"""Idle-aware local admission (LP-0MVGD9Z8Z0055Y5J).

Routing used to be purely size/economics-driven, so a large or cache-cold
prompt was bypassed to a paid remote provider even while both GPU slots sat
unused (187 ``local_dispatch_denied`` in an hour where local busy was 2.5 %).

These tests cover the override that spends obvious idle capacity:

- AC1: ``local_idle_admit_enabled`` (default **false**) / ``local_idle_admit_seconds``
  (default 60), validated, and disabled when the local model is not healthy;
- AC2: idle > threshold + empty queue ⇒ admit; idle < threshold ⇒ today's
  bypass; queue non-empty ⇒ today's bypass; model loading/switch ⇒ today's
  bypass;
- AC3: cannot fire while a lease/prefill is in flight, and the admission is
  visible as a greppable ``routing_idle_admit`` line (parsed by
  ``proxy-usage-analysis`` as the ``local_idle_admit`` counter).
"""

from __future__ import annotations

import asyncio
import time

import pytest
from proxy.local_activity import (
    activity_snapshot,
    last_local_activity_ts,
    local_idle_seconds,
    record_local_activity,
    reset_local_activity,
)
from proxy.provider import (
    DEFAULT_IDLE_ADMIT_SECONDS,
    IDLE_ADMIT_REASON,
    IDLE_OVERRIDABLE_SKIP_REASONS,
    _evaluate_idle_admission_override,
    _idle_admit_config,
    _local_idle_blocker,
    _should_skip_local,
)

from proxy import contention_queue, prefill_eta

MODEL = "Qwen3"
SESSION = "sess-idle-1"
COLD = 38_000


@pytest.fixture(autouse=True)
def _clean_idle_state(monkeypatch):
    """Reset the idle clock, the calibration and the local server globals."""
    import proxy.server as server

    reset_local_activity()
    prefill_eta.reset()
    monkeypatch.setattr(server, "backend_ready", True, raising=False)
    monkeypatch.setattr(server, "llama_process", object(), raising=False)
    monkeypatch.setattr(server, "local_active_queries", 0, raising=False)
    monkeypatch.setattr(server, "local_generating_queries", 0, raising=False)
    monkeypatch.setattr(server, "local_prefill_in_flight", {}, raising=False)
    monkeypatch.setattr(server, "model_switch_refcount", 0, raising=False)
    monkeypatch.setattr(server, "background_loads", {}, raising=False)
    monkeypatch.setattr(
        server, "backend_recovery_state", {"in_progress": False}, raising=False
    )
    contention_queue.reset()
    yield
    reset_local_activity()
    prefill_eta.reset()
    contention_queue.reset()


def _idle_for(seconds: float) -> None:
    """Pretend the local backend has been idle for *seconds*."""
    record_local_activity("dispatch", now=time.monotonic() - seconds)


def _config(**overrides):
    server = {
        "local_large_context_cold_cache_threshold": COLD,
        "local_large_context_warm_cache_threshold": 0,
    }
    server.update(overrides)
    return {"server": server}


def _enabled_config(**overrides):
    base = {
        "local_idle_admit_enabled": True,
        "local_idle_admit_seconds": 60,
    }
    base.update(overrides)
    return _config(**base)


def _provider_cfg(endpoint: str | None = None) -> dict:
    cfg = {"name": "local-qwen3", "type": "local", "llama_model": MODEL}
    if endpoint:
        cfg["endpoint"] = endpoint
    return cfg


GUARD_BUDGET = 25_000
GUARD_DEADLINE = 120.0
GUARD_SAFETY = 0.95


def _guard_config(**overrides) -> dict:
    """The shipped post-change admission configuration, idle override enabled."""
    server = {
        "local_idle_admit_enabled": True,
        "local_idle_admit_seconds": 60,
        "contention_queue_max_wait_seconds": 60,
        "local_prefill_deadline_seconds": int(GUARD_DEADLINE),
        "local_cold_prefill_budget_tokens": GUARD_BUDGET,
        "local_admission_guard_budget_safety_fraction": GUARD_SAFETY,
        "local_admission_guard_policy": "reject",
        "local_admission_guard_min_samples": 5,
        "local_admission_guard_eta_percentile_min_tokens": 20_000,
        "local_admission_guard_eta_percentile": 75,
    }
    server.update(overrides)
    return _config(**server)


def _calibrate(tokens: int, service_ms: float, count: int = 6) -> None:
    """Seed an identical-sample bucket so the ``(model, bucket)`` is calibrated."""
    for _ in range(count):
        prefill_eta.record_dispatch_sample(MODEL, tokens, 0.0, service_ms)


def _body(tokens: int = 120_000) -> dict:
    # ~4 chars/token is close enough for the shared estimator.
    return {"messages": [{"role": "user", "content": "x" * (tokens * 4)}]}


class TestLocalActivityClock:
    def test_reset_starts_the_clock(self):
        reset_local_activity(now=1_000.0)
        assert last_local_activity_ts() == pytest.approx(1_000.0)
        assert local_idle_seconds(now=1_030.0) == pytest.approx(30.0)

    def test_record_resets_idle_age(self):
        _idle_for(500)
        assert local_idle_seconds() >= 499

        record_local_activity("chunk")

        assert local_idle_seconds() < 1

    def test_idle_is_never_negative(self):
        reset_local_activity(now=2_000.0)

        assert local_idle_seconds(now=1_500.0) == 0.0

    def test_snapshot_exposes_source_and_timestamps(self):
        record_local_activity("chunk", now=5_000.0, wall=1_700_000_000.0)

        snapshot = activity_snapshot(now=5_120.0)

        assert snapshot["idle_seconds"] == pytest.approx(120.0)
        assert snapshot["last_activity_source"] == "chunk"
        assert snapshot["last_activity_at"].startswith("2023-11-14")


class TestIdleAdmitConfig:
    """AC1 — knobs are default-off and validated."""

    def test_disabled_by_default(self):
        assert _idle_admit_config({}) == (False, DEFAULT_IDLE_ADMIT_SECONDS)

    def test_enabled_with_configured_threshold(self):
        assert _idle_admit_config(
            {"local_idle_admit_enabled": True, "local_idle_admit_seconds": 120}
        ) == (True, 120.0)

    @pytest.mark.parametrize("bad", [0, -5, "not-a-number", None])
    def test_non_positive_or_malformed_threshold_disables(self, bad):
        enabled, seconds = _idle_admit_config(
            {"local_idle_admit_enabled": True, "local_idle_admit_seconds": bad}
        )
        assert enabled is False

    def test_missing_threshold_defaults_to_60_when_enabled(self):
        assert _idle_admit_config({"local_idle_admit_enabled": True}) == (True, 60.0)


class TestIdleBlockers:
    """AC2/AC3 — every non-idle signal keeps today's behaviour."""

    def test_healthy_idle_backend_has_no_blocker(self):
        assert _local_idle_blocker("http://localhost:8080") is None

    def test_self_healing_blocks(self, monkeypatch):
        import proxy.server as server

        monkeypatch.setattr(
            server, "backend_recovery_state", {"in_progress": True}, raising=False
        )
        assert _local_idle_blocker("http://localhost:8080") == "self_healing"

    def test_model_switch_refcount_blocks(self, monkeypatch):
        import proxy.server as server

        monkeypatch.setattr(server, "model_switch_refcount", 1, raising=False)
        assert _local_idle_blocker("http://localhost:8080") == "model_switch"

    def test_background_load_blocks(self, monkeypatch):
        import proxy.server as server

        monkeypatch.setattr(
            server, "background_loads", {"qwen": True}, raising=False
        )
        assert _local_idle_blocker("http://localhost:8080") == "model_loading"

    def test_backend_not_ready_blocks_default_endpoint(self, monkeypatch):
        import proxy.server as server

        monkeypatch.setattr(server, "backend_ready", False, raising=False)
        assert _local_idle_blocker(None) == "backend_not_ready"

    def test_missing_llama_process_blocks_default_endpoint(self, monkeypatch):
        import proxy.server as server

        monkeypatch.setattr(server, "llama_process", None, raising=False)
        assert _local_idle_blocker(None) == "backend_not_running"

    def test_explicit_endpoint_skips_process_gate(self, monkeypatch):
        """An explicit provider endpoint is an externally managed server, so
        the single-process ready gate does not apply (LP-0MRPILSMW004T4H8)."""
        import proxy.server as server

        monkeypatch.setattr(server, "backend_ready", False, raising=False)
        monkeypatch.setattr(server, "llama_process", None, raising=False)
        assert _local_idle_blocker("http://192.168.0.199:8080") is None

    def test_active_lease_blocks(self, monkeypatch):
        import proxy.server as server

        monkeypatch.setattr(server, "local_active_queries", 1, raising=False)
        assert _local_idle_blocker("http://localhost:8080") == "local_active"

    def test_generating_blocks(self, monkeypatch):
        import proxy.server as server

        monkeypatch.setattr(server, "local_generating_queries", 1, raising=False)
        assert _local_idle_blocker("http://localhost:8080") == "generating"

    def test_prefill_in_flight_blocks(self, monkeypatch):
        import proxy.server as server

        monkeypatch.setattr(
            server, "local_prefill_in_flight", {"sess": 1}, raising=False
        )
        assert _local_idle_blocker("http://localhost:8080") == "prefill_in_flight"

    def test_non_empty_queue_blocks(self):
        class _Waiter:
            pass

        contention_queue._waiters.append(_Waiter())
        try:
            assert _local_idle_blocker("http://localhost:8080") == "queue_not_empty"
        finally:
            contention_queue.reset()


class TestIdleAdmissionOverride:
    """AC2 — the decision matrix of the override."""

    def test_disabled_returns_none_even_when_idle(self):
        _idle_for(600)

        assert (
            _evaluate_idle_admission_override(_config(), _provider_cfg(), 120_000)
            is None
        )

    def test_below_threshold_keeps_the_bypass(self):
        _idle_for(30)

        assert (
            _evaluate_idle_admission_override(
                _enabled_config(), _provider_cfg(), 120_000
            )
            is None
        )

    def test_idle_and_healthy_admits(self):
        _idle_for(90)

        decision = _evaluate_idle_admission_override(
            _enabled_config(), _provider_cfg(), 120_000
        )

        assert decision is not None
        assert decision["reason"] == IDLE_ADMIT_REASON
        assert decision["idle_seconds"] >= 60
        assert decision["threshold_seconds"] == 60.0

    def test_queue_not_empty_keeps_the_bypass(self):
        _idle_for(600)

        class _Waiter:
            pass

        contention_queue._waiters.append(_Waiter())
        try:
            assert (
                _evaluate_idle_admission_override(
                    _enabled_config(), _provider_cfg(), 120_000
                )
                is None
            )
        finally:
            contention_queue.reset()

    def test_model_loading_keeps_the_bypass(self, monkeypatch):
        import proxy.server as server

        monkeypatch.setattr(
            server, "background_loads", {"qwen": True}, raising=False
        )
        _idle_for(600)

        assert (
            _evaluate_idle_admission_override(
                _enabled_config(), _provider_cfg(), 120_000
            )
            is None
        )

    def test_active_request_keeps_the_bypass(self, monkeypatch):
        import proxy.server as server

        monkeypatch.setattr(server, "local_active_queries", 2, raising=False)
        _idle_for(600)

        assert (
            _evaluate_idle_admission_override(
                _enabled_config(), _provider_cfg(), 120_000
            )
            is None
        )

    def test_override_turns_the_economic_bypass_into_a_local_admission(self):
        """AC1/AC2 end-to-end at the gate: the same request the economic
        cold-cache check bypasses is admitted once the GPU is idle."""
        body = _body()
        skip, reason = _should_skip_local(
            MODEL, SESSION, body, COLD, estimated_tokens=120_000
        )
        assert (skip, reason) == (True, "large_context_bypass")

        _idle_for(180)
        decision = _evaluate_idle_admission_override(
            _enabled_config(), _provider_cfg(), 120_000
        )

        assert decision is not None
        # The physical cap is NOT overridable: context_too_large would have
        # been reported by _should_skip_local instead of large_context_bypass.


class TestIdleAdmitPrefillFeasibility:
    """AC1/AC2/AC3/AC4 — the idle admit is priced against prefill feasibility.

    LP-0MVRCLPVI0057116: an idle GPU is not permission to start a prefill it
    cannot finish before ``local_prefill_deadline_seconds``. The eligibility
    verdict must agree with the shared admission guard (fail-safe: it may only
    ever *refuse* admissions the override would otherwise have made), and the
    guard knobs must be read per dispatch so ``POST /admin/reload-config``
    retunes it without a restart.
    """

    def test_projected_prefill_beyond_the_safety_budget_is_refused(self):
        """Calibrated pricing over ``0.95 × 120 s`` refuses before dispatch."""
        _calibrate(20_000, 117_000.0)  # 117 s of prefill for a 20k prompt
        _idle_for(600)

        verdict = _evaluate_idle_admission_override(
            _guard_config(), _provider_cfg(), 20_000, 0.0
        )

        assert verdict is not None, "idle capacity applies, so a verdict is returned"
        assert verdict["admitted"] is False
        assert verdict["guard_reason"] in (
            "prefill_eta_exceeds_safety_budget",
            "prefill_eta_exceeds_deadline",
            "eta_exceeds_budget",
        )
        decision = verdict["guard_decision"]
        assert decision is not None and decision.admit is False
        assert decision.eta_prefill_seconds is not None
        assert decision.eta_prefill_seconds > GUARD_SAFETY * GUARD_DEADLINE

    def test_calibrated_prefill_inside_the_deadline_is_admitted(self):
        """A prompt that fits the budget and deadline is still admitted."""
        _calibrate(20_000, 40_000.0)  # 40 s: inside the 60 s budget and 114 s safety
        _idle_for(600)

        verdict = _evaluate_idle_admission_override(
            _guard_config(), _provider_cfg(), 20_000, 0.0
        )

        assert verdict is not None and verdict["admitted"] is True
        assert verdict["guard_decision"] is not None
        assert verdict["guard_decision"].admit is True

    def test_uncalibrated_bucket_keeps_the_idle_admit_fail_open(self):
        """No samples for the bucket: the guard is inert, the admit stands."""
        _idle_for(600)

        verdict = _evaluate_idle_admission_override(
            _guard_config(), _provider_cfg(), 20_000, 0.0
        )

        assert verdict is not None and verdict["admitted"] is True
        assert verdict["guard_decision"].reason == "insufficient_samples"

    def test_flat_budget_refuses_even_from_an_idle_gpu(self):
        """The flat cap is calibration-independent: 25,001 new tokens refuse."""
        _calibrate(20_000, 40_000.0)  # a *fast* calibrated bucket must not admit it
        _idle_for(600)

        verdict = _evaluate_idle_admission_override(
            _guard_config(), _provider_cfg(), 25_001, 0.0
        )

        assert verdict is not None and verdict["admitted"] is False
        assert verdict["guard_reason"] == "prefill_budget_exceeded"

    def test_eligibility_never_admits_more_than_the_guard(self):
        """AC2 fail-safe: the idle verdict is the guard's, never a wider one."""
        cases = [
            (40_000, 0.0, None, False),        # over the flat cap -> refuse
            (20_000, 0.0, 117_000.0, False),   # projected over the deadline -> refuse
            (20_000, 0.0, 40_000.0, True),     # within every bound -> admit
        ]
        for tokens, ratio, service_ms, expected in cases:
            prefill_eta.reset()
            if service_ms:
                _calibrate(tokens, service_ms)
            _idle_for(600)

            verdict = _evaluate_idle_admission_override(
                _guard_config(), _provider_cfg(), tokens, ratio
            )

            assert verdict is not None
            assert verdict["admitted"] is expected, (tokens, service_ms)
            guard_decision = verdict["guard_decision"]
            if guard_decision is not None:
                assert verdict["admitted"] is bool(guard_decision.admit)

    def test_guard_knobs_are_read_per_dispatch_hot_reloadable(self):
        """AC3: the eligibility bound is resolved per call (no restart)."""
        _calibrate(20_000, 117_000.0)
        _idle_for(600)

        refused = _evaluate_idle_admission_override(
            _guard_config(), _provider_cfg(), 20_000, 0.0
        )
        assert refused is not None and refused["admitted"] is False

        # Hot-reload: disable the cold budget AND the deadline-relative term.
        reloaded = _evaluate_idle_admission_override(
            _guard_config(
                local_cold_prefill_budget_tokens=0,
                local_admission_guard_budget_safety_fraction=0.0,
            ),
            _provider_cfg(),
            20_000,
            0.0,
        )
        assert reloaded is not None and reloaded["admitted"] is True
        assert reloaded["guard_reason"] is None

    def test_physical_cap_and_lease_gate_are_not_overridable(self):
        """AC4: only the economic/residency misses are idle-overridable.

        ``context_too_large`` is the physical KV-cap refusal and
        ``local_lease_active`` is the dispatch lease gate; neither is in the
        override's eligibility set, so the tuning can never turn either into an
        admission. The idle gate is additionally blocked while a request/lease
        is active (``test_active_request_keeps_the_bypass``).
        """
        assert "large_context_bypass" in IDLE_OVERRIDABLE_SKIP_REASONS
        assert "context_too_large" not in IDLE_OVERRIDABLE_SKIP_REASONS
        assert "local_lease_active" not in IDLE_OVERRIDABLE_SKIP_REASONS


class TestStatusPayload:
    """AC1 — idle age is published on /llama/local/status."""

    def test_status_includes_local_idle_seconds(self, monkeypatch):
        import proxy.server as server

        async def _fake_query():
            return {"llama_server_running": False, "n_ctx": None}

        monkeypatch.setattr(server, "query_llama_status", _fake_query, raising=False)
        from proxy.handlers import get_llama_local_status
        from starlette.requests import Request

        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/llama/local/status",
                "headers": [],
                "query_string": b"",
                "client": ("127.0.0.1", 4321),
                "server": ("testserver", 80),
                "scheme": "http",
                "root_path": "",
            }
        )
        _idle_for(120)

        payload = asyncio.run(get_llama_local_status(request))

        assert payload["local_idle_seconds"] >= 100
        assert payload["local_last_activity_source"] == "dispatch"
        assert payload["local_last_activity_at"]
