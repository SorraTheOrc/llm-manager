"""Last-resort local admission when the GPU is idle and remotes are exhausted.

LP-0MVWAMY8M001KE4W. The idle-admit rung (LP-0MVGD9Z8Z0055Y5J) overrides the
economic bypass on an idle GPU, but it is still priced by the flat
``local_cold_prefill_budget_tokens`` cap (25,000). Measured over the 24 h window
2026-10-29 23:43:30 → 2026-10-30 23:43:30:

- the idle-admit ladder fired 1,337 times and refused 1,291 (96.6 %) on
  ``routing_idle_admit_refused reason=prefill_budget_exceeded`` (``new_tokens``
  p50 79,136 — all inside the 120k warm cap, ``idle_seconds`` p50 197 s);
- 236 of the window's 686 ``All providers exhausted`` warnings landed inside
  idle gaps that carried 98 inbound requests and **0 streams**;
- the remote chain was empty for those gaps (``usage_limit_reset_pending``,
  ``quota_exhausted``, ``outside available_times``).

The rung under test is the LAST rung of the local admission ladder: when every
remote provider is measured unavailable **now** and the GPU has been idle for
``local_last_resort_idle_seconds`` (default 300 s), the prompt is served
locally with its own size-appropriate deadline instead of answering a
retryable 503.

Coverage map: AC1 (admit), AC2 (unreachable while a remote is available), AC3
(size-scaled deadline replaces the normal one), AC4 (concurrency bounded at
1), AC5 (never above the warm cap), AC6 (fail-safe), AC7 (kill switch at 0
restores today's decision matrix), AC9 (structured ``local_last_resort_admit``
log line with the fields the ≥ 24 h measurement needs).
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace

import proxy.provider as provider
import pytest
from proxy.local_activity import record_local_activity, reset_local_activity
from proxy.provider import (
    DEFAULT_LAST_RESORT_DEADLINE_SECONDS,
    DEFAULT_LAST_RESORT_IDLE_SECONDS,
    LAST_RESORT_REASON,
    LOCAL_LAST_RESORT_ATTR,
    MAX_LAST_RESORT_DEADLINE_SECONDS,
    _evaluate_last_resort_admission,
    _last_resort_config,
    _log_last_resort_admit,
    _prefill_admission_guard,
    _remote_chain_has_capacity,
    _should_skip_local,
)

from proxy import contention_queue, router_helpers

MODEL = "Qwen3"
SESSION = "sess-last-resort-1"
COLD = 38_000
WARM = 120_000


@pytest.fixture(autouse=True)
def _clean_local_state(monkeypatch):
    """Reset idle/queue/cooldown state and pin the local server globals."""
    import proxy.server as server

    reset_local_activity()
    contention_queue.reset()
    provider._provider_unavailable_until.clear()
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
    yield
    reset_local_activity()
    contention_queue.reset()
    provider._provider_unavailable_until.clear()


def _idle_for(seconds: float) -> None:
    record_local_activity("dispatch", now=time.monotonic() - seconds)


def _config(**overrides) -> dict:
    server = {
        "local_large_context_cold_cache_threshold": COLD,
        "local_large_context_warm_cache_threshold": WARM,
    }
    server.update(overrides)
    return {"server": server}


def _provider_cfg(name: str = "local-qwen3") -> dict:
    return {"name": name, "type": "local", "llama_model": MODEL}


def _remote(name: str, **overrides) -> dict:
    entry = {"name": name, "type": "remote", "provider": name, "model": MODEL}
    entry.update(overrides)
    return entry


def _chain(*, remotes_available: bool, config: dict | None = None) -> dict:
    """A local-first chain whose remote entries are available or exhausted.

    "Exhausted" is the same state the resolver reads: every remote entry in a
    real cooldown, which is what ``usage_limit_reset_pending`` /
    ``quota_exhausted`` / ``outside available_times`` look like to
    :func:`proxy.provider._remote_chain_has_capacity`.
    """
    remotes = [_remote("remote-a"), _remote("remote-b")]
    if not remotes_available:
        for entry in remotes:
            provider.mark_provider_unavailable(entry["name"], 600)
    return {"name": MODEL, "providers": [_provider_cfg(), *remotes]}


def _request() -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace())


def _decide(
    *,
    config: dict | None = None,
    chain: dict | None = None,
    request=None,
    tokens: int = 80_000,
    ratio: float = 0.0,
):
    return _evaluate_last_resort_admission(
        config if config is not None else _config(),
        _provider_cfg(),
        chain if chain is not None else _chain(remotes_available=False),
        request if request is not None else _request(),
        tokens,
        ratio,
    )


class TestLastResortConfig:
    """AC7 — the knobs and their defaults."""

    def test_defaults_are_300s_idle_and_600s_deadline(self):
        assert _last_resort_config({}) == (
            DEFAULT_LAST_RESORT_IDLE_SECONDS,
            DEFAULT_LAST_RESORT_DEADLINE_SECONDS,
        )
        assert DEFAULT_LAST_RESORT_IDLE_SECONDS == 300.0
        assert DEFAULT_LAST_RESORT_DEADLINE_SECONDS == 600.0

    def test_configured_values_are_read(self):
        assert _last_resort_config(
            {
                "local_last_resort_idle_seconds": 600,
                "local_last_resort_deadline_seconds": 450,
            }
        ) == (600.0, 450.0)

    @pytest.mark.parametrize("bad", [0, -1, "not-a-number", None, True])
    def test_non_positive_or_malformed_idle_disables_the_rung(self, bad):
        """A kill switch and a fail-safe: no idle threshold ⇒ today's path."""
        idle, _deadline = _last_resort_config(
            {"local_last_resort_idle_seconds": bad}
        )
        assert idle == 0.0

    @pytest.mark.parametrize("bad", [0, -5, "not-a-number", None, False])
    def test_malformed_deadline_falls_back_to_the_default(self, bad):
        """The rung must never admit a dispatch that has no deadline."""
        _idle, deadline = _last_resort_config(
            {"local_last_resort_deadline_seconds": bad}
        )
        assert deadline == DEFAULT_LAST_RESORT_DEADLINE_SECONDS

    def test_deadline_is_hard_capped(self):
        _idle, deadline = _last_resort_config(
            {"local_last_resort_deadline_seconds": 100_000}
        )
        assert deadline == MAX_LAST_RESORT_DEADLINE_SECONDS


class TestLastResortDecision:
    """AC1/AC2/AC5 — the admission decision matrix."""

    def test_idle_gpu_with_exhausted_chain_admits(self):
        """AC1: the request that used to be a 503 is dispatched locally."""
        _idle_for(600)

        decision = _decide()

        assert decision is not None
        assert decision["admitted"] is True
        assert decision["reason"] == LAST_RESORT_REASON
        assert decision["remote_exhausted"] is True
        assert decision["estimated_tokens"] == 80_000
        assert decision["new_tokens"] == 80_000
        assert decision["deadline_s"] == DEFAULT_LAST_RESORT_DEADLINE_SECONDS
        assert decision["idle_seconds"] >= DEFAULT_LAST_RESORT_IDLE_SECONDS
        assert decision["threshold_seconds"] == DEFAULT_LAST_RESORT_IDLE_SECONDS
        assert decision["warm_cap"] == WARM

    def test_new_tokens_are_priced_off_the_cached_ratio(self):
        _idle_for(600)

        decision = _decide(tokens=60_000, ratio=0.5)

        assert decision is not None
        assert decision["estimated_tokens"] == 60_000
        assert decision["new_tokens"] == 30_000

    def test_any_available_remote_keeps_todays_path(self):
        """AC2: the rung is unreachable in the normal path.

        The decision *input* is asserted, not only the outcome: the same
        availability predicate the resolver uses reports capacity, and the rung
        then declines.
        """
        _idle_for(600)
        chain = _chain(remotes_available=True)
        request = _request()

        assert _remote_chain_has_capacity(request, chain, _config()) is True
        assert _decide(chain=chain, request=request) is None

    def test_idle_below_the_threshold_keeps_todays_path(self):
        _idle_for(30)

        assert _decide() is None

    def test_prompt_above_the_warm_cap_is_never_admitted(self):
        """AC5: a prompt that cannot fit a slot is never a last resort."""
        _idle_for(600)

        assert _decide(tokens=WARM + 1) is None

    def test_prompt_at_the_warm_cap_is_admitted(self):
        _idle_for(600)

        assert _decide(tokens=WARM) is not None

    def test_disabled_warm_cap_declines(self):
        """AC5 fail-safe: with no warm cap the fit cannot be verified."""
        _idle_for(600)
        config = _config(local_large_context_warm_cache_threshold=0)

        assert _decide(config=config) is None

    def test_per_entry_context_ceiling_tightens_the_warm_cap(self):
        _idle_for(600)
        provider_cfg = _provider_cfg()
        provider_cfg["max_context_tokens"] = 50_000

        decision = _evaluate_last_resort_admission(
            _config(),
            provider_cfg,
            _chain(remotes_available=False),
            _request(),
            60_000,
            0.0,
        )

        assert decision is None
        assert (
            _evaluate_last_resort_admission(
                _config(),
                provider_cfg,
                _chain(remotes_available=False),
                _request(),
                40_000,
                0.0,
            )
            is not None
        )


class TestLastResortConcurrencyBound:
    """AC4 — concurrency is bounded at 1: only a provably idle GPU admits."""

    @pytest.mark.parametrize(
        "attr, value, blocker",
        [
            ("local_active_queries", 1, "local_active"),
            ("local_generating_queries", 1, "generating"),
            ("local_prefill_in_flight", {"other-session": 1.0}, "prefill_in_flight"),
            ("model_switch_refcount", 1, "model_switch"),
        ],
    )
    def test_in_flight_local_work_keeps_todays_path(
        self, monkeypatch, attr, value, blocker
    ):
        import proxy.server as server

        monkeypatch.setattr(server, attr, value, raising=False)
        _idle_for(600)

        assert _decide() is None

    def test_queued_waiter_keeps_todays_path(self):
        class _Waiter:
            pass

        contention_queue._waiters.append(_Waiter())
        _idle_for(600)

        assert _decide() is None


class TestLastResortFailSafe:
    """AC6 — an error denies, never widening admission."""

    def test_remote_availability_error_denies(self, monkeypatch):
        _idle_for(600)

        def _boom(*_args, **_kwargs):
            raise RuntimeError("availability unavailable")

        monkeypatch.setattr(provider, "_remote_chain_has_capacity", _boom)

        assert _decide() is None

    def test_missing_idle_clock_denies(self, monkeypatch):
        import proxy.local_activity as local_activity

        def _boom(*_args, **_kwargs):
            raise RuntimeError("no clock")

        monkeypatch.setattr(local_activity, "local_idle_seconds", _boom)

        assert _decide() is None

    def test_unmeasured_prompt_size_denies(self):
        _idle_for(600)

        assert _decide(tokens=0) is None


class TestKillSwitch:
    """AC7 — ``local_last_resort_idle_seconds: 0`` restores today's matrix."""

    def test_zero_disables_every_admission(self):
        _idle_for(10_000)
        config = _config(local_last_resort_idle_seconds=0)

        for tokens in (1_000, 40_000, 80_000, WARM):
            assert _decide(config=config, tokens=tokens) is None

    def test_zero_disables_even_with_an_exhausted_chain(self):
        _idle_for(10_000)
        config = _config(local_last_resort_idle_seconds=0)

        assert (
            _decide(
                config=config,
                chain=_chain(remotes_available=False),
                request=_request(),
            )
            is None
        )

    def test_enabled_rung_is_the_only_difference(self):
        _idle_for(10_000)
        disabled = _config(local_last_resort_idle_seconds=0)
        enabled = _config(local_last_resort_idle_seconds=300)

        assert _decide(config=disabled) is None
        assert _decide(config=enabled) is not None


class TestLastResortAtTheGate:
    """AC1/AC3 at the ladder level: the rung turns a skip into a dispatch.

    The decision-level tests above exercise ``_evaluate_last_resort_admission``
    directly; this class drives the two neighbouring gates that decide whether
    the admitted dispatch actually reaches the GPU — the routing skip that
    precedes the rung and the prefill-ETA admission guard that follows it.
    """

    def test_the_rung_converts_a_cold_cache_bypass_into_an_admission(self):
        body = {"messages": [{"role": "user", "content": "x" * (80_000 * 4)}]}
        assert _should_skip_local(
            MODEL, SESSION, body, COLD, estimated_tokens=80_000
        ) == (True, "large_context_bypass")

        _idle_for(600)

        decision = _decide(tokens=80_000)
        assert decision is not None and decision["admitted"] is True

    def _guard_config(self) -> dict:
        return _config(
            local_cold_prefill_budget_tokens=25_000,
            local_admission_guard_policy="reject",
            local_prefill_deadline_seconds=120,
        )

    def test_admission_guard_is_inert_for_a_last_resort_dispatch(self):
        """The flat cap / normal deadline are the bounds the rung overrules."""
        refused = asyncio.run(
            _prefill_admission_guard(
                self._guard_config(),
                _provider_cfg(),
                80_000,
                SESSION,
                priority="medium",
                max_local=2,
                reason_label="eta_exceeds_budget_direct",
                free_slot=True,
            )
        )
        assert refused is not None and refused.admit is False
        assert refused.reason == "prefill_budget_exceeded"

        inert = asyncio.run(
            _prefill_admission_guard(
                self._guard_config(),
                _provider_cfg(),
                80_000,
                SESSION,
                priority="medium",
                max_local=2,
                reason_label="eta_exceeds_budget_direct",
                free_slot=True,
                last_resort_admitted=True,
            )
        )
        assert inert is None


class TestLastResortDeadline:
    """AC3 — the dispatch uses its own deadline, not the normal one."""

    def _srv(self, **overrides) -> SimpleNamespace:
        server = {
            "local_prefill_deadline_seconds": 120,
            "local_dispatch_first_byte_timeout_seconds": 840,
            "local_dispatch_max_prefill_seconds": 900,
        }
        server.update(overrides)
        return SimpleNamespace(config={"server": server}, logger=logging.getLogger("x"))

    def _stamped_request(self, config: dict, tokens: int = 80_000):
        _idle_for(600)
        decision = _decide(config=config, tokens=tokens)
        assert decision is not None
        request = _request()
        setattr(request, LOCAL_LAST_RESORT_ATTR, dict(decision))
        return request

    def test_deadline_helpers_use_the_last_resort_deadline(self):
        config = _config(local_last_resort_deadline_seconds=600)
        srv = self._srv()
        request = self._stamped_request(config)

        assert router_helpers._get_prefill_deadline_seconds(srv, request) == 600.0
        assert router_helpers._get_first_byte_timeout_seconds(srv, request) == 600.0

    def test_unstamped_requests_keep_the_normal_deadline(self):
        srv = self._srv()

        assert router_helpers._get_prefill_deadline_seconds(srv, _request()) == 120.0
        assert router_helpers._get_first_byte_timeout_seconds(srv, _request()) == 120.0

    def test_disabled_normal_deadline_still_bounds_the_dispatch(self):
        """``local_prefill_deadline_seconds: 0`` no longer means unbounded."""
        config = _config(local_last_resort_deadline_seconds=600)
        srv = self._srv(local_prefill_deadline_seconds=0)
        request = self._stamped_request(config)

        assert router_helpers._get_prefill_deadline_seconds(srv, request) == 600.0

    def test_dispatch_survives_past_the_normal_deadline(self):
        """The prefill outlives ``local_prefill_deadline_seconds`` and completes."""
        config = _config(
            local_last_resort_deadline_seconds=0.6,
        )
        srv = self._srv(local_prefill_deadline_seconds=0.2)
        request = self._stamped_request(config)

        ceiling = router_helpers._get_first_byte_timeout_seconds(srv, request)
        assert ceiling == 0.6

        async def _first_byte_after(delay: float):
            await asyncio.sleep(delay)
            return ("stream", "ok")

        result = asyncio.run(
            router_helpers._await_first_byte_with_prefill_monitor(
                srv,
                _first_byte_after(0.4),
                timeout=ceiling,
                poll_seconds=0,
                warn_seconds=10,
            )
        )

        assert result[1] == "ok"

    def test_normal_deadline_would_have_cancelled_the_same_prefill(self):
        srv = self._srv(local_prefill_deadline_seconds=0.2)
        plain = _request()

        ceiling = router_helpers._get_first_byte_timeout_seconds(srv, plain)
        assert ceiling == 0.2

        async def _first_byte_after(delay: float):
            await asyncio.sleep(delay)
            return ("stream", "ok")

        with pytest.raises(TimeoutError):
            asyncio.run(
                router_helpers._await_first_byte_with_prefill_monitor(
                    srv,
                    _first_byte_after(0.4),
                    timeout=ceiling,
                    poll_seconds=0,
                    warn_seconds=10,
                )
            )


class TestLastResortObservability:
    """AC9 — the structured line the measurement window depends on."""

    def test_log_line_carries_every_required_field(self, caplog):
        _idle_for(600)
        decision = _decide(tokens=79_136, ratio=0.0)
        assert decision is not None

        with caplog.at_level(logging.INFO, logger="llama-proxy"):
            _log_last_resort_admit(decision, "local-qwen3", MODEL, SESSION)

        lines = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("local_last_resort_admit")
        ]
        assert len(lines) == 1
        line = lines[0]
        for field in (
            "estimated_tokens=79136",
            "new_tokens=79136",
            "idle_seconds=",
            "deadline_s=600.0",
            "remote_exhausted=true",
            f"session={SESSION}",
        ):
            assert field in line
        # ``turn=`` is omitted when no turn is bound (byte-compatible logs);
        # when it is bound it must appear.
        from proxy.turn_id import get_turn_id, reset_turn_id

        assert get_turn_id() is None or "turn=" in line
        reset_turn_id()

    def test_attribute_constant_is_shared_with_the_deadline_reader(self):
        """A drifting attribute name would silently disable the deadline."""
        assert LOCAL_LAST_RESORT_ATTR == router_helpers.LOCAL_LAST_RESORT_ATTR


class TestShippedConfig:
    """AC7 (default ON) — the shipped config declares the rung."""

    def test_shipped_config_enables_the_rung_by_default(self):
        from pathlib import Path

        import yaml

        repo_root = Path(__file__).resolve().parents[2]
        server = (
            yaml.safe_load((repo_root / "proxy" / "config.yaml").read_text())
        )["server"]

        # Declared (not merely defaulted) so an operator can find the knobs and
        # AC7's default-ON state is visible in the file, then read back through
        # the same reader the proxy uses.
        assert "local_last_resort_idle_seconds" in server
        assert "local_last_resort_deadline_seconds" in server
        idle, deadline = _last_resort_config(server)
        assert idle == 300.0
        assert deadline == 600.0
