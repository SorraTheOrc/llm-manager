"""Deadline-aware local admission bound (LP-0MVKUKEU80022C3U).

Verifies that the prefill-ETA guard refuses local dispatch when the projected
first-byte ETA exceeds ``local_prefill_deadline_seconds`` (default 90 s).

Acceptance criteria tested:

- AC1: Projected ETA >= deadline is refused on both the queue and direct
  paths with ``reason=prefill_eta_exceeds_deadline``.
- AC2: Projected ETA < deadline keeps today's admission behaviour (no new
  refusals; existing guard tests unchanged).
- AC3: Boundary (ETA == deadline) refuses; ``local_prefill_deadline_seconds: 0``
  disables the bound and is covered by a test.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import pytest

from proxy import prefill_eta


@pytest.fixture(autouse=True)
def _clean_calibration():
    prefill_eta.reset()
    yield
    prefill_eta.reset()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**server_overrides) -> dict:
    server = {
        "session_slot_pool_size": 1,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": 60,
        "contention_queue_max_depth": 8,
    }
    server.update(server_overrides)
    return {"server": server}


def _seed_samples(
    model: str = "Qwen3",
    tokens: int = 100_000,
    ratio: float = 0.0,
    ms_per_token: float = 3.9,
    count: int = 6,
) -> None:
    """Record *count* synthetic dispatches at *ms_per_token*."""
    for _ in range(count):
        prefill_eta.record_dispatch_sample(
            model, tokens, ratio, tokens * ms_per_token
        )


# ---------------------------------------------------------------------------
# AC3: configuration resolution
# ---------------------------------------------------------------------------

def test_default_deadline_is_90_seconds():
    """AC3: default prefill deadline is 90 s."""
    cfg = prefill_eta.resolve_config(_cfg())
    assert cfg.prefill_deadline_seconds == 90.0


def test_deadline_overridden_by_config():
    """AC3: a config value overrides the default."""
    cfg = prefill_eta.resolve_config(_cfg(local_prefill_deadline_seconds=45))
    assert cfg.prefill_deadline_seconds == 45.0


def test_disabled_deadline_is_zero():
    """AC3: 0 disables the bound."""
    cfg = prefill_eta.resolve_config(_cfg(local_prefill_deadline_seconds=0))
    assert cfg.prefill_deadline_seconds == 0.0


def test_malformed_deadline_falls_back_to_default():
    """A non-numeric value falls back to the default, never raises."""
    cfg = prefill_eta.resolve_config(
        _cfg(local_prefill_deadline_seconds="not-a-number")
    )
    assert cfg.prefill_deadline_seconds == 90.0


def test_none_deadline_uses_default():
    """None is treated as missing — default 90 s."""
    cfg = prefill_eta.resolve_config(
        _cfg(local_prefill_deadline_seconds=None)
    )
    assert cfg.prefill_deadline_seconds == 90.0


# ---------------------------------------------------------------------------
# AC1: deadline refusal on the queue path
# ---------------------------------------------------------------------------

def test_deadline_refuses_when_eta_exceeds_deadline():
    """AC1: projected ETA >= deadline ⇒ refused with reason=prefill_eta_exceeds_deadline."""
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    decision = prefill_eta.evaluate_admission(_cfg(), "Qwen3", 100_000, 0.0, "low")

    assert decision.admit is False
    assert decision.reason == "prefill_eta_exceeds_deadline"
    assert decision.calibrated is True
    assert decision.budget_seconds is not None
    assert decision.budget_seconds > 0
    # The deadline field is populated on the decision.
    assert decision.prefill_deadline_seconds == 90.0
    # The own-prefill ETA exceeds the deadline.
    assert decision.eta_prefill_seconds is not None
    assert decision.eta_prefill_seconds >= decision.prefill_deadline_seconds


def test_deadline_refusal_has_active_true():
    """The deadline refusal is an active decision (logged by usage analysis)."""
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    decision = prefill_eta.evaluate_admission(_cfg(), "Qwen3", 100_000, 0.0, "low")

    assert decision.active is True
    assert decision.reason == "prefill_eta_exceeds_deadline"


def test_deadline_dominates_when_both_bounds_fire():
    """When both bounds fire the deadline is the reason: it is the binding
    constraint (that prefill will be cancelled at the deadline regardless)."""
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    # Tight budget (30 s) + default deadline: both fire for a 390 s ETA.
    cfg = _cfg(contention_queue_max_wait_seconds=30, local_prefill_deadline_seconds=90)
    decision = prefill_eta.evaluate_admission(cfg, "Qwen3", 100_000, 0.0, "low")

    assert decision.admit is False
    assert decision.reason == "prefill_eta_exceeds_deadline"


# ---------------------------------------------------------------------------
# AC2: deadline < admission → no new refusals (existing behaviour preserved)
# ---------------------------------------------------------------------------

def test_eta_within_deadline_is_admitted():
    """AC2: projected ETA < deadline keeps today's admission (no new refusals)."""
    # Calibrate a 60k bucket cache-cold at 3.9 ms/token (same bucket as the
    # request below) — prefill_ms becomes 3.9 * 1.2 = 4.68 ms/token.
    _seed_samples(tokens=60_000, ratio=0.0, ms_per_token=3.9, count=6)
    # A 95 %-cached 60k prompt: 57k coded at 0.5 ms + 3k new at 4.68 ms
    # ≈ 42.5 s < 60 s budget and < 90 s deadline ⇒ admitted exactly as today.
    decision = prefill_eta.evaluate_admission(_cfg(), "Qwen3", 60_000, 0.95, "low")

    assert decision.admit is True
    assert decision.reason == "within_budget"
    assert decision.eta_prefill_seconds is not None
    assert decision.eta_prefill_seconds < decision.prefill_deadline_seconds


def test_uncalibrated_bucket_still_admits_even_with_deadline():
    """When calibration is absent the guard is inert — today's behaviour unchanged."""
    decision = prefill_eta.evaluate_admission(_cfg(), "Qwen3", 100_000, 0.0, "low")

    assert decision.admit is True
    assert decision.active is False
    assert decision.reason == "insufficient_samples"


# ---------------------------------------------------------------------------
# AC3: boundary (ETA == deadline) refuses
# ---------------------------------------------------------------------------

def test_boundary_eta_equals_deadline_refuses():
    """AC3: boundary case — ETA == deadline is a refusal (the check is >=).

    Calibration is clamped so the effective prefill cost is at least
    ``MIN_PREFILL_MS_PER_TOKEN * CONSERVATISM_FACTOR`` = 1.2 ms/token.
    A 75k-token cache-cold prompt at the clamped minimum therefore costs
    exactly 75_000 * 1.2 / 1000 = 90 s — the default deadline, the boundary.
    """
    for _ in range(6):
        prefill_eta.record_dispatch_sample("Qwen3", 75_000, 0.0, 75_000 * 1.0)

    cfg = _cfg(local_prefill_deadline_seconds=90)
    decision = prefill_eta.evaluate_admission(cfg, "Qwen3", 75_000, 0.0, "low")

    assert decision.calibrated is True
    assert decision.eta_prefill_seconds == pytest.approx(90.0, rel=1e-6)
    assert decision.reason == "prefill_eta_exceeds_deadline"
    assert decision.admit is False


def test_disabled_deadline_does_not_refuse():
    """AC3: 0 disables the deadline bound — the deadline adds nothing on top of
    the budget bound.  A prompt within budget but far beyond the deadline is
    admitted exactly as today when the deadline is 0."""
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    # Effective ETA = 468 s (100k × 4.68 ms).  Budget 600 s > ETA: only the
    # deadline could refuse this.
    cfg = _cfg(
        contention_queue_max_wait_seconds=600,
        local_prefill_deadline_seconds=0,
    )
    decision = prefill_eta.evaluate_admission(cfg, "Qwen3", 100_000, 0.0, "low")

    assert decision.admit is True
    assert decision.reason == "within_budget"
    assert decision.prefill_deadline_seconds == 0.0
    assert decision.reason != "prefill_eta_exceeds_deadline"


# ---------------------------------------------------------------------------
# observe mode: deadline refusal is logged but not enforced
# ---------------------------------------------------------------------------

def test_observe_mode_logs_deadline_refusal():
    """In observe mode the deadline refusal is logged but not enforced."""
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    cfg = _cfg(
        local_admission_guard_policy="observe",
        local_prefill_deadline_seconds=90,
    )
    decision = prefill_eta.evaluate_admission(cfg, "Qwen3", 100_000, 0.0, "low")

    assert decision.admit is True  # observe mode never enforces
    assert decision.reason == "prefill_eta_exceeds_deadline"
    assert decision.policy == "observe"


def test_off_mode_skips_all_guards():
    """Policy off: guard is inert regardless of deadline."""
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    cfg = _cfg(
        local_admission_guard_policy="off",
        local_prefill_deadline_seconds=90,
    )
    decision = prefill_eta.evaluate_admission(cfg, "Qwen3", 100_000, 0.0, "low")

    assert decision.admit is True
    assert decision.reason == "policy_off"
    assert decision.active is False


# ---------------------------------------------------------------------------
# record_guard_decision: deadline refusal is logged correctly
# ---------------------------------------------------------------------------

def test_record_guard_decision_logs_deadline_refusal(caplog):
    """The log line carries the deadline reason and the deadline value."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    cfg = _cfg()
    decision = prefill_eta.evaluate_admission(cfg, "Qwen3", 100_000, 0.0, "low")

    assert decision.admit is False
    assert decision.reason == "prefill_eta_exceeds_deadline"

    prefill_eta.record_guard_decision(decision)

    line = next(
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("local_admission_guard")
    )
    assert "prefill_eta_exceeds_deadline" in line
    assert "deadline_s=90.0" in line
    assert "decision=reject" in line


# ---------------------------------------------------------------------------
# Direct dispatch path: reason propagates correctly
# ---------------------------------------------------------------------------

def test_direct_path_uses_deadline_reason_in_log(caplog):
    """The direct dispatch path logs ``prefill_eta_exceeds_deadline`` not
    ``eta_exceeds_budget_direct`` when the deadline is the cause."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    cfg = _cfg()
    decision = prefill_eta.evaluate_admission(cfg, "Qwen3", 100_000, 0.0, "low")

    assert decision.admit is False
    assert decision.reason == "prefill_eta_exceeds_deadline"

    # Simulate what _prefill_admission_guard does for the direct path:
    reason_label = "eta_exceeds_budget_direct"
    log_label = reason_label
    if reason_label and str(decision.reason).startswith("prefill_eta_exceeds_deadline"):
        log_label = decision.reason

    prefill_eta.record_guard_decision(
        decision, reason_label=log_label, free_slot=True,
    )

    line = next(
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("local_admission_guard")
    )
    assert "prefill_eta_exceeds_deadline" in line
    assert "free_slot=true" in line
    assert "eta_exceeds_budget_direct" not in line


def test_direct_path_budget_refusal_keeps_original_label(caplog):
    """When the budget (not deadline) fires, the direct label is preserved."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    # A 15k cache-cold prompt costs 15k * 4.68 ms ≈ 70.2 s: above the 60 s
    # budget but below the 90 s deadline ⇒ ONLY the budget bound fires and
    # the direct path keeps its ``…_direct`` log label.
    _seed_samples(tokens=15_000, ratio=0.0, ms_per_token=3.9, count=6)
    cfg = _cfg(
        contention_queue_max_wait_seconds=60,
        local_prefill_deadline_seconds=90,
    )
    decision = prefill_eta.evaluate_admission(cfg, "Qwen3", 15_000, 0.0, "low")

    assert decision.admit is False
    assert decision.reason == "eta_exceeds_budget"
    assert decision.eta_prefill_seconds < decision.prefill_deadline_seconds

    reason_label = "eta_exceeds_budget_direct"
    log_label = reason_label  # stays

    prefill_eta.record_guard_decision(
        decision, reason_label=log_label, free_slot=True,
    )

    line = next(
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("local_admission_guard")
    )
    assert "eta_exceeds_budget_direct" in line
    assert "prefill_eta_exceeds_deadline" not in line


# ---------------------------------------------------------------------------
# AC1 end-to-end: both dispatch paths refuse and reach the next provider
# ---------------------------------------------------------------------------

class _BusyConcurrency:
    """All slots busy ⇒ the queue-entry guard runs."""

    def __call__(self, config, endpoint=None) -> tuple[int, int]:
        return (1, 1)


class _FreeConcurrency:
    """A free slot ⇒ the direct/free-slot guard runs."""

    def __call__(self, config, endpoint=None) -> tuple[int, int]:
        return (0, 1)


def _ok():
    import json

    from fastapi import Response

    return Response(
        content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
        status_code=200,
        media_type="application/json",
    )


class _DummyRequest:
    def __init__(self, body: bytes = b'{"model":"test"}'):
        self._body = body
        self.headers = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


def _integration_cfg(deadline: float = 90, **server_overrides) -> dict:
    server = {
        "session_slot_pool_size": 1,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": 60,
        "contention_queue_max_depth": 4,
        "local_prefill_deadline_seconds": deadline,
    }
    server.update(server_overrides)
    return {"provider_cooldown_seconds": 60, "server": server}


@pytest.fixture
def mixed_model_config():
    return {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {
                "name": "remote-fallback",
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
            },
        ],
        "aliases": ["hybrid*"],
    }


@contextmanager
def _guard_patches(tokens: int):
    from unittest.mock import AsyncMock, patch

    with (
        patch(
            "proxy.provider._estimate_effective_prompt_tokens_for_routing",
            new=AsyncMock(return_value=tokens),
        ),
        patch("proxy.provider._get_tokenizer_for_model", lambda *_: (None, 1.0)),
        patch("proxy.provider._should_skip_local", lambda *a, **k: (False, None)),
        patch("proxy.provider._get_cached_ratio", lambda *_: 0.0),
    ):
        yield


@pytest.mark.asyncio
async def test_queue_path_refuses_on_deadline_and_reaches_next_provider(
    mixed_model_config, caplog
):
    """AC1 (queue path): slots busy ⇒ guard refuses with the deadline reason
    and the request dispatches to the remote provider."""
    from unittest.mock import patch

    import proxy.provider as provider

    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok()

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _BusyConcurrency()),
        patch("proxy.provider.request_priority", return_value="low"),
        _guard_patches(tokens=100_000),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", mixed_model_config,
            _integration_cfg(deadline=90),
        )

    assert result.status_code == 200
    assert calls == ["remote"], "a deadline-refused request must reach the next provider"
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "local_admission_guard decision=reject" in messages
    assert "reason=prefill_eta_exceeds_deadline" in messages
    assert "free_slot=false" in messages, "the queue-entry guard must be the one refusing"


@pytest.mark.asyncio
async def test_direct_path_refuses_on_deadline_and_reaches_next_provider(
    mixed_model_config, caplog
):
    """AC1 (direct/free-slot path): a free slot does not bypass the deadline."""
    from unittest.mock import patch

    import proxy.provider as provider

    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok()

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _FreeConcurrency()),
        patch("proxy.provider.request_priority", return_value="low"),
        _guard_patches(tokens=100_000),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", mixed_model_config,
            _integration_cfg(deadline=90),
        )

    assert result.status_code == 200
    assert calls == ["remote"], "a free slot must NOT bypass the deadline bound"
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "local_admission_guard decision=reject" in messages
    assert "reason=prefill_eta_exceeds_deadline" in messages
    assert "free_slot=true" in messages
