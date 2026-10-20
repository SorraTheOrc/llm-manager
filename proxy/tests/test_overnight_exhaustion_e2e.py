"""End-to-end overnight-gap exhaustion contract (LP-0MUI68NTH004SZMG).

One deterministic integration test composing the real overnight blockers and
asserting the terminal client contract delivered by the epic's child fixes:

- all remote providers outside their ``available_times`` window,
- at least one account under ``usage_limit_reset_pending``,
- (lease variant) the single local slot held by another session,

then asserts: truthful 503, stable machine-readable ``code``, quarantine
visible in diagnostics, truthful ``Retry-After``, and a bounded hold that never
sleeps through ``chain_hold_max_cycles`` when ``retry_after`` is hours away.

The retry-timer contract is pinned (LP-0MV728V6M0013J4G): under the injected
clock the asserted ``retry_after`` equals the exact soonest availability (the
``available_times`` window edge), the ``Retry-After`` header and ``retry_after``
body field carry the same integer, both exhaustion variants (``all_exhausted``
and ``outside_time_window``) are covered through the fallback chain, and ``0``
is asserted exactly when a provider can serve immediately.

Determinism: the UTC clock is injected and the hold sleep is monkeypatched to
record requested durations instead of sleeping (no real sleep > 1s).

Regression evidence: all three tests fail against the pre-fix commit
``52307b7f`` (the hold is unbounded at 300s and the response has no ``code``
and no quarantine visibility), and pass once the epic's children land.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import proxy.provider as provider
import pytest
from fastapi import Response

_FIXED_NOW = datetime(2026, 1, 5, 13, 0, 0, tzinfo=UTC)  # 13:00 UTC
_OVERNIGHT_WINDOW = ["00:00-01:00"]  # excludes 13:00 → next edge ~11h away


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return _FIXED_NOW


class _DummyRequest:
    def __init__(self, body: bytes = b'{"model":"test"}'):
        self._body = body
        self.headers = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body

    async def is_disconnected(self):
        return False


class _FreeConcurrency:
    def __call__(self, config, endpoint=None):
        return (0, 1)


def _lease_active_response() -> Response:
    return Response(
        content=json.dumps({
            "error": {
                "code": "no_slots_available",
                "type": "server_busy",
                "message": "Local dispatch lease is held by another session",
            },
            "total_slots": 1,
            "available_slots": 0,
            "reason": "local_lease_active",
        }).encode("utf-8"),
        status_code=503,
        media_type="application/json",
    )


@pytest.fixture(autouse=True)
def _reset_state():
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()
    yield
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()


@pytest.fixture(autouse=True)
def _pin_fast_mode(monkeypatch):
    monkeypatch.setattr("proxy.priority.request_priority", lambda *a, **k: "medium")


def _remote_providers_for_overnight() -> dict:
    quarantined = {
        "name": "opencode-go-3",
        "type": "remote",
        "provider": "opencode-go",
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": "OPENCODE_3_API_KEY",
        "model": "deepseek-v4-flash",
        "available_times": _OVERNIGHT_WINDOW,
    }
    windowed = {
        "name": "opencode-go-2",
        "type": "remote",
        "provider": "opencode-go",
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": "OPENCODE_2_API_KEY",
        "model": "deepseek-v4-flash",
        "available_times": _OVERNIGHT_WINDOW,
    }
    return {"providers": [quarantined, windowed]}


async def _collect(result) -> str:
    chunks = []
    async for chunk in result.body_iterator:
        chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk))
    return "".join(chunks)


@pytest.mark.asyncio
async def test_overnight_gap_window_plus_quarantine_contract(monkeypatch):
    """Windows + usage-limit quarantine composed: truthful terminal 503 with a
    stable code, quarantine visibility, truthful Retry-After, and no hold."""
    model = _remote_providers_for_overnight()
    quarantined = model["providers"][0]
    reset_seconds = 9 * 3600  # ~9h, like the observed opencode-go-3
    provider._usage_reset_at[
        provider._usage_limit_account_key(quarantined)
    ] = time.time() + reset_seconds

    hold_requests: list[float] = []

    async def _record_hold(_req, seconds):
        hold_requests.append(seconds)
        return False

    monkeypatch.setattr(provider, "_hold_sleep", _record_hold)

    with patch("proxy.provider.datetime", _FixedDateTime):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", model,
            {"provider_cooldown_seconds": 60,
             "chain_hold_seconds": 300, "chain_hold_max_cycles": 3,
             # Raise the client cap so the pinned honest wait is advertised
             # unclamped (AC2: retry_after equals the real soonest
             # availability, not the default 3600 s ceiling).
             "max_retry_after_seconds": 100000},
        )

    assert result.status_code == 503
    body = json.loads(result.body)
    # Truthful error + stable machine-readable discriminator.
    assert body["error"] == "All providers exhausted"
    assert body["code"] == "all_exhausted"

    # Quarantine visibility: reset seconds + distinct reason, not folded into
    # the window reason.
    assert "opencode-go-3" in body["unavailable_providers"]
    assert body["unavailable_providers"]["opencode-go-3"] >= reset_seconds - 5
    statuses = {d["provider"]: d for d in body["diagnostics"]}
    assert statuses["opencode-go-3"]["status"] == "usage_limit_reset"
    assert statuses["opencode-go-2"]["status"] == "outside_time_window"

    # Truthful Retry-After, pinned to the real soonest availability (AC2):
    # under _FixedDateTime (13:00 UTC) the next "00:00-01:00" window edge is
    # 11 h away, ahead of the 9 h usage-limit reset, so the advertised wait is
    # exactly that window edge.
    window_edge = 11 * 3600
    assert provider._compute_retry_after(
        model_config=model, now_utc=_FIXED_NOW
    ) == window_edge
    assert body["retry_after"] == window_edge
    assert result.headers.get("retry-after") == str(body["retry_after"])
    # AC5: the 11 h timer exceeds the 300 s × 3 hold budget, so the exhaustion
    # is surfaced immediately with the timer, never held beyond it.
    assert hold_requests == [], "a >budget retry_after must not hold/cycle"


@pytest.mark.asyncio
async def test_overnight_gap_lease_denied_remote_window_contract(monkeypatch):
    """Lease-denied local slot + all remotes out of window composed: a prompt
    terminal error within the contention-queue budget, never a silent
    multi-cycle hold."""

    model = {
        "providers": [
            {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"},
            {
                "name": "opencode-go-2",
                "type": "remote",
                "provider": "opencode-go",
                "endpoint": "https://opencode.ai/zen/go",
                "api_key_env": "OPENCODE_2_API_KEY",
                "model": "deepseek-v4-flash",
                "available_times": _OVERNIGHT_WINDOW,
            },
        ]
    }
    cfg = {
        "provider_cooldown_seconds": 60,
        "chain_hold_seconds": 300,
        "chain_hold_max_cycles": 3,
        "server": {
            "session_slot_pool_size": 1,
            "contention_queue_policy": "queue",
            "contention_queue_max_wait_seconds": 120,
            "contention_queue_max_depth": 4,
        },
    }
    hold_requests: list[float] = []

    async def _record_hold(_req, seconds):
        hold_requests.append(seconds)
        return False

    async def _lease_denied(_req, _path):
        return _lease_active_response()

    monkeypatch.setattr(provider, "_hold_sleep", _record_hold)
    monkeypatch.setattr(provider, "_get_local_concurrency_info", _FreeConcurrency())

    with (
        patch("proxy.router.proxy_to_local", _lease_denied),
        patch("proxy.server.proxy_to_remote",
              AsyncMock(side_effect=AssertionError("out-of-window remote must not run"))),
        patch("proxy.provider.datetime", _FixedDateTime),
        patch.object(provider, "_maybe_queue_for_local_slot",
                     new=AsyncMock(return_value=("fallback", None, 120.0))),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", model, cfg
        )

    assert result.status_code == 503
    body = json.loads(result.body)
    assert body["error"] == "All providers exhausted"
    assert body["code"] == "all_exhausted"
    diag = json.dumps(body.get("diagnostics", []))
    assert "local_lease_active" in diag or "fallback_after_queue" in diag
    assert body["retry_after"] > 0
    assert hold_requests == [], "lease denial + windowed remotes must not silently hold"


@pytest.mark.asyncio
async def test_overnight_gap_hold_bounded_by_small_retry_after(monkeypatch):
    """Medium/small retry_after: the hold tracks real availability and the
    next cycle runs (no real sleep)."""
    cycle_responses = [
        Response(
            content=json.dumps({"error": "All providers exhausted", "retry_after": 2}).encode(),
            status_code=503, media_type="application/json",
            headers={"Retry-After": "2"},
        ),
    ]

    async def cycle_fn(_req, _path, _model, _config):
        if cycle_responses:
            raise provider.ChainExhaustedError(cycle_responses.pop(0))
        return Response(status_code=200, content=b"ok")

    holds: list[float] = []

    async def _record_hold(_req, seconds):
        holds.append(seconds)
        return False

    monkeypatch.setattr(provider, "_hold_sleep", _record_hold)
    result = await provider._run_chain_cycles(
        _DummyRequest(), "v1/chat/completions", {},
        {"chain_hold_seconds": 300, "chain_hold_max_cycles": 3},
        cycle_fn,
    )

    assert result.status_code == 200
    assert holds == [2.0], "the hold must equal the real retry_after (2s)"
    assert max(holds) <= 2, "hold must never exceed the computed retry_after"


@pytest.mark.asyncio
async def test_overnight_sole_cause_window_code_and_timer(monkeypatch):
    """Sole-cause time-window exhaustion through the fallback chain: the
    distinguished ``outside_time_window`` 503 (LP-0MV728V6M0013J4G AC3)
    carries the same pinned retry-timer contract as the generic variant."""
    model = _remote_providers_for_overnight()  # both remotes outside the window
    hold_requests: list[float] = []

    async def _record_hold(_req, seconds):
        hold_requests.append(seconds)
        return False

    monkeypatch.setattr(provider, "_hold_sleep", _record_hold)
    monkeypatch.setattr(
        "proxy.server.proxy_to_remote",
        AsyncMock(side_effect=AssertionError("out-of-window remote must not run")),
    )

    with patch("proxy.provider.datetime", _FixedDateTime):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", model,
            {"provider_cooldown_seconds": 60,
             "chain_hold_seconds": 300, "chain_hold_max_cycles": 3,
             "max_retry_after_seconds": 100000},
        )

    assert result.status_code == 503
    body = json.loads(result.body)
    # AC4: the human-readable prose is retained; AC3: the stable code
    # discriminates the time-window variant.
    assert body["error"] == "All providers exhausted"
    assert body["code"] == "outside_time_window"
    assert all(
        d["status"] == "outside_time_window" for d in body["diagnostics"]
    )

    # AC2: the pinned timer equals the real window-edge wait under
    # _FixedDateTime, and the wire advertises the same value in header and
    # body (AC1).
    window_edge = 11 * 3600
    assert provider._compute_retry_after(
        model_config=model, now_utc=_FIXED_NOW
    ) == window_edge
    assert body["retry_after"] == window_edge
    assert result.headers.get("retry-after") == str(body["retry_after"])

    # AC5: the timer exceeds the hold budget (900 s), so no hold is attempted.
    assert hold_requests == [], "window-only exhaustion must not hold/cycle"


def test_exhausted_zero_when_availability_exists():
    """AC2 edge: ``retry_after`` is 0 exactly when a provider can serve now.

    Even with ``available_times`` configured, a provider currently inside its
    window contributes no wait, so the pinned clock and the built response
    agree on 0 (deterministic under ``_FixedDateTime``).
    """
    in_window = {
        "name": "opencode-go-2",
        "type": "remote",
        "provider": "opencode-go",
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": "OPENCODE_2_API_KEY",
        "model": "deepseek-v4-flash",
        "available_times": ["12:00-14:00"],  # contains _FIXED_NOW (13:00 UTC)
    }
    model = {"providers": [in_window]}
    assert provider._compute_retry_after(
        model_config=model, now_utc=_FIXED_NOW
    ) == 0
    with patch("proxy.provider.datetime", _FixedDateTime):
        r = provider._build_exhausted_response(
            unavailable_providers={}, diagnostics=[], model_config=model,
        )
    body = json.loads(r.body)
    assert body["retry_after"] == 0
    assert r.headers.get("retry-after") == "0"
