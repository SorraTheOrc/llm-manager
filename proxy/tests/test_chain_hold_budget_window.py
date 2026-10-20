"""Chain-hold on free-tier budget-window exhaustion (LP-0MVF0N9SA0071LWM).

When a fallback chain exhausts *only* because the proactive free-tier request
budget (``rate_limit_rpm`` sliding window, see ``proxy.provider_budget``) is
momentarily full, the exhaustion payload carries no cooldown and no
usage-limit quarantine (``unavailable={}``), so ``_compute_retry_after`` has
no candidate and returned ``0``. ``_bounded_hold_seconds`` then treated the
exhaustion as "instant recovery" and held ``0.0`` s, replaying every
``chain_hold_max_cycles`` cycle inside the same full window and surfacing a
spurious "All providers exhausted" to the client.

The behaviour these tests pin down:

- AC1: a chain whose only blockers are budget skips derives a POSITIVE
  ``retry_after`` from the rolling-window rollover, so the chain hold waits
  (bounded) and the next cycle succeeds once the window frees.
- AC2: a genuine instant-recovery exhaustion (real ``retry_after == 0``) still
  retries with no added latency.
- AC3: the cumulative hold stays within ``chain_hold_total_seconds`` and thus
  strictly below the client's 900 s deadline.
- AC4: the 503 exhaustion body carries an accurate ``Retry-After`` /
  ``retry_after`` derived from the window rollover.
"""

from __future__ import annotations

import json
import math

import proxy.provider as provider
import pytest
from fastapi import Response
from proxy.provider_budget import RATE_LIMIT_WINDOW_SECONDS, reset_provider_budget_state
from proxy.rate_limiter import get_rate_limiter

FREE_ENTRY = "munitoai-free"
# The measured production scenario: 13 requests admitted into an 11-request
# window (the limiter admits up to ``limit_rpm``; the extra two came in before
# the window filled — the log line is "13/11 requests in 60s window").
MEASURED_LIMIT = 11
MEASURED_RESERVE = 4


class _Req:
    """Minimal request stub (mirrors the other chain-hold tests)."""

    def __init__(self, stream: bool = False):
        self._body = b'{"model":"test","stream":true}' if stream else b'{"model":"test"}'
        self.headers = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body

    async def is_disconnected(self):
        return False


def _model_config(limit: int = MEASURED_LIMIT) -> dict:
    return {
        "name": "v1/chat/completions",
        "providers": [
            {"name": FREE_ENTRY, "type": "remote", "rate_limit_rpm": limit},
        ],
    }


def _budget_skip_attempts(limit: int = MEASURED_LIMIT) -> list[dict]:
    """Diagnostics recorded when the budget gate denied every provider."""
    return [
        {
            "provider": FREE_ENTRY,
            "type": "remote",
            "status": "rate_limited",
            "status_code": 429,
            "reason": "rate_limited",
            "limit_rpm": limit,
            "effective_limit": limit - MEASURED_RESERVE,
            "reserve_slots": MEASURED_RESERVE,
            "remaining": 2,
            "elapsed_wait_seconds": 0.0,
        }
    ]


async def _fill_window(key: str = FREE_ENTRY, limit: int = MEASURED_LIMIT) -> None:
    """Fill the rolling window so the gate denies the next dispatch."""
    limiter = get_rate_limiter()
    for _ in range(limit):
        allowed = await limiter.check_and_increment(
            key, limit, RATE_LIMIT_WINDOW_SECONDS
        )
        assert allowed, "window must not overflow while filling"
    assert (
        limiter.seconds_until_capacity(key, limit, RATE_LIMIT_WINDOW_SECONDS) > 0
    ), "a full window must have a positive rollover"


def _payload(response: Response) -> dict:
    body = response.body
    if isinstance(body, bytes):
        body = body.decode()
    return json.loads(body)


@pytest.fixture(autouse=True)
def _reset_state():
    reset_provider_budget_state()
    limiter = get_rate_limiter()
    limiter.reset(FREE_ENTRY)
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    yield
    limiter.reset(FREE_ENTRY)
    reset_provider_budget_state()
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()


# ---------------------------------------------------------------------------
# AC4: the exhaustion body advertises the window rollover
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_skip_exhaustion_advertises_window_rollover():
    """AC4: retry_after + Retry-After come from the rolling-window rollover."""
    await _fill_window()
    expected = math.ceil(
        get_rate_limiter().seconds_until_capacity(
            FREE_ENTRY, MEASURED_LIMIT, RATE_LIMIT_WINDOW_SECONDS
        )
    )
    assert expected > 0

    response = provider._build_exhausted_response(
        unavailable_providers={},
        diagnostics=_budget_skip_attempts(),
        model_config=_model_config(),
        config={},
    )

    assert response.status_code == 503
    payload = _payload(response)
    assert payload["retry_after"] == expected
    assert response.headers["Retry-After"] == str(expected)
    # The rollover can never exceed the window it is derived from.
    assert 0 < payload["retry_after"] <= RATE_LIMIT_WINDOW_SECONDS


@pytest.mark.asyncio
async def test_budget_skip_rollover_uses_earliest_freeing_provider():
    """A chain succeeds as soon as SOME budget-skipped provider frees."""
    await _fill_window()
    second = "second-free-tier"
    limiter = get_rate_limiter()
    limiter.reset(second)
    try:
        # The second entry has only just filled: its oldest entry expires
        # later than the first entry's, so the first is the earlier rollover.
        await _fill_window(second, 2)
        attempts = _budget_skip_attempts() + [
            {
                "provider": second,
                "type": "remote",
                "status": "rate_limited",
                "reason": "rate_limited",
                "limit_rpm": 2,
                "effective_limit": 2,
                "remaining": 0,
            }
        ]
        model_config = _model_config()
        model_config["providers"].append(
            {"name": second, "type": "remote", "rate_limit_rpm": 2}
        )
        response = provider._build_exhausted_response(
            unavailable_providers={},
            diagnostics=attempts,
            model_config=model_config,
            config={},
        )
        expected = math.ceil(
            limiter.seconds_until_capacity(
                FREE_ENTRY, MEASURED_LIMIT, RATE_LIMIT_WINDOW_SECONDS
            )
        )
        assert _payload(response)["retry_after"] == expected
    finally:
        limiter.reset(second)


# ---------------------------------------------------------------------------
# AC1: positive hold, and the next cycle succeeds once the window frees
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_skip_exhaustion_holds_positive_and_next_cycle_succeeds(monkeypatch):
    """AC1 regression: the measured 13/11 budget-window exhaustion."""
    await _fill_window()
    exhaustion = provider._build_exhausted_response(
        unavailable_providers={},
        diagnostics=_budget_skip_attempts(),
        model_config=_model_config(),
        config={},
    )
    advertised = _payload(exhaustion)["retry_after"]
    assert advertised > 0, "budget-window exhaustion must advertise a wait"

    holds: list[float] = []
    calls: list[int] = []

    async def fake_hold(_request, seconds):
        # The hold elapses and the window rolls over: drop the recorded
        # dispatch timestamps so the next cycle finds capacity.
        holds.append(seconds)
        get_rate_limiter().reset(FREE_ENTRY)
        return False

    async def cycle_fn(_request, _path, _model, _config):
        calls.append(1)
        if len(calls) == 1:
            raise provider.ChainExhaustedError(exhaustion)
        return Response(status_code=200, content=b"ok")

    monkeypatch.setattr(provider, "_hold_sleep", fake_hold)
    result = await provider._run_chain_cycles(
        _Req(),
        "v1/chat/completions",
        _model_config(),
        {"chain_hold_seconds": 300, "chain_hold_max_cycles": 3},
        cycle_fn,
    )

    assert result.status_code == 200, "the cycle after the rollover must succeed"
    assert holds == [float(advertised)], "the hold waits for the window rollover"
    assert len(calls) == 2, "one exhausted cycle, then one successful cycle"


@pytest.mark.asyncio
async def test_rollover_beyond_hold_budget_returns_immediately(monkeypatch):
    """A rollover that cannot fit the hold budget must not hold at all.

    Holding for less than the window rollover could never make the next cycle
    succeed, so the exhaustion is surfaced at once (LP-0MU56ZKQD005SX08's
    doomed-hold guard, now reachable from a window-derived ``retry_after``).
    """
    await _fill_window()
    exhaustion = provider._build_exhausted_response(
        unavailable_providers={},
        diagnostics=_budget_skip_attempts(),
        model_config=_model_config(),
        config={},
    )
    holds: list[float] = []

    async def fake_hold(_request, seconds):
        holds.append(seconds)
        return False

    async def cycle_fn(_request, _path, _model, _config):
        raise provider.ChainExhaustedError(exhaustion)

    monkeypatch.setattr(provider, "_hold_sleep", fake_hold)
    result = await provider._run_chain_cycles(
        _Req(),
        "v1/chat/completions",
        _model_config(),
        {"chain_hold_seconds": 5, "chain_hold_max_cycles": 1},
        cycle_fn,
    )

    assert result.status_code == 503
    assert holds == [], "a doomed hold is never entered"
    assert result.headers.get("Retry-After") is not None


@pytest.mark.asyncio
async def test_streaming_budget_skip_hold_emits_positive_feedback(monkeypatch):
    """The streaming hold path also waits for the window rollover."""
    await _fill_window()
    exhaustion = provider._build_exhausted_response(
        unavailable_providers={},
        diagnostics=_budget_skip_attempts(),
        model_config=_model_config(),
        config={},
    )
    advertised = _payload(exhaustion)["retry_after"]

    held: list[float] = []

    async def fake_feedback(_request, hold_seconds, comment):
        # The hold elapses and the window rolls over.
        held.append(hold_seconds)
        get_rate_limiter().reset(FREE_ENTRY)
        yield (comment + "\n\n").encode("utf-8")

    calls: list[int] = []

    async def cycle_fn(_request, _path, _model, _config):
        calls.append(1)
        if len(calls) == 1:
            raise provider.ChainExhaustedError(exhaustion)
        return Response(status_code=200, content=b"ok")

    monkeypatch.setattr(provider, "_hold_feedback", fake_feedback)
    result = await provider._run_chain_cycles(
        _Req(stream=True),
        "v1/chat/completions",
        _model_config(),
        {"chain_hold_seconds": 300, "chain_hold_max_cycles": 3},
        cycle_fn,
    )

    chunks = []
    async for chunk in result.body_iterator:
        chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk))
    collected = "".join(chunks)

    assert f"in {advertised}s" in collected, "SSE feedback names the rollover wait"
    assert held == [float(advertised)], "the streaming hold waits for the rollover"


# ---------------------------------------------------------------------------
# AC2: genuine instant-recovery exclusions keep zero added latency
# ---------------------------------------------------------------------------


def test_zero_retry_after_without_budget_skips_keeps_zero_hold():
    """AC2: a real 0 ``retry_after`` (instant recovery) adds no latency."""
    response = provider._build_exhausted_response(
        unavailable_providers={},
        diagnostics=[],
        model_config=_model_config(),
        config={},
    )
    assert _payload(response)["retry_after"] == 0
    assert provider._bounded_hold_seconds(300.0, response, 3) == 0.0


def test_explicit_zero_retry_after_response_keeps_zero_hold():
    """AC2: an exhaustion response that explicitly says 0 still retries now."""
    response = Response(
        content=json.dumps(
            {"error": "All providers exhausted", "retry_after": 0}
        ).encode(),
        status_code=503,
        media_type="application/json",
        headers={"Retry-After": "0"},
    )
    assert provider._bounded_hold_seconds(300.0, response, 3) == 0.0


@pytest.mark.asyncio
async def test_budget_derivation_skipped_when_a_real_cooldown_sets_retry_after():
    """A cooldown-derived wait wins: only cooldown-free exhaustion uses the window."""
    await _fill_window()
    response = provider._build_exhausted_response(
        unavailable_providers={FREE_ENTRY: 120},
        diagnostics=_budget_skip_attempts(),
        model_config=_model_config(),
        config={},
    )
    assert _payload(response)["retry_after"] == 120


def test_budget_derivation_ignores_free_capacity_and_non_budget_attempts():
    """No derivation when the window is not the blocker."""
    limiter = get_rate_limiter()
    limiter.reset(FREE_ENTRY)
    # Empty window: capacity is available now → no invented wait.
    assert provider._budget_window_retry_after(_budget_skip_attempts()) == 0
    # Non-budget diagnostics are not a budget-window blocker.
    assert (
        provider._budget_window_retry_after(
            [{"provider": FREE_ENTRY, "status": "http_error", "reason": "HTTP 500"}]
        )
        == 0
    )
    assert provider._budget_window_retry_after([]) == 0
    assert provider._budget_window_retry_after(None) == 0


@pytest.mark.asyncio
async def test_rate_limited_waited_attempt_is_not_a_blocker():
    """An attempt that waited then dispatched is not a budget blocker."""
    await _fill_window()
    assert (
        provider._budget_window_retry_after(
            [
                {
                    "provider": FREE_ENTRY,
                    "status": "rate_limited_waited",
                    "reason": "allowed_after_wait",
                    "limit_rpm": MEASURED_LIMIT,
                    "effective_limit": MEASURED_LIMIT,
                }
            ]
        )
        == 0
    )


# ---------------------------------------------------------------------------
# AC3: cumulative hold stays within chain_hold_total_seconds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_window_holds_respect_total_budget(monkeypatch):
    """AC3: cumulative wait never exceeds ``chain_hold_total_seconds``."""
    await _fill_window()
    exhaustion = provider._build_exhausted_response(
        unavailable_providers={},
        diagnostics=_budget_skip_attempts(),
        model_config=_model_config(),
        config={},
    )
    advertised = _payload(exhaustion)["retry_after"]

    holds: list[float] = []

    async def fake_hold(_request, seconds):
        # Never roll the window over: every cycle re-exhausts at the budget.
        holds.append(seconds)
        return False

    async def cycle_fn(_request, _path, _model, _config):
        raise provider.ChainExhaustedError(exhaustion)

    monkeypatch.setattr(provider, "_hold_sleep", fake_hold)
    result = await provider._run_chain_cycles(
        _Req(),
        "v1/chat/completions",
        _model_config(),
        {
            "chain_hold_seconds": 300,
            "chain_hold_max_cycles": 3,
            # Two full rollovers worth of budget, i.e. an even split.
            "chain_hold_total_seconds": float(advertised * 2),
        },
        cycle_fn,
    )

    assert result.status_code == 503
    assert sum(holds) == pytest.approx(float(advertised * 2), abs=0.01)
    assert sum(holds) < 900, "strictly below the client's 900 s deadline"
