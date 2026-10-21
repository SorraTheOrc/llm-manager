"""Tests for per-provider request-budget gating (LP-0MVDQ1GMO00801PL).

The free-tier provider entry (``munitoai-free``) has a request-rate allowance —
N requests per rolling 60-second window. This module implements the gate that:

- dispatches to the entry only while the window still has capacity (AC1);
- makes an over-budget request from a *new* session skip the entry immediately,
  with no waiting at all (AC2);
- lets a request whose session is already using the entry wait a bounded time
  for the window to free capacity, then pass through if it never frees (AC3);
- records/expires the per-session affinity that decides "in progress" (AC4);
- logs the decision with limit/remaining/reason (AC5).

The tests below assert observable behaviour: dispatch counts through the public
fallback entry points, the returned decision, the recorded attempts, and the
log lines — not implementation details.
"""

import json
import time
from unittest.mock import patch

import proxy.provider as provider
import pytest
from fastapi import Response
from proxy.provider_budget import (
    acquire_provider_budget,
    provider_affinity_active,
    record_provider_affinity,
    reset_provider_budget_state,
)
from proxy.rate_limiter import SlidingWindowRateLimiter, get_rate_limiter

from proxy import provider_budget

FREE_ENTRY = "test-free-tier"
PAID_ENTRY = "test-paid-tier"


class _DummyRequest:
    """Minimal request stub (mirrors the other provider fallback tests)."""

    def __init__(self, body: bytes = b'{"model":"test"}', session_id: str | None = None):
        self._body = body
        self.headers = {"x-session-id": session_id} if session_id else {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body

    async def is_disconnected(self):
        return False


class _DisconnectedRequest(_DummyRequest):
    async def is_disconnected(self):
        return True


def _ok_json_response() -> Response:
    return Response(
        content=json.dumps(
            {
                "id": "cmpl-test",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"}}
                ],
            }
        ),
        status_code=200,
        media_type="application/json",
    )


def _error_response(status_code: int = 429) -> Response:
    return Response(
        content=json.dumps({"error": {"message": "rate limited"}}),
        status_code=status_code,
        media_type="application/json",
    )


@pytest.fixture(autouse=True)
def reset_state():
    """Clear limiter buckets, affinity records and cooldowns around each test."""
    get_rate_limiter().reset(FREE_ENTRY)
    get_rate_limiter().reset(PAID_ENTRY)
    reset_provider_budget_state()
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    yield
    get_rate_limiter().reset(FREE_ENTRY)
    get_rate_limiter().reset(PAID_ENTRY)
    reset_provider_budget_state()
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()


@pytest.fixture
def clock(monkeypatch):
    """Controllable clock shared by the limiter and the budget module."""
    state = {"now": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: state["now"])
    return state


@pytest.fixture
def sleeps(monkeypatch, clock):
    """Replace the disconnect-aware sleep: advance the fake clock instead."""
    recorded: list[float] = []

    async def _fake_sleep(request, seconds):
        recorded.append(seconds)
        clock["now"] += seconds
        return False

    monkeypatch.setattr(provider_budget, "_sleep_with_disconnect", _fake_sleep)
    return recorded


def _free_provider_cfg(rpm: int = 2, name: str = FREE_ENTRY) -> dict:
    return {
        "name": name,
        "type": "remote",
        "endpoint": "https://free.example.com/v1",
        "rate_limit_rpm": rpm,
    }


def _budget_config(**server_overrides) -> dict:
    server = {"provider_rate_limit_wait_max_seconds": 60, "provider_affinity_ttl_seconds": 900}
    server.update(server_overrides)
    return {"server": server}


# ---------------------------------------------------------------------------
# rate_limiter: capacity reporting used by the bounded wait
# ---------------------------------------------------------------------------


def test_seconds_until_capacity_reports_window_expiry(clock):
    """seconds_until_capacity() reports when the oldest call leaves the window."""
    limiter = SlidingWindowRateLimiter()
    limiter._windows[FREE_ENTRY].extend([clock["now"], clock["now"]])

    # Full: capacity frees when the oldest entry is 60s old.
    assert limiter.seconds_until_capacity(FREE_ENTRY, max_rpm=2) == pytest.approx(60.0)

    clock["now"] += 15.0
    assert limiter.seconds_until_capacity(FREE_ENTRY, max_rpm=2) == pytest.approx(45.0)

    clock["now"] += 100.0
    assert limiter.seconds_until_capacity(FREE_ENTRY, max_rpm=2) == 0.0


# ---------------------------------------------------------------------------
# AC1 — the budget gate engages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_allows_up_to_limit_then_denies():
    """AC1: exactly rate_limit_rpm dispatches fit in one window."""
    cfg = _free_provider_cfg(rpm=2)
    config = _budget_config()

    first = await acquire_provider_budget(cfg, config, session_id="s1")
    second = await acquire_provider_budget(cfg, config, session_id="s2")
    third = await acquire_provider_budget(cfg, config, session_id="s3")

    assert first.allowed and second.allowed
    assert not third.allowed
    assert third.limit_rpm == 2
    assert third.remaining == 0


@pytest.mark.asyncio
async def test_unset_rate_limit_is_unlimited():
    """No rate_limit_rpm => today's behaviour, no budget recorded."""
    cfg = {"name": FREE_ENTRY, "type": "remote", "endpoint": "https://x/v1"}
    config = _budget_config()

    for _ in range(50):
        decision = await acquire_provider_budget(cfg, config, session_id="s1")
        assert decision.allowed
        assert decision.reason == "unlimited"
    assert get_rate_limiter().remaining(FREE_ENTRY, max_rpm=5) == 5


# ---------------------------------------------------------------------------
# AC2 — new sessions pass through, never wait
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_session_over_budget_never_waits(sleeps):
    """AC2: no affinity => immediate denial, no sleep at all."""
    cfg = _free_provider_cfg(rpm=1)
    config = _budget_config()

    assert (await acquire_provider_budget(cfg, config, session_id="new-a")).allowed
    decision = await acquire_provider_budget(cfg, config, session_id="new-b")

    assert not decision.allowed
    assert decision.reason == "rate_limited"
    assert decision.waited_seconds == 0.0
    assert sleeps == [], "a new session must never sleep waiting for capacity"


@pytest.mark.asyncio
async def test_sessionless_request_never_waits(sleeps):
    """A request with no session identity cannot be 'in progress' => no wait."""
    cfg = _free_provider_cfg(rpm=1)
    config = _budget_config()

    assert (await acquire_provider_budget(cfg, config, session_id=None)).allowed
    decision = await acquire_provider_budget(cfg, config, session_id=None)

    assert not decision.allowed
    assert sleeps == []


@pytest.mark.asyncio
async def test_skip_log_names_limit_remaining_and_reason(caplog):
    """AC5: the skip decision is logged with entry, limit, remaining and why."""
    cfg = _free_provider_cfg(rpm=1)
    config = _budget_config()

    assert (await acquire_provider_budget(cfg, config, session_id="a")).allowed
    with caplog.at_level("WARNING", logger="llama-proxy.provider_budget"):
        decision = await acquire_provider_budget(cfg, config, session_id="b")

    assert not decision.allowed
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert FREE_ENTRY in text
    assert "limit=1" in text
    assert "remaining=0" in text
    assert "reason=rate_limited" in text


# ---------------------------------------------------------------------------
# AC3 — in-progress sessions wait, bounded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_affinity_session_waits_then_dispatches(sleeps):
    """AC3: an in-progress session waits for the window, then gets the slot."""
    cfg = _free_provider_cfg(rpm=1)
    config = _budget_config()

    assert (await acquire_provider_budget(cfg, config, session_id="busy")).allowed
    record_provider_affinity("busy", FREE_ENTRY)

    decision = await acquire_provider_budget(cfg, config, session_id="busy")

    assert decision.allowed
    assert decision.reason == "allowed_after_wait"
    assert decision.waited_seconds == pytest.approx(60.0)
    assert sleeps == [pytest.approx(60.0)]


@pytest.mark.asyncio
async def test_affinity_session_passes_through_after_wait_cap(sleeps):
    """AC3: waiting is capped; after the cap the request falls through."""
    cfg = _free_provider_cfg(rpm=1)
    config = _budget_config(provider_rate_limit_wait_max_seconds=5)

    assert (await acquire_provider_budget(cfg, config, session_id="busy")).allowed
    record_provider_affinity("busy", FREE_ENTRY)

    decision = await acquire_provider_budget(cfg, config, session_id="busy")

    assert not decision.allowed
    assert decision.reason == "rate_limited_after_wait"
    assert decision.waited_seconds <= 5.0
    assert sum(sleeps) == pytest.approx(5.0, abs=0.1)


@pytest.mark.asyncio
async def test_wait_aborts_on_client_disconnect():
    """AC3: a disconnected client stops waiting immediately."""
    cfg = _free_provider_cfg(rpm=1)
    config = _budget_config()

    assert (
        await acquire_provider_budget(
            cfg, config, session_id="gone", request=_DummyRequest()
        )
    ).allowed
    record_provider_affinity("gone", FREE_ENTRY)

    started = time.time()
    decision = await acquire_provider_budget(
        cfg, config, session_id="gone", request=_DisconnectedRequest()
    )
    elapsed = time.time() - started

    assert not decision.allowed
    assert decision.reason == "rate_limited_disconnect"
    assert elapsed < 5.0, "disconnect must abort the wait promptly"


@pytest.mark.asyncio
async def test_waiter_depth_cap_denies_extra_waiters(monkeypatch, clock):
    """AC3: concurrent waiters are capped so they cannot pile up."""
    import asyncio

    cfg = _free_provider_cfg(rpm=1)
    config = _budget_config(provider_rate_limit_wait_max_waiters=1)

    assert (await acquire_provider_budget(cfg, config, session_id="first")).allowed
    record_provider_affinity("first", FREE_ENTRY)
    record_provider_affinity("second", FREE_ENTRY)
    record_provider_affinity("third", FREE_ENTRY)

    started = asyncio.Event()
    release = asyncio.Event()

    async def _blocking_sleep(request, seconds):
        started.set()
        await release.wait()
        return True  # simulate disconnect to exit the waiter cleanly

    monkeypatch.setattr(provider_budget, "_sleep_with_disconnect", _blocking_sleep)

    waiter = asyncio.create_task(
        acquire_provider_budget(cfg, config, session_id="second")
    )
    await started.wait()
    try:
        decision = await acquire_provider_budget(cfg, config, session_id="third")
        assert not decision.allowed
        assert decision.reason == "rate_limited_waiters"
    finally:
        release.set()
        await waiter


# ---------------------------------------------------------------------------
# AC4 — affinity is recorded and expires
# ---------------------------------------------------------------------------


def test_affinity_expires_after_ttl(clock):
    """AC4: affinity is TTL-scoped."""
    config = _budget_config(provider_affinity_ttl_seconds=60)
    record_provider_affinity("s1", FREE_ENTRY, config=config)

    assert provider_affinity_active("s1", FREE_ENTRY, config=config)
    clock["now"] += 59.0
    assert provider_affinity_active("s1", FREE_ENTRY, config=config)
    clock["now"] += 2.0
    assert not provider_affinity_active("s1", FREE_ENTRY, config=config)


def test_affinity_is_scoped_to_session_and_entry():
    """AC4: affinity never leaks across sessions or to a sibling entry."""
    record_provider_affinity("s1", FREE_ENTRY)

    assert provider_affinity_active("s1", FREE_ENTRY)
    assert not provider_affinity_active("s2", FREE_ENTRY)
    assert not provider_affinity_active("s1", PAID_ENTRY)


# ---------------------------------------------------------------------------
# AC1/AC2 — integration through the remote fallback chain
# ---------------------------------------------------------------------------


def _chain_config() -> dict:
    return {
        "providers": [
            {
                "name": FREE_ENTRY,
                "type": "remote",
                "endpoint": "https://free.example.com/v1",
                "rate_limit_rpm": 1,
            },
            {
                "name": PAID_ENTRY,
                "type": "remote",
                "endpoint": "https://paid.example.com/v1",
            },
        ],
    }


@pytest.mark.asyncio
async def test_remote_fallback_skips_gated_entry_for_new_session(caplog):
    """AC1+AC2: the second (different) session never reaches the free entry."""
    call_order: list[str] = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        call_order.append(provider_cfg["name"])
        return _error_response(429)

    model_config = _chain_config()
    config = {"provider_cooldown_seconds": 0, "provider_rate_limit_wait_max_seconds": 60}

    with patch("proxy.provider._get_proxy_to_remote", return_value=_mock_proxy_to_remote):
        await provider.proxy_with_remote_fallback(
            _DummyRequest(session_id="session-a"),
            "v1/chat/completions",
            model_config,
            config,
        )
        first_calls = list(call_order)
        call_order.clear()

        await provider.proxy_with_remote_fallback(
            _DummyRequest(session_id="session-b"),
            "v1/chat/completions",
            model_config,
            config,
        )

    assert first_calls[0] == FREE_ENTRY
    assert call_order == [PAID_ENTRY], (
        f"new session must skip the over-budget free entry, got {call_order}"
    )


@pytest.mark.asyncio
async def test_remote_fallback_uses_gated_entry_when_budget_available():
    """AC1: with spare capacity the free entry is still the first tier."""
    call_order: list[str] = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        call_order.append(provider_cfg["name"])
        return _ok_json_response()

    with patch("proxy.provider._get_proxy_to_remote", return_value=_mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(session_id="session-a"),
            "v1/chat/completions",
            _chain_config(),
            {"provider_cooldown_seconds": 0},
        )

    assert result.status_code == 200
    assert call_order == [FREE_ENTRY]


@pytest.mark.asyncio
async def test_remote_fallback_preserves_behaviour_without_budget():
    """AC6: an entry without rate_limit_rpm is never gated."""
    call_order: list[str] = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        call_order.append(provider_cfg["name"])
        return _ok_json_response()

    model_config = {"providers": [
        {"name": FREE_ENTRY, "type": "remote", "endpoint": "https://free.example.com/v1"},
    ]}

    with patch("proxy.provider._get_proxy_to_remote", return_value=_mock_proxy_to_remote):
        for _ in range(10):
            result = await provider.proxy_with_remote_fallback(
                _DummyRequest(session_id="session-a"),
                "v1/chat/completions",
                model_config,
                {"provider_cooldown_seconds": 0},
            )
            assert result.status_code == 200

    assert call_order == [FREE_ENTRY] * 10


@pytest.mark.asyncio
async def test_affinity_recorded_only_after_successful_dispatch():
    """AC4: a 2xx dispatch records affinity; a 429 does not."""
    cfg = _free_provider_cfg(rpm=5)

    provider._maybe_record_provider_affinity(cfg, "s-ok", _ok_json_response())
    provider._maybe_record_provider_affinity(cfg, "s-fail", _error_response(429))

    assert provider_affinity_active("s-ok", FREE_ENTRY)
    assert not provider_affinity_active("s-fail", FREE_ENTRY)


@pytest.mark.asyncio
async def test_combined_chain_skips_gated_remote_entry_for_new_session():
    """AC2 through the combined local+remote chain (second entry point)."""
    call_order: list[str] = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        call_order.append(provider_cfg["name"])
        return _ok_json_response()

    async def _mock_proxy_to_local(_req, _path):
        call_order.append("local")
        return Response(
            content=json.dumps({
                "error": {
                    "type": "server_busy",
                    "code": "no_slots_available",
                    "reason": "local_lease_active",
                },
                "total_slots": 1,
                "available_slots": 0,
            }),
            status_code=503,
            media_type="application/json",
        )

    model_config = {
        "providers": [
            {"name": "local", "type": "local", "llama_model": "Qwen3"},
            {
                "name": FREE_ENTRY,
                "type": "remote",
                "endpoint": "https://free.example.com/v1",
                "rate_limit_rpm": 1,
            },
            {
                "name": PAID_ENTRY,
                "type": "remote",
                "endpoint": "https://paid.example.com/v1",
            },
        ],
    }
    config = {
        "provider_cooldown_seconds": 0,
        "local_large_context_cold_cache_threshold": 0,
        "local_large_context_warm_cache_threshold": 0,
        "session_slot_pool_size": 4,
        "local_slot_retry_attempts": 0,
        "local_slot_retry_delay_seconds": 0,
    }

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote), patch(
        "proxy.router.proxy_to_local", _mock_proxy_to_local
    ):
        for session in ("session-a", "session-b"):
            call_order.clear()
            await provider.proxy_with_fallback(
                _DummyRequest(session_id=session),
                "v1/chat/completions",
                model_config,
                config,
            )
            if session == "session-a":
                assert FREE_ENTRY in call_order
            else:
                assert FREE_ENTRY not in call_order, (
                    f"new session must skip the over-budget free entry, got {call_order}"
                )
                assert PAID_ENTRY in call_order


# ---------------------------------------------------------------------------
# AC5 — remaining capacity is visible in provider diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_format_budget_status_reports_remaining_capacity():
    """Unbudgeted entries show nothing; gated entries show remaining/limit."""
    ungated = {"name": FREE_ENTRY, "type": "remote", "endpoint": "https://x/v1"}
    assert provider.format_budget_status(ungated) == ""

    cfg = _free_provider_cfg(rpm=5)
    assert provider.format_budget_status(cfg) == "budget 5/5 per 60s"

    for _ in range(2):
        await acquire_provider_budget(cfg, _budget_config(), session_id="s1")

    assert provider.format_budget_status(cfg) == "budget 3/5 per 60s"


def test_home_page_shows_free_tier_budget():
    """The provider diagnostics table renders the remaining free-tier budget."""
    from proxy.ui import _build_home_model_rows

    get_rate_limiter().reset(FREE_ENTRY)
    get_rate_limiter()._windows[FREE_ENTRY].extend([time.monotonic()] * 3)

    class _Srv:
        config = {
            "models": {
                "code": {
                    "providers": [
                        {
                            "name": FREE_ENTRY,
                            "type": "remote",
                            "endpoint": "https://free.example.com/v1",
                            "rate_limit_rpm": 10,
                        }
                    ]
                }
            }
        }

    html = _build_home_model_rows(_Srv())
    assert "budget 7/10 per 60s" in html


# ---------------------------------------------------------------------------
# LP-0MVDUN5CY009VD20 — reserve for in-progress sessions + 15s wait cap
# ---------------------------------------------------------------------------


def _reserve_config(**server_overrides) -> dict:
    server = {
        "provider_rate_limit_wait_max_seconds": 15,
        "provider_affinity_ttl_seconds": 900,
        "provider_rate_limit_reserve_ratio": 0.3,
    }
    server.update(server_overrides)
    return {"server": server}


@pytest.mark.asyncio
async def test_reserve_limits_new_sessions_but_not_in_progress_sessions():
    """AC1: new sessions stop at rpm-reserve; an affinity session gets the rest."""
    cfg = _free_provider_cfg(rpm=10)
    config = _reserve_config()

    for i in range(7):
        decision = await acquire_provider_budget(cfg, config, session_id=f"new-{i}")
        assert decision.allowed, f"new session {i} should fit inside the unreserved window"

    blocked = await acquire_provider_budget(cfg, config, session_id="new-blocked")
    assert not blocked.allowed, "the 8th new-session dispatch must be refused (3 slots reserved)"
    assert blocked.reason == "rate_limited"
    assert blocked.effective_limit == 7
    assert blocked.reserve_slots == 3

    record_provider_affinity("veteran", FREE_ENTRY, config=config)
    for i in range(3):
        decision = await acquire_provider_budget(cfg, config, session_id="veteran")
        assert decision.allowed, f"in-progress session should use reserved slot {i + 1}"

    exhausted = await acquire_provider_budget(cfg, config, session_id="veteran")
    assert not exhausted.allowed
    assert exhausted.reason == "rate_limited_after_wait"


@pytest.mark.asyncio
async def test_zero_reserve_ratio_preserves_full_window_for_new_sessions():
    """AC2: ratio 0 disables the reserve (previous behaviour)."""
    cfg = _free_provider_cfg(rpm=4)
    config = _reserve_config(provider_rate_limit_reserve_ratio=0)

    for i in range(4):
        assert (await acquire_provider_budget(cfg, config, session_id=f"s-{i}")).allowed
    assert not (await acquire_provider_budget(cfg, config, session_id="s-4")).allowed


@pytest.mark.asyncio
async def test_reserve_is_clamped_for_small_and_saturated_budgets():
    """AC2: rpm=1 reserves nothing; ratio 1.0 still leaves one slot for new sessions."""
    one_rpm = _free_provider_cfg(rpm=1)
    config = _reserve_config()
    assert (await acquire_provider_budget(one_rpm, config, session_id="first")).allowed
    denied = await acquire_provider_budget(one_rpm, config, session_id="second")
    assert not denied.allowed and denied.reserve_slots == 0

    get_rate_limiter().reset(FREE_ENTRY)
    saturating = _free_provider_cfg(rpm=4)
    config = _reserve_config(provider_rate_limit_reserve_ratio=1.0)
    assert (await acquire_provider_budget(saturating, config, session_id="only")).allowed
    denied = await acquire_provider_budget(saturating, config, session_id="next")
    assert not denied.allowed
    assert denied.reserve_slots == 3
    assert denied.effective_limit == 1


@pytest.mark.asyncio
async def test_wait_cap_defaults_to_15_seconds(clock, sleeps):
    """AC3: the default wait cap is 15s and no wait exceeds it."""
    assert provider_budget.wait_max_seconds({}) == 15.0

    cfg = _free_provider_cfg(rpm=1)
    config = {"server": {"provider_affinity_ttl_seconds": 900}}

    assert (await acquire_provider_budget(cfg, config, session_id="busy")).allowed
    record_provider_affinity("busy", FREE_ENTRY, config=config)

    # The single slot is held for the full 60s window, so a 15s cap cannot
    # succeed: the request must give up at the cap and pass through.
    decision = await acquire_provider_budget(cfg, config, session_id="busy")
    assert not decision.allowed
    assert decision.reason == "rate_limited_after_wait"
    assert decision.waited_seconds == pytest.approx(15.0)
    assert sleeps and all(w <= 15.0 for w in sleeps)


@pytest.mark.asyncio
async def test_in_progress_session_waits_and_wins_within_the_15s_cap(clock, sleeps):
    """AC3: a wait that can succeed inside the cap still dispatches on the tier."""
    cfg = _free_provider_cfg(rpm=1)
    config = {"server": {"provider_affinity_ttl_seconds": 900}}

    # Seed the window so the single slot frees 5s from now (55s already elapsed).
    get_rate_limiter()._windows[FREE_ENTRY].append(clock["now"] - 55.0)
    record_provider_affinity("busy", FREE_ENTRY, config=config)

    decision = await acquire_provider_budget(cfg, config, session_id="busy")

    assert decision.allowed
    assert decision.reason == "allowed_after_wait"
    assert decision.waited_seconds == pytest.approx(5.0, abs=0.2)
    assert sleeps and sleeps[0] <= 15.0


@pytest.mark.asyncio
async def test_in_progress_session_passes_through_when_cap_expires(clock, sleeps):
    """AC3: once the (short) cap expires the in-progress request passes through."""
    cfg = _free_provider_cfg(rpm=1)
    config = {"server": {"provider_affinity_ttl_seconds": 900,
                         "provider_rate_limit_wait_max_seconds": 5}}

    assert (await acquire_provider_budget(cfg, config, session_id="busy")).allowed
    record_provider_affinity("busy", FREE_ENTRY, config=config)

    decision = await acquire_provider_budget(cfg, config, session_id="busy")
    assert not decision.allowed
    assert decision.reason == "rate_limited_after_wait"
    assert decision.waited_seconds <= 5.0


@pytest.mark.asyncio
async def test_new_session_never_waits_even_with_a_reserve():
    """AC4: the reserve must not introduce waiting for new sessions."""
    sleeps: list[float] = []

    async def _record_sleep(request, seconds):
        sleeps.append(seconds)
        return False

    cfg = _free_provider_cfg(rpm=10)
    config = _reserve_config()

    with patch.object(provider_budget, "_sleep_with_disconnect", _record_sleep):
        for i in range(7):
            assert (await acquire_provider_budget(cfg, config, session_id=f"n-{i}")).allowed
        denied = await acquire_provider_budget(cfg, config, session_id="n-over")

    assert not denied.allowed
    assert sleeps == []


@pytest.mark.asyncio
async def test_reserve_recorded_in_attempt_diagnostics(caplog):
    """AC5: the skip log names the effective cap and the reserve."""
    cfg = _free_provider_cfg(rpm=10)
    config = _reserve_config()

    with caplog.at_level("WARNING", logger="llama-proxy.provider_budget"):
        for i in range(7):
            await acquire_provider_budget(cfg, config, session_id=f"d-{i}")
        await acquire_provider_budget(cfg, config, session_id="d-over")

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "limit=7" in text
    assert "reserve=3" in text


@pytest.mark.asyncio
async def test_remote_fallback_keeps_slots_free_for_in_progress_sessions():
    """AC1 end-to-end: the 8th new session reaches the paid tier, not the free one."""
    call_order: list[str] = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        call_order.append(provider_cfg["name"])
        return _ok_json_response()

    model_config = {
        "providers": [
            {
                "name": FREE_ENTRY,
                "type": "remote",
                "endpoint": "https://free.example.com/v1",
                "rate_limit_rpm": 10,
            },
            {
                "name": PAID_ENTRY,
                "type": "remote",
                "endpoint": "https://paid.example.com/v1",
            },
        ]
    }
    config = {
        "provider_cooldown_seconds": 0,
        "provider_rate_limit_reserve_ratio": 0.3,
        "provider_rate_limit_wait_max_seconds": 15,
    }

    with patch("proxy.provider._get_proxy_to_remote", return_value=_mock_proxy_to_remote):
        for i in range(8):
            call_order.clear()
            await provider.proxy_with_remote_fallback(
                _DummyRequest(session_id=f"agent-{i}"),
                "v1/chat/completions",
                model_config,
                config,
            )
            if i < 7:
                assert call_order == [FREE_ENTRY], f"session {i} should use the free tier"
            else:
                assert call_order == [PAID_ENTRY], (
                    f"session {i} must skip to the paid tier, got {call_order}"
                )


# ---------------------------------------------------------------------------
# LP-0MVDVK5TM004TQQV — model / entry level ``budget`` blocks
# ---------------------------------------------------------------------------
#
# Precedence under test: provider entry > ``models.<name>.budget`` >
# ``server.*`` > built-in default. Every knob is optional at every layer, so a
# config without any ``budget`` block must behave exactly as before.

#: knob -> (server config key, server value, model value, entry value, default)
_BUDGET_KNOBS = {
    "reserve_ratio": ("provider_rate_limit_reserve_ratio", 0.5, 0.2, 0.1, 0.3),
    "wait_max_seconds": ("provider_rate_limit_wait_max_seconds", 12.0, 7.0, 2.0, 15.0),
    "max_waiters": ("provider_rate_limit_wait_max_waiters", 6, 3, 1, 8),
    "affinity_ttl_seconds": ("provider_affinity_ttl_seconds", 600.0, 300.0, 60.0, 900.0),
}


def _model_with_budget(
    budget: dict | None = None,
    *,
    rpm: int = 10,
    entry_budget: dict | None = None,
    name: str | None = None,
) -> dict:
    """A ``models.<name>`` block carrying a ``budget`` block and one entry."""
    entry = _free_provider_cfg(rpm=rpm)
    if entry_budget is not None:
        entry["budget"] = entry_budget
    model: dict = {"providers": [entry]}
    if budget is not None:
        model["budget"] = budget
    if name is not None:
        model["name"] = name
    return model


@pytest.mark.parametrize("knob", sorted(_BUDGET_KNOBS))
def test_budget_precedence_entry_then_model_then_server_then_default(knob):
    """AC2: every layer of the entry > model > server > default chain."""
    server_key, server_value, model_value, entry_value, default = _BUDGET_KNOBS[knob]
    entry_only = _free_provider_cfg(rpm=10)

    # 1. nothing configured anywhere -> built-in default
    settings = provider_budget.resolve_budget_settings(entry_only, {}, None)
    assert getattr(settings, knob) == default
    assert settings.sources[knob] == "default"

    # 2. server only
    config = {"server": {server_key: server_value}}
    settings = provider_budget.resolve_budget_settings(entry_only, config, None)
    assert getattr(settings, knob) == server_value
    assert settings.sources[knob] == "server"

    # 3. server + model -> model wins
    model = _model_with_budget({knob: model_value}, rpm=10)
    settings = provider_budget.resolve_budget_settings(model["providers"][0], config, model)
    assert getattr(settings, knob) == model_value
    assert settings.sources[knob] == "model"

    # 4. entry + model + server -> entry wins
    overridden = _model_with_budget({knob: model_value}, rpm=10, entry_budget={knob: entry_value})
    settings = provider_budget.resolve_budget_settings(overridden["providers"][0], config, overridden)
    assert getattr(settings, knob) == entry_value
    assert settings.sources[knob] == "entry"

    # 5. entry only
    settings = provider_budget.resolve_budget_settings(overridden["providers"][0], {}, overridden)
    assert getattr(settings, knob) == entry_value
    assert settings.sources[knob] == "entry"

    # 6. model only (no server value configured)
    settings = provider_budget.resolve_budget_settings(model["providers"][0], {}, model)
    assert getattr(settings, knob) == model_value
    assert settings.sources[knob] == "model"


@pytest.mark.asyncio
async def test_model_budget_reserve_ratio_is_honoured():
    """AC1: a model ``budget.reserve_ratio`` beats the server-level ratio."""
    model = _model_with_budget({"reserve_ratio": 0.5}, rpm=10)
    entry = model["providers"][0]
    config = _reserve_config(provider_rate_limit_reserve_ratio=0.3)

    for i in range(5):
        decision = await acquire_provider_budget(
            entry, config, session_id=f"n-{i}", model_config=model
        )
        assert decision.allowed, f"new session {i} should fit the 5 unreserved slots"

    denied = await acquire_provider_budget(entry, config, session_id="n-over", model_config=model)
    assert not denied.allowed
    assert denied.reserve_slots == 5, "model ratio 0.5 of 10, not the server's 0.3"
    assert denied.effective_limit == 5


@pytest.mark.asyncio
async def test_model_budget_wait_max_seconds_is_honoured(clock, sleeps):
    """AC1: a model ``budget.wait_max_seconds`` caps the in-progress wait."""
    model = _model_with_budget({"wait_max_seconds": 4}, rpm=1)
    entry = model["providers"][0]
    config = _reserve_config(provider_rate_limit_wait_max_seconds=15)

    assert (
        await acquire_provider_budget(entry, config, session_id="busy", model_config=model)
    ).allowed
    provider._maybe_record_provider_affinity(
        entry, "busy", _ok_json_response(), config, model_config=model
    )

    decision = await acquire_provider_budget(entry, config, session_id="busy", model_config=model)

    assert not decision.allowed
    assert decision.reason == "rate_limited_after_wait"
    assert decision.waited_seconds == pytest.approx(4.0)
    assert sum(sleeps) == pytest.approx(4.0, abs=0.1)


@pytest.mark.asyncio
async def test_model_budget_max_waiters_is_honoured():
    """AC1: a model ``budget.max_waiters`` of 0 denies every wait."""
    model = _model_with_budget({"max_waiters": 0}, rpm=1)
    entry = model["providers"][0]
    config = _reserve_config(provider_rate_limit_wait_max_waiters=8)

    assert (
        await acquire_provider_budget(entry, config, session_id="busy", model_config=model)
    ).allowed
    provider._maybe_record_provider_affinity(
        entry, "busy", _ok_json_response(), config, model_config=model
    )

    decision = await acquire_provider_budget(entry, config, session_id="busy", model_config=model)

    assert not decision.allowed
    assert decision.reason == "rate_limited_waiters"


@pytest.mark.asyncio
async def test_model_budget_affinity_ttl_is_honoured(clock, sleeps):
    """AC1: affinity recorded under a model TTL expires on that TTL, not the server's."""
    plan = _model_with_budget({"affinity_ttl_seconds": 60}, rpm=1, name="plan")
    code = _model_with_budget({"affinity_ttl_seconds": 900}, rpm=1, name="code")
    entry = plan["providers"][0]
    config = _reserve_config(provider_affinity_ttl_seconds=900)

    assert (await acquire_provider_budget(entry, config, session_id="plan-1", model_config=plan)).allowed
    provider._maybe_record_provider_affinity(
        entry, "plan-1", _ok_json_response(), config, model_config=plan
    )
    provider._maybe_record_provider_affinity(
        entry, "code-1", _ok_json_response(), config, model_config=code
    )

    # Past the 60 s plan TTL but well inside the 900 s code TTL.
    clock["now"] += 61.0
    assert (
        await acquire_provider_budget(entry, config, session_id="filler", model_config=None)
    ).allowed, "window rolled over, make the entry full again"

    # plan-1 is a new session again -> passes through immediately, no wait.
    sleeps.clear()
    plan_decision = await acquire_provider_budget(
        entry, config, session_id="plan-1", model_config=plan
    )
    assert not plan_decision.allowed
    assert plan_decision.reason == "rate_limited"
    assert sleeps == [], "the 60 s model TTL must have expired plan-1's affinity"

    # code-1 is still in progress -> waits for capacity.
    code_decision = await acquire_provider_budget(
        entry, config, session_id="code-1", model_config=code
    )
    assert not code_decision.allowed
    assert code_decision.reason == "rate_limited_after_wait"
    assert code_decision.waited_seconds > 0.0


@pytest.mark.asyncio
async def test_entry_budget_overrides_model_budget_in_the_gate():
    """AC1: an entry ``budget`` block wins over the model block."""
    model = _model_with_budget(
        {"reserve_ratio": 0.5}, rpm=10, entry_budget={"reserve_ratio": 0.0}
    )
    entry = model["providers"][0]
    config = _reserve_config(provider_rate_limit_reserve_ratio=0.3)

    for i in range(10):
        assert (
            await acquire_provider_budget(entry, config, session_id=f"s-{i}", model_config=model)
        ).allowed, f"the entry override disables the reserve, so dispatch {i} must fit"

    denied = await acquire_provider_budget(entry, config, session_id="s-over", model_config=model)
    assert not denied.allowed
    assert denied.reserve_slots == 0
    assert denied.effective_limit == 10


@pytest.mark.asyncio
async def test_no_budget_blocks_keep_todays_decisions_and_log_lines(caplog):
    """AC3: without any ``budget`` block, decisions and logs are byte-identical."""
    cfg = _free_provider_cfg(rpm=10)
    config = _reserve_config()  # 0.3 / 15 / 8 / 900 — the shipped server block

    for i in range(7):
        decision = await acquire_provider_budget(cfg, config, session_id=f"new-{i}")
        assert decision.allowed
        assert (decision.reserve_slots, decision.effective_limit, decision.reason) == (
            3,
            7,
            "allowed",
        )
        assert decision.source_label == "", "no model/entry block => no source annotation"

    with caplog.at_level("WARNING", logger="llama-proxy.provider_budget"):
        denied = await acquire_provider_budget(
            cfg, config, session_id="abcdefgh-trailing"
        )

    assert (denied.reserve_slots, denied.effective_limit, denied.reason) == (
        3,
        7,
        "rate_limited",
    )
    skip_lines = [
        r.getMessage() for r in caplog.records if "Provider budget exhausted" in r.getMessage()
    ]
    assert skip_lines[-1] == (
        "Provider budget exhausted: skipping provider=test-free-tier limit=7 reserve=3 "
        "remaining=3 reason=rate_limited session_affinity=false session=abcdefgh"
    ), "the shipped-config log line must not change"


@pytest.mark.parametrize("knob", sorted(_BUDGET_KNOBS))
def test_malformed_budget_value_falls_back_to_the_next_layer_with_warning(knob, caplog):
    """AC4: a malformed model value warns and defers to the server value."""
    server_key, server_value, _model_value, _entry_value, _default = _BUDGET_KNOBS[knob]
    model = _model_with_budget({knob: "not-a-number"}, rpm=10)
    config = {"server": {server_key: server_value}}

    with caplog.at_level("WARNING", logger="llama-proxy.provider_budget"):
        settings = provider_budget.resolve_budget_settings(model["providers"][0], config, model)

    assert getattr(settings, knob) == server_value
    assert settings.sources[knob] == "server"
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert f"key={knob}" in text
    assert "layer=model" in text


@pytest.mark.parametrize(
    "knob,raw,expected",
    [
        ("reserve_ratio", 7.5, 1.0),
        ("wait_max_seconds", -3, 0.0),
        ("max_waiters", -2, 0),
        ("affinity_ttl_seconds", -5, 0.0),
    ],
)
def test_out_of_range_budget_value_is_clamped_with_warning(knob, raw, expected, caplog):
    """AC4: out-of-range values clamp into range (as server knobs always have)."""
    model = _model_with_budget({knob: raw}, rpm=10)

    with caplog.at_level("WARNING", logger="llama-proxy.provider_budget"):
        settings = provider_budget.resolve_budget_settings(model["providers"][0], {}, model)

    assert getattr(settings, knob) == expected
    assert settings.sources[knob] == "model"
    assert "clamped" in "\n".join(r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_malformed_budget_block_never_blocks_a_request(caplog):
    """AC4: garbage in the block is ignored, the request still dispatches."""
    model = _model_with_budget(
        {
            "reserve_ratio": "high",
            "wait_max_seconds": [],
            "max_waiters": {},
            "affinity_ttl_seconds": None,
        },
        rpm=10,
    )
    entry = model["providers"][0]

    with caplog.at_level("WARNING", logger="llama-proxy.provider_budget"):
        decision = await acquire_provider_budget(entry, {}, session_id="s-1", model_config=model)

    assert decision.allowed
    assert decision.reason == "allowed"


def test_non_mapping_budget_block_is_ignored_with_warning(caplog):
    """AC4: ``budget: fast`` (not a mapping) is ignored, not fatal."""
    model = {"providers": [_free_provider_cfg(rpm=10)], "budget": "fast"}

    with caplog.at_level("WARNING", logger="llama-proxy.provider_budget"):
        settings = provider_budget.resolve_budget_settings(model["providers"][0], {}, model)

    assert settings.reserve_ratio == 0.3
    assert settings.sources["reserve_ratio"] == "default"
    assert "layer=model" in "\n".join(r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_budget_window_stays_shared_across_models():
    """AC5: the bucket is per provider entry — two models share one window."""
    plan = _model_with_budget({"reserve_ratio": 0.0}, rpm=10, name="plan")
    code = _model_with_budget({"reserve_ratio": 0.0}, rpm=10, name="code")
    config = _reserve_config()

    for i in range(5):
        assert (
            await acquire_provider_budget(
                plan["providers"][0], config, session_id=f"plan-{i}", model_config=plan
            )
        ).allowed
    for i in range(5):
        assert (
            await acquire_provider_budget(
                code["providers"][0], config, session_id=f"code-{i}", model_config=code
            )
        ).allowed

    denied = await acquire_provider_budget(
        code["providers"][0], config, session_id="code-over", model_config=code
    )
    assert not denied.allowed, "a second model must not get a second 10/min window"
    assert get_rate_limiter().remaining(FREE_ENTRY, max_rpm=10) == 0


@pytest.mark.asyncio
async def test_source_layer_is_visible_in_the_skip_log(caplog):
    """AC6: the skip log names the layer that supplied the effective knob."""
    model = _model_with_budget({"reserve_ratio": 0.5}, rpm=10)
    entry = model["providers"][0]
    config = _reserve_config()

    with caplog.at_level("WARNING", logger="llama-proxy.provider_budget"):
        for i in range(5):
            await acquire_provider_budget(entry, config, session_id=f"n-{i}", model_config=model)
        denied = await acquire_provider_budget(entry, config, session_id="n-over", model_config=model)

    assert denied.source_label == "reserve_ratio@model"
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "budget_source=reserve_ratio@model" in text


def test_budget_diagnostics_are_empty_without_a_model_block():
    """AC6: unchanged behaviour when no model/entry layer contributed a value."""
    decision = provider_budget.BudgetDecision(
        allowed=False, reason="rate_limited", reserve_slots=3, effective_limit=7
    )
    assert provider._budget_diagnostics(decision) == {}


@pytest.mark.asyncio
async def test_attempt_diagnostics_carry_the_budget_source_layer():
    """AC6: the exhaustion payload's attempt entry names the source layer."""

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        return _ok_json_response()

    model = _model_with_budget({"reserve_ratio": 0.0}, rpm=1)
    config = {"provider_cooldown_seconds": 0}

    with patch("proxy.provider._get_proxy_to_remote", return_value=_mock_proxy_to_remote):
        first = await provider.proxy_with_remote_fallback(
            _DummyRequest(session_id="first"), "v1/chat/completions", model, config
        )
        blocked = await provider.proxy_with_remote_fallback(
            _DummyRequest(session_id="second"), "v1/chat/completions", model, config
        )

    assert first.status_code == 200
    payload = json.loads(blocked.body)
    rate_limited = [
        attempt for attempt in payload["diagnostics"] if attempt.get("status") == "rate_limited"
    ]
    assert rate_limited, f"expected a rate_limited attempt, got {payload['diagnostics']}"
    assert rate_limited[0]["budget_source"] == "reserve_ratio@model"


@pytest.mark.asyncio
async def test_both_fallback_entry_points_pass_the_model_config_to_the_gate(monkeypatch):
    """AC7: both real entry points hand the request's model config to the gate."""
    seen: list[dict | None] = []
    real_acquire = provider.acquire_provider_budget

    async def _spy(provider_cfg, config, session_id, request=None, model_config=None):
        seen.append(model_config)
        return await real_acquire(
            provider_cfg, config, session_id, request, model_config=model_config
        )

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        return _ok_json_response()

    monkeypatch.setattr(provider, "acquire_provider_budget", _spy)

    model = _model_with_budget({"wait_max_seconds": 3}, rpm=5)
    with patch("proxy.provider._get_proxy_to_remote", return_value=_mock_proxy_to_remote):
        await provider.proxy_with_remote_fallback(
            _DummyRequest(session_id="remote-1"),
            "v1/chat/completions",
            model,
            {"provider_cooldown_seconds": 0},
        )
    assert seen == [model], "the remote cycle must pass the model config"

    seen.clear()
    with patch("proxy.provider._get_proxy_to_remote", return_value=_mock_proxy_to_remote):
        await provider.proxy_with_fallback(
            _DummyRequest(session_id="combined-1"),
            "v1/chat/completions",
            model,
            {"provider_cooldown_seconds": 0},
        )
    assert seen == [model], "the combined cycle must pass the model config"
