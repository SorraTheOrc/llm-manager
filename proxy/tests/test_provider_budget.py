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
