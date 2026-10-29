"""Post-deadline prefill-thrash cooldown (LP-0MVKUKFHH005UDWL).

Root cause 3 of 3 (LP-0MVKN9IRP003XJH6): after ``prefill_deadline_exceeded``
cancels a local dispatch, the client/agent re-dispatches the identical
prompt and the whole prefill burns again — the 2026-10-23 incident logged 41
local POSTs (35 inside 02:00-03:00) re-prefilling near-duplicates of the
same 28k-124k-token prompts (93,038 x2, 93,006 x3, …) with zero tokens
delivered.

This item adds a per-session cooldown:

1. On ``prefill_deadline_exceeded`` the session records a cooldown window
   (``server.local_prefill_thrash_cooldown_seconds``, default 600; 0
   disables).
2. During the window that session's local routing skips
   (``routing_skip_local reason=prefill_thrash_cooldown``) and reaches the
   next provider; other sessions are unaffected.
3. The window clears on expiry and on a successful local first byte.

Acceptance criteria covered:

- AC1: after a deadline event, the next request from that session skips
  local with ``reason=prefill_thrash_cooldown`` and reaches the next
  provider.
- AC2: a different session is still admitted locally while the first is
  cooling down (cross-session isolation).
- AC3: the cooldown clears on expiry and on a successful local first byte;
  ``0`` disables it.
- AC4: the skip is logged once per request with the remaining cooldown.

Clock injection mirrors ``test_prefill_thrash_wedge.py``: ``router_helpers.time``
is replaced by a shim so every cooldown timestamp reads an injected monotonic
clock — no real sleeps and deterministic expiry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Response

from proxy import contention_queue
from proxy import router_helpers as helpers

DEFAULT_COOLDOWN_SECONDS = 600.0

SESSION_A = "sess-cooling-a"
SESSION_B = "sess-other-b"


# ---------------------------------------------------------------------------
# Injected clock (mirrors test_prefill_thrash_wedge.py)
# ---------------------------------------------------------------------------


class _InjectedClock:
    """Monotonic clock that only advances when the test drives it."""

    def __init__(self) -> None:
        self._now = 0.0

    @property
    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class _TimeShim:
    """Replace ``router_helpers.time`` so only ``monotonic`` is injected."""

    def __init__(self, clock: _InjectedClock) -> None:
        self._clock = clock

    def monotonic(self) -> float:
        return self._clock.now

    def __getattr__(self, name: str):
        return getattr(time, name)


@pytest.fixture
def clock() -> _InjectedClock:
    return _InjectedClock()


@pytest.fixture(autouse=True)
def _reset_cooldowns(monkeypatch, clock: _InjectedClock):
    """Isolate cooldown state and the injected clock around every test."""
    monkeypatch.setattr(helpers, "time", _TimeShim(clock))
    helpers._reset_prefill_thrash_cooldowns()
    yield
    helpers._reset_prefill_thrash_cooldowns()


def _server_config(**overrides) -> dict:
    """A server config with the cooldown knob explicitly set."""
    cfg = {
        "server": {
            "local_prefill_thrash_cooldown_seconds": DEFAULT_COOLDOWN_SECONDS,
        }
    }
    cfg["server"].update(overrides)
    return cfg


# ═════════════════════════════════════════════════════════════════════════════
# Cooldown state primitives (router_helpers)
# ═════════════════════════════════════════════════════════════════════════════


def test_record_sets_default_cooldown_window(clock):
    """AC1: recording the cooldown arms a fresh window measured from now."""
    deadline = helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), now=clock.now
    )
    assert deadline == DEFAULT_COOLDOWN_SECONDS
    remaining = helpers._prefill_thrash_cooldown_remaining(
        SESSION_A, _server_config(), now=clock.now
    )
    assert remaining == DEFAULT_COOLDOWN_SECONDS


def test_config_override_and_malformed_fallback(clock):
    """The knob overrides the default; a malformed value falls back to the
    default (fail-safe: the protection stays on)."""
    cfg = _server_config(local_prefill_thrash_cooldown_seconds=30)
    deadline = helpers._record_prefill_thrash_cooldown(SESSION_A, cfg, now=0)
    assert deadline == 30.0

    helpers._reset_prefill_thrash_cooldowns()
    bad = _server_config(local_prefill_thrash_cooldown_seconds="not-a-number")
    assert helpers._get_prefill_thrash_cooldown_seconds(bad) == DEFAULT_COOLDOWN_SECONDS
    helpers._record_prefill_thrash_cooldown(SESSION_A, bad, now=0)
    assert (
        helpers._prefill_thrash_cooldown_remaining(SESSION_A, bad, now=10)
        == DEFAULT_COOLDOWN_SECONDS - 10
    )


def test_cooldown_decays_and_clears_on_expiry(clock):
    """AC3: the remaining time decays with the clock and the window is pruned
    the moment it expires — the session is free again."""
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), now=clock.now
    )
    clock.advance(300)
    assert (
        helpers._prefill_thrash_cooldown_remaining(SESSION_A, _server_config(), now=clock.now)
        == 300
    )
    clock.advance(300)
    assert (
        helpers._prefill_thrash_cooldown_remaining(SESSION_A, _server_config(), now=clock.now)
        == 0
    )
    # Expired windows are pruned, not just reported as 0.
    assert SESSION_A not in helpers._prefill_thrash_cooldowns


def test_zero_disables_the_cooldown(clock):
    """AC3: 0 disables — recording arms nothing and remaining stays 0."""
    cfg = _server_config(local_prefill_thrash_cooldown_seconds=0)
    assert helpers._record_prefill_thrash_cooldown(SESSION_A, cfg, now=0) == 0
    assert helpers._prefill_thrash_cooldown_remaining(SESSION_A, cfg, now=0) == 0
    assert SESSION_A not in helpers._prefill_thrash_cooldowns


def test_live_disable_clears_an_active_window(clock):
    """A live config flip to 0 clears an already-armed window (documented
    rollback: the cooldown no longer suppresses local routing)."""
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), now=clock.now
    )
    assert (
        helpers._prefill_thrash_cooldown_remaining(SESSION_A, _server_config(), now=clock.now)
        > 0
    )
    off = _server_config(local_prefill_thrash_cooldown_seconds=0)
    assert helpers._prefill_thrash_cooldown_remaining(SESSION_A, off, now=clock.now) == 0
    assert SESSION_A not in helpers._prefill_thrash_cooldowns


def test_cross_session_isolated(clock):
    """AC2: a cooldown on one session leaves another session untouched."""
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), now=clock.now
    )
    cfg = _server_config()
    assert helpers._prefill_thrash_cooldown_remaining(SESSION_A, cfg, now=clock.now) > 0
    assert helpers._prefill_thrash_cooldown_remaining(SESSION_B, cfg, now=clock.now) == 0


def test_repeat_deadline_rearms_the_window(clock):
    """A second deadline event restarts the window from now (the session has
    thrashed again, so the cooldown starts fresh)."""
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), now=clock.now
    )
    clock.advance(500)
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), now=clock.now
    )
    assert (
        helpers._prefill_thrash_cooldown_remaining(SESSION_A, _server_config(), now=clock.now)
        == DEFAULT_COOLDOWN_SECONDS
    )


def test_successful_first_byte_clears_the_cooldown(clock):
    """AC3: clearing on a successful local first byte frees the session
    immediately — a healthy dispatch proves prefills work again."""
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), now=clock.now
    )
    assert helpers._clear_prefill_thrash_cooldown(SESSION_A) is True
    assert SESSION_A not in helpers._prefill_thrash_cooldowns
    assert (
        helpers._prefill_thrash_cooldown_remaining(SESSION_A, _server_config(), now=clock.now)
        == 0
    )


# ---------------------------------------------------------------------------
# Recovery-fraction gating of the clear path (LP-0MVQITN6P008DPLB)
#
# Measured 2026-10-26 09:06 -> 2026-10-27 09:06: 89 of 114 prefill-deadline
# evictions were repeats inside 33 (session, prompt) clusters, and the repeats
# violated the configured 600 s window (e.g. 01a178bd cancelled at 15:25:57,
# 15:28:05 and 15:30:20 on the same 46,134-token prompt). The clear-on-any-
# success rule let a small warm dispatch (first byte ~1.5 s) wipe the window
# that the large doomed prompt had just armed.
# ---------------------------------------------------------------------------

ARMED_TOKENS = 46_134


def test_small_success_does_not_clear_a_large_cancel_window(clock):
    """The bug: a 6,000-token success must not release a 46k-token cooldown."""
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), now=clock.now, tokens=ARMED_TOKENS
    )
    assert (
        helpers._clear_prefill_thrash_cooldown(
            SESSION_A, tokens=6_000, config=_server_config()
        )
        is False
    )
    assert (
        helpers._prefill_thrash_cooldown_remaining(
            SESSION_A, _server_config(), now=clock.now
        )
        == DEFAULT_COOLDOWN_SECONDS
    )


def test_comparable_success_clears_the_window(clock):
    """A dispatch of the failing size proves recovery and releases the window."""
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), now=clock.now, tokens=ARMED_TOKENS
    )
    assert (
        helpers._clear_prefill_thrash_cooldown(
            SESSION_A, tokens=ARMED_TOKENS, config=_server_config()
        )
        is True
    )
    assert (
        helpers._prefill_thrash_cooldown_remaining(
            SESSION_A, _server_config(), now=clock.now
        )
        == 0
    )


def test_success_just_below_the_fraction_does_not_clear(clock):
    """Boundary: 89 % of the armed size is still below the 0.9 default."""
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), now=clock.now, tokens=ARMED_TOKENS
    )
    assert (
        helpers._clear_prefill_thrash_cooldown(
            SESSION_A, tokens=int(ARMED_TOKENS * 0.89), config=_server_config()
        )
        is False
    )


def test_success_at_the_fraction_boundary_clears(clock):
    """The comparison is inclusive at the configured fraction."""
    threshold = math.ceil(ARMED_TOKENS * 0.9)
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), now=clock.now, tokens=ARMED_TOKENS
    )
    assert (
        helpers._clear_prefill_thrash_cooldown(
            SESSION_A, tokens=threshold, config=_server_config()
        )
        is True
    )


def test_arming_without_a_size_keeps_clear_on_any_success(clock):
    """Fail-open: an armed window with no recorded size cannot gate, so the
    pre-change behaviour (clear on any success) is preserved."""
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), now=clock.now
    )
    assert (
        helpers._clear_prefill_thrash_cooldown(
            SESSION_A, tokens=1_000, config=_server_config()
        )
        is True
    )


def test_clearing_without_a_size_keeps_todays_behaviour(clock):
    """Fail-open: a caller that cannot supply the size is never worse off."""
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), now=clock.now, tokens=ARMED_TOKENS
    )
    assert helpers._clear_prefill_thrash_cooldown(SESSION_A) is True


def test_zero_recovery_fraction_restores_clear_on_any_success(clock):
    """Rollback: fraction 0 restores the pre-fix clear-on-any-success rule."""
    cfg = _server_config(local_prefill_thrash_recovery_fraction=0)
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, cfg, now=clock.now, tokens=ARMED_TOKENS
    )
    assert (
        helpers._clear_prefill_thrash_cooldown(
            SESSION_A, tokens=100, config=cfg
        )
        is True
    )


def test_recovery_fraction_config_resolution():
    """Default 0.9; a valid override wins; malformed falls back to the default."""
    assert (
        helpers._get_prefill_thrash_recovery_fraction({})
        == helpers.DEFAULT_PREFILL_THRASH_RECOVERY_FRACTION
    )
    assert (
        helpers._get_prefill_thrash_recovery_fraction(
            _server_config(local_prefill_thrash_recovery_fraction=0.5)
        )
        == 0.5
    )
    assert (
        helpers._get_prefill_thrash_recovery_fraction(
            _server_config(local_prefill_thrash_recovery_fraction="nonsense")
        )
        == helpers.DEFAULT_PREFILL_THRASH_RECOVERY_FRACTION
    )
    # Disabled: 0 and negatives both resolve to 0, not to the default.
    assert (
        helpers._get_prefill_thrash_recovery_fraction(
            _server_config(local_prefill_thrash_recovery_fraction=0)
        )
        == 0.0
    )
    assert (
        helpers._get_prefill_thrash_recovery_fraction(
            _server_config(local_prefill_thrash_recovery_fraction=-3)
        )
        == 0.0
    )


def test_recovery_fraction_above_one_is_clamped(clock):
    """A fraction > 1 would make recovery impossible; clamp it to 1.0."""
    assert (
        helpers._get_prefill_thrash_recovery_fraction(
            _server_config(local_prefill_thrash_recovery_fraction=5)
        )
        == 1.0
    )


def test_clear_is_cross_session_isolated(clock):
    """Clearing session B never touches session A's armed window."""
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), now=clock.now, tokens=ARMED_TOKENS
    )
    assert (
        helpers._clear_prefill_thrash_cooldown(
            SESSION_B, tokens=ARMED_TOKENS, config=_server_config()
        )
        is False
    )
    assert (
        helpers._prefill_thrash_cooldown_remaining(
            SESSION_A, _server_config(), now=clock.now
        )
        == DEFAULT_COOLDOWN_SECONDS
    )


def test_reset_clears_recorded_sizes(clock):
    """Test isolation: the reset hook drops the armed sizes too."""
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), now=clock.now, tokens=ARMED_TOKENS
    )
    helpers._reset_prefill_thrash_cooldowns()
    assert helpers._prefill_thrash_cancel_tokens == {}


# ═════════════════════════════════════════════════════════════════════════════
# Routing behaviour (provider fallback chain)
# ═════════════════════════════════════════════════════════════════════════════


class _DummyRequest:
    """Minimal request stub (same shape as test_contention_queue.py)."""

    def __init__(self, body: bytes = b'{"model":"test"}', headers: dict | None = None):
        self._body = body
        self.headers = headers or {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


def _ok_response() -> Response:
    return Response(
        content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
        status_code=200,
        media_type="application/json",
    )


class _MutableConcurrency:
    """Slot availability stand-in: a free slot so local is dispatchable."""

    def __init__(self, active: int = 0, max_: int = 1):
        self.active = active
        self.max = max_

    def __call__(self, config) -> tuple[int, int]:
        return (self.active, self.max)


@pytest.fixture(autouse=True)
def _reset_queue():
    contention_queue.reset()
    yield
    contention_queue.reset()


@pytest.fixture
def mixed_model_config():
    """A model config with both local and remote providers."""
    return {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {
                "name": "remote-fallback",
                "type": "remote",
                "endpoint": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
        ],
        "aliases": ["hybrid*"],
    }


@pytest.fixture
def fallback_harness(monkeypatch, mixed_model_config):
    """Wire the provider fallback chain with in-memory local/remote stubs.

    Returns ``(request_body, config, call_log, models)`` where *call_log*
    records which provider actually served the request.
    """
    import proxy.provider as provider

    concurrency = _MutableConcurrency(active=0, max_=1)
    call_log: list[str] = []

    async def _mock_proxy_to_local(_req, _path):
        call_log.append("local")
        return _ok_response()

    async def _mock_proxy_to_remote(_req, _path, _pc):
        call_log.append("remote")
        return _ok_response()

    monkeypatch.setattr(provider, "_get_local_concurrency_info", concurrency)
    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
    ):

        def run(request, cfg):
            return provider.proxy_with_fallback(
                request, "v1/chat/completions", mixed_model_config, cfg
            )

        run.call_log = call_log
        run.concurrency = concurrency
        yield run
    return None


def _session_request(session_id: str | None) -> _DummyRequest:
    # Header keys must be lowercase: ``_resolve_session_id_header`` reads a
    # plain dict with ``headers.get("x-session-id", …)``.
    headers = {"x-session-id": session_id} if session_id else {}
    body = json.dumps(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "short prompt"}],
        }
    ).encode()
    return _DummyRequest(body=body, headers=headers)


@pytest.mark.asyncio
async def test_cooling_session_skips_local_and_reaches_remote(
    fallback_harness, clock, caplog,
):
    """AC1 + AC4: while the session cools down, local is skipped once with
    ``reason=prefill_thrash_cooldown`` (remaining included) and the request
    reaches the next (remote) provider."""

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), now=clock.now
    )

    resp = await fallback_harness(
        _session_request(SESSION_A), _server_config()
    )
    assert resp.status_code == 200
    assert fallback_harness.call_log == ["remote"], (
        "a cooling session must never be dispatched locally"
    )

    messages = " ".join(r.getMessage() for r in caplog.records)
    # AC4: the skip itself is logged exactly once per request, carrying the
    # remaining cooldown and the ``prefill_thrash_cooldown`` reason.
    assert messages.count("routing_skip_local") == 1
    assert "reason=prefill_thrash_cooldown" in messages
    assert "cooldown_remaining=" in messages


@pytest.mark.asyncio
async def test_cross_session_isolation_at_routing(fallback_harness, clock):
    """AC2: while session A cools down, session B is still admitted locally."""
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), now=clock.now
    )

    resp_b = await fallback_harness(
        _session_request(SESSION_B), _server_config()
    )
    assert resp_b.status_code == 200
    assert fallback_harness.call_log == ["local"], (
        "a different session must be unaffected by A's cooldown"
    )


@pytest.mark.asyncio
async def test_expired_cooldown_routes_local_again(fallback_harness, clock):
    """AC3: after the window expires the same session is dispatched locally
    again (no stale suppression)."""
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), now=clock.now
    )
    clock.advance(DEFAULT_COOLDOWN_SECONDS)

    resp = await fallback_harness(_session_request(SESSION_A), _server_config())
    assert resp.status_code == 200
    assert fallback_harness.call_log == ["local"], (
        "an expired cooldown must not suppress local routing"
    )


@pytest.mark.asyncio
async def test_cleared_cooldown_routes_local_again(fallback_harness, clock):
    """AC3: clearing on a successful local first byte restores local routing
    for the session (router.py clears on first byte)."""
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), now=clock.now
    )
    helpers._clear_prefill_thrash_cooldown(SESSION_A)

    resp = await fallback_harness(_session_request(SESSION_A), _server_config())
    assert resp.status_code == 200
    assert fallback_harness.call_log == ["local"]


@pytest.mark.asyncio
async def test_zero_disables_skip_at_routing(fallback_harness, clock):
    """AC3 (rollback): a 0 knob never arms the cooldown, so local routing is
    unchanged even after a deadline event would have recorded one."""
    cfg_off = _server_config(local_prefill_thrash_cooldown_seconds=0)
    assert helpers._record_prefill_thrash_cooldown(SESSION_A, cfg_off, now=0) == 0

    resp = await fallback_harness(_session_request(SESSION_A), cfg_off)
    assert resp.status_code == 200
    assert fallback_harness.call_log == ["local"]


# ═════════════════════════════════════════════════════════════════════════════
# Router wiring: deadline records the cooldown; first byte clears it
# ═════════════════════════════════════════════════════════════════════════════


def _install_router_harness(monkeypatch, server_cfg: dict):
    """Minimal ``proxy_to_local`` harness (mirrors test_cold_start_admission).

    Returns an object whose ``monitor`` attribute can be replaced so tests can
    force either a deadline timeout or a fast first byte.
    """
    from unittest.mock import AsyncMock

    from proxy import server as srv

    monkeypatch.setattr(srv, "config", server_cfg)
    proc = MagicMock()
    proc.poll.return_value = None
    monkeypatch.setattr(srv, "llama_process", proc)
    monkeypatch.setattr(srv, "backend_ready", True)
    monkeypatch.setattr(srv, "current_model", "Qwen3")
    monkeypatch.setattr(srv, "active_queries", 0)
    monkeypatch.setattr(srv, "active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "local_active_queries", 0)
    monkeypatch.setattr(srv, "local_active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "local_generating_queries", 0)
    monkeypatch.setattr(srv, "local_generating_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "local_generating_sessions", set())
    monkeypatch.setattr(srv, "local_dispatch_records", {})
    monkeypatch.setattr(srv, "local_dispatch_records_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "local_prefill_in_flight", {})
    monkeypatch.setattr(srv, "local_prefill_in_flight_lock", asyncio.Lock())

    import proxy.router as router_mod

    monkeypatch.setattr(router_mod, "_is_self_healing_active", lambda: False)
    monkeypatch.setattr(
        router_mod,
        "_handle_session",
        AsyncMock(
            return_value={
                "session_id": SESSION_A,
                "session_id_header": SESSION_A,
                "session_explicit": True,
                "session_created": False,
                "is_delta_request": False,
                "session_fallback_reason": None,
                "delta_messages": None,
                "original_message_count": 1,
                "body_json": {
                    "model": "plan",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                "body_override": None,
            }
        ),
    )
    monkeypatch.setattr(router_mod, "_build_slot_context", lambda *_: (None, None, 3.0))
    monkeypatch.setattr(router_mod, "_resolve_slot_model_name", lambda model, *_: model)
    monkeypatch.setattr(router_mod, "_check_slot_availability", AsyncMock(return_value=None))
    monkeypatch.setattr(
        router_mod, "_update_session_and_slot", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        router_mod,
        "_call_with_empty_retry",
        AsyncMock(
            return_value=SimpleNamespace(
                status_code=200,
                headers={"content-type": "application/json"},
                content=b'{"choices":[]}',
            )
        ),
    )
    return router_mod


def _router_server_cfg(**overrides) -> dict:
    cfg = {
        "server": {
            "llama_server_port": 8080,
            "session_slot_pool_size": 3,
            "max_concurrent_queries": 16,
            "local_prefill_deadline_seconds": 90,
            "local_dispatch_first_byte_timeout_seconds": 840,
            "local_dispatch_max_prefill_seconds": 900,
            "local_dispatch_no_progress_timeout_seconds": 180,
            "local_dispatch_lease_timeout_seconds": 30,
            "local_dispatch_lease_chunk_refresh_buffer_seconds": 30,
            "local_max_concurrent_prefills": 0,
            "local_prefill_thrash_cooldown_seconds": DEFAULT_COOLDOWN_SECONDS,
        }
    }
    cfg["server"].update(overrides)
    return cfg


async def _run_proxy_to_local(monkeypatch, server_cfg: dict) -> tuple[Response, object]:
    import json as _json

    from proxy.router import proxy_to_local

    router_mod = _install_router_harness(monkeypatch, server_cfg)
    req = _DummyRequest(
        _json.dumps(
            {
                "model": "plan",
                "messages": [{"role": "user", "content": "hello"}],
            }
        ).encode("utf-8"),
        headers={"X-Session-Id": SESSION_A},
    )
    resp = await proxy_to_local(req, "v1/chat/completions")
    return resp, router_mod


def _router_request() -> _DummyRequest:
    """A streaming request — the prefill-deadline mechanism (and therefore
    the cooldown arming / first-byte clear) lives on the streaming dispatch
    path in router.py."""
    return _DummyRequest(
        json.dumps(
            {
                "model": "plan",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            }
        ).encode("utf-8"),
        headers={"x-session-id": SESSION_A},
    )


@pytest.mark.asyncio
async def test_router_deadline_records_the_cooldown(monkeypatch, clock, caplog):
    """AC1 (wiring): when the prefill deadline cancels a local dispatch,
    router.py records the session's cooldown window (measurable through the
    shared primitives the provider check reads)."""
    from proxy.router import proxy_to_local

    server_cfg = _router_server_cfg()
    router_mod = _install_router_harness(monkeypatch, server_cfg)


    caplog.set_level(logging.INFO, logger="llama-proxy")

    async def _deadline_fires(srv, pending, **kwargs):
        # Discard the never-started upstream request cleanly (the real
        # monitor cancels/awaits it; the stub must not leak the coroutine).
        pending.close()
        raise TimeoutError()

    monkeypatch.setattr(router_mod, "_await_first_byte_with_prefill_monitor", _deadline_fires)

    req = _router_request()
    resp = await proxy_to_local(req, "v1/chat/completions")
    assert resp.status_code == 503

    remaining = helpers._prefill_thrash_cooldown_remaining(
        SESSION_A, server_cfg, now=clock.now
    )
    assert remaining == DEFAULT_COOLDOWN_SECONDS, (
        "the prefill deadline must arm the session's cooldown window"
    )
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "prefill_deadline_exceeded" in messages
    assert "prefill_thrash_cooldown_started" in messages


@pytest.mark.asyncio
async def test_router_successful_first_byte_clears_the_cooldown(monkeypatch, clock):
    """AC3 (wiring): a successful local first byte clears the session's
    cooldown, so the next request may route local again."""
    from proxy.router import proxy_to_local

    from proxy import server as srv

    server_cfg = _router_server_cfg()
    router_mod = _install_router_harness(monkeypatch, server_cfg)

    # The session is cooling down from an earlier deadline event.
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, server_cfg, now=clock.now
    )
    assert (
        helpers._prefill_thrash_cooldown_remaining(SESSION_A, server_cfg, now=clock.now)
        > 0
    )

    upstream = SimpleNamespace(
        status_code=200,
        headers={"content-type": "application/json"},
        content=(
            b'{"id":"test","object":"chat.completion","choices":['
            b'{"index":0,"message":{"role":"assistant","content":"ok"},'
            b'"finish_reason":"stop"}]}'
        ),
    )
    monkeypatch.setattr(
        router_mod, "_call_with_backend_retries", AsyncMock(return_value=upstream)
    )

    monkeypatch.setattr(srv, "logger", MagicMock())

    async def _fast_first_byte(srv, pending, **kwargs):
        # Discard the never-awaited upstream request; a healthy dispatch
        # bypasses the backend call in this harness.
        pending.close()
        # A healthy (non-SSE buffered) upstream answer: the first byte arrived
        # inside the deadline.
        upstream = SimpleNamespace(
            status_code=200,
            headers={
                "content-type": "application/json",
                "x-session-restored": "1",
            },
        )
        upstream.aread = AsyncMock(
            return_value=(
                b'{"id":"test","object":"chat.completion","choices":'
                b'[{"index":0,"message":{"role":"assistant",'
                b'"content":"ok"},"finish_reason":"stop"}]}'
            )
        )
        return AsyncMock(), upstream

    monkeypatch.setattr(
        router_mod, "_await_first_byte_with_prefill_monitor", _fast_first_byte
    )

    req = _router_request()
    resp = await proxy_to_local(req, "v1/chat/completions")
    assert resp.status_code == 200
    assert (
        helpers._prefill_thrash_cooldown_remaining(SESSION_A, server_cfg, now=clock.now)
        == 0
    ), "a successful local first byte must clear the session's cooldown"


# ═════════════════════════════════════════════════════════════════════════════
# Cooldown scope and free-GPU release (LP-0MVT2KSC6004UZSD)
#
# Parent: LP-0MVRMI5HK00558TB — AC2 (observable behaviour) and AC3 (coverage).
# Implementation items: LP-0MVT2KSXY002O75C (fingerprint scope + free-GPU
# release) and LP-0MVT2KSN3007QYFK (abandoned-slot registry the release reads).
#
# Measured defect (2026-10-24 20:00 → 2026-10-25 07:00, proxy logs):
# 916 ``routing_skip_local reason=prefill_thrash_cooldown`` skips across 46
# sessions — 241 of them for 21 ``herdr-*`` sessions and 336 (36.7 %) where the
# nearest ≤120 s status poll reported ``local_in_flight_requests=0`` (308 of
# those ``capacity_hint=idle``, ``available_slots=2``). A 600 s session-wide
# veto armed by one doomed prompt starved unrelated, much smaller prompts on a
# demonstrably free GPU.
#
# Red-first encoding: the defect assertions are pinned with
# ``pytest.mark.xfail(strict=True)`` so this test-only item can land with a
# green suite; the implementation item removes each marker as it turns the
# assertion green (``strict`` = an unexpectedly passing test fails the suite,
# so a marker cannot be forgotten).
# ═════════════════════════════════════════════════════════════════════════════

"""Red-first markers removed.

The scope/free-GPU assertions were pinned with
``pytest.mark.xfail(strict=True)`` by the prerequisite test item
(LP-0MVT2KSC6004UZSD). LP-0MVT2KSXY002O75C removed every marker as it turned
the assertions green; they now run unmarked, which is the only state that
proves the fix.
"""

PROMPT_P = "x" * 40_000  # the doomed large prompt
PROMPT_Q = "short unrelated prompt"

#: A fresh ``/slots`` snapshot proving the GPU free: two idle slots, nothing
#: processing.
_GPU_FREE_SNAPSHOT = [
    {"slot_id": 0, "is_processing": False, "n_decoded": None, "context_tokens": None},
    {"slot_id": 1, "is_processing": False, "n_decoded": None, "context_tokens": None},
]


def _request_with_prompt(session_id: str | None, text: str) -> _DummyRequest:
    """A request for *session_id* carrying *text* as the user prompt."""
    headers = {"x-session-id": session_id} if session_id else {}
    body = json.dumps(
        {"model": "test", "messages": [{"role": "user", "content": text}]}
    ).encode()
    return _DummyRequest(body=body, headers=headers)


def _body_for_prompt(text: str) -> dict:
    return {"model": "test", "messages": [{"role": "user", "content": text}]}


def _fingerprint(text: str) -> str:
    """The prompt fingerprint the implementation derives from the payload."""
    fp = helpers._prefill_thrash_fingerprint(_body_for_prompt(text))
    assert isinstance(fp, str) and fp, "a fingerprint must be derivable"
    return fp


@pytest.fixture
def gpu_free(monkeypatch, clock):
    """A fresh snapshot proving the GPU free, with no live local dispatch.

    Composes the same evidence the admission path already reads: the cached
    ``/slots`` detail snapshot aged by ``observability._last_slot_details_at``
    (via ``proxy.residency``) plus the live dispatch-record count.
    """
    import proxy.observability as obs
    import proxy.server as server

    clock.advance(1_000.0)
    monkeypatch.setattr(obs, "_last_slot_details_cache", [dict(s) for s in _GPU_FREE_SNAPSHOT])
    monkeypatch.setattr(obs, "_last_slot_details_at", clock.now)
    monkeypatch.setattr(server, "local_dispatch_records", {})
    monkeypatch.setattr(server, "local_dispatch_records_lock", asyncio.Lock())
    monkeypatch.setattr(server, "local_prefill_in_flight", {})
    with patch.dict("proxy.session._slot_owners", {}, clear=True):
        yield clock


def _mark_slot_processing():
    """Replace the ``gpu_free`` snapshot with one slot still processing."""
    import proxy.observability as obs

    return [
        {"slot_id": 0, "is_processing": True, "n_decoded": 4, "context_tokens": 40_000},
        {"slot_id": 1, "is_processing": False, "n_decoded": None, "context_tokens": None},
    ]


# ---------------------------------------------------------------------------
# Prompt fingerprint
# ---------------------------------------------------------------------------


class TestPromptFingerprint:
    """AC2: the window is keyed by (session, prompt fingerprint)."""

    def test_same_prompt_yields_the_same_fingerprint(self):
        assert _fingerprint(PROMPT_P) == _fingerprint(PROMPT_P)

    def test_different_prompt_yields_a_different_fingerprint(self):
        assert _fingerprint(PROMPT_P) != _fingerprint(PROMPT_Q)

    def test_message_order_is_significant(self):
        a = helpers._prefill_thrash_fingerprint(
            {"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]}
        )
        b = helpers._prefill_thrash_fingerprint(
            {"messages": [{"role": "user", "content": "b"}, {"role": "user", "content": "a"}]}
        )
        assert a != b

    def test_unavailable_prompt_yields_none(self):
        """A payload with no usable messages cannot be fingerprinted, so the
        gate fails open to its session-wide behaviour."""
        assert helpers._prefill_thrash_fingerprint(None) is None
        assert helpers._prefill_thrash_fingerprint({}) is None
        assert helpers._prefill_thrash_fingerprint({"messages": []}) is None
        assert helpers._prefill_thrash_fingerprint({"messages": "nope"}) is None

    def test_fingerprint_is_bounded(self):
        """The key is a fixed-size digest, not the prompt itself."""
        fp = _fingerprint(PROMPT_P)
        assert len(fp) <= 64
        assert PROMPT_P not in fp


# ---------------------------------------------------------------------------
# Arming / gate scope (router_helpers primitives)
# ---------------------------------------------------------------------------


class TestCooldownScope:
    """AC2: only the re-dispatch the window was armed for is refused."""

    def test_same_prompt_is_still_refused(self, clock):
        """AC2 (i): the gate's intent is preserved."""
        cfg = _server_config()
        helpers._record_prefill_thrash_cooldown(
            SESSION_A, cfg, now=clock.now, tokens=40_000, fingerprint=_fingerprint(PROMPT_P)
        )

        assert (
            helpers._prefill_thrash_cooldown_remaining(
                SESSION_A, cfg, now=clock.now, fingerprint=_fingerprint(PROMPT_P)
            )
            == DEFAULT_COOLDOWN_SECONDS
        )

    def test_different_prompt_is_admitted(self, clock):
        """AC2 (ii): an unrelated prompt from the same session is not vetoed."""
        cfg = _server_config()
        helpers._record_prefill_thrash_cooldown(
            SESSION_A, cfg, now=clock.now, tokens=40_000, fingerprint=_fingerprint(PROMPT_P)
        )

        assert (
            helpers._prefill_thrash_cooldown_remaining(
                SESSION_A, cfg, now=clock.now, fingerprint=_fingerprint(PROMPT_Q)
            )
            == 0
        )

    def test_the_armed_prompt_stays_refused_while_another_is_admitted(self, clock):
        """Both directions in one window: the fingerprint keys are independent."""
        cfg = _server_config()
        fp_p = _fingerprint(PROMPT_P)
        helpers._record_prefill_thrash_cooldown(
            SESSION_A, cfg, now=clock.now, tokens=40_000, fingerprint=fp_p
        )

        assert (
            helpers._prefill_thrash_cooldown_remaining(
                SESSION_A, cfg, now=clock.now, fingerprint=_fingerprint(PROMPT_Q)
            )
            == 0
        )
        assert (
            helpers._prefill_thrash_cooldown_remaining(
                SESSION_A, cfg, now=clock.now, fingerprint=fp_p
            )
            > 0
        )

    def test_two_prompts_arm_independent_windows(self, clock):
        cfg = _server_config()
        fp_p, fp_q = _fingerprint(PROMPT_P), _fingerprint(PROMPT_Q)
        helpers._record_prefill_thrash_cooldown(
            SESSION_A, cfg, now=clock.now, tokens=40_000, fingerprint=fp_p
        )
        clock.advance(120)
        helpers._record_prefill_thrash_cooldown(
            SESSION_A, cfg, now=clock.now, tokens=len(PROMPT_Q), fingerprint=fp_q
        )

        clock.advance(10)
        remaining_p = helpers._prefill_thrash_cooldown_remaining(
            SESSION_A, cfg, now=clock.now, fingerprint=fp_p
        )
        remaining_q = helpers._prefill_thrash_cooldown_remaining(
            SESSION_A, cfg, now=clock.now, fingerprint=fp_q
        )
        assert remaining_p == DEFAULT_COOLDOWN_SECONDS - 130
        assert remaining_q == DEFAULT_COOLDOWN_SECONDS - 10

    def test_arming_without_a_fingerprint_keeps_session_wide_behaviour(self, clock):
        """Fail-open: an arming site that cannot fingerprint must not silently
        weaken the gate — the session-wide window stands."""
        cfg = _server_config()
        helpers._record_prefill_thrash_cooldown(
            SESSION_A, cfg, now=clock.now, tokens=40_000
        )

        assert (
            helpers._prefill_thrash_cooldown_remaining(
                SESSION_A, cfg, now=clock.now, fingerprint=_fingerprint(PROMPT_Q)
            )
            > 0
        )

    def test_gate_without_a_fingerprint_keeps_session_wide_behaviour(self, clock):
        """Fail-open on the other side: a gate site that cannot fingerprint a
        request treats it as possibly the failing prompt."""
        cfg = _server_config()
        helpers._record_prefill_thrash_cooldown(
            SESSION_A, cfg, now=clock.now, tokens=40_000, fingerprint=_fingerprint(PROMPT_P)
        )

        assert (
            helpers._prefill_thrash_cooldown_remaining(
                SESSION_A, cfg, now=clock.now, fingerprint=None
            )
            > 0
        )

    def test_cross_session_isolation_with_fingerprints(self, clock):
        cfg = _server_config()
        fp_p = _fingerprint(PROMPT_P)
        helpers._record_prefill_thrash_cooldown(
            SESSION_A, cfg, now=clock.now, tokens=40_000, fingerprint=fp_p
        )

        assert (
            helpers._prefill_thrash_cooldown_remaining(
                SESSION_B, cfg, now=clock.now, fingerprint=fp_p
            )
            == 0
        )

    def test_a_comparable_success_clears_only_the_matching_fingerprint(self, clock):
        """The recovery-fraction clear (LP-0MVQITN6P008DPLB) is per
        fingerprint: a comparable success proves *that* prompt works."""
        cfg = _server_config()
        fp_p, fp_q = _fingerprint(PROMPT_P), _fingerprint(PROMPT_Q)
        helpers._record_prefill_thrash_cooldown(
            SESSION_A, cfg, now=clock.now, tokens=40_000, fingerprint=fp_p
        )
        helpers._record_prefill_thrash_cooldown(
            SESSION_A, cfg, now=clock.now, tokens=40_000, fingerprint=fp_q
        )

        assert (
            helpers._clear_prefill_thrash_cooldown(
                SESSION_A, tokens=40_000, config=cfg, fingerprint=fp_p
            )
            is True
        )
        assert (
            helpers._prefill_thrash_cooldown_remaining(
                SESSION_A, cfg, now=clock.now, fingerprint=fp_p
            )
            == 0
        )
        assert (
            helpers._prefill_thrash_cooldown_remaining(
                SESSION_A, cfg, now=clock.now, fingerprint=fp_q
            )
            > 0
        )

    def test_zero_knob_still_disables_arming_and_skipping(self, clock):
        """Rollback: ``local_prefill_thrash_cooldown_seconds: 0``."""
        cfg = _server_config(local_prefill_thrash_cooldown_seconds=0)
        assert (
            helpers._record_prefill_thrash_cooldown(
                SESSION_A, cfg, now=clock.now, tokens=40_000,
                fingerprint=_fingerprint(PROMPT_P),
            )
            == 0
        )
        assert (
            helpers._prefill_thrash_cooldown_remaining(
                SESSION_A, cfg, now=clock.now, fingerprint=_fingerprint(PROMPT_P)
            )
            == 0
        )


# ---------------------------------------------------------------------------
# Free-GPU release (router_helpers helper + provider gate)
# ---------------------------------------------------------------------------


class TestFreeGpuRelease:
    """AC2: a window must never outlive the condition that justified it."""

    def test_free_gpu_releases_an_armed_window(self, gpu_free, caplog):
        clock = gpu_free
        cfg = _server_config()
        fp_p = _fingerprint(PROMPT_P)
        helpers._record_prefill_thrash_cooldown(
            SESSION_A, cfg, now=clock.now, tokens=40_000, fingerprint=fp_p
        )
        caplog.set_level(logging.INFO, logger="llama-proxy")

        released = helpers._release_prefill_thrash_cooldown_on_free_gpu(
            SESSION_A, cfg, now=clock.now
        )

        assert released is True
        assert (
            helpers._prefill_thrash_cooldown_remaining(
                SESSION_A, cfg, now=clock.now, fingerprint=fp_p
            )
            == 0
        )
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "prefill_thrash_cooldown_released" in messages
        assert "reason=free_gpu" in messages

    def test_live_dispatch_never_releases(self, gpu_free):
        """A live dispatch record means the pool is working, not free."""
        import proxy.server as server

        clock = gpu_free
        server.local_dispatch_records = {
            "live-session": {"active": True, "expires_at": clock.now + 60}
        }
        cfg = _server_config()
        helpers._record_prefill_thrash_cooldown(
            SESSION_A, cfg, now=clock.now, tokens=40_000,
            fingerprint=_fingerprint(PROMPT_P),
        )

        assert (
            helpers._release_prefill_thrash_cooldown_on_free_gpu(
                SESSION_A, cfg, now=clock.now
            )
            is False
        )
        assert (
            helpers._prefill_thrash_cooldown_remaining(SESSION_A, cfg, now=clock.now)
            > 0
        )

    def test_stale_snapshot_never_releases(self, gpu_free, monkeypatch):
        """A fossil ``/slots`` snapshot proves nothing about the GPU now."""
        import proxy.observability as obs

        clock = gpu_free
        monkeypatch.setattr(obs, "_last_slot_details_at", clock.now - 120.0)
        cfg = _server_config()
        helpers._record_prefill_thrash_cooldown(
            SESSION_A, cfg, now=clock.now, tokens=40_000,
            fingerprint=_fingerprint(PROMPT_P),
        )

        assert (
            helpers._release_prefill_thrash_cooldown_on_free_gpu(
                SESSION_A, cfg, now=clock.now
            )
            is False
        )
        assert (
            helpers._prefill_thrash_cooldown_remaining(SESSION_A, cfg, now=clock.now)
            > 0
        )

    def test_abandoned_but_processing_slot_never_releases(self, gpu_free, monkeypatch):
        """A ghost still prefilling means the GPU is busy: keep the window."""
        import proxy.observability as obs

        clock = gpu_free
        monkeypatch.setattr(obs, "_last_slot_details_cache", _mark_slot_processing())
        helpers._record_abandoned_slot(
            0, session_id="herdr-ghost", reason="prefill_deadline", now=clock.now
        )
        cfg = _server_config()
        helpers._record_prefill_thrash_cooldown(
            SESSION_A, cfg, now=clock.now, tokens=40_000,
            fingerprint=_fingerprint(PROMPT_P),
        )

        assert (
            helpers._release_prefill_thrash_cooldown_on_free_gpu(
                SESSION_A, cfg, now=clock.now
            )
            is False
        )
        assert (
            helpers._prefill_thrash_cooldown_remaining(
                SESSION_A, cfg, now=clock.now, fingerprint=_fingerprint(PROMPT_P)
            )
            > 0
        )

    def test_live_processing_slot_never_releases(self, gpu_free, monkeypatch):
        """Genuine live occupancy blocks the release even without a ghost."""
        import proxy.observability as obs

        clock = gpu_free
        monkeypatch.setattr(obs, "_last_slot_details_cache", _mark_slot_processing())
        cfg = _server_config()
        helpers._record_prefill_thrash_cooldown(
            SESSION_A, cfg, now=clock.now, tokens=40_000,
            fingerprint=_fingerprint(PROMPT_P),
        )

        assert (
            helpers._release_prefill_thrash_cooldown_on_free_gpu(
                SESSION_A, cfg, now=clock.now
            )
            is False
        )

    def test_no_window_armed_is_a_no_op(self, gpu_free):
        """Nothing to release → False and no lifecycle change."""
        clock = gpu_free
        assert (
            helpers._release_prefill_thrash_cooldown_on_free_gpu(
                SESSION_A, _server_config(), now=clock.now
            )
            is False
        )


# ---------------------------------------------------------------------------
# End-to-end through the provider fallback chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_different_prompt_is_dispatched_locally_on_a_free_gpu(
    fallback_harness, gpu_free, clock, caplog,
):
    """AC2 (ii): the measured 336 free-GPU skips must not happen."""
    cfg = _server_config()
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, cfg, now=clock.now, tokens=40_000,
        fingerprint=_fingerprint(PROMPT_P),
    )
    caplog.set_level(logging.INFO, logger="llama-proxy.provider")

    resp = await fallback_harness(_request_with_prompt(SESSION_A, PROMPT_Q), cfg)

    assert resp.status_code == 200
    assert fallback_harness.call_log == ["local"], (
        "an unrelated prompt must be routed local when the GPU is free"
    )
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "reason=prefill_thrash_cooldown" not in messages


@pytest.mark.asyncio
async def test_same_prompt_is_still_refused_at_the_gate_on_a_busy_gpu(
    fallback_harness, gpu_free, monkeypatch, clock, caplog,
):
    """AC2 (i): the 41-local-POST thrash stays suppressed while the GPU is
    busy finishing the abandoned prefill."""
    import proxy.observability as obs

    monkeypatch.setattr(obs, "_last_slot_details_cache", _mark_slot_processing())
    helpers._record_abandoned_slot(
        0, session_id="herdr-ghost", reason="prefill_deadline", now=clock.now
    )
    cfg = _server_config()
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, cfg, now=clock.now, tokens=40_000,
        fingerprint=_fingerprint(PROMPT_P),
    )
    caplog.set_level(logging.INFO, logger="llama-proxy.provider")

    resp = await fallback_harness(_request_with_prompt(SESSION_A, PROMPT_P), cfg)

    assert resp.status_code == 200
    assert fallback_harness.call_log == ["remote"]
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert messages.count("routing_skip_local") == 1
    assert "reason=prefill_thrash_cooldown" in messages
    assert "cooldown_remaining=" in messages


@pytest.mark.asyncio
async def test_failing_prompt_is_still_skipped_when_evidence_is_stale(
    fallback_harness, gpu_free, monkeypatch, clock,
):
    """The release is not a blanket disable: with no fresh evidence the window
    stays armed and the *failing* prompt is still refused.

    (The unrelated-prompt case is asserted separately: LP-0MVT2KSXY002O75C
    narrowed the veto to the failing prompt, so a different prompt is admitted
    regardless of the evidence — that is AC2 (ii), not a release.)
    """
    import proxy.observability as obs

    monkeypatch.setattr(obs, "_last_slot_details_at", clock.now - 120.0)
    cfg = _server_config()
    helpers._record_prefill_thrash_cooldown(
        SESSION_A, cfg, now=clock.now, tokens=40_000,
        fingerprint=_fingerprint(PROMPT_P),
    )

    resp = await fallback_harness(_request_with_prompt(SESSION_A, PROMPT_P), cfg)

    assert resp.status_code == 200
    assert fallback_harness.call_log == ["remote"]


@pytest.mark.asyncio
async def test_zero_knob_routes_the_failing_prompt_locally(
    fallback_harness, clock,
):
    """Rollback knob: 0 disables the skip even for the failing prompt."""
    cfg = _server_config(local_prefill_thrash_cooldown_seconds=0)

    resp = await fallback_harness(_request_with_prompt(SESSION_A, PROMPT_P), cfg)

    assert resp.status_code == 200
    assert fallback_harness.call_log == ["local"]


# ═════════════════════════════════════════════════════════════════════════════
# Deadline branch wiring: the abandoned slot is attributable
# (LP-0MVT2KSN3007QYFK → parent LP-0MVRMI5HK00558TB AC1)
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_router_deadline_records_the_abandoned_slot(monkeypatch, clock):
    """AC1 (wiring): the deadline branch hands the dispatched slot id to the
    release, so the ghost llama-server keeps computing is attributable on the
    status payload instead of reading as live proxy-owned work."""
    from proxy.router import proxy_to_local

    server_cfg = _router_server_cfg()
    router_mod = _install_router_harness(monkeypatch, server_cfg)
    # Dispatch to a real slot id so the release has an identity to record.
    monkeypatch.setattr(
        router_mod, "_build_slot_context", lambda *_a, **_k: (1, "slot-1.bin", 3.0)
    )
    monkeypatch.setattr(
        router_mod, "_restore_slot_snapshot", AsyncMock(return_value=False)
    )

    async def _deadline_fires(srv, pending, **kwargs):
        pending.close()
        raise TimeoutError()

    monkeypatch.setattr(
        router_mod, "_await_first_byte_with_prefill_monitor", _deadline_fires
    )

    resp = await proxy_to_local(_router_request(), "v1/chat/completions")
    assert resp.status_code == 503

    entry = helpers._abandoned_slot_for(1, now=clock.now)
    assert entry is not None, (
        "the prefill deadline must record the abandoned slot identity "
        "(parent LP-0MVRMI5HK00558TB AC1)"
    )
    assert entry["reason"] == "prefill_deadline"
    assert entry["session_id"] == SESSION_A
