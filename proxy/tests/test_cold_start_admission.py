"""
Hermetic tests for cold-start admission control (LP-0MUCEFCAT005NFNN).

Context
-------
After a proxy restart the co-located llama-server reloads the model cold
(no prompt cache). Several large-context sessions dispatching at once each
issue a full GPU prefill concurrently; first bytes take minutes and the
local pool wedges. The cold-start window caps concurrent local dispatches
(default 1) until the first prefill completes and warms the cache.

These tests verify hermetically (no live llama-server):

1. A ``backend_ready`` False->True transition arms the cold window.
2. ``is_cold`` is True within the grace window and self-expires after it.
3. ``mark_warm`` lifts the cap immediately.
4. ``effective_max_concurrent`` clamps during cold and is unchanged warm.
5. Config defaults / overrides / disable.
6. Integration: N concurrent explicit sessions after a simulated restart
   never exceed the cold cap; each admitted request proceeds and the cap
   lifts once warm.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from proxy import cold_start


@pytest.fixture(autouse=True)
def _reset_cold_start():
    """Reset the module-level cold-start state around every test."""
    cold_start.reset()
    yield
    cold_start.reset()


def _server_config(**overrides) -> dict:
    cfg = {"session_slot_pool_size": 3}
    cfg.update(overrides)
    return {"server": cfg}


# ═══════════════════════════════════════════════════════════════════════════════
# State machine
# ═══════════════════════════════════════════════════════════════════════════════


def test_model_loaded_arms_cold():
    """AC: a model-load event enters the cold window with a fresh clock."""
    assert not cold_start.is_cold(_server_config())

    cold_start.note_model_loaded()
    assert cold_start.is_cold(_server_config())

    # A repeat load (e.g. a model switch) re-arms with a fresh clock.
    first_loaded_at = cold_start._state.loaded_at
    cold_start.note_model_loaded(now=first_loaded_at + 10)
    assert cold_start._state.cold is True
    assert cold_start._state.loaded_at == first_loaded_at + 10


def test_backend_down_clears_cold():
    """A restart (not-ready) clears the cold state so the next load re-arms."""
    cold_start.note_model_loaded()
    assert cold_start.is_cold(_server_config())

    cold_start.note_backend_down()
    assert not cold_start.is_cold(_server_config())
    assert cold_start._state.loaded_at is None

    # Next model load re-arms with a fresh clock.
    cold_start.note_model_loaded()
    assert cold_start.is_cold(_server_config())


def test_is_cold_self_expires_after_grace():
    """AC: the cold window self-expires after ``local_cold_start_grace_seconds``."""
    config = _server_config(local_cold_start_grace_seconds=30)
    cold_start.note_model_loaded()
    loaded_at = cold_start._state.loaded_at

    assert cold_start.is_cold(config, now=loaded_at + 29)
    assert not cold_start.is_cold(config, now=loaded_at + 30)
    assert not cold_start.is_cold(config, now=loaded_at + 100)


def test_grace_zero_disables_cold_window():
    """grace <= 0 disables the cold window entirely."""
    config = _server_config(local_cold_start_grace_seconds=0)
    cold_start.note_model_loaded()
    assert not cold_start.is_cold(config)


def test_mark_warm_lifts_cap():
    """AC: a completed warm prefill lifts the cap immediately."""
    cold_start.note_model_loaded()
    assert cold_start.is_cold(_server_config())
    cold_start.mark_warm()
    assert not cold_start.is_cold(_server_config())


# ═══════════════════════════════════════════════════════════════════════════════
# Config resolution
# ═══════════════════════════════════════════════════════════════════════════════


def test_config_defaults():
    cfg = _server_config()
    assert cold_start.grace_seconds(cfg) == 120.0
    assert cold_start.max_concurrent(cfg) == 1
    assert cold_start.retry_after_seconds(cfg) == 15


def test_config_overrides():
    cfg = _server_config(
        local_cold_start_grace_seconds=60,
        local_cold_start_max_concurrent=2,
        local_cold_start_retry_after_seconds=5,
    )
    assert cold_start.grace_seconds(cfg) == 60.0
    assert cold_start.max_concurrent(cfg) == 2
    assert cold_start.retry_after_seconds(cfg) == 5


def test_config_invalid_values_fall_back_to_defaults():
    cfg = _server_config(
        local_cold_start_grace_seconds="not-a-number",
        local_cold_start_max_concurrent=0,  # clamped to >= 1
        local_cold_start_retry_after_seconds=None,
    )
    assert cold_start.grace_seconds(cfg) == 120.0
    assert cold_start.max_concurrent(cfg) == 1
    assert cold_start.retry_after_seconds(cfg) == 15


def test_config_accepts_flat_server_section():
    """A bare server-section dict (no outer 'server' key) is accepted."""
    flat = {"local_cold_start_grace_seconds": 45, "local_cold_start_max_concurrent": 2}
    assert cold_start.grace_seconds(flat) == 45.0
    assert cold_start.max_concurrent(flat) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# effective_max_concurrent
# ═══════════════════════════════════════════════════════════════════════════════


def test_effective_max_clamps_when_cold():
    cfg = _server_config(local_cold_start_max_concurrent=1)
    cold_start.note_model_loaded()
    assert cold_start.effective_max_concurrent(cfg, 3) == 1


def test_effective_max_unchanged_when_warm():
    cfg = _server_config(local_cold_start_max_concurrent=1)
    assert cold_start.effective_max_concurrent(cfg, 3) == 3

    cold_start.note_model_loaded()
    cold_start.mark_warm()
    assert cold_start.effective_max_concurrent(cfg, 3) == 3


def test_effective_max_respects_lower_configured_max():
    """The cap never raises the configured max."""
    cfg = _server_config(local_cold_start_max_concurrent=5)
    cold_start.note_model_loaded()
    assert cold_start.effective_max_concurrent(cfg, 2) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: concurrent sessions after a simulated restart
# ═══════════════════════════════════════════════════════════════════════════════


def _make_dispatch_srv(config: dict) -> SimpleNamespace:
    return SimpleNamespace(
        config=config,
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
        local_generating_queries=0,
        local_generating_sessions=set(),
        local_generating_queries_lock=asyncio.Lock(),
        local_prefill_in_flight={},
        local_prefill_in_flight_lock=asyncio.Lock(),
        logger=MagicMock(),
    )


@pytest.mark.asyncio
async def test_cold_start_caps_concurrent_dispatches_then_lifts():
    """AC: after a simulated restart, N concurrent large local prompts do
    not all prefill at once; the cap lifts once the first request warms."""
    from proxy.router_helpers import _try_acquire_local_dispatch

    cfg = _server_config(
        session_slot_pool_size=3,
        local_cold_start_grace_seconds=120,
        local_cold_start_max_concurrent=1,
    )
    srv = _make_dispatch_srv(cfg)

    # Simulate the restart: backend becomes ready (model just loaded).
    cold_start.note_backend_down()
    cold_start.note_model_loaded()
    assert cold_start.is_cold(cfg)

    configured_max = 3
    cold_max = cold_start.effective_max_concurrent(cfg, configured_max)
    assert cold_max == 1

    # First large prompt is admitted.
    acquired_a, *_ = await _try_acquire_local_dispatch(
        srv, max_local=cold_max, session_key="large-a", backend="local",
        model_name="qwen",
    )
    assert acquired_a is True

    # Concurrent large prompts B/C are denied while A prefills.
    acquired_b, *_ = await _try_acquire_local_dispatch(
        srv, max_local=cold_max, session_key="large-b", backend="local",
        model_name="qwen",
    )
    assert acquired_b is False, "Second concurrent prefill must be deferred while cold"

    # A's prefill completes -> first data byte -> cache warm -> cap lifts.
    cold_start.mark_warm()
    assert not cold_start.is_cold(cfg)
    warm_max = cold_start.effective_max_concurrent(cfg, configured_max)
    assert warm_max == configured_max

    acquired_b2, *_ = await _try_acquire_local_dispatch(
        srv, max_local=warm_max, session_key="large-b", backend="local",
        model_name="qwen",
    )
    assert acquired_b2 is True, "Deferred session must be admitted once warm"


@pytest.mark.asyncio
async def test_cold_start_defers_only_until_grace_expires():
    """AC: the cap is bounded by the grace window, not permanent."""
    cfg = _server_config(
        session_slot_pool_size=3,
        local_cold_start_grace_seconds=10,
        local_cold_start_max_concurrent=1,
    )
    cold_start.note_model_loaded()

    # Within grace: capped.
    assert cold_start.effective_max_concurrent(cfg, 3) == 1

    # Simulate the grace window elapsing (loaded 15s ago, grace 10s).
    cold_start._state.loaded_at = time.monotonic() - 15
    assert not cold_start.is_cold(cfg)
    assert cold_start.effective_max_concurrent(cfg, 3) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Router integration: cold-start 503 + Retry-After
# ═══════════════════════════════════════════════════════════════════════════════


class _DummyRequest:
    def __init__(self, body: bytes):
        self._body = body
        self.headers = {"content-type": "application/json"}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


def _install_router_harness(monkeypatch, server_cfg: dict):
    """Install the minimal server state + mocks proxy_to_local needs to reach
    the local dispatch gate and return its 503."""
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
    monkeypatch.setattr(srv, "local_active_queries", 1)
    monkeypatch.setattr(srv, "local_active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "local_generating_queries", 0)
    monkeypatch.setattr(srv, "local_generating_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "local_generating_sessions", set())
    monkeypatch.setattr(
        srv,
        "local_dispatch_records",
        {
            "sess-a": {
                "backend": "local",
                "started_at": time.monotonic(),
                "active": True,
                "expires_at": time.monotonic() + 1000,
                "last_progress_ts": time.monotonic(),
            }
        },
    )
    monkeypatch.setattr(srv, "local_dispatch_records_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "local_prefill_in_flight", {"sess-a": True})
    monkeypatch.setattr(srv, "local_prefill_in_flight_lock", asyncio.Lock())

    import proxy.router as router_mod

    monkeypatch.setattr(router_mod, "_is_self_healing_active", lambda: False)
    monkeypatch.setattr(
        router_mod,
        "_handle_session",
        AsyncMock(
            return_value={
                "session_id": "sess-b",
                "session_id_header": "sess-b",
                "session_explicit": True,
                "session_created": False,
                "is_delta_request": False,
                "session_fallback_reason": None,
                "delta_messages": None,
                "original_message_count": 1,
                "body_json": {
                    "model": "plan",
                    "messages": [{"role": "user", "content": "x" * 100_000}],
                },
                "body_override": None,
            }
        ),
    )
    monkeypatch.setattr(router_mod, "_build_slot_context", lambda *_: (None, None, 3.0))
    monkeypatch.setattr(router_mod, "_resolve_slot_model_name", lambda model, *_: model)
    monkeypatch.setattr(router_mod, "_check_slot_availability", AsyncMock(return_value=None))


@pytest.mark.asyncio
async def test_proxy_to_local_cold_start_returns_retryable_503(monkeypatch):
    """AC: a deferred local request during cold start gets 503 + Retry-After
    with reason=cold_start (distinct from a warm lease denial)."""
    import json as _json

    from proxy.router import proxy_to_local

    server_cfg = {
        "server": {
            "llama_server_port": 8080,
            "session_slot_pool_size": 3,
            "max_concurrent_queries": 16,
            "local_cold_start_grace_seconds": 120,
            "local_cold_start_max_concurrent": 1,
            "local_cold_start_retry_after_seconds": 7,
        }
    }
    _install_router_harness(monkeypatch, server_cfg)

    # Simulate the restart: backend just became ready.
    cold_start.note_backend_down()
    cold_start.note_model_loaded()
    assert cold_start.is_cold(server_cfg)

    req = _DummyRequest(
        _json.dumps(
            {
                "model": "plan",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            }
        ).encode("utf-8")
    )
    resp = await proxy_to_local(req, "v1/chat/completions")

    assert resp.status_code == 503
    payload = _json.loads(resp.body)
    assert payload["reason"] == "cold_start"
    assert payload["error"]["code"] == "cold_start"
    assert resp.headers.get("Retry-After") == "7"


@pytest.mark.asyncio
async def test_proxy_to_local_warm_bypasses_cold_cap(monkeypatch):
    """AC: in warm state the full pool is available (no cold deferral)."""
    import json as _json

    from proxy.router import proxy_to_local

    from proxy import server as srv

    server_cfg = {
        "server": {
            "llama_server_port": 8080,
            "session_slot_pool_size": 3,
            "max_concurrent_queries": 16,
            "local_cold_start_grace_seconds": 120,
            "local_cold_start_max_concurrent": 1,
        }
    }
    _install_router_harness(monkeypatch, server_cfg)
    # Backend has been ready and warm for a while: arm then lift the cold
    # window so the request sees the full pool.
    cold_start.note_model_loaded()
    cold_start.mark_warm()
    assert not cold_start.is_cold(server_cfg)

    req = _DummyRequest(
        _json.dumps(
            {
                "model": "plan",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            }
        ).encode("utf-8")
    )
    resp = await proxy_to_local(req, "v1/chat/completions")
    # The dispatch gate did not defer for cold start.
    payload = _json.loads(resp.body)
    assert payload.get("reason") != "cold_start", (
        "Warm state must not cold-defer a request when the pool has room"
    )
