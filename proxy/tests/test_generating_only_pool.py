"""
Unit tests for generating-only pool occupancy.

Tests that only generating slots count toward session_slot_pool_size:
- Slots are NOT counted during prefill phase
- Slots ARE counted when generating (first-byte onward)
- Slots are released immediately on stream end (no 30s inactive hold)
- Edge cases: concurrent streams, rapid prefill/generate transitions
"""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


class _DummyRequest:
    def __init__(self, body: bytes):
        self._body = body
        self.headers = {"content-type": "application/json"}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


# ---------------------------------------------------------------------------
# AC1: Generating-only pool occupancy — increment only on generating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_increment_local_active_queries_counts_at_dispatch():
    """_increment_local_active_queries still increments for dispatch tracking.

    The total counter tracks dispatches (for lease ownership); the new
    generating-only counter is separate.
    """
    from proxy.router_helpers import (
        _increment_local_active_queries,
    )

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 60}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
    )

    await _increment_local_active_queries(
        srv, session_key="sess-1", backend="local"
    )

    assert srv.local_active_queries == 1
    assert "sess-1" in srv.local_dispatch_records
    assert srv.local_dispatch_records["sess-1"]["active"] is True


@pytest.mark.asyncio
async def test_generating_only_counter_not_incremented_at_dispatch():
    """Generating-only counter must NOT be incremented during dispatch (prefill phase).

    The key insight of this change: prefill time does NOT count against
    session_slot_pool_size. Only generating time (first-byte onward) counts.
    """
    from proxy.router_helpers import (
        _get_generating_only_count,
        _increment_generating_only_slot,
        _increment_local_active_queries,
    )

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 60}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_generating_queries=0,
        local_generating_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
    )

    # Dispatch happens (prefill starts) — total increments, generating does NOT
    await _increment_local_active_queries(
        srv, session_key="sess-1", backend="local"
    )

    assert srv.local_active_queries == 1
    # Generating-only counter is NOT incremented during dispatch
    assert _get_generating_only_count(srv) == 0


@pytest.mark.asyncio
async def test_generating_only_counter_incremented_on_first_byte():
    """Generating-only counter IS incremented when first chunk arrives."""
    from proxy.router_helpers import (
        _get_generating_only_count,
        _increment_generating_only_slot,
    )

    srv = SimpleNamespace(
        local_generating_queries=0,
        local_generating_queries_lock=asyncio.Lock(),
    )

    # First chunk arrives (first-byte) — generating counter increments
    await _increment_generating_only_slot(srv, session_key="sess-1")

    assert _get_generating_only_count(srv) == 1


@pytest.mark.asyncio
async def test_generating_only_counter_decremented_on_stream_end():
    """Generating-only counter IS decremented when stream ends."""
    from proxy.router_helpers import (
        _decrement_generating_only_slot,
        _get_generating_only_count,
        _increment_generating_only_slot,
    )

    srv = SimpleNamespace(
        local_generating_queries=0,
        local_generating_queries_lock=asyncio.Lock(),
    )

    await _increment_generating_only_slot(srv, session_key="sess-1")
    assert _get_generating_only_count(srv) == 1

    await _decrement_generating_only_slot(srv, session_key="sess-1")
    assert _get_generating_only_count(srv) == 0


@pytest.mark.asyncio
async def test_no_post_stream_inactive_hold_for_generating_counter():
    """After stream end, generating counter is released immediately — no hold.

    The old behavior kept `local_active_queries` incremented for 30s after
    stream end. The new behavior releases the generating-only slot immediately
    so it can be reused by another session during the next request's prefill.
    """
    from proxy.router_helpers import (
        _decrement_generating_only_slot,
        _decrement_local_active_queries,
        _get_generating_only_count,
        _increment_generating_only_slot,
        _increment_local_active_queries,
    )

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 180}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_generating_queries=0,
        local_generating_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
    )

    # 1. Dispatch (prefill starts) — total counts, generating does NOT
    await _increment_local_active_queries(
        srv, session_key="sess-1", backend="local"
    )
    assert srv.local_active_queries == 1
    assert _get_generating_only_count(srv) == 0

    # 2. First byte — generating starts counting
    await _increment_generating_only_slot(srv, session_key="sess-1")
    assert _get_generating_only_count(srv) == 1

    # 3. Stream end — releasing generating counter immediately, no hold
    await _decrement_generating_only_slot(srv, session_key="sess-1")
    assert _get_generating_only_count(srv) == 0

    # 4. Also mark dispatch record inactive (lease persists for ownership)
    await _decrement_local_active_queries(srv, session_key="sess-1")
    # Total counter goes to 0 (was 1 from increment_local, decremented)
    assert srv.local_active_queries == 0
    # Generating counter stays 0
    assert _get_generating_only_count(srv) == 0


# ---------------------------------------------------------------------------
# AC2: Pool gate uses generating-only count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_try_acquire_checks_generating_only_count():
    """_try_acquire_local_dispatch must gate on generating-only count.

    The pool should read as having a free slot when a request is in prefill
    (generating count = 0) even if local_active_queries > 0.
    """
    from proxy.router_helpers import (
        _get_generating_only_count,
        _try_acquire_local_dispatch,
    )

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 60}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_generating_queries=0,
        local_generating_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
    )

    # Session A is in prefill: dispatch tracked, generating NOT counted
    await _try_acquire_local_dispatch(
        srv, max_local=1, session_key="sess-a", backend="local"
    )

    # Session B should be allowed — generating count is 0
    acquired_b, owner_b, _, _ = await _try_acquire_local_dispatch(
        srv, max_local=1, session_key="sess-b", backend="local"
    )

    # With generating-only: both can dispatch because generating count < max_local
    # (sess-a is in prefill, not generating yet)
    assert acquired_b is True, (
        "Session B should acquire when only sess-a is in prefill (generating=0)"
    )


@pytest.mark.asyncio
async def test_try_acquire_denied_when_generating_at_capacity():
    """When generating count equals max_local, new dispatches are denied."""
    from proxy.router_helpers import (
        _decrement_local_active_queries,
        _get_generating_only_count,
        _increment_generating_only_slot,
        _try_acquire_local_dispatch,
    )

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 60}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_generating_queries=0,
        local_generating_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
    )

    # Session A acquires and enters generating phase
    await _try_acquire_local_dispatch(
        srv, max_local=1, session_key="sess-a", backend="local"
    )
    await _increment_generating_only_slot(srv, session_key="sess-a")
    assert _get_generating_only_count(srv) == 1

    # Session B tries to acquire — should be denied (generating at capacity)
    acquired_b, owner_b, active_b, _ = await _try_acquire_local_dispatch(
        srv, max_local=1, session_key="sess-b", backend="local"
    )

    assert acquired_b is False, (
        "Session B should be denied when sess-a is generating (generating=1/1)"
    )
    assert owner_b == "sess-a"
    assert active_b == 1


# ---------------------------------------------------------------------------
# AC3: Concurrent streams with generating-only semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_prefill_allows_multiple_dispatches():
    """Multiple sessions can dispatch concurrently during prefill (generating=0).

    This is the key benefit: if llama-server has capacity for 3 concurrent
    prefill requests but only 1 generating, the old pool would block at 1
    (since active=1), but the new pool allows 3 prefill dispatches.
    """
    from proxy.router_helpers import (
        _get_generating_only_count,
        _try_acquire_local_dispatch,
    )

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 60}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_generating_queries=0,
        local_generating_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
    )

    # Three sessions dispatch concurrently (all in prefill)
    for i in range(3):
        acquired, owner, _, _ = await _try_acquire_local_dispatch(
            srv, max_local=3, session_key=f"sess-{i}", backend="local"
        )
        assert acquired is True, f"sess-{i} should acquire during prefill"

    assert _get_generating_only_count(srv) == 0, (
        "No sessions are generating yet, so generating count should be 0"
    )


@pytest.mark.asyncio
async def test_generating_phase_blocks_new_dispatches():
    """When sessions are generating, new dispatches are blocked."""
    from proxy.router_helpers import (
        _get_generating_only_count,
        _increment_generating_only_slot,
        _try_acquire_local_dispatch,
    )

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 60}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_generating_queries=0,
        local_generating_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
    )

    # Three sessions dispatch and all enter generating phase
    for i in range(3):
        await _try_acquire_local_dispatch(
            srv, max_local=3, session_key=f"sess-{i}", backend="local"
        )
        await _increment_generating_only_slot(srv, session_key=f"sess-{i}")

    assert _get_generating_only_count(srv) == 3

    # Fourth session should be denied
    acquired, owner, _, _ = await _try_acquire_local_dispatch(
        srv, max_local=3, session_key="sess-4", backend="local"
    )
    assert acquired is False, (
        "sess-4 should be denied when all 3 slots are generating"
    )


# ---------------------------------------------------------------------------
# AC4: Lease mechanism preserved for dispatch ownership
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adaptive_lease_preserved_for_in_flight_prompts():
    """Adaptive prefill lease still applies for in-flight prompt tracking.

    The lease (60 + tokens*0.015 cap 1500s + prefill-progress) is for
    dispatch ownership tracking — it remains for in-flight prompts.
    The change is ONLY to what counts against the pool (generating-only).
    """
    from proxy.router_helpers import (
        _decrement_local_active_queries,
        _get_lease_timeout_seconds,
        _increment_local_active_queries,
    )

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 60}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
    )

    await _increment_local_active_queries(
        srv, session_key="sess-1", backend="local"
    )
    assert srv.local_dispatch_records["sess-1"]["active"] is True
    assert srv.local_dispatch_records["sess-1"]["expires_at"] > time.monotonic()

    # Stream ends — lease still persists for ownership (inactive hold)
    await _decrement_local_active_queries(srv, session_key="sess-1")
    assert "sess-1" in srv.local_dispatch_records
    assert srv.local_dispatch_records["sess-1"]["active"] is False
    assert srv.local_dispatch_records["sess-1"]["expires_at"] > time.monotonic()


@pytest.mark.asyncio
async def test_lease_timeout_config_respected():
    """Config-specified lease timeout should override the default."""
    from proxy.router_helpers import _get_lease_timeout_seconds

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 300}},
    )

    timeout = _get_lease_timeout_seconds(srv)
    assert timeout == 300.0


@pytest.mark.asyncio
async def test_default_lease_timeout_is_60_seconds():
    """Default lease timeout should be 60s."""
    from proxy.router_helpers import _get_lease_timeout_seconds

    srv = SimpleNamespace(
        config={"server": {}},
    )

    timeout = _get_lease_timeout_seconds(srv)
    assert timeout == 60.0


# ---------------------------------------------------------------------------
# Edge case: Rapid prefill/generate transitions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rapid_prefill_then_generate():
    """Handle rapid transition from prefill to generating for the same session.

    Even if prefill completes quickly (e.g., cached prompt), the session
    should correctly increment generating count when first byte arrives.
    """
    from proxy.router_helpers import (
        _decrement_generating_only_slot,
        _get_generating_only_count,
        _increment_generating_only_slot,
        _try_acquire_local_dispatch,
    )

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 60}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_generating_queries=0,
        local_generating_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
    )

    # Session dispatches (prefill)
    await _try_acquire_local_dispatch(
        srv, max_local=1, session_key="sess-1", backend="local"
    )
    assert _get_generating_only_count(srv) == 0

    # Prefill completes, first byte arrives
    await _increment_generating_only_slot(srv, session_key="sess-1")
    assert _get_generating_only_count(srv) == 1

    # Another session trying to acquire should be blocked
    acquired, owner, _, _ = await _try_acquire_local_dispatch(
        srv, max_local=1, session_key="sess-2", backend="local"
    )
    assert acquired is False
    assert owner == "sess-1"


@pytest.mark.asyncio
async def test_double_increment_generating_is_safe():
    """Incrementing generating count twice for same session should be safe (idempotent).

    In practice, duplicate first-byte events should not break the counter.
    """
    from proxy.router_helpers import (
        _decrement_generating_only_slot,
        _get_generating_only_count,
        _increment_generating_only_slot,
    )

    srv = SimpleNamespace(
        local_generating_queries=0,
        local_generating_queries_lock=asyncio.Lock(),
    )

    # Increment twice (should only count once)
    await _increment_generating_only_slot(srv, session_key="sess-1")
    await _increment_generating_only_slot(srv, session_key="sess-1")

    assert _get_generating_only_count(srv) == 1, (
        "Duplicate increment should be idempotent"
    )


@pytest.mark.asyncio
async def test_decrement_nonexistent_generating_is_safe():
    """Decrementing generating count for a session that never incremented should be safe."""
    from proxy.router_helpers import (
        _decrement_generating_only_slot,
        _get_generating_only_count,
    )

    srv = SimpleNamespace(
        local_generating_queries=0,
        local_generating_queries_lock=asyncio.Lock(),
    )

    # Decrement without increment — should not go negative
    await _decrement_generating_only_slot(srv, session_key="sess-1")
    assert _get_generating_only_count(srv) == 0, (
        "Counter should not go negative"
    )


# ---------------------------------------------------------------------------
# Integration: proxy_to_local with generating-only semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_to_local_allows_dispatch_when_generating_below_cap(monkeypatch):
    """proxy_to_local should allow dispatch when generating count < max_local.

    Even if local_active_queries > 0 (some sessions in prefill), if generating
    count < session_slot_pool_size, the dispatch gate should pass.
    """
    from proxy.router import proxy_to_local

    from proxy import server as srv

    monkeypatch.setattr(
        srv,
        "config",
        {
            "server": {
                "llama_server_port": 8080,
                "session_slot_pool_size": 1,
                "max_concurrent_queries": 16,
            }
        },
    )
    proc = MagicMock()
    proc.poll.return_value = None
    monkeypatch.setattr(srv, "llama_process", proc)
    monkeypatch.setattr(srv, "backend_ready", True)
    monkeypatch.setattr(srv, "current_model", "Qwen3")
    monkeypatch.setattr(srv, "active_queries", 0)
    monkeypatch.setattr(srv, "active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "local_active_queries", 1)  # sess-a dispatched
    monkeypatch.setattr(srv, "local_active_queries_lock", asyncio.Lock())
    # Generating count is 0 (sess-a is in prefill, not generating yet)
    monkeypatch.setattr(srv, "local_generating_queries", 0)
    monkeypatch.setattr(srv, "local_generating_queries_lock", asyncio.Lock())
    monkeypatch.setattr(
        srv,
        "local_dispatch_records",
        {
            "sess-a": {
                "backend": "local",
                "started_at": 1.0,
                "active": True,
                "expires_at": 10**12,
            }
        },
    )
    monkeypatch.setattr(srv, "local_dispatch_records_lock", asyncio.Lock())

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
                    "messages": [{"role": "user", "content": "hi"}],
                },
                "body_override": None,
            }
        ),
    )
    monkeypatch.setattr(router_mod, "_build_slot_context", lambda *_: (None, None, 3.0))
    monkeypatch.setattr(router_mod, "_resolve_slot_model_name", lambda model, *_: model)
    monkeypatch.setattr(router_mod, "_check_slot_availability", AsyncMock(return_value=None))

    # Mock the backend call so proxy_to_local proceeds past dispatch gate
    class _FakeBackendResponse:
        status_code = 200
        content = b'{"id":"chatcmpl-generating-only","choices":[{"message":{"role":"assistant","content":"hi"}}]}'
        headers = {"content-type": "application/json"}

    async def _fake_backend_call(*args, **kwargs):
        return _FakeBackendResponse()

    monkeypatch.setattr(router_mod, "_call_with_backend_retries", _fake_backend_call)
    monkeypatch.setattr(router_mod, "_call_with_empty_retry", _fake_backend_call)

    req = _DummyRequest(
        body=json.dumps(
            {
                "model": "plan",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            }
        ).encode("utf-8")
    )

    resp = await proxy_to_local(req, "v1/chat/completions")

    # With generating=0 < pool_size=1, sess-b should be allowed through
    assert resp.status_code == 200, (
        "sess-b should dispatch when sess-a is in prefill (generating=0 < pool_size=1)"
    )


@pytest.mark.asyncio
async def test_proxy_to_local_denies_when_generating_at_pool_cap(monkeypatch):
    """proxy_to_local should deny dispatch when generating count >= max_local.

    When a session is actively generating and holding a pool slot, new
    dispatches should be rejected with 503/no_slots_available.
    """
    from proxy.router import proxy_to_local

    from proxy import server as srv

    monkeypatch.setattr(
        srv,
        "config",
        {
            "server": {
                "llama_server_port": 8080,
                "session_slot_pool_size": 1,
                "max_concurrent_queries": 16,
            }
        },
    )
    proc = MagicMock()
    proc.poll.return_value = None
    monkeypatch.setattr(srv, "llama_process", proc)
    monkeypatch.setattr(srv, "backend_ready", True)
    monkeypatch.setattr(srv, "current_model", "Qwen3")
    monkeypatch.setattr(srv, "active_queries", 0)
    monkeypatch.setattr(srv, "active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "local_active_queries", 1)  # sess-a dispatched
    monkeypatch.setattr(srv, "local_active_queries_lock", asyncio.Lock())
    # Generating count is 1 (sess-a is actively generating)
    monkeypatch.setattr(srv, "local_generating_queries", 1)
    monkeypatch.setattr(srv, "local_generating_queries_lock", asyncio.Lock())
    monkeypatch.setattr(
        srv,
        "local_dispatch_records",
        {
            "sess-a": {
                "backend": "local",
                "started_at": 1.0,
                "active": True,
                "expires_at": 10**12,
            }
        },
    )
    monkeypatch.setattr(srv, "local_dispatch_records_lock", asyncio.Lock())

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
                    "messages": [{"role": "user", "content": "hi"}],
                },
                "body_override": None,
            }
        ),
    )
    monkeypatch.setattr(router_mod, "_build_slot_context", lambda *_: (None, None, 3.0))
    monkeypatch.setattr(router_mod, "_resolve_slot_model_name", lambda model, *_: model)
    monkeypatch.setattr(router_mod, "_check_slot_availability", AsyncMock(return_value=None))

    req = _DummyRequest(
        body=json.dumps(
            {
                "model": "plan",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            }
        ).encode("utf-8")
    )

    resp = await proxy_to_local(req, "v1/chat/completions")

    # With generating=1 >= pool_size=1, sess-b should be denied
    assert resp.status_code == 503
    payload = json.loads(resp.body)
    assert payload["error"]["code"] == "no_slots_available"
    assert payload["local_owner_session_id"] == "sess-a"
