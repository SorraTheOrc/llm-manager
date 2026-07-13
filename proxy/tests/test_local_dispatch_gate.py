import asyncio
import json
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


@pytest.mark.asyncio
async def test_proxy_to_local_rejects_when_local_dispatch_busy(monkeypatch):
    """proxy_to_local should return 503 when another local request is active."""
    from proxy import server as srv
    from proxy.router import proxy_to_local

    # Minimal server state
    monkeypatch.setattr(
        srv,
        "config",
        {
            "server": {
                "llama_server_port": 8080,
                "local_max_concurrent_queries": 1,
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
    monkeypatch.setattr(srv, "local_active_queries", 1)
    monkeypatch.setattr(srv, "local_active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(
        srv,
        "local_dispatch_records",
        {
            "owner-session": {
                "backend": "local",
                "started_at": 1.0,
                "active": True,
                "expires_at": 999999999.0,
            }
        },
    )
    monkeypatch.setattr(srv, "local_dispatch_records_lock", asyncio.Lock())

    # Avoid unrelated paths
    import proxy.router as router_mod

    monkeypatch.setattr(router_mod, "_is_self_healing_active", lambda: False)
    monkeypatch.setattr(
        router_mod,
        "_handle_session",
        AsyncMock(
            return_value={
                "session_id": "new-session",
                "session_id_header": "new-session",
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
    monkeypatch.setattr(router_mod, "_get_job_scheduler", lambda: None)
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

    assert resp.status_code == 503
    payload = json.loads(resp.body)
    assert payload["error"]["code"] == "no_slots_available"
    assert payload["local_owner_session_id"] == "owner-session"


@pytest.mark.asyncio
async def test_proxy_to_local_rejects_when_other_session_holds_unexpired_lease(monkeypatch):
    """No-preemption policy should reject non-owner even when no active request exists."""
    from proxy import server as srv
    from proxy.router import proxy_to_local

    monkeypatch.setattr(
        srv,
        "config",
        {
            "server": {
                "llama_server_port": 8080,
                "local_max_concurrent_queries": 1,
                "local_dispatch_lease_timeout_seconds": 180,
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
    monkeypatch.setattr(srv, "local_active_queries", 0)
    monkeypatch.setattr(srv, "local_active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(
        srv,
        "local_dispatch_records",
        {
            "owner-session": {
                "backend": "local",
                "started_at": 1.0,
                "active": False,
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
                "session_id": "new-session",
                "session_id_header": "new-session",
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
    monkeypatch.setattr(router_mod, "_get_job_scheduler", lambda: None)
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
    assert resp.status_code == 503
    payload = json.loads(resp.body)
    assert payload["local_owner_session_id"] == "owner-session"


@pytest.mark.asyncio
async def test_local_dispatch_tracking_helpers_keep_lease_between_turns():
    """Releasing active local request should keep an inactive lease for the owner session."""
    from proxy.router_helpers import _increment_local_active_queries, _decrement_local_active_queries

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 180}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
    )

    await _increment_local_active_queries(srv, session_key="sess-1", backend="local")
    assert srv.local_active_queries == 1
    assert "sess-1" in srv.local_dispatch_records
    assert srv.local_dispatch_records["sess-1"]["active"] is True

    await _decrement_local_active_queries(srv, session_key="sess-1")
    assert srv.local_active_queries == 0
    assert "sess-1" in srv.local_dispatch_records
    assert srv.local_dispatch_records["sess-1"]["active"] is False
    assert srv.local_dispatch_records["sess-1"]["expires_at"] > 0


@pytest.mark.asyncio
async def test_try_acquire_denies_non_owner_during_unexpired_lease():
    """No-preemption policy: non-owner cannot acquire local while owner lease is active."""
    from proxy.router_helpers import _try_acquire_local_dispatch

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 180}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={
            "sess-owner": {
                "backend": "local",
                "started_at": 1.0,
                "active": False,
                "expires_at": 10**12,
            }
        },
        local_dispatch_records_lock=asyncio.Lock(),
    )

    acquired, owner, active, retry_after = await _try_acquire_local_dispatch(
        srv,
        max_local=1,
        session_key="sess-other",
        backend="local",
    )

    assert acquired is False
    assert owner == "sess-owner"
    assert active == 0
    assert retry_after >= 1


@pytest.mark.asyncio
async def test_try_acquire_allows_new_owner_after_lease_expiry():
    """When prior lease expires, a different session can acquire local."""
    from proxy.router_helpers import _try_acquire_local_dispatch

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 180}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={
            "sess-owner": {
                "backend": "local",
                "started_at": 1.0,
                "active": False,
                "expires_at": 0.0,
            }
        },
        local_dispatch_records_lock=asyncio.Lock(),
    )

    acquired, owner, active, retry_after = await _try_acquire_local_dispatch(
        srv,
        max_local=1,
        session_key="sess-new",
        backend="local",
    )

    assert acquired is True
    assert owner is None
    assert active == 1
    assert retry_after >= 1
    assert "sess-new" in srv.local_dispatch_records
    assert srv.local_dispatch_records["sess-new"]["active"] is True


@pytest.mark.asyncio
async def test_cleanup_stale_preserves_expired_active_lease():
    """
    _cleanup_stale_local_dispatch must NOT remove active lease records
    even when their expires_at has passed.  Active records represent
    in-flight requests that may take longer than the 60s lease timeout.
    """
    from proxy.router_helpers import _cleanup_stale_local_dispatch

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 180}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={
            # Expired active lease — should NOT be cleaned (in-flight)
            "sess-stuck": {
                "backend": "local",
                "started_at": 1.0,
                "active": True,
                "expires_at": 0.0,  # expired
            },
            # Valid inactive lease that hasn't expired — should NOT be cleaned
            "sess-valid": {
                "backend": "local",
                "started_at": 1.0,
                "active": False,
                "expires_at": 10**12,  # far in the future
            },
            # Expired inactive lease — should be cleaned
            "sess-expired-inactive": {
                "backend": "local",
                "started_at": 1.0,
                "active": False,
                "expires_at": 0.0,
            },
        },
        local_dispatch_records_lock=asyncio.Lock(),
    )

    removed = await _cleanup_stale_local_dispatch(srv)

    # The expired active lease must be PRESERVED (in-flight request)
    assert "sess-stuck" in srv.local_dispatch_records, (
        "Expired active lease must be preserved (in-flight request)"
    )
    # The expired inactive lease should be removed
    assert "sess-expired-inactive" not in srv.local_dispatch_records
    # The valid future lease should still exist
    assert "sess-valid" in srv.local_dispatch_records
    assert removed == 1, "Expected 1 lease removed (expired-inactive only)"


@pytest.mark.asyncio
async def test_cleanup_stale_preserves_active_unexpired_lease():
    """
    _cleanup_stale_local_dispatch must NOT remove active lease records
    whose expires_at is still in the future (legitimate in-flight requests).
    """
    from proxy.router_helpers import _cleanup_stale_local_dispatch

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 180}},
        local_active_queries=1,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={
            "sess-active": {
                "backend": "local",
                "started_at": 100.0,
                "active": True,
                "expires_at": 10**12,  # far in the future
            },
        },
        local_dispatch_records_lock=asyncio.Lock(),
    )

    removed = await _cleanup_stale_local_dispatch(srv)

    assert "sess-active" in srv.local_dispatch_records, (
        "Active unexpired lease must be preserved"
    )
    assert removed == 0, "No leases should have been removed"
    assert srv.local_dispatch_records["sess-active"]["active"] is True


# ---------------------------------------------------------------------------
# Cross-session handoff tests (LP-0MRHV4UYE0013F6P)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reduced_default_timeout_is_60_seconds():
    """The default lease timeout should be 60s (reduced from 180s)."""
    from proxy.router_helpers import _get_lease_timeout_seconds

    srv = SimpleNamespace(
        config={"server": {}},
    )

    timeout = _get_lease_timeout_seconds(srv)
    assert timeout == 60.0, f"Expected 60.0, got {timeout}"


@pytest.mark.asyncio
async def test_configurable_timeout_still_respected():
    """A config-specified timeout should still override the default."""
    from proxy.router_helpers import _get_lease_timeout_seconds

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 300}},
    )

    timeout = _get_lease_timeout_seconds(srv)
    assert timeout == 300.0, f"Expected 300.0, got {timeout}"


@pytest.mark.asyncio
async def test_release_local_dispatch_on_session_eviction():
    """_release_local_dispatch should remove the dispatch record for an evicted session.

    This replicates the core action of the eviction callback — when a
    session is evicted from the session manager, its dispatch lease
    should be released so another session can acquire it.
    """
    from proxy.router_helpers import _release_local_dispatch, _increment_local_active_queries

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 180}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
        logger=MagicMock(),
    )

    # Create a dispatch record for session "sess-a"
    await _increment_local_active_queries(srv, session_key="sess-a", backend="local")
    assert "sess-a" in srv.local_dispatch_records

    # Simulate stream finish (mark inactive)
    from proxy.router_helpers import _decrement_local_active_queries
    await _decrement_local_active_queries(srv, session_key="sess-a")
    assert srv.local_dispatch_records["sess-a"]["active"] is False
    assert "sess-a" in srv.local_dispatch_records  # lease persists

    # Now release the lease (like the eviction callback would do)
    removed = await _release_local_dispatch(srv, "sess-a")
    assert removed is True
    assert "sess-a" not in srv.local_dispatch_records, (
        "Dispatch record should be removed on explicit release"
    )


@pytest.mark.asyncio
async def test_cross_session_handoff_after_lease_release():
    """After lease release, a different session can acquire the local backend.

    This simulates:
    1. Session A acquires the local backend and finishes.
    2. The lease is explicitly released (via eviction callback or /v1/leases/release).
    3. Session B can immediately acquire the local backend.
    """
    from proxy.router_helpers import (
        _try_acquire_local_dispatch,
        _increment_local_active_queries,
        _release_local_dispatch,
    )

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 180}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
        logger=MagicMock(),
    )

    # 1. Session A acquires the local backend
    await _increment_local_active_queries(srv, session_key="sess-a", backend="local")
    assert srv.local_active_queries == 1

    # 2. Session A finishes its stream (mark inactive, lease remains)
    from proxy.router_helpers import _decrement_local_active_queries
    await _decrement_local_active_queries(srv, session_key="sess-a")
    assert srv.local_active_queries == 0
    assert "sess-a" in srv.local_dispatch_records  # lease persists

    # 3. Before handoff: Session B tries to acquire — should be denied
    acquired_before, owner, _, _ = await _try_acquire_local_dispatch(
        srv, max_local=1, session_key="sess-b", backend="local",
    )
    assert acquired_before is False, (
        "Session B should be denied while Session A's lease is held"
    )
    assert owner == "sess-a"

    # 4. Release Sesion A's lease (simulating eviction callback)
    await _release_local_dispatch(srv, "sess-a")
    assert "sess-a" not in srv.local_dispatch_records

    # 5. After release: Session B tries to acquire — should succeed
    acquired_after, _, _, _ = await _try_acquire_local_dispatch(
        srv, max_local=1, session_key="sess-b", backend="local",
    )
    assert acquired_after is True, (
        "Session B should be able to acquire after Session A's lease is released"
    )


# ---------------------------------------------------------------------------
# N-aware dispatch lease tests (LP-0MRI8J7WR0035ZE1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_n2_allows_two_concurrent_sessions():
    """With N=2, two different sessions can concurrently hold active dispatch leases."""
    from proxy.router_helpers import _try_acquire_local_dispatch

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 180}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
    )

    # Session A acquires
    acquired_a, owner_a, _, _ = await _try_acquire_local_dispatch(
        srv, max_local=2, session_key="sess-a", backend="local",
    )
    assert acquired_a is True
    assert owner_a is None

    # Session B acquires (should succeed with N=2)
    acquired_b, owner_b, _, _ = await _try_acquire_local_dispatch(
        srv, max_local=2, session_key="sess-b", backend="local",
    )
    assert acquired_b is True, "Session B should be allowed with N=2"
    assert owner_b is None


@pytest.mark.asyncio
async def test_n2_blocks_third_session():
    """With N=2 and two active leases held, a third session is denied."""
    from proxy.router_helpers import _try_acquire_local_dispatch

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 180}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
    )

    # Session A acquires
    await _try_acquire_local_dispatch(srv, max_local=2, session_key="sess-a", backend="local")
    # Session B acquires
    await _try_acquire_local_dispatch(srv, max_local=2, session_key="sess-b", backend="local")

    # Session C tries - should be denied
    acquired_c, owner_c, _, _ = await _try_acquire_local_dispatch(
        srv, max_local=2, session_key="sess-c", backend="local",
    )
    assert acquired_c is False, "Session C should be denied when 2 slots occupied"
    assert owner_c in ("sess-a", "sess-b")


@pytest.mark.asyncio
async def test_n1_backward_compatible():
    """With N=1, behaviour is identical to the current single-session dispatch."""
    from proxy.router_helpers import _try_acquire_local_dispatch

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 180}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
    )

    # Session A acquires
    acquired_a, _, _, _ = await _try_acquire_local_dispatch(
        srv, max_local=1, session_key="sess-a", backend="local",
    )
    assert acquired_a is True

    # Session B tries - should be denied (backward compat)
    acquired_b, owner_b, _, _ = await _try_acquire_local_dispatch(
        srv, max_local=1, session_key="sess-b", backend="local",
    )
    assert acquired_b is False
    assert owner_b == "sess-a"


@pytest.mark.asyncio
async def test_n3_inactive_lease_reserves_slot():
    """No-preemption policy preserved: inactive leases reserve their slot.

    With N=3, session A has an inactive lease, sessions B and C fill
    the remaining slots, session D is blocked.
    """
    from proxy.router_helpers import _try_acquire_local_dispatch, _decrement_local_active_queries

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 180}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
    )

    # Session A acquires and finishes (inactive lease)
    await _try_acquire_local_dispatch(srv, max_local=3, session_key="sess-a", backend="local")
    await _decrement_local_active_queries(srv, session_key="sess-a")
    assert srv.local_dispatch_records["sess-a"]["active"] is False

    # Session B acquires (active)
    acquired_b, _, _, _ = await _try_acquire_local_dispatch(
        srv, max_local=3, session_key="sess-b", backend="local",
    )
    assert acquired_b is True

    # Session C acquires (active) - fills the third slot
    acquired_c, _, _, _ = await _try_acquire_local_dispatch(
        srv, max_local=3, session_key="sess-c", backend="local",
    )
    assert acquired_c is True

    # Session D tries - should be denied (all 3 slots occupied by A, B, C)
    acquired_d, owner_d, _, _ = await _try_acquire_local_dispatch(
        srv, max_local=3, session_key="sess-d", backend="local",
    )
    assert acquired_d is False, "Session D should be denied when all N=3 slots occupied"
    assert owner_d is not None


@pytest.mark.asyncio
async def test_expired_lease_frees_slot():
    """Expired leases are cleaned and their slots become available for other sessions."""
    from proxy.router_helpers import _try_acquire_local_dispatch

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 180}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={
            "sess-expired": {
                "backend": "local",
                "started_at": 1.0,
                "active": False,
                "expires_at": 0.0,  # already expired
            },
        },
        local_dispatch_records_lock=asyncio.Lock(),
    )

    acquired, owner, _, _ = await _try_acquire_local_dispatch(
        srv, max_local=1, session_key="sess-new", backend="local",
    )
    assert acquired is True, "New session should acquire after expired lease is cleaned"
    assert "sess-expired" not in srv.local_dispatch_records, (
        "Expired lease should have been removed"
    )
    assert "sess-new" in srv.local_dispatch_records


@pytest.mark.asyncio
async def test_get_local_max_concurrent_queries_reads_session_slot_pool_size():
    """_get_local_max_concurrent_queries reads from session_slot_pool_size."""
    from proxy.router import _get_local_max_concurrent_queries

    # session_slot_pool_size should take precedence
    result = _get_local_max_concurrent_queries({
        "session_slot_pool_size": 3,
        "local_max_concurrent_queries": 1,
    })
    assert result == 3, f"Expected 3, got {result}"

    # without session_slot_pool_size, fall back to local_max_concurrent_queries
    result = _get_local_max_concurrent_queries({
        "local_max_concurrent_queries": 2,
    })
    assert result == 2, f"Expected 2, got {result}"

    # with neither, default to 1
    result = _get_local_max_concurrent_queries({})
    assert result == 1, f"Expected 1, got {result}"

    # session_slot_pool_size=1 should still work
    result = _get_local_max_concurrent_queries({
        "session_slot_pool_size": 1,
    })
    assert result == 1, f"Expected 1, got {result}"


# ---------------------------------------------------------------------------
# N=2 concurrent dispatch integration tests (LP-0MRI8JB6W0022XT9)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_n2_integration_third_session_blocked_via_proxy_to_local(monkeypatch):
    """With N=2 and two active sessions, a third is blocked via proxy_to_local."""
    from proxy import server as srv
    from proxy.router import proxy_to_local, _get_local_max_concurrent_queries

    # Config: N=2 via session_slot_pool_size (also set local_max_concurrent_queries
    # for backward compat so tests work with both old and new code)
    monkeypatch.setattr(
        srv,
        "config",
        {
            "server": {
                "llama_server_port": 8080,
                "session_slot_pool_size": 2,
                "local_max_concurrent_queries": 2,
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
    # Simulate 2 active local queries (sess-a and sess-b)
    monkeypatch.setattr(srv, "local_active_queries", 2)
    monkeypatch.setattr(srv, "local_active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(
        srv,
        "local_dispatch_records",
        {
            "sess-a": {
                "backend": "local",
                "started_at": 1.0,
                "active": True,
                "expires_at": 10**12,
            },
            "sess-b": {
                "backend": "local",
                "started_at": 2.0,
                "active": True,
                "expires_at": 10**12,
            },
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
                "session_id": "sess-c",
                "session_id_header": "sess-c",
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
    monkeypatch.setattr(router_mod, "_get_job_scheduler", lambda: None)
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

    assert resp.status_code == 503
    payload = json.loads(resp.body)
    assert payload["error"]["code"] == "no_slots_available"
    assert payload["local_owner_session_id"] in ("sess-a", "sess-b")


@pytest.mark.asyncio
async def test_n2_integration_release_then_retry(monkeypatch):
    """After one session completes, a blocked session can acquire on retry."""
    from proxy import server as srv
    from proxy.router import proxy_to_local, _get_local_max_concurrent_queries

    monkeypatch.setattr(
        srv,
        "config",
        {
            "server": {
                "llama_server_port": 8080,
                "session_slot_pool_size": 2,
                "local_max_concurrent_queries": 2,
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
    # Simulate 1 active local query (sess-a) after sess-b's lease was released
    monkeypatch.setattr(srv, "local_active_queries", 1)
    monkeypatch.setattr(srv, "local_active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(
        srv,
        "local_dispatch_records",
        {
            "sess-a": {
                "backend": "local",
                "started_at": 1.0,
                "active": True,
                "expires_at": 10**12,
            },
            # sess-b's lease has been released
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
                "session_id": "sess-c",
                "session_id_header": "sess-c",
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
    monkeypatch.setattr(router_mod, "_get_job_scheduler", lambda: None)
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

    # With only 1 active lease (sess-a), sess-c should be allowed through
    # the dispatch gate. Note: this test verifies dispatch gate passes, but
    # the actual backend call may fail — we only check that it's NOT a 503
    # from the dispatch gate.
    assert resp.status_code != 503, (
        "Session should not receive 503 when only 1 of N=2 slots is occupied"
    )


@pytest.mark.asyncio
async def test_n1_backward_compat_integration(monkeypatch):
    """N=1 backward compat: single session proceeds, second session blocked."""
    from proxy import server as srv
    from proxy.router import proxy_to_local

    monkeypatch.setattr(
        srv,
        "config",
        {
            "server": {
                "llama_server_port": 8080,
                "session_slot_pool_size": 1,
                "local_max_concurrent_queries": 1,
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
    monkeypatch.setattr(srv, "local_active_queries", 1)
    monkeypatch.setattr(srv, "local_active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(
        srv,
        "local_dispatch_records",
        {
            "sess-a": {
                "backend": "local",
                "started_at": 1.0,
                "active": True,
                "expires_at": 10**12,
            },
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
    monkeypatch.setattr(router_mod, "_get_job_scheduler", lambda: None)
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

    assert resp.status_code == 503
    payload = json.loads(resp.body)
    assert payload["error"]["code"] == "no_slots_available"
    assert payload["local_owner_session_id"] == "sess-a"
