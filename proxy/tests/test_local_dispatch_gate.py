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
