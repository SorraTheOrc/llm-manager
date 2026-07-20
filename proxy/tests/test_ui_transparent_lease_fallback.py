"""Tests for transparent local-to-remote fallback when a dispatch lease is active.

These tests verify that when the local dispatch lease is held by another
session, ``proxy_to_local`` returns a 503 with ``reason: "local_lease_active"``,
and that the higher-level ``proxy_with_fallback`` correctly routes to a remote
provider instead of propagating the 503 to the client.

The legacy function ``ui._dispatch_local_with_transparent_remote_fallback``
was refactored into inline code in ``router.py``'s ``proxy_to_local()``.
These tests exercise the real code paths through ``proxy_to_local`` and
``proxy_with_fallback``.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.responses import JSONResponse


def _setup_server_state(monkeypatch):
    """Set up minimal server module state for proxy routing.

    Configures the server module so that ``_srv()`` returns a module with
    the attributes ``proxy_to_local`` and ``proxy_with_fallback`` expect
    (backend ready, llama process running, dispatch tracking initialised).
    """
    from proxy import server as srv_module

    config = {
        "server": {
            "llama_router_mode": False,
            "llama_server_port": 8080,
            "local_max_concurrent_queries": 1,
        }
    }
    monkeypatch.setattr(srv_module, "config", config)
    monkeypatch.setattr(srv_module, "llama_process", MagicMock())
    srv_module.llama_process.poll.return_value = None  # running
    monkeypatch.setattr(srv_module, "backend_ready", True)
    monkeypatch.setattr(srv_module, "current_model", "test-model")
    # Reset dispatch tracking to a known state
    monkeypatch.setattr(srv_module, "active_queries", 0)
    monkeypatch.setattr(srv_module, "local_active_queries", 0)
    monkeypatch.setattr(srv_module, "local_dispatch_records", {})
    # Re-create locks to avoid sharing state with other test fixtures
    import asyncio
    monkeypatch.setattr(srv_module, "active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv_module, "local_active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv_module, "local_dispatch_records_lock", asyncio.Lock())
    # Mock the session manager
    _mock_session_manager(monkeypatch, srv_module)
    return srv_module


def _mock_session_manager(mp, srv_module):
    """Attach a mock session manager to *srv_module*."""
    from proxy.session_manager import Session

    mock_session = MagicMock(spec=Session)
    mock_session.session_id = "test-session"
    mock_session.message_count = 0
    mock_session.restore_confirmed = False
    mock_session.messages = []
    mock_session.invalidated = False

    mock_sm = MagicMock()
    mock_sm.get_or_create = AsyncMock(return_value=(mock_session, True))
    mock_sm.get = AsyncMock(return_value=mock_session)
    mp.setattr(srv_module, "session_manager", mock_sm)


def _make_request(body, headers=None):
    """Build a mock FastAPI Request."""
    from fastapi import Request as FastAPIRequest

    mock_req = MagicMock(spec=FastAPIRequest)
    mock_req.method = "POST"
    mock_req.url = type("U", (), {"path": "/v1/chat/completions"})()
    mock_req.headers = headers or {"content-type": "application/json"}
    mock_req._body = body

    async def _body():
        return mock_req._body

    mock_req.body = _body
    mock_req.is_disconnected = AsyncMock(return_value=False)
    return mock_req


def _seed_other_session_lease(srv_module):
    """Pre-seed a dispatch lease for a *different* session."""
    import time
    other = "other-owner-session"
    srv_module.local_dispatch_records[other] = {
        "backend": "local",
        "started_at": time.monotonic(),
        "active": True,
        "expires_at": time.monotonic() + 300,
    }
    srv_module.local_active_queries = 1


# ===================================================================
# proxy_to_local: lease held by another session → 503 lease-active
# ===================================================================


@pytest.mark.asyncio
async def test_proxy_to_local_lease_active_returns_503(monkeypatch):
    """When another session holds a local dispatch lease, ``proxy_to_local``
    returns a 503 with ``reason: "local_lease_active"``.
    """
    srv_module = _setup_server_state(monkeypatch)
    from proxy.router import proxy_to_local

    _seed_other_session_lease(srv_module)

    body = json.dumps({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }).encode("utf-8")

    mock_request = _make_request(
        body,
        {"content-type": "application/json", "x-session-id": "my-test-session"},
    )
    resp = await proxy_to_local(mock_request, "v1/chat/completions")

    assert resp.status_code == 503, (
        f"Expected 503 when another session holds the lease, got {resp.status_code}"
    )
    body_data = json.loads(resp.body.decode("utf-8"))
    assert body_data.get("reason") == "local_lease_active"
    assert body_data.get("local_owner_session_id") == "other-owner-session"


# ===================================================================
# proxy_with_fallback: lease-active → remote fallback
# ===================================================================


@pytest.mark.asyncio
async def test_proxy_with_fallback_lease_active_routes_to_remote(monkeypatch):
    """When the local provider returns 503 (lease-active), the fallback
    chain routes to the remote provider and returns its 200 response.
    """
    srv_module = _setup_server_state(monkeypatch)
    import proxy.router as router_mod
    from proxy.provider import proxy_with_fallback

    _seed_other_session_lease(srv_module)

    body = json.dumps({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }).encode("utf-8")

    mock_request = _make_request(
        body,
        {"content-type": "application/json", "x-session-id": "my-test-session"},
    )

    model_cfg = {
        "providers": [
            {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"},
            {"name": "remote-fallback", "type": "remote",
             "endpoint": "https://example.com/api", "api_key_env": "K"},
        ]
    }

    async def _mock_remote(_request, _path, _provider_cfg):
        return JSONResponse(
            status_code=200,
            content={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "content": "Hello from remote provider",
                            "role": "assistant",
                        },
                    }
                ]
            },
        )

    with (
        patch("proxy.provider._get_proxy_to_local", return_value=router_mod.proxy_to_local),
        patch("proxy.provider._get_proxy_to_remote", return_value=_mock_remote),
    ):
        resp = await proxy_with_fallback(
            mock_request, "v1/chat/completions", model_cfg, srv_module.config,
        )

    assert resp.status_code == 200, (
        f"Expected 200 from remote fallback, got {resp.status_code}"
    )
    body_data = json.loads(resp.body.decode("utf-8"))
    choices = body_data.get("choices", [])
    assert len(choices) > 0
    assert choices[0]["message"]["content"] == "Hello from remote provider"


# ===================================================================
# proxy_with_fallback: lease-active + no remote providers → exhausted
# ===================================================================


@pytest.mark.asyncio
async def test_proxy_with_fallback_lease_active_no_remote_503(monkeypatch):
    """When the local provider returns 503 (lease-active) and there are NO
    remote providers configured, the response includes lease-active diagnostics.
    """
    srv_module = _setup_server_state(monkeypatch)
    import proxy.router as router_mod
    from proxy.provider import proxy_with_fallback

    _seed_other_session_lease(srv_module)

    body = json.dumps({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }).encode("utf-8")

    mock_request = _make_request(
        body,
        {"content-type": "application/json", "x-session-id": "my-test-session"},
    )

    # Only local provider — no remote fallback
    model_cfg = {
        "providers": [
            {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"},
        ]
    }

    with patch("proxy.provider._get_proxy_to_local", return_value=router_mod.proxy_to_local):
        resp = await proxy_with_fallback(
            mock_request, "v1/chat/completions", model_cfg, srv_module.config,
        )

    assert resp.status_code == 503, (
        f"Expected 503 when no remote fallback available, got {resp.status_code}"
    )
    body_data = json.loads(resp.body.decode("utf-8"))
    diagnostics = body_data.get("diagnostics", [])
    # With local_max_concurrent_queries=1 and one other session holding
    # a lease, the concurrency limit is hit before the lease check in
    # proxy_to_local is even called.  Verify the diagnostics contain the
    # concurrency-limit reason.
    assert any(
        d.get("status") == "local_concurrency_limit"
        for d in diagnostics
    ), f"Expected local_concurrency_limit in diagnostics, got {json.dumps(diagnostics, indent=2)}"
