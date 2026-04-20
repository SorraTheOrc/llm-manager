import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Response, HTTPException

import proxy.server as server


class DummyRequest:
    def __init__(self, body: bytes):
        self._body = body
        self.headers = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/embeddings"})

    async def body(self):
        return self._body


@pytest.fixture(autouse=True)
def reset_server_state(monkeypatch):
    """Reset mutable global server state between tests."""
    # Replace dicts with fresh instances to avoid cross-test leakage
    monkeypatch.setattr(server, 'background_loads', {})
    monkeypatch.setattr(server, 'model_last_used', {})
    # Ensure no running process is visible
    monkeypatch.setattr(server, 'llama_process', None)
    server.current_model = None
    server.last_start_failure = None
    # Reset numeric refcount
    try:
        server.model_switch_refcount = 0
    except Exception:
        pass
    yield


@pytest.fixture
def mock_config():
    return {
        "models": {
            "embed": {
                "aliases": ["embeddings", "embed"],
                "llama_model": "mxbai-embed",
                "type": "local",
            },
            "gemma4": {
                "aliases": ["gemma4"],
                "llama_model": "gemma4",
                "type": "local",
            },
        },
        "server": {
            "llama_router_mode": True,
            "embeddings_model": "mxbai-embed",
            "llama_server_port": 8080,
            "llama_embed_load_timeout": 5,
            "llama_model_load_timeout": 10,
            "llama_startup_timeout": 5,
        },
    }


@pytest.mark.asyncio
async def test_ensure_model_loaded_router_success(monkeypatch, mock_config):
    """ensure_model_loaded should start router (if needed) and load the model."""
    monkeypatch.setattr(server, 'config', mock_config)

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    monkeypatch.setattr(server, 'start_llama_server', lambda model: fake_proc)
    monkeypatch.setattr(server, 'wait_for_llama_server', AsyncMock(return_value=True))
    monkeypatch.setattr(server, 'router_load_model', AsyncMock(return_value=True))
    monkeypatch.setattr(server, 'router_wait_for_model', AsyncMock(return_value=True))

    # Pre-conditions
    server.current_model = None
    monkeypatch.setattr(server, 'llama_process', None)

    ok = await server.ensure_model_loaded('embed')
    assert ok is True
    assert server.current_model == 'mxbai-embed'


@pytest.mark.asyncio
async def test_create_embeddings_schedules_background_load_when_not_loaded(monkeypatch, mock_config):
    """create_embeddings should schedule a background load and return 503 when model not loaded."""
    monkeypatch.setattr(server, 'config', mock_config)
    monkeypatch.setattr(server, 'current_model', None)

    # router reports model not present
    monkeypatch.setattr(server, 'router_is_model_loaded', AsyncMock(return_value=False))

    scheduled = {}

    def fake_schedule(model_name: str) -> bool:
        scheduled['model'] = model_name
        return True

    monkeypatch.setattr(server, 'schedule_background_load', fake_schedule)

    body = json.dumps({"model": "embeddings", "input": "hello"}).encode('utf-8')
    req = DummyRequest(body)

    with pytest.raises(HTTPException) as excinfo:
        await server.create_embeddings(req)

    assert excinfo.value.status_code == 503
    assert scheduled.get('model') in ("embeddings", "mxbai-embed")


@pytest.mark.asyncio
async def test_create_embeddings_serves_when_router_reports_model_loaded(monkeypatch, mock_config):
    """When router reports the model loaded, create_embeddings should serve immediately via proxy_to_local."""
    monkeypatch.setattr(server, 'config', mock_config)
    # Simulate a running local process (llama_process) so proxy_to_local is reachable
    monkeypatch.setattr(server, 'llama_process', MagicMock())

    monkeypatch.setattr(server, 'router_is_model_loaded', AsyncMock(return_value=True))

    async def fake_proxy_to_local(request, path: str) -> Response:
        return Response(content=b'OK', status_code=200)

    monkeypatch.setattr(server, 'proxy_to_local', fake_proxy_to_local)

    body = json.dumps({"model": "embeddings", "input": "hello"}).encode('utf-8')
    req = DummyRequest(body)
    resp = await server.create_embeddings(req)

    assert isinstance(resp, Response)
    assert resp.status_code == 200
    assert server.current_model == 'mxbai-embed'


@pytest.mark.asyncio
async def test_failed_load_leaves_stable_error_state(monkeypatch, mock_config):
    """When router load fails the function should return False and broadcast an error."""
    monkeypatch.setattr(server, 'config', mock_config)

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    monkeypatch.setattr(server, 'start_llama_server', lambda model: fake_proc)
    monkeypatch.setattr(server, 'wait_for_llama_server', AsyncMock(return_value=True))
    monkeypatch.setattr(server, 'router_load_model', AsyncMock(return_value=False))

    events = []

    async def fake_broadcast_status(event_type: str, data: dict):
        events.append((event_type, data))

    monkeypatch.setattr(server, 'broadcast_status', fake_broadcast_status)

    res = await server.ensure_model_loaded('embed')
    assert res is False
    assert server.current_model is None
    assert any(e[0] == 'error' for e in events)
