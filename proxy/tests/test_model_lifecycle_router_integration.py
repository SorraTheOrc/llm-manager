import json
import pytest
import httpx
from unittest.mock import MagicMock, AsyncMock

import proxy.server as server

pytestmark = pytest.mark.refactor_parity


@pytest.fixture(autouse=True)
def reset_server_state(monkeypatch):
    """Reset mutable global server state between tests.

    Prevents test-ordering pollution from preceding tests that directly
    modify shared module-level state (e.g. current_model, llama_process)
    without restoring it.
    """
    monkeypatch.setattr(server, 'background_loads', {})
    monkeypatch.setattr(server, 'model_last_used', {})
    monkeypatch.setattr(server, 'llama_process', None)
    monkeypatch.setattr(server, '_http_client', None)
    monkeypatch.setattr(server, '_remote_http_client', None)
    server.current_model = None
    server.last_start_failure = None
    server.backend_ready = False
    try:
        server.model_switch_refcount = 0
    except Exception:
        pass
    yield


@pytest.mark.asyncio
@pytest.mark.slow
async def test_stub_router_integration(monkeypatch):
    """Minimal stub-based integration-style test that emulates router endpoints using httpx.MockTransport.

    This test verifies that ensure_model_loaded interacts with router endpoints
    (/models/load and /models) to load and observe a model in router-mode.
    """
    mock_config = {
        "models": {
            "embed": {
                "aliases": ["embeddings", "embed"],
                "llama_model": "mxbai-embed",
                "type": "local",
            }
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

    monkeypatch.setattr(server, 'config', mock_config)

    loaded = set()

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        if path == '/health' and method == 'GET':
            return httpx.Response(200)
        if path == '/models/load' and method == 'POST':
            try:
                payload = json.loads(request.content.decode('utf-8') if request.content else '{}')
            except Exception:
                payload = {}
            model = payload.get('model')
            if model:
                loaded.add(model)
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(400, json={"error": "missing model"})
        if path == '/models' and method == 'GET':
            return httpx.Response(
                200,
                json={"data": [{"id": m, "status": {"value": "loaded"}} for m in loaded]},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(server, '_http_client', client)

    # Simulate starting a router-mode llama-server process
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    monkeypatch.setattr(server, 'start_llama_server', lambda model, **kwargs: fake_proc)
    monkeypatch.setattr(server, 'wait_for_llama_server', AsyncMock(return_value=True))

    # Ensure initial state
    server.current_model = None
    monkeypatch.setattr(server, 'llama_process', None)

    ok = await server.ensure_model_loaded('embed')

    # Cleanup
    await client.aclose()

    assert ok is True
    assert server.current_model == 'mxbai-embed'
    assert 'mxbai-embed' in loaded


@pytest.mark.asyncio
async def test_stub_router_fixture_resets_polluted_state(monkeypatch):
    """Regression test: verify the reset_server_state fixture handles polluted state.

    Even when preceding tests (e.g. test_switching_sse.py, test_alias_plan_code_integration)
    leave shared server state dirty, the autouse fixture restores it so
    ensure_model_loaded works reliably.
    """
    from proxy import server as srv

    # Verify fixture reset_server_state already cleared these.
    # Without the fixture, these would be polluted by preceding tests.
    assert srv.current_model is None
    assert srv.llama_process is None
    assert srv.last_start_failure is None
    assert srv.backend_ready is False
    assert srv._http_client is None
    assert srv._remote_http_client is None
