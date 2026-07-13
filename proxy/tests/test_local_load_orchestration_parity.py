"""Parity tests for shared local-model load orchestration.

Verifies that the extracted ``_dispatch_local_model_load`` helper produces
equivalent behavior for both embeddings and chat/completions endpoints,
and that endpoint-specific logic (input validation for embeddings, system
prompt composition for chat) is preserved.

See work item LP-0MR6Y0VKF0068KK1.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import pytest
from fastapi import Response, HTTPException
from starlette.responses import JSONResponse

import proxy.server as server
import proxy.ui as ui
from proxy.lifecycle import _model_loading_response as _real_model_loading_response
from proxy.lifecycle import get_local_model_name as _real_get_local_model_name
from proxy.ui import _dispatch_local_model_load

pytestmark = pytest.mark.refactor_parity


class DummyRequest:
    """Minimal Request-like object for test injection."""

    def __init__(self, body: bytes, headers=None):
        self._body = body
        self.headers = headers or {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/embeddings"})()

    async def body(self):
        return self._body


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_server_state(monkeypatch):
    """Reset mutable global server state between tests."""
    monkeypatch.setattr(server, "background_loads", {})
    monkeypatch.setattr(server, "model_last_used", {})
    monkeypatch.setattr(server, "llama_process", None)
    server.current_model = None
    server.last_start_failure = None
    # Restore functions that may have been replaced by other tests
    server.get_local_model_name = _real_get_local_model_name
    server._model_loading_response = _real_model_loading_response
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
                "providers": [
                    {"name": "local-embed", "type": "local", "llama_model": "mxbai-embed"}
                ],
            },
            "gemma4": {
                "aliases": ["gemma4"],
                "providers": [
                    {"name": "local-gemma4", "type": "local", "llama_model": "gemma4"}
                ],
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


@pytest.fixture
def mock_srv(monkeypatch, mock_config):
    """Create a mock server module with essential lifecycle methods."""
    monkeypatch.setattr(server, "config", mock_config)

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    monkeypatch.setattr(server, "start_llama_server", lambda model: fake_proc)
    monkeypatch.setattr(server, "wait_for_llama_server", AsyncMock(return_value=True))
    monkeypatch.setattr(server, "router_load_model", AsyncMock(return_value=True))
    monkeypatch.setattr(server, "router_wait_for_model", AsyncMock(return_value=True))
    monkeypatch.setattr(server, "router_is_model_loaded", AsyncMock(return_value=True))
    monkeypatch.setattr(server, "router_list_models", AsyncMock(return_value=[]))
    monkeypatch.setattr(server, "schedule_background_load", lambda model: True)
    monkeypatch.setattr(server, "ensure_model_loaded", AsyncMock(return_value=True))

    server.logger = MagicMock()
    server.proxy_to_local = AsyncMock(return_value=Response("local", status_code=200))
    server.proxy_to_remote = AsyncMock(return_value=Response("remote", status_code=200))

    return server


# ── Helper to call create_embeddings / _do_proxy_openai_api ───────────────

def _patch_srv(monkeypatch, mock_srv):
    """Patch both ui._srv and lifecycle._srv to return mock_srv.

    Without this, ``get_model_config`` and ``_model_loading_response``
    (both in lifecycle.py) use their own ``_srv()`` which imports a
    potentially unpatched ``proxy.server`` module, leading to state
    leakage across tests.
    """
    import proxy.lifecycle as lifecycle
    monkeypatch.setattr(ui, "_srv", lambda: mock_srv)
    monkeypatch.setattr(lifecycle, "_srv", lambda: mock_srv)


async def _call_create_embeddings(monkeypatch, body_dict, mock_srv):
    """Call create_embeddings with a JSON body dict."""
    _patch_srv(monkeypatch, mock_srv)
    request = DummyRequest(json.dumps(body_dict).encode())
    return await ui.create_embeddings(request)


async def _call_chat(monkeypatch, body_dict, path, mock_srv):
    """Call _do_proxy_openai_api with a JSON body dict."""
    _patch_srv(monkeypatch, mock_srv)
    body = json.dumps(body_dict).encode()
    return await ui._do_proxy_openai_api(
        DummyRequest(body),
        path,
        body,
        mock_srv,
    )


# ── Tests: Endpoint-specific logic preserved ─────────────────────────────

class TestEndpointSpecificLogic:
    """Endpoint-specific validation/composition must remain unchanged."""

    @pytest.mark.asyncio
    async def test_create_embeddings_validates_input_required(self, monkeypatch, mock_srv):
        """Embeddings endpoint rejects requests missing 'input'."""
        monkeypatch.setattr(ui, "_srv", lambda: mock_srv)
        body = json.dumps({"model": "embed"}).encode()
        with pytest.raises(HTTPException) as exc:
            await ui.create_embeddings(DummyRequest(body))
        assert exc.value.status_code == 400
        assert "input" in exc.value.detail

    @pytest.mark.asyncio
    async def test_create_embeddings_validates_input_type(self, monkeypatch, mock_srv):
        """Embeddings endpoint rejects non-string/non-array input."""
        monkeypatch.setattr(ui, "_srv", lambda: mock_srv)
        body = json.dumps({"model": "embed", "input": 42}).encode()
        with pytest.raises(HTTPException) as exc:
            await ui.create_embeddings(DummyRequest(body))
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_create_embeddings_validates_empty_input_array(self, monkeypatch, mock_srv):
        """Embeddings endpoint rejects empty input array."""
        monkeypatch.setattr(ui, "_srv", lambda: mock_srv)
        body = json.dumps({"model": "embed", "input": []}).encode()
        with pytest.raises(HTTPException) as exc:
            await ui.create_embeddings(DummyRequest(body))
        assert exc.value.status_code == 400


# ── Tests: Shared orchestration parity between endpoints ─────────────────

class TestLocalLoadOrchestrationParity:
    """Behavioral parity between embeddings and chat local-model dispatch."""

    @pytest.mark.asyncio
    async def test_active_model_fast_path_parity(self, monkeypatch, mock_srv):
        """Both endpoints serve immediately when model is already active."""
        mock_srv.current_model = "mxbai-embed"
        mock_srv.llama_process = MagicMock()
        mock_srv.llama_process.poll.return_value = None

        # Embeddings
        body_emb = {"model": "embed", "input": "test text"}
        resp_emb = await _call_create_embeddings(monkeypatch, body_emb, mock_srv)
        assert resp_emb.status_code == 200

        # Chat
        body_chat = {"model": "embed", "messages": [{"role": "user", "content": "hi"}]}
        resp_chat = await _call_chat(monkeypatch, body_chat, "chat/completions", mock_srv)
        assert resp_chat.status_code == 200

    @pytest.mark.asyncio
    async def test_model_loading_response_shape_parity(self, monkeypatch, mock_srv):
        """Both endpoints return equivalent model_loading response when model not ready.

        Tests ``_dispatch_local_model_load`` directly to avoid state leakage
        from handler-level routing (``get_model_config``, system prompt
        composition).
        """
        mock_srv.current_model = None
        mock_srv.llama_process = None
        mock_srv.router_is_model_loaded = AsyncMock(return_value=False)
        mock_srv.router_load_model = AsyncMock(return_value=False)
        import proxy.lifecycle as lifecycle
        monkeypatch.setattr(lifecycle, "_srv", lambda: mock_srv)

        model_cfg = {
            "aliases": ["embeddings", "embed"],
            "providers": [
                {"name": "local-embed", "type": "local", "llama_model": "mxbai-embed"}
            ],
        }
        request = DummyRequest(json.dumps({"model": "embed", "input": "test"}).encode())

        with patch("proxy.router._get_job_scheduler", return_value=None):
            with patch("proxy.session._build_slot_context", return_value=(None, "", None)):
                with patch("proxy.session._resolve_session_id_header", return_value=(None, {})):

                    # Embeddings via shared helper (no grace window)
                    resp_emb = await _dispatch_local_model_load(
                        request, mock_srv, model_cfg, "embed", "v1/embeddings",
                        enable_grace_window=False,
                    )
                    assert resp_emb.status_code == 503
                    emb_content = json.loads(resp_emb.body)

                    # Chat via shared helper (with grace window)
                    resp_chat = await _dispatch_local_model_load(
                        request, mock_srv, model_cfg, "embed", "v1/chat/completions",
                        enable_grace_window=True,
                    )
                    assert resp_chat.status_code == 503
                    chat_content = json.loads(resp_chat.body)

        # Both should have the same structure
        for key in ("error", "status", "target_model", "scheduled", "current_model"):
            assert key in emb_content, f"Embeddings response missing key: {key}"
            assert key in chat_content, f"Chat response missing key: {key}"

        # Endpoint field differs by design
        assert emb_content["endpoint"] == "/v1/embeddings"
        assert chat_content["endpoint"] == "/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_remote_fallback_before_model_loading_parity(self, monkeypatch, mock_srv):
        """Both endpoints attempt remote fallback before returning model_loading."""
        mock_srv.current_model = None
        mock_srv.llama_process = None
        mock_srv.router_is_model_loaded = AsyncMock(return_value=False)

        mock_remote_cfg = {
            "embed": {
                "aliases": ["embed"],
                "providers": [
                    {"name": "remote-openai", "type": "remote", "endpoint": "https://api.openai.com/v1"},
                ],
            },
        }
        monkeypatch.setattr(mock_srv, "config", {
            "models": mock_remote_cfg,
            "server": {"llama_router_mode": False},
        })

        # Mock proxy_with_remote_fallback to return an error, forcing model_loading
        async def remote_fallback_error(*args, **kwargs):
            return Response("error", status_code=503)

        import proxy.provider
        monkeypatch.setattr(proxy.provider, "proxy_with_remote_fallback", remote_fallback_error)

        with patch("proxy.router._get_job_scheduler", return_value=None):
            with patch("proxy.session._build_slot_context", return_value=(None, "", None)):
                with patch("proxy.session._resolve_session_id_header", return_value=(None, {})):

                    # Embeddings
                    body_emb = {"model": "embed", "input": "test"}
                    resp_emb = await _call_create_embeddings(monkeypatch, body_emb, mock_srv)
                    assert resp_emb.status_code == 503

                    # Chat
                    body_chat = {"model": "embed", "messages": [{"role": "user", "content": "hi"}]}
                    resp_chat = await _call_chat(monkeypatch, body_chat, "chat/completions", mock_srv)
                    assert resp_chat.status_code == 503

    @pytest.mark.asyncio
    async def test_success_path_when_router_reports_loaded_parity(self, monkeypatch, mock_srv):
        """Both endpoints serve directly when router indicates model is loaded."""
        mock_srv.current_model = None
        mock_srv.llama_process = None
        mock_srv.router_is_model_loaded = AsyncMock(return_value=True)

        # Embeddings
        body_emb = {"model": "embed", "input": "test"}
        resp_emb = await _call_create_embeddings(monkeypatch, body_emb, mock_srv)
        assert resp_emb.status_code == 200

        # Chat
        mock_srv.router_is_model_loaded = AsyncMock(return_value=True)
        body_chat = {"model": "embed", "messages": [{"role": "user", "content": "hi"}]}
        resp_chat = await _call_chat(monkeypatch, body_chat, "chat/completions", mock_srv)
        assert resp_chat.status_code == 200

    @pytest.mark.asyncio
    async def test_scheduler_reenter_heuristic_parity(self, monkeypatch, mock_srv):
        """Both endpoints return model_loading when scheduler detects existing slot."""
        mock_srv.current_model = None
        mock_srv.llama_process = None
        mock_srv.router_is_model_loaded = AsyncMock(return_value=False)
        # Prevent grace window from succeeding
        mock_srv.router_load_model = AsyncMock(return_value=False)

        # Mock scheduler to return a slot reenter result
        fake_job_scheduler = MagicMock()
        fake_job_scheduler.reenter_job = AsyncMock(return_value={"job_id": "test-job"})

        with patch("proxy.router._get_job_scheduler", return_value=fake_job_scheduler):
            with patch("proxy.session._resolve_session_id_header", return_value=("session-123", {})):
                with patch("proxy.session._build_slot_context", return_value=(None, "", None)):

                    body_emb = {"model": "embed", "input": "test"}
                    resp_emb = await _call_create_embeddings(monkeypatch, body_emb, mock_srv)
                    assert resp_emb.status_code == 503

                    body_chat = {"model": "embed", "messages": [{"role": "user", "content": "hi"}]}
                    resp_chat = await _call_chat(monkeypatch, body_chat, "chat/completions", mock_srv)
                    assert resp_chat.status_code == 503

    @pytest.mark.asyncio
    async def test_slot_file_heuristic_parity(self, monkeypatch, mock_srv, tmp_path):
        """Both endpoints return model_loading when slot file exists."""
        mock_srv.current_model = None
        mock_srv.llama_process = None
        mock_srv.router_is_model_loaded = AsyncMock(return_value=False)
        # Prevent grace window from succeeding
        mock_srv.router_load_model = AsyncMock(return_value=False)

        slot_file = tmp_path / "slots" / "session-slot-123.json"
        slot_file.parent.mkdir(parents=True)
        slot_file.write_text("{}")

        with patch("proxy.router._get_job_scheduler", return_value=None):
            with patch("proxy.session._resolve_session_id_header", return_value=("session-123", {})):
                with patch("proxy.session._build_slot_context",
                           return_value=(None, str(slot_file), {})):

                    body_emb = {"model": "embed", "input": "test"}
                    resp_emb = await _call_create_embeddings(monkeypatch, body_emb, mock_srv)
                    assert resp_emb.status_code == 503

                    body_chat = {"model": "embed", "messages": [{"role": "user", "content": "hi"}]}
                    resp_chat = await _call_chat(monkeypatch, body_chat, "chat/completions", mock_srv)
                    assert resp_chat.status_code == 503

    @pytest.mark.asyncio
    async def test_remote_provider_error_returns_model_loading_parity(self, monkeypatch, mock_srv):
        """Both endpoints return model_loading when all remote providers fail."""
        mock_srv.current_model = None
        mock_srv.llama_process = None
        mock_srv.router_is_model_loaded = AsyncMock(return_value=False)

        # Model must be LOCAL with remote fallback providers, not remote-only
        mock_remote_cfg = {
            "embed": {
                "aliases": ["embed"],
                "providers": [
                    {"name": "local-embed", "type": "local", "llama_model": "mxbai-embed"},
                    {"name": "remote-openai", "type": "remote"},
                ],
            },
        }
        monkeypatch.setattr(mock_srv, "config", {
            "models": mock_remote_cfg,
            "server": {"llama_router_mode": False},
        })

        async def remote_raise(*args, **kwargs):
            raise RuntimeError("Remote failed")

        import proxy.provider
        monkeypatch.setattr(proxy.provider, "proxy_with_remote_fallback", remote_raise)

        with patch("proxy.router._get_job_scheduler", return_value=None):
            with patch("proxy.session._build_slot_context", return_value=(None, "", None)):
                with patch("proxy.session._resolve_session_id_header", return_value=(None, {})):

                    body_emb = {"model": "embed", "input": "test"}
                    resp_emb = await _call_create_embeddings(monkeypatch, body_emb, mock_srv)
                    assert resp_emb.status_code == 503

                    body_chat = {"model": "embed", "messages": [{"role": "user", "content": "hi"}]}
                    resp_chat = await _call_chat(monkeypatch, body_chat, "chat/completions", mock_srv)
                    assert resp_chat.status_code == 503


# ── Tests: Unknown model / default remote handling preserved ─────────────

class TestDefaultRemoteFallback:
    """Unknown-model path preserved for both endpoints."""

    @pytest.mark.asyncio
    async def test_unknown_model_with_default_remote_embeddings(self, monkeypatch, mock_srv):
        """Embeddings falls through to default remote when model unknown."""
        monkeypatch.setattr(mock_srv, "current_model", None)
        monkeypatch.setattr(mock_srv, "config", {
            "default_remote": {"enabled": True, "endpoint": "https://remote.example.com"},
        })
        monkeypatch.setattr(ui, "_srv", lambda: mock_srv)
        mock_srv.proxy_to_remote = AsyncMock(return_value=Response("remote ok", status_code=200))

        body = json.dumps({"model": "unknown-model", "input": "test"}).encode()
        resp = await ui.create_embeddings(DummyRequest(body))
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_unknown_model_with_default_remote_chat(self, monkeypatch, mock_srv):
        """Chat falls through to default remote when model unknown."""
        monkeypatch.setattr(mock_srv, "current_model", None)
        monkeypatch.setattr(mock_srv, "config", {
            "default_remote": {"enabled": True, "endpoint": "https://remote.example.com"},
        })
        monkeypatch.setattr(ui, "_srv", lambda: mock_srv)
        mock_srv.proxy_to_remote = AsyncMock(return_value=Response("remote ok", status_code=200))

        body = json.dumps({"model": "unknown-model", "messages": [{"role": "user", "content": "hi"}]}).encode()
        resp = await ui._do_proxy_openai_api(DummyRequest(body), "chat/completions", body, mock_srv)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_unknown_model_no_default_remote_embeddings(self, monkeypatch, mock_srv):
        """Embeddings raises HTTPException when model unknown and no remote."""
        monkeypatch.setattr(mock_srv, "current_model", None)
        monkeypatch.setattr(mock_srv, "config", {})
        monkeypatch.setattr(ui, "_srv", lambda: mock_srv)

        body = json.dumps({"model": "no-such-model", "input": "test"}).encode()
        with pytest.raises(HTTPException) as exc:
            await ui.create_embeddings(DummyRequest(body))
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_model_no_default_remote_chat(self, monkeypatch, mock_srv):
        """Chat raises HTTPException when model unknown and no remote."""
        monkeypatch.setattr(mock_srv, "current_model", None)
        monkeypatch.setattr(mock_srv, "config", {})
        monkeypatch.setattr(ui, "_srv", lambda: mock_srv)

        body = json.dumps({"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]}).encode()
        with pytest.raises(HTTPException) as exc:
            await ui._do_proxy_openai_api(DummyRequest(body), "chat/completions", body, mock_srv)
        assert exc.value.status_code == 400


# ── Tests: Remote-only model handling ────────────────────────────────────

class TestRemoteModelHandling:
    """Remote model handling preserved for both endpoints."""

    @pytest.mark.asyncio
    async def test_remote_model_embeddings(self, monkeypatch, mock_srv):
        """Remote-only model routes to proxy_with_remote_fallback."""
        monkeypatch.setattr(mock_srv, "config", {
            "models": {
                "gpt4": {
                    "providers": [{"name": "openai", "type": "remote"}],
                },
            },
        })
        monkeypatch.setattr(ui, "_srv", lambda: mock_srv)

        import proxy.provider
        monkeypatch.setattr(proxy.provider, "proxy_with_remote_fallback",
                            AsyncMock(return_value=Response("remote ok", status_code=200)))

        body = json.dumps({"model": "gpt4", "input": "test"}).encode()
        request = DummyRequest(body)
        resp = await ui.create_embeddings(request)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_remote_model_chat(self, monkeypatch, mock_srv):
        """Remote-only model routes to proxy_with_remote_fallback."""
        monkeypatch.setattr(mock_srv, "config", {
            "models": {
                "gpt4": {
                    "providers": [{"name": "openai", "type": "remote"}],
                },
            },
        })
        monkeypatch.setattr(ui, "_srv", lambda: mock_srv)

        import proxy.provider
        monkeypatch.setattr(proxy.provider, "proxy_with_remote_fallback",
                            AsyncMock(return_value=Response("remote ok", status_code=200)))

        body = json.dumps({"model": "gpt4", "messages": [{"role": "user", "content": "hi"}]}).encode()
        resp = await ui._do_proxy_openai_api(DummyRequest(body), "chat/completions", body, mock_srv)
        assert resp.status_code == 200
