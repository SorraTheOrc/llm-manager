"""
Tests for X-Resolved-Model response header in proxy responses.

Verifies that:
1. proxy_to_local() adds X-Resolved-Model: local/<model> to streaming responses
2. proxy_to_local() adds X-Resolved-Model: local/<model> to non-streaming responses
3. proxy_to_remote() adds X-Resolved-Model: <provider>/<model> to responses
4. proxy_with_fallback() overrides X-Resolved-Model with the actual provider/model
5. proxy_with_remote_fallback() sets X-Resolved-Model with the actual provider/model
6. Non-fallback direct calls to proxy_to_local() also set the header
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import pytest
from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

import proxy.router as router
import proxy.proxy_remote as proxy_remote
import proxy.provider as provider
from proxy.router import proxy_to_local
from proxy.proxy_remote import proxy_to_remote
from proxy.provider import (
    proxy_with_fallback,
    proxy_with_remote_fallback,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


class AsyncIterator:
    """Helper to turn a list into an async iterator."""

    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        return self._iterator()

    async def _iterator(self):
        for item in self.items:
            yield item


def _mock_upstream_response(
    status_code: int = 200,
    content: bytes = b'{"id":"test","choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"Hello!"}}]}',
    content_type: str = "application/json",
):
    """Build a synchronous mock Response (plain object, not httpx spec)."""
    return type("MockResponse", (), {
        "status_code": status_code,
        "content": content,
        "headers": {"content-type": content_type},
    })()


def _mock_streaming_upstream_response(
    status_code: int = 200,
    chunks: list = None,
    content_type: str = "text/event-stream",
):
    """Build a mock httpx streaming response."""
    if chunks is None:
        chunks = [
            b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0}]}\n\n",
            b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}\n\n",
            b"data: [DONE]\n\n",
        ]

    async def _aiter():
        for c in chunks:
            yield c

    MockStreamResponse = type("MockStreamResponse", (), {
        "status_code": status_code,
        "headers": {"content-type": content_type},
        "aiter_bytes": staticmethod(_aiter),
        "aread": AsyncMock(return_value=b"".join(chunks)),
    })

    class MockCM:
        async def __aenter__(self):
            return MockStreamResponse()

        async def __aexit__(self, *args):
            pass

    return MockCM(), MockStreamResponse()


def _make_mock_client(mock_response):
    """Create a mock httpx.AsyncClient that returns mock_response on stream()."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_response)
    cm.__aexit__ = AsyncMock(return_value=None)

    client_instance = MagicMock()
    client_instance.stream = MagicMock(return_value=cm)

    mock_client_cls = MagicMock(return_value=client_instance)
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=client_instance)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

    return mock_client_cls


def _dummy_request(body: dict, stream: bool = False):
    """Build a minimal dummy Request that proxy_to_local can consume."""
    payload = {**body}
    if stream:
        payload["stream"] = True
    body_bytes = json.dumps(payload).encode("utf-8")

    class DummyRequest:
        headers = {"host": "localhost"}
        method = "POST"
        url = type("U", (), {"path": "/v1/chat/completions"})()

        async def body(self):
            return body_bytes

        async def is_disconnected(self):
            return False

    return DummyRequest()


def _make_mock_remote_response(status_code=200, body=None, content_type="text/event-stream"):
    """Create a mock response for remote streaming tests."""
    if body is None:
        body = b'data: {"choices":[{"delta":{"content":"Hi"},"index":0}]}\n\ndata: [DONE]\n\n'

    mock_resp = MagicMock()
    type(mock_resp).status_code = PropertyMock(return_value=status_code)
    mock_resp.headers = {"content-type": content_type}
    mock_resp.aiter_bytes = MagicMock(return_value=AsyncIterator([body]))
    mock_resp.aread = AsyncMock(return_value=body)
    return mock_resp


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_cooldown():
    """Reset cooldown state between tests to avoid cross-test leakage."""
    provider._provider_unavailable_until.clear()
    yield


@pytest.fixture(autouse=True)
def _mock_server_state(monkeypatch):
    """Mock server-level state for proxy_to_local tests."""
    import proxy.server as server
    monkeypatch.setattr(server, "config", {
        "server": {
            "llama_router_mode": False,
            "llama_server_port": 8080,
            "max_concurrent_queries": 4,
            "local_max_concurrent_queries": 1,
            "llama_request_timeout": 30,
            "session_single_flight_mode": "bypass",
            "disconnect_cleanup_timeout": 1,
        }
    })
    monkeypatch.setattr(server, "active_queries", 0)
    monkeypatch.setattr(server, "local_active_queries", 0)
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "llama_process", MagicMock(poll=lambda: None, pid=1))
    monkeypatch.setattr(server, "current_model", "Qwen3")
    monkeypatch.setattr(server, "session_manager", MagicMock())
    monkeypatch.setattr(server, "logger", MagicMock())

    monkeypatch.setattr("proxy.router._get_job_scheduler", lambda: None)
    monkeypatch.setattr("proxy.router._is_self_healing_active", lambda: False)
    monkeypatch.setattr("proxy.router._restore_slot_snapshot", AsyncMock(return_value=False))
    monkeypatch.setattr("proxy.router._save_slot_snapshot", AsyncMock(return_value=False))
    monkeypatch.setattr("proxy.router._build_slot_context", MagicMock(return_value=(None, None, 3.0)))
    monkeypatch.setattr("proxy.router._handle_session", AsyncMock(return_value={
        "session_id": "test-session-id",
        "session_created": True,
        "is_delta_request": False,
        "session_fallback_reason": None,
        "delta_messages": [],
        "original_message_count": 1,
        "body_override": None,
        "body_json": None,
        "session_explicit": False,
    }))
    monkeypatch.setattr("proxy.session._resolve_log_path", MagicMock(return_value=MagicMock(
        exists=lambda: False,
        stat=lambda: MagicMock(st_size=0),
    )))
    monkeypatch.setattr("proxy.router._check_slot_availability", AsyncMock(return_value=None))


@pytest.fixture
def mock_request():
    """Create a mock Request for remote proxy tests."""
    req = MagicMock(spec=Request)
    req.method = "POST"
    req.url.path = "/v1/chat/completions"
    req.is_disconnected = AsyncMock(return_value=False)
    req.body = AsyncMock(return_value=b'{"model":"test","messages":[{"role":"user","content":"hi"}]}')
    return req


# ═══════════════════════════════════════════════════════════════════════════════
# AC1: proxy_to_local streaming adds X-Resolved-Model header
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_proxy_to_local_streaming_adds_resolved_model_header(monkeypatch):
    """proxy_to_local() adds X-Resolved-Model: local/Qwen3 to streaming responses."""
    req = _dummy_request({"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}]}, stream=True)

    cm, upstream_resp = _mock_streaming_upstream_response(
        status_code=200,
        content_type="text/event-stream",
    )

    async def _open_stream_once():
        return cm, upstream_resp

    monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, upstream_resp)))
    monkeypatch.setattr("proxy.router._call_with_empty_retry", AsyncMock())

    with patch("proxy.router.httpx.AsyncClient") as mock_client:
        mock_client.return_value = MagicMock()
        result = await proxy_to_local(req, "v1/chat/completions")

    assert isinstance(result, StreamingResponse), f"Expected StreamingResponse, got {type(result).__name__}"

    # Check headers
    headers = dict(result.headers)
    assert "x-resolved-model" in headers, "X-Resolved-Model header missing from streaming response"
    assert headers["x-resolved-model"] == "local/Qwen3", \
        f"Expected local/Qwen3, got {headers['X-Resolved-Model']}"


# ═══════════════════════════════════════════════════════════════════════════════
# AC2: proxy_to_local non-streaming adds X-Resolved-Model header
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_proxy_to_local_non_streaming_adds_resolved_model_header(monkeypatch):
    """proxy_to_local() adds X-Resolved-Model: local/Qwen3 to non-streaming responses."""
    req = _dummy_request({"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}]}, stream=False)

    upstream_resp = _mock_upstream_response(
        status_code=200,
        content=b'{"id":"test","choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"Hello!"}}]}',
        content_type="application/json",
    )

    monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=upstream_resp))
    monkeypatch.setattr("proxy.router._call_with_empty_retry", AsyncMock(return_value=upstream_resp))
    monkeypatch.setattr("proxy.router._schedule_recv_token_increment", AsyncMock())

    result = await proxy_to_local(req, "v1/chat/completions")

    assert isinstance(result, Response), f"Expected Response, got {type(result).__name__}"
    headers = dict(result.headers)
    assert "x-resolved-model" in headers, "X-Resolved-Model header missing from non-streaming response"
    assert headers["x-resolved-model"] == "local/Qwen3", \
        f"Expected local/Qwen3, got {headers['X-Resolved-Model']}"


# ═══════════════════════════════════════════════════════════════════════════════
# AC3: proxy_to_remote streaming adds X-Resolved-Model header
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_proxy_to_remote_streaming_adds_resolved_model_header(mock_request):
    """proxy_to_remote() adds X-Resolved-Model: opencode/deepseek-v4-flash-free to streaming responses."""
    provider_cfg = {
        "name": "opencode",
        "type": "remote",
        "endpoint": "https://api.opencode.com/v1",
        "model": "deepseek-v4-flash-free",
    }

    sse_chunk = b'data: {"choices":[{"delta":{"content":"Hi"},"index":0}]}\n\ndata: [DONE]\n\n'
    mock_resp = _make_mock_remote_response(status_code=200, body=sse_chunk)
    mock_client_cls = _make_mock_client(mock_resp)

    # Override body to include stream=true for streaming path
    mock_request.body = AsyncMock(return_value=b'{"model":"test","stream":true}')

    with patch("proxy.proxy_remote.httpx.AsyncClient", mock_client_cls):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        result = await proxy_to_remote(
                            request=mock_request,
                            path="v1/chat/completions",
                            model_config=provider_cfg,
                        )

    assert isinstance(result, StreamingResponse), f"Expected StreamingResponse, got {type(result).__name__}"
    headers = dict(result.headers)
    assert "x-resolved-model" in headers, "X-Resolved-Model header missing from remote streaming response"
    assert headers["x-resolved-model"] == "opencode/deepseek-v4-flash-free", \
        f"Expected opencode/deepseek-v4-flash-free, got {headers['X-Resolved-Model']}"


# ═══════════════════════════════════════════════════════════════════════════════
# AC4: proxy_to_remote non-streaming adds X-Resolved-Model header
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_proxy_to_remote_non_streaming_adds_resolved_model_header(mock_request):
    """proxy_to_remote() adds X-Resolved-Model: deepseek/deepseek-chat to non-streaming responses."""
    provider_cfg = {
        "name": "deepseek",
        "type": "remote",
        "endpoint": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    }

    mock_resp = MagicMock()
    type(mock_resp).status_code = PropertyMock(return_value=200)
    mock_resp.headers = {"content-type": "application/json"}
    mock_resp.content = b'{"id":"test","choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"Hello!"}}]}'

    with patch("proxy.proxy_remote.httpx.AsyncClient") as mock_client_cls:
        client_instance = MagicMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=client_instance)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        method_mock = AsyncMock(return_value=mock_resp)
        setattr(client_instance, "post", method_mock)

        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response"):
                with patch("proxy.proxy_remote.log_request"):
                    result = await proxy_to_remote(
                        request=mock_request,
                        path="v1/chat/completions",
                        model_config=provider_cfg,
                    )

    assert isinstance(result, Response), f"Expected Response, got {type(result).__name__}"
    headers = dict(result.headers)
    assert "x-resolved-model" in headers, "X-Resolved-Model header missing from remote non-streaming response"
    assert headers["x-resolved-model"] == "deepseek/deepseek-chat", \
        f"Expected deepseek/deepseek-chat, got {headers['X-Resolved-Model']}"


# ═══════════════════════════════════════════════════════════════════════════════
# AC5: proxy_with_fallback adds X-Resolved-Model with correct provider/model
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_proxy_with_fallback_sets_resolved_model_header(monkeypatch):
    """proxy_with_fallback() sets X-Resolved-Model with the provider/model that succeeded."""
    model_config = {
        "providers": [
            {
                "name": "local-llama",
                "type": "local",
                "llama_model": "Qwen3",
            },
            {
                "name": "remote-fallback",
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
                "model": "fallback-model",
            },
        ]
    }

    server_config = {
        "server": {
            "provider_cooldown_seconds": 60,
            "local_slot_exhaustion_retry_attempts": 0,
            "local_slot_exhaustion_retry_delay_seconds": 0.2,
            "local_max_concurrent_queries": 4,
        }
    }

    req = _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=False)

    # Mock proxy_to_local to return a successful response
    local_response = Response(
        content=b'{"id":"test","choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"Hello!"}}]}',
        status_code=200,
        headers={"content-type": "application/json"},
    )

    async def mock_proxy_to_local(request, path):
        return local_response

    # Patch _get_proxy_to_local to return our mock
    with patch("proxy.provider._get_proxy_to_local", return_value=mock_proxy_to_local):
        result = await proxy_with_fallback(req, "v1/chat/completions", model_config, server_config)

    assert isinstance(result, Response), f"Expected Response, got {type(result).__name__}"
    headers = dict(result.headers)

    # Should have both X-Provider and X-Resolved-Model
    assert "x-provider" in headers, "X-Provider header missing"
    assert "x-resolved-model" in headers, "X-Resolved-Model header missing"
    assert headers["x-provider"] == "local-llama", \
        f"Expected X-Provider=local-llama, got {headers['X-Provider']}"
    assert headers["x-resolved-model"] == "local-llama/Qwen3", \
        f"Expected X-Resolved-Model=local-llama/Qwen3, got {headers['X-Resolved-Model']}"


# ═══════════════════════════════════════════════════════════════════════════════
# AC6: proxy_with_remote_fallback adds X-Resolved-Model
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_proxy_with_remote_fallback_sets_resolved_model_header():
    """proxy_with_remote_fallback() sets X-Resolved-Model with the provider/model."""
    model_config = {
        "providers": [
            {
                "name": "primary",
                "type": "remote",
                "endpoint": "https://api.primary.com/v1",
                "model": "primary-model",
            },
        ]
    }

    server_config = {"provider_cooldown_seconds": 60}

    req = MagicMock(spec=Request)
    req.method = "POST"
    req.url.path = "/v1/chat/completions"
    req.is_disconnected = AsyncMock(return_value=False)
    req.body = AsyncMock(return_value=b'{"model":"test","messages":[{"role":"user","content":"hi"}]}')

    # Create a non-streaming success response from remote
    remote_response = Response(
        content=b'{"id":"test","choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"Hello!"}}]}',
        status_code=200,
        headers={"content-type": "application/json"},
    )

    async def mock_proxy_to_remote(request, path, provider_cfg):
        return remote_response

    with patch("proxy.provider._get_proxy_to_remote", return_value=mock_proxy_to_remote):
        result = await proxy_with_remote_fallback(req, "v1/chat/completions", model_config, server_config)

    assert isinstance(result, Response), f"Expected Response, got {type(result).__name__}"
    headers = dict(result.headers)

    assert "x-provider" in headers, "X-Provider header missing"
    assert "x-resolved-model" in headers, "X-Resolved-Model header missing"
    assert headers["x-provider"] == "primary", \
        f"Expected X-Provider=primary, got {headers['X-Provider']}"
    assert headers["x-resolved-model"] == "primary/primary-model", \
        f"Expected X-Resolved-Model=primary/primary-model, got {headers['X-Resolved-Model']}"


# ═══════════════════════════════════════════════════════════════════════════════
# AC7: Fallback overrides local header with resolved provider/model
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_fallback_overrides_resolved_model_header(monkeypatch):
    """proxy_with_fallback() overrides proxy_to_local's X-Resolved-Model with fallback provider/model."""
    model_config = {
        "providers": [
            {
                "name": "local-llama",
                "type": "local",
                "llama_model": "Qwen3",
            },
            {
                "name": "remote-fallback",
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
                "model": "fallback-model",
            },
        ]
    }

    server_config = {
        "server": {
            "provider_cooldown_seconds": 60,
            "local_slot_exhaustion_retry_attempts": 0,
            "local_slot_exhaustion_retry_delay_seconds": 0.2,
            "local_max_concurrent_queries": 4,
        }
    }

    req = _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=False)

    # First provider (local) fails with 503, then fallback to remote succeeds
    error_response = Response(
        content=b'{"error":"slot_exhaustion","status":503}',
        status_code=503,
        headers={"content-type": "application/json"},
    )

    remote_success_response = Response(
        content=b'{"id":"test","choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"Hello!"}}]}',
        status_code=200,
        headers={"content-type": "application/json"},
    )

    call_count = 0

    async def mock_proxy_to_local(request, path):
        nonlocal call_count
        call_count += 1
        # Always return error for local
        return error_response

    async def mock_proxy_to_remote(request, path, provider_cfg):
        return remote_success_response

    with patch("proxy.provider._get_proxy_to_local", return_value=mock_proxy_to_local):
        with patch("proxy.provider._get_proxy_to_remote", return_value=mock_proxy_to_remote):
            # Also need to patch _is_slot_exhaustion_response to avoid cooldown
            with patch("proxy.provider._is_slot_exhaustion_response", return_value=True):
                result = await proxy_with_fallback(req, "v1/chat/completions", model_config, server_config)

    assert isinstance(result, Response), f"Expected Response, got {type(result).__name__}"
    headers = dict(result.headers)

    assert "x-resolved-model" in headers, "X-Resolved-Model header missing"
    # Should show the fallback provider, not the local one
    assert headers["x-resolved-model"] == "remote-fallback/fallback-model", \
        f"Expected remote-fallback/fallback-model, got {headers['X-Resolved-Model']}"


# ═══════════════════════════════════════════════════════════════════════════════
# AC8: proxy_to_remote uses model_config name for streaming header
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_proxy_to_remote_streaming_uses_config_name(mock_request):
    """proxy_to_remote streaming uses model_config name and model for the header."""
    provider_cfg = {
        "name": "opencode-go",
        "type": "remote",
        "endpoint": "https://api.opencode-go.com/v1",
        "model": "deepseek-v4-flash",
    }

    sse_chunk = b'data: {"choices":[{"delta":{"content":"Hi"},"index":0}]}\n\ndata: [DONE]\n\n'
    mock_resp = _make_mock_remote_response(status_code=200, body=sse_chunk)
    mock_client_cls = _make_mock_client(mock_resp)

    # Override body to include stream=true for streaming path
    mock_request.body = AsyncMock(return_value=b'{"model":"test","stream":true}')

    with patch("proxy.proxy_remote.httpx.AsyncClient", mock_client_cls):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        result = await proxy_to_remote(
                            request=mock_request,
                            path="v1/chat/completions",
                            model_config=provider_cfg,
                        )

    assert isinstance(result, StreamingResponse), f"Expected StreamingResponse, got {type(result).__name__}"
    headers = dict(result.headers)
    assert "x-resolved-model" in headers, "X-Resolved-Model header missing"
    assert headers["x-resolved-model"] == "opencode-go/deepseek-v4-flash", \
        f"Expected opencode-go/deepseek-v4-flash, got {headers['X-Resolved-Model']}"


# ═══════════════════════════════════════════════════════════════════════════════
# AC9: X-Resolved-Model reflects upstream_model override, not original model name
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_proxy_to_remote_header_reflects_upstream_model_override(mock_request):
    """proxy_to_remote uses the upstream model after override, not the original alias."""
    provider_cfg = {
        "name": "opencode",
        "type": "remote",
        "endpoint": "https://api.opencode.com/v1",
        "model": "deepseek-v4-flash-free",  # upstream model override
    }

    # Body sends "plan" but provider cfg overrides to "deepseek-v4-flash-free"
    sse_chunk = b'data: {"choices":[{"delta":{"content":"Hi"},"index":0}]}\n\ndata: [DONE]\n\n'
    mock_resp = _make_mock_remote_response(status_code=200, body=sse_chunk)
    mock_client_cls = _make_mock_client(mock_resp)

    # Override body to include stream=true for streaming path
    mock_request.body = AsyncMock(return_value=b'{"model":"plan","messages":[{"role":"user","content":"hi"}],"stream":true}')

    with patch("proxy.proxy_remote.httpx.AsyncClient", mock_client_cls):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        result = await proxy_to_remote(
                            request=mock_request,
                            path="v1/chat/completions",
                            model_config=provider_cfg,
                        )

    assert isinstance(result, StreamingResponse), f"Expected StreamingResponse, got {type(result).__name__}"
    headers = dict(result.headers)
    assert "x-resolved-model" in headers, "X-Resolved-Model header missing"
    assert headers["x-resolved-model"] == "opencode/deepseek-v4-flash-free", \
        f"Expected opencode/deepseek-v4-flash-free, got {headers['X-Resolved-Model']}"
