"""
Tests for client disconnect detection and cleanup.

Verifies that:
1. request.is_disconnected() is called periodically during streaming (AC5)
2. On disconnect, cleanup functions execute correctly (AC5)
3. Active queries counter does not leak (AC5)
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import pytest
from fastapi.responses import StreamingResponse


class AsyncIterator:
    """Helper to turn a list into an async iterator."""

    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        return self._iterator()

    async def _iterator(self):
        for item in self.items:
            yield item


# ===================================================================
# Proxy remote streaming disconnect tests
# ===================================================================


def _make_mock_remote_stream_response(chunks, status=200):
    """Create a mock httpx response with streaming bytes."""
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.headers = {"content-type": "text/event-stream"}
    mock_resp.aiter_bytes = lambda: AsyncIterator(chunks)
    return mock_resp


@pytest.mark.asyncio
async def test_remote_stream_disconnect_detected():
    """Verify disconnect detection in proxy_remote streaming (AC5)."""
    from proxy.proxy_remote import _handle_remote_streaming

    mock_request = MagicMock(spec=["method", "url", "headers", "is_disconnected"])
    mock_request.method = "POST"
    type(mock_request.url).path = PropertyMock(return_value="/v1/chat/completions")
    mock_request.headers = {}
    mock_request.is_disconnected = AsyncMock(return_value=False)

    # Need 11+ chunks to trigger the disconnect check counter (check every 10)
    chunks = [
        b'data: {"choices": [{"delta": {"content": "A"}, "index": 0}]}\n\n',
    ] * 12

    with patch("proxy.proxy_remote.httpx.AsyncClient") as mock_client_cls:
        mock_cm = AsyncMock()
        mock_resp = _make_mock_remote_stream_response(chunks)
        mock_cm.__aenter__.return_value = mock_resp

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client

        with patch("proxy.proxy_remote.log_response_chunk"):
            result = await _handle_remote_streaming(
                mock_request,
                "http://fake.api/v1/chat/completions",
                {"Authorization": "Bearer test-key"},
                json.dumps({"model": "test-model", "stream": True}).encode(),
                {"model": "test-model", "stream": True},
                "test-model",
                httpx.Timeout(300),
            )

            assert isinstance(result, StreamingResponse)

            collected = b""
            async for chunk in result.body_iterator:
                collected += chunk

            assert mock_request.is_disconnected.called, (
                "request.is_disconnected() should be checked during remote streaming"
            )


@pytest.mark.asyncio
async def test_remote_stream_disconnect_cleanup():
    """Verify cleanup runs when disconnect detected in remote streaming."""
    from proxy.proxy_remote import _handle_remote_streaming

    mock_request = MagicMock(spec=["method", "url", "headers", "is_disconnected"])
    mock_request.method = "POST"
    type(mock_request.url).path = PropertyMock(return_value="/v1/chat/completions")
    mock_request.headers = {}
    mock_request.is_disconnected = AsyncMock(return_value=True)

    chunks = [
        b'data: {"choices": [{"delta": {"content": "Hi"}, "index": 0}]}\n\n',
    ]

    with patch("proxy.proxy_remote.httpx.AsyncClient") as mock_client_cls:
        mock_cm = AsyncMock()
        mock_resp = _make_mock_remote_stream_response(chunks)
        mock_cm.__aenter__.return_value = mock_resp

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client

        with patch("proxy.proxy_remote.log_response_chunk"):
            result = await _handle_remote_streaming(
                mock_request,
                "http://fake.api/v1/chat/completions",
                {"Authorization": "Bearer test-key"},
                json.dumps({"model": "test-model", "stream": True}).encode(),
                {"model": "test-model", "stream": True},
                "test-model",
                httpx.Timeout(300),
            )

            async for chunk in result.body_iterator:
                pass

            assert mock_cm.__aexit__.called, (
                "Context manager __aexit__ should be called on disconnect"
            )
            assert mock_client.aclose.called, (
                "httpx client aclose should be called on disconnect"
            )


# ===================================================================
# Router streaming disconnect detection
# ===================================================================

def _make_mock_request(is_disconnected=False):
    """Create a mock Starlette Request."""
    mock_req = MagicMock(spec=["is_disconnected", "headers", "method", "url"])
    mock_req.is_disconnected = AsyncMock(return_value=is_disconnected)
    mock_req.method = "POST"
    type(mock_req.url).path = PropertyMock(return_value="/v1/chat/completions")
    mock_req.headers = {}
    return mock_req


def _make_mock_stream_response(chunks, status=200):
    """Create a mock httpx response with streaming bytes."""
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.headers = {"content-type": "text/event-stream"}
    mock_resp.aiter_bytes = lambda: AsyncIterator(chunks)
    return mock_resp


@pytest.fixture
def mock_server():
    """Create a minimal mock server object."""
    srv = MagicMock()
    srv.active_queries = 0
    srv.active_queries_lock = asyncio.Lock()
    srv.backend_ready = True
    srv.llama_process = MagicMock()
    srv.current_model = "test-model"
    srv.logger = MagicMock()
    srv.session_manager = AsyncMock()
    srv.config = {
        "server": {
            "max_concurrent_queries": 4,
            "session_single_flight_mode": "queue",
            "session_single_flight_max_queue_depth": 1,
        }
    }
    return srv


@pytest.mark.asyncio
async def test_router_stream_disconnect_calls_is_disconnected(mock_server):
    """Verify request.is_disconnected() is called during streaming (AC5)."""
    from proxy.router import proxy_to_local

    mock_req = _make_mock_request(is_disconnected=False)

    chunks = [b'data: test\n\n'] * 15  # 15 chunks to ensure counter reaches 10

    mock_resp = _make_mock_stream_response(chunks)

    with (
        patch("proxy.router._srv", return_value=mock_server),
        patch("proxy.router._is_self_healing_active", return_value=False),
        patch("proxy.router._check_slot_availability", return_value=None),
        patch("proxy.router._increment_active_queries"),
        patch("proxy.router._decrement_active_queries"),
        patch("proxy.router._handle_session") as mock_session,
        patch("proxy.router._build_slot_context", return_value=(None, None, 3.0)),
        patch("proxy.router._call_with_backend_retries", new_callable=AsyncMock) as mock_retry,
        patch("proxy.router.httpx.AsyncClient") as mock_client_cls,
        patch("proxy.router._schedule_token_increment"),
        patch("proxy.router._schedule_recv_token_increment"),
        patch("proxy.router.count_text_tokens", return_value=5),
        patch("proxy.router._extract_delta_text_from_sse_chunk", return_value=""),
        patch("proxy.router.evaluate_stream_guardrail", return_value=None),
        patch("proxy.router.log_request"),
        patch("proxy.router.log_response_chunk"),
    ):
        mock_session.return_value = {
            "session_id": "test-session-123",
            "session_created": True,
            "is_delta_request": False,
            "session_fallback_reason": None,
            "delta_messages": None,
            "body_override": None,
            "body_json": {"messages": [], "model": "test-model", "stream": True},
            "original_message_count": 0,
        }

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_resp
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client
        mock_retry.return_value = (mock_cm, mock_resp)

        mock_req.body = AsyncMock(return_value=json.dumps(
            {"messages": [], "model": "test-model", "stream": True}
        ).encode())

        result = await proxy_to_local(mock_req, "v1/chat/completions")
        assert isinstance(result, StreamingResponse), (
            f"Expected StreamingResponse, got {type(result).__name__}"
        )

        collected = b""
        async for chunk in result.body_iterator:
            collected += chunk

        assert mock_req.is_disconnected.called, (
            "request.is_disconnected() should have been called during streaming"
        )


@pytest.mark.asyncio
async def test_router_stream_disconnect_triggers_cleanup(mock_server):
    """On disconnect, cleanup functions execute correctly (AC5)."""
    from proxy.router import proxy_to_local

    mock_req = _make_mock_request(is_disconnected=True)

    chunks = [b'data: test\n\n']

    mock_resp = _make_mock_stream_response(chunks)

    with (
        patch("proxy.router._srv", return_value=mock_server),
        patch("proxy.router._is_self_healing_active", return_value=False),
        patch("proxy.router._check_slot_availability", return_value=None),
        patch("proxy.router._increment_active_queries"),
        patch("proxy.router._decrement_active_queries") as mock_decrement,
        patch("proxy.router._handle_session") as mock_session,
        patch("proxy.router._build_slot_context", return_value=(None, None, 3.0)),
        patch("proxy.router._call_with_backend_retries", new_callable=AsyncMock) as mock_retry,
        patch("proxy.router.httpx.AsyncClient") as mock_client_cls,
        patch("proxy.router._schedule_token_increment"),
        patch("proxy.router._schedule_recv_token_increment"),
        patch("proxy.router.count_text_tokens", return_value=5),
        patch("proxy.router._extract_delta_text_from_sse_chunk", return_value=""),
        patch("proxy.router.evaluate_stream_guardrail", return_value=None),
        patch("proxy.router.log_request"),
        patch("proxy.router.log_response_chunk"),
    ):
        mock_session.return_value = {
            "session_id": "test-session-123",
            "session_created": True,
            "is_delta_request": False,
            "session_fallback_reason": None,
            "delta_messages": None,
            "body_override": None,
            "body_json": {"messages": [], "model": "test-model", "stream": True},
            "original_message_count": 0,
        }

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_resp
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client
        mock_retry.return_value = (mock_cm, mock_resp)

        mock_req.body = AsyncMock(return_value=json.dumps(
            {"messages": [], "model": "test-model", "stream": True}
        ).encode())

        result = await proxy_to_local(mock_req, "v1/chat/completions")
        assert isinstance(result, StreamingResponse), (
            f"Expected StreamingResponse, got {type(result).__name__}"
        )

        collected = b""
        async for chunk in result.body_iterator:
            collected += chunk

        assert mock_decrement.called, (
            "_decrement_active_queries should have been called on cleanup"
        )
        assert mock_client.aclose.called, (
            "httpx client aclose should be called on cleanup"
        )
        assert mock_cm.__aexit__.called, (
            "stream context manager __aexit__ should be called on cleanup"
        )


@pytest.mark.asyncio
async def test_router_stream_disconnect_active_queries_not_leaked(mock_server):
    """Active queries counter does not leak on disconnect (AC5)."""
    from proxy.router import proxy_to_local

    mock_req = _make_mock_request(is_disconnected=True)
    session_id = "test-session-456"

    chunks = [b'data: test\n\n']

    mock_resp = _make_mock_stream_response(chunks)

    with (
        patch("proxy.router._srv", return_value=mock_server),
        patch("proxy.router._is_self_healing_active", return_value=False),
        patch("proxy.router._check_slot_availability", return_value=None),
        patch("proxy.router._increment_active_queries"),
        patch("proxy.router._handle_session") as mock_session,
        patch("proxy.router._build_slot_context", return_value=(None, None, 3.0)),
        patch("proxy.router._call_with_backend_retries", new_callable=AsyncMock) as mock_retry,
        patch("proxy.router.httpx.AsyncClient") as mock_client_cls,
        patch("proxy.router._schedule_token_increment"),
        patch("proxy.router._schedule_recv_token_increment"),
        patch("proxy.router.count_text_tokens", return_value=5),
        patch("proxy.router._extract_delta_text_from_sse_chunk", return_value=""),
        patch("proxy.router.evaluate_stream_guardrail", return_value=None),
        patch("proxy.router.log_request"),
        patch("proxy.router.log_response_chunk"),
        patch("proxy.router._decrement_active_queries") as mock_decrement,
    ):
        mock_session.return_value = {
            "session_id": session_id,
            "session_created": True,
            "is_delta_request": False,
            "session_fallback_reason": None,
            "delta_messages": None,
            "body_override": None,
            "body_json": {"messages": [], "model": "test-model", "stream": True},
            "original_message_count": 0,
        }

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_resp
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client
        mock_retry.return_value = (mock_cm, mock_resp)

        mock_req.body = AsyncMock(return_value=json.dumps(
            {"messages": [], "model": "test-model", "stream": True}
        ).encode())

        result = await proxy_to_local(mock_req, "v1/chat/completions")
        assert isinstance(result, StreamingResponse), (
            f"Expected StreamingResponse, got {type(result).__name__}"
        )
        async for chunk in result.body_iterator:
            pass

        assert mock_decrement.called, (
            "active_queries must be decremented on disconnect"
        )


# ===================================================================
# Non-streaming disconnect tests
# ===================================================================

@pytest.mark.asyncio
async def test_non_streaming_disconnect_before_request(mock_server):
    """Non-streaming path detects disconnect before sending to backend."""
    from proxy.router import proxy_to_local

    mock_req = _make_mock_request(is_disconnected=True)
    mock_req.body = AsyncMock(return_value=json.dumps(
        {"messages": [{"role": "user", "content": "Hello"}], "model": "test-model"}
    ).encode())

    with (
        patch("proxy.router._srv", return_value=mock_server),
        patch("proxy.router._is_self_healing_active", return_value=False),
        patch("proxy.router._check_slot_availability", return_value=None),
        patch("proxy.router._increment_active_queries"),
        patch("proxy.router._decrement_active_queries"),
        patch("proxy.router._handle_session") as mock_session,
        patch("proxy.router._build_slot_context", return_value=(None, None, 3.0)),
        patch("proxy.router._call_with_backend_retries", new_callable=AsyncMock),
        patch("proxy.router._call_with_empty_retry", new_callable=AsyncMock) as mock_empty_retry,
        patch("proxy.router._estimate_tokens_sent", return_value=10),
        patch("proxy.router._schedule_token_increment"),
        patch("proxy.router._schedule_recv_token_increment"),
        patch("proxy.router.log_request"),
        patch("proxy.router.log_response"),
    ):
        mock_resp_for_nonstream = MagicMock()
        mock_resp_for_nonstream.status_code = 200
        mock_resp_for_nonstream.headers = {"content-type": "application/json"}
        mock_resp_for_nonstream.content = b'{"choices": [{"message": {"content": "Hi"}}]}'
        mock_empty_retry.return_value = mock_resp_for_nonstream

        mock_session.return_value = {
            "session_id": "test-session-ns",
            "session_created": False,
            "is_delta_request": False,
            "session_fallback_reason": None,
            "delta_messages": None,
            "body_override": None,
            "body_json": {"messages": [{"role": "user", "content": "Hello"}], "model": "test-model"},
            "original_message_count": 1,
        }

        result = await proxy_to_local(mock_req, "v1/chat/completions")

        assert result is not None
