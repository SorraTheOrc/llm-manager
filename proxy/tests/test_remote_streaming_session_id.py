"""
Tests for _handle_remote_streaming() accepting session_id parameter.

Verifies that:
1. _handle_remote_streaming() accepts session_id keyword argument without TypeError.
2. When session_id is provided, fire-and-forget recording is scheduled.
3. Streaming without session_id continues to work (no regression).
4. Error-path streaming with session_id also schedules recording.
"""

import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import pytest
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from proxy.proxy_remote import _handle_remote_streaming


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


def _make_mock_response(status_code: int, headers: dict, sse_chunks=None, error_body=None):
    """Create a mock HTTP response for testing."""
    mock_resp = MagicMock()
    type(mock_resp).status_code = PropertyMock(return_value=status_code)
    mock_resp.headers = headers
    if sse_chunks is not None:
        mock_resp.aiter_bytes = MagicMock(return_value=AsyncIterator(sse_chunks))
    if error_body is not None:
        mock_resp.aread = AsyncMock(return_value=error_body)
    return mock_resp


def _make_mock_client(mock_response):
    """Create a mock httpx.AsyncClient that returns mock_response on stream().

    Returns a MagicMock that replaces httpx.AsyncClient. When called,
    returns a client mock whose stream() method returns a context manager
    whose __aenter__ returns mock_response.
    """
    # The context manager returned by client.stream()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_response)
    cm.__aexit__ = AsyncMock(return_value=None)

    # The client instance returned by httpx.AsyncClient()
    client_instance = MagicMock()
    client_instance.stream = MagicMock(return_value=cm)

    # The class mock that replaces httpx.AsyncClient
    mock_client_cls = MagicMock(return_value=client_instance)
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=client_instance)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

    return mock_client_cls


@pytest.fixture
def mock_request():
    """Create a mock Request object."""
    req = MagicMock(spec=Request)
    req.method = "POST"
    req.url.path = "/v1/chat/completions"
    req.is_disconnected = AsyncMock(return_value=False)
    return req


# ═══════════════════════════════════════════════════════════════════════════════
# AC1/AC2: session_id keyword is accepted (no TypeError), recording is scheduled
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_handle_remote_streaming_accepts_session_id(mock_request):
    """_handle_remote_streaming() does not reject session_id keyword argument (AC1/AC2)."""
    sse_chunks = [
        b"data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}, "index": 0}]}).encode() + b"\n\n",
        b"data: [DONE]\n\n",
    ]

    mock_resp = _make_mock_response(
        status_code=200,
        headers={"content-type": "text/event-stream", "content-length": "999"},
        sse_chunks=sse_chunks,
    )
    mock_client_cls = _make_mock_client(mock_resp)

    with patch("proxy.proxy_remote.httpx.AsyncClient", mock_client_cls):
        with patch("proxy.proxy_remote._schedule_traffic_recording") as mock_record:
            with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
                with patch("proxy.proxy_remote.log_response_chunk"):
                    with patch("proxy.proxy_remote.log_response"):
                        with patch("proxy.proxy_remote.log_request"):
                            result = await _handle_remote_streaming(
                                request=mock_request,
                                target_url="https://api.example.com/v1/chat/completions",
                                headers={"Authorization": "Bearer test"},
                                body=b'{"stream": true, "model": "test"}',
                                body_json={"stream": True, "model": "test"},
                                model_name="test-model",
                                remote_timeout=httpx.Timeout(30.0),
                                session_id="test-session-123",
                            )

                            # Consume the streaming response while patches are active.
                            # The generator's finally block runs during consumption.
                            collected = [chunk async for chunk in result.body_iterator]

    # Should return a StreamingResponse (not crash with TypeError)
    assert isinstance(result, StreamingResponse), \
        f"Expected StreamingResponse, got {type(result).__name__}"
    assert len(collected) > 0, "Should have collected streamed chunks"

    # Verify recording was scheduled for the response payload
    response_record_calls = [
        call for call in mock_record.call_args_list
        if call.kwargs.get("session_id") == "test-session-123"
        and "response_payload" in call.kwargs
    ]
    assert len(response_record_calls) >= 1, \
        "_schedule_traffic_recording should have been called with session_id and response_payload"


# ═══════════════════════════════════════════════════════════════════════════════
# AC4: No session_id - streaming still works (regression protection)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_handle_remote_streaming_without_session_id(mock_request):
    """_handle_remote_streaming() works without session_id argument (AC4)."""
    sse_chunks = [
        b"data: " + json.dumps({"choices": [{"delta": {"content": "World"}, "index": 0}]}).encode() + b"\n\n",
        b"data: [DONE]\n\n",
    ]

    mock_resp = _make_mock_response(
        status_code=200,
        headers={"content-type": "text/event-stream", "content-length": "999"},
        sse_chunks=sse_chunks,
    )
    mock_client_cls = _make_mock_client(mock_resp)

    with patch("proxy.proxy_remote.httpx.AsyncClient", mock_client_cls):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        # No session_id keyword argument
                        result = await _handle_remote_streaming(
                            request=mock_request,
                            target_url="https://api.example.com/v1/chat/completions",
                            headers={"Authorization": "Bearer test"},
                            body=b'{"stream": true, "model": "test"}',
                            body_json={"stream": True, "model": "test"},
                            model_name="test-model",
                            remote_timeout=httpx.Timeout(30.0),
                        )

    assert isinstance(result, StreamingResponse), \
        f"Expected StreamingResponse, got {type(result).__name__}"


# ═══════════════════════════════════════════════════════════════════════════════
# AC3: Error path with session_id - schedules recording
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_handle_remote_streaming_error_path_with_session_id(mock_request):
    """Error-path streaming with session_id schedules recording (AC3)."""
    error_body = b'{"error": "upstream failure"}'

    mock_resp = _make_mock_response(
        status_code=500,
        headers={"content-type": "application/json"},
        error_body=error_body,
    )
    mock_client_cls = _make_mock_client(mock_resp)

    with patch("proxy.proxy_remote.httpx.AsyncClient", mock_client_cls):
        with patch("proxy.proxy_remote._schedule_traffic_recording") as mock_record:
            with patch("proxy.proxy_remote.log_response"):
                with patch("proxy.proxy_remote._srv") as mock_srv:
                    mock_srv.return_value = MagicMock(logger=MagicMock(warning=MagicMock()))
                    result = await _handle_remote_streaming(
                        request=mock_request,
                        target_url="https://api.example.com/v1/chat/completions",
                        headers={"Authorization": "Bearer test"},
                        body=b'{"stream": true, "model": "test"}',
                        body_json={"stream": True, "model": "test"},
                        model_name="test-model",
                        remote_timeout=httpx.Timeout(30.0),
                        session_id="test-session-456",
                    )

    assert isinstance(result, Response), \
        f"Expected Response (error path), got {type(result).__name__}"

    # Verify _schedule_traffic_recording was called with response_payload
    response_record_calls = [
        call for call in mock_record.call_args_list
        if call.kwargs.get("session_id") == "test-session-456"
        and call.kwargs.get("response_payload") == error_body
    ]
    assert len(response_record_calls) >= 1, \
        "Error path should call _schedule_traffic_recording with response_payload"


# ═══════════════════════════════════════════════════════════════════════════════
# AC2: Function signature accepts session_id as optional parameter
# ═══════════════════════════════════════════════════════════════════════════════


def test_handle_remote_streaming_signature_has_session_id():
    """Verify _handle_remote_streaming() has session_id as an optional parameter."""
    import inspect

    sig = inspect.signature(_handle_remote_streaming)
    assert "session_id" in sig.parameters, \
        "session_id parameter missing from _handle_remote_streaming()"

    param = sig.parameters["session_id"]
    assert param.default is None, \
        f"session_id default should be None, got {param.default}"
    assert param.kind == param.KEYWORD_ONLY or param.kind == param.POSITIONAL_OR_KEYWORD, \
        "session_id should be a keyword argument"


# ═══════════════════════════════════════════════════════════════════════════════
# Regression: session_id is None by default
# ═══════════════════════════════════════════════════════════════════════════════


def test_handle_remote_streaming_session_id_default_is_none():
    """session_id defaults to None for backward compatibility."""
    import inspect

    sig = inspect.signature(_handle_remote_streaming)
    param = sig.parameters["session_id"]
    assert param.default is None, "session_id must default to None"


# ═══════════════════════════════════════════════════════════════════════════════
# Session header forwarding (LP-0MRE8GD1H0028CGN)
# ═══════════════════════════════════════════════════════════════════════════════


def _make_mock_request_with_session_headers(headers_dict):
    """Create a mock FastAPI Request with specific headers."""
    req = MagicMock(spec=Request)
    req.method = "POST"
    req.url.path = "/v1/chat/completions"
    req.headers = headers_dict
    req.body = AsyncMock(return_value=b'{"model":"test","messages":[]}')
    return req


@pytest.mark.asyncio
async def test_proxy_to_remote_forward_session_headers_default():
    """By default (forward_session_headers not set), session affinity headers
    are NOT stripped from outgoing requests.

    AC1: Default behavior preserves session affinity headers for upstream.
    """
    import proxy.proxy_remote as pr

    mock_req = _make_mock_request_with_session_headers({
        "x-session-id": "sess-test-123",
        "content-type": "application/json",
        "x-client-request-id": "req-456",
    })

    model_config = {
        "endpoint": "https://api.example.com/v1",
        "type": "remote",
        "name": "test-provider",
    }

    captured_headers = {}

    mock_client = MagicMock()
    async def _capture_post(url, **kwargs):
        captured_headers.update(kwargs.get("headers", {}))
        return _make_buffered_response()
    mock_client.post = AsyncMock(side_effect=_capture_post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch.object(pr, "_srv") as mock_srv:
        mock_srv.return_value.config = {"server": {}}
        mock_srv.return_value.logger = MagicMock()
        mock_srv.return_value.current_model = "test"
        # Ensure pooling is skipped (no _remote_http_client) so the test
        # exercises the per-request AsyncClient path (LP-0MRE8G3JK0099Y4J).
        mock_srv.return_value._remote_http_client = None

        with patch.object(pr, "normalize_upstream_request_headers", return_value=dict(mock_req.headers)):
            with patch.object(pr, "log_request"):
                with patch.object(pr, "log_response"):
                    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=mock_client):
                        await pr.proxy_to_remote(
                            request=mock_req,
                            path="v1/chat/completions",
                            model_config=model_config,
                        )

    # x-session-id should be present in forwarded headers (not stripped)
    assert any("session-id" in k.lower() or "session_id" in k.lower() or "client-request" in k.lower()
               for k in captured_headers.keys()), (
        f"Session headers should be forwarded, got: {list(captured_headers.keys())}"
    )


def _make_buffered_response(status_code=200):
    """Create a mock Response for non-streaming path."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"content-type": "application/json"}
    resp.content = b'{"ok":true}'
    return resp


@pytest.mark.asyncio
async def test_proxy_to_remote_strip_session_headers_when_disabled():
    """When forward_session_headers is False, session affinity headers
    ARE stripped from outgoing requests.

    AC2: forward_session_headers: false strips session headers.
    """
    import proxy.proxy_remote as pr

    mock_req = _make_mock_request_with_session_headers({
        "x-session-id": "sess-test-123",
        "content-type": "application/json",
    })

    model_config = {
        "endpoint": "https://api.example.com/v1",
        "type": "remote",
        "name": "test-provider",
        "forward_session_headers": False,
    }

    captured_headers = {}

    mock_client = MagicMock()
    async def _capture_post(url, **kwargs):
        captured_headers.update(kwargs.get("headers", {}))
        return _make_buffered_response()
    mock_client.post = AsyncMock(side_effect=_capture_post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch.object(pr, "_srv") as mock_srv:
        mock_srv.return_value.config = {"server": {}}
        mock_srv.return_value.logger = MagicMock()
        mock_srv.return_value.current_model = "test"
        # Ensure pooling is skipped (no _remote_http_client) so the test
        # exercises the per-request AsyncClient path (LP-0MRE8G3JK0099Y4J).
        mock_srv.return_value._remote_http_client = None

        with patch.object(pr, "normalize_upstream_request_headers", return_value=dict(mock_req.headers)):
            with patch.object(pr, "log_request"):
                with patch.object(pr, "log_response"):
                    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=mock_client):
                        await pr.proxy_to_remote(
                            request=mock_req,
                            path="v1/chat/completions",
                            model_config=model_config,
                        )

    # x-session-id should NOT be present in forwarded headers
    session_headers_found = [
        k for k in captured_headers.keys()
        if any(s in k.lower() for s in ["session-id", "session_id", "client-request"])
    ]
    assert not session_headers_found, (
        f"Session headers should be stripped, found: {session_headers_found}"
    )



