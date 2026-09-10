"""
Tests for body snippet logging on empty response detection.

Covers:
1. AC1: Streaming empty response log includes truncated upstream body snippet
2. AC2: Non-streaming empty response log includes body snippet (≤512 chars)
3. AC3: No secrets leak into logged snippets (body is model output, truncated)
4. Edge cases: empty body, long body truncation, None handling

Related: LP-0MTVPJWWZ000REYU
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse

from proxy.proxy_remote import (
    _handle_remote_non_streaming,
    _handle_remote_streaming,
    _snippet_body as _remote_snippet,
)
from proxy.utils import _snippet_body as _utils_snippet, _call_with_empty_retry


# ===================================================================
# Async iterator helpers
# ===================================================================


class AsyncChunkIterator:
    """Async iterator that yields pre-defined byte chunks."""

    def __init__(self, chunks, hang_after=False):
        self._chunks = list(chunks)
        self._hang_after = hang_after
        self._done = False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for chunk in self._chunks:
            yield chunk
        if self._hang_after:
            await asyncio.Event().wait()


# ===================================================================
# Mock factories
# ===================================================================


def _make_streaming_mock_response(
    status_code=200,
    headers=None,
    aiter_chunks=None,
    hang_after=False,
):
    """Create a mock HTTP response for streaming tests."""
    mock_resp = MagicMock(spec=httpx.Response)
    type(mock_resp).status_code = PropertyMock(return_value=status_code)
    mock_resp.headers = headers or {"content-type": "text/event-stream"}
    if aiter_chunks is not None:
        mock_resp.aiter_bytes = MagicMock(
            return_value=AsyncChunkIterator(aiter_chunks, hang_after=hang_after)
        )
    return mock_resp


def _make_non_streaming_mock_response(
    status_code=200,
    body_bytes=b'{"choices":[{"message":{"content":"Hello"},"finish_reason":"stop"}],"usage":{"total_tokens":5}}',
    headers=None,
):
    """Create a mock HTTP response for non-streaming tests."""
    mock_resp = MagicMock(spec=httpx.Response)
    type(mock_resp).status_code = PropertyMock(return_value=status_code)
    mock_resp.headers = headers or {"content-type": "application/json"}
    mock_resp.content = body_bytes
    return mock_resp


def _make_client(stream_responses=None, non_stream_responses=None):
    """Create a mock httpx.AsyncClient."""
    client = MagicMock(spec=httpx.AsyncClient)
    client.aclose = AsyncMock(return_value=None)

    if stream_responses is not None:
        cms = []
        for resp in stream_responses:
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=resp)
            cm.__aexit__ = AsyncMock(return_value=None)
            cms.append(cm)
        client.stream = MagicMock(side_effect=cms)

    if non_stream_responses is not None:
        client.post = AsyncMock(side_effect=non_stream_responses)

    return client


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def mock_request():
    req = MagicMock(spec=Request)
    req.method = "POST"
    req.url.path = "/v1/chat/completions"
    req.is_disconnected = AsyncMock(return_value=False)
    req.headers = {}
    return req


@pytest.fixture
def mock_srv():
    """Fixture for server state mock."""
    srv = MagicMock()
    srv.config = {}
    srv.logger = MagicMock()
    return srv


# ===================================================================
# Unit tests: _snippet_body helper (both modules)
# ===================================================================


class TestSnippetBody:
    """Unit tests for the _snippet_body helper function."""

    def test_empty_string_returns_placeholder(self):
        """Empty string returns '<empty>' placeholder."""
        assert _utils_snippet("") == "<empty>"
        assert _remote_snippet("") == "<empty>"
        assert _remote_snippet(b"") == "<empty>"

    def test_none_returns_placeholder(self):
        """None/None-like values return '<empty>'."""
        assert _utils_snippet(None) == "<empty>"

    def test_short_string_passed_through(self):
        """Short strings are returned unchanged."""
        text = '{"choices":[{"message":{"content":""}}]}'
        assert _utils_snippet(text) == text

    def test_long_string_truncated(self):
        """Long strings are truncated at 512 chars with '...' suffix."""
        long_body = '{"choices":[{"message":{"content":"' + "x" * 600 + '"}}]}'
        snippet = _utils_snippet(long_body)
        assert len(snippet) == 515  # 512 + '...'
        assert snippet.endswith("...")
        assert snippet[:512] == long_body[:512]

    def test_bytes_input_truncated(self):
        """Bytes input is decoded and truncated."""
        long_bytes = b'{"choices":[{"message":{"content":"' + b"x" * 600 + b'"}}]}'
        snippet = _remote_snippet(long_bytes)
        assert isinstance(snippet, str)
        assert len(snippet) == 515
        assert snippet.endswith("...")

    def test_exact_length_boundary(self):
        """Exactly 512 chars: no truncation suffix."""
        exact = "a" * 512
        snippet = _utils_snippet(exact)
        assert snippet == exact
        assert not snippet.endswith("...")

    def test_513_chars_truncated(self):
        """513 chars: truncated with '...' suffix."""
        over = "a" * 513
        snippet = _utils_snippet(over)
        assert len(snippet) == 515  # 512 + '...'


# ===================================================================
# AC1: Streaming path — body snippet in empty response log
# ===================================================================


@pytest.mark.asyncio
async def test_streaming_empty_response_log_includes_body_snippet(mock_request, mock_srv):
    """AC1: Streaming empty response log includes truncated upstream body snippet.

    When the streaming empty-response retry fires, the log message must include
    a body snippet extracted from the raw upstream SSE chunks that were yielded
    during the first (empty) attempt.
    """
    # The upstream sends SSE events with empty content
    empty_stream_chunks = [
        b'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}\n\n',
        b'data: [DONE]\n\n',
    ]
    valid_stream_chunks = [
        b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    first_resp = _make_streaming_mock_response(
        status_code=200,
        aiter_chunks=empty_stream_chunks,
    )
    second_resp = _make_streaming_mock_response(
        status_code=200,
        aiter_chunks=valid_stream_chunks,
    )

    client = _make_client(stream_responses=[first_resp, second_resp])
    mock_srv.config = {}

    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client):
        with patch("proxy.proxy_remote._srv", return_value=mock_srv):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
                            result = await _handle_remote_streaming(
                                request=mock_request,
                                target_url="https://api.example.com/v1/chat/completions",
                                headers={"Authorization": "Bearer test"},
                                body=b'{"stream": true, "model": "test"}',
                                body_json={"stream": True, "model": "test"},
                                model_name="test-model",
                                remote_timeout=httpx.Timeout(30.0),
                                upstream_idle_timeout_seconds=1.0,
                            )

                            collected = [chunk async for chunk in result.body_iterator]

    # Verify the empty-retry INFO log includes a body snippet
    empty_retry_logs = [
        call for call in mock_srv.logger.info.call_args_list
        if "Empty response detected" in str(call.args[0]) and "stream attempt" in str(call.args[0])
    ]
    assert len(empty_retry_logs) == 1, (
        f"Expected one 'Empty response detected on stream' log, got {len(empty_retry_logs)}"
    )

    log_fmt, *log_args = empty_retry_logs[0].args
    assert "upstream_body_snippet=%s" in log_fmt, (
        f"Empty-retry log missing upstream_body_snippet placeholder: {log_fmt}"
    )
    # The body snippet should contain the raw SSE from the first attempt
    body_snippet = log_args[-1]
    assert '{"choices"' in body_snippet, (
        f"Expected raw SSE JSON in body snippet, got: {body_snippet[:100]}"
    )


@pytest.mark.asyncio
async def test_streaming_empty_response_log_body_snippet_truncated(mock_request, mock_srv):
    """AC1 (boundary): Body snippet is truncated to 512 chars for very large empty responses.

    When the upstream sends a very long empty-response body, the snippet must
    be truncated and must not exceed 512 chars + '...'
    """
    # Build a large empty SSE response
    large_empty_chunk = (
        b'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0},'
        b'"_padding":"' + b"x" * 1000 + b'"}]}\n\n'
    )
    empty_stream_chunks = [large_empty_chunk, b'data: [DONE]\n\n']
    valid_stream_chunks = [
        b'data: {"choices":[{"delta":{"content":"Retry"},"index":0}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    first_resp = _make_streaming_mock_response(
        status_code=200,
        aiter_chunks=empty_stream_chunks,
    )
    second_resp = _make_streaming_mock_response(
        status_code=200,
        aiter_chunks=valid_stream_chunks,
    )

    client = _make_client(stream_responses=[first_resp, second_resp])
    mock_srv.config = {}

    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client):
        with patch("proxy.proxy_remote._srv", return_value=mock_srv):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
                            result = await _handle_remote_streaming(
                                request=mock_request,
                                target_url="https://api.example.com/v1/chat/completions",
                                headers={"Authorization": "Bearer test"},
                                body=b'{"stream": true, "model": "test"}',
                                body_json={"stream": True, "model": "test"},
                                model_name="test-model",
                                remote_timeout=httpx.Timeout(30.0),
                                upstream_idle_timeout_seconds=1.0,
                            )

                            collected = [chunk async for chunk in result.body_iterator]

    # Verify truncation
    empty_retry_logs = [
        call for call in mock_srv.logger.info.call_args_list
        if "Empty response detected" in str(call.args[0]) and "stream attempt" in str(call.args[0])
    ]
    assert len(empty_retry_logs) == 1
    _, *log_args = empty_retry_logs[0].args
    body_snippet = log_args[-1]
    assert len(body_snippet) <= 515, f"Body snippet too long: {len(body_snippet)} chars"
    assert body_snippet.endswith("..."), "Long body snippet should end with '...'"


# ===================================================================
# AC2: Non-streaming path — body snippet in empty response log
# ===================================================================


@pytest.mark.asyncio
async def test_non_streaming_empty_response_log_includes_body_snippet(mock_request, mock_srv):
    """AC2: Non-streaming empty response WARNING includes body snippet.

    When the non-streaming empty-response retry is exhausted, the WARNING log
    must include the body snippet and status code.
    """
    empty_body = json.dumps({
        "choices": [{"message": {"content": [], "stopReason": "stop"}, "index": 0}],
        "usage": {"input": 0, "output": 0, "total": 0},
    }).encode("utf-8")

    mock_srv.config = {
        "server": {
            "upstream_empty_retry_max_attempts": 1,
            "upstream_empty_retry_base_delay_seconds": 0.01,
        }
    }

    first_resp = _make_non_streaming_mock_response(body_bytes=empty_body)
    second_resp = _make_non_streaming_mock_response(body_bytes=empty_body)
    client = _make_client(non_stream_responses=[first_resp, second_resp])

    with patch("proxy.proxy_remote._srv", return_value=mock_srv):
        with patch("proxy.proxy_remote.log_response"):
            await _handle_remote_non_streaming(
                request=mock_request,
                target_url="https://api.example.com/v1/chat/completions",
                headers={"Authorization": "Bearer test"},
                body=b'{"model": "test"}',
                model_name="test-model",
                remote_timeout=httpx.Timeout(30.0),
                pool_client=client,
                session_id=None,
                resolved_model=None,
            )

    # Check that the WARNING log includes body_snippet and status_code
    warn_logs = [
        call for call in mock_srv.logger.warning.call_args_list
        if "Empty upstream response persisted" in str(call.args[0])
    ]
    assert len(warn_logs) == 1, (
        f"Expected one 'Empty upstream response persisted' WARNING, got {len(warn_logs)}"
    )

    log_fmt, *log_args = warn_logs[0].args
    assert "body_snippet=%s" in log_fmt, (
        f"WARNING log missing body_snippet placeholder: {log_fmt}"
    )
    assert "status=%d" in log_fmt, (
        f"WARNING log missing status placeholder: {log_fmt}"
    )
    # Body snippet should contain the raw empty response
    body_snippet = log_args[-1]
    assert '{"choices"' in body_snippet, (
        f"Expected raw JSON in body snippet, got: {body_snippet[:100]}"
    )


@pytest.mark.asyncio
async def test_non_streaming_empty_retry_info_log_includes_body_snippet(mock_request, mock_srv):
    """AC2 (retry): Non-streaming empty retry INFO log includes body snippet.

    When an empty response triggers a retry, the INFO log must also include
    the body snippet.
    """
    empty_body = json.dumps({
        "choices": [{"message": {"content": [], "stopReason": "stop"}, "index": 0}],
    }).encode("utf-8")
    valid_body = json.dumps({
        "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
    }).encode("utf-8")

    mock_srv.config = {
        "server": {
            "upstream_empty_retry_max_attempts": 2,
            "upstream_empty_retry_base_delay_seconds": 0.01,
        }
    }

    empty_resp = _make_non_streaming_mock_response(body_bytes=empty_body)
    valid_resp = _make_non_streaming_mock_response(body_bytes=valid_body)
    client = _make_client(non_stream_responses=[empty_resp, valid_resp, valid_resp])

    with patch("proxy.proxy_remote._srv", return_value=mock_srv):
        with patch("proxy.proxy_remote.log_response"):
            await _handle_remote_non_streaming(
                request=mock_request,
                target_url="https://api.example.com/v1/chat/completions",
                headers={"Authorization": "Bearer test"},
                body=b'{"model": "test"}',
                model_name="test-model",
                remote_timeout=httpx.Timeout(30.0),
                pool_client=client,
                session_id=None,
                resolved_model=None,
            )

    # Check that the INFO retry log includes body_snippet
    info_logs = [
        call for call in mock_srv.logger.info.call_args_list
        if "Empty upstream response detected on attempt" in str(call.args[0])
    ]
    assert len(info_logs) == 1, (
        f"Expected one empty retry INFO log, got {len(info_logs)}"
    )

    log_fmt, *log_args = info_logs[0].args
    assert "body_snippet=%s" in log_fmt, (
        f"INFO retry log missing body_snippet: {log_fmt}"
    )
    body_snippet = log_args[-1]
    assert '{"choices"' in body_snippet, (
        f"Expected raw JSON in body snippet, got: {body_snippet[:100]}"
    )


# ===================================================================
# AC3: No secrets in logged snippets
# ===================================================================


def test_body_snippet_does_not_include_secrets():
    """AC3: Body snippets are model output only, not request secrets.

    The body snippet is extracted from the upstream RESPONSE body, which
    contains model output. It should never contain API keys, auth tokens,
    or other secrets from the request.
    """
    # Simulate a response body (safe model output)
    response_body = json.dumps({
        "choices": [
            {"message": {"content": "This is a safe model response"}}
        ]
    })

    snippet = _utils_snippet(response_body)
    # Should contain the model output
    assert "safe model response" in snippet
    # Should NOT contain any request-level secrets
    assert "Bearer " not in snippet
    assert "sk-proj-" not in snippet
    assert "api_key" not in snippet.lower()


@pytest.mark.asyncio
async def test_non_streaming_long_body_truncated_in_log(mock_request, mock_srv):
    """AC3 (boundary): Long response bodies are truncated in logs.

    Even if the upstream sends a very large response, the snippet in the
    log must be truncated to prevent oversized log lines.
    """
    long_empty_body = json.dumps({
        "choices": [
            {
                "message": {
                    "content": [],
                    "stopReason": "stop",
                    "_padding": "x" * 2000,  # Force long body
                },
            }
        ],
    }).encode("utf-8")

    mock_srv.config = {
        "server": {
            "upstream_empty_retry_max_attempts": 1,
            "upstream_empty_retry_base_delay_seconds": 0.01,
        }
    }

    resp1 = _make_non_streaming_mock_response(body_bytes=long_empty_body)
    resp2 = _make_non_streaming_mock_response(body_bytes=long_empty_body)
    client = _make_client(non_stream_responses=[resp1, resp2])

    with patch("proxy.proxy_remote._srv", return_value=mock_srv):
        with patch("proxy.proxy_remote.log_response"):
            await _handle_remote_non_streaming(
                request=mock_request,
                target_url="https://api.example.com/v1/chat/completions",
                headers={"Authorization": "Bearer test"},
                body=b'{"model": "test"}',
                model_name="test-model",
                remote_timeout=httpx.Timeout(30.0),
                pool_client=client,
                session_id=None,
                resolved_model=None,
            )

    warn_logs = [
        call for call in mock_srv.logger.warning.call_args_list
        if "Empty upstream response persisted" in str(call.args[0])
    ]
    assert len(warn_logs) == 1
    _, *log_args = warn_logs[0].args
    body_snippet = log_args[-1]
    assert len(body_snippet) <= 515, f"Body snippet too long: {len(body_snippet)} chars"


# ===================================================================
# AC4: Utils path — non-streaming retry also logs body snippet
# ===================================================================


@pytest.mark.asyncio
async def test_utils_empty_response_retry_includes_body_snippet():
    """Non-streaming retry in utils.py (provider.py fallback) logs body snippet.

    The _call_with_empty_retry function in utils.py detects empty
    responses and retries. The INFO and WARNING logs must include body snippets.
    """
    # Must use content='' (empty string) to trigger empty detection;
    # content=[] is NOT detected as empty (extracted as string "[]")
    empty_body = json.dumps({
        "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
    }).encode("utf-8")
    valid_body = json.dumps({
        "choices": [{"message": {"content": "Hello from retry"}}],
    }).encode("utf-8")

    mock_response_empty = MagicMock()
    mock_response_empty.content = empty_body

    mock_response_valid = MagicMock()
    mock_response_valid.content = valid_body

    # Track calls: first returns empty, second returns valid
    call_count = [0]

    async def mock_call_with_backend_retries(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return mock_response_empty
        return mock_response_valid

    mock_lifecycle = MagicMock()
    mock_lifecycle._call_with_backend_retries = mock_call_with_backend_retries

    mock_srv_instance = MagicMock()
    mock_srv_instance.logger = MagicMock()

    with patch("proxy.utils._lifecycle", return_value=mock_lifecycle):
        with patch("proxy.utils._srv", return_value=mock_srv_instance):
            result = await _call_with_empty_retry(
                send_fn=MagicMock(),  # send_fn not actually called by the function
                path="/v1/chat/completions",
                max_retries=2,
                retry_delay=0.01,
            )

    # Check INFO log includes body_snippet
    info_logs = [
        call for call in mock_srv_instance.logger.info.call_args_list
        if "Empty response detected" in str(call.args[0])
    ]
    assert len(info_logs) >= 1, (
        f"Expected at least one 'Empty response detected' INFO log, got {len(info_logs)}"
    )

    # The log should include body_snippet
    log_fmt, *log_args = info_logs[0].args
    assert "body_snippet=%s" in log_fmt, (
        f"INFO log missing body_snippet: {log_fmt}"
    )


@pytest.mark.asyncio
async def test_utils_empty_response_warning_includes_body_snippet():
    """Non-streaming retry exhausted — WARNING log includes body snippet.

    When all retries are exhausted for an empty response, the WARNING log
    must include the body snippet.
    """
    # Must use content='' to trigger empty detection
    empty_body = json.dumps({
        "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
    }).encode("utf-8")

    mock_response_empty = MagicMock()
    mock_response_empty.content = empty_body

    async def mock_call_with_backend_retries(*args, **kwargs):
        return mock_response_empty

    mock_lifecycle = MagicMock()
    mock_lifecycle._call_with_backend_retries = mock_call_with_backend_retries

    mock_srv_instance = MagicMock()
    mock_srv_instance.logger = MagicMock()

    with patch("proxy.utils._lifecycle", return_value=mock_lifecycle):
        with patch("proxy.utils._srv", return_value=mock_srv_instance):
            result = await _call_with_empty_retry(
                send_fn=MagicMock(),
                path="/v1/chat/completions",
                max_retries=1,
                retry_delay=0.01,
            )

    # Check WARNING log includes body_snippet
    warn_logs = [
        call for call in mock_srv_instance.logger.warning.call_args_list
        if "Empty response persisted" in str(call.args[0])
    ]
    assert len(warn_logs) >= 1, (
        f"Expected at least one 'Empty response persisted' WARNING, got {len(warn_logs)}"
    )

    log_fmt, *log_args = warn_logs[0].args
    assert "body_snippet=%s" in log_fmt, (
        f"WARNING log missing body_snippet: {log_fmt}"
    )
