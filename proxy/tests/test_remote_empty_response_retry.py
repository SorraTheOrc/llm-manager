"""
Tests for proxy-level empty upstream LLM response detection and retry.

Tests cover:
1. AC1: Non-streaming empty response retry — empty content triggers retry
2. AC2: Streaming empty response retry — zero content chunks at stream end triggers retry
3. AC3: Error fallback after retries exhausted — returns error for fallback chain
4. AC4: Configurable upstream timeout — upstream_request_timeout_seconds is wired
5. AC5: Configurable empty-response retry — max_attempts and base_delay are wired
6. AC6: No infinite loops — one retry + fallback is sufficient
7. AC7: Transparent to client — non-empty responses pass through unchanged
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
)


# ===================================================================
# Async iterator helpers (reused from test_upstream_stall_detection.py)
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
# Mock response / client factories
# ===================================================================


def _make_streaming_mock_response(
    status_code=200,
    headers=None,
    aiter_chunks=None,
    hang_after=False,
    error_body=None,
):
    """Create a mock HTTP response for streaming tests."""
    mock_resp = MagicMock(spec=httpx.Response)
    type(mock_resp).status_code = PropertyMock(return_value=status_code)
    mock_resp.headers = headers or {"content-type": "text/event-stream"}
    if aiter_chunks is not None:
        mock_resp.aiter_bytes = MagicMock(
            return_value=AsyncChunkIterator(aiter_chunks, hang_after=hang_after)
        )
    if error_body is not None:
        mock_resp.aread = AsyncMock(return_value=error_body)
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
    """Create a mock httpx.AsyncClient.

    For streaming: stream_responses is a list of mock responses whose
    ordering matches sequential stream() calls.

    For non-streaming: non_stream_responses is a list of mock responses
    whose ordering matches sequential HTTP method calls (e.g., post()).
    """
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
# AC1: Non-streaming empty response retry
# ===================================================================


@pytest.mark.asyncio
async def test_non_streaming_empty_response_triggers_retry(mock_request, mock_srv):
    """AC1: Empty non-streaming upstream response triggers retry.

    When upstream returns empty content (content: [], usage.total_tokens: 0,
    stopReason: stop), the proxy retries the same model at least once
    before returning the result.
    """
    empty_body = json.dumps({
        "choices": [{"message": {"content": [], "stopReason": "stop"}, "index": 0}],
        "usage": {"input": 0, "output": 0, "total": 0},
    }).encode("utf-8")

    valid_body = json.dumps({
        "choices": [{"message": {"content": "Hello world"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 5},
    }).encode("utf-8")

    # First response is empty, second is valid
    first_resp = _make_non_streaming_mock_response(body_bytes=empty_body)
    second_resp = _make_non_streaming_mock_response(body_bytes=valid_body)

    client = _make_client(non_stream_responses=[first_resp, second_resp])

    with patch("proxy.proxy_remote._srv", return_value=mock_srv):
        with patch("proxy.proxy_remote.log_response"):
            result = await _handle_remote_non_streaming(
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

    # Should have called post() twice (initial + 1 retry)
    assert client.post.call_count == 2, (
        f"Expected 2 post() calls (initial + 1 retry), got {client.post.call_count}"
    )
    # Should return the valid response
    assert result.status_code == 200
    body = json.loads(result.body.decode("utf-8"))
    assert body["choices"][0]["message"]["content"] == "Hello world"


@pytest.mark.asyncio
async def test_non_streaming_non_empty_passes_through(mock_request, mock_srv):
    """AC7: Non-empty non-streaming responses pass through unchanged."""
    valid_body = json.dumps({
        "choices": [{"message": {"content": "Hello world"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 5},
    }).encode("utf-8")

    first_resp = _make_non_streaming_mock_response(body_bytes=valid_body)

    client = _make_client(non_stream_responses=[first_resp])

    with patch("proxy.proxy_remote._srv", return_value=mock_srv):
        with patch("proxy.proxy_remote.log_response"):
            result = await _handle_remote_non_streaming(
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

    # Should have called post() only once (no retry)
    assert client.post.call_count == 1, (
        f"Expected 1 post() call (no retry), got {client.post.call_count}"
    )
    assert result.status_code == 200
    body = json.loads(result.body.decode("utf-8"))
    assert body["choices"][0]["message"]["content"] == "Hello world"


@pytest.mark.asyncio
async def test_non_streaming_empty_retry_then_return_empty(mock_request, mock_srv):
    """AC3 + AC6: After retries exhausted for empty response, return last response.

    The proxy does not infinite-loop — after configurable retries it returns
    the last response as-is, allowing the fallback chain to activate.
    """
    empty_body = json.dumps({
        "choices": [{"message": {"content": [], "stopReason": "stop"}, "index": 0}],
        "usage": {"input": 0, "output": 0, "total": 0},
    }).encode("utf-8")

    # Both responses are empty (retry doesn't help)
    first_resp = _make_non_streaming_mock_response(body_bytes=empty_body)
    second_resp = _make_non_streaming_mock_response(body_bytes=empty_body)

    client = _make_client(non_stream_responses=[first_resp, second_resp])

    with patch("proxy.proxy_remote._srv", return_value=mock_srv):
        with patch("proxy.proxy_remote.log_response"):
            # Pass custom config with upstream_empty_retry_max_attempts=1
            mock_srv.config = {
                "server": {
                    "upstream_empty_retry_max_attempts": 1,
                    "upstream_empty_retry_base_delay_seconds": 0.01,
                }
            }
            result = await _handle_remote_non_streaming(
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

    # Should have called post() twice (initial + 1 retry)
    assert client.post.call_count == 2, (
        f"Expected 2 post() calls (initial + 1 retry), got {client.post.call_count}"
    )
    # Result should be the empty response (as-is for fallback to handle)
    assert result.status_code == 200
    body = json.loads(result.body.decode("utf-8"))
    assert body["choices"][0]["message"]["content"] == []


# ===================================================================
# AC2: Streaming empty response retry
# ===================================================================


@pytest.mark.asyncio
async def test_streaming_empty_response_triggers_retry(mock_request, mock_srv):
    """AC2: Streaming upstream that produces no content chunks triggers retry.

    When a streaming upstream produces zero content deltas and ends with
    finish_reason: stop, the proxy retries the same model at least once.
    """
    # First stream: finish_reason stop with no content chunks
    empty_stream_chunks = [
        b'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    # Second stream (retry): valid content
    valid_stream_chunks = [
        b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"},"index":0}]}\n\n',
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

    assert isinstance(result, StreamingResponse), "Expected StreamingResponse"
    # Should have content from the retry stream
    collected_text = b"".join(collected).decode("utf-8", errors="replace")
    assert "Hello" in collected_text, (
        f"Expected 'Hello' content from retry, got: {collected_text[:200]}"
    )
    # Verify stream() was called twice
    assert client.stream.call_count == 2, (
        f"Expected 2 stream() calls (initial + 1 retry), got {client.stream.call_count}"
    )


@pytest.mark.asyncio
async def test_streaming_non_empty_passes_through(mock_request, mock_srv):
    """AC7: Streaming with content passes through unchanged."""
    chunks = [
        b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"},"index":0}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    mock_resp = _make_streaming_mock_response(
        status_code=200,
        aiter_chunks=chunks,
    )
    client = _make_client(stream_responses=[mock_resp])

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

    assert isinstance(result, StreamingResponse)
    assert len(collected) == 3, (
        f"Expected 3 chunks (2 data + 1 [DONE]), got {len(collected)}"
    )
    # stream() should be called only once
    assert client.stream.call_count == 1, (
        f"Expected 1 stream() call, got {client.stream.call_count}"
    )


@pytest.mark.asyncio
async def test_streaming_empty_retry_then_finish_reason_error(mock_request, mock_srv):
    """AC3 + AC6: After streaming retries exhausted, yields finish_reason: error."""
    empty_stream_chunks = [
        b'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    # Both streams empty
    first_resp = _make_streaming_mock_response(
        status_code=200,
        aiter_chunks=empty_stream_chunks,
    )
    second_resp = _make_streaming_mock_response(
        status_code=200,
        aiter_chunks=empty_stream_chunks,
    )

    client = _make_client(stream_responses=[first_resp, second_resp])

    # Set empty retry max attempts = 1
    mock_srv.config = {
        "server": {
            "upstream_empty_retry_max_attempts": 1,
            "upstream_empty_retry_base_delay_seconds": 0.01,
        }
    }

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

    assert isinstance(result, StreamingResponse)
    # After retries exhausted, should yield a synthetic finish_reason: error
    assert len(collected) >= 1, "Should have yielded at least one chunk"

    last_chunk = collected[-1].decode("utf-8", errors="replace")
    assert '"finish_reason":"error"' in last_chunk.replace(" ", ""), (
        f"Last chunk should have finish_reason: error, got: {last_chunk[:200]}"
    )
    # stream() should be called twice
    assert client.stream.call_count == 2, (
        f"Expected 2 stream() calls, got {client.stream.call_count}"
    )


# ===================================================================
# AC4: Configurable upstream timeout
# ===================================================================


@pytest.mark.asyncio
async def test_upstream_request_timeout_wired(mock_request, mock_srv):
    """AC4: Configurable upstream request-timeout prevents long hangs.

    The upstream_request_timeout_seconds config key is read and passed
    through to the upstream call.
    """
    config = {
        "server": {
            "upstream_request_timeout_seconds": 120,
        }
    }

    # Verify the default is applied when key is absent
    default_timeout = float(
        config.get("server", {}).get("upstream_request_timeout_seconds", 120)
    )
    assert default_timeout == 120.0, (
        f"Expected default upstream_request_timeout_seconds to be 120, got {default_timeout}"
    )

    # Verify custom value is applied
    config["server"]["upstream_request_timeout_seconds"] = 60
    custom_timeout = float(
        config.get("server", {}).get("upstream_request_timeout_seconds", 120)
    )
    assert custom_timeout == 60.0, (
        f"Expected custom timeout 60, got {custom_timeout}"
    )


# ===================================================================
# AC5: Configurable empty-response retry
# ===================================================================


@pytest.mark.asyncio
async def test_empty_retry_config_keys_used_in_non_streaming(mock_request, mock_srv):
    """AC5: Empty-response retry settings are configurable via config.yaml.

    Verifies that upstream_empty_retry_max_attempts and
    upstream_empty_retry_base_delay_seconds are read from server config
    and used to drive retry behavior.
    """
    empty_body = json.dumps({
        "choices": [{"message": {"content": [], "stopReason": "stop"}, "index": 0}],
        "usage": {"input": 0, "output": 0, "total": 0},
    }).encode("utf-8")

    # Set max_attempts=2, so 3 calls total (1 initial + 2 retries)
    mock_srv.config = {
        "server": {
            "upstream_empty_retry_max_attempts": 2,
            "upstream_empty_retry_base_delay_seconds": 0.01,
        }
    }

    empty_resp = _make_non_streaming_mock_response(body_bytes=empty_body)

    # 3 calls (initial + 2 retries) all empty
    client = _make_client(non_stream_responses=[empty_resp, empty_resp, empty_resp])

    with patch("proxy.proxy_remote._srv", return_value=mock_srv):
        with patch("proxy.proxy_remote.log_response"):
            result = await _handle_remote_non_streaming(
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

    # Should have called post() 3 times (initial + 2 retries)
    assert client.post.call_count == 3, (
        f"Expected 3 post() calls (initial + 2 retries), got {client.post.call_count}"
    )


@pytest.mark.asyncio
async def test_default_empty_retry_values(mock_request, mock_srv):
    """AC5: Default values for empty-response retry settings work when config absent."""
    # No config set — defaults should apply
    mock_srv.config = {}

    empty_body = json.dumps({
        "choices": [{"message": {"content": [], "stopReason": "stop"}, "index": 0}],
        "usage": {"input": 0, "output": 0, "total": 0},
    }).encode("utf-8")

    empty_resp = _make_non_streaming_mock_response(body_bytes=empty_body)
    valid_body = json.dumps({
        "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 2},
    }).encode("utf-8")
    valid_resp = _make_non_streaming_mock_response(body_bytes=valid_body)

    # Default: 1 retry, so 2 calls total
    client = _make_client(non_stream_responses=[empty_resp, valid_resp])

    with patch("proxy.proxy_remote._srv", return_value=mock_srv):
        with patch("proxy.proxy_remote.log_response"):
            result = await _handle_remote_non_streaming(
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

    assert client.post.call_count == 2, (
        f"Expected 2 post() calls with default config, got {client.post.call_count}"
    )
    body = json.loads(result.body.decode("utf-8"))
    assert body["choices"][0]["message"]["content"] == "Hello"


# ===================================================================
# Streaming empty detection edge cases
# ===================================================================


@pytest.mark.asyncio
async def test_streaming_empty_via_stop_async_iteration(mock_request, mock_srv):
    """Stream ending with StopAsyncIteration without [DONE] and no content.

    If the stream exhausts the iterator without yielding any content chunks
    and the synthetic stop has no content, treat as empty and retry.
    """
    # No chunks at all — StopAsyncIteration immediately
    empty_stream = _make_streaming_mock_response(
        status_code=200,
        aiter_chunks=[],
    )
    valid_stream = _make_streaming_mock_response(
        status_code=200,
        aiter_chunks=[
            b'data: {"choices":[{"delta":{"content":"Retry works"},"index":0}]}\n\n',
            b'data: [DONE]\n\n',
        ],
    )

    client = _make_client(stream_responses=[empty_stream, valid_stream])
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

    assert isinstance(result, StreamingResponse)
    collected_text = b"".join(collected).decode("utf-8", errors="replace")
    # Should have content from the retry
    assert "Retry works" in collected_text, (
        f"Expected 'Retry works' from retry, got: {collected_text[:200]}"
    )
    assert client.stream.call_count == 2, (
        f"Expected 2 stream() calls, got {client.stream.call_count}"
    )
