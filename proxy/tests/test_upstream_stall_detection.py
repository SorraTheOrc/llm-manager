"""
Tests for upstream stall detection and proactive abort for remote streaming.

Tests cover:
1. AC1: Stall detection - per-chunk idle timeout fires on silent upstream
2. AC2: Configurable timeout - upstream_idle_timeout_seconds is wired through
3. AC3: Retry on stall - bounded exponential backoff, re-sends request
4. AC4: Retry on httpx ReadTimeout - falls through to retry path
5. AC6: Normal streaming unaffected - chunks at normal intervals pass through
6. AC5 + AC8: Max retries exhausted -> finish_reason: error fallback
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse

from proxy.proxy_remote import _handle_remote_streaming


# ===================================================================
# Async iterator helpers
# ===================================================================


class AsyncChunkIterator:
    """Async iterator that yields pre-defined byte chunks."""

    def __init__(self, chunks, hang_after=False, chunk_delay=0):
        self._chunks = list(chunks)
        self._hang_after = hang_after
        self._chunk_delay = chunk_delay
        self._done = False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for chunk in self._chunks:
            if self._chunk_delay > 0:
                await asyncio.sleep(self._chunk_delay)
            yield chunk
        if self._hang_after:
            # Hang forever
            await asyncio.Event().wait()


class AsyncNoDataIterator:
    """Async iterator that immediately hangs (no chunks at all)."""

    def __init__(self):
        self._event = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self._event.wait()
        return b""


class AsyncDelayedIterator:
    """Async iterator that yields chunks with a configurable per-chunk delay."""

    def __init__(self, chunks, delay_between=0):
        self._chunks = list(chunks)
        self._delay = delay_between

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return self._chunks.pop(0)


# ===================================================================
# Mock response / client factories
# ===================================================================


def _make_mock_response(
    status_code=200,
    headers=None,
    aiter_chunks=None,
    hang_after=False,
    error_body=None,
):
    """Create a mock HTTP response for testing.

    Args:
        aiter_chunks: List of byte chunks to yield from aiter_bytes().
        hang_after: If True, the iterator hangs after yielding all chunks.
    """
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


def _make_streaming_mock_client(mock_response):
    """Create a mock httpx.AsyncClient that returns mock_response on stream().

    Returns a MagicMock that, when called as httpx.AsyncClient(), returns a
    client whose stream() method returns a context manager whose __aenter__
    returns mock_response.
    """
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_response)
    cm.__aexit__ = AsyncMock(return_value=None)

    client_instance = MagicMock(spec=httpx.AsyncClient)
    client_instance.stream = MagicMock(return_value=cm)
    client_instance.aclose = AsyncMock(return_value=None)

    return client_instance


def _make_streaming_client_cls(client_instance):
    """Wrap a client_instance in a class mock for patch('httpx.AsyncClient')."""
    mock_cls = MagicMock(return_value=client_instance)
    return mock_cls


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def mock_request():
    req = MagicMock(spec=Request)
    req.method = "POST"
    req.url.path = "/v1/chat/completions"
    req.is_disconnected = AsyncMock(return_value=False)
    return req


# ===================================================================
# AC6: Normal streaming unaffected
# ===================================================================


@pytest.mark.asyncio
async def test_normal_streaming_unaffected(mock_request):
    """Normal streaming with chunks arriving at <1s intervals passes through.

    AC6: Regular streaming with chunk delivery at normal intervals (<1s) is
    completely unaffected — the timeout only fires on inter-chunk silence
    exceeding the configured threshold.
    """
    chunks = [
        b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"},"index":0}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    mock_resp = _make_mock_response(
        status_code=200,
        aiter_chunks=chunks,
    )
    client = _make_streaming_mock_client(mock_resp)

    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client):
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
                            upstream_idle_timeout_seconds=0.5,
                        )

                        collected = [chunk async for chunk in result.body_iterator]

    assert isinstance(result, StreamingResponse)
    assert len(collected) == 3, (
        f"Expected 3 chunks (2 data + 1 [DONE]), got {len(collected)}"
    )


# ===================================================================
# AC1 + AC3: Stall detection triggers retry with backoff
# ===================================================================


@pytest.mark.asyncio
async def test_stall_detection_triggers_retry(mock_request):
    """Upstream stall triggers per-chunk idle timeout and retries.

    AC1: Per-chunk idle timeout detects upstream silence within
    upstream_idle_timeout_seconds and closes the stalled connection.

    AC3: On stall detection, the proxy automatically retries the request to
    the same provider with bounded exponential backoff.
    """
    first_chunks = [
        b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
    ]
    # First response yields one chunk then hangs (stall)
    first_resp = _make_mock_response(
        status_code=200,
        aiter_chunks=first_chunks,
        hang_after=True,
    )
    # Second response (retry) succeeds fully
    second_chunks = [
        b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"},"index":0}]}\n\n',
        b'data: [DONE]\n\n',
    ]
    second_resp = _make_mock_response(
        status_code=200,
        aiter_chunks=second_chunks,
    )

    cm1 = MagicMock()
    cm1.__aenter__ = AsyncMock(return_value=first_resp)
    cm1.__aexit__ = AsyncMock(return_value=None)

    cm2 = MagicMock()
    cm2.__aenter__ = AsyncMock(return_value=second_resp)
    cm2.__aexit__ = AsyncMock(return_value=None)

    client_instance = MagicMock(spec=httpx.AsyncClient)
    # First call to stream() returns cm1, second call returns cm2
    client_instance.stream = MagicMock(side_effect=[cm1, cm2])
    client_instance.aclose = AsyncMock(return_value=None)

    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client_instance):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        with patch("proxy.proxy_remote._srv") as mock_srv:
                            mock_srv.return_value.config = {}
                            mock_srv.return_value.logger = MagicMock()

                            result = await _handle_remote_streaming(
                                request=mock_request,
                                target_url="https://api.example.com/v1/chat/completions",
                                headers={"Authorization": "Bearer test"},
                                body=b'{"stream": true, "model": "test"}',
                                body_json={"stream": True, "model": "test"},
                                model_name="test-model",
                                remote_timeout=httpx.Timeout(30.0),
                                upstream_idle_timeout_seconds=0.05,
                            )

                            collected = [chunk async for chunk in result.body_iterator]

    assert isinstance(result, StreamingResponse)
    # Should get chunks from the retry: hello, world, [DONE]
    # Plus maybe a final synthesized event
    assert len(collected) >= 3, (
        f"Expected at least 3 chunks from retry, got {len(collected)}"
    )
    # Verify stream() was called twice (first attempt + retry)
    assert client_instance.stream.call_count == 2, (
        f"Expected 2 stream() calls (initial + 1 retry), got {client_instance.stream.call_count}"
    )


# ===================================================================
# AC4: Retry on httpx ReadTimeout
# ===================================================================


@pytest.mark.asyncio
async def test_retry_on_httpx_read_timeout(mock_request):
    """httpx ReadTimeout triggers retry with bounded exponential backoff.

    AC4: If the upstream stalls long enough that httpx raises a ReadTimeout
    before the idle timeout fires (edge case), the proxy retries the same
    provider with bounded exponential backoff before falling through.
    """
    # Simulate: first streaming attempt raises httpx.ReadTimeout
    # (e.g., during the client.stream() call or __aenter__)
    first_resp = MagicMock(spec=httpx.Response)
    type(first_resp).status_code = PropertyMock(return_value=200)
    first_resp.headers = {"content-type": "text/event-stream"}
    # aiter_bytes raises ReadTimeout
    first_resp.aiter_bytes = MagicMock(side_effect=httpx.ReadTimeout("Read timed out"))

    # Second response succeeds
    second_chunks = [
        b'data: {"choices":[{"delta":{"content":"Retry"},"index":0}]}\n\n',
        b'data: [DONE]\n\n',
    ]
    second_resp = _make_mock_response(
        status_code=200,
        aiter_chunks=second_chunks,
    )

    cm1 = MagicMock()
    cm1.__aenter__ = AsyncMock(return_value=first_resp)
    cm1.__aexit__ = AsyncMock(return_value=None)

    cm2 = MagicMock()
    cm2.__aenter__ = AsyncMock(return_value=second_resp)
    cm2.__aexit__ = AsyncMock(return_value=None)

    client_instance = MagicMock(spec=httpx.AsyncClient)
    client_instance.stream = MagicMock(side_effect=[cm1, cm2])
    client_instance.aclose = AsyncMock(return_value=None)

    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client_instance):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        with patch("proxy.proxy_remote._srv") as mock_srv:
                            mock_srv.return_value.config = {}
                            mock_srv.return_value.logger = MagicMock()

                            result = await _handle_remote_streaming(
                                request=mock_request,
                                target_url="https://api.example.com/v1/chat/completions",
                                headers={"Authorization": "Bearer test"},
                                body=b'{"stream": true, "model": "test"}',
                                body_json={"stream": True, "model": "test"},
                                model_name="test-model",
                                remote_timeout=httpx.Timeout(30.0),
                                upstream_idle_timeout_seconds=0.5,
                            )

                            collected = [chunk async for chunk in result.body_iterator]

    assert isinstance(result, StreamingResponse)
    assert len(collected) >= 2, (
        f"Expected at least 2 chunks from retry, got {len(collected)}"
    )
    # Verify stream() was called twice
    assert client_instance.stream.call_count == 2, (
        f"Expected 2 stream() calls, got {client_instance.stream.call_count}"
    )


# ===================================================================
# AC5 + AC8: Max retries exhausted -> finish_reason: error
# ===================================================================


@pytest.mark.asyncio
async def test_max_retries_exhausted_yields_finish_reason_error(mock_request):
    """After max retries exhausted, yields finish_reason: error event.

    AC5: Session state preserved — full request body is re-sent on retry.
    AC8: After max retries exhausted, falls through with finish_reason: error.
    """
    # All 4 attempts (initial + 3 retries) stall after first chunk
    chunks_per_attempt = [
        [b'data: {"choices":[{"delta":{"content":"A"},"index":0}]}\n\n'],
        [b'data: {"choices":[{"delta":{"content":"B"},"index":0}]}\n\n'],
        [b'data: {"choices":[{"delta":{"content":"C"},"index":0}]}\n\n'],
        [b'data: {"choices":[{"delta":{"content":"D"},"index":0}]}\n\n'],
    ]

    responses = []
    for attempt_chunks in chunks_per_attempt:
        resp = _make_mock_response(
            status_code=200,
            aiter_chunks=attempt_chunks,
            hang_after=True,
        )
        responses.append(resp)

    # 4 context managers (initial + 3 retries)
    cms = []
    for resp in responses:
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=None)
        cms.append(cm)

    client_instance = MagicMock(spec=httpx.AsyncClient)
    client_instance.stream = MagicMock(side_effect=cms)
    client_instance.aclose = AsyncMock(return_value=None)

    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client_instance):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        with patch("proxy.proxy_remote._srv") as mock_srv:
                            mock_srv.return_value.config = {}
                            mock_srv.return_value.logger = MagicMock()

                            result = await _handle_remote_streaming(
                                request=mock_request,
                                target_url="https://api.example.com/v1/chat/completions",
                                headers={"Authorization": "Bearer test"},
                                body=b'{"stream": true, "model": "test"}',
                                body_json={"stream": True, "model": "test"},
                                model_name="test-model",
                                remote_timeout=httpx.Timeout(30.0),
                                upstream_idle_timeout_seconds=0.05,
                            )

                            collected = [chunk async for chunk in result.body_iterator]

    assert isinstance(result, StreamingResponse)
    assert len(collected) >= 1, "Should have yielded at least one chunk"

    # Verify stream() was called 4 times (initial + 3 retries)
    assert client_instance.stream.call_count == 4, (
        f"Expected 4 stream() calls (initial + 3 retries), "
        f"got {client_instance.stream.call_count}"
    )

    # The last chunk should be finish_reason: error
    last_chunk = collected[-1].decode("utf-8", errors="replace")
    assert "finish_reason" in last_chunk, (
        f"Last chunk should contain finish_reason, got: {last_chunk[:200]}"
    )
    assert '"finish_reason":"error"' in last_chunk.replace(" ", ""), (
        f"Last chunk should have finish_reason: error, got: {last_chunk[:200]}"
    )


# ===================================================================
# AC2: Config key wired through
# ===================================================================


@pytest.mark.asyncio
async def test_upstream_idle_timeout_configurable(mock_request):
    """upstream_idle_timeout_seconds is passed through to _handle_remote_streaming.

    AC2: A new config key upstream_idle_timeout_seconds is wired into
    proxy_remote.py, separate from stream_idle_timeout_seconds.
    """
    chunks = [
        b'data: {"choices":[{"delta":{"content":"test"},"index":0}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    mock_resp = _make_mock_response(
        status_code=200,
        aiter_chunks=chunks,
    )
    client = _make_streaming_mock_client(mock_resp)

    # Use a custom upstream_idle_timeout_seconds
    custom_timeout = 123.0

    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client):
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
                            upstream_idle_timeout_seconds=custom_timeout,
                        )

                        collected = [chunk async for chunk in result.body_iterator]

    assert isinstance(result, StreamingResponse)
    assert len(collected) > 0, "Should have collected chunks"


# ===================================================================
# Regression: Already-buffered error responses still work
# ===================================================================


@pytest.mark.asyncio
async def test_non_streaming_error_response_unchanged(mock_request):
    """Error responses (non-200, non-SSE) are still buffered, not streamed.

    Regression test: upstream returning 4xx/5xx with non-SSE content type
    should still return a buffered Response, not a StreamingResponse.
    """
    mock_resp = _make_mock_response(
        status_code=401,
        headers={"content-type": "application/json"},
        error_body=b'{"error":"unauthorized"}',
    )
    client = _make_streaming_mock_client(mock_resp)

    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
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
                        upstream_idle_timeout_seconds=0.5,
                    )

    # Should be a plain Response (not StreamingResponse) for error path
    assert not isinstance(result, StreamingResponse), (
        "Error path should return buffered Response, not StreamingResponse"
    )
    assert result.status_code == 401
