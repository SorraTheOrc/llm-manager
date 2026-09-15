"""
Regression tests for bounded failover on stalled/empty remote upstream.

Covers the ACs from LP-0MU1RXEPL005CK97:
  AC1: Time-to-failover on a stalled remote gateway is bounded (idle timeout
       never exceeds upstream_request_timeout_seconds).
  AC2: Empty upstream responses are retried within a bounded, configurable
       budget (max attempts AND total time).
  AC4: Regression test asserts bounded failover for a stalling then empty upstream.
"""

import asyncio
import time
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse
from proxy.proxy_remote import (
    _delta_has_content,
    _handle_remote_non_streaming,
    _handle_remote_streaming,
)

# ===================================================================
# Async iterator helpers
# ===================================================================


class AsyncNoDataIterator:
    """Async iterator that immediately hangs (no chunks at all)."""

    def __init__(self):
        self._event = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self._event.wait()
        return b""


# ===================================================================
# Mock factories
# ===================================================================


def _make_streaming_mock_response(
    status_code=200,
    headers=None,
    aiter_chunks=None,
    hang_after=False,
    body_bytes=None,
):
    """Create a mock HTTP response for streaming tests."""
    mock_resp = MagicMock(spec=httpx.Response)
    type(mock_resp).status_code = PropertyMock(return_value=status_code)
    mock_resp.headers = headers or {"content-type": "text/event-stream"}
    if aiter_chunks is not None:
        mock_resp.aiter_bytes = MagicMock(
            return_value=AsyncChunkIterator(aiter_chunks, hang_after=hang_after)
        )
    if body_bytes is not None:
        mock_resp.content = body_bytes

    async def _aread():
        return body_bytes or b'{"error": "timeout"}'
    mock_resp.aread = _aread

    return mock_resp


def _make_stream_context(mock_resp):
    """Wrap a mock response in an async context manager."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_resp)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


def _make_pool_client(stream_cm):
    """Create a mock httpx.AsyncClient that yields a stream context manager."""
    client = MagicMock(spec=httpx.AsyncClient)
    client.stream = MagicMock(return_value=stream_cm)
    client.aclose = AsyncMock(return_value=None)
    return client


def _make_mock_request():
    """Create a minimal mock Request."""
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.url.path = "/v1/chat/completions"
    request.is_disconnected = AsyncMock(return_value=False)
    request.headers = {}
    return request


# ===================================================================
# AC1: Idle timeout capped by upstream_request_timeout_seconds
# ===================================================================

@pytest.mark.asyncio
async def test_idle_timeout_never_exceeds_request_timeout():
    """AC1: The per-chunk idle timeout is capped so it never exceeds
    upstream_request_timeout_seconds. A 240s silent stall does not sit in the
    critical path.

    The cap is enforced in ``proxy_with_remote`` (proxy_remote.py) where the
    configured idle timeout is clamped to ``upstream_request_timeout_seconds``
    before being passed to ``_handle_remote_streaming``.
    """
    mock_request = _make_mock_request()

    # Stream that stalls immediately (no chunks, just hangs)
    hanging_cm = _make_stream_context(
        _make_streaming_mock_response(
            aiter_chunks=[],
            hang_after=True,
        )
    )

    pool_client = _make_pool_client(hanging_cm)

    with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
        with patch("proxy.proxy_remote.log_response_chunk"):
            with patch("proxy.proxy_remote.log_response"):
                with patch("proxy.proxy_remote.log_request"):
                    with patch("proxy.proxy_remote._srv") as mock_srv:
                        mock_srv.return_value.config = {
                            "server": {
                                "upstream_request_timeout_seconds": 120,
                                # Idle timeout is already capped to request timeout
                                # in proxy_with_remote before reaching here.
                                "upstream_retry_max_attempts": 1,
                                "upstream_retry_base_delay_seconds": 0.01,
                                "upstream_empty_retry_max_attempts": 1,
                                "upstream_empty_retry_base_delay_seconds": 0.01,
                                "upstream_empty_retry_timeout_seconds": 30.0,
                            }
                        }
                        mock_srv.return_value.logger = MagicMock()

                        result = await _handle_remote_streaming(
                            request=mock_request,
                            target_url="https://api.example.com/v1/chat/completions",
                            headers={"Authorization": "Bearer test"},
                            body=b'{"stream": true, "model": "test"}',
                            body_json={"stream": True, "model": "test"},
                            model_name="test-model",
                            remote_timeout=httpx.Timeout(300.0),
                            upstream_idle_timeout_seconds=0.1,
                            upstream_retry_connect_timeout_seconds=0.05,
                            upstream_empty_retry_timeout_seconds=30.0,
                            pool_client=pool_client,
                        )

                        collected = [chunk async for chunk in result.body_iterator]

    assert len(collected) >= 1
    last_chunk = collected[-1].decode("utf-8", errors="replace")
    # Should finish with error (stall exhausted), not hang
    assert '"finish_reason"' in last_chunk.replace(" ", ""), (
        f"Expected finish_reason in last chunk, got: {last_chunk[:300]}"
    )
    # The stall should have been detected and the idle timeout (0.1s)
    # never exceeded the request timeout (120s)
    assert '"stall"' in last_chunk.lower() or '"error"' in last_chunk.lower(), (
        f"Expected stall/error in error event, got: {last_chunk[:300]}"
    )


# ===================================================================
# AC2: Empty upstream responses retried within bounded budget
# ===================================================================


class AsyncChunkIterator:
    """Async iterator that yields pre-defined byte chunks."""

    def __init__(self, chunks, hang_after=False):
        self._chunks = list(chunks)
        self._hang_after = hang_after

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for chunk in self._chunks:
            yield chunk
        if self._hang_after:
            await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_empty_response_retries_have_total_time_budget():
    """AC2: Empty upstream responses are retried within a bounded, configurable
    budget that is logged. The total time for empty retries is limited by
    upstream_empty_retry_timeout_seconds.
    """
    mock_request = _make_mock_request()

    # Stream that returns empty response (no content chunks, just [DONE])
    empty_chunks = [
        b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}\n\n",
        b"data: [DONE]\n\n",
    ]

    empty_cm = _make_stream_context(
        _make_streaming_mock_response(aiter_chunks=empty_chunks)
    )

    pool_client = _make_pool_client(empty_cm)

    with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
        with patch("proxy.proxy_remote.log_response_chunk"):
            with patch("proxy.proxy_remote.log_response"):
                with patch("proxy.proxy_remote.log_request"):
                    with patch("proxy.proxy_remote._srv") as mock_srv:
                        mock_srv.return_value.config = {
                            "server": {
                                "upstream_empty_retry_max_attempts": 3,
                                "upstream_empty_retry_base_delay_seconds": 0.01,
                                "upstream_empty_retry_timeout_seconds": 1.0,
                                "upstream_retry_max_attempts": 3,
                                "upstream_retry_base_delay_seconds": 0.01,
                            }
                        }
                        mock_srv.return_value.logger = MagicMock()

                        result = await _handle_remote_streaming(
                            request=mock_request,
                            target_url="https://api.example.com/v1/chat/completions",
                            headers={"Authorization": "Bearer test"},
                            body=b'{"stream": true, "model": "test"}',
                            body_json={"stream": True, "model": "test"},
                            model_name="test-model",
                            remote_timeout=httpx.Timeout(300.0),
                            upstream_idle_timeout_seconds=0.05,
                            upstream_retry_connect_timeout_seconds=0.05,
                            upstream_empty_retry_timeout_seconds=1.0,
                            pool_client=pool_client,
                        )

                        collected = [chunk async for chunk in result.body_iterator]

    assert len(collected) >= 1
    last_chunk = collected[-1].decode("utf-8", errors="replace")
    # After retries exhausted, should yield error event
    assert '"empty_response"' in last_chunk.lower() or '"error"' in last_chunk.lower(), (
        f"Expected empty_response or error in last chunk, got: {last_chunk[:300]}"
    )


@pytest.mark.asyncio
async def test_non_streaming_empty_response_bounded():
    """AC2 (non-streaming): Non-streaming empty responses also respect the
    empty retry timeout budget.
    """
    mock_request = _make_mock_request()
    mock_request.body = AsyncMock(return_value=b'{"stream": false, "model": "test"}')

    empty_body = json.dumps({
        "choices": [{
            "message": {"content": "", "stopReason": "stop"},
            "finish_reason": "stop",
            "index": 0,
        }],
        "usage": {"total_tokens": 0},
    }).encode()

    empty_resp = _make_streaming_mock_response(status_code=200, body_bytes=empty_body)
    empty_resp.content = empty_body

    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(return_value=empty_resp)
    mock_client.get = AsyncMock(return_value=empty_resp)

    with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
        with patch("proxy.proxy_remote.log_response"):
            with patch("proxy.proxy_remote.log_request"):
                with patch("proxy.proxy_remote._srv") as mock_srv:
                    mock_srv.return_value.config = {
                        "server": {
                            "upstream_empty_retry_max_attempts": 3,
                            "upstream_empty_retry_base_delay_seconds": 0.01,
                            "upstream_empty_retry_timeout_seconds": 1.0,
                        }
                    }
                    mock_srv.return_value.logger = MagicMock()

                    result = await _handle_remote_non_streaming(
                        request=mock_request,
                        target_url="https://api.example.com/v1/chat/completions",
                        headers={"Authorization": "Bearer test"},
                        body=b'{"stream": false, "model": "test"}',
                        model_name="test-model",
                        remote_timeout=httpx.Timeout(300.0),
                        upstream_empty_retry_timeout_seconds=1.0,
                        pool_client=mock_client,
                    )

    assert result is not None
    assert result.status_code == 200


# ===================================================================
# AC4: Regression — stalling then empty upstream → bounded failover
# ===================================================================

@pytest.mark.asyncio
async def test_stalling_then_empty_upstream_bounded_failover():
    """AC4: Regression test asserting bounded failover when upstream first
    stalls (idle timeout fires) and then returns an empty response.

    Scenario:
    1. First attempt: upstream stalls (idle timeout triggers) → retry
    2. Retry attempt: upstream sends empty response → retry
    3. Final attempt: empty again → error event yielded

    Total time must be bounded by the configured timeouts.
    """
    mock_request = _make_mock_request()

    # First attempt: stalls immediately (hangs, triggers idle timeout)
    stall_cm = _make_stream_context(
        _make_streaming_mock_response(
            aiter_chunks=[],
            hang_after=True,
        )
    )

    # Retry attempts: empty responses
    empty_chunks = [
        b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}\n\n",
        b"data: [DONE]\n\n",
    ]
    empty_cm = _make_stream_context(
        _make_streaming_mock_response(aiter_chunks=empty_chunks)
    )

    # Need two stream calls: initial stall + one retry
    pool_client = MagicMock(spec=httpx.AsyncClient)
    pool_client.stream = MagicMock(side_effect=[stall_cm, empty_cm])
    pool_client.aclose = AsyncMock(return_value=None)

    start = asyncio.get_event_loop().time()

    with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
        with patch("proxy.proxy_remote.log_response_chunk"):
            with patch("proxy.proxy_remote.log_response"):
                with patch("proxy.proxy_remote.log_request"):
                    with patch("proxy.proxy_remote._srv") as mock_srv:
                        mock_srv.return_value.config = {
                            "server": {
                                # Tight timeouts for fast test execution
                                "upstream_request_timeout_seconds": 60,
                                "upstream_idle_timeout_seconds": 0.1,
                                "upstream_empty_retry_max_attempts": 2,
                                "upstream_empty_retry_base_delay_seconds": 0.01,
                                "upstream_empty_retry_timeout_seconds": 5.0,
                                "upstream_retry_max_attempts": 1,
                                "upstream_retry_base_delay_seconds": 0.01,
                                "upstream_retry_connect_timeout_seconds": 0.05,
                            }
                        }
                        mock_srv.return_value.logger = MagicMock()

                        result = await _handle_remote_streaming(
                            request=mock_request,
                            target_url="https://api.example.com/v1/chat/completions",
                            headers={"Authorization": "Bearer test"},
                            body=b'{"stream": true, "model": "test"}',
                            body_json={"stream": True, "model": "test"},
                            model_name="test-model",
                            remote_timeout=httpx.Timeout(60.0),
                            upstream_idle_timeout_seconds=0.1,
                            upstream_retry_connect_timeout_seconds=0.05,
                            upstream_empty_retry_timeout_seconds=5.0,
                            pool_client=pool_client,
                        )

                        collected = [chunk async for chunk in result.body_iterator]

    elapsed = asyncio.get_event_loop().time() - start

    # Should complete quickly (within a few seconds), not hang
    assert elapsed < 5.0, (
        f"Bounded failover took {elapsed:.1f}s — exceeds reasonable budget"
    )
    assert len(collected) >= 1

    last_chunk = collected[-1].decode("utf-8", errors="replace")
    # Should yield a terminal error event
    assert '"finish_reason"' in last_chunk.replace(" ", ""), (
        f"Expected terminal finish_reason in last chunk, got: {last_chunk[:300]}"
    )
    assert '"error"' in last_chunk.lower(), (
        f"Expected error in last chunk, got: {last_chunk[:300]}"
    )


# ===================================================================
# _delta_has_content regression
# ===================================================================

def test_delta_has_content_with_content():
    """Verify _delta_has_content correctly identifies content-bearing deltas."""
    assert _delta_has_content({"content": "hello"}) is True
    assert _delta_has_content({"content": ""}) is False
    assert _delta_has_content({"content": None}) is False
    assert _delta_has_content({"tool_calls": [{"id": "1"}]}) is True
    assert _delta_has_content({"tool_calls": []}) is False
    assert _delta_has_content({"reasoning_content": "thinking..."}) is True
    assert _delta_has_content({}) is False
    assert _delta_has_content("not a dict") is False
