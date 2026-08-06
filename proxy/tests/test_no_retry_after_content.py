"""
Tests for the content-aware remote streaming retry policy (LP-0MS9FR9LG002AJ4C).

Context: Pi sessions intermittently fail with `Error: Provider finish_reason:
error` after the proxy's Tier-1 upstream-retry chain exhausts mid-stream. Slow
reasoning upstreams (opencode-go / deepseek-v4-flash) can pause >30s between
chunks while still being alive; the proxy misclassified those pauses as stalls,
re-sent the whole multi-hundred-KB request up to 3 more times, and finally
surfaced a synthetic `finish_reason: error`.

Fix (this work item):
1. `upstream_idle_timeout_seconds` raised 30 -> 120 -> 240 (config + code default) so
   long-but-alive reasoning pauses no longer misfire stall detection.
2. Tier-1 retries only occur while ZERO content-bearing chunks have been
   delivered. Once any content has been sent to the client, a stall (idle
   timeout or httpx ReadTimeout) terminates the stream immediately with a
   synthetic `finish_reason: error` — the whole request is never re-sent, so
   failure time stays bounded and the client can retry with full context.

Tests cover:
(a) idle-timeout tuning behavior (long chunk gaps tolerated),
(b) no-retry-after-content policy,
(c) retry-before-content still works,
(d) config defaults (code-level fallback is 240),
(e) content-detection edge cases (reasoning_content / tool_calls count as
    content, so a stall after those also does not retry),
(f) Tier-3 circuit breaker still records after-content stalls.
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
    """Async iterator that yields pre-defined byte chunks, optionally with an
    inter-chunk delay, then optionally hangs forever."""

    def __init__(self, chunks, hang_after=False, chunk_delay=0):
        self._chunks = list(chunks)
        self._hang_after = hang_after
        self._chunk_delay = chunk_delay

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for chunk in self._chunks:
            if self._chunk_delay > 0:
                await asyncio.sleep(self._chunk_delay)
            yield chunk
        if self._hang_after:
            await asyncio.Event().wait()


class ErrorAsyncIterator:
    """Async iterator that yields chunks then raises on the next anext call."""

    def __init__(self, chunks, exc):
        self._chunks = list(chunks)
        self._exc = exc

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._chunks:
            return self._chunks.pop(0)
        raise self._exc


# ===================================================================
# Mock response / client factories
# ===================================================================


def _make_mock_response(
    status_code=200,
    headers=None,
    aiter_chunks=None,
    hang_after=False,
    chunk_delay=0,
):
    """Create a mock HTTP response for testing."""
    mock_resp = MagicMock(spec=httpx.Response)
    type(mock_resp).status_code = PropertyMock(return_value=status_code)
    mock_resp.headers = headers or {"content-type": "text/event-stream"}
    if aiter_chunks is not None:
        mock_resp.aiter_bytes = MagicMock(
            return_value=AsyncChunkIterator(
                aiter_chunks, hang_after=hang_after, chunk_delay=chunk_delay
            )
        )
    return mock_resp


def _make_error_response(chunks, exc_cls):
    """Create a mock response whose aiter_bytes raises after yielding chunks."""
    mock_resp = MagicMock(spec=httpx.Response)
    type(mock_resp).status_code = PropertyMock(return_value=200)
    mock_resp.headers = {"content-type": "text/event-stream"}
    mock_resp.aiter_bytes = MagicMock(
        return_value=ErrorAsyncIterator(chunks, exc_cls("simulated"))
    )
    return mock_resp


def _make_streaming_client(responses):
    """Create a mock httpx.AsyncClient whose stream() returns the given
    responses in order (one context manager per response)."""
    cms = []
    for resp in responses:
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=None)
        cms.append(cm)
    client_instance = MagicMock(spec=httpx.AsyncClient)
    client_instance.stream = MagicMock(side_effect=cms)
    client_instance.aclose = AsyncMock(return_value=None)
    return client_instance


def _content_chunk(text):
    return (
        f'data: {json.dumps({"choices": [{"delta": {"content": text}, "index": 0}]})}\n\n'
    ).encode()


def _reasoning_chunk(text):
    return (
        f'data: {json.dumps({"choices": [{"delta": {"reasoning_content": text}, "index": 0}]})}\n\n'
    ).encode()


def _tool_calls_chunk():
    delta = {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
            }
        ]
    }
    return (
        f'data: {json.dumps({"choices": [{"delta": delta, "index": 0}]})}\n\n'
    ).encode()


def _done_chunk():
    return b"data: [DONE]\n\n"


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


async def _run_streaming(client, mock_request, **kwargs):
    """Invoke _handle_remote_streaming with sensible defaults and return
    (result, collected_chunks)."""
    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client):
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
                                **kwargs,
                            )
                            collected = [
                                chunk async for chunk in result.body_iterator
                            ]
    return result, collected


def _last_finish_reason(collected):
    """Extract the last finish_reason value present in the collected chunks."""
    reasons = []
    for chunk in collected:
        s = chunk.decode("utf-8", errors="replace")
        for line in s.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            try:
                j = json.loads(payload)
                for choice in j.get("choices", []):
                    if choice.get("finish_reason") is not None:
                        reasons.append(choice["finish_reason"])
            except Exception:
                pass
    return reasons[-1] if reasons else None


# ===================================================================
# (b) No-retry-after-content policy
# ===================================================================


@pytest.mark.asyncio
async def test_no_retry_after_content_delivered(mock_request):
    """A stall after a content-bearing chunk terminates the stream immediately.

    AC: When a remote stream stalls after at least one content-bearing chunk
    has been delivered, the proxy does NOT restart the whole request; it
    terminates with a synthetic finish_reason: error so the client can retry
    with full context.
    """
    first_resp = _make_mock_response(
        status_code=200,
        aiter_chunks=[_content_chunk("Hello")],
        hang_after=True,
    )
    client = _make_streaming_client([first_resp])

    result, collected = await _run_streaming(client, mock_request)

    assert isinstance(result, StreamingResponse)
    assert len(collected) >= 2, (
        f"Expected content chunk + finish_reason: error, got {len(collected)} chunks"
    )
    # The whole request must NOT be re-sent once content was delivered.
    assert client.stream.call_count == 1, (
        f"Expected 1 stream() call (no retry after content), "
        f"got {client.stream.call_count}"
    )
    assert _last_finish_reason(collected) == "error", (
        f"Expected final finish_reason 'error', got {_last_finish_reason(collected)}"
    )
    # Content delivered before the error must be preserved.
    assert b"Hello" in collected[0], "Original content should be preserved"


@pytest.mark.asyncio
async def test_no_retry_after_reasoning_content_delivered(mock_request):
    """A stall after a reasoning_content chunk does not retry.

    Reasoning-only streams must be treated as content (LP-0MS8XAPXT009W3CL);
    a stall after reasoning output therefore terminates immediately instead
    of re-sending the whole request.
    """
    first_resp = _make_mock_response(
        status_code=200,
        aiter_chunks=[_reasoning_chunk("thinking...")],
        hang_after=True,
    )
    client = _make_streaming_client([first_resp])

    result, collected = await _run_streaming(client, mock_request)

    assert isinstance(result, StreamingResponse)
    assert client.stream.call_count == 1, (
        f"Expected 1 stream() call (no retry after reasoning content), "
        f"got {client.stream.call_count}"
    )
    assert _last_finish_reason(collected) == "error"


@pytest.mark.asyncio
async def test_no_retry_after_tool_calls_delivered(mock_request):
    """A stall after a tool_calls delta does not retry.

    Tool-call-only streams must be treated as content (LP-0MS8XAPXT009W3CL);
    a stall after tool-call output terminates immediately instead of
    re-sending the whole request.
    """
    first_resp = _make_mock_response(
        status_code=200,
        aiter_chunks=[_tool_calls_chunk()],
        hang_after=True,
    )
    client = _make_streaming_client([first_resp])

    result, collected = await _run_streaming(client, mock_request)

    assert isinstance(result, StreamingResponse)
    assert client.stream.call_count == 1, (
        f"Expected 1 stream() call (no retry after tool_calls), "
        f"got {client.stream.call_count}"
    )
    assert _last_finish_reason(collected) == "error"


@pytest.mark.asyncio
async def test_readtimeout_after_content_no_retry(mock_request):
    """httpx ReadTimeout after content also terminates without retry.

    AC: Retries (Tier 1) only occur while zero content has been delivered.
    """
    first_resp = _make_error_response([_content_chunk("Hello")], httpx.ReadTimeout)
    client = _make_streaming_client([first_resp])

    result, collected = await _run_streaming(client, mock_request)

    assert isinstance(result, StreamingResponse)
    assert client.stream.call_count == 1, (
        f"Expected 1 stream() call (no retry after content on ReadTimeout), "
        f"got {client.stream.call_count}"
    )
    assert _last_finish_reason(collected) == "error"
    assert b"Hello" in collected[0], "Original content should be preserved"


# ===================================================================
# (c) Retry-before-content still works
# ===================================================================


@pytest.mark.asyncio
async def test_retry_before_content_still_works(mock_request):
    """A stall while zero content has been delivered still retries.

    AC: Retry count/backoff behavior for the zero-content case is unchanged
    (bounded exponential backoff, then finish_reason: error on exhaustion).
    """
    # Attempt 1: hangs with no content (zero-content stall).
    stall_resp = _make_mock_response(
        status_code=200,
        aiter_chunks=[],
        hang_after=True,
    )
    # Attempt 2 (retry): completes normally.
    ok_resp = _make_mock_response(
        status_code=200,
        aiter_chunks=[
            _content_chunk("Hello"),
            _content_chunk(" world"),
            _done_chunk(),
        ],
    )
    client = _make_streaming_client([stall_resp, ok_resp])

    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        with patch("proxy.proxy_remote._srv") as mock_srv:
                            mock_srv.return_value.config = {
                                "server": {
                                    "upstream_retry_base_delay_seconds": 0.01,
                                    "upstream_retry_max_delay_seconds": 0.02,
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
                                remote_timeout=httpx.Timeout(30.0),
                                upstream_idle_timeout_seconds=0.05,
                            )
                            collected = [
                                chunk async for chunk in result.body_iterator
                            ]

    assert isinstance(result, StreamingResponse)
    # Zero-content stall triggered exactly one retry that succeeded.
    assert client.stream.call_count == 2, (
        f"Expected 2 stream() calls (initial + retry), got {client.stream.call_count}"
    )
    joined = b"".join(collected)
    assert b"Hello" in joined and b"world" in joined, (
        "Retry should deliver the content"
    )
    assert b"[DONE]" in joined or _last_finish_reason(collected) == "stop", (
        "Retry should complete the stream normally"
    )


# ===================================================================
# (a) Idle-timeout tuning behavior
# ===================================================================


@pytest.mark.asyncio
async def test_raised_idle_timeout_tolerates_long_chunk_gap(mock_request):
    """A long inter-chunk gap shorter than the configured idle timeout is NOT
    treated as a stall — the stream completes normally on a single connection.

    Simulates the real-world 60-90s reasoning pauses on large prompts
    proportionally: the default upstream idle timeout is raised to 240s (from
    120s), so gaps up to ~180s are tolerated. Here a 0.4s gap with a 1.0s idle
    timeout (same ratio) must not kill or retry the stream.
    """
    chunks = [
        _content_chunk("Hello"),
        _content_chunk(" world"),
        _done_chunk(),
    ]
    mock_resp = _make_mock_response(
        status_code=200,
        aiter_chunks=chunks,
        chunk_delay=0.4,
    )
    client = _make_streaming_client([mock_resp])

    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client):
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
                                # Raised idle timeout — the tuning under test.
                                upstream_idle_timeout_seconds=1.0,
                            )
                            collected = [
                                chunk async for chunk in result.body_iterator
                            ]

    assert isinstance(result, StreamingResponse)
    assert client.stream.call_count == 1, (
        f"Expected 1 stream() call (no stall kill / no retry), "
        f"got {client.stream.call_count}"
    )
    joined = b"".join(collected)
    assert b"Hello" in joined and b"world" in joined, (
        "All chunks should pass through despite the long gap"
    )
    assert b"[DONE]" in joined, "Stream should complete normally"


# ===================================================================
# (d) Config defaults: code-level fallback is 240
# ===================================================================


@pytest.mark.asyncio
async def test_config_default_idle_timeout_is_240(mock_request):
    """When the config key is absent, the code-level fallback default for
    upstream_idle_timeout_seconds is 240 (matching the config value).

    AC: Config defaults — the fallback used by _handle_remote_streaming when
    the key is missing is 240 seconds (LP-0MSF5I7XN009ENWQ raises the default
    from 120 to 240 for LP-0MSF1PUM90099ZSW F4).
    """
    real_wait_for = asyncio.wait_for
    captured_timeouts = []

    async def _capture_wait_for(coro, timeout=None):
        captured_timeouts.append(timeout)
        return await real_wait_for(coro, timeout)

    chunks = [
        _content_chunk("Hello"),
        _done_chunk(),
    ]
    mock_resp = _make_mock_response(status_code=200, aiter_chunks=chunks)
    client = _make_streaming_client([mock_resp])

    with patch("proxy.proxy_remote.asyncio.wait_for", side_effect=_capture_wait_for):
        with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client):
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
                                    # No upstream_idle_timeout_seconds — fallback
                                    # default resolution is under test.
                                )
                                collected = [
                                    chunk async for chunk in result.body_iterator
                                ]

    assert isinstance(result, StreamingResponse)
    assert len(collected) >= 1, "Stream should complete"
    assert captured_timeouts, "wait_for should have been called for chunk reads"
    # The chunk-read wait_for uses the resolved fallback default (240.0).
    # Nothing else in this flow uses 240, so its presence uniquely identifies
    # the per-chunk idle timeout.
    assert 240.0 in captured_timeouts, (
        f"Expected fallback idle timeout 240.0 in {captured_timeouts}"
    )


# ===================================================================
# (f) Tier-3 circuit breaker records after-content stalls
# ===================================================================


@pytest.mark.asyncio
async def test_after_content_stall_records_circuit_breaker(mock_request):
    """An after-content stall termination still records the stall in the
    Tier-3 circuit breaker (LP-0MRFEXXVC001RYKB) so repeated stalls accumulate
    toward provider cooldown.
    """
    first_resp = _make_mock_response(
        status_code=200,
        aiter_chunks=[_content_chunk("Hello")],
        hang_after=True,
    )
    client = _make_streaming_client([first_resp])

    mock_cb = MagicMock(return_value=False)

    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        with patch("proxy.proxy_remote._srv") as mock_srv:
                            mock_srv.return_value.config = {}
                            mock_srv.return_value.logger = MagicMock()
                            with patch(
                                "proxy.proxy_remote._check_stall_circuit_breaker",
                                mock_cb,
                            ):
                                result = await _handle_remote_streaming(
                                    request=mock_request,
                                    target_url="https://api.example.com/v1/chat/completions",
                                    headers={"Authorization": "Bearer test"},
                                    body=b'{"stream": true, "model": "test"}',
                                    body_json={"stream": True, "model": "test"},
                                    model_name="test-model",
                                    remote_timeout=httpx.Timeout(30.0),
                                    upstream_idle_timeout_seconds=0.05,
                                    provider="opencode-deepseek-free",
                                )
                                collected = [
                                    chunk async for chunk in result.body_iterator
                                ]

    assert isinstance(result, StreamingResponse)
    assert _last_finish_reason(collected) == "error"
    assert mock_cb.call_count == 1, (
        f"Expected 1 circuit-breaker call after content stall, "
        f"got {mock_cb.call_count}"
    )
    provider_arg = mock_cb.call_args[0][0]
    assert provider_arg == "opencode-deepseek-free", (
        f"Expected provider 'opencode-deepseek-free', got '{provider_arg}'"
    )
