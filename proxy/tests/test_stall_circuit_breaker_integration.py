"""
Integration tests for circuit breaker wiring in the retry-exhausted path.

Tests verify the call chain from ``_handle_remote_streaming()`` in
``proxy_remote.py`` through to ``_check_stall_circuit_breaker()`` when
per-stream Tier 1 retries exhaust.

Uses mocked ``_check_stall_circuit_breaker`` to verify it is called with
the correct provider name, and verifies that the circuit breaker does NOT
fire in scenarios where stalls should not be recorded.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse

from proxy.proxy_remote import _handle_remote_streaming


# ===================================================================
# Mock helpers (reused from test_upstream_stall_detection.py)
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


def _make_mock_response(
    status_code=200,
    headers=None,
    aiter_chunks=None,
    hang_after=False,
    error_body=None,
):
    """Create a mock HTTP response for testing."""
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
# AC1: Retry exhaustion calls circuit breaker
# ===================================================================


@pytest.mark.asyncio
async def test_retry_exhaustion_calls_circuit_breaker(mock_request):
    """Retry exhaustion in _handle_remote_streaming calls _check_stall_circuit_breaker.

    AC1: When max retries are exhausted, _check_stall_circuit_breaker()
    is called with the correct provider name.
    """
    # All 4 attempts stall after first chunk (initial + 3 retries)
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

    cms = []
    for resp in responses:
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=None)
        cms.append(cm)

    client_instance = MagicMock(spec=httpx.AsyncClient)
    client_instance.stream = MagicMock(side_effect=cms)
    client_instance.aclose = AsyncMock(return_value=None)

    # Mock the circuit breaker to track calls
    mock_cb = AsyncMock(return_value=False)

    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client_instance):
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
                                )

                                collected = [
                                    chunk async for chunk in result.body_iterator
                                ]

    assert isinstance(result, StreamingResponse)
    assert len(collected) >= 1, "Should have yielded at least one chunk"

    # _check_stall_circuit_breaker should have been called once (at max retries)
    assert mock_cb.call_count == 1, (
        f"Expected 1 call to _check_stall_circuit_breaker, got {mock_cb.call_count}"
    )

    # Verify the provider name was passed correctly
    call_args = mock_cb.call_args
    assert call_args is not None
    provider_arg = call_args[0][0]
    assert provider_arg == "remote", (
        f"Expected provider='remote', got '{provider_arg}'"
    )

    # The last chunk should be finish_reason: error
    last_chunk = collected[-1].decode("utf-8", errors="replace")
    assert '"finish_reason":"error"' in last_chunk.replace(" ", ""), (
        f"Last chunk should have finish_reason: error, got: {last_chunk[:200]}"
    )


# ===================================================================
# AC2: Circuit breaker triggered with provider name
# ===================================================================


@pytest.mark.asyncio
async def test_circuit_breaker_called_with_provider_name(mock_request):
    """_check_stall_circuit_breaker is called with the configured provider name.

    AC2: When _handle_remote_streaming is invoked with a named provider,
    that name is passed to the circuit breaker.
    """
    # All attempts stall (to trigger retry exhaustion)
    chunk = [b'data: {"choices":[{"delta":{"content":"A"},"index":0}]}\n\n']
    responses = []
    for i in range(4):
        resp = _make_mock_response(
            status_code=200, aiter_chunks=chunk, hang_after=True
        )
        responses.append(resp)

    cms = []
    for resp in responses:
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=None)
        cms.append(cm)

    client_instance = MagicMock(spec=httpx.AsyncClient)
    client_instance.stream = MagicMock(side_effect=cms)
    client_instance.aclose = AsyncMock(return_value=None)

    mock_cb = AsyncMock(return_value=False)

    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client_instance):
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

                                _collected = [
                                    chunk async for chunk in result.body_iterator
                                ]

    assert isinstance(result, StreamingResponse)

    # Verify the provider name was passed
    call_args = mock_cb.call_args
    assert call_args is not None, "check_stall_circuit_breaker was not called"
    provider_arg = call_args[0][0]
    assert provider_arg == "opencode-deepseek-free", (
        f"Expected provider='opencode-deepseek-free', got '{provider_arg}'"
    )

    # Verify config dict was passed as second arg
    config_arg = call_args[0][1]
    assert isinstance(config_arg, dict), (
        "Second argument should be a config dict"
    )


# ===================================================================
# AC3: Normal streaming does NOT trigger circuit breaker
# ===================================================================


@pytest.mark.asyncio
async def test_normal_streaming_does_not_trigger_cb(mock_request):
    """Normal streaming (no stall) does not call the circuit breaker.

    AC3: When streaming completes normally with [DONE], the circuit
    breaker should NOT be called.
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
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_resp)
    cm.__aexit__ = AsyncMock(return_value=None)

    client_instance = MagicMock(spec=httpx.AsyncClient)
    client_instance.stream = MagicMock(return_value=cm)
    client_instance.aclose = AsyncMock(return_value=None)

    mock_cb = AsyncMock(return_value=False)

    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client_instance):
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
                                    upstream_idle_timeout_seconds=0.5,
                                    provider="opencode-deepseek-free",
                                )

                                collected = [
                                    chunk async for chunk in result.body_iterator
                                ]

    assert isinstance(result, StreamingResponse)
    assert len(collected) == 3, (
        f"Expected 3 chunks, got {len(collected)}"
    )

    # Circuit breaker should NOT have been called
    assert mock_cb.call_count == 0, (
        f"Expected 0 calls to _check_stall_circuit_breaker, "
        f"got {mock_cb.call_count}"
    )


# ===================================================================
# AC4: Client disconnect (GeneratorExit) does NOT trigger circuit breaker
# ===================================================================


@pytest.mark.asyncio
async def test_client_disconnect_does_not_trigger_cb(mock_request):
    """Client disconnect should not trigger the circuit breaker.

    AC4: When the client disconnects (GeneratorExit), the circuit breaker
    should NOT be called since the stall was not caused by the upstream.
    """
    chunks = [
        b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
    ]

    mock_resp = _make_mock_response(
        status_code=200,
        aiter_chunks=chunks,
        hang_after=True,
    )
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_resp)
    cm.__aexit__ = AsyncMock(return_value=None)

    client_instance = MagicMock(spec=httpx.AsyncClient)
    client_instance.stream = MagicMock(return_value=cm)
    client_instance.aclose = AsyncMock(return_value=None)

    mock_cb = AsyncMock(return_value=False)

    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client_instance):
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
                                    upstream_idle_timeout_seconds=0.5,
                                    provider="opencode-deepseek-free",
                                )

                                # Collect only the first chunk to trigger generator
                                # closure (simulating client disconnect)
                                iterator = result.body_iterator.__aiter__()
                                _first_chunk = await iterator.__anext__()

    # Circuit breaker should NOT have been called (GeneratorExit, not stall)
    assert mock_cb.call_count == 0, (
        f"Expected 0 calls to _check_stall_circuit_breaker on client disconnect, "
        f"got {mock_cb.call_count}"
    )


# ===================================================================
# AC5: Non-timeout errors do NOT trigger circuit breaker
# ===================================================================


@pytest.mark.asyncio
async def test_non_timeout_error_does_not_trigger_cb(mock_request):
    """Non-timeout stream errors should not trigger the circuit breaker.

    AC5: httpx.RemoteProtocolError is not a stall and should not
    trigger the circuit breaker.
    """
    # First response raises a RemoteProtocolError
    first_resp = MagicMock(spec=httpx.Response)
    type(first_resp).status_code = PropertyMock(return_value=200)
    first_resp.headers = {"content-type": "text/event-stream"}
    first_resp.aiter_bytes = MagicMock(
        side_effect=httpx.RemoteProtocolError("Connection reset")
    )

    cm1 = MagicMock()
    cm1.__aenter__ = AsyncMock(return_value=first_resp)
    cm1.__aexit__ = AsyncMock(return_value=None)

    client_instance = MagicMock(spec=httpx.AsyncClient)
    client_instance.stream = MagicMock(return_value=cm1)
    client_instance.aclose = AsyncMock(return_value=None)

    mock_cb = AsyncMock(return_value=False)

    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client_instance):
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
                                    upstream_idle_timeout_seconds=0.5,
                                    provider="opencode-deepseek-free",
                                )

                                _collected = [
                                    chunk async for chunk in result.body_iterator
                                ]

    assert isinstance(result, StreamingResponse)

    # Circuit breaker should NOT have been called (RemoteProtocolError
    # is handled differently — it yields finish_reason:error but does
    # NOT retry, so retry_count never reaches max_retries in the outer loop)
    assert mock_cb.call_count == 0, (
        f"Expected 0 calls to _check_stall_circuit_breaker for non-timeout error, "
        f"got {mock_cb.call_count}"
    )
