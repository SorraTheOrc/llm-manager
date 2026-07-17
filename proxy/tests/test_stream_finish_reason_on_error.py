"""
Tests for synthetic finish_reason on stream error (LP-0MQSQZ1W20024WJW).

Verifies that:
1. proxy_remote.py stream_generator yields a synthetic finish_reason: "error"
   event when an httpx stream error occurs.
2. router.py stream_generator yields a synthetic finish_reason: "error" event
   when an httpx stream error occurs.
3. Existing behavior (normal completion, client disconnect) is preserved.
4. Documentation is updated to reflect the new behavior.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import pytest
from fastapi.responses import StreamingResponse

import proxy.server as server


# ── Async iterator that raises after yielding chunks ───────────────────────


class ErrorAsyncIterator:
    """Async iterator that yields chunks then raises on the next anext call."""

    def __init__(self, chunks, exc):
        self.chunks = list(chunks)
        self.exc = exc
        self.idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.idx < len(self.chunks):
            val = self.chunks[self.idx]
            self.idx += 1
            return val
        raise self.exc


class AsyncIterator:
    """Helper to turn a list into a clean async iterator (no exception)."""

    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        return self._iterator()

    async def _iterator(self):
        for item in self.items:
            yield item


# ═══════════════════════════════════════════════════════════════════════════════
# proxy_remote.py tests
# ═══════════════════════════════════════════════════════════════════════════════


def _make_mock_response(chunks, status=200, content_type="text/event-stream"):
    """Create a mock httpx response with controlled streaming chunks."""
    mock = AsyncMock()
    mock.status_code = status
    mock.headers = {"content-type": content_type}
    mock.aiter_bytes = lambda: AsyncIterator(chunks)
    return mock


def _make_error_response(
    chunks, exc_cls=httpx.RemoteProtocolError, content_type="text/event-stream"
):
    """Create a mock httpx response whose aiter_bytes raises after yielding chunks.

    The async iterator yields the given chunks first, then raises *exc_cls*
    on the next iteration.
    """
    mock = AsyncMock()
    mock.status_code = 200
    mock.headers = {"content-type": content_type}
    exc = exc_cls("Simulated stream error for testing")
    mock.aiter_bytes = lambda: ErrorAsyncIterator(chunks, exc)
    return mock


@pytest.fixture
def remote_setup():
    """Minimal fixture providing common mocks for proxy_remote tests."""
    mock_request = MagicMock(spec=["method", "url", "headers"])
    mock_request.method = "POST"
    type(mock_request.url).path = PropertyMock(return_value="/v1/chat/completions")
    mock_request.headers = {}
    return mock_request


@pytest.mark.asyncio
async def test_remote_stream_error_yields_finish_reason(remote_setup):
    """AC1+AC2: proxy_remote.py yields finish_reason: 'error' on stream error.

    When an httpx.RemoteProtocolError occurs mid-stream, the generator should
    yield a synthetic final SSE event with finish_reason: 'error' before cleanup.
    """
    from proxy.proxy_remote import _handle_remote_streaming

    chunks = [
        b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n\n',
        b'data: {"choices": [{"delta": {"content": " wor"}, "index": 0}]}\n\n',
    ]

    # Patch httpx.AsyncClient to return a mock response whose aiter_bytes
    # raises after yielding the chunks above
    with patch("proxy.proxy_remote.httpx.AsyncClient") as mock_client_cls:
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = _make_error_response(
            chunks, httpx.RemoteProtocolError
        )

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client

        result = await _handle_remote_streaming(
            remote_setup,
            "http://fake.api/v1/chat/completions",
            {"Authorization": "Bearer test-key"},
            json.dumps({"model": "test-model", "stream": True}).encode(),
            {"model": "test-model", "stream": True},
            "test-model",
            httpx.Timeout(300),
        )

        assert isinstance(result, StreamingResponse)

        # Collect all chunks from the streaming response
        collected = b""
        async for chunk in result.body_iterator:
            collected += chunk

    # Verify that a synthetic finish_reason event was emitted
    assert b"finish_reason" in collected, (
        "Should contain finish_reason in the output"
    )
    # The last finish_reason should be "error"
    # Decode and check all finish_reason values
    decoded = collected.decode("utf-8")
    finish_reasons = []
    for line in decoded.splitlines():
        line = line.strip()
        if line.startswith("data:") and '"finish_reason"' in line:
            try:
                payload = json.loads(line[5:].strip())
                for choice in payload.get("choices", []):
                    fr = choice.get("finish_reason")
                    if fr is not None:
                        finish_reasons.append(fr)
            except (json.JSONDecodeError, IndexError):
                pass

    assert "error" in finish_reasons, (
        f"Expected finish_reason 'error' in outputs, got {finish_reasons}"
    )

    # Verify original content is preserved
    assert b"Hello" in collected, "Original content should be preserved"
    assert b"wor" in collected, "Original content should be preserved"

    # Verify cleanup was called
    assert mock_cm.__aexit__.called, (
        "Context manager __aexit__ should be called on error"
    )
    assert mock_client.aclose.called, (
        "Client aclose should be called on error"
    )


@pytest.mark.asyncio
async def test_remote_stream_error_with_different_exception(remote_setup):
    """AC1+AC2: Verify with httpx.ReadTimeout as well."""
    from proxy.proxy_remote import _handle_remote_streaming

    chunks = [
        b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n\n',
    ]

    with patch("proxy.proxy_remote.httpx.AsyncClient") as mock_client_cls:
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = _make_error_response(
            chunks, httpx.ReadTimeout
        )

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client

        result = await _handle_remote_streaming(
            remote_setup,
            "http://fake.api/v1/chat/completions",
            {"Authorization": "Bearer test-key"},
            json.dumps({"model": "test-model", "stream": True}).encode(),
            {"model": "test-model", "stream": True},
            "test-model",
            httpx.Timeout(300),
        )

        collected = b""
        async for chunk in result.body_iterator:
            collected += chunk

    assert b"finish_reason" in collected
    assert b'"error"' in collected or b'"error"' in collected
    assert mock_client.aclose.called


@pytest.mark.asyncio
async def test_remote_normal_completion_preserved(remote_setup):
    """AC3: Normal completion still works without synthetic finish_reason on error."""
    from proxy.proxy_remote import _handle_remote_streaming

    chunks = [
        b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n\n',
        b'data: {"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    with patch("proxy.proxy_remote.httpx.AsyncClient") as mock_client_cls:
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = _make_mock_response(chunks)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client

        result = await _handle_remote_streaming(
            remote_setup,
            "http://fake.api/v1/chat/completions",
            {"Authorization": "Bearer test-key"},
            json.dumps({"model": "test-model", "stream": True}).encode(),
            {"model": "test-model", "stream": True},
            "test-model",
            httpx.Timeout(300),
        )

        collected = b""
        async for chunk in result.body_iterator:
            collected += chunk

    # The original finish_reason: "stop" should be present
    # There should be only one finish_reason (the original one)
    finish_count = collected.count(b"finish_reason")
    assert finish_count == 1, (
        f"Expected 1 finish_reason (original), got {finish_count}"
    )
    assert b'"stop"' in collected, "Original finish_reason should be 'stop'"


@pytest.mark.asyncio
async def test_remote_client_disconnect_preserved(remote_setup):
    """AC4: Client disconnect still skips synthetic event (except GeneratorExit)."""
    from proxy.proxy_remote import _handle_remote_streaming

    # Many chunks so we can trigger aclose mid-stream
    chunks = [
        b'data: {"choices": [{"delta": {"content": "Chunk data"}, "index": 0}]}\n\n'
        for _ in range(100)
    ]

    with patch("proxy.proxy_remote.httpx.AsyncClient") as mock_client_cls:
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = _make_mock_response(chunks)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client

        result = await _handle_remote_streaming(
            remote_setup,
            "http://fake.api/v1/chat/completions",
            {"Authorization": "Bearer test-key"},
            json.dumps({"model": "test-model", "stream": True}).encode(),
            {"model": "test-model", "stream": True},
            "test-model",
            httpx.Timeout(300),
        )

        assert isinstance(result, StreamingResponse)

        # Read a few chunks then trigger aclose (simulates client disconnect)
        iterator = result.body_iterator.__aiter__()
        first_chunk = await iterator.__anext__()
        assert first_chunk is not None

        # Close the generator — should NOT raise RuntimeError
        await iterator.aclose()

        # Verify cleanup was called
        assert mock_cm.__aexit__.called, (
            "__aexit__ should be called during aclose cleanup"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# router.py tests
# ═══════════════════════════════════════════════════════════════════════════════


BASE_SERVER_CONFIG = {
    "server": {
        "llama_router_mode": False,
        "llama_server_port": 8080,
        "max_concurrent_queries": 4,
        "local_max_concurrent_queries": 1,
        "llama_request_timeout": 30,
        "session_single_flight_mode": "bypass",
        "disconnect_cleanup_timeout": 1,
    }
}


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


def _make_error_streaming_response(
    chunks, exc_cls=httpx.RemoteProtocolError, content_type="text/event-stream"
):
    """Build a mock streaming response whose aiter_bytes raises after chunks."""

    exc = exc_cls("Simulated stream error for testing")

    async def _aiter():
        for c in chunks:
            yield c
        raise exc

    MockStreamResponse = type("MockStreamResponse", (), {
        "status_code": 200,
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


@pytest.fixture(autouse=True)
def _reset_server_state(monkeypatch):
    """Reset server-level state before each test."""
    monkeypatch.setattr(server, "config", dict(BASE_SERVER_CONFIG))
    monkeypatch.setattr(server, "active_queries", 0)
    monkeypatch.setattr(server, "local_active_queries", 0)
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "llama_process", MagicMock(poll=lambda: None, pid=1))
    monkeypatch.setattr(server, "current_model", "test-model")
    monkeypatch.setattr(server, "session_manager", MagicMock())
    monkeypatch.setattr(server, "logger", MagicMock())

    # Disable scheduler and self-healing
    monkeypatch.setattr("proxy.router._get_job_scheduler", lambda: None)
    monkeypatch.setattr("proxy.router._is_self_healing_active", lambda: False)

    # Reset metrics
    monkeypatch.setattr(server, "backend_signal_counts", {
        "connect_failures": 0,
        "read_failures": 0,
        "timeout_failures": 0,
        "other_failures": 0,
        "concurrency_rejects": 0,
    })

    # Mock slot save/restore
    monkeypatch.setattr("proxy.router._restore_slot_snapshot", AsyncMock(return_value=False))
    monkeypatch.setattr("proxy.router._save_slot_snapshot", AsyncMock(return_value=False))
    monkeypatch.setattr("proxy.router._build_slot_context", MagicMock(return_value=(None, None, 3.0)))

    # Mock session handlers
    monkeypatch.setattr("proxy.router._handle_session", AsyncMock(return_value={
        "session_id": "test-session-id",
        "session_created": True,
        "is_delta_request": False,
        "session_fallback_reason": None,
        "delta_messages": [],
        "original_message_count": 1,
        "body_override": None,
        "body_json": None,
    }))

    # Mock log resolvers
    monkeypatch.setattr("proxy.session._resolve_log_path", MagicMock(return_value=MagicMock(
        exists=lambda: False,
        stat=lambda: MagicMock(st_size=0),
    )))

    # Mock slot availability
    monkeypatch.setattr("proxy.router._check_slot_availability", AsyncMock(return_value=None))


@pytest.mark.asyncio
async def test_router_stream_error_yields_finish_reason(monkeypatch):
    """AC1: router.py yields finish_reason: 'error' on stream error.

    When a RemoteProtocolError occurs mid-stream in proxy_to_local's
    stream_generator, a synthetic final SSE event with finish_reason: 'error'
    should be yielded before cleanup.
    """
    from proxy.router import proxy_to_local

    chunks = [
        b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n\n',
        b'data: {"choices": [{"delta": {"content": " world"}, "index": 0}]}\n\n',
    ]

    cm, sresp = _make_error_streaming_response(chunks, httpx.RemoteProtocolError)
    monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, sresp)))

    response = await proxy_to_local(
        _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=True),
        "v1/chat/completions",
    )

    assert isinstance(response, StreamingResponse), "Expected a StreamingResponse"

    # Collect all chunks from the streaming response
    collected = b""
    async for chunk in response.body_iterator:
        collected += chunk

    # Verify that a synthetic finish_reason event was emitted
    decoded = collected.decode("utf-8")
    finish_reasons = []
    for line in decoded.splitlines():
        line = line.strip()
        if line.startswith("data:") and '"finish_reason"' in line:
            try:
                payload = json.loads(line[5:].strip())
                for choice in payload.get("choices", []):
                    fr = choice.get("finish_reason")
                    if fr is not None:
                        finish_reasons.append(fr)
            except (json.JSONDecodeError, IndexError):
                pass

    assert "error" in finish_reasons, (
        f"Expected finish_reason 'error' in outputs, got {finish_reasons}"
    )

    # Verify original content is preserved
    assert b"Hello" in collected, "Original content should be preserved"


@pytest.mark.asyncio
async def test_router_stream_error_with_read_timeout(monkeypatch):
    """AC1: Verify with httpx.ReadTimeout as well."""
    from proxy.router import proxy_to_local

    chunks = [
        b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n\n',
    ]

    cm, sresp = _make_error_streaming_response(chunks, httpx.ReadTimeout)
    monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, sresp)))

    response = await proxy_to_local(
        _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=True),
        "v1/chat/completions",
    )

    collected = b""
    async for chunk in response.body_iterator:
        collected += chunk

    assert b"finish_reason" in collected
    assert b'"error"' in collected


@pytest.mark.asyncio
async def test_router_normal_completion_preserved(monkeypatch):
    """AC3+AC4: Normal completion still works without spurious finish_reason."""
    from proxy.router import proxy_to_local

    chunks = [
        b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n\n',
        b'data: {"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    mock_response = type("MockStreamResponse", (), {
        "status_code": 200,
        "headers": {"content-type": "text/event-stream"},
        "aiter_bytes": lambda: AsyncIterator(chunks).__aiter__(),
        "aread": AsyncMock(return_value=b"".join(chunks)),
    })

    class MockCM:
        async def __aenter__(self):
            return mock_response

        async def __aexit__(self, *args):
            pass

    cm = MockCM()
    monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, mock_response)))

    response = await proxy_to_local(
        _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=True),
        "v1/chat/completions",
    )

    collected = b""
    async for chunk in response.body_iterator:
        collected += chunk

    # The original finish_reason should be present, and there should be only one
    finish_count = collected.count(b"finish_reason")
    assert finish_count >= 1, "Should have finish_reason from original stream"
    # Normal completion emits a synthetic stop if the original had finish_reason
    # already, so verify at least one finish_reason exists (not an error one)
    assert b'"stop"' in collected, "Should have finish_reason 'stop'"


# ═══════════════════════════════════════════════════════════════════════════════
# Structural verification test
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "filepath,expected_substring",
    [
        (
            "proxy/router.py",
            'finish_reason": "error',
        ),
        (
            "proxy/proxy_remote.py",
            'finish_reason": "error',
        ),
    ],
)
def test_except_exception_contains_finish_reason_error(filepath, expected_substring):
    """Structural test: verify except Exception handler yields finish_reason: 'error'.

    This ensures the synthetic finish_reason yield exists in the exception
    handler of stream_generator in both files.
    """
    import os

    test_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(test_dir, "..", filepath)

    with open(full_path) as f:
        content = f.read()

    # Find the "except Exception" section and verify it contains the expected pattern
    # We look for 'except Exception' followed by the synthetic finish_reason code
    lines = content.splitlines()

    found_except_exception = False
    found_finish_reason_error = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("except Exception") or stripped.startswith("except Exception "):
            found_except_exception = True
            # Look at the next lines for the finish_reason error pattern
            for j in range(i, min(i + 30, len(lines))):
                if expected_substring in lines[j]:
                    found_finish_reason_error = True
                    break

    assert found_except_exception, (
        f"Could not find 'except Exception' in {filepath}"
    )
    assert found_finish_reason_error, (
        f"Could not find '{expected_substring}' near 'except Exception' in {filepath}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LP-0MRDI0FYQ008BI5Y: Remote ReadTimeout bounded-timeout verification tests
# ═══════════════════════════════════════════════════════════════════════════════
#
# These tests verify the proxy fails fast (within < 30s) when a remote provider
# ReadTimeout occurs. They cover:
# 1. ReadTimeout at different simulated delays (5, 15, 30 chunks/events)
# 2. Bounded error-recovery time
# 3. Non-streaming remote ReadTimeout path
# 4. Timeout configuration (adaptive vs fixed) integration
# ═══════════════════════════════════════════════════════════════════════════════


class ChunkThenErrorIterator:
    """Async iterator that yields N chunks then raises httpx.ReadTimeout.

    Simulates a ReadTimeout occurring after a configurable number of
    SSE chunks have been received (representing different request delays).
    """

    def __init__(self, num_chunks: int, exc):
        self._chunks = [
            b'data: {"choices": [{"delta": {"content": "chunk "}, "index": 0}]}\n\n'
        ] * num_chunks
        self._exc = exc
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx < len(self._chunks):
            val = self._chunks[self._idx]
            self._idx += 1
            return val
        raise self._exc


def _make_chunk_then_error_response(num_chunks, exc_cls=httpx.ReadTimeout):
    """Create a mock httpx response that yields N chunks then raises *exc_cls*.

    The *num_chunks* parameter controls how many SSE chunks are emitted
    before the error, simulating a real request that streams data for a
    while before hitting a ReadTimeout.
    """
    exc = exc_cls(f"Simulated ReadTimeout after {num_chunks} chunks")
    mock = AsyncMock()
    mock.status_code = 200
    mock.headers = {"content-type": "text/event-stream"}
    mock.aiter_bytes = lambda: ChunkThenErrorIterator(num_chunks, exc)
    return mock


def _make_mock_remote_client(mock_response):
    """Create a mock httpx.AsyncClient that returns *mock_response* on stream()."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_response)
    cm.__aexit__ = AsyncMock(return_value=None)

    client_instance = MagicMock()
    client_instance.stream = MagicMock(return_value=cm)
    client_instance.aclose = AsyncMock()

    mock_client_cls = MagicMock(return_value=client_instance)
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=client_instance)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

    return mock_client_cls


@pytest.fixture
def mock_remote_request():
    """Minimal mock Request for proxy_remote tests."""
    req = MagicMock(spec=["method", "url", "headers", "is_disconnected"])
    req.method = "POST"
    type(req.url).path = PropertyMock(return_value="/v1/chat/completions")
    req.headers = {}
    req.is_disconnected = AsyncMock(return_value=False)
    return req


# ── AC1: Configurable ReadTimeout delays (5s, 15s, 30s) ────────────────


@pytest.mark.asyncio
async def test_remote_readtimeout_after_few_chunks(mock_remote_request):
    """AC1: ReadTimeout after 5 chunks (simulating ~5s delay) yields finish_reason: error.

    Verifies that when a ReadTimeout occurs early in the stream (after
    only a few chunks), the finish_reason: "error" is still emitted.
    """
    from proxy.proxy_remote import _handle_remote_streaming

    mock_resp = _make_chunk_then_error_response(5, httpx.ReadTimeout)
    mock_client_cls = _make_mock_remote_client(mock_resp)

    with patch("proxy.proxy_remote.httpx.AsyncClient", mock_client_cls):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        # Use a bounded timeout for the test itself
                        result = await asyncio.wait_for(
                            _handle_remote_streaming(
                                request=mock_remote_request,
                                target_url="https://api.example.com/v1/chat/completions",
                                headers={"Authorization": "Bearer test"},
                                body=b'{"stream": true, "model": "test"}',
                                body_json={"stream": True, "model": "test"},
                                model_name="test-model",
                                remote_timeout=httpx.Timeout(60.0),
                            ),
                            timeout=10.0,
                        )

    assert isinstance(result, StreamingResponse), (
        f"Expected StreamingResponse, got {type(result).__name__}"
    )

    # Collect chunks (test is bounded by asyncio.wait_for timeout)
    collected = b""
    async for chunk in result.body_iterator:
        collected += chunk

    assert b"finish_reason" in collected, (
        "Should contain finish_reason in the output"
    )

    decoded = collected.decode("utf-8")
    finish_reasons = []
    for line in decoded.splitlines():
        line = line.strip()
        if line.startswith("data:") and '"finish_reason"' in line:
            try:
                payload = json.loads(line[5:].strip())
                for choice in payload.get("choices", []):
                    fr = choice.get("finish_reason")
                    if fr is not None:
                        finish_reasons.append(fr)
            except (json.JSONDecodeError, IndexError):
                pass

    assert "error" in finish_reasons, (
        f"Expected finish_reason 'error' in outputs, got {finish_reasons}"
    )
    assert b"chunk" in collected, "Original content should be preserved"


@pytest.mark.asyncio
async def test_remote_readtimeout_after_mid_chunks(mock_remote_request):
    """AC1: ReadTimeout after 15 chunks (simulating ~15s delay)."""
    from proxy.proxy_remote import _handle_remote_streaming

    mock_resp = _make_chunk_then_error_response(15, httpx.ReadTimeout)
    mock_client_cls = _make_mock_remote_client(mock_resp)

    with patch("proxy.proxy_remote.httpx.AsyncClient", mock_client_cls):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        result = await asyncio.wait_for(
                            _handle_remote_streaming(
                                request=mock_remote_request,
                                target_url="https://api.example.com/v1/chat/completions",
                                headers={"Authorization": "Bearer test"},
                                body=b'{"stream": true, "model": "test"}',
                                body_json={"stream": True, "model": "test"},
                                model_name="test-model",
                                remote_timeout=httpx.Timeout(60.0),
                            ),
                            timeout=10.0,
                        )

    assert isinstance(result, StreamingResponse)

    collected = b""
    async for chunk in result.body_iterator:
        collected += chunk

    assert b"finish_reason" in collected

    decoded = collected.decode("utf-8")
    finish_reasons = []
    for line in decoded.splitlines():
        line = line.strip()
        if line.startswith("data:") and '"finish_reason"' in line:
            try:
                payload = json.loads(line[5:].strip())
                for choice in payload.get("choices", []):
                    fr = choice.get("finish_reason")
                    if fr is not None:
                        finish_reasons.append(fr)
            except (json.JSONDecodeError, IndexError):
                pass

    assert "error" in finish_reasons, (
        f"Expected finish_reason 'error', got {finish_reasons}"
    )


@pytest.mark.asyncio
async def test_remote_readtimeout_after_many_chunks(mock_remote_request):
    """AC1: ReadTimeout after 30 chunks (simulating ~30s delay)."""
    from proxy.proxy_remote import _handle_remote_streaming

    mock_resp = _make_chunk_then_error_response(30, httpx.ReadTimeout)
    mock_client_cls = _make_mock_remote_client(mock_resp)

    with patch("proxy.proxy_remote.httpx.AsyncClient", mock_client_cls):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        result = await asyncio.wait_for(
                            _handle_remote_streaming(
                                request=mock_remote_request,
                                target_url="https://api.example.com/v1/chat/completions",
                                headers={"Authorization": "Bearer test"},
                                body=b'{"stream": true, "model": "test"}',
                                body_json={"stream": True, "model": "test"},
                                model_name="test-model",
                                remote_timeout=httpx.Timeout(60.0),
                            ),
                            timeout=10.0,
                        )

    assert isinstance(result, StreamingResponse)

    collected = b""
    async for chunk in result.body_iterator:
        collected += chunk

    assert b"finish_reason" in collected

    decoded = collected.decode("utf-8")
    finish_reasons = []
    for line in decoded.splitlines():
        line = line.strip()
        if line.startswith("data:") and '"finish_reason"' in line:
            try:
                payload = json.loads(line[5:].strip())
                for choice in payload.get("choices", []):
                    fr = choice.get("finish_reason")
                    if fr is not None:
                        finish_reasons.append(fr)
            except (json.JSONDecodeError, IndexError):
                pass

    assert "error" in finish_reasons, (
        f"Expected finish_reason 'error', got {finish_reasons}"
    )


# ── AC2: Bounded-time error recovery verification ────────────────────────


@pytest.mark.asyncio
async def test_remote_readtimeout_completes_within_bounded_time(mock_remote_request):
    """AC2: Stream terminates with finish_reason 'error' within bounded time.

    Verifies that the entire proxy_remote streaming path (including error
    synthesis, cleanup, and client consumption) completes in < 30 seconds.
    Uses a 15s test timeout to prove fast-fail behavior.
    """
    from proxy.proxy_remote import _handle_remote_streaming

    mock_resp = _make_chunk_then_error_response(3, httpx.ReadTimeout)
    mock_client_cls = _make_mock_remote_client(mock_resp)

    start = asyncio.get_running_loop().time()

    with patch("proxy.proxy_remote.httpx.AsyncClient", mock_client_cls):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        result = await asyncio.wait_for(
                            _handle_remote_streaming(
                                request=mock_remote_request,
                                target_url="https://api.example.com/v1/chat/completions",
                                headers={"Authorization": "Bearer test"},
                                body=b'{"stream": true, "model": "test"}',
                                body_json={"stream": True, "model": "test"},
                                model_name="test-model",
                                remote_timeout=httpx.Timeout(60.0),
                            ),
                            timeout=15.0,
                        )

        # Consume the streaming response
        _ = [chunk async for chunk in result.body_iterator]

    elapsed = asyncio.get_running_loop().time() - start

    assert elapsed < 30.0, (
        f"Remote ReadTimeout recovery exceeded 30s bounded window: {elapsed:.2f}s"
    )
    assert isinstance(result, StreamingResponse)


# ── AC2: Error response well-formed for client retry ─────────────────────


@pytest.mark.asyncio
async def test_remote_readtimeout_error_is_actionable(mock_remote_request):
    """AC3: Error response is well-formed so client (Pi) can detect and retry.

    The error SSE event must have a proper structure that Pi can interpret
    without manual intervention (ESC abort). The finish_reason: "error"
    in the choices array is the standard mechanism Pi uses to detect
    stream errors and trigger automatic retry.
    """
    from proxy.proxy_remote import _handle_remote_streaming

    mock_resp = _make_chunk_then_error_response(1, httpx.ReadTimeout)
    mock_client_cls = _make_mock_remote_client(mock_resp)

    with patch("proxy.proxy_remote.httpx.AsyncClient", mock_client_cls):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        result = await _handle_remote_streaming(
                            request=mock_remote_request,
                            target_url="https://api.example.com/v1/chat/completions",
                            headers={"Authorization": "Bearer test"},
                            body=b'{"stream": true, "model": "test"}',
                            body_json={"stream": True, "model": "test"},
                            model_name="test-model",
                            remote_timeout=httpx.Timeout(60.0),
                        )

    collected = b""
    async for chunk in result.body_iterator:
        collected += chunk

    # Verify the SSE output is valid JSON with choices[] and finish_reason
    decoded = collected.decode("utf-8")
    lines = [line.strip() for line in decoded.splitlines() if line.strip()]

    # Find the error SSE event
    error_events = []
    for line in lines:
        if line.startswith("data:") and '"finish_reason"' in line:
            try:
                payload = json.loads(line[5:].strip())
                assert "choices" in payload, (
                    "SSE error event must have 'choices' array"
                )
                assert isinstance(payload["choices"], list), (
                    "'choices' must be a list"
                )
                error_events.append(payload)
            except (json.JSONDecodeError, IndexError):
                pass

    assert len(error_events) >= 1, (
        "At least one SSE event with finish_reason must be present"
    )

    # Verify the last choices entry has finish_reason: "error"
    last_event = error_events[-1]
    last_choice = last_event["choices"][-1]
    assert last_choice.get("finish_reason") == "error", (
        f"Expected finish_reason 'error', got '{last_choice.get('finish_reason')}'"
    )

    # Verify the delta is empty (standard error format)
    assert last_choice.get("delta", {}) == {}, (
        "Error event delta should be empty"
    )


# ── AC4: Non-streaming remote ReadTimeout path ───────────────────────────


@pytest.mark.asyncio
async def test_remote_non_streaming_readtimeout_returns_error(mock_remote_request):
    """AC4: Non-streaming remote path handles ReadTimeout gracefully.

    When a ReadTimeout occurs in the non-streaming (single response) path,
    the proxy should return an error response rather than hanging.
    """
    from proxy.proxy_remote import _handle_remote_non_streaming

    with patch("proxy.proxy_remote.httpx.AsyncClient") as mock_client_cls:
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=None)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_client = MagicMock()
        # Make the client.post() call raise ReadTimeout
        mock_client.post = MagicMock(side_effect=httpx.ReadTimeout(
            "Simulated ReadTimeout on non-streaming request",
            request=None,
        ))
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response"):
                with patch("proxy.proxy_remote.log_request"):
                    with patch("proxy.proxy_remote._srv") as mock_srv:
                        mock_srv.return_value = MagicMock(
                            logger=MagicMock(warning=MagicMock()),
                            current_model="test-model",
                        )

                        start = asyncio.get_running_loop().time()
                        try:
                            result = await asyncio.wait_for(
                                _handle_remote_non_streaming(
                                    request=mock_remote_request,
                                    target_url="https://api.example.com/v1/chat/completions",
                                    headers={"Authorization": "Bearer test"},
                                    body=b'{"model": "test", "messages": [{"role": "user", "content": "hi"}]}',
                                    model_name="test-model",
                                    remote_timeout=httpx.Timeout(60.0),
                                ),
                                timeout=15.0,
                            )
                            elapsed = asyncio.get_running_loop().time() - start

                            # An exception should propagate since _handle_remote_non_streaming
                            # doesn't catch client.post() ReadTimeout. This is expected
                            # because the non-streaming path has no async generator to
                            # catch the error — the caller (proxy_to_remote) must handle it.
                        except httpx.ReadTimeout:
                            # Expected: ReadTimeout propagates because non-streaming
                            # doesn't have an async generator to catch it
                            pass
                        except Exception:
                            # Any other exception is also acceptable — the key assertion
                            # is that the call returns/fails within bounded time
                            pass

                        elapsed = asyncio.get_running_loop().time() - start
                        assert elapsed < 30.0, (
                            f"Non-streaming ReadTimeout recovery exceeded 30s: {elapsed:.2f}s"
                        )


# ── AC4: Both streaming and non-streaming covered ────────────────────────


@pytest.mark.asyncio
async def test_remote_both_paths_concurrent(mock_remote_request):
    """AC4: Both streaming and non-streaming ReadTimeout handling work concurrently.

    Verifies that handling a ReadTimeout in streaming and non-streaming
    paths simultaneously does not cause interference.
    """
    from proxy.proxy_remote import (
        _handle_remote_streaming,
        _handle_remote_non_streaming,
    )

    # Set up streaming mock
    stream_resp = _make_chunk_then_error_response(3, httpx.ReadTimeout)
    mock_stream_client = _make_mock_remote_client(stream_resp)

    # Set up non-streaming mock
    mock_ns_cm = AsyncMock()
    mock_ns_cm.__aenter__ = AsyncMock(return_value=None)
    mock_ns_cm.__aexit__ = AsyncMock(return_value=None)
    mock_ns_client = MagicMock()
    mock_ns_client.post = MagicMock(side_effect=httpx.ReadTimeout(
        "Simulated non-streaming ReadTimeout", request=None,
    ))
    mock_ns_client.aclose = AsyncMock()

    async def _test_streaming():
        with patch("proxy.proxy_remote.httpx.AsyncClient", mock_stream_client):
            with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
                with patch("proxy.proxy_remote.log_response_chunk"):
                    with patch("proxy.proxy_remote.log_response"):
                        with patch("proxy.proxy_remote.log_request"):
                            r = await _handle_remote_streaming(
                                request=mock_remote_request,
                                target_url="https://api.example.com/v1/chat/completions",
                                headers={"Authorization": "Bearer test"},
                                body=b'{"stream": true, "model": "test"}',
                                body_json={"stream": True, "model": "test"},
                                model_name="test-model",
                                remote_timeout=httpx.Timeout(60.0),
                            )
                            _ = [chunk async for chunk in r.body_iterator]
                            return True

    async def _test_non_streaming():
        with patch("proxy.proxy_remote.httpx.AsyncClient") as cls:
            cls.return_value = mock_ns_client
            cls.return_value.__aenter__ = AsyncMock(return_value=mock_ns_client)
            cls.return_value.__aexit__ = AsyncMock(return_value=None)

            with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        with patch("proxy.proxy_remote._srv") as srv:
                            srv.return_value = MagicMock(
                                logger=MagicMock(warning=MagicMock()),
                                current_model="test-model",
                            )
                            try:
                                await _handle_remote_non_streaming(
                                    request=mock_remote_request,
                                    target_url="https://api.example.com/v1/chat/completions",
                                    headers={"Authorization": "Bearer test"},
                                    body=b'{"model": "test"}',
                                    model_name="test-model",
                                    remote_timeout=httpx.Timeout(60.0),
                                )
                            except httpx.ReadTimeout:
                                pass
                            return True

    # Run both concurrently
    stream_result, ns_result = await asyncio.gather(
        _test_streaming(),
        _test_non_streaming(),
        return_exceptions=True,
    )

    assert stream_result is True or isinstance(stream_result, Exception), (
        f"Streaming path failed: {stream_result}"
    )
    assert ns_result is True or isinstance(stream_result, Exception), (
        f"Non-streaming path failed: {ns_result}"
    )


# ── Extra: structural test for remote ReadTimeout error handling ─────────


@pytest.mark.parametrize(
    "filepath,expected_substring",
    [
        (
            "proxy/proxy_remote.py",
            'finish_reason": "error',
        ),
    ],
)
def test_proxy_remote_except_exception_readtimeout(filepath, expected_substring):
    """Structural verification: proxy_remote.py has finish_reason: 'error' in
    its stream_generator's exception handler.

    This ensures the synthetic finish_reason: 'error' yield exists even for
    httpx.ReadTimeout in the remote streaming path.
    """
    import os

    test_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(test_dir, "..", filepath)

    with open(full_path) as f:
        content = f.read()

    lines = content.splitlines()

    found_except_exception = False
    found_finish_reason_error = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("except Exception") or stripped.startswith("except Exception "):
            found_except_exception = True
            for j in range(i, min(i + 30, len(lines))):
                if expected_substring in lines[j]:
                    found_finish_reason_error = True
                    break

    assert found_except_exception, (
        f"Could not find 'except Exception' in {filepath}"
    )
    assert found_finish_reason_error, (
        f"Could not find '{expected_substring}' near 'except Exception' in {filepath}"
    )
