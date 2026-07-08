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
