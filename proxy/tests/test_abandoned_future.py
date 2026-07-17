"""
Tests for abandoned asyncio future cleanup in streaming path (LP-0MRCMKG9O004XE0Q).

Verifies that:
1. stream_generator() in router.py retrieves exceptions from the pending
   _stream_iter future when the loop exits, preventing "Task exception was
   never retrieved" warnings.
2. The fix does NOT cancel the in-flight _stream_iter future (CRITICAL
   constraint from LP-0MQTHP828000JYM6).
3. Normal completion and error handling continue to work correctly.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import proxy.server as server


BASE_SERVER_CONFIG = {
    "server": {
        "llama_router_mode": False,
        "llama_server_port": 8080,
        "max_concurrent_queries": 4,
        "local_max_concurrent_queries": 1,
        "llama_request_timeout": 30,
        "session_single_flight_mode": "bypass",
        "disconnect_cleanup_timeout": 1,
        "stream_heartbeat_interval_seconds": 0.05,
        "stream_idle_timeout_seconds": 0.3,
        "session_guardrail_max_runtime_seconds": 3600,
        "session_guardrail_max_completion_tokens": 4096,
        "session_guardrail_repetition_min_pattern_chars": 100,
        "session_guardrail_repetition_min_repeats": 3,
        "session_guardrail_invalidate_on_cutoff": False,
        "session_guardrail_invalidate_on_repetition": False,
        "session_guardrail_max_token_rate": 0,
        "session_guardrail_token_rate_window_seconds": 60,
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


def _make_mock_cm(aiter_func):
    """Create (cm, response) for router tests.

    Returns (context_manager, response_object) matching the contract
    of _call_with_backend_retries in the streaming path.
    """
    exc = None

    async def _aiter():
        async for chunk in aiter_func():
            yield chunk

    mock_resp = type("MockStreamResponse", (), {
        "status_code": 200,
        "headers": {"content-type": "text/event-stream"},
        "aiter_bytes": staticmethod(aiter_func),
        "aread": AsyncMock(return_value=b""),
    })

    class _MockCM:
        async def __aenter__(self):
            return mock_resp()
        async def __aexit__(self, *args):
            pass

    return _MockCM(), mock_resp()


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


async def _collect_streamed_chunks(resp):
    """Collect all chunks from a StreamingResponse."""
    collected = b""
    async for chunk in resp.body_iterator:
        collected += chunk
    return collected


def _assert_no_unretrieved_warnings(task_exception_logger):
    """Assert no 'Task exception was never retrieved' warnings were logged."""
    warning_calls = [
        call for call in task_exception_logger.warning.call_args_list
        if "exception was never retrieved" in str(call)
    ]
    debug_calls = [
        call for call in task_exception_logger.debug.call_args_list
        if "exception was never retrieved" in str(call)
    ]
    assert not warning_calls, (
        f"No 'Task exception was never retrieved' warnings expected, "
        f"got {len(warning_calls)}: {warning_calls}"
    )
    assert not debug_calls, (
        f"No 'Task exception was never retrieved' debug messages expected, "
        f"got {len(debug_calls)}: {debug_calls}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_abandoned_future_cleaned_up_on_stream_error(monkeypatch):
    """AC1+AC5: Pending _stream_iter future cleaned up on stream error.

    When a stream error occurs mid-stream (e.g., httpx.RemoteProtocolError),
    the exception handler yields finish_reason='error', and the finally block
    must retrieve the pending _stream_iter future's exception to prevent
    "Task exception was never retrieved" warnings.
    """
    from proxy.router import proxy_to_local

    async def _error_aiter():
        yield b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n\n'
        raise httpx.RemoteProtocolError("Simulated stream error")

    cm, resp = _make_mock_cm(_error_aiter)
    monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, resp)))

    task_exception_logger = MagicMock()
    monkeypatch.setattr("asyncio.log.logger", task_exception_logger)

    response = await proxy_to_local(
        _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=True),
        "v1/chat/completions",
    )

    collected = await _collect_streamed_chunks(response)

    # Verify the error handling still works (AC5)
    assert b"Hello" in collected
    assert b'"error"' in collected

    # Verify NO "Task exception was never retrieved" warnings
    _assert_no_unretrieved_warnings(task_exception_logger)


@pytest.mark.asyncio
async def test_abandoned_future_cleaned_up_on_read_timeout(monkeypatch):
    """AC5: Works with httpx.ReadTimeout as well."""
    from proxy.router import proxy_to_local

    async def _error_aiter():
        yield b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n\n'
        raise httpx.ReadTimeout("Simulated read timeout")

    cm, resp = _make_mock_cm(_error_aiter)
    monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, resp)))

    task_exception_logger = MagicMock()
    monkeypatch.setattr("asyncio.log.logger", task_exception_logger)

    response = await proxy_to_local(
        _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=True),
        "v1/chat/completions",
    )

    collected = await _collect_streamed_chunks(response)

    assert b"Hello" in collected
    assert b'"error"' in collected
    _assert_no_unretrieved_warnings(task_exception_logger)


@pytest.mark.asyncio
async def test_normal_completion_no_warnings(monkeypatch):
    """AC3: Normal completion produces no unretrieved exception warnings."""
    from proxy.router import proxy_to_local

    async def _normal_aiter():
        yield b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n\n'
        yield b'data: {"choices": [{"delta": {"content": " world"}, "index": 0}]}\n\n'
        yield b'data: {"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}\n\n'
        yield b"data: [DONE]\n\n"

    cm, resp = _make_mock_cm(_normal_aiter)
    monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, resp)))

    task_exception_logger = MagicMock()
    monkeypatch.setattr("asyncio.log.logger", task_exception_logger)

    response = await proxy_to_local(
        _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=True),
        "v1/chat/completions",
    )

    collected = await _collect_streamed_chunks(response)

    assert b"Hello" in collected
    assert b"world" in collected
    _assert_no_unretrieved_warnings(task_exception_logger)


@pytest.mark.asyncio
async def test_generator_exit_no_warnings(monkeypatch):
    """AC1+AC3: GeneratorExit (generator.close()) produces no warnings.

    When the streaming generator is forcibly closed (e.g., by the FastAPI
    framework on client disconnect), the GeneratorExit handler runs and
    the finally block must clean up the pending _stream_iter future.
    """
    from proxy.router import proxy_to_local

    async def _normal_aiter():
        yield b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n\n'
        yield b'data: {"choices": [{"delta": {"content": " world"}, "index": 0}]}\n\n'

    cm, resp = _make_mock_cm(_normal_aiter)
    monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, resp)))

    task_exception_logger = MagicMock()
    monkeypatch.setattr("asyncio.log.logger", task_exception_logger)

    response = await proxy_to_local(
        _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=True),
        "v1/chat/completions",
    )

    # Read one chunk then close the generator (simulates client disconnect)
    gen = response.body_iterator.__aiter__()
    chunk = await gen.__anext__()
    assert chunk is not None

    # Close the generator — this triggers GeneratorExit path
    await gen.aclose()

    # Give asyncio a moment to process any pending callbacks
    await asyncio.sleep(0.01)

    _assert_no_unretrieved_warnings(task_exception_logger)


@pytest.mark.asyncio
async def test_idle_timeout_cleans_up_pending_future(monkeypatch):
    """AC1+AC2: Pending _stream_iter future cleaned up on idle timeout.

    When the streaming loop exits via idle timeout (heartbeat fires and
    remaining budget is exhausted), the pending _stream_iter future must
    not be cancelled (CRITICAL constraint) and must not produce warnings.
    """
    from proxy.router import proxy_to_local

    # Use a tight config: short idle timeout, quick heartbeat
    monkeypatch.setattr(server, "config", {
        "server": {
            **BASE_SERVER_CONFIG["server"],
            "stream_idle_timeout_seconds": 0.2,
            "stream_heartbeat_interval_seconds": 0.05,
            "session_guardrail_max_runtime_seconds": 1.0,
        }
    })

    # An iterator that yields one chunk then hangs forever
    # (second __anext__() never returns)
    _hang_event = asyncio.Event()

    async def _hanging_aiter():
        yield b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n\n'
        await _hang_event.wait()

    cm, resp = _make_mock_cm(_hanging_aiter)
    monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, resp)))

    task_exception_logger = MagicMock()
    monkeypatch.setattr("asyncio.log.logger", task_exception_logger)

    response = await proxy_to_local(
        _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=True),
        "v1/chat/completions",
    )

    collected = await _collect_streamed_chunks(response)

    # First chunk should be delivered
    assert b"Hello" in collected

    # The idle timeout should fire, and no warnings should appear
    _assert_no_unretrieved_warnings(task_exception_logger)

    # Clean up the hanging event
    _hang_event.set()


@pytest.mark.asyncio
async def test_dispatch_lease_refreshed_during_streaming(monkeypatch):
    """The dispatch lease expires_at should be refreshed on each data chunk.

    LP-0MRDKV44T003FRBP: When a streaming response delivers real data,
    the dispatch record's expires_at must be extended so long-running
    streams (> local_dispatch_lease_timeout_seconds) do not lose their
    lease mid-stream.
    """
    from proxy.router import proxy_to_local

    # Set up dispatch tracking on server state
    monkeypatch.setattr(server, "local_dispatch_records", {})
    monkeypatch.setattr(server, "local_dispatch_records_lock", asyncio.Lock())
    monkeypatch.setattr(server, "local_active_queries", 0)
    monkeypatch.setattr(server, "local_active_queries_lock", asyncio.Lock())

    async def _aiter():
        yield b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n\n'
        yield b'data: {"choices": [{"delta": {"content": " world"}, "index": 0}]}\n\n'
        yield b'data: {"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}\n\n'
        yield b"data: [DONE]\n\n"

    cm, resp = _make_mock_cm(_aiter)
    monkeypatch.setattr(
        "proxy.router._call_with_backend_retries",
        AsyncMock(return_value=(cm, resp)),
    )

    # Use an explicit session so the dispatch lease path is exercised
    monkeypatch.setattr(
        "proxy.router._handle_session",
        AsyncMock(
            return_value={
                "session_id": "test-session-lease-refresh",
                "session_id_header": "test-session-lease-refresh",
                "session_explicit": True,
                "session_created": False,
                "is_delta_request": False,
                "session_fallback_reason": None,
                "delta_messages": [],
                "original_message_count": 1,
                "body_override": None,
                "body_json": None,
            }
        ),
    )

    monkeypatch.setattr(
        "proxy.router._build_slot_context",
        MagicMock(return_value=(None, None, 3.0)),
    )
    monkeypatch.setattr(
        "proxy.router._check_slot_availability",
        AsyncMock(return_value=None),
    )

    response = await proxy_to_local(
        _dummy_request(
            {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            stream=True,
        ),
        "v1/chat/completions",
    )

    # Consume the first chunk (triggers lease refresh in stream_generator)
    gen = response.body_iterator.__aiter__()
    chunk1 = await gen.__anext__()
    assert chunk1 is not None

    # After consuming one chunk, the dispatch record should exist and have
    # a future expires_at (refreshed when _has_actual_data was True).
    assert "test-session-lease-refresh" in server.local_dispatch_records, (
        "Dispatch record should exist for the explicit session"
    )
    record = server.local_dispatch_records["test-session-lease-refresh"]
    assert record["expires_at"] > time.monotonic(), (
        f"Dispatch lease expires_at ({record['expires_at']}) should be in the "
        f"future, but time.monotonic()={time.monotonic()}"
    )

    # Consume remaining chunks (clean shutdown)
    try:
        while True:
            await gen.__anext__()
    except StopAsyncIteration:
        pass


@pytest.mark.asyncio
async def test_dispatch_lease_adaptive_timeout_large_prompt(monkeypatch):
    """The dispatch lease initial timeout should be extended for large prompts.

    LP-0MRDUQ9QC003LDDP: Large-context requests with cache prefill >180s must
    have an extended initial lease timeout so the lease covers the prefill phase
    before any data chunks arrive. For a ~48K-token prompt, the extension is
    approximately 48K * 0.015s = 720s, giving a total lease of ~900s.
    """
    from proxy.router_helpers import _try_acquire_local_dispatch, _get_lease_timeout_seconds

    # Set up dispatch tracking on server state
    monkeypatch.setattr(server, "local_dispatch_records", {})
    monkeypatch.setattr(server, "local_dispatch_records_lock", asyncio.Lock())
    monkeypatch.setattr(server, "local_active_queries", 0)
    monkeypatch.setattr(server, "local_active_queries_lock", asyncio.Lock())

    # Configure server with base 180s lease timeout
    monkeypatch.setattr(
        server, "config",
        {"server": {"local_dispatch_lease_timeout_seconds": 180}},
    )
    monkeypatch.setattr(server, "logger", MagicMock())

    # Large prompt (~48K tokens: 48000 * 4 chars/token = 192000 chars)
    large_body = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "hello " * 48000},
        ],
    }

    acquired, owner, count, retry = await _try_acquire_local_dispatch(
        server,
        max_local=1,
        session_key="test-large-prompt",
        backend="local",
        body_json=large_body,
    )

    assert acquired is True
    record = server.local_dispatch_records["test-large-prompt"]

    base_timeout = _get_lease_timeout_seconds(server)
    actual_lease = record["expires_at"] - record["started_at"]

    # For a 48K-token prompt: base=180s, extension ~720s => ~900s total
    # The lease should be significantly longer than the base 180s
    assert actual_lease > base_timeout + 300, (
        f"Expected lease extended for large prompt (48K tokens), "
        f"got {actual_lease:.0f}s (base={base_timeout}s)"
    )
    # But it shouldn't exceed the max (1500s)
    assert actual_lease <= 1500, (
        f"Expected lease capped at 1500s max, got {actual_lease:.0f}s"
    )


@pytest.mark.asyncio
async def test_dispatch_lease_adaptive_timeout_small_prompt(monkeypatch):
    """Small prompts should use the base lease timeout without significant extension.

    LP-0MRDUQ9QC003LDDP: For small requests (<180s), the lease timeout
    should remain at the base 180s without significant adaptive extension.
    """
    from proxy.router_helpers import _try_acquire_local_dispatch, _get_lease_timeout_seconds

    # Set up dispatch tracking on server state
    monkeypatch.setattr(server, "local_dispatch_records", {})
    monkeypatch.setattr(server, "local_dispatch_records_lock", asyncio.Lock())
    monkeypatch.setattr(server, "local_active_queries", 0)
    monkeypatch.setattr(server, "local_active_queries_lock", asyncio.Lock())

    monkeypatch.setattr(
        server, "config",
        {"server": {"local_dispatch_lease_timeout_seconds": 180}},
    )
    monkeypatch.setattr(server, "logger", MagicMock())

    # Small prompt (~10 tokens)
    small_body = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
        ],
    }

    acquired, owner, count, retry = await _try_acquire_local_dispatch(
        server,
        max_local=1,
        session_key="test-small-prompt",
        backend="local",
        body_json=small_body,
    )

    assert acquired is True
    record = server.local_dispatch_records["test-small-prompt"]

    base_timeout = _get_lease_timeout_seconds(server)
    actual_lease = record["expires_at"] - record["started_at"]

    # For a small prompt, the lease should be close to the base timeout
    # (extension is negligible: ~10 tokens * 0.015 = 0.15s)
    assert abs(actual_lease - base_timeout) < 10, (
        f"Expected lease close to base timeout for small prompt, "
        f"got {actual_lease:.0f}s (base={base_timeout}s)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Structural verification test
# ═══════════════════════════════════════════════════════════════════════════════


def test_finally_block_uses_exception_not_cancel():
    """Structural test: Verify the finally block uses .exception() not .cancel().

    The abandonment fix (LP-0MRCMKG9O004XE0Q) replaces _stream_iter.cancel()
    with _stream_iter.exception() to retrieve the exception without cancelling
    the in-flight httpx read (CRITICAL constraint).
    """
    import os

    test_dir = os.path.dirname(os.path.abspath(__file__))
    router_path = os.path.join(test_dir, "..", "proxy", "router.py")

    with open(router_path) as f:
        content = f.read()

    lines = content.splitlines()

    # Find the cleanup section comment
    cleanup_section_start = None
    for i, line in enumerate(lines):
        if "Clean up the pending _stream_iter" in line:
            cleanup_section_start = i
            break

    assert cleanup_section_start is not None, (
        "Could not find _stream_iter cleanup section"
    )

    # Look at the next 25 lines for .cancel() and .exception()
    cancel_found = False
    exception_found = False
    for j in range(cleanup_section_start, min(cleanup_section_start + 25, len(lines))):
        line = lines[j]
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if ".cancel()" in stripped:
            cancel_found = True
        if ".exception()" in stripped:
            exception_found = True

    assert not cancel_found, (
        "Cleanup section should NOT use .cancel() on _stream_iter "
        "(CRITICAL constraint from LP-0MQTHP828000JYM6)"
    )
    assert exception_found, (
        "Cleanup section should use .exception() to retrieve exception "
        "without cancelling the in-flight httpx read"
    )
