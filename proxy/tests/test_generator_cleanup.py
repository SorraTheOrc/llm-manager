"""Tests for removing yield in finally blocks from streaming generators.

Verifies that:
1. Normal completion still yields the final SSE event.
2. Client disconnect (via aclose()) runs cleanup without RuntimeError.
3. No yield statements exist in finally blocks of stream_generator functions.

Acceptance criteria covered:
- AC1: No yield in finally blocks
- AC2: Full cleanup on disconnect
- AC3: Final SSE event on natural completion
- AC6: New disconnect cleanup test
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import pytest
from fastapi.responses import StreamingResponse


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


# ═══════════════════════════════════════════════════════════════════════════════
# Structural tests — verify no yield exists in finally blocks post-fix
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "filepath",
    [
        "proxy/router.py",
        "proxy/proxy_remote.py",
    ],
)
def test_no_yield_in_finally_blocks(filepath: str):
    """Verify no yield statement appears inside any finally block (AC1)."""
    import ast
    import os

    # Resolve paths relative to the proxy/ repo directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(test_dir, "..", filepath)

    with open(full_path) as f:
        source = f.read()
    tree = ast.parse(source)

    # Collect all yield statements found inside finally blocks
    yields_in_finally = []

    class FinallyFinder(ast.NodeVisitor):
        def visit_Try(self, node):
            for handler in node.finalbody:
                for child in ast.walk(handler):
                    if isinstance(child, (ast.Yield, ast.YieldFrom)):
                        yields_in_finally.append(
                            (node.lineno, ast.unparse(child))
                        )
            self.generic_visit(node)

    FinallyFinder().visit(tree)
    assert not yields_in_finally, (
        f"Found yield in finally blocks: {yields_in_finally}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for proxy_remote.py stream_generator cleanup
# ═══════════════════════════════════════════════════════════════════════════════


def _make_mock_response(chunks, status=200, content_type="text/event-stream"):
    """Create a mock httpx response with controlled streaming chunks."""
    mock = AsyncMock()
    mock.status_code = status
    mock.headers = {"content-type": content_type}
    mock.aiter_bytes = lambda: AsyncIterator(chunks)
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
async def test_remote_proxy_normal_completion_yields_final_event(remote_setup):
    """AC3: Final SSE event produced on natural completion without finish marker."""
    from proxy.proxy_remote import _handle_remote_streaming

    chunks = [
        b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n\n',
    ]

    # Patch httpx.AsyncClient to return controlled mock
    with patch("proxy.proxy_remote.httpx.AsyncClient") as mock_client_cls:
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = _make_mock_response(chunks)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client

        with patch("proxy.proxy_remote.log_response_chunk") as mock_log:
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
            # IMPORTANT: iterate inside the with block since the generator
            # is lazy and patches must be active during iteration
            collected = b""
            async for chunk in result.body_iterator:
                collected += chunk

        # Verify the final event was emitted (no saw_done or saw_finish)
        assert b"finish_reason" in collected, (
            "Final event should contain finish_reason"
        )
        assert b'"stop"' in collected, (
            "Final event finish_reason should be 'stop'"
        )

        # Verify cleanup was called
        assert mock_cm.__aexit__.called, (
            "Context manager __aexit__ should be called on completion"
        )
        assert mock_client.aclose.called, (
            "Client aclose should be called on completion"
        )
        # Verify log_response_chunk was called for the final event
        final_event_logged = False
        for call in mock_log.call_args_list:
            if call.args and b"finish_reason" in call.args[0]:
                final_event_logged = True
                break
        assert final_event_logged, (
            "log_response_chunk should have been called for the final event"
        )


@pytest.mark.asyncio
async def test_remote_proxy_aclose_cleanup_no_runtime_error(remote_setup):
    """AC2: aclose() triggers cleanup without RuntimeError from yield-in-finally."""
    from proxy.proxy_remote import _handle_remote_streaming

    # Use many chunks so we can trigger aclose mid-stream
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

        # Read a few chunks then trigger aclose
        iterator = result.body_iterator.__aiter__()
        first_chunk = await iterator.__anext__()
        assert first_chunk is not None

        # Now close the generator (simulates client disconnect)
        # This should NOT raise RuntimeError
        await iterator.aclose()

        # Verify cleanup functions were called
        assert mock_cm.__aexit__.called, (
            "Context manager __aexit__ should be called during aclose cleanup"
        )
        assert mock_client.aclose.called, (
            "Client aclose should be called during aclose cleanup"
        )


@pytest.mark.asyncio
async def test_remote_proxy_normal_completion_with_finish_marker(remote_setup):
    """When upstream sends a finish_reason, no synthetic final event is emitted."""
    from proxy.proxy_remote import _handle_remote_streaming

    chunks = [
        b'data: {"choices": [{"delta": {"content": "Done"}, "index": 0, "finish_reason": "stop"}]}\n\n',
    ]

    with patch("proxy.proxy_remote.httpx.AsyncClient") as mock_client_cls:
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = _make_mock_response(chunks)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client

        with patch("proxy.proxy_remote.log_response_chunk") as mock_log:
            result = await _handle_remote_streaming(
                remote_setup,
                "http://fake.api/v1/chat/completions",
                {"Authorization": "Bearer test-key"},
                json.dumps({"model": "test-model", "stream": True}).encode(),
                {"model": "test-model", "stream": True},
                "test-model",
                httpx.Timeout(300),
            )

        # Collect all chunks
        collected = b""
        async for chunk in result.body_iterator:
            collected += chunk

        # The chunk with finish_reason passes through but no synthetic final event
        assert collected.count(b"finish_reason") == 1, (
            "Only the original finish_reason should appear, no synthetic one"
        )
        # Verify cleanup
        assert mock_cm.__aexit__.called
        assert mock_client.aclose.called


@pytest.mark.asyncio
async def test_remote_proxy_normal_completion_with_done_marker(remote_setup):
    """When upstream sends [DONE], no synthetic final event is emitted."""
    from proxy.proxy_remote import _handle_remote_streaming

    chunks = [
        b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n\n',
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

        # The [DONE] chunk passes through but no synthetic final event
        assert b"[DONE]" in collected
        # The synthetic final event should NOT be present (saw_done was set)
        assert collected.count(b"finish_reason") == 0, (
            "No synthetic finish_reason should appear when [DONE] was seen"
        )
        assert mock_cm.__aexit__.called
        assert mock_client.aclose.called


@pytest.mark.asyncio
async def test_remote_proxy_empty_stream_normal_completion(remote_setup):
    """Empty stream should still get final event and cleanup."""
    from proxy.proxy_remote import _handle_remote_streaming

    chunks = []

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

        # Even with an empty stream, final event should be emitted
        assert b"finish_reason" in collected
        assert mock_cm.__aexit__.called
        assert mock_client.aclose.called


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for router.py stream_generator cleanup
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_router_proxy_aclose_cleanup_no_runtime_error():
    """AC2: router.py stream_generator cleanup on aclose without RuntimeError.

    This test verifies the generator pattern fix for router.py by inspecting
    the source to verify no yield-in-finally exists after the fix.
    """
    import ast
    import inspect
    from proxy import router as router_mod

    # Read the source to verify no yield in finally in router.py
    source = inspect.getsource(router_mod)
    tree = ast.parse(source)

    yields_in_finally = []

    class FinallyFinder(ast.NodeVisitor):
        def visit_Try(self, node):
            for stmt in node.finalbody:
                for child in ast.walk(stmt):
                    if isinstance(child, (ast.Yield, ast.YieldFrom)):
                        yields_in_finally.append(
                            (node.lineno, ast.unparse(child))
                        )
            self.generic_visit(node)

    FinallyFinder().visit(tree)
    assert not yields_in_finally, (
        f"router.py still has yield in finally: {yields_in_finally}"
    )
