"""Tests for the stream idle timeout with heartbeat keepalive.

Tests cover:
- Heartbeat events are emitted during long pre-fill (keeps client alive)
- Between-chunks timeout is shorter than pre-fill timeout
- Completely hung upstream is caught after max_runtime_seconds
- Normal streaming (with [DONE]) completes without interference
"""

import asyncio
import json
from typing import AsyncGenerator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===================================================================
# Mock helpers
# ===================================================================

# Sentinel used to make an async iterator hang forever
_HANG_SENTINEL = object()


class _HangAsyncIterator:
    """An async iterator that yields given items then hangs forever."""

    def __init__(self, items: List[bytes]):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._items:
            return self._items.pop(0)
        # Block forever
        await asyncio.Event().wait()
        raise StopAsyncIteration  # unreachable


class _EmptyHangAsyncIterator:
    """An async iterator that immediately hangs (no items at all)."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.Event().wait()
        raise StopAsyncIteration  # unreachable


class _HangIterator:
    """Async iterator that yields given items, then blocks forever."""

    def __init__(self, items: List[bytes], prefill_delay: float = 0.0):
        self._items = list(items)
        self._prefill_delay = prefill_delay
        self._done_prefill = False
        self._event = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._done_prefill:
            self._done_prefill = True
            if self._prefill_delay > 0:
                await asyncio.sleep(self._prefill_delay)
        if self._items:
            return self._items.pop(0)
        # Block forever — never raise StopAsyncIteration
        await self._event.wait()
        return b""  # unreachable


class MockHangAfterChunksResponse:
    """aiter_bytes yields given chunks then hangs via custom iterator."""

    def __init__(self, chunks: List[str], prefill_delay: float = 0.0):
        self.chunks = chunks
        self.prefill_delay = prefill_delay
        self.status_code = 200
        self.headers = {"content-type": "text/event-stream"}

    def aiter_bytes(self):
        return _HangIterator(
            [c.encode("utf-8") for c in self.chunks],
            prefill_delay=self.prefill_delay,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


class MockNormalStreamResponse:
    """Simulates a normal httpx response that ends with [DONE]."""

    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.status_code = 200
        self.headers = {"content-type": "text/event-stream"}

    async def aiter_bytes(self) -> AsyncGenerator[bytes, None]:
        for chunk in self.chunks:
            yield chunk.encode("utf-8")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


class MockStreamContextManager:
    """Async context manager returned by ``client.stream()``."""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return None


class MockStreamingAsyncClient:
    """Mock ``httpx.AsyncClient`` that supports ``.stream()``."""

    def __init__(self, response):
        self._response = response
        self.calls: list = []

    def stream(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        return MockStreamContextManager(self._response)

    async def aclose(self):
        pass


# ===================================================================
# Mock server config helpers
# ===================================================================


def _make_config(
    stream_idle_timeout: float = 0.1,
    max_runtime: float = 3600.0,
    heartbeat_interval: float = 0.05,
) -> dict:
    """Return a minimal server config for testing (short intervals)."""
    return {
        "server": {
            "llama_router_mode": False,
            "llama_server_port": 8080,
            "session_guardrail_max_runtime_seconds": max_runtime,
            "session_guardrail_max_completion_tokens": 2048,
            "session_guardrail_repetition_min_pattern_chars": 64,
            "session_guardrail_repetition_min_repeats": 10,
            "session_guardrail_invalidate_on_cutoff": False,
            "session_guardrail_invalidate_on_repetition": False,
            "session_guardrail_max_token_rate": 0,
            "stream_idle_timeout_seconds": stream_idle_timeout,
            "stream_heartbeat_interval_seconds": heartbeat_interval,
        }
    }


async def _collect_stream(resp) -> List[str]:
    """Collect all chunks from a streaming response as strings."""
    collected = []
    async for chunk in resp.body_iterator:
        raw = chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="replace")
        collected.append(raw)
    return collected


async def _make_mock_request(body: dict):
    """Create a mock FastAPI request object."""
    from fastapi import Request as FastAPIRequest

    encoded = json.dumps(body).encode("utf-8")
    mock_req = MagicMock(spec=FastAPIRequest)
    mock_req.method = "POST"
    mock_req.url = type("U", (), {"path": "/v1/chat/completions"})()
    mock_req.headers = {"content-type": "application/json"}
    mock_req._body = encoded

    async def mock_body():
        return mock_req._body

    mock_req.body = mock_body
    return mock_req


def _setup_basic_mocks(monkeypatch):
    """Set up the common server state mocks needed for proxy_to_local."""
    from proxy import server as srv_module
    import proxy.router as router_mod

    monkeypatch.setattr(srv_module, "llama_process", MagicMock())
    monkeypatch.setattr(srv_module, "backend_ready", True)
    monkeypatch.setattr(srv_module, "current_model", "test-model")
    monkeypatch.setattr(srv_module, "active_queries", 0)
    monkeypatch.setattr(srv_module, "active_queries_lock", asyncio.Lock())

    monkeypatch.setattr(router_mod, "_is_self_healing_active", lambda: False)
    monkeypatch.setattr(router_mod, "_get_job_scheduler", lambda: None)
    monkeypatch.setattr(router_mod, "_check_slot_availability", AsyncMock(return_value=None))

    return srv_module, router_mod


# ===================================================================
# Tests
# ===================================================================


@pytest.mark.asyncio
async def test_heartbeat_during_long_prefill(monkeypatch):
    """
    When the upstream takes a long time for pre-fill (first chunk delayed),
    the proxy should emit heartbeat events to keep the client alive.
    """
    from proxy.router import proxy_to_local

    srv_module, router_mod = _setup_basic_mocks(monkeypatch)
    monkeypatch.setattr(srv_module, "config", _make_config(
        stream_idle_timeout=0.5,
        max_runtime=3600,
        heartbeat_interval=0.05,
    ))

    # Simulate a 0.12s pre-fill delay — should get ~2 heartbeats
    chunk1 = 'data: {"choices":[{"delta":{"content":"hello"},"index":0}]}\n\n'
    mock_response = MockHangAfterChunksResponse([chunk1], prefill_delay=0.12)
    mock_client = MockStreamingAsyncClient(mock_response)

    with patch("proxy.router.httpx.AsyncClient", return_value=mock_client):
        mock_req = await _make_mock_request({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        })
        resp = await proxy_to_local(mock_req, "v1/chat/completions")

    assert resp is not None
    assert resp.status_code == 200

    collected = await _collect_stream(resp)

    # Should have heartbeats before the content chunk
    heartbeat_count = sum(
        1 for c in collected
        if '"type":"heartbeat"' in c or '"type": "heartbeat"' in c
    )
    assert heartbeat_count >= 1, (
        f"Expected at least 1 heartbeat during pre-fill, got {heartbeat_count} in {collected}"
    )

    # The content chunk should still be delivered
    assert any('"content":"hello"' in c or '"content": "hello"' in c for c in collected), (
        "Content chunk missing after pre-fill delay"
    )


@pytest.mark.asyncio
async def test_between_chunks_timeout_shorter_than_prefill(monkeypatch):
    """
    After the first chunk arrives, idle timeout should use the shorter
    stream_idle_timeout_seconds, not the longer max_runtime_seconds budget.
    """
    from proxy.router import proxy_to_local

    srv_module, router_mod = _setup_basic_mocks(monkeypatch)
    # stream_idle_timeout of 0.05s, max_runtime of 3600s
    monkeypatch.setattr(srv_module, "config", _make_config(
        stream_idle_timeout=0.05,
        max_runtime=3600,
        heartbeat_interval=0.05,
    ))

    # Response yields one chunk then hangs
    chunk1 = 'data: {"choices":[{"delta":{"content":"hello"},"index":0}]}\n\n'
    mock_response = MockHangAfterChunksResponse([chunk1])
    mock_client = MockStreamingAsyncClient(mock_response)

    with patch("proxy.router.httpx.AsyncClient", return_value=mock_client):
        mock_req = await _make_mock_request({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        })
        resp = await proxy_to_local(mock_req, "v1/chat/completions")

    assert resp is not None
    assert resp.status_code == 200

    collected = await _collect_stream(resp)

    # Should have the content chunk + a synthesised finish_reason
    # (the hang after first chunk should be caught by the short timeout)
    assert len(collected) >= 2, (
        f"Expected at least 2 chunks (content + finish_reason), got {len(collected)}"
    )
    assert any("finish_reason" in c for c in collected), (
        "Expected finish_reason after between-chunks timeout"
    )


@pytest.mark.asyncio
async def test_complete_hang_caught_by_runtime_budget(monkeypatch):
    """
    An upstream that sends no data at all (completely hung) should be
    caught by the max_runtime_seconds budget, not hang forever.
    """
    from proxy.router import proxy_to_local

    srv_module, router_mod = _setup_basic_mocks(monkeypatch)
    # Very short max_runtime for testing (0.2s), short heartbeat
    monkeypatch.setattr(srv_module, "config", _make_config(
        stream_idle_timeout=999,
        max_runtime=0.2,
        heartbeat_interval=0.05,
    ))

    # Mock that responds with aiter never returning
    mock_response = MockHangAfterChunksResponse([])
    mock_client = MockStreamingAsyncClient(mock_response)

    with patch("proxy.router.httpx.AsyncClient", return_value=mock_client):
        mock_req = await _make_mock_request({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        })
        resp = await proxy_to_local(mock_req, "v1/chat/completions")

    assert resp is not None
    assert resp.status_code == 200

    collected = await _collect_stream(resp)

    # Should have heartbeats during the wait and finish_reason at the end
    assert any("finish_reason" in c for c in collected), (
        "Expected finish_reason after complete hang timeout"
    )
    assert any("heartbeat" in c for c in collected), (
        "Expected heartbeats during the hang"
    )


@pytest.mark.asyncio
async def test_normal_stream_unaffected(monkeypatch):
    """Normal streaming with proper [DONE] should not be affected."""
    from proxy.router import proxy_to_local

    srv_module, router_mod = _setup_basic_mocks(monkeypatch)
    monkeypatch.setattr(srv_module, "config", _make_config(
        stream_idle_timeout=0.1,
        max_runtime=3600,
        heartbeat_interval=0.05,
    ))

    chunk1 = 'data: {"choices":[{"delta":{"content":"hello"},"index":0}]}\n\n'
    chunk_done = "data: [DONE]\n\n"
    mock_response = MockNormalStreamResponse([chunk1, chunk_done])
    mock_client = MockStreamingAsyncClient(mock_response)

    with patch("proxy.router.httpx.AsyncClient", return_value=mock_client):
        mock_req = await _make_mock_request({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        })
        resp = await proxy_to_local(mock_req, "v1/chat/completions")

    assert resp is not None
    assert resp.status_code == 200

    collected = await _collect_stream(resp)

    assert len(collected) >= 2
    assert any("[DONE]" in c or "finish_reason" in c for c in collected), (
        "Expected [DONE] or finish_reason in normal stream output"
    )
