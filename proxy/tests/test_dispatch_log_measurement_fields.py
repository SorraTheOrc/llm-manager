"""Dispatch-event measurement fields for the idle-admission impact tool.

LP-0MVGHXD59008UOUD (measuring the idle-aware local admission override from
LP-0MVGD9Z8Z0055Y5J) needs two facts attached to every local dispatch:

  * ``priority=`` — so the interactive tier's first-byte latency can be
    compared before/after (AC4);
  * ``estimated_tokens=`` — so an idle-admitted dispatch can be compared with
    non-idle dispatches **in the same context bucket** (AC2).

Both values are already computed inside ``proxy_to_local``; this change emits
them on the existing ``dispatch_first_byte_ms`` line. The line must stay
backwards compatible: every pre-existing field keeps its position and format
so existing log consumers (``scripts/dispatch_latency_verification.py``,
``proxy-usage-analysis``) parse unchanged lines.

These tests drive a real local stream through ``proxy_to_local`` with a mocked
upstream and assert the *observable* log line, not the source text.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _MockStreamResponse:
    """Simulates an httpx streaming response that yields SSE bytes."""

    def __init__(self, chunks: list[str]):
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


class _MockStreamContext:
    def __init__(self, response: _MockStreamResponse):
        self._response = response

    async def __aenter__(self) -> _MockStreamResponse:
        return self._response

    async def __aexit__(self, *exc):
        return None


class _MockStreamingClient:
    """Minimal ``httpx.AsyncClient`` stand-in that streams fixed SSE chunks."""

    def __init__(self, chunks: list[str], **kwargs):
        self._chunks = chunks

    def stream(self, method: str, url: str, **kwargs) -> _MockStreamContext:
        return _MockStreamContext(_MockStreamResponse(self._chunks))

    async def aclose(self):
        pass


@pytest.fixture(autouse=True)
def _server_state(monkeypatch):
    """Minimal local-dispatch server state (mirrors the token-rate tests)."""
    from proxy import server as srv_module

    config = {
        "server": {
            "llama_router_mode": False,
            "llama_server_port": 8080,
        }
    }
    monkeypatch.setattr(srv_module, "config", config)
    monkeypatch.setattr(srv_module, "llama_process", MagicMock())
    srv_module.llama_process.poll.return_value = None
    monkeypatch.setattr(srv_module, "backend_ready", True)
    monkeypatch.setattr(srv_module, "current_model", "test-model")
    monkeypatch.setattr(srv_module, "active_queries", 0)
    monkeypatch.setattr(srv_module, "local_active_queries", 0)
    monkeypatch.setattr(srv_module, "local_generating_queries", 0)
    monkeypatch.setattr(srv_module, "local_prefill_in_flight", {})
    monkeypatch.setattr(srv_module, "local_dispatch_records", {})
    monkeypatch.setattr(srv_module, "active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv_module, "local_active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv_module, "local_generating_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv_module, "local_prefill_in_flight_lock", asyncio.Lock())
    monkeypatch.setattr(srv_module, "local_dispatch_records_lock", asyncio.Lock())
    monkeypatch.setattr(srv_module, "local_generating_sessions", set())

    mock_sm = MagicMock()
    from proxy.session_manager import Session

    mock_session = MagicMock(spec=Session)
    mock_session.session_id = "measure-session"
    mock_session.message_count = 0
    mock_session.restore_confirmed = False
    mock_session.messages = []
    mock_session.invalidated = False
    mock_sm.get_or_create = AsyncMock(return_value=(mock_session, True))
    mock_sm.get = AsyncMock(return_value=mock_session)
    monkeypatch.setattr(srv_module, "session_manager", mock_sm)


def _sse_chunks() -> list[str]:
    payload = json.dumps({"choices": [{"index": 0, "delta": {"content": "hello"}}]})
    final = json.dumps(
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    )
    return [f"data: {payload}\n\n", f"data: {final}\n\n", "data: [DONE]\n\n"]


async def _drive_local_stream(headers: dict | None = None):
    """Run one streaming local dispatch and return the captured log records."""
    from fastapi import Request as FastAPIRequest
    from proxy.router import proxy_to_local

    body = json.dumps(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }
    ).encode("utf-8")

    request = MagicMock(spec=FastAPIRequest)
    request.method = "POST"
    request.url = type("U", (), {"path": "/v1/chat/completions"})()
    request.headers = headers or {"content-type": "application/json"}
    request._body = body

    async def _body():
        return body

    request.body = _body
    request.state = MagicMock()

    client = _MockStreamingClient(_sse_chunks())
    with patch("proxy.router.httpx.AsyncClient", return_value=client):
        response = await proxy_to_local(request, "v1/chat/completions")
        async for _ in response.body_iterator:
            pass
    return response


@pytest.mark.asyncio
async def test_dispatch_line_carries_priority_and_prompt_size(caplog):
    """The dispatch event carries the AC4 priority and AC2 prompt size."""
    import logging

    caplog.set_level(logging.INFO)
    response = await _drive_local_stream()
    assert response.status_code == 200

    lines = [
        record.getMessage()
        for record in caplog.records
        if "dispatch_first_byte_ms=" in record.getMessage()
    ]
    assert len(lines) == 1, lines
    line = lines[0]
    assert "priority=" in line
    assert "estimated_tokens=" in line
    # The pre-existing contract is untouched and still positioned first.
    assert line.startswith("dispatch_first_byte_ms=")
    assert " dispatch_to_first_byte_ms=" in line
    assert " cache_state=" in line
    assert " cached_ratio=" in line
    assert " queue_wait_ms=" in line
    assert " session=" in line
    assert " model=" in line


@pytest.mark.asyncio
async def test_dispatch_line_reports_the_requested_priority_tier(caplog, monkeypatch):
    """A high-priority request is logged at the high tier, not the default.

    The effective tier is capped to ``low`` while the configured cheap window
    (``proxy/config.yaml``, 01:00-10:00 local) is active, so an unpinned clock
    made this assertion depend on the wall-clock time the suite happened to run
    at: green by day, red between 01:00 and 10:00 (LP-0MVK7WR8X006TUX9). Pin the
    clock outside the window so the requested tier is what is observed.
    """
    import logging

    from proxy import priority as priority_mod

    class _MiddayDatetime(priority_mod.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 1, 12, 0)

    monkeypatch.setattr(priority_mod, "datetime", _MiddayDatetime)

    caplog.set_level(logging.INFO)
    await _drive_local_stream(
        headers={"content-type": "application/json", "X-Priority": "high"}
    )
    line = next(
        record.getMessage()
        for record in caplog.records
        if "dispatch_first_byte_ms=" in record.getMessage()
    )
    assert "priority=high" in line


@pytest.mark.asyncio
async def test_dispatch_line_prompt_size_is_a_non_negative_integer(caplog):
    """``estimated_tokens=`` is always an integer (the estimator may fail open)."""
    import logging

    caplog.set_level(logging.INFO)
    await _drive_local_stream()
    line = next(
        record.getMessage()
        for record in caplog.records
        if "dispatch_first_byte_ms=" in record.getMessage()
    )
    tokens = line.split("estimated_tokens=")[1].split()[0]
    assert tokens.isdigit()
    assert int(tokens) >= 0
