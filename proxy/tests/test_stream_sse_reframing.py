"""Regression tests for SSE re-framing on the client-facing output path.

LP-0MUOBUPBC002GYTL: the proxy used to forward raw upstream ``aiter_bytes()``
reads verbatim, then append synthetic/retry events. When a read ended mid-event
(or contained a complete event plus the start of the next), a later synthetic
or retry event was concatenated with the dangling partial and the client's SSE
decoder merged both into one event with a non-JSON ``data:`` payload
("malformed server-sent event JSON").

These tests pin the three client-visible cases from the work item:

- AC5  case A: complete event + start of the next in one read, then a stall
  after content → only complete events reach the client and the synthetic
  error is a separate event.
- AC6  case B: an empty-response retry whose first attempt ends mid-event →
  the retry stream's first event is not concatenated with the partial.
- AC7  clean stop: a terminal event and a trailing partial arrive in the same
  read → the incomplete event is not forwarded.
- AC4  the local provider path gets the same guarantee.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import httpx
import proxy.server as server
import pytest
from fastapi import Request
from proxy.proxy_remote import _handle_remote_streaming

# Distinctive marker so a leaked partial is unambiguously detectable.
PARTIAL_MARKER = b"SECRET_PARTIAL_CONTENT"


# ===================================================================
# Shared assertion helpers
# ===================================================================


def _assert_all_events_well_formed(chunks: list[bytes]) -> list[bytes]:
    """Assert every chunk is a complete SSE event and return the payloads.

    A complete event ends in a blank line (``\\n\\n`` or ``\\r\\n\\r\\n``) and
    every ``data:`` payload is valid JSON (or the ``[DONE]`` sentinel).
    """
    payloads: list[bytes] = []
    for chunk in chunks:
        assert chunk.endswith((b"\n\n", b"\r\n\r\n")), (
            f"Forwarded a non-event chunk (not blank-line terminated): {chunk!r}"
        )
        for raw_line in chunk.splitlines():
            line = raw_line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                payloads.append(payload)
                continue
            try:
                json.loads(payload)
            except Exception as exc:  # pragma: no cover - failure path
                raise AssertionError(
                    f"Forwarded a non-JSON data payload: {payload!r}"
                ) from exc
            payloads.append(payload)
    return payloads


def _joined(chunks: list[bytes]) -> bytes:
    return b"".join(chunks)


# ===================================================================
# Remote streaming harness
# ===================================================================


class AsyncChunkIterator:
    """Async iterator yielding pre-defined byte chunks, optionally hanging."""

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


def _make_mock_response(aiter_chunks, hang_after=False):
    mock_resp = MagicMock(spec=httpx.Response)
    type(mock_resp).status_code = PropertyMock(return_value=200)
    mock_resp.headers = {"content-type": "text/event-stream"}
    mock_resp.aiter_bytes = MagicMock(
        return_value=AsyncChunkIterator(aiter_chunks, hang_after=hang_after)
    )
    return mock_resp


class _MockCM:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        return None


def _make_client(responses):
    client = MagicMock(spec=httpx.AsyncClient)
    client.stream = MagicMock(side_effect=[_MockCM(r) for r in responses])
    client.aclose = AsyncMock(return_value=None)
    return client


def _make_request():
    req = MagicMock(spec=Request)
    req.method = "POST"
    req.url.path = "/v1/chat/completions"
    req.is_disconnected = AsyncMock(return_value=False)
    return req


async def _run_remote(client, mock_srv, request, idle_timeout=0.05):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("proxy.proxy_remote.httpx.AsyncClient", lambda *a, **k: client)
        mp.setattr("proxy.proxy_remote._srv", lambda: mock_srv)
        mp.setattr("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock())
        mp.setattr("proxy.proxy_remote.log_response_chunk", MagicMock())
        mp.setattr("proxy.proxy_remote.log_response", MagicMock())
        mp.setattr("proxy.proxy_remote.log_request", MagicMock())
        result = await _handle_remote_streaming(
            request=request,
            target_url="https://api.example.com/v1/chat/completions",
            headers={"Authorization": "Bearer test"},
            body=b'{"stream": true, "model": "test"}',
            body_json={"stream": True, "model": "test"},
            model_name="test-model",
            remote_timeout=httpx.Timeout(30.0),
            provider="test-provider",
            upstream_idle_timeout_seconds=idle_timeout,
        )
        return [chunk async for chunk in result.body_iterator]


def _mock_srv(config=None, logger=None):
    m = MagicMock()
    m.config = config if config is not None else {}
    m.logger = logger or MagicMock()
    return m


# ===================================================================
# AC5 — case A: stall after content with a trailing partial
# ===================================================================


@pytest.mark.asyncio
async def test_stall_after_content_drops_trailing_partial():
    """AC5: a complete event + the start of the next in one read, then a stall.

    The client must receive the complete event and the synthetic error as two
    separate, parseable events; the dangling partial must never be forwarded.
    """
    complete = (
        b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'
    )
    partial = b'data: {"choices":[{"delta":{"content":"' + PARTIAL_MARKER
    first = _make_mock_response([complete + partial], hang_after=True)
    client = _make_client([first])

    chunks = await _run_remote(client, _mock_srv(), _make_request())

    payloads = _assert_all_events_well_formed(chunks)
    assert PARTIAL_MARKER not in _joined(chunks), (
        "The dangling partial event was forwarded to the client"
    )
    assert any(b'"finish_reason": "error"' in p for p in payloads), (
        "Expected a synthetic finish_reason:error event"
    )
    assert any(b"stall_after_content" in p for p in payloads), (
        "Expected the synthetic error to be classified as stall_after_content"
    )
    # The synthetic error is its own event, not merged with the content event.
    error_events = [c for c in chunks if b'"finish_reason": "error"' in c]
    assert len(error_events) == 1
    assert error_events[0].count(b"data:") == 1


# ===================================================================
# AC6 — case B: empty-response retry whose first attempt ends mid-event
# ===================================================================


@pytest.mark.asyncio
async def test_empty_response_retry_does_not_concatenate_first_partial():
    """AC6: the retry stream's first event is not joined to the first attempt's bytes."""
    partial = b'data: {"choices":[{"delta":{"content":"' + PARTIAL_MARKER
    first = _make_mock_response([partial])
    retry_events = [
        b'data: {"choices":[{"delta":{"content":"Retry"},"index":0}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    second = _make_mock_response(retry_events)
    client = _make_client([first, second])

    config = {
        "server": {
            "upstream_empty_retry_max_attempts": 1,
            "upstream_empty_retry_base_delay_seconds": 0.0,
        }
    }
    chunks = await _run_remote(client, _mock_srv(config), _make_request())

    payloads = _assert_all_events_well_formed(chunks)
    assert PARTIAL_MARKER not in _joined(chunks), (
        "The first attempt's partial was concatenated with the retry stream"
    )
    assert client.stream.call_count == 2, "Expected an empty-response retry"
    # The first event the client sees is the retry's content event, whole.
    assert payloads[0] == (
        b'{"choices":[{"delta":{"content":"Retry"},"index":0}]}'
    )


# ===================================================================
# AC7 — clean stop with a trailing partial in the same read
# ===================================================================


@pytest.mark.asyncio
async def test_clean_stop_drops_trailing_partial():
    """AC7: a terminal event + trailing partial in one read; partial not forwarded."""
    terminal = (
        b'data: {"choices":[{"delta":{"content":"Done"},'
        b'"finish_reason":"stop","index":0}]}\n\n'
    )
    stop_payload = b'"finish_reason":"stop"'
    partial = b'data: {"choices":[{"delta":{"content":"' + PARTIAL_MARKER
    response = _make_mock_response([terminal + partial])
    client = _make_client([response])

    chunks = await _run_remote(client, _mock_srv(), _make_request())

    _assert_all_events_well_formed(chunks)
    assert PARTIAL_MARKER not in _joined(chunks), (
        "The incomplete event after a clean stop was forwarded"
    )
    assert any(stop_payload in c for c in chunks)


# ===================================================================
# AC4 — local provider path gets the same framing guarantee
# ===================================================================

BASE_SERVER_CONFIG = {
    "server": {
        "llama_router_mode": False,
        "llama_server_port": 8080,
        "max_concurrent_queries": 4,
        "session_slot_pool_size": 1,
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


def _dummy_request(body):
    payload = {**body, "stream": True}
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


def _make_local_cm(aiter_func):
    mock_resp = type("MockStreamResponse", (), {
        "status_code": 200,
        "headers": {"content-type": "text/event-stream"},
        "aiter_bytes": staticmethod(aiter_func),
        "aread": AsyncMock(return_value=b""),
    })

    class _LocalCM:
        async def __aenter__(self):
            return mock_resp()

        async def __aexit__(self, *args):
            pass

    return _LocalCM(), mock_resp()


@pytest.fixture
def _reset_router_state(monkeypatch):
    monkeypatch.setattr(server, "config", dict(BASE_SERVER_CONFIG))
    monkeypatch.setattr(server, "active_queries", 0)
    monkeypatch.setattr(server, "local_active_queries", 0)
    monkeypatch.setattr(server, "local_generating_queries", 0)
    monkeypatch.setattr(server, "local_generating_queries_lock", asyncio.Lock())
    monkeypatch.setattr(server, "local_generating_sessions", set())
    monkeypatch.setattr(server, "local_prefill_in_flight", {})
    monkeypatch.setattr(server, "local_prefill_in_flight_lock", asyncio.Lock())
    monkeypatch.setattr(server, "local_dispatch_records", {})
    monkeypatch.setattr(server, "local_dispatch_records_lock", asyncio.Lock())
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "llama_process", MagicMock(poll=lambda: None, pid=1))
    monkeypatch.setattr(server, "current_model", "test-model")
    monkeypatch.setattr(server, "session_manager", MagicMock())
    monkeypatch.setattr(server, "logger", MagicMock())
    monkeypatch.setattr(server, "backend_signal_counts", {
        "connect_failures": 0,
        "read_failures": 0,
        "timeout_failures": 0,
        "other_failures": 0,
        "concurrency_rejects": 0,
    })
    monkeypatch.setattr("proxy.router._is_self_healing_active", lambda: False)
    monkeypatch.setattr("proxy.router._restore_slot_snapshot", AsyncMock(return_value=False))
    monkeypatch.setattr("proxy.router._save_slot_snapshot", AsyncMock(return_value=False))
    monkeypatch.setattr("proxy.router._build_slot_context", MagicMock(return_value=(None, None, 3.0)))
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
    monkeypatch.setattr("proxy.session._resolve_log_path", MagicMock(return_value=MagicMock(
        exists=lambda: False,
        stat=lambda: MagicMock(st_size=0),
    )))
    monkeypatch.setattr("proxy.router._check_slot_availability", AsyncMock(return_value=None))


@pytest.mark.asyncio
async def test_local_stream_drops_trailing_partial_and_frames_final(_reset_router_state, monkeypatch):
    """AC4: the local path frames raw bytes and never forwards a partial event.

    The backend delivers a complete content event plus the start of the next in
    one read, then closes. The client must receive the complete event and the
    proxy-synthesised terminal events as separate, parseable events.
    """
    from proxy.router import proxy_to_local

    complete = (
        b'data: {"choices":[{"delta":{"content":"Local"},"index":0}]}\n\n'
    )
    partial = b'data: {"choices":[{"delta":{"content":"' + PARTIAL_MARKER

    async def _aiter():
        yield complete + partial

    cm, resp = _make_local_cm(_aiter)
    monkeypatch.setattr(
        "proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, resp))
    )

    response = await proxy_to_local(
        _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}),
        "v1/chat/completions",
    )
    chunks = [chunk async for chunk in response.body_iterator]

    _assert_all_events_well_formed(chunks)
    assert PARTIAL_MARKER not in _joined(chunks), (
        "The local path forwarded an incomplete event"
    )
    assert any(b'"finish_reason": "stop"' in c for c in chunks), (
        "Expected the local path to synthesise a terminal stop event"
    )
    assert any(c.strip() == b"data: [DONE]" for c in chunks), (
        "Expected the local path to emit [DONE]"
    )
