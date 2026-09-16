"""Contract tests for compaction metadata response headers (LP-0MTYGZ1DI0004QP8).

Frozen wire contract consumed by the Pi client extension (SA-0MTYGZIWF000ZLU0).
When the proxy applies live server-side compaction it must surface the decision
on the HTTP response so the client can mirror the compacted dispatch base on
its next turn:

``X-Compaction-Occurred: true``
``X-Compaction-Marker: <base64(<_SUMMARY_MARKER><summary><_SUMMARY_MARKER_END>)>``
``X-Compaction-Turns-Summarized: <decimal int>``
``X-Compaction-Recent-Turns-Kept: <decimal int>``

TDD red phase (child LP-0MU44KR5F0015D7A): the headers are emitted by the
implementation child LP-0MU44KSMS002DBWA. The positive-contract tests below are
``xfail(strict=True)`` so the suite stays green on ``dev`` while the assertions
genuinely fail against the current code. ``strict=True`` makes the red phase
self-clearing: as soon as the headers land the tests XPASS, turning them into
failures that force removal of the marker (AC7).

The negative-path tests (dry-run / noop / below-trigger / remote_with_guidance
/ fail-safe) are green in both phases — they assert that non-compaction
responses never carry the compaction header set (AC3/AC4).
"""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import proxy.provider as provider
import pytest
from fastapi import Response
from fastapi.responses import StreamingResponse
from proxy.compaction import _SUMMARY_MARKER, _SUMMARY_MARKER_END, _summary_message
from proxy.router import proxy_to_local

# ── Helpers ──────────────────────────────────────────────────────────────────


class AsyncIterator:
    """Turn a list of chunks into an async iterator (mirrors upstream SSE)."""

    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        return self._iterator()

    async def _iterator(self):
        for item in self.items:
            yield item


def _mock_upstream_response(
    status_code: int = 200,
    content: bytes = (
        b'{"id":"test","choices":[{"finish_reason":"stop","index":0,'
        b'"message":{"role":"assistant","content":"Hello!"}}]}'
    ),
    content_type: str = "application/json",
):
    """Build a synchronous mock upstream Response (plain object)."""
    return type(
        "MockResponse",
        (),
        {
            "status_code": status_code,
            "content": content,
            "headers": {"content-type": content_type},
        },
    )()


def _mock_streaming_upstream_response(
    status_code: int = 200,
    content_type: str = "text/event-stream",
):
    """Build a mock httpx streaming upstream response."""
    chunks = [
        b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    async def _aiter():
        for c in chunks:
            yield c

    mock_stream_response = type(
        "MockStreamResponse",
        (),
        {
            "status_code": status_code,
            "headers": {"content-type": content_type},
            "aiter_bytes": staticmethod(_aiter),
            "aread": AsyncMock(return_value=b"".join(chunks)),
        },
    )

    class MockCM:
        async def __aenter__(self):
            return mock_stream_response()

        async def __aexit__(self, *args):
            pass

    return MockCM(), mock_stream_response()


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


def _headers(response) -> dict[str, str]:
    """Lower-case header view of a Starlette/FastAPI response."""
    return {k.lower(): v for k, v in dict(response.headers).items()}


def _assert_no_compaction_headers(response) -> None:
    """AC3/AC4: non-compaction responses carry none of the four headers."""
    headers = _headers(response)
    for name in (
        "x-compaction-occurred",
        "x-compaction-marker",
        "x-compaction-turns-summarized",
        "x-compaction-recent-turns-kept",
    ):
        assert name not in headers, (
            f"{name} unexpectedly present on a non-compaction response: "
            f"{headers.get(name)!r}"
        )


# ── Session-result doubles ───────────────────────────────────────────────────


def _session_result(**overrides) -> dict:
    result = {
        "session_id": "test-session-id",
        "session_id_header": None,
        "session_created": False,
        "is_delta_request": False,
        "session_fallback_reason": None,
        "delta_messages": None,
        "body_json": None,
        "body_override": None,
        "original_message_count": 1,
        "session_explicit": False,
    }
    result.update(overrides)
    return result


def _compaction_result(
    summary_text: str = "Middle turns folded into this summary.",
    turns: int = 3,
    recent: int = 2,
) -> dict:
    """A session result as produced by _handle_session on live compaction."""
    return _session_result(
        compaction_applied=True,
        compaction_summary_text=summary_text,
        compaction_turns_summarized=turns,
        compaction_recent_turns_kept=recent,
        compaction_estimated_before=120000,
        compaction_reason="compacted_within_target",
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_cooldown():
    """Reset provider cooldown state between tests to avoid leakage."""
    provider._provider_unavailable_until.clear()
    provider._usage_reset_at.clear()
    yield


@pytest.fixture(autouse=True)
def _mock_server_state(monkeypatch):
    """Mock server-level state for proxy_to_local tests.

    Mirrors the fixture in test_resolved_model_header.py; the default
    ``_handle_session`` double returns a *non-compaction* session result so the
    negative-path tests need no per-test setup.
    """
    import proxy.server as server

    monkeypatch.setattr(
        server,
        "config",
        {
            "server": {
                "llama_router_mode": False,
                "llama_server_port": 8080,
                "max_concurrent_queries": 4,
                "session_slot_pool_size": 1,
                "llama_request_timeout": 30,
                "session_single_flight_mode": "bypass",
                "disconnect_cleanup_timeout": 1,
            }
        },
    )
    monkeypatch.setattr(server, "active_queries", 0)
    monkeypatch.setattr(server, "local_active_queries", 0)
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "llama_process", MagicMock(poll=lambda: None, pid=1))
    monkeypatch.setattr(server, "current_model", "Qwen3")
    monkeypatch.setattr(server, "session_manager", MagicMock())
    monkeypatch.setattr(server, "logger", MagicMock())

    monkeypatch.setattr("proxy.router._is_self_healing_active", lambda: False)
    monkeypatch.setattr("proxy.router._restore_slot_snapshot", AsyncMock(return_value=False))
    monkeypatch.setattr("proxy.router._save_slot_snapshot", AsyncMock(return_value=False))
    monkeypatch.setattr(
        "proxy.router._build_slot_context", MagicMock(return_value=(None, None, 3.0))
    )
    monkeypatch.setattr(
        "proxy.router._handle_session",
        AsyncMock(return_value=_session_result()),
    )
    monkeypatch.setattr(
        "proxy.session._resolve_log_path",
        MagicMock(
            return_value=MagicMock(exists=lambda: False, stat=lambda: MagicMock(st_size=0))
        ),
    )
    monkeypatch.setattr("proxy.router._check_slot_availability", AsyncMock(return_value=None))


# ═══════════════════════════════════════════════════════════════════════════════
# AC1/AC5: applied compaction emits the full header set (streaming + buffered)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.xfail(strict=True, reason="headers emitted by LP-0MU44KSMS002DBWA")
@pytest.mark.asyncio
async def test_streaming_applied_compaction_emits_header_set(monkeypatch):
    """Live compaction over SSE sets the four headers on the StreamingResponse."""
    summary = "Streaming compaction summary."
    monkeypatch.setattr(
        "proxy.router._handle_session",
        AsyncMock(return_value=_compaction_result(summary, turns=4, recent=1)),
    )
    req = _dummy_request(
        {"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}]},
        stream=True,
    )
    cm, upstream = _mock_streaming_upstream_response(status_code=200)
    monkeypatch.setattr(
        "proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, upstream))
    )
    monkeypatch.setattr("proxy.router._call_with_empty_retry", AsyncMock())

    result = await proxy_to_local(req, "v1/chat/completions")

    assert isinstance(result, StreamingResponse)
    headers = _headers(result)
    assert headers.get("x-compaction-occurred") == "true"
    assert headers.get("x-compaction-turns-summarized") == "4"
    assert headers.get("x-compaction-recent-turns-kept") == "1"
    assert "x-compaction-marker" in headers


@pytest.mark.xfail(strict=True, reason="headers emitted by LP-0MU44KSMS002DBWA")
@pytest.mark.asyncio
async def test_buffered_applied_compaction_emits_header_set(monkeypatch):
    """Live compaction over a buffered response sets the four headers."""
    summary = "Buffered compaction summary."
    monkeypatch.setattr(
        "proxy.router._handle_session",
        AsyncMock(return_value=_compaction_result(summary, turns=7, recent=5)),
    )
    req = _dummy_request(
        {"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}]},
        stream=False,
    )
    upstream = _mock_upstream_response(status_code=200, content_type="application/json")
    monkeypatch.setattr(
        "proxy.router._call_with_backend_retries", AsyncMock(return_value=upstream)
    )
    monkeypatch.setattr(
        "proxy.router._call_with_empty_retry", AsyncMock(return_value=upstream)
    )
    monkeypatch.setattr("proxy.router._schedule_recv_token_increment", AsyncMock())

    result = await proxy_to_local(req, "v1/chat/completions")

    assert isinstance(result, Response)
    headers = _headers(result)
    assert headers.get("x-compaction-occurred") == "true"
    assert headers.get("x-compaction-turns-summarized") == "7"
    assert headers.get("x-compaction-recent-turns-kept") == "5"
    assert "x-compaction-marker" in headers


# ═══════════════════════════════════════════════════════════════════════════════
# AC2: the marker header round-trips to the exact injected summary message
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.xfail(strict=True, reason="headers emitted by LP-0MU44KSMS002DBWA")
@pytest.mark.asyncio
async def test_marker_decodes_to_exact_injected_summary_message(monkeypatch):
    """X-Compaction-Marker base64-decodes to the verbatim injected message."""
    summary = "Exact summary text with <xml-looking> content & symbols."
    monkeypatch.setattr(
        "proxy.router._handle_session",
        AsyncMock(return_value=_compaction_result(summary)),
    )
    req = _dummy_request(
        {"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}]},
        stream=True,
    )
    cm, upstream = _mock_streaming_upstream_response(status_code=200)
    monkeypatch.setattr(
        "proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, upstream))
    )
    monkeypatch.setattr("proxy.router._call_with_empty_retry", AsyncMock())

    result = await proxy_to_local(req, "v1/chat/completions")

    headers = _headers(result)
    assert "x-compaction-marker" in headers
    decoded = base64.b64decode(headers["x-compaction-marker"]).decode("utf-8")
    # The injected message is the delimiters wrapped around the summary text.
    assert decoded == _summary_message(summary)["content"]
    assert decoded.startswith(_SUMMARY_MARKER)
    assert decoded.endswith(_SUMMARY_MARKER_END)


@pytest.mark.xfail(strict=True, reason="headers emitted by LP-0MU44KSMS002DBWA")
@pytest.mark.asyncio
async def test_turn_count_headers_are_decimal_integers(monkeypatch):
    """Both turn-count headers carry canonical decimal integers."""
    monkeypatch.setattr(
        "proxy.router._handle_session",
        AsyncMock(return_value=_compaction_result(turns=12, recent=0)),
    )
    req = _dummy_request(
        {"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}]},
        stream=False,
    )
    upstream = _mock_upstream_response(status_code=200, content_type="application/json")
    monkeypatch.setattr(
        "proxy.router._call_with_backend_retries", AsyncMock(return_value=upstream)
    )
    monkeypatch.setattr(
        "proxy.router._call_with_empty_retry", AsyncMock(return_value=upstream)
    )
    monkeypatch.setattr("proxy.router._schedule_recv_token_increment", AsyncMock())

    result = await proxy_to_local(req, "v1/chat/completions")

    headers = _headers(result)
    summarized = headers.get("x-compaction-turns-summarized", "")
    kept = headers.get("x-compaction-recent-turns-kept", "")
    assert summarized.isdigit() and str(int(summarized)) == summarized
    assert kept.isdigit() and str(int(kept)) == kept
    assert summarized == "12"
    assert kept == "0"


# ═══════════════════════════════════════════════════════════════════════════════
# AC3: non-compaction paths never emit the compaction header set
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("scenario", ["noop", "below_trigger", "dry_run"])
@pytest.mark.asyncio
async def test_non_compaction_streaming_sets_no_headers(monkeypatch, scenario):
    """noop / below-trigger / dry-run streaming responses omit every header."""
    req = _dummy_request(
        {"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}]},
        stream=True,
    )
    cm, upstream = _mock_streaming_upstream_response(status_code=200)
    monkeypatch.setattr(
        "proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, upstream))
    )
    monkeypatch.setattr("proxy.router._call_with_empty_retry", AsyncMock())

    result = await proxy_to_local(req, "v1/chat/completions")

    assert isinstance(result, StreamingResponse)
    _assert_no_compaction_headers(result)


@pytest.mark.parametrize("scenario", ["noop", "below_trigger", "dry_run"])
@pytest.mark.asyncio
async def test_non_compaction_buffered_sets_no_headers(monkeypatch, scenario):
    """noop / below-trigger / dry-run buffered responses omit every header."""
    req = _dummy_request(
        {"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}]},
        stream=False,
    )
    upstream = _mock_upstream_response(status_code=200, content_type="application/json")
    monkeypatch.setattr(
        "proxy.router._call_with_backend_retries", AsyncMock(return_value=upstream)
    )
    monkeypatch.setattr(
        "proxy.router._call_with_empty_retry", AsyncMock(return_value=upstream)
    )
    monkeypatch.setattr("proxy.router._schedule_recv_token_increment", AsyncMock())

    result = await proxy_to_local(req, "v1/chat/completions")

    assert isinstance(result, Response)
    _assert_no_compaction_headers(result)


@pytest.mark.asyncio
async def test_remote_with_guidance_sets_no_headers(monkeypatch):
    """remote_with_guidance escalates remote; it never emits compaction headers."""
    monkeypatch.setattr(
        "proxy.router._handle_session",
        AsyncMock(
            return_value=_session_result(
                compaction_remote_with_guidance=True,
                compaction_estimated_before=120000,
                compaction_reason="summarizer_unavailable",
            )
        ),
    )
    routed = Response(status_code=200, content=b'{"ok":true}')
    route_mock = AsyncMock(return_value=routed)
    monkeypatch.setattr("proxy.router._route_remote_with_compaction_guidance", route_mock)

    req = _dummy_request(
        {"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}]},
        stream=True,
    )

    result = await proxy_to_local(req, "v1/chat/completions")

    assert result is routed
    route_mock.assert_awaited_once()
    _assert_no_compaction_headers(result)


# ═══════════════════════════════════════════════════════════════════════════════
# AC4: fail-safe — missing metadata never yields a malformed header set
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_applied_without_summary_text_sets_no_headers(monkeypatch):
    """compaction_applied with no summary text is a no-op for headers (AC4)."""
    monkeypatch.setattr(
        "proxy.router._handle_session",
        AsyncMock(
            return_value=_session_result(
                compaction_applied=True,
                compaction_summary_text=None,
                compaction_turns_summarized=3,
                compaction_recent_turns_kept=2,
            )
        ),
    )
    req = _dummy_request(
        {"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}]},
        stream=True,
    )
    cm, upstream = _mock_streaming_upstream_response(status_code=200)
    monkeypatch.setattr(
        "proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, upstream))
    )
    monkeypatch.setattr("proxy.router._call_with_empty_retry", AsyncMock())

    result = await proxy_to_local(req, "v1/chat/completions")

    assert isinstance(result, StreamingResponse)
    _assert_no_compaction_headers(result)


@pytest.mark.asyncio
async def test_header_emission_failure_does_not_break_dispatch(monkeypatch):
    """AC4: an error while building headers never changes dispatch behaviour.

    The response body/status must be delivered unchanged even when the header
    block cannot be assembled (here: a malformed metadata value that raises on
    encode).
    """
    monkeypatch.setattr(
        "proxy.router._handle_session",
        AsyncMock(
            return_value=_session_result(
                compaction_applied=True,
                # Non-str summary text cannot be encoded by the marker builder.
                compaction_summary_text=object(),
                compaction_turns_summarized=3,
                compaction_recent_turns_kept=2,
            )
        ),
    )
    req = _dummy_request(
        {"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}]},
        stream=False,
    )
    upstream = _mock_upstream_response(status_code=200, content_type="application/json")
    monkeypatch.setattr(
        "proxy.router._call_with_backend_retries", AsyncMock(return_value=upstream)
    )
    monkeypatch.setattr(
        "proxy.router._call_with_empty_retry", AsyncMock(return_value=upstream)
    )
    monkeypatch.setattr("proxy.router._schedule_recv_token_increment", AsyncMock())

    result = await proxy_to_local(req, "v1/chat/completions")

    assert isinstance(result, Response)
    assert result.status_code == 200
    assert result.body == upstream.content
    _assert_no_compaction_headers(result)
