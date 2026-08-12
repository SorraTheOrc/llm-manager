"""
Tests for the mid-stream re-route decision matrix (LP-0MSF1PUM90099ZSW).

Covers the re-route boundary defined by the plan (F1 contract for F2/F3):

- A reasoning-only stall (reasoning_content chunks delivered, zero tool_calls,
  zero final-answer content, then a terminal error) re-routes the SAME request
  to the next provider in the chain, so the client receives a usable
  completion without being told to retry (AC1).
- A tool-call-only stall (tool_calls deltas delivered, no tool-result
  round-trip, then a terminal error) terminates with the enriched error and
  does NOT re-route (AC2).
- A content-committed stall (final-answer content already forwarded) terminates
  with the enriched error and does NOT re-route (AC3).
- All-providers-exhausted surfaces the enriched error (provider/model/reason),
  never a bare finish_reason or a silent abort (AC4).
- Bounded re-route: never loops back to an already-failed provider in the same
  request; per-provider cooldown (Tier-2) applies (AC5).
- Healthy streaming passthrough has no regression (AC6).

Implementation detail: the pre-flight (provider.py) buffers intermediate
chunks (reasoning/tool_calls) until the commit point (first final-answer
content chunk) or a terminal event. A terminal event with only reasoning
delivered raises `StreamingRecoverableAfterReasoningError`, which both fallback
chains catch and treat as a re-route (same request, next provider).
"""

import json
from unittest.mock import AsyncMock, patch

import proxy.provider as provider
import pytest
from fastapi import Response
from fastapi.responses import StreamingResponse


class _DummyRequest:
    """Minimal request stub (mirrors test_stream_error_fallback._DummyRequest)."""

    def __init__(self, body: bytes = b'{"model":"test"}'):
        self._body = body
        self.headers = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body

    async def is_disconnected(self):
        return False


@pytest.fixture(autouse=True)
def reset_cooldown_state():
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    yield


@pytest.fixture
def two_provider_config():
    return {
        "providers": [
            {
                "name": "remote-primary",
                "type": "remote",
                "endpoint": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
            {
                "name": "remote-fallback",
                "type": "remote",
                "endpoint": "https://api.anthropic.com/v1",
                "api_key_env": "ANTHROPIC_API_KEY",
            },
        ],
        "aliases": ["test*"],
    }


def _sse_data(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _reasoning_chunk(text: str) -> bytes:
    return _sse_data({"choices": [{"delta": {"reasoning_content": text}, "index": 0}]})


def _tool_calls_chunk() -> bytes:
    return _sse_data({
        "choices": [
            {"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                       "function": {"name": "get_weather", "arguments": '{"city":"SF"}'}}]},
             "index": 0}
        ]
    })


def _content_chunk(text: str) -> bytes:
    return _sse_data({"choices": [{"delta": {"content": text}, "index": 0}]})


def _error_event(error_type: str = "stall_after_content") -> bytes:
    return _sse_data({
        "choices": [
            {"delta": {}, "finish_reason": "error", "index": 0,
             "error": {
                 "type": error_type,
                 "message": "Upstream idle timeout",
                 "provider": "remote-primary",
                 "model": "test-model",
                 "entry": "remote-primary-1",
                 "suggested_action": "Retry the request with full context, or route to a healthier provider",
             }},
        ]
    })


def _done_marker() -> bytes:
    return b"data: [DONE]\n\n"


def _reasoning_only_stall_stream():
    """Reasoning chunks, then a terminal error. Zero tool_calls, zero content."""
    return [
        _reasoning_chunk("Let me think"),
        _reasoning_chunk(" about this problem"),
        _error_event(),
    ]


def _tool_calls_only_stall_stream():
    """Tool-call delta(s), then a terminal error. Zero final content."""
    return [
        _tool_calls_chunk(),
        _error_event(),
    ]


def _content_then_error_stream():
    """Final-answer content, then a terminal error (after-content stall)."""
    return [
        _content_chunk("Partial answer"),
        _error_event(),
    ]


def _healthy_stream():
    return [
        _reasoning_chunk("Thinking"),
        _content_chunk("Hello"),
        _content_chunk(" world"),
        _sse_data({"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}),
        _done_marker(),
    ]


def _make_streaming_response(chunks, status=200):
    async def _body():
        for c in chunks:
            yield c

    return StreamingResponse(_body(), status_code=status, media_type="text/event-stream")


def _ok_json_response():
    return Response(
        content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
        status_code=200,
        media_type="application/json",
    )


async def _collect(result: Response) -> str:
    """Collect a StreamingResponse body into a decoded string."""
    body = b"".join([c async for c in result.body_iterator])
    return body.decode("utf-8", errors="replace")


# ===================================================================
# Chunk classification (F2 contract: finer-grained categories)
# ===================================================================


def test_classify_distinguishes_final_content_tool_calls_reasoning():
    """The pre-flight classification must tell final content apart from
    intermediate tool_calls/reasoning (LP-0MS8XAPXT009W3CL refinement)."""
    classify = provider._classify_stream_chunk

    # Final-answer content only
    h_fc, h_tc, h_rc, term, done = classify(_content_chunk("Hi"))
    assert h_fc and not h_tc and not h_rc and not term and not done

    # Tool calls only
    h_fc, h_tc, h_rc, term, done = classify(_tool_calls_chunk())
    assert h_tc and not h_fc and not h_rc and not term and not done

    # Reasoning only
    h_fc, h_tc, h_rc, term, done = classify(_reasoning_chunk("hm"))
    assert h_rc and not h_fc and not h_tc and not term and not done

    # Terminal error
    h_fc, h_tc, h_rc, term, done = classify(_error_event())
    assert term and not h_fc and not h_tc and not h_rc

    # Done marker
    h_fc, h_tc, h_rc, term, done = classify(_done_marker())
    assert done and not term and not h_fc and not h_tc and not h_rc


def test_classify_ignores_keepalive_comments():
    """Keep-alive comment lines are ignored (no classification fires)."""
    h_fc, h_tc, h_rc, term, done = provider._classify_stream_chunk(b": keep-alive\n\n")
    assert not (h_fc or h_tc or h_rc or term or done)


# ===================================================================
# AC1: reasoning-only stall re-routes to the next provider
# ===================================================================


@pytest.mark.asyncio
async def test_reasoning_only_stall_reroutes_to_next_provider(two_provider_config):
    """A stream that delivered reasoning_content then a terminal error
    (zero tool_calls, zero final content) re-routes the SAME request to the
    next provider; the client receives the fallback's completion."""
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_streaming_response(_reasoning_only_stall_stream())
        return _ok_json_response()

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", two_provider_config,
            {"provider_cooldown_seconds": 60},
        )

    assert call_count == 2, (
        f"Reasoning-only stall must re-route to next provider, got {call_count} calls"
    )
    assert result.status_code == 200
    body = json.loads(result.body.decode()) if isinstance(result.body, bytes) else json.loads(result.body)
    assert body["choices"][0]["message"]["content"] == "ok"
    # Failed provider must be in cooldown (Tier-2)
    assert provider._is_provider_unavailable("remote-primary")


@pytest.mark.asyncio
async def test_reasoning_only_stall_reroutes_in_proxy_with_fallback(two_provider_config):
    """Same re-route behavior through the combined local+remote chain."""
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_streaming_response(_reasoning_only_stall_stream())
        return _ok_json_response()

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", two_provider_config,
            {"provider_cooldown_seconds": 60},
        )

    assert call_count == 2
    assert result.status_code == 200
    body = json.loads(result.body.decode()) if isinstance(result.body, bytes) else json.loads(result.body)
    assert body["choices"][0]["message"]["content"] == "ok"
    assert provider._is_provider_unavailable("remote-primary")


@pytest.mark.asyncio
async def test_reroute_emits_sse_comment(two_provider_config):
    """On a mid-stream re-route, an SSE comment marks the switch so operators
    / pi can see it: ``: re-route provider=a->b reason=stall_after_reasoning``."""
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_streaming_response(_reasoning_only_stall_stream())
        return _make_streaming_response(_healthy_stream())

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", two_provider_config,
            {"provider_cooldown_seconds": 60},
        )

    assert call_count == 2
    collected = await _collect(result)
    assert ": re-route provider=remote-primary->remote-fallback" in collected, (
        f"Expected SSE re-route comment, got: {collected!r}"
    )
    assert "stall_after_reasoning" in collected
    # The fallback's content must be present (usable completion).
    assert "Hello" in collected


# ===================================================================
# AC2: tool-call-only stall terminates, no re-route
# ===================================================================


@pytest.mark.asyncio
async def test_tool_calls_only_stall_terminates_no_reroute(two_provider_config):
    """Tool-call deltas delivered then a stall: terminate with the enriched
    error; never re-route (no tool-result round-trip means re-routing would
    re-plan the request)."""
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        return _make_streaming_response(_tool_calls_only_stall_stream())

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", two_provider_config,
            {"provider_cooldown_seconds": 60},
        )

    assert call_count == 1, (
        f"Tool-call-only stall must NOT re-route, got {call_count} calls"
    )
    collected = await _collect(result)
    assert '"tool_calls"' in collected, "Tool-call deltas must reach the client"
    assert '"finish_reason": "error"' in collected, "Terminal error must reach the client"


# ===================================================================
# AC3: content-committed stall terminates, no re-route
# ===================================================================


@pytest.mark.asyncio
async def test_content_committed_stall_terminates_no_reroute(two_provider_config):
    """Once final-answer content is forwarded, a stall terminates with the
    enriched error as today (LP-0MS9FR9LG002AJ4C preserved)."""
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        return _make_streaming_response(_content_then_error_stream())

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", two_provider_config,
            {"provider_cooldown_seconds": 60},
        )

    assert call_count == 1, (
        f"Content-committed stall must NOT re-route, got {call_count} calls"
    )
    collected = await _collect(result)
    assert "Partial answer" in collected, "Forwarded content must reach the client"
    assert '"finish_reason": "error"' in collected


# ===================================================================
# AC4: all providers exhausted surfaces the enriched error
# ===================================================================


@pytest.mark.asyncio
async def test_all_providers_exhausted_surfaces_enriched_error(two_provider_config):
    """Both providers stall after reasoning; the client receives an enriched
    finish_reason:error (provider/model/reason present), never a bare abort."""
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        return _make_streaming_response(_reasoning_only_stall_stream())

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", two_provider_config,
            {"provider_cooldown_seconds": 60},
        )

    assert call_count == 2, "Both providers must be attempted"
    collected = await _collect(result)
    assert '"finish_reason": "error"' in collected
    assert '"provider"' in collected, "Enriched error payload must include provider"
    assert '"model"' in collected
    # First provider is in cooldown; the last (no next provider) passes through.
    assert provider._is_provider_unavailable("remote-primary")


# ===================================================================
# AC5: bounded re-route / cooldown / no loop-back
# ===================================================================


@pytest.mark.asyncio
async def test_bounded_reroute_respects_remaining_providers():
    """With three providers where the first two stall after reasoning, the
    chain tries each once (no loop-back) and the third succeeds."""
    config = {
        "providers": [
            {"name": "p1", "type": "remote", "endpoint": "https://a/v1", "api_key_env": "A"},
            {"name": "p2", "type": "remote", "endpoint": "https://b/v1", "api_key_env": "B"},
            {"name": "p3", "type": "remote", "endpoint": "https://c/v1", "api_key_env": "C"},
        ],
        "aliases": ["test*"],
    }
    call_order: list[str] = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        name = provider_cfg["name"]
        call_order.append(name)
        if name in ("p1", "p2"):
            return _make_streaming_response(_reasoning_only_stall_stream())
        return _ok_json_response()

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", config,
            {"provider_cooldown_seconds": 60},
        )

    assert call_order == ["p1", "p2", "p3"], (
        f"Expected each provider tried once in order, got {call_order}"
    )
    assert result.status_code == 200
    body = json.loads(result.body.decode()) if isinstance(result.body, bytes) else json.loads(result.body)
    assert body["choices"][0]["message"]["content"] == "ok"
    assert provider._is_provider_unavailable("p1")
    assert provider._is_provider_unavailable("p2")
    assert not provider._is_provider_unavailable("p3")


@pytest.mark.asyncio
async def test_cooldown_skips_recently_failed_provider(two_provider_config):
    """A provider already in Tier-2 cooldown is skipped: the request routes
    directly to the healthy provider (no re-try of the cooldown provider)."""
    provider.mark_provider_unavailable("remote-primary", 3600)  # 1h cooldown
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        return _ok_json_response()

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", two_provider_config,
            {"provider_cooldown_seconds": 60},
        )

    assert call_count == 1, (
        f"Cooldown provider must be skipped, got {call_count} calls"
    )
    body = json.loads(result.body.decode()) if isinstance(result.body, bytes) else json.loads(result.body)
    assert body["choices"][0]["message"]["content"] == "ok"


# ===================================================================
# AC6: healthy streaming passthrough regression
# ===================================================================


@pytest.mark.asyncio
async def test_healthy_streaming_passthrough_unchanged(two_provider_config):
    """A healthy stream (reasoning then content then stop) passes through
    unchanged: one provider call, X-Provider header, all content delivered."""
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        return _make_streaming_response(_healthy_stream())

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", two_provider_config,
            {"provider_cooldown_seconds": 60},
        )

    assert call_count == 1
    assert isinstance(result, StreamingResponse)
    assert result.headers.get("X-Provider") == "remote-primary"
    collected = await _collect(result)
    assert "Hello" in collected and " world" in collected
    assert '"finish_reason": "stop"' in collected
    assert not provider._is_provider_unavailable("remote-primary")


# ===================================================================
# Pre-flight buffering contract (F2)
# ===================================================================


@pytest.mark.asyncio
async def test_preflight_buffers_intermediate_chunks_then_commits():
    """Before the commit point (first final-answer content chunk), the
    pre-flight buffers intermediate chunks; at commit it replays them in
    order followed by the live stream (no reordering, no duplication)."""
    chunks = [
        _reasoning_chunk("r1"),
        _tool_calls_chunk(),
        _reasoning_chunk("r2"),
        _content_chunk("answer"),
        _content_chunk(" continued"),
        _sse_data({"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}),
    ]
    response = _make_streaming_response(chunks)

    wrapped = await provider._preflight_streaming_response(
        response, _DummyRequest(), "remote-primary"
    )

    collected = await _collect(wrapped)
    # All chunks present in order.
    assert "r1" in collected
    assert '"tool_calls"' in collected
    assert "r2" in collected
    assert "answer" in collected
    assert " continued" in collected
    assert '"finish_reason": "stop"' in collected
    # Order preserved: reasoning before content.
    assert collected.index("r1") < collected.index("answer")
    assert collected.index("r2") < collected.index("answer")


@pytest.mark.asyncio
async def test_preflight_raises_recoverable_after_reasoning_error():
    """A reasoning-only terminal event (zero tool_calls, zero content) raises
    StreamingRecoverableAfterReasoningError with provider + reason."""
    response = _make_streaming_response(_reasoning_only_stall_stream())

    with pytest.raises(provider.StreamingRecoverableAfterReasoningError) as excinfo:
        await provider._preflight_streaming_response(
            response, _DummyRequest(), "remote-primary"
        )

    assert excinfo.value.provider_name == "remote-primary"
    assert "stall_after_reasoning" in excinfo.value.reason


@pytest.mark.asyncio
async def test_preflight_tool_calls_only_stall_does_not_raise():
    """A tool-call-only terminal event does NOT raise the re-route exception;
    the pre-flight hands the stream back (terminate path, no re-route)."""
    response = _make_streaming_response(_tool_calls_only_stall_stream())

    wrapped = await provider._preflight_streaming_response(
        response, _DummyRequest(), "remote-primary"
    )

    collected = await _collect(wrapped)
    assert '"tool_calls"' in collected
    assert '"finish_reason": "error"' in collected


@pytest.mark.asyncio
async def test_preflight_content_committed_does_not_raise():
    """Once final content is seen, the pre-flight commits (no exception)."""
    response = _make_streaming_response(_content_then_error_stream())

    wrapped = await provider._preflight_streaming_response(
        response, _DummyRequest(), "remote-primary"
    )

    collected = await _collect(wrapped)
    assert "Partial answer" in collected
    assert '"finish_reason": "error"' in collected
