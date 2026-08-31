"""
Tests for OpenAI Responses API support in the remote proxy path (LP-0MTGK5DQO001Y8H0).

Covers the three translation layers:
1. Request translation: chat/completions body → responses API body
   (``_translate_chat_to_responses``).
2. Streaming response translation: responses SSE events → chat/completions SSE
   (``_translate_responses_stream``).
3. Non-streaming response translation: responses JSON → chat/completions JSON
   (``_translate_responses_to_chat``).

The muse models on opencode.ai (muse-spark-1.2-contributor(-free)) only
respond via the Responses API (/v1/responses); chat/completions returns
HTTP 500. These tests pin the translation behaviour so the proxy can serve
them through its existing chat/completions-facing API.
"""

import json

import pytest
from proxy.proxy_remote import (
    _translate_chat_to_responses,
    _translate_responses_stream,
    _translate_responses_to_chat,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Request translation: chat/completions body → responses API body
# ═══════════════════════════════════════════════════════════════════════════════

def test_request_translation_messages_to_input():
    """messages → input; basic roles pass through."""
    body = {
        "model": "plan",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
        "stream": True,
    }
    out = _translate_chat_to_responses(body)
    assert out["input"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    assert out["model"] == "plan"
    assert out["stream"] is True


def test_request_translation_max_tokens_mapping():
    """max_tokens/max_completion_tokens → max_output_tokens."""
    body = {"model": "m", "messages": [], "max_tokens": 4096}
    out = _translate_chat_to_responses(body)
    assert out["max_output_tokens"] == 4096
    assert "max_tokens" not in out

    body2 = {"model": "m", "messages": [], "max_completion_tokens": 2048}
    out2 = _translate_chat_to_responses(body2)
    assert out2["max_output_tokens"] == 2048
    assert "max_completion_tokens" not in out2


def test_request_translation_tool_messages():
    """tool-role messages → function_call_output items; assistant tool_calls → function_call items."""
    body = {
        "model": "m",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": '{"a":1,"b":2}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "3"},
        ],
    }
    out = _translate_chat_to_responses(body)
    # assistant tool_calls become a function_call item
    assert out["input"][0] == {
        "type": "function_call",
        "call_id": "call_abc",
        "name": "calculator",
        "arguments": '{"a":1,"b":2}',
    }
    # tool result becomes function_call_output
    assert out["input"][1] == {
        "type": "function_call_output",
        "call_id": "call_abc",
        "output": "3",
    }


def test_request_translation_strips_unsupported_fields():
    """Chat-only fields (store, stream_options, n, user, logprobs) are stripped."""
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "store": True,
        "stream_options": {"include_usage": True},
        "n": 2,
        "user": "u1",
        "logprobs": True,
        "temperature": 0.5,
    }
    out = _translate_chat_to_responses(body)
    for key in ("store", "stream_options", "n", "user", "logprobs"):
        assert key not in out, f"{key} should be stripped"
    assert out["temperature"] == 0.5
    assert out["input"] == [{"role": "user", "content": "hi"}]


def test_request_translation_reasoning_effort():
    """reasoning_effort → reasoning.effort (responses API shape)."""
    body = {"model": "m", "messages": [], "reasoning_effort": "high"}
    out = _translate_chat_to_responses(body)
    assert out.get("reasoning") == {"effort": "high"}
    assert "reasoning_effort" not in out


def test_request_translation_tools_and_choice_pass_through():
    """tools are normalized to Responses shape; tool_choice retained."""
    tools = [{"type": "function", "name": "f", "description": "d",
              "parameters": {"type": "object", "properties": {}}}]
    body = {"model": "m", "messages": [], "tools": tools, "tool_choice": "auto"}
    out = _translate_chat_to_responses(body)
    assert out["tools"] == tools
    assert out["tool_choice"] == "auto"


def test_request_translation_tools_function_nested_unwrapped():
    """chat tool (function nested) -> Responses tool (name at top level).

    Regression for upstream 400: ``tools[0] missing required field name`` —
    the Responses API requires ``name`` at the tool top level, while
    chat/completions nests it under ``function``.
    """
    tools = [{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Adds two numbers",
            "parameters": {"type": "object", "properties": {"a": {"type": "number"}}},
            "strict": True,
        },
    }]
    body = {"model": "m", "messages": [], "tools": tools}
    out = _translate_chat_to_responses(body)
    assert out["tools"] == [{
        "type": "function",
        "name": "calculator",
        "description": "Adds two numbers",
        "parameters": {"type": "object", "properties": {"a": {"type": "number"}}},
        "strict": True,
    }]
    # no nested function key remains
    assert "function" not in out["tools"][0]


def test_request_translation_content_parts_normalized():
    """chat content parts (text/image_url) -> Responses input parts (input_text/input_image).

    Regression for upstream 400: ``input[N].content did not match any supported
    type`` — the Responses API rejects chat-format part types.
    """
    body = {
        "model": "m",
        "messages": [
            {"role": "system", "content": "plain string ok"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "high"}},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ],
    }
    out = _translate_chat_to_responses(body)
    # string content passes through
    assert out["input"][0]["content"] == "plain string ok"
    # text part -> input_text
    assert out["input"][1]["content"][0] == {"type": "input_text", "text": "hello"}
    # image_url part -> input_image with url + detail
    img = out["input"][1]["content"][1]
    assert img["type"] == "input_image"
    assert img["image_url"] == "data:image/png;base64,AAAA"
    assert img["detail"] == "high"
    # assistant content parts normalized too
    assert out["input"][2]["content"][0] == {"type": "input_text", "text": "hi"}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Non-streaming response translation: responses JSON → chat/completions JSON
# ═══════════════════════════════════════════════════════════════════════════════

def test_non_streaming_response_translation_text():
    """output[].message content → choices[0].message.content; usage mapped."""
    resp = {
        "id": "resp_123",
        "object": "response",
        "status": "completed",
        "created_at": 1788154809,
        "model": "muse-spark-1.2-contributor-free",
        "output": [
            {"type": "reasoning", "status": "completed", "summary": []},
            {
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "apple", "annotations": []}],
            },
        ],
        "usage": {"input_tokens": 13, "output_tokens": 190, "total_tokens": 203},
    }
    out = _translate_responses_to_chat(resp)
    assert out["object"] == "chat.completion"
    assert out["id"] == "resp_123"
    assert out["model"] == "muse-spark-1.2-contributor-free"
    assert out["created"] == 1788154809
    assert out["choices"][0]["message"]["role"] == "assistant"
    assert out["choices"][0]["message"]["content"] == "apple"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"] == {"prompt_tokens": 13, "completion_tokens": 190, "total_tokens": 203}


def test_non_streaming_response_translation_function_call():
    """output[].function_call → choices[0].message.tool_calls; finish_reason tool_calls."""
    resp = {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "model": "m",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "calculator",
                "arguments": '{"a":2.0,"b":3.0}',
                "status": "completed",
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    }
    out = _translate_responses_to_chat(resp)
    tc = out["choices"][0]["message"]["tool_calls"]
    assert tc == [
        {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"a":2.0,"b":3.0}'},
        }
    ]
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    assert out["choices"][0]["message"]["content"] is None


def test_non_streaming_response_translation_incomplete():
    """status incomplete (max_output_tokens) → finish_reason length."""
    resp = {
        "id": "resp_2",
        "object": "response",
        "status": "incomplete",
        "model": "m",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "partial"}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    }
    out = _translate_responses_to_chat(resp)
    assert out["choices"][0]["message"]["content"] == "partial"
    assert out["choices"][0]["finish_reason"] == "length"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Streaming response translation: responses SSE → chat/completions SSE
# ═══════════════════════════════════════════════════════════════════════════════

class _AsyncBytesIter:
    """Async iterator over byte chunks (simulates upstream aiter_bytes)."""

    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)


def _sse_event(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _parse_json_chunks(chunks):
    """Parse translated SSE chunks; skip non-JSON data lines like `data: [DONE]`."""
    parsed = []
    for c in chunks:
        text = c.decode()
        for line in text.splitlines():
            if line.startswith("data: ") and line[6:].strip() not in ("[DONE]", ""):
                try:
                    parsed.append(json.loads(line[6:].strip()))
                except ValueError:
                    continue
    return parsed


async def _collect_translated(chunks):
    out = []
    async for chunk in _translate_responses_stream(_AsyncBytesIter(chunks)):
        out.append(chunk)
    return out


@pytest.mark.asyncio
async def test_streaming_translation_content_deltas():
    """response.output_text.delta events → chat/completions content deltas."""
    events = [
        _sse_event("response.created", {"type": "response.created"}),
        _sse_event("response.output_item.added", {
            "type": "response.output_item.added", "output_index": 0,
            "item": {"type": "reasoning", "status": "in_progress"},
        }),
        _sse_event("response.output_text.delta", {
            "type": "response.output_text.delta", "output_index": 1,
            "content_index": 0, "delta": "Hello",
        }),
        _sse_event("response.output_text.delta", {
            "type": "response.output_text.delta", "output_index": 1,
            "content_index": 0, "delta": " world",
        }),
        _sse_event("response.completed", {
            "type": "response.completed",
            "response": {"id": "resp_1", "status": "completed", "model": "m",
                         "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}},
        }),
        _sse_event("ping", {"type": "ping", "cost": "0"}),
    ]
    chunks = await _collect_translated(events)

    parsed = _parse_json_chunks(chunks)
    # content deltas only
    contents = [
        p["choices"][0]["delta"].get("content")
        for p in parsed
        if p["choices"][0]["delta"].get("content") is not None
    ]
    assert contents == ["Hello", " world"]

    # finish chunk with finish_reason stop
    finish = [p for p in parsed if p["choices"][0].get("finish_reason") == "stop"]
    assert len(finish) == 1, "exactly one finish chunk expected"

    # [DONE] terminator present
    assert any(c.decode().strip() == "data: [DONE]" for c in chunks)


@pytest.mark.asyncio
async def test_streaming_translation_tool_calls():
    """function_call output_item + arguments.delta → tool_calls delta chunks."""
    events = [
        _sse_event("response.output_item.added", {
            "type": "response.output_item.added", "output_index": 2,
            "item": {"type": "function_call", "name": "calculator",
                     "call_id": "call_abc", "arguments": ""},
        }),
        _sse_event("response.function_call_arguments.delta", {
            "type": "response.function_call_arguments.delta", "output_index": 2,
            "item_id": "fc_1", "delta": '{"a":2.0',
        }),
        _sse_event("response.function_call_arguments.delta", {
            "type": "response.function_call_arguments.delta", "output_index": 2,
            "item_id": "fc_1", "delta": ',"b":3.0}',
        }),
        _sse_event("response.completed", {
            "type": "response.completed",
            "response": {"id": "resp_2", "status": "completed", "model": "m",
                         "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}},
        }),
    ]
    chunks = await _collect_translated(events)

    parsed = _parse_json_chunks(chunks)
    tool_chunks = [
        p["choices"][0]["delta"]["tool_calls"]
        for p in parsed
        if "tool_calls" in p["choices"][0].get("delta", {})
    ]
    assert len(tool_chunks) == 3, "one tool_calls chunk per added/delta event"
    # first chunk carries id + name
    first = tool_chunks[0][0]
    assert first["id"] == "call_abc"
    assert first["function"]["name"] == "calculator"
    # arguments accumulate across delta chunks
    args = "".join(tc[0]["function"]["arguments"] for tc in tool_chunks)
    assert args == '{"a":2.0,"b":3.0}'

    # finish_reason tool_calls on the final chunk
    finish = [p for p in parsed if p["choices"][0].get("finish_reason") == "tool_calls"]
    assert len(finish) == 1


@pytest.mark.asyncio
async def test_streaming_translation_partial_events_buffered():
    """Events split across byte chunks are buffered and reassembled."""
    event = _sse_event("response.output_text.delta", {
        "type": "response.output_text.delta", "output_index": 1,
        "content_index": 0, "delta": "buffered",
    })
    # Split the event bytes at an arbitrary point
    mid = len(event) // 2
    chunks = await _collect_translated([event[:mid], event[mid:]])
    parsed = _parse_json_chunks(chunks)
    contents = [
        p["choices"][0]["delta"].get("content")
        for p in parsed
        if p["choices"][0]["delta"].get("content") is not None
    ]
    assert contents == ["buffered"]


@pytest.mark.asyncio
async def test_streaming_translation_empty_stream_terminates():
    """Stream ending without response.completed still terminates cleanly (no hang)."""
    events = [
        _sse_event("response.created", {"type": "response.created"}),
        _sse_event("ping", {"type": "ping"}),
    ]
    chunks = await _collect_translated(events)
    # No content, no [DONE] — but iteration completes without error
    assert chunks == []


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Handler wiring: responses_mode flag on the streaming/non-streaming handlers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_mock_response(status_code=200, headers=None, body_bytes=b"", aiter_chunks=None):
    """Build a minimal httpx.Response lookalike for handler tests."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, PropertyMock

    mock = MagicMock()
    mock.status_code = status_code
    mock.headers = headers or {"content-type": "text/event-stream"}
    mock.content = body_bytes
    if aiter_chunks is not None:
        async def _iter_chunks():
            for c in aiter_chunks:
                yield c
        mock.aiter_bytes = MagicMock(return_value=_iter_chunks())
    return mock


class _FakeRequest:
    method = "POST"

    @property
    def url(self):
        from types import SimpleNamespace
        return SimpleNamespace(path="/v1/chat/completions")

    async def body(self):
        return b"{}"


@pytest.mark.asyncio
async def test_non_streaming_handler_translates_responses_body():
    """responses_mode=True: upstream Responses JSON is returned as chat-completions JSON."""
    from unittest.mock import AsyncMock, patch

    import httpx
    from proxy.proxy_remote import _handle_remote_non_streaming
    from starlette.responses import Response as StarletteResponse

    upstream_body = json.dumps({
        "id": "resp_x",
        "object": "response",
        "status": "completed",
        "model": "muse-spark-1.2-contributor-free",
        "output": [
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "hello from muse"}]}
        ],
        "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
    }).encode()

    async def fake_do_request(*args, **kwargs):
        return _make_mock_response(status_code=200, headers={"content-type": "application/json"}, body_bytes=upstream_body)

    with patch(
        "proxy.proxy_remote._handle_remote_non_streaming",
        side_effect=None,
    ):
        pass  # placeholder; real invocation below

    # Rebind the handler's internal _do_request through the pool client path.
    pool_client = AsyncMock()
    pool_client.post = AsyncMock(side_effect=fake_do_request)

    resp = await _handle_remote_non_streaming(
        _FakeRequest(),
        target_url="https://opencode.ai/zen/v1/responses",
        headers={},
        body=b"{}",
        model_name="muse-spark-1.2-contributor-free",
        remote_timeout=httpx.Timeout(30.0),
        pool_client=pool_client,
        responses_mode=True,
    )
    assert isinstance(resp, StarletteResponse)
    body = json.loads(resp.body.decode())
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hello from muse"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    # upstream status preserved
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_non_streaming_handler_passes_through_error_status():
    """responses_mode=True but non-2xx: upstream error body passes through untranslated."""
    from unittest.mock import AsyncMock, patch

    import httpx
    from proxy.proxy_remote import _handle_remote_non_streaming
    from starlette.responses import Response as StarletteResponse

    err_body = json.dumps({
        "type": "error",
        "error": {"type": "error", "message": "Internal server error"},
    }).encode()

    pool_client = AsyncMock()
    pool_client.post = AsyncMock(return_value=_make_mock_response(
        status_code=500,
        headers={"content-type": "application/json"},
        body_bytes=err_body,
    ))

    resp = await _handle_remote_non_streaming(
        _FakeRequest(),
        target_url="https://opencode.ai/zen/v1/responses",
        headers={},
        body=b"{}",
        model_name="m",
        remote_timeout=httpx.Timeout(30.0),
        pool_client=pool_client,
        responses_mode=True,
    )
    assert resp.status_code == 500
    body = json.loads(resp.body.decode())
    # Untranslated: still a responses-shape error document, NOT empty chat JSON
    assert body.get("type") == "error"
    assert "choices" not in body


@pytest.mark.asyncio
async def test_streaming_handler_wraps_aiter_when_responses_mode():
    """responses_mode=True on the streaming handler translates Responses SSE to chat SSE."""
    from unittest.mock import AsyncMock, patch

    import httpx
    from proxy.proxy_remote import _handle_remote_streaming
    from starlette.responses import Response as StarletteResponse

    events = [
        _sse_event("response.output_text.delta", {
            "type": "response.output_text.delta", "output_index": 1,
            "content_index": 0, "delta": "streamed",
        }),
        _sse_event("response.completed", {
            "type": "response.completed",
            "response": {"id": "resp_s", "status": "completed", "model": "m"},
        }),
    ]

    stream_resp = _make_mock_response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        aiter_chunks=events,
    )
    pool_client = AsyncMock()
    spec = _FakeRequest()

    # Patch the internal per-request _do_request via the pool client, then
    # consume the returned StreamingResponse body to observe translated SSE.
    import httpx as _httpx

    from proxy import proxy_remote as pr

    # The handler builds a fresh AsyncClient when pool_client is None; use the
    # pool client and stub its stream() call.
    pool_client.stream = AsyncMock(return_value=stream_resp.__aiter__())

    # Simpler: exercise the streaming handler directly using a context-manager
    # client via patched httpx.AsyncClient.
    from unittest.mock import MagicMock

    class _FakeACL:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, *a, **k):
            return _FakeStreamCtx(stream_resp)

    class _FakeStreamCtx:
        def __init__(self, resp):
            self.resp = resp

        async def __aenter__(self):
            return self.resp

        async def __aexit__(self, *a):
            return False

    with patch.object(_httpx, "AsyncClient", return_value=_FakeACL()) as _mc:
        resp = await _handle_remote_streaming(
            spec,
            target_url="https://opencode.ai/zen/v1/responses",
            headers={},
            body=b"{}",
            body_json={},
            model_name="m",
            remote_timeout=_httpx.Timeout(30.0),
            pool_client=None,
            responses_mode=True,
        )

    assert isinstance(resp, StarletteResponse)
    collected = b""
    async for chunk in resp.body_iterator:
        collected += chunk
    text = collected.decode()
    assert '"content": "streamed"' in text
    # The handler terminates on the finish chunk (saw_finish) the same way it
    # does for chat-completions upstreams; the [DONE] terminator is consumed
    # internally as the completion signal.
    assert '"finish_reason": "stop"' in text
    assert "data: [DONE]" not in text
