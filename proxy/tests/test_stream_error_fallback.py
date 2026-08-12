"""
Tests for fallback-to-next-provider on pre-content streaming failures and the
enriched stream-error SSE payload (LP-0MSETOTWY000SU0Z).

Covers the Aug-3 error-analysis recommendations (proxy/docs/error-analysis-
2026-08-03.md):

- Recommendation 1 (recovery-first): a remote streaming response that fails
  BEFORE any content-bearing chunk is delivered (stall retries exhausted,
  empty response, stream exception) re-routes to the next provider in the
  configured chain instead of surfacing a bare `finish_reason: error`.
- Recommendation 2 (informative error): every synthetic `finish_reason: error`
  SSE event carries a structured `error` payload (type, message, provider,
  model, entry, suggested_action) so the client can act instead of seeing an
  unspecified error.
- After-content failures (LP-0MS9FR9LG002AJ4C) still terminate with the error
  event; they never re-route (re-sending would duplicate output).
"""

import json
from unittest.mock import AsyncMock, patch

import proxy.provider as provider
import pytest
from fastapi import Response
from fastapi.responses import StreamingResponse


class _DummyRequest:
    """Minimal request stub (mirrors test_provider_fallback._DummyRequest)."""

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


def _error_stream_chunks():
    """SSE bytes: keep-alive comments then a bare finish_reason:error event.

    Mirrors the real opencode-go stall signature from the Aug-4 session
    recordings (keep-alives + `{"delta": {}, "finish_reason": "error"}`).
    """
    return [
        b": keep-alive\n\n",
        b": keep-alive\n\n",
        b'data: {"choices": [{"delta": {}, "finish_reason": "error", "index": 0}]}\n\n',
    ]


def _content_stream_chunks():
    return [
        b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n\n',
        b'data: {"choices": [{"delta": {"content": " world"}, "index": 0}]}\n\n',
        b'data: {"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}\n\n',
    ]


def _after_content_error_stream_chunks():
    """Content delivered, THEN a terminal error (after-content stall)."""
    return [
        b'data: {"choices": [{"delta": {"content": "Partial"}, "index": 0}]}\n\n',
        b'data: {"choices": [{"delta": {}, "finish_reason": "error", "index": 0}]}\n\n',
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


# ===================================================================
# Recommendation 1: pre-content failures re-route to the next provider
# ===================================================================


@pytest.mark.asyncio
async def test_precontent_finish_reason_error_triggers_fallback(two_provider_config):
    """A streaming response whose first meaningful event is finish_reason:error
    (zero content delivered) falls back to the next provider."""
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_streaming_response(_error_stream_chunks())
        return _ok_json_response()

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", two_provider_config,
            {"provider_cooldown_seconds": 60},
        )

    assert call_count == 2, (
        f"Expected fallback after pre-content error, got {call_count} calls"
    )
    assert result.status_code == 200
    body = json.loads(result.body.decode()) if isinstance(result.body, bytes) else json.loads(result.body)
    assert body["choices"][0]["message"]["content"] == "ok"
    # Failed provider must be in cooldown (Tier-2)
    assert provider._is_provider_unavailable("remote-primary")


@pytest.mark.asyncio
async def test_precontent_error_all_providers_exhausted(two_provider_config):
    """When every provider fails pre-content, the chain is exhausted: both
    providers are attempted and the final stream terminates with a
    finish_reason: error event (never a silent abort). The last provider's
    stream is passed through as-is — its enriched payload is emitted by
    _handle_remote_streaming (covered by
    test_handle_remote_streaming_stall_exhaustion_emits_enriched_error)."""
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        return _make_streaming_response(_error_stream_chunks())

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", two_provider_config,
            {"provider_cooldown_seconds": 60},
        )

    # Both providers were tried; the final stream terminates with an error.
    assert call_count == 2
    assert isinstance(result, StreamingResponse)
    collected = b"".join([c async for c in result.body_iterator])
    decoded = collected.decode("utf-8", errors="replace")
    assert '"finish_reason": "error"' in decoded, (
        f"Expected a terminal error event in output: {decoded!r}"
    )
    # The re-routed provider must be in cooldown (Tier-2). The last provider
    # is passed through as-is (no next provider to fall back to), so it is not
    # marked unavailable here.
    assert provider._is_provider_unavailable("remote-primary")


@pytest.mark.asyncio
async def test_streaming_success_passes_through_unchanged(two_provider_config):
    """A healthy streaming response with content is NOT re-routed; the client
    sees the content and the X-Provider header."""
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        return _make_streaming_response(_content_stream_chunks())

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", two_provider_config,
            {"provider_cooldown_seconds": 60},
        )

    assert call_count == 1, (
        f"Expected no fallback for healthy stream, got {call_count} calls"
    )
    assert isinstance(result, StreamingResponse)
    assert result.headers.get("X-Provider") == "remote-primary"
    collected = b"".join([c async for c in result.body_iterator])
    assert b"Hello" in collected and b"world" in collected


@pytest.mark.asyncio
async def test_after_content_error_does_not_fallback(two_provider_config):
    """Content delivered then error (after-content stall) never re-routes:
    the stream terminates with the error event as per LP-0MS9FR9LG002AJ4C."""
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        return _make_streaming_response(_after_content_error_stream_chunks())

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", two_provider_config,
            {"provider_cooldown_seconds": 60},
        )

    assert call_count == 1, (
        f"After-content failure must NOT re-route, got {call_count} calls"
    )
    collected = b"".join([c async for c in result.body_iterator])
    assert b"Partial" in collected, "Partial content must reach the client"
    assert b"finish_reason" in collected and b'"error"' in collected


# ===================================================================
# Edge cases: client disconnect and last-provider pass-through
# ===================================================================


class _DisconnectingRequest(_DummyRequest):
    """Request that reports disconnected=True immediately."""

    async def is_disconnected(self):
        return True


@pytest.mark.asyncio
async def test_preflight_client_disconnect_does_not_reroute(two_provider_config):
    """If the client disconnects during pre-flight, the response is handed
    back as-is and no re-route to the next provider is attempted (AC4:
    re-routing to a dead client would be wasteful)."""
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        return _make_streaming_response(_error_stream_chunks())

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            _DisconnectingRequest(), "v1/chat/completions", two_provider_config,
            {"provider_cooldown_seconds": 60},
        )

    assert call_count == 1, (
        f"Expected NO re-route on client disconnect, got {call_count} calls"
    )
    assert isinstance(result, StreamingResponse)


@pytest.mark.asyncio
async def test_last_provider_stream_passes_through(two_provider_config):
    """The last provider in the chain is NOT pre-flighted (no remaining
    provider to fall back to), so its stream reaches the client as-is."""
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        return _make_streaming_response(_content_stream_chunks())

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        # Single-provider config: no next provider, no pre-flight.
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions",
            {"providers": [two_provider_config["providers"][0]], "aliases": ["solo*"]},
            {"provider_cooldown_seconds": 60},
        )

    assert call_count == 1
    assert isinstance(result, StreamingResponse)
    collected = b"".join([c async for c in result.body_iterator])
    assert b"Hello" in collected
    # Provider NOT marked unavailable (no failure, nothing to fall back to).
    assert not provider._is_provider_unavailable("remote-primary")


# ===================================================================
# Recommendation 2: enriched error payload on synthetic error events
# ===================================================================


def test_build_stream_error_event_shape():
    """The shared helper emits a structured error object alongside the
    (backward-compatible) finish_reason: error."""
    from proxy.proxy_remote import _build_stream_error_event

    event = _build_stream_error_event(
        provider="opencode-go",
        model="deepseek-v4-flash",
        entry="opencode-go-2-deepseek",
        error_type="stall_exhausted",
        message="Upstream stalled repeatedly; retries exhausted",
        suggested_action="Provider placed in cooldown; next provider will be used",
        session_id="sess-123",
    )
    choice = event["choices"][0]
    assert choice["finish_reason"] == "error"
    assert choice["delta"] == {}
    err = choice["error"]
    assert err["type"] == "stall_exhausted"
    assert "retries exhausted" in err["message"]
    assert err["provider"] == "opencode-go"
    assert err["model"] == "deepseek-v4-flash"
    assert err["entry"] == "opencode-go-2-deepseek"
    assert err["session_id"] == "sess-123"
    assert "next provider" in err["suggested_action"]


def test_build_stream_error_event_defaults():
    """Defaults: unknown provider/model and a generic message/type."""
    from proxy.proxy_remote import _build_stream_error_event

    event = _build_stream_error_event()
    choice = event["choices"][0]
    assert choice["finish_reason"] == "error"
    err = choice["error"]
    assert err["type"] == "stream_error"
    assert err["provider"] == "unknown"
    assert err["model"] == "unknown"
    assert err["message"]


@pytest.mark.asyncio
async def test_handle_remote_streaming_stall_exhaustion_emits_enriched_error():
    """End-to-end: _handle_remote_streaming with zero content and retries
    exhausted emits the enriched error event (provider/model/entry present)."""
    import asyncio
    from unittest.mock import MagicMock, PropertyMock

    import httpx
    from fastapi import Request
    from proxy.proxy_remote import _handle_remote_streaming

    class AsyncChunkIterator:
        def __init__(self, chunks):
            self._chunks = list(chunks)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._chunks:
                return self._chunks.pop(0)
            await asyncio.Event().wait()  # hang forever (stall)

    mock_resp = MagicMock(spec=httpx.Response)
    type(mock_resp).status_code = PropertyMock(return_value=200)
    mock_resp.headers = {"content-type": "text/event-stream"}
    mock_resp.aiter_bytes = MagicMock(
        return_value=AsyncChunkIterator([b": keep-alive\n\n"])
    )

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_resp)
    cm.__aexit__ = AsyncMock(return_value=None)
    client = MagicMock(spec=httpx.AsyncClient)
    client.stream = MagicMock(return_value=cm)
    client.aclose = AsyncMock(return_value=None)

    req = MagicMock(spec=Request)
    req.method = "POST"
    req.url.path = "/v1/chat/completions"
    req.is_disconnected = AsyncMock(return_value=False)

    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        with patch("proxy.proxy_remote._srv") as mock_srv:
                            mock_srv.return_value.config = {
                                "server": {
                                    "upstream_retry_max_attempts": 0,
                                    "upstream_retry_base_delay_seconds": 0.01,
                                }
                            }
                            mock_srv.return_value.logger = MagicMock()
                            result = await _handle_remote_streaming(
                                request=req,
                                target_url="https://api.example.com/v1/chat/completions",
                                headers={"Authorization": "Bearer test"},
                                body=b'{"stream": true, "model": "test"}',
                                body_json={"stream": True, "model": "test"},
                                model_name="test-model",
                                remote_timeout=httpx.Timeout(30.0),
                                upstream_idle_timeout_seconds=0.05,
                                provider="opencode-go",
                                entry="opencode-go-2-deepseek",
                            )
                            collected = b"".join(
                                [c async for c in result.body_iterator]
                            )

    decoded = collected.decode("utf-8", errors="replace")
    error_events = []
    for line in decoded.splitlines():
        line = line.strip()
        if line.startswith("data:") and '"finish_reason"' in line:
            payload = json.loads(line[5:].strip())
            for choice in payload.get("choices", []):
                if choice.get("finish_reason") == "error":
                    error_events.append(choice)

    assert error_events, f"Expected an error event, got: {decoded!r}"
    last = error_events[-1]
    assert last["delta"] == {}
    err = last.get("error")
    assert err is not None, "Enriched error payload missing from error event"
    assert err["provider"] == "opencode-go"
    assert err["model"] == "test-model"
    assert err["entry"] == "opencode-go-2-deepseek"
    assert err["type"] in {"stall_exhausted", "stream_error"}
