"""Measured token usage for local streams (LP-0MVMHNVKT003PRIK).

Local ``Stream finished`` lines never carried ``tokens=`` because llama-server
reports usage in a chunk of its own, *after* the ``finish_reason`` chunk, and
the proxy never asked for it (``stream_options.include_usage``). Without
measured usage the usage-driven cache-credit path (``update_cached_ratio`` /
``record_prompt_prefix``) could never fire for local, so a session that missed
the slot-snapshot seeding stayed cold forever and was then bypassed by the
cold-cache token cap on every later turn.

These tests cover:

- AC1 (streaming): the local request asks llama-server for usage, the measured
  ``tokens=prompt/completion/total`` lands on the ``Stream finished`` line, and
  the client-visible stream is byte-compatible (the usage-only chunk is not
  forwarded unless the client asked for it).
- AC1 (non-streaming): a buffered local reply logs the same line, marked
  ``stream=false``.
- AC1 (no usage): a stream that never delivers a usage chunk still logs its
  terminal line (no ``tokens=``), so no line is lost.
- AC2/AC3: measured usage feeds ``update_cached_ratio`` +
  ``record_prompt_prefix``, and the next turn (40K total / ~10K new) routes
  local — the cold cap applies to NEW, uncached tokens.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.responses import StreamingResponse

MODEL = "Qwen3"
SESSION = "sess-local-usage-1"
COLD_CAP = 38000  # the 38K cold-cache cap
WARM_CAP = 100000  # per-slot warm (total-context) cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _usage_chunk(prompt: int, completion: int, cached: int) -> bytes:
    """The usage-only chunk llama.cpp emits with ``include_usage``."""
    payload = {
        "choices": [],
        "created": 0,
        "id": "chatcmpl-usage",
        "model": MODEL,
        "object": "chat.completion.chunk",
        "usage": {
            "completion_tokens": completion,
            "prompt_tokens": prompt,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


def _finish_chunk(reason: str = "stop") -> bytes:
    payload = {
        "choices": [{"finish_reason": reason, "index": 0, "delta": {}}],
        "created": 0,
        "id": "chatcmpl-finish",
        "model": MODEL,
        "object": "chat.completion.chunk",
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


_CONTENT_CHUNK = (
    b'data: {"choices":[{"delta":{"content":"hi"},"index":0}],'
    b'"created":0,"id":"chatcmpl-1","model":"Qwen3",'
    b'"object":"chat.completion.chunk"}\n\n'
)
_DONE = b"data: [DONE]\n\n"


def _finished_lines(logger_mock: MagicMock) -> list[str]:
    lines = []
    for call in logger_mock.info.call_args_list:
        template = call.args[0]
        if not str(template).startswith("Stream finished"):
            continue
        lines.append(template % call.args[1:] if len(call.args) > 1 else str(template))
    return lines


# ---------------------------------------------------------------------------
# router_helpers: the emit path (AC1)
# ---------------------------------------------------------------------------


class TestStreamFinishLine:
    """The terminal line carries the measured usage exactly once."""

    def test_finish_chunk_with_following_usage_chunk_reports_tokens(self):
        from proxy.router_helpers import log_response_chunk

        logger = MagicMock()
        srv = MagicMock(logger=logger)
        with patch("proxy.router_helpers._srv", return_value=srv):
            log_response_chunk(
                _finish_chunk(),
                session_id=SESSION,
                model=MODEL,
                provider="local",
                expect_usage=True,
            )
            # Nothing logged yet: the usage chunk has not arrived.
            assert _finished_lines(logger) == []
            log_response_chunk(
                _usage_chunk(1200, 50, 0),
                session_id=SESSION,
                model=MODEL,
                provider="local",
                expect_usage=True,
            )
            log_response_chunk(
                _DONE,
                session_id=SESSION,
                model=MODEL,
                provider="local",
                expect_usage=True,
            )

        lines = _finished_lines(logger)
        assert len(lines) == 1, lines
        assert "Stream finished: reason=stop tokens=1200/50/1250" in lines[0]
        assert f"session={SESSION}" in lines[0]
        assert "provider=local" in lines[0]
        assert f"model={MODEL}" in lines[0]

    def test_stream_without_usage_chunk_still_logs_terminal_line(self):
        """An aborted/usage-less stream never loses its terminal line."""
        from proxy.router_helpers import log_response_chunk

        logger = MagicMock()
        srv = MagicMock(logger=logger)
        with patch("proxy.router_helpers._srv", return_value=srv):
            log_response_chunk(
                _finish_chunk(),
                session_id=SESSION,
                model=MODEL,
                provider="local",
                expect_usage=True,
            )
            log_response_chunk(
                _DONE,
                session_id=SESSION,
                model=MODEL,
                provider="local",
                expect_usage=True,
            )

        lines = _finished_lines(logger)
        assert len(lines) == 1, lines
        assert "Stream finished: reason=stop" in lines[0]
        assert "tokens=" not in lines[0]

    def test_end_of_stream_flush_covers_no_done_marker(self):
        """The finally-path flush emits a deferred line when ``[DONE]`` never came."""
        from proxy.router_helpers import (
            flush_pending_stream_finish,
            log_response_chunk,
        )

        logger = MagicMock()
        srv = MagicMock(logger=logger)
        with patch("proxy.router_helpers._srv", return_value=srv):
            log_response_chunk(
                _finish_chunk("length"),
                session_id=SESSION,
                model=MODEL,
                provider="local",
                expect_usage=True,
            )
            assert _finished_lines(logger) == []
            assert flush_pending_stream_finish(
                SESSION, provider="local", model=MODEL
            )
            # A second flush is a no-op (never duplicate terminal lines).
            assert not flush_pending_stream_finish(
                SESSION, provider="local", model=MODEL
            )

        lines = _finished_lines(logger)
        assert len(lines) == 1, lines
        assert "reason=length" in lines[0]

    def test_without_expect_usage_behaviour_is_unchanged(self):
        """Remote providers (no ``expect_usage``) log on the finish chunk."""
        from proxy.router_helpers import log_response_chunk

        logger = MagicMock()
        srv = MagicMock(logger=logger)
        with patch("proxy.router_helpers._srv", return_value=srv):
            log_response_chunk(
                _finish_chunk(),
                session_id="remote-1",
                model="deepseek",
                provider="deepseek",
            )

        lines = _finished_lines(logger)
        assert len(lines) == 1
        assert "tokens=" not in lines[0]

    def test_usage_in_finish_chunk_is_unchanged(self):
        """A provider that reports usage inline keeps today's exact format."""
        from proxy.router_helpers import log_response_chunk

        payload = {
            "choices": [{"finish_reason": "stop", "index": 0, "delta": {}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
        }
        logger = MagicMock()
        srv = MagicMock(logger=logger)
        with patch("proxy.router_helpers._srv", return_value=srv):
            log_response_chunk(
                f"data: {json.dumps(payload)}\n\n".encode(),
                session_id="remote-1",
                model="deepseek",
                provider="deepseek",
            )

        lines = _finished_lines(logger)
        assert len(lines) == 1
        assert "tokens=10/2/12" in lines[0]


class TestUsageOnlyChunkDetection:
    def test_usage_only_chunk_is_detected(self):
        from proxy.router_helpers import is_usage_only_chunk

        assert is_usage_only_chunk(_usage_chunk(10, 2, 0)) is True

    def test_content_chunk_is_not_usage_only(self):
        from proxy.router_helpers import is_usage_only_chunk

        assert is_usage_only_chunk(_CONTENT_CHUNK) is False
        assert is_usage_only_chunk(_finish_chunk()) is False
        assert is_usage_only_chunk(_DONE) is False
        assert is_usage_only_chunk(b"") is False


class TestNonStreamingUsageLine:
    """AC1 non-streaming path: buffered replies report measured usage."""

    def test_buffered_reply_logs_tokens_and_stream_false(self):
        from proxy.router_helpers import log_non_stream_usage

        body = {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
            "usage": {
                "prompt_tokens": 3000,
                "completion_tokens": 120,
                "total_tokens": 3120,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
        logger = MagicMock()
        srv = MagicMock(logger=logger)
        with patch("proxy.router_helpers._srv", return_value=srv):
            log_non_stream_usage(
                json.dumps(body).encode(),
                session_id=SESSION,
                model=MODEL,
                provider="local",
            )

        lines = _finished_lines(logger)
        assert len(lines) == 1, lines
        line = lines[0]
        assert "Stream finished: reason=stop" in line
        assert "tokens=3000/120/3120" in line
        assert "stream=false" in line
        assert f"session={SESSION}" in line
        assert "provider=local" in line

    def test_buffered_reply_without_usage_logs_nothing(self):
        from proxy.router_helpers import log_non_stream_usage

        logger = MagicMock()
        srv = MagicMock(logger=logger)
        with patch("proxy.router_helpers._srv", return_value=srv):
            log_non_stream_usage(
                b'{"choices":[{"index":0,"message":{"content":"hi"}}]}',
                session_id=SESSION,
                model=MODEL,
                provider="local",
            )
            log_non_stream_usage(None, session_id=SESSION, provider="local")
            log_non_stream_usage(b"not json", session_id=SESSION, provider="local")

        assert _finished_lines(logger) == []


# ---------------------------------------------------------------------------
# Router integration: include_usage injection + client-visible stream
# ---------------------------------------------------------------------------


def _mock_server():
    srv = MagicMock()
    srv.config = {
        "server": {
            "max_concurrent_queries": 4,
            "session_single_flight_mode": "queue",
            "session_single_flight_max_queue_depth": 1,
        }
    }
    srv.active_queries = 0
    srv.local_active_queries = 0
    srv.backend_ready = True
    srv.llama_process = MagicMock()
    srv.current_model = MODEL
    srv.logger = MagicMock()
    srv.session_manager = AsyncMock()
    srv.local_dispatch_records = {}
    srv.local_dispatch_records_lock = AsyncMock()
    return srv


def _mock_request(body: dict):
    req = MagicMock()
    req.method = "POST"
    req.headers = {"host": "localhost"}
    req.url = type("U", (), {"path": "/v1/chat/completions"})()
    req.body = AsyncMock(return_value=json.dumps(body).encode())
    req.is_disconnected = AsyncMock(return_value=False)
    return req


class _StreamResponse:
    def __init__(self, chunks: list[bytes]):
        self.status_code = 200
        self.headers = {"content-type": "text/event-stream"}
        self.content = b""
        self._chunks = chunks

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aread(self):
        return b""


class _CM:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


async def _drive_local_stream(body: dict, chunks: list[bytes]):
    """Run the local streaming path and return (client bytes, upstream body, srv)."""
    import proxy.router as router

    srv = _mock_server()
    response = _StreamResponse(chunks)
    cm = _CM(response)
    client = MagicMock()
    client.stream = MagicMock(return_value=cm)
    client.aclose = AsyncMock()

    async def _session(*args, **_kwargs):
        # ``_handle_session(srv, body_json, server_config, headers)``: echo the
        # router's parsed body so the injected ``stream_options`` is forwarded.
        body_json = args[1] if len(args) > 1 else {}
        return {
            "session_id": SESSION,
            "session_created": True,
            "is_delta_request": False,
            "session_fallback_reason": None,
            "delta_messages": None,
            "body_override": None,
            "body_json": body_json,
            "original_message_count": 1,
            "session_explicit": False,
        }

    stack = ExitStack()
    patches = [
        patch("proxy.router._srv", return_value=srv),
        patch("proxy.router_helpers._srv", return_value=srv),
        patch("proxy.router._is_self_healing_active", return_value=False),
        patch("proxy.router._check_slot_availability", return_value=None),
        patch("proxy.router._handle_session", side_effect=_session),
        patch("proxy.router._build_slot_context", return_value=(None, None, 3.0)),
        patch("proxy.router.httpx.AsyncClient", return_value=client),
        patch("proxy.router._schedule_token_increment"),
        patch("proxy.router._schedule_recv_token_increment"),
        patch("proxy.router.count_text_tokens", return_value=5),
        patch("proxy.router._extract_delta_text_from_sse_chunk", return_value=""),
        patch("proxy.router.evaluate_stream_guardrail", return_value=None),
        patch("proxy.router.log_request"),
        patch(
            "proxy.router._detect_restore_signal_from_llama_log", return_value=False
        ),
        patch("proxy.session._resolve_log_path", return_value=Path("/nonexistent")),
        patch("proxy.router._update_session_and_slot", new=AsyncMock()),
        patch("proxy.router._cleanup_after_request", new=AsyncMock()),
        patch("proxy.router._increment_local_active_queries", new=AsyncMock()),
        patch("proxy.router._increment_active_queries", new=AsyncMock()),
        patch("proxy.router._increment_per_model_query", new=AsyncMock()),
        patch(
            "proxy.router_helpers._decrement_generating_only_slot", new=AsyncMock()
        ),
        patch(
            "proxy.router_helpers._increment_generating_only_slot", new=AsyncMock()
        ),
    ]
    async def _retry(factory, **_kwargs):
        """Run the real stream factory() so the client call args are recorded."""
        return await factory()

    patches.insert(0, patch(
        "proxy.router._call_with_backend_retries", side_effect=_retry
    ))
    try:
        for p in patches:
            stack.enter_context(p)
        result = await router.proxy_to_local(
            _mock_request(body), "v1/chat/completions"
        )
        assert isinstance(result, StreamingResponse)
        received = b""
        async for chunk in result.body_iterator:
            received += chunk
    finally:
        # Always restore patches: a leaked ``httpx.AsyncClient`` patch would
        # break every later test in the session.
        stack.close()

    sent_body = json.loads(client.stream.call_args.kwargs["content"])
    return received, sent_body, srv


@pytest.mark.asyncio
async def test_local_stream_requests_usage_and_hides_the_usage_chunk():
    """AC1: usage is requested upstream, logged, and not leaked to the client."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    received, sent_body, srv = await _drive_local_stream(
        body,
        [_CONTENT_CHUNK, _finish_chunk(), _usage_chunk(1200, 50, 0), _DONE],
    )

    assert sent_body["stream_options"]["include_usage"] is True
    # The client asked for no usage, so its stream is unchanged.
    assert b'"usage"' not in received
    assert _finish_chunk() in received
    assert _DONE in received

    lines = _finished_lines(srv.logger)
    assert len(lines) == 1, lines
    assert "Stream finished: reason=stop tokens=1200/50/1250" in lines[0]
    assert "provider=local" in lines[0]


@pytest.mark.asyncio
async def test_client_requested_usage_is_forwarded_and_logged():
    """A client that asks for usage still receives it; the line still has tokens."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    received, sent_body, srv = await _drive_local_stream(
        body,
        [_CONTENT_CHUNK, _finish_chunk(), _usage_chunk(400, 8, 380), _DONE],
    )

    assert sent_body["stream_options"]["include_usage"] is True
    assert _usage_chunk(400, 8, 380) in received

    lines = _finished_lines(srv.logger)
    assert len(lines) == 1, lines
    assert "tokens=400/8/408" in lines[0]


# ---------------------------------------------------------------------------
# AC2/AC3: measured usage -> cache credit -> cold-cap routing
# ---------------------------------------------------------------------------


def _turn1_messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "turn 1: summarise the module layout"},
    ]


def _turn2_messages() -> list[dict]:
    return _turn1_messages() + [
        {"role": "assistant", "content": "The proxy routes requests to providers."},
        {"role": "user", "content": "turn 2: now explain the routing decision"},
    ]


@pytest.fixture(autouse=True)
def _clean_provider_state():
    """Isolate process-global ratio/prefix state between tests."""
    import proxy.provider as provider

    provider._last_cached_ratio.clear()
    provider._cached_ratio_source.clear()
    provider._last_prompt_prefix.clear()
    yield
    provider._last_cached_ratio.clear()
    provider._cached_ratio_source.clear()
    provider._last_prompt_prefix.clear()


@pytest.mark.asyncio
async def test_router_records_measured_usage_for_local_streams():
    """AC2: the router's post-stream accounting consumes the usage block."""
    import proxy.provider as provider
    import proxy.router as router

    srv = _mock_server()
    sse = b"".join([_CONTENT_CHUNK, _finish_chunk(), _usage_chunk(30000, 200, 0)])
    response = _StreamResponse([sse])
    response.headers = {}

    body_json = {
        "model": MODEL,
        "messages": _turn1_messages(),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    with patch(
        "proxy.router._detect_restore_signal_from_llama_log", return_value=False
    ), patch("proxy.router._has_explicit_restore_signal", return_value=False):
        await router._update_session_and_slot(
            srv,
            SESSION,
            body_json,
            False,
            None,
            1,
            response,
            8080,
            None,
            None,
            3.0,
            None,
            False,
            upstream_status=200,
            slot_save_allowed=False,
            collected_content=[sse.decode("utf-8")],
        )

    assert provider._get_cached_ratio(MODEL, SESSION) == pytest.approx(0.0)
    assert provider._last_prompt_prefix[(MODEL, SESSION)][0] == 30000


@pytest.mark.asyncio
async def test_sequential_turns_stay_local_under_the_cold_cap():
    """AC2/AC3: 30K then 40K total / ~10K new keeps routing local.

    Without the measured credit the 40K turn is costed as 40K NEW tokens and
    is skipped by the 38K cold cap; with the measured turn-1 usage the new
    tokens are only ~13K, so local is served — the cap applies to new
    (uncached) tokens, total context being bounded separately by the warm cap.
    """
    import proxy.provider as provider
    import proxy.router as router

    # Turn 1 — below the cold cap on its own, routes local.
    skip, reason = provider._should_skip_local(
        MODEL,
        SESSION,
        {"messages": _turn1_messages()},
        COLD_CAP,
        estimated_tokens=30000,
        warm_cache_threshold=WARM_CAP,
    )
    assert (skip, reason) == (False, None)

    # Turn 1 is then served locally and reports measured usage (30K prompt,
    # nothing cached yet) — the router applies exactly this accounting.
    srv = _mock_server()
    sse = b"".join([_CONTENT_CHUNK, _finish_chunk(), _usage_chunk(30000, 200, 0)])
    response = _StreamResponse([sse])
    response.headers = {}
    with patch(
        "proxy.router._detect_restore_signal_from_llama_log", return_value=False
    ), patch("proxy.router._has_explicit_restore_signal", return_value=False):
        await router._update_session_and_slot(
            srv,
            SESSION,
            {
                "model": MODEL,
                "messages": _turn1_messages(),
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            False,
            None,
            1,
            response,
            8080,
            None,
            None,
            3.0,
            None,
            False,
            upstream_status=200,
            slot_save_allowed=False,
            collected_content=[sse.decode("utf-8")],
        )

    # Turn 2 — 40K total. Pre-fix (cold ratio 0.0) the 40K new tokens exceed
    # the 38K cap; the measured prefix credit prices the real ~10K of new work.
    ratio = provider.expected_cached_ratio(
        MODEL, SESSION, 40000, _turn2_messages()
    )
    new_tokens = int(40000 * (1 - ratio))
    assert new_tokens < COLD_CAP, (
        f"expected only the ~10K uncached tokens to be priced, got {new_tokens}"
    )
    assert 10000 <= new_tokens <= 14000

    skip, reason = provider._should_skip_local(
        MODEL,
        SESSION,
        {"messages": _turn2_messages()},
        COLD_CAP,
        estimated_tokens=40000,
        warm_cache_threshold=WARM_CAP,
    )
    assert (skip, reason) == (False, None)

    # And the counterfactual: with no measured turn 1 the same turn is skipped.
    provider._last_cached_ratio.clear()
    provider._cached_ratio_source.clear()
    provider._last_prompt_prefix.clear()
    cold_ratio = provider.expected_cached_ratio(
        MODEL, SESSION, 40000, _turn2_messages()
    )
    assert int(40000 * (1 - cold_ratio)) > COLD_CAP
    skip, reason = provider._should_skip_local(
        MODEL,
        SESSION,
        {"messages": _turn2_messages()},
        COLD_CAP,
        estimated_tokens=40000,
        warm_cache_threshold=WARM_CAP,
    )
    assert (skip, reason) == (True, "large_context_bypass")
