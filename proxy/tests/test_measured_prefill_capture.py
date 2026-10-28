"""Measured prefill capture from local ``prompt_progress`` (LP-0MVR7DAFF006HBRX).

Parent: LP-0MVQIT9QY007QU27 (``local_eviction_after_acceptance
reason=prefill_deadline``, producer-approved Option D).

The parent's finding was that warmth is *predicted*, never *measured*: all
5,456 ``routing_resident_prefix`` decisions in the measured window carried
``n_past_actual=none``, so a clobbered slot owner is admitted on a claimed
``reuse_ratio=1.0`` and then re-prefills the whole prompt until the 120 s
deadline releases it. The ground truth is llama-server's own
``prompt_progress`` event (contract pinned by :mod:`proxy.prompt_progress`,
F1): when the request carries ``return_progress: true`` the first event's
``cache`` is the exact reused prefix against ``total``.

This item is the *capture* half (F2), purely additive — it wires the injection
and the parse, records the measurement, and logs it; the early-abort decision
is F4.

AC mapping (this item):
  AC1 injection of ``return_progress`` on local streams only, never leaked into
      recorded/stored payloads;
  AC2 ``prompt_progress`` events parsed from the reframed stream and the latest
      ``{total, cache, processed, time_ms}`` recorded per dispatch;
  AC3 the dispatch record exposes ``cache / total`` and ``total - cache``;
  AC4 the client-visible stream is byte-for-byte unchanged (progress events are
      proxy-internal and withheld);
  AC5 fail-open on missing/malformed events and on capture exceptions;
  AC6 the ``prefill_progress_measured`` line carries the attributable fields.

Harness: the local-streaming driver mirrors
``test_local_stream_usage_tokens.py`` (the established ``proxy_to_local``
integration harness) and extends it to record the client/proxy payloads the
proxy stores and to expose the dispatch-lease record.
"""

from __future__ import annotations

import json
import re
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import proxy.measured_prefill as measured_prefill
import pytest
from fastapi.responses import StreamingResponse
from proxy.measured_prefill import (
    PREFILL_PROGRESS_EVENT,
    MeasuredPrefill,
    capture_enabled,
    get_measured_prefill,
    inject_return_progress,
)
from proxy.prompt_progress import parse_prompt_progress_event

MODEL = "Qwen3"
SESSION = "sess-measured-prefill-1"

#: The b8782 event shape verified live (F1): ``prompt_progress`` is a top-level
#: key next to ``choices`` — never nested inside the delta.
_CONTENT_CHUNK = (
    b'data: {"choices":[{"delta":{"content":"hi"},"index":0}],'
    b'"created":0,"id":"chatcmpl-1","model":"Qwen3",'
    b'"object":"chat.completion.chunk"}\n\n'
)
_FINISH_CHUNK = (
    b'data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}],'
    b'"created":0,"id":"chatcmpl-finish","model":"Qwen3",'
    b'"object":"chat.completion.chunk"}\n\n'
)
_DONE = b"data: [DONE]\n\n"


def _progress_event(total: int, cache: int, processed: int = 0, time_ms: int = 0) -> bytes:
    """One b8782 ``prompt_progress`` chunk (the shape the server emits)."""
    payload = {
        "choices": [
            {
                "finish_reason": None,
                "index": 0,
                "delta": {"role": "assistant", "content": None},
            }
        ],
        "created": 0,
        "id": "chatcmpl-progress",
        "model": MODEL,
        "system_fingerprint": "b8782-e97492369",
        "object": "chat.completion.chunk",
        "prompt_progress": {
            "total": total,
            "cache": cache,
            "processed": processed,
            "time_ms": time_ms,
        },
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


# ---------------------------------------------------------------------------
# Unit: injection (AC1)
# ---------------------------------------------------------------------------


class TestInjection:
    """The upstream body carries the key; the parsed client body never does."""

    def test_streaming_body_gets_return_progress(self):
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        upstream = inject_return_progress(body)
        assert upstream is not None
        sent = json.loads(upstream)
        assert sent["return_progress"] is True
        # The caller's parsed body (the recorded/stored payload) is untouched.
        assert set(body) == {"model", "messages", "stream"}
        # Everything else is forwarded verbatim.
        assert sent["model"] == MODEL
        assert sent["messages"] == body["messages"]

    def test_non_streaming_body_is_not_injected(self):
        assert (
            inject_return_progress(
                {"model": MODEL, "messages": [], "stream": False}
            )
            is None
        )

    def test_client_supplied_return_progress_is_not_rewritten(self):
        body = {"model": MODEL, "stream": True, "return_progress": False}
        assert inject_return_progress(body) is None
        assert body["return_progress"] is False

    def test_non_mapping_body_is_not_injected(self):
        assert inject_return_progress(None) is None
        assert inject_return_progress([1, 2, 3]) is None


# ---------------------------------------------------------------------------
# Unit: capture_enabled (parent AC4 rollback key)
# ---------------------------------------------------------------------------


class TestCaptureEnabled:
    def test_default_enabled(self):
        assert capture_enabled({}) is True
        assert capture_enabled(None) is True

    def test_rollback_key_disables_capture(self):
        assert capture_enabled({"local_measured_cache_abort_enabled": 0}) is False
        assert (
            capture_enabled({"local_measured_cache_abort_enabled": "false"})
            is False
        )

    def test_malformed_config_fails_open_to_pre_change_behaviour(self):
        assert (
            capture_enabled({"local_measured_cache_abort_enabled": "maybe"})
            is False
        )


# ---------------------------------------------------------------------------
# Unit: the measured record (AC3/AC6)
# ---------------------------------------------------------------------------


class TestMeasuredPrefillRecord:
    def test_derived_values_match_the_stream(self):
        progress = parse_prompt_progress_event(
            _progress_event(total=56_000, cache=200)
        )
        assert progress is not None
        measured = MeasuredPrefill.from_progress(
            progress, priced_new_tokens=150.0
        )
        assert measured.total == 56_000
        assert measured.cache == 200
        assert measured.processed == 0
        assert measured.measured_cached_ratio == pytest.approx(200 / 56_000)
        assert measured.measured_new_tokens == 55_800
        record = measured.as_dict()
        assert record["measured_cached_ratio"] == pytest.approx(200 / 56_000)
        assert record["measured_new_tokens"] == 55_800
        assert record["priced_new_tokens"] == 150.0

    def test_warm_measurement_reports_near_full_reuse(self):
        progress = parse_prompt_progress_event(
            _progress_event(total=30_000, cache=29_970)
        )
        assert progress is not None
        measured = MeasuredPrefill.from_progress(progress)
        assert measured.measured_cached_ratio == pytest.approx(0.999)
        assert measured.measured_new_tokens == 30

    def test_log_fields_carry_the_attributable_numbers(self):
        progress = parse_prompt_progress_event(
            _progress_event(total=56_000, cache=200, processed=0, time_ms=12)
        )
        assert progress is not None
        fields = MeasuredPrefill.from_progress(
            progress, priced_new_tokens=150.0
        ).log_fields()
        parsed = dict(re.findall(r"(\w+)=(\S+)", fields))
        assert parsed["total"] == "56000"
        assert parsed["cache"] == "200"
        assert parsed["processed"] == "0"
        assert parsed["measured_cached_ratio"] == "0.0036"
        assert parsed["priced_new_tokens"] == "150"
        assert parsed["measured_new_tokens"] == "55800"

    def test_unknown_fields_are_logged_as_unknown(self):
        measured = MeasuredPrefill(
            total=None, cache=None, processed=None, time_ms=None
        )
        fields = measured.log_fields()
        assert "total=unknown" in fields
        assert "measured_cached_ratio=unknown" in fields
        assert "priced_new_tokens=unknown" in fields


# ---------------------------------------------------------------------------
# Router integration harness
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
    # The admission-time pricing snapshot the routing layer stamps on the
    # request (LP-0MVODJ3P6000AIUM); AC6's ``priced_new_tokens`` reads it.
    req._local_admission_evidence = {
        "estimated_tokens": 56_000,
        "cached_ratio": 0.0,
        "new_tokens": 150,
    }
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


def _recorded_payloads(record_mock: MagicMock) -> list[dict]:
    """Every payload the proxy handed to traffic recording (client + proxy)."""
    payloads = []
    for call in record_mock.call_args_list:
        for name in ("client_payload", "proxy_payload"):
            payload = call.kwargs.get(name)
            if isinstance(payload, dict):
                payloads.append(payload)
    return payloads


async def _drive_local_stream(
    body: dict,
    chunks: list[bytes],
    *,
    server_config_extra: dict | None = None,
    preload_dispatch_record: bool = True,
):
    """Run the local streaming path and return its observable outcomes."""
    import proxy.router as router

    srv = _mock_server()
    srv.config["server"].update(server_config_extra or {})
    if preload_dispatch_record:
        # ``endpoint`` is None on this path, so the lease key is the bare
        # session id (``_dispatch_lease_key``).
        srv.local_dispatch_records[SESSION] = {"active": True}
    response = _StreamResponse(chunks)
    cm = _CM(response)
    client = MagicMock()
    client.stream = MagicMock(return_value=cm)
    client.aclose = AsyncMock()

    async def _session(*args, **_kwargs):
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

    async def _retry(factory, **_kwargs):
        return await factory()

    record_mock = MagicMock()
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
        patch("proxy.router._schedule_traffic_recording", record_mock),
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
        # Deterministic prompt estimate for the AC6 ``estimated_tokens`` path.
        patch(
            "proxy.provider._estimate_effective_prompt_tokens_for_routing",
            new=AsyncMock(return_value=56_000),
        ),
    ]
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
        stack.close()

    sent_body = json.loads(client.stream.call_args.kwargs["content"])
    return received, sent_body, srv, record_mock


def _stream_body() -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }


def _measured_log_lines(logger: MagicMock) -> list[str]:
    lines = []
    for call in logger.info.call_args_list:
        template = str(call.args[0]) if call.args else ""
        rendered = (
            template % tuple(call.args[1:])
            if "%s" in template and len(call.args) > 1
            else template
        )
        if rendered.startswith(PREFILL_PROGRESS_EVENT):
            lines.append(rendered)
    return lines


@pytest.fixture(autouse=True)
def _clean_measured_prefill_store():
    measured_prefill.reset_measured_prefill()
    yield
    measured_prefill.reset_measured_prefill()


# ---------------------------------------------------------------------------
# AC1 — injection on the real local streaming path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_stream_injects_return_progress_without_leaking_it():
    """AC1: upstream gets the key; recorded client/proxy payloads do not."""
    received, sent_body, _srv, record_mock = await _drive_local_stream(
        _stream_body(), [_CONTENT_CHUNK, _FINISH_CHUNK, _DONE]
    )

    assert sent_body["return_progress"] is True

    payloads = _recorded_payloads(record_mock)
    assert payloads, "expected client/proxy payloads to be recorded"
    for payload in payloads:
        assert "return_progress" not in payload

    # The client-visible stream still ends normally.
    assert _FINISH_CHUNK in received
    assert _DONE in received


@pytest.mark.asyncio
async def test_rollback_key_disables_injection_and_capture():
    """Parent AC4: the rollback key restores the pre-change request body."""
    received, sent_body, srv, _record_mock = await _drive_local_stream(
        _stream_body(),
        [_progress_event(56_000, 200), _CONTENT_CHUNK, _FINISH_CHUNK, _DONE],
        server_config_extra={"local_measured_cache_abort_enabled": 0},
    )

    assert "return_progress" not in sent_body
    assert get_measured_prefill(MODEL, SESSION) is None
    assert _measured_log_lines(srv.logger) == []
    # With capture disabled the progress chunk is just an upstream event.
    assert _progress_event(56_000, 200) in received


# ---------------------------------------------------------------------------
# AC2/AC3 — capture + dispatch-record exposure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_progress_events_are_captured_per_dispatch():
    """AC2/AC3: the latest measurement is recorded and exposed."""
    # b8782 fixes ``cache`` (``n_prompt_tokens_cache``) before prefill starts;
    # only ``processed``/``time_ms`` advance with later events.
    first = _progress_event(56_000, 200, processed=0, time_ms=0)
    later = _progress_event(56_000, 200, processed=8_192, time_ms=410)
    received, _sent, srv, _record = await _drive_local_stream(
        _stream_body(), [first, _CONTENT_CHUNK, later, _FINISH_CHUNK, _DONE]
    )

    measured = get_measured_prefill(MODEL, SESSION)
    assert measured is not None
    assert (measured.total, measured.cache) == (56_000, 200)
    # The latest event wins in the store (progress advanced).
    assert measured.processed == 8_192
    assert measured.time_ms == 410.0
    assert measured.measured_cached_ratio == pytest.approx(200 / 56_000)
    assert measured.measured_new_tokens == 55_800

    # AC3: the dispatch record itself exposes the derived numbers.
    record = srv.local_dispatch_records[SESSION]["measured_prefill"]
    assert record["measured_cached_ratio"] == pytest.approx(200 / 56_000)
    assert record["measured_new_tokens"] == 55_800
    assert record["priced_new_tokens"] == 150.0

    # AC4: no progress chunk reached the client.
    assert first not in received
    assert later not in received
    assert _CONTENT_CHUNK in received


@pytest.mark.asyncio
async def test_missing_dispatch_record_still_exposes_measurement():
    """AC3: capture does not depend on the (optional) lease record existing."""
    _received, _sent, _srv, _record = await _drive_local_stream(
        _stream_body(),
        [_progress_event(1_000, 100), _CONTENT_CHUNK, _FINISH_CHUNK, _DONE],
        preload_dispatch_record=False,
    )
    measured = get_measured_prefill(MODEL, SESSION)
    assert measured is not None
    assert measured.measured_new_tokens == 900


# ---------------------------------------------------------------------------
# AC4 — client-visible stream is byte-for-byte unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_stream_is_unchanged_when_progress_is_injected():
    """AC4: progress events are internal — the client sees the same bytes."""
    baseline, _sent, _srv, _record = await _drive_local_stream(
        _stream_body(), [_CONTENT_CHUNK, _FINISH_CHUNK, _DONE]
    )

    measured_prefill.reset_measured_prefill()
    with_progress, _sent2, _srv2, _record2 = await _drive_local_stream(
        _stream_body(),
        [
            _progress_event(56_000, 200, processed=0, time_ms=0),
            _CONTENT_CHUNK,
            _progress_event(56_000, 54_000, processed=2_048, time_ms=120),
            _FINISH_CHUNK,
            _DONE,
        ],
    )

    assert with_progress == baseline
    assert b"prompt_progress" not in with_progress


@pytest.mark.asyncio
async def test_client_requested_progress_is_forwarded_unchanged():
    """A client that asked for ``return_progress`` keeps its own events."""
    body = _stream_body()
    body["return_progress"] = True
    progress = _progress_event(56_000, 200)
    received, sent_body, _srv, _record = await _drive_local_stream(
        body, [progress, _CONTENT_CHUNK, _FINISH_CHUNK, _DONE]
    )

    assert sent_body["return_progress"] is True
    assert progress in received
    # Capture still observes the client's events.
    assert get_measured_prefill(MODEL, SESSION) is not None


# ---------------------------------------------------------------------------
# AC5 — fail-open
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_without_progress_events_is_unchanged():
    """AC5: no measurement and no log line, dispatch behaviour as today."""
    received, sent_body, srv, _record = await _drive_local_stream(
        _stream_body(), [_CONTENT_CHUNK, _FINISH_CHUNK, _DONE]
    )

    assert sent_body["return_progress"] is True
    assert get_measured_prefill(MODEL, SESSION) is None
    assert _measured_log_lines(srv.logger) == []
    assert received == b"".join([_CONTENT_CHUNK, _FINISH_CHUNK, _DONE])


@pytest.mark.asyncio
async def test_malformed_progress_event_does_not_break_the_stream():
    """AC5: a malformed event is forwarded untouched; no measurement."""
    malformed = b'data: {"choices":[],"prompt_progress":"not-an-object"}\n\n'
    received, _sent, srv, _record = await _drive_local_stream(
        _stream_body(), [malformed, _CONTENT_CHUNK, _FINISH_CHUNK, _DONE]
    )

    assert get_measured_prefill(MODEL, SESSION) is None
    assert _measured_log_lines(srv.logger) == []
    assert malformed in received
    assert _FINISH_CHUNK in received


@pytest.mark.asyncio
async def test_non_numeric_progress_fields_tolerate_to_unknown():
    """AC5: wrong-typed fields never raise; the line reports ``unknown``."""
    odd = b'data: {"prompt_progress":{"total":"x","cache":null}}\n\n'
    received, _sent, srv, _record = await _drive_local_stream(
        _stream_body(), [odd, _CONTENT_CHUNK, _FINISH_CHUNK, _DONE]
    )

    measured = get_measured_prefill(MODEL, SESSION)
    assert measured is not None
    assert measured.total is None and measured.cache is None
    assert measured.measured_cached_ratio is None
    lines = _measured_log_lines(srv.logger)
    assert len(lines) == 1
    assert "measured_cached_ratio=unknown" in lines[0]
    assert _FINISH_CHUNK in received


@pytest.mark.asyncio
async def test_capture_exception_never_escapes_the_stream_loop():
    """AC5: an exception in capture leaves the dispatch behaviour intact."""
    with patch.object(
        MeasuredPrefill, "from_progress", side_effect=RuntimeError("boom")
    ):
        received, _sent, srv, _record = await _drive_local_stream(
            _stream_body(),
            [_progress_event(56_000, 200), _CONTENT_CHUNK, _FINISH_CHUNK, _DONE],
        )

    assert get_measured_prefill(MODEL, SESSION) is None
    assert _measured_log_lines(srv.logger) == []
    # The stream still terminates with the synthetic markers.
    assert _FINISH_CHUNK in received
    assert _DONE in received


# ---------------------------------------------------------------------------
# AC6 — observability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prefill_progress_measured_line_carries_attributable_fields():
    """AC6: one structured line per dispatch with the measured/priced numbers."""
    received, _sent, srv, _record = await _drive_local_stream(
        _stream_body(),
        [
            _progress_event(56_000, 200, processed=0, time_ms=0),
            _CONTENT_CHUNK,
            _FINISH_CHUNK,
            _DONE,
        ],
    )

    lines = _measured_log_lines(srv.logger)
    assert len(lines) == 1, lines
    line = lines[0]
    assert f"session={SESSION}" in line
    assert f"model={MODEL}" in line
    fields = dict(re.findall(r"(\w+)=(\S+)", line))
    assert fields["total"] == "56000"
    assert fields["cache"] == "200"
    assert fields["processed"] == "0"
    assert fields["measured_cached_ratio"] == "0.0036"
    assert fields["priced_new_tokens"] == "150"
    assert _FINISH_CHUNK in received


# ---------------------------------------------------------------------------
# AC1 — non-local providers never receive the key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remote_streaming_forward_body_is_untouched():
    """AC1: the remote path forwards the client body verbatim."""
    from proxy.proxy_remote import _handle_remote_streaming

    body = json.dumps(
        {
            "model": "remote-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
    ).encode()

    sse = [_CONTENT_CHUNK, _FINISH_CHUNK, _DONE]
    response = MagicMock()
    response.status_code = 200
    response.headers = {"content-type": "text/event-stream"}

    async def _aiter():
        for chunk in sse:
            yield chunk

    response.aiter_bytes = MagicMock(return_value=_aiter())
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    client = MagicMock()
    client.stream = MagicMock(return_value=cm)
    client.aclose = AsyncMock()

    request = MagicMock()
    request.method = "POST"
    request.url.path = "/v1/chat/completions"
    request.is_disconnected = AsyncMock(return_value=False)

    with ExitStack() as stack:
        stack.enter_context(
            patch("proxy.proxy_remote.httpx.AsyncClient", MagicMock(return_value=client))
        )
        stack.enter_context(patch("proxy.proxy_remote._schedule_traffic_recording"))
        stack.enter_context(
            patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock())
        )
        stack.enter_context(patch("proxy.proxy_remote.log_response_chunk"))
        stack.enter_context(patch("proxy.proxy_remote.log_response"))
        stack.enter_context(patch("proxy.proxy_remote.log_request"))
        result = await _handle_remote_streaming(
            request=request,
            target_url="https://api.example.com/v1/chat/completions",
            headers={"Authorization": "Bearer test"},
            body=body,
            body_json=json.loads(body),
            model_name="remote-model",
            remote_timeout=httpx.Timeout(60.0),
            session_id="sess-remote-1",
        )
        if isinstance(result, StreamingResponse):
            async for _chunk in result.body_iterator:
                pass

    forwarded = client.stream.call_args.kwargs.get("content")
    assert forwarded == body
