"""Early abort on measured prefill cache deficit — router wiring (F4).

LP-0MVR7IN0O000DPXN / parent LP-0MVQIT9QY007QU27 (producer-approved Option D).

F3 (``proxy.measured_cache_abort``) supplies the pure decision, F1/F2
(``proxy.prompt_progress`` / ``proxy.measured_prefill``) supply the measured
``prompt_progress`` capture. This item wires the decision into
``proxy.router.proxy_to_local``'s streaming dispatch:

* the abort is decided **before** the response is committed, by peeking the
  SSE stream for the first ``prompt_progress`` measurement (once headers have
  been sent a mid-stream failure can no longer detour to a remote provider);
* a firing decision releases the accepted dispatch in seconds, logs
  ``local_eviction_after_acceptance reason=measured_cache_deficit`` with the
  measured evidence and returns the same retryable 503 the 120 s deadline
  returns, so the provider chain detours to a remote provider;
* every failure mode (no progress, malformed event, capture/decision
  exception, read error, disabled feature) degrades to today's behaviour.

AC mapping:
  AC1 early release (seconds, not the 120 s deadline) — integration test over
      a replayed fake-warm SSE stream;
  AC2 warm stream untouched (``cache ≈ 0.999 × total``), client bytes
      unchanged;
  AC3 the early release falls through to the remote failover path;
  AC4 the rollback key restores the exact pre-change behaviour;
  AC5 fail open on missing/malformed/erroring measurement;
  AC6 the enriched eviction snapshot carries the measured evidence, and the
      120 s deadline eviction is not double-fired for the same dispatch;
  AC7 lease/marker/counter release and the prefill-thrash cooldown.

Harness: mirrors the established ``proxy_to_local`` integration harness
(``test_measured_prefill_capture.py``) and drives the real lease bookkeeping
(``test_lease_truthfulness.py``) so the release path is exercised end to end
rather than asserted from source.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import proxy.router as router
import proxy.router_helpers as helpers
import proxy.server as server
import pytest
from fastapi.responses import JSONResponse, StreamingResponse

MODEL = "Qwen3"
SESSION = "sess-measured-cache-abort-1"
PRICED_NEW_TOKENS = 150
PROMPT_TOKENS = 56_000

SERVER_CONFIG = {
    "server": {
        "llama_router_mode": False,
        "llama_server_port": 8080,
        "max_concurrent_queries": 4,
        "session_slot_pool_size": 1,
        "local_dispatch_lease_timeout_seconds": 30,
        "local_dispatch_lease_per_token_seconds": 0.0,
        "local_dispatch_lease_max_seconds": 1500,
        "local_dispatch_lease_prefill_poll_seconds": 0.05,
        "local_dispatch_lease_prefill_buffer_seconds": 0.5,
        "local_dispatch_no_progress_timeout_seconds": 180,
        "local_dispatch_max_prefill_seconds": 900,
        # The feature must beat this by a wide margin (AC1).
        "local_prefill_deadline_seconds": 120,
        "local_dispatch_first_byte_timeout_seconds": 840,
        "local_dispatch_prefill_observability_warn_seconds": 0,
        "session_single_flight_mode": "bypass",
        "disconnect_cleanup_timeout": 1,
        "stream_heartbeat_interval_seconds": 0.05,
        "stream_idle_timeout_seconds": 0.3,
        "local_max_concurrent_prefills": 1,
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


# ---------------------------------------------------------------------------
# b8782 stream fixtures (the shapes verified live for F1)
# ---------------------------------------------------------------------------

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
    """One b8782 ``prompt_progress`` chunk (top-level key, integer fields)."""
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


def _fake_warm_progress() -> bytes:
    """The measured fake-warm signature: 56k prompt, 200-token reused prefix."""
    return _progress_event(PROMPT_TOKENS, 200)


def _warm_progress() -> bytes:
    """A genuinely warm dispatch: ``cache ≈ 0.999 × total``."""
    return _progress_event(PROMPT_TOKENS, int(PROMPT_TOKENS * 0.999))


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _StreamResponse:
    """Minimal ``httpx.Response`` stand-in whose body is a chunk list."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        raise_after: int | None = None,
        hang_after: int | None = None,
    ):
        self.status_code = 200
        self.headers = {"content-type": "text/event-stream"}
        self.content = b""
        self._chunks = chunks
        self._raise_after = raise_after
        self._hang_after = hang_after

    async def aiter_bytes(self):
        for index, chunk in enumerate(self._chunks):
            if self._raise_after is not None and index >= self._raise_after:
                raise RuntimeError("upstream read failed")
            yield chunk
        if self._hang_after is not None:
            await asyncio.sleep(30)

    async def aread(self):
        return b""


class _CM:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _Request:
    """Minimal FastAPI ``Request`` stand-in (client body + disconnect probe)."""

    method = "POST"
    headers = {"host": "localhost", "x-session-id": SESSION}
    url = type("U", (), {"path": "/v1/chat/completions"})()
    _local_admission_evidence = {
        "estimated_tokens": PROMPT_TOKENS,
        "cached_ratio": 0.0,
        "new_tokens": PRICED_NEW_TOKENS,
    }

    def __init__(self, body: dict):
        self._body = json.dumps(body).encode()

    async def body(self):
        return self._body

    async def is_disconnected(self):
        return False


def _install_server(monkeypatch, server_config_extra: dict | None = None) -> None:
    """Install the ``proxy.server`` state the real lease bookkeeping touches."""
    config = json.loads(json.dumps(SERVER_CONFIG))
    config["server"].update(server_config_extra or {})
    monkeypatch.setattr(server, "config", config)
    monkeypatch.setattr(server, "active_queries", 0)
    monkeypatch.setattr(server, "active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(server, "local_active_queries", 0)
    monkeypatch.setattr(server, "local_active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(server, "local_generating_queries", 0)
    monkeypatch.setattr(server, "local_generating_queries_lock", asyncio.Lock())
    monkeypatch.setattr(server, "local_generating_sessions", set())
    monkeypatch.setattr(server, "local_prefill_in_flight", {})
    monkeypatch.setattr(server, "local_prefill_in_flight_lock", asyncio.Lock())
    monkeypatch.setattr(server, "local_dispatch_records", {})
    monkeypatch.setattr(server, "local_dispatch_records_lock", asyncio.Lock())
    monkeypatch.setattr(server, "per_model_queries", {})
    monkeypatch.setattr(server, "per_model_queries_lock", asyncio.Lock())
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "llama_process", MagicMock(poll=lambda: None, pid=1))
    monkeypatch.setattr(server, "current_model", MODEL)
    monkeypatch.setattr(server, "session_manager", MagicMock())
    monkeypatch.setattr(server, "logger", MagicMock())


def _patch_router(monkeypatch, response: _StreamResponse, *, sent_body_sink: list):
    """Patch everything around the dispatch so the real router code runs."""

    client = MagicMock()
    client.stream = MagicMock(return_value=_CM(response))
    client.aclose = AsyncMock()

    def _capture_stream(*_args, **kwargs):
        try:
            sent_body_sink.append(json.loads(kwargs.get("content") or b"{}"))
        except Exception:
            sent_body_sink.append({})
        return _CM(response)

    client.stream = MagicMock(side_effect=_capture_stream)

    async def _session(*args, **_kwargs):
        body_json = args[1] if len(args) > 1 else {}
        return {
            "session_id": SESSION,
            "session_id_header": SESSION,
            "session_created": False,
            "is_delta_request": False,
            "session_fallback_reason": None,
            "delta_messages": [],
            "body_override": None,
            "body_json": body_json,
            "original_message_count": 1,
            "session_explicit": True,
        }

    async def _retry(factory, **_kwargs):
        return await factory()

    monkeypatch.setattr(router, "_is_self_healing_active", lambda: False)
    monkeypatch.setattr(router, "_check_slot_availability", AsyncMock(return_value=None))
    monkeypatch.setattr(router, "_handle_session", _session)
    monkeypatch.setattr(router, "_build_slot_context", MagicMock(return_value=(None, None, 3.0)))
    monkeypatch.setattr(router.httpx, "AsyncClient", MagicMock(return_value=client))
    monkeypatch.setattr(router, "_schedule_token_increment", AsyncMock())
    monkeypatch.setattr(router, "_schedule_recv_token_increment", AsyncMock())
    monkeypatch.setattr(router, "_schedule_traffic_recording", MagicMock())
    monkeypatch.setattr(router, "count_text_tokens", MagicMock(return_value=5))
    monkeypatch.setattr(router, "_extract_delta_text_from_sse_chunk", MagicMock(return_value=""))
    monkeypatch.setattr(router, "evaluate_stream_guardrail", MagicMock(return_value=None))
    monkeypatch.setattr(router, "log_request", MagicMock())
    monkeypatch.setattr(router, "log_response_chunk", MagicMock())
    monkeypatch.setattr(router, "_call_with_backend_retries", _retry)
    monkeypatch.setattr(router, "_update_session_and_slot", AsyncMock())
    monkeypatch.setattr(router, "_restore_slot_snapshot", AsyncMock(return_value=False))
    monkeypatch.setattr(
        router, "_detect_restore_signal_from_llama_log", MagicMock(return_value=False)
    )
    monkeypatch.setattr("proxy.session._resolve_log_path", MagicMock(return_value=Path("/nonexistent")))
    monkeypatch.setattr(
        "proxy.provider._estimate_effective_prompt_tokens_for_routing",
        AsyncMock(return_value=PROMPT_TOKENS),
    )
    return client


async def _drive_local(
    chunks: list[bytes],
    *,
    monkeypatch,
    server_config_extra=None,
    body_extra=None,
    raise_after=None,
    hang_after=None,
):
    """Run one local streaming dispatch through the real ``proxy_to_local``.

    Returns ``(response, received_bytes, sent_upstream_body)``.
    """
    helpers._reset_prefill_thrash_cooldowns()
    _install_server(monkeypatch, server_config_extra)
    sent_body: list[dict] = []
    response_obj = _StreamResponse(
        chunks, raise_after=raise_after, hang_after=hang_after
    )
    _patch_router(monkeypatch, response_obj, sent_body_sink=sent_body)

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    body.update(body_extra or {})

    result = await router.proxy_to_local(_Request(body), "v1/chat/completions")

    received = b""
    if isinstance(result, StreamingResponse):
        async for chunk in result.body_iterator:
            received += chunk
    return result, received, (sent_body[-1] if sent_body else {})


def _log_lines(logger: MagicMock, level: str) -> list[str]:
    """Render ``logger.<level>`` calls back into their formatted lines."""
    lines: list[str] = []
    for call in getattr(logger, level).call_args_list:
        template = str(call.args[0]) if call.args else ""
        args = call.args[1:]
        if args and "%" in template:
            try:
                lines.append(template % args)
                continue
            except (TypeError, ValueError):
                pass
        lines.append(template + (" " + " ".join(str(a) for a in args) if args else ""))
    return lines


def _all_log_lines(level: str) -> list[str]:
    return _log_lines(server.logger, level)


# ---------------------------------------------------------------------------
# AC1 / AC6 / AC7 — early release on a measured cache deficit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_warm_progress_releases_the_dispatch_early(monkeypatch):
    """AC1/AC6/AC7: fake-warm measurement ⇒ early release + remote detour.

    The dispatch is released with ``reason=measured_cache_deficit`` in
    seconds (the 120 s deadline never fires), the enriched eviction snapshot
    carries the measured evidence, and the lease/marker/counters are released
    exactly like the deadline path.
    """
    started = time.monotonic()
    result, received, _sent = await _drive_local(
        [_fake_warm_progress(), _CONTENT_CHUNK, _FINISH_CHUNK, _DONE],
        monkeypatch=monkeypatch,
    )
    elapsed = time.monotonic() - started

    # AC1: released with the retryable 503 the provider chain detours on,
    # long before the 120 s deadline (and without waiting for the stream).
    assert isinstance(result, JSONResponse), (
        "an early abort must return the retryable error response, not a "
        f"committed stream (got {type(result).__name__})"
    )
    assert result.status_code == 503
    assert received == b""
    assert elapsed < 10, f"expected a seconds-scale release, took {elapsed:.1f}s"

    # AC1/AC6: attributable release events.
    infos = _all_log_lines("info")
    warnings = _all_log_lines("warning")
    errors = _all_log_lines("error")
    assert any(
        "lease_released" in line and "reason=measured_cache_deficit" in line
        for line in infos
    ), f"expected lease_released reason=measured_cache_deficit, got {infos[-8:]}"
    assert any(
        "measured_cache_abort" in line and "session=" + SESSION in line
        for line in warnings
    ), f"expected the measured_cache_abort detour line, got {warnings[-8:]}"
    # AC6: the measurement is still logged for the aborted dispatch.
    assert any(
        line.startswith("prefill_progress_measured") and f"session={SESSION}" in line
        for line in infos
    ), f"expected prefill_progress_measured, got {infos[-8:]}"

    eviction = [line for line in errors if "local_eviction_after_acceptance" in line]
    assert len(eviction) == 1, eviction
    line = eviction[0]
    assert f"session={SESSION}" in line
    assert "reason=measured_cache_deficit" in line
    assert "measured_cache=200" in line
    assert f"total={PROMPT_TOKENS}" in line
    assert f"priced_new_tokens={PRICED_NEW_TOKENS}" in line
    assert "elapsed_s=" in line
    # AC6: the 120 s deadline eviction is never double-fired for this dispatch.
    assert not any("reason=prefill_deadline" in ln for ln in errors), eviction
    assert not any("prefill_deadline_exceeded" in ln for ln in warnings)

    # AC7: no leaked lease, no leaked in-flight marker, counters decremented.
    assert server.local_dispatch_records == {}
    assert SESSION not in server.local_prefill_in_flight
    assert server.local_active_queries == 0
    assert server.active_queries == 0

    # AC7: the cooldown machinery still applies to the session.
    assert helpers._prefill_thrash_cooldown_remaining(SESSION, server.config) > 0

    # AC1: the measurement landed on the dispatch record before release.
    assert any(
        "prefill_progress_measured" in line and "cache=200" in line
        for line in infos
    )


# ---------------------------------------------------------------------------
# AC2 — warm path untouched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warm_progress_never_trips_the_abort(monkeypatch):
    """AC2: ``cache ≈ 0.999 × total`` streams the full normal response."""
    chunks = [_warm_progress(), _CONTENT_CHUNK, _FINISH_CHUNK, _DONE]
    result, received, _sent = await _drive_local(chunks, monkeypatch=monkeypatch)

    assert isinstance(result, StreamingResponse)
    assert result.status_code == 200
    # Progress chunks are proxy-internal: the client sees the same bytes as a
    # dispatch that produced no progress events at all.
    assert received == _CONTENT_CHUNK + _FINISH_CHUNK + _DONE
    assert b"prompt_progress" not in received

    assert not any(
        "measured_cache_deficit" in line for line in _all_log_lines("error")
    )
    assert not any(
        "measured_cache_abort" in line for line in _all_log_lines("warning")
    )
    # The warm path still reports its measurement (capture is unchanged).
    assert any(
        line.startswith("prefill_progress_measured")
        for line in _all_log_lines("info")
    )
    assert server.local_dispatch_records, "a served dispatch keeps its lease record"


@pytest.mark.asyncio
async def test_client_requested_progress_is_replayed_unchanged(monkeypatch):
    """AC2: a client that asked for ``return_progress`` keeps its own events."""
    chunks = [_warm_progress(), _CONTENT_CHUNK, _FINISH_CHUNK, _DONE]
    result, received, sent = await _drive_local(
        chunks, monkeypatch=monkeypatch, body_extra={"return_progress": True}
    )

    assert sent.get("return_progress") is True
    assert _warm_progress() in received
    assert received == b"".join(chunks)


# ---------------------------------------------------------------------------
# AC4 — rollback knob
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rollback_key_restores_pre_change_behaviour(monkeypatch):
    """AC4: ``0`` disables injection, capture and the early release."""
    chunks = [_fake_warm_progress(), _CONTENT_CHUNK, _FINISH_CHUNK, _DONE]
    result, received, sent = await _drive_local(
        chunks,
        monkeypatch=monkeypatch,
        server_config_extra={"local_measured_cache_abort_enabled": 0},
    )

    assert "return_progress" not in sent
    assert isinstance(result, StreamingResponse)
    # Pre-change behaviour: the progress chunk is just upstream data and the
    # dispatch is never released early.
    assert received == b"".join(chunks)
    assert not any(
        "measured_cache_deficit" in line for line in _all_log_lines("error")
    )
    assert not any(
        line.startswith("prefill_progress_measured")
        for line in _all_log_lines("info")
    )


# ---------------------------------------------------------------------------
# AC5 — fail open
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_progress_fails_open(monkeypatch):
    """AC5: no ``prompt_progress`` at all ⇒ today's behaviour."""
    chunks = [_CONTENT_CHUNK, _FINISH_CHUNK, _DONE]
    result, received, sent = await _drive_local(chunks, monkeypatch=monkeypatch)

    assert sent.get("return_progress") is True
    assert isinstance(result, StreamingResponse)
    assert received == b"".join(chunks)
    assert not any("measured_cache_deficit" in ln for ln in _all_log_lines("error"))


@pytest.mark.asyncio
async def test_malformed_progress_fails_open(monkeypatch):
    """AC5: unparseable measurement ⇒ continue, bytes untouched."""
    malformed = b'data: {"choices":[],"prompt_progress":"not-an-object"}\n\n'
    chunks = [malformed, _CONTENT_CHUNK, _FINISH_CHUNK, _DONE]
    result, received, _sent = await _drive_local(chunks, monkeypatch=monkeypatch)

    assert isinstance(result, StreamingResponse)
    assert received == b"".join(chunks)
    assert not any("measured_cache_deficit" in ln for ln in _all_log_lines("error"))


@pytest.mark.asyncio
async def test_capture_exception_fails_open(monkeypatch):
    """AC5: an exception in the capture path never aborts the dispatch."""
    from proxy.measured_prefill import MeasuredPrefill

    chunks = [_fake_warm_progress(), _CONTENT_CHUNK, _FINISH_CHUNK, _DONE]
    with patch.object(MeasuredPrefill, "from_progress", side_effect=RuntimeError("boom")):
        result, received, _sent = await _drive_local(chunks, monkeypatch=monkeypatch)

    assert isinstance(result, StreamingResponse)
    assert _FINISH_CHUNK in received and _DONE in received
    assert not any("measured_cache_deficit" in ln for ln in _all_log_lines("error"))


@pytest.mark.asyncio
async def test_read_error_during_peek_fails_open(monkeypatch):
    """AC5: a read error while peeking keeps the bytes and reports the error."""
    chunks = [b'data: {"choices":[],"pro', _CONTENT_CHUNK, _FINISH_CHUNK]
    result, received, _sent = await _drive_local(
        chunks, monkeypatch=monkeypatch, raise_after=2
    )

    assert isinstance(result, StreamingResponse)
    # The complete content chunk read before the error still reaches the
    # client, and the stream terminates with the synthetic error markers.
    assert _CONTENT_CHUNK in received
    assert b"[DONE]" in received
    assert not any("measured_cache_deficit" in ln for ln in _all_log_lines("error"))


@pytest.mark.asyncio
async def test_prefill_deadline_still_fires_while_peeking(monkeypatch):
    """A warm measurement then silence still detours at the prefill deadline.

    With ``return_progress`` injected the SSE headers arrive at prefill
    start, so the peek — not the first-byte monitor — is where
    ``local_prefill_deadline_seconds`` now bounds the wait. Losing that bound
    would turn a stalled prefill into a multi-minute client hang instead of
    today's remote detour.
    """
    result, received, _sent = await _drive_local(
        [_warm_progress()],
        monkeypatch=monkeypatch,
        server_config_extra={"local_prefill_deadline_seconds": 0.2},
        hang_after=1,
    )

    assert isinstance(result, JSONResponse), (
        "the deadline must still detour with a retryable error response"
    )
    assert result.status_code == 503
    assert received == b""

    warnings = _all_log_lines("warning")
    errors = _all_log_lines("error")
    assert any("prefill_deadline_exceeded" in line for line in warnings), warnings
    assert any(
        "reason=prefill_deadline" in line and "local_eviction_after_acceptance" in line
        for line in errors
    ), errors
    assert not any("measured_cache_deficit" in line for line in errors)
    assert server.local_dispatch_records == {}
    assert SESSION not in server.local_prefill_in_flight


# ---------------------------------------------------------------------------
# AC3 — the early release detours to the remote failover path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_early_release_detours_to_remote_provider(monkeypatch):
    """AC3: after the early release the client is served by a remote provider."""
    import proxy.provider as provider

    helpers._reset_prefill_thrash_cooldowns()
    _install_server(monkeypatch)

    local_calls: list[str] = []
    remote_calls: list[str] = []

    async def _local_via_router(request, path, endpoint=None):
        local_calls.append(path)
        sent_body: list[dict] = []
        _patch_router(
            monkeypatch,
            _StreamResponse(
                [_fake_warm_progress(), _CONTENT_CHUNK, _FINISH_CHUNK, _DONE]
            ),
            sent_body_sink=sent_body,
        )
        return await router.proxy_to_local(request, path)

    async def _remote(request, path, provider_cfg):
        remote_calls.append(provider_cfg.get("name"))
        return StreamingResponse(
            iter([_CONTENT_CHUNK, _FINISH_CHUNK, _DONE]),
            media_type="text/event-stream",
            status_code=200,
        )

    model_cfg = {
        "name": MODEL,
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": MODEL},
            {
                "name": "remote-fallback",
                "type": "remote",
                "endpoint": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
        ],
    }

    monkeypatch.setattr(provider, "_get_local_concurrency_info", lambda *a, **k: (0, 1))
    monkeypatch.setattr(server, "proxy_to_local", _local_via_router)
    monkeypatch.setattr(server, "proxy_to_remote", _remote)

    body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True}
    result = await provider.proxy_with_fallback(
        _Request(body), "v1/chat/completions", model_cfg, server.config
    )

    received = b""
    if isinstance(result, StreamingResponse):
        async for chunk in result.body_iterator:
            received += chunk

    assert local_calls == ["v1/chat/completions"], "local was attempted first"
    assert remote_calls == ["remote-fallback"], (
        "the early release must fall through to the remote provider"
    )
    assert _FINISH_CHUNK in received and _DONE in received, (
        "the client must receive a completed response"
    )
    assert any(
        "reason=measured_cache_deficit" in line
        for line in _all_log_lines("error")
    )


# ---------------------------------------------------------------------------
# Unit: the peek helper + the release helper
# ---------------------------------------------------------------------------


class _FakeAiter:
    """Async iterator over pre-baked chunks (tracks consumption)."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)
        self.consumed = 0
        self._iterator = self._generate()

    async def _generate(self):
        for chunk in self._chunks:
            self.consumed += 1
            yield chunk

    def __aiter__(self):
        return self

    def __anext__(self):
        return self._iterator.__anext__()


@pytest.mark.asyncio
async def test_prefetch_decides_once_and_replays_every_consumed_chunk():
    """The peek stops at the first client-visible event, losing no bytes.

    A non-aborting progress event does NOT end the peek: it keeps reading so
    ``local_prefill_deadline_seconds`` still bounds the pre-first-byte wait
    (see ``test_prefill_deadline_still_fires_while_peeking``).
    """
    from proxy.measured_cache_abort import AbortSettings
    from proxy.utils import SSEEventReframer

    chunks = [_warm_progress(), _CONTENT_CHUNK, _FINISH_CHUNK]
    reframer = SSEEventReframer()
    prefetch = await helpers._prefetch_measured_cache_abort(
        _FakeAiter(chunks),
        reframer,
        settings=AbortSettings(enabled=True, bound_tokens=2_000, min_deficit_ratio=0.25),
        priced_new_tokens=PRICED_NEW_TOKENS,
        timeout=1.0,
    )

    assert prefetch.abort is False
    assert prefetch.complete is False
    # Both the progress read and the first client-visible read are replayed.
    assert prefetch.chunks == [chunks[0], chunks[1]]
    assert prefetch.measured is not None
    assert prefetch.measured.cache == int(PROMPT_TOKENS * 0.999)


@pytest.mark.asyncio
async def test_prefetch_reports_the_abort_decision():
    """A qualifying fake-warm measurement returns the F3 decision."""
    from proxy.measured_cache_abort import (
        MEASURED_CACHE_DEFICIT_REASON,
        AbortSettings,
    )
    from proxy.utils import SSEEventReframer

    chunks = [_fake_warm_progress()]
    prefetch = await helpers._prefetch_measured_cache_abort(
        _FakeAiter(chunks),
        SSEEventReframer(),
        settings=AbortSettings(),
        priced_new_tokens=PRICED_NEW_TOKENS,
        timeout=1.0,
    )

    assert prefetch.abort is True
    assert prefetch.decision.reason == MEASURED_CACHE_DEFICIT_REASON
    assert prefetch.measured.total == PROMPT_TOKENS
    assert prefetch.measured.cache == 200


@pytest.mark.asyncio
async def test_prefetch_fails_open_when_disabled():
    """A disabled feature never peeks (no bytes consumed)."""
    from proxy.measured_cache_abort import AbortSettings
    from proxy.utils import SSEEventReframer

    aiter = _FakeAiter([_fake_warm_progress()])
    prefetch = await helpers._prefetch_measured_cache_abort(
        aiter,
        SSEEventReframer(),
        settings=AbortSettings(enabled=False),
        priced_new_tokens=PRICED_NEW_TOKENS,
        timeout=1.0,
    )

    assert prefetch.abort is False
    assert prefetch.chunks == []
    assert aiter.consumed == 0


@pytest.mark.asyncio
async def test_prefetch_raises_timeout_when_nothing_arrives():
    """The prefill deadline still bounds the pre-first-byte wait."""
    from proxy.measured_cache_abort import AbortSettings
    from proxy.utils import SSEEventReframer

    class _SilentAiter:
        async def __anext__(self):
            await asyncio.sleep(30)

    with pytest.raises(TimeoutError):
        await helpers._prefetch_measured_cache_abort(
            _SilentAiter(),
            SSEEventReframer(),
            settings=AbortSettings(),
            priced_new_tokens=PRICED_NEW_TOKENS,
            timeout=0.05,
        )


@pytest.mark.asyncio
async def test_release_helper_is_idempotent(monkeypatch):
    """AC6: the eviction snapshot is emitted once, never double-fired."""
    _install_server(monkeypatch)
    server.local_dispatch_records[SESSION] = {
        "active": True,
        "started_at": time.monotonic(),
        "first_byte_seen": False,
    }
    server.local_prefill_in_flight[SESSION] = True

    from proxy.measured_prefill import MeasuredPrefill

    measured = MeasuredPrefill(
        total=PROMPT_TOKENS,
        cache=200,
        processed=0,
        time_ms=0.0,
        priced_new_tokens=float(PRICED_NEW_TOKENS),
    )

    assert await helpers._release_measured_cache_abort_dispatch(
        server, SESSION, measured=measured, priced_new_tokens=PRICED_NEW_TOKENS
    )
    assert await helpers._release_measured_cache_abort_dispatch(
        server, SESSION, measured=measured, priced_new_tokens=PRICED_NEW_TOKENS
    ) is False

    evictions = [
        line
        for line in _all_log_lines("error")
        if "local_eviction_after_acceptance" in line
    ]
    assert len(evictions) == 1, evictions
    assert "reason=measured_cache_deficit" in evictions[0]
    assert SESSION not in server.local_prefill_in_flight
