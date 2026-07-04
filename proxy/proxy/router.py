"""
Router Module

Core proxy routing function (proxy_to_local) for routing requests to the
local llama-server with session-based incremental ingestion.

Helper functions and remote-proxying have been moved to router_helpers.py
and proxy_remote.py respectively to keep individual modules focused.

Functions in this module:
    - proxy_to_local: Route to local llama-server with session handling
"""

import asyncio
import json
import time
from typing import Optional

import httpx
from fastapi import HTTPException, Request, Response
from fastapi.responses import StreamingResponse

# Lazy server import — avoids circular imports when server.py imports us
def _srv():
    import proxy.server as _m
    return _m

# Imports from sibling extracted modules
from proxy.lifecycle import (  # noqa: E402
    _is_self_healing_active,
    _self_healing_response,
    _resolve_slot_model_name,
    get_model_config,
)
from proxy.session import (  # noqa: E402
    SessionSingleFlightRejected,
    _build_slot_context,
    _detect_restore_signal_from_llama_log,
    _detect_restore_signal_from_log_slice,
    _has_explicit_restore_signal,
    _restore_slot_snapshot,
    _save_slot_snapshot,
    _record_guardrail_cutoff,
    _record_restore_success,
    evaluate_stream_guardrail,
    _should_invalidate_on_guardrail,
    merge_session_history_for_update,
    session_single_flight_coordinator,
    slot_lock_coordinator,
)
from proxy.observability import (  # noqa: E402
    _record_backend_signal,
)
import proxy.metrics as metrics  # noqa: E402
# legacy alias for convenience
record_http_error = metrics.record_http_error

from proxy.utils import (  # noqa: E402
    _extract_assistant_content_from_sse,
    _extract_delta_text_from_sse_chunk,
    count_text_tokens,
)

from proxy.slot_scheduler import JobScheduler, AdmitResult  # noqa: E402

# Imports from sibling router helpers
from .router_helpers import (  # noqa: E402
    _build_backend_error_response,
    _build_backend_unavailable_response,
    _check_slot_availability,
    _compute_request_timeout,
    _decrement_active_queries,
    _decrement_local_active_queries,
    _estimate_tokens_sent,
    _handle_session,
    _increment_active_queries,
    _increment_local_active_queries,
    _normalize_outgoing_headers,
    _schedule_recv_token_increment,
    _schedule_token_increment,
    _call_with_backend_retries,
    _call_with_empty_retry,
    normalize_upstream_request_headers,
    log_request,
    log_response,
    log_response_chunk,
)


# ===================================================================
# Job-level slot scheduler (global, lazy-initialized from config)
# ===================================================================

_job_scheduler: Optional[JobScheduler] = None
_job_scheduler_initialized: bool = False


def _get_job_scheduler() -> Optional[JobScheduler]:
    """
    Return the global JobScheduler, initialising it from config on first call.
    Returns None if slot management is not configured.
    """
    global _job_scheduler, _job_scheduler_initialized
    if _job_scheduler_initialized:
        return _job_scheduler

    _job_scheduler_initialized = True
    srv = _srv()
    slot_config = srv.config.get("slot_management", {})
    if not slot_config:
        return None

    pool_size = int(slot_config.get("slot_pool_size", 0) or 0)
    if pool_size < 1:
        return None

    _job_scheduler = JobScheduler(
        pool_size=pool_size,
        max_queue_depth=int(slot_config.get("slot_queue_max_depth", 16) or 16),
        job_timeout=float(slot_config.get("slot_job_timeout_seconds", 300.0) or 300.0),
        queue_overflow_retry_after=float(
            slot_config.get("slot_queue_overflow_retry_after", 900) or 900
        ),
    )
    # Wire scheduler into SlotLockCoordinator
    slot_lock_coordinator.set_scheduler(_job_scheduler)
    return _job_scheduler


def _scheduler_has_idle_slot() -> bool:
    """
    Check whether the JobScheduler has at least one idle slot.

    Returns ``True`` if there is an idle slot (the request can be served),
    ``False`` if all slots are busy (the request would be queued).
    Returns ``True`` when no scheduler is active (no slot management).
    """
    scheduler = _get_job_scheduler()
    if scheduler is None:
        return True
    return scheduler.has_idle_slot()


def _get_local_max_concurrent_queries(server_config: dict) -> int:
    """
    Read the local-model concurrency limit from config.

    Returns the configured ``local_max_concurrent_queries`` value
    (default 1). This limit is separate from the global
    ``max_concurrent_queries`` which applies to remote providers.
    """
    try:
        val = server_config.get("local_max_concurrent_queries", 1)
        return max(1, int(val or 1))
    except (ValueError, TypeError):
        return 1


def _get_local_active_count(srv) -> int:
    """
    Get the current number of active local requests.

    Returns the count stored on the server for local-provider requests.
    """
    try:
        return int(getattr(srv, 'local_active_queries', 0) or 0)
    except (ValueError, TypeError):
        return 0


# Core proxy routing: Local llama-server dispatch

async def proxy_to_local(request: Request, path: str) -> Response:
    """Proxy request to local llama-server with session-based incremental ingestion.

    Uses session headers (X-Session-Id, session_id, X-Client-Request-Id,
    X-Session-Affinity) to track per-session message history and forward
    only new messages (delta) on subsequent requests.
    """
    srv = _srv()
    server_config = srv.config.get("server", {})
    llama_port = server_config.get("llama_server_port", 8080)
    target_url = f"http://localhost:{llama_port}/{path}"

    # Self-healing is active — record 5xx with reason "self_healing"
    if _is_self_healing_active():
        record_http_error("v1/chat/completions", "5xx", "self_healing")
        return _self_healing_response(path)

    # LP-0MQ4GQ2LO005PZPY: Return 503 immediately when backend is unavailable.
    if not srv.backend_ready or srv.llama_process is None:
        return _build_backend_unavailable_response(srv, path)

    # Get request body (keep original for logging before any modifications)
    body = await request.body()
    body_for_logging = body

    # Parse body once and determine method/key/model for attribution
    try:
        body_json = json.loads(body) if body else {}
    except Exception:
        body_json = {}

    # Session handling – incremental prompt ingestion
    session_result = await _handle_session(
        srv, body_json, server_config, request.headers
    )
    session_id = session_result["session_id"]
    session_created = session_result["session_created"]
    is_delta_request = session_result["is_delta_request"]
    session_fallback_reason = session_result["session_fallback_reason"]
    delta_messages = session_result["delta_messages"]
    original_message_count = session_result["original_message_count"]
    if session_result["body_override"] is not None:
        body = session_result["body_override"]
        body_json = session_result["body_json"]

    slot_id = None
    slot_filename = None
    slot_timeout = 3.0
    slot_enabled = False

    # Try job-level scheduler (slot management) first
    scheduler = _get_job_scheduler()
    if scheduler is not None and session_id:
        admit_result = await scheduler.reenter_job(session_id)
        if admit_result is None:
            admit_result = await scheduler.admit_job(session_id)
        if isinstance(admit_result, AdmitResult):
            if admit_result.kind == "ASSIGNED":
                slot_id = admit_result.slot_id
                # Job owns the slot — no save/restore needed
                slot_enabled = False
                srv.logger.debug(
                    "scheduler assigned slot %s to session %s",
                    slot_id, session_id[:8] if session_id else "unknown",
                )
                await scheduler.mark_request_start(slot_id)
            elif admit_result.kind == "QUEUED":
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Slot unavailable: session {session_id[:8]} queued "
                        f"at position {admit_result.position}"
                    ),
                    headers={"Retry-After": "30"},
                )
            elif admit_result.kind == "REJECTED_503":
                raise HTTPException(
                    status_code=503,
                    detail=f"Queue full. Try again later.",
                    headers={
                        "Retry-After": str(
                            int(admit_result.retry_after)
                            if admit_result.retry_after
                            else 900
                        )
                    },
                )

    # Fall back to hash-based slot context if no scheduler assigned a slot
    if slot_id is None:
        slot_id, slot_filename, slot_timeout = _build_slot_context(
            server_config, session_id
        )
        slot_enabled = slot_id is not None and slot_filename is not None

    # Log request with resolved session_id and slot_id (LP-0MQQSM1V7004QOGL)
    log_request(
        request,
        body_for_logging,
        "local",
        session_id=session_id,
        slot_id=slot_id if slot_id is not None else "none",
    )

    method = request.method.upper()
    key = f"{method} {request.url.path} -> local"
    model_name = None
    try:
        model_name = body_json.get("model")
    except Exception:
        model_name = None
    if not model_name:
        model_name = srv.current_model

    slot_model_name = _resolve_slot_model_name(
        model_name, srv.current_model, server_config
    )

    if server_config.get("llama_router_mode", False) and isinstance(
        body_json, dict
    ):
        if slot_model_name and body_json.get("model") != slot_model_name:
            body_json["model"] = slot_model_name
            body = json.dumps(body_json).encode("utf-8")

    if slot_model_name:
        model_name = slot_model_name

    slot_model_payload = (
        slot_model_name
        if server_config.get("llama_router_mode", False)
        else None
    )

    single_flight_mode = server_config.get("session_single_flight_mode", "queue")
    single_flight_max_queue_depth = int(
        server_config.get("session_single_flight_max_queue_depth", 1) or 1
    )

    # Check concurrency limit
    max_queries = server_config.get("max_concurrent_queries", 4)
    try:
        async with srv.active_queries_lock:
            cur_active = srv.active_queries
    except Exception:
        cur_active = 0

    if cur_active >= max_queries:
        # Concurrency limit reached. Attempt to serve from remote providers
        # for the requested model before rejecting, since remote calls do not
        # consume local backend slots.
        try:
            model_cfg = srv.get_model_config(model_name)
            if model_cfg:
                providers = model_cfg.get("providers") or []
                remote_providers = [p for p in providers if isinstance(p, dict) and p.get("type") == "remote"]
                if remote_providers:
                    from proxy.provider import proxy_with_remote_fallback
                    remote_cfg = {"providers": remote_providers}
                    try:
                        resp = await proxy_with_remote_fallback(request, f"v1/{path}", remote_cfg, srv.config)
                        return resp
                    except Exception:
                        srv.logger.exception("Remote fallback during concurrency limit failed; will return concurrency 503")
        except Exception:
            srv.logger.debug("Failed to attempt remote fallback under concurrency limit", exc_info=True)

        # No remote providers or remote attempts failed — reject due to concurrency
        _record_backend_signal("concurrency_rejects")
        srv.logger.warning(
            "concurrency_reject active=%s max=%s path=%s",
            cur_active,
            max_queries,
            path,
        )
        # Concurrency limit reached — record 5xx with reason "concurrency_rejected"
        record_http_error(
            "v1/chat/completions", "5xx", "concurrency_rejected"
        )
        raise HTTPException(
            status_code=503,
            detail=f"Server overloaded: {cur_active} queries active. Retry later.",
        )

    # Check slot availability
    slot_response = await _check_slot_availability(
        srv, server_config, llama_port, slot_model_name, model_name, path
    )
    if slot_response is not None:
        return slot_response

    # Mark active query
    await _increment_active_queries(srv)
    await _increment_local_active_queries(srv)

    # Token accounting
    tokens_sent = _estimate_tokens_sent(body, body_json, model_name)
    await _schedule_token_increment(key, tokens_sent)

    # Forward headers (strip hop-by-hop transport headers)
    headers = normalize_upstream_request_headers(request.headers)

    from proxy.session import _resolve_log_path  # noqa: E402
    llama_log_path = _resolve_log_path("llama")
    try:
        llama_log_offset = (
            llama_log_path.stat().st_size
            if llama_log_path.exists()
            else 0
        )
    except Exception:
        llama_log_offset = 0

    is_streaming = body_json.get("stream", False)

    # Compute request timeout (adaptive if enabled)
    request_timeout = _compute_request_timeout(server_config, body_json)

    if is_streaming:
        session_guard = session_single_flight_coordinator.acquire(
            session_id,
            single_flight_mode,
            single_flight_max_queue_depth,
        )
        slot_guard = slot_lock_coordinator.acquire(slot_id)
        try:
            async with session_guard:
                async with slot_guard:
                    if slot_enabled:
                        restored = await _restore_slot_snapshot(
                            llama_port,
                            slot_id,
                            slot_filename,
                            slot_timeout,
                            model=slot_model_payload,
                        )
                        if restored:
                            srv.logger.info(
                                "slot_restore success session=%s slot=%s",
                                session_id[:8] if session_id else "unknown",
                                slot_id,
                            )
                    slot_save_allowed = slot_enabled

                    client = httpx.AsyncClient(timeout=request_timeout)

                    async def _open_stream_once():
                        stream_cm = client.stream(
                            request.method,
                            target_url,
                            headers=headers,
                            content=body,
                        )
                        stream_resp = await stream_cm.__aenter__()
                        return stream_cm, stream_resp

                    # Enter the stream with bounded retries on transient backend failures
                    try:
                        cm, response = await _call_with_backend_retries(
                            _open_stream_once,
                            path=path,
                            stream=True,
                        )
                        srv.backend_ready = True
                        restore_signal_detected = _has_explicit_restore_signal(
                            dict(response.headers), None
                        )
                        if session_id and not restore_signal_detected:
                            restore_signal_detected = (
                                _detect_restore_signal_from_llama_log(
                                    session_id
                                )
                            )
                    except Exception:
                        srv.backend_ready = False
                        await _decrement_active_queries(srv)
                        try:
                            await client.aclose()
                        except Exception:
                            pass
                        # Self-healing became active during streaming — record 5xx with reason "self_healing"
                        if _is_self_healing_active():
                            record_http_error(
                                "v1/chat/completions", "5xx", "self_healing"
                            )
                            return _self_healing_response(path)
                        # Backend connection/read error — record 5xx with reason "backend_error"
                        record_http_error(
                            "v1/chat/completions", "5xx", "backend_error"
                        )
                        return _build_backend_error_response(
                            srv, path, session_id, session_created,
                            is_delta_request, session_fallback_reason,
                        )
                    upstream_status = response.status_code
                    upstream_content_type = response.headers.get(
                        "content-type", ""
                    )

                    # Return buffered response for non-SSE upstream errors.
                    if upstream_status >= 400 or "text/event-stream" not in upstream_content_type.lower():
                        try:
                            body_bytes = await response.aread()
                        except Exception:
                            body_bytes = b""
                        try:
                            await cm.__aexit__(None, None, None)
                        except Exception:
                            pass
                        try:
                            await client.aclose()
                        except Exception:
                            pass

                        # Upstream returned 5xx — record 5xx with reason "upstream_error"
                        if upstream_status >= 500:
                            record_http_error(
                                "v1/chat/completions", "5xx", "upstream_error"
                            )

                        err_headers = _normalize_outgoing_headers(
                            dict(response.headers), buffered=True
                        )
                        if session_id:
                            err_headers["X-Session-Id"] = session_id
                            err_headers[
                                "X-Session-Created"
                            ] = (
                                "true"
                                if session_created
                                else "false"
                            )
                            err_headers[
                                "X-Session-Delta"
                            ] = (
                                "true"
                                if is_delta_request
                                else "false"
                            )
                            if session_fallback_reason:
                                err_headers[
                                    "X-Session-Fallback-Reason"
                                ] = session_fallback_reason
                        await _decrement_active_queries(srv)
                        return Response(
                            content=body_bytes,
                            status_code=upstream_status,
                            headers=err_headers,
                        )

                    outgoing_headers = _normalize_outgoing_headers(
                        dict(response.headers), buffered=False
                    )
                    if "cache-control" not in {
                        k.lower()
                        for k in outgoing_headers.keys()
                    }:
                        outgoing_headers["Cache-Control"] = "no-cache"

                    if session_id:
                        outgoing_headers["X-Session-Id"] = session_id
                        outgoing_headers[
                            "X-Session-Created"
                        ] = (
                            "true"
                            if session_created
                            else "false"
                        )
                        outgoing_headers[
                            "X-Session-Delta"
                        ] = (
                            "true"
                            if is_delta_request
                            else "false"
                        )
                        if session_fallback_reason:
                            outgoing_headers[
                                "X-Session-Fallback-Reason"
                            ] = session_fallback_reason
                    media_type = response.headers.get(
                        "content-type", "text/event-stream"
                    )

                    guardrail_reason: Optional[str] = None
                    guardrail_response_text = ""
                    completion_tokens_total = 0
                    stream_start = time.monotonic()
                    chunk_history: list[tuple[float, str]] = []
                    max_runtime_seconds = float(
                        server_config.get(
                            "session_guardrail_max_runtime_seconds", 1800
                        )
                        or 1800
                    )
                    max_completion_tokens = int(
                        server_config.get(
                            "session_guardrail_max_completion_tokens",
                            2048,
                        )
                        or 2048
                    )
                    repetition_min_pattern_chars = int(
                        server_config.get(
                            "session_guardrail_repetition_min_pattern_chars",
                            64,
                        )
                        or 64
                    )
                    repetition_min_repeats = int(
                        server_config.get(
                            "session_guardrail_repetition_min_repeats",
                            10,
                        )
                        or 10
                    )
                    invalidate_on_guardrail = bool(
                        server_config.get(
                            "session_guardrail_invalidate_on_cutoff", True
                        )
                    )
                    invalidate_on_repetition = (
                        server_config.get(
                            "session_guardrail_invalidate_on_repetition",
                            False,
                        )
                    )
                    max_token_rate = int(
                        server_config.get(
                            "session_guardrail_max_token_rate", 0
                        )
                        or 0
                    )
                    token_rate_window_seconds = int(
                        server_config.get(
                            "session_guardrail_token_rate_window_seconds",
                            5,
                        )
                        or 5
                    )

                    async def stream_generator():
                        nonlocal guardrail_reason, guardrail_response_text, completion_tokens_total, slot_save_allowed, chunk_history
                        # Track assistant response for session history update
                        collected_content: list[str] = []
                        saw_done = False
                        saw_finish = False
                        # Client disconnect detection (LP-0MQTHP828000JYM6)
                        disconnected = False
                        _disconnect_check_count = 0
                        try:
                            async for chunk in response.aiter_bytes():
                                # count tokens in this chunk (best-effort)
                                try:
                                    chunk_text = chunk.decode(
                                        "utf-8", errors="replace"
                                    )
                                    chunk_tokens = count_text_tokens(
                                        chunk_text, model_name
                                    )
                                    delta_text = (
                                        _extract_delta_text_from_sse_chunk(
                                            chunk_text
                                        )
                                    )
                                    if delta_text:
                                        completion_tokens_total += (
                                            count_text_tokens(
                                                delta_text, model_name
                                            )
                                        )
                                        guardrail_response_text = (
                                            guardrail_response_text
                                            + delta_text
                                        )[-2000:]
                                    try:
                                        loop = asyncio.get_running_loop()
                                        loop.create_task(
                                            _increment_tokens(
                                                "recv",
                                                key,
                                                chunk_tokens,
                                            )
                                        )
                                    except RuntimeError:
                                        asyncio.run(
                                            _increment_tokens(
                                                "recv",
                                                key,
                                                chunk_tokens,
                                            )
                                        )
                                except Exception:
                                    chunk_text = ""
                                    delta_text = ""

                                # Track chunk for token-rate guardrail (rolling window)
                                now_ts = time.monotonic()
                                chunk_history.append((now_ts, chunk_text))

                                # Emit token-rate metrics (best-effort) — gauge + histogram per-session
                                try:
                                    # Only emit if prometheus metrics enabled and we have a session_id
                                    if getattr(metrics, '_enabled', False) and session_id:
                                        # Compute instantaneous rate from last two chunks
                                        if len(chunk_history) >= 2:
                                            t_prev, _ = chunk_history[-2]
                                            t_curr, _ = chunk_history[-1]
                                            elapsed = t_curr - t_prev
                                            if elapsed > 0:
                                                # chunk_tokens computed earlier represents tokens in this raw chunk
                                                try:
                                                    token_rate = float(chunk_tokens) / float(elapsed)
                                                except Exception:
                                                    token_rate = 0.0
                                                try:
                                                    # Use session_id label to scope metrics
                                                    metrics.llama_token_rate_gauge.labels(session_id=session_id).set(token_rate)
                                                    metrics.llama_token_rate_histogram.labels(session_id=session_id).observe(token_rate)
                                                except Exception:
                                                    pass
                                except Exception:
                                    pass

                                # Inspect SSE-style 'data:' lines for finish indicators
                                try:
                                    txt = chunk.decode(
                                        "utf-8", errors="replace"
                                    )
                                    for line in txt.splitlines():
                                        line = line.strip()
                                        if not line.startswith(
                                            "data:"
                                        ):
                                            continue
                                        payload = line[5:].strip()
                                        if payload == "[DONE]":
                                            saw_done = True
                                        else:
                                            try:
                                                j = json.loads(payload)
                                                for choice in j.get(
                                                    "choices", []
                                                ):
                                                    if (
                                                        choice.get(
                                                            "finish_reason"
                                                        )
                                                        is not None
                                                    ):
                                                        saw_finish = True
                                            except Exception:
                                                # ignore non-json payloads
                                                pass
                                except Exception:
                                    pass

                                if not guardrail_reason:
                                    guardrail_reason = evaluate_stream_guardrail(
                                        runtime_seconds=time.monotonic()
                                        - stream_start,
                                        completion_tokens=completion_tokens_total,
                                        response_text=guardrail_response_text,
                                        max_runtime_seconds=max_runtime_seconds,
                                        max_completion_tokens=max_completion_tokens,
                                        repetition_min_pattern_chars=repetition_min_pattern_chars,
                                        repetition_min_repeats=repetition_min_repeats,
                                        chunk_history=chunk_history,
                                        max_token_rate=max_token_rate,
                                        token_rate_window_seconds=token_rate_window_seconds,
                                    )
                                    if guardrail_reason:
                                        _record_guardrail_cutoff(
                                            guardrail_reason
                                        )
                                        srv.logger.warning(
                                            "session_guardrail_cutoff session=%s reason=%s",
                                            session_id[:8]
                                            if session_id
                                            else "unknown",
                                            guardrail_reason,
                                        )
                                        should_invalidate = (
                                            _should_invalidate_on_guardrail(
                                                guardrail_reason,
                                                invalidate_on_guardrail,
                                                bool(
                                                    invalidate_on_repetition
                                                ),
                                            )
                                        )
                                        if session_id and should_invalidate:
                                            await _invalidate_session_and_slot(
                                                session_id,
                                                f"guardrail_{guardrail_reason}",
                                                slot_filename,
                                                scheduler=scheduler,
                                                scheduler_slot_id=slot_id,
                                            )
                                            slot_save_allowed = False
                                        break

                                # Collect content for session history if session is active
                                if session_id:
                                    try:
                                        collected_content.append(
                                            chunk.decode(
                                                "utf-8", errors="replace"
                                            )
                                        )
                                    except Exception:
                                        pass

                                # Check for client disconnect periodically (LP-0MQTHP828000JYM6)
                                _disconnect_check_count += 1
                                if _disconnect_check_count % 10 == 0:
                                    try:
                                        _dc = await request.is_disconnected()
                                        if isinstance(_dc, bool) and _dc:
                                            disconnected = True
                                            srv.logger.info(
                                                "client_disconnect session=%s slot=%s",
                                                session_id[:8] if session_id else "unknown",
                                                slot_id,
                                            )
                                            break
                                    except Exception:
                                        pass

                                yield chunk
                                log_response_chunk(chunk)
                            # Synthesize final SSE event if upstream closed without finish marker.
                            # Skip if client disconnected (LP-0MQTHP828000JYM6)
                            if not disconnected and not saw_done and not saw_finish:
                                finish_reason = (
                                    "stop"
                                    if not guardrail_reason
                                    else "stop"
                                )
                                final_obj = {
                                    "choices": [
                                        {
                                            "delta": {},
                                            "finish_reason": finish_reason,
                                            "index": 0,
                                        }
                                    ]
                                }
                                final_bytes = (
                                    f"data: {json.dumps(final_obj)}\n\n"
                                ).encode("utf-8")
                                yield final_bytes
                                log_response_chunk(
                                    final_bytes
                                )
                        except GeneratorExit:
                            # Client disconnected or generator is being closed.
                            # Skip the final event yield and proceed directly to cleanup.
                            pass
                        except Exception as exc:
                            # httpx stream error (e.g. RemoteProtocolError, ReadTimeout).
                            # Log and let the finally block handle cleanup so backend_ready
                            # is not spuriously set to False (which would cooldown the
                            # local provider and trigger fallback to remotes).
                            srv.logger.warning(
                                "Stream error during local response for model=%s: %s",
                                model_name,
                                type(exc).__name__,
                            )
                        finally:
                            # Strict restore confirmation state comes from explicit backend signal.
                            if session_id:
                                try:
                                    if not restore_signal_detected:
                                        restore_signal_detected = (
                                            _detect_restore_signal_from_log_slice(
                                                llama_log_path,
                                                llama_log_offset,
                                            )
                                        )
                                    if restore_signal_detected:
                                        _record_restore_success()
                                    await srv.session_manager.set_restore_confirmed(
                                        session_id, restore_signal_detected
                                    )
                                except Exception:
                                    srv.logger.debug(
                                        "Failed to set restore-confirmed state",
                                        exc_info=True,
                                    )

                            # Update session history with the full conversation
                            if (
                                session_id
                                and original_message_count > 0
                                and (
                                    collected_content
                                    or not guardrail_reason
                                )
                            ):
                                try:
                                    if collected_content:
                                        full_response = "".join(
                                            collected_content
                                        )
                                        assistant_content = (
                                            _extract_assistant_content_from_sse(
                                                full_response
                                            )
                                        )
                                        existing_messages = []
                                        if (
                                            is_delta_request
                                            and delta_messages
                                        ):
                                            session_obj = (
                                                await srv.session_manager.get(
                                                    session_id
                                                )
                                            )
                                            if session_obj:
                                                existing_messages = list(
                                                    session_obj.messages
                                                )
                                        full_messages = (
                                            merge_session_history_for_update(
                                                existing_messages=existing_messages,
                                                request_messages=list(
                                                    body_json.get(
                                                        "messages", []
                                                    )
                                                ),
                                                delta_messages=delta_messages,
                                                is_delta_request=is_delta_request,
                                                assistant_content=assistant_content,
                                            )
                                        )
                                        await srv.session_manager.update_messages(
                                            session_id, full_messages
                                        )
                                    else:
                                        if (
                                            not is_delta_request
                                            and original_message_count
                                            > 0
                                        ):
                                            await srv.session_manager.update_messages(
                                                session_id,
                                                body_json.get(
                                                    "messages", []
                                                ),
                                            )
                                        elif (
                                            is_delta_request
                                            and delta_messages
                                        ):
                                            await srv.session_manager.append_messages(
                                                session_id, delta_messages
                                            )
                                except Exception:
                                    srv.logger.debug(
                                        f"Failed to update session {session_id[:8]}... history",
                                        exc_info=True,
                                    )

                            if (
                                slot_save_allowed
                                and slot_enabled
                                and upstream_status < 400
                            ):
                                saved = await _save_slot_snapshot(
                                    llama_port,
                                    slot_id,
                                    slot_filename,
                                    slot_timeout,
                                    model=slot_model_payload,
                                )
                                if saved:
                                    srv.logger.info(
                                        "slot_save success session=%s slot=%s",
                                        session_id[:8]
                                        if session_id
                                        else "unknown",
                                        slot_id,
                                    )

                            try:
                                await cm.__aexit__(None, None, None)
                            except Exception:
                                pass
                            try:
                                disconnect_cleanup_timeout = server_config.get("disconnect_cleanup_timeout", 5.0)
                                await asyncio.wait_for(client.aclose(), timeout=disconnect_cleanup_timeout)
                            except (asyncio.TimeoutError, Exception):
                                pass
                            # If client disconnected, release scheduler slot entirely (LP-0MQTHP828000JYM6)
                            if scheduler is not None:
                                if disconnected and session_id:
                                    await scheduler.remove_job(session_id)
                                elif slot_id is not None:
                                    await scheduler.mark_request_end(slot_id)
                            await _decrement_active_queries(srv)

                    return StreamingResponse(
                        stream_generator(),
                        media_type=media_type,
                        headers=outgoing_headers,
                        status_code=upstream_status,
                    )
        except SessionSingleFlightRejected as exc:
            await _decrement_active_queries(srv)
            payload = {
                "error": {
                    "type": "session_single_flight",
                    "code": "session_single_flight",
                    "message": "Another request is already active for this session",
                    "reason": exc.reason,
                },
                "status": 429,
                "session_id": session_id,
                "mode": single_flight_mode,
            }
            return JSONResponse(status_code=429, content=payload)
    else:
        session_guard = session_single_flight_coordinator.acquire(
            session_id,
            single_flight_mode,
            single_flight_max_queue_depth,
        )
        slot_guard = slot_lock_coordinator.acquire(slot_id)
        try:
            async with session_guard:
                async with slot_guard:
                    if slot_enabled:
                        restored = await _restore_slot_snapshot(
                            llama_port,
                            slot_id,
                            slot_filename,
                            slot_timeout,
                            model=slot_model_payload,
                        )
                        if restored:
                            srv.logger.info(
                                "slot_restore success session=%s slot=%s",
                                session_id[:8] if session_id else "unknown",
                                slot_id,
                            )
                    slot_save_allowed = slot_enabled

                    try:
                        async with httpx.AsyncClient(
                            timeout=request_timeout
                        ) as client:
                            method = request.method.lower()

                            async def _send_once():
                                return await getattr(
                                    client, method
                                )(
                                    target_url,
                                    headers=headers,
                                    content=body,
                                )

                            try:
                                response = (
                                    await _call_with_backend_retries(
                                        _send_once, path=path, stream=False
                                    )
                                )
                                srv.backend_ready = True
                            except Exception:
                                srv.backend_ready = False
                                # Self-healing became active during retry — record 5xx with reason "self_healing"
                                if _is_self_healing_active():
                                    record_http_error(
                                        "v1/chat/completions",
                                        "5xx",
                                        "self_healing",
                                    )
                                    return _self_healing_response(path)
                                # Backend error during retry — record 5xx with reason "backend_error"
                                record_http_error(
                                    "v1/chat/completions",
                                    "5xx",
                                    "backend_error",
                                )
                                return _build_backend_error_response(
                                    srv, path, session_id, session_created,
                                    is_delta_request, session_fallback_reason,
                                )

                            response = await _call_with_empty_retry(
                                _send_once, path=path
                            )

                            recv_tokens = 0
                            try:
                                resp_text = response.content.decode(
                                    "utf-8", errors="replace"
                                )
                                recv_tokens = count_text_tokens(
                                    resp_text, model_name
                                )
                                await _schedule_recv_token_increment(
                                    key, recv_tokens
                                )
                            except Exception:
                                pass

                            # Note: Hard completion_tokens cutoff has been removed.
                            # Loop detection via repetition check is used instead.
                            # The max_completion_tokens config is now ignored.

                            # Update session history for non-streaming responses
                            if (
                                session_id
                                and isinstance(body_json, dict)
                                and "messages" in body_json
                            ):
                                try:
                                    resp_content = response.content.decode(
                                        "utf-8", errors="replace"
                                    )
                                    resp_json = (
                                        json.loads(resp_content)
                                        if resp_content
                                        else {}
                                    )
                                    restore_signal_detected = (
                                        _has_explicit_restore_signal(
                                            dict(response.headers),
                                            (
                                                resp_json
                                                if isinstance(
                                                    resp_json, dict
                                                )
                                                else None
                                            ),
                                        )
                                    )
                                    if session_id and not restore_signal_detected:
                                        restore_signal_detected = (
                                            _detect_restore_signal_from_llama_log(
                                                session_id
                                            )
                                        )
                                    if session_id and not restore_signal_detected:
                                        restore_signal_detected = (
                                            _detect_restore_signal_from_log_slice(
                                                llama_log_path,
                                                llama_log_offset,
                                            )
                                        )
                                    if restore_signal_detected:
                                        _record_restore_success()
                                    await srv.session_manager.set_restore_confirmed(
                                        session_id, restore_signal_detected
                                    )
                                    assistant_content = _extract_assistant_content(
                                        resp_json
                                    )
                                    existing_messages = []
                                    if (
                                        is_delta_request
                                        and delta_messages
                                    ):
                                        session_obj = (
                                            await srv.session_manager.get(
                                                session_id
                                            )
                                        )
                                        if session_obj:
                                            existing_messages = list(
                                                session_obj.messages
                                            )
                                    full_messages = (
                                        merge_session_history_for_update(
                                            existing_messages=existing_messages,
                                            request_messages=list(
                                                body_json.get(
                                                    "messages", []
                                                )
                                            ),
                                            delta_messages=delta_messages,
                                            is_delta_request=is_delta_request,
                                            assistant_content=assistant_content,
                                        )
                                    )
                                    await srv.session_manager.update_messages(
                                        session_id, full_messages
                                    )
                                except Exception:
                                    srv.logger.debug(
                                        f"Failed to update session {session_id[:8]}... history",
                                        exc_info=True,
                                    )

                            if (
                                slot_save_allowed
                                and slot_enabled
                                and response.status_code < 400
                            ):
                                saved = await _save_slot_snapshot(
                                    llama_port,
                                    slot_id,
                                    slot_filename,
                                    slot_timeout,
                                    model=slot_model_payload,
                                )
                                if saved:
                                    srv.logger.info(
                                        "slot_save success session=%s slot=%s",
                                        session_id[:8]
                                        if session_id
                                        else "unknown",
                                        slot_id,
                                    )

                            log_response(
                                response.status_code, response.content
                            )

                            # Build response headers with session info
                            resp_headers = _normalize_outgoing_headers(
                                dict(response.headers), buffered=True
                            )
                            if session_id:
                                resp_headers["X-Session-Id"] = session_id
                                resp_headers[
                                    "X-Session-Created"
                                ] = (
                                    "true"
                                    if session_created
                                    else "false"
                                )
                                resp_headers[
                                    "X-Session-Delta"
                                ] = (
                                    "true"
                                    if is_delta_request
                                    else "false"
                                )
                                if session_fallback_reason:
                                    resp_headers[
                                        "X-Session-Fallback-Reason"
                                    ] = session_fallback_reason

                            return Response(
                                content=response.content,
                                status_code=response.status_code,
                                headers=resp_headers,
                            )
                    finally:
                        await _decrement_active_queries(srv)
                        await _decrement_local_active_queries(srv)
                        if scheduler is not None and slot_id is not None:
                            await scheduler.mark_request_end(slot_id)
        except SessionSingleFlightRejected as exc:
            await _decrement_active_queries(srv)
            await _decrement_local_active_queries(srv)
            payload = {
                "error": {
                    "type": "session_single_flight",
                    "code": "session_single_flight",
                    "message": "Another request is already active for this session",
                    "reason": exc.reason,
                },
                "status": 429,
                "session_id": session_id,
                "mode": single_flight_mode,
            }
            return JSONResponse(status_code=429, content=payload)

# Backward-compatibility re-exports for tests
from .router_helpers import (  # noqa: E402, F401
    log_request,
    log_response,
    log_response_chunk,
)
from .proxy_remote import (  # noqa: E402, F401
    proxy_to_remote,
)
