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
from fastapi.responses import JSONResponse, StreamingResponse

# Lazy server import — avoids circular imports when server.py imports us
def _srv():
    import proxy.server as _m
    return _m

# Imports from sibling extracted modules
from proxy.lifecycle import (  # noqa: E402
    _is_self_healing_active,
    _self_healing_response,
    _resolve_slot_model_name,
    _compute_adaptive_timeout,
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
    extract_streamed_assistant_message_from_sse,
    merge_session_history_for_update,
    session_single_flight_coordinator,
    slot_lock_coordinator,
)
from proxy.observability import (  # noqa: E402
    _increment_tokens,
    _record_backend_signal,
)
import proxy.metrics as metrics  # noqa: E402
# legacy alias for convenience
record_http_error = metrics.record_http_error

from proxy.utils import (  # noqa: E402
    _extract_assistant_content,
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
    _cleanup_stale_local_dispatch,
    _compute_request_timeout,
    _decrement_active_queries,
    _decrement_local_active_queries,
    _estimate_tokens_sent,
    _get_request_preview,
    _handle_session,
    _increment_active_queries,
    _increment_local_active_queries,
    _normalize_outgoing_headers,
    _schedule_recv_token_increment,
    _schedule_token_increment,
    _try_acquire_local_dispatch,
    _get_lease_timeout_seconds,
    _call_with_backend_retries,
    _call_with_empty_retry,
    normalize_upstream_request_headers,
    log_request,
    log_response,
    log_response_chunk,
)

from .router_helpers import (  # noqa: E402, F401
    _schedule_traffic_recording,
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
        srv.logger.info(
            "scheduler not initialized: slot_management config missing, "
            "using hash-based slot assignment",
        )
        return None

    pool_size = int(slot_config.get("slot_pool_size", 0) or 0)
    if pool_size < 1:
        srv.logger.info(
            "scheduler not initialized: pool_size=%s < 1, "
            "using hash-based slot assignment",
            pool_size,
        )
        return None

    _job_scheduler = JobScheduler(
        pool_size=pool_size,
        max_queue_depth=int(slot_config.get("slot_queue_max_depth", 16) or 16),
        job_timeout=float(slot_config.get("slot_job_timeout_seconds", 300.0) or 300.0),
        queue_overflow_retry_after=float(
            slot_config.get("slot_queue_overflow_retry_after", 900) or 900
        ),
        max_request_duration=float(
            slot_config.get("slot_max_request_duration_seconds", 600.0) or 600.0
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

    Returns the configured parallel session count, which determines how
    many concurrent sessions can hold local dispatch leases simultaneously.

    The primary config key is ``session_slot_pool_size`` (same value that
    controls ``--parallel`` in llama-server). If not set, falls back to
    the legacy ``local_max_concurrent_queries`` key.  Defaults to 1.

    This limit is separate from the global ``max_concurrent_queries``
    which applies to remote providers.
    """
    try:
        # Primary: session_slot_pool_size (configurable parallel session count)
        val = server_config.get("session_slot_pool_size", None)
        if val is None:
            # Fallback: local_max_concurrent_queries for backward compatibility
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


# ===================================================================
# Extracted helpers for proxy_to_local
# ===================================================================


def _build_session_headers(
    session_id: Optional[str],
    session_created: bool,
    is_delta_request: bool,
    session_fallback_reason: Optional[str],
) -> dict:
    """Build the X-Session-* response headers common to both paths."""
    headers = {}
    if session_id:
        headers["X-Session-Id"] = session_id
        headers["X-Session-Created"] = "true" if session_created else "false"
        headers["X-Session-Delta"] = "true" if is_delta_request else "false"
        if session_fallback_reason:
            headers["X-Session-Fallback-Reason"] = session_fallback_reason
    return headers


def _get_guardrail_config(server_config: dict) -> dict:
    """Extract guardrail parameters from server config.

    Defaults (when config keys are absent or falsy):
        max_runtime_seconds: 1800 (30 minutes) — acts as safety cap for adaptive budget
        max_completion_tokens: 2048
        repetition_min_pattern_chars: 64
        repetition_min_repeats: 10
        invalidate_on_guardrail: False
    """
    return {
        "max_runtime_seconds": float(
            server_config.get("session_guardrail_max_runtime_seconds", 1800) or 1800
        ),
        "max_completion_tokens": int(
            server_config.get("session_guardrail_max_completion_tokens", 2048) or 2048
        ),
        "repetition_min_pattern_chars": int(
            server_config.get("session_guardrail_repetition_min_pattern_chars", 64) or 64
        ),
        "repetition_min_repeats": int(
            server_config.get("session_guardrail_repetition_min_repeats", 10) or 10
        ),
        "invalidate_on_guardrail": bool(
            server_config.get("session_guardrail_invalidate_on_cutoff", True)
        ),
        "invalidate_on_repetition": server_config.get(
            "session_guardrail_invalidate_on_repetition", False
        ),
        "max_token_rate": int(
            server_config.get("session_guardrail_max_token_rate", 0) or 0
        ),
        "token_rate_window_seconds": int(
            server_config.get("session_guardrail_token_rate_window_seconds", 5) or 5
        ),
    }


async def _update_session_and_slot(
    srv,
    session_id: Optional[str],
    body_json: dict,
    is_delta_request: bool,
    delta_messages: list,
    original_message_count: int,
    response,
    llama_port: int,
    slot_id: Optional[str],
    slot_filename: Optional[str],
    slot_timeout: float,
    slot_model_payload: Optional[str],
    slot_enabled: bool,
    upstream_status: int,
    slot_save_allowed: bool = True,
    collected_content: Optional[list] = None,
    llama_log_path=None,
    llama_log_offset: int = 0,
) -> None:
    """Update session history and save slot snapshot after a response.

    Shared by both streaming and buffered paths.
    """
    if not session_id:
        return

    # Restore signal detection and confirmation
    try:
        resp_content = (
            response.content.decode("utf-8", errors="replace")
            if hasattr(response, "content") and isinstance(getattr(response, 'content', None), (bytes, str))
            else None
        )
        restore_signal_detected = _has_explicit_restore_signal(
            dict(response.headers) if hasattr(response, "headers") else {},
            json.loads(resp_content) if resp_content else None,
        )
    except Exception:
        restore_signal_detected = False
    if not restore_signal_detected:
        restore_signal_detected = _detect_restore_signal_from_llama_log(session_id)
    if not restore_signal_detected and llama_log_path is not None:
        restore_signal_detected = _detect_restore_signal_from_log_slice(
            llama_log_path, llama_log_offset
        )
    if restore_signal_detected:
        _record_restore_success()
    try:
        await srv.session_manager.set_restore_confirmed(session_id, restore_signal_detected)
    except Exception:
        srv.logger.debug(
            "Failed to set restore-confirmed state", exc_info=True
        )

    # Update session history
    if (
        session_id
        and isinstance(body_json, dict)
        and "messages" in body_json
        and original_message_count > 0
    ):
        try:
            if collected_content is not None and collected_content:
                full_response = "".join(collected_content)
                assistant_content = _extract_assistant_content_from_sse(full_response)
                assistant_message = extract_streamed_assistant_message_from_sse(
                    full_response
                )
                existing_messages = []
                if is_delta_request and delta_messages:
                    session_obj = await srv.session_manager.get(session_id)
                    if session_obj:
                        existing_messages = list(session_obj.messages)
                full_messages = merge_session_history_for_update(
                    existing_messages=existing_messages,
                    request_messages=list(body_json.get("messages", [])),
                    delta_messages=delta_messages,
                    is_delta_request=is_delta_request,
                    assistant_content=assistant_content,
                    assistant_message=assistant_message,
                )
                await srv.session_manager.update_messages(session_id, full_messages)
            elif hasattr(response, "content") and isinstance(getattr(response, 'content', None), (bytes, str)):
                # Buffered path: parse JSON response
                resp_content = response.content.decode("utf-8", errors="replace")
                resp_json = json.loads(resp_content) if resp_content else {}
                assistant_content = _extract_assistant_content(resp_json)
                assistant_message = None
                if isinstance(resp_json, dict):
                    choices = resp_json.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        maybe_message = choices[0].get("message")
                        if isinstance(maybe_message, dict):
                            assistant_message = maybe_message
                existing_messages = []
                if is_delta_request and delta_messages:
                    session_obj = await srv.session_manager.get(session_id)
                    if session_obj:
                        existing_messages = list(session_obj.messages)
                full_messages = merge_session_history_for_update(
                    existing_messages=existing_messages,
                    request_messages=list(body_json.get("messages", [])),
                    delta_messages=delta_messages,
                    is_delta_request=is_delta_request,
                    assistant_content=assistant_content,
                    assistant_message=assistant_message,
                )
                await srv.session_manager.update_messages(session_id, full_messages)
            else:
                if not is_delta_request and original_message_count > 0:
                    await srv.session_manager.update_messages(
                        session_id, body_json.get("messages", [])
                    )
                elif is_delta_request and delta_messages:
                    await srv.session_manager.append_messages(session_id, delta_messages)
        except Exception:
            srv.logger.debug(
                f"Failed to update session {session_id[:8]}... history",
                exc_info=True,
            )

    # Save slot snapshot if enabled
    if slot_save_allowed and slot_enabled and upstream_status < 400:
        try:
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
                    session_id[:8] if session_id else "unknown",
                    slot_id,
                )
        except Exception:
            srv.logger.debug("slot_save failed", exc_info=True)


async def _release_scheduler_and_decrement(
    srv,
    scheduler,
    session_id: Optional[str],
    slot_id: Optional[int],
    disconnected: bool = False,
    decrement_local: bool = True,
    session_explicit: bool = False,
) -> None:
    """Release scheduler slot and decrement active query counters.

    When *disconnected* is True and *session_id* is known, any dispatch
    lease record for that session is also removed immediately (the client
    is gone, so no lease should persist).

    When *session_explicit* is True and *session_id* is known, the
    corresponding dispatch record is marked as inactive with a future
    expires_at timestamp, keeping the lease alive for a returning session.
    """
    if scheduler is not None:
        if disconnected and session_id:
            await scheduler.remove_job(session_id)
        elif slot_id is not None:
            await scheduler.mark_request_end(slot_id)
    await _decrement_active_queries(srv)
    if decrement_local:
        await _decrement_local_active_queries(
            srv,
            session_key=session_id,
        )
        # For non-explicit sessions (no session affinity), immediately
        # remove the dispatch record instead of letting it linger with
        # a 60-second inactive lease — these one-shot sessions won't
        # return, so accumulating inactive records would block slots.
        if not session_explicit and session_id:
            try:
                lock = getattr(srv, "local_dispatch_records_lock", None)
                if lock is not None:
                    async with lock:
                        records = getattr(srv, "local_dispatch_records", {})
                        if session_id in records:
                            del records[session_id]
                            try:
                                srv.logger.info(
                                    "lease_released session=%s reason=non_explicit",
                                    session_id[:8] if session_id else "unknown",
                                )
                            except Exception:
                                pass
            except Exception:
                pass

    # On client disconnect, immediately remove the dispatch lease record
    if disconnected and session_id:
        try:
            lock = getattr(srv, "local_dispatch_records_lock", None)
            if lock is not None:
                async with lock:
                    records = getattr(srv, "local_dispatch_records", {})
                    if session_id in records:
                        del records[session_id]
                        try:
                            srv.logger.info(
                                "lease_released session=%s reason=disconnect",
                                session_id[:8] if session_id else "unknown",
                            )
                        except Exception:
                            pass
        except Exception:
            pass


# ===================================================================
# Core proxy routing: Local llama-server dispatch
# ===================================================================

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
    session_explicit = session_result.get("session_explicit", False)
    if session_result["body_override"] is not None:
        body = session_result["body_override"]
        body_json = session_result["body_json"]

    # Capture original client→proxy request payload for recording (LP-0MR8FEKK6005V9ML)
    _client_request_payload = body_json

    # Determine model name from request for recording context
    _recording_model = None
    try:
        if isinstance(body_json, dict):
            _recording_model = body_json.get("model") or srv.current_model
    except Exception:
        pass

    # Schedule fire-and-forget recording of the client→proxy request
    if session_id and _client_request_payload:
        _schedule_traffic_recording(
            session_id=session_id,
            client_payload=_client_request_payload,
            model=_recording_model,
        )

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

    # Capture the processed proxy→provider request payload for recording (LP-0MR8FEKK6005V9ML)
    _proxy_provider_payload = body_json if session_id else None
    if _proxy_provider_payload:
        _schedule_traffic_recording(
            session_id=session_id,
            proxy_payload=_proxy_provider_payload,
            model=model_name,
            provider="local",
        )

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

    # -------------------------------------------------------------------
    # Local dispatch gating — no-preemption lease check
    # Only applies to explicitly-provided sessions (X-Session-Id header).
    # Anonymous/auto-generated sessions are ephemeral and should not
    # acquire a persistent lease.
    # -------------------------------------------------------------------
    if session_id and session_explicit:
        local_max = _get_local_max_concurrent_queries(server_config)
        acquired, owner, active_count, retry_after = await _try_acquire_local_dispatch(
            srv,
            max_local=local_max,
            session_key=session_id,
            backend="local",
            body_json=body_json if isinstance(body_json, dict) else None,
        )
        if not acquired:
            srv.logger.info(
                "local_dispatch_denied session=%s owner=%s active=%s",
                session_id[:8] if session_id else "unknown",
                owner[:8] if owner else "none",
                active_count,
            )
            _record_backend_signal("local_dispatch_denied")

            payload = {
                "error": {
                    "type": "server_busy",
                    "code": "no_slots_available",
                    "message": (
                        f"Local backend busy. Owner session "
                        f"{(owner[:8] + '...') if owner else 'unknown'} "
                        f"holds the lease."
                    ),
                },
                "status": 503,
                "retry_after": max(1, int(retry_after)),
                "reason": "local_lease_active",
                "local_owner_session_id": owner,
            }
            return JSONResponse(status_code=503, content=payload)

    # Check slot availability
    slot_response = await _check_slot_availability(
        srv, server_config, llama_port, slot_model_name, model_name, path
    )
    if slot_response is not None:
        return slot_response

    # Mark active query
    await _increment_active_queries(srv)
    # Only increment local_active_queries if _try_acquire_local_dispatch did not
    # already do so (LP-0MR96QL8400022BW: double-increment bug). When the lease
    # check above ran (session_id and session_explicit), _try_acquire_local_dispatch
    # already incremented local_active_queries and created the dispatch record.
    if not (session_id and session_explicit):
        await _increment_local_active_queries(
            srv,
            session_key=session_id,
            backend="local",
        )

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
                        await _release_scheduler_and_decrement(
                            srv, scheduler, session_id, slot_id,
                            decrement_local=True,
                            session_explicit=session_explicit,
                        )
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
                        err_headers.update(
                            _build_session_headers(
                                session_id, session_created,
                                is_delta_request, session_fallback_reason,
                            )
                        )
                        await _release_scheduler_and_decrement(
                            srv, scheduler, session_id, slot_id,
                            decrement_local=True,
                            session_explicit=session_explicit,
                        )
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

                    outgoing_headers.update(
                        _build_session_headers(
                            session_id, session_created,
                            is_delta_request, session_fallback_reason,
                        )
                    )
                    # LP-0MR4ZIGDT004A3E1: Surface resolved provider/model for Pi extension
                    outgoing_headers["X-Resolved-Model"] = f"local/{model_name}"
                    media_type = response.headers.get(
                        "content-type", "text/event-stream"
                    )

                    guardrail_reason: Optional[str] = None
                    guardrail_response_text = ""
                    completion_tokens_total = 0
                    stream_start = time.monotonic()
                    chunk_history: list[tuple[float, str]] = []
                    gc = _get_guardrail_config(server_config)
                    max_runtime_seconds = gc["max_runtime_seconds"]
                    # Compute adaptive guardrail budget when adaptive timeout is enabled
                    # (LP-0MRB9AZDJ00716OT).  Reuses the same adaptive timeout formula
                    # from lifecycle.py that is already used for the HTTP request timeout.
                    _adaptive_enabled = server_config.get(
                        "llama_adaptive_timeout_enabled", False
                    )
                    if _adaptive_enabled and isinstance(body_json, dict):
                        _adaptive_base = float(
                            server_config.get(
                                "llama_adaptive_timeout_base_seconds", 60
                            )
                        )
                        _adaptive_per_token = float(
                            server_config.get(
                                "llama_adaptive_timeout_per_token_seconds", 0.01
                            )
                        )
                        runtime_budget = _compute_adaptive_timeout(
                            body_json,
                            _adaptive_base,
                            _adaptive_per_token,
                            max_runtime_seconds,
                        )
                    else:
                        runtime_budget = float(max_runtime_seconds)
                    max_completion_tokens = gc["max_completion_tokens"]
                    repetition_min_pattern_chars = gc["repetition_min_pattern_chars"]
                    repetition_min_repeats = gc["repetition_min_repeats"]
                    invalidate_on_guardrail = gc["invalidate_on_guardrail"]
                    invalidate_on_repetition = gc["invalidate_on_repetition"]
                    max_token_rate = gc["max_token_rate"]
                    token_rate_window_seconds = gc["token_rate_window_seconds"]
                    stream_idle_timeout = float(
                        server_config.get("stream_idle_timeout_seconds", 30) or 30
                    )
                    stream_heartbeat_interval = float(
                        server_config.get("stream_heartbeat_interval_seconds", 10) or 10
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

                        # Log stream started with session context (LP-0MR90HJED005WI1Z)
                        try:
                            _request_preview = _get_request_preview(body_json)
                            srv.logger.info(
                                "Stream started: provider=local model=%s session=%s request=%s",
                                model_name,
                                session_id or "unknown",
                                _request_preview or "",
                            )
                        except Exception:
                            pass

                        try:
                            # Use asyncio.wait(FIRST_COMPLETED) to concurrently
                            # listen for two events:
                            #   1. A chunk from the upstream (aiter_bytes)
                            #   2. A heartbeat interval expiry
                            #
                            # Unlike asyncio.wait_for(), this approach does NOT
                            # cancel the pending tasks when the heartbeat fires.
                            # Cancelling an in-flight httpx read would destroy the
                            # underlying HTTP connection (llama-server sees "Connection
                            # handling canceled").
                            #
                            # Budget tracking:
                            #   Phase 1 (pre-fill / first chunk): budget =
                            #     max_runtime_seconds (long — large prompt ingestion).
                            #   Phase 2 (between chunks): budget =
                            #     stream_idle_timeout_seconds (short).
                            _stream_aiter = response.aiter_bytes().__aiter__()
                            _stream_iter = asyncio.ensure_future(
                                _stream_aiter.__anext__()
                            )
                            _heartbeat_interval = stream_heartbeat_interval
                            remaining_budget = runtime_budget
                            while True:
                                _hb_task = asyncio.ensure_future(
                                    asyncio.sleep(_heartbeat_interval)
                                )
                                done, pending = await asyncio.wait(
                                    [_stream_iter, _hb_task],
                                    return_when=asyncio.FIRST_COMPLETED,
                                )
                                # CRITICAL: only cancel the heartbeat task (the
                                # one we just created), NEVER cancel _stream_iter
                                # (the pending upstream read).  Cancelling an
                                # in-flight httpx read would destroy the HTTP
                                # connection to llama-server.
                                if _hb_task in done:
                                    # Heartbeat interval elapsed with no chunk.
                                    if remaining_budget <= _heartbeat_interval:
                                        srv.logger.info(
                                            "stream_idle_timeout session=%s "
                                            "idle=%.1fs budget=%.1fs",
                                            session_id[:8] if session_id else "unknown",
                                            stream_idle_timeout,
                                            remaining_budget,
                                        )
                                        break
                                    remaining_budget -= _heartbeat_interval
                                    # Build heartbeat JSON with token progress (LP-0MRDFUHMP005SFU2)
                                    _pct = (
                                        round(completion_tokens_total / max_completion_tokens * 100, 1)
                                        if max_completion_tokens > 0
                                        else 0.0
                                    )
                                    _hb = (
                                        'data: {"type":"heartbeat",'
                                        f'"tokens":{completion_tokens_total},'
                                        f'"max_tokens":{max_completion_tokens},'
                                        f'"pct":{_pct}}}' + '\n\n'
                                    ).encode("utf-8")
                                    yield _hb
                                    continue

                                # A chunk arrived — cancel the heartbeat task
                                _hb_task.cancel()
                                try:
                                    await _hb_task
                                except asyncio.CancelledError:
                                    pass

                                try:
                                    chunk = _stream_iter.result()
                                except StopAsyncIteration:
                                    break

                                # ── process this chunk ──────────────────────
                                try:
                                    chunk_text = chunk.decode(
                                        "utf-8", errors="replace"
                                    )
                                except Exception:
                                    chunk_text = ""

                                chunk_tokens = count_text_tokens(
                                    chunk_text, model_name
                                )
                                delta_text = _extract_delta_text_from_sse_chunk(
                                    chunk_text
                                )
                                if delta_text:
                                    completion_tokens_total += count_text_tokens(
                                        delta_text, model_name
                                    )
                                    guardrail_response_text = (
                                        guardrail_response_text + delta_text
                                    )[-2000:]
                                try:
                                    loop = asyncio.get_running_loop()
                                    loop.create_task(
                                        _increment_tokens("recv", key, chunk_tokens)
                                    )
                                    loop.create_task(
                                        _increment_tokens("recv", f"response:{key}", chunk_tokens)
                                    )
                                except RuntimeError:
                                    asyncio.run(
                                        _increment_tokens("recv", key, chunk_tokens)
                                    )
                                    asyncio.run(
                                        _increment_tokens("recv", f"response:{key}", chunk_tokens)
                                    )

                                now_ts = time.monotonic()
                                chunk_history.append((now_ts, chunk_text))

                                # token-rate metrics
                                try:
                                    if getattr(metrics, '_enabled', False) and session_id:
                                        if len(chunk_history) >= 2:
                                            t_prev, _ = chunk_history[-2]
                                            t_curr, _ = chunk_history[-1]
                                            elapsed = t_curr - t_prev
                                            if elapsed > 0:
                                                token_rate = float(chunk_tokens) / float(elapsed)
                                                metrics.llama_token_rate_gauge.labels(session_id=session_id).set(token_rate)
                                                metrics.llama_token_rate_histogram.labels(session_id=session_id).observe(token_rate)
                                except Exception:
                                    pass

                                # Determine if this chunk carries actual SSE data
                                # (as opposed to a keepalive comment ":").
                                # Only actual data chunks reset the between-chunks
                                # budget, preventing premature timeout on slow
                                # upstream processing.
                                txt = chunk.decode("utf-8", errors="replace")
                                _has_actual_data = bool(
                                    txt.strip()
                                    and not txt.strip().startswith(":")
                                )

                                # SSE finish indicators
                                try:
                                    for line in txt.splitlines():
                                        line = line.strip()
                                        if not line.startswith("data:"):
                                            continue
                                        payload = line[5:].strip()
                                        if payload == "[DONE]":
                                            saw_done = True
                                            _has_actual_data = True
                                        else:
                                            try:
                                                j = json.loads(payload)
                                                for choice in j.get("choices", []):
                                                    if choice.get("finish_reason") is not None:
                                                        saw_finish = True
                                            except Exception:
                                                pass
                                except Exception:
                                    pass

                                if _has_actual_data:
                                    remaining_budget = float(stream_idle_timeout)

                                # Refresh dispatch lease expiry for long-running
                                # streams (LP-0MRDKV44T003FRBP).  Extend the lease
                                # whenever real data arrives, not on heartbeats,
                                # so that streams lasting longer than
                                # local_dispatch_lease_timeout_seconds do not lose
                                # their lease mid-stream.
                                if _has_actual_data and session_id and session_explicit:
                                    try:
                                        _lease_lock = getattr(srv, 'local_dispatch_records_lock', None)
                                        if _lease_lock is not None:
                                            _lease_timeout = _get_lease_timeout_seconds(srv)
                                            async with _lease_lock:
                                                if session_id in srv.local_dispatch_records:
                                                    srv.local_dispatch_records[session_id]['expires_at'] = (
                                                        time.monotonic() + _lease_timeout
                                                    )
                                    except Exception:
                                        pass

                                # guardrail check
                                if not guardrail_reason:
                                    guardrail_reason = evaluate_stream_guardrail(
                                        runtime_seconds=time.monotonic() - stream_start,
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
                                        _record_guardrail_cutoff(guardrail_reason)
                                        srv.logger.warning(
                                            "session_guardrail_cutoff session=%s reason=%s",
                                            session_id[:8] if session_id else "unknown",
                                            guardrail_reason,
                                        )
                                        should_invalidate = _should_invalidate_on_guardrail(
                                            guardrail_reason,
                                            invalidate_on_guardrail,
                                            bool(invalidate_on_repetition),
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

                                # collect session history
                                if session_id:
                                    try:
                                        collected_content.append(
                                            chunk.decode("utf-8", errors="replace")
                                        )
                                    except Exception:
                                        pass

                                # client disconnect
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
                                log_response_chunk(chunk, session_id=session_id, model=model_name, provider="local", body_json=body_json)

                                # Prepare the next anext task
                                _stream_iter = asyncio.ensure_future(
                                    _stream_aiter.__anext__()
                                )

                            # Synthesize final SSE event if upstream closed without finish marker.
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
                                log_response_chunk(final_bytes, session_id=session_id, model=model_name, provider="local", body_json=body_json)
                        except GeneratorExit:
                            # Client disconnected or generator is being closed.
                            # Skip the final event yield and proceed directly to cleanup.
                            pass
                        except Exception as exc:
                            # httpx stream error (e.g. RemoteProtocolError, ReadTimeout).
                            # Log and let the finally block handle cleanup so backend_ready
                            # is not spuriously set to False (which would cooldown the
                            # local provider and trigger fallback to remotes).
                            try:
                                _error_type = type(exc).__name__
                                srv.logger.warning(
                                    "Stream error: session=%s provider=local model=%s error=%s",
                                    session_id or "unknown",
                                    model_name,
                                    _error_type,
                                )
                            except Exception:
                                pass
                            # Synthesize a final SSE event so the client receives a
                            # proper finish_reason marker even on stream error.
                            final_obj = {
                                "choices": [
                                    {"delta": {}, "finish_reason": "error", "index": 0}
                                ]
                            }
                            final_bytes = (
                                f"data: {json.dumps(final_obj)}\n\n"
                            ).encode("utf-8")
                            yield final_bytes
                            log_response_chunk(final_bytes, session_id=session_id, model=model_name, provider="local", body_json=body_json)
                        finally:
                            # Record assembled streaming response (fire-and-forget)
                            if session_id and collected_content:
                                _stream_full_response = "".join(collected_content)
                                _schedule_traffic_recording(
                                    session_id=session_id,
                                    response_payload=_stream_full_response,
                                    model=model_name,
                                    provider="local",
                                )

                            # Update session history and save slot (shared helper)
                            await _update_session_and_slot(
                                srv, session_id, body_json,
                                is_delta_request, delta_messages,
                                original_message_count,
                                response,
                                llama_port, slot_id, slot_filename,
                                slot_timeout, slot_model_payload,
                                slot_enabled,
                                upstream_status=upstream_status,
                                slot_save_allowed=slot_save_allowed,
                                collected_content=collected_content,
                                llama_log_path=llama_log_path,
                                llama_log_offset=llama_log_offset,
                            )

                            # Wrap both cm.__aexit__ and client.aclose() with a
                            # configurable timeout so that an unresponsive upstream
                            # (llama-server stalled mid-stream) does not block the
                            # generator cleanup, which would prevent session counter
                            # and scheduler slot release (LP-0MRE7CMVZ002D2QU).
                            disconnect_cleanup_timeout = server_config.get("disconnect_cleanup_timeout", 5.0)
                            try:
                                await asyncio.wait_for(
                                    cm.__aexit__(None, None, None),
                                    timeout=disconnect_cleanup_timeout,
                                )
                            except (asyncio.TimeoutError, Exception):
                                pass
                            try:
                                await asyncio.wait_for(client.aclose(), timeout=disconnect_cleanup_timeout)
                            except (asyncio.TimeoutError, Exception):
                                pass
                            # Clean up the pending _stream_iter future if the
                            # stream_generator used FIRST_COMPLETED waiting.
                            # CRITICAL: NEVER cancel _stream_iter — cancelling an
                            # in-flight httpx read would destroy the underlying HTTP
                            # connection to llama-server (LP-0MQTHP828000JYM6).
                            # Instead, retrieve the exception (if any) to prevent
                            # "Task exception was never retrieved" warnings from
                            # abandoned asyncio futures (LP-0MRCMKG9O004XE0Q).
                            try:
                                if _stream_iter is not None:
                                    if _stream_iter.done():
                                        # Retrieve the exception (if any) to prevent
                                        # the "never retrieved" warning.  Safe: returns
                                        # None if the future completed successfully.
                                        _stream_iter.exception()
                                    else:
                                        # Future is still pending (in-flight httpx
                                        # read).  Do NOT cancel (CRITICAL constraint).
                                        # Attach a done callback that retrieves the
                                        # exception to prevent the warning when the
                                        # future eventually completes.
                                        def _suppress_abandoned_future(fut):
                                            try:
                                                fut.exception()
                                            except (asyncio.InvalidStateError, Exception):
                                                pass
                                        _stream_iter.add_done_callback(
                                            _suppress_abandoned_future
                                        )
                            except (NameError, AttributeError):
                                # _stream_iter may not exist in all code paths
                                pass

                            # Decrement local active queries now that the stream
                            # has finished (LP-0MR96QL8400022BW: streaming path was
                            # not decrementing local_active_queries, causing subsequent
                            # requests to the same session to be rejected with 503).
                            await _release_scheduler_and_decrement(
                                srv, scheduler, session_id, slot_id,
                                disconnected=disconnected,
                                decrement_local=True,
                                session_explicit=session_explicit,
                            )

                    return StreamingResponse(
                        stream_generator(),
                        media_type=media_type,
                        headers=outgoing_headers,
                        status_code=upstream_status,
                    )
        except SessionSingleFlightRejected as exc:
            await _release_scheduler_and_decrement(
                srv, scheduler, session_id, slot_id,
                decrement_local=False,
            )
            # Clean up any dispatch record that was created before the rejection
            if session_explicit and session_id:
                try:
                    lock = getattr(srv, "local_dispatch_records_lock", None)
                    if lock is not None:
                        async with lock:
                            if session_id in getattr(srv, "local_dispatch_records", {}):
                                del srv.local_dispatch_records[session_id]
                except Exception:
                    pass
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

                            # Record provider→client response (fire-and-forget)
                            if session_id and hasattr(response, "content"):
                                _schedule_traffic_recording(
                                    session_id=session_id,
                                    response_payload=response.content,
                                    model=model_name,
                                    provider="local",
                                )

                            # Update session history and save slot (shared helper)
                            await _update_session_and_slot(
                                srv, session_id, body_json,
                                is_delta_request, delta_messages,
                                original_message_count,
                                response,
                                llama_port, slot_id, slot_filename,
                                slot_timeout, slot_model_payload,
                                slot_enabled,
                                upstream_status=response.status_code,
                                llama_log_path=llama_log_path,
                                llama_log_offset=llama_log_offset,
                            )

                            log_response(
                                response.status_code, response.content
                            )

                            # Build response headers with session info
                            resp_headers = _normalize_outgoing_headers(
                                dict(response.headers), buffered=True
                            )
                            resp_headers.update(
                                _build_session_headers(
                                    session_id, session_created,
                                    is_delta_request, session_fallback_reason,
                                )
                            )
                            # LP-0MR4ZIGDT004A3E1: Surface resolved provider/model for Pi extension
                            resp_headers["X-Resolved-Model"] = f"local/{model_name}"

                            return Response(
                                content=response.content,
                                status_code=response.status_code,
                                headers=resp_headers,
                            )
                    finally:
                        await _release_scheduler_and_decrement(
                            srv, scheduler, session_id, slot_id,
                            decrement_local=True,
                            session_explicit=session_explicit,
                        )
        except SessionSingleFlightRejected as exc:
            await _release_scheduler_and_decrement(
                srv, scheduler, session_id, slot_id,
                decrement_local=True,
            )
            # Clean up any dispatch record that was created before the rejection
            if session_explicit and session_id:
                try:
                    lock = getattr(srv, "local_dispatch_records_lock", None)
                    if lock is not None:
                        async with lock:
                            if session_id in getattr(srv, "local_dispatch_records", {}):
                                del srv.local_dispatch_records[session_id]
                except Exception:
                    pass
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
