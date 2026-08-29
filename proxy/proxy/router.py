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
import logging
import time

import httpx
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("llama-proxy.router")


# Lazy server import — avoids circular imports when server.py imports us
def _srv():
    import proxy.server as _m
    return _m


def _log_local_stream_client_disconnect(srv, session_id, model_name):
    """Log a terminal ``Stream finished: reason=client_disconnect`` line.

    Emitted when a local stream is aborted because the client disconnected
    mid-stream (in-loop ``is_disconnected()`` check, GeneratorExit from the
    ASGI framework closing the generator, or the disconnect reaper cancelling
    the in-flight request task). The line is parseable by the
    proxy-usage-analysis log_parser as a ``stream_finished`` event with
    reason ``client_disconnect`` (distinct from the ``reason=error``
    synthetic events), so the stream pairs and its compute time becomes known
    instead of being reported as "aborted or still running"
    (LP-0MSVRRTAB0078TMK). Logging is best-effort and must never change
    stream behaviour (AC4).
    """
    try:
        srv.logger.info(
            "Stream finished: reason=client_disconnect session=%s provider=local model=%s",
            session_id or "unknown",
            model_name,
        )
    except Exception:
        pass

# Imports from sibling extracted modules
import proxy.metrics as metrics  # noqa: E402
from proxy.lifecycle import (  # noqa: E402
    _compute_adaptive_timeout,
    _is_self_healing_active,
    _resolve_slot_model_name,
    _self_healing_response,
)
from proxy.observability import (  # noqa: E402
    _increment_tokens,
    _record_backend_signal,
)
from proxy.session import (  # noqa: E402
    SessionSingleFlightRejectedError,
    _build_slot_context,
    _detect_restore_signal_from_llama_log,
    _detect_restore_signal_from_log_slice,
    _has_explicit_restore_signal,
    _invalidate_session_and_slot,
    _record_guardrail_cutoff,
    _record_restore_success,
    _restore_slot_snapshot,
    _save_slot_snapshot,
    _should_invalidate_on_guardrail,
    evaluate_stream_guardrail,
    extract_streamed_assistant_message_from_sse,
    merge_session_history_for_update,
    session_single_flight_coordinator,
    slot_lock_coordinator,
)

# legacy alias for convenience
record_http_error = metrics.record_http_error

from proxy.utils import (  # noqa: E402
    _extract_assistant_content,
    _extract_assistant_content_from_sse,
    _extract_delta_text_from_sse_chunk,
    count_text_tokens,
)

# Imports from sibling router helpers
from .router_helpers import (  # noqa: E402  # noqa: E402, F401
    _apply_queue_wait_to_timeout,
    _build_backend_error_response,
    _build_backend_unavailable_response,
    _call_with_backend_retries,
    _call_with_empty_retry,
    _check_slot_availability,
    _client_identity_extra,
    _compute_request_timeout,
    _decrement_active_queries,
    _decrement_local_active_queries,
    _decrement_per_model_query,
    _estimate_tokens_sent,
    _extend_lease_during_prefill,
    _get_chunk_refresh_buffer_seconds,
    _get_lease_timeout_seconds,
    _get_request_preview,
    _handle_session,
    _increment_active_queries,
    _increment_local_active_queries,
    _increment_per_model_query,
    _normalize_outgoing_headers,
    _schedule_recv_token_increment,
    _schedule_token_increment,
    _schedule_traffic_recording,
    _try_acquire_local_dispatch,
    log_request,
    log_response,
    log_response_chunk,
    normalize_upstream_request_headers,
)


def _get_local_max_concurrent_queries(server_config: dict) -> int:
    """
    Read the local-model concurrency limit from config.

    Returns the configured parallel session count, which determines how
    many concurrent sessions can hold local dispatch leases simultaneously.

    Reads ``session_slot_pool_size`` (same value that controls
    ``--parallel`` in llama-server). The legacy
    ``local_max_concurrent_queries`` fallback was removed (LP-0MTCZ35X7009IZKE)
    and a missing value is caught at launch by
    ``validate_local_routing_config``.  Defaults to 1.

    This limit is separate from the global ``max_concurrent_queries``
    which applies to remote providers.
    """
    try:
        return max(1, int(server_config.get("session_slot_pool_size", 1) or 1))
    except (ValueError, TypeError):
        return 1


def _get_contention_queue_config(server_config: dict) -> dict:
    """
    Resolve the per-mode contention-queue policy and caps with sane clamps.

    Returns a dict with keys:
      - ``policy``: "queue" or "fallback" (absent keys default to fallback,
        fast-mode behavior)
      - ``max_wait_seconds``: clamped to [1, session_guardrail_max_runtime_seconds]
      - ``max_depth``: clamped to [1, 16]

    Invalid values are logged and clamped, never crash (LP-0MSORQVK50012Q4D
    F2 AC3/AC4). The policy applies during cheap operating mode only (the
    caller gates on proxy.mode.read_mode() == "cheap").
    """
    policy = str(
        server_config.get("contention_queue_policy", "fallback") or "fallback"
    ).strip().lower()
    if policy not in ("queue", "fallback"):
        logger.warning(
            "Invalid contention_queue_policy=%r — coercing to 'fallback' "
            "(valid: queue, fallback)",
            server_config.get("contention_queue_policy"),
        )
        policy = "fallback"
    # Wait cap: [1, session_guardrail_max_runtime_seconds]
    try:
        max_runtime = int(
            server_config.get("session_guardrail_max_runtime_seconds", 1800) or 1800
        )
    except (TypeError, ValueError):
        logger.warning(
            "Invalid session_guardrail_max_runtime_seconds=%r — using default 1800",
            server_config.get("session_guardrail_max_runtime_seconds"),
        )
        max_runtime = 1800
    raw_wait = server_config.get("contention_queue_max_wait_seconds", 60)
    try:
        wait = float(raw_wait) if raw_wait is not None else 60.0
    except (TypeError, ValueError):
        logger.warning(
            "Invalid contention_queue_max_wait_seconds=%r — using default 60",
            raw_wait,
        )
        wait = 60.0
    max_wait_seconds = max(1.0, min(wait, float(max(1, max_runtime))))
    if max_wait_seconds != wait:
        logger.warning(
            "contention_queue_max_wait_seconds=%r clamped to %s "
            "(bounds [1, session_guardrail_max_runtime_seconds=%s])",
            raw_wait, max_wait_seconds, max_runtime,
        )
    # Depth cap: [1, 16]
    raw_depth = server_config.get("contention_queue_max_depth", 4)
    try:
        depth = int(raw_depth) if raw_depth is not None else 4
    except (TypeError, ValueError):
        logger.warning(
            "Invalid contention_queue_max_depth=%r — using default 4",
            raw_depth,
        )
        depth = 4
    max_depth = max(1, min(depth, 16))
    if max_depth != depth:
        logger.warning(
            "contention_queue_max_depth=%r clamped to %s (bounds [1, 16])",
            raw_depth, max_depth,
        )
    return {
        "policy": policy,
        "max_wait_seconds": max_wait_seconds,
        "max_depth": max_depth,
    }


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
    session_id: str | None,
    session_created: bool,
    is_delta_request: bool,
    session_fallback_reason: str | None,
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
            server_config.get("session_guardrail_max_completion_tokens", 16384) or 16384
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
    session_id: str | None,
    body_json: dict,
    is_delta_request: bool,
    delta_messages: list,
    original_message_count: int,
    response,
    llama_port: int,
    slot_id: str | None,
    slot_filename: str | None,
    slot_timeout: float,
    slot_model_payload: str | None,
    slot_enabled: bool,
    upstream_status: int,
    slot_save_allowed: bool = True,
    collected_content: list | None = None,
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

    # Update per-session cached-tokens ratio from REAL usage data
    # (LP-0MS9GAN2P009KK6G). The final usage chunk in the local response
    # reports prompt_tokens + prompt_tokens_details.cached_tokens; use the
    # true cache-hit fraction when available. The save-success 1.0 below
    # remains the fallback when no usage data is present.
    _real_usage_applied = False
    if session_id and srv.current_model:
        try:
            from proxy.provider import (
                _extract_cached_tokens_from_usage,
                _extract_usage_from_sse_text,
                update_cached_ratio,
            )
            _full_text = None
            if collected_content is not None and collected_content:
                _full_text = "".join(collected_content)
            elif hasattr(response, "content") and isinstance(
                getattr(response, "content", None), (bytes, str)
            ):
                _full_text = response.content.decode(
                    "utf-8", errors="replace"
                )
            _usage = None
            if _full_text:
                # Streaming SSE path: the final chunk carries a usage event.
                _usage = _extract_usage_from_sse_text(_full_text)
                if _usage is None:
                    # Buffered path: JSON body with a top-level usage field.
                    try:
                        _body = json.loads(_full_text)
                        if isinstance(_body, dict) and isinstance(
                            _body.get("usage"), dict
                        ):
                            _usage = _body["usage"]
                    except Exception:
                        _usage = None
            if isinstance(_usage, dict):
                _cached = _extract_cached_tokens_from_usage(_usage)
                _prompt = int(_usage.get("prompt_tokens", 0) or 0)
                if _prompt > 0:
                    update_cached_ratio(
                        srv.current_model,
                        session_id,
                        cached_tokens=_cached,
                        prompt_tokens=_prompt,
                    )
                    _real_usage_applied = True
        except Exception:
            srv.logger.debug(
                "update_cached_ratio from usage failed", exc_info=True
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
                # Update per-session cached-tokens ratio (LP-0MRMMBZ7T007ER59)
                # as a FALLBACK when no real usage data was reported.
                if session_id and srv.current_model and not _real_usage_applied:
                    try:
                        from proxy.provider import update_cached_ratio
                        update_cached_ratio(
                            srv.current_model,
                            session_id,
                            cached_tokens=1,
                            prompt_tokens=1,
                        )
                    except Exception:
                        srv.logger.debug(
                            "update_cached_ratio failed", exc_info=True
                        )

        except Exception:
            srv.logger.debug("slot_save failed", exc_info=True)


async def _cleanup_after_request(
    srv,
    session_id: str | None,
    disconnected: bool = False,
    decrement_local: bool = True,
    session_explicit: bool = False,
    model_name: str | None = None,
    request: Request | None = None,
) -> None:
    """Decrement active query counters and clean up dispatch records.

    The dispatch lease system (``_try_acquire_local_dispatch``) handles
    concurrency gating, session ownership, and timeout-based release
    independently. This cleanup ensures counters and lease records are
    properly released after a request completes.

    When *disconnected* is True and *session_id* is known, any dispatch
    lease record for that session is also removed immediately (the client
    is gone, so no lease should persist).

    When *session_explicit* is True and *session_id* is known, the
    corresponding dispatch record is marked as inactive with a future
    expires_at timestamp, keeping the lease alive for a returning session.

    When *model_name* is provided, the per-model active query counter
    is also decremented.

    When *request* is provided, ``lease_released`` log events carry the
    caller's client identity (``client_ip`` / ``client_port``) for poller
    attribution (LP-0MSKV3IEQ004ZV88).
    """
    await _decrement_active_queries(srv)
    await _decrement_per_model_query(srv, model_name)
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
                                    session_id if session_id else "unknown",
                                    extra=_client_identity_extra(request),
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
                                session_id if session_id else "unknown",
                                extra=_client_identity_extra(request),
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

    # Use hash-based slot context (dispatch lease system handles concurrency gating)
    slot_id, slot_filename, slot_timeout = _build_slot_context(
        server_config, session_id, body_json
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
    single_flight_queue_timeout = float(
        server_config.get("session_single_flight_queue_timeout_seconds", 120) or 120
    )

    # ── Hard local-routing cap (LP-0MTBOX45O005LD1S) ──
    # Check BEFORE concurrency/lease/slot gating so requests above the cap
    # never acquire a dispatch lease or consume slot resources.
    # Fast mode: skip local with context_too_large when above cap.
    # Cheap mode: return a 429 compaction-gate response (no silent remote).
    #
    # The estimate MUST use the provider's authoritative pipeline
    # (``_get_tokenizer_for_model`` + ``_estimate_effective_prompt_tokens_for_routing``)
    # — native tokenizer + session history + multiplier — IDENTICAL to the
    # provider smart-routing block (provider.py ``_should_skip_local`` path),
    # so the gate fires whenever the provider would skip local.  The body-
    # only tiktoken estimate (``_estimate_tokens_sent``, which returns a DICT
    # and would crash the cap comparison) undercounts session-heavy /
    # native-tokenized requests (~1.69x vs Qwen3).  With a lower gate
    # estimate an over-cap cheap request passes the gate, then the provider
    # block fires ``context_too_large`` → routes to the next REMOTE provider
    # = silent remote fallback in cheap mode (AC2 violation).
    try:
        from proxy.provider import (
            _estimate_effective_prompt_tokens_for_routing,
            _get_tokenizer_for_model,
        )
        _model_cfg = srv.get_model_config(model_name) if model_name else None
        _tokenizer, _multiplier = _get_tokenizer_for_model(_model_cfg, server_config)
        _estimated_tokens = await _estimate_effective_prompt_tokens_for_routing(
            request, body_json, tokenizer=_tokenizer,
        )
        if _multiplier != 1.0:
            _estimated_tokens = int(_estimated_tokens * _multiplier)
    except Exception:
        # Fail-open: an estimate error must never break local routing.  The
        # provider smart-routing block re-estimates per attempt anyway.
        _estimated_tokens = 0
    try:
        from proxy.mode import read_mode as _read_mode
        _mode = _read_mode()
    except Exception:
        _mode = "fast"
    from proxy.provider import (
        check_hard_routing_cap,
        compute_hard_routing_cap,
    )
    if check_hard_routing_cap(_estimated_tokens, _mode, server_config):
        _cap = compute_hard_routing_cap(_mode, server_config)
        if _mode == "cheap":
            # Cheap mode: return a 429 compaction-gate response.
            srv.logger.info(
                "compaction_gate session=%s tokens=%d cap=%d mode=%s",
                session_id or "anonymous",
                _estimated_tokens,
                _cap,
                _mode,
            )
            from proxy.provider import _build_compaction_gate_response
            return _build_compaction_gate_response(
                _estimated_tokens, _cap, _mode, session_id, model_name,
            )
        else:
            # Fast mode: skip local with context_too_large.
            srv.logger.info(
                "context_too_large session=%s tokens=%d cap=%d mode=%s",
                session_id or "anonymous",
                _estimated_tokens,
                _cap,
                _mode,
            )
            _record_backend_signal("context_too_large")
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Context ({_estimated_tokens} tokens) exceeds the "
                    f"{model_name or 'model'} local routing cap ({_cap} tokens) "
                    "in fast mode. Compact the session and retry."
                ),
                headers={"X-Context-Too-Large": "true"},
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
    acquired = False
    if session_id and session_explicit:
        local_max = _get_local_max_concurrent_queries(server_config)
        acquired, owner, active_count, retry_after = await _try_acquire_local_dispatch(
            srv,
            max_local=local_max,
            session_key=session_id,
            backend="local",
            body_json=body_json if isinstance(body_json, dict) else None,
            model_name=model_name,
        )
        if not acquired:
            srv.logger.info(
                "local_dispatch_denied session=%s owner=%s active=%s",
                session_id if session_id else "unknown",
                owner if owner else "none",
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

    # Check slot availability — skipped when the dispatch lease was acquired
    # (the lease already gates concurrency to session_slot_pool_size; the
    # router /slots?model= check is redundant and can take 5-7s under load,
    # LP-0MTDGBRPU003Z7KU).
    _lease_held = bool(session_id and session_explicit) and acquired
    slot_response = await _check_slot_availability(
        srv, server_config, llama_port, slot_model_name, model_name, path,
        lease_held=_lease_held,
    )
    if slot_response is not None:
        return slot_response

    # Mark active query
    await _increment_active_queries(srv)
    try:
        await _increment_per_model_query(srv, model_name)
    except Exception:
        srv.logger.warning("Failed to increment per-model query counter", exc_info=True)
    # Only increment local_active_queries if _try_acquire_local_dispatch did not
    # already do so (LP-0MR96QL8400022BW: double-increment bug). When the lease
    # check above ran (session_id and session_explicit), _try_acquire_local_dispatch
    # already incremented local_active_queries and created the dispatch record.
    if not (session_id and session_explicit):
        # Anonymous/non-explicit sessions: apply the adaptive lease timeout
        # for large prompts so the lease covers the prefill phase (which
        # produces no stream chunks to refresh it) instead of expiring after
        # the base 60s and being orphan-cleaned mid-prefill
        # (LP-0MSEHMMBK0062ZPI).
        await _increment_local_active_queries(
            srv,
            session_key=session_id,
            backend="local",
            body_json=body_json if isinstance(body_json, dict) else None,
            model_name=model_name,
        )

    # Token accounting
    tokens_sent = _estimate_tokens_sent(body, body_json, model_name)
    await _schedule_token_increment(key, tokens_sent)

    # Forward headers (strip hop-by-hop transport headers)
    headers = normalize_upstream_request_headers(request.headers)

    from proxy.session import _resolve_log_path
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
    # Contention-queue wait subtracts from the client-visible adaptive
    # timeout budget (Q2=a, LP-0MSORQVK50012Q4D): total (wait + serve) stays
    # within llama_adaptive_timeout_* so clients never see queue wait + serve
    # exceed the adaptive envelope. provider.py sets this attribute when a
    # queued request wins a slot.
    _queue_wait = float(getattr(request, "_contention_queue_wait_seconds", 0.0) or 0.0)
    if _queue_wait > 0:
        request_timeout = _apply_queue_wait_to_timeout(request_timeout, _queue_wait)

    if is_streaming:
        session_guard = session_single_flight_coordinator.acquire(
            session_id,
            single_flight_mode,
            single_flight_max_queue_depth,
            queue_timeout_seconds=single_flight_queue_timeout,
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
                        await _cleanup_after_request(
                            srv, session_id,
                            decrement_local=True,
                            session_explicit=session_explicit,
                            request=request,
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
                        await _cleanup_after_request(
                            srv, session_id,
                            decrement_local=True,
                            session_explicit=session_explicit,
                            request=request,
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

                    guardrail_reason: str | None = None
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
                        server_config.get("stream_idle_timeout_seconds", 120) or 120
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
                            # ── Prefill-phase dispatch-lease tracking ──
                            # The prefill phase emits no stream data chunks, so
                            # the chunk-based lease refresh below never fires;
                            # instead we poll llama-server for observed prefill
                            # progress on the heartbeat branch and extend the
                            # lease while progress advances (LP-0MSE05J53004C6EL).
                            _saw_actual_data = False
                            _last_prefill_poll = 0.0
                            _prefill_progress = 0
                            _prefill_poll_seconds = float(
                                server_config.get(
                                    "local_dispatch_lease_prefill_poll_seconds", 10
                                ) or 10
                            )
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
                                    if remaining_budget < _heartbeat_interval:
                                        srv.logger.info(
                                            "stream_idle_timeout session=%s "
                                            "idle=%.1fs budget=%.1fs",
                                            session_id[:8] if session_id else "unknown",
                                            stream_idle_timeout,
                                            remaining_budget,
                                        )
                                        break
                                    remaining_budget -= _heartbeat_interval
                                    # ── Prefill-phase lease extension ──
                                    # While the request is still in the prefill
                                    # phase (no actual data chunk yet) and the
                                    # session is explicit, poll llama-server for
                                    # observed prefill progress at the configured
                                    # cadence and extend the dispatch lease by
                                    # the safety buffer while progress advances.
                                    # This covers very large prefills beyond the
                                    # adaptive token-estimate cap (1500s); the
                                    # existing chunk-refresh path takes over once
                                    # the first data chunk arrives.
                                    if (
                                        not _saw_actual_data
                                        and session_id
                                        and session_explicit
                                        and _prefill_poll_seconds > 0
                                    ):
                                        _now_ts = time.monotonic()
                                        if (
                                            _now_ts - _last_prefill_poll
                                            >= _prefill_poll_seconds
                                        ):
                                            _last_prefill_poll = _now_ts
                                            _prefill_progress, _extended = (
                                                await _extend_lease_during_prefill(
                                                    srv,
                                                    session_id,
                                                    llama_port=llama_port,
                                                    model_name=model_name,
                                                    slot_id=slot_id,
                                                    last_progress=_prefill_progress,
                                                )
                                            )
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
                                    # Prefill phase is over: the first actual data
                                    # chunk has arrived, so stop progress-based
                                    # lease extension — the chunk-refresh path
                                    # below takes over (LP-0MSE05J53004C6EL).
                                    _saw_actual_data = True

                                # Refresh dispatch lease expiry for long-running
                                # streams (LP-0MRDKV44T003FRBP).  Extend the lease
                                # whenever real data arrives, not on heartbeats,
                                # so that streams lasting longer than
                                # local_dispatch_lease_timeout_seconds do not lose
                                # their lease mid-stream.
                                #
                                # Non-explicit (anonymous) sessions also refresh
                                # here: they acquire an adaptive lease but have no
                                # prefill-progress path, so data-chunk refresh is
                                # their only runtime protection against a 15s base
                                # lease expiring during long silent generation
                                # (LP-0MSUO6HLX0089MNQ). They refresh by a dedicated
                                # safety buffer (default 30s) instead of the base
                                # lease timeout.
                                if _has_actual_data and session_id:
                                    try:
                                        _lease_lock = getattr(srv, 'local_dispatch_records_lock', None)
                                        if _lease_lock is not None:
                                            if session_explicit:
                                                _lease_timeout = _get_lease_timeout_seconds(srv)
                                            else:
                                                _lease_timeout = _get_chunk_refresh_buffer_seconds(srv)
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
                                            # Terminal line so the aborted stream
                                            # is attributable (LP-0MSVRRTAB0078TMK)
                                            _log_local_stream_client_disconnect(
                                                srv, session_id, model_name
                                            )
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
                                ).encode()
                                yield final_bytes
                                log_response_chunk(final_bytes, session_id=session_id, model=model_name, provider="local", body_json=body_json)
                                # Emit [DONE] marker after synthetic finish event
                                # so client agents detect stream completion (LP-0MS14PM7I0077MXD)
                                done_bytes = b"data: [DONE]\n\n"
                                yield done_bytes
                                log_response_chunk(done_bytes, session_id=session_id, model=model_name, provider="local", body_json=body_json)
                        except GeneratorExit:
                            # Client disconnected or generator is being closed.
                            # Log a terminal line so the aborted stream is
                            # attributable (LP-0MSVRRTAB0078TMK), then skip the
                            # final event yield and proceed directly to cleanup.
                            _log_local_stream_client_disconnect(
                                srv, session_id, model_name
                            )
                        except asyncio.CancelledError:
                            # The disconnect reaper (LP-0MQTHP828000JYM6) cancels
                            # the in-flight request task when the client drops
                            # the connection. Log a terminal line so the aborted
                            # stream is attributable, then re-raise so the task
                            # cancellation propagates normally (LP-0MSVRRTAB0078TMK).
                            _log_local_stream_client_disconnect(
                                srv, session_id, model_name
                            )
                            raise
                        except Exception as exc:
                            # httpx stream error (e.g. RemoteProtocolError, ReadTimeout).
                            # Log and let the finally block handle cleanup so backend_ready
                            # is not spuriously set to False (which would cooldown the
                            # local provider and trigger fallback to remotes).
                            #
                            # Proxy-code bug classification (LP-0MSDRRPV0001TCLX):
                            # NameError/AttributeError raised in the loop are proxy-side
                            # coding bugs (undefined name / bad attribute access), never an
                            # upstream fault. On Aug 3 a missing import raised NameError 3x
                            # and was masked as a generic stream error with no traceback.
                            # Log a full traceback for these so they are self-diagnosing,
                            # while still emitting the synthetic finish_reason:error / [DONE]
                            # terminal events clients depend on.
                            try:
                                _error_type = type(exc).__name__
                                if isinstance(exc, (NameError, AttributeError)):
                                    srv.logger.exception(
                                        "PROXY-CODE BUG in local stream loop: %s - this "
                                        "is NOT an upstream stream error (session=%s "
                                        "provider=local model=%s); fix router.py",
                                        _error_type,
                                        session_id or "unknown",
                                        model_name,
                                    )
                                else:
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
                            final_obj = _build_stream_error_event(
                                provider="local",
                                model=model_name,
                                error_type="stream_exception",
                                message=f"Local stream error ({_error_type}); llama-server may be unhealthy",
                                suggested_action="Check llama-server logs; the request may be retried",
                                session_id=session_id,
                            )
                            final_bytes = _stream_error_event_bytes(
                                provider="local",
                                model=model_name,
                                error_type="stream_exception",
                                message=f"Local stream error ({_error_type}); llama-server may be unhealthy",
                                suggested_action="Check llama-server logs; the request may be retried",
                                session_id=session_id,
                            )
                            yield final_bytes
                            log_response_chunk(final_bytes, session_id=session_id, model=model_name, provider="local", body_json=body_json)
                            # Emit [DONE] marker after synthetic error finish event
                            # so client agents detect stream completion (LP-0MS14PM7I0077MXD)
                            done_bytes = b"data: [DONE]\n\n"
                            yield done_bytes
                            log_response_chunk(done_bytes, session_id=session_id, model=model_name, provider="local", body_json=body_json)
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
                            # and dispatch lease release (LP-0MRE7CMVZ002D2QU).
                            disconnect_cleanup_timeout = server_config.get("disconnect_cleanup_timeout", 5.0)
                            try:
                                await asyncio.wait_for(
                                    cm.__aexit__(None, None, None),
                                    timeout=disconnect_cleanup_timeout,
                                )
                            except (TimeoutError, Exception):
                                pass
                            try:
                                await asyncio.wait_for(client.aclose(), timeout=disconnect_cleanup_timeout)
                            except (TimeoutError, Exception):
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
                            await _cleanup_after_request(
                                srv, session_id,
                                disconnected=disconnected,
                                decrement_local=True,
                                session_explicit=session_explicit,
                                model_name=model_name,
                                request=request,
                            )

                    return StreamingResponse(
                        stream_generator(),
                        media_type=media_type,
                        headers=outgoing_headers,
                        status_code=upstream_status,
                    )
        except SessionSingleFlightRejectedError as exc:
            await _cleanup_after_request(
                srv, session_id,
                decrement_local=False,
                model_name=model_name,
                request=request,
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
            queue_timeout_seconds=single_flight_queue_timeout,
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
                        await _cleanup_after_request(
                            srv, session_id,
                            decrement_local=True,
                            session_explicit=session_explicit,
                            request=request,
                        )
        except SessionSingleFlightRejectedError as exc:
            await _cleanup_after_request(
                srv, session_id,
                decrement_local=True,
                request=request,
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
from .proxy_remote import (  # noqa: E402, F401
    _build_stream_error_event,
    _stream_error_event_bytes,
    proxy_to_remote,
)
from .router_helpers import (  # noqa: E402, F811
    log_request,
    log_response,
    log_response_chunk,
)
