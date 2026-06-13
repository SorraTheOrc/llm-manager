#!/usr/bin/env python3
"""
LLama Proxy Server

A proxy server that routes OpenAI API requests to either a local llama-server
or remote API services based on configuration.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from fnmatch import fnmatch
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx
import shutil
import io
import traceback
import threading
from datetime import timedelta
import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from proxy.session_manager import SessionManager, DEFAULT_SESSION_TTL_SECONDS
import proxy.metrics as metrics
from proxy.metrics import record_http_error

# Global state
llama_process: Optional[subprocess.Popen] = None
llama_log_file: Optional[Any] = None
current_model: Optional[str] = None
last_start_failure: Optional[str] = None
model_switch_lock = asyncio.Lock()
# Mark if a background model load is in progress (model_name -> True)
background_loads: dict = {}
# Track last-used timestamps for models (model_id -> ISO8601 string)
model_last_used: dict = {}
# Reference count for model switch/load operations to provide a
# cross-task (and cross-thread-safe observable) indicator that a model
# switch is in progress. We prefer a simple integer refcount rather
# than relying solely on asyncio.Lock.locked() because background
# loaders may run in different event loops/threads during tests and
# startup, making Lock visibility unreliable across contexts.
model_switch_refcount: int = 0
# Lock to protect model_switch_refcount updates across threads
model_switch_refcount_lock = threading.Lock()

# Session manager for incremental prompt ingestion
session_manager: SessionManager = SessionManager(ttl_seconds=DEFAULT_SESSION_TTL_SECONDS)


# Shared httpx client for connection pooling
_http_client: Optional[httpx.AsyncClient] = None
# Cache for which llama-server endpoint successfully provided status
_llama_status_endpoint_cache: Optional[str] = None
# Record recent failures for endpoints to avoid hammering endpoints that 404
_llama_status_endpoint_failures: dict = {}
# One-time discovery markers: avoid repeated discovery for the same process
_llama_status_discovered: bool = False
_llama_status_discovered_pid: Optional[int] = None

#   extract_progress_data, poll_slots_for_model, start_slot_polling, format_progress
# The module-level state they reference remains here.
from .handlers import extract_progress_data, format_progress, poll_slots_for_model, start_slot_polling  # noqa: F401

# Polling state for /slots API (model -> latest data)
slot_polling_state: dict = {}
# Internal record of active polling tasks (model -> asyncio.Task)
_slot_polling_tasks: dict = {}

# Request counting
request_counts: dict = {}
counts_lock = asyncio.Lock()
counts_filename = "request_counts.json"
# Dirty flags and persist tasks
counts_dirty = False
counts_persist_task: Optional[asyncio.Task] = None
periodic_broadcast_task: Optional[asyncio.Task] = None

# Active local queries counter
active_queries: int = 0
active_queries_lock = asyncio.Lock()

# Backend resilience/observability signals
# Health/readiness signal for local backend
backend_ready: bool = False

# Self-healing state for backend recovery attempts
backend_recovery_state: dict = {
    "in_progress": False,
    "attempt_timestamps": [],
    "max_attempts": 3,
    "window_seconds": 300,
    "retry_after_seconds": 30,
    "last_failure": None,
}

# Background watchdog task (started in lifespan)
backend_watchdog_task: Optional[asyncio.Task] = None

# Token counting
token_counts: dict = {}
token_lock = asyncio.Lock()
token_counts_filename = "token_counts.json"
# Dirty flags and persist tasks for tokens
tokens_dirty = False
tokens_persist_task: Optional[asyncio.Task] = None


# Optional process metrics (psutil)
try:
    import psutil
except Exception:
    psutil = None


config: dict = {}


log_dir: Optional[Path] = None
logger: logging.Logger = logging.getLogger("llama-proxy")


# SSE clients for real-time status updates




async def proxy_to_local(request: Request, path: str) -> Response:
    """Proxy request to local llama-server with optional session-based
    incremental ingestion.

    When a request includes a supported session header (``X-Session-Id``,
    ``session_id``, ``X-Client-Request-Id``, or ``X-Session-Affinity``),
    or the proxy generates one, the proxy tracks per-session message
    history so that only new messages are forwarded to llama-server on
    subsequent requests within the same session. llama-server's
    ``session_id`` and ``cache_prompt`` parameters are used to preserve
    the KV cache across requests.
    """
    global active_queries, backend_ready
    server_config = config.get("server", {})
    llama_port = server_config.get("llama_server_port", 8080)
    target_url = f"http://localhost:{llama_port}/{path}"

    if _is_self_healing_active():
        record_http_error("v1/chat/completions", "5xx", "self_healing")
        return _self_healing_response(path)

    # LP-0MQ4GQ2LO005PZPY: Return 503 immediately when backend is unavailable.
    # This covers the gap between a backend crash and watchdog detection/recovery,
    # preventing clients from seeing raw 500s from the dead backend.
    if not backend_ready or llama_process is None:
        retry_after = _self_heal_retry_after_seconds()
        headers_err = {"Retry-After": str(retry_after), "Cache-Control": "no-store"}
        payload = {
            "error": {
                "type": "backend_unavailable",
                "code": "backend_unavailable",
                "message": "Backend is not available, please retry later",
            },
            "status": 503,
            "path": f"/{path.lstrip('/')}",
            "retry_after": retry_after,
        }
        record_http_error("v1/chat/completions", "5xx", "backend_unavailable")
        return JSONResponse(status_code=503, content=payload, headers=headers_err)

    # Get request body
    body = await request.body()

    # Log request
    log_request(request, body, "local")

    # Parse body once and determine method/key/model for attribution
    try:
        body_json = json.loads(body) if body else {}
    except Exception:
        body_json = {}

    requested_model_name = None
    try:
        requested_model_name = body_json.get("model")
    except Exception:
        requested_model_name = None
    model_cfg = get_model_config(requested_model_name) if requested_model_name else get_model_config(current_model)
    force_full_prompt = _should_force_full_prompt(model_cfg)

    # ------------------------------------------------------------------
    # Session handling – incremental prompt ingestion
    # ------------------------------------------------------------------
    session_id_header, session_header_source = _resolve_session_id_header(request.headers)
    session_id: Optional[str] = None
    session_created = False
    delta_messages: Optional[List[Dict[str, Any]]] = None
    is_delta_request = False
    session_fallback_reason: Optional[str] = None
    original_message_count = 0

    if isinstance(body_json, dict) and "messages" in body_json:
        original_message_count = len(body_json["messages"])
        _log_session_header_resolution(session_id_header, session_header_source)
        try:
            session, session_created = await session_manager.get_or_create(
                session_id_header
            )
            session_id = session.session_id

            if not session_created and session.message_count > 0:
                delta_messages, history_matches = session_manager.compute_delta(
                    session.messages, body_json["messages"]
                )
                is_delta_request, session_fallback_reason = _classify_delta_routing(
                    history_matches=history_matches,
                    delta_message_count=len(delta_messages),
                    restore_confirmed=bool(session.restore_confirmed),
                    require_restore_signal=bool(
                        server_config.get("session_require_restore_signal", False)
                    ),
                    force_full_prompt=force_full_prompt,
                )

                if is_delta_request:
                    body_json["messages"] = list(delta_messages)
                    try:
                        _record_delta_payload_bytes(
                            len(json.dumps(delta_messages, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
                        )
                    except Exception:
                        pass
                    logger.info(
                        f"Session {session_id[:8]}... strict restore confirmed; "
                        f"forwarding delta ({len(delta_messages)} new messages)"
                    )
                else:
                    if session_fallback_reason == "history_mismatch":
                        _, slot_filename, _ = _build_slot_context(server_config, session_id)
                        await _invalidate_session_and_slot(
                            session_id,
                            "history_mismatch",
                            slot_filename,
                        )
                        session, session_created = await session_manager.get_or_create(
                            session_id
                        )
                    if session_fallback_reason:
                        _record_restore_fallback(session_fallback_reason)
                    logger.info(
                        f"Session {session_id[:8]}... history match={history_matches} "
                        f"delta_messages={len(delta_messages)} using full prompt "
                        f"reason={session_fallback_reason or 'none'}"
                    )
            elif session_created:
                session_fallback_reason = "no_existing_history"

            # Add session_id and cache_prompt to request body for llama-server
            body_json["cache_prompt"] = True
            body_json["session_id"] = session_id
            body = json.dumps(body_json).encode("utf-8")
        except Exception:
            # Session handling failed – fall back to full history
            logger.warning(
                "Session handling failed, falling back to full history",
                exc_info=True,
            )
            session_id = None
            is_delta_request = False
            session_fallback_reason = "session_handling_error"
    else:
        # Not a chat-completion request – no session handling
        session_id = None

    slot_id, slot_filename, slot_timeout = _build_slot_context(server_config, session_id)
    slot_enabled = slot_id is not None and slot_filename is not None

    method = request.method.upper()
    key = f"{method} {request.url.path} -> local"
    # determine model for token attribution (fallback to current_model)
    model_name = None
    try:
        model_name = body_json.get("model")
    except Exception:
        model_name = None
    if not model_name:
        model_name = current_model

    slot_model_name = _resolve_slot_model_name(model_name, current_model, server_config)

    # If router mode is enabled, translate model aliases to llama preset ids
    if server_config.get("llama_router_mode", False) and isinstance(body_json, dict):
        if slot_model_name and body_json.get("model") != slot_model_name:
            body_json["model"] = slot_model_name
            body = json.dumps(body_json).encode("utf-8")

    if slot_model_name:
        model_name = slot_model_name

    slot_model_payload = slot_model_name if server_config.get("llama_router_mode", False) else None

    single_flight_mode = server_config.get("session_single_flight_mode", "queue")
    single_flight_max_queue_depth = int(
        server_config.get("session_single_flight_max_queue_depth", 1) or 1
    )

    # Check concurrency limit before accepting request
    max_queries = server_config.get("max_concurrent_queries", 4)
    try:
        async with active_queries_lock:
            if active_queries >= max_queries:
                _record_backend_signal("concurrency_rejects")
                logger.warning(
                    "concurrency_reject active=%s max=%s path=%s",
                    active_queries,
                    max_queries,
                    path,
                )
                raise HTTPException(status_code=503, detail=f"Server overloaded: {active_queries} queries active. Retry later.")
    except HTTPException:
        raise
    except Exception:
        pass

    # Check llama-server slot availability for chat requests
    if path == "v1/chat/completions" or path.endswith("chat/completions"):
        try:
            # Determine model name for slot query
            slot_model = slot_model_name or model_name or current_model or "Qwen3"
            slots_url = f"http://localhost:{llama_port}/slots?model={slot_model}"
            client = _http_client if _http_client else httpx.AsyncClient(timeout=httpx.Timeout(5.0))
            slots_resp = await client.get(slots_url, timeout=5.0)
            if slots_resp.status_code == 200:
                slots_data = slots_resp.json()
                available_slots = 0
                total_slots = 0
                if isinstance(slots_data, list):
                    total_slots = len(slots_data)
                    available_slots = sum(1 for s in slots_data if not s.get("is_processing", True))
                if available_slots == 0 and total_slots > 0:
                    logger.warning(f"No available slots ({total_slots} total), rejecting request")
                    retry_after = int(server_config.get("slot_unavailable_retry_after", 5) or 5)
                    record_http_error("v1/chat/completions", "5xx", "slot_exhaustion")
                    return JSONResponse(
                        status_code=503,
                        content={
                            "error": {
                                "type": "server_busy",
                                "code": "no_slots_available",
                                "message": f"Model server busy: 0/{total_slots} slots available. Please retry later.",
                            },
                            "status": 503,
                            "retry_after": retry_after,
                            "total_slots": total_slots,
                            "available_slots": 0,
                        },
                        headers={"Retry-After": str(retry_after), "Cache-Control": "no-store"},
                    )
        except HTTPException:
            raise
        except Exception:
            # If slot check fails, proceed anyway (best effort)
            pass

    # Mark that a local query is active for status reporting
    try:
        async with active_queries_lock:
            active_queries += 1
    except Exception:
        pass

    # Token accounting: estimate tokens sent
    try:
        tokens_sent = 0
        # Chat-like payloads
        if isinstance(body_json, dict) and 'messages' in body_json:
            for m in body_json.get('messages', []):
                tokens_sent += count_text_tokens(str(m.get('content', '')), model_name)
        elif isinstance(body_json, dict) and 'input' in body_json:
            inp = body_json['input']
            if isinstance(inp, list):
                for it in inp:
                    tokens_sent += count_text_tokens(str(it), model_name)
            else:
                tokens_sent += count_text_tokens(str(inp), model_name)
        else:
            tokens_sent += count_text_tokens(body.decode('utf-8', errors='replace'), model_name)
    except Exception:
        tokens_sent = 0

    # schedule token increment
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_increment_tokens('sent', key, tokens_sent))
    except RuntimeError:
        asyncio.run(_increment_tokens('sent', key, tokens_sent))
    except Exception:
        pass
    
    # Forward headers (excluding host)
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    
    # Capture llama log cursor so we can attribute restore events during this request.
    llama_log_path = _resolve_log_path("llama")
    try:
        llama_log_offset = llama_log_path.stat().st_size if llama_log_path.exists() else 0
    except Exception:
        llama_log_offset = 0

    # Check if streaming is requested
    is_streaming = body_json.get("stream", False)
    
    # Compute request timeout - use adaptive timeout if enabled
    adaptive_enabled = server_config.get("llama_adaptive_timeout_enabled", False)
    if adaptive_enabled and body_json:
        base_timeout = float(server_config.get("llama_adaptive_timeout_base_seconds", 60))
        per_token_timeout = float(server_config.get("llama_adaptive_timeout_per_token_seconds", 0.01))
        max_timeout = float(server_config.get("llama_request_timeout", 300))
        timeout_seconds = _compute_adaptive_timeout(body_json, base_timeout, per_token_timeout, max_timeout)
        logger.debug(
            "Adaptive timeout: tokens=%d timeout=%.1fs",
            _estimate_prompt_tokens(body_json),
            timeout_seconds,
        )
    else:
        timeout_seconds = server_config.get("llama_request_timeout", 300)
    request_timeout = httpx.Timeout(timeout_seconds)
    
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
                            logger.info(
                                "slot_restore success session=%s slot=%s",
                                session_id[:8] if session_id else "unknown",
                                slot_id,
                            )
                    slot_save_allowed = slot_enabled

                    # Streaming response - client must stay open during streaming
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
                    backend_ready = True
                    restore_signal_detected = _has_explicit_restore_signal(dict(response.headers), None)
                    if session_id and not restore_signal_detected:
                        restore_signal_detected = _detect_restore_signal_from_llama_log(session_id)
                except Exception:
                    backend_ready = False
                    # If stream setup failed, ensure active_queries is decremented
                    try:
                        async with active_queries_lock:
                            active_queries = max(0, active_queries - 1)
                    except Exception:
                        pass
                    try:
                        await client.aclose()
                    except Exception:
                        pass
                    if _is_self_healing_active():
                        record_http_error("v1/chat/completions", "5xx", "self_healing")
                        return _self_healing_response(path)
                    # Return a 503 response indicating backend error instead of raising
                    record_http_error("v1/chat/completions", "5xx", "backend_error")
                    retry_after = _self_heal_retry_after_seconds()
                    headers_err = {"Retry-After": str(retry_after), "Cache-Control": "no-store"}
                    if session_id:
                        headers_err["X-Session-Id"] = session_id
                        headers_err["X-Session-Created"] = "true" if session_created else "false"
                        headers_err["X-Session-Delta"] = "true" if is_delta_request else "false"
                        if session_fallback_reason:
                            headers_err["X-Session-Fallback-Reason"] = session_fallback_reason
                    payload = {
                        "error": {
                            "type": "backend_error",
                            "code": "backend_error",
                            "message": "Backend unavailable, please retry later"
                        },
                        "status": 503,
                        "path": f"/{path}",
                        "retry_after": retry_after,
                    }
                    return JSONResponse(status_code=503, content=payload, headers=headers_err)
                upstream_status = response.status_code
                upstream_content_type = response.headers.get('content-type', '')

                # If upstream returned an error (or a non-SSE payload), return a buffered
                # response with the real status code so clients don't parse it as SSE.
                if upstream_status >= 400 or 'text/event-stream' not in upstream_content_type.lower():
                    try:
                        body_bytes = await response.aread()
                    except Exception:
                        body_bytes = b''
                    try:
                        await cm.__aexit__(None, None, None)
                    except Exception:
                        pass
                    try:
                        await client.aclose()
                    except Exception:
                        pass

                    err_headers = _normalize_outgoing_headers(dict(response.headers), buffered=True)
                    if session_id:
                        err_headers["X-Session-Id"] = session_id
                        err_headers["X-Session-Created"] = "true" if session_created else "false"
                        err_headers["X-Session-Delta"] = "true" if is_delta_request else "false"
                        if session_fallback_reason:
                            err_headers["X-Session-Fallback-Reason"] = session_fallback_reason
                    return Response(
                        content=body_bytes,
                        status_code=upstream_status,
                        headers=err_headers,
                    )

                # Normalize backend headers for streaming (remove content-length if TE present)
                outgoing_headers = _normalize_outgoing_headers(dict(response.headers), buffered=False)
                # Ensure Cache-Control is present
                if 'cache-control' not in {k.lower() for k in outgoing_headers.keys()}:
                    outgoing_headers['Cache-Control'] = 'no-cache'

                # Add session header for incremental ingestion
                if session_id:
                    outgoing_headers["X-Session-Id"] = session_id
                    outgoing_headers["X-Session-Created"] = "true" if session_created else "false"
                    outgoing_headers["X-Session-Delta"] = "true" if is_delta_request else "false"
                    if session_fallback_reason:
                        outgoing_headers["X-Session-Fallback-Reason"] = session_fallback_reason
                media_type = response.headers.get('content-type', 'text/event-stream')

                guardrail_reason: Optional[str] = None
                guardrail_response_text = ""
                completion_tokens_total = 0
                stream_start = time.monotonic()
                max_runtime_seconds = float(server_config.get("session_guardrail_max_runtime_seconds", 120) or 120)
                max_completion_tokens = int(server_config.get("session_guardrail_max_completion_tokens", 2048) or 2048)
                repetition_min_pattern_chars = int(
                    server_config.get("session_guardrail_repetition_min_pattern_chars", 64) or 64
                )
                repetition_min_repeats = int(
                    server_config.get("session_guardrail_repetition_min_repeats", 10) or 10
                )
                invalidate_on_guardrail = bool(
                    server_config.get("session_guardrail_invalidate_on_cutoff", True)
                )
                invalidate_on_repetition = server_config.get(
                    "session_guardrail_invalidate_on_repetition",
                    False,
                )

                async def stream_generator():
                    global active_queries
                    nonlocal guardrail_reason, guardrail_response_text, completion_tokens_total, slot_save_allowed
                    # Track assistant response for session history update
                    collected_content: list[str] = []
                    saw_done = False
                    saw_finish = False
                    try:
                        async for chunk in response.aiter_bytes():
                            # count tokens in this chunk (best-effort)
                            try:
                                chunk_text = chunk.decode('utf-8', errors='replace')
                                chunk_tokens = count_text_tokens(chunk_text, model_name)
                                delta_text = _extract_delta_text_from_sse_chunk(chunk_text)
                                if delta_text:
                                    completion_tokens_total += count_text_tokens(delta_text, model_name)
                                    guardrail_response_text = (guardrail_response_text + delta_text)[-2000:]
                                try:
                                    loop = asyncio.get_running_loop()
                                    loop.create_task(_increment_tokens('recv', key, chunk_tokens))
                                except RuntimeError:
                                    asyncio.run(_increment_tokens('recv', key, chunk_tokens))
                            except Exception:
                                chunk_text = ""
                                delta_text = ""

                            # Inspect SSE-style 'data:' lines for finish indicators
                            try:
                                txt = chunk.decode('utf-8', errors='replace')
                                for line in txt.splitlines():
                                    line = line.strip()
                                    if not line.startswith('data:'):
                                        continue
                                    payload = line[5:].strip()
                                    if payload == '[DONE]':
                                        saw_done = True
                                    else:
                                        try:
                                            j = json.loads(payload)
                                            for choice in j.get('choices', []):
                                                if choice.get('finish_reason') is not None:
                                                    saw_finish = True
                                        except Exception:
                                            # ignore non-json payloads
                                            pass
                            except Exception:
                                pass

                            if not guardrail_reason:
                                guardrail_reason = evaluate_stream_guardrail(
                                    runtime_seconds=time.monotonic() - stream_start,
                                    completion_tokens=completion_tokens_total,
                                    response_text=guardrail_response_text,
                                    max_runtime_seconds=max_runtime_seconds,
                                    max_completion_tokens=max_completion_tokens,
                                    repetition_min_pattern_chars=repetition_min_pattern_chars,
                                    repetition_min_repeats=repetition_min_repeats,
                                )
                                if guardrail_reason:
                                    _record_guardrail_cutoff(guardrail_reason)
                                    logger.warning(
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

                            # Collect content for session history if session is active
                            if session_id:
                                try:
                                    collected_content.append(chunk.decode('utf-8', errors='replace'))
                                except Exception:
                                    pass

                            yield chunk
                            log_response_chunk(chunk)
                    finally:
                        # If the upstream closed the stream without sending a final finish marker,
                        # synthesize a final SSE event so clients expecting a finish_reason get one.
                        try:
                            if not saw_done and not saw_finish:
                                try:
                                    finish_reason = "stop" if not guardrail_reason else "stop"
                                    final_obj = {"choices": [{"delta": {}, "finish_reason": finish_reason, "index": 0}]}
                                    final_bytes = (f"data: {json.dumps(final_obj)}\n\n").encode('utf-8')
                                    yield final_bytes
                                    log_response_chunk(final_bytes)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # Strict restore confirmation state comes from explicit backend signal.
                        if session_id:
                            try:
                                if not restore_signal_detected:
                                    restore_signal_detected = _detect_restore_signal_from_log_slice(
                                        llama_log_path,
                                        llama_log_offset,
                                    )
                                if restore_signal_detected:
                                    _record_restore_success()
                                await session_manager.set_restore_confirmed(session_id, restore_signal_detected)
                            except Exception:
                                logger.debug("Failed to set restore-confirmed state", exc_info=True)

                        # Update session history with the full conversation
                        if session_id and original_message_count > 0 and (collected_content or not guardrail_reason):
                            try:
                                if collected_content:
                                    full_response = ''.join(collected_content)
                                    assistant_content = _extract_assistant_content_from_sse(full_response)
                                    existing_messages = []
                                    if is_delta_request and delta_messages:
                                        session_obj = await session_manager.get(session_id)
                                        if session_obj:
                                            existing_messages = list(session_obj.messages)
                                    full_messages = merge_session_history_for_update(
                                        existing_messages=existing_messages,
                                        request_messages=list(body_json.get('messages', [])),
                                        delta_messages=delta_messages,
                                        is_delta_request=is_delta_request,
                                        assistant_content=assistant_content,
                                    )
                                    await session_manager.update_messages(session_id, full_messages)
                                else:
                                    if not is_delta_request and original_message_count > 0:
                                        await session_manager.update_messages(session_id, body_json.get('messages', []))
                                    elif is_delta_request and delta_messages:
                                        await session_manager.append_messages(session_id, delta_messages)
                            except Exception:
                                logger.debug(f"Failed to update session {session_id[:8]}... history", exc_info=True)

                        if slot_save_allowed and slot_enabled and upstream_status < 400:
                            saved = await _save_slot_snapshot(
                                llama_port,
                                slot_id,
                                slot_filename,
                                slot_timeout,
                                model=slot_model_payload,
                            )
                            if saved:
                                logger.info(
                                    "slot_save success session=%s slot=%s",
                                    session_id[:8] if session_id else "unknown",
                                    slot_id,
                                )

                        try:
                            await cm.__aexit__(None, None, None)
                        except Exception:
                            pass
                        try:
                            await client.aclose()
                        except Exception:
                            pass
                        # decrement active queries when streaming finishes
                        try:
                            async with active_queries_lock:
                                active_queries = max(0, active_queries - 1)
                        except Exception:
                            pass

                return StreamingResponse(
                    stream_generator(),
                    media_type=media_type,
                    headers=outgoing_headers,
                    status_code=upstream_status,
                )
        except SessionSingleFlightRejected as exc:
            try:
                async with active_queries_lock:
                    active_queries = max(0, active_queries - 1)
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
                            logger.info(
                                "slot_restore success session=%s slot=%s",
                                session_id[:8] if session_id else "unknown",
                                slot_id,
                            )
                    slot_save_allowed = slot_enabled

                    # Non-streaming response
                    try:
                        async with httpx.AsyncClient(timeout=request_timeout) as client:
                            method = request.method.lower()

                            async def _send_once():
                                return await getattr(client, method)(
                                    target_url,
                                    headers=headers,
                                    content=body,
                                )

                            try:
                                response = await _call_with_backend_retries(_send_once, path=path, stream=False)
                                backend_ready = True
                            except Exception:
                                backend_ready = False
                                if _is_self_healing_active():
                                    record_http_error("v1/chat/completions", "5xx", "self_healing")
                                    return _self_healing_response(path)
                                # Return a 503 response indicating backend error instead of raising
                                record_http_error("v1/chat/completions", "5xx", "backend_error")
                                retry_after = _self_heal_retry_after_seconds()
                                headers_err = {"Retry-After": str(retry_after), "Cache-Control": "no-store"}
                                if session_id:
                                    headers_err["X-Session-Id"] = session_id
                                    headers_err["X-Session-Created"] = "true" if session_created else "false"
                                    headers_err["X-Session-Delta"] = "true" if is_delta_request else "false"
                                    if session_fallback_reason:
                                        headers_err["X-Session-Fallback-Reason"] = session_fallback_reason
                                payload = {
                                    "error": {
                                        "type": "backend_error",
                                        "code": "backend_error",
                                        "message": "Backend unavailable, please retry later"
                                    },
                                    "status": 503,
                                    "path": f"/{path}",
                                    "retry_after": retry_after,
                                }
                                return JSONResponse(status_code=503, content=payload, headers=headers_err)

                            # Retry on empty response (no content and no tool call in reasoning_content)
                            response = await _call_with_empty_retry(_send_once, path=path)

                            recv_tokens = 0
                            # Non-streaming: count tokens in response body
                            try:
                                resp_text = response.content.decode('utf-8', errors='replace')
                                recv_tokens = count_text_tokens(resp_text, model_name)
                                try:
                                    loop = asyncio.get_running_loop()
                                    loop.create_task(_increment_tokens('recv', key, recv_tokens))
                                except RuntimeError:
                                    asyncio.run(_increment_tokens('recv', key, recv_tokens))
                            except Exception:
                                pass

                            # Note: Hard completion_tokens cutoff has been removed.
                            # Loop detection via repetition check is used instead.
                            # The max_completion_tokens config is now ignored.

                            # Update session history for non-streaming responses
                            if session_id and isinstance(body_json, dict) and 'messages' in body_json:
                                try:
                                    resp_content = response.content.decode('utf-8', errors='replace')
                                    resp_json = json.loads(resp_content) if resp_content else {}
                                    restore_signal_detected = _has_explicit_restore_signal(
                                        dict(response.headers),
                                        resp_json if isinstance(resp_json, dict) else None,
                                    )
                                    if session_id and not restore_signal_detected:
                                        restore_signal_detected = _detect_restore_signal_from_llama_log(session_id)
                                    if session_id and not restore_signal_detected:
                                        restore_signal_detected = _detect_restore_signal_from_log_slice(
                                            llama_log_path,
                                            llama_log_offset,
                                        )
                                    if restore_signal_detected:
                                        _record_restore_success()
                                    await session_manager.set_restore_confirmed(session_id, restore_signal_detected)
                                    assistant_content = _extract_assistant_content(resp_json)
                                    existing_messages = []
                                    if is_delta_request and delta_messages:
                                        session_obj = await session_manager.get(session_id)
                                        if session_obj:
                                            existing_messages = list(session_obj.messages)
                                    full_messages = merge_session_history_for_update(
                                        existing_messages=existing_messages,
                                        request_messages=list(body_json.get('messages', [])),
                                        delta_messages=delta_messages,
                                        is_delta_request=is_delta_request,
                                        assistant_content=assistant_content,
                                    )
                                    await session_manager.update_messages(session_id, full_messages)
                                except Exception:
                                    logger.debug(f"Failed to update session {session_id[:8]}... history", exc_info=True)

                            if slot_save_allowed and slot_enabled and response.status_code < 400:
                                saved = await _save_slot_snapshot(
                                    llama_port,
                                    slot_id,
                                    slot_filename,
                                    slot_timeout,
                                    model=slot_model_payload,
                                )
                                if saved:
                                    logger.info(
                                        "slot_save success session=%s slot=%s",
                                        session_id[:8] if session_id else "unknown",
                                        slot_id,
                                    )

                            log_response(response.status_code, response.content)

                            # Build response headers with session info
                            resp_headers = _normalize_outgoing_headers(dict(response.headers), buffered=True)
                            if session_id:
                                resp_headers["X-Session-Id"] = session_id
                                resp_headers["X-Session-Created"] = "true" if session_created else "false"
                                resp_headers["X-Session-Delta"] = "true" if is_delta_request else "false"
                                if session_fallback_reason:
                                    resp_headers["X-Session-Fallback-Reason"] = session_fallback_reason

                            return Response(
                                content=response.content,
                                status_code=response.status_code,
                                headers=resp_headers
                            )
                    finally:
                        # decrement active queries when non-streaming finishes or failures occur
                        try:
                            async with active_queries_lock:
                                active_queries = max(0, active_queries - 1)
                        except Exception:
                            pass
        except SessionSingleFlightRejected as exc:
            try:
                async with active_queries_lock:
                    active_queries = max(0, active_queries - 1)
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


async def proxy_to_remote(
    request: Request, 
    path: str, 
    model_config: dict
) -> Response:
    """Proxy request to remote API endpoint."""
    endpoint = model_config.get("endpoint", "")
    target_url = f"{endpoint}/{path}"
    
    # Get request body
    body = await request.body()
    
    # Log request
    log_request(request, body, "remote", endpoint)
    
    # Get API key
    api_key = None
    api_key_env = model_config.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env)
    if not api_key:
        api_key = model_config.get("api_key")
    
    # Forward headers
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    
    # Add API key
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Add custom headers from config
    custom_headers = model_config.get("headers", {})
    headers.update(custom_headers)
    
    body_json = json.loads(body) if body else {}
    # Determine model name for attribution (may be provided in body)
    model_name = None
    try:
        model_name = body_json.get('model')
    except Exception:
        model_name = None
    if not model_name:
        model_name = current_model or model_config.get('name') or model_config.get('id') or 'unknown'

    remote_timeout = httpx.Timeout(config.get("server", {}).get("llama_request_timeout", 300))
    is_streaming = body_json.get("stream", False)
    
    if is_streaming:
        # Streaming response - client must stay open during streaming
        client = httpx.AsyncClient(timeout=remote_timeout)
        cm = client.stream(
            request.method,
            target_url,
            headers=headers,
            content=body
        )

        response = await cm.__aenter__()
        upstream_status = response.status_code
        upstream_content_type = response.headers.get('content-type', '')

        # If upstream returned an error (or non-SSE payload), return a buffered
        # response with the real status code so clients can handle it as an API error.
        if upstream_status >= 400 or 'text/event-stream' not in upstream_content_type.lower():
            try:
                body_bytes = await response.aread()
            except Exception:
                body_bytes = b''
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass
            try:
                await client.aclose()
            except Exception:
                pass
            return Response(
                content=body_bytes,
                status_code=upstream_status,
                headers=_normalize_outgoing_headers(dict(response.headers), buffered=True),
            )

        outgoing_headers = _normalize_outgoing_headers(dict(response.headers), buffered=False)
        if 'cache-control' not in {k.lower() for k in outgoing_headers.keys()}:
            outgoing_headers['Cache-Control'] = 'no-cache'

        media_type = response.headers.get('content-type', 'text/event-stream')

        async def stream_generator():
            saw_done = False
            saw_finish = False
            try:
                async for chunk in response.aiter_bytes():
                    # parse OpenAI-style SSE chunks for delta content when possible
                    try:
                        s = chunk.decode('utf-8', errors='replace')
                        # look for lines starting with 'data: '
                        texts = []
                        for line in s.splitlines():
                            line = line.strip()
                            if not line.startswith('data:'):
                                continue
                            payload = line[5:].strip()
                            if payload == '[DONE]':
                                saw_done = True
                                continue
                            try:
                                j = json.loads(payload)
                                # detect finish_reason if present
                                for choice in j.get('choices', []):
                                    if choice.get('finish_reason') is not None:
                                        saw_finish = True
                                # extract any delta.content fields
                                for choice in j.get('choices', []):
                                    delta = choice.get('delta', {})
                                    if isinstance(delta, dict) and 'content' in delta:
                                        texts.append(str(delta.get('content', '')))
                            except Exception:
                                # fallback: treat payload as plain text
                                texts.append(payload)
                        if texts:
                            chunk_text = '\n'.join(texts)
                            chunk_tokens = count_text_tokens(chunk_text, model_name)
                            try:
                                loop = asyncio.get_running_loop()
                                loop.create_task(_increment_tokens('recv', f"{request.method.upper()} {request.url.path} -> remote", chunk_tokens))
                            except RuntimeError:
                                asyncio.run(_increment_tokens('recv', f"{request.method.upper()} {request.url.path} -> remote", chunk_tokens))
                    except Exception:
                        pass
                    yield chunk
                    log_response_chunk(chunk)
            finally:
                # If upstream closed without finish markers, synthesize a final SSE chunk
                try:
                    if not saw_done and not saw_finish:
                        try:
                            final_obj = {"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}
                            final_bytes = (f"data: {json.dumps(final_obj)}\n\n").encode('utf-8')
                            yield final_bytes
                            log_response_chunk(final_bytes)
                        except Exception:
                            pass
                    await cm.__aexit__(None, None, None)
                except Exception:
                    try:
                        await cm.__aexit__(None, None, None)
                    except Exception:
                        pass
                try:
                    await client.aclose()
                except Exception:
                    pass

        return StreamingResponse(
            stream_generator(),
            media_type=media_type,
            headers=outgoing_headers,
            status_code=upstream_status,
        )
    else:
        async with httpx.AsyncClient(timeout=remote_timeout) as client:
            method = request.method.lower()
            response = await getattr(client, method)(
                target_url,
                headers=headers,
                content=body
            )

            # Non-streaming: count tokens in response
            try:
                resp_text = response.content.decode('utf-8', errors='replace')
                # Use determined model_name (may be None)
                recv_tokens = count_text_tokens(resp_text, model_name)
                key = f"{request.method.upper()} {request.url.path} -> remote"
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(_increment_tokens('recv', key, recv_tokens))
                except RuntimeError:
                    asyncio.run(_increment_tokens('recv', key, recv_tokens))
            except Exception:
                pass

            log_response(response.status_code, response.content)

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=_normalize_outgoing_headers(dict(response.headers), buffered=True)
            )


def log_request(
    request: Request, 
    body: bytes, 
    target_type: str, 
    endpoint: str = "localhost"
):
    """Log incoming request."""
    try:
        body_str = body.decode("utf-8")[:2000] if body else ""
        logger.info(
            f"REQUEST [{target_type}] {request.method} {request.url.path} "
            f"-> {endpoint} | Body: {body_str}"
        )
    except Exception as e:
        logger.error(f"Error logging request: {e}")
    # Update request counts asynchronously
    try:
        path = request.url.path
        method = request.method.upper()

        # Determine model if available in body or current_model
        model_name = None
        try:
            body_json = json.loads(body) if body else {}
            model_name = body_json.get('model')
        except Exception:
            model_name = None

        if not model_name:
            model_name = current_model

        # Key: by endpoint only
        endpoint_key = f"{method} {path} -> {target_type}"

        # Increment endpoint key only
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_increment_count(endpoint_key))
        except RuntimeError:
            asyncio.run(_increment_count(endpoint_key))
    except Exception:
        pass


def log_response(status_code: int, content: bytes):
    """Log response."""
    try:
        content_str = content.decode("utf-8")[:2000] if content else ""
        logger.info(f"RESPONSE [{status_code}] | Body: {content_str}")
    except Exception as e:
        logger.error(f"Error logging response: {e}")


def log_response_chunk(chunk: bytes):
    """Log streaming response chunk.
    
    The ContentOnlyConsoleHandler extracts and displays just the content to console.
    Raw JSON is written to the log file only.
    """
    try:
        chunk_str = chunk.decode("utf-8")[:500] if chunk else ""
        logger.info(f"STREAM CHUNK | {chunk_str}")
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global config, logger, llama_process, _http_client, backend_watchdog_task, backend_ready, backend_recovery_state
    
    # Startup
    config = load_config()
    logger = setup_logging(config)
    logger.info("Starting LLama Proxy Server")
    backend_ready = False
    backend_recovery_state = {
        "in_progress": False,
        "attempt_timestamps": [],
        "max_attempts": int(config.get("server", {}).get("llama_self_heal_max_attempts", 3) or 3),
        "window_seconds": int(config.get("server", {}).get("llama_self_heal_window_seconds", 300) or 300),
        "retry_after_seconds": int(config.get("server", {}).get("llama_self_heal_retry_after_seconds", 30) or 30),
        "last_failure": None,
    }

    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(5.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
    )

    # One-time podman rootless state reset. After a reboot, crash-loop, or
    # when the service was previously run under incompatible systemd sandbox
    # settings, stale catatonit pause processes accumulate and leave podman
    # unable to create user namespaces ("invalid internal status",
    # "newuidmap: write to uid_map failed: Operation not permitted").
    #
    # The fix is to: (1) kill any stale pause processes, (2) run
    # `podman system migrate` to clean up internal state.
    # NOTE: this stops all running containers, so it must only run once here,
    # never inside the retry loop of start_llama_server.
    try:
        # Kill stale catatonit pause processes that accumulate during
        # crash-loop restarts and poison podman's namespace state.
        subprocess.run(
            ["pkill", "-9", "catatonit"],
            capture_output=True, timeout=5
        )
        time.sleep(1)  # let kernel clean up namespaces
        migrate_result = subprocess.run(
            ["podman", "system", "migrate"],
            capture_output=True, text=True, timeout=30
        )
        if migrate_result.returncode == 0:
            logger.info("podman system migrate completed successfully")
        else:
            logger.warning(
                f"podman system migrate returned {migrate_result.returncode}: "
                f"{migrate_result.stderr.strip()}"
            )
    except FileNotFoundError:
        logger.warning("podman not found in PATH, skipping system migrate")
    except Exception as e:
        logger.warning(f"podman system migrate failed: {e}")

    # Load the default model in a background task so uvicorn can finish
    # startup and begin accepting connections immediately. This avoids
    # blocking systemd (which waits for the lifespan to complete) for the
    # full model-load time (potentially 5+ minutes) and also handles the
    # boot-order race where distrobox/podman isn't ready yet.
    # Read default_model from config; default to gemma4 if not present
    default_model = config.get("default_model", "gemma4")
    router_mode = config.get("server", {}).get("llama_router_mode", False)
    router_preload = config.get("server", {}).get("llama_router_preload", [])
    router_preload_list = list(router_preload) if isinstance(router_preload, list) else []
    if router_mode:
        if "embeddings" not in router_preload_list:
            router_preload_list.append("embeddings")
        if default_model and default_model not in router_preload_list:
            router_preload_list.append(default_model)

    async def _load_default_model():
        """Load the default model with retries, running in the background."""
        global current_model, llama_process, backend_ready
        max_attempts = 6
        retry_delays = [0, 30, 60, 120, 240, 300]  # first attempt immediate
        for attempt, delay in enumerate(retry_delays[:max_attempts], 1):
            if delay > 0:
                logger.info(
                    f"Background retry {attempt}/{max_attempts} for "
                    f"default model '{default_model}' in {delay}s"
                )
                await asyncio.sleep(delay)
            # If something else loaded a model while we were waiting, stop
            if current_model is not None and not router_mode:
                logger.info("Model already loaded by another request, stopping background loader")
                return
            try:
                if router_mode:
                    if llama_process is None or llama_process.poll() is not None:
                        llama_process = start_llama_server(None)
                        if llama_process is None:
                            raise RuntimeError("Failed to start router-mode llama-server")
                    if not await wait_for_llama_server(config.get("server", {}).get("llama_startup_timeout", 300)):
                        raise RuntimeError("Router-mode llama-server failed to become ready")
                    backend_ready = True

                    resolved = []
                    if router_preload_list:
                        resolved = [get_local_model_name(name) or name for name in router_preload_list]
                        if not await router_preload_models(resolved):
                            raise RuntimeError(f"Router preload failed for {router_preload_list}")
                        logger.info(f"Router preload complete: {router_preload_list}")
                        if resolved:
                            current_model = resolved[0]
                    return

                if await ensure_model_loaded(default_model):
                    backend_ready = True
                    logger.info(f"Default model '{default_model}' loaded successfully")
                    return
            except Exception as e:
                logger.error(f"Exception loading default model (attempt {attempt}): {e}")
            logger.warning(f"Attempt {attempt}/{max_attempts} to load default model failed")
        logger.error(
            f"All attempts exhausted for default model '{default_model}'. "
            f"Model will be loaded on first matching request."
        )

    loop = asyncio.get_running_loop()
    loop.create_task(_load_default_model())

    if backend_watchdog_task is None:
        backend_watchdog_task = loop.create_task(_backend_watchdog_loop())

    # Load persisted request counts and token counts
    load_counts()
    load_token_counts()
    # Start background persist loops
    try:
        loop = asyncio.get_running_loop()
        global counts_persist_task, tokens_persist_task
        if counts_persist_task is None:
            counts_persist_task = loop.create_task(_counts_persist_loop())
        if tokens_persist_task is None:
            tokens_persist_task = loop.create_task(_tokens_persist_loop())
        global periodic_broadcast_task
        if periodic_broadcast_task is None:
            periodic_broadcast_task = loop.create_task(_periodic_broadcast_loop())
    except RuntimeError:
        pass

    # Start session manager cleanup task
    try:
        session_manager.start_cleanup_task()
    except Exception as e:
        logger.warning(f"Failed to start session cleanup task: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLama Proxy Server")
    # Stop session manager cleanup task
    try:
        session_manager.stop_cleanup_task()
    except Exception:
        pass
    if _http_client is not None:
        try:
            await _http_client.aclose()
        except Exception:
            pass
        _http_client = None

    if backend_watchdog_task is not None:
        backend_watchdog_task.cancel()
        try:
            await backend_watchdog_task
        except Exception:
            pass
        backend_watchdog_task = None

    backend_ready = False
    stop_llama_server()


app = FastAPI(
    title="LLama Proxy Server",
    description="Proxy server for routing OpenAI API requests",
    lifespan=lifespan
)

# Include handlers from the extracted handlers module
from . import handlers  # noqa: E402
app.include_router(handlers.router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return await _ui_index(request)

@app.get("/events")
async def status_events():
    return await _ui_status_events()

def _resolve_log_path(source: str = "proxy") -> Path:
    """Resolve the log file path for a given source.
    
    Args:
        source: Either 'proxy' for proxy.log or 'llama' for llama-server.log
        
    Returns:
        Path to the requested log file
    """
    if source == "llama":
        if log_dir:
            return log_dir / "llama-server.log"
        else:
            return Path(__file__).parent / "logs" / "llama-server.log"
    else:
        if log_dir:
            return log_dir / "proxy.log"
        else:
            return Path(__file__).parent / "logs" / "proxy.log"


@app.get("/logs/tail")
async def tail_logs(request: Request, lines: int = 100, source: str = "proxy"):
    return await _ui_tail_logs(request, lines, source)

@app.get("/logs")
async def view_logs(request: Request):
    return await _ui_view_logs(request)

@app.post("/v1/embeddings")
async def create_embeddings(request: Request):
    return await _ui_create_embeddings(request)

@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def proxy_openai_api(request: Request, path: str):
    return await _ui_proxy_openai_api(request, path)

@app.post("/admin/switch-model/{model_name}")
async def switch_model(model_name: str):
    return await _ui_switch_model(model_name)

def main():
    """Main entry point."""
    import uvicorn
    
    # Load config for server settings
    cfg = load_config()
    server_cfg = cfg.get("server", {})
    
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8000)
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


# ---------------------------------------------------------------------------
# Backward-compatibility re-exports for tests that import these from server
# ---------------------------------------------------------------------------
from .handlers import (  # noqa: E402, F401
    get_llama_local_status,
    health_check,
    list_models,
    prometheus_metrics,
    admin_metrics,
    admin_dump_counts,
    admin_stop_server,
    admin_reset_counts,
    admin_list_sessions,
    admin_delete_session,
    reload_config,
)
from .lifecycle import (  # noqa: E402, F401
    _self_heal_retry_after_seconds,
    _is_self_healing_active,
    _self_healing_response,
    _backend_recovery_snapshot,
    _worker_process_unhealthy,
    _prune_recovery_attempts,
    _attempt_router_self_heal,
    _backend_watchdog_loop,
    _inc_model_switch_refcount,
    _dec_model_switch_refcount,
    schedule_background_load,
    _model_loading_response,
    _is_retryable_backend_exception,
    _compute_retry_delay,
    _estimate_prompt_tokens,
    _compute_adaptive_timeout,
    _call_with_backend_retries,
    _probe_backend_reachable,
    get_model_config,
    _should_force_full_prompt,
    get_local_model_name,
    _resolve_slot_model_name,
    wait_for_llama_server,
    router_load_model,
    router_list_models,
    _extract_router_model_ids,
    router_is_model_loaded,
    router_wait_for_model,
    router_preload_models,
    start_llama_server,
    rotate_llama_logs,
    stop_llama_server,
    ensure_model_loaded,
    slot_polling_state,
    _slot_polling_tasks,
)
from .session import (  # noqa: E402, F401
    session_restore_observability,
    session_single_flight_observability,
    session_guardrail_observability,
    _record_restore_success,
    _record_restore_fallback,
    _record_delta_payload_bytes,
    _record_single_flight_queue,
    _record_single_flight_reject,
    _record_guardrail_cutoff,
    _record_session_invalidation,
    _detect_restore_signal_from_log_slice,
    _detect_restore_signal_from_llama_log,
    extract_streamed_content_from_chunk,
    ContentOnlyConsoleHandler,
    # Session coordination helpers
    _sanitize_session_id,
    _slot_id_for_session,
    _slot_filename_for_session,
    _build_slot_context,
    _call_slot_endpoint,
    _restore_slot_snapshot,
    _save_slot_snapshot,
    _ensure_slot_dir,
    _slot_persistence_enabled,
    _invalidate_session_and_slot,
    _should_cutoff_for_repetition,
    evaluate_stream_guardrail,
    _should_invalidate_on_guardrail,
    merge_session_history_for_update,
    _classify_delta_routing,
    _has_explicit_restore_signal,
    _resolve_session_id_header,
    _log_session_header_resolution,
    SlotLockCoordinator,
    slot_lock_coordinator,
    SessionSingleFlightRejected,
    SessionSingleFlightCoordinator,
    session_single_flight_coordinator,
)
from .observability import (  # noqa: E402, F401
    backend_signal_counts,
    _record_backend_signal,
    _classify_backend_exception,
    sse_clients,
    log_tail_clients,
    broadcast_status,
    broadcast_status_sync,
    _counts_file_path,
    load_counts,
    save_counts_sync,
    _token_file_path,
    load_token_counts,
    save_token_counts_sync,
    save_token_counts,
    save_counts,
    _counts_persist_loop,
    _tokens_persist_loop,
    query_llama_status,
    _periodic_broadcast_loop,
    _increment_count,
    _increment_count_multi,
    _increment_tokens,
)
from .utils import (  # noqa: E402, F401
    _get_tiktoken_encoding_for_model,
    count_text_tokens,
    _extract_tool_call_from_reasoning,
    _extract_assistant_content,
    _call_with_empty_retry,
    _is_empty_response,
    _extract_assistant_content_from_sse,
    _extract_delta_text_from_sse_chunk,
    _normalize_outgoing_headers,
    normalize_provider_name,
    load_config,
    setup_logging,
)

from .ui import (  # noqa: E402, F401
    index as _ui_index,
    status_events as _ui_status_events,
    tail_logs as _ui_tail_logs,
    view_logs as _ui_view_logs,
    create_embeddings as _ui_create_embeddings,
    proxy_openai_api as _ui_proxy_openai_api,
    switch_model as _ui_switch_model,
)


if __name__ == "__main__":
    main()
