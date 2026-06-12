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
# Session restore observability



def _ensure_slot_dir(slot_path: Optional[str]) -> Optional[Path]:
    if not slot_path:
        return None
    try:
        path = Path(slot_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return None


def _slot_persistence_enabled(slot_path: Optional[Path | str], slot_pool_size: int) -> bool:
    return bool(slot_path and slot_pool_size > 0)


async def _call_slot_endpoint(
    llama_port: int,
    slot_id: int,
    action: str,
    filename: str,
    timeout: float,
    model: Optional[str] = None,
) -> bool:
    if not filename:
        return False
    url = f"http://localhost:{llama_port}/slots/{slot_id}?action={action}"
    payload = {"filename": Path(filename).name}
    if model:
        payload["model"] = model
    client = _http_client if _http_client else httpx.AsyncClient(timeout=timeout)
    try:
        response = await client.post(url, json=payload, timeout=timeout)
        return getattr(response, "status_code", None) == 200
    except Exception as exc:
        logger.warning("slot_%s failed slot=%s error=%s", action, slot_id, exc)
        return False
    finally:
        if not _http_client:
            try:
                await client.aclose()
            except Exception:
                pass


async def _restore_slot_snapshot(
    llama_port: int,
    slot_id: int,
    filename: str,
    timeout: float,
    model: Optional[str] = None,
) -> bool:
    try:
        if not Path(filename).exists():
            return False
    except Exception:
        return False
    return await _call_slot_endpoint(
        llama_port,
        slot_id,
        "restore",
        filename,
        timeout,
        model=model,
    )


async def _save_slot_snapshot(
    llama_port: int,
    slot_id: int,
    filename: str,
    timeout: float,
    model: Optional[str] = None,
) -> bool:
    return await _call_slot_endpoint(
        llama_port,
        slot_id,
        "save",
        filename,
        timeout,
        model=model,
    )


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


async def _call_with_empty_retry(
    send_fn: Callable[[], Awaitable[httpx.Response]],
    path: str,
    max_retries: int = 2,
    retry_delay: float = 0.5,
) -> httpx.Response:
    """Call send_fn, retrying on empty responses up to max_retries times.

    A response is considered "empty" when it has no text content and no tool
    call embedded in reasoning_content (e.g. the model returned only thinking
    output without actual content). Retries use retry_delay seconds between
    attempts. Session context is preserved across retries because send_fn
    captures the same headers/body.

    Returns the first non-empty response, or the last response if all retries
    are exhausted.
    """
    for attempt in range(max_retries + 1):
        response = await _call_with_backend_retries(send_fn, path=path, stream=False)
        try:
            content = response.content.decode('utf-8', errors='replace')
            resp_json = json.loads(content) if content else {}
        except Exception:
            return response  # not valid JSON or content not readable, use as-is

        if _is_empty_response(content or "", resp_json):
            if attempt < max_retries:
                logger.info(
                    "Empty response detected on attempt %s/%s, retrying...",
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(retry_delay)
            else:
                logger.warning(
                    "Empty response persisted after %s retries, returning empty response",
                    max_retries,
                )
        else:
            if attempt > 0:
                logger.info(
                    "Retry attempt %s produced non-empty response",
                    attempt + 1,
                )
            return response

    return response


def _sanitize_session_id(session_id: str) -> str:
    if not session_id:
        return ""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", session_id)


def _slot_id_for_session(session_id: str, pool_size: int) -> Optional[int]:
    if not session_id or pool_size <= 0:
        return None
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % int(pool_size)


def _slot_filename_for_session(session_id: str, base_dir: Path | str) -> str:
    safe_id = _sanitize_session_id(session_id)
    return str(Path(base_dir) / f"slot_{safe_id}.bin")


def _build_slot_context(
    server_config: dict,
    session_id: Optional[str],
) -> tuple[Optional[int], Optional[str], float]:
    slot_path = server_config.get("session_slot_save_path")
    slot_pool_size = int(server_config.get("session_slot_pool_size", 0) or 0)
    slot_timeout = float(server_config.get("session_slot_timeout_seconds", 3.0) or 3.0)
    slot_dir = _ensure_slot_dir(slot_path)
    if not session_id or not _slot_persistence_enabled(slot_dir, slot_pool_size):
        return None, None, slot_timeout
    slot_id = _slot_id_for_session(session_id, slot_pool_size)
    if slot_id is None:
        return None, None, slot_timeout
    return slot_id, _slot_filename_for_session(session_id, slot_dir), slot_timeout


async def _invalidate_session_and_slot(
    session_id: Optional[str],
    reason: str,
    slot_filename: Optional[str],
) -> None:
    if session_id:
        try:
            await session_manager.invalidate(session_id)
        except Exception:
            pass
    if reason:
        _record_session_invalidation(reason)
    if slot_filename:
        try:
            Path(slot_filename).unlink(missing_ok=True)
        except Exception:
            pass


class SlotLockCoordinator:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._locks: dict[int, asyncio.Lock] = {}

    def acquire(self, slot_id: Optional[int]):
        @asynccontextmanager
        async def _guard():
            if slot_id is None:
                yield
                return
            async with self._lock:
                lock = self._locks.get(slot_id)
                if lock is None:
                    lock = asyncio.Lock()
                    self._locks[slot_id] = lock
            await lock.acquire()
            try:
                yield
            finally:
                lock.release()

        return _guard()


slot_lock_coordinator = SlotLockCoordinator()


class SessionSingleFlightRejected(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class SessionSingleFlightCoordinator:
    def __init__(self) -> None:
        self._state_lock = asyncio.Lock()
        self._states: dict[str, dict[str, Any]] = {}

    async def _get_state(self, session_id: str) -> dict[str, Any]:
        async with self._state_lock:
            state = self._states.get(session_id)
            if state is None:
                state = {"lock": asyncio.Lock(), "waiters": 0, "active": False}
                self._states[session_id] = state
            return state

    def acquire(self, session_id: Optional[str], mode: str, max_queue_depth: int):
        @asynccontextmanager
        async def _guard():
            if not session_id:
                yield
                return

            state = await self._get_state(session_id)
            mode_norm = (mode or "queue").strip().lower()
            if mode_norm not in {"queue", "reject"}:
                mode_norm = "queue"

            is_waiting = False
            async with self._state_lock:
                if state["lock"].locked():
                    if mode_norm == "reject":
                        _record_single_flight_reject()
                        raise SessionSingleFlightRejected("active_inflight")
                    if max_queue_depth is not None and state["waiters"] >= max_queue_depth:
                        _record_single_flight_reject()
                        raise SessionSingleFlightRejected("queue_full")
                    state["waiters"] += 1
                    is_waiting = True
                    _record_single_flight_queue()

            await state["lock"].acquire()
            async with self._state_lock:
                if is_waiting:
                    state["waiters"] = max(0, state["waiters"] - 1)
                state["active"] = True
                session_single_flight_observability["active_sessions_current"] = sum(
                    1 for s in self._states.values() if s.get("active")
                )
                session_single_flight_observability["queue_depth_current"] = sum(
                    int(s.get("waiters", 0)) for s in self._states.values()
                )

            try:
                yield
            finally:
                state["lock"].release()
                async with self._state_lock:
                    state["active"] = False
                    if not state["lock"].locked() and state["waiters"] == 0:
                        self._states.pop(session_id, None)
                    session_single_flight_observability["active_sessions_current"] = sum(
                        1 for s in self._states.values() if s.get("active")
                    )
                    session_single_flight_observability["queue_depth_current"] = sum(
                        int(s.get("waiters", 0)) for s in self._states.values()
                    )

        return _guard()

    def metrics_snapshot(self) -> dict:
        return dict(session_single_flight_observability)


def _should_cutoff_for_repetition(
    response_text: str,
    min_pattern_chars: int,
    min_repeats: int,
) -> bool:
    if not response_text:
        return False
    pattern_len = max(1, int(min_pattern_chars))
    repeats = max(2, int(min_repeats))
    tail_len = pattern_len * repeats
    if len(response_text) < tail_len:
        return False
    tail = response_text[-tail_len:]
    pattern = tail[-pattern_len:]
    if not pattern.strip():
        return False
    return tail == pattern * repeats


def evaluate_stream_guardrail(
    runtime_seconds: float,
    completion_tokens: int,
    response_text: str,
    max_runtime_seconds: Optional[float],
    max_completion_tokens: Optional[int],
    repetition_min_pattern_chars: int,
    repetition_min_repeats: int,
) -> Optional[str]:
    """Evaluate whether the stream should be stopped due to guardrail violations.

    Priority order:
    1. Runtime cutoff - indicates a true runaway loop
    2. Repetition detection - indicates the model is stuck in a loop

    Note: The hard completion_tokens cutoff has been removed. Legitimate long
    responses should not be cut off. Loop detection via repetition check is
    used instead to catch runaway generation.
    """
    if max_runtime_seconds and runtime_seconds >= max_runtime_seconds:
        return "runtime"
    if _should_cutoff_for_repetition(response_text, repetition_min_pattern_chars, repetition_min_repeats):
        return "repetition"
    return None


def _should_invalidate_on_guardrail(
    guardrail_reason: Optional[str],
    invalidate_on_cutoff: bool,
    invalidate_on_repetition: bool,
) -> bool:
    """Determine whether a session should be invalidated due to a guardrail.

    By default:
    - "runtime" guardrail invalidates the session (indicates true runaway loop)
    - "repetition" guardrail does NOT invalidate by default (let client retry)
    - "completion_tokens" guardrail never invalidates (removed in favor of loop detection)
    """
    if not guardrail_reason:
        return False
    if guardrail_reason == "repetition":
        return bool(invalidate_on_repetition)
    # completion_tokens reason should never cause invalidation
    # (it's no longer a guardrail reason, but handle defensively)
    if guardrail_reason == "completion_tokens":
        return False
    return bool(invalidate_on_cutoff)


def merge_session_history_for_update(
    existing_messages: List[Dict[str, Any]],
    request_messages: List[Dict[str, Any]],
    delta_messages: Optional[List[Dict[str, Any]]],
    is_delta_request: bool,
    assistant_content: Optional[str],
) -> List[Dict[str, Any]]:
    if is_delta_request and delta_messages:
        merged = list(existing_messages) + list(delta_messages)
    else:
        merged = list(request_messages)

    if assistant_content:
        if not merged or merged[-1].get("role") != "assistant" or merged[-1].get("content") != assistant_content:
            merged.append({"role": "assistant", "content": assistant_content})
    return merged


# Single-flight coordinator for per-session concurrency
session_single_flight_coordinator = SessionSingleFlightCoordinator()


def _classify_delta_routing(
    history_matches: bool,
    delta_message_count: int,
    restore_confirmed: bool,
    require_restore_signal: bool = True,
    force_full_prompt: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Decide whether to use delta routing.

    When ``require_restore_signal`` is True, delta routing requires explicit
    restore confirmation from backend signals/logs.
    """
    if not history_matches:
        return False, "history_mismatch"
    if delta_message_count <= 0:
        return False, "no_new_messages"
    if force_full_prompt:
        return False, "delta_disabled"
    if require_restore_signal and not restore_confirmed:
        return False, "missing_restore_signal"
    return True, None


def _has_explicit_restore_signal(
    response_headers: Dict[str, str],
    response_json: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return True only when explicit backend restore evidence is present."""
    header_candidates = {
        "x-llama-session-restored",
        "x-session-restored",
        "x-llama-cache-restored",
        "x-kv-cache-restored",
        "x-cache-restored",
    }
    for key, value in response_headers.items():
        if key.lower() not in header_candidates:
            continue
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "restored", "hit"}:
            return True

    if isinstance(response_json, dict):
        for field in (
            "session_restored",
            "cache_restored",
            "restore_success",
            "kv_cache_restored",
        ):
            if response_json.get(field) is True:
                return True
    return False





log_dir: Optional[Path] = None
logger: logging.Logger = logging.getLogger("llama-proxy")




# SSE clients for real-time status updates





def _log_session_header_resolution(
    session_id_header: Optional[str],
    header_source: Optional[str],
) -> None:
    """Log whether a session header was provided on the request."""
    try:
        if header_source:
            prefix = session_id_header[:8] if session_id_header else "unknown"
            logger.info(
                "Session header resolved: source=%s session=%s...",
                header_source,
                prefix,
            )
        else:
            logger.info("No session header provided; proxy will generate session id")
    except Exception:
        pass


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
    """Serve the index page with API documentation."""
    # Build models table rows and quick link buttons for local models
    models_rows = ""
    model_buttons = ""
    model_options = ""
    for name, cfg in config.get("models", {}).items():
        model_type = cfg.get("type", "unknown")
        aliases = ", ".join(cfg.get("aliases", [])) or "—"
        endpoint = cfg.get("endpoint", "localhost:8080") if model_type == "remote" else "Local llama-server"
        type_badge = f'<span class="badge badge-local">Local</span>' if model_type == "local" else f'<span class="badge badge-remote">Remote</span>'
        
        # Build model dropdown options
        # Consider both the config key (name) and the underlying llama_model when
        # deciding which option is selected so UIs that compare against the
        # resolved llama-server id still show the correct active model.
        selected = ""
        try:
            lm = cfg.get("llama_model", name)
            if name == current_model or lm == current_model:
                selected = "selected"
        except Exception:
            selected = "selected" if name == current_model else ""
        type_label = "Local" if model_type == "local" else "Remote"
        model_options += f'<option value="{name}" {selected}>{name} ({type_label})</option>'
        
        # Add switch button for local models that aren't currently loaded
        action_cell = ""
        if model_type == "local":
            llama_model = cfg.get("llama_model", name)
            # Consider model active when either the user-visible name or the
            # resolved llama_model matches the current_model state.
            if llama_model != current_model and name != current_model:
                action_cell = f'<button class="btn-switch" onclick="switchModel(\'{name}\')">Load Model</button>'
                model_buttons += f'<button class="btn-switch btn-model" onclick="switchModel(\'{name}\')">Load {name}</button>'
            else:
                action_cell = '<span class="badge badge-active">Active</span>'
        
        models_rows += f"""
        <tr>
            <td><code>{name}</code></td>
            <td>{type_badge}</td>
            <td><code>{aliases}</code></td>
            <td>{endpoint}</td>
            <td>{action_cell}</td>
        </tr>"""

    # Build list of local model names for JavaScript
    import json
    local_model_names = [name for name, cfg in config.get("models", {}).items() if cfg.get("type") == "local"]
    local_model_names_json = json.dumps(local_model_names)

    router_mode = config.get("server", {}).get("llama_router_mode", False)
    router_models = None
    if router_mode:
        router_models = await router_list_models()

    # Prefer configured provider host when present (e.g. Tailscale mapping)
    providers_cfg = config.get('providers') if isinstance(config.get('providers'), dict) else {}
    proxy_cfg = providers_cfg.get('Proxy') if providers_cfg else None
    provider_host = None
    if isinstance(proxy_cfg, dict):
        provider_host = proxy_cfg.get('host') or proxy_cfg.get('url') or proxy_cfg.get('base')
    provider_host_html = f'<div class="status-item"><strong>Provider:</strong> <code id="providerHost">{provider_host}</code></div>' if provider_host else ''
    # Base URL from incoming request (includes scheme and host:port)
    base = provider_host.rstrip('/') if provider_host else str(request.base_url).rstrip('/')

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLama Proxy Server</title>
    <style>
        :root {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #1f2940;
            --text-primary: #eee;
            --text-secondary: #aaa;
            --accent: #4f8cff;
            --accent-hover: #6ba1ff;
            --success: #4caf50;
            --warning: #ff9800;
            --border: #2a3a5a;
            --log-error: #b00020;
            --log-warning: #f59e0b;
            --log-info: #60a5fa;
            --log-debug: #9ca3af;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--accent), #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .subtitle {{ color: var(--text-secondary); margin-bottom: 2rem; font-size: 1.1rem; }}
        .status-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 1.5rem;
            background: var(--bg-card);
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
        }}
        .status-item {{ display: flex; align-items: center; gap: 0.5rem; }}
        .status-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        .card {{
            background: var(--bg-card);
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border);
        }}
        .card h2 {{
            font-size: 1.3rem;
            margin-bottom: 1rem;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        .card-header h2 {{
            margin-bottom: 0;
        }}
        .model-select-wrapper {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .endpoint-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }}
        .endpoint {{
            background: var(--bg-secondary);
            padding: 1rem;
            border-radius: 6px;
            border-left: 3px solid var(--accent);
        }}
        .endpoint-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.5rem;
        }}
        .method {{
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            text-transform: uppercase;
        }}
        .method-get {{ background: #2e7d32; color: #fff; }}
        .method-post {{ background: #1565c0; color: #fff; }}
        .endpoint-path {{ font-family: monospace; font-size: 0.95rem; color: var(--text-primary); }}
        .endpoint-desc {{ font-size: 0.85rem; color: var(--text-secondary); }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        th, td {{
            text-align: left;
            padding: 0.75rem;
            border-bottom: 1px solid var(--border);
        }}
        th {{ color: var(--text-secondary); font-weight: 500; font-size: 0.85rem; text-transform: uppercase; }}
        .badge {{
            font-size: 0.75rem;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-weight: 500;
        }}
        .badge-local {{ background: #2e7d32; color: #fff; }}
        .badge-remote {{ background: #7b1fa2; color: #fff; }}
        code {{
            background: var(--bg-primary);
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-size: 0.85rem;
        }}
        pre {{
            background: var(--bg-primary);
            padding: 1rem;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 0.85rem;
            margin-top: 1rem;
        }}
        .nav-links {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 2rem;
        }}
        .nav-links a {{
            color: var(--accent);
            text-decoration: none;
            padding: 0.5rem 1rem;
            background: var(--bg-card);
            border-radius: 6px;
            border: 1px solid var(--border);
            transition: all 0.2s;
        }}
        .nav-links a:hover {{
            background: var(--accent);
            color: #fff;
            border-color: var(--accent);
        }}
        .btn-model {{
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }}
        .section-title {{
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }}
        .btn-switch {{
            background: var(--accent);
            color: #fff;
            border: none;
            padding: 0.4rem 0.8rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
            font-weight: 500;
            transition: all 0.2s;
        }}
        .btn-switch:hover {{
            background: var(--accent-hover);
        }}
        .btn-switch:disabled {{
            background: var(--text-secondary);
            cursor: not-allowed;
        }}
        .badge-active {{
            background: var(--success);
            color: #fff;
            font-size: 0.75rem;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-weight: 500;
        }}
        .status-message {{
            position: fixed;
            top: 1rem;
            right: 1rem;
            padding: 1rem 1.5rem;
            border-radius: 6px;
            font-weight: 500;
            z-index: 1000;
            display: none;
        }}
        .status-message.success {{
            background: var(--success);
            color: #fff;
        }}
        .status-message.error {{
            background: #d32f2f;
            color: #fff;
        }}
        .status-message.loading {{
            background: var(--warning);
            color: #000;
        }}
        .quick-test {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-top: 1rem;
        }}
        .test-input-area, .test-output-area {{
            display: flex;
            flex-direction: column;
        }}
        .test-input-area label, .test-output-area label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }}
        .test-input {{
            width: 100%;
            min-height: 120px;
            padding: 0.75rem;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.9rem;
            resize: vertical;
        }}
        .test-input:focus {{
            outline: none;
            border-color: var(--accent);
        }}
        .test-output {{
            width: 100%;
            min-height: 120px;
            max-height: 300px;
            padding: 0.75rem;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            font-family: monospace;
            font-size: 0.85rem;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .test-hint {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
        }}
        .test-status {{
            font-size: 0.8rem;
            margin-top: 0.5rem;
            color: var(--text-secondary);
        }}
        .test-status.streaming {{
            color: var(--success);
        }}
        .test-status.error {{
            color: #d32f2f;
        }}
        .btn-test {{
            background: transparent;
            color: var(--accent);
            border: 1px solid var(--accent);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.7rem;
            font-weight: 500;
            margin-left: auto;
            transition: all 0.2s;
        }}
        .btn-test:hover {{
            background: var(--accent);
            color: #fff;
        }}
        .api-test-section {{
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid var(--accent);
        }}
        .api-test-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            color: var(--accent);
        }}
        .model-select-label {{
            color: var(--text-secondary);
            font-size: 0.85rem;
        }}
        .model-select {{
            background: var(--bg-primary);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 0.4rem 0.6rem;
            font-size: 0.85rem;
            cursor: pointer;
        }}
        .model-select:focus {{
            outline: none;
            border-color: var(--accent);
        }}
        .btn-close {{
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 1.5rem;
            cursor: pointer;
            line-height: 1;
            padding: 0 0.25rem;
        }}
        .btn-close:hover {{
            color: var(--text-primary);
        }}
        .api-test-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }}
        .api-test-input-area, .api-test-output-area {{
            display: flex;
            flex-direction: column;
        }}
        .api-test-input-area label, .api-test-output-area label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }}
        .api-test-pre {{
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.75rem;
            font-family: monospace;
            font-size: 0.8rem;
            color: var(--text-primary);
            overflow: auto;
            max-height: 300px;
            min-height: 100px;
            white-space: pre-wrap;
            word-wrap: break-word;
            margin: 0;
        }}
        .stats-panel {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 2rem;
            overflow: hidden;
        }}
        .stats-panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 1rem;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            font-weight: 500;
            color: var(--accent);
        }}
        .btn-close-stats {{
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 1.2rem;
            cursor: pointer;
            padding: 0 0.25rem;
            line-height: 1;
        }}
        .btn-close-stats:hover {{
            color: var(--text-primary);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1px;
            background: var(--border);
        }}
        .stats-item {{
            display: flex;
            flex-direction: column;
            padding: 0.75rem 1rem;
            background: var(--bg-card);
        }}
        .stats-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.25rem;
        }}
        .stats-value {{
            font-size: 0.95rem;
            color: var(--text-primary);
            font-family: monospace;
        }}
        .stats-unknown {{
            color: var(--warning);
            font-style: italic;
            cursor: help;
        }}
        .stats-toggle {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 0.25rem 0.5rem;
            font-size: 0.75rem;
            color: var(--accent);
            cursor: pointer;
            margin-left: 1rem;
        }}
        .stats-toggle:hover {{
            background: var(--accent);
            color: #fff;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>LLama Proxy Server</h1>
        <p class="subtitle">OpenAI-compatible API proxy for local and remote LLM models</p>

        <div class="status-bar">
            <div class="status-item">
                <div class="status-dot"></div>
                <span>Proxy Running</span>
            </div>
            {provider_host_html}
            <div class="status-item">
                <strong>Current Model:</strong>
                <code id="currentModelStatus">{current_model or 'None'}</code>
            </div>
            <div class="status-item" id="routerModeStatus" data-router-mode="{'true' if router_mode else 'false'}" style="display: {'flex' if router_mode else 'none'};">
                <strong>Router:</strong>
                <span id="routerModeLabel">{'Enabled' if router_mode else 'Disabled'}</span>
            </div>
            <div class="status-item">
                <strong>llama-server:</strong>
                <span id="llamaServerStatus">{'Running' if llama_process and llama_process.poll() is None else 'Stopped'}</span>
            </div>
        </div>

        <div id="statsPanel" class="stats-panel" style="display: none;">
            <div class="stats-panel-header">
                <span>Model Statistics</span>
                <button class="btn-close-stats" onclick="toggleStatsPanel()">&times;</button>
            </div>
            <div class="stats-grid">
                <div class="stats-item">
                    <span class="stats-label">Model</span>
                    <span class="stats-value" id="statsModel">-</span>
                </div>
                <div class="stats-item">
                    <span class="stats-label">Llama-server status</span>
                    <span class="stats-value" id="statsLlamaStatus">-</span>
                </div>
                <div class="stats-item">
                    <span class="stats-label">Max context</span>
                    <span class="stats-value" id="statsNCtx">
                        <span class="stats-val">-</span>
                        <span class="stats-unknown" title="Value not available from backend" style="display:none;">unknown</span>
                    </span>
                </div>
                <div class="stats-item">
                    <span class="stats-label">KV cache tokens</span>
                    <span class="stats-value" id="statsKvCache">
                        <span class="stats-val">-</span>
                        <span class="stats-unknown" title="Value not available from backend" style="display:none;">unknown</span>
                    </span>
                </div>
                <div class="stats-item">
                    <span class="stats-label">Total tokens sent</span>
                    <span class="stats-value" id="statsTokensSent">0</span>
                </div>
                <div class="stats-item">
                    <span class="stats-label">Total tokens received</span>
                    <span class="stats-value" id="statsTokensRecv">0</span>
                </div>
            </div>
        </div>

        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            <p class="section-title" style="margin-bottom: 0;">Quick Links</p>
            <button class="stats-toggle" onclick="toggleStatsPanel()">Show Model Stats</button>
        </div>
        <div class="nav-links">
            <a href="/health">Health Check</a>
            <a href="/v1/models">List Models</a>
            <a href="/docs">OpenAPI Docs</a>
            <a href="/redoc">ReDoc</a>
            <a href="/logs">View Logs</a>
            {model_buttons}
        </div>

        <div class="card">
            <h2>Quick Test</h2>
            <p style="color: var(--text-secondary); margin-bottom: 0.5rem;">
                Send a message to test the current model. Press Enter to send (Shift+Enter for new line).
            </p>
            <div class="quick-test">
                <div class="test-input-area">
                    <label>Input</label>
                    <textarea id="testInput" class="test-input" placeholder="Type your message here..."></textarea>
                    <p class="test-hint">Press Enter to send, Shift+Enter for new line</p>
                </div>
                <div class="test-output-area">
                    <label>Response</label>
                    <div id="testOutput" class="test-output"></div>
                    <p id="testStatus" class="test-status"></p>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Configured Models</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model ID</th>
                        <th>Type</th>
                        <th>Aliases (supports wildcards)</th>
                        <th>Endpoint</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {models_rows}
                </tbody>
            </table>
        </div>

        <div class="card">
            <div class="card-header">
                <h2>API Passthrough Endpoints</h2>
                <div class="model-select-wrapper">
                    <label for="modelSelect" class="model-select-label">Test with model:</label>
                    <select id="modelSelect" class="model-select" onchange="updateTestRequest()">
                        {model_options}
                    </select>
                </div>
            </div>
            <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                These endpoints are fully compatible with the OpenAI API. Requests are automatically routed to local llama-server or remote APIs based on the model specified. Click "Test" to try each endpoint.
            </p>
            <div class="endpoint-grid">
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method method-post">POST</span>
                        <span class="endpoint-path">/v1/chat/completions</span>
                        <button class="btn-test" onclick="testEndpoint('chat')">Test</button>
                    </div>
                    <p class="endpoint-desc">Chat completions - send messages and get AI responses</p>
                </div>
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method method-post">POST</span>
                        <span class="endpoint-path">/v1/completions</span>
                        <button class="btn-test" onclick="testEndpoint('completions')">Test</button>
                    </div>
                    <p class="endpoint-desc">Text completions - complete a prompt</p>
                </div>
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method method-get">GET</span>
                        <span class="endpoint-path">/v1/models</span>
                        <button class="btn-test" onclick="testEndpoint('models')">Test</button>
                    </div>
                    <p class="endpoint-desc">List all available models</p>
                </div>
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method method-post">POST</span>
                        <span class="endpoint-path">/v1/embeddings</span>
                        <button class="btn-test" onclick="testEndpoint('embeddings')">Test</button>
                    </div>
                    <p class="endpoint-desc">Generate embeddings for text</p>
                </div>
            </div>
            
            <div id="apiTestSection" class="api-test-section" style="display: none; margin-top: 1.5rem;">
                <div class="api-test-header">
                    <strong id="apiTestTitle">Test Request</strong>
                    <button class="btn-close" onclick="closeApiTest()">&times;</button>
                </div>
                <div class="api-test-grid">
                    <div class="api-test-input-area">
                        <label>Request</label>
                        <pre id="apiTestRequest" class="api-test-pre"></pre>
                    </div>
                    <div class="api-test-output-area">
                        <label>Response</label>
                        <pre id="apiTestResponse" class="api-test-pre"></pre>
                        <p id="apiTestStatus" class="test-status"></p>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Admin Endpoints</h2>
            <div class="endpoint-grid">
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method method-get">GET</span>
                        <span class="endpoint-path">/health</span>
                        <button class="btn-switch" style="margin-left:auto;" onclick="refreshStatus()">Refresh</button>
                    </div>
                    <p class="endpoint-desc">Health check - returns server and model status</p>
                </div>
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method method-post">POST</span>
                        <span class="endpoint-path">/admin/reload-config</span>
                        <button class="btn-switch" style="margin-left:auto;" onclick="reloadConfig()">Reload</button>
                    </div>
                    <p class="endpoint-desc">Reload configuration from config.yaml</p>
                </div>
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method method-post">POST</span>
                        <span class="endpoint-path">/admin/switch-model/{{model}}</span>
                        <button class="btn-switch" style="margin-left:auto;" onclick="adminSwitchModel()">Switch To Selected</button>
                    </div>
                    <p class="endpoint-desc">Switch the llama-server to the model selected in the dropdown above</p>
                </div>
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method method-post">POST</span>
                        <span class="endpoint-path">/admin/stop-server</span>
                        <button class="btn-switch" style="margin-left:auto; background:#d32f2f;" onclick="stopServer()">Stop</button>
                    </div>
                    <p class="endpoint-desc">Stop the llama-server process (requires confirmation)</p>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Model Routing</h2>
            <p style="color: var(--text-secondary);">
                The proxy automatically routes requests based on the <code>model</code> parameter in your API request:
            </p>
            <ul style="margin: 1rem 0 0 1.5rem; color: var(--text-secondary);">
                <li><strong>Local models</strong> are served by llama-server running in a distrobox container</li>
                <li><strong>Remote models</strong> are proxied to external APIs (OpenAI, Anthropic, etc.)</li>
                <li><strong>Wildcard aliases</strong> like <code>gpt-*</code> match any model starting with that prefix</li>
                <li>If a model switch is needed, the server will automatically load the new model</li>
            </ul>
        </div>
        
        <div class="card">
            <h2>API Endpoints</h2>
            <pre style="background:var(--bg-primary); padding:1rem; border-radius:6px; color:var(--text-secondary);">==========================================
API Endpoints
==========================================

  Health check:     GET  {base}/health
  List models:      GET  {base}/v1/models
  Chat completions: POST {base}/v1/chat/completions
  Completions:      POST {base}/v1/completions

  Admin endpoints:
    Reload config:  POST {base}/admin/reload-config
    Switch model:   POST {base}/admin/switch-model/{{model}}
    Stop server:    POST {base}/admin/stop-server
</pre>
        </div>
    </div>

    <div id="statusMessage" class="status-message"></div>

    <script>
        async function switchModel(modelName) {{
            const statusEl = document.getElementById('statusMessage');
            const currentModelEl = document.getElementById('currentModelStatus');
            const llamaStatusEl = document.getElementById('llamaServerStatus');
            const btn = event.target;
            
            // Store original values for error recovery
            const originalModel = currentModelEl.textContent;
            const originalLlamaStatus = llamaStatusEl.textContent;
            
            // Show loading state - update status bar immediately
            btn.disabled = true;
            btn.textContent = 'Loading...';
            currentModelEl.textContent = `Switching to ${{modelName}}...`;
            llamaStatusEl.textContent = 'Switching';
            statusEl.className = 'status-message loading';
            statusEl.textContent = `Switching model to ${{modelName}}... This may take a few minutes.`;
            statusEl.style.display = 'block';
            
            try {{
                const response = await fetch(`/admin/switch-model/${{modelName}}`, {{
                    method: 'POST'
                }});
                
                const data = await response.json();
                
                if (response.ok) {{
                    statusEl.className = 'status-message loading';
                    statusEl.textContent = `Switch requested for ${{modelName}}. Waiting for readiness...`;
                    statusEl.style.display = 'block';
                    btn.disabled = false;
                    btn.textContent = 'Load Model';
                }} else {{
                    throw new Error(data.detail || 'Failed to switch model');
                }}
            }} catch (error) {{
                // Restore original values on error
                currentModelEl.textContent = originalModel;
                llamaStatusEl.textContent = originalLlamaStatus;
                statusEl.className = 'status-message error';
                statusEl.textContent = `Error: ${{error.message}}`;
                btn.disabled = false;
                btn.textContent = 'Load Model';
                // Hide error after 5 seconds
                setTimeout(() => statusEl.style.display = 'none', 5000);
            }}
        }}

        // Status bar elements
        const currentModelEl = document.getElementById('currentModelStatus');
        const llamaStatusEl = document.getElementById('llamaServerStatus');
        const statusEl = document.getElementById('statusMessage');
        const routerModeEl = document.getElementById('routerModeStatus');
        const routerModeLabel = document.getElementById('routerModeLabel');
        
        // Stats panel elements
        const statsPanel = document.getElementById('statsPanel');
        const statsModelEl = document.getElementById('statsModel');
        const statsLlamaStatusEl = document.getElementById('statsLlamaStatus');
        const statsNCtxEl = document.getElementById('statsNCtx');
        const statsKvCacheEl = document.getElementById('statsKvCache');
        const statsTokensSentEl = document.getElementById('statsTokensSent');
        const statsTokensRecvEl = document.getElementById('statsTokensRecv');
        
        // Track the actual current model (updated after successful operations)
        let actualCurrentModel = '{current_model or "None"}';
        const routerModeEnabled = Boolean(window.__ROUTER_MODE);
        const routerModels = window.__ROUTER_MODELS;

        if (routerModeEl) {{
            const serverFlag = routerModeEl.dataset.routerMode === 'true';
            const enabled = routerModeEnabled || serverFlag;
            routerModeEl.style.display = enabled ? 'flex' : 'none';
            if (routerModeLabel) routerModeLabel.textContent = enabled ? 'Enabled' : 'Disabled';
        }}
        
        // Toggle stats panel visibility
        function toggleStatsPanel() {{
            if (statsPanel.style.display === 'none') {{
                statsPanel.style.display = 'block';
                document.querySelector('.stats-toggle').textContent = 'Hide Model Stats';
            }} else {{
                statsPanel.style.display = 'none';
                document.querySelector('.stats-toggle').textContent = 'Show Model Stats';
            }}
        }}
        
        // Update stats panel with new values (only if changed)
        function updateStatsPanel(data) {{
            if (!data) return;
            
            if (data.current_model !== undefined) {{
                const val = data.current_model || '-';
                if (statsModelEl.textContent !== val) {{
                    statsModelEl.textContent = val;
                }}
            }}

            if (data.loaded_models !== undefined) {{
                const val = data.loaded_models.length ? data.loaded_models.join(', ') : '-';
                if (statsModelEl.textContent !== val) {{
                    statsModelEl.textContent = val;
                }}
            }}
            
            if (data.llama_server_running !== undefined) {{
                const val = data.llama_server_running ? 'Running' : 'Stopped';
                if (statsLlamaStatusEl.textContent !== val) {{
                    statsLlamaStatusEl.textContent = val;
                }}
            }}
            
            if (data.n_ctx !== undefined) {{
                const valSpan = statsNCtxEl.querySelector('.stats-val');
                const unknownSpan = statsNCtxEl.querySelector('.stats-unknown');
                if (data.n_ctx === null || data.n_ctx === undefined) {{
                    if (valSpan) valSpan.textContent = '-';
                    if (unknownSpan) unknownSpan.style.display = 'inline';
                }} else {{
                    if (valSpan) valSpan.textContent = data.n_ctx;
                    if (unknownSpan) unknownSpan.style.display = 'none';
                }}
            }}
            
            if (data.kv_cache_tokens !== undefined) {{
                const valSpan = statsKvCacheEl.querySelector('.stats-val');
                const unknownSpan = statsKvCacheEl.querySelector('.stats-unknown');
                if (data.kv_cache_tokens === null || data.kv_cache_tokens === undefined) {{
                    if (valSpan) valSpan.textContent = '-';
                    if (unknownSpan) unknownSpan.style.display = 'inline';
                }} else {{
                    if (valSpan) valSpan.textContent = data.kv_cache_tokens;
                    if (unknownSpan) unknownSpan.style.display = 'none';
                }}
            }}
            
            if (data.total_sent !== undefined) {{
                const val = String(data.total_sent);
                if (statsTokensSentEl.textContent !== val) {{
                    statsTokensSentEl.textContent = val;
                }}
            }}
            
            if (data.total_recv !== undefined) {{
                const val = String(data.total_recv);
                if (statsTokensRecvEl.textContent !== val) {{
                    statsTokensRecvEl.textContent = val;
                }}
            }}
        }}
        
        // Helper function to show model switching status
        function showSwitchingStatus(targetModel) {{
            currentModelEl.textContent = `Switching to ${{targetModel}}...`;
            llamaStatusEl.textContent = 'Switching';
            statusEl.className = 'status-message loading';
            statusEl.textContent = `Switching model to ${{targetModel}}... This may take a few minutes.`;
            statusEl.style.display = 'block';
        }}
        
        // Helper function to update status after successful model load
        function showModelReady(modelName) {{
            actualCurrentModel = modelName;
            currentModelEl.textContent = modelName;
            llamaStatusEl.textContent = 'Running';
            statusEl.className = 'status-message success';
            statusEl.textContent = `Model ${{modelName}} is ready`;
            // Hide success message after 3 seconds
            setTimeout(() => statusEl.style.display = 'none', 3000);
        }}
        
        // Helper function to check if model switch is needed and show status
        function checkAndShowSwitchStatus(targetModel) {{
            // Check if this is a local model that might need switching
            const localModels = {local_model_names_json};
            const isLocal = localModels.some(m => targetModel.toLowerCase().startsWith(m.toLowerCase()));
            
            if (isLocal && targetModel !== actualCurrentModel) {{
                showSwitchingStatus(targetModel);
                return true;
            }}
            return false;
        }}
        
        // Helper to refresh status from server
        async function refreshStatus() {{
            try {{
                const response = await fetch('/health');
                const data = await response.json();
                if (data.current_model) {{
                    actualCurrentModel = data.current_model;
                    currentModelEl.textContent = data.current_model;
                    llamaStatusEl.textContent = data.llama_server_running ? 'Running' : 'Stopped';
                }}
                if (data.loaded_models) {{
                    statsModelEl.textContent = data.loaded_models.join(', ');
                }}
            }} catch (e) {{
                // Ignore errors
            }}
        }}

        // Subscribe to Server-Sent Events for real-time status updates
        function connectSSE() {{
            const eventSource = new EventSource('/events');
            
            eventSource.onmessage = (event) => {{
                try {{
                    const data = JSON.parse(event.data);
                    
                    switch (data.type) {{
                        case 'status':
                            if (data.current_model) {{
                                actualCurrentModel = data.current_model;
                                currentModelEl.textContent = data.current_model;
                            }}
                            if (data.loaded_models) {{
                                statsModelEl.textContent = data.loaded_models.join(', ');
                            }}
                            llamaStatusEl.textContent = data.llama_server_running ? 'Running' : 'Stopped';
                            updateStatsPanel(data);
                            break;
                            
                        case 'switching':
                            // Model switch started
                            showSwitchingStatus(data.target_model);
                            break;
                            
                        case 'ready':
                            // Model switch completed successfully
                            showModelReady(data.current_model);
                            break;
                            
                        case 'error':
                            // Model switch failed
                            currentModelEl.textContent = data.current_model || 'None';
                            llamaStatusEl.textContent = data.llama_server_running ? 'Running' : 'Stopped';
                            statusEl.className = 'status-message error';
                            statusEl.textContent = data.message || 'An error occurred';
                            statusEl.style.display = 'block';
                            setTimeout(() => statusEl.style.display = 'none', 5000);
                            break;
                    }}
                }} catch (e) {{
                    console.error('Error parsing SSE message:', e);
                }}
            }};
            
            eventSource.onerror = () => {{
                // Reconnect after a delay
                eventSource.close();
                setTimeout(connectSSE, 5000);
            }};
        }}
        
        // Start SSE connection
        connectSSE();

        // Admin button handlers
        async function reloadConfig() {{
            try {{
                const resp = await fetch('/admin/reload-config', {{ method: 'POST' }});
                const data = await resp.json();
                statusEl.className = 'status-message success';
                statusEl.textContent = data.message || 'Config reloaded';
                statusEl.style.display = 'block';
                setTimeout(() => statusEl.style.display = 'none', 3000);
                await refreshStatus();
            }} catch (e) {{
                statusEl.className = 'status-message error';
                statusEl.textContent = 'Failed to reload config';
                statusEl.style.display = 'block';
                setTimeout(() => statusEl.style.display = 'none', 5000);
            }}
        }}

        async function adminSwitchModel() {{
            const selected = modelSelect ? modelSelect.value : null;
            if (!selected) return;
            try {{
                showSwitchingStatus(selected);
                const resp = await fetch(`/admin/switch-model/${{selected}}`, {{ method: 'POST' }});
                const data = await resp.json();
                if (resp.ok) {{
                    statusEl.className = 'status-message loading';
                    statusEl.textContent = data.message || `Switch requested for ${{selected}}. Waiting for readiness...`;
                    statusEl.style.display = 'block';
                }} else {{
                    throw new Error(data.detail || 'Switch failed');
                }}
            }} catch (e) {{
                statusEl.className = 'status-message error';
                statusEl.textContent = `Error: ${{e.message}}`;
                statusEl.style.display = 'block';
                setTimeout(() => statusEl.style.display = 'none', 5000);
                await refreshStatus();
            }}
        }}

        async function stopServer() {{
            if (!confirm('Are you sure you want to stop the llama-server?')) return;
            try {{
                const resp = await fetch('/admin/stop-server', {{ method: 'POST' }});
                const data = await resp.json();
                statusEl.className = 'status-message success';
                statusEl.textContent = data.message || 'llama-server stopped';
                statusEl.style.display = 'block';
                setTimeout(() => statusEl.style.display = 'none', 3000);
                await refreshStatus();
            }} catch (e) {{
                statusEl.className = 'status-message error';
                statusEl.textContent = 'Failed to stop server';
                statusEl.style.display = 'block';
                setTimeout(() => statusEl.style.display = 'none', 5000);
            }}
        }}

        // Quick Test functionality
        const testInput = document.getElementById('testInput');
        const testOutput = document.getElementById('testOutput');
        const testStatus = document.getElementById('testStatus');
        let isStreaming = false;

        testInput.addEventListener('keydown', async (e) => {{
            if (e.key === 'Enter' && !e.shiftKey) {{
                e.preventDefault();
                if (isStreaming) return;
                
                const message = testInput.value.trim();
                if (!message) return;
                
                await sendTestMessage(message);
            }}
        }});

        async function sendTestMessage(message) {{
            isStreaming = true;
            testOutput.textContent = '';
            testStatus.textContent = 'Connecting...';
            testStatus.className = 'test-status';
            
            try {{
                const response = await fetch('/v1/chat/completions', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify({{
                        model: actualCurrentModel,
                        messages: [{{ role: 'user', content: message }}],
                        stream: true
                    }})
                }});
                
                if (!response.ok) {{
                    const err = await response.json();
                    throw new Error(err.detail || `HTTP ${{response.status}}`);
                }}
                
                testStatus.textContent = 'Streaming...';
                testStatus.className = 'test-status streaming';
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                
                while (true) {{
                    const {{ done, value }} = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');
                    
                    for (const line of lines) {{
                        if (line.startsWith('data: ')) {{
                            const data = line.slice(6);
                            if (data === '[DONE]') continue;
                            
                            try {{
                                const json = JSON.parse(data);
                                const content = json.choices?.[0]?.delta?.content;
                                if (content) {{
                                    testOutput.textContent += content;
                                    testOutput.scrollTop = testOutput.scrollHeight;
                                }}
                            }} catch {{
                                // Skip invalid JSON
                            }}
                        }}
                    }}
                }}
                
                testStatus.textContent = 'Complete';
                testStatus.className = 'test-status';
            }} catch (error) {{
                testStatus.textContent = `Error: ${{error.message}}`;
                testStatus.className = 'test-status error';
            }} finally {{
                isStreaming = false;
                // Refresh status bar in case model changed
                await refreshStatus();
            }}
        }}

        // API Endpoint Test functionality
        const apiTestSection = document.getElementById('apiTestSection');
        const apiTestTitle = document.getElementById('apiTestTitle');
        const apiTestRequest = document.getElementById('apiTestRequest');
        const apiTestResponse = document.getElementById('apiTestResponse');
        const apiTestStatus = document.getElementById('apiTestStatus');
        const modelSelect = document.getElementById('modelSelect');
        
        let currentEndpointType = null;

        function getTestExample(endpointType) {{
            const selectedModel = modelSelect ? modelSelect.value : '{current_model or "qwen3"}';
            
            const examples = {{
                chat: {{
                    title: 'POST /v1/chat/completions',
                    method: 'POST',
                    url: '/v1/chat/completions',
                    body: {{
                        model: selectedModel,
                        messages: [{{ role: 'user', content: 'Say hello in exactly 3 words.' }}],
                        max_tokens: 50
                    }}
                }},
                completions: {{
                    title: 'POST /v1/completions',
                    method: 'POST',
                    url: '/v1/completions',
                    body: {{
                        model: selectedModel,
                        prompt: 'The quick brown fox',
                        max_tokens: 30
                    }}
                }},
                models: {{
                    title: 'GET /v1/models',
                    method: 'GET',
                    url: '/v1/models',
                    body: null
                }},
                embeddings: {{
                    title: 'POST /v1/embeddings',
                    method: 'POST',
                    url: '/v1/embeddings',
                    body: {{
                        model: selectedModel,
                        input: 'Hello, world!'
                    }}
                }}
            }};
            
            return examples[endpointType];
        }}
        
        function updateTestRequest() {{
            if (!currentEndpointType) return;
            
            const example = getTestExample(currentEndpointType);
            if (example && example.body) {{
                apiTestRequest.textContent = JSON.stringify(example.body, null, 2);
            }}
        }}

        async function testEndpoint(endpointType) {{
            currentEndpointType = endpointType;
            const example = getTestExample(endpointType);
            if (!example) return;

            // Check if we need to show model switching status
            const selectedModel = modelSelect ? modelSelect.value : actualCurrentModel;
            const willSwitch = checkAndShowSwitchStatus(selectedModel);

            // Show the test section
            apiTestSection.style.display = 'block';
            apiTestTitle.textContent = example.title;
            
            // Format and display the request
            if (example.body) {{
                apiTestRequest.textContent = JSON.stringify(example.body, null, 2);
            }} else {{
                apiTestRequest.textContent = '(No request body - GET request)';
            }}
            
            apiTestResponse.textContent = '';
            apiTestStatus.textContent = willSwitch ? 'Switching model...' : 'Sending request...';
            apiTestStatus.className = 'test-status';

            // Scroll to the test section
            apiTestSection.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});

            try {{
                const fetchOptions = {{
                    method: example.method,
                    headers: {{}}
                }};

                if (example.body) {{
                    fetchOptions.headers['Content-Type'] = 'application/json';
                    fetchOptions.body = JSON.stringify(example.body);
                }}

                const response = await fetch(example.url, fetchOptions);
                // Prefer JSON parsing, but gracefully handle non-JSON responses (HTML/text)
                let data;
                const contentType = response.headers.get('content-type') || '';
                if (contentType.includes('application/json')) {{
                    try {{
                        data = await response.json();
                    }} catch (e) {{
                        // Malformed JSON despite content-type; fall back to text
                        data = await response.text();
                    }}
                }} else {{
                    // Not JSON - try to parse as JSON, otherwise keep as plain text
                    const txt = await response.text();
                    try {{
                        data = JSON.parse(txt);
                    }} catch (e) {{
                        data = txt;
                    }}
                }}

                // Update status bar after request completes (model may have switched)
                await refreshStatus();

                const formatted = typeof data === 'string' ? data : JSON.stringify(data, null, 2);

                if (response.ok) {{
                    apiTestResponse.textContent = formatted;
                    apiTestStatus.textContent = `Success (HTTP ${{response.status}})`;
                    apiTestStatus.className = 'test-status streaming';
                }} else {{
                    apiTestResponse.textContent = formatted;
                    apiTestStatus.textContent = `Error (HTTP ${{response.status}})`;
                    apiTestStatus.className = 'test-status error';
                }}
            }} catch (error) {{
                apiTestResponse.textContent = error.message;
                apiTestStatus.textContent = 'Request failed';
                apiTestStatus.className = 'test-status error';
                // Still try to refresh status on error
                await refreshStatus();
            }}
        }}

        function closeApiTest() {{
            apiTestSection.style.display = 'none';
            currentEndpointType = null;
        }}
    </script>
</body>
</html>"""
    html_content = html_content.replace(
        '</body>',
        f'<script>window.__ROUTER_MODE = {json.dumps(router_mode)}; window.__ROUTER_MODELS = {json.dumps(router_models)};</script></body>'
    )
    return HTMLResponse(content=html_content)








@app.get("/events")
async def status_events():
    """Server-Sent Events endpoint for real-time status updates."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    sse_clients.add(queue)

    async def event_generator():
        try:
            llama_status = await query_llama_status()
            total_sent = token_counts.get("total_sent", 0)
            total_recv = token_counts.get("total_recv", 0)
            
            loaded_models = None
            if llama_status.get("router_mode"):
                router_models = await router_list_models()
                loaded_models = _extract_router_model_ids(router_models)

            initial_status = json.dumps({
                "type": "status",
                "current_model": current_model,
                "loaded_models": loaded_models,
                "llama_server_running": llama_status["llama_server_running"],
                "n_ctx": llama_status["n_ctx"],
                "kv_cache_tokens": llama_status["kv_cache_tokens"],
                "total_sent": total_sent,
                "total_recv": total_recv
            })
            yield f"data: {initial_status}\n\n"

            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield message
                except asyncio.TimeoutError:
                    # keepalive comment
                    yield ": keepalive\n\n"
        finally:
            sse_clients.discard(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )





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
    """Stream a log file as Server-Sent Events (SSE).

    Query params:
    - lines: number of previous lines to include initially (default 100)
    - source: which log to tail: 'proxy' (default) or 'llama' for llama-server.log

    Sends an initial SSE message with key `initial` containing the last
    `lines` lines, then streams new lines as they are appended with key
    `line`. Includes a `source` field to identify which log the data belongs to.
    """
    # Validate source parameter
    if source not in ("proxy", "llama"):
        source = "proxy"
    
    log_path = _resolve_log_path(source)

    async def event_generator():
        # local reference to counts queue for cleanup in finally - ensure always defined
        _local_counts_queue = None

        try:
            if not log_path.exists():
                err = {"error": "log_not_found", "path": str(log_path)}
                yield f"data: {json.dumps(err)}\n\n"
                return

            # Helper to read last N lines in a thread
            def read_last_n(n: int) -> str:
                # Read in binary for efficient seeking
                with open(log_path, "rb") as f:
                    f.seek(0, 2)
                    filesize = f.tell()
                    block_size = 1024
                    data = b""
                    # Read backwards until we have enough lines or hit BOF
                    while filesize > 0 and data.count(b"\n") <= n:
                        read_size = min(block_size, filesize)
                        f.seek(filesize - read_size)
                        chunk = f.read(read_size)
                        data = chunk + data
                        filesize -= read_size
                    lines_bytes = data.splitlines()[-n:]
                    return b"\n".join(lines_bytes).decode("utf-8", errors="replace")

            # Send initial block of lines
            initial = await asyncio.to_thread(read_last_n, lines)
            yield f"data: {json.dumps({'initial': initial, 'source': source})}\n\n"

            # Register for counts updates
            counts_queue: asyncio.Queue | None = None
            try:
                counts_queue = asyncio.Queue(maxsize=10)
                log_tail_clients.add(counts_queue)
            except Exception:
                counts_queue = None

            # Start following the file
            last_pos = log_path.stat().st_size
            # local reference to the counts queue
            _local_counts_queue = counts_queue if counts_queue is not None else None

            while True:
                # If client disconnected, stop
                if await asyncio.sleep(0):
                    pass

                # Small sleep / wait for counts updates to avoid busy loop
                try:
                    # Wait briefly for any counts/tokens updates to arrive on the queue.
                    update = None
                    if _local_counts_queue is not None:
                        try:
                            update = await asyncio.wait_for(_local_counts_queue.get(), timeout=0.25)
                        except asyncio.TimeoutError:
                            update = None
                    else:
                        await asyncio.sleep(0.25)
                except asyncio.CancelledError:
                    break

                # If we got an update, send it immediately and continue (don't wait for file checks)
                if update is not None:
                    try:
                        yield f"data: {json.dumps(update)}\n\n"
                    except Exception:
                        pass
                    continue
                # If file was rotated/recreated, reset position
                try:
                    cur_stat = log_path.stat()
                except FileNotFoundError:
                    # File disappeared; notify and exit
                    yield f"data: {json.dumps({'info': 'log_rotated_or_removed', 'source': source})}\n\n"
                    break

                cur_size = cur_stat.st_size
                if cur_size < last_pos:
                    # File truncated/rotated
                    last_pos = 0

                if cur_size > last_pos:
                    # Read new data
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(last_pos)
                        new = f.read()
                    last_pos = cur_size

                    # Send each new line as its own SSE message
                    for line in new.splitlines():
                        yield f"data: {json.dumps({'line': line, 'source': source})}\n\n"
                else:
                    # No new file data; send keepalive
                    yield ": keepalive\n\n"
        finally:
            # Cleanup
            try:
                if _local_counts_queue is not None:
                    log_tail_clients.discard(_local_counts_queue)
            except Exception:
                pass
            return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/logs")
async def view_logs(request: Request):
    """Simple web UI to view both proxy and llama-server logs using SSE from /logs/tail."""
    base = str(request.base_url).rstrip('/')
    async def get_counts_html():
        items = []
        async with counts_lock:
            for k, v in request_counts.items():
                items.append((k, v))
        items.sort(key=lambda x: (-x[1], x[0]))
        rows = '\n'.join([f'<div class="line">{k}: <strong>{v}</strong></div>' for k, v in items])
        if not rows:
            rows = '<div class="muted">No requests recorded yet.</div>'
        return rows

    counts_html = await get_counts_html()
    async def get_tokens_html():
        items = []
        async with token_lock:
            for k, v in token_counts.items():
                items.append((k, v))
        totals = []
        for k, v in items:
            if k.startswith('total_'):
                totals.append((k, v))
        other = [(k, v) for k, v in items if not k.startswith('total_')]
        other.sort(key=lambda x: (-x[1], x[0]))
        rows = ''
        for k, v in totals:
            rows += f'<div class="line">{k}: <strong>{v}</strong></div>'
        for k, v in other:
            rows += f'<div class="line">{k}: <strong>{v}</strong></div>'
        if not rows:
            rows = '<div class="muted">No token stats yet.</div>'
        return rows

    tokens_html = await get_tokens_html()

    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log Viewer</title>
    <style>
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #1f2940;
            --text-primary: #eee;
            --text-secondary: #aaa;
            --accent: #4f8cff;
            --accent-hover: #6ba1ff;
            --accent-llama: #ff8c4f;
            --accent-llama-hover: #ffa66b;
            --success: #4caf50;
            --warning: #ff9800;
            --border: #2a3a5a;
            --log-error: #b00020;
            --log-warning: #f59e0b;
            --log-info: #60a5fa;
            --log-debug: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 1rem;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { display:flex; gap:1rem; align-items:center; margin-bottom:1rem; flex-wrap:wrap; }
        .header-left { display:flex; align-items:center; gap:1rem; }
        .controls { display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap; }
        .controls-llama { display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap; }
        input[type=number] { width:5rem; padding:0.35rem; border-radius:6px; border:1px solid var(--border); background:var(--bg-card); color:var(--text-primary); }
        button { border:none; padding:0.4rem 0.6rem; border-radius:6px; cursor:pointer; }
        button.proxy-btn { background:var(--accent); color:#fff; }
        button.proxy-btn:hover { background:var(--accent-hover); }
        button.llama-btn { background:var(--accent-llama); color:#fff; }
        button.llama-btn:hover { background:var(--accent-llama-hover); }
        button:disabled { opacity:0.5; cursor:not-allowed; }
        .log-panes { display:grid; grid-template-columns:1fr 1fr; gap:1rem; height: calc(100vh - 200px); }
        .pane { display:flex; flex-direction:column; overflow:hidden; }
        .pane-header { display:flex; align-items:center; gap:0.5rem; padding:0.5rem; background:var(--bg-card); border-radius:8px 8px 0 0; border:1px solid var(--border); border-bottom:none; }
        .pane-header.proxy { color:var(--accent); font-weight:600; }
        .pane-header.llama { color:var(--accent-llama); font-weight:600; }
        .pane-controls { display:flex; gap:0.3rem; align-items:center; flex-wrap:wrap; }
        .pane-controls input[type=number] { width:4rem; }
        .pane-controls button { padding:0.25rem 0.5rem; font-size:0.85rem; }
        .pane-controls button.proxy-btn { background:var(--accent); color:#fff; }
        .pane-controls button.llama-btn { background:var(--accent-llama); color:#fff; }
        .pane-controls button.connected.proxy-btn { background:var(--success); }
        .pane-controls button.connected.llama-btn { background:var(--success); }
        .log { flex:1; overflow:auto; padding:0.75rem; font-family: monospace; background: linear-gradient(180deg, var(--bg-card), rgba(15,18,30,1)); border:1px solid var(--border); border-radius:0 0 8px 8px; white-space:pre-wrap; }
        .line { padding:0 0 2px 0; border-bottom:1px solid rgba(255,255,255,0.02); font-size:0.85rem; }
        .muted { color:var(--text-secondary); font-size:0.9rem; }
        .summary { margin-bottom:0.75rem; }
        .summary h3 { margin:0 0 0.5rem 0; color:var(--accent); }
        .summary-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.75rem; }
        .summary-card { background:var(--bg-card); padding:0.75rem; border-radius:6px; border:1px solid var(--border); max-height:140px; overflow:auto; }
        .summary-card h4 { color:var(--text-secondary); font-size:0.85rem; margin-bottom:0.25rem; }
        .pane-label { font-size:0.9rem; min-width:80px; }
    </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="header-left">
        <a href="/" style="color:var(--accent); text-decoration:none; font-weight:600;">Home</a>
      </div>
      <div style="margin-left:auto; display:flex; gap:0.5rem; align-items:center;">
        <label class="muted" style="font-size:0.85rem;">Shared Lines:</label>
        <input id="sharedLines" type="number" value="200" min="1" style="width:5rem;" />
      </div>
    </div>

    <div class="summary">
      <h3>Request Summary</h3>
      <div class="summary-grid">
        <div class="summary-card">
          <h4>Counts</h4>
          <div id="counts"></div>
          <pre id="rawCounts" style="display:none; margin-top:8px; font-size:0.75rem; color:var(--text-secondary);"></pre>
        </div>
        <div class="summary-card">
          <h4>Tokens</h4>
          <div id="tokens"></div>
          <pre id="rawTokens" style="display:none; margin-top:8px; font-size:0.75rem; color:var(--text-secondary);"></pre>
        </div>
      </div>
    </div>

    <div class="log-panes">
      <!-- Proxy Log Pane -->
      <div class="pane" id="proxyPane">
        <div class="pane-header proxy">
          <span class="pane-label">Proxy log</span>
          <div class="pane-controls">
            <input id="proxyLines" type="number" value="200" min="1" />
            <button id="proxyConnect" class="proxy-btn">Connect</button>
            <button id="proxyDisconnect" class="proxy-btn" disabled>Disconnect</button>
            <button id="proxyClear" class="proxy-btn">Clear</button>
            <label style="display:flex; align-items:center; gap:0.25rem; color:var(--text-secondary); font-size:0.8rem;">
              <input id="proxyAutoscroll" type="checkbox" checked /> Auto
            </label>
            <button id="proxyDownload" class="proxy-btn">Download</button>
          </div>
        </div>
        <div id="proxyLog" class="log"></div>
      </div>

      <!-- Llama-server Log Pane -->
      <div class="pane" id="llamaPane">
        <div class="pane-header llama">
          <span class="pane-label">Llama-server log</span>
          <div class="pane-controls">
            <input id="llamaLines" type="number" value="200" min="1" />
            <button id="llamaConnect" class="llama-btn">Connect</button>
            <button id="llamaDisconnect" class="llama-btn" disabled>Disconnect</button>
            <button id="llamaClear" class="llama-btn">Clear</button>
            <label style="display:flex; align-items:center; gap:0.25rem; color:var(--text-secondary); font-size:0.8rem;">
              <input id="llamaAutoscroll" type="checkbox" checked /> Auto
            </label>
            <button id="llamaDownload" class="llama-btn">Download</button>
          </div>
        </div>
        <div id="llamaLog" class="log"></div>
      </div>
    </div>
  </div>

  <script>
    const endpointDefs = [
      { label: 'Chat', path: '/v1/chat/completions' },
      { label: 'Completions', path: '/v1/completions' },
      { label: 'Embeddings', path: '/v1/embeddings' },
      { label: 'Models', path: '/v1/models' }
    ];

    let latestCounts = {};
    let latestTokens = {};
    let esProxy = null;
    let esLlama = null;

    const proxyLog = document.getElementById('proxyLog');
    const llamaLog = document.getElementById('llamaLog');
    const proxyLinesInput = document.getElementById('proxyLines');
    const llamaLinesInput = document.getElementById('llamaLines');
    const sharedLinesInput = document.getElementById('sharedLines');
    const proxyConnectBtn = document.getElementById('proxyConnect');
    const proxyDisconnectBtn = document.getElementById('proxyDisconnect');
    const llamaConnectBtn = document.getElementById('llamaConnect');
    const llamaDisconnectBtn = document.getElementById('llamaDisconnect');
    const proxyAutoscrollCb = document.getElementById('proxyAutoscroll');
    const llamaAutoscrollCb = document.getElementById('llamaAutoscroll');

    // Sync shared lines input with individual inputs
    sharedLinesInput.addEventListener('change', () => {
      const n = sharedLinesInput.value;
      proxyLinesInput.value = n;
      llamaLinesInput.value = n;
    });
    proxyLinesInput.addEventListener('change', () => {
      sharedLinesInput.value = proxyLinesInput.value;
    });
    llamaLinesInput.addEventListener('change', () => {
      sharedLinesInput.value = llamaLinesInput.value;
    });

    function appendLine(logEl, autoscrollCb, text) {
      const div = document.createElement('div');
      div.className = 'line';
      div.textContent = text;
      logEl.appendChild(div);
      if (autoscrollCb.checked) {
        logEl.scrollTop = logEl.scrollHeight;
      }
    }

    function renderSummary() {
      try {
        const countsEl = document.getElementById('counts');
        const tokensEl = document.getElementById('tokens');
        const counts = latestCounts || {};
        const tokens = latestTokens || {};
        const countsParts = [];
        const tokensParts = [];

        for (const def of endpointDefs) {
          const label = def.label;
          const path = def.path;
          let reqTotal = 0;
          for (const [k,v] of Object.entries(counts)) {
            try {
              const m = k.match(/^[A-Z]+\s+(\S+)\s+->/);
              const reqPath = m ? m[1] : null;
              const pathNoV1 = path.replace(/^\/v1\//, '/');
              if (reqPath && (reqPath === path || reqPath === pathNoV1) && !k.includes('-> model:')) {
                reqTotal += Number(v || 0);
              }
            } catch (e) { /* ignore */ }
          }

          let sent = 0, recv = 0;
          for (const [k,v] of Object.entries(tokens)) {
            try {
              if (!k) continue;
              const n = Number(v || 0);
              if (k.startsWith('sent:') && k.includes(path)) sent += n;
              if (k.startsWith('recv:') && k.includes(path)) recv += n;
            } catch (e) { /* ignore */ }
          }

          countsParts.push(`<div class="line">${label}: <strong>${reqTotal}</strong></div>`);
          tokensParts.push(`<div class="line">${label}: <strong>sent ${sent}</strong> <span style="margin-left:8px;">recv <strong>${recv}</strong></span></div>`);
        }

        if (countsEl) countsEl.innerHTML = countsParts.join('') || '<div class="muted">No requests recorded yet.</div>';
        if (tokensEl) tokensEl.innerHTML = tokensParts.join('') || '<div class="muted">No token stats yet.</div>';

        const rawCountsEl = document.getElementById('rawCounts');
        const rawTokensEl = document.getElementById('rawTokens');
        if (rawCountsEl) rawCountsEl.textContent = JSON.stringify(counts, null, 2);
        if (rawTokensEl) rawTokensEl.textContent = JSON.stringify(tokens, null, 2);
      } catch (e) { /* ignore */ }
    }

    try {
      if (window.__INITIAL_STATS) {
        latestCounts = window.__INITIAL_STATS.counts || {};
        latestTokens = window.__INITIAL_STATS.tokens || {};
        renderSummary();
      }
    } catch (e) { /* ignore */ }

    function handleMessage(logEl, autoscrollCb, obj) {
      if (obj.initial) {
        appendLine(logEl, autoscrollCb, '--- initial log ---');
        obj.initial.split(String.fromCharCode(10)).forEach(l => appendLine(logEl, autoscrollCb, l));
        appendLine(logEl, autoscrollCb, '--- end initial ---');
      } else if (obj.line) {
        appendLine(logEl, autoscrollCb, obj.line);
      } else if (obj.counts) {
        try {
          latestCounts = obj.counts || {};
          renderSummary();
        } catch (e) { /* ignore */ }
      } else if (obj.tokens) {
        try {
          latestTokens = obj.tokens || {};
          renderSummary();
        } catch (e) { /* ignore */ }
      } else if (obj.info) {
        appendLine(logEl, autoscrollCb, '[info] ' + obj.info);
      } else if (obj.error) {
        appendLine(logEl, autoscrollCb, '[error] ' + JSON.stringify(obj));
      }
    }

    function connectProxy() {
      if (esProxy) return;
      const n = Math.max(1, parseInt(proxyLinesInput.value || '200'));
      const url = '/logs/tail?lines=' + encodeURIComponent(n) + '&source=proxy';
      esProxy = new EventSource(url);
      proxyConnectBtn.disabled = true;
      proxyDisconnectBtn.disabled = false;
      proxyConnectBtn.classList.add('connected');

      esProxy.onmessage = e => {
        try {
          const obj = JSON.parse(e.data);
          handleMessage(proxyLog, proxyAutoscrollCb, obj);
        } catch (err) {
          appendLine(proxyLog, proxyAutoscrollCb, e.data);
        }
      };

      esProxy.onerror = () => {
        appendLine(proxyLog, proxyAutoscrollCb, '[connection closed]');
        disconnectProxy();
      };
    }

    function disconnectProxy() {
      if (!esProxy) return;
      esProxy.close();
      esProxy = null;
      proxyConnectBtn.disabled = false;
      proxyDisconnectBtn.disabled = true;
      proxyConnectBtn.classList.remove('connected');
    }

    function connectLlama() {
      if (esLlama) return;
      const n = Math.max(1, parseInt(llamaLinesInput.value || '200'));
      const url = '/logs/tail?lines=' + encodeURIComponent(n) + '&source=llama';
      esLlama = new EventSource(url);
      llamaConnectBtn.disabled = true;
      llamaDisconnectBtn.disabled = false;
      llamaConnectBtn.classList.add('connected');

      esLlama.onmessage = e => {
        try {
          const obj = JSON.parse(e.data);
          handleMessage(llamaLog, llamaAutoscrollCb, obj);
        } catch (err) {
          appendLine(llamaLog, llamaAutoscrollCb, e.data);
        }
      };

      esLlama.onerror = () => {
        appendLine(llamaLog, llamaAutoscrollCb, '[connection closed]');
        disconnectLlama();
      };
    }

    function disconnectLlama() {
      if (!esLlama) return;
      esLlama.close();
      esLlama = null;
      llamaConnectBtn.disabled = false;
      llamaDisconnectBtn.disabled = true;
      llamaConnectBtn.classList.remove('connected');
    }

    function downloadLog(logEl, filename) {
      const text = Array.from(logEl.querySelectorAll('.line')).map(n => n.textContent).join(String.fromCharCode(10));
      const blob = new Blob([text], {type: 'text/plain'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    }

    proxyConnectBtn.addEventListener('click', connectProxy);
    proxyDisconnectBtn.addEventListener('click', disconnectProxy);
    document.getElementById('proxyClear').addEventListener('click', () => proxyLog.innerHTML = '');
    document.getElementById('proxyDownload').addEventListener('click', () => downloadLog(proxyLog, 'proxy.log'));

    llamaConnectBtn.addEventListener('click', connectLlama);
    llamaDisconnectBtn.addEventListener('click', disconnectLlama);
    document.getElementById('llamaClear').addEventListener('click', () => llamaLog.innerHTML = '');
    document.getElementById('llamaDownload').addEventListener('click', () => downloadLog(llamaLog, 'llama-server.log'));

    connectProxy();
    connectLlama();

    window.addEventListener('beforeunload', () => {
      if (esProxy) esProxy.close();
      if (esLlama) esLlama.close();
    });
  </script>
</body>
</html>"""

    # Prepare JSON snapshot for client-side rendering
    # Use shallow copies under locks
    async with counts_lock:
        counts_snapshot = dict(request_counts)
    async with token_lock:
        tokens_snapshot = dict(token_counts)

    model_list = list(config.get("models", {}).keys())
    router_mode = config.get("server", {}).get("llama_router_mode", False)
    router_models = None
    if router_mode:
        router_models = await router_list_models()
    model_list_json = json.dumps(model_list)
    initial_stats_json = json.dumps({"counts": counts_snapshot, "tokens": tokens_snapshot})

    # Replace placeholders with empty containers; client will render using INITIAL_STATS
    html = html.replace('{counts_html}', '')
    html = html.replace('{tokens_html}', '')

    # Inject initial stats and model list script before </body>
    html = html.replace('</body>', f'<script>window.__INITIAL_STATS = {initial_stats_json}; window.__MODEL_LIST = {model_list_json}; window.__ROUTER_MODE = {json.dumps(router_mode)}; window.__ROUTER_MODELS = {json.dumps(router_models)};</script></body>')
    return HTMLResponse(content=html)


@app.post("/v1/embeddings")
async def create_embeddings(request: Request):
    """
    Dedicated endpoint for embeddings requests.
    Validates the request and routes to the appropriate backend.
    
    The OpenAI embeddings API expects:
    - model: string (required)
    - input: string or array of strings (required)
    - encoding_format: string (optional, "float" or "base64")
    - dimensions: integer (optional)
    - user: string (optional)
    """
    global current_model
    # Parse request body
    body = await request.body()
    if not body:
        raise HTTPException(
            status_code=400,
            detail="Request body is required"
        )
    
    try:
        body_json = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON in request body"
        )
    
    # Validate required fields
    if "input" not in body_json:
        raise HTTPException(
            status_code=400,
            detail="'input' field is required for embeddings requests"
        )
    
    input_value = body_json["input"]
    if not isinstance(input_value, (str, list)):
        raise HTTPException(
            status_code=400,
            detail="'input' must be a string or an array of strings"
        )
    
    if isinstance(input_value, list):
        if len(input_value) == 0:
            raise HTTPException(
                status_code=400,
                detail="'input' array must not be empty"
            )
        if not all(isinstance(item, (str, int, list)) for item in input_value):
            raise HTTPException(
                status_code=400,
                detail="'input' array elements must be strings, integers, or arrays"
            )
    
    # Resolve model
    model_name = body_json.get("model")
    if not model_name and current_model:
        model_name = current_model
    
    model_cfg = get_model_config(model_name) if model_name else None
    
    if model_cfg is None:
        # Check if default remote is enabled
        default_remote = config.get("default_remote", {})
        if default_remote.get("enabled", False):
            return await proxy_to_remote(request, "v1/embeddings", default_remote)
        
        # If we have a current model loaded, try local
        if current_model:
            return await proxy_to_local(request, "v1/embeddings")
        
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name}. No default remote configured."
        )
    
    if model_cfg.get("type") == "local":
        server_config = config.get("server", {})
        router_mode = server_config.get("llama_router_mode", False)
        # Check if we need to switch models
        llama_model = model_cfg.get("llama_model")
        if not isinstance(llama_model, str) or not llama_model:
            raise HTTPException(
                status_code=500,
                detail=f"Local model configuration missing llama_model for: {model_name}"
            )
        llama_model_str: str = llama_model

        # If model already active and process running, proceed immediately
        if current_model == llama_model_str and llama_process is not None and (llama_process.poll() is None):
            return await proxy_to_local(request, "v1/embeddings")

        # Try a fast router-mode check: model may already be loaded in router
        if router_mode:
            try:
                if await router_is_model_loaded(llama_model_str):
                    logger.info(f"Router reports model {llama_model_str} already loaded; serving request immediately")
                    current_model = llama_model_str
                    return await proxy_to_local(request, "v1/embeddings")
            except Exception:
                # Non-fatal: fall through to scheduling background load
                logger.debug("Fast router check failed; scheduling background load")

        # Otherwise, schedule a background load and return 503 immediately
        target_model: str = model_name if isinstance(model_name, str) and model_name else llama_model_str
        scheduled = schedule_background_load(target_model)
        logger.info(f"Scheduled background load for embeddings request: {target_model} scheduled={scheduled}")
        return _model_loading_response(
            requested_model=model_name if isinstance(model_name, str) else None,
            target_model=target_model,
            scheduled=scheduled,
            endpoint="/v1/embeddings",
        )
    
    elif model_cfg.get("type") == "remote":
        return await proxy_to_remote(request, "v1/embeddings", model_cfg)
    
    raise HTTPException(
        status_code=500,
        detail=f"Invalid model configuration for: {model_name}"
    )


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def proxy_openai_api(request: Request, path: str):
    """
    Main proxy endpoint for OpenAI API requests.
    Routes to local llama-server or remote API based on model.
    """
    # Get the request body to determine the model
    global current_model
    body = await request.body()
    body_json = {}
    model_name = None
    
    if body:
        try:
            body_json = json.loads(body)
            model_name = body_json.get("model")
        except json.JSONDecodeError:
            pass
    
    # If no model specified, use the currently loaded model
    if not model_name and current_model:
        model_name = current_model
    
    # Get model configuration
    model_cfg = get_model_config(model_name) if model_name else None
    
    if model_cfg is None:
        # Check if default remote is enabled
        default_remote = config.get("default_remote", {})
        if default_remote.get("enabled", False):
            return await proxy_to_remote(request, f"v1/{path}", default_remote)
        
        # If we have a current model loaded, use that
        if current_model:
            return await proxy_to_local(request, f"v1/{path}")
        
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name}. No default remote configured."
        )
    
    if model_cfg.get("type") == "local":
        server_config = config.get("server", {})
        router_mode = server_config.get("llama_router_mode", False)
        # Check if we need to switch models
        llama_model = model_cfg.get("llama_model")
        if not isinstance(llama_model, str) or not llama_model:
            raise HTTPException(
                status_code=500,
                detail=f"Local model configuration missing llama_model for: {model_name}"
            )
        llama_model_str: str = llama_model

        # If model already active and process running, proceed immediately
        if current_model == llama_model_str and llama_process is not None and (llama_process.poll() is None):
            return await proxy_to_local(request, f"v1/{path}")

        # Try a fast router-mode check: model may already be loaded in router
        if router_mode:
            try:
                if await router_is_model_loaded(llama_model_str):
                    logger.info(f"Router reports model {llama_model_str} already loaded; serving request immediately")
                    current_model = llama_model_str
                    return await proxy_to_local(request, f"v1/{path}")
            except Exception:
                logger.debug("Fast router check failed; scheduling background load")

        # Otherwise, schedule background load and return 503 so client doesn't hang
        target_model: str = model_name if isinstance(model_name, str) and model_name else llama_model_str
        scheduled = schedule_background_load(target_model)
        logger.info(f"Scheduled background load for request: model={target_model} scheduled={scheduled}")
        return _model_loading_response(
            requested_model=model_name if isinstance(model_name, str) else None,
            target_model=target_model,
            scheduled=scheduled,
            endpoint=f"/v1/{path}",
        )
    
    elif model_cfg.get("type") == "remote":
        return await proxy_to_remote(request, f"v1/{path}", model_cfg)
    
    raise HTTPException(
        status_code=500,
        detail=f"Invalid model configuration for: {model_name}"
    )





@app.post("/admin/switch-model/{model_name}")
async def switch_model(model_name: str):
    """Manually switch to a different model."""
    model_cfg = get_model_config(model_name)
    
    if model_cfg is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")
    
    if model_cfg.get("type") != "local":
        raise HTTPException(
            status_code=400, 
            detail=f"Model {model_name} is not a local model"
        )
    
    logger.info(f"Admin switch-model requested: {model_name}; current_model before: {current_model}; llama_running: {llama_process is not None and llama_process.poll() is None}")
    if await ensure_model_loaded(model_name):
        logger.info(f"Admin switch-model succeeded: requested={model_name} current_model after: {current_model}")
        return {
            "status": "success",
            "message": f"Switched to model: {model_name}",
            "current_model": current_model,
            "llama_server_running": llama_process is not None and llama_process.poll() is None,
            "last_start_failure": last_start_failure,
        }
    else:
        # Return the last captured start failure when available to aid UI debugging
        detail_msg = f"Failed to switch to model: {model_name}"
        if last_start_failure:
            detail_msg = detail_msg + "\n\nLast start failure:\n" + last_start_failure
        logger.error(f"Admin switch-model failed: {model_name}; reason: {detail_msg}")
        raise HTTPException(
            status_code=500,
            detail=detail_msg
        )





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
    _is_empty_response,
    _extract_assistant_content_from_sse,
    _extract_delta_text_from_sse_chunk,
    _normalize_outgoing_headers,
    normalize_provider_name,
    load_config,
    setup_logging,
)


if __name__ == "__main__":
    main()
