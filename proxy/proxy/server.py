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


def _inc_model_switch_refcount() -> None:
    """Increment the global model_switch_refcount in a thread-safe way."""
    global model_switch_refcount
    try:
        with model_switch_refcount_lock:
            model_switch_refcount += 1
    except Exception:
        # Best-effort: swallow errors so status reporting never raises
        pass


def _dec_model_switch_refcount() -> None:
    """Decrement the global model_switch_refcount (never below zero)."""
    global model_switch_refcount
    try:
        with model_switch_refcount_lock:
            model_switch_refcount = max(0, model_switch_refcount - 1)
    except Exception:
        pass
# Shared httpx client for connection pooling
_http_client: Optional[httpx.AsyncClient] = None
# Cache for which llama-server endpoint successfully provided status
_llama_status_endpoint_cache: Optional[str] = None
# Record recent failures for endpoints to avoid hammering endpoints that 404
_llama_status_endpoint_failures: dict = {}
# One-time discovery markers: avoid repeated discovery for the same process
_llama_status_discovered: bool = False
_llama_status_discovered_pid: Optional[int] = None

def schedule_background_load(model_name: str) -> bool:
    """Schedule model load in background if not already running.

    Returns True if a background load was started, False if one is already in progress.
    """
    if not model_name:
        return False
    if background_loads.get(model_name):
        return False

    background_loads[model_name] = True

    # Increment the global refcount so status checks observe switching even
    # when the background worker runs in another thread/event loop.
    try:
        _inc_model_switch_refcount()
    except Exception:
        pass

    async def _bg():
        global background_loads
        try:
            logger.info(f"Background model load started: {model_name}")
            ok = await ensure_model_loaded(model_name)
            if ok:
                logger.info(f"Background model load succeeded: {model_name}")
            else:
                logger.error(f"Background model load failed: {model_name}")
        except Exception:
            logger.exception(f"Exception during background model load for {model_name}")
        finally:
            # Decrement the refcount when done so status reflects completion.
            try:
                _dec_model_switch_refcount()
            except Exception:
                pass
            background_loads.pop(model_name, None)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_bg())
    except RuntimeError:
        # No running loop; spawn a new thread to run background load
        def _run_sync():
            asyncio.run(_bg())
        t = threading.Thread(target=_run_sync, daemon=True)
        t.start()

    return True

    
def extract_progress_data(line: Optional[str]) -> Optional[Tuple[int, float]]:
    """Extract `n_tokens` and `progress` from llama-server stdout progress lines.

    Returns a tuple (n_tokens: int, progress: float) or None if the line does
    not contain valid progress data.

    Examples of supported lines:
        "slot 1 : prompt processing progress, n_tokens = 26988, progress = 0.658083"
    The parser is conservative: it requires both `n_tokens` and `progress`
    fields and at least one comma to avoid false positives from unrelated
    text that mentions the word 'progress'.
    """
    if not isinstance(line, str):
        return None
    text = line.strip()
    if not text:
        return None
    # Require explicit field markers to avoid false positives
    if 'n_tokens' not in text or 'progress' not in text:
        return None
    # Require at least one comma present to resemble expected formats
    if ',' not in text:
        return None
    try:
        m_tokens = re.search(r'\bn_tokens\s*=\s*(\d+)\b', text, flags=re.IGNORECASE)
        m_progress = re.search(r'\bprogress\s*=\s*([0-9]+(?:\.[0-9]+)?)\b', text, flags=re.IGNORECASE)
        if not m_tokens or not m_progress:
            return None
        n_tokens = int(m_tokens.group(1))
        progress = float(m_progress.group(1))
        return (n_tokens, progress)
    except Exception:
        return None

# Polling state for /slots API (model -> latest data)
slot_polling_state: dict = {}
# Internal record of active polling tasks (model -> asyncio.Task)
_slot_polling_tasks: dict = {}

async def poll_slots_for_model(model: str, llama_port: int = 0, interval: float = 0.5, max_polls: Optional[int] = None) -> None:
    """Poll the llama-server `/slots` endpoint for a given model and update
    `slot_polling_state[model]` with a dict containing `is_processing` and
    `n_decoded` (or None).

    This function is intentionally conservative: it tolerates a variety of
    response formats (list-of-slots or single-slot dict) and never raises
    on parse errors — instead it records `None` values for missing data.

    The optional ``max_polls`` parameter is useful for tests to limit the
    number of iterations.
    """
    polls = 0
    if not model:
        return None
    slots_url = f"http://localhost:{llama_port}/slots?model={model}"
    while True:
        try:
            client = _http_client if _http_client else httpx.AsyncClient(timeout=httpx.Timeout(5.0))
            # Support either an injected async client instance or creating a
            # temporary AsyncClient context when none is provided.
            if getattr(client, '__aenter__', None):
                # client supports async context manager
                async with client as c:
                    resp = await c.get(slots_url, timeout=5.0)
            else:
                resp = await client.get(slots_url, timeout=5.0)

            if resp is not None and getattr(resp, 'status_code', None) == 200:
                try:
                    data = resp.json()
                except Exception:
                    data = None

                n_decoded = None
                is_processing = False
                if isinstance(data, list):
                    if data:
                        slot = data[0]
                        is_processing = bool(slot.get('is_processing', False))
                        next_token = slot.get('next_token') if isinstance(slot.get('next_token'), dict) else None
                        if next_token is not None and 'n_decoded' in next_token:
                            n_decoded = next_token.get('n_decoded')
                        else:
                            # Fallback to top-level field if present
                            n_decoded = slot.get('n_decoded')
                elif isinstance(data, dict):
                    is_processing = bool(data.get('is_processing', False))
                    next_token = data.get('next_token')
                    if isinstance(next_token, dict) and 'n_decoded' in next_token:
                        n_decoded = next_token.get('n_decoded')
                    else:
                        n_decoded = data.get('n_decoded')

                slot_polling_state[model] = {'is_processing': is_processing, 'n_decoded': n_decoded}
            else:
                # Non-200 or missing response — mark as not processing
                slot_polling_state[model] = {'is_processing': False, 'n_decoded': None}
        except Exception:
            # On any error, record a safe fallback and continue
            slot_polling_state[model] = {'is_processing': False, 'n_decoded': None}

        polls += 1
        if max_polls is not None and polls >= max_polls:
            break

        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break


def start_slot_polling(model: str, llama_port: int, interval: float = 0.5) -> None:
    """Start an asyncio task to poll the slots endpoint for `model`.

    If an active poller already exists for the model it will not start a
    duplicate. If called from a non-async context (no running loop) a
    background thread will be created that runs its own asyncio loop.
    """
    if not model:
        return None
    if model in _slot_polling_tasks and not _slot_polling_tasks[model].done():
        return None
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(poll_slots_for_model(model, llama_port, interval, None))
        _slot_polling_tasks[model] = task
    except RuntimeError:
        # No running event loop in this thread — spawn a thread to run one
        def _run():
            try:
                asyncio.run(poll_slots_for_model(model, llama_port, interval, None))
            except Exception:
                pass
        t = threading.Thread(target=_run, daemon=True)
        t.start()

def format_progress(n_tokens: int, total_tokens: int, progress: float) -> str:
    """Return a formatted, ANSI-dimmed, in-place-updating progress string.

    The returned string begins with a carriage return ("\r") so it can be
    written to stderr repeatedly to update a single console line in-place.

    Percentage is truncated to an integer (no decimals) to match UX
    expectations (e.g. 0.658 -> 65%). The output is ANSI-dimmed and the
    ANSI reset code is appended so subsequent console output is unaffected.
    """
    try:
        pct = int(max(0, min(100, int(progress * 100))))
    except Exception:
        try:
            pct = int(max(0, min(100, int(float(progress) * 100))))
        except Exception:
            pct = 0
    # Truncate rather than round (floor) to match example behaviour.
    pct = int(max(0, min(100, int(progress * 100)))) if isinstance(progress, (int, float)) else pct
    dim = "\x1b[2m"
    reset = "\x1b[0m"
    body = f"Processing {n_tokens}/{total_tokens} tokens ({pct}%)"
    return f"\r{dim}{body}{reset}"

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
backend_signal_counts: dict = {
    "connect_failures": 0,
    "read_failures": 0,
    "timeout_failures": 0,
    "other_failures": 0,
    "concurrency_rejects": 0,
}

# Session restore observability
session_restore_observability: dict = {
    "restore_success_total": 0,
    "restore_fallback_total": {},
    "delta_payload_bytes_total": 0,
}

# Session single-flight observability
session_single_flight_observability: dict = {
    "queue_events_total": 0,
    "reject_events_total": 0,
    "active_sessions_current": 0,
    "queue_depth_current": 0,
}

# Guardrail & invalidation observability
session_guardrail_observability: dict = {
    "guardrail_cutoff_total": 0,
    "guardrail_cutoff_reasons": {},
    "session_invalidation_total": 0,
    "session_invalidation_reasons": {},
}


def _record_restore_success() -> None:
    try:
        session_restore_observability["restore_success_total"] = int(
            session_restore_observability.get("restore_success_total", 0)
        ) + 1
    except Exception:
        pass


def _record_restore_fallback(reason: str) -> None:
    if not reason:
        return
    try:
        bucket = session_restore_observability.setdefault("restore_fallback_total", {})
        bucket[reason] = int(bucket.get(reason, 0)) + 1
    except Exception:
        pass


def _record_delta_payload_bytes(value: int) -> None:
    if value <= 0:
        return
    try:
        session_restore_observability["delta_payload_bytes_total"] = int(
            session_restore_observability.get("delta_payload_bytes_total", 0)
        ) + int(value)
    except Exception:
        pass


def _record_single_flight_queue() -> None:
    try:
        session_single_flight_observability["queue_events_total"] = int(
            session_single_flight_observability.get("queue_events_total", 0)
        ) + 1
    except Exception:
        pass


def _record_single_flight_reject() -> None:
    try:
        session_single_flight_observability["reject_events_total"] = int(
            session_single_flight_observability.get("reject_events_total", 0)
        ) + 1
    except Exception:
        pass


def _record_guardrail_cutoff(reason: str) -> None:
    if not reason:
        return
    try:
        session_guardrail_observability["guardrail_cutoff_total"] = int(
            session_guardrail_observability.get("guardrail_cutoff_total", 0)
        ) + 1
        bucket = session_guardrail_observability.setdefault("guardrail_cutoff_reasons", {})
        bucket[reason] = int(bucket.get(reason, 0)) + 1
    except Exception:
        pass


def _record_session_invalidation(reason: str) -> None:
    if not reason:
        return
    try:
        session_guardrail_observability["session_invalidation_total"] = int(
            session_guardrail_observability.get("session_invalidation_total", 0)
        ) + 1
        bucket = session_guardrail_observability.setdefault("session_invalidation_reasons", {})
        bucket[reason] = int(bucket.get(reason, 0)) + 1
    except Exception:
        pass


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


# Optional tokenizer (tiktoken)
try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None

# Optional process metrics (psutil)
try:
    import psutil
except Exception:
    psutil = None

def _get_tiktoken_encoding_for_model(model_name: str | None):
    if not tiktoken:
        return None
    try:
        if model_name:
            return tiktoken.encoding_for_model(model_name)
    except Exception:
        pass
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

def count_text_tokens(text: str, model_name: str | None = None) -> int:
    """Count tokens in text using tiktoken if available, otherwise a heuristic."""
    if not text:
        return 0
    enc = _get_tiktoken_encoding_for_model(model_name)
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # heuristic: 1 token ~ 4 bytes UTF-8
    return max(1, len(text.encode('utf-8')) // 4)


def _model_loading_response(requested_model: Optional[str], target_model: str, scheduled: bool, endpoint: str) -> JSONResponse:
    """Build a consistent JSON 503 payload when a model is loading."""
    retry_after = int(config.get("server", {}).get("model_loading_retry_after", 30) or 30)
    payload = {
        "error": {
            "type": "model_loading",
            "code": "model_loading",
            "message": f"Model {target_model} is loading, retry shortly"
        },
        "status": 503,
        "requested_model": requested_model,
        "target_model": target_model,
        "scheduled": bool(scheduled),
        "current_model": current_model,
        "llama_server_running": llama_process is not None and llama_process.poll() is None,
        "retry_after": retry_after,
        "endpoint": endpoint,
    }
    return JSONResponse(
        status_code=503,
        content=payload,
        headers={"Retry-After": str(retry_after), "Cache-Control": "no-store"},
    )


def _record_backend_signal(signal_name: str) -> None:
    """Increment a backend signal counter for observability."""
    try:
        if signal_name in backend_signal_counts:
            backend_signal_counts[signal_name] = int(backend_signal_counts.get(signal_name, 0)) + 1
    except Exception:
        pass


def _classify_backend_exception(exc: Exception) -> str:
    """Map backend transport exceptions to signal buckets."""
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return "connect_failures"
    if isinstance(exc, (httpx.ReadError,)):
        return "read_failures"
    if isinstance(exc, (httpx.ReadTimeout, httpx.TimeoutException)):
        return "timeout_failures"
    return "other_failures"


def _is_retryable_backend_exception(exc: Exception) -> bool:
    """Return True when an exception is a retryable backend transport failure."""
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError, httpx.ReadTimeout, httpx.TimeoutException))


def _compute_retry_delay(attempt: int, base_delay: float, max_delay: float, jitter_ratio: float) -> float:
    """Compute bounded exponential backoff with jitter."""
    base = max(0.0, float(base_delay)) * (2 ** max(0, attempt - 1))
    delay = min(base, max(0.0, float(max_delay)))
    if jitter_ratio > 0 and delay > 0:
        jitter = delay * min(max(float(jitter_ratio), 0.0), 1.0)
        delay += jitter * (2 * (os.urandom(1)[0] / 255.0) - 1.0)
        delay = max(0.0, min(delay, max(0.0, float(max_delay))))
    return delay


def _estimate_prompt_tokens(body_json: dict) -> int:
    """Estimate prompt token count from request body.
    
    Returns estimated token count based on message content length.
    Uses a heuristic of ~4 bytes per token for UTF-8 text.
    """
    if not isinstance(body_json, dict):
        return 0
    messages = body_json.get("messages", [])
    if not messages:
        return 0
    # Concatenate all message content
    total_chars = 0
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # Handle array content (e.g., multimodal)
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        total_chars += len(str(item["text"]))
    # Heuristic: ~4 bytes per token
    return max(1, total_chars // 4)


def _compute_adaptive_timeout(
    body_json: dict,
    base_timeout: float,
    per_token_timeout: float,
    max_timeout: float,
) -> float:
    """Compute adaptive timeout based on prompt size.
    
    Timeout = min(base_timeout + per_token_timeout * estimated_tokens, max_timeout)
    
    This allows larger prompts to have longer timeouts while keeping
    a reasonable upper bound.
    """
    estimated_tokens = _estimate_prompt_tokens(body_json)
    adaptive = base_timeout + (per_token_timeout * estimated_tokens)
    return min(adaptive, max_timeout)


async def _call_with_backend_retries(call_factory, path: str, stream: bool = False):
    """Execute backend call with bounded retries on connect/read failures."""
    server_cfg = config.get("server", {})
    max_attempts = int(server_cfg.get("backend_retry_attempts", 3) or 3)
    base_delay = float(server_cfg.get("backend_retry_base_delay_seconds", 0.25) or 0.25)
    max_delay = float(server_cfg.get("backend_retry_max_delay_seconds", 2.0) or 2.0)
    jitter_ratio = float(server_cfg.get("backend_retry_jitter_ratio", 0.25) or 0.25)

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await call_factory()
        except Exception as exc:
            last_exc = exc
            if not _is_retryable_backend_exception(exc) or attempt >= max_attempts:
                _record_backend_signal(_classify_backend_exception(exc))
                try:
                    global backend_ready
                    backend_ready = False
                except Exception:
                    pass
                raise

            signal_name = _classify_backend_exception(exc)
            _record_backend_signal(signal_name)
            delay = _compute_retry_delay(attempt, base_delay, max_delay, jitter_ratio)
            logger.warning(
                "backend_retry path=%s stream=%s attempt=%s/%s delay=%.3fs signal=%s error=%s",
                path,
                stream,
                attempt,
                max_attempts,
                delay,
                signal_name,
                type(exc).__name__,
            )
            await asyncio.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("backend retry loop exhausted without exception")


def _self_heal_retry_after_seconds() -> int:
    server_cfg = config.get("server", {}) if isinstance(config, dict) else {}
    configured = server_cfg.get("llama_self_heal_retry_after_seconds", backend_recovery_state.get("retry_after_seconds", 30))
    try:
        retry_after = int(configured or 30)
    except Exception:
        retry_after = 30
    retry_after = max(1, retry_after)
    backend_recovery_state["retry_after_seconds"] = retry_after
    return retry_after


def _is_self_healing_active() -> bool:
    try:
        return bool(backend_recovery_state.get("in_progress"))
    except Exception:
        return False


def _self_healing_response(path: str) -> JSONResponse:
    retry_after = _self_heal_retry_after_seconds()
    message = "Backend error detected, team is working on recovery. Please retry after 30 seconds."
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "type": "backend_recovery",
                "code": "backend_recovery_in_progress",
                "message": message,
            },
            "status": 503,
            "retry_after": retry_after,
            "path": f"/{path.lstrip('/')}",
        },
        headers={"Retry-After": str(retry_after), "Cache-Control": "no-store"},
    )


def _backend_recovery_snapshot() -> dict:
    state = dict(backend_recovery_state)
    attempts = state.get("attempt_timestamps")
    state["attempt_count"] = len(attempts) if isinstance(attempts, list) else 0
    return state


config: dict = {}


def normalize_provider_name(name: Optional[str]) -> Optional[str]:
    """Normalize provider display/identifier strings.

    Current compatibility rules:
      - "Local Proxy" (case-insensitive) is treated as "Proxy".
    """
    if not name:
        return name
    try:
        n = str(name).strip()
    except Exception:
        return name
    if n.lower() == "local proxy":
        return "Proxy"
    return n



def _extract_tool_call_from_reasoning(reasoning_content: Optional[str]) -> Optional[str]:
    """Extract a tool call XML pattern from reasoning_content.

    When a model with thinking mode enabled (like Qwen3) generates tool calls
    during its thinking phase, they appear in reasoning_content rather than
    content. This function extracts well-formed <function=...>...</function>
    patterns from reasoning content.

    Returns the matched tool call XML string, or None if no tool call found.
    """
    if not reasoning_content:
        return None
    # Match <function=...>...</function> block
    match = re.search(r'<function=[^>]*>.*?</function>', reasoning_content, re.DOTALL)
    if match:
        return match.group(0)
    return None



def _extract_assistant_content(resp_json: dict) -> Optional[str]:
    """Extract assistant content from a non-streaming OpenAI API response.

    Looks for choices[0].message.content and returns it.
    If content is null but reasoning_content contains a tool call
    pattern (<function=...>...</function>), the tool call is extracted
    and returned instead.
    Returns None if unable to extract content or tool call.
    """
    try:
        choices = resp_json.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if content is not None:
                return str(content)
            # Fall back to extracting tool call from reasoning_content
            reasoning_content = message.get("reasoning_content")
            tool_call = _extract_tool_call_from_reasoning(reasoning_content)
            if tool_call:
                logger.info(
                    "Extracted tool call from reasoning_content (non-streaming): %.80s",
                    tool_call,
                )
                return tool_call
    except Exception:
        pass
    return None


def _is_empty_response(response_text: str, resp_json: Optional[dict] = None) -> bool:
    """Check if a response is effectively empty (no content, no tool calls).

    Used to detect cases where the model generates thinking content but
    produces no actual output. Returns True if the response has no usable
    content.

    When resp_json is provided (OpenAI-style response), emptiness is determined
    by the presence of assistant content (text or tool calls) in the JSON
    structure, not by the raw text length.
    When resp_json is not provided, falls back to checking if response_text
    is blank or whitespace-only.
    """
    if resp_json:
        # For JSON API responses, check the structured content
        content = _extract_assistant_content(resp_json)
        if content:
            return False
        # Check reasoning_content for tool calls
        try:
            choices = resp_json.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                rc = message.get("reasoning_content")
                if rc and _extract_tool_call_from_reasoning(rc):
                    return False
        except Exception:
            pass
        return True
    # Fallback: check if the raw response text is blank or whitespace-only
    if response_text and response_text.strip():
        return False
    return True


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


def _extract_assistant_content_from_sse(sse_text: str) -> Optional[str]:
    """Extract concatenated assistant content from SSE stream text.

    Parses 'data: {json}' lines, extracting delta.content from each chunk.
    If no content is found, falls back to checking delta.reasoning_content
    for embedded tool call XML patterns (<function=...>...</function>).
    Returns concatenated content string, tool call string, or None.
    """
    parts: list[str] = []
    reasoning_parts: list[str] = []
    for line in sse_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            j = json.loads(payload)
            for choice in j.get("choices", []):
                delta = choice.get("delta", {})
                if isinstance(delta, dict):
                    if "content" in delta and delta["content"] is not None:
                        parts.append(str(delta["content"]))
                    # Collect reasoning_content regardless for fallback
                    if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                        reasoning_parts.append(str(delta["reasoning_content"]))
        except Exception:
            continue

    # If we have regular content, return it (preferred)
    if parts:
        return "".join(parts)

    # Fall back to extracting tool call from accumulated reasoning_content
    if reasoning_parts:
        full_reasoning = "".join(reasoning_parts)
        tool_call = _extract_tool_call_from_reasoning(full_reasoning)
        if tool_call:
            logger.info(
                "Extracted tool call from reasoning_content (streaming): %.80s",
                tool_call,
            )
            return tool_call

    return None


def _extract_delta_text_from_sse_chunk(chunk_text: str) -> str:
    """Extract assistant delta content from a single SSE chunk.

    Uses delta.content and delta.reasoning_content fields and ignores wrapper JSON.
    """
    if not chunk_text:
        return ""
    parts: list[str] = []
    for line in chunk_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            j = json.loads(payload)
        except Exception:
            continue
        for choice in j.get("choices", []):
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                continue
            for key in ("reasoning_content", "content"):
                value = delta.get(key)
                if value is not None:
                    parts.append(str(value))
    return "".join(parts)


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
    if max_runtime_seconds and runtime_seconds >= max_runtime_seconds:
        return "runtime"
    if max_completion_tokens and completion_tokens >= max_completion_tokens:
        return "completion_tokens"
    if _should_cutoff_for_repetition(response_text, repetition_min_pattern_chars, repetition_min_repeats):
        return "repetition"
    return None


def _should_invalidate_on_guardrail(
    guardrail_reason: Optional[str],
    invalidate_on_cutoff: bool,
    invalidate_on_repetition: bool,
) -> bool:
    if not guardrail_reason:
        return False
    if guardrail_reason == "repetition":
        return bool(invalidate_on_repetition)
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


def _detect_restore_signal_from_log_slice(
    log_path: Path,
    start_offset: int,
) -> bool:
    """Return True when restore evidence exists in newly appended log bytes."""
    if not log_path.exists():
        return False
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(max(0, int(start_offset)))
            data = f.read()
    except Exception:
        return False

    if not data:
        return False

    text = data.lower()
    phrases = (
        "restored context checkpoint",
        "load_session",
        "session restore",
        "restore session",
        "loading kv cache",
        "kv cache restored",
    )
    return any(p in text for p in phrases)


def _detect_restore_signal_from_llama_log(
    session_id: Optional[str],
    log_path: Optional[Path] = None,
    lookback_lines: int = 400,
) -> bool:
    """Best-effort compatibility signal from llama-server logs.

    Prefer session-id-specific lines when available.
    """
    if not session_id:
        return False

    target_path = log_path or _resolve_log_path("llama")
    if not target_path.exists():
        return False

    try:
        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-max(1, int(lookback_lines)):]
    except Exception:
        return False

    sid = str(session_id).strip()
    sid_lower = sid.lower()
    phrases = (
        "load_session",
        "session restore",
        "restore session",
        "loading kv cache",
        "kv cache restored",
        "restored context checkpoint",
    )

    for line in reversed(lines):
        text = line.strip().lower()
        if sid_lower in text and any(p in text for p in phrases):
            return True

    # Fallback: if no session id appears in log format, accept recent restore phrases.
    return any(any(p in line.strip().lower() for p in phrases) for line in lines)


log_dir: Optional[Path] = None
logger: logging.Logger = logging.getLogger("llama-proxy")


def extract_streamed_content_from_chunk(chunk_str: str) -> Optional[str]:
    """Extract concatenated delta.content and delta.reasoning_content strings from an SSE chunk.

    Returns the concatenated content (may include newlines as provided by the delta values)
    or None if no parseable content is found. Handles both 'content' and 'reasoning_content'
    fields used by models like Qwen3 during their thinking/reasoning phase.
    """
    if not chunk_str:
        return None
    try:
        contents: list[str] = []
        # First attempt: parse lines prefixed with 'data:' (SSE style)
        for line in chunk_str.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                try:
                    j = json.loads(payload)
                except Exception:
                    # ignore non-json data: lines
                    continue
                if isinstance(j, dict):
                    choices = j.get("choices") or []
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta") or {}
                        if isinstance(delta, dict):
                            # Extract both content and reasoning_content
                            for key in ("reasoning_content", "content"):
                                content_piece = delta.get(key)
                                if content_piece is not None:
                                    contents.append(str(content_piece))
        if contents:
            return "".join(contents)

        # Second attempt: try to parse the whole chunk as JSON
        s = chunk_str.strip()
        if s:
            try:
                j = json.loads(s)
                if isinstance(j, dict):
                    choices = j.get("choices") or []
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta") or {}
                        if isinstance(delta, dict):
                            # Extract both content and reasoning_content
                            for key in ("reasoning_content", "content"):
                                content_piece = delta.get(key)
                                if content_piece is not None:
                                    contents.append(str(content_piece))
                if contents:
                    return "".join(contents)
            except Exception:
                pass
    except Exception:
        pass
    return None


class ContentOnlyConsoleHandler(logging.StreamHandler):
    """Console handler that prints only streamed content for STREAM CHUNK records.

    For log records whose formatted message begins with the prefix
    "STREAM CHUNK | ", this handler will attempt to extract delta.content
    values from any JSON payloads inside the chunk and write only the
    concatenated content to the console stream (without adding extra
    newlines). Raw JSON is never displayed in the console - only extracted
    text content is shown.

    - reasoning_content is displayed in dim/grey
    - content is displayed in bold

    For other records, normal formatting is used.
    """

    PREFIX = "STREAM CHUNK | "
    # ANSI escape codes
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    def _extract_and_format_content(self, chunk_str: str) -> Optional[str]:
        """Extract content from chunk and apply formatting based on type.
        
        Returns formatted string with ANSI codes, or None if no content found.
        """
        if not chunk_str:
            return None
        
        parts: list[str] = []
        for line in chunk_str.splitlines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                continue
            try:
                j = json.loads(payload)
            except Exception:
                continue
            for choice in j.get("choices", []):
                delta = choice.get("delta", {})
                if not isinstance(delta, dict):
                    continue
                # Check reasoning_content first (dim)
                reasoning = delta.get("reasoning_content")
                if reasoning is not None:
                    parts.append(f"{self.DIM}{reasoning}{self.RESET}")
                # Check content (bold)
                content = delta.get("content")
                if content is not None:
                    parts.append(f"{self.BOLD}{content}{self.RESET}")
        
        return "".join(parts) if parts else None

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - exercised by integration
        try:
            msg = record.getMessage()
            if isinstance(msg, str) and msg.startswith(self.PREFIX):
                chunk_str = msg[len(self.PREFIX):]
                formatted = self._extract_and_format_content(chunk_str)
                if formatted:
                    # Ensure stream exists
                    if getattr(self, 'stream', None) is None:
                        self.stream = sys.stderr
                    try:
                        # Write formatted content. Do not append extra newline;
                        # the content may include its own newline characters.
                        self.stream.write(formatted)
                        try:
                            self.flush()
                        except Exception:
                            pass
                    except Exception:
                        pass
                # Always return for STREAM CHUNK - never show raw JSON in console
                # Raw JSON is written to the log file by the file handler
                return
            # Not a stream chunk — use default formatting
            super().emit(record)
        except Exception:
            # Best-effort: do not allow logging errors to crash application
            try:
                super().emit(record)
            except Exception:
                pass

# SSE clients for real-time status updates
sse_clients: set[asyncio.Queue] = set()
# SSE clients for log tail updates (counts + other notifications)
log_tail_clients: set[asyncio.Queue] = set()


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.environ.get(
            "LLAMA_PROXY_CONFIG",
            str(Path(__file__).parent.parent / "config.yaml")
        )
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> logging.Logger:
    """Setup logging with time-based rotation."""
    global log_dir

    # Check for dev mode
    is_dev = os.environ.get("LLAMA_PROXY_DEV") == "1"

    log_config = config.get("logging", {})
    rotation_hours = log_config.get("rotation_hours", 6)
    retention_days = log_config.get("retention_days", 90)
    log_level = log_config.get("level", "INFO")

    if is_dev:
        # Dev mode: use XDG-based dev log directory with DEBUG level
        xdg_state = os.environ.get("XDG_STATE_HOME", os.path.join(os.path.expanduser("~"), ".local", "state"))
        log_dir = Path(xdg_state) / "llama-proxy-dev" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_level = "DEBUG"
        print(f"[INFO] Dev mode: using log directory {log_dir} at level {log_level}")
    else:
        log_dir = Path(log_config.get("directory", "/var/log/llama-proxy"))
        # Try to create log directory, fall back to local logs directory if permission denied
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            log_dir = Path(__file__).parent / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Using local log directory: {log_dir}")

    # Calculate backup count based on retention days and rotation interval
    # (retention_days * 24 hours / rotation_hours)
    backup_count = (retention_days * 24) // rotation_hours

    # Create logger
    logger = logging.getLogger("llama-proxy")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # File handler with rotation
    log_file = log_dir / "proxy.log"
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="H",  # Hourly rotation
        interval=rotation_hours,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(file_handler)
    
    # Console handler for debugging (content-only for STREAM CHUNK messages)
    console_handler = ContentOnlyConsoleHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(console_handler)
    
    return logger


async def broadcast_status(event_type: str, data: dict):
    """Broadcast a status event to all connected SSE clients."""
    event_data = json.dumps({"type": event_type, **data})
    message = f"data: {event_data}\n\n"
    
    # Send to all connected clients
    dead_clients = set()
    for queue in sse_clients:
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            dead_clients.add(queue)
    
    # Clean up dead clients
    for client in dead_clients:
        sse_clients.discard(client)


def broadcast_status_sync(event_type: str, data: dict):
    """Synchronous wrapper to broadcast status (for use in sync code)."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(broadcast_status(event_type, data))
    except RuntimeError:
        # No running loop, skip broadcast
        pass


def _counts_file_path() -> Path:
    """Return path to the persisted counts file inside log_dir (or local logs)."""
    if log_dir:
        return log_dir / counts_filename
    return Path(__file__).parent / "logs" / counts_filename


def load_counts():
    global request_counts
    try:
        path = _counts_file_path()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                request_counts = json.load(f)
        else:
            request_counts = {}
    except Exception:
        request_counts = {}


def save_counts_sync():
    try:
        path = _counts_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(request_counts, f, indent=2)
        tmp.replace(path)
    except Exception as e:
        logger.error(f"Failed to persist request counts: {e}")


def _token_file_path() -> Path:
    if log_dir:
        return log_dir / token_counts_filename
    return Path(__file__).parent / "logs" / token_counts_filename


def load_token_counts():
    global token_counts
    try:
        path = _token_file_path()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                token_counts = json.load(f)
        else:
            token_counts = {}
    except Exception:
        token_counts = {}


def save_token_counts_sync():
    try:
        path = _token_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(token_counts, f, indent=2)
        tmp.replace(path)
    except Exception as e:
        logger.error(f"Failed to persist token counts: {e}")


async def save_token_counts():
    await asyncio.to_thread(save_token_counts_sync)


async def save_counts():
    await asyncio.to_thread(save_counts_sync)


async def _counts_persist_loop():
    global counts_dirty, counts_persist_task
    try:
        while True:
            await asyncio.sleep(2.0)
            if counts_dirty:
                try:
                    await save_counts()
                    counts_dirty = False
                except Exception:
                    pass
    finally:
        counts_persist_task = None


async def _tokens_persist_loop():
    global tokens_dirty, tokens_persist_task
    try:
        while True:
            await asyncio.sleep(2.0)
            if tokens_dirty:
                try:
                    await save_token_counts()
                    tokens_dirty = False
                except Exception:
                    pass
    finally:
        tokens_persist_task = None


async def query_llama_status() -> dict:
    """
    Query llama-server HTTP endpoints for model metadata.

    Attempts HTTP GET to /model then /status and returns parsed JSON if successful.
    If those endpoints are not present or do not include n_ctx / KV cache values,
    returns null for those fields.

    Returns dict with:
      - n_ctx: max context size (int) or None
      - kv_cache_tokens: KV cache token count (int) or None
      - llama_server_running: bool
      - router_mode: bool
    """
    # allow updating/reading endpoint cache/failures
    global _llama_status_endpoint_cache, _llama_status_endpoint_failures

    result = {
        "n_ctx": None,
        "kv_cache_tokens": None,
        "llama_server_running": llama_process is not None and llama_process.poll() is None,
        "router_mode": False
    }

    if not result["llama_server_running"]:
        return result

    server_config = config.get("server", {})
    llama_port = server_config.get("llama_server_port", 8080)

    # One-time discovery: probe metadata endpoints once per llama-server process
    client = _http_client if _http_client else httpx.AsyncClient(timeout=5.0)
    try:
        async def _do_discovery_if_needed():
            """Discover a working metadata endpoint once per process.

            Returns a (n_ctx, kv_cache_tokens) tuple discovered during probing
            so callers can use the data without re-requesting mocked responses
            (important for tests that provide a sequence of responses).
            """
            global _llama_status_endpoint_cache, _llama_status_discovered, _llama_status_discovered_pid
            # If discovery was already done for this pid, skip.
            try:
                current_pid = getattr(llama_process, 'pid', None)
            except Exception:
                current_pid = None
            if _llama_status_discovered and _llama_status_discovered_pid == current_pid:
                return None, None

            endpoints = ["/model", "/status", "/models", "/v1/models"]
            found_n = None
            found_kv = None
            for endpoint in endpoints:
                try:
                    url = f"http://localhost:{llama_port}{endpoint}"
                    resp = await client.get(url, timeout=5.0)
                    if getattr(resp, 'status_code', None) == 200:
                        # remember endpoint
                        if not _llama_status_endpoint_cache:
                            _llama_status_endpoint_cache = endpoint
                        _llama_status_endpoint_failures.pop(endpoint, None)

                        # attempt to parse JSON/text for n_ctx / kv fields
                        data = None
                        if hasattr(resp, 'json'):
                            try:
                                maybe = resp.json()
                                data = await maybe if asyncio.iscoroutine(maybe) else maybe
                            except Exception:
                                data = None
                        if data is None and hasattr(resp, 'text'):
                            try:
                                txt = resp.text if not asyncio.iscoroutine(resp.text) else await resp.text
                                data = json.loads(txt)
                            except Exception:
                                data = None

                        if isinstance(data, dict):
                            if found_n is None:
                                found_n = data.get("n_ctx") or data.get("n_ctx_total")
                            if found_kv is None:
                                found_kv = data.get("kv_cache_tokens") or data.get("kv_cache_token_count")

                        # If we've discovered both values, stop probing
                        if found_n is not None and found_kv is not None:
                            break
                except Exception:
                    # ignore and try next
                    continue

            # Mark discovery done for this pid even if none found
            _llama_status_discovered = True
            try:
                _llama_status_discovered_pid = getattr(llama_process, 'pid', None)
            except Exception:
                _llama_status_discovered_pid = None

            return found_n, found_kv

        # Run discovery and capture any discovered metadata so we can
        # return parsed values immediately (important for tests that
        # provide a sequence of mocked responses consumed during discovery).
        found_n, found_kv = await _do_discovery_if_needed()
        if found_n is not None:
            result["n_ctx"] = found_n
        if found_kv is not None:
            result["kv_cache_tokens"] = found_kv

        # If we have a cached endpoint, try it first
        if _llama_status_endpoint_cache:
            try:
                url = f"http://localhost:{llama_port}{_llama_status_endpoint_cache}"
                response = await client.get(url, timeout=5.0)
                if getattr(response, 'status_code', None) == 200:
                    data = None
                    if hasattr(response, 'json'):
                        try:
                            maybe = response.json()
                            data = await maybe if asyncio.iscoroutine(maybe) else maybe
                        except Exception:
                            data = None
                    if data is None and hasattr(response, 'text'):
                        try:
                            txt = response.text if not asyncio.iscoroutine(response.text) else await response.text
                            data = json.loads(txt)
                        except Exception:
                            data = None

                    if isinstance(data, dict):
                        if result["n_ctx"] is None:
                            result["n_ctx"] = data.get("n_ctx") or data.get("n_ctx_total")
                        if result["kv_cache_tokens"] is None:
                            result["kv_cache_tokens"] = (
                                data.get("kv_cache_tokens") or data.get("kv_cache_token_count")
                            )
                else:
                    # cache no longer valid; clear and allow future rediscovery
                    _llama_status_endpoint_failures[_llama_status_endpoint_cache] = time.time()
                    _llama_status_endpoint_cache = None
                    _llama_status_discovered = False
                    _llama_status_discovered_pid = None
            except Exception:
                # ignore and fallthrough to props check
                pass

        # As a fallback, check /props to detect router mode
        try:
            props_url = f"http://localhost:{llama_port}/props"
            response = await client.get(props_url, timeout=5.0)
            if getattr(response, "status_code", None) == 200:
                props = None
                if hasattr(response, "json"):
                    try:
                        maybe = response.json()
                        props = await maybe if asyncio.iscoroutine(maybe) else maybe
                    except Exception:
                        props = None
                if props is None and hasattr(response, "text"):
                    try:
                        txt = response.text if not asyncio.iscoroutine(response.text) else await response.text
                        props = json.loads(txt)
                    except Exception:
                        props = None
                if isinstance(props, dict):
                    result["router_mode"] = True
        except Exception:
            pass
    finally:
        if not _http_client:
            try:
                await client.aclose()
            except Exception:
                pass

    return result


async def _periodic_broadcast_loop():
    """Periodically broadcast current counts/tokens to connected log-tail clients.

    This ensures UI updates even if no direct increment message races occur.
    Also queries llama-server for status and broadcasts stats.
    """
    global periodic_broadcast_task
    try:
        while True:
            try:
                await asyncio.sleep(1.0)
                snap_c = {}
                snap_t = {}
                async with counts_lock:
                    snap_c = dict(request_counts)
                async with token_lock:
                    snap_t = dict(token_counts)

                llama_status = await query_llama_status()
                
                total_sent = token_counts.get("total_sent", 0)
                total_recv = token_counts.get("total_recv", 0)

                if log_tail_clients:
                    for q in list(log_tail_clients):
                        try:
                            q.put_nowait({
                                "counts": snap_c, 
                                "tokens": snap_t,
                                "llama_status": llama_status,
                                "total_sent": total_sent,
                                "total_recv": total_recv
                            })
                        except asyncio.QueueFull:
                            continue
                
                if sse_clients:
                    status_data = {
                        "type": "status",
                        "current_model": current_model,
                        "llama_server_running": llama_status["llama_server_running"],
                        "n_ctx": llama_status["n_ctx"],
                        "kv_cache_tokens": llama_status["kv_cache_tokens"],
                        "total_sent": total_sent,
                        "total_recv": total_recv
                    }
                    event_data = json.dumps(status_data)
                    message = f"data: {event_data}\n\n"
                    dead_clients = set()
                    for q in sse_clients:
                        try:
                            q.put_nowait(message)
                        except asyncio.QueueFull:
                            dead_clients.add(q)
                    for client in dead_clients:
                        sse_clients.discard(client)
                        
            except asyncio.CancelledError:
                break
            except Exception:
                # ignore transient errors
                pass
    finally:
        periodic_broadcast_task = None


async def _increment_count(key: str):
    """Increment the in-memory counter for a request key and persist."""
    try:
        async with counts_lock:
            prev = request_counts.get(key, 0)
            request_counts[key] = prev + 1
            # mark dirty but don't persist immediately; background task will persist
            global counts_dirty
            counts_dirty = True
            logger.debug(f"_increment_count: key={key} prev={prev} new={request_counts[key]}")
    except Exception as e:
        logger.error(f"Error incrementing request count: {e}")
    # Broadcast updated counts to connected log tail clients
    try:
        snapshot = None
        async with counts_lock:
            snapshot = dict(request_counts)

        # Send to all connected log-tail queues
        for q in list(log_tail_clients):
            try:
                q.put_nowait({"counts": snapshot})
            except asyncio.QueueFull:
                # skip slow listeners
                continue
    except Exception:
        pass


async def _increment_count_multi(keys: list[str]):
    """Increment multiple request count keys in one go and broadcast snapshot."""
    try:
        async with counts_lock:
            # log previous values for debugging
            prevs = {k: request_counts.get(k, 0) for k in keys}
            for key in keys:
                request_counts[key] = prevs.get(key, 0) + 1
            global counts_dirty
            counts_dirty = True
            logger.debug(f"_increment_count_multi: keys={keys} prevs={prevs} new_vals={{k: request_counts[k] for k in keys}}")
    except Exception as e:
        logger.error(f"Error incrementing request counts: {e}")

    # Broadcast snapshot
    try:
        snapshot = None
        async with counts_lock:
            snapshot = dict(request_counts)

        for q in list(log_tail_clients):
            try:
                q.put_nowait({"counts": snapshot})
            except asyncio.QueueFull:
                continue
    except Exception:
        pass


async def _increment_tokens(key_prefix: str, key: str, n: int):
    """Increment token counts and persist; key_prefix is 'sent' or 'recv'."""
    try:
        async with token_lock:
            pk = key_prefix + ':' + key
            prev = token_counts.get(pk, 0)
            token_counts[pk] = prev + n
            total_key = 'total_sent' if key_prefix == 'sent' else 'total_recv'
            prev_total = token_counts.get(total_key, 0)
            token_counts[total_key] = prev_total + n
            global tokens_dirty
            tokens_dirty = True
            logger.debug(f"_increment_tokens: prefix={key_prefix} key={key} n={n} prev={prev} new={token_counts[pk]} total_prev={prev_total} total_new={token_counts[total_key]}")
    except Exception as e:
        logger.error(f"Error incrementing token counts: {e}")
    # Broadcast token snapshot
    try:
        snap = None
        async with token_lock:
            snap = dict(token_counts)
        for q in list(log_tail_clients):
            try:
                q.put_nowait({"tokens": snap})
            except asyncio.QueueFull:
                continue
    except Exception:
        pass


def get_model_config(model_name: Optional[str]) -> Optional[dict]:
    """
    Get model configuration by name or alias.
    
    Supports wildcard patterns in aliases using fnmatch syntax:
    - '*' matches any sequence of characters
    - '?' matches any single character
    - '[seq]' matches any character in seq
    - '[!seq]' matches any character not in seq
    
    Examples:
    - 'gpt*' matches 'gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'
    - 'claude-3-*' matches 'claude-3-opus', 'claude-3-sonnet'
    """
    if model_name is None:
        return None
    
    models = config.get("models", {})
    
    # Direct match
    if model_name in models:
        return models[model_name]
    
    model_name_lower = model_name.lower()
    
    # Check exact aliases first (higher priority)
    for name, model_cfg in models.items():
        aliases = model_cfg.get("aliases", [])
        for alias in aliases:
            alias_lower = alias.lower()
            # Skip wildcard patterns in first pass
            if '*' in alias or '?' in alias or '[' in alias:
                continue
            if model_name_lower == alias_lower:
                return model_cfg
    
    # Check wildcard patterns
    for name, model_cfg in models.items():
        aliases = model_cfg.get("aliases", [])
        for alias in aliases:
            alias_lower = alias.lower()
            # Only process wildcard patterns
            if '*' in alias or '?' in alias or '[' in alias:
                if fnmatch(model_name_lower, alias_lower):
                    return model_cfg
    
    return None


def _should_force_full_prompt(model_cfg: Optional[dict]) -> bool:
    if not isinstance(model_cfg, dict):
        return False
    return bool(model_cfg.get("force_full_prompt") or model_cfg.get("disable_delta"))


def get_local_model_name(model_name: Optional[str]) -> Optional[str]:
    """Get the llama model name for a given model."""
    model_cfg = get_model_config(model_name)
    if model_cfg and model_cfg.get("type") == "local":
        llama_model = model_cfg.get("llama_model")
        if not isinstance(llama_model, str) or not llama_model:
            raise HTTPException(
                status_code=500,
                detail=f"Local model configuration missing llama_model for: {model_name}"
            )
        llama_model_str: str = llama_model
        if llama_model:
            return llama_model
        return model_name
    return None


def _resolve_slot_model_name(
    requested_model: Optional[str],
    current_model: Optional[str],
    server_config: dict,
) -> Optional[str]:
    """Resolve the llama model name used for slot endpoints in router mode."""
    candidate = requested_model or current_model
    if not candidate:
        return None
    if server_config.get("llama_router_mode", False):
        try:
            return get_local_model_name(candidate) or candidate
        except HTTPException:
            return candidate
    return candidate


async def wait_for_llama_server(timeout: int = 300) -> bool:
    """Wait for llama-server to be ready."""
    server_config = config.get("server", {})
    llama_port = server_config.get("llama_server_port", 8080)
    health_url = f"http://localhost:{llama_port}/health"
    
    start_time = time.time()
    client = _http_client if _http_client else httpx.AsyncClient(timeout=httpx.Timeout(5.0))
    try:
        while time.time() - start_time < timeout:
            # Check if llama process died
            if llama_process is not None and llama_process.poll() is not None:
                exit_code = llama_process.returncode
                logger.error(f"llama-server process exited with code {exit_code}")
                return False
            
            try:
                response = await client.get(health_url, timeout=5)
                if response.status_code == 200:
                    logger.info("llama-server is ready")
                    return True
            except asyncio.CancelledError:
                logger.info("Wait for llama-server cancelled")
                raise
            except Exception:
                pass
            try:
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                logger.info("Wait for llama-server cancelled")
                raise
    finally:
        if not _http_client:
            try:
                await client.aclose()
            except Exception:
                pass
    
    logger.error(f"llama-server failed to start within {timeout} seconds")
    return False


async def router_load_model(model_name: str) -> bool:
    """Request router-mode llama-server to load a model."""
    server_config = config.get("server", {})
    llama_port = server_config.get("llama_server_port", 8080)
    url = f"http://localhost:{llama_port}/models/load"
    payload = {"model": model_name}

    client = _http_client if _http_client else httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    try:
        try:
            response = await client.post(url, json=payload, timeout=30)
            if response.status_code != 200:
                body = response.text
                body_l = (body or "").lower()
                if response.status_code == 400 and ("already loaded" in body_l or "already running" in body_l):
                    logger.info(f"Router model already loaded: {model_name}")
                    # update last-used timestamp when model already loaded/running
                    try:
                        model_last_used[model_name] = datetime.utcnow().isoformat()
                    except Exception:
                        pass
                    return True
                logger.error(f"Router load failed for {model_name}: {response.status_code} {body}")
                return False
            # Update last-used timestamp on successful load
            try:
                model_last_used[model_name] = datetime.utcnow().isoformat()
            except Exception:
                pass
            try:
                metrics.record_model_loaded(model_name)
            except Exception:
                pass
            return True
        except Exception as e:
            logger.error(f"Router load failed for {model_name}: {e}")
            return False
    finally:
        if not _http_client:
            try:
                await client.aclose()
            except Exception:
                pass


async def router_list_models() -> Optional[dict]:
    """List models from router-mode llama-server."""
    server_config = config.get("server", {})
    llama_port = server_config.get("llama_server_port", 8080)
    url = f"http://localhost:{llama_port}/models"

    client = _http_client if _http_client else httpx.AsyncClient(timeout=5.0)
    try:
        response = await client.get(url, timeout=5.0)
        if response.status_code != 200:
            logger.warning(f"Router list models failed: {response.status_code} {response.text}")
            return None
        return response.json()
    except Exception as e:
        logger.warning(f"Router list models failed: {e}")
        return None
    finally:
        if not _http_client:
            try:
                await client.aclose()
            except Exception:
                pass


def _extract_router_model_ids(router_models: Optional[dict]) -> list[str]:
    if not isinstance(router_models, dict):
        return []
    models_payload = router_models.get("data") or router_models.get("models") or []
    if isinstance(models_payload, list):
        return [str(m.get("id")) for m in models_payload if isinstance(m, dict) and m.get("id")]
    return []


async def router_is_model_loaded(model_name: str) -> bool:
    router_models = await router_list_models()
    if not isinstance(router_models, dict):
        return False
    models_payload = router_models.get("data") or router_models.get("models") or []
    if not isinstance(models_payload, list):
        return False

    for m in models_payload:
        if not (isinstance(m, dict) and m.get("id") == model_name):
            continue

        # Different llama-server/router builds expose model readiness with
        # varying schemas. Treat model presence as loaded when no explicit
        # status is provided to avoid false negatives that can pin
        # background_loads and keep returning scheduled=False + 503.
        status = m.get("status")
        if status is None:
            return True

        if isinstance(status, dict):
            value = str(status.get("value", "")).strip().lower()
            if value in {"loaded", "ready", "running", "active"}:
                return True
            if value in {"loading", "unloaded", "error", "failed"}:
                return False
            # Unknown status value: fall back to presence as loaded.
            return True

        if isinstance(status, str):
            value = status.strip().lower()
            if value in {"loaded", "ready", "running", "active"}:
                return True
            if value in {"loading", "unloaded", "error", "failed"}:
                return False

        # Presence with unrecognized status shape/value: consider loaded.
        return True

    return False


async def router_wait_for_model(model_name: str, timeout: int = 300, interval: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await router_is_model_loaded(model_name):
            return True
        await asyncio.sleep(interval)
    return False


async def router_preload_models(model_names: list[str]) -> bool:
    """Preload a list of models in router mode."""
    for model_name in model_names:
        if not await router_load_model(model_name):
            return False
    return True


def _normalize_outgoing_headers(in_headers: dict, buffered: bool = False) -> dict:
    """Normalize headers before sending to clients.

    - If buffered=True (we are sending a full body via Response), remove
      any Transfer-Encoding header so frameworks/servers may set a proper
      Content-Length for the buffered body.
    - If buffered=False (we are streaming and will not pre-compute a
      Content-Length), remove Content-Length if Transfer-Encoding is present
      to avoid sending both headers.
    """
    if not in_headers:
        return {}
    lc_map = {k.lower(): k for k in in_headers.keys()}
    out = dict(in_headers)

    if buffered:
        # We're returning a buffered body; ensure Transfer-Encoding is not forwarded
        if 'transfer-encoding' in lc_map:
            out.pop(lc_map['transfer-encoding'], None)
    else:
        # Streaming or unknown delivery: do not forward Content-Length when TE exists
        if 'transfer-encoding' in lc_map and 'content-length' in lc_map:
            out.pop(lc_map['content-length'], None)

    return out


def start_llama_server(model: Optional[str]) -> Optional[subprocess.Popen]:
    """Start the llama-server with the specified model inside distrobox."""
    global llama_process, llama_log_file, current_model, last_start_failure
    
    server_config = config.get("server", {})
    # Default to the repository root `start-llama.sh` if not specified in config
    script_path = server_config.get(
        "llama_start_script",
        str(Path(__file__).parent.parent / "start-llama.sh")
    )
    distrobox_name = server_config.get("distrobox_name", "llama")
    llama_port = server_config.get("llama_server_port", 8080)
    
    # Set environment variables
    env = os.environ.copy()
    env["PORT"] = str(llama_port)

    # Allow overriding which llama-server binary the start script should invoke
    # inside the distrobox. This is useful when the host has a custom build
    # (e.g. ~/llama.cpp/build/bin/llama-server) but the container image has a
    # different system-installed binary. The start script can honour the
    # LLAMA_SERVER_BIN env var (if present) to use the specified binary path.
    llama_server_bin = server_config.get("llama_server_bin")
    if llama_server_bin:
        env["LLAMA_SERVER_BIN"] = str(llama_server_bin)
    slot_save_path = server_config.get("session_slot_save_path")
    if slot_save_path:
        env["LLAMA_SLOT_SAVE_PATH"] = str(slot_save_path)
    # Export a flag so the start script can include `--no-mmap` for router
    # launches started by the proxy. Default to enabling no-mmap for router
    # processes unless explicitly disabled in config.
    try:
        if server_config.get("llama_no_mmap", True):
            env["LLAMA_SERVER_NO_MMAP"] = "1"
    except Exception:
        pass
    
    router_mode = False
    if model is None:
        router_mode = True
    elif isinstance(model, str) and model.strip().lower() == "router":
        router_mode = True

    mode_label = "router" if router_mode else f"model: {model}"
    logger.info(f"Starting llama-server with {mode_label} in distrobox '{distrobox_name}'")

    # Rotate llama-server logs (keep last 15)
    if log_dir:
        llama_log_path = log_dir / "llama-server.log"
        rotate_llama_logs(llama_log_path, keep=15)
        llama_log_file = open(llama_log_path, "w")
    else:
        llama_log_file = subprocess.DEVNULL

    # Try running via distrobox first; if distrobox is not available or fails,
    # fall back to running the start script directly. Capture and log errors
    # so failures after reboot are diagnosable. Implement a short retry loop
    # with backoff to tolerate boot-order races.
    if router_mode:
        llama_models_max = server_config.get("llama_models_max")
        if llama_models_max:
            env["LLAMA_MODELS_MAX"] = str(llama_models_max)
        llama_models_preset = server_config.get("llama_models_preset")
        if llama_models_preset:
            env["LLAMA_MODELS_PRESET"] = str(llama_models_preset)

        distrobox_cmd = ["distrobox", "enter", distrobox_name, "--", script_path, "router"]
    else:
        if model is None:
            msg = "Model name is required when not running in router mode"
            logger.error(msg)
            last_start_failure = msg
            broadcast_status_sync("error", {"message": msg, "current_model": None, "llama_server_running": False})
            return None
        distrobox_cmd = ["distrobox", "enter", distrobox_name, "--", script_path, model]

    # Helper to start a subprocess and capture immediate stderr/stdout if it
    # exits quickly. Returns a tuple (Popen|None, captured_output_str|None).
    def _spawn_and_capture(cmd):
        try:
            # Use pipes so we can capture early output if the child exits
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
        except FileNotFoundError as e:
            logger.warning(f"Command not found when starting llama-server: {cmd[0]}: {e}")
            return None, f"Command not found: {cmd[0]}: {e}"
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Failed to spawn command {cmd}: {e}\n{tb}")
            return None, f"Spawn failed: {e}\n{tb}"

        # Give the child a short window to produce output and check if it exits
        try:
            outs, _ = proc.communicate(timeout=3)
            # If communicate returned, the process exited quickly; include output
            return None, outs
        except subprocess.TimeoutExpired:
            # Process is still running — good.
            # Reattach stdout to file for long-running logging
            try:
                if llama_log_file and proc.stdout:
                    # Write the already-read portion (if any) and keep piping
                    # from proc.stdout asynchronously is complex; as a compromise
                    # we'll spawn a background thread to stream output to file.
                    import threading

                    def _stream_output(src, dst):
                        try:
                            for line in src:
                                dst.write(line)
                                dst.flush()
                                # Display prompt processing progress to console
                                try:
                                    line_str = line.decode('utf-8', errors='replace') if isinstance(line, bytes) else str(line)
                                    # Detect prompt processing progress lines from llama-server
                                    # Format: "slot N : prompt processing ..." or similar
                                    if 'prompt processing' in line_str.lower() or 'prompt eval' in line_str.lower():
                                        # Try to parse structured progress information and show a
                                        # concise, in-place progress indicator. Fall back to
                                        # printing the raw line if parsing/display fails.
                                        try:
                                            parsed = extract_progress_data(line_str)
                                            if parsed:
                                                n_tokens, prog_frac = parsed
                                                # Estimate total tokens when progress fraction is available
                                                try:
                                                    total_tokens = int(n_tokens / prog_frac) if prog_frac and prog_frac > 0 else n_tokens
                                                except Exception:
                                                    total_tokens = n_tokens
                                                try:
                                                    progress_str = format_progress(n_tokens, total_tokens, prog_frac)
                                                    # Write the progress string in-place (no newline)
                                                    sys.stderr.write(progress_str)
                                                    sys.stderr.flush()
                                                except Exception:
                                                    # Formatting failed — fall back to raw line
                                                    progress_line = line_str.strip()
                                                    if progress_line:
                                                        sys.stderr.write(f"\r{progress_line}\n")
                                                        sys.stderr.flush()
                                            else:
                                                # No structured progress found — print raw line
                                                progress_line = line_str.strip()
                                                if progress_line:
                                                    sys.stderr.write(f"\r{progress_line}\n")
                                                    sys.stderr.flush()
                                        except Exception:
                                            try:
                                                progress_line = line_str.strip()
                                                if progress_line:
                                                    sys.stderr.write(f"\r{progress_line}\n")
                                                    sys.stderr.flush()
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    t = threading.Thread(target=_stream_output, args=(proc.stdout, llama_log_file), daemon=True)
                    t.start()
            except Exception:
                pass
            return proc, None

    # Retry loop configuration
    retries = 4
    backoff = 3  # seconds base (initial retry period)
    tried_cmds = []

    # The server MUST run inside distrobox/container. Do not fall back to
    # running the start script directly on the host. This avoids running
    # models outside a controlled container environment.
    if not shutil.which("distrobox"):
        msg = "distrobox not found in PATH; llama-server must be started inside distrobox"
        logger.error(msg)
        last_start_failure = msg
        broadcast_status_sync("error", {"message": msg, "current_model": None, "llama_server_running": False})
        return None

    # Try distrobox only
    for attempt in range(retries):
        proc, out = _spawn_and_capture(distrobox_cmd)
        tried_cmds.append((distrobox_cmd, out))
        if proc is not None:
            # Do not set `current_model` here. The proxy must only mark a
            # model as active after the llama-server is actually ready and
            # the model has finished loading. `ensure_model_loaded` is
            # responsible for setting `current_model` once startup and
            # model load succeed. Setting it here would reflect the
            # requested model prematurely (while still switching).
            return proc
        # If out is present, the command exited quickly with output; log and retry
        logger.warning(f"distrobox attempt {attempt+1} failed quickly: {out}")
        time.sleep(backoff * (2 ** attempt))

    # All distrobox attempts failed — assemble a helpful diagnostic message
    msg_lines = ["Failed to start llama-server using distrobox. Attempts:"]
    for cmd, out in tried_cmds:
        cmd_str = " ".join(cmd)
        snippet = (out or "(no immediate output)").strip()
        if len(snippet) > 1000:
            snippet = snippet[:1000] + "...[truncated]"
        msg_lines.append(f"- {cmd_str}: {snippet}")
    msg_lines.append("")
    msg_lines.append("Hints:")
    msg_lines.append(" - Ensure distrobox/podman rootless runtime is available (enable user linger: sudo loginctl enable-linger <user>)")
    msg_lines.append(" - Ensure /etc/subuid and /etc/subgid contain mappings for the user and that /usr/bin/newuidmap and newgidmap are setuid root")
    msg = "\n".join(msg_lines)
    logger.error(msg)
    # record last failure for diagnostics and broadcast
    last_start_failure = msg
    broadcast_status_sync("error", {"message": msg, "current_model": None, "llama_server_running": False})
    return None


def rotate_llama_logs(current_log: Path, keep: int = 15):
    """Rotate llama-server logs, keeping the last N copies."""
    if not current_log.exists():
        return
    
    # Find existing rotated logs
    log_dir = current_log.parent
    base_name = current_log.stem
    suffix = current_log.suffix
    
    # Get all existing rotated logs sorted by number (descending)
    rotated_logs = []
    for f in log_dir.glob(f"{base_name}.*{suffix}"):
        try:
            num = int(f.stem.split(".")[-1])
            rotated_logs.append((num, f))
        except ValueError:
            continue
    
    rotated_logs.sort(key=lambda x: x[0], reverse=True)
    
    # Delete logs beyond the keep limit (accounting for the new rotation)
    for num, f in rotated_logs:
        if num >= keep:
            f.unlink()
            logger.debug(f"Deleted old llama-server log: {f}")
    
    # Rotate existing logs (N -> N+1)
    for num, f in rotated_logs:
        if num < keep:
            new_name = log_dir / f"{base_name}.{num + 1}{suffix}"
            f.rename(new_name)
    
    # Rotate current log to .1
    if current_log.exists():
        current_log.rename(log_dir / f"{base_name}.1{suffix}")


def stop_llama_server():
    """Stop the currently running llama-server."""
    global llama_process, llama_log_file, current_model, backend_ready
    
    server_config = config.get("server", {})
    distrobox_name = server_config.get("distrobox_name", "llama")
    
    # First, try to kill llama-server inside the distrobox
    try:
        subprocess.run(
            ["distrobox", "enter", distrobox_name, "--", "pkill", "-f", "llama-server"],
            timeout=10,
            capture_output=True
        )
        logger.info("Sent kill signal to llama-server inside distrobox")
    except Exception as e:
        logger.warning(f"Failed to kill llama-server inside distrobox: {e}")
    
    if llama_process is not None:
        pid = getattr(llama_process, 'pid', 'N/A')
        logger.info(f"Stopping llama-server wrapper (PID: {pid})")
        # Only clean up process and model state if llama_process looks like
        # a real subprocess (has terminate/kill/wait methods). If it's a
        # test mock or invalid object, skip process cleanup.
        is_real_process = hasattr(llama_process, 'terminate') or hasattr(llama_process, 'kill')
        if is_real_process:
            previous_model = current_model
            llama_process.terminate()
            try:
                llama_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("llama-server wrapper did not terminate gracefully, killing...")
                if hasattr(llama_process, 'kill'):
                    llama_process.kill()
                if hasattr(llama_process, 'wait'):
                    llama_process.wait()
            llama_process = None
            try:
                if previous_model:
                    metrics.record_model_unloaded(previous_model)
            except Exception:
                pass
            current_model = None
            backend_ready = False
            logger.info("llama-server stopped")
        else:
            llama_process = None
            backend_ready = False
            logger.info("llama-server stop skipped (no valid process)")
    
    # Close log file if open
    if llama_log_file is not None and llama_log_file != subprocess.DEVNULL:
        try:
            llama_log_file.close()
        except Exception:
            pass
        llama_log_file = None


async def ensure_model_loaded(requested_model: Optional[str]) -> bool:
    """
    Ensure the requested model is loaded in llama-server.
    Returns True if the model is ready, False if there was an error.
    """
    global llama_process, current_model, backend_ready

    llama_model = get_local_model_name(requested_model)
    if llama_model is None:
        backend_ready = False
        return False

    server_config = config.get("server", {})
    router_mode = server_config.get("llama_router_mode", False)

    # Use a try/finally around the model switch lock so we can reliably
    # decrement the global refcount if this invocation incremented it.
    incremented_here = False
    try:
        async with model_switch_lock:
            if current_model == llama_model and llama_process is not None:
                # Check if process is still running
                if llama_process.poll() is None:
                    backend_ready = True
                    return True
                else:
                    logger.warning("llama-server process died, restarting...")

            # If no background load marker exists for this model then this
            # synchronous path should increment the refcount so status
            # endpoints observe switching across threads/loops.
            try:
                if not background_loads.get(llama_model):
                    _inc_model_switch_refcount()
                    incremented_here = True
            except Exception:
                # Best-effort: do not fail switching due to refcount errors
                pass

            # Broadcast that we're switching models
            await broadcast_status("switching", {
                "target_model": llama_model,
                "previous_model": current_model
            })

            timeout = server_config.get("llama_startup_timeout", 300)

            if router_mode:
                if llama_process is None or llama_process.poll() is not None:
                    llama_process = start_llama_server(None)

                    if llama_process is None:
                        logger.error("start_llama_server failed to spawn router process")
                        backend_ready = False
                        return False

                    if not await wait_for_llama_server(timeout):
                        await broadcast_status("error", {
                            "message": "Failed to start router-mode llama-server",
                            "current_model": None,
                            "llama_server_running": False
                        })
                        stop_llama_server()
                        backend_ready = False
                        return False

                if not await router_load_model(llama_model):
                    await broadcast_status("error", {
                        "message": f"Failed to load model {llama_model} via router",
                        "current_model": None,
                        "llama_server_running": True
                    })
                    backend_ready = False
                    return False

                # Enforce embeddings pinned: ensure embeddings preset remains loaded
                embeddings_preset = server_config.get("embeddings_model")
                if embeddings_preset:
                    try:
                        await router_load_model(embeddings_preset)
                        await router_wait_for_model(embeddings_preset, timeout=server_config.get("llama_embed_load_timeout", 30))
                    except Exception:
                        logger.warning("Failed to ensure embeddings preset is loaded/pinned")

                load_timeout = server_config.get("llama_model_load_timeout", timeout)
                if not await router_wait_for_model(llama_model, timeout=load_timeout):
                    await broadcast_status("error", {
                        "message": f"Timed out waiting for model {llama_model} to load",
                        "current_model": None,
                        "llama_server_running": True
                    })
                    backend_ready = False
                    return False

                current_model = llama_model
                try:
                    metrics.record_model_loaded(llama_model)
                except Exception:
                    pass
                await broadcast_status("ready", {
                    "current_model": llama_model,
                    "llama_server_running": True
                })
                backend_ready = True
                return True

            # Need to switch models or restart
            stop_llama_server()

            llama_process = start_llama_server(llama_model)

            # If starting the process failed immediately (start_llama_server returns None),
            # fail fast instead of waiting the full timeout. start_llama_server already
            # broadcasts a detailed error message.
            if llama_process is None:
                logger.error(f"start_llama_server failed to spawn process for model {llama_model}")
                backend_ready = False
                return False

            if await wait_for_llama_server(timeout):
                current_model = llama_model
                try:
                    metrics.record_model_loaded(llama_model)
                except Exception:
                    pass
                # Broadcast success
                await broadcast_status("ready", {
                    "current_model": llama_model,
                    "llama_server_running": True
                })
                backend_ready = True
                return True
            else:
                # Broadcast failure
                await broadcast_status("error", {
                    "message": f"Failed to load model {llama_model}",
                    "current_model": None,
                    "llama_server_running": False
                })
                stop_llama_server()
                backend_ready = False
                return False
    finally:
        # Ensure we decrement the refcount if we incremented it above.
        if incremented_here:
            try:
                _dec_model_switch_refcount()
            except Exception:
                pass


def _resolve_session_id_header(headers: Any) -> Tuple[Optional[str], Optional[str]]:
    """Resolve session id from supported client headers.

    Priority order:
    1. X-Session-Id (proxy-native)
    2. session_id (OpenAI cache header)
    3. X-Client-Request-Id (OpenAI-compatible)
    4. X-Session-Affinity (Anthropic-compatible)
    """
    if headers is None:
        return None, None
    candidates = [
        ("x-session-id", headers.get("x-session-id")),
        ("session_id", headers.get("session_id")),
        ("x-client-request-id", headers.get("x-client-request-id")),
        ("x-session-affinity", headers.get("x-session-affinity")),
    ]
    for name, value in candidates:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped, name
    return None, None


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
                        return _self_healing_response(path)
                    # Return a 503 response indicating backend error instead of raising
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
                                    return _self_healing_response(path)
                                # Return a 503 response indicating backend error instead of raising
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

                            max_completion_tokens = int(
                                server_config.get("session_guardrail_max_completion_tokens", 2048) or 2048
                            )
                            invalidate_on_guardrail = bool(
                                server_config.get("session_guardrail_invalidate_on_cutoff", True)
                            )
                            invalidate_on_repetition = server_config.get(
                                "session_guardrail_invalidate_on_repetition",
                                False,
                            )
                            if max_completion_tokens and recv_tokens >= max_completion_tokens:
                                _record_guardrail_cutoff("completion_tokens")
                                should_invalidate = _should_invalidate_on_guardrail(
                                    "completion_tokens",
                                    invalidate_on_guardrail,
                                    bool(invalidate_on_repetition),
                                )
                                if session_id and should_invalidate:
                                    await _invalidate_session_and_slot(
                                        session_id,
                                        "guardrail_completion_tokens",
                                        slot_filename,
                                    )
                                    slot_save_allowed = False

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


def _worker_process_unhealthy(proc: Optional[subprocess.Popen]) -> bool:
    """Detect unhealthy llama worker states (for example zombie children)."""
    if proc is None or psutil is None:
        return False

    pid = getattr(proc, "pid", None)
    if not pid:
        return False

    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
    except Exception:
        return False

    zombie_statuses = {"zombie", "dead"}
    try:
        zombie_statuses.add(str(psutil.STATUS_ZOMBIE).lower())
    except Exception:
        pass
    try:
        zombie_statuses.add(str(psutil.STATUS_DEAD).lower())
    except Exception:
        pass

    for child in children:
        try:
            status = str(child.status()).lower()
        except Exception:
            continue
        if status in zombie_statuses:
            logger.error(
                "watchdog detected unhealthy worker pid=%s status=%s parent_pid=%s",
                getattr(child, "pid", "unknown"),
                status,
                pid,
            )
            return True

    return False


def _prune_recovery_attempts(attempts: list[float], now_ts: float, window_seconds: int) -> list[float]:
    window = max(1, int(window_seconds))
    return [float(ts) for ts in attempts if now_ts - float(ts) <= window]


async def _attempt_router_self_heal() -> bool:
    """Attempt router-mode self-healing with capped exponential backoff."""
    global llama_process, current_model, backend_ready

    server_cfg = config.get("server", {}) if isinstance(config, dict) else {}
    max_attempts = max(1, int(server_cfg.get("llama_self_heal_max_attempts", 3) or 3))
    window_seconds = max(1, int(server_cfg.get("llama_self_heal_window_seconds", 300) or 300))
    base_backoff = max(0.0, float(server_cfg.get("llama_self_heal_backoff_base_seconds", 1.0) or 1.0))
    startup_timeout = int(server_cfg.get("llama_startup_timeout", 300) or 300)
    retry_after = _self_heal_retry_after_seconds()

    now_ts = time.time()
    attempts = backend_recovery_state.get("attempt_timestamps", [])
    if not isinstance(attempts, list):
        attempts = []
    attempts = _prune_recovery_attempts(attempts, now_ts, window_seconds)

    backend_recovery_state["attempt_timestamps"] = attempts
    backend_recovery_state["max_attempts"] = max_attempts
    backend_recovery_state["window_seconds"] = window_seconds
    backend_recovery_state["retry_after_seconds"] = retry_after

    if len(attempts) >= max_attempts:
        backend_recovery_state["in_progress"] = False
        backend_recovery_state["last_failure"] = (
            f"self-heal throttled: max {max_attempts} attempts in {window_seconds}s"
        )
        logger.error(
            "self-heal giving up: max attempts reached (%s attempts in %ss); manual intervention required",
            max_attempts,
            window_seconds,
        )
        return False

    backend_recovery_state["in_progress"] = True
    remaining = max_attempts - len(attempts)

    try:
        for local_attempt in range(remaining):
            attempt_started = time.time()
            attempts.append(attempt_started)
            backend_recovery_state["attempt_timestamps"] = attempts
            attempt_number = len(attempts)

            logger.warning(
                "self-heal attempt %s/%s started (window=%ss)",
                attempt_number,
                max_attempts,
                window_seconds,
            )

            try:
                restarted = start_llama_server(None)
                if restarted is None:
                    raise RuntimeError("start_llama_server returned None")

                llama_process = restarted
                backend_ready = await wait_for_llama_server(startup_timeout)
                if backend_ready:
                    backend_recovery_state["last_failure"] = None
                    logger.info("self-heal succeeded on attempt %s/%s", attempt_number, max_attempts)
                    return True

                raise RuntimeError("wait_for_llama_server returned False")
            except Exception as exc:
                backend_ready = False
                llama_process = None
                current_model = None
                backend_recovery_state["last_failure"] = str(exc)
                logger.error(
                    "self-heal attempt %s/%s failed: %s",
                    attempt_number,
                    max_attempts,
                    exc,
                )

            if local_attempt < remaining - 1:
                delay = base_backoff * (2 ** local_attempt)
                logger.warning("self-heal backoff sleeping %.1fs before retry", delay)
                await asyncio.sleep(delay)

        logger.error(
            "self-heal exhausted after %s attempt(s) within %ss; manual intervention required",
            remaining,
            window_seconds,
        )
        return False
    finally:
        backend_recovery_state["in_progress"] = False


async def _backend_watchdog_loop() -> None:
    """Watch local backend process and trigger best-effort recovery."""
    global llama_process, current_model, backend_ready

    while True:
        try:
            interval = float(config.get("server", {}).get("llama_watchdog_interval_seconds", 5.0) or 5.0)
            await asyncio.sleep(max(0.0, interval))

            proc = llama_process

            # LP-0MQ4GQ2LO005PZPY: If process is None (crashed or never started),
            # attempt restart in router mode instead of skipping.
            if proc is None:
                router_mode = bool(config.get("server", {}).get("llama_router_mode", False))
                if router_mode and not backend_ready:
                    logger.warning("watchdog: llama_process is None, attempting restart")
                    recovered = await _attempt_router_self_heal()
                    logger.info("watchdog restart-from-none recovered=%s", recovered)
                continue

            code = None
            try:
                code = proc.poll()
            except Exception:
                code = None

            worker_unhealthy = False
            if code is None:
                worker_unhealthy = _worker_process_unhealthy(proc)
                if not worker_unhealthy:
                    continue

            router_mode = bool(config.get("server", {}).get("llama_router_mode", False))
            if code is None and worker_unhealthy:
                logger.error("watchdog detected unhealthy worker while main process is alive model=%s", current_model)
                try:
                    if hasattr(proc, "terminate"):
                        proc.terminate()
                except Exception:
                    pass
            else:
                logger.error("watchdog detected llama-server exit code=%s model=%s", code, current_model)

            backend_ready = False
            _record_backend_signal("other_failures")
            llama_process = None
            current_model = None

            if router_mode:
                recovered = await _attempt_router_self_heal()
                logger.info("watchdog router self-heal recovered=%s", recovered)

        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("watchdog loop error")


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


@app.get("/llama/local/status")
async def get_llama_local_status():
    """Return a small JSON object describing local llama-server status.

    Fields:
      - active_query: bool
      - model_switch_in_progress: bool
      - current_model: str | None
      - llama_server_running: bool
    """
    # Query llama-server for runtime info (n_ctx etc. not included here)
    try:
        status = await query_llama_status()
    except Exception:
        # If querying fails, assume server not running
        status = {"llama_server_running": False, "n_ctx": None, "kv_cache_tokens": None, "router_mode": False}

    llama_running = bool(status.get("llama_server_running", False))

    # Determine if a model switch/load is in progress.
    # Consider multiple indicators:
    #  - explicit refcount for background/synchronous loads (`model_switch_refcount`)
    #  - the model_switch_lock (held during ensure_model_loaded)
    #  - any scheduled background loads in `background_loads`.
    try:
        switch_in_progress = (model_switch_refcount > 0) if 'model_switch_refcount' in globals() else False
    except Exception:
        switch_in_progress = False

    # Fall back to lock visibility
    if not switch_in_progress:
        try:
            switch_in_progress = model_switch_lock.locked() if model_switch_lock is not None else False
        except Exception:
            switch_in_progress = switch_in_progress

    # Also consider any scheduled background loads
    if not switch_in_progress:
        try:
            switch_in_progress = bool(background_loads)
        except Exception:
            switch_in_progress = switch_in_progress

    # current_model should be null when server not running
    cm = current_model if llama_running else None

    # active_query is true when we have at least one in-flight local request
    active = False
    try:
        async with active_queries_lock:
            active = (active_queries > 0)
    except Exception:
        active = False

    result = {
        "active_query": bool(active),
        "model_switch_in_progress": bool(switch_in_progress),
        "current_model": cm,
        "llama_server_running": bool(llama_running),
    }

    return result


async def _probe_backend_reachable(llama_port: int) -> bool:
    """Actively probe backend reachability so /health reflects real connectivity."""
    if llama_port <= 0:
        return False

    server_cfg = config.get("server", {}) if isinstance(config, dict) else {}
    router_mode = bool(server_cfg.get("llama_router_mode", False))
    timeout_seconds = float(server_cfg.get("llama_backend_probe_timeout_seconds", 2.0) or 2.0)

    probe_paths: list[str] = []
    if router_mode and current_model:
        probe_paths.append(f"/slots?model={current_model}")
    if router_mode:
        probe_paths.extend(["/models", "/health"])
    else:
        probe_paths.append("/health")

    client = _http_client if _http_client else httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))
    try:
        for probe_path in probe_paths:
            try:
                url = f"http://localhost:{llama_port}{probe_path}"
                response = await client.get(url, timeout=timeout_seconds)
                status_code = int(getattr(response, "status_code", 0) or 0)
                if 200 <= status_code < 500:
                    return True
            except Exception:
                continue
        return False
    finally:
        if not _http_client:
            try:
                await client.aclose()
            except Exception:
                pass


@app.get("/health")
async def health_check():
    """Health check endpoint with readiness gating."""
    server_cfg = config.get("server", {}) if isinstance(config, dict) else {}
    router_mode = bool(server_cfg.get("llama_router_mode", False))
    loaded_models = None
    if router_mode:
        router_models = await router_list_models()
        loaded_models = _extract_router_model_ids(router_models)

    llama_running = llama_process is not None and llama_process.poll() is None
    llama_port = int(server_cfg.get("llama_server_port", 8080) or 8080)
    backend_reachable = bool(llama_running and await _probe_backend_reachable(llama_port))
    self_healing = _is_self_healing_active()
    ready = bool(llama_running and backend_ready and backend_reachable and not self_healing)

    return {
        "status": "healthy" if ready else "degraded",
        "ready": ready,
        "current_model": current_model,
        "loaded_models": loaded_models,
        "llama_server_running": llama_running,
        "backend_reachable": backend_reachable,
        "self_healing_in_progress": self_healing,
        "backend_recovery": _backend_recovery_snapshot(),
        "backend_signals": dict(backend_signal_counts),
    }


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


@app.get("/v1/models")
async def list_models():
    """List available models from proxy configuration."""
    models_list = []
    
    for name, cfg in config.get("models", {}).items():
        models_list.append({
            "id": name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local" if cfg.get("type") == "local" else "remote",
            "type": cfg.get("type"),
            "aliases": cfg.get("aliases", [])
        })
    
    return {
        "object": "list",
        "data": models_list
    }


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


@app.post("/admin/reload-config")
async def reload_config():
    """Reload configuration file."""
    global config
    try:
        config = load_config()
        logger.info("Configuration reloaded")
        return {"status": "success", "message": "Configuration reloaded"}
    except Exception as e:
        logger.error(f"Failed to reload config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/admin/stop-server")
async def admin_stop_server():
    """Stop the llama-server."""
    stop_llama_server()
    return {"status": "success", "message": "llama-server stopped"}


@app.get("/admin/dump-counts")
async def admin_dump_counts():
    """Return in-memory request and token counts for debugging."""
    # Snapshot under locks to avoid races
    snap_c = {}
    snap_t = {}
    async with counts_lock:
        snap_c = dict(request_counts)
    async with token_lock:
        snap_t = dict(token_counts)
    return {"counts": snap_c, "tokens": snap_t}


@app.get("/admin/metrics")
async def admin_metrics():
    """Return router/memory/metrics for observability."""
    server_config = config.get("server", {})
    models_max = server_config.get("llama_models_max")
    router_mode = server_config.get("llama_router_mode", False)
    loaded_models = None
    if router_mode:
        router_models = await router_list_models()
        loaded_models = _extract_router_model_ids(router_models)

    per_model = {}
    for m in loaded_models or []:
        per_model[m] = {"last_used": model_last_used.get(m), "rss_bytes": None}

    process_rss = None
    try:
        if 'psutil' in globals() and psutil and llama_process is not None:
            pid = getattr(llama_process, 'pid', None)
            if pid:
                p = psutil.Process(pid)
                mem = p.memory_info()
                process_rss = getattr(mem, 'rss', None)
    except Exception:
        process_rss = None

    # Estimate per-model RSS bytes when possible (approximation for router-mode)
    if process_rss is not None and loaded_models:
        try:
            per = int(process_rss // len(loaded_models))
            for m in loaded_models:
                per_model[m]['rss_bytes'] = per
        except Exception:
            for m in loaded_models:
                per_model[m]['rss_bytes'] = None

    # Update Prometheus metrics (best-effort)
    try:
        metrics.update_metrics(process_rss, loaded_models)
    except Exception:
        pass

    return {
        "models_max": models_max,
        "loaded_models": loaded_models,
        "per_model": per_model,
        "process_rss_bytes": process_rss,
        "session_metrics": session_manager.get_metrics(),
        "restore_success_total": int(session_restore_observability.get("restore_success_total", 0)),
        "restore_fallback_total": dict(session_restore_observability.get("restore_fallback_total", {})),
        "delta_payload_bytes_total": int(session_restore_observability.get("delta_payload_bytes_total", 0)),
        "single_flight_metrics": dict(session_single_flight_observability),
        "guardrail_metrics": {
            "guardrail_cutoff_total": int(session_guardrail_observability.get("guardrail_cutoff_total", 0)),
            "guardrail_cutoff_reasons": dict(session_guardrail_observability.get("guardrail_cutoff_reasons", {})),
            "session_invalidation_total": int(session_guardrail_observability.get("session_invalidation_total", 0)),
            "session_invalidation_reasons": dict(session_guardrail_observability.get("session_invalidation_reasons", {})),
        },
        "backend_ready": bool(backend_ready),
        "backend_recovery": _backend_recovery_snapshot(),
        "backend_signals": dict(backend_signal_counts),
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus scrape endpoint (text/plain, exposition format)."""
    try:
        payload, content_type = metrics.generate_metrics_payload()
        return Response(content=payload, media_type=content_type)
    except Exception:
        raise HTTPException(status_code=503, detail="Prometheus metrics unavailable")


@app.post("/admin/reset-counts")
async def admin_reset_counts():
    """Reset in-memory and persisted request/token counts to empty.

    This clears the in-memory dictionaries and triggers an immediate persist.
    """
    global request_counts, token_counts, counts_dirty, tokens_dirty
    async with counts_lock:
        request_counts = {}
        counts_dirty = True
    async with token_lock:
        token_counts = {}
        tokens_dirty = True

    # Trigger async saves (background)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(save_counts())
        loop.create_task(save_token_counts())
    except RuntimeError:
        # fallback
        await save_counts()
        await save_token_counts()

    # Broadcast an empty snapshot to log tail clients
    for q in list(log_tail_clients):
        try:
            q.put_nowait({"counts": {}, "tokens": {}})
        except Exception:
            continue

    return {"status": "success", "message": "Counts reset"}


@app.get("/admin/sessions")
async def admin_list_sessions():
    """List all active sessions with their metadata."""
    sessions = []
    for session_id in list(session_manager._sessions.keys()):
        info = session_manager.get_session_info(session_id)
        if info is not None:
            sessions.append(info)
    return {"sessions": sessions, "total": len(sessions)}


@app.delete("/admin/sessions/{session_id}")
async def admin_delete_session(session_id: str):
    """Delete a specific session by ID."""
    removed = await session_manager.remove(session_id)
    if removed:
        return {"status": "success", "message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


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


if __name__ == "__main__":
    main()
