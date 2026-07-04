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

# Version information captured at startup (llama-server and ROCm)
llama_server_version: str = "unknown"
rocm_version: str = "unknown"

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

# Active local queries counter (global, all providers)
active_queries: int = 0
active_queries_lock = asyncio.Lock()

# Local-only active queries counter (LP-0MR5MAJNM005R905)
local_active_queries: int = 0
local_active_queries_lock = asyncio.Lock()

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

# Background model health monitoring task (started in lifespan)
model_health_task: Optional[asyncio.Task] = None

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


async def _capture_llama_server_version() -> str:
    """Capture the llama-server version by running `llama-server --version`.

    Runs once at startup. Returns the version string or "unknown" on failure.
    """
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["llama-server", "--version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            logger.warning("llama-server --version returned %d", result.returncode)
            return "unknown"
        stdout = result.stdout.strip()
        if not stdout:
            return "unknown"
        # Return the first non-empty line as the version string
        # Typical outputs: "build: 4321 (abc12345)" or "version: X.Y.Z (deadbeef)"
        lines = stdout.split("\n")
        for line in lines:
            line = line.strip()
            if line:
                return line
        return "unknown"
    except FileNotFoundError:
        logger.warning("llama-server not found in PATH")
        return "unknown"
    except subprocess.TimeoutExpired:
        logger.warning("llama-server --version timed out")
        return "unknown"
    except Exception as e:
        logger.warning("Failed to capture llama-server version: %s", e)
        return "unknown"


async def _capture_rocm_version() -> str:
    """Capture the ROCm version by running `rocm-smi --showtag`.

    Falls back to `rocm-smi --version` if --showtag is unavailable.
    Returns the version string or "unknown" on failure.
    """
    # Try primary command: rocm-smi --showtag
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["rocm-smi", "--showtag"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            stdout = result.stdout.strip()
            if stdout:
                return stdout
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        logger.warning("rocm-smi --showtag failed: %s", e)

    # Fallback: rocm-smi --version
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["rocm-smi", "--version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            stdout = result.stdout.strip()
            if stdout:
                return stdout
    except FileNotFoundError:
        logger.warning("rocm-smi not found in PATH")
    except subprocess.TimeoutExpired:
        logger.warning("rocm-smi --version timed out")
    except Exception as e:
        logger.warning("rocm-smi --version failed: %s", e)

    return "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global config, logger, llama_process, _http_client, backend_watchdog_task, model_health_task, backend_ready, backend_recovery_state
    
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

    # Capture llama-server and ROCm versions asynchronously at startup.
    # These are best-effort, non-blocking captures. If either command fails
    # or is unavailable, the version remains "unknown" and the server continues.
    loop = asyncio.get_running_loop()

    async def _capture_versions():
        global llama_server_version, rocm_version
        llama_server_version = await _capture_llama_server_version()
        rocm_version = await _capture_rocm_version()
        logger.info(
            "Version info: llama-server=%s, ROCm=%s",
            llama_server_version, rocm_version
        )

    loop.create_task(_capture_versions())

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
    # boot-order race where podman isn't ready yet.
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

    if model_health_task is None:
        model_health_task = loop.create_task(_router_model_health_loop())

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


@app.get("/debug/prompt")
async def debug_prompt(request: Request, alias: str = "", full: bool = False):
    """Debug endpoint to inspect resolved prompt for a given alias.

    Gated by server.debug config flag or localhost access.
    Returns {
        alias, resolved: bool, mode, source_path, content_preview, size_bytes
    }
    """
    from proxy.prompt_resolver import resolve_system_prompt

    debug_mode = config.get("server", {}).get("debug", False)
    client_host = request.client.host if request.client else ""
    is_local = client_host in ("127.0.0.1", "::1", "localhost")

    if not debug_mode and not is_local:
        raise HTTPException(status_code=403, detail="Debug endpoint is not enabled")

    if not alias:
        raise HTTPException(status_code=400, detail="Query parameter 'alias' is required")

    model_cfg = get_model_config(alias)
    if model_cfg is None:
        return JSONResponse({
            "alias": alias,
            "resolved": False,
            "mode": None,
            "source_path": None,
            "content_preview": None,
            "size_bytes": None,
            "reason": "Model config not found",
        })

    prompt_result = resolve_system_prompt(alias, model_cfg)

    if prompt_result is None:
        return JSONResponse({
            "alias": alias,
            "resolved": False,
            "mode": model_cfg.get("system_prompt", {}).get("mode") if isinstance(model_cfg.get("system_prompt"), dict) else None,
            "source_path": None,
            "content_preview": None,
            "size_bytes": None,
            "reason": "No prompt file found or configured",
        })

    content = prompt_result["content"]
    content_bytes = content.encode("utf-8")
    size_bytes = len(content_bytes)

    if full and debug_mode:
        content_preview = content
    else:
        content_preview = content[:200] + ("..." if len(content) > 200 else "")

    return JSONResponse({
        "alias": alias,
        "resolved": True,
        "mode": prompt_result["mode"],
        "source_path": prompt_result["source"],
        "content_preview": content_preview,
        "size_bytes": size_bytes,
    })


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
    _router_model_health_loop,
    _extract_model_port_from_args,
    _probe_model_instance,
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
from .router import (  # noqa: E402, F401
    proxy_to_local,
    proxy_to_remote,
    log_request,
    log_response,
    log_response_chunk,
)
from .provider import (  # noqa: E402, F401
    resolve_provider,
    mark_provider_unavailable,
    proxy_with_remote_fallback,
    proxy_with_fallback,
    _provider_unavailable_until,
    _is_provider_unavailable,
    _is_connection_error,
    _is_http_error_status,
    _is_slot_exhaustion_response,
    _parse_retry_after,
    _compute_cooldown,
)


if __name__ == "__main__":
    main()
