"""
Lifecycle Module

Model start/stop/load, self-healing, backend recovery, and watchdog
behaviour isolated from the monolithic server.py.

Uses a lazy server import (_srv()) to access module-level state without
circular import issues.
"""

import asyncio
import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Awaitable, Dict, List, Optional, Tuple

import httpx
import subprocess
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Lazy server import — avoids circular imports when server.py imports us
# ---------------------------------------------------------------------------
def _srv():
    import proxy.server as _m
    return _m


# ===================================================================
# Backend-recovery helpers
# ===================================================================

def _self_heal_retry_after_seconds() -> int:
    srv = _srv()
    server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}
    configured = server_cfg.get(
        "llama_self_heal_retry_after_seconds",
        srv.backend_recovery_state.get("retry_after_seconds", 30),
    )
    try:
        retry_after = int(configured or 30)
    except Exception:
        retry_after = 30
    retry_after = max(1, retry_after)
    srv.backend_recovery_state["retry_after_seconds"] = retry_after
    return retry_after


def _is_self_healing_active() -> bool:
    srv = _srv()
    try:
        return bool(srv.backend_recovery_state.get("in_progress"))
    except Exception:
        return False


def _self_healing_response(path: str) -> JSONResponse:
    srv = _srv()
    retry_after = srv._self_heal_retry_after_seconds()
    message = (
        "Backend error detected, team is working on recovery. "
        "Please retry after 30 seconds."
    )
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
    srv = _srv()
    state = dict(srv.backend_recovery_state)
    attempts = state.get("attempt_timestamps")
    state["attempt_count"] = len(attempts) if isinstance(attempts, list) else 0
    return state


# ===================================================================
# Worker-health helper
# ===================================================================

def _worker_process_unhealthy(proc: Optional[subprocess.Popen]) -> bool:
    """Detect unhealthy llama worker states (for example zombie children)."""
    srv = _srv()
    if proc is None or srv.psutil is None:
        return False

    pid = getattr(proc, "pid", None)
    if not pid:
        return False

    try:
        parent = srv.psutil.Process(pid)
        children = parent.children(recursive=True)
    except Exception:
        return False

    zombie_statuses = {"zombie", "dead"}
    try:
        zombie_statuses.add(str(srv.psutil.STATUS_ZOMBIE).lower())
    except Exception:
        pass
    try:
        zombie_statuses.add(str(srv.psutil.STATUS_DEAD).lower())
    except Exception:
        pass

    for child in children:
        try:
            status = str(child.status()).lower()
        except Exception:
            continue
        if status in zombie_statuses:
            srv.logger.error(
                "watchdog detected unhealthy worker pid=%s status=%s parent_pid=%s",
                getattr(child, "pid", "unknown"),
                status,
                pid,
            )
            return True

    return False


def _prune_recovery_attempts(
    attempts: list[float], now_ts: float, window_seconds: int
) -> list[float]:
    window = max(1, int(window_seconds))
    return [float(ts) for ts in attempts if now_ts - float(ts) <= window]


# ===================================================================
# Router self-healing
# ===================================================================

async def _attempt_router_self_heal() -> bool:
    """Attempt router-mode self-healing with capped exponential backoff."""
    srv = _srv()

    server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}
    max_attempts = max(
        1, int(server_cfg.get("llama_self_heal_max_attempts", 3) or 3)
    )
    window_seconds = max(
        1, int(server_cfg.get("llama_self_heal_window_seconds", 300) or 300)
    )
    base_backoff = max(
        0.0,
        float(server_cfg.get("llama_self_heal_backoff_base_seconds", 1.0) or 1.0),
    )
    startup_timeout = int(server_cfg.get("llama_startup_timeout", 300) or 300)
    retry_after = srv._self_heal_retry_after_seconds()

    now_ts = time.time()
    attempts = srv.backend_recovery_state.get("attempt_timestamps", [])
    if not isinstance(attempts, list):
        attempts = []
    attempts = srv._prune_recovery_attempts(attempts, now_ts, window_seconds)

    srv.backend_recovery_state["attempt_timestamps"] = attempts
    srv.backend_recovery_state["max_attempts"] = max_attempts
    srv.backend_recovery_state["window_seconds"] = window_seconds
    srv.backend_recovery_state["retry_after_seconds"] = retry_after

    if len(attempts) >= max_attempts:
        srv.backend_recovery_state["in_progress"] = False
        srv.backend_recovery_state["last_failure"] = (
            f"self-heal throttled: max {max_attempts} attempts in "
            f"{window_seconds}s"
        )
        srv.logger.error(
            "self-heal giving up: max attempts reached (%s attempts in %ss); "
            "manual intervention required",
            max_attempts,
            window_seconds,
        )
        return False

    srv.backend_recovery_state["in_progress"] = True
    remaining = max_attempts - len(attempts)

    try:
        for local_attempt in range(remaining):
            attempt_started = time.time()
            attempts.append(attempt_started)
            srv.backend_recovery_state["attempt_timestamps"] = attempts
            attempt_number = len(attempts)

            srv.logger.warning(
                "self-heal attempt %s/%s started (window=%ss)",
                attempt_number,
                max_attempts,
                window_seconds,
            )

            try:
                restarted = srv.start_llama_server(None)
                if restarted is None:
                    raise RuntimeError("start_llama_server returned None")

                srv.llama_process = restarted
                srv.backend_ready = await srv.wait_for_llama_server(startup_timeout)
                if srv.backend_ready:
                    srv.backend_recovery_state["last_failure"] = None
                    srv.logger.info(
                        "self-heal succeeded on attempt %s/%s",
                        attempt_number,
                        max_attempts,
                    )
                    return True

                raise RuntimeError("wait_for_llama_server returned False")
            except Exception as exc:
                srv.backend_ready = False
                srv.llama_process = None
                srv.current_model = None
                srv.backend_recovery_state["last_failure"] = str(exc)
                srv.logger.error(
                    "self-heal attempt %s/%s failed: %s",
                    attempt_number,
                    max_attempts,
                    exc,
                )

            if local_attempt < remaining - 1:
                delay = base_backoff * (2**local_attempt)
                srv.logger.warning(
                    "self-heal backoff sleeping %.1fs before retry", delay
                )
                await asyncio.sleep(delay)

        srv.logger.error(
            "self-heal exhausted after %s attempt(s) within %ss; "
            "manual intervention required",
            remaining,
            window_seconds,
        )
        return False
    finally:
        srv.backend_recovery_state["in_progress"] = False


# ===================================================================
# Backend watchdog
# ===================================================================

async def _backend_watchdog_loop() -> None:
    """Watch local backend process and trigger best-effort recovery."""
    srv = _srv()

    while True:
        try:
            interval = float(
                srv.config.get("server", {}).get(
                    "llama_watchdog_interval_seconds", 5.0
                )
                or 5.0
            )
            await asyncio.sleep(max(0.0, interval))

            proc = srv.llama_process

            # LP-0MQ4GQ2LO005PZPY: If process is None (crashed or never
            # started), attempt restart in router mode instead of skipping.
            if proc is None:
                router_mode = bool(
                    srv.config.get("server", {}).get("llama_router_mode", False)
                )
                if router_mode and not srv.backend_ready:
                    srv.logger.warning(
                        "watchdog: llama_process is None, attempting restart"
                    )
                    recovered = await srv._attempt_router_self_heal()
                    srv.logger.info(
                        "watchdog restart-from-none recovered=%s", recovered
                    )
                continue

            code = None
            try:
                code = proc.poll()
            except Exception:
                code = None

            worker_unhealthy = False
            if code is None:
                worker_unhealthy = srv._worker_process_unhealthy(proc)
                if not worker_unhealthy:
                    continue

            router_mode = bool(
                srv.config.get("server", {}).get("llama_router_mode", False)
            )
            if code is None and worker_unhealthy:
                srv.logger.error(
                    "watchdog detected unhealthy worker while main process "
                    "is alive model=%s",
                    srv.current_model,
                )
                try:
                    if hasattr(proc, "terminate"):
                        proc.terminate()
                except Exception:
                    pass
            else:
                srv.logger.error(
                    "watchdog detected llama-server exit code=%s model=%s",
                    code,
                    srv.current_model,
                )

            srv.backend_ready = False
            srv._record_backend_signal("other_failures")
            srv.llama_process = None
            srv.current_model = None

            if router_mode:
                recovered = await srv._attempt_router_self_heal()
                srv.logger.info(
                    "watchdog router self-heal recovered=%s", recovered
                )

        except asyncio.CancelledError:
            return
        except Exception:
            srv.logger.exception("watchdog loop error")

# ===================================================================
# Model loading and lifecycle orchestration helpers
# (extracted from monolithic server.py)
# ===================================================================

def _inc_model_switch_refcount() -> None:
    """Increment the global model_switch_refcount in a thread-safe way."""
    srv = _srv()
    try:
       with srv.model_switch_refcount_lock:
           srv.model_switch_refcount += 1
    except Exception:
       # Best-effort: swallow errors so status reporting never raises
       pass



def _dec_model_switch_refcount() -> None:
    """Decrement the global model_switch_refcount (never below zero)."""
    srv = _srv()
    try:
       with srv.model_switch_refcount_lock:
           srv.model_switch_refcount = max(0, srv.model_switch_refcount - 1)
    except Exception:
        pass


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

    
# Functions extracted to handlers.py:
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




def _model_loading_response(requested_model: Optional[str], target_model: str, scheduled: bool, endpoint: str) -> JSONResponse:
    """Build a consistent JSON 503 payload when a model is loading."""
    srv = _srv()
    retry_after = int(srv.config.get("server", {}).get("model_loading_retry_after", 30) or 30)
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
       "srv.current_model": srv.current_model,
       "llama_server_running": srv.llama_process is not None and llama_process.poll() is None,
       "retry_after": retry_after,
       "endpoint": endpoint,
    }
    return JSONResponse(
       status_code=503,
       content=payload,
       headers={"Retry-After": str(retry_after), "Cache-Control": "no-store"},
    )






def _is_retryable_backend_exception(exc: Exception) -> bool:
    """Return True when an exception is a retryable backend transport failure."""
    srv = _srv()
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError, httpx.ReadTimeout, httpx.TimeoutException))



def _compute_retry_delay(attempt: int, base_delay: float, max_delay: float, jitter_ratio: float) -> float:
    """Compute bounded exponential backoff with jitter."""
    srv = _srv()
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
    srv = _srv()
    server_cfg = srv.config.get("server", {})
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
           if not srv._is_retryable_backend_exception(exc) or attempt >= max_attempts:
               srv._record_backend_signal(srv._classify_backend_exception(exc))
               try:
                   srv.backend_ready = False
               except Exception:
                   pass
               raise

           signal_name = srv._classify_backend_exception(exc)
           srv._record_backend_signal(signal_name)
           delay = srv._compute_retry_delay(attempt, base_delay, max_delay, jitter_ratio)
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






async def _probe_backend_reachable(llama_port: int) -> bool:
    """Actively probe backend reachability so /health reflects real connectivity."""
    srv = _srv()
    if llama_port <= 0:
       return False
    server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}
    router_mode = bool(server_cfg.get("llama_router_mode", False))
    timeout_seconds = float(server_cfg.get("llama_backend_probe_timeout_seconds", 2.0) or 2.0)
    probe_paths: list[str] = []
    if router_mode and srv.current_model:
       probe_paths.append(f"/slots?model={srv.current_model}")
    if router_mode:
       probe_paths.extend(["/models", "/health"])
    else:
       probe_paths.append("/health")
    client = srv._http_client if srv._http_client else httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))
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
       if not srv._http_client:
           try:
               await client.aclose()
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
    
    models = srv.config.get("models", {})
    
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
    srv = _srv()
    if not isinstance(model_cfg, dict):
       return False
    return bool(model_cfg.get("force_full_prompt") or model_cfg.get("disable_delta"))



def get_local_model_name(model_name: Optional[str]) -> Optional[str]:
    """Get the llama model name for a given model."""
    srv = _srv()
    model_cfg = srv.get_model_config(model_name)
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
    srv = _srv()
    candidate = requested_model or srv.current_model
    if not candidate:
       return None
    if server_config.get("llama_router_mode", False):
       try:
           return srv.get_local_model_name(candidate) or candidate
       except HTTPException:
           return candidate
    return candidate



async def wait_for_llama_server(timeout: int = 300) -> bool:
    """Wait for llama-server to be ready."""
    srv = _srv()
    server_config = srv.config.get("server", {})
    llama_port = server_config.get("llama_server_port", 8080)
    health_url = f"http://localhost:{llama_port}/health"
    
    start_time = time.time()
    client = srv._http_client if srv._http_client else httpx.AsyncClient(timeout=httpx.Timeout(5.0))
    try:
       while time.time() - start_time < timeout:
           # Check if llama process died
           if srv.llama_process is not None and llama_process.poll() is not None:
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
       if not srv._http_client:
           try:
               await client.aclose()
           except Exception:
               pass
    
    logger.error(f"llama-server failed to start within {timeout} seconds")
    return False



async def router_load_model(model_name: str) -> bool:
    """Request router-mode llama-server to load a model."""
    srv = _srv()
    server_config = srv.config.get("server", {})
    llama_port = server_config.get("llama_server_port", 8080)
    url = f"http://localhost:{llama_port}/models/load"
    payload = {"model": model_name}

    client = srv._http_client if srv._http_client else httpx.AsyncClient(timeout=httpx.Timeout(30.0))
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
                       srv.model_last_used[model_name] = datetime.utcnow().isoformat()
                   except Exception:
                       pass
                   return True
               logger.error(f"Router load failed for {model_name}: {response.status_code} {body}")
               return False
           # Update last-used timestamp on successful load
           try:
               srv.model_last_used[model_name] = datetime.utcnow().isoformat()
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
       if not srv._http_client:
           try:
               await client.aclose()
           except Exception:
               pass



async def router_list_models() -> Optional[dict]:
    """List models from router-mode llama-server."""
    srv = _srv()
    server_config = srv.config.get("server", {})
    llama_port = server_config.get("llama_server_port", 8080)
    url = f"http://localhost:{llama_port}/models"

    client = srv._http_client if srv._http_client else httpx.AsyncClient(timeout=5.0)
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
       if not srv._http_client:
           try:
               await client.aclose()
           except Exception:
               pass



def _extract_router_model_ids(router_models: Optional[dict]) -> list[str]:
    srv = _srv()
    if not isinstance(router_models, dict):
       return []
    models_payload = router_models.get("data") or router_models.get("models") or []
    if isinstance(models_payload, list):
       return [str(m.get("id")) for m in models_payload if isinstance(m, dict) and m.get("id")]
    return []



async def router_is_model_loaded(model_name: str) -> bool:
    srv = _srv()
    router_models = await srv.router_list_models()
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
       # srv.background_loads and keep returning scheduled=False + 503.
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
    srv = _srv()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
       if await srv.router_is_model_loaded(model_name):
           return True
       await asyncio.sleep(interval)
    return False



async def router_preload_models(model_names: list[str]) -> bool:
    """Preload a list of models in router mode."""
    srv = _srv()
    for model_name in model_names:
       if not await srv.router_load_model(model_name):
           return False
    return True



def start_llama_server(model: Optional[str]) -> Optional[subprocess.Popen]:
    """Start the llama-server with the specified model inside distrobox."""
    srv = _srv()
    
    server_config = srv.config.get("server", {})
    # Default to the repository root `start-llama.sh` if not specified in srv.config
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
    if srv.log_dir:
       llama_log_path = srv.log_dir / "llama-server.log"
       srv.rotate_llama_logs(llama_log_path, keep=15)
       srv.llama_log_file = open(llama_log_path, "w")
    else:
       srv.llama_log_file = subprocess.DEVNULL

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
           srv.last_start_failure = msg
           srv.broadcast_status_sync("error", {"message": msg, "srv.current_model": None, "llama_server_running": False})
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
               if srv.llama_log_file and proc.stdout:
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
                                           parsed = srv.extract_progress_data(line_str)
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

                   t = threading.Thread(target=_stream_output, args=(proc.stdout, srv.llama_log_file), daemon=True)
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
       srv.last_start_failure = msg
       srv.broadcast_status_sync("error", {"message": msg, "srv.current_model": None, "llama_server_running": False})
       return None

    # Try distrobox only
    for attempt in range(retries):
       proc, out = _spawn_and_capture(distrobox_cmd)
       tried_cmds.append((distrobox_cmd, out))
       if proc is not None:
           # Do not set `srv.current_model` here. The proxy must only mark a
           # model as active after the llama-server is actually ready and
           # the model has finished loading. `ensure_model_loaded` is
           # responsible for setting `srv.current_model` once startup and
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
    srv.last_start_failure = msg
    srv.broadcast_status_sync("error", {"message": msg, "srv.current_model": None, "llama_server_running": False})
    return None



def rotate_llama_logs(current_log: Path, keep: int = 15):
    """Rotate llama-server logs, keeping the last N copies."""
    srv = _srv()
    if not current_log.exists():
       return
    
    # Find existing rotated logs
    srv.log_dir = current_log.parent
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
           new_name = srv.log_dir / f"{base_name}.{num + 1}{suffix}"
           f.rename(new_name)
    
    # Rotate current log to .1
    if current_log.exists():
       current_log.rename(srv.log_dir / f"{base_name}.1{suffix}")



def stop_llama_server():
    """Stop the currently running llama-server."""
    srv = _srv()
    
    server_config = srv.config.get("server", {})
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
    
    if srv.llama_process is not None:
       pid = getattr(srv.llama_process, 'pid', 'N/A')
       logger.info(f"Stopping llama-server wrapper (PID: {pid})")
       # Only clean up process and model state if srv.llama_process looks like
       # a real subprocess (has terminate/kill/wait methods). If it's a
       # test mock or invalid object, skip process cleanup.
       is_real_process = hasattr(srv.llama_process, 'terminate') or hasattr(srv.llama_process, 'kill')
       if is_real_process:
           previous_model = srv.current_model
           llama_process.terminate()
           try:
               llama_process.wait(timeout=30)
           except subprocess.TimeoutExpired:
               logger.warning("llama-server wrapper did not terminate gracefully, killing...")
               if hasattr(srv.llama_process, 'kill'):
                   llama_process.kill()
               if hasattr(srv.llama_process, 'wait'):
                   llama_process.wait()
           srv.llama_process = None
           try:
               if previous_model:
                   metrics.record_model_unloaded(previous_model)
           except Exception:
               pass
           srv.current_model = None
           srv.backend_ready = False
           logger.info("llama-server stopped")
       else:
           srv.llama_process = None
           srv.backend_ready = False
           logger.info("llama-server stop skipped (no valid process)")
    
    # Close log file if open
    if srv.llama_log_file is not None and srv.llama_log_file != subprocess.DEVNULL:
       try:
           llama_log_file.close()
       except Exception:
           pass
       srv.llama_log_file = None



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

    server_config = srv.config.get("server", {})
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
           await srv.broadcast_status("switching", {
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
                       await srv.broadcast_status("error", {
                           "message": "Failed to start router-mode llama-server",
                           "current_model": None,
                           "llama_server_running": False
                       })
                       stop_llama_server()
                       backend_ready = False
                       return False

               if not await router_load_model(llama_model):
                   await srv.broadcast_status("error", {
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
                   await srv.broadcast_status("error", {
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
               await srv.broadcast_status("ready", {
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
               await srv.broadcast_status("ready", {
                   "current_model": llama_model,
                   "llama_server_running": True
               })
               backend_ready = True
               return True
           else:
               # Broadcast failure
               await srv.broadcast_status("error", {
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



