"""
Lifecycle Module

Model start/stop/load, model loading, router-model loading, refcounting,
and background load orchestration isolated from the monolithic server.py.

Uses a lazy server import (_srv()) to access module-level state without
circular import issues.
"""

import asyncio
import logging
import os
import subprocess
import threading
import time
import traceback
from datetime import datetime, timedelta
from fnmatch import fnmatch
from pathlib import Path

import httpx
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Lazy server import — avoids circular imports when server.py imports us
# ---------------------------------------------------------------------------
def _srv():
    import proxy.server as _m
    return _m


# ===================================================================
# Backward-compatibility re-exports
#   The functions below were extracted to backend_health.py.  These
#   re-exports keep existing import chains (server.py, router.py) and
#   test monkey-patches on server.* working without modification.
# ===================================================================
from .backend_health import (  # noqa: F401
    _attempt_router_self_heal,
    _attempt_tts_self_heal,
    _backend_recovery_snapshot,
    _backend_watchdog_loop,
    _get_tts_self_heal_max_attempts,
    _get_tts_watchdog_interval,
    _is_self_healing_active,
    _prune_recovery_attempts,
    _self_heal_retry_after_seconds,
    _self_healing_response,
    _tts_recovery_snapshot,
    _tts_watchdog_loop,
    _worker_process_unhealthy,
)

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
    srv = _srv()
    """Schedule model load in background if not already running.

    Returns True if a background load was started, False if one is already in progress.
    """
    if not model_name:
       return False
    if srv.background_loads.get(model_name):
       return False

    srv.background_loads[model_name] = True

    # Increment the global refcount so status checks observe switching even
    # when the background worker runs in another thread/event loop.
    try:
       srv._inc_model_switch_refcount()
    except Exception:
       pass

    async def _bg():

       try:
           srv.logger.info(f"Background model load started: {model_name}")
           ok = await srv.ensure_model_loaded(model_name)
           if ok:
               srv.logger.info(f"Background model load succeeded: {model_name}")
           else:
               srv.logger.error(f"Background model load failed: {model_name}")
       except Exception:
           srv.logger.exception(f"Exception during background model load for {model_name}")
       finally:
           # Decrement the refcount when done so status reflects completion.
           try:
               srv._dec_model_switch_refcount()
           except Exception:
               pass
           srv.background_loads.pop(model_name, None)

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
counts_persist_task: asyncio.Task | None = None
periodic_broadcast_task: asyncio.Task | None = None

# Active local queries counter
active_queries: int = 0
active_queries_lock = asyncio.Lock()

# Backend resilience/observability signals
# Session restore observability




def _model_loading_response(requested_model: str | None, target_model: str, scheduled: bool, endpoint: str) -> JSONResponse:
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
       "current_model": srv.current_model,
       "llama_server_running": srv.llama_process is not None and srv.llama_process.poll() is None,
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
    srv = _srv()
    server_cfg = srv.config.get("server", {})
    max_attempts = int(server_cfg.get("backend_retry_attempts", 3) or 3)
    base_delay = float(server_cfg.get("backend_retry_base_delay_seconds", 0.25) or 0.25)
    max_delay = float(server_cfg.get("backend_retry_max_delay_seconds", 2.0) or 2.0)
    jitter_ratio = float(server_cfg.get("backend_retry_jitter_ratio", 0.25) or 0.25)

    last_exc: Exception | None = None
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
           srv.logger.warning(
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



def get_model_config(model_name: str | None) -> dict | None:
    srv = _srv()
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



def _should_force_full_prompt(model_cfg: dict | None) -> bool:
    if not isinstance(model_cfg, dict):
       return False
    return bool(model_cfg.get("force_full_prompt") or model_cfg.get("disable_delta"))



def get_local_model_name(model_name: str | None) -> str | None:
    """Get the llama model name for a given model.

    Supports both the legacy top-level ``type: local`` + ``llama_model`` and the
    new ``providers`` list schema where a local provider contains ``type: local``
    and ``llama_model``.
    """
    srv = _srv()
    model_cfg = srv.get_model_config(model_name)
    if not model_cfg:
       return None

    # First, prefer an explicit top-level llama_model (backwards compat)
    llama_model = model_cfg.get("llama_model")

    # If not present, inspect providers list for a local provider
    if not llama_model:
       providers = model_cfg.get("providers") or []
       if isinstance(providers, list):
           for p in providers:
               if isinstance(p, dict) and p.get("type") == "local" and p.get("llama_model"):
                   llama_model = p.get("llama_model")
                   break

    # If still not found, fall back to legacy type field where appropriate
    if not llama_model and model_cfg.get("type") == "local":
       llama_model = model_cfg.get("llama_model")

    if not isinstance(llama_model, str) or not llama_model:
       # Not a local model or missing config
       return None

    return llama_model



def _resolve_slot_model_name(
    requested_model: str | None,
    current_model: str | None,
    server_config: dict,
) -> str | None:
    """Resolve the llama model name used for slot endpoints in router mode."""
    srv = _srv()
    candidate = requested_model or current_model
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
           if srv.llama_process is not None and srv.llama_process.poll() is not None:
               exit_code = srv.llama_process.returncode
               srv.logger.error(f"llama-server process exited with code {exit_code}")
               return False

           try:
               response = await client.get(health_url, timeout=5)
               if response.status_code == 200:
                   srv.logger.info("llama-server is ready")
                   return True
           except asyncio.CancelledError:
               srv.logger.info("Wait for llama-server cancelled")
               raise
           except Exception:
               pass
           try:
               await asyncio.sleep(2)
           except asyncio.CancelledError:
               srv.logger.info("Wait for llama-server cancelled")
               raise
    finally:
       if not srv._http_client:
           try:
               await client.aclose()
           except Exception:
               pass

    srv.logger.error(f"llama-server failed to start within {timeout} seconds")
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
                   srv.logger.info(f"Router model already loaded: {model_name}")
                   # update last-used timestamp when model already loaded/running
                   try:
                       srv.model_last_used[model_name] = datetime.utcnow().isoformat()
                   except Exception:
                       pass
                   return True
               srv.logger.error(f"Router load failed for {model_name}: {response.status_code} {body}")
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
           srv.logger.error(f"Router load failed for {model_name}: {e}")
           return False
    finally:
       if not srv._http_client:
           try:
               await client.aclose()
           except Exception:
               pass



async def router_list_models() -> dict | None:
    """List models from router-mode llama-server."""
    srv = _srv()
    server_config = srv.config.get("server", {})
    llama_port = server_config.get("llama_server_port", 8080)
    url = f"http://localhost:{llama_port}/models"

    client = srv._http_client if srv._http_client else httpx.AsyncClient(timeout=5.0)
    try:
       response = await client.get(url, timeout=5.0)
       if response.status_code != 200:
           srv.logger.warning(f"Router list models failed: {response.status_code} {response.text}")
           return None
       return response.json()
    except Exception as e:
       srv.logger.warning(f"Router list models failed: {e}")
       return None
    finally:
       if not srv._http_client:
           try:
               await client.aclose()
           except Exception:
               pass



def _extract_router_model_ids(router_models: dict | None) -> list[str]:
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


# ---------------------------------------------------------------------------
# Log watcher — monitors llama-server logs for unload_lru eviction events
# ---------------------------------------------------------------------------


def _parse_unload_lru(line):
    """Check if a log line contains an unload_lru event.

    Returns True if the line contains 'unload_lru', None otherwise.
    Handles both str and bytes inputs gracefully.
    """
    if not isinstance(line, str):
        try:
            line = str(line)
        except Exception:
            return None
    if not line:
        return None
    if "unload_lru" in line.lower():
        return True
    return None


class _UnloadLruTracker:
    """Tracks unload_lru events within a configurable rolling time window.

    Attributes:
        window_minutes: Rolling window duration (minutes).
        threshold: Number of events that triggers an alert.
        alerted: Whether the threshold has been breached since last reset.
    """

    def __init__(self, window_minutes: int = 5, threshold: int = 3):
        self.window_minutes = window_minutes
        self.threshold = threshold
        self._events: list[datetime] = []
        self.alerted = False

    def record(self):
        """Record an unload_lru event at the current time."""
        self._events.append(datetime.now())
        self.prune()

    def prune(self):
        """Remove events outside the rolling window."""
        cutoff = datetime.now() - timedelta(minutes=self.window_minutes)
        self._events = [e for e in self._events if e >= cutoff]

    def count(self) -> int:
        """Return the number of events in the current window."""
        self.prune()
        return len(self._events)


def _check_unload_lru_threshold(tracker: _UnloadLruTracker) -> bool:
    """Check if the tracker's event count has reached the threshold.

    Returns True if threshold is met or exceeded, False otherwise.
    Sets tracker.alerted = True when triggered.
    """
    if tracker.count() >= tracker.threshold and not tracker.alerted:
        tracker.alerted = True
        return True
    return False


# ---------------------------------------------------------------------------
# Spawn helper — extracted from start_llama_server() for testability
# ---------------------------------------------------------------------------

def spawn_and_capture(
    cmd: list[str],
    env: dict,
    log_file,
    logger: logging.Logger,
    model_name: str = "unknown",
) -> tuple:
    """Start a subprocess and capture immediate stderr/stdout if it exits quickly.

    Args:
        cmd: Command list to execute.
        env: Environment variables.
        log_file: File-like object for streaming output.
        logger: Logger instance.
        model_name: Short model name for progress display (e.g. "Qwen3", "gemma4").

    Returns a tuple (Popen|None, captured_output_str|None).
    """
    try:
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

    try:
        outs, _ = proc.communicate(timeout=3)
        return None, outs
    except subprocess.TimeoutExpired:
        try:
            if log_file and proc.stdout:
                t = threading.Thread(target=_stream_output, args=(proc.stdout, log_file, model_name, logger), daemon=True)
                t.start()
        except Exception:
            pass
        return proc, None


def _stream_output(src, dst, model_name: str = "unknown", logger: logging.Logger | None = None):
    """Stream lines from src to dst, logging prompt progress as timestamped INFO entries.

    Instead of writing progress updates to stderr with carriage-return overwrites,
    this function logs clean progress entries via the Python logging system at
    each 10% progress milestone (10%, 20%, …, 100%) per slot.

    Per-slot progress thresholds are tracked in-memory so each slot independently
    triggers log entries only when crossing a new 10% boundary.  The state is
    scoped to the lifetime of this thread (one per llama-server process).

    Args:
        src: Source stream (e.g. subprocess stdout).
        dst: Destination stream (e.g. log file).
        model_name: Short model name for progress display (e.g. "Qwen3", "gemma4").
            When ``"unknown"`` (router mode with no initial model), the function
            will dynamically look up ``srv.current_model`` from server state.
        logger: Logger instance for progress entries.  If ``None``, progress
            entries are silently skipped (no-op).
    """
    # Per-slot progress threshold tracking.
    # Maps slot_id -> last logged 10%-bucket index (0-10, where 0 = 0%, 10 = 100%).
    # A slot not in the dict has never been logged.
    _last_logged_pct: dict[int, int] = {}
    first_progress_time = None
    try:
        for line in src:
            dst.write(line)
            dst.flush()
            # Parse and log prompt processing progress
            try:
                line_str = line.decode('utf-8', errors='replace') if isinstance(line, bytes) else str(line)
                if 'prompt processing' in line_str.lower():
                    parsed = extract_progress_data(line_str)
                    if parsed:
                        slot_id, n_tokens, progress = parsed
                        total_tokens = int(n_tokens / progress) if progress > 0 else n_tokens

                        # Track elapsed time for average tokens/sec calculation
                        now = time.monotonic()
                        if first_progress_time is None:
                            first_progress_time = now
                        elapsed = now - first_progress_time
                        if elapsed < 0.5:
                            tokens_per_sec = None
                        else:
                            tokens_per_sec = n_tokens / elapsed

                        # Dynamically resolve model name for router mode
                        resolved_name = model_name
                        if resolved_name == "unknown":
                            try:
                                srv = _srv()
                                if getattr(srv, 'current_model', None):
                                    resolved_name = srv.current_model
                            except Exception:
                                pass

                        progress_str = format_progress(
                            n_tokens, total_tokens, progress,
                            model_name=resolved_name,
                            slot_id=slot_id,
                            tokens_per_sec=tokens_per_sec,
                        )

                        # Threshold-based logging: log at start, at each 10%
                        # milestone, and at completion (progress >= 1.0).
                        current_threshold = int(min(progress, 1.0) * 10)
                        last = _last_logged_pct.get(slot_id)

                        if last is None:
                            # First progress data for this slot — always log.
                            _last_logged_pct[slot_id] = current_threshold
                            if logger:
                                logger.info(progress_str)
                        elif current_threshold > last:
                            # Crossed a new 10% boundary.
                            _last_logged_pct[slot_id] = current_threshold
                            if logger:
                                logger.info(progress_str)
                        elif progress >= 1.0 and last < 10:
                            # Final milestone (progress >= 100%) — ensure logged.
                            _last_logged_pct[slot_id] = 10
                            if logger:
                                logger.info(progress_str)
            except Exception:
                if logger:
                    logger.exception("Error processing progress line in _stream_output")
                else:
                    import traceback
                    traceback.print_exc()
    except Exception:
        if logger:
            logger.exception("Fatal error in _stream_output thread")
        else:
            import traceback
            traceback.print_exc()


def start_llama_server(model: str | None, display_name: str | None = None) -> subprocess.Popen | None:
    """Start the llama-server with the specified model.

    Args:
        model: Model name for the start script command (llama model name or None for router mode).
        display_name: Short model name for progress display (e.g. "Qwen3", "gemma4").
            When not provided, falls back to ``model``, then ``srv.current_model``,
            then ``"unknown"``.

    Returns a subprocess.Popen object when a long-running process is started,
    or None when startup failed after retries.
    """
    srv = _srv()
    # Resolve the display name for progress output
    _display = display_name or model or getattr(srv, 'current_model', None) or "unknown"

    server_config = srv.config.get("server", {})
    # Default to the repository root `start-llama.sh` if not specified in srv.config
    script_path = server_config.get(
       "llama_start_script",
       str(Path(__file__).parent.parent.parent / "start-llama.sh")
    )
    llama_port = server_config.get("llama_server_port", 8080)

    # Set environment variables
    env = os.environ.copy()
    env["PORT"] = str(llama_port)

    # Allow overriding which llama-server binary the start script should invoke
    llama_server_bin = server_config.get("llama_server_bin")
    if llama_server_bin:
       env["LLAMA_SERVER_BIN"] = str(llama_server_bin)
    slot_save_path = server_config.get("session_slot_save_path")
    if slot_save_path:
       env["LLAMA_SLOT_SAVE_PATH"] = str(slot_save_path)

    # Export session_slot_pool_size as LLAMA_PARALLEL so the start script can
    # align --parallel / -np with the proxy's pool size. Falls back to 1 when
    # not configured — matches the historical default in start-llama.sh.
    slot_pool_size = server_config.get("session_slot_pool_size", 1)
    env["LLAMA_PARALLEL"] = str(int(slot_pool_size or 1))

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
    srv.logger.info(f"Starting llama-server with {mode_label}")

    # Rotate llama-server logs (keep last 15)
    if srv.log_dir:
       llama_log_path = srv.log_dir / "llama-server.log"
       srv.rotate_llama_logs(llama_log_path, keep=15)
       srv.llama_log_file = open(llama_log_path, "w")
    else:
       srv.llama_log_file = subprocess.DEVNULL

    # Build the command for the configured start script
    if router_mode:
       llama_models_max = server_config.get("llama_models_max")
       if llama_models_max:
           env["LLAMA_MODELS_MAX"] = str(llama_models_max)
       llama_models_preset = server_config.get("llama_models_preset")
       if llama_models_preset:
           env["LLAMA_MODELS_PRESET"] = str(llama_models_preset)

       cmd = [script_path, "router"]
    else:
       if model is None:
           msg = "Model name is required when not running in router mode"
           srv.logger.error(msg)
           srv.last_start_failure = msg
           srv.broadcast_status_sync("error", {"message": msg, "current_model": None, "llama_server_running": False})
           return None
       cmd = [script_path, model]

    # Host-first startup: if llama_allow_host_fallback is enabled AND the
    # configured start script is different from the default start-llama.sh,
    # attempt one host-start before falling back to the configured script.
    host_start_script = str(Path(__file__).parent.parent.parent / "start-llama.sh")
    allow_host_fallback = bool(server_config.get("llama_allow_host_fallback", False))

    if allow_host_fallback and script_path != host_start_script:
        host_cmd = [host_start_script]
        if router_mode:
            host_cmd.append("router")
        elif model is not None:
            host_cmd.append(model)
        else:
            host_cmd.append("router")

        srv.logger.info(f"Attempting host-first startup with: {' '.join(host_cmd)}")
        host_proc, host_out = spawn_and_capture(host_cmd, env, srv.llama_log_file, srv.logger, model_name=_display)

        if host_proc is not None:
            srv.logger.info(f"Host-first startup succeeded with command: {' '.join(host_cmd)}")
            return host_proc

        srv.logger.warning(
            f"Host-first startup failed (falling back to configured script "
            f"'{script_path}'): {host_out}"
        )

    # Retry loop configuration
    retries = 4
    backoff = 3  # seconds base
    tried_cmds = []

    for attempt in range(retries):
       proc, out = spawn_and_capture(cmd, env, srv.llama_log_file, srv.logger, model_name=_display)
       tried_cmds.append((cmd, out))
       if proc is not None:
           srv.logger.info(f"Started llama-server with command: {' '.join(cmd)}")
           return proc
       srv.logger.warning(f"Attempt {attempt+1} failed quickly: {out}")
       time.sleep(backoff * (2 ** attempt))

    # All attempts failed — assemble diagnostic
    msg_lines = ["Failed to start llama-server. Attempts:"]
    for cmd, out in tried_cmds:
       cmd_str = " ".join(cmd)
       snippet = (out or "(no immediate output)").strip()
       if len(snippet) > 1000:
           snippet = snippet[:1000] + "...[truncated]"
       msg_lines.append(f"- {cmd_str}: {snippet}")
    msg_lines.append("")
    msg_lines.append("Hints:")
    msg_lines.append(" - Ensure podman/container runtime is available and the start script is executable")
    msg_lines.append(" - Check the llama-server binary is installed and configured correctly")
    msg = "\n".join(msg_lines)
    srv.logger.error(msg)
    srv.last_start_failure = msg
    srv.broadcast_status_sync("error", {"message": msg, "current_model": None, "llama_server_running": False})
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
    for f in srv.log_dir.glob(f"{base_name}.*{suffix}"):
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
           srv.logger.debug(f"Deleted old llama-server log: {f}")

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

    if srv.llama_process is not None:
       pid = getattr(srv.llama_process, 'pid', 'N/A')
       srv.logger.info(f"Stopping llama-server wrapper (PID: {pid})")
       # Only clean up process and model state if srv.llama_process looks like
       # a real subprocess (has terminate/kill/wait methods). If it's a
       # test mock or invalid object, skip process cleanup.
       is_real_process = hasattr(srv.llama_process, 'terminate') or hasattr(srv.llama_process, 'kill')
       if is_real_process:
           previous_model = srv.current_model
           srv.llama_process.terminate()
           try:
               srv.llama_process.wait(timeout=30)
           except subprocess.TimeoutExpired:
               srv.logger.warning("llama-server wrapper did not terminate gracefully, killing...")
               if hasattr(srv.llama_process, 'kill'):
                   srv.llama_process.kill()
               if hasattr(srv.llama_process, 'wait'):
                   srv.llama_process.wait()
           srv.llama_process = None
           try:
               if previous_model:
                   metrics.record_model_unloaded(previous_model)
           except Exception:
               pass
           srv.current_model = None
           srv.backend_ready = False
           srv.logger.info("llama-server stopped")
       else:
           srv.llama_process = None
           srv.backend_ready = False
           srv.logger.info("llama-server stop skipped (no valid process)")

    # Close log file if open
    if srv.llama_log_file is not None and srv.llama_log_file != subprocess.DEVNULL:
       try:
           srv.llama_log_file.close()
       except Exception:
           pass
       srv.llama_log_file = None


# ---------------------------------------------------------------------------
# TTS server lifecycle
# ---------------------------------------------------------------------------


def start_tts_server() -> subprocess.Popen | None:
    """Start the qwentts TTS server using the configured start script.

    Returns a subprocess.Popen when successful, None on failure.
    """
    srv = _srv()

    server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}
    script_path = server_cfg.get(
        "tts_start_script",
        str(Path(__file__).parent.parent / "scripts" / "start-qwentts.sh")
    )
    tts_port = int(server_cfg.get("tts_server_port", 8081))
    tts_model_path = str(server_cfg.get("tts_model_path", ""))
    tts_codec_path = str(server_cfg.get("tts_codec_path", ""))

    if not os.path.isfile(script_path):
        srv.logger.warning(f"TTS start script not found: {script_path}")
        return None

    env = os.environ.copy()
    env["QWTTS_PORT"] = str(tts_port)
    if tts_model_path:
        env["QWTTS_MODEL"] = tts_model_path
    if tts_codec_path:
        env["QWTTS_CODEC"] = tts_codec_path

    cmd = [script_path]
    cmd.extend(["--port", str(tts_port)])
    if tts_model_path and os.path.isfile(tts_model_path):
        cmd.extend(["--model", tts_model_path])
    if tts_codec_path and os.path.isfile(tts_codec_path):
        cmd.extend(["--codec", tts_codec_path])
    cmd.extend(["--lang", "english"])

    srv.logger.info(f"Starting TTS server: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(cmd, env=env)
    except FileNotFoundError:
        srv.logger.warning(f"Command not found: {cmd[0]}")
        return None
    except Exception as e:
        srv.logger.error(f"Failed to start TTS server: {e}")
        return None

    srv.logger.info(f"Started TTS server (PID: {proc.pid})")
    return proc


def stop_tts_server():
    """Stop the currently running qwentts TTS server."""
    srv = _srv()

    if srv.tts_process is not None:
        pid = getattr(srv.tts_process, 'pid', 'N/A')
        srv.logger.info(f"Stopping TTS server (PID: {pid})")
        is_real_process = hasattr(srv.tts_process, 'terminate') or hasattr(srv.tts_process, 'kill')
        if is_real_process:
            srv.tts_process.terminate()
            try:
                srv.tts_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                srv.logger.warning("TTS server did not terminate gracefully, killing...")
                if hasattr(srv.tts_process, 'kill'):
                    srv.tts_process.kill()
                if hasattr(srv.tts_process, 'wait'):
                    srv.tts_process.wait()
            srv.tts_process = None
            srv.logger.info("TTS server stopped")
        else:
            srv.tts_process = None
            srv.logger.info("TTS server stop skipped (no valid process)")
    else:
        srv.logger.info("TTS server not running, nothing to stop")


async def wait_for_tts_server(tts_port: int = 8081, timeout: int = 30) -> bool:
    """Wait for the TTS server to become ready by polling its health endpoint."""
    srv = _srv()
    health_url = f"http://localhost:{tts_port}/v1/audio/speech"

    start_time = time.time()
    client = srv._http_client if srv._http_client else httpx.AsyncClient(timeout=httpx.Timeout(5.0))
    try:
        while time.time() - start_time < timeout:
            # Check if tts process died
            if srv.tts_process is not None and hasattr(srv.tts_process, 'poll') and srv.tts_process.poll() is not None:
                exit_code = srv.tts_process.returncode
                srv.logger.error(f"TTS server process exited with code {exit_code}")
                return False

            try:
                response = await client.get(health_url, timeout=5)
                # Any response (including 4xx) means the server is running
                if int(getattr(response, 'status_code', 0) or 0) > 0:
                    srv.logger.info(f"TTS server is ready on port {tts_port}")
                    return True
            except asyncio.CancelledError:
                srv.logger.info("Wait for TTS server cancelled")
                raise
            except Exception:
                pass

            try:
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                srv.logger.info("Wait for TTS server cancelled")
                raise
    finally:
        if not srv._http_client:
            try:
                await client.aclose()
            except Exception:
                pass

    srv.logger.error(f"TTS server failed to start within {timeout} seconds")
    return False


async def ensure_model_loaded(requested_model: str | None) -> bool:
    srv = _srv()
    """
    Ensure the requested model is loaded in llama-server.
    Returns True if the model is ready, False if there was an error.
    """


    llama_model = srv.get_local_model_name(requested_model)
    if llama_model is None:
       srv.backend_ready = False
       return False

    server_config = srv.config.get("server", {})
    router_mode = server_config.get("llama_router_mode", False)

    # Use a try/finally around the model switch lock so we can reliably
    # decrement the global refcount if this invocation incremented it.
    incremented_here = False
    try:
       async with srv.model_switch_lock:
           if srv.current_model == llama_model and srv.llama_process is not None:
               # Check if process is still running
               if srv.llama_process.poll() is None:
                   srv.backend_ready = True
                   return True
               else:
                   srv.logger.warning("llama-server process died, restarting...")

           # If no background load marker exists for this model then this
           # synchronous path should increment the refcount so status
           # endpoints observe switching across threads/loops.
           try:
               if not srv.background_loads.get(llama_model):
                   srv._inc_model_switch_refcount()
                   incremented_here = True
           except Exception:
               # Best-effort: do not fail switching due to refcount errors
               pass

           # Broadcast that we're switching models
           await srv.broadcast_status("switching", {
               "target_model": llama_model,
               "previous_model": srv.current_model
           })

           timeout = server_config.get("llama_startup_timeout", 300)

           # Router mode: ensure server process is running and models loaded
           if router_mode:
               if srv.llama_process is None or srv.llama_process.poll() is not None:
                   try:
                       srv.llama_process = srv.start_llama_server(None, display_name=requested_model)
                   except TypeError:
                       # Backwards-compatible call for older test monkeypatches
                       srv.llama_process = srv.start_llama_server(None)

                   if srv.llama_process is None:
                       srv.logger.error("start_llama_server failed to spawn router process")
                       srv.backend_ready = False
                       return False

                   # Wait for the backend to become reachable
                   if not await srv.wait_for_llama_server(timeout):
                       await srv.broadcast_status("error", {
                           "message": "Failed to start router-mode llama-server",
                           "current_model": None,
                           "llama_server_running": False
                       })
                       srv.stop_llama_server()
                       srv.backend_ready = False
                       return False

               if not await srv.router_load_model(llama_model):
                   await srv.broadcast_status("error", {
                       "message": f"Failed to load model {llama_model} via router",
                       "current_model": None,
                       "llama_server_running": True
                   })
                   srv.backend_ready = False
                   return False

               # Enforce embeddings pinned: ensure embeddings preset remains loaded
               embeddings_preset = server_config.get("embeddings_model")
               if embeddings_preset:
                   try:
                       await srv.router_load_model(embeddings_preset)
                       await srv.router_wait_for_model(embeddings_preset, timeout=server_config.get("llama_embed_load_timeout", 30))
                   except Exception:
                       srv.logger.warning("Failed to ensure embeddings preset is loaded/pinned")

               load_timeout = server_config.get("llama_model_load_timeout", timeout)
               if not await srv.router_wait_for_model(llama_model, timeout=load_timeout):
                   await srv.broadcast_status("error", {
                       "message": f"Timed out waiting for model {llama_model} to load",
                       "current_model": None,
                       "llama_server_running": True
                   })
                   srv.backend_ready = False
                   return False

               srv.current_model = llama_model
               try:
                   metrics.record_model_loaded(llama_model)
               except Exception:
                   pass

               await srv.broadcast_status("ready", {
                   "current_model": llama_model,
                   "llama_server_running": True
               })
               srv.backend_ready = True
               return True

           # Need to switch models or restart (single-model path)
           srv.stop_llama_server()

           # Start llama-server via the configured start script
           try:
               srv.llama_process = srv.start_llama_server(llama_model, display_name=requested_model)
           except TypeError:
               srv.llama_process = srv.start_llama_server(llama_model)

           # If starting the process failed immediately (start_llama_server returns None),
           # fail fast instead of waiting the full timeout. start_llama_server already
           # broadcasts a detailed error message.
           if srv.llama_process is None:
               srv.logger.error(f"start_llama_server failed to spawn process for model {llama_model}")
               srv.backend_ready = False
               return False

           # Wait for the backend to become reachable
           if await srv.wait_for_llama_server(timeout):
               srv.current_model = llama_model
               try:
                   metrics.record_model_loaded(llama_model)
               except Exception:
                   pass
               # Broadcast success
               await srv.broadcast_status("ready", {
                   "current_model": llama_model,
                   "llama_server_running": True
               })
               srv.backend_ready = True
               return True
           else:
               # Broadcast failure
               await srv.broadcast_status("error", {
                   "message": f"Failed to load model {llama_model}",
                   "current_model": None,
                   "llama_server_running": False
               })
               srv.stop_llama_server()
               srv.backend_ready = False
               return False
    finally:
       # Ensure we decrement the refcount if we incremented it above.
       if incremented_here:
           try:
               srv._dec_model_switch_refcount()
           except Exception:
               pass


# ---------------------------------------------------------------------------
# Backward-compatibility re-exports — router model health monitoring
# (moved to backend_health.py)
# ---------------------------------------------------------------------------
from .backend_health import (  # noqa: E402, F401
    _extract_model_port_from_args,
    _probe_model_instance,
    _router_model_health_loop,
)


