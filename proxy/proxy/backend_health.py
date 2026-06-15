"""
Backend Health Module

Backend recovery, self-healing, watchdog monitoring, and worker-health
functions extracted from the monolithic lifecycle.py.

Uses a lazy server import (_srv()) to access module-level state without
circular import issues.
"""

import asyncio
import httpx
import subprocess
import time
from typing import Optional

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
# Router model health monitoring
# ===================================================================

def _extract_model_port_from_args(args: list) -> Optional[int]:
    """Extract the model instance port from its argument list.

    The args list comes from the router's /models endpoint status.args
    field and contains things like ``--port 41649``.
    """
    if not isinstance(args, list):
        return None
    try:
        for i, arg in enumerate(args):
            if arg == "--port" and i + 1 < len(args):
                return int(args[i + 1])
    except (ValueError, IndexError):
        pass
    return None


async def _router_model_health_loop() -> None:
    """Periodically check all loaded models' reachability in router mode.

    Queries the router's ``/models`` endpoint, then probes each model that
    reports ``status.value == "loaded"`` by connecting to its port.  If a
    model claims to be loaded but its port is unreachable, triggers an
    unload+reload cycle to recover the dead instance.

    This addresses the case where llama.cpp's child-monitoring thread fails
    to promptly detect a child crash and update the model status, leaving
    the router permanently proxying requests to a dead port.
    """
    srv = _srv()

    while True:
        try:
            interval = float(
                srv.config.get("server", {}).get(
                    "llama_model_health_interval_seconds", 30.0
                )
                or 30.0
            )
            await asyncio.sleep(max(5.0, interval))

            router_mode = bool(
                srv.config.get("server", {}).get("llama_router_mode", False)
            )
            if not router_mode:
                continue

            # Don't interfere while the watchdog is actively recovering
            if _is_self_healing_active():
                continue

            # Get the list of models from the router
            models_data = await srv.router_list_models()
            if not isinstance(models_data, dict):
                continue

            models_payload = models_data.get("data") or models_data.get("models") or []
            if not isinstance(models_payload, list):
                continue

            router_port = int(
                srv.config.get("server", {}).get("llama_server_port", 8080)
            )
            router_host = "127.0.0.1"

            for model_entry in models_payload:
                if not isinstance(model_entry, dict):
                    continue

                model_id = model_entry.get("id")
                raw_status = model_entry.get("status", {})

                # Extract the status value string
                if isinstance(raw_status, str):
                    status_value = raw_status.lower()
                    args = []
                elif isinstance(raw_status, dict):
                    status_value = str(raw_status.get("value", "")).lower()
                    args = raw_status.get("args", [])
                else:
                    continue

                if status_value != "loaded":
                    continue
                if not model_id:
                    continue

                # Extract port from the instance args (--port N)
                port = _extract_model_port_from_args(args)
                if port is None or port <= 0:
                    srv.logger.debug(
                        "model_health: cannot determine port for "
                        "loaded model %s, skipping",
                        model_id,
                    )
                    continue

                # Probe the model's port with a simple health check
                reachable = await _probe_model_instance(
                    router_host, port
                )
                if reachable:
                    continue

                # Model reports "loaded" but its port is unreachable
                srv.logger.error(
                    "model_health: model %s (port %d) is reported as "
                    "loaded but unreachable, triggering recovery",
                    model_id,
                    port,
                )

                # Unload first (may 400 if already unloaded, that's fine)
                srv.logger.info(
                    "model_health: unloading dead model %s",
                    model_id,
                )
                try:
                    client = (
                        srv._http_client
                        if srv._http_client
                        else httpx.AsyncClient(timeout=10.0)
                    )
                    try:
                        await client.post(
                            f"http://{router_host}:{router_port}/models/unload",
                            json={"model": model_id},
                            timeout=10.0,
                        )
                    finally:
                        if not srv._http_client:
                            await client.aclose()
                except Exception as exc:
                    srv.logger.warning(
                        "model_health: unload request for %s failed: %s",
                        model_id,
                        exc,
                    )

                # Reload the model
                srv.logger.info(
                    "model_health: reloading model %s",
                    model_id,
                )
                loaded = await srv.router_load_model(model_id)
                if loaded:
                    srv.logger.info(
                        "model_health: successfully reloaded model %s",
                        model_id,
                    )
                else:
                    srv.logger.error(
                        "model_health: failed to reload model %s",
                        model_id,
                    )

        except asyncio.CancelledError:
            return
        except Exception:
            srv.logger.exception("model health loop error")


async def _probe_model_instance(
    host: str, port: int, timeout: float = 5.0
) -> bool:
    """Probe whether a model instance is reachable on its port.

    Performs a simple GET to ``/health`` on the given host:port.
    Returns True if the endpoint responds with HTTP 200.
    """
    if port <= 0:
        return False
    try:
        client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
        try:
            url = f"http://{host}:{port}/health"
            response = await client.get(url, timeout=timeout)
            return response.status_code == 200
        finally:
            await client.aclose()
    except Exception:
        return False
