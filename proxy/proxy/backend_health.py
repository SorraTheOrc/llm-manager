"""
Backend Health Module

Backend recovery, self-healing, watchdog monitoring, and worker-health
functions extracted from the monolithic lifecycle.py.

Uses a lazy server import (_srv()) to access module-level state without
circular import issues.

Backward-compatible re-exports:
    Functions moved to per-server backend modules (backends/llama.py,
    backends/tts.py) are re-exported here so that existing import paths
    (from proxy.backend_health import ...) continue to work.
"""

import subprocess
import time

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

def _worker_process_unhealthy(proc: subprocess.Popen | None) -> bool:
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


def _coerce_float(value, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_int(value, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _extract_model_port_from_args(args: list) -> int | None:
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


# ===================================================================
# Backward-compatible re-exports
#
# Functions moved to proxy/proxy/backends/llama.py
# Functions moved to proxy/proxy/backends/tts.py
#
# These re-exports preserve existing import chains (server.py, lifecycle.py,
# router_helpers.py, tests) so they continue to work without modification.
# ===================================================================
from .backends.llama import (  # noqa: E402, F401
    _attempt_router_self_heal,
    _backend_watchdog_loop,
    _probe_model_instance,
    _probe_model_instance_with_retries,
    _router_model_health_loop,
)

from .backends.tts import (  # noqa: E402, F401
    _attempt_tts_self_heal,
    _get_tts_self_heal_max_attempts,
    _get_tts_self_heal_probe_timeout,
    _get_tts_self_heal_window,
    _get_tts_watchdog_interval,
    _tts_recovery_snapshot,
    _tts_watchdog_loop,
)
