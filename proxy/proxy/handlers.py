"""
HTTP request handlers for the proxy server, extracted from the monolithic
server.py module.

Each handler is a plain async function that receives a FastAPI ``Request``
object (and optional path/query parameters) and returns a Response.

Module-level access to server globals (config, current_model, etc.) is
performed via lazy import of ``proxy.proxy.server`` at function-call time
to avoid circular-import problems during the transitional refactor period.
"""

import asyncio
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from proxy.provider import get_model_type

logger = logging.getLogger("llama-proxy")

# APIRouter for use by the main server app
router = APIRouter()


# ---------------------------------------------------------------------------
# Internal helpers  (lazy server import)
# ---------------------------------------------------------------------------

def _srv():
    """Lazy import of the server module to avoid circular imports."""
    import proxy.server as _m
    return _m


# ---------------------------------------------------------------------------
# Progress parsing helpers
# ---------------------------------------------------------------------------

def extract_progress_data(line: Optional[str]) -> Optional[Tuple[int, float]]:
    """Extract ``n_tokens`` and ``progress`` from llama-server stdout progress lines.

    Returns a tuple (n_tokens: int, progress: float) or None if the line does
    not contain valid progress data.
    """
    if not isinstance(line, str):
        return None
    text = line.strip()
    if not text:
        return None
    if 'n_tokens' not in text or 'progress' not in text:
        return None
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


# NOTE: slot_polling_state and _slot_polling_tasks remain defined in server.py
# because tests access them via the server module. Handlers access them
# through the lazy _srv() helper.


async def poll_slots_for_model(
    model: str,
    llama_port: int = 0,
    interval: float = 0.5,
    max_polls: Optional[int] = None,
) -> None:
    """Poll the llama-server `/slots` endpoint for a given model and update
    ``slot_polling_state[model]``.
    """
    polls = 0
    if not model:
        return None
    slots_url = f"http://localhost:{llama_port}/slots?model={model}"
    while True:
        try:
            srv = _srv()
            client = srv._http_client if srv._http_client else httpx.AsyncClient(timeout=httpx.Timeout(5.0))
            if getattr(client, '__aenter__', None):
                async with client as c:
                    resp = await c.get(slots_url, timeout=5.0)
            else:
                resp = await client.get(slots_url, timeout=5.0)

            n_decoded = None
            is_processing = False
            if resp is not None and getattr(resp, 'status_code', None) == 200:
                try:
                    data = resp.json()
                except Exception:
                    data = None
                if isinstance(data, list):
                    if data:
                        slot = data[0]
                        is_processing = bool(slot.get('is_processing', False))
                        next_token = slot.get('next_token') if isinstance(slot.get('next_token'), dict) else None
                        if next_token is not None and 'n_decoded' in next_token:
                            n_decoded = next_token.get('n_decoded')
                        else:
                            n_decoded = slot.get('n_decoded')
                elif isinstance(data, dict):
                    is_processing = bool(data.get('is_processing', False))
                    next_token = data.get('next_token')
                    if isinstance(next_token, dict) and 'n_decoded' in next_token:
                        n_decoded = next_token.get('n_decoded')
                    else:
                        n_decoded = data.get('n_decoded')
                # Access server's slot_polling_state
                srv.slot_polling_state[model] = {'is_processing': is_processing, 'n_decoded': n_decoded}
            else:
                srv.slot_polling_state[model] = {'is_processing': False, 'n_decoded': None}
        except Exception:
            try:
                _srv().slot_polling_state[model] = {'is_processing': False, 'n_decoded': None}
            except Exception:
                pass

        polls += 1
        if max_polls is not None and polls >= max_polls:
            break
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break


def start_slot_polling(model: str, llama_port: int, interval: float = 0.5) -> None:
    """Start an asyncio task to poll the slots endpoint for ``model``."""
    srv = _srv()
    if not model:
        return None
    if model in srv._slot_polling_tasks and not srv._slot_polling_tasks[model].done():
        return None
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(poll_slots_for_model(model, llama_port, interval, None))
        srv._slot_polling_tasks[model] = task
    except RuntimeError:
        def _run():
            try:
                asyncio.run(poll_slots_for_model(model, llama_port, interval, None))
            except Exception:
                pass
        t = threading.Thread(target=_run, daemon=True)
        t.start()


def format_progress(n_tokens: int, total_tokens: int, progress: float) -> str:
    """Return a formatted, ANSI-dimmed, in-place-updating progress string."""
    try:
        pct = int(max(0, min(100, int(progress * 100))))
    except Exception:
        try:
            pct = int(max(0, min(100, int(float(progress) * 100))))
        except Exception:
            pct = 0
    pct = int(max(0, min(100, int(progress * 100)))) if isinstance(progress, (int, float)) else pct
    dim = "\x1b[2m"
    reset = "\x1b[0m"
    body = f"Processing {n_tokens}/{total_tokens} tokens ({pct}%)"
    return f"\r{dim}{body}{reset}"


# ---------------------------------------------------------------------------
# /llama/local/status
# ---------------------------------------------------------------------------

@router.get("/llama/local/status")
async def get_llama_local_status():
    """Return a small JSON object describing local llama-server status.

    The endpoint is designed to be non-blocking even when the underlying
    llama-server is busy. ``query_llama_status()`` is wrapped with a short
    timeout so the endpoint itself remains responsive under load (target
    response time < 5 s).

    Fields returned::

        {"active_query": bool,
         "model_switch_in_progress": bool,
         "current_model": str | None,
         "llama_server_running": bool}

    Timeout is configurable via the ``STATUS_QUERY_TIMEOUT`` env var
    (seconds, default 1.0).
    """
    import os  # noqa: local import for config access

    # -- query_llama_status with timeout (non-blocking guarantee) ---------
    srv = _srv()
    timeout = float(os.environ.get("STATUS_QUERY_TIMEOUT", "1.0"))
    try:
        status = await asyncio.wait_for(srv.query_llama_status(), timeout=timeout)
    except (asyncio.TimeoutError, Exception):
        status = {"llama_server_running": False, "n_ctx": None, "kv_cache_tokens": None, "router_mode": False}

    llama_running = bool(status.get("llama_server_running", False))

    # -- model_switch_in_progress (thread-safe read of refcount) ----------
    switch_in_progress = False
    try:
        if hasattr(srv, "model_switch_refcount"):
            with srv.model_switch_refcount_lock:  # type: ignore[arg-type]
                refcount = srv.model_switch_refcount
            switch_in_progress = refcount > 0
    except Exception:
        switch_in_progress = False

    # -- secondary indicators ---------------------------------------------
    if not switch_in_progress:
        try:
            switch_in_progress = srv.model_switch_lock.locked() if srv.model_switch_lock is not None else False
        except Exception:
            pass

    if not switch_in_progress:
        try:
            switch_in_progress = bool(srv.background_loads)
        except Exception:
            pass

    cm = srv.current_model if llama_running else None

    # -- active queries (non-blocking snapshot) ---------------------------
    active = False
    try:
        async with srv.active_queries_lock:
            active = srv.active_queries > 0
    except Exception:
        active = False

    return {
        "active_query": bool(active),
        "model_switch_in_progress": bool(switch_in_progress),
        "current_model": cm,
        "llama_server_running": bool(llama_running),
    }


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@router.get("/health")
async def health_check():
    """Health check endpoint with readiness gating."""
    srv = _srv()
    server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}
    router_mode = bool(server_cfg.get("llama_router_mode", False))
    loaded_models = None
    if router_mode:
        router_models = await srv.router_list_models()
        loaded_models = srv._extract_router_model_ids(router_models)

    llama_running = srv.llama_process is not None and srv.llama_process.poll() is None
    llama_port = int(server_cfg.get("llama_server_port", 8080) or 8080)
    backend_reachable = bool(llama_running and await srv._probe_backend_reachable(llama_port))
    self_healing = srv._is_self_healing_active()
    ready = bool(llama_running and srv.backend_ready and backend_reachable and not self_healing)

    return {
        "status": "healthy" if ready else "degraded",
        "ready": ready,
        "current_model": srv.current_model,
        "loaded_models": loaded_models,
        "llama_server_running": llama_running,
        "backend_reachable": backend_reachable,
        "self_healing_in_progress": self_healing,
        "backend_recovery": srv._backend_recovery_snapshot(),
        "backend_signals": dict(srv.backend_signal_counts),
    }


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

@router.get("/v1/models")
async def list_models():
    """List available models from proxy configuration."""
    srv = _srv()
    models_list = []
    for name, cfg in srv.config.get("models", {}).items():
        mtype = get_model_type(cfg)
        models_list.append({
            "id": name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local" if mtype == "local" else ("remote" if mtype == "remote" else "unknown"),
            "type": mtype or "unknown",
            "aliases": cfg.get("aliases", []),
        })
    return {"object": "list", "data": models_list}


# ---------------------------------------------------------------------------
# /metrics  (Prometheus)
# ---------------------------------------------------------------------------

@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus scrape endpoint (text/plain, exposition format)."""
    srv = _srv()
    try:
        payload, content_type = srv.metrics.generate_metrics_payload()
        return Response(content=payload, media_type=content_type)
    except Exception:
        raise HTTPException(status_code=503, detail="Prometheus metrics unavailable")


# ---------------------------------------------------------------------------
# /admin/metrics
# ---------------------------------------------------------------------------

@router.get("/admin/metrics")
async def admin_metrics():
    """Return router/memory/metrics for observability."""
    srv = _srv()
    server_config = srv.config.get("server", {})
    models_max = server_config.get("llama_models_max")
    router_mode = server_config.get("llama_router_mode", False)
    loaded_models = None
    if router_mode:
        router_models = await srv.router_list_models()
        loaded_models = srv._extract_router_model_ids(router_models)

    per_model = {}
    for m in loaded_models or []:
        per_model[m] = {"last_used": srv.model_last_used.get(m), "rss_bytes": None}

    process_rss = None
    try:
        if 'psutil' in dir(srv) and srv.psutil and srv.llama_process is not None:
            pid = getattr(srv.llama_process, 'pid', None)
            if pid:
                p = srv.psutil.Process(pid)
                mem = p.memory_info()
                process_rss = getattr(mem, 'rss', None)
    except Exception:
        process_rss = None

    if process_rss is not None and loaded_models:
        try:
            per = int(process_rss // len(loaded_models))
            for m in loaded_models:
                per_model[m]['rss_bytes'] = per
        except Exception:
            for m in loaded_models:
                per_model[m]['rss_bytes'] = None

    try:
        srv.metrics.update_metrics(process_rss, loaded_models)
    except Exception:
        pass

    return {
        "models_max": models_max,
        "loaded_models": loaded_models,
        "per_model": per_model,
        "process_rss_bytes": process_rss,
        "session_metrics": srv.session_manager.get_metrics(),
        "restore_success_total": int(srv.session_restore_observability.get("restore_success_total", 0)),
        "restore_fallback_total": dict(srv.session_restore_observability.get("restore_fallback_total", {})),
        "delta_payload_bytes_total": int(srv.session_restore_observability.get("delta_payload_bytes_total", 0)),
        "single_flight_metrics": dict(srv.session_single_flight_observability),
        "guardrail_metrics": {
            "guardrail_cutoff_total": int(srv.session_guardrail_observability.get("guardrail_cutoff_total", 0)),
            "guardrail_cutoff_reasons": dict(srv.session_guardrail_observability.get("guardrail_cutoff_reasons", {})),
            "session_invalidation_total": int(srv.session_guardrail_observability.get("session_invalidation_total", 0)),
            "session_invalidation_reasons": dict(srv.session_guardrail_observability.get("session_invalidation_reasons", {})),
        },
        "backend_ready": bool(srv.backend_ready),
        "backend_recovery": srv._backend_recovery_snapshot(),
        "backend_signals": dict(srv.backend_signal_counts),
    }


# ---------------------------------------------------------------------------
# /admin/dump-counts
# ---------------------------------------------------------------------------

@router.get("/admin/dump-counts")
async def admin_dump_counts():
    """Return in-memory request and token counts for debugging."""
    srv = _srv()
    snap_c = {}
    snap_t = {}
    async with srv.counts_lock:
        snap_c = dict(srv.request_counts)
    async with srv.token_lock:
        snap_t = dict(srv.token_counts)
    return {"counts": snap_c, "tokens": snap_t}


# ---------------------------------------------------------------------------
# /admin/stop-server
# ---------------------------------------------------------------------------

@router.post("/admin/stop-server")
async def admin_stop_server():
    """Stop the llama-server."""
    srv = _srv()
    srv.stop_llama_server()
    return {"status": "success", "message": "llama-server stopped"}


# ---------------------------------------------------------------------------
# /admin/reset-counts
# ---------------------------------------------------------------------------

@router.post("/admin/reset-counts")
async def admin_reset_counts():
    """Reset in-memory and persisted request/token counts to empty."""
    srv = _srv()
    async with srv.counts_lock:
        srv.request_counts = {}
        srv.counts_dirty = True
    async with srv.token_lock:
        srv.token_counts = {}
        srv.tokens_dirty = True
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(srv.save_counts())
        loop.create_task(srv.save_token_counts())
    except RuntimeError:
        await srv.save_counts()
        await srv.save_token_counts()
    for q in list(srv.log_tail_clients):
        try:
            q.put_nowait({"counts": {}, "tokens": {}})
        except Exception:
            continue
    return {"status": "success", "message": "Counts reset"}


# ---------------------------------------------------------------------------
# /admin/sessions
# ---------------------------------------------------------------------------

@router.get("/admin/sessions")
async def admin_list_sessions():
    """List all active sessions with their metadata."""
    srv = _srv()
    sessions = []
    for session_id in list(srv.session_manager._sessions.keys()):
        info = srv.session_manager.get_session_info(session_id)
        if info is not None:
            sessions.append(info)
    return {"sessions": sessions, "total": len(sessions)}


@router.delete("/admin/sessions/{session_id}")
async def admin_delete_session(session_id: str):
    """Delete a specific session by ID."""
    srv = _srv()
    removed = await srv.session_manager.remove(session_id)
    if removed:
        return {"status": "success", "message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


# ---------------------------------------------------------------------------
# /admin/reload-config
# ---------------------------------------------------------------------------

@router.post("/admin/reload-config")
async def reload_config():
    """Reload the configuration from disk."""
    srv = _srv()
    try:
        srv.config = srv.load_config()
        logger.info("Configuration reloaded successfully via admin endpoint")
        return {"status": "success", "message": "Configuration reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
