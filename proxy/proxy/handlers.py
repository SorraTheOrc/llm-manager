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

def extract_progress_data(line: Optional[str]) -> Optional[Tuple[int, int, float]]:
    """Extract ``slot_id``, ``n_tokens`` and ``progress`` from llama-server stdout progress lines.

    Returns a tuple (slot_id: int, n_tokens: int, progress: float) or None if the line does
    not contain valid progress data. When slot ID is not found, defaults to 0.
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
        m_slot = re.search(r'\bslot\s+(\d+)\b', text, flags=re.IGNORECASE)
        m_tokens = re.search(r'\bn_tokens\s*=\s*(\d+)\b', text, flags=re.IGNORECASE)
        m_progress = re.search(r'\bprogress\s*=\s*([0-9]+(?:\.[0-9]+)?)\b', text, flags=re.IGNORECASE)
        if not m_tokens or not m_progress:
            return None
        slot_id = int(m_slot.group(1)) if m_slot else 0
        n_tokens = int(m_tokens.group(1))
        progress = float(m_progress.group(1))
        return (slot_id, n_tokens, progress)
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
                        # Aggregate across ALL slots, not just data[0]
                        is_processing = any(
                            bool(slot.get('is_processing', False)) for slot in data
                        )
                        n_decoded = None
                        for slot in data:
                            next_token = slot.get('next_token') if isinstance(slot.get('next_token'), dict) else None
                            if next_token is not None and 'n_decoded' in next_token:
                                slot_n = next_token.get('n_decoded')
                            else:
                                slot_n = slot.get('n_decoded')
                            if slot_n is not None:
                                if n_decoded is None or slot_n > n_decoded:
                                    n_decoded = slot_n
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


def format_progress(
    n_tokens: int,
    total_tokens: int,
    progress: float,
    model_name: str = "unknown",
    slot_id: int = 0,
    tokens_per_sec: Optional[float] = None,
) -> str:
    """Return a clean, log-friendly progress string without terminal control characters.

    Includes model/slot prefix and tokens-per-second rate when available.
    Output is suitable for logging via the Python logging system (no ANSI
    escape codes, no carriage returns).
    """
    try:
        pct = int(max(0, min(100, int(progress * 100))))
    except Exception:
        pct = 0

    # Build TPS suffix
    if tokens_per_sec is not None:
        tps_str = f" @ {tokens_per_sec:.1f} tok/s"
    else:
        tps_str = " @ --.- tok/s"

    return f"[slot:{slot_id} {model_name}] Processing {n_tokens}/{total_tokens} tokens ({pct}%){tps_str}"


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
         "llama_server_running": bool,
         "available_slots": int,
         "total_slots": int,
         "local_owner_session_id": str | None,
         "local_owner_lease_remaining_seconds": float | None}

    ``available_slots`` and ``total_slots`` reflect the model-serving slot
    state reported by llama-server's ``/slots`` endpoint. When the server is
    not running or the query fails both default to 0.

    Timeout is configurable via the ``STATUS_QUERY_TIMEOUT`` env var
    (seconds, default 1.0).

    Each call is logged with a ``status_request`` structured message that
    includes the response fields and request latency (ms).
    """
    import os  # noqa: local import for config access

    _start = time.monotonic()

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

    # -- slots query (lightweight, short timeout) -------------------------
    available_slots = 0
    total_slots = 0
    if llama_running:
        try:
            server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}
            llama_port = int(server_cfg.get("llama_server_port", 8080) or 8080)
            client = srv._http_client if srv._http_client else httpx.AsyncClient(timeout=5.0)
            from proxy.observability import _query_slots
            available_slots, total_slots = await _query_slots(client, llama_port, timeout=2.0)
        except Exception:
            # slots query is best-effort; default to 0 on failure
            pass

    # -- local dispatch lease info (LP-0MR9G183O004SJLO) --------------------
    local_owner_session_id = None
    local_owner_lease_remaining_seconds = None
    try:
        lock = getattr(srv, "local_dispatch_records_lock", None)
        records = getattr(srv, "local_dispatch_records", {})
        if lock is not None:
            async with lock:
                for sid, rec in list(records.items()):
                    if rec.get("active"):
                        local_owner_session_id = sid
                        remaining = rec.get("expires_at", 0) - time.monotonic()
                        local_owner_lease_remaining_seconds = max(0.0, remaining)
                        break
    except Exception:
        pass

    # -- structured log entry with latency --------------------------------
    _latency_ms = int((time.monotonic() - _start) * 1000)
    logger.info(
        "status_request",
        extra={
            "latency_ms": _latency_ms,
            "llama_server_running": llama_running,
            "active_query": active,
            "model_switch_in_progress": switch_in_progress,
            "current_model": cm,
            "available_slots": available_slots,
            "total_slots": total_slots,
            "local_owner_session_id": local_owner_session_id,
            "local_owner_lease_remaining_seconds": local_owner_lease_remaining_seconds,
        },
    )

    return {
        "active_query": bool(active),
        "model_switch_in_progress": bool(switch_in_progress),
        "current_model": cm,
        "llama_server_running": bool(llama_running),
        "available_slots": available_slots,
        "total_slots": total_slots,
        "local_owner_session_id": local_owner_session_id,
        "local_owner_lease_remaining_seconds": local_owner_lease_remaining_seconds,
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
# /admin/sessions — session DELETE only; GET is served by ui.py list_all_sessions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# /v1/leases/release  —  Proactive lease release (LP-0MRFOF7XO003T7CT)
# ---------------------------------------------------------------------------

@router.post("/v1/leases/release")
async def release_lease(request: Request):
    """Explicitly release the dispatch lease for the caller's session.

    Accepts a JSON body with a ``session_id`` field and removes the
    corresponding dispatch record from ``local_dispatch_records``,
    allowing other sessions to acquire the local backend immediately.

    The endpoint is idempotent: calling it with a session_id that has
    no matching lease returns ``200 OK`` with ``{"status": "ok"}``.

    Returns ``400 Bad Request`` when ``session_id`` is missing, empty,
    or ``null``.
    """
    srv = _srv()

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON body",
        )

    session_id = body.get("session_id")
    if not session_id or not isinstance(session_id, str) or not session_id.strip():
        raise HTTPException(
            status_code=400,
            detail="session_id is required",
        )

    session_id = session_id.strip()

    try:
        from proxy.router_helpers import _release_local_dispatch
        await _release_local_dispatch(srv, session_id)
    except Exception as e:
        logger.exception(
            "Failed to release dispatch lease for session %s",
            session_id[:8] if session_id else "unknown",
        )
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# /v1/voices  —  List available TTS voices
# ---------------------------------------------------------------------------

@router.get("/v1/voices")
async def list_voices():
    """List available TTS voices from the tts-server.

    Forwards a GET request to the tts-server ``/v1/voices`` endpoint
    and returns the JSON response with available speaker voices.

    Returns ``200 OK`` with the voices list on success.
    Returns ``502 Bad Gateway`` when the tts-server is unreachable.
    """
    srv = _srv()
    server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}

    tts_host = server_cfg.get("tts_server_host", "localhost")
    tts_port = server_cfg.get("tts_server_port", 8081)
    tts_url = f"http://{tts_host}:{tts_port}/v1/voices"

    try:
        client = srv._http_client if srv._http_client else None
        if client is None:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as c:
                resp = await c.get(tts_url, timeout=10.0)
        else:
            try:
                resp = await client.get(tts_url, timeout=10.0)
            except RuntimeError:
                async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as c:
                    resp = await c.get(tts_url, timeout=10.0)
    except httpx.ConnectError as e:
        logger.error("TTS server unreachable at %s: %s", tts_url, e)
        raise HTTPException(status_code=502, detail=f"TTS server unreachable: {e}")
    except httpx.TimeoutException as e:
        logger.error("TTS server timeout at %s: %s", tts_url, e)
        raise HTTPException(status_code=502, detail=f"TTS server timeout: {e}")
    except Exception as e:
        logger.exception("Unexpected error forwarding TTS request to %s", tts_url)
        raise HTTPException(status_code=502, detail=f"TTS server error: {e}")

    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ---------------------------------------------------------------------------
# /v1/audio/speech  —  TTS (Speech Synthesis)
# ---------------------------------------------------------------------------

@router.post("/v1/audio/speech")
async def create_speech(request: Request):
    """Generate speech from text using the local TTS server backend.

    OpenAI-compatible endpoint that forwards requests to the qwentts.cpp
    ``tts-server`` (which should be running alongside the proxy).

    Request body (JSON)::

        {
            "model": "qwen3-tts",
            "input": "Text to convert to speech",
            "voice": "default",
            "response_format": "wav"
        }

    Returns audio content with ``Content-Type: audio/wav`` on success.
    Returns ``400 Bad Request`` for invalid/missing parameters.
    Returns ``502 Bad Gateway`` when the tts-server is unreachable.
    """
    srv = _srv()
    server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}

    # Parse request body
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Validate required parameters
    model = body.get("model")
    input_text = body.get("input")

    if not model or not isinstance(model, str):
        raise HTTPException(status_code=400, detail="model is required and must be a string")

    if not isinstance(input_text, str) or not input_text.strip():
        raise HTTPException(status_code=400, detail="input is required and must be a non-empty string")

    # Optional parameters
    voice = body.get("voice", "")
    response_format = body.get("response_format", "wav")
    instructions = body.get("instructions", "")

    # Input length guard
    if len(input_text) > 10000:
        raise HTTPException(status_code=400, detail="input exceeds maximum length of 10000 characters")

    # Build tts-server URL from config
    tts_host = server_cfg.get("tts_server_host", "localhost")
    tts_port = server_cfg.get("tts_server_port", 8081)
    tts_url = f"http://{tts_host}:{tts_port}/v1/audio/speech"

    # Forward request to tts-server (omit empty optional fields)
    forward_body = {
        "model": model,
        "input": input_text,
        "response_format": response_format,
    }
    if voice:
        forward_body["voice"] = voice
    if instructions:
        forward_body["instructions"] = instructions

    try:
        client = srv._http_client if srv._http_client else None
        if client is None:
            # No shared client — create a temporary one
            async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as c:
                resp = await c.post(tts_url, json=forward_body, timeout=180.0)
        else:
            try:
                # Use shared client directly without entering/closing it
                resp = await client.post(tts_url, json=forward_body, timeout=180.0)
            except RuntimeError:
                # Shared client was closed by another handler — fall back to temp
                async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as c:
                    resp = await c.post(tts_url, json=forward_body, timeout=180.0)
    except httpx.ConnectError as e:
        logger.error("TTS server unreachable at %s: %s", tts_url, e)
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "tts_error",
                    "code": "tts_server_unreachable",
                    "message": f"TTS server unreachable: {e}",
                },
                "status": 502,
                "path": "/v1/audio/speech",
            },
        )
    except httpx.TimeoutException as e:
        logger.error("TTS server timeout at %s: %s", tts_url, e)
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "tts_error",
                    "code": "tts_server_timeout",
                    "message": f"TTS server timeout: {e}",
                },
                "status": 502,
                "path": "/v1/audio/speech",
            },
        )
    except Exception as e:
        logger.exception("Unexpected error forwarding TTS request to %s", tts_url)
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "tts_error",
                    "code": "tts_server_error",
                    "message": f"TTS server error: {e}",
                },
                "status": 502,
                "path": "/v1/audio/speech",
            },
        )

    # Check for backend HTTP errors (>=400) and return structured 502
    if resp.status_code >= 400:
        try:
            root_body = json.loads(resp.content.decode("utf-8", errors="replace"))
        except Exception:
            root_body = resp.content.decode("utf-8", errors="replace")
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "tts_error",
                    "code": "tts_server_error",
                    "message": f"TTS server returned HTTP {resp.status_code}",
                },
                "status": 502,
                "path": "/v1/audio/speech",
                "root_cause": {
                    "backend_status_code": resp.status_code,
                    "backend_body": root_body,
                },
            },
        )

    # Return the audio response from tts-server
    content_type = resp.headers.get("content-type", "audio/wav")
    return Response(content=resp.content, media_type=content_type, status_code=resp.status_code)
