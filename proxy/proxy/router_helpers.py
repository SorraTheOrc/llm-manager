"""
Router Helpers Module

Shared helper functions used by router.py (local routing) and
proxy_remote.py (remote proxying). Extracted to keep individual routing
modules focused and under the ~1000-line guideline.

Includes:
- Request/response logging helpers
- Error response builders (503 with session headers, slot exhaustion)
- Session handling helper
- Concurrency/slot helpers
- Backend retry wrappers
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Lazy server import — avoids circular imports when server.py imports us
# ---------------------------------------------------------------------------
def _srv():
    import proxy.server as _m
    return _m


# ===================================================================
# Request/response logging helpers
# ===================================================================

def log_request(
    request: Request,
    body: bytes,
    source: str,
    endpoint: str = "",
) -> None:
    """Log incoming request details."""
    srv = _srv()
    try:
        method = request.method
        url = str(request.url)
        body_preview = body.decode("utf-8", errors="replace")[:500]
        session_headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower()
            in (
                "x-session-id",
                "x-client-request-id",
                "x-session-affinity",
            )
        }
        if source == "remote" and endpoint:
            srv.logger.info(
                f"[{source}] {method} {url} -> {endpoint} "
                f"body={body_preview} session={session_headers}"
            )
        else:
            srv.logger.info(
                f"[{source}] {method} {url} "
                f"body={body_preview} session={session_headers}"
            )
    except Exception:
        srv.logger.debug("Failed to log request", exc_info=True)


def log_response(status_code: int, content: bytes) -> None:
    """Log response status and size."""
    srv = _srv()
    try:
        srv.logger.info(
            f"Response: {status_code} ({len(content)} bytes)"
        )
    except Exception:
        pass


def log_response_chunk(chunk: bytes) -> None:
    """Log streaming response chunk.

    The ContentOnlyConsoleHandler extracts and displays just the content to console.
    Raw JSON is written to the log file only.
    """
    srv = _srv()
    try:
        chunk_str = chunk.decode("utf-8")[:500] if chunk else ""
        srv.logger.info(f"STREAM CHUNK | {chunk_str}")
    except Exception:
        pass


# ===================================================================
# Backend retry wrappers
# ===================================================================

def _call_with_backend_retries(*args, **kwargs):
    """Wrapper around _call_with_backend_retries to support monkey-patching.
    
    Accesses via _srv() so that tests can monkeypatch server-level
    references (back-compat with test patterns).
    """
    return _srv()._call_with_backend_retries(*args, **kwargs)


def _call_with_empty_retry(*args, **kwargs):
    """Wrapper around _call_with_empty_retry to support monkey-patching.
    
    Accesses via _srv() so that tests can monkeypatch server-level
    references (back-compat with test patterns).
    """
    return _srv()._call_with_empty_retry(*args, **kwargs)


# ===================================================================
# Error response builders
# ===================================================================

def _build_backend_error_response(
    srv,
    path: str,
    session_id: Optional[str],
    session_created: bool,
    is_delta_request: bool,
    session_fallback_reason: Optional[str],
    retry_after: Optional[int] = None,
) -> JSONResponse:
    """Build a 503 error response with session information headers.
    
    Used by both streaming and non-streaming paths in proxy_to_local
    when the backend is unavailable or returns an error.
    """
    if retry_after is None:
        from proxy.backend_health import _self_heal_retry_after_seconds
        retry_after = _self_heal_retry_after_seconds()
    
    headers = {
        "Retry-After": str(retry_after),
        "Cache-Control": "no-store",
    }
    if session_id:
        headers["X-Session-Id"] = session_id
        headers["X-Session-Created"] = "true" if session_created else "false"
        headers["X-Session-Delta"] = "true" if is_delta_request else "false"
        if session_fallback_reason:
            headers["X-Session-Fallback-Reason"] = session_fallback_reason
    
    payload = {
        "error": {
            "type": "backend_error",
            "code": "backend_error",
            "message": "Backend unavailable, please retry later",
        },
        "status": 503,
        "path": f"/{path}",
        "retry_after": retry_after,
    }
    return JSONResponse(status_code=503, content=payload, headers=headers)


def _build_backend_unavailable_response(
    srv, path: str
) -> JSONResponse:
    """Build a 503 response for backend_unavailable state.
    
    Called before any session processing has happened, so no session
    headers are included.
    """
    from proxy.backend_health import _self_heal_retry_after_seconds
    from proxy.metrics import record_http_error
    
    retry_after = _self_heal_retry_after_seconds()
    record_http_error("v1/chat/completions", "5xx", "backend_unavailable")
    
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "type": "backend_unavailable",
                "code": "backend_unavailable",
                "message": "Backend is not available, please retry later",
            },
            "status": 503,
            "path": f"/{path.lstrip('/')}",
            "retry_after": retry_after,
        },
        headers={"Retry-After": str(retry_after), "Cache-Control": "no-store"},
    )


def _build_slot_exhaustion_response(
    server_config: dict, srv, total_slots: int
) -> JSONResponse:
    """Build a 503 response when all llama-server slots are busy."""
    from proxy.metrics import record_http_error
    
    retry_after = int(
        server_config.get("slot_unavailable_retry_after", 5) or 5
    )
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
        headers={
            "Retry-After": str(retry_after),
            "Cache-Control": "no-store",
        },
    )


# ===================================================================
# Concurrency helpers
# ===================================================================

async def _decrement_active_queries(srv) -> None:
    """Safely decrement the active queries counter."""
    try:
        async with srv.active_queries_lock:
            srv.active_queries = max(0, srv.active_queries - 1)
    except Exception:
        pass


async def _increment_active_queries(srv) -> None:
    """Safely increment the active queries counter."""
    try:
        async with srv.active_queries_lock:
            srv.active_queries += 1
    except Exception:
        pass


# ===================================================================
# Header normalization helpers
# ===================================================================

def _normalize_outgoing_headers(
    headers: dict, buffered: bool = False
) -> dict:
    """Normalize outgoing response headers.
    
    Removes content-length when transfer-encoding is present (for streaming).
    """
    result = dict(headers)
    if buffered:
        pass
    else:
        # For streaming, remove content-length if TE is present
        if "transfer-encoding" in {k.lower() for k in result.keys()}:
            for k in list(result.keys()):
                if k.lower() == "content-length":
                    del result[k]
    return result


# ===================================================================
# Session handling helper
# ===================================================================

async def _handle_session(
    srv,
    body_json: dict,
    server_config: dict,
    request_headers,
) -> dict:
    """Handle session resolution and delta calculation.
    
    Returns a dict with session_id, session_created, is_delta_request,
    session_fallback_reason, delta_messages, and updated body_json/body.
    """
    from proxy.session import (
        _resolve_session_id_header,
        _log_session_header_resolution,
        _classify_delta_routing,
        _record_delta_payload_bytes,
        _record_restore_fallback,
        _build_slot_context,
        _invalidate_session_and_slot,
    )
    
    result = {
        "session_id": None,
        "session_created": False,
        "is_delta_request": False,
        "session_fallback_reason": None,
        "delta_messages": None,
        "body_json": body_json,
        "body_override": None,
        "original_message_count": 0,
    }
    
    session_id_header, session_header_source = _resolve_session_id_header(
        request_headers
    )
    
    if isinstance(body_json, dict) and "messages" in body_json:
        result["original_message_count"] = len(body_json["messages"])
        _log_session_header_resolution(session_id_header, session_header_source)
        
        try:
            session, session_created = await srv.session_manager.get_or_create(
                session_id_header
            )
            result["session_id"] = session.session_id
            result["session_created"] = session_created
            
            if not session_created and session.message_count > 0:
                delta_messages, history_matches = srv.session_manager.compute_delta(
                    session.messages, body_json["messages"]
                )
                is_delta_request, session_fallback_reason = _classify_delta_routing(
                    history_matches=history_matches,
                    delta_message_count=len(delta_messages),
                    restore_confirmed=bool(session.restore_confirmed),
                    require_restore_signal=bool(
                        server_config.get("session_require_restore_signal", False)
                    ),
                    force_full_prompt=_should_force_full_prompt_from_config(
                        body_json, server_config
                    ),
                )
                result["is_delta_request"] = is_delta_request
                result["session_fallback_reason"] = session_fallback_reason
                result["delta_messages"] = delta_messages
                
                if is_delta_request:
                    body_json["messages"] = list(delta_messages)
                    try:
                        _record_delta_payload_bytes(
                            len(
                                json.dumps(
                                    delta_messages,
                                    separators=(",", ":"),
                                    ensure_ascii=False,
                                ).encode("utf-8")
                            )
                        )
                    except Exception:
                        pass
                    srv.logger.info(
                        f"Session {result['session_id'][:8]}... strict restore confirmed; "
                        f"forwarding delta ({len(delta_messages)} new messages)"
                    )
                else:
                    if session_fallback_reason == "history_mismatch":
                        from proxy.session import _build_slot_context, _invalidate_session_and_slot
                        _, slot_filename, _ = _build_slot_context(
                            server_config, result["session_id"]
                        )
                        await _invalidate_session_and_slot(
                            result["session_id"],
                            "history_mismatch",
                            slot_filename,
                        )
                        session, session_created = await srv.session_manager.get_or_create(
                            result["session_id"]
                        )
                        result["session_created"] = session_created
                    if session_fallback_reason:
                        _record_restore_fallback(session_fallback_reason)
                    srv.logger.info(
                        f"Session {result['session_id'][:8]}... history match={history_matches} "
                        f"delta_messages={len(delta_messages)} using full prompt "
                        f"reason={session_fallback_reason or 'none'}"
                    )
            elif session_created:
                result["session_fallback_reason"] = "no_existing_history"
            
            # Add session_id and cache_prompt to request body for llama-server
            body_json["cache_prompt"] = True
            body_json["session_id"] = result["session_id"]
            result["body_override"] = json.dumps(body_json).encode("utf-8")
        except Exception:
            srv.logger.warning(
                "Session handling failed, falling back to full history",
                exc_info=True,
            )
            result["session_id"] = None
            result["is_delta_request"] = False
            result["session_fallback_reason"] = "session_handling_error"
    else:
        result["session_id"] = None
    
    return result


def _should_force_full_prompt_from_config(
    body_json: dict, server_config: dict
) -> bool:
    """Determine if a full prompt should be forced based on config."""
    # Simplified version of lifecycle._should_force_full_prompt
    try:
        model_name = body_json.get("model") if body_json else None
        if model_name:
            from proxy.lifecycle import get_model_config
            cfg = get_model_config(model_name)
            if cfg:
                return bool(cfg.get("force_full_prompt", False))
    except Exception:
        pass
    return bool(server_config.get("force_full_prompts", False))


# ===================================================================
# Token accounting helpers
# ===================================================================

def _estimate_tokens_sent(
    body: bytes, body_json: dict, model_name: Optional[str]
) -> int:
    """Estimate tokens sent in the request body."""
    from proxy.utils import count_text_tokens
    
    try:
        tokens_sent = 0
        if isinstance(body_json, dict) and "messages" in body_json:
            for m in body_json.get("messages", []):
                tokens_sent += count_text_tokens(
                    str(m.get("content", "")), model_name
                )
        elif isinstance(body_json, dict) and "input" in body_json:
            inp = body_json["input"]
            if isinstance(inp, list):
                for it in inp:
                    tokens_sent += count_text_tokens(str(it), model_name)
            else:
                tokens_sent += count_text_tokens(str(inp), model_name)
        else:
            tokens_sent += count_text_tokens(
                body.decode("utf-8", errors="replace"), model_name
            )
    except Exception:
        tokens_sent = 0
    return tokens_sent


async def _schedule_token_increment(
    key: str, tokens: int
) -> None:
    """Schedule a token increment in the running event loop."""
    from proxy.observability import _increment_tokens
    
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_increment_tokens("sent", key, tokens))
    except RuntimeError:
        asyncio.run(_increment_tokens("sent", key, tokens))
    except Exception:
        pass


async def _schedule_recv_token_increment(
    key: str, tokens: int
) -> None:
    """Schedule a received token increment."""
    from proxy.observability import _increment_tokens
    
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_increment_tokens("recv", key, tokens))
    except RuntimeError:
        asyncio.run(_increment_tokens("recv", key, tokens))
    except Exception:
        pass


# ===================================================================
# Slot availability check
# ===================================================================

async def _check_slot_availability(
    srv,
    server_config: dict,
    llama_port: int,
    slot_model_name: Optional[str],
    model_name: Optional[str],
    path: str,
) -> Optional[JSONResponse]:
    """Check llama-server slot availability.
    
    Returns a 503 JSONResponse if no slots are available, None otherwise.
    """
    if not (path == "v1/chat/completions" or path.endswith("chat/completions")):
        return None
    
    try:
        slot_model = (
            slot_model_name or model_name or srv.current_model or "Qwen3"
        )
        slots_url = f"http://localhost:{llama_port}/slots?model={slot_model}"
        client = (
            srv._http_client
            if srv._http_client
            else httpx.AsyncClient(timeout=httpx.Timeout(5.0))
        )
        slots_resp = await client.get(slots_url, timeout=5.0)
        if slots_resp.status_code == 200:
            slots_data = slots_resp.json()
            available_slots = 0
            total_slots = 0
            if isinstance(slots_data, list):
                total_slots = len(slots_data)
                available_slots = sum(
                    1
                    for s in slots_data
                    if not s.get("is_processing", True)
                )
            if available_slots == 0 and total_slots > 0:
                return _build_slot_exhaustion_response(
                    server_config, srv, total_slots
                )
    except HTTPException:
        raise
    except Exception:
        pass  # best effort
    
    return None


# ===================================================================
# Request timeout computation
# ===================================================================

def _compute_request_timeout(
    server_config: dict,
    body_json: dict,
) -> httpx.Timeout:
    """Compute the request timeout, using adaptive timeout if enabled."""
    from proxy.lifecycle import _compute_adaptive_timeout, _estimate_prompt_tokens
    
    adaptive_enabled = server_config.get("llama_adaptive_timeout_enabled", False)
    if adaptive_enabled and body_json:
        base_timeout = float(
            server_config.get("llama_adaptive_timeout_base_seconds", 60)
        )
        per_token_timeout = float(
            server_config.get("llama_adaptive_timeout_per_token_seconds", 0.01)
        )
        max_timeout = float(server_config.get("llama_request_timeout", 300))
        timeout_seconds = _compute_adaptive_timeout(
            body_json, base_timeout, per_token_timeout, max_timeout
        )
        _srv().logger.debug(
            "Adaptive timeout: tokens=%d timeout=%.1fs",
            _estimate_prompt_tokens(body_json),
            timeout_seconds,
        )
    else:
        timeout_seconds = server_config.get("llama_request_timeout", 300)
    return httpx.Timeout(timeout_seconds)
