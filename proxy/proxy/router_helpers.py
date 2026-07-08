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
from typing import Any, Dict, List, Optional, Tuple, Mapping, Union

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

def _get_request_preview(body_json: Optional[Union[dict, bytes]]) -> str:
    """Extract the first 80 characters of the first non-system user message.

    Parses the JSON body to find the first message whose ``role`` is not
    ``"system"`` and returns the first 80 characters of its ``content``
    field, appending ``...`` if the content is longer than 80 characters.

    Returns an empty string if the body cannot be parsed, contains no
    messages, or contains only system messages.

    Parameters
    ----------
    body_json : dict or bytes or None
        The request body as a parsed dict, raw bytes, or None.

    Returns
    -------
    str
        The request preview (max 83 characters including ``...``).
    """
    try:
        if isinstance(body_json, bytes):
            body_json = json.loads(body_json.decode("utf-8", errors="replace"))
        if not isinstance(body_json, dict):
            return ""
        messages = body_json.get("messages")
        if not isinstance(messages, list):
            return ""
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "system":
                continue
            content = msg.get("content", "")
            if not content:
                continue
            content_str = str(content)
            if len(content_str) > 80:
                return content_str[:80] + "..."
            return content_str
    except Exception:
        pass
    return ""


def _strip_system_messages_from_preview(body: bytes) -> str:
    """Produce a body preview that excludes system message content.

    Parses the JSON body and filters out any ``{"role": "system"...}``
    messages before serialising back to a preview string.  Returns the
    raw body decoded (capped at 500 chars) when JSON parsing fails or the
    body does not contain a ``"messages"`` list.

    This prevents sensitive system-prompt content from appearing in proxy
    logs while still exposing user-facing message content for debugging.
    """
    preview = body.decode("utf-8", errors="replace")[:500]

    try:
        body_json = json.loads(body) if isinstance(body, bytes) else body
        if isinstance(body_json, dict) and "messages" in body_json:
            filtered_messages = [
                msg
                for msg in body_json["messages"]
                if isinstance(msg, dict) and msg.get("role") != "system"
            ]
            if filtered_messages != body_json["messages"]:
                # System messages were present and removed — rebuild JSON.
                body_json = dict(body_json)
                body_json["messages"] = filtered_messages
                preview = json.dumps(body_json, ensure_ascii=False)[:500]
    except Exception:
        # If JSON parsing fails, return the raw preview (existing behaviour).
        pass

    return preview


def log_request(
    request: Request,
    body: bytes,
    source: str,
    endpoint: str = "",
    *,
    session_id: Optional[str] = None,
    slot_id: Optional[str] = "none",
) -> None:
    """Log incoming request details.

    Parameters
    ----------
    request : Request
        The incoming FastAPI request.
    body : bytes
        The raw request body.
    source : str
        Routing source label (``"local"`` or ``"remote"``).
    endpoint : str, optional
        Remote endpoint URL (used only for ``source == "remote"``).
    session_id : str, optional
        Resolved session ID (the internal session identifier). When
        provided it is included in the log line as
        ``session_id=<value>``.
    slot_id : str, optional
        Assigned slot identifier. Defaults to ``"none"``. When a slot
        is assigned the actual ID is logged; otherwise the placeholder
        ``"none"`` or ``"queued"`` is used.

    Notes
    -----
    - System message content is stripped from the body preview to avoid
      leaking sensitive system-prompt data in proxy logs.
    - This function is the single source of truth for request logging and
      is called by both ``proxy_to_local`` and ``proxy_to_remote``.
    """
    srv = _srv()
    try:
        method = request.method
        url = str(request.url)
        body_preview = _strip_system_messages_from_preview(body)
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

        # Build session info portion
        session_parts = [f"session={session_headers}"]
        if session_id is not None:
            session_parts.insert(0, f"session_id={session_id}")
        if slot_id is not None:
            session_parts.append(f"slot={slot_id}")

        session_info = " ".join(session_parts)

        if source == "remote" and endpoint:
            srv.logger.info(
                f"[{source}] {method} {url} -> {endpoint} "
                f"body={body_preview} {session_info}"
            )
        else:
            srv.logger.info(
                f"[{source}] {method} {url} "
                f"body={body_preview} {session_info}"
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


def log_response_chunk(
    chunk: bytes,
    session_id: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    body_json: Optional[Union[dict, bytes]] = None,
) -> None:
    """Log streaming response chunk.

    The ContentOnlyConsoleHandler no longer displays streaming content to the
    console (LP-0MR90HJED005WI1Z). Raw JSON is written to the log file only.

    If the chunk contains a ``finish_reason`` in any ``choices[]`` entry,
    an enhanced ``Stream finished: reason=<reason>`` log line is emitted
    so the stop reason (and optional token usage) appears in both console
    and file logs. When *session_id*, *model*, and *provider* are provided,
    they are appended to the log line.

    When *body_json* is provided, a request preview (first 80 characters of
    the first non-system user message) is included in the finished line.
    """
    srv = _srv()
    try:
        chunk_str = chunk.decode("utf-8")[:500] if chunk else ""
        srv.logger.info(f"STREAM CHUNK | {chunk_str}")
    except Exception:
        pass

    # Detect finish_reason and log stop-reason line (LP-0MQZXHHHO0063YCI)
    try:
        if not chunk:
            return
        chunk_full = chunk.decode("utf-8", errors="replace")
        for line in chunk_full.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                continue
            try:
                j = json.loads(payload)
            except Exception:
                continue
            if not isinstance(j, dict):
                continue
            # Look for finish_reason in any choices[] entry
            finish_reason = None
            for choice in j.get("choices", []):
                if isinstance(choice, dict):
                    fr = choice.get("finish_reason")
                    if fr is not None:
                        finish_reason = fr
                        break
            if finish_reason is not None:
                parts = [f"Stream finished: reason={finish_reason}"]
                usage = j.get("usage")
                if isinstance(usage, dict):
                    pt = usage.get("prompt_tokens")
                    ct = usage.get("completion_tokens")
                    tt = usage.get("total_tokens")
                    if pt is not None or ct is not None or tt is not None:
                        parts.append(f"tokens={pt or 0}/{ct or 0}/{tt or 0}")
                # Add session, provider, model and request preview (LP-0MR90HJED005WI1Z)
                if session_id:
                    parts.append(f"session={session_id}")
                if provider:
                    parts.append(f"provider={provider}")
                if model:
                    parts.append(f"model={model}")
                if body_json is not None:
                    preview = _get_request_preview(body_json)
                    if preview:
                        parts.append(f"request={preview}")
                srv.logger.info(" ".join(parts))
    except Exception:
        pass


# ===================================================================
# Upstream request header normalization
# ===================================================================

_HOP_BY_HOP_REQUEST_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "expect",
    "proxy-connection",
}


def normalize_upstream_request_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    """Normalize inbound headers before proxying to upstream/local backends.

    Removes hop-by-hop transport headers that can produce malformed upstream
    requests (especially when forwarding from one HTTP connection to another).
    Also strips headers referenced by the Connection header token list.
    """
    if not headers:
        return {}

    connection_tokens: set[str] = set()
    for k, v in headers.items():
        if str(k).lower() == "connection":
            try:
                connection_tokens.update(
                    token.strip().lower()
                    for token in str(v).split(",")
                    if token and token.strip()
                )
            except Exception:
                pass

    out: Dict[str, str] = {}
    for k, v in headers.items():
        lk = str(k).lower()
        if lk in ("host", "content-length"):
            continue
        if lk in _HOP_BY_HOP_REQUEST_HEADERS:
            continue
        if lk in connection_tokens:
            continue
        out[str(k)] = str(v)

    return out


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
    # Backend not ready or process missing — record 5xx with reason "backend_unavailable"
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
    # All llama-server slots busy — record 5xx with reason "slot_exhaustion"
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


def _get_lease_timeout_seconds(srv) -> float:
    """Return the configured lease timeout in seconds (default 180)."""
    try:
        server_cfg = srv.config.get("server", {})
        return float(
            server_cfg.get("local_dispatch_lease_timeout_seconds", 180) or 180
        )
    except (ValueError, TypeError):
        return 180.0


async def _decrement_local_active_queries(
    srv,
    session_key: Optional[str] = None,
) -> None:
    """Safely decrement the local-only active queries counter.

    When *session_key* is provided, the corresponding dispatch record
    (if any) is marked as inactive with a future *expires_at* timestamp,
    keeping the lease alive for the owner session until the timeout.
    """
    try:
        async with srv.local_active_queries_lock:
            srv.local_active_queries = max(0, srv.local_active_queries - 1)
    except Exception:
        pass

    if session_key is not None:
        try:
            lock = getattr(srv, "local_dispatch_records_lock", None)
            if lock is not None:
                lease_timeout = _get_lease_timeout_seconds(srv)
                async with lock:
                    if session_key in srv.local_dispatch_records:
                        srv.local_dispatch_records[session_key]["active"] = False
                        srv.local_dispatch_records[session_key]["expires_at"] = (
                            time.monotonic() + lease_timeout
                        )
        except Exception:
            pass


async def _increment_local_active_queries(
    srv,
    session_key: Optional[str] = None,
    backend: Optional[str] = None,
) -> None:
    """Safely increment the local-only active queries counter.

    When *session_key* and *backend* are provided, a corresponding
    dispatch record is created in *local_dispatch_records* to track
    lease ownership.
    """
    try:
        async with srv.local_active_queries_lock:
            srv.local_active_queries += 1
    except Exception:
        pass

    if session_key is not None and backend is not None:
        try:
            lock = getattr(srv, "local_dispatch_records_lock", None)
            if lock is not None:
                lease_timeout = _get_lease_timeout_seconds(srv)
                async with lock:
                    srv.local_dispatch_records[session_key] = {
                        "backend": backend,
                        "started_at": time.monotonic(),
                        "active": True,
                        "expires_at": time.monotonic() + lease_timeout,
                    }
        except Exception:
            pass


async def _try_acquire_local_dispatch(
    srv,
    max_local: int,
    session_key: str,
    backend: str,
) -> tuple:
    """Try to acquire the local dispatch for *session_key*.

    Returns ``(acquired, owner, active_count, retry_after)`` where:

    - *acquired* is True if the local backend was acquired for the caller.
    - *owner* is the session ID that currently holds the lease (or None).
    - *active_count* is the current number of active local queries after
      acquisition (or 0 if denied).
    - *retry_after* is a suggested retry delay in seconds (minimum 1).

    The no-preemption policy means that a non-owner session cannot
    acquire the local backend while an unexpired lease exists, even if
    the owner has no active request in flight.

    If the server does not have *local_dispatch_records* or
    *local_dispatch_records_lock* attributes (legacy state), the function
    silently returns ``(True, None, 0, 1.0)`` to allow the request.
    """
    # Guard: skip if dispatch tracking is not initialised
    if not hasattr(srv, "local_dispatch_records") or not hasattr(srv, "local_dispatch_records_lock"):
        return (True, None, 0, 1.0)

    lease_timeout = _get_lease_timeout_seconds(srv)
    now = time.monotonic()

    try:
        async with srv.local_dispatch_records_lock:
            # Check for existing leases from other sessions
            for existing_key, record in list(srv.local_dispatch_records.items()):
                if existing_key == session_key:
                    # Same session -- check if previous lease expired
                    if not record.get("active") and record.get("expires_at", 0) <= now:
                        # Lease expired, remove old record and allow re-acquisition
                        del srv.local_dispatch_records[existing_key]
                        continue
                    # Same session with valid lease -- allow
                    continue

                # Different session -- check whether it blocks acquisition
                if record.get("active") or record.get("expires_at", 0) > now:
                    # Owner session has an active or unexpired lease
                    owner = existing_key
                    active_count = getattr(srv, "local_active_queries", 0)
                    retry_after = max(1.0, record.get("expires_at", now) - now)
                    return (False, owner, active_count, retry_after)

            # No blocking lease -- check concurrency cap
            async with srv.local_active_queries_lock:
                if srv.local_active_queries >= max_local:
                    # At concurrency limit -- find the owner of the active request
                    active_owner = None
                    for ek, er in srv.local_dispatch_records.items():
                        if er.get("active"):
                            active_owner = ek
                            break
                    return (
                        False,
                        active_owner,
                        srv.local_active_queries,
                        max(1.0, lease_timeout),
                    )

                # Acquire: increment counter and create dispatch record
                srv.local_active_queries += 1

            srv.local_dispatch_records[session_key] = {
                "backend": backend,
                "started_at": now,
                "active": True,
                "expires_at": now + lease_timeout,
            }

        return (True, None, getattr(srv, "local_active_queries", 0), max(1.0, lease_timeout))
    except Exception:
        return (True, None, 0, 1.0)


async def _cleanup_stale_local_dispatch(srv) -> int:
    """Remove stale lease records from *local_dispatch_records*.

    A record is stale when its *expires_at* timestamp has passed
    (``time.monotonic() > expires_at``), regardless of whether it is
    *active* or *inactive*.  This ensures abandoned/crashed requests
    (where *active* stays ``True`` permanently) are eventually cleaned.
    Each removed record is logged with its session ID and reason.

    Returns the number of records removed.
    """
    now = time.monotonic()
    removed = 0
    try:
        async with srv.local_dispatch_records_lock:
            stale_ids = [
                sid
                for sid, record in srv.local_dispatch_records.items()
                if record.get("expires_at", 0) <= now
            ]
            for sid in stale_ids:
                del srv.local_dispatch_records[sid]
                removed += 1
                try:
                    srv.logger.info(
                        "lease_released session=%s reason=idle_timeout",
                        sid[:8] if sid else "unknown",
                    )
                except Exception:
                    pass
    except Exception:
        pass
    return removed


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
        "session_id_header": None,
        "session_created": False,
        "is_delta_request": False,
        "session_fallback_reason": None,
        "delta_messages": None,
        "body_json": body_json,
        "body_override": None,
        "original_message_count": 0,
        "session_explicit": False,
    }
    
    session_id_header, session_header_source = _resolve_session_id_header(
        request_headers
    )
    result["session_id_header"] = session_id_header
    result["session_explicit"] = session_id_header is not None
    
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
) -> dict:
    """Estimate tokens sent in the request body, broken down by category.

    Returns a dict with keys ``user``, ``assistant``, ``tool``, ``system``
    mapping to integer token counts for each message role.

    Category mapping:
    - **User messages**: tokens in ``role: "user"`` message content
    - **Agent responses**: tokens in ``role: "assistant"`` message content
    - **Tool Calls**: tokens in ``role: "tool"`` message content +
      tool_use content blocks within assistant messages + ``tools`` array
      definitions
    - **System Prompt**: tokens in the ``system`` field or
      ``role: "system"`` message content

    Falls back to the ``user`` category for non-message formats
    (e.g. raw ``/v1/completions`` input).
    """
    from proxy.utils import count_text_tokens

    result = {"user": 0, "assistant": 0, "tool": 0, "system": 0}

    try:
        if isinstance(body_json, dict) and "messages" in body_json:
            messages = body_json.get("messages", [])
            # First pass: count content tokens per role
            for m in messages:
                role = m.get("role", "")
                content = str(m.get("content", ""))
                tokens = count_text_tokens(content, model_name)
                if role == "user":
                    result["user"] += tokens
                elif role == "assistant":
                    result["assistant"] += tokens
                elif role == "tool":
                    result["tool"] += tokens
                elif role == "system":
                    result["system"] += tokens
                else:
                    # Unknown role — attribute to user as fallback
                    result["user"] += tokens

            # Second pass: count tool_calls embedded in assistant messages
            for m in messages:
                if m.get("role") == "assistant":
                    tool_calls = m.get("tool_calls")
                    if isinstance(tool_calls, list):
                        for tc in tool_calls:
                            func = tc.get("function", {})
                            args_str = func.get("arguments", "")
                            if args_str:
                                result["tool"] += count_text_tokens(
                                    str(args_str), model_name
                                )
                            name_str = func.get("name", "")
                            if name_str:
                                result["tool"] += count_text_tokens(
                                    str(name_str), model_name
                                )

            # Count tools array definitions
            tools = body_json.get("tools")
            if tools:
                import json as _json
                result["tool"] += count_text_tokens(
                    _json.dumps(tools, separators=(",", ":"), ensure_ascii=False),
                    model_name,
                )

        elif isinstance(body_json, dict) and "input" in body_json:
            inp = body_json["input"]
            if isinstance(inp, list):
                for it in inp:
                    result["user"] += count_text_tokens(str(it), model_name)
            else:
                result["user"] += count_text_tokens(str(inp), model_name)
        else:
            result["user"] += count_text_tokens(
                body.decode("utf-8", errors="replace"), model_name
            )
    except Exception:
        result = {"user": 0, "assistant": 0, "tool": 0, "system": 0}
    return result


async def _schedule_token_increment(
    key: str, tokens: Any
) -> None:
    """Schedule a token increment in the running event loop.

    Accepts either:
    - A ``dict`` with per-category keys (``user``, ``assistant``, ``tool``,
      ``system``) for the new per-category breakdown.
    - An ``int`` for backward compatibility with existing callers and tests.

    When a dict is provided, both the category-prefixed keys
    (``sent:<category>:<key>``) and the flat total key (``sent:<key>``)
    are incremented.
    """
    from proxy.observability import _increment_tokens

    try:
        loop = asyncio.get_running_loop()
        if isinstance(tokens, dict):
            total = 0
            for category, count in tokens.items():
                if count > 0:
                    total += count
                    loop.create_task(
                        _increment_tokens("sent", f"{category}:{key}", count)
                    )
            if total > 0:
                loop.create_task(_increment_tokens("sent", key, total))
        else:
            # Legacy int path
            loop.create_task(_increment_tokens("sent", key, int(tokens)))
    except RuntimeError:
        if isinstance(tokens, dict):
            total = 0
            for category, count in tokens.items():
                if count > 0:
                    total += count
                    asyncio.run(
                        _increment_tokens("sent", f"{category}:{key}", count)
                    )
            if total > 0:
                asyncio.run(_increment_tokens("sent", key, total))
        else:
            asyncio.run(_increment_tokens("sent", key, int(tokens)))
    except Exception:
        pass


async def _schedule_recv_token_increment(
    key: str, tokens: int
) -> None:
    """Schedule a received token increment.

    Stores both the flat recv key (``recv:<key>``) for backward
    compatibility and the category-prefixed key
    (``recv:response:<key>``) for the per-category breakdown.
    """
    from proxy.observability import _increment_tokens

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_increment_tokens("recv", key, tokens))
        loop.create_task(
            _increment_tokens("recv", f"response:{key}", tokens)
        )
    except RuntimeError:
        asyncio.run(_increment_tokens("recv", key, tokens))
        asyncio.run(
            _increment_tokens("recv", f"response:{key}", tokens)
        )
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
        max_timeout = float(server_config.get("llama_request_timeout", 1800))
        timeout_seconds = _compute_adaptive_timeout(
            body_json, base_timeout, per_token_timeout, max_timeout
        )
        _srv().logger.debug(
            "Adaptive timeout: tokens=%d timeout=%.1fs",
            _estimate_prompt_tokens(body_json),
            timeout_seconds,
        )
    else:
        timeout_seconds = server_config.get("llama_request_timeout", 1800)
    return httpx.Timeout(timeout_seconds)


# ---------------------------------------------------------------------------
# Session traffic recording helpers (LP-0MR8FEKK6005V9ML)
# ---------------------------------------------------------------------------


def _schedule_traffic_recording(
    session_id: str,
    client_payload: Optional[Any] = None,
    proxy_payload: Optional[Any] = None,
    response_payload: Optional[Any] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> None:
    """Schedule fire-and-forget recording of session traffic.

    Records the client→proxy request, proxy→provider request, and
    provider→client response for a single proxied call. All writes
    are dispatched to the event loop as background tasks and do not
    block the caller.

    Args:
        session_id: The session identifier for the call being recorded.
        client_payload: The original client→proxy request payload.
        proxy_payload: The processed proxy→provider request payload.
        response_payload: The assembled provider→client response.
        model: Optional model name to include in recording metadata.
        provider: Optional provider name to include in recording metadata.
    """
    if not session_id:
        return

    try:
        from proxy.session_recorder import SessionRecorder

        loop = asyncio.get_running_loop()
        recorder = SessionRecorder.from_config(_srv().config)

        if client_payload is not None:
            loop.create_task(
                recorder.record_request(
                    session_id, "client_to_proxy", client_payload,
                    model=model, provider=provider,
                )
            )

        if proxy_payload is not None:
            loop.create_task(
                recorder.record_request(
                    session_id, "proxy_to_provider", proxy_payload,
                    model=model, provider=provider,
                )
            )

        if response_payload is not None:
            # Try to parse string payload as JSON for consistent format
            if isinstance(response_payload, str):
                try:
                    parsed = json.loads(response_payload)
                except (json.JSONDecodeError, ValueError):
                    parsed = response_payload
            elif isinstance(response_payload, bytes):
                try:
                    parsed = json.loads(response_payload.decode("utf-8", errors="replace"))
                except (json.JSONDecodeError, ValueError):
                    parsed = response_payload.decode("utf-8", errors="replace")
            else:
                parsed = response_payload

            loop.create_task(
                recorder.record_response(
                    session_id, "provider_to_client", parsed,
                    model=model, provider=provider,
                )
            )
    except Exception as exc:
        try:
            _srv().logger.warning(
                "Failed to schedule session recording for %s: %s",
                session_id[:8] if session_id else "unknown",
                exc,
            )
        except Exception:
            pass
