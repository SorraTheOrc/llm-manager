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
import os
import time
from collections.abc import Mapping
from typing import Any

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

def _get_request_preview(body_json: dict | bytes | None) -> str:
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
    session_id: str | None = None,
    slot_id: str | None = "none",
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
    session_id: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    body_json: dict | bytes | None = None,
    entry: str | None = None,
) -> None:
    """Log streaming response chunk.

    The ContentOnlyConsoleHandler no longer displays streaming content to the
    console (LP-0MR90HJED005WI1Z). Raw JSON is written to the log file only.

    Per-chunk ``STREAM CHUNK | ...`` lines are logged at DEBUG level by
    default so normal (INFO-level) operation does not write millions of
    chunk lines per day (LP-0MS9GAN2P002NR4M). When verbose chunk logging is
    enabled (config ``logging.verbose_chunks`` or ``LLAMA_PROXY_VERBOSE=1``),
    they are emitted at INFO level for debugging stream issues.

    If the chunk contains a ``finish_reason`` in any ``choices[]`` entry,
    an enhanced ``Stream finished: reason=<reason>`` log line is emitted
    so the stop reason (and optional token usage) appears in both console
    and file logs. When *session_id*, *model*, *provider*, and *entry* are
    provided, they are appended to the log line. *entry* carries the config
    entry name (e.g. ``opencode-go-2-deepseek``) so per-account traffic is
    attributable (LP-0MSC7F7BG0043TE1); it is omitted when absent.

    When *body_json* is provided, a request preview (first 80 characters of
    the first non-system user message) is included in the finished line.
    """
    srv = _srv()
    try:
        chunk_str = chunk.decode("utf-8")[:500] if chunk else ""
        if getattr(srv, "verbose_chunks", False):
            srv.logger.info(f"STREAM CHUNK | {chunk_str}")
        else:
            srv.logger.debug(f"STREAM CHUNK | {chunk_str}")
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
                if entry:
                    parts.append(f"entry={entry}")
                if body_json is not None:
                    preview = _get_request_preview(body_json)
                    if preview:
                        parts.append(f"request={preview}")
                srv.logger.info(" ".join(parts))

                # Log a WARNING when the upstream provider truncated the response
                # due to reaching its max_tokens (LP-0MS4C6E2L004HLLZ).
                if finish_reason == "length":
                    _parts = ["Response truncated: finish_reason=length"]
                    if session_id:
                        _parts.append(f"session={session_id}")
                    if model:
                        _parts.append(f"model={model}")
                    if entry:
                        _parts.append(f"entry={entry}")
                    if isinstance(usage, dict):
                        ct = usage.get("completion_tokens")
                        if ct is not None:
                            _parts.append(f"completion_tokens={ct}")
                    srv.logger.warning(" ".join(_parts))
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


def normalize_upstream_request_headers(headers: Mapping[str, str]) -> dict[str, str]:
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

    out: dict[str, str] = {}
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
    session_id: str | None,
    session_created: bool,
    is_delta_request: bool,
    session_fallback_reason: str | None,
    retry_after: int | None = None,
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


async def _increment_per_model_query(srv, model_name: str | None) -> None:
    """Increment per-model active query counter.

    When *model_name* is ``None`` or empty, this is a no-op.
    """
    if not model_name:
        return
    try:
        async with srv.per_model_queries_lock:
            srv.per_model_queries[model_name] = srv.per_model_queries.get(model_name, 0) + 1
    except AttributeError as exc:
        # Gracefully handle missing server attributes (e.g., in tests)
        import proxy.server as _s
        _s.logger.debug("per_model_query_lock not available: %s", exc)
    except Exception:
        pass


async def _decrement_per_model_query(srv, model_name: str | None) -> None:
    """Decrement per-model active query counter.

    When *model_name* is ``None`` or empty, this is a no-op.
    The counter is clamped to a minimum of 0.
    """
    if not model_name:
        return
    try:
        async with srv.per_model_queries_lock:
            current = srv.per_model_queries.get(model_name, 0)
            if current > 0:
                srv.per_model_queries[model_name] = current - 1
            else:
                srv.per_model_queries[model_name] = 0
    except Exception:
        pass


async def _get_per_model_queries(srv) -> dict[str, int]:
    """Return a snapshot of per-model query counts."""
    try:
        async with srv.per_model_queries_lock:
            return dict(srv.per_model_queries)
    except Exception:
        return {}


def _apply_queue_wait_to_timeout(
    request_timeout: httpx.Timeout,
    queue_wait_seconds: float,
) -> httpx.Timeout:
    """Reduce the adaptive timeout by the elapsed contention-queue wait.

    Q2=a (LP-0MSORQVK50012Q4D): the queued wait subtracts from the
    client-visible adaptive timeout budget, so total (wait + serve) stays
    within ``llama_adaptive_timeout_*``. Floors at 1s so a near-cap wait still
    leaves a minimal serve window.
    """
    try:
        base = float(request_timeout.connect)
    except Exception:
        return request_timeout
    reduced = max(1.0, base - max(0.0, float(queue_wait_seconds)))
    return httpx.Timeout(reduced)


def _get_lease_timeout_seconds(srv) -> float:
    """Return the configured lease timeout in seconds (default 180)."""
    try:
        server_cfg = srv.config.get("server", {})
        return float(
            server_cfg.get("local_dispatch_lease_timeout_seconds", 60) or 60
        )
    except Exception:
        return 60.0


def _get_chunk_refresh_buffer_seconds(srv) -> float:
    """Return the chunk-refresh safety buffer in seconds (default 30).

    Applied to non-explicit (anonymous) sessions: each data-chunk arrival
    on an active stream pushes ``expires_at`` out to ``now + buffer`` so a
    15s base lease cannot orphan-clean a stream mid-generation when gaps
    between chunks exceed the base (LP-0MSUO6HLX0089MNQ). Explicit sessions
    already refresh on chunks with the full lease timeout; anonymous
    sessions get this dedicated buffer so the refresh is generous without
    depending on the (short) base lease.
    """
    try:
        server_cfg = srv.config.get("server", {})
        return float(
            server_cfg.get(
                "local_dispatch_lease_chunk_refresh_buffer_seconds", 30
            )
            or 30
        )
    except Exception:
        return 30.0


def _get_adaptive_lease_timeout_seconds(
    srv,
    body_json: dict | None = None,
) -> float:
    """Return an adaptive lease timeout based on estimated prompt tokens.

    For large-context requests, the base lease timeout (default 180s) may
    be insufficient to cover the cache prefill phase where no data chunks
    arrive to refresh the lease. This function extends the lease timeout
    based on estimated prompt tokens using the same per-token multiplier
    as the request timeout computation.

    Formula:
        adaptive_timeout = min(base_timeout + tokens * per_token_secs, max_lease)

    When *body_json* is not provided or the adaptive timeout is not enabled,
    the base lease timeout is returned unchanged (LP-0MRDUQ9QC003LDDP).
    """
    base_timeout = _get_lease_timeout_seconds(srv)

    if body_json is None:
        return base_timeout

    try:
        from proxy.lifecycle import _estimate_prompt_tokens

        server_cfg = srv.config.get("server", {})
        per_token_seconds = float(
            server_cfg.get(
                "local_dispatch_lease_per_token_seconds",
                server_cfg.get("llama_adaptive_timeout_per_token_seconds", 0.015),
            )
        )
        max_lease = float(
            server_cfg.get("local_dispatch_lease_max_seconds", 1500)
        )
        estimated_tokens = _estimate_prompt_tokens(body_json)
        adaptive = base_timeout + (estimated_tokens * per_token_seconds)
        return min(adaptive, max_lease)
    except Exception:
        return base_timeout


def _get_prefill_lease_config(srv) -> tuple[float, float]:
    """Return the prefill-progress lease config ``(poll_seconds, buffer_seconds)``.

    - *poll_seconds* — cadence (seconds) at which the proxy polls llama-server
      for observed prefill progress during the prefill phase of an explicit-
      session request. ``0`` disables progress-based extension entirely.
    - *buffer_seconds* — safety buffer added to ``expires_at`` after each
      observed progress advance. Must comfortably exceed the dispatch
      cleanup-loop cadence (~10s) so a refresh cannot lose to cleanup.

    Defaults: (10, 30). (LP-0MSE05J53004C6EL)
    """
    try:
        server_cfg = srv.config.get("server", {})
        raw_poll = server_cfg.get("local_dispatch_lease_prefill_poll_seconds", 10)
        poll_seconds = float(raw_poll) if raw_poll is not None else 10.0
        raw_buffer = server_cfg.get("local_dispatch_lease_prefill_buffer_seconds", 30)
        buffer_seconds = float(raw_buffer) if raw_buffer is not None else 30.0
        return poll_seconds, buffer_seconds
    except Exception:
        return 10.0, 30.0


async def _query_prefill_progress(
    srv,
    llama_port: int,
    model_name: str | None = None,
    slot_id: int | None = None,
) -> tuple[int | None, bool]:
    """Observe llama-server prefill state: ``(progress, alive)``.

    Non-blocking: every query is wrapped in ``asyncio.wait_for`` using the
    same ``STATUS_QUERY_TIMEOUT`` (default 1.0s) pattern as the
    ``/llama/local/status`` endpoint, so the stream loop is never blocked
    waiting on llama-server.

    Progress sources, in preference order:

    1. Per-slot: ``/slots`` -> per-slot state from ``_query_slots_progress``
       (``n_past`` / ``n_prompt_tokens_processed`` when the build reports
       them, plus the ``is_processing`` liveness flag).
    2. Aggregate: ``query_llama_status()`` -> ``kv_cache_tokens`` (or
       ``n_past`` if present).

    Returns ``(progress, alive)``:

    - *progress* is the latest observed prefill progress (``None`` when no
      numeric progress can be observed).
    - *alive* is True when the slot is observed actively processing
      (``is_processing``), even when the llama.cpp build exposes no numeric
      progress fields (b8782 removed ``n_past``/``n_prompt_tokens_processed``
      from ``/slots``; LP-0MSUO5Z0K007HBSS). This lets the caller extend the
      lease on liveness rather than progress advance alone.

    When progress is unobservable AND the slot is not observed processing,
    a throttled warning is logged so silent query failures are visible in
    production (LP-0MSUO5Z0K007HBSS AC2).
    """
    timeout = float(os.environ.get("STATUS_QUERY_TIMEOUT", "1.0"))
    alive = False

    if slot_id is not None:
        try:
            from proxy.observability import _query_slots_progress

            states = await asyncio.wait_for(
                _query_slots_progress(
                    llama_port, timeout=timeout, model=model_name
                ),
                timeout=timeout + 0.5,
            )
            state = states.get(slot_id)
            if isinstance(state, dict):
                alive = bool(state.get("processing", False))
                value = state.get("progress")
                if isinstance(value, (int, float)) and value >= 0:
                    return int(value), alive
        except Exception:
            pass

    try:
        from proxy.observability import query_llama_status

        status = await asyncio.wait_for(query_llama_status(), timeout=timeout)
        for key in ("kv_cache_tokens", "n_past"):
            value = status.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return int(value), True
    except Exception:
        pass

    if not alive:
        _warn_prefill_progress_unobservable(srv)
    return None, alive


_last_prefill_progress_warn_ts = 0.0
"""Monotonic timestamp of the last 'prefill progress unobservable' warning.

Used to throttle the warning to one per ``_PREFILL_WARN_INTERVAL`` seconds so
a permanently-broken progress source cannot spam the log at the 10s poll
cadence (LP-0MSUO5Z0K007HBSS AC2).
"""

_PREFILL_WARN_INTERVAL = 60.0
"""Seconds between repeated 'prefill progress unobservable' warnings."""


def _warn_prefill_progress_unobservable(srv) -> None:
    """Throttled warning when neither progress source can observe prefill
    progress (per-slot query failed / returned nothing usable, and the
    aggregate status query yielded no token counts).

    The prefill poll runs every ``local_dispatch_lease_prefill_poll_seconds``
    (default 10s) per stream, so without throttling a persistent failure
    would emit a warning every poll. Warn at most once per
    ``_PREFILL_WARN_INTERVAL`` seconds (LP-0MSUO5Z0K007HBSS AC2).
    """
    global _last_prefill_progress_warn_ts
    now = time.monotonic()
    if now - _last_prefill_progress_warn_ts < _PREFILL_WARN_INTERVAL:
        return
    _last_prefill_progress_warn_ts = now
    try:
        srv.logger.warning(
            "prefill_progress_unobservable: no per-slot or aggregate prefill "
            "progress; lease extension relies on the adaptive acquisition "
            "estimate and liveness signal"
        )
    except Exception:
        pass


async def _extend_lease_during_prefill(
    srv,
    session_key: str,
    *,
    llama_port: int,
    model_name: str | None = None,
    slot_id: int | None = None,
    last_progress: int = 0,
) -> tuple[int, bool]:
    """Observe prefill state and extend the dispatch lease while advancing.

    During the prefill phase of an explicit-session request (dispatched, no
    stream data chunks yet), llama-server reports per-slot state that
    advances as the cache prefill progresses. While the slot is observed
    alive (``is_processing``) or the reported progress advances, the
    session's dispatch lease ``expires_at`` is pushed out to
    ``now + safety buffer`` so a very large prefill — beyond the adaptive
    token-estimate cap of 1500s — cannot lose its lease mid-prefill
    (LP-0MSE05J53004C6EL).

    Extension triggers:

    - **Progress advance** — observed numeric progress (per-slot
      ``n_past``/``n_prompt_tokens_processed`` or aggregate
      ``kv_cache_tokens``) is greater than *last_progress*.
    - **Liveness** — no numeric progress is reported by the llama.cpp build
      (b8782 removed the fields from ``/slots``) but the slot is observed
      actively processing (``is_processing``); the lease is extended on
      liveness so streams are never orphaned mid-prefill just because the
      build stopped reporting a counter (LP-0MSUO5Z0K007HBSS).

    Returns ``(last_progress, extended)``:

    - *last_progress* is the latest observed progress the caller should
      pass back on the next poll so extension stops when progress stalls.
    - *extended* is True when ``expires_at`` was pushed out.

    When progress is unobservable AND the slot is not alive, the lease is
    left untouched — it keeps the adaptive token-estimate value applied at
    acquisition rather than being dropped (fallback). When disabled
    (``local_dispatch_lease_prefill_poll_seconds: 0``) this is a no-op.
    """
    poll_seconds, buffer_seconds = _get_prefill_lease_config(srv)
    if poll_seconds <= 0 or buffer_seconds <= 0:
        return last_progress, False

    progress, alive = await _query_prefill_progress(
        srv, llama_port, model_name=model_name, slot_id=slot_id
    )
    advancing = progress is not None and progress > last_progress
    if not advancing and not alive:
        # Unobservable or stalled: no extension. Unobservable keeps the
        # adaptive estimate applied at acquisition (fallback).
        return last_progress, False

    extended = False
    try:
        lock = getattr(srv, "local_dispatch_records_lock", None)
        if lock is not None:
            async with lock:
                record = srv.local_dispatch_records.get(session_key)
                if record is not None and record.get("active"):
                    record["expires_at"] = time.monotonic() + buffer_seconds
                    extended = True
                    try:
                        if advancing:
                            srv.logger.info(
                                "lease_extended_during_prefill session=%s progress=%d buffer=%.0fs",
                                session_key[:8] if session_key else "unknown",
                                progress,
                                buffer_seconds,
                            )
                        else:
                            srv.logger.info(
                                "lease_extended_during_prefill session=%s liveness=1 buffer=%.0fs",
                                session_key[:8] if session_key else "unknown",
                                buffer_seconds,
                            )
                    except Exception:
                        pass
    except Exception:
        pass
    if advancing:
        return progress, extended
    return last_progress, extended


async def _decrement_local_active_queries(
    srv,
    session_key: str | None = None,
) -> None:
    """Safely decrement the local-only active queries counter.

    When *session_key* is provided, the corresponding dispatch record
    (if any) is marked as inactive with a future *expires_at* timestamp,
    keeping the lease alive for the owner session until the timeout.
    """
    try:
        async with srv.local_active_queries_lock:
            srv.local_active_queries = max(0, srv.local_active_queries - 1)
    except Exception as exc:
        session_hint = f" session={session_key[:8]}" if session_key else ""
        try:
            srv.logger.warning(
                "Failed to decrement local_active_queries: %s: %s%s",
                type(exc).__name__,
                exc,
                session_hint,
            )
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
                        try:
                            srv.logger.info(
                                "lease_renewed session=%s timeout=%.0fs",
                                session_key[:8] if session_key else "unknown",
                                lease_timeout,
                            )
                        except Exception as exc:
                            try:
                                srv.logger.warning(
                                    "Failed to log lease_renewed for session=%s: %s: %s",
                                    session_key[:8] if session_key else "unknown",
                                    type(exc).__name__,
                                    exc,
                                )
                            except Exception:
                                pass
        except Exception as exc:
            try:
                srv.logger.warning(
                    "Failed to mark dispatch record inactive for session=%s: %s: %s",
                    session_key[:8] if session_key else "unknown",
                    type(exc).__name__,
                    exc,
                )
            except Exception:
                pass

    # A local slot freed (stream end / request completion) — wake the
    # cross-session contention queue (LP-0MSORQVK50012Q4D AC2). Best-effort:
    # an idle queue is a no-op.
    try:
        from proxy.contention_queue import wake

        await wake(1)
    except Exception:
        pass


async def _increment_local_active_queries(
    srv,
    session_key: str | None = None,
    backend: str | None = None,
    body_json: dict | None = None,
    model_name: str | None = None,
) -> None:
    """Safely increment the local-only active queries counter.

    When *session_key* and *backend* are provided, a corresponding
    dispatch record is created in *local_dispatch_records* to track
    lease ownership. *model_name* is stored on the record so orphan
    cleanup can verify the session's slot against llama-server's
    ``/slots`` before freeing the lease (LP-0MSUO6XRP001MCB2).

    When *body_json* is provided, the lease timeout is extended
    adaptively based on the estimated prompt token count, so that
    large-prompt prefills (which produce no stream chunks to refresh
    the lease) hold their lease for the full prefill duration. This
    applies the adaptive lease to anonymous/non-explicit sessions, not
    just explicit ones (LP-0MSEHMMBK0062ZPI).
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
                lease_timeout = _get_adaptive_lease_timeout_seconds(srv, body_json)
                async with lock:
                    srv.local_dispatch_records[session_key] = {
                        "backend": backend,
                        "started_at": time.monotonic(),
                        "active": True,
                        "expires_at": time.monotonic() + lease_timeout,
                        "model_name": model_name,
                    }
        except Exception:
            pass


async def _try_acquire_local_dispatch(
    srv,
    max_local: int,
    session_key: str,
    backend: str,
    body_json: dict | None = None,
    model_name: str | None = None,
) -> tuple:
    """Try to acquire the local dispatch for *session_key*.

    Returns ``(acquired, owner, active_count, retry_after)`` where:

    - *acquired* is True if the local backend was acquired for the caller.
    - *owner* is the session ID that currently holds the lease (or None).
    - *active_count* is the current number of active local queries after
      acquisition (or 0 if denied).
    - *retry_after* is a suggested retry delay in seconds (minimum 1).

    N-aware dispatch: allows up to *max_local* concurrent sessions to hold
    dispatch leases. When fewer than *max_local* leases are held by *other*
    sessions, a new session can acquire. Same-session re-acquisition is
    always permitted.

    The no-preemption policy means that an inactive lease (post-request
    cooldown) reserves its slot for the owning session. Other sessions are
    blocked only when the total number of occupied slots (active + inactive
    unexpired leases from other sessions) reaches *max_local*.

    Expired lease records (inactive and past their *expires_at* threshold)
    are cleaned before the occupancy check, freeing their slots.

    Lock ordering: ``local_active_queries_lock`` is acquired before
    ``local_dispatch_records_lock`` so the records-based occupancy count
    and the ``local_active_queries`` counter check are atomic with respect
    to all counter mutators. This matches the lock order used by
    ``_increment_local_active_queries``/``_decrement_local_active_queries``
    and prevents a TOCTOU false 503 under concurrent anonymous-session
    increments (LP-0MS8ZM98R000M8AN).

    When *body_json* is provided, the lease timeout is extended adaptively
    based on the estimated prompt token count. This prevents mid-stream
    lease expiry during the cache prefill phase for large-context requests
    (LP-0MRDUQ9QC003LDDP).

    If the server does not have *local_dispatch_records* or
    *local_dispatch_records_lock* attributes (legacy state), the function
    silently returns ``(True, None, 0, 1.0)`` to allow the request.
    """
    # Guard: skip if dispatch tracking is not initialised
    if not hasattr(srv, "local_dispatch_records") or not hasattr(srv, "local_dispatch_records_lock"):
        return (True, None, 0, 1.0)

    lease_timeout = _get_adaptive_lease_timeout_seconds(srv, body_json)
    now = time.monotonic()

    # Lock ordering: acquire local_active_queries_lock BEFORE
    # local_dispatch_records_lock so the occupancy count (records) and the
    # active-counter check are atomic w.r.t. all counter mutators
    # (_increment_local_active_queries / _decrement_local_active_queries,
    # which themselves take the active lock first). This closes the TOCTOU
    # window in which an anonymous-session increment could land between the
    # records count and the counter check and falsely deny an explicit
    # session that had a free slot (LP-0MS8ZM98R000M8AN).
    try:
        async with srv.local_active_queries_lock:
            async with srv.local_dispatch_records_lock:
                # ... (cleaning, checking, acquiring logic)
                for existing_key, record in list(srv.local_dispatch_records.items()):
                    if not record.get("active") and record.get("expires_at", 0) <= now:
                        del srv.local_dispatch_records[existing_key]
                        try:
                            from proxy.session import _free_slot_assignment
                            _free_slot_assignment(existing_key)
                        except Exception:
                            pass

                own_record = srv.local_dispatch_records.get(session_key)
                own_has_lease = (
                    own_record is not None
                    and (
                        own_record.get("active")
                        or own_record.get("expires_at", 0) > now
                    )
                )

                if not own_has_lease:
                    occupied_by_others = 0
                    first_occupied_owner = None
                    for existing_key, record in srv.local_dispatch_records.items():
                        if existing_key == session_key:
                            continue
                        if record.get("active") or record.get("expires_at", 0) > now:
                            occupied_by_others += 1
                            if first_occupied_owner is None:
                                first_occupied_owner = existing_key

                    if occupied_by_others >= max_local:
                        active_count = getattr(srv, "local_active_queries", 0)
                        retry_after = max(1.0, lease_timeout)
                        return (False, first_occupied_owner, active_count, retry_after)

                if srv.local_active_queries >= max_local and not own_has_lease:
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

                srv.local_active_queries += 1

                srv.local_dispatch_records[session_key] = {
                    "backend": backend,
                    "started_at": now,
                    "active": True,
                    "expires_at": now + lease_timeout,
                    "model_name": model_name,
                }

            return (True, None, getattr(srv, "local_active_queries", 0), max(1.0, lease_timeout))
    except Exception:
        return (True, None, 0, 1.0)


def _client_identity_extra(request: Request | None) -> dict:
    """Build the client-identity ``extra`` dict for structured log events.

    Returns the resolved ``client_ip`` / ``client_ip_source`` / ``client_port``
    when a Request is in scope, or an empty dict when it is not (background
    cleanup paths degrade gracefully — identity omitted, event still logged)
    (LP-0MSKV3IEQ004ZV88).
    """
    if request is None:
        return {}
    try:
        from proxy.handlers import _resolve_client_ip, _resolve_client_port
        client_ip, client_ip_source = _resolve_client_ip(request)
        return {
            "client_ip": client_ip,
            "client_ip_source": client_ip_source,
            "client_port": _resolve_client_port(request),
        }
    except Exception:
        return {}


async def _release_local_dispatch(srv, session_id: str, request: Request | None = None) -> bool:
    """Explicitly release the dispatch lease for *session_id*.

    Removes the dispatch record from ``local_dispatch_records`` under
    the existing lock. Returns ``True`` if a record was removed, or
    ``False`` if no matching record existed (idempotent no-op).

    This is the programmatic equivalent of what the
    ``POST /v1/leases/release`` endpoint provides, allowing callers
    (e.g. other internal components) to proactively release a lease
    without going through the HTTP layer.
    """
    removed = False
    try:
        async with srv.local_dispatch_records_lock:
            if session_id in srv.local_dispatch_records:
                del srv.local_dispatch_records[session_id]
                removed = True
                try:
                    srv.logger.info(
                        "lease_released session=%s reason=explicit_release",
                        session_id[:8] if session_id else "unknown",
                        extra=_client_identity_extra(request),
                    )
                except Exception:
                    pass
    except Exception:
        raise
    # Free the slot registry entry
    if session_id:
        try:
            from proxy.session import _free_slot_assignment
            _free_slot_assignment(session_id)
        except Exception:
            pass
    # A slot-persistence / lease release frees the backend — wake the
    # cross-session contention queue (LP-0MSORQVK50012Q4D AC2).
    try:
        from proxy.contention_queue import wake

        await wake(1)
    except Exception:
        pass
    return removed


async def _query_slot_processing(srv, session_id: str, model_name: str | None) -> bool:
    """Check whether the session's llama-server slot is still processing.

    Queries llama-server ``/slots`` (via ``_query_slots_progress``) for the
    slot assigned to *session_id* and returns its ``is_processing`` flag —
    the only per-slot liveness signal llama.cpp b8782 exposes. Used by
    orphan cleanup to avoid freeing the dispatch lease / slot registry
    entry for a stream that is still generating on the backend
    (LP-0MSUO6XRP001MCB2).

    Returns False (do not treat as alive) when the session has no slot
    assignment, no *model_name* is known (the router requires a model
    param), or the query fails/times out — the caller then proceeds with
    normal orphan cleanup.
    """
    if not session_id or not model_name:
        return False
    try:
        from proxy.session import _assigned_slot_for_session

        slot_id = _assigned_slot_for_session(session_id)
        if slot_id is None:
            return False  # no slot assignment — cannot verify

        server_cfg = srv.config.get("server", {})
        llama_port = server_cfg.get("llama_server_port", 8080)

        from proxy.observability import _query_slots_progress

        timeout = float(os.environ.get("STATUS_QUERY_TIMEOUT", "1.0"))
        states = await asyncio.wait_for(
            _query_slots_progress(llama_port, timeout=timeout, model=model_name),
            timeout=timeout + 0.5,
        )
        state = states.get(slot_id)
        if isinstance(state, dict):
            return bool(state.get("processing", False))
        return False
    except Exception:
        return False


async def _cleanup_stale_local_dispatch(srv) -> int:
    """Remove stale lease records from *local_dispatch_records*.

    Two categories of stale records are cleaned:

    1. **Inactive records** whose *expires_at* has passed — these represent
       sessions that finished their request but whose idle lease timeout
       has expired. Logged at INFO level with ``reason=idle_timeout``.

    2. **Active records** whose *expires_at* has passed — these represent
       abandoned/orphaned streams where the stream was started but never
       finished (no *active=False* transition). Before freeing, the record's
       slot is verified against llama-server ``/slots``: if the slot is
       still processing, the lease is extended by the chunk-refresh buffer
       instead (``lease_verified_active ... stream_abandoned=False``), so a
       long silent generation cannot lose its lease mid-flight
       (LP-0MSUO6XRP001MCB2). Genuinely idle slots are orphan-cleaned as
       before, logged at WARNING level with ``reason=orphan_cleanup`` plus
       an INFO-level ``lease_released reason=orphan_cleanup`` for parity
       with existing log consumers.

    Active records whose *expires_at* is still in the future are preserved
    (legitimate in-flight requests).

    Returns the number of records removed.
    """
    now = time.monotonic()
    removed = 0
    verify_candidates: list[tuple[str, dict]] = []

    # Phase 1 (under lock): collect expired records. Inactive records are
    # freed immediately; expired ACTIVE records are deferred to phase 3 so
    # the slot liveness check runs OUTSIDE the records lock (it performs
    # an HTTP query to llama-server).
    try:
        async with srv.local_dispatch_records_lock:
            for sid, record in list(srv.local_dispatch_records.items()):
                expires_at = record.get("expires_at", 0)
                if expires_at > now:
                    continue  # still within valid window

                active = record.get("active", False)
                if not active:
                    # Normal idle timeout for inactive records
                    del srv.local_dispatch_records[sid]
                    removed += 1
                    # Free the slot registry entry so the slot can be
                    # reused by a new session (LP-0MSB0RP7F000U0WJ)
                    try:
                        from proxy.session import _free_slot_assignment
                        _free_slot_assignment(sid)
                    except Exception:
                        pass
                    try:
                        srv.logger.info(
                            "lease_released session=%s reason=idle_timeout",
                            sid[:8] if sid else "unknown",
                        )
                    except Exception:
                        pass
                else:
                    verify_candidates.append((sid, record))
    except Exception:
        pass

    # Phase 2 (outside the lock): verify which candidates still have a
    # processing slot on llama-server. A failed / unverifiable query means
    # "not verified alive" — the record is orphan-cleaned below (fail-open,
    # matching pre-existing behaviour).
    alive: set[str] = set()
    for sid, record in verify_candidates:
        try:
            if await _query_slot_processing(
                srv, sid, record.get("model_name")
            ):
                alive.add(sid)
        except Exception:
            pass

    # Phase 3 (under lock again): apply the verdict. Re-check the record
    # still exists and is still expired — it may have been refreshed by the
    # stream loop (chunk-refresh / prefill extension) or released since
    # phase 1.
    if verify_candidates:
        try:
            async with srv.local_dispatch_records_lock:
                for sid, record in verify_candidates:
                    current = srv.local_dispatch_records.get(sid)
                    if current is None:
                        continue  # released concurrently — nothing to do
                    if current.get("expires_at", 0) > time.monotonic():
                        continue  # refreshed since phase 1 — preserved
                    if sid in alive:
                        # Slot still generating: extend the lease instead of
                        # freeing (LP-0MSUO6XRP001MCB2).
                        current["expires_at"] = (
                            time.monotonic() + _get_chunk_refresh_buffer_seconds(srv)
                        )
                        try:
                            srv.logger.info(
                                "lease_verified_active session=%s "
                                "reason=active_slot stream_abandoned=False",
                                sid[:8] if sid else "unknown",
                            )
                        except Exception:
                            pass
                        continue
                    # Genuinely orphaned active record past its expires_at
                    del srv.local_dispatch_records[sid]
                    removed += 1
                    # Free the slot registry entry so the slot can be
                    # reused by a new session (LP-0MSB0RP7F000U0WJ)
                    try:
                        from proxy.session import _free_slot_assignment
                        _free_slot_assignment(sid)
                    except Exception:
                        pass
                    # Decrement local_active_queries for orphaned records
                    # that never completed through the normal request path
                    # (LP-0MRKVN93I000XXXX: cleanup orphaned active queries)
                    try:
                        srv.local_active_queries = max(
                            0, int(getattr(srv, 'local_active_queries', 0) or 0) - 1
                        )
                    except Exception:
                        pass
                    try:
                        srv.logger.warning(
                            "lease_released session=%s reason=orphan_cleanup "
                            "stream_abandoned=True",
                            sid[:8] if sid else "unknown",
                        )
                        srv.logger.info(
                            "lease_released session=%s reason=orphan_cleanup",
                            sid[:8] if sid else "unknown",
                        )
                    except Exception:
                        pass
        except Exception:
            pass
    # Removed stale leases frees slots — wake the cross-session contention
    # queue (LP-0MSORQVK50012Q4D AC2).
    if removed > 0:
        try:
            from proxy.contention_queue import wake

            await wake(removed)
        except Exception:
            pass
    return removed


async def _recover_stuck_local_active_queries(srv) -> None:
    """Detect and reset a stuck ``local_active_queries`` counter.

    If ``local_active_queries > 0`` and **no** active dispatch records
    exist, the counter is likely stuck (e.g., from a swallowed exception
    in ``_decrement_local_active_queries``). Resets to 0 and logs a
    WARNING-level message.

    Designed to be called from ``_dispatch_cleanup_loop`` (server.py)
    after ``_cleanup_stale_local_dispatch``, providing a periodic
    self-recovery mechanism that runs every 10 seconds.

    In legacy mode (no ``local_dispatch_records`` attribute), the counter
    is reset whenever it is > 0, since there is no dispatch-record-based
    way to distinguish a stuck counter from legitimate in-flight requests.
    """
    try:
        records = getattr(srv, "local_dispatch_records", None)
        if records is not None:
            async with srv.local_active_queries_lock:
                if srv.local_active_queries > 0:
                    # Read the records snapshot (not holding records_lock).
                    # This is a self-correcting check that runs every 10s,
                    # so a slightly stale snapshot is acceptable.
                    has_active = any(
                        r.get("active", False) for r in records.values()
                    )
                    if not has_active:
                        prev = srv.local_active_queries
                        srv.local_active_queries = 0
                        try:
                            srv.logger.warning(
                                "local_active_queries counter recovered: "
                                "reset from %d to 0 (no active dispatch "
                                "records)",
                                prev,
                            )
                        except Exception:
                            pass
                        try:
                            from proxy.contention_queue import wake_all

                            await wake_all()
                        except Exception:
                            pass
        else:
            # Legacy mode: no dispatch records system
            async with srv.local_active_queries_lock:
                if srv.local_active_queries > 0:
                    prev = srv.local_active_queries
                    srv.local_active_queries = 0
                    try:
                        srv.logger.warning(
                            "local_active_queries counter recovered: "
                            "reset from %d to 0 (legacy mode, no "
                            "dispatch records)",
                            prev,
                        )
                    except Exception:
                        pass
                    try:
                        from proxy.contention_queue import wake_all

                        await wake_all()
                    except Exception:
                        pass
    except Exception:
        pass


async def _recover_stuck_global_active_queries(srv) -> None:
    """Detect and reset a stuck ``active_queries`` counter.

    If ``active_queries > 0`` but there is no evidence of in-flight
    work — ``local_active_queries == 0`` and no active dispatch
    records — the global counter is likely stuck (e.g., an abandoned
    stream whose decrement path never ran). Resets to 0 and logs a
    WARNING-level message.

    Mirrors ``_recover_stuck_local_active_queries`` and is designed to
    be called from ``_dispatch_cleanup_loop`` (server.py) right after
    the local recovery, providing a periodic in-process self-recovery
    mechanism (no proxy restart required).

    Rationale for the no-active-work check: the global counter is only
    incremented on the main routing path (router.py), which always
    increments ``local_active_queries`` as well — either via
    ``_try_acquire_local_dispatch`` for explicit sessions or via
    ``_increment_local_active_queries`` for anonymous ones. Remote
    concurrency-limit fallback never touches the global counter.
    Therefore a positive ``active_queries`` with no local activity
    means the counter leaked and can be safely reset.

    In legacy mode (no ``local_dispatch_records`` attribute) the only
    remaining signal is the local counter: if it is 0 and the global
    counter is positive, the global counter leaked.
    """
    try:
        records = getattr(srv, "local_dispatch_records", None)
        if records is not None:
            async with srv.local_active_queries_lock:
                has_active = any(
                    r.get("active", False) for r in records.values()
                )
            if has_active:
                # Legitimate in-flight local work — keep the counter.
                return

        async with srv.local_active_queries_lock:
            local_active = int(getattr(srv, "local_active_queries", 0) or 0)
        if local_active > 0:
            # Local requests in flight — keep the global counter.
            return

        async with srv.active_queries_lock:
            if srv.active_queries > 0:
                prev = srv.active_queries
                srv.active_queries = 0
                try:
                    srv.logger.warning(
                        "active_queries counter recovered: "
                        "reset from %d to 0 (no active dispatch "
                        "records or local queries in flight)",
                        prev,
                    )
                except Exception:
                    pass
    except Exception:
        pass


# ===================================================================
# Header normalization helpers
# ===================================================================

# Upstream response headers that MUST NOT be forwarded to downstream clients.
# These are either set automatically by uvicorn/Starlette (date, server),
# are hop-by-hop (connection), or are incorrect after httpx auto-decompression
# (content-encoding). Forwarding them causes duplicate headers, violated RFCs,
# or downstream decompression failures.
_OUTGOING_STRIPPED_HEADERS = {
    "content-encoding",
    "date",
    "server",
    "connection",
}


def _normalize_outgoing_headers(
    headers: dict, buffered: bool = False
) -> dict:
    """Normalize outgoing response headers.

    Strips upstream headers that should not be forwarded to the downstream
    client:

    - ``content-encoding`` — httpx auto-decompresses upstream bodies;
      forwarding the encoding header causes downstream clients to attempt
      double-decompression and fail.
    - ``date``, ``server`` — uvicorn/Starlette set these automatically;
      forwarding them creates duplicate RFC-violating headers.
    - ``connection`` — hop-by-hop header managed by the HTTP stack.
    - ``content-length`` — removed when ``transfer-encoding`` is present
      (for streaming).
    """
    result = dict(headers)

    # Always strip headers that uvicorn/Starlette set or that are incorrect
    # after httpx processing
    for k in list(result.keys()):
        if k.lower() in _OUTGOING_STRIPPED_HEADERS:
            del result[k]

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
        _build_slot_context,
        _classify_delta_routing,
        _invalidate_session_and_slot,
        _log_session_header_resolution,
        _record_delta_payload_bytes,
        _record_restore_fallback,
        _resolve_session_id_header,
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
    body: bytes, body_json: dict, model_name: str | None
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
    slot_model_name: str | None,
    model_name: str | None,
    path: str,
) -> JSONResponse | None:
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
    remote: bool = False,
) -> httpx.Timeout:
    """Compute the request timeout, using adaptive timeout if enabled.

    Parameters
    ----------
    server_config : dict
        The server configuration dictionary.
    body_json : dict
        The parsed request body, used to estimate prompt tokens.
    remote : bool, optional
        When True, use remote-specific timeout override keys
        (``llama_remote_request_timeout_base_seconds`` and
        ``llama_remote_request_timeout_per_token_seconds``) if
        configured. Falls back to the local keys when remote-specific
        keys are not set.  Defaults to False (local path).

    Returns
    -------
    httpx.Timeout
        The computed timeout value.
    """
    from proxy.lifecycle import _compute_adaptive_timeout, _estimate_prompt_tokens

    adaptive_enabled = server_config.get("llama_adaptive_timeout_enabled", False)
    if adaptive_enabled and body_json:
        # Resolve base and per-token timeout: use remote-specific keys when
        # *remote=True* and they are explicitly configured; otherwise fall
        # back to the local/default keys for backward compatibility.
        if remote:
            base_timeout = float(
                server_config.get(
                    "llama_remote_request_timeout_base_seconds",
                    server_config.get("llama_adaptive_timeout_base_seconds", 60),
                )
            )
            per_token_timeout = float(
                server_config.get(
                    "llama_remote_request_timeout_per_token_seconds",
                    server_config.get("llama_adaptive_timeout_per_token_seconds", 0.01),
                )
            )
        else:
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
            "Adaptive timeout: tokens=%d timeout=%.1fs%s",
            _estimate_prompt_tokens(body_json),
            timeout_seconds,
            " remote=True" if remote else "",
        )
    else:
        timeout_seconds = server_config.get("llama_request_timeout", 1800)
    return httpx.Timeout(timeout_seconds)


# ---------------------------------------------------------------------------
# Session traffic recording helpers (LP-0MR8FEKK6005V9ML)
# ---------------------------------------------------------------------------


def _schedule_traffic_recording(
    session_id: str,
    client_payload: Any | None = None,
    proxy_payload: Any | None = None,
    response_payload: Any | None = None,
    model: str | None = None,
    provider: str | None = None,
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
