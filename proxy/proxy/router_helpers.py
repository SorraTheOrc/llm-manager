
# <!-- REFACTOR-LP-0MTRWI65I005VXOC
# smell: formatting
# severity: high
# description: Do not assign a `lambda` expression, use a `def`
# -->
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
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

# Per-turn correlation id stamped on every structured line these helpers emit
# (LP-0MVMBM467005BT4A). Read from the request's bound context so the shared
# helpers keep their existing signatures.
from proxy.turn_id import bind_turn_id, get_turn_id, turn_field


# ---------------------------------------------------------------------------
# Lazy server import — avoids circular imports when server.py imports us
# ---------------------------------------------------------------------------
def _srv():
    import proxy.server as _m
    return _m


# ===================================================================
# Per-endpoint dispatch-lease key helpers (LP-0MRPILSMW004T4H8)
# ===================================================================

def _dispatch_lease_key(endpoint: str | None, session_id: str) -> str | tuple:
    """Build the dispatch-lease record key.

    Records are keyed by ``(endpoint, session_id)`` so each local
    llama-server instance (identified by its endpoint URL) tracks its
    concurrency independently (LP-0MRPILSMW004T4H8). When *endpoint* is
    None or empty (legacy single-server mode / tests), the bare
    *session_id* string is used for backward compatibility. The literal
    legacy label ``"local"`` is also treated as no-endpoint so existing
    callers that pass ``backend="local"`` keep using plain session-id keys.
    """
    if not endpoint or endpoint == "local":
        return session_id
    return (endpoint, session_id)


def _dispatch_key_session_id(key) -> str:
    """Return the session-id component of a dispatch-lease key.

    Accepts both legacy string keys (returned unchanged) and per-endpoint
    ``(endpoint, session_id)`` tuple keys.
    """
    if isinstance(key, tuple) and len(key) == 2:
        return key[1]
    return key


def _parse_endpoint_url(url: str) -> tuple[str, int]:
    """Parse an endpoint URL into ``(host, port)``.

    Returns ``(host, port)`` for use in slot-path derivation.
    """
    m = re.match(r'https?://([^/:]+):?(\d*)', url)
    if m:
        host = m.group(1)
        port = int(m.group(2) or 8080)
        return host, port
    return "localhost", 8080


def _endpoint_from_record(record: dict) -> str | None:
    """Extract the endpoint URL from a dispatch record's 'backend' field.

    The backend field stores the endpoint URL (or "local" for legacy
    single-server mode). Returns None when no endpoint URL is found.
    """
    backend = record.get("backend")
    if backend and backend.startswith("http"):
        return backend
    return None


def _request_dedup_hash(body_json: dict | None) -> str | None:
    """Return a stable hash of a request's message list for retry dedup.

    Used by the session single-flight coordinator to detect a client retry
    of the *same* prompt while the original is still in flight
    (LP-0MUCEFCV5006UKSZ). Only the ``messages`` list is hashed so retries
    that differ only in sampling parameters (which clients may vary) still
    collapse onto the original prefill. Returns None when there is no
    usable message list (caller then skips duplicate detection).
    """
    try:
        if not isinstance(body_json, dict):
            return None
        messages = body_json.get("messages")
        if not isinstance(messages, list):
            return None
        canonical = json.dumps(
            messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        import hashlib

        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except Exception:
        return None


# ===================================================================
# Request/response logging helpers
# ===================================================================

def _first_message_content(body_json: dict | bytes | None) -> str:
    """Return the full content of the first non-system message.

    This is the shared first-message extraction shared by the log body
    preview (:func:`_get_request_preview`) and the structured prompt log
    (:mod:`proxy.prompt_log`), so both apply the same redaction policy: the
    first message whose ``role`` is not ``"system"``, never system-prompt
    content.

    Parameters
    ----------
    body_json : dict or bytes or None
        The request body as a parsed dict, raw bytes, or None.

    Returns
    -------
    str
        The message content (uncapped), or ``""`` when the body cannot be
        parsed, contains no messages, or contains only system messages.
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
            return str(content)
    except Exception:
        pass
    return ""


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
    content_str = _first_message_content(body_json)
    if len(content_str) > 80:
        return content_str[:80] + "..."
    return content_str


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
    - The per-turn correlation id (``turn=``) is appended last when a turn is
      bound; a caller that has not bound one gets one generated here so the
      dispatch line is never unattributed (LP-0MVMBM467005BT4A AC1).
    """
    srv = _srv()
    try:
        method = request.method
        url = str(request.url)
        body_preview = _strip_system_messages_from_preview(body)
        # Bind (or keep) the request's per-turn id: the line below always
        # carries ``turn=``, even when the caller dispatched straight into
        # this helper (LP-0MVMBM467005BT4A AC1).
        bind_turn_id(request)
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
                f"body={body_preview} {session_info}{turn_field(request)}"
            )
        else:
            srv.logger.info(
                f"[{source}] {method} {url} "
                f"body={body_preview} {session_info}{turn_field(request)}"
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


# ---------------------------------------------------------------------------
# Deferred ``Stream finished`` emission (LP-0MVMHNVKT003PRIK)
# ---------------------------------------------------------------------------
#
# llama.cpp (and any OpenAI-compatible upstream honouring
# ``stream_options.include_usage``) reports the usage block in a *separate*
# final chunk that arrives AFTER the ``finish_reason`` chunk:
#
#   data: {"choices":[{"finish_reason":"stop", ...}]}
#   data: {"choices":[], "usage":{"prompt_tokens":N, ...}}
#   data: [DONE]
#
# The ``Stream finished`` line is emitted from the finish_reason chunk, so the
# measured token counts were never on it — local streams never carried
# ``tokens=`` and the usage-driven cache-credit path was dead for local.  When
# the caller knows a usage chunk is expected (``expect_usage=True``), the
# finish parts are deferred and emitted — with ``tokens=p/c/t`` — as soon as
# the usage chunk arrives.  A stream that ends without a usage chunk (aborted,
# old upstream) is flushed at ``[DONE]`` or by
# :func:`flush_pending_stream_finish`, marked ``tokens=missing`` so "no data"
# is distinguishable from "zero tokens" (LP-0MVMBM6N400541EH).

#: Marker appended to a terminal ``Stream finished`` line whose upstream
#: reported no usage (LP-0MVMBM6N400541EH AC1). It sits exactly where
#: ``tokens=p/c/t`` sits, so every consumer that matches
#: ``tokens=(\d+)/(\d+)/(\d+)`` simply does not match it — the "no data"
#: case is explicitly visible in the line and never mistaken for a count.
TOKENS_MISSING = "tokens=missing"

#: Upper bound on deferred per-stream finish parts (aborted streams are
#: flushed explicitly; the cap only guards against a leak if a caller never
#: flushes).
_PENDING_STREAM_FINISH_LIMIT = 64

#: Deferred ``Stream finished`` parts keyed by stream identity.
_pending_stream_finish: dict[tuple[str, str, str, str, str], list[str]] = {}


def _stream_finish_key(
    session_id: str | None,
    provider: str | None,
    model: str | None,
    entry: str | None,
) -> tuple[str, str, str, str, str]:
    """Identity of one stream for the deferred-finish buffer.

    The per-turn correlation id (LP-0MVMBM467005BT4A) is part of the identity:
    two interleaved turns of one session that share provider/model/entry must
    not be able to complete each other's deferred ``Stream finished`` line
    (an empty component when no turn is bound keeps the pre-change behaviour
    for direct helper calls and legacy logs).
    """
    return (
        session_id or "",
        provider or "",
        model or "",
        entry or "",
        get_turn_id() or "",
    )


def _usage_token_field(usage: Any) -> str | None:
    """Render a ``usage`` dict as ``tokens=prompt/completion/total``.

    Returns ``None`` when *usage* is absent or carries no token counts; the
    caller then emits :data:`TOKENS_MISSING` instead, so a line that genuinely
    has no usage says so rather than omitting the field
    (LP-0MVMBM6N400541EH AC1).
    """
    if not isinstance(usage, dict):
        return None
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    tt = usage.get("total_tokens")
    if pt is None and ct is None and tt is None:
        return None
    return f"tokens={pt or 0}/{ct or 0}/{tt or 0}"


def _defer_stream_finish(
    key: tuple[str, str, str, str, str], parts: list[str]
) -> None:
    """Buffer *parts* until the stream's usage chunk arrives."""
    if len(_pending_stream_finish) >= _PENDING_STREAM_FINISH_LIMIT:
        try:
            _pending_stream_finish.pop(next(iter(_pending_stream_finish)))
        except (StopIteration, KeyError):
            pass
    _pending_stream_finish[key] = parts


def _flush_deferred_stream_finish(
    key: tuple[str, str, str, str, str], token_field: str | None
) -> bool:
    """Emit a buffered ``Stream finished`` line, optionally with usage.

    The token field (or :data:`TOKENS_MISSING` when the stream never delivered
    usage) is inserted directly after the ``reason=`` field so the line keeps
    the historical ``Stream finished: reason=<r> tokens=p/c/t`` shape.
    Returns True when a deferred line was emitted.
    """
    parts = _pending_stream_finish.pop(key, None)
    if parts is None:
        return False
    parts.insert(1, token_field or TOKENS_MISSING)
    try:
        _srv().logger.info(" ".join(parts))
    except Exception:
        pass
    return True


def flush_pending_stream_finish(
    session_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    entry: str | None = None,
) -> bool:
    """Flush a deferred ``Stream finished`` line that never saw usage.

    Called at end-of-stream by the local streaming loop so an aborted or
    usage-less stream still logs its terminal line, marked ``tokens=missing``.
    """
    return _flush_deferred_stream_finish(
        _stream_finish_key(session_id, provider, model, entry), None
    )


def is_usage_only_chunk(chunk: bytes) -> bool:
    """Whether *chunk* carries only an OpenAI ``usage`` block (no choices).

    Used by the local streaming loop to keep the client-visible SSE stream
    byte-compatible when the proxy requested ``include_usage`` on the client's
    behalf: the measurement chunk is logged but not forwarded.
    """
    if not chunk:
        return False
    saw_usage = False
    try:
        text = chunk.decode("utf-8", errors="replace")
    except Exception:
        return False
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            return False
        if not isinstance(obj, dict):
            return False
        if obj.get("choices"):
            return False
        if isinstance(obj.get("usage"), dict):
            saw_usage = True
    return saw_usage


def log_non_stream_usage(
    content: bytes | str | None,
    session_id: str | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> None:
    """Emit the terminal ``Stream finished`` line for a non-streamed reply.

    Non-streaming local responses carry llama-server's ``usage`` block in the
    JSON body, so the measured token counts are logged here (AC1's
    non-streaming path) with ``stream=false`` marking it as a single buffered
    response rather than a chunked stream. A completion-shaped body that
    carries no reportable usage is marked ``tokens=missing`` rather than being
    dropped silently (LP-0MVMBM6N400541EH AC1); a body that is not a
    completion at all logs nothing.
    """
    if not content:
        return
    try:
        text = (
            content.decode("utf-8", errors="replace")
            if isinstance(content, (bytes, bytearray))
            else str(content)
        )
        body = json.loads(text) if text else None
    except Exception:
        return
    if not isinstance(body, dict):
        return
    choices = body.get("choices")
    token_field = _usage_token_field(body.get("usage"))
    if token_field is None and not (isinstance(choices, list) and choices):
        # Not a completion reply (e.g. an error body): there is no terminal
        # stream line to annotate.
        return
    finish_reason = "stop"
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish_reason = choices[0].get("finish_reason") or finish_reason
    parts = [
        f"Stream finished: reason={finish_reason}",
        token_field or TOKENS_MISSING,
        "stream=false",
    ]
    if session_id:
        parts.append(f"session={session_id}")
    if provider:
        parts.append(f"provider={provider}")
    if model:
        parts.append(f"model={model}")
    # Per-turn correlation id (LP-0MVMBM467005BT4A AC1): the buffered reply's
    # terminal line joins the same turn as its dispatch lines.
    turn = turn_field()
    if turn:
        parts.append(turn.strip())
    try:
        _srv().logger.info(" ".join(parts))
    except Exception:
        pass


def log_response_chunk(
    chunk: bytes,
    session_id: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    body_json: dict | bytes | None = None,
    entry: str | None = None,
    expect_usage: bool = False,
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

    For ``finish_reason: error`` events carrying an enriched ``error``
    object (LP-0MSETOTWY000SU0Z), the line additionally carries
    ``error_type``, ``error_message``, and ``suggested_action`` so the
    proxy-usage analysis tool can classify the informative-error fallback
    (LP-0MT60S55M000TK1H).

    When *body_json* is provided, a request preview (first 80 characters of
    the first non-system user message) is included in the finished line.

    *expect_usage* declares that the upstream was asked for an
    OpenAI ``usage`` block (``stream_options.include_usage``), which arrives
    in a chunk of its own after the ``finish_reason`` chunk
    (LP-0MVMHNVKT003PRIK). The ``Stream finished`` line is then deferred by
    one chunk so it can carry the measured ``tokens=prompt/completion/total``;
    a stream that never delivers usage is flushed unchanged on ``[DONE]`` or
    by :func:`flush_pending_stream_finish`.
    """
    srv = _srv()
    stream_key = _stream_finish_key(session_id, provider, model, entry)
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
                if expect_usage:
                    # No usage chunk arrived (aborted / usage-less upstream):
                    # emit the deferred terminal line without token counts.
                    _flush_deferred_stream_finish(stream_key, None)
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
            usage = j.get("usage")
            token_field = _usage_token_field(usage)
            if finish_reason is None:
                # A usage-only chunk following the finish chunk completes the
                # deferred ``Stream finished`` line with measured tokens
                # (LP-0MVMHNVKT003PRIK AC1).
                if token_field is not None and expect_usage:
                    _flush_deferred_stream_finish(stream_key, token_field)
                continue
            parts = [f"Stream finished: reason={finish_reason}"]
            if token_field is not None:
                parts.append(token_field)
            elif not expect_usage:
                # No usage reported and none was requested: say so explicitly
                # rather than omitting the field (LP-0MVMBM6N400541EH AC1).
                # With ``expect_usage`` the line is deferred instead and gets
                # the marker only if the usage chunk never arrives.
                parts.append(TOKENS_MISSING)
            # LP-0MT60S55M000TK1H: when the error is a synthetic stream-error
            # event (LP-0MSETOTWY000SU0Z) carry the enriched error payload so
            # the proxy-usage analysis tool can classify it (previously the
            # log line reported "no error payload").
            if finish_reason == "error":
                for choice in j.get("choices", []):
                    if isinstance(choice, dict):
                        err = choice.get("error")
                        if isinstance(err, dict):
                            err_type = err.get("type")
                            err_msg = err.get("message")
                            err_action = err.get("suggested_action")
                            if err_type:
                                parts.append(f"error_type={err_type}")
                            if err_msg:
                                parts.append(f"error_message={err_msg}")
                            if err_action:
                                parts.append(f"suggested_action={err_action}")
                            break  # only use the first choice's error
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
            # Per-turn correlation id (LP-0MVMBM467005BT4A AC1): appended last
            # so every existing field (and every end-anchored consumer regex)
            # is unchanged. Absent when no turn is bound (legacy shape).
            turn = turn_field()
            if turn:
                parts.append(turn.strip())
            if expect_usage and token_field is None:
                # The usage block arrives in the NEXT chunk: defer so the
                # line can carry the measured counts (LP-0MVMHNVKT003PRIK).
                _defer_stream_finish(stream_key, parts)
            else:
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


def _get_no_progress_timeout_seconds(srv) -> float:
    """Return the no-progress watchdog timeout in seconds (default 180).

    When a dispatch record has ``active=True`` but has had no observed
    progress advance or first-byte arrival for this duration, the watchdog
    treats it as a no-progress wedge and releases the lease (LP-0MUCEFB8E003YVFF).
    Returns 0 to disable the watchdog entirely.
    """
    try:
        server_cfg = srv.config.get("server", {})
        raw = server_cfg.get("local_dispatch_no_progress_timeout_seconds", 180)
        return float(raw) if raw is not None else 180.0
    except Exception:
        return 180.0


def _get_max_prefill_seconds(srv) -> float:
    """Return the max prefill lifetime in seconds (default 900).

    A hard ceiling on how long any single dispatch record may remain active
    from its ``started_at`` timestamp. Exceeding this ceiling always releases
    the lease regardless of observed progress (LP-0MUCEFB8E003YVFF).
    Returns 0 to disable the ceiling entirely.
    """
    try:
        server_cfg = srv.config.get("server", {})
        raw = server_cfg.get("local_dispatch_max_prefill_seconds", 900)
        return float(raw) if raw is not None else 900.0
    except Exception:
        return 900.0


def _get_first_byte_timeout_seconds(srv) -> float:
    """Return the initial first-byte wait ceiling in seconds (default 840).

    Bounds how long a local dispatch may wait for its first data chunk before
    the proxy aborts server-side with a retryable 503 (LP-0MUCEFB8E003YVFF).
    The default is deliberately STRICTLY BELOW the client's 900-second
    queue-timeout deadline, so the proxy always responds before the client
    gives up — the legacy default of ``local_dispatch_max_prefill_seconds``
    (900) raced the client exactly (``dispatch_first_byte_timeout ...
    ceiling=900s``, LP-0MU1S617L002Q9PR AC1). The lifetime wedge bound
    (``local_dispatch_max_prefill_seconds``) is unchanged. Returns 0 to
    disable the first-byte ceiling entirely.

    The effective ceiling is the tighter of this value and
    :func:`_get_prefill_deadline_seconds` — the prefill deadline
    (LP-0MVIQ0G2M0006QF6) is a *safety detour* deliberately set well below
    the wedge bound so a stalled local prefill becomes a fast remote answer
    instead of a multi-minute hang.
    """
    try:
        server_cfg = srv.config.get("server", {})
        raw = server_cfg.get("local_dispatch_first_byte_timeout_seconds", 840)
        ceiling = float(raw) if raw is not None else 840.0
    except Exception:
        ceiling = 840.0
    deadline = _get_prefill_deadline_seconds(srv)
    if deadline > 0:
        # Tighten toward the deadline; a zero wedge bound means "no ceiling".
        return deadline if ceiling <= 0 else min(ceiling, deadline)
    return ceiling


def _get_prefill_deadline_seconds(srv) -> float:
    """Return the prefill deadline in seconds (default 90; 0 disables).

    A hard backstop *independent of ETA-estimator quality*
    (LP-0MVIQ0G2M0006QF6 AC3): if a local dispatch has produced no first byte
    by this deadline, the pending upstream request is cancelled and the local
    dispatch aborts with the existing retryable 503, so the chain falls
    through to the next provider instead of leaving the client on a silent
    socket for minutes (baseline: 48 local dispatches >120 s, max 592 s).

    It is deliberately set well ABOVE the admission budget
    (``contention_queue_max_wait_seconds``, 60 s) so a request legitimately
    admitted inside its budget is never cut off, and strictly BELOW
    ``local_dispatch_first_byte_timeout_seconds`` (840 s) so it is the bound
    that actually fires. Precedence: the prefill deadline wins over the wedge
    bound; both are bounded by ``local_dispatch_max_prefill_seconds``.
    """
    try:
        server_cfg = srv.config.get("server", {})
        raw = server_cfg.get("local_prefill_deadline_seconds", 90)
        return float(raw) if raw is not None else 90.0
    except Exception:
        return 90.0
    try:
        server_cfg = srv.config.get("server", {})
        raw = server_cfg.get("local_dispatch_first_byte_timeout_seconds", 840)
        return float(raw) if raw is not None else 840.0
    except Exception:
        return 840.0


# ===================================================================
# Post-deadline prefill-thrash cooldown (LP-0MVKUKFHH005UDWL)
# ===================================================================
#
# Root cause 3 of 3 of the 2026-10-23 wedged prefill incident
# (LP-0MVKN9IRP003XJH6): after ``prefill_deadline_exceeded`` cancels a local
# dispatch, the client re-dispatched the *identical* prompt and burned the
# whole prefill again — 41 local POSTs in the incident hour (35 inside
# 02:00-03:00) re-prefilling near-duplicates of the same 28k-124k-token
# prompts (93,038 x2, 93,006 x3, …) with zero tokens delivered.
#
# The cooldown is a per-session, in-process window armed by router.py when
# the prefill deadline fires. While it is armed, local dispatch is refused for
# that session before any local resource is acquired — provider.py's chain gate
# skips local and the acquisition choke points refuse it — so the request
# reaches the next provider immediately instead of re-prefilling. Both sites
# log ``routing_skip_local reason=prefill_thrash_cooldown`` (the choke points
# add ``entry=router_dispatch`` / ``entry=lease_acquire``; the choke-point
# half of the gate is LP-0MVSAMN14001DMTY AC2).
# Other sessions are untouched. The window is cleared on expiry (lazy prune)
# and on a successful local first byte, and ``0`` disables it (documented
# rollback).

DEFAULT_PREFILL_THRASH_COOLDOWN_SECONDS = 600.0
"""Default post-deadline cooldown for a thrashing session (10 minutes)."""

DEFAULT_PREFILL_THRASH_RECOVERY_FRACTION = 0.9
"""Share of the cancelled prompt a success must reach to clear the window.

A successful local dispatch clears a session's post-deadline cooldown only
when it prefilled at least this fraction of the prompt that armed the window
(LP-0MVQITN6P008DPLB). The window is cleared to restore local service once the
failing case demonstrably works again — not because some *unrelated*, much
smaller dispatch happened to succeed. Measured 2026-10-26 09:06→2026-10-27
09:06: 89 of 114 ``prefill_deadline`` evictions were repeats inside 33
``(session, prompt)`` clusters, re-admitted because a ~1.5 s warm dispatch had
cleared the window the large doomed prompt had just armed (e.g. session
``01a178bd…`` cancelled at 15:25:57, 15:28:05 and 15:30:20 on the same
46,134-token prompt, against a 600 s cooldown).

``0`` restores the pre-fix clear-on-any-success rule (documented rollback).
"""

#: ``(session id, prompt fingerprint)`` → monotonic deadline.
#:
#: The window is keyed by **(session, prompt fingerprint)**, not by session
#: alone (LP-0MVT2KSXY002O75C, parent LP-0MVRMI5HK00558TB AC2). A session-wide
#: veto armed by one doomed 60 k prompt also refused the next, unrelated (and
#: usually much smaller) prompt for a full 600 s: measured 2026-10-24 20:00 →
#: 2026-10-25 07:00, 916 ``routing_skip_local reason=prefill_thrash_cooldown``
#: skips across 46 sessions, 241 of them for 21 ``herdr-*`` sessions and 336
#: where the proxy's own nearest status poll reported the GPU free.
#:
#: The fingerprint is the empty string for a *session-wide* window — what a
#: caller with no usable fingerprint arms, and what every gate consults as a
#: fallback, so a site that cannot fingerprint a request fails open to the
#: pre-fix behaviour instead of silently losing the gate (see
#: :func:`_prefill_thrash_fingerprint`).
#:
#: In-process by design: a proxy restart resets the routing state, exactly
#: like the dispatch leases it guards.
_prefill_thrash_cooldowns: dict[tuple[str, str], float] = {}

#: ``(session id, prompt fingerprint)`` → prompt size (tokens) that armed the
#: window. Recorded so the clear path can compare a success against the case
#: that failed; an absent entry (unknown size) keeps the fail-open
#: clear-on-any-success path.
_prefill_thrash_cancel_tokens: dict[tuple[str, str], int] = {}

#: Length of the prompt fingerprint prefix (hex chars). Bounded so the key
#: cannot grow with the prompt: a 32-char SHA-256 prefix keeps collision risk
#: negligible while staying readable in a log line.
_PREFILL_THRASH_FINGERPRINT_HEX_CHARS = 32


def _prefill_thrash_turn_messages(messages: list) -> list:
    """Return the messages of the *current turn* (the ones being prefilled).

    The arming site and the gate site hold different bodies for the same
    request (LP-0MVSAMN14001DMTY AC2): when session delta routing is active,
    ``_handle_session`` rewrites ``body_json["messages"]`` to the delta it is
    about to prefill, so ``router.py`` arms the window from the *delta* while
    ``provider.py``'s chain gate fingerprints the *original* client body it
    parsed. Fingerprinting the whole list therefore produced two different
    digests for one request and the window never matched — the same
    non-engagement, one level down (the 2026-10-26 incident logged **zero**
    ``routing_skip_local reason=prefill_thrash_cooldown`` lines for the
    affected session across four successive local dispatches).

    The delta is always a *trailing suffix* of the original list (the client
    appends the new turn), so fingerprinting everything after the last
    assistant message yields the same digest for both bodies while staying a
    prompt-scoped identity: a different turn (different content, order or
    tools) still yields a different digest. A body with no assistant message
    (a first turn, or the delta itself) keeps the whole list, which is also
    what makes the two bodies agree.
    """
    if not isinstance(messages, list):
        return []
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, Mapping) and message.get("role") == "assistant":
            turn = messages[index + 1:]
            # A trailing assistant message (no new turn) keeps the whole list
            # rather than fingerprinting "nothing".
            return turn or messages
    return messages


def _prefill_thrash_fingerprint(body_json) -> str | None:
    """Return a stable, bounded fingerprint of the request's prompt.

    Used to scope the post-deadline prefill-thrash cooldown to the prompt that
    actually failed (LP-0MVT2KSXY002O75C). Derived only from ``messages`` —
    the part that determines what gets prefilled — with sorted keys and
    compact separators so the same payload always yields the same digest and
    message order is significant.

    The digest covers the **current turn** (:func:`_prefill_thrash_turn_messages`)
    rather than the whole message list, so the router's post-session delta
    body and ``provider.py``'s original client body fingerprint identically
    for the same request (LP-0MVSAMN14001DMTY AC2).

    Fail-open: returns ``None`` when the payload carries no usable messages
    (streaming bookkeeping-only bodies, malformed JSON, a non-list
    ``messages``). Callers then fall back to the session-wide window, so a
    site that cannot fingerprint never stops engaging the gate.

    Cost: one ``json.dumps`` + one ``sha256`` over the current turn, at the
    arming site (once per abandoned dispatch) and the gate site (once per
    local candidate). No body re-read: the router and
    ``proxy_with_fallback`` already hold the parsed payload.
    """
    if not isinstance(body_json, Mapping):
        return None
    messages = body_json.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    turn = _prefill_thrash_turn_messages(messages)
    if not turn:
        return None
    try:
        canonical = json.dumps(
            turn,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        return None
    try:
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except Exception:
        return None
    return digest[:_PREFILL_THRASH_FINGERPRINT_HEX_CHARS]


def _prefill_thrash_key(session_id, fingerprint) -> tuple[str, str] | None:
    """Build the cooldown store key (``None`` when there is no session)."""
    if not session_id:
        return None
    return (str(session_id), str(fingerprint or ""))


def _prefill_thrash_keys_for(session_id, fingerprint) -> list[tuple[str, str]]:
    """Keys a gate at ``(session, fingerprint)`` must consult, in order.

    - With a fingerprint: the exact ``(session, fingerprint)`` window, then the
      session-wide (empty-fingerprint) window — the latter is what a caller
      that could not fingerprint *armed*, so it must still be consulted.
    - Without one: every window of the session. A gate that cannot fingerprint
      the request has to assume it might be the failing prompt, which is the
      documented fail-open direction (the gate keeps engaging; it can never
      silently stop refusing).
    """
    if not session_id:
        return []
    sid = str(session_id)
    if not fingerprint:
        return [key for key in _prefill_thrash_cooldowns if key[0] == sid] or [
            (sid, "")
        ]
    return [(sid, str(fingerprint)), (sid, "")]


def _drop_prefill_thrash_session(session_id) -> None:
    """Forget every window (all fingerprints) belonging to *session_id*."""
    if not session_id:
        return
    sid = str(session_id)
    for store in (_prefill_thrash_cooldowns, _prefill_thrash_cancel_tokens):
        for key in [k for k in store if k[0] == sid]:
            store.pop(key, None)


def _prefill_thrash_config_section(config) -> Mapping:
    """Return the ``server`` config section of *config* (flat configs too)."""
    if not isinstance(config, Mapping):
        return {}
    server = config.get("server")
    if isinstance(server, Mapping):
        return server
    return config


def _get_prefill_thrash_cooldown_seconds(config) -> float:
    """Return the post-deadline cooldown in seconds (default 600; 0 disables).

    ``server.local_prefill_thrash_cooldown_seconds`` is the rollback knob: 0
    disables the cooldown entirely (both the arming on a deadline event and
    the routing skip). A missing or malformed value falls back to the default
    so a config typo cannot silently switch the protection off.
    """
    section = _prefill_thrash_config_section(config)
    raw = section.get(
        "local_prefill_thrash_cooldown_seconds",
        DEFAULT_PREFILL_THRASH_COOLDOWN_SECONDS,
    )
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_PREFILL_THRASH_COOLDOWN_SECONDS
    return max(0.0, seconds)


def _get_prefill_thrash_recovery_fraction(config) -> float:
    """Resolve ``server.local_prefill_thrash_recovery_fraction`` (default 0.9).

    Returns the share of the cancelled prompt size a successful local dispatch
    must reach before it may clear the cooldown window. ``0`` (or a
    non-positive / malformed-but-non-zero value) disables the gate, restoring
    the pre-fix clear-on-any-success behaviour; a malformed *string* falls back
    to the default so a config typo cannot silently switch the protection off.
    A value above 1.0 is clamped to 1.0 — anything higher would make recovery
    impossible and strand the session off local forever.
    """
    section = _prefill_thrash_config_section(config)
    raw = section.get(
        "local_prefill_thrash_recovery_fraction",
        DEFAULT_PREFILL_THRASH_RECOVERY_FRACTION,
    )
    try:
        fraction = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_PREFILL_THRASH_RECOVERY_FRACTION
    if fraction <= 0.0:
        return 0.0
    return min(1.0, fraction)


def _record_prefill_thrash_cooldown(
    session_id, config, now=None, tokens=None, fingerprint=None
) -> float:
    """Arm (or re-arm) a cooldown window; return its monotonic deadline.

    Called by router.py when ``prefill_deadline_exceeded`` (or the measured
    cache-deficit early abort) cancels a local dispatch. Returns the monotonic
    deadline, or 0 when the cooldown is disabled or no session id is known (a
    cooldown needs a session to scope to — an anonymous one-shot request has
    nothing to suppress).

    *fingerprint* scopes the window to the prompt that failed
    (:func:`_prefill_thrash_fingerprint`, LP-0MVT2KSXY002O75C). Only that
    prompt is then refused; an unrelated prompt from the same session is
    admitted. ``None`` keeps the pre-fix **session-wide** window, which is the
    documented fail-open behaviour when the arming site cannot fingerprint
    the payload.

    *tokens* is the prompt size that failed. It is recorded so a later success
    can be compared against it (see
    :func:`_clear_prefill_thrash_cooldown`); when it is absent or not a
    positive integer the window is still armed, but the clear path falls back
    to the fail-open clear-on-any-success behaviour.
    """
    if not session_id:
        return 0.0
    seconds = _get_prefill_thrash_cooldown_seconds(config)
    if seconds <= 0:
        _drop_prefill_thrash_session(session_id)
        return 0.0
    key = _prefill_thrash_key(session_id, fingerprint)
    if key is None:  # pragma: no cover - guarded by the session check above
        return 0.0
    _now = time.monotonic() if now is None else now
    deadline = _now + seconds
    _prefill_thrash_cooldowns[key] = deadline
    try:
        size = int(tokens or 0)
    except (TypeError, ValueError):
        size = 0
    if size > 0:
        _prefill_thrash_cancel_tokens[key] = size
    else:
        _prefill_thrash_cancel_tokens.pop(key, None)
    return deadline


def _prefill_thrash_cooldown_remaining(
    session_id, config=None, now=None, fingerprint=None
) -> float:
    """Return the seconds left in a cooldown window (0 when clear).

    Consults the ``(session, fingerprint)`` window when *fingerprint* is
    known, and always the session-wide window as the fail-open fallback, so a
    gate site that cannot fingerprint a request still refuses what might be
    the failing prompt (LP-0MVT2KSXY002O75C).

    Expired windows are pruned here, so expiry needs no timer. A live config
    flip to 0 also clears the session's windows (documented rollback).
    """
    keys = _prefill_thrash_keys_for(session_id, fingerprint)
    if not keys:
        return 0.0
    if config is not None and _get_prefill_thrash_cooldown_seconds(config) <= 0:
        _drop_prefill_thrash_session(session_id)
        return 0.0
    _now = time.monotonic() if now is None else now
    remaining = 0.0
    for key in keys:
        deadline = _prefill_thrash_cooldowns.get(key)
        if deadline is None:
            continue
        left = deadline - _now
        if left <= 0:
            # Expired: prune the window (and its armed size) lazily.
            _prefill_thrash_cooldowns.pop(key, None)
            _prefill_thrash_cancel_tokens.pop(key, None)
            continue
        if left > remaining:
            remaining = left
    return remaining


def _clear_prefill_thrash_cooldown(
    session_id, tokens=None, config=None, fingerprint=None
) -> bool:
    """Clear a cooldown window; return True when one was armed.

    Called by router.py on a successful local first byte. A healthy dispatch
    proves prefill works for this session again — but only for *that* size.
    When the window was armed by a large prompt that the deadline cancelled, a
    much smaller success is no evidence that the large prompt works now, and
    clearing on it re-admits the doomed dispatch (LP-0MVQITN6P008DPLB).

    The window is therefore cleared only when the successful dispatch prefilled
    at least ``local_prefill_thrash_recovery_fraction`` (default 0.9) of the
    size that armed it. Fail-open: when the armed size or the succeeding size
    is unknown, or the fraction is 0, the pre-fix clear-on-any-success
    behaviour is preserved rather than guessed at.

    *fingerprint* scopes the clear to the prompt that just succeeded
    (LP-0MVT2KSXY002O75C): a comparable success on prompt P must not release
    the window armed by prompt Q. When *fingerprint* is ``None`` the call is
    treated as "the caller cannot tell which prompt succeeded" and every
    window of the session is evaluated (pre-fix behaviour, fail-open).
    """
    if not session_id:
        return False
    if fingerprint:
        keys = _prefill_thrash_keys_for(session_id, fingerprint)
    else:
        sid = str(session_id)
        keys = [key for key in _prefill_thrash_cooldowns if key[0] == sid]
    if not keys:
        return False
    fraction = _get_prefill_thrash_recovery_fraction(config)
    cleared = False
    try:
        succeeded = int(tokens) if tokens is not None else 0
    except (TypeError, ValueError):
        succeeded = 0
    for key in keys:
        armed = _prefill_thrash_cancel_tokens.get(key)
        if fraction > 0.0 and armed:
            if succeeded > 0 and succeeded < armed * fraction:
                # Below the fraction: keep this window armed.
                continue
        _prefill_thrash_cancel_tokens.pop(key, None)
        if _prefill_thrash_cooldowns.pop(key, None) is not None:
            cleared = True
    return cleared


def _gpu_demonstrably_free(now=None, endpoint: str | None = None) -> bool:
    """Return True only when the proxy's own fresh evidence proves the GPU free.

    Evidence used — the same the admission path already reads, no extra poll
    (LP-0MVT2KSXY002O75C AC2):

    1. **No live local dispatch**: no dispatch record is ``active``. A record
       is proxy-owned in-flight work, so its existence blocks the release.
    2. **A fresh ``/slots`` snapshot**: :func:`proxy.residency.get_residency_view`
       reports ``known`` (snapshot present and younger than
       ``RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS``). A stale or missing snapshot
       proves nothing and never releases.
    3. **No processing slot at all** — including a slot the proxy has
       abandoned. While llama-server is still processing anything the GPU is
       genuinely busy, so the window stays armed; that is precisely why AC2
       had to be sequenced after AC1 (LP-0MVRMI5HK00558TB AC1 cancels the
       abandoned upstream stream so the ghost window is short instead of
       lasting the whole prefill).

    Fail-closed by construction: every unknown (no snapshot, no dispatch
    registry, unreadable entry) answers "not free", so the cooldown keeps
    engaging rather than being released on a guess.
    """
    _now = time.monotonic() if now is None else now
    # 1. Live local dispatch records.
    try:
        srv = _srv()
        records = getattr(srv, "local_dispatch_records", None)
        if records is not None:
            for record in records.values():
                if isinstance(record, dict) and record.get("active"):
                    return False
    except Exception:
        return False
    # 2 + 3. Fresh snapshot and nothing processing.
    try:
        from proxy.residency import get_residency_view

        view = get_residency_view(now=_now, endpoint=endpoint)
    except Exception:
        return False
    if not view.known:
        return False
    for slot in view.slots:
        if slot.is_processing:
            return False
    return True


def _release_prefill_thrash_cooldown_on_free_gpu(
    session_id, config=None, now=None, endpoint: str | None = None, fingerprint=None
) -> bool:
    """Release a cooldown window when a fresh snapshot proves the GPU free.

    The second half of AC2 (LP-0MVT2KSXY002O75C): a window armed by one
    abandonment must not veto local routing while the GPU is demonstrably
    free. Measured baseline — 336 of 916 skips happened with the proxy's own
    nearest status poll reporting no live dispatch — is what this removes.

    The release is deliberately **not** a blanket disable: it requires the
    freshness, liveness and ghost evidence of :func:`_gpu_demonstrably_free`,
    so a stale snapshot, a live dispatch or a ghost still processing keeps the
    window armed and the failing prompt refused.

    When the evidence holds, every window of the session (all fingerprints) is
    released: the justification for *all* of them ("the GPU could still be
    finishing the abandoned prefill") is gone. Logged once, with an
    attributable reason, so the release is measurable from logs alone.

    Returns True when at least one window was released.
    """
    keys = _prefill_thrash_keys_for(session_id, fingerprint)
    if not keys:
        return False
    remaining = _prefill_thrash_cooldown_remaining(
        session_id, config, now=now, fingerprint=fingerprint
    )
    if remaining <= 0:
        return False
    if not _gpu_demonstrably_free(now=now, endpoint=endpoint):
        return False
    _drop_prefill_thrash_session(session_id)
    try:
        srv = _srv()
        srv.logger.info(
            "prefill_thrash_cooldown_released session=%s reason=free_gpu "
            "cooldown_remaining=%.0fs endpoint=%s",
            session_id if session_id else "unknown",
            remaining,
            endpoint or "default",
        )
    except Exception:
        pass
    return True


def _reset_prefill_thrash_cooldowns() -> None:
    """Clear every cooldown window (test isolation / explicit reset)."""
    _prefill_thrash_cooldowns.clear()
    _prefill_thrash_cancel_tokens.clear()


# ===================================================================
# Post-deadline local dispatch gate (LP-0MVSAMN14001DMTY AC1/AC2)
#
# Measured defect (2026-10-26 10:39-10:47, ``proxy.log.2026-10-26_131429``):
# a local dispatch cancelled by ``prefill_deadline_exceeded`` released its
# lease truthfully, and ~150 ms later the *same request* re-dispatched local
# and acquired a **fresh** lease, which no watchdog progress could ever
# release until the 180 s no-progress watchdog (≈308 s of unproductive hold,
# other sessions denied 311-322 s each). The armed 600 s thrash cooldown never
# engaged: zero ``routing_skip_local reason=prefill_thrash_cooldown`` lines
# for the affected session across four successive dispatches, because the
# in-request retry loops re-enter ``proxy_to_local`` *after* provider.py's
# chain gate has already been passed.
#
# Two gates therefore sit at the acquisition choke points themselves, where no
# dispatch path can skip them:
#
# 1. **Session cooldown** (AC1) — the same
#    ``_prefill_thrash_cooldown_remaining`` provider.py consults, checked in
#    ``_try_acquire_local_dispatch`` (the lease choke point) and in
#    ``proxy_to_local`` before *any* local resource is acquired (the gate for
#    the anonymous ``_increment_local_active_queries`` path and other direct
#    callers).
# 2. **Request-scoped terminal marker** (AC2) — ``prefill_deadline_exceeded``
#    marks the request object, so local is terminal for that request even when
#    the session cooldown cannot be resolved (a header-less request arms the
#    window under the session id the router generated, which no gate site can
#    re-derive from the headers). The marker lives on the ``Request`` object,
#    so a concurrent unrelated request from the same session is unaffected.
# ===================================================================

#: ``Request`` attribute holding the monotonic time at which this request had a
#: local dispatch cancelled by ``prefill_deadline_exceeded`` (or the measured
#: cache-deficit early abort). Request-scoped by construction: it dies with the
#: ``Request`` object, so it can never leak across requests or sessions.
_LOCAL_DISPATCH_TERMINAL_ATTR = "_local_dispatch_post_deadline_ts"


def _mark_local_dispatch_terminal(request, now=None) -> float | None:
    """Mark *request* as terminal for local dispatch for the rest of its life.

    Called by ``router.py`` when a local dispatch is cancelled before its first
    byte — ``prefill_deadline_exceeded`` (and the measured cache-deficit early
    abort). The deadline detour is then terminal for local **within that
    request**: the same request can never acquire a fresh dispatch lease after
    the release, so the 2026-10-26 in-request re-dispatch (a fresh ``[local]
    POST`` 147-150 ms after ``lease_released reason=prefill_deadline``) cannot
    happen. Mechanism, scope and the evidence line
    (``routing_skip_local reason=post_deadline_terminal``):
    LP-0MVSAMN14001DMTY AC2, verified by LP-0MVSAMS5O006OAIS.

    Scope is deliberately the request, not the session: the session-wide
    cooldown is what suppresses *later* requests, while a concurrent request
    from the same session (a different ``Request`` object) keeps its own
    admission decision.

    Returns the recorded monotonic timestamp, or ``None`` when the request
    object cannot carry attributes (``None``, a slotted test double).
    """
    if request is None:
        return None
    _now = time.monotonic() if now is None else now
    try:
        setattr(request, _LOCAL_DISPATCH_TERMINAL_ATTR, _now)
    except Exception:
        return None
    return _now


def _local_dispatch_terminal_at(request) -> float | None:
    """Return when *request* became terminal for local dispatch, else ``None``."""
    if request is None:
        return None
    try:
        value = getattr(request, _LOCAL_DISPATCH_TERMINAL_ATTR, None)
    except Exception:
        return None
    return value if isinstance(value, (int, float)) else None


def _local_dispatch_gate(
    srv,
    session_key,
    *,
    request=None,
    config=None,
    fingerprint=None,
    entry="router",
) -> tuple[bool, str, float]:
    """Consult the post-deadline local gates at a dispatch acquisition point.

    Returns ``(refused, reason, remaining_seconds)``:

    - ``(True, "post_deadline_terminal", 0.0)`` — this request already had a
      local dispatch cancelled post-deadline: local is terminal for it
      (LP-0MVSAMN14001DMTY AC2).
    - ``(True, "prefill_thrash_cooldown", remaining)`` — the session's
      post-deadline window is armed for this prompt (AC1). ``remaining`` is the
      seconds left, so a caller can surface ``Retry-After``.
    - ``(False, "", 0.0)`` — admitted; nothing armed.

    Both refusals log ``routing_skip_local … reason=<reason>`` so the gate is
    attributable in ``proxy.log`` from the dispatch side (provider.py logs the
    same event from the chain side; ``entry=`` names the site). The existing
    log vocabulary is kept — ``routing_skip_local``,
    ``prefill_thrash_cooldown_started``, ``lease_released``.

    The free-GPU release (:func:`_release_prefill_thrash_cooldown_on_free_gpu`)
    is deliberately **not** attempted here. It belongs to provider.py's chain
    entry, which runs before this point for every new request: by the time this
    choke point is reached the window has already been released when the
    proxy's own evidence proved the GPU free. Keeping the choke point purely
    deterministic is what makes it a reliable safety net for the paths that
    bypass the chain (in-request retries, ``ui.py``'s direct dispatch) instead
    of a second, racy release site.

    Fail-open by construction: an unknown session, a disabled rollback knob
    (``local_prefill_thrash_cooldown_seconds: 0``) or an expired window all
    return ``(False, …, 0.0)``.
    """
    terminal_at = _local_dispatch_terminal_at(request)
    if terminal_at is not None:
        _log_local_dispatch_gate_skip(
            srv, session_key, "post_deadline_terminal", 0.0, entry
        )
        return (True, "post_deadline_terminal", 0.0)
    remaining = _prefill_thrash_cooldown_remaining(
        session_key,
        config if config is not None else _server_config_of(srv),
        fingerprint=fingerprint,
    )
    if remaining > 0:
        _log_local_dispatch_gate_skip(
            srv, session_key, "prefill_thrash_cooldown", remaining, entry
        )
        return (True, "prefill_thrash_cooldown", remaining)
    return (False, "", 0.0)


def _log_local_dispatch_gate_skip(
    srv, session_key, reason: str, remaining: float, entry: str
) -> None:
    """Log a dispatch-side gate refusal (never raises; logging is optional)."""
    try:
        srv.logger.info(
            "routing_skip_local session=%s reason=%s cooldown_remaining=%.0fs "
            "entry=%s → refusing local dispatch, chain falls to next provider",
            session_key if session_key else "unknown",
            reason,
            remaining,
            entry,
        )
    except Exception:
        pass


# ===================================================================
# Abandoned local dispatch registry (LP-0MVRMI5HK00558TB AC1)
#
# Measured defect (LP-0MVOG7IP80003MLV, 2026-10-24 20:00 -> 2026-10-25
# 07:00): 332 local dispatches were abandoned by the proxy while
# llama-server kept computing them, so 247 fresh ``/slots`` polls reported
# ``capacity_hint=saturated`` with ``local_in_flight_requests=0`` — work
# nobody was waiting for, truthfully reported from a snapshot that cannot
# tell who abandoned the slot.
#
# Two mechanisms carry AC1, and this registry is the second one:
#
# 1. **Cancel the upstream stream** on every path that holds a handle
#    (deadline, session eviction — the watchdog and client-disconnect paths
#    already did; LP-0MVIQ0FJC000YLX0 / LP-0MVLFVY6K003VYEL). llama-server
#    aborts at its next write to the socket, which ``return_progress``
#    guarantees happens during prefill, so the ghost window is short.
# 2. **Record the abandoned slot identity** here, so the status payload can
#    annotate a slot that is still processing as *busy, not ours* even when
#    the backend has not noticed the cancel yet. The annotation is what makes
#    the AC1 measurement (``saturated`` without live work ~ 0) hold
#    regardless of how promptly the backend reacts.
#
# The registry is in-process by design: it resets with the routing state it
# describes (a proxy restart ends every ghost it could have recorded).
# ===================================================================

#: Abandonment reasons the parent AC names (LP-0MVRMI5HK00558TB AC1).
#: Every one of them must be recorded so the annotation is per-reason
#: attributable. The set is documentation plus the assertion the regression
#: harness makes — recording itself accepts any reason (the measured
#: cache-deficit early abort is an abandonment too, and its own reason string
#: is already attributable), it is never silently rewritten.
ABANDONED_SLOT_REASONS = frozenset(
    {
        "prefill_deadline",
        "session_evicted",
        "no_progress_watchdog",
        "client_disconnect",
    }
)

#: How long an abandonment stays attributable. Bounded so a recycled slot id
#: cannot inherit a ghost's annotation forever, and comfortably longer than
#: the 180 s no-progress watchdog plus the 600 s prefill-thrash cooldown, so
#: the annotation outlives the window in which the ghost can still be seen
#: processing (the deadline is 90 s; a stuck prefill is capped far below this).
DEFAULT_ABANDONED_SLOT_TTL_SECONDS = 900.0

ABANDONED_SLOT_ENDPOINT_WIDE: int = -1
"""Registry key for an abandonment whose **slot identity is unknown**.

A documented sentinel, deliberately an ``int`` because the registry is keyed
the same way and deliberately negative because llama-server slot ids are always
non-negative — so the endpoint-wide entry can never collide with a real slot
(LP-0MVUVHAQP001IRKK AC1).

Readers must go through :func:`_abandoned_slot_endpoint_wide`, **never**
:func:`_abandoned_slot_for`: an endpoint-wide entry names no slot, so applying
it to a live slot is only safe behind the "no live dispatch record / lease
owner" guard of the status builder (LP-0MVUVHB1V006HA5I).
"""

_abandoned_slots: dict[tuple[str, int], dict] = {}
"""``{(endpoint_or_empty, key): entry}`` for abandoned local dispatches.

``key`` is the slot id when it is known, otherwise
:data:`ABANDONED_SLOT_ENDPOINT_WIDE`.

``entry`` carries ``slot_id`` (``None`` for an endpoint-wide entry),
``endpoint``, ``session_id``, ``reason``, ``abandoned_at``, ``ttl_seconds``
and ``scope`` (``"slot"`` | ``"endpoint"``) — always attributable, never a
bare boolean (LP-0MVRMI5HK00558TB AC1). An abandonment is never silently
dropped: an unresolvable identity records the endpoint-wide entry so nothing
the proxy gave up on is unattributable (LP-0MVUVHAQP001IRKK AC1).

Memory bound: the key is the slot identity, so the registry can never hold
more entries than there are ``(endpoint, slot)`` pairs plus **one**
endpoint-wide entry per endpoint — re-abandoning the same slot (or the same
endpoint without an identity) overwrites/refreshes its entry. Expiry is lazy
(pruned on read and by the periodic sweep), so no timer task is needed.
"""


def _abandoned_slot_key(endpoint: str | None, slot_id: int) -> tuple[str, int]:
    """Build the per-endpoint registry key (mirrors session._slot_registry_key)."""
    return (endpoint or "", slot_id)


def _prune_abandoned_slots(now: float | None = None) -> int:
    """Drop expired entries; return how many were removed.

    Expiry is lazy (evaluated on read and on this sweep) so the registry needs
    no timer of its own.
    """
    _now = time.monotonic() if now is None else now
    expired = [
        key
        for key, entry in _abandoned_slots.items()
        if _now >= float(entry.get("abandoned_at", 0.0))
        + float(entry.get("ttl_seconds", 0.0))
    ]
    for key in expired:
        _abandoned_slots.pop(key, None)
    return len(expired)


def _abandoned_slot_is_real_id(slot_id: Any) -> bool:
    """Return True when *slot_id* can name a real llama-server slot.

    llama-server slot ids are non-negative integers; everything else (``None``,
    a string, a bool, the endpoint-wide sentinel) is an **unresolved** identity.
    """
    return (
        isinstance(slot_id, int)
        and not isinstance(slot_id, bool)
        and slot_id >= 0
    )


def _record_abandoned_slot(
    slot_id: int | None,
    *,
    session_id: str | None = None,
    endpoint: str | None = None,
    reason: str,
    now: float | None = None,
    ttl: float | None = None,
) -> dict:
    """Record that the proxy gave up on *slot_id* for *reason*.

    Called from every abandonment path (deadline, session eviction, watchdog,
    client disconnect, orphan sweep, measured cache-deficit abort) AFTER the
    upstream cancel has been attempted, so the status payload can annotate a
    slot that is still processing as "abandoned" rather than as live
    proxy-owned work (LP-0MVRMI5HK00558TB AC1). Re-recording the same slot
    refreshes the window (a slot abandoned twice must not expire from the
    first event).

    **Never drops an abandonment** (LP-0MVUVHAQP001IRKK AC1): when *slot_id* is
    not a usable slot identity the entry is stored endpoint-wide under the
    :data:`ABANDONED_SLOT_ENDPOINT_WIDE` sentinel with ``slot_id=None`` and
    ``scope="endpoint"``, so a release that could not resolve a slot is still
    attributable to the endpoint it gave work up on.

    Returns the stored entry.
    """
    _now = time.monotonic() if now is None else now
    try:
        _ttl = float(ttl) if ttl is not None else DEFAULT_ABANDONED_SLOT_TTL_SECONDS
    except (TypeError, ValueError):
        _ttl = DEFAULT_ABANDONED_SLOT_TTL_SECONDS
    if _ttl <= 0:
        _ttl = DEFAULT_ABANDONED_SLOT_TTL_SECONDS
    resolved_slot_id = int(slot_id) if _abandoned_slot_is_real_id(slot_id) else None
    scope = "slot" if resolved_slot_id is not None else "endpoint"
    key_id = (
        resolved_slot_id
        if resolved_slot_id is not None
        else ABANDONED_SLOT_ENDPOINT_WIDE
    )
    entry = {
        "slot_id": resolved_slot_id,
        "endpoint": endpoint or "",
        "session_id": session_id,
        "reason": str(reason),
        "abandoned_at": _now,
        "ttl_seconds": _ttl,
        "scope": scope,
    }
    _abandoned_slots[_abandoned_slot_key(endpoint, key_id)] = entry
    return entry


def _abandoned_slot_for(
    slot_id: int | None,
    *,
    endpoint: str | None = None,
    now: float | None = None,
) -> dict | None:
    """Return the live abandonment entry for *slot_id* (``None`` when none).

    Matches the exact endpoint first, then the legacy/default endpoint entry,
    then any endpoint carrying that slot id — so an entry written by a path
    that could not resolve the endpoint still annotates the slot the status
    payload reads (the annotation must never silently vanish because two call
    sites disagree about the endpoint spelling).

    Only a **real** slot id can match: an endpoint-wide entry (recorded with no
    identity, keyed by :data:`ABANDONED_SLOT_ENDPOINT_WIDE`) is never returned
    here, so this lookup can never mis-attribute a ghost to a live slot. Use
    :func:`_abandoned_slot_endpoint_wide` for that entry behind its guard.
    """
    if not _abandoned_slot_is_real_id(slot_id):
        return None
    _now = time.monotonic() if now is None else now
    _prune_abandoned_slots(now=_now)
    keys = [_abandoned_slot_key(endpoint, slot_id)]
    fallback = _abandoned_slot_key(None, slot_id)
    if fallback not in keys:
        keys.append(fallback)
    for key in keys:
        entry = _abandoned_slots.get(key)
        if entry is not None:
            return entry
    # Last resort: the same slot id recorded under a different endpoint label
    # (multi-backend deployments; the local status payload describes the
    # current local child, which may or may not carry an explicit label).
    for (ep, sid), entry in _abandoned_slots.items():
        if entry.get("slot_id") is not None and sid == slot_id:
            return entry
    return None


def _abandoned_slot_endpoint_wide(
    *, endpoint: str | None = None, now: float | None = None
) -> dict | None:
    """Return the live **endpoint-wide** abandonment entry (``None`` when none).

    The only sanctioned reader of an entry recorded without a slot identity.
    Resolved the same way as :func:`_abandoned_slot_for` — exact endpoint first,
    then the legacy/default label, then any endpoint — so an entry written by a
    path that carried a different endpoint spelling is still readable.

    A caller **must** apply the "no live dispatch record / no lease owner"
    guard before annotating a slot from this entry: it names no slot, so it can
    only ever describe a draining ghost (LP-0MVUVHB1V006HA5I).
    """
    _now = time.monotonic() if now is None else now
    _prune_abandoned_slots(now=_now)
    keys = [_abandoned_slot_key(endpoint, ABANDONED_SLOT_ENDPOINT_WIDE)]
    fallback = _abandoned_slot_key(None, ABANDONED_SLOT_ENDPOINT_WIDE)
    if fallback not in keys:
        keys.append(fallback)
    for key in keys:
        entry = _abandoned_slots.get(key)
        if entry is not None:
            return entry
    for (ep, sid), entry in _abandoned_slots.items():
        if sid == ABANDONED_SLOT_ENDPOINT_WIDE:
            return entry
    return None


def _abandoned_slot_count(
    *, endpoint: str | None = None, now: float | None = None
) -> int:
    """Count live abandonment entries (top-level status payload field)."""
    _now = time.monotonic() if now is None else now
    _prune_abandoned_slots(now=_now)
    if endpoint is None:
        return len(_abandoned_slots)
    want = endpoint or ""
    return sum(1 for key in _abandoned_slots if key[0] == want)


def _reset_abandoned_slots() -> None:
    """Clear the registry (test isolation / explicit reset)."""
    _abandoned_slots.clear()


def _abandoned_slot_lookup(
    session_id: str | None, endpoint: str | None = None
) -> int | None:
    """Return the llama-server slot *session_id* is assigned, if any.

    Read-only (never assigns): the abandonment paths call this *before* the
    slot assignment is freed, so the registry entry can name the slot the
    ghost actually occupies. Best-effort — an unknown slot is recorded
    endpoint-wide by :func:`_record_abandoned_slot` rather than dropped.
    """
    if not session_id:
        return None
    try:
        from proxy.session import _assigned_slot_for_session

        # The dispatch-lease key treats the legacy literal ``"local"`` as "no
        # endpoint" (``_dispatch_lease_key``) while the slot registry stores
        # the empty label, so normalise before looking up — otherwise an
        # identity that exists silently fails to resolve and the abandonment
        # is recorded endpoint-wide for no reason (LP-0MVUVHAQP001IRKK).
        lookup_endpoint = "" if endpoint == "local" else endpoint
        return _assigned_slot_for_session(session_id, endpoint=lookup_endpoint)
    except Exception:
        return None


def _observed_abandoned_slot(
    session_id: str | None, endpoint: str | None = None
) -> int | None:
    """Return the slot llama-server **currently reports** for *session_id*.

    Reads the in-process ``/slots``-derived pairing
    (``observability._processing_slot_assignments``, slot id → session id). This
    is the backend's own view, independent of the proxy's slot assignment, so
    comparing it with the recorded id is what makes an identity mismatch
    diagnosable (LP-0MVUVHAQP001IRKK). Read-only and synchronous; ``None`` when
    the pairing has no entry for the session (no status poll yet, or already
    cleaned up).
    """
    if not session_id:
        return None
    try:
        from proxy import observability as _observability

        assignments = getattr(_observability, "_processing_slot_assignments", None)
        if not isinstance(assignments, dict):
            return None
        for slot_id, owner in list(assignments.items()):
            if owner == session_id and _abandoned_slot_is_real_id(slot_id):
                return int(slot_id)
    except Exception:
        return None
    return None


@dataclass(frozen=True)
class AbandonedSlotIdentity:
    """Slot identity resolved for an abandonment release.

    ``slot_id`` is the slot the proxy recorded (or ``None`` when it could not
    resolve one), ``source`` is the provenance published as ``slot_source=``
    (``"explicit"`` | ``"assignment"`` | ``"endpoint_wide"``) and
    ``observed_slot_id`` is the slot llama-server reported for the session when
    the release was taken. ``slot_id != observed_slot_id`` (both known) is the
    shape-1 mismatch that used to be silent.
    """

    slot_id: int | None
    source: str
    observed_slot_id: int | None


SLOT_SOURCE_EXPLICIT = "explicit"
SLOT_SOURCE_ASSIGNMENT = "assignment"
SLOT_SOURCE_ENDPOINT_WIDE = "endpoint_wide"


def _resolve_abandoned_slot_identity(
    session_id: str | None,
    *,
    endpoint: str | None = None,
    explicit: int | None = None,
    record: Mapping[str, Any] | None = None,
) -> AbandonedSlotIdentity:
    """Resolve the slot identity to record for an abandonment release.

    Precedence: the caller's *explicit* slot id, then the dispatch *record*'s
    pinned ``slot_id`` (captured at admission, so it survives a later assignment
    free), then the live session assignment. All three are read **before** the
    caller frees the assignment; an unresolvable identity yields ``source ==
    "endpoint_wide"`` and the caller records the endpoint-wide entry instead of
    dropping the abandonment.
    """
    if _abandoned_slot_is_real_id(explicit):
        slot_id: int | None = int(explicit)  # type: ignore[arg-type]
        source = SLOT_SOURCE_EXPLICIT
    else:
        pinned = record.get("slot_id") if isinstance(record, Mapping) else None
        if _abandoned_slot_is_real_id(pinned):
            slot_id = int(pinned)  # type: ignore[arg-type]
            source = SLOT_SOURCE_ASSIGNMENT
        else:
            assigned = _abandoned_slot_lookup(session_id, endpoint)
            if _abandoned_slot_is_real_id(assigned):
                slot_id = int(assigned)  # type: ignore[arg-type]
                source = SLOT_SOURCE_ASSIGNMENT
            else:
                slot_id = None
                source = SLOT_SOURCE_ENDPOINT_WIDE
    return AbandonedSlotIdentity(
        slot_id=slot_id,
        source=source,
        observed_slot_id=_observed_abandoned_slot(session_id, endpoint),
    )


def _record_abandoned_identity(
    srv,
    identity: AbandonedSlotIdentity,
    *,
    session_id: str | None,
    endpoint: str | None = None,
    reason: str,
) -> dict:
    """Record *identity* and publish the ``slot_source`` provenance on the log.

    Re-verifies the resolved id against the slot llama-server actually reports
    (captured in *identity* before the assignment was freed). On mismatch the
    release logs ``abandoned_slot_mismatch`` with ``recorded_slot=`` /
    ``observed_slot=`` and records the endpoint-wide entry **as well as** the
    recorded exact entry, so shape 1 (attributable but unmatched) becomes
    diagnosable from production logs instead of silent (LP-0MVUVHAQP001IRKK).

    Every recording logs ``abandoned_slot_recorded ... slot_source=`` so the
    intake OPEN QUESTION (why attributable polls showed
    ``local_abandoned_slots=0``) is answerable from the logs alone. Never
    raises: an abandonment log line must never break the release path.
    Returns the entry that represents the abandonment (endpoint-wide when the
    identity was unresolvable or mismatched, else the exact entry).
    """
    logger = getattr(srv, "logger", None)
    recorded = identity.slot_id
    observed = identity.observed_slot_id
    mismatch = (
        recorded is not None and observed is not None and observed != recorded
    )
    if mismatch:
        try:
            if logger is not None:
                logger.warning(
                    "abandoned_slot_mismatch session=%s endpoint=%s "
                    "recorded_slot=%s observed_slot=%s reason=%s",
                    session_id if session_id else "unknown",
                    endpoint or "default",
                    recorded,
                    observed,
                    reason,
                )
        except Exception:
            pass
    exact_entry = None
    if recorded is not None:
        exact_entry = _record_abandoned_slot(
            recorded,
            session_id=session_id,
            endpoint=endpoint,
            reason=reason,
        )
        _log_abandoned_slot_recorded(
            logger, exact_entry, identity.source, session_id, endpoint
        )
    wide_entry = None
    if recorded is None or mismatch:
        wide_entry = _record_abandoned_slot(
            None,
            session_id=session_id,
            endpoint=endpoint,
            reason=reason,
        )
        _log_abandoned_slot_recorded(
            logger, wide_entry, SLOT_SOURCE_ENDPOINT_WIDE, session_id, endpoint
        )
    return wide_entry if wide_entry is not None else exact_entry


def _pin_dispatch_slot(
    srv, record_key, session_key: str | None, endpoint: str | None = None
) -> None:
    """Pin the dispatch record's slot identity at admission (best effort).

    Captured while the session's slot assignment exists, so the abandonment
    paths can name the slot the proxy gave up on even after the assignment has
    been freed, and the recorded id can later be re-verified against the slot
    llama-server reports (LP-0MVUVHAQP001IRKK). ``_dispatch_evidence`` fills
    the same ``slot_id`` key with ``None``, hence the explicit overwrite once
    the record exists. A missing record / unresolvable assignment is a no-op.
    """
    try:
        records = getattr(srv, "local_dispatch_records", None)
        if not isinstance(records, dict):
            return
        record = records.get(record_key)
        if not isinstance(record, dict):
            return
        slot_id = _abandoned_slot_lookup(session_key, endpoint)
        if _abandoned_slot_is_real_id(slot_id):
            record["slot_id"] = int(slot_id)  # type: ignore[arg-type]
    except Exception:
        pass


def _log_abandoned_slot_recorded(
    logger,
    entry: Mapping[str, Any] | None,
    slot_source: str,
    session_id: str | None,
    endpoint: str | None,
) -> None:
    """Emit one ``abandoned_slot_recorded`` line carrying the provenance."""
    if entry is None:
        return
    try:
        if logger is None:
            return
        logger.info(
            "abandoned_slot_recorded session=%s endpoint=%s slot=%s "
            "slot_source=%s reason=%s ttl_s=%.0f",
            session_id if session_id else "unknown",
            endpoint or "default",
            entry.get("slot_id"),
            slot_source,
            entry.get("reason"),
            entry.get("ttl_seconds", 0.0),
        )
    except Exception:
        pass


#: Cached-token ratio at or above which a local dispatch counts as a
#: warm-cache request for the re-scoped AC2 first-byte metric
#: (LP-0MV6CFMU9005WU2E). Matching the routing intent: a request whose prompt
#: is largely served from the prompt cache is warm.
_WARM_CACHE_RATIO_THRESHOLD = 0.5


def _classify_cache_state(cached_ratio: float, source: str | None = None) -> str:
    """Return the cache-state label for a local dispatch's routing ratio.

    Labels:

    - ``"restored"`` — the ratio was seeded from a persisted slot snapshot and
      reaches the warm threshold, i.e. a restored-prefix win
      (LP-0MVGD9YXH008O304 AC4);
    - ``"warm"`` — *cached_ratio* >= _WARM_CACHE_RATIO_THRESHOLD;
    - ``"cold"`` — everything else.

    Keeping ``restored`` distinct from ``warm`` lets reporting separate
    restored-prefix wins from genuine prompt-cache hits. A restored ratio
    below the warm threshold classifies cold so a weak credit never inflates
    the restored/warm statistics.

    Fail-open: non-numeric / NaN / missing values classify COLD so the
    warm-cache p99 target is never diluted by unknown cases.
    """
    try:
        warm = float(cached_ratio) >= _WARM_CACHE_RATIO_THRESHOLD
    except (TypeError, ValueError):
        return "cold"
    if warm and source == "restored":
        return "restored"
    return "warm" if warm else "cold"


def _lookup_dispatch_cache_state(model_name: str, session_id: str | None):
    """Return ``(cache_state, cached_ratio)`` for a local dispatch.

    Reads the routing cached-token ratio for ``(model_name, session_id)``
    (``provider._get_cached_ratio`` — 0.0 when absent/unknown) together with
    its provenance label, so a ratio seeded from a persisted slot snapshot is
    reported as ``restored`` rather than ``warm`` (LP-0MVGD9YXH008O304 AC4).
    Feeds the ``dispatch_first_byte_ms`` log line so warm-cache dispatches are
    identifiable and the warm-cache p99 can be measured
    (LP-0MV6CFMU9005WU2E). Never raises: the metric path must not break the
    streaming loop (provider failures classify cold/0.0).
    """
    try:
        from proxy.provider import _get_cached_ratio, _get_cached_ratio_source

        ratio = float(_get_cached_ratio(model_name, session_id) or 0.0)
        source = _get_cached_ratio_source(model_name, session_id)
    except Exception:
        return "cold", 0.0
    return _classify_cache_state(ratio, source), ratio


async def _query_prefill_progress(
    srv,
    llama_port: int,
    model_name: str | None = None,
    slot_id: int | None = None,
    endpoint: str | None = None,
) -> tuple[int | None, bool]:
    """Observe llama-server prefill state: ``(progress, alive)``.

    Non-blocking: every query is wrapped in ``asyncio.wait_for`` using the
    same ``STATUS_QUERY_TIMEOUT`` (default 1.0s) pattern as the
    ``/llama/local/status`` endpoint, so the stream loop is never blocked
    waiting on llama-server.

    When *endpoint* is provided, the queries target that specific
    llama-server instance (LP-0MRPILSMW004T4H8). Otherwise the legacy
    ``localhost:{llama_port}`` URL is used.

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

            # LP-0MVP7525P008RXMF AC2: this probe runs *during* the prefill it
            # observes, so it must never traverse the router (the router
            # serializes /slots behind the busy child and 500s on cancel).
            _slots_endpoint, _slots_port = _resolve_local_slots_target(
                srv, model=model_name, endpoint=endpoint, router_port=llama_port
            )
            states = await asyncio.wait_for(
                _query_slots_progress(
                    _slots_port, timeout=timeout, model=model_name,
                    endpoint=_slots_endpoint,
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
    endpoint: str | None = None,
    llama_port: int,
    model_name: str | None = None,
    slot_id: int | None = None,
    last_progress: int = 0,
    prefill_in_flight: bool = False,
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

    When *endpoint* is provided, the dispatch record lookup uses a
    per-endpoint key so the correct server's lease is extended
    (LP-0MRPILSMW004T4H8).

    Extension triggers:

    - **Progress advance** — observed numeric progress (per-slot
      ``n_past``/``n_prompt_tokens_processed`` or aggregate
      ``kv_cache_tokens``) is greater than *last_progress*.
    - **Liveness** — the slot is observed processing (``is_processing``)
      while no numeric progress is reported (LP-0MUXQ5QL40027VZ4). This is
      the only usable signal on the deployed llama.cpp b8782 build, which
      removed the numeric per-slot fields. Without it, every prefill longer
      than ``local_dispatch_no_progress_timeout_seconds`` was released by
      the watchdog, its upstream stream cancelled, and the client retried
      into a fresh full prefill — an unbounded kill/retry loop.
    - **In-flight request** — *prefill_in_flight* is True, i.e. the caller
      still holds an open, error-free upstream request that is awaiting its
      first byte (LP-0MUXTW4MD003TVIC). This is the *only* reliable
      prefill-phase liveness signal under load: llama-server holds ``/slots``
      for the duration of a prompt-processing batch (measured 8.7s, then
      >15s), so the per-slot query times out at ``STATUS_QUERY_TIMEOUT`` and
      ``is_processing`` is unobservable; a queued request's slot additionally
      reports ``is_processing=False``. Relying on ``/slots`` alone left the
      extension a no-op in exactly the heavy-load case it targets. The hard
      ``local_dispatch_max_prefill_seconds`` ceiling remains the wedge bound.
      The in-flight trigger is accepted **only for the caller's own
      session** — ``local_prefill_in_flight`` is process-wide, so session A's
      open request must never refresh session B's lease
      (LP-0MVKUKFSO008GI6O AC2).

    Extension is scoped to the caller's own record: the record is addressed by
    ``_dispatch_lease_key(endpoint, session_key)``, which embeds the session,
    and only that record is mutated.

    An open request is **not** progress. The in-flight trigger pushes
    ``expires_at`` out but does **not** refresh ``last_progress_ts``, so a
    genuinely progress-less in-flight record is still released by the
    ``local_dispatch_no_progress_timeout_seconds`` (180s) watchdog
    (LP-0MVKUKFSO008GI6O AC4). Refreshing the no-progress clock on the
    in-flight signal alone is what sustained the 2026-10-23 ~4 h wedge:
    ``lease_extended_during_prefill progress=None alive=False in_flight=True``
    repeated every 10s defeated the watchdog while no byte was ever produced.
    Only observed numeric progress or observed slot liveness counts as
    progress (LP-0MUXQ5QL40027VZ4 liveness behaviour is unchanged).

    Liveness cannot hold a lease indefinitely: the
    ``local_dispatch_max_prefill_seconds`` ceiling in
    ``_watchdog_no_progress`` still releases a record once its total active
    time exceeds the ceiling, so a wedged slot that keeps reporting
    ``is_processing`` is bounded (LP-0MUCEFB8E003YVFF wedge protection).

    Returns ``(last_progress, extended)``:

    - *last_progress* is the latest numeric progress the caller should pass
      back on the next poll so numeric extension stops when progress stalls.
      When only liveness is observable the caller's previous baseline is
      returned unchanged (never ``None``).
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
        srv, llama_port, model_name=model_name, slot_id=slot_id,
        endpoint=endpoint,
    )
    advancing = progress is not None and progress > last_progress
    # LP-0MUXQ5QL40027VZ4: extend on a numeric progress advance OR on
    # observed liveness during the prefill phase. On the deployed b8782
    # build numeric progress is unobservable, so liveness is the only usable
    # signal; the max-prefill ceiling in `_watchdog_no_progress` still bounds
    # a wedged slot. Liveness-only extension was removed in
    # LP-0MUCEFB8E003YVFF on the assumption that numeric progress would be
    # available — it is not, so the removal turned every long prefill into a
    # watchdog kill.
    # LP-0MUXTW4MD003TVIC: extend while the upstream request is demonstrably
    # still in flight (``prefill_in_flight``), even when /slots is unobservable.
    # `/slots` blocks for the whole prompt-processing batch, so without this
    # the extension never fired under load and the watchdog killed every
    # prefill/queue wait at 180s. The max-prefill ceiling still bounds a wedge.
    # LP-0MVKUKFSO008GI6O AC2: the in-flight signal is only usable when the
    # marker belongs to *this* session — ``local_prefill_in_flight`` is a
    # process-wide map, so another session's open request must never keep this
    # record's lease alive ("truthful lease and in-flight accounting").
    in_flight_own = False
    if prefill_in_flight:
        try:
            _markers = getattr(srv, "local_prefill_in_flight", None)
            in_flight_own = bool(_markers) and session_key in _markers
        except Exception:
            in_flight_own = False
    if not advancing and not alive and not in_flight_own:
        # Unobservable and not processing: no extension. Keeps the adaptive
        # estimate applied at acquisition (fallback).
        return last_progress, False

    # Preserve the caller's numeric baseline when only liveness is observed.
    observed_progress = progress if progress is not None else last_progress

    extended = False
    try:
        lock = getattr(srv, "local_dispatch_records_lock", None)
        if lock is not None:
            async with lock:
                # LP-0MVKUKFSO008GI6O AC2: the extension is scoped to the
                # caller's own session record (``_dispatch_lease_key`` embeds
                # the session), so session A's dispatch can never refresh
                # session B's lease — the two-session isolation is guaranteed
                # by key construction, not by incidental absence of markers.
                record_key = _dispatch_lease_key(endpoint, session_key)
                record = srv.local_dispatch_records.get(record_key)
                if record is not None and record.get("active"):
                    record["expires_at"] = time.monotonic() + buffer_seconds
                    if progress is not None:
                        record["last_progress"] = progress
                    # LP-0MVKUKFSO008GI6O AC4: an open request is not
                    # progress. The in-flight signal (``prefill_in_flight``)
                    # extends the lease but must NOT refresh the no-progress
                    # clock, otherwise a wedged in-flight prefill defeats the
                    # 180s watchdog and holds the pool for hours (2026-10-23
                    # incident; ``last_progress_ts`` was refreshed every 10s
                    # with ``progress=None alive=False in_flight=True``). The
                    # clock advances only on observed numeric progress or
                    # liveness — a genuinely progress-less in-flight record is
                    # still released by ``_watchdog_no_progress``.
                    if advancing or alive:
                        record["last_progress_ts"] = time.monotonic()
                    extended = True
                    try:
                        srv.logger.info(
                            "lease_extended_during_prefill session=%s progress=%s "
                            "alive=%s in_flight=%s buffer=%.0fs",
                            session_key if session_key else "unknown",
                            progress,
                            alive,
                            in_flight_own,
                            buffer_seconds,
                        )
                    except Exception:
                        pass
    except Exception:
        pass
    return observed_progress, extended


async def _await_first_byte_with_prefill_monitor(
    srv,
    open_coro,
    *,
    timeout: float = 0.0,
    poll_seconds: float = 0.0,
    warn_seconds: float = 30.0,
    session_id: str | None = None,
    endpoint: str | None = None,
    llama_port: int = 8080,
    model_name: str | None = None,
    slot_id: int | None = None,
    request: Request | None = None,
    dispatch_session_id: str | None = None,
) -> tuple:
    """Await the initial upstream response while monitoring prefill progress.

    ``stream_cm.__aenter__()`` blocks until llama-server returns the SSE
    response headers, which for this build arrive with the first token.
    The post-headers prefill-progress poll in the streaming loop therefore
    never runs during a real prefill (LP-0MUCEFC0T009V7ZY).

    This helper races *open_coro* (the first-byte wait) against a periodic
    prefill-progress poll, so that:

    - Progress observed before the first byte extends the dispatch lease via
      ``_extend_lease_during_prefill`` (numeric progress advance OR observed
      slot liveness — LP-0MUXQ5QL40027VZ4; liveness was briefly removed in
      LP-0MUCEFB8E003YVFF and restored because b8782 exposes no numeric
      prefill progress). Crucially, because the poll below only runs while
      *open_task* is still pending, it also passes ``prefill_in_flight=True``
      — the in-flight request itself is the authoritative prefill-phase
      liveness signal when ``/slots`` is unobservable under load
      (LP-0MUXTW4MD003TVIC).
    - A distinct ``prefill_wait_exceeds_threshold`` warning is emitted once
      the initial wait exceeds *warn_seconds*, making a long pre-header wait
      visible in production (previously this path was silent).
    - *timeout* bounds the total wait (``local_dispatch_max_prefill_seconds``);
      exceeding it raises ``asyncio.TimeoutError`` so the caller can abort the
      request with a retryable status.

    *open_coro* is awaited directly (no polling, no timeout) when both
    *poll_seconds* and *timeout* are non-positive, preserving the previous
    behaviour for disabled configurations.

    LP-0MVLFVY6K003VYEL: when *request* is given, each poll tick also checks
    ``request.is_disconnected()``. A prefill emits no chunks, so the
    post-first-byte disconnect checks never run while the dispatch awaits its
    first byte — a client that died mid-prefill previously left its dispatch
    record and ``local_prefill_in_flight`` marker behind until the
    ``idle_timeout`` sweep (~5.5-6 min), refusing other sessions with
    ``reason=local_lease_active``. On a *confirmed* disconnect
    (``is_disconnected()`` returning True — never inferred from elapsed time)
    the dispatch is released via ``_release_client_disconnect_dispatch``
    (reason ``client_disconnect``) and ``ClientDisconnectedDuringPrefillError`` is
    raised. The check rides the existing poll cadence, so no extra task is
    created; with polling disabled (``poll_seconds <= 0``) the disconnect
    reaper's cancellation + verdict covers the same case.

    *dispatch_session_id* is the session key used for the release above and
    defaults to *session_id*; the router passes the real session id there even
    for non-explicit sessions, whose dispatch records are keyed by it.

    The pending *open_coro* task is cancelled if this helper is cancelled or
    raises, so a timed-out first-byte wait never leaks an upstream stream.
    """
    _release_key = dispatch_session_id or session_id
    open_task = asyncio.ensure_future(open_coro)
    waited = 0.0
    warned = False
    last_progress = 0
    try:
        while True:
            if timeout > 0:
                remaining = timeout - waited
                if remaining <= 0:
                    raise TimeoutError()
                step = min(poll_seconds, remaining) if poll_seconds > 0 else remaining
            else:
                step = poll_seconds if poll_seconds > 0 else None

            if step is None:
                # No polling and no timeout: await directly.
                return await open_task

            done, _pending = await asyncio.wait({open_task}, timeout=step)
            if open_task in done:
                return open_task.result()

            waited += step

            # Client-dead prefill release (LP-0MVLFVY6K003VYEL). A local
            # prefill emits no chunks, so the in-loop disconnect check in
            # ``stream_generator`` cannot see a client that disappeared while
            # the dispatch awaited its first byte. Poll on the same cadence as
            # the lease extension (no new background task) and release the
            # dispatch on a *confirmed* disconnect. Only an explicit
            # ``is_disconnected()`` True releases — never elapsed time — so a
            # live but slow prefill (>180 s) is never cut short here.
            if request is not None and poll_seconds > 0:
                _client_gone = False
                try:
                    _client_gone = await request.is_disconnected()
                except Exception:
                    _client_gone = False
                if isinstance(_client_gone, bool) and _client_gone:
                    try:
                        srv.logger.info(
                            "client_disconnect_during_prefill session=%s "
                            "waited=%.0fs slot=%s",
                            _release_key or "unknown",
                            waited,
                            slot_id,
                        )
                    except Exception:
                        pass
                    _release_client_disconnect_dispatch(
                        srv, _release_key, endpoint=endpoint, request=request
                    )
                    raise ClientDisconnectedDuringPrefillError(_release_key)

            # Poll prefill progress and extend the lease. The request is
            # still awaiting its first byte (open_task is pending), so it is
            # in flight: that alone is a valid liveness signal even when
            # /slots times out (LP-0MUXTW4MD003TVIC).
            if session_id:
                try:
                    last_progress, _extended = await _extend_lease_during_prefill(
                        srv,
                        session_id,
                        endpoint=endpoint,
                        llama_port=llama_port,
                        model_name=model_name,
                        slot_id=slot_id,
                        last_progress=last_progress,
                        prefill_in_flight=True,
                    )
                except Exception:
                    pass

            # Observable event once the initial wait exceeds the threshold.
            if warn_seconds > 0 and not warned and waited >= warn_seconds:
                warned = True
                try:
                    srv.logger.warning(
                        "prefill_wait_exceeds_threshold session=%s waited=%.0fs "
                        "progress=%s",
                        session_id or "unknown",
                        waited,
                        last_progress,
                    )
                except Exception:
                    pass
    except BaseException:
        if not open_task.done():
            open_task.cancel()
        raise


async def _decrement_local_active_queries(
    srv,
    session_key: str | None = None,
    backend: str | None = None,
) -> None:
    """Safely decrement the local-only active queries counter.

    When *session_key* is provided, the corresponding dispatch record
    (if any) is marked as inactive with a future *expires_at* timestamp,
    keeping the lease alive for the owner session until the timeout.

    The *backend* parameter specifies the endpoint URL of the llama-server
    instance; it is used to build a per-endpoint dispatch-lease key so the
    correct server's record is updated (LP-0MRPILSMW004T4H8).
    """
    try:
        async with srv.local_active_queries_lock:
            srv.local_active_queries = max(0, srv.local_active_queries - 1)
    except Exception as exc:
        session_hint = f" session={session_key}" if session_key else ""
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
                record_key = _dispatch_lease_key(backend, session_key)
                async with lock:
                    record_key = _dispatch_lease_key(backend, session_key)

                    if record_key in srv.local_dispatch_records:
                        srv.local_dispatch_records[record_key]["active"] = False
                        srv.local_dispatch_records[record_key]["expires_at"] = (
                            time.monotonic() + lease_timeout
                        )
                        try:
                            srv.logger.info(
                                "lease_renewed session=%s timeout=%.0fs",
                                _dispatch_key_session_id(record_key) if _dispatch_key_session_id(record_key) else "unknown",

                                lease_timeout,
                            )
                        except Exception as exc:
                            try:
                                srv.logger.warning(
                                    "Failed to log lease_renewed for session=%s: %s: %s",
                                    _dispatch_key_session_id(record_key) if _dispatch_key_session_id(record_key) else "unknown",

                                    type(exc).__name__,
                                    exc,
                                )
                            except Exception:
                                pass
        except Exception as exc:
            try:
                srv.logger.warning(
                    "Failed to mark dispatch record inactive for session=%s: %s: %s",
                    session_key if session_key else "unknown",
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
    admission_evidence: dict | None = None,
) -> None:
    """Safely increment the local-only active queries counter.

    When *session_key* and *backend* are provided, a corresponding
    dispatch record is created in *local_dispatch_records* to track
    lease ownership. The record is keyed by ``(endpoint, session_key)``
    so each llama-server instance tracks its concurrency independently
    (LP-0MRPILSMW004T4H8). *model_name* is stored on the record so orphan
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
                record_key = _dispatch_lease_key(backend, session_key)
                async with lock:
                    record_key = _dispatch_lease_key(backend, session_key)

                    srv.local_dispatch_records[record_key] = {
                        "backend": backend,
                        "started_at": time.monotonic(),
                        "active": True,
                        "expires_at": time.monotonic() + lease_timeout,
                        "model_name": model_name,
                        "endpoint": backend,
                        # LP-0MUCEFB8E003YVFF: no-progress watchdog baseline.
                        # A record that never makes progress is bounded by
                        # ``local_dispatch_no_progress_timeout_seconds`` from
                        # creation (no waiting on an unobservable signal).
                        "last_progress": 0,
                        "last_progress_ts": time.monotonic(),
                        # LP-0MVN4LCED00661MX AC2: the admission evidence the
                        # post-acceptance eviction ERROR reports.
                        # LP-0MVODJ3P6000AIUM AC1/AC4: verbatim admission-time
                        # snapshot when one was threaded; explicit ``None``
                        # markers (never a recompute) otherwise.
                        **_dispatch_evidence(
                            body_json,
                            model_name,
                            session_key,
                            admission_evidence=admission_evidence,
                        ),
                    }
                    _pin_dispatch_slot(
                        srv, record_key, session_key, backend
                    )
        except Exception:
            pass


def _get_generating_only_count(srv) -> int:
    """Return the current generating-only pool occupancy."""
    try:
        return max(0, int(getattr(srv, "local_generating_queries", 0) or 0))
    except Exception:
        return 0


DEFAULT_MAX_CONCURRENT_PREFILLS = 1
"""Default cap on concurrent local prefills (serialised prefills, LP-0MVGDUUV3001XFRJ)."""


def _max_concurrent_prefills(server_config: dict | None) -> int:
    """Return ``server.local_max_concurrent_prefills`` (0 = cap disabled).

    Measured 2026-10-19 → 10-20: 47.8 % of local prefills shared the GPU with at
    least one other prefill, halving per-request prefill throughput (p50 85 s
    overlapped vs 23.4 s solitary) while aggregate throughput stayed flat at
    ~254 tok/s — concurrency adds latency, not capacity. Prefills are therefore
    serialised by default.

    Validation (AC1): a missing value defaults to 1, ``0`` disables the cap
    (restoring the previous slot-pool-sized behaviour), and a malformed or
    negative value falls back to the default rather than silently disabling a
    latency protection.
    """
    if not isinstance(server_config, dict):
        return DEFAULT_MAX_CONCURRENT_PREFILLS
    raw = server_config.get(
        "local_max_concurrent_prefills", DEFAULT_MAX_CONCURRENT_PREFILLS
    )
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_CONCURRENT_PREFILLS
    if value < 0:
        return DEFAULT_MAX_CONCURRENT_PREFILLS
    return value


def _effective_prefill_cap(server_config: dict | None, max_local: int) -> int:
    """Return the effective prefill concurrency cap for a slot pool of *max_local*.

    The configured cap can never exceed the slot pool (a larger value is
    equivalent to no cap), and ``0`` means "disabled" — the pool size is then
    the cap, which is the behaviour before LP-0MVGDUUV3001XFRJ.
    """
    try:
        pool = max(1, int(max_local or 1))
    except (TypeError, ValueError):
        pool = 1
    configured = _max_concurrent_prefills(server_config)
    if configured <= 0:
        return pool
    return min(configured, pool)


def _server_config_of(srv) -> dict:
    """Return the ``server`` config block from *srv* (fail-open to ``{}``)."""
    try:
        cfg = srv.config
        return cfg.get("server", cfg) if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _get_prefill_in_flight_count(srv) -> int:
    """Return the number of sessions currently in prefill (dispatch→first-byte).

    Prefill-aware guard (LP-0MTJET4I5009EHNX): caps concurrent prefills to
    ``parallel`` (``session_slot_pool_size`` / ``max_local``) so the llama-
    server internal queue does not saturate under generating-only occupancy.
    """
    try:
        d = getattr(srv, "local_prefill_in_flight", None)
        if d is None:
            return 0
        return max(0, int(len(d)))
    except Exception:
        return 0


def _get_live_prefill_in_flight_count(srv) -> int:
    """Return the number of in-flight prefills backed by a live dispatch record.

    LP-0MVKUKFSO008GI6O AC3: a prefill-in-flight marker alone is not
    evidence that a prefill is genuinely in progress — it must be backed by a
    *live* dispatch record in ``local_dispatch_records``, i.e. an ``active``
    record (a dispatch that is still running: prefill or generation).

    An orphaned marker (record deleted by the prefill-deadline release, a
    non-explicit cleanup, an orphan sweep or a session eviction) and a marker
    whose record was retained only as an inactive lease (the pre-fix
    ``lease_renewed`` retention) must not inflate ``busy_slots`` /
    ``in_flight_prefills`` for the rest of the pool: the 2026-10-23 incident
    shows a single stuck marker raising ``available_slots=0`` for every other
    session while the dispatch it represented was long gone.

    Never raises: a routing decision must not break on an observability gap.
    """
    try:
        markers = getattr(srv, "local_prefill_in_flight", None)
        if not markers:
            return 0
        records = getattr(srv, "local_dispatch_records", None) or {}
        live_sessions: set[str] = set()
        for key, record in list(records.items()):
            if not isinstance(record, dict) or not record.get("active"):
                continue
            session_id = _dispatch_key_session_id(key)
            if session_id is not None:
                live_sessions.add(session_id)
        return max(0, sum(1 for s in markers if s in live_sessions))
    except Exception:
        return 0


def _live_prefill_remaining_seconds(
    srv,
    *,
    exclude_session: str | None = None,
    endpoint: str | None = None,
    now: float | None = None,
) -> float:
    """Total remaining prefill service owed by live in-flight prefills.

    LP-0MVOFVCA2008DQVE: the admission guard's occupancy term must not price an
    occupant at the *newcomer's* own prefill. Each dispatch record carries the
    calibrated prefill ETA captured when it was admitted
    (``_dispatch_evidence``: ``eta_prefill_s``) plus its ``started_at``
    (``time.monotonic()``).

    LP-0MVS9YK9P006M1NQ: the remaining service is priced from the occupant's
    remaining *uncached new tokens* at the contended co-residency rate instead
    of ``max(0, eta_prefill_s - (now - started_at))``. The recorded
    ``eta_prefill_s`` is the optimistic, uncontended admission ETA (measured
    2-4x optimistic when the other slot is generating), so decaying it
    under-priced the occupant and converged to ~0 the moment the occupant
    passed its optimistic ETA -- exactly when it was most expensive
    (2026-10-28 00:28:55: a cold 65-89k-token audit prompt priced
    ``eta_occupancy_s=16.0`` and the queue never freed). Each record also
    carries ``new_tokens`` / ``cached_ratio``/ ``model_name`` verbatim
    (LP-0MVODJ3P6000AIUM), so the remaining work can be re-priced:

    ``progress = min(1, elapsed / contended_full_service)`` -- progress is
    measured against the *contended* full duration of the occupant's whole new
    token load (``estimate_prefill_seconds(..., contended=True)``), never the
    optimistic ``eta_prefill_s``; ``remaining_new = new_tokens * (1 - progress)``;
    ``remaining_s = estimate_prefill_seconds(remaining_new, cached_ratio=0,
    contended=True)``.

    Only records that are *genuinely prefilling* are priced: ``active`` (a
    released/inactive lease is not consuming the gate) and no first byte yet
    (``first_byte_seen`` false -- once a byte is served the prefill is done).

    The requesting session's own record is excluded via *exclude_session*: a
    same-session re-entry never waits behind itself. When *endpoint* is given,
    records belonging to another endpoint are ignored (endpoint-aware pools).

    Fail-open by construction (LP-0MVS9YK9P006M1NQ AC3): a missing/invalid
    ``new_tokens``, ``eta_prefill_s`` or ``started_at``, malformed timestamps,
    a missing record set, or an uncalibrated contended window (where the
    estimator would have to *guess* the slowdown) all contribute nothing,
    leaving the symmetric ``eta * busy_slots`` floor in charge. Never raises --
    a routing decision must not break on an observability gap.
    """
    try:
        records = getattr(srv, "local_dispatch_records", None) or {}
        if not isinstance(records, dict) or not records:
            return 0.0
        _now = time.monotonic() if now is None else float(now)
        config = getattr(srv, "config", None)
        total = 0.0
        for key, record in list(records.items()):
            if not isinstance(record, dict):
                continue
            if not record.get("active"):
                continue
            if record.get("first_byte_seen"):
                continue
            session_id = _dispatch_key_session_id(key)
            if exclude_session is not None and session_id == exclude_session:
                continue
            record_endpoint = record.get("endpoint")
            if (
                endpoint is not None
                and record_endpoint is not None
                and record_endpoint != endpoint
            ):
                continue
            try:
                eta = float(record.get("eta_prefill_s"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(eta) or eta <= 0.0:
                continue
            try:
                started_at = float(record.get("started_at"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(started_at):
                continue
            try:
                new_tokens = int(record.get("new_tokens"))
            except (TypeError, ValueError):
                continue
            if new_tokens <= 0:
                continue
            # Lazy import: keeps router_helpers free of proxy.* module-level
            # dependencies (mirrors provider.py's lazy ``prefill_eta`` use).
            from proxy.prefill_eta import estimate_prefill_seconds

            elapsed = max(0.0, _now - started_at)
            full_contended_s, coeffs = estimate_prefill_seconds(
                record.get("model_name"),
                new_tokens,
                0.0,
                config,
                contended=True,
            )
            if not coeffs.calibrated:
                # Never price from a guess: an uncalibrated contended window
                # (with no calibrated solitary fallback either) leaves the
                # symmetric floor in charge, byte-for-byte the old fail-open.
                continue
            if full_contended_s <= 0.0:
                continue
            # Progress is measured against the contended full service of the
            # occupant's whole token load -- never the optimistic recorded
            # ETA, which would cap progress at 1.0 the moment the occupant
            # passed it and re-introduce the convergence-to-zero defect. The
            # ``max(full, eta)`` guard also keeps a conservative occupant ETA
            # from accelerating the decay.
            reference = max(full_contended_s, eta)
            progress = min(1.0, elapsed / reference) if reference > 0.0 else 1.0
            remaining_new = new_tokens * (1.0 - progress)
            remaining_s, _ = estimate_prefill_seconds(
                record.get("model_name"),
                int(round(remaining_new)),
                0.0,
                config,
                contended=True,
            )
            total += max(0.0, remaining_s)
        return max(0.0, total)
    except Exception:
        return 0.0


async def _increment_generating_only_slot(srv, session_key: str | None = None) -> None:
    """Increment generating-only slot once per session (idempotent).

    Prefill-aware guard: the session's prefill hold is released on first-byte
    so the prefill cap slot is freed for waiters.

    Adds *session_key* to ``local_generating_sessions`` (a set tracking which
    sessions are generating) and increments ``local_generating_queries``.
    This counter gates dispatch to the local model via ``_try_acquire_local_dispatch``
    — when it reaches ``max_local``, new dispatches are denied.

    Self-healing: if the corresponding decrement (``_decrement_generating_only_slot``)
    is skipped due to an exception in the streaming generator's finally block,
    the ``_dispatch_cleanup_loop`` periodically reclaims stale entries via
    ``_recover_stuck_generating_queries`` (LP-0MTYAWDCQ006RGYU).
    """
    # Prefill-aware guard: clear the prefill hold on first-byte.
    try:
        _p = getattr(srv, "local_prefill_in_flight", None)
        if _p is not None and session_key and session_key in _p:
            _plock = getattr(srv, "local_prefill_in_flight_lock", None)
            if _plock is not None:
                async with _plock:
                    _p.pop(session_key, None)
            else:
                _p.pop(session_key, None)
            try:
                from proxy.contention_queue import wake
                await wake(1)
            except Exception:
                pass
    except Exception:
        pass
    if not hasattr(srv, "local_generating_queries"):
        return
    # Ensure the per-session set exists so duplicate increments are idempotent
    # even for test SimpleNamespace fixtures that lack the attribute.
    if not hasattr(srv, "local_generating_sessions") or getattr(srv, "local_generating_sessions", None) is None:
        try:
            srv.local_generating_sessions = set()  # type: ignore[attr-defined]
        except Exception:
            pass
    try:
        lock = getattr(srv, "local_generating_queries_lock", None)
        sessions: set | None = getattr(srv, "local_generating_sessions", None)
        if lock is not None:
            async with lock:
                if session_key and sessions is not None and session_key in sessions:
                    return
                srv.local_generating_queries = int(getattr(srv, "local_generating_queries", 0) or 0) + 1
                if session_key and sessions is not None:
                    sessions.add(session_key)
        else:
            if session_key and sessions is not None and session_key in sessions:
                return
            srv.local_generating_queries = int(getattr(srv, "local_generating_queries", 0) or 0) + 1
            if session_key and sessions is not None:
                sessions.add(session_key)
    except Exception:
        pass


async def _decrement_generating_only_slot(srv, session_key: str | None = None) -> None:
    """Decrement generating-only slot for *session_key* (safe / not negative).

    Also clears any remaining prefill hold (prefill-only aborts).

    Called from the streaming generator's ``finally`` block (``router.py:1790``)
    after stream termination. If this call is skipped — e.g., due to an
    exception in the preceding slot-save path (``router.py:1700-1722``)
    — the session key leaks into ``local_generating_queries`` /
    ``local_generating_sessions`` and wedges the dispatch pool at capacity.

    Self-healing: the dispatch cleanup loop periodically reclaims such stale
    entries via ``_recover_stuck_generating_queries`` (LP-0MTYAWDCQ006RGYU).
    """
    # Clear prefill hold for prefill-only sessions (no generating).
    try:
        _p = getattr(srv, "local_prefill_in_flight", None)
        if _p is not None and session_key and session_key in _p:
            _plock = getattr(srv, "local_prefill_in_flight_lock", None)
            if _plock is not None:
                async with _plock:
                    _p.pop(session_key, None)
            else:
                _p.pop(session_key, None)
            try:
                from proxy.contention_queue import wake
                await wake(1)
            except Exception:
                pass
    except Exception:
        pass
    if not hasattr(srv, "local_generating_queries"):
        return
    if not hasattr(srv, "local_generating_sessions") or getattr(srv, "local_generating_sessions", None) is None:
        try:
            srv.local_generating_sessions = set()  # type: ignore[attr-defined]
        except Exception:
            pass
    try:
        lock = getattr(srv, "local_generating_queries_lock", None)
        sessions: set | None = getattr(srv, "local_generating_sessions", None)
        if lock is not None and sessions is not None:
            async with lock:
                if session_key and session_key not in sessions:
                    return
                srv.local_generating_queries = max(0, int(getattr(srv, "local_generating_queries", 0) or 0) - 1)
                if session_key:
                    sessions.discard(session_key)
        elif lock is not None:
            async with lock:
                srv.local_generating_queries = max(0, int(getattr(srv, "local_generating_queries", 0) or 0) - 1)
        else:
            if sessions is not None and session_key and session_key not in sessions:
                return
            srv.local_generating_queries = max(0, int(getattr(srv, "local_generating_queries", 0) or 0) - 1)
            if session_key and sessions is not None:
                sessions.discard(session_key)
    except Exception:
        pass
    try:
        from proxy.contention_queue import wake

        await wake(1)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Session tenancy bound (LP-0MVN1IQTF0037WTG AC2)
#
# Per-request dispatch leases bound *in-flight* requests but not *which*
# sessions occupy the pool over time: 27 distinct sessions cycled through 2
# slots in 5.4 h on 2026-10-24, each newcomer evicting the resident session's
# KV prefix. Tenancy bounds the number of distinct sessions that may hold a
# local reservation at once, so a further session's large prompt follows the
# normal chain instead of evicting a resident.
#
# A tenancy is a *lease*, not a lock: it is derived from the existing dispatch
# records (no new lifecycle), it is always re-acquirable by the session that
# already holds it (a session's own next turn is never blocked), and it
# idle-lapses so a genuinely idle resident cannot gate the pool forever.
# ---------------------------------------------------------------------------

#: Skip reason emitted when the tenancy bound refuses a new session.
TENANCY_SKIP_REASON: str = "tenant_limit"

#: Default idle-lapse for a tenancy (seconds). Derived from the adaptive
#: dispatch lease (``local_dispatch_lease_timeout_seconds``, 30 s base) so a
#: tenancy is bounded by the same lifecycle that already governs slot
#: ownership; in every case it stays well inside the 180 s no-progress
#: watchdog and the 900 s lifetime ceiling (AC6).
DEFAULT_TENANCY_IDLE_SECONDS: float = 30.0


@dataclass(frozen=True)
class TenancyDecision:
    """Outcome of the tenancy check for one session's large prompt."""

    admitted: bool
    reason: str | None
    occupants: tuple[str, ...] = ()
    tenant_limit: int = 0


def resolve_session_tenants(config: dict | None, max_local: int | None = None) -> int:
    """Resolve ``server.local_session_tenants`` (default: the slot count).

    ``0`` disables the bound (unlimited tenants), matching the
    "0 disables" convention of the other admission knobs. Malformed values
    fall back to the slot count so a typo can never widen the bound silently.
    """
    server_cfg = (config or {}).get("server") or {}
    raw = server_cfg.get("local_session_tenants")
    if raw is None:
        return _default_tenants(server_cfg, max_local)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        # Malformed: fall back to the slot count, never to "unlimited".
        return _default_tenants(server_cfg, max_local)
    return max(0, value)


def _default_tenants(server_cfg: dict, max_local: int | None) -> int:
    """Slot count used as the tenancy default ("N tenants, N slots")."""
    slots = max_local
    if slots is None:
        try:
            slots = int(server_cfg.get("session_slot_pool_size", 1) or 1)
        except (TypeError, ValueError):
            slots = 1
    return max(1, int(slots))


def resolve_tenancy_idle_seconds(config: dict | None) -> float:
    """Resolve ``server.local_session_tenancy_idle_seconds``.

    Defaults to the adaptive dispatch-lease base
    (``server.local_dispatch_lease_timeout_seconds``, itself defaulting to
    :data:`DEFAULT_TENANCY_IDLE_SECONDS`) so tenancy lapses on the same clock
    as the lease that holds the slot. A non-positive/malformed value falls back
    to the default (a zero would otherwise make tenancy lapse instantly and
    silently disable the bound).
    """
    server_cfg = (config or {}).get("server") or {}
    raw = server_cfg.get("local_session_tenancy_idle_seconds")
    if raw is None:
        raw = server_cfg.get("local_dispatch_lease_timeout_seconds")
    if raw is None:
        return DEFAULT_TENANCY_IDLE_SECONDS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_TENANCY_IDLE_SECONDS
    if value <= 0.0:
        return DEFAULT_TENANCY_IDLE_SECONDS
    return value


def evaluate_session_tenancy(
    srv,
    session_key: str | None,
    *,
    endpoint: str | None = None,
    tenants: int | None = None,
    idle_seconds: float | None = None,
    now: float | None = None,
    view=None,
) -> TenancyDecision:
    """Decide whether *session_key* may hold a local tenancy right now.

    Occupants are the distinct sessions that provably hold the pool:

    - sessions with a **live dispatch record** in the same endpoint scope
      (``active`` or an unexpired no-preemption lease), and
    - sessions whose slot is ``is_processing`` in the residency view (they are
      actively prefill/decode-ing, so their prefix is resident).

    A session already among the occupants is always admitted (same-session
    re-entry: a session's own next turn is never blocked behind itself). An
    occupant is dropped once its tenancy idle-lapses (``now - last activity >=``
    *idle_seconds*), so an abandoned resident cannot gate the pool.

    Fail-open: any unexpected error admits (admission must never break because
    a registry read went wrong), and ``tenants <= 0`` disables the bound.

    Returns:
        A :class:`TenancyDecision`. ``admitted`` False only when a *new*
        session would exceed the bound; ``reason`` is then
        :data:`TENANCY_SKIP_REASON`.
    """
    if tenants is None:
        tenants = resolve_session_tenants(None)
    if idle_seconds is None:
        idle_seconds = DEFAULT_TENANCY_IDLE_SECONDS
    if tenants <= 0 or not session_key:
        return TenancyDecision(True, None, (), tenants)

    current = time.monotonic() if now is None else float(now)
    ep_scope = endpoint or ""
    occupants: list[str] = []

    try:
        records = dict(getattr(srv, "local_dispatch_records", None) or {})
    except Exception:
        return TenancyDecision(True, None, (), tenants)

    # Snapshot the record dicts so a concurrent lease refresh cannot mutate a
    # record mid-decision.
    for key, record in records.items():
        if not isinstance(record, dict):
            continue
        key_endpoint = key[0] if isinstance(key, tuple) and key else ""
        if key_endpoint != ep_scope:
            continue
        owner = _dispatch_key_session_id(key)
        if not owner or owner in occupants:
            continue
        live = bool(record.get("active")) or float(
            record.get("expires_at") or 0.0
        ) > current
        if not live:
            continue
        last_activity = record.get("last_progress_ts") or record.get("started_at")
        try:
            lapsed = (
                last_activity is not None
                and (current - float(last_activity)) >= float(idle_seconds)
            )
        except (TypeError, ValueError):
            lapsed = False
        if lapsed:
            continue
        occupants.append(owner)

    if view is not None and getattr(view, "known", False):
        try:
            processing = view.processing_sessions(endpoint=endpoint)
        except Exception:
            processing = ()
        for owner in processing:
            if owner and owner not in occupants:
                occupants.append(owner)

    if session_key in occupants:
        return TenancyDecision(True, None, tuple(occupants), tenants)
    if len(occupants) >= tenants:
        return TenancyDecision(False, TENANCY_SKIP_REASON, tuple(occupants), tenants)
    return TenancyDecision(True, None, tuple(occupants), tenants)


async def _save_evicted_snapshot(
    srv,
    session_key: str | None,
    *,
    endpoint: str | None,
    plan,
    reason: str,
) -> None:
    """Detached: persist an evicted session's KV snapshot (AC6).

    Runs as a detached task so a reclaim path never blocks the cleanup cycle
    on a multi-second KV save (and never holds the dispatch-records lock while
    doing it). Fail-soft: a gated or failed save is reported by
    :func:`proxy.session.save_evicted_session_snapshot`, never raised.
    """
    try:
        from proxy import session as session_mod

        await session_mod.save_evicted_session_snapshot(
            session_key,
            _server_config_of(srv),
            endpoint=endpoint,
            reason=reason,
            plan=plan,
        )
    except Exception:
        pass


def _record_local_eviction(
    srv,
    session_key: str | None,
    *,
    endpoint: str | None = None,
    reason: str,
    slot_id: int | None = None,
    slot_owner_session: str | None = None,
) -> bool:
    """Emit the shared eviction event and schedule the snapshot save (AC6).

    One chokepoint for every tenancy/lease reclaim, so the ERROR-level
    ``local_eviction`` record (the convention owned by
    ``LP-0MVN3KMGI009BZ0D`` AC7, see :mod:`proxy.admission_events`) and the
    "save the evicted prefix so the client restores warm" step can never
    drift apart.

    Returns whether a snapshot save was planned (i.e. not gated off).
    """
    plan = None
    try:
        from proxy import session as session_mod

        plan = session_mod.plan_eviction_snapshot(
            session_key, _server_config_of(srv), endpoint=endpoint
        )
    except Exception:
        plan = None

    slot_owner = slot_owner_session or session_key
    try:
        from proxy.admission_events import log_local_eviction

        log_local_eviction(
            srv,
            session_key,
            reason,
            endpoint=endpoint,
            slot_id=slot_id if slot_id is not None else (plan[0] if plan else None),
            slot_owner_session=slot_owner,
            snapshot_planned=plan is not None,
        )
    except Exception:
        pass

    # LP-0MVT2QFAP0082BKR (AC eviction): the reclaimed slot no longer holds this
    # session's prefix, so its captured measurement is stale. Every reclaim
    # reason funnels through this chokepoint, so clearing here covers them all:
    # the next admission for the session must report
    # ``unavailable:no_prior_measurement`` instead of reusing a ratio measured
    # against a prefix the pool has dropped. The routed model is deliberately
    # not known here, so every model's measurement for the session goes.
    # Fail-open: clearing never raises and never changes a decision.
    try:
        from proxy.measured_prefill import clear_measured_prefill

        clear_measured_prefill(None, session_key)
    except Exception:
        pass

    if plan is not None:
        _spawn_detached_release(
            _save_evicted_snapshot(
                srv, session_key, endpoint=endpoint, plan=plan, reason=reason
            )
        )
    return plan is not None


def _dispatch_evidence(
    body_json: dict | None,
    model_name: str | None,
    session_key: str | None,
    *,
    admission_evidence: dict | None = None,
) -> dict:
    """Accepted-dispatch evidence for the post-acceptance eviction event.

    LP-0MVN4LCED00661MX AC2: the eviction ERROR must carry the numbers the
    admission model used, so a defect is diagnosable without log archaeology.
    Captured ONCE at record creation (cheap, best-effort) and never recomputed
    on release — the values must describe the decision that admitted the
    dispatch, not the state at eviction.

    LP-0MVODJ3P6000AIUM (AC1/AC4): the authoritative source is the admission-
    time snapshot threaded from the routing/admission decision
    (``admission_evidence``, stamped once on the request before the first
    local dispatch — see ``provider._stamp_local_admission_evidence``). The
    numbers are stored verbatim, never recomputed from the request body or a
    drifted cached-ratio sample at record creation.

    Fallback (AC4): when no admission decision is available (a record created
    outside the guarded routing path, e.g. legacy callers), the pricing fields
    are emitted as explicit ``None`` markers — the eviction event renders them
    as ``0``/``unknown`` — rather than silently recomputed from drifted inputs,
    so a ``new_tokens > budget`` field is unambiguous evidence of an over-
    budget admission and never a re-pricing artefact.

    ``body_json`` / ``model_name`` / ``session_key`` are retained for call-site
    compatibility only: the prompt is deliberately NOT re-estimated and the
    cached ratio is deliberately NOT re-sampled here (that recompute was the
    defect this fixes).
    """
    evidence: dict = {
        "estimated_tokens": None,
        "cached_ratio": None,
        "new_tokens": None,
        "eta_prefill_s": None,
        "slot_id": None,
        "llama_task_id": None,
        "first_byte_seen": False,
    }
    if not isinstance(admission_evidence, dict):
        return evidence
    try:
        estimated = admission_evidence.get("estimated_tokens")
        if estimated is not None:
            evidence["estimated_tokens"] = max(0, int(estimated))
        ratio = admission_evidence.get("cached_ratio")
        if ratio is not None:
            try:
                evidence["cached_ratio"] = min(1.0, max(0.0, float(ratio)))
            except (TypeError, ValueError):
                evidence["cached_ratio"] = None
        new_tokens = admission_evidence.get("new_tokens")
        if new_tokens is not None:
            evidence["new_tokens"] = max(0, int(new_tokens))
        eta = admission_evidence.get("eta_prefill_s")
        if eta is not None:
            try:
                value = float(eta)
                evidence["eta_prefill_s"] = value if value == value else None
            except (TypeError, ValueError):
                evidence["eta_prefill_s"] = None
    except Exception:
        pass
    return evidence


def mark_local_first_byte(srv, session_key: str | None, endpoint: str | None = None) -> None:
    """Flag the session's dispatch record as having served a first byte (AC3).

    After this the dispatch is no longer "accepted and unserved", so a later
    release is normal churn and must NOT be reported as an eviction error.
    Best-effort: a missing record or lock is a no-op.
    """
    try:
        key = _dispatch_lease_key(endpoint, session_key)
        records = getattr(srv, "local_dispatch_records", None)
        if records is None:
            return
        record = records.get(key)
        if record is None:
            # Legacy string-key form (pre per-endpoint keys).
            record = records.get(session_key) if session_key else None
        if record is not None:
            record["first_byte_seen"] = True
    except Exception:
        pass


def _emit_eviction_after_acceptance(
    srv,
    session_key: str | None,
    reason: str,
    record: dict | None,
    *,
    endpoint: str | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> bool:
    """Report an accepted-and-unserved dispatch release (AC1/AC7).

    Thin wrapper over :func:`proxy.admission_events.record_eviction_after_acceptance`
    so the release paths share one call shape; *extra_fields* carries the
    caller-specific measured evidence (LP-0MVR7IN0O000DPXN AC6). Never raises:
    an eviction path is already an exceptional path.
    """
    try:
        from proxy.admission_events import record_eviction_after_acceptance

        return record_eviction_after_acceptance(
            srv,
            session_key,
            reason,
            record,
            endpoint=endpoint,
            extra_fields=extra_fields,
        )
    except Exception:
        return False


async def _try_acquire_local_dispatch(
    srv,
    max_local: int,
    session_key: str,
    backend: str,
    body_json: dict | None = None,
    model_name: str | None = None,
    admission_evidence: dict | None = None,
    request=None,
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

    Post-deadline thrash gate (LP-0MVSAMN14001DMTY AC1/AC2): this is the
    **lease acquisition choke point**, so every path that can obtain a local
    dispatch lease — the provider chain, the provider's in-request local retry
    loops (``local_slot_exhaustion_retry`` / ``local_http_retry``), ``ui.py``'s
    direct dispatch, the contention-queue re-dispatch — passes through
    :func:`_local_dispatch_gate` *before* the record is created. A request
    whose local dispatch was cancelled by the prefill deadline is refused
    (``reason=post_deadline_terminal``) and a session whose post-deadline
    cooldown is armed for this prompt is refused
    (``reason=prefill_thrash_cooldown``), so the ~150 ms post-release
    re-acquisition observed on 2026-10-26 is impossible even on the paths that
    never consult provider.py's chain gate.
    """
    # Guard: skip if dispatch tracking is not initialised
    if not hasattr(srv, "local_dispatch_records") or not hasattr(srv, "local_dispatch_records_lock"):
        return (True, None, 0, 1.0)

    _gate_refused, _gate_reason, _gate_remaining = _local_dispatch_gate(
        srv,
        session_key,
        request=request,
        fingerprint=(
            _prefill_thrash_fingerprint(body_json)
            if isinstance(body_json, dict)
            else None
        ),
        entry="lease_acquire",
    )
    if _gate_refused:
        return (False, None, 0, max(1.0, _gate_remaining))

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
    record_key = _dispatch_lease_key(backend, session_key)
    # The record dict's key type tells us whether the deployment is endpoint-
    # aware (multi-backend, LP-0MRPILSMW004T4H8): once any per-endpoint tuple
    # key exists, per-endpoint records are the authoritative occupancy source
    # and the global counter is no longer consulted. In legacy single-server
    # mode (no tuple keys), the global ``local_active_queries`` counter keeps
    # its original deny semantics.
    has_endpoint_records = any(
        isinstance(k, tuple) for k in srv.local_dispatch_records
    )
    endpoint_scope = (
        backend if (isinstance(record_key, tuple) and backend and backend != "local")
        else ""
    )
    try:
        async with srv.local_active_queries_lock:
            async with srv.local_dispatch_records_lock:
                # Multi-backend per-endpoint scoping (LP-0MRPILSMW004T4H8):
                # record keys are ``(endpoint, session_key)``. Derive the
                # record key + endpoint scope so only records in the same
                # endpoint scope count toward occupancy.
                record_key = _dispatch_lease_key(backend, session_key)
                has_endpoint_records = any(
                    isinstance(k, tuple) for k in srv.local_dispatch_records
                )
                endpoint_scope = (
                    backend if (isinstance(record_key, tuple) and backend and backend != "local")
                    else ""
                )
                for existing_key, record in list(srv.local_dispatch_records.items()):
                    if not record.get("active") and record.get("expires_at", 0) <= now:
                        del srv.local_dispatch_records[existing_key]
                        try:
                            from proxy.session import _free_slot_assignment
                            _free_slot_assignment(_dispatch_key_session_id(existing_key))
                        except Exception:
                            pass

                own_record = srv.local_dispatch_records.get(record_key)
                own_has_lease = (
                    own_record is not None
                    and (
                        own_record.get("active")
                        or own_record.get("expires_at", 0) > now
                    )
                )

                occupied_by_others = 0
                first_occupied_owner = None
                for existing_key, record in srv.local_dispatch_records.items():
                    if existing_key == record_key:
                        continue
                    if has_endpoint_records:
                        # Endpoint-aware mode: only records in the same
                        # endpoint scope count toward this server's pool.
                        ek_endpoint = existing_key[0] if isinstance(existing_key, tuple) else ""
                        if ek_endpoint != endpoint_scope:
                            continue
                    if record.get("active") or record.get("expires_at", 0) > now:
                        occupied_by_others += 1
                        if first_occupied_owner is None:
                            first_occupied_owner = _dispatch_key_session_id(existing_key)
                _generating_count = _get_generating_only_count(srv)
                _prefill_count = _get_prefill_in_flight_count(srv)
                has_generating_state = (
                    hasattr(srv, "local_generating_queries")
                    or hasattr(srv, "local_generating_sessions")
                    or hasattr(srv, "local_generating_queries_lock")
                )
                if not own_has_lease:
                    if has_generating_state:
                        # Generating-only pool (LP-0MTH7JX82000YS5N): prefill
                        # dispatches do not count against the cap — only
                        # generating sessions do. Inactive leases (cooldown)
                        # are not counted here; they are covered by the
                        # streaming-phase counter.
                        if _generating_count >= max_local:
                            active_owner = None
                            for ek, er in srv.local_dispatch_records.items():
                                if ek != record_key and er.get("active"):
                                    active_owner = _dispatch_key_session_id(ek)
                                    break
                            if active_owner is None:
                                active_owner = first_occupied_owner
                            return (False, active_owner, _generating_count, max(1.0, lease_timeout))
                        # Prefill-aware guard (LP-0MTJET4I5009EHNX): cap
                        # concurrent prefills to ``parallel`` (max_local) so
                        # the llama-server internal queue does not saturate.
                        # Serialised by default since LP-0MVGDUUV3001XFRJ: at
                        # most ``local_max_concurrent_prefills`` (default 1)
                        # prefills share the GPU, because prefills do not scale
                        # (47.8 % shared the GPU, halving per-request prefill
                        # throughput at flat aggregate tok/s).
                        if _prefill_count >= _effective_prefill_cap(
                            _server_config_of(srv), max_local
                        ):
                            active_owner = None
                            for ek, er in srv.local_dispatch_records.items():
                                if ek != record_key and er.get("active"):
                                    active_owner = _dispatch_key_session_id(ek)
                                    break
                            if active_owner is None:
                                active_owner = first_occupied_owner
                            return (False, active_owner, _prefill_count, max(1.0, lease_timeout))
                        # Also enforce the traditional slot-cap when no
                        # generating backing state is present (legacy).
                    else:
                        if occupied_by_others >= max_local:
                            return (False, first_occupied_owner, occupied_by_others, max(1.0, lease_timeout))
                else:
                    # Lease-holding session re-acquisition (LP-0MU09CAL3001MP0Z).
                    # No-preemption: allow this session to re-acquire, but
                    # still respect the generating-only cap so that
                    # ``active`` never exceeds ``max_local``.
                    if has_generating_state and _generating_count >= max_local:
                        active_owner = None
                        for ek, er in srv.local_dispatch_records.items():
                            if ek != record_key and er.get("active"):
                                active_owner = _dispatch_key_session_id(ek)
                                break
                        if active_owner is None:
                            active_owner = first_occupied_owner
                        return (False, active_owner, _generating_count, max(1.0, lease_timeout))


                srv.local_active_queries += 1

                srv.local_dispatch_records[record_key] = {
                    "backend": backend,
                    "started_at": now,
                    "active": True,
                    "expires_at": now + lease_timeout,
                    "model_name": model_name,
                    "endpoint": backend,
                    # LP-0MUCEFB8E003YVFF: no-progress watchdog baseline.
                    "last_progress": 0,
                    "last_progress_ts": now,
                    # LP-0MVN4LCED00661MX AC2: admitted-dispatch evidence.
                    # LP-0MVODJ3P6000AIUM AC1/AC4: verbatim admission-time
                    # snapshot when one was threaded; explicit ``None`` markers
                    # (never a recompute) otherwise.
                    **_dispatch_evidence(
                        body_json,
                        model_name,
                        session_key,
                        admission_evidence=admission_evidence,
                    ),
                }
                # LP-0MVUVHAQP001IRKK: pin the slot identity on the record at
                # admission (before any release can free the assignment) so
                # every abandonment path can name the slot it gave up on.
                _pin_dispatch_slot(srv, record_key, session_key, backend)
                # Register prefill hold for the prefill-aware guard.
                try:
                    _p = getattr(srv, "local_prefill_in_flight", None)
                    if _p is not None:
                        _p[session_key] = True
                except Exception:
                    pass
                # Lease-owner retention priority (LP-0MVI2VRT2000V1AP): note
                # the grant so this session keeps contention-queue priority
                # for ``contention_queue_lease_grace_seconds`` after it
                # releases, instead of re-queueing behind a newcomer.
                try:
                    from proxy import contention_queue

                    contention_queue.note_local_grant(session_key)
                except Exception:
                    pass

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


async def _release_local_dispatch(srv, session_id: str, request: Request | None = None, endpoint: str | None = None) -> bool:
    """Explicitly release the dispatch lease for *session_id*.

    Removes the dispatch record from ``local_dispatch_records`` under
    the existing lock. Returns ``True`` if a record was removed, or
    ``False`` if no matching record existed (idempotent no-op).

    When *endpoint* is provided, only the per-endpoint record for that
    llama-server instance is released (LP-0MRPILSMW004T4H8); otherwise
    the legacy single-server record keyed by *session_id* is removed.

    This is the programmatic equivalent of what the
    ``POST /v1/leases/release`` endpoint provides, allowing callers
    (e.g. other internal components) to proactively release a lease
    without going through the HTTP layer.
    """
    removed = False
    record_key = _dispatch_lease_key(endpoint, session_id)
    try:
        async with srv.local_dispatch_records_lock:
            if record_key in srv.local_dispatch_records:
                del srv.local_dispatch_records[record_key]
                removed = True
                try:
                    srv.logger.info(
                        "lease_released session=%s reason=explicit_release",
                        session_id if session_id else "unknown",

                        extra=_client_identity_extra(request),
                    )
                except Exception:
                    pass
            elif endpoint and (endpoint, session_id) in srv.local_dispatch_records:
                # Belt-and-braces: tuple-key form for any caller that passes
                # the endpoint directly.
                del srv.local_dispatch_records[(endpoint, session_id)]
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
    try:
        _p = getattr(srv, "local_prefill_in_flight", None)
        if _p is not None and session_id and session_id in _p:
            _p.pop(session_id, None)
    except Exception:
        pass
    if session_id:
        try:
            from proxy.session import _free_slot_assignment
            _free_slot_assignment(session_id, endpoint=endpoint)
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


#: Machine-readable reasons reported by :func:`summarise_on_session_slot` and
#: logged by the local→remote boundary hook (parent LP-0MVRBQBDF0037OZ5 AC4).
BOUNDARY_REASON_COMPACT = "boundary_compact"
BOUNDARY_REASON_NO_SLOT = "boundary_no_slot"
BOUNDARY_REASON_SLOT_UNAVAILABLE = "boundary_slot_unavailable"
BOUNDARY_REASON_SUMMARIZER_FAILED = "boundary_summarizer_failed"


def _log_boundary_event(srv, level: str, message: str, **fields) -> None:
    """Emit a boundary-compaction line through ``srv.logger`` when present.

    Logging must never break dispatch, so a missing logger or a failing
    handler is swallowed (the request still proceeds remote untouched).
    """
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    try:
        logger = getattr(srv, "logger", None)
        if logger is None:
            return
        getattr(logger, level)(f"{message} {payload}".strip())
    except Exception:
        pass


async def summarise_on_session_slot(
    srv,
    session_id: str,
    messages: list[dict],
    *,
    config: dict | None = None,
    summarizer_builder=None,
    endpoint: str | None = None,
) -> dict:
    """Summarise *messages* on the local slot already assigned to *session_id*.

    Local→remote boundary helper (LP-0MVTDU91C003Q9YZ, parent AC2/AC3): the
    session is leaving the local model, so its own slot is disposable — the
    summary is produced there instead of on another session's warm slot or on
    a dedicated summariser backend.

    Ordering contract: the session's local dispatch lease is released the
    instant the summariser returns (success, failure **or** timeout) and
    therefore **before** the caller's remote stream starts — no local slot
    stays pinned to a session now served remotely.

    The slot is looked up with the **read-only**
    ``proxy.session._assigned_slot_for_session``; this helper never assigns a
    slot (a session without one is skipped, never given another session's).

    Args:
        srv: Proxy server object (provides ``local_dispatch_records`` and the
            optional ``logger``).
        session_id: Session whose slot is reused.
        messages: Messages to summarise (the turns being compacted away).
        config: Proxy config dict; defaults to ``srv.config``. Supplies the
            llama-server port and the summariser knobs — no new timeout knob
            is introduced (``compaction_summarizer_timeout`` bounds the call).
        summarizer_builder: Optional ``Callable[[int], Summarizer]`` called
            exactly once with the resolved slot id. Production defaults to
            ``build_slot_pinned_summarizer(config, slot_id, llama_port=…)``;
            tests inject fakes here.
        endpoint: Optional per-endpoint local llama-server URL, used for both
            the slot lookup and the lease release.

    Returns:
        ``{"slot_id", "summary", "reason", "released"}``. ``summary`` is the
        summary text on success and ``None`` otherwise; ``reason`` is one of
        ``boundary_compact``, ``boundary_no_slot``,
        ``boundary_slot_unavailable`` or ``boundary_summarizer_failed``.
    """
    cfg = config if isinstance(config, dict) else None
    if cfg is None:
        srv_config = getattr(srv, "config", None)
        cfg = srv_config if isinstance(srv_config, dict) else {}
    server_cfg = cfg.get("server") if isinstance(cfg.get("server"), dict) else {}
    try:
        llama_port = int(server_cfg.get("llama_server_port", 8080))
    except (TypeError, ValueError):
        llama_port = 8080
    if endpoint:
        try:
            _, endpoint_port = _parse_endpoint_url(endpoint)
            llama_port = endpoint_port
        except Exception:
            pass

    # Read-only slot lookup — never ``_slot_id_for_session`` (which assigns).
    slot_id = None
    try:
        from proxy.session import _assigned_slot_for_session

        if endpoint:
            slot_id = _assigned_slot_for_session(session_id, endpoint=endpoint)
        else:
            slot_id = _assigned_slot_for_session(session_id)
    except Exception:
        slot_id = None

    reason = BOUNDARY_REASON_NO_SLOT
    summary: str | None = None
    released = False
    try:
        if slot_id is None:
            # No slot of its own: never borrow another session's.
            _log_boundary_event(
                srv,
                "warning",
                "boundary_compaction_skipped",
                session=session_id,
                reason=BOUNDARY_REASON_NO_SLOT,
            )
        else:
            if summarizer_builder is not None:
                summarizer = summarizer_builder(slot_id)
            else:
                from proxy.compaction_summarizer import (
                    build_slot_pinned_summarizer,
                )

                summarizer = build_slot_pinned_summarizer(
                    cfg, slot_id, llama_port=llama_port
                )
            try:
                text = summarizer(messages)
            except httpx.TransportError:
                # Connection-level failure: the pinned slot could not be
                # reached at all.
                reason = BOUNDARY_REASON_SLOT_UNAVAILABLE
                _log_boundary_event(
                    srv,
                    "warning",
                    "boundary_compaction_skipped",
                    session=session_id,
                    slot=slot_id,
                    reason=reason,
                )
            except Exception as exc:
                reason = BOUNDARY_REASON_SUMMARIZER_FAILED
                _log_boundary_event(
                    srv,
                    "warning",
                    "boundary_compaction_skipped",
                    session=session_id,
                    slot=slot_id,
                    reason=reason,
                    error=type(exc).__name__,
                )
            else:
                text = text if isinstance(text, str) else None
                if text and text.strip():
                    summary = text
                    reason = BOUNDARY_REASON_COMPACT
                    _log_boundary_event(
                        srv,
                        "info",
                        "boundary_compaction_summarised",
                        session=session_id,
                        slot=slot_id,
                        reason=reason,
                    )
                else:
                    # Blank text and the fail-open ``EmptySummary`` sentinel
                    # both mean "no usable summary" — never injected.
                    reason = BOUNDARY_REASON_SUMMARIZER_FAILED
                    _log_boundary_event(
                        srv,
                        "warning",
                        "boundary_compaction_skipped",
                        session=session_id,
                        slot=slot_id,
                        reason=reason,
                    )
    finally:
        # Immediate release: this runs before the helper returns, so the
        # caller cannot start the remote stream while the slot is still held —
        # even when the summariser raised or timed out. It also runs on the
        # no-slot path so no stale lease survives.
        try:
            released = await _release_local_dispatch(
                srv, session_id, endpoint=endpoint
            )
        except Exception:
            released = False

    return {
        "slot_id": slot_id,
        "summary": summary,
        "reason": reason,
        "released": released,
    }


async def _release_unserved_dispatch(
    srv,
    session_key: str | None,
    endpoint: str | None = None,
    request: Request | None = None,
    *,
    release_reason: str,
    eviction_reason: str,
    extra_eviction_fields: Mapping[str, Any] | None = None,
    abandoned_slot_id: int | None = None,
    upstream_stream_cm: Any | None = None,
) -> tuple[bool, dict | None]:
    """Release an ACCEPTED-but-unserved dispatch lease and in-flight marker.

    Shared core of every "cancel the local dispatch before its first byte"
    path — the prefill deadline (LP-0MVKUKFSO008GI6O) and the measured
    cache-deficit early abort (LP-0MVR7IN0O000DPXN) — so the lease, marker,
    eviction snapshot, counters and contention-queue wake stay identical
    instead of drifting apart in two copies.

    Deletes the dispatch record (any key form — legacy string or per-endpoint
    tuple) and pops the marker for *session_key* (and only that session),
    emitting ``lease_released session=... reason=<release_reason>``. Emits the
    post-acceptance eviction snapshot with *eviction_reason* (and any extra
    measured evidence) before the slot assignment is freed, so a returning
    session restores its resident prefix warm.

    AC1 (LP-0MVRMI5HK00558TB): an abandoned dispatch must stop consuming the
    GPU and stop being reported as live work.

    - **Cancel the upstream stream** (*upstream_stream_cm*, or the released
      record's own pinned ``stream_cm``) exactly as the watchdog and
      client-disconnect releases do (LP-0MVIQ0FJC000YLX0). This build aborts
      at llama-server's next socket write — ``return_progress`` guarantees one
      during prefill — so the abort is prompt (evidence: 42 of 82
      ``Connection handling canceled`` events on 2026-10-28 cancelled a task
      that never reached ``prompt processing done``).
    - **Record the abandoned slot identity** (*abandoned_slot_id*, then the
      released record's pinned ``slot_id``, then the session's slot
      assignment — all captured before the assignment is freed) so a slot the
      backend has not released yet is annotated "busy, not ours" in the status
      payload and its ``capacity_hint`` is never ``saturated`` (never
      ``idle`` either). An identity that cannot be resolved records the
      endpoint-wide entry; a recorded-vs-observed mismatch logs
      ``abandoned_slot_mismatch`` **and** records the endpoint-wide entry
      (LP-0MVUVHAQP001IRKK).

    Returns ``(removed, released_record)``. Fail-soft: these are exceptional
    paths and must never raise. Wakes the cross-session contention queue when
    state was released.

    Ordering (LP-0MVSAMN14001DMTY AC3): the dispatch record and this session's
    ``local_prefill_in_flight`` marker are removed **first**, then the upstream
    stream is closed and *awaited* (bounded by
    ``server.disconnect_cleanup_timeout``) before returning. Both halves matter
    for a retry that lands immediately afterwards (the incident's 147-150 ms):
    ``_extend_lease_during_prefill`` needs an active record **and** the
    session's own marker, so nothing can re-extend a cancelled dispatch, and a
    marker can never outlive the dispatch it marks.
    """""
    if not session_key:
        return (False, None)
    removed = False
    released_record = None
    record_key = _dispatch_lease_key(endpoint, session_key)
    try:
        async with srv.local_dispatch_records_lock:
            if record_key in srv.local_dispatch_records:
                released_record = srv.local_dispatch_records.pop(record_key)
                removed = True
            elif endpoint and (endpoint, session_key) in srv.local_dispatch_records:
                # Belt-and-braces: tuple-key form for callers that pass the
                # endpoint directly (mirrors ``_release_local_dispatch``).
                released_record = srv.local_dispatch_records.pop(
                    (endpoint, session_key)
                )
                removed = True
            if removed:
                try:
                    srv.logger.info(
                        "lease_released session=%s reason=%s "
                        "endpoint=%s",
                        session_key if session_key else "unknown",
                        release_reason,
                        endpoint or "default",
                        extra=_client_identity_extra(request),
                    )
                except Exception:
                    pass
    except Exception:
        pass
    # Pop the per-session prefill marker (own session only). A cancelled
    # prefill is no longer in flight and must not count toward the prefill
    # cap / gate occupancy (LP-0MVKUKFSO008GI6O AC1 + AC3).
    marker_popped = False
    try:
        _p = getattr(srv, "local_prefill_in_flight", None)
        if _p is not None and session_key in _p:
            _plock = getattr(srv, "local_prefill_in_flight_lock", None)
            if _plock is not None:
                async with _plock:
                    _p.pop(session_key, None)
            else:
                _p.pop(session_key, None)
            marker_popped = True
    except Exception:
        pass
    if removed or marker_popped:
        # LP-0MVN1IQTF0037WTG AC6: a cancelled dispatch evicts the session's
        # lease. Record it through the shared eviction event and plan the KV
        # snapshot *before* the slot assignment is freed below, so the
        # session that retries restores its resident prefix warm.
        _record_local_eviction(
            srv,
            session_key,
            endpoint=endpoint,
            reason="lease_evicted",
        )
        # LP-0MVN4LCED00661MX AC1: an ACCEPTED dispatch cancelled before its
        # first byte is the defect signal the whole budget/pricing model
        # exists to prevent — reported at ERROR.
        _emit_eviction_after_acceptance(
            srv,
            session_key,
            eviction_reason,
            released_record,
            endpoint=endpoint,
            extra_fields=extra_eviction_fields,
        )
        try:
            from proxy.contention_queue import wake

            await wake(1)
        except Exception:
            pass
    # ── AC1 (LP-0MVRMI5HK00558TB): end the ghost, then make it attributable ──
    # 1. Cancel the upstream stream so llama-server stops computing work nobody
    #    is waiting for. The watchdog and client-disconnect paths already do
    #    this (LP-0MVIQ0FJC000YLX0); the deadline and measured-cache-deficit
    #    paths reach it through this shared core. The record's pinned
    #    ``stream_cm`` (set after the SSE headers arrive, LP-0MUGQHPJ50046LV8)
    #    is the fallback handle when the caller did not pass one.
    stream_cm = upstream_stream_cm
    if stream_cm is None and isinstance(released_record, dict):
        stream_cm = released_record.get("stream_cm")
    if stream_cm is not None:
        try:
            await asyncio.wait_for(
                _close_stream_cm(stream_cm),
                timeout=_abandoned_stream_close_timeout(srv),
            )
        except BaseException:
            pass
    # 2. Record the abandoned slot identity (bounded TTL) so the status payload
    #    can annotate a slot the backend has not released yet as "busy, not
    #    ours". The reason recorded is the eviction/release reason the caller
    #    already publishes, keeping logs and payload attributable to one word.
    #    An unresolvable identity records the endpoint-wide entry instead of
    #    dropping the abandonment, and a recorded-vs-observed mismatch is
    #    logged rather than matched silently (LP-0MVUVHAQP001IRKK).
    if removed or marker_popped:
        try:
            _record_abandoned_identity(
                srv,
                _resolve_abandoned_slot_identity(
                    session_key,
                    endpoint=endpoint,
                    explicit=abandoned_slot_id,
                    record=released_record,
                ),
                session_id=session_key,
                endpoint=endpoint,
                reason=release_reason,
            )
        except Exception:
            pass
    return (removed, released_record)


def _abandoned_stream_close_timeout(srv=None) -> float:
    """Bound on closing an abandoned upstream stream (never wedge a request).

    Best effort by construction: a stream that will not close must not delay
    the release path, so the ghost is annotated instead of awaited forever.

    The bound is the request's own ``disconnect_cleanup_timeout`` (default
    5 s, the value the deadline / measured-abort twins use for the same
    cleanup steps) so the release path honours one configured budget instead of
    a hard-coded copy of it (LP-0MVSAMN14001DMTY AC3). A missing, malformed or
    non-positive value keeps the 5 s default — a config typo must not make the
    close unbounded.
    """
    default = 5.0
    if srv is None:
        try:
            srv = _srv()
        except Exception:
            return default
    try:
        raw = _server_config_of(srv).get("disconnect_cleanup_timeout", default)
        timeout = float(raw)
    except Exception:
        return default
    return timeout if timeout > 0 else default


async def _release_prefill_deadline_dispatch(
    srv,
    session_key: str | None,
    endpoint: str | None = None,
    request: Request | None = None,
    *,
    abandoned_slot_id: int | None = None,
    upstream_stream_cm: Any | None = None,
) -> bool:
    """Release the dispatch lease + in-flight marker of a deadline-cancelled dispatch.

    LP-0MVKUKFSO008GI6O AC1: when ``prefill_deadline_exceeded`` cancels a
    local dispatch that never produced a first byte, the session's lease must
    be released on the spot — the pre-fix cleanup only marked the record
    inactive with a *future* ``expires_at`` (``lease_renewed``) and left the
    ``local_prefill_in_flight`` marker in place, so a wedged session re-
    acquired the lease for hours (2026-10-23 incident: ~4 h hold, 2,056
    other-session denials, ``available_slots=0``).

    Delegates to :func:`_release_unserved_dispatch` (shared with the measured
    cache-deficit early abort), emitting ``lease_released session=...
    reason=prefill_deadline`` and the ``local_eviction_after_acceptance
    reason=prefill_deadline`` snapshot. Returns True when a record was
    removed.

    AC1 (LP-0MVRMI5HK00558TB): the deadline used to only ``client.aclose()``
    the downstream connection, which llama-server does not act on until it
    next writes — so the prefill ran to completion as a ghost. The caller
    passes the captured upstream stream (*upstream_stream_cm*) and the slot id
    it dispatched to (*abandoned_slot_id*), so the shared core cancels the
    real task and records the abandoned slot for the status annotation.
    """
    removed, _record = await _release_unserved_dispatch(
        srv,
        session_key,
        endpoint=endpoint,
        request=request,
        release_reason="prefill_deadline",
        eviction_reason="prefill_deadline",
        abandoned_slot_id=abandoned_slot_id,
        upstream_stream_cm=upstream_stream_cm,
    )
    return removed


async def _release_measured_cache_abort_dispatch(
    srv,
    session_key: str | None,
    endpoint: str | None = None,
    request: Request | None = None,
    *,
    measured: Any | None = None,
    priced_new_tokens: float | None = None,
    abandoned_slot_id: int | None = None,
    upstream_stream_cm: Any | None = None,
) -> bool:
    """Release the lease of a dispatch aborted on a measured cache deficit.

    The early-abort twin of :func:`_release_prefill_deadline_dispatch`
    (LP-0MVR7IN0O000DPXN AC1/AC6/AC7): identical lease + marker + counter
    release, but logged as ``reason=measured_cache_deficit`` so the early
    detour is attributable and never counted as a 120 s deadline eviction. The
    eviction snapshot additionally carries the *measured* evidence
    (``measured_cache`` / ``total`` / ``priced_new_tokens``) the decision used.

    ``abandoned_slot_id`` / ``upstream_stream_cm`` are forwarded exactly like
    the deadline twin (LP-0MVUVHAQP001IRKK): the caller passes the dispatch
    record's pinned slot identity and the live upstream stream, so the upstream
    cancel and the abandonment annotation both fire on the **earliest** abort
    path — the one that used to pass neither and therefore relied on a
    post-free session lookup.

    Returns True when a record was removed.
    """
    from proxy.measured_cache_abort import MEASURED_CACHE_DEFICIT_REASON

    extra_fields: dict[str, Any] = {}
    try:
        total = getattr(measured, "total", None)
        cache = getattr(measured, "cache", None)
        if total is not None:
            extra_fields["total"] = total
        if cache is not None:
            extra_fields["measured_cache"] = cache
        if priced_new_tokens is not None:
            extra_fields["priced_new_tokens"] = priced_new_tokens
    except Exception:
        extra_fields = {}

    removed, _record = await _release_unserved_dispatch(
        srv,
        session_key,
        endpoint=endpoint,
        request=request,
        release_reason=MEASURED_CACHE_DEFICIT_REASON,
        eviction_reason=MEASURED_CACHE_DEFICIT_REASON,
        extra_eviction_fields=extra_fields or None,
        abandoned_slot_id=abandoned_slot_id,
        upstream_stream_cm=upstream_stream_cm,
    )
    return removed


@dataclass
class MeasuredAbortPrefetch:
    """Outcome of :func:`_prefetch_measured_cache_abort` (LP-0MVR7IN0O000DPXN).

    The router replays ``chunks`` (the raw upstream reads already consumed,
    in order) through its normal stream loop before reading anything new, so
    the peek never loses bytes and never changes the client-visible stream.
    """

    chunks: list[bytes]
    abort: bool = False
    decision: Any | None = None
    measured: Any | None = None
    complete: bool = False
    error: BaseException | None = None


async def _prefetch_measured_cache_abort(
    aiter: Any,
    reframer: Any,
    *,
    settings: Any | None = None,
    priced_new_tokens: float | None = None,
    timeout: float = 5.0,
    max_chunks: int = 1024,
    chunks_sink: list[bytes] | None = None,
) -> MeasuredAbortPrefetch:
    """Peek the local SSE stream for the first ``prompt_progress`` measurement.

    Option D (LP-0MVR7IN0O000DPXN AC1) needs the abort decision BEFORE the
    response is committed to the client — once headers/bytes have been sent a
    mid-stream failure can no longer detour the request to a remote provider.
    This helper reads ahead of the stream reframer: it consumes raw chunks
    (returned for byte-identical replay) until one of:

    - a complete reframed event carries ``prompt_progress`` → run the F3
      decision ONCE. Abort ⇒ ``abort=True`` with the decision and the
      :class:`proxy.measured_prefill.MeasuredPrefill` measurement. No abort ⇒
      keep peeking: the first event's ``cache`` is the final reused prefix, so
      later progress events add no decision information (F1/F3), but staying
      in the peek keeps ``local_prefill_deadline_seconds`` bounding the whole
      pre-first-byte wait (with ``return_progress`` the SSE headers arrive at
      prefill START, so no later code path enforces it);
    - a complete reframed non-progress event arrives → stop peeking and replay
      (that event is the first client-visible byte; the peek must not delay
      it — this also reproduces the pre-change "headers at first token"
      timing);
    - the stream ends (``complete=True``), a read raises (``error``), the
      deadline elapses (``asyncio.TimeoutError`` — the caller runs today's
      deadline detour), or the *max_chunks* bound is hit (continue).

    Fail-open by contract (AC5): an unexpected exception while observing an
    event degrades to *continue* with the consumed chunks replayed, never a
    raised error. ``settings``/``priced_new_tokens`` come from the router's
    fail-open-resolved configuration; ``None`` prices simply continue.

    *chunks_sink* lets the caller see every raw read even when this coroutine
    is cancelled (the router bounds the peek with the same pre-first-byte
    monitor it uses for the header wait, LP-0MVR7IXAI002QXXD). The caller then
    replays those bytes through a fresh ``SSEEventReframer``, so a cancelled
    peek never loses or re-orders a byte.

    *aiter* is the dispatch's ``aiter_bytes()`` iterator and *reframer* its
    :class:`proxy.utils.SSEEventReframer`, both owned by the caller: the peek
    must consume the SAME cursor and event buffer the normal stream loop
    continues with, otherwise a split event would be lost (or the body
    re-read). Neither is re-created here.
    """
    from proxy.measured_cache_abort import (
        AbortSettings,
        decide_measured_cache_abort,
    )
    from proxy.measured_prefill import MeasuredPrefill
    from proxy.prompt_progress import PromptProgressCapture

    effective = AbortSettings() if settings is None else settings
    if not isinstance(effective, AbortSettings) or not effective.enabled:
        return MeasuredAbortPrefetch(chunks=[], abort=False)

    chunks: list[bytes] = [] if chunks_sink is None else chunks_sink
    capture = PromptProgressCapture()
    decided_measured = None
    deadline_at = time.monotonic() + timeout if timeout > 0 else None

    while len(chunks) < max_chunks:
        remaining = None
        if deadline_at is not None:
            remaining = deadline_at - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()
        read_task = asyncio.ensure_future(aiter.__anext__())
        try:
            done, _pending = await asyncio.wait({read_task}, timeout=remaining)
        except BaseException:
            read_task.cancel()
            try:
                await read_task
            except BaseException:
                pass
            raise
        if not done:
            read_task.cancel()
            try:
                await read_task
            except BaseException:
                pass
            raise TimeoutError()
        try:
            chunk = read_task.result()
        except StopAsyncIteration:
            return MeasuredAbortPrefetch(chunks=chunks, complete=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # read error — fail open, replay + raise later
            return MeasuredAbortPrefetch(chunks=chunks, error=exc)
        chunks.append(chunk)

        try:
            for event in reframer.feed(chunk):
                progress = capture.observe(event)
                if progress is None:
                    # First complete client-visible event: stop peeking so the
                    # replay releases it immediately (AC2 latency).
                    return MeasuredAbortPrefetch(
                        chunks=chunks, measured=decided_measured
                    )
                if decided_measured is not None:
                    # Only the FIRST progress event decides (its ``cache`` is
                    # the final reused prefix); later ones add no decision
                    # information, so keep peeking for the deadline bound.
                    continue
                measured = MeasuredPrefill.from_progress(
                    progress, priced_new_tokens=priced_new_tokens
                )
                decided_measured = measured
                decision = decide_measured_cache_abort(
                    progress, priced_new_tokens, effective
                )
                if decision.abort:
                    return MeasuredAbortPrefetch(
                        chunks=chunks, abort=True, decision=decision, measured=measured
                    )
                # No abort: keep peeking so ``local_prefill_deadline_seconds``
                # keeps bounding the whole pre-first-byte wait.
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            raise
        except Exception:  # AC5: an observation error degrades
            return MeasuredAbortPrefetch(chunks=chunks, measured=decided_measured)
        # No complete event yet (partial/empty read): keep reading.
    return MeasuredAbortPrefetch(chunks=chunks, measured=decided_measured)


class ClientDisconnectedDuringPrefillError(Exception):
    """The client that dispatched a local request is gone before first byte.

    Raised by ``_await_first_byte_with_prefill_monitor`` when it observes a
    confirmed ``request.is_disconnected()`` inside the prefill/first-byte
    wait (LP-0MVLFVY6K003VYEL). The dispatch record, the
    ``local_prefill_in_flight`` marker and the session's slot assignment have
    already been released when this is raised, and the pending upstream
    first-byte wait is cancelled by the monitor, so the caller only has to
    finish its own accounting (counters, client close) and stop serving a
    client that is not there any more.
    """

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id
        super().__init__(
            f"client disconnected during prefill (session={session_id or 'unknown'})"
        )


_detached_release_tasks: set[asyncio.Task] = set()
"""Strong references to fire-and-forget release tasks.

``_release_client_disconnect_dispatch`` must not await anything (a second
cancellation would skip an awaiting release — LP-0MVLFVY6K003VYEL AC3), so
its upstream-stream close and contention-queue wake are scheduled as
detached tasks. Holding a reference keeps the event loop from garbage
collecting them mid-flight; each task discards itself on completion.
"""


def _reap_detached_release(task: asyncio.Task) -> None:
    """Drop a completed detached release task, retrieving its exception."""
    _detached_release_tasks.discard(task)
    try:
        task.exception()
    except BaseException:
        pass


def _spawn_detached_release(coro) -> None:
    """Schedule *coro* as a detached task (never awaited by the caller).

    Used by the client-disconnect release so it can complete synchronously
    (no ``await`` = no cancellation point). Fail-soft: without a running
    loop the coroutine is closed instead of leaking an "never awaited"
    warning.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        try:
            coro.close()
        except Exception:
            pass
        return
    try:
        task = loop.create_task(coro)
    except Exception:
        try:
            coro.close()
        except Exception:
            pass
        return
    _detached_release_tasks.add(task)
    task.add_done_callback(_reap_detached_release)


async def _close_stream_cm(stream_cm) -> None:
    """Close a captured upstream stream context manager (best effort)."""
    try:
        await stream_cm.__aexit__(None, None, None)
    except BaseException:
        pass


def _release_client_disconnect_dispatch(
    srv,
    session_key: str | None,
    *,
    endpoint: str | None = None,
    request: Request | None = None,
) -> bool:
    """Release a local dispatch whose client died before the first byte.

    LP-0MVLFVY6K003VYEL: a client that disappears mid-prefill (agent killed,
    CLI CTRL-C, network drop) produces no chunks, so none of the
    post-first-byte disconnect checks can see it. The dispatch record then
    survived as an inactive lease with a fresh ``expires_at`` plus its
    ``local_prefill_in_flight`` marker, and other sessions were refused with
    ``local_dispatch_denied ... reason=local_lease_active`` until the cleanup
    loop's ``idle_timeout`` sweep fired ~5.5-6 minutes later (observation
    2026-10-23 20:58-21:04). This helper releases the dispatch on the spot:

    - deletes the dispatch record (legacy string **and** per-endpoint tuple
      key forms), so the slot is no longer occupied;
    - pops this session's ``local_prefill_in_flight`` marker (own session
      only) so the prefill cap / admission guard stop counting it;
    - frees the session's slot assignment so the slot registry can be
      reused;
    - cancels the captured upstream stream (``stream_cm``) so llama-server
      is not left prefilling for a client that cannot consume the tokens —
      the same behaviour as the prefill-deadline and watchdog releases;
    - wakes the cross-session contention queue so a queued session is
      admitted immediately;
    - logs ``lease_released session=... reason=client_disconnect`` — distinct
      from ``disconnect`` (stream-end path), ``prefill_deadline``,
      ``idle_timeout`` and ``no_progress_watchdog``, so the recovery path is
      attributable.

    **Synchronous by design** (AC3): the release contains no ``await``, so a
    second ``CancelledError`` delivered to the reaper-cancelled request task
    can never interrupt it half-way (the pre-fix ``finally`` cleanup awaited
    the records lock and ``stream_cm.__aexit__`` and could be skipped
    entirely). The two awaiting steps — closing the upstream stream and
    waking the contention queue — are scheduled as detached tasks. Dict
    mutations without an ``await`` cannot interleave with the lock-holding
    readers on a single-threaded event loop, so skipping
    ``local_dispatch_records_lock`` here is safe and is what makes the
    release cancellation-proof.

    Idempotent: a missing record/marker is a no-op and logs nothing, so the
    disconnect release racing the deadline release, the watchdog or the
    orphan sweep logs at most once. Returns True when a record or marker was
    released.

    The per-request counters (``active_queries``, ``local_active_queries``,
    the per-model counter) are deliberately **not** touched here: the caller
    pairs this release with ``_cleanup_after_request(..., decrement_local=
    True)``, which is the single owner of those decrements. Decrementing in
    both places would release another session's capacity
    (``local_active_queries`` is a pool-wide counter).
    """
    if not session_key:
        return False

    removed = False
    stream_cm = None
    released_record = None
    records = getattr(srv, "local_dispatch_records", None)
    if records is not None:
        try:
            keys = [_dispatch_lease_key(endpoint, session_key)]
            if endpoint:
                # Belt-and-braces tuple-key form (mirrors
                # ``_release_prefill_deadline_dispatch``).
                keys.append((endpoint, session_key))
            for key in keys:
                record = records.pop(key, None)
                if isinstance(record, dict):
                    released_record = record
                    stream_cm = record.get("stream_cm")
                    removed = True
                    break
        except Exception:
            removed = False

    if removed:
        try:
            srv.logger.info(
                "lease_released session=%s reason=client_disconnect endpoint=%s",
                session_key,
                endpoint or "default",
                extra=_client_identity_extra(request),
            )
        except Exception:
            pass

    marker_popped = False
    try:
        markers = getattr(srv, "local_prefill_in_flight", None)
        if markers is not None and session_key in markers:
            markers.pop(session_key, None)
            marker_popped = True
    except Exception:
        marker_popped = False

    # AC1 (LP-0MVRMI5HK00558TB): resolve the slot identity **before** the
    # assignment is freed below (the released record's pinned ``slot_id`` first,
    # then the live assignment), so the abandonment is attributable to a slot —
    # or, when neither resolves, to an endpoint-wide entry
    # (LP-0MVUVHAQP001IRKK).
    identity = _resolve_abandoned_slot_identity(
        session_key, endpoint=endpoint, record=released_record
    )

    if removed or marker_popped:
        # LP-0MVN1IQTF0037WTG AC6: an observed client disconnect releases the
        # reservation immediately (one cleanup cycle, no waiting). Record the
        # eviction through the shared event and plan the KV snapshot *before*
        # the slot assignment is freed below, so a client that reconnects
        # restores its resident prefix warm.
        _record_local_eviction(
            srv,
            session_key,
            endpoint=endpoint,
            reason="lease_evicted",
        )
        # LP-0MVN4LCED00661MX AC1 (``lease_evicted``): an accepted dispatch
        # whose lease was reclaimed before the first byte.
        _emit_eviction_after_acceptance(
            srv,
            session_key,
            "lease_evicted",
            released_record,
            endpoint=endpoint,
        )
        try:
            from proxy.session import _free_slot_assignment

            _free_slot_assignment(session_key, endpoint=endpoint)
        except Exception:
            pass

    if stream_cm is not None:
        _spawn_detached_release(_close_stream_cm(stream_cm))

    # AC1 (LP-0MVRMI5HK00558TB): a client that vanished mid-prefill abandoned
    # this dispatch. The slot identity is captured before the assignment was
    # freed above; recording it lets the status payload annotate a slot the
    # backend has not released yet as "busy, not ours". An unresolvable
    # identity records the endpoint-wide entry instead of dropping it
    # (LP-0MVUVHAQP001IRKK).
    if removed or marker_popped:
        _record_abandoned_identity(
            srv,
            identity,
            session_id=session_key,
            endpoint=endpoint,
            reason="client_disconnect",
        )

    if removed or marker_popped:
        try:
            from proxy.contention_queue import wake

            _spawn_detached_release(wake(1))
        except Exception:
            pass

    return removed or marker_popped


async def _query_slot_processing(srv, session_id: str, model_name: str | None, endpoint: str | None = None) -> bool:
    """Check whether the session's llama-server slot is still processing.

    Queries llama-server ``/slots`` (via ``_query_slots_progress``) for the
    slot assigned to *session_id* and returns its ``is_processing`` flag —
    the only per-slot liveness signal llama.cpp b8782 exposes. Used by
    orphan cleanup to avoid freeing the dispatch lease / slot registry
    entry for a stream that is still generating on the backend
    (LP-0MSUO6XRP001MCB2).

    When *endpoint* is provided (a full ``http://host:port`` URL), the
    query targets that llama-server instance so per-endpoint dispatch
    records are verified against the correct server (LP-0MRPILSMW004T4H8).

    Returns False (probe inconclusive) when the session has no slot
    assignment, no *model_name* is known, or the query fails/times out.
    The caller must apply fail-closed semantics (LP-0MVIQ0FJC000YLX0):
    if the record has recent ``last_progress_ts``, treat it as alive
    rather than orphaning on an inconclusive probe.
    """
    if not session_id or not model_name:
        return False
    try:
        from proxy.session import _assigned_slot_for_session

        if endpoint:
            slot_id = _assigned_slot_for_session(session_id, endpoint=endpoint)
        else:
            # Legacy single-server path — call with session_id only so tests
            # that mock the helper with a 1-arg lambda keep working.
            slot_id = _assigned_slot_for_session(session_id)
        if slot_id is None:
            return False  # no slot assignment — cannot verify

        server_cfg = srv.config.get("server", {})
        llama_port = server_cfg.get("llama_server_port", 8080)
        if endpoint:
            _, llama_port = _parse_endpoint_url(endpoint)

        from proxy.observability import _query_slots_progress

        timeout = float(os.environ.get("STATUS_QUERY_TIMEOUT", "1.0"))
        # LP-0MVP7525P008RXMF AC2: lease-liveness probes must not traverse the
        # router — resolve to the discovered child port (the pool of this probe
        # is exactly the giant prefill that makes the router 500).
        _slots_endpoint, _slots_port = _resolve_local_slots_target(
            srv, model=model_name, endpoint=endpoint, router_port=llama_port
        )
        states = await asyncio.wait_for(
            _query_slots_progress(
                _slots_port, timeout=timeout, model=model_name,
                endpoint=_slots_endpoint,
            ),
            timeout=timeout + 0.5,
        )
        state = states.get(slot_id)
        if isinstance(state, dict):
            return bool(state.get("processing", False))
        return False
    except Exception:
        return False


def _watchdog_no_progress(srv, session_key: str, record: dict) -> bool:
    """Check whether a dispatch record has exceeded the no-progress timeout.

    Returns ``True`` when the record qualifies for no-progress release:

    - The record is ``active=True`` and carries a ``last_progress_ts``
      baseline (always set at record creation by
      ``_try_acquire_local_dispatch`` / ``_increment_local_active_queries``).
    - ``local_dispatch_no_progress_timeout_seconds`` is non-zero and the
      record's ``last_progress_ts`` is older than the timeout — meaning no
      progress has been observed for that long.
    - Or the record's ``started_at`` exceeds ``local_dispatch_max_prefill_seconds``
      (hard ceiling on any single request's lifetime, even with progress).

    Records without a ``last_progress_ts`` baseline (legacy records created
    before this feature) return ``False`` — they fall through to the
    pre-existing expiry/orphan-cleanup path rather than being released by the
    watchdog. This keeps the change surgical and avoids misclassifying
    records whose age was never tracked on a progress clock.

    When ``True``, the caller should release the lease so the session can
    fall through to the next backend (LP-0MUCEFB8E003YVFF).
    """
    no_progress_timeout = _get_no_progress_timeout_seconds(srv)
    max_prefill = _get_max_prefill_seconds(srv)
    now = time.monotonic()

    # The watchdog only has an opinion about records whose progress clock it
    # owns (``last_progress_ts``). This is set at creation and refreshed on
    # every observed progress advance, every observed prefill liveness signal
    # (LP-0MUXQ5QL40027VZ4), every in-flight prefill refresh
    # (LP-0MUXTW4MD003TVIC), and every data chunk. A record that is being
    # liveness- or in-flight-extended therefore falls through to the
    # ``max_prefill`` ceiling below, which bounds a wedged slot that keeps
    # reporting ``is_processing`` or never answers.
    last_progress_ts = record.get("last_progress_ts")
    if last_progress_ts is None:
        return False

    # No-progress timeout: no observed progress for longer than the timeout.
    if no_progress_timeout > 0 and now - last_progress_ts > no_progress_timeout:
        return True

    # Max-prefill ceiling: total active time exceeds the ceiling.
    if max_prefill > 0:
        started_at = record.get("started_at")
        if started_at is not None and now - started_at > max_prefill:
            return True  # exceeded max prefill ceiling

    return False


async def _cleanup_stale_local_dispatch(srv) -> int:
    """Remove stale lease records from *local_dispatch_records*.

    Three categories of stale records are cleaned:

    0. **Active records with no observed progress** (LP-0MUCEFB8E003YVFF) —
       the no-progress watchdog releases any active record whose
       ``last_progress_ts`` is older than
       ``local_dispatch_no_progress_timeout_seconds``, or whose
       ``started_at`` exceeds ``local_dispatch_max_prefill_seconds``, even
       when ``expires_at`` is still in the future (a wedged request's
       adaptive lease can be far out). Logged at WARNING level with
       ``reason=no_progress_watchdog``.

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
       (LP-0MSUO6XRP001MCB2). LP-0MVIQ0FJC000YLX0: also extended when the
       /slots probe is inconclusive (fail-closed) but the record has recent
       ``last_progress_ts``. Genuinely idle slots are orphan-cleaned as
       before, logged at WARNING level with ``reason=orphan_cleanup`` plus
       an INFO-level ``lease_released reason=orphan_cleanup`` for parity.
       On genuine orphan cleanup, the upstream HTTP stream is cancelled
       (``stream_cm.__aexit__``) so the llama-server slot is actually freed.
       with existing log consumers.

    Active records whose *expires_at* is still in the future are preserved
    (legitimate in-flight requests).

    Returns the number of records removed.
    """
    now = time.monotonic()
    removed = 0
    verify_candidates: list[tuple[str, dict]] = []
    # LP-0MUGQHPJ50046LV8: deferred watchdog-released records (stream_cm, slot
    # freeing, preflight tracking) — processed after releasing the lock.
    deferred_watchdog: list[tuple[str, str, str | None, object]] = []

    # Phase 1 (under lock): collect expired records. Inactive records are
    # freed immediately; expired ACTIVE records are deferred to phase 3 so
    # the slot liveness check runs OUTSIDE the records lock (it performs
    # an HTTP query to llama-server). Records are keyed per-endpoint
    # (LP-0MRPILSMW004T4H8); ``_dispatch_key_session_id`` recovers the
    # session portion for logging and slot-registry lookup.
    try:
        async with srv.local_dispatch_records_lock:
            for sid, record in list(srv.local_dispatch_records.items()):
                active = record.get("active", False)
                # LP-0MUCEFB8E003YVFF: no-progress watchdog. Runs on every
                # active record regardless of ``expires_at`` — a wedged
                # record's lease may be far in the future (adaptive lease),
                # so expiry alone cannot catch it. Pure time computation, so
                # safe to run under the lock.
                if active and _watchdog_no_progress(
                    srv, _dispatch_key_session_id(sid), record
                ):
                    session_key = _dispatch_key_session_id(sid)
                    endpoint_from_record = _endpoint_from_record(record)
                    # LP-0MUGQHPJ50046LV8: capture the upstream stream context
                    # manager before deleting the record so we can cancel the
                    # in-flight request after releasing the lock.
                    stream_cm = record.pop("stream_cm", None)
                    del srv.local_dispatch_records[sid]
                    removed += 1
                    # LP-0MVN4LCED00661MX AC1: the record is captured before
                    # deletion so the eviction event carries its evidence.
                    released_record = dict(record)
                    # Decrement counter while still under the lock for
                    # atomicity (LP-0MUCEFB8E003YVFF).
                    try:
                        srv.local_active_queries = max(
                            0, int(getattr(srv, 'local_active_queries', 0) or 0) - 1
                        )
                    except Exception:
                        pass
                    try:
                        srv.logger.warning(
                            "lease_released session=%s reason=no_progress_watchdog "
                            "endpoint=%s",
                            session_key if session_key else "unknown",
                            endpoint_from_record or "default",
                        )
                    except Exception:
                        pass
                    # Deferred: free slot assignment and preflight tracking
                    # after the lock, and close the upstream stream.
                    deferred_watchdog.append(
                        (
                            sid,
                            session_key,
                            endpoint_from_record,
                            stream_cm,
                            released_record,
                        )
                    )
                    continue

                expires_at = record.get("expires_at", 0)
                if expires_at > now:
                    continue  # still within valid window

                if not active:
                    # Normal idle timeout for inactive records
                    idle_record = dict(record)
                    del srv.local_dispatch_records[sid]
                    removed += 1
                    # LP-0MVN1IQTF0037WTG AC6: record the eviction and plan
                    # the KV snapshot BEFORE the slot assignment is released
                    # (the plan needs it); the slot is then freed for reuse.
                    _record_local_eviction(
                        srv,
                        _dispatch_key_session_id(sid),
                        endpoint=_endpoint_from_record(record),
                        reason="slot_assignment_freed",
                    )
                    # Free the slot registry entry so the slot can be
                    # reused by a new session (LP-0MSB0RP7F000U0WJ)
                    try:
                        from proxy.session import _free_slot_assignment
                        _free_slot_assignment(
                            _dispatch_key_session_id(sid),
                            endpoint=_endpoint_from_record(record),
                        )
                    except Exception:
                        pass
                    try:
                        _p = getattr(srv, "local_prefill_in_flight", None)
                        if _p is not None and _dispatch_key_session_id(sid) in _p:
                            _p.pop(_dispatch_key_session_id(sid), None)

                    except Exception:
                        pass
                    # LP-0MVN4LCED00661MX AC1 (``idle_timeout``): classified by
                    # the record's ``first_byte_seen`` flag, NOT by this path —
                    # an inactive record that already served its response is
                    # normal churn and is suppressed by the emitter (AC3).
                    _emit_eviction_after_acceptance(
                        srv,
                        _dispatch_key_session_id(sid),
                        "idle_timeout",
                        idle_record,
                        endpoint=_endpoint_from_record(record),
                    )
                    try:
                        srv.logger.info(
                            "lease_released session=%s reason=idle_timeout endpoint=%s",
                            _dispatch_key_session_id(sid) if _dispatch_key_session_id(sid) else "unknown",

                            _endpoint_from_record(record) or "default",
                        )
                    except Exception:
                        pass
                else:
                    verify_candidates.append((sid, record))
    except Exception:
        pass

    # Phase 2 (outside the lock): verify which candidates still have a
    # processing slot on llama-server. A failed/unverifiable query means
    # the record is not in ``alive`` — Phase 3 applies fail-closed semantics
    # (LP-0MVIQ0FJC000YLX0): if the record has recent ``last_progress_ts``,
    # extend the lease rather than orphaning. The no-progress watchdog already
    # ran in phase 1, so records reaching here have shown recent progress.
    alive: set[str] = set()
    for sid, record in verify_candidates:
        try:
            if await _query_slot_processing(
                srv,
                _dispatch_key_session_id(sid),
                record.get("model_name"),
                endpoint=_endpoint_from_record(record),
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
                        # freeing (LP-0MSUO6XRP001MCB2). The watchdog already
                        # filtered out no-progress wedges above, so a slot
                        # verified as alive at this point is making real
                        # progress — extend on its behalf.
                        current["expires_at"] = (
                            time.monotonic() + _get_chunk_refresh_buffer_seconds(srv)
                        )
                        try:
                            srv.logger.info(
                                "lease_verified_active session=%s "
                                "reason=active_slot stream_abandoned=False",
                                _dispatch_key_session_id(sid) if _dispatch_key_session_id(sid) else "unknown",

                            )
                        except Exception:
                            pass
                        continue
                    # LP-0MVIQ0FJC000YLX0: fail-closed on unverifiable liveness.
                    # If the /slots probe failed or timed out (session not in
                    # ``alive``) but the record has recent progress, treat it
                    # as alive and extend the lease rather than orphaning. This
                    # prevents orphaning streams that are still generating but
                    # whose /slots query timed out under heavy prefill load.
                    no_progress_timeout = _get_no_progress_timeout_seconds(srv)
                    last_progress_ts = current.get("last_progress_ts")
                    if last_progress_ts is not None and no_progress_timeout > 0:
                        now = time.monotonic()
                        if now - last_progress_ts <= no_progress_timeout:
                            # Recent progress detected — extend the lease.
                            current["expires_at"] = (
                                time.monotonic()
                                + _get_chunk_refresh_buffer_seconds(srv)
                            )
                            try:
                                srv.logger.info(
                                    "lease_verified_active session=%s "
                                    "reason=recent_progress stream_abandoned=False",
                                    _dispatch_key_session_id(sid) if _dispatch_key_session_id(sid) else "unknown",
                                )
                            except Exception:
                                pass
                            continue
                    # Genuinely orphaned active record past its expires_at
                    # Capture stream_cm before deleting so we can cancel
                    # the upstream request and actually free the llama-server slot.
                    stream_cm = current.pop("stream_cm", None)
                    # AC1 (LP-0MVRMI5HK00558TB): resolve the slot identity while
                    # the assignment still exists, so the abandonment recorded
                    # after the release below is attributable to a slot (or to
                    # an endpoint-wide entry when it cannot be resolved,
                    # LP-0MVUVHAQP001IRKK).
                    orphan_identity = _resolve_abandoned_slot_identity(
                        _dispatch_key_session_id(sid),
                        endpoint=_endpoint_from_record(record),
                        record=record,
                    )
                    del srv.local_dispatch_records[sid]
                    try:
                        _p = getattr(srv, "local_prefill_in_flight", None)
                        if _p is not None and _dispatch_key_session_id(sid) in _p:
                            _p.pop(_dispatch_key_session_id(sid), None)
                    except Exception:
                        pass
                    removed += 1
                    # LP-0MVIQ0FJC000YLX0: cancel the upstream stream so the
                    # llama-server slot is actually freed. Without this the
                    # slot stays occupied on the backend while the proxy
                    # thinks it is free (LP-0MUGQHPJ50046LV8).
                    if stream_cm is not None:
                        try:
                            await stream_cm.__aexit__(None, None, None)
                        except Exception:
                            pass
                    # LP-0MVN1IQTF0037WTG AC6: emit the shared eviction event
                    # and plan the KV snapshot before the assignment is freed.
                    _record_local_eviction(
                        srv,
                        _dispatch_key_session_id(sid),
                        endpoint=_endpoint_from_record(record),
                        reason="lease_evicted",
                    )
                    # LP-0MVN4LCED00661MX AC1 (``orphan_cleanup``): the orphan
                    # sweep released an accepted dispatch with no live slot.
                    _emit_eviction_after_acceptance(
                        srv,
                        _dispatch_key_session_id(sid),
                        "orphan_cleanup",
                        record,
                        endpoint=_endpoint_from_record(record),
                    )
                    # Free the slot registry entry so the slot can be
                    # reused by a new session (LP-0MSB0RP7F000U0WJ)
                    try:
                        from proxy.session import _free_slot_assignment
                        _free_slot_assignment(
                            _dispatch_key_session_id(sid),
                            endpoint=_endpoint_from_record(record),
                        )
                    except Exception:
                        pass
                    # AC1 (LP-0MVRMI5HK00558TB): the sweep abandoned this
                    # dispatch and cancelled its upstream stream above; record
                    # the slot so a backend that has not released it yet is
                    # annotated "busy, not ours" — endpoint-wide when the
                    # identity was unresolvable (LP-0MVUVHAQP001IRKK).
                    _record_abandoned_identity(
                        srv,
                        orphan_identity,
                        session_id=_dispatch_key_session_id(sid),
                        endpoint=_endpoint_from_record(record),
                        reason="orphan_cleanup",
                    )
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
                            "stream_abandoned=True endpoint=%s",
                            _dispatch_key_session_id(sid) if _dispatch_key_session_id(sid) else "unknown",

                            _endpoint_from_record(record) or "default",
                        )
                        srv.logger.info(
                            "lease_released session=%s reason=orphan_cleanup",
                            _dispatch_key_session_id(sid) if _dispatch_key_session_id(sid) else "unknown",

                        )
                    except Exception:
                        pass
        except Exception:
            pass
    # LP-0MUGQHPJ50046LV8: close upstream streams for watchdog-released
    # records and free their slot assignments (deferred past the lock).
    if deferred_watchdog:
        for (
            sid,
            session_key,
            endpoint_from_record,
            stream_cm,
            released_record,
        ) in deferred_watchdog:
            try:
                # Close the upstream httpx stream to cancel the in-flight
                # request so the llama-server slot is actually freed.
                if stream_cm is not None:
                    try:
                        await stream_cm.__aexit__(None, None, None)
                    except Exception:
                        # Idempotent — harmless if the stream already closed.
                        pass
            except Exception:
                pass
            # Free the slot registry entry (LP-0MSB0RP7F000U0WJ).
            # LP-0MVN1IQTF0037WTG AC6: a watchdog-released lease is a
            # tenancy that expired without observed activity — record it and
            # plan the KV snapshot before the assignment is freed, so a client
            # that comes back restores warm instead of pre-filling cold.
            _record_local_eviction(
                srv,
                session_key,
                endpoint=endpoint_from_record,
                reason="tenancy_lease_expired",
            )
            # LP-0MVN4LCED00661MX AC1 (``no_progress_watchdog``): the watchdog
            # released an accepted dispatch that never produced a first byte.
            _emit_eviction_after_acceptance(
                srv,
                session_key,
                "no_progress_watchdog",
                released_record,
                endpoint=endpoint_from_record,
            )
            # AC1 (LP-0MVRMI5HK00558TB): the watchdog abandoned this dispatch.
            # Resolve the slot identity before the assignment is freed (the
            # record's pin first, then the assignment), then record it so a slot
            # the backend has not released yet is annotated "busy, not ours"
            # instead of reading as live work. An unresolvable identity records
            # the endpoint-wide entry (LP-0MVUVHAQP001IRKK).
            watchdog_identity = _resolve_abandoned_slot_identity(
                session_key,
                endpoint=endpoint_from_record,
                record=released_record,
            )
            try:
                from proxy.session import _free_slot_assignment
                _free_slot_assignment(
                    session_key,
                    endpoint=endpoint_from_record,
                )
            except Exception:
                pass
            _record_abandoned_identity(
                srv,
                watchdog_identity,
                session_id=session_key,
                endpoint=endpoint_from_record,
                reason="no_progress_watchdog",
            )
            # Pop from preflight tracking.
            try:
                _p = getattr(srv, "local_prefill_in_flight", None)
                if _p is not None and session_key in _p:
                    _p.pop(session_key, None)
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


async def _recover_stuck_generating_queries(srv) -> None:
    """Detect and reclaim stale ``local_generating_queries`` / ``local_generating_sessions`` entries.

    Session keys stuck in ``local_generating_sessions`` (from an aborted
    ``_decrement_generating_only_slot`` in the streaming generator's finally
    block, ``router.py:1700-1801``) wedge the local dispatch pool when
    ``local_generating_queries >= max_local``. Nothing in
    ``_dispatch_cleanup_loop`` previously reclaimed these entries, causing
    prolonged outages (e.g. 2h 11m, 2026-09-11; LP-0MTYAWDCQ006RGYU).

    This function reconciles ``local_generating_queries`` and
    ``local_generating_sessions`` against ``local_dispatch_records``:

    - For each session key in ``local_generating_sessions``, check whether
      an active dispatch record exists.
    - Keys with **no** active record are stale — removed from the set and
      ``local_generating_queries`` is reset to the count of remaining
      legitimate entries.
    - If **no** active dispatch records exist at all, all generating state
      is considered stale: the session set is emptied and the counter is
      reset to 0. This also reclaims a counter-only leak (positive counter
      with an empty session set), which can occur for anonymous sessions
      where increment/decrement touch the counter but not the set.
    - If the set is empty and the counter is 0 there is nothing to do.

    Designed to be called from ``_dispatch_cleanup_loop`` (server.py)
    as part of the periodic self-healing cycle, providing a bounded
    recovery mechanism that runs every 10 seconds. O(n) in the number
    of generating sessions; no llama-server HTTP calls.

    Lock ordering: acquires ``local_generating_queries_lock`` only. Does
    not acquire ``local_dispatch_records_lock`` — reads the records
    snapshot without holding the lock. This is acceptable because the
    check is self-correcting (runs every 10s) and a slightly stale
    snapshot cannot cause harm (false negatives simply defer recovery;
    false positives are prevented by the ``active`` flag check).

    On reclaim, emits a WARNING log and wakes contention-queue waiters
    so blocked dispatch retries can proceed.
    """
    try:
        generating_sessions: set | None = getattr(
            srv, "local_generating_sessions", None
        )
        if generating_sessions is None:
            # No generating-session tracking at all — nothing to reconcile.
            return

        prev_count = int(
            getattr(srv, "local_generating_queries", 0) or 0
        )
        if not generating_sessions and prev_count == 0:
            return  # Nothing to do

        generating_lock = getattr(
            srv, "local_generating_queries_lock", None
        )
        records = getattr(srv, "local_dispatch_records", None)

        if records is not None:
            has_active = any(
                r.get("active", False) for r in records.values()
            )
        else:
            has_active = False  # legacy mode: no dispatch records

        if not has_active:
            # No in-flight streams anywhere — all generating state is stale
            # (this includes a counter-only leak with an empty session set).
            stale_keys = set(generating_sessions)
            new_count = 0
        else:
            # Identify stale keys: those without an active dispatch record.
            stale_keys = set()
            for key in generating_sessions:
                # Support both string keys and tuple (endpoint, session)
                # keys.
                record = records.get(key)
                if record is None and not isinstance(key, tuple):
                    # Direct string-key lookup failed. Try the legacy
                    # ("local", session) tuple key.
                    record = records.get(("local", key))
                    # If that also fails, the dispatch record may be keyed
                    # by a full endpoint URL tuple (endpoint_url, session).
                    # Search through all tuple keys for a match.
                    if record is None:
                        for rk in records:
                            if isinstance(rk, tuple) and len(rk) == 2:
                                if rk[1] == key:
                                    record = records[rk]
                                    break
                if record is None or not record.get("active", False):
                    stale_keys.add(key)
            # The session set is authoritative for the legitimate count.
            new_count = len(generating_sessions) - len(stale_keys)

        if not stale_keys and prev_count == new_count:
            return  # Nothing to reclaim

        prev_sessions_count = len(generating_sessions)

        if generating_lock is not None:
            async with generating_lock:
                for key in stale_keys:
                    generating_sessions.discard(key)
                srv.local_generating_queries = max(0, new_count)
        else:
            for key in stale_keys:
                generating_sessions.discard(key)
            srv.local_generating_queries = max(0, new_count)

        new_count = int(getattr(srv, "local_generating_queries", 0) or 0)

        # Log the recovery.
        try:
            srv.logger.warning(
                "local_generating_queries counter recovered: "
                "reclaimed %d stale session(s) (reset from %d to %d, "
                "%d sessions removed from local_generating_sessions)",
                prev_count - new_count,
                prev_count,
                new_count,
                prev_sessions_count - len(generating_sessions),
            )
        except Exception:
            pass

        # Wake contention-queue waiters so blocked dispatches can retry.
        try:
            from proxy.contention_queue import wake_all

            await wake_all()
        except Exception:
            pass

    except Exception:
        pass


async def _recover_stuck_local_active_queries(srv) -> None:
    """Detect and repair a stuck ``local_active_queries`` counter.

    The counter can only be as large as the number of truly in-flight local
    requests (LP-0MVKSXR4P001V0CM AC2). A leak that outlives the request —
    e.g. a swallowed exception in ``_decrement_local_active_queries`` — is
    repaired by clamping the counter to the number of **active** dispatch
    records at every tick, rather than only resetting to 0 when *no* active
    records exist. Without the clamp, a leak that coexists with any in-flight
    request (``has_active`` true) survived indefinitely and made
    ``/llama/local/status`` report busy while the GPU was idle (2026-10-22
    night: 66 consecutive ``freeSlots: 0`` refusals).

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
                    active_count = _count_active_local_dispatch_records(srv)
                    active_count = active_count if active_count is not None else 0
                    if srv.local_active_queries > active_count:
                        prev = srv.local_active_queries
                        srv.local_active_queries = active_count
                        try:
                            srv.logger.warning(
                                "local_active_queries counter recovered: "
                                "reset from %d to %d (%d active dispatch "
                                "records)",
                                prev,
                                active_count,
                                active_count,
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


def _count_active_local_dispatch_records(srv) -> int | None:
    """Count dispatch records still marked ``active`` (in-flight local work).

    Returns ``None`` when the server has no dispatch-record registry (legacy
    mode) so callers can distinguish "no records system" from "zero
    in-flight". Best-effort snapshot without a lock, matching the periodic
    recovery loop's tolerance for a slightly stale view.
    """
    records = getattr(srv, "local_dispatch_records", None)
    if records is None:
        return None
    try:
        return sum(
            1
            for r in records.values()
            if isinstance(r, dict) and r.get("active")
        )
    except Exception:
        return None


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


def _evaluate_session_compaction(
    srv,
    session_id: str,
    session_messages: list,
    mode: str,
    summarizer=None,
    estimate_tokens=None,
    churn_collector=None,
    logger_obj=None,
) -> dict:
    """Prompt-assembly-time compaction decision for a session (integration).

    Parent LP-0MTCWE8NG003P0SD: proactive session compaction at
    prompt-assembly / session-persistence time. Wires the pure decision
    (``decide_session_compaction``) to the router's session machinery:

    - Warn-only dry-run (default, AC8 gate not passed): returns
      ``applied=False`` and the ORIGINAL list — the persistence layer
      stores exactly what it would have stored before. Advisory log +
      churn only.
    - Live (opt-in ``compaction_dry_run: false``): when the session
      exceeds the trigger and is compactable, returns ``applied=True``
      with the COMPACTED history for the caller to persist; the slot/KV
      is invalidated by the caller so the next request re-prefills
      cleanly from the compacted history.
    - ``remote_with_guidance``: returns ``applied=False`` and the reason;
      the dispatcher must route remote with guidance rather than dispatch
      local near-full-slot.

    Fail-open: any exception falls back to the untouched plan (original
    messages, ``applied=False``) and logs a warning — compaction must
    never break dispatch.

    Args:
        srv: Server object (config at ``srv.config``).
        session_id: The session being assembled.
        session_messages: The full session message list.
        mode: "fast" or "cheap" compaction-budget key (derived from the
            request's effective priority by the caller — ``low`` → "cheap",
            otherwise "fast"; LP-0MVDT94F0003VHUA AC6).
        summarizer: Production summarizer callable (see compaction module).
        estimate_tokens: Production routing estimator.

    Returns:
        The ``decide_session_compaction`` result (always includes
        ``messages``, ``applied``, ``dry_run``).
    """
    try:
        from proxy.compaction import decide_session_compaction

        server_config = getattr(srv, "config", {}).get("server", {})
        return decide_session_compaction(
            session_messages,
            server_config,
            mode,
            summarizer=summarizer,
            estimate_tokens=estimate_tokens,
            session_id=session_id or "",
            churn_collector=churn_collector,
            logger_obj=logger_obj,
        )
    except Exception:
        getattr(getattr(srv, "logger", None), "warning", lambda *a, **k: None)(
            "Session compaction evaluation failed; dispatching untouched "
            "(session=%s)",
            (session_id or "")[:8],
            exc_info=True,
        )
        return {
            "action": "noop",
            "applied": False,
            "dry_run": True,
            "messages": session_messages,
            "reason": "evaluation_failed",
        }


async def evaluate_and_apply_compaction(
    srv,
    result: dict,
    body_json: dict,
    server_config: dict,
    *,
    session=None,
    session_messages: list | None = None,
    summarizer=None,
    estimate_tokens=None,
    mode: str | None = None,
) -> dict:
    """Evaluate session compaction and apply the decision to ``result``.

    Shared by the local-dispatch path (``_handle_session``) and the router
    bypass path (``_proxy_with_fallback_cycle``) so an oversized session is
    always evaluated for compaction — including sessions the router skips
    for local dispatch on size grounds (LP-0MU5ARWSP001BYYB).

    On a live compact the helper mutates ``body_json`` in place to the
    compacted history and sets the same ``result`` keys the inline
    ``_handle_session`` block did: ``body_override``, ``compaction_applied``,
    ``compaction_estimated_before``, ``compaction_reason``,
    ``compaction_summary_text``, ``compaction_turns_summarized``,
    ``compaction_recent_turns_kept``. On live ``remote_with_guidance`` it
    sets ``result["compaction_remote_with_guidance"]``.

    Idempotent: ``result["compaction_evaluated"]`` gates a second call in the
    same request cycle so compaction never runs twice.

    Fail-open: an exception is logged and leaves dispatch unchanged, exactly
    like the inline block.

    Args:
        srv: Server object (config at ``srv.config``; session manager at
            ``srv.session_manager``; logger at ``srv.logger``).
        result: Per-request session result dict (mutated in place).
        body_json: Parsed request body (mutated in place on live compact).
        server_config: The ``server`` config mapping.
        session: Optional session object exposing ``.messages``.
        session_messages: Optional explicit stored history, used when
            ``session`` is not available.
        summarizer / estimate_tokens: Optional injectable production
            callables (tests); built from config when omitted.
        mode: Optional "fast"/"cheap" compaction-budget override; defaults
            to "fast" (the unified profile, LP-0MVDT94F0003VHUA AC6).

    Returns:
        Outcome dict: ``evaluated``, ``action``, ``applied``, ``dry_run``,
        ``estimated_before``, ``estimated_after``, ``reason``, and
        ``messages`` (compacted messages when applied, else ``None``).
    """
    outcome: dict[str, Any] = {
        "evaluated": False,
        "action": "noop",
        "applied": False,
        "dry_run": True,
        "estimated_before": 0,
        "estimated_after": 0,
        "reason": None,
        "messages": None,
    }

    # Idempotent gating — compaction already evaluated this cycle.
    if result.get("compaction_evaluated"):
        outcome["reason"] = "already_evaluated"
        return outcome

    try:
        from proxy.compaction_summarizer import build_compact_summarizer
        from proxy.provider import (
            _estimate_prompt_tokens_for_routing,
            _get_tokenizer_for_model,
        )

        # Build the production summarizer + token estimator once per request
        # so decide_session_compaction has real capabilities rather than the
        # always-None defaults that caused the compaction hang
        # (LP-0MTPK77WG009A4VH). build_compact_summarizer uses the
        # remote-only ``models.compact`` chain (Muse -> DeepSeek) so summaries
        # never contend with the local GPU slots; it falls back to the local
        # summarizer when ``models.compact`` is absent (LP-0MTT0O74N009E7N2).
        _llama_port = server_config.get("llama_server_port", 8080)
        _top_cfg = getattr(srv, "config", None)
        if not isinstance(_top_cfg, dict):
            _top_cfg = {"server": dict(server_config)}
        if summarizer is None:
            summarizer = build_compact_summarizer(
                _top_cfg,
                llama_port=_llama_port,
            )

        # Resolve the native tokenizer for the model so the compaction
        # trigger uses the same tokenizer + multiplier as routing
        # (LP-0MU5A84YU003YOTY).
        _model_name = body_json.get("model") if isinstance(body_json, dict) else None
        _model_config: dict = {}
        if _model_name:
            try:
                from proxy.lifecycle import get_model_config

                _model_config = get_model_config(_model_name) or {}
            except Exception:
                _model_config = {}
        _model_config = _model_config or {}
        if estimate_tokens is None:
            _tokenizer, _tok_multiplier = _get_tokenizer_for_model(
                _model_config, server_config
            )

            # The summarizer timeout resolves inside build_local_summarizer
            # from config (``compaction_summarizer_timeout``, default 600 s —
            # ``_DEFAULT_SUMMARIZER_TIMEOUT_SECONDS``): a 30 s timeout could
            # not wait for the single local slot to free up while a long
            # generating request held it, so every compaction failed
            # (summarizer_failed/timeout) and the session was routed remote
            # with guidance, stalling until the 900 s upstream timeout. 600 s
            # exceeds the observed worst-case stall (max
            # dispatch_first_byte_ms 713 s). See LP-0MU1RXEY10075TUU.
            def estimate_tokens(msgs):
                return _estimate_prompt_tokens_for_routing(
                    {"messages": msgs}, tokenizer=_tokenizer
                )

        # The full history this request produces: the persistent session
        # history PLUS this request's new turn(s). Compaction must operate on
        # this produced history — evaluating only the stored history would
        # drop the current turn from the compacted dispatch body
        # (LP-0MTVXP7DG00613ZB AC2).
        _stored_messages = getattr(session, "messages", None)
        if _stored_messages is None:
            _stored_messages = session_messages
        _delta_for_compaction = result.get("delta_messages")
        if result.get("is_delta_request") and _delta_for_compaction:
            _pre_compaction_messages = list(_stored_messages or []) + list(
                _delta_for_compaction
            )
        else:
            _pre_compaction_messages = list(
                (body_json.get("messages") if isinstance(body_json, dict) else None)
                or _stored_messages
                or []
            )

        _session_id = result.get("session_id") or ""
        # Compaction budget selection (LP-0MVDT94F0003VHUA AC6): the retired
        # mode axis is replaced by the request's effective priority — ``low``
        # uses the conservative (cheap) budget, everything else the fast
        # budget. Callers may still pass an explicit value; without a
        # request object the medium (default) priority applies → fast budget.
        if mode is None:
            _mode = "fast"
        else:
            _mode = mode

        # Mark evaluated BEFORE the await so a re-entrant call short-circuits.
        result["compaction_evaluated"] = True

        # The summarizer call blocks on the local llama-server slot (up to
        # ``compaction_summarizer_timeout``, default 600 s on a 1-slot backend
        # where a long generating request holds the slot). Run the whole
        # evaluation in a worker thread so the event loop keeps serving other
        # requests (notably the streaming request whose slot we are waiting
        # on) instead of freezing for the whole wait (LP-0MU1RXEY10075TUU).
        _compaction = await asyncio.to_thread(
            _evaluate_session_compaction,
            srv,
            _session_id,
            _pre_compaction_messages,
            _mode,
            summarizer=summarizer,
            estimate_tokens=estimate_tokens,
        )

        outcome["action"] = _compaction.get("action", "noop")
        outcome["applied"] = bool(_compaction.get("applied"))
        outcome["dry_run"] = bool(_compaction.get("dry_run", True))
        outcome["estimated_before"] = int(
            _compaction.get("estimated_before", 0) or 0
        )
        outcome["estimated_after"] = int(
            _compaction.get("estimated_after", 0) or 0
        )
        outcome["reason"] = _compaction.get("reason")
        outcome["evaluated"] = True

        if (
            _compaction.get("action") == "compact"
            and _compaction.get("applied")
            and not _compaction.get("dry_run")
        ):
            # Live: dispatch the compacted history as a full prompt (its
            # prefix no longer matches the client's history).
            body_json["messages"] = list(_compaction["messages"])
            body_json["cache_prompt"] = True
            body_json["session_id"] = _session_id
            result["body_override"] = json.dumps(body_json).encode("utf-8")
            result["is_delta_request"] = False
            result["delta_messages"] = None
            result["compaction_applied"] = True
            result["compaction_estimated_before"] = outcome["estimated_before"]
            result["compaction_reason"] = _compaction.get("reason")
            # LP-0MTYGZ1DI0004QP8: surface the compaction decision so
            # proxy_to_local can emit the X-Compaction-* bridge headers.
            # Set only on the live-applied path (AC6).
            result["compaction_summary_text"] = _compaction.get("summary_text")
            result["compaction_turns_summarized"] = int(
                _compaction.get("turns_summarized", 0) or 0
            )
            result["compaction_recent_turns_kept"] = int(
                _compaction.get("recent_turns_kept", 0) or 0
            )
            outcome["messages"] = list(_compaction["messages"])
            _logger = getattr(srv, "logger", None)
            if _logger is not None:
                from proxy.compaction import _budget_to_priority

                _logger.info(
                    "session_compaction applied session=%s priority=%s "
                    "est_before=%d est_after=%d",
                    str(_session_id)[:8],
                    _budget_to_priority(_compaction.get("mode")),
                    outcome["estimated_before"],
                    outcome["estimated_after"],
                )
            # After compaction, update the session's message history so
            # downstream token estimates (e.g. routing_estimate_session)
            # reflect the compacted count, not the pre-compaction value, and
            # record the client/base anchors the next request needs to heal
            # the sync break (AC2).
            try:
                if _session_id:
                    await srv.session_manager.update_messages(
                        _session_id,
                        list(_compaction["messages"]),
                    )
                    await srv.session_manager.mark_compacted(
                        _session_id,
                        len(_pre_compaction_messages),
                        len(_compaction["messages"]),
                    )
            except Exception:
                pass  # non-fatal: routing estimate still uses body messages
        elif (
            _compaction.get("action") == "remote_with_guidance"
            and not _compaction.get("dry_run")
        ):
            # Never dispatch local near-full-slot when the session cannot be
            # compacted; the dispatcher must escalate remote WITH guidance.
            result["compaction_remote_with_guidance"] = True
            result["compaction_estimated_before"] = outcome["estimated_before"]
            result["compaction_reason"] = _compaction.get("reason")
    except Exception:
        # Fail-open: record that evaluation was attempted (so it is not
        # retried this cycle) and leave dispatch unchanged.
        result["compaction_evaluated"] = True
        outcome["evaluated"] = True
        outcome["reason"] = "evaluation_failed"
        _logger = getattr(srv, "logger", None)
        if _logger is not None:
            _logger.warning(
                "Compaction evaluation failed; continuing unchanged "
                "(session=%s)",
                str(result.get("session_id") or "")[:8],
                exc_info=True,
            )

    return outcome


def _apply_post_compaction_heal(
    srv,
    session,
    body_json: dict,
) -> list | None:
    """Heal a post-compaction client/session sync break (AC2).

    Thin, fail-open wrapper around
    :func:`proxy.session_manager.compute_post_compaction_delta`: returns the
    healed delta when the incoming history still aligns with the stored
    compacted base, otherwise ``None`` so the caller keeps the existing
    ``history_mismatch`` invalidation behavior.
    """
    try:
        from proxy.session_manager import compute_post_compaction_delta

        incoming = body_json.get("messages", []) if isinstance(body_json, dict) else []
        return compute_post_compaction_delta(session, incoming)
    except Exception:
        srv.logger.debug("Post-compaction heal evaluation failed", exc_info=True)
        return None


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
                        # Post-compaction resync (LP-0MTVXP7DG00613ZB AC2):
                        # a live compaction leaves the client holding its
                        # pre-compaction history. When the stored compacted
                        # recent turns + appended messages still align with
                        # the incoming history, accept the new tail as a
                        # delta against the compacted base instead of
                        # invalidating the session (which would undo the
                        # compaction and drop the KV cache).
                        _healed = _apply_post_compaction_heal(
                            srv, session, body_json
                        )
                        if _healed is not None:
                            delta_messages = _healed
                            result["delta_messages"] = _healed
                            result["is_delta_request"] = True
                            result["session_fallback_reason"] = None
                            session_fallback_reason = None
                            body_json["messages"] = list(_healed)
                            try:
                                _record_delta_payload_bytes(
                                    len(
                                        json.dumps(
                                            _healed,
                                            separators=(",", ":"),
                                            ensure_ascii=False,
                                        ).encode("utf-8")
                                    )
                                )
                            except Exception:
                                pass
                            srv.logger.info(
                                "post_compaction_resync session=%s delta_messages=%d",
                                result["session_id"][:8],
                                len(_healed),
                            )
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

            # Proactive session compaction at prompt-assembly time
            # (LP-0MTCWE8NG003P0SD / LP-0MTGBQ01A000ZFT9 wiring). Evaluates
            # the persistent history this request produces. Warn-only
            # dry-run (default, AC8 gate pending) logs advisories only —
            # zero dispatch change. Opt-in live enforcement rewrites the
            # dispatch body to the compacted full history and marks the
            # request full-prompt so forward + persistence stay consistent.
            #
            # Extracted into the shared ``evaluate_and_apply_compaction``
            # helper (LP-0MU5IBXD4000POV0) so the router bypass path can run
            # the same evaluation for oversized sessions it would otherwise
            # skip entirely (LP-0MU5ARWSP001BYYB).
            await evaluate_and_apply_compaction(
                srv,
                result,
                body_json,
                server_config,
                session=session,
            )

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
# Local child-port discovery
# ===================================================================

_QWEN3_SPAWN_RE = re.compile(r"name=(\S+) on port (\d+)")
# Alias kept for backward-compat; new code uses _SPAWN_RE.
_SPAWN_RE = _QWEN3_SPAWN_RE

#: Live, rotation-proof child-port signal (LP-0MVP7525P008RXMF AC1). The router
#: writes one of these per proxied request, e.g. ``srv  proxy_reques: proxying
#: request to model Qwen3 on port 41395``. Unlike the spawn line it keeps
#: appearing for the whole life of the process, so it survives logrotate — the
#: 2026-10-26 window had 38,761 of these and zero spawn lines in the current
#: ``llama-server.log``, which is exactly why discovery returned ``None`` for the
#: lifetime of the pid and every ``/slots`` poll silently used the router.
_PROXY_REQUEST_RE = re.compile(r"proxying request to model (\S+) on port (\d+)")

_CHILD_PORT_HEAD_SCAN_BYTES = 256 * 1024
"""Head window scanned for startup spawn lines (unchanged behaviour)."""

_CHILD_PORT_TAIL_SCAN_BYTES = 1024 * 1024
"""Tail window scanned for the live ``proxying request ... on port`` signal.

Only read on a cache miss / re-verification (see the TTLs below), so the
per-request cost stays at zero once a port is cached.
"""

_child_port_cache: dict[Any, int] = {}
"""Cached **positive** child port per (llama-server pid, normalized model).

The router llama-server (port 8080) spawns child instances on dynamic ports
(``--port 0``). The child port is discoverable from the llama-server log:

1. ``spawning server instance with name=Qwen3 on port 58113`` — written once at
   startup and lost at the next logrotate.
2. ``proxying request to model Qwen3 on port 58113`` — written for every
   proxied request, so it is present in any non-trivial current log
   (LP-0MVP7525P008RXMF AC1).

The router serializes ``GET /slots?model=...`` behind the busy child's
generation loop (LP-0MTDGBRPU003Z7KU: measured 5-7s via the router vs 0.17s
direct), so availability/status/progress checks must target the child port.

LP-0MTP1FQXH004JYEF: the lookup filters by target model, because the *first*
spawn line is ``mxbai-embed`` (embeddings child); querying it reported
``3/3`` idle while the Qwen3 child was busy.

Cache policy (LP-0MVP7525P008RXMF): only positives are cached here (a negative
result is cached separately with a short TTL, so a signal that appears later is
picked up instead of being pinned for the life of the pid); a positive entry is
re-verified every :func:`_child_port_cache_ttl` seconds so a child restart with
the same parent pid cannot pin a stale port forever. A bare ``int`` pid key is
still honoured for backward compatibility.
"""

_child_port_cache_at: dict[Any, float] = {}
"""``time.monotonic()`` timestamp of each positive cache entry (re-verify TTL)."""

_child_port_miss_cache: dict[tuple[int, str, str], float] = {}
"""Negative cache: ``(pid, model, log path)`` -> monotonic time of the miss.

TTL-bounded by :func:`_child_port_miss_ttl`. Without this the helper would
re-read the log on every poll while no child port is discoverable; with a
permanent cache it would pin ``None`` for the whole life of the pid (the
2026-10-26 regression). The log path is part of the key so unit tests that
use a fresh ``tmp_path`` per case never collide.
"""


# -------------------------------------------------------------------
# Discovery tuning (env-overridable so an operator can shorten the
# re-verification window without a code change)
# -------------------------------------------------------------------

def _child_port_cache_ttl() -> float:
    """Seconds a positive child-port result is trusted before re-verification."""
    try:
        return max(0.0, float(os.environ.get("CHILD_PORT_CACHE_TTL", "300")))
    except (TypeError, ValueError):
        return 300.0


def _child_port_miss_ttl() -> float:
    """Seconds a negative child-port result is cached (then re-scanned)."""
    try:
        return max(0.0, float(os.environ.get("CHILD_PORT_MISS_TTL", "30")))
    except (TypeError, ValueError):
        return 30.0


_CHILD_PORT_FALLBACK_WARN_INTERVAL = 300.0
"""Seconds between repeated 'no child port / falling back to router' warnings."""

_last_child_port_fallback_warn_ts = 0.0
"""Monotonic timestamp of the last child-port fallback warning (throttle)."""


def _is_sparse_log(log_path: Path) -> bool:
    """Return True when *log_path* is sparse/NUL-padded/empty (logrotate).

    A non-sparse file with real content (even without a spawn line) returns
    False so :func:`_discover_local_child_port` does NOT fall back to global
    rotated logs — this preserves tmp_path isolation in tests
    (LP-0MTP1FQXH004JYEF).
    """
    try:
        if not log_path.exists():
            return False
        size = log_path.stat().st_size
        if size == 0:
            return True
        read_size = min(8192, size)
        with open(log_path, "rb") as f:
            chunk = f.read(read_size)
        if not chunk:
            return True
        # Sparse if >50% NULs or stripped text is empty/short whitespace.
        nul_ratio = chunk.count(b"\x00") / len(chunk)
        if nul_ratio > 0.5:
            return True
        text = chunk.decode("utf-8", errors="replace").replace("\x00", "").strip()
        return len(text) == 0
    except Exception:
        return False


def _llama_log_path(srv) -> Path:
    """Resolve the llama-server log path for a given server context."""
    log_dir = getattr(srv, "log_dir", None)
    if log_dir:
        return Path(log_dir) / "llama-server.log"
    return Path(__file__).parent / "logs" / "llama-server.log"


def _read_log_window(log_path: Path, *, tail: bool, window: int) -> str:
    """Read up to *window* bytes from the head or tail of *log_path*.

    Returns the decoded text with NUL padding (sparse logrotate files) stripped,
    or an empty string when the file is missing/unreadable/empty. A tail read
    can start mid-line; the partial first line simply does not match either
    port pattern, so the result is unaffected.
    """
    try:
        if not log_path.exists():
            return ""
        size = log_path.stat().st_size
        if size == 0 or window <= 0:
            return ""
        n = min(window, size)
        offset = size - n if tail else 0
        with open(log_path, "rb") as f:
            f.seek(offset)
            raw = f.read(n)
        return raw.decode("utf-8", errors="replace").replace("\x00", "")
    except Exception:
        return ""


def _last_matching_port(
    text: str, target_model: str | None, pattern: re.Pattern[str]
) -> int | None:
    """Return the port on the *last* line of *text* matching *pattern*.

    The last match wins so a restart / child re-spawn (which appends a newer
    line) is picked up. When *target_model* is given only an exact
    case-insensitive model match is accepted; otherwise a non-embed model is
    preferred over an embed one (the first spawn line is ``mxbai-embed`` —
    LP-0MTP1FQXH004JYEF).
    """
    if not text:
        return None
    want = (target_model or "").strip().lower() or None
    matches: list[tuple[str, int]] = []
    for line in text.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        try:
            name = str(m.group(1)).strip()
            port = int(m.group(2))
        except (TypeError, ValueError, IndexError):
            continue
        matches.append((name, port))
    if not matches:
        return None
    if want is not None:
        filtered = [p for n, p in matches if n.lower() == want]
        if filtered:
            return filtered[-1]
        return None
    non_embed = [(n, p) for n, p in matches if "embed" not in n.lower()]
    if non_embed:
        return non_embed[-1][1]
    return matches[-1][1]


def _scan_spawn_ports(log_path: Path, target_model: str | None) -> int | None:
    """Scan the head of one log file for startup spawn ports.

    Returns the *last* matching ``spawning server instance with name=<model>
    on port <port>`` line in the first
    :data:`_CHILD_PORT_HEAD_SCAN_BYTES` bytes, filtered by *target_model*
    (case-insensitive). ``None`` means "no matching spawn line in this
    window" — the caller distinguishes sparse vs. real content via
    :func:`_is_sparse_log` and may fall back to
    :func:`_scan_proxy_request_ports` (the live source) or a rotated log.
    """
    text = _read_log_window(
        log_path, tail=False, window=_CHILD_PORT_HEAD_SCAN_BYTES
    )
    return _last_matching_port(text, target_model, _SPAWN_RE)


def _scan_proxy_request_ports(
    log_path: Path, target_model: str | None
) -> int | None:
    """Scan the tail of one log file for the *live* child-port signal.

    Returns the port from the last ``proxying request to model <model> on port
    <port>`` line in the tail window — the rotation-proof source introduced by
    LP-0MVP7525P008RXMF AC1. Unlike the startup spawn line this keeps being
    written for the whole life of the process, so it survives logrotate.
    """
    text = _read_log_window(
        log_path, tail=True, window=_CHILD_PORT_TAIL_SCAN_BYTES
    )
    return _last_matching_port(text, target_model, _PROXY_REQUEST_RE)


def _scan_child_port_signals(
    log_path: Path, target_model: str | None
) -> int | None:
    """Return the newest child-port signal in a single log file.

    Tail first (live ``proxying request ... on port`` lines — newer than
    anything in the head window and present even when the startup spawn line
    has been rotated away), then the head window's spawn line (a fresh restart
    writes it before any request is proxied, so the very first polls after a
    restart still resolve — documented risk of the live source).
    """
    port = _scan_proxy_request_ports(log_path, target_model)
    if port is not None:
        return port
    return _scan_spawn_ports(log_path, target_model)


def _rotated_log_candidates(log_path: Path) -> list[Path]:
    """Return rotated llama-server logs, newest first, excluding *log_path*.

    Scans both the log's own directory and the canonical
    ``/var/log/llama-proxy`` location (so a worktree/test log dir can still
    find a real spawn line); compressed archives are skipped.
    """
    candidates: list[Path] = []
    try:
        parent = log_path.parent
        for base in {parent, Path("/var/log/llama-proxy")}:
            try:
                if not base.exists():
                    continue
                for p in base.glob("llama-server*.log*"):
                    if p.suffix == ".gz":
                        continue
                    try:
                        if p.resolve() == log_path.resolve():
                            continue
                    except Exception:
                        if p == log_path:
                            continue
                    candidates.append(p)
            except Exception:
                continue
        candidates.sort(
            key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True
        )
    except Exception:
        return []
    return candidates


def _remember_child_port(cache_key: Any, port: int, now: float) -> int:
    """Cache a positive discovery result and return it."""
    _child_port_cache[cache_key] = int(port)
    _child_port_cache_at[cache_key] = now
    return int(port)


def _discover_local_child_port(srv, model: str | None = None) -> int | None:
    """Discover the local model child port from the llama-server log.

    The router (``llama_server_port``) spawns model children on dynamic ports
    (``--port 0``); the child port is discoverable from the llama-server log:

    1. ``spawning server instance with name=<model> on port <port>`` — written
       near the top of the fresh log on startup, lost at the next logrotate.
    2. ``proxying request to model <model> on port <port>`` — written for every
       proxied request, so it is present in any live current log
       (LP-0MVP7525P008RXMF AC1).

    Args:
        srv: Server module / namespace with ``llama_process.pid`` and
            ``log_dir`` / ``current_model``.
        model: Optional target model name to filter by (e.g. ``"Qwen3"``).
            When ``None`` the function uses ``srv.current_model`` or
            ``"Qwen3"`` as the target. Matching is case-insensitive.
            Passing the wrong model previously returned the embed child port
            (first spawn line ``mxbai-embed``) — LP-0MTP1FQXH004JYEF.

    Returns the port for the target model, or ``None`` when the log is
    missing/unreadable or contains no matching child-port signal (caller
    should fall back to the router port). NUL padding from sparse logrotate
    files is stripped before matching.

    Cache policy (LP-0MVP7525P008RXMF AC1): positives are cached per
    ``(pid, normalized target model)`` and re-verified every
    :func:`_child_port_cache_ttl` seconds (a child restart with the same
    parent pid cannot pin a stale port); negatives are cached per
    ``(pid, model, log-path)`` with the short TTL from
    :func:`_child_port_miss_ttl`, so a signal that appears later (e.g. the
    first proxied request after a restart) is picked up instead of being
    pinned for the life of the pid. A new pid (restart) re-parses.
    """
    try:
        proc = getattr(srv, "llama_process", None)
        pid = getattr(proc, "pid", None) if proc is not None else None
        if pid is None:
            return None
        # Resolve the target model: explicit arg wins, then current_model.
        target = (model or getattr(srv, "current_model", None) or "Qwen3")
        target = str(target).strip() if target is not None else "Qwen3"
        if not target:
            target = "Qwen3"
        cache_key = (int(pid), target.lower())
        log_path = _llama_log_path(srv)
        now = time.monotonic()
        # --- Positive cache (re-verified every _child_port_cache_ttl()) ---
        cached = _child_port_cache.get(cache_key)
        if not isinstance(cached, int):
            # Compatibility: legacy cache used int pid -> port; honour it for
            # the default Qwen3 target so old tests that populate {pid: port}
            # still hit.
            legacy = _child_port_cache.get(int(pid))  # type: ignore[arg-type]
            if legacy is not None and target.lower() in ("qwen3",):
                cached = legacy
        if isinstance(cached, int):
            at = _child_port_cache_at.get(cache_key, 0.0)
            if now - at < _child_port_cache_ttl():
                return cached
            # TTL expired: re-verify. A model reload / child re-spawn with the
            # same parent pid can move the child port; the live signal at the
            # log tail reflects the newest mapping. If no newer signal exists
            # we keep serving the last known-good port instead of regressing
            # to the router (LP-0MVP7525P008RXMF AC1 — sticky positive).
            fresh = _scan_child_port_signals(log_path, target)
            if fresh is not None:
                return _remember_child_port(cache_key, fresh, now)
            _child_port_cache_at[cache_key] = now
            return cached
        # --- Negative cache (TTL-bounded so a late signal is picked up) ---
        miss_key = (int(pid), target.lower(), str(log_path))
        miss_at = _child_port_miss_cache.get(miss_key)
        if miss_at is not None and now - miss_at < _child_port_miss_ttl():
            return None
        # Missing file -> no fallback scan (test isolation: tmp_path that
        # does not contain the log should stay None, not find the global
        # /var/log/llama-proxy file — LP-0MTP1FQXH004JYEF).
        if not log_path.exists():
            _child_port_miss_cache[miss_key] = now
            return None
        # 1. Live signal first: the proxying-request line survives logrotate
        #    (the 2026-10-26 production log had 38,761 of them and zero spawn
        #    lines — the exact shape that previously pinned None for the whole
        #    pid lifetime). LP-0MVP7525P008RXMF AC1.
        port = _scan_child_port_signals(log_path, target)
        if port is not None:
            return _remember_child_port(cache_key, port, now)
        # Primary had real content but no matching child-port signal for this
        # target (e.g. test file "[58113] main: model loaded" or a
        # non-Qwen3-only log) — do NOT fall back to global
        # /var/log/llama-proxy rotated logs, otherwise tmp_path isolation
        # leaks into tests. Only fall back when the primary is
        # sparse/NUL-padded/empty (logrotate case) — LP-0MTP1FQXH004JYEF.
        if not _is_sparse_log(log_path):
            _child_port_miss_cache[miss_key] = now
            return None
        # Sparse current log (e.g. NUL-padded after logrotate) — scan
        # rotated logs (llama-server*.log) by mtime descending for the most
        # recent matching signal. Only when the primary file exists but had
        # no matching signal; a missing file already returned above.
        try:
            for cand in _rotated_log_candidates(log_path)[:8]:
                cand_port = _scan_child_port_signals(cand, target)
                if cand_port is not None:
                    return _remember_child_port(cache_key, cand_port, now)
        except Exception:
            pass
        _child_port_miss_cache[miss_key] = now
        return None
    except Exception:
        return None


_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"}
"""Hosts that resolve to the local router instance (never rewritten)."""


def _warn_child_port_fallback(srv, model: str | None, router_port: int) -> None:
    """Throttled warning + counter when a proxy ``/slots`` query falls back
    to the router port because no child port could be discovered
    (LP-0MVP7525P008RXMF AC3).

    The old implementation swallowed the discovery failure
    (``except Exception: pass``) and silently used the router for the whole
    life of the pid — the 2026-10-26 log had 1,754 router 500s and zero child
    500s because of it. Now the fallback is observable in ``proxy.log``
    (throttled to one line per :data:`_CHILD_PORT_FALLBACK_WARN_INTERVAL`
    seconds) and in the ``llama_slots_child_port_fallbacks_total`` counter.

    In single-server mode (``llama_router_mode`` off) the router port *is* the
    model instance, so the fallback is expected behaviour: we log at debug
    level there and still count it under ``no_child_port_single_server``.
    """
    global _last_child_port_fallback_warn_ts
    cfg = getattr(srv, "config", None)
    server_cfg = cfg.get("server", {}) if isinstance(cfg, dict) else {}
    router_mode = bool(server_cfg.get("llama_router_mode", False))
    try:
        from proxy.metrics import record_slots_child_port_fallback

        record_slots_child_port_fallback(
            "no_child_port_router_mode"
            if router_mode
            else "no_child_port_single_server"
        )
    except Exception:
        pass
    now = time.monotonic()
    if now - _last_child_port_fallback_warn_ts < _CHILD_PORT_FALLBACK_WARN_INTERVAL:
        return
    _last_child_port_fallback_warn_ts = now
    message = (
        "slots_child_port_unavailable: no model child port discovered for "
        f"model={model or getattr(srv, 'current_model', None) or 'unknown'}; "
        f"falling back to router port {router_port} — every proxy-initiated "
        "/slots poll will traverse the router and can 500 during giant "
        "prefills (LP-0MVP7525P008RXMF AC3)"
    )
    try:
        if router_mode:
            srv.logger.warning(message)
        else:
            srv.logger.debug(message)
    except Exception:
        pass


def _resolve_local_slots_target(
    srv,
    model: str | None = None,
    endpoint: str | None = None,
    router_port: int | None = None,
) -> tuple[str | None, int]:
    """Resolve ``(endpoint, port)`` for a proxy-initiated ``/slots`` query.

    Single shared path for every proxy ``/slots`` call site (status handler,
    SSE payload, slot-availability check, progress/liveness probes) so no
    query silently traverses the router (LP-0MVP7525P008RXMF AC2).

    Rules:

    - When *endpoint* points at a genuinely different llama-server instance
      (non-loopback host, or a loopback port that is not the configured router
      port — e.g. it is already a child port), the endpoint is returned
      untouched.
    - When *endpoint* is the local router (loopback host on
      ``server.llama_server_port``) or omitted, the discovered child port is
      returned as ``(None, child_port)``; when discovery yields nothing, the
      router port is returned after a throttled warning + counter
      (``_warn_child_port_fallback``, AC3).

    Returns ``(endpoint_or_None, port)``: pass the tuple to
    ``_query_slots*`` so it builds the URL from the resolved port instead of
    the router.
    """
    cfg = getattr(srv, "config", None)
    server_cfg = cfg.get("server", {}) if isinstance(cfg, dict) else {}
    if router_port is None:
        try:
            router_port = int(server_cfg.get("llama_server_port", 8080) or 8080)
        except (TypeError, ValueError):
            router_port = 8080
    else:
        router_port = int(router_port)
    if endpoint:
        host, port = _parse_endpoint_url(endpoint)
        if host not in _LOOPBACK_HOSTS or port != router_port:
            # A genuinely different instance — the router-proxied-500 problem
            # does not apply, never rewrite it.
            return endpoint, port
    child = _discover_local_child_port(srv, model=model)
    if child is not None:
        return None, child
    _warn_child_port_fallback(srv, model, router_port)
    return None, router_port


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
    lease_held: bool = False,

    endpoint: str | None = None,
) -> JSONResponse | None:
    """Check llama-server slot availability.

    Returns a 503 JSONResponse if no slots are available, None otherwise.

    When *lease_held* is True the check is skipped entirely: the caller has
    already acquired a dispatch lease (``_try_acquire_local_dispatch``),
    which gates concurrency to ``session_slot_pool_size`` — the /slots query
    would be redundant and, via the router, can take 5-7s under load
    (LP-0MTDGBRPU003Z7KU).

    The query targets the discovered local child port when one is known —
    including when *endpoint* is the local router URL (the local provider
    endpoint in router mode) — because the router serializes
    ``GET /slots?model=...`` behind the busy child's generation loop and turns a
    cancelled proxied poll into an HTTP 500. A genuinely different
    llama-server instance passed as *endpoint* is used unchanged. Resolution is
    centralised in :func:`_resolve_local_slots_target`
    (LP-0MVP7525P008RXMF AC2).

    The query uses a dedicated per-call ``httpx.AsyncClient`` with a short
    timeout (``session_slot_availability_timeout_seconds``, default 2.0) so a
    slow router/child response fails fast and can never exhaust the shared
    ``_http_client`` pool (LP-0MTDH2U6V0062TUF).

    """
    if lease_held:
        return None
    if not (path == "v1/chat/completions" or path.endswith("chat/completions")):
        return None

    try:
        slot_model = (
            slot_model_name or model_name or srv.current_model or "Qwen3"
        )
        # LP-0MVP7525P008RXMF AC2: single shared resolution path — a
        # *local* endpoint (the router) is rewritten to the discovered child
        # port so this availability check never serializes behind the busy
        # child via the router (LP-0MTDGBRPU003Z7KU measured 5-7s vs 0.17s).
        slots_endpoint, slots_port = _resolve_local_slots_target(
            srv, model=slot_model, endpoint=endpoint, router_port=llama_port
        )
        if slots_endpoint:
            slots_url = f"{slots_endpoint.rstrip('/')}/slots?model={slot_model}"
        else:
            slots_url = f"http://localhost:{slots_port}/slots?model={slot_model}"
        availability_timeout = float(
            server_config.get(
                "session_slot_availability_timeout_seconds", 2.0
            )
            or 2.0

        )
        # Dedicated per-call client with a short timeout — never borrow the
        # shared _http_client, so a slow /slots response cannot hold shared
        # pool connections and starve slot_save/status under multi-session
        # load (LP-0MTDH2U6V0062TUF).
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(availability_timeout)
        ) as client:
            slots_resp = await client.get(slots_url, timeout=availability_timeout)
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


def _other_session_dispatch_active(
    srv,
    session_id: str | None,
    endpoint: str | None = None,
    now: float | None = None,
) -> bool:
    """True when *another* session holds a live dispatch in this endpoint.

    LP-0MVS9Y76R004B5M8: co-residency evidence at first-byte time. Prefill is
    serialised but prefill+decode batching is not, so an occupant that is
    *generating* halves a newcomer's prefill throughput. This helper answers
    "was another session's dispatch live while this prefill ran?" from the live
    dispatch records, complementing the admission-time ``busy_slots`` snapshot
    stamped on the request (the two are ORed by the sample recorder).

    Scope matches the admission gate: only records in the same *endpoint*
    scope count (endpoint-aware pools), and the requesting session's own
    record never makes its own dispatch contended. A record is live when it is
    ``active`` or holds an unexpired no-preemption lease — a released record is
    not consuming the gate.

    Fail-open: any error/missing state returns False (never blocks a stream
    and never fabricates a contended sample).
    """
    try:
        records = getattr(srv, "local_dispatch_records", None)
        if not isinstance(records, dict) or not records:
            return False
        _now = time.monotonic() if now is None else float(now)
        ep_scope = endpoint or ""
        for key, record in list(records.items()):
            if not isinstance(record, dict):
                continue
            key_endpoint = key[0] if isinstance(key, tuple) and key else ""
            if key_endpoint != ep_scope:
                continue
            owner = _dispatch_key_session_id(key)
            if not owner or owner == session_id:
                continue
            live = bool(record.get("active")) or float(
                record.get("expires_at") or 0.0
            ) > _now
            if live:
                return True
        return False
    except Exception:
        return False


def dispatch_sample_contended(
    srv,
    session_id: str | None,
    *,
    endpoint: str | None = None,
    admission_busy_slots: int | None = None,
    now: float | None = None,
) -> bool:
    """Co-residency flag for a first-byte calibration sample.

    LP-0MVS9Y76R004B5M8: a prefill observed while the gate was contended
    belongs in the contended calibration window. Two independent signals are
    ORed (either marks the sample contended):

    - the **admission-time** ``busy_slots`` snapshot stamped on the request
      (``>= 1``), and
    - a **first-byte-time live check** that *another* session's dispatch
      record is still active for the same endpoint
      (:func:`_other_session_dispatch_active`).

    Fail-open: an unparseable/absent snapshot contributes 0 and a missing
    record set contributes False, so a stream is never mis-tagged because
    observability state is missing.
    """
    try:
        busy = max(0, int(admission_busy_slots or 0))
    except (TypeError, ValueError):
        busy = 0
    if busy >= 1:
        return True
    return _other_session_dispatch_active(
        srv, session_id, endpoint=endpoint, now=now
    )
