"""
Provider Module

Provider resolution and fallback logic for model requests.

Provides:
- `resolve_provider()`: Select the next available provider for a model config
- `proxy_with_remote_fallback()`: Remote provider fallback loop
- Cooldown tracking: Mark providers as temporarily unavailable after failures
"""

import asyncio
import json
import logging
import time
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException, Response
from fastapi.responses import StreamingResponse

from proxy.utils import _is_empty_response

logger = logging.getLogger("llama-proxy.provider")

# ---------------------------------------------------------------------------
# Cooldown / circuit-breaker state
# ---------------------------------------------------------------------------

# In-memory cooldown tracking: provider_name -> expiry_timestamp (seconds since epoch)
_provider_unavailable_until: Dict[str, float] = {}

# Consecutive failure count for exponential backoff: provider_name -> count
# Incremented on each failure, reset to 0 on success.
_provider_failure_count: Dict[str, int] = {}

# Exponential backoff constants (remote providers only)
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 45.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_provider(
    model_config: dict,
    failed_provider: Optional[str] = None,
) -> Optional[dict]:
    """Get the next available provider for a model config.

    Iterates through the model's ordered `providers` list and returns the
    first provider that:
    - Is not the `failed_provider` (if specified)
    - Is not in cooldown (if marked unavailable)

    Args:
        model_config: Model configuration dict. Must contain a ``providers``
                      key whose value is an ordered list of provider configs.
        failed_provider: Optional name of a provider that just failed and
                         should be skipped.

    Returns:
        A provider config dict (with keys ``name``, ``type``, etc.), or
        ``None`` if no provider is available.
    """
    providers: Optional[List[Dict[str, Any]]] = model_config.get("providers")
    if not providers:
        return None

    for provider_cfg in providers:
        name = provider_cfg.get("name", "")
        if failed_provider and name == failed_provider:
            continue
        if _is_provider_unavailable(name):
            continue
        return provider_cfg

    return None


def get_model_type(model_config: dict) -> Optional[str]:
    """Determine the model type from the providers list.

    Returns ``"local"`` if the first provider is a local provider,
    ``"remote"`` if it is remote, or ``None`` if no providers are defined.

    This replaces the legacy ``model_config["type"]`` field.
    """
    providers: Optional[List[Dict[str, Any]]] = model_config.get("providers")
    if not providers:
        return None
    first = providers[0]
    ptype = first.get("type")
    if ptype in ("local", "remote"):
        return ptype
    return None


def get_local_model_name_from_providers(model_config: dict) -> Optional[str]:
    """Extract the llama_model name from the providers list.

    Searches the ordered ``providers`` list for a local provider and
    returns its ``llama_model`` value.

    This replaces the legacy ``model_config["llama_model"]`` field.

    Returns ``None`` if no local provider is found.
    """
    providers: Optional[List[Dict[str, Any]]] = model_config.get("providers")
    if not providers:
        return None
    for p in providers:
        if isinstance(p, dict) and p.get("type") == "local":
            return p.get("llama_model")
    return None


def get_remote_endpoint(model_config: dict) -> Optional[str]:
    """Extract the endpoint URL from the providers list.

    Searches the ordered ``providers`` list for a remote provider and
    returns its ``endpoint`` value.

    This replaces the legacy ``model_config["endpoint"]`` field.

    Returns ``None`` if no remote provider is found.
    """
    providers: Optional[List[Dict[str, Any]]] = model_config.get("providers")
    if not providers:
        return None
    for p in providers:
        if isinstance(p, dict) and p.get("type") == "remote":
            return p.get("endpoint")
    return None


def mark_provider_unavailable(
    provider_name: str,
    cooldown_seconds: float,
    use_exponential_backoff: bool = False,
) -> None:
    """Mark a provider as unavailable for the given cooldown duration.

    When *use_exponential_backoff* is ``True``, the actual cooldown is
    computed via exponential backoff based on consecutive failure count
    instead of using *cooldown_seconds* directly:

        cooldown = min(BACKOFF_BASE * 2^failure_count, BACKOFF_MAX)

    The failure count is incremented on each call and reset to 0 on
    successful provider response via ``_reset_provider_failure_count()``.

    Args:
        provider_name: Name of the provider to mark.
        cooldown_seconds: Number of seconds the provider should be
                          considered unavailable (used as max when
                          *use_exponential_backoff* is ``True``).
        use_exponential_backoff: If ``True``, apply exponential backoff.
                                 Default is ``False``.
    """
    if use_exponential_backoff:
        count = _provider_failure_count.get(provider_name, 0)
        backoff = min(
            _BACKOFF_BASE_SECONDS * (2 ** count),
            _BACKOFF_MAX_SECONDS,
        )
        cooldown_seconds = min(backoff, cooldown_seconds)
        _provider_failure_count[provider_name] = count + 1

    _provider_unavailable_until[provider_name] = time.time() + cooldown_seconds


def _reset_provider_failure_count(provider_name: str) -> None:
    """Reset the consecutive failure count for a provider on success.

    Removes the provider from the failure count dict so that the next
    failure starts with the base backoff interval.
    """
    _provider_failure_count.pop(provider_name, None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_provider_unavailable(provider_name: str) -> bool:
    """Check if a provider is currently in cooldown.

    Returns ``True`` if the provider is marked unavailable and its cooldown
    has not yet expired.  Expired entries are cleaned up lazily.

    This check is global — it reads from the shared module-level dict, so
    any session calling this function sees the same cooldown state. There is
    no per-session isolation of cooldown state.
    """
    expiry = _provider_unavailable_until.get(provider_name)
    if expiry is None:
        return False
    if time.time() >= expiry:
        # Cooldown expired — clean up entry
        del _provider_unavailable_until[provider_name]
        return False
    return True


def _parse_retry_after(response: Response) -> Optional[float]:
    """Parse Retry-After header from a response.

    Supports both integer seconds and HTTP-date formats.

    Returns:
        Number of seconds to wait, or ``None`` if no Retry-After header.
    """
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return None
    # Try integer seconds first
    try:
        return float(retry_after)
    except ValueError:
        pass
    # Try HTTP-date format
    try:
        dt = parsedate_to_datetime(retry_after)
        now = time.time()
        dt_ts = dt.timestamp()
        if dt_ts > now:
            return dt_ts - now
        return 0.0
    except Exception:
        return None


def _compute_cooldown(
    cooldown_seconds: float,
    response: Optional[Response] = None,
) -> float:
    """Compute the effective cooldown duration.

    Uses the larger of the configured cooldown and any Retry-After header
    value present in the response.
    """
    if response is None:
        return cooldown_seconds
    retry_after = _parse_retry_after(response)
    if retry_after is not None:
        return max(cooldown_seconds, retry_after)
    return cooldown_seconds


def _get_cooldown_seconds(config: dict) -> float:
    """Read ``provider_cooldown_seconds`` from config, supporting both flat and nested formats.

    Checks ``config["provider_cooldown_seconds"]`` (flat) first for backward
    compatibility with unit tests, then falls back to
    ``config["server"]["provider_cooldown_seconds"]`` (nested) for production
    configs loaded from ``config.yaml``.  Defaults to 60.
    """
    val = config.get("provider_cooldown_seconds")
    if val is None:
        val = config.get("server", {}).get("provider_cooldown_seconds", 60)
    return float(val)


def _get_local_slot_retry_attempts(config: dict) -> int:
    """Read local slot-exhaustion retry attempts from config.

    Supports flat key (tests) and nested server key (runtime).
    """
    val = config.get("local_slot_exhaustion_retry_attempts")
    if val is None:
        val = config.get("server", {}).get("local_slot_exhaustion_retry_attempts", 0)
    try:
        return max(0, int(val or 0))
    except Exception:
        return 0


def _get_local_slot_retry_delay_seconds(config: dict) -> float:
    """Read local slot-exhaustion retry delay (seconds) from config."""
    val = config.get("local_slot_exhaustion_retry_delay_seconds")
    if val is None:
        val = config.get("server", {}).get("local_slot_exhaustion_retry_delay_seconds", 0.2)
    try:
        return max(0.0, float(val or 0.0))
    except Exception:
        return 0.2


def _get_slot_unavailable_retry_after(config: dict) -> float:
    """Read the short cooldown for slot-exhaustion (slot busy), default 5s.

    Distinct from provider_cooldown_seconds: a busy slot frees quickly (when
    the in-flight request finishes), so we use a short cooldown so the next
    request can retry local soon instead of waiting the full provider cooldown.
    """
    val = config.get("slot_unavailable_retry_after")
    if val is None:
        val = config.get("server", {}).get("slot_unavailable_retry_after", 5)
    try:
        return max(1.0, float(val or 5))
    except Exception:
        return 5.0


def _is_streaming_response(response: Response) -> bool:
    """Return True when response is a StreamingResponse (body is a generator).

    Such responses cannot be inspected for emptiness and should be treated as
    success when their status is 2xx.
    """
    return isinstance(response, StreamingResponse)


def _response_body_text(response: Response) -> str:
    """Best-effort extraction of text body for diagnostics/classification."""
    try:
        if hasattr(response, 'content'):
            b = response.content
        elif hasattr(response, 'body'):
            b = response.body
        else:
            b = None
        if b:
            return b.decode('utf-8', errors='replace') if isinstance(b, (bytes, bytearray)) else str(b)
    except Exception:
        return ""
    return ""


def _add_provider_header(response: Response, provider_name: str) -> Response:
    """Add X-Provider header to a response."""
    response.headers.append("X-Provider", provider_name)
    return response


def _build_resolved_model_value(provider_cfg: dict) -> Optional[str]:
    """Build the X-Resolved-Model header value from a provider config.

    Returns ``<provider-name>/<model-id>`` or ``None`` if the config
    doesn't have the required fields.

    For local providers, uses ``llama_model`` as the model ID.
    For remote providers, uses ``model`` (upstream model ID).

    The provider name is taken from the ``provider`` field first (actual
    provider brand name), falling back to ``name`` (provider entry name)
    for backward compatibility.  A warning is logged when a remote provider
    entry lacks the ``provider`` field.
    """
    provider_name = provider_cfg.get("provider") or provider_cfg.get("name")
    if not provider_name:
        return None
    model_id = provider_cfg.get("llama_model") or provider_cfg.get("model")
    if not model_id:
        return None
    # Warn when a remote provider entry is missing the ``provider`` field
    if not provider_cfg.get("provider") and provider_cfg.get("type") == "remote":
        logger.warning(
            "Remote provider entry %r is missing the 'provider' field; "
            "X-Resolved-Model header will use 'name' (%r) instead of the "
            "actual provider brand name. Add 'provider: <brand>' to the "
            "provider config to fix this.",
            provider_cfg.get("name"),
            provider_name,
        )
    return f"{provider_name}/{model_id}"


def _add_resolved_model_header(response: Response, provider_cfg: dict) -> Response:
    """Add X-Resolved-Model header to a response based on provider config.

    Sets the header using ``_build_resolved_model_value()``. Overwrites
    any existing value so the fallback's resolved provider takes priority.
    """
    value = _build_resolved_model_value(provider_cfg)
    if value:
        response.headers["X-Resolved-Model"] = value
    return response


def _build_exhausted_response(all_local_slot_exhaustion: bool = False, total_slots: int = 0, unavailable_providers: Optional[dict] = None, diagnostics: Optional[List[Dict[str, Any]]] = None) -> Response:
    """Build the response when all providers are exhausted.

    Args:
        all_local_slot_exhaustion: If ``True``, all providers exhausted due to
                                   slot exhaustion (returns HTTP 429).
                                   Otherwise, returns HTTP 503 with JSON body.
        total_slots: Total number of slots across local providers (used only
                     for the slot-exhaustion 429 text body).
        unavailable_providers: Optional mapping of provider -> remaining cooldown seconds
                               to include in the 503 JSON payload for diagnostics.
        diagnostics: Optional list of per-provider attempt diagnostics (order-preserving)
    """
    if all_local_slot_exhaustion:
        # total_slots may be 0 if unknown; still format per acceptance criteria
        return Response(
            content=(f"Model server busy: 0/{int(total_slots)} slots available. Retry later.").encode("utf-8"),
            status_code=429,
            media_type="text/plain",
        )

    payload = {"error": "All providers exhausted", "retry_after": 60}
    if unavailable_providers:
        # Attach diagnostic info about which providers are in cooldown
        try:
            payload["unavailable_providers"] = unavailable_providers
        except Exception:
            pass

    if diagnostics:
        try:
            # Include a sanitized diagnostics list to aid troubleshooting
            payload["diagnostics"] = diagnostics
        except Exception:
            pass

    return Response(
        content=json.dumps(payload).encode("utf-8"),
        status_code=503,
        media_type="application/json",
    )


def _is_connection_error(exc: Exception) -> bool:
    """Check if an exception is a connection-related error."""
    return isinstance(exc, (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.TimeoutException,
        httpx.RemoteProtocolError,
        httpx.NetworkError,
    ))


def _is_http_error_status(status_code: int) -> bool:
    """Check if an HTTP status code indicates a provider failure."""
    return status_code >= 400


# Lazy import to avoid circular dependency
# NOTE: We do NOT cache the result so that tests can patch
# proxy.proxy_remote.proxy_to_remote between calls.


def _get_proxy_to_remote():
    """Lazily import proxy_to_remote.

    Uses ``proxy.server`` as the source so that any monkeypatches
    applied to the server module (e.g. in tests) are picked up.
    """
    from proxy.server import proxy_to_remote
    return proxy_to_remote


def _get_proxy_to_local():
    """Lazily import proxy_to_local.

    Select the best available implementation:
    - If `proxy.server.proxy_to_local` has been monkeypatched (differs from
      the router implementation), prefer that so server-level patches are used.
    - Otherwise, prefer `proxy.router.proxy_to_local` when available (so tests
      that patch the router are respected).
    - Fallback to whichever one is importable.
    """
    try:
        import proxy.router as _router
    except Exception:
        _router = None

    try:
        import proxy.server as _server
    except Exception:
        _server = None

    # If both router and server provide proxy_to_local, try to detect which
    # one has been monkeypatched by tests.  Prefer the implementation that
    # appears to come from outside the package (i.e., its __module__ is not
    # the original module name), otherwise prefer router by default.
    router_fn = getattr(_router, 'proxy_to_local', None) if _router is not None else None
    server_fn = getattr(_server, 'proxy_to_local', None) if _server is not None else None

    if router_fn is not None and server_fn is None:
        return router_fn
    if server_fn is not None and router_fn is None:
        return server_fn

    if router_fn is not None and server_fn is not None:
        router_modname = getattr(router_fn, '__module__', '')
        server_modname = getattr(server_fn, '__module__', '')
        # If router function appears to be patched (not from proxy.router), prefer it
        if not router_modname.startswith('proxy.router') and server_modname.startswith('proxy.server'):
            return router_fn
        # If server function appears to be patched, prefer it
        if not server_modname.startswith('proxy.server') and router_modname.startswith('proxy.router'):
            return server_fn
        # Fallback: prefer router implementation
        return router_fn

    raise ImportError('No proxy_to_local implementation available')


def _get_scheduler_has_idle_slot():
    """Lazily import and check whether the JobScheduler has an idle slot.

    Returns True if there is at least one idle slot, False if all slots
    are busy, or True if no scheduler is active (no slot management).
    """
    try:
        from proxy.router import _scheduler_has_idle_slot as _check
        return _check()
    except Exception:
        return True


def _get_local_concurrency_info(config: dict) -> tuple:
    """Lazily import and return (current_local_active, max_local) from config.

    Returns the current local active query count and the configured
    local_max_concurrent_queries limit.  Defaults to (0, 1) on error.
    """
    cur_active = 0
    max_local = 1
    try:
        import proxy.server as _srv
        cur_active = max(0, int(getattr(_srv, 'local_active_queries', 0) or 0))
    except Exception:
        pass
    try:
        server_cfg = config.get("server", config)
        max_local = max(1, int(server_cfg.get("local_max_concurrent_queries", 1) or 1))
    except (ValueError, TypeError):
        pass
    return (cur_active, max_local)


def _parse_slot_exhaustion(response):
    """Parse a slot-exhaustion response and return slot info.

    Returns a dict with keys:
      - total_slots
      - available_slots
      - reason (optional)
      - local_owner_session_id (optional)

    when the response indicates slot exhaustion, otherwise returns None.

    Handles two response formats:

    1. Proxy-generated (``_build_slot_exhaustion_response``):

           {"error": {"code": "no_slots_available", ...}, "total_slots": 1}

    2. Llama-server native (flat):

           {"type": "server_busy", "code": "no_slots_available", ...}
    """
    try:
        if response.status_code != 503:
            return None
        import json
        body = json.loads(response.body)

        # Format 1: nested error.code
        error = body.get("error", {})
        if isinstance(error, dict) and error.get("code") == "no_slots_available":
            total = int(body.get("total_slots", 0) or 0)
            avail = int(body.get("available_slots", 0) or 0)
            reason = body.get("reason") or error.get("reason")
            owner = body.get("local_owner_session_id")
            return {
                "total_slots": total,
                "available_slots": avail,
                "reason": reason,
                "local_owner_session_id": owner,
            }

        # Format 2: flat top-level code (llama-server native)
        if body.get("code") == "no_slots_available":
            total = int(body.get("total_slots", 0) or 0)
            avail = int(body.get("available_slots", 0) or 0)
            reason = body.get("reason")
            owner = body.get("local_owner_session_id")
            return {
                "total_slots": total,
                "available_slots": avail,
                "reason": reason,
                "local_owner_session_id": owner,
            }
    except Exception:
        pass
    return None


def _is_slot_exhaustion_response(response) -> bool:
    """Backward-compatible boolean check for slot exhaustion."""
    return _parse_slot_exhaustion(response) is not None


def _is_local_lease_active_response(response) -> bool:
    """Return True when response indicates local lease-active contention."""
    try:
        slot_info = _parse_slot_exhaustion(response)
        if isinstance(slot_info, dict):
            reason = str(slot_info.get("reason") or "").strip().lower()
            if reason == "local_lease_active":
                return True
    except Exception:
        pass

    # Fallback heuristic in case payload shape is unexpected.
    try:
        body_text = _response_body_text(response).lower()
        return "local_lease_active" in body_text
    except Exception:
        return False


def _resolve_provider_with_exclusions(
    model_config: dict,
    excluded_provider_names: set[str],
) -> Optional[dict]:
    """Resolve next available provider while excluding names tried this request."""
    providers: Optional[List[Dict[str, Any]]] = model_config.get("providers")
    if not providers:
        return None

    for provider_cfg in providers:
        name = provider_cfg.get("name", "")
        if name in excluded_provider_names:
            continue
        if _is_provider_unavailable(name):
            continue
        return provider_cfg
    return None


def _is_model_loading_response(response: Response, body_text: str) -> bool:
    """Return True when a 503 response represents transient model loading."""
    if int(getattr(response, "status_code", 0) or 0) != 503:
        return False

    try:
        payload = json.loads(body_text) if body_text else None
    except Exception:
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            code = str(error.get("code", "")).strip().lower()
            err_type = str(error.get("type", "")).strip().lower()
            message = str(error.get("message", "")).strip().lower()
            if code == "model_loading" or err_type == "model_loading":
                return True
            if "model" in message and "loading" in message:
                return True
        elif isinstance(error, str):
            lowered = error.strip().lower()
            if "model_loading" in lowered or ("model" in lowered and "loading" in lowered):
                return True

    lowered_body = (body_text or "").strip().lower()
    if "model_loading" in lowered_body:
        return True
    if "model" in lowered_body and "loading" in lowered_body:
        return True

    return False


# ---------------------------------------------------------------------------
# Shared fallback primitives (extracted from proxy_with_remote_fallback and
# proxy_with_fallback to eliminate duplicated state-machine logic)
# ---------------------------------------------------------------------------


def _record_attempt(attempts: List[Dict[str, Any]], **fields) -> None:
    """Append a diagnostic attempt entry to the attempts list.

    Each entry records which provider was tried, the outcome, and optional
    diagnostic payload (status code, body snippet, cooldown, etc.).
    """
    attempts.append(dict(fields))


def _handle_streaming_success(
    response: Response,
    provider_name: str,
    provider_type: str,
    attempts: List[Dict[str, Any]],
    prev_provider: Optional[str],
    fallback_reason: Optional[str],
    path: str,
) -> Optional[Response]:
    """If *response* is a 2xx StreamingResponse, record the attempt, add
    the ``X-Provider`` header, log the fallback (if one occurred), and
    return the augmented response.

    Returns ``None`` if *response* is **not** a streaming success (caller
    should continue normal processing).
    """
    if _is_streaming_response(response) and int(getattr(response, "status_code", 0) or 0) < 400:
        _record_attempt(
            attempts,
            provider=provider_name,
            type=provider_type,
            status="streaming_success",
            status_code=int(getattr(response, "status_code", 0) or 0),
        )
        # Reset exponential-backoff failure count on success
        _reset_provider_failure_count(provider_name)
        result = _add_provider_header(response, provider_name)
        if prev_provider:
            logger.info(
                "Fallback triggered for model=%s, from=%s, to=%s, reason=%s",
                path, prev_provider, provider_name, fallback_reason or "streaming",
            )
        return result
    return None


def _build_fallback_success_response(
    response: Response,
    provider_name: str,
    provider_type: str,
    attempts: List[Dict[str, Any]],
    prev_provider: Optional[str],
    fallback_reason: Optional[str],
    path: str,
    body_text: str = "",
    status_override: str = "success",
) -> Response:
    """Record a successful provider attempt, add the ``X-Provider`` header,
    log the fallback (if one occurred), and return the augmented response.

    This is the normal (non-streaming) success path used by both fallback
    entrypoints when a provider returns a successful response.
    """
    _record_attempt(
        attempts,
        provider=provider_name,
        type=provider_type,
        status=status_override,
        status_code=int(getattr(response, "status_code", 0) or 0),
        body_snippet=(body_text[:512] if body_text else None),
    )
    # Reset exponential-backoff failure count on success
    _reset_provider_failure_count(provider_name)
    result = _add_provider_header(response, provider_name)
    if prev_provider:
        logger.info(
            "Fallback triggered for model=%s, from=%s, to=%s, reason=%s",
            path,
            prev_provider,
            provider_name,
            fallback_reason or "unknown",
        )
    return result


def _handle_connection_error_in_fallback(
    exc: Exception,
    provider_name: str,
    provider_type: str,
    cooldown_seconds: float,
    attempts: List[Dict[str, Any]],
) -> bool:
    """If *exc* is a connection error, mark the provider unavailable, record
    a diagnostic attempt entry, and return ``True`` (caller should ``continue``
    to the next provider).

    Applies exponential backoff for remote providers (capped at configured
    *cooldown_seconds*, so setting it to 0 disables backoff entirely).

    Returns ``False`` if *exc* is **not** a connection error (caller should
    re-raise or handle differently).
    """
    if _is_connection_error(exc):
        cooldown = cooldown_seconds
        if provider_type == "remote" and cooldown_seconds > 0:
            count = _provider_failure_count.get(provider_name, 0)
            backoff = min(
                _BACKOFF_BASE_SECONDS * (2 ** count),
                _BACKOFF_MAX_SECONDS,
            )
            cooldown = min(backoff, cooldown_seconds)
            _provider_failure_count[provider_name] = count + 1
        mark_provider_unavailable(provider_name, cooldown)
        _record_attempt(
            attempts,
            provider=provider_name,
            type=provider_type,
            status="connection_error",
            error=str(type(exc).__name__),
        )
        return True
    return False


def _handle_http_error_with_cooldown(
    response: Response,
    provider_name: str,
    provider_type: str,
    cooldown_seconds: float,
    attempts: List[Dict[str, Any]],
    body_text: str,
) -> float:
    """Handle an HTTP error response: compute effective cooldown, mark the
    provider unavailable, record a diagnostic attempt entry, and return the
    effective cooldown duration.

    Applies exponential backoff for remote providers.

    The caller is responsible for setting ``fallback_reason``, ``prev_provider``,
    and ``all_slot_exhaustion`` after calling this function, and for issuing
    ``continue``.
    """
    # Parse Retry-After separately so we can respect it alongside backoff
    retry_after = _parse_retry_after(response)

    cooldown = cooldown_seconds
    if provider_type == "remote" and cooldown_seconds > 0:
        count = _provider_failure_count.get(provider_name, 0)
        backoff = min(
            _BACKOFF_BASE_SECONDS * (2 ** count),
            _BACKOFF_MAX_SECONDS,
        )
        cooldown = min(backoff, cooldown_seconds)
        _provider_failure_count[provider_name] = count + 1

    # Respect Retry-After header regardless of backoff
    if retry_after is not None:
        cooldown = max(cooldown, retry_after)

    mark_provider_unavailable(provider_name, cooldown)
    _record_attempt(
        attempts,
        provider=provider_name,
        type=provider_type,
        status="http_error",
        status_code=int(response.status_code),
        body_snippet=(body_text[:512] if body_text else None),
        cooldown_seconds=cooldown,
    )
    return cooldown


def _handle_empty_response_with_cooldown(
    response: Response,
    provider_name: str,
    provider_type: str,
    cooldown_seconds: float,
    attempts: List[Dict[str, Any]],
    body_text: str,
) -> float:
    """Handle an empty (non-reasoning) successful response: compute effective
    cooldown, mark the provider unavailable, record a diagnostic attempt entry,
    and return the effective cooldown duration.

    Applies exponential backoff for remote providers.

    The caller is responsible for setting ``fallback_reason``, ``prev_provider``,
    and ``all_slot_exhaustion`` after calling this function, and for issuing
    ``continue``.
    """
    # Parse Retry-After separately so we can respect it alongside backoff
    retry_after = _parse_retry_after(response)

    cooldown = cooldown_seconds
    if provider_type == "remote" and cooldown_seconds > 0:
        count = _provider_failure_count.get(provider_name, 0)
        backoff = min(
            _BACKOFF_BASE_SECONDS * (2 ** count),
            _BACKOFF_MAX_SECONDS,
        )
        cooldown = min(backoff, cooldown_seconds)
        _provider_failure_count[provider_name] = count + 1

    # Respect Retry-After header regardless of backoff
    if retry_after is not None:
        cooldown = max(cooldown, retry_after)

    mark_provider_unavailable(provider_name, cooldown)
    _record_attempt(
        attempts,
        provider=provider_name,
        type=provider_type,
        status="empty_response",
        status_code=int(getattr(response, "status_code", 0) or 0),
        body_snippet=(body_text[:512] if body_text else None),
        cooldown_seconds=cooldown,
    )
    return cooldown


def _resolve_reasoning_content_promotion(
    response: Response,
    provider_name: str,
    provider_type: str,
    attempts: List[Dict[str, Any]],
    prev_provider: Optional[str],
    fallback_reason: Optional[str],
    path: str,
    body_text: str,
) -> Optional[Response]:
    """If the response body contains ``reasoning_content``, treat this
    empty-but-meaningful response as a success (promote it).  Records the
    attempt, adds the provider header, logs the fallback, and returns the
    augmented response.

    Returns ``None`` if the body does **not** contain ``reasoning_content``
    (caller should continue with empty-response cooldown logic).
    """
    body_l = (body_text or "").lower()
    if "reasoning_content" in body_l:
        _record_attempt(
            attempts,
            provider=provider_name,
            type=provider_type,
            status="promoted_reasoning",
            status_code=int(getattr(response, "status_code", 0) or 0),
            body_snippet=(body_text[:512] if body_text else None),
        )
        # Reset exponential-backoff failure count on success
        _reset_provider_failure_count(provider_name)
        result = _add_provider_header(response, provider_name)
        if prev_provider:
            logger.info(
                "Fallback triggered for model=%s, from=%s, to=%s, reason=%s",
                path,
                prev_provider,
                provider_name,
                fallback_reason or "promoted_reasoning",
            )
        return result
    return None


def _log_exhausted_providers(model_config: dict, path: str) -> Dict[str, int]:
    """Log diagnostic details about which providers are in cooldown and return
    the mapping of provider name to remaining cooldown seconds.
    """
    unavailable: Dict[str, int] = {}
    try:
        provider_names = [p.get("name") for p in model_config.get("providers", []) if isinstance(p, dict)]
        for n in provider_names:
            exp = _provider_unavailable_until.get(n)
            if exp:
                unavailable[n] = int(max(0, exp - time.time()))
        logger.warning("All providers exhausted for model=%s; unavailable=%s", path, unavailable)
    except Exception:
        pass
    return unavailable


async def proxy_with_remote_fallback(
    request,
    path: str,
    model_config: dict,
    config: dict,
) -> Response:
    """Proxy a request to a remote model with provider fallback.

    Iterates through the model's configured providers (in order) and
    returns the first successful response.  On failure (connection error
    or HTTP status >= 400), the provider is marked with a cooldown and
    the next provider is tried.

    Args:
        request: The incoming FastAPI Request.
        path: The API path to proxy (e.g., ``v1/chat/completions``).
        model_config: Model configuration dict with a ``providers`` list.
        config: Server configuration dict (for ``provider_cooldown_seconds``).

    Returns:
        A ``Response`` from a successful provider, or a 503/429 error
        response if all providers are exhausted.
    """
    cooldown_seconds = _get_cooldown_seconds(config)
    all_slot_exhaustion = True
    any_provider_tried = False
    prev_provider: Optional[str] = None
    fallback_reason: Optional[str] = None

    ptr = _get_proxy_to_remote()

    # Diagnostics: record attempts (ordered) for inclusion in exhausted responses
    attempts: List[Dict[str, Any]] = []
    attempted_provider_names: set[str] = set()

    # Preserve first model-loading response so single-provider models
    # do not collapse into generic "All providers exhausted".
    first_model_loading_response: Optional[Response] = None

    while True:
        provider_cfg = _resolve_provider_with_exclusions(model_config, attempted_provider_names)
        if provider_cfg is None:
            break

        provider_name = provider_cfg.get("name", "unknown")
        attempted_provider_names.add(provider_name)
        provider_type = provider_cfg.get("type", "remote")
        try:
            # Mark that we attempted this provider
            any_provider_tried = True

            # Proactive rate-limit check for remote providers
            provider_rpm = int(provider_cfg.get("rate_limit_rpm", 0) or 0)
            if provider_rpm > 0:
                from proxy.rate_limiter import get_rate_limiter
                allowed = await get_rate_limiter().check_and_increment(
                    provider_name, provider_rpm, window_seconds=60
                )
                if not allowed:
                    logger.warning(
                        "Rate limited: skipping provider=%s model=%s (limit=%d rpm)",
                        provider_name, path, provider_rpm,
                    )
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type="remote",
                        status="rate_limited",
                        status_code=429,
                    )
                    continue

            response = await ptr(request, path, provider_cfg)

            # Shared primitive: handle streaming success
            stream_result = _handle_streaming_success(
                response, provider_name, provider_type, attempts,
                prev_provider, fallback_reason, path,
            )
            if stream_result is not None:
                return stream_result

            # Safely extract a small body snippet for diagnostics
            body_text = _response_body_text(response)

            # Check for HTTP error status
            if _is_http_error_status(response.status_code):
                if _is_model_loading_response(response, body_text):
                    fallback_reason = "model_loading"
                    prev_provider = provider_name
                    if first_model_loading_response is None:
                        first_model_loading_response = response
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="model_loading",
                        status_code=int(response.status_code),
                        body_snippet=(body_text[:512] if body_text else None),
                    )
                    all_slot_exhaustion = False
                    continue

                # Shared primitive: HTTP error with cooldown
                _handle_http_error_with_cooldown(
                    response, provider_name, provider_type,
                    cooldown_seconds, attempts, body_text,
                )
                fallback_reason = f"HTTP {response.status_code}"
                prev_provider = provider_name
                if response.status_code != 429:
                    all_slot_exhaustion = False
                continue

            # Treat empty successful responses as failures to allow fallback
            try:
                resp_json = None
                try:
                    resp_json = json.loads(body_text) if body_text else None
                except Exception:
                    resp_json = None
                if _is_empty_response(body_text or '', resp_json):
                    # Shared primitive: check for reasoning_content promotion
                    promoted = _resolve_reasoning_content_promotion(
                        response, provider_name, provider_type, attempts,
                        prev_provider, fallback_reason, path, body_text,
                    )
                    if promoted is not None:
                        # LP-0MR4ZIGDT004A3E1: Surface resolved provider/model for Pi extension
                        _add_resolved_model_header(promoted, provider_cfg)
                        return promoted

                    # Shared primitive: empty response with cooldown
                    _handle_empty_response_with_cooldown(
                        response, provider_name, provider_type,
                        cooldown_seconds, attempts, body_text,
                    )
                    fallback_reason = "empty_response"
                    prev_provider = provider_name
                    all_slot_exhaustion = False
                    continue
            except Exception:
                pass

            # Shared primitive: success path
            result = _build_fallback_success_response(
                response, provider_name, provider_type, attempts,
                prev_provider, fallback_reason, path, body_text,
            )
            # LP-0MR4ZIGDT004A3E1: Surface resolved provider/model for Pi extension
            _add_resolved_model_header(result, provider_cfg)
            return result

        except Exception as exc:
            # Shared primitive: handle connection errors
            if _handle_connection_error_in_fallback(
                exc, provider_name, provider_type, cooldown_seconds, attempts,
            ):
                any_provider_tried = True
                fallback_reason = str(type(exc).__name__)
                prev_provider = provider_name
                all_slot_exhaustion = False
                continue
            # Non-connection error — propagate
            raise

    # All providers exhausted — log diagnostic details
    unavailable = _log_exhausted_providers(model_config, path)

    if not any_provider_tried:
        return _build_exhausted_response(all_local_slot_exhaustion=False, unavailable_providers=unavailable, diagnostics=attempts)

    if first_model_loading_response is not None:
        logger.info(
            "Returning model_loading response instead of generic exhausted message for model=%s",
            path,
        )
        return first_model_loading_response

    return _build_exhausted_response(all_local_slot_exhaustion=all_slot_exhaustion, unavailable_providers=unavailable, diagnostics=attempts)


async def proxy_with_fallback(
    request,
    path: str,
    model_config: dict,
    config: dict,
) -> Response:
    """Proxy a request with fallback across both local and remote providers.

    Iterates through the model's configured providers (in order) and
    tries each one.  Local providers are dispatched via ``proxy_to_local``,
    remote providers via ``proxy_to_remote``.  On failure (connection error,
    HTTP status >= 400 for remote, slot exhaustion for local), the provider
    enters cooldown and the next provider is tried.

    Args:
        request: The incoming FastAPI Request.
        path: The API path to proxy.
        model_config: Model configuration dict with a ``providers`` list.
        config: Server configuration dict.

    Returns:
        A ``Response`` from a successful provider, or a 503/429 error
        response if all providers are exhausted.
    """
    cooldown_seconds = _get_cooldown_seconds(config)
    local_slot_retry_attempts = _get_local_slot_retry_attempts(config)
    local_slot_retry_delay_seconds = _get_local_slot_retry_delay_seconds(config)
    slot_unavailable_cooldown = _get_slot_unavailable_retry_after(config)
    all_slot_exhaustion = True
    any_provider_tried = False
    prev_provider: Optional[str] = None
    fallback_reason: Optional[str] = None

    # Accumulate slot counts when local providers report slot exhaustion
    total_slots_sum = 0
    available_slots_sum = 0

    # Track the first error response so we can return it when all providers
    # are exhausted, instead of the generic "All providers exhausted" message.
    # This preserves the actual error (e.g. backend_unavailable, concurrency)
    # that a single-provider model would have returned directly.
    _first_error_response = None

    ptr_remote = _get_proxy_to_remote()
    ptr_local = _get_proxy_to_local()

    # Diagnostics: record attempts (ordered) for inclusion in exhausted responses
    attempts: List[Dict[str, Any]] = []
    attempted_provider_names: set[str] = set()

    while True:
        provider_cfg = _resolve_provider_with_exclusions(model_config, attempted_provider_names)
        if provider_cfg is None and fallback_reason == "local_lease_active":
            # Local lease-active is expected contention, not provider failure.
            # For transparent fallback, allow trying the next remote provider
            # even if it is currently in cooldown.
            providers = model_config.get("providers") or []
            for candidate in providers:
                if not isinstance(candidate, dict):
                    continue
                candidate_name = candidate.get("name", "")
                if candidate_name in attempted_provider_names:
                    continue
                if candidate.get("type") != "remote":
                    continue
                provider_cfg = candidate
                break

        if provider_cfg is None:
            break

        provider_name = provider_cfg.get("name", "unknown")
        attempted_provider_names.add(provider_name)
        provider_type = provider_cfg.get("type", "remote")

        try:
            # Mark attempt
            any_provider_tried = True
            if provider_type == "local":
                # Queue bypass (LP-0MR5MAJNM005R905): if scheduler has no
                # idle slots, skip local provider immediately without marking
                # it as unavailable — the provider is busy, not failed.
                if not _get_scheduler_has_idle_slot():
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="slot_busy_skip_queue",
                    )
                    fallback_reason = "slot_busy_skip_queue"
                    prev_provider = provider_name
                    continue

                # Local concurrency limit check (LP-0MR5MAJNM005R905):
                # if local_max_concurrent_queries is exceeded, skip to next
                # provider without marking local as unavailable.
                cur_local, max_local = _get_local_concurrency_info(config)
                if cur_local >= max_local:
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="local_concurrency_limit",
                        active=cur_local,
                        max=max_local,
                    )
                    fallback_reason = "local_concurrency_limit"
                    prev_provider = provider_name
                    continue

                response = await ptr_local(request, path)
            else:
                # Proactive rate-limit check for remote providers
                # (LP-0MQNRDUP4008KT6T: rate limiter for remote models)
                provider_rpm = int(provider_cfg.get("rate_limit_rpm", 0) or 0)
                if provider_rpm > 0:
                    from proxy.rate_limiter import get_rate_limiter
                    allowed = await get_rate_limiter().check_and_increment(
                        provider_name, provider_rpm, window_seconds=60
                    )
                    if not allowed:
                        logger.warning(
                            "Rate limited: skipping provider=%s model=%s (limit=%d rpm)",
                            provider_name, path, provider_rpm,
                        )
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type="remote",
                            status="rate_limited",
                            status_code=429,
                        )
                        fallback_reason = "rate_limited"
                        prev_provider = provider_name
                        all_slot_exhaustion = False
                        continue

                response = await ptr_remote(request, path, provider_cfg)

            # Shared primitive: handle streaming success
            stream_result = _handle_streaming_success(
                response, provider_name, provider_type, attempts,
                prev_provider, fallback_reason, path,
            )
            if stream_result is not None:
                # LP-0MR4ZIGDT004A3E1: Surface resolved provider/model for Pi extension
                _add_resolved_model_header(stream_result, provider_cfg)
                return stream_result

            # Capture the first non-success response so we can return it when
            # all providers are exhausted (instead of the generic exhausted message).
            # Do not capture local slot-exhaustion responses here — they are
            # routing signals (busy/lease-active), not terminal provider errors.
            if _first_error_response is None and response.status_code >= 400:
                _first_slot_info = _parse_slot_exhaustion(response)
                if not (provider_type == "local" and _first_slot_info is not None):
                    _first_error_response = response

            # Extract small response snippet for diagnostics
            body_text = _response_body_text(response)

            # Check for slot exhaustion (local model)
            slot_info = _parse_slot_exhaustion(response)
            if slot_info:
                slot_reason = str(slot_info.get("reason") or "").strip().lower()

                # Lease-aware behavior: when local is reserved for another
                # session, do not retry local and do not put local in cooldown.
                # Route to the next provider in the chain immediately.
                if provider_type == "local" and slot_reason == "local_lease_active":
                    fallback_reason = "local_lease_active"
                    prev_provider = provider_name
                    total_slots_sum += int(slot_info.get("total_slots", 0) or 0)
                    available_slots_sum += int(slot_info.get("available_slots", 0) or 0)
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="local_lease_active",
                        slot_info=slot_info,
                    )
                    all_slot_exhaustion = False
                    continue

                # Optional local retry window for startup races where router/model
                # is loaded but slot probes briefly report 0 available.
                if provider_type == "local" and local_slot_retry_attempts > 0:
                    resolved_after_retry = False
                    for retry_idx in range(1, local_slot_retry_attempts + 1):
                        if local_slot_retry_delay_seconds > 0:
                            await asyncio.sleep(local_slot_retry_delay_seconds)

                        retry_response = await ptr_local(request, path)
                        retry_body_text = _response_body_text(retry_response)
                        retry_slot_info = _parse_slot_exhaustion(retry_response)

                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="slot_exhaustion_retry",
                            retry_attempt=retry_idx,
                            status_code=int(getattr(retry_response, "status_code", 0) or 0),
                            slot_info=retry_slot_info,
                            body_snippet=(retry_body_text[:512] if retry_body_text else None),
                        )

                        if retry_slot_info:
                            # Preserve lease-aware semantics during retry loop.
                            retry_reason = str(retry_slot_info.get("reason") or "").strip().lower()
                            if retry_reason == "local_lease_active":
                                slot_info = retry_slot_info
                                break
                            slot_info = retry_slot_info
                            continue

                        response = retry_response
                        body_text = retry_body_text
                        slot_info = None
                        resolved_after_retry = True
                        break

                    if resolved_after_retry and slot_info is None:
                        # Continue evaluating updated response below.
                        pass
                    else:
                        # If retries ended with lease-active, skip cooldown and
                        # route to next provider immediately.
                        final_reason = str((slot_info or {}).get("reason") or "").strip().lower()
                        if provider_type == "local" and final_reason == "local_lease_active":
                            fallback_reason = "local_lease_active"
                            prev_provider = provider_name
                            total_slots_sum += int((slot_info or {}).get("total_slots", 0) or 0)
                            available_slots_sum += int((slot_info or {}).get("available_slots", 0) or 0)
                            _record_attempt(
                                attempts,
                                provider=provider_name,
                                type=provider_type,
                                status="local_lease_active",
                                slot_info=slot_info,
                            )
                            all_slot_exhaustion = False
                            continue

                        mark_provider_unavailable(provider_name, slot_unavailable_cooldown)
                        fallback_reason = "slot_exhaustion"
                        prev_provider = provider_name
                        total_slots_sum += int(slot_info.get("total_slots", 0) or 0)
                        available_slots_sum += int(slot_info.get("available_slots", 0) or 0)
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="slot_exhaustion",
                            slot_info=slot_info,
                        )
                        continue
                else:
                    mark_provider_unavailable(provider_name, slot_unavailable_cooldown)
                    fallback_reason = "slot_exhaustion"
                    prev_provider = provider_name
                    total_slots_sum += int(slot_info.get("total_slots", 0) or 0)
                    available_slots_sum += int(slot_info.get("available_slots", 0) or 0)
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="slot_exhaustion",
                        slot_info=slot_info,
                    )
                    continue

            # Check for HTTP error status
            if _is_http_error_status(response.status_code):
                if _is_model_loading_response(response, body_text):
                    fallback_reason = "model_loading"
                    prev_provider = provider_name
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="model_loading",
                        status_code=int(response.status_code),
                        body_snippet=(body_text[:512] if body_text else None),
                    )
                    all_slot_exhaustion = False
                    continue

                # Local 5xx can be transient right after startup (slot routing,
                # backend warm-up). Retry local a few times before falling back.
                if provider_type == "local" and int(response.status_code) >= 500 and local_slot_retry_attempts > 0:
                    for retry_idx in range(1, local_slot_retry_attempts + 1):
                        if local_slot_retry_delay_seconds > 0:
                            await asyncio.sleep(local_slot_retry_delay_seconds)

                        retry_response = await ptr_local(request, path)
                        retry_body_text = _response_body_text(retry_response)
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="local_http_retry",
                            retry_attempt=retry_idx,
                            status_code=int(getattr(retry_response, "status_code", 0) or 0),
                            body_snippet=(retry_body_text[:512] if retry_body_text else None),
                        )

                        response = retry_response
                        body_text = retry_body_text
                        if not _is_http_error_status(response.status_code):
                            break

                    if _is_model_loading_response(response, body_text):
                        fallback_reason = "model_loading"
                        prev_provider = provider_name
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="model_loading",
                            status_code=int(response.status_code),
                            body_snippet=(body_text[:512] if body_text else None),
                        )
                        all_slot_exhaustion = False
                        continue

                if _is_http_error_status(response.status_code):
                    # Local 4xx responses are typically request-shape
                    # incompatibilities (e.g. optional OpenAI fields unsupported
                    # by llama-server), not provider health failures. Allow
                    # same-request fallback, but do not poison local provider
                    # cooldown across subsequent requests.
                    if provider_type == "local" and 400 <= int(response.status_code) < 500:
                        fallback_reason = f"HTTP {response.status_code}"
                        prev_provider = provider_name
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="http_error_no_cooldown",
                            status_code=int(response.status_code),
                            body_snippet=(body_text[:512] if body_text else None),
                        )
                        all_slot_exhaustion = False
                        continue

                    # Shared primitive: HTTP error with cooldown
                    _handle_http_error_with_cooldown(
                        response, provider_name, provider_type,
                        cooldown_seconds, attempts, body_text,
                    )
                    fallback_reason = f"HTTP {response.status_code}"
                    prev_provider = provider_name
                    if response.status_code != 429:
                        all_slot_exhaustion = False
                    continue

            # Treat empty successful responses as failures to allow fallback
            try:
                resp_json = None
                try:
                    resp_json = json.loads(body_text) if body_text else None
                except Exception:
                    resp_json = None
                if _is_empty_response(body_text or '', resp_json):
                    # Shared primitive: check for reasoning_content promotion
                    promoted = _resolve_reasoning_content_promotion(
                        response, provider_name, provider_type, attempts,
                        prev_provider, fallback_reason, path, body_text,
                    )
                    if promoted is not None:
                        # LP-0MR4ZIGDT004A3E1: Surface resolved provider/model for Pi extension
                        _add_resolved_model_header(promoted, provider_cfg)
                        return promoted

                    # Local empty 200 can be transient (slot busy/cancelled
                    # right after a previous request). Retry locally before
                    # falling back to remote providers.
                    if provider_type == "local" and local_slot_retry_attempts > 0:
                        resolved_after_empty_retry = False
                        for retry_idx in range(1, local_slot_retry_attempts + 1):
                            if local_slot_retry_delay_seconds > 0:
                                await asyncio.sleep(local_slot_retry_delay_seconds)
                            retry_response = await ptr_local(request, path)
                            retry_body_text = _response_body_text(retry_response)
                            _record_attempt(
                                attempts,
                                provider=provider_name,
                                type=provider_type,
                                status="local_empty_retry",
                                retry_attempt=retry_idx,
                                status_code=int(getattr(retry_response, "status_code", 0) or 0),
                                body_snippet=(retry_body_text[:512] if retry_body_text else None),
                            )
                            try:
                                retry_resp_json = json.loads(retry_body_text) if retry_body_text else None
                            except Exception:
                                retry_resp_json = None
                            if not _is_empty_response(retry_body_text or "", retry_resp_json):
                                response = retry_response
                                body_text = retry_body_text
                                resolved_after_empty_retry = True
                                break

                        if resolved_after_empty_retry:
                            # Shared primitive: check reasoning_content after retry
                            promoted2 = _resolve_reasoning_content_promotion(
                                response, provider_name, provider_type, attempts,
                                prev_provider, fallback_reason, path, body_text,
                            )
                            if promoted2 is not None:
                                # LP-0MR4ZIGDT004A3E1: Surface resolved provider/model
                                _add_resolved_model_header(promoted2, provider_cfg)
                                return promoted2
                            # Fall through to success path below.
                            pass
                        else:
                            # Shared primitive: empty response with cooldown
                            _handle_empty_response_with_cooldown(
                                response, provider_name, provider_type,
                                cooldown_seconds, attempts, body_text,
                            )
                            fallback_reason = "empty_response"
                            prev_provider = provider_name
                            all_slot_exhaustion = False
                            continue
                    else:
                        # Shared primitive: empty response with cooldown
                        _handle_empty_response_with_cooldown(
                            response, provider_name, provider_type,
                            cooldown_seconds, attempts, body_text,
                        )
                        fallback_reason = "empty_response"
                        prev_provider = provider_name
                        all_slot_exhaustion = False
                        continue
            except Exception:
                pass

            # Shared primitive: success path
            result = _build_fallback_success_response(
                response, provider_name, provider_type, attempts,
                prev_provider, fallback_reason, path, body_text,
            )
            # LP-0MR4ZIGDT004A3E1: Surface resolved provider/model for Pi extension
            _add_resolved_model_header(result, provider_cfg)
            return result

        except Exception as exc:
            # Shared primitive: handle connection errors
            if _handle_connection_error_in_fallback(
                exc, provider_name, provider_type, cooldown_seconds, attempts,
            ):
                any_provider_tried = True
                fallback_reason = str(type(exc).__name__)
                prev_provider = provider_name
                all_slot_exhaustion = False
                continue
            # HTTPException from the local provider (e.g., backend busy, slot
            # queue full, concurrency limit) should also trigger fallback.
            # Only 5xx responses are retryable via fallback; 4xx errors are
            # client errors that should propagate.
            if isinstance(exc, HTTPException) and exc.status_code >= 500:
                any_provider_tried = True

                # Local 5xx HTTPException can be transient (slot warm-up,
                # brief concurrency spike right after restart). Retry local a
                # few times before falling back to remote providers.
                if provider_type == "local" and local_slot_retry_attempts > 0:
                    retry_exc: Optional[Exception] = exc
                    resolved_response: Optional[Response] = None
                    for retry_idx in range(1, local_slot_retry_attempts + 1):
                        if local_slot_retry_delay_seconds > 0:
                            await asyncio.sleep(local_slot_retry_delay_seconds)
                        try:
                            retry_response = await ptr_local(request, path)
                        except Exception as inner_exc:
                            retry_exc = inner_exc
                            _record_attempt(
                                attempts,
                                provider=provider_name,
                                type=provider_type,
                                status="local_http_exception_retry",
                                retry_attempt=retry_idx,
                                error=str(type(inner_exc).__name__),
                            )
                            continue
                        retry_exc = None
                        resolved_response = retry_response
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="local_http_exception_retry",
                            retry_attempt=retry_idx,
                            status_code=int(getattr(retry_response, "status_code", 0) or 0),
                        )
                        break

                    if retry_exc is None and resolved_response is not None:
                        # Local retry succeeded; re-enter success-path checks by
                        # re-processing the resolved response.
                        response = resolved_response
                        body_text = _response_body_text(response)
                        slot_info = _parse_slot_exhaustion(response)
                        if slot_info is None and not _is_http_error_status(response.status_code):
                            # Success — record and return below via normal path.
                            result = _build_fallback_success_response(
                                response, provider_name, provider_type, attempts,
                                prev_provider, fallback_reason, path, body_text,
                                status_override="success_after_http_exception_retry",
                            )
                            # LP-0MR4ZIGDT004A3E1: Surface resolved provider/model
                            _add_resolved_model_header(result, provider_cfg)
                            return result
                        # Retry produced a response but still slot-exhaustion/error;
                        # fall through to normal handling by continuing the loop.
                        continue

                if _first_error_response is None:
                    # Capture the first HTTPException as an error response so
                    # the actual error (e.g. concurrency limit, slot queue) is
                    # preserved instead of replaced by the generic exhausted message.
                    _first_error_response = Response(
                        content=json.dumps({
                            "error": {
                                "type": "backend_error",
                                "code": "backend_error",
                                "message": str(exc.detail),
                            },
                            "status": exc.status_code,
                        }).encode("utf-8"),
                        status_code=exc.status_code,
                        media_type="application/json",
                    )
                mark_provider_unavailable(provider_name, cooldown_seconds)
                fallback_reason = f"HTTPException {exc.status_code}"
                prev_provider = provider_name
                _record_attempt(
                    attempts,
                    provider=provider_name,
                    type=provider_type,
                    status="http_exception",
                    status_code=exc.status_code,
                    error=str(exc),
                )
                all_slot_exhaustion = False
                continue
            # Non-connection error — propagate
            raise

    # All providers exhausted — log diagnostic details
    unavailable = _log_exhausted_providers(model_config, path)

    if not any_provider_tried:
        return _build_exhausted_response(all_local_slot_exhaustion=False, unavailable_providers=unavailable, diagnostics=attempts)

    # If all failures were slot exhaustion, include total slots in message
    if all_slot_exhaustion:
        return _build_exhausted_response(all_local_slot_exhaustion=True, total_slots=total_slots_sum, unavailable_providers=unavailable, diagnostics=attempts)

    # When all providers are exhausted, return the first provider's actual
    # error response instead of the generic "All providers exhausted"
    # message.  This preserves the real error (e.g. backend_unavailable,
    # concurrency limit, slot exhaustion, backend error) that the client
    # would have received from a single-provider model or direct call.
    #
    # Exception: if the first error is local lease-active contention and the
    # model has remote providers, do not return that local routing signal to
    # clients; prefer generic exhausted/remote error semantics.
    if _first_error_response is not None:
        has_remote_provider = any(
            isinstance(p, dict) and p.get("type") == "remote"
            for p in (model_config.get("providers") or [])
        )
        if has_remote_provider and _is_local_lease_active_response(_first_error_response):
            logger.info(
                "Suppressing local_lease_active first error response for model=%s; "
                "remote fallback chain present",
                path,
            )
        else:
            logger.info(
                "Returning first provider error response instead of generic exhausted "
                "message for model=%s (status=%s)",
                path, _first_error_response.status_code,
            )
            return _first_error_response

    return _build_exhausted_response(all_local_slot_exhaustion=False, unavailable_providers=unavailable, diagnostics=attempts)
