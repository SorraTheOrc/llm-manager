"""
Provider Module

Provider resolution and fallback logic for model requests.

Provides:
- `resolve_provider()`: Select the next available provider for a model config
- `proxy_with_remote_fallback()`: Remote provider fallback loop
- Cooldown tracking: Mark providers as temporarily unavailable after failures
"""

import json
import logging
import time
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import Response


logger = logging.getLogger("llama-proxy.provider")

# ---------------------------------------------------------------------------
# Cooldown / circuit-breaker state
# ---------------------------------------------------------------------------

# In-memory cooldown tracking: provider_name -> expiry_timestamp (seconds since epoch)
_provider_unavailable_until: Dict[str, float] = {}


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
) -> None:
    """Mark a provider as unavailable for the given cooldown duration.

    During the cooldown period ``resolve_provider()`` will skip this
    provider.

    Args:
        provider_name: Name of the provider to mark.
        cooldown_seconds: Number of seconds the provider should be
                          considered unavailable.
    """
    _provider_unavailable_until[provider_name] = time.time() + cooldown_seconds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_provider_unavailable(provider_name: str) -> bool:
    """Check if a provider is currently in cooldown.

    Returns ``True`` if the provider is marked unavailable and its cooldown
    has not yet expired.  Expired entries are cleaned up lazily.
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


def _add_provider_header(response: Response, provider_name: str) -> Response:
    """Add X-Provider header to a response."""
    response.headers.append("X-Provider", provider_name)
    return response


def _build_exhausted_response(all_local_slot_exhaustion: bool = False, total_slots: int = 0) -> Response:
    """Build the response when all providers are exhausted.

    Args:
        all_local_slot_exhaustion: If ``True``, all providers exhausted due to
                                   slot exhaustion (returns HTTP 429).
                                   Otherwise, returns HTTP 503 with JSON body.
        total_slots: Total number of slots across local providers (used only
                     for the slot-exhaustion 429 text body).
    """
    if all_local_slot_exhaustion:
        # total_slots may be 0 if unknown; still format per acceptance criteria
        return Response(
            content=(f"Model server busy: 0/{int(total_slots)} slots available. Retry later.").encode("utf-8"),
            status_code=429,
            media_type="text/plain",
        )
    return Response(
        content=json.dumps({"error": "All providers exhausted", "retry_after": 60}).encode("utf-8"),
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
    """Return an async callable that dispatches to the current proxy_to_local.

    We resolve the implementation at call time (dynamic import) so that
    unit tests that monkeypatch either ``proxy.router.proxy_to_local`` or
    ``proxy.server.proxy_to_local`` are both honored. This avoids fragile
    import-time binding and makes the provider fallback logic more robust
    in test environments.
    """
    async def _call(request, path):
        # Dynamic import to pick up runtime monkeypatches
        try:
            import importlib
            router_mod = importlib.import_module("proxy.router")
            fn_router = getattr(router_mod, "proxy_to_local", None)
        except Exception:
            fn_router = None
        try:
            import importlib as _importlib2
            server_mod = _importlib2.import_module("proxy.server")
            fn_server = getattr(server_mod, "proxy_to_local", None)
        except Exception:
            fn_server = None

        # Prefer a server-level patch (monkeypatching proxy.server.proxy_to_local)
        # if present; otherwise prefer a router-level patch. This order makes
        # tests that patch server.proxy_to_local succeed while still honoring
        # tests that patch router.proxy_to_local.
        chosen = None
        if fn_server is not None and fn_server is not fn_router:
            chosen = fn_server
        elif fn_router is not None:
            chosen = fn_router
        elif fn_server is not None:
            chosen = fn_server
        else:
            raise RuntimeError("proxy_to_local function is not available")

        return await chosen(request, path)

    return _call


def _parse_slot_exhaustion(response):
    """Parse a slot-exhaustion response and return slot info.

    Returns a dict with keys 'total_slots' and 'available_slots' when the
    response indicates slot exhaustion, otherwise returns None.
    """
    try:
        if response.status_code != 503:
            return None
        import json
        body = json.loads(response.body)
        error = body.get("error", {})
        if isinstance(error, dict) and error.get("code") == "no_slots_available":
            total = int(body.get("total_slots", 0) or 0)
            avail = int(body.get("available_slots", 0) or 0)
            return {"total_slots": total, "available_slots": avail}
    except Exception:
        pass
    return None


def _is_slot_exhaustion_response(response) -> bool:
    """Backward-compatible boolean check for slot exhaustion."""
    return _parse_slot_exhaustion(response) is not None


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

    while True:
        provider_cfg = resolve_provider(model_config)
        if provider_cfg is None:
            break

        provider_name = provider_cfg.get("name", "unknown")
        try:
            response = await ptr(request, path, provider_cfg)
            any_provider_tried = True

            # Check for HTTP error status
            if _is_http_error_status(response.status_code):
                effective_cooldown = _compute_cooldown(cooldown_seconds, response)
                mark_provider_unavailable(provider_name, effective_cooldown)
                fallback_reason = f"HTTP {response.status_code}"
                prev_provider = provider_name
                # Track whether the failure was slot-exhaustion-like
                if response.status_code != 429:
                    all_slot_exhaustion = False
                continue

            # Success — add X-Provider header and log if fallback occurred
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

        except Exception as exc:
            if _is_connection_error(exc):
                any_provider_tried = True
                mark_provider_unavailable(provider_name, cooldown_seconds)
                fallback_reason = str(type(exc).__name__)
                prev_provider = provider_name
                all_slot_exhaustion = False
                continue
            # Non-connection error — propagate
            raise

    # All providers exhausted
    if not any_provider_tried:
        # No providers were available at all (all in cooldown or none defined)
        return _build_exhausted_response(all_local_slot_exhaustion=False)
    return _build_exhausted_response(all_local_slot_exhaustion=all_slot_exhaustion)


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
    all_slot_exhaustion = True
    any_provider_tried = False
    prev_provider: Optional[str] = None
    fallback_reason: Optional[str] = None

    # Accumulate slot counts when local providers report slot exhaustion
    total_slots_sum = 0
    available_slots_sum = 0

    ptr_remote = _get_proxy_to_remote()
    ptr_local = _get_proxy_to_local()

    while True:
        provider_cfg = resolve_provider(model_config)
        if provider_cfg is None:
            break

        provider_name = provider_cfg.get("name", "unknown")
        provider_type = provider_cfg.get("type", "remote")

        try:
            if provider_type == "local":
                response = await ptr_local(request, path)
            else:
                response = await ptr_remote(request, path, provider_cfg)

            any_provider_tried = True

            # Check for slot exhaustion (local model)
            slot_info = _parse_slot_exhaustion(response)
            if slot_info:
                mark_provider_unavailable(provider_name, cooldown_seconds)
                fallback_reason = "slot_exhaustion"
                prev_provider = provider_name
                total_slots_sum += int(slot_info.get("total_slots", 0) or 0)
                available_slots_sum += int(slot_info.get("available_slots", 0) or 0)
                continue

            # Check for HTTP error status (remote provider)
            if _is_http_error_status(response.status_code):
                effective_cooldown = _compute_cooldown(cooldown_seconds, response)
                mark_provider_unavailable(provider_name, effective_cooldown)
                fallback_reason = f"HTTP {response.status_code}"
                prev_provider = provider_name
                if response.status_code != 429:
                    all_slot_exhaustion = False
                continue

            # Success — add X-Provider header and log if fallback occurred
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

        except Exception as exc:
            if _is_connection_error(exc):
                any_provider_tried = True
                mark_provider_unavailable(provider_name, cooldown_seconds)
                fallback_reason = str(type(exc).__name__)
                prev_provider = provider_name
                all_slot_exhaustion = False
                continue
            # Non-connection error — propagate
            raise

    # All providers exhausted
    if not any_provider_tried:
        return _build_exhausted_response(all_local_slot_exhaustion=False)
    # If all failures were slot exhaustion, include total slots in message
    if all_slot_exhaustion:
        return _build_exhausted_response(all_local_slot_exhaustion=True, total_slots=total_slots_sum)
    return _build_exhausted_response(all_local_slot_exhaustion=False)
