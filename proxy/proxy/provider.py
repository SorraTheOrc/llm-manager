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
from fastapi import HTTPException, Response

from proxy.utils import _is_empty_response

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
    unavailable: Dict[str, int] = {}
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
            response = await ptr(request, path, provider_cfg)

            # Safely extract a small body snippet for diagnostics
            body_text = ""
            try:
                if hasattr(response, 'content'):
                    b = response.content
                elif hasattr(response, 'body'):
                    b = response.body
                else:
                    b = None
                if b:
                    body_text = b.decode('utf-8', errors='replace') if isinstance(b, (bytes, bytearray)) else str(b)
            except Exception:
                body_text = ""

            # Check for HTTP error status
            if _is_http_error_status(response.status_code):
                if _is_model_loading_response(response, body_text):
                    fallback_reason = "model_loading"
                    prev_provider = provider_name
                    if first_model_loading_response is None:
                        first_model_loading_response = response
                    attempts.append({
                        "provider": provider_name,
                        "type": provider_type,
                        "status": "model_loading",
                        "status_code": int(response.status_code),
                        "body_snippet": (body_text[:512] if body_text else None),
                    })
                    all_slot_exhaustion = False
                    continue

                effective_cooldown = _compute_cooldown(cooldown_seconds, response)
                mark_provider_unavailable(provider_name, effective_cooldown)
                fallback_reason = f"HTTP {response.status_code}"
                prev_provider = provider_name
                # Record diagnostic
                attempts.append({
                    "provider": provider_name,
                    "type": provider_type,
                    "status": "http_error",
                    "status_code": int(response.status_code),
                    "body_snippet": (body_text[:512] if body_text else None),
                    "cooldown_seconds": effective_cooldown,
                })
                # Track whether the failure was slot-exhaustion-like
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
                    # If the upstream included reasoning_content in the payload,
                    # promote it and return success instead of marking provider down.
                    body_l = (body_text or "").lower()
                    if "reasoning_content" in body_l:
                        attempts.append({
                            "provider": provider_name,
                            "type": provider_type,
                            "status": "promoted_reasoning",
                            "status_code": int(getattr(response, 'status_code', 0) or 0),
                            "body_snippet": (body_text[:512] if body_text else None),
                        })
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

                    effective_cooldown = _compute_cooldown(cooldown_seconds, response)
                    mark_provider_unavailable(provider_name, effective_cooldown)
                    fallback_reason = "empty_response"
                    prev_provider = provider_name
                    attempts.append({
                        "provider": provider_name,
                        "type": provider_type,
                        "status": "empty_response",
                        "status_code": int(getattr(response, 'status_code', 0) or 0),
                        "body_snippet": (body_text[:512] if body_text else None),
                        "cooldown_seconds": effective_cooldown,
                    })
                    all_slot_exhaustion = False
                    continue
            except Exception:
                pass

            # Success — record diagnostic and return
            attempts.append({
                "provider": provider_name,
                "type": provider_type,
                "status": "success",
                "status_code": int(getattr(response, 'status_code', 0) or 0),
                "body_snippet": (body_text[:512] if body_text else None),
            })

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
            # Connection errors: mark provider unavailable and continue
            if _is_connection_error(exc):
                any_provider_tried = True
                mark_provider_unavailable(provider_name, cooldown_seconds)
                fallback_reason = str(type(exc).__name__)
                prev_provider = provider_name
                attempts.append({
                    "provider": provider_name,
                    "type": provider_type,
                    "status": "connection_error",
                    "error": str(type(exc).__name__),
                })
                all_slot_exhaustion = False
                continue
            # Non-connection error — propagate
            raise

    # All providers exhausted
    # Log diagnostic details about provider cooldowns to aid troubleshooting
    try:
        provider_names = [p.get('name') for p in model_config.get('providers', []) if isinstance(p, dict)]
        unavailable = {}
        for n in provider_names:
            exp = _provider_unavailable_until.get(n)
            if exp:
                unavailable[n] = int(max(0, exp - time.time()))
        logger.warning("All providers exhausted for model=%s; unavailable=%s", path, unavailable)
    except Exception:
        pass

    if not any_provider_tried:
        # No providers were available at all (all in cooldown or none defined)
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
    unavailable: Dict[str, int] = {}
    attempted_provider_names: set[str] = set()

    while True:
        provider_cfg = _resolve_provider_with_exclusions(model_config, attempted_provider_names)
        if provider_cfg is None:
            break

        provider_name = provider_cfg.get("name", "unknown")
        attempted_provider_names.add(provider_name)
        provider_type = provider_cfg.get("type", "remote")

        try:
            # Mark attempt
            any_provider_tried = True
            if provider_type == "local":
                response = await ptr_local(request, path)
            else:
                response = await ptr_remote(request, path, provider_cfg)

            # Capture the first non-success response so we can return it when
            # all providers are exhausted (instead of the generic exhausted message).
            if _first_error_response is None and response.status_code >= 400:
                _first_error_response = response

            # Extract small response snippet for diagnostics
            body_text = ""
            try:
                if hasattr(response, 'content'):
                    b = response.content
                elif hasattr(response, 'body'):
                    b = response.body
                else:
                    b = None
                if b:
                    body_text = b.decode('utf-8', errors='replace') if isinstance(b, (bytes, bytearray)) else str(b)
            except Exception:
                body_text = ""

            # Check for slot exhaustion (local model)
            slot_info = _parse_slot_exhaustion(response)
            if slot_info:
                mark_provider_unavailable(provider_name, cooldown_seconds)
                fallback_reason = "slot_exhaustion"
                prev_provider = provider_name
                total_slots_sum += int(slot_info.get("total_slots", 0) or 0)
                available_slots_sum += int(slot_info.get("available_slots", 0) or 0)
                attempts.append({
                    "provider": provider_name,
                    "type": provider_type,
                    "status": "slot_exhaustion",
                    "slot_info": slot_info,
                })
                continue

            # Check for HTTP error status (remote provider)
            if _is_http_error_status(response.status_code):
                if _is_model_loading_response(response, body_text):
                    fallback_reason = "model_loading"
                    prev_provider = provider_name
                    attempts.append({
                        "provider": provider_name,
                        "type": provider_type,
                        "status": "model_loading",
                        "status_code": int(response.status_code),
                        "body_snippet": (body_text[:512] if body_text else None),
                    })
                    all_slot_exhaustion = False
                    continue

                effective_cooldown = _compute_cooldown(cooldown_seconds, response)
                mark_provider_unavailable(provider_name, effective_cooldown)
                fallback_reason = f"HTTP {response.status_code}"
                prev_provider = provider_name
                attempts.append({
                    "provider": provider_name,
                    "type": provider_type,
                    "status": "http_error",
                    "status_code": int(response.status_code),
                    "body_snippet": (body_text[:512] if body_text else None),
                    "cooldown_seconds": effective_cooldown,
                })
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
                    # If the upstream included reasoning_content in the payload,
                    # promote it and return success instead of marking provider down.
                    body_l = (body_text or "").lower()
                    if "reasoning_content" in body_l:
                        attempts.append({
                            "provider": provider_name,
                            "type": provider_type,
                            "status": "promoted_reasoning",
                            "status_code": int(getattr(response, 'status_code', 0) or 0),
                            "body_snippet": (body_text[:512] if body_text else None),
                        })
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

                    effective_cooldown = _compute_cooldown(cooldown_seconds, response)
                    mark_provider_unavailable(provider_name, effective_cooldown)
                    fallback_reason = "empty_response"
                    prev_provider = provider_name
                    attempts.append({
                        "provider": provider_name,
                        "type": provider_type,
                        "status": "empty_response",
                        "status_code": int(getattr(response, 'status_code', 0) or 0),
                        "body_snippet": (body_text[:512] if body_text else None),
                        "cooldown_seconds": effective_cooldown,
                    })
                    all_slot_exhaustion = False
                    continue
            except Exception:
                pass

            # Success — record diagnostic and return
            attempts.append({
                "provider": provider_name,
                "type": provider_type,
                "status": "success",
                "status_code": int(getattr(response, 'status_code', 0) or 0),
                "body_snippet": (body_text[:512] if body_text else None),
            })

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
                attempts.append({
                    "provider": provider_name,
                    "type": provider_type,
                    "status": "connection_error",
                    "error": str(type(exc).__name__),
                })
                all_slot_exhaustion = False
                continue
            # HTTPException from the local provider (e.g., backend busy, slot
            # queue full, concurrency limit) should also trigger fallback.
            # Only 5xx responses are retryable via fallback; 4xx errors are
            # client errors that should propagate.
            if isinstance(exc, HTTPException) and exc.status_code >= 500:
                any_provider_tried = True
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
                attempts.append({
                    "provider": provider_name,
                    "type": provider_type,
                    "status": "http_exception",
                    "status_code": exc.status_code,
                    "error": str(exc),
                })
                all_slot_exhaustion = False
                continue
            # Non-connection error — propagate
            raise

    # All providers exhausted
    # Log diagnostic details about provider cooldowns to aid troubleshooting
    try:
        provider_names = [p.get('name') for p in model_config.get('providers', []) if isinstance(p, dict)]
        unavailable = {}
        for n in provider_names:
            exp = _provider_unavailable_until.get(n)
            if exp:
                unavailable[n] = int(max(0, exp - time.time()))
        logger.warning("All providers exhausted for model=%s; unavailable=%s", path, unavailable)
    except Exception:
        pass

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
    if _first_error_response is not None:
        logger.info(
            "Returning first provider error response instead of generic exhausted "
            "message for model=%s (status=%s)",
            path, _first_error_response.status_code,
        )
        return _first_error_response

    return _build_exhausted_response(all_local_slot_exhaustion=False, unavailable_providers=unavailable, diagnostics=attempts)
