"""
Remote Proxy Module

Remote API proxying function (proxy_to_remote) extracted from the
monolithic router.py. Handles forwarding requests to remote API
endpoints (e.g., OpenAI, Anthropic) with streaming support.

Uses a lazy server import (_srv()) to access module-level state without
circular import issues.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from .router_helpers import (
    _get_request_preview,
    _srv,
    log_request,
    log_response,
    log_response_chunk,
    _compute_request_timeout,
    _schedule_recv_token_increment,
    _normalize_outgoing_headers,
    normalize_upstream_request_headers,
    _schedule_traffic_recording,
)

# Import utils functions used by this module
from proxy.utils import count_text_tokens  # noqa: E402


# ---------------------------------------------------------------------------
# Auth.json fallback helpers
# ---------------------------------------------------------------------------


def _get_auth_json_path() -> Path:
    """Return the path to pi's auth.json."""
    return Path.home() / ".pi" / "agent" / "auth.json"


def _try_pi_auth_json(provider_name: str) -> Optional[str]:
    """Attempt to resolve an API key from ~/.pi/agent/auth.json.

    Performs a case-insensitive lookup matching *provider_name* against
    keys in the auth JSON file.  Strip trailing ``_api_key`` suffix from
    *provider_name* before lookup.

    The resolution order follows the ``start-proxy.sh`` logic:
      1. Exact match (case-insensitive) on the provider name
      2. ``api_key_env``-style names (e.g. ``OPENCODE_API_KEY``) are matched
         by stripping the ``_api_key`` suffix and looking up the stem
         (e.g. ``OPENCODE`` -> ``opencode``)

    Returns the API key string (from the ``key`` field of a matching
    ``api_key``-type entry), or ``None`` if no match is found.
    """
    path = _get_auth_json_path()
    if not path.exists():
        return None

    try:
        auth_data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if not isinstance(auth_data, dict):
        return None

    lookup_key = provider_name.lower()

    # For OPENCODE_API_KEY-style env vars, prefer opencode-go then opencode
    if lookup_key == "opencode_api_key":
        for preferred in ("opencode-go", "opencode"):
            entry = auth_data.get(preferred)
            if isinstance(entry, dict) and entry.get("type") == "api_key":
                key = entry.get("key")
                if key:
                    return str(key)

    # Exact lowercase match
    entry = auth_data.get(lookup_key)
    if isinstance(entry, dict) and entry.get("type") == "api_key":
        key = entry.get("key")
        if key:
            return str(key)

    # Strip _API_KEY suffix and retry (for env-var-style names)
    if lookup_key.endswith("_api_key"):
        stem = lookup_key[:-8]
        entry = auth_data.get(stem)
        if isinstance(entry, dict) and entry.get("type") == "api_key":
            key = entry.get("key")
            if key:
                return str(key)

    return None


# A conservative OpenAI-compatible subset for remote chat completions.
# Unknown/experimental client keys can trigger 4xx on some providers.
_REMOTE_CHAT_FIELD_ALLOWLIST = {
    "model",
    "messages",
    "stream",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "n",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "seed",
    "logit_bias",
    "user",
    "reasoning_effort",
}


def _sanitize_remote_chat_payload(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize chat-completions payload for remote providers.

    Keeps a conservative OpenAI-compatible field subset for
    ``v1/chat/completions``. This improves cross-provider compatibility
    when clients include local-only or experimental fields.
    """
    if not isinstance(payload, dict):
        return payload
    if not (path == "v1/chat/completions" or str(path).endswith("chat/completions")):
        return payload

    sanitized = {k: v for k, v in payload.items() if k in _REMOTE_CHAT_FIELD_ALLOWLIST}
    dropped = sorted(k for k in payload.keys() if k not in sanitized)
    if dropped:
        try:
            _srv().logger.info(
                "[remote] stripped unsupported chat-completions fields: %s",
                ",".join(dropped),
            )
        except Exception:
            pass
    return sanitized


async def proxy_to_remote(
    request: Request,
    path: str,
    model_config: dict,
) -> Response:
    """Proxy request to remote API endpoint."""
    endpoint = model_config.get("endpoint", "")
    target_url = f"{endpoint}/{path}"

    # Get request body
    body = await request.body()

    # Log request (remote path has no slot concept; slot_id defaults to "none")
    log_request(request, body, "remote", endpoint)

    # Get API key
    api_key = None
    api_key_env = model_config.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env)
    if not api_key:
        api_key = model_config.get("api_key")
    if not api_key:
        # Fall back to pi's auth.json
        api_key = _try_pi_auth_json(api_key_env or "")

    # Forward headers (strip hop-by-hop transport headers)
    headers = normalize_upstream_request_headers(request.headers)

    # Remove local/proxy auth/session headers before forwarding.
    # In particular, prevent duplicate Authorization variants
    # (e.g. "authorization" + "Authorization") which can trigger
    # Cloudflare 400 Bad Request on upstream.
    for hk in list(headers.keys()):
        hkl = str(hk).lower()
        if hkl in {
            "authorization",
            "x-session-id",
            "x-client-request-id",
            "x-session-affinity",
            "session_id",
        }:
            headers.pop(hk, None)

    # Add API key
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Add custom headers from config
    custom_headers = model_config.get("headers", {})
    headers.update(custom_headers)

    body_json = json.loads(body) if body else {}
    if not isinstance(body_json, dict):
        body_json = {}

    # Sanitize request-shape for remote compatibility before model override.
    body_json = _sanitize_remote_chat_payload(path, body_json)

    # Override model name in body if provider config specifies an upstream model ID.
    # This allows the proxy to present a different model name to the remote API
    # than what the client originally sent (e.g. "deepseek-v4-flash-free" for a
    # model alias like "qwen3-fallback").
    upstream_model = model_config.get("model")
    if upstream_model and body_json.get("model"):
        body_json["model"] = upstream_model

    body = json.dumps(body_json).encode("utf-8")

    # Determine model name for attribution (may be provided in body)
    model_name = None
    try:
        model_name = body_json.get("model")
    except Exception:
        model_name = None
    if not model_name:
        model_name = _srv().current_model or model_config.get("name") or model_config.get("id") or "unknown"

    # Resolve session ID from headers for recording (LP-0MR8FEKK6005V9ML)
    _remote_session_id = (
        request.headers.get("x-session-id")
        or request.headers.get("session_id")
        or request.headers.get("x-client-request-id")
        or None
    )

    # Schedule fire-and-forget recording of client→proxy and proxy→provider requests
    if _remote_session_id:
        _schedule_traffic_recording(
            session_id=_remote_session_id,
            client_payload=body_json,
            proxy_payload=body_json,
            model=model_name,
            provider="remote",
        )

    server_config = _srv().config.get("server", {})
    remote_timeout = _compute_request_timeout(server_config, body_json, remote=True)
    is_streaming = body_json.get("stream", False)

    # LP-0MR4ZIGDT004A3E1: Build resolved model string for X-Resolved-Model header
    # Use the ``provider`` field (actual provider brand name) if present,
    # falling back to ``name`` (provider entry name) for backward compatibility.
    _provider_name = model_config.get("provider") or model_config.get("name", "unknown")
    _resolved_model_id = body_json.get("model", "unknown")
    _resolved_model_header = f"{_provider_name}/{_resolved_model_id}"
    # Warn when a remote provider entry is missing the ``provider`` field
    if not model_config.get("provider") and model_config.get("type") == "remote":
        _srv().logger.warning(
            "Remote provider entry %r is missing the 'provider' field; "
            "X-Resolved-Model header will use 'name' (%r) instead of the "
            "actual provider brand name. Add 'provider: <brand>' to the "
            "provider config to fix this.",
            model_config.get("name"),
            _provider_name,
        )

    # Read upstream idle timeout from config (LP-0MRE52D3C001KP1H)
    _upstream_idle_timeout = float(
        server_config.get("upstream_idle_timeout_seconds", 60) or 60
    )
    # Read upstream retry connect timeout from config (LP-0MRE8FYKV008WOTB)
    _upstream_retry_connect_timeout = float(
        server_config.get("upstream_retry_connect_timeout_seconds", 30) or 30
    )

    # Get shared HTTP connection pool for remote upstream requests (LP-0MRE8G3JK0099Y4J)
    _pool_client = getattr(_srv(), "_remote_http_client", None)

    if is_streaming:
        if _remote_session_id:
            return await _handle_remote_streaming(
                request, target_url, headers, body, body_json,
                model_name, remote_timeout,
                resolved_model=_resolved_model_header,
                session_id=_remote_session_id,
                provider=_provider_name,
                upstream_idle_timeout_seconds=_upstream_idle_timeout,
                upstream_retry_connect_timeout_seconds=_upstream_retry_connect_timeout,
                pool_client=_pool_client,
            )
        return await _handle_remote_streaming(
            request, target_url, headers, body, body_json,
            model_name, remote_timeout,
            resolved_model=_resolved_model_header,
            provider=_provider_name,
            upstream_idle_timeout_seconds=_upstream_idle_timeout,
            upstream_retry_connect_timeout_seconds=_upstream_retry_connect_timeout,
            pool_client=_pool_client,
        )
    else:
        if _remote_session_id:
            return await _handle_remote_non_streaming(
                request, target_url, headers, body, model_name, remote_timeout,
                resolved_model=_resolved_model_header,
                session_id=_remote_session_id,
                pool_client=_pool_client,
            )
        return await _handle_remote_non_streaming(
            request, target_url, headers, body, model_name, remote_timeout,
            resolved_model=_resolved_model_header,
            pool_client=_pool_client,
        )


async def _handle_remote_streaming(
    request: Request,
    target_url: str,
    headers: dict,
    body: bytes,
    body_json: dict,
    model_name: str,
    remote_timeout: httpx.Timeout,
    resolved_model: Optional[str] = None,
    session_id: Optional[str] = None,
    provider: Optional[str] = None,
    upstream_idle_timeout_seconds: Optional[float] = None,
    upstream_retry_connect_timeout_seconds: Optional[float] = None,
    pool_client: Optional[httpx.AsyncClient] = None,
) -> Response:
    """Handle streaming remote proxy request with upstream stall detection and retry.

    Features:
    - Per-chunk idle timeout: detects upstream silence within
      *upstream_idle_timeout_seconds* and closes the stalled connection.
    - Automatic retry: on stall detection (asyncio.TimeoutError) or httpx
      ReadTimeout, retries the same provider with bounded exponential backoff
      (1s, 2s, 4s; max 3 retries).
    - Fallthrough: after max retries exhausted, yields a synthetic
      ``finish_reason: error`` event so the caller (provider.py fallback chain)
      can route to the next provider.
    """
    # Resolve upstream_idle_timeout_seconds from parameter or config
    if upstream_idle_timeout_seconds is None:
        try:
            upstream_idle_timeout_seconds = float(
                _srv().config.get("server", {}).get(
                    "upstream_idle_timeout_seconds", 60
                ) or 60
            )
        except Exception:
            upstream_idle_timeout_seconds = 60.0

    # Resolve upstream_retry_connect_timeout_seconds from parameter or config
    if upstream_retry_connect_timeout_seconds is None:
        try:
            upstream_retry_connect_timeout_seconds = float(
                _srv().config.get("server", {}).get(
                    "upstream_retry_connect_timeout_seconds", 30
                ) or 30
            )
        except Exception:
            upstream_retry_connect_timeout_seconds = 30.0

    max_retries = 3
    retry_base_delay = 1.0
    retry_max_delay = 4.0

    # We need to manage client/context manager lifecycle for retries.
    # Use the shared pool client or create a fallback if unavailable.
    _pool_client = pool_client
    if _pool_client is None:
        _pool_client = httpx.AsyncClient(timeout=remote_timeout)
    _owns_client = pool_client is None  # Track if we need to close the client
    client = _pool_client
    cm = client.stream(
        request.method,
        target_url,
        headers=headers,
        content=body,
        timeout=remote_timeout,
    )

    response = await cm.__aenter__()
    upstream_status = response.status_code
    upstream_content_type = response.headers.get("content-type", "")

    # If upstream returned an error (or non-SSE payload), return a buffered response
    if upstream_status >= 400 or "text/event-stream" not in upstream_content_type.lower():
        try:
            body_bytes = await response.aread()
        except Exception:
            body_bytes = b""
        try:
            # Keep error-path visibility parity with non-streaming calls.
            log_response(upstream_status, body_bytes or b"")
            if upstream_status >= 400:
                err_preview = (body_bytes or b"").decode("utf-8", errors="replace")[:500]
                _srv().logger.warning(
                    "[remote] upstream error status=%s url=%s body=%s",
                    upstream_status,
                    target_url,
                    err_preview,
                )
        except Exception:
            pass
        try:
            await cm.__aexit__(None, None, None)
        except Exception:
            pass
        if _owns_client:
            try:
                await client.aclose()
            except Exception:
                pass
        # Record provider->client response for error path (fire-and-forget)
        if session_id:
            _schedule_traffic_recording(
                session_id=session_id,
                response_payload=body_bytes,
            )

        _err_headers = _normalize_outgoing_headers(dict(response.headers), buffered=True)
        # LP-0MR4ZIGDT004A3E1: Include resolved model info in error path
        if resolved_model:
            _err_headers["X-Resolved-Model"] = resolved_model
        return Response(
            content=body_bytes,
            status_code=upstream_status,
            headers=_err_headers,
        )

    outgoing_headers = _normalize_outgoing_headers(dict(response.headers), buffered=False)
    if "cache-control" not in {k.lower() for k in outgoing_headers.keys()}:
        outgoing_headers["Cache-Control"] = "no-cache"

    # LP-0MR4ZIGDT004A3E1: Surface resolved provider/model for Pi extension
    if resolved_model:
        outgoing_headers["X-Resolved-Model"] = resolved_model

    media_type = response.headers.get("content-type", "text/event-stream")
    key = f"{request.method.upper()} {request.url.path} -> remote"

    async def stream_generator():
        saw_done = False
        saw_finish = False
        # Client disconnect detection (LP-0MQTHP828000JYM6)
        disconnected = False
        _disconnect_check_count = 0
        # Collect chunks for session recording (LP-0MR94O16S000WFQ0)
        collected_chunks = [] if session_id else None

        # Log stream started with session context (LP-0MR90HJED005WI1Z)
        try:
            _request_preview = _get_request_preview(body_json)
            _srv().logger.info(
                "Stream started: provider=%s model=%s session=%s request=%s",
                provider or "remote",
                model_name,
                session_id or "unknown",
                _request_preview or "",
            )
        except Exception:
            pass

        # Per-chunk idle timeout and retry state (LP-0MRE52D3C001KP1H)
        _retry_count = 0
        _current_client = client
        _current_cm = cm
        _current_response = response
        _should_retry = False

        # Outer loop: retry on stall/ReadTimeout (initial attempt counts as
        # iteration 0; retries are iterations 1..max_retries)
        while True:
            if _retry_count >= max_retries:
                # Max retries exhausted — yield synthetic finish_reason: error
                # and stop. The caller (provider.py fallback chain) will see
                # this error event and route to the next provider.
                try:
                    _srv().logger.warning(
                        "Upstream stall: max retries exhausted session=%s provider=%s model=%s retries=%d",
                        session_id or "unknown",
                        provider or "remote",
                        model_name,
                        _retry_count,
                    )
                except Exception:
                    pass
                _final_error_obj = {
                    "choices": [
                        {"delta": {}, "finish_reason": "error", "index": 0}
                    ]
                }
                _final_error_bytes = (
                    f"data: {json.dumps(_final_error_obj)}\n\n"
                ).encode("utf-8")
                if collected_chunks is not None:
                    collected_chunks.append(_final_error_bytes)
                yield _final_error_bytes
                log_response_chunk(_final_error_bytes, session_id=session_id, model=model_name, provider=provider, body_json=body_json)
                break

            if _should_retry:
                _should_retry = False
                _retry_count += 1
                # Bounded exponential backoff
                _backoff_delay = min(
                    retry_base_delay * (2 ** (_retry_count - 1)),
                    retry_max_delay,
                )
                try:
                    _srv().logger.info(
                        "Upstream stall: retrying session=%s provider=%s model=%s attempt=%d backoff=%.1fs",
                        session_id or "unknown",
                        provider or "remote",
                        model_name,
                        _retry_count,
                        _backoff_delay,
                    )
                except Exception:
                    pass
                await asyncio.sleep(_backoff_delay)

                # Create fresh stream on the pool client for retry
                try:
                    _current_client = _pool_client
                    _current_cm = _pool_client.stream(
                        request.method,
                        target_url,
                        headers=headers,
                        content=body,
                        timeout=remote_timeout,
                    )
                    _current_response = await asyncio.wait_for(
                        _current_cm.__aenter__(),
                        timeout=upstream_retry_connect_timeout_seconds,
                    )
                    _retry_upstream_status = _current_response.status_code
                    _retry_upstream_ct = _current_response.headers.get("content-type", "")

                    if _retry_upstream_status >= 400 or "text/event-stream" not in _retry_upstream_ct.lower():
                        # Retry failed (non-streaming response) — retry loop will
                        # catch this and continue to next retry or max out.
                        try:
                            await _current_cm.__aexit__(None, None, None)
                        except Exception:
                            pass
                        try:
                            await _current_client.aclose()
                        except Exception:
                            pass
                        _retry_count += 1
                        if _retry_count >= max_retries:
                            continue  # Will exit on next outer iteration
                        _should_retry = True
                        continue
                except Exception as _reconnect_err:
                    # Connection failed on retry — continue retry loop
                    _retry_count += 1
                    if _retry_count >= max_retries:
                        continue  # Will exit on next outer iteration
                    _should_retry = True
                    continue

            # Inner loop: read chunks with per-chunk idle timeout
            # Initialize or reset per-stream state
            saw_done = False
            saw_finish = False
            disconnected = False
            _disconnect_check_count = 0

            try:
                _aiter = _current_response.aiter_bytes().__aiter__()
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            _aiter.__anext__(),
                            timeout=upstream_idle_timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        # Upstream stall detected — no data for
                        # upstream_idle_timeout_seconds. Break inner loop to
                        # trigger retry.
                        try:
                            _srv().logger.warning(
                                "Upstream stall detected: idle timeout session=%s "
                                "provider=%s model=%s timeout=%.1fs",
                                session_id or "unknown",
                                provider or "remote",
                                model_name,
                                upstream_idle_timeout_seconds,
                            )
                        except Exception:
                            pass
                        break

                    try:
                        s = chunk.decode("utf-8", errors="replace")
                        texts = []
                        for line in s.splitlines():
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == "[DONE]":
                                saw_done = True
                                continue
                            try:
                                j = json.loads(payload)
                                for choice in j.get("choices", []):
                                    if choice.get("finish_reason") is not None:
                                        saw_finish = True
                                for choice in j.get("choices", []):
                                    delta = choice.get("delta", {})
                                    if isinstance(delta, dict) and "content" in delta:
                                        texts.append(str(delta.get("content", "")))
                            except Exception:
                                texts.append(payload)
                        if texts:
                            chunk_text = "\n".join(texts)
                            chunk_tokens = count_text_tokens(chunk_text, model_name)
                            await _schedule_recv_token_increment(key, chunk_tokens)
                    except Exception:
                        pass

                    # Check for client disconnect periodically (LP-0MQTHP828000JYM6)
                    _disconnect_check_count += 1
                    if _disconnect_check_count % 10 == 0:
                        try:
                            _dc = await request.is_disconnected()
                            if isinstance(_dc, bool) and _dc:
                                disconnected = True
                                break
                        except Exception:
                            pass

                    if collected_chunks is not None:
                        collected_chunks.append(chunk)
                    yield chunk
                    log_response_chunk(chunk, session_id=session_id, model=model_name, provider=provider, body_json=body_json)

                    if saw_done or saw_finish:
                        break

                if disconnected:
                    # Client disconnected — stop streaming entirely, no retry
                    break

                if saw_done or saw_finish:
                    # Stream completed normally on this attempt — stop outer loop
                    break

                # If we break out of the inner loop without saw_done/saw_finish
                # and without disconnect, it's a stall (asyncio.TimeoutError).
                # Set retry flag to reconnect with backoff.
                _should_retry = True

            except StopAsyncIteration:
                # Normal exhaustion of the upstream iterator (no [DONE] received).
                # Synthesize final stop event as in the original code, then stop.
                if not saw_done and not saw_finish:
                    _final_stop_obj = {
                        "choices": [
                            {"delta": {}, "finish_reason": "stop", "index": 0}
                        ]
                    }
                    _final_stop_bytes = (
                        f"data: {json.dumps(_final_stop_obj)}\n\n"
                    ).encode("utf-8")
                    if collected_chunks is not None:
                        collected_chunks.append(_final_stop_bytes)
                    yield _final_stop_bytes
                    log_response_chunk(_final_stop_bytes, session_id=session_id, model=model_name, provider=provider, body_json=body_json)
                break
            except httpx.ReadTimeout:
                # httpx ReadTimeout before idle timeout (edge case) — retry
                try:
                    _srv().logger.warning(
                        "Upstream ReadTimeout session=%s provider=%s model=%s",
                        session_id or "unknown",
                        provider or "remote",
                        model_name,
                    )
                except Exception:
                    pass
                _should_retry = True
            except GeneratorExit:
                # Client disconnected or generator is being closed.
                # Skip the final event yield and proceed directly to cleanup.
                break
            except Exception as exc:
                # httpx stream error (e.g. RemoteProtocolError).
                # Yield a synthetic final SSE event so the client receives a
                # proper finish_reason marker even on stream error.
                # Do NOT retry on non-timeout errors.
                try:
                    _error_type = type(exc).__name__
                    _srv().logger.warning(
                        "Stream error: session=%s provider=%s model=%s error=%s",
                        session_id or "unknown",
                        provider or "remote",
                        model_name,
                        _error_type,
                    )
                except Exception:
                    pass
                _final_obj = {
                    "choices": [
                        {"delta": {}, "finish_reason": "error", "index": 0}
                    ]
                }
                _final_bytes = (
                    f"data: {json.dumps(_final_obj)}\n\n"
                ).encode("utf-8")
                if collected_chunks is not None:
                    collected_chunks.append(_final_bytes)
                yield _final_bytes
                log_response_chunk(_final_bytes, session_id=session_id, model=model_name, provider=provider, body_json=body_json)
                break
            finally:
                # Clean up the current connection (client+cm) after each
                # attempt. For the final attempt, this runs both here and
                # in the outer finally block; close() is idempotent.
                if not (_retry_count >= max_retries and not _should_retry):
                    # Only clean up if we might retry; final cleanup is in outer finally
                    if _should_retry or saw_done or saw_finish or disconnected:
                        try:
                            await _current_cm.__aexit__(None, None, None)
                        except Exception:
                            pass
                        if _owns_client:
                            try:
                                disconnect_cleanup_timeout = _srv().config.get("server", {}).get("disconnect_cleanup_timeout", 5.0)
                                await asyncio.wait_for(_current_client.aclose(), timeout=disconnect_cleanup_timeout)
                            except (asyncio.TimeoutError, Exception):
                                pass

            if saw_done or saw_finish or disconnected:
                break

            # If _should_retry is True, the outer loop will handle backoff
            # and reconnect on next iteration.

        # Finally block outside the while loop: ensures final cleanup of
        # the last active connection.
        try:
            await _current_cm.__aexit__(None, None, None)
        except Exception:
            try:
                await _current_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if _owns_client:
            try:
                disconnect_cleanup_timeout = _srv().config.get("server", {}).get("disconnect_cleanup_timeout", 5.0)
                await asyncio.wait_for(_current_client.aclose(), timeout=disconnect_cleanup_timeout)
            except (asyncio.TimeoutError, Exception):
                pass

        # Record provider->client response for streaming path (fire-and-forget)
        if session_id and collected_chunks is not None:
            response_body = b"".join(collected_chunks)
            _schedule_traffic_recording(
                session_id=session_id,
                response_payload=response_body,
                model=model_name,
                provider="remote",
            )

    return StreamingResponse(
        stream_generator(),
        media_type=media_type,
        headers=outgoing_headers,
        status_code=upstream_status,
    )


async def _handle_remote_non_streaming(
    request: Request,
    target_url: str,
    headers: dict,
    body: bytes,
    model_name: str,
    remote_timeout: httpx.Timeout,
    resolved_model: Optional[str] = None,
    session_id: Optional[str] = None,
    pool_client: Optional[httpx.AsyncClient] = None,
) -> Response:
    """Handle non-streaming remote proxy request."""
    key = f"{request.method.upper()} {request.url.path} -> remote"
    
    if pool_client is not None:
        method = request.method.lower()
        response = await getattr(pool_client, method)(
            target_url,
            headers=headers,
            content=body,
            timeout=remote_timeout,
        )
    else:
        async with httpx.AsyncClient(timeout=remote_timeout) as client:
            method = request.method.lower()
            response = await getattr(client, method)(
                target_url,
                headers=headers,
                content=body,
            )

        # Non-streaming: count tokens in response
        try:
            resp_text = response.content.decode("utf-8", errors="replace")
            recv_tokens = count_text_tokens(resp_text, model_name)
            await _schedule_recv_token_increment(key, recv_tokens)
        except Exception:
            pass

        log_response(response.status_code, response.content)

        # Record provider->client response (fire-and-forget)
        if session_id:
            _schedule_traffic_recording(
                session_id=session_id,
                response_payload=response.content,
                model=model_name,
                provider="remote",
            )

        _ns_headers = _normalize_outgoing_headers(dict(response.headers), buffered=True)
        # LP-0MR4ZIGDT004A3E1: Surface resolved provider/model for Pi extension
        if resolved_model:
            _ns_headers["X-Resolved-Model"] = resolved_model
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=_ns_headers,
        )



