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
    _srv,
    log_request,
    log_response,
    log_response_chunk,
    _schedule_recv_token_increment,
    _normalize_outgoing_headers,
    normalize_upstream_request_headers,
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

    # Log request
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

    remote_timeout = httpx.Timeout(_srv().config.get("server", {}).get("llama_request_timeout", 300))
    is_streaming = body_json.get("stream", False)

    if is_streaming:
        return await _handle_remote_streaming(
            request, target_url, headers, body, body_json,
            model_name, remote_timeout,
        )
    else:
        return await _handle_remote_non_streaming(
            request, target_url, headers, body, model_name, remote_timeout,
        )


async def _handle_remote_streaming(
    request: Request,
    target_url: str,
    headers: dict,
    body: bytes,
    body_json: dict,
    model_name: str,
    remote_timeout: httpx.Timeout,
) -> Response:
    """Handle streaming remote proxy request."""
    client = httpx.AsyncClient(timeout=remote_timeout)
    cm = client.stream(
        request.method,
        target_url,
        headers=headers,
        content=body,
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
        try:
            await client.aclose()
        except Exception:
            pass
        return Response(
            content=body_bytes,
            status_code=upstream_status,
            headers=_normalize_outgoing_headers(dict(response.headers), buffered=True),
        )

    outgoing_headers = _normalize_outgoing_headers(dict(response.headers), buffered=False)
    if "cache-control" not in {k.lower() for k in outgoing_headers.keys()}:
        outgoing_headers["Cache-Control"] = "no-cache"

    media_type = response.headers.get("content-type", "text/event-stream")
    key = f"{request.method.upper()} {request.url.path} -> remote"

    async def stream_generator():
        saw_done = False
        saw_finish = False
        # Client disconnect detection (LP-0MQTHP828000JYM6)
        disconnected = False
        _disconnect_check_count = 0
        try:
            async for chunk in response.aiter_bytes():
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

                yield chunk
                log_response_chunk(chunk)
            # Synthesize final SSE event if upstream closed without finish marker.
            # Skip if client disconnected (LP-0MQTHP828000JYM6)
            if not disconnected and not saw_done and not saw_finish:
                final_obj = {
                    "choices": [
                        {"delta": {}, "finish_reason": "stop", "index": 0}
                    ]
                }
                final_bytes = (
                    f"data: {json.dumps(final_obj)}\n\n"
                ).encode("utf-8")
                yield final_bytes
                log_response_chunk(final_bytes)
        except GeneratorExit:
            # Client disconnected or generator is being closed.
            # Skip the final event yield and proceed directly to cleanup.
            pass
        finally:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:
                    pass
            try:
                disconnect_cleanup_timeout = _srv().config.get("server", {}).get("disconnect_cleanup_timeout", 5.0)
                await asyncio.wait_for(client.aclose(), timeout=disconnect_cleanup_timeout)
            except (asyncio.TimeoutError, Exception):
                pass

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
) -> Response:
    """Handle non-streaming remote proxy request."""
    key = f"{request.method.upper()} {request.url.path} -> remote"
    
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

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=_normalize_outgoing_headers(dict(response.headers), buffered=True),
        )



