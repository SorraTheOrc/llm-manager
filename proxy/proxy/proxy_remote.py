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
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

# Import utils functions used by this module
from proxy.utils import count_text_tokens

from .router_helpers import (
    _compute_request_timeout,
    _get_request_preview,
    _normalize_outgoing_headers,
    _schedule_recv_token_increment,
    _schedule_traffic_recording,
    _srv,
    log_request,
    log_response,
    log_response_chunk,
    normalize_upstream_request_headers,
)

# Import stall circuit breaker for Tier 3 cross-request stall tracking
from .stall_circuit_breaker import _check_stall_circuit_breaker

# ---------------------------------------------------------------------------
# Auth.json fallback helpers
# ---------------------------------------------------------------------------


def _get_auth_json_path() -> Path:
    """Return the path to pi's auth.json."""
    return Path.home() / ".pi" / "agent" / "auth.json"


def _try_pi_auth_json(provider_name: str) -> str | None:
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


def _sanitize_remote_chat_payload(path: str, payload: dict[str, Any]) -> dict[str, Any]:
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


def _sanitize_remote_messages(messages: list[Any]) -> list[Any]:
    """Sanitize accumulated chat message history for remote sends.

    Remote providers (opencode zen/go, api.deepseek.com) reject malformed
    tool-call/tool-result sequences with HTTP 400 (LP-0MSC1BNP90017L9K).
    RCA (F1) confirmed the rejected shapes:

      - tool message missing ``tool_call_id``
      - tool message with dangling ``tool_call_id`` (no matching assistant tool_calls)
      - assistant ``tool_calls`` entry missing ``id``
      - empty ``tool_calls`` array

    Hybrid policy (always-on, no config flag):

      - **Repair** where unambiguous: assistant ``content: null`` -> ``""``
        when ``tool_calls`` present; missing ``function.arguments`` -> ``""``;
        missing ``type`` -> ``"function"``.
      - **Prune** where not: tool messages with missing/dangling
        ``tool_call_id``; assistant ``tool_calls`` entries missing ``id``;
        empty ``tool_calls`` arrays (key removed).
      - **Preserve** truncated ``function.arguments`` JSON — RCA showed it is
        accepted by both zen/go and deepseek; do not alter valid semantics.

    Valid tool-call sequences pass through unchanged (regression guards:
    LP-0MS8XAPXT009W3CL, LP-0MQP3Q8DN0047J1H).

    Each mutation is logged at DEBUG for diagnosability.
    """
    if not isinstance(messages, list):
        return messages

    def _log(action: str, index: int, detail: str) -> None:
        try:
            _srv().logger.debug(
                "[remote] sanitizer: %s messages[%d] %s", action, index, detail,
            )
        except Exception:
            pass

    # First pass: sanitize assistant messages and collect valid tool_call ids.
    valid_tool_call_ids: set[str] = set()
    sanitized: list[Any] = []
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            sanitized.append(msg)
            continue
        role = msg.get("role")
        if role == "assistant":
            cleaned = dict(msg)
            # reasoning_content round-trip repair (LP-0MSGU3JNU0092AFQ):
            # remote thinking-mode providers (Console / Console Go / deepseek)
            # reject the whole request with HTTP 400 when ANY assistant message
            # lacks the ``reasoning_content`` field. Clients (e.g. opencode)
            # drop the empty ``reasoning_content: ""`` that the upstream emitted
            # on tool-call-only turns, so the field is absent on those messages
            # when the history is re-sent. Inject ``""`` (matching upstream
            # emission) where the field is missing or null — additive only;
            # existing values are never touched.
            if cleaned.get("reasoning_content") is None:
                _log("repair", index, "missing/null reasoning_content -> ''")
                cleaned["reasoning_content"] = ""
            tool_calls = cleaned.get("tool_calls")
            if isinstance(tool_calls, list):
                if not tool_calls:
                    # Empty tool_calls array -> 400; remove the key entirely.
                    _log("prune", index, "empty tool_calls array")
                    del cleaned["tool_calls"]
                else:
                    # Repair content:null -> "" when tool_calls present.
                    if cleaned.get("content") is None:
                        _log("repair", index, "content null -> ''")
                        cleaned["content"] = ""

                    valid_entries: list[Any] = []
                    for entry in tool_calls:
                        if not isinstance(entry, dict):
                            _log("prune", index, "non-dict tool_calls entry")
                            continue
                        if not entry.get("id"):
                            # RCA: missing id -> 400 on both zen/go and deepseek.
                            _log("prune", index, "tool_calls entry missing id")
                            continue
                        entry = dict(entry)
                        if not entry.get("type"):
                            _log("repair", index, f"tool_calls[{entry.get('id')}] missing type -> 'function'")
                            entry["type"] = "function"
                        fn = entry.get("function")
                        if not isinstance(fn, dict):
                            _log("prune", index, f"tool_calls[{entry.get('id')}] missing function")
                            continue
                        fn = dict(fn)
                        if not fn.get("name"):
                            _log("prune", index, f"tool_calls[{entry.get('id')}] missing function.name")
                            continue
                        if "arguments" not in fn:
                            _log("repair", index, f"tool_calls[{entry.get('id')}] missing arguments -> ''")
                            fn["arguments"] = ""
                        entry["function"] = fn
                        valid_entries.append(entry)
                        valid_tool_call_ids.add(str(entry["id"]))

                    if valid_entries:
                        cleaned["tool_calls"] = valid_entries
                    else:
                        _log("prune", index, "all tool_calls entries invalid")
                        del cleaned["tool_calls"]
            sanitized.append(cleaned)
        else:
            sanitized.append(msg)

    # Second pass: prune tool messages with missing or dangling tool_call_id.
    result: list[Any] = []
    for index, msg in enumerate(sanitized):
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tool_call_id = msg.get("tool_call_id")
            if not tool_call_id:
                _log("prune", index, "tool message missing tool_call_id")
                continue
            if str(tool_call_id) not in valid_tool_call_ids:
                _log("prune", index, f"tool message dangling tool_call_id {tool_call_id}")
                continue
        result.append(msg)
    return result


# ---------------------------------------------------------------------------
# OpenAI Responses API translation (LP-0MTGK5DQO001Y8H0)
# ---------------------------------------------------------------------------
#
# Some remote providers (opencode.ai zen/go for the muse model family) expose
# ONLY the Responses API (``/v1/responses``). Chat-completions requests to
# those models return HTTP 500, so the proxy translates between the two wire
# formats:
#
#   - Request: chat/completions body --messages--> ``input`` items,
#     ``max_tokens`` --> ``max_output_tokens``, tool-role messages -->
#     ``function_call_output`` items, assistant ``tool_calls`` -->
#     ``function_call`` items.
#   - Streaming response: Responses SSE events (``response.output_text.delta``,
#     ``response.function_call_arguments.delta``, ``response.completed``) are
#     converted into chat/completions SSE chunks so the existing proxy
#     fallback/watchdog/retry machinery keeps working unchanged.
#   - Non-streaming response: Responses JSON (``output[]``, ``usage``) is
#     mapped to the chat/completions JSON shape.
#
# Enable per-provider with ``api: openai-responses`` on the remote provider
# config entry. See proxy/docs/routing.md and proxy/config.yaml.
# ---------------------------------------------------------------------------


def _normalize_responses_content(content: Any) -> Any:
    """Normalize chat-format message content to Responses-API input content.

    The Responses API ``input`` items accept content as a plain string or an
    array of typed parts (``input_text`` / ``input_image`` / ``input_file``).
    Chat/completions messages use ``text`` / ``image_url`` part types, which
    the Responses API rejects with ``input[N].content did not match any
    supported type``. Mapping:

      - ``{"type": "text", "text": ...}``  -> ``{"type": "input_text", "text": ...}``
      - ``{"type": "image_url", "image_url": {"url": ...}}`` -> ``{"type": "input_image", "image_url": <url>}``
      - strings and already-valid parts pass through unchanged.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    parts: list[Any] = []
    for part in content:
        if not isinstance(part, dict):
            parts.append(part)
            continue
        ptype = part.get("type")
        if ptype == "text":
            parts.append({"type": "input_text", "text": part.get("text", "")})
        elif ptype == "image_url":
            iu = part.get("image_url")
            url = iu.get("url") if isinstance(iu, dict) else iu
            new_part: dict[str, Any] = {"type": "input_image", "image_url": url or ""}
            if isinstance(iu, dict) and iu.get("detail"):
                new_part["detail"] = iu["detail"]
            parts.append(new_part)
        else:
            # Already-valid Responses part type (input_text/input_image/...) or
            # unknown: pass through unchanged.
            parts.append(part)
    return parts


def _translate_chat_to_responses(body_json: dict) -> dict:
    """Translate a chat/completions request body to the Responses API format.

    Mapping (best-effort):

      - ``messages`` -> ``input``; system/user/assistant roles pass through;
        ``tool`` role becomes ``{type: function_call_output}``; assistant
        ``tool_calls`` become ``{type: function_call}`` items.
      - ``max_tokens`` / ``max_completion_tokens`` -> ``max_output_tokens``.
      - ``reasoning_effort`` -> ``reasoning.effort`` (Responses shape).
      - Streaming/tools/temperature/top_p/stop pass through unchanged.

    Unsupported chat-only keys (store, stream_options, n, ...) are dropped;
    the upstream sanitizer already strips most of them before this runs.
    """
    if not isinstance(body_json, dict):
        return body_json

    out: dict[str, Any] = {}
    if body_json.get("model"):
        out["model"] = body_json["model"]

    messages = body_json.get("messages")
    if isinstance(messages, list):
        input_items: list[Any] = []
        for msg in messages:
            if not isinstance(msg, dict):
                input_items.append(msg)
                continue
            role = msg.get("role")
            if role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id") or "",
                    "output": msg.get("content") if isinstance(msg.get("content"), str) else json.dumps(msg.get("content")),
                })
            elif role == "assistant" and isinstance(msg.get("tool_calls"), list) and msg.get("tool_calls"):
                # Assistant tool-call turn: keep content (if any) then emit
                # one function_call item per tool call.
                content = msg.get("content")
                if content:
                    input_items.append({"role": "assistant", "content": _normalize_responses_content(content)})
                for tc in msg["tool_calls"]:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    input_items.append({
                        "type": "function_call",
                        "call_id": tc.get("id") or "",
                        "name": fn.get("name") or "",
                        "arguments": fn.get("arguments") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments") or {}),
                    })
            else:
                norm = dict(msg)
                if "content" in norm:
                    norm["content"] = _normalize_responses_content(norm["content"])
                input_items.append(norm)
        out["input"] = input_items

    if body_json.get("max_tokens") is not None:
        out["max_output_tokens"] = body_json["max_tokens"]
    elif body_json.get("max_completion_tokens") is not None:
        out["max_output_tokens"] = body_json["max_completion_tokens"]

    for key in ("stream", "temperature", "top_p", "stop"):
        if body_json.get(key) is not None:
            out[key] = body_json[key]

    # Tools: chat/completions nests function name/description/parameters under
    # ``function``; the Responses API requires ``name`` at the tool top level
    # (``tools[0] missing required field name`` otherwise).
    tools = body_json.get("tools")
    if isinstance(tools, list):
        normalized_tools: list[Any] = []
        for tool in tools:
            if not isinstance(tool, dict):
                normalized_tools.append(tool)
                continue
            fn = tool.get("function")
            if tool.get("type") == "function" and isinstance(fn, dict):
                norm_tool: dict[str, Any] = {"type": "function"}
                for k in ("name", "description", "parameters", "strict"):
                    if fn.get(k) is not None:
                        norm_tool[k] = fn[k]
                normalized_tools.append(norm_tool)
            else:
                normalized_tools.append(tool)
        out["tools"] = normalized_tools
    elif tools is not None:
        out["tools"] = tools

    if body_json.get("tool_choice") is not None:
        out["tool_choice"] = body_json["tool_choice"]

    if body_json.get("reasoning_effort"):
        out["reasoning"] = {"effort": body_json["reasoning_effort"]}

    return out


def _translate_responses_to_chat(resp_json: dict) -> dict:
    """Translate a non-streaming Responses API JSON body to chat/completions shape.

    Mapping:
      - ``output[].message.content[].text`` -> ``choices[0].message.content``
        (concatenated over the message body).
      - ``output[].function_call`` -> ``choices[0].message.tool_calls``.
      - ``usage.{input,output,total}_tokens`` -> ``usage.{prompt,completion,total}_tokens``.
      - ``status`` -> ``finish_reason`` (completed->stop, incomplete->length,
        failed->error; tool-call output -> tool_calls).

    Reasoning items (which carry ``encrypted_content`` on zen/go) are dropped;
    they are not usable as assistant reasoning_content anyway.
    """
    if not isinstance(resp_json, dict):
        return resp_json

    output = resp_json.get("output") or []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            content = item.get("content") or []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    t = part.get("text")
                    if isinstance(t, str):
                        text_parts.append(t)
        elif itype == "function_call":
            tool_calls.append({
                "id": item.get("call_id") or "",
                "type": "function",
                "function": {
                    "name": item.get("name") or "",
                    "arguments": item.get("arguments") if isinstance(item.get("arguments"), str) else json.dumps(item.get("arguments") or {}),
                },
            })

    status = resp_json.get("status")
    if tool_calls:
        finish_reason = "tool_calls"
    elif status == "incomplete":
        finish_reason = "length"
    elif status == "failed":
        finish_reason = "error"
    else:
        finish_reason = "stop"

    message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = resp_json.get("usage") or {}
    return {
        "id": resp_json.get("id") or "resp-unknown",
        "object": "chat.completion",
        "created": resp_json.get("created_at"),
        "model": resp_json.get("model"),
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


def _sse_chat_completion_chunk(delta: dict, finish_reason: str | None = None) -> bytes:
    """Serialize one chat/completions SSE ``data:`` chunk as bytes."""
    choice: dict[str, Any] = {"index": 0, "delta": delta}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return f"data: {json.dumps({'choices': [choice]})}\n\n".encode()


async def _translate_responses_stream(aiter) -> Any:
    """Wrap a Responses API SSE byte iterator, yielding chat/completions SSE chunks.

    Consumes ``response.output_text.delta`` (-> content delta), function-call
    item add / ``function_call_arguments.delta`` (-> tool_calls deltas), and
    ``response.completed`` (-> finish chunk + ``[DONE]``). All other events
    (created, in_progress, reasoning items, ping, ...) are skipped. The
    ``response.failed`` event yields a ``finish_reason: error`` chunk so the
    proxy fallback chain can route to the next provider.
    """
    buffer = b""
    # Track emitted tool-call index by responses output_index so argument
    # deltas accumulate onto the right tool_call delta chunk.
    _tool_index_by_output: dict[int, int] = {}
    _next_tool_index = 0
    _stream_saw_tool_calls = False
    _stream_status = "in_progress"

    async def _feed(chunk: bytes):
        nonlocal buffer
        buffer += chunk

    async def _process_events():
        nonlocal buffer, _next_tool_index, _stream_saw_tool_calls, _stream_status
        while b"\n\n" in buffer:
            event_block, buffer = buffer.split(b"\n\n", 1)
            event_block = event_block.strip()
            if not event_block:
                continue
            event_type = None
            data_str = None
            for line in event_block.split(b"\n"):
                line = line.strip()
                if line.startswith(b"event:"):
                    event_type = line[6:].strip().decode("utf-8", errors="replace")
                elif line.startswith(b"data:"):
                    data_str = line[5:].strip()
            if event_type is None or data_str is None:
                continue
            try:
                data = json.loads(data_str.decode("utf-8", errors="replace"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue

            if event_type == "response.output_text.delta":
                delta = data.get("delta")
                if isinstance(delta, str) and delta:
                    yield _sse_chat_completion_chunk({"content": delta})
            elif event_type == "response.output_item.added":
                item = data.get("item") or {}
                if isinstance(item, dict) and item.get("type") == "function_call":
                    output_index = data.get("output_index")
                    _tool_index_by_output[output_index] = _next_tool_index
                    _next_tool_index += 1
                    _stream_saw_tool_calls = True
                    fn = {"name": item.get("name") or "", "arguments": ""}
                    yield _sse_chat_completion_chunk({
                        "tool_calls": [{
                            "index": _tool_index_by_output[output_index],
                            "id": item.get("id") or item.get("call_id") or "",
                            "type": "function",
                            "function": fn,
                        }]
                    })
            elif event_type == "response.function_call_arguments.delta":
                output_index = data.get("output_index")
                tool_idx = _tool_index_by_output.get(output_index)
                if tool_idx is not None:
                    delta_args = data.get("delta")
                    if isinstance(delta_args, str) and delta_args:
                        yield _sse_chat_completion_chunk({
                            "tool_calls": [{"index": tool_idx, "function": {"arguments": delta_args}}]
                        })
            elif event_type == "response.completed":
                resp_obj = data.get("response") or {}
                _stream_status = resp_obj.get("status") or "completed"
                if _stream_saw_tool_calls:
                    yield _sse_chat_completion_chunk({}, finish_reason="tool_calls")
                elif _stream_status == "incomplete":
                    yield _sse_chat_completion_chunk({}, finish_reason="length")
                else:
                    yield _sse_chat_completion_chunk({}, finish_reason="stop")
                yield b"data: [DONE]\n\n"
            elif event_type == "response.failed":
                yield _sse_chat_completion_chunk({}, finish_reason="error")
                yield b"data: [DONE]\n\n"
            # All other events -> skipped

    async for chunk in aiter:
        await _feed(chunk)
        async for out in _process_events():
            yield out
    # Stream ended without a terminal event: flush any trailing partial block
    # (unlikely for a well-formed stream) and stop.
    return


async def proxy_to_remote(
    request: Request,
    path: str,
    model_config: dict,
) -> Response:
    """Proxy request to remote API endpoint."""
    endpoint = model_config.get("endpoint", "")

    # Responses-API mode (LP-0MTGK5DQO001Y8H0): providers configured with
    # ``api: openai-responses`` expose only /v1/responses. Route to the
    # responses path and translate the body/response between the two wire
    # formats.
    responses_mode = model_config.get("api") == "openai-responses"
    if responses_mode and str(path).endswith("chat/completions"):
        path = str(path)[: -len("chat/completions")] + "responses"

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

    # Determine whether to forward session affinity headers (LP-0MRE8GD1H0028CGN).
    # Default: forward (True) to maintain session locality for upstream providers
    # that support it. Set forward_session_headers: false on a provider config
    # to opt out for incompatible upstreams.
    _forward_session_headers = model_config.get("forward_session_headers", True)
    if _forward_session_headers is None:
        _forward_session_headers = True

    # Remove local/proxy auth/session headers before forwarding.
    # In particular, prevent duplicate Authorization variants
    # (e.g. "authorization" + "Authorization") which can trigger
    # Cloudflare 400 Bad Request on upstream.
    # Session affinity headers are stripped only when forward_session_headers
    # is False (or absent with backward-compatible default).
    _session_header_keys = {"x-session-id", "x-client-request-id", "x-session-affinity", "session_id"}
    for hk in list(headers.keys()):
        hkl = str(hk).lower()
        if hkl == "authorization":
            headers.pop(hk, None)
        elif not _forward_session_headers and hkl in _session_header_keys:
            headers.pop(hk, None)

    # Add API key
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Add custom headers from config
    custom_headers = model_config.get("headers", {})
    headers.update(custom_headers)

    # Add provider attribution headers from config (LP-0MRE8GHNG003R8YX)
    # Allows operators to forward provider-specific headers (e.g.,
    # HTTP-Referer, X-OpenRouter-Title for OpenRouter billing).
    attribution_headers = model_config.get("attribution_headers", {})
    if attribution_headers and isinstance(attribution_headers, dict):
        headers.update(attribution_headers)

    body_json = json.loads(body) if body else {}
    if not isinstance(body_json, dict):
        body_json = {}

    if responses_mode:
        # Keep the chat-format tool-call repair (it runs on the inbound chat
        # messages), then translate the whole body to the Responses API shape.
        if isinstance(body_json.get("messages"), list):
            body_json["messages"] = _sanitize_remote_messages(body_json["messages"])
        body_json = _translate_chat_to_responses(body_json)
        # The upstream must receive the translated (responses-shaped) body.
        body = json.dumps(body_json).encode("utf-8")
    else:
        # Sanitize request-shape for remote compatibility before model override.
        body_json = _sanitize_remote_chat_payload(path, body_json)

        # Sanitize accumulated tool-call/tool-result message history before remote
        # sends (LP-0MSC1BNP90017L9K): remote providers reject malformed tool-call
        # sequences with HTTP 400 (missing/dangling tool_call_id, missing id/type,
        # empty tool_calls). Always-on; repairs where unambiguous, prunes otherwise.
        if isinstance(body_json.get("messages"), list):
            body_json["messages"] = _sanitize_remote_messages(body_json["messages"])

    # Override model name in body if provider config specifies an upstream model ID.
    # This allows the proxy to present a different model name to the remote API
    # than what the client originally sent (e.g. "deepseek-v4-flash" for a
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

    # Config entry name for stream-level log attribution (LP-0MSC7F7BG0043TE1).
    # Distinct from model_name (the body model ID): multiple config entries can
    # share the same provider+model, so entry=<name> lets per-account traffic
    # be distinguished in logs. None when the config entry has no name.
    entry_name = model_config.get("name")

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

    # Apply upstream request-level timeout cap (LP-0MRF77A0E0026B9T).
    # If configured, cap the read timeout to prevent 15+ minute silent hangs
    # when the upstream is slow to respond. This is different from the
    # per-chunk idle timeout (upstream_idle_timeout_seconds) which detects
    # mid-stream stalls.
    _upstream_request_timeout = float(
        server_config.get("upstream_request_timeout_seconds", 120) or 120
    )
    if isinstance(remote_timeout, httpx.Timeout):
        _capped_read = remote_timeout.read
        if _capped_read is not None and _capped_read > _upstream_request_timeout:
            remote_timeout = httpx.Timeout(
                connect=remote_timeout.connect,
                read=_upstream_request_timeout,
                write=remote_timeout.write,
                pool=remote_timeout.pool,
            )

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
    # Default raised 60 -> 120 -> 240 to tolerate long reasoning pauses on
    # remote upstreams (LP-0MS9FR9LG002AJ4C; LP-0MSF5I7XN009ENWQ raises the
    # default to 240s for LP-0MSF1PUM90099ZSW F4). Keep in sync with the
    # fallback in _handle_remote_streaming and proxy/config.yaml.
    _upstream_idle_timeout = float(
        server_config.get("upstream_idle_timeout_seconds", 240) or 240
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
                entry=entry_name,
                upstream_idle_timeout_seconds=_upstream_idle_timeout,
                upstream_retry_connect_timeout_seconds=_upstream_retry_connect_timeout,
                pool_client=_pool_client,
                responses_mode=responses_mode,
            )
        return await _handle_remote_streaming(
            request, target_url, headers, body, body_json,
            model_name, remote_timeout,
            resolved_model=_resolved_model_header,
            provider=_provider_name,
            entry=entry_name,
            upstream_idle_timeout_seconds=_upstream_idle_timeout,
            upstream_retry_connect_timeout_seconds=_upstream_retry_connect_timeout,
            pool_client=_pool_client,
            responses_mode=responses_mode,
        )
    else:
        if _remote_session_id:
            return await _handle_remote_non_streaming(
                request, target_url, headers, body, model_name, remote_timeout,
                resolved_model=_resolved_model_header,
                session_id=_remote_session_id,
                pool_client=_pool_client,
                responses_mode=responses_mode,
            )
        return await _handle_remote_non_streaming(
            request, target_url, headers, body, model_name, remote_timeout,
            resolved_model=_resolved_model_header,
            pool_client=_pool_client,
            responses_mode=responses_mode,
        )


def _build_stream_error_event(
    provider: str | None = None,
    model: str | None = None,
    entry: str | None = None,
    error_type: str = "stream_error",
    message: str = "Upstream stream error",
    suggested_action: str | None = None,
    session_id: str | None = None,
) -> dict:
    """Build an enriched synthetic ``finish_reason: error`` SSE event.

    Replaces the previously bare ``{"delta": {}, "finish_reason": "error"}``
    payload (LP-0MSETOTWY000SU0Z / proxy/docs/error-analysis-2026-08-03.md
    Recommendation 2). The event keeps ``finish_reason: error`` and an empty
    ``delta`` (backward compatible with existing clients) and adds a
    structured ``error`` object (type/message/provider/model/entry/
    suggested_action/session_id) so the operator/agent can act instead of
    seeing an unspecified error.

    Args:
        provider: The provider brand (e.g. ``opencode-go``) that failed.
        model: The model id (e.g. ``deepseek-v4-flash``).
        entry: The config entry name (e.g. ``opencode-go-2-deepseek``).
        error_type: Failure class (stall_exhausted, empty_response,
            stream_exception, stall_after_content, ...).
        message: Human-readable one-liner with the underlying cause.
        suggested_action: Static remediation guidance for this failure type.
        session_id: The proxy session id, when available.

    Returns:
        The SSE event dict (one ``choices`` entry with finish_reason: error
        and an ``error`` payload).
    """
    error_payload: dict[str, Any] = {
        "type": error_type,
        "message": message,
        "provider": provider or "unknown",
        "model": model or "unknown",
    }
    if entry:
        error_payload["entry"] = entry
    if suggested_action:
        error_payload["suggested_action"] = suggested_action
    if session_id:
        error_payload["session_id"] = session_id
    return {
        "choices": [
            {
                "delta": {},
                "finish_reason": "error",
                "index": 0,
                "error": error_payload,
            }
        ]
    }


def _stream_error_event_bytes(
    provider: str | None = None,
    model: str | None = None,
    entry: str | None = None,
    error_type: str = "stream_error",
    message: str = "Upstream stream error",
    suggested_action: str | None = None,
    session_id: str | None = None,
) -> bytes:
    """Serialize :func:`_build_stream_error_event` to an SSE ``data:`` chunk."""
    return f"data: {json.dumps(_build_stream_error_event(provider=provider, model=model, entry=entry, error_type=error_type, message=message, suggested_action=suggested_action, session_id=session_id))}\n\n".encode()


def _delta_has_content(delta: dict) -> bool:
    """True if a stream delta carries meaningful output.

    Counts non-empty ``content``, a non-empty ``tool_calls`` list, and
    non-empty ``reasoning_content`` as content so that tool-call-only and
    reasoning-only streams are not misclassified as empty responses
    (LP-0MS8XAPXT009W3CL).
    """
    if not isinstance(delta, dict):
        return False
    c = delta.get("content")
    if isinstance(c, str) and c.strip():
        return True
    tc = delta.get("tool_calls")
    if isinstance(tc, list) and tc:
        return True
    rc = delta.get("reasoning_content")
    if isinstance(rc, str) and rc.strip():
        return True
    return False


async def _handle_remote_streaming(
    request: Request,
    target_url: str,
    headers: dict,
    body: bytes,
    body_json: dict,
    model_name: str,
    remote_timeout: httpx.Timeout,
    resolved_model: str | None = None,
    session_id: str | None = None,
    provider: str | None = None,
    entry: str | None = None,
    upstream_idle_timeout_seconds: float | None = None,
    upstream_retry_connect_timeout_seconds: float | None = None,
    upstream_max_stream_duration_seconds: float | None = None,
    upstream_activity_timeout_seconds: float | None = None,
    pool_client: httpx.AsyncClient | None = None,
    responses_mode: bool = False,
) -> Response:
    """Handle streaming remote proxy request with upstream stall detection and retry.

    When ``responses_mode`` is True (``api: openai-responses`` provider), the
    upstream emits Responses API SSE events; the byte iterator is wrapped with
    :func:`_translate_responses_stream` so the events surface as chat/completions
    SSE (content/tool_calls deltas, finish chunk, [DONE]) and every fallback/
    watchdog/retry mechanism below keeps working unchanged (LP-0MTGK5DQO001Y8H0).


    Features:
    - Per-chunk idle timeout: detects upstream silence within
      *upstream_idle_timeout_seconds* and closes the stalled connection.
    - Watchdog budgets (LP-0MSVP7ZML003XZTJ): *upstream_max_stream_duration_seconds*
      caps total remote stream lifetime; *upstream_activity_timeout_seconds* caps
      time since the last CONTENT-bearing chunk. Either expiry terminates the
      stream with a synthetic ``finish_reason: error`` (error.type
      ``stream_max_duration`` / ``stream_activity_timeout``) and NO retry — a
      "connected but idle" upstream (heartbeats flowing, no content) defeats the
      per-chunk idle timeout and previously held proxy state indefinitely.
    - Automatic retry: on stall detection (asyncio.TimeoutError) or httpx
      ReadTimeout, retries the same provider with bounded exponential backoff
      (1s, 2s, 4s; max 3 retries).
    - Content-aware retry (LP-0MS9FR9LG002AJ4C): Tier-1 retries only occur
      while zero content-bearing chunks have been delivered. Once any content
      has been sent to the client, a stall terminates the stream immediately
      with a synthetic ``finish_reason: error`` (no whole-request retry) so
      the client can retry with full context.
    - Fallthrough: after max retries exhausted, yields a synthetic
      ``finish_reason: error`` event so the caller (provider.py fallback chain)
      can route to the next provider.
    """
    # Resolve upstream_idle_timeout_seconds from parameter or config
    if upstream_idle_timeout_seconds is None:
        try:
            upstream_idle_timeout_seconds = float(
                _srv().config.get("server", {}).get(
                    "upstream_idle_timeout_seconds", 240
                ) or 240
            )
        except Exception:
            upstream_idle_timeout_seconds = 240.0

    # Resolve remote-stream watchdog budgets (LP-0MSVP7ZML003XZTJ).
    # A "connected but idle" upstream that never goes SILENT (heartbeats /
    # keep-alives flowing, empty deltas) defeats the per-chunk idle timeout
    # above and can hold proxy state (local_active_query, slots) for hours.
    # These bounded deadlines terminate such streams with a synthetic
    # finish_reason: error and NO retry (restarting a stuck stream re-sticks
    # it), and increment a metric so runaway streams surface as alerts.
    #
    # - upstream_max_stream_duration_seconds (default 14400 = 4h): hard cap
    #   on total remote stream lifetime.
    # - upstream_activity_timeout_seconds (default 1800 = 30 min): max time
    #   since the last CONTENT-bearing chunk; heartbeats/empty deltas do not
    #   count as progress.
    if upstream_max_stream_duration_seconds is None:
        try:
            upstream_max_stream_duration_seconds = float(
                _srv().config.get("server", {}).get(
                    "upstream_max_stream_duration_seconds", 14400
                ) or 14400
            )
        except Exception:
            upstream_max_stream_duration_seconds = 14400.0
    if upstream_activity_timeout_seconds is None:
        try:
            upstream_activity_timeout_seconds = float(
                _srv().config.get("server", {}).get(
                    "upstream_activity_timeout_seconds", 1800
                ) or 1800
            )
        except Exception:
            upstream_activity_timeout_seconds = 1800.0

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

    # Read retry config from server settings (LP-0MRE8G94H005ZBLV)
    try:
        max_retries = int(
            _srv().config.get("server", {}).get(
                "upstream_retry_max_attempts", 3
            ) or 3
        )
    except Exception:
        max_retries = 3
    try:
        retry_base_delay = float(
            _srv().config.get("server", {}).get(
                "upstream_retry_base_delay_seconds", 2.0
            ) or 2.0
        )
    except Exception:
        retry_base_delay = 2.0
    try:
        retry_max_delay = float(
            _srv().config.get("server", {}).get(
                "upstream_retry_max_delay_seconds", 60.0
            ) or 60.0
        )
    except Exception:
        retry_max_delay = 60.0

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

    # Read empty-response retry config (LP-0MRF77A0E0026B9T)
    try:
        empty_max_attempts = int(
            _srv().config.get("server", {}).get("upstream_empty_retry_max_attempts", 1) or 1
        )
    except Exception:
        empty_max_attempts = 1
    try:
        empty_base_delay = float(
            _srv().config.get("server", {}).get("upstream_empty_retry_base_delay_seconds", 0.5) or 0.5
        )
    except Exception:
        empty_base_delay = 0.5

    media_type = response.headers.get("content-type", "text/event-stream")
    key = f"{request.method.upper()} {request.url.path} -> remote"

    async def stream_generator():
        saw_done = False
        saw_finish = False
        # Content tracking for empty-response detection (LP-0MRF77A0E0026B9T)
        # _has_content is set when a chunk carries meaningful output: non-empty
        # content, tool_calls, or reasoning_content (LP-0MS8XAPXT009W3CL).
        _has_content = False
        # Diagnostic flags for empty-retry logging (LP-0MS8XAPXT009W3CL)
        _saw_tool_calls = False
        _saw_reasoning = False
        # Client disconnect detection (LP-0MQTHP828000JYM6)
        disconnected = False
        _disconnect_check_count = 0
        # Collect chunks for session recording (LP-0MR94O16S000WFQ0)
        collected_chunks = [] if session_id else None

        # Log stream started with session context (LP-0MR90HJED005WI1Z)
        try:
            _request_preview = _get_request_preview(body_json)
            _srv().logger.info(
                "Stream started: provider=%s model=%s session=%s request=%s%s",
                provider or "remote",
                model_name,
                session_id or "unknown",
                _request_preview or "",
                f" entry={entry}" if entry else "",
            )
        except Exception:
            pass

        # Per-chunk idle timeout and retry state (LP-0MRE52D3C001KP1H)
        _retry_count = 0
        # Remote-stream watchdog state (LP-0MSVP7ZML003XZTJ): bounded
        # deadlines that terminate a "connected but idle" stream which the
        # per-chunk idle timeout cannot catch (heartbeats flowing, no
        # content progress). Updated on every content-bearing chunk.
        _watchdog_started = time.monotonic()
        _last_content_at = _watchdog_started
        _watchdog_terminated_reason: str | None = None
        # Empty-response retry state (LP-0MRF77A0E0026B9T)
        _empty_retry_count = 0
        _should_empty_retry = False
        _current_client = client
        _current_cm = cm
        _current_response = response
        _should_retry = False
        # After-content termination flag (LP-0MS9FR9LG002AJ4C): set when a
        # stall/ReadTimeout occurs after content-bearing chunks were already
        # delivered; the stream then terminates immediately with a synthetic
        # finish_reason: error instead of restarting the whole request.
        _terminate_after_content = False

        # Outer loop: retry on stall/ReadTimeout (initial attempt counts as
        # iteration 0; retries are iterations 1..max_retries) or
        # empty-response retry (LP-0MRF77A0E0026B9T).
        while True:
            if _should_empty_retry:
                _should_empty_retry = False
                _empty_retry_count += 1
                try:
                    _srv().logger.info(
                        "Empty response detected on stream attempt %s/%s, "
                        "retrying in %.2fs (provider=%s model=%s "
                        "saw_tool_calls=%s saw_reasoning=%s)",
                        _empty_retry_count,
                        empty_max_attempts + 1,
                        empty_base_delay,
                        provider or "remote",
                        model_name,
                        _saw_tool_calls,
                        _saw_reasoning,
                    )
                except Exception:
                    pass
                await asyncio.sleep(empty_base_delay)
                # Create fresh stream connection for empty retry
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
                    _empty_upstream_status = _current_response.status_code
                    _empty_upstream_ct = _current_response.headers.get("content-type", "")

                    if _empty_upstream_status >= 400 or "text/event-stream" not in _empty_upstream_ct.lower():
                        # Retry returned a non-streaming response — don't retry further
                        try:
                            await _current_cm.__aexit__(None, None, None)
                        except Exception:
                            pass
                        # Yield error and exit
                        _final_error_obj = _build_stream_error_event(
                            provider=provider,
                            model=model_name,
                            entry=entry,
                            error_type="empty_response",
                            message="Retry returned a non-streaming/HTTP error after an empty upstream response",
                            suggested_action="Upstream returned no content; check upstream status or route manually",
                            session_id=session_id,
                        )
                        _final_error_bytes = (
                            f"data: {json.dumps(_final_error_obj)}\n\n"
                        ).encode()
                        if collected_chunks is not None:
                            collected_chunks.append(_final_error_bytes)
                        yield _final_error_bytes
                        log_response_chunk(_final_error_bytes, session_id=session_id, model=model_name, provider=provider, body_json=body_json, entry=entry)
                        break

                    # Reset stream state for the new connection
                    _has_content = False
                    saw_done = False
                    saw_finish = False
                    continue
                except Exception:
                    # Connection failed on empty retry — continue to next retry
                    # or fall through to exhaustion
                    if _empty_retry_count > empty_max_attempts:
                        pass  # Will be caught below
                    else:
                        continue

            # Empty-response retry exhaustion (LP-0MRF77A0E0026B9T)
            # Check if we've exhausted empty retries and need to yield an error.
            # This can happen when:
            #   - All retries returned empty responses (no content chunks)
            #   - Empty retry connection failed (exception case above)
            if _empty_retry_count > empty_max_attempts:
                try:
                    _srv().logger.warning(
                        "Empty upstream response: max retries exhausted "
                        "session=%s provider=%s model=%s retries=%d "
                        "saw_tool_calls=%s saw_reasoning=%s",
                        session_id or "unknown",
                        provider or "remote",
                        model_name,
                        _empty_retry_count,
                        _saw_tool_calls,
                        _saw_reasoning,
                    )
                except Exception:
                    pass
                _final_error_obj = _build_stream_error_event(
                    provider=provider,
                    model=model_name,
                    entry=entry,
                    error_type="empty_response",
                    message=f"Empty upstream response after {_empty_retry_count} retries",
                    suggested_action="Upstream returned no content; check upstream status or route manually",
                    session_id=session_id,
                )
                _final_error_bytes = (
                    f"data: {json.dumps(_final_error_obj)}\n\n"
                ).encode()
                if collected_chunks is not None:
                    collected_chunks.append(_final_error_bytes)
                yield _final_error_bytes
                log_response_chunk(_final_error_bytes, session_id=session_id, model=model_name, provider=provider, body_json=body_json, entry=entry)
                break

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

                # Wire into Tier 3 cross-request stall circuit breaker
                # (LP-0MRFEXXVC001RYKB). Record the stall so repeated
                # failures across requests accumulate and trigger provider
                # cooldown.
                try:
                    _config = _srv().config if hasattr(_srv(), 'config') else {}
                    _check_stall_circuit_breaker(
                        provider or "remote",
                        _config,
                    )
                except Exception:
                    pass

                _final_error_obj = _build_stream_error_event(
                    provider=provider,
                    model=model_name,
                    entry=entry,
                    error_type="stall_exhausted",
                    message=f"Upstream stalled repeatedly ({_retry_count} retries exhausted; idle timeout {upstream_idle_timeout_seconds:.0f}s)",
                    suggested_action="Provider placed in cooldown; the next provider in the chain will be used",
                    session_id=session_id,
                )
                _final_error_bytes = (
                    f"data: {json.dumps(_final_error_obj)}\n\n"
                ).encode()
                if collected_chunks is not None:
                    collected_chunks.append(_final_error_bytes)
                yield _final_error_bytes
                log_response_chunk(_final_error_bytes, session_id=session_id, model=model_name, provider=provider, body_json=body_json, entry=entry)
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
                        if _owns_client:
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
                _raw_aiter = _current_response.aiter_bytes().__aiter__()
                _aiter = _translate_responses_stream(_raw_aiter) if responses_mode else _raw_aiter
                while True:
                    # Watchdog-bounded read (LP-0MSVP7ZML003XZTJ): the
                    # effective per-read budget is the minimum of the idle
                    # timeout and the remaining max-duration / activity
                    # budgets, so a deadline that elapses classifies as its
                    # own reason instead of the ordinary idle-stall retry.
                    _now = time.monotonic()
                    _duration_remaining = (
                        _watchdog_started + upstream_max_stream_duration_seconds
                    ) - _now
                    _activity_remaining = (
                        _last_content_at + upstream_activity_timeout_seconds
                    ) - _now
                    _read_budget = min(
                        upstream_idle_timeout_seconds,
                        _duration_remaining,
                        _activity_remaining,
                    )
                    try:
                        chunk = await asyncio.wait_for(
                            _aiter.__anext__(),
                            timeout=max(0.0, _read_budget),
                        )
                    except TimeoutError:
                        _now = time.monotonic()
                        if _now >= _watchdog_started + upstream_max_stream_duration_seconds:
                            # Hard total-lifetime cap exceeded → terminate, no retry.
                            _watchdog_terminated_reason = "stream_max_duration"
                            break
                        if _now >= _last_content_at + upstream_activity_timeout_seconds:
                            # Connected-but-idle: no content progress within the
                            # activity budget → terminate, no retry.
                            _watchdog_terminated_reason = "stream_activity_timeout"
                            break
                        # Ordinary per-chunk idle stall (true silence) within
                        # the watchdog budgets → existing retry path.
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
                                    if _delta_has_content(delta):
                                        _has_content = True
                                        # Content-bearing chunk = activity
                                        # progress (LP-0MSVP7ZML003XZTJ):
                                        # resets the activity watchdog.
                                        _last_content_at = time.monotonic()
                                    # Token counting for text content remains
                                    # unchanged (content only).
                                    if isinstance(delta, dict) and "content" in delta:
                                        d_content = delta.get("content")
                                        if d_content is not None and d_content != "":
                                            texts.append(str(d_content))
                                    # Track tool_calls/reasoning separately for
                                    # empty-retry diagnostics (LP-0MS8XAPXT009W3CL).
                                    if isinstance(delta, dict) and delta.get("tool_calls"):
                                        _saw_tool_calls = True
                                        _last_content_at = time.monotonic()
                                    if isinstance(delta, dict):
                                        _rc = delta.get("reasoning_content")
                                        if isinstance(_rc, str) and _rc.strip():
                                            _saw_reasoning = True
                                            _last_content_at = time.monotonic()
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
                    log_response_chunk(chunk, session_id=session_id, model=model_name, provider=provider, body_json=body_json, entry=entry)

                    if saw_done or saw_finish:
                        break

                if disconnected:
                    # Client disconnected — stop streaming entirely, no retry
                    break

                if saw_done or saw_finish:
                    # Stream completed normally. Check for empty response
                    # (no content chunks received). Retry if configurable
                    # empty-retry attempts remain (LP-0MRF77A0E0026B9T).
                    if not _has_content and _empty_retry_count < empty_max_attempts:
                        # Empty response detected — retry with empty-retry backoff
                        _should_empty_retry = True
                        continue
                    # If no content and retries exhausted, yield synthetic error
                    # so the caller (provider.py fallback chain) can route to
                    # the next provider.
                    if not _has_content:
                        _final_empty_error_obj = _build_stream_error_event(
                            provider=provider,
                            model=model_name,
                            entry=entry,
                            error_type="empty_response",
                            message="Upstream stream completed with no content",
                            suggested_action="Upstream returned no content; check upstream status or route manually",
                            session_id=session_id,
                        )
                        _final_empty_error_bytes = (
                            f"data: {json.dumps(_final_empty_error_obj)}\n\n"
                        ).encode()
                        if collected_chunks is not None:
                            collected_chunks.append(_final_empty_error_bytes)
                        yield _final_empty_error_bytes
                        log_response_chunk(_final_empty_error_bytes, session_id=session_id, model=model_name, provider=provider, body_json=body_json, entry=entry)
                    # Has content (with or without retries) or retries exhausted — stop outer loop
                    break

                # If we break out of the inner loop without saw_done/saw_finish
                # and without disconnect, it's a stall (asyncio.TimeoutError)
                # or a watchdog termination (LP-0MSVP7ZML003XZTJ).
                # Content-aware retry (LP-0MS9FR9LG002AJ4C): only retry while
                # zero content-bearing chunks were delivered. Once any content
                # has been sent to the client, restarting the whole request
                # would re-send a huge prompt and duplicate output, so the
                # stream terminates immediately instead.
                if _watchdog_terminated_reason:
                    # Watchdog fired (max-duration or activity timeout): the
                    # stream is stuck — restarting it would just re-stick it.
                    # Terminate with a synthetic error + alert metric.
                    try:
                        from proxy.metrics import record_remote_stream_terminated

                        record_remote_stream_terminated(_watchdog_terminated_reason)
                    except Exception:
                        pass
                    _watchdog_message = (
                        f"Upstream stream exceeded max duration "
                        f"({upstream_max_stream_duration_seconds:.0f}s)"
                        if _watchdog_terminated_reason == "stream_max_duration"
                        else f"Upstream stream made no content progress for "
                        f"{upstream_activity_timeout_seconds:.0f}s "
                        f"(connected but idle)"
                    )
                    try:
                        _srv().logger.warning(
                            "Remote stream %s session=%s provider=%s model=%s "
                            "duration=%.1fs",
                            _watchdog_terminated_reason,
                            session_id or "unknown",
                            provider or "remote",
                            model_name,
                            time.monotonic() - _watchdog_started,
                        )
                    except Exception:
                        pass
                    _final_error_obj = _build_stream_error_event(
                        provider=provider,
                        model=model_name,
                        entry=entry,
                        error_type=_watchdog_terminated_reason,
                        message=_watchdog_message,
                        suggested_action=(
                            "Upstream is stuck (connected but idle). Retry the "
                            "request, or check upstream health."
                        ),
                        session_id=session_id,
                    )
                    _final_error_bytes = (
                        f"data: {json.dumps(_final_error_obj)}\n\n"
                    ).encode()
                    if collected_chunks is not None:
                        collected_chunks.append(_final_error_bytes)
                    yield _final_error_bytes
                    log_response_chunk(_final_error_bytes, session_id=session_id, model=model_name, provider=provider, body_json=body_json, entry=entry)
                    break
                if _has_content:
                    _terminate_after_content = True
                else:
                    # Zero content delivered — safe to retry the whole request
                    # with bounded exponential backoff.
                    _should_retry = True

            except StopAsyncIteration:
                # Normal exhaustion of the upstream iterator (no [DONE] received).
                # Synthesize final stop event as in the original code.
                if not saw_done and not saw_finish:
                    # Check for empty response (no content received)
                    # and retry if configurable retries remain (LP-0MRF77A0E0026B9T).
                    if not _has_content and _empty_retry_count < empty_max_attempts:
                        _should_empty_retry = True
                        continue

                    # If no content (and retries exhausted/exhausted above), yield
                    # synthetic error so the fallback chain can activate.
                    if not _has_content:
                        _final_empty_error_obj = _build_stream_error_event(
                            provider=provider,
                            model=model_name,
                            entry=entry,
                            error_type="empty_response",
                            message="Upstream closed without delivering content",
                            suggested_action="Upstream returned no content; check upstream status or route manually",
                            session_id=session_id,
                        )
                        _final_empty_error_bytes = (
                            f"data: {json.dumps(_final_empty_error_obj)}\n\n"
                        ).encode()
                        if collected_chunks is not None:
                            collected_chunks.append(_final_empty_error_bytes)
                        yield _final_empty_error_bytes
                        log_response_chunk(_final_empty_error_bytes, session_id=session_id, model=model_name, provider=provider, body_json=body_json, entry=entry)
                    else:
                        # Stream had content but ended without [DONE] — yield stop
                        _final_stop_obj = {
                            "choices": [
                                {"delta": {}, "finish_reason": "stop", "index": 0}
                            ]
                        }
                        _final_stop_bytes = (
                            f"data: {json.dumps(_final_stop_obj)}\n\n"
                        ).encode()
                        if collected_chunks is not None:
                            collected_chunks.append(_final_stop_bytes)
                        yield _final_stop_bytes
                        log_response_chunk(_final_stop_bytes, session_id=session_id, model=model_name, provider=provider, body_json=body_json, entry=entry)
                break
            except httpx.ReadTimeout:
                # httpx ReadTimeout before idle timeout (edge case). Content-
                # aware retry: only retry while zero content was delivered
                # (LP-0MS9FR9LG002AJ4C); after content, terminate immediately.
                try:
                    _srv().logger.warning(
                        "Upstream ReadTimeout session=%s provider=%s model=%s",
                        session_id or "unknown",
                        provider or "remote",
                        model_name,
                    )
                except Exception:
                    pass
                if _has_content:
                    _terminate_after_content = True
                else:
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
                        "Stream error: session=%s provider=%s model=%s error=%s%s",
                        session_id or "unknown",
                        provider or "remote",
                        model_name,
                        _error_type,
                        f" entry={entry}" if entry else "",
                    )
                except Exception:
                    pass
                _final_obj = _build_stream_error_event(
                    provider=provider,
                    model=model_name,
                    entry=entry,
                    error_type="stream_exception",
                    message=f"Proxy stream error ({_error_type}); upstream may be unhealthy",
                    suggested_action="Check proxy/upstream logs; the next provider in the chain may be used",
                    session_id=session_id,
                )
                _final_bytes = (
                    f"data: {json.dumps(_final_obj)}\n\n"
                ).encode()
                if collected_chunks is not None:
                    collected_chunks.append(_final_bytes)
                yield _final_bytes
                log_response_chunk(_final_bytes, session_id=session_id, model=model_name, provider=provider, body_json=body_json, entry=entry)
                break
            finally:
                # Clean up the current connection (client+cm) after each
                # attempt. For the final attempt, this runs both here and
                # in the outer finally block; close() is idempotent.
                if not (_retry_count >= max_retries and not _should_retry):
                    # Only clean up if we might retry; final cleanup is in outer finally
                    if _should_retry or _should_empty_retry or saw_done or saw_finish or disconnected:
                        try:
                            await _current_cm.__aexit__(None, None, None)
                        except Exception:
                            pass
                        if _owns_client:
                            try:
                                disconnect_cleanup_timeout = _srv().config.get("server", {}).get("disconnect_cleanup_timeout", 5.0)
                                await asyncio.wait_for(_current_client.aclose(), timeout=disconnect_cleanup_timeout)
                            except (TimeoutError, Exception):
                                pass

            # After-content stall/ReadTimeout: terminate the stream immediately
            # with a synthetic finish_reason: error instead of restarting the
            # whole request (LP-0MS9FR9LG002AJ4C). The client sees a clear
            # terminal state quickly and can retry with full context.
            if _terminate_after_content:
                try:
                    _srv().logger.warning(
                        "Upstream stall after content delivered: terminating "
                        "stream without retry session=%s provider=%s model=%s "
                        "timeout=%.1fs",
                        session_id or "unknown",
                        provider or "remote",
                        model_name,
                        upstream_idle_timeout_seconds,
                    )
                except Exception:
                    pass
                # Record the stall in the Tier 3 circuit breaker so repeated
                # stalls (before or after content) accumulate toward provider
                # cooldown (LP-0MRFEXXVC001RYKB).
                try:
                    _config = _srv().config if hasattr(_srv(), 'config') else {}
                    _check_stall_circuit_breaker(
                        provider or "remote",
                        _config,
                    )
                except Exception:
                    pass
                _final_error_obj = _build_stream_error_event(
                    provider=provider,
                    model=model_name,
                    entry=entry,
                    error_type="stall_after_content",
                    message=f"Upstream idle timeout after content delivered ({upstream_idle_timeout_seconds:.0f}s no data)",
                    suggested_action="Retry the request with full context, or route to a healthier provider",
                    session_id=session_id,
                )
                _final_error_bytes = (
                    f"data: {json.dumps(_final_error_obj)}\n\n"
                ).encode()
                if collected_chunks is not None:
                    collected_chunks.append(_final_error_bytes)
                yield _final_error_bytes
                log_response_chunk(_final_error_bytes, session_id=session_id, model=model_name, provider=provider, body_json=body_json, entry=entry)
                break

            if saw_done or saw_finish or disconnected:
                # Don't break if we're about to handle an empty-response retry;
                # the outer loop will reconnect (LP-0MRF77A0E0026B9T).
                if not _should_empty_retry:
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
            except (TimeoutError, Exception):
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


def _is_empty_remote_response(resp_json: dict) -> bool:
    """Check if a remote upstream response is semantically empty.

    Detects the specific pattern observed when free-tier upstream LLMs
    return a well-formed 200 OK with no usable content:
      - choices[0].message.content is empty (``[]``, ``""``, or absent)
      - choices[0].message.stopReason or finish_reason is ``stop``
      - choices[0].usage.total_tokens is 0 (or usage absent)

    This is used for retry-on-empty logic in the remote provider path.
    It differs from ``_is_empty_response`` in ``utils.py`` which targets
    the local llama-server path and its reasoning_content extraction.

    Args:
        resp_json: Parsed JSON body of the upstream response.

    Returns:
        True if the response is semantically empty (retry-eligible).
    """
    if not isinstance(resp_json, dict):
        return False
    try:
        choices = resp_json.get("choices", [])
        if not choices or not isinstance(choices, list):
            return False
        choice = choices[0]
        if not isinstance(choice, dict):
            return False

        message = choice.get("message", {})
        if not isinstance(message, dict):
            return False

        # Check content: must be empty (None, [], "", or absent)
        content = message.get("content")
        _is_content_empty = (
            content is None
            or (isinstance(content, list) and len(content) == 0)
            or (isinstance(content, str) and content.strip() == "")
        )
        if not _is_content_empty:
            return False

        # Check stopReason / finish_reason is "stop"
        stop_reason = message.get("stopReason") or choice.get("finish_reason")
        if stop_reason != "stop":
            return False

        # Check usage: total_tokens is 0 or absent
        usage = resp_json.get("usage", {})
        if isinstance(usage, dict):
            total = usage.get("total_tokens") or usage.get("total")
            if total is not None and total != 0:
                return False

        return True
    except Exception:
        return False


async def _handle_remote_non_streaming(
    request: Request,
    target_url: str,
    headers: dict,
    body: bytes,
    model_name: str,
    remote_timeout: httpx.Timeout,
    resolved_model: str | None = None,
    session_id: str | None = None,
    pool_client: httpx.AsyncClient | None = None,
    responses_mode: bool = False,
) -> Response:
    """Handle non-streaming remote proxy request with empty-response retry.

    When ``responses_mode`` is True (``api: openai-responses`` provider), the
    upstream body is a Responses API JSON document; it is translated to the
    chat/completions shape before the empty-response check and before being
    returned to the client (LP-0MTGK5DQO001Y8H0).


    Features:
    - Detects semantically empty upstream responses (empty content,
      stopReason: stop, total_tokens: 0) and retries the same provider
      with a configurable number of attempts and base delay.
    - After retries exhausted, returns the last response as-is so the
      caller (``proxy_with_fallback`` in provider.py) can route to the
      next provider in the fallback chain.
    - Non-empty responses pass through unchanged (no retry).
    """
    key = f"{request.method.upper()} {request.url.path} -> remote"

    # Read empty-response retry config from server settings
    server_config = _srv().config.get("server", {})
    try:
        empty_max_attempts = int(
            server_config.get("upstream_empty_retry_max_attempts", 1) or 1
        )
    except Exception:
        empty_max_attempts = 1
    try:
        empty_base_delay = float(
            server_config.get("upstream_empty_retry_base_delay_seconds", 0.5) or 0.5
        )
    except Exception:
        empty_base_delay = 0.5

    async def _do_request() -> httpx.Response:
        """Make one upstream request and return the response."""
        if pool_client is not None:
            method = request.method.lower()
            return await getattr(pool_client, method)(
                target_url,
                headers=headers,
                content=body,
                timeout=remote_timeout,
            )
        else:
            async with httpx.AsyncClient(timeout=remote_timeout) as client:
                method = request.method.lower()
                return await getattr(client, method)(
                    target_url,
                    headers=headers,
                    content=body,
                )

    last_response = None
    translated_body = None
    for attempt in range(empty_max_attempts + 1):
        response = await _do_request()
        last_response = response

        # Parse response body to check for emptiness
        try:
            resp_text = response.content.decode("utf-8", errors="replace")
            resp_json = json.loads(resp_text) if resp_text else {}
        except Exception:
            # Not valid JSON — no retry, use as-is
            break

        if responses_mode and isinstance(resp_json, dict) and response.status_code < 400:
            # Translate Responses API JSON -> chat/completions JSON so the
            # empty check and the client-visible body use the same shape.
            # Non-2xx responses pass through untranslated: upstream error
            # JSON has no ``output`` items and must reach the client/fallback
            # chain as-is.
            resp_json = _translate_responses_to_chat(resp_json)
            translated_body = json.dumps(resp_json).encode("utf-8")

        if _is_empty_remote_response(resp_json):
            if attempt < empty_max_attempts:
                try:
                    _srv().logger.info(
                        "Empty upstream response detected on attempt %s/%s, "
                        "retrying in %.2fs (model=%s)",
                        attempt + 1,
                        empty_max_attempts + 1,
                        empty_base_delay,
                        model_name,
                    )
                except Exception:
                    pass
                await asyncio.sleep(empty_base_delay)
            else:
                try:
                    _srv().logger.warning(
                        "Empty upstream response persisted after %s/%s retries, "
                        "returning empty response for fallback (model=%s)",
                        attempt + 1,
                        empty_max_attempts + 1,
                        model_name,
                    )
                except Exception:
                    pass
        else:
            # Non-empty response — no further retry
            break

    # Log and return the last response
    response = last_response

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

    _out_content = translated_body if translated_body is not None else response.content
    return Response(
        content=_out_content,
        status_code=response.status_code,
        headers=_ns_headers,
    )



