"""
Proxy-side local summarizer for session compaction (LP-0MTPMJG1P0038D32).

Production ``Summarizer`` callable backed by the local llama-server
(Qwen3). Used by the compaction planner (proxy/compaction.py) to fold
middle turns; fail-open so compaction never blocks dispatch.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from proxy.compaction import EmptySummary

logger = logging.getLogger("llama-proxy.compaction_summarizer")


def _transport_failure_kind(exc: BaseException) -> str:
    """Classify a summarizer transport exception into a stable kind string.

    Used to populate :class:`proxy.compaction.EmptySummary` so the compaction
    planner can log why summarization failed (LP-0MTXGU8T00066WVH).
    """
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.ConnectError):
        return "connect"
    return "transport"


# ---------------------------------------------------------------------------
# File-operation tracking (R2, LP-0MTTPXI1Y003YFOU)
#
# Mirrors Pi's utils.js extractFileOpsFromMessage / computeFileLists /
# formatFileOperations: read / write / edit tool calls in the summarised
# assistant messages are collected programmatically (never left to the
# model) and appended to the summary as sorted ``<read-files>`` /
# ``<modified-files>`` XML sections. A file that was both read and modified
# is reported under ``<modified-files>`` only. Sections are omitted entirely
# when no file operations exist.
# ---------------------------------------------------------------------------
_FILE_TOOL_NAMES = frozenset({"read", "write", "edit"})


def _iter_tool_calls(messages: list[dict[str, Any]]):
    """Yield ``(name, arguments)`` for every file-ish tool call in messages.

    Recognises OpenAI-style ``assistant.tool_calls[].function`` entries and
    tolerates a flat ``{name, arguments}`` shape. Non-assistant messages and
    messages without tool_calls contribute nothing.
    """
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if isinstance(function, dict):
                yield function.get("name"), function.get("arguments")
            else:
                yield tool_call.get("name"), tool_call.get("arguments")


def _parse_tool_path(arguments: Any) -> str | None:
    """Extract the ``path`` argument from JSON-string or dict tool arguments.

    Returns ``None`` (never raises) for malformed JSON, non-dict payloads,
    or a missing/empty path.
    """
    if isinstance(arguments, str):
        try:
            data = json.loads(arguments)
        except (ValueError, TypeError):
            return None
    elif isinstance(arguments, dict):
        data = arguments
    else:
        return None
    if not isinstance(data, dict):
        return None
    path = data.get("path")
    return path if isinstance(path, str) and path.strip() else None


def extract_file_operations(
    messages: list[dict[str, Any]],
) -> dict[str, set[str]]:
    """Extract deduplicated file operations from the summarised messages.

    Args:
        messages: OpenAI-style message list (the summarizer's middle turns).

    Returns:
        ``{"read": set[str], "modified": set[str]}`` where ``modified`` is
        the union of write/edit paths and ``read`` excludes any path that was
        also modified (Pi's computeFileLists semantics). Sets are unordered;
        callers sort when rendering (see ``format_file_operations``).
    """
    read: set[str] = set()
    modified: set[str] = set()
    for name, arguments in _iter_tool_calls(messages):
        if not isinstance(name, str) or name not in _FILE_TOOL_NAMES:
            continue
        path = _parse_tool_path(arguments)
        if not path:
            continue
        if name == "read":
            read.add(path)
        else:  # write / edit
            modified.add(path)
    return {"read": read - modified, "modified": modified}


def format_file_operations(read_files, modified_files) -> str:
    """Render sorted ``<read-files>`` / ``<modified-files>`` XML sections.

    Sections are sorted for deterministic output; empty sections are omitted
    and a result with no sections returns ``""`` (callers append nothing).
    The leading blank line separates the sections from the summary body,
    matching Pi's formatFileOperations.
    """
    sections: list[str] = []
    if read_files:
        sections.append("<read-files>\n" + "\n".join(sorted(read_files)) + "\n</read-files>")
    if modified_files:
        sections.append("<modified-files>\n" + "\n".join(sorted(modified_files)) + "\n</modified-files>")
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Operator-overridable system prompt (LP-0MTTSL2AW000A5OG)
#
# The compaction summarizer's system prompt defaults to Pi's verbatim
# SUMMARIZATION_SYSTEM_PROMPT (proxy.provider._SUMMARIZER_SYSTEM_PROMPT). An
# operator may override it per-installation without code changes by dropping
# a UTF-8 text file (<= 64 KB) at:
#
#   <repo>/.sorraAgents/prompts/compaction-summarizer.txt
#
# which is the same override directory proxy/proxy/prompt_resolver.py uses
# for model-alias system prompts. Resolution is fail-open: an absent,
# oversized, invalid-UTF-8, or empty file logs a warning and falls back to
# the code default. No caching (files are tiny and re-read per summarizer
# build so operators iterate without restarting — same policy as
# prompt_resolver).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_OVERRIDE_DIR = _REPO_ROOT / ".sorraAgents" / "prompts"
_SYSTEM_PROMPT_OVERRIDE_FILENAME = "compaction-summarizer.txt"
_MAX_SYSTEM_PROMPT_SIZE = 64 * 1024  # mirror prompt_resolver.MAX_PROMPT_SIZE


def _resolve_system_prompt_override() -> str | None:
    """Read the operator override for the summarizer system prompt, if any.

    Looks for ``<repo>/.sorraAgents/prompts/compaction-summarizer.txt``.
    Returns its (stripped) content when present and valid (UTF-8, <= 64 KB),
    else ``None`` so callers fall back to
    ``proxy.provider._SUMMARIZER_SYSTEM_PROMPT``. Fail-open: invalid files
    log a warning and are ignored, never raising.
    """
    candidate = _OVERRIDE_DIR / _SYSTEM_PROMPT_OVERRIDE_FILENAME
    try:
        if not candidate.is_file():
            return None
        if candidate.stat().st_size > _MAX_SYSTEM_PROMPT_SIZE:
            logger.warning(
                "compaction summarizer system-prompt override %s exceeds %d bytes; using code default",
                candidate,
                _MAX_SYSTEM_PROMPT_SIZE,
            )
            return None
        content = candidate.read_bytes().decode("utf-8").strip()
    except UnicodeDecodeError:
        logger.warning(
            "compaction summarizer system-prompt override %s is not valid UTF-8; using code default",
            candidate,
        )
        return None
    except OSError as exc:
        logger.warning(
            "failed to read compaction summarizer system-prompt override %s: %s; using code default",
            candidate,
            exc,
        )
        return None
    return content or None


# Import-failure fallbacks (kept constant-case so ruff N811 stays quiet; the
# ``from proxy.provider import ...`` in ``build_local_summarizer`` virtually
# never fails since both live in the same package).
_FALLBACK_SUMMARIZER_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a "
    "conversation between a user and an AI coding assistant, then produce "
    "a structured summary following the exact format specified.\n"
    "\n"
    "Do NOT continue the conversation. Do NOT respond to any questions "
    "in the conversation. ONLY output the structured summary."
)
_FALLBACK_SUMMARIZATION_PROMPT = (
    "The messages above are a conversation to summarize. Create a "
    "structured context checkpoint summary that another LLM will use "
    "to continue the work. Keep each section concise. Preserve exact "
    "file paths, function names, and error messages."
)
_FALLBACK_UPDATE_SUMMARIZATION_PROMPT = (
    "The messages above are NEW conversation messages to incorporate into "
    "the existing summary provided in <previous-summary> tags. Update the "
    "existing structured summary, PRESERVING all existing information and "
    "adding new progress, decisions, and context. Preserve exact file paths, "
    "function names, and error messages."
)


def _message_content_text(msg: dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(p for p in parts if p)
    return str(content) if content is not None else ""


def _transcript_for_summarizer(middle_messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for m in middle_messages:
        role = str(m.get("role", "user"))
        text = _message_content_text(m).strip()
        if not text:
            continue
        lines.append(f"{role}: {text}")
    return "\n\n".join(lines)


def _load_prompt_constants() -> tuple[str, str, str]:
    """Return ``(system_prompt, format_template, update_template)`` defaults.

    Imports Pi's verbatim ``_SUMMARIZER_SYSTEM_PROMPT`` /
    ``_SUMMARIZATION_PROMPT`` / ``_UPDATE_SUMMARIZATION_PROMPT`` from
    ``proxy.provider``. Falls back to the built-in copies above on any import
    error (same-package import virtually never fails) so the summarizer
    keeps working in degraded conditions.
    """
    try:
        from proxy.provider import (
            _SUMMARIZATION_PROMPT,
            _SUMMARIZER_SYSTEM_PROMPT,
            _UPDATE_SUMMARIZATION_PROMPT,
        )
    except Exception:
        return (
            _FALLBACK_SUMMARIZER_SYSTEM_PROMPT,
            _FALLBACK_SUMMARIZATION_PROMPT,
            _FALLBACK_UPDATE_SUMMARIZATION_PROMPT,
        )
    return _SUMMARIZER_SYSTEM_PROMPT, _SUMMARIZATION_PROMPT, _UPDATE_SUMMARIZATION_PROMPT


def build_local_summarizer(
    config: dict | None,
    llama_port: int = 8080,
    timeout_seconds: float | None = None,
):
    """Build a production Summarizer backed by the local llama-server.

    The returned callable matches ``proxy.compaction.Summarizer``:
    ``Callable[[list[dict]], str]`` — takes the middle messages and
    returns the summary text. Empty input returns "" without a network
    call. Any transport / HTTP / parse error is fail-open: the call returns
    an :class:`proxy.compaction.EmptySummary` — a falsy ``str`` subclass
    that compares equal to ``""`` but carries the failure ``kind`` and the
    number of ``attempts`` made. The compaction planner treats any blank
    summary (sentinel or plain ``""``) as a summarizer failure and routes
    ``remote_with_guidance`` rather than applying an empty summary
    (LP-0MTXGU8T00066WVH); dispatch is therefore never blocked.

    Args:
        config: Proxy config dict (read via ``compaction_config`` for
            model name, max_tokens and the timeout). ``None`` uses defaults.
        llama_port: Local llama-server port (default 8080).
        timeout_seconds: HTTP timeout for the summarization call. ``None``
            resolves ``config["server"]["compaction_summarizer_timeout"]``
            (default 600 s — see ``_DEFAULT_SUMMARIZER_TIMEOUT_SECONDS``)
            so the summarizer can wait for the single local slot to free up
            while a long generating request holds it (LP-0MU1RXEY10075TUU).

    Returns:
        A ``Summarizer`` callable.
    """
    from proxy.provider import compaction_config

    cfg = compaction_config(config or {})
    model_name = cfg.get("summarizer_model_name") or "Qwen3"
    max_tokens = int(cfg.get("summarizer_max_tokens") or 512)
    _system_default, _format_template, _update_template = _load_prompt_constants()
    _system_prompt = _resolve_system_prompt_override() or _system_default
    retries = int(cfg.get("summarizer_retries") or 0)
    retry_delay = float(cfg.get("summarizer_retry_delay_seconds") or 0.0)
    if timeout_seconds is None:
        timeout_seconds = float(cfg.get("summarizer_timeout_seconds") or 600.0)

    def _summarizer(
        middle_messages: list[dict[str, Any]],
        previous_summary: str | None = None,
    ) -> str:
        if not middle_messages:
            return ""
        transcript = _transcript_for_summarizer(middle_messages)
        if not transcript.strip():
            return ""

        # R5 (LP-0MTTPXIIX005Y0Z9) — Pi's message split: the transcript is
        # serialised inside <conversation> tags so the model summarises
        # rather than continues the conversation; a previous compaction
        # summary is passed in a <previous-summary> block and switches the
        # template to the incremental UPDATE prompt.
        user_content = f"<conversation>\n{transcript}\n</conversation>"
        if previous_summary:
            user_content += f"\n\n<previous-summary>\n{previous_summary}\n</previous-summary>"
            template = _update_template
        else:
            template = _format_template
        user_content += f"\n\n{template}"

        body = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": _system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "stream": False,
            "temperature": 0.2,
        }
        url = f"http://localhost:{int(llama_port)}/v1/chat/completions"
        timeout = httpx.Timeout(float(timeout_seconds))
        attempts = retries + 1  # initial attempt + configured retries
        try:
            # R3 (LP-0MTTPXIAB0031AC4): retry TRANSIENT failures only —
            # connection errors / read timeouts (httpx.TransportError) and
            # HTTP 5xx / 429. Other 4xx (auth, model-not-found) are permanent
            # and fail open immediately. Each retry is logged at WARNING;
            # after exhaustion the summarizer returns "" (fail-open).
            for attempt in range(attempts):
                try:
                    with httpx.Client(timeout=timeout) as client:
                        resp = client.post(url, json=body)
                except httpx.TransportError as exc:
                    if attempt < retries:
                        logger.warning(
                            "local summarizer transport error (attempt %d/%d): %s; retrying in %.2fs",
                            attempt + 1,
                            attempts,
                            exc,
                            retry_delay,
                        )
                        if retry_delay > 0:
                            time.sleep(retry_delay)
                        continue
                    logger.warning(
                        "local summarizer call failed after %d attempts: %s",
                        attempts,
                        exc,
                        exc_info=True,
                    )
                    return EmptySummary(_transport_failure_kind(exc), attempts)
                if resp.status_code == 200:
                    break
                transient = resp.status_code >= 500 or resp.status_code == 429
                if transient and attempt < retries:
                    logger.warning(
                        "local summarizer non-200 status=%s (attempt %d/%d); retrying in %.2fs",
                        resp.status_code,
                        attempt + 1,
                        attempts,
                        retry_delay,
                    )
                    if retry_delay > 0:
                        time.sleep(retry_delay)
                    continue
                logger.warning(
                    "local summarizer non-200 status=%s body=%.500s",
                    resp.status_code,
                    getattr(resp, "text", "") or "",
                )
                return EmptySummary(f"http_{resp.status_code}", attempt + 1)
            data = resp.json()
            choices = data.get("choices") if isinstance(data, dict) else None
            if not choices:
                logger.warning("local summarizer empty choices payload=%.500s", str(data)[:500])
                return EmptySummary("malformed_response", attempts)
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            content = (msg or {}).get("content") if isinstance(msg, dict) else None
            if not isinstance(content, str):
                # Some servers return content as list parts
                if isinstance(content, list):
                    content = "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
                else:
                    content = str(content or "")
            content = content.strip()
            if not content:
                return EmptySummary("empty_completion", attempts)
            # R2: append file-operation XML (read/modified tool calls in the
            # summarised turns) to the model's summary, Pi-style.
            ops = extract_file_operations(middle_messages)
            return content + format_file_operations(ops["read"], ops["modified"])
        except Exception as exc:
            logger.warning("local summarizer call failed: %s", exc, exc_info=True)
            return EmptySummary(type(exc).__name__, attempts)

    return _summarizer


# ---------------------------------------------------------------------------
# Remote-only ``compact`` summarizer (LP-0MTT0O74N009E7N2)
#
# Under local-slot saturation every local summarizer call timed out (49/49 at
# 30 s) because the single slot was held by a long generating request. The
# ``models.compact`` chain therefore contains NO local tier: Muse via
# opencode-go first, DeepSeek (api.deepseek.com) second. ``build_compact_summarizer``
# resolves the declared providers itself and calls each one directly over
# httpx — reusing the proxy's provider auth / failure-domain plumbing
# (``resolve_provider``, pi ``auth.json`` fallback, Responses-API translation)
# without going through localhost:8080 (AC2/AC5). Fail-open: when every tier
# fails it returns an :class:`proxy.compaction.EmptySummary` (falsy ``""``) so
# compaction routes ``remote_with_guidance`` instead of blocking dispatch.
#
# Backward compatibility (AC3): when ``models.compact`` is absent the builder
# delegates to :func:`build_local_summarizer`, so an operator who still uses
# ``server.summarizer_model: {type: local, llama_model: Qwen3}`` keeps the
# previous local behaviour unchanged.
# ---------------------------------------------------------------------------
_COMPACT_MODEL_KEY = "compact"


def _compact_model_config(config: dict | None) -> dict | None:
    """Return ``models.compact`` when it declares at least one provider.

    Returns ``None`` for a missing/invalid ``models.compact`` so the caller
    can fall back to the local summarizer (AC3).
    """
    if not isinstance(config, dict):
        return None
    models = config.get("models")
    if not isinstance(models, dict):
        return None
    model_cfg = models.get(_COMPACT_MODEL_KEY)
    if not isinstance(model_cfg, dict):
        return None
    providers = model_cfg.get("providers")
    if not isinstance(providers, list) or not providers:
        return None
    return model_cfg


def _remote_chat_url(provider_cfg: dict) -> str:
    """Return the upstream URL for a provider entry.

    Mirrors ``proxy_remote.proxy_to_remote``: ``api: openai-responses``
    providers are called on ``/v1/responses``; everything else uses
    ``/v1/chat/completions``.
    """
    endpoint = str(provider_cfg.get("endpoint") or "").rstrip("/")
    if provider_cfg.get("api") == "openai-responses":
        return f"{endpoint}/v1/responses"
    return f"{endpoint}/v1/chat/completions"


def _resolve_remote_api_key(provider_cfg: dict) -> str | None:
    """Resolve a provider API key from env, inline config, or pi auth.json.

    Same precedence as ``proxy_remote.proxy_to_remote``:
    ``api_key_env`` -> inline ``api_key`` -> ``~/.pi/agent/auth.json``.
    """
    import os

    api_key = None
    api_key_env = provider_cfg.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env)
    if not api_key:
        api_key = provider_cfg.get("api_key")
    if not api_key:
        try:
            from proxy.proxy_remote import _try_pi_auth_json

            api_key = _try_pi_auth_json(api_key_env or "")
        except Exception:
            api_key = None
    return api_key


def _extract_summary_content(payload: Any) -> str:
    """Extract assistant text from a chat/completions-shaped payload.

    Tolerates list-form ``content`` parts; returns a stripped ``str`` (empty
    when the payload carries no usable content).
    """
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else None
    msg = first.get("message") if isinstance(first, dict) else None
    content = msg.get("content") if isinstance(msg, dict) else None
    if not isinstance(content, str):
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        else:
            content = str(content or "")
    return content.strip()


def _post_compact_tier(
    provider_cfg: dict,
    body: dict[str, Any],
    timeout: float,
    *,
    opencode_session: str | None = None,
) -> tuple[str, str]:
    """POST one ``compact`` tier and return ``(content, failure_kind)``.

    ``content`` is the summary text on success (and ``failure_kind`` is
    ``""``); on failure ``content`` is ``""`` and ``failure_kind`` is a
    stable machine-readable reason (``"timeout"``, ``"connect"``,
    ``"transport"``, ``"http_<status>"``, ``"malformed_response"``,
    ``"empty_completion"``). Never raises.
    """
    responses_mode = provider_cfg.get("api") == "openai-responses"
    url = _remote_chat_url(provider_cfg)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = _resolve_remote_api_key(provider_cfg)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    custom_headers = provider_cfg.get("headers")
    if isinstance(custom_headers, dict):
        headers.update(custom_headers)
    attribution_headers = provider_cfg.get("attribution_headers")
    if isinstance(attribution_headers, dict):
        headers.update(attribution_headers)

    translate_to_responses = None
    translate_to_chat = None
    try:
        from proxy.proxy_remote import (
            _is_opencode_upstream,
            _sanitize_header_value,
            _translate_chat_to_responses,
            _translate_responses_to_chat,
        )

        translate_to_responses = _translate_chat_to_responses
        translate_to_chat = _translate_responses_to_chat
        # opencode.ai gateways reject requests without an x-opencode-session
        # header (HTTP 400 MissingSessionID, LP-0MTR3CHEP007S699); synthesize
        # a per-call value since compaction has no client session id.
        if opencode_session and _is_opencode_upstream(provider_cfg.get("endpoint", "")):
            headers["x-opencode-session"] = _sanitize_header_value(opencode_session)
    except Exception:  # pragma: no cover - same-package import virtually never fails
        logger.debug("compact summarizer: Responses-API helpers unavailable", exc_info=True)

    outbound = dict(body)
    if responses_mode and translate_to_responses is not None:
        outbound = translate_to_responses(outbound)

    try:
        with httpx.Client(timeout=httpx.Timeout(float(timeout))) as client:
            resp = client.post(url, json=outbound, headers=headers)
    except httpx.TransportError as exc:
        return "", _transport_failure_kind(exc)
    except Exception as exc:
        logger.warning("compact summarizer transport failure: %s", exc, exc_info=True)
        return "", type(exc).__name__

    if resp.status_code != 200:
        logger.warning(
            "compact summarizer non-200 status=%s body=%.500s",
            resp.status_code,
            getattr(resp, "text", "") or "",
        )
        return "", f"http_{resp.status_code}"
    try:
        data = resp.json()
    except Exception:
        return "", "malformed_response"
    if responses_mode and translate_to_chat is not None:
        try:
            data = translate_to_chat(data)
        except Exception:
            return "", "malformed_response"
    content = _extract_summary_content(data)
    if not content:
        return "", "empty_completion"
    return content, ""


def build_compact_summarizer(
    config: dict | None,
    llama_port: int = 8080,
    timeout_seconds: float | None = None,
):
    """Build a production Summarizer backed by the remote ``compact`` model.

    The chain is declared as ``models.compact`` in the proxy config (Muse via
    opencode-go, then DeepSeek via api.deepseek.com — strictly remote, no
    local tier). Providers are resolved with ``proxy.provider.resolve_provider``
    so cooldowns, ``available_times`` windows and failure-domain grouping are
    honoured exactly as on the normal dispatch path. Each failed tier is
    logged at WARNING with its reason before the next tier is tried; when
    every tier fails the summarizer returns an
    :class:`proxy.compaction.EmptySummary` (falsy ``""``) — fail-open, so
    compaction never blocks dispatch.

    Backward compatibility (AC3): when ``models.compact`` is absent the
    returned callable is :func:`build_local_summarizer` (the previous local
    Qwen3 behaviour), so operators still on
    ``server.summarizer_model: {type: local, llama_model: Qwen3}`` are
    unaffected.

    Args:
        config: Proxy config dict (``models.compact`` selects the remote
            chain; ``server.compaction_summarizer_timeout`` bounds each tier).
            ``None`` falls back to the local summarizer.
        llama_port: Local llama-server port used only by the local fallback.
        timeout_seconds: Per-tier HTTP timeout. ``None`` resolves
            ``models.compact.timeout_seconds`` when present, else
            ``server.compaction_summarizer_timeout`` (default 600 s).

    Returns:
        A ``Summarizer`` callable.
    """
    compact_cfg = _compact_model_config(config)
    if compact_cfg is None:
        logger.info(
            "models.compact not configured; using the local summarizer "
            "(set models.compact to route compaction through the remote chain)"
        )
        return build_local_summarizer(
            config,
            llama_port=llama_port,
            timeout_seconds=timeout_seconds,
        )

    from proxy.provider import compaction_config, resolve_provider

    cfg = compaction_config(config or {})
    max_tokens = int(cfg.get("summarizer_max_tokens") or 512)
    if timeout_seconds is None:
        timeout_seconds = float(cfg.get("summarizer_timeout_seconds") or 600.0)
    try:
        per_tier_timeout = float(compact_cfg.get("timeout_seconds") or timeout_seconds)
    except (TypeError, ValueError):
        per_tier_timeout = float(timeout_seconds)

    _system_default, _format_template, _update_template = _load_prompt_constants()
    _system_prompt = _resolve_system_prompt_override() or _system_default
    declared_providers = compact_cfg.get("providers") or []
    tier_count = max(1, len(declared_providers))

    def _summarizer(
        middle_messages: list[dict[str, Any]],
        previous_summary: str | None = None,
    ) -> str:
        if not middle_messages:
            return ""
        transcript = _transcript_for_summarizer(middle_messages)
        if not transcript.strip():
            return ""

        user_content = f"<conversation>\n{transcript}\n</conversation>"
        if previous_summary:
            user_content += f"\n\n<previous-summary>\n{previous_summary}\n</previous-summary>"
            template = _update_template
        else:
            template = _format_template
        user_content += f"\n\n{template}"

        # Per-call opencode session id; opencode.ai rejects requests without
        # the header and compaction has no client session to derive it from.
        opencode_session = f"compact-{uuid.uuid4().hex}"
        attempts = 0
        last_kind = "no_provider"
        failed_provider: str | None = None
        while attempts < tier_count:
            provider = resolve_provider(compact_cfg, failed_provider=failed_provider)
            if provider is None:
                break
            name = str(provider.get("name") or "?")
            attempts += 1
            body = {
                "model": provider.get("model") or _COMPACT_MODEL_KEY,
                "messages": [
                    {"role": "system", "content": _system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": max_tokens,
                "stream": False,
                "temperature": 0.2,
            }
            content, kind = _post_compact_tier(
                provider,
                body,
                per_tier_timeout,
                opencode_session=opencode_session,
            )
            if content:
                ops = extract_file_operations(middle_messages)
                return content + format_file_operations(ops["read"], ops["modified"])
            last_kind = kind or "unknown"
            logger.warning(
                "compact summarizer provider=%s failed (%s); falling back to next "
                "provider",
                name,
                last_kind,
            )
            failed_provider = name

        if attempts == 0:
            logger.warning(
                "compact summarizer: no eligible provider in models.compact chain"
            )
            return EmptySummary("no_provider", 0)
        return EmptySummary(last_kind, attempts)

    return _summarizer
