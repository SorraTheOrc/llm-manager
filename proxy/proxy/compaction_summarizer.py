"""
Proxy-side local summarizer for session compaction (LP-0MTPMJG1P0038D32).

Production ``Summarizer`` callable backed by the local llama-server
(Qwen3). Used by the compaction planner (proxy/compaction.py) to fold
middle turns; fail-open so compaction never blocks dispatch.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

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


def _load_prompt_constants() -> tuple[str, str]:
    """Return ``(system_prompt, format_template)`` defaults from provider.

    Imports Pi's verbatim ``_SUMMARIZER_SYSTEM_PROMPT`` / ``_SUMMARIZATION_PROMPT``
    from ``proxy.provider``. Falls back to the built-in copies above on any
    import error (same-package import virtually never fails) so the
    summarizer keeps working in degraded conditions.
    """
    try:
        from proxy.provider import _SUMMARIZATION_PROMPT, _SUMMARIZER_SYSTEM_PROMPT
    except Exception:
        return _FALLBACK_SUMMARIZER_SYSTEM_PROMPT, _FALLBACK_SUMMARIZATION_PROMPT
    return _SUMMARIZER_SYSTEM_PROMPT, _SUMMARIZATION_PROMPT


def build_local_summarizer(
    config: dict | None,
    llama_port: int = 8080,
    timeout_seconds: float = 30.0,
):
    """Build a production Summarizer backed by the local llama-server.

    The returned callable matches ``proxy.compaction.Summarizer``:
    ``Callable[[list[dict]], str]`` — takes the middle messages and
    returns the summary text. Empty input returns "" without a network
    call. Any transport / HTTP / parse error is fail-open (warning log,
    return "") so compaction never blocks dispatch.

    Args:
        config: Proxy config dict (read via ``compaction_config`` for
            model name and max_tokens). ``None`` uses defaults.
        llama_port: Local llama-server port (default 8080).
        timeout_seconds: HTTP timeout for the summarization call.

    Returns:
        A ``Summarizer`` callable.
    """
    from proxy.provider import compaction_config

    cfg = compaction_config(config or {})
    model_name = cfg.get("summarizer_model_name") or "Qwen3"
    max_tokens = int(cfg.get("summarizer_max_tokens") or 512)
    _system_default, _format_template = _load_prompt_constants()
    _system_prompt = _resolve_system_prompt_override() or _system_default

    def _summarizer(middle_messages: list[dict[str, Any]]) -> str:
        if not middle_messages:
            return ""
        transcript = _transcript_for_summarizer(middle_messages)
        if not transcript.strip():
            return ""

        body = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": _system_prompt},
                # Pi's split: the transcript is serialised inside
                # <conversation> tags and the structured format template is
                # appended after them (both in the USER message) so the model
                # summarises rather than continues the conversation.
                {
                    "role": "user",
                    "content": (f"<conversation>\n{transcript}\n</conversation>\n\n{_format_template}"),
                },
            ],
            "max_tokens": max_tokens,
            "stream": False,
            "temperature": 0.2,
        }
        url = f"http://localhost:{int(llama_port)}/v1/chat/completions"
        try:
            timeout = httpx.Timeout(float(timeout_seconds))
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=body)
            if resp.status_code != 200:
                logger.warning(
                    "local summarizer non-200 status=%s body=%.500s",
                    resp.status_code,
                    getattr(resp, "text", "") or "",
                )
                return ""
            data = resp.json()
            choices = data.get("choices") if isinstance(data, dict) else None
            if not choices:
                logger.warning("local summarizer empty choices payload=%.500s", str(data)[:500])
                return ""
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
                return ""
            # R2: append file-operation XML (read/modified tool calls in the
            # summarised turns) to the model's summary, Pi-style.
            ops = extract_file_operations(middle_messages)
            return content + format_file_operations(ops["read"], ops["modified"])
        except Exception as exc:
            logger.warning("local summarizer call failed: %s", exc, exc_info=True)
            return ""

    return _summarizer
