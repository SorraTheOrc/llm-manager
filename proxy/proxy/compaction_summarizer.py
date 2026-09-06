
# <!-- REFACTOR-LP-0MTPNMF3Q001VKA6
# smell: naming
# severity: medium
# description: Variable `_PROMPT` in function should be lowercase
# -->
"""
Proxy-side local summarizer for session compaction (LP-0MTPMJG1P0038D32).

Production ``Summarizer`` callable backed by the local llama-server
(Qwen3). Used by the compaction planner (proxy/compaction.py) to fold
middle turns; fail-open so compaction never blocks dispatch.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


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
    try:
        from proxy.provider import _SUMMARIZER_SYSTEM_PROMPT as _PROMPT
    except Exception:
        _PROMPT = (
            "Summarise the middle portion of this conversation for context retention. "
            "Preserve essential instructions, decisions, and key facts."
        )

    def _summarizer(middle_messages: list[dict[str, Any]]) -> str:
        if not middle_messages:
            return ""
        transcript = _transcript_for_summarizer(middle_messages)
        if not transcript.strip():
            return ""

        body = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": _PROMPT},
                {"role": "user", "content": transcript},
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
                    content = "\n".join(
                        str(p.get("text", "")) for p in content if isinstance(p, dict)
                    )
                else:
                    content = str(content or "")
            return content.strip()
        except Exception as exc:
            logger.warning("local summarizer call failed: %s", exc, exc_info=True)
            return ""

    return _summarizer
