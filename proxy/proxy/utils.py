"""
Utility Module

Pure helper functions and text-processing utilities extracted from the
monolithic server.py. Functions in this module do NOT depend on server
module-level state unless accessed via the lazy _srv() import pattern.
"""

import json
import logging
import os
import re
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import yaml

from proxy.session import ContentOnlyConsoleHandler


# ---------------------------------------------------------------------------
# Lazy server import — avoids circular imports when server.py imports us
# ---------------------------------------------------------------------------
def _srv():
    import proxy.server as _m
    return _m


# ===================================================================
# Token counting helpers
# ===================================================================

def _get_tiktoken_encoding_for_model(model_name: str | None):
    """Get the tiktoken encoding for a given model name."""
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        if model_name:
            return tiktoken.encoding_for_model(model_name)
    except Exception:
        pass
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_text_tokens(text: str, model_name: str | None = None) -> int:
    """Count tokens in text using tiktoken if available, otherwise a heuristic."""
    if not text:
        return 0
    enc = _get_tiktoken_encoding_for_model(model_name)
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # heuristic: 1 token ~ 4 bytes UTF-8
    return max(1, len(text.encode('utf-8')) // 4)


# ===================================================================
# Provider / identifier helpers
# ===================================================================

def normalize_provider_name(name: Optional[str]) -> Optional[str]:
    """Normalize provider display/identifier strings.

    Current compatibility rules:
      - "Local Proxy" (case-insensitive) is treated as "Proxy".
    """
    if not name:
        return name
    try:
        n = str(name).strip()
    except Exception:
        return name
    if n.lower() == "local proxy":
        return "Proxy"
    return n


# ===================================================================
# Tool call and response extraction helpers
# ===================================================================

def _extract_tool_call_from_reasoning(reasoning_content: Optional[str]) -> Optional[str]:
    """Extract a tool call XML pattern from reasoning_content.

    When a model with thinking mode enabled (like Qwen3) generates tool calls
    during its thinking phase, they appear in reasoning_content rather than
    content. This function extracts well-formed <function=...>...</function>
    patterns from reasoning content.

    Returns the matched tool call XML string, or None if no tool call found.
    """
    if not reasoning_content:
        return None
    # Match <function=...>...</function> block
    match = re.search(r'<function=[^>]*>.*?</function>', reasoning_content, re.DOTALL)
    if match:
        return match.group(0)
    return None


def _extract_assistant_content(resp_json: dict) -> Optional[str]:
    """Extract assistant content from a non-streaming OpenAI API response.

    Looks for choices[0].message.content and returns it.
    If content is null but reasoning_content contains a tool call
    pattern (<function=...>...</function>), the tool call is extracted
    and returned instead.
    Returns None if unable to extract content or tool call.
    """
    srv = _srv()
    try:
        choices = resp_json.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if content is not None:
                return str(content)
            # Fall back to extracting tool call from reasoning_content
            reasoning_content = message.get("reasoning_content")
            tool_call = _extract_tool_call_from_reasoning(reasoning_content)
            if tool_call:
                srv.logger.info(
                    "Extracted tool call from reasoning_content (non-streaming): %.80s",
                    tool_call,
                )
                return tool_call
    except Exception:
        pass
    return None


def _is_empty_response(response_text: str, resp_json: Optional[dict] = None) -> bool:
    """Check if a response is effectively empty (no content, no tool calls).

    Used to detect cases where the model generates thinking content but
    produces no actual output. Returns True if the response has no usable
    content.

    When resp_json is provided (OpenAI-style response), emptiness is determined
    by the presence of assistant content (text or tool calls) in the JSON
    structure, not by the raw text length.
    When resp_json is not provided, falls back to checking if response_text
    is blank or whitespace-only.
    """
    if resp_json:
        # For JSON API responses, check the structured content
        content = _extract_assistant_content(resp_json)
        if content:
            return False
        # Check reasoning_content for tool calls
        try:
            choices = resp_json.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                rc = message.get("reasoning_content")
                if rc and _extract_tool_call_from_reasoning(rc):
                    return False
        except Exception:
            pass
        return True
    # Fallback: check if the raw response text is blank or whitespace-only
    if response_text and response_text.strip():
        return False
    return True


def _extract_assistant_content_from_sse(sse_text: str) -> Optional[str]:
    """Extract concatenated assistant content from SSE stream text.

    Parses 'data: {json}' lines, extracting delta.content from each chunk.
    If no content is found, falls back to checking delta.reasoning_content
    for embedded tool call XML patterns (<function=...>...</function>).
    Returns concatenated content string, tool call string, or None.
    """
    srv = _srv()
    parts: list[str] = []
    reasoning_parts: list[str] = []
    for line in sse_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            j = json.loads(payload)
            for choice in j.get("choices", []):
                delta = choice.get("delta", {})
                if isinstance(delta, dict):
                    if "content" in delta and delta["content"] is not None:
                        parts.append(str(delta["content"]))
                    # Collect reasoning_content regardless for fallback
                    if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                        reasoning_parts.append(str(delta["reasoning_content"]))
        except Exception:
            continue

    # If we have regular content, return it (preferred)
    if parts:
        return "".join(parts)

    # Fall back to extracting tool call from accumulated reasoning_content
    if reasoning_parts:
        full_reasoning = "".join(reasoning_parts)
        tool_call = _extract_tool_call_from_reasoning(full_reasoning)
        if tool_call:
            srv.logger.info(
                "Extracted tool call from reasoning_content (streaming): %.80s",
                tool_call,
            )
            return tool_call

    return None


def _extract_delta_text_from_sse_chunk(chunk_text: str) -> str:
    """Extract assistant delta content from a single SSE chunk.

    Uses delta.content and delta.reasoning_content fields and ignores wrapper JSON.
    """
    if not chunk_text:
        return ""
    parts: list[str] = []
    for line in chunk_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            j = json.loads(payload)
        except Exception:
            continue
        for choice in j.get("choices", []):
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                continue
            for key in ("reasoning_content", "content"):
                value = delta.get(key)
                if value is not None:
                    parts.append(str(value))
    return "".join(parts)


# ===================================================================
# Header normalization
# ===================================================================

def _normalize_outgoing_headers(in_headers: dict, buffered: bool = False) -> dict:
    """Normalize headers before sending to clients.

    - If buffered=True (we are sending a full body via Response), remove
      any Transfer-Encoding header so frameworks/servers may set a proper
      Content-Length for the buffered body.
    - If buffered=False (we are streaming and will not pre-compute a
      Content-Length), remove Content-Length if Transfer-Encoding is present
      to avoid sending both headers.
    """
    if not in_headers:
        return {}
    lc_map = {k.lower(): k for k in in_headers.keys()}
    out = dict(in_headers)

    if buffered:
        # We're returning a buffered body; ensure Transfer-Encoding is not forwarded
        if 'transfer-encoding' in lc_map:
            out.pop(lc_map['transfer-encoding'], None)
    else:
        # Streaming or unknown delivery: do not forward Content-Length when TE exists
        if 'transfer-encoding' in lc_map and 'content-length' in lc_map:
            out.pop(lc_map['content-length'], None)

    return out


# ===================================================================
# Config loading
# ===================================================================

def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.environ.get(
            "LLAMA_PROXY_CONFIG",
            str(Path(__file__).parent.parent / "config.yaml")
        )

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> logging.Logger:
    """Setup logging with time-based rotation.

    Creates a TimedRotatingFileHandler and a console handler (using
    ContentOnlyConsoleHandler from the session module). Assigns the
    log_dir to the server module for use by other modules.
    """
    srv = _srv()

    # Check for dev mode
    is_dev = os.environ.get("LLAMA_PROXY_DEV") == "1"

    log_config = config.get("logging", {})
    rotation_hours = log_config.get("rotation_hours", 6)
    retention_days = log_config.get("retention_days", 90)
    log_level = log_config.get("level", "INFO")

    if is_dev:
        # Dev mode: use XDG-based dev log directory with DEBUG level
        xdg_state = os.environ.get("XDG_STATE_HOME", os.path.join(os.path.expanduser("~"), ".local", "state"))
        dir_path = Path(xdg_state) / "llama-proxy-dev" / "logs"
        dir_path.mkdir(parents=True, exist_ok=True)
        log_level = "DEBUG"
        print(f"[INFO] Dev mode: using log directory {dir_path} at level {log_level}")
    else:
        dir_path = Path(log_config.get("directory", "/var/log/llama-proxy"))
        # Try to create log directory, fall back to local logs directory if permission denied
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            dir_path = Path(__file__).parent / "logs"
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Using local log directory: {dir_path}")

    # Assign log_dir to the server module for use by other modules
    srv.log_dir = dir_path

    # Calculate backup count based on retention days and rotation interval
    backup_count = (retention_days * 24) // rotation_hours

    # Create logger
    logger = logging.getLogger("llama-proxy")
    logger.setLevel(getattr(logging, log_level.upper()))

    # File handler with rotation
    log_file = dir_path / "proxy.log"
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="H",  # Hourly rotation
        interval=rotation_hours,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(file_handler)

    # Console handler for debugging
    console_handler = ContentOnlyConsoleHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(console_handler)

    return logger
