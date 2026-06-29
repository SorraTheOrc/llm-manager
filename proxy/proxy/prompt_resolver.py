"""
Prompt Resolver Module

Resolves and loads system prompt files for model aliases according to the
following precedence:

1. Local override: .sorraAgents/prompts/<alias>.txt (project-local)
2. Repo default: The path specified in model_cfg.system_prompt.file
3. No prompt if neither exists

Configuration notes (see config.yaml):
  models:
    <model_name>:
      system_prompt:
        mode: "override" | "prepend"      # required
        file: "proxy/prompts/<name>.txt"   # path relative to repo root

Size limit: MAX_PROMPT_SIZE (64 KB). Oversized files are logged and skipped.
No caching: files are read on every resolve_system_prompt() call.
"""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("llama-proxy")

# Maximum allowed prompt file size (64 KB)
MAX_PROMPT_SIZE: int = 64 * 1024

# Default override directory (project-local, relative to repo root)
OVERRIDE_DIR_RELATIVE: str = ".sorraAgents/prompts"

# Repo root: resolved from the location of this file (proxy/proxy/)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT: Path = _REPO_ROOT

# Override directory path
OVERRIDE_DIR: Path = REPO_ROOT / OVERRIDE_DIR_RELATIVE

# Valid modes
VALID_MODES = {"override", "prepend"}


def validate_prompt_config(model_cfg: dict) -> None:
    """Validate system_prompt configuration for a model entry.

    Args:
        model_cfg: A single model configuration dict from config.yaml.

    Raises:
        ValueError: If system_prompt exists but has missing or invalid mode.
    """
    prompt_cfg = model_cfg.get("system_prompt")
    if prompt_cfg is None:
        return  # No system_prompt configured - valid

    if not isinstance(prompt_cfg, dict):
        raise ValueError(
            f"system_prompt must be a dict with 'file' and 'mode' keys, "
            f"got {type(prompt_cfg).__name__}"
        )

    mode = prompt_cfg.get("mode")
    if not mode:
        raise ValueError(
            "system_prompt.mode is required when system_prompt is specified. "
            "Must be one of: override, prepend"
        )
    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid system_prompt.mode: '{mode}'. "
            f"Must be one of: {', '.join(sorted(VALID_MODES))}"
        )

    file_val = prompt_cfg.get("file")
    if not file_val:
        raise ValueError(
            "system_prompt.file is required when system_prompt is specified"
        )


def resolve_system_prompt(
    alias: str,
    model_cfg: dict,
) -> Optional[dict]:
    """Resolve and load a system prompt for the given alias/model.

    Args:
        alias: The model name or alias being requested.
        model_cfg: The resolved model configuration dict (from get_model_config).

    Returns:
        A dict with keys:
          - content (str): The prompt file content.
          - mode (str): "override" or "prepend".
          - source (str): Absolute path to the source file.
        Returns None if no prompt is configured or the file cannot be loaded.
    """
    prompt_cfg = model_cfg.get("system_prompt")
    if prompt_cfg is None:
        return None

    mode = prompt_cfg.get("mode", "prepend")
    file_path_str = prompt_cfg.get("file", "")

    if not file_path_str:
        return None

    # Determine candidates for the prompt file location.
    candidates = _resolve_candidates(alias, model_cfg, file_path_str)

    for candidate_path in candidates:
        if not candidate_path.exists() or not candidate_path.is_file():
            continue

        # Check file size first
        try:
            file_size = candidate_path.stat().st_size
        except OSError:
            continue

        if file_size > MAX_PROMPT_SIZE:
            logger.warning(
                "Prompt file %s exceeds size limit (%d bytes > %d bytes), skipping",
                candidate_path, file_size, MAX_PROMPT_SIZE,
            )
            # If the override file is too big, fall through to repo default
            continue

        # Read and validate UTF-8 encoding
        try:
            content_bytes = candidate_path.read_bytes()
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning(
                "Prompt file %s is not valid UTF-8, skipping", candidate_path
            )
            continue
        except OSError as e:
            logger.warning("Failed to read prompt file %s: %s", candidate_path, e)
            continue

        return {
            "content": content,
            "mode": mode,
            "source": str(candidate_path.resolve()),
        }

    # No valid file found — log the first non-existent path as a hint
    if candidates:
        logger.warning(
            "Prompt file not found or invalid: tried %s", candidates[0]
        )

    return None


def _resolve_candidates(
    alias: str,
    model_cfg: dict,
    file_path_str: str,
) -> list:
    """Build an ordered list of candidate file paths for prompt resolution.

    Search order:
    1. OVERRIDE_DIR/<alias>.txt  (for each alias including model name)
    2. OVERRIDE_DIR/<model_name>.txt
    3. The repo default path from config (relative or absolute)
    """
    candidates = []
    model_name = model_cfg.get("llama_model", "") or model_cfg.get("name", "")

    # All names to try for override lookup
    names_to_try = [alias]
    if model_name and model_name != alias:
        names_to_try.append(model_name)

    # Also try all aliases from the config
    aliases = model_cfg.get("aliases", [])
    for a in aliases:
        # Only add exact aliases (skip wildcard patterns for override file lookup)
        if a not in names_to_try and "*" not in a and "?" not in a:
            names_to_try.append(a)

    # 1 & 2. Check override dir for each name
    for name in names_to_try:
        candidates.append(OVERRIDE_DIR / f"{name}.txt")

    # 3. Repo default path
    repo_path = _resolve_repo_path(file_path_str)
    if repo_path not in candidates:
        candidates.append(repo_path)

    return candidates


def compose_messages(
    messages: list,
    prompt_result: Optional[dict],
) -> list:
    """Apply a resolved system prompt to a list of chat messages.

    Args:
        messages: The original list of message dicts (from request body).
        prompt_result: The result from resolve_system_prompt(), or None.

    Returns:
        Modified list of messages with the system prompt applied.
        If prompt_result is None, returns messages unchanged.
    """
    if prompt_result is None:
        return list(messages)

    content = prompt_result["content"]
    mode = prompt_result["mode"]

    if mode == "override":
        # Replace all system messages with a single new system message
        non_system = [m for m in messages if m.get("role") != "system"]
        return [{"role": "system", "content": content}] + non_system

    elif mode == "prepend":
        # Insert the prompt as the first system message
        result = list(messages)
        result.insert(0, {"role": "system", "content": content})
        return result

    else:
        # Unknown mode – pass through unchanged
        logger.warning("Unknown system_prompt mode: %s, skipping", mode)
        return list(messages)


def _resolve_repo_path(file_path_str: str) -> Path:
    """Resolve the configured file path against the repo root."""
    p = Path(file_path_str)
    if p.is_absolute():
        return p
    return REPO_ROOT / p
