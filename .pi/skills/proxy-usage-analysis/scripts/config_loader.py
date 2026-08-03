"""Read the proxy configuration values the analysis and recommendations need.

The source of truth is ``proxy/config.yaml`` in the llm project (referenced,
never modified). Values are read from any depth so the loader works whether
the keys sit at the top level or under a parent key (the real file nests them
under ``server:``).

YAML parsing uses ``yaml`` when available (already installed); a small regex
fallback (``_parse_config_text_regex``) covers the keys this skill needs when
the module is unavailable, so there are no hard third-party dependencies.
"""

from __future__ import annotations

import re
from pathlib import Path


# Scalar keys the skill reads (recommendations + slot schedule fallback).
SCALAR_KEYS = (
    "session_slot_pool_size",
    "session_slot_max_prompt_tokens",
    "local_large_context_cold_cache_threshold",
    "local_large_context_warm_cache_threshold",
    "local_model_ctx_size",
)


def _find_nested(cfg: dict, key: str):
    """Return the first value of ``key`` found at any depth (or ``None``)."""
    if key in cfg:
        return cfg[key]
    for value in cfg.values():
        if isinstance(value, dict):
            found = _find_nested(value, key)
            if found is not None:
                return found
    return None


def _normalize(cfg: dict | None) -> dict | None:
    """Flatten the values the skill needs into a simple dict."""
    if not cfg:
        return None
    result: dict = {}
    for key in SCALAR_KEYS:
        value = _find_nested(cfg, key)
        if value is not None:
            result[key] = int(value)
    schedule = _find_nested(cfg, "slot_schedule")
    if isinstance(schedule, dict):
        entries = []
        for e in schedule.get("entries") or []:
            if isinstance(e, dict) and e.get("time") is not None:
                entries.append((str(e["time"]), int(e.get("slots", 0))))
            elif isinstance(e, (tuple, list)) and len(e) == 2:
                entries.append((str(e[0]), int(e[1])))
        result["slot_schedule"] = {
            "enabled": bool(schedule.get("enabled", True)),
            "entries": entries,
        }
    return result or None


def parse_config_text(text: str) -> dict | None:
    """Parse config YAML text into the normalized dict the skill uses."""
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(text)
    except Exception:  # intentionally broad: fall back to the regex parser
        cfg = None
    if cfg is None:
        cfg = _parse_config_text_regex(text)
    return _normalize(cfg)


def _parse_config_text_regex(text: str) -> dict:
    """Minimal regex parser for the specific keys the skill needs.

    Comments (``# ...``) are stripped first so commented example blocks such
    as the one at the top of the real ``slot_schedule`` documentation do not
    confuse the section scanner.
    """
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.split("#", 1)[0].rstrip()
        if stripped.strip():
            lines.append(stripped)
    body = "\n".join(lines)

    cfg: dict = {}
    for key in SCALAR_KEYS:
        m = re.search(rf"^\s*{re.escape(key)}:\s*(\d+)\s*$", body, re.M)
        if m:
            cfg[key] = int(m.group(1))

    matches = [i for i, line in enumerate(lines) if re.match(r"^\s*slot_schedule:\s*$", line)]
    if matches:
        start = matches[-1]  # last occurrence (the real one; examples are comments)
        key_indent = len(lines[start]) - len(lines[start].lstrip())
        enabled = True
        entries: list[tuple[str, int]] = []
        pending_time: str | None = None
        for line in lines[start + 1 :]:
            indent = len(line) - len(line.lstrip())
            if indent <= key_indent:
                break
            m_enabled = re.match(r"^\s*enabled:\s*(true|false)\s*$", line)
            if m_enabled:
                enabled = m_enabled.group(1) == "true"
                continue
            m_time = re.match(r"^\s*-\s*time:\s*[\"']?([\d:]+)[\"']?\s*$", line)
            if m_time:
                pending_time = m_time.group(1)
                continue
            m_slots = re.match(r"^\s*slots:\s*(\d+)\s*$", line)
            if m_slots and pending_time is not None:
                entries.append((pending_time, int(m_slots.group(1))))
                pending_time = None
        cfg["slot_schedule"] = {"enabled": enabled, "entries": entries}
    return cfg


def find_config_path(explicit: str | None = None, start: Path | None = None) -> Path | None:
    """Locate ``proxy/config.yaml``.

    An explicit ``--config`` path wins. Otherwise walk up from ``start``
    (default: the current working directory) looking for ``proxy/config.yaml``.
    """
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    cursor = Path(start or Path.cwd()).resolve()
    for _ in range(8):
        candidate = cursor / "proxy" / "config.yaml"
        if candidate.is_file():
            return candidate
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    return None


def load_proxy_config(path: Path | None) -> dict | None:
    """Load and normalize the proxy config from ``path`` (``None`` if absent)."""
    if path is None or not Path(path).is_file():
        return None
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_config_text(text)
