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
        ctx_by_time: dict[str, int] = {}
        for e in schedule.get("entries") or []:
            if isinstance(e, dict) and e.get("time") is not None:
                time_str = str(e["time"])
                entries.append((time_str, int(e.get("slots", 0))))
                if e.get("ctx_size") is not None:
                    ctx_by_time[time_str] = int(e["ctx_size"])
            elif isinstance(e, (tuple, list)) and len(e) == 2:
                entries.append((str(e[0]), int(e[1])))
        result["slot_schedule"] = {
            "enabled": bool(schedule.get("enabled", True)),
            "entries": entries,
        }
        if ctx_by_time:
            result["slot_schedule"]["ctx_by_time"] = ctx_by_time
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
        ctx_by_time: dict[str, int] = {}
        pending: dict | None = None
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
                if pending is not None and pending["slots"] is not None:
                    entries.append((pending["time"], pending["slots"]))
                    if pending["ctx"] is not None:
                        ctx_by_time[pending["time"]] = pending["ctx"]
                pending = {"time": m_time.group(1), "slots": None, "ctx": None}
                continue
            m_slots = re.match(r"^\s*slots:\s*(\d+)\s*$", line)
            if m_slots and pending is not None:
                pending["slots"] = int(m_slots.group(1))
                continue
            m_ctx = re.match(r"^\s*ctx_size:\s*(\d+)\s*$", line)
            if m_ctx and pending is not None:
                pending["ctx"] = int(m_ctx.group(1))
                continue
        if pending is not None and pending["slots"] is not None:
            entries.append((pending["time"], pending["slots"]))
            if pending["ctx"] is not None:
                ctx_by_time[pending["time"]] = pending["ctx"]
        cfg["slot_schedule"] = {"enabled": enabled, "entries": entries}
        if ctx_by_time:
            cfg["slot_schedule"]["ctx_by_time"] = ctx_by_time
    return cfg


def find_config_base_path(explicit: str | None = None, start: Path | None = None) -> Path | None:
    """Locate ``proxy/config.yaml`` (without the mode preference).

    An explicit ``--config`` path wins. Otherwise walk up from ``start``
    (default: the current working directory) looking for ``proxy/config.yaml``.
    Unlike :func:`find_config_path`, this returns the *base* config so the
    caller can discover the sibling mode profiles (``config-fast.yaml`` /
    ``config-cheap.yaml``) and the persisted ``proxy/.mode``.
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


def read_mode(config_yaml: Path) -> str | None:
    """Return the persisted operating mode (``proxy/.mode``), or ``None``.

    Only ``fast``/``cheap`` are valid; anything else (missing file, garbage)
    returns ``None`` (fail-open, LP-0MSLMYEEU002IBH6).
    """
    mode_file = config_yaml.parent / ".mode"
    try:
        mode = mode_file.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return None
    return mode if mode in ("fast", "cheap") else None


def discover_configs(explicit: str | None = None, start: Path | None = None) -> dict:
    """Locate ``proxy/config.yaml`` and load everything the analysis needs.

    Returns a dict with:

    - ``base``: the base ``proxy/config.yaml`` path (or ``None``);
    - ``profiles``: ``{"default": <parsed>, "fast": <parsed>, "cheap":
      <parsed>}`` (mode profiles are ``None`` when the file is absent);
    - ``analysis_mode``: the persisted ``proxy/.mode`` value when it selects
      an existing profile (else ``None``);
    - ``analysis_config``: the mode-selected config (the profile for
      ``analysis_mode``, else the default) — the backward-compatible
      single-config view used by the recommendations.
    """
    base = find_config_base_path(explicit, start)
    profiles: dict[str, dict | None] = {"default": None, "fast": None, "cheap": None}
    analysis_mode: str | None = None
    if base is not None:
        profiles["default"] = load_proxy_config(base)
        for mode in ("fast", "cheap"):
            profile_path = base.parent / f"config-{mode}.yaml"
            if profile_path.is_file():
                profiles[mode] = load_proxy_config(profile_path)
        mode = read_mode(base)
        if mode is not None and profiles[mode] is not None:
            analysis_mode = mode
    analysis_config = profiles.get(analysis_mode) if analysis_mode else profiles["default"]
    return {
        "base": base,
        "profiles": profiles,
        "analysis_mode": analysis_mode,
        "analysis_config": analysis_config,
    }


def find_config_path(explicit: str | None = None, start: Path | None = None) -> Path | None:
    """Locate ``proxy/config.yaml`` (mode-aware).

    An explicit ``--config`` path wins. Otherwise walk up from ``start``
    (default: the current working directory) looking for ``proxy/config.yaml``.
    When the persisted operating mode (``proxy/.mode``, LP-0MSLMYEEU002IBH6)
    selects a profile, the mode-selected file (``config-fast.yaml`` /
    ``config-cheap.yaml``) is returned instead so bucketing/recommendations
    read the config the running proxy actually uses; ``config.yaml`` remains
    the default when no mode is persisted (or the mode file is missing).
    """
    base = find_config_base_path(explicit, start)
    if base is None:
        return None
    return _mode_preferred_config(base)


def _mode_preferred_config(config_yaml: Path) -> Path:
    """Return the mode-selected profile when a valid mode is persisted.

    Reads ``proxy/.mode`` next to ``config.yaml``; ``fast``/``cheap`` select
    ``config-fast.yaml``/``config-cheap.yaml`` when those files exist.
    Anything else (missing/invalid mode, missing profile) falls back to
    ``config.yaml`` (the default/fallback profile).
    """
    mode_file = config_yaml.parent / ".mode"
    try:
        mode = mode_file.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return config_yaml
    if mode in ("fast", "cheap"):
        selected = config_yaml.parent / f"config-{mode}.yaml"
        if selected.is_file():
            return selected
    return config_yaml


def load_proxy_config(path: Path | None) -> dict | None:
    """Load and normalize the proxy config from ``path`` (``None`` if absent)."""
    if path is None or not Path(path).is_file():
        return None
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_config_text(text)
