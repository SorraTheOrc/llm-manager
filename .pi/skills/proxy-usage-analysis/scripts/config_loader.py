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
from datetime import datetime as _datetime
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
    priority_window = normalize_priority_window(_find_nested(cfg, "priority_window"))
    if priority_window is not None:
        result["priority_window"] = priority_window
    models = _find_nested(cfg, "models")
    if isinstance(models, dict):
        # Carried through so a deduplicated mode overlay still exposes the
        # full model/provider surface of the merged config.
        result["models"] = models
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


def parse_window_minutes(value: object, default: int) -> int:
    """Parse a ``"HH:MM"`` clock bound to minutes from midnight (local time).

    Mirrors ``proxy/proxy/priority.py::in_cheap_window``: the bounds are local
    wall-clock times, so no UTC conversion is applied. Invalid or missing
    values return ``default`` (LP-0MVKSXXIR006OBI2 AC4).
    """
    try:
        hh, mm = str(value).strip().split(":")[:2]
        hours, minutes = int(hh), int(mm)
    except (ValueError, AttributeError, TypeError):
        return int(default)
    if not (0 <= hours < 24 and 0 <= minutes < 60):
        return int(default)
    return hours * 60 + minutes


def _as_bool(value: object, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("true", "yes", "1", "on")


def normalize_priority_window(window: dict | None) -> dict | None:
    """Canonical ``priority_window`` block (``None`` when absent).

    Returns ``{"enabled", "start", "end", "start_minutes", "end_minutes"}``
    so every consumer sees the same parsed shape; ``end <= start`` wraps past
    midnight (e.g. ``22:00``–``06:00``). Defaults match the shipped config
    (01:00–10:00, the retired mode-schedule cheap window).
    """
    if not isinstance(window, dict):
        return None
    start = str(window.get("start") or "01:00")
    end = str(window.get("end") or "10:00")
    return {
        "enabled": _as_bool(window.get("enabled"), True),
        "start": start,
        "end": end,
        "start_minutes": parse_window_minutes(start, 1 * 60),
        "end_minutes": parse_window_minutes(end, 10 * 60),
    }


def priority_window_bounds(config: dict | None) -> dict | None:
    """The parsed ``server.priority_window`` block of an effective config."""
    if not isinstance(config, dict):
        return None
    window = config.get("priority_window")
    if isinstance(window, dict):
        # Already normalised by ``_normalize``; re-normalising is harmless and
        # tolerates a raw dict from a caller that skipped the loader.
        return normalize_priority_window(window)
    return normalize_priority_window(_find_nested(config, "priority_window"))


def _parse_config_raw(text: str) -> dict:
    """Parse config YAML text into a raw dict (yaml, else the regex parser)."""
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(text)
    except Exception:  # intentionally broad: fall back to the regex parser
        cfg = None
    if not isinstance(cfg, dict):
        cfg = _parse_config_text_regex(text)
    return cfg


def parse_config_text(text: str) -> dict | None:
    """Parse config YAML text into the normalized dict the skill uses."""
    return _normalize(_parse_config_raw(text))


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge *overlay* onto *base* (scalars/lists replace).

    Mirrors ``proxy.config_merge.deep_merge`` so the standalone skill resolves
    the same base+overlay config the proxy does (LP-0MU0W548B0029255).
    """
    result = dict(base)
    for key, value in overlay.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result


_MODE_OVERLAY_RE = re.compile(r"^config-(?:fast|cheap)\.yaml$")


def _base_for_overlay(path: Path) -> Path | None:
    """Sibling ``config.yaml`` when *path* is a deduplicated mode overlay.

    The mode profiles are overlays that no longer repeat base-inherited values
    (for example ``local_model_ctx_size``), so the loader must merge them onto
    the base to see the effective config.
    """
    if _MODE_OVERLAY_RE.match(path.name):
        base = path.parent / "config.yaml"
        return base if base.is_file() else None
    return None


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

    window_matches = [
        i for i, line in enumerate(lines) if re.match(r"^\s*priority_window:\s*$", line)
    ]
    if window_matches:
        start_idx = window_matches[-1]
        key_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        window: dict = {}
        for line in lines[start_idx + 1 :]:
            indent = len(line) - len(line.lstrip())
            if indent <= key_indent:
                break
            m_field = re.match(r"^\s*(enabled|start|end):\s*(.+?)\s*$", line)
            if not m_field:
                continue
            field, raw_value = m_field.group(1), m_field.group(2).strip().strip("\"'")
            window[field] = raw_value
        if window:
            cfg["priority_window"] = window
    return cfg


def find_config_base_path(explicit: str | None = None, start: Path | None = None) -> Path | None:
    """Locate ``proxy/config.yaml`` (without the mode preference).

    An explicit ``--config`` path wins. Otherwise walk up from ``start``
    (default: the current working directory) looking for ``proxy/config.yaml``
    (the single authoritative config; the mode profiles and persisted
    ``proxy/.mode`` were removed, LP-0MVDT94F0003VHUA AC4).
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


def priority_window_label(config_yaml: Path) -> str | None:
    """Return the priority-window state label (``"low"``/``"standard"``).

    The fast/cheap mode axis was removed (LP-0MVDT94F0003VHUA AC4): there is
    no persisted ``.mode`` file. The label is derived from the unified
    config's ``server.priority_window`` (cheap window) at call time.
    """
    cfg = load_proxy_config(config_yaml) or {}
    window = priority_window_bounds(cfg)
    if window is None or not window["enabled"]:
        return "standard"
    start = window["start_minutes"]
    end = window["end_minutes"]
    now = _datetime.now()
    minutes = now.hour * 60 + now.minute + now.second / 60.0
    inside = (start <= minutes < end) if start <= end else (
        minutes >= start or minutes < end
    )
    return "low" if inside else "standard"


def discover_configs(explicit: str | None = None, start: Path | None = None) -> dict:
    """Locate ``proxy/config.yaml`` and load everything the analysis needs.

    ``config.yaml`` is the single authoritative config (the fast/cheap mode
    overlays were removed, LP-0MVDT94F0003VHUA AC4). Returns a dict with:

    - ``base``: the ``proxy/config.yaml`` path (or ``None``);
    - ``profiles``: ``{"default": <parsed>}`` (historical shape retained);
    - ``analysis_mode``: legacy key retained as ``None`` (there is no mode);
    - ``analysis_config``: the unified config;
    - ``priority_window``: the current cheap-window label (``low`` inside,
      else ``standard``);
    - ``priority_window_bounds``: the parsed window bounds (``start_minutes`` /
      ``end_minutes`` / ``enabled``) used to bucket historical windows;
      ``None`` when the config has no priority window.
    """
    base = find_config_base_path(explicit, start)
    profiles: dict[str, dict | None] = {"default": None}
    analysis_mode: str | None = None
    label: str | None = None
    if base is not None:
        profiles["default"] = load_proxy_config(base)
        label = priority_window_label(base)
    return {
        "base": base,
        "profiles": profiles,
        "analysis_mode": analysis_mode,
        "priority_window": label,
        "priority_window_bounds": priority_window_bounds(profiles["default"]),
        "analysis_config": profiles["default"],
    }


def find_config_path(explicit: str | None = None, start: Path | None = None) -> Path | None:
    """Locate ``proxy/config.yaml`` (the single unified config)."""
    return find_config_base_path(explicit, start)


def load_proxy_config(path: Path | None) -> dict | None:
    """Load and normalize the proxy config from ``path`` (``None`` if absent).

    ``config.yaml`` is the single authoritative config (the mode overlays
    were removed, LP-0MVDT94F0003VHUA AC4). A legacy custom overlay path is
    deep-merged onto the base first, so inherited
    values such as ``local_model_ctx_size`` are present.
    """
    if path is None or not Path(path).is_file():
        return None
    path = Path(path)
    cfg = _parse_config_raw(path.read_text(encoding="utf-8", errors="replace"))
    base_path = _base_for_overlay(path)
    if base_path is not None:
        base_cfg = _parse_config_raw(
            base_path.read_text(encoding="utf-8", errors="replace")
        )
        cfg = _deep_merge(base_cfg, cfg)
    return _normalize(cfg)
