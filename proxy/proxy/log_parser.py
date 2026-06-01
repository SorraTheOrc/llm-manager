"""Log severity detection utilities.

This module provides a conservative heuristic parser used to detect
log severity from a text line. It mirrors the client-side heuristics
so unit tests can validate the same behaviour.
"""
from __future__ import annotations

import json
import re
from typing import Optional

SEVERITY_ORDER = ("error", "warning", "info", "debug", "unknown")


def detect_severity(line: str) -> str:
    """Detect severity for a single log line.

    Returns one of: 'error', 'warning', 'info', 'debug', 'unknown'.
    Uses conservative heuristics:
     - If the line is JSON and contains level/severity-like fields, prefer them
     - Look for common leading tokens: ERROR, WARN, INFO, DEBUG
     - Look for square-bracketed level markers like "[ERROR]"
    """
    if not line:
        return "unknown"

    text = line.strip()

    # Try JSON parsing first
    if text.startswith("{") or text.startswith("["):
        try:
            obj = json.loads(text)
            for key in ("level", "severity", "levelname", "type", "log_level", "level_name"):
                if isinstance(obj, dict) and key in obj:
                    val = obj.get(key)
                    if isinstance(val, str):
                        lv = val.lower()
                        if "error" in lv:
                            return "error"
                        if "warn" in lv:
                            return "warning"
                        if "info" in lv:
                            return "info"
                        if "debug" in lv or "trace" in lv:
                            return "debug"
        except Exception:
            pass

    # Leading token heuristic
    parts = re.split(r"\s+", text, maxsplit=1)
    if parts:
        prefix = re.sub(r"[:\[\]]+$", "", parts[0]).upper()
        if prefix in ("ERROR", "ERR", "FATAL"):
            return "error"
        if prefix in ("WARN", "WARNING"):
            return "warning"
        if prefix in ("INFO", "NOTICE"):
            return "info"
        if prefix in ("DEBUG", "TRACE"):
            return "debug"

    # Square bracket marker
    m = re.search(r"\[\s*(ERROR|ERR|WARN|WARNING|INFO|DEBUG|TRACE)\s*\]", text, flags=re.IGNORECASE)
    if m:
        p = m.group(1).upper()
        if p.startswith("ERR"):
            return "error"
        if p.startswith("WARN"):
            return "warning"
        if p.startswith("INFO"):
            return "info"
        if p.startswith("DEBUG") or p.startswith("TRACE"):
            return "debug"

    return "unknown"
