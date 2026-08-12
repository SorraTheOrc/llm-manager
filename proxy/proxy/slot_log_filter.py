"""Pure per-slot log filtering helpers for the Web UI "Slots" tab.

This module implements the slot-log relevance rule (LP-0MSHET5SI000LYSK):

- **llama-server.log** lines are attributed to a slot by their slot id
  marker, observed live as ``id <N> |`` (e.g.
  ``[57463] slot update_slots: id  2 | task 209403 | n_tokens = 16750, ...``)
  or the ``id=<N>`` form used by progress lines
  (``slot update_slots: id=5 n_tokens=4096 progress=0.17``).
  Non-slot lines (``srv  log_server_r: ...``) carry no marker and are
  excluded from every slot section.

- **proxy.log** lines are attributed primarily by their ``session=<uuid>``
  marker against the slot's mapped dispatch session, with ``slot=<n>`` as a
  fallback when no session mapping is available.  Lines tagged ``slot=none``
  (session not yet assigned to a persistence slot) or carrying no markers
  are "unmapped" and excluded from slot sections (they remain visible on the
  "All Logs" tab).

The functions are pure and side-effect free so they can be unit tested in
isolation and reused by any streaming path.
"""

from __future__ import annotations

import re

__all__ = [
    "extract_llama_slot_id",
    "extract_proxy_session_id",
    "extract_proxy_slot_id",
    "line_matches_slot",
    "filter_log_lines_for_slot",
]

# Slot id marker in llama-server.log lines: `id <N> |` or `id=<N>`.
_LLAMA_SLOT_ID_RE = re.compile(r"\bid\s*[= ]\s*(\d+)")
# Some llama-server builds also write `slot <N> | ...` without an id keyword.
_LLAMA_BARE_SLOT_RE = re.compile(r"\bslot\s+(\d+)\b")

# proxy.log markers.
_PROXY_SESSION_RE = re.compile(r"\bsession=([0-9a-fA-F-]{8,})")
_PROXY_SLOT_RE = re.compile(r"\bslot=(\d+)")

# Shape used to detect llama-server lines for auto source detection:
# `slot <word>: ...` prefix (update_slots / print_timing / launch_slot_ / release).
_LLAMA_LINE_SHAPE_RE = re.compile(r"\bslot\s+[a-z_]+\s*:", flags=re.IGNORECASE)


def extract_llama_slot_id(line: str) -> int | None:
    """Extract the llama-server slot id from a slot log line.

    Returns the slot id for lines like
    ``slot update_slots: id  2 | task ...`` or ``slot update_slots: id=5 ...``,
    otherwise ``None``.
    """
    if not isinstance(line, str):
        return None
    m = _LLAMA_SLOT_ID_RE.search(line)
    if m:
        return int(m.group(1))
    m = _LLAMA_BARE_SLOT_RE.search(line)
    if m:
        return int(m.group(1))
    return None


def extract_proxy_session_id(line: str) -> str | None:
    """Extract the ``session=<uuid>`` value from a proxy.log line, or ``None``."""
    if not isinstance(line, str):
        return None
    m = _PROXY_SESSION_RE.search(line)
    return m.group(1) if m else None


def extract_proxy_slot_id(line: str) -> int | None:
    """Extract the ``slot=<n>`` marker from a proxy.log line, or ``None``.

    ``slot=none`` (session not yet assigned to a persistence slot) is not a
    numeric marker and yields ``None``.
    """
    if not isinstance(line, str):
        return None
    m = _PROXY_SLOT_RE.search(line)
    return int(m.group(1)) if m else None


def _detect_source(line: str) -> str | None:
    """Best-effort source detection for a log line.

    Returns ``"llama"`` for llama-server slot-shaped lines, ``"proxy"`` for
    lines carrying proxy markers, and ``None`` for unattributable lines.
    """
    if _LLAMA_LINE_SHAPE_RE.search(line):
        return "llama"
    if _PROXY_SESSION_RE.search(line) or _PROXY_SLOT_RE.search(line) or "slot=none" in line:
        return "proxy"
    return None


def line_matches_slot(
    line: str,
    slot_id: int,
    session_id: str | None = None,
    source: str | None = None,
) -> bool:
    """Return ``True`` when *line* is relevant to the given llama-server slot.

    Args:
        line: A single log line (proxy.log or llama-server.log).
        slot_id: The llama-server slot id (``slot_id`` from ``/slots``).
        session_id: The dispatch session currently mapped to the slot
            (from the slot→session map).  Used to attribute proxy.log lines
            via their ``session=<uuid>`` marker.
        source: Explicit source: ``"llama"`` or ``"proxy"``.  When ``None``
            (or unknown) the source is auto-detected from the line shape.

    Matching rules:

    - llama: the line's ``id <N>`` / ``id=<N>`` (or bare ``slot <N>``)
      marker must equal *slot_id*; non-slot lines never match.
    - proxy: the line matches if its ``session=<uuid>`` equals *session_id*
      (when provided); otherwise the ``slot=<n>`` marker must equal
      *slot_id*.  ``slot=none`` / marker-less lines never match.
    """
    if not isinstance(line, str) or not line.strip():
        return False

    src = source if source in ("llama", "proxy") else _detect_source(line)
    if src == "llama":
        sid = extract_llama_slot_id(line)
        return sid is not None and sid == slot_id

    if src == "proxy":
        if session_id:
            sid = extract_proxy_session_id(line)
            if sid is not None and sid == session_id:
                return True
        pid = extract_proxy_slot_id(line)
        if pid is not None and pid == slot_id:
            return True
        return False

    return False


def filter_log_lines_for_slot(
    lines,
    slot_id: int,
    session_id: str | None = None,
    source: str | None = None,
) -> list[str]:
    """Return only the lines from *lines* relevant to the given slot.

    Args:
        lines: Iterable of log lines (a bare string is treated as one line).
        slot_id: The llama-server slot id.
        session_id: Optional mapped dispatch session for proxy.log attribution.
        source: Optional explicit source (``"llama"`` / ``"proxy"``); when
            ``None`` each line's source is auto-detected.

    Returns a new list containing the matching lines in original order.
    """
    if isinstance(lines, str):
        lines = [lines]
    return [
        line
        for line in lines
        if line_matches_slot(line, slot_id, session_id=session_id, source=source)
    ]
