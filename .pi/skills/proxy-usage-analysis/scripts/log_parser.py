"""Streaming, line-based parser for llama-proxy structured INFO log lines.

The proxy logs lines of the form::

    %Y-%m-%d %H:%M:%S,%f - LEVEL - message

Only a small set of structured message prefixes carry the signal this skill
needs (``Stream started``, ``Stream finished``, ``Fallback triggered``,
``routing_skip_local``, ``local_dispatch_denied``). Everything else is
ignored. Parsing is tolerant: missing fields default to ``None`` and a line
that cannot be parsed is skipped (never fatal), so truncated request payloads
and log-format drift do not break the analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

# Log line prefix: "2026-08-02 13:58:32,260 - INFO - <message>"
LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) - (\w+) - (.*)$")

# Field extractors (tolerant: may match nothing).
RE_PROVIDER = re.compile(r"\bprovider=(\S+)")
RE_MODEL = re.compile(r"\bmodel=(\S+)")
RE_SESSION = re.compile(r"\bsession=([A-Za-z0-9_.-]+)")
RE_TOKENS = re.compile(r"\btokens=(\d+)/(\d+)/(\d+)")
RE_FALLBACK = re.compile(
    r"Fallback triggered for model=(\S+?), from=(\S+?), to=(\S+?), reason=([\w ]+)$"
)
# routing_skip_local reason sits between "reason=" and the "→" arrow.
RE_ROUTING_SKIP_REASON = re.compile(r"\breason=([\w_]+)\s*→")
RE_DISPATCH_DENIED = re.compile(r"session=([A-Za-z0-9_.-]+) owner=([A-Za-z0-9_.-]+) active=(\d+)")

# Rotated log naming: proxy.log.YYYY-MM-DD_HH (the timestamp is the rotation time).
ROTATED_NAME_RE = re.compile(r"^proxy\.log\.(\d{4})-(\d{2})-(\d{2})_(\d{2})$")

STREAM_STARTED = "Stream started"
STREAM_FINISHED = "Stream finished"
FALLBACK = "Fallback triggered"
ROUTING_SKIP = "routing_skip_local"
DISPATCH_DENIED = "local_dispatch_denied"

# Events within this many seconds *before* a session's first remote stream are
# candidates for attributing a session-less "Fallback triggered" line to that
# session.
FALLBACK_ATTRIBUTION_WINDOW_SECONDS = 60

LOCAL_PROVIDER = "local"


@dataclass
class LogEvent:
    """One parsed structured log line.

    ``kind`` is one of ``stream_started``, ``stream_finished``, ``fallback``,
    ``routing_skip``, ``dispatch_denied``. Only the fields relevant to each
    kind are populated.
    """

    kind: str
    ts: datetime
    provider: str | None = None
    model: str | None = None
    session: str | None = None
    reason: str | None = None
    prompt: int | None = None
    completion: int | None = None
    total: int | None = None
    src: str | None = None
    dst: str | None = None
    owner: str | None = None
    active: int | None = None


def _first(pattern: re.Pattern, text: str) -> str | None:
    m = pattern.search(text)
    return m.group(1) if m else None


def _session_from(text: str) -> str | None:
    token = _first(RE_SESSION, text)
    if token is None or token == "unknown":
        return None
    return token


def parse_log_line(line: str) -> LogEvent | None:
    """Parse one log line into a :class:`LogEvent`, or ``None`` if the line
    carries no relevant signal (or cannot be parsed).

    ``session=unknown`` is treated as unattributed (``session=None``).
    """
    if not line:
        return None
    m = LINE_RE.match(line)
    if m is None:
        return None
    date_part, ms_part, _level, msg = m.groups()
    try:
        ts = datetime.strptime(date_part, "%Y-%m-%d %H:%M:%S").replace(
            microsecond=int(ms_part) * 1000
        )
    except ValueError:
        return None

    if msg.startswith(STREAM_STARTED):
        return LogEvent(
            "stream_started",
            ts,
            provider=_first(RE_PROVIDER, msg),
            model=_first(RE_MODEL, msg),
            session=_session_from(msg),
        )
    if msg.startswith(STREAM_FINISHED):
        tokens = RE_TOKENS.search(msg)
        prompt = completion = total = None
        if tokens:
            prompt, completion, total = (int(v) for v in tokens.groups())
        return LogEvent(
            "stream_finished",
            ts,
            provider=_first(RE_PROVIDER, msg),
            model=_first(RE_MODEL, msg),
            session=_session_from(msg),
            reason=_first(RE_ROUTING_SKIP_REASON, msg) or _first(
                re.compile(r"\breason=(\w+)"), msg
            ),
            prompt=prompt,
            completion=completion,
            total=total,
        )
    if msg.startswith(FALLBACK):
        m2 = RE_FALLBACK.search(msg)
        if m2 is None:
            return None
        fmodel, src, dst, reason = m2.groups()
        return LogEvent("fallback", ts, reason=reason, src=src, dst=dst)
    if msg.startswith(ROUTING_SKIP):
        return LogEvent(
            "routing_skip",
            ts,
            session=_session_from(msg),
            reason=_first(RE_ROUTING_SKIP_REASON, msg),
        )
    if msg.startswith(DISPATCH_DENIED):
        m2 = RE_DISPATCH_DENIED.search(msg)
        if m2 is None:
            return None
        session, owner, active = m2.groups()
        return LogEvent(
            "dispatch_denied", ts, session=session, owner=owner, active=int(active)
        )
    return None


def iter_events(path: Path, window_start: datetime, window_end: datetime) -> Iterator[LogEvent]:
    """Stream-parse ``path`` line by line, yielding only events whose
    timestamp falls inside ``[window_start, window_end]``.

    The file is never loaded into memory: lines are read one at a time and a
    cheap prefix check skips non-log lines before regex parsing.
    """
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if len(line) < 24 or not line[:4].isdigit():
                continue
            ev = parse_log_line(line)
            if ev is None:
                continue
            if window_start <= ev.ts <= window_end:
                yield ev


def discover_log_files(log_dir: Path, window_start: datetime) -> list[Path]:
    """Return the log files in ``log_dir`` that can overlap the analysis
    window, sorted by name.

    The live ``proxy.log`` is always included. Rotated files
    (``proxy.log.YYYY-MM-DD_HH``) end at their name-encoded rotation time, so
    a rotated file overlaps the window iff rotation_time >= window_start.
    Files with unrecognised names fall back to an mtime check (conservative;
    per-line timestamp filtering remains authoritative).
    """
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return []
    candidates: list[Path] = []
    for p in sorted(log_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name == "proxy.log":
            candidates.append(p)
            continue
        if not p.name.startswith("proxy.log."):
            continue
        m = ROTATED_NAME_RE.match(p.name)
        if m is not None:
            y, mo, d, h = (int(x) for x in m.groups())
            rotation = datetime(y, mo, d, h)
            if rotation >= window_start:
                candidates.append(p)
        else:
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
            except OSError:
                continue
            if mtime >= window_start:
                candidates.append(p)
    return sorted(candidates, key=lambda p: p.name)
