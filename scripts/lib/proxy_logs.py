"""Shared proxy-log discovery, gzip-aware opening and coverage measurement.

The deployment writes the proxy log stream under **two** rotation schemes, so
callers must not assume a single naming pattern:

- in-process ``TimedRotatingFileHandler`` → ``proxy.log.YYYY-MM-DD_HH`` (dot)
- logrotate safety net (``dateformat -%Y-%m-%d_%H``) → ``proxy.log-YYYY-MM-DD_HH``
  (dash)

Both may be gzip-compressed (``.gz``). Before this module each analysis script
carried its own copy of the discovery logic; the copies drifted and silently
dropped dash-named files (which carry the daytime window) and compressed files.
Consolidating them here means a naming/compression change is fixed once.

This module is dependency-light on purpose: standard library only, so it can be
imported from both ``proxy/scripts/*`` and repo-root ``scripts/*`` without
pulling in the heavy ``proxy.proxy.utils`` dependencies. It intentionally does
**not** import the ``proxy-usage-analysis`` skill's ``log_parser`` (agent
infrastructure under ``.pi/``); that module is the behavioural reference only.

No filename- or mtime-based time filtering is performed here: rotated files
routinely hold data past their encoded rotation time, so callers keep per-line
timestamp filtering as the authoritative window boundary.

:func:`measure_log_coverage` turns the same discovery into a verdict: it
reports an adjacent-file gap (the 2026-10-23 18:29:12 -> 22:04:16 hole shape)
when two consecutive files are further apart than a bound, so a verification
window with a missing interval fails instead of reading as a partial success
(LP-0MVLHMAWV008EKTI).

Parent: LP-0MU148SHI004WHQM (child LP-0MUIEK483002TZ54).
"""

from __future__ import annotations

import gzip
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

#: Live, currently-written log.
_LIVE_LOG_NAME = "proxy.log"
#: In-process ``TimedRotatingFileHandler`` rotated files.
_DOT_PREFIX = "proxy.log."
#: Logrotate safety-net rotated files (``dateformat -%Y-%m-%d_%H``).
_DASH_PREFIX = "proxy.log-"


def discover_proxy_log_files(log_dir: Path) -> list[Path]:
    """Return every proxy log file in *log_dir*, sorted by name.

    Includes the live ``proxy.log`` and both rotated naming schemes
    (``proxy.log.*`` and ``proxy.log-*``), whether plain or ``.gz``. Directories
    whose names happen to match a prefix are ignored. Returns an empty list when
    *log_dir* is not a directory (missing, or a regular file).

    No time-based filtering is applied — a rotated file's name-encoded time does
    not bound its content.
    """
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return []
    candidates = [
        path
        for path in log_dir.iterdir()
        if path.is_file()
        and (
            path.name == _LIVE_LOG_NAME
            or path.name.startswith(_DOT_PREFIX)
            or path.name.startswith(_DASH_PREFIX)
        )
    ]
    return sorted(candidates, key=lambda path: path.name)


def open_proxy_log_text(path: Path):
    """Open a proxy log for text reading, transparently decompressing ``.gz``.

    Rotated files may be gzip-compressed (logrotate ``compress``, or the
    in-process handler's ``.gz`` retention). Detecting the ``.gz`` suffix here
    keeps callers free of compression concerns; without it a compressed file is
    read as raw bytes, yields no parseable lines, and is silently counted as
    scanned. Plain files are read as UTF-8 with ``errors="replace"`` so a stray
    non-UTF-8 byte never aborts a log scan.
    """
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def iter_proxy_log_lines(log_dir: Path) -> Iterator[tuple[Path, str]]:
    """Yield ``(path, line)`` for every discovered proxy log, in name order.

    Pairs each line with its source file so callers can attribute findings.
    Files are streamed (never loaded whole). An unreadable file emits a warning
    to stderr and is skipped, so one bad file does not abort the scan — matching
    the failure behaviour the analysis scripts had before consolidation.
    """
    for path in discover_proxy_log_files(log_dir):
        try:
            with open_proxy_log_text(path) as fh:
                for line in fh:
                    yield path, line
        except OSError as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Coverage: does the discovered log set actually cover the requested window?
# ---------------------------------------------------------------------------

#: Leading timestamp of every proxy log line (``2026-10-23 18:29:12,229 - ...``).
_LINE_TS_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})")


@dataclass(frozen=True)
class FileSpan:
    """The window of lines one log file contributes (after window filtering)."""

    path: Path
    first: datetime
    last: datetime
    line_count: int


@dataclass(frozen=True)
class CoverageGap:
    """A hole between two adjacent files' windows (a lost/unmeasurable window)."""

    before: Path
    before_last: datetime
    after: Path
    after_first: datetime
    gap_seconds: float


@dataclass
class LogCoverage:
    """Result of :func:`measure_log_coverage`.

    ``gaps`` are **adjacent-file** gaps: a boundary between two files whose
    first/last lines are further apart than the threshold. Ordinary idle time
    *inside* one file is not a gap — the incident this guards against
    (2026-10-23: ``proxy.log.2026-10-23_12`` ended 18:29:12, the next file
    started 22:04:16) is exactly a missing file boundary.

    ``window_start_gap_seconds`` / ``window_end_gap_seconds`` are informational:
    the lead-in before the first retained line and the distance from the last
    line to the requested end (a live window has no end yet), so they do not
    fail the verdict.
    """

    spans: list[FileSpan] = field(default_factory=list)
    gaps: list[CoverageGap] = field(default_factory=list)
    max_gap_seconds: float = 3600.0
    window_start_gap_seconds: float | None = None
    window_end_gap_seconds: float | None = None

    @property
    def covered(self) -> bool:
        """True when at least one file matched and no adjacent-file gap exists."""
        return bool(self.spans) and not self.gaps


def parse_line_timestamp(line: str) -> datetime | None:
    """Return the timestamp at the start of a proxy log line, or ``None``."""
    match = _LINE_TS_RE.match(line)
    if not match:
        return None
    try:
        return datetime(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def measure_log_coverage(
    log_dir: Path,
    since: datetime | None = None,
    until: datetime | None = None,
    max_gap_seconds: float = 3600.0,
) -> LogCoverage:
    """Measure how completely the log files in *log_dir* cover a window.

    For every discovered file (both rotation schemes, plain or ``.gz``) the
    first/last timestamps of the lines inside ``[since, until]`` are collected;
    files are then ordered by their content and every adjacent boundary whose
    distance exceeds *max_gap_seconds* is reported as a :class:`CoverageGap`.

    This turns a silently partial read into an explicit verdict: without it
    ``--measure`` reported on whatever it happened to parse, so a 3 h 35 m hole
    looked like a normal window (LP-0MVLHMAWV008EKTI).

    Files with no lines inside the window are ignored (an archived file from a
    different day is not a boundary). Overlapping files (logrotate
    ``copytruncate`` duplicates a tail) never produce a gap.
    """
    log_dir = Path(log_dir)
    spans: list[FileSpan] = []
    for path in discover_proxy_log_files(log_dir):
        first: datetime | None = None
        last: datetime | None = None
        count = 0
        try:
            with open_proxy_log_text(path) as fh:
                for line in fh:
                    ts = parse_line_timestamp(line)
                    if ts is None:
                        continue
                    if since is not None and ts < since:
                        continue
                    if until is not None and ts > until:
                        continue
                    if first is None or ts < first:
                        first = ts
                    if last is None or ts > last:
                        last = ts
                    count += 1
        except OSError as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
            continue
        if first is not None and last is not None:
            spans.append(FileSpan(path=path, first=first, last=last, line_count=count))

    # Order by content, not by filename: the two naming schemes sort
    # ambiguously and a rotated file routinely holds data past its name.
    spans.sort(key=lambda span: (span.first, span.last, span.path.name))

    gaps: list[CoverageGap] = []
    for previous, following in zip(spans, spans[1:]):
        gap_seconds = (following.first - previous.last).total_seconds()
        if gap_seconds > max_gap_seconds:
            gaps.append(
                CoverageGap(
                    before=previous.path,
                    before_last=previous.last,
                    after=following.path,
                    after_first=following.first,
                    gap_seconds=gap_seconds,
                )
            )

    result = LogCoverage(
        spans=spans, gaps=gaps, max_gap_seconds=max_gap_seconds
    )
    if spans:
        if since is not None:
            result.window_start_gap_seconds = (
                spans[0].first - since
            ).total_seconds()
        if until is not None:
            result.window_end_gap_seconds = (
                until - max(span.last for span in spans)
            ).total_seconds()
    return result
