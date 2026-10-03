"""Shared proxy-log discovery and gzip-aware opening.

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

Parent: LP-0MU148SHI004WHQM (child LP-0MUIEK483002TZ54).
"""

from __future__ import annotations

import gzip
import sys
from collections.abc import Iterator
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
