#!/usr/bin/env python3
"""Quantify the ``local_active_queries`` counter leak from proxy logs.

LP-0MVYWKG0T0081EGC (parent LP-0MVWF78QK0060TZH, AC1). The proxy's
``local_active_queries`` counter drifts upward and is repaired by the
self-healing clamp in ``_recover_stuck_local_active_queries``, which logs

    local_active_queries counter recovered: reset from N to M (M active dispatch records)

at WARNING. The item's original baseline (136 warnings / 359 local streams for
2026-10-31 00:00-13:05) was measured by hand and appears to have mixed in the
previous day's rotated file. This tool replaces that measurement with a
deterministic one so the post-fix target (AC6) compares against a reproducible
figure:

* (a) total ``local_active_queries counter recovered`` warnings in the window,
* (b) the ``reset from N to M`` delta distribution,
* (c) the leak ratio = warnings / local ``Stream finished`` events, and
* (d) every log file used, with its own first/last timestamp and per-file
      counts, so a rotated file counted twice is visible rather than assumed.

Log files are discovered by ``scripts/lib/proxy_logs.py`` (dot
``proxy.log.YYYY-MM-DD_HH`` and dash ``proxy.log-YYYY-MM-DD_HH`` rotation
schemes, plain or ``.gz``) and each file is read exactly once. A line that is
byte-identical in two files (logrotate ``copytruncate`` duplicates a tail) is
counted once and reported as a duplicate, so double-counting across rotated
files is impossible.

Window bounds are compared against the log lines' own wall-clock timestamps:
the proxy log carries no UTC offset, so pass naive timestamps, exactly as they
appear in the log.

A "local stream" is a ``Stream finished`` line whose ``provider=`` value is the
local pool (``local``, or any ``local-*`` variant). Remote providers (e.g.
``provider=OpenCode``) are excluded from the denominator.

Usage:
  ./scripts/analyze_local_active_queries_leak.py \\
      --window '2026-10-31T00:00:00,2026-10-31T13:14:00' --window-closed
  ./scripts/analyze_local_active_queries_leak.py \\
      --window '2026-11-01T05:00:00,2026-11-02T06:49:00' --window-closed --json
  ./scripts/analyze_local_active_queries_leak.py \\
      --files proxy.log.2026-10-31_003749.gz,proxy.log.2026-10-31_00.gz \\
      --window '2026-10-31T00:00:00,2026-10-31T13:14:00'

Exit codes:
  0 - success and the requested window is covered (a window with zero warnings
      is a result, not an error)
  1 - log directory missing/unreadable or an unexpected read error
  2 - invalid arguments (bad window, unknown explicit file)
  3 - coverage gap: the retained files do not cover the requested window, so
      the measured ratio would describe a shorter window than it claims
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lib.proxy_logs import (  # noqa: E402
    discover_proxy_log_files,
    open_proxy_log_text,
    parse_line_timestamp,
)

DEFAULT_LOG_DIR = Path("/var/log/llama-proxy")
#: Two adjacent files further apart than this are reported as a coverage gap.
DEFAULT_MAX_GAP_SECONDS = 3600.0

# 2026-10-31 00:42:39,487 - WARNING - local_active_queries counter recovered: reset from 3 to 2 (2 active dispatch records)
# Post-LP-0MVYLHNI000AQ7X the line may carry identifying fields after the
# parenthesis, so only the reset pair is matched.
_RECOVERY_RE = re.compile(
    r"local_active_queries counter recovered: reset from (\d+) to (\d+)"
)
# 2026-10-31 00:42:05,127 - INFO - Stream finished: reason=tool_calls tokens=... session=... provider=local model=Qwen3 ...
_STREAM_FINISHED_RE = re.compile(r"Stream finished:.*?\bprovider=(\S+)")


@dataclass
class FileTally:
    """Per-file contribution to the report (AC1 requirement d)."""

    path: Path
    #: First/last timestamp anywhere in the file, used for the retention verdict.
    scan_first: dt.datetime | None = None
    scan_last: dt.datetime | None = None
    #: First/last timestamp of the lines that fall inside the window.
    first: dt.datetime | None = None
    last: dt.datetime | None = None
    lines: int = 0
    recovery_warnings: int = 0
    local_streams: int = 0
    duplicates_dropped: int = 0


def parse_window(spec: str) -> tuple[dt.datetime, dt.datetime]:
    """Parse ``start,end`` (ISO-8601, log wall clock) into two naive datetimes."""
    try:
        start_text, end_text = spec.split(",", 1)
    except ValueError as exc:
        raise ValueError(
            "window must be 'start,end' (e.g. "
            "'2026-10-31T00:00:00,2026-10-31T13:14:00')"
        ) from exc
    bounds: list[dt.datetime] = []
    for raw in (start_text, end_text):
        text = raw.strip()
        try:
            parsed = dt.datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"invalid window timestamp {text!r}: {exc}") from exc
        if parsed.tzinfo is not None:
            raise ValueError(
                f"window timestamp {text!r} carries a UTC offset, but the proxy "
                "log has none; pass the log's own wall clock (naive)"
            )
        bounds.append(parsed)
    start, end = bounds
    if start >= end:
        raise ValueError("window start must be before window end")
    return start, end


def resolve_files(log_dir: Path, names: list[str]) -> list[Path]:
    """Resolve an explicit ``--files`` list against *log_dir* (absolute wins)."""
    paths: list[Path] = []
    for name in names:
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = Path(log_dir) / candidate
        if not candidate.is_file():
            raise FileNotFoundError(f"explicit log file not found: {candidate}")
        paths.append(candidate)
    return paths


def _first_timestamp(path: Path) -> dt.datetime | None:
    """The first parseable timestamp in *path* (``None`` when it has none)."""
    try:
        with open_proxy_log_text(path) as handle:
            for line in handle:
                stamp = parse_line_timestamp(line)
                if stamp is not None:
                    return stamp
    except OSError as exc:
        print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
    return None


def _is_cross_file_duplicate(
    seen: dict[tuple[str, str], str], key: tuple[str, str], path: Path
) -> bool:
    """True when *key* was already counted from a **different** file.

    Within one file identical lines are kept (they may be genuinely repeated
    events); only a copy carried over by rotation is dropped.
    """
    previous = seen.get(key)
    if previous is not None and previous != path.name:
        return True
    seen[key] = previous or path.name
    return False


def analyse(
    log_dir: Path,
    start: dt.datetime,
    end: dt.datetime,
    *,
    files: list[str] | None = None,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    window_closed: bool = False,
) -> dict:
    """Return the leak report for the ``[start, end]`` window."""
    log_dir = Path(log_dir)
    paths = (
        resolve_files(log_dir, files)
        if files
        else discover_proxy_log_files(log_dir)
    )

    tallies: list[FileTally] = []
    #: Files whose first line is already past the window (append-ordered, so
    #: they cannot contribute); recorded so the file set stays explicit.
    skipped_after_window: list[tuple[str, dt.datetime]] = []
    seen: dict[tuple[str, str], str] = {}
    deltas: Counter[str] = Counter()

    for path in paths:
        first_ts = _first_timestamp(path)
        if first_ts is not None and first_ts > end:
            skipped_after_window.append((path.name, first_ts))
            continue
        tally = FileTally(path=path)
        try:
            with open_proxy_log_text(path) as handle:
                for line in handle:
                    stamp = parse_line_timestamp(line)
                    if stamp is None:
                        continue
                    if tally.scan_first is None or stamp < tally.scan_first:
                        tally.scan_first = stamp
                    if tally.scan_last is None or stamp > tally.scan_last:
                        tally.scan_last = stamp
                    if stamp < start or stamp > end:
                        continue
                    tally.lines += 1
                    if tally.first is None or stamp < tally.first:
                        tally.first = stamp
                    if tally.last is None or stamp > tally.last:
                        tally.last = stamp
                    if (match := _RECOVERY_RE.search(line)) is not None:
                        if _is_cross_file_duplicate(
                            seen, ("recovery", line.strip()), path
                        ):
                            tally.duplicates_dropped += 1
                            continue
                        tally.recovery_warnings += 1
                        deltas[f"{match.group(1)}\u2192{match.group(2)}"] += 1
                        continue
                    if (match := _STREAM_FINISHED_RE.search(line)) is not None:
                        if not match.group(1).startswith("local"):
                            continue
                        if _is_cross_file_duplicate(
                            seen, ("stream", line.strip()), path
                        ):
                            tally.duplicates_dropped += 1
                            continue
                        tally.local_streams += 1
        except OSError as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
            continue
        tallies.append(tally)

    used = [tally for tally in tallies if tally.lines]
    warnings = sum(tally.recovery_warnings for tally in used)
    streams = sum(tally.local_streams for tally in used)
    duplicates = sum(tally.duplicates_dropped for tally in used)
    hours = (end - start).total_seconds() / 3600.0

    return {
        "window": {
            "start": start.isoformat(sep=" "),
            "end": end.isoformat(sep=" "),
            "hours": round(hours, 2),
            "closed": window_closed,
        },
        "recovery_warnings": {
            "total": warnings,
            "per_hour": round(warnings / hours, 2) if hours else None,
        },
        "delta_distribution": _sorted_counts(deltas),
        "local_streams": {
            "total": streams,
            "per_hour": round(streams / hours, 2) if hours else None,
        },
        "leak_ratio": {
            "warnings_per_local_stream": (
                round(warnings / streams, 3) if streams else None
            ),
            "local_streams_per_warning": (
                round(streams / warnings, 2) if warnings else None
            ),
        },
        "files": [_file_report(tally) for tally in used],
        "files_scanned": len(paths),
        "files_skipped_after_window": [
            name for name, _stamp in skipped_after_window
        ],
        "duplicates_dropped": duplicates,
        "coverage": _coverage(
            tallies,
            skipped_after_window,
            start,
            end,
            max_gap_seconds=max_gap_seconds,
            window_closed=window_closed,
        ),
    }


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    """Counts ordered by size then key, so the dominant delta reads first."""
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _file_report(tally: FileTally) -> dict:
    """The explicit per-file row AC1(d) asks for."""
    return {
        "file": tally.path.name,
        "first": tally.first.isoformat(sep=" ") if tally.first else None,
        "last": tally.last.isoformat(sep=" ") if tally.last else None,
        "lines": tally.lines,
        "recovery_warnings": tally.recovery_warnings,
        "local_streams": tally.local_streams,
        "duplicates_dropped": tally.duplicates_dropped,
    }


def _coverage(
    tallies: list[FileTally],
    skipped_after_window: list[tuple[str, dt.datetime]],
    start: dt.datetime,
    end: dt.datetime,
    *,
    max_gap_seconds: float,
    window_closed: bool,
) -> dict:
    """Verdict on whether the retained files actually cover the window.

    Without it a missing rotation would read as a lower leak rate instead of
    an unmeasurable window (the 2026-10-23 18:29 -> 22:04 hole shape that
    ``scripts/lib/proxy_logs.py`` guards against).
    """
    spans = sorted(
        (tally for tally in tallies if tally.first is not None),
        key=lambda tally: (tally.first, tally.last, tally.path.name),
    )
    gaps: list[dict] = []
    for previous, following in zip(spans, spans[1:]):
        gap_seconds = (following.first - previous.last).total_seconds()
        if gap_seconds > max_gap_seconds:
            gaps.append(
                {
                    "kind": "between_files",
                    "before": previous.path.name,
                    "before_last": previous.last.isoformat(sep=" "),
                    "after": following.path.name,
                    "after_first": following.first.isoformat(sep=" "),
                    "gap_seconds": gap_seconds,
                }
            )

    head_gap = (spans[0].first - start).total_seconds() if spans else None
    if head_gap is not None and head_gap > max_gap_seconds:
        gaps.insert(
            0,
            {
                "kind": "window_head",
                "before": None,
                "before_last": start.isoformat(sep=" "),
                "after": spans[0].path.name,
                "after_first": spans[0].first.isoformat(sep=" "),
                "gap_seconds": head_gap,
            },
        )
    tail_gap = (end - spans[-1].last).total_seconds() if spans else None
    if window_closed and tail_gap is not None and tail_gap > max_gap_seconds:
        gaps.append(
            {
                "kind": "window_tail",
                "before": spans[-1].path.name,
                "before_last": spans[-1].last.isoformat(sep=" "),
                "after": None,
                "after_first": end.isoformat(sep=" "),
                "gap_seconds": tail_gap,
            }
        )

    retention_first = _min_stamp(
        [tally.scan_first for tally in tallies]
        + [stamp for _name, stamp in skipped_after_window]
    )
    retention_last = _max_stamp(
        [tally.scan_last for tally in tallies]
        + [stamp for _name, stamp in skipped_after_window]
    )
    covered = bool(spans) and not gaps
    if not spans:
        # Distinguish "the retained logs never reach this window" (a gap) from
        # a quiet window the retention does span (an empty result).
        if retention_first is None or retention_last is None:
            kind = "no_log_files"
        elif end < retention_first or start > retention_last:
            kind = "outside_retention"
        else:
            kind = None
        if kind is not None:
            gaps.append(
                {
                    "kind": kind,
                    "before": None,
                    "before_last": start.isoformat(sep=" "),
                    "after": None,
                    "after_first": end.isoformat(sep=" "),
                    "gap_seconds": (end - start).total_seconds(),
                }
            )
        else:
            covered = True

    return {
        "covered": covered,
        "max_gap_seconds": max_gap_seconds,
        "window_closed": window_closed,
        "retention_first": (
            retention_first.isoformat(sep=" ") if retention_first else None
        ),
        "retention_last": (
            retention_last.isoformat(sep=" ") if retention_last else None
        ),
        "spans": [
            {
                "file": tally.path.name,
                "first": tally.first.isoformat(sep=" "),
                "last": tally.last.isoformat(sep=" "),
                "lines": tally.lines,
            }
            for tally in spans
        ],
        "gaps": gaps,
        "window_start_gap_seconds": head_gap,
        "window_end_gap_seconds": tail_gap,
    }


def _min_stamp(stamps: list[dt.datetime | None]) -> dt.datetime | None:
    present = [stamp for stamp in stamps if stamp is not None]
    return min(present) if present else None


def _max_stamp(stamps: list[dt.datetime | None]) -> dt.datetime | None:
    present = [stamp for stamp in stamps if stamp is not None]
    return max(present) if present else None


def render(report: dict) -> str:
    """Render the report as a human-readable summary."""
    ratio = report["leak_ratio"]
    coverage = report["coverage"]
    lines = [
        f"window: {report['window']['start']} .. {report['window']['end']} "
        f"({report['window']['hours']} h, closed={report['window']['closed']})",
        f"coverage: {'covered' if coverage['covered'] else 'COVERAGE GAP'} "
        f"(max gap {coverage['max_gap_seconds']:.0f} s, "
        f"retention {coverage['retention_first']} .. {coverage['retention_last']})",
        f"recovery warnings: {report['recovery_warnings']['total']} "
        f"({report['recovery_warnings']['per_hour']}/h)",
        f"delta distribution: {report['delta_distribution']}",
        f"local streams: {report['local_streams']['total']} "
        f"({report['local_streams']['per_hour']}/h)",
        f"leak ratio: {ratio['warnings_per_local_stream']} warnings per local "
        f"stream (1 per {ratio['local_streams_per_warning']} local streams)",
        f"files used ({len(report['files'])} of {report['files_scanned']} "
        f"scanned; {report['duplicates_dropped']} duplicate lines dropped):",
    ]
    for row in report["files"]:
        lines.append(
            f"  {row['file']}  {row['first']} .. {row['last']}  "
            f"lines={row['lines']} warnings={row['recovery_warnings']} "
            f"local_streams={row['local_streams']}"
            + (
                f" duplicates={row['duplicates_dropped']}"
                if row["duplicates_dropped"]
                else ""
            )
        )
    if report["files_skipped_after_window"]:
        lines.append(
            "files after the window (not read): "
            + ", ".join(report["files_skipped_after_window"])
        )
    for gap in coverage["gaps"]:
        lines.append(
            f"  GAP {gap['kind']}: {gap['before']} "
            f"({gap['before_last']}) -> {gap['after']} "
            f"({gap['after_first']}) = {gap['gap_seconds']:.0f} s"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help=f"proxy log directory (default {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--window",
        required=True,
        help="start,end in the log's wall clock (ISO-8601, no UTC offset)",
    )
    parser.add_argument(
        "--files",
        default=None,
        help="comma-separated explicit log files (relative to --log-dir)",
    )
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=DEFAULT_MAX_GAP_SECONDS,
        help=f"adjacent-file gap that fails the coverage verdict "
        f"(default {DEFAULT_MAX_GAP_SECONDS:.0f} s)",
    )
    parser.add_argument(
        "--window-closed",
        action="store_true",
        help="the window has fully elapsed, so a long tail hole is a gap too",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"error: log directory not found: {log_dir}", file=sys.stderr)
        return 1
    try:
        start, end = parse_window(args.window)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    files = [name for name in (args.files or "").split(",") if name.strip()]
    try:
        report = analyse(
            log_dir,
            start,
            end,
            files=files or None,
            max_gap_seconds=args.max_gap_seconds,
            window_closed=args.window_closed,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2) if args.json else render(report))
    if not report["coverage"]["covered"]:
        print(
            "error: coverage gap — the retained logs do not cover the requested "
            "window, so the measured ratio would describe a shorter window than "
            "it claims (see coverage.gaps)",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
