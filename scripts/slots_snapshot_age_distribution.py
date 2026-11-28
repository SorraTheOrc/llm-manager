#!/usr/bin/env python3
"""Snapshot-age distribution for ``/llama/local/status`` polls.

The reproducible **tightened reading** of a ``slots_stale`` window
(LP-0MWLZ68SF0052E7Z, F6 / LP-0MVVCO1N7004TNGY AC2, plan question Q3).

``scripts/capacity_signal_verification.py`` judges ``slots_stale`` at the
**delivered** ceiling — ``slots_snapshot_age_seconds > slots_max_snapshot_age_seconds``
(300 s by default, ``observability.local_slots_max_snapshot_age_seconds``:
``max(120, 10 × 30 s poll cadence)``) — and counts a poll that cannot date its
snapshot as stale. That is the F4 contract (``status_schema_version: 8``), and
it answers "did the proxy serve a snapshot it considers unusably old?".

It does **not** answer the plan's Q3 question, "what would ``slots_stale`` be
if the ceiling were 2 s (two 1 s refresher cycles)?", nor does it publish the
snapshot-age distribution. Without that second reading a `< 5 %` verdict can
pass on a ceiling so loose that a poll is served a minutes-old snapshot and
still counts fresh. This tool reports both halves of the same poll population:

* the **as-configured** reading, re-derived here so the two numbers cannot drift
  (``over_ceiling`` + ``undatable`` == the tool's contract ``slots_stale``);
* **any number of tighter thresholds** (default 2 s), each with the count, the
  rate over the *datable* polls and over *all* status polls, and the
  ``capacity_hint`` split of the polls that cross it — the "how much of it sits
  in the prefill-blocked backend regime" figure.

Usage::

  ./scripts/slots_snapshot_age_distribution.py \\
      --since '2026-11-18 10:25:00' --until '2026-11-19 10:25:00' --json
  ./scripts/slots_snapshot_age_distribution.py --since '...' --until '...' \\
      --threshold-seconds 2 --threshold-seconds 300 --json

Window bounds are **local wall-clock** (the proxy log's own timestamps, and the
same convention as ``capacity_signal_verification.py``): ``--since`` inclusive,
``--until`` exclusive. Discovery and per-line filtering go through
``scripts/lib/proxy_logs.py``, so a rotated file's name never bounds its
content and both rotation naming schemes (dot/dash, plain/``.gz``) are read.

Exit codes: 0 — report written; 1 — no log directory / no proxy logs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lib.capacity_signals import (  # noqa: E402
    in_window,
    is_status_poll,
    percentile,
    share,
    slots_age_fields,
)
from lib.proxy_logs import (  # noqa: E402
    discover_proxy_log_files,
    open_proxy_log_text,
    parse_line_timestamp,
)

TOOL = "scripts/slots_snapshot_age_distribution.py"
DEFAULT_LOG_DIR = Path("/var/log/llama-proxy")
DEFAULT_THRESHOLD_SECONDS = 2.0
UNKNOWN_HINT = "unknown"

_HINT_RE = re.compile(r"\bcapacity_hint=(\S+)")
_BOUND_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")


def parse_bound(value: str | None) -> dt.datetime | None:
    """Parse a ``YYYY-MM-DD [HH:MM[:SS]]`` local wall-clock bound."""
    if not value:
        return None
    for fmt in _BOUND_FORMATS:
        try:
            return dt.datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"could not parse {value!r}; use 'YYYY-MM-DD [HH:MM[:SS]]'")


def _parse_age(value: str | None) -> float | None:
    """Parse an age field; ``None`` for the literal ``None``/NaN/unparseable."""
    if value is None or value == "None":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN
        return None
    return parsed


def collect(
    log_dir: Path,
    since: dt.datetime | None,
    until: dt.datetime | None,
) -> tuple[list[dict], Counter]:
    """Return ``(polls, ceilings)`` for the window.

    Each poll is ``{"age": float | None, "has_age_field": bool, "hint": str}``:
    ``age`` is ``None`` for an undatable snapshot (the literal ``None``) and for
    a pre-F1 line that carries no age field at all, which ``has_age_field``
    distinguishes. ``ceilings`` counts the published
    ``slots_max_snapshot_age_seconds`` values.
    """
    polls: list[dict] = []
    ceilings: Counter = Counter()
    for path in discover_proxy_log_files(log_dir):
        with open_proxy_log_text(path) as handle:
            for line in handle:
                if not is_status_poll(line):
                    continue
                if not in_window(parse_line_timestamp(line), since, until):
                    continue
                fields = slots_age_fields(line)
                hint_match = _HINT_RE.search(line)
                polls.append(
                    {
                        "age": _parse_age(fields.get("slots_snapshot_age_seconds")),
                        "has_age_field": "slots_snapshot_age_seconds" in fields,
                        "hint": hint_match.group(1) if hint_match else UNKNOWN_HINT,
                    }
                )
                ceiling = _parse_age(fields.get("slots_max_snapshot_age_seconds"))
                if ceiling is not None:
                    ceilings[ceiling] += 1
    return polls, ceilings


def summarise(polls: list[dict], ceilings: Counter, thresholds: list[float]) -> dict:
    """Build the report payload from the collected poll sample."""
    datable = sorted(p["age"] for p in polls if p["age"] is not None)
    undatable = sum(1 for p in polls if p["has_age_field"] and p["age"] is None)
    no_age_field = sum(1 for p in polls if not p["has_age_field"])
    status_polls = len(polls)
    ceiling = ceilings.most_common(1)[0][0] if ceilings else None
    over_ceiling = sum(1 for age in datable if ceiling is not None and age > ceiling)

    readings = []
    for threshold in sorted(set(thresholds)):
        over = [p for p in polls if p["age"] is not None and p["age"] > threshold]
        split: Counter = Counter(p["hint"] for p in over)
        readings.append(
            {
                "threshold_seconds": threshold,
                "over": len(over),
                "rate_of_datable": share(len(over), len(datable)),
                "rate_of_status_polls": share(len(over), status_polls),
                "by_capacity_hint": dict(sorted(split.items())),
            }
        )

    return {
        "tool": TOOL,
        "generated_at": dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "polls": {
            "status_polls": status_polls,
            "polls_with_age_field": len(datable) + undatable,
            "datable": len(datable),
            "undatable": undatable,
            "no_age_field": no_age_field,
        },
        "ceiling_seconds": ceiling,
        "age_seconds": {
            "count": len(datable),
            "min": datable[0] if datable else None,
            "p50": percentile(datable, 50),
            "p90": percentile(datable, 90),
            "p99": percentile(datable, 99),
            "max": datable[-1] if datable else None,
        },
        "contract_stale": {
            # age > ceiling **or** no datable snapshot (the F4 definition)
            "count": over_ceiling + undatable,
            "rate_of_status_polls": share(over_ceiling + undatable, status_polls),
        },
        "capacity_hint_counts": dict(sorted(Counter(p["hint"] for p in polls).items())),
        "thresholds": readings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--since", default=None, help="inclusive lower bound (local)")
    parser.add_argument("--until", default=None, help="exclusive upper bound (local)")
    parser.add_argument(
        "--threshold-seconds",
        type=float,
        action="append",
        default=None,
        help=(f"age threshold to report (repeatable; default {DEFAULT_THRESHOLD_SECONDS:g})"),
    )
    parser.add_argument("--json", action="store_true", help="machine-readable JSON")
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"error: log directory not found: {log_dir}", file=sys.stderr)
        return 1
    if not discover_proxy_log_files(log_dir):
        print(f"error: no proxy logs found in {log_dir}", file=sys.stderr)
        return 1
    try:
        since = parse_bound(args.since)
        until = parse_bound(args.until)
    except argparse.ArgumentTypeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    polls, ceilings = collect(log_dir, since, until)
    thresholds = args.threshold_seconds or [DEFAULT_THRESHOLD_SECONDS]
    report = summarise(polls, ceilings, thresholds)
    report["log_dir"] = str(log_dir)
    report["window"] = {
        "since": since.strftime("%Y-%m-%d %H:%M:%S") if since else None,
        "until": until.strftime("%Y-%m-%d %H:%M:%S") if until else None,
        "half_open": True,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
