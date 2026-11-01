#!/usr/bin/env python3
"""Attribute ``slot_ratio_seed outcome=missing_sidecar`` lines to a write path.

LP-0MVVI8TEU006KTOZ. The 2026-10-29/30 post-fix window reported 605
``missing_sidecar`` seeds (snapshot present, no ``.ratio.json`` sidecar) and 0
``found``. To pin the writer side rather than guess, this script walks the
proxy logs for a window and, for every ``missing_sidecar`` seed, finds the most
recent successful slot save for that session and classifies it:

* ``eviction``  — ``slot_save session=P ... saved=True source=eviction``
* ``router``    — ``slot_save success session=P slot=N``
* ``no-save-in-window`` — no successful save for ``P`` inside the window

It also tallies the seed-outcome vocabulary, successful saves per path,
persistence skips (e.g. ``slot_busy``) and the per-session attribution. Log
files may be compressed (``.gz``); each file is read exactly once.

Sessions are logged truncated to 8 characters, so attribution is by that
prefix; collisions would only merge two sessions' saves, never invent one.

Usage:
  ./scripts/attribute_missing_sidecar.py \\
      --log-dir /var/log/llama-proxy \\
      --window '2026-10-29T12:58:32,2026-10-30T12:58:32'
  ./scripts/attribute_missing_sidecar.py --log-dir /var/log/llama-proxy \\
      --window '2026-10-29T12:58:32,2026-10-30T12:58:32' --json

Exit codes:
  0 - success (even when a window contains no samples)
  1 - log directory missing / unreadable / unexpected error
  2 - invalid arguments (bad window specification)
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) - ")
SEED_RE = re.compile(r"slot_ratio_seed outcome=(\w+) session=(\S+)")
EVICT_RE = re.compile(
    r"slot_save session=(\S+) slot=(\S+) reason=(\S+) saved=(True|False) source=eviction"
)
ROUTER_RE = re.compile(r"slot_save success session=(\S+) slot=(\S+)")
SKIP_RE = re.compile(
    r"slot persistence skipped session=(\S+) slot=\S+ reason=(\S+)"
)


def parse_ts(line: str) -> dt.datetime | None:
    """Return the line's timestamp (UTC) or ``None`` for an un-timestamped line."""
    match = LINE_RE.match(line)
    if not match:
        return None
    try:
        return dt.datetime.fromisoformat(
            f"{match.group(1)}.{match.group(2)}"
        ).replace(tzinfo=dt.UTC)
    except ValueError:
        return None


def parse_window(spec: str) -> tuple[dt.datetime, dt.datetime]:
    """Parse ``start,end`` (ISO-8601, ``Z``/offset optional) into two UTC times."""
    try:
        start_text, end_text = spec.split(",", 1)
    except ValueError as exc:
        raise ValueError(
            "window must be 'start,end' (e.g. '2026-10-29T12:58:32,2026-10-30T12:58:32')"
        ) from exc
    try:
        start = dt.datetime.fromisoformat(start_text.replace("Z", "+00:00"))
        end = dt.datetime.fromisoformat(end_text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid window timestamp: {exc}") from exc
    if start.tzinfo is None:
        start = start.replace(tzinfo=dt.UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=dt.UTC)
    if start >= end:
        raise ValueError("window start must be before window end")
    return start.astimezone(dt.UTC), end.astimezone(dt.UTC)


def iter_log_lines(log_dir: Path):
    """Yield every line from the proxy logs in *log_dir* (``.gz`` aware)."""
    for path in sorted(log_dir.glob("proxy.log*")):
        if not path.is_file():
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        try:
            with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
                yield from handle
        except OSError:
            continue


def analyse(log_dir: Path, start: dt.datetime, end: dt.datetime) -> dict:
    """Return the attribution report for ``[start, end]``."""
    seeds: list[tuple[dt.datetime, str, str]] = []
    saves: list[tuple[dt.datetime, str, str]] = []
    skips: Counter[str] = Counter()

    for line in iter_log_lines(log_dir):
        ts = parse_ts(line)
        if ts is None or ts < start or ts > end:
            continue
        if match := SEED_RE.search(line):
            seeds.append((ts, match.group(1), match.group(2)))
            continue
        if match := EVICT_RE.search(line):
            if match.group(4) == "True":
                saves.append((ts, match.group(1), "eviction"))
            else:
                skips["eviction_save_failed"] += 1
            continue
        if match := ROUTER_RE.search(line):
            saves.append((ts, match.group(1), "router"))
            continue
        if match := SKIP_RE.search(line):
            skips[match.group(2)] += 1

    saves_by_session: dict[str, list[tuple[dt.datetime, str]]] = defaultdict(list)
    for ts, session, kind in saves:
        saves_by_session[session].append((ts, kind))
    for entries in saves_by_session.values():
        entries.sort()

    attribution: Counter[str] = Counter()
    per_session: dict[str, Counter[str]] = defaultdict(Counter)
    for ts, outcome, session in seeds:
        if outcome != "missing_sidecar":
            continue
        prior = [kind for (save_ts, kind) in saves_by_session.get(session, []) if save_ts <= ts]
        kind = prior[-1] if prior else "no-save-in-window"
        attribution[kind] += 1
        per_session[session][kind] += 1

    return {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "seed_outcomes": dict(sorted(Counter(o for _, o, _ in seeds).items())),
        "successful_saves": dict(sorted(Counter(k for _, _, k in saves).items())),
        "persistence_skips": dict(sorted(skips.items())),
        "missing_sidecar_seeds": sum(attribution.values()),
        "missing_sidecar_attribution": dict(sorted(attribution.items())),
        "distinct_missing_sidecar_sessions": len(per_session),
        "sessions": {
            session: dict(sorted(counts.items()))
            for session, counts in sorted(
                per_session.items(), key=lambda kv: -sum(kv[1].values())
            )
        },
    }


def render_text(report: dict) -> str:
    """Render the report as a short human-readable summary."""
    lines = [
        f"window: {report['window']['start']} .. {report['window']['end']}",
        f"seed outcomes: {report['seed_outcomes']}",
        f"successful saves: {report['successful_saves']}",
        f"persistence skips: {report['persistence_skips']}",
        (
            "missing_sidecar attribution: "
            f"{report['missing_sidecar_attribution']} "
            f"(total {report['missing_sidecar_seeds']} across "
            f"{report['distinct_missing_sidecar_sessions']} sessions)"
        ),
        "per-session attribution:",
    ]
    for session, counts in report["sessions"].items():
        total = sum(counts.values())
        lines.append(f"  {session}: total={total} {counts}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log-dir", required=True, help="proxy log directory")
    parser.add_argument(
        "--window", required=True, help="start,end (ISO-8601; Z/offset optional)"
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

    report = analyse(log_dir, start, end)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
