#!/usr/bin/env python3
"""Attribute slot-save failures and missing-sidecar seeds by save path.

LP-0MVVX6RM3001P4EY. Triage the 66-67% "router-path" slot-save failure rate
measured on the 2026-10-29→31 window. The prior measurement counted every
``slot_save failed slot=N …`` warning as a router failure, but that line
carries neither ``session=`` nor ``source=``: it is emitted by
``_call_slot_endpoint`` for BOTH the router save path and the detached
eviction save path, which logs its own paired ``slot_save session=…
saved=False source=eviction`` INFO line at the same timestamp. This script

* counts save attempts per path (router / eviction) with ok vs failed,
  pairing each ``slot_save failed`` warning with a same-slot eviction line
  within ±2 s (else it is a router failure);
* breaks failures down by reason (``error=ReadTimeout/ReadTimeout`` etc.),
  the applied timeout and the proxy busy-state snapshot at failure;
* tallies ``slot persistence skipped reason=…`` counts;
* states whether a save-duration distribution is available: the success path
  logs duration only at DEBUG (``slot_save ok slot=N elapsed=… timeout=…``)
  and the deployed level is INFO, so success durations are NOT logged —
  reported explicitly instead of estimated;
* attributes ``slot_ratio_seed outcome=missing_sidecar|missing_snapshot`` to
  the session's most recent save attempt, including failed attempts, so the
  seeds caused by a failed save (AC4) are quantified.

Log files may be compressed (``.gz``) under either rotation scheme (discovery
via ``scripts/lib/proxy_logs.py``); each file is read exactly once.

Usage:
  ./scripts/attribute_slot_save_failures.py \\
      --log-dir /var/log/llama-proxy \\
      --window '2026-10-29T12:58:32,2026-10-31T04:09:00'
  ./scripts/attribute_slot_save_failures.py --log-dir /var/log/llama-proxy \\
      --window '2026-10-29T12:58:32,2026-10-30T12:58:32' --json

Exit codes:
  0 - success (even when a window contains no samples)
  1 - log directory missing / unreadable / unexpected error
  2 - invalid arguments (bad window specification)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter
from pathlib import Path

from lib.proxy_logs import iter_proxy_log_lines

LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) - \w+ - "
)
# 2026-08-06 01:57:19,894 - WARNING - slot_save failed slot=2 error=ReadTimeout/ReadTimeout elapsed=3.0s timeout=3.0s busy={...}
# (optionally -- deployed logs after LP-0MVVX6RM3001P4EY -- ending in: source=eviction|router)
_FAIL_RE = re.compile(
    r"slot_save failed slot=(\d+) error=([^\s]+) elapsed=([\d.]+)s "
    r"timeout=([\d.]+)s busy=(\{.*\})( source=(\S+))?$"
)
# fallback for non-error failures: status=... body=...
_FAIL_STATUS_RE = re.compile(
    r"slot_save failed slot=(\d+) status=\S+ body=\S+ elapsed=([\d.]+)s "
    r"timeout=([\d.]+)s busy=(\{.*\})( source=(\S+))?$"
)
# 2026-08-06 01:55:47,533 - INFO - slot_save success session=019fd490 slot=0
_ROUTER_OK_RE = re.compile(r"slot_save success session=(\S+) slot=(\d+)")
# 2026-11-02 03:51:19,410 - INFO - slot_save session=herdr-17 slot=0 reason=lease_evicted saved=False source=eviction
_EVICTION_RE = re.compile(
    r"slot_save session=(\S+) slot=(\d+) reason=(\S+) saved=(True|False) source=eviction"
)
# 2026-11-02 03:58:40,626 - INFO - slot persistence skipped session=herdr-17 slot=0 reason=slot_busy active_sessions=1
_SKIP_RE = re.compile(r"slot persistence skipped session=\S+ slot=\S+ reason=(\S+)")
# 2026-10-30 10:00:00,000 - INFO - slot_ratio_seed outcome=missing_snapshot session=01a187be ...
_SEED_RE = re.compile(r"slot_ratio_seed outcome=(\w+) session=(\S+)")
# 2026-08-06 01:55:13,618 - DEBUG - slot_save ok slot=2 elapsed=1.2s timeout=3.0s
_SAVE_OK_DEBUG_RE = re.compile(r"slot_save ok slot=(\d+) elapsed=([\d.]+)s")


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


def _second_key(ts: dt.datetime) -> dt.datetime:
    """Bucket a timestamp to the whole second for failure/eviction pairing.

    The failure WARNING and the eviction INFO line share the same wall-clock
    second but differ in milliseconds (and the WARNING line may even precede
    the INFO line), so pairing uses second-granularity slots.
    """
    return dt.datetime.fromtimestamp(int(ts.timestamp()), tz=dt.UTC)


def analyse(log_dir: Path, start: dt.datetime, end: dt.datetime) -> dict:
    """Return the slot-save failure attribution report for ``[start, end]``."""
    # attempts: tuple(ts, path, ok, session|None, slot)
    attempts: list[tuple[dt.datetime, str, bool, str | None, int]] = []
    # eviction saved=False lines keyed by (second, slot); a ``slot_save
    # failed`` warning carrying no session/source is paired with one of these
    # so the failure can be attributed to the eviction path. Pairing happens
    # after the scan because the INFO line follows the WARNING in the file.
    eviction_fail_by_key: dict[tuple[dt.datetime, int], dict] = {}
    pending_failures: list[dict] = []
    skips: Counter[str] = Counter()
    seeds: list[tuple[dt.datetime, str, str]] = []
    success_duration_lines = 0

    for _path, line in iter_proxy_log_lines(log_dir):
        # Success-duration availability is window-independent: count the
        # DEBUG duration line anywhere in the logs so the report does not
        # claim a duration distribution exists for a window that happens to
        # lack DEBUG lines.
        if _SAVE_OK_DEBUG_RE.search(line):
            success_duration_lines += 1
        ts = parse_ts(line)
        if ts is None or ts < start or ts > end:
            continue
        if match := _EVICTION_RE.search(line):
            ok = match.group(4) == "True"
            attempts.append(
                (ts, "eviction", ok, match.group(1), int(match.group(2)))
            )
            if not ok:
                eviction_fail_by_key[(_second_key(ts), int(match.group(2)))] = {
                    "session": match.group(1),
                    "reason": match.group(3),
                }
            continue
        if match := _ROUTER_OK_RE.search(line):
            attempts.append(
                (ts, "router", True, match.group(1), int(match.group(2)))
            )
            continue
        if match := _FAIL_RE.search(line):
            pending_failures.append(
                {
                    "ts": ts,
                    "slot": int(match.group(1)),
                    "error": match.group(2),
                    "elapsed_s": float(match.group(3)),
                    "timeout_s": float(match.group(4)),
                    "busy": _parse_busy(match.group(5)),
                    "source_logged": match.group(7),
                }
            )
            continue
        if match := _FAIL_STATUS_RE.search(line):
            pending_failures.append(
                {
                    "ts": ts,
                    "slot": int(match.group(1)),
                    "error": "status",
                    "elapsed_s": float(match.group(2)),
                    "timeout_s": float(match.group(3)),
                    "busy": _parse_busy(match.group(4)),
                    "source_logged": match.group(6),
                }
            )
            continue
        if match := _SKIP_RE.search(line):
            skips[match.group(1)] += 1
            continue
        if match := _SEED_RE.search(line):
            seeds.append((ts, match.group(1), match.group(2)))
            continue

    failures: list[dict] = []
    for pending in pending_failures:
        ev = eviction_fail_by_key.pop(
            (_second_key(pending["ts"]), pending["slot"]), None
        )
        if pending["source_logged"] in ("eviction", "router"):
            # Deployed logs after the fix carry an explicit source= field;
            # prefer it over the timestamp-pairing inference.
            path = pending["source_logged"]
        else:
            path = "eviction" if ev is not None else "router"
        session = ev["session"] if ev else None
        if path == "router":
            # Eviction failed attempts are already recorded from their INFO
            # line; a router failure only exists as the WARNING line.
            attempts.append(
                (pending["ts"], "router", False, None, pending["slot"])
            )
        failures.append(
            {
                "ts": pending["ts"].strftime("%Y-%m-%d %H:%M:%S"),
                "path": path,
                "slot": pending["slot"],
                "session": session,
                "error": pending["error"],
                "elapsed_s": pending["elapsed_s"],
                "timeout_s": pending["timeout_s"],
                "busy": pending["busy"],
            }
        )

    router_ok = sum(1 for _t, p, ok, _s, _slot in attempts if p == "router" and ok)
    router_fail = sum(1 for _t, p, ok, _s, _slot in attempts if p == "router" and not ok)
    eviction_ok = sum(1 for _t, p, ok, _s, _slot in attempts if p == "eviction" and ok)
    eviction_fail = sum(1 for _t, p, ok, _s, _slot in attempts if p == "eviction" and not ok)

    by_session: dict[str, list[tuple[dt.datetime, str, bool]]] = {}
    for ts, path, ok, session, _slot in attempts:
        if not session:
            continue
        by_session.setdefault(session, []).append((ts, path, ok))
    for entries in by_session.values():
        entries.sort(key=lambda e: e[0])

    seed_attribution: dict[str, Counter[str]] = {}
    for ts, outcome, session in seeds:
        if outcome not in ("missing_sidecar", "missing_snapshot"):
            continue
        prior = [
            (path, ok)
            for (ats, path, ok) in by_session.get(session, [])
            if ats <= ts
        ]
        if prior:
            path, ok = prior[-1]
            kind = path if ok else f"{path}-failed"
        else:
            kind = "no-save-in-window"
        seed_attribution.setdefault(outcome, Counter())[kind] += 1

    # AC4 headline: seeds whose most recent save attempt *failed*. A
    # ``no-save-in-window`` seed is NOT proof of a failed save (the session
    # may simply have had no attempt inside the window), so it is reported in
    # ``seeds_by_last_attempt`` but excluded here.
    failed_save_attr = {
        outcome: counts.get("eviction-failed", 0) + counts.get("router-failed", 0)
        for outcome, counts in seed_attribution.items()
    }

    return {
        "window": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "hours": round((end - start).total_seconds() / 3600.0, 2),
        },
        "per_path": {
            "router": {
                "attempts": router_ok + router_fail,
                "ok": router_ok,
                "failed": router_fail,
            },
            "eviction": {
                "attempts": eviction_ok + eviction_fail,
                "ok": eviction_ok,
                "failed": eviction_fail,
            },
        },
        "failure_reasons": dict(sorted(Counter(f["error"] for f in failures).items())),
        "failure_elapsed_seconds": dict(
            sorted(Counter(f"{f['elapsed_s']:.1f}" for f in failures).items())
        ),
        "timeout_used_seconds": dict(
            sorted(Counter(f"{f['timeout_s']:.1f}" for f in failures).items())
        ),
        "busy_at_failure": _busy_rollup(failures),
        "success_duration_logged": success_duration_lines > 0,
        "failures": sorted(failures, key=lambda f: f["ts"]),
        "persistence_skips": dict(sorted(skips.items())),
        "seed_outcomes": dict(
            sorted(Counter(o for _t, o, _s in seeds).items())
        ),
        "seeds_by_last_attempt": {
            outcome: dict(sorted(counts.items()))
            for outcome, counts in sorted(seed_attribution.items())
        },
        "failed_save_attribution": dict(sorted(failed_save_attr.items())),
    }


def _parse_busy(raw: str) -> dict:
    """Best-effort parse of the trailing ``busy={...}`` snapshot."""
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    return {"raw": raw}


def _busy_rollup(failures: list[dict]) -> dict:
    """Summarise the concurrency signal at each failure (hypothesis 2)."""
    active = Counter()
    local = Counter()
    alone = 0
    for f in failures:
        busy = f["busy"]
        aq = busy.get("active_queries")
        lq = busy.get("local_active_queries")
        if isinstance(aq, int):
            active[aq] += 1
        if isinstance(lq, int):
            local[lq] += 1
        if aq == 0 and lq == 0:
            alone += 1
    return {
        "failures_with_zero_proxy_queries": alone,
        "active_queries_distribution": dict(sorted(active.items())),
        "local_active_queries_distribution": dict(sorted(local.items())),
    }


def render_text(report: dict) -> str:
    """Render the report as a short human-readable summary."""
    pp = report["per_path"]
    lines = [
        f"window: {report['window']['start']} .. {report['window']['end']} "
        f"({report['window']['hours']} h)",
        (
            f"router: {pp['router']['attempts']} attempts "
            f"({pp['router']['ok']} ok / {pp['router']['failed']} failed)"
        ),
        (
            f"eviction: {pp['eviction']['attempts']} attempts "
            f"({pp['eviction']['ok']} ok / {pp['eviction']['failed']} failed)"
        ),
        f"failure reasons: {report['failure_reasons']}",
        f"failure elapsed (s): {report['failure_elapsed_seconds']}",
        f"timeout used (s): {report['timeout_used_seconds']}",
        f"failures at zero proxy queries: "
        f"{report['busy_at_failure']['failures_with_zero_proxy_queries']}",
        f"success duration distribution logged: {report['success_duration_logged']}",
        f"persistence skips: {report['persistence_skips']}",
        f"seed outcomes: {report['seed_outcomes']}",
        f"seeds by last save attempt: {report['seeds_by_last_attempt']}",
        f"failed-save-attributable seeds: {report['failed_save_attribution']}",
    ]
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
