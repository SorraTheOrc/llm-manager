#!/usr/bin/env python3
"""Correlate slot save/restore failures with concurrent local load.

Root-cause correlation for LP-0MSI1RWLM007N367 (F1): slot save/restore
ReadTimeouts persist under concurrent load (~1.8% of saves). This script
maps proxy-side slot_save/slot_restore events to:

1. the adaptive-timeout cadence — gaps between consecutive failures for a
   slot (a burst of ~25-70s gaps at exactly the computed timeout window
   proves the proxy waited the FULL window before giving up);
2. concurrent local load — local stream starts/finishes around each failure
   (failures cluster when other local streams are active);
3. llama-server KV serialization activity — prompt_save/prompt_load lines
   (llama-server logs lack timestamps, so this is a count, not a timing;
   the F1 proxy instrumentation adds per-request elapsed time going forward).

Usage:
  ./scripts/slot-persistence-correlate.py                          # default /var/log/llama-proxy
  ./scripts/slot-persistence-correlate.py --log-dir /var/log/llama-proxy
  ./scripts/slot-persistence-correlate.py --start 2026-08-06 --end 2026-08-07
  ./scripts/slot-persistence-correlate.py --json                    # machine-readable
  ./scripts/slot-persistence-correlate.py --window 120              # load window ±120s

Exit codes:
  0 - success
  1 - no log files found / unexpected error
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# 2026-08-06 01:57:19,894 - WARNING - slot_save failed slot=2 error=... elapsed=... timeout=... busy=...
_FAIL_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} - \w+ - "
    r"slot_(?P<action>save|restore) failed slot=(?P<slot>\d+) "
    r"(?P<rest>.*)$"
)
# 2026-08-06 01:55:47,533 - INFO - slot_save success session=019fd490 slot=0
_SUCCESS_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} - \w+ - "
    r"slot_(?P<action>save|restore) success session=(?P<session>\S+) slot=(?P<slot>\d+)$"
)
# 2026-08-06 01:55:13,618 - INFO - Stream started: provider=local model=Qwen3 session=...
# 2026-08-06 01:55:13,618 - INFO - Stream started: provider=local model=Qwen3 session=...
# 2026-08-06 01:55:19,388 - INFO - Stream finished: reason=tool_calls session=... provider=local model=Qwen3 ...
# (started lines put provider=local BEFORE session=; finished lines put it AFTER)
_STREAM_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} - \w+ - "
    r"Stream (?:started|finished): .*?"
    r"(?:session=(?P<session>\S+).*?provider=local|"
    r"provider=local.*?session=(?P<session2>\S+))"
)
# 2026-08-06 01:55:13,618 - INFO - Fallback triggered ... reason=local_concurrency_limit
_FALLBACK_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} - \w+ - "
    r"Fallback triggered .*reason=(?P<reason>\S+)$"
)
# llama-server: [36051] srv   prompt_save:  - saving prompt with length 26234, total state size = 335.517 MiB
_LLAMA_SAVE_RE = re.compile(r"prompt_save:")
_LLAMA_LOAD_RE = re.compile(r"prompt_load:")

_TS_FMT = "%Y-%m-%d %H:%M:%S"


def _parse_ts(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, _TS_FMT)


def _iter_proxy_logs(log_dir: Path):
    """Yield lines from proxy.log then rotated proxy.log.* (oldest last)."""
    live = log_dir / "proxy.log"
    rotated = sorted(
        (p for p in log_dir.glob("proxy.log.*") if p.is_file()),
        key=lambda p: p.name,
    )
    files = [live] + rotated if live.exists() else rotated
    for path in files:
        try:
            with path.open(errors="replace") as fh:
                yield from fh
        except OSError as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)


def _iter_llama_logs(log_dir: Path):
    live = log_dir / "llama-server.log"
    rotated = sorted(
        (p for p in log_dir.glob("llama-server.log.*") if p.is_file()),
        key=lambda p: p.name,
    )
    files = [live] + rotated if live.exists() else rotated
    for path in files:
        try:
            with path.open(errors="replace") as fh:
                yield from fh
        except OSError as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(log_dir: Path, start: dt.datetime | None, end: dt.datetime | None,
            window: int, stream_cap: int = 900) -> dict:
    failures: list[dict] = []
    successes: list[dict] = []
    # session -> sorted [(ts, kind)] for local streams (full history, so
    # active-at-failure counts stay correct across window boundaries)
    sessions: dict[str, list[tuple[dt.datetime, str]]] = defaultdict(list)
    fallback_events: list[tuple[dt.datetime, str]] = []  # (ts, reason)
    llama_saves = 0
    llama_loads = 0

    for line in _iter_proxy_logs(log_dir):
        m = _FAIL_RE.match(line)
        if m:
            ts = _parse_ts(m.group("ts"))
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            failures.append({
                "ts": ts.strftime(_TS_FMT),
                "action": m.group("action"),
                "slot": int(m.group("slot")),
                "detail": m.group("rest").strip(),
            })
            continue
        m = _SUCCESS_RE.match(line)
        if m:
            ts = _parse_ts(m.group("ts"))
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            successes.append({
                "ts": ts.strftime(_TS_FMT),
                "action": m.group("action"),
                "slot": int(m.group("slot")),
                "session": m.group("session"),
            })
            continue
        # Load-context events (streams, fallbacks) are collected over the
        # FULL log history so active-stream counts at failure time remain
        # correct even when the failure window starts mid-stream.
        m = _STREAM_RE.match(line)
        if m:
            session = m.group("session") or m.group("session2")
            if not session:
                continue
            kind = "started" if "started" in line else "finished"
            sessions[session].append((_parse_ts(m.group("ts")), kind))
            continue
        m = _FALLBACK_RE.match(line)
        if m:
            fallback_events.append((_parse_ts(m.group("ts")), m.group("reason")))

    for line in _iter_llama_logs(log_dir):
        if _LLAMA_SAVE_RE.search(line):
            llama_saves += 1
        elif _LLAMA_LOAD_RE.search(line):
            llama_loads += 1

    # --- Per-slot failure cadence (gaps between consecutive failures) ---
    by_slot: dict[int, list[dt.datetime]] = defaultdict(list)
    for f in failures:
        by_slot[f["slot"]].append(_parse_ts(f["ts"]))
    cadence: dict[str, list[int]] = {}
    for slot, tss in sorted(by_slot.items()):
        gaps = [
            int((tss[i] - tss[i - 1]).total_seconds())
            for i in range(1, len(tss))
        ]
        cadence[f"slot_{slot}"] = gaps

    # --- Load context per failure: active local streams at failure instant ---
    # A session is "active at ts" when its most recent local stream event at
    # or before ts was "started" (not yet finished). Stale "started" events
    # older than *stream_cap* seconds are treated as finished so crashed or
    # log-rotation-truncated streams don't inflate the count forever.
    for events in sessions.values():
        events.sort(key=lambda e: e[0])

    def _active_streams(ts: dt.datetime) -> int:
        count = 0
        for events in sessions.values():
            last: tuple[dt.datetime, str] | None = None
            for ets, kind in events:
                if ets > ts:
                    break
                last = (ets, kind)
            if last is None:
                continue
            ets, kind = last
            if kind == "started" and (ts - ets).total_seconds() <= stream_cap:
                count += 1
        return count

    for f in failures:
        ts = _parse_ts(f["ts"])
        f["active_local_streams_approx"] = _active_streams(ts)
        f["concurrency_fallbacks_in_window"] = sum(
            1
            for fts, reason in fallback_events
            if abs((fts - ts).total_seconds()) <= window
            and reason == "local_concurrency_limit"
        )

    failure_actions = Counter(f["action"] for f in failures)
    success_actions = Counter(s["action"] for s in successes)

    return {
        "window_seconds": window,
        "totals": {
            "slot_save_failed": failure_actions["save"],
            "slot_save_success": success_actions["save"],
            "slot_restore_failed": failure_actions["restore"],
            "slot_restore_success": success_actions["restore"],
            "llama_prompt_save": llama_saves,
            "llama_prompt_load": llama_loads,
        },
        "failure_rate_pct": {
            "save": _rate(failure_actions["save"], success_actions["save"]),
            "restore": _rate(failure_actions["restore"], success_actions["restore"]),
        },
        "failures": sorted(failures, key=lambda f: f["ts"]),
        "cadence_seconds_by_slot": cadence,
        "load_context": {
            "failures_with_local_streams": sum(
                1 for f in failures if f["active_local_streams_approx"] > 0
            ),
            "failures_without_local_streams": sum(
                1 for f in failures if f["active_local_streams_approx"] == 0
            ),
            "max_active_local_streams_at_failure": max(
                (f["active_local_streams_approx"] for f in failures), default=0
            ),
        },
    }


def _rate(failed: int, success: int) -> float:
    total = failed + success
    if total == 0:
        return 0.0
    return round(100.0 * failed / total, 2)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_markdown(report: dict) -> str:
    t = report["totals"]
    lines = [
        "# Slot save/restore failure correlation",
        "",
        f"- Window: ±{report['window_seconds']}s around each failure",
        f"- slot_save: {t['slot_save_failed']} failed / {t['slot_save_success']} success "
        f"({report['failure_rate_pct']['save']}%)",
        f"- slot_restore: {t['slot_restore_failed']} failed / {t['slot_restore_success']} success "
        f"({report['failure_rate_pct']['restore']}%)",
        f"- llama-server prompt_save lines: {t['llama_prompt_save']}; "
        f"prompt_load lines: {t['llama_prompt_load']} (no timestamps in llama-server logs)",
        "",
        "## Load context",
        "",
        f"- Failures with ≥1 concurrent local stream: {report['load_context']['failures_with_local_streams']}",
        f"- Failures with no concurrent local stream: {report['load_context']['failures_without_local_streams']}",
        f"- Max concurrent local streams at a failure: {report['load_context']['max_active_local_streams_at_failure']}",
        "",
        "## Failure cadence (gaps between consecutive failures, per slot)",
        "",
    ]
    if report["cadence_seconds_by_slot"]:
        lines.append("| Slot | Consecutive-failure gaps (s) |")
        lines.append("|------|------------------------------|")
        for slot, gaps in report["cadence_seconds_by_slot"].items():
            gap_str = ", ".join(str(g) for g in gaps) if gaps else "—"
            lines.append(f"| {slot} | {gap_str} |")
        lines.append("")
        lines.append(
            "Gaps at the adaptive-timeout cadence (~25-70s = 3s base + "
            "0.001s/token × est tokens) indicate the proxy waited the FULL "
            "window before the ReadTimeout — the save was starved behind "
            "concurrent slot work, not interrupted mid-copy."
        )
    else:
        lines.append("No failures in the analysed window.")
    lines.append("")
    lines.append("## Failure events")
    lines.append("")
    if report["failures"]:
        lines.append("| Timestamp | Action | Slot | Detail | Load |")
        lines.append("|-----------|--------|------|--------|------|")
        for f in report["failures"]:
            lines.append(
                f"| {f['ts']} | {f['action']} | {f['slot']} | "
                f"{f['detail']} | {f['active_local_streams_approx']} streams |"
            )
    else:
        lines.append("No failure events.")
    lines.append("")
    lines.append(
        "Newer failures include elapsed=.../timeout=.../busy={...} fields "
        "(LP-0MSI1RWLM007N367 F1 instrumentation) for exact latency vs "
        "timeout comparison."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="/var/log/llama-proxy",
                        help="directory containing proxy.log*/llama-server.log*")
    parser.add_argument("--start", help="start timestamp filter (YYYY-MM-DD[ HH:MM:SS])")
    parser.add_argument("--end", help="end timestamp filter (YYYY-MM-DD[ HH:MM:SS])")
    parser.add_argument("--window", type=int, default=120,
                        help="load-context window in seconds around each failure")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"error: log directory not found: {log_dir}", file=sys.stderr)
        return 1

    start = _parse_optional_ts(args.start)
    end = _parse_optional_ts(args.end)
    if start and end and start > end:
        print("error: --start must be before --end", file=sys.stderr)
        return 1

    try:
        report = analyze(log_dir, start, end, args.window)
    except Exception as exc:
        print(f"error: analysis failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(render_markdown(report))
    return 0


def _parse_optional_ts(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise SystemExit(f"error: invalid timestamp: {value!r} (use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)")


if __name__ == "__main__":
    sys.exit(main())
