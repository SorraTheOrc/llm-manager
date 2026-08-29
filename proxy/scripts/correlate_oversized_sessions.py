#!/usr/bin/env python3
"""Correlate oversized sessions with decode stalls and fallback storms.

F2 deliverable (LP-0MTC8A2UB0040NKQ, parent LP-0MTAQNAQT002L746): demonstrate
the causal chain "oversized sessions -> full re-prefills -> slot
monopolization -> decode collapse -> fallback storms" for the 2026-08-26
incident, and quantify how much prefill work was wasted on sessions that
could never fit in a local slot.

Data model

Two log families, one timestamped, one not:

1. **Proxy logs** (``/var/log/llama-proxy/proxy.log*``, timestamped):
   - ``routing_check``: per-request routing-time session context estimate
     (``estimated_tokens``, ``warm_threshold`` = per-slot clamp; mode via
     clamp 83285 fast / 100000 cheap, as in analyze_context_distribution.py)
   - ``context_pressure``: warnings when a session's estimated context crosses
     0.80 of the per-slot capacity (``estimated_tokens``, ``per_slot_ctx``,
     ``ratio``)
   - ``routing_skip_local reason=context_too_large|large_context_bypass``:
     hard-cap skips (fallback storms)
   - ``local_dispatch_denied``: queue/slot contention denials
   - ``upstream error status=5xx``: backend 5xx responses
   These give the hour-by-hour timeline.

2. **llama-server logs** (``/var/log/llama-proxy/llama-server*.log``, **no
   timestamps** — llama.cpp only logs ``[pid] ...`` lines): per-request
   ``prompt eval time = X ms / N tokens (R tokens per second)`` (prefill) and
   ``eval time = X ms / N tokens (R tokens per second)`` (decode). File
   rotation mtimes delimit windows; the incident window (Aug 25 22:00 ->
   Aug 26 01:00) is covered by `llama-server.14.log`, which contains most of
   the sub-1-t/s decode observations. Decode-rate evidence is correlated at
   file-window granularity, never hour-exact.

Correlation outputs (see AC1-AC3):
- Hourly timeline of pressure warnings / skips / denials / 5xx (Aug 26).
- Session-level: top session by prefill work (sum of ``estimated_tokens``
  over its routing checks), peak estimated context, ``ratio > 1.0`` flag
  ("could never fit"), its skip/fallback correlation.
- Wasted prefill: sum of ``estimated_tokens`` over routing checks where the
  session's context ratio exceeded 1.0 (the request was locally prefilled in
  full although the context can never be resident in one slot — the KV
  persists nothing useful and the next turn re-prefills again).

CLI::

    python3 proxy/scripts/correlate_oversized_sessions.py \\
        --log-dir /var/log/llama-proxy \\
        --day 2026-08-26 \\
        --output-dir proxy/docs/context-compaction-eval
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from analyze_context_distribution import (
    TS_FMT,
    discover_log_files,
    iter_log_lines,
    parse_routing_sample,
)

FAST_CAP = 83285  # per-slot cap, fast (87.4K / 3 - 4096 headroom)
CHEAP_CAP = 61440  # static clamp, cheap

PRESSURE_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?context_pressure "
    r"session=([A-Za-z0-9_.-]+) estimated_tokens=(\d+) per_slot_ctx=(\d+) ratio=([\d.]+)"
)
SKIP_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?routing_skip_local .*?reason=(\w+)"
)
DENIED_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?local_dispatch_denied"
)
UPSTREAM_5XX_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?upstream error status=(5\d\d)"
)
LLAMA_EVAL_RE = re.compile(
    r"(?<!prompt )eval time\s*=\s*([\d.]+)\s+ms\s+/\s+(\d+)\s+tokens\s*\(\s*([\d.]+)\s+ms per token,\s*([\d.]+)\s+tokens per second\)"
)
LLAMA_PREFILL_RE = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s+ms\s+/\s+(\d+)\s+tokens\s*\(\s*([\d.]+)\s+ms per token,\s*([\d.]+)\s+tokens per second\)"
)


def iter_llama_lines(path: Path):
    """Yield text lines from a llama-server log (plain or .gz).

    llama.cpp logs start with ``[pid]`` (or ``srv`` for boot/exit lines), so
    the shared proxy iterator (which skips lines not beginning with a date)
    cannot be reused here. Lines are read with errors replaced — llama.cpp
    occasionally emits binary garbage mid-stream.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            yield line


@dataclass
class HourlyEvent:
    """One timestamped event type with a per-hour count."""

    name: str
    hour_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, ts: datetime) -> None:
        self.hour_counts[ts.strftime("%H:00")] += 1

    @property
    def total(self) -> int:
        return sum(self.hour_counts.values())


@dataclass
class SessionCorrelation:
    """Per-session correlation summary for the day."""

    session: str
    mode: str = "other"
    checks: int = 0
    peak_est: int = 0
    prefill_work: int = 0        # sum of estimated_tokens over routing checks
    wasted_work: int = 0         # prefill work on checks with ratio > 1.0
    ratio_gt_one: int = 0        # number of checks where est > per_slot (can't fit)
    skip_count: int = 0          # routing_skip_local lines for this session
    pressure_count: int = 0      # context_pressure warnings for this session


@dataclass
class DayCorrelation:
    """Full correlation result for one calendar day."""

    day: str
    hours: list[tuple[str, int, int, int, int, int]] = field(default_factory=list)
    # (hour_label, pressure, skips, denied, upstream_5xx, local_checks)
    sessions: dict[str, SessionCorrelation] = field(default_factory=dict)
    total_routing_checks: int = 0
    total_wasted_work: int = 0
    total_prefill_work: int = 0
    total_skips: int = 0
    total_pressure: int = 0


def llama_log_windows(log_dir: Path) -> list[tuple[str, datetime]]:
    """(filename, window_end) for llama-server logs, ordered by rotation time."""
    out: list[tuple[str, datetime]] = []
    for p in sorted(log_dir.iterdir()):
        name = p.name
        if not (name.startswith("llama-server") and name.endswith(".log")):
            continue
        if name == "llama-server.log":
            continue  # current, unbounded
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        out.append((name, mtime))
    out.sort(key=lambda t: t[1])
    return out


def decode_stats_for_window(
    log_dir: Path,
    all_windows: list[tuple[str, datetime]],
    start: datetime,
    end: datetime,
):
    """Aggregate llama-server decode/prefill stats for the [start, end) window.

    llama-server logs carry no timestamps; the rotation mtime of each file is
    used as its window close time, so files whose close time falls inside the
    window are attributed to it (approximation, documented in the report).
    """
    slow_decodes: list[tuple[float, int]] = []   # (tps, tokens)
    decode_rates: list[float] = []
    prefill_events: list[tuple[float, int, float]] = []  # (ms, tokens, tokens_per_s)
    prefill_tokens = 0
    for name, close in all_windows:
        if not (start <= close < end):
            continue
        p = log_dir / name
        for line in iter_llama_lines(p):
            m2 = LLAMA_PREFILL_RE.search(line)
            if m2:
                prefill_events.append(
                    (float(m2.group(1)), int(m2.group(2)), float(m2.group(4)))
                )
                prefill_tokens += int(m2.group(2))
                continue
            m = LLAMA_EVAL_RE.search(line)
            if m:
                tps = float(m.group(4))
                tokens = int(m.group(2))
                decode_rates.append(tps)
                if tps < 1.0:
                    slow_decodes.append((tps, tokens))
                continue
    return {
        "windows": [n for n, _ in all_windows if start <= _ < end],
        "decode_obs": len(decode_rates),
        "decode_median_tps": round(statistics.median(decode_rates), 2)
        if decode_rates
        else None,
        "decode_min_tps": round(min(decode_rates), 3) if decode_rates else None,
        "slow_decodes_lt_1tps": len(slow_decodes),
        "slow_decodes_examples": sorted(slow_decodes, key=lambda t: t[0])[:5],
        "prefill_events": len(prefill_events),
        "prefill_total_tokens": prefill_tokens,
        "max_prefill_tokens": max((t for _, t, _ in prefill_events), default=0),
    }


def analyze_day(
    log_dir: Path, day: datetime, caps: dict[str, int]
) -> DayCorrelation:
    """Build the day's correlation from proxy logs."""
    start = day
    end = day + timedelta(days=1)
    res = DayCorrelation(day=day.strftime("%Y-%m-%d"))
    events = {
        "pressure": HourlyEvent("context_pressure"),
        "skips": HourlyEvent("routing_skip_local"),
        "denied": HourlyEvent("local_dispatch_denied"),
        "upstream5xx": HourlyEvent("upstream 5xx"),
    }
    pressure_of_session: dict[str, int] = defaultdict(int)
    skip_of_session: dict[str, int] = defaultdict(int)

    for path in discover_log_files(log_dir):
        for line in iter_log_lines(path):
            ts_line = line[:19]
            try:
                ts = datetime.strptime(ts_line, TS_FMT)
            except ValueError:
                continue
            if not (start <= ts < end):
                continue

            if "context_pressure" in line:
                m = PRESSURE_LINE_RE.search(line)
                if m:
                    events["pressure"].add(ts)
                    pressure_of_session[m.group(2)] += 1
                continue
            if "routing_skip_local" in line:
                m = SKIP_LINE_RE.search(line)
                if m:
                    events["skips"].add(ts)
                    sess = line.split("session=", 1)[1].split()[0].rstrip(",")
                    if sess and sess != "unknown" and sess[0].isalnum():
                        skip_of_session[sess] += 1
                continue
            if "local_dispatch_denied" in line:
                events["denied"].add(ts)
                continue
            if "upstream error status=5" in line:
                m = UPSTREAM_5XX_RE.search(line)
                if m:
                    events["upstream5xx"].add(ts)
                continue
            sample = parse_routing_sample(line)
            if sample is not None:
                res.total_routing_checks += 1
                sess = sample.session
                if sess == "unknown":
                    continue
                agg = res.sessions.setdefault(
                    sess, SessionCorrelation(session=sess)
                )
                agg.checks += 1
                agg.peak_est = max(agg.peak_est, sample.estimated_tokens)
                agg.prefill_work += sample.estimated_tokens
                cap = caps.get(sample.mode, 0)
                if cap and sample.estimated_tokens > cap:
                    agg.ratio_gt_one += 1
                    agg.wasted_work += sample.estimated_tokens
                if sample.mode in ("fast", "cheap"):
                    agg.mode = sample.mode

    # merge pressure/skip tallies
    for s, n in pressure_of_session.items():
        if s in res.sessions:
            res.sessions[s].pressure_count = n
    for s, n in skip_of_session.items():
        if s in res.sessions:
            res.sessions[s].skip_count = n

    res.total_pressure = events["pressure"].total
    res.total_skips = events["skips"].total
    res.total_prefill_work = sum(s.prefill_work for s in res.sessions.values())
    res.total_wasted_work = sum(s.wasted_work for s in res.sessions.values())

    # hour labels 00:00..23:00, bucketed by hour (HH:00)
    by_hour: dict[str, dict] = defaultdict(dict)
    for name, ev in events.items():
        for h, c in ev.hour_counts.items():
            by_hour[h][name] = c
    per_hour_checks: Counter = Counter()
    for path in discover_log_files(log_dir):
        for line in iter_log_lines(path):
            sample = parse_routing_sample(line)
            if sample is None:
                continue
            if start <= sample.ts < end:
                per_hour_checks[sample.ts.strftime("%H:00")] += 1
    res.hours = [
        (
            f"{h:02d}:00",
            by_hour.get(f"{h:02d}:00", {}).get("pressure", 0),
            by_hour.get(f"{h:02d}:00", {}).get("skips", 0),
            by_hour.get(f"{h:02d}:00", {}).get("denied", 0),
            by_hour.get(f"{h:02d}:00", {}).get("upstream5xx", 0),
            per_hour_checks.get(f"{h:02d}:00", 0),
        )
        for h in range(24)
    ]
    return res


def build_report(
    log_dir: Path, day: datetime, caps: dict[str, int]
) -> dict:
    """Full machine-readable correlation report for the day."""
    res = analyze_day(log_dir, day, caps)
    windows = llama_log_windows(log_dir)
    # llama-server attribution window: the incident calendar day plus slack on
    # both sides so rotation-boundary files (evening peak lands in the file
    # closed at ~01:00 next day) are attributed. Approximation documented in
    # the report (llama.cpp logs carry no timestamps).
    start = day - timedelta(hours=6)
    end = day + timedelta(hours=30)
    llm = decode_stats_for_window(log_dir, windows, start, end)

    top_sessions = sorted(
        res.sessions.values(),
        key=lambda s: s.wasted_work,
        reverse=True,
    )[:15]

    return {
        "day": res.day,
        "caps": dict(caps),
        "totals": {
            "routing_checks": res.total_routing_checks,
            "prefill_work_tokens": res.total_prefill_work,
            "wasted_work_tokens": res.total_wasted_work,
            "wasted_pct_of_prefill": round(
                res.total_wasted_work / res.total_prefill_work * 100, 1
            )
            if res.total_prefill_work
            else 0.0,
            "context_pressure_warnings": res.total_pressure,
            "routing_skips": res.total_skips,
            "sessions": len(res.sessions),
        },
        "hourly_timeline": [
            {
                "hour": h,
                "pressure_warnings": p,
                "routing_skips": s,
                "dispatch_denied": d,
                "upstream_5xx": u,
                "routing_checks": c,
            }
            for (h, p, s, d, u, c) in res.hours
        ],
        "top_sessions": [
            {
                "session": s.session,
                "mode": s.mode,
                "checks": s.checks,
                "peak_estimated_tokens": s.peak_est,
                "prefill_work_tokens": s.prefill_work,
                "wasted_work_tokens": s.wasted_work,
                "checks_ratio_gt_1": s.ratio_gt_one,
                "skips": s.skip_count,
                "pressure_warnings": s.pressure_count,
            }
            for s in top_sessions
        ],
        "llama_server_decode": llm,
    }


def render_markdown(report: dict) -> str:
    """Render the correlation report as Markdown (evidence tables)."""
    lines: list[str] = []
    lines.append(f"# Oversized-session correlation ({report['day']})")
    lines.append("")
    lines.append(
        "Proxy-log timeline (timestamped) correlated with llama-server "
        "prefill/decode evidence (no timestamps in llama.cpp logs — attributed "
        "by file rotation window). Session context = `estimated_tokens` at "
        "routing time."
    )
    lines.append("")
    t = report["totals"]
    lines.append("## Totals")
    lines.append("")
    lines.append(
        f"- Routing checks: **{t['routing_checks']}** | sessions: **{t['sessions']}**"
    )
    lines.append(
        f"- context_pressure warnings: **{t['context_pressure_warnings']}** | "
        f"routing_skip_local: **{t['routing_skips']}**"
    )
    lines.append(
        f"- Estimated prefill work: **{t['prefill_work_tokens']:,} tokens**"
    )
    lines.append(
        f"- **Wasted prefill (ratio > 1.0): {t['wasted_work_tokens']:,} tokens "
        f"({t['wasted_pct_of_prefill']}% of all prefill)**"
    )
    lines.append("")
    lines.append("## Hourly timeline")
    lines.append("")
    lines.append("| Hour | Pressure | Skips | Dispatch denied | Upstream 5xx | Routing checks |")
    lines.append("|---|---|---|---|---|---|")
    for row in report["hourly_timeline"]:
        if row["pressure_warnings"] or row["routing_skips"] or row["routing_checks"]:
            lines.append(
                f"| {row['hour']} | {row['pressure_warnings']} | "
                f"{row['routing_skips']} | {row['dispatch_denied']} | "
                f"{row['upstream_5xx']} | {row['routing_checks']} |"
            )
    lines.append("")
    lines.append("## Top sessions by wasted prefill work")
    lines.append("")
    lines.append(
        "| Session | Mode | Checks | Peak est. ctx | Prefill (tokens) | "
        "Wasted (tokens) | Checks ratio>1 | Skips | Pressure |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in report["top_sessions"]:
        lines.append(
            f"| `{s['session'][:32]}` | {s['mode']} | {s['checks']} | "
            f"{s['peak_estimated_tokens']:,} | {s['prefill_work_tokens']:,} | "
            f"{s['wasted_work_tokens']:,} | {s['checks_ratio_gt_1']} | "
            f"{s['skips']} | {s['pressure_warnings']} |"
        )
    lines.append("")
    d = report["llama_server_decode"]
    lines.append("## llama-server decode/prefill evidence (window)")
    lines.append("")
    lines.append(f"- Files in window: {', '.join(d['windows']) or 'none'}")
    lines.append(
        f"- Decode observations: **{d['decode_obs']}** | median "
        f"{d['decode_median_tps']} t/s | min {d['decode_min_tps']} t/s"
    )
    lines.append(f"- **Slow decodes (< 1 t/s): {d['slow_decodes_lt_1tps']}**")
    lines.append(f"- Prefill events: {d['prefill_events']} | total "
                 f"{d['prefill_total_tokens']:,} tokens | max prefill "
                 f"{d['max_prefill_tokens']:,} tokens")
    if d["slow_decodes_examples"]:
        ex = ", ".join(f"{tps} t/s ({n} tok)" for tps, n in d["slow_decodes_examples"])
        lines.append(f"- Slowest examples: {ex}")
    lines.append("")
    lines.append("## Caveats and methodology")
    lines.append("")
    lines.append(
        "- **llama.cpp logs carry no timestamps**; decode/prefill evidence is "
        "attributed by rotation-file close time, not hour-exact event time."
    )
    lines.append(
        "- **Proxy log gap on Aug 26 22:00-24:00**: rotated logs stop at "
        "22:00:22 (proxy.log.2026-08-26_16) and resume at 00:00:03 Aug 27 "
        "(proxy.log.2026-08-27_01); ~2h of event data (including most of hour "
        "22) is absent. Earlier cited figures for hour 22 (4,541 fallbacks, "
        "280 backend 5xx, 42.7M prefill tokens, max 85,724) predate the "
        "calendar-day reconstruction and cover other windows; this report "
        "recomputes from the calendar day with the gap called out."
    )
    lines.append(
        "- **Wasted prefill** counts ``estimated_tokens`` on routing checks whose "
        "context exceeded the per-slot clamp (ratio > 1.0): such a session can "
        "never be resident in one slot, so every turn is a full re-prefill that "
        "persists no reusable KV — the prefill work is lost."
    )
    lines.append("")
    return "\n".join(lines)


def write_artifacts(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jp = output_dir / "correlation.json"
    mp = output_dir / "correlation.md"
    jp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mp.write_text(render_markdown(report) + "\n", encoding="utf-8")
    return jp, mp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="correlate_oversized_sessions.py",
        description=(
            "Correlate oversized sessions with decode stalls and fallback storms "
            "(evaluation only; no behavior change)."
        ),
    )
    parser.add_argument(
        "--log-dir", default="/var/log/llama-proxy",
        help="dir containing proxy.log* and llama-server*.log",
    )
    parser.add_argument(
        "--day", default="2026-08-26", help="incident calendar day (default 2026-08-26)"
    )
    parser.add_argument(
        "--output-dir", default="proxy/docs/context-compaction-eval",
        help="output dir for correlation.json / correlation.md",
    )
    parser.add_argument("--fast-cap", type=int, default=FAST_CAP)
    parser.add_argument("--cheap-cap", type=int, default=CHEAP_CAP)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"error: log dir not found: {log_dir}", file=sys.stderr)
        return 2
    try:
        day = datetime.strptime(args.day, "%Y-%m-%d")
    except ValueError:
        print("error: --day must be YYYY-MM-DD", file=sys.stderr)
        return 2
    report = build_report(log_dir, day, {"fast": args.fast_cap, "cheap": args.cheap_cap})
    jp, mp = write_artifacts(report, Path(args.output_dir))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
        print(f"artifacts: {jp} {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
