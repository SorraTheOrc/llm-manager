"""CSV and Markdown report generation, plus the end-to-end analysis runner.

Outputs (per acceptance criteria):

- ``daytime_sessions.csv`` — one row per daytime session (10:00-23:59, 6 slots
  per the configured schedule) covering ALL sessions in the window.
- ``nighttime_sessions.csv`` — one row per nighttime session (00:00-09:59).
- ``report.md`` — the aggregate Markdown report with highlighted,
  data-backed recommendations.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from pathlib import Path


import aggregation
import bucketing
import config_loader
import log_parser
import recommendations
from aggregation import AnalysisResult, SessionStats

CSV_COLUMNS = [
    "session_id",
    "start_time",
    "end_time",
    "duration_seconds",
    "messages",
    "start_context_size",
    "avg_context_size",
    "max_context_size",
    "avg_response_size",
    "max_response_size",
    "initial_provider",
    "initial_model",
    "remote_move_time",
    "fallback_reason",
    "bucket",
    "slots",
    "local_requests",
    "remote_requests",
    "dispatch_denied",
]

TS_FMT = "%Y-%m-%d %H:%M:%S"


@dataclass
class AnalysisRun:
    summary: AnalysisResult
    files: list[Path] = field(default_factory=list)


def _fmt_ts(ts: datetime | None) -> str:
    return ts.strftime(TS_FMT) if ts else ""


def _session_row(s: SessionStats) -> dict:
    return {
        "session_id": s.session_id,
        "start_time": _fmt_ts(s.start),
        "end_time": _fmt_ts(s.end),
        "duration_seconds": f"{s.duration_seconds:.1f}",
        "messages": str(s.messages),
        "start_context_size": str(s.start_context_size) if s.start_context_size is not None else "",
        "avg_context_size": f"{s.avg_context_size:.1f}" if s.avg_context_size is not None else "",
        "max_context_size": str(s.max_context_size) if s.max_context_size is not None else "",
        "avg_response_size": f"{s.avg_response_size:.1f}" if s.avg_response_size is not None else "",
        "max_response_size": str(s.max_response_size) if s.max_response_size is not None else "",
        "initial_provider": s.initial_provider or "",
        "initial_model": s.initial_model or "",
        "remote_move_time": _fmt_ts(s.remote_move_time),
        "fallback_reason": s.fallback_reason or "",
        "bucket": s.bucket or "",
        "slots": str(s.slots) if s.slots else "",
        "local_requests": str(s.local_requests),
        "remote_requests": str(s.remote_requests),
        "dispatch_denied": str(s.dispatch_denied),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_csvs(summary: AnalysisResult, out_dir: Path) -> tuple[Path, Path]:
    """Write the daytime and nighttime session CSVs; returns their paths."""
    day_rows, night_rows = [], []
    for s in sorted(summary.sessions.values(), key=lambda x: x.start):
        row = _session_row(s)
        (day_rows if s.bucket != "night" else night_rows).append(row)
    day_path = out_dir / "daytime_sessions.csv"
    night_path = out_dir / "nighttime_sessions.csv"
    _write_csv(day_path, day_rows)
    _write_csv(night_path, night_rows)
    return day_path, night_path


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def build_report(summary: AnalysisResult, config: dict | None) -> str:
    recs = recommendations.generate_recommendations(summary, config)
    hours = (summary.window_end - summary.window_start).total_seconds() / 3600.0
    sessions = list(summary.sessions.values())
    total = summary.total_requests

    local_only = [s for s in sessions if s.remote_requests == 0]
    fell_back = [s for s in sessions if s.fell_back]
    remote_only = [s for s in sessions if s.local_requests == 0 and s.remote_requests > 0]

    fallback_rate = (len(summary.fallback_events) / total) if total else 0.0
    schedule = bucketing.schedule_from_config(
        config, (config or {}).get("session_slot_pool_size")
    )
    profile = _bucket_profile(
        sessions,
        schedule,
        summary.fallback_events,
        summary.routing_skip_events,
        summary.dispatch_denied_events,
    )

    lines: list[str] = []
    ap = lines.append
    ap("# Proxy Usage Analysis Report")
    ap("")
    ap(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ap(f"- Window: {_fmt_dt(summary.window_start)} → {_fmt_dt(summary.window_end)} ({hours:.1f}h)")
    ap(f"- Sessions in window: **{len(sessions)}** | Requests: **{total}** "
       f"(local {summary.local_requests} / remote {summary.remote_requests})")
    ap(f"- Fallback events: **{len(summary.fallback_events)}** "
       f"({fallback_rate * 100:.1f}% of requests) | dispatch denied: {summary.dispatch_denied_count}")
    ap(f"- Unattributed stream events (no session UUID): {summary.unattributed_events}")
    ap(f"- Lines parsed: {summary.total_lines} | lines skipped: {summary.lines_skipped}")

    ap("")
    ap("## Session summary")
    ap("")
    ap("| Metric | Total | Day | Night |")
    ap("|---|---|---|---|")
    d, n = profile["day"], profile["night"]
    ap(f"| Sessions | {len(sessions)} | {d['sessions']} ({_pct(d['sessions'], len(sessions)):.1f}%) | "
       f"{n['sessions']} ({_pct(n['sessions'], len(sessions)):.1f}%) |")
    ap(f"| Requests | {total} | {d['requests']} ({_pct(d['requests'], total):.1f}%) | "
       f"{n['requests']} ({_pct(n['requests'], total):.1f}%) |")
    ap(f"| Local requests | {summary.local_requests} ({_pct(summary.local_requests, total):.1f}%) | "
       f"{d['local']} ({_pct(d['local'], d['requests']):.1f}%) | "
       f"{n['local']} ({_pct(n['local'], n['requests']):.1f}%) |")
    ap(f"| Remote requests | {summary.remote_requests} ({_pct(summary.remote_requests, total):.1f}%) | "
       f"{d['remote']} ({_pct(d['remote'], d['requests']):.1f}%) | "
       f"{n['remote']} ({_pct(n['remote'], n['requests']):.1f}%) |")
    ap(f"| Local-only sessions | {len(local_only)} ({_pct(len(local_only), len(sessions)):.1f}%) | "
       f"{d['local_only']} ({_pct(d['local_only'], d['sessions']):.1f}%) | "
       f"{n['local_only']} ({_pct(n['local_only'], n['sessions']):.1f}%) |")
    ap(f"| Fell back (local → remote) | {len(fell_back)} ({_pct(len(fell_back), len(sessions)):.1f}%) | "
       f"{d['fell_back']} ({_pct(d['fell_back'], d['sessions']):.1f}%) | "
       f"{n['fell_back']} ({_pct(n['fell_back'], n['sessions']):.1f}%) |")
    ap(f"| Remote-only (never used local) | {len(remote_only)} ({_pct(len(remote_only), len(sessions)):.1f}%) | "
       f"{d['remote_only']} ({_pct(d['remote_only'], d['sessions']):.1f}%) | "
       f"{n['remote_only']} ({_pct(n['remote_only'], n['sessions']):.1f}%) |")
    day_fb = sum(d["fallback_reasons"].values())
    night_fb = sum(n["fallback_reasons"].values())
    ap(f"| Fallback events | {len(summary.fallback_events)} ({fallback_rate * 100:.1f}%) | "
       f"{day_fb} ({_pct(day_fb, len(summary.fallback_events)):.1f}%) | "
       f"{night_fb} ({_pct(night_fb, len(summary.fallback_events)):.1f}%) |")
    ap(f"| Dispatch denied | {summary.dispatch_denied_count} | "
       f"{d['dispatch_denied']} ({_pct(d['dispatch_denied'], summary.dispatch_denied_count):.1f}%) | "
       f"{n['dispatch_denied']} ({_pct(n['dispatch_denied'], summary.dispatch_denied_count):.1f}%) |")
    total_avg, total_max = _ctx_stats(
        [s.max_context_size for s in sessions if s.max_context_size is not None]
    )
    day_avg, day_max = _ctx_stats(d["ctx"])
    night_avg, night_max = _ctx_stats(n["ctx"])
    ap(f"| Avg max context | {total_avg} | {day_avg} | {night_avg} |")
    ap(f"| Highest context | {total_max} | {day_max} | {night_max} |")

    if summary.fallback_reason_counts:
        ap("")
        ap("## Fallback reasons")
        ap("")
        ap("| Reason | Total | % of fallbacks | Day | Night |")
        ap("|---|---|---|---|---|")
        for reason, count in summary.fallback_reason_counts.most_common():
            d = profile["day"]["fallback_reasons"].get(reason, 0)
            n = profile["night"]["fallback_reasons"].get(reason, 0)
            ap(f"| {reason} | {count} | {_pct(count, len(summary.fallback_events)):.1f}% | "
               f"{d} ({_pct(d, count):.1f}%) | {n} ({_pct(n, count):.1f}%) |")

    if summary.routing_skip_reason_counts:
        ap("")
        ap("## routing_skip_local reasons")
        ap("")
        ap("| Reason | Total | % of skips | Day | Night |")
        ap("|---|---|---|---|---|")
        for reason, count in summary.routing_skip_reason_counts.most_common():
            d = profile["day"]["routing_skip_reasons"].get(reason, 0)
            n = profile["night"]["routing_skip_reasons"].get(reason, 0)
            ap(f"| {reason} | {count} | {_pct(count, len(summary.routing_skip_events)):.1f}% | "
               f"{d} ({_pct(d, count):.1f}%) | {n} ({_pct(n, count):.1f}%) |")

    initial = Counter((s.initial_provider, s.initial_model) for s in sessions)
    ap("")
    ap("## Per-model breakdown (initial assignment)")
    ap("")
    ap("| Provider | Model | Sessions | Day | Night | Requests | Fell back |")
    ap("|---|---|---|---|---|---|---|")
    for (provider, model), count in initial.most_common():
        s_list = [s for s in sessions if s.initial_provider == provider and s.initial_model == model]
        day = sum(1 for s in s_list if _bucket_key(s.bucket) == "day")
        night = len(s_list) - day
        reqs = sum(s.messages for s in s_list)
        fb = sum(1 for s in s_list if s.fell_back)
        ap(f"| {provider} | {model} | {count} | {day} ({_pct(day, count):.1f}%) | "
           f"{night} ({_pct(night, count):.1f}%) | {reqs} | {fb} |")

    ap("")
    ap("## Recommendations")
    ap("")
    if not recs:
        ap("_No issues detected._")
    for r in recs:
        ap(f"### [{r.severity.upper()}] {r.title}")
        ap("")
        ap(f"> Evidence: {r.evidence}")
        ap("")
        ap(r.detail)
        ap("")

    ap("## Notes and limitations")
    ap("")
    ap(
        "- Sessions are identified by their session UUID; context/response sizes use the "
        "authoritative per-request `tokens=prompt/completion/total` from `Stream finished` lines "
        "(log-line payloads are truncated and are never used for sizes)."
    )
    ap(
        "- A session is included when it has at least one `Stream started` inside the window; "
        "day/night bucketing uses the session start time and the slot schedule in proxy/config.yaml."
    )
    ap(
        "- Sessions spanning a slot-schedule transition may observe 503s during the drain window; "
        "those are expected and not treated as errors."
    )
    ap(
        "- `Fallback triggered` lines carry no session UUID; per-session attribution prefers the "
        "session's own `routing_skip_local` line and otherwise the nearest fallback event within 60s."
    )
    ap(
        "- Related context: work item LP-0MSAOQTJS000FFVM (evaluate increasing local ctx-size) can "
        "use this report's `large_context_bypass` data. See the skill's SKILL.md for interpretation."
    )
    return "\n".join(lines) + "\n"


def _pct(part: int, total: int) -> float:
    return (part / total * 100.0) if total else 0.0


def _bucket_key(bucket: str | None) -> str:
    return "night" if bucket == "night" else "day"


def _ctx_stats(values: list[int]) -> tuple[object, object]:
    """Return (avg, max) for context sizes, or ("-", "-") when empty."""
    if not values:
        return "-", "-"
    return round(sum(values) / len(values)), max(values)


def _bucket_profile(
    sessions: list[SessionStats],
    schedule: bucketing.SlotSchedule,
    fallback_events: list[log_parser.LogEvent],
    routing_skip_events: list[log_parser.LogEvent],
    dispatch_denied_events: list[log_parser.LogEvent],
) -> dict:
    """Per-bucket (day/night) totals for the report's summary tables.

    Covers sessions, requests, local/remote split, classification counts,
    context sizes, dispatch denials, and per-reason counters. Events without a
    session (fallbacks, routing skips, dispatch denials) are bucketed by their
    own timestamp.
    """
    buckets = {
        "day": {
            "sessions": 0, "requests": 0, "local": 0, "remote": 0,
            "local_only": 0, "fell_back": 0, "remote_only": 0,
            "dispatch_denied": 0, "ctx": [],
            "fallback_reasons": Counter(), "routing_skip_reasons": Counter(),
        },
        "night": {
            "sessions": 0, "requests": 0, "local": 0, "remote": 0,
            "local_only": 0, "fell_back": 0, "remote_only": 0,
            "dispatch_denied": 0, "ctx": [],
            "fallback_reasons": Counter(), "routing_skip_reasons": Counter(),
        },
    }
    for s in sessions:
        b = buckets[_bucket_key(s.bucket)]
        b["sessions"] += 1
        b["requests"] += s.messages
        b["local"] += s.local_requests
        b["remote"] += s.remote_requests
        if s.remote_requests == 0:
            b["local_only"] += 1
        if s.fell_back:
            b["fell_back"] += 1
        if s.local_requests == 0 and s.remote_requests > 0:
            b["remote_only"] += 1
        if s.max_context_size is not None:
            b["ctx"].append(s.max_context_size)
    for ev in fallback_events:
        if not ev.reason:
            continue
        label = schedule.period_for(ev.ts).label if schedule.periods else "day"
        buckets[_bucket_key(label)]["fallback_reasons"][ev.reason] += 1
    for ev in routing_skip_events:
        if not ev.reason:
            continue
        label = schedule.period_for(ev.ts).label if schedule.periods else "day"
        buckets[_bucket_key(label)]["routing_skip_reasons"][ev.reason] += 1
    for ev in dispatch_denied_events:
        label = schedule.period_for(ev.ts).label if schedule.periods else "day"
        buckets[_bucket_key(label)]["dispatch_denied"] += 1
    return buckets


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_analysis(
    log_dir: Path,
    window_start: datetime,
    window_end: datetime,
    output_dir: Path,
    config: dict | None = None,
) -> AnalysisRun:
    """Discover log files, stream-parse them, aggregate sessions, and write
    the CSVs and report into ``output_dir``."""
    log_dir = Path(log_dir)
    output_dir = Path(output_dir)
    if config is None:
        config = config_loader.load_proxy_config(config_loader.find_config_path())
    schedule = bucketing.schedule_from_config(config, (config or {}).get("session_slot_pool_size"))

    files = log_parser.discover_log_files(log_dir, window_start)
    events = chain.from_iterable(
        log_parser.iter_events(f, window_start, window_end) for f in files
    )
    summary = aggregation.aggregate(events, window_start, window_end, schedule)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csvs(summary, output_dir)
    report_path = output_dir / "report.md"
    report_path.write_text(build_report(summary, config), encoding="utf-8")
    return AnalysisRun(summary=summary, files=files)


def summary_to_json(summary: AnalysisResult) -> dict:
    """Machine-readable summary of the analysis (one dict; JSON-serialisable)."""
    sessions = list(summary.sessions.values())
    local_only = sum(1 for s in sessions if s.remote_requests == 0)
    fell_back = sum(1 for s in sessions if s.fell_back)
    remote_only = sum(1 for s in sessions if s.local_requests == 0 and s.remote_requests > 0)
    total = summary.total_requests
    return {
        "window_start": _fmt_ts(summary.window_start),
        "window_end": _fmt_ts(summary.window_end),
        "sessions": len(sessions),
        "local_only_sessions": local_only,
        "fallback_sessions": fell_back,
        "remote_only_sessions": remote_only,
        "total_requests": total,
        "local_requests": summary.local_requests,
        "remote_requests": summary.remote_requests,
        "fallback_events": len(summary.fallback_events),
        "fallback_rate": round((len(summary.fallback_events) / total) if total else 0.0, 4),
        "dispatch_denied": summary.dispatch_denied_count,
        "unattributed_events": summary.unattributed_events,
        "day_sessions": sum(1 for s in sessions if s.bucket != "night"),
        "night_sessions": sum(1 for s in sessions if s.bucket == "night"),
        "recommendations": len(recommendations.generate_recommendations(summary, None)),
    }
