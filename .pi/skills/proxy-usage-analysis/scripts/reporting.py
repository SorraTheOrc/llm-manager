"""CSV and Markdown report generation, plus the end-to-end analysis runner.

Outputs (per acceptance criteria):

- ``fast_sessions.csv`` — one row per **fast** session (the period(s)
  with the fewest slots per the configured ``slot_schedule`` in the active
  config profile) covering ALL sessions in the window.
- ``cheap_sessions.csv`` — one row per **cheap** session (the period(s)
  with the most slots; produced only when the schedule has differing slot
  counts).
- ``report.md`` — the aggregate Markdown report with highlighted,
  data-backed recommendations.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from pathlib import Path

import aggregation
import bucketing
import config_loader
import llama_log_parser
import log_parser
import recommendations
from aggregation import AnalysisResult, SessionStats
from llama_log_parser import CHEAP, FAST, TOTAL

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
    "decode_tok_s",
]

TS_FMT = "%Y-%m-%d %H:%M:%S"


@dataclass
class AnalysisRun:
    summary: AnalysisResult
    files: list[Path] = field(default_factory=list)
    archived_to: Path | None = None


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
        "decode_tok_s": f"{s.decode_tok_s:.1f}" if s.decode_tok_s is not None else "",
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames or CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_csvs(summary: AnalysisResult, out_dir: Path) -> tuple[Path, Path]:
    """Write the fast and cheap session CSVs; returns their paths."""
    fast_rows, cheap_rows = [], []
    for s in sorted(summary.sessions.values(), key=lambda x: x.start):
        row = _session_row(s)
        (fast_rows if s.bucket != "cheap" else cheap_rows).append(row)
    fast_path = out_dir / "fast_sessions.csv"
    cheap_path = out_dir / "cheap_sessions.csv"
    _write_csv(fast_path, fast_rows)
    _write_csv(cheap_path, cheap_rows)
    return fast_path, cheap_path


# Output artifacts that get archived (moved into a dated subdirectory)
# before a fresh run overwrites them. Anything else in the output dir
# (e.g. cron.log) is left untouched.
ARCHIVE_ARTIFACTS = [
    "report.md",
    "fast_sessions.csv",
    "cheap_sessions.csv",
    "errors.csv",
    "errors.json",
]


def _archive_existing_outputs(output_dir: Path, now: datetime | None = None) -> Path | None:
    """Move existing report artifacts into a dated archive subdirectory.

    Artifacts are named by the run date (``YYYY-MM-DD``); when that directory
    already exists (a same-day repeat, or a manual archive), a ``_2``, ``_3``
    ... suffix is appended so archives are never overwritten. Returns the
    archive directory path, or ``None`` when the output directory contained no
    artifacts to archive (a pristine dir is left untouched). Only the skill's
    own artifacts (``ARCHIVE_ARTIFACTS``) are moved.
    """
    existing = [output_dir / name for name in ARCHIVE_ARTIFACTS if (output_dir / name).exists()]
    if not existing:
        return None
    if now is None:
        now = datetime.now()
    stamp = now.strftime("%Y-%m-%d")
    archive_dir = output_dir / stamp
    n = 2
    while archive_dir.exists():
        archive_dir = output_dir / f"{stamp}_{n}"
        n += 1
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        path.replace(archive_dir / path.name)
    return archive_dir


ERROR_CSV_COLUMNS = [
    "error_type",
    "timestamp",
    "provider",
    "model",
    "session",
    "entry",
    "error_detail",
    "status",
    "attempt",
    "signal",
    "source_file",
    "evidence",
]


ERROR_TYPE_LABELS = {
    "stream_finish_error": "Stream finished: reason=error",
    "stream_error": "Stream error",
    "slot_save_error": "slot_save failed",
    "backend_retry": "backend_retry",
    "upstream_http_error": "upstream HTTP error",
}


def _error_row(e: log_parser.LogEvent) -> dict:
    return {
        "error_type": e.kind,
        "timestamp": _fmt_ts(e.ts),
        "provider": e.provider or "",
        "model": e.model or "",
        "session": e.session or "",
        "entry": e.entry or "",
        "error_detail": e.error or "",
        "status": str(e.status) if e.status is not None else "",
        "attempt": e.attempt or "",
        "signal": e.signal or "",
        "source_file": e.src_file or "",
        "evidence": (e.raw or "").strip(),
    }


# JSON key used when a provider or model is not derivable from the log line.
UNKNOWN_LABEL = "(unknown)"


def error_provider_model_json(summary: AnalysisResult) -> dict:
    """Nested error breakdown ``{error_type: {provider: {model: count}}}``.

    Rows are ordered by count (descending) so the JSON is deterministic for
    consumers. A provider/model that is not derivable from the log line is
    keyed as ``(unknown)`` so the breakdown is self-describing.
    """
    out: dict[str, dict[str, dict[str, int]]] = {}
    for (kind, provider, model), count in summary.error_provider_model_counts.most_common():
        p = provider or UNKNOWN_LABEL
        m = model or UNKNOWN_LABEL
        out.setdefault(kind, {}).setdefault(p, {})[m] = count
    return out


def write_error_artifacts(summary: AnalysisResult, out_dir: Path) -> tuple[Path, Path]:
    """Write ``errors.csv`` (one row per error event) and ``errors.json``
    (aggregated counts by error type); returns their paths."""
    events = sorted(summary.error_events, key=lambda e: e.ts)
    csv_path = out_dir / "errors.csv"
    if events:
        _write_csv(csv_path, [_error_row(e) for e in events], fieldnames=ERROR_CSV_COLUMNS)
    else:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=ERROR_CSV_COLUMNS)
            writer.writeheader()

    json_path = out_dir / "errors.json"
    by_type = dict(summary.error_counts.most_common())
    payload = {
        "total": len(events),
        "by_type": by_type,
        "by_provider_model": error_provider_model_json(summary),
        "window_start": _fmt_ts(summary.window_start),
        "window_end": _fmt_ts(summary.window_end),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return csv_path, json_path


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def build_report(
    summary: AnalysisResult,
    config: dict | None,
    speed: llama_log_parser.SpeedStats | None = None,
) -> str:
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
    ap("| Metric | Total | Fast | Cheap |")
    ap("|---|---|---|---|")
    d, n = profile["fast"], profile["cheap"]
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
    fast_fb = sum(d["fallback_reasons"].values())
    cheap_fb = sum(n["fallback_reasons"].values())
    ap(f"| Fallback events | {len(summary.fallback_events)} ({fallback_rate * 100:.1f}%) | "
       f"{fast_fb} ({_pct(fast_fb, len(summary.fallback_events)):.1f}%) | "
       f"{cheap_fb} ({_pct(cheap_fb, len(summary.fallback_events)):.1f}%) |")
    ap(f"| Queued → dispatched local | {len(summary.contention_dispatch_events)} | "
       f"- | {len(summary.contention_dispatch_events)} |")
    ap(f"| Fallback after queue | {len(summary.contention_fallback_events)} | "
       f"- | {len(summary.contention_fallback_events)} |")
    ap(f"| Dispatch denied | {summary.dispatch_denied_count} | "
       f"{d['dispatch_denied']} ({_pct(d['dispatch_denied'], summary.dispatch_denied_count):.1f}%) | "
       f"{n['dispatch_denied']} ({_pct(n['dispatch_denied'], summary.dispatch_denied_count):.1f}%) |")
    total_avg, total_max = _ctx_stats(
        [s.max_context_size for s in sessions if s.max_context_size is not None]
    )
    fast_avg, fast_max = _ctx_stats(d["ctx"])
    cheap_avg, cheap_max = _ctx_stats(n["ctx"])
    ap(f"| Avg max context | {total_avg} | {fast_avg} | {cheap_avg} |")
    ap(f"| Highest context | {total_max} | {fast_max} | {cheap_max} |")

    if summary.fallback_reason_counts:
        ap("")
        ap("## Fallback reasons")
        ap("")
        ap("| Reason | Total | % of fallbacks | Fast | Cheap |")
        ap("|---|---|---|---|---|")
        for reason, count in summary.fallback_reason_counts.most_common():
            d = profile["fast"]["fallback_reasons"].get(reason, 0)
            n = profile["cheap"]["fallback_reasons"].get(reason, 0)
            ap(f"| {reason} | {count} | {_pct(count, len(summary.fallback_events)):.1f}% | "
               f"{d} ({_pct(d, count):.1f}%) | {n} ({_pct(n, count):.1f}%) |")

    _append_error_section(ap, summary)

    if summary.routing_skip_reason_counts:
        ap("")
        ap("## routing_skip_local reasons")
        ap("")
        ap("| Reason | Total | % of skips | Fast | Cheap |")
        ap("|---|---|---|---|---|")
        for reason, count in summary.routing_skip_reason_counts.most_common():
            d = profile["fast"]["routing_skip_reasons"].get(reason, 0)
            n = profile["cheap"]["routing_skip_reasons"].get(reason, 0)
            ap(f"| {reason} | {count} | {_pct(count, len(summary.routing_skip_events)):.1f}% | "
               f"{d} ({_pct(d, count):.1f}%) | {n} ({_pct(n, count):.1f}%) |")

    initial = Counter((s.initial_provider, s.initial_model) for s in sessions)
    ap("")
    ap("## Per-model breakdown (initial assignment)")
    ap("")
    ap("| Provider | Model | Sessions | Fast | Cheap | Requests | Fell back |")
    ap("|---|---|---|---|---|---|---|")
    for (provider, model), count in initial.most_common():
        s_list = [s for s in sessions if s.initial_provider == provider and s.initial_model == model]
        fast = sum(1 for s in s_list if _bucket_key(s.bucket) == "fast")
        cheap = len(s_list) - fast
        reqs = sum(s.messages for s in s_list)
        fb = sum(1 for s in s_list if s.fell_back)
        ap(f"| {provider} | {model} | {count} | {fast} ({_pct(fast, count):.1f}%) | "
           f"{cheap} ({_pct(cheap, count):.1f}%) | {reqs} | {fb} |")

    _append_speed_section(ap, "Decode speed", "decode", speed)
    _append_speed_section(ap, "Prompt eval speed", "prompt_eval", speed)

    _append_busy_section(ap, summary, schedule)

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
        "fast/cheap bucketing uses the session start time and the slot schedule in the active config profile."
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
    ap(
        "- Decode/prompt-eval speeds come from llama-server eval-timing lines, filtered to the Qwen3 "
        "child port (discovered per log file). llama-server.log lines carry no timestamps, so the "
        "fast/cheap split and window filtering are approximate: each sample is bucketed by its log "
        "file's last-write time. Files whose Qwen3 port cannot be discovered are skipped."
    )
    return "\n".join(lines) + "\n"


def _append_error_section(ap, summary: AnalysisResult) -> None:
    """Append the ``## Error analysis`` section to the report.

    Taxonomy table (error type, count, evidence excerpt) plus a
    ``### Provider/model breakdown`` table (error type × provider × model ×
    count) and a pointer to the remediation recommendations and the
    ``errors.csv`` / ``errors.json`` artifacts. The section is omitted when
    the window has no error events.
    """
    if not summary.error_events:
        return
    ap("")
    ap("## Error analysis")
    ap("")
    ap("| Error type | Count | Evidence excerpt |")
    ap("|---|---|---|")
    counts = summary.error_counts
    for kind, count in counts.most_common():
        first = next((e for e in summary.error_events if e.kind == kind), None)
        excerpt = ""
        if first and first.raw:
            excerpt = first.raw.strip()
            if len(excerpt) > 100:
                excerpt = excerpt[:100] + "…"
        ap(f"| {ERROR_TYPE_LABELS.get(kind, kind)} | {count} | `{excerpt}` |")
    ap("")
    ap("### Provider/model breakdown")
    ap("")
    ap("| Error type | Provider | Model | Count |")
    ap("|---|---|---|---|")
    pm = summary.error_provider_model_counts
    for (kind, provider, model), count in sorted(
        pm.items(),
        key=lambda kv: (-kv[1], kv[0][0], kv[0][1] or "", kv[0][2] or ""),
    ):
        label = ERROR_TYPE_LABELS.get(kind, kind)
        ap(f"| {label} | {provider or '-'} | {model or '-'} | {count} |")
    ap("")
    ap(
        f"- {len(summary.error_events)} error event(s) in window — see `errors.csv` / `errors.json` "
        "and the remediation recommendations below (recovery-first silent continue, informative-error "
        "fallback, ctx-size pressure, upstream 429 cooldown)."
    )


def _fmt_duration(seconds: float) -> str:
    """Format seconds as ``Hh MMm`` (or ``Ns`` when under a minute)."""
    seconds = round(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _busy_cell(busy: float, window: float) -> str:
    return f"{_fmt_duration(busy)} ({busy / window * 100:.1f}%)" if window else f"{_fmt_duration(busy)}"


def _append_busy_section(ap, summary: AnalysisResult, schedule: bucketing.SlotSchedule) -> None:
    """Append the ``## Local model utilization`` section to the report.

    Covers busy/idle time over the window (union of active local streams,
    clipped to the window), total compute (slot-seconds), concurrency, and
    the fast/cheap + hourly busy profile. The section is omitted when the
    window has no local traffic.
    """
    if summary.busy is None:
        return
    b = summary.busy
    ap("")
    ap("## Local model utilization")
    ap("")
    ap("Busy = at least one local slot actively generating (local streams paired "
       "per session, clipped to the window, overlapping streams merged).")
    ap("")
    ap("| Metric | Total | Fast | Cheap |")
    ap("|---|---|---|---|")
    fast_win = b.fast_window_seconds or 0.0
    cheap_win = b.cheap_window_seconds or 0.0
    ap(f"| Busy time | {_busy_cell(b.busy_seconds, b.window_seconds)} | "
       f"{_busy_cell(b.fast_busy_seconds, fast_win)} | {_busy_cell(b.cheap_busy_seconds, cheap_win)} |")
    ap(f"| Idle time | {_busy_cell(b.idle_seconds, b.window_seconds)} | "
       f"{_busy_cell(fast_win - b.fast_busy_seconds, fast_win)} | "
       f"{_busy_cell(cheap_win - b.cheap_busy_seconds, cheap_win)} |")
    ap(f"| Streams served | {b.streams} | - | - |")
    ap(f"| Avg stream duration | {b.avg_stream_duration:.1f}s | - | - |")
    ap(f"| Total compute (slot-time) | {_fmt_duration(b.total_compute_seconds)} | - | - |")
    ap(f"| Avg concurrency (while busy) | {b.avg_concurrency:.2f} | - | - |")
    ap(f"| Peak concurrency | {b.peak_concurrency} | - | - |")
    if b.hourly_busy:
        ap("")
        ap("Busy seconds by hour:")
        ap("")
        ap("| Hour | Busy |")
        ap("|---|---|")
        for hour, seconds in b.hourly_busy:
            ap(f"| {hour:02d}:00-{hour + 1:02d}:00 | {_fmt_duration(seconds)} |")
    if b.unfinished_streams:
        ap("")
        ap(f"_Note: {b.unfinished_streams} local stream(s) started without a logged "
           "`Stream finished` in the available logs (aborted or still running); "
           "their compute time is unknown, so busy time is a conservative lower bound._")
    ap("")
    ap("Method: streams are paired per session (FIFO) across the full log with a "
       f"1h margin beyond the window ({log_parser.BUSY_WINDOW_MARGIN}), then clipped "
       "to the window so boundary-crossing streams are counted exactly; fast/cheap "
       "split follows the slot schedule.")


def _pct(part: int, total: int) -> float:
    return (part / total * 100.0) if total else 0.0


def _speed_cell(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "-"


def _append_speed_section(
    ap,
    title: str,
    kind: str,
    speed: llama_log_parser.SpeedStats | None,
) -> None:
    """Append a speed section (``## Decode speed`` / ``## Prompt eval speed``)
    to the report, one row per (model, bucket) with samples / median / p90 / p10.
    """
    ap("")
    ap(f"## {title}")
    ap("")
    if speed is None:
        ap("_No llama-server eval timing samples in window._")
        return
    buckets = speed.decode if kind == "decode" else speed.prompt_eval
    total = buckets[TOTAL]
    if total.count == 0:
        ap("_No llama-server eval timing samples in window._")
        return
    ap("| Model | Bucket | Samples | Median (tok/s) | p90 (tok/s) | p10 (tok/s) |")
    ap("|---|---|---|---|---|---|")
    for bucket_key in (TOTAL, FAST, CHEAP):
        b = buckets[bucket_key]
        if b.count == 0:
            continue
        # All samples are from the single local model (Qwen3); keep the model
        # name generic so a future second local model renders per-model rows.
        label = {"total": "Total", "fast": "Fast", "cheap": "Cheap"}[bucket_key]
        ap(f"| Qwen3 | {label} | {b.count} | {_speed_cell(b.median)} | "
           f"{_speed_cell(b.p90)} | {_speed_cell(b.p10)} |")
    if speed.files_skipped:
        ap("")
        ap(f"_Note: {speed.files_skipped} llama-server log file(s) skipped "
           "(Qwen3 child port not found)._")


def _bucket_key(bucket: str | None) -> str:
    return "cheap" if bucket == "cheap" else "fast"


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
    """Per-bucket (fast/cheap) totals for the report's summary tables.

    Covers sessions, requests, local/remote split, classification counts,
    context sizes, dispatch denials, and per-reason counters. Events without a
    session (fallbacks, routing skips, dispatch denials) are bucketed by their
    own timestamp.
    """
    buckets = {
        "fast": {
            "sessions": 0, "requests": 0, "local": 0, "remote": 0,
            "local_only": 0, "fell_back": 0, "remote_only": 0,
            "dispatch_denied": 0, "ctx": [],
            "fallback_reasons": Counter(), "routing_skip_reasons": Counter(),
        },
        "cheap": {
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
        label = schedule.period_for(ev.ts).label if schedule.periods else "fast"
        buckets[_bucket_key(label)]["fallback_reasons"][ev.reason] += 1
    for ev in routing_skip_events:
        if not ev.reason:
            continue
        label = schedule.period_for(ev.ts).label if schedule.periods else "fast"
        buckets[_bucket_key(label)]["routing_skip_reasons"][ev.reason] += 1
    for ev in dispatch_denied_events:
        label = schedule.period_for(ev.ts).label if schedule.periods else "fast"
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
    llama_log_dir: Path | None = None,
) -> AnalysisRun:
    """Discover log files, stream-parse them, aggregate sessions, and write
    the CSVs and report into ``output_dir``.

    ``llama_log_dir`` defaults to ``log_dir`` (the proxy and llama-server
    logs live in the same directory). llama-server eval-timing samples are
    parsed for the decode/prompt-eval speed sections; missing or unparseable
    llama-server files are skipped, never fatal.
    """
    log_dir = Path(log_dir)
    output_dir = Path(output_dir)
    if llama_log_dir is None:
        llama_log_dir = log_dir
    if config is None:
        config = config_loader.load_proxy_config(config_loader.find_config_path())
    schedule = bucketing.schedule_from_config(config, (config or {}).get("session_slot_pool_size"))

    files = log_parser.discover_log_files(log_dir, window_start)
    events = chain.from_iterable(
        log_parser.iter_events(
            f, window_start, window_end, margin=log_parser.BUSY_WINDOW_MARGIN
        )
        for f in files
    )
    summary = aggregation.aggregate(events, window_start, window_end, schedule)

    llama_files = llama_log_parser.discover_llama_logs(llama_log_dir, window_start)
    summary.speed = llama_log_parser.build_speed_stats(
        llama_files, window_start, window_end, schedule
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    archived_to = _archive_existing_outputs(output_dir)
    write_csvs(summary, output_dir)
    write_error_artifacts(summary, output_dir)
    report_path = output_dir / "report.md"
    report_path.write_text(build_report(summary, config, summary.speed), encoding="utf-8")
    return AnalysisRun(summary=summary, files=files, archived_to=archived_to)


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
        "contention_dispatch": len(summary.contention_dispatch_events),
        "contention_fallback_after_queue": len(summary.contention_fallback_events),
        "contention_queued_duration_seconds": round(
            sum(float(e.queued_duration or 0.0) for e in summary.contention_dispatch_events)
            + sum(float(e.queued_duration or 0.0) for e in summary.contention_fallback_events),
            3,
        ),
        "dispatch_denied": summary.dispatch_denied_count,
        "unattributed_events": summary.unattributed_events,
        "fast_sessions": sum(1 for s in sessions if s.bucket != "cheap"),
        "cheap_sessions": sum(1 for s in sessions if s.bucket == "cheap"),
        "errors": len(summary.error_events),
        "errors_by_type": dict(summary.error_counts.most_common()),
        "errors_by_provider_model": error_provider_model_json(summary),
        "recommendations": len(recommendations.generate_recommendations(summary, None)),
        "local_busy": _busy_json(summary.busy),
        "decode_speed": _speed_json(summary.speed) if summary.speed else None,
        "prompt_eval_speed": _speed_json(summary.speed, "prompt_eval") if summary.speed else None,
    }


def _busy_json(busy: aggregation.BusyStats | None) -> dict | None:
    """JSON-friendly local-model utilization summary (None when no local traffic)."""
    if busy is None:
        return None
    return {
        "window_seconds": busy.window_seconds,
        "busy_seconds": busy.busy_seconds,
        "busy_pct": round(busy.busy_pct, 1),
        "idle_seconds": round(busy.idle_seconds, 1),
        "idle_pct": round(busy.idle_pct, 1),
        "total_compute_seconds": busy.total_compute_seconds,
        "streams": busy.streams,
        "avg_stream_duration_seconds": busy.avg_stream_duration,
        "peak_concurrency": busy.peak_concurrency,
        "avg_concurrency": busy.avg_concurrency,
        "unfinished_streams": busy.unfinished_streams,
        "fast_busy_seconds": busy.fast_busy_seconds,
        "cheap_busy_seconds": busy.cheap_busy_seconds,
        "fast_window_seconds": busy.fast_window_seconds,
        "cheap_window_seconds": busy.cheap_window_seconds,
        "hourly_busy": busy.hourly_busy,
    }


def _speed_json(speed: llama_log_parser.SpeedStats, kind: str = "decode") -> dict:
    """JSON-friendly speed summary (total bucket only)."""
    buckets = speed.decode if kind == "decode" else speed.prompt_eval
    b = buckets[TOTAL]
    return {
        "samples": b.count,
        "median_tok_s": b.median,
        "p90_tok_s": b.p90,
        "p10_tok_s": b.p10,
        "files_parsed": speed.files_parsed,
        "files_skipped": speed.files_skipped,
    }
