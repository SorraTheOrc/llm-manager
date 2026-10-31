
"""CSV and Markdown report generation, plus the end-to-end analysis runner.

Outputs (per acceptance criteria):

- ``standard_sessions.csv`` — one row per **standard** session (started
  outside the configured priority window) covering ALL sessions in the
  window.
- ``low_priority_sessions.csv`` — one row per **low** session (started
  inside the cheap priority window; LP-0MVDT94F0003VHUA AC6).
- ``report.md`` — the aggregate Markdown report with highlighted,
  data-backed recommendations.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from itertools import chain
from pathlib import Path

import aggregation
import bucketing
import config_loader
import gpu_busy
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
    "fallback_attribution",
    "bucket",
    "slots",
    "ctx_size",
    "local_requests",
    "remote_requests",
    "local_turns_answered",
    "local_turns_unanswered",
    "local_turn_pct",
    "context_at_fallback",
    "context_at_fallback_source",
    "dispatch_denied",
    "decode_tok_s",
    # Measured-usage coverage per session (LP-0MVMBM6N400541EH AC3): the
    # per-session record of how many terminal lines reported no usage, so a
    # blank context column is explainable rather than looking like a bug.
    "streams_with_usage",
    "streams_without_usage",
]

TS_FMT = "%Y-%m-%d %H:%M:%S"

# --- Contention-prediction measurement (LP-0MVGR2OXT0096OGK) --------------
#
# The item measures whether the payload's ``lease_state`` /
# ``prefill_eta_seconds`` signals predict real local contention:
#
# * M1 — per-hour denial attribution by ``local_admission_guard`` reason plus
#   the guard's measured ``busy_slots`` / ``in_flight_prefills`` / ``free_slot``
#   (the denial line itself carries no ``lease_state``), reduced to an
#   ``idle_leased`` / ``prefill_bound`` / ``saturated`` / ``unknown`` verdict.
# * M2 — the window's local busy fraction beside those denials (already
#   computed by :class:`aggregation.BusyStats`; rendered here for comparison).
# * M3 — predicted-versus-observed prefill duration at p50/p90, comparing the
#   ``prefill_eta_seconds`` published on ``status_request`` polls against the
#   llama-server ``prompt eval time`` samples for dispatches that ran.
#
# The baseline these are compared against (measured before the payload change,
# from this item's description):
CONTENTION_BASELINE = {
    "window": "2026-10-19 07:23 → 2026-10-20 07:23",
    "denials": 2641,
    "hours": 24.0,
    "busy_pct": 29.3,
    "worst_hour": "2026-10-20 04:00–05:00",
    "worst_hour_denials": 187,
    "worst_hour_busy_pct": 2.5,
    "idle_lease_records": 2633,
    "source": "LP-0MVGR2OXT0096OGK description (pre-change baseline)",
}

# M3 matched per-dispatch tolerance: the guard's priced ETA for a dispatch is
# called a match when the median relative error against that same dispatch's
# observed first-byte time is within this percentage (two-sided). It applies
# ONLY to matched pairs (LP-0MVSD5RJW008F1IB AC1/AC3); the cross-population
# "bound vs served remainder" comparison is a one-sided bound test below.
PREFILL_ETA_MATCH_TOLERANCE_PCT = 25.0

# M3 bound test (LP-0MVSD5RJW008F1IB AC3): below this many observed dispatches
# the window cannot decide whether the published cache-cold bound held, so the
# verdict is ``inconclusive`` rather than a verdict on noise.
PREFILL_ETA_BOUND_MIN_SAMPLES = 30

# M1 verdict threshold: the share of attributed denials whose decision block
# measured zero local work (``idle_leased``) above which the lease/busy signal
# is judged NOT to reflect reality.
IDLE_LEASED_SHARE_THRESHOLD_PCT = 50.0


@dataclass
class AnalysisRun:
    summary: AnalysisResult
    files: list[Path] = field(default_factory=list)
    archived_to: Path | None = None
    # Fast/cheap mode schedule map (per-mode profiles + mode timeline from the
    # logs); None when not built (e.g. no config discovered).
    mode_map: object | None = None


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
        "fallback_attribution": s.fallback_attribution or "",
        "bucket": s.bucket or "",
        "slots": str(s.slots) if s.slots else "",
        "ctx_size": str(s.ctx_size) if s.ctx_size else "",
        "local_requests": str(s.local_requests),
        "remote_requests": str(s.remote_requests),
        # Local-turn coverage (LP-0MVM93DE6009JGBU AC5): the per-session data
        # behind the report's "Local model usage by session" section, so the
        # numbers are auditable from an artifact instead of the Markdown only.
        "local_turns_answered": str(s.local_turns_answered),
        "local_turns_unanswered": str(s.local_turns_unanswered),
        "local_turn_pct": (
            f"{s.local_turn_pct:.1f}" if s.local_turn_pct is not None else ""
        ),
        "context_at_fallback": (
            str(s.context_at_fallback)
            if s.context_at_fallback is not None
            else ""
        ),
        "context_at_fallback_source": s.context_at_fallback_source or "",
        "dispatch_denied": str(s.dispatch_denied),
        "decode_tok_s": f"{s.decode_tok_s:.1f}" if s.decode_tok_s is not None else "",
        "streams_with_usage": str(getattr(s, "streams_with_usage", 0)),
        "streams_without_usage": str(getattr(s, "streams_without_usage", 0)),
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames or CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_csvs(summary: AnalysisResult, out_dir: Path) -> tuple[Path, Path]:
    """Write the standard and low-priority session CSVs; returns their paths."""
    standard_rows, low_rows = [], []
    for s in sorted(summary.sessions.values(), key=lambda x: x.start):
        row = _session_row(s)
        (standard_rows if s.bucket != "cheap" else low_rows).append(row)
    standard_path = out_dir / "standard_sessions.csv"
    low_path = out_dir / "low_priority_sessions.csv"
    _write_csv(standard_path, standard_rows)
    _write_csv(low_path, low_rows)
    return standard_path, low_path


# Output artifacts that get archived (moved into a dated subdirectory)
# before a fresh run overwrites them. Anything else in the output dir
# (e.g. cron.log) is left untouched.
ARCHIVE_ARTIFACTS = [
    "report.md",
    "standard_sessions.csv",
    "low_priority_sessions.csv",
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
    "root_cause",
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
    # Enriched stream-error payload (LP-0MT6322OT00900OX): fold the
    # error_type / message / suggested_action into the detail cell so the
    # CSV/JSON artifacts surface the informative-error coverage.
    detail = e.error or ""
    if e.kind == "stream_finish_error" and e.error_type:
        detail = e.error_type
        if e.error_message:
            detail += f": {e.error_message}"
        if e.suggested_action:
            detail += f" ({e.suggested_action})"
    return {
        "error_type": e.kind,
        "root_cause": e.root_cause or "",
        "timestamp": _fmt_ts(e.ts),
        "provider": e.provider or "",
        "model": e.model or "",
        "session": e.session or "",
        "entry": e.entry or "",
        "error_detail": detail,
        "status": str(e.status) if e.status is not None else "",
        "attempt": e.attempt or "",
        "signal": e.signal or "",
        "source_file": e.src_file or "",
        "evidence": (e.raw or "").strip(),
    }


# JSON key used when a provider or model is not derivable from the log line.
UNKNOWN_LABEL = "(unknown)"

# Bucket label for an unmeasured residency decision whose line predates the
# additive ``measurement=`` field (LP-0MVT2QHMU0008F2V): old logs are not
# reclassified, and "no provenance recorded" is distinct from a reason the
# proxy actually emitted.
UNRECORDED_MEASUREMENT_LABEL = "unrecorded"


def _residency_measurement_breakdown(
    summary: AnalysisResult,
) -> tuple[int, Counter]:
    """Return ``(measured_count, unmeasured_reason_counts)`` for residency.

    The single source of truth behind both the rendered breakdown
    (``_append_residency_section``) and the machine-readable keys in
    ``summary_to_json`` (LP-0MVT2QHMU0008F2V); the two can never disagree.

    A decision is *measured* when it carries a numeric ``reuse_ratio``. An
    unmeasured decision is bucketed by its parsed ``measurement=`` value
    verbatim (``unavailable:no_slot``, ...), or ``unrecorded`` when the line
    predates the field — never fabricated into a measurement.
    """
    events = list(getattr(summary, "residency_events", []) or [])
    measured_count = sum(1 for ev in events if ev.reuse_ratio is not None)
    reasons = Counter(
        ev.measurement or UNRECORDED_MEASUREMENT_LABEL
        for ev in events
        if ev.reuse_ratio is None
    )
    return measured_count, reasons


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


def root_cause_json(summary: AnalysisResult) -> dict:
    """Root-cause breakdown ``{error_type: {root_cause: count}}`` for stream
    errors (LP-0MUQO82LJ008F1FS).

    Only stream errors carry a root cause; kinds with no classified event are
    omitted. Counts are data-driven (no hardcoded totals) and the inner keys
    are sorted so the JSON is deterministic.
    """
    counts = summary.root_cause_counts
    return {
        kind: {rc: n for rc, n in sorted(inner.items())}
        for kind, inner in sorted(counts.items())
    }


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
    by_status = {
        str(status): count
        for status, count in sorted(summary.upstream_error_by_status.items())
    }
    by_status_provider = {
        str(status): {
            provider: count
            for provider, count in sorted(providers.items(), key=lambda kv: -kv[1])
        }
        for status, providers in sorted(summary.upstream_error_by_status_provider.items())
    }
    payload = {
        "total": len(events),
        "by_type": by_type,
        "by_provider_model": error_provider_model_json(summary),
        "by_root_cause": root_cause_json(summary),
        "upstream_by_status": by_status,
        "upstream_by_status_provider": by_status_provider,
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


# ---------------------------------------------------------------------------
# Local-turn coverage (LP-0MVM93DE6009JGBU)
# ---------------------------------------------------------------------------
#
# "Which sessions actually used the local model, how many turns did it answer,
# what share of the session's turns were local, and why did each session move
# to remote?" Everything below is derived from data the pipeline already
# collects (per-session turns and fallback attribution), so the section needs no
# new data source:
#
# - a session is *covered* when it has at least one local turn (whether or not
#   that turn was answered — an unanswered turn is flagged, never dropped);
# - ``% local turns`` = local / (local + remote) × 100, 100 % for a session
#   that never used remote;
# - the summary table's percentile rows use linear interpolation (the shared
#   ``_percentile`` convention) over the covered sessions; the total row sums
#   the turn columns and recomputes the share from those sums, and shows the
#   mean of the sessions that have token data for the context column;
# - the reason distribution is its own counts-and-percent table because a
#   reason is categorical and has no meaningful percentile;
# - ``context size at fallback`` is the session's own evidence (see
#   ``aggregation.CONTEXT_AT_FALLBACK_*``), rendered ``n/a`` — with a named
#   limitation — when the logs carry no tokens at all.
LOCAL_TURN_PERCENTILES = ((10, "P10"), (50, "P50"), (90, "P90"))


@dataclass(frozen=True)
class LocalTurnRow:
    """One local-using session's coverage row (report + JSON)."""

    session_id: str
    local_turns: int
    answered_turns: int
    unanswered_turns: int
    remote_turns: int
    local_turn_pct: float
    reason: str
    context_at_fallback: int | None
    context_source: str | None
    # Terminal streams in this session that reported no usage
    # (LP-0MVMBM6N400541EH AC3): the concrete reason a session's
    # ``context size at fallback`` can read ``n/a`` — those lines carry
    # ``tokens=missing``, so they contribute no prompt tokens to any figure.
    streams_without_usage: int = 0


NO_LOCAL_TURNS_TEXT = "No session in the window used the local model."


def _fallback_reason_label(session: SessionStats) -> str:
    """Reason label for the per-session coverage tables.

    The session's own attributed ``fallback_reason`` when it has one; otherwise
    its attribution state, so a session that never left local reads
    ``no_remote_fallback`` and one with no own evidence reads ``unattributed``
    (LP-0MVMBMSWR0036RES — a reason is never inherited from a neighbour).
    """
    if session.fallback_reason:
        return session.fallback_reason
    if session.fallback_attribution:
        return session.fallback_attribution
    if session.remote_requests == 0:
        return aggregation.FALLBACK_ATTRIBUTION_NO_REMOTE
    return aggregation.FALLBACK_ATTRIBUTION_UNATTRIBUTED


def local_turn_rows(
    sessions: dict[str, SessionStats] | list[SessionStats],
) -> list[LocalTurnRow]:
    """Coverage rows for the sessions with at least one local turn.

    Sorted by ``# turns local`` **descending** (more local turns is the goal,
    so the best sessions read first), then by session id so equal counts keep a
    deterministic order.
    """
    candidates = sessions.values() if isinstance(sessions, dict) else sessions
    rows = [
        LocalTurnRow(
            session_id=s.session_id,
            local_turns=s.local_requests,
            answered_turns=s.local_turns_answered,
            unanswered_turns=s.local_turns_unanswered,
            remote_turns=s.remote_requests,
            local_turn_pct=s.local_turn_pct
            if s.local_turn_pct is not None
            else 0.0,
            reason=_fallback_reason_label(s),
            context_at_fallback=s.context_at_fallback,
            context_source=s.context_at_fallback_source,
            streams_without_usage=getattr(s, "streams_without_usage", 0),
        )
        for s in candidates
        if s.local_requests > 0
    ]
    rows.sort(key=lambda r: (-r.local_turns, r.session_id))
    return rows


def local_turn_coverage(summary: AnalysisResult) -> dict:
    """Machine-readable local-turn coverage (shared by the report and JSON).

    Returns the covered rows, the percentile/total table rows and the
    fallback-reason counts-and-percent rows, so the Markdown section and the
    ``--json`` summary can never disagree.
    """
    rows = local_turn_rows(summary.sessions)
    with_local = len(rows)
    local_values = [float(r.local_turns) for r in rows]
    remote_values = [float(r.remote_turns) for r in rows]
    pct_values = [r.local_turn_pct for r in rows]
    context_values = [
        float(r.context_at_fallback)
        for r in rows
        if r.context_at_fallback is not None
    ]

    percentiles: list[dict] = []
    for percentile, label in LOCAL_TURN_PERCENTILES:
        context = _percentile(context_values, percentile)
        percentiles.append(
            {
                "label": label,
                "local_turns": _round1(_percentile(local_values, percentile)),
                "remote_turns": _round1(_percentile(remote_values, percentile)),
                "local_turn_pct": _round1(_percentile(pct_values, percentile)),
                "context_at_fallback": (
                    int(round(context)) if context is not None else None
                ),
            }
        )

    total_local = sum(r.local_turns for r in rows)
    total_remote = sum(r.remote_turns for r in rows)
    total_turns = total_local + total_remote
    total = {
        "local_turns": total_local,
        "remote_turns": total_remote,
        "local_turn_pct": round(total_local / total_turns * 100.0, 1)
        if total_turns
        else None,
        "context_at_fallback": (
            int(round(sum(context_values) / len(context_values)))
            if context_values
            else None
        ),
        "answered_turns": sum(r.answered_turns for r in rows),
        "unanswered_turns": sum(r.unanswered_turns for r in rows),
        "context_samples": len(context_values),
    }

    reason_counts = Counter(r.reason for r in rows)
    reasons = [
        {
            "reason": reason,
            "sessions": count,
            "pct_of_sessions_with_local": round(count / with_local * 100.0, 1)
            if with_local
            else 0.0,
        }
        for reason, count in sorted(
            reason_counts.items(), key=lambda kv: (-kv[1], kv[0])
        )
    ]

    # Interpretation notes (deterministic, data-gated). These are counted over
    # ALL sessions in the window, not just the covered rows: the decision-first
    # finding is precisely about the sessions that reached remote *without* a
    # local turn, so they can never appear in the per-session table.
    all_sessions = list(summary.sessions.values())
    reached_remote = [s for s in all_sessions if s.remote_requests > 0]
    declined_before_local = [s for s in reached_remote if s.local_requests == 0]
    context_fallbacks = [
        s
        for s in all_sessions
        if _fallback_reason_label(s) in aggregation.CONTEXT_FALLBACK_REASONS
    ]
    context_before_local = [s for s in context_fallbacks if s.local_requests == 0]
    budget_exhausted = [
        s for s in all_sessions if log_parser.is_provider_budget_reason(_fallback_reason_label(s))
    ]

    return {
        "sessions_in_window": len(summary.sessions),
        "sessions_with_local": with_local,
        # Sessions whose terminal streams reported no usage
        # (LP-0MVMBM6N400541EH): the concrete cause of a context-at-fallback
        # figure that has to render ``n/a``.
        "sessions_without_usage": sum(
            1 for r in rows if r.streams_without_usage
        ),
        "percentiles": percentiles,
        "total": total,
        "reasons": reasons,
        "sessions": [
            {
                "session_id": r.session_id,
                "local_turns": r.local_turns,
                "answered_turns": r.answered_turns,
                "unanswered_turns": r.unanswered_turns,
                "remote_turns": r.remote_turns,
                "local_turn_pct": r.local_turn_pct,
                "reason": r.reason,
                "context_at_fallback": r.context_at_fallback,
                "context_at_fallback_source": r.context_source,
                # Per-session no-usage count (LP-0MVMBM6N400541EH AC3): names
                # why a session's context-at-fallback figure is ``n/a``.
                "streams_without_usage": r.streams_without_usage,
            }
            for r in rows
        ],
        "notes": {
            "sessions_reaching_remote": len(reached_remote),
            "declined_before_local_turn": len(declined_before_local),
            "context_fallback_sessions": len(context_fallbacks),
            "context_fallback_before_local_turn": len(context_before_local),
            "provider_budget_exhausted_sessions": len(budget_exhausted),
        },
    }


def _round1(value: float | None) -> float | None:
    return round(value, 1) if value is not None else None


def _fmt_count(value: float | None) -> str:
    """Render a percentile figure: integers without a decimal, else one place."""
    if value is None:
        return "-"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _fmt_fallback_tokens(value: int | None) -> str:
    """Render a context-at-fallback figure, or an explicit ``n/a``."""
    return "n/a" if value is None else f"{value:,}"


def _append_local_turn_coverage_section(
    ap, summary: AnalysisResult
) -> None:
    """Append the ``## Local model usage by session`` section.

    Three deterministic tables (summary percentiles, fallback-reason counts,
    per-session sampling) plus data-gated interpretation notes. Renders a named
    no-local-traffic line instead of an empty section when no session used the
    local model (LP-0MVM93DE6009JGBU AC1–AC4).
    """
    coverage = local_turn_coverage(summary)
    ap("")
    ap("## Local model usage by session")
    ap("")
    if not coverage["sessions_with_local"]:
        ap(NO_LOCAL_TURNS_TEXT)
        return

    total = coverage["total"]
    total_turns = total["local_turns"] + total["remote_turns"]
    ap(
        "Sessions with at least one local turn (a local `Stream started` paired "
        "with a local `Stream finished`): "
        f"**{coverage['sessions_with_local']} of "
        f"{coverage['sessions_in_window']}** sessions "
        f"({_pct(coverage['sessions_with_local'], coverage['sessions_in_window']):.1f}%), "
        f"**{total['local_turns']}** local turns "
        f"({total['answered_turns']} answered, {total['unanswered_turns']} unanswered) "
        f"and {total['remote_turns']} remote turns "
        f"({_pct(total['local_turns'], total_turns):.1f}% local overall)."
    )
    ap("")
    ap("| Percentile | # turns local | # turns remote | % local turns | Context size at fallback |")
    ap("|---|---|---|---|---|")
    for row in coverage["percentiles"]:
        ap(
            f"| {row['label']} | {_fmt_count(row['local_turns'])} | "
            f"{_fmt_count(row['remote_turns'])} | "
            f"{_fmt_count(row['local_turn_pct'])}% | "
            f"{_fmt_fallback_tokens(row['context_at_fallback'])} |"
        )
    total_row = (
        f"| Total | {total['local_turns']} | {total['remote_turns']} | "
        f"{_fmt_count(total['local_turn_pct'])}% | "
        f"{_fmt_fallback_tokens(total['context_at_fallback'])} |"
    )
    ap(total_row)
    ap("")
    ap(
        "P10/P50/P90 are percentiles (linear interpolation) over the sessions "
        "with ≥1 local turn. The total row sums the turn columns, recomputes "
        "`% local turns` from those sums, and shows the mean context at "
        "fallback of the sessions with token data. Context size at fallback "
        "covers the "
        f"{total['context_samples']} of {coverage['sessions_with_local']} sessions "
        "with token evidence at their fallback; sessions with none are "
        "excluded from that column's percentiles rather than counted as 0."
    )

    ap("")
    ap("### Fallback reason distribution")
    ap("")
    ap("| Reason | Sessions | % of sessions-with-local |")
    ap("|---|---|---|")
    for row in coverage["reasons"]:
        ap(
            f"| {row['reason']} | {row['sessions']} | "
            f"{row['pct_of_sessions_with_local']:.1f}% |"
        )
    ap("")
    ap(
        "`no_remote_fallback` marks a session that never used a remote provider; "
        "`unattributed` means it reached remote with no own evidence "
        "(LP-0MVMBMSWR0036RES — a session never inherits a neighbour's reason)."
    )

    ap("")
    ap("### Per-session coverage")
    ap("")
    ap(
        "| Session ID | # turns local | # turns remote | % local turns | "
        "Reason for remote fallback | Context size at fallback | "
        "Streams without usage |"
    )
    ap("|---|---|---|---|---|---|---|")
    for row in coverage["sessions"]:
        local_cell = str(row["local_turns"])
        if row["unanswered_turns"]:
            local_cell += f" ({row['unanswered_turns']} unanswered)"
        ap(
            f"| {row['session_id']} | {local_cell} | {row['remote_turns']} | "
            f"{row['local_turn_pct']:.1f}% | {row['reason']} | "
            f"{_fmt_fallback_tokens(row['context_at_fallback'])} | "
            f"{row.get('streams_without_usage', 0)} |"
        )
    ap("")
    ap(
        "Sorted by `# turns local` descending (more local turns is the goal), "
        "then by session id. `% local turns` = local / (local + remote) × 100 "
        "for that session; an unanswered local turn is counted and flagged in "
        "the local column rather than dropped. `Streams without usage` counts "
        "the session's terminal lines that reported no usage — they carry "
        "`tokens=missing` (LP-0MVMBM6N400541EH) — which is why its context "
        "figures can be `n/a`."
    )

    notes = coverage["notes"]
    if notes["sessions_reaching_remote"]:
        ap("")
        ap("Interpretation:")
        ap(
            "- Decision-first routing: "
            f"{notes['declined_before_local_turn']} of "
            f"{notes['sessions_reaching_remote']} sessions that reached remote "
            f"never served a local turn "
            f"({_pct(notes['declined_before_local_turn'], notes['sessions_reaching_remote']):.1f}%) "
            "— local is declined pre-emptively, not after a failed attempt."
        )
        if notes["context_fallback_sessions"]:
            ap(
                "- Context-size fallbacks "
                "(`context_too_large` / `large_context_bypass` / "
                "`provider_context_exceeded`): "
                f"{notes['context_fallback_before_local_turn']} of "
                f"{notes['context_fallback_sessions']} were declined before the "
                "session's first local turn. These are `estimated_tokens` "
                "threshold decisions, so they fire only once a prompt grows past "
                "the configured cold/warm thresholds — never on a session's first "
                "message."
            )
        if notes["provider_budget_exhausted_sessions"]:
            ap(
                "- `provider_budget_exhausted` is the proxy's own per-provider "
                "RPM budget pacing (`provider_budget.py`: the configured "
                "`rate_limit_rpm` window for that provider is exhausted, so the "
                "request passes to the next provider immediately) — not an "
                "upstream HTTP 429; those surface as "
                "`upstream_http_error status=429` events instead."
            )

    missing = coverage["sessions_with_local"] - coverage["total"]["context_samples"]
    if missing:
        # LP-0MVMBM6N400541EH AC4: when the affected sessions actually logged
        # usage-less streams, name the marker as the cause. A session can also
        # have no context figure simply because it never fell back (no
        # ``routing_skip_local`` and no remote stream), so the marker is only
        # named when the evidence shows it.
        marker_note = ""
        if coverage["sessions_without_usage"]:
            marker_note = (
                " Terminal streams that reported no usage carry the explicit "
                "`tokens=missing` marker (LP-0MVMBM6N400541EH) and are counted "
                "in the `Streams without usage` column, so their `n/a` is "
                "explained by the logs rather than being an unexplained gap."
            )
        ap("")
        ap(
            f"**Limitation:** {missing} of {coverage['sessions_with_local']} "
            "session(s) have no token evidence at their fallback — their "
            "`routing_skip_local` line carried no `estimated_tokens=`, no remote "
            "stream reported usage and the session itself logged no prompt "
            "tokens." + marker_note + " They render `n/a`; the report never "
            "fabricates an estimate."
        )


def _ap_fallback_attribution(ap, summary: AnalysisResult) -> None:
    """Render the per-session fallback-attribution states (LP-0MVMBMSWR0036RES).

    States are mutually exclusive and cover every session: own evidence
    (``own_fallback``/``own_skip``), the legacy nearest-line heuristic that only
    session-less rotated logs can reach (``legacy_nearest``), a lease/concurrency
    refusal (``lease_denied``), no own evidence at all (``unattributed``), and
    sessions that never left local (``no_remote_fallback``).
    """
    counts = summary.fallback_attribution_counts
    if not counts:
        return
    parts = [
        f"{state} **{counts.get(state, 0)}**"
        for state in aggregation.FALLBACK_ATTRIBUTION_ORDER
        if counts.get(state, 0)
    ]
    ap("- Session fallback attribution: " + " | ".join(parts))


def build_report(
    summary: AnalysisResult,
    config: dict | None,
    speed: llama_log_parser.SpeedStats | None = None,
    mode_map: bucketing.ModeScheduleMap | None = None,
    profiles: dict | None = None,
) -> str:
    recs = recommendations.generate_recommendations(summary, config, mode_map)
    hours = (summary.window_end - summary.window_start).total_seconds() / 3600.0
    sessions = list(summary.sessions.values())
    total = summary.total_requests

    local_only = [s for s in sessions if s.remote_requests == 0]
    fell_back = [s for s in sessions if s.fell_back]
    remote_only = [s for s in sessions if s.local_requests == 0 and s.remote_requests > 0]

    fallback_rate = (len(summary.fallback_events) / total) if total else 0.0
    # Prefill-ETA admission-guard tallies (LP-0MVGDUV5E005UWBQ AC2).
    guard_events = getattr(summary, "local_admission_events", []) or []
    local_admission_rejects = sum(
        1 for e in guard_events if getattr(e, "decision", None) == "reject"
    )
    local_admission_admits = sum(
        1 for e in guard_events if getattr(e, "decision", None) == "admit"
    )
    schedule = bucketing.schedule_from_config(
        config, (config or {}).get("session_slot_pool_size")
    )
    profile = _bucket_profile(
        sessions,
        schedule,
        summary.fallback_events,
        summary.routing_skip_events,
        summary.dispatch_denied_events,
        mode_map,
        summary.provider_context_skip_events,
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
    ap(f"- Local-skip events: **{len(summary.routing_skip_events)}** | "
       f"hold-cycle re-emissions (excluded from demand): **{summary.hold_cycle_reemission_count}**")
    ap(f"- Provider context-ceiling skips: **{len(summary.provider_context_skip_events)}** "
       "(per-provider skip counted, not a provider failure)")
    ap(f"- Unattributed stream events (no session UUID): {summary.unattributed_events}")
    ap(f"- Lines parsed: {summary.total_lines} | lines skipped: {summary.lines_skipped}")
    _ap_fallback_attribution(ap, summary)

    _append_summary_by_hour(ap, summary)

    ap("")
    ap("## Session summary")
    ap("")
    ap("| Metric | Total | Standard | Low |")
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
       f"{fast_fb} ({_pct(fast_fb, d['requests']):.1f}%) | "
       f"{cheap_fb} ({_pct(cheap_fb, n['requests']):.1f}%) |")
    ap(f"| Queued → dispatched local | {len(summary.contention_dispatch_events)} | "
       f"- | {len(summary.contention_dispatch_events)} |")
    ap(f"| Fallback after queue | {len(summary.contention_fallback_events)} | "
       f"- | {len(summary.contention_fallback_events)} |")
    ap(f"| Local admission refused (prefill ETA) | {local_admission_rejects} | "
       f"- | {local_admission_rejects} |")
    ap(f"| Local admission admitted (prefill ETA) | {local_admission_admits} | "
       f"- | {local_admission_admits} |")
    # Idle-aware local admission (LP-0MVGD9Z8Z0055Y5J): economic cold-cache
    # bypasses overridden because the GPU had been idle for at least the
    # configured threshold. Local-only by construction (it *is* a local serve),
    # so it sits in the local column.
    idle_admits = len(getattr(summary, "idle_admit_events", []) or [])
    ap(f"| Local admitted while GPU idle (cold-cache override) | {idle_admits} | "
       f"- | {idle_admits} |")
    ap(f"| Dispatch denied | {summary.dispatch_denied_count} | "
       f"{d['dispatch_denied']} ({_pct(d['dispatch_denied'], summary.dispatch_denied_count):.1f}%) | "
       f"{n['dispatch_denied']} ({_pct(n['dispatch_denied'], summary.dispatch_denied_count):.1f}%) |")
    _append_compaction_section(ap, summary, config, profiles)
    total_avg, total_max = _ctx_stats(
        [s.max_context_size for s in sessions if s.max_context_size is not None]
    )
    fast_avg, fast_max = _ctx_stats(d["ctx"])
    cheap_avg, cheap_max = _ctx_stats(n["ctx"])
    ap(f"| Avg max context | {total_avg} | {fast_avg} | {cheap_avg} |")
    ap(f"| Highest context | {total_max} | {fast_max} | {cheap_max} |")

    _append_total_tokens_section(ap, summary)

    # Local-turn coverage (LP-0MVM93DE6009JGBU): which sessions used local, how
    # many turns it answered, the per-session share and the reason each session
    # moved to remote — read straight from the session aggregates above.
    _append_local_turn_coverage_section(ap, summary)

    if summary.fallback_reason_counts:
        ap("")
        ap("## Fallback reasons")
        ap("")
        ap("| Reason | Total | % of fallbacks | Standard | Low |")
        ap("|---|---|---|---|---|")
        for reason, count in summary.fallback_reason_counts.most_common():
            d = profile["fast"]["fallback_reasons"].get(reason, 0)
            n = profile["cheap"]["fallback_reasons"].get(reason, 0)
            ap(f"| {reason} | {count} | {_pct(count, len(summary.fallback_events)):.1f}% | "
               f"{d} ({_pct(d, fast_fb):.1f}%) | {n} ({_pct(n, cheap_fb):.1f}%) |")

    _append_error_section(ap, summary)

    if summary.routing_skip_reason_counts:
        fast_skips = sum(profile["fast"]["routing_skip_reasons"].values())
        cheap_skips = sum(profile["cheap"]["routing_skip_reasons"].values())
        ap("")
        ap("## routing_skip_local reasons")
        ap("")
        ap("| Reason | Total | % of skips | Standard | Low |")
        ap("|---|---|---|---|---|")
        for reason, count in summary.routing_skip_reason_counts.most_common():
            d = profile["fast"]["routing_skip_reasons"].get(reason, 0)
            n = profile["cheap"]["routing_skip_reasons"].get(reason, 0)
            ap(f"| {reason} | {count} | {_pct(count, len(summary.routing_skip_events)):.1f}% | "
               f"{d} ({_pct(d, fast_skips):.1f}%) | {n} ({_pct(n, cheap_skips):.1f}%) |")

    # Provider context-ceiling skips (LP-0MVMBM1Y3001TPRT): their own
    # breakdown, immediately after the local-skip table so all skip evidence
    # reads together and its `% of skips` column is comparable.
    _append_provider_context_skip_section(ap, summary, schedule, mode_map)

    # Residency-aware admission (LP-0MVN1IQTF0037WTG AC5): resident-vs-evicted
    # reuse split and per-session slot-owner churn, immediately after the skip
    # evidence they explain.
    _append_residency_section(ap, summary)

    # Prefill-regression checks (LP-0MVOAQDB4002D56G, parent AC4): the four
    # admission-budget numbers the 24 h certification reads, split into user
    # and agent-audit traffic.
    _append_prefill_regression_section(ap, summary)

    # Contended-pricing observability (LP-0MVS9YKHL007UBBZ, parent AC1/AC4):
    # admission verdicts split by the rate bucket the ETA was priced from and
    # by the conversion harness's new-token bands, next to the other admission
    # evidence.
    _append_admission_bucket_section(ap, summary)

    # Hold-cycle routing-skip re-emissions (LP-0MUIEO6IP0030CT3): surfaced
    # separately so a stuck hold is not read as repeated client retries.
    _append_hold_cycle_section(ap, summary)

    initial = Counter((s.initial_provider, s.initial_model) for s in sessions)
    ap("")
    ap("## Per-model breakdown (initial assignment)")
    ap("")
    ap("| Provider | Model | Sessions | Standard | Low | Requests | Fell back |")
    ap("|---|---|---|---|---|---|---|")
    total_fast_sessions = profile["fast"]["sessions"]
    total_cheap_sessions = profile["cheap"]["sessions"]
    for (provider, model), count in initial.most_common():
        s_list = [s for s in sessions if s.initial_provider == provider and s.initial_model == model]
        fast = sum(1 for s in s_list if _bucket_key(s.bucket) == "fast")
        cheap = len(s_list) - fast
        reqs = sum(s.messages for s in s_list)
        fb = sum(1 for s in s_list if s.fell_back)
        ap(f"| {provider} | {model} | {count} | {fast} ({_pct(fast, total_fast_sessions):.1f}%) | "
           f"{cheap} ({_pct(cheap, total_cheap_sessions):.1f}%) | {reqs} | {fb} |")

    # TTFT section: Total / Fast / Cheap split by session bucket
    # (LP-0MTSSM5SO003PKU0, LP-0MUI56OCI008EGML).
    _append_ttft_section(ap, summary.ttft_events, summary.sessions)
    _append_ttc_section(ap, sessions)
    _append_speed_section(ap, "Decode speed", "decode", speed)
    _append_speed_section(ap, "Prompt eval speed", "prompt_eval", speed)

    _append_busy_section(ap, summary, schedule)
    _append_prefill_vs_decode_section(ap, summary)
    _append_contention_prediction_section(ap, summary, speed)

    _append_mode_timeline_section(ap, summary, mode_map)

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
        "(log-line payloads are truncated and are never used for sizes). Local streams report "
        "llama-server's measured usage since LP-0MVMHNVKT003PRIK (the proxy requests "
        "`stream_options.include_usage`), so local prompt sizes and per-model local tokens are "
        "measured rather than inferred from the session's initial provider. A terminal line "
        "whose upstream reported no usage carries the explicit marker `tokens=missing` "
        "(LP-0MVMBM6N400541EH) — that is the cause of any missing context/response figure, and "
        "those lines are counted per session in the **Total tokens** section rather than "
        "silently treated as zero. Logs written before that marker carry no `tokens=` at all and "
        "are counted the same way."
    )
    ap(
        "- A session is included when it has at least one `Stream started` inside the window; "
        "fast/cheap bucketing uses the timestamp of that first in-window `Stream started` and "
        "the profile active at that time. The profile comes from the reconstructed operating-mode "
        "timeline when the logs carry `Mode scheduler` lines for scheduled transitions plus the "
        "`Grandfathering: enabled; ... (current=<mode>)` marker for manual switches "
        "(LP-0MT1EE315007AKXG; windows without mode transitions fall through, LP-0MSPZUD4G007IYGH); "
        "the retired mode schedule leaves the configured `server.priority_window` as the cheap-mode "
        "signal, so a window with no mode evidence buckets traffic inside it as cheap/low and "
        "everything else as fast (LP-0MVKSXXIR006OBI2). Windows with neither fall back to the "
        "slot schedule of the analysis-time config."
    )
    ap(
        "- Sessions spanning a slot-schedule transition may observe 503s during the drain window; "
        "those are expected and not treated as errors."
    )
    ap(
        "- `Fallback triggered` lines carry `session=`/`attempt=` plus budget evidence "
        "(`limit=`/`remaining=`/`waited=`) when the reason is "
        "`provider_budget_exhausted*`. A session's "
        "reason therefore comes from its own evidence (its `Fallback triggered` or "
        "`routing_skip_local` line); the nearest-line-within-60s heuristic applies only to "
        "session-less lines from rotated logs predating the session stamp, so a session never "
        "inherits another session's reason. Sessions with no fallback line but a local-dispatch "
        "lease refusal are labelled `lease_denied` (`local_lease_active`); any other session "
        "without own evidence is `unattributed`, and a session that never used remote is "
        "`no_remote_fallback` (see the `fallback_attribution` CSV column)."
    )
    ap(
        "- Every structured line a request emits also carries a per-turn correlation id "
        "(`turn=<8 hex>`, LP-0MVMBM467005BT4A). Where ids are present the pairing is "
        "**exact** — fallbacks, routing skips, dispatch denials, stream start/finish and "
        "TTFT lines are joined by id, so interleaved turns of one session can no longer "
        "swap reasons or pair with the wrong stream. Lines without ids (rotated logs "
        "predating the field) still use the per-session FIFO / nearest-line heuristics "
        "described above."
    )
    ap(
        "- Related context: work item LP-0MSAOQTJS000FFVM (evaluate increasing local ctx-size) can "
        "use this report's `large_context_bypass` data. See the skill's SKILL.md for interpretation."
    )
    ap(
        "- Decode/prompt-eval speeds come from llama-server eval-timing lines, "
        "filtered to the Qwen3 child port (discovered per log file from the "
        "router's `name=Qwen3 on port` spawn line, or inferred when rotation "
        "split the spawn line away and the file's eval lines carry a single "
        "unambiguous port). llama-server.log lines carry no timestamps, so the "
        "fast/cheap split and window filtering are approximate: each sample is "
        "bucketed by its log file's last-write time. Rotated files follow both "
        "the numbered (`llama-server.N.log`) and daily "
        "(`llama-server.log-YYYY-MM-DD[.gz]`) naming schemes. Files whose Qwen3 "
        "port cannot be determined are skipped, and a window with no samples "
        "prints a named reason (log path, files scanned, lines matched) instead "
        "of an unattributed blank (LP-0MVJNIZRO008S031)."
    )
    ap(
        "- Compaction events (LP-0MTHCTLAF00147IT): the proxy emits `compaction_event` lines for "
        "every compaction decision (compact / remote_with_guidance / backstop_*). `dry_run=true` "
        "indicates the prompt was rejected without sending to the model; `dry_run=false` means "
        "compaction actually ran. Churn warnings (`compaction_churn`) signal compaction rate "
        "exceeding the target; backstops (`compaction_backstop`) are forced truncations when "
        "compaction cannot keep pace. The trigger thresholds in the report are schedule-aware "
        "(from ``slot_schedule.ctx_by_time``), not the static ``local_model_ctx_size`` fallback; "
        "cheap live triggers can exceed fast despite the file fallback being lower "
        "(LP-0MTNIJQ8U007AGVW)."
    )
    return "\n".join(lines) + "\n"


def _fmt_window(priority_window: object) -> str:
    """``HH:MM–HH:MM`` rendering of a priority window (with a wrap marker)."""
    start = float(getattr(priority_window, "start_minutes", 0))
    end = float(getattr(priority_window, "end_minutes", 0))
    wraps = bool(getattr(priority_window, "wraps", False))

    def _hhmm(minutes: float) -> str:
        total = int(minutes) % (24 * 60)
        return f"{total // 60:02d}:{total % 60:02d}"

    suffix = " (wraps past midnight)" if wraps else ""
    return f"{_hhmm(start)}–{_hhmm(end)}{suffix}"


def _append_mode_timeline_section(
    ap: callable, summary: AnalysisResult, mode_map: bucketing.ModeScheduleMap | None
) -> None:
    """Append the ``## Priority window timeline & restarts`` section to the report.

    Lists every observed mode transition and restart event in timestamp
    order, displaying the attribution (source / actor / reason) when
    available. Restart rows show the restart kind (``spawned`` /
    ``complete`` / ``startup``) plus the requested operating mode when the
    log line carried one (``mode=`` marker); mode-switch rows show the
    applied/requested mode. Events without markers render ``unknown`` for
    the absent fields so historical logs degrade gracefully (backward
    compatible, LP-0MVC0S6U5003SEDJ).

    Source values display consistently as the canonical ``scheduler`` /
    ``worker`` / ``api`` / ``manual`` labels (LP-0MVC144KS001PDR4).

    When the window's fast/cheap profile came from the configured
    ``server.priority_window`` rather than a reconstructed mode schedule (the
    mode timeline is retired in the unified config), a note naming that source
    is rendered so the attribution is auditable (LP-0MVKSXXIR006OBI2 AC3).
    """
    transitions = mode_map.transitions if mode_map else []
    restarts = summary.restart_events_sorted
    priority_window = getattr(mode_map, "priority_window", None)

    # Merge mode transitions and restart events into a single timeline.
    # (timestamp, event_type, mode_or_kind, source, actor, reason)
    timeline: list[tuple[datetime, str, str, str | None, str | None, str | None]] = []
    for tr in transitions:
        timeline.append((tr.ts, "mode_switch", tr.mode, tr.source, tr.actor, tr.reason))
    for ev in restarts:
        if ev.kind != "restart":
            continue
        kind = ev.restart_kind or "-"
        if ev.mode:
            # The set_mode() spawn carries the mode being switched to;
            # surface it as the requested mode next to the restart kind
            # (LP-0MVC144KS001PDR4 AC2).
            kind = f"{kind} → {ev.mode}"
        timeline.append((ev.ts, "restart", kind, ev.source, ev.actor, ev.reason))

    timeline.sort(key=lambda x: x[0])
    if not timeline and priority_window is None:
        return

    ap("")
    ap("## Priority window timeline & restarts")
    ap("")
    if priority_window is not None and not transitions:
        # AC3: name the attribution source explicitly.
        ap(
            "Mode schedule: **retired** — no `Mode scheduler` transition lines "
            "were found in this window, so fast/cheap attribution is derived from "
            f"the configured `{priority_window.source}` "
            f"({_fmt_window(priority_window)}, local time): traffic inside the "
            "window is bucketed **cheap** (low priority), traffic outside it "
            "**fast**. The window is the only cheap-mode signal in the unified "
            "config (LP-0MVDT94F0003VHUA)."
        )
        ap("")
    elif priority_window is not None and transitions:
        ap(
            "Mode schedule: **active for this window** — the logs carry "
            "`Mode scheduler` / `Grandfathering` evidence, so fast/cheap "
            "attribution follows the reconstructed mode timeline and the "
            f"configured `{priority_window.source}` "
            f"({_fmt_window(priority_window)}) is not applied."
        )
        ap("")
    if not timeline:
        return
    ap(
        "Each line shows the timestamp, event type, and attribution "
        "(source / actor / reason) when available. Restart rows also show the "
        "restart kind and, when logged, the requested operating mode; "
        "mode-switch rows show the applied mode. Events without markers show "
        "\"unknown\" for attribution (historical logs degrade gracefully, "
        "LP-0MVC0S6U5003SEDJ)."
    )
    ap("")
    ap("| Timestamp | Event | Mode / Kind | Source | Actor | Reason |")
    ap("|---|---|---|---|---|---|")

    for ts, event_type, mode_or_kind, source, actor, reason in timeline:
        ts_str = _fmt_dt(ts)
        # Source renders verbatim as parsed — already one of the canonical
        # ``scheduler`` / ``worker`` / ``api`` / ``manual`` labels (the
        # tolerant extractor in log_parser only accepts ``[a-z0-9_]+``
        # tokens); ``None`` (no marker) renders "unknown" so historical
        # logs degrade gracefully (LP-0MVC144KS001PDR4 AC4).
        src = source if source else "unknown"
        act = actor if actor else "unknown"
        rsn = reason if reason else "unknown"
        ap(f"| {ts_str} | {event_type} | {mode_or_kind} | {src} | {act} | {rsn} |")


def _resolve_trigger_for(ctx_size: int, slots: int) -> int:
    """Compute the compaction trigger for one (ctx_size, slots) pair.

    Mirrors the proxy: ``round_half_up(0.70 * (ctx_size // slots - 4096))``
    (``compaction_trigger_ratio`` \u00d7 the per-slot clamp in
    ``proxy/proxy/compaction.py``).
    """
    per_slot = ctx_size // slots - 4096
    if per_slot <= 0:
        return 0
    raw = Decimal(str(0.70)) * Decimal(str(per_slot))
    return int(raw.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _mode_trigger(cfg: dict | None, shared: dict | None) -> int:
    """Resolve the compaction trigger for ONE operating mode's profile config.

    Uses the mode's own slot pool (``session_slot_pool_size``) against its
    effective context sizes: the static ``local_model_ctx_size`` plus any
    per-period ``slot_schedule.ctx_by_time`` overrides (the schedule may
    raise the window above the file fallback, LP-0MTNIJQ8U007AGVW). The
    highest trigger across those contexts is returned (a mode whose schedule
    pins a larger ctx can fire later than its static fallback implies).
    Missing values fall back to the shared config (``shared``) so a profile
    that omits a key still resolves deterministically.
    """
    if cfg is None:
        cfg = {}
    slots = cfg.get("session_slot_pool_size")
    if slots is None:
        slots = (shared or {}).get("session_slot_pool_size")
    if not slots:
        return 0
    ctx = cfg.get("local_model_ctx_size")
    if ctx is None:
        ctx = (shared or {}).get("local_model_ctx_size")
    candidates = [ctx] if ctx else []
    schedule = cfg.get("slot_schedule") if isinstance(cfg, dict) else None
    if isinstance(schedule, dict):
        candidates.extend(
            int(v) for v in (schedule.get("ctx_by_time") or {}).values() if v
        )
    triggers = [
        _resolve_trigger_for(int(ctx_size), int(slots))
        for ctx_size in candidates
        if int(ctx_size) > 0
    ]
    return max(triggers) if triggers else 0


def _compute_trigger_thresholds(
    config: dict | None, profiles: dict | None = None
) -> dict[str, int]:
    """Resolve the compaction trigger thresholds for fast and cheap modes.

    Returns ``{"fast": <trigger>, "cheap": <trigger>}`` where each trigger
    is ``round_half_up(0.70 * (ctx_size // slots - 4096))``.

    Fast and cheap are separate operating-mode profiles (``config-fast.yaml``
    / ``config-cheap.yaml``), each with its own slot pool and ctx schedule; the
    report therefore resolves each mode's trigger from ITS OWN profile
    (``profiles`` = ``config_loader.discover_configs()["profiles"]``). When
    ``profiles`` is absent (tests, single-config callers) the one config
    applies to both modes.
    """
    if profiles:
        # Production: per-mode profiles from discover_configs().
        fast_cfg = profiles.get("fast") or config or profiles.get("default")
        cheap_cfg = profiles.get("cheap") or config or profiles.get("default")
        if fast_cfg is None and cheap_cfg is None:
            return {"fast": 0, "cheap": 0}
        return {
            "fast": _mode_trigger(fast_cfg, config),
            "cheap": _mode_trigger(cheap_cfg, config),
        }

    # Legacy single-config path (tests / no mode split): one config covers both.
    if config is None or config.get("session_slot_pool_size") is None:
        return {"fast": 0, "cheap": 0}
    t = _mode_trigger(config, {})
    return {"fast": t, "cheap": t}


def _append_compaction_dry_run_estimate(
    ap, sessions: list, config: dict | None = None, profiles: dict | None = None
) -> None:
    """Append a dry-run hypothetical estimate when no compaction events occurred.

    Uses each session's ``max_context_size`` as a proxy for whether the
    compaction trigger would have fired. Sessions are compared against their
    own mode's trigger (``profiles`` = ``discover_configs()["profiles"]``)
    so a fast bucket is judged against the fast profile's threshold and a
    cheap bucket against the cheap profile's. The estimate is clearly
    labelled hypothetical with a warning that no history was mutated.
    """
    thresholds = _compute_trigger_thresholds(config, profiles)
    fast_t = thresholds.get("fast", 0)
    cheap_t = thresholds.get("cheap", 0)
    if not sessions or (fast_t == 0 and cheap_t == 0):
        # No thresholds to estimate against
        return

    fast_count = 0
    cheap_count = 0
    for s in sessions:
        max_ctx = s.max_context_size if s.max_context_size is not None else 0
        bucket = _bucket_key(s.bucket)
        if bucket == "fast":
            if max_ctx > fast_t:
                fast_count += 1
        else:  # cheap
            if max_ctx > cheap_t:
                cheap_count += 1

    total = fast_count + cheap_count
    n = len(sessions)

    ap("")
    ap("DRY-RUN ESTIMATE (hypothetical \u2014 no compaction ran)")
    ap("")
    ap(f"- Would-have-triggered: **{total}** / {n} (fast {fast_count} / cheap {cheap_count})")
    ap(
        f"- Triggers used: fast ≈{fast_t:,} / cheap ≈{cheap_t:,} "
        f"(`compaction_trigger_ratio=0.70` \u00d7 per-slot clamp)"
    )
    ap(
        "\u26a0\ufe0f  This is a rough estimate using max-context as a proxy; "
        "the server's session estimator is more nuanced (new-token economics "
        "with cached_ratio). No history was mutated, no dispatch change."
    )
    ap("")


def _append_compaction_section(
    ap, summary: AnalysisResult, config: dict | None = None, profiles: dict | None = None
) -> None:
    """Append the ``## Server-side compaction`` section to the report.

    Reports server-side proactive session compaction telemetry
    (LP-0MTHCTLAF00147IT): total events, Total/Fast/Cheap split (using the
    event's ``mode`` field; sessions bucketed the same way stay aligned),
    action breakdown, dry-run labelling, token/turn economics, and
    fallback-avoidance impact (compacted sessions that stayed local vs fell
    back, and the estimate of avoided fallbacks computed from the
    pre/post token estimates against the effective large-context thresholds).
    ``profiles`` (``discover_configs()["profiles"]``) supplies each mode's
    own config so the reported triggers are per-mode (LP-0MTSU19N9007TYZC).
    """
    compactions = [e for e in summary.compaction_events if e.kind == "compaction_event"]
    backstops = [e for e in summary.compaction_events if e.kind == "compaction_backstop"]
    churns = [e for e in summary.compaction_events if e.kind == "compaction_churn"]
    has_events = bool(compactions or backstops or churns)

    ap("")
    ap("## Server-side compaction")
    ap("")

    if not has_events:
        ap("No compactions observed in window.")
        ap("")
        thresholds = _compute_trigger_thresholds(config, profiles)
        fast_t = thresholds.get("fast", 0)
        cheap_t = thresholds.get("cheap", 0)
        ap(
            "_The server emits `compaction_event` lines only when the session "
            "estimate exceeds the per-mode trigger ("
            f"fast ≈{fast_t:,} / cheap ≈{cheap_t:,}, "
            "`compaction_trigger_ratio=0.70` × per-slot clamp); below trigger or "
            "without a session history the window is expected to be empty._"
        )
        ap(
            "_Default mode is dry-run advisory (`compaction_dry_run: true`) — "
            "no dispatch change even when events fire. Flip "
            "`server.compaction_dry_run: false` for live enforcement._"
        )
        # Dry-run estimate: how many sessions *would* have triggered
        summary_sessions = list(summary.sessions.values())
        if summary_sessions:
            _append_compaction_dry_run_estimate(ap, summary_sessions, config, profiles)
        return

    # --- Aggregate stats via the shared impact helper ---
    impact = aggregation.compaction_impact(summary.sessions, summary.compaction_events, config)
    sessions_with_compaction = impact["sessions_with_compaction"]
    by_bucket = impact["bucket_counts"]
    bucket_dry_live = impact["bucket_dry_live"]
    dry_live = impact["dry_run"]
    stayed_local = impact["stayed_local"]
    fell_back = impact["fell_back"]
    would_have_avoided = impact["would_have_avoided"]
    would_still_fallback = impact["would_still_fallback"]
    avoided = impact["avoided"]
    effective_warm = impact["effective_warm"]
    backstop_sessions = impact["backstop_sessions"]

    total_compactions = len(compactions) + len(backstops)
    all_dry = (dry_live.get("live", 0) == 0) and (dry_live.get("dry_run", 0) > 0)
    heading_note = "Advisory (dry_run) — no dispatch change" if all_dry else None
    if heading_note:
        ap(f"_{heading_note}._")
        ap("")

    # --- Top-line table: Total / Fast / Cheap split (same column convention
    # as the Session summary). Counts come from compaction_event.action +
    # backstop events; live vs dry_run breakdown reflects the event-level
    # ``dry_run`` flag. ---
    fast_total = by_bucket.get("fast", 0)
    cheap_total = by_bucket.get("cheap", 0)
    unknown_bucket = total_compactions - fast_total - cheap_total
    # Backstop flag: backstops carry no mode; Cheap/Fast columns are pure
    # compacted-session counts.
    if unknown_bucket:
        ap(f"Total compaction events: **{total_compactions}** (" f"Total {total_compactions} / Fast {fast_total} / Cheap {cheap_total} / unknown-mode {unknown_bucket})")
    else:
        ap(f"Total compaction events: **{total_compactions}** (Total {total_compactions} / Standard {fast_total} / Low {cheap_total})")
    ap(f"- Sessions with compaction: {sessions_with_compaction} / {len(summary.sessions)} "
       f"({(_pct(sessions_with_compaction, len(summary.sessions)) if len(summary.sessions) else 0):.1f}%)")

    # Bucket × action × dry_run breakdown.
    ap("")
    ap("| Metric | Total | Standard | Low |")
    ap("|---|---|---|---|")
    # Combine compaction_event + backstop actions for the action breakdown.
    # Count per action per bucket from the raw events (includes dry-run flag).
    def _action_bucket_count(action: str, bucket: str) -> int:
        return sum(1 for e in summary.compaction_events
                   if e.action == action and (e.mode == bucket) and e.kind == "compaction_event")
    def _total_action_count(action: str) -> int:
        return sum(1 for e in summary.compaction_events if e.action == action)
    actions = sorted(set(e.action for e in summary.compaction_events if e.action)) or []
    for action in actions:
        t = _total_action_count(action)
        f = _action_bucket_count(action, "fast")
        c = _action_bucket_count(action, "cheap")
        ap(f"| action={action} | {t} | {f} | {c} |")
    # Dry-run vs applied split (matches acceptance criteria AC2).
    ap(f"| dry_run=true (advisory) | {dry_live.get('dry_run', 0)} | {bucket_dry_live.get('fast', {}).get('dry_run', 0)} | {bucket_dry_live.get('cheap', {}).get('dry_run', 0)} |")
    ap(f"| dry_run=false (applied) | {dry_live.get('live', 0)} | {bucket_dry_live.get('fast', {}).get('live', 0)} | {bucket_dry_live.get('cheap', {}).get('live', 0)} |")
    if backstop_sessions:
        ap(f"| backstop sessions | {backstop_sessions} | - | - |")

    # --- Token / turn economics (per average sample) ---
    pre_sum = 0
    post_sum = 0
    samples = 0
    turns_summ = 0
    turns_drop = 0
    for e in compactions:
        if e.pre_tokens is not None and e.post_tokens is not None:
            pre_sum += e.pre_tokens
            post_sum += e.post_tokens
            samples += 1
        if e.turns_summarized is not None:
            turns_summ += e.turns_summarized
        if e.turns_dropped is not None:
            turns_drop += e.turns_dropped
    for e in backstops:
        if e.pre_tokens is not None and e.post_tokens is not None:
            pre_sum += e.pre_tokens
            post_sum += e.post_tokens
            samples += 1
        if e.turns_dropped is not None:
            turns_drop += e.turns_dropped
    ap("")
    if samples:
        avg_pre = round(pre_sum / samples)
        avg_post = round(post_sum / samples)
        saved = avg_pre - avg_post
        saved_pct = (saved / avg_pre * 100.0) if avg_pre else 0.0
        ap(f"- Avg pre_tokens: {avg_pre:,} | post_tokens: {avg_post:,} | " f"saved {saved:,} ({saved_pct:.1f}%)")
        ap(f"- Avg turns summarized: {turns_summ / max(samples, 1):.1f} | turns dropped (backstop): {turns_drop}")
    else:
        ap("- Token/turn economics unavailable (no pre/post token estimates in window).")

    # --- Fallback-avoidance impact ---
    # Use a bold label (not a "##"/"###" heading) so section parsers that
    # split on "## " (e.g. tests) keep this content inside the
    # "## Server-side compaction" section.
    ap("")
    ap("**Fallback-avoidance impact**")
    ap("")
    ap(f"- Compacted sessions that stayed local: {stayed_local} / {sessions_with_compaction}")
    ap(f"- Compacted sessions that still fell back: {fell_back} / {sessions_with_compaction}")
    if effective_warm is not None:
        ap(f"_Effective warm threshold (estimated fallback boundary): {effective_warm:,}_")
        if all_dry:
            ap(f"- Would-have-avoided fallbacks (dry-run, advisory): {would_have_avoided} (pre ≥ threshold → post < threshold)")
            ap(f"- Would-still-fallback (dry-run, stayed over threshold): {would_still_fallback}")
        else:
            ap(f"- Avoided fallbacks (live, pre ≥ threshold → post < threshold): {avoided}")
            if would_have_avoided or would_still_fallback:
                ap(f"- Would-have-avoided (remaining dry-run, advisory): {would_have_avoided}")
    else:
        ap("_Threshold-unaware impact estimate not available (no config thresholds in analysis)._")

    if churns:
        ap("")
        ap("Churn warnings")
        ap("")
        ap(f"_Compaction churn (> target rate per session/hour): {len(churns)}_")
        for e in churns:
            ap(f"- session={e.session or '?'} count={e.churn_count} rate={e.churn_rate}/h")

    # When the entire window was dry-run, clearly state it so the
    # estimates are not overstated.
    if all_dry and (would_have_avoided or avoided or would_still_fallback):
        ap("")
        ap("_Impact estimates in this window are hypothetical: no live dispatch change; " "the session history was not mutated._")


def _append_provider_context_skip_section(
    ap,
    summary: AnalysisResult,
    schedule: bucketing.SlotSchedule,
    mode_map: bucketing.ModeScheduleMap | None = None,
) -> None:
    """Append the ``## Provider context skips`` section to the report.

    ``routing_skip_provider_context`` lines (LP-0MVMBM1Y3001TPRT) are
    per-provider context-ceiling skips: the estimated prompt exceeded that
    remote entry's ``max_context_tokens``, so the entry was skipped for this
    request only (no cooldown, no failure-domain exclusion). They are the
    session-bearing explanation for chains that the session-less
    ``Fallback triggered`` lines cannot attribute, so they get their own
    breakdown instead of being folded into the local-skip table.

    Rendered only when the window carries such events. ``% of skips`` uses
    **all skip events** as its denominator (``routing_skip_local`` +
    provider-context skips), so the two tables stay comparable; the
    Standard/Low cells are category-relative like the other reason tables.
    """
    events = summary.provider_context_skip_events
    if not events:
        return
    local_skips = len(summary.routing_skip_events)
    total_skips = local_skips + len(events)
    by_provider: Counter = Counter()
    providers_by_bucket: dict[str, Counter] = {"fast": Counter(), "cheap": Counter()}
    limits_by_provider: dict[str, Counter] = {}
    types_by_provider: dict[str, Counter] = {}
    estimates: list[int] = []
    for ev in events:
        provider = ev.provider or UNKNOWN_LABEL
        by_provider[provider] += 1
        label = bucketing.bucket_label(ev.ts, schedule, mode_map)
        providers_by_bucket["cheap" if label == "cheap" else "fast"][provider] += 1
        if ev.limit is not None:
            limits_by_provider.setdefault(provider, Counter())[ev.limit] += 1
        types_by_provider.setdefault(provider, Counter())[
            ev.provider_type or UNKNOWN_LABEL
        ] += 1
        if ev.estimated_tokens is not None:
            estimates.append(ev.estimated_tokens)

    fast_total = sum(providers_by_bucket["fast"].values())
    cheap_total = sum(providers_by_bucket["cheap"].values())

    ap("")
    ap("## Provider context skips")
    ap("")
    ap("`routing_skip_provider_context` events: per-request **size** mismatches "
       "where the estimated prompt exceeded that remote entry's "
       "`max_context_tokens`. They are not provider failures (no cooldown, no "
       "failure-domain exclusion) and the entry stays eligible for smaller "
       "requests. Percentages are shares of **all skip events** "
       "(`routing_skip_local` + provider-context skips).")
    ap("")
    ap(f"- Provider context skips: **{len(events)}** "
       f"({_pct(len(events), total_skips):.1f}% of {total_skips} skip events: "
       f"{local_skips} `routing_skip_local` + {len(events)} provider-context)")
    if estimates:
        estimates.sort()
        ap(f"- Estimated prompt tokens that hit a ceiling: median "
           f"**{int(_percentile(estimates, 50))}**, p90 "
           f"**{int(_percentile(estimates, 90))}**, max **{estimates[-1]}** "
           f"({len(estimates)} of {len(events)} events carried an estimate)")
    no_limit = sum(1 for ev in events if ev.limit is None)
    if no_limit:
        ap(f"- **{no_limit}** event(s) carried no `limit=` field (older rotated "
           "logs): their ceiling is reported as `not logged`.")
    ap("")
    ap("| Provider | Type | Total | % of skips | Standard | Low | Ceiling (limit) |")
    ap("|---|---|---|---|---|---|---|")
    for provider, count in by_provider.most_common():
        fast = providers_by_bucket["fast"].get(provider, 0)
        cheap = providers_by_bucket["cheap"].get(provider, 0)
        types = ", ".join(
            t for t, _ in types_by_provider.get(provider, Counter()).most_common()
        ) or UNKNOWN_LABEL
        limits = limits_by_provider.get(provider, Counter())
        ceiling = ", ".join(f"{limit} ({n})" for limit, n in limits.most_common())
        if not ceiling:
            ceiling = "not logged"
        ap(f"| {provider} | {types} | {count} | {_pct(count, total_skips):.1f}% | "
           f"{fast} ({_pct(fast, fast_total):.1f}%) | "
           f"{cheap} ({_pct(cheap, cheap_total):.1f}%) | {ceiling} |")


def _append_residency_section(ap, summary: AnalysisResult) -> None:
    """Append the ``## Residency-aware admission`` section to the report.

    LP-0MVN1IQTF0037WTG AC5. Two questions the pre-change report could not
    answer:

    - **Resident vs evicted reuse.** `routing_resident_prefix` lines carry the
      *measured* reusable prefix (`n_past_actual`) next to the request estimate
      (`estimated_tokens`), so admission can be split into genuinely resident
      work (ratio at/above the configured threshold) versus evicted work (the
      session held no slot, or held one whose prefix had been replaced). The
      2026-10-24 baseline — median reuse 15.6 % for prompts >20k tokens — is
      exactly this split, and it is invisible in the predicted
      `cached_ratio` figures.
    - **Slot-owner churn.** Which sessions owned the slot at decision time.
      A session whose `slot_owner_session` changes between decisions lost its
      resident prefix to another tenant; counting distinct owners per session
      per hour is the churn signal behind the 27-sessions-through-2-slots
      observation.

    Also lists post-acceptance evictions (`local_eviction`) by reason so a
    regression in reclaim behaviour (dead sessions gating the pool) shows up
    as data rather than as a client-visible stall.

    **Measurement provenance (LP-0MVT2QHMU0008F2V AC4).** The pre-change line
    could only say `reuse_ratio=none`, which made "no slot to measure", "no
    prior measurement for the pair" and "capture switched off" look identical
    — the 2026-10-27 window logged 3,607 decisions with 0 measured and the AC4
    reuse rule could never fire. The additive `measurement=` field
    (`measured` / `unavailable:<reason>`, closed vocabulary from
    LP-0MVT2Q1SD009JLKD) is now parsed onto `LogEvent.measurement` and this
    section reports:

    - the measured / unmeasured decision split (unchanged line);
    - a per-reason **Unmeasured reason** table, counted with
      `Counter(ev.measurement or "unrecorded")` over unmeasured decisions, so
      each cause gets its own count and remedy (a line predating the field
      counts as `unrecorded`);
    - an explicit **all-unmeasured finding** when `measured == 0` and there
      are residency decisions — "decisions logged but all unmeasured" is a
      broken measurement ladder, never a quiet window.

    **Downstream handoff (LP-0MVQ4X1S30034BQ1 classifier).** The same counts are
    exposed machine-readably by `summary_to_json` as the additive keys
    `residency_measured_decisions` (int) and
    `residency_unmeasured_reasons` (`{reason: count}`; reasons are the raw
    `measurement=` values, `unrecorded` for old lines). Both are computed by
    `_residency_measurement_breakdown`, so the report and the JSON can never
    disagree. No existing key changes meaning or value.
    """
    residency_events = list(getattr(summary, "residency_events", []) or [])
    eviction_events = list(getattr(summary, "eviction_events", []) or [])
    after_acceptance = list(
        getattr(summary, "eviction_after_acceptance_events", []) or []
    )
    if not residency_events and not eviction_events and not after_acceptance:
        return

    measured = [ev for ev in residency_events if ev.reuse_ratio is not None]
    measured_count, unmeasured_reasons = _residency_measurement_breakdown(summary)
    unmeasured_count = len(residency_events) - measured_count
    resident = [ev for ev in measured if ev.decision == "admit"]
    evicted = [
        ev
        for ev in measured
        if ev.decision == "resident_prefix_insufficient"
    ]
    no_slot = [
        ev
        for ev in residency_events
        if ev.reuse_ratio is None
        and ev.decision == "resident_prefix_insufficient"
    ]
    tenant_denials = [ev for ev in residency_events if ev.decision == "tenant_limit"]

    ap("")
    ap("## Residency-aware admission")
    ap("")
    ap("`routing_resident_prefix` decisions carry the **measured** reusable "
       "prefix (`n_past_actual`) next to the request estimate, so resident and "
       "evicted reuse are separable — the predicted `cached_ratio` cannot do "
       "this under contention.")
    ap("")
    ap(f"- Residency decisions: **{len(residency_events)}** "
       f"(measured ratio: {len(measured)}, no measurement: "
       f"{len(residency_events) - len(measured)})")
    if unmeasured_count and not measured_count:
        # The exact 3607/0 production condition (LP-0MVRZXPH70025ZNR): a bare
        # "measured ratio: 0" reads as a quiet window. Say it as a finding.
        dominant, dominant_count = unmeasured_reasons.most_common(1)[0]
        ap(f"- **Finding: all {unmeasured_count} residency decisions in this "
           f"window are unmeasured** (dominant reason: `{dominant}`, "
           f"{dominant_count} of {unmeasured_count}). The measurement ladder "
           "is not emitting a numeric prefix — this is a defect signal, not a "
           "quiet window.")
    ap(f"- Resident admissions (ratio at/above threshold): **{len(resident)}**")
    ap(f"- Evicted denials (`resident_prefix_insufficient`): **{len(evicted)}** "
       f"with a measured prefix, **{len(no_slot)}** with no slot at all")
    ap(f"- Tenant-limit denials (`tenant_limit`): **{len(tenant_denials)}**")
    if eviction_events:
        ap(f"- Post-acceptance evictions (`local_eviction`): "
           f"**{len(eviction_events)}**")
    if unmeasured_count:
        # Per-reason breakdown (LP-0MVT2QHMU0008F2V AC4): "unmeasurable" is
        # three different operator problems; each gets its own count.
        ap("")
        ap("| Unmeasured reason | Decisions | Share |")
        ap("|---|---|---|")
        for reason, count in unmeasured_reasons.most_common():
            ap(f"| `{reason}` | {count} | "
               f"{_pct(count, unmeasured_count):.1f}% |")

    if measured:
        def _ratio_stats(events) -> str:
            ratios = sorted(ev.reuse_ratio for ev in events)
            if not ratios:
                return "none"
            return (
                f"median {_percentile(ratios, 50) * 100:.1f}%, "
                f"p10 {_percentile(ratios, 10) * 100:.1f}%, "
                f"max {ratios[-1] * 100:.1f}%"
            )

        ap("")
        ap("| Class | Decisions | Reuse ratio | Est. prompt tokens | Median new tokens |")
        ap("|---|---|---|---|---|")
        for label, events in (("Resident (admitted)", resident), ("Evicted (denied)", evicted)):
            if not events:
                continue
            estimates = [
                ev.estimated_tokens for ev in events if ev.estimated_tokens
            ]
            new_tokens = sorted(max(0, ev.estimated_tokens - ev.n_past_actual)
                                for ev in events
                                if ev.estimated_tokens and ev.n_past_actual is not None)
            ap(f"| {label} | {len(events)} | {_ratio_stats(events)} | "
               f"{int(_percentile(estimates, 50)) if estimates else '-'} | "
               f"{int(_percentile(new_tokens, 50)) if new_tokens else '-'} |")

    # Slot-owner churn per session: distinct owners seen across decisions.
    owners_by_session: dict[str, set] = {}
    decisions_by_session: Counter = Counter()
    for ev in residency_events:
        if not ev.session:
            continue
        decisions_by_session[ev.session] += 1
        if ev.slot_owner_session and ev.slot_owner_session != "none":
            owners_by_session.setdefault(ev.session, set()).add(ev.slot_owner_session)
    churned = {
        session: owners
        for session, owners in owners_by_session.items()
        if len(owners) > 1
    }
    ap("")
    ap(f"- Sessions whose slot owner changed between decisions "
       f"(slot churn): **{len(churned)}** of "
       f"{len(decisions_by_session)} session(s) making residency decisions")
    if churned:
        ap("")
        ap("| Session | Decisions | Slot owners seen |")
        ap("|---|---|---|")
        for session, owners in sorted(
            churned.items(), key=lambda kv: -len(kv[1])
        )[:10]:
            names = ", ".join(sorted(owners)[:5])
            if len(owners) > 5:
                names += f", … (+{len(owners) - 5})"
            ap(f"| {session[:8]} | {decisions_by_session[session]} | {names} |")

    if after_acceptance:
        # LP-0MVN4LCQJ0034S5S AC3: the counter, its per-reason breakdown and
        # the alert threshold, inline. This is a defect signal — any count for
        # within-budget prompts is an alert, not a trend to watch.
        by_reason = Counter(ev.reason or UNKNOWN_LABEL for ev in after_acceptance)
        ap("")
        ap(f"- **Post-acceptance evictions, ERROR "
           f"(`local_eviction_after_acceptance`): "
           f"{len(after_acceptance)}**")
        ap("  | Reason | Count | Share |")
        ap("  |---|---|---|")
        for reason, count in by_reason.most_common():
            ap(f"  | {reason} | {count} | "
               f"{_pct(count, len(after_acceptance)):.1f}% |")
        ap("")
        ap("  **Alert threshold: >0.** An accepted local dispatch evicted before "
           "its first byte is a defect in the budget/pricing model for that "
           "prompt band — file a critical `test-failure` work item via the "
           "triage path (see the skill's triage rule), do not treat it as "
           "expected churn.")

    if eviction_events:
        reasons = Counter(ev.reason or UNKNOWN_LABEL for ev in eviction_events)
        snapshots = sum(1 for ev in eviction_events if ev.snapshot_planned)
        ap("")
        ap("| Eviction reason | Count | Share | Snapshot saved |")
        ap("|---|---|---|---|")
        for reason, count in reasons.most_common():
            ap(f"| {reason} | {count} | "
               f"{_pct(count, len(eviction_events)):.1f}% | "
               f"{snapshots} |")


def _append_prefill_regression_section(ap, summary: AnalysisResult) -> None:
    """Append the ``## Prefill admission regression`` section (AC4 checks).

    The cold-prefill admission budget (LP-0MVN3KMGI009BZ0D) is certified over
    a 24 h window by four checks, which used to require hand-written log
    scans. They are rendered together here, with the deadline count split into
    ``user`` and agent ``audit`` traffic (LP-0MVOAQDB4002D56G) because audit
    gate runs are real inference but not the workload the budget is priced
    for. Omitted entirely when every check is zero, like the report's other
    conditional sections.
    """
    deadline_events = len(getattr(summary, "prefill_deadline_events", []) or [])
    by_class = _prefill_deadline_by_class(summary)
    budget_refusals = sum(
        1
        for e in getattr(summary, "local_admission_events", []) or []
        if getattr(e, "decision", None) == "reject"
        and (getattr(e, "reason", None) or "") == "prefill_budget_exceeded"
    )
    wasted = _wasted_prefill_tokens(summary)
    exhausted = len(getattr(summary, "provider_exhausted_events", []) or [])
    upstream_503 = sum(
        1
        for e in getattr(summary, "error_events", []) or []
        if getattr(e, "kind", None) == "upstream_http_error"
        and getattr(e, "status", None) == 503
    )
    if not any((deadline_events, budget_refusals, wasted, exhausted, upstream_503)):
        return

    evictions = list(getattr(summary, "eviction_after_acceptance_events", []) or [])
    ap("")
    ap("## Prefill admission regression")
    ap("")
    ap("Cold-prefill admission budget checks (LP-0MVN3KMGI009BZ0D AC4). "
       "`user` / `audit` splits the deadline count by session id: `audit-*` "
       "sessions are agent audit runs, not user traffic.")
    ap("")
    ap("| Check | Value | Meaning |")
    ap("|---|---|---|")
    ap(f"| `prefill_deadline_exceeded` | {deadline_events} "
       f"(user {by_class['user']} / audit {by_class['audit']}) | "
       "admitted local prefills cancelled at the deadline |")
    ap(f"| `prefill_budget_exceeded` refusals | {budget_refusals} | "
       "over-budget prompts refused at admission (never dispatched) |")
    ap(f"| Wasted prefill tokens | {wasted} | "
       "tokens computed then discarded by deadline cancellations |")
    ap(f"| Provider exhaustion | {exhausted} | "
       "client-visible `All providers exhausted` failures |")
    ap(f"| Upstream `status=503` | {upstream_503} | provider-side 5xx responses |")
    if evictions:
        ap("")
        ap("Post-acceptance evictions in the window, by reason "
           "(`local_evictions_after_acceptance`):")
        ap("")
        ap("| Reason | Count |")
        ap("|---|---|")
        for reason, count in Counter(
            ev.reason or UNKNOWN_LABEL for ev in evictions
        ).most_common():
            ap(f"| {reason} | {count} |")


def _append_admission_bucket_section(ap, summary: AnalysisResult) -> None:
    """Append the ``## Admission verdicts by rate bucket and prompt band`` section.

    LP-0MVS9YKHL007UBBZ (parent LP-0MVS8T6F4006DGG3): the contended-pricing
    fix prices a contended admission against the contended rate bucket and
    falls back to the solitary window (logged ``rate_bucket=solitary_fallback``)
    while that window is uncalibrated. Surfacing the split here makes a
    regression to the optimistic rate visible in the standard report instead
    of requiring ad-hoc log archaeology, and the per-band conversion counts
    are read against the same bands the admission-conversion harness
    (``scripts/prefill_admission_conversion.py``) reports.

    Omitted entirely when the window holds no guard decision, like the
    report's other conditional sections. Buckets/bands with no verdict are
    omitted rather than zero-filled: an absent ``contended`` row means the
    guard never priced a contended admission, which is itself the signal.
    """
    breakdown = aggregation.admission_breakdown(summary)
    by_bucket = breakdown["by_rate_bucket"]
    by_band = breakdown["by_token_band"]
    if not by_bucket and not by_band:
        return

    ap("")
    ap("## Admission verdicts by rate bucket and prompt band")
    ap("")
    ap("`local_admission_guard` decisions in the window, split by the pricing "
       "bucket the prefill ETA was calibrated from (`rate_bucket`) and by the "
       "new-token bands the admission-conversion harness "
       "(`scripts/prefill_admission_conversion.py`) measures. `contended` "
       "prices against the contended window; `solitary_fallback` means the "
       "contended window was uncalibrated so the solitary window was used — "
       "the deliberately-optimistic residual case, so a high "
       "`solitary_fallback` share is the signal that the contended window "
       "still needs samples. `unknown` is a pre-change line with no bucket "
       "field (or a banded line with no priced `new_tokens`).")
    ap("")
    ap("### By pricing rate bucket")
    ap("")
    ap("| Rate bucket | Admitted | Refused | Refused share |")
    ap("|---|---:|---:|---:|")
    for bucket, verdicts in by_bucket.items():
        total = verdicts["admit"] + verdicts["reject"]
        ap(f"| {bucket} | {verdicts['admit']} | {verdicts['reject']} | "
           f"{_pct(verdicts['reject'], total):.1f}% |")
    ap("")
    ap("### By new-token band (conversion-harness bands)")
    ap("")
    ap("| Band | Admitted | Refused | Conversions | Unconverted | Conversion share |")
    ap("|---|---:|---:|---:|---:|---:|")
    for band, entry in by_band.items():
        share = entry["conversion_share"]
        share_cell = f"{share * 100:.1f}%" if share is not None else "-"
        ap(f"| {band} | {entry['admit']} | {entry['reject']} | "
           f"{entry['converted']} | {entry['unconverted']} | {share_cell} |")
    ap("")
    ap("Conversions join a session's `decision=admit` to a later **local** "
       "first byte (`dispatch_first_byte_ms`); an admitted session with no "
       "local first byte counts as unconverted, and a remote first byte never "
       "converts a local admission. A band with no admitted session shows `-` "
       "rather than 0%.")


def _append_hold_cycle_section(ap, summary: AnalysisResult) -> None:
    """Append the ``## Hold-cycle re-emissions`` section to the report.

    ``routing_skip_local`` events logged with ``chain_hold_cycle > 0`` are
    re-emissions of a single held request restarted across chain-hold cycles
    (LP-0MU5AIAAY003KVM0), not genuine new client demand. Aggregation keeps
    them out of every demand/local-skip counter (the hourly ``local_skip``
    totals, the ``routing_skip_local reasons`` table, and its percentages);
    this section surfaces them separately so an operator can tell a stuck
    hold from a client retry storm. Omitted when the window has no hold-cycle
    re-emissions.

    Background on the demand-vs-re-emission distinction: cycle 0 is the first
    (pre-hold) cycle and is genuine demand; every hold-restart increments the
    cycle, so ``cycle > 0`` skips are the same request held
    (``chain_hold_max_cycles`` bounds the re-emissions).
    """
    events = summary.hold_cycle_routing_skip_events
    if not events:
        return
    ap("")
    ap("## Hold-cycle re-emissions")
    ap("")
    ap("`routing_skip_local` events with `chain_hold_cycle>0` are re-emissions "
       "of a request held across chain-hold cycles (LP-0MU5AIAAY003KVM0), not "
       "genuine new client demand. They are excluded from the local-skip "
       "counts, the `routing_skip_local reasons` table, and the fallback / "
       "demand ratios in this report, and surfaced here separately so a stuck "
       "hold is distinguishable from a client retry storm.")
    ap("")
    ap(f"- Hold-cycle routing-skip re-emissions: **{len(events)}** "
       "(the cycle-0 occurrence of each held request is still counted in the "
       "demand aggregates above)")
    by_session: Counter = Counter()
    for ev in events:
        by_session[ev.session or UNKNOWN_LABEL] += 1
    ap("")
    ap("| Session | Re-emissions |")
    ap("|---|---|")
    for session, count in by_session.most_common():
        ap(f"| {session} | {count} |")


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
    ap("| Error type | Status | Count | Evidence excerpt |")
    ap("|---|---|---|---|")
    counts = summary.error_counts
    for kind, count in counts.most_common():
        first = next((e for e in summary.error_events if e.kind == kind), None)
        excerpt = ""
        if first and first.raw:
            excerpt = first.raw.strip()
            if len(excerpt) > 100:
                excerpt = excerpt[:100] + "…"
        # Split upstream HTTP errors into one row per status code.
        if kind == "upstream_http_error":
            by_status = {}
            for e in summary.error_events:
                if e.kind == "upstream_http_error" and e.status is not None:
                    by_status[e.status] = by_status.get(e.status, 0) + 1
            for status, scount in sorted(by_status.items()):
                ap(f"| {ERROR_TYPE_LABELS.get(kind, kind)} | {status} | {scount} | `{excerpt}` |")
            unlabeled = count - sum(by_status.values())
            if unlabeled > 0:
                ap(f"| {ERROR_TYPE_LABELS.get(kind, kind)} | - | {unlabeled} | `{excerpt}` |")
            continue
        ap(f"| {ERROR_TYPE_LABELS.get(kind, kind)} | - | {count} | `{excerpt}` |")
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
    # Upstream errors broken out by HTTP status code.
    if summary.upstream_error_by_status:
        ap("### Upstream HTTP error breakdown by status")
        ap("")
        ap("| Status | Count | Provider breakdown |")
        ap("|---|---|---|")
        for status, count in sorted(summary.upstream_error_by_status.items()):
            providers = ", ".join(
                f"{p} ({c})"
                for p, c in sorted(
                    summary.upstream_error_by_status_provider.get(status, {}).items(),
                    key=lambda kv: -kv[1],
                )
            )
            ap(f"| {status} | {count} | {providers} |")
        ap("")
    # Enriched stream-error coverage (LP-0MT6322OT00900OX): when stream
    # finish errors carry the informative error payload, report it so the
    # operator can verify informative-error coverage without grepping logs.
    enriched = [
        e for e in summary.error_events
        if e.kind == "stream_finish_error" and e.error_type
    ]
    if enriched:
        types = sorted({e.error_type for e in enriched if e.error_type})
        ap(
            f"- {len(enriched)} of the `stream_finish_error` events carry the enriched error "
            f"payload (types: {', '.join(types) or 'n/a'}) — informative-error fallback is "
            "reaching the log line."
        )
    # Root-cause breakdown (LP-0MUQO862C004UT7T): render the F1 classification
    # for stream errors so the operator sees which residual cause dominates
    # (e.g. a stall-after-content terminal class vs restart kills).
    root_causes = summary.root_cause_counts
    if root_causes:
        ap("### Root-cause breakdown")
        ap("")
        ap("| Error type | Root cause | Count |")
        ap("|---|---|---|")
        for kind in sorted(root_causes):
            label = ERROR_TYPE_LABELS.get(kind, kind)
            for rc, count in sorted(
                root_causes[kind].items(), key=lambda kv: (-kv[1], kv[0])
            ):
                ap(f"| {label} | {rc} | {count} |")
        ap("")
    ap(
        f"- {len(summary.error_events)} error event(s) in window — see `errors.csv` / `errors.json` "
        "and the remediation recommendations below (recovery-first silent continue, informative-error "
        "fallback, ctx-size pressure, upstream HTTP status-specific remediation)."
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


# Per-hour activity columns for the ``## Summary by hour`` table
# (LP-0MTYAZFGN003BYQS): internal counter key -> display label. Counted from
# each event's own timestamp, so long-running sessions contribute to every
# hour they touch instead of only their first-request hour.
ACTIVITY_COLUMNS = (
    ("requests_started", "Requests started"),
    ("local_attempts", "Local attempts"),
    ("local_served", "Local served"),
    ("local_dispatch_attempts", "Local dispatch"),
    ("local_denials", "Lease denials"),
    ("fallbacks", "Fallbacks"),
    ("local_skip", "Local-skip"),
)

# Footnote defining the ``State`` column's vocabulary
# (LP-0MVKUKG3F0035I85). Rendered once, immediately after the table, so the
# operator can decode ``lease-blocked``/``prefill-only-no-decode`` rows without
# consulting REFERENCE.md.
AVAILABILITY_FOOTNOTE = (
    "_Hour state (derived only from the logs; states are not exclusive and are "
    "`+`-joined when several apply):_"
)
AVAILABILITY_STATE_NOTES = (
    (
        aggregation.AVAILABILITY_SERVING,
        "at least one local stream completed (a local `Stream finished`), so the "
        "local model actually decoded tokens.",
    ),
    (
        aggregation.AVAILABILITY_PREFILL_ONLY_NO_DECODE,
        "local work was dispatched (`[local] POST` to llama-server) but no local "
        "stream completed, so prompts were prefilled without any decode.",
    ),
    (
        aggregation.AVAILABILITY_LEASE_BLOCKED,
        "local demand was refused because another session held the local dispatch "
        "lease (`local_dispatch_denied` / `reason=local_lease_active`).",
    ),
    (
        aggregation.AVAILABILITY_IDLE,
        "no local dispatch attempt, no refusal and no completed local stream: no "
        "local demand at all (a wedged or refused hour is never reported idle).",
    ),
)


def _session_classification(s: SessionStats) -> str:
    """Journey classification of one session (``local_only`` / ``fell_back``
    / ``remote_only``), matching the Session summary table's definitions:

    - ``local_only`` — never used a remote provider (``remote_requests == 0``);
    - ``fell_back`` — started local and moved to remote
      (``remote_move_time`` set, LP-0MTFO210Q0044TTF: the session's journey
      applies to the whole session regardless of when the fallback occurred);
    - ``remote_only`` — never used local (``local_requests == 0`` and remote
      traffic).

    The three buckets are mutually exclusive and partition all sessions
    (every session has at least one in-window ``Stream started``).
    """
    if s.remote_requests == 0:
        return "local_only"
    if s.fell_back:
        return "fell_back"
    return "remote_only"


def _hourly_session_classification(sessions: list[SessionStats]) -> dict[datetime, Counter]:
    """Session-classification counts keyed by the session's start hour.

    Each session is counted once, in the **absolute top-of-hour** bucket in
    which its **first request** (first in-window ``Stream started``) began.
    Retained for the JSON summary (``hourly_session_classification``); the
    report table itself now uses the per-hour activity counters
    (LP-0MTYAZFGN003BYQS). Absolute-hour keying matches
    ``BusyStats.hourly_busy``, so a rolling window never collides two
    different days' same hour-of-day.
    """
    by_hour: dict[datetime, Counter] = {}
    for s in sessions:
        key = s.start.replace(minute=0, second=0, microsecond=0)
        by_hour.setdefault(key, Counter())[_session_classification(s)] += 1
    return by_hour


def _append_summary_by_hour(ap, summary: AnalysisResult) -> None:
    """Append the top-of-report ``## Summary by hour`` table.

    One row per **absolute wall-clock hour** of the report window
    (window-bounded buckets, partial first/last hours truncated to the
    window edges, idle hours listed as ``0s``), with busy time / busy-%
    columns (busy seconds ÷ window-bounded bucket duration, defensively
    clamped to 100%) and seven per-hour **activity** columns counted by each
    event's own timestamp (LP-0MTYAZFGN003BYQS): ``Requests started`` (all
    ``Stream started`` events), ``Local attempts`` (``Stream started`` with
    ``provider=local``), ``Local served`` (``Stream finished`` with
    ``provider=local``), ``Fallbacks`` (``Fallback triggered`` events), and
    ``Local-skip`` (``routing_skip_local`` events) — replacing the previous
    session-first-hour classification columns. ``Local dispatch`` (every
    ``[local] POST`` attempt to llama-server, LP-0MVKUKG3F0035I85) and
    ``Lease denials`` (``local_dispatch_denied`` refusals) were added so the
    ``State`` column's verdict is auditable from the row itself. A **State**
    column (LP-0MVKUKG3F0035I85) sits next to ``Hour``, carrying the
    availability classification (``serving`` / ``prefill-only-no-decode`` /
    ``lease-blocked`` / ``idle``, ``+``-joined when several apply) so an hour
    whose gate was wedged no longer reads as ``0s / 0.0%`` idle; the state
    vocabulary is defined in the footnote printed after the table. Buckets are
    keyed by absolute hour, so a rolling window that straddles an hour emits
    two distinct edge partials rather than two rows sharing one hour-of-day
    bucket. Always rendered — also when the window has no local traffic (busy
    columns then read ``0s`` / ``0.0%``); with no events at all the body shows
    a ``No data`` note. Ends with a Totals row: the window-level state (the
    classification of the summed window evidence), window busy totals and the
    sum of each activity column over the window.
    """
    sessions = list(summary.sessions.values())
    hourly = dict(summary.busy.hourly_busy) if summary.busy else {}
    end = summary.window_end
    has_data = bool(sessions) or bool(summary.hourly_activity)

    ap("")
    ap("## Summary by hour")
    ap("")
    # State column (LP-0MVKUKG3F0035I85): per-hour availability classification
    # derived only from the logs (serving / prefill-only-no-decode /
    # lease-blocked / idle), so a wedged gate stops reading as ``0s / 0.0%``
    # idle. Placed next to ``Hour`` so the operator sees the outage before the
    # busy columns.
    ap("| Hour | State | Busy | % | " + " | ".join(label for _, label in ACTIVITY_COLUMNS) + " |")
    ap("|" + "---|" * (4 + len(ACTIVITY_COLUMNS)))
    activity_totals: Counter = Counter()
    window_evidence = Counter()
    if not has_data:
        ap("| _No session data in window._ |" + " - |" * (3 + len(ACTIVITY_COLUMNS)))
    else:
        # Window-bounded hour buckets keyed by ABSOLUTE hour
        # (LP-0MTYAZFGN003BYQS): each bucket's key is the top of its hour, so
        # the two edge partials of a rolling window never share a key and no
        # busy duration is double-counted. Busy seconds come from
        # ``BusyStats.hourly_busy`` (absolute hour -> seconds), already
        # window-correct because intervals are clipped to the window before
        # attribution.
        cursor = summary.window_start
        while cursor < end:
            bucket_key = cursor.replace(minute=0, second=0, microsecond=0)
            bucket_end = min(bucket_key + timedelta(hours=1), end)
            duration = (bucket_end - cursor).total_seconds()
            seconds = hourly.get(bucket_key, 0.0)
            # Defensive clamp: absolute-hour keys make a >100% row
            # impossible, but never emit one even if inputs drift.
            pct = min(100.0, (seconds / duration * 100.0)) if duration else 0.0
            counts = summary.hourly_activity.get(bucket_key, Counter())
            state = aggregation.hour_availability(bucket_key, counts).label
            for key in aggregation.AVAILABILITY_COUNTERS:
                window_evidence[key] += counts.get(key, 0)
            cells = []
            for key, _label in ACTIVITY_COLUMNS:
                value = counts.get(key, 0)
                activity_totals[key] += value
                cells.append(str(value))
            ap(
                f"| {cursor:%H:%M}-{bucket_end:%H:%M} | {state} | {_fmt_duration(seconds)} | {pct:.1f}% | "
                + " | ".join(cells)
                + " |"
            )
            cursor = bucket_end

    busy_seconds = summary.busy.busy_seconds if summary.busy else 0.0
    busy_pct = summary.busy.busy_pct if summary.busy else 0.0
    window_state = aggregation.classify_hour(
        summary.window_start,
        local_dispatch_attempts=window_evidence.get("local_dispatch_attempts", 0),
        local_served=window_evidence.get("local_served", 0),
        local_denials=window_evidence.get("local_denials", 0),
        lease_active_fallbacks=window_evidence.get("lease_active_fallbacks", 0),
    ).label
    ap(
        f"| Totals | {window_state} | {_fmt_duration(busy_seconds)} | {busy_pct:.1f}% | "
        + " | ".join(str(activity_totals.get(key, 0)) for key, _label in ACTIVITY_COLUMNS)
        + " |"
    )
    ap("")
    ap(AVAILABILITY_FOOTNOTE)
    for state_name, note in AVAILABILITY_STATE_NOTES:
        ap(f"- `{state_name}` — {note}")


def _append_busy_section(ap, summary: AnalysisResult, schedule: bucketing.SlotSchedule) -> None:
    """Append the ``## Local model utilization`` section to the report.

    Busy is the union of **dispatch → finish** intervals (prefill + decode,
    LP-0MVJN9H2I00116FP): at least one local slot occupied from the stream's
    back-calculated dispatch time until its ``Stream finished`` line. The
    definition change is stated in the section header, the previous
    decode-only figure is retained as the labelled **lower bound**, and the
    dispatch → finish union is the labelled **upper bound** (a stream with no
    first-byte line contributes its decode interval only, so the upper bound
    is never overstated). All figures — busy/idle, total compute
    (slot-seconds), concurrency, and the fast/cheap split — are computed on
    the new intervals. The section is omitted when the window has no local
    traffic. (The former hourly busy profile moved to the top of the report
    as the ``## Summary by hour`` table, LP-0MTFO210Q0044TTF.)
    """
    if summary.busy is None:
        return
    b = summary.busy
    ap("")
    ap("## Local model utilization")
    ap("")
    ap("Busy = at least one local slot active **from dispatch to finish** "
       "(prefill + decode). This redefines busy: it previously counted decode only "
       "(`Stream started` → `Stream finished`), which excluded prompt processing. "
       "The decode-only figure is retained below as the lower bound.")
    ap("")
    ap("| Metric | Total | Standard | Low |")
    ap("|---|---|---|---|")
    fast_win = b.fast_window_seconds or 0.0
    cheap_win = b.cheap_window_seconds or 0.0
    ap(f"| Busy time — upper bound (dispatch → finish) | "
       f"{_busy_cell(b.busy_seconds, b.window_seconds)} | "
       f"{_busy_cell(b.fast_busy_seconds, fast_win)} | {_busy_cell(b.cheap_busy_seconds, cheap_win)} |")
    ap(f"| Decode-only busy — lower bound (`Stream started` → finish) | "
       f"{_busy_cell(b.decode_busy_seconds, b.window_seconds)} | "
       f"{_busy_cell(b.decode_fast_busy_seconds, fast_win)} | "
       f"{_busy_cell(b.decode_cheap_busy_seconds, cheap_win)} |")
    ap(f"| Prefill busy (upper − lower) | {_fmt_duration(b.prefill_busy_seconds)} | - | - |")
    ap(f"| Idle time | {_busy_cell(b.idle_seconds, b.window_seconds)} | "
       f"{_busy_cell(fast_win - b.fast_busy_seconds, fast_win)} | "
       f"{_busy_cell(cheap_win - b.cheap_busy_seconds, cheap_win)} |")
    ap(f"| Streams served | {b.streams} | - | - |")
    ap(f"| Avg stream duration | {b.avg_stream_duration:.1f}s | - | - |")
    ap(f"| Total compute (slot-time) | {_fmt_duration(b.total_compute_seconds)} | - | - |")
    ap(f"| Avg concurrency (while busy) | {b.avg_concurrency:.2f} | - | - |")
    ap(f"| Peak concurrency | {b.peak_concurrency} | - | - |")
    # GPU-based busy time from Prometheus power consumption (hardware-side).
    gpu = summary.gpu_busy
    if gpu is not None:
        ap(f"| GPU busy — power-based (hardware) | "
           f"{_busy_cell(gpu.busy_seconds, gpu.total_seconds)} | - | - |")
    ap("")
    ap(f"_Bounds: lower = {aggregation.BUSY_LOWER_BOUND_LABEL}; upper = "
       f"{aggregation.BUSY_UPPER_BOUND_LABEL}. {b.prefill_extended_streams} of "
       f"{b.streams} stream(s) contributed a measured prefill (a first-byte line); "
       "the rest fall back to their decode interval, so the upper bound is never "
       "overstated. The count covers the paired streams whose busy interval "
       "falls inside the window, so it never exceeds Streams served; the "
       "`Prefill vs decode` Samples column counts first-byte samples anchored "
       "in the window, unfinished streams included, and may legitimately exceed "
       "Streams served._")
    if b.unfinished_streams or b.pre_window_unfinished:
        ap("")
        ap(f"_Note: {b.unfinished_streams} local stream(s) started inside the window "
           "(or within its 1h busy-pairing margin) without a logged "
           "`Stream finished` in the available logs (aborted or still running); "
           "their compute time is unknown, so busy time is a conservative lower bound._")
        if b.pre_window_unfinished:
            ap(f"_Additionally, {b.pre_window_unfinished} pre-window leftover stream(s) "
               f"started more than 1h before the window (from earlier windows) also lack a "
               f"logged `Stream finished`; they do not affect this window's busy time._")
    ap("")
    ap("Method: streams are paired exactly by their per-turn id (`turn=`) and, "
       "for pre-change lines without an id, per session FIFO across the "
       f"margin-widened event stream (48h collection margin, max({log_parser.BUSY_WINDOW_MARGIN} "
       "busy-pairing, 48h mode timeline)), each pair extended back to its "
       "back-calculated dispatch time (`ttft.ts - ttft_ms/1000`) when a "
       "`dispatch_first_byte_ms` line exists, then clipped to the window so "
       "boundary-crossing streams are counted exactly; unfinished streams are those "
       "started inside the window or within its 1h busy-pairing margin with no "
       "logged finish (earlier starts are pre-window leftovers); fast/cheap split "
       "follows the slot schedule.")
    # GPU-based busy-time note (hardware signal, not log-derived).
    if gpu is not None:
        ap("")
        ap("_The **GPU busy** row uses Prometheus power-consumption data "
           "(`avg_over_time(gpu:computing:bool[Nh])`, evaluated every 15 s): "
           "1 while the GPU draws ≥ 90 W (the recording-rule threshold), 0 "
           "otherwise. It is **hardware activity, not local model "
           "utilisation**: it counts GPU work outside the proxy's slot "
           "accounting (other workloads, non-local requests) and misses "
           "proxy-served work during log gaps. The decision figure for local "
           "model utilisation is the log-derived `local_busy` "
           "(`dispatch → finish`) above; this row is supplementary hardware "
           "telemetry and does not depend on the proxy's stream lifecycle "
           "logs._")


def _phase_cell(value: float | None) -> str:
    """Format a slot-seconds percentile (``-`` when there are no samples)."""
    return f"{value:.1f}" if value is not None else "-"


def _phase_row(
    label: str,
    stats: aggregation.SlotSecondsStats | None,
    pct: float,
) -> str:
    """One phase row of the ``## Prefill vs decode`` table."""
    if stats is None or not stats.has_samples:
        return f"| {label} | - | - | - | - | - | - | 0 |"
    return (
        f"| {label} | {_fmt_duration(stats.total_seconds)} | {pct:.1f}% | "
        f"{_phase_cell(stats.p10)} | {_phase_cell(stats.p50)} | "
        f"{_phase_cell(stats.p90)} | {_phase_cell(stats.max_seconds)} | "
        f"{stats.count} |"
    )


def _phase_split(busy: aggregation.BusyStats) -> tuple[float, float, float, float]:
    """``(prefill_total, decode_total, combined, prefill_pct)`` in slot-seconds.

    ``combined`` is the measured phase slot-time (prefill + decode) and the
    returned prefill percentage is its complement-base, so
    ``prefill_pct + (100 - prefill_pct) == 100`` exactly.
    """
    prefill = busy.prefill_stats
    decode = busy.decode_stats
    prefill_total = prefill.total_seconds if prefill is not None else 0.0
    decode_total = decode.total_seconds if decode is not None else 0.0
    combined = round(prefill_total + decode_total, 1)
    prefill_pct = round(prefill_total / combined * 100.0, 1) if combined else 0.0
    return prefill_total, decode_total, combined, prefill_pct


def _append_prefill_vs_decode_section(ap, summary: AnalysisResult) -> None:
    """Append the ``## Prefill vs decode`` section to the report.

    Splits local phase **slot time** into prefill (dispatch → first byte) and
    decode (first byte → finish), each as an absolute slot-seconds total and as
    a percentage of the combined phase slot-time (always summing to 100%),
    with p10/p50/p90/max per phase (LP-0MVJNDOUU0059MMO). Every figure is
    slot-seconds — the integral of concurrency — never wall-clock, which the
    header states explicitly. A window without prefill samples prints a named
    reason and points at the llama-server eval-timing diagnostic instead of
    leaving an unattributed blank; a window without local traffic prints a
    named "no local traffic" reason.
    """
    ap("")
    ap("## Prefill vs decode")
    ap("")
    busy = summary.busy
    if busy is None:
        ap(
            "_No local traffic in window: no local slot time to attribute to "
            "prefill or decode._"
        )
        return
    ap(
        "Where local slot time goes, by phase. Every figure here is "
        "**slot-seconds** — the integral of concurrency (the sum of the "
        "per-stream phase durations), **not wall-clock**: two concurrent 10 s "
        "prefills are 20 slot-seconds but only 10 s of wall clock. The phase "
        "totals are therefore not comparable with the wall-clock busy/idle "
        "figures in `## Local model utilization`."
    )
    ap("")
    ap(
        "| Phase | Slot-seconds | % of phase slot-time | p10 (s) | Median (s) "
        "| p90 (s) | Max (s) | Samples |"
    )
    ap("|---|---|---|---|---|---|---|---|")
    prefill = busy.prefill_stats
    decode = busy.decode_stats
    _prefill_total, _decode_total, combined, prefill_pct = _phase_split(busy)
    decode_pct = round(100.0 - prefill_pct, 1) if combined else 0.0
    ap(_phase_row("Prefill (dispatch → first byte)", prefill, prefill_pct))
    ap(_phase_row("Decode (first byte → finish)", decode, decode_pct))
    known = (prefill.count if prefill is not None else 0) + (
        decode.count if decode is not None else 0
    )
    total_pct = "100.0%" if combined else "-"
    ap(f"| Total | {_fmt_duration(combined)} | {total_pct} | - | - | - | - | {known} |")
    ap("")
    ap(
        "Percentages are of the combined measured phase slot-time (prefill + "
        "decode) and always sum to 100.0%. A phase the logs cannot measure is "
        "excluded from the percentiles and totals rather than counted as 0 s."
    )
    unknown_parts: list[str] = []
    if prefill is not None and prefill.unknown_count:
        unknown_parts.append(f"{prefill.unknown_count} unknown prefill")
    if decode is not None and decode.unknown_count:
        unknown_parts.append(
            f"{decode.unknown_count} unknown decode (no `Stream finished`)"
        )
    if unknown_parts:
        ap("")
        ap(
            "_Unknown phase durations (aborted or still-running streams): "
            f"{', '.join(unknown_parts)}; they are excluded from the "
            "percentiles and totals rather than counted as 0 s._"
        )
    if prefill is None or not prefill.has_samples:
        if prefill is None:
            reason = (
                "no local TTFT events were supplied to the analysis, so no "
                "`dispatch_first_byte_ms` first-byte line was available"
            )
        else:
            reason = (
                f"none of the {prefill.stream_count} local stream(s) carried "
                "a `dispatch_first_byte_ms` first-byte line "
                f"({prefill.unknown_count} unknown prefill)"
            )
        if combined:
            tail = (
                f"the decode figures above are {decode_pct:.1f}% of measured "
                "phase slot-time only"
            )
        else:
            tail = "no phase slot time could be measured"
        ap("")
        ap(
            f"_No prefill samples in window — {reason}. Prefill slot-time is "
            f"unattributed; {tail}. For the llama-server-side breakdown see "
            "the `## Decode speed` / `## Prompt eval speed` eval-timing "
            "diagnostic sections._"
        )


def _append_contention_prediction_section(
    ap, summary: AnalysisResult, speed: llama_log_parser.SpeedStats | None
) -> None:
    """Append the ``## Local contention prediction vs reality`` section (M1-M3).

    LP-0MVGR2OXT0096OGK: measures whether the payload's ``lease_state`` /
    ``prefill_eta_seconds`` signals predict real local contention. Rendered
    after the busy sections so the M2 busy fractions are already on the page;
    the three measurements and the explicit matches/does-not-match verdicts
    follow (the same numbers are also in ``--json`` under
    ``denial_attribution`` and ``prefill_eta_error``). Skipped when the window
    carries neither a denial nor a status poll, so unchanged windows render
    unchanged reports.
    """
    denials = int(summary.dispatch_denied_count)
    statuses = getattr(summary, "status_request_events", []) or []
    if not denials and not statuses:
        return

    att = denial_attribution_json(summary)
    eta = prefill_eta_error_json(summary, speed)
    hours = max((summary.window_end - summary.window_start).total_seconds() / 3600.0, 1e-9)
    busy_pct = summary.busy.busy_pct if summary.busy else 0.0

    ap("")
    ap("## Local contention prediction vs reality")
    ap("")
    ap("Measures whether the payload's `lease_state` / `prefill_eta_seconds` "
       "signals predict **real** local contention (LP-0MVGR2OXT0096OGK). "
       "`local_dispatch_denied` lines carry no `lease_state`, so each denial is "
       "joined to its decision-block `local_admission_guard` record "
       f"(nearest within ±{aggregation.DENIAL_GUARD_JOIN_WINDOW_SECONDS:.0f}s) and "
       "the pool state at denial time is derived from the guard's measured "
       "`busy_slots` / `in_flight_prefills` / `free_slot` fields.")
    ap("")
    ap("### M1 — denial attribution (pool state at denial time)")
    ap("")
    ap("| Hour | Denials | idle_leased | prefill_bound | saturated | unknown | "
       "unattributed | Local busy % |")
    ap("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in att["per_hour"]:
        ap(
            f"| {row['hour']} | {row['denials']} | {row['idle_leased']} | "
            f"{row['prefill_bound']} | {row['saturated']} | {row['unknown']} | "
            f"{row['unattributed']} | {row['busy_pct']:.1f}% |"
        )
    total_row = att["by_verdict"]
    ap(
        f"| **Total** | **{att['denials']}** | **{total_row['idle_leased']}** | "
        f"**{total_row['prefill_bound']}** | **{total_row['saturated']}** | "
        f"**{total_row['unknown']}** | **{att['unattributed']}** | {busy_pct:.1f}% |"
    )
    ap("")
    if att["by_guard_reason"]:
        ap("**Guard reason split (attributed denials):** "
           + ", ".join(f"{k} {v}" for k, v in att["by_guard_reason"].items())
           + ".")
    if att["capacity_hint"]:
        hints = ", ".join(f"{k} {v}" for k, v in att["capacity_hint"].items())
        ap(f"**Payload's own claim (`capacity_hint` over status polls):** {hints}.")
    ap("")
    ap("### M2 — local busy fraction vs denials")
    ap("")
    baseline = CONTENTION_BASELINE
    now_dph = att["denials"] / hours
    base_dph = baseline["denials"] / baseline["hours"]
    ap("| Metric | This window | Baseline (pre-change) |")
    ap("|---|---|---|")
    ap(f"| Window | {_fmt_dt(summary.window_start)} → {_fmt_dt(summary.window_end)} "
       f"({hours:.1f}h) | {baseline['window']} ({baseline['hours']:.0f}h) |")
    ap(f"| Dispatch denials | {att['denials']} | **{baseline['denials']}** |")
    ap(f"| Denials per hour | {now_dph:.1f} | {base_dph:.1f} |")
    ap(f"| Local busy (dispatch → finish) | {busy_pct:.1f}% | {baseline['busy_pct']}% |")
    if att["per_hour"]:
        worst = max(att["per_hour"], key=lambda r: r["denials"])
        worst_cell = (
            f"{worst['hour']} ({worst['denials']} denials at {worst['busy_pct']:.1f}% busy)"
        )
    else:
        worst_cell = "-"
    ap(f"| Worst hour | {worst_cell} | "
       f"{baseline['worst_hour']} ({baseline['worst_hour_denials']} denials at "
       f"{baseline['worst_hour_busy_pct']}% busy) |")
    ap("")
    idle_share = att["idle_leased_share_pct"]
    if idle_share is None:
        denial_verdict_label = "inconclusive"
        denial_sentence = (
            "No denial could be attributed to a guard record — the split is "
            "inconclusive."
        )
    elif idle_share >= IDLE_LEASED_SHARE_THRESHOLD_PCT:
        denial_verdict_label = "does not match reality"
        denial_sentence = (
            f"{idle_share:.1f}% of attributed denials fired while the guard "
            "measured **zero** local work (no slot decoding, no prefill in "
            "flight) — the idle-lease false-busy pattern."
        )
    else:
        denial_verdict_label = "matches reality"
        denial_sentence = (
            f"Only {idle_share:.1f}% of attributed denials fired with the pool "
            "measured idle — most denials tracked genuine prefill/decode work."
        )
    if eta["verdict"] == "bound_holds":
        eta_verdict_label = "holds"
        eta_sentence = (
            "The served p90 stayed within the published cache-cold reference "
            f"bound (observed p90 {eta['observed']['p90']} s ≤ predicted p50 "
            f"{eta['predicted']['p50']} s, n={eta['bound']['observed_samples']})."
        )
    elif eta["verdict"] == "bound_violated":
        eta_verdict_label = "was violated"
        eta_sentence = (
            "Served prefill durations exceeded the published cache-cold "
            f"reference bound (observed p90 {eta['observed']['p90']} s > "
            f"predicted p50 {eta['predicted']['p50']} s, "
            f"n={eta['bound']['observed_samples']}) — the bound is "
            "under-stated on real traffic."
        )
    else:
        eta_verdict_label = "inconclusive"
        eta_sentence = (
            "The bound-vs-served comparison is inconclusive (fewer than "
            f"{eta['bound']['min_samples']} observed dispatches, or no "
            "published polls) so the bound is neither confirmed nor refuted."
        )
    ap(f"**Denial signal (`lease_state`): {denial_verdict_label}.** {denial_sentence}")
    ap(f"**Prefill-ETA bound (`prefill_eta_seconds`, cache-cold reference bound): "
       f"{eta_verdict_label}.** {eta_sentence}")
    if "does not match reality" in denial_verdict_label or eta_verdict_label in (
        "was violated",
        "inconclusive",
    ):
        overall_label = "do not unambiguously match reality"
    else:
        overall_label = "match reality"
    ap(f"**Overall: the pre-change false-busy dispatch failures are driven by "
       f"signals that {overall_label} (denial volume {att['denials']} vs baseline "
       f"{baseline['denials']}, denials/hour {now_dph:.1f} vs {base_dph:.1f}, busy "
       f"{busy_pct:.1f}% vs {baseline['busy_pct']}%).**")
    ap("")
    ap("### M3 — prefill_eta_seconds accuracy (bound vs served remainder, p50/p90)")
    ap("")
    ap("**Comparison: bound vs served remainder.** The first table is the "
       "historical quantile-matched comparison. Read it as *does the published "
       "cache-cold reference bound hold against traffic that actually ran*, not "
       "as a per-request calibration score: the prediction is a cache-cold "
       "reference-prompt **bound** while the observation is the **uncached "
       "remainder** of mostly cache-warm prompts, so a large predicted−observed "
       "gap is expected by construction (the two populations differ). The "
       "verdict is therefore one-sided — `bound_holds` when the served p90 "
       "stays within the published p50, `bound_violated` otherwise, and "
       f"`inconclusive` below {eta['bound']['min_samples']} observed "
       "dispatches.")
    ap("")
    ap("| Quantile | Predicted (status polls) | Observed (prompt eval) | "
       "Error (pred−obs) | Relative |")
    ap("|---|---:|---:|---:|---:|")
    pred = eta["predicted"]
    obs = eta["observed"]
    ap(f"| p50 | {pred['p50'] if pred['p50'] is not None else '-'} s "
       f"(n={pred['samples']}) | {obs['p50'] if obs['p50'] is not None else '-'} s "
       f"(n={obs['samples']}) | {eta['error_p50'] if eta['error_p50'] is not None else '-'} s | "
       f"{eta['relative_error_p50_pct'] if eta['relative_error_p50_pct'] is not None else '-'}% |")
    ap(f"| p90 | {pred['p90'] if pred['p90'] is not None else '-'} s | "
       f"{obs['p90'] if obs['p90'] is not None else '-'} s | "
       f"{eta['error_p90'] if eta['error_p90'] is not None else '-'} s | "
       f"{eta['relative_error_p90_pct'] if eta['relative_error_p90_pct'] is not None else '-'}% |")
    ap("")
    ap("Predicted = the calibrated cache-cold reference bound published on "
       "`status_request` polls; observed = llama-server `prompt eval time` "
       "durations for dispatches that ran. The two streams cannot be paired "
       "1:1 (llama-server eval lines carry no per-line timestamps), so the "
       "comparison is quantile-matched — a cross-population comparison, "
       "labelled and bounded as above.")
    ap(f"**(prefill ETA bound verdict: {eta['verdict']}; observed p90 "
       f"{eta['bound']['observed_p90'] if eta['bound']['observed_p90'] is not None else '-'} s "
       f"vs predicted p50 "
       f"{eta['bound']['predicted_p50'] if eta['bound']['predicted_p50'] is not None else '-'} s, "
       f"margin {'-' if eta['bound']['margin_seconds'] is None else str(eta['bound']['margin_seconds']) + ' s'})**")
    ap("")
    matched = eta["matched"]
    ap("### M3b — matched per-dispatch prefill ETA accuracy (guard decision → "
       "that dispatch's first byte)")
    ap("")
    ap("Predicted = the ETA the admission guard priced **for this dispatch** "
       "(`eta_prefill_s`, the own-prefill term); observed = that same "
       "dispatch's `dispatch_to_first_byte_ms`. Joined on the `session=` id, so "
       "predicted and observed describe the same request — the only comparison "
       "here where the "
       f"{eta['matched']['tolerance_pct']:.0f}% tolerance is a calibration score.")
    ap("")
    ap("| Metric | Value |")
    ap("|---|---:|")
    ap(f"| Matched pairs | {matched['pairs']} "
       f"(admitted {matched['admitted_pairs']}, observe-policy "
       f"{matched['observe_pairs']}) |")
    ap(f"| Enforced refusals (never paired) | "
       f"{matched['enforced_refusals']} |")
    ap(f"| Unmatched dispatches | {matched['unmatched_dispatches']} "
       f"of {matched['dispatches']} |")
    ap(f"| Predicted p50 / p90 | "
       f"{matched['predicted_p50'] if matched['predicted_p50'] is not None else '-'} s / "
       f"{matched['predicted_p90'] if matched['predicted_p90'] is not None else '-'} s |")
    ap(f"| Observed p50 / p90 | "
       f"{matched['observed_p50'] if matched['observed_p50'] is not None else '-'} s / "
       f"{matched['observed_p90'] if matched['observed_p90'] is not None else '-'} s |")
    ap(f"| Relative error p50 / p90 | "
       f"{matched['relative_error_p50_pct'] if matched['relative_error_p50_pct'] is not None else '-'}% / "
       f"{matched['relative_error_p90_pct'] if matched['relative_error_p90_pct'] is not None else '-'}% |")
    ap("")
    ap("Unmatched dispatches are never folded into the matched figure: an "
       "inert (uncalibrated) guard, or an inactive `observe`-mode decision, "
       "logs no priced ETA and cannot be scored, and an enforced refusal "
       "never owns a first byte. The verdict uses the **median** per-pair "
       "relative error; its p90 is tail-dominated by very fast cache-warm "
       "dispatches (a 0.3 s first byte against a several-second ETA is a huge "
       "percentage). "
       f"**(matched prefill ETA verdict: {matched['verdict']})**")
    ap("")


def _fmt_tokens(n: int) -> str:
    """Format a token count with comma separators."""
    return f"{n:,}"


def _append_total_tokens_section(ap, summary) -> None:
    """Render the ``## Total tokens`` section immediately before fallback reasons.

    Shows Total (Input + Output = Total) with fast/cheap and by-model
    breakdowns, each with absolute + % share.  Shares are of the grand total
    (``input + output``) so fast% + cheap% = 100% and model %s sum to 100%.
    Rendered every run including zero-traffic windows (``0`` / ``100.0%``
    are well-defined for an empty window and ``_pct`` guards zero division).
    Token totals are summed from authoritative ``tokens=prompt/completion/total``
    on ``Stream finished`` lines (never payload truncation).  Model unknown
    renders as the ``(unknown)`` placeholder from ``aggregation.UNKNOWN_LABEL``.
    """
    grand_input, grand_output, grand_total = summary.total_tokens
    buckets = summary.token_buckets
    fast = buckets.get("fast")
    cheap = buckets.get("cheap")
    # Missing buckets (empty window, single-bucket window) are treated as 0.
    fast_in, fast_out, fast_total = fast if fast is not None else (0, 0, 0)
    cheap_in, cheap_out, cheap_total = cheap if cheap is not None else (0, 0, 0)
    ap("")
    ap("## Total tokens")
    ap("")
    ap("Total = prompt (input) + completion (output) summed from authoritative "
       "``tokens=prompt/completion/total`` on ``Stream finished`` lines.  "
       "Input = sum(prompt), Output = sum(completion), Total = Input + Output.  "
       "Shares are of the grand Total Tokens, so standard% + low% = 100% and model "
       "%s sum to 100%.  Streams without ``tokens=`` contribute ``0`` and are "
       "counted below; sessions without a bucket (unattributed) are excluded from "
       "the bucket share.")
    ap("")
    # Measured-usage coverage (LP-0MVMHNVKT003PRIK AC4): local streams report
    # llama-server's usage block since this work item (the proxy requests
    # ``stream_options.include_usage``), so real local prompt sizes are
    # measurable. A non-zero count means some terminal lines still carry no
    # usage (older rotated logs, providers that omit it, aborted streams).
    # Since LP-0MVMBM6N400541EH the proxy marks those lines explicitly with
    # ``tokens=missing``, so "no data" is distinguishable from "zero tokens"
    # and the per-session count is carried in the *Local model usage by
    # session* coverage table and the per-session CSVs.
    without = getattr(summary, "streams_without_usage", 0)
    measured = getattr(summary, "streams_finished", 0)
    total_finished = measured + without
    if total_finished:
        by_provider = getattr(summary, "streams_without_usage_by_provider", None) or {}
        provider_part = ""
        if without:
            provider_part = " (" + ", ".join(
                f"{provider} {count}"
                for provider, count in sorted(by_provider.items())
            ) + ")"
        ap(
            f"Measured usage: **{measured:,} of {total_finished:,}** "
            f"``Stream finished`` lines carry ``tokens=``."
        )
        ap("")
        ap(
            f"Streams without usage: **{without:,}**{provider_part}. Those lines "
            "carry ``tokens=missing`` (logs written before "
            "LP-0MVMBM6N400541EH simply omit the field). Their context/response "
            "sizes are unknown (not zero) and are excluded from the token "
            "totals and the local prompt-size figures."
        )
        ap("")
    ap("| Kind | Count | % of total |")
    ap("|---|---|---|")
    ap(f"| Input (prompt) | {_fmt_tokens(grand_input)} | {_pct(grand_input, grand_total):.1f}% |")
    ap(f"| Output (completion) | {_fmt_tokens(grand_output)} | {_pct(grand_output, grand_total):.1f}% |")
    ap(f"| Total | **{_fmt_tokens(grand_total)}** | 100.0% |")
    ap("")
    ap("### Tokens — standard vs low")
    ap("")
    ap("Bucketing follows the existing mode-aware fast/cheap logic (mode timeline / "
       "slot schedule keyed by session start; unattributed streams are excluded from "
       "the share).")
    ap("")
    ap("| Bucket | Input | % input | Output | % output | Total | % total |")
    ap("|---|---|---|---|---|---|---|")
    ap(f"| Standard | {_fmt_tokens(fast_in)} | {_pct(fast_in, grand_input):.1f}% | "
       f"{_fmt_tokens(fast_out)} | {_pct(fast_out, grand_output):.1f}% | "
       f"{_fmt_tokens(fast_total)} | {_pct(fast_total, grand_total):.1f}% |")
    ap(f"| Low | {_fmt_tokens(cheap_in)} | {_pct(cheap_in, grand_input):.1f}% | "
       f"{_fmt_tokens(cheap_out)} | {_pct(cheap_out, grand_output):.1f}% | "
       f"{_fmt_tokens(cheap_total)} | {_pct(cheap_total, grand_total):.1f}% |")
    ap("")
    ap("### Tokens — by model")
    ap("")
    ap("Grouped by the ``(provider, model)`` stamped on each ``Stream finished`` "
       "line — the provider/model that actually served the stream — sorted by "
       "Total desc; unknown provider/model render as the placeholder "
       "``(unknown)``. A session that starts local and falls back to a remote "
       "provider therefore splits its tokens across both instead of piling them "
       "onto its initial provider (LP-0MVMHNVKT003PRIK). Local rows are real "
       "measured prompt/completion counts, so they can be compared against "
       "remote rows.")
    ap("")
    if grand_total == 0:
        ap("_No tokens in window._")
    else:
        ap("| Provider | Model | Input | % input | Output | % output | Total | % total |")
        ap("|---|---|---|---|---|---|---|---|")
        for (provider, model), (inp, out, total) in summary.token_by_model.items():
            ap(f"| {provider} | {model} | {_fmt_tokens(inp)} | {_pct(inp, grand_input):.1f}% | "
               f"{_fmt_tokens(out)} | {_pct(out, grand_output):.1f}% | "
               f"{_fmt_tokens(total)} | {_pct(total, grand_total):.1f}% |")


def _pct(part: int, total: int) -> float:
    return (part / total * 100.0) if total else 0.0


AUDIT_SESSION_PREFIX = "audit-"


def _is_audit_session(session: str | None) -> bool:
    """Whether *session* is agent audit traffic rather than user traffic.

    Audit runs (``/skill:audit`` gate sweeps, ``herdr`` child sessions) use
    session ids prefixed ``audit-``. They issue genuine local inference and
    therefore genuinely consume prefill, but they are not the workload the
    prefill budget is priced for (LP-0MVOAQDB4002D56G).
    """
    return bool(session) and str(session).startswith(AUDIT_SESSION_PREFIX)


def _prefill_deadline_by_class(summary: AnalysisResult) -> dict[str, int]:
    """``prefill_deadline_exceeded`` counts split user vs agent-audit traffic."""
    counts = {"user": 0, "audit": 0}
    for event in getattr(summary, "prefill_deadline_events", []) or []:
        key = "audit" if _is_audit_session(getattr(event, "session", None)) else "user"
        counts[key] += 1
    return counts


def _wasted_prefill_tokens(summary: AnalysisResult) -> int:
    """Prefill tokens computed and discarded by deadline cancellations.

    Sums ``new_tokens`` over ``local_eviction_after_acceptance`` events whose
    ``reason`` is ``prefill_deadline``: those are the tokens the GPU had to
    compute for a dispatch that was then abandoned before its first byte
    (parent AC4's waste check). Other eviction reasons are excluded — their
    ``new_tokens`` did not necessarily cost a fresh prefill.

    Token semantics (LP-0MVODJ3P6000AIUM AC5): ``new_tokens`` is the
    **admission-time** value the routing/admission decision priced and the
    dispatch record stored verbatim — never a value recomputed at eviction
    time. An event whose record carried no admission snapshot contributes
    ``0`` (explicit ``unknown`` marker), so the figure can never over-account
    a re-priced prompt.
    """
    total = 0
    for event in getattr(summary, "eviction_after_acceptance_events", []) or []:
        if (getattr(event, "reason", None) or "") != "prefill_deadline":
            continue
        try:
            total += int(getattr(event, "new_tokens", None) or 0)
        except (TypeError, ValueError):
            continue
    return total


def _percentile(values: list[float], pct: float) -> float | None:
    """Return the ``pct``-th percentile of *values* (0–100), or ``None`` when empty."""
    if not values:
        return None
    s = sorted(values)
    k = (pct / 100.0) * (len(s) - 1)
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def _ttft_cell(value: float | None) -> str:
    """Format a TTFT value (milliseconds) for the report table."""
    if value is None:
        return "-"
    # Display in milliseconds; values > 1s show seconds
    if value >= 1000:
        return f"{value/1000:.1f}s"
    return f"{value:.0f}ms"


def _ttft_buckets(
    ttft_events: list[log_parser.LogEvent],
    sessions: dict[str, SessionStats],
) -> dict[str, list[float]]:
    """Partition TTFT samples (ms) into ``total`` / ``fast`` / ``cheap`` lists.

    Uses the same mode-aware session bucketing as the rest of the report:
    ``_bucket_key(session.bucket)``, where any non-``cheap`` bucket counts as
    fast. Every sample lands in ``total``; ``fast`` and ``cheap`` partition the
    samples whose session resolves against *sessions*. A sample whose session is
    absent (e.g. a TTFT line whose stream started outside the window) is counted
    in ``total`` only, so ``fast + cheap`` may be less than ``total`` in that
    rare case.
    """
    buckets: dict[str, list[float]] = {"total": [], "fast": [], "cheap": []}
    for event in ttft_events:
        value = event.ttft_ms
        if value is None:
            continue
        buckets["total"].append(value)
        session = sessions.get(event.session) if event.session else None
        if session is not None:
            buckets[_bucket_key(session.bucket)].append(value)
    return buckets


def _append_ttft_section(
    ap,
    ttft_events: list[log_parser.LogEvent],
    sessions: dict[str, SessionStats],
) -> None:
    """Append the ``## Time to first token`` section to the report.

    Shows p10 / median / p90 of TTFT (dispatch → first token), parsed from both
    local ``dispatch_first_byte_ms=`` and remote ``ttft_seconds=`` log lines,
    split into Total / Fast / Cheap rows using the mode-aware session bucketing
    (``_bucket_key``, LP-0MUI56OCI008EGML).
    """
    ap("")
    ap("## Time to first token")
    ap("")
    buckets = _ttft_buckets(ttft_events, sessions)
    if not buckets["total"]:
        ap("_No TTFT data in window._")
        return
    ap("Percentiles of time from dispatch to first token (ms), from local")
    ap("``dispatch_first_byte_ms=`` and remote ``ttft_seconds=`` log lines.")
    ap("")
    ap("| Bucket | Samples | p10 | Median | p90 |")
    ap("|---|---|---|---|---|")
    for label, bucket_key in (("Total", "total"), ("Fast", "fast"), ("Cheap", "cheap")):
        values = buckets[bucket_key]
        ap(f"| {label} | {len(values)} | "
           f"{_ttft_cell(_percentile(values, 10))} | "
           f"{_ttft_cell(_percentile(values, 50))} | "
           f"{_ttft_cell(_percentile(values, 90))} |")
    ap("")


def _append_ttc_section(
    ap,
    sessions: list[SessionStats],
) -> None:
    """Append the ``## Time to completion`` section to the report.

    Shows p10 / median / p90 of session duration (stream-start → stream-end),
    split by Total / Fast / Cheap.
    """
    ap("")
    ap("## Time to completion")
    ap("")
    if not sessions:
        ap("_No sessions in window._")
        return
    ap("Percentiles of session duration (``Stream started`` → last ``Stream finished`` in-window).")
    ap("")
    ap("| Bucket | Sessions | p10 (s) | Median (s) | p90 (s) |")
    ap("|---|---|---|---|---|")
    for label, bucket_key in (("Total", "all"), ("Fast", "fast"), ("Cheap", "cheap")):
        if bucket_key == "all":
            durations = [s.duration_seconds for s in sessions]
        else:
            durations = [s.duration_seconds for s in sessions if _bucket_key(s.bucket) == bucket_key]
        p10 = _percentile(durations, 10)
        med = _percentile(durations, 50)
        p90 = _percentile(durations, 90)
        ap(f"| {label} | {len(durations)} | "
           f"{_speed_cell(p10)} | {_speed_cell(med)} | {_speed_cell(p90)} |")
    ap("")


def _speed_cell(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "-"


def _append_speed_diagnostic(ap, speed: llama_log_parser.SpeedStats) -> None:
    """Render the named, self-diagnosing reason for an empty eval-timing
    window (LP-0MVJNIZRO008S031): the log path scanned, the files found, the
    eval-timing lines matched, and the distinguishing cause(s) — verbosity /
    build, rotation, or attribution failure — instead of an unattributed
    blank.
    """
    ap("")
    ap("**Why no eval-timing samples**")
    ap("")
    ap(f"- Log directory scanned: `{speed.log_dir or '<unknown>'}`")
    if speed.files_discovered:
        ap(
            f"- Files scanned ({len(speed.files_discovered)}): "
            + llama_log_parser._summarise_names(
                [f"`{n}`" for n in speed.files_discovered]
            )
        )
    else:
        ap("- Files scanned: **none**")
    total_eval = sum(d.eval_lines for d in speed.file_diagnostics)
    matched = sum(d.samples_in_window for d in speed.file_diagnostics)
    ap(
        f"- Files parsed: {speed.files_parsed} | skipped: "
        f"{speed.files_skipped} | outside window (rotation): "
        f"{len(speed.outside_window_files)} | unrecognised rotation names: "
        f"{len(speed.ignored_logs)}"
    )
    ap(f"- Eval-timing lines found: {total_eval} | matched to Qwen3: {matched}")
    ap("")
    for reason in llama_log_parser.empty_window_reasons(speed):
        ap(f"- {reason}")
    if speed.file_diagnostics:
        rows = [d for d in speed.file_diagnostics if d.status != llama_log_parser.STATUS_OUTSIDE_WINDOW]
        if rows:
            ap("")
            ap("| File | Status | Qwen3 port (source) | Lines | Eval lines (matched/total) | Samples |")
            ap("|---|---|---|---|---|---|")
            for d in rows:
                port = f"{d.port} ({d.port_source})" if d.port is not None else "-"
                ap(
                    f"| `{d.name}` | {d.status} | {port} | {d.lines_scanned} | "
                    f"{d.eval_lines_matched}/{d.eval_lines} | {d.samples_in_window} |"
                )


def _append_speed_section(
    ap,
    title: str,
    kind: str,
    speed: llama_log_parser.SpeedStats | None,
) -> None:
    """Append a speed section (``## Decode speed`` / ``## Prompt eval speed``)
    to the report, one row per (model, bucket) with samples / median / p90 / p10.

    When the window has no samples the section renders a named,
    self-diagnosing reason (log path, files scanned, lines matched and the
    cause) instead of an unattributed blank (LP-0MVJNIZRO008S031); the
    diagnostic is additive and the table layout is unchanged when samples do
    exist.
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
        _append_speed_diagnostic(ap, speed)
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
           "(Qwen3 child port not found)._", )


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
    mode_map: bucketing.ModeScheduleMap | None = None,
    provider_context_skip_events: list[log_parser.LogEvent] | None = None,
) -> dict:
    """Per-bucket (fast/cheap) totals for the report's summary tables.

    Covers sessions, requests, local/remote split, classification counts,
    context sizes, dispatch denials, and per-reason counters. Events without a
    session (fallbacks, routing skips, provider context skips, dispatch
    denials) are bucketed by their own timestamp using
    :func:`bucketing.bucket_label` (mode timeline, else the configured
    priority window, else the slot schedule; LP-0MVKSXXIR006OBI2).
    """
    def _label_at(ts: datetime) -> str:
        return bucketing.bucket_label(ts, schedule, mode_map)

    buckets = {
        "fast": {
            "sessions": 0, "requests": 0, "local": 0, "remote": 0,
            "local_only": 0, "fell_back": 0, "remote_only": 0,
            "dispatch_denied": 0, "ctx": [],
            "fallback_reasons": Counter(), "routing_skip_reasons": Counter(),
            "provider_context_skips": Counter(),
        },
        "cheap": {
            "sessions": 0, "requests": 0, "local": 0, "remote": 0,
            "local_only": 0, "fell_back": 0, "remote_only": 0,
            "dispatch_denied": 0, "ctx": [],
            "fallback_reasons": Counter(), "routing_skip_reasons": Counter(),
            "provider_context_skips": Counter(),
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
        label = _label_at(ev.ts)
        buckets[_bucket_key(label)]["fallback_reasons"][ev.reason] += 1
    for ev in routing_skip_events:
        if not ev.reason:
            continue
        label = _label_at(ev.ts)
        buckets[_bucket_key(label)]["routing_skip_reasons"][ev.reason] += 1
    for ev in dispatch_denied_events:
        label = _label_at(ev.ts)
        buckets[_bucket_key(label)]["dispatch_denied"] += 1
    for ev in provider_context_skip_events or ():
        label = _label_at(ev.ts)
        buckets[_bucket_key(label)]["provider_context_skips"][
            ev.provider or UNKNOWN_LABEL
        ] += 1
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
    mode_map: bucketing.ModeScheduleMap | None = None,
) -> AnalysisRun:
    """Discover log files, stream-parse them, aggregate sessions, and write
    the CSVs and report into ``output_dir``.

    ``llama_log_dir`` defaults to ``log_dir`` (the proxy and llama-server
    logs live in the same directory). llama-server eval-timing samples are
    parsed for the decode/prompt-eval speed sections; missing or unparseable
    llama-server files are skipped, never fatal.

    Fast/cheap bucketing is mode-aware (LP-0MSPZUD4G007IYGH): the mode
    timeline is reconstructed from ``Mode scheduler`` lines in the logs
    (scheduled transitions) plus the ``Grandfathering: enabled; other-mode
    config ... (current=<mode>)`` marker (manual ``POST /admin/set-mode``
    switches, LP-0MT1EE315007AKXG), and each session is bucketed by the
    profile active at its start. ``mode_map``
    may be passed explicitly (tests); otherwise it is built from the
    discovered config profiles. When the logs carry no mode transitions the
    configured ``server.priority_window`` supplies the fast/cheap schedule
    (LP-0MVKSXXIR006OBI2), and only a window with neither signal falls back to
    the legacy slot-schedule attribution.
    """
    log_dir = Path(log_dir)
    output_dir = Path(output_dir)
    if llama_log_dir is None:
        llama_log_dir = log_dir
    profiles: dict | None = None
    if mode_map is None:
        configs = config_loader.discover_configs()
        profiles = configs["profiles"]
        if config is None:
            config = configs["analysis_config"]
        mode_map = bucketing.ModeScheduleMap.from_profiles(
            configs["profiles"],
            configs["analysis_mode"],
            (config or {}).get("session_slot_pool_size"),
            priority_window=bucketing.PriorityWindow.from_config(config),
        )
    schedule = bucketing.schedule_from_config(config, (config or {}).get("session_slot_pool_size"))

    files = log_parser.discover_log_files(log_dir, window_start)
    # The effective margin covers both the busy-time pairing (1h) and the mode
    # timeline (48h) in a single streaming pass.
    margin = max(log_parser.BUSY_WINDOW_MARGIN, log_parser.MODE_TIMELINE_MARGIN)
    events = chain.from_iterable(
        log_parser.iter_events(
            f, window_start, window_end, margin=margin
        )
        for f in files
    )
    summary = aggregation.aggregate(events, window_start, window_end, schedule, mode_map)

    llama_files = llama_log_parser.discover_llama_logs(llama_log_dir, window_start)
    summary.speed = llama_log_parser.build_speed_stats(
        llama_files, window_start, window_end, schedule, mode_map, log_dir=llama_log_dir
    )

    # GPU busy-time from Prometheus power consumption (best-effort).
    if summary.busy is not None:
        ws = window_start.timestamp()
        we = window_end.timestamp()
        summary.gpu_busy = gpu_busy.compute_gpu_busy(ws, we)

    output_dir.mkdir(parents=True, exist_ok=True)
    archived_to = _archive_existing_outputs(output_dir)
    write_csvs(summary, output_dir)
    write_error_artifacts(summary, output_dir)
    report_path = output_dir / "report.md"
    report_path.write_text(
        build_report(summary, config, summary.speed, mode_map, profiles=profiles),
        encoding="utf-8",
    )
    return AnalysisRun(summary=summary, files=files, archived_to=archived_to, mode_map=mode_map)


def provider_context_skips_json(
    summary: AnalysisResult, mode_map: bucketing.ModeScheduleMap | None = None
) -> dict:
    """Provider context-ceiling skip summary for the ``--json`` output.

    LP-0MVMBM1Y3001TPRT: ``routing_skip_provider_context`` events are
    per-request size mismatches against a remote entry's ``max_context_tokens``
    (not provider failures). ``share_of_skips`` uses all skip events
    (``routing_skip_local`` + provider-context) as its denominator, matching
    the report's ``% of skips`` column. An event whose line carried no
    ``limit`` is keyed as ``"(unknown)"`` so the absence is visible instead of
    silently dropped (AC8).
    """
    events = summary.provider_context_skip_events
    schedule = bucketing.schedule_from_config(None, None)
    standard = low = 0
    by_provider: Counter = Counter()
    by_limit: Counter = Counter()
    for ev in events:
        by_provider[ev.provider or UNKNOWN_LABEL] += 1
        by_limit[str(ev.limit) if ev.limit is not None else UNKNOWN_LABEL] += 1
        if bucketing.bucket_label(ev.ts, schedule, mode_map) == "cheap":
            low += 1
        else:
            standard += 1
    total_skips = len(summary.routing_skip_events) + len(events)
    return {
        "total": len(events),
        "share_of_skips": round(len(events) / total_skips, 4) if total_skips else 0.0,
        "standard": standard,
        "low": low,
        "by_provider": dict(by_provider.most_common()),
        "by_limit": dict(by_limit.most_common()),
    }


def summary_to_json(summary: AnalysisResult, mode_map: bucketing.ModeScheduleMap | None = None) -> dict:
    """Machine-readable summary of the analysis (one dict; JSON-serialisable)."""
    sessions = list(summary.sessions.values())
    local_only = sum(1 for s in sessions if s.remote_requests == 0)
    fell_back = sum(1 for s in sessions if s.fell_back)
    remote_only = sum(1 for s in sessions if s.local_requests == 0 and s.remote_requests > 0)
    total = summary.total_requests
    # Contention-prediction block (LP-0MVGR2OXT0096OGK): computed once here and
    # shared by the three keys so the attribution split never drifts between
    # ``denial_attribution`` and ``status_capacity_hint``.
    _contention = denial_attribution_json(summary)
    # Contended-pricing split (LP-0MVS9YKHL007UBBZ AC2): the same computation
    # feeds the markdown section, so the two views cannot disagree.
    _admission = aggregation.admission_breakdown(summary)
    # Measurement provenance (LP-0MVT2QHMU0008F2V): computed once so the two
    # additive keys below are always consistent with each other.
    _measured_decisions, _unmeasured_reasons = _residency_measurement_breakdown(
        summary
    )
    return {
        "window_start": _fmt_ts(summary.window_start),
        "window_end": _fmt_ts(summary.window_end),
        "sessions": len(sessions),
        "local_only_sessions": local_only,
        "fallback_sessions": fell_back,
        "remote_only_sessions": remote_only,
        "hourly_activity": _hourly_activity_json(summary),
        "hourly_session_classification": _hourly_session_classification_json(sessions),
        "total_requests": total,
        "local_requests": summary.local_requests,
        "remote_requests": summary.remote_requests,
        "fallback_events": len(summary.fallback_events),
        "fallback_rate": round((len(summary.fallback_events) / total) if total else 0.0, 4),
        # Per-session fallback-attribution states (LP-0MVMBMSWR0036RES AC5).
        "fallback_attribution": dict(summary.fallback_attribution_counts.most_common()),
        "contention_dispatch": len(summary.contention_dispatch_events),
        "contention_fallback_after_queue": len(summary.contention_fallback_events),
        # Prefill-ETA admission guard (LP-0MVGDUV5E005UWBQ AC2): verdicts taken
        # at the contention-queue point, and how many of the refusals were
        # actually enforced (``enforced=false`` rows are an ``observe``
        # rollout, i.e. would-be refusals).
        "local_admission_reject": sum(
            1
            for e in getattr(summary, "local_admission_events", []) or []
            if getattr(e, "decision", None) == "reject"
        ),
        "local_admission_reject_enforced": sum(
            1
            for e in getattr(summary, "local_admission_events", []) or []
            if getattr(e, "decision", None) == "reject"
            and getattr(e, "enforced", None) != "false"
        ),
        "local_admission_admit": sum(
            1
            for e in getattr(summary, "local_admission_events", []) or []
            if getattr(e, "decision", None) == "admit"
        ),
        # Contended-pricing split (LP-0MVS9YKHL007UBBZ AC2): the admit/reject
        # verdicts above, split by the pricing ``rate_bucket`` the ETA was
        # calibrated from and by the admission-conversion harness's new-token
        # bands (with per-band admission → local first-byte conversion). Both
        # are ADDITIVE, and buckets/bands with no verdict are omitted rather
        # than zero-filled.
        "local_admission_by_rate_bucket": _admission["by_rate_bucket"],
        "local_admission_by_token_band": _admission["by_token_band"],
        # Idle-aware local admission (LP-0MVGD9Z8Z0055Y5J AC3): economic
        # bypasses served locally because the GPU was idle.
        "local_idle_admit": len(getattr(summary, "idle_admit_events", []) or []),
        # Residency-aware admission (LP-0MVN1IQTF0037WTG AC5): admission
        # decisions carrying a measured reusable prefix, split by verdict, plus
        # post-acceptance evictions by reason.
        "residency_decisions": len(getattr(summary, "residency_events", []) or []),
        "residency_resident_admits": sum(
            1
            for e in getattr(summary, "residency_events", []) or []
            if getattr(e, "decision", None) == "admit"
        ),
        "residency_evicted_denials": sum(
            1
            for e in getattr(summary, "residency_events", []) or []
            if getattr(e, "decision", None) == "resident_prefix_insufficient"
        ),
        "residency_tenant_limit_denials": sum(
            1
            for e in getattr(summary, "residency_events", []) or []
            if getattr(e, "decision", None) == "tenant_limit"
        ),
        # Measurement provenance (LP-0MVT2QHMU0008F2V AC4, downstream handoff to
        # LP-0MVQ4X1S30034BQ1's classifier): the measured decision count and the
        # per-reason unmeasured breakdown. Both are ADDITIVE — every key above
        # keeps its value and meaning. Reasons are the raw ``measurement=``
        # values; ``unrecorded`` covers lines predating the field.
        "residency_measured_decisions": _measured_decisions,
        "residency_unmeasured_reasons": dict(_unmeasured_reasons),
        "local_evictions": len(getattr(summary, "eviction_events", []) or []),
        # Parent AC7(a), LP-0MVN4LCQJ0034S5S AC2: accepted dispatches evicted
        # before their first byte. Both keys are ADDITIVE — nothing above
        # changes meaning or value.
        "local_evictions_after_acceptance": len(
            getattr(summary, "eviction_after_acceptance_events", []) or []
        ),
        "local_evictions_after_acceptance_by_reason": dict(
            Counter(
                ev.reason or UNKNOWN_LABEL
                for ev in (
                    getattr(summary, "eviction_after_acceptance_events", []) or []
                )
            )
        ),
        # LP-0MVOAQDB4002D56G (parent AC4): the prefill-regression checks the
        # 24 h certification needs. All ADDITIVE — no key above changes value
        # or meaning.
        #
        # ``audit`` class = a session id beginning ``audit-`` (agent audit
        # traffic from parallel gate runs); every other session, including an
        # unattributed one, is ``user`` traffic. Both are reported so the
        # re-scoped AC4 can gate on user traffic while showing the total.
        "prefill_deadline_events": len(
            getattr(summary, "prefill_deadline_events", []) or []
        ),
        "prefill_deadline_events_by_class": _prefill_deadline_by_class(summary),
        "prefill_budget_refusals": sum(
            1
            for e in getattr(summary, "local_admission_events", []) or []
            if getattr(e, "decision", None) == "reject"
            and (getattr(e, "reason", None) or "") == "prefill_budget_exceeded"
        ),
        "wasted_prefill_tokens": _wasted_prefill_tokens(summary),
        "provider_exhausted_errors": len(
            getattr(summary, "provider_exhausted_events", []) or []
        ),
        "upstream_status_503_errors": sum(
            1
            for e in getattr(summary, "error_events", []) or []
            if getattr(e, "kind", None) == "upstream_http_error"
            and getattr(e, "status", None) == 503
        ),
        "contention_queued_duration_seconds": round(
            sum(float(e.queued_duration or 0.0) for e in summary.contention_dispatch_events)
            + sum(float(e.queued_duration or 0.0) for e in summary.contention_fallback_events),
            3,
        ),
        "dispatch_denied": summary.dispatch_denied_count,
        # Contention-prediction measurements (LP-0MVGR2OXT0096OGK): M1 denial
        # attribution by guard reason/pool state, the payload's own
        # capacity_hint claim, and M3 predicted-vs-observed prefill ETA error.
        # All ADDITIVE — no key above changes value or meaning.
        "denial_attribution": _contention,
        "prefill_eta_error": prefill_eta_error_json(summary, summary.speed),
        "status_capacity_hint": _contention["capacity_hint"],
        "unattributed_events": summary.unattributed_events,
        # Measured-usage coverage (LP-0MVMHNVKT003PRIK AC4): terminal lines with
        # measured token counts vs. terminal lines with none (unknown, not zero).
        "streams_with_usage": getattr(summary, "streams_finished", 0),
        "streams_without_usage": getattr(summary, "streams_without_usage", 0),
        "streams_without_usage_by_provider": dict(
            getattr(summary, "streams_without_usage_by_provider", None) or {}
        ),
        "hold_cycle_routing_skip_events": summary.hold_cycle_reemission_count,
        "provider_context_skips": provider_context_skips_json(summary, mode_map),
        "standard_sessions": sum(1 for s in sessions if s.bucket != "cheap"),
        "low_priority_sessions": sum(1 for s in sessions if s.bucket == "cheap"),
        "errors": len(summary.error_events),
        "errors_by_type": dict(summary.error_counts.most_common()),
        "errors_by_provider_model": error_provider_model_json(summary),
        "errors_by_root_cause": root_cause_json(summary),
        "compaction": compaction_json(summary),
        "recommendations": len(recommendations.generate_recommendations(summary, None, mode_map)),
        "local_busy": _busy_json(summary.busy),
        "gpu_busy": _gpu_busy_json(summary.gpu_busy) if summary.gpu_busy else None,
        "local_prefill": _prefill_json(summary.busy),
        "decode_speed": _speed_json(summary.speed) if summary.speed else None,
        "prompt_eval_speed": _speed_json(summary.speed, "prompt_eval") if summary.speed else None,
        "ttft": _ttft_json(summary.ttft_events, summary.sessions),
        # Local-turn coverage (LP-0MVM93DE6009JGBU): the same derivation the
        # Markdown section renders, so `--json` consumers need not scrape it.
        "local_turn_coverage": local_turn_coverage(summary),
    }


def _hourly_session_classification_json(sessions: list[SessionStats]) -> list[dict]:
    """JSON-friendly per-hour session-classification counts (absolute hour).

    Keyed by the absolute top-of-hour in which each session's first request
    started (LP-0MTYAZFGN003BYQS); the report table no longer shows these
    columns, but the machine-readable summary keeps the journey split.
    """
    return [
        {
            "hour": _fmt_ts(hour),
            "local_only": counts["local_only"],
            "fell_back": counts["fell_back"],
            "remote_only": counts["remote_only"],
        }
        for hour, counts in sorted(_hourly_session_classification(sessions).items())
    ]


def _hourly_activity_json(summary: AnalysisResult) -> list[dict]:
    """JSON-friendly per-hour activity counters (absolute hour -> counts).

    Mirrors the ``## Summary by hour`` activity columns so agents can query
    per-hour traffic without scraping the Markdown (LP-0MTYAZFGN003BYQS).

    Each entry also carries the hour's local-availability classification
    (LP-0MVKUKG3F0035I85): ``state`` is the rendered ``+``-joined label and
    ``states`` the ordered tuple, accompanied by the raw evidence counters
    (``local_dispatch_attempts`` / ``local_served`` / ``local_denials`` /
    ``lease_active_fallbacks``) so a consumer can audit the verdict instead of
    trusting it.
    """
    entries: list[dict] = []
    for hour, counts in sorted(summary.hourly_activity.items()):
        availability = aggregation.hour_availability(hour, counts)
        entries.append(
            {
                "hour": _fmt_ts(hour),
                "state": availability.label,
                "states": list(availability.states),
                "requests_started": counts.get("requests_started", 0),
                "local_attempts": counts.get("local_attempts", 0),
                "local_served": counts.get("local_served", 0),
                "local_dispatch_attempts": availability.local_dispatch_attempts,
                "local_denials": availability.local_denials,
                "lease_active_fallbacks": availability.lease_active_fallbacks,
                "fallbacks": counts.get("fallbacks", 0),
                "local_skip": counts.get("local_skip", 0),
            }
        )
    return entries


def _ttft_json(
    ttft_events: list[log_parser.LogEvent],
    sessions: dict[str, SessionStats],
) -> dict | None:
    """Machine-readable TTFT summary for ``summary_to_json``.

    Returns p10 / median / p90 in milliseconds for Total/Fast/Cheap buckets
    (Fast/Cheap via mode-aware session bucketing, LP-0MUI56OCI008EGML), with
    ``null`` percentiles for a bucket that has no samples; returns ``None`` when
    no TTFT data is available.
    """
    buckets = _ttft_buckets(ttft_events, sessions)
    if not buckets["total"]:
        return None

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"samples": 0, "p10": None, "median": None, "p90": None}
        return {
            "samples": len(values),
            "p10": round(_percentile(values, 10), 1),
            "median": round(_percentile(values, 50), 1),
            "p90": round(_percentile(values, 90), 1),
        }

    return {
        "total": _stats(buckets["total"]),
        "fast": _stats(buckets["fast"]),
        "cheap": _stats(buckets["cheap"]),
    }


def _phase_json(stats: aggregation.SlotSecondsStats | None) -> dict:
    """JSON-friendly slot-seconds statistics for one local phase.

    Absent stats render as zeros/Nones (no samples), never as a missing key, so
    a JSON consumer can always read ``prefill`` and ``decode``.
    """
    if stats is None:
        return {
            "slot_seconds": 0.0,
            "pct": 0.0,
            "samples": 0,
            "unknown_streams": 0,
            "p10": None,
            "p50": None,
            "p90": None,
            "max_seconds": None,
        }
    return {
        "slot_seconds": stats.total_seconds,
        "pct": 0.0,  # filled by :func:`_prefill_vs_decode_json`
        "samples": stats.count,
        "unknown_streams": stats.unknown_count,
        "p10": stats.p10,
        "p50": stats.p50,
        "p90": stats.p90,
        "max_seconds": stats.max_seconds,
    }


def _prefill_vs_decode_json(busy: aggregation.BusyStats) -> dict:
    """Machine-readable phase slot-time split (LP-0MVJNDOUU0059MMO)."""
    _prefill_total, _decode_total, combined, prefill_pct = _phase_split(busy)
    prefill = _phase_json(busy.prefill_stats)
    decode = _phase_json(busy.decode_stats)
    prefill["pct"] = prefill_pct
    decode["pct"] = round(100.0 - prefill_pct, 1) if combined else 0.0
    return {
        "unit": aggregation.SLOT_SECONDS_UNIT,
        "combined_slot_seconds": combined,
        "prefill": prefill,
        "decode": decode,
    }


def _prefill_json(busy: aggregation.BusyStats | None) -> dict | None:
    """JSON-friendly local prefill statistics for ``summary_to_json``
    (LP-0MVJNW6SN008AKH3).

    ``None`` when the window has no local traffic (mirrors ``local_busy``); a
    zeros/Nones object when there is local traffic but no measurable prefill,
    so a consumer can always read the keys. Every duration is
    **slot-seconds** (the sum of the per-stream prefill work, not the
    wall-clock union on :class:`aggregation.BusyStats`), labelled via
    ``unit``; streams without an observable prefill are counted in
    ``unknown_prefill`` and excluded from the percentiles.
    """
    if busy is None:
        return None
    stats = busy.prefill_stats
    if stats is None or not stats.has_samples:
        return {
            "unit": aggregation.SLOT_SECONDS_UNIT,
            "slot_seconds": 0.0,
            "count": 0,
            "streams": stats.stream_count if stats is not None else 0,
            "unknown_prefill": stats.unknown_count if stats is not None else 0,
            "p10": None,
            "p50": None,
            "p90": None,
            "max_seconds": None,
        }
    return {
        "unit": aggregation.SLOT_SECONDS_UNIT,
        "slot_seconds": stats.total_seconds,
        "count": stats.count,
        "streams": stats.stream_count,
        "unknown_prefill": stats.unknown_count,
        "p10": stats.p10,
        "p50": stats.p50,
        "p90": stats.p90,
        "max_seconds": stats.max_seconds,
    }


def _busy_json(busy: aggregation.BusyStats | None) -> dict | None:
    """JSON-friendly local-model utilization summary (None when no local traffic).

    ``busy_seconds`` is the **dispatch → finish** union
    (LP-0MVJN9H2I00116FP); the retained decode-only union lands in
    ``decode_busy_seconds`` (the labelled lower bound), the wall-clock
    prefill span (upper − lower) in ``prefill_busy_seconds``, and the
    definition marker in ``definition`` so a consumer never needs to infer
    the semantics (LP-0MVJNW6SN008AKH3).
    """
    if busy is None:
        return None
    return {
        "window_seconds": busy.window_seconds,
        "busy_seconds": busy.busy_seconds,
        "busy_pct": round(busy.busy_pct, 1),
        # Busy definition (LP-0MVJNW6SN008AKH3): the dispatch → finish
        # union, with the retained decode-only figure as the lower bound and
        # the wall-clock prefill span (upper − lower) alongside.
        "definition": busy.busy_definition,
        "decode_busy_seconds": busy.decode_busy_seconds,
        "prefill_busy_seconds": busy.prefill_busy_seconds,
        "prefill_extended_streams": busy.prefill_extended_streams,
        "upper_bound_label": aggregation.BUSY_UPPER_BOUND_LABEL,
        "lower_bound_label": aggregation.BUSY_LOWER_BOUND_LABEL,
        "idle_seconds": round(busy.idle_seconds, 1),
        "idle_pct": round(busy.idle_pct, 1),
        "total_compute_seconds": busy.total_compute_seconds,
        "streams": busy.streams,
        "avg_stream_duration_seconds": busy.avg_stream_duration,
        "peak_concurrency": busy.peak_concurrency,
        "avg_concurrency": busy.avg_concurrency,
        "unfinished_streams": busy.unfinished_streams,
        "pre_window_unfinished": busy.pre_window_unfinished,
        "fast_busy_seconds": busy.fast_busy_seconds,
        "cheap_busy_seconds": busy.cheap_busy_seconds,
        "fast_window_seconds": busy.fast_window_seconds,
        "cheap_window_seconds": busy.cheap_window_seconds,
        "hourly_busy": [
            {"hour": _fmt_ts(hour), "seconds": seconds}
            for hour, seconds in busy.hourly_busy
        ],
        # Phase slot-time split (LP-0MVJNDOUU0059MMO): prefill vs decode as
        # slot-seconds and % of combined phase slot-time (sums to 100%).
        "prefill_vs_decode": _prefill_vs_decode_json(busy),
    }


def _gpu_busy_json(gpu: object | None) -> dict | None:
    """JSON-friendly GPU busy-time from Prometheus power consumption.

    Supplementary hardware telemetry, **not** a local-utilisation signal:
    measures when the GPU drew compute-level power (``gpu:computing:bool``),
    independent of proxy log events, so it counts GPU work outside the
    proxy's slot accounting. Decisions use the log-derived ``local_busy``
    (``dispatch → finish``); this block is its hardware-side complement.
    Returns ``None`` when the query fails.
    """
    if gpu is None:
        return None
    return {
        "computing_fraction": round(gpu.computing_fraction, 4),
        "busy_seconds": round(gpu.busy_seconds, 1),
        "idle_seconds": round(gpu.idle_seconds, 1),
        "busy_pct": round(gpu.busy_pct, 1),
        "idle_pct": round(gpu.idle_pct, 1),
        "total_seconds": round(gpu.total_seconds, 1),
        # Smoothed power trace (1-min samples, derived from 15-s raw).
        "smoothed_power": [
            {"timestamp": int(ts), "power_watts": round(w, 2)}
            for ts, w in gpu.smoothed_power_values
        ],
    }


def _matched_prefill_eta(summary: AnalysisResult) -> dict:
    """Matched per-dispatch prefill-ETA accuracy (LP-0MVSD5RJW008F1IB AC1).

    Joins each local dispatch's first byte to the ETA the *admission guard
    priced for that same dispatch*: predicted = the guard record's
    ``eta_prefill_s`` (the own-prefill term — not ``projected_wait_s``, which
    adds occupancy the observed first byte already includes once), observed =
    that dispatch's ``dispatch_to_first_byte_ms``. The join key is the
    ``session=`` identifier both lines carry.

    Reporting rules from the brief:

    - a dispatch with no matching guard record is **unmatched** and never
      folded into the matched figure (the guard is inert for uncalibrated
      buckets and logs nothing for inactive observe-mode decisions);
    - observe-policy pairs (a would-be refusal priced without enforcement)
      are valid *pricing* measurements and are included, but counted
      separately from ordinary admitted dispatches so a rollout window stays
      interpretable. The log's ``enforced`` token means "the refusal was
      applied" — it is ``false`` for every admitted request, so the ``policy``
      token decides the split;
    - an **enforced refusal never pairs**: it refused that request, so a later
      first byte in the same session belongs to a different request. Those
      records are counted in ``enforced_refusals`` and excluded from the
      candidate set (the dispatch becomes unmatched);
    - guard records whose priced ETA is ``0`` (the calibration-independent
      cold-prefill budget refusal) are not estimates and are skipped.
    """
    def _num(value) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else None

    guards: dict[str, list[tuple[datetime, float, str | None]]] = {}
    enforced_refusals = 0
    for event in getattr(summary, "local_admission_events", []) or []:
        session = getattr(event, "session", None)
        eta = _num(getattr(event, "eta_prefill_s", None))
        if not session or eta is None or eta <= 0.0:
            continue
        if getattr(event, "enforced", None) == "true":
            # A refusal that was actually applied: the request did not
            # dispatch, so it is never a pairing candidate.
            enforced_refusals += 1
            continue
        guards.setdefault(session, []).append(
            (event.ts, eta, getattr(event, "policy", None))
        )
    for entries in guards.values():
        entries.sort(key=lambda item: item[0])

    dispatches = 0
    unmatched = 0
    pairs: list[tuple[float, float, str | None]] = []
    for event in getattr(summary, "ttft_events", []) or []:
        if getattr(event, "ttft_source", None) not in (None, "local"):
            continue
        service_ms = _num(getattr(event, "dispatch_to_first_byte_ms", None))
        session = getattr(event, "session", None)
        if service_ms is None or service_ms <= 0.0:
            continue
        dispatches += 1
        candidates = guards.get(session) if session else None
        if not candidates:
            unmatched += 1
            continue
        # The guard fires at admission, before the first byte: take the last
        # decision at or before it (falling back to the earliest recorded one).
        chosen = candidates[0]
        for candidate in candidates:
            if candidate[0] <= event.ts:
                chosen = candidate
        pairs.append((chosen[1], service_ms / 1000.0, chosen[2]))

    predicted = [p for p, _o, _pol in pairs]
    observed = [o for _p, o, _pol in pairs]
    relative = [
        (p - o) / o * 100.0 for p, o in zip(predicted, observed) if o > 0
    ]
    rel_p50 = _percentile(relative, 50)
    rel_p90 = _percentile(relative, 90)
    if not pairs or rel_p50 is None:
        verdict = "inconclusive"
    elif abs(rel_p50) <= PREFILL_ETA_MATCH_TOLERANCE_PCT:
        verdict = "matches_reality"
    elif rel_p50 > 0:
        verdict = "over_predicts"
    else:
        verdict = "under_predicts"

    def _r(value: float | None) -> float | None:
        return round(value, 2) if value is not None else None

    return {
        "pairs": len(pairs),
        "dispatches": dispatches,
        "unmatched_dispatches": unmatched,
        # Mutually exclusive split of the matched pairs: an observe-policy
        # would-be refusal, or an ordinary admitted dispatch. Enforced
        # refusals are excluded from pairing entirely (``enforced_refusals``
        # counts the guard records dropped for that reason).
        "admitted_pairs": sum(1 for _p, _o, pol in pairs if pol != "observe"),
        "observe_pairs": sum(1 for _p, _o, pol in pairs if pol == "observe"),
        "enforced_refusals": enforced_refusals,
        "predicted_p50": _r(_percentile(predicted, 50)),
        "predicted_p90": _r(_percentile(predicted, 90)),
        "observed_p50": _r(_percentile(observed, 50)),
        "observed_p90": _r(_percentile(observed, 90)),
        "relative_error_p50_pct": _r(rel_p50),
        "relative_error_p90_pct": _r(rel_p90),
        "tolerance_pct": PREFILL_ETA_MATCH_TOLERANCE_PCT,
        "verdict": verdict,
    }


def prefill_eta_error_json(
    summary: AnalysisResult, speed: llama_log_parser.SpeedStats | None
) -> dict:
    """M3 (LP-0MVGR2OXT0096OGK): predicted-vs-observed prefill ETA error.

    Two comparisons, because a cache-cold reference bound and a per-request
    estimate answer different questions (LP-0MVSD5RJW008F1IB AC1/AC3):

    - **bound vs served remainder** (the historical quantile-matched table).
      Predicted: every non-``None`` ``prefill_eta_seconds`` published on
      ``status_request`` polls inside the window. Observed: the llama-server
      ``prompt eval time`` durations (seconds) for the dispatches that ran.
      The two streams cannot be joined 1:1 (llama-server eval lines carry no
      per-line timestamps — see REFERENCE.md limitations), so the comparison
      is quantile-matched. The prediction is a cache-cold reference-prompt
      *bound* while the observation is the *uncached remainder* of mostly-warm
      prompts, so a large gap is expected; the only meaningful question is
      whether the bound is **violated** (observed p90 above predicted p50).
      That is reported one-sided as ``bound_holds`` / ``bound_violated`` /
      ``inconclusive`` (below :data:`PREFILL_ETA_BOUND_MIN_SAMPLES` observed
      samples) — never as a two-sided calibration verdict.
    - **matched per-dispatch accuracy** (``matched``): the ETA the guard priced
      for a dispatch vs that same dispatch's ``dispatch_to_first_byte_ms``,
      where the 25 % two-sided tolerance *is* meaningful
      (:func:`_matched_prefill_eta`).

    The pre-existing keys (``predicted`` / ``observed`` / ``error_p50`` /
    ``error_p90`` / ``relative_error_*`` / ``tolerance_pct`` / ``verdict``)
    keep their names and shapes for trend continuity; ``verdict`` now reports
    the one-sided bound test, and ``comparison`` states which comparison it
    describes.
    """
    predicted = sorted(
        float(e.prefill_eta_seconds)
        for e in getattr(summary, "status_request_events", []) or []
        if getattr(e, "prefill_eta_seconds", None) is not None
    )
    observed = sorted(float(v) for v in (getattr(speed, "prompt_eval_seconds", None) or []))
    pred_p50 = _percentile(predicted, 50)
    pred_p90 = _percentile(predicted, 90)
    obs_p50 = _percentile(observed, 50)
    obs_p90 = _percentile(observed, 90)
    # Duration percentiles are reported to 2 dp (seconds) so JSON consumers
    # never see float-representation dust like 8.600000000000001.
    pred_p50 = round(pred_p50, 2) if pred_p50 is not None else None
    pred_p90 = round(pred_p90, 2) if pred_p90 is not None else None
    obs_p50 = round(obs_p50, 2) if obs_p50 is not None else None
    obs_p90 = round(obs_p90, 2) if obs_p90 is not None else None

    def _error(pred: float | None, obs: float | None) -> float | None:
        if pred is None or obs is None:
            return None
        return round(pred - obs, 1)

    def _relative(pred: float | None, obs: float | None) -> float | None:
        if pred is None or obs is None or not obs:
            return None
        return round((pred - obs) / obs * 100.0, 1)

    err_p50 = _error(pred_p50, obs_p50)
    err_p90 = _error(pred_p90, obs_p90)
    rel_p50 = _relative(pred_p50, obs_p50)
    rel_p90 = _relative(pred_p90, obs_p90)
    # One-sided bound test (Q3 = replace the two-sided tolerance here): the
    # cache-cold bound must not be exceeded by the served p90. Below the
    # sample floor the window cannot decide, and ``inconclusive`` says so.
    if (
        pred_p50 is None
        or obs_p90 is None
        or len(observed) < PREFILL_ETA_BOUND_MIN_SAMPLES
    ):
        bound_verdict = "inconclusive"
        bound_holds: bool | None = None
    elif obs_p90 <= pred_p50:
        bound_verdict = "bound_holds"
        bound_holds = True
    else:
        bound_verdict = "bound_violated"
        bound_holds = False
    return {
        "comparison": "bound_vs_served_remainder",
        "predicted": {
            "samples": len(predicted),
            "p50": pred_p50,
            "p90": pred_p90,
        },
        "observed": {
            "samples": len(observed),
            "p50": obs_p50,
            "p90": obs_p90,
        },
        "error_p50": err_p50,
        "error_p90": err_p90,
        "relative_error_p50_pct": rel_p50,
        "relative_error_p90_pct": rel_p90,
        "tolerance_pct": PREFILL_ETA_MATCH_TOLERANCE_PCT,
        # One-sided bound test (AC3): observed p90 must stay within the
        # published p50 bound.
        "bound": {
            "verdict": bound_verdict,
            "holds": bound_holds,
            "predicted_p50": pred_p50,
            "observed_p90": obs_p90,
            "observed_samples": len(observed),
            "min_samples": PREFILL_ETA_BOUND_MIN_SAMPLES,
            "margin_seconds": (
                round(pred_p50 - obs_p90, 2)
                if pred_p50 is not None and obs_p90 is not None
                else None
            ),
        },
        "verdict": bound_verdict,
        # Matched per-dispatch accuracy (AC1): the 25 % two-sided tolerance
        # applies here, where predicted and observed describe one dispatch.
        "matched": _matched_prefill_eta(summary),
    }


def denial_attribution_json(summary: AnalysisResult) -> dict:
    """M1/M2 JSON surface (LP-0MVGR2OXT0096OGK): denial attribution split.

    ``local_dispatch_denied`` lines carry no ``lease_state``; each is joined to
    its decision-block ``local_admission_guard`` reject (nearest within
    :data:`aggregation.DENIAL_GUARD_JOIN_WINDOW_SECONDS`, inclusive) and the
    pool-state verdict is derived from the guard's measured fields. Denials
    with no guard in the block are ``unattributed``. Per-hour rows pair the
    attribution split with the hour's local busy fraction (M2 corroboration:
    a denial hour whose busy% is ~0 is a false-busy hour).

    ``capacity_hint`` — the payload field under test — is summarised from the
    ``status_request`` polls so the payload's own claim is compared against
    the measured reality (the same ``denials`` can coincide with an
    ``idle``-claimed pool).
    """
    attrs = getattr(summary, "denial_attributions", []) or []
    statuses = getattr(summary, "status_request_events", []) or []
    denials_total = int(summary.dispatch_denied_count)
    attributed = len(attrs)
    by_verdict: Counter[str] = Counter(a.verdict for a in attrs)
    by_reason: Counter[str] = Counter(a.guard_reason for a in attrs if a.guard_reason)
    capacity: Counter[str] = Counter(e.capacity_hint for e in statuses if e.capacity_hint)

    busy_hourly = dict(
        getattr(getattr(summary, "busy", None), "hourly_busy", []) or []
    )
    denials_by_hour: Counter[datetime] = Counter(
        e.ts.replace(minute=0, second=0, microsecond=0)
        for e in getattr(summary, "dispatch_denied_events", []) or []
    )
    verdicts_by_hour: dict[datetime, Counter[str]] = {}
    for a in attrs:
        hk = a.ts.replace(minute=0, second=0, microsecond=0)
        verdicts_by_hour.setdefault(hk, Counter())[a.verdict] += 1

    per_hour: list[dict] = []
    for hk in sorted(denials_by_hour):
        window_seconds = max(
            0.0,
            (
                min(summary.window_end, hk + timedelta(hours=1))
                - max(summary.window_start, hk)
            ).total_seconds(),
        )
        busy_seconds = float(busy_hourly.get(hk, 0.0))
        vh = verdicts_by_hour.get(hk, Counter())
        row: dict = {
            "hour": _fmt_ts(hk),
            "denials": denials_by_hour[hk],
            "attributed": sum(vh.values()),
            "unattributed": denials_by_hour[hk] - sum(vh.values()),
            "busy_seconds": round(busy_seconds, 1),
            "busy_pct": round(busy_seconds / window_seconds * 100.0, 1) if window_seconds else 0.0,
        }
        for verdict in aggregation.DENIAL_VERDICT_ORDER:
            row[verdict] = vh.get(verdict, 0)
        per_hour.append(row)

    idle_share_pct = (
        round(by_verdict.get(aggregation.DENIAL_VERDICT_IDLE_LEASED, 0) / attributed * 100.0, 1)
        if attributed
        else None
    )
    if not attributed:
        verdict = "inconclusive"
    elif idle_share_pct is not None and idle_share_pct >= IDLE_LEASED_SHARE_THRESHOLD_PCT:
        verdict = "does_not_match_reality"
    else:
        verdict = "matches_reality"

    return {
        "denials": denials_total,
        "attributed": attributed,
        "unattributed": denials_total - attributed,
        "join_window_seconds": aggregation.DENIAL_GUARD_JOIN_WINDOW_SECONDS,
        "idle_leased_share_pct": idle_share_pct,
        "by_verdict": {v: by_verdict.get(v, 0) for v in aggregation.DENIAL_VERDICT_ORDER},
        "by_guard_reason": dict(sorted(by_reason.items(), key=lambda kv: (-kv[1], kv[0]))),
        "per_hour": per_hour,
        "capacity_hint": dict(sorted(capacity.items(), key=lambda kv: (-kv[1], kv[0]))),
        "verdict": verdict,
    }


def compaction_json(summary: AnalysisResult) -> dict:
    """Machine-readable compaction summary for ``summary_to_json``.

    Built from ``AnalysisResult.compaction_events`` and the session stats;
    suitable for ``--json | jq .compaction``. Structure mirrors the report
    section: top-line counts, per-bucket / per-action breakdown, token and
    turn economics, and fallback-avoidance impact. No mode-timeline
    approximation is baked in — it is streaming-discovered at report time.
    """
    events = summary.compaction_events
    impact = aggregation.compaction_impact(summary.sessions, events, None)
    pre_sum = 0
    post_sum = 0
    counted = 0
    for e in events:
        if e.pre_tokens is not None and e.post_tokens is not None:
            pre_sum += e.pre_tokens
            post_sum += e.post_tokens
            counted += 1
    avg_pre = round(pre_sum / counted) if counted else None
    avg_post = round(post_sum / counted) if counted else None
    return {
        "events": len(events),
        "compact_events": sum(1 for e in events if e.kind == "compaction_event"),
        "backstop_events": sum(1 for e in events if e.kind == "compaction_backstop"),
        "churn_events": sum(1 for e in events if e.kind == "compaction_churn"),
        "sessions_with_compaction": impact["sessions_with_compaction"],
        "by_action": impact["by_action"],
        "by_reason": impact["by_reason"],
        "bucket_counts": impact["bucket_counts"],
        "bucket_dry_live": impact["bucket_dry_live"],
        "dry_run": impact["dry_run"],
        "avg_pre_tokens": avg_pre,
        "avg_post_tokens": avg_post,
        "avg_tokens_saved": (avg_pre - avg_post) if (avg_pre is not None and avg_post is not None) else None,
        "stayed_local": impact["stayed_local"],
        "fell_back": impact["fell_back"],
        "would_have_avoided": impact["would_have_avoided"],
        "would_still_fallback": impact["would_still_fallback"],
        "avoided": impact["avoided"],
        "backstop_sessions": impact["backstop_sessions"],
    }


def _speed_json(speed: llama_log_parser.SpeedStats, kind: str = "decode") -> dict:
    """JSON-friendly speed summary (total bucket only), plus the diagnostic
    context that lets an empty window explain itself (LP-0MVJNIZRO008S031):
    the log directory scanned, the files discovered/outside/unrecognised and
    the per-file scan outcomes.
    """
    buckets = speed.decode if kind == "decode" else speed.prompt_eval
    b = buckets[TOTAL]
    out = {
        "samples": b.count,
        "median_tok_s": b.median,
        "p90_tok_s": b.p90,
        "p10_tok_s": b.p10,
        "files_parsed": speed.files_parsed,
        "files_skipped": speed.files_skipped,
    }
    out["log_dir"] = speed.log_dir
    out["files_discovered"] = speed.files_discovered
    out["files_outside_window"] = list(speed.outside_window_files or [])
    out["files_unrecognised"] = list(speed.ignored_logs or [])
    out["qwen3_ports"] = list(speed.qwen3_ports or [])
    out["eval_lines_total"] = sum(d.eval_lines for d in speed.file_diagnostics)
    out["eval_lines_matched"] = sum(d.samples_in_window for d in speed.file_diagnostics)
    out["file_diagnostics"] = [
        {
            "file": d.name,
            "status": d.status,
            "reason": d.reason,
            "port": d.port,
            "port_source": d.port_source,
            "mtime": d.mtime,
            "lines_scanned": d.lines_scanned,
            "eval_lines": d.eval_lines,
            "eval_lines_matched": d.eval_lines_matched,
            "samples_in_window": d.samples_in_window,
            "ports_seen": d.ports_seen,
        }
        for d in speed.file_diagnostics
    ]
    if b.count == 0:
        out["empty_window_reasons"] = llama_log_parser.empty_window_reasons(speed)
    return out
