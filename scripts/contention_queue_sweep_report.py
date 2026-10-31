#!/usr/bin/env python3
"""Residual contention-queue report for the 60 s → 30 s sweep.

Measurement tool behind LP-0MVT110PJ003XYLP (child of LP-0MVIKPRMD000Y97R, the
contention-queue sweep). The sweep shortens
``contention_queue_max_wait_seconds`` (60 → 30) and
``contention_queue_max_depth`` (8 → 2) so the queue becomes a straggler net
instead of a 60 s parking lot. Its flip criteria are stated against **one
windowed definition**, so a ≥ 24 h pre-flip baseline at 60 s / depth 8 and the
post-flip verdict are directly comparable — that definition is this report.

It is a pure log analyser: it never changes proxy behaviour and never requires
the proxy to be running. It scans an arbitrary window of the proxy log stream
(``/var/log/llama-proxy`` — live plus rotated files under either naming scheme,
plain or ``.gz``) via :mod:`scripts.lib.proxy_logs`, and reports three
sections:

**(a) Ticket-backed dispatches** — ``local_admission_reserve_claimed`` lines
(a cooperating client presented a reservation ticket and skipped the queue).
Reported for context, **not** a flip gate: neither flip criterion is phrased
against it.

**(b) Residual queue wins** — ``contention_queue_dispatch`` lines (a queued
request got a local slot within the budget). Count, the share whose
``queued_duration`` is at or below ``--sweep-budget-seconds`` (the budget the
sweep applies — this is the "what would the shorter budget still have won"
number), and the wait distribution (p50/p75/p90/max).

**(c) Residual queue losses** — ``contention_queue_fallback_after_queue``
lines (the request parked and then went remote anyway: dead wait). Count,
aggregate wait, wait-per-hour and the split by ``reason``. Only the two
lease-retention emit sites in ``proxy/proxy/provider.py`` carry
``reason=local_lease_active``; a line with no ``reason=`` is counted as
``none`` and a line with any other reason is counted under that reason — no
loss is ever dropped, so the split always sums to the loss count.

The residual framing matters: "residual" is what the queue still does once
cooperating clients reserve capacity, and the loss split says how much of the
remaining dead wait is the lease-retention branch (an owner-level problem) as
opposed to genuine capacity contention.

Telemetry contract: the **log events are the source of truth**; this tool adds
no status counters (the cumulative ones were deliberately removed in
LP-0MU4NACW9001WKBP).

Usage::

  # Last 24 h (the default), human-readable:
  ./scripts/contention_queue_sweep_report.py

  # A stated pre-flip baseline window, machine readable:
  ./scripts/contention_queue_sweep_report.py \\
      --since '2026-10-30 05:00' --until '2026-10-31 05:00' --json \\
      > docs/dev/contention-queue-sweep-baseline.json

  # Markdown section for a work-item comment:
  ./scripts/contention_queue_sweep_report.py --hours 24 --markdown

Exit codes (mirroring ``scripts/idle_admit_impact.py``):
  0 - measurement complete
  2 - measurement incomplete (no logs / thin window / coverage gap) — the
      numbers are still printed, but the run never silently passes
  1 - invalid input (bad path, bad timestamp, bad budget/window)

Parent: LP-0MVIKPRMD000Y97R.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from lib.proxy_logs import (  # noqa: E402
    discover_proxy_log_files,
    iter_proxy_log_lines,
    measure_log_coverage,
    parse_line_timestamp,
)

_REPO_ROOT = SCRIPT_DIR.parent
_DEFAULT_LOG_DIR = Path("/var/log/llama-proxy")

#: The budget the sweep applies (``contention_queue_max_wait_seconds`` 60 → 30).
_DEFAULT_SWEEP_BUDGET_S = 30.0

#: The baseline precondition the parent's AC1 trigger demands: >= 24 h at
#: 60 s / depth 8 before the flip.
_DEFAULT_MIN_HOURS = 24.0

#: Adjacent-file log-coverage hole that makes a window unmeasurable (the
#: 2026-10-23 18:29 → 22:04 shape guarded by ``measure_log_coverage``).
_DEFAULT_MAX_GAP_S = 3600.0

DEFAULT_HOURS = 24.0

#: Log message bodies (the text after the logging prefix) this report counts.
_MARKER_TICKET = "local_admission_reserve_claimed"
_MARKER_WIN = "contention_queue_dispatch"
_MARKER_LOSS = "contention_queue_fallback_after_queue"

#: ``reason=local_lease_active`` on both loss/win lines, and ``none`` when the
#: line carries no ``reason=`` at all. The two groups the flip discussion is
#: phrased in: lease-active losses vs everything else.
_REASON_LEASE_ACTIVE = "local_lease_active"
_REASON_NONE = "none"

#: Prefix of a logging *message body*: ``<asctime> - [<logger> - ]<LEVEL> - ``.
#: The anchor matters: a proxied request body (an agent prompt quoting a probe
#: line) can contain a marker verbatim, and an unanchored substring match counts
#: it (the LP-0MVV9JWL9003A6XR inflation class). Only the **leftmost** prefix is
#: considered, so text inside a message can never be mistaken for the event.
_LOG_MESSAGE_PREFIX = re.compile(
    r"(?:^|\s-\s)(?:[A-Za-z0-9_.]+\s-\s)?"
    r"(?:DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\s-\s+"
)


# ---------------------------------------------------------------------------
# Read-only reuse of the canonical log parser
# ---------------------------------------------------------------------------

#: The ``proxy-usage-analysis`` skill's parser is the behavioural reference for
#: the contention-queue line shape (LP-0MSORQVK50012Q4D F4 AC3). It lives under
#: ``.pi/`` agent infrastructure and is loaded (never modified, never vendored)
#: so the field extractors cannot drift from what the proxy emits.
_SKILL_LOG_PARSER_PATH = (
    _REPO_ROOT / ".pi" / "skills" / "proxy-usage-analysis" / "scripts" / "log_parser.py"
)


def _load_skill_log_parser():
    """Load the skill log parser read-only, or return ``None``.

    A checkout that does not ship ``.pi/`` (an installed package, a vendored
    copy) degrades to the local fallback patterns rather than failing the run;
    :data:`PARSER_SOURCE` records which one was used so a report is never
    ambiguous about the field definitions behind its numbers.
    """
    if not _SKILL_LOG_PARSER_PATH.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "_proxy_usage_analysis_log_parser", _SKILL_LOG_PARSER_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        return None
    module = importlib.util.module_from_spec(spec)
    # The module must be registered before execution: it defines a dataclass,
    # and ``dataclasses`` resolves ``cls.__module__`` through ``sys.modules``
    # (a missing entry raises AttributeError on Python 3.12).
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # pragma: no cover - a broken skill module must not break the report
        sys.modules.pop(spec.name, None)
        return None
    return module


_SKILL_LOG_PARSER = _load_skill_log_parser()

#: ``session=<id>`` — ticket and queue lines all carry it.
_SESSION_RE = getattr(_SKILL_LOG_PARSER, "RE_SESSION", None) or re.compile(
    r"\bsession=([A-Za-z0-9_.:-]+)"
)
#: ``queued_duration=12.34s`` — seconds the request waited in the queue.
_QUEUED_RE = getattr(_SKILL_LOG_PARSER, "RE_CONTENTION_DURATION", None) or re.compile(
    r"queued_duration=([\d.]+)s"
)
#: Bare ``reason=<token>`` (the queue lines carry no routing-skip arrow).
_REASON_RE = getattr(_SKILL_LOG_PARSER, "RE_REASON_KV", None) or re.compile(
    r"\breason=([\w_]+)"
)

PARSER_SOURCE = (
    "proxy-usage-analysis/log_parser"
    if _SKILL_LOG_PARSER is not None
    else "builtin"
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueueEvent:
    """One residual contention-queue outcome inside the window."""

    kind: str  # "win" (dispatched local) | "loss" (fell back after queueing)
    ts: dt.datetime
    session: str | None
    queued_seconds: float
    reason: str  # raw ``reason=`` token, or "none" when the line carries none


def _message_body(line: str) -> str | None:
    """Return the text after the log line's leftmost message prefix."""
    match = _LOG_MESSAGE_PREFIX.search(line)
    if match is None:
        return None
    return line[match.end() :]


def _starts_with_marker(body: str, marker: str) -> bool:
    """Whether *body* begins with *marker* as a whole token, not a prefix of one."""
    if not body.startswith(marker):
        return False
    rest = body[len(marker) :]
    return rest == "" or rest[0] in " \t"


def _first(pattern: re.Pattern, text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def _queued_seconds(body: str) -> float | None:
    raw = _first(_QUEUED_RE, body)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:  # pragma: no cover - regex guarantees a numeric shape
        return None


def parse_queue_event(line: str, ts: dt.datetime) -> QueueEvent | None:
    """Parse a contention-queue outcome line, or ``None`` when it is not one.

    Tolerant by design: a missing ``queued_duration`` degrades to a 0 s wait and
    an absent ``reason=`` becomes ``none`` rather than dropping the event, so
    log-format drift cannot silently remove losses from the split.
    """
    body = _message_body(line)
    if body is None:
        return None
    if _starts_with_marker(body, _MARKER_WIN):
        kind = "win"
    elif _starts_with_marker(body, _MARKER_LOSS):
        kind = "loss"
    else:
        return None
    return QueueEvent(
        kind=kind,
        ts=ts,
        session=_first(_SESSION_RE, body),
        queued_seconds=_queued_seconds(body) or 0.0,
        reason=_first(_REASON_RE, body) or _REASON_NONE,
    )


def is_ticket_claim(line: str) -> bool:
    """Whether *line* is an anchored ``local_admission_reserve_claimed`` event."""
    body = _message_body(line)
    return body is not None and _starts_with_marker(body, _MARKER_TICKET)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile; an empty list yields 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _wait_stats(values: list[float]) -> dict:
    """Count/sum/mean plus the p50/p75/p90/max wait distribution, in seconds."""
    if not values:
        return {
            "n": 0,
            "total": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    return {
        "n": len(values),
        "total": round(sum(values), 2),
        "mean": round(sum(values) / len(values), 2),
        "p50": round(_percentile(values, 50), 2),
        "p75": round(_percentile(values, 75), 2),
        "p90": round(_percentile(values, 90), 2),
        "max": round(max(values), 2),
    }


def _per_hour(count: float, hours: float) -> float:
    return round(count / hours, 2) if hours > 0 else 0.0


def _share_pct(part: float, whole: float) -> float | None:
    """``part`` as a percentage of ``whole``; ``None`` when ``whole`` is zero."""
    if whole <= 0:
        return None
    return round(100.0 * part / whole, 1)


def summarise(
    events: list[QueueEvent],
    *,
    ticket_count: int,
    sweep_budget_seconds: float,
    window_hours: float,
) -> dict:
    """Build the three report sections plus the queue-level denominator."""
    wins = [event for event in events if event.kind == "win"]
    losses = [event for event in events if event.kind == "loss"]
    win_waits = [event.queued_seconds for event in wins]
    loss_waits = [event.queued_seconds for event in losses]

    at_or_below = sum(1 for wait in win_waits if wait <= sweep_budget_seconds)

    by_reason: dict[str, dict] = {}
    for event in losses:
        bucket = by_reason.setdefault(
            event.reason,
            {"count": 0, "aggregate_wait_seconds": 0.0},
        )
        bucket["count"] += 1
        bucket["aggregate_wait_seconds"] += event.queued_seconds
    for reason, bucket in by_reason.items():
        bucket["aggregate_wait_seconds"] = round(bucket["aggregate_wait_seconds"], 2)
        bucket["mean_wait_seconds"] = round(
            bucket["aggregate_wait_seconds"] / bucket["count"], 2
        )
        bucket["share_of_losses_pct"] = _share_pct(bucket["count"], len(losses))
    by_reason = dict(
        sorted(by_reason.items(), key=lambda item: (-item[1]["count"], item[0]))
    )

    queue_events = len(wins) + len(losses)
    return {
        "ticket_backed": {
            "count": ticket_count,
            "per_hour": _per_hour(ticket_count, window_hours),
            "note": (
                "reported for context only — ticket-backed dispatches are not a "
                "flip gate for the sweep"
            ),
        },
        "residual_wins": {
            "count": len(wins),
            "per_hour": _per_hour(len(wins), window_hours),
            "sweep_budget_seconds": sweep_budget_seconds,
            "at_or_below_budget": at_or_below,
            "over_budget": len(wins) - at_or_below,
            "share_at_or_below_budget_pct": _share_pct(at_or_below, len(wins)),
            "wait_seconds": _wait_stats(win_waits),
        },
        "residual_losses": {
            "count": len(losses),
            "per_hour": _per_hour(len(losses), window_hours),
            "aggregate_wait_seconds": round(sum(loss_waits), 2),
            "wait_per_hour_seconds": _per_hour(sum(loss_waits), window_hours),
            "mean_wait_seconds": round(sum(loss_waits) / len(losses), 2)
            if losses
            else 0.0,
            "wait_seconds": _wait_stats(loss_waits),
            "by_reason": by_reason,
            "reason_groups": {
                _REASON_LEASE_ACTIVE: by_reason.get(_REASON_LEASE_ACTIVE, {}).get(
                    "count", 0
                ),
                _REASON_NONE: by_reason.get(_REASON_NONE, {}).get("count", 0),
                "other": sum(
                    bucket["count"]
                    for reason, bucket in by_reason.items()
                    if reason not in (_REASON_LEASE_ACTIVE, _REASON_NONE)
                ),
            },
        },
        "queue_events": {
            "count": queue_events,
            "per_hour": _per_hour(queue_events, window_hours),
            "win_share_pct": _share_pct(len(wins), queue_events),
            "loss_share_pct": _share_pct(len(losses), queue_events),
        },
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _pct_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value}%"


def render(meta: dict, sections: dict, status: dict) -> str:
    window = meta["window"]
    coverage = meta["coverage"]
    wins = sections["residual_wins"]
    losses = sections["residual_losses"]
    queue = sections["queue_events"]

    out: list[str] = []
    out.append("Contention-queue sweep report (LP-0MVT110PJ003XYLP)")
    out.append("=" * 66)
    out.append(f"log_dir     : {meta['log_dir']}")
    out.append(
        f"window      : {window['since']} .. {window['until']} "
        f"({window['hours']} h)"
    )
    out.append(f"sweep budget: {meta['sweep_budget_seconds']} s")
    out.append(
        f"coverage    : {coverage['files']} file(s) with data, "
        f"{len(coverage['gaps'])} adjacent-file gap(s), "
        f"data {coverage['data_start'] or '(none)'} .. {coverage['data_end'] or '(none)'}"
    )
    out.append(f"parser      : {meta['parser']}")
    out.append(f"generated_at: {meta['generated_at']}")
    out.append("")

    out.append("(a) Ticket-backed dispatches — local_admission_reserve_claimed")
    out.append("-" * 66)
    tickets = sections["ticket_backed"]
    out.append(
        f"count        : {tickets['count']}  ({tickets['per_hour']}/h)"
    )
    out.append(f"note         : {tickets['note']}")
    out.append("")

    out.append("(b) Residual queue wins — contention_queue_dispatch")
    out.append("-" * 66)
    waits = wins["wait_seconds"]
    out.append(f"count        : {wins['count']}  ({wins['per_hour']}/h)")
    out.append(
        f"at/below {wins['sweep_budget_seconds']:.0f}s : {wins['at_or_below_budget']} "
        f"({_pct_text(wins['share_at_or_below_budget_pct'])} of wins); "
        f"over budget {wins['over_budget']}"
    )
    out.append(
        f"wait seconds : p50 {waits['p50']} / p75 {waits['p75']} / "
        f"p90 {waits['p90']} / max {waits['max']}  (mean {waits['mean']}, "
        f"total {waits['total']})"
    )
    out.append("")

    out.append("(c) Residual queue losses — contention_queue_fallback_after_queue")
    out.append("-" * 66)
    out.append(f"count        : {losses['count']}  ({losses['per_hour']}/h)")
    out.append(
        f"aggregate wait: {losses['aggregate_wait_seconds']} s "
        f"({losses['wait_per_hour_seconds']} s/h, mean "
        f"{losses['mean_wait_seconds']} s)"
    )
    out.append(
        f"wait seconds : p50 {losses['wait_seconds']['p50']} / "
        f"p75 {losses['wait_seconds']['p75']} / "
        f"p90 {losses['wait_seconds']['p90']} / "
        f"max {losses['wait_seconds']['max']}"
    )
    groups = losses["reason_groups"]
    out.append(
        f"reason split : local_lease_active {groups['local_lease_active']} / "
        f"none {groups['none']} / other {groups['other']}"
    )
    for reason, bucket in losses["by_reason"].items():
        out.append(
            f"  {reason:<24} count {bucket['count']:<5} "
            f"wait {bucket['aggregate_wait_seconds']} s "
            f"({_pct_text(bucket['share_of_losses_pct'])} of losses)"
        )
    out.append("")

    out.append(
        f"Queue outcomes: {queue['count']} event(s) ({queue['per_hour']}/h) — "
        f"wins {_pct_text(queue['win_share_pct'])} / "
        f"losses {_pct_text(queue['loss_share_pct'])}"
    )
    out.append("")
    out.append(f"STATUS: {'COMPLETE' if status['complete'] else 'INCOMPLETE'}")
    for reason in status["reasons"]:
        out.append(f"  - {reason}")
    return "\n".join(out)


def render_markdown(meta: dict, sections: dict, status: dict) -> str:
    """Compact markdown section for a work-item comment or docs artifact."""
    window = meta["window"]
    wins = sections["residual_wins"]
    losses = sections["residual_losses"]
    queue = sections["queue_events"]
    tickets = sections["ticket_backed"]

    out: list[str] = []
    out.append("### Contention-queue sweep report (LP-0MVT110PJ003XYLP)")
    out.append("")
    out.append(
        f"- window: {window['since']} → {window['until']} ({window['hours']} h)"
    )
    out.append(f"- sweep budget: {meta['sweep_budget_seconds']} s")
    out.append(f"- log_dir: `{meta['log_dir']}`")
    out.append("")

    out.append("### (a) Ticket-backed dispatches")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---:|")
    out.append(f"| `local_admission_reserve_claimed` | {tickets['count']} |")
    out.append(f"| per hour | {tickets['per_hour']} |")
    out.append("")
    out.append(f"_{tickets['note']}._")
    out.append("")

    out.append("### (b) Residual queue wins")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---:|")
    out.append(f"| count | {wins['count']} |")
    out.append(f"| per hour | {wins['per_hour']} |")
    out.append(f"| at or below {wins['sweep_budget_seconds']:.0f} s | {wins['at_or_below_budget']} ({_pct_text(wins['share_at_or_below_budget_pct'])}) |")
    out.append(f"| over budget | {wins['over_budget']} |")
    out.append(f"| wait p50 (s) | {wins['wait_seconds']['p50']} |")
    out.append(f"| wait p75 (s) | {wins['wait_seconds']['p75']} |")
    out.append(f"| wait p90 (s) | {wins['wait_seconds']['p90']} |")
    out.append(f"| wait max (s) | {wins['wait_seconds']['max']} |")
    out.append("")

    out.append("### (c) Residual queue losses")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---:|")
    out.append(f"| count | {losses['count']} |")
    out.append(f"| per hour | {losses['per_hour']} |")
    out.append(f"| aggregate dead wait (s) | {losses['aggregate_wait_seconds']} |")
    out.append(f"| dead wait per hour (s) | {losses['wait_per_hour_seconds']} |")
    out.append(f"| mean wait (s) | {losses['mean_wait_seconds']} |")
    out.append("")
    out.append("| reason | count | aggregate wait (s) | share of losses |")
    out.append("|---|---:|---:|---:|")
    for reason, bucket in losses["by_reason"].items():
        out.append(
            f"| `{reason}` | {bucket['count']} | "
            f"{bucket['aggregate_wait_seconds']} | "
            f"{_pct_text(bucket['share_of_losses_pct'])} |"
        )
    out.append("")
    groups = losses["reason_groups"]
    out.append(
        f"Reason groups: `local_lease_active` {groups['local_lease_active']} / "
        f"`none` {groups['none']} / other {groups['other']}."
    )
    out.append("")
    out.append(
        f"Queue outcomes: {queue['count']} event(s) ({queue['per_hour']}/h) — "
        f"wins {_pct_text(queue['win_share_pct'])} / "
        f"losses {_pct_text(queue['loss_share_pct'])}."
    )
    out.append("")
    if status["complete"]:
        out.append("**Measurement:** complete.")
    else:
        out.append("**Measurement:** incomplete —")
        for reason in status["reasons"]:
            out.append(f"- {reason}")
    out.append("")
    out.append(
        f"_Generated by `scripts/contention_queue_sweep_report.py` at "
        f"{meta['generated_at']}._"
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_bound(value: str) -> dt.datetime:
    """Parse a CLI ``--since``/``--until`` timestamp (naive local time)."""
    text = value.strip()
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"invalid timestamp {value!r}; use 'YYYY-MM-DD HH:MM[:SS]' or ISO 8601"
    )


def _resolve_window(
    *,
    since: dt.datetime | None,
    until: dt.datetime | None,
    hours: float,
    now: dt.datetime,
) -> tuple[dt.datetime, dt.datetime]:
    """Resolve the window: explicit bounds win, otherwise *hours* back from *until*.

    Sub-second precision is dropped so a ``--hours`` window reads as whole
    seconds in the report (the log lines carry milliseconds, the window does
    not need them).
    """
    end = (until or now).replace(microsecond=0)
    start = since if since is not None else end - dt.timedelta(hours=hours)
    if start >= end:
        raise ValueError(
            f"window start must be before its end (start {start}, end {end})"
        )
    return start, end


def _build_meta(args, log_dir: Path, start, end, coverage) -> dict:
    return {
        "tool": "scripts/contention_queue_sweep_report.py",
        "work_item": "LP-0MVT110PJ003XYLP",
        "parent_work_item": "LP-0MVIKPRMD000Y97R",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "log_dir": str(log_dir),
        "parser": PARSER_SOURCE,
        "sweep_budget_seconds": args.sweep_budget_seconds,
        "min_hours": args.min_hours,
        "max_gap_seconds": args.max_gap_seconds,
        "window": {
            "since": start.isoformat(sep=" "),
            "until": end.isoformat(sep=" "),
            "hours": round((end - start).total_seconds() / 3600.0, 2),
        },
        "coverage": coverage,
    }


def _coverage_summary(log_dir: Path, start, end, max_gap_seconds: float) -> dict:
    measured = measure_log_coverage(
        log_dir, since=start, until=end, max_gap_seconds=max_gap_seconds
    )
    return {
        "files": len(measured.spans),
        "gaps": [
            {
                "before": gap.before.name,
                "before_last": gap.before_last.isoformat(sep=" "),
                "after": gap.after.name,
                "after_first": gap.after_first.isoformat(sep=" "),
                "gap_seconds": round(gap.gap_seconds, 1),
            }
            for gap in measured.gaps
        ],
        "covered": measured.covered,
        "data_start": measured.spans[0].first.isoformat(sep=" ") if measured.spans else None,
        "data_end": max(span.last for span in measured.spans).isoformat(sep=" ")
        if measured.spans
        else None,
    }


def _status(window_hours: float, coverage: dict, *, min_hours: float, discovered: int) -> dict:
    """Decide whether the measurement is complete, and say why when it is not."""
    reasons: list[str] = []
    if discovered == 0:
        reasons.append(
            "no proxy logs found in the log directory — nothing to measure"
        )
    elif coverage["files"] == 0:
        reasons.append(
            "no proxy log data in the requested window — the window is empty"
        )
    if window_hours < min_hours:
        reasons.append(
            f"thin window: {window_hours} h is shorter than the {min_hours} h "
            "pre-flip baseline minimum"
        )
    for gap in coverage["gaps"]:
        reasons.append(
            f"coverage gap: {gap['gap_seconds']} s between {gap['before']} "
            f"(last {gap['before_last']}) and {gap['after']} "
            f"(first {gap['after_first']}) — the window is incomplete"
        )
    return {"complete": not reasons, "reasons": reasons}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--log-dir", default=str(_DEFAULT_LOG_DIR), help="proxy log directory"
    )
    parser.add_argument(
        "--since", default=None, help="window start ('YYYY-MM-DD HH:MM[:SS]', ISO 8601)"
    )
    parser.add_argument(
        "--until",
        default=None,
        help="window end (defaults to now; inclusive)",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=DEFAULT_HOURS,
        help=(
            "window length used when --since is omitted (default "
            f"{DEFAULT_HOURS:.0f}); ignored when --since is given"
        ),
    )
    parser.add_argument(
        "--sweep-budget-seconds",
        type=float,
        default=_DEFAULT_SWEEP_BUDGET_S,
        help=(
            "budget the sweep applies, used for the win at-or-below share "
            f"(default {_DEFAULT_SWEEP_BUDGET_S:.0f})"
        ),
    )
    parser.add_argument(
        "--min-hours",
        type=float,
        default=_DEFAULT_MIN_HOURS,
        help=(
            "minimum window length for a complete measurement (default "
            f"{_DEFAULT_MIN_HOURS:.0f}, the pre-flip baseline precondition)"
        ),
    )
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=_DEFAULT_MAX_GAP_S,
        help=(
            "adjacent-file log gap that makes the window unmeasurable "
            f"(default {_DEFAULT_MAX_GAP_S:.0f})"
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--markdown", action="store_true", help="emit a markdown section")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse: invalid input, never the "incomplete" code
        return 0 if not exc.code else 1

    if args.sweep_budget_seconds <= 0:
        print(
            f"error: --sweep-budget-seconds must be positive, got "
            f"{args.sweep_budget_seconds}",
            file=sys.stderr,
        )
        return 1
    if args.min_hours < 0:
        print(f"error: --min-hours must be >= 0, got {args.min_hours}", file=sys.stderr)
        return 1
    if args.max_gap_seconds <= 0:
        print(
            f"error: --max-gap-seconds must be positive, got "
            f"{args.max_gap_seconds}",
            file=sys.stderr,
        )
        return 1

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"error: log directory not found: {log_dir}", file=sys.stderr)
        return 1

    try:
        since = _parse_bound(args.since) if args.since else None
        until = _parse_bound(args.until) if args.until else None
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        start, end = _resolve_window(
            since=since, until=until, hours=args.hours, now=dt.datetime.now()
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Single streaming pass over every discovered log file (both rotation
    # schemes, plain or .gz); per-line timestamps are the authoritative window
    # boundary, never the filename.
    events: list[QueueEvent] = []
    ticket_count = 0
    for _path, line in iter_proxy_log_lines(log_dir):
        ts = parse_line_timestamp(line)
        if ts is None or ts < start or ts > end:
            continue
        event = parse_queue_event(line, ts)
        if event is not None:
            events.append(event)
        elif is_ticket_claim(line):
            ticket_count += 1

    discovered = len(discover_proxy_log_files(log_dir))
    coverage = _coverage_summary(log_dir, start, end, args.max_gap_seconds)
    meta = _build_meta(args, log_dir, start, end, coverage)
    window_hours = meta["window"]["hours"]
    status = _status(
        window_hours,
        coverage,
        min_hours=args.min_hours,
        discovered=discovered,
    )
    sections = summarise(
        events,
        ticket_count=ticket_count,
        sweep_budget_seconds=args.sweep_budget_seconds,
        window_hours=window_hours,
    )

    if args.json:
        print(
            json.dumps(
                {"meta": meta, "sections": sections, "status": status}, indent=2
            )
        )
    elif args.markdown:
        print(render_markdown(meta, sections, status))
    else:
        print(render(meta, sections, status))
    return 0 if status["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
