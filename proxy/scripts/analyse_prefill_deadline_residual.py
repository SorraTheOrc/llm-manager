#!/usr/bin/env python3
"""Measure and attribute ``local_eviction_after_acceptance reason=prefill_deadline``.

LP-0MVRMOT0N005136O (evidence half of the AC3 closure verdict on
LP-0MVO8H7VQ003B12I). Given a proxy log window this produces the dataset the
closure verdict needs:

- the count of post-acceptance evictions **per reason** (eviction-scoped, see
  :func:`count_evictions_by_reason`), so the residual ``prefill_deadline``
  count and the Option D ``measured_cache_deficit`` conversion are reported
  side by side (AC5) — a fall in one must never be read as the residual
  disappearing;
- the **user-traffic vs audit-session split** using the same classifier as the
  parent AC4 measurement (:data:`AUDIT_SESSION_PREFIX`, mirroring
  ``.pi/skills/proxy-usage-analysis/scripts/reporting.py``), with the hourly
  rate for each class (AC3);
- the ``estimated_tokens`` / ``new_tokens`` / ``eta_prefill_s`` / ``elapsed_s``
  distributions of the ``prefill_deadline`` events, so a window can be compared
  with the 22-event/7.4 h baseline (LP-0MVO8H7VQ003B12I) and the 114-event
  fix-item window (LP-0MVQIT9QY007QU27) (AC3);
- a **per-event cause attribution** into occupancy / capacity / predicted-warmth
  with the evidence columns the rules consumed (AC4), including whether another
  tenant held the pool, whether the backend reported *any* prefill progress in
  the dispatch window, and whether the dispatch started inside an
  abandoned-prefill backlog window.

Profile of the evidence, and why it is proxy-log-only
-----------------------------------------------------

Every figure is derived from the on-disk log stream, so a proxy restart inside
the window is a **recorded re-base event**, not a reason to discard the window:
no in-process counter is consumed. (The same reasoning the 2026-10-25
post-deploy measurement recorded for its certified window.)

``llama-server.log`` is deliberately **not** used for per-event attribution: its
lines carry no wall-clock timestamp, so a per-event join is impossible. It is
used only for the optional window-level accounting cross-check in
:func:`backend_prefill_accounting`, which compares the number of completed
prefills in ``llama-server`` with the number of local streams the proxy actually
consumed — the quantitative signature of prefills abandoned in the backend.

Event grammar consumed (all from ``admission_events`` / ``measured_prefill`` /
the dispatch monitor; see ``proxy/proxy/admission_events.py``)::

    local_eviction_after_acceptance session=S reason=R estimated_tokens=N
        new_tokens=N eta_prefill_s=F elapsed_s=F slot_id=S llama_task_id=T endpoint=E
    measured_cache_abort session=S model=M reason=measured_cache_deficit total=N
        cache=N measured_new_tokens=N priced_new_tokens=F deficit_ratio=F ...
    prefill_progress_measured session=S model=M total=N cache=N processed=N
        time_ms=F measured_cached_ratio=F measured_new_tokens=N priced_new_tokens=F
    lease_extended_during_prefill session=S progress=X alive=X in_flight=X buffer=X
    status_request ... available_slots=N capacity_hint=H
        local_in_flight_requests=N local_owner_session_id=S ...
    Stream finished: reason=R tokens=A/B/C session=S provider=local ...

Usage::

    python3 proxy/scripts/analyse_prefill_deadline_residual.py \\
        --start "2026-10-29 14:08:10" --end "2026-10-30 14:08:10" --json

    # optional backend accounting cross-check for a llama-server log that
    # covers exactly the same window
    python3 proxy/scripts/analyse_prefill_deadline_residual.py \\
        --start ... --end ... --llama-log /var/log/llama-proxy/llama-server.log-2026-10-30

The JSON payload is the recorded dataset; the prose report is written from it
(``docs/dev/prefill-deadline-residual.md``) so the ACs stay reproducible after
the log files rotate away.

Analysis only: nothing here reads or changes proxy configuration, so it cannot
alter an admission decision or a deadline.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_LOG_DIR = "/var/log/llama-proxy"
DEFAULT_LOG_GLOB = "proxy.log*"

#: Eviction reasons that release an ACCEPTED dispatch before its first byte
#: (``admission_events.EVICTION_REASONS_AFTER_ACCEPTANCE``). The residual class
#: and the Option D conversion are the first two.
REASON_PREFILL_DEADLINE = "prefill_deadline"
REASON_MEASURED_CACHE_DEFICIT = "measured_cache_deficit"

#: The parent AC4 classifier (LP-0MVOAQDB4002D56G): agent audit runs use session
#: ids prefixed ``audit-``; every other session — including an unattributed one
#: — is user traffic. Mirrors ``reporting.AUDIT_SESSION_PREFIX`` so the two
#: reports stay comparable; do not "improve" it here, that would break AC2.
AUDIT_SESSION_PREFIX = "audit-"

#: Admission priced at most this fraction of the prompt as new compute ⇒ the
#: decision asserted ≈90 %+ cache reuse (a *warmth prediction*, not a
#: measurement). The 0.10 line is above every fake-warm observation in the
#: reference windows (17–182 priced tokens on ~56 k prompts ⇒ ≤0.004) and below
#: every honest cold dispatch (which prices 100 %), so it separates the classes
#: without a fitted threshold.
PRICED_WARM_RATIO_MAX = 0.10

#: A measured re-prefill this many times larger than the priced compute means
#: the warmth prediction was materially wrong (not rounding). F5's replay model
#: used the same order of magnitude (priced 131 vs measured 12 727 = 97×).
MEASURED_DEFICIT_RATIO_MIN = 4.0

#: Conservative prompt-eval rate for the *abandoned-prefill backlog* estimate
#: (tokens/s). This is F5's replay rate (LP-0MVR7IXAI002QXXD: the baseline
#: window's prompt-eval p10) and the same value LP-0MVS03KCV005BTGE used to
#: price early-abort waste, so the backlog figure is comparable with both.
ABANDONED_PREFILL_RATE_TOK_S = 43.1

#: Slack allowed between an abort and a following dispatch that is treated as
#: the same retry burst (the router retries immediately after the 503).
BACKLOG_START_SLACK_SECONDS = 5.0

#: How far back a same-prompt measurement is borrowed as cluster evidence.
CLUSTER_HORIZON_SECONDS = 900.0

LINE_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
FIELD = re.compile(r"([A-Za-z_]+)=(\S+)")

ATTRIBUTION_OCCUPANCY = "occupancy"
ATTRIBUTION_CAPACITY = "capacity"
ATTRIBUTION_PREDICTED_WARMTH = "predicted-warmth"

RULES = {
    ATTRIBUTION_OCCUPANCY: (
        "another tenant occupied the slot pool during the dispatch window: a "
        "lease_extended_during_prefill from a DIFFERENT session, or a different "
        "session's provider=local stream finishing inside the window, or a "
        "status poll reporting local_in_flight_requests>=2"
    ),
    ATTRIBUTION_PREDICTED_WARMTH: (
        "no other tenant, and admission priced <="
        f"{PRICED_WARM_RATIO_MAX:.0%} of the prompt as new compute (a warmth "
        "prediction), while the evidence contradicts that price: either the "
        "backend reported NO prefill progress at all inside the 120 s window, "
        "or the same prompt cluster measured a re-prefill >="
        f"{MEASURED_DEFICIT_RATIO_MIN:g}x the priced compute"
    ),
    ATTRIBUTION_CAPACITY: (
        "no other tenant and an honest cold price (priced ratio >"
        f"{PRICED_WARM_RATIO_MAX:.0%}): the GPU could not deliver the priced "
        "prefill inside the deadline"
    ),
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvictionEvent:
    ts: datetime
    session: str
    reason: str
    estimated_tokens: int
    new_tokens: int
    eta_prefill_s: float | None
    elapsed_s: float
    endpoint: str

    @property
    def dispatch_start(self) -> datetime:
        return self.ts - timedelta(seconds=self.elapsed_s)

    @property
    def priced_cached_ratio(self) -> float:
        if self.estimated_tokens <= 0:
            return 0.0
        return max(0.0, 1.0 - (self.new_tokens / self.estimated_tokens))


@dataclass(frozen=True)
class Measurement:
    ts: datetime
    session: str
    total: int
    cache: int
    measured_new_tokens: int
    priced_new_tokens: float | None

    @property
    def measured_cached_ratio(self) -> float:
        return (self.cache / self.total) if self.total > 0 else 0.0


@dataclass(frozen=True)
class ProgressSignature:
    """The ``progress=0`` signature the 22-event baseline was attributed by.

    The parent AC4 measurement classified its events by
    ``lease_extended_during_prefill … progress=None alive=False in_flight=True``
    repeated for the whole lease plus
    ``prefill_wait_exceeds_threshold waited=30s progress=0``. Those two lines
    are captured here so the same signature is reported against any window
    (AC2: the two reports must stay comparable).
    """

    ts: datetime
    session: str
    kind: str
    progress: str


@dataclass(frozen=True)
class StatusPoll:
    ts: datetime
    in_flight_requests: int
    available_slots: int | None
    capacity_hint: str


@dataclass
class WindowLog:
    """Everything parsed from one window, indexed for attribution."""

    evictions: list[EvictionEvent] = field(default_factory=list)
    measurements: list[Measurement] = field(default_factory=list)
    lease_extensions: list[tuple[datetime, str]] = field(default_factory=list)
    progress_signatures: list[ProgressSignature] = field(default_factory=list)
    local_completions: list[tuple[datetime, str]] = field(default_factory=list)
    status_polls: list[StatusPoll] = field(default_factory=list)
    lines_in_window: int = 0
    duplicate_lines_dropped: int = 0
    files_read: int = 0

    def measurements_for(self, session: str, start: datetime, end: datetime) -> list[Measurement]:
        return [
            m for m in self.measurements if m.session == session and start <= m.ts <= end
        ]

def _seconds(value: float) -> timedelta:
    return timedelta(seconds=value)


def parse_timestamp(line: str) -> datetime | None:
    match = LINE_TS.match(line)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(
        microsecond=int(match.group(2)) * 1000
    )


def parse_fields(line: str) -> dict[str, str]:
    return dict(FIELD.findall(line))


def _int(value: str | None) -> int:
    try:
        return max(0, int(str(value)))
    except (TypeError, ValueError):
        return 0


def _float(value: str | None) -> float | None:
    if value is None or value == "unknown":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def read_log_lines(paths, failures: list[str] | None = None) -> list[str]:
    """Read every line of *paths*, transparently decompressing ``.gz`` files.

    Unreadable files are skipped and their names appended to *failures*, so a
    caller can report a silently short window instead of publishing a count
    that quietly lost a rotation file.
    """
    lines: list[str] = []
    for path in paths:
        opener = gzip.open if str(path).endswith(".gz") else open
        try:
            with opener(path, "rt", errors="replace") as handle:  # type: ignore[call-overload]
                lines.extend(handle.read().splitlines())
        except OSError:
            if failures is not None:
                failures.append(str(path))
            continue
    return lines


def parse_window(lines, start: datetime, end: datetime) -> WindowLog:
    """Parse *lines* into a :class:`WindowLog` restricted to ``[start, end)``.

    Identical lines are de-duplicated: rotation preserves pre-restart copies of
    the same log file, so without this the same eviction can be counted twice.
    ``duplicate_lines_dropped`` reports the number removed so the figure is
    auditable rather than silent.
    """
    log = WindowLog()
    seen: set[str] = set()
    for raw in lines:
        ts = parse_timestamp(raw)
        if ts is None or not (start <= ts < end):
            continue
        log.lines_in_window += 1
        if raw in seen:
            log.duplicate_lines_dropped += 1
            continue
        seen.add(raw)
        _parse_line(log, ts, raw)
    return log


def _parse_line(log: WindowLog, ts: datetime, line: str) -> None:
    if "local_eviction_after_acceptance " in line:
        fields = parse_fields(line)
        reason = fields.get("reason")
        if not reason:
            return
        log.evictions.append(
            EvictionEvent(
                ts=ts,
                session=fields.get("session") or "unknown",
                reason=reason,
                estimated_tokens=_int(fields.get("estimated_tokens")),
                new_tokens=_int(fields.get("new_tokens")),
                eta_prefill_s=_float(fields.get("eta_prefill_s")),
                elapsed_s=_float(fields.get("elapsed_s")) or 0.0,
                endpoint=fields.get("endpoint") or "unknown",
            )
        )
        return
    if "prefill_progress_measured " in line:
        fields = parse_fields(line)
        log.measurements.append(
            Measurement(
                ts=ts,
                session=fields.get("session") or "unknown",
                total=_int(fields.get("total")),
                cache=_int(fields.get("cache")),
                measured_new_tokens=_int(fields.get("measured_new_tokens")),
                priced_new_tokens=_float(fields.get("priced_new_tokens")),
            )
        )
        return
    if "lease_extended_during_prefill " in line:
        fields = parse_fields(line)
        session = fields.get("session")
        if session:
            log.lease_extensions.append((ts, session))
            log.progress_signatures.append(
                ProgressSignature(ts, session, "lease_extended_during_prefill", fields.get("progress") or "unknown")
            )
        return
    if "prefill_wait_exceeds_threshold " in line:
        fields = parse_fields(line)
        session = fields.get("session")
        if session:
            log.progress_signatures.append(
                ProgressSignature(ts, session, "prefill_wait_exceeds_threshold", fields.get("progress") or "unknown")
            )
        return
    if "Stream finished" in line and "provider=local" in line:
        session = parse_fields(line).get("session")
        if session:
            log.local_completions.append((ts, session))
        return
    if "status_request " in line:
        fields = parse_fields(line)
        log.status_polls.append(
            StatusPoll(
                ts=ts,
                in_flight_requests=_int(fields.get("local_in_flight_requests")),
                available_slots=(
                    _int(fields.get("available_slots"))
                    if fields.get("available_slots")
                    else None
                ),
                capacity_hint=fields.get("capacity_hint") or "unknown",
            )
        )


# ---------------------------------------------------------------------------
# Counting / split / distributions
# ---------------------------------------------------------------------------


def count_evictions_by_reason(evictions) -> dict[str, int]:
    """Count *eviction-scoped* events, keyed by ``reason``.

    Counting must filter on the ``local_eviction_after_acceptance`` line itself:
    the same ``reason=<...>`` string also appears on the ``measured_cache_abort``
    and lease-release lines of one event, so a raw ``grep -o | wc -l`` on
    ``reason=`` over-counts (LP-0MVS03KCV005BTGE recorded the trap; this
    function is the eviction-scoped form).
    """
    return dict(Counter(event.reason for event in evictions))


def is_audit_session(session: str | None) -> bool:
    """Parent AC4 classifier: ``audit-`` sessions are agent audit traffic."""
    return bool(session) and str(session).startswith(AUDIT_SESSION_PREFIX)


def session_class(session: str | None) -> str:
    return "audit" if is_audit_session(session) else "user"


def _percentiles(values, points=(10, 50, 90)) -> dict[str, float | int | None]:
    numbers = sorted(float(v) for v in values if v is not None)
    if not numbers:
        return {"count": 0, **{f"p{p}": None for p in points}, "min": None, "max": None, "mean": None}
    result: dict[str, float | int | None] = {"count": len(numbers)}
    for point in points:
        result[f"p{point}"] = round(_percentile(numbers, point), 3)
    result["min"] = round(numbers[0], 3)
    result["max"] = round(numbers[-1], 3)
    result["mean"] = round(sum(numbers) / len(numbers), 3)
    return result


def _percentile(sorted_values, point: float) -> float:
    if not sorted_values:
        raise ValueError("empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * (point / 100.0)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = rank - low
    return float(sorted_values[low]) + (float(sorted_values[high]) - float(sorted_values[low])) * fraction


def distributions(events) -> dict[str, dict]:
    return {
        "estimated_tokens": _percentiles(e.estimated_tokens for e in events),
        "new_tokens": _percentiles(e.new_tokens for e in events),
        "eta_prefill_s": _percentiles(e.eta_prefill_s for e in events),
        "elapsed_s": _percentiles(e.elapsed_s for e in events),
        "priced_cached_ratio": _percentiles(e.priced_cached_ratio for e in events),
    }


def reason_split(events, hours: float) -> dict[str, dict]:
    """Per-reason total, user/audit split and hourly rate for each class."""
    by_reason: dict[str, dict] = {}
    for reason, group in _group_by(events, lambda e: e.reason).items():
        user = sum(1 for e in group if session_class(e.session) == "user")
        audit = len(group) - user
        by_reason[reason] = {
            "total": len(group),
            "user": user,
            "audit": audit,
            "user_per_hour": round(user / hours, 3) if hours else None,
            "audit_per_hour": round(audit / hours, 3) if hours else None,
            "total_per_hour": round(len(group) / hours, 3) if hours else None,
            "distinct_sessions": len({e.session for e in group}),
            "distinct_user_sessions": len({e.session for e in group if session_class(e.session) == "user"}),
        }
    return by_reason


def _group_by(items, key) -> dict:
    grouped: dict = defaultdict(list)
    for item in items:
        grouped[key(item)].append(item)
    return dict(grouped)


# ---------------------------------------------------------------------------
# Per-event attribution (AC4)
# ---------------------------------------------------------------------------


def abandoned_prefill_windows(log: WindowLog) -> list[tuple[datetime, datetime, str, int, str]]:
    """Estimate how long each released dispatch kept the backend busy.

    A released dispatch is **not** cancelled at ``llama-server``: the proxy drops
    its lease and detours, but the prompt continues to be evaluated (measured on
    2026-10-29: 1 092 completed prefills at the backend versus 807 local streams
    the proxy consumed + 232 early releases). Each release therefore occupies the
    shared GPU for ``new tokens / rate`` after the release. The duration is an
    estimate — priced from the *measured* deficit when the release logged one
    (``measured_cache_deficit``), otherwise from the admission-priced
    ``new_tokens`` — and is reported as such.

    Returns ``(start, end, reason, tokens, session)`` tuples.
    """
    windows = []
    for event in log.evictions:
        tokens = 0
        if event.reason == REASON_MEASURED_CACHE_DEFICIT:
            candidates = log.measurements_for(
                event.session, _seconds_before(event.ts, 5.0), _seconds_after(event.ts, 5.0)
            )
            if candidates:
                tokens = candidates[0].measured_new_tokens
        if tokens <= 0:
            tokens = event.new_tokens
        if tokens <= 0:
            continue
        duration = tokens / ABANDONED_PREFILL_RATE_TOK_S
        windows.append((event.ts, _seconds_after(event.ts, duration), event.reason, tokens, event.session))
    return windows


def _seconds_before(ts: datetime, seconds: float) -> datetime:
    return ts - timedelta(seconds=seconds)


def _seconds_after(ts: datetime, seconds: float) -> datetime:
    return ts + timedelta(seconds=seconds)


def occupancy_evidence(log: WindowLog, event: EvictionEvent) -> dict:
    """Other-tenant evidence for one dispatch window (the original hypothesis)."""
    start, end = event.dispatch_start, event.ts
    other_leases = sorted(
        {
            session
            for ts, session in log.lease_extensions
            if session != event.session and start <= ts <= end
        }
    )
    other_completions = sorted(
        {
            session
            for ts, session in log.local_completions
            if session != event.session and start <= ts <= end
        }
    )
    polls = [p for p in log.status_polls if start <= p.ts <= end]
    max_in_flight = max((p.in_flight_requests for p in polls), default=0)
    own_signatures = [s for s in log.progress_signatures if s.session == event.session and start <= s.ts <= end]
    zero_progress = [s for s in own_signatures if s.progress in ("None", "0", "unknown")]
    min_available = min((p.available_slots for p in polls if p.available_slots is not None), default=None)
    return {
        "other_session_leases": other_leases,
        "other_session_local_completions": other_completions,
        "in_flight_requests_max": max_in_flight,
        "available_slots_min": min_available,
        "status_polls": len(polls),
        "tenant_present": bool(other_leases or other_completions or max_in_flight >= 2),
        "own_lease_extensions": sum(
            1 for s in own_signatures if s.kind == "lease_extended_during_prefill"
        ),
        "own_zero_progress_signatures": len(zero_progress),
        "own_prefill_wait_threshold_events": sum(
            1 for s in own_signatures if s.kind == "prefill_wait_exceeds_threshold"
        ),
    }


def cluster_measurement(log: WindowLog, event: EvictionEvent) -> Measurement | None:
    """The nearest same-session measurement whose prompt size matches *event*.

    A retry burst re-sends the same prompt, so a measured dispatch of the same
    ``total`` (within 1 %) inside the burst measures the warmth of the whole
    cluster even when the evicted dispatch itself reported no progress. The
    burst horizon is :data:`CLUSTER_HORIZON_SECONDS` of wall time — wide enough
    for the observed 40-minute retry bursts, narrow enough that an unrelated
    dispatch of a similarly sized prompt is not borrowed.
    """
    candidates = [
        m
        for m in log.measurements
        if m.session == event.session
        and _seconds_before(event.dispatch_start, CLUSTER_HORIZON_SECONDS) <= m.ts <= _seconds_after(event.ts, 30.0)
        and event.estimated_tokens > 0
        and abs(m.total - event.estimated_tokens) <= max(64, 0.01 * event.estimated_tokens)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda m: abs((m.ts - event.ts).total_seconds()))


def classify_event(log: WindowLog, event: EvictionEvent, occupancy: dict) -> tuple[str, dict]:
    """Classify one event into occupancy / capacity / predicted-warmth (AC4)."""
    in_window = log.measurements_for(event.session, event.dispatch_start, event.ts)
    cluster = cluster_measurement(log, event)
    priced_ratio = (event.new_tokens / event.estimated_tokens) if event.estimated_tokens else 1.0
    measured_deficit_ratio = None
    if cluster is not None and cluster.priced_new_tokens:
        measured_deficit_ratio = cluster.measured_new_tokens / cluster.priced_new_tokens

    evidence = {
        "priced_ratio": round(priced_ratio, 5),
        "priced_cached_ratio": round(event.priced_cached_ratio, 5),
        "backend_progress_observed": bool(in_window),
        "backend_progress_events": len(in_window),
        "cluster_measured_new_tokens": cluster.measured_new_tokens if cluster else None,
        "cluster_priced_new_tokens": cluster.priced_new_tokens if cluster else None,
        "cluster_measured_deficit_ratio": (
            round(measured_deficit_ratio, 3) if measured_deficit_ratio is not None else None
        ),
        "cluster_measured_cached_ratio": (
            round(cluster.measured_cached_ratio, 5) if cluster else None
        ),
    }

    if occupancy["tenant_present"]:
        return ATTRIBUTION_OCCUPANCY, evidence
    warmth_predicted = priced_ratio <= PRICED_WARM_RATIO_MAX
    contradicted = (
        not evidence["backend_progress_observed"]
        or (
            measured_deficit_ratio is not None
            and measured_deficit_ratio >= MEASURED_DEFICIT_RATIO_MIN
        )
    )
    if warmth_predicted and contradicted:
        return ATTRIBUTION_PREDICTED_WARMTH, evidence
    return ATTRIBUTION_CAPACITY, evidence


def backlog_evidence(orphan_windows, event: EvictionEvent) -> dict:
    """Whether the dispatch started inside an abandoned-prefill backlog window."""
    start = event.dispatch_start
    covering = [
        (end, reason, tokens, session)
        for begin, end, reason, tokens, session in orphan_windows
        if begin <= _seconds_after(start, BACKLOG_START_SLACK_SECONDS) and end >= start
    ]
    if not covering:
        return {"abandoned_prefill_overlap": False, "abandoned_prefill_remaining_s": None}
    end, reason, tokens, _session = max(covering, key=lambda item: item[0])
    return {
        "abandoned_prefill_overlap": True,
        "abandoned_prefill_remaining_s": round((end - start).total_seconds(), 1),
        "abandoned_prefill_reason": reason,
        "abandoned_prefill_tokens": tokens,
    }


def attribute_events(log: WindowLog) -> list[dict]:
    """One row per ``prefill_deadline`` event with its evidence and verdict."""
    orphans = abandoned_prefill_windows(log)
    rows = []
    for event in [e for e in log.evictions if e.reason == REASON_PREFILL_DEADLINE]:
        occupancy = occupancy_evidence(log, event)
        verdict, evidence = classify_event(log, event, occupancy)
        rows.append(
            {
                "ts": event.ts.isoformat(sep=" ", timespec="seconds"),
                "session": event.session,
                "session_class": session_class(event.session),
                "estimated_tokens": event.estimated_tokens,
                "new_tokens": event.new_tokens,
                "eta_prefill_s": event.eta_prefill_s,
                "elapsed_s": event.elapsed_s,
                "attribution": verdict,
                "evidence": {
                    **evidence,
                    **{
                        "other_session_leases": occupancy["other_session_leases"],
                        "other_session_local_completions": occupancy[
                            "other_session_local_completions"
                        ],
                        "in_flight_requests_max": occupancy["in_flight_requests_max"],
                        "available_slots_min": occupancy["available_slots_min"],
                        "own_lease_extensions": occupancy["own_lease_extensions"],
                        "own_zero_progress_signatures": occupancy[
                            "own_zero_progress_signatures"
                        ],
                        "own_prefill_wait_threshold_events": occupancy[
                            "own_prefill_wait_threshold_events"
                        ],
                    },
                    **backlog_evidence(orphans, event),
                },
            }
        )
    return rows


def attribution_summary(rows, capture_available: bool = True) -> dict:
    counts = Counter(row["attribution"] for row in rows)
    by_class: dict[str, dict[str, int]] = {}
    for row in rows:
        by_class.setdefault(row["session_class"], Counter())[row["attribution"]] += 1
    return {
        "totals": {name: counts.get(name, 0) for name in RULES},
        "by_session_class": {name: dict(counter) for name, counter in by_class.items()},
        "rules": RULES,
        "capture_available": capture_available,
        "capture_caveat": (
            None
            if capture_available
            else (
                "no prefill_progress_measured line exists in this window, so the "
                "engine did not report measured progress for ANY dispatch. The "
                "predicted-warmth rule then rests on the admission price alone "
                "(priced ratio) and the zero-progress signature lines; read the "
                "deficit column as unmeasured."
            )
        ),
        "abandoned_prefill_overlap": {
            "events": sum(1 for row in rows if row["evidence"]["abandoned_prefill_overlap"]),
            "total": len(rows),
        },
    }


# ---------------------------------------------------------------------------
# Optional backend accounting cross-check
# ---------------------------------------------------------------------------


def backend_prefill_accounting(log: WindowLog, llama_log_path: str) -> dict:
    """Compare backend-completed prefills with proxy-consumed local streams.

    ``llama-server.log`` has no wall-clock timestamps, so this is valid only
    when the file covers exactly the analysed window (the day file of a
    logrotate-daily setup does). ``abandoned_prefills`` is then the number of
    prefills the GPU finished that no proxy stream consumed — the quantitative
    signature of an abandoned-prefill backlog.
    """
    path = Path(llama_log_path)
    if not path.exists():
        return {"llama_log": str(path), "error": "not found"}
    completed = sum(1 for line in read_log_lines([path]) if "prompt eval time" in line)
    consumed = len(log.local_completions)
    released = len(log.evictions)
    return {
        "llama_log": str(path),
        "backend_completed_prefills": completed,
        "proxy_consumed_local_streams": consumed,
        "proxy_released_dispatches": released,
        "abandoned_prefills": completed - consumed,
        "note": (
            "Valid only when the llama-server log covers exactly the analysed "
            "window; llama-server lines carry no wall-clock timestamp."
        ),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(
    log: WindowLog,
    start: datetime,
    end: datetime,
    *,
    log_dir: str | None = None,
    llama_log: str | None = None,
) -> dict:
    hours = max((end - start).total_seconds() / 3600.0, 0.0)
    rows = attribute_events(log)
    by_reason = count_evictions_by_reason(log.evictions)
    deadline_events = [e for e in log.evictions if e.reason == REASON_PREFILL_DEADLINE]
    report = {
        "window": {
            "start": start.isoformat(sep=" ", timespec="seconds"),
            "end": end.isoformat(sep=" ", timespec="seconds"),
            "hours": round(hours, 3),
        },
        "source": {
            "log_dir": log_dir,
            "files_read": log.files_read,
            "lines_in_window": log.lines_in_window,
            "duplicate_lines_dropped": log.duplicate_lines_dropped,
            "restarts_in_window": _restart_events(log_dir, start, end) if log_dir else [],
        },
        "evictions": {
            "total": len(log.evictions),
            "by_reason": by_reason,
            "per_hour": round(len(log.evictions) / hours, 3) if hours else None,
        },
        "reason_split": reason_split(log.evictions, hours),
        "distributions": {
            "prefill_deadline": distributions(deadline_events),
        },
        "attribution": attribution_summary(rows, capture_available=bool(log.measurements)),
        "events": rows,
    }
    if llama_log:
        report["backend_prefill_accounting"] = backend_prefill_accounting(log, llama_log)
    return report


def _restart_events(log_dir: str, start: datetime, end: datetime) -> list[dict]:
    """Proxy restarts inside the window (recorded as re-base events, AC1)."""
    lines = read_log_lines(sorted(Path(log_dir).glob(DEFAULT_LOG_GLOB)))
    restarts = []
    for line in lines:
        if "Rotated previous proxy log aside" not in line:
            continue
        ts = parse_timestamp(line)
        if ts is None or not (start <= ts < end):
            continue
        restarts.append(
            {
                "ts": ts.isoformat(sep=" ", timespec="seconds"),
                "line": line.split(" - INFO - ", 1)[-1].strip(),
            }
        )
    return restarts


def _parse_window_arg(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyse_prefill_deadline_residual.py",
        description=(
            "Count, split and attribute local_eviction_after_acceptance "
            "reason=prefill_deadline events in a proxy log window."
        ),
    )
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help=f"proxy log directory (default {DEFAULT_LOG_DIR})")
    parser.add_argument("--log-glob", default=DEFAULT_LOG_GLOB, help=f"file glob inside --log-dir (default {DEFAULT_LOG_GLOB})")
    parser.add_argument("--start", required=True, help="window start, 'YYYY-MM-DD HH:MM:SS' (UTC)")
    parser.add_argument("--end", required=True, help="window end, 'YYYY-MM-DD HH:MM:SS' (UTC, exclusive)")
    parser.add_argument("--llama-log", help="llama-server log covering exactly the window (optional cross-check)")
    parser.add_argument("--output", help="write the JSON report here instead of stdout")
    parser.add_argument("--json", action="store_true", help="emit the JSON dataset (default: a short text summary)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start = _parse_window_arg(args.start)
    end = _parse_window_arg(args.end)
    if start >= end:
        print("error: --start must be before --end", file=sys.stderr)
        return 2
    paths = sorted(Path(args.log_dir).glob(args.log_glob))
    if not paths:
        print(f"error: no log files matched {args.log_glob} in {args.log_dir}", file=sys.stderr)
        return 1
    failures: list[str] = []
    log = parse_window(read_log_lines(paths, failures), start, end)
    log.files_read = len(paths) - len(failures)
    for failed in failures:
        print(f"warning: unreadable log file skipped: {failed}", file=sys.stderr)
    report = build_report(log, start, end, log_dir=args.log_dir, llama_log=args.llama_log)
    if failures:
        report["source"]["unreadable_files"] = failures
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).expanduser().write_text(payload + "\n")
    if args.json:
        print(payload)
        return 0
    print(_text_summary(report))
    if args.output:
        print(f"\nJSON dataset written to {Path(args.output).expanduser()}")
    return 0


def _text_summary(report: dict) -> str:
    window = report["window"]
    reasons = report["evictions"]["by_reason"]
    lines = [
        f"Window {window['start']} → {window['end']} ({window['hours']} h)",
        f"Evictions after acceptance: {report['evictions']['total']} "
        + ", ".join(f"{name}={count}" for name, count in sorted(reasons.items())),
    ]
    split = report["reason_split"].get(REASON_PREFILL_DEADLINE)
    if split:
        lines.append(
            f"prefill_deadline: {split['total']} "
            f"(user {split['user']} / audit {split['audit']}) — "
            f"user {split['user_per_hour']}/h, audit {split['audit_per_hour']}/h"
        )
    attribution = report["attribution"]["totals"]
    lines.append(
        "Attribution: " + ", ".join(f"{name}={count}" for name, count in sorted(attribution.items()))
    )
    if not report["attribution"]["capture_available"]:
        lines.append("(no measured-prefill capture in this window: attributions rest on price alone)")
    restarts = report["source"].get("restarts_in_window") or []
    if restarts:
        lines.append(f"Re-base events (restarts) in window: {len(restarts)}")
        for restart in restarts:
            lines.append(f"  {restart['ts']}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
