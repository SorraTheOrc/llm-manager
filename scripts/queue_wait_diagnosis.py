#!/usr/bin/env python3
"""Per-wait diagnosis of abandoned contention-queue waits.

Diagnosis harness for **LP-0MVV29XPT003OXI5** (parent `LP-0MVV1V213002QAIR`).
It answers, per queue wait rather than per session, *why* a wait the admission
guard admitted still expired at the ``contention_queue_max_wait_seconds``
budget:

- the guard's own projection for the **same request** (``projected_wait_s`` /
  ``eta_occupancy_s`` / ``busy_slots`` / ``rate_bucket``), against the wait
  actually observed;
- the class of **blocker** that held the slot when the wait enqueued —
  ``prefill_in_flight`` (the holder was still prefilling) or
  ``lease_holder_generating`` (the holder had already served its first byte and
  was decoding, so no prefill service was owed) or ``unknown``.

This is a *pure log analyser*: it never changes proxy behaviour and never
requires the proxy to be running. Rotation, gzip handling and line discovery are
delegated to :mod:`scripts.lib.proxy_logs`; the admission/queue line vocabulary
(``local_admission_guard``, ``contention_queue_wait``,
``contention_queue_fallback_after_queue``, ``contention_queue_dispatch``) is
**reused** from :mod:`scripts.prefill_admission_conversion` — this module adds
no third parser for those lines, only the two lease markers the blocker
classification needs (``lease_extended_during_prefill``, ``lease_released``).

## Own-request correlation

A wait's guard projection is only meaningful if it belongs to the *same
request*. Correlation is resolved in this order and the choice is recorded on
every row in ``correlation``:

1. ``own_request`` — the wait's own ``new_tokens`` (the outcome line's, else the
   enqueue line's) equals the ``new_tokens`` of a preceding, not-yet-consumed
   admission for the same session. This is positive own-request evidence and
   ``ambiguous`` is ``false``.
2. ``session_preceding`` — the log shape carries no own-request pricing (the
   2026-10-29/30 window predates ``LP-0MVV29WR0009DY1A``), so the nearest
   preceding, not-yet-consumed admission for the session is used. The join is
   *never silent*: ``ambiguous`` is ``true`` and ``candidates`` reports how many
   outstanding admissions it could have been (``1`` means the join was in fact
   forced by the event order).
3. ``mismatched`` — own-request pricing exists and a preceding admission exists,
   but none matches it, so the admission belongs to a **different request** in
   the same session. The wait is **not** joined, its guard fields stay ``null``
   and ``ambiguous`` is ``true`` (LP-0MVV29XPT003OXI5 AC2b).
4. ``none`` — no preceding admission for the session at all. Guard fields
   ``null``, ``ambiguous`` ``false`` (AC2c).

## Blocker class

Reconstructed from lease evidence in a bounded lookback before the enqueue
instant, newest evidence per session wins:

- ``dispatch_first_byte_ms`` → the session is past its prefill and is
  **generating**;
- ``lease_extended_during_prefill`` → the session is still **prefilling**;
- ``lease_released`` → the session no longer holds the slot (state cleared).

The classification is deliberately evidence-only: a session with no lease
marker inside the lookback is ``unknown``, never guessed. ``blocker_is_self``
flags the case where the only candidate holder is the waiting session itself
(the ``local_lease_active`` re-queue path), so a self-blocked wait is visible
rather than folded into the population.

## New-token band

``band``/``band_evidence`` re-derive the per-wait band with the
``LP-0MVV29WR0009DY1A`` own-request preference, so the record can prove per
wait where a previously phantom band came from (a *later* admission is never
borrowed — the defect that put ``<8k`` waits into ``32k-64k``).

Usage::

    scripts/queue_wait_diagnosis.py \
        --since "2026-10-29 00:10:18" --until "2026-10-30 00:10:18" --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

try:  # script executed directly (``python3 scripts/...``) vs imported as a module
    from lib.proxy_logs import (
        discover_proxy_log_files,
        iter_proxy_log_lines,
        measure_log_coverage,
        parse_line_timestamp,
    )
    from prefill_admission_conversion import (
        _KV_REASON,
        _KV_SESSION,
        Event,
        _first,
        _message_starts_with,
        band_for,
        parse_event,
        parse_timestamp,
    )
except ImportError:  # pragma: no cover - import path for ``scripts`` on sys.path
    from scripts.lib.proxy_logs import (  # type: ignore[no-redef]
        discover_proxy_log_files,
        iter_proxy_log_lines,
        measure_log_coverage,
        parse_line_timestamp,
    )
    from scripts.prefill_admission_conversion import (  # type: ignore[no-redef]
        _KV_REASON,
        _KV_SESSION,
        Event,
        _first,
        _message_starts_with,
        band_for,
        parse_event,
        parse_timestamp,
    )

DEFAULT_LOG_DIR = Path("/var/log/llama-proxy")

#: Marker on the line that reports a still-prefilling lease holder.
LEASE_PREFILL_MARKER = "lease_extended_during_prefill"
#: Marker on the line that reports a released lease.
LEASE_RELEASE_MARKER = "lease_released"

#: Blocker classes (AC1). ``unknown`` is reported, never guessed.
BLOCKER_PREFILL = "prefill_in_flight"
BLOCKER_GENERATING = "lease_holder_generating"
BLOCKER_UNKNOWN = "unknown"
BLOCKER_CLASSES: tuple[str, ...] = (
    BLOCKER_PREFILL,
    BLOCKER_GENERATING,
    BLOCKER_UNKNOWN,
)

#: Correlation outcomes (AC1/AC2). Only ``own_request`` is positive same-request
#: evidence; ``session_preceding`` and ``mismatched`` are flagged ambiguous.
CORRELATION_OWN = "own_request"
CORRELATION_SESSION = "session_preceding"
CORRELATION_MISMATCH = "mismatched"
CORRELATION_NONE = "none"
CORRELATIONS: tuple[str, ...] = (
    CORRELATION_OWN,
    CORRELATION_SESSION,
    CORRELATION_MISMATCH,
    CORRELATION_NONE,
)

#: How far before an enqueue a lease marker may sit and still describe the
#: holder (seconds). Defaults to ``local_prefill_deadline_seconds`` (120): a
#: holder silent for longer than that is no longer evidenced as holding.
DEFAULT_BLOCKER_LOOKBACK_S = 120.0

#: How close an enqueue line must be to ``outcome_ts - queued_duration`` to be
#: paired with that outcome (seconds). Measured pairs agree to a few ms; the
#: tolerance only absorbs the queue's polling granularity.
PAIR_TOLERANCE_S = 2.0

#: Projected-wait distribution edges for the abandoned-wait summary (seconds).
PROJECTED_BUCKETS: tuple[tuple[str, float, float | None], ...] = (
    ("<20", 0.0, 20.0),
    ("20-30", 20.0, 30.0),
    ("30-40", 30.0, 40.0),
    ("40-50", 40.0, 50.0),
    ("50-60", 50.0, 60.0),
    (">=60", 60.0, None),
)


def projected_bucket(seconds: float | None) -> str:
    """Return the distribution bucket label for a projected wait, or ``unknown``."""
    if seconds is None:
        return "unknown"
    for label, lower, upper in PROJECTED_BUCKETS:
        if seconds >= lower and (upper is None or seconds < upper):
            return label
    return "unknown"


def _projected_bucket_order() -> list[str]:
    return [label for label, _, _ in PROJECTED_BUCKETS] + ["unknown"]


@dataclass
class LeaseEvent:
    """One lease-lifecycle marker that describes who holds the slot."""

    #: ``prefill`` | ``generating`` | ``released``
    kind: str
    ts: datetime
    session: str | None
    reason: str | None = None


def parse_lease_event(line: str, ts: datetime) -> LeaseEvent | None:
    """Parse a lease marker into a :class:`LeaseEvent`, or ``None``.

    Anchored to the start of the logging message body (the harness's
    ``_message_starts_with``) so a proxied request body that merely quotes a
    marker can never be read as lease evidence.
    """
    if _message_starts_with(line, LEASE_RELEASE_MARKER):
        return LeaseEvent(
            kind="released",
            ts=ts,
            session=_first(_KV_SESSION, line),
            reason=_first(_KV_REASON, line),
        )
    if _message_starts_with(line, LEASE_PREFILL_MARKER):
        return LeaseEvent(
            kind="prefill", ts=ts, session=_first(_KV_SESSION, line)
        )
    return None


@dataclass
class QueueWait:
    """One queue wait: its enqueue evidence and the outcome that concluded it."""

    enqueue: Event | None
    outcome: Event | None

    @property
    def session(self) -> str | None:
        if self.outcome is not None and self.outcome.session is not None:
            return self.outcome.session
        return self.enqueue.session if self.enqueue is not None else None

    @property
    def enqueue_ts(self) -> datetime | None:
        if self.enqueue is not None:
            return self.enqueue.ts
        if self.outcome is not None and self.outcome.queued_seconds is not None:
            # The enqueue line is missing (or unpaired): the outcome's own
            # ``queued_duration`` dates the enqueue exactly.
            return self.outcome.ts - _seconds(self.outcome.queued_seconds)
        return None

    @property
    def observed_wait_s(self) -> float | None:
        return self.outcome.queued_seconds if self.outcome is not None else None

    @property
    def outcome_kind(self) -> str | None:
        if self.outcome is None:
            return None
        return "abandoned" if self.outcome.kind == "queue_fallback" else "dispatched"

    @property
    def trigger(self) -> str:
        for event in (self.outcome, self.enqueue):
            if event is not None and event.trigger:
                return event.trigger
        return "unknown"

    @property
    def cause(self) -> str | None:
        return self.outcome.cause if self.outcome is not None else None

    @property
    def own_new_tokens(self) -> int | None:
        """Own-request pricing, from the outcome line then the enqueue line."""
        for event in (self.outcome, self.enqueue):
            if event is not None and event.new_tokens is not None:
                return event.new_tokens
        return None


def _seconds(value: float) -> timedelta:
    return timedelta(seconds=float(value))


@dataclass
class Row:
    """One emitted diagnosis row (AC1)."""

    session: str | None
    enqueue_ts: datetime | None
    outcome_ts: datetime | None
    outcome: str | None
    trigger: str
    cause: str | None
    observed_wait_s: float | None
    own_new_tokens: int | None
    correlation: str
    ambiguous: bool
    candidates: int
    guard: Event | None
    band: str
    band_evidence: str
    #: The band the **pre-fix** attribution rule would have assigned: the
    #: session's latest admission at or before the wait, else the session's
    #: latest admission *even when it is later*. Reproduced here so the record
    #: can prove per wait whether a phantom band was an attribution artefact
    #: (``legacy_band`` differs from ``band``) or genuine (AC3c).
    legacy_band: str
    blocker_class: str
    blocker_session: str | None
    blocker_is_self: bool

    def to_dict(self) -> dict:
        guard = self.guard
        return {
            "session": self.session,
            "enqueue_ts": _iso(self.enqueue_ts),
            "outcome_ts": _iso(self.outcome_ts),
            "outcome": self.outcome,
            "trigger": self.trigger,
            "cause": self.cause,
            "observed_wait_s": _round(self.observed_wait_s),
            "own_new_tokens": self.own_new_tokens,
            "correlation": self.correlation,
            "ambiguous": self.ambiguous,
            "candidates": self.candidates,
            "guard": (
                None
                if guard is None
                else {
                    "decision": guard.decision,
                    "reason": guard.reason,
                    "projected_wait_s": guard.projected_wait_s,
                    "eta_prefill_s": guard.eta_prefill_s,
                    "eta_occupancy_s": guard.eta_occupancy_s,
                    "busy_slots": guard.busy_slots,
                    "in_flight_prefills": guard.in_flight_prefills,
                    "rate_bucket": guard.rate_bucket,
                    "new_tokens": guard.new_tokens,
                    "budget_s": guard.budget_s,
                }
            ),
            "band": self.band,
            "band_evidence": self.band_evidence,
            "legacy_band": self.legacy_band,
            "blocker_class": self.blocker_class,
            "blocker_session": self.blocker_session,
            "blocker_is_self": self.blocker_is_self,
        }


def _iso(value: datetime | None) -> str | None:
    return value.isoformat(sep=" ") if value is not None else None


def _round(value: float | None) -> float | None:
    return round(float(value), 2) if value is not None else None


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


@dataclass
class Collected:
    """Everything one scan produced, split by the role each line plays."""

    admissions: list[Event] = field(default_factory=list)
    enqueues: list[Event] = field(default_factory=list)
    outcomes: list[Event] = field(default_factory=list)
    leases: list[LeaseEvent] = field(default_factory=list)
    files: set[str] = field(default_factory=set)


def collect(log_dir: Path, since: datetime, until: datetime) -> Collected:
    """Scan *log_dir* for in-window events relevant to the diagnosis."""
    collected = Collected()
    for path, line in iter_proxy_log_lines(log_dir):
        ts = parse_line_timestamp(line)
        if ts is None or ts < since or ts > until:
            continue
        lease = parse_lease_event(line, ts)
        if lease is not None:
            collected.leases.append(lease)
            collected.files.add(path.name)
            continue
        event = parse_event(line, ts)
        if event is None:
            continue
        if event.kind == "admission":
            collected.admissions.append(event)
        elif event.kind == "queue_wait":
            collected.enqueues.append(event)
        elif event.kind in ("queue_dispatch", "queue_fallback"):
            collected.outcomes.append(event)
        elif event.kind == "first_byte":
            # A served first byte is the transition out of prefill: from this
            # instant the session holds the slot while *generating*.
            collected.leases.append(
                LeaseEvent(kind="generating", ts=ts, session=event.session)
            )
        else:
            continue
        collected.files.add(path.name)
    # Files are read in name order, which is not chronological across the two
    # rotation schemes, so replay the lease evidence in time order.
    collected.leases.sort(key=lambda lease: lease.ts)
    return collected


def pair_waits(
    enqueues: list[Event], outcomes: list[Event]
) -> list[QueueWait]:
    """Pair each queue *outcome* with its enqueue line, chronologically.

    The post-change log shape (2026-10-29/30) logs no session on the enqueue
    line, so the pair is resolved by time: the outcome's own ``queued_duration``
    dates the enqueue, and an unconsumed enqueue line within
    :data:`PAIR_TOLERANCE_S` of that instant is its enqueue. A session match is
    preferred when both lines carry one. Enqueues with no outcome in the window
    are returned as their own rows (``outcome`` ``None``) rather than dropped.
    """
    remaining = list(enqueues)
    waits: list[QueueWait] = []
    for outcome in sorted(outcomes, key=lambda event: event.ts):
        target = None
        if outcome.queued_seconds is not None:
            target = outcome.ts - _seconds(outcome.queued_seconds)
        best: tuple[tuple[int, float], Event] | None = None
        if target is not None:
            for candidate in remaining:
                if (
                    candidate.session is not None
                    and outcome.session is not None
                    and candidate.session != outcome.session
                ):
                    continue
                delta = abs((candidate.ts - target).total_seconds())
                if delta > PAIR_TOLERANCE_S:
                    continue
                same_session = (
                    candidate.session is not None
                    and candidate.session == outcome.session
                )
                score = (0 if same_session else 1, delta)
                if best is None or score < best[0]:
                    best = (score, candidate)
        enqueue = best[1] if best is not None else None
        if enqueue is not None:
            remaining.remove(enqueue)
        waits.append(QueueWait(enqueue=enqueue, outcome=outcome))
    for enqueue in remaining:
        waits.append(QueueWait(enqueue=enqueue, outcome=None))
    waits.sort(
        key=lambda wait: (
            wait.enqueue_ts is None,
            wait.enqueue_ts or datetime.min,
        )
    )
    return waits


def blocker_at(
    ts: datetime | None,
    leases: list[LeaseEvent],
    *,
    lookback_s: float = DEFAULT_BLOCKER_LOOKBACK_S,
) -> tuple[str, str | None]:
    """Return ``(blocker_class, blocker_session)`` for an enqueue at *ts*.

    Lease evidence is replayed in time order and the newest state per session
    wins: a ``dispatch_first_byte_ms`` means that session has left prefill and
    is generating, a ``lease_extended_during_prefill`` means it is still
    prefilling, and a ``lease_released`` clears it. Only evidence inside
    ``lookback_s`` of *ts* counts, so a session whose last marker is older than
    the bound is not treated as the current holder.
    """
    if ts is None:
        return BLOCKER_UNKNOWN, None
    states: dict[str, tuple[datetime, str]] = {}
    for lease in leases:
        if lease.ts > ts:
            break
        if lease.session is None:
            continue
        if lease.kind == "released":
            states.pop(lease.session, None)
            continue
        states[lease.session] = (
            lease.ts,
            BLOCKER_GENERATING if lease.kind == "generating" else BLOCKER_PREFILL,
        )
    newest: tuple[datetime, str, str] | None = None
    for session, (state_ts, state) in states.items():
        if (ts - state_ts).total_seconds() > lookback_s:
            continue
        if newest is None or state_ts > newest[0]:
            newest = (state_ts, state, session)
    if newest is None:
        return BLOCKER_UNKNOWN, None
    return newest[1], newest[2]


def _admissions_by_session(admissions: list[Event]) -> dict[str, list[Event]]:
    grouped: dict[str, list[Event]] = {}
    for admission in sorted(admissions, key=lambda event: event.ts):
        if admission.session is None:
            continue
        grouped.setdefault(admission.session, []).append(admission)
    return grouped


def _legacy_band(admissions, at: datetime | None) -> str:
    """Band a wait with the **pre-fix** attribution rule (``LP-0MVV29WR0009DY1A``).

    The defect the phantom ``32k-64k`` row came from: band by the session's
    latest admission at or before the wait, and when none precedes it, by the
    session's latest admission anyway — even one hours *after* the queue event.
    Reproduced only to prove per wait which rows moved.
    """
    ordered = sorted(admissions, key=lambda event: event.ts)
    if not ordered:
        return band_for(None)
    preceding = [
        event for event in ordered if at is not None and event.ts <= at
    ]
    chosen = (preceding or ordered)[-1]
    return band_for(chosen.new_tokens)


def build_rows(
    collected: Collected,
    *,
    blocker_lookback_s: float = DEFAULT_BLOCKER_LOOKBACK_S,
) -> list[Row]:
    """Resolve correlation, band and blocker class for every queue wait."""
    waits = pair_waits(collected.enqueues, collected.outcomes)
    by_session = _admissions_by_session(collected.admissions)
    consumed: dict[str, set[int]] = {}
    rows: list[Row] = []
    for wait in waits:
        session = wait.session
        correlation = CORRELATION_NONE
        ambiguous = False
        candidates = 0
        guard: Event | None = None
        enqueue_ts = wait.enqueue_ts
        if session is not None and enqueue_ts is not None:
            used = consumed.setdefault(session, set())
            preceding = [
                admission
                for admission in by_session.get(session, ())
                if admission.ts <= enqueue_ts and id(admission) not in used
            ]
            candidates = len(preceding)
            own = wait.own_new_tokens
            if preceding:
                if own is not None:
                    matching = [
                        admission
                        for admission in preceding
                        if admission.new_tokens == own
                    ]
                    if matching:
                        guard = matching[-1]
                        correlation = CORRELATION_OWN
                        used.add(id(guard))
                    else:
                        # The only admissions belong to other requests in this
                        # session: never join them (AC2b).
                        correlation = CORRELATION_MISMATCH
                        ambiguous = True
                else:
                    # No own-request pricing in this log shape: the documented
                    # fallback join, always flagged (AC1).
                    guard = preceding[-1]
                    correlation = CORRELATION_SESSION
                    ambiguous = True
                    used.add(id(guard))
        # Banding mirrors LP-0MVV29WR0009DY1A: own outcome pricing, else own
        # enqueue pricing, else the correlated admission, else unknown. A later
        # admission is never borrowed.
        if wait.outcome is not None and wait.outcome.new_tokens is not None:
            band_tokens = wait.outcome.new_tokens
            band_evidence = "own_outcome"
        elif wait.enqueue is not None and wait.enqueue.new_tokens is not None:
            band_tokens = wait.enqueue.new_tokens
            band_evidence = "own_enqueue"
        elif guard is not None and guard.new_tokens is not None:
            band_tokens = guard.new_tokens
            band_evidence = "session_chain"
        else:
            band_tokens = None
            band_evidence = "unknown"
        blocker_class, blocker_session = blocker_at(
            enqueue_ts, collected.leases, lookback_s=blocker_lookback_s
        )
        legacy_band = _legacy_band(
            by_session.get(session, ()) if session is not None else (),
            wait.outcome.ts if wait.outcome is not None else enqueue_ts,
        )
        rows.append(
            Row(
                session=session,
                enqueue_ts=enqueue_ts,
                outcome_ts=wait.outcome.ts if wait.outcome is not None else None,
                outcome=wait.outcome_kind,
                trigger=wait.trigger,
                cause=wait.cause,
                observed_wait_s=wait.observed_wait_s,
                own_new_tokens=wait.own_new_tokens,
                correlation=correlation,
                ambiguous=ambiguous,
                candidates=candidates,
                guard=guard,
                band=band_for(band_tokens),
                band_evidence=band_evidence,
                legacy_band=legacy_band,
                blocker_class=blocker_class,
                blocker_session=blocker_session,
                blocker_is_self=(
                    blocker_session is not None and blocker_session == session
                ),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _count_all(values, domain: tuple[str, ...]) -> dict[str, int]:
    """Count *values* over a fixed *domain*, keeping zero-count classes.

    A class with no rows is reported as ``0`` rather than omitted, so a reader
    (or a record) never has to infer "absent" from "zero".
    """
    counts = {value: 0 for value in domain}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def _guard_projection_summary(rows: list[Row]) -> dict:
    """Projection-vs-observed summary for one outcome class.

    ``below_budget`` is the population the parent's hypothesis is about: the
    guard *admitted* the wait with a projection inside the queue budget, and the
    wait still expired at that budget.
    """
    projected = [
        row.guard.projected_wait_s
        for row in rows
        if row.guard is not None and row.guard.projected_wait_s is not None
    ]
    observed = [
        row.observed_wait_s
        for row in rows
        if row.observed_wait_s is not None
    ]
    below_budget = [
        row
        for row in rows
        if row.guard is not None
        and row.guard.projected_wait_s is not None
        and row.guard.budget_s is not None
        and row.guard.projected_wait_s < row.guard.budget_s
    ]
    gaps = [
        round(row.observed_wait_s - row.guard.projected_wait_s, 2)
        for row in rows
        if row.guard is not None
        and row.guard.projected_wait_s is not None
        and row.observed_wait_s is not None
    ]
    return {
        "rows": len(rows),
        "with_guard": len(projected),
        "admits": sum(1 for row in rows if row.guard is not None and row.guard.decision == "admit"),
        "admits_below_budget": len(below_budget),
        "admits_below_budget_own_request": sum(
            1 for row in below_budget if row.correlation == CORRELATION_OWN
        ),
        "admits_below_budget_ambiguous": sum(
            1 for row in below_budget if row.ambiguous
        ),
        "median_projected_s": _median(projected),
        "median_observed_s": _median(observed),
        "projected_buckets": _count(projected_bucket(value) for value in projected),
        "observed_minus_projected_s": {
            "median": _median(gaps),
            "min": min(gaps) if gaps else None,
            "max": max(gaps) if gaps else None,
        },
        "projected_lt_observed": sum(1 for gap in gaps if gap > 0),
        "median_eta_prefill_s": _median(
            [
                row.guard.eta_prefill_s
                for row in rows
                if row.guard is not None and row.guard.eta_prefill_s is not None
            ]
        ),
        "median_eta_occupancy_s": _median(
            [
                row.guard.eta_occupancy_s
                for row in rows
                if row.guard is not None and row.guard.eta_occupancy_s is not None
            ]
        ),
        #: Rows whose occupancy term is exactly the symmetric ``eta * busy_slots``
        #: floor, i.e. the occupant's measured remaining service added nothing.
        #: This is the discriminator between "the occupant is priced" and "the
        #: occupant is priced as if it were the newcomer's own prefill".
        "occupancy_is_symmetric_floor": sum(
            1
            for row in rows
            if row.guard is not None
            and row.guard.eta_prefill_s is not None
            and row.guard.eta_occupancy_s is not None
            and row.guard.busy_slots is not None
            and abs(
                row.guard.eta_occupancy_s
                - row.guard.eta_prefill_s * row.guard.busy_slots
            )
            < 0.15
        ),
        "occupancy_raised_by_measurement": sum(
            1
            for row in rows
            if row.guard is not None
            and row.guard.eta_prefill_s is not None
            and row.guard.eta_occupancy_s is not None
            and row.guard.busy_slots is not None
            and row.guard.eta_occupancy_s
            > row.guard.eta_prefill_s * row.guard.busy_slots + 0.15
        ),
        "median_busy_slots": _median(
            [
                float(row.guard.busy_slots)
                for row in rows
                if row.guard is not None and row.guard.busy_slots is not None
            ]
        ),
        "busy_slots": _count(
            str(row.guard.busy_slots)
            for row in rows
            if row.guard is not None and row.guard.busy_slots is not None
        ),
        "in_flight_prefills": _count(
            str(row.guard.in_flight_prefills)
            for row in rows
            if row.guard is not None and row.guard.in_flight_prefills is not None
        ),
        "rate_buckets": _count(
            row.guard.rate_bucket or "unknown"
            for row in rows
            if row.guard is not None
        ),
    }


def _blocker_matrix(rows: list[Row]) -> dict:
    """Blocker class distribution, per trigger and per outcome."""
    by_class = _count_all((row.blocker_class for row in rows), BLOCKER_CLASSES)
    by_class_trigger: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_class_trigger.setdefault(row.trigger, {})
        bucket[row.blocker_class] = bucket.get(row.blocker_class, 0) + 1
    abandoned_by_class_trigger: dict[str, dict[str, int]] = {}
    for row in rows:
        if row.outcome != "abandoned":
            continue
        bucket = abandoned_by_class_trigger.setdefault(row.trigger, {})
        bucket[row.blocker_class] = bucket.get(row.blocker_class, 0) + 1
    return {
        "by_class": by_class,
        "by_class_and_trigger": {
            trigger: dict(sorted(bucket.items()))
            for trigger, bucket in sorted(by_class_trigger.items())
        },
        "abandoned_by_class": _count_all(
            (row.blocker_class for row in rows if row.outcome == "abandoned"),
            BLOCKER_CLASSES,
        ),
        "abandoned_by_class_and_trigger": {
            trigger: dict(sorted(bucket.items()))
            for trigger, bucket in sorted(abandoned_by_class_trigger.items())
        },
        "abandoned_by_class_and_own_request": _count(
            row.blocker_class
            for row in rows
            if row.outcome == "abandoned" and row.correlation == CORRELATION_OWN
        ),
        "blocker_is_self": sum(1 for row in rows if row.blocker_is_self),
        #: Blocker class against the guard's own occupancy observation. The two
        #: are not the same quantity — ``in_flight_prefills`` is process-wide
        #: while the blocker class names the slot holder — so this is a
        #: contingency table, not an agreement score. ``generating`` with
        #: ``in_flight_prefills=0`` is direct corroboration (nothing was
        #: prefilling, so the holder could only be decoding).
        "by_class_and_in_flight_prefills": _cross_tab(
            rows,
            lambda row: row.blocker_class,
            lambda row: (
                str(row.guard.in_flight_prefills)
                if row.guard is not None
                and row.guard.in_flight_prefills is not None
                else "no_guard"
            ),
        ),
    }


def _cross_tab(rows: list[Row], row_key, col_key) -> dict[str, dict[str, int]]:
    """Count *rows* into a ``row_key -> col_key -> count`` contingency table."""
    table: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = table.setdefault(row_key(row), {})
        column = col_key(row)
        bucket[column] = bucket.get(column, 0) + 1
    return {key: dict(sorted(value.items())) for key, value in sorted(table.items())}


def build_summary(rows: list[Row]) -> dict:
    """Aggregate *rows* into the diagnosis summary the record quotes."""
    abandoned = [row for row in rows if row.outcome == "abandoned"]
    dispatched = [row for row in rows if row.outcome == "dispatched"]
    by_band: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_band.setdefault(
            row.band, {"queued": 0, "dispatched": 0, "abandoned": 0, "no_outcome": 0}
        )
        bucket["queued"] += 1
        bucket[row.outcome or "no_outcome"] += 1
    return {
        "queue_waits": len(rows),
        "by_outcome": _count(row.outcome or "no_outcome" for row in rows),
        "by_trigger": _count(row.trigger for row in rows),
        "by_cause": _count(row.cause or "unknown" for row in rows),
        "abandoned_by_trigger": _count(row.trigger for row in abandoned),
        "abandoned_by_cause": _count(row.cause or "unknown" for row in abandoned),
        "correlation": {
            **_count_all((row.correlation for row in rows), CORRELATIONS),
            "ambiguous": sum(1 for row in rows if row.ambiguous),
            "unambiguous_candidates": sum(
                1 for row in rows if row.candidates == 1
            ),
        },
        "abandoned_correlation": {
            **_count_all((row.correlation for row in abandoned), CORRELATIONS),
            "ambiguous": sum(1 for row in abandoned if row.ambiguous),
            "unambiguous_candidates": sum(
                1 for row in abandoned if row.candidates == 1
            ),
        },
        "abandoned_guard": _guard_projection_summary(abandoned),
        "dispatched_guard": _guard_projection_summary(dispatched),
        "blocker": _blocker_matrix(rows),
        "bands": {band: bucket for band, bucket in sorted(by_band.items())},
        "band_evidence": _count_all(
            (row.band_evidence for row in rows),
            ("own_outcome", "own_enqueue", "session_chain", "unknown"),
        ),
        "abandoned_band_evidence": _count_all(
            (row.band_evidence for row in abandoned),
            ("own_outcome", "own_enqueue", "session_chain", "unknown"),
        ),
        #: Per-wait proof of AC3c: what the **pre-fix** rule would have banded,
        #: and how many waits the own-request rule moved out of a band.
        "legacy_bands": _legacy_band_counts(rows),
        "rebanded": sum(1 for row in rows if row.legacy_band != row.band),
        "rebanded_abandoned": sum(
            1 for row in abandoned if row.legacy_band != row.band
        ),
        "projected_bucket_order": _projected_bucket_order(),
    }


def _legacy_band_counts(rows: list[Row]) -> dict[str, dict[str, int]]:
    """Count rows by the pre-fix band rule, split by outcome."""
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = counts.setdefault(
            row.legacy_band,
            {"queued": 0, "dispatched": 0, "abandoned": 0, "no_outcome": 0},
        )
        bucket["queued"] += 1
        bucket[row.outcome or "no_outcome"] += 1
    return {band: bucket for band, bucket in sorted(counts.items())}


def analyse(
    log_dir: Path,
    since: datetime,
    until: datetime,
    *,
    blocker_lookback_s: float = DEFAULT_BLOCKER_LOOKBACK_S,
) -> dict:
    """Return the full JSON diagnosis for the window."""
    collected = collect(log_dir, since, until)
    rows = build_rows(collected, blocker_lookback_s=blocker_lookback_s)
    coverage = measure_log_coverage(log_dir, since, until)
    return {
        "since": since.isoformat(sep=" "),
        "until": until.isoformat(sep=" "),
        "log_dir": str(log_dir),
        "blocker_lookback_s": blocker_lookback_s,
        "summary": build_summary(rows),
        "rows": [row.to_dict() for row in rows],
        "log_files": sorted(collected.files),
        "log_files_scanned": [
            path.name for path in discover_proxy_log_files(log_dir)
        ],
        "coverage": {
            "covered": coverage.covered,
            "files": [span.path.name for span in coverage.spans],
            "window_start_gap_seconds": (
                round(coverage.window_start_gap_seconds, 1)
                if coverage.window_start_gap_seconds is not None
                else None
            ),
            "window_end_gap_seconds": (
                round(coverage.window_end_gap_seconds, 1)
                if coverage.window_end_gap_seconds is not None
                else None
            ),
            "gaps": [
                {
                    "before": gap.before.name,
                    "before_last": gap.before_last.isoformat(sep=" "),
                    "after": gap.after.name,
                    "after_first": gap.after_first.isoformat(sep=" "),
                    "gap_seconds": round(gap.gap_seconds, 1),
                }
                for gap in coverage.gaps
            ],
        },
    }


def render_text(report: dict) -> str:
    """Render a compact human-readable digest of the report."""
    summary = report["summary"]
    abandoned = summary["abandoned_guard"]
    dispatched = summary["dispatched_guard"]
    lines = [
        f"Window: {report['since']} -> {report['until']}",
        f"Log dir: {report['log_dir']}",
        f"Coverage complete: {report['coverage']['covered']} "
        f"(gaps: {len(report['coverage']['gaps'])})",
        "",
        f"Queue waits: {summary['queue_waits']} {summary['by_outcome']}",
        f"By trigger: {summary['by_trigger']}",
        f"Abandoned by cause: {summary['abandoned_by_cause']}",
        f"Correlation: {summary['correlation']}",
        f"Abandoned correlation: {summary['abandoned_correlation']}",
        "",
        "Abandoned waits — guard projection:",
        f"  with guard: {abandoned['with_guard']} admits: {abandoned['admits']} "
        f"admits below budget: {abandoned['admits_below_budget']} "
        f"(own-request {abandoned['admits_below_budget_own_request']}, "
        f"ambiguous {abandoned['admits_below_budget_ambiguous']})",
        f"  median projected: {abandoned['median_projected_s']}s "
        f"median observed: {abandoned['median_observed_s']}s "
        f"median gap: {abandoned['observed_minus_projected_s']['median']}s",
        f"  projected buckets: {abandoned['projected_buckets']}",
        "Dispatched waits — guard projection:",
        f"  with guard: {dispatched['with_guard']} "
        f"median projected: {dispatched['median_projected_s']}s "
        f"median observed: {dispatched['median_observed_s']}s",
        "",
        f"Blocker by class: {summary['blocker']['by_class']}",
        f"Blocker abandoned by class: {summary['blocker']['abandoned_by_class']}",
        "Blocker abandoned by class and trigger: "
        f"{summary['blocker']['abandoned_by_class_and_trigger']}",
        f"Blocker is the waiting session itself: {summary['blocker']['blocker_is_self']}",
        "",
        f"Bands: {summary['bands']}",
        f"Band evidence: {summary['band_evidence']}",
        f"Abandoned band evidence: {summary['abandoned_band_evidence']}",
        f"Legacy (pre-fix) bands: {summary['legacy_bands']}",
        f"Rebanded by the own-request rule: {summary['rebanded']} "
        f"(abandoned {summary['rebanded_abandoned']})",
        f"Occupancy is the symmetric floor: {abandoned['occupancy_is_symmetric_floor']}"
        f"/{abandoned['with_guard']} (raised by measurement: "
        f"{abandoned['occupancy_raised_by_measurement']})",
        "Blocker class vs guard in_flight_prefills: "
        f"{summary['blocker']['by_class_and_in_flight_prefills']}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Per-wait diagnosis of abandoned contention-queue waits: the "
            "same-request admission projection against the observed wait, plus "
            "the class of blocker that held the slot "
            "(LP-0MVV29XPT003OXI5)."
        )
    )
    parser.add_argument(
        "--since", required=True, type=parse_timestamp, help="window start (inclusive)"
    )
    parser.add_argument(
        "--until", required=True, type=parse_timestamp, help="window end (inclusive)"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"proxy log directory (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--blocker-lookback",
        type=float,
        default=DEFAULT_BLOCKER_LOOKBACK_S,
        help=(
            "seconds before an enqueue that a lease marker may sit and still "
            f"describe the holder (default: {DEFAULT_BLOCKER_LOOKBACK_S})"
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.until < args.since:
        print("error: --until must not precede --since", file=sys.stderr)
        return 2
    report = analyse(
        args.log_dir,
        args.since,
        args.until,
        blocker_lookback_s=args.blocker_lookback,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
