"""Session grouping and aggregation over parsed proxy log events.

A "session" is identified by its session UUID (``session=<uuid>`` on
``Stream started`` / ``Stream finished`` lines). Per-session stats follow the
work item's decisions:

- one row per session (start/avg/max context size, avg/max response size);
- context/response sizes come from the authoritative per-request
  ``tokens=prompt/completion/total`` on ``Stream finished`` lines;
- the initial model assignment is the provider/model of the session's first
  ``Stream started`` line;
- the move to a remote model is attributed from the session's first
  ``routing_skip_local`` line (carries session + reason), falling back to the
  nearest ``Fallback triggered`` line within 60s before the first remote
  stream, and finally to the first remote stream time itself;
- day/night bucket and slot count come from the slot schedule, keyed by
  session start time.

Window semantics: only events with ``window_start <= ts <= window_end`` are
aggregated. A session is included iff it has at least one in-window
``Stream started``; its start is the first in-window stream.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List

import bucketing
from log_parser import (
    FALLBACK_ATTRIBUTION_WINDOW_SECONDS,
    LOCAL_PROVIDER,
    LogEvent,
)

_EMPTY = object()


@dataclass
class SessionStats:
    """Aggregated statistics for one session."""

    session_id: str
    start: datetime
    end: datetime
    duration_seconds: float
    messages: int
    local_requests: int
    remote_requests: int
    start_context_size: int | None
    avg_context_size: float | None
    max_context_size: int | None
    avg_response_size: float | None
    max_response_size: int | None
    initial_provider: str
    initial_model: str
    remote_provider: str | None
    remote_model: str | None
    remote_move_time: datetime | None
    fallback_reason: str | None
    bucket: str | None
    slots: int | None
    dispatch_denied: int
    routing_skips: int

    @property
    def fell_back(self) -> bool:
        return self.remote_move_time is not None


@dataclass
class AnalysisResult:
    window_start: datetime
    window_end: datetime
    sessions: Dict[str, SessionStats]
    fallback_events: List[LogEvent]
    routing_skip_events: List[LogEvent]
    dispatch_denied_count: int
    unattributed_events: int
    lines_skipped: int
    total_lines: int
    dispatch_denied_events: List[LogEvent] = field(default_factory=list)

    @property
    def total_requests(self) -> int:
        return sum(s.messages for s in self.sessions.values())

    @property
    def local_requests(self) -> int:
        return sum(s.local_requests for s in self.sessions.values())

    @property
    def remote_requests(self) -> int:
        return sum(s.remote_requests for s in self.sessions.values())

    @property
    def fallback_reason_counts(self) -> Counter:
        return Counter(e.reason for e in self.fallback_events if e.reason)

    @property
    def routing_skip_reason_counts(self) -> Counter:
        return Counter(e.reason for e in self.routing_skip_events if e.reason)


class _SessionBuilder:
    """Accumulates raw event data for one session while streaming."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.started: List[LogEvent] = []
        self.finished: List[LogEvent] = []
        self.last_seen: datetime | None = None
        self.dispatch_denied = 0
        self.routing_skips: List[LogEvent] = []

    def add(self, ev: LogEvent) -> None:
        self.last_seen = ev.ts
        if ev.kind == "stream_started":
            self.started.append(ev)
        elif ev.kind == "stream_finished":
            self.finished.append(ev)
        elif ev.kind == "routing_skip":
            self.routing_skips.append(ev)
        elif ev.kind == "dispatch_denied":
            self.dispatch_denied += 1


def _mean(values: list) -> float | None:
    return round(sum(values) / len(values), 1) if values else None


def _first_remote(events: Iterable[LogEvent]) -> LogEvent | None:
    for ev in events:
        if ev.provider and ev.provider != LOCAL_PROVIDER:
            return ev
    return None


def _attribute_fallback(
    session_id: str,
    first_remote_ts: datetime,
    routing_skips: Dict[str, List[LogEvent]],
    fallback_events: List[LogEvent],
) -> tuple[datetime | None, str | None]:
    """Find (move_time, reason) for a session that reached a remote provider.

    Priority:
    1. The session's own ``routing_skip_local`` line at/before the first
       remote stream (authoritative: carries the session UUID and reason).
    2. The nearest session-less ``Fallback triggered`` line within
       ``FALLBACK_ATTRIBUTION_WINDOW_SECONDS`` before the first remote stream.
    Returns ``(None, None)`` when there is no evidence.
    """
    skips = [e for e in routing_skips.get(session_id, []) if e.ts <= first_remote_ts]
    if skips:
        e = max(skips, key=lambda x: x.ts)
        return e.ts, e.reason
    earliest = first_remote_ts - timedelta(seconds=FALLBACK_ATTRIBUTION_WINDOW_SECONDS)
    candidates = [e for e in fallback_events if earliest <= e.ts <= first_remote_ts]
    if candidates:
        e = max(candidates, key=lambda x: x.ts)
        return e.ts, e.reason
    return None, None


def _build_session(
    builder: _SessionBuilder,
    routing_skips: Dict[str, List[LogEvent]],
    fallback_events: List[LogEvent],
    schedule: bucketing.SlotSchedule,
) -> SessionStats:
    started = sorted(builder.started, key=lambda e: e.ts)
    finished = sorted(builder.finished, key=lambda e: e.ts)

    first = started[0] if started else (finished[0] if finished else None)
    start = first.ts if first is not None else builder.last_seen
    end = builder.last_seen or start

    first_provider = first.provider if first is not None else ""
    first_model = first.model if first is not None else ""

    prompt_tokens = [e.prompt for e in finished if e.prompt is not None]
    completion_tokens = [e.completion for e in finished if e.completion is not None]

    first_remote = _first_remote(started)
    has_local = any(e.provider == LOCAL_PROVIDER for e in started)

    if first_remote is not None:
        move_ts, reason = _attribute_fallback(
            builder.session_id, first_remote.ts, routing_skips, fallback_events
        )
        if move_ts is None:
            move_ts = first_remote.ts if has_local else None
    else:
        move_ts, reason = None, None

    period = schedule.period_for(start) if schedule.periods else None

    local_req = sum(1 for e in started if e.provider == LOCAL_PROVIDER)
    remote_req = len(started) - local_req

    return SessionStats(
        session_id=builder.session_id,
        start=start,
        end=end,
        duration_seconds=round((end - start).total_seconds(), 1),
        messages=len(started),
        local_requests=local_req,
        remote_requests=remote_req,
        start_context_size=prompt_tokens[0] if prompt_tokens else None,
        avg_context_size=_mean(prompt_tokens),
        max_context_size=max(prompt_tokens) if prompt_tokens else None,
        avg_response_size=_mean(completion_tokens),
        max_response_size=max(completion_tokens) if completion_tokens else None,
        initial_provider=first_provider,
        initial_model=first_model,
        remote_provider=first_remote.provider if first_remote else None,
        remote_model=first_remote.model if first_remote else None,
        remote_move_time=move_ts,
        fallback_reason=reason,
        bucket=period.label if period else None,
        slots=period.slots if period else None,
        dispatch_denied=builder.dispatch_denied,
        routing_skips=len(builder.routing_skips),
    )


def aggregate(
    events: Iterable[LogEvent],
    window_start: datetime,
    window_end: datetime,
    schedule: bucketing.SlotSchedule,
) -> AnalysisResult:
    """Group in-window events into sessions and compute per-session stats.

    ``events`` may be any iterable of parsed :class:`LogEvent` (e.g. chained
    across multiple log files). Events outside the window are ignored.
    """
    builders: Dict[str, _SessionBuilder] = {}
    routing_skips: Dict[str, List[LogEvent]] = {}
    fallback_events: List[LogEvent] = []
    routing_skip_events: List[LogEvent] = []
    dispatch_denied_events: List[LogEvent] = []
    dispatch_denied = 0
    unattributed = 0
    lines_skipped = 0
    total_lines = 0

    for ev in events:
        total_lines += 1
        if not (window_start <= ev.ts <= window_end):
            continue
        if ev.kind == "fallback":
            fallback_events.append(ev)
            continue
        if ev.kind == "dispatch_denied":
            dispatch_denied += 1
            dispatch_denied_events.append(ev)
            if ev.session:
                builders.setdefault(ev.session, _SessionBuilder(ev.session)).add(ev)
            continue
        if ev.kind == "routing_skip":
            routing_skip_events.append(ev)
            if ev.session:
                routing_skips.setdefault(ev.session, []).append(ev)
                builders.setdefault(ev.session, _SessionBuilder(ev.session)).add(ev)
            continue
        if ev.kind in ("stream_started", "stream_finished"):
            if not ev.session:
                unattributed += 1
                continue
            builder = builders.setdefault(ev.session, _SessionBuilder(ev.session))
            builder.add(ev)
            continue
        lines_skipped += 1

    sessions: Dict[str, SessionStats] = {}
    for sid, builder in builders.items():
        # Require at least one in-window stream *started* event (events have
        # already been window-filtered above).
        if not builder.started:
            continue
        sessions[sid] = _build_session(builder, routing_skips, fallback_events, schedule)

    return AnalysisResult(
        window_start=window_start,
        window_end=window_end,
        sessions=sessions,
        fallback_events=fallback_events,
        routing_skip_events=routing_skip_events,
        dispatch_denied_count=dispatch_denied,
        unattributed_events=unattributed,
        lines_skipped=lines_skipped,
        total_lines=total_lines,
        dispatch_denied_events=dispatch_denied_events,
    )
