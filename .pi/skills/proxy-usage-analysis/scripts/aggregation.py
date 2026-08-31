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
- fast/cheap bucket and slot count come from the slot schedule, keyed by
  session start time.

Window semantics: only events with ``window_start <= ts <= window_end`` are
aggregated. A session is included iff it has at least one in-window
``Stream started``; its start is the first in-window stream.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import bucketing
from log_parser import (
    BUSY_WINDOW_MARGIN,
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
    # Per-period context window of the profile active for this session (from
    # the mode profile's slot_schedule ctx_size, or its global
    # local_model_ctx_size when the entry did not pin one); None when not
    # derivable.
    ctx_size: int | None = None
    # Decode speed derived from local streams: total local completion tokens /
    # local active span (first→last local stream event). None when not derivable.
    decode_tok_s: float | None = None

    @property
    def fell_back(self) -> bool:
        return self.remote_move_time is not None


@dataclass
class AnalysisResult:
    window_start: datetime
    window_end: datetime
    sessions: dict[str, SessionStats]
    fallback_events: list[LogEvent]
    routing_skip_events: list[LogEvent]
    dispatch_denied_count: int
    unattributed_events: int
    lines_skipped: int
    total_lines: int
    dispatch_denied_events: list[LogEvent] = field(default_factory=list)
    # Contention-queue events (LP-0MSORQVK50012Q4D F4 AC3): queued requests
    # dispatched local after a slot freed, and queued requests that fell back
    # to a remote provider after the wait/depth caps were exceeded.
    contention_dispatch_events: list[LogEvent] = field(default_factory=list)
    contention_fallback_events: list[LogEvent] = field(default_factory=list)
    # Parsed error events (stream errors, slot_save failures, backend_retry
    # timeouts, upstream HTTP errors) inside the window.
    error_events: list[LogEvent] = field(default_factory=list)
    # llama-server decode/prompt-eval speed stats (set by reporting.run_analysis).
    speed: object | None = None
    # Local-model utilization (busy time etc.); None when no local traffic.
    busy: BusyStats | None = None

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

    @property
    def error_counts(self) -> Counter:
        """Error events grouped by error type (event kind)."""
        return Counter(e.kind for e in self.error_events)

    @property
    def error_provider_model_counts(self) -> Counter:
        """Error events grouped by (error type, provider, model)."""
        return Counter((e.kind, e.provider, e.model) for e in self.error_events)

    @property
    def upstream_error_by_status(self) -> dict[int, int]:
        """Upstream HTTP error events grouped by HTTP status code."""
        counts: dict[int, int] = {}
        for e in self.error_events:
            if e.kind == "upstream_http_error" and e.status is not None:
                counts[e.status] = counts.get(e.status, 0) + 1
        return counts

    @property
    def upstream_error_by_status_provider(self) -> dict[int, dict[str, int]]:
        """Upstream HTTP error events grouped by (status, provider)."""
        result: dict[int, dict[str, int]] = {}
        for e in self.error_events:
            if e.kind == "upstream_http_error" and e.status is not None:
                provider = e.provider or "(unknown)"
                if e.status not in result:
                    result[e.status] = {}
                result[e.status][provider] = result[e.status].get(provider, 0) + 1
        return result


@dataclass
class BusyStats:
    """Local-model utilization over the analysis window.

    "Busy" means at least one local slot is actively generating: streams are
    paired per session (FIFO across the full event stream, not just the
    window), each pair is clipped to ``[window_start, window_end]``, and the
    clipped intervals are merged so overlapping streams do not double-count.

    ``total_compute_seconds`` is the sum of all clipped stream durations
    (slot-seconds, i.e. the integral of concurrency); ``busy_seconds`` is the
    union of active intervals. ``avg_concurrency`` = total / busy.
    """

    window_seconds: float
    busy_seconds: float
    total_compute_seconds: float
    streams: int
    peak_concurrency: int
    avg_concurrency: float
    avg_stream_duration: float
    unfinished_streams: int
    # Streams that started more than ``BUSY_WINDOW_MARGIN`` before the window
    # start and have no logged finish: stale pre-window leftovers from earlier
    # windows, tracked separately from in-window aborts (LP-0MSVRRO3L0056N6C).
    pre_window_unfinished: int
    # Busy seconds attributed to fast/cheap periods (from the slot schedule)
    # and to each hour of the window (hour-of-day -> seconds).
    fast_busy_seconds: float
    cheap_busy_seconds: float
    fast_window_seconds: float
    cheap_window_seconds: float
    hourly_busy: list[tuple[int, float]]

    @property
    def busy_pct(self) -> float:
        return (self.busy_seconds / self.window_seconds * 100.0) if self.window_seconds else 0.0

    @property
    def idle_seconds(self) -> float:
        return max(0.0, self.window_seconds - self.busy_seconds)

    @property
    def idle_pct(self) -> float:
        return (self.idle_seconds / self.window_seconds * 100.0) if self.window_seconds else 0.0


class _SessionBuilder:
    """Accumulates raw event data for one session while streaming."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.started: list[LogEvent] = []
        self.finished: list[LogEvent] = []
        self.last_seen: datetime | None = None
        self.dispatch_denied = 0
        self.routing_skips: list[LogEvent] = []
        # Local-only stream tracking (decode-speed fallback derivation).
        self.local_first: datetime | None = None
        self.local_last: datetime | None = None
        self.local_completion = 0

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
        if ev.provider == LOCAL_PROVIDER and ev.kind in ("stream_started", "stream_finished"):
            if self.local_first is None or ev.ts < self.local_first:
                self.local_first = ev.ts
            if self.local_last is None or ev.ts > self.local_last:
                self.local_last = ev.ts
            if ev.kind == "stream_finished" and ev.completion is not None:
                self.local_completion += ev.completion


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
    routing_skips: dict[str, list[LogEvent]],
    fallback_events: list[LogEvent],
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
    routing_skips: dict[str, list[LogEvent]],
    fallback_events: list[LogEvent],
    schedule: bucketing.SlotSchedule,
    mode_map: bucketing.ModeScheduleMap | None = None,
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

    # Bucket by the operating mode active at the session's first in-window
    # stream (LP-0MSPZUD4G007IYGH): when the logs show mode transitions, the
    # period is the mode's profile period labelled with the mode name; without
    # transitions (a single-mode window) the legacy slot-count bucketing from
    # the analysis-time schedule applies unchanged.
    period: bucketing.SlotPeriod | None = None
    if mode_map is not None and mode_map.transitions:
        period = mode_map.period_for(start)
    elif schedule.periods:
        period = schedule.period_for(start)

    local_req = sum(1 for e in started if e.provider == LOCAL_PROVIDER)
    remote_req = len(started) - local_req

    # Decode speed fallback: total local completion tokens / local active span
    # (first→last local stream event). Only derivable with local completions
    # and a positive span; conservative (includes inter-request gaps).
    decode_tok_s: float | None = None
    if builder.local_completion > 0 and builder.local_first is not None and builder.local_last is not None:
        span = (builder.local_last - builder.local_first).total_seconds()
        if span > 0:
            decode_tok_s = round(builder.local_completion / span, 1)

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
        ctx_size=period.ctx_size if period else None,
        dispatch_denied=builder.dispatch_denied,
        routing_skips=len(builder.routing_skips),
        decode_tok_s=decode_tok_s,
    )


def _segment_boundaries(
    start: datetime,
    end: datetime,
    periods: list[bucketing.SlotPeriod],
) -> list[datetime]:
    """All timestamps inside ``(start, end)`` where the hour-of-day or a
    fast/cheap period (slot schedule) boundary falls. Used to attribute busy
    seconds to hours and fast/cheap buckets exactly."""
    boundaries = {start, end}
    t = start.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    while t < end:
        boundaries.add(t)
        t += timedelta(hours=1)
    for period in periods:
        for minutes in (period.start_minutes, period.end_minutes):
            day = start.replace(hour=0, minute=0, second=0, microsecond=0)
            while day <= end:
                cand = day + timedelta(minutes=minutes)
                if start < cand < end:
                    boundaries.add(cand)
                day += timedelta(days=1)
    return sorted(boundaries)



def _period_label_at(
    dt: datetime,
    schedule: bucketing.SlotSchedule,
    mode_map: bucketing.ModeScheduleMap | None,
) -> str:
    """Fast/cheap label for ``dt``: mode-aware when the logs show mode
    transitions (LP-0MSPZUD4G007IYGH), else the legacy slot-count label from
    the analysis-time schedule."""
    if mode_map is not None and mode_map.transitions:
        return mode_map.period_for(dt).label
    return schedule.period_for(dt).label if schedule.periods else "fast"



def _attribute_interval(
    interval_start: datetime,
    interval_end: datetime,
    periods: list[bucketing.SlotPeriod],
    label_at,
    bucket_busy: dict[str, float],
    hourly: dict[int, float],
) -> None:
    """Add one merged busy interval's seconds to the fast/cheap and hourly
    buckets it overlaps, splitting at hour and period boundaries."""
    b = _segment_boundaries(interval_start, interval_end, periods)
    for lo, hi in zip(b, b[1:]):
        if hi <= lo:
            continue
        mid = lo + (hi - lo) / 2
        label = label_at(mid)
        bucket_busy[label] = bucket_busy.get(label, 0.0) + (hi - lo).total_seconds()
        hourly[mid.hour] = hourly.get(mid.hour, 0.0) + (hi - lo).total_seconds()


def compute_busy_stats(
    local_events: Iterable[LogEvent],
    window_start: datetime,
    window_end: datetime,
    schedule: bucketing.SlotSchedule,
    mode_map: bucketing.ModeScheduleMap | None = None,
) -> BusyStats | None:
    """Compute local-model utilization from local stream events.

    ``local_events`` may span a margin beyond the window (see
    ``log_parser.iter_events(margin=...)``); each stream is clipped back to
    ``[window_start, window_end]``. Streams whose start has no paired finish
    are counted in ``unfinished_streams`` (their compute time is unknown, so
    busy time is a conservative lower bound) **only when the stream started
    inside the window or within ``BUSY_WINDOW_MARGIN`` before it**; streams
    that started more than ``BUSY_WINDOW_MARGIN`` before the window are stale
    pre-window leftovers from earlier windows and are tracked separately in
    ``pre_window_unfinished`` (LP-0MSVRRO3L0056N6C). Streams started after
    the window end are not counted (they belong to the next window).
    Fast/cheap attribution is mode-aware when the logs show mode transitions
    (LP-0MSPZUD4G007IYGH), so busy seconds during cheap hours count as cheap
    even when the analysis runs in fast mode. Returns ``None`` when there is
    no local traffic in the window.
    """
    started: dict[str, list[datetime]] = {}
    finished: dict[str, list[datetime]] = {}
    for ev in local_events:
        if ev.session is None:
            continue
        bucket = started if ev.kind == "stream_started" else finished
        bucket.setdefault(ev.session, []).append(ev.ts)

    intervals: list[tuple[datetime, datetime]] = []
    unfinished = 0
    pre_window_unfinished = 0
    margin_start = window_start - BUSY_WINDOW_MARGIN
    for session_id, starts in started.items():
        ss = sorted(starts)
        ff = sorted(finished.get(session_id, []))
        for b, e in zip(ss, ff):
            cb, ce = max(b, window_start), min(e, window_end)
            if ce > cb:
                intervals.append((cb, ce))
        # FIFO pairing: the earliest ``len(ff)`` starts pair with finishes;
        # the remaining starts are unpaired (no logged finish). Only starts
        # within the busy margin before the window (or inside it) count as
        # this window's unfinished; earlier starts are pre-window leftovers;
        # starts after the window end belong to the next window.
        for ts in ss[len(ff):]:
            if ts < margin_start:
                pre_window_unfinished += 1
            elif ts <= window_end:
                unfinished += 1
    if not intervals:
        return None

    intervals.sort()
    merged: list[list[datetime]] = []
    for cb, ce in intervals:
        if merged and cb <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], ce)
        else:
            merged.append([cb, ce])

    busy_seconds = sum((e - s).total_seconds() for s, e in merged)
    total_compute = sum((e - s).total_seconds() for s, e in intervals)
    window_seconds = (window_end - window_start).total_seconds()

    # Peak concurrency via a sweep over interval endpoints.
    sweep: list[tuple[datetime, int]] = []
    for s, e in intervals:
        sweep.append((s, +1))
        sweep.append((e, -1))
    sweep.sort(key=lambda x: (x[0], x[1]))
    cur = peak = 0
    for _t, delta in sweep:
        cur += delta
        peak = max(peak, cur)

    mode_aware = mode_map is not None and bool(mode_map.transitions)
    periods = mode_map.all_periods if mode_aware else schedule.periods

    def label_at(mid: datetime) -> str:
        return _period_label_at(mid, schedule, mode_map)

    bucket_busy: dict[str, float] = {}
    hourly: dict[int, float] = {}
    for s, e in merged:
        _attribute_interval(s, e, periods, label_at, bucket_busy, hourly)

    fast_window = cheap_window = 0.0
    bounds = _segment_boundaries(window_start, window_end, periods)
    for lo, hi in zip(bounds, bounds[1:]):
        if hi <= lo:
            continue
        mid = lo + (hi - lo) / 2
        label = _period_label_at(mid, schedule, mode_map)
        if label == "cheap":
            cheap_window += (hi - lo).total_seconds()
        else:
            fast_window += (hi - lo).total_seconds()

    return BusyStats(
        window_seconds=window_seconds,
        busy_seconds=round(busy_seconds, 1),
        total_compute_seconds=round(total_compute, 1),
        streams=len(intervals),
        peak_concurrency=peak,
        avg_concurrency=round(total_compute / busy_seconds, 2) if busy_seconds else 0.0,
        avg_stream_duration=round(total_compute / len(intervals), 1) if intervals else 0.0,
        unfinished_streams=unfinished,
        pre_window_unfinished=pre_window_unfinished,
        fast_busy_seconds=round(bucket_busy.get("fast", 0.0), 1),
        cheap_busy_seconds=round(bucket_busy.get("cheap", 0.0), 1),
        fast_window_seconds=round(fast_window, 1),
        cheap_window_seconds=round(cheap_window, 1),
        hourly_busy=sorted(hourly.items()),
    )


def aggregate(
    events: Iterable[LogEvent],
    window_start: datetime,
    window_end: datetime,
    schedule: bucketing.SlotSchedule,
    mode_map: bucketing.ModeScheduleMap | None = None,
) -> AnalysisResult:
    """Group in-window events into sessions and compute per-session stats.

    ``events`` may be any iterable of parsed :class:`LogEvent` (e.g. chained
    across multiple log files). Events outside the window are ignored.

    When ``mode_map`` is given, mode-switch events (``Mode scheduler:
    applied scheduled mode`` for scheduled transitions plus the
    ``Grandfathering: enabled; ... (current=<mode>)`` marker for manual
    switches, LP-0MT1EE315007AKXG; yielded by ``iter_events`` across the
    margin-widened window) are collected into a mode timeline, and each
    session is bucketed by the mode active at its first in-window stream
    (LP-0MSPZUD4G007IYGH).
    """
    builders: dict[str, _SessionBuilder] = {}
    routing_skips: dict[str, list[LogEvent]] = {}
    fallback_events: list[LogEvent] = []
    routing_skip_events: list[LogEvent] = []
    dispatch_denied_events: list[LogEvent] = []
    error_events: list[LogEvent] = []
    contention_dispatch_events: list[LogEvent] = []
    contention_fallback_events: list[LogEvent] = []
    dispatch_denied = 0
    unattributed = 0
    lines_skipped = 0
    total_lines = 0
    # Local stream events across the (margin-widened) event stream for the
    # busy-time calculation; ``iter_events`` yields a margin beyond the window
    # so boundary-crossing streams pair correctly (clipped in compute_busy_stats).
    local_stream_events: list[LogEvent] = []
    # Operating-mode transitions across the (margin-widened) event stream:
    # the mode timeline used for fast/cheap session bucketing.
    mode_transitions: list[tuple[datetime, str]] = []

    for ev in events:
        if ev.kind == "mode_switch":
            if ev.mode in ("fast", "cheap"):
                mode_transitions.append((ev.ts, ev.mode))
            if window_start <= ev.ts <= window_end:
                total_lines += 1
            continue
        if ev.provider == LOCAL_PROVIDER and ev.kind in ("stream_started", "stream_finished"):
            local_stream_events.append(ev)
        if not (window_start <= ev.ts <= window_end):
            continue
        total_lines += 1
        if ev.kind in (
            "stream_error",
            "stream_finish_error",
            "slot_save_error",
            "backend_retry",
            "upstream_http_error",
        ):
            error_events.append(ev)
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
        if ev.kind == "contention_dispatch":
            contention_dispatch_events.append(ev)
            continue
        if ev.kind == "contention_fallback_after_queue":
            contention_fallback_events.append(ev)
            continue
        if ev.kind in ("stream_started", "stream_finished"):
            if not ev.session:
                unattributed += 1
                continue
            builder = builders.setdefault(ev.session, _SessionBuilder(ev.session))
            builder.add(ev)
            continue
        lines_skipped += 1

    if mode_map is not None:
        mode_map.transitions = sorted(mode_transitions)

    sessions: dict[str, SessionStats] = {}
    for sid, builder in builders.items():
        # Require at least one in-window stream *started* event (events have
        # already been window-filtered above).
        if not builder.started:
            continue
        sessions[sid] = _build_session(builder, routing_skips, fallback_events, schedule, mode_map)

    busy = compute_busy_stats(local_stream_events, window_start, window_end, schedule, mode_map)

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
        error_events=error_events,
        contention_dispatch_events=contention_dispatch_events,
        contention_fallback_events=contention_fallback_events,
        busy=busy,
    )
