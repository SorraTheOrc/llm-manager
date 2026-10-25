"""Session grouping and aggregation over parsed proxy log events.

A "session" is identified by its session UUID (``session=<uuid>`` on
``Stream started`` / ``Stream finished`` lines). Per-session stats follow the
work item's decisions:

- one row per session (start/avg/max context size, avg/max response size);
- context/response sizes come from the authoritative per-request
  ``tokens=prompt/completion/total`` on ``Stream finished`` lines;
- the initial model assignment is the provider/model of the session's first
  ``Stream started`` line;
- the move to a remote model is attributed from the session's own
  ``routing_skip_local`` or ``routing_skip_provider_context`` line (both carry
  the session and the reason; the provider-context skip also names the
  provider and the ceiling that blocked that attempt), or its own
  session-bearing ``Fallback triggered`` line, falling back to the nearest
  session-less ``Fallback triggered`` line within 60s before the first remote
  stream (rotated logs predating the session stamp only), then to a local-lease
  refusal, and finally to the first remote stream time itself. Another
  session's reason is never inherited — the provenance is recorded in
  ``SessionStats.fallback_attribution``;
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
from bucketing import ModeTransition
from log_parser import (
    BUSY_WINDOW_MARGIN,
    FALLBACK_ATTRIBUTION_WINDOW_SECONDS,
    LOCAL_PROVIDER,
    LogEvent,
    PrefillMatch,
    pair_ttft_to_streams,
)

# Hourly local-availability states (LP-0MVKUKG3F0035I85): each hour's state is
# derived **only from timestamped proxy-log evidence** (llama-server eval-timing
# lines carry no per-line timestamps and cannot be attributed to an hour
# deterministically — documented in REFERENCE.md "Limitations"). The four
# states distinguish a wedged gate (lease held by a prefill that never decodes)
# from a genuinely idle GPU, which busy-% alone cannot: busy is computed only
# from paired ``Stream started``/``Stream finished`` events, so an outage where
# work never yields a first byte renders as 0.0% idle.
AVAILABILITY_SERVING = "serving"
AVAILABILITY_PREFILL_ONLY_NO_DECODE = "prefill-only-no-decode"
AVAILABILITY_LEASE_BLOCKED = "lease-blocked"
AVAILABILITY_IDLE = "idle"
# Canonical display order for multi-state hours (serving/prefill are derived
# from decode/dispatch evidence; lease-blocked from refusal evidence).
AVAILABILITY_ORDER = (
    AVAILABILITY_SERVING,
    AVAILABILITY_PREFILL_ONLY_NO_DECODE,
    AVAILABILITY_LEASE_BLOCKED,
)
# Proxy ``reason=`` value that marks a local-lease refusal.
LEASE_ACTIVE_REASON = "local_lease_active"

# Per-session fallback-attribution states (LP-0MVMBMSWR0036RES AC5). Since the
# proxy stamps every ``Fallback triggered`` line with ``session=``, a session's
# reason is taken **only from its own evidence**; the legacy nearest-line
# heuristic survives for rotated logs predating the change (their lines carry
# no ``session=``), so historical windows keep working.
#
# - ``own_fallback`` — the session's own ``Fallback triggered`` line;
# - ``own_skip`` — the session's own ``routing_skip_local`` line (unchanged
#   precedence over the fallback line, which normally carries the same reason);
# - ``own_provider_skip`` — the session's own ``routing_skip_provider_context``
#   line (LP-0MVMBM1Y3001TPRT): a per-provider context-ceiling skip. It outranks
#   the session's own fallback line because it names the provider and the
#   ``limit`` that blocked that chain step, while ``own_skip`` still outranks it
#   (a local-side skip explains why local was not used at all, and the digest
#   of provider-context skips is reported separately);
# - ``legacy_nearest`` — a session-less fallback line within
#   :data:`FALLBACK_ATTRIBUTION_WINDOW_SECONDS` (pre-change rotated logs only);
# - ``lease_denied`` — no fallback line, but the session was refused the local
#   dispatch by another session's lease (``local_dispatch_denied``); the cause
#   is lease/concurrency contention, not an inherited reason;
# - ``unattributed`` — reached remote with no own evidence at all (never
#   inherits another session's reason);
# - ``no_remote_fallback`` — the session never used a remote provider, so no
#   fallback occurred.
FALLBACK_ATTRIBUTION_OWN_FALLBACK = "own_fallback"
FALLBACK_ATTRIBUTION_OWN_SKIP = "own_skip"
# Provider context-ceiling skip (LP-0MVMBM1Y3001TPRT): own session-bearing
# evidence that outranks the own-fallback state, see the comment block above.
FALLBACK_ATTRIBUTION_OWN_PROVIDER_SKIP = "own_provider_skip"
FALLBACK_ATTRIBUTION_LEGACY_NEAREST = "legacy_nearest"
FALLBACK_ATTRIBUTION_LEASE_DENIED = "lease_denied"
FALLBACK_ATTRIBUTION_UNATTRIBUTED = "unattributed"
FALLBACK_ATTRIBUTION_NO_REMOTE = "no_remote_fallback"

# Canonical display order for the report's attribution summary.
FALLBACK_ATTRIBUTION_ORDER = (
    FALLBACK_ATTRIBUTION_OWN_FALLBACK,
    FALLBACK_ATTRIBUTION_OWN_SKIP,
    FALLBACK_ATTRIBUTION_OWN_PROVIDER_SKIP,
    FALLBACK_ATTRIBUTION_LEASE_DENIED,
    FALLBACK_ATTRIBUTION_UNATTRIBUTED,
    FALLBACK_ATTRIBUTION_LEGACY_NEAREST,
    FALLBACK_ATTRIBUTION_NO_REMOTE,
)

# Extra per-hour counters feeding the availability classification, layered on
# the LP-0MTYAZFGN003BYQS counters: ``local_dispatch_attempts`` (every
# ``[local] POST`` to llama-server, including attempts that never stream),
# ``local_denials`` (``local_dispatch_denied`` refusals) and
# ``lease_active_fallbacks`` (``Fallback triggered`` /
# ``contention_queue_fallback_after_queue`` with ``reason=local_lease_active``).
AVAILABILITY_COUNTERS = (
    "local_dispatch_attempts",
    "local_served",
    "local_denials",
    "lease_active_fallbacks",
)

UNKNOWN_LABEL = "(unknown)"

# Unit of the slot-seconds figures on :class:`SlotSecondsStats` (and its
# :class:`PrefillStats` / :class:`DecodeStats` subclasses) and on
# :attr:`BusyStats.prefill_stats` / :attr:`BusyStats.decode_stats`
# (LP-0MVJN45LF009CO2V, LP-0MVJNDOUU0059MMO). A phase is reported as
# slot-seconds — the integral of concurrency, i.e. the sum of the per-stream
# phase durations — which is NOT wall-clock elapsed time.
SLOT_SECONDS_UNIT = "slot-seconds"

# Back-compat alias for the original prefill-only name (LP-0MVJN45LF009CO2V).
PREFILL_UNIT = SLOT_SECONDS_UNIT

# Busy definition (LP-0MVJN9H2I00116FP). Local "busy" is at least one local slot
# occupied **from dispatch to finish** (prefill + decode), not decode only: the
# interval starts at the stream's back-calculated dispatch time
# (``ttft.ts - ttft_ms/1000``) and ends at its ``Stream finished`` line. The
# previous decode-only figure is retained as
# :attr:`BusyStats.decode_busy_seconds` and labels the **lower bound** of the
# busy figure, while the dispatch → finish union labels the **upper bound**:
# a stream with no first-byte line contributes only its decode interval, so the
# upper bound is never overstated by a fabricated prefill.
BUSY_DEFINITION = "dispatch_to_finish"
BUSY_UPPER_BOUND_LABEL = "dispatch → finish union (prefill + decode)"
BUSY_LOWER_BOUND_LABEL = "decode-only union (Stream started → Stream finished)"

# Root-cause classification for stream errors (LP-0MUQO82LJ008F1FS).
# A stream error is the client-visible synthetic ``finish_reason: error``
# (``stream_finish_error``) or its proxy-side companion (``stream_error``);
# the taxonomy attributes it to one of four deterministic buckets,
# validated against the 2026-09-25 → 2026-09-26 window (24 / 6 / 1).
ROOT_CAUSE_STALL = "upstream_stall_after_content"
ROOT_CAUSE_RESTART = "mode_switch_restart"
ROOT_CAUSE_REMOTE = "remote_chain_exhaustion"
ROOT_CAUSE_OTHER = "other"

# Enriched error type that maps to :data:`ROOT_CAUSE_STALL` (the proxy sets
# ``error_type=stall_after_content`` on the synthetic finish event).
ERROR_TYPE_STALL_AFTER_CONTENT = "stall_after_content"

# A restart / mode-switch marker within this many seconds of a local stream
# error classifies it as a restart kill. Evidence from the 2026-09-25 →
# 2026-09-26 window: a 17:27:58 restart preceded 2 kills at 17:28:00; a
# 14:10:19 restart preceded 1 kill; the 10:00 scheduled switch preceded 3
# kills at 10:00:36-37. The explicit restart marker is logged immediately
# before the kill and the ``restart_services`` completion marker immediately
# after, so a symmetric window is used to tolerate log-flush ordering; 120s
# is far wider than the observed ≤3s gap yet tight enough to avoid
# correlating an unrelated local error with a distant restart.
RESTART_CORRELATION_WINDOW_SECONDS = 120

# Error kinds that are stream-level failures and therefore carry a root
# cause. Other error kinds (slot_save, backend_retry, upstream HTTP) are
# infrastructure/informational signals and leave ``root_cause`` unset.
STREAM_ERROR_KINDS = ("stream_finish_error", "stream_error")


def _near_restart(ts: datetime, restart_signals: Iterable[datetime]) -> bool:
    """Return True when *ts* is within the restart-correlation window of a
    restart or mode-switch marker."""
    for marker_ts in restart_signals:
        delta = abs((ts - marker_ts).total_seconds())
        if delta <= RESTART_CORRELATION_WINDOW_SECONDS:
            return True
    return False


def classify_root_cause(
    ev: LogEvent, restart_signals: Iterable[datetime]
) -> str | None:
    """Classify a parsed stream error into a deterministic root cause.

    Rules (validated against the 2026-09-25 → 2026-09-26 window):

    - ``upstream_stall_after_content`` — the enriched ``error_type`` is
      ``stall_after_content`` (the upstream stalled *after* content was
      delivered, so the proxy cannot re-route).
    - ``mode_switch_restart`` — a ``local`` stream error correlated with a
      restart / mode-switch marker within
      :data:`RESTART_CORRELATION_WINDOW_SECONDS` (the restart killed the
      in-flight llama-server stream).
    - ``remote_chain_exhaustion`` — any other remote-provider stream error.
    - ``other`` — anything else (including a local error with no restart
      marker); never fabricates a cause from missing evidence.

    Returns ``None`` for non-stream error kinds so the CSV/JSON only carry a
    root cause where it is meaningful.
    """
    if ev.kind not in STREAM_ERROR_KINDS:
        return None
    if ev.error_type == ERROR_TYPE_STALL_AFTER_CONTENT:
        return ROOT_CAUSE_STALL
    if ev.provider == LOCAL_PROVIDER and _near_restart(ev.ts, restart_signals):
        return ROOT_CAUSE_RESTART
    if ev.provider and ev.provider != LOCAL_PROVIDER:
        return ROOT_CAUSE_REMOTE
    return ROOT_CAUSE_OTHER


def _bucket_key(bucket: str | None) -> str:
    return "cheap" if bucket == "cheap" else "fast"


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
    # Provenance of ``fallback_reason`` (LP-0MVMBMSWR0036RES AC5): one of the
    # ``FALLBACK_ATTRIBUTION_*`` states, so a consumer can tell own evidence
    # from the legacy nearest-line heuristic and from a session with none.
    fallback_attribution: str | None = None
    # Per-period context window of the profile active for this session (from
    # the mode profile's slot_schedule ctx_size, or its global
    # local_model_ctx_size when the entry did not pin one); None when not
    # derivable.
    ctx_size: int | None = None
    # Decode speed derived from local streams: total local completion tokens /
    # local active span (first→last local stream event). None when not derivable.
    decode_tok_s: float | None = None
    # Token totals from this session's Stream finished events.
    input_tokens: int = 0
    output_tokens: int = 0

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
    # Prefill-ETA admission-guard decisions (LP-0MVGDUV5E005UWBQ): one event per
    # decision taken at the contention-queue point, with ``decision`` = the
    # guard verdict (``admit`` / ``reject``), ``enforced`` = whether it was
    # applied (``false`` in ``observe`` mode), and the projected prefill ETA /
    # wait budget in seconds.
    local_admission_events: list[LogEvent] = field(default_factory=list)
    # Idle-aware local admission (LP-0MVGD9Z8Z0055Y5J): one event per economic
    # cold-cache bypass that was overridden because the GPU had been idle for
    # at least ``threshold_seconds`` (``idle_seconds`` = measured idle age).
    idle_admit_events: list[LogEvent] = field(default_factory=list)
    # Residency-aware admission decisions (LP-0MVN1IQTF0037WTG AC5): one event
    # per decision with the *measured* reusable prefix (``n_past_actual``), the
    # request estimate, the resulting ``reuse_ratio``, the slot owner at
    # decision time and whether the ratio came from the live measurement, from
    # slot ownership, or was unavailable.
    residency_events: list[LogEvent] = field(default_factory=list)
    # Post-acceptance eviction events (LP-0MVN1IQTF0037WTG AC6): a resident
    # session's lease/slot was reclaimed, with the closed ``reason`` vocabulary
    # and whether its KV snapshot was persisted for the returning client.
    eviction_events: list[LogEvent] = field(default_factory=list)
    # Parsed error events (stream errors, slot_save failures, backend_retry
    # timeouts, upstream HTTP errors) inside the window.
    error_events: list[LogEvent] = field(default_factory=list)
    # Compaction events (LP-0MTHCTLAF00147IT): compaction_event, backstop, and
    # churn events observed inside the window.
    compaction_events: list[LogEvent] = field(default_factory=list)
    # llama-server decode/prompt-eval speed stats (set by reporting.run_analysis).
    speed: object | None = None
    # Local-model utilization (busy time etc.); None when no local traffic.
    busy: BusyStats | None = None
    # Per-hour activity counters for the report's ``## Summary by hour``
    # table (LP-0MTYAZFGN003BYQS): absolute top-of-hour datetime -> Counter
    # over ``requests_started`` / ``local_attempts`` / ``local_served`` /
    # ``fallbacks`` / ``local_skip``. Counted by each event's own timestamp
    # rather than by session start hour, so a long-running session
    # contributes to every hour it touches. Includes in-window stream events
    # that carry no session UUID.
    hourly_activity: dict[datetime, Counter] = field(default_factory=dict)
    # TTFT (Time to First Token) events parsed from log lines
    # (LP-0MTSSM5SO003PKU0).
    ttft_events: list[LogEvent] = field(default_factory=list)
    # Hold-cycle routing-skip re-emissions (LP-0MUIEO6IP0030CT3):
    # ``routing_skip`` events with ``chain_hold_cycle > 0`` are re-emissions
    # of a single held request restarted across chain-hold cycles
    # (LP-0MU5AIAAY003KVM0), NOT genuine new client demand. They are kept out
    # of every demand/local-skip aggregate (hourly ``local_skip``,
    # ``routing_skip_events``, per-session ``routing_skips``,
    # ``routing_skip_reason_counts``) and surfaced here separately so a stuck
    # hold is distinguishable from a client retry storm.
    hold_cycle_routing_skip_events: list[LogEvent] = field(default_factory=list)
    # Provider context-ceiling skips (LP-0MVMBM1Y3001TPRT): one event per
    # provider whose ``max_context_tokens`` ceiling the estimated prompt
    # exceeded. These are per-request size mismatches recorded by the proxy
    # *before* a remote attempt, so they explain fallback chains that no
    # session-less ``Fallback triggered`` line can attribute.
    provider_context_skip_events: list[LogEvent] = field(default_factory=list)
    # Restart events (``restart`` kind, LP-0MUQO82LJ008F1FS) observed inside
    # the window, carrying the attribution markers (source/actor/reason,
    # LP-0MVC0S6U5003SEDJ) parsed from the log lines so the report can render
    # restart causes deterministically instead of heuristically
    # (LP-0MVC144KS001PDR4). Includes spawn/complete markers and the
    # start-proxy.sh --restart startup attribution line. Restart events
    # already contribute their timestamp to ``restart_signals`` for
    # root-cause correlation (unchanged).
    restart_events: list[LogEvent] = field(default_factory=list)

    @property
    def restart_events_sorted(self) -> list[LogEvent]:
        """Restart events sorted by timestamp (report rendering order)."""
        return sorted(self.restart_events, key=lambda e: e.ts)

    @property
    def hold_cycle_reemission_count(self) -> int:
        """Number of hold-cycle routing-skip re-emissions
        (``chain_hold_cycle > 0``) in the window."""
        return len(self.hold_cycle_routing_skip_events)

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
    def fallback_attribution_counts(self) -> Counter:
        """Session count per fallback-attribution state (LP-0MVMBMSWR0036RES)."""
        return Counter(
            s.fallback_attribution
            for s in self.sessions.values()
            if s.fallback_attribution
        )

    @property
    def routing_skip_reason_counts(self) -> Counter:
        return Counter(e.reason for e in self.routing_skip_events if e.reason)

    @property
    def provider_context_skip_provider_counts(self) -> Counter:
        """Provider context-ceiling skips grouped by provider (LP-0MVMBM1Y3001TPRT)."""
        return Counter(
            e.provider or "(unknown)" for e in self.provider_context_skip_events
        )

    @property
    def provider_context_skip_limit_counts(self) -> Counter:
        """Provider context-ceiling skips grouped by the ``limit`` they hit.

        An event whose line carried no ``limit`` (older rotated logs, AC8) is
        keyed as ``None`` so the caller can report it as unattributed rather
        than silently dropping it.
        """
        return Counter(e.limit for e in self.provider_context_skip_events)

    @property
    def error_counts(self) -> Counter:
        """Error events grouped by error type (event kind)."""
        return Counter(e.kind for e in self.error_events)

    @property
    def error_provider_model_counts(self) -> Counter:
        """Error events grouped by (error type, provider, model)."""
        return Counter((e.kind, e.provider, e.model) for e in self.error_events)

    @property
    def root_cause_counts(self) -> dict[str, dict[str, int]]:
        """Stream-error root-cause counts keyed by error kind then root cause
        (LP-0MUQO82LJ008F1FS).

        Only error events carrying a root cause (``stream_finish_error`` /
        ``stream_error``) are included, so the breakdown is self-describing
        and remains data-driven.
        """
        out: dict[str, Counter] = {}
        for e in self.error_events:
            if e.root_cause:
                out.setdefault(e.kind, Counter())[e.root_cause] += 1
        return {kind: dict(counts) for kind, counts in out.items()}

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

    @property
    def total_tokens(self) -> tuple[int, int, int]:
        """Total tokens across all sessions: (input, output, total).

        *input* = sum of prompt tokens, *output* = sum of completion tokens,
        *total* = input + output (source-of-truth from ``tokens=prompt/completion/total``
        on ``Stream finished`` lines).
        """
        inp = sum(s.input_tokens for s in self.sessions.values())
        out = sum(s.output_tokens for s in self.sessions.values())
        return inp, out, inp + out

    @property
    def token_buckets(self) -> dict[str, tuple[int, int, int]]:
        """Token totals split by fast/cheap bucket: ``{label: (input, output, total)}``."""
        buckets: dict[str, list[tuple[int, int]]] = {}
        for s in self.sessions.values():
            if s.bucket is None:
                continue
            key = _bucket_key(s.bucket)
            buckets.setdefault(key, []).append((s.input_tokens, s.output_tokens))
        result: dict[str, tuple[int, int, int]] = {}
        for key, pairs in buckets.items():
            inp = sum(p[0] for p in pairs)
            out = sum(p[1] for p in pairs)
            result[key] = (inp, out, inp + out)
        return result

    @property
    def token_by_model(self) -> dict[tuple[str, str], tuple[int, int, int]]:
        """Token totals grouped by ``(provider, model)`` sorted by total descending."""
        totals: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for s in self.sessions.values():
            key = (s.initial_provider or UNKNOWN_LABEL, s.initial_model or UNKNOWN_LABEL)
            totals.setdefault(key, []).append((s.input_tokens, s.output_tokens))
        result: dict[tuple[str, str], tuple[int, int, int]] = {}
        for key, pairs in totals.items():
            inp = sum(p[0] for p in pairs)
            out = sum(p[1] for p in pairs)
            result[key] = (inp, out, inp + out)
        return dict(sorted(result.items(), key=lambda kv: kv[1][2], reverse=True))


def _percentile(sorted_values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile (numpy default) over sorted values.

    Matches the percentile convention used by the rest of the skill
    (``reporting._percentile`` / ``llama_log_parser._percentile``); returns
    ``None`` for an empty sample set and rounds to one decimal place.
    """
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return round(sorted_values[0], 1)
    pos = p / 100.0 * (n - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, n - 1)
    value = sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac
    return round(value, 1)


@dataclass(frozen=True)
class PrefillSample:
    """One local stream's prefill measurement.

    ``seconds`` is ``None`` for an unknown prefill — a stream that produced no
    first-byte line (aborted request, log gap), so ``first_byte_ts`` is then
    ``None`` as well. ``session`` keeps the pairing attributable per stream.
    """

    session: str | None
    first_byte_ts: datetime | None
    seconds: float | None

    @property
    def known(self) -> bool:
        """Whether this stream's prefill duration is observable."""
        return self.seconds is not None


@dataclass(frozen=True)
class DecodeSample:
    """One local stream's decode (first byte → finish) measurement.

    ``seconds`` is ``None`` for an unknown decode — a stream with no logged
    ``Stream finished`` line (aborted or still running), so its decode time is
    unknown rather than zero. ``session`` keeps the pairing attributable per
    stream.
    """

    session: str | None
    seconds: float | None

    @property
    def known(self) -> bool:
        """Whether this stream's decode duration is observable."""
        return self.seconds is not None


@dataclass(frozen=True)
class SlotSecondsStats:
    """Shared slot-seconds statistics for one local phase.

    Every figure here is **slot-time, not wall-clock**: ``total_seconds`` is
    the sum of the per-stream phase durations (the total phase *work*), so two
    concurrent 10 s prefills contribute 20 slot-seconds while occupying only
    10 s of wall clock. Phase totals are therefore deliberately distinct from
    the wall-clock unions on :class:`BusyStats` and must not be compared
    against them.

    ``streams`` holds one sample per local stream (in stream order). Streams
    whose phase duration cannot be observed are **unknown**: excluded from
    ``samples`` and the percentiles, but retained in
    ``streams``/``stream_count`` so the unknown time is visible rather than
    silently dropped.
    """

    streams: tuple[PrefillSample | DecodeSample, ...] = ()
    unit: str = SLOT_SECONDS_UNIT

    @property
    def samples(self) -> tuple[float, ...]:
        """Known per-stream phase durations in seconds (percentile input)."""
        return tuple(s.seconds for s in self.streams if s.seconds is not None)

    @property
    def count(self) -> int:
        """Number of streams with an observable phase duration."""
        return len(self.samples)

    @property
    def unknown_count(self) -> int:
        """Number of streams whose phase duration could not be observed."""
        return sum(1 for stream in self.streams if not stream.known)

    @property
    def stream_count(self) -> int:
        """All local streams considered: known + unknown phase duration."""
        return len(self.streams)

    @property
    def has_samples(self) -> bool:
        """Whether any stream had an observable phase duration."""
        return self.count > 0

    @property
    def total_seconds(self) -> float:
        """Window total in slot-seconds (sum of the per-stream durations).

        This is total phase *work*, not the wall-clock union of the phase
        intervals (see the class docstring).
        """
        return round(sum(self.samples), 1)

    @property
    def p10(self) -> float | None:
        return _percentile(sorted(self.samples), 10)

    @property
    def p50(self) -> float | None:
        return _percentile(sorted(self.samples), 50)

    @property
    def p90(self) -> float | None:
        return _percentile(sorted(self.samples), 90)

    @property
    def max_seconds(self) -> float | None:
        return round(max(self.samples), 1) if self.samples else None


@dataclass(frozen=True)
class PrefillStats(SlotSecondsStats):
    """Per-stream local prefill (dispatch → first byte) statistics.

    See :class:`SlotSecondsStats` for the shared slot-seconds semantics. A
    stream whose first-byte line is missing is **unknown prefill**.
    """

    streams: tuple[PrefillSample, ...] = ()


@dataclass(frozen=True)
class DecodeStats(SlotSecondsStats):
    """Per-stream local decode (first byte → finish) statistics.

    See :class:`SlotSecondsStats` for the shared slot-seconds semantics. A
    stream with no logged ``Stream finished`` line is **unknown decode**.
    """

    streams: tuple[DecodeSample, ...] = ()


@dataclass
class BusyStats:
    """Local-model utilization over the analysis window.

    "Busy" means at least one local slot is occupied **from dispatch to
    finish** (prefill + decode, LP-0MVJN9H2I00116FP): streams are paired per
    session (FIFO across the full event stream, not just the window), each
    pair's interval starts at its back-calculated dispatch time
    (``ttft.ts - ttft_ms/1000``, falling back to ``Stream started`` when the
    stream has no first-byte line) and ends at ``Stream finished``; each pair
    is clipped to ``[window_start, window_end]`` and the clipped intervals are
    merged so overlapping streams do not double-count.

    ``total_compute_seconds`` is the sum of all clipped stream durations
    (slot-seconds, i.e. the integral of concurrency) on those dispatch →
    finish intervals; ``busy_seconds`` is their union. ``avg_concurrency`` =
    total / busy. The decode-only union (``Stream started`` → ``Stream
    finished``) is retained in :attr:`decode_busy_seconds` — the lower bound
    of the busy figure, whose labelled upper bound is :attr:`busy_seconds`
    (see :data:`BUSY_UPPER_BOUND_LABEL` / :data:`BUSY_LOWER_BOUND_LABEL`).
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
    # and to each absolute wall-clock hour of the window (top-of-hour
    # datetime -> seconds, LP-0MTYAZFGN003BYQS). Absolute-hour keying (not
    # hour-of-day) keeps rolling-window buckets distinct: a 24h window that
    # straddles an hour yields two edge partial buckets with different
    # absolute-hour keys, so no busy duration is double-counted and no row
    # can exceed 100%.
    fast_busy_seconds: float
    cheap_busy_seconds: float
    fast_window_seconds: float
    cheap_window_seconds: float
    hourly_busy: list[tuple[datetime, float]]
    # Per-stream local prefill (dispatch → first byte) statistics
    # (LP-0MVJN45LF009CO2V). Populated when the caller supplies the local
    # TTFT events; ``None`` when they were not supplied at all, so a caller
    # that never asked for prefill is distinguishable from one that found no
    # first-byte lines (an empty :class:`PrefillStats`).
    prefill_stats: PrefillStats | None = None
    # Per-stream local decode (first byte → finish) statistics
    # (LP-0MVJNDOUU0059MMO). Populated alongside ``prefill_stats`` so the
    # report can split phase slot time; unlike prefill it needs no TTFT
    # events, so it is present whenever there is local traffic.
    decode_stats: DecodeStats | None = None
    # Decode-only busy union (``Stream started`` → ``Stream finished``),
    # clipped to the window — the lower bound of the busy figure
    # (LP-0MVJN9H2I00116FP). It is what the report called "busy" before the
    # dispatch → finish redefinition and stays visible for comparability.
    decode_busy_seconds: float = 0.0
    # The same decode-only union split across fast/cheap periods.
    decode_fast_busy_seconds: float = 0.0
    decode_cheap_busy_seconds: float = 0.0
    # Number of paired streams whose busy interval was extended backwards to
    # a measured dispatch time; the rest contribute their decode interval
    # only (unknown prefill), which keeps the upper bound honest.
    prefill_extended_streams: int = 0

    @property
    def busy_definition(self) -> str:
        """Marker for the busy definition (``dispatch_to_finish``)."""
        return BUSY_DEFINITION

    @property
    def busy_upper_bound_seconds(self) -> float:
        """Upper bound of the busy figure: the dispatch → finish union."""
        return self.busy_seconds

    @property
    def busy_lower_bound_seconds(self) -> float:
        """Lower bound of the busy figure: the decode-only union."""
        return self.decode_busy_seconds

    @property
    def prefill_busy_seconds(self) -> float:
        """Wall-clock prefill occupancy: upper bound − lower bound.

        This is the extra union time contributed by the measured prefills
        (slot-time already spent before the first byte). It is a wall-clock
        union, not the per-stream slot-seconds sum in
        :class:`PrefillStats.total_seconds`.
        """
        return round(max(0.0, self.busy_seconds - self.decode_busy_seconds), 1)

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
    dispatch_denied: int = 0,
    provider_context_skips: dict[str, list[LogEvent]] | None = None,
) -> tuple[datetime | None, str | None, str]:
    """Find (move_time, reason, attribution) for a session that reached remote.

    Priority — a session's own evidence always wins, and another session's
    reason is never inherited (LP-0MVMBMSWR0036RES):

    1. The session's own ``routing_skip_local`` line at/before the first remote
       stream (unchanged precedence; it names why local was skipped).
    2. The session's own ``routing_skip_provider_context`` line at/before the
       first remote stream (LP-0MVMBM1Y3001TPRT): also session-bearing, and it
       names the provider + ``limit`` whose context ceiling blocked that
       attempt, so it outranks the generic fallback line.
    3. The session's own session-bearing ``Fallback triggered`` line at/before
       the first remote stream (the actual decision line).
    4. The nearest **session-less** ``Fallback triggered`` line within
       ``FALLBACK_ATTRIBUTION_WINDOW_SECONDS`` before the first remote stream —
       only rotated logs predating the session stamp can reach this branch; a
       session-bearing line belongs to the session it names.
    5. ``lease_denied`` when the session was refused local dispatch by another
       session's lease (``local_dispatch_denied`` evidence) — the cause is
       concurrency/lease contention, reported as such instead of inheriting a
       neighbouring fallback's reason.
    6. ``unattributed`` when there is no evidence at all.

    A ``routing_skip_local`` line wins over a provider-context skip because it
    explains why local was not used at all, while provider-context skips are
    downstream per-provider ceiling hits recorded while walking the chain;
    both reasons are reported separately (per-session reason and the
    provider-context breakdown), so nothing is lost by this precedence.

    Returns ``(None, None, attribution)`` when there is no timestamped
    evidence.
    """
    skips = [e for e in routing_skips.get(session_id, []) if e.ts <= first_remote_ts]
    if skips:
        e = max(skips, key=lambda x: x.ts)
        return e.ts, e.reason, FALLBACK_ATTRIBUTION_OWN_SKIP
    if provider_context_skips:
        ctx_skips = [
            e
            for e in provider_context_skips.get(session_id, [])
            if e.ts <= first_remote_ts
        ]
        if ctx_skips:
            e = max(ctx_skips, key=lambda x: x.ts)
            return e.ts, e.reason, FALLBACK_ATTRIBUTION_OWN_PROVIDER_SKIP
    own = [
        e
        for e in fallback_events
        if e.session == session_id and e.ts <= first_remote_ts
    ]
    if own:
        e = max(own, key=lambda x: x.ts)
        return e.ts, e.reason, FALLBACK_ATTRIBUTION_OWN_FALLBACK
    earliest = first_remote_ts - timedelta(seconds=FALLBACK_ATTRIBUTION_WINDOW_SECONDS)
    candidates = [
        e
        for e in fallback_events
        if not e.session and earliest <= e.ts <= first_remote_ts
    ]
    if candidates:
        e = max(candidates, key=lambda x: x.ts)
        return e.ts, e.reason, FALLBACK_ATTRIBUTION_LEGACY_NEAREST
    if dispatch_denied > 0:
        # The proxy logged ``local_dispatch_denied session=<this session>
        # owner=<other session>``: local was refused by a lease held
        # elsewhere, so the move is lease/concurrency contention.
        return None, LEASE_ACTIVE_REASON, FALLBACK_ATTRIBUTION_LEASE_DENIED
    return None, None, FALLBACK_ATTRIBUTION_UNATTRIBUTED


def _build_session(
    builder: _SessionBuilder,
    routing_skips: dict[str, list[LogEvent]],
    fallback_events: list[LogEvent],
    schedule: bucketing.SlotSchedule,
    mode_map: bucketing.ModeScheduleMap | None = None,
    provider_context_skips: dict[str, list[LogEvent]] | None = None,
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
        move_ts, reason, attribution = _attribute_fallback(
            builder.session_id,
            first_remote.ts,
            routing_skips,
            fallback_events,
            dispatch_denied=builder.dispatch_denied,
            provider_context_skips=provider_context_skips,
        )
        if move_ts is None:
            move_ts = first_remote.ts if has_local else None
    else:
        move_ts, reason, attribution = None, None, FALLBACK_ATTRIBUTION_NO_REMOTE

    # Bucket by the operating mode active at the session's first in-window
    # stream (LP-0MSPZUD4G007IYGH): when the logs show mode transitions, the
    # period is the mode's profile period labelled with the mode name; when
    # they do not, the configured ``server.priority_window`` supplies the
    # fast/cheap split (LP-0MVKSXXIR006OBI2), falling back to the legacy
    # slot-count bucketing from the analysis-time schedule.
    period: bucketing.SlotPeriod | None = bucketing.bucket_period(
        start, schedule, mode_map
    )

    local_req = sum(1 for e in started if e.provider == LOCAL_PROVIDER)
    remote_req = len(started) - local_req

    # Token totals for this session.
    input_tokens = sum(prompt_tokens)
    output_tokens = sum(completion_tokens)

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
        fallback_attribution=attribution,
        bucket=period.label if period else None,
        slots=period.slots if period else None,
        ctx_size=period.ctx_size if period else None,
        dispatch_denied=builder.dispatch_denied,
        routing_skips=len(builder.routing_skips),
        decode_tok_s=decode_tok_s,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
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
    """Fast/cheap label for ``dt``.

    Mode-aware when the logs show mode transitions (LP-0MSPZUD4G007IYGH), else
    the configured priority window (LP-0MVKSXXIR006OBI2), else the legacy
    slot-count label from the analysis-time schedule.
    """
    return bucketing.bucket_label(dt, schedule, mode_map)



def _attribute_interval(
    interval_start: datetime,
    interval_end: datetime,
    periods: list[bucketing.SlotPeriod],
    label_at,
    bucket_busy: dict[str, float],
    hourly: dict[datetime, float],
) -> None:
    """Add one merged busy interval's seconds to the fast/cheap and hourly
    buckets it overlaps, splitting at hour and period boundaries.

    ``hourly`` is keyed by the segment's **absolute top-of-hour** datetime
    (not hour-of-day). ``_segment_boundaries`` already splits at hour
    boundaries, so each segment lies inside a single absolute hour
    (LP-0MTYAZFGN003BYQS)."""
    b = _segment_boundaries(interval_start, interval_end, periods)
    for lo, hi in zip(b, b[1:]):
        if hi <= lo:
            continue
        mid = lo + (hi - lo) / 2
        label = label_at(mid)
        bucket_busy[label] = bucket_busy.get(label, 0.0) + (hi - lo).total_seconds()
        hour_key = mid.replace(minute=0, second=0, microsecond=0)
        hourly[hour_key] = hourly.get(hour_key, 0.0) + (hi - lo).total_seconds()


def _prefill_seconds(match: PrefillMatch) -> float | None:
    """Prefill duration (dispatch → first byte) for one paired stream.

    Resolution order, most authoritative first:

    1. ``dispatch_first_byte_ms`` (the paired TTFT event's ``ttft_ms``): the
       dispatch line is written at first byte and carries the total interval
       including the single-slot queue wait.
    2. ``dispatch_to_first_byte_ms + queue_wait_ms``: the logged service/queue
       split of the same interval, summed back to the total when
       ``dispatch_first_byte_ms`` itself is missing from a rotated log.
    3. Pairing the dispatch and first-byte timestamps: ``ttft.ts - stream.ts``.
       Only a **positive** difference is accepted — the proxy logs
       ``Stream started`` at first byte, so a zero/negative delta carries no
       evidence and the stream stays unknown prefill rather than claiming a
       0 s prefill.

    Returns ``None`` when no source yields evidence (unknown prefill).
    """
    ttft = match.ttft
    if ttft is None:
        return None
    if ttft.ttft_ms is not None:
        return max(0.0, ttft.ttft_ms / 1000.0)
    if ttft.dispatch_to_first_byte_ms is not None and ttft.queue_wait_ms is not None:
        return max(0.0, (ttft.dispatch_to_first_byte_ms + ttft.queue_wait_ms) / 1000.0)
    delta = (ttft.ts - match.stream.ts).total_seconds()
    return delta if delta > 0 else None


def _merge_intervals(
    intervals: Iterable[tuple[datetime, datetime]],
) -> list[list[datetime]]:
    """Merge overlapping/touching ``(start, end)`` intervals into a union."""
    ordered = sorted(intervals)
    merged: list[list[datetime]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _compute_prefill_stats(
    matches: Iterable[PrefillMatch],
    window_start: datetime,
    window_end: datetime,
) -> PrefillStats:
    """Per-stream local prefill statistics for the window.

    ``matches`` are the streams paired FIFO to their TTFT events per session
    (the same pairing the busy intervals use), so each stream gets at most
    one prefill interval ``[ttft.ts - ttft_ms/1000, ttft.ts]``. A stream is
    attributed to the window whose bounds contain its first-byte timestamp
    (its ``Stream started`` timestamp when it has no first-byte line, which
    the proxy logs at the same instant); intervals are **not** clipped, so a
    dispatch that started before ``window_start`` keeps its full duration and
    a first byte after ``window_end`` belongs to the next window (no
    double-counting).
    """
    streams: list[PrefillSample] = []
    for match in matches:
        anchor = match.ttft.ts if match.ttft is not None else match.stream.ts
        if not (window_start <= anchor <= window_end):
            continue
        streams.append(
            PrefillSample(
                session=match.stream.session,
                first_byte_ts=match.ttft.ts if match.ttft is not None else None,
                seconds=_prefill_seconds(match),
            )
        )
    return PrefillStats(streams=tuple(streams))


def compute_busy_stats(
    local_events: Iterable[LogEvent],
    window_start: datetime,
    window_end: datetime,
    schedule: bucketing.SlotSchedule,
    mode_map: bucketing.ModeScheduleMap | None = None,
    *,
    ttft_events: Iterable[LogEvent] | None = None,
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
    (LP-0MSPZUD4G007IYGH), falls back to the configured
    ``server.priority_window`` when they do not (LP-0MVKSXXIR006OBI2), and
    otherwise uses the legacy slot schedule, so busy seconds during cheap
    hours count as cheap even when the analysis runs in fast mode. Returns
    ``None`` when there is no local traffic in the window.

    Busy is the union of **dispatch → finish** intervals (prefill + decode,
    LP-0MVJN9H2I00116FP): each paired stream's interval starts at
    ``ttft.ts - ttft_ms/1000`` when a first-byte line is available and at
    ``Stream started`` otherwise, and ends at ``Stream finished``. Busy,
    total compute, peak/average concurrency, the fast/cheap split and the
    hourly split are all computed on those intervals. The decode-only union
    (``Stream started`` → ``Stream finished``) is retained as
    :attr:`BusyStats.decode_busy_seconds`, the labelled lower bound of the
    busy figure; the dispatch → finish union is the labelled upper bound.
    A stream with no first-byte line falls back to its decode interval, so it
    can only ever contribute its lower-bound time — the busy figure is never
    overstated.

    ``ttft_events`` are the local ``dispatch_first_byte_ms`` TTFT events
    (LP-0MVJN45LF009CO2V). When supplied they are paired FIFO to the local
    streams per session and reported as :attr:`BusyStats.prefill_stats`:
    p10/p50/p90/max plus the window total of the prefill interval
    ``[ttft.ts - ttft_ms/1000, ttft.ts]``. Every prefill figure is
    **slot-time, not wall-clock** (the per-stream durations are summed, not
    unioned), unlike ``busy_seconds``. When ``dispatch_first_byte_ms`` is
    absent the duration falls back to the logged
    ``dispatch_to_first_byte_ms + queue_wait_ms`` split and then to pairing the
    dispatch and first-byte timestamps (see :func:`_prefill_seconds`). Streams
    with no first-byte line are counted as unknown prefill: excluded from the
    percentiles, retained in the stream count. ``prefill_stats`` is ``None``
    only when ``ttft_events`` was not supplied at all — in that case every
    busy interval falls back to its decode interval, so ``busy_seconds`` and
    ``decode_busy_seconds`` coincide.

    ``decode_stats`` (:class:`DecodeStats`, LP-0MVJNDOUU0059MMO) reports the
    same paired streams' decode durations (first byte → finish) as
    slot-seconds with p10/p50/p90/max, attributed to the window containing the
    stream's first byte and not clipped (matching the prefill samples), so the
    report can split phase slot time into prefill vs decode. It needs no TTFT
    events, so it is present whenever local traffic exists; streams without a
    logged finish are unknown decode.
    """
    started: dict[str, list[LogEvent]] = {}
    finished: dict[str, list[LogEvent]] = {}
    started_events: list[LogEvent] = []
    for ev in local_events:
        if ev.session is None:
            continue
        if ev.kind == "stream_started":
            started.setdefault(ev.session, []).append(ev)
            started_events.append(ev)
        else:
            finished.setdefault(ev.session, []).append(ev)

    ttft_list = list(ttft_events) if ttft_events is not None else None
    matches = (
        pair_ttft_to_streams(ttft_list, started_events) if ttft_list is not None else []
    )
    # Dispatch start per paired stream: ``ttft.ts - prefill_seconds`` — the
    # prefill interval start from LP-0MVJN9H2I00116FP AC1. Streams with unknown
    # prefill are absent from the map and fall back to their decode interval.
    dispatch_start: dict[int, datetime] = {}
    for match in matches:
        seconds = _prefill_seconds(match)
        if seconds is not None and match.ttft is not None:
            dispatch_start[id(match.stream)] = match.ttft.ts - timedelta(seconds=seconds)

    intervals: list[tuple[datetime, datetime]] = []
    decode_intervals: list[tuple[datetime, datetime]] = []
    decode_samples: list[DecodeSample] = []
    prefill_extended = 0
    unfinished = 0
    pre_window_unfinished = 0
    margin_start = window_start - BUSY_WINDOW_MARGIN
    for session_id, session_starts in started.items():
        ss = sorted(session_starts, key=lambda e: e.ts)
        ff = sorted(finished.get(session_id, []), key=lambda e: e.ts)
        for start_ev, finish_ev in zip(ss, ff):
            # Decode-only interval (``Stream started`` → ``Stream finished``).
            d_lo, d_hi = max(start_ev.ts, window_start), min(finish_ev.ts, window_end)
            if d_hi > d_lo:
                decode_intervals.append((d_lo, d_hi))
            # Decode slot-seconds (LP-0MVJNDOUU0059MMO): the per-stream
            # duration, attributed to the window containing its first byte
            # (``Stream started``) and not clipped — matching the prefill
            # sample attribution, so both phases cover the same streams.
            if window_start <= start_ev.ts <= window_end:
                decode_samples.append(
                    DecodeSample(
                        session=session_id,
                        seconds=max(0.0, (finish_ev.ts - start_ev.ts).total_seconds()),
                    )
                )
            # Busy interval (dispatch → finish): the same pair extended back to
            # its measured dispatch time when a first-byte line exists.
            measured_dispatch = dispatch_start.get(id(start_ev))
            if measured_dispatch is not None:
                busy_start = measured_dispatch
                prefill_extended += 1
            else:
                busy_start = start_ev.ts
            b_lo, b_hi = max(busy_start, window_start), min(finish_ev.ts, window_end)
            if b_hi > b_lo:
                intervals.append((b_lo, b_hi))
        # FIFO pairing: the earliest ``len(ff)`` starts pair with finishes;
        # the remaining starts are unpaired (no logged finish). Only starts
        # within the busy margin before the window (or inside it) count as
        # this window's unfinished; earlier starts are pre-window leftovers;
        # starts after the window end belong to the next window.
        for start_ev in ss[len(ff):]:
            if start_ev.ts < margin_start:
                pre_window_unfinished += 1
            elif start_ev.ts <= window_end:
                unfinished += 1
                # No finish line: the decode duration is unknown, not zero.
                if window_start <= start_ev.ts:
                    decode_samples.append(DecodeSample(session=session_id, seconds=None))
    prefill_stats = (
        _compute_prefill_stats(matches, window_start, window_end)
        if ttft_list is not None
        else None
    )
    if not intervals:
        # No paired stream duration. Prefill evidence (a known sample or an
        # unknown-prefill stream) can still exist when every stream is
        # unfinished, so only report "no local traffic" when there is none.
        if prefill_stats is None or prefill_stats.stream_count == 0:
            return None

    merged = _merge_intervals(intervals)
    decode_merged = _merge_intervals(decode_intervals)

    busy_seconds = sum((e - s).total_seconds() for s, e in merged)
    decode_busy = sum((e - s).total_seconds() for s, e in decode_merged)
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

    periods = bucketing.bucket_periods(schedule, mode_map)

    def label_at(mid: datetime) -> str:
        return _period_label_at(mid, schedule, mode_map)

    bucket_busy: dict[str, float] = {}
    hourly: dict[datetime, float] = {}
    for s, e in merged:
        _attribute_interval(s, e, periods, label_at, bucket_busy, hourly)
    # The retained decode-only lower bound is bucketed the same way (its
    # hourly split is not reported, so the throwaway dict is intentional).
    decode_bucket_busy: dict[str, float] = {}
    for s, e in decode_merged:
        _attribute_interval(s, e, periods, label_at, decode_bucket_busy, {})

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
        prefill_stats=prefill_stats,
        decode_stats=DecodeStats(streams=tuple(decode_samples)),
        decode_busy_seconds=round(decode_busy, 1),
        decode_fast_busy_seconds=round(decode_bucket_busy.get("fast", 0.0), 1),
        decode_cheap_busy_seconds=round(decode_bucket_busy.get("cheap", 0.0), 1),
        prefill_extended_streams=prefill_extended,
    )


def compaction_impact(
    sessions: dict[str, SessionStats],
    compaction_events: list[LogEvent],
    config: dict | None = None,
) -> dict:
    """Impact analysis for compacted sessions (LP-0MTHCTLAF00147IT AC3).

    Correlates each compacted session with its subsequent in-window
    routing/fallback outcome (stayed local vs fell back) and estimates
    fallback avoidance: sessions where ``pre_tokens`` met the effective
    large-context threshold but ``post_tokens`` dropped below it would have
    been a large-context bypass without compaction.

    ``config`` supplies the effective large-context thresholds (clamped to
    the per-slot context, matching the routing machinery's
    ``_effective_large_context_thresholds``). When ``config`` is ``None``
    the avoided-fallback heuristic is skipped.

    The compaction event's own ``dry_run`` flag distinguishes "would-have-
    avoided" (advisory) from "avoided" (live). The threshold clamp is
    approximated here from ``local_model_ctx_size / slots`` minus output
    headroom (4096) and the warm-threshold hard-cap, matching
    ``_effective_large_context_thresholds``. When the config clamp cannot be
    computed the raw configured thresholds are compared directly.
    """
    from collections import Counter as _Counter

    effective_warm: int | None = None
    effective_cold: int | None = None
    if config:
        try:
            ctx = int(config.get("local_model_ctx_size") or 0)
            slots = int(config.get("session_slot_pool_size") or 0)
            # Prefer the routing machinery's helper when available; fall back to
            # the same arithmetic so tests (which lack proxy/ on sys.path) still
            # exercise the clamp.
            try:
                from proxy.provider import effective_per_slot_threshold as _cap_fn  # type: ignore[import]

                cap = _cap_fn(ctx, slots)
            except Exception:
                per_slot = (ctx // slots) if slots else 0
                cap = (per_slot - 4096) if per_slot > 4096 else 0
            cold_cfg = int(config.get("local_large_context_cold_cache_threshold") or 0) or None
            warm_cfg = int(config.get("local_large_context_warm_cache_threshold") or 0) or None
            effective_warm = min(warm_cfg, cap) if warm_cfg and cap else (warm_cfg or cap or None)
            effective_cold = cold_cfg  # cold threshold clamp is NOT applied (it is economic-only).
        except Exception:
            effective_warm = None
            effective_cold = None

    sessions_with_compaction: set[str] = {e.session for e in compaction_events if e.kind == "compaction_event" and e.session}
    by_action: _Counter = _Counter()
    by_reason: _Counter = _Counter()
    dry_live: _Counter = _Counter()
    bucket_counts: _Counter = _Counter()
    bucket_dry_live: dict[str, _Counter] = {}
    stayed_local = 0
    fell_back = 0
    would_have_avoided = 0  # dry_run=true, pre above threshold, post below.
    would_still_fallback = 0  # dry_run=true, pre and post both above.
    avoided = 0  # live, pre above / post below (sparse until live rollout).
    backstop_sessions: set[str] = set()

    for ev in compaction_events:
        if ev.kind == "compaction_event":
            if ev.action:
                by_action[ev.action] += 1
            if ev.reason:
                by_reason[ev.reason] += 1
            lab = "dry_run" if ev.dry_run is True else ("live" if ev.dry_run is False else "unknown")
            dry_live[lab] += 1
            if ev.mode:
                bucket_counts[ev.mode] += 1
                bucket_dry_live.setdefault(ev.mode, _Counter())[lab] += 1
            if ev.action in ("backstop_dropped", "backstop_exhausted"):
                if ev.session:
                    backstop_sessions.add(ev.session)
            # Fallback-avoidance heuristic: pre above threshold -> would bypass, post below -> fit.
            if ev.pre_tokens is not None and ev.post_tokens is not None and effective_warm:
                pre_hit = ev.pre_tokens >= effective_warm
                post_fit = ev.post_tokens < effective_warm
                if pre_hit and post_fit:
                    if ev.dry_run is True:
                        would_have_avoided += 1
                    elif ev.dry_run is False:
                        avoided += 1
                elif pre_hit and not post_fit and ev.dry_run is True:
                    would_still_fallback += 1
        elif ev.kind == "compaction_backstop":
            # Backstop is a supplementary machine line; also counts as backstop.* action.
            if ev.action:
                by_action[ev.action] += 1
            dry_live["live"] += 1

    # Session-level routing correlation uses the aggregated session stats.
    for sid in sessions_with_compaction:
        sess = sessions.get(sid)
        if sess is None:
            continue
        if sess.fell_back:
            fell_back += 1
        else:
            stayed_local += 1

    return {
        "sessions_with_compaction": len(sessions_with_compaction),
        "by_action": dict(by_action),
        "by_reason": dict(by_reason),
        "dry_run": {"dry_run": dry_live.get("dry_run", 0), "live": dry_live.get("live", 0), "unknown": dry_live.get("unknown", 0)},
        "bucket_counts": dict(bucket_counts),
        "bucket_dry_live": {k: dict(v) for k, v in bucket_dry_live.items()},
        "stayed_local": stayed_local,
        "fell_back": fell_back,
        "would_have_avoided": would_have_avoided,
        "would_still_fallback": would_still_fallback,
        "avoided": avoided,
        "effective_warm": effective_warm,
        "effective_cold": effective_cold,
        "backstop_sessions": len(backstop_sessions),
    }


@dataclass(frozen=True)
class HourAvailability:
    """One hour's local-model availability classification
    (LP-0MVKUKG3F0035I85).

    ``states`` is a **non-exclusive, canonically ordered** tuple of
    :data:`AVAILABILITY_ORDER` members (or ``(AVAILABILITY_IDLE,)``): a wedged
    hour is simultaneously ``prefill-only-no-decode`` (dispatch attempts with
    no completed local stream) and ``lease-blocked`` (demand refused by the
    foreign lease), and a busy hour can be ``serving`` while still refusing
    other sessions. ``label`` is the render/JSON form: ``+``-joined states.

    The counters are the raw evidence, kept so the classification is
    auditable from the report and the ``--json`` summary.
    """

    hour: datetime
    states: tuple[str, ...]
    local_dispatch_attempts: int = 0
    local_served: int = 0
    local_denials: int = 0
    lease_active_fallbacks: int = 0

    @property
    def label(self) -> str:
        return "+".join(self.states)

    @property
    def is_idle(self) -> bool:
        return self.states == (AVAILABILITY_IDLE,)


def classify_hour(
    hour: datetime,
    *,
    local_dispatch_attempts: int = 0,
    local_served: int = 0,
    local_denials: int = 0,
    lease_active_fallbacks: int = 0,
) -> HourAvailability:
    """Classify one hour from timestamped proxy-log evidence.

    Deterministic and priority-free (states accumulate instead of competing):

    - ``serving`` — at least one local ``Stream finished``: decode happened.
    - ``prefill-only-no-decode`` — local dispatch was attempted
      (``[local] POST`` to llama-server) but no local stream completed, i.e.
      prompt work with no decode. Suppressed when ``serving`` — a completed
      stream already proves decode.
    - ``lease-blocked`` — local demand was refused because another session
      held the dispatch lease (``local_dispatch_denied`` or a
      ``reason=local_lease_active`` fallback).
    - ``idle`` — no dispatch attempt, no refusal and no completed local
      stream: genuinely no local demand (never a false positive for a wedged
      or refused hour).

    Only the log evidence decides; nothing is inferred from wall-clock level
    or from the absence of config. Prefill evidence comes from the proxy's own
    dispatch-attempt marker because llama-server eval lines carry no
    timestamps (see ``REFERENCE.md`` → Limitations).
    """
    present: set[str] = set()
    if local_served > 0:
        present.add(AVAILABILITY_SERVING)
    elif local_dispatch_attempts > 0:
        present.add(AVAILABILITY_PREFILL_ONLY_NO_DECODE)
    if local_denials > 0 or lease_active_fallbacks > 0:
        present.add(AVAILABILITY_LEASE_BLOCKED)
    # Canonical order (:data:`AVAILABILITY_ORDER`), independent of the
    # accumulation order above.
    states = tuple(s for s in AVAILABILITY_ORDER if s in present)
    if not states:
        states = (AVAILABILITY_IDLE,)
    return HourAvailability(
        hour=hour,
        states=tuple(states),
        local_dispatch_attempts=local_dispatch_attempts,
        local_served=local_served,
        local_denials=local_denials,
        lease_active_fallbacks=lease_active_fallbacks,
    )


def hour_availability(hour: datetime, counts: Counter) -> HourAvailability:
    """Classify one hourly-activity counter bucket
    (``AnalysisResult.hourly_activity``)."""
    return classify_hour(
        hour,
        local_dispatch_attempts=counts.get("local_dispatch_attempts", 0),
        local_served=counts.get("local_served", 0),
        local_denials=counts.get("local_denials", 0),
        lease_active_fallbacks=counts.get("lease_active_fallbacks", 0),
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
    # Provider context-ceiling skips indexed by session (LP-0MVMBM1Y3001TPRT):
    # per-session attribution input, see ``_attribute_fallback``.
    provider_context_skips: dict[str, list[LogEvent]] = {}
    fallback_events: list[LogEvent] = []
    provider_context_skip_events: list[LogEvent] = []
    routing_skip_events: list[LogEvent] = []
    dispatch_denied_events: list[LogEvent] = []
    error_events: list[LogEvent] = []
    contention_dispatch_events: list[LogEvent] = []
    contention_fallback_events: list[LogEvent] = []
    local_admission_events: list[LogEvent] = []
    idle_admit_events: list[LogEvent] = []
    residency_events: list[LogEvent] = []
    eviction_events: list[LogEvent] = []
    compaction_events: list[LogEvent] = []
    # TTFT events (LP-0MTSSM5SO003PKU0)
    ttft_events: list[LogEvent] = []
    # Hold-cycle routing-skip re-emissions (LP-0MUIEO6IP0030CT3):
    # ``chain_hold_cycle > 0`` routing_skip events, recorded separately from
    # the demand aggregates.
    hold_cycle_routing_skip_events: list[LogEvent] = []
    # Per-hour activity counters (LP-0MTYAZFGN003BYQS): absolute top-of-hour
    # datetime -> Counter, counted by each event's own timestamp.
    hourly_activity: dict[datetime, Counter] = {}
    dispatch_denied = 0
    unattributed = 0
    lines_skipped = 0
    total_lines = 0
    # Local stream events across the (margin-widened) event stream for the
    # busy-time calculation; ``iter_events`` yields a margin beyond the window
    # so boundary-crossing streams pair correctly (clipped in compute_busy_stats).
    local_stream_events: list[LogEvent] = []
    # Local TTFT events (``dispatch_first_byte_ms`` lines) across the same
    # margin-widened stream: paired FIFO to the local streams to derive the
    # per-stream prefill statistics (LP-0MVJN45LF009CO2V). The dispatch line
    # carries no ``provider=`` field, so ``provider is None`` marks it as local
    # (remote ``ttft_seconds=`` lines always name their provider).
    local_ttft_events: list[LogEvent] = []
    # Operating-mode transitions across the (margin-widened) event stream:
    # the mode timeline used for fast/cheap session bucketing. Carries the
    # attribution markers (source/actor/reason) parsed by log_parser
    # (LP-0MVC0S6U5003SEDJ) so the report can render who/what requested the
    # switch (LP-0MVC144KS001PDR4).
    mode_transitions: list[ModeTransition] = []
    # Restart / mode-switch marker timestamps across the (margin-widened)
    # event stream (LP-0MUQO82LJ008F1FS): used to correlate local stream
    # errors with the restart that killed them. Collected before the window
    # filter so an error near the window edge still finds its marker.
    restart_signals: list[datetime] = []
    # In-window restart events carrying attribution (LP-0MVC144KS001PDR4),
    # collected alongside the restart-signal markers above.
    restart_events: list[LogEvent] = []

    for ev in events:
        if ev.kind == "mode_switch":
            if ev.mode in ("fast", "cheap"):
                mode_transitions.append(
                    ModeTransition(
                        ts=ev.ts,
                        mode=ev.mode,
                        source=ev.source,
                        actor=ev.actor,
                        reason=ev.reason,
                    )
                )
            restart_signals.append(ev.ts)
            if window_start <= ev.ts <= window_end:
                total_lines += 1
            continue
        if ev.kind == "restart":
            restart_signals.append(ev.ts)
            if window_start <= ev.ts <= window_end:
                total_lines += 1
                # Surface attribution (LP-0MVC144KS001PDR4): keep the restart
                # event (with its source/actor/reason markers) for the report.
                restart_events.append(ev)
            continue
        if ev.provider == LOCAL_PROVIDER and ev.kind in ("stream_started", "stream_finished"):
            local_stream_events.append(ev)
        if ev.kind == "ttft" and ev.provider in (None, LOCAL_PROVIDER):
            local_ttft_events.append(ev)
        if not (window_start <= ev.ts <= window_end):
            continue
        total_lines += 1
        # Per-hour activity counters (LP-0MTYAZFGN003BYQS): count by the
        # event's own timestamp, so a long-running session contributes to
        # every hour it touches rather than only its first-request hour.
        hour_key = ev.ts.replace(minute=0, second=0, microsecond=0)
        if ev.kind == "stream_started":
            counts = hourly_activity.setdefault(hour_key, Counter())
            counts["requests_started"] += 1
            if ev.provider == LOCAL_PROVIDER:
                counts["local_attempts"] += 1
        elif ev.kind == "stream_finished" and ev.provider == LOCAL_PROVIDER:
            hourly_activity.setdefault(hour_key, Counter())["local_served"] += 1
        elif ev.kind == "local_dispatch":
            # A ``[local] POST`` dispatch attempt (LP-0MVKUKG3F0035I85): counted
            # regardless of whether the attempt ever produced a first byte or a
            # ``Stream finished``, so wedged/prefill-only hours stay visible.
            hourly_activity.setdefault(hour_key, Counter())["local_dispatch_attempts"] += 1
        elif ev.kind == "dispatch_denied":
            hourly_activity.setdefault(hour_key, Counter())["local_denials"] += 1
        elif ev.kind == "fallback":
            counts = hourly_activity.setdefault(hour_key, Counter())
            counts["fallbacks"] += 1
            if ev.reason == LEASE_ACTIVE_REASON:
                counts["lease_active_fallbacks"] += 1
        elif ev.kind == "contention_fallback_after_queue" and ev.reason == LEASE_ACTIVE_REASON:
            # Contention-queue fallback refused by the local lease
            # (``queued_duration=60s reason=local_lease_active``), a second
            # ``local_lease_active`` refusal shape (LP-0MVKUKG3F0035I85).
            hourly_activity.setdefault(hour_key, Counter())["lease_active_fallbacks"] += 1
        elif ev.kind == "routing_skip":
            # Demand counter for genuine skips only: a ``chain_hold_cycle > 0``
            # event is a hold-cycle re-emission, not new client demand
            # (LP-0MUIEO6IP0030CT3).
            if ev.chain_hold_cycle is None or ev.chain_hold_cycle <= 0:
                hourly_activity.setdefault(hour_key, Counter())["local_skip"] += 1
        if ev.kind in (
            "stream_error",
            "stream_finish_error",
            "slot_save_error",
            "backend_retry",
            "upstream_http_error",
        ):
            error_events.append(ev)
            continue
        if ev.kind == "local_dispatch":
            # Dispatch-attempt marker: consumed by the hourly counter above;
            # nothing else aggregates it (no session is attributed to these
            # lines — the echoed request body may itself contain ``session=``
            # text, which is why the parser extracts no fields).
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
            if ev.chain_hold_cycle is not None and ev.chain_hold_cycle > 0:
                # Hold-cycle re-emission: keep out of every demand aggregate
                # (routing_skip_events, per-session routing_skips,
                # routing_skip_reason_counts) and record separately.
                hold_cycle_routing_skip_events.append(ev)
            else:
                routing_skip_events.append(ev)
                if ev.session:
                    routing_skips.setdefault(ev.session, []).append(ev)
                    builders.setdefault(ev.session, _SessionBuilder(ev.session)).add(ev)
            continue
        if ev.kind == "provider_context_skip":
            # Provider context-ceiling skip (LP-0MVMBM1Y3001TPRT): a skip, not a
            # stream/fallback event — it never contributes to request, token or
            # fallback-event counts. It is collected globally for the report's
            # provider-context breakdown and indexed by session so
            # ``_attribute_fallback`` can prefer this session's own line.
            provider_context_skip_events.append(ev)
            if ev.session:
                provider_context_skips.setdefault(ev.session, []).append(ev)
                builders.setdefault(ev.session, _SessionBuilder(ev.session)).add(ev)
            continue
        if ev.kind == "contention_dispatch":
            contention_dispatch_events.append(ev)
            continue
        if ev.kind == "contention_fallback_after_queue":
            contention_fallback_events.append(ev)
            continue
        if ev.kind == "local_admission_guard":
            local_admission_events.append(ev)
            continue
        if ev.kind == "idle_admit":
            idle_admit_events.append(ev)
            continue
        if ev.kind == "residency_decision":
            residency_events.append(ev)
            continue
        if ev.kind == "local_eviction":
            eviction_events.append(ev)
            continue
        if ev.kind in ("compaction_event", "compaction_backstop", "compaction_churn"):
            compaction_events.append(ev)
            continue
        if ev.kind == "ttft":
            ttft_events.append(ev)
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

    # Root-cause classification (LP-0MUQO82LJ008F1FS): every stream-level
    # error is attributed to a deterministic cause using the restart / mode
    # markers collected above. Non-stream errors keep ``root_cause=None``.
    for ev in error_events:
        ev.root_cause = classify_root_cause(ev, restart_signals)

    sessions: dict[str, SessionStats] = {}
    for sid, builder in builders.items():
        # Require at least one in-window stream *started* event (events have
        # already been window-filtered above).
        if not builder.started:
            continue
        sessions[sid] = _build_session(
            builder,
            routing_skips,
            fallback_events,
            schedule,
            mode_map,
            provider_context_skips,
        )

    busy = compute_busy_stats(
        local_stream_events,
        window_start,
        window_end,
        schedule,
        mode_map,
        ttft_events=local_ttft_events,
    )

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
        local_admission_events=local_admission_events,
        idle_admit_events=idle_admit_events,
        residency_events=residency_events,
        eviction_events=eviction_events,
        compaction_events=compaction_events,
        busy=busy,
        hourly_activity=hourly_activity,
        ttft_events=ttft_events,
        hold_cycle_routing_skip_events=hold_cycle_routing_skip_events,
        provider_context_skip_events=provider_context_skip_events,
        restart_events=restart_events,
    )
