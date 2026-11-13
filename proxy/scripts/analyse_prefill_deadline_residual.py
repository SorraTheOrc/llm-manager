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

``llama-server.log`` is used for two optional surfaces, both valid only when the
file covers exactly the analysed window (its lines carry no wall-clock
timestamp): the release→backend-cancel joiner
(:func:`release_cancel_accounting`), which attributes every proxy release of an
accepted local dispatch to a backend task lifecycle segment, and the window
accounting cross-check in :func:`backend_prefill_accounting`, which compares the
number of completed prefills in ``llama-server`` with the number of local
streams the proxy actually consumed. The joiner's seconds basis
(``seconds_basis_tok_s``) is derived **only from Qwen3-attributed prompt-eval
lines**: a two-server glob that also carries the embeddings child's ~100 tok/s
lines would otherwise pollute the median that prices cancel→stop latency
(LP-0MW957SKX0079QKC; see :func:`_attribute_prompt_eval_rates`). When a
llama-server log is supplied the abandoned-prefill windows are bounded by the
**observed** backend stop
(:func:`abandoned_prefill_windows`); without one they degrade to the legacy
token/rate estimate, and the report says so (`measured_available=False`).

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

    # release→cancel accounting + corrected window model: point at the
    # llama-server log(s) covering exactly the same window (comma-separated
    # paths/globs); every release is then joined to a backend task lifecycle
    python3 proxy/scripts/analyse_prefill_deadline_residual.py \\
        --start ... --end ... \\
        --llama-log /var/log/llama-proxy/llama-server.log-2026-10-30

The JSON payload is the recorded dataset; the prose report is written from it
(``docs/dev/prefill-deadline-residual.md``) so the ACs stay reproducible after
the log files rotate away.

Analysis only: nothing here reads or changes proxy configuration, so it cannot
alter an admission decision or a deadline.
"""

from __future__ import annotations

import argparse
import glob
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
#:
#: **Retained only as the legacy comparison basis** (LP-0MVW8S8ZK001ZZRV). It
#: prices a whole remaining prompt, which the measurement refuted: a released
#: prefill is cancelled at the backend's next socket write, so the backlog is
#: the in-flight `ubatch` (p90 cancel→stop 6.6 s at the measured 155.51 tok/s
#: on the 2026-10-29 window), not `tokens / 43.1` seconds — two to three orders
#: of magnitude smaller. The model of record is ``basis="measured_stop"`` in
#: :func:`abandoned_prefill_windows` (observed stop latency and tokens actually
#: evaluated after the cancel); this constant still drives
#: ``basis="legacy_rate_estimate"`` rows and ``legacy_duration_s`` so the
#: published legacy figures stay reproducible and comparable.
#: See ``docs/dev/abandoned-prefill-release-cancel.md``.
ABANDONED_PREFILL_RATE_TOK_S = 43.1

#: Slack allowed between an abort and a following dispatch that is treated as
#: the same retry burst (the router retries immediately after the 503).
BACKLOG_START_SLACK_SECONDS = 5.0

#: How far back a same-prompt measurement is borrowed as cluster evidence.
CLUSTER_HORIZON_SECONDS = 900.0

LINE_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
FIELD = re.compile(r"([A-Za-z_]+)=(\S+)")

#: ``llama-server.log`` task-lifecycle grammar (the log carries no wall-clock
#: timestamp; only the order of these lines is available).
LLAMA_LAUNCH = re.compile(r"launch_slot_: id\s+(\d+)\s*\|\s*task\s+(-?\d+)\s*\|")
LLAMA_PROGRESS = re.compile(
    r"update_slots: id\s+(\d+)\s*\|\s*task\s+(-?\d+)\s*\|\s*"
    r"prompt processing progress, n_tokens = (\d+)"
)
LLAMA_TASK_CANCEL = re.compile(r"stop: cancel task, id_task = (-?\d+)")
LLAMA_STOP = re.compile(
    r"release: id\s+(\d+)\s*\|\s*task\s+(-?\d+)\s*\|\s*"
    r"stop processing: n_tokens = (\d+)"
)
LLAMA_OPERATOR_CANCEL = re.compile(
    r"operator\(\): http client error: Connection handling canceled"
)
#: A ``prompt eval time`` line's rate in tokens/s. Kept unanchored so the
#: previous matching breadth (including a ``slot print_timing:`` prefix) is
#: preserved — the attribution below only changes *which* lines contribute.
LLAMA_EVAL_TIME = re.compile(
    r"prompt eval time\s*=.*?([\d.]+)\s*tokens per second"
)
#: The ``[<port>]`` child prefix a router-mode line starts with. The prefix is
#: the backend instance's listening port (see ``llama_log_parser.py``); a bare
#: ``slot print_timing:`` line has no port.
LLAMA_EVAL_PORT = re.compile(r"^\s*\[(\d+)\]\s*")
#: Router parent line announcing a child instance and its port:
#: ``spawning server instance with name=Qwen3 on port 32999``. Mirrors
#: ``llama_log_parser.SPAWN_MODEL_PORT_RE`` so the two tools agree on how a
#: model is bound to a port.
LLAMA_SPAWN_MODEL_PORT = re.compile(r"name=([A-Za-z0-9._-]+) on port (\d+)")

#: The local model the seconds basis must be derived from. The embeddings child
#: (``name=mxbai-embed …``) also emits ``prompt eval time`` lines at roughly half
#: Qwen3's rate; mixing them pollutes the median (LP-0MW957SKX0079QKC).
LLAMA_QWEN3_MODEL = "Qwen3"

#: How a file's Qwen3 port was discovered (matches
#: ``llama_log_parser._resolve_port``).
LLAMA_PORT_SOURCE_SPAWN = "spawn_line"
LLAMA_PORT_SOURCE_INFERRED = "inferred"
LLAMA_PORT_SOURCE_NONE = "none"

#: Machine-readable ``unjoined_reason`` vocabulary for the release→cancel joiner.
UNJOINED_NO_LLAMA_LOG = "no_llama_log"
UNJOINED_NO_BACKEND_SEGMENT = "no_backend_segment"
UNJOINED_CANCEL_WITHOUT_TASK_ID = "cancel_without_task_id"

#: Coverage caveat for every llama-server-derived surface (the log has no clock).
LLAMA_COVERAGE_NOTE = (
    "Valid only when the llama-server log covers exactly the analysed window; "
    "llama-server lines carry no wall-clock timestamp, so releases are joined "
    "to backend task segments in log order (a proxy line's explicit "
    "llama_task_id/slot_id takes precedence when present)."
)

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
    llama_task_id: int | None = None
    slot_id: int | None = None

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
    #: LP-0MW8VP0XR007UGQF: backend slots the measured signals reported as
    #: occupied (union of ``/slots`` ``is_processing`` and fresh log-derived
    #: progress). ``None`` in a window captured before the field existed.
    measured_in_flight_requests: int | None = None


@dataclass
class LlamaTaskLifecycle:
    """One backend task segment parsed from ``llama-server.log`` (log order).

    A segment begins at its ``launch_slot_: … processing task`` line and
    accumulates the ``prompt processing progress`` batches, any
    ``stop: cancel task`` line, the ``prompt eval time`` line and the terminal
    ``release: … stop processing`` line. Bare ``operator(): … Connection
    handling canceled`` lines are *not* segments (they carry no task id); they
    are tracked separately on :class:`LlamaLifecycle` so a release paired with
    one is reported as an unjoined-with-cancel, never as a joined task.
    """

    task_id: int | None = None
    slot_id: int | None = None
    launch_index: int = -1
    line_index: int = -1
    progress: list[int] = field(default_factory=list)
    cancel_seen: bool = False
    cancel_after_n_tokens: int = 0
    stop_seen: bool = False
    stop_n_tokens: int | None = None
    eval_time_seen: bool = False
    eval_time_after_cancel: bool = False

    @property
    def progress_after_cancel(self) -> list[int]:
        """Progress lines recorded after the task's cancel line."""
        if not self.cancel_seen:
            return []
        return [n for n in self.progress if n > self.cancel_after_n_tokens]

    @property
    def tokens_after_cancel(self) -> int:
        """Tokens evaluated after the cancel, from the progress lines only."""
        if not self.cancel_seen or not self.progress:
            return 0
        return max(0, self.progress[-1] - self.cancel_after_n_tokens)


@dataclass
class LlamaLifecycle:
    """Parsed ``llama-server.log``: ordered task segments and the observed rate.

    ``observed_rate_tok_s`` is the median ``… tokens per second`` value from the
    file's **Qwen3-attributed** ``prompt eval time`` lines — the only seconds
    basis the joiner and the measured window model may use (never
    :data:`ABANDONED_PREFILL_RATE_TOK_S`). The Qwen3 port comes from the router's
    ``name=Qwen3 on port <n>`` spawn line, falling back to the single
    otherwise-unannounced eval-line port; the embeddings child's lines are
    excluded (LP-0MW957SKX0079QKC). When no port can be resolved the basis is
    ``None`` rather than a median over mixed models, and
    :attr:`rate_unavailable_reason` says why.

    ``operator_cancels`` holds the line indices of bare ``operator(): …
    Connection handling canceled`` lines — today's cancel shape, which carries no
    task id — so they can be paired (as unjoined-with-cancel) with the releases
    that triggered them.

    The attribution fields are the diagnostic surface AC2/AC3 require: which
    ports were used, which were excluded, how many eval lines contributed, and
    (when the basis is ``None``) the reason.
    """

    segments: list[LlamaTaskLifecycle] = field(default_factory=list)
    operator_cancels: list[int] = field(default_factory=list)
    observed_rate_tok_s: float | None = None
    source: str | None = None
    lines: int = 0
    qwen3_ports: list[int] = field(default_factory=list)
    port_source: str = LLAMA_PORT_SOURCE_NONE
    excluded_ports: list[int] = field(default_factory=list)
    contributing_eval_lines: int = 0
    excluded_eval_lines: int = 0
    eval_lines_total: int = 0
    rate_unavailable_reason: str | None = None


@dataclass
class ReleaseJoin:
    """The outcome of joining one proxy release to the backend log.

    ``segment`` is the :class:`LlamaTaskLifecycle` the release maps to, or
    ``None`` when the release is unjoined (including a release paired with a
    bare ``operator()`` cancel, which has no task lifecycle).
    ``operator_cancel`` records that the release was paired with such a cancel
    line (``cancel_observed=True`` with ``task_id=None``). ``join_basis``
    states how the join was made (AC1: record how each join was made).
    """

    segment: LlamaTaskLifecycle | None = None
    operator_cancel: bool = False
    join_basis: str = "unjoined"


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


def _maybe_int(value: str | None) -> int | None:
    """``int`` or ``None`` — ``unknown``/absent ids must not become ``0``."""
    if value is None or value == "unknown":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
                llama_task_id=_maybe_int(fields.get("llama_task_id")),
                slot_id=_maybe_int(fields.get("slot_id")),
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
                # LP-0MW8VP0XR007UGQF AC2/AC3: the measured backend occupancy is
                # the same value the availability surface publishes, so the
                # analyser no longer under-counts a busy second slot. ``None``
                # for pre-field windows keeps the fallback to the dispatch
                # count (and so the two reports stay comparable).
                measured_in_flight_requests=(
                    _int(fields.get("measured_in_flight_requests"))
                    if fields.get("measured_in_flight_requests")
                    else None
                ),
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
# Release→cancel accounting (backend task lifecycles from llama-server.log)
# ---------------------------------------------------------------------------


def _ports_text(ports) -> str:
    """Render a port list for a diagnostic reason (never an empty string)."""
    listed = ", ".join(str(port) for port in ports)
    return listed or "none"


def _attribute_prompt_eval_rates(lines: list[str]) -> dict:
    """Attribute every ``prompt eval time`` line to a model port (AC1–AC3).

    Rules mirror ``.pi/skills/proxy-usage-analysis/scripts/llama_log_parser.py``:

    1. **Spawn line.** Every ``name=<model> on port <n>`` line binds a port to a
       model. Eval lines whose port was announced for Qwen3 contribute; ports
       announced for another model (the embeddings child) are excluded. A
       restart may announce a new Qwen3 port, so *all* Qwen3-announced ports
       contribute (the reference parser resolves one port per file; the merged
       log is resolved across files here).
    2. **Single unannounced port.** With no Qwen3 spawn line — the rotation split
       case — a single eval-line port not announced for another model is
       inferred as Qwen3.
    3. **Unusable.** Anything else (multiple unannounced ports, only non-Qwen3
       ports, portless lines) yields an empty rate list and a reason: the basis
       is ``None`` rather than a median polluted by another model.

    Returns the attributed ``rates`` plus the diagnostic counts AC2/AC3 require
    (contributing/excluded eval lines, Qwen3 and excluded ports, the reason the
    basis is unavailable).
    """
    announced: dict[int, str] = {}
    eval_port_counts: Counter[int | None] = Counter()
    samples: list[tuple[int | None, float]] = []
    for line in lines:
        if line.lstrip().startswith("#"):
            continue
        spawn = LLAMA_SPAWN_MODEL_PORT.search(line)
        if spawn is not None:
            announced.setdefault(int(spawn.group(2)), spawn.group(1))
        match = LLAMA_EVAL_TIME.search(line)
        if match is None:
            continue
        prefix = LLAMA_EVAL_PORT.match(line)
        port = int(prefix.group(1)) if prefix is not None else None
        rate = float(match.group(1))
        if not _lazy_is_finite(rate):
            continue
        eval_port_counts[port] += 1
        samples.append((port, rate))

    total = len(samples)
    qwen3_announced = sorted(
        port for port, model in announced.items() if model == LLAMA_QWEN3_MODEL
    )
    other_announced = sorted(
        port for port, model in announced.items() if model != LLAMA_QWEN3_MODEL
    )

    attributed_ports: list[int] = []
    port_source = LLAMA_PORT_SOURCE_NONE
    reason: str | None = None
    if qwen3_announced:
        port_source = LLAMA_PORT_SOURCE_SPAWN
        attributed_ports = [port for port in qwen3_announced if eval_port_counts.get(port)]
    else:
        unannounced = sorted(
            port
            for port in eval_port_counts
            if port is not None and port not in other_announced
        )
        if len(unannounced) == 1:
            port_source = LLAMA_PORT_SOURCE_INFERRED
            attributed_ports = unannounced
        elif total == 0:
            reason = "no 'prompt eval time' lines in the llama-server log"
        elif len(unannounced) > 1:
            reason = (
                "no Qwen3 spawn line and eval-timing lines carry multiple "
                f"unannounced ports ({_ports_text(unannounced)}); the Qwen3 "
                "port is ambiguous"
            )
        elif eval_port_counts.get(None):
            reason = (
                "no Qwen3 spawn line and the eval-timing lines carry no "
                "[port] prefix, so the model cannot be attributed"
            )
        else:
            reason = (
                "no Qwen3 spawn line and every eval-timing line belongs to a "
                f"non-Qwen3 child port ({_ports_text(other_announced)})"
            )

    attributed = [rate for port, rate in samples if port in set(attributed_ports)]
    if attributed:
        reason = None
    elif reason is None:
        seen = sorted(port for port in eval_port_counts if port is not None)
        if not total:
            reason = "no 'prompt eval time' lines in the llama-server log"
        else:
            reason = (
                "no 'prompt eval time' line on the announced Qwen3 port(s) "
                f"({_ports_text(qwen3_announced)}); eval ports seen: "
                f"{_ports_text(seen)}"
            )

    excluded_ports = sorted(
        set(other_announced)
        | {
            port
            for port in eval_port_counts
            if port is not None and port not in set(attributed_ports)
        }
    )
    return {
        "rates": attributed,
        "qwen3_ports": attributed_ports or qwen3_announced,
        "port_source": port_source,
        "excluded_ports": excluded_ports,
        "contributing_eval_lines": len(attributed),
        "excluded_eval_lines": total - len(attributed),
        "eval_lines_total": total,
        "reason": reason,
    }


def parse_llama_lifecycle(lines: list[str], source: str | None = None) -> LlamaLifecycle:
    """Parse ``llama-server.log`` *lines* into ordered task segments.

    ``launch_slot_: … processing task`` lines start a new segment (the
    ``task -1`` sampler-chain launches are slot setup and are skipped); a bare
    ``operator(): … Connection handling canceled`` line records an
    operator-cancel event; ``prompt processing progress`` / ``stop: cancel
    task`` / ``release: … stop processing`` / ``prompt eval time`` lines attach
    to the current segment. ``observed_rate_tok_s`` is the median ``… tokens
    per second`` value of the **Qwen3-attributed** eval lines (see
    :func:`_attribute_prompt_eval_rates`) — the only seconds basis the joiner
    and the measured window model may use.
    """
    lifecycle = LlamaLifecycle(source=source, lines=len(lines))
    current: LlamaTaskLifecycle | None = None
    for index, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        launch = LLAMA_LAUNCH.search(line)
        if launch and int(launch.group(2)) != -1:
            current = LlamaTaskLifecycle(
                task_id=int(launch.group(2)),
                slot_id=int(launch.group(1)),
                launch_index=index,
                line_index=index,
            )
            lifecycle.segments.append(current)
            continue
        if LLAMA_OPERATOR_CANCEL.search(line):
            lifecycle.operator_cancels.append(index)
            continue
        if current is None:
            continue
        progress = LLAMA_PROGRESS.search(line)
        if progress:
            current.progress.append(int(progress.group(3)))
            continue
        cancel = LLAMA_TASK_CANCEL.search(line)
        if cancel:
            current.cancel_seen = True
            # The tokens already evaluated at the cancel position: the last
            # progress line before the cancel (the tokens-after-cancel count is
            # ``final n_tokens − this``).
            current.cancel_after_n_tokens = current.progress[-1] if current.progress else 0
            continue
        stop = LLAMA_STOP.search(line)
        if stop:
            current.stop_seen = True
            current.stop_n_tokens = int(stop.group(3))
            continue
        eval_line = LLAMA_EVAL_TIME.search(line)
        if eval_line:
            current.eval_time_seen = True
            if current.cancel_seen:
                current.eval_time_after_cancel = True
            continue
    attribution = _attribute_prompt_eval_rates(lines)
    lifecycle.qwen3_ports = attribution["qwen3_ports"]
    lifecycle.port_source = attribution["port_source"]
    lifecycle.excluded_ports = attribution["excluded_ports"]
    lifecycle.contributing_eval_lines = attribution["contributing_eval_lines"]
    lifecycle.excluded_eval_lines = attribution["excluded_eval_lines"]
    lifecycle.eval_lines_total = attribution["eval_lines_total"]
    lifecycle.rate_unavailable_reason = attribution["reason"]
    rates = sorted(attribution["rates"])
    if rates:
        lifecycle.observed_rate_tok_s = _percentile(rates, 50.0)
    return lifecycle


def _lazy_is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _llama_segment_timeline(llama: LlamaLifecycle) -> list[LlamaTaskLifecycle | None]:
    """One entry per backend lifecycle event in log order; ``None`` = operator cancel.

    Task segments and bare operator-cancel lines are merged by line index so the
    joiner can walk both in the order the backend logged them. ``None`` entries
    keep their own logical position (a release paired with one is
    unjoined-with-cancel).
    """
    entries: list[tuple[int, LlamaTaskLifecycle | None]] = [
        (segment.line_index, segment) for segment in llama.segments
    ]
    entries.extend((index, None) for index in llama.operator_cancels)
    entries.sort(key=lambda item: item[0])
    return [entry for _, entry in entries]


def _pair_releases(
    order: list[EvictionEvent], llama: LlamaLifecycle | None
) -> list[ReleaseJoin]:
    """Join each release to a backend lifecycle entry (AC: nothing is dropped).

    Priority (work item step 2): the release's explicit ``llama_task_id``;
    else its explicit ``slot_id`` matched against unused segments; else the next
    unused lifecycle entry in log order. ``join_basis`` records which rule won;
    a release with no remaining entry stays unjoined
    (:data:`UNJOINED_NO_BACKEND_SEGMENT`), or is paired with the next bare
    ``operator()`` cancel (:data:`UNJOINED_CANCEL_WITHOUT_TASK_ID`).
    """
    joins: list[ReleaseJoin] = [ReleaseJoin() for _ in order]
    if llama is None:
        return joins
    timeline = _llama_segment_timeline(llama)
    used: set[int] = set()

    def _explicit(event: EvictionEvent) -> int | None:
        if event.llama_task_id is not None:
            for index, entry in enumerate(timeline):
                if index not in used and entry is not None and entry.task_id == event.llama_task_id:
                    return index
        if event.slot_id is not None:
            for index, entry in enumerate(timeline):
                if index not in used and entry is not None and entry.slot_id == event.slot_id:
                    return index
        return None

    for i, event in enumerate(order):
        index = _explicit(event)
        if index is None:
            continue
        entry = timeline[index]
        used.add(index)
        basis = "llama_task_id" if event.llama_task_id is not None else "slot_id"
        joins[i] = ReleaseJoin(segment=entry, join_basis=basis)  # type: ignore[arg-type]

    cursor = 0
    for i, event in enumerate(order):
        if joins[i].segment is not None or joins[i].operator_cancel:
            continue
        index = None
        if event.slot_id is not None:
            for candidate in range(cursor, len(timeline)):
                entry = timeline[candidate]
                if candidate in used or entry is None or entry.slot_id != event.slot_id:
                    continue
                index = candidate
                break
        if index is None:
            for candidate in range(cursor, len(timeline)):
                if candidate in used:
                    continue
                index = candidate
                break
        if index is None:
            continue
        entry = timeline[index]
        used.add(index)
        cursor = index + 1
        if entry is None:
            joins[i] = ReleaseJoin(operator_cancel=True, join_basis="log_order")
        else:
            joins[i] = ReleaseJoin(
                segment=entry,
                join_basis="slot_id" if event.slot_id is not None else "log_order",
            )
    return joins


def seconds_basis_payload(llama: LlamaLifecycle | None) -> dict:
    """The seconds-basis provenance block (AC2/AC3).

    Reports the Qwen3-only median rate, the ports it came from, the excluded
    (embeddings) ports, the contributing/excluded eval-line counts and — when
    the rate is ``None`` — the reason it could not be attributed. This is the
    diagnostic surface that shows an operator *why* the basis moved instead of
    silently publishing a median over mixed models.
    """
    if llama is None:
        return {
            "tok_s": None,
            "model": LLAMA_QWEN3_MODEL,
            "qwen3_ports": [],
            "port_source": LLAMA_PORT_SOURCE_NONE,
            "excluded_ports": [],
            "contributing_eval_lines": 0,
            "excluded_eval_lines": 0,
            "eval_lines_total": 0,
            "reason": "no llama-server log supplied (--llama-log)",
        }
    return {
        "tok_s": llama.observed_rate_tok_s,
        "model": LLAMA_QWEN3_MODEL,
        "qwen3_ports": list(llama.qwen3_ports),
        "port_source": llama.port_source,
        "excluded_ports": list(llama.excluded_ports),
        "contributing_eval_lines": llama.contributing_eval_lines,
        "excluded_eval_lines": llama.excluded_eval_lines,
        "eval_lines_total": llama.eval_lines_total,
        "reason": llama.rate_unavailable_reason,
    }


def release_cancel_accounting(
    log: WindowLog,
    llama: LlamaLifecycle | None = None,
) -> dict:
    """Attribute every proxy release of an accepted dispatch to a backend task.

    Given a window's evictions and the (optional) parsed ``llama-server.log``
    covering exactly that window, emit one row per release — joined,
    unjoined-with-reason, or explained-by-another-mechanism — plus the window
    aggregates AC1 requires. Log-only: no in-process counter is consumed.
    """
    order = sorted(log.evictions, key=lambda event: (event.ts, event.session))
    joins = _pair_releases(order, llama)
    rows: list[dict] = []
    cancel_to_stop: list[float] = []
    tokens_total = 0
    observed_cancel = 0
    unjoined_count = 0
    unjoined_reasons: Counter[str] = Counter()
    for event, join in zip(order, joins):
        segment = join.segment
        unjoined = segment is None
        if llama is None:
            unjoined_reason = UNJOINED_NO_LLAMA_LOG
        elif segment is None and join.operator_cancel:
            unjoined_reason = UNJOINED_CANCEL_WITHOUT_TASK_ID
        elif segment is None:
            unjoined_reason = UNJOINED_NO_BACKEND_SEGMENT
        else:
            unjoined_reason = None
        if unjoined:
            unjoined_count += 1
            if unjoined_reason is not None:
                unjoined_reasons[unjoined_reason] += 1
        cancel_observed = bool(
            (segment is not None and segment.cancel_seen) or join.operator_cancel
        )
        if cancel_observed:
            observed_cancel += 1
        tokens_after = segment.tokens_after_cancel if segment is not None else 0
        tokens_total += tokens_after
        if cancel_observed and segment is not None and segment.stop_seen:
            if tokens_after > 0 and llama is not None and llama.observed_rate_tok_s:
                cancel_to_stop_s = round(tokens_after / llama.observed_rate_tok_s, 3)
            else:
                cancel_to_stop_s = 0.0
        else:
            cancel_to_stop_s = None
        if cancel_to_stop_s is not None:
            cancel_to_stop.append(cancel_to_stop_s)
        rows.append(
            {
                "session": event.session,
                "reason": event.reason,
                "released_at": event.ts.isoformat(sep=" ", timespec="seconds"),
                "task_id": segment.task_id if segment is not None else None,
                "slot_id": segment.slot_id if segment is not None else None,
                "join": join.join_basis,
                "unjoined": unjoined,
                "unjoined_reason": unjoined_reason,
                "cancel_observed": cancel_observed,
                "progress_lines_after_cancel": (
                    len(segment.progress_after_cancel) if segment is not None else 0
                ),
                "tokens_evaluated_after_cancel": tokens_after,
                "cancel_to_stop_s": cancel_to_stop_s,
                "prompt_eval_time_emitted": bool(
                    segment is not None and segment.eval_time_seen
                ),
            }
        )
    total = len(order)
    explained = sum(
        1 for join in joins if join.segment is not None and join.segment.stop_seen
    )
    return {
        "releases_total": total,
        "releases_with_observed_cancel": observed_cancel,
        "releases_without_explained_backend_stop": {
            "count": total - explained,
            "share": round((total - explained) / total, 3) if total else 0.0,
        },
        "releases_unjoined": {
            "count": unjoined_count,
            "reasons": dict(unjoined_reasons),
        },
        "cancel_to_stop_seconds": _percentiles(cancel_to_stop),
        "tokens_evaluated_after_cancel": {
            "total": tokens_total,
            "per_release": {
                row["session"]: row["tokens_evaluated_after_cancel"] for row in rows
            },
        },
        "measured_available": any(
            join.segment is not None and join.segment.stop_seen for join in joins
        ),
        "seconds_basis_tok_s": (
            llama.observed_rate_tok_s if llama is not None else None
        ),
        "seconds_basis": seconds_basis_payload(llama),
        "note": LLAMA_COVERAGE_NOTE,
        "releases": rows,
    }


def _llama_log_paths(value: str | None) -> list[Path]:
    """Expand the ``--llama-log`` value: comma-separated paths and/or globs."""
    if not value:
        return []
    paths: list[Path] = []
    seen: set[str] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        expanded = str(Path(part).expanduser())
        matches = sorted(glob.glob(expanded))
        for match in (matches or [expanded]):
            if match in seen:
                continue
            seen.add(match)
            paths.append(Path(match))
    return paths


def _load_llama_lifecycle(value: str | None) -> tuple[LlamaLifecycle | None, str | None]:
    """Read and parse the llama-server log(s) behind ``--llama-log``.

    Returns ``(lifecycle, llama_spec)`` — ``lifecycle`` is ``None`` when no
    file resolved or every file was unreadable; ``llama_spec`` is the original
    CLI value, kept so the report can say which log(s) it was asked to use.
    """
    if not value:
        return None, None
    paths = _llama_log_paths(value)
    if not paths:
        return None, value
    failures: list[str] = []
    lines = read_log_lines(paths, failures)
    if not lines:
        return None, value
    source = ",".join(str(path) for path in paths)
    return parse_llama_lifecycle(lines, source=source), value


def _llama_coverage(
    llama_spec: str | None, llama: LlamaLifecycle | None, log: WindowLog
) -> dict:
    """How well the backend log can cover this window (AC2 coverage caveat)."""
    if llama_spec is None:
        return {
            "available": False,
            "reason": "no_llama_log",
            "note": LLAMA_COVERAGE_NOTE,
        }
    if llama is None:
        return {
            "available": False,
            "reason": "unreadable_or_no_matching_files",
            "llama_log": llama_spec,
            "note": LLAMA_COVERAGE_NOTE,
        }
    entries = len(_llama_segment_timeline(llama))
    releases = len(log.evictions)
    return {
        "available": True,
        "llama_log": llama_spec,
        "entries": entries,
        "releases": releases,
        "verdict": "exact" if entries == releases else "mismatch",
        "note": LLAMA_COVERAGE_NOTE,
    }


def _coverage_window(
    windows, event: EvictionEvent
) -> dict:
    """Whether *event*'s dispatch start falls inside any *windows* (tuple rows)."""
    start = event.dispatch_start
    covering = [
        (end, reason, tokens, session)
        for begin, end, reason, tokens, session in windows
        if begin <= _seconds_after(start, BACKLOG_START_SLACK_SECONDS) and end >= start
    ]
    if not covering:
        return {"overlap": False, "remaining_s": None}
    end, reason, tokens, _session = max(covering, key=lambda item: item[0])
    return {
        "overlap": True,
        "remaining_s": round((end - start).total_seconds(), 1),
        "reason": reason,
        "tokens": tokens,
    }


# ---------------------------------------------------------------------------
# Per-event attribution (AC4)
# ---------------------------------------------------------------------------


def _legacy_priced_tokens(log: WindowLog, event: EvictionEvent) -> int:
    """Tokens the legacy estimator prices into *event*'s abandoned window.

    The measured deficit logged beside a ``measured_cache_deficit`` release is
    preferred (it is the tokens the retry actually had to recompute); otherwise
    the admission-priced ``new_tokens`` is used. Unchanged provenance from the
    pre-correction model, kept so the legacy basis stays comparable.
    """
    tokens = 0
    if event.reason == REASON_MEASURED_CACHE_DEFICIT:
        candidates = log.measurements_for(
            event.session, _seconds_before(event.ts, 5.0), _seconds_after(event.ts, 5.0)
        )
        if candidates:
            tokens = candidates[0].measured_new_tokens
    if tokens <= 0:
        tokens = event.new_tokens
    return tokens


def abandoned_prefill_windows(
    log: WindowLog,
    paired: list[LlamaTaskLifecycle | None] | None = None,
    llama: LlamaLifecycle | None = None,
) -> list[dict]:
    """One window row per released dispatch, bounded by what was **observed**.

    What is measured (``basis="measured_stop"``, ``bounded=True``): when the
    release is joined to a backend task that logged both a
    ``stop: cancel task, id_task = N`` line and the terminal
    ``slot release: … stop processing`` line, the window ends at the observed
    stop. Its length is the tokens evaluated **after** the cancel (last
    ``n_tokens`` minus the ``n_tokens`` at the cancel position, from the
    ``prompt processing progress`` lines, never from a token-rate constant)
    divided by the **observed** prompt-eval rate of the same log — the median
    ``… tokens per second`` value of its ``prompt eval time`` lines. No wall
    clock exists in ``llama-server.log``, so the stop is positioned at
    ``release + measured stop latency``; the window is therefore valid only for
    a log that covers exactly the analysed window (same caveat as
    :func:`backend_prefill_accounting`).

    What is estimated (``basis="legacy_rate_estimate"``, ``bounded=False``):
    the released dispatch's priced tokens divided by
    :data:`ABANDONED_PREFILL_RATE_TOK_S` (43.1 tok/s — F5's replay rate,
    LP-0MVR7IXAI002QXXD: the baseline window's prompt-eval p10, the same value
    LP-0MVS03KCV005BTGE used to price early-abort waste). That constant prices a
    *whole* remaining prompt and is known to overstate a stopped prefill
    several-fold (measured prompt-eval p50 on the 2026-10-29 log is 158 tok/s),
    so a legacy row is labelled ``estimated=True`` and never mixed silently with
    a measured one. ``legacy_duration_s`` is always exported for comparison.

    Rows are keyed by ``start`` (the proxy release timestamp) and carry
    ``end``/``duration_s``/``bounded``/``basis``/``estimated``/``tokens``/
    ``measured_stop_s``/``legacy_duration_s`` plus the release's
    ``reason``/``session`` for the overlap accounting.
    """
    order = sorted(log.evictions, key=lambda e: (e.ts, e.session))
    paired = paired if paired is not None else [None] * len(order)
    rows = []
    for event, seg in zip(order, paired):
        tokens = _legacy_priced_tokens(log, event)
        if tokens <= 0:
            continue
        legacy_duration = tokens / ABANDONED_PREFILL_RATE_TOK_S
        measured_stop_s: float | None = None
        bounded = False
        basis = "legacy_rate_estimate"
        window_tokens = tokens
        if (
            seg is not None
            and seg.cancel_seen
            and seg.stop_seen
            and llama is not None
            and llama.observed_rate_tok_s
        ):
            held = seg.tokens_after_cancel
            if held > 0:
                measured_stop_s = held / llama.observed_rate_tok_s
                bounded = True
                basis = "measured_stop"
                window_tokens = held
        duration = measured_stop_s if bounded else legacy_duration
        assert duration is not None
        rows.append(
            {
                "start": event.ts.isoformat(sep=" ", timespec="seconds"),
                "end": _seconds_after(event.ts, duration).isoformat(
                    sep=" ", timespec="seconds"
                ),
                "duration_s": round(duration, 3),
                "bounded": bounded,
                "basis": basis,
                "estimated": not bounded,
                "tokens": window_tokens,
                "measured_stop_s": (
                    round(measured_stop_s, 3) if measured_stop_s is not None else None
                ),
                "legacy_duration_s": round(legacy_duration, 3),
                "reason": event.reason,
                "session": event.session,
            }
        )
    return rows


def _parse_window_row_start(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def _basis_rows(window_rows, basis: str) -> list[tuple[datetime, datetime, str, int, str]]:
    """Covering tuples ``(begin, end, reason, tokens, session)`` for one basis."""
    out: list[tuple[datetime, datetime, str, int, str]] = []
    for row in window_rows:
        start = _parse_window_row_start(row["start"])
        if basis == "measured_stop":
            if not row["bounded"] or row["measured_stop_s"] is None:
                continue
            end = _seconds_after(start, row["measured_stop_s"])
        else:
            end = _seconds_after(start, row["legacy_duration_s"])
        out.append((start, end, row["reason"], row["tokens"], row["session"]))
    return out


def _windows_covering(windows, event: EvictionEvent) -> list[tuple[datetime, str, int, str]]:
    """Windows whose span contains *event*'s dispatch start (same slack rule)."""
    start = event.dispatch_start
    return [
        (end, reason, tokens, session)
        for begin, end, reason, tokens, session in windows
        if begin <= _seconds_after(start, BACKLOG_START_SLACK_SECONDS) and end >= start
    ]


def _dual_basis_overlap(window_rows, deadline_events, measured_available: bool) -> dict:
    """The 58/60-style overlap on both bases, stating the verdict basis."""
    legacy_rows = _basis_rows(window_rows, "legacy_rate_estimate")
    measured_rows = _basis_rows(window_rows, "measured_stop")
    total = len(deadline_events)
    legacy_events = sum(1 for e in deadline_events if _windows_covering(legacy_rows, e))
    if measured_available:
        measured_events: int | None = sum(
            1 for e in deadline_events if _windows_covering(measured_rows, e)
        )
        verdict_basis = "measured_stop"
        verdict_events = measured_events
    else:
        measured_events = None
        verdict_basis = "legacy_rate_estimate"
        verdict_events = legacy_events
    return {
        "events": verdict_events,
        "total": total,
        "basis": verdict_basis,
        "measured": {
            "events": measured_events,
            "total": total,
            "basis": "measured_stop",
            "available": measured_available,
        },
        "legacy": {
            "events": legacy_events,
            "total": total,
            "basis": "legacy_rate_estimate",
            "available": True,
        },
    }


def _zero_progress_while_draining(log: WindowLog, window_rows, verdict_basis: str) -> dict:
    """Accepted dispatches that expired their lease with no measured progress
    while their dispatch started inside an abandoned window (parent AC3 second
    half), reported separately from the overlap share."""
    windows = _basis_rows(window_rows, verdict_basis)
    count = 0
    for event in log.evictions:
        if log.measurements_for(event.session, event.dispatch_start, event.ts):
            continue
        if _windows_covering(windows, event):
            count += 1
    return {"count": count, "total": len(log.evictions), "basis": verdict_basis}


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
    # LP-0MW8VP0XR007UGQF AC2: take the larger of the dispatch-record count and
    # the measured backend occupancy. Under a long prefill the dispatch count
    # can stay at 1 while a second slot is genuinely busy (the 2026-11-06
    # window attributed occupancy in only 9 of 60 deadline events); the
    # measured count is the one that sees it. ``None`` measured values (older
    # windows) fall back to the dispatch count, so the two reports stay
    # comparable.
    def _in_flight(p: StatusPoll) -> int:
        measured = p.measured_in_flight_requests
        if measured is None:
            return p.in_flight_requests
        return max(p.in_flight_requests, measured)

    max_in_flight = max((_in_flight(p) for p in polls), default=0)
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
    """Whether the dispatch started inside an abandoned-prefill backlog window.

    *orphan_windows* is the covering-tuple list ``(begin, end, reason, tokens,
    session)`` of one basis (see :func:`_basis_rows`). Kept for callers that
    want a single-basis answer; :func:`attribute_events` uses
    :func:`_coverage_window` on both bases instead.
    """
    result = _coverage_window(orphan_windows, event)
    if not result["overlap"]:
        return {"abandoned_prefill_overlap": False, "abandoned_prefill_remaining_s": None}
    return {
        "abandoned_prefill_overlap": True,
        "abandoned_prefill_remaining_s": result["remaining_s"],
        "abandoned_prefill_reason": result["reason"],
        "abandoned_prefill_tokens": result["tokens"],
    }


def attribute_events(
    log: WindowLog,
    *,
    window_rows: list[dict] | None = None,
    measured_available: bool = False,
) -> list[dict]:
    """One row per ``prefill_deadline`` event with its evidence and verdict.

    The abandoned-prefill overlap is reported on the *verdict* basis (measured
    when the backend log supplied cancel→stop evidence, otherwise the legacy
    estimate) with both basis counts alongside (AC3/AC4 of the window-model
    contract) plus the reason/tokens of the covering window on the verdict
    basis (backwards-compatible with the pre-correction evidence columns).
    """
    if window_rows is None:
        window_rows = abandoned_prefill_windows(log)
    verdict_basis = (
        "measured_stop" if measured_available else "legacy_rate_estimate"
    )
    legacy_windows = _basis_rows(window_rows, "legacy_rate_estimate")
    measured_windows = (
        _basis_rows(window_rows, "measured_stop") if measured_available else None
    )
    rows = []
    for event in [e for e in log.evictions if e.reason == REASON_PREFILL_DEADLINE]:
        occupancy = occupancy_evidence(log, event)
        verdict, evidence = classify_event(log, event, occupancy)
        measured = (
            _coverage_window(measured_windows, event) if measured_windows is not None else None
        )
        legacy = _coverage_window(legacy_windows, event)
        used = measured if measured is not None else legacy
        overlap_fields: dict = {
            "abandoned_prefill_overlap": bool(used and used["overlap"]),
            "abandoned_prefill_basis": verdict_basis,
            "abandoned_prefill_overlap_measured": bool(
                measured is not None and measured["overlap"]
            ),
            "abandoned_prefill_overlap_legacy": legacy["overlap"],
        }
        if used and used["overlap"]:
            overlap_fields["abandoned_prefill_remaining_s"] = used["remaining_s"]
            overlap_fields["abandoned_prefill_reason"] = used["reason"]
            overlap_fields["abandoned_prefill_tokens"] = used["tokens"]
        else:
            overlap_fields["abandoned_prefill_remaining_s"] = None
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
                    **overlap_fields,
                },
            }
        )
    return rows


def attribution_summary(
    rows,
    capture_available: bool = True,
    *,
    abandoned_prefill_overlap: dict | None = None,
    zero_progress_while_draining: dict | None = None,
) -> dict:
    counts = Counter(row["attribution"] for row in rows)
    by_class: dict[str, dict[str, int]] = {}
    for row in rows:
        by_class.setdefault(row["session_class"], Counter())[row["attribution"]] += 1
    if abandoned_prefill_overlap is None:
        abandoned_prefill_overlap = {
            "events": sum(1 for row in rows if row["evidence"]["abandoned_prefill_overlap"]),
            "total": len(rows),
        }
    summary = {
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
        "abandoned_prefill_overlap": abandoned_prefill_overlap,
    }
    if zero_progress_while_draining is not None:
        summary["zero_progress_while_draining"] = zero_progress_while_draining
    return summary


# ---------------------------------------------------------------------------
# Optional backend accounting cross-check
# ---------------------------------------------------------------------------


def backend_prefill_accounting(log: WindowLog, llama_log: str) -> dict:
    """Compare backend-completed prefills with proxy-consumed local streams.

    ``llama-server.log`` has no wall-clock timestamps, so this is valid only
    when the file covers exactly the analysed window (the day file of a
    logrotate-daily setup does). ``abandoned_prefills`` is then the number of
    prefills the GPU finished that no proxy stream consumed — the quantitative
    signature of an abandoned-prefill backlog.  *llama_log* may be a
    comma-separated list of paths/globs, like ``--llama-log``.
    """
    paths = _llama_log_paths(llama_log)
    if not paths:
        return {"llama_log": llama_log, "error": "not found"}
    completed = sum(
        1
        for line in read_log_lines(paths)
        if "prompt eval time" in line
    )
    consumed = len(log.local_completions)
    released = len(log.evictions)
    return {
        "llama_log": ",".join(str(path) for path in paths),
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
    llama, llama_spec = _load_llama_lifecycle(llama_log)
    order = sorted(log.evictions, key=lambda event: (event.ts, event.session))
    joins = _pair_releases(order, llama)
    paired = [join.segment for join in joins]
    window_rows = abandoned_prefill_windows(log, paired=paired, llama=llama)
    # A measured-stop window needs the observed rate to convert post-cancel
    # tokens to seconds; without an attributable Qwen3 basis the window model
    # honestly degrades to the legacy basis (LP-0MW957SKX0079QKC AC3).
    measured_available = (
        any(join.segment is not None and join.segment.stop_seen for join in joins)
        and llama is not None
        and llama.observed_rate_tok_s is not None
    )
    verdict_basis = (
        "measured_stop" if measured_available else "legacy_rate_estimate"
    )
    rows = attribute_events(
        log, window_rows=window_rows, measured_available=measured_available
    )
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
        "abandoned_prefill_windows": {
            "legacy_rate_tok_s": ABANDONED_PREFILL_RATE_TOK_S,
            "seconds_basis_tok_s": (
                llama.observed_rate_tok_s if llama is not None else None
            ),
            "basis": verdict_basis,
            "measured_available": measured_available,
            "coverage": _llama_coverage(llama_spec, llama, log),
            "note": LLAMA_COVERAGE_NOTE,
            "rows": window_rows,
        },
        # AC2/AC3: the Qwen3-only basis, with the excluded ports and
        # contributing line counts, so an operator can see why it moved (or
        # why it is None). Always present; ``reason`` names a missing log.
        "seconds_basis": seconds_basis_payload(llama),
        "attribution": attribution_summary(
            rows,
            capture_available=bool(log.measurements),
            abandoned_prefill_overlap=_dual_basis_overlap(
                window_rows, deadline_events, measured_available
            ),
            zero_progress_while_draining=_zero_progress_while_draining(
                log, window_rows, verdict_basis
            ),
        ),
        "events": rows,
    }
    if llama_log:
        report["release_cancel_accounting"] = release_cancel_accounting(log, llama=llama)
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
    parser.add_argument(
        "--llama-log",
        help=(
            "llama-server log/path-glob(s) covering exactly the window "
            "(comma-separated; enables release→cancel accounting and bounds "
            "the abandoned-prefill windows by observed backend stops)"
        ),
    )
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
    basis = report.get("seconds_basis")
    if basis is not None:
        if basis["tok_s"] is None:
            lines.append(f"Seconds basis: unavailable — {basis['reason']}")
        else:
            lines.append(
                f"Seconds basis: {basis['tok_s']} tok/s from {basis['model']} "
                f"({basis['contributing_eval_lines']} prompt-eval line(s), "
                f"port_source={basis['port_source']}); excluded ports: "
                f"{basis['excluded_ports'] or 'none'}"
            )
    windows = report.get("abandoned_prefill_windows")
    if windows:
        bounded = sum(1 for row in windows["rows"] if row["bounded"])
        lines.append(
            f"Abandoned-prefill windows: {len(windows['rows'])} releases, "
            f"{bounded} bounded by an observed backend stop "
            f"(basis={windows['basis']}, measured_available={windows['measured_available']})"
        )
    accounting = report.get("release_cancel_accounting")
    if accounting:
        unexplained = accounting["releases_without_explained_backend_stop"]
        unjoined = accounting["releases_unjoined"]
        stop = accounting["cancel_to_stop_seconds"]
        tokens = accounting["tokens_evaluated_after_cancel"]
        lines.append(
            f"Release→cancel: {accounting['releases_total']} releases, "
            f"{accounting['releases_with_observed_cancel']} with an observed "
            f"backend cancel, {unexplained['count']} without an explained stop "
            f"({unexplained['share']:.2%}), {unjoined['count']} unjoined"
        )
        lines.append(
            f"  cancel→stop latency s: p10={stop['p10']} p50={stop['p50']} "
            f"p90={stop['p90']} (n={stop['count']}); "
            f"tokens evaluated after cancel: {tokens['total']}"
        )
    attribution = report["attribution"]["totals"]
    lines.append(
        "Attribution: " + ", ".join(f"{name}={count}" for name, count in sorted(attribution.items()))
    )
    overlap = report["attribution"].get("abandoned_prefill_overlap")
    if overlap and "total" in overlap:
        lines.append(
            f"prefill_deadline overlap: {overlap['events']}/{overlap['total']} "
            f"(basis={overlap.get('basis', 'legacy_rate_estimate')})"
        )
    if not report["attribution"]["capture_available"]:
        lines.append("(no measured-prefill capture in this window: attributions rest on price alone)")
    resting = report.get("attribution", {}).get("zero_progress_while_draining")
    if resting:
        lines.append(
            f"Zero-progress lease expiries inside an abandoned window: "
            f"{resting['count']}/{resting['total']} (basis={resting['basis']})"
        )
    restarts = report["source"].get("restarts_in_window") or []
    if restarts:
        lines.append(f"Re-base events (restarts) in window: {len(restarts)}")
        for restart in restarts:
            lines.append(f"  {restart['ts']}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
