#!/usr/bin/env python3
"""Attribute ``local_concurrency_limit`` fall-throughs to the requester's own record.

LP-0MWNHBI6A001V5DD (parent LP-0MWMSQR3O002YNPW AC5). The parent fixes the
routing concurrency gate so the requesting session's **own** dispatch record
stops counting as occupancy while it holds no streaming dispatch token:

* an **idle lease** — ``active: false``, the slot a finished turn keeps
  reserved for its owner during the no-preemption cooldown
  (LP-0MW0VH2LB008JSKQ), and
* **finished-but-not-yet-released** — ``active: true`` with an empty
  ``streaming_dispatch_tokens`` set: the response reached its terminal chunk
  but request teardown has not released the credit yet
  (LP-0MWNHBG7D002NH5S).

A raw ``reason=local_concurrency_limit`` count cannot show whether the fix
worked, because the same line is still emitted for genuine contention. This
tool reconstructs each fall-through's **occupancy composition** from
``proxy.log`` and splits the events into:

``own_non_streaming_attributable``
    The requester's own record was present and held **no** streaming dispatch
    token (idle lease or finished-but-not-released), and the *other* sessions'
    records alone were below ``session_slot_pool_size``. Pre-fix that own
    record made ``cur_local >= max_local``; the fix excludes it and admits.
    This is the class the fix is expected to drive to ~0.

``own_streaming_gated``
    The requester's own record held a streaming dispatch token (a genuinely
    concurrent local request from the same session). Genuine in-flight work is
    still gated, before and after the fix.

``other_sessions_contention``
    The other sessions' present records already filled the pool, so the
    fall-through is real contention whether or not the own record was counted.
    This class is expected to remain after the fix.

``unattributable``
    Neither the own record nor the other sessions' records explain a
    ``cur_local >= max_local`` refusal at the log timestamp: the own record is
    absent/non-occupying and the others alone are below the pool size. This
    covers the inherent skew (a ``Fallback triggered`` line is emitted when the
    *next* provider's stream begins, several seconds after the routing
    decision), a prefill-cap refusal, or a log gap. Reported rather than
    folded into a headline number so a real shift is visible.

The own-record state is reconstructed from the local dispatch lifecycle lines
the proxy already emits:

============================  ============================================
line                          transition
============================  ============================================
``local_preference_resolved``
  ``outcome=local``           local dispatch admitted (a streaming token)
``Stream finished:`` …        terminal chunk: streaming token leaves, the
  ``provider=local``          record stays ``active`` (finished-unreleased)
``lease_renewed session=``    teardown: idle lease retained for the timeout
``lease_released session=``   record released (absent)
``local_eviction`` …          idle lease expired / record evicted (absent)
  ``session=``
============================  ============================================

Method mirrors the reference measurement on LP-0MW0VH2LB008JSKQ ("split the
``local_concurrency_limit`` refusals by whether the session's own local turn
finished within the lease timeout"), extended to the finished-but-not-released
window the parent fixes.

Timing caveat: a ``Fallback triggered`` line is emitted when the *next*
provider's stream begins (``_handle_streaming_success``), a few seconds after
the routing decision that refused local. The own record can therefore have
moved from finished-unreleased to idle between decision and log line. Both are
non-streaming, so the attribute class is unaffected; the sub-shape is taken
from the state at the log timestamp, and a recent local finish inside the lease
timeout also marks the event as own-attributable (the reference method's
lookback), which covers the skew.

Occupancy caveat: ``others_present`` is reconstructed from the same lifecycle
lines, not read from the gate. A record the reconstruction misses (a log gap, a
non-streaming local serve) makes the other-session count a *lower* bound, so
``own_non_streaming_attributable`` is an **upper** bound on the true
own-attributable count. Anonymous (``session=unknown``) callers share one
record and are excluded from occupancy, which only strengthens the upper bound.
Confirming the fix therefore errs on the safe side: a post-fix ~0 under an
upper-bound measure is strong evidence.

Usage::

    python3 attribute_local_concurrency.py \
        --start '2026-11-18 22:00:00' --until '2026-11-19 22:00:00' [--json]

``--json`` emits the recordable summary (per-class counts, window, filter and
method) suitable for a work-item comment. ``--rows`` adds the per-event
attribution.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

DEFAULT_LOG_DIR = "/var/log/llama-proxy"
DEFAULT_MAX_LOCAL = 2
DEFAULT_LEASE_TIMEOUT_SECONDS = 30.0
#: Lines are read this far before the window start so a dispatch admitted
#: before the window (and still running, or inside its idle lease) is in the
#: reconstructed state when the first in-window event is classified. It never
#: adds events: only fall-throughs inside the window are recorded.
DEFAULT_WARMUP_SECONDS = 6 * 3600.0
#: A ``Stream finished`` record is ``active`` until teardown re-news the lease
#: (measured ~1.4 s). If teardown never logs (a crash, a log gap) the record
#: must not stay "present" forever, so the finished-unreleased state is bounded
#: by a conservative grace well above the observed teardown latency.
DEFAULT_TEARDOWN_GRACE_SECONDS = 60.0

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ ")
KV = re.compile(r"(\w+)=([^\s]+)")

FALLBACK = "Fallback triggered"
FALLBACK_REASON = "reason=local_concurrency_limit"
PREFERENCE_RESOLVED = "local_preference_resolved "
STREAM_FINISHED = "Stream finished:"
LEASE_RENEWED = "lease_renewed "
LEASE_RELEASED = "lease_released "
LOCAL_EVICTION = "local_eviction "

#: Attribute classes, in report order.
OWN_NON_STREAMING_ATTRIBUTABLE = "own_non_streaming_attributable"
OWN_STREAMING_GATED = "own_streaming_gated"
OTHER_SESSIONS_CONTENTION = "other_sessions_contention"
UNATTRIBUTABLE = "unattributable"
ATTRIBUTION_CLASSES = (
    OWN_NON_STREAMING_ATTRIBUTABLE,
    OWN_STREAMING_GATED,
    OTHER_SESSIONS_CONTENTION,
    UNATTRIBUTABLE,
)

#: Own-record states reconstructed at the fall-through timestamp.
OWN_STREAMING = "streaming"
OWN_FINISHED_UNRELEASED = "finished_unreleased"
OWN_IDLE_LEASE = "idle_lease"
OWN_ABSENT_STATE = "absent"
OWN_STATES = (
    OWN_STREAMING,
    OWN_FINISHED_UNRELEASED,
    OWN_IDLE_LEASE,
    OWN_ABSENT_STATE,
)

#: The most a session may go unseen before its stale state is dropped. Only
#: affects memory, never the classification: a dropped session is by definition
#: absent at every later timestamp.
_STATE_TTL = timedelta(hours=1)

#: Session ids that do not name a real per-session dispatch record. Anonymous
#: callers share one record, so folding them into a single pseudo-session would
#: mis-price occupancy; they are skipped as "others" and cannot be attributed
#: as the requester's own.
_ANONYMOUS_SESSIONS = frozenset({"", "unknown"})


def _is_named_session(session: str | None) -> bool:
    return bool(session) and session not in _ANONYMOUS_SESSIONS


@dataclass
class _Session:
    """Reconstructed lifecycle of one session's local dispatch record."""

    streaming: int = 0
    present: bool = False
    expires_at: datetime | None = None
    #: ``finished`` while a terminal chunk has landed but teardown has not
    #: re-newsed the lease; ``idle`` after teardown; ``None`` before either.
    phase: str | None = None
    last_seen: datetime | None = None

    def note_seen(self, moment: datetime) -> None:
        self.last_seen = moment

    def is_present(self, moment: datetime) -> bool:
        if self.streaming > 0:
            return True
        if not self.present:
            return False
        return self.expires_at is None or moment <= self.expires_at

    def own_state(self, moment: datetime) -> str:
        if self.streaming > 0:
            return OWN_STREAMING
        if not self.is_present(moment):
            return OWN_ABSENT_STATE
        if self.phase == "finished":
            return OWN_FINISHED_UNRELEASED
        return OWN_IDLE_LEASE


class _State:
    """Processes relevant log lines in order and answers occupancy queries."""

    def __init__(self) -> None:
        self.sessions: dict[str, _Session] = {}
        self._last_finish: dict[str, datetime] = {}
        self._last_admit: dict[str, datetime] = {}

    def _session(self, session: str, moment: datetime) -> _Session:
        state = self.sessions.get(session)
        if state is None:
            state = _Session()
            self.sessions[session] = state
        state.note_seen(moment)
        return state

    def apply(self, kind: str, fields: dict[str, str], moment: datetime) -> None:
        """Apply one state-transition line."""
        session = fields.get("session") or fields.get("session_id")
        if not _is_named_session(session):
            return
        if kind == "admit_local":
            state = self._session(session, moment)
            state.streaming += 1
            state.present = True
            state.expires_at = None
            state.phase = "streaming"
            self._last_admit[session] = moment
        elif kind == "stream_finished_local":
            state = self._session(session, moment)
            state.streaming = max(0, state.streaming - 1)
            if state.streaming == 0:
                state.present = True
                state.expires_at = moment + timedelta(
                    seconds=self._teardown_grace
                )
                state.phase = "finished"
            self._last_finish[session] = moment
        elif kind == "lease_renewed":
            state = self._session(session, moment)
            if state.streaming == 0:
                state.present = True
                state.expires_at = moment + timedelta(
                    seconds=self._lease_timeout
                )
                state.phase = "idle"
        elif kind in ("lease_released", "local_eviction"):
            state = self._session(session, moment)
            # A release/eviction is the record's teardown, the strongest
            # "absent" signal: reset the whole lifecycle so a dispatch that
            # never logged a matching ``Stream finished`` (a non-streaming
            # local serve, or a prefill-aborted dispatch) cannot leak as a
            # perpetually-present record.
            state.streaming = 0
            state.present = False
            state.expires_at = None
            state.phase = None

    def configure(self, lease_timeout: float, teardown_grace: float) -> None:
        self._lease_timeout = lease_timeout
        self._teardown_grace = teardown_grace

    def others_present(self, session: str, moment: datetime) -> int:
        count = 0
        for key, state in self.sessions.items():
            if key == session:
                continue
            if state.is_present(moment):
                count += 1
        return count

    def recent_own_finish(self, session: str, moment: datetime) -> bool:
        """True when the session's own local turn finished inside the lease
        timeout and no new local dispatch has been admitted since.

        This is the reference method's lookback (LP-0MW0VH2LB008JSKQ) and it
        covers the decision/log-line skew: the routing decision can precede the
        ``Fallback triggered`` line by a few seconds, during which the own
        record moves from finished-unreleased to idle.
        """
        finished = self._last_finish.get(session)
        if finished is None:
            return False
        if (moment - finished).total_seconds() > self._lease_timeout:
            return False
        admitted = self._last_admit.get(session)
        return admitted is None or admitted <= finished

    def prune(self, moment: datetime) -> None:
        cutoff = moment - _STATE_TTL
        stale = [
            key
            for key, state in self.sessions.items()
            if state.streaming == 0
            and not state.present
            and state.last_seen is not None
            and state.last_seen < cutoff
        ]
        for key in stale:
            del self.sessions[key]
            self._last_finish.pop(key, None)
            self._last_admit.pop(key, None)


def _parse_ts(line: str) -> datetime | None:
    match = TS.match(line)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _kv(line: str, anchor: str) -> dict[str, str]:
    index = line.find(anchor)
    if index < 0:
        return {}
    return dict(KV.findall(line[index + len(anchor):]))


def _classify_line(line: str) -> tuple[str, str] | None:
    """Return ``(kind, anchor)`` for a relevant line, else ``None``."""
    if FALLBACK in line and FALLBACK_REASON in line:
        return ("fallthrough", FALLBACK)
    if PREFERENCE_RESOLVED in line and " outcome=local" in line:
        return ("admit_local", PREFERENCE_RESOLVED)
    if STREAM_FINISHED in line and " provider=local" in line:
        return ("stream_finished_local", STREAM_FINISHED)
    if LEASE_RENEWED in line:
        return ("lease_renewed", LEASE_RENEWED)
    if LEASE_RELEASED in line:
        return ("lease_released", LEASE_RELEASED)
    if LOCAL_EVICTION in line:
        return ("local_eviction", LOCAL_EVICTION)
    return None


def _open(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, errors="replace")


def collect(log_dir: str, window_start: datetime, window_end: datetime) -> list:
    """Collect the relevant in-window log records, de-duplicated and ordered.

    Rotated files are read whole and filtered to the window; the records are
    then sorted by timestamp and exact-duplicate records (an overlapping
    rotation) are dropped, so the state machine sees each transition once.
    """
    records: list[tuple[datetime, str, str, str]] = []
    for path in sorted(glob.glob(f"{log_dir}/proxy.log*")):
        try:
            handle = _open(path)
        except OSError:
            continue
        with handle:
            for line in handle:
                if "=" not in line:
                    continue
                moment = _parse_ts(line)
                if moment is None or moment < window_start or moment > window_end:
                    continue
                matched = _classify_line(line)
                if matched is None:
                    continue
                kind, anchor = matched
                records.append((moment, kind, anchor, line.rstrip("\n")))
    records.sort(key=lambda item: (item[0], item[3]))
    deduped: list[tuple[datetime, str, str, str]] = []
    previous: tuple[datetime, str] | None = None
    for record in records:
        key = (record[0], record[3])
        if key == previous:
            continue
        deduped.append(record)
        previous = key
    return deduped


def attribute(
    log_dir: str,
    window_start: datetime,
    window_end: datetime,
    max_local: int = DEFAULT_MAX_LOCAL,
    lease_timeout_seconds: float = DEFAULT_LEASE_TIMEOUT_SECONDS,
    teardown_grace_seconds: float = DEFAULT_TEARDOWN_GRACE_SECONDS,
    warmup_seconds: float = DEFAULT_WARMUP_SECONDS,
    include_rows: bool = False,
) -> dict:
    """Classify every ``local_concurrency_limit`` fall-through in the window."""
    state = _State()
    state.configure(lease_timeout_seconds, teardown_grace_seconds)
    rows: list[dict] = []
    processed = 0
    read_from = window_start - timedelta(seconds=max(0.0, warmup_seconds))
    for moment, kind, anchor, line in collect(log_dir, read_from, window_end):
        processed += 1
        if kind == "fallthrough":
            if moment < window_start:
                continue
            fields = _kv(line, anchor)
            session = fields.get("session", "")
            own_state = state.sessions[session].own_state(moment) if session in state.sessions else OWN_ABSENT_STATE
            recent_finish = state.recent_own_finish(session, moment)
            others = state.others_present(session, moment)
            streaming = own_state == OWN_STREAMING
            own_present = own_state != OWN_ABSENT_STATE
            own_non_streaming = (own_present and not streaming) or (
                recent_finish and not streaming
            )
            if streaming:
                attribution = OWN_STREAMING_GATED
            elif others >= max_local:
                attribution = OTHER_SESSIONS_CONTENTION
            elif own_non_streaming:
                attribution = OWN_NON_STREAMING_ATTRIBUTABLE
            else:
                attribution = UNATTRIBUTABLE
            rows.append(
                {
                    "ts": moment.isoformat(sep=" "),
                    "session": session,
                    "turn": fields.get("turn"),
                    "attempt": fields.get("attempt"),
                    "own_state": own_state,
                    "own_recent_finish_within_lease": recent_finish,
                    "others_present": others,
                    "attribution": attribution,
                }
            )
        else:
            state.apply(kind, _kv(line, anchor), moment)
        if processed % 5000 == 0:
            state.prune(moment)

    by_class = {name: 0 for name in ATTRIBUTION_CLASSES}
    by_own_state = {name: 0 for name in OWN_STATES}
    for row in rows:
        by_class[row["attribution"]] += 1
        by_own_state[row["own_state"]] += 1
    summary = {
        "window": {
            "start": window_start.isoformat(sep=" "),
            "end": window_end.isoformat(sep=" "),
            "hours": round((window_end - window_start).total_seconds() / 3600, 3),
        },
        "method": {
            "filter": "Fallback triggered … reason=local_concurrency_limit",
            "own_record": (
                "state reconstructed from local_preference_resolved outcome=local, "
                "Stream finished provider=local, lease_renewed, lease_released and "
                "local_eviction"
            ),
            "occupancy": "other sessions' present dispatch records at the event",
            "max_local": max_local,
            "lease_timeout_seconds": lease_timeout_seconds,
            "teardown_grace_seconds": teardown_grace_seconds,
            "warmup_seconds": warmup_seconds,
            "reference": "LP-0MW0VH2LB008JSKQ method, extended to the "
            "finished-but-not-released window (LP-0MWNHBG7D002NH5S)",
        },
        "events": len(rows),
        "by_attribution": by_class,
        "by_own_state": by_own_state,
        "own_record_attributable": by_class[OWN_NON_STREAMING_ATTRIBUTABLE],
        "genuine_contention": by_class[OWN_STREAMING_GATED]
        + by_class[OTHER_SESSIONS_CONTENTION],
        "unattributable": by_class[UNATTRIBUTABLE],
        "own_record_present": sum(
            1 for row in rows if row["own_state"] != OWN_ABSENT_STATE
        ),
    }
    if include_rows:
        summary["rows"] = rows
    return summary


def _parse_iso(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.fromisoformat(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="attribute_local_concurrency.py",
        description=(
            "Classify local_concurrency_limit fall-throughs by whether the "
            "requesting session's own dispatch record was counted occupancy."
        ),
    )
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--until", required=True, help="YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--max-local", type=int, default=DEFAULT_MAX_LOCAL)
    parser.add_argument(
        "--lease-timeout",
        type=float,
        default=DEFAULT_LEASE_TIMEOUT_SECONDS,
        help="local_dispatch_lease_timeout_seconds (default 30)",
    )
    parser.add_argument(
        "--teardown-grace",
        type=float,
        default=DEFAULT_TEARDOWN_GRACE_SECONDS,
        help="bound on the finished-but-not-released window (default 60 s)",
    )
    parser.add_argument(
        "--warmup-hours",
        type=float,
        default=DEFAULT_WARMUP_SECONDS / 3600.0,
        help=(
            "read this many hours before --start to establish in-flight state "
            "(default 6; never adds events)"
        ),
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--rows", action="store_true", help="include per-event attribution"
    )
    args = parser.parse_args(argv)

    window_start = _parse_iso(args.start)
    window_end = _parse_iso(args.until)
    if window_start >= window_end:
        print("error: --start must be before --until")
        return 2
    summary = attribute(
        args.log_dir,
        window_start,
        window_end,
        max_local=args.max_local,
        lease_timeout_seconds=args.lease_timeout,
        teardown_grace_seconds=args.teardown_grace,
        warmup_seconds=max(0.0, args.warmup_hours) * 3600.0,
        include_rows=args.rows,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    print(
        f"local_concurrency_limit fall-throughs: {summary['events']} "
        f"over {summary['window']['hours']:.1f} h"
    )
    for name in ATTRIBUTION_CLASSES:
        print(f"  {summary['by_attribution'][name]:6d}  {name}")
    print(
        f"  {summary['own_record_attributable']:6d}  own_record_attributable  "
        f"(the fix's target: expected ~0 post-fix)"
    )
    print(
        f"  {summary['genuine_contention']:6d}  genuine_contention  "
        f"(expected to remain)"
    )
    print("own-record state at the event: ")
    for name in OWN_STATES:
        print(f"  {summary['by_own_state'][name]:6d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
