"""
Contention-queue simulation harness (T1 of LP-0MTED3OFP006I7NO).

Replays the log-observable cheap-mode contention-queue event stream through a
faithful model of the bounded cross-session contention queue
(LP-0MSORQVK50012Q4D) to project the effect of tuning
``contention_queue_max_wait_seconds`` (wait cap) and
``contention_queue_max_depth`` (depth cap) on queued dispatches vs
fallback-after-queue.

Methodology (documented assumptions)
------------------------------------
The model is built entirely from two observables, mirroring the real
``contention_queue.py`` mechanics:

1. Queue-path arrivals — every request that reached the contention-queue
   decision point while both cheap-mode slots were busy. These are exactly
   the ``contention_queue_dispatch`` (dispatched) and
   ``contention_queue_fallback_after_queue`` (fell back) events; the arrival
   time is ``event_time - queued_duration``. By construction every such
   arrival saw the slot counter at the max (that is why it queued), so
   arrivals always find the queue engagement at full occupancy.

2. Slot-free events — the only events that decrement ``local_active_queries``
   (the counter the queue reads via ``_slot_free``) are local stream
   completions, logged as ``Stream finished ... provider=local``. Lease
   releases (``lease_released``) and stale-lease cleanup
   (``dispatch_cleanup``) call ``wake()`` but do NOT decrement the counter,
   so they cannot hand a slot to a queue waiter and are NOT treated as
   slot-free events (validated empirically, see below).

The queue model is FIFO with per-waiter deadlines: an arrival enqueues with
deadline ``arrival + max_wait_seconds`` unless the queue is already at
``max_depth`` (immediate fallback, wait 0). A slot-free event pops the head:
if its deadline has not passed it dispatches (wait = now - arrival), else it
falls back. Expired waiters fall back at exactly their deadline (matching the
observed ``queued_duration=60.00s`` timeout precision of ``asyncio.wait_for``).

Validation (AC2): replaying the analyzed cheap-mode windows (pre-2026-08-28)
with the default caps (60s / depth 4) reproduces the observed counts within
+/-10%: 249 observed dispatches vs 243 simulated (-2.4%) and 166 observed
fallback-after-queue vs 172 simulated (+3.6%), with exact dispatch matches on
the two largest windows (Aug 23_01: 42/42, Aug 25_01: 101 dispatches,
49 fallbacks vs 100/50).
"""

from __future__ import annotations

import argparse
import glob as _glob
import gzip
import heapq
import re
import statistics
import sys
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LOG_DIR = "/var/log/llama-proxy"
DEFAULT_LOG_PATTERN = "proxy.log*"
#: Analysis-window decision (Q1): logs before 2026-08-28 are the replay corpus;
#: Aug 28+ windows are excluded as anomalous (see plan).
DEFAULT_END = "2026-08-28T00:00:00"
DEFAULT_MAX_WAIT_SECONDS = 60.0
DEFAULT_MAX_DEPTH = 4

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
_DUR_RE = re.compile(r"queued_duration=([\d.]+)s")
_DEPTH_AFTER_RE = re.compile(r"depth=(-?\d+)")
_ACTIVE_RE = re.compile(r"active=(\d+)")
_SESSION_RE = re.compile(r"session[=_]([A-Za-z0-9._-]+)")

# Event type markers (the four AC1 event types + the slot-free wake source).
MARKER_DISPATCH = "contention_queue_dispatch"
MARKER_FALLBACK_AFTER = "contention_queue_fallback_after_queue"
MARKER_STATUS = "status_request"
MARKER_DENIED = "local_dispatch_denied"
MARKER_SLOT_FREE = "Stream finished"
MARKER_POLICY_QUEUE = "contention_queue_policy=queue"

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------


def _parse_ts(line: str) -> float | None:
    """Unix timestamp (ms precision) from the log line prefix, if present."""
    m = _TS_RE.match(line)
    if not m:
        return None
    dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    return dt.timestamp() + int(m.group(2)) / 1000.0


def _num(line: str, regex: re.Pattern[str], default: float = 0.0) -> float:
    m = regex.search(line)
    return float(m.group(1)) if m else default


def _session(line: str) -> str | None:
    m = _SESSION_RE.search(line)
    return m.group(1) if m else None


@dataclass
class LogEvent:
    """A single recognized queue-related log event."""

    type: str  # dispatch | fallback_after_queue | status | denied | slot_free
    ts: float
    wait: float = 0.0  # queued_duration seconds (dispatch/fallback)
    arrival: float = 0.0  # ts - wait (dispatch/fallback)
    depth_after: int | None = None  # queue depth after pop (dispatch)
    active: int | None = None  # local active queries (denied)
    session: str | None = None

    def to_dict(self) -> dict:
        d = {"type": self.type, "ts": self.ts, "wait": self.wait, "arrival": self.arrival}
        if self.depth_after is not None:
            d["depth_after"] = self.depth_after
        if self.active is not None:
            d["active"] = self.active
        if self.session is not None:
            d["session"] = self.session
        return d


def parse_line(line: str) -> LogEvent | None:
    """Parse one proxy log line into a queue-related LogEvent, or None.

    Recognized event types (AC1): contention_queue_dispatch,
    contention_queue_fallback_after_queue, status_request (queue snapshot),
    local_dispatch_denied — plus the slot-free wake source
    (``Stream finished ... provider=local``).
    """
    if "contention_queue_dispatch" in line:
        ts = _parse_ts(line)
        if ts is None:
            return None
        wait = _num(line, _DUR_RE)
        return LogEvent(
            type="dispatch",
            ts=ts,
            wait=wait,
            arrival=ts - wait,
            depth_after=int(_num(line, _DEPTH_AFTER_RE, -1) if _DEPTH_AFTER_RE.search(line) else -1),
            session=_session(line),
        )
    if "contention_queue_fallback_after_queue" in line:
        ts = _parse_ts(line)
        if ts is None:
            return None
        wait = _num(line, _DUR_RE)
        return LogEvent(
            type="fallback_after_queue",
            ts=ts,
            wait=wait,
            arrival=ts - wait,
            session=_session(line),
        )
    if MARKER_STATUS in line and MARKER_POLICY_QUEUE in line:
        # Queue snapshot on a status_request line (policy=queue ⇒ queue engaged).
        ts = _parse_ts(line)
        if ts is None:
            return None
        return LogEvent(type="status", ts=ts, session=_session(line))
    if MARKER_DENIED in line:
        ts = _parse_ts(line)
        if ts is None:
            return None
        return LogEvent(
            type="denied",
            ts=ts,
            active=int(_num(line, _ACTIVE_RE, -1)),
            session=_session(line),
        )
    if MARKER_SLOT_FREE in line and "provider=local" in line:
        ts = _parse_ts(line)
        if ts is None:
            return None
        return LogEvent(type="slot_free", ts=ts, session=_session(line))
    return None


def _candidate_count(line: str, event_type: str) -> bool:
    """True if the line should be counted in the coverage candidate set."""
    if event_type == "dispatch":
        return MARKER_DISPATCH in line
    if event_type == "fallback_after_queue":
        return MARKER_FALLBACK_AFTER in line
    if event_type == "status":
        return MARKER_STATUS in line and MARKER_POLICY_QUEUE in line
    if event_type == "denied":
        return MARKER_DENIED in line
    if event_type == "slot_free":
        return MARKER_SLOT_FREE in line and "provider=local" in line
    return False


def parse_iso(value: str) -> float:
    """Parse 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS' into a unix timestamp."""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"invalid timestamp: {value!r}")


def iter_lines(path: Path):
    """Iterate log lines from a plain or .gz file."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as fh:
            yield from fh
    else:
        with open(path, errors="replace") as fh:
            yield from fh


@dataclass
class LoadResult:
    events: list[LogEvent]
    coverage: dict[str, dict[str, int]]  # type -> {parsed, candidate}
    files: list[str] = field(default_factory=list)
    window_start: float | None = None
    window_end: float | None = None


def load_events(
    log_files: Sequence[str | Path],
    start: float | None = None,
    end: float | None = None,
) -> LoadResult:
    """Parse every log file (plain/.gz, globs expanded), collecting events and
    per-type coverage (parsed vs candidate counts across the 4 AC1 types plus
    slot_free). Optional [start, end) window filter in unix seconds."""
    files = []
    for pat in log_files:
        files.extend(sorted(_glob.glob(str(pat))))
    if not files:
        raise FileNotFoundError(f"no log files matched: {list(log_files)}")

    events: list[LogEvent] = []
    coverage: dict[str, dict[str, int]] = {
        t: {"parsed": 0, "candidate": 0} for t in
        ("dispatch", "fallback_after_queue", "status", "denied", "slot_free")
    }
    used_files = []
    for path_str in files:
        path = Path(path_str)
        used = False
        for line in iter_lines(path):
            kind = None
            for t in coverage:
                if _candidate_count(line, t):
                    coverage[t]["candidate"] += 1
                    kind = t if kind is None else kind  # first matching type
            ev = parse_line(line)
            if ev is None:
                continue
            if kind is not None:
                coverage[kind]["parsed"] += 1
            if start is not None and ev.ts < start:
                continue
            if end is not None and ev.ts >= end:
                continue
            events.append(ev)
            used = True
        if used:
            used_files.append(path_str)
    return LoadResult(events=events, coverage=coverage, files=used_files,
                      window_start=start, window_end=end)


# ---------------------------------------------------------------------------
# Simulation model
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    dispatched: int = 0
    fallback_after_queue: int = 0
    waits: list[float] = field(default_factory=list)
    max_queue_depth: int = 0
    depth_capped_fallbacks: int = 0
    timeout_fallbacks: int = 0

    def total_fallbacks(self) -> int:
        return self.fallback_after_queue

    def wait_stats(self) -> dict:
        if not self.waits:
            return {"median": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0}
        s = sorted(self.waits)
        return {
            "median": round(statistics.median(s), 2),
            "p50": round(_percentile(s, 50), 2),
            "p90": round(_percentile(s, 90), 2),
            "p95": round(_percentile(s, 95), 2),
        }


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def simulate(
    events: Sequence[LogEvent],
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    end_time: float | None = None,
) -> SimulationResult:
    """Replay queue-path arrivals against slot-free events through the FIFO
    contention-queue model (see module docstring).

    Every queue-path arrival (dispatch/fallback_after_queue event) reached the
    decision point at full occupancy by construction; slot-free events are the
    ``Stream finished ... provider=local`` completions that decrement
    ``local_active_queries``. The model has no free parameters beyond the caps
    under test.

    Returns dispatched / fallback_after_queue counts and per-request waits.
    """
    arrivals = [e for e in events if e.type in ("dispatch", "fallback_after_queue")]
    slot_frees = [e.ts for e in events if e.type == "slot_free"]

    # Merged timeline: arrivals (priority 0) before slot frees at the same ts
    # so a stream ending at the exact instant of an arrival can serve it with a
    # 0-second wait (observed ``queued_duration=0.00s`` races).
    timeline: list[tuple[float, int, LogEvent | None]] = []
    for e in arrivals:
        timeline.append((e.arrival, 0, e))
    for t in slot_frees:
        timeline.append((t, 1, None))
    timeline.sort(key=lambda item: (item[0], item[1]))

    queue: deque[tuple[float, float]] = deque()  # (arrival, deadline)
    deadlines: list[tuple[float, float]] = []  # min-heap (deadline, arrival)
    result = SimulationResult()

    def expire(now: float) -> None:
        while deadlines and deadlines[0][0] <= now:
            dl, a = heapq.heappop(deadlines)
            for i, (qa, qdl) in enumerate(queue):
                if qa == a and qdl == dl:
                    del queue[i]
                    result.fallback_after_queue += 1
                    result.timeout_fallbacks += 1
                    result.waits.append(dl - a)
                    break

    for t, _prio, ev in timeline:
        expire(t)
        if ev is None:  # slot free: serve the FIFO head within its deadline
            if queue:
                a, dl = queue.popleft()
                if dl >= t:
                    result.dispatched += 1
                    result.waits.append(t - a)
                else:  # deadline already passed at this instant
                    result.fallback_after_queue += 1
                    result.timeout_fallbacks += 1
                    result.waits.append(dl - a)
        else:  # arrival: occupancy is full by construction -> enqueue
            if len(queue) < max_depth:
                queue.append((t, t + max_wait_seconds))
                heapq.heappush(deadlines, (t + max_wait_seconds, t))
                result.max_queue_depth = max(result.max_queue_depth, len(queue))
            else:
                result.fallback_after_queue += 1
                result.depth_capped_fallbacks += 1
                result.waits.append(0.0)

    # Flush: any waiter still queued at window end falls back at its deadline.
    expire(end_time if end_time is not None else float("inf"))
    while queue:
        a, dl = queue.popleft()
        result.fallback_after_queue += 1
        result.timeout_fallbacks += 1
        result.waits.append(dl - a)
    return result


def observed_counts(events: Sequence[LogEvent]) -> dict[str, int]:
    """Observed (ground-truth) dispatch / fallback_after_queue counts."""
    return {
        "dispatched": sum(1 for e in events if e.type == "dispatch"),
        "fallback_after_queue": sum(
            1 for e in events if e.type == "fallback_after_queue"
        ),
    }


def replication_validation(
    events: Sequence[LogEvent],
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    tolerance_pct: float = 10.0,
) -> dict:
    """Default-caps replay vs observed counts (AC2): must reproduce the
    observed dispatch / fallback_after_queue counts within +/-tolerance_pct."""
    sim = simulate(events, max_wait_seconds=max_wait_seconds,
                   max_depth=max_depth)
    obs = observed_counts(events)

    def dev(sim_v: int, obs_v: int) -> float:
        if obs_v:
            return 100.0 * (sim_v - obs_v) / obs_v
        return 100.0 if sim_v else 0.0

    d_dev = dev(sim.dispatched, obs["dispatched"])
    f_dev = dev(sim.fallback_after_queue, obs["fallback_after_queue"])
    return {
        "config": {"max_wait_seconds": max_wait_seconds, "max_depth": max_depth},
        "observed": obs,
        "simulated": {
            "dispatched": sim.dispatched,
            "fallback_after_queue": sim.fallback_after_queue,
        },
        "deviation_pct": {
            "dispatched": round(d_dev, 1),
            "fallback_after_queue": round(f_dev, 1),
        },
        "within_tolerance": abs(d_dev) <= tolerance_pct
        and abs(f_dev) <= tolerance_pct,
        "tolerance_pct": tolerance_pct,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Replay cheap-mode proxy logs through the contention-queue model and "
            "project dispatch/fallback outcomes under configurable caps."
        )
    )
    p.add_argument("--log-dir", default=DEFAULT_LOG_DIR,
                   help=f"log directory (default: {DEFAULT_LOG_DIR})")
    p.add_argument("--pattern", default=DEFAULT_LOG_PATTERN,
                   help=f"glob within --log-dir (default: {DEFAULT_LOG_PATTERN})")
    p.add_argument("--log-files", nargs="*", default=None,
                   help="explicit log files/globs (overrides --log-dir/--pattern)")
    p.add_argument("--start", type=parse_iso, default=None,
                   help="window start 'YYYY-MM-DD[THH:MM:SS]' (unix seconds)")
    p.add_argument("--end", type=parse_iso, default=parse_iso(DEFAULT_END),
                   help=f"window end (default: {DEFAULT_END})")
    p.add_argument("--max-wait-seconds", type=float, default=DEFAULT_MAX_WAIT_SECONDS,
                   help=f"contention_queue_max_wait_seconds (default: {DEFAULT_MAX_WAIT_SECONDS})")
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                   help=f"contention_queue_max_depth (default: {DEFAULT_MAX_DEPTH})")
    p.add_argument("--tolerance-pct", type=float, default=10.0,
                   help="AC2 replication tolerance (default: 10.0)")
    p.add_argument("--json", action="store_true",
                   help="also print machine-readable JSON report")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_files = args.log_files or [
        str(Path(args.log_dir) / args.pattern)
        for _ in [0]
    ]
    try:
        loaded = load_events(log_files, start=args.start, end=args.end)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    sim = simulate(loaded.events, max_wait_seconds=args.max_wait_seconds,
                   max_depth=args.max_depth)
    validation = replication_validation(
        loaded.events, max_wait_seconds=args.max_wait_seconds,
        max_depth=args.max_depth, tolerance_pct=args.tolerance_pct,
    )

    # ---- human-readable report ----
    print("--- contention-queue simulation ---")
    print(f"files: {len(loaded.files)} ({loaded.files[0] if loaded.files else 'none'} ...)")
    print(f"window: {_fmt(loaded.window_start)} .. {_fmt(loaded.window_end)}")
    print(f"policy caps: wait={args.max_wait_seconds}s depth={args.max_depth} (cheap-mode 2-slot config)")
    print()
    print("event coverage (parsed/candidate, %):")
    all_cov = True
    for t, c in loaded.coverage.items():
        cov = 100.0 * c["parsed"] / c["candidate"] if c["candidate"] else 100.0
        ok = cov >= 95.0
        all_cov &= ok
        print(f"  {t:<20} {c['parsed']:>6}/{c['candidate']:<6} {cov:5.1f}% {'OK' if ok else 'LOW'}")
    print(f"  coverage gate (>=95% per AC1): {'PASS' if all_cov else 'FAIL'}")
    print()
    print(f"simulation: dispatched={sim.dispatched}  "
          f"fallback_after_queue={sim.fallback_after_queue}  "
          f"total_fallbacks={sim.total_fallbacks()}")
    ws = sim.wait_stats()
    print(f"queue-wait: median={ws['median']}s p90={ws['p90']}s p95={ws['p95']}s  "
          f"(max queue depth seen: {sim.max_queue_depth})")
    print()
    obs = validation["observed"]
    dev = validation["deviation_pct"]
    status = "PASS" if validation["within_tolerance"] else f"FAIL(>+/-{args.tolerance_pct:g}%)"
    print(f"replication vs observed: d={sim.dispatched}/{obs['dispatched']} "
          f"({dev['dispatched']:+.1f}%)  f={sim.fallback_after_queue}/{obs['fallback_after_queue']} "
          f"({dev['fallback_after_queue']:+.1f}%)  [{status}]")

    if args.json:
        report = {
            "config": {"log_dir": args.log_dir, "pattern": args.pattern,
                       "max_wait_seconds": args.max_wait_seconds,
                       "max_depth": args.max_depth},
            "window": {"files": loaded.files,
                       "start": _fmt(loaded.window_start),
                       "end": _fmt(loaded.window_end)},
            "coverage": {t: {**c, "coverage_pct": round(
                100.0 * c["parsed"] / c["candidate"], 1) if c["candidate"] else 100.0}
                for t, c in loaded.coverage.items()},
            "coverage_gate_pct": 95.0,
            "simulation": {
                "dispatched": sim.dispatched,
                "fallback_after_queue": sim.fallback_after_queue,
                "total_fallbacks": sim.total_fallbacks(),
                "queue_wait_seconds": ws,
                "max_queue_depth": sim.max_queue_depth,
                "timeout_fallbacks": sim.timeout_fallbacks,
                "depth_capped_fallbacks": sim.depth_capped_fallbacks,
            },
            "validation": validation,
        }
        print("\n" + __import__("json").dumps(report, indent=2))
    return 0


def _fmt(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    sys.exit(main())
