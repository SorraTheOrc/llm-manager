"""
Burst profile quantification (T2 of LP-0MTED3OFP006I7NO).

Quantifies the bursty fan-out profile of cheap-mode traffic from the replayed
pre-2026-08-28 windows, consuming the T1 log-based simulation harness
(``contention_queue_simulation``):

1. Concurrent local requests during bursts — distribution of
   ``local_active_queries`` (from ``local_dispatch_denied active=``) sampled
   inside burst windows, plus ``available_slots`` from status snapshots.
2. Queue depth observed — ``contention_queue_depth`` from status_request
   snapshots and ``depth=`` (depth after pop) on dispatch lines.
3. Queue-wait durations — ``queued_duration`` on
   ``contention_queue_dispatch`` vs ``contention_queue_fallback_after_queue``.
4. Fallbacks while the queue was non-empty — ``contention_queue_fallback_after_queue``
   and ``Fallback triggered reason=local_concurrency_limit`` events matched
   against the nearest status_request snapshot's queue depth within a delta.

Burst definition (explicit, applied consistently): a maximal run of
full-occupancy contention events (queue-path arrivals plus immediate denials)
whose inter-event gaps are <= BURST_GAP_SECONDS (default 30s), with at
least BURST_MIN_ARRIVALS events. The 30s gap threshold sits above the
observed median (~5s) / p90 (~220s, minutes-long quiet stretches split out)
inter-event gap so intra-burst fan-out is captured while quiet inter-burst
stretches (minutes+) are split.

Generates the committed report ``docs/dev/contention-queue-burst-profile.md``.
"""

from __future__ import annotations

import argparse
import glob as _glob
import gzip
import json
import re
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:  # same import fallback pattern as the benchmark tests
    from proxy.benchmarks import contention_queue_simulation as cs
except ImportError:  # pragma: no cover - exercised when run as a script
    import importlib.util
    import sys

    _here = Path(__file__).resolve().parent  # proxy/benchmarks
    _spec = importlib.util.spec_from_file_location(
        "contention_queue_simulation", _here / "contention_queue_simulation.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _mod  # dataclasses need the module registered
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    cs = _mod  # type: ignore[assignment]

DEFAULT_LOG_DIR = cs.DEFAULT_LOG_DIR
DEFAULT_LOG_PATTERN = cs.DEFAULT_LOG_PATTERN
DEFAULT_END = cs.DEFAULT_END
DEFAULT_BURST_GAP_SECONDS = 30.0
DEFAULT_BURST_MIN_ARRIVALS = 2
DEFAULT_SNAPSHOT_DELTA_SECONDS = 30.0
DEFAULT_REPORT = "docs/dev/contention-queue-burst-profile.md"

_FALLBACK_REASON_RE = re.compile(r"reason=([a-z_]+)")
_MARKER_FALLBACK_TRIGGERED = "Fallback triggered"
_REASON_LOCAL_CONCURRENCY = "local_concurrency_limit"
_DEPTH_RE = re.compile(r"contention_queue_depth=(\d+)")
_SLOTS_RE = re.compile(r"available_slots=(\d+)")

#: Events that hit the full-occupancy contention decision point: queue-path
#: arrivals (dispatch/fallback_after_queue) plus immediate denials. Together
#: they capture the fan-out "hit" stream that drives bursts (the denied stream
#: alone matches the parent analysis: median gap ~4s, p90 ~21s; queue-path
#: arrivals alone are far sparser, median ~31s).
BURST_TYPES = ("dispatch", "fallback_after_queue", "denied")

# ---------------------------------------------------------------------------
# Raw-line helpers (fields the T1 harness does not retain)
# ---------------------------------------------------------------------------


def iter_lines(path: Path):
    """Iterate log lines from a plain or .gz file (same as the T1 harness)."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as fh:
            yield from fh
    else:
        with open(path, errors="replace") as fh:
            yield from fh


def parse_fallback_triggered(line: str) -> dict | None:
    """Parse a ``Fallback triggered ... reason=`` line into {ts, reason}."""
    if _MARKER_FALLBACK_TRIGGERED not in line:
        return None
    ts = cs._parse_ts(line)
    if ts is None:
        return None
    m = _FALLBACK_REASON_RE.search(line)
    return {"ts": ts, "reason": m.group(1) if m else "unknown"}


# ---------------------------------------------------------------------------
# Burst / distribution helpers
# ---------------------------------------------------------------------------


def _event_time(e: cs.LogEvent) -> float:
    """Time of a contention event: arrival for queue-path events (ts - wait),
    ts for immediate denials."""
    if e.type in ("dispatch", "fallback_after_queue"):
        return e.arrival
    return e.ts


def inter_arrival_gaps(events: Sequence[cs.LogEvent],
                        types: Sequence[str] = BURST_TYPES) -> list[float]:
    """Gaps between consecutive full-occupancy events (seconds)."""
    arrivals = sorted(_event_time(e) for e in events if e.type in types)
    return [b - a for a, b in zip(arrivals, arrivals[1:])]


def detect_bursts(
    events: Sequence[cs.LogEvent],
    gap_seconds: float = DEFAULT_BURST_GAP_SECONDS,
    min_arrivals: int = DEFAULT_BURST_MIN_ARRIVALS,
    types: Sequence[str] = BURST_TYPES,
) -> list[tuple[float, float, int]]:
    """Maximal runs of full-occupancy events with inter-event gaps <= gap.

    Returns (burst_start, burst_end, arrival_count) tuples.
    """
    arrivals = [_event_time(e) for e in events if e.type in types]
    arrivals.sort()
    bursts: list[list[float]] = []
    for t in arrivals:
        if bursts and t - bursts[-1][-1] <= gap_seconds:
            bursts[-1].append(t)
        else:
            bursts.append([t])
    return [(b[0], b[-1], len(b)) for b in bursts if len(b) >= min_arrivals]


def nearest_depth_bucket(snapshot_depths: dict, ts: float,
                         delta: float) -> str | None:
    """Classify the nearest status snapshot within +-delta by depth.

    Returns 'gt0' (queue non-empty), 'eq0' (queue empty), or None when no
    snapshot exists within the delta (not derivable).
    """
    best_dt: float | None = None
    best_depth: int | None = None
    for sts, d in snapshot_depths.items():
        dt = abs(sts - ts)
        if dt <= delta and (best_dt is None or dt < best_dt):
            best_dt = dt
            best_depth = d
    if best_depth is None:
        return None
    return "gt0" if best_depth > 0 else "eq0"


def _pct(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (k - lo) * (sorted_values[hi] - sorted_values[lo])


def _dist_stats(values: list[int]) -> dict:
    if not values:
        return {"n": 0, "max": 0, "p50": 0, "p90": 0, "mean": 0.0, "histogram": {}}
    s = sorted(values)
    return {
        "n": len(values),
        "max": s[-1],
        "p50": int(_pct(s, 50)),
        "p90": int(_pct(s, 90)),
        "mean": round(statistics.mean(s), 2),
        "histogram": {v: values.count(v) for v in sorted(set(values))},
    }


def _fmt(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


@dataclass
class ProfileResult:
    window_start: float | None = None
    window_end: float | None = None
    files: list[str] = field(default_factory=list)
    arrivals: int = 0
    gaps: list[float] = field(default_factory=list)
    bursts: list[tuple[float, float, int]] = field(default_factory=list)
    denied_active: list[int] = field(default_factory=list)
    available_slots: list[int] = field(default_factory=list)
    status_depth: list[int] = field(default_factory=list)
    dispatch_depth_after: list[int] = field(default_factory=list)
    dispatch_waits: list[float] = field(default_factory=list)
    fallback_waits: list[float] = field(default_factory=list)
    fallback_triggered_cheap: list[dict] = field(default_factory=list)
    fallback_after_queue_total: int = 0
    fallback_triggered_total: int = 0
    # three-way split of coincidence with a non-empty queue:
    # {'gt0': .., 'eq0': .., 'no_snapshot': ..}
    fallback_after_queue_split: dict = field(default_factory=dict)
    fallback_triggered_split: dict = field(default_factory=dict)


def compute_profile(
    events: Sequence[cs.LogEvent],
    status_depth: dict,
    available_slots: Sequence[int],
    fallback_triggered: Sequence[dict],
    window_start: float | None = None,
    window_end: float | None = None,
    files: Sequence[str] | None = None,
    burst_gap_seconds: float = DEFAULT_BURST_GAP_SECONDS,
    min_arrivals: int = DEFAULT_BURST_MIN_ARRIVALS,
    snapshot_delta: float = DEFAULT_SNAPSHOT_DELTA_SECONDS,
) -> ProfileResult:
    res = ProfileResult(window_start=window_start, window_end=window_end,
                        files=list(files or []))
    res.arrivals = sum(1 for e in events if e.type in BURST_TYPES)
    res.gaps = inter_arrival_gaps(events)
    res.bursts = detect_bursts(events, gap_seconds=burst_gap_seconds,
                               min_arrivals=min_arrivals)
    burst_bounds = [(b[0], b[1]) for b in res.bursts]

    res.status_depth = list(status_depth.values())
    res.available_slots = list(available_slots)
    res.fallback_triggered_cheap = [
        f for f in fallback_triggered if f["reason"] == _REASON_LOCAL_CONCURRENCY
    ]

    for e in events:
        if e.type == "denied" and e.active is not None:
            if any(s <= e.ts <= e0 for s, e0 in burst_bounds):
                res.denied_active.append(e.active)
        elif e.type == "dispatch":
            res.dispatch_waits.append(e.wait)
            if e.depth_after is not None and e.depth_after >= 0:
                res.dispatch_depth_after.append(e.depth_after)
        elif e.type == "fallback_after_queue":
            res.fallback_waits.append(e.wait)

    fbq = [e for e in events if e.type == "fallback_after_queue"]
    res.fallback_after_queue_total = len(fbq)
    res.fallback_after_queue_split = _bucket_counts(
        fbq, status_depth, snapshot_delta, lambda e: e.ts)
    res.fallback_triggered_total = len(res.fallback_triggered_cheap)
    res.fallback_triggered_split = _bucket_counts(
        res.fallback_triggered_cheap, status_depth, snapshot_delta,
        lambda f: f["ts"])
    return res


def _bucket_counts(items: Sequence, snapshots: dict, delta: float,
                   ts_of) -> dict:
    buckets = {"gt0": 0, "eq0": 0, "no_snapshot": 0}
    for item in items:
        bucket = nearest_depth_bucket(snapshots, ts_of(item), delta)
        buckets[bucket if bucket is not None else "no_snapshot"] += 1
    return buckets


def load_profile(
    log_files: Sequence[str | Path],
    start: float | None = None,
    end: float | None = None,
    burst_gap_seconds: float = DEFAULT_BURST_GAP_SECONDS,
    min_arrivals: int = DEFAULT_BURST_MIN_ARRIVALS,
    snapshot_delta: float = DEFAULT_SNAPSHOT_DELTA_SECONDS,
) -> ProfileResult:
    """Load events via the T1 harness; enrich with raw status ``depth`` /
    ``available_slots`` and ``Fallback triggered`` lines; compute metrics."""
    try:
        loaded = cs.load_events(log_files, start=start, end=end)
    except FileNotFoundError:
        raise

    status_depth: dict[float, int] = {}
    available: list[int] = []
    fallback_triggered: list[dict] = []
    for pat in log_files:
        for path_str in _glob.glob(str(pat)):
            for line in iter_lines(Path(path_str)):
                ts = cs._parse_ts(line)
                if ts is None:
                    continue
                if start is not None and ts < start:
                    continue
                if end is not None and ts >= end:
                    continue
                if _MARKER_FALLBACK_TRIGGERED in line:
                    f = parse_fallback_triggered(line)
                    if f is not None:
                        fallback_triggered.append(f)
                    continue
                if "status_request" in line:
                    m = _DEPTH_RE.search(line)
                    if m:
                        status_depth[ts] = int(m.group(1))
                        # available_slots is only meaningful on the same
                        # cheap-tier snapshot stream; status lines without
                        # contention_queue_depth are other tiers (e.g. the
                        # mxbai-embed embedding pool).
                        m2 = _SLOTS_RE.search(line)
                        if m2:
                            available.append(int(m2.group(1)))

    # Actual span of the queue-path event stream (dispatch / fallback /
    # denied / slot-free), excluding spillover lines that only carry
    # snapshots or fallback-triggered markers.
    ev_ts = [e.ts for e in loaded.events]
    ev_start = min(ev_ts) if ev_ts else None
    ev_end = max(ev_ts) if ev_ts else None

    return compute_profile(
        loaded.events, status_depth, available, fallback_triggered,
        window_start=loaded.window_start or ev_start,
        window_end=loaded.window_end or ev_end,
        files=loaded.files, burst_gap_seconds=burst_gap_seconds,
        min_arrivals=min_arrivals, snapshot_delta=snapshot_delta,
    )


# ---------------------------------------------------------------------------
# Serialization / report
# ---------------------------------------------------------------------------


def serialize(res: ProfileResult) -> dict:
    g = sorted(res.gaps)
    dw = sorted(res.dispatch_waits)
    return {
        "window": {"start": _fmt(res.window_start), "end": _fmt(res.window_end),
                   "files": res.files},
        "arrivals": res.arrivals,
        "inter_arrival_gaps_seconds": {
            "n": len(res.gaps), "median": round(_pct(g, 50), 2),
            "p90": round(_pct(g, 90), 2), "p99": round(_pct(g, 99), 2),
            "max": round(g[-1], 2) if g else 0.0,
        },
        "bursts": {
            "definition": (
                f"runs of full-occupancy contention events (denied + "
                f"queue-path arrivals), inter-event gap <= "
                f"{DEFAULT_BURST_GAP_SECONDS:.0f}s, >= "
                f"{DEFAULT_BURST_MIN_ARRIVALS} events"
            ),
            "count": len(res.bursts),
            "arrivals_per_burst": _dist_stats([b[2] for b in res.bursts]),
            "burst_duration_seconds": _duration_stats(res.bursts),
        },
        "concurrency": {
            "denied_active_inside_bursts": _dist_stats(res.denied_active),
            "available_slots": _dist_stats(res.available_slots),
        },
        "queue_depth": {
            "status_snapshot": _dist_stats(res.status_depth),
            "dispatch_after_pop": _dist_stats(res.dispatch_depth_after),
        },
        "queue_wait_durations_seconds": {
            "dispatched": {"n": len(res.dispatch_waits),
                           "median": round(_pct(dw, 50), 2),
                           "p90": round(_pct(dw, 90), 2),
                           "p95": round(_pct(dw, 95), 2),
                           "max": round(dw[-1], 2) if dw else 0.0},
            "fallback_after_queue": {
                "n": len(res.fallback_waits),
                "values": sorted(set(res.fallback_waits)),
            },
        },
        "fallbacks_while_queue_nonempty": {
            "fallback_after_queue": {
                "total": res.fallback_after_queue_total,
                **res.fallback_after_queue_split,
            },
            "fallback_triggered_cheap": {
                "total": res.fallback_triggered_total,
                **res.fallback_triggered_split,
            },
            "snapshot_delta_seconds": DEFAULT_SNAPSHOT_DELTA_SECONDS,
        },
    }


def _duration_stats(bursts: list[tuple[float, float, int]]) -> dict:
    if not bursts:
        return {"n": 0, "median": 0.0, "p90": 0.0, "max": 0.0}
    d = sorted(b[1] - b[0] for b in bursts)
    return {"n": len(d), "median": round(_pct(d, 50), 2),
            "p90": round(_pct(d, 90), 2), "max": round(d[-1], 2)}


def render_markdown(res: ProfileResult) -> str:
    s = serialize(res)
    fb = s["fallbacks_while_queue_nonempty"]
    lines = [
        "# Contention-Queue Burst Profile (cheap-mode fan-out)",
        "",
        f"**Window:** {s['window']['start'] or 'start'} -> {s['window']['end'] or 'end'}",
        f"**Files:** {len(s['window']['files'])}",
        "**Work item:** LP-0MTF0G4VH003DXB9 (T2 of LP-0MTED3OFP006I7NO)",
        "",
        "## 1. Concurrent local requests during bursts",
        "",
        f"- Full-occupancy contention events (denied + queue-path arrivals): "
        f"**{s['arrivals']}**",
        f"- Inter-arrival gaps: median **{s['inter_arrival_gaps_seconds']['median']}s**, "
        f"p90 **{s['inter_arrival_gaps_seconds']['p90']}s**, "
        f"p99 **{s['inter_arrival_gaps_seconds']['p99']}s**",
        f"- Burst definition: {s['bursts']['definition']}",
        f"- Bursts detected: **{s['bursts']['count']}**",
        f"- Arrivals per burst: median {s['bursts']['arrivals_per_burst']['p50']}, "
        f"max {s['bursts']['arrivals_per_burst']['max']}",
        f"- Burst duration: median {s['bursts']['burst_duration_seconds']['median']}s, "
        f"max {s['bursts']['burst_duration_seconds']['max']}s",
        "",
        "`local_active_queries` (from `local_dispatch_denied active=`, sampled inside bursts):",
        "",
        _render_table("value", s["concurrency"]["denied_active_inside_bursts"]["histogram"]),
        "",
        "`available_slots` (status_request snapshots):",
        "",
        _render_table("value", s["concurrency"]["available_slots"]["histogram"]),
        "",
        "## 2. Queue depth observed",
        "",
        "`contention_queue_depth` (status_request snapshots):",
        "",
        _render_table("depth", s["queue_depth"]["status_snapshot"]["histogram"]),
        "",
        "`depth` after pop (contention_queue_dispatch lines):",
        "",
        _render_table("depth", s["queue_depth"]["dispatch_after_pop"]["histogram"]),
        "",
        "## 3. Queue-wait durations",
        "",
        "Dispatched (`queued_duration` on contention_queue_dispatch):",
        "",
        f"- n={s['queue_wait_durations_seconds']['dispatched']['n']}, "
        f"median {s['queue_wait_durations_seconds']['dispatched']['median']}s, "
        f"p90 {s['queue_wait_durations_seconds']['dispatched']['p90']}s, "
        f"p95 {s['queue_wait_durations_seconds']['dispatched']['p95']}s, "
        f"max {s['queue_wait_durations_seconds']['dispatched']['max']}s",
        "",
        "Fell back (`queued_duration` on contention_queue_fallback_after_queue):",
        "",
        f"- n={s['queue_wait_durations_seconds']['fallback_after_queue']['n']}, "
        f"values {s['queue_wait_durations_seconds']['fallback_after_queue']['values']} "
        f"(== wait cap)",
        "",
        "## 4. Fallbacks while the queue was non-empty",
        "",
        f"- `contention_queue_fallback_after_queue` "
        f"({res.fallback_after_queue_total} total): "
        f"{res.fallback_after_queue_split['gt0']} with queue depth > 0, "
        f"{res.fallback_after_queue_split['eq0']} with queue depth 0, "
        f"{res.fallback_after_queue_split['no_snapshot']} without a snapshot "
        f"within {fb['snapshot_delta_seconds']:.0f}s",
        f"- `Fallback triggered reason=local_concurrency_limit` "
        f"({res.fallback_triggered_total} total): "
        f"{res.fallback_triggered_split['gt0']} with queue depth > 0, "
        f"{res.fallback_triggered_split['eq0']} with queue depth 0, "
        f"{res.fallback_triggered_split['no_snapshot']} without a snapshot "
        f"within {fb['snapshot_delta_seconds']:.0f}s",
        "",
        "## Methodology",
        "",
        "- Script: `python3 proxy/benchmarks/contention_queue_profile.py "
        "[--log-files ...] [--start ...] [--end ...] --report "
        "docs/dev/contention-queue-burst-profile.md`",
        "- Parsing: T1 harness `contention_queue_simulation.load_events`; "
        "status `contention_queue_depth`/`available_slots` and "
        "`Fallback triggered` lines parsed by this script",
        "- `available_slots` is collected only from cheap-tier snapshot lines "
        "(those carrying `contention_queue_depth`); status lines from other "
        "pools (e.g. the mxbai-embed embedding tier) are excluded",
        "- Metric 4 splits events by the nearest status snapshot's queue depth "
        "within +-30s: `gt0` (queue non-empty), `eq0` (verified empty), "
        "`no_snapshot` (not derivable — no snapshot within the delta; these "
        "fallbacks occur in hours without cheap-tier snapshot traffic)",
        "",
    ]
    return "\n".join(lines)


def _render_table(label: str, hist: dict) -> str:
    if not hist:
        return "_(no samples)_"
    rows = [f"| {label} | count |", "|---|---|"]
    rows += [f"| {k} | {v} |" for k, v in sorted(hist.items())]
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Quantify the cheap-mode burst profile (T2 contention "
                    "queue analysis) from proxy logs."
    )
    p.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    p.add_argument("--pattern", default=DEFAULT_LOG_PATTERN)
    p.add_argument("--log-files", nargs="*", default=None)
    p.add_argument("--start", type=cs.parse_iso, default=None)
    p.add_argument("--end", type=cs.parse_iso, default=cs.parse_iso(DEFAULT_END))
    p.add_argument("--burst-gap", type=float, default=DEFAULT_BURST_GAP_SECONDS)
    p.add_argument("--burst-min-arrivals", type=int, default=DEFAULT_BURST_MIN_ARRIVALS)
    p.add_argument("--snapshot-delta", type=float, default=DEFAULT_SNAPSHOT_DELTA_SECONDS)
    p.add_argument("--report", default=DEFAULT_REPORT,
                   help="output markdown report path "
                        "(default: docs/dev/contention-queue-burst-profile.md)")
    p.add_argument("--json", action="store_true", help="also print JSON report")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_files = args.log_files or [str(Path(args.log_dir) / args.pattern)]
    res = load_profile(log_files, start=args.start, end=args.end,
                       burst_gap_seconds=args.burst_gap,
                       min_arrivals=args.burst_min_arrivals,
                       snapshot_delta=args.snapshot_delta)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(res))

    s = serialize(res)
    print("--- contention-queue burst profile ---")
    print(f"files: {len(res.files)}  window: {_fmt(res.window_start)} .. {_fmt(res.window_end)}")
    print(f"arrivals: {res.arrivals}  bursts: {len(res.bursts)}")
    g = s["inter_arrival_gaps_seconds"]
    print(f"inter-arrival: median={g['median']}s p90={g['p90']}s p99={g['p99']}s")
    print(f"denied active (inside bursts): "
          f"{s['concurrency']['denied_active_inside_bursts']['histogram']}")
    print(f"status queue depth: {s['queue_depth']['status_snapshot']['histogram']}")
    dw = s["queue_wait_durations_seconds"]["dispatched"]
    print(f"dispatch waits: n={dw['n']} median={dw['median']}s p90={dw['p90']}s "
          f"p95={dw['p95']}s max={dw['max']}s")
    print(f"fallbacks while queue non-empty: fb_after_queue "
          f"{res.fallback_after_queue_split['gt0']}/"
          f"{res.fallback_after_queue_total} (eq0: "
          f"{res.fallback_after_queue_split['eq0']}, no_snapshot: "
          f"{res.fallback_after_queue_split['no_snapshot']})")
    ft = res.fallback_triggered_split
    print(f"fallback_triggered(local_concurrency_limit): "
          f"{res.fallback_triggered_total} total (gt0: {ft['gt0']}, "
          f"eq0: {ft['eq0']}, no_snapshot: {ft['no_snapshot']})")
    print(f"report written: {report_path}")

    if args.json:
        print("\n" + json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
