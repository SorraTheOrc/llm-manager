#!/usr/bin/env python3
"""Mode-flip classifier for the llama-proxy logs (LP-0MV9UNE75004903J).

Investigation deliverable: quantify how often the llama-proxy flips between
**fast** and **cheap** operating modes, split by trigger source, and whether
the studied 24 h window was unusual.

Attribution model (see
``docs/mode-flip-analysis-LP-0MV9UNE75004903J.md``):

- A **flip** is a real fast<->cheap transition. The proxy skips redundant
  switches (``set_mode`` returns a no-op when the persisted mode already
  matches), so every "Mode-switch restart spawned" == one real transition.
- The *active mode at startup* is read from the
  ``Grandfathering: enabled; other-mode config config-<other>.yaml
  (current=<mode>)`` line that follows each "Starting LLama Proxy Server"
  marker. A flip is a change of startup mode between consecutive restarts.
- Trigger classification (deterministic, ``source``-based since
  LP-0MVAPNC2900145RH):

  The proxy now logs attribution markers on restart and mode-switch lines
  (``source=``/``actor=``/``reason=``). The ``classify_trigger()`` function
  uses these markers when present, falling back to heuristic classification
  for historical unattributed lines:

  | ``source`` value | Flip direction | Classification |
  |------------------|----------------|----------------|
  | ``scheduler``    | any            | ``schedule``   |
  | ``worker``       | to-fast        | ``to-fast``    |
  | ``worker``       | to-cheap       | ``to-cheap``   |
  | ``api``          | to-fast        | ``to-fast``    |
  | ``api``          | to-cheap       | ``to-cheap``   |
  | ``manual``       | to-fast        | ``to-fast``    |
  | ``manual``       | to-cheap       | ``to-cheap``   |
  | *absent*         | any            | Heuristic fallback (sched_marker / direction-based) |

  Prior to attribution markers, classification used these heuristics:
  - **schedule**: a "Mode scheduler: applied scheduled mode <mode>" marker
    within +/-10 min of the flip (the built-in 01:00 cheap / 10:00 fast
    schedule). Markers appear only when the scheduled mode *differs* from the
    persisted mode, so a scheduled boundary already satisfied by an
    activity-driven switch in the interim is a no-op and never logged.
  - **fast-on-command / manual**: any other flip INTO fast. The ContextHub
    ``mode-switch-worker.ts`` fire-and-forget POSTs ``/admin/set-mode
    {"mode":"fast"}`` on every agent-route command; an operator may also POST
    directly.
  - **idle-cheap / manual**: any other flip INTO cheap. The worker (leader
    only) POSTs ``cheap`` when the operator has been idle >= 15 min AND the
    proxy reports idle AND no model cooldown is active.

  Without markers the proxy did not log the ``/admin/set-mode`` caller, so
  worker vs manual was not separable from the proxy log alone (the original
  report argued worker dominance from the timing signature — median inter-flip
  gap == the 30-min fast->cheap cooldown — and the shared
  ``mode-activity.json`` broadcast). With the attribution markers those two
  sources are now distinguished directly.

The pure functions below (``parse_event`` .. ``summarise``) are unit-tested in
``tests/test_mode_flip_analysis.py``; ``main`` only does I/O and rendering.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Allow both ``python3 scripts/analyze_mode_flips.py`` (cwd=root) and import
# from the test suite.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.proxy_logs import iter_proxy_log_lines

# ── Log-line parsing ─────────────────────────────────────────────────

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
_START_RE = re.compile(r"Starting LLama Proxy Server")
_SPAWN_RE = re.compile(r"Mode-switch restart spawned")
_SCHED_RE = re.compile(r"Mode scheduler: applied scheduled mode (\w+)")
_MODE_RE = re.compile(
    r"other-mode config config-(\w+)\.yaml \(current=(\w+)\)"
)
#: Attribution marker ``source=<value>`` appended to scheduler / grandfathering /
#: restart-spawned lines (LP-0MVC1450O000LJHV, LP-0MVAPNC2900145RH).  Values
#: are lower-case tokens.
_RE_ATTRIBUTION_SOURCE = re.compile(r"\bsource=([a-z0-9_]+)")

#: A startup mode line follows its start marker within this many seconds.
MODE_ATTACH_WINDOW_SECONDS = 30
#: A schedule marker is attributed to a restart within this many seconds.
SCHEDULE_ATTACH_WINDOW_SECONDS = 600
#: The proxy's fast->cheap cooldown (MODE_SWITCH_COOLDOWN_SECONDS).
COOLDOWN_SECONDS = 30 * 60
#: Boundary band: the restart marker lags the set_mode call by ~10-15 s.
COOLDOWN_BOUNDARY_MAX_MINUTES = 30.5

EVENT_KINDS = ("start", "spawn", "sched", "mode")
#: Chronological tie-break: spawn precedes the start it causes, which
#: precedes that process's own "current=<mode>" line.
KIND_PRIORITY = {"spawn": 0, "start": 1, "mode": 2, "sched": 3}


def parse_timestamp(line: str) -> datetime | None:
    """Return the line's timestamp, or None when the line is not timestamped."""
    m = _TS_RE.match(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")


def parse_event(line: str) -> tuple[str, object, str | None] | None:
    """Classify a log line into ``(kind, payload, source)`` or None.

    Kinds: ``start`` (process start), ``spawn`` (mode-switch restart spawned),
    ``sched`` (scheduler applied a scheduled mode -> payload is the mode),
    ``mode`` (startup active mode -> payload is the current mode).

    *source* is the attribution marker (``scheduler`` / ``worker`` / ``api`` /
    ``manual``) when present on the line, else ``None``.
    """
    if _START_RE.search(line):
        return "start", None, None
    if _SPAWN_RE.search(line):
        src = _RE_ATTRIBUTION_SOURCE.search(line)
        return "spawn", None, src.group(1) if src else None
    if m := _SCHED_RE.search(line):
        src = _RE_ATTRIBUTION_SOURCE.search(line)
        return "sched", m.group(1), src.group(1) if src else None
    if m := _MODE_RE.search(line):
        src = _RE_ATTRIBUTION_SOURCE.search(line)
        return "mode", m.group(2), src.group(1) if src else None
    return None


def collect_events(log_dir: Path) -> list[tuple[datetime, str, object, str | None]]:
    """Parse every event from the proxy logs under *log_dir*, time-sorted.

    Uses the shared ``lib.proxy_logs`` discovery/opening helper so both
    rotation schemes (``proxy.log.*`` / ``proxy.log-*``) and gzip compression
    are handled. Untimestamped or non-matching lines are skipped.

    Returns ``(ts, kind, payload, source)`` tuples; *source* is the
    attribution marker when present, else ``None``.
    """
    events: list[tuple[datetime, str, object, str | None]] = []
    for _path, line in iter_proxy_log_lines(log_dir):
        ts = parse_timestamp(line)
        if ts is None:
            continue
        parsed = parse_event(line)
        if parsed is None:
            continue
        kind, payload, source = parsed
        events.append((ts, kind, payload, source))
    events.sort(key=lambda e: (e[0], KIND_PRIORITY[e[1]]))
    return events


def build_restarts(
    events: list[tuple[datetime, str, object, str | None]],
) -> list[dict]:
    """Group events into restart records ``{ts, mode, sched_marker, source}``.

    A ``start`` event opens a record; the first ``mode`` event within
    ``MODE_ATTACH_WINDOW_SECONDS`` fills its ``mode``. Records without a mode
    (window-boundary truncation) keep ``mode=None``. The *source* attribution
    (``scheduler`` / ``worker`` / ``api`` / ``manual``) is carried through from
    the ``mode`` event when present (LP-0MVC1450O000LJHV, LP-0MVAPNC2900145RH).
    """
    restarts: list[dict] = []
    for ts, kind, payload, source in events:
        if kind == "start":
            restarts.append({"ts": ts, "mode": None, "sched_marker": None, "source": None})
        elif kind == "mode" and restarts:
            rec = restarts[-1]
            if rec["mode"] is None and (ts - rec["ts"]).total_seconds() <= (
                MODE_ATTACH_WINDOW_SECONDS
            ):
                rec["mode"] = payload
                rec["source"] = source
    return restarts


def attach_schedule_markers(
    restarts: list[dict],
    scheds: list[tuple[datetime, str]],
) -> None:
    """Attach each schedule marker to the restart it caused (in place).

    A marker is logged by the scheduler tick immediately before the restart
    spawn, so it belongs to the *next* restart within
    ``SCHEDULE_ATTACH_WINDOW_SECONDS``. Markers and restarts are matched
    one-to-one by nearest timestamp so a marker is never attached twice.
    """
    used: set[int] = set()
    for rec in restarts:
        best_idx: int | None = None
        best_delta: float | None = None
        for idx, (mts, _mode) in enumerate(scheds):
            if idx in used:
                continue
            delta = abs((rec["ts"] - mts).total_seconds())
            if delta <= SCHEDULE_ATTACH_WINDOW_SECONDS and (
                best_delta is None or delta < best_delta
            ):
                best_delta = delta
                best_idx = idx
        if best_idx is not None:
            used.add(best_idx)
            rec["sched_marker"] = scheds[best_idx][1]
            # When the schedule marker itself carries source=scheduler, also
            # record it on the restart for classify_trigger().
            if rec.get("source") is None and scheds[best_idx][1] == "scheduler":
                rec["source"] = "scheduler"


def compute_flips(restarts: list[dict]) -> list[dict]:
    """Return flips between consecutive restarts whose startup mode differs."""
    known = [r for r in restarts if r["mode"]]
    flips: list[dict] = []
    for prev, cur in zip(known, known[1:]):
        if prev["mode"] != cur["mode"]:
            flips.append(
                {
                    "at": cur["ts"],
                    "prev_from": prev["mode"],
                    "to": cur["mode"],
                    "sched_marker": cur.get("sched_marker"),
                    "source": cur.get("source"),
                }
            )
    return flips


def classify_trigger(flip: dict) -> str:
    """Classify a flip using deterministic attribution when available.

    Uses the ``source`` field (LP-0MVC1450O000LJHV, LP-0MVAPNC2900145RH) when
    present:

    - ``source=scheduler`` → ``schedule``
    - ``source=worker`` → ``to-cheap`` when flipping to cheap, ``to-fast`` otherwise
    - ``source=api`` → ``to-fast`` when flipping to fast, ``to-cheap`` otherwise
    - ``source=manual`` → ``to-fast`` when flipping to fast, ``to-cheap`` otherwise

    Falls back to the existing heuristic (sched_marker / direction-based) when
    no attribution is available.
    """
    source = flip.get("source")
    if source == "scheduler":
        return "schedule"
    if source in ("worker", "api", "manual"):
        return "to-fast" if flip["to"] == "fast" else "to-cheap"
    # Heuristic fallback: sched_marker or direction-based classification.
    if flip.get("sched_marker"):
        return "schedule"
    return "to-fast" if flip["to"] == "fast" else "to-cheap"


def _dwells(restarts: list[dict], mode: str) -> list[tuple[datetime, float, datetime]]:
    """Dwell intervals (start, minutes, next-change) spent in *mode*."""
    known = [r for r in restarts if r["mode"]]
    out: list[tuple[datetime, float, datetime]] = []
    for a, b in zip(known, known[1:]):
        if a["mode"] == mode and b["mode"] != mode:
            out.append((a["ts"], (b["ts"] - a["ts"]).total_seconds() / 60.0, b["ts"]))
    return out


def summarise(
    restarts: list[dict],
    flips: list[dict],
    window: tuple[datetime, datetime] | None = None,
) -> dict:
    """Build the machine-readable analysis summary.

    *window* optionally restricts the studied-window counts to
    ``[from, to)``.
    """
    by_day: dict[str, Counter] = defaultdict(Counter)
    for f in flips:
        by_day[f["at"].date().isoformat()][classify_trigger(f)] += 1
    trigger_totals: Counter = Counter()
    for day in by_day:
        trigger_totals.update(by_day[day])

    by_hour = Counter(f["at"].hour for f in flips)

    gaps = [
        (b["at"] - a["at"]).total_seconds() / 60.0
        for a, b in zip(flips, flips[1:])
    ]
    cooldown_breaches = [
        b["at"].isoformat()
        for a, b in zip(flips, flips[1:])
        if b["to"] == "cheap" and (b["at"] - a["at"]).total_seconds() < (
            COOLDOWN_SECONDS - 60
        )
    ]
    cooldown_boundary = sum(
        1
        for a, b in zip(flips, flips[1:])
        if b["to"] == "cheap"
        and COOLDOWN_SECONDS / 60.0 - 1.0
        <= (b["at"] - a["at"]).total_seconds() / 60.0
        <= COOLDOWN_BOUNDARY_MAX_MINUTES
    )

    cheap_dwells = _dwells(restarts, "cheap")
    wasted_threshold_minutes = 5.0
    wasted = [d for d in cheap_dwells if d[1] < wasted_threshold_minutes]
    fast_dwells = _dwells(restarts, "fast")

    def _pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        idx = min(len(values) - 1, int(p * len(values)))
        return sorted(values)[idx]

    summary: dict = {
        "restarts_with_mode": len([r for r in restarts if r["mode"]]),
        "flips": len(flips),
        "trigger_totals": dict(trigger_totals),
        "per_day": {d: dict(by_day[d]) for d in sorted(by_day)},
        "by_hour": {h: by_hour.get(h, 0) for h in range(24)},
        "gap_minutes": {
            "n": len(gaps),
            "median": round(_median(gaps), 1) if gaps else 0.0,
            "p25": round(_pct(gaps, 0.25), 1) if gaps else 0.0,
            "p75": round(_pct(gaps, 0.75), 1) if gaps else 0.0,
            "max": round(max(gaps), 1) if gaps else 0.0,
        },
        "cooldown_breaches": cooldown_breaches,
        "cooldown_boundary_hits": cooldown_boundary,
        "cheap_dwells": {
            "n": len(cheap_dwells),
            "median": round(_median([d[1] for d in cheap_dwells]), 1)
            if cheap_dwells
            else 0.0,
            "wasted_under_5m": len(wasted),
        },
        "fast_dwells": {
            "n": len(fast_dwells),
            "median": round(_median([d[1] for d in fast_dwells]), 1)
            if fast_dwells
            else 0.0,
            "min": round(min((d[1] for d in fast_dwells), default=0.0), 1),
        },
    }
    if window is not None:
        w0, w1 = window
        summary["studied_window"] = {
            "from": w0.isoformat(),
            "to": w1.isoformat(),
            "starts": len([r for r in restarts if w0 <= r["ts"] < w1]),
            "flips": len([f for f in flips if w0 <= f["at"] < w1]),
            "by_trigger": dict(
                Counter(classify_trigger(f) for f in flips if w0 <= f["at"] < w1)
            ),
            "to_fast": [f["at"].isoformat() for f in flips if w0 <= f["at"] < w1 and f["to"] == "fast"],
            "to_cheap": [f["at"].isoformat() for f in flips if w0 <= f["at"] < w1 and f["to"] == "cheap"],
            "flip_list": [
                {
                    "at": f["at"].isoformat(),
                    "from": f["prev_from"],
                    "to": f["to"],
                    "trigger": classify_trigger(f),
                }
                for f in flips
                if w0 <= f["at"] < w1
            ],
        }
    return summary


def filter_restarts(
    restarts: list[dict],
    start: datetime | None,
    end: datetime | None,
) -> list[dict]:
    """Restrict restart records to ``[start, end)`` (either may be None)."""
    return [
        r
        for r in restarts
        if (start is None or r["ts"] >= start) and (end is None or r["ts"] < end)
    ]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


# ── Rendering ────────────────────────────────────────────────────────

def render(summary: dict) -> str:
    """Render the human-readable report for *summary*."""
    lines: list[str] = []
    lines.append(
        f"Restarts with mode: {summary['restarts_with_mode']}   "
        f"Flips: {summary['flips']}   "
        f"(restarts_with_mode - flips <= 1 == continuous window)"
    )
    t = summary["trigger_totals"]
    lines.append(
        f"Trigger totals: schedule={t.get('schedule', 0)} "
        f"to-fast={t.get('to-fast', 0)} to-cheap={t.get('to-cheap', 0)}"
    )
    lines.append("")
    lines.append("Per-day flips by trigger:")
    lines.append(f"{'Date':<12}{'total':>6}{'schedule':>10}{'to-fast':>10}{'to-cheap':>11}")
    for day, counts in summary["per_day"].items():
        tot = sum(counts.values())
        lines.append(
            f"{day:<12}{tot:>6}{counts.get('schedule', 0):>10}"
            f"{counts.get('to-fast', 0):>10}{counts.get('to-cheap', 0):>11}"
        )
    lines.append("")
    lines.append("Flips by hour (UTC):")
    for h in range(24):
        n = summary["by_hour"][h]
        lines.append(f"  {h:02d}:00 {n:>3} {'#' * n}")
    g = summary["gap_minutes"]
    lines.append("")
    lines.append(
        f"Inter-flip gap (min): n={g['n']} median={g['median']} "
        f"p25={g['p25']} p75={g['p75']} max={g['max']}"
    )
    lines.append(
        f"Cooldown breaches (<29 min fast->cheap): "
        f"{len(summary['cooldown_breaches'])}; "
        f"boundary-band hits [29,30.5]: {summary['cooldown_boundary_hits']}"
    )
    cd = summary["cheap_dwells"]
    fd = summary["fast_dwells"]
    lines.append(
        f"Cheap dwells: n={cd['n']} median={cd['median']}m "
        f"wasted(<5m)={cd['wasted_under_5m']} ({100*cd['wasted_under_5m']//max(cd['n'],1)}%)"
    )
    lines.append(
        f"Fast dwells: n={fd['n']} median={fd['median']}m min={fd['min']}m "
        f"(>= ~30m cooldown)"
    )
    if "studied_window" in summary:
        sw = summary["studied_window"]
        lines.append("")
        lines.append(
            f"Studied window {sw['from']} -> {sw['to']}: "
            f"starts={sw['starts']} flips={sw['flips']} by_trigger={sw['by_trigger']}"
        )
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────

DEFAULT_LOG_DIR = Path("/var/log/llama-proxy")
#: The originally investigated window (2026-10-14 14:02 -> 2026-10-15 14:02 UTC).
STUDIED_WINDOW = (
    datetime(2026, 10, 14, 14, 2),
    datetime(2026, 10, 15, 14, 2),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--json", action="store_true", help="emit JSON summary")
    parser.add_argument(
        "--window",
        default=f"{STUDIED_WINDOW[0]:%Y-%m-%d %H:%M},{STUDIED_WINDOW[1]:%Y-%m-%d %H:%M}",
        help="studied window 'START,END' (UTC), or 'none' to skip",
    )
    parser.add_argument(
        "--from",
        dest="from_ts",
        default=None,
        help="restrict the dataset to restarts at/after this 'YYYY-MM-DD HH:MM' UTC",
    )
    parser.add_argument(
        "--to",
        dest="to_ts",
        default=None,
        help="restrict the dataset to restarts before this 'YYYY-MM-DD HH:MM' UTC",
    )
    args = parser.parse_args(argv)

    events = collect_events(args.log_dir)
    restarts = build_restarts(events)
    scheds = [(ts, p) for ts, kind, p in events if kind == "sched"]
    attach_schedule_markers(restarts, scheds)

    def _parse_opt(value: str | None) -> datetime | None:
        return datetime.strptime(value.strip(), "%Y-%m-%d %H:%M") if value else None

    date_from, date_to = _parse_opt(args.from_ts), _parse_opt(args.to_ts)
    restarts = filter_restarts(restarts, date_from, date_to)
    flips = compute_flips(restarts)

    window = None
    if args.window and args.window.strip().lower() != "none":
        start_s, _, end_s = args.window.partition(",")
        window = (
            datetime.strptime(start_s.strip(), "%Y-%m-%d %H:%M"),
            datetime.strptime(end_s.strip(), "%Y-%m-%d %H:%M"),
        )

    summary = summarise(restarts, flips, window=window)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(render(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
