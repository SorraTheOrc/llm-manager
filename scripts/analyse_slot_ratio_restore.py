#!/usr/bin/env python3
"""Measure restored-slot-prefix warmness (LP-0MVGGPSVM003W35K).

LP-0MVGD9YXH008O304 made a session whose KV prefix is already in a persisted
slot snapshot score as warm at routing time.  Its instrumentation is:

* ``routing_check`` / ``routing_skip_local`` / ``routing_economic_bypass_local``
  carry ``cache_state=cold|warm|restored`` (``restored`` = ratio seeded from a
  snapshot sidecar and above the warm threshold);
* ``slot_ratio_restored session=… cached_tokens=… estimated_tokens=…
  cached_ratio=…`` is logged once per successful seed;
* local ``dispatch_first_byte_ms=… cache_state=…`` records the prefill time of
  every local serve.

This script extracts those metrics for a set of windows so a before/after
comparison is reproducible instead of hand-grepped:

* share of routing decisions with ``cached_ratio=0.00``,
* ``large_context_bypass`` (and other local-skip) counts,
* local serve share (local / all ``Stream started`` lines),
* local ``dispatch_first_byte_ms`` p10/p50/p90/max, per ``cache_state``,
* ``cache_state=restored`` dispatches counted separately from ``warm``/``cold``,
* ``slot_ratio_restored`` seed events (the only direct evidence that the
  restored-prefix credit was applied),
* ``slot_save`` success/failure and ``slot_restore failed`` rates (AC4
  confound check), and
* the ``slot_<session>.bin.ratio.json`` sidecar inventory on disk.

Log files may be compressed (``.gz``); every matched file is read exactly once,
so no line is double-counted and no compressed history is skipped (the classic
``cat proxy.log*`` extraction silently drops every ``.gz`` file that follows a
NUL byte — that is why this script uses :func:`read_log_lines`).

Usage:
  ./scripts/analyse_slot_ratio_restore.py                  # default windows
  ./scripts/analyse_slot_ratio_restore.py --json           # machine readable
  ./scripts/analyse_slot_ratio_restore.py --start '2026-10-19 07:23:12' \\
      --end '2026-10-20 07:23:12'                              # one window
  ./scripts/analyse_slot_ratio_restore.py --window baseline=2026-10-19T07:23:12,2026-10-20T07:23:12
  ./scripts/analyse_slot_ratio_restore.py \\
      --slot-cache-dir ./slot-cache                         # + sidecar inventory

Exit codes:
  0 - success (even when a window contains no samples)
  1 - log directory missing / unexpected error
  2 - invalid arguments (bad window specification)
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LOG_DIR = Path("/var/log/llama-proxy")

#: First sidecar written by LP-0MVGD9YXH008O304 (the feature's deploy marker).
DEFAULT_DEPLOY_START = "2026-10-20 13:27:34"

#: The window the intake baselines were measured over.
DEFAULT_BASELINE_START = "2026-10-19 07:23:12"
DEFAULT_BASELINE_END = "2026-10-20 07:23:12"

#: Context buckets used for the like-for-like prefill comparison (AC3).
DEFAULT_BUCKETS = (38_000, 120_000)

RE_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
RE_SESSION = re.compile(r"session=(\S+)")
RE_ESTIMATED = re.compile(r"estimated_tokens=(\d+)")
RE_CACHED_RATIO = re.compile(r"cached_ratio=([\d.]+)")
RE_CACHE_STATE = re.compile(r"cache_state=(\w+)")
RE_DISPATCH = re.compile(r"dispatch_first_byte_ms=([\d.]+)")
RE_REASON = re.compile(r"reason=(\w+)")


def read_log_lines(path: Path):
    """Yield decoded lines from *path*, transparently decompressing ``.gz``."""
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            yield from fh
        return
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        yield from fh


def parse_ts(text: str) -> dt.datetime | None:
    """Parse a proxy log timestamp (``YYYY-mm-dd HH:MM:SS``)."""
    try:
        return dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def _percentiles(values: list[float]) -> dict | None:
    """Return count/p10/p50/p90/max/sum for *values* (linear interpolation)."""
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)

    def at(p: float) -> float:
        idx = p * (n - 1)
        lo, hi = math.floor(idx), math.ceil(idx)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)

    return {
        "count": n,
        "p10": round(at(0.10), 2),
        "p50": round(at(0.50), 2),
        "p90": round(at(0.90), 2),
        "max": round(ordered[-1], 2),
        "sum_hours": round(sum(ordered) / 3600.0, 3),
    }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@dataclass
class WindowMetrics:
    """All metrics collected for one analysis window."""

    name: str
    start: dt.datetime
    end: dt.datetime
    lines: int = 0
    stream_started: int = 0
    stream_started_local: int = 0
    routing_check: int = 0
    routing_check_ratio_zero: int = 0
    routing_ratio_sum: float = 0.0
    routing_state: Counter = field(default_factory=Counter)
    skip_local: int = 0
    economic_bypass: int = 0
    skip_reasons: Counter = field(default_factory=Counter)
    dispatches: list[float] = field(default_factory=list)
    dispatches_by_state: dict[str, list[float]] = field(default_factory=dict)
    # (timestamp, session, seconds, cache_state) so the AC3 like-for-like
    # bucketing can pair each dispatch with its routing estimate.
    dispatch_records: list[tuple] = field(default_factory=list)
    slot_save_ok: int = 0
    slot_save_failed: int = 0
    slot_restore_failed: int = 0
    slot_ratio_restored: int = 0
    restored_any: int = 0
    # session -> [(ts, estimated_tokens)] for the AC3 bucket pairing
    routing_estimates: dict[str, list[tuple[dt.datetime, int]]] = field(default_factory=dict)

    def add_line(self, line: str, ts: dt.datetime) -> None:
        self.lines += 1
        if "routing_check " in line:
            self.routing_check += 1
            ratio = RE_CACHED_RATIO.search(line)
            if ratio:
                value = float(ratio.group(1))
                self.routing_ratio_sum += value
                if value == 0.0:
                    self.routing_check_ratio_zero += 1
            state = RE_CACHE_STATE.search(line)
            if state:
                self.routing_state[state.group(1)] += 1
                if state.group(1) == "restored":
                    self.restored_any += 1
            est = RE_ESTIMATED.search(line)
            session = RE_SESSION.search(line)
            if est and session:
                self.routing_estimates.setdefault(session.group(1), []).append((ts, int(est.group(1))))
            return
        if "routing_skip_local " in line or "routing_economic_bypass_local " in line:
            if "routing_skip_local " in line:
                self.skip_local += 1
            else:
                self.economic_bypass += 1
            reason = RE_REASON.search(line)
            if reason:
                self.skip_reasons[reason.group(1)] += 1
            state = RE_CACHE_STATE.search(line)
            if state and state.group(1) == "restored":
                self.restored_any += 1
            return
        if "dispatch_first_byte_ms=" in line:
            dispatch = RE_DISPATCH.search(line)
            if dispatch:
                seconds = float(dispatch.group(1)) / 1000.0
                self.dispatches.append(seconds)
                state = RE_CACHE_STATE.search(line)
                key = state.group(1) if state else "unknown"
                self.dispatches_by_state.setdefault(key, []).append(seconds)
                session = RE_SESSION.search(line)
                self.dispatch_records.append((ts, session.group(1) if session else None, seconds, key))
            return
        if "Stream started: provider=local" in line:
            self.stream_started += 1
            self.stream_started_local += 1
            return
        if "Stream started: provider=" in line:
            self.stream_started += 1
            return
        if "slot_save success" in line:
            self.slot_save_ok += 1
            return
        if "slot_save failed" in line:
            self.slot_save_failed += 1
            return
        if "slot_restore failed" in line:
            self.slot_restore_failed += 1
            return
        if "slot_ratio_restored" in line:
            self.slot_ratio_restored += 1
            return
        if "cache_state=restored" in line:
            self.restored_any += 1

    # -- derived views -----------------------------------------------------

    def summary(self) -> dict:
        total_saves = self.slot_save_ok + self.slot_save_failed
        return {
            "window": {
                "name": self.name,
                "start": self.start.isoformat(sep=" "),
                "end": self.end.isoformat(sep=" "),
                "hours": round((self.end - self.start).total_seconds() / 3600.0, 2),
            },
            "lines": self.lines,
            "streams": {
                "total": self.stream_started,
                "local": self.stream_started_local,
                "local_share_pct": (
                    round(100.0 * self.stream_started_local / self.stream_started, 2) if self.stream_started else None
                ),
            },
            "routing": {
                "decisions": self.routing_check,
                "cached_ratio_zero": self.routing_check_ratio_zero,
                "cached_ratio_zero_pct": (
                    round(100.0 * self.routing_check_ratio_zero / self.routing_check, 2) if self.routing_check else None
                ),
                "mean_cached_ratio": (
                    round(self.routing_ratio_sum / self.routing_check, 4) if self.routing_check else None
                ),
                "by_cache_state": dict(sorted(self.routing_state.items())),
            },
            "local_skip": {
                "total": self.skip_local + self.economic_bypass,
                "routing_skip_local": self.skip_local,
                "economic_bypass": self.economic_bypass,
                "large_context_bypass": self.skip_reasons.get("large_context_bypass", 0),
                "reasons": dict(self.skip_reasons.most_common()),
            },
            "local_prefill_ms": {
                "all": _percentiles(self.dispatches),
                "by_cache_state": {
                    state: _percentiles(values) for state, values in sorted(self.dispatches_by_state.items())
                },
                "by_context_bucket": self.bucket_pairs(DEFAULT_BUCKETS),
            },
            "slots": {
                "save_ok": self.slot_save_ok,
                "save_failed": self.slot_save_failed,
                "save_failure_pct": (round(100.0 * self.slot_save_failed / total_saves, 2) if total_saves else None),
                "restore_failed": self.slot_restore_failed,
                "ratio_restored_events": self.slot_ratio_restored,
                "cache_state_restored_lines": self.restored_any,
            },
        }

    def bucket_pairs(self, bucket_edges: tuple[int, ...]) -> dict:
        """Pair local dispatches with the preceding routing estimate per session.

        Returns ``{"edges": [...], "by_bucket": {bucket: {state: percentiles}}}``
        so warm and cold dispatches of a comparable prompt size can be compared
        (AC3's like-for-like rule). A dispatch is paired with the most recent
        ``routing_check`` for the same session within one hour — the decision
        that authorised the dispatch.
        """
        edges = sorted(bucket_edges)

        def bucket_of(estimate: int) -> str:
            if estimate <= edges[0]:
                return f"<= {edges[0]:,}"
            if estimate <= edges[1]:
                return f"{edges[0]:,}-{edges[1]:,}"
            return f"> {edges[1]:,}"

        result: dict[str, dict[str, list[float]]] = {}
        for ts, session, seconds, state in self.dispatch_records:
            if not session:
                continue
            estimates = self.routing_estimates.get(session) or []
            candidates = [(t, est) for t, est in estimates if t <= ts <= t + dt.timedelta(hours=1)]
            if not candidates:
                continue
            _, estimate = max(candidates, key=lambda pair: pair[0])
            result.setdefault(bucket_of(estimate), {}).setdefault(state, []).append(seconds)
        return {
            "edges": edges,
            "by_bucket": {
                bucket: {state: _percentiles(values) for state, values in sorted(states.items())}
                for bucket, states in sorted(result.items())
            },
        }


def collect(
    log_dir: Path,
    windows: list[tuple[str, dt.datetime, dt.datetime]],
    file_glob: str = "proxy.log*",
) -> list[WindowMetrics]:
    """Scan every matching log file once and return per-window metrics."""
    metrics = [WindowMetrics(name, start, end) for name, start, end in windows]
    for path in sorted(log_dir.glob(file_glob)):
        if not path.is_file():
            continue
        for line in read_log_lines(path):
            match = RE_TS.match(line)
            if not match:
                continue
            ts = parse_ts(match.group(1))
            if ts is None:
                continue
            for item in metrics:
                if item.start <= ts <= item.end:
                    # Windows may overlap (full span and last 24 h); each one is an
                    # independent measurement, so a line counts in every window
                    # that contains it. Do NOT break on the first match.
                    item.add_line(line, ts)
    return metrics


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(value: float | None, unit: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:g}{unit}"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}%"


def render_markdown(results: list[dict]) -> str:
    """Render a side-by-side comparison table for the given window summaries."""
    headers = [item["window"]["name"] for item in results]
    rows = [
        ("Window", [f"{i['window']['start']} → {i['window']['end']}" for i in results]),
        ("Hours", [_fmt(i["window"]["hours"]) for i in results]),
        (
            "Routing decisions (`routing_check`)",
            [_fmt(i["routing"]["decisions"]) for i in results],
        ),
        (
            "`cached_ratio=0.00` decisions",
            [f"{i['routing']['cached_ratio_zero']} ({_pct(i['routing']['cached_ratio_zero_pct'])})" for i in results],
        ),
        ("Mean `cached_ratio`", [_fmt(i["routing"]["mean_cached_ratio"]) for i in results]),
        (
            "`large_context_bypass`",
            [_fmt(i["local_skip"]["large_context_bypass"]) for i in results],
        ),
        ("Local skips (total)", [_fmt(i["local_skip"]["total"]) for i in results]),
        (
            "Local serve share",
            [
                f"{i['streams']['local']} / {i['streams']['total']} ({_pct(i['streams']['local_share_pct'])})"
                for i in results
            ],
        ),
    ]
    prefill = [i["local_prefill_ms"]["all"] for i in results]
    for key, label in (
        ("count", "Local prefill samples"),
        ("p10", "Local prefill p10 (s)"),
        ("p50", "Local prefill p50 (s)"),
        ("p90", "Local prefill p90 (s)"),
        ("max", "Local prefill max (s)"),
    ):
        rows.append((label, [_fmt(p[key]) if p else "—" for p in prefill]))
    for state in ("restored", "warm", "cold"):
        vals = [i["local_prefill_ms"]["by_cache_state"].get(state) for i in results]
        rows.append(
            (
                f"  `cache_state={state}` dispatches",
                [f"n={v['count']}, p50={v['p50']:g}s, p90={v['p90']:g}s" if v else "—" for v in vals],
            )
        )
    rows.extend(
        [
            (
                "`slot_ratio_restored` seeds",
                [_fmt(i["slots"]["ratio_restored_events"]) for i in results],
            ),
            (
                "`cache_state=restored` lines",
                [_fmt(i["slots"]["cache_state_restored_lines"]) for i in results],
            ),
            (
                "`slot_save` ok / failed",
                [
                    f"{i['slots']['save_ok']} / {i['slots']['save_failed']} ({_pct(i['slots']['save_failure_pct'])})"
                    for i in results
                ],
            ),
            (
                "`slot_restore failed`",
                [_fmt(i["slots"]["restore_failed"]) for i in results],
            ),
        ]
    )
    width = max(len(label) for label, _ in rows) + 2
    widths = [max(len(h), *(len(row[i]) for _, row in rows)) for i, h in enumerate(headers)]
    out = [
        "| " + "Metric".ljust(width) + " | " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |",
        "|" + "-" * (width + 2) + "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    for label, row in rows:
        out.append("| " + label.ljust(width) + " | " + " | ".join(cell.ljust(w) for cell, w in zip(row, widths)) + " |")
    buckets = [i["local_prefill_ms"]["by_context_bucket"]["by_bucket"] for i in results]
    if any(buckets):
        bucket_names: list[str] = []
        for bucket in buckets:
            for name in bucket:
                if name not in bucket_names:
                    bucket_names.append(name)
        out.append("")
        out.append("Local prefill by context bucket (routing estimate; states side by side):")
        head = ("Bucket", "State", *headers)
        rows2 = []
        for name in bucket_names:
            for state in ("restored", "warm", "cold"):
                cells = []
                for bucket in buckets:
                    stats = bucket.get(name, {}).get(state)
                    cells.append(f"n={stats['count']}, p50={stats['p50']:g}s, p90={stats['p90']:g}s" if stats else "—")
                rows2.append((name, state, *cells))
        w0 = max(len(r[0]) for r in rows2) + 2
        w1 = max(len(r[1]) for r in rows2) + 2
        out.append(
            "| "
            + head[0].ljust(w0)
            + " | "
            + head[1].ljust(w1)
            + " | "
            + " | ".join(h.ljust(w) for h, w in zip(headers, widths))
            + " |"
        )
        out.append("|" + "-" * (w0 + 2) + "|" + "-" * (w1 + 2) + "|" + "|".join("-" * (w + 2) for w in widths) + "|")
        for name, state, *cells in rows2:
            out.append(
                "| "
                + name.ljust(w0)
                + " | "
                + state.ljust(w1)
                + " | "
                + " | ".join(c.ljust(w) for c, w in zip(cells, widths))
                + " |"
            )
    return "\n".join(out)


def sidecar_inventory(slot_root: Path | None) -> dict:
    """Count ``*.ratio.json`` sidecars under *slot_root* (recursive)."""
    if slot_root is None or not slot_root.is_dir():
        return {
            "root": str(slot_root) if slot_root else None,
            "count": 0,
            "newest": None,
            "oldest": None,
            "by_model": {},
        }
    files = sorted(slot_root.rglob("*.bin.ratio.json"))
    models: Counter = Counter()
    stamps: list[float] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        models[str(payload.get("model") or "unknown")] += 1
        try:
            stamps.append(path.stat().st_mtime)
        except OSError:
            pass

    def iso(ts: float | None) -> str | None:
        return None if ts is None else dt.datetime.fromtimestamp(ts).isoformat(sep=" ", timespec="seconds")

    return {
        "root": str(slot_root),
        "count": len(files),
        "newest": iso(max(stamps)) if stamps else None,
        "oldest": iso(min(stamps)) if stamps else None,
        "by_model": dict(models.most_common()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_window(spec: str) -> tuple[str, dt.datetime, dt.datetime]:
    """Parse ``name=START,END`` where the timestamps are ISO-ish."""
    if "=" not in spec or "," not in spec:
        raise argparse.ArgumentTypeError(
            "window must be name=START,END (e.g. baseline=2026-10-19T07:23:12,2026-10-20T07:23:12)"
        )
    name, rest = spec.split("=", 1)
    start_text, end_text = rest.split(",", 1)
    start = _parse_iso(start_text)
    end = _parse_iso(end_text)
    if start is None or end is None:
        raise argparse.ArgumentTypeError(f"unparseable window timestamps: {spec!r}")
    if end <= start:
        raise argparse.ArgumentTypeError(f"window end must be after start: {spec!r}")
    return name.strip() or "window", start, end


def _parse_iso(text: str) -> dt.datetime | None:
    text = text.strip().replace("T", " ").replace("Z", "")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def default_windows(
    baseline_start: str,
    baseline_end: str,
    deploy_start: str,
    report_end: dt.datetime,
) -> list[tuple[str, dt.datetime, dt.datetime]]:
    """Return the three windows this measurement compares."""
    b_start = _parse_iso(baseline_start)
    b_end = _parse_iso(baseline_end)
    d_start = _parse_iso(deploy_start)
    if b_start is None or b_end is None or d_start is None:
        raise SystemExit("error: invalid default window timestamps")
    if report_end <= d_start:
        raise SystemExit("error: the report time is before the deploy start — no post-deploy window exists")
    windows = [
        ("baseline (intake)", b_start, b_end),
        ("post-deploy (full)", d_start, report_end),
    ]
    last24 = report_end - dt.timedelta(hours=24)
    if last24 > d_start:
        windows.append(("post-deploy (last 24h)", last24, report_end))
    return windows


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    p.add_argument("--glob", default="proxy.log*", help="log filename glob inside --log-dir (default: proxy.log*)")
    p.add_argument(
        "--window",
        action="append",
        type=_parse_window,
        default=None,
        metavar="NAME=START,END",
        help="explicit window (repeatable); overrides the defaults",
    )
    p.add_argument("--baseline-start", default=DEFAULT_BASELINE_START)
    p.add_argument("--baseline-end", default=DEFAULT_BASELINE_END)
    p.add_argument(
        "--deploy-start", default=DEFAULT_DEPLOY_START, help="post-deploy window start (default: first sidecar write)"
    )
    p.add_argument("--report-end", default=None, help="post-deploy window end (default: now)")
    p.add_argument("--slot-cache-dir", type=Path, default=None, help="slot snapshot root for the sidecar inventory")
    p.add_argument("--end", default=None, help="shorthand for a single window --start/--end")
    p.add_argument("--start", default=None, help="single-window start; pair with --end")
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.add_argument("--quiet", action="store_true", help="suppress the markdown table")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if not args.log_dir.is_dir():
        print(f"error: log directory not found: {args.log_dir}", file=sys.stderr)
        return 1

    if args.start or args.end:
        if not (args.start and args.end):
            print("error: --start and --end must be given together", file=sys.stderr)
            return 2
        start, end = _parse_iso(args.start), _parse_iso(args.end)
        if start is None or end is None or end <= start:
            print("error: invalid --start/--end window", file=sys.stderr)
            return 2
        windows = [("window", start, end)]
    elif args.window:
        windows = list(args.window)
    else:
        report_end = _parse_iso(args.report_end) if args.report_end else dt.datetime.now()
        assert report_end is not None
        windows = default_windows(args.baseline_start, args.baseline_end, args.deploy_start, report_end)

    collected = collect(args.log_dir, windows, args.glob)
    results = [item.summary() for item in collected]
    inventory = sidecar_inventory(args.slot_cache_dir)

    if args.json:
        payload = {
            "meta": {
                "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "log_dir": str(args.log_dir),
                "glob": args.glob,
                "slot_cache_dir": str(args.slot_cache_dir) if args.slot_cache_dir else None,
            },
            "windows": results,
            "sidecars": inventory,
            "restored_engagement": {
                "ratio_restored_events": sum(item["slots"]["ratio_restored_events"] for item in results),
                "cache_state_restored_lines": sum(item["slots"]["cache_state_restored_lines"] for item in results),
                "sidecars_on_disk": inventory["count"],
                "engaged": any(
                    item["slots"]["ratio_restored_events"] or item["slots"]["cache_state_restored_lines"]
                    for item in results
                ),
            },
        }
        print(json.dumps(payload, indent=2))
        return 0

    if not args.quiet:
        print(render_markdown(results))
        print()
        if inventory["count"]:
            print(
                f"Sidecars on disk: {inventory['count']} "
                f"(oldest {inventory['oldest']}, newest {inventory['newest']}) "
                f"models={inventory['by_model']}"
            )
        else:
            print("Sidecars on disk: 0 (writer never produced one for this slot root)")
        restored = sum(item["slots"]["ratio_restored_events"] for item in results)
        if restored == 0:
            print(
                "Restored-prefix warmness did NOT engage: 0 `slot_ratio_restored` "
                "events and 0 `cache_state=restored` lines in these windows — "
                "prefill/latency movement cannot be attributed to the feature."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
