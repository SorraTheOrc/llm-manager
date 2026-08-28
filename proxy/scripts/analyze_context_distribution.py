#!/usr/bin/env python3
"""Session estimated-context distribution analysis for proactive compaction.

F1 deliverable (LP-0MTC87GBV0031F4B, parent LP-0MTAQNAQT002L746): derive the
distribution of session estimated-context sizes from the llama-proxy logs for
the 2026-08-24..26 window and count how many sessions breach the per-mode
per-slot cap, so proactive compaction can be sized before any behavior change.

Why the raw logs and not the report CSVs?

The daily proxy-usage reports (``~/proxy-usage-reports/2026-08-2X/``) carry
session metadata (bucket, fallback reason, request counts) but their context
columns (``start_context_size`` / ``avg_context_size`` / ``max_context_size``)
are empty: the ``Stream finished: tokens=p/c/t`` payload the reporting
pipeline keys on is not emitted in this deployment. The authoritative
per-session context signal is the proxy's own routing-time estimate in the
``routing_check`` log line — ``estimated_tokens`` (accumulated session
history + new prompt, computed with the native tokenizer). Every local
routing decision logs one. ``context_pressure`` warnings (same log) flag the
sessions that cross the 0.80 warn ratio, and ``routing_skip_local`` lines
count the ``context_too_large`` / ``large_context_bypass`` hard-cap skips.

Mode classification

The ``warm_threshold`` value in each ``routing_check`` line is the effective
per-slot clamp at routing time (LP-0MSAZXXDY005AWA1):
    fast  3 slots x 262144 -> min(100000, 262144//3 - 4096) = 83285
    cheap 2 slots x 262144 -> min(100000, 262144//2 - 4096) = 100000
So ``warm_threshold == 83285`` -> fast, ``== 100000`` -> cheap. Sessions are
bucketed by the majority warm_threshold of their samples.

Breach caps (from the parent AC1 / F1 AC2):
    fast  83285  (per-slot cap, 87.4K-slot / 3 - 4096 headroom)
    cheap 61440  (static clamp)

CLI::

    python3 proxy/scripts/analyze_context_distribution.py \\
        --log-dir /var/log/llama-proxy \\
        --days 2026-08-24 2026-08-25 2026-08-26 \\
        --output-dir proxy/docs/context-compaction-eval

Writes ``distribution.json`` + ``distribution.md``; ``--json`` prints the
machine-readable report instead of the markdown summary.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# Effective per-slot clamp values that identify the operating mode at
# routing time (LP-0MSAZXXDY005AWA1; see module docstring).
FAST_WARM_THRESHOLD = 83285
CHEAP_WARM_THRESHOLD = 100000

# Breach caps per mode (parent AC1 / F1 AC2).
FAST_CAP = 83285
CHEAP_CAP = 61440

TS_FMT = "%Y-%m-%d %H:%M:%S"

ROUTING_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?routing_check .*?"
    r"estimated_tokens=(\d+) .*?warm_threshold=(\d+) .*?session=([A-Za-z0-9_.-]+)"
)
# The proxy logs ``session=unknown`` for routing checks it cannot attribute to
# a session (per the skill parser's ``session=unknown`` convention). These
# carry no per-session signal, so they are excluded from session aggregation
# (their token volumes are still counted nowhere — they are not session
# context, just unattributed routing noise).
UNATTRIBUTED_SESSION = "unknown"
PRESSURE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?context_pressure "
    r"session=([A-Za-z0-9_.-]+) estimated_tokens=(\d+) per_slot_ctx=(\d+) "
    r"ratio=([\d.]+)"
)
SKIP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?routing_skip_local .*?"
    r"reason=(\w+) .*?session=([A-Za-z0-9_.-]+)"
)


@dataclass
class EstimateSample:
    """One routing-time context estimate for a session."""

    ts: datetime
    session: str
    estimated_tokens: int
    warm_threshold: int

    @property
    def mode(self) -> str:
        """Fast/cheap mode at routing time, from the effective warm clamp."""
        if self.warm_threshold == FAST_WARM_THRESHOLD:
            return "fast"
        if self.warm_threshold == CHEAP_WARM_THRESHOLD:
            return "cheap"
        return "other"


@dataclass
class SessionAggregate:
    """Aggregated routing-time context signal for one session (one day)."""

    session: str
    samples: list[int] = field(default_factory=list)
    modes: Counter = field(default_factory=Counter)

    def add(self, n_tokens: int, mode: str) -> None:
        self.samples.append(n_tokens)
        self.modes[mode] += 1

    @property
    def mode(self) -> str:
        """Dominant mode for the session's routing checks."""
        best = self.modes.most_common(1)
        return best[0][0] if best else "other"

    @property
    def max_tokens(self) -> int | None:
        return max(self.samples) if self.samples else None

    @property
    def avg_tokens(self) -> float | None:
        return statistics.mean(self.samples) if self.samples else None

    @property
    def last_tokens(self) -> int | None:
        return self.samples[-1] if self.samples else None

    @property
    def count(self) -> int:
        return len(self.samples)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Nearest-rank percentile on an already-sorted list.

    Uses ``ceil(pct/100 * N)`` (exclusive upper bound) so the p50 of a
    4-element list is the 2nd element, matching common percentile semantics.
    """
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    rank = max(1, (pct / 100.0) * n)
    idx = min(n - 1, round_up(rank) - 1)
    return sorted_vals[idx]


def round_up(x: float) -> int:
    """Ceiling for positive floats (int(np.ceil) without the dependency)."""
    return int(x) if x == int(x) else int(x) + 1


def distribution_stats(values: list[float]) -> dict:
    """Median/mean/p90/p95/max over a list of context sizes."""
    if not values:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    sv = sorted(values)
    return {
        "count": len(values),
        "median": round(statistics.median(values), 1),
        "mean": round(statistics.mean(values), 1),
        "p90": round(_percentile(sv, 90), 1),
        "p95": round(_percentile(sv, 95), 1),
        "max": round(max(values), 1),
    }


def iter_log_lines(path: Path):
    """Yield text lines from a proxy log (transparent .gz handling)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if len(line) < 24 or not line[:4].isdigit():
                continue
            yield line


def parse_routing_sample(line: str) -> EstimateSample | None:
    """Parse a ``routing_check`` line into an :class:`EstimateSample`."""
    m = ROUTING_RE.match(line)
    if m is None:
        return None
    ts_s, est, warm, session = m.groups()
    return EstimateSample(
        ts=datetime.strptime(ts_s, TS_FMT),
        session=session,
        estimated_tokens=int(est),
        warm_threshold=int(warm),
    )


def parse_pressure(line: str) -> tuple | None:
    """Parse a ``context_pressure`` warning line -> (session, est, per_slot, ratio)."""
    m = PRESSURE_RE.match(line)
    if m is None:
        return None
    _ts, session, est, per_slot, ratio = m.groups()
    return (session, int(est), int(per_slot), float(ratio))


def parse_skip(line: str) -> tuple | None:
    """Parse a ``routing_skip_local`` line -> (window_ts, reason, session)."""
    m = SKIP_RE.match(line)
    if m is None:
        return None
    ts_s, reason, session = m.groups()
    return (datetime.strptime(ts_s, TS_FMT), reason, session)


def discover_log_files(log_dir: Path) -> list[Path]:
    """All plain + rotated proxy log files, sorted by name."""
    if not log_dir.is_dir():
        return []
    return sorted(
        p
        for p in log_dir.iterdir()
        if p.is_file()
        and (p.name == "proxy.log" or p.name.startswith("proxy.log."))
    )


@dataclass
class DayResult:
    """Per-day extraction: session aggregates + warning/skip tallies."""

    day: str
    sessions: dict[str, SessionAggregate] = field(default_factory=dict)
    pressure_count: int = 0
    pressure_sessions: set = field(default_factory=set)
    skip_counts: Counter = field(default_factory=Counter)


def analyze_day(
    log_dir: Path, day: datetime, include_skip: bool = True
) -> DayResult:
    """Parse every proxy log for one calendar day (00:00 -> 24:00)."""
    start = day
    end = day + timedelta(days=1)
    res = DayResult(day=day.strftime("%Y-%m-%d"))
    for path in discover_log_files(log_dir):
        for line in iter_log_lines(path):
            sample = parse_routing_sample(line)
            if sample is not None:
                if (
                    start <= sample.ts < end
                    and sample.session != UNATTRIBUTED_SESSION
                ):
                    agg = res.sessions.setdefault(
                        sample.session, SessionAggregate(sample.session)
                    )
                    agg.add(sample.estimated_tokens, sample.mode)
                continue
            if "context_pressure" in line:
                p = parse_pressure(line)
                if p is None:
                    continue
                ts_s = line[:19]
                ts = datetime.strptime(ts_s, TS_FMT)
                if start <= ts < end:
                    res.pressure_count += 1
                    res.pressure_sessions.add(p[0])
                continue
            if include_skip and "routing_skip_local" in line:
                s = parse_skip(line)
                if s is None:
                    continue
                ts, reason, _session = s
                if start <= ts < end:
                    res.skip_counts[reason] += 1
    return res


def summarize_modes(
    day: DayResult,
) -> dict[str, dict]:
    """Per-mode per-session-max and per-request distribution stats."""
    by_mode: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"session_max": [], "sample": []}
    )
    for agg in day.sessions.values():
        mode = agg.mode
        if agg.max_tokens is not None:
            by_mode[mode]["session_max"].append(float(agg.max_tokens))
        by_mode[mode]["sample"].extend(float(t) for t in agg.samples)
    out: dict[str, dict] = {}
    for mode, buckets in sorted(by_mode.items()):
        out[mode] = {
            "sessions": distribution_stats(buckets["session_max"]),
            "requests": distribution_stats(buckets["sample"]),
        }
    return out


def breach_summary(
    day: DayResult, caps: dict[str, int]
) -> dict[str, dict]:
    """Per-mode session breach counts at the configured caps."""
    per_mode: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "breach": 0}
    )
    for agg in day.sessions.values():
        mode = agg.mode if agg.mode in caps else "other"
        per_mode[mode]["total"] += 1
        if agg.max_tokens is not None and agg.max_tokens >= caps.get(mode, 0):
            per_mode[mode]["breach"] += 1
    out: dict[str, dict] = {}
    for mode, counts in sorted(per_mode.items()):
        total = counts["total"]
        breach = counts["breach"]
        out[mode] = {
            "sessions": total,
            "breach": breach,
            "breach_pct": round(breach / total * 100, 1) if total else 0.0,
            "cap": caps.get(mode),
        }
    return out


def build_report(
    log_dir: Path, days: list[datetime], caps: dict[str, int]
) -> dict:
    """Full machine-readable report across all requested days."""
    results = [analyze_day(log_dir, d) for d in days]
    return {
        "window": f"{days[0]:%Y-%m-%d}..{days[-1]:%Y-%m-%d}",
        "caps": dict(caps),
        "days": {
            r.day: {
                "sessions": len(r.sessions),
                "distributions": summarize_modes(r),
                "breaches": breach_summary(r, caps),
                "context_pressure_warnings": r.pressure_count,
                "context_pressure_sessions": len(r.pressure_sessions),
                "routing_skips": dict(r.skip_counts),
            }
            for r in results
        },
    }


def render_markdown(report: dict) -> str:
    """Render the report as the evidence Markdown table set."""
    lines: list[str] = []
    lines.append("# Session estimated-context distribution (2026-08-24..26)")
    lines.append("")
    lines.append(
        "Derived from `routing_check` log lines (`estimated_tokens`, the proxy's "
        "routing-time session-context estimate). Mode per the effective per-slot "
        "warm clamp (fast 83285 / cheap 100000). Breach caps: fast 83285, cheap 61440."
    )
    lines.append("")
    # --- Trend summary (AC3) ---
    days = list(report["days"].items())
    lines.append("## Trend across days")
    lines.append("")
    lines.append("| Day | Sessions | Pressure warnings | Breach fast | Breach cheap |")
    lines.append("|---|---|---|---|---|")
    for day, d in days:
        bf = d["breaches"].get("fast", {})
        bc = d["breaches"].get("cheap", {})
        lines.append(
            f"| {day} | {d['sessions']} | {d['context_pressure_warnings']} "
            f"| {bf.get('breach', '-')}/{bf.get('sessions', '-')} "
            f"| {bc.get('breach', '-')}/{bc.get('sessions', '-')} |"
        )
    lines.append("")
    for day, d in days:
        lines.append(f"## {day}")
        lines.append("")
        lines.append(f"- Sessions with routing checks: **{d['sessions']}**")
        lines.append(
            f"- `context_pressure` warnings: **{d['context_pressure_warnings']}** "
            f"({d['context_pressure_sessions']} sessions)"
        )
        skips = d["routing_skips"]
        if skips:
            skip_str = ", ".join(f"{k}={v}" for k, v in sorted(skips.items()))
        else:
            skip_str = "none"
        lines.append(f"- routing_skip_local: {skip_str}")
        lines.append("")
        lines.append("### Distribution (per-session max estimated context)")
        lines.append("")
        lines.append("| Mode | Sessions | Median | Mean | p90 | p95 | Max |")
        lines.append("|---|---|---|---|---|---|---|")
        for mode, stats in d["distributions"].items():
            s = stats["sessions"]
            lines.append(
                f"| {mode} | {s['count']} | {s['median']:.0f} | {s['mean']:.0f} "
                f"| {s['p90']:.0f} | {s['p95']:.0f} | {s['max']:.0f} |"
            )
        lines.append("")
        lines.append("### Breach counts vs per-mode cap")
        lines.append("")
        lines.append("| Mode | Sessions | Breach (>= cap) | % | Cap |")
        lines.append("|---|---|---|---|---|")
        for mode, b in d["breaches"].items():
            lines.append(
                f"| {mode} | {b['sessions']} | {b['breach']} | "
                f"{b['breach_pct']}% | {b['cap']} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_artifacts(report: dict, output_dir: Path) -> tuple[Path, Path]:
    """Write distribution.json + distribution.md into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "distribution.json"
    md_path = output_dir / "distribution.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(report) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze_context_distribution.py",
        description=(
            "Session estimated-context distribution + breach counts per mode "
            "from llama-proxy logs (evaluation only; no behavior change)."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default="/var/log/llama-proxy",
        help="dir containing proxy.log* (default: /var/log/llama-proxy)",
    )
    parser.add_argument(
        "--days",
        nargs="+",
        default=["2026-08-24", "2026-08-25", "2026-08-26"],
        help="calendar days to analyze, YYYY-MM-DD (default: 2026-08-24..26)",
    )
    parser.add_argument(
        "--output-dir",
        default="proxy/docs/context-compaction-eval",
        help="output dir for distribution.json / distribution.md",
    )
    parser.add_argument(
        "--fast-cap", type=int, default=FAST_CAP, help="fast breach cap (default 83285)"
    )
    parser.add_argument(
        "--cheap-cap", type=int, default=CHEAP_CAP, help="cheap breach cap (default 61440)"
    )
    parser.add_argument(
        "--json", action="store_true", help="print JSON report instead of markdown"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"error: log dir not found: {log_dir}", file=sys.stderr)
        return 2
    try:
        days = [datetime.strptime(d, "%Y-%m-%d") for d in args.days]
    except ValueError:
        print("error: --days must be YYYY-MM-DD", file=sys.stderr)
        return 2
    caps = {"fast": args.fast_cap, "cheap": args.cheap_cap}
    report = build_report(log_dir, days, caps)
    json_path, md_path = write_artifacts(report, Path(args.output_dir))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
        print(f"artifacts: {json_path} {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())