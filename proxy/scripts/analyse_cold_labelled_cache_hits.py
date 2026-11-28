#!/usr/bin/env python3
"""Confirm that sub-second ``cold`` dispatches were prompt-cache hits.

LP-0MWRQVLFY003BK5L (evidence half of LP-0MWRQ6ZID000HZAZ).

Why
----

The proxy prices a local dispatch's cache warmth from a *per-(model, session)*
table (``provider._get_cached_ratio``). When that table has no entry — a fresh
session, or one whose entry was invalidated/evicted — the dispatch is logged
``cache_state=cold cached_ratio=0.000``. llama-server's prompt cache is
**global and cross-session**: an identical (or shared-prefix) prompt is restored
from ``server_prompt_cache`` regardless of which session sent it. The proxy
therefore cannot see a large class of genuine prompt-cache hits.

On the 2026-11-21 window the symptom was 572 dispatches with
``estimated_tokens > 3000`` whose time-to-first-byte was under 0.8 s while
logged ``cached_ratio=0.0`` — an implied prefill rate orders of magnitude above
the engine's measured throughput. This script proves what actually happened from
llama-server's own signals.

Two independent signals
-----------------------

1. **Proxy-side measured reuse** (``prefill_progress_measured`` lines). When
   ``local_measured_cache_abort_enabled`` is on, the proxy injects
   ``return_progress`` and llama-server reports
   ``cache = n_prompt_tokens_cache`` against ``total``. The measurement is
   emitted immediately *before* the dispatch line for the same session, so it is
   joined here FIFO per session (each measurement is consumed once). This is the
   server's reuse number, not the routing prediction.
2. **llama-server slot log** (optional ``--llama-log``). Every task carries
   ``new prompt ... task.n_tokens = N`` and a following
   ``n_tokens = X, memory_seq_rm [X, end)`` line: X is the retained (reused)
   prefix and ``N - X`` the tokens actually prefilled. The ``prompt eval time``
   lines give the aggregate tokens prefilled and the engine's bulk throughput.

Usage
-----

    # Human summary over the default 2026-11-21 window
    ./proxy/scripts/analyse_cold_labelled_cache_hits.py --date 2026-11-21

    # Machine readable, with the independent llama-server cross-check
    ./proxy/scripts/analyse_cold_labelled_cache_hits.py --date 2026-11-21 \
        --llama-log /var/log/llama-proxy/llama-server.log-2026-11-22 --json

    # Any other window (proxy log lines carry the authoritative timestamp)
    ./proxy/scripts/analyse_cold_labelled_cache_hits.py \
        --start '2026-11-21 00:00' --end '2026-11-22 00:00'

Exit codes: 0 success, 1 no proxy logs / unexpected error.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

#: Default llama-proxy log directory (proxy.log* and llama-server.log* live
#: side by side).
DEFAULT_LOG_DIR = Path("/var/log/llama-proxy")

#: A dispatch is "sub-second cold-labelled" when its time-to-first-byte is
#: below this bound and its routing ratio was below the warm threshold.
DEFAULT_MAX_FIRST_BYTE_MS = 800.0
#: ...and it carried at least this many estimated prompt tokens (the small ones
#: are simply fast and prove nothing).
DEFAULT_MIN_ESTIMATED_TOKENS = 3000
#: A measurement is paired to a dispatch only when it precedes it by at most
#: this many seconds (the pair is emitted back-to-back).
DEFAULT_JOIN_WINDOW_S = 10.0

_DISPATCH_RE = re.compile(
    r"^(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) .*?"
    r"dispatch_first_byte_ms=(?P<fb>[0-9.]+) "
    r"dispatch_to_first_byte_ms=(?P<tfb>[0-9.]+) "
    r"cache_state=(?P<state>\w+) "
    r"cached_ratio=(?P<ratio>[0-9.]+) "
    r"queue_wait_ms=(?P<qw>[0-9.]+) "
    r"session=(?P<session>\S+) "
    r"model=(?P<model>\S+) "
    r"priority=(?P<prio>\S+) "
    r"estimated_tokens=(?P<est>\d+) "
    r"turn=(?P<turn>\w+)"
)

_MEASURED_RE = re.compile(
    r"^(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) .*?prefill_progress_measured "
    r"session=(?P<session>\S+) "
    r"model=(?P<model>\S+) "
    r"total=(?P<total>\d+) "
    r"cache=(?P<cache>\d+) "
    r"processed=(?P<processed>\d+) "
    r"time_ms=(?P<time>[0-9.]+) "
    r"measured_cached_ratio=(?P<mcr>[0-9.]+) "
    r"measured_new_tokens=(?P<mnt>\d+) "
    r"priced_new_tokens=(?P<pnt>\d+)"
)

_LLAMA_NEW_RE = re.compile(
    r"task (?P<task>\d+) \| new prompt, n_ctx_slot = \d+, n_keep = \d+, "
    r"task\.n_tokens = (?P<tokens>\d+)"
)
# ``n_tokens = 9878, memory_seq_rm [9878, end)`` — the first field is the
# retained (reused) prefix; the brackets repeat it.
_LLAMA_RM_RE = re.compile(
    r"task (?P<task>\d+) \| n_tokens = (?P<n>\d+), memory_seq_rm \[(?P<a>\d+), end\)"
)
_LLAMA_EVAL_RE = re.compile(
    r"^\[(?P<pid>\d+)\] prompt eval time =\s+(?P<ms>[0-9.]+) ms /\s+"
    r"(?P<tokens>\d+) tokens"
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_timestamp(raw: str) -> datetime:
    """Parse a proxy-log leading timestamp (``%Y-%m-%d %H:%M:%S,%f``)."""
    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S,%f")


@dataclass(frozen=True)
class Dispatch:
    """One ``dispatch_first_byte_ms`` log line."""

    ts: datetime
    first_byte_ms: float
    to_first_byte_ms: float
    cache_state: str
    cached_ratio: float
    session: str
    model: str
    priority: str
    estimated_tokens: int
    turn: str

    @property
    def cold_labelled(self) -> bool:
        """True when routing priced no cache (the label this item audits)."""
        return self.cache_state == "cold" or self.cached_ratio == 0.0


@dataclass(frozen=True)
class Measurement:
    """One ``prefill_progress_measured`` line (llama-server reuse)."""

    ts: datetime
    session: str
    total: int
    cache: int
    measured_cached_ratio: float
    measured_new_tokens: int
    priced_new_tokens: int


def parse_dispatch(line: str) -> Dispatch | None:
    """Return the :class:`Dispatch` a log line describes, or ``None``."""
    match = _DISPATCH_RE.match(line)
    if not match:
        return None
    return Dispatch(
        ts=parse_timestamp(match["ts"]),
        first_byte_ms=float(match["fb"]),
        to_first_byte_ms=float(match["tfb"]),
        cache_state=match["state"],
        cached_ratio=float(match["ratio"]),
        session=match["session"],
        model=match["model"],
        priority=match["prio"],
        estimated_tokens=int(match["est"]),
        turn=match["turn"],
    )


def parse_measurement(line: str) -> Measurement | None:
    """Return the :class:`Measurement` a log line describes, or ``None``."""
    match = _MEASURED_RE.match(line)
    if not match:
        return None
    return Measurement(
        ts=parse_timestamp(match["ts"]),
        session=match["session"],
        total=int(match["total"]),
        cache=int(match["cache"]),
        measured_cached_ratio=float(match["mcr"]),
        measured_new_tokens=int(match["mnt"]),
        priced_new_tokens=int(match["pnt"]),
    )


def read_log_lines(paths, failures: list[str] | None = None) -> list[str]:
    """Read every line of *paths*, transparently decompressing ``.gz``.

    Unreadable files are skipped and named in *failures* so a silently short
    window is reported instead of publishing a count that lost a rotation file.
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
    return lines


def discover_proxy_logs(log_dir: Path) -> list[Path]:
    """Every proxy log file in *log_dir*, both rotation schemes, sorted."""
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return []
    return sorted(
        path
        for path in log_dir.iterdir()
        if path.is_file()
        and (
            path.name == "proxy.log"
            or path.name.startswith("proxy.log.")
            or path.name.startswith("proxy.log-")
        )
    )


# ---------------------------------------------------------------------------
# Joining measurements to dispatches
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedDispatch:
    """A dispatch with the llama-server measurement that preceded it."""

    dispatch: Dispatch
    measurement: Measurement | None
    gap_seconds: float | None


def join_measurements(
    events,
    join_window_s: float = DEFAULT_JOIN_WINDOW_S,
) -> list[PairedDispatch]:
    """Pair each dispatch with the measurement emitted just before it.

    *events* is an iterable of :class:`Dispatch` / :class:`Measurement` in log
    order (it may span several files, already sorted by name). A measurement is
    kept per session and **consumed** by the first dispatch that follows it
    within *join_window_s*; a stale measurement therefore cannot be re-counted
    or attached to a later dispatch. Dispatches with no fresh measurement keep
    ``measurement=None``.
    """
    last: dict[str, Measurement] = {}
    paired: list[PairedDispatch] = []
    for event in events:
        if isinstance(event, Measurement):
            last[event.session] = event
            continue
        assert isinstance(event, Dispatch)
        previous = last.pop(event.session, None)
        if previous is None:
            paired.append(PairedDispatch(event, None, None))
            continue
        gap = (event.ts - previous.ts).total_seconds()
        if 0.0 <= gap <= join_window_s:
            paired.append(PairedDispatch(event, previous, gap))
        else:
            paired.append(PairedDispatch(event, None, gap))
    return paired


# ---------------------------------------------------------------------------
# llama-server slot-log cross-check
# ---------------------------------------------------------------------------


def parse_prompt_eval(line: str) -> dict | None:
    """Parse a llama-server ``prompt eval time`` line, or return ``None``."""
    match = _LLAMA_EVAL_RE.search(line)
    if not match:
        return None
    tokens = int(match["tokens"])
    ms = float(match["ms"])
    return {
        "tokens": tokens,
        "ms": ms,
        "tok_s": round(tokens / (ms / 1000.0), 1) if ms else 0.0,
        "line": line.strip(),
    }


def parse_llama_reuse(lines) -> dict:
    """Derive per-task prompt reuse from a llama-server slot log.

    Each task's ``new prompt ... task.n_tokens = N`` gives the prompt size; the
    following ``n_tokens = X, memory_seq_rm [X, end)`` line gives the retained
    prefix X (reused) and ``N - X`` is what was actually prefilled. The
    ``prompt eval time`` lines give the aggregate tokens prefilled and the bulk
    throughput, independent of the proxy entirely.
    """
    totals: dict[str, int] = {}
    reuse: dict[str, int] = {}
    eval_tokens = 0
    eval_ms = 0.0
    eval_lines = 0
    for line in lines:
        match = _LLAMA_NEW_RE.search(line)
        if match:
            totals[match["task"]] = int(match["tokens"])
            continue
        match = _LLAMA_RM_RE.search(line)
        if match:
            reuse[match["task"]] = int(match["a"])
            continue
        evaluated = parse_prompt_eval(line)
        if evaluated is not None:
            eval_lines += 1
            eval_tokens += evaluated["tokens"]
            eval_ms += evaluated["ms"]

    matched = [task for task in totals if task in reuse]
    prompt_tokens = sum(totals[task] for task in matched)
    reused_tokens = sum(reuse[task] for task in matched)
    result: dict = {
        "tasks": len(totals),
        "tasks_with_reuse_line": len(matched),
        "prompt_tokens": prompt_tokens,
        "reused_tokens": reused_tokens,
        "reuse_pct": round(100.0 * reused_tokens / prompt_tokens, 2) if prompt_tokens else 0.0,
        "prompt_eval_lines": eval_lines,
        "prompt_eval_tokens": eval_tokens,
        "prompt_eval_aggregate_tok_s": round(eval_tokens / (eval_ms / 1000.0), 1) if eval_ms else 0.0,
    }
    if matched:
        sizes = Counter(totals[task] for task in matched)
        dominant_size, dominant_count = sizes.most_common(1)[0]
        dominant = [task for task in matched if totals[task] == dominant_size]
        dominant_reused = sum(reuse[task] for task in dominant)
        dominant_prompt = sum(totals[task] for task in dominant)
        result.update(
            {
                "dominant_prompt_tokens": dominant_size,
                "dominant_prompt_count": dominant_count,
                "dominant_reuse_pct": round(
                    100.0 * dominant_reused / dominant_prompt, 2
                ) if dominant_prompt else 0.0,
            }
        )
    return result


# ---------------------------------------------------------------------------
# Statistics + summary
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return ordered[idx]


def _stats(values: list[float], decimals: int = 2) -> dict:
    return {
        "samples": len(values),
        "median": round(_percentile(values, 50), decimals) if values else 0.0,
        "p10": round(_percentile(values, 10), decimals) if values else 0.0,
        "p90": round(_percentile(values, 90), decimals) if values else 0.0,
        "min": round(min(values), decimals) if values else 0.0,
        "max": round(max(values), decimals) if values else 0.0,
    }


def summarise(
    paired: list[PairedDispatch],
    max_first_byte_ms: float = DEFAULT_MAX_FIRST_BYTE_MS,
    min_estimated_tokens: int = DEFAULT_MIN_ESTIMATED_TOKENS,
    engine_tok_s: float | None = None,
) -> dict:
    """Reduce the paired dispatches to the evidence summary.

    The headline is the set of cold-labelled dispatches whose implied prefill
    rate (``estimated_tokens / time-to-first-byte``) is far above anything the
    engine can compute inline. For each of them the joined llama-server
    measurement is reported, so "warm prefix the proxy did not price" is a
    measured fact rather than an inference.
    """
    dispatches = [pair.dispatch for pair in paired]
    cold = [pair for pair in paired if pair.dispatch.cold_labelled]
    suspect = [
        pair
        for pair in cold
        if pair.dispatch.to_first_byte_ms < max_first_byte_ms
        and pair.dispatch.estimated_tokens > min_estimated_tokens
    ]
    measured = [pair for pair in suspect if pair.measurement is not None]

    implied = [
        pair.dispatch.estimated_tokens / (pair.dispatch.to_first_byte_ms / 1000.0)
        for pair in suspect
        if pair.dispatch.to_first_byte_ms > 0
    ]
    mcrs = [pair.measurement.measured_cached_ratio for pair in measured]
    new_tokens = [float(pair.measurement.measured_new_tokens) for pair in measured]
    reused = sum(pair.measurement.cache for pair in measured)
    total = sum(pair.measurement.total for pair in measured)

    summary: dict = {
        "dispatches": len(dispatches),
        "cold_labelled": len(cold),
        "cold_labelled_sub_second": len(suspect),
        "with_measurement": len(measured),
        "measurement_coverage_pct": round(100.0 * len(measured) / len(suspect), 2) if suspect else 0.0,
        "distinct_sessions": len({pair.dispatch.session for pair in suspect}),
        "estimated_tokens_sum": sum(pair.dispatch.estimated_tokens for pair in suspect),
        "implied_prefill_tok_s": _stats(implied),
        "first_byte_ms": _stats([pair.dispatch.to_first_byte_ms for pair in suspect]),
        "measured_cached_ratio": _stats(mcrs, decimals=4),
        "measured_new_tokens": _stats(new_tokens),
        "reused_tokens": reused,
        "measured_total_tokens": total,
        "measured_reuse_pct": round(100.0 * reused / total, 2) if total else 0.0,
        "thresholds": {
            "max_first_byte_ms": max_first_byte_ms,
            "min_estimated_tokens": min_estimated_tokens,
        },
    }
    if engine_tok_s:
        summary["engine_aggregate_tok_s"] = engine_tok_s
        summary["implied_vs_engine_ratio"] = (
            round(summary["implied_prefill_tok_s"]["median"] / engine_tok_s, 1)
            if engine_tok_s
            else 0.0
        )
    if suspect:
        order = sorted(pair.dispatch.ts for pair in suspect)
        summary["window_first"] = order[0].isoformat(sep=" ", timespec="seconds")
        summary["window_last"] = order[-1].isoformat(sep=" ", timespec="seconds")
        dominant = Counter(pair.dispatch.estimated_tokens for pair in suspect).most_common(1)[0]
        summary["dominant_estimated_tokens"] = dominant[0]
        summary["dominant_count"] = dominant[1]
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_day(raw: str) -> tuple[datetime, datetime]:
    day = datetime.strptime(raw, "%Y-%m-%d")
    return day, day.replace(hour=23, minute=59, second=59, microsecond=999999)


def _parse_bound(raw: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"unrecognised timestamp: {raw!r}")


def _collect_events(lines, start: datetime, end: datetime):
    """Yield Dispatch/Measurement objects whose timestamp is inside the window."""
    for line in lines:
        dispatch = parse_dispatch(line)
        if dispatch is not None:
            if start <= dispatch.ts <= end:
                yield dispatch
            continue
        measurement = parse_measurement(line)
        if measurement is not None and start <= measurement.ts <= end:
            yield measurement


def _render(summary: dict, llama: dict | None) -> str:
    lines = [
        "cold-labelled sub-second prompt-cache analysis",
        "=" * 46,
        f"window                        : {summary.get('window_first', 'n/a')} -> "
        f"{summary.get('window_last', 'n/a')}",
        f"dispatches in window          : {summary['dispatches']:,}",
        f"cold-labelled                 : {summary['cold_labelled']:,}",
        f"  + sub-{summary['thresholds']['max_first_byte_ms']:.0f} ms and "
        f"> {summary['thresholds']['min_estimated_tokens']:,} est tokens: "
        f"{summary['cold_labelled_sub_second']:,}",
        f"distinct sessions             : {summary['distinct_sessions']}",
        f"measured server-side reuse    : {summary['with_measurement']:,} "
        f"({summary['measurement_coverage_pct']}% coverage)",
        f"measured cached ratio         : median "
        f"{summary['measured_cached_ratio']['median']} "
        f"(p10 {summary['measured_cached_ratio']['p10']} / "
        f"p90 {summary['measured_cached_ratio']['p90']}, "
        f"min {summary['measured_cached_ratio']['min']})",
        f"measured new tokens/request   : median {summary['measured_new_tokens']['median']} "
        f"(max {summary['measured_new_tokens']['max']:.0f})",
        f"reused / measured total       : {summary['reused_tokens']:,} / "
        f"{summary['measured_total_tokens']:,} ({summary['measured_reuse_pct']}%)",
        f"implied prefill throughput    : median "
        f"{summary['implied_prefill_tok_s']['median']:,.0f} tok/s "
        f"(p10 {summary['implied_prefill_tok_s']['p10']:,.0f} / "
        f"p90 {summary['implied_prefill_tok_s']['p90']:,.0f})",
        f"first byte (time-to-first)    : median {summary['first_byte_ms']['median']:.1f} ms",
    ]
    if "engine_aggregate_tok_s" in summary:
        lines.append(
            f"engine aggregate prefill rate : {summary['engine_aggregate_tok_s']:,.1f} tok/s "
            f"({summary['implied_vs_engine_ratio']:,.0f}x lower than implied)"
        )
    if llama:
        lines += [
            "",
            "llama-server slot-log cross-check",
            "-" * 46,
            f"tasks / with reuse line       : {llama['tasks']:,} / "
            f"{llama['tasks_with_reuse_line']:,}",
            f"reused / prompt tokens        : {llama['reused_tokens']:,} / "
            f"{llama['prompt_tokens']:,} ({llama['reuse_pct']}%)",
            f"prompt eval lines / tokens    : {llama['prompt_eval_lines']:,} / "
            f"{llama['prompt_eval_tokens']:,}",
            f"prompt eval aggregate rate    : {llama['prompt_eval_aggregate_tok_s']:,.1f} tok/s",
        ]
        if "dominant_prompt_tokens" in llama:
            lines.append(
                f"dominant repeated prompt      : {llama['dominant_prompt_tokens']:,} tokens "
                f"x {llama['dominant_prompt_count']:,} "
                f"(reuse {llama['dominant_reuse_pct']}%)"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--date", help="single UTC day, YYYY-MM-DD (ignores --start/--end)")
    parser.add_argument("--start", type=_parse_bound, help="window start (inclusive)")
    parser.add_argument("--end", type=_parse_bound, help="window end (inclusive)")
    parser.add_argument(
        "--llama-log",
        type=Path,
        default=None,
        help="llama-server slot log covering the window, for the independent cross-check",
    )
    parser.add_argument(
        "--max-first-byte-ms", type=float, default=DEFAULT_MAX_FIRST_BYTE_MS
    )
    parser.add_argument(
        "--min-estimated-tokens", type=int, default=DEFAULT_MIN_ESTIMATED_TOKENS
    )
    parser.add_argument(
        "--join-window-s", type=float, default=DEFAULT_JOIN_WINDOW_S,
        help="max seconds between a measurement and the dispatch it precedes",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.date:
        if args.start or args.end:
            parser.error("--date cannot be combined with --start/--end")
        start, end = _parse_day(args.date)
    elif args.start or args.end:
        start = args.start or datetime.min
        end = args.end or datetime.max
    else:
        parser.error("one of --date, --start or --end is required")

    failures: list[str] = []
    proxy_logs = discover_proxy_logs(args.log_dir)
    if not proxy_logs:
        print(f"error: no proxy logs found in {args.log_dir}", file=sys.stderr)
        return 1
    lines = read_log_lines(proxy_logs, failures)

    events = list(_collect_events(lines, start, end))
    # File names are not chronological (the rotation schemes are offset), so
    # sort by the authoritative line timestamp before the FIFO join — otherwise
    # a measurement in one file could be joined against a dispatch from an
    # earlier file, and the reported window bounds would follow file order.
    events.sort(key=lambda event: event.ts)
    paired = join_measurements(events, args.join_window_s)

    llama = None
    engine_tok_s = None
    if args.llama_log:
        if not args.llama_log.is_file():
            print(f"error: llama log not found: {args.llama_log}", file=sys.stderr)
            return 1
        llama = parse_llama_reuse(read_log_lines([args.llama_log], failures))
        engine_tok_s = llama.get("prompt_eval_aggregate_tok_s") or None

    summary = summarise(
        paired,
        args.max_first_byte_ms,
        args.min_estimated_tokens,
        engine_tok_s=engine_tok_s,
    )

    if args.json:
        result = {
            "meta": {
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "log_dir": str(args.log_dir),
                "proxy_files": [path.name for path in proxy_logs],
                "unreadable_files": failures,
                "llama_log": str(args.llama_log) if args.llama_log else None,
                "join_window_s": args.join_window_s,
            },
            "summary": summary,
        }
        if llama is not None:
            result["llama_server_cross_check"] = llama
        print(json.dumps(result, indent=2, sort_keys=False))
    else:
        print(_render(summary, llama))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
