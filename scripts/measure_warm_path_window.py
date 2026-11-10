#!/usr/bin/env python3
"""Measure the large-prompt warm path over a proxy log window.

Why this exists (LP-0MVW8S8OZ00449YR)
-------------------------------------

The post-change ``prefill_deadline`` measurement must show **no warm-path or
throughput regression** against a matched pre-change window (AC4/AC5 of
LP-0MVVEQW5Y00962T2, evidence child LP-0MVW8S8OZ00449YR). Two of the three
metrics that comparison needs were only ever hand-derived:

* local completions per hour, and
* large-prompt **warm** completion latency p50/p90.

The sibling instruments cover the rest: ``analyse_prefill_deadline_residual.py``
owns the eviction reasons and attribution, and
``dispatch_latency_verification.py`` owns warm/cold service latency for *all*
prompt sizes. Neither restricts the warm latency population to large prompts,
which is the population the abandoned-drain fix and the measured-cache abort
both target — so this tool supplies exactly that slice, from the same log
surfaces, and records the window it read.

What is measured (and why each field is the one it is)
-----------------------------------------------------

``dispatch_first_byte_ms`` lines (one per local dispatch, carrying the prompt
estimate and the proxy's own cache label)::

    dispatch_first_byte_ms=65394.3 dispatch_to_first_byte_ms=65392.6 \
      cache_state=restored cached_ratio=0.879 queue_wait_ms=1.7 \
      session=<id> model=<m> priority=<p> estimated_tokens=58797 turn=<8 hex>

``Stream finished`` lines (one per completed stream, carrying measured usage)::

    Stream finished: reason=<r> tokens=<prompt>/<completion>/<total> \
      session=<id> provider=<provider> model=<m> request=[...]

Metrics:

``local_completions``
    ``Stream finished`` lines whose provider is exactly ``local`` and that
    carry measured usage. This is the throughput denominator: the client
    received a complete local stream. ``tokens=missing`` lines are excluded
    (no measured prompt size, so no band can be assigned).
``large``
    prompt estimate (dispatch line ``estimated_tokens``, completion line
    prompt tokens) ``>= --large-tokens`` (default 20 000). This is the band the
    residual evictions and the early-abort decisions live in.
``warm``
    the proxy's own ``cache_state=warm`` label. ``cold`` and ``restored`` are
    reported separately instead of being folded in: ``restored`` is a distinct
    cache path (KV restored, not served warm by the resident prefix), and
    merging it would change the definition the earlier published tables used.
``service_ms``
    ``dispatch_to_first_byte_ms`` — first byte **minus** the single-slot queue
    wait. Queue wait is admission-side, so service time is the prefill+first
    token cost the warm path is held to.
``first_byte_under_threshold``
    share of large warm dispatches reaching a first byte under
    ``--first-byte-fast-ms`` (default 10 000 ms), the "warm and actually fast"
    shape the earlier windows reported as ``48 of 59``.

Coverage is measured with the shared :mod:`lib.proxy_logs` implementation, so a
window whose rotated segments were pruned is reported as an uncovered window
rather than as a small sample (the 2026-10-29 segments behind the pre-change
baseline have rotated away; only windows the logs still cover are measurable).

Usage::

    ./scripts/measure_warm_path_window.py \
        --since "2026-11-06 14:15:56" --until "2026-11-07 14:15:56" --json

Exit codes: 0 — measured; 1 — no proxy logs found or unexpected error.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.proxy_logs import (
    iter_proxy_log_lines,
    measure_log_coverage,
    parse_line_timestamp,
)

#: Default prompt size (tokens) at which a dispatch counts as "large" — the band
#: the residual evictions, the measured-cache early abort and the earlier
#: published warm tables all use.
DEFAULT_LARGE_TOKENS = 20_000
#: Default "fast enough" first byte for the warm share metric (10 s), matching
#: the published "large warm completion" definition.
DEFAULT_FIRST_BYTE_FAST_MS = 10_000.0

_TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")

#: ``dispatch_first_byte_ms`` line (local dispatch only). ``estimated_tokens``
#: became optional in the grammar, so it is parsed separately and a dispatch
#: without it is counted as an unknown-size dispatch rather than mis-banded.
_DISPATCH_RE = re.compile(
    r"dispatch_first_byte_ms=(?P<first_byte>[0-9.]+) "
    r"dispatch_to_first_byte_ms=(?P<service>[0-9.]+) "
    r"cache_state=(?P<state>[A-Za-z]+) "
    r"cached_ratio=(?P<ratio>[0-9.]+) "
    r"queue_wait_ms=(?P<queue_wait>[0-9.]+)"
)
_ESTIMATED_TOKENS_RE = re.compile(r"\bestimated_tokens=(?P<tokens>\d+)")

#: ``Stream finished`` terminal line. Provider names may contain spaces
#: (``provider=OpenCode Go 2``), so the value is captured up to ``model=``
#: rather than as a single token; anchoring on the fixed field order keeps a
#: quoted log line inside ``request=[...]`` from being read as the provider.
_STREAM_FINISHED_RE = re.compile(
    r"Stream finished: .*?\btokens=(?P<prompt>\d+)/(?P<completion>\d+)/(?P<total>\d+) "
    r"session=\S+ provider=(?P<provider>.+?) model=\S+"
)


def _parse_bound(text: str) -> dt.datetime:
    for fmt in _TS_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"invalid timestamp {text!r} (expected 'YYYY-MM-DD [HH:MM[:SS]]')")


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile; empty list -> 0.0.

    Same convention as ``scripts/dispatch_latency_verification.py`` so the two
    instruments report comparable percentiles for the windows they share.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _dist_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "p50": round(_percentile(values, 50), 1),
        "p90": round(_percentile(values, 90), 1),
        "p99": round(_percentile(values, 99), 1),
        "max": round(max(values), 1),
    }


def analyse(
    log_dir: Path,
    *,
    since: dt.datetime | None = None,
    until: dt.datetime | None = None,
    large_tokens: int = DEFAULT_LARGE_TOKENS,
    first_byte_fast_ms: float = DEFAULT_FIRST_BYTE_FAST_MS,
) -> dict:
    """Return the warm-path metrics for the window ``[since, until)``."""
    completion_prompts: list[float] = []
    large_completions = 0
    completion_usage_missing = 0

    dispatch_total = 0
    dispatch_other_provider = 0
    unknown_size = 0
    state_breakdown: dict[str, int] = {}
    warm_service: list[float] = []
    large_warm_service: list[float] = []
    large_warm_first_byte: list[float] = []
    large_cold_service: list[float] = []
    large_unknown_state_service: list[float] = []
    first_seen: dt.datetime | None = None
    last_seen: dt.datetime | None = None
    lines_in_window = 0

    for _path, line in iter_proxy_log_lines(log_dir):
        ts = parse_line_timestamp(line)
        if ts is None:
            continue
        if since is not None and ts < since:
            continue
        if until is not None and ts >= until:
            continue
        lines_in_window += 1
        if first_seen is None:
            first_seen = ts
        last_seen = ts

        if "Stream finished" in line:
            if "tokens=missing" in line:
                completion_usage_missing += 1
                continue
            match = _STREAM_FINISHED_RE.search(line)
            if not match:
                continue
            if match.group("provider").strip() != "local":
                continue
            prompt_tokens = float(match.group("prompt"))
            completion_prompts.append(prompt_tokens)
            if prompt_tokens >= large_tokens:
                large_completions += 1
            continue

        match = _DISPATCH_RE.search(line)
        if not match:
            continue
        dispatch_total += 1
        state = match.group("state")
        state_breakdown[state] = state_breakdown.get(state, 0) + 1
        service = float(match.group("service"))
        first_byte = float(match.group("first_byte"))
        size_match = _ESTIMATED_TOKENS_RE.search(line)
        size = int(size_match.group("tokens")) if size_match else None
        if state == "warm":
            warm_service.append(service)
        if size is None:
            unknown_size += 1
            continue
        if size < large_tokens:
            continue
        if state == "warm":
            large_warm_service.append(service)
            large_warm_first_byte.append(first_byte)
        elif state == "cold":
            large_cold_service.append(service)
        else:
            large_unknown_state_service.append(service)

    if since is not None and until is not None:
        hours = (until - since).total_seconds() / 3600.0
    elif first_seen is not None and last_seen is not None:
        hours = max((last_seen - first_seen).total_seconds() / 3600.0, 0.0)
    else:
        hours = 0.0

    def _per_hour(count: int) -> float:
        return round(count / hours, 3) if hours > 0 else 0.0

    fast_hits = sum(1 for value in large_warm_first_byte if value < first_byte_fast_ms)
    coverage = measure_log_coverage(log_dir, since, until)

    return {
        "label": None,
        "window": {
            "since": since.isoformat(sep=" ") if since else None,
            "until": until.isoformat(sep=" ") if until else None,
            "hours": round(hours, 3),
            "first_seen": first_seen.isoformat(sep=" ") if first_seen else None,
            "last_seen": last_seen.isoformat(sep=" ") if last_seen else None,
            "lines_in_window": lines_in_window,
        },
        "definitions": {
            "large_tokens": large_tokens,
            "warm": "proxy cache_state=warm on the dispatch_first_byte_ms line",
            "service_ms": "dispatch_to_first_byte_ms (first byte minus queue wait)",
            "first_byte_fast_ms": first_byte_fast_ms,
            "local_completion": ("Stream finished with provider=local and measured tokens"),
        },
        "coverage": {
            "covered": coverage.covered,
            "files_with_lines": len(coverage.spans),
            "gaps": [
                {
                    "after": gap.after.name,
                    "before": gap.before.name,
                    "gap_seconds": round(gap.gap_seconds, 1),
                }
                for gap in coverage.gaps
            ],
            "window_start_gap_seconds": (
                round(coverage.window_start_gap_seconds, 1) if coverage.window_start_gap_seconds is not None else None
            ),
            "window_end_gap_seconds": (
                round(coverage.window_end_gap_seconds, 1) if coverage.window_end_gap_seconds is not None else None
            ),
        },
        "local_completions": {
            "total": len(completion_prompts),
            "per_hour": _per_hour(len(completion_prompts)),
            "usage_missing": completion_usage_missing,
            "prompt_tokens": _dist_stats(completion_prompts),
            "large": {
                "total": large_completions,
                "per_hour": _per_hour(large_completions),
            },
        },
        "dispatches": {
            "total": dispatch_total,
            "per_hour": _per_hour(dispatch_total),
            "cache_state_breakdown": dict(sorted(state_breakdown.items())),
            "unknown_prompt_estimate": unknown_size,
            "other_provider_events_ignored": dispatch_other_provider,
        },
        "warm": {
            "total": len(warm_service),
            "per_hour": _per_hour(len(warm_service)),
            "service_ms": _dist_stats(warm_service),
        },
        "large_warm": {
            "total": len(large_warm_service),
            "per_hour": _per_hour(len(large_warm_service)),
            "service_ms": _dist_stats(large_warm_service),
            "first_byte_under_threshold": {
                "threshold_ms": first_byte_fast_ms,
                "count": fast_hits,
                "share": round(fast_hits / len(large_warm_service), 4) if large_warm_service else None,
            },
        },
        "large_cold": {
            "total": len(large_cold_service),
            "service_ms": _dist_stats(large_cold_service),
        },
        "large_other_cache_state": {
            "total": len(large_unknown_state_service),
            "service_ms": _dist_stats(large_unknown_state_service),
        },
    }


def _render(report: dict) -> str:
    window = report["window"]
    definitions = report["definitions"]
    completions = report["local_completions"]
    warm = report["large_warm"]
    cold = report["large_cold"]
    coverage = report["coverage"]
    lines = [
        "Warm-path window measurement",
        "=" * 30,
        f"window          : {window['since']} -> {window['until']} ({window['hours']} h)",
        f"lines in window : {window['lines_in_window']} across {coverage['files_with_lines']} files",
        f"coverage        : {'covered' if coverage['covered'] else 'GAPS'} "
        f"(adjacent-file gaps: {len(coverage['gaps'])})",
        f"large threshold : {definitions['large_tokens']} prompt tokens",
        "",
        f"local completions      : {completions['total']} ({completions['per_hour']}/h)",
        f"large local completions: {completions['large']['total']} ({completions['large']['per_hour']}/h)",
        f"large warm dispatches  : {warm['total']} ({warm['per_hour']}/h)",
        f"large warm service ms  : p50 {warm['service_ms']['p50']} / "
        f"p90 {warm['service_ms']['p90']} / max {warm['service_ms']['max']}",
        f"large warm < {definitions['first_byte_fast_ms']:.0f} ms: "
        f"{warm['first_byte_under_threshold']['count']} of {warm['total']} "
        f"(share {warm['first_byte_under_threshold']['share']})",
        f"large cold service ms  : p50 {cold['service_ms']['p50']} / p90 {cold['service_ms']['p90']}",
    ]
    if coverage["gaps"]:
        lines.append("")
        lines.append("WARNING: the window is not fully covered by the retained logs:")
        for gap in coverage["gaps"]:
            lines.append(f"  {gap['before']} -> {gap['after']} ({gap['gap_seconds']} s missing)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="measure_warm_path_window.py",
        description=(
            "Measure local completions per hour and large-prompt warm "
            "dispatch service latency over a proxy log window "
            "(LP-0MVW8S8OZ00449YR AC5)."
        ),
        epilog=(
            "Example: measure_warm_path_window.py --since '2026-11-06 14:15:56' --until '2026-11-07 14:15:56' --json"
        ),
    )
    parser.add_argument("--log-dir", default="/var/log/llama-proxy")
    parser.add_argument("--since", required=True, help="inclusive lower bound")
    parser.add_argument("--until", required=True, help="exclusive upper bound")
    parser.add_argument(
        "--large-tokens",
        type=int,
        default=DEFAULT_LARGE_TOKENS,
        help=f"prompt tokens that make a dispatch 'large' (default {DEFAULT_LARGE_TOKENS})",
    )
    parser.add_argument(
        "--first-byte-fast-ms",
        type=float,
        default=DEFAULT_FIRST_BYTE_FAST_MS,
        help=(f"first-byte threshold for the 'warm and fast' share (default {DEFAULT_FIRST_BYTE_FAST_MS:.0f})"),
    )
    parser.add_argument("--label", help="free-form label recorded in the output")
    parser.add_argument("--output", help="write the JSON report here")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"error: no proxy logs at {log_dir}", file=sys.stderr)
        return 1
    report = analyse(
        log_dir,
        since=_parse_bound(args.since),
        until=_parse_bound(args.until),
        large_tokens=args.large_tokens,
        first_byte_fast_ms=args.first_byte_fast_ms,
    )
    report["label"] = args.label
    if report["window"]["lines_in_window"] == 0:
        print(
            f"error: no log lines in {args.since} .. {args.until} (rotated away?)",
            file=sys.stderr,
        )
        payload = json.dumps(report, indent=2, sort_keys=True)
        if args.json:
            print(payload)
        return 1
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
