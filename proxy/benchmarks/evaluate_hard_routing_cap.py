#!/usr/bin/env python3
"""Hard local-routing cap evaluation -- replay llama-server log data.

Replays prefill and decode events from llama-server logs to quantify the
impact of a proposed hard local-routing cap (LP-0MTAQNAIH001RN1S):

1. Prefill-volume projection: how many prefill events / tokens a candidate
   hard cap would gate (never dispatched local), per mode.
2. Decode-contention correlation: how often decode collapses (<5 t/s,
   severe <2 t/s) while ANOTHER slot runs a large prefill -- the iGPU
   memory-bandwidth saturation mechanism (0.14-0.2 t/s observations,
   2026-08-26 incident).
3. Mode-specific thresholds from the live configs.

Methodology note: llama-server logs carry no wall-clock timestamps, so
event proximity is measured in sequential-log lines within ``--window``
lines (default 800, tuned to the observed decode-collapse signal). The
current slot is tracked from each ``slot update_slots: id X`` line; eval
lines are attributed to that slot. "Cross-slot" prefills are those of any
OTHER slot -- the shared-memory bandwidth consumers that starve the
decoding slot.

Usage:
    python3 proxy/benchmarks/evaluate_hard_routing_cap.py \
        --logs /var/log/llama-proxy/llama-server.log* \
        [--mode fast|cheap] [--hard-cap 70000]
"""
import argparse
import glob
import json
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants (mirroring proxy/provider.py and the mode configs)
# ---------------------------------------------------------------------------
OUTPUT_HEADROOM = 4096

MODES = {
    "fast": {
        # config-fast.yaml / supersede LP-0MSY0SDAS0031Y7F:
        # 3 slots x 262144 (per-slot 87381, clamp 83285).
        "cold_cache_threshold": 38000,
        "warm_cache_threshold_config": 100000,
        "local_model_ctx_size": 262144,
        "session_slot_pool_size": 3,
        "routing_clamp": 262144 // 3 - OUTPUT_HEADROOM,  # 83285
    },
    "cheap": {
        # config-cheap.yaml / LP-0MSMZOAJW002UR2A:
        # 2 slots x 262144 (per-slot 131072, clamp min(100000, ...)=100000).
        "cold_cache_threshold": 42000,
        "warm_cache_threshold_config": 100000,
        "local_model_ctx_size": 262144,
        "session_slot_pool_size": 2,
        "routing_clamp": min(100000, 262144 // 2 - OUTPUT_HEADROOM),  # 100000
    },
}

# Regex patterns for llama-server log lines
SLOT_LINE = re.compile(r"slot update_slots: id\s+(\d+)\s+\|\s+task\s+(\d+)")
NEW_PROMPT = re.compile(r"new prompt,.*task\.n_tokens\s*=\s*(\d+)")
PREFILL_DONE = re.compile(
    r"prompt processing done,\s+n_tokens\s*=\s*(\d+),\s+batch\.n_tokens\s*=\s*(\d+)"
)
EVAL_TIME = re.compile(
    r"eval time\s*=\s*([\d.]+)\s+ms\s+/\s+(\d+)\s+tokens\s+"
    r"\(\s*([\d.]+)\s+ms per token,\s+([\d.]+)\s+tokens per second\)"
)


def warm_threshold(mode_cfg: dict) -> int:
    """The runtime-clamped warm threshold (hard capacity limit)."""
    return min(mode_cfg["warm_cache_threshold_config"], mode_cfg["routing_clamp"])


def parse_log_files(paths: list[str]) -> list[dict]:
    """Parse llama-server log files into slot-attributed events.

    Returns a list of dicts with keys: line, slot, type (new_prompt /
    prefill_done / eval), and payload fields (n_tokens, tps, ...).
    """
    events = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)):
            cur_slot = None
            try:
                with open(path, errors="replace") as f:
                    for line_no, line in enumerate(f):
                        m = SLOT_LINE.search(line)
                        if m:
                            cur_slot = int(m.group(1))
                        pd = PREFILL_DONE.search(line)
                        if pd:
                            events.append({
                                "line": line_no, "slot": cur_slot,
                                "type": "prefill_done",
                                "n_tokens": int(pd.group(1)),
                            })
                            continue
                        et = EVAL_TIME.search(line)
                        if et:
                            events.append({
                                "line": line_no, "slot": cur_slot,
                                "type": "eval",
                                "eval_time_ms": float(et.group(1)),
                                "eval_tokens": int(et.group(2)),
                                "ms_per_token": float(et.group(3)),
                                "tps": float(et.group(4)),
                            })
                            continue
            except OSError as exc:
                print(f"skip {path}: {exc}", file=sys.stderr)
    return events


def summarize_prefills(events: list[dict], mode_cfg: dict,
                       hard_cap: int | None = None,
                       cap_basis: str = "n_tokens") -> dict:
    """Prefill distribution and gating summary for a mode.

    ``hard_cap``: candidate hard routing cap on the per-request estimate.
    Events above the cap (and above the current clamped warm threshold)
    are classified ``above_cap``; the rest retain the current-policy
    bucket (under_cold / in_band / context_too_large).
    """
    prefills = [e for e in events if e["type"] == "prefill_done"]
    n = len(prefills)
    if n == 0:
        return {"n": 0}

    counts = defaultdict(int)
    tokens = defaultdict(int)
    for p in prefills:
        size = p[cap_basis]
        cold = mode_cfg["cold_cache_threshold"]
        warm = warm_threshold(mode_cfg)
        if hard_cap is not None and size > hard_cap:
            kind = "above_cap"
        elif size > warm:
            kind = "context_too_large"
        elif size <= cold:
            kind = "under_cold"
        else:
            kind = "in_band"
        counts[kind] += 1
        tokens[kind] += p["n_tokens"]

    total_tokens = sum(tokens.values())
    summary = {
        "n": n,
        "total_tokens": total_tokens,
        "mean_tokens": int(total_tokens / n),
        "max_tokens": max(p["n_tokens"] for p in prefills),
        "hard_cap": hard_cap,
    }
    for kind in ("under_cold", "in_band", "context_too_large", "above_cap"):
        c = counts.get(kind, 0)
        t = tokens.get(kind, 0)
        if c:
            summary[kind] = {"count": c, "pct": 100.0 * c / n,
                             "tokens": t, "pct_tokens": 100.0 * t / max(total_tokens, 1)}
    return summary


def decode_collapse_analysis(events: list[dict], window: int = 800) -> dict:
    """Correlate decode collapses with CROSS-slot large prefills.

    For each eval: find the largest prefill of a DIFFERENT slot within
    ``window`` lines (the bandwidth consumer), bucket decode speed by it.
    Also reports how many collapses are explained by a nearby >=50K
    cross-slot prefill.
    """
    evals = [e for e in events if e["type"] == "eval"]
    prefills = [e for e in events if e["type"] == "prefill_done"]
    n_eval = len(evals)
    if n_eval == 0:
        return {"n": 0}

    buckets: dict[str, list] = defaultdict(list)
    collapsed_total = 0
    collapsed_with_big_cross = 0
    for e in evals:
        tps = e["tps"]
        cross = [
            p["n_tokens"] for p in prefills
            if p["slot"] != e["slot"] and abs(p["line"] - e["line"]) <= window
        ]
        cross_max = max(cross) if cross else 0
        if cross_max < 10000:
            key = "cross<10K"
        elif cross_max < 30000:
            key = "10-30K"
        elif cross_max < 50000:
            key = "30-50K"
        else:
            key = "50K+"
        buckets[key].append(tps)

        if tps < 5.0:
            collapsed_total += 1
            if cross_max >= 50000:
                collapsed_with_big_cross += 1

    return {
        "n": n_eval,
        "collapsed_lt5": collapsed_total,
        "collapsed_lt5_pct": 100.0 * collapsed_total / n_eval,
        "collapsed_with_50k_cross_prefill": collapsed_with_big_cross,
        "buckets": {
            k: {
                "count": len(v),
                "avg_tps": round(sum(v) / len(v), 2),
                "min_tps": round(min(v), 2),
                "collapsed_lt5": sum(1 for t in v if t < 5.0),
                "collapsed_lt2": sum(1 for t in v if t < 2.0),
            }
            for k, v in sorted(buckets.items())
        },
        "window_lines": window,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", nargs="+", required=True,
                        help="llama-server log path(s) or glob patterns")
    parser.add_argument("--mode", choices=["fast", "cheap"], default="fast",
                        help="Mode profile whose thresholds are applied")
    parser.add_argument("--hard-cap", type=int, default=None,
                        help="Candidate hard routing cap (absent = current policy)")
    parser.add_argument("--window", type=int, default=800,
                        help="Line-distance window for cross-slot correlation")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args(argv)

    events = parse_log_files(args.logs)
    if not events:
        print("No events parsed from logs", file=sys.stderr)
        return 1

    mode_cfg = MODES[args.mode]
    warm = warm_threshold(mode_cfg)
    cold = mode_cfg["cold_cache_threshold"]
    print(f"Mode: {args.mode} (cold {cold}, warm-clamped {warm}, "
          f"routing_clamp {mode_cfg['routing_clamp']})")
    print(f"Events: {len(events)} "
          f"({sum(1 for e in events if e['type']=='prefill_done')} prefills, "
          f"{sum(1 for e in events if e['type']=='eval')} evals)\n")

    current = summarize_prefills(events, mode_cfg)
    cap = None if not args.hard_cap else args.hard_cap
    capped = summarize_prefills(events, mode_cfg, hard_cap=cap)

    print("--- Prefill distribution (current policy) ---")
    for kind in ("under_cold", "in_band", "context_too_large"):
        if kind in current:
            s = current[kind]
            print(f"  {kind:>18}: {s['count']:>5} events ({s['pct']:.1f}%), "
                  f"{s['tokens']:>10} tokens ({s['pct_tokens']:.1f}%)")
    print(f"  Total: {current['n']} events, {current['total_tokens']} tokens "
          f"(mean {current['mean_tokens']}, max {current['max_tokens']})\n")

    if cap:
        print(f"--- With hard cap {cap} ---")
        if "above_cap" in capped:
            s = capped["above_cap"]
            print(f"  {'above_cap':>18}: {s['count']:>5} events ({s['pct']:.1f}%), "
                  f"{s['tokens']:>10} tokens ({s['pct_tokens']:.1f}%)")
        for kind in ("under_cold", "in_band", "context_too_large"):
            if kind in capped:
                s = capped[kind]
                print(f"  {kind:>18}: {s['count']:>5} events ({s['pct']:.1f}%), "
                      f"{s['tokens']:>10} tokens ({s['pct_tokens']:.1f}%)")
        extra = capped.get("above_cap", {}).get("count", 0) - \
            current.get("context_too_large", {}).get("count", 0)
        extra_tokens = capped.get("above_cap", {}).get("tokens", 0) - \
            current.get("context_too_large", {}).get("tokens", 0)
        print(f"  → additionally gated beyond current clamp: {extra} events / "
              f"{extra_tokens} tokens (now routed remote or compacted)\n")

    dec = decode_collapse_analysis(events, window=args.window)
    print(f"--- Decode collapse vs CROSS-slot prefill (line window {args.window}) ---")
    print(f"  Evals: {dec['n']}")
    print(f"  Collapsed <5 t/s: {dec['collapsed_lt5']} "
          f"({dec['collapsed_lt5_pct']:.1f}%), of which with a >=50K "
          f"cross-slot prefill nearby: {dec['collapsed_with_50k_cross_prefill']}")
    print("  Avg decode t/s by cross-slot prefill size:")
    for k, v in dec["buckets"].items():
        print(f"    {k:>8}: n={v['count']:>5}  avg={v['avg_tps']:>6.2f}  "
              f"min={v['min_tps']:>6.2f}  <5tps={v['collapsed_lt5']} "
              f"(<2tps={v['collapsed_lt2']})")

    if args.json:
        out = {
            "mode": args.mode,
            "thresholds": {"cold": cold, "warm_clamped": warm,
                           "routing_clamp": mode_cfg["routing_clamp"]},
            "current": current,
            "with_cap": capped,
            "decode": dec,
        }
        print("\n" + json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())