#!/usr/bin/env python3
"""F5: Session-retention projection per config (LP-0MSC95W7M009VMEM).

Loads the proxy-usage daytime session CSV and projects, per candidate
config, which sessions would stay fully local (no remote fallback).

Methodology:
- A session's peak context (max_context_size) is compared against each
  config's effective routing clamp (per_slot_ctx - 4096 output headroom).
  If max_context <= clamp, the session is not blocked by the
  warm_cache_bypass / large_context_bypass path (context-based routing).
- Concurrency: a config with S slots can serve at most S concurrent
  sessions. We project concurrency-limited sessions separately using the
  dispatch_denied / local_concurrency_limit signal from the CSV.
- Persistence-cap variant: session_slot_max_prompt_tokens (12288) skips
  KV save/restore above 12.3K prompt tokens, forcing cold re-prefill on
  handoff. Raising the cap changes warm-cache behavior, so sessions whose
  max_context exceeds the cap but is below the routing clamp would benefit.

Usage:
    python3 proxy/benchmarks/project_session_retention.py \
        --csv ~/proxy-usage-reports/2026-08-03/daytime_sessions.csv \
        [--json]
"""
import argparse
import csv
import json
import sys

OUTPUT_HEADROOM = 4096
PERSISTENCE_CAP_CURRENT = 12288


def per_slot_ctx(total_ctx: int, slots: int) -> int:
    return total_ctx // slots


def routing_clamp(total_ctx: int, slots: int) -> int:
    return per_slot_ctx(total_ctx, slots) - OUTPUT_HEADROOM


def classify_session(row: dict, clamp: int, slots: int, peak_concurrency: int | None = None) -> dict:
    """Classify whether a session stays full-local under a config.

    If ``peak_concurrency`` is provided, a context-eligible session is
    additionally gated by slot availability: with S slots and peak
    concurrency C, the probability a session holds a slot when needed is
    modeled as ``min(1.0, S / C)`` (first-order approximation; real slot
    scheduling is dynamic).
    """
    max_ctx = float(row.get("max_context_size") or 0)
    local_req = int(row.get("local_requests") or 0)
    remote_req = int(row.get("remote_requests") or 0)
    fallback = (row.get("fallback_reason") or "").strip()
    dispatch_denied = int(row.get("dispatch_denied") or 0)

    # Context-based routing: does peak context fit the clamp?
    ctx_ok = max_ctx <= clamp

    # Concurrency: was this session denied local due to slot scarcity?
    concurrency_blocked = (
        fallback == "local_concurrency_limit"
        or dispatch_denied > 0
        or fallback == "local_lease_active"
    )

    # Actual outcome in the observed config (4 slots, clamp 61440 = Option 1)
    actually_local = local_req > 0 and remote_req == 0

    if ctx_ok and not concurrency_blocked:
        projection = "full_local"
    elif not ctx_ok:
        projection = "context_bypass"  # would bypass local on context
    else:
        projection = "concurrency_blocked"

    # Concurrency-aware retention probability: how likely a context-eligible
    # session gets a slot under this config's slot count.
    slot_avail = 1.0
    if peak_concurrency and peak_concurrency > 0:
        slot_avail = min(1.0, slots / peak_concurrency)

    return {
        "session_id": row.get("session_id"),
        "max_context": max_ctx,
        "local_requests": local_req,
        "remote_requests": remote_req,
        "fallback_reason": fallback,
        "clamp": clamp,
        "slots": slots,
        "projection": projection,
        "ctx_ok": ctx_ok,
        "concurrency_blocked": concurrency_blocked,
        "actually_local": actually_local,
        "slot_avail": slot_avail,
        "expected_local": (slot_avail if ctx_ok and not concurrency_blocked else 0.0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to daytime_sessions.csv")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args(argv)

    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        print("No sessions in CSV", file=sys.stderr)
        return 1

    # Configs: (name, total_ctx, slots)
    configs = [
        ("baseline 3x43.7K", 131072, 3),
        ("4x65.5K", 262144, 4),
        ("3x87.4K", 262144, 3),
        ("2x131K", 262144, 2),
    ]

    # Only sessions with context data can be classified on the context axis.
    with_ctx = [r for r in rows if r.get("max_context_size")]
    print(f"Sessions: {len(rows)} total, {len(with_ctx)} with max_context data\n")

    # Peak concurrent sessions from start/end times (drives slot-availability).
    import datetime

    def _peak_concurrency() -> int:
        events = []
        for r in rows:
            st, et = r.get("start_time"), r.get("end_time")
            if not st or not et:
                continue
            try:
                s0 = datetime.datetime.strptime(st, "%Y-%m-%d %H:%M:%S")
                e0 = datetime.datetime.strptime(et, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            events.append((s0, 1))
            events.append((e0, -1))
        if not events:
            return 0
        events.sort(key=lambda x: x[0])
        cur = peak = 0
        for _t, d in events:
            cur += d
            peak = max(peak, cur)
        return peak

    peak_conc = _peak_concurrency()
    print(f"Peak concurrent sessions: {peak_conc}\n")

    table = []
    for name, total_ctx, slots in configs:
        clamp = routing_clamp(total_ctx, slots)
        projections = [classify_session(r, clamp, slots, peak_conc) for r in with_ctx]
        full_local = sum(1 for p in projections if p["projection"] == "full_local")
        ctx_bypass = sum(1 for p in projections if p["projection"] == "context_bypass")
        conc_blocked = sum(1 for p in projections if p["projection"] == "concurrency_blocked")
        # Concurrency-aware expected local retention: context-eligible sessions
        # weighted by slot availability (slots / peak concurrency, capped at 1).
        expected_local = sum(p["expected_local"] for p in projections)
        expected_local_pct = 100 * expected_local / len(with_ctx) if with_ctx else 0.0

        # Persistence-cap variant: raise cap to clamp (so all ctx-ok sessions
        # get warm-cache persistence). Sessions whose max_context <= clamp but
        # > 12288 (current cap) would newly benefit.
        raised_cap_local = sum(
            1 for p in projections
            if p["projection"] == "full_local" or (p["ctx_ok"] and p["max_context"] > PERSISTENCE_CAP_CURRENT)
        )
        newly_full = sum(
            1 for p in projections
            if p["ctx_ok"] and p["max_context"] > PERSISTENCE_CAP_CURRENT
        )

        entry = {
            "config": name,
            "total_ctx": total_ctx,
            "slots": slots,
            "clamp": clamp,
            "n_sessions": len(with_ctx),
            "full_local": full_local,
            "full_local_pct": 100 * full_local / len(with_ctx),
            "context_bypass": ctx_bypass,
            "concurrency_blocked": conc_blocked,
            "expected_local": round(expected_local, 1),
            "expected_local_pct": round(expected_local_pct, 1),
            "raised_cap_full_local": raised_cap_local,
            "raised_cap_full_local_pct": 100 * raised_cap_local / len(with_ctx),
            "newly_full_above_cap": newly_full,
        }
        table.append(entry)

        print(f"--- {name} (clamp {clamp}, {slots} slots) ---")
        print(f"  full_local (context-only):  {full_local:>3}/{len(with_ctx)} ({100*full_local/len(with_ctx):.1f}%)")
        print(f"  expected_local (concurrency-aware): {expected_local:.1f} ({expected_local_pct:.1f}%)")
        print(f"  context_bypass:        {ctx_bypass:>3} (peak ctx > clamp)")
        print(f"  concurrency_blocked:   {conc_blocked:>3}")
        print(f"  raised-cap full_local: {raised_cap_local:>3} ({100*raised_cap_local/len(with_ctx):.1f}%)  [+{newly_full} sessions above 12.3K cap]")
        print()

    # Identify sessions that become full-local under raised cap for 2x131K
    best = max(table, key=lambda t: t["raised_cap_full_local"])
    clamp = best["clamp"]
    newly_sessions = [
        classify_session(r, clamp, best["slots"])["session_id"]
        for r in with_ctx
        if float(r.get("max_context_size") or 0) <= clamp
        and float(r.get("max_context_size") or 0) > PERSISTENCE_CAP_CURRENT
    ]
    print(f"Sessions becoming full-local under raised cap ({best['config']}): {len(newly_sessions)}")
    for sid in newly_sessions[:20]:
        print(f"  {sid}")
    if len(newly_sessions) > 20:
        print(f"  ... and {len(newly_sessions)-20} more")

    if args.json:
        print("\n" + json.dumps({"configs": table, "newly_full_sessions": newly_sessions}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
