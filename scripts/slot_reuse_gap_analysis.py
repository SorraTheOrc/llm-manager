#!/usr/bin/env python3
"""Save/restore reuse-gap root-cause analysis (F2, LP-0MTCMEOHB002X1JN).

Correlates slot save vs restore events and gating factors to explain why
~95% of context checkpoints are never reused (the 2026-08-26 incident),
with timeline correlation to snapshot writes (22:02-23:09) and proxy state.

Factors analysed (per-factor counts from the F1 corpus):
  1. Context-size gating     — routing_skip_local reason=context_too_large /
                               large_context_bypass (sessions too big to ever
                               get persistence)
  2. Load-aware gating       — slot_save failures under concurrent load
                               (busy_info.slot_busy / PoolTimeout)
  3. Circuit-breaker cooldown— slot persistence disabled after consecutive
                               failures (persistence_cooldown events)
  4. slots_stale             — status_request polls with slots_stale=true
  5. GET /slots 500s         — llama-server access-log GET /slots 500 counts
  6. Lease churn / affinity  — lease_released reason=orphan_cleanup (stream
                               abandoned → session↔slot mapping broken) and
                               reason=session_evicted
  7. Cross-reference        — per-session: sessions gated out by
                               context_too_large NEVER get a restore (the
                               oversized-session gap)

Usage:
  ./scripts/slot_reuse_gap_analysis.py                                # defaults
  ./scripts/slot_reuse_gap_analysis.py --log-dir /var/log/llama-proxy
  ./scripts/slot_reuse_gap_analysis.py --start 2026-08-26 --end 2026-08-27
  ./scripts/slot_reuse_gap_analysis.py --llama-file '*2026-08-27*'
  ./scripts/slot_reuse_gap_analysis.py --json                         # JSON out
  ./scripts/slot_reuse_gap_analysis.py --summary                      # compact

Exit codes:
  0 - success
  1 - log directory missing / unexpected error
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Reuse the F1 harness parsers so the corpus stays consistent.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import slot_persistence_harness as harness  # noqa: E402

_TS_FMT = "%Y-%m-%d %H:%M:%S"


def _parse_ts(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, _TS_FMT)


def analyze_reuse_gap(
    log_dir: Path,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    llama_file_glob: str | None = None,
    half_hour_buckets: bool = False,
) -> dict:
    """Run the reuse-gap analysis over the F1 corpus + factor extraction.

    Returns:
        dict with keys:
          - factor_breakdown: per-factor counts & rates
          - hourly_timeline:  half-hour/HH:00 buckets of proxy events
          - llama_per_file:   per-file llama checkpoint creates/restores
          - session_cross_reference: session → {saves, restores, skips}
    """
    corpus = harness.analyze(log_dir, None, start, end, llama_file_glob)

    # ------------------------------------------------------------------
    # Factor 1: context-size gating (routing_skip_local reasons)
    # ------------------------------------------------------------------
    skip_reasons = Counter(
        e["details"].get("reason", "unknown")
        for e in corpus["skip_events"]
    )
    context_gating = {
        "context_too_large": skip_reasons.get("context_too_large", 0),
        "large_context_bypass": skip_reasons.get("large_context_bypass", 0),
        "total": skip_reasons.get("context_too_large", 0)
                 + skip_reasons.get("large_context_bypass", 0),
    }

    # ------------------------------------------------------------------
    # Factor 2: load-aware gating — slot_save failures under load
    # ------------------------------------------------------------------
    save_failures = [e for e in corpus["slot_save_events"] if e.get("status") == "failure"]
    load_gating = {
        "slot_save_failures": len(save_failures),
        "failures_with_slot_busy": sum(
            1 for e in save_failures
            if (e.get("busy_info") or {}).get("slot_busy") is True
        ),
        "failures_with_pool_timeout": sum(
            1 for e in save_failures if "PoolTimeout" in e.get("error", "")
        ),
    }

    # ------------------------------------------------------------------
    # Factor 3: circuit-breaker cooldown (persistence disabled events)
    # ------------------------------------------------------------------
    cooldown_events = [
        e for e in corpus["skip_events"]
        if e.get("event_type") == "persistence_cooldown"
    ]
    cooldown = {
        "persistence_disabled_events": len(cooldown_events),
    }

    # ------------------------------------------------------------------
    # Factor 4: slots_stale
    # ------------------------------------------------------------------
    stale_polls = [
        s for s in corpus["slots_status_codes"] if s.get("slots_stale")
    ]
    slots_stale = {
        "stale_polls": len(stale_polls),
        "total_polls": len(corpus["slots_status_codes"]),
        "stale_pct": _safe_pct(len(stale_polls), len(corpus["slots_status_codes"])),
    }

    # ------------------------------------------------------------------
    # Factor 5: GET /slots 500s (llama-server access log)
    # ------------------------------------------------------------------
    slots_500 = [a for a in corpus["llama_slots_access"] if a["status"] == 500]
    slots_total = len(corpus["llama_slots_access"])
    slots_500_factor = {
        "five_hundreds": len(slots_500),
        "total_polls": slots_total,
        "five_hundred_pct": _safe_pct(len(slots_500), slots_total),
    }

    # ------------------------------------------------------------------
    # Factor 6: lease churn / affinity breaks
    # ------------------------------------------------------------------
    lease_events = corpus["lease_events"]
    orphan_releases = [
        e for e in lease_events
        if e.get("reason") == "orphan_cleanup"
    ]
    evicted_releases = [
        e for e in lease_events
        if e.get("reason") == "session_evicted"
    ]
    lease_churn = {
        "total_lease_events": len(lease_events),
        "orphan_releases": len(orphan_releases),
        "evicted_releases": len(evicted_releases),
        "renewed": sum(1 for e in lease_events if e.get("event") == "lease_renewed"),
    }

    # ------------------------------------------------------------------
    # Proxy slot persistence restore rate + llama checkpoint restore rate
    # ------------------------------------------------------------------
    save_events = corpus["slot_save_events"]
    restore_events = corpus["slot_restore_events"]
    proxy_save_success = sum(1 for e in save_events if e.get("status") == "success")
    proxy_restore_success = sum(1 for e in restore_events if e.get("status") == "success")
    proxy_save_failure = sum(1 for e in save_events if e.get("status") == "failure")
    proxy_restore_failure = sum(1 for e in restore_events if e.get("status") == "failure")

    llama_created = len(corpus["llama_checkpoint_events"])
    llama_restored = len(corpus["llama_checkpoint_restore_events"])

    # ------------------------------------------------------------------
    # Timeline: bucket proxy events into HH:00 (or HH:30) slots
    # ------------------------------------------------------------------
    timeline: dict[str, dict] = defaultdict(lambda: defaultdict(int))

    def _bucket(ts_str: str) -> str:
        ts = _parse_ts(ts_str)
        if half_hour_buckets:
            hh = ts.strftime("%H")
            mm = "30" if ts.minute >= 30 else "00"
            return f"{hh}:{mm}"
        return ts.strftime("%H:00")

    for e in save_events:
        timeline[_bucket(e["ts"])]["saves"] += 1
    for e in restore_events:
        timeline[_bucket(e["ts"])]["restores"] += 1
    for e in corpus["skip_events"]:
        if e.get("event_type") == "routing_skip":
            timeline[_bucket(e["ts"])]["skips"] += 1
    for s in stale_polls:
        timeline[_bucket(s["ts"])]["stale_polls"] += 1
    for e in orphan_releases:
        timeline[_bucket(e["ts"])]["orphan_releases"] += 1
    for e in evicted_releases:
        timeline[_bucket(e["ts"])]["evicted_releases"] += 1

    # ------------------------------------------------------------------
    # Per-file llama checkpoint attribution (no timestamps in llama logs)
    # ------------------------------------------------------------------
    llama_per_file: dict[str, dict] = {}
    for name, stats in corpus["llama_files_seen"].items():
        llama_per_file[name] = {
            "created": stats["created_checkpoints"],
            "restored": stats["restored_checkpoints"],
        }

    # ------------------------------------------------------------------
    # Cross-reference: session-level saves/restores/skips
    # ------------------------------------------------------------------
    session_ref: dict[str, dict] = defaultdict(lambda: {"saves": 0, "restores": 0, "skips": 0})
    for e in save_events:
        s = e.get("session")
        if s:
            session_ref[s]["saves"] += 1
    for e in restore_events:
        s = e.get("session")
        if s:
            session_ref[s]["restores"] += 1
    for e in corpus["skip_events"]:
        s = e["details"].get("session")
        if s:
            session_ref[s]["skips"] += 1

    return {
        "factor_breakdown": {
            "context_gating": context_gating,
            "load_gating": load_gating,
            "circuit_breaker_cooldown": cooldown,
            "slots_stale": slots_stale,
            "slots_500": slots_500_factor,
            "lease_churn": lease_churn,
            # restore-rate gap
            "proxy_slot_saves": proxy_save_success,
            "proxy_slot_save_failures": proxy_save_failure,
            "proxy_slot_restores": proxy_restore_success,
            "proxy_slot_restore_failures": proxy_restore_failure,
            "proxy_slot_restore_rate_pct": _safe_pct(proxy_restore_success, proxy_save_success),
            "llama_checkpoints_created": llama_created,
            "llama_checkpoints_restored": llama_restored,
            "llama_checkpoint_restore_rate_pct": _safe_pct(llama_restored, llama_created),
            "llama_checkpoints_unrestored_pct": _safe_pct(llama_created - llama_restored, llama_created),
        },
        "hourly_timeline": {k: dict(v) for k, v in sorted(timeline.items())},
        "llama_per_file": llama_per_file,
        "session_cross_reference": {
            k: dict(v) for k, v in sorted(session_ref.items())
        },
    }


def _safe_pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def render_markdown(res: dict) -> str:
    f = res["factor_breakdown"]
    lines = [
        "# Save/restore reuse-gap root-cause analysis",
        "",
        "## Factor breakdown",
        "",
        f"- Proxy slot persistence: {f['proxy_slot_saves']} saves → "
        f"{f['proxy_slot_restores']} restores "
        f"({f['proxy_slot_restore_rate_pct']}% restore rate; "
        f"{f['proxy_slot_save_failures']} save failures / "
        f"{f['proxy_slot_restore_failures']} restore failures)",
        f"- llama-server native checkpoints: {f['llama_checkpoints_created']} created → "
        f"{f['llama_checkpoints_restored']} restored "
        f"({f['llama_checkpoint_restore_rate_pct']}% restore rate; "
        f"{f['llama_checkpoints_unrestored_pct']}% unrestored — the ~95% gap)",
        "",
        "### Gating factors",
        "",
        f"- Context-size gating: {f['context_gating']}",
        f"- Load-aware gating: {f['load_gating']}",
        f"- Circuit-breaker cooldown: {f['circuit_breaker_cooldown']}",
        f"- slots_stale polls: {f['slots_stale']}",
        f"- GET /slots HTTP 500: {f['slots_500']}",
        f"- Lease churn: {f['lease_churn']}",
        "",
        "## Timeline (proxy events, per HH:00 bucket)",
        "",
    ]
    for bucket, counts in res["hourly_timeline"].items():
        lines.append(f"- `{bucket}`: {counts}")
    lines.append("")
    lines.append("## Per-file llama checkpoint attribution")
    lines.append("")
    for name, counts in res["llama_per_file"].items():
        lines.append(f"- `{name}`: created={counts['created']} restored={counts['restored']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="/var/log/llama-proxy",
                        help="directory containing proxy.log*/llama-server.log*")
    parser.add_argument("--start", help="start timestamp filter (YYYY-MM-DD[ HH:MM:SS])")
    parser.add_argument("--end", help="end timestamp filter (YYYY-MM-DD[ HH:MM:SS])")
    parser.add_argument("--llama-file", default=None,
                        help="restrict llama log parsing to a filename glob (e.g. '*2026-08-27*')")
    parser.add_argument("--json", action="store_true", default=True,
                        help="emit JSON output (default)")
    parser.add_argument("--summary", action="store_true",
                        help="emit compact summary (factor breakdown only)")
    parser.add_argument("--half-hour", action="store_true",
                        help="bucket timeline into 30-minute slots")
    parser.add_argument("--compact", action="store_true",
                        help="emit compact JSON (no indentation)")
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"error: log directory not found: {log_dir}", file=sys.stderr)
        return 1

    start = harness._parse_optional_ts(args.start)
    end = harness._parse_optional_ts(args.end)
    if start and end and start > end:
        print("error: --start must be before --end", file=sys.stderr)
        return 1

    try:
        res = analyze_reuse_gap(log_dir, start, end, args.llama_file, args.half_hour)
    except Exception as exc:
        print(f"error: analysis failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    if args.summary:
        print(json.dumps({"factor_breakdown": res["factor_breakdown"]},
                         indent=None if args.compact else 2, default=str))
    elif args.json:
        print(json.dumps(res, indent=None if args.compact else 2, default=str))
    else:
        print(render_markdown(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
