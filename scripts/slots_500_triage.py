#!/usr/bin/env python3
"""GET /slots HTTP 500-storm triage analysis (F3, LP-0MTCMEV1G0022A35).

Root-causes the ~9% HTTP 500 rate on GET /slots and the HTTP 400s, and
correlates 500-storm windows with concurrent giant prefills.

Key mechanism (from llama-server access-log evidence):
- The ROUTER (llama-server in router mode) answers every GET /slots call.
  Model instances answer only 200.
- When the router proxies GET /slots to a busy model instance (mid
  giant-prefill), the model cancels the proxied connection
  ('operator(): http client error: Connection handling canceled') and the
  router returns HTTP 500.
- HTTP 400s arise when the router answers /slots without a resolvable
  model (llama-server requires ?model= — LP-0MSHW2AXJ009DO3S).

Produced outputs (JSON):
  - classification: 500s/400s by responder (router vs model instance) and
    proximate cause (router-proxy-cancel, transient-busy, restart-race)
  - correlation: 500-window ↔ prefill/checkpoint/cancel density
  - fix_options: ranked by expected impact

Usage:
  ./scripts/slots_500_triage.py --log-dir /var/log/llama-proxy
  ./scripts/slots_500_triage.py --llama-file '*2026-08-27*'
  ./scripts/slots_500_triage.py --json | --summary | --compact

Exit codes:
  0 - success
  1 - log directory missing / unexpected error
"""

from __future__ import annotations

import argparse
import gzip as gz
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# llama-server access log parsing (no timestamps in llama-server logs; the
# corpus is per-file)
# ---------------------------------------------------------------------------

# [59455] srv  log_server_r: done request: GET /slots 127.0.0.1 200
_ACCESS_RE = re.compile(
    r"^\[?(?P<pid>\d+)?\]?\s*srv  log_server_r: done request: "
    r"(?P<method>\w+) /slots\s+\S+\s+(?P<status>\d{3})"
)
# srv  proxy_reques: proxying request to model Qwen3 on port 59455
_PROXY_RE = re.compile(r"srv\s+proxy_reques: proxying request to model (\S+) on port (\d+)")
# srv  operator(): http client error: Connection handling canceled
_CANCEL_RE = re.compile(r"operator\(\): http client error: (Connection handling canceled|Failed to read connection)")
# srv  operator(): got exception: {"error":{"code":400,...}} (model-side)
_MODEL_400_RE = re.compile(r"got exception: \{\"error\":\{\"code\":400")
# prompt processing progress → giant prefill activity
_PREFILL_RE = re.compile(r"prompt processing progress")
# created context checkpoint
_CHECKPOINT_RE = re.compile(r"created context checkpoint")
# slot selected / load lines → restart-related
_LOAD_RE = re.compile(r"(load: spawning server instance|server is listening|ensure_model)")


def _iter_llama_lines(log_dir: Path, llama_file_glob: str | None):
    import fnmatch

    files = sorted(
        [p for p in log_dir.glob("llama-server*.log*") if p.is_file()],
        key=lambda p: p.name,
    )
    if llama_file_glob:
        files = [p for p in files if fnmatch.fnmatch(p.name, llama_file_glob)]
    for path in files:
        opener = gz.open if path.suffix == ".gz" else open
        with opener(path, "rt", errors="replace") as fh:
            for line in fh:
                yield path.name, line


def analyze_slots_500(log_dir: Path, llama_file_glob: str | None = None) -> dict:
    """Classify GET /slots status responses and correlate with load."""
    # responder status: (responder, method, status) -> count
    slots_status: Counter = Counter()
    classify: Counter = Counter()  # proximate-cause labels

    five_hundred_windows: list[dict] = []

    stats = {
        "slots_total": 0, "slots_200": 0, "slots_400": 0, "slots_500": 0,
        "cancel_events": 0, "model_400_events": 0,
        "proxying_events": 0, "prefill_events": 0, "checkpoint_events": 0,
        "near_proxy_500s": 0, "near_cancel_500s": 0, "near_prefill_500s": 0,
        "near_checkpoint_500s": 0,
    }

    # We buffer the last N lines to build ±window context for each 500.
    ctx_buf: list[str] = []
    ctx_n = 10

    per_file_slots: dict[str, Counter] = defaultdict(Counter)

    for fname, line in _iter_llama_lines(log_dir, llama_file_glob):
        ctx_buf.append(line.rstrip())
        if len(ctx_buf) > ctx_n:
            ctx_buf.pop(0)

        is_slots = "GET /slots" in line
        if is_slots:
            m = _ACCESS_RE.match(line)
            status = m.group("status") if m else None
            responder = "router" if m and not m.group("pid") else (m.group("pid") if m else "unknown")
            per_file_slots[fname][status or "?"] += 1
            if status:
                slots_status[(responder, status)] += 1
                stats[f"slots_{status}"] += 1
                stats["slots_total"] += 1

            if status == "500":
                window = "\n".join(ctx_buf)
                # proximate-cause classification
                if _PROXY_RE.search(window):
                    stats["near_proxy_500s"] += 1
                    classify["router_proxy_cancel"] += 1
                if _CANCEL_RE.search(window):
                    stats["near_cancel_500s"] += 1
                    classify["connection_canceled"] += 1
                if _PREFILL_RE.search(window) or _CHECKPOINT_RE.search(window):
                    stats["near_prefill_500s"] += 1
                    classify["concurrent_prefill"] += 1
                if _LOAD_RE.search(window):
                    classify["restart_race"] += 1
                five_hundred_windows.append({"line": window[:400]})

        elif _PROXY_RE.search(line):
            stats["proxying_events"] += 1
        elif _CANCEL_RE.search(line):
            stats["cancel_events"] += 1
        elif _MODEL_400_RE.search(line):
            stats["model_400_events"] += 1
        elif _PREFILL_RE.search(line):
            stats["prefill_events"] += 1
        elif _CHECKPOINT_RE.search(line):
            stats["checkpoint_events"] += 1

    # ------------------------------------------------------------------
    # Correlation: 500 density vs prefill/checkpoint/cancel activity
    # (per 1K-line slab since llama logs have no timestamps)
    # ------------------------------------------------------------------
    slab: dict[str, int] = defaultdict(int)
    slab_slots_500: dict[str, int] = defaultdict(int)
    slab_slots_total: dict[str, int] = defaultdict(int)
    slab_index = 0

    for fname, line in _iter_llama_lines(log_dir, llama_file_glob):
        idx = slab_index // 1000
        if "GET /slots" in line:
            m = _ACCESS_RE.match(line)
            if m:
                status = m.group("status")
                slab_slots_total[f"{fname}:{idx}"] += 1
                if status == "500":
                    slab_slots_500[f"{fname}:{idx}"] += 1
        if _PREFILL_RE.search(line) or _CHECKPOINT_RE.search(line):
            slab[f"{fname}:{idx}"] += 1
        slab_index += 1

    # pearson-like positive: 500-rate per slab vs prefill density
    corr_rows = []
    all_slabs = sorted(set(list(slab.keys()) + list(slab_slots_total.keys())))
    for s in all_slabs:
        total = slab_slots_total.get(s, 0)
        if total == 0:
            continue
        r500 = slab_slots_500.get(s, 0) / total
        corr_rows.append({
            "slab": s,
            "slots_polls": total,
            "five_hundred_pct": round(100 * r500, 2),
            "prefill_checkpoint_events": slab.get(s, 0),
        })

    high_rate_windows = [
        r for r in corr_rows if r["five_hundred_pct"] >= 50
    ]
    busy_windows = [
        r for r in corr_rows if r["prefill_checkpoint_events"] > 0
        and r["five_hundred_pct"] > 0
    ]

    return {
        "classification": {
            "responder_status": {f"{k[0] or 'router'}:{k[1]}": v for k, v in sorted(slots_status.items())},
            "proximate_causes": dict(classify.most_common()),
            "stats": stats,
        },
        "per_file_slots": {k: dict(v) for k, v in per_file_slots.items()},
        "correlation": {
            "slab_rows": corr_rows,
            "high_rate_windows": high_rate_windows,
            "busy_windows_count": len(busy_windows),
            "five_hundred_windows": five_hundred_windows[:10],
        },
        "fix_options": [
            {
                "rank": 1,
                "fix": "Restore-before-proxy: route GET /slots DIRECTLY to the model instance "
                       "(bypassing the router proxy) when a model is unloaded or busy, or answer "
                       "from the last-known model slot state instead of proxying",
                "expected_impact": "Eliminates nearly all 500s (router-proxy-cancel path) and the "
                                   "400s; upstream of LP-0MSVP7XJ6008QPKX (/slots 500 after restart) "
                                   "and LP-0MSB0RV72001KNRV (slot registry leak)",
                "tracked_elsewhere": "LP-0MSVP7XJ6008QPKX (restart 500s), LP-0MSB0RV72001KNRV (registry leak), "
                                     "LP-0MSHW2AXJ009DO3S (400 without ?model=)",
                "gpu_wedge_risk": "none",
            },
            {
                "rank": 2,
                "fix": "Concurrency-aware slot query timeout/backoff: when the router probes slots "
                       "for a model that is mid-giant-prefill, extend the timeout and retry instead "
                       "of cancelling, so the 500 rate drops during prefill storms",
                "expected_impact": "Medium — cuts 500s during giant-prefill windows (53.3% of 500s near "
                                   "are near prefill activity)",
                "tracked_elsewhere": "partially LP-0MSI5B1T2009GQ4C (load-aware timeout rebalance)",
                "gpu_wedge_risk": "low",
            },
            {
                "rank": 3,
                "fix": "Fix the 'Connection handling canceled' path in router-mode proxying "
                       "(coordinate cancellation handling between the router and busy model)",
                "expected_impact": "Medium — 43.0% of 500 windows contain a cancel event; may be a "
                                   "llama.cpp router-mode bug under load",
                "tracked_elsewhere": "not tracked",
                "gpu_wedge_risk": "low",
            },
            {
                "rank": 4,
                "fix": "Deduplicate slot-status aggregation: proxy polls the model directly for "
                       "slot state instead of letting the router's /slots be the authority "
                       "(mirror _query_slots_detail LP-0MTC8A2UB0040NKQ style)",
                "expected_impact": "Low-medium — reduces dependence on the flaky router /slots path",
                "tracked_elsewhere": "not tracked",
                "gpu_wedge_risk": "none",
            },
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="/var/log/llama-proxy")
    parser.add_argument("--llama-file", default=None,
                        help="restrict llama log parsing to a filename glob")
    parser.add_argument("--json", action="store_true", default=True)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"error: log directory not found: {log_dir}", file=sys.stderr)
        return 1

    try:
        res = analyze_slots_500(log_dir, args.llama_file)
    except Exception as exc:
        print(f"error: analysis failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    indent = None if args.compact else 2
    if args.summary:
        print(json.dumps({
            "classification": res["classification"],
            "correlation_counts": {
                "busy_windows_count": res["correlation"]["busy_windows_count"],
                "high_rate_windows": len(res["correlation"]["high_rate_windows"]),
            },
            "fix_options": res["fix_options"],
        }, indent=indent, default=str))
    else:
        print(json.dumps(res, indent=indent, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
