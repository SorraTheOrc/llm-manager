#!/usr/bin/env python3
"""Restore-rate metric, per-mode targets & fix ranking (F4, LP-0MTCMG18A008ZKVT).

Quantifies the current KV slot restore rate with a Wilson confidence
interval, defines per-mode target restore rates (fast vs cheap), and ranks
fix options by expected prefill-token savings and GPU-wedge risk — feeding
the F5 mode-specific recommendation and the final F6 evaluation report.

Metrics:
  - proxy_slot_restore_rate:   proxy-side slot_save→slot_restore success
                               (the persistence mechanism designed for reuse)
  - llama_native_restore_rate: llama-server 'created context checkpoint' →
                               'restored context checkpoint' (the incident's
                               ~95%-unrestored number)
  - prefill tokens/day re-prefilled because restores fail → the savings pool

Fix ranking:
  Each option carries recovered_fraction × extra_restores × avg_prefill_tokens
  → expected_savings_tokens, and a GPU-wedge risk rating derived from
  LP-0MS91DHPZ001VWQO (large-context saves can wedge the GPU; any cap raise
  must carry a timeout/cooldown plan).

Usage:
  ./scripts/restore_metric_fix_ranking.py --baseline docs/dev/slot-persistence-baseline-2026-08-26.json
  ./scripts/restore_metric_fix_ranking.py --json | --markdown

Exit codes:
  0 - success
  1 - baseline file missing/invalid / unexpected error
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a proportion; returns (center, lo, hi).

    Robust for small samples and degenerate rates (0% / 100%) where the
    normal approximation collapses.
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center, max(0.0, center - half), min(1.0, center + half)


def compute_current_rates(baseline_metrics: dict) -> dict:
    """Current restore rates for both mechanisms + prefill pool."""
    k_p = int(baseline_metrics.get("slot_restore_success", 0))
    n_p = int(baseline_metrics.get("total_slot_saves", 0))
    c_p, lo_p, hi_p = wilson_ci(k_p, n_p)

    k_n = int(baseline_metrics.get("llama_checkpoints_restored", 0))
    n_n = int(baseline_metrics.get("llama_checkpoints_created", 0))
    c_n, lo_n, hi_n = wilson_ci(k_n, n_n)

    prefill_done = int(baseline_metrics.get("prompt_done_tokens_total", 0))

    return {
        "proxy": {
            "k": k_p, "n": n_p,
            "rate_pct": round(100 * c_p, 2),
            "ci_pct": {"lo": round(100 * lo_p, 2), "hi": round(100 * hi_p, 2)},
        },
        "llama_native": {
            "k": k_n, "n": n_n,
            "rate_pct": round(100 * c_n, 2),
            "ci_pct": {"lo": round(100 * lo_n, 2), "hi": round(100 * hi_n, 2)},
        },
        "prefill_done_tokens": prefill_done,
        "prefill_reprefill_daily_tokens": round(
            prefill_done * (1 - c_n)),  # tokens re-prefilled at current rate
    }


def define_targets() -> dict:
    """Per-mode target restore rates with rationale.

    Targets are on the PROXY slot-persistence rate (the mechanism the fixes
    can move); the native-checkpoint rate is the incident's signal but not a
    directly actionable knob.
    """
    return {
        "fast": {
            "priority": "ttft_p95",
            "rationale": (
                "Fast mode tolerates fallbacks for speed. Restore reuse cuts "
                "TTFT/P95 on large sessions by eliminating full re-prefill; "
                "target is a high reliable-reuse rate so large-context turns "
                "do not pay minutes of prefill."
            ),
            "target_rate_pct": 95.0,
            "persistence_cap_tokens": 83285,
            "slots": 3,
        },
        "cheap": {
            "priority": "cost_local_utilization",
            "rationale": (
                "Cheap mode must avoid fallbacks (cost). Reliable restore is "
                "the lever that keeps oversized sessions local without full "
                "re-prefill; target favors local utilization over TTFT."
            ),
            "target_rate_pct": 90.0,
            "persistence_cap_tokens": 126976,
            "slots": 2,
        },
    }


def rank_fixes(fixes: dict, pool_tokens: int, current_rate: float) -> list[dict]:
    """Rank fix options by expected prefill-token savings (desc), ties
    broken by lower GPU-wedge risk.

    Savings model: the daily re-prefill pool (prompt_done_tokens_total) is
    what is spent because restores fail at the current rate. Each fix
    recovers a fraction of the addressable gap:

        savings = pool_tokens × (1 - current_rate) × rate_gain × recovery

    where ``rate_gain`` is the fraction of the restore-rate gap that fix
    closes and ``recovery`` is the share of those turns that actually
    restore. Both are derived from the F1/F2/F3 evidence.
    """
    scored = []
    risk_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    gap = 1.0 - current_rate  # fraction of prefill spent due to restore failure
    for key, f in fixes.items():
        savings = pool_tokens * gap * f["rate_gain"] * f["recovered_fraction"]
        scored.append({
            "id": key,
            "name": f["name"],
            "expected_savings_tokens": int(savings),
            "gpu_wedge_risk": f["gpu_wedge_risk"],
            "rate_gain": f["rate_gain"],
            "recovered_fraction": f["recovered_fraction"],
        })
    scored.sort(key=lambda s: (-s["expected_savings_tokens"],
                               risk_order.get(s["gpu_wedge_risk"], 9)))
    for i, s in enumerate(scored, 1):
        s["rank"] = i
    return scored


# The four mandated fix-option classes, parameterised from the F1/F2/F3
# corpus data (2026-08-26) and the GPU-wedge constraint (LP-0MS91DHPZ001VWQO).
#
# rate_gain: fraction of the restore-rate gap each fix closes (F2 dominance:
# size-gating is the dominant factor; affinity/lease churn secondary; /slots
# 500s & staleness compound). recovered_fraction: share of those turns that
# actually restore (mitigates the GPU-wedge constraint — saves can time out
# under load, LP-0MSI1RWLM007N367).
DEFAULT_FIX_OPTIONS = {
    "raise_cap": {
        "name": "Raise persistence cap to routing clamp with timeout+cooldown plan",
        "description": (
            "Current session_slot_max_prompt_tokens (fast 83,285 / cheap "
            "126,976) skips persistence for oversized sessions — exactly the "
            "sessions that need reuse (F2: 38/48 sessions gated, 0 saves). "
            "Raise to the per-slot routing clamp and add the GPU-wedge "
            "timeout/cooldown from LP-0MS91DHPZ001VWQO."
        ),
        "rate_gain": 0.65,   # dominant factor per F2 (38/48 sessions gated)
        "recovered_fraction": 0.8,
        "gpu_wedge_risk": "medium",
    },
    "affinity_fix": {
        "name": "Session↔slot affinity / ownership continuity fix",
        "description": (
            "Lease churn (166 orphan + 141 evicted releases on 2026-08-26) "
            "breaks session↔slot mapping so a session cannot restore to its "
            "slot next turn (F2). Fix orphan cleanup / registry continuity "
            "(LP-0MSB0RV72001KNRV leak class)."
        ),
        "rate_gain": 0.15,
        "recovered_fraction": 0.5,
        "gpu_wedge_risk": "low",
    },
    "restore_before_save": {
        "name": "Restore-before-save ordering for hot sessions",
        "description": (
            "When a session reuses the same slot across turns, restore "
            "before saving so the warm KV cache is reused; ordering fix, "
            "no GPU footprint change."
        ),
        "rate_gain": 0.05,
        "recovered_fraction": 0.2,
        "gpu_wedge_risk": "none",
    },
    "relax_skip_when_busy": {
        "name": "Relax load-aware skip-when-busy for same-session reuse",
        "description": (
            "session_slot_skip_when_busy (LP-0MSI1RWLM007N367) skips "
            "persistence when another session streams. Relax to allow "
            "same-slot saves during moderate load with the adaptive "
            "timeout floor (only 2 save failures on the incident day — "
            "the gate is conservative vs observed failures)."
        ),
        "rate_gain": 0.03,
        "recovered_fraction": 0.1,
        "gpu_wedge_risk": "low",
    },
}


def render_markdown(res: dict) -> str:
    p = res["current_rates"]["proxy"]
    n = res["current_rates"]["llama_native"]
    lines = [
        "# Restore-rate metric, per-mode targets & fix ranking (F4)",
        "",
        "## Current restore rate (2026-08-26, Wilson 95% CI)",
        "",
        f"- **Proxy slot persistence**: {p['rate_pct']}% "
        f"({p['k']}/{p['n']}; CI {p['ci_pct']['lo']}–{p['ci_pct']['hi']}%)",
        f"- **llama-server native checkpoints**: {n['rate_pct']}% "
        f"({n['k']}/{n['n']}; CI {n['ci_pct']['lo']}–{n['ci_pct']['hi']}%) "
        f"— the incident's ~5% number",
        f"- Prefill tokens re-prefilled daily at current rate: "
        f"{res['current_rates']['prefill_reprefill_daily_tokens']:,} "
        f"(pool for fix-savings math)",
        "",
        "## Per-mode targets",
        "",
    ]
    for mode, t in res["targets"].items():
        lines.append(
            f"- **{mode}** ({t['priority']}): target {t['target_rate_pct']}% — "
            f"{t['rationale']} (cap {t['persistence_cap_tokens']}, {t['slots']} slots)"
        )
    lines.append("")
    lines.append("## Fix options ranked by expected prefill-token savings")
    lines.append("")
    lines.append("| Rank | Fix | Expected savings (tokens) | GPU-wedge risk |")
    lines.append("|------|-----|---------------------------|----------------|")
    for f in res["ranking"]:
        lines.append(
            f"| {f['rank']} | {f['name']} | {f['expected_savings_tokens']:,} "
            f"| {f['gpu_wedge_risk']} |"
        )
    lines.append("")
    lines.append("See docs/dev/save-restore-reuse-gap-root-cause.md (F2) and "
                 "docs/dev/slots-500-triage.md (F3) for the log-based evidence "
                 "behind each option. Any cap raise carries the timeout/cooldown "
                 "plan from LP-0MS91DHPZ001VWQO.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", default=None,
        help="F1 baseline summary JSON (default: docs/dev/slot-persistence-baseline-2026-08-26.json)",
    )
    parser.add_argument("--json", action="store_true", default=True)
    parser.add_argument("--markdown", action="store_true",
                        help="emit Markdown report instead of JSON")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline) if args.baseline else (
        SCRIPT_DIR.parent / "docs/dev/slot-persistence-baseline-2026-08-26.json")
    if not baseline_path.exists():
        print(f"error: baseline file not found: {baseline_path}", file=sys.stderr)
        return 1
    try:
        baseline = json.loads(baseline_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"error: invalid baseline JSON: {exc}", file=sys.stderr)
        return 1

    metrics = baseline.get("baseline_metrics", baseline)
    current = compute_current_rates(metrics)
    targets = define_targets()
    pool = current["prefill_done_tokens"]
    current_rate = current["llama_native"]["rate_pct"] / 100.0
    ranking = rank_fixes(DEFAULT_FIX_OPTIONS, pool, current_rate)
    res = {
        "current_rates": current,
        "targets": targets,
        "ranking": ranking,
    }

    if args.markdown:
        print(render_markdown(res))
    else:
        print(json.dumps(res, indent=None if args.compact else 2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
