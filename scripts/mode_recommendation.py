#!/usr/bin/env python3
"""Mode-specific recommendation & follow-up brief (F5, LP-0MTCMGJBM007AQ55).

Consolidates the F4 fix ranking into per-mode fix sets (fast / cheap) with
concrete persistence-cap and GPU-wedge timeout/cooldown values, estimates
fallback-rate / TTFT-P95 impact, and emits the follow-up implementation
brief for the parent's AC #5 (no code change now — evaluation only).

GPU-wedge contract (LP-0MS91DHPZ001VWQO): any persistence-cap raise MUST
carry an explicit timeout/cooldown plan — the current config values are
preserved as the floor (max timeout 60s, cooldown 300s, max consecutive
failures 3, skip-when-busy on) so saves cannot wedge the GPU.

Usage:
  ./scripts/mode_recommendation.py --json | --markdown | --compact

Exit codes:
  0 - success
  1 - unexpected error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import restore_metric_fix_ranking as metric  # noqa: E402

# Current config values (proxy/config-fast.yaml / proxy/config-cheap.yaml),
# preserved as the GPU-wedge mitigation floor.
GPU_WEDGE_PLAN = {
    "timeout_base_seconds": 3.0,
    "timeout_per_token_seconds": 0.0015,
    "max_timeout_seconds": 60,
    "failure_cooldown_seconds": 300,
    "max_consecutive_failures": 3,
    "skip_when_busy": True,
}


def build_recommendation() -> dict:
    """Per-mode recommendation from the F4 fix ranking + config.

    Fix selection per mode:
      - fast: prioritize TTFT/P95 → keep the highest-savings options;
        cap raise is the head of the ranking.
      - cheap: prioritize cost/local-utilization → same head, with the
        cap raise enabling local reuse of oversized sessions (fallback
        avoidance is the point).
    Both modes select the F4 ranking head (cap raise + affinity fix) as the
    primary set; restore-before-save is included as the low-risk ordering
    improvement. Relaxing skip-when-busy is NOT selected (only 2 save
    failures observed; the gate's conservatism is acceptable and keeps the
    GPU-wedge floor intact).
    """
    # Load F4 ranking from the baseline corpus for realistic savings numbers.
    baseline_path = SCRIPT_DIR.parent / "docs/dev/slot-persistence-baseline-2026-08-26.json"
    current_rate = 0.0488  # fallback if baseline missing
    pool = 0
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        metrics = baseline.get("baseline_metrics", baseline)
        cur = metric.compute_current_rates(metrics)
        current_rate = cur["llama_native"]["rate_pct"] / 100.0
        pool = cur["prefill_done_tokens"]

    fixes = metric.rank_fixes(metric.DEFAULT_FIX_OPTIONS, pool, current_rate)

    def _mode_set(mode: str, slots: int, cap: int, priority: str,
                  fallback_dir: str, ttft_dir: str) -> dict:
        # select: head (rank 1) + affinity (rank 2) + ordering (rank 3)
        selected = [f for f in fixes if f["rank"] in (1, 2, 3)]
        total_savings = sum(f["expected_savings_tokens"] for f in selected)
        return {
            "slots": slots,
            "persistence_cap_tokens": cap,
            "priority": priority,
            "selected_fixes": selected,
            "expected_savings_tokens_total": total_savings,
            "gpu_wedge_plan": dict(GPU_WEDGE_PLAN),
            "expected_impact": {
                "fallback_rate": {
                    "direction": fallback_dir,
                    "rationale": (
                        f"{mode} mode: restore reuse keeps oversized sessions "
                        "local without full re-prefill, so fewer turns fall "
                        "back to remote providers."
                    ),
                },
                "ttft_p95": {
                    "direction": ttft_dir,
                    "rationale": (
                        f"{mode} mode: eliminating minutes-long full "
                        "re-prefills on large sessions directly lowers "
                        "TTFT/P95."
                    ),
                },
            },
        }

    fast = _mode_set("fast", slots=3, cap=83285, priority="ttft_p95",
                     fallback_dir="down", ttft_dir="down")
    cheap = _mode_set("cheap", slots=2, cap=126976, priority="cost_local_utilization",
                      fallback_dir="down", ttft_dir="down")

    notes = {"converged": fast["persistence_cap_tokens"] == cheap["persistence_cap_tokens"]}

    return {
        "fast": fast,
        "cheap": cheap,
        "follow_up_brief": {
            "title": "Implement KV slot persistence fixes from the reuse-gap evaluation",
            "cap_fast": fast["persistence_cap_tokens"],
            "cap_cheap": cheap["persistence_cap_tokens"],
            "gpu_wedge_plan": GPU_WEDGE_PLAN,
            "evaluation_parent": "LP-0MTAQNB7J0094X71",
            "no_code_change_now": True,
        },
        "notes": notes,
    }


def render_markdown(res: dict) -> str:
    lines = [
        "# Mode-specific recommendation (F5)",
        "",
    ]
    for mode, label in (("fast", "Fast"), ("cheap", "Cheap")):
        m = res[mode]
        lines.append(f"## {label} mode ({m['slots']} slots, cap {m['persistence_cap_tokens']:,})")
        lines.append("")
        lines.append(f"- Priority: `{m['priority']}`")
        lines.append(f"- Selected fixes: {', '.join(f['name'] for f in m['selected_fixes'])}")
        lines.append(f"- Expected savings: {m['expected_savings_tokens_total']:,} tokens/day")
        lines.append(f"- Fallback rate: {m['expected_impact']['fallback_rate']['direction']} — "
                     f"{m['expected_impact']['fallback_rate']['rationale']}")
        lines.append(f"- TTFT/P95: {m['expected_impact']['ttft_p95']['direction']} — "
                     f"{m['expected_impact']['ttft_p95']['rationale']}")
        lines.append("")
        lines.append("### GPU-wedge plan (LP-0MS91DHPZ001VWQO)")
        lines.append("")
        plan = m["gpu_wedge_plan"]
        lines.append(
            f"- timeout base {plan['timeout_base_seconds']}s + "
            f"{plan['timeout_per_token_seconds']}s/token, cap {plan['max_timeout_seconds']}s; "
            f"cooldown {plan['failure_cooldown_seconds']}s after "
            f"{plan['max_consecutive_failures']} consecutive failures; "
            f"skip-when-busy {'on' if plan['skip_when_busy'] else 'off'}."
        )
        lines.append("")

    lines.append("## Follow-up implementation brief")
    lines.append("")
    b = res["follow_up_brief"]
    lines.append(f"- **{b['title']}**")
    lines.append(f"- caps: fast {b['cap_fast']:,} / cheap {b['cap_cheap']:,} "
                 "(evaluation only — no code change now)")
    lines.append(f"- references {b['evaluation_parent']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", default=True)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)

    try:
        res = build_recommendation()
    except Exception as exc:
        print(f"error: recommendation failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    if args.markdown:
        print(render_markdown(res))
    else:
        print(json.dumps(res, indent=None if args.compact else 2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
