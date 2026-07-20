#!/usr/bin/env python3
"""
Benchmark comparison tool for KV quantization experiments.

Compares baseline and candidate benchmark results, computes deltas, and
produces a Markdown report with gating policy checks.

Usage:
    python -m proxy.benchmarks.compare_results baseline.json candidate.json
    python -m proxy.benchmarks.compare_results baseline.json candidate.json --json
    python -m proxy.benchmarks.compare_results baseline.json candidate.json \
        --memory-threshold 20 --latency-threshold 15 --tps-threshold 15
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Default gating thresholds
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    "memory_reduction_pct": 25.0,
    "max_latency_regression_pct": 10.0,
    "max_tps_regression_pct": 10.0,
    "max_ttft_regression_pct": 10.0,
}


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------


def safe_pct(baseline: float | None, candidate: float | None) -> float | None:
    """Compute percentage change: ((candidate - baseline) / baseline) * 100.

    Returns None when baseline is None, zero, or both are None.
    """
    if baseline is None or candidate is None:
        return None
    if baseline == 0:
        return None
    return (candidate - baseline) / baseline * 100


def compute_deltas(baseline: dict, candidate: dict) -> dict:
    """Compute percentage deltas between baseline and candidate summaries.

    Returns a dict with delta fields.
    """
    bs = baseline.get("summary", {})
    cs = candidate.get("summary", {})

    deltas = {
        "duration_delta_pct": safe_pct(
            bs.get("avg_total_duration_seconds"),
            cs.get("avg_total_duration_seconds"),
        ),
        "tps_delta_pct": safe_pct(
            bs.get("avg_tokens_per_second"),
            cs.get("avg_tokens_per_second"),
        ),
        "ttft_delta_pct": safe_pct(
            bs.get("avg_time_to_first_token_seconds"),
            cs.get("avg_time_to_first_token_seconds"),
        ),
        "memory_delta_pct": safe_pct(
            bs.get("memory_snapshot_bytes"),
            cs.get("memory_snapshot_bytes"),
        ),
    }

    # Compute absolute completion token change
    b_tokens = bs.get("total_completion_tokens", 0)
    c_tokens = cs.get("total_completion_tokens", 0)
    deltas["completion_tokens_delta"] = c_tokens - b_tokens
    deltas["completion_tokens_delta_pct"] = safe_pct(
        float(b_tokens) if b_tokens else None,
        float(c_tokens) if c_tokens else None,
    )

    return deltas


# ---------------------------------------------------------------------------
# Gating check
# ---------------------------------------------------------------------------


def check_gates(
    deltas: dict, baseline: dict, candidate: dict, thresholds: dict | None = None
) -> dict:
    """Evaluate gating policy thresholds against computed deltas.

    Returns a dict with pass/fail results keyed by gate name.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    gates = {}

    # Memory gate: candidate must show at least threshold% reduction
    mem_delta = deltas.get("memory_delta_pct")
    if mem_delta is not None:
        # A negative delta = memory decreased (good)
        reduction = -mem_delta
        gates["memory_reduction_gate"] = {
            "pass": reduction >= thresholds["memory_reduction_pct"],
            "actual_reduction_pct": round(reduction, 1),
            "threshold_pct": thresholds["memory_reduction_pct"],
            "detail": (
                f"Memory reduction: {reduction:.1f}% "
                f"(threshold: >= {thresholds['memory_reduction_pct']}%)"
            ),
        }
    else:
        gates["memory_reduction_gate"] = {
            "pass": None,
            "actual_reduction_pct": None,
            "threshold_pct": thresholds["memory_reduction_pct"],
            "detail": "Memory data not available — gate skipped",
        }

    # Latency gate: candidate latency must not regress more than threshold%
    dur_delta = deltas.get("duration_delta_pct")
    if dur_delta is not None:
        # Positive delta = slower (regression)
        gates["latency_regression_gate"] = {
            "pass": dur_delta <= thresholds["max_latency_regression_pct"],
            "actual_regression_pct": round(dur_delta, 1),
            "threshold_pct": thresholds["max_latency_regression_pct"],
            "detail": (
                f"Latency delta: {dur_delta:+.1f}% "
                f"(threshold: <= +{thresholds['max_latency_regression_pct']}%)"
            ),
        }
    else:
        gates["latency_regression_gate"] = {
            "pass": None,
            "actual_regression_pct": None,
            "threshold_pct": thresholds["max_latency_regression_pct"],
            "detail": "Latency data not available — gate skipped",
        }

    # TPS gate: candidate TPS must not regress more than threshold%
    tps_delta = deltas.get("tps_delta_pct")
    if tps_delta is not None:
        # Negative delta = slower (regression)
        gates["tps_regression_gate"] = {
            "pass": tps_delta >= -thresholds["max_tps_regression_pct"],
            "actual_regression_pct": round(-tps_delta, 1) if tps_delta < 0 else 0.0,
            "threshold_pct": thresholds["max_tps_regression_pct"],
            "detail": (
                f"TPS delta: {tps_delta:+.1f}% "
                f"(threshold: >= -{thresholds['max_tps_regression_pct']}%)"
            ),
        }
    else:
        gates["tps_regression_gate"] = {
            "pass": None,
            "actual_regression_pct": None,
            "threshold_pct": thresholds["max_tps_regression_pct"],
            "detail": "TPS data not available — gate skipped",
        }

    # TTFT gate: candidate TTFT must not regress more than threshold%
    ttft_delta = deltas.get("ttft_delta_pct")
    if ttft_delta is not None:
        gates["ttft_regression_gate"] = {
            "pass": ttft_delta <= thresholds["max_ttft_regression_pct"],
            "actual_regression_pct": round(ttft_delta, 1),
            "threshold_pct": thresholds["max_ttft_regression_pct"],
            "detail": (
                f"TTFT delta: {ttft_delta:+.1f}% "
                f"(threshold: <= +{thresholds['max_ttft_regression_pct']}%)"
            ),
        }
    else:
        gates["ttft_regression_gate"] = {
            "pass": None,
            "actual_regression_pct": None,
            "threshold_pct": thresholds["max_ttft_regression_pct"],
            "detail": "TTFT data not available — gate skipped",
        }

    # Overall result
    results = [
        g["pass"]
        for g in gates.values()
        if g["pass"] is not None
    ]
    gates["overall_pass"] = all(results) if results else None

    return gates


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------


def generate_report(
    baseline: dict,
    candidate: dict,
    deltas: dict,
    gates: dict | None = None,
    thresholds: dict | None = None,
) -> str:
    """Generate a Markdown comparison report."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    if gates is None:
        gates = check_gates(deltas, baseline, candidate, thresholds)

    bc = baseline.get("config", {})
    cc = candidate.get("config", {})
    bs = baseline.get("summary", {})
    cs = candidate.get("summary", {})

    lines = []
    lines.append("# Benchmark Comparison Report")
    lines.append("")
    lines.append(
        f"Generated: {candidate.get('timestamp', 'N/A')}"
    )
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Property | Baseline | Candidate |")
    lines.append("|----------|----------|-----------|")
    lines.append(f"| Run type | {bc.get('run_type', 'N/A')} | {cc.get('run_type', 'N/A')} |")
    lines.append(f"| Model | {bc.get('model', 'N/A')} | {cc.get('model', 'N/A')} |")
    lines.append(f"| Quantization | {bc.get('quantization', 'N/A') or 'N/A'} | {cc.get('quantization', 'N/A') or 'N/A'} |")
    lines.append(f"| Context size | {bc.get('ctx_size', 'N/A')} | {cc.get('ctx_size', 'N/A')} |")
    lines.append(f"| Requests | {bs.get('total_requests', 'N/A')} | {cs.get('total_requests', 'N/A')} |")
    lines.append("")

    # Summary comparison
    lines.append("## Summary Comparison")
    lines.append("")
    lines.append("| Metric | Baseline | Candidate | Delta (%) |")
    lines.append("|--------|----------|-----------|-----------|")

    def add_row(metric: str, b_val: Any, c_val: Any, delta_key: str):
        b_str = f"{b_val:.2f}" if isinstance(b_val, (int, float)) else str(b_val) if b_val else "N/A"
        c_str = f"{c_val:.2f}" if isinstance(c_val, (int, float)) else str(c_val) if c_val else "N/A"
        d_val = deltas.get(delta_key)
        d_str = f"{d_val:+.2f}%" if d_val is not None else "N/A"
        lines.append(f"| {metric} | {b_str} | {c_str} | {d_str} |")

    add_row("Avg duration (s)", bs.get("avg_total_duration_seconds"), cs.get("avg_total_duration_seconds"), "duration_delta_pct")
    add_row("Avg TPS", bs.get("avg_tokens_per_second"), cs.get("avg_tokens_per_second"), "tps_delta_pct")
    add_row("Avg TTFT (s)", bs.get("avg_time_to_first_token_seconds"), cs.get("avg_time_to_first_token_seconds"), "ttft_delta_pct")
    add_row("Total completion tokens", bs.get("total_completion_tokens"), cs.get("total_completion_tokens"), "completion_tokens_delta_pct")

    # Memory row (display in MB if available)
    b_mem = bs.get("memory_snapshot_bytes")
    c_mem = cs.get("memory_snapshot_bytes")
    if b_mem and c_mem:
        b_mem_mb = b_mem / (1024 * 1024)
        c_mem_mb = c_mem / (1024 * 1024)
        add_row("Memory RSS (MB)", round(b_mem_mb, 1), round(c_mem_mb, 1), "memory_delta_pct")
    else:
        lines.append(f"| Memory RSS | {'N/A'} | {'N/A'} | N/A |")

    lines.append("")
    lines.append(f"*Errors: baseline={bs.get('errors', 0)}, candidate={cs.get('errors', 0)}*")
    lines.append("")

    # Gating policy results
    lines.append("## Gating Policy Results")
    lines.append("")
    lines.append("| Gate | Result | Detail |")
    lines.append("|------|--------|--------|")

    for gate_name, gate_result in gates.items():
        if gate_name == "overall_pass":
            continue
        result_str = (
            "✅ PASS" if gate_result["pass"] is True
            else "❌ FAIL" if gate_result["pass"] is False
            else "⬜ SKIP"
        )
        lines.append(f"| {gate_name} | {result_str} | {gate_result['detail']} |")

    lines.append("")

    overall = gates.get("overall_pass")
    if overall is True:
        lines.append("### ✅ Overall: ALL GATES PASSED")
        lines.append("")
        lines.append("The candidate configuration meets all gating thresholds and is safe to roll out.")
    elif overall is False:
        lines.append("### ❌ Overall: GATES FAILED")
        lines.append("")
        lines.append(
            "One or more gating thresholds were not met. "
            "Review the failures above and adjust the candidate configuration "
            "before rolling out."
        )
    else:
        lines.append("### ⬜ Overall: INCONCLUSIVE")
        lines.append("")
        lines.append(
            "Some gates could not be evaluated due to missing data. "
            "Review the skipped gates above and re-run with complete metrics."
        )

    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def generate_json_output(deltas: dict, gates: dict) -> dict:
    """Generate a JSON-serializable comparison result."""
    return {
        "deltas": deltas,
        "gates": gates,
        "overall_pass": gates.get("overall_pass"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare baseline and candidate benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("baseline", type=str, help="Path to baseline benchmark JSON")
    parser.add_argument("candidate", type=str, help="Path to candidate benchmark JSON")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output comparison result as JSON instead of Markdown report",
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=DEFAULT_THRESHOLDS["memory_reduction_pct"],
        help=f"Minimum memory reduction %% (default: {DEFAULT_THRESHOLDS['memory_reduction_pct']})",
    )
    parser.add_argument(
        "--latency-threshold",
        type=float,
        default=DEFAULT_THRESHOLDS["max_latency_regression_pct"],
        help=f"Max latency regression %% (default: {DEFAULT_THRESHOLDS['max_latency_regression_pct']})",
    )
    parser.add_argument(
        "--tps-threshold",
        type=float,
        default=DEFAULT_THRESHOLDS["max_tps_regression_pct"],
        help=f"Max TPS regression %% (default: {DEFAULT_THRESHOLDS['max_tps_regression_pct']})",
    )
    parser.add_argument(
        "--ttft-threshold",
        type=float,
        default=DEFAULT_THRESHOLDS["max_ttft_regression_pct"],
        help=f"Max TTFT regression %% (default: {DEFAULT_THRESHOLDS['max_ttft_regression_pct']})",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the comparison tool."""
    args = parse_args(argv)

    # Load input files
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)

    if not baseline_path.exists():
        print(f"Error: baseline file not found: {baseline_path}", file=sys.stderr)
        sys.exit(1)
    if not candidate_path.exists():
        print(f"Error: candidate file not found: {candidate_path}", file=sys.stderr)
        sys.exit(1)

    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(candidate_path) as f:
        candidate = json.load(f)

    # Custom thresholds
    thresholds = {
        "memory_reduction_pct": args.memory_threshold,
        "max_latency_regression_pct": args.latency_threshold,
        "max_tps_regression_pct": args.tps_threshold,
        "max_ttft_regression_pct": args.ttft_threshold,
    }

    # Compute deltas
    deltas = compute_deltas(baseline, candidate)

    # Check gating
    gates = check_gates(deltas, baseline, candidate, thresholds)

    if args.json:
        output = generate_json_output(deltas, gates)
        print(json.dumps(output, indent=2))
    else:
        report = generate_report(baseline, candidate, deltas, gates, thresholds)
        print(report)

    # Exit code: 0 if all gates pass, 1 if any fail, 2 if inconclusive
    overall = gates.get("overall_pass")
    if overall is True:
        sys.exit(0)
    elif overall is False:
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
