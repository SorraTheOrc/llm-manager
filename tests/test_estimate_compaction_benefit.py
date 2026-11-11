#!/usr/bin/env python3
"""Tests for the compaction-benefit estimator (LP-0MWCEGXEK000U6G6).

The estimator costs the proxy-side session-compaction lever against the
2026-11-01 measured basis. These tests pin the arithmetic the spike's option
table quotes, so a later edit cannot quietly change what the lever is worth:

- the guard's own prefill model (cold/warm, linearity, input validation);
- the admission boundaries that decide admissibility: the flat cold-prefill
  cap (25 000 new tokens, refused on `>` only), the busy-gate queue bound
  (projected wait = 2 x own eta against 30 s), and the free-gate safety budget
  (171 s);
- the routing size tests (38 000 always-local band, 120 000 economic warm cap,
  residency ladder above the band);
- the headline claim: a prompt cap converts cold refused traffic **only** at or
  below the 25 000-token budget — the shipped 30k/38k targets and the intake's
  57k cap admit nothing;
- the two ceilings that bound the answer: the measured serialised-capacity
  ceiling (never exceeded) and the window's idle gate seconds;
- the window arithmetic itself (baseline shares reproduce, reconstruction is
  within 3 % of offered tokens);
- the committed register re-verifying from its own embedded inputs, and a
  tampered register being reported as drift.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from estimate_compaction_benefit import (  # noqa: E402
    CAPS,
    DEFAULT_BASIS,
    GATE_MIX,
    GUARD,
    ROUTING,
    WARM_REUSE,
    admissible,
    admission_probability,
    all_ok,
    build_register,
    checks,
    cost_cap,
    decile_grid,
    guard_verdict,
    load_basis,
    prefill_seconds,
    quantile,
    recheck_register,
    reconstruct_sizes,
    routing_verdict,
)

REGISTER = REPO_ROOT / "docs" / "dev" / "compaction-benefit-estimate-2026-11-01.json"


# ---------------------------------------------------------------------------
# Synthetic inputs: a window whose refused traffic is entirely cold, sized in
# the ranges the 2026-11-01 window actually contains.
# ---------------------------------------------------------------------------
def _classes(grid: list[float], requests: int = 100) -> list[dict]:
    return [
        {
            "reason": "context_too_large",
            "sessions": 1,
            "requests": requests,
            "requests_with_size_evidence": requests,
            "mean_tokens": sum(grid) / len(grid),
            "size_grid": grid,
        }
    ]


COLD_LARGE = _classes([90_000, 95_000, 100_000, 105_000, 110_000,
                       115_000, 120_000, 130_000, 150_000, 190_000])


# ---------------------------------------------------------------------------
# The guard's prefill model
# ---------------------------------------------------------------------------
def test_cold_prefill_uses_the_measured_per_token_coefficient() -> None:
    # 25 000 new tokens at 3.9 ms/token.
    assert prefill_seconds(25_000, 1.0) == pytest.approx(97.5)
    # 60 715 tokens (the window's median) priced cold.
    assert prefill_seconds(60_714.6, 1.0) == pytest.approx(236.79, abs=0.01)


def test_warm_prefill_prices_cached_tokens_cheaply() -> None:
    # 10 % new at 3.9 ms + 90 % cached at 0.5 ms = 0.84 ms/token.
    assert prefill_seconds(25_000, WARM_REUSE) == pytest.approx(21.0)
    assert prefill_seconds(25_000, 0.0) == pytest.approx(12.5)


def test_prefill_is_linear_in_prompt_size_at_a_fixed_cache_state() -> None:
    one = prefill_seconds(10_000, 1.0)
    two = prefill_seconds(20_000, 1.0)
    assert two == pytest.approx(2.0 * one)


@pytest.mark.parametrize("tokens,reuse", [(-1, 1.0), (1_000, -0.1), (1_000, 1.1)])
def test_prefill_rejects_impossible_inputs(tokens: float, reuse: float) -> None:
    with pytest.raises(ValueError):
        prefill_seconds(tokens, reuse)


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------
def test_quantile_interpolates_like_the_proxy() -> None:
    assert quantile([10, 20, 30, 40], 50) == pytest.approx(25.0)
    assert quantile([10], 90) == 10.0
    assert quantile([0, 100], 10) == pytest.approx(10.0)


def test_quantile_of_an_empty_sample_is_loud() -> None:
    with pytest.raises(ValueError):
        quantile([], 50)


def test_decile_grid_preserves_shape_and_mean() -> None:
    sample = [float(v) for v in range(1, 101)]
    grid = decile_grid(sample)
    assert len(grid) == 10
    assert grid == sorted(grid)
    assert sum(grid) / len(grid) == pytest.approx(sum(sample) / len(sample), rel=0.02)


# ---------------------------------------------------------------------------
# Admission boundaries
# ---------------------------------------------------------------------------
def test_flat_cold_prefill_cap_admits_exactly_the_budget() -> None:
    cap = GUARD["flat_cap_tokens"]
    assert guard_verdict(cap, 1.0, "free") == "within_budget"
    assert guard_verdict(cap + 1, 1.0, "free") == "prefill_budget_exceeded"


def test_busy_gate_projects_two_prefills_against_the_queue_budget() -> None:
    ceiling = GUARD["queue_budget_seconds"] / 2.0 / (GUARD["ms_per_new_token"] / 1000.0)
    assert guard_verdict(ceiling, 1.0, "busy") == "within_budget"
    assert guard_verdict(ceiling + 1, 1.0, "busy") == "eta_exceeds_budget"
    # The same prompt is admitted when the gate is free.
    assert guard_verdict(ceiling + 1, 1.0, "free") == "within_budget"


def test_free_gate_eta_bound_fires_behind_a_cache_warm_prompt() -> None:
    # With reuse 0.10 a prompt prices at 0.84 ms/token, so it reaches the 171 s
    # safety budget at ~203.6k tokens while still inside the flat cap. (A cold
    # prompt never gets here: the 25 000-token flat cap fires first.)
    ms_per_token = 0.1 * GUARD["ms_per_new_token"] + 0.9 * GUARD["ms_per_coded_token"]
    bound_tokens = GUARD["safety_budget_seconds"] / (ms_per_token / 1000.0)
    assert guard_verdict(bound_tokens, WARM_REUSE, "free") == "within_budget"
    assert (
        guard_verdict(bound_tokens + 1, WARM_REUSE, "free")
        == "prefill_eta_exceeds_safety_budget"
    )
    # A COLD prompt of the same size is refused much earlier, by the flat cap.
    assert guard_verdict(bound_tokens, 1.0, "free") == "prefill_budget_exceeded"


# ---------------------------------------------------------------------------
# Routing size tests
# ---------------------------------------------------------------------------
def test_routing_size_tests() -> None:
    assert routing_verdict(ROUTING["always_local_tokens"], 1.0) == "admit"
    assert (
        routing_verdict(ROUTING["always_local_tokens"] + 1, 1.0)
        == "residency_required"
    )
    assert routing_verdict(ROUTING["warm_cap_tokens"], 1.0) == "residency_required"
    assert routing_verdict(ROUTING["warm_cap_tokens"] + 1, 1.0) == "context_too_large"
    # A cache-warm prompt above the band is priced against its new tokens and
    # passes the ladder.
    assert routing_verdict(ROUTING["always_local_tokens"] + 1, WARM_REUSE) == "admit"


def test_admission_probability_weights_the_measured_gate_mix() -> None:
    # Cold 25 000 tokens is admitted only when the gate is free.
    assert admission_probability(25_000, 1.0, GATE_MIX) == pytest.approx(
        GATE_MIX["free"]
    )
    # Cold 30 000 exceeds the flat cap in either gate state.
    assert admission_probability(30_000, 1.0, GATE_MIX) == 0.0


# ---------------------------------------------------------------------------
# The headline: only a cap <= the cold-prefill budget converts cold traffic
# ---------------------------------------------------------------------------
def test_only_a_cap_at_or_below_the_cold_budget_converts_cold_prompts() -> None:
    def newly_admitted(cap: int | None) -> float:
        return cost_cap(COLD_LARGE, cap, 1.0, GATE_MIX, DEFAULT_BASIS)[
            "newly_admitted_streams"
        ]

    assert newly_admitted(None) == 0.0
    assert newly_admitted(57_000) == 0.0
    assert newly_admitted(38_000) == 0.0
    assert newly_admitted(30_000) == 0.0
    assert newly_admitted(25_000) > 0.0


def test_capping_is_monotone_in_served_tokens() -> None:
    entries = [
        cost_cap(COLD_LARGE, cap, 1.0, GATE_MIX, DEFAULT_BASIS)
        for cap, _ in CAPS
    ]
    served = [entry["local_prompt_tokens"] for entry in entries]
    assert served == sorted(served)


def test_served_volume_is_bounded_by_idle_gate_seconds() -> None:
    entry = cost_cap(COLD_LARGE, 25_000, 1.0, GATE_MIX, DEFAULT_BASIS)
    idle = DEFAULT_BASIS["window_seconds"] - DEFAULT_BASIS["local_busy_seconds"]
    bound = idle * entry["engine_rate_tok_s"]
    assert entry["added_tokens"] <= bound + 1e-6
    # The demand is far larger than the idle gate can absorb, so the gate (not
    # the demand) decides the served volume.
    assert entry["demand_gate_seconds"] > idle
    assert entry["served_streams"] < entry["newly_admitted_streams"]


def test_share_never_exceeds_the_measured_capacity_ceiling() -> None:
    basis = dict(DEFAULT_BASIS)
    basis["local_prompt_tokens"] = basis["capacity_tokens"] - 1.0
    entry = cost_cap(COLD_LARGE, 25_000, 1.0, GATE_MIX, basis)
    # Served volume is clamped at the capacity ceiling before the share is
    # derived, so rounding can never push the quoted share past it.
    assert entry["local_prompt_tokens"] <= basis["capacity_tokens"] + 1e-6
    ceiling = 100.0 * basis["capacity_tokens"] / basis["offered_prompt_tokens"]
    assert entry["local_token_share_pct"] <= ceiling + 5e-4


def test_cap_arithmetic_without_any_refused_traffic_is_a_no_op() -> None:
    entry = cost_cap(_classes([1_000.0]), 25_000, 1.0, GATE_MIX, DEFAULT_BASIS)
    assert entry["newly_admitted_streams"] == 0.0
    assert entry["added_tokens"] == 0.0


# ---------------------------------------------------------------------------
# Register assembly
# ---------------------------------------------------------------------------
def _sessions() -> list[dict]:
    return [
        {"session_id": "a", "messages": "10", "avg_context_size": "90000",
         "fallback_reason": "large_context_bypass"},
        {"session_id": "b", "messages": "5", "avg_context_size": "200000",
         "fallback_reason": "context_too_large"},
        {"session_id": "c", "messages": "4", "avg_context_size": "60000",
         "fallback_reason": "resident_prefix_insufficient"},
        {"session_id": "d", "messages": "3", "avg_context_size": None,
         "fallback_reason": "local_lease_active"},
        {"session_id": "e", "messages": "2", "avg_context_size": "20000",
         "fallback_reason": ""},
    ]


def test_reconstruct_sizes_expands_sessions_and_skips_missing_evidence() -> None:
    sizes = reconstruct_sizes(_sessions())
    assert len(sizes) == 21  # 10 + 5 + 4 + 2; the null session contributes none
    assert sizes.count(90_000.0) == 10
    assert reconstruct_sizes(_sessions(), "context_too_large") == [200_000.0] * 5


def test_register_reproduces_the_measured_baseline_shares() -> None:
    register = build_register(_sessions(), dict(DEFAULT_BASIS))
    uncapped = register["thresholds_cold"][0]
    assert uncapped["tokens"] is None
    assert uncapped["local_token_share_pct"] == pytest.approx(0.699, abs=1e-3)
    assert uncapped["local_stream_share_pct"] == pytest.approx(2.573, abs=1e-3)
    assert uncapped["added_tokens"] == 0.0


def test_register_records_the_shipped_compaction_state() -> None:
    register = build_register(_sessions(), dict(DEFAULT_BASIS))
    state = register["compaction_state"]
    assert state["assembly_time_trigger_ratio"] == 0.0
    assert state["assembly_time_enabled"] is False
    assert state["summarizer_model"] == "Qwen2.5-7B"
    assert state["summarizer_parallel"] == 1
    assert state["remote_boundary_compaction"] is False


def _consistent_basis(sessions: list[dict]) -> dict:
    """A basis whose measured shares match *sessions* (a self-consistent window)."""
    total = sum(reconstruct_sizes(sessions))
    return {
        "window_seconds": 2835.0,
        "offered_prompt_tokens": total,
        "offered_requests": 21,
        "local_prompt_tokens": 0.00699 * total,
        "local_streams": 0.02573 * 21,
        "local_busy_seconds": 1223.0,
        "capacity_tokens": 0.0162 * total,
        "source_report": "synthetic test window",
    }


def test_register_checks_all_pass_on_a_self_consistent_window() -> None:
    register = build_register(_sessions(), _consistent_basis(_sessions()))
    assert register["checks"]
    assert all_ok(register), [c for c in register["checks"] if not c["ok"]]


def test_checks_refuse_an_inconsistent_window() -> None:
    # The synthetic sessions hold ~2 M prompt tokens, so they cannot reproduce
    # the 2026-11-01 offered load: the calibration check must say so.
    register = build_register(_sessions(), dict(DEFAULT_BASIS))
    failing = {entry["claim"] for entry in register["checks"] if not entry["ok"]}
    assert "reconstructed size total within 3 % of offered prompt tokens" in failing
    # The structural checks are independent of the window and still pass.
    assert not any("never exceeds the capacity ceiling" in c for c in failing)
    assert not any("flat cap" in c for c in failing)
    assert not any("scale linearly" in c for c in failing)


def test_checks_are_derived_not_hard_coded() -> None:
    register = build_register(_sessions(), dict(DEFAULT_BASIS))
    tampered = json.loads(json.dumps(register))
    tampered["verdicts"]["busy_gate_ceiling_tokens"] = 1.0
    assert not all(entry["ok"] for entry in checks(tampered))


# ---------------------------------------------------------------------------
# Basis loading
# ---------------------------------------------------------------------------
def test_load_basis_reads_the_committed_baseline() -> None:
    path = REPO_ROOT / "docs" / "dev" / "local-share-baseline-2026-11-01.json"
    basis = load_basis(path)
    assert basis["offered_prompt_tokens"] == 125_577_780.0
    assert basis["local_prompt_tokens"] == 877_908.0
    assert basis["local_busy_seconds"] == 1223.0
    assert basis["capacity_tokens"] == pytest.approx(2_035_052.4775143089)


def test_load_basis_without_a_path_uses_the_recorded_defaults() -> None:
    assert load_basis(None) == DEFAULT_BASIS


# ---------------------------------------------------------------------------
# The committed register
# ---------------------------------------------------------------------------
def test_committed_register_reproduces_from_its_own_inputs() -> None:
    register = json.loads(REGISTER.read_text(encoding="utf-8"))
    recheck = recheck_register(register)
    assert recheck["ok"], recheck["drift"]
    assert recheck["recorded_checks_ok"]


def test_committed_register_drift_is_detected() -> None:
    register = json.loads(REGISTER.read_text(encoding="utf-8"))
    register["thresholds_cold"][-1]["added_tokens"] += 1.0
    recheck = recheck_register(register)
    assert not recheck["ok"]
    assert any(entry["section"] == "thresholds_cold" for entry in recheck["drift"])


def test_committed_register_keeps_the_window_and_read_only_claim() -> None:
    register = json.loads(REGISTER.read_text(encoding="utf-8"))
    assert register["read_only"] is True
    assert register["work_item"] == "LP-0MWCEGXEK000U6G6"
    assert register["window"]["seconds"] == pytest.approx(2835.0)
    assert abs(register["inputs"]["size_distribution"]["reconstruction_vs_offered_pct"]) <= 3.0


# ---------------------------------------------------------------------------
# The report quotes the register, not stale numbers
# ---------------------------------------------------------------------------
def test_report_quotes_the_register_figures() -> None:
    register = json.loads(REGISTER.read_text(encoding="utf-8"))
    report = (
        REPO_ROOT / "docs" / "dev" / "compaction-benefit-estimate-2026-11-01.md"
    ).read_text(encoding="utf-8")
    cap_25 = register["thresholds_cold"][-1]
    quoted = [
        f"{cap_25['local_token_share_pct']:.3f} %",  # 1.025 %
        f"{cap_25['local_stream_share_pct']:.3f} %",  # 3.488 %
        f"{register['verdicts']['busy_gate_ceiling_tokens']:,.0f}",  # 3,846
        f"{register['per_slot_cost']['total_vram_gb_estimate']:.2f}",  # 6.25
        f"{register['prefill_curve'][0]['cold_guard_default']:,.1f}",  # 236.8
        f"{register['prefill_curve'][1]['cold_guard_default']:,.1f}",  # 475.0
        f"{register['throughput']['idle_gate_seconds']:,.0f}",  # 1,612
    ]
    # The report uses thin-space digit grouping ("3 846") where the register
    # uses commas ("3,846"); compare with whitespace and commas normalised.
    def _normalise(text: str) -> str:
        return re.sub(r"[\s,]", "", text)

    haystack = _normalise(report)
    missing = [value for value in quoted if _normalise(value) not in haystack]
    assert not missing, f"report no longer quotes: {missing}"
