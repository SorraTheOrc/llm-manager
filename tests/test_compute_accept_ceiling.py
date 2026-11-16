#!/usr/bin/env python3
"""Tests for the accept-the-ceiling arithmetic (LP-0MWCEI8QV007NIC0).

The analyser computes the hardware prefill ceiling and prices the "accept"
option for the spike's AC1 comparison table. These tests pin the arithmetic a
downstream decision leans on, so a later edit cannot quietly move the ceiling:

- the model's arithmetic on hand-checkable inputs (tier 1 is rate x time, tier 2
  divides by the reuse gap, tier 3 divides by the local prompt size);
- monotonicity and axis dominance: reuse moves the ceiling further than the
  whole measured engine-rate band does;
- the reconciliation with the ceiling of record, and that it is algebraic
  (the identity holds for *any* engine rate once reuse is implied);
- the ≥10 % requirements in each denominator and that they are never mixed;
- the lever bound: parallel prefill and geometry contribute zero, compaction is
  additive, quantisation is multiplicative, and the total stays inside the
  point ceiling's headroom;
- the accept option's zero throughput uplift and zero per-slot cost;
- loud failure on an incomplete or corrupt register (never a silent zero);
- the committed register re-verifying from its own inputs, a tampered register
  being reported as drift, and the committed artefact staying in step with the
  script.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from compute_accept_ceiling import (  # noqa: E402
    SCENARIOS,
    RegisterError,
    all_ok,
    bound_levers,
    build_register,
    check,
    derive,
    load_inputs,
    main,
    verify_register,
)

ARTEFACT = REPO_ROOT / "docs" / "dev" / "accept-ceiling-arithmetic-2026-11-01.json"
REPORT = REPO_ROOT / "docs" / "dev" / "accept-ceiling-arithmetic-2026-11-01.md"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def inputs() -> dict:
    return load_inputs(REPO_ROOT)


@pytest.fixture(scope="module")
def derived(inputs: dict) -> dict:
    return derive(inputs)


@pytest.fixture(scope="module")
def register() -> dict:
    return build_register(REPO_ROOT)


def _synthetic_inputs(**overrides) -> dict:
    """A minimal, hand-checkable input block (2 000 s, 100 tok/s, no decode)."""
    inputs = {
        "window": {"start": "a", "end": "b", "seconds": 2000.0, "source": "synthetic"},
        "measured": {
            "offered_prompt_tokens": 1_000_000.0,
            "offered_streams": 1_000.0,
            "local_prompt_tokens": 100_000.0,
            "local_streams": 10.0,
            "local_busy_seconds": 500.0,
            "local_completion_tokens": 1_000.0,
            "local_prompt_tokens_per_stream": 10_000.0,
            "offered_prompt_tokens_per_stream": 1_000.0,
            "local_rate_tok_s": 200.0,
            "offered_prompt_tok_s": 500.0,
            "canonical_ceiling_tokens": 400_000.0,
            "canonical_token_share_pct": 40.0,
            "canonical_stream_share_pct": 40.0,
            "source": "synthetic",
        },
        "engine": {
            "rate_min_tok_s": 80.0,
            "rate_point_tok_s": 100.0,
            "rate_max_tok_s": 120.0,
            "rate_min_label": "synthetic",
            "rate_point_label": "synthetic",
            "rate_max_label": "synthetic",
            "solo_p50_tok_s": 120.0,
            "paired_aggregate_tok_s": 120.0,
            "source": "synthetic",
        },
        "decode": {"rate_tok_s": 50.0, "slot_seconds": 20.0, "share_of_busy_pct": 0.0, "source": "synthetic"},
        "reuse": {
            "reuse_min": 0.25,
            "reuse_max": 0.75,
            "reuse_min_label": "synthetic",
            "reuse_max_label": "synthetic",
            "source": "synthetic",
        },
        "denominators": {
            "token": {"label": "tokens", "value": 1_000_000.0},
            "stream": {"label": "streams", "value": 1_000.0, "tokens_per_stream": 10_000.0},
            "ruled_out": {},
            "source": "synthetic",
        },
        "levers": {
            "compaction": {
                "label": "synthetic",
                "cap_tokens": 25_000,
                "added_tokens": 10_000.0,
                "served_streams": 1.0,
                "idle_gate_seconds": 100.0,
                "token_share_delta_pp": 1.0,
                "stream_share_delta_pp": 0.1,
                "per_slot_cost": {},
                "source": "synthetic",
            },
            "parallel_prefill": {"verdict": "rejected", "prefill_gain_pct": 0.0, "source": "synthetic"},
            "quantisation": {
                "prefill_gain_high_pct": 0.0,
                "prefill_gain_low_pct": 0.0,
                "capacity_token_share_after_pct": 10.0,
                "capacity_stream_share_after_pct": 10.0,
                "source": "synthetic",
            },
            "geometry": {"verdict": "none", "prefill_gain_pct": 0.0, "source": "synthetic"},
        },
        "base_shares": {"token_share_pct": 10.0, "stream_share_pct": 10.0, "source": "synthetic"},
        "published": {"stream_ceiling_pct": 40.0, "stream_definition_pct": 40.0, "token_definition_pct": 40.0},
        "parent_targets": {
            "ac9_local_share_baseline_pct": 2.8,
            "ac9_denominator": "streams",
            "ac1_target_pct": 10.0,
            "ac1_denominator": "streams",
        },
    }
    inputs.update(overrides)
    return inputs


# ---------------------------------------------------------------------------
# the arithmetic itself
# ---------------------------------------------------------------------------


def test_tier1_is_rate_times_available_gate_time(derived: dict, inputs: dict) -> None:
    """Tier 1 is exactly ``prefill_seconds x rate`` — no reuse, no denominators."""
    hardware = derived["hardware_prefill_ceiling"]
    for name in SCENARIOS:
        expected = (
            derived["prefill_seconds_available"] * inputs["engine"][f"rate_{name}_tok_s"]
        )
        assert hardware["prefill_tokens"][name] == pytest.approx(expected, rel=1e-6)
    assert hardware["point_pct"] == pytest.approx(
        100.0
        * hardware["prefill_tokens"]["point"]
        / inputs["measured"]["offered_prompt_tokens"],
        abs=1e-3,
    )


def test_prefill_gate_time_excludes_the_measured_decode_reserve(derived: dict, inputs: dict) -> None:
    """The reserve is a *share* of gate time, which is what makes it scalable."""
    expected = inputs["window"]["seconds"] * (1.0 - inputs["decode"]["share_of_busy_pct"] / 100.0)
    assert derived["prefill_seconds_available"] == pytest.approx(expected, abs=1e-2)
    assert derived["prefill_seconds_available"] < inputs["window"]["seconds"]


def test_tier2_divides_the_reuse_gap(derived: dict) -> None:
    """Delivered = prefilled / (1 - reuse), for every scenario."""
    for name in SCENARIOS:
        row = derived["ceiling"][name]
        assert row["delivered_tokens"] == pytest.approx(
            row["prefill_capacity_tokens"] / (1.0 - row["reuse_fraction"]), rel=1e-6
        )


def test_tier3_divides_by_the_local_prompt_size(derived: dict, inputs: dict) -> None:
    """Stream share uses the *current local* mean prompt size, not the offered one."""
    tokens_per_stream = inputs["denominators"]["stream"]["tokens_per_stream"]
    offered_streams = inputs["denominators"]["stream"]["value"]
    for name in SCENARIOS:
        row = derived["ceiling"][name]
        assert row["servable_streams"] == pytest.approx(
            row["delivered_tokens"] / tokens_per_stream, abs=1e-2
        )
        assert row["stream_share_pct"] == pytest.approx(
            100.0 * row["servable_streams"] / offered_streams, abs=1e-3
        )


def test_synthetic_inputs_give_hand_checkable_numbers() -> None:
    """2 000 s at 100 tok/s with no decode reserve is 200 000 prefilled tokens."""
    derived = derive(_synthetic_inputs())
    point = derived["ceiling"]["point"]
    assert point["prefill_capacity_tokens"] == pytest.approx(200_000.0, rel=1e-9)
    # reuse implied by the synthetic window: 1 - (500 s x 100 tok/s)/100 000 = 0.5
    assert derived["reuse_fractions"]["point"] == pytest.approx(0.5, abs=1e-9)
    assert point["delivered_tokens"] == pytest.approx(400_000.0, rel=1e-9)
    assert point["token_share_pct"] == pytest.approx(40.0, abs=1e-3)  # 400k of 1M
    assert point["stream_share_pct"] == pytest.approx(4.0, abs=1e-3)  # 40 of 1 000 streams


def test_no_decode_reserve_means_prefill_uses_the_whole_window() -> None:
    derived = derive(_synthetic_inputs())
    assert derived["prefill_seconds_available"] == pytest.approx(2000.0, abs=1e-2)


# ---------------------------------------------------------------------------
# reconciliation and axis dominance
# ---------------------------------------------------------------------------


def test_point_model_reproduces_the_ceiling_of_record(derived: dict, inputs: dict) -> None:
    """The point row must land on the committed ceiling, not near it."""
    rec = derived["reconciliation"]
    assert rec["modelled_delivered_tokens"] == pytest.approx(
        inputs["measured"]["canonical_ceiling_tokens"], rel=1e-6
    )
    assert rec["modelled_stream_share_pct"] == pytest.approx(
        inputs["measured"]["canonical_stream_share_pct"], abs=0.01
    )


def test_the_reconciliation_is_algebraic_at_any_engine_rate(inputs: dict) -> None:
    """Reuse is implied from occupancy, so the rate cancels: the model reduces
    to the duty-cycle extrapolation. This is why the equality is *not* a
    validation of the engine rate, and it is stated as such in the docs."""
    for rate in (50.0, 100.0, 150.0):
        patched = copy.deepcopy(_synthetic_inputs())
        for key in ("rate_min_tok_s", "rate_point_tok_s", "rate_max_tok_s"):
            patched["engine"][key] = rate
        derived = derive(patched)
        expected = (
            patched["window"]["seconds"]
            / patched["measured"]["local_busy_seconds"]
            * patched["measured"]["local_prompt_tokens"]
        )
        assert derived["ceiling"]["point"]["delivered_tokens"] == pytest.approx(
            expected, rel=1e-9
        )


def test_reuse_moves_the_ceiling_further_than_the_rate_band(derived: dict) -> None:
    """Axis dominance: the decision hinges on reuse, so a lever that raises it
    (compaction) is worth more than one that raises the rate by a few percent."""
    grid = derived["grid"]
    rate_only = (
        grid["max"]["reuse_point"]["token_share_pct"]
        / grid["min"]["reuse_point"]["token_share_pct"]
    )
    reuse_only = (
        grid["point"]["reuse_max"]["token_share_pct"]
        / grid["point"]["reuse_min"]["token_share_pct"]
    )
    assert reuse_only > rate_only
    assert reuse_only > 5.0  # measured reuse spans 0.261-0.900


def test_implied_reuse_sits_between_the_two_measurements(derived: dict, inputs: dict) -> None:
    """The window ran at higher reuse than the aggregate and below the admitted
    population's warmth — an admitted-population bias, so both bounds are real."""
    assert (
        inputs["reuse"]["reuse_min"]
        < derived["reuse_fractions"]["point"]
        < inputs["reuse"]["reuse_max"]
    )


def test_ceiling_band_is_ordered_and_bounded(derived: dict) -> None:
    ceiling = derived["ceiling"]
    tokens = [ceiling[name]["token_share_pct"] for name in SCENARIOS]
    streams = [ceiling[name]["stream_share_pct"] for name in SCENARIOS]
    assert tokens == sorted(tokens)
    assert streams == sorted(streams)
    # The point row of record sits between the measured corners.
    assert tokens[0] < 1.621 < tokens[2]
    assert streams[0] < 5.964 < streams[2]


def test_degenerate_reuse_is_rejected_instead_of_dividing_by_zero() -> None:
    """A fully reused stream would make the ceiling infinite; say so, don't crash."""
    patched = copy.deepcopy(_synthetic_inputs())
    patched["reuse"]["reuse_max"] = 1.0
    with pytest.raises(RegisterError):
        derive(patched)


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


def test_ten_percent_token_share_is_unreachable(derived: dict) -> None:
    target = derived["targets"]["spike_ac1_token"]
    assert target["denominator"] == "token"
    assert not target["reachable_at_point"]
    assert not target["reachable_within_measured_band"]
    assert target["factor_vs_point_capacity"] > 6.0
    # The "what it would take" remedies are both far outside the measured band.
    assert target["rate_needed_at_point_reuse_tok_s"] > 5.0 * max(
        derived["engine_rates_tok_s"].values()
    )
    assert target["reuse_needed_at_point_rate"] > 0.95


def test_ten_percent_stream_share_needs_more_than_the_point_ceiling(derived: dict) -> None:
    target = derived["targets"]["spike_ac1_stream"]
    assert target["denominator"] == "stream"
    assert not target["reachable_at_point"]
    assert 1.5 < target["factor_vs_point_capacity"] < 2.0
    assert 350.0 < target["rate_needed_at_point_reuse_tok_s"] < 500.0


def test_parent_ac9_baseline_is_reachable(derived: dict) -> None:
    target = derived["targets"]["parent_ac9_stream"]
    assert target["reachable_at_point"]
    assert target["factor_vs_point_capacity"] < 1.0


def test_denominators_are_never_mixed(inputs: dict, derived: dict) -> None:
    """A stream target converts through the *local* mean prompt size; a token
    target does not convert at all. Mixing them is the spike's original bug."""
    tokens_per_stream = inputs["denominators"]["stream"]["tokens_per_stream"]
    offered_streams = inputs["denominators"]["stream"]["value"]
    stream_target = derived["targets"]["spike_ac1_stream"]
    token_target = derived["targets"]["spike_ac1_token"]
    assert stream_target["required_tokens"] == pytest.approx(
        0.10 * offered_streams * tokens_per_stream, abs=0.1
    )
    assert token_target["required_tokens"] == pytest.approx(
        0.10 * inputs["denominators"]["token"]["value"], abs=0.1
    )
    assert token_target["required_tokens"] != pytest.approx(
        stream_target["required_tokens"], rel=1e-3
    )


# ---------------------------------------------------------------------------
# lever bound
# ---------------------------------------------------------------------------


def test_zero_gain_levers_contribute_nothing(inputs: dict, derived: dict) -> None:
    """Parallel prefill (1.00x) and slot geometry (GPU-bound) are priced at 0."""
    levers = bound_levers(inputs, derived)
    assert levers["rate_factors"]["parallel_prefill"] == 1.0
    assert levers["rate_factors"]["geometry"] == 1.0


def test_compaction_is_additive_and_quantisation_multiplicative(inputs: dict, derived: dict) -> None:
    levers = bound_levers(inputs, derived)
    compaction_tokens = float(inputs["levers"]["compaction"]["added_tokens"])
    factor = levers["rate_factors"]["quantisation_high"]
    high = levers["combined_high"]
    low = levers["combined_low"]
    assert high["compaction_added_tokens"] == pytest.approx(compaction_tokens * factor, rel=1e-6)
    assert high["token_share_after_pct"] > low["token_share_after_pct"] > inputs["base_shares"]["token_share_pct"]


def test_the_levers_do_not_close_the_gap(derived: dict, inputs: dict) -> None:
    levers = bound_levers(inputs, derived)
    gap = levers["gap_to_target"]
    assert not gap["token"]["closed"]
    assert not gap["stream"]["closed"]
    assert 0.0 < gap["token"]["gap_closed_pct_of_gap"] < 10.0
    assert 0.0 < gap["stream"]["gap_closed_pct_of_gap"] < 25.0
    assert gap["token"]["remaining_factor"] > 9.0
    assert gap["stream"]["remaining_factor"] > 2.5


def test_lever_tokens_stay_inside_the_point_headroom(derived: dict, inputs: dict) -> None:
    """The levers convert headroom; they cannot exceed the ceiling's spare room."""
    levers = bound_levers(inputs, derived)
    added = (
        levers["combined_high"]["compaction_added_tokens"]
        + levers["combined_high"]["quantisation_freed_capacity_tokens"]
    )
    headroom = (
        derived["ceiling"]["point"]["delivered_tokens"]
        - inputs["measured"]["local_prompt_tokens"]
    )
    assert added < headroom


# ---------------------------------------------------------------------------
# the accept option row
# ---------------------------------------------------------------------------


def test_accept_option_buys_nothing_and_costs_nothing(register: dict) -> None:
    option = register["accept_option"]
    assert option["throughput_tok_s"]["rate_added_by_option_tok_s"] == 0.0
    assert option["per_slot_resource_cost"]["gpu_memory_gib"] == 0.0
    assert option["per_slot_resource_cost"]["config_changes"] == 0
    # It endorses the measured ceiling rather than a new one.
    assert option["local_share_pct"]["token_point"] == pytest.approx(1.6206, abs=1e-3)
    assert option["local_share_pct"]["stream_point"] == pytest.approx(5.9637, abs=1e-3)


# ---------------------------------------------------------------------------
# failure modes and the committed artefact
# ---------------------------------------------------------------------------


def test_passing_tier_three_as_a_missing_key_raises(inputs: dict) -> None:
    broken = copy.deepcopy(inputs)
    del broken["denominators"]["stream"]["tokens_per_stream"]
    with pytest.raises(KeyError):
        derive(broken)


def test_an_impossible_reuse_fraction_is_reported_not_silently_clamped() -> None:
    """A window that delivered less than the gate could prefill is not a valid
    input: reuse would be negative, and the tool must say so."""
    broken = copy.deepcopy(_synthetic_inputs())
    broken["measured"]["local_prompt_tokens"] = 10_000.0  # < 500 s x 100 tok/s
    with pytest.raises(RegisterError):
        derive(broken)


def test_missing_source_register_is_a_loud_failure(tmp_path: Path) -> None:
    with pytest.raises(RegisterError):
        load_inputs(tmp_path)


def test_verify_register_accepts_the_committed_artefact(register: dict) -> None:
    assert verify_register(register) == []


def test_verify_register_detects_a_tampered_derived_block(register: dict) -> None:
    tampered = copy.deepcopy(register)
    tampered["derived"]["ceiling"]["point"]["token_share_pct"] = 9.99
    drift = verify_register(tampered)
    assert drift, "a hand-edited derived figure must be reported as drift"
    assert any("token_share_pct" in entry["claim"] for entry in drift)


def test_verify_register_detects_tampered_inputs(register: dict) -> None:
    tampered = copy.deepcopy(register)
    tampered["inputs"]["engine"]["rate_max_tok_s"] = 999.0
    assert verify_register(tampered)


def test_all_checks_reproduce(register: dict) -> None:
    assert all_ok(register)
    assert len(register["checks"]) >= 8


def test_a_refuted_check_fails_the_gate(register: dict) -> None:
    tampered = copy.deepcopy(register)
    tampered["checks"][0]["ok"] = False
    tampered["checks"][0]["verdict"] = "refuted"
    assert not all_ok(tampered)


def test_committed_artefact_matches_a_fresh_build(register: dict) -> None:
    """Drift guard: the committed register must be what the script produces."""
    committed = json.loads(ARTEFACT.read_text(encoding="utf-8"))
    assert committed["derived"] == register["derived"]
    assert committed["levers_bound"] == register["levers_bound"]
    assert committed["accept_option"] == register["accept_option"]
    assert committed["checks"] == register["checks"]
    assert committed["inputs"] == register["inputs"]


def _squash(text: str) -> str:
    """Drop the thousands separators/grouping spaces the report styles with."""
    return text.replace(",", "").replace("\u202f", "").replace("\u00a0", "").replace(" ", "")


def test_report_quotes_the_registers_figures(register: dict) -> None:
    """Doc-fidelity guard: the page a decision-maker reads must carry the
    register's numbers, so the two cannot drift apart silently."""
    text = _squash(REPORT.read_text(encoding="utf-8"))
    derived = register["derived"]
    hardware = derived["hardware_prefill_ceiling"]
    ceiling = derived["ceiling"]
    levers = register["levers_bound"]

    expected = [
        # tier 1 (2 dp headline + 3 dp table cells)
        f"**{hardware['min_pct']:.2f} / {hardware['point_pct']:.2f} / {hardware['max_pct']:.2f} %**",
        # tiers 2/3 (3 dp cells)
        f"**{ceiling['min']['token_share_pct']:.3f} %**",
        f"**{ceiling['point']['token_share_pct']:.3f} %**",
        f"**{ceiling['max']['token_share_pct']:.3f} %**",
        f"**{ceiling['min']['stream_share_pct']:.3f} %**",
        f"**{ceiling['point']['stream_share_pct']:.3f} %**",
        f"**{ceiling['max']['stream_share_pct']:.3f} %**",
        # tier-1 table cells
        f"**{hardware['min_pct']:.3f} %**",
        f"**{hardware['point_pct']:.3f} %**",
        f"**{hardware['max_pct']:.3f} %**",
        # ≥10 % requirements
        f"{derived['targets']['spike_ac1_token']['rate_needed_at_point_reuse_tok_s']:,.1f} tok/s",
        f"{derived['targets']['spike_ac1_stream']['rate_needed_at_point_reuse_tok_s']:,.1f} tok/s",
        # lever bound
        f"**{levers['combined_high']['token_share_after_pct']:.4f} %**",
        f"**{levers['combined_high']['stream_share_after_pct']:.4f} %**",
        f"**{levers['gap_to_target']['token']['remaining_factor']:.2f}x**",
        f"**{levers['gap_to_target']['stream']['remaining_factor']:.2f}x**",
    ]
    missing = [item for item in expected if _squash(item) not in text]
    assert missing == [], f"report is out of step with the register: {missing}"


def test_report_names_the_fixed_denominators(register: dict) -> None:
    """The reconciliation child pinned the denominators; the page must say so."""
    text = REPORT.read_text(encoding="utf-8")
    assert "LP-0MWCEGJDP009WURD" in text
    assert "offered prompt tokens" in text
    assert "offered streams" in text


def test_cli_check_passes_on_the_committed_artefact() -> None:
    assert main(["--register", str(ARTEFACT), "--check"]) == 0


def test_cli_json_emits_the_register(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["work_item"] == "LP-0MWCEI8QV007NIC0"
    assert payload["read_only"] is True


def test_check_reports_every_assertion_as_reproducing(inputs: dict, derived: dict) -> None:
    """``check`` must not leave a claim unjudged."""
    levers = bound_levers(inputs, derived)
    entries = check(inputs, derived, levers)
    assert entries
    for entry in entries:
        assert entry["verdict"] in {"reproduces", "refuted"}
        assert entry["ok"] is (entry["verdict"] == "reproduces")
