#!/usr/bin/env python3
"""Tests for the costed lever comparison and share target (LP-0MWCEII0V003V76A).

The assembly script puts four independently measured lever families on one unit
set and converts a candidate stream-share target into capacity, cache-reuse and
gate-time terms. These tests pin what a downstream decision leans on, so a later
edit cannot quietly move a row or a target:

- every family reports a throughput, a per-slot resource cost and **both** share
  denominators (a family that quotes one denominator against the other's
  baseline is wrong by the prompt-size mix);
- the target arithmetic is hand-checkable: streams x local prompt size, a
  capacity multiple, and gate-seconds = volume / rate;
- the gate-time test actually separates a routing target from a throughput
  target (the all-cold bound is what decides it);
- the properties the comparison claims (parallel prefill flat, compaction inert
  above the cold-prefill budget, quantisation under 2 % prefill) fail the checks
  when the underlying figures are tampered with;
- loud failure on an incomplete source register, never a silent zero;
- the committed register re-verifies from its own inputs, reports drift on a
  tampered field, and stays in step with the committed document.

Every assertion goes through the module's public functions or the committed
artefacts — none greps the source for a string.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from compare_prefill_levers import (  # noqa: E402
    RECOMMENDED_STREAM_SHARE_PCT,
    SOURCE_PATHS,
    RegisterError,
    _target,
    all_ok,
    build_register,
    check,
    derive,
    load_inputs,
    main,
    verify_register,
)

ARTEFACT = REPO_ROOT / "docs" / "dev" / "local-prefill-lever-comparison-2026-11-11.json"
REPORT = REPO_ROOT / "docs" / "dev" / "local-prefill-lever-comparison-2026-11-11.md"

FAMILY_KEYS = (
    "a_session_compaction",
    "b_parallel_prefill",
    "c_quantisation_geometry",
    "d_accept_ceiling",
)


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
def register(inputs: dict) -> dict:
    return build_register(REPO_ROOT)


def _families(derived: dict) -> dict[str, dict]:
    return {family["key"]: family for family in derived["families"]}


# ---------------------------------------------------------------------------
# structure: one unit set, both denominators
# ---------------------------------------------------------------------------


def test_all_four_families_are_present_in_spike_order(derived: dict) -> None:
    assert tuple(family["key"] for family in derived["families"]) == FAMILY_KEYS


def test_every_family_reports_throughput_cost_and_both_denominators(derived: dict) -> None:
    for family in derived["families"]:
        assert family["throughput"], family["key"]
        assert family["per_slot_resource_cost"], family["key"]
        share = family["local_share"]
        # Both denominators, always: a family without one of these cannot be
        # compared with a target expressed in the other.
        assert "token_share_delta_pp" in share, family["key"]
        assert "stream_share_delta_pp" in share, family["key"]
        assert family["work_item"].startswith("LP-"), family["key"]
        assert family["artefact"].endswith(".md"), family["key"]


def test_denominators_are_pinned_to_the_measured_basis(inputs: dict, derived: dict) -> None:
    token = derived["denominators"]["token"]
    stream = derived["denominators"]["stream"]
    measured = inputs["measured"]
    assert token["actual_pct"] == pytest.approx(measured["local_token_share_pct"], abs=1e-4)
    assert stream["actual_pct"] == pytest.approx(measured["local_stream_share_pct"], abs=1e-4)
    # The point ceiling on each denominator is the committed one, not a new one.
    assert token["ceiling_point_pct"] == pytest.approx(measured["canonical_token_share_pct"], rel=0.02)
    assert stream["ceiling_point_pct"] == pytest.approx(
        measured["canonical_stream_share_pct"], rel=0.02
    )


def test_denominator_bridge_is_the_prompt_size_mix(inputs: dict, derived: dict) -> None:
    mix = derived["denominators"]["prompt_size_mix"]
    expected = (
        inputs["measured"]["offered_prompt_tokens_per_stream"]
        / inputs["measured"]["local_prompt_tokens_per_stream"]
    )
    assert mix["offered_over_local_ratio"] == pytest.approx(expected, rel=1e-3)
    # Sanity: the mix is a factor, never ~1, otherwise the two denominators could
    # be used interchangeably and this whole comparison would be simpler.
    assert mix["offered_over_local_ratio"] > 2.0


def test_compaction_deltas_use_its_own_per_stream_size(derived: dict) -> None:
    """The converted stream carries the 25k cap, not the 19 085-token local mean.

    Getting this wrong is exactly the "mixed denominators" error the comparison
    exists to prevent, so it is pinned rather than left to documentation.
    """
    compaction = _families(derived)["a_session_compaction"]["local_share"]
    tokens = float(compaction["added_delivered_tokens"])
    streams = float(compaction["served_streams"])
    assert tokens / streams == pytest.approx(25_000, rel=0.05)
    ratio = float(compaction["stream_share_delta_pp"]) / float(compaction["token_share_delta_pp"])
    local_mean = derived["denominators"]["prompt_size_mix"]["local_prompt_tokens_per_stream"]
    offered_mean = derived["denominators"]["prompt_size_mix"]["offered_prompt_tokens_per_stream"]
    assert ratio == pytest.approx(offered_mean / (tokens / streams), rel=0.01)
    assert ratio != pytest.approx(offered_mean / local_mean, rel=0.05)


# ---------------------------------------------------------------------------
# family arithmetic
# ---------------------------------------------------------------------------


def test_parallel_prefill_row_is_flat_and_expensive(derived: dict) -> None:
    throughput = _families(derived)["b_parallel_prefill"]["throughput"]
    assert throughput["serial_aggregate_tok_s_at_44_770"] > 100
    assert throughput["parallel_aggregate_tok_s_at_44_770"] == pytest.approx(
        throughput["serial_aggregate_tok_s_at_44_770"], rel=0.05
    )
    assert throughput["wall_time_factor"] > 1.5
    assert throughput["per_request_ratio"] < 1.0
    cost = _families(derived)["b_parallel_prefill"]["per_slot_resource_cost"]
    # Two slots' checkpoint state for no additional throughput.
    assert cost["checkpoint_state_gib_total"] == pytest.approx(
        2 * cost["checkpoint_state_gib_per_slot"], rel=0.1
    )


def test_compaction_converts_only_below_the_cold_prefill_budget(derived: dict) -> None:
    share = _families(derived)["a_session_compaction"]["local_share"]
    assert share["shipped_targets_convert_nothing"] is True
    assert share["added_delivered_tokens"] > 0
    assert share["stream_share_delta_pp"] > share["token_share_delta_pp"]
    for cap, delta in share["shipped_targets_pp"].items():
        assert delta["token_share_delta_pp"] == 0.0, cap
        assert delta["stream_share_delta_pp"] == 0.0, cap


def test_compaction_gain_is_bounded_by_idle_gate_time(derived: dict) -> None:
    family = _families(derived)["a_session_compaction"]
    demand = family["throughput"]["demand_gate_seconds_at_25k"]
    idle = family["throughput"]["idle_gate_seconds"]
    # The lever's own estimate says demand exceeds idle gate time by ~17x, which
    # is why the converted volume is gate-bound rather than demand-bound.
    assert demand > 10 * idle
    added = family["local_share"]["added_delivered_tokens"]
    rate = family["throughput"]["gate_rate_tok_s"]
    assert added == pytest.approx(idle * rate, rel=0.01)


def test_quantisation_prefill_effect_is_bounded_by_weight_time_share(derived: dict) -> None:
    throughput = _families(derived)["c_quantisation_geometry"]["throughput"]
    for candidate in throughput["candidates"]:
        assert candidate["prefill_gain_measured_share_pct"] < 2.0
        # The byte-ratio ceiling is an order of magnitude larger: it is the bound
        # the measurement refuted, so the two must not be confused.
        assert candidate["prefill_gain_ceiling_pct"] > 10 * candidate[
            "prefill_gain_measured_share_pct"
        ] / 2
        assert candidate["weight_share_pct_of_prefill_pass"] < 10.0
    share = _families(derived)["c_quantisation_geometry"]["local_share"]
    assert share["geometry_token_share_delta_pp"] == 0.0
    assert share["geometry_stream_share_delta_pp"] == 0.0


def test_accept_option_adds_no_capacity(derived: dict) -> None:
    family = _families(derived)["d_accept_ceiling"]
    assert family["throughput"]["rate_added_tok_s"] == 0.0
    assert family["per_slot_resource_cost"]["gpu_memory_gib"] == 0.0
    assert family["per_slot_resource_cost"]["config_changes"] == 0
    assert family["local_share"]["ac9_reachable"] is True
    assert family["local_share"]["ac1_10pct_reachable"] is False


def test_combined_bound_does_not_reach_ten_percent_of_streams(derived: dict) -> None:
    bound = derived["combined_bound"]
    assert bound["stream_share_after_pct"] < 10.0
    assert bound["token_share_after_pct"] < 10.0
    assert bound["remaining_factor_to_10pct"]["stream"] > 2.0
    assert bound["gap_closed_pct_of_gap"]["stream"] < 20.0
    assert bound["rate_factors"]["parallel_prefill"] == 1.0
    assert bound["rate_factors"]["geometry"] == 1.0


# ---------------------------------------------------------------------------
# target arithmetic
# ---------------------------------------------------------------------------


def test_target_arithmetic_is_hand_checkable(inputs: dict) -> None:
    target = _target(inputs, 3.5, "test")
    measured = inputs["measured"]
    expected_streams = 0.035 * measured["offered_streams"]
    expected_tokens = expected_streams * measured["local_prompt_tokens_per_stream"]
    assert target["required_streams"] == pytest.approx(expected_streams, rel=1e-3)
    assert target["required_delivered_tokens"] == pytest.approx(expected_tokens, rel=1e-3)
    assert target["delta_delivered_tokens_vs_measured"] == pytest.approx(
        expected_tokens - measured["local_prompt_tokens"], rel=1e-3
    )
    # Gate seconds are volume / rate, by definition.
    rate = inputs["ceiling"]["point"]["engine_rate_tok_s"]
    assert target["gate_seconds_if_marginal_cold"] == pytest.approx(
        (expected_tokens - measured["local_prompt_tokens"]) / rate, rel=1e-3
    )
    reuse = inputs["reuse_fractions"]["point"]
    assert target["gate_seconds_at_point_reuse"] == pytest.approx(
        target["gate_seconds_if_marginal_cold"] * (1.0 - reuse), rel=1e-3
    )
    assert target["factor_vs_point_capacity"] == pytest.approx(
        expected_tokens / inputs["ceiling"]["point"]["delivered_tokens"], rel=1e-3
    )


def test_target_gate_cost_grows_with_the_target(inputs: dict) -> None:
    small = _target(inputs, 2.8, "small")
    large = _target(inputs, 5.0, "large")
    assert large["gate_seconds_if_marginal_cold"] > small["gate_seconds_if_marginal_cold"]
    assert large["factor_vs_point_capacity"] > small["factor_vs_point_capacity"]
    assert large["reuse_needed_at_point_rate"] > small["reuse_needed_at_point_rate"]


def test_no_capacity_is_required_below_the_ceiling(inputs: dict) -> None:
    """At or below the point ceiling the reuse requirement never exceeds it."""
    point_reuse = inputs["reuse_fractions"]["point"]
    for pct, _ in ((2.8, ""), (3.5, ""), (5.0, "")):
        target = _target(inputs, pct, "candidate")
        assert target["reachable_at_point"] is True
        assert target["reuse_needed_at_point_rate"] < point_reuse
    beyond = _target(inputs, 10.0, "beyond")
    assert beyond["reachable_at_point"] is False


def test_recommended_target_sits_inside_the_gate_and_above_the_baseline(
    inputs: dict, derived: dict
) -> None:
    target = derived["share_target"]["arithmetic"]
    assert target["target_stream_share_pct"] == RECOMMENDED_STREAM_SHARE_PCT
    assert target["gate_seconds_if_marginal_cold"] < target["idle_gate_seconds"]
    assert target["above_ac9_baseline_factor"] > 1.2
    assert target["factor_vs_point_capacity"] < 0.75
    # The token companion is the same volume, not a second target.
    assert target["equivalent_token_share_pct"] == pytest.approx(
        derived["share_target"]["recommended_token_share_companion_pct"], abs=0.05
    )
    assert derived["share_target"]["candidates"]


def test_target_table_labels_the_measured_and_ceiling_rows(derived: dict) -> None:
    labels = {
        candidate["label"]: candidate
        for candidate in derived["share_target"]["candidates"]
    }
    measured = labels["measured (2026-11-01 window)"]
    assert measured["delta_delivered_tokens_vs_measured"] == 0.0
    ceiling = labels["point stream ceiling"]
    assert ceiling["factor_vs_point_capacity"] == pytest.approx(1.0, abs=0.01)
    assert labels["parent AC9 baseline"]["target_stream_share_pct"] == pytest.approx(2.8)


# ---------------------------------------------------------------------------
# checks: the claims survive tampering
# ---------------------------------------------------------------------------


def test_committed_inputs_pass_every_check(register: dict) -> None:
    refuted = [entry for entry in register["checks"] if not entry["ok"]]
    assert not refuted, [entry["claim"] for entry in refuted]
    assert all_ok(register)


def test_tampered_parallel_prefill_ratio_refutes_the_flat_claim(inputs: dict) -> None:
    tampered = copy.deepcopy(inputs)
    tampered["parallel_prefill"]["derived"]["D1.aggregate_ratio_median"] = 1.9
    checks = check(tampered, derive(tampered))
    assert any(
        entry["claim"].startswith("parallel prefill adds no aggregate")
        and entry["verdict"] == "refuted"
        for entry in checks
    )


def test_tampered_compaction_cap_refutes_the_conversion_claim(inputs: dict) -> None:
    tampered = copy.deepcopy(inputs)
    # Pretend every shipped target converts something: the claim that only the
    # cold-prefill budget does must then fail.
    tampered["compaction"]["cold_rows"]["38000"]["added_tokens"] = 12345.0
    checks = check(tampered, derive(tampered))
    assert any(
        "cold-prefill budget" in entry["claim"] and entry["verdict"] == "refuted"
        for entry in checks
    )


def test_tampered_quantisation_gain_refutes_the_bound(inputs: dict) -> None:
    tampered = copy.deepcopy(inputs)
    tampered["quantisation"]["candidates"]["ud_q4_k_s_measured"][
        "prefill_gain_measured_share_pct"
    ] = 25.0
    checks = check(tampered, derive(tampered))
    assert any(
        entry["claim"].startswith("quantisation moves prefill") and not entry["ok"]
        for entry in checks
    )


def test_a_target_beyond_the_idle_gate_refutes_the_routing_claim(inputs: dict) -> None:
    """The gate-time check is what distinguishes routing from capacity."""
    tampered = copy.deepcopy(inputs)
    tampered["compaction"]["idle_gate_seconds"] = 100.0
    checks = check(tampered, derive(tampered))
    assert any(
        "idle prefill gate" in entry["claim"] and entry["verdict"] == "refuted"
        for entry in checks
    )


# ---------------------------------------------------------------------------
# loud failure, never a silent zero
# ---------------------------------------------------------------------------


def test_missing_source_register_raises(tmp_path: Path) -> None:
    for key, path in SOURCE_PATHS.items():
        if key == "compaction":
            continue
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((REPO_ROOT / path).read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(RegisterError):
        load_inputs(tmp_path)


def test_missing_block_in_a_register_raises(tmp_path: Path) -> None:
    for key, path in SOURCE_PATHS.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
        if key == "prefill_concurrency":
            data.pop("derived", None)
        target.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RegisterError):
        load_inputs(tmp_path)


def test_corrupt_json_raises_register_error(tmp_path: Path) -> None:
    for key, path in SOURCE_PATHS.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        body = (REPO_ROOT / path).read_text(encoding="utf-8")
        target.write_text("{not json" if key == "local_share_baseline" else body, encoding="utf-8")
    with pytest.raises(RegisterError):
        load_inputs(tmp_path)


# ---------------------------------------------------------------------------
# committed artefact: drift guard and CLI
# ---------------------------------------------------------------------------


def test_committed_register_matches_the_script(register: dict) -> None:
    committed = json.loads(ARTEFACT.read_text(encoding="utf-8"))
    # Volatile identity fields are the only ones allowed to differ.
    for key in ("work_item", "spike_work_item", "target_item", "title"):
        assert committed[key] == register[key], key
    assert committed["derived"] == json.loads(json.dumps(register["derived"]))
    assert committed["checks"] == register["checks"]


def test_committed_register_has_no_drift(register: dict) -> None:
    assert verify_register(json.loads(ARTEFACT.read_text(encoding="utf-8"))) == []
    assert verify_register(register) == []


def test_tampered_register_reports_drift(register: dict) -> None:
    tampered = copy.deepcopy(json.loads(ARTEFACT.read_text(encoding="utf-8")))
    tampered["derived"]["share_target"]["recommended_stream_share_pct"] = 9.9
    drift = verify_register(tampered)
    assert drift
    assert any("share_target" in entry["claim"] for entry in drift)


def test_cli_json_and_check_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["work_item"] == "LP-0MWCEII0V003V76A"
    assert main(["--register", str(ARTEFACT), "--check"]) == 0
    out = capsys.readouterr().out
    assert "no drift" in out


# ---------------------------------------------------------------------------
# document <-> register consistency
# ---------------------------------------------------------------------------


def _table_rows(document: str, header_first_cell: str) -> list[list[str]]:
    rows: list[list[str]] = []
    inside = False
    for line in document.splitlines():
        if not line.startswith("|"):
            inside = False
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == header_first_cell:
            inside = True
            continue
        if not inside:
            continue
        if all(set(cell) <= {"-", ":", " "} for cell in cells if cell):
            continue
        rows.append(cells)
    return rows


def _number(cell: str) -> float:
    cleaned = (
        cell.replace("*", "").replace("%", "").replace("pp", "").replace(" ", "").replace(",", "")
    )
    return float(cleaned)


def test_document_comparison_table_matches_the_register(register: dict) -> None:
    document = REPORT.read_text(encoding="utf-8")
    rows = _table_rows(document, "Family")
    assert len(rows) == 4, rows
    families = {family["key"]: family for family in register["derived"]["families"]}
    order = ["a_session_compaction", "b_parallel_prefill", "c_quantisation_geometry", "d_accept_ceiling"]
    for row, key in zip(rows, order):
        assert row[0].startswith("("), row[0]
        assert _number(row[3]) == pytest.approx(
            families[key]["local_share"]["token_share_delta_pp"], abs=0.006
        ), key
        assert _number(row[4]) == pytest.approx(
            families[key]["local_share"]["stream_share_delta_pp"], abs=0.006
        ), key
        assert row[5], key


def test_document_target_table_matches_the_register(register: dict) -> None:
    document = REPORT.read_text(encoding="utf-8")
    rows = _table_rows(document, "Stream share target")
    candidates = {
        round(candidate["target_stream_share_pct"], 4): candidate
        for candidate in register["derived"]["share_target"]["candidates"]
    }
    assert rows, "target table not found"
    assert {_number(row[0]) for row in rows} == set(candidates)
    for row in rows:
        pct = _number(row[0])
        candidate = candidates[pct]
        assert _number(row[1]) == pytest.approx(candidate["required_streams"], abs=0.01), pct
        assert _number(row[2]) == pytest.approx(
            candidate["required_delivered_tokens"], abs=1.0
        ), pct
        assert _number(row[3]) == pytest.approx(candidate["factor_vs_point_capacity"], abs=0.002), pct
        assert _number(row[6]) == pytest.approx(
            candidate["equivalent_token_share_pct"], abs=0.002
        ), pct


def test_document_quotes_the_recommended_target_and_bound(register: dict) -> None:
    document = REPORT.read_text(encoding="utf-8")
    target = register["derived"]["share_target"]["arithmetic"]
    bound = register["derived"]["combined_bound"]
    assert f"{target['target_stream_share_pct']:.1f} %" in document
    assert f"{target['equivalent_token_share_pct']:.3f} %" in document
    assert f"{bound['stream_share_after_pct']:.3f} %" in document
    assert "LP-0MVWEZCB0006WPBQ" in document
    # AC4 traceability: each producing child is named in the document.
    for family in register["derived"]["families"]:
        assert family["work_item"] in document, family["work_item"]
