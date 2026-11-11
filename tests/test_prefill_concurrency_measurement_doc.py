#!/usr/bin/env python3
"""Fidelity, provenance and arithmetic checks for the parallel-prefill measurement.

LP-0MWCEHDRV000B7HD answers "does opening the prefill gate raise aggregate
prefill throughput?" from two production windows and a controlled engine probe.
Its conclusions are numbers, and a prose document cannot be trusted to keep
numbers attached to their artefacts — so this module makes those promises
executable:

1. **No unsourced number.** Every figure registered in
   ``docs/dev/prefill-concurrency-baseline-2026-11-11.json`` renders its value in
   the report, so editing one side without the other fails.
2. **Arithmetic recomputes.** Aggregate throughput is recomputed from the raw
   artefacts as ``sum(tokens) / sum(engine ms)`` — the per-request rates are
   checked separately, so the per-request vs aggregate distinction the item
   turns on cannot be silently conflated.
3. **Experiment arithmetic reproduces.** Each probe round's aggregate rate is
   recomputed from ``prompt_tokens / window_seconds``, and the registered
   serial/parallel medians and their derived ratio are recomputed from those
   rounds.
4. **Derived formulas are honest.** Every derived figure recomputes from its
   registered inputs with its registered formula documented.
5. **The verdict is stated, with both AC axes.** The document carries the
   adopt/reject decision and the throughput/per-slot-cost comparison the spike's
   AC1 table consumes.

No llama-server, GPU, network or live host log is read: the artefacts are the
committed record.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

DOC_PATH = REPO_ROOT / "docs/dev/prefill-concurrency-2026-11-11.md"
REGISTER_PATH = REPO_ROOT / "docs/dev/prefill-concurrency-baseline-2026-11-11.json"

PRODUCTION_ARTEFACTS = {
    "P1": REPO_ROOT / "docs/dev/measurements/prefill-concurrency-2026-11-11-production-2026-11-10.json",
    "P2": REPO_ROOT / "docs/dev/measurements/prefill-concurrency-2026-11-11-production-2026-11-11.json",
}

EXPERIMENT_ARTEFACTS = {
    ("c1", 1): REPO_ROOT / "docs/dev/measurements/prefill-concurrency-2026-11-11-c1-r1.json",
    ("c2", 1): REPO_ROOT / "docs/dev/measurements/prefill-concurrency-2026-11-11-c2-r1.json",
    ("c1", 2): REPO_ROOT / "docs/dev/measurements/prefill-concurrency-2026-11-11-c1-r2.json",
    ("c2", 2): REPO_ROOT / "docs/dev/measurements/prefill-concurrency-2026-11-11-c2-r2.json",
    ("c1", 3): REPO_ROOT / "docs/dev/measurements/prefill-concurrency-2026-11-11-c1-r3.json",
    ("c2", 3): REPO_ROOT / "docs/dev/measurements/prefill-concurrency-2026-11-11-c2-r3.json",
}

#: Two-sided tolerance for published one-decimal rates (rounding to 0.1).
TOLERANCE = 0.16


@pytest.fixture(scope="module")
def register() -> dict:
    return json.loads(REGISTER_PATH.read_text())


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC_PATH.read_text()


@pytest.fixture(scope="module")
def production() -> dict[str, dict]:
    return {name: json.loads(path.read_text()) for name, path in PRODUCTION_ARTEFACTS.items()}


@pytest.fixture(scope="module")
def experiment() -> dict[tuple[str, int], dict]:
    return {key: json.loads(path.read_text()) for key, path in EXPERIMENT_ARTEFACTS.items()}


def fig(register: dict, figure_id: str) -> float:
    block = register["figures"].get(figure_id) or register["derived"].get(figure_id)
    assert block is not None, f"figure {figure_id} is not registered"
    return float(block["value"])


# ---------------------------------------------------------------------------
# 1. No unsourced number: every registered figure is in the report
# ---------------------------------------------------------------------------


def test_every_registered_figure_is_rendered_in_the_report(register: dict, doc: str) -> None:
    missing = []
    for section in ("figures", "derived"):
        for figure_id, block in register[section].items():
            assert block["provenance"] in register["provenance_classes"], (
                f"{figure_id}: unknown provenance {block['provenance']}"
            )
            if block["rendered"] not in doc:
                missing.append(f"{figure_id} -> {block['rendered']}")
    assert not missing, f"registered figures missing from the report: {missing}"


def test_every_measured_figure_names_its_source(register: dict) -> None:
    for figure_id, block in register["figures"].items():
        assert block.get("source"), f"{figure_id} has no source"
    for figure_id, block in register["derived"].items():
        assert block.get("formula"), f"{figure_id} has no formula"
        assert block.get("inputs"), f"{figure_id} has no inputs"


# ---------------------------------------------------------------------------
# 2. Production windows recompute from their artefacts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "window,prefix",
    [
        ("P1", "P1"),
        ("P2", "P2"),
    ],
)
def test_production_figures_recompute_from_the_artefact(
    register: dict, production: dict, window: str, prefix: str
) -> None:
    artefact = production[window]

    def check(figure_id: str, expected: float) -> None:
        assert fig(register, figure_id) == pytest.approx(expected, abs=TOLERANCE), (
            f"{figure_id} disagrees with the {window} artefact"
        )

    check(f"{prefix}.requests", artefact["requests"])
    check(f"{prefix}.prompt_tokens", artefact["prompt_tokens"])
    check(f"{prefix}.aggregate_tps", artefact["aggregate_tokens_per_second"])
    check(f"{prefix}.per_request_p50", artefact["per_request"]["p50"])
    check(f"{prefix}.per_request_p90", artefact["per_request"]["p90"])

    large = artefact["buckets"]["20k-40k"]
    check(f"{prefix}.bucket_20_40k_requests", large["requests"])
    check(
        f"{prefix}.bucket_20_40k_aggregate_tps",
        large["aggregate_tokens_per_second"],
    )
    check(f"{prefix}.bucket_20_40k_per_request_p50", large["per_request"]["p50"])
    check(f"{prefix}.bucket_20_40k_per_request_p90", large["per_request"]["p90"])
    check(f"{prefix}.bucket_40_80k_requests", artefact["buckets"]["40k-80k"]["requests"])

    # The aggregate is tokens over engine seconds — never the mean of the
    # per-request rates, which would be a different (and flattering) number.
    recomputed = artefact["prompt_tokens"] / (artefact["prompt_ms"] / 1000.0)
    assert artefact["aggregate_tokens_per_second"] == pytest.approx(recomputed, abs=TOLERANCE), (
        f"{window}: aggregate is not tokens/engine-seconds"
    )


def test_p2_40_80k_bucket_is_the_items_own_probe(register: dict, production: dict) -> None:
    bucket = production["P2"]["buckets"]["40k-80k"]
    assert bucket["requests"] == int(fig(register, "P2.bucket_40_80k_requests"))
    assert bucket["aggregate_tokens_per_second"] == pytest.approx(
        fig(register, "P2.bucket_40_80k_aggregate_tps"), abs=TOLERANCE
    )
    # The probe delivers the same token count in both arms, which is why the
    # bucket entry can be tied back to the experiment.
    assert bucket["prompt_tokens"] == int(fig(register, "E1.prompt_tokens"))


# ---------------------------------------------------------------------------
# 3. Experiment rounds recompute; the aggregate ratio follows from them
# ---------------------------------------------------------------------------


def _round_aggregate(artefact: dict) -> dict:
    round_row = artefact["rounds"][0]
    recomputed = round_row["prompt_tokens"] / round_row["window_seconds"]
    assert round_row["tokens_per_second"] == pytest.approx(recomputed, abs=TOLERANCE)
    return round_row


def test_serial_rounds_carry_one_request_at_full_rate(experiment: dict) -> None:
    for rep in (1, 2, 3):
        artefact = experiment[("c1", rep)]
        assert artefact["concurrency"] == 1
        round_row = _round_aggregate(artefact)
        row = artefact["rows"][0]
        # A single request's rate IS the aggregate: the gate is closed. (It
        # differs only by the wall-vs-engine timing overhead, <1 %.)
        assert row["prompt_per_second"] == pytest.approx(round_row["tokens_per_second"], rel=0.02)


def test_parallel_rounds_carry_two_requests_but_no_extra_aggregate(
    experiment: dict,
) -> None:
    for rep in (1, 2, 3):
        artefact = experiment[("c2", rep)]
        assert artefact["concurrency"] == 2
        round_row = _round_aggregate(artefact)
        assert len(artefact["rows"]) == 2
        assert round_row["prompt_tokens"] == 2 * artefact["rows"][0]["prompt_tokens"]
        # The window is the whole round, not the sum of the per-request rates.
        assert round_row["window_seconds"] >= max(row["wall_seconds"] for row in artefact["rows"])


def test_registered_medians_match_the_rounds(register: dict, experiment: dict) -> None:
    serial = [experiment[("c1", rep)]["rounds"][0]["tokens_per_second"] for rep in (1, 2, 3)]
    parallel = [experiment[("c2", rep)]["rounds"][0]["tokens_per_second"] for rep in (1, 2, 3)]
    serial_windows = [experiment[("c1", rep)]["rounds"][0]["window_seconds"] for rep in (1, 2, 3)]
    parallel_windows = [experiment[("c2", rep)]["rounds"][0]["window_seconds"] for rep in (1, 2, 3)]
    serial_per_request = [experiment[("c1", rep)]["rows"][0]["prompt_per_second"] for rep in (1, 2, 3)]
    slowest_parallel = [min(row["prompt_per_second"] for row in experiment[("c2", rep)]["rows"]) for rep in (1, 2, 3)]

    assert fig(register, "E1.c1_aggregate_tps_median") == pytest.approx(statistics.median(serial), abs=TOLERANCE)
    assert fig(register, "E1.c1_aggregate_tps_max") == pytest.approx(max(serial), abs=TOLERANCE)
    assert fig(register, "E1.c2_aggregate_tps_median") == pytest.approx(statistics.median(parallel), abs=TOLERANCE)
    assert fig(register, "E1.c2_aggregate_tps_min") == pytest.approx(min(parallel), abs=TOLERANCE)
    assert fig(register, "E1.c1_window_s_median") == pytest.approx(statistics.median(serial_windows), abs=TOLERANCE)
    assert fig(register, "E1.c2_window_s_median") == pytest.approx(statistics.median(parallel_windows), abs=TOLERANCE)
    assert fig(register, "E1.c1_per_request_p50_median") == pytest.approx(
        statistics.median(serial_per_request), abs=TOLERANCE
    )
    assert fig(register, "E1.c2_per_request_min_median") == pytest.approx(
        statistics.median(slowest_parallel), abs=TOLERANCE
    )
    assert fig(register, "E1.prompt_tokens") == pytest.approx(experiment[("c1", 1)]["rows"][0]["prompt_tokens"])


# ---------------------------------------------------------------------------
# 4. Derived arithmetic
# ---------------------------------------------------------------------------


def test_derived_ratios_and_daily_capacity_recompute(register: dict) -> None:
    c1 = fig(register, "E1.c1_aggregate_tps_median")
    c2 = fig(register, "E1.c2_aggregate_tps_median")
    c1_window = fig(register, "E1.c1_window_s_median")
    c2_window = fig(register, "E1.c2_window_s_median")
    c2_slowest = fig(register, "E1.c2_per_request_min_median")
    c1_per_request = fig(register, "E1.c1_per_request_p50_median")

    assert fig(register, "D1.aggregate_ratio_median") == pytest.approx(round(c2 / c1, 2), abs=0.01)
    assert fig(register, "D2.window_factor_median") == pytest.approx(round(c2_window / c1_window, 2), abs=0.01)
    assert fig(register, "D3.per_request_ratio_median") == pytest.approx(
        round(c2_slowest / c1_per_request, 2), abs=0.01
    )
    assert fig(register, "D4.single_gate_daily_tokens_at_40k") == pytest.approx(round(86400 * c1))
    assert fig(register, "D5.parallel_daily_tokens_measured_at_40k") == pytest.approx(round(86400 * c2))
    assert fig(register, "D6.daily_tokens_at_20_40k") == pytest.approx(
        round(86400 * fig(register, "P2.bucket_20_40k_aggregate_tps"))
    )
    assert fig(register, "D7.per_slot_checkpoint_gib") == pytest.approx(
        round(32 * fig(register, "E1.per_checkpoint_mib") / 1024, 1), abs=0.05
    )


def test_the_measured_verdict_is_reject_and_is_arithmetically_supported(
    register: dict,
) -> None:
    verdict = register["verdict"]
    assert verdict["decision"] == "reject"
    # The verdict's load-bearing fact: concurrency did not raise the aggregate.
    assert fig(register, "D1.aggregate_ratio_median") <= 1.01
    # ... while it did double the wall window and slow the slowest request.
    assert fig(register, "D2.window_factor_median") >= 1.9
    assert fig(register, "D3.per_request_ratio_median") < 0.9


# ---------------------------------------------------------------------------
# 5. The report states the AC answers the parent spike consumes
# ---------------------------------------------------------------------------


def test_report_states_verdict_and_both_ac_axes(doc: str) -> None:
    assert "Reject parallel prefill" in doc
    # AC1: the re-measurement names its windows and both rate views.
    assert "aggregate throughput" in doc
    assert "per-request" in doc
    # AC2: the test is explicit about how concurrency was forced.
    assert "concurrency 2" in doc
    # AC3: the reconciliation names the prior evidence it reconciles with.
    assert "W3" in doc
    # AC4: the comparison table carries throughput and per-slot cost.
    assert "Per-slot cost" in doc
    assert "131,072" in doc
    # AC5: the verdict names the shipped lever and its rollback.
    assert "local_max_concurrent_prefills" in doc
    assert "rollback" in doc.lower()
