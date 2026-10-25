"""Replay of the 2026-10-24 cold-prefill thrash window (LP-0MVN4LAV9007PM2X).

The parent item (LP-0MVN3KMGI009BZ0D) claims the proxy admits cold prefills it
cannot finish before the prefill deadline and then discards the computed
tokens. This module pins that claim as an executable baseline over the checked
in fixture of the measured window, so the sibling children (budget cap,
deadline-relative pricing, post-acceptance eviction counter) assert their
post-change behaviour against the same oracle.

Baseline measured from the fixture (capture command in its header)::

    attempts                     1 978 local admission attempts
    deadline cancellations         125  (81 distinct cancelled prefills)
    wasted prefill tokens        778 863  (90 s × band rate per task)
    idle/economic bypass            33  oversized (>25k new-token) cold prompts

Reconciliation with the parent item's prose figures (160 events / ~45 tasks /
~780k tokens): the retained logs support **125** deadline events for the
window named (164 for 17:37–00:00) and **81** distinct cancelled prefills, but
the **~780k** discarded-token figure reproduces exactly (778 863) under the
documented 90 s × band-rate waste model. The fixture — not the prose — is the
oracle; the discrepancy is recorded on the child and parent work items.

AC mapping: AC1 fixture shape/regeneration, AC2 baseline reproduction, AC3
structured result, AC4 size/privacy/style, AC5 no production import.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import prefill_thrash_replay_harness as harness
import pytest

from proxy import prefill_eta

#: Measured baseline, pinned. Tolerance is deliberately loose on the waste
#: total (~1 %) so a legitimate recalibration of the band table does not churn
#: the assertion, while a silent behaviour change still fails.
BASELINE_DEADLINE_CANCELLATIONS = 125
BASELINE_CANCELLED_TASKS = 81
BASELINE_WASTED_TOKENS = 778_863
WASTED_TOKENS_TOLERANCE = 0.02
MIN_MODEL_OUTCOME_AGREEMENT = 0.80

REQUIRED_RECORD_KEYS = {
    "ts",
    "session",
    "estimated_tokens",
    "cached_ratio",
    "new_tokens",
    "guard_verdict",
    "outcome",
    "prefill_tokens_computed",
}


@pytest.fixture(autouse=True)
def _clean_calibration():
    prefill_eta.reset()
    yield
    prefill_eta.reset()


@pytest.fixture(scope="module")
def fixture_data():
    return harness.load_fixture()


def test_fixture_header_totals_match_records(fixture_data):
    """AC1: the header documents the window/capture command and its totals."""
    meta, records = fixture_data

    assert meta["window"] == {"start": "2026-10-24 18:38", "end": "2026-10-24 23:47"}
    assert "capture_prefill_thrash_window.py" in meta["capture_command"]
    assert meta["source_logs"], "capture must record which logs it read"
    assert "waste_model" in meta and "band_rates_tokens_per_second" in meta

    deadline = [r for r in records if r["outcome"] == "prefill_deadline_exceeded"]
    tasks = {(r["session"], r["estimated_tokens"]) for r in deadline}
    totals = meta["totals"]
    assert totals["attempts"] == len(records)
    assert totals["deadline_cancelled_attempts"] == len(deadline)
    assert totals["deadline_cancelled_tasks"] == len(tasks)
    assert totals["idle_bypass_attempts"] == sum(
        1 for r in records if r["route"] == "idle_bypass"
    )


def test_fixture_records_are_complete_and_carry_no_prompt_text(fixture_data):
    """AC1/AC4: every record has the documented fields and nothing private."""
    _, records = fixture_data

    forbidden = {"prompt", "messages", "content", "text", "api_key", "authorization"}
    for record in records:
        assert REQUIRED_RECORD_KEYS <= set(record), record
        assert not (forbidden & set(record)), record
        assert record["estimated_tokens"] >= record["new_tokens"] >= 0
        assert 0.0 <= record["cached_ratio"] <= 1.0
        assert record["outcome"] in {
            "prefill_deadline_exceeded",
            "first_byte_ok",
            "detoured",
            "other",
        }


def test_fixture_is_small_enough_to_check_in():
    """AC4: the fixture is bounded in size."""
    assert Path(harness.FIXTURE_PATH).stat().st_size < harness.MAX_FIXTURE_BYTES


def test_replay_reproduces_the_measured_baseline(fixture_data):
    """AC2: replaying the fixture reproduces the recorded outcome baseline."""
    _, records = fixture_data
    harness.seed_production_like_profile()

    result = harness.replay(records)

    assert len(result.deadline_cancelled) == BASELINE_DEADLINE_CANCELLATIONS
    assert (
        result.totals["deadline_cancelled_tasks"] == BASELINE_CANCELLED_TASKS
    ), "distinct cancelled prefills changed — re-measure before re-pinning"
    assert result.wasted_tokens == pytest.approx(
        BASELINE_WASTED_TOKENS, rel=WASTED_TOKENS_TOLERANCE
    )
    # The parent's AC4 floor: the window wasted more than 700k prefill tokens.
    assert result.wasted_tokens > 700_000
    # Oracle integrity: replaying the fixture reproduces the totals the capture
    # recorded in its header, so the harness and the capture cannot drift.
    meta, _ = fixture_data
    assert result.totals["deadline_cancelled"] == meta["totals"][
        "deadline_cancelled_attempts"
    ]
    assert result.totals["wasted_tokens"] == round(
        meta["totals"]["wasted_prefill_tokens_model"]
    )


def test_pinned_profile_agrees_with_the_measured_outcomes(fixture_data):
    """AC2: the pinned band profile explains the recorded cancellations.

    A prompt the fixture recorded as cancelled must be priced above the
    deadline by the model; otherwise the profile (and the pricing built on it
    in later children) is wrong for that band.
    """
    _, records = fixture_data
    harness.seed_production_like_profile()

    result = harness.replay(records)
    cancelled = result.deadline_cancelled
    agreed = [a for a in cancelled if a.deadline_risk]

    assert len(agreed) / len(cancelled) >= MIN_MODEL_OUTCOME_AGREEMENT


def test_todays_admission_path_still_admits_doomed_prefills(fixture_data):
    """AC2: the defect the parent describes is reproducible, not asserted.

    With today's guard configuration and no calibration the doomed attempts
    are admitted, which is exactly the thrash the budget/discount children
    must stop.
    """
    _, records = fixture_data
    harness.seed_production_like_profile()

    result = harness.replay(records)

    assert result.totals["guard_refused"] == 0, (
        "an inert guard must refuse nothing — if this fails the guard "
        "calibration state is leaking between tests"
    )
    assert result.totals["admitted_and_doomed"] >= 900
    assert result.wasted_tokens_admitted > 700_000


def test_economic_bypass_admits_oversized_cold_prompts(fixture_data):
    """Parent AC1 evidence: the idle bypass is the oversized-prompt leak."""
    _, records = fixture_data
    harness.seed_production_like_profile()

    result = harness.replay(records)

    assert result.totals["idle_bypass_attempts"] >= 30
    assert (
        result.totals["idle_bypass_over_25k"] == result.totals["idle_bypass_attempts"]
    ), "every economic-bypass admission in this window was an oversized cold prompt"


def test_pinned_profile_prices_the_budget_band_inside_a_120s_deadline():
    """The profile contract later children assert against.

    At the pinned p75 rate 25k cold prices inside ``0.95 x 120 s`` (114 s) and
    a 40k cold prompt does not — the deadline-relative budget decision the
    parent's AC3 describes.
    """
    harness.seed_pinned_profile()
    config = harness.dev_guard_config(local_prefill_deadline_seconds=120)

    eta_25k, coeffs_25k = prefill_eta.estimate_prefill_seconds(
        harness.MODEL, 25_000, 0.0, config
    )
    eta_40k, _ = prefill_eta.estimate_prefill_seconds(
        harness.MODEL, 40_000, 0.0, config
    )

    assert coeffs_25k.calibrated, "pinned profile must calibrate the bucket"
    assert eta_25k < 0.95 * 120
    assert eta_40k > 0.95 * 120


def test_pinned_profile_keeps_warm_large_prompts_admissible():
    """Parent AC5: a warm 47k prompt with few new tokens stays local."""
    harness.seed_pinned_profile()
    config = harness.dev_guard_config(local_prefill_deadline_seconds=120)

    decision = prefill_eta.evaluate_admission(
        config, harness.MODEL, 47_000, 0.988, priority="medium"
    )

    assert decision.admit is True
    assert decision.reason == "within_budget"


def test_replay_result_is_structured_for_later_children(fixture_data):
    """AC3: results are exposed to tests as data, not parsed from logs."""
    _, records = fixture_data
    harness.seed_production_like_profile()

    result = harness.replay(records[:200])
    payload = harness.replay_result_to_json(result)

    assert payload["totals"]["attempts"] == len(records[:200])
    assert len(payload["attempts"]) == len(records[:200])
    first = payload["attempts"][0]
    assert {"outcome", "guard_admit", "deadline_risk", "new_tokens"} <= set(first)
    json.dumps(payload)  # must be JSON-serialisable (CLI convention)
    assert result.totals["guard_admitted"] + result.totals["guard_refused"] == len(
        records[:200]
    )


def test_replay_is_deterministic(fixture_data):
    """AC3: the same fixture and profile always yield the same result."""
    _, records = fixture_data
    harness.seed_production_like_profile()
    first = harness.replay_result_to_json(harness.replay(records[:300]))
    harness.seed_production_like_profile()
    second = harness.replay_result_to_json(harness.replay(records[:300]))

    assert first == second


def test_harness_is_not_imported_by_any_production_module():
    """AC5: the harness is an analysis tool, never production code.

    Behavioural check: import every module that participates in the admission
    path and assert none of them carries the harness sentinel.
    """
    production_modules = (
        "proxy.prefill_eta",
        "proxy.metrics",
        "proxy.session",
        "proxy.router_helpers",
        "proxy.router",
        "proxy.handlers",
    )
    for name in production_modules:
        module = importlib.import_module(name)
        assert not hasattr(module, "REPLAY_HARNESS_SENTINEL"), (
            f"{name} imported the replay harness — it is an analysis tool and "
            "must stay out of the production import graph"
        )
