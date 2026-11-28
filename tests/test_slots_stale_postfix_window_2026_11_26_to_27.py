#!/usr/bin/env python3
"""Artefact-contract tests for the re-measured ``slots_stale`` 24 h window.

LP-0MWSD6D24002J0K7 re-measures the undatable share of ``/llama/local/status``
polls on a ≥ 24 h restart-free window and commits the evidence as
``docs/dev/slots-stale-postfix-window-2026-11-26-to-27.{md,json}``. Its ACs are
numbers, so they are pinned here instead of trusted to prose:

1. **The window is what it claims.** Exactly 24 h, half-open, 0 in-window
   startup markers, coverage complete with no gaps (AC1/AC2).
2. **Both figures are consistent and moved.** The before figure is the
   precedent window's 4.47 % (557/12,450) and the after figure is below the
   item's 0.5 % target and below the ``slots_stale`` contract rate (AC2/AC4).
3. **The agreement check recomputes.** ``slots_stale`` == over-age + undatable,
   and the second reader (``slots_snapshot_age_distribution.py``) reports the
   same count (AC4).
4. **The per-hour breakdown reconciles.** Every hour from window start to
   window end is present, the hourly polls and stale counts sum to the totals,
   and no hour exceeds the item's 0.5 % target (AC5).
5. **Every residual stale poll is explained.** No over-age poll exists, and
   each of the residual stale polls names its own snapshot source and an
   explanation — none is an undatable cache entry (the old provenance gap)
   (AC7).
6. **The calendar records the measurement.** ``measurement-windows.md`` bills
   this item's window in § 3.4 and no longer lists it as an open window (AC1).

No production log is read: the committed artefacts are the record.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs" / "dev"

REPORT_PATH = DOCS_DIR / "slots-stale-postfix-window-2026-11-26-to-27.md"
DATASET_PATH = DOCS_DIR / "slots-stale-postfix-window-2026-11-26-to-27.json"
CALENDAR_PATH = DOCS_DIR / "measurement-windows.md"

WORK_ITEM = "LP-0MWSD6D24002J0K7"
PARENT_ITEM = "LP-0MWRXFO0I005UBVM"

#: The item's target: undatable polls must fall below 0.5 % of polls.
UNDATABLE_TARGET = 0.005

#: The precedent window's figures (docs/dev/slots-stale-postfix-window-2026-11-18-to-19.json).
BEFORE_POLLS = 12_450
BEFORE_STALE = 608
BEFORE_UNDATABLE = 557

WINDOW_SINCE = datetime(2026, 11, 26, 10, 18, 0)
WINDOW_UNTIL = datetime(2026, 11, 27, 10, 18, 0)


@pytest.fixture(scope="module")
def dataset() -> dict:
    return json.loads(DATASET_PATH.read_text())


@pytest.fixture(scope="module")
def report() -> str:
    return REPORT_PATH.read_text()


def _pct(value: float) -> str:
    """Render a rate the way the report writes it (two decimals)."""
    return f"{value * 100:.2f} %"


def test_dataset_and_report_are_committed_and_linked():
    assert DATASET_PATH.is_file(), "dataset must be committed under docs/dev/"
    assert REPORT_PATH.is_file(), "report must be committed under docs/dev/"
    assert "[`slots-stale-postfix-window-2026-11-26-to-27.json`]" in REPORT_PATH.read_text()


def test_dataset_identifies_the_item_and_its_parent(dataset):
    assert dataset["work_item"] == WORK_ITEM
    assert dataset["parent"] == PARENT_ITEM
    assert {e["commit"] for e in dataset["fix_under_test"]} >= {"aa6115bcc", "5ef9b5ebb"}


def test_window_is_exactly_24_hours_and_half_open(dataset):
    window = dataset["window"]
    since = datetime.strptime(window["since_local"], "%Y-%m-%d %H:%M:%S")
    until = datetime.strptime(window["until_local"], "%Y-%m-%d %H:%M:%S")
    assert since == WINDOW_SINCE
    assert until == WINDOW_UNTIL
    assert until - since == timedelta(hours=24)
    assert window["half_open"] is True
    assert window["window_closed"] is True


def test_window_is_restart_free_and_fully_covered(dataset):
    """AC1/AC2: no restart inside the window, and no missing log interval."""
    integrity = dataset["restart_integrity"]
    assert integrity["in_window_startup_markers"] == 0
    assert integrity["in_window_marker_files_checked"], "the marker scan must name its files"

    anchor = integrity["anchor_restart_marker"]
    # The anchor marker itself must sit outside the half-open window.
    assert "2026-11-26 10:17:49" in anchor

    coverage = dataset["coverage"]
    assert coverage["covered"] is True
    assert coverage["gaps"] == []
    assert coverage["measured_hours"] == 24.0
    assert coverage["window_hours"] == 24.0
    assert coverage["window_start_gap_seconds"] == 0.0
    assert coverage["window_end_gap_seconds"] == 0.0
    assert len(coverage["spans"]) >= 2


def test_before_figure_matches_the_precedent_window(dataset):
    before = dataset["before_reading"]
    assert before["status_polls"] == BEFORE_POLLS
    assert before["slots_stale"]["count"] == BEFORE_STALE
    assert before["stale_composition"]["undatable"] == BEFORE_UNDATABLE
    # 557 / 12,450 == 4.47 %
    assert before["undatable_share"]["rate"] == pytest.approx(0.0447, abs=5e-5)


def test_after_figure_clears_the_target_and_improves_on_before(dataset):
    """AC2/AC4: undatable share falls from 4.47 % to below 0.5 %."""
    after = dataset["after_reading"]
    polls = after["status_polls"]
    undatable = after["stale_composition"]["undatable"]
    over_age = after["stale_composition"]["over_age"]

    assert polls > 12_000, "a 24 h window at a ~30 s cadence cannot be this small"
    assert after["undatable_share"]["count"] == undatable
    assert after["undatable_share"]["of_polls"] == polls
    assert undatable / polls == pytest.approx(after["undatable_share"]["rate"], abs=1e-6)
    assert undatable / polls < UNDATABLE_TARGET
    assert after["slots_stale"]["count"] / polls < UNDATABLE_TARGET
    # A 100x improvement over the baseline is the point of the re-measurement.
    assert after["undatable_share"]["rate"] * 100 < dataset["before_reading"]["undatable_share"]["rate"]
    assert over_age == 0
    assert after["unattributed_stale_polls"] == 0


def test_agreement_check_recomputes(dataset):
    """AC4: slots_stale == over-age + undatable, and both readers agree."""
    after = dataset["after_reading"]
    agreement = dataset["agreement_check_f4"]

    assert agreement["capacity_signal_verification_slots_stale"] == after["slots_stale"]["count"]
    assert agreement["over_age_plus_undatable"] == (
        after["stale_composition"]["over_age"] + after["stale_composition"]["undatable"]
    )
    assert agreement["agrees"] is True
    assert agreement["slots_snapshot_age_distribution_contract_stale"] == after["slots_stale"]["count"]
    assert agreement["slots_snapshot_age_distribution_undatable"] == after["stale_composition"]["undatable"]


def test_tightened_thresholds_are_reported_and_non_increasing(dataset):
    """AC5: both threshold figures are published, and a looser ceiling is never worse."""
    tightened = dataset["tightened_reading"]
    by_threshold = {t["threshold_seconds"]: t for t in tightened["thresholds"]}
    assert set(by_threshold) >= {2.0, 5.0, 300.0}

    over = [by_threshold[t]["over"] for t in (2.0, 5.0, 300.0)]
    assert over == sorted(over, reverse=True)
    assert by_threshold[300.0]["over"] == dataset["after_reading"]["stale_composition"]["over_age"]

    polls = tightened["polls"]
    assert polls["status_polls"] == dataset["after_reading"]["status_polls"]
    assert polls["datable"] + polls["undatable"] == polls["status_polls"]
    assert tightened["contract_stale"]["count"] == polls["undatable"]


def test_per_hour_breakdown_reconciles_with_the_totals(dataset):
    """AC5: the hourly rows sum to the window's poll and stale counts."""
    hours = dataset["per_hour"]["hours"]
    assert len(hours) == 25, "a half-open 24 h window spans 25 clock-hour buckets"

    buckets = [datetime.strptime(h["hour_utc"], "%Y-%m-%dT%H:00Z") for h in hours]
    assert buckets == sorted(buckets)
    assert buckets[0] == WINDOW_SINCE.replace(minute=0)
    assert buckets[-1] == WINDOW_UNTIL.replace(minute=0)
    assert all(b - a == timedelta(hours=1) for a, b in zip(buckets, buckets[1:]))

    totals = dataset["per_hour"]["totals"]
    assert totals["polls"] == sum(h["polls"] for h in hours) == dataset["after_reading"]["status_polls"]
    assert totals["slots_stale"] == sum(h["slots_stale"] for h in hours)
    assert totals["slots_stale"] == dataset["after_reading"]["slots_stale"]["count"]

    for hour in hours:
        assert hour["slots_stale"] <= hour["polls"]
        if hour["polls"]:
            assert hour["slots_stale"] / hour["polls"] < UNDATABLE_TARGET


def test_every_residual_stale_poll_is_explained(dataset):
    """AC7: no over-age poll, and each residual poll names its own cause."""
    stale_polls = dataset["after_reading"]["stale_polls"]
    assert len(stale_polls) == dataset["after_reading"]["stale_composition"]["undatable"]
    for poll in stale_polls:
        # The old provenance gap was a counts-cache fallback; the residual is a
        # poll with no snapshot at all, which is a different (and explainable) state.
        assert poll["snapshot_source"] == "none"
        assert poll["slots_snapshot_age_seconds"] is None
        assert poll["explanation"].strip()
        assert poll["ts_utc"].startswith("2026-11-26T")
        assert WINDOW_SINCE <= datetime.strptime(poll["ts_utc"], "%Y-%m-%dT%H:%M:%SZ") < WINDOW_UNTIL


def test_report_states_both_figures_and_the_met_verdict(report, dataset):
    """The prose cannot drift from the numbers it quotes."""
    after_rate = dataset["after_reading"]["undatable_share"]["rate"]
    before_rate = dataset["before_reading"]["undatable_share"]["rate"]

    assert _pct(before_rate) in report  # 4.47 %
    assert f"{after_rate * 100:.4f} %" in report  # 0.0368 %
    assert "13.6× inside" in report
    assert "MET" in report
    for ac in ("AC1", "AC2/AC4", "AC4 agreement", "AC5", "AC6", "AC7"):
        assert ac in report, f"the outcome table no longer reports {ac}"
    assert "slots-stale-postfix-window-2026-11-26-to-27.json" in report


def test_calendar_records_the_window_as_measured():
    """AC1: the calendar billed the window before it closed and now records the result."""
    calendar = CALENDAR_PATH.read_text()
    sections = calendar.split("### 3.4 Measured windows")
    assert len(sections) == 2, "measurement-windows.md lost its § 3.4 measured-windows table"

    measured_rows = [
        line for line in sections[1].splitlines() if line.startswith("| **") and WORK_ITEM in line
    ]
    assert measured_rows, f"§ 3.4 carries no measured row for {WORK_ITEM}"
    row = measured_rows[0]
    assert "2026-11-26T10:18:00Z → 2026-11-27T10:18:00Z" in row
    assert "slots-stale-postfix-window-2026-11-26-to-27.md" in row

    open_rows = [
        line
        for line in sections[0].splitlines()
        if line.startswith("| **") and WORK_ITEM in line and "Done" not in line
    ]
    assert not open_rows, f"{WORK_ITEM} is still billed as an open window in § 3"
