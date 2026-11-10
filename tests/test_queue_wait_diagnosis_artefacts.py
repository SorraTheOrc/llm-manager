#!/usr/bin/env python3
"""Artefact contract for the per-wait queue-abandonment diagnosis record.

`LP-0MVV29XPT003OXI5` commits a diagnosis record plus the two JSON artefacts it
quotes. Prose alone cannot pin the properties that make those numbers usable, so
they are asserted here instead:

1. **the windows are the deploy boundary pair** — a window that starts or ends
   elsewhere is not diurnally matched to the post-change record and is not the
   population the parent item asks about;
2. **the totals decompose** over the outcome split and the per-trigger split, so
   a hand-edited JSON is caught here rather than in an operator's analysis;
3. **the record's quoted numbers are the artefact's numbers** — the headline
   counts, the phantom-band reproduction and the correlation counts must all be
   present in the committed JSON, so the page cannot drift from its evidence;
4. **the record's relative links resolve**, so a moved artefact or renamed page
   fails loudly instead of leaving a dead reference.

No production log, no network: the committed artefacts are read as data.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs" / "dev"
DOC_PATH = DOCS_DIR / "queue-wait-abandonment-diagnosis.md"
ARTEFACT_PATH = DOCS_DIR / "queue-wait-abandonment-diagnosis.json"
BASELINE_PATH = (
    DOCS_DIR / "queue-wait-abandonment-diagnosis-diurnal-baseline.json"
)

#: ``52085f09`` started serving at this instant; the post-change window starts
#: here and the diurnally matched pre-change window ends here.
DEPLOY_AT = datetime(2026, 10, 29, 0, 10, 18)
POST_WINDOW_START = DEPLOY_AT
POST_WINDOW_END = datetime(2026, 10, 30, 0, 10, 18)
PRE_WINDOW_START = datetime(2026, 10, 28, 0, 10, 18)

#: The population the parent item reports for this window (committed
#: ``prefill-admission-conversion-postchange.json``): 517 enqueued, 203
#: dispatched, 314 abandoned.
POST_ENQUEUED = 517
POST_DISPATCHED = 203
POST_ABANDONED = 314


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _document() -> dict:
    return _load(ARTEFACT_PATH)


def _baseline() -> dict:
    return _load(BASELINE_PATH)


def _text() -> str:
    return DOC_PATH.read_text()


def test_post_window_is_the_deploy_anchored_window():
    """The primary artefact measures the post-change window and nothing else."""
    document = _document()

    assert datetime.fromisoformat(document["since"]) == POST_WINDOW_START
    assert datetime.fromisoformat(document["until"]) == POST_WINDOW_END


def test_baseline_window_is_diurnally_matched_to_the_deploy():
    """The comparator ends exactly at the deploy, so it is pre-change code."""
    baseline = _baseline()

    assert datetime.fromisoformat(baseline["since"]) == PRE_WINDOW_START
    assert datetime.fromisoformat(baseline["until"]) == DEPLOY_AT


def test_post_window_totals_match_the_committed_conversion_record():
    """The rows are the same population the post-change record counts."""
    summary = _document()["summary"]

    assert summary["queue_waits"] == POST_ENQUEUED
    assert summary["by_outcome"] == {
        "abandoned": POST_ABANDONED,
        "dispatched": POST_DISPATCHED,
    }
    assert len(_document()["rows"]) == POST_ENQUEUED


def test_totals_decompose_over_the_split_dimensions():
    """Outcome, trigger and blocker splits each add up to the total."""
    summary = _document()["summary"]

    assert sum(summary["by_outcome"].values()) == summary["queue_waits"]
    assert sum(summary["by_trigger"].values()) == summary["queue_waits"]
    assert sum(summary["abandoned_by_trigger"].values()) == POST_ABANDONED
    assert (
        sum(summary["blocker"]["abandoned_by_class"].values())
        == POST_ABANDONED
    )
    for bucket in summary["bands"].values():
        assert bucket["queued"] == (
            bucket["dispatched"] + bucket["abandoned"] + bucket["no_outcome"]
        )
    for trigger, split in summary["blocker"]["abandoned_by_class_and_trigger"].items():
        assert sum(split.values()) == summary["abandoned_by_trigger"][trigger]


def test_correlation_never_silently_joins_a_mismatch():
    """Every joined row is accounted for; mismatches carry no guard at all."""
    document = _document()
    correlation = document["summary"]["correlation"]

    assert sum(
        correlation[key]
        for key in ("own_request", "session_preceding", "mismatched", "none")
    ) == document["summary"]["queue_waits"]
    for row in document["rows"]:
        if row["correlation"] == "mismatched":
            assert row["guard"] is None
            assert row["ambiguous"] is True
        elif row["correlation"] == "own_request":
            assert row["ambiguous"] is False
            assert row["guard"] is not None
        elif row["correlation"] == "session_preceding":
            assert row["ambiguous"] is True
            assert row["guard"] is not None
        else:
            assert row["guard"] is None
            assert row["ambiguous"] is False


def test_headline_numbers_are_the_artefacts_numbers():
    """The record's headline counts are present in the committed JSON."""
    summary = _document()["summary"]
    abandoned = summary["abandoned_guard"]

    assert summary["abandoned_correlation"]["session_preceding"] == 169
    assert summary["abandoned_correlation"]["none"] == 145
    assert abandoned["with_guard"] == 169
    assert abandoned["admits"] == 163
    assert abandoned["admits_below_budget"] == 163
    assert abandoned["median_projected_s"] == 28.4
    assert abandoned["median_observed_s"] == 60.0
    assert abandoned["observed_minus_projected_s"]["median"] == 31.6
    assert abandoned["occupancy_is_symmetric_floor"] == 163
    assert abandoned["occupancy_raised_by_measurement"] == 6
    assert abandoned["projected_buckets"]["<20"] == 75
    assert summary["blocker"]["abandoned_by_class"] == {
        "lease_holder_generating": 42,
        "prefill_in_flight": 205,
        "unknown": 67,
    }
    assert summary["dispatched_guard"]["median_projected_s"] == 21.95
    assert summary["dispatched_guard"]["median_observed_s"] == 28.33


def test_phantom_band_is_reproduced_and_resolved():
    """AC3(c): the pre-fix rule reproduces 45/41 in ``32k-64k``; the fix empties it."""
    summary = _document()["summary"]

    assert summary["legacy_bands"]["32k-64k"] == {
        "queued": 45,
        "dispatched": 4,
        "abandoned": 41,
        "no_outcome": 0,
    }
    assert "32k-64k" not in summary["bands"]
    assert ">=64k" not in summary["bands"]
    assert summary["rebanded"] == 98
    assert summary["rebanded_abandoned"] == 78
    # Every re-banded wait is accounted for: no wait is banded from a *later*
    # admission, so the phantom band cannot reappear.
    moved = sum(
        1
        for row in _document()["rows"]
        if row["legacy_band"] != row["band"]
    )
    assert moved == summary["rebanded"]


def test_baseline_artefact_reproduces_the_committed_diurnal_share():
    """The surviving-slice comparator matches the committed baseline record."""
    summary = _baseline()["summary"]

    assert summary["by_outcome"] == {
        "abandoned": 36,
        "dispatched": 31,
        "no_outcome": 1,
    }
    outcomes = summary["by_outcome"]["abandoned"] + summary["by_outcome"]["dispatched"]
    assert outcomes == 67
    assert round(summary["by_outcome"]["abandoned"] / outcomes, 4) == 0.5373


def test_baseline_artefact_records_its_incomplete_coverage():
    """Retention pruned the window's start; the artefact must say so."""
    coverage = _baseline()["coverage"]

    assert coverage["covered"] is False
    assert coverage["window_start_gap_seconds"] > 0
    assert len(coverage["files"]) == 2
    assert coverage["gaps"], "the internal retention gap must be recorded"


def test_post_artefact_coverage_is_complete():
    coverage = _document()["coverage"]

    assert coverage["covered"] is True
    assert coverage["gaps"] == []
    assert coverage["window_start_gap_seconds"] <= 1.0
    assert coverage["window_end_gap_seconds"] == 0.0


def test_document_quotes_the_share_and_the_phantom_row():
    text = _text()

    assert "60.7" in text and "(314/517)" in text
    assert "53.7" in text and "(36/67)" in text
    assert "45 queued / 41 abandoned" in text
    assert "163 of 169" in text
    assert "46.2 %" in text
    assert "13.9 %" in text


def test_document_relative_links_resolve():
    """Every relative markdown link in the record points at a real file."""
    text = _text()
    links = re.findall(r"\]\(([^)#]+?)\)", text)
    relative = [link for link in links if not link.startswith(("http://", "https://"))]

    assert relative, "the record must link its artefacts"
    missing = [link for link in relative if not (DOCS_DIR / link).resolve().exists()]
    assert missing == []


def test_document_embeds_the_regenerating_commands():
    """AC5: the command that regenerates each committed JSON is in the record."""
    text = _text()

    for path in (ARTEFACT_PATH, BASELINE_PATH):
        assert path.name in text, f"{path.name} is not referenced by the record"
    assert "scripts/queue_wait_diagnosis.py" in text
    assert '--since "2026-10-29 00:10:18"' in text
    assert '--since "2026-10-28 00:10:18"' in text


@pytest.mark.parametrize(
    "key", ["blocker_lookback_s", "since", "until", "log_dir", "rows", "summary"]
)
def test_artefact_carries_its_provenance(key):
    assert key in _document()
