#!/usr/bin/env python3
"""Artefact contract for the post-fix admission→first-byte conversion record.

`LP-0MVV29YP6006PLNP` re-measures the 24 h post-fix abandoned-queue-wait share
after the occupant-hold blocker pricing (`bed25a46c`,
`LP-0MVV29YDQ00437N6`). The committed artefact pins the properties prose
cannot:

1. **the window is the re-anchored, re-verified window** — the nominal
   fix-anchored window was voided by mid-window redeploys and a backend
   crash-loop; the record measures the first clean 24 h
   (`2026-11-06 14:16:00` → `2026-11-07 14:16:00`, the zero-kill validated
   anchor from `LP-0MW3K7Z7T0094JK6`) and must not drift from those bounds;
2. **the totals decompose** over the per-band splits, so a hand-edited JSON is
   caught here rather than in an operator's analysis (AC2's per-band/trigger/
   cause claims, AC3's eviction counts, AC4's "no hand-typed numbers");
3. **the record's quoted numbers are the artefact's numbers** — the headline
   share, the triggers/causes/A3 eviction reasons, and the rate-bucket sanity
   gate must all be present in the committed JSON;
4. **the record's relative links resolve**, so a moved artefact or renamed page
   fails loudly instead of leaving a dead reference.

No production log, no network: the committed artefacts are read as data.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs" / "dev"
DOC_PATH = DOCS_DIR / "prefill-admission-conversion-postfix.md"
ARTEFACT_PATH = DOCS_DIR / "prefill-admission-conversion-postfix.json"
DIAGNOSIS_PATH = (
    DOCS_DIR / "queue-wait-diagnosis-postfix-2026-11-06-24h.json"
)

#: The re-anchored window: first whole minute after the 2026-11-06 14:15:52
#: restart (the zero-kill validated anchor, LP-0MW3K7Z7T0094JK6), 24 h long
#: and restart-free until 2026-11-10 14:18:30.
WINDOW_START = datetime(2026, 11, 6, 14, 16, 0)
WINDOW_END = datetime(2026, 11, 7, 14, 16, 0)
WINDOW_HOURS = 24

#: Headline population (committed artefact): 28 queued waits, 23 abandoned,
#: 5 dispatched → 82.1 %.
ENQUEUED = 28
ABANDONED = 23
DISPATCHED = 5


def _artefact() -> dict:
    return json.loads(ARTEFACT_PATH.read_text())


def _diagnosis() -> dict:
    return json.loads(DIAGNOSIS_PATH.read_text())


def _text() -> str:
    return DOC_PATH.read_text()


def test_window_is_the_re_anchored_clean_window():
    """The artefact measures exactly the re-anchored 24 h window."""
    document = _artefact()
    since = datetime.fromisoformat(document["since"])
    until = datetime.fromisoformat(document["until"])

    assert since == WINDOW_START, (
        f"window starts {since}, not the re-anchored instant {WINDOW_START} "
        "(the nominal fix-anchored window was void)"
    )
    assert until == WINDOW_END, (
        f"window ends {until}, expected {WINDOW_END}"
    )
    assert (until - since).total_seconds() == WINDOW_HOURS * 3600, (
        f"window is {(until - since).total_seconds() / 3600} h, not "
        f"{WINDOW_HOURS} h"
    )


def test_totals_decompose_over_bands_and_triggers():
    """Every published total is the sum the harness would recompute.

    Pins "no hand-typed numbers": the share is the harness's round(4) of
    abandoned / outcomes, and the totals are the band sums.
    """
    document = _artefact()
    totals = document["totals"]
    bands = document["bands"]

    assert totals["queued_waits"] == (
        totals["queue_dispatched"] + totals["queue_abandoned"]
    )
    assert totals["abandoned_share"] == round(
        totals["queue_abandoned"] / totals["queued_waits"], 4
    )

    for key in ("admit", "reject"):
        assert totals["admission_verdicts"][key] == sum(
            band["admission_verdicts"][key] for band in bands.values()
        ), f"total {key} verdicts do not equal the per-band sum"

    for total_key, band_key in (
        ("queue_dispatched", "queue_dispatched"),
        ("queue_abandoned", "queue_abandoned"),
        ("queued_waits", "queued_waits"),
        ("queue_wait_seconds", "queue_wait_seconds"),
        ("evictions", "evictions"),
    ):
        band_sum = sum(band[band_key] for band in bands.values())
        if total_key == "queue_wait_seconds":
            assert abs(totals[total_key] - band_sum) < 0.5, total_key
        else:
            assert totals[total_key] == band_sum, (
                f"{total_key} is {totals[total_key]}, bands sum to {band_sum}"
            )

    for split in ("queue_dispatched_by_trigger", "queue_abandoned_by_trigger"):
        per_band: dict[str, int] = {}
        for band in bands.values():
            for trigger, count in band[split].items():
                per_band[trigger] = per_band.get(trigger, 0) + count
        assert per_band == totals[split], f"{split} does not match the band sum"

    # The enqueue denominator is auditable: priced enqueues + unbanded ones.
    assert document["queued_waits_enqueued"] == (
        document["queued_waits_enqueued_unbanded"]
        + sum(band["queue_enqueued"] for band in bands.values())
    )
    assert document["queued_waits_enqueued"] >= totals["queued_waits"]


def test_contributing_archives_are_a_subset_of_the_scanned_inventory():
    """Provenance is intact: what contributed was actually scanned."""
    document = _artefact()
    contributing = document["log_files"]
    scanned = set(document["log_files_scanned"])

    assert contributing, "the artefact records no contributing log archive"
    assert set(contributing) <= scanned, (
        "contributing archives missing from log_files_scanned: "
        f"{sorted(set(contributing) - scanned)}"
    )
    assert document["queue_banding"], "the queue-wait banding provenance is gone"


def test_banding_is_all_own_evidence():
    """Post-fix log shape: every wait bands from its own outcome line."""
    document = _artefact()

    assert document["queue_banding"] == {
        "own_enqueue": 0,
        "own_outcome": ENQUEUED,
        "session_chain": 0,
        "unknown": 0,
    }
    assert document["queued_waits_enqueued_unbanded"] == 0


def test_headline_counts_are_the_artefacts_counts():
    """28 / 23 / 5 and the 82.1 % share are the artefact's, not hand-typed."""
    document = _artefact()
    totals = document["totals"]

    assert totals["queue_enqueued"] == ENQUEUED
    assert totals["queue_abandoned"] == ABANDONED
    assert totals["queue_dispatched"] == DISPATCHED
    assert totals["abandoned_share"] == 0.8214
    assert document["evictions"]["total"] == 80
    assert document["evictions"]["by_reason"] == {
        "lease_evicted": 1,
        "measured_cache_deficit": 40,
        "no_progress_watchdog": 3,
        "prefill_deadline": 36,
    }


def test_trigger_and_cause_splits_are_the_artefacts():
    """The record's per-trigger/per-cause rows match the committed JSON."""
    document = _artefact()
    totals = document["totals"]

    assert totals["queue_enqueued_by_trigger"] == {
        "concurrency_limit": 10,
        "local_lease_active": 18,
    }
    assert totals["queue_abandoned_by_trigger"] == {
        "concurrency_limit": 9,
        "local_lease_active": 14,
    }
    assert totals["queue_abandoned_by_cause"] == {"queue_wait_expired": 23}


def test_sanity_gate_rate_buckets_are_the_artefacts():
    """The window-validity gate rows are the artefact's per-band buckets."""
    document = _artefact()
    expected = {
        "<8k": {"contended": 642, "solitary": 2779, "solitary_fallback": 1},
        "8k-20k": {"contended": 462, "solitary": 115},
        "20k-32k": {"contended": 120, "solitary": 233},
        "32k-64k": {"solitary": 443},
        ">=64k": {"solitary": 378},
    }
    for band, buckets in expected.items():
        assert document["bands"][band]["rate_buckets"] == buckets, band


def test_diagnosis_artefact_matches_the_conversion_record():
    """The per-wait diagnosis agrees with the conversion totals."""

    summary = _diagnosis()["summary"]

    assert summary["queue_waits"] == ENQUEUED
    assert summary["by_outcome"] == {
        "abandoned": ABANDONED,
        "dispatched": DISPATCHED,
    }
    assert summary["by_trigger"] == {
        "concurrency_limit": 10,
        "local_lease_active": 18,
    }
    assert summary["correlation"]["own_request"] == ENQUEUED
    assert summary["abandoned_by_cause"]["queue_wait_expired"] == ABANDONED
    assert summary["blocker"]["abandoned_by_class"] == {
        "lease_holder_generating": 7,
        "prefill_in_flight": 16,
        "unknown": 0,
    }


def test_document_quotes_the_committed_share_and_counts():
    """The doc quotes the artefact's own numbers (AC2/AC3/AC4)."""
    text = _text()

    assert "82.1 %" in text and "(23/28)" in text
    assert "60.7 %" in text and "(314/517)" in text
    assert "52.6 %" in text and "(71/135)" in text
    assert "85 %" in text
    assert "queue_wait_expired: 23" in text
    assert "2031" not in text  # no stray hand-typed figure


def test_document_links_resolve():
    """Every relative doc/artefact link in the record points at a real file."""
    text = _text()
    links = re.findall(r"\]\(([^)#]+?)\)", text)
    relative = [
        link for link in links if not link.startswith(("http://", "https://"))
    ]

    assert relative, "the record no longer links any doc or artefact"
    missing = [
        link for link in relative if not (DOCS_DIR / link).resolve().exists()
    ]
    assert missing == []


def test_document_embeds_the_regenerating_commands():
    """AC4: the command that regenerates the committed JSON is in the record."""
    text = _text()

    for path in (ARTEFACT_PATH, DIAGNOSIS_PATH):
        assert path.name in text, f"{path.name} is not referenced by the record"
    assert "scripts/prefill_admission_conversion.py" in text
    assert '--since "2026-11-06 14:16:00"' in text
    assert '--until "2026-11-07 14:16:00"' in text
    assert "scripts/queue_wait_diagnosis.py" in text


def test_artefact_carries_its_provenance():
    for key in ("since", "until", "log_dir", "log_files", "totals", "bands"):
        assert key in _artefact()
