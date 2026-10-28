"""Doc-consistency tests for the pool-4 go/no-go decision record.

Item: LP-0MVOG8U4D0026YCL (C5, parent LP-0MVN3KPEC007WWL5). The deliverable is
``docs/slot-snapshot-pool-evaluation-LP-0MVN3KPEC007WWL5.md`` — the decision
record for widening the session-slot snapshot pool 2 → 4. Parent AC2 requires
the verdict to be evidence-backed and reproducible; parent AC4 requires the
measured blockers to be recorded; parent AC5 requires the record to assert that
the live slot count and per-slot context stay unchanged.

The record is documentation, so these tests are content contracts against the
two committed, machine-readable evidence artefacts the record must reproduce:

* C3 ``proxy/benchmarks/results/slot_snapshot_bands_20261028/summary.json`` —
  per-band save/restore cost (unloaded + under load), disk footprint, restore
  effectiveness;
* C4 ``proxy/benchmarks/results/pool_bakeoff_2026_10_24/summary.json`` — the
  bake-off arms' reuse metrics, restore success, exclusions and gate verdicts.

If a number is edited in the record without the artefact (or vice versa), or the
record's verdict stops matching the gates, these fail — the decision must never
silently drift from the evidence it claims to rest on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

#: Repo root (``proxy/tests/`` → ``proxy/`` → repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]

DOC_PATH = REPO_ROOT / "docs/slot-snapshot-pool-evaluation-LP-0MVN3KPEC007WWL5.md"
C3_SUMMARY = (
    REPO_ROOT
    / "proxy/benchmarks/results/slot_snapshot_bands_20261028/summary.json"
)
C4_SUMMARY = (
    REPO_ROOT / "proxy/benchmarks/results/pool_bakeoff_2026_10_24/summary.json"
)

#: The C2 §4.3 go/no-go thresholds the record must state numerically
#: (G1: reuse uplift; G2: restore success vs suppression ceiling).
G1_MEDIAN_PCT = 31.2
G1_AGGREGATE_PCT = 36.1
G2_RESTORE_SUCCESS_PCT = 90.0
G2_EXCLUSION_CEILING_PCT = 20.0

#: Alternative owning items the record must link (decision traceability).
ALTERNATIVE_LINKS = (
    "LP-0MVGD9ZKI004ETTE",  # virtual slots (candidate C)
    "LP-0MVGCQ3QN009HRE8",  # slot-count re-evaluation
    "LP-0MVN1IQTF0037WTG",  # tenancy / resident-prefix scheduling
)


def _doc_text() -> str:
    assert DOC_PATH.is_file(), f"evaluation doc missing: {DOC_PATH}"
    return DOC_PATH.read_text(encoding="utf-8")


def _c3() -> dict:
    return json.loads(C3_SUMMARY.read_text(encoding="utf-8"))


def _c4() -> dict:
    return json.loads(C4_SUMMARY.read_text(encoding="utf-8"))


def _assert_in_doc(text: str, needle: str, message: str) -> None:
    assert needle in text, f"{message}\nMissing from {DOC_PATH.name}: {needle!r}"


# ---------------------------------------------------------------------------
# 1. The record exists, carries an explicit verdict, and the same verdict the
#    C4 gates produce.
# ---------------------------------------------------------------------------


def test_record_exists_with_explicit_verdict():
    doc = _doc_text()
    _assert_in_doc(
        doc,
        "NO-GO",
        "Parent AC2 requires an explicit GO or NO-GO verdict in the record.",
    )


def test_record_verdict_matches_c4_gates():
    """The record's verdict must agree with C4's gate verdicts for arm1.

    C4 measured G1 and G2 failing with G3–G6 passing; the record must not
    contradict that (a no-go is forced by G1∪G2 alone).
    """
    c4 = _c4()
    arm1 = c4["arms"]["arm1"]
    assert arm1["gates"]["G1"]["pass"] is False
    assert arm1["gates"]["G2"]["pass"] is False
    for gate in ("G3", "G4", "G5", "G6"):
        assert arm1["gates"][gate]["pass"] is True, f"{gate} must pass on arm1"
    doc = _doc_text()
    _assert_in_doc(
        doc,
        "G1",
        "The record must cite the failing gates (G1 reuse uplift, G2 suppression).",
    )
    _assert_in_doc(
        doc,
        "G2",
        "The record must cite the failing gates (G1 reuse uplift, G2 suppression).",
    )


# ---------------------------------------------------------------------------
# 2. C3 evidence table: per-band save/restore cost reproduces summary.json.
# ---------------------------------------------------------------------------

#: (band, prompt_total_tokens, unloaded save s, unloaded restore s, under-load
#: save s, under-load restore s) — the record's band table must agree with C3.
C3_BAND_ROWS = (
    ("10k", 9942, 0.034, 0.021, 0.049, 0.04),
    ("25k", 24942, 0.052, 0.032, 0.072, 0.052),
    ("40k", 39942, 0.095, 0.049, 0.111, 0.067),
    ("60k+", 59942, 0.125, 0.071, 0.291, 0.089),
)


def test_c3_band_numbers_match_artefact():
    """The record's save/restore numbers equal the C3 summary (rounded to 3 dp)."""
    c3 = _c3()
    for band, tokens, save_u, restore_u, save_l, restore_l in C3_BAND_ROWS:
        band_row = next(b for b in c3["bands"] if b["band"] == band)
        assert band_row["prompt_total_tokens"] == tokens
        assert band_row["unloaded"]["save_seconds"] == pytest.approx(save_u, abs=0.0005)
        assert band_row["unloaded"]["restore_seconds"] == pytest.approx(
            restore_u, abs=0.0005
        )
        assert band_row["under_load"]["save_seconds"] == pytest.approx(save_l, abs=0.0005)
        assert band_row["under_load"]["restore_seconds"] == pytest.approx(
            restore_l, abs=0.0005
        )
    doc = _doc_text()
    for band, _, _, _, _, _ in C3_BAND_ROWS:
        _assert_in_doc(
            doc,
            band,
            f"The record's C3 evidence table must contain band {band!r}.",
        )


def test_c3_zero_failures_are_stated():
    """The record must carry C3's zero-timeout/zero-failure totals (parent AC4:
    latency is *not* a blocker)."""
    c3 = _c3()
    assert c3["totals"]["timeout_exceeded_operations"] == 0
    assert c3["totals"]["failed_operations"] == 0
    doc = _doc_text()
    _assert_in_doc(
        doc,
        "0 timeout-exceeded",
        "Record must state C3's zero timeout-exceeded save/restore operations.",
    )
    _assert_in_doc(
        doc,
        "0 failed",
        "Record must state C3's zero failed save/restore operations.",
    )


def test_c3_restore_cost_within_adaptive_timeout():
    """Parent AC1 / C3: restore cost must be inside the adaptive timeout at every
    band — the record must state that the mechanism is not latency-bound."""
    c3 = _c3()
    for band_row in c3["bands"]:
        restore = band_row["under_load"]["restore_seconds"]
        assert restore < band_row["timeout_seconds"], (
            f"restore {band_row['band']} exceeds its adaptive timeout"
        )
    doc = _doc_text()
    _assert_in_doc(
        doc,
        "0.02–0.09 s",
        "Record must cite the measured restore-cost range 0.02–0.09 s.",
    )


# ---------------------------------------------------------------------------
# 3. C4 evidence table: bake-off arms and gate verdicts reproduce summary.json.
# ---------------------------------------------------------------------------


def test_c4_arm_numbers_match_artefact():
    """Calibrated aggregate / median >20k reuse for the comparable arms."""
    c4 = _c4()
    arm0, arm1 = c4["arms"]["arm0"], c4["arms"]["arm1"]
    assert arm0["config"]["comparable"] is True
    assert arm1["config"]["comparable"] is True
    doc = _doc_text()
    for label, arm in (("arm0", arm0), ("arm1", arm1)):
        agg = arm["calibrated"]["aggregate_pct"]
        med = arm["calibrated"]["median_pct_gt20k"]
        # Record must carry the reuse numbers for both comparable arms.
        _assert_in_doc(
            doc,
            f"{agg}",
            f"Record must reproduce {label} calibrated aggregate reuse {agg} %.",
        )
        _assert_in_doc(
            doc,
            f"{med}",
            f"Record must reproduce {label} calibrated median >20k reuse {med} %.",
        )


def test_c4_g1_thresholds_state_break_even_numerically():
    """G1's numeric go-thresholds must appear in the record (the break-even
    condition of parent AC2: reuse uplift must at least double the >20k median
    and add 10 pp aggregate)."""
    doc = _doc_text()
    _assert_in_doc(
        doc,
        f">= {G1_MEDIAN_PCT} %",
        "Record must state the G1 median go-threshold 31.2 % numerically.",
    )
    _assert_in_doc(
        doc,
        f">= {G1_AGGREGATE_PCT} %",
        "Record must state the G1 aggregate go-threshold 36.1 % numerically.",
    )


def test_c4_g2_suppression_share_matches_artefact():
    """G2's suppression share (skip_when_busy + cooldown over persistence
    decisions) must reproduce the 56.1 % C4 measured and the 20 % ceiling."""
    c4 = _c4()
    arm1 = c4["arms"]["arm1"]["counters"]
    attempts = arm1["attempts"]
    busy = arm1["excluded_slot_busy"]
    cooldown = arm1["excluded_cooldown"]
    share = (busy + cooldown) / attempts
    assert share == pytest.approx(0.5611, abs=0.0005)
    doc = _doc_text()
    _assert_in_doc(
        doc,
        "56.1 %",
        "Record must state the measured skip_when_busy suppression share 56.1 %.",
    )
    _assert_in_doc(
        doc,
        f"<= {G2_EXCLUSION_CEILING_PCT} %",
        "Record must state the G2 suppression ceiling 20 % numerically.",
    )


def test_c4_restore_success_is_not_a_blocker():
    """Parent AC4: record the measured restore-success — it must reproduce C4's
    100 % success (i.e. restore *failures* are not the blocker)."""
    c4 = _c4()
    for arm_name in ("arm0", "arm1"):
        arm = c4["arms"][arm_name]
        assert arm["restore_success_rate"] == 1.0
    doc = _doc_text()
    _assert_in_doc(
        doc,
        "100 %",
        "Record must state the measured restore-success rate (100 %).",
    )


def test_c4_candidate_a_is_called_out_as_ac5_supersession():
    """Parent AC5: candidate A changes live slots / per-slot ctx, so the record
    must call it out as an AC5 supersession needing producer sign-off, never a
    silent change."""
    c4 = _c4()
    arm2 = c4["arms"]["arm2"]
    assert arm2["config"]["live_slots"] == 4
    assert arm2["config"]["comparable"] is False
    doc = _doc_text()
    _assert_in_doc(
        doc,
        "AC5",
        "Record must assert the AC5 invariant explicitly.",
    )
    _assert_in_doc(
        doc,
        "supersession",
        "Any proposed live-geometry change must be called out as an AC5 "
        "supersession requiring producer sign-off.",
    )


def test_record_links_alternative_owning_items():
    """The record must trace to the owning items of the alternatives."""
    doc = _doc_text()
    for item_id in ALTERNATIVE_LINKS:
        _assert_in_doc(
            doc,
            item_id,
            f"Record must link owning alternative item {item_id}.",
        )


def test_ac5_invariant_asserted():
    """Parent AC5: the record must explicitly state that the decision keeps the
    live slot count (2) and per-slot context (clamp 126,976) unchanged."""
    doc = _doc_text()
    _assert_in_doc(
        doc,
        "126,976",
        "Record must state the per-slot context clamp (126,976) it preserves.",
    )
    _assert_in_doc(
        doc,
        "2",
        "Record must state the live slot count (2) it preserves.",
    )


def test_no_go_blocks_c6_c7():
    """The record must state that C6 (retention/eviction) and C7
    (observability) remain blocked on a no-go (they are not unblocked with the
    required spec)."""
    doc = _doc_text()
    _assert_in_doc(
        doc,
        "LP-0MVOG8UE60050SKA",
        "Record must name C6 (retention/eviction item) as remaining blocked.",
    )
    _assert_in_doc(
        doc,
        "LP-0MVOG8UNX003Z9WN",
        "Record must name C7 (observability item) as remaining blocked.",
    )
