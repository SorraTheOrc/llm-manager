"""Pool-size / snapshot-restore bake-off replay tests (LP-0MVOG8TUM005705R).

Parent AC2 asks whether widening the session-slot snapshot pool (2 -> 4)
raises prefix reuse on the measured 2026-10-24 thrash window. C2
(`docs/dev/slot-snapshot-pool-sizing-LP-0MVN3KPEC007WWL5.md` §3/§4) defines
the candidate configurations (candidate B: decoupled retained-snapshot pool
of 4; candidate A: literal pool-4 geometry, non-comparable) and the bake-off
protocol (metrics, gates G1-G6). This module pins the offline replay that
answers AC2:

* :mod:`pool_snapshot_bakeoff` models the snapshot pool (live slots + retained
  snapshots, LRU/disk-capped) over the checked-in fixture, driving the **real**
  production predicates (slot assignment, busy gate, circuit breaker, context
  cap) with the same documented stub sources the residency tests use.
* Restore/save cost is pinned from C3's committed measurement
  (`proxy/benchmarks/results/slot_snapshot_bands_20261028/summary.json`).
* Percentiles are delegated to ``scripts/analyse_slot_ratio_restore._percentiles``
  so the reuse uplift uses the usage report's percentile definition, never an
  ad-hoc one.

AC mapping: pool-size/snapshot model + gates (AC1), pool-4 replay reporting
uplift and restore-success rate (AC2), usage-report percentile definitions
(AC3), per-config exclusion counts (AC4), determinism + boundary + no-
production-import (AC5), committed JSON artefact (AC6).
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
from pathlib import Path

import pool_snapshot_bakeoff as bakeoff
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSE_SCRIPT = REPO_ROOT / "scripts" / "analyse_slot_ratio_restore.py"
COMMITTED_ARTEFACT = (
    REPO_ROOT
    / "proxy"
    / "benchmarks"
    / "results"
    / "pool_bakeoff_2026_10_24"
    / "summary.json"
)

#: The measured baseline this bake-off must reproduce (AC3 reference).
MEASURED_AGGREGATE_PCT = 26.1
MEASURED_MEDIAN_PCT = 15.6
MEASURED_COLD_OF = (28, 227)

LARGE_PROMPT = 20_000


def _load_analyse_script():
    spec = importlib.util.spec_from_file_location(
        "analyse_slot_ratio_restore_bakeoff", ANALYSE_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyse = _load_analyse_script()


@pytest.fixture(scope="module")
def fixture_data():
    """The deduped 2026-10-24 window (1016 unique admission attempts)."""
    _, records = bakeoff.load_records()
    return bakeoff.unique_attempts(records)


@pytest.fixture
def clean_production_state():
    """Reset the production-side global registries the sim drives."""
    import proxy.session as session_mod

    session_mod._slot_owners.clear()
    session_mod._slot_failure_state.clear()
    yield
    session_mod._slot_owners.clear()
    session_mod._slot_failure_state.clear()


# ---------------------------------------------------------------------------
# Fixture + arms (AC1/AC2 shape)
# ---------------------------------------------------------------------------


def test_fixture_dedupes_duplicate_admission_records(fixture_data):
    """The capture logs each attempt twice (queue + large_context_bypass)."""
    assert len(fixture_data) == 1016
    assert len(fixture_data) < 1978
    keys = {(r["ts"], r["session"], r["estimated_tokens"]) for r in fixture_data}
    assert len(keys) == len(fixture_data)
    for record in fixture_data:
        assert 0 < record["estimated_tokens"]
        assert 0 <= record["cached_ratio"] <= 1.0


def test_arms_defined_per_c2_protocol():
    """The four arms match C2 §3/§4: clamps from the production helper."""
    arm0 = bakeoff.ARMS["arm0"]
    arm1 = bakeoff.ARMS["arm1"]
    arm2 = bakeoff.ARMS["arm2"]
    arm3 = bakeoff.ARMS["arm3"]

    assert (arm0.live_slots, arm0.retained_snapshot_pool) == (2, None)
    assert (arm1.live_slots, arm1.retained_snapshot_pool) == (2, 4)
    assert (arm2.live_slots, arm2.retained_snapshot_pool) == (4, None)
    assert (arm3.live_slots, arm3.retained_snapshot_pool) == (2, 4)

    # C2 §2 table: pool 2 clamp 126,976, pool 4 clamp 61,440 — produced by
    # the same production helper, not hand-computed.
    assert arm0.clamp == bakeoff.effective_per_slot_clamp(2) == 126_976
    assert arm2.clamp == bakeoff.effective_per_slot_clamp(4) == 61_440
    assert arm0.clamp > arm2.clamp

    # Gating / comparability per C2 §4.1 (candidate A is NOT a bake-off arm).
    assert arm0.skip_when_busy and arm1.skip_when_busy and arm2.skip_when_busy
    assert arm3.skip_when_busy is False
    assert arm0.comparable and arm1.comparable
    assert not arm2.comparable and not arm3.comparable


# ---------------------------------------------------------------------------
# Determinism + replay shape (AC5)
# ---------------------------------------------------------------------------


def test_replay_is_deterministic(fixture_data, clean_production_state):
    first = bakeoff.bakeoff(fixture_data)
    second = bakeoff.bakeoff(fixture_data)
    assert bakeoff.bakeoff_result_to_json(first) == bakeoff.bakeoff_result_to_json(
        second
    )


def test_all_arms_report_metric_and_exclusion_fields(fixture_data, clean_production_state):
    report = bakeoff.bakeoff(fixture_data)
    assert set(report["arms"]) == {"arm0", "arm1", "arm2", "arm3"}
    for arm in report["arms"].values():
        assert arm["config"]["live_slots"] in (2, 4)
        assert "exclusions" in arm
        for category in (
            "context_too_large",
            "slot_busy",
            "cooldown",
            "timeout_exceeded",
            "pool_exhausted",
        ):
            assert category in arm["exclusions"]
            assert arm["exclusions"][category] >= 0
        assert arm["exclusions"]["total"] == sum(
            arm["exclusions"][c]
            for c in ("context_too_large", "slot_busy", "cooldown", "timeout_exceeded")
        )
        assert arm["save_skips"]["total"] == sum(
            arm["save_skips"][k] for k in ("slot_busy", "cooldown", "timeout_exceeded")
        )
        assert 0.0 <= arm["restore_success_rate"] <= 1.0


def test_gate_verdicts_present_for_comparable_arms(fixture_data, clean_production_state):
    report = bakeoff.bakeoff(fixture_data)
    for name in ("arm0", "arm1"):
        gates = report["arms"][name]["gates"]
        for gate in ("G1", "G2", "G3", "G4", "G5", "G6"):
            assert gate in gates
            assert gates[gate]["pass"] in (True, False)
            assert gates[gate]["rationale"]
    # Candidate A is not a bake-off arm: gates are n/a, not fabricated.
    assert report["arms"]["arm2"]["gates"]["G1"]["pass"] == "n/a"


# ---------------------------------------------------------------------------
# Pool model boundaries (AC5)
# ---------------------------------------------------------------------------


def test_retained_pool_cap_evicts_least_recently_used(fixture_data, clean_production_state):
    """Capped retained pool keeps the 4 most-recent sessions, drops the rest."""
    pool = bakeoff.RetentionPool(cap_count=4, cap_bytes=None)
    for index in range(6):
        pool.upsert(f"session-{index}", bakeoff.Snapshot(tokens=10_000, last_active=float(index)))
    assert sorted(pool.session_ids()) == ["session-2", "session-3", "session-4", "session-5"]
    assert "session-0" not in pool and "session-1" not in pool
    assert len(pool) == 4
    assert pool.get("session-4").tokens == 10_000


def test_pool_larger_than_live_slots_keeps_extra_snapshots(fixture_data, clean_production_state):
    """Retained pool of 4 > 2 live slots: extra sessions stay restorable."""
    sim = bakeoff.SnapshotPoolSim(bakeoff.ARMS["arm1"])
    sim.simulate(
        _craft_attempts(
            ("s-a", 15_000),
            ("s-b", 15_000),
            ("s-c", 15_000),
            ("s-d", 15_000),
            ("s-e", 15_000),
            gap_seconds=60.0,  # turns do not overlap; eviction saves are allowed
        )
    )
    assert sim.arm.live_slots == 2
    # Four sessions keep a restorable snapshot even though only two slots exist
    # — the boundary case C2 §3 candidate B depends on.
    assert len(sim.retained.session_ids()) == 4
    assert sim.retained.get("s-c") is not None
    # A pool of 2 would keep only the two most recently saved.
    tight = bakeoff.SnapshotPoolSim(dataclasses.replace(bakeoff.ARMS["arm1"], retained_snapshot_pool=2))
    tight.simulate(
        _craft_attempts(
            ("s-a", 15_000),
            ("s-b", 15_000),
            ("s-c", 15_000),
            ("s-d", 15_000),
            ("s-e", 15_000),
            gap_seconds=60.0,
        )
    )
    assert sorted(tight.retained.session_ids()) == ["s-d", "s-e"]


def test_disk_cap_bounds_retention_by_measured_bytes(fixture_data, clean_production_state):
    """The 20 GiB cleanup cap bounds retention via C3's bytes-per-token."""
    pool = bakeoff.RetentionPool(cap_count=None, cap_bytes=20 * 1024)
    for index in range(40):
        pool.upsert(f"session-{index}", bakeoff.Snapshot(tokens=10_000, last_active=float(index)))
    assert pool.byte_usage_mib <= 20 * 1024
    # Candidate B's count cap of 4 is far tighter than the byte cap, so it is
    # the effective bound there.
    capped = bakeoff.RetentionPool(cap_count=4, cap_bytes=20 * 1024)
    for index in range(40):
        capped.upsert(
            f"session-{index}", bakeoff.Snapshot(tokens=10_000, last_active=float(index))
        )
    assert len(capped.session_ids()) == 4


# ---------------------------------------------------------------------------
# Persistence gates via the real production predicates (AC1/AC4)
# ---------------------------------------------------------------------------


def test_skip_when_busy_exclusions_are_counted(fixture_data, clean_production_state):
    """A request while another session's serve is in flight skips save/restore."""
    sim = bakeoff.SnapshotPoolSim(bakeoff.ARMS["arm1"])
    sim.simulate(
        _craft_attempts(
            ("s-x", 10_000),  # populates slot 0 and saves its snapshot
            ("s-a", 100_000, 100_000),  # long prefill: occupies its slot
            ("s-b", 10_000),  # starts while s-a is still serving
            ("s-x", 10_000),  # returns while s-a is still serving
        )
    )
    # Two persistence decisions suppressed by the busy gate (s-b and s-x),
    # plus the eviction save of s-b's KV — production behaviour, and the
    # attribution the AC requires.
    assert sim.counters["excluded_slot_busy"] >= 1, sim.counters
    assert sim.counters["save_skipped_slot_busy"] >= 1, sim.counters
    # The same corpus with the busy gate off performs the save and the restore.
    relaxed = bakeoff.SnapshotPoolSim(bakeoff.ARMS["arm3"])
    relaxed.simulate(
        _craft_attempts(
            ("s-x", 10_000),
            ("s-a", 100_000, 100_000),
            ("s-b", 10_000),
            ("s-x", 10_000),
        )
    )
    assert relaxed.counters["excluded_slot_busy"] == 0
    assert relaxed.counters["save_skipped_slot_busy"] == 0
    assert relaxed.counters["restore_opportunities"] >= 1
    assert relaxed.counters["restore_success"] >= 1


def test_circuit_breaker_cooldown_blocks_then_recovers(fixture_data, clean_production_state):
    """After N consecutive failures the slot cools down, then recovers."""
    sim = bakeoff.SnapshotPoolSim(
        bakeoff.ARMS["arm1"], fail_restores_every=1, fail_saves_every=1
    )
    # a: cold; save a fails (#1). b: evict-save a fails (#2), save b fails (#3)
    # => the slot's breaker trips (3 failures in the 300 s window).
    sim.simulate(_craft_attempts(("s-a", 10_000), ("s-b", 10_000), ("s-c", 10_000)))
    assert sim.counters["save_failed"] >= 2
    sim.simulate(_craft_attempts(("s-a", 10_000), ("s-b", 10_000)))
    # The breaker suppresses persistence for the slot until the cooldown ends.
    assert sim.counters["save_skipped_cooldown"] >= 1, sim.counters
    assert sim.counters["excluded_cooldown"] + sim.counters["save_skipped_cooldown"] >= 2
    # After the cooldown window elapses the breaker resets and work resumes.
    before = sim.counters["save_skipped_cooldown"]
    sim.simulate(_craft_attempts(("s-a", 10_000), start_seconds=400.0))
    assert sim.counters["save_skipped_cooldown"] == before
    assert sim.counters["save_failed"] >= 1  # persisted again (and fails by injection)


def test_context_too_large_exclusions_grow_when_clamp_shrinks(fixture_data, clean_production_state):
    """Candidate A's 61,440 clamp excludes far more restores than 126,976."""
    arm0 = bakeoff.run_arm(bakeoff.ARMS["arm0"], fixture_data)
    arm2 = bakeoff.run_arm(bakeoff.ARMS["arm2"], fixture_data)
    assert arm0["exclusions"]["context_too_large"] < arm2["exclusions"][
        "context_too_large"
    ]
    assert arm0["exclusions"]["context_too_large"] >= 0


def test_timeout_exceeded_is_counted_when_restore_exceeds_budget(
    fixture_data, clean_production_state
):
    """A restored prompt whose measured cost exceeds the adaptive timeout."""
    sim = bakeoff.SnapshotPoolSim(bakeoff.ARMS["arm1"])
    sim.simulate(
        _craft_attempts(
            ("s-a", bakeoff.SLOT_CTX),
            ("s-b", 10_000),
            ("s-a", 10_000),
        )
    )
    # Restores at the clamp sit well inside the 60 s adaptive cap with C3's
    # measured cost, so the timeout bucket is zero for the fixture — but the
    # counter exists and is exercised by an injected-budget run below.
    assert sim.counters["excluded_timeout"] == 0
    sim.timeout_for = lambda tokens: 0.001  # pathological: nothing fits
    sim.simulate(_craft_attempts(("s-b", 10_000), ("s-a", 10_000), start_seconds=600.0))
    assert sim.counters["excluded_timeout"] >= 1
    assert sim.counters["save_skipped_timeout"] >= 1


def test_local_serve_share_guards_against_serving_less(fixture_data, clean_production_state):
    """C2 §4.3 G5: an arm may not buy reuse by serving fewer prompts locally."""
    report = bakeoff.bakeoff(fixture_data)
    for arm in report["arms"].values():
        share = arm["raw"]["local_serve_share"]
        assert 0.0 < share <= 1.0
        assert arm["raw"]["local_serves"] + arm["raw"]["pool_exhausted"] + arm["exclusions"][
            "context_too_large"
        ] == arm["raw"]["total_attempts"]
    # Candidate B keeps live geometry, so its local serve share is unchanged;
    # candidate A's smaller clamp collapses it (the AC5-violating trade).
    assert report["arms"]["arm1"]["raw"]["local_serve_share"] == pytest.approx(
        report["arms"]["arm0"]["raw"]["local_serve_share"]
    )
    assert report["arms"]["arm2"]["raw"]["local_serve_share"] < (
        report["arms"]["arm0"]["raw"]["local_serve_share"]
    )
    assert report["arms"]["arm1"]["gates"]["G5"]["pass"] is True


def test_no_uplift_is_attributable_not_merely_observed(fixture_data, clean_production_state):
    """AC4: every candidate's result names the channel that bounded it."""
    report = bakeoff.bakeoff(fixture_data)
    arm1 = report["arms"]["arm1"]
    # Candidate B as specified (a cap of 4 retained snapshots) cannot raise
    # reuse: restores are rare and the cap drops snapshots the baseline keeps.
    # Restores are a small minority of local serves (the reuse that exists comes
    # from residency, not from snapshot restoration).
    assert arm1["raw"]["restore"]["success"] < 0.15 * arm1["raw"]["attempts"]
    assert arm1["raw"]["retention_evictions"] > 0
    assert arm1["gates"]["G1"]["pass"] is False
    assert "paired per-session median delta" in arm1["gates"]["G1"]["rationale"]
    # The dominant exclusion channels are reported per configuration.
    assert arm1["exclusions"]["context_too_large"] > arm1["raw"]["attempts"]
    assert arm1["exclusions"]["slot_busy"] > 0
    # The diagnostic arm (busy gate relaxed) is the biggest same-geometry lever
    # and is still short of the G1 threshold.
    arm3 = report["arms"]["arm3"]
    assert arm3["delta_vs_arm0"]["median_pp_gt20k"] > arm1["delta_vs_arm0"]["median_pp_gt20k"]
    assert arm3["gates"]["G1"]["pass"] == "n/a"


# ---------------------------------------------------------------------------
# Usage-report percentile definitions (AC3)
# ---------------------------------------------------------------------------


def test_percentile_definition_matches_the_usage_report(caplog):
    """The median uses scripts/analyse_slot_ratio_restore._percentiles, not a
    hand-rolled quantile (the AC's 'no ad-hoc percentile definition')."""
    values = [0.0, 0.0, 0.5, 0.8, 1.0]
    ours = bakeoff.percentiles_pct(values)
    theirs = analyse._percentiles([v * 100.0 for v in values])
    assert ours["p50"] == theirs["p50"]
    assert ours["p10"] == theirs["p10"]
    assert ours["p90"] == theirs["p90"]
    # Feeding percentages keeps the shared helper's 2-dp rounding at 0.01 pp.
    assert ours["p50"] == 50.0


def test_reported_baseline_matches_the_measured_reference(fixture_data, clean_production_state):
    """AC3: the pinned 26.1 / 15.6 / 28-of-227 reference is carried verbatim."""
    report = bakeoff.bakeoff(fixture_data)
    ref = report["baseline_ref"]
    assert ref["aggregate_pct"] == MEASURED_AGGREGATE_PCT
    assert ref["median_pct_gt20k"] == MEASURED_MEDIAN_PCT
    assert tuple(ref["cold_of"]) == MEASURED_COLD_OF


def test_calibrated_arm0_reproduces_the_measured_baseline(
    fixture_data, clean_production_state
):
    """The proportional calibration anchors arm0 to the measured baseline."""
    report = bakeoff.bakeoff(fixture_data)
    arm0 = report["arms"]["arm0"]["calibrated"]
    assert arm0["aggregate_pct"] == pytest.approx(MEASURED_AGGREGATE_PCT, abs=0.1)
    assert arm0["median_pct_gt20k"] == pytest.approx(MEASURED_MEDIAN_PCT, abs=0.1)
    # All arms scale with the same per-metric factor, so arm order is preserved.
    raw0 = report["arms"]["arm0"]["raw"]
    raw1 = report["arms"]["arm1"]["raw"]
    assert report["arms"]["arm1"]["calibrated"]["median_pct_gt20k"] == pytest.approx(
        raw1["gt20k"]["median_pct"] / raw0["gt20k"]["median_pct"] * MEASURED_MEDIAN_PCT,
        abs=0.01,  # both sides are rounded to 0.01 pp in the report
    )


# ---------------------------------------------------------------------------
# JSON artefact (AC6)
# ---------------------------------------------------------------------------


def test_bakeoff_json_is_serialisable_and_committed(fixture_data, clean_production_state):
    report = bakeoff.bakeoff(fixture_data)
    payload = bakeoff.bakeoff_result_to_json(report)
    json.dumps(payload)  # must be JSON-serialisable
    assert getattr(bakeoff, "BAKEOFF_SENTINEL", None) is not None
    fresh = json.dumps(payload, sort_keys=True, indent=2)
    assert COMMITTED_ARTEFACT.exists(), (
        "the bake-off result artefact must be committed next to the replay "
        "(AC6: C5 cites machine-readable evidence)"
    )
    assert COMMITTED_ARTEFACT.read_text(encoding="utf-8") == fresh + "\n", (
        "the committed artefact drifted from a fresh run — re-generate it with "
        "the bake-off writer and commit both"
    )


# ---------------------------------------------------------------------------
# Restore-cost model fed from C3 (AC1 "C3 measured restore cost")
# ---------------------------------------------------------------------------


def test_restore_cost_model_is_consistent_with_c3_measurement(caplog):
    """Pinned band costs match the committed C3 summary within tolerance."""
    summary = json.loads(
        (REPO_ROOT / "proxy" / "benchmarks" / "results" / "slot_snapshot_bands_20261028" / "summary.json").read_text(encoding="utf-8")
    )
    model = bakeoff.RestoreCostModel()
    for band in summary["bands"]:
        tokens = band["target_tokens"]
        measured = band["under_load"]["restore_seconds"]
        assert model.restore_seconds(tokens) == pytest.approx(measured, rel=0.25), (
            f"pinned restore cost for the {band['band']} band drifted from C3"
        )
    # The adaptive timeout is the production model from config.
    assert model.timeout_seconds(25_000) == bakeoff.PER_TOKEN_TIMEOUT_SECONDS * 25_000 + bakeoff.BASE_TIMEOUT_SECONDS
    assert model.timeout_seconds(120_000) == bakeoff.MAX_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Human-readable report (RESULTS_NOTE.md) agrees with the evidence (AC2/AC4)
# ---------------------------------------------------------------------------

RESULTS_NOTE = COMMITTED_ARTEFACT.parent / "RESULTS_NOTE.md"


def _note_arm_row(note_text: str, arm_name: str) -> str:
    """Return the RESULTS_NOTE table row for *arm_name*.

    The note's arm table is the human-readable form of the numbers C5 cites, so
    it must stay in step with ``summary.json``; this locates that one row.
    """
    for line in note_text.splitlines():
        if line.startswith(f"| {arm_name} "):
            return line
    raise AssertionError(f"RESULTS_NOTE.md has no table row for {arm_name}")


@pytest.mark.parametrize("arm_name", ["arm0", "arm1", "arm2", "arm3"])
def test_results_note_reports_the_artefact_numbers(arm_name: str):
    """The note's per-arm row quotes the committed artefact's own numbers.

    A decision record (C5) cites the note; if the note drifts from
    ``summary.json`` the citation silently becomes wrong. Each asserted value is
    read back out of the committed JSON, never hard-coded here.
    """
    artefact = json.loads(COMMITTED_ARTEFACT.read_text(encoding="utf-8"))
    arm = artefact["arms"][arm_name]
    row = _note_arm_row(RESULTS_NOTE.read_text(encoding="utf-8"), arm_name)

    calibrated = arm["calibrated"]
    assert f"{calibrated['aggregate_pct']:.2f} %" in row, row
    assert f"{calibrated['median_pct_gt20k']:.2f} %" in row, row

    cold, total = calibrated["cold_of"]
    assert f"{cold} / {total}" in row, row

    exclusions = arm["exclusions"]
    assert (
        f"{exclusions['slot_busy']} / {exclusions['cooldown']} / "
        f"{exclusions['context_too_large']} / {exclusions['timeout_exceeded']}" in row
    ), row

    restore = arm["raw"]["restore"]
    assert f"{restore['success']} / {restore['success'] + restore['failed']}" in row, row


# ---------------------------------------------------------------------------
# No production module imports the harness (AC5)
# ---------------------------------------------------------------------------


def test_harness_not_imported_by_any_production_module():
    """The bake-off is an analysis tool — never in the production import graph."""
    production_modules = (
        "proxy.prefill_eta",
        "proxy.metrics",
        "proxy.session",
        "proxy.provider",
        "proxy.slot_cache_ratio",
        "proxy.router_helpers",
        "proxy.router",
        "proxy.handlers",
        "proxy.residency",
        "proxy.contention_queue",
        "proxy.observability",
    )
    for name in production_modules:
        module = importlib.import_module(name)
        assert not hasattr(module, bakeoff.BAKEOFF_SENTINEL), (
            f"{name} imported the bake-off harness — it is an analysis tool "
            "and must stay out of the production import graph"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _craft_attempts(
    *pairs: tuple[str, int] | tuple[str, int, int],
    gap_seconds: float = 5.0,
    start_seconds: float = 0.0,
) -> list[dict]:
    """Build tiny deterministic attempt corpora for gate-level unit tests.

    Each spec is ``(session, estimated_tokens)`` or
    ``(session, estimated_tokens, new_tokens)`` (default new_tokens 1,000 →
    ~3.6 s prefill at the pinned 281 t/s, so 5 s gaps do not overlap).
    Successive attempts advance the sim clock by *gap_seconds*; the whole
    batch can be shifted forward with *start_seconds* so repeated ``simulate``
    calls keep a monotonic timeline (e.g. past a cooldown window).
    """
    import datetime as dt

    base = dt.datetime(2026, 10, 24, 18, 38, 0)
    attempts = []
    for index, spec in enumerate(pairs):
        session, tokens = spec[0], spec[1]
        new_tokens = spec[2] if len(spec) > 2 else min(tokens, 1_000)
        ts = base + dt.timedelta(seconds=start_seconds + index * gap_seconds)
        attempts.append(
            {
                "ts": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "session": session,
                "estimated_tokens": tokens,
                "new_tokens": new_tokens,
                "cached_ratio": 0.0,
                "route": "queue",
                "outcome": "other",
            }
        )
    return attempts
