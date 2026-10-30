# <!-- REFACTOR-COMMENT: LP-0MVGR2OXT0096OGK -- new M1/M3 measurement tests -->
"""Unit tests for the contention-prediction measurements (LP-0MVGR2OXT0096OGK).

The item verifies that the payload's ``lease_state`` / ``prefill_eta_seconds``
signals predict real local contention. Because ``local_dispatch_denied`` lines
carry no ``lease_state``, M1 attributes each denial to the adjacent
``local_admission_guard`` record (guard reason + ``busy_slots`` /
``in_flight_prefills`` / ``free_slot``) and derives a pool-state verdict
(``idle_leased`` / ``prefill_bound`` / ``saturated``). M3 compares the
parked ``prefill_eta_seconds`` distribution against observed llama-server
``prompt eval`` durations at p50/p90.

These tests are fixture-driven (real log-line shapes from
/var/log/llama-proxy) so no production log or GPU is needed. Test-first per
the item's method requirement: the two CLI aggregations and their report /
JSON rendering are all covered here before the implementation lands.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import aggregation
import log_parser
import pytest
import reporting
from aggregation import (
    DenialAttribution,
    attribute_denials,
    classify_denial_verdict,
)

WINDOW_START = datetime(2026, 8, 2, 14, 0, 0)
WINDOW_END = datetime(2026, 8, 2, 15, 0, 0)

# --- Real guard/denial line shapes (2026-10-28 proxy.log) ------------------

GUARD_IDLE = (
    "2026-08-02 14:05:00,000 - INFO - local_admission_guard decision=reject "
    "reason=prefill_budget_exceeded enforced=true eta_prefill_s=0.0 "
    "eta_occupancy_s=0.0 projected_wait_s=0.0 budget_s=60.0 deadline_s=120.0 "
    "deadline_budget_s=114.0 new_tokens=79899 budget_tokens=25000 busy_slots=0 "
    "in_flight_prefills=0 free_slot=false priority=medium policy=reject "
    "model=Qwen3 samples=0 provider=local-qwen3 session=unknown"
)
GUARD_PREFILL = (
    "2026-08-02 14:05:30,000 - INFO - local_admission_guard decision=reject "
    "reason=eta_exceeds_budget enforced=true eta_prefill_s=28.3 "
    "eta_occupancy_s=39.4 projected_wait_s=67.7 budget_s=60.0 deadline_s=120.0 "
    "deadline_budget_s=114.0 new_tokens=12953 budget_tokens=25000 busy_slots=1 "
    "in_flight_prefills=1 free_slot=false priority=low policy=reject "
    "model=Qwen3 samples=200 provider=local-qwen3 session=unknown"
)
GUARD_SATURATED = (
    "2026-08-02 14:06:00,000 - INFO - local_admission_guard decision=reject "
    "reason=prefill_budget_exceeded enforced=true eta_prefill_s=0.0 "
    "eta_occupancy_s=0.0 projected_wait_s=0.0 budget_s=60.0 deadline_s=120.0 "
    "deadline_budget_s=114.0 new_tokens=55902 budget_tokens=25000 busy_slots=2 "
    "in_flight_prefills=0 free_slot=false priority=medium policy=reject "
    "model=Qwen3 samples=0 provider=local-qwen3 session=unknown"
)
# A legacy guard line (pre-payload) with no pool-state fields: verdict: unknown.
GUARD_LEGACY = (
    "2026-08-02 14:07:00,000 - INFO - local_admission_guard decision=reject "
    "reason=eta_exceeds_budget enforced=true eta_prefill_s=468.0 budget_s=120.0 "
    "priority=low policy=reject model=Qwen3 samples=0 provider=local-qwen3 "
    "session=unknown"
)
DENIAL_NEAR_IDLE = (
    "2026-08-02 14:05:00,025 - INFO - local_dispatch_denied session=s1 "
    "owner=o1 active=2 cold_start=False"
)
DENIAL_NEAR_PREFILL = (
    "2026-08-02 14:05:30,020 - INFO - local_dispatch_denied session=s2 "
    "owner=o2 active=1 cold_start=False"
)
DENIAL_LONE = (
    "2026-08-02 14:40:00,000 - INFO - local_dispatch_denied session=s3 "
    "owner=o3 active=1 cold_start=False"
)
# A denial exactly 60s before a guard: on the join-window boundary (inclusive).
DENIAL_BOUNDARY = (
    "2026-08-02 14:06:00,000 - INFO - local_dispatch_denied session=s4 "
    "owner=o4 active=1 cold_start=False"
)

STATUS_ETA = (
    "2026-08-02 14:08:00,000 - INFO - status_request active_query=false "
    "available_slots=2 capacity_hint=idle client_ip=192.168.0.199 "
    "client_ip_source=direct client_port=42868 current_model=Qwen3 latency_ms=10 "
    "llama_server_running=true local_active_query=false local_idle_seconds=5.0 "
    "local_in_flight_requests=0 local_owner_lease_remaining_seconds=None "
    "local_owner_session_id=None model_switch_in_progress=false "
    "prefill_eta_seconds=91.6 queue_eta_seconds=183.2 slots_detail_stale=false "
    "slots_stale=false status_schema_version=3 total_slots=2"
)
STATUS_ETA_NONE = STATUS_ETA.replace("prefill_eta_seconds=91.6", "prefill_eta_seconds=None")
STATUS_ETA_PREFILL = STATUS_ETA.replace("capacity_hint=idle", "capacity_hint=prefill_bound")
STATUS_LEGACY = (
    "2026-08-02 14:09:00,000 - INFO - status_request active_query=false "
    "available_slots=2 current_model=Qwen3 latency_ms=10 llama_server_running=true "
    "local_active_query=false slots_detail_stale=true slots_stale=true "
    "status_schema_version=1 total_slots=2"
)

QWEN3_SPAWN = "srv load: spawning server instance with name=Qwen3 on port 32999"


def _schedule():
    import bucketing

    return bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])


def _events(lines: list[str]) -> list[object]:
    return [e for e in (log_parser.parse_log_line(ln) for ln in lines) if e is not None]


# ---------------------------------------------------------------------------
# Log-line parsing: guard pool-state fields + status_request
# ---------------------------------------------------------------------------


class TestGuardPoolStateFields:
    """M1: the guard record's busy_slots / in_flight_prefills / free_slot."""

    def test_parses_pool_state_fields(self):
        ev = log_parser.parse_log_line(GUARD_IDLE)
        assert ev.kind == "local_admission_guard"
        assert ev.decision == "reject"
        assert ev.reason == "prefill_budget_exceeded"
        assert ev.busy_slots == 0
        assert ev.in_flight_prefills == 0
        assert ev.free_slot == "false"

    def test_parses_nonzero_pool_state(self):
        ev = log_parser.parse_log_line(GUARD_PREFILL)
        assert ev.busy_slots == 1
        assert ev.in_flight_prefills == 1
        assert ev.free_slot == "false"

    def test_legacy_guard_without_pool_state_is_none(self):
        ev = log_parser.parse_log_line(GUARD_LEGACY)
        assert ev.kind == "local_admission_guard"
        assert ev.busy_slots is None
        assert ev.in_flight_prefills is None
        assert ev.free_slot is None


class TestStatusRequestParsing:
    """M3: status_request lines carry the predicted prefill ETA and capacity."""

    def test_parses_prefill_eta_and_capacity_hint(self):
        ev = log_parser.parse_log_line(STATUS_ETA)
        assert ev.kind == "status_request"
        assert ev.prefill_eta_seconds == pytest.approx(91.6)
        assert ev.capacity_hint == "idle"
        assert ev.status_schema_version == 3

    def test_none_eta_parses_as_none(self):
        ev = log_parser.parse_log_line(STATUS_ETA_NONE)
        assert ev.kind == "status_request"
        assert ev.prefill_eta_seconds is None
        assert ev.capacity_hint == "idle"

    def test_legacy_status_request_without_eta(self):
        ev = log_parser.parse_log_line(STATUS_LEGACY)
        assert ev.kind == "status_request"
        assert ev.prefill_eta_seconds is None
        assert ev.capacity_hint is None
        assert ev.status_schema_version == 1


# ---------------------------------------------------------------------------
# M1: denial verdict classification
# ---------------------------------------------------------------------------


class TestDenialVerdictClassification:
    """Derived pool-state verdict from the guard record's measured fields."""

    def test_idle_leased_when_no_slot_busy_and_no_prefill(self):
        ev = log_parser.parse_log_line(GUARD_IDLE)
        assert classify_denial_verdict(ev) == "idle_leased"

    def test_prefill_bound_when_a_prefill_is_in_flight(self):
        ev = log_parser.parse_log_line(GUARD_PREFILL)
        assert classify_denial_verdict(ev) == "prefill_bound"

    def test_saturated_when_all_slots_decode(self):
        ev = log_parser.parse_log_line(GUARD_SATURATED)
        assert classify_denial_verdict(ev) == "saturated"

    def test_prefill_wins_over_saturate_when_prefill_in_flight(self):
        # busy=2 means the pool is full, but a prefill is the gating resource:
        # prefill_bound is the dominant condition (serialised prefills).
        ev = log_parser.parse_log_line(
            GUARD_SATURATED.replace("busy_slots=2 in_flight_prefills=0", "busy_slots=2 in_flight_prefills=1")
        )
        assert classify_denial_verdict(ev) == "prefill_bound"

    def test_unknown_when_pool_state_fields_missing(self):
        ev = log_parser.parse_log_line(GUARD_LEGACY)
        assert classify_denial_verdict(ev) == "unknown"

    def test_idle_leased_when_slot_free_but_gpu_idle(self):
        # free_slot=true still means the GPU measured zero work: idle_leased.
        ev = log_parser.parse_log_line(
            GUARD_IDLE.replace("free_slot=false", "free_slot=true")
        )
        assert classify_denial_verdict(ev) == "idle_leased"


# ---------------------------------------------------------------------------
# M1: denial → guard attribution (timestamp-block join)
# ---------------------------------------------------------------------------


class TestDenialAttribution:
    """Join each `local_dispatch_denied` to its adjacent guard record."""

    def test_joins_nearest_guard_within_window(self):
        denial = _events([DENIAL_NEAR_IDLE])[0]
        guard = _events([GUARD_IDLE])[0]
        attrs = attribute_denials([denial], [guard])
        assert len(attrs) == 1
        a = attrs[0]
        assert isinstance(a, DenialAttribution)
        assert a.ts == denial.ts
        assert a.guard_ts == guard.ts
        assert a.guard_reason == "prefill_budget_exceeded"
        assert a.verdict == "idle_leased"
        assert a.busy_slots == 0
        assert a.in_flight_prefills == 0
        assert a.free_slot == "false"

    def test_unattributed_when_no_guard_in_window(self):
        denial = _events([DENIAL_LONE])[0]
        guard = _events([GUARD_IDLE])[0]  # 35 minutes away
        attrs = attribute_denials([denial], [guard])
        assert attrs == []  # outside the join window: not attributed

    def test_boundary_window_inclusive(self):
        # GUARD_LEGACY sits at 14:07:00, exactly 60s after the denial.
        denial = _events([DENIAL_BOUNDARY])[0]
        guard = _events([GUARD_LEGACY])[0]
        attrs = attribute_denials([denial], [guard], window_seconds=60.0)
        assert len(attrs) == 1
        assert attrs[0].verdict == "unknown"  # legacy guard, no pool state

    def test_nearest_guard_selected(self):
        denial = _events([DENIAL_NEAR_IDLE])  # 14:05:00.025
        # 25 ms away (the real decision block) vs 10 s away (a later burst).
        nearer = _events([GUARD_IDLE])[0]
        farther = _events([GUARD_PREFILL.replace("14:05:30,000", "14:05:10,000")])[0]
        attrs = attribute_denials(denial, [farther, nearer])
        assert len(attrs) == 1
        assert attrs[0].guard_ts == nearer.ts
        assert attrs[0].verdict == "idle_leased"

    def test_all_denials_attributed_through_aggregate(self):
        lines = [GUARD_IDLE, DENIAL_NEAR_IDLE, GUARD_PREFILL, DENIAL_NEAR_PREFILL, DENIAL_LONE]
        res = aggregation.aggregate(
            _events(lines), WINDOW_START, WINDOW_END, _schedule()
        )
        assert res.dispatch_denied_count == 3
        attrs = res.denial_attributions
        assert len(attrs) == 2  # DENIAL_LONE has no guard in its 60s block
        verdicts = sorted(a.verdict for a in attrs)
        assert verdicts == ["idle_leased", "prefill_bound"]

    def test_status_requests_collected_through_aggregate(self):
        lines = [STATUS_ETA, STATUS_ETA_NONE, STATUS_LEGACY]
        res = aggregation.aggregate(
            _events(lines), WINDOW_START, WINDOW_END, _schedule()
        )
        assert len(res.status_request_events) == 3
        etas = [e.prefill_eta_seconds for e in res.status_request_events]
        assert etas == [pytest.approx(91.6), None, None]


# ---------------------------------------------------------------------------
# M3: predicted vs observed prefill ETA error (p50/p90)
# ---------------------------------------------------------------------------


class TestPrefillEtaError:
    """Quantile-matched predicted-vs-observed comparison (the CLI's M3)."""

    def test_error_and_relative_error_at_p50_p90(self):
        summary = type(
            "Sum",
            (),
            {
                "status_request_events": _events(
                    [
                        STATUS_ETA.replace("prefill_eta_seconds=91.6", "prefill_eta_seconds=50.0"),
                        STATUS_ETA.replace("prefill_eta_seconds=91.6", "prefill_eta_seconds=100.0"),
                        STATUS_ETA.replace("prefill_eta_seconds=91.6", "prefill_eta_seconds=120.0"),
                    ]
                )
            },
        )()
        speed = type("Speed", (), {"prompt_eval_seconds": [2.0, 3.0, 10.0]})()
        out = reporting.prefill_eta_error_json(summary, speed)
        assert out["predicted"]["samples"] == 3
        assert out["observed"]["samples"] == 3
        # predicted sorted [50, 100, 120] -> p50 = 100, p90 = 116
        assert out["predicted"]["p50"] == 100.0
        assert out["predicted"]["p90"] == 116.0
        # observed sorted [2, 3, 10] -> p50 = 3, p90 = 8.6
        assert out["observed"]["p50"] == 3.0
        assert out["observed"]["p90"] == 8.6
        assert out["error_p50"] == pytest.approx(97.0)
        assert out["error_p90"] == pytest.approx(107.4)
        assert out["relative_error_p50_pct"] == pytest.approx(97.0 / 3.0 * 100, abs=0.1)
        assert out["relative_error_p90_pct"] == pytest.approx(107.4 / 8.6 * 100, abs=0.1)

    def test_empty_observed_yields_inconclusive(self):
        summary = type("Sum", (), {"status_request_events": _events([STATUS_ETA])})()
        speed = type("Speed", (), {"prompt_eval_seconds": []})()
        out = reporting.prefill_eta_error_json(summary, speed)
        assert out["observed"]["samples"] == 0
        assert out["observed"]["p50"] is None
        assert out["verdict"] == "inconclusive"

    def test_no_predictions_yields_inconclusive(self):
        summary = type("Sum", (), {"status_request_events": _events([STATUS_ETA_NONE])})()
        speed = type("Speed", (), {"prompt_eval_seconds": [2.0, 3.0]})()
        out = reporting.prefill_eta_error_json(summary, speed)
        assert out["predicted"]["samples"] == 0
        assert out["verdict"] == "inconclusive"

    def test_cross_population_comparison_is_labelled_and_one_sided(self):
        """LP-0MVSD5RJW008F1IB AC3: no calibration verdict without a matched join.

        A cache-cold bound compared with the served remainder can never
        "over-predict" in a meaningful sense, so the ambiguous two-sided
        wording is gone; the crossing is named and the below-sample-floor
        outcome is ``inconclusive`` (here: 1 observed dispatch < 30).
        """
        summary = type(
            "Sum",
            (),
            {"status_request_events": _events([STATUS_ETA.replace("prefill_eta_seconds=91.6", "prefill_eta_seconds=8.0")])},
        )()
        speed = type("Speed", (), {"prompt_eval_seconds": [2.0]})()
        out = reporting.prefill_eta_error_json(summary, speed)

        assert out["comparison"] == "bound_vs_served_remainder"
        assert out["verdict"] == "inconclusive"
        assert out["bound"]["holds"] is None
        # The legacy keys survive for trend continuity.
        for key in (
            "predicted",
            "observed",
            "error_p50",
            "error_p90",
            "relative_error_p50_pct",
            "relative_error_p90_pct",
            "tolerance_pct",
            "verdict",
        ):
            assert key in out

    def test_bound_holds_when_served_p90_stays_within_the_predicted_p50(self):
        summary = type("Sum", (), {"status_request_events": _events([STATUS_ETA])})()
        speed = type("Speed", (), {"prompt_eval_seconds": [5.0] * 30})()
        out = reporting.prefill_eta_error_json(summary, speed)

        assert out["bound"]["verdict"] == "bound_holds"
        assert out["bound"]["holds"] is True
        assert out["verdict"] == "bound_holds"
        assert out["bound"]["margin_seconds"] == pytest.approx(91.6 - 5.0, abs=0.11)

    def test_bound_violated_when_served_p90_exceeds_the_predicted_p50(self):
        summary = type(
            "Sum",
            (),
            {"status_request_events": _events([STATUS_ETA.replace("prefill_eta_seconds=91.6", "prefill_eta_seconds=20.0")])},
        )()
        # p90 of [5, 10, 30] repeated 10x = 30 s > the 20 s published bound.
        speed = type("Speed", (), {"prompt_eval_seconds": [5.0, 10.0, 30.0] * 10})()
        out = reporting.prefill_eta_error_json(summary, speed)

        assert out["bound"]["verdict"] == "bound_violated"
        assert out["bound"]["holds"] is False
        assert out["bound"]["margin_seconds"] < 0

    def test_below_the_sample_floor_is_inconclusive_not_a_verdict(self):
        summary = type(
            "Sum",
            (),
            {"status_request_events": _events([STATUS_ETA.replace("prefill_eta_seconds=91.6", "prefill_eta_seconds=1.0")])},
        )()
        # 29 observed dispatches: one short of the floor, so even a blatant
        # violation is not reported as one.
        speed = type("Speed", (), {"prompt_eval_seconds": [60.0] * 29})()
        out = reporting.prefill_eta_error_json(summary, speed)

        assert out["bound"]["verdict"] == "inconclusive"
        assert out["bound"]["holds"] is None
        assert out["bound"]["observed_samples"] == 29
        assert out["bound"]["min_samples"] == reporting.PREFILL_ETA_BOUND_MIN_SAMPLES


# ---------------------------------------------------------------------------
# Report + JSON surface (the CLI deliverable)
# ---------------------------------------------------------------------------


class TestContentionReportAndJson:
    """the deterministic CLI exposes M1/M2/M3 in report.md and --json."""

    def _run(self, tmp_path: Path):
        log_dir = tmp_path / "logs"
        out_dir = tmp_path / "out"
        log_dir.mkdir()
        proxy_lines = [
            GUARD_IDLE, DENIAL_NEAR_IDLE,
            GUARD_PREFILL, DENIAL_NEAR_PREFILL,
            GUARD_SATURATED,
            DENIAL_LONE,   # unattributed
            STATUS_ETA,
            STATUS_ETA_PREFILL,
            STATUS_ETA_NONE,
        ]
        (log_dir / "proxy.log").write_text("\n".join(proxy_lines) + "\n")
        (log_dir / "llama-server.log").write_text(
            QWEN3_SPAWN
            + "\n[32999] prompt eval time =    2000.00 ms /    10 tokens (  200.00 ms per token,     5.00 tokens per second)\n"
            + "[32999] prompt eval time =    3000.00 ms /    10 tokens (  300.00 ms per token,     3.33 tokens per second)\n"
            + "[32999] prompt eval time =   10000.00 ms /    10 tokens ( 1000.00 ms per token,     1.00 tokens per second)\n"
        )
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=out_dir,
            config=None,
            llama_log_dir=log_dir,
        )
        return result, (out_dir / "report.md").read_text()

    def test_report_contains_contention_section(self, tmp_path):
        _result, md = self._run(tmp_path)
        section = md.split("## Local contention prediction vs reality", 1)[1]
        # M1 per-hour denial attribution plus the capacity-hint split.
        assert "| idle_leased |" in section
        assert "prefill_bound" in section
        assert "saturated" in section
        assert "capacity_hint" in section
        # M2 busy fraction reference.
        assert "busy" in section.lower()
        # M3 predicted/observed percentiles and an explicit verdict line.
        assert "p50" in section and "p90" in section
        assert "does not match reality" in section or "match reality" in section

    def test_json_exposes_measurements(self, tmp_path):
        result, _md = self._run(tmp_path)
        data = reporting.summary_to_json(result.summary, result.mode_map)
        json.dumps(data)  # serialisable
        att = data["denial_attribution"]
        assert att["denials"] == 3
        assert att["attributed"] == 2
        assert att["unattributed"] == 1
        assert att["by_verdict"]["idle_leased"] == 1
        assert att["by_verdict"]["prefill_bound"] == 1
        # GUARD_SATURATED has no adjacent denial in the fixture: only the two
        # guards directly joined to a denial contribute.
        assert att["by_guard_reason"]["prefill_budget_exceeded"] == 1
        assert att["by_guard_reason"]["eta_exceeds_budget"] == 1
        # capacity_hint distribution from status_request events.
        assert data["status_capacity_hint"]["idle"] == 2
        assert data["status_capacity_hint"]["prefill_bound"] == 1
        # M3 report keys.
        err = data["prefill_eta_error"]
        assert err["predicted"]["samples"] == 2  # STATUS_ETA + STATUS_ETA_PREFILL
        assert err["observed"]["samples"] == 3
        # The cross-population crossing is named and cannot claim a calibration
        # defect; 3 observations are below the bound-test floor.
        assert err["comparison"] == "bound_vs_served_remainder"
        assert err["verdict"] == "inconclusive"
        # Matched accuracy is reported (no guard records in this fixture: the
        # dispatches are unmatched, never folded into a fabricated match).
        assert err["matched"]["verdict"] == "inconclusive"
        assert err["matched"]["pairs"] == 0

    def test_per_hour_rows_carry_busy_fraction(self, tmp_path):
        result, _md = self._run(tmp_path)
        att = reporting.summary_to_json(result.summary, result.mode_map)["denial_attribution"]
        rows = att["per_hour"]
        assert len(rows) == 1
        hour = rows[0]
        assert hour["denials"] == 3
        assert hour["unattributed"] == 1
        # The local-busy fraction for the hour is present (M2 corroboration),
        # even when the window has no attributable busy streams.
        assert "busy_pct" in hour


# ---------------------------------------------------------------------------
# LP-0MVSD5RJW008F1IB AC6: the poll log carries the calibration context
# ---------------------------------------------------------------------------

# Real shape of the enriched line: the proxy appends the calibration context to
# ``status_request`` (log fields are emitted alphabetically, so the enrichment
# lands beside the existing values without disturbing them).
STATUS_ETA_ENRICHED = (
    "2026-08-02 14:10:00,000 - INFO - status_request active_query=false "
    "available_slots=2 capacity_hint=prefill_bound client_ip=192.168.0.199 "
    "client_ip_source=direct client_port=42868 current_model=Qwen3 "
    "latency_ms=10 llama_server_running=true local_active_query=false "
    "local_idle_seconds=5.0 local_in_flight_requests=0 "
    "local_owner_lease_remaining_seconds=None local_owner_session_id=None "
    "model_switch_in_progress=false "
    "prefill_eta_assumed_cached_ratio=0.0 prefill_eta_calibrated=true "
    "prefill_eta_kind=reference_cold_bound prefill_eta_rate_bucket=contended "
    "prefill_eta_reference_tokens=25000 prefill_eta_samples=12 "
    "prefill_eta_seconds=97.5 queue_eta_seconds=195.0 slots_detail_stale=false "
    "slots_stale=false status_schema_version=4 total_slots=2"
)


class TestStatusRequestCalibrationContext:
    """AC6: poll lines say WHICH reference size and calibration produced the bound."""

    def test_parses_the_appended_calibration_context(self):
        ev = log_parser.parse_log_line(STATUS_ETA_ENRICHED)

        assert ev.kind == "status_request"
        assert ev.prefill_eta_reference_tokens == 25_000
        assert ev.prefill_eta_calibrated is True
        assert ev.prefill_eta_rate_bucket == "contended"
        assert ev.prefill_eta_samples == 12
        assert ev.prefill_eta_kind == "reference_cold_bound"
        assert ev.prefill_eta_assumed_cached_ratio == 0.0

    def test_fields_reach_status_request_events_through_the_aggregate(self):
        res = aggregation.aggregate(
            _events([STATUS_ETA_ENRICHED, STATUS_ETA]),
            WINDOW_START,
            WINDOW_END,
            _schedule(),
        )

        enriched, legacy = res.status_request_events
        assert enriched.prefill_eta_reference_tokens == 25_000
        assert enriched.prefill_eta_calibrated is True
        assert enriched.prefill_eta_rate_bucket == "contended"
        assert enriched.prefill_eta_samples == 12
        # Pre-change lines keep ``None`` — old logs are never reclassified.
        assert legacy.prefill_eta_reference_tokens is None
        assert legacy.prefill_eta_calibrated is None

    def test_existing_fields_still_parse_unchanged_on_the_enriched_line(self):
        """Regression pin: the additive fields disturb nothing old."""
        ev = log_parser.parse_log_line(STATUS_ETA_ENRICHED)

        assert ev.prefill_eta_seconds == pytest.approx(97.5)
        assert ev.capacity_hint == "prefill_bound"
        assert ev.status_schema_version == 4
        assert ev.available_slots == 2
        assert ev.total_slots == 2

    def test_pre_change_line_yields_none_for_the_new_fields(self):
        ev = log_parser.parse_log_line(STATUS_ETA)

        assert ev.prefill_eta_reference_tokens is None
        assert ev.prefill_eta_calibrated is None
        assert ev.prefill_eta_rate_bucket is None
        assert ev.prefill_eta_samples is None
        assert ev.prefill_eta_kind is None
        assert ev.prefill_eta_assumed_cached_ratio is None

    def test_in_repo_log_consumer_still_reads_the_published_eta(self):
        """Regression pin (AC6b): ``scripts/idle_admit_impact.py`` keeps working.

        It parses ``prefill_eta_seconds`` out of the same ``status_request``
        line; the appended fields must not disturb that read.
        """
        import sys

        repo_root = Path(__file__).resolve().parents[4]
        sys.path.insert(0, str(repo_root / "scripts"))
        try:
            import idle_admit_impact
        finally:
            sys.path.pop(0)

        event = idle_admit_impact.parse_event(
            STATUS_ETA_ENRICHED, WINDOW_START
        )

        assert event is not None
        assert event["kind"] == "status_request"
        assert event["prefill_eta_seconds"] == pytest.approx(97.5)


# ---------------------------------------------------------------------------
# LP-0MVSD5RJW008F1IB AC1: matched per-dispatch accuracy
# ---------------------------------------------------------------------------

# Guard decisions carry the session id, so a dispatch's first byte can be
# joined to the ETA the guard priced for THAT dispatch. ``eta_prefill_s`` is
# the own-prefill term (not ``projected_wait_s``, which adds occupancy).
GUARD_SESSION_A = (
    "2026-08-02 14:20:00,000 - INFO - local_admission_guard decision=admit "
    "reason=within_budget enforced=true eta_prefill_s=30.0 "
    "eta_occupancy_s=0.0 projected_wait_s=30.0 budget_s=60.0 deadline_s=120.0 "
    "deadline_budget_s=114.0 new_tokens=7692 budget_tokens=25000 busy_slots=0 "
    "in_flight_prefills=0 free_slot=true priority=low policy=reject "
    "model=Qwen3 samples=20 provider=local-qwen3 session=11111111-aaaa"
)
GUARD_SESSION_B_OBSERVE = (
    "2026-08-02 14:21:00,000 - INFO - local_admission_guard decision=reject "
    "reason=eta_exceeds_budget enforced=false eta_prefill_s=60.0 "
    "eta_occupancy_s=0.0 projected_wait_s=60.0 budget_s=60.0 deadline_s=120.0 "
    "deadline_budget_s=114.0 new_tokens=15384 budget_tokens=25000 busy_slots=0 "
    "in_flight_prefills=0 free_slot=true priority=low policy=observe "
    "model=Qwen3 samples=20 provider=local-qwen3 session=22222222-bbbb"
)
GUARD_SESSION_ZERO_ETA = (
    "2026-08-02 14:22:00,000 - INFO - local_admission_guard decision=reject "
    "reason=prefill_budget_exceeded enforced=true eta_prefill_s=0.0 "
    "eta_occupancy_s=0.0 projected_wait_s=0.0 budget_s=60.0 deadline_s=120.0 "
    "deadline_budget_s=114.0 new_tokens=79899 budget_tokens=25000 busy_slots=0 "
    "in_flight_prefills=0 free_slot=false priority=medium policy=reject "
    "model=Qwen3 samples=0 provider=local-qwen3 session=33333333-cccc"
)


def _dispatch(session: str, service_ms: float, ts: str = "14:20:10,000") -> str:
    return (
        f"2026-08-02 {ts} - INFO - dispatch_first_byte_ms={service_ms} "
        f"dispatch_to_first_byte_ms={service_ms} cache_state=cold "
        "cached_ratio=0.000 queue_wait_ms=0.0 "
        f"session={session} model=Qwen3"
    )


def _matched_summary(*lines: str):
    events = _events(list(lines))
    return type(
        "Sum",
        (),
        {
            "status_request_events": [],
            "local_admission_events": [
                e for e in events if e.kind == "local_admission_guard"
            ],
            "ttft_events": [e for e in events if e.kind == "ttft"],
        },
    )()


class TestMatchedPrefillEtaAccuracy:
    """AC1: predicted and observed describe the SAME dispatch."""

    def test_seeded_pair_renders_the_known_relative_error(self):
        """Guard priced 30 s; that dispatch took 15 s → +100 % → over_predicts."""
        summary = _matched_summary(
            GUARD_SESSION_A, _dispatch("11111111-aaaa", 15_000.0)
        )
        matched = reporting.prefill_eta_error_json(summary, None)["matched"]

        assert matched["pairs"] == 1
        assert matched["enforced_pairs"] == 1
        assert matched["observe_pairs"] == 0
        assert matched["predicted_p50"] == pytest.approx(30.0)
        assert matched["observed_p50"] == pytest.approx(15.0)
        assert matched["relative_error_p50_pct"] == pytest.approx(100.0)
        assert matched["verdict"] == "over_predicts"

    def test_seeded_match_within_tolerance_is_a_match(self):
        """Guard 30 s vs a 28 s first byte → +7.1 % → matches_reality."""
        summary = _matched_summary(
            GUARD_SESSION_A, _dispatch("11111111-aaaa", 28_000.0)
        )
        matched = reporting.prefill_eta_error_json(summary, None)["matched"]

        assert matched["relative_error_p50_pct"] == pytest.approx(7.1, abs=0.05)
        assert matched["verdict"] == "matches_reality"

    def test_under_prediction_is_reported(self):
        summary = _matched_summary(
            GUARD_SESSION_A, _dispatch("11111111-aaaa", 60_000.0)
        )
        matched = reporting.prefill_eta_error_json(summary, None)["matched"]

        assert matched["relative_error_p50_pct"] == pytest.approx(-50.0)
        assert matched["verdict"] == "under_predicts"

    def test_describes_the_same_dispatch_not_a_quantile_mix(self):
        """Two sessions priced differently keep their own pairs.

        The matched figure must not mix a cheap dispatch's prediction with an
        expensive dispatch's observation (which a quantile match would).
        """
        summary = _matched_summary(
            GUARD_SESSION_A,
            _dispatch("11111111-aaaa", 30_000.0),
            GUARD_SESSION_B_OBSERVE,
            _dispatch("22222222-bbbb", 120_000.0),
        )
        matched = reporting.prefill_eta_error_json(summary, None)["matched"]

        assert matched["pairs"] == 2
        # Pair 1: 30 vs 30 → 0 %. Pair 2: 60 vs 120 → -50 %. p50 = -25 %.
        assert matched["relative_error_p50_pct"] == pytest.approx(-25.0, abs=0.1)
        # The observe-mode pair is a valid pricing measurement, reported apart.
        assert matched["observe_pairs"] == 1
        assert matched["enforced_pairs"] == 1

    def test_unmatched_dispatch_is_counted_never_folded_in(self):
        """No guard record for the session (inert guard) → unmatched, not scored."""
        summary = _matched_summary(_dispatch("99999999-zzzz", 30_000.0))
        matched = reporting.prefill_eta_error_json(summary, None)["matched"]

        assert matched["pairs"] == 0
        assert matched["dispatches"] == 1
        assert matched["unmatched_dispatches"] == 1
        assert matched["verdict"] == "inconclusive"

    def test_zero_eta_guard_record_is_not_a_priced_estimate(self):
        """The calibration-independent budget refusal carries eta 0 → skip."""
        summary = _matched_summary(
            GUARD_SESSION_ZERO_ETA, _dispatch("33333333-cccc", 30_000.0)
        )
        matched = reporting.prefill_eta_error_json(summary, None)["matched"]

        assert matched["pairs"] == 0
        assert matched["unmatched_dispatches"] == 1


# ---------------------------------------------------------------------------
# LP-0MVSD5RJW008F1IB AC1/AC3: the report renders both figures, labelled
# ---------------------------------------------------------------------------


class TestContentionReportLabelsTheTwoComparisons:

    def test_report_labels_bound_and_matched_comparisons(self, tmp_path: Path):
        log_dir = tmp_path / "logs"
        out_dir = tmp_path / "out"
        log_dir.mkdir()
        proxy_lines = [
            STATUS_ETA,
            STATUS_ETA_ENRICHED,
            GUARD_SESSION_A,
            _dispatch("11111111-aaaa", 30_000.0, ts="14:20:10,000"),
        ]
        (log_dir / "proxy.log").write_text("\n".join(proxy_lines) + "\n")
        (log_dir / "llama-server.log").write_text(
            QWEN3_SPAWN
            + "\n[32999] prompt eval time =    2000.00 ms /    10 tokens "
            "(  200.00 ms per token,     5.00 tokens per second)\n"
        )
        reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=out_dir,
            config=None,
            llama_log_dir=log_dir,
        )
        md = (out_dir / "report.md").read_text()

        # The cross-population crossing is named for what it is...
        assert "bound vs served remainder" in md
        assert "M3b — matched per-dispatch prefill ETA accuracy" in md
        # ...and no longer asserts a calibration defect from that comparison.
        assert "over_predicts observed prefill" not in md
        assert "bound verdict" in md
        # The matched figure renders its pair split and its own verdict.
        assert "Unmatched dispatches" in md
        assert "matched prefill ETA verdict" in md
