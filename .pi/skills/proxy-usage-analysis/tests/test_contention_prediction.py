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

    def test_over_prediction_verdict(self):
        summary = type(
            "Sum",
            (),
            {"status_request_events": _events([STATUS_ETA.replace("prefill_eta_seconds=91.6", "prefill_eta_seconds=8.0")])},
        )()
        speed = type("Speed", (), {"prompt_eval_seconds": [2.0]})()
        out = reporting.prefill_eta_error_json(summary, speed)
        # predicted p50 = 8.0 vs observed p50 = 2.0 -> +300%: the estimate
        # over-predicts real prefill duration (the harmful direction: it
        # refuses work that would have been servable).
        assert out["verdict"] == "over_predicts"

    def test_within_tolerance_verdict(self):
        summary = type(
            "Sum",
            (),
            {"status_request_events": _events([STATUS_ETA.replace("prefill_eta_seconds=91.6", "prefill_eta_seconds=10.5")])},
        )()
        speed = type("Speed", (), {"prompt_eval_seconds": [10.0]})()
        out = reporting.prefill_eta_error_json(summary, speed)
        assert out["verdict"] == "matches_reality"


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
        assert err["verdict"] in ("over_predicts", "under_predicts", "inconclusive")
        # Verdict must be decisive: predicted (p50 91.6/120) far exceeds the
        # observed prompt-eval durations in the fixture.
        assert err["verdict"] == "over_predicts"

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
