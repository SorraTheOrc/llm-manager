"""Prefill-deadline residual measurement and attribution (LP-0MVRMOT0N005136O).

The script under test (``proxy/scripts/analyse_prefill_deadline_residual.py``)
produces the dataset the AC3 closure verdict on LP-0MVO8H7VQ003B12I needs. The
tests pin the *contract* the report claims, not the implementation:

- counting is eviction-scoped (the same ``reason=`` string also appears on the
  ``measured_cache_abort`` line of one event, so a naive ``grep -o`` doubles it);
- the user/audit split uses the parent-AC4 classifier (``audit-`` prefix) and
  reports an hourly rate per class;
- each ``prefill_deadline`` event is attributed to occupancy / capacity /
  predicted-warmth from recorded evidence, and the evidence columns travel with
  the verdict so a reader can re-derive it;
- a window with no ``prefill_progress_measured`` capture at all (any pre-F4
  window, e.g. the 114-event fix-item baseline) still produces a report — the
  measurement simply degrades to "backend progress unobserved" instead of
  crashing or silently claiming warmth was measured.

The fixture is a hand-built 1 h slice covering every branch; the real window's
figures are recorded on the work item and in
``docs/dev/prefill-deadline-residual.md``.
"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "prefill_deadline_residual_synthetic.log"
WINDOW_START = datetime(2026, 1, 1, 0, 0, 0)
WINDOW_END = datetime(2026, 1, 1, 1, 0, 0)


@pytest.fixture(name="script")
def _script():
    """Import the analysis script from ``proxy/scripts`` (no package there)."""
    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return importlib.import_module("analyse_prefill_deadline_residual")


@pytest.fixture(name="report")
def _report(script):
    log = script.parse_window(script.read_log_lines([FIXTURE]), WINDOW_START, WINDOW_END)
    log.files_read = 1
    return script.build_report(log, WINDOW_START, WINDOW_END)


def _deadline_rows(report):
    return [row for row in report["events"]]


def _row_for(report, session):
    rows = [row for row in _deadline_rows(report) if row["session"] == session]
    assert len(rows) == 1, f"expected one deadline event for {session}, got {len(rows)}"
    return rows[0]


def test_window_boundaries_are_respected_and_duplicates_dropped(script, report):
    """Only [start, end) is measured, and a rotated copy is not double counted."""
    sessions = {row["session"] for row in _deadline_rows(report)}
    assert "before-window" not in sessions
    assert "after-window" not in sessions
    # The fixture repeats user-e's eviction line verbatim (rotation copies).
    assert report["source"]["duplicate_lines_dropped"] == 1
    assert report["reason_split"]["prefill_deadline"]["distinct_sessions"] == 8
    assert report["window"]["hours"] == 1.0


def test_reason_counts_are_eviction_scoped(report):
    """`reason=` on the measured_cache_abort line must not inflate the counts."""
    by_reason = report["evictions"]["by_reason"]
    assert by_reason["prefill_deadline"] == 8
    assert by_reason["measured_cache_deficit"] == 2
    assert by_reason["lease_evicted"] == 1
    assert report["evictions"]["total"] == 11


def test_user_audit_split_uses_the_parent_ac4_classifier(script, report):
    """``audit-`` prefixed sessions are audit traffic; everything else is user."""
    assert script.AUDIT_SESSION_PREFIX == "audit-"
    split = report["reason_split"]["prefill_deadline"]
    assert (split["user"], split["audit"]) == (7, 1)
    assert split["user_per_hour"] == 7.0
    assert split["audit_per_hour"] == 1.0
    assert _row_for(report, "audit-WL-0MVPRZU0A000TSWD-parent-1a2b3c4d")["session_class"] == "audit"


def test_distributions_cover_the_recorded_columns(report):
    """AC3 needs the token/ETA/elapsed shape, not just the totals."""
    dist = report["distributions"]["prefill_deadline"]
    assert dist["estimated_tokens"]["count"] == 8
    assert dist["estimated_tokens"]["max"] == 47120
    assert dist["estimated_tokens"]["min"] == 0
    assert dist["elapsed_s"]["min"] == pytest.approx(120.0)
    # eta_prefill_s=unknown on one event must not poison the distribution.
    assert dist["eta_prefill_s"]["count"] == 7
    # new_tokens sorted: 0, 173, 173, 15942, 20000, 25000, 30000, 40000 — the
    # documented linear-interpolation percentile puts p50 halfway between the
    # two middle samples (17971), not on a sample.
    assert dist["new_tokens"]["p50"] == pytest.approx(17971.0)


def test_occupancy_attribution_from_an_other_session_lease(report):
    row = _row_for(report, "user-b")
    assert row["attribution"] == "occupancy"
    assert row["evidence"]["other_session_leases"] == ["other-a"]


def test_progress_zero_signature_is_reported_for_the_victim(report):
    """AC2: the baseline's ``progress=0`` signature is portable to any window."""
    evidence = _row_for(report, "user-b")["evidence"]
    assert evidence["own_lease_extensions"] == 1
    assert evidence["own_prefill_wait_threshold_events"] == 1
    assert evidence["own_zero_progress_signatures"] == 2
    # The other session's lease must not be attributed to the victim.
    assert _row_for(report, "user-e")["evidence"]["own_lease_extensions"] == 0


def test_occupancy_attribution_from_an_other_session_local_stream(report):
    row = _row_for(report, "user-f")
    assert row["attribution"] == "occupancy"
    assert row["evidence"]["other_session_local_completions"] == ["other-b"]


def test_occupancy_attribution_from_in_flight_two(report):
    row = _row_for(report, "user-h")
    assert row["attribution"] == "occupancy"
    assert row["evidence"]["in_flight_requests_max"] == 2
    assert row["evidence"]["available_slots_min"] == 0


def test_predicted_warmth_when_the_backend_reported_no_progress(report):
    """A tiny priced share of a large prompt with zero backend progress."""
    row = _row_for(report, "user-a")
    assert row["attribution"] == "predicted-warmth"
    assert row["evidence"]["priced_ratio"] < 0.01
    assert row["evidence"]["backend_progress_observed"] is False


def test_predicted_warmth_from_a_cluster_measurement(report):
    """A retry of a measured prompt inherits that cluster's measured deficit."""
    row = _row_for(report, "user-d")
    assert row["attribution"] == "predicted-warmth"
    assert row["evidence"]["backend_progress_observed"] is True
    assert row["evidence"]["cluster_measured_new_tokens"] == 35267
    assert row["evidence"]["cluster_measured_deficit_ratio"] >= 4.0


def test_capacity_attribution_for_honestly_priced_cold_dispatch(report):
    """A fully cold price with no other tenant is a GPU-capacity miss."""
    cold = _row_for(report, "user-e")
    assert cold["attribution"] == "capacity"
    assert cold["evidence"]["priced_ratio"] == 1.0
    assert cold["evidence"]["other_session_leases"] == []
    assert _row_for(report, "user-g")["attribution"] == "capacity"


def test_abandoned_prefill_backlog_evidence_is_recorded(report):
    """A dispatch that starts inside an abandoned-prefill window says so."""
    row = _row_for(report, "user-d")
    assert row["evidence"]["abandoned_prefill_overlap"] is True
    assert row["evidence"]["abandoned_prefill_reason"] == "measured_cache_deficit"
    assert row["evidence"]["abandoned_prefill_remaining_s"] > 0


def test_attribution_totals_and_rules_are_reported(report):
    attribution = report["attribution"]
    assert attribution["totals"] == {"occupancy": 3, "capacity": 3, "predicted-warmth": 2}
    assert attribution["by_session_class"]["audit"] == {"capacity": 1}
    assert attribution["abandoned_prefill_overlap"]["total"] == 8
    assert set(attribution["rules"]) == {"occupancy", "capacity", "predicted-warmth"}
    assert attribution["capture_available"] is True
    assert attribution["capture_caveat"] is None


def test_pre_f4_window_without_progress_capture_still_reports(script, tmp_path):
    """No ``prefill_progress_measured`` lines at all (pre-F4): degrade, don't crash."""
    log_file = tmp_path / "proxy.log"
    log_file.write_text(
        "2026-01-01 00:10:00,000 - ERROR - local_eviction_after_acceptance "
        "session=user-1 reason=prefill_deadline estimated_tokens=56000 new_tokens=182 "
        "eta_prefill_s=13.8 elapsed_s=120.1 slot_id=unknown llama_task_id=unknown "
        "endpoint=default\n"
    )
    log = script.parse_window(script.read_log_lines([log_file]), WINDOW_START, WINDOW_END)
    report = script.build_report(log, WINDOW_START, WINDOW_END)

    row = report["events"][0]
    assert row["attribution"] == "predicted-warmth"
    assert row["evidence"]["backend_progress_observed"] is False
    assert row["evidence"]["cluster_measured_new_tokens"] is None
    assert report["attribution"]["totals"]["occupancy"] == 0
    # The report must say the deficit was never measured, not imply it was.
    assert report["attribution"]["capture_available"] is False
    assert "admission price alone" in report["attribution"]["capture_caveat"]


def test_cli_emits_the_json_dataset(script, tmp_path, capsys):
    out = tmp_path / "dataset.json"
    code = script.main(
        [
            "--log-dir",
            str(FIXTURE.parent),
            "--log-glob",
            FIXTURE.name,
            "--start",
            "2026-01-01 00:00:00",
            "--end",
            "2026-01-01 01:00:00",
            "--output",
            str(out),
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["reason_split"]["prefill_deadline"]["total"] == 8
    assert json.loads(out.read_text())["attribution"]["totals"]["occupancy"] == 3


def test_cli_text_summary_states_the_verdict_inputs(script, tmp_path, capsys):
    code = script.main(
        [
            "--log-dir",
            str(FIXTURE.parent),
            "--log-glob",
            FIXTURE.name,
            "--start",
            "2026-01-01 00:00:00",
            "--end",
            "2026-01-01 01:00:00",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "prefill_deadline: 8 (user 7 / audit 1)" in out
    assert "occupancy=3" in out


def test_backend_accounting_flags_abandoned_prefills(script, tmp_path):
    """Optional cross-check: backend-completed prefills vs proxy-consumed streams."""
    llama_log = tmp_path / "llama-server.log"
    llama_log.write_text(
        "".join("slot print_timing: total time = 1.0 ms / 1 tokens\n" for _ in range(5))
        + "slot print_timing: prompt eval time = 10.0 ms / 100 tokens\n"
    )
    log = script.parse_window(script.read_log_lines([FIXTURE]), WINDOW_START, WINDOW_END)
    report = script.build_report(log, WINDOW_START, WINDOW_END, llama_log=str(llama_log))

    accounting = report["backend_prefill_accounting"]
    assert accounting["backend_completed_prefills"] == 1
    assert accounting["proxy_consumed_local_streams"] == 1
    assert accounting["abandoned_prefills"] == 0
