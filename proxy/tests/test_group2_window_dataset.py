"""Integrity pin for the group-2 restart-free measurement deliverable.

Work item LP-0MW8EAU8P0079280 (AC3/AC4 of the group-2 validation epic
LP-0MW7V14DN007F127). The deliverable is *evidence*, not behaviour: a committed
dataset (`docs/dev/prefill-deadline-residual-group2-24h.json`) and its
Markdown report. Nothing else covers those two files, so a later hand edit —
re-widowing the measurement, dropping the ``--llama-log`` basis back to
``legacy_rate_estimate``, or deleting the quoted command — would silently
change what the window certifies.

Pattern: ``test_contention_queue_docs.py`` (committed-artefact drift pin). The
numbers below are **recomputed from the recorded payload** wherever possible
(``window.hours``, per-reason counts, the (b2) composite) so this test pins
*self-consistency*, not a frozen copy of the report's prose.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "docs" / "dev" / "prefill-deadline-residual-group2-24h.json"
REPORT_PATH = REPO_ROOT / "docs" / "dev" / "prefill-deadline-residual-group2-24h.md"

#: The window is anchored on the single group-2 restart
#: (``docs/dev/group-2-restart.md``): proxy PID 3720326, started
#: 2026-11-16T15:20:47Z. A dataset whose start moves is measuring a different
#: window and must not reuse this path.
WINDOW_START = "2026-11-16 15:20:47"

#: AC3's floor. The measured span may be longer (this window is 26.987 h).
MIN_WINDOW_HOURS = 24.0

#: The exact ``--llama-log`` flag value AC3/AC4 require in the invoked command.
LLAMA_LOG_FLAG = '--llama-log "/var/log/llama-proxy/llama-server*.log"'

#: The (b2) certified criterion's two reasons plus the certified rate.
B2_REASONS = ("prefill_deadline", "measured_cache_deficit")
B2_RATE_GATE_PER_HOUR = 0.5


def _dataset() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _report() -> str:
    return REPORT_PATH.read_text(encoding="utf-8")


def test_dataset_and_report_are_committed() -> None:
    assert DATASET_PATH.is_file(), f"missing committed dataset: {DATASET_PATH}"
    assert REPORT_PATH.is_file(), f"missing committed report: {REPORT_PATH}"


def test_window_is_anchored_on_the_group2_restart_and_at_least_24h() -> None:
    window = _dataset()["window"]
    assert window["start"] == WINDOW_START
    assert window["hours"] >= MIN_WINDOW_HOURS
    # ``hours`` is the rounded span the analyser derived from start/end, so a
    # hand-edited end time that kept the old hours would fail here.
    assert round(window["hours"], 2) == 26.99


def test_dataset_records_no_restart_other_than_the_windows_own_boot() -> None:
    source = _dataset()["source"]
    restarts = source["restarts_in_window"]
    assert len(restarts) == 1, restarts
    # The single entry is the window's own log rotation, not a re-base.
    assert "Rotated previous proxy log aside" in restarts[0]["line"]
    assert restarts[0]["ts"] == "2026-11-16 15:20:48"


def test_abandoned_window_basis_comes_from_the_llama_log_not_the_legacy_estimate() -> None:
    payload = _dataset()
    abandoned = payload["abandoned_prefill_windows"]
    assert abandoned["basis"] == "measured_stop"
    assert abandoned["measured_available"] is True
    assert abandoned["coverage"]["available"] is True
    assert abandoned["coverage"]["llama_log"] == "/var/log/llama-proxy/llama-server*.log"
    # The overlap verdict the report quotes rests on the same basis.
    assert payload["attribution"]["abandoned_prefill_overlap"]["basis"] == "measured_stop"
    assert payload["release_cancel_accounting"]["measured_available"] is True
    assert payload["seconds_basis"]["tok_s"] > 0


def test_eviction_totals_agree_with_the_per_reason_split() -> None:
    payload = _dataset()
    by_reason = payload["evictions"]["by_reason"]
    split = payload["reason_split"]
    assert sum(by_reason.values()) == payload["evictions"]["total"]
    assert set(by_reason) == set(split)
    for reason, total in by_reason.items():
        row = split[reason]
        assert row["user"] + row["audit"] == total, reason
        assert row["total"] == total, reason


def test_b2_composite_and_stall_rate_are_reproducible_from_the_payload() -> None:
    payload = _dataset()
    hours = payload["window"]["hours"]
    user_events = sum(payload["reason_split"][reason]["user"] for reason in B2_REASONS)
    assert user_events == 4  # 1 prefill_deadline + 3 measured_cache_deficit

    combined_rate = user_events / hours
    assert combined_rate < B2_RATE_GATE_PER_HOUR
    # The published per-reason rates must be the recorded counts over the
    # recorded hours, within the payload's 3-decimal rounding.
    per_reason_rate = sum(
        payload["reason_split"][reason]["user_per_hour"] for reason in B2_REASONS
    )
    assert abs(per_reason_rate - combined_rate) < 1e-3

    stalls = payload["stall_cost"]["by_reason"]
    user_stall_s = sum(
        stalls[reason]["by_session_class"]["user"]["stall_seconds"] for reason in B2_REASONS
    )
    measured_rate = sum(
        stalls[reason]["by_session_class"]["user"]["stall_seconds_per_hour"]
        for reason in B2_REASONS
    )
    # A detour and a full-deadline stall must stay distinguishable in the
    # payload: the deficit leg is seconds, the deadline leg is the backstop.
    assert stalls["measured_cache_deficit"]["by_session_class"]["user"]["percentiles"]["p50"] < 1.0
    assert (
        stalls["prefill_deadline"]["by_session_class"]["user"]["stall_seconds"] == 180.1
    )
    assert abs(measured_rate - user_stall_s / hours) < 1e-3


def test_report_quotes_the_exact_command_and_the_basis_assertion() -> None:
    report = _report()
    assert LLAMA_LOG_FLAG in report
    assert '--start "2026-11-16 15:20:47"' in report
    assert '--output docs/dev/prefill-deadline-residual-group2-24h.json' in report
    # AC3's wording is explicit: the report must say the basis is the measured
    # backend stop, never the legacy rate estimate.
    assert "measured_stop" in report
    assert "legacy_rate_estimate" in report
    # The window record AC3 asks for (start, end, hours) is in the document.
    assert "26.987 h" in report
    assert "2026-11-17 18:20:00" in report
