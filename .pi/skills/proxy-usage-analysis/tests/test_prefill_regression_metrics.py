"""Prefill-regression metrics in the usage report (LP-0MVOAQDB4002D56G).

Parent LP-0MVN3KMGI009BZ0D (re-scoped AC4) certifies the cold-prefill admission
budget over a 24 h window with four checks. Before this change three of them
required hand-written log scans because the report exposed only the after-
acceptance eviction counter. This module verifies the additive metrics:

- AC1 ``prefill_deadline_events`` (+ user/audit split; an ``audit-`` session id
  is agent audit traffic, everything else is user traffic);
- AC2 ``prefill_budget_refusals`` (only ``reason=prefill_budget_exceeded``);
- AC3 ``wasted_prefill_tokens`` (cancelled ``new_tokens`` of deadline evictions);
- AC4 ``provider_exhausted_errors`` and ``upstream_status_503_errors``;
- AC5 the markdown report renders the checks and omits the section when empty;
- AC6 strictly additive: the pre-existing keys keep their values.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import aggregation  # noqa: E402
import bucketing  # noqa: E402
import log_parser  # noqa: E402
import reporting  # noqa: E402

WINDOW_START = datetime(2026, 10, 25, 9, 0, 0)
WINDOW_END = datetime(2026, 10, 25, 11, 0, 0)
TS = "2026-10-25 10:00:00,123"
USER_SESSION = "herdr-1792954719-3003559-13442"
AUDIT_SESSION = "audit-LP-0MVN1IQTF0037WTG-phase2_deep-b4f1"


def _analyse(lines: list[str]):
    events = [
        e for e in (log_parser.parse_log_line(ln) for ln in lines) if e is not None
    ]
    schedule = bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])
    return aggregation.aggregate(events, WINDOW_START, WINDOW_END, schedule)


def _deadline_line(session: str | None = USER_SESSION, deadline: str = "120s") -> str:
    suffix = f" session={session}" if session else ""
    return (
        f"{TS} - WARNING - prefill_deadline_exceeded{suffix} deadline={deadline} "
        "\u2192 cancelling local dispatch and detouring to the next provider"
    )


def _guard_line(reason: str = "prefill_budget_exceeded", decision: str = "reject") -> str:
    return (
        f"{TS} - INFO - local_admission_guard decision={decision} reason={reason} "
        "enforced=true eta_prefill_s=0.0 eta_occupancy_s=0.0 projected_wait_s=0.0 "
        "budget_s=60.0 deadline_s=120.0 deadline_budget_s=114.0 new_tokens=25383 "
        "budget_tokens=25000 busy_slots=2 in_flight_prefills=1 free_slot=false "
        "priority=medium policy=reject model=Qwen3 samples=0 provider=local-qwen3 "
        f"session={USER_SESSION}"
    )


def _eviction_line(reason: str = "prefill_deadline", new_tokens: int = 12_734) -> str:
    return (
        f"{TS} - ERROR - local_eviction_after_acceptance session={USER_SESSION} "
        f"reason={reason} estimated_tokens=56580 new_tokens={new_tokens} "
        "eta_prefill_s=201.5 elapsed_s=133.8 slot_id=unknown "
        "llama_task_id=unknown endpoint=http://localhost:8080"
    )


def _exhausted_line() -> str:
    return (
        f"{TS} - WARNING - All providers exhausted for model=v1/chat/completions; "
        "unavailable={'local-qwen3': 59}; usage_limit_reset=None"
    )


def _upstream_line(status: int) -> str:
    return (
        f"{TS} - WARNING - [remote] upstream error status={status} "
        "url=https://api.munito.ai/inference/v1/chat/completions "
        'body={"error":{"message":"Service Unavailable"}}'
    )


# ---------------------------------------------------------------------------
# AC1 — parsing the deadline event
# ---------------------------------------------------------------------------

def test_deadline_line_parses_with_session_and_deadline():
    event = log_parser.parse_log_line(_deadline_line(deadline="120s"))

    assert event is not None
    assert event.kind == "prefill_deadline_exceeded"
    assert event.session == USER_SESSION
    assert event.deadline_seconds == pytest.approx(120.0)


def test_deadline_line_without_session_still_parses():
    event = log_parser.parse_log_line(_deadline_line(session=None))

    assert event is not None
    assert event.kind == "prefill_deadline_exceeded"
    assert event.session is None


def test_exhausted_line_parses_with_model():
    event = log_parser.parse_log_line(_exhausted_line())

    assert event is not None
    assert event.kind == "provider_exhausted"
    assert event.model == "v1/chat/completions"


# ---------------------------------------------------------------------------
# AC1/AC3/AC4 — aggregates in the JSON summary
# ---------------------------------------------------------------------------

def test_deadline_events_are_counted_and_split_user_vs_audit():
    summary = _analyse(
        [
            _deadline_line(USER_SESSION),
            _deadline_line("herdr-1792954719-3003559-13442"),
            _deadline_line(AUDIT_SESSION),
        ]
    )
    payload = reporting.summary_to_json(summary)

    assert payload["prefill_deadline_events"] == 3
    assert payload["prefill_deadline_events_by_class"] == {"user": 2, "audit": 1}


def test_budget_refusals_count_only_the_budget_reason():
    summary = _analyse(
        [
            _guard_line("prefill_budget_exceeded"),
            _guard_line("prefill_budget_exceeded"),
            _guard_line("eta_exceeds_budget"),
            _guard_line("prefill_budget_exceeded", decision="admit"),
        ]
    )
    payload = reporting.summary_to_json(summary)

    assert payload["prefill_budget_refusals"] == 2


def test_wasted_tokens_sum_only_deadline_evictions():
    summary = _analyse(
        [
            _eviction_line("prefill_deadline", new_tokens=12_734),
            _eviction_line("prefill_deadline", new_tokens=4_291),
            _eviction_line("slot_assignment_freed", new_tokens=9_999),
            _eviction_line("no_progress_watchdog", new_tokens=500),
        ]
    )
    payload = reporting.summary_to_json(summary)

    assert payload["wasted_prefill_tokens"] == 12_734 + 4_291
    # Additive only: the pre-existing keys keep their meaning (AC6).
    assert payload["local_evictions_after_acceptance"] == 4
    assert payload["local_evictions_after_acceptance_by_reason"] == {
        "prefill_deadline": 2,
        "slot_assignment_freed": 1,
        "no_progress_watchdog": 1,
    }


def test_provider_exhaustion_and_upstream_503_are_counted_separately():
    summary = _analyse(
        [
            _exhausted_line(),
            _exhausted_line(),
            _upstream_line(503),
            _upstream_line(429),
        ]
    )
    payload = reporting.summary_to_json(summary)

    assert payload["provider_exhausted_errors"] == 2
    assert payload["upstream_status_503_errors"] == 1


def test_all_new_keys_are_zero_on_an_empty_window():
    payload = reporting.summary_to_json(_analyse([]))

    assert payload["prefill_deadline_events"] == 0
    assert payload["prefill_deadline_events_by_class"] == {"user": 0, "audit": 0}
    assert payload["prefill_budget_refusals"] == 0
    assert payload["wasted_prefill_tokens"] == 0
    assert payload["provider_exhausted_errors"] == 0
    assert payload["upstream_status_503_errors"] == 0


# ---------------------------------------------------------------------------
# AC5 — rendered report section
# ---------------------------------------------------------------------------

def test_report_renders_the_prefill_regression_section():
    summary = _analyse(
        [
            _deadline_line(USER_SESSION),
            _deadline_line(AUDIT_SESSION),
            _guard_line(),
            _eviction_line("prefill_deadline", new_tokens=1_000),
            _exhausted_line(),
            _upstream_line(503),
        ]
    )
    report = reporting.build_report(summary, {})

    assert "## Prefill admission regression" in report
    for fragment in (
        "prefill_deadline_exceeded",
        "user",
        "audit",
        "prefill_budget_exceeded",
        "1000",
        "Provider exhaustion",
    ):
        assert fragment in report


def test_report_omits_the_section_when_every_check_is_zero():
    report = reporting.build_report(_analyse([]), {})

    assert "## Prefill admission regression" not in report
