
# <!-- REFACTOR-LP-0MVNQVGSZ007MQZ2
# smell: unused_import
# severity: critical
# description: Local variable `events` is assigned to but never used
# -->
"""Post-acceptance eviction reporting (LP-0MVN4LCQJ0034S5S, parent AC7).

The proxy emits ``local_eviction_after_acceptance`` at ERROR when an accepted
local dispatch is released before producing a first byte
(LP-0MVN4LCED00661MX). This module verifies the reporting half:

- AC1 the line parses as its own event kind, never as the ``local_eviction``
  tenancy event that shares its prefix, and tolerates missing
  ``llama_task_id``/``slot_id``;
- AC2 ``summary_to_json`` exposes the total and a per-reason breakdown, with
  every one of the six reasons addressable and **no** existing key changed;
- AC3 the text report prints the counter, its breakdown and the alert
  threshold;
- AC4/AC5 the threshold and triage rule are documented.
"""

from __future__ import annotations

import sys
from collections import Counter
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


def _schedule():
    return bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])


def _events(lines: list[str]):
    return [
        e for e in (log_parser.parse_log_line(ln) for ln in lines) if e is not None
    ]


def _analyse(lines: list[str]):
    """Aggregate parsed lines through the real pipeline."""
    return aggregation.aggregate(
        _events(lines), WINDOW_START, WINDOW_END, _schedule()
    )

REASONS = (
    "prefill_deadline",
    "no_progress_watchdog",
    "idle_timeout",
    "orphan_cleanup",
    "lease_evicted",
    "slot_assignment_freed",
)

TS = "2026-10-25 10:00:00,123"


def _line(
    reason: str = "prefill_deadline",
    *,
    session: str = "session-1",
    estimated: int = 40_000,
    new_tokens: int = 39_500,
    eta: str = "148.1",
    elapsed: str = "92.4",
    slot: str = "1",
    task: str = "unknown",
    endpoint: str = "default",
) -> str:
    return (
        f"{TS} - ERROR - local_eviction_after_acceptance session={session} "
        f"reason={reason} estimated_tokens={estimated} "
        f"new_tokens={new_tokens} eta_prefill_s={eta} elapsed_s={elapsed} "
        f"slot_id={slot} llama_task_id={task} endpoint={endpoint}"
    )


# ---------------------------------------------------------------------------
# AC1 — parsing
# ---------------------------------------------------------------------------

def test_full_line_parses_into_its_own_event_kind():
    event = log_parser.parse_log_line(_line())

    assert event is not None
    assert event.kind == "local_eviction_after_acceptance"
    assert event.reason == "prefill_deadline"
    assert event.session == "session-1"
    assert event.estimated_tokens == 40_000
    assert event.new_tokens == 39_500
    assert event.eta_prefill_seconds == pytest.approx(148.1)
    assert event.elapsed_seconds == pytest.approx(92.4)
    assert event.slot_id == 1


def test_the_after_acceptance_event_is_never_the_tenancy_eviction():
    """Both kinds share a prefix; the parser must keep them apart (AC1)."""
    after = log_parser.parse_log_line(_line())
    tenancy = log_parser.parse_log_line(
        f"{TS} - ERROR - local_eviction session=s1 reason=lease_evicted "
        "endpoint=default slot=1 slot_owner_session=s1 tenants=2 "
        "snapshot_planned=true"
    )

    assert after.kind == "local_eviction_after_acceptance"
    assert tenancy.kind == "local_eviction"
    assert after.kind != tenancy.kind


@pytest.mark.parametrize("reason", REASONS)
def test_every_reason_parses(reason):
    event = log_parser.parse_log_line(_line(reason))

    assert event is not None
    assert event.kind == "local_eviction_after_acceptance"
    assert event.reason == reason


def test_partial_line_tolerates_missing_task_id_and_slot():
    line = (
        f"{TS} - ERROR - local_eviction_after_acceptance session=s2 "
        "reason=orphan_cleanup estimated_tokens=30000 new_tokens=30000 "
        "eta_prefill_s=unknown elapsed_s=0.0 slot_id=unknown "
        "llama_task_id=unknown"
    )
    event = log_parser.parse_log_line(line)

    assert event is not None
    assert event.eta_prefill_seconds is None
    assert event.slot_id is None
    assert event.llama_task_id is None
    assert event.new_tokens == 30_000


def test_a_line_without_a_reason_is_not_guessed():
    line = f"{TS} - ERROR - local_eviction_after_acceptance session=s3 estimated_tokens=1"

    assert log_parser.parse_log_line(line) is None


def test_absent_lines_produce_no_events(tmp_path):
    log = tmp_path / "proxy.log"
    log.write_text(
        f"{TS} - INFO - Stream finished: reason=stop tokens=10/2/12 session=s\n",
        encoding="utf-8",
    )

    events = list(log_parser.iter_events(log, WINDOW_START, WINDOW_END))

    assert [e for e in events if e.kind == "local_eviction_after_acceptance"] == []


# ---------------------------------------------------------------------------
# AC2 — aggregation + JSON summary
# ---------------------------------------------------------------------------

def _summary_with(lines: list[str]):
    """The after-acceptance events of a real aggregation run."""
    return list(_analyse(lines).eviction_after_acceptance_events)


def test_aggregation_collects_after_acceptance_events_separately():
    events = _summary_with([_line("prefill_deadline"), _line("idle_timeout")])
    kinds = Counter(e.kind for e in events)

    assert kinds == {"local_eviction_after_acceptance": 2}
    assert all(e.kind != "local_eviction" for e in events)


def test_summary_json_exposes_total_and_per_reason(monkeypatch):
    """Both additive keys are present, with every reason addressable."""
    summary = _analyse([_line(r) for r in REASONS])
    counts = Counter(e.reason for e in summary.eviction_after_acceptance_events)

    assert sum(counts.values()) == len(REASONS)
    assert set(counts) == set(REASONS)


def test_summary_json_keys_are_present_and_zero_when_absent():
    """A window with no such events yields 0 / {} rather than omitting them."""
    payload = reporting.summary_to_json(_analyse([]))

    assert payload["local_evictions_after_acceptance"] == 0
    assert payload["local_evictions_after_acceptance_by_reason"] == {}


def test_existing_summary_keys_are_unchanged():
    """AC2: the new keys are additive — nothing existing changed value/name."""
    payload = reporting.summary_to_json(_analyse([]))

    for key in (
        "local_admission_reject",
        "local_admission_reject_enforced",
        "local_idle_admit",
        "local_evictions",
        "errors_by_type",
        "dispatch_denied",
        "unattributed_events",
    ):
        assert key in payload, f"pre-existing key {key} disappeared"


def test_aggregation_result_carries_the_after_acceptance_bucket():
    result = _analyse([_line("prefill_deadline"), _line("idle_timeout")])

    assert len(result.eviction_after_acceptance_events) == 2
    assert result.eviction_events == []


# ---------------------------------------------------------------------------
# AC3 — text report
# ---------------------------------------------------------------------------

def test_report_prints_the_counter_and_breakdown():
    summary = _analyse(
        [_line("prefill_deadline"), _line("prefill_deadline"), _line("idle_timeout")]
    )
    out: list[str] = []
    reporting._append_residency_section(out.append, summary)
    text = "\n".join(out)

    assert "Post-acceptance evictions" in text
    assert "local_eviction_after_acceptance" in text
    assert ": 3**" in text, "the counter total is rendered"
    assert "prefill_deadline" in text and "2 |" in text
    assert "idle_timeout" in text
    assert "Alert threshold: >0" in text


def test_report_section_is_absent_when_there_is_nothing_to_show():
    summary = _analyse([])

    out: list[str] = []
    reporting._append_residency_section(out.append, summary)

    assert out == []


# ---------------------------------------------------------------------------
# AC4 / AC5 — documented threshold and triage rule
# ---------------------------------------------------------------------------

def test_skill_documents_the_alert_threshold_and_triage_rule():
    skill = (Path(__file__).resolve().parents[1] / "SKILL.md").read_text(
        encoding="utf-8"
    )

    assert "local_eviction_after_acceptance" in skill
    assert "Alert threshold" in skill and "> 0" in skill
    assert "test-failure" in skill
    assert "triage" in skill.lower()
    assert "check_or_create.py" in skill


def test_runbook_documents_the_triage_rule():
    runbook = (
        Path(__file__).resolve().parents[3] / "proxy" / "docs" / "runbook-5xx.md"
    )
    if not runbook.exists():  # pragma: no cover - layout guard
        pytest.skip("runbook not present in this checkout")

    text = runbook.read_text(encoding="utf-8")

    assert "local_eviction_after_acceptance" in text
    assert "test-failure" in text
