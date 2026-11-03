"""Chain-end taxonomy: hold vs fail-fast (LP-0MW0KZR7A003UFQK AC3).

A model whose whole fallback chain cannot serve ends one of two ways, and the
report must classify them distinctly:

- ``Chain exhausted for model=<p>; holding <N>s before restarting cycle ...``
  — the bounded hold. Real latency.
- ``chain_planned_unavailable ... reason=deterministic_residual_chain
  hold=skipped ...`` — the fail-fast refusal (every remaining provider
  deterministically skipped or terminally refused). **No** hold latency.

Before the fail-fast existed, the second shape was indistinguishable from the
first in the report: it consumed the full ``chain_hold_seconds x
chain_hold_max_cycles`` budget and the hold line was all that remained
(measured 2026-11-03 08:41: three 300 s holds on a chain where every provider
was skipped for a reason hours-to-days away, which burned an unattended audit
run's per-call budget).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import aggregation  # noqa: E402
import bucketing  # noqa: E402
import log_parser  # noqa: E402
import reporting  # noqa: E402

WINDOW_START = datetime(2026, 11, 3, 8, 0, 0)
WINDOW_END = datetime(2026, 11, 3, 10, 0, 0)
TS = "2026-11-03 08:41:42,000"


def _analyse(lines: list[str]):
    events = [
        e for e in (log_parser.parse_log_line(ln) for ln in lines) if e is not None
    ]
    schedule = bucketing.schedule_from_entries([("00:00", 8)])
    return aggregation.aggregate(events, WINDOW_START, WINDOW_END, schedule)


def _hold_line(seconds: float = 300.0, cycle: int = 0) -> str:
    return (
        f"{TS} - WARNING - Chain exhausted for model=v1/chat/completions; holding "
        f"{seconds:.0f}s before restarting cycle from the first provider "
        f"(chain_hold_cycle={cycle}, cycle={cycle}, max_cycles=3)"
    )


def _fail_fast_line(retry_after: int | None = 90) -> str:
    retry = "" if retry_after is None else f" retry_after={retry_after}"
    return (
        f"{TS} - INFO - chain_planned_unavailable model=v1/chat/completions "
        f"reason=deterministic_residual_chain hold=skipped{retry} providers=6 "
        "deterministic_skips=munitoai-free:usage_limit_reset_pending,"
        "opencode-go-3:quota_exhausted terminal=none"
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_hold_line_parses_as_a_hold():
    event = log_parser.parse_log_line(_hold_line(seconds=300))

    assert event is not None
    assert event.kind == "chain_hold"
    assert event.hold == "taken"
    assert event.hold_seconds == 300.0
    assert event.model == "v1/chat/completions"


def test_fail_fast_line_parses_as_its_own_kind():
    event = log_parser.parse_log_line(_fail_fast_line(retry_after=90))

    assert event is not None
    assert event.kind == "chain_planned_unavailable"
    assert event.hold == "skipped"
    assert event.reason == "deterministic_residual_chain"
    assert event.retry_after == 90
    assert event.model == "v1/chat/completions"


def test_the_two_chain_end_kinds_never_collapse():
    hold = log_parser.parse_log_line(_hold_line())
    fail_fast = log_parser.parse_log_line(_fail_fast_line())

    assert hold.kind != fail_fast.kind
    assert hold.hold == "taken"
    assert fail_fast.hold == "skipped"


def test_a_fail_fast_without_a_known_reset_keeps_retry_after_null():
    event = log_parser.parse_log_line(_fail_fast_line(retry_after=None))

    assert event is not None
    assert event.retry_after is None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_aggregation_keeps_the_two_outcomes_apart():
    summary = _analyse(
        [_hold_line(), _hold_line(cycle=1), _fail_fast_line(), _fail_fast_line()]
    )

    assert len(summary.chain_hold_events) == 2
    assert len(summary.chain_planned_unavailable_events) == 2


def test_a_window_with_only_a_fail_fast_reports_no_hold():
    summary = _analyse([_fail_fast_line()])

    assert summary.chain_hold_events == []
    assert len(summary.chain_planned_unavailable_events) == 1


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_report_states_that_a_fail_fast_is_not_hold_latency():
    summary = _analyse([_hold_line(), _fail_fast_line()])
    lines: list[str] = []

    reporting._append_chain_end_section(lines.append, summary)
    text = "\n".join(lines)

    assert "## Chain-end outcomes" in text
    assert "`chain_hold` | 1" in text
    assert "`chain_planned_unavailable` | 1" in text
    assert "never counted as one" in text
    assert "deterministic_residual_chain" in text


def test_report_omits_the_section_without_a_chain_end_event():
    summary = _analyse([])
    lines: list[str] = []

    reporting._append_chain_end_section(lines.append, summary)

    assert lines == []
