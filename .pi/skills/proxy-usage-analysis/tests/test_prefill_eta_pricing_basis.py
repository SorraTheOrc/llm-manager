"""Measured-basis coverage + p90 in the usage report (LP-0MWRQ6LSV009DVP4).

Parent AC3/AC4: the 24 h report must make visible the share of ETA-priced
admissions whose ``new_tokens`` price was backed by a **measurement** (vs a
prediction), and must publish the p90 as well as the p50 of the prefill-ETA
absolute relative error, so the improvement can be stated as a measured
before/after.

All additions are **additive**: no existing key is renamed, removed or
retyped, and a window with no local admissions renders an explicit ``n/a``
rather than a fabricated ``0``.
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

WINDOW_START = datetime(2026, 11, 21, 5, 0, 0)
WINDOW_END = datetime(2026, 11, 22, 5, 0, 0)
TS = "2026-11-21 10:00:00,123"

SESSION_HIT = "sess-measured"
SESSION_MISS = "sess-predicted"
SESSION_LEGACY = "sess-legacy"


def _guard_line(
    session: str,
    *,
    basis: str | None,
    eta: float = 41.2,
    decision: str = "admit",
    enforced: str = "false",
    ts: str = TS,
) -> str:
    basis_field = f" eta_pricing_basis={basis}" if basis is not None else ""
    return (
        f"{ts} - INFO - local_admission_guard decision={decision} reason=within_budget "
        f"enforced={enforced} eta_prefill_s={eta:.1f} budget_s=60.0 "
        "new_tokens=1000 rate_bucket=solitary busy_slots=0 in_flight_prefills=0 "
        "free_slot=false priority=medium policy=reject model=Qwen3 samples=12 "
        f"provider=local-qwen3 session={session}{basis_field}"
    )


def _first_byte_line(session: str, ms: float = 8000.0, ts: str = TS) -> str:
    return (
        f"{ts} - INFO - dispatch_first_byte_ms={ms} "
        f"dispatch_to_first_byte_ms={ms} cache_state=cold cached_ratio=0.000 "
        f"queue_wait_ms=3.3 session={session} model=Qwen3 priority=high "
        "estimated_tokens=22726"
    )


STATUS_LINE = (
    "2026-11-21 10:05:00,000 - INFO - status_request active_query=false "
    "available_slots=2 capacity_hint=idle prefill_eta_seconds=91.6 "
    "queue_eta_seconds=183.2 status_schema_version=3 total_slots=2"
)


def _analyse(lines: list[str]):
    events = [
        e for e in (log_parser.parse_log_line(ln) for ln in lines) if e is not None
    ]
    schedule = bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])
    return aggregation.aggregate(events, WINDOW_START, WINDOW_END, schedule)


# ---------------------------------------------------------------------------
# AC3 — the parser carries the pricing basis
# ---------------------------------------------------------------------------

def test_guard_parser_carries_pricing_basis():
    measured = log_parser.parse_log_line(_guard_line(SESSION_HIT, basis="measured"))
    predicted = log_parser.parse_log_line(_guard_line(SESSION_MISS, basis="predicted"))
    legacy = log_parser.parse_log_line(_guard_line(SESSION_LEGACY, basis=None))

    assert measured is not None and measured.eta_pricing_basis == "measured"
    assert predicted is not None and predicted.eta_pricing_basis == "predicted"
    assert legacy is not None and legacy.eta_pricing_basis is None


# ---------------------------------------------------------------------------
# AC3 — the basis split
# ---------------------------------------------------------------------------

def test_basis_share_counts_measured_and_predicted():
    summary = _analyse(
        [
            _guard_line(SESSION_HIT, basis="measured"),
            _guard_line(SESSION_MISS, basis="predicted"),
            _guard_line(SESSION_LEGACY, basis=None),
        ]
    )
    share = aggregation.prefill_eta_basis_share(summary)

    assert share["measured"] == 1
    assert share["predicted"] == 1
    assert share["unknown"] == 1
    assert share["total"] == 3
    # share is over the classified (measured + predicted) population.
    assert share["measured_share_pct"] == 50.0


def test_basis_share_empty_window_is_na_not_zero():
    share = aggregation.prefill_eta_basis_share(_analyse([STATUS_LINE]))
    assert share["total"] == 0
    assert share["measured_share_pct"] is None


def test_basis_share_ignores_enforced_refusals():
    summary = _analyse(
        [
            _guard_line(
                SESSION_HIT, basis="measured", decision="reject", enforced="true"
            ),
            _guard_line(SESSION_MISS, basis="predicted"),
        ]
    )
    share = aggregation.prefill_eta_basis_share(summary)
    # The enforced refusal never dispatched; it is not an ETA-priced admission.
    assert share["measured"] == 0
    assert share["predicted"] == 1


# ---------------------------------------------------------------------------
# AC4 — p90 alongside p50
# ---------------------------------------------------------------------------

def test_accuracy_publishes_p90_on_matched_pairs():
    lines = [
        _guard_line(SESSION_MISS, basis="predicted", eta=40.0),
        _first_byte_line(SESSION_MISS, ms=8000.0),  # error 400 %
    ]
    summary = _analyse(lines)
    p50 = aggregation.prefill_eta_accuracy(summary)
    p90 = aggregation.prefill_eta_accuracy_p90(summary)
    assert p50 is not None
    assert p90 is not None
    assert p90 >= p50


def test_accuracy_p90_none_without_pairs():
    assert aggregation.prefill_eta_accuracy_p90(_analyse([STATUS_LINE])) is None


# ---------------------------------------------------------------------------
# AC3/AC4 — the JSON is additive and carries the new keys
# ---------------------------------------------------------------------------

def test_json_adds_basis_split_and_accuracy_without_touching_existing_keys():
    summary = _analyse(
        [
            _guard_line(SESSION_HIT, basis="measured"),
            _first_byte_line(SESSION_HIT, ms=8000.0),
            STATUS_LINE,
        ]
    )
    block = reporting.prefill_eta_error_json(summary, summary.speed)

    # Existing keys are untouched.
    for key in (
        "comparison", "predicted", "observed", "error_p50", "error_p90",
        "relative_error_p50_pct", "relative_error_p90_pct", "tolerance_pct",
        "bound", "verdict", "matched",
    ):
        assert key in block
    # Additive keys are present and shaped.
    assert block["basis_split"]["measured"] == 1
    assert "measured_share_pct" in block["basis_split"]
    assert set(block["accuracy"]) == {"samples", "p50_pct", "p90_pct"}


def test_rendered_section_shows_basis_share_row():
    summary = _analyse(
        [
            _guard_line(SESSION_HIT, basis="measured"),
            _first_byte_line(SESSION_HIT, ms=8000.0),
            STATUS_LINE,
        ]
    )
    lines: list[str] = []
    reporting._append_contention_prediction_section(
        lines.append, summary, summary.speed
    )
    text = "\n".join(lines)
    assert "ETA pricing basis" in text
    assert "Measured-basis share" in text
    assert "Absolute relative error p50 / p90" in text
    # The parent's 2.3 % baseline is quoted for context.
    assert "2.3 %" in text
