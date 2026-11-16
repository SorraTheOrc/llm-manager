"""Stall-weighted prefill-deadline rate in the daily usage report
(LP-0MW7V30AZ007JB5Z).

Extends the wasted-GPU-compute surface (LP-0MW2SQNHJ0044PV6) with the figure
the (b2) event rate cannot produce: seconds of user waiting per hour. The
report already reuses the committed analyser, so this module only has to prove
that

- the fixture window's two reasons are priced *differently* even though the
  event rate counts them identically (AC1/AC2);
- the summary, the detail table and the appendix render, in the right order,
  with the definitions appendix stating the non-equivalence and quoting the
  measured p50/p90 (AC3);
- ``--json`` gains a ``stall_cost`` block and every pre-existing key keeps its
  name, meaning and type (AC4);
- the rendered section is pinned by a golden file (AC7).

Every expectation is derived from the fixture by hand, never read back from the
implementation.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SCRIPTS = TESTS_DIR.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import aggregation  # noqa: E402
import bucketing  # noqa: E402
import reporting  # noqa: E402
import wasted_compute  # noqa: E402

GOLDEN = TESTS_DIR / "golden" / "wasted_compute_stall_cost_report.md"

WINDOW_START = datetime(2026, 11, 4, 16, 0, 0)
WINDOW_END = datetime(2026, 11, 5, 2, 0, 0)  # 10 h
EXPECTED_HOURS = 10.0

#: The fixture's deadline events stall for the full backstop; the deficit abort
#: gives up in 0.4 s. Two of the deadline events are user, one is audit.
DEADLINE_USER_STALL_S = 360.2
DEFICIT_USER_STALL_S = 0.4


def _eviction(
    ts: str,
    session: str,
    reason: str,
    *,
    estimated_tokens: int = 60_000,
    new_tokens: int = 3_000,
    elapsed_s: float = 180.1,
    eta_prefill_s: str = "95.0",
) -> str:
    return (
        f"{ts} - ERROR - local_eviction_after_acceptance session={session} "
        f"reason={reason} estimated_tokens={estimated_tokens} "
        f"new_tokens={new_tokens} eta_prefill_s={eta_prefill_s} "
        f"elapsed_s={elapsed_s} slot_id=unknown llama_task_id=unknown "
        "endpoint=default"
    )


#: Two user deadline stalls (180.1 s each), one audit deadline stall, one user
#: cache-deficit abort at 0.4 s and one user backend-restart eviction at 3.1 s.
WINDOW_LINES = [
    _eviction("2026-11-04 17:10:00,000", "herdr-1", "prefill_deadline"),
    _eviction("2026-11-04 18:20:00,000", "herdr-2", "prefill_deadline"),
    _eviction("2026-11-04 19:30:00,000", "audit-1", "prefill_deadline"),
    _eviction(
        "2026-11-04 20:40:00,000",
        "herdr-3",
        "measured_cache_deficit",
        estimated_tokens=40_000,
        new_tokens=400,
        elapsed_s=0.4,
    ),
    _eviction(
        "2026-11-04 21:50:00,000",
        "herdr-4",
        "backend_restart",
        estimated_tokens=32_347,
        new_tokens=16_816,
        elapsed_s=3.1,
    ),
]


def _write_window(tmp_path: Path) -> Path:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "proxy.log").write_text(
        "\n".join(WINDOW_LINES) + "\n", encoding="utf-8"
    )
    return log_dir


def _analyse(log_dir: Path, **kwargs) -> wasted_compute.WastedCompute:
    return wasted_compute.analyse_window(
        log_dir, WINDOW_START, WINDOW_END, llama_log_dir=log_dir, **kwargs
    )


def _empty_summary():
    schedule = bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])
    return aggregation.aggregate([], WINDOW_START, WINDOW_END, schedule)


@pytest.fixture(name="result")
def _result(tmp_path):
    return _analyse(_write_window(tmp_path))


# ---------------------------------------------------------------------------
# AC1/AC2 — the stall cost is normalised from the analyser's payload
# ---------------------------------------------------------------------------


def test_stall_cost_is_normalised_per_reason_and_class(result):
    assert result.status == "ok", result.reason
    stall = result.stall_cost
    assert stall is not None
    assert stall.hours == pytest.approx(EXPECTED_HOURS)
    assert stall.deadline_seconds == pytest.approx(180.0)
    assert stall.total_events == 5

    deadline = stall.deadline_row
    assert deadline is not None
    assert deadline.events == 3
    assert (deadline.user_events, deadline.audit_events) == (2, 1)
    assert deadline.user_stall_seconds == pytest.approx(DEADLINE_USER_STALL_S, abs=0.01)
    # 360.2 s over a 10 h window.
    assert deadline.user_stall_seconds_per_hour == pytest.approx(36.02, abs=0.01)
    assert deadline.user_percentiles["p50"] == pytest.approx(180.1)

    deficit = stall.deficit_row
    assert deficit is not None
    assert deficit.user_stall_seconds == pytest.approx(DEFICIT_USER_STALL_S, abs=0.01)
    assert deficit.user_stall_seconds_per_hour == pytest.approx(0.04, abs=0.001)
    assert deficit.user_buckets["under_1s"] == {"count": 1, "share": 1.0}
    # The whole reason's waiting is 900x smaller than the deadline stall while
    # the event counts are 1 and 2 — the contrast the new figure exists for.
    assert deficit.user_stall_seconds * 100 < deadline.user_stall_seconds


def test_event_rate_is_unchanged_beside_the_stall_rate(result):
    """AC1: the stall figure is an addition, not a replacement."""
    # 2 user deadline + 1 user cache-deficit = the (b2) composite, 3 events.
    assert result.composite_user_events == 3
    assert result.composite_user_per_hour == pytest.approx(0.3)
    # Same population, priced in seconds.
    assert result.stall_cost.composite_user_events == 3
    assert result.stall_cost.composite_user_stall_seconds == pytest.approx(
        DEADLINE_USER_STALL_S + DEFICIT_USER_STALL_S, abs=0.01
    )
    assert result.stall_cost.composite_user_stall_seconds_per_hour == pytest.approx(
        36.06, abs=0.01
    )


def test_composite_percentiles_agree_with_the_analyser_arithmetic(result):
    """The composite uses the same interpolation the analyser published with."""
    stall = result.stall_cost
    # Three samples, 0.4 + 180.1 + 180.1: the median is a deadline stall.
    assert stall.composite_user_percentiles["count"] == 3
    assert stall.composite_user_percentiles["p50"] == pytest.approx(180.1)
    assert stall.composite_user_percentiles["p90"] == pytest.approx(180.1)
    assert stall.composite_user_percentiles["max"] == pytest.approx(180.1)
    # And it agrees with the analyser's own figure for a single-reason
    # population (the deficit abort alone), proving one interpolation rule.
    assert stall.deficit_row.user_percentiles["p50"] == pytest.approx(0.4)


def test_stall_buckets_use_the_reported_deadline(result):
    stall = result.stall_cost
    assert stall.composite_user_buckets["at_or_near_deadline"] == {
        "count": 2,
        "share": 0.6667,
        "threshold_s": 179.0,
    }
    assert stall.composite_user_buckets["under_1s"] == {"count": 1, "share": 0.3333}


def test_a_payload_without_a_stall_block_degrades_to_none():
    """An analyser payload from before this surface existed must not crash."""
    assert wasted_compute.normalise_stall_cost(None, hours=24.0) is None
    assert wasted_compute.normalise_stall_cost({}, hours=24.0) is None
    dataset = {
        "window": {"hours": 24.0},
        "source": {"restarts_in_window": []},
        "evictions": {"total": 0, "by_reason": {}, "per_hour": 0.0},
        "reason_split": {},
        "attribution": {
            "capture_available": True,
            "totals": {"occupancy": 0, "capacity": 0, "predicted-warmth": 0},
            "abandoned_prefill_overlap": {"events": 0, "total": 0},
        },
        "events": [],
    }
    assert wasted_compute.normalise(dataset).stall_cost is None


# ---------------------------------------------------------------------------
# AC1/AC2/AC3 — rendering
# ---------------------------------------------------------------------------


def test_summary_and_detail_state_the_stall_weighted_rate(result):
    markdown = reporting.build_report(_empty_summary(), None, wasted=result)

    # Summary row sits in the headline table, next to the per-event rate.
    assert "User-class prefill **stall**" in markdown
    assert "36.1 s/h" in markdown

    # Detail section names each reason with its percentiles and buckets.
    assert "#### Stall cost (what the event rate cannot see)" in markdown
    assert "| `prefill_deadline` | 3 (2 / 1) | 360.2 s | 36.0 s/h |" in markdown
    assert "| `measured_cache_deficit` | 1 (1 / 0) | 0.4 s | 0.0 s/h |" in markdown
    assert "**User-class composite** (the (b2) population, stall-weighted):" in markdown
    assert "**36.1 s/h**" in markdown


def test_appendix_states_the_non_equivalence_with_measured_evidence(result):
    """AC3: definitions state the difference and quote this window's p50/p90."""
    markdown = reporting.build_report(_empty_summary(), None, wasted=result)
    appendix = markdown[markdown.index("## Appendix: wasted GPU compute") :]

    assert (
        "**`measured_cache_deficit` and `prefill_deadline` are not equivalent.**"
        in appendix
    )
    # The *why*: early detour versus a full-deadline wait.
    assert "the early detector **working**" in appendix
    assert "waiting the whole backstop" in appendix
    # The measured evidence, per reason, in one sentence.
    assert "`measured_cache_deficit` p50 0.4 s / p90 0.4 s over 1 event(s)" in appendix
    assert "`prefill_deadline` p50 180.1 s / p90 180.1 s over 2 event(s)" in appendix
    assert "1 (100%) of them under 1 s" in appendix
    # And the instruction on how to read the pair.
    assert "whether the defect is occurring" in appendix
    assert "what it costs" in appendix
    # The definitions table gained the stall terms.
    assert "| Stall-weighted rate |" in appendix
    assert "| Stall percentiles |" in appendix


def test_detail_section_is_omitted_when_there_is_no_stall_block(tmp_path):
    """A payload without the block renders the old report, not zeros."""
    result = _analyse(_write_window(tmp_path))
    without = wasted_compute.WastedCompute(
        **{**_as_kwargs(result), "stall_cost": None}
    )
    markdown = reporting.build_report(_empty_summary(), None, wasted=without)
    assert "#### Stall cost" not in markdown
    assert "User-class prefill **stall**" not in markdown


def _as_kwargs(result: wasted_compute.WastedCompute) -> dict:
    """Dataclass field values of *result* (so a variant can be constructed)."""
    from dataclasses import fields

    return {field.name: getattr(result, field.name) for field in fields(result)}


# ---------------------------------------------------------------------------
# AC4 — additive JSON
# ---------------------------------------------------------------------------


def test_json_gains_a_stall_cost_block_and_keeps_the_old_keys(result):
    data = reporting.summary_to_json(_empty_summary(), None, None, wasted=result)
    block = data["wasted_compute"]

    stall = block["stall_cost"]
    assert stall["hours"] == pytest.approx(EXPECTED_HOURS)
    assert stall["deadline_seconds"] == pytest.approx(180.0)
    assert stall["composite_user_events"] == 3
    assert stall["composite_user_stall_seconds"] == pytest.approx(360.6, abs=0.01)
    assert stall["composite_user_stall_seconds_per_hour"] == pytest.approx(36.06, abs=0.01)
    assert stall["deadline_user_percentiles"]["p50"] == pytest.approx(180.1)
    reasons = {row["reason"]: row for row in stall["reasons"]}
    assert reasons["prefill_deadline"]["user_stall_seconds_per_hour"] == pytest.approx(
        36.02, abs=0.01
    )
    assert reasons["measured_cache_deficit"]["user_buckets"]["under_1s"] == {
        "count": 1,
        "share": 1.0,
    }

    # AC4 — the pre-existing keys keep their name and meaning.
    assert block["composite_user_events"] == 3
    assert block["composite_user_per_hour"] == pytest.approx(0.3)
    assert block["total_events"] == 5
    assert block["waste"]["wasted_tokens"] == 9_000
    assert block["verdict"] == "inconclusive"


def test_json_is_serialisable_and_stall_cost_is_null_without_the_analysis():
    import json as _json

    data = reporting.summary_to_json(_empty_summary(), None, None)
    assert data["wasted_compute"] is None
    _json.dumps(data)


# ---------------------------------------------------------------------------
# AC7 — golden file for the rendered section
# ---------------------------------------------------------------------------


def _wasted_compute_sections(markdown: str) -> str:
    """The rendered wasted-compute surface, in document order.

    The three renderings are non-contiguous by design (summary near the top,
    detail in the body, appendix last), so the golden file holds them
    concatenated in the order they appear — a change to any of them, or to
    their relative order, fails the comparison.
    """
    summary = markdown.index("## Wasted GPU compute (summary)\n")
    detail = markdown.index("## Wasted GPU compute\n")
    appendix = markdown.index("## Appendix: wasted GPU compute")
    end = markdown.find("\n## ", appendix + 10)
    appendix_text = markdown[appendix:] if end == -1 else markdown[appendix:end]
    return "\n".join(
        [
            _section(markdown, summary),
            _section(markdown, detail),
            appendix_text.rstrip() + "\n",
        ]
    )


def _section(markdown: str, start: int) -> str:
    end = markdown.find("\n## ", start + 10)
    return markdown[start:] if end == -1 else markdown[start:end]


def test_rendered_stall_cost_section_matches_the_golden_file(result, tmp_path):
    markdown = reporting.build_report(_empty_summary(), None, wasted=result)
    rendered = _wasted_compute_sections(markdown)

    if not GOLDEN.exists():  # pragma: no cover - first run writes the baseline
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(rendered, encoding="utf-8")
        pytest.fail(f"golden file created at {GOLDEN}; review it and re-run")
    assert rendered == GOLDEN.read_text(encoding="utf-8")
