"""Context size at fallback by percentile band (LP-0MVZ6EAZR001DX5E).

The nightly report must answer *does the dominant reason for falling back
change as context grows, and does the uncached work per turn grow with it?*
in a ``## Context size at fallback by percentile band`` section rendered
immediately after ``## Local model usage by session``. This module pins:

- the cohort (sessions with >=1 local turn **and** token evidence at their
  fallback) and the exclusion accounting (a session without evidence is
  counted as excluded, never as 0);
- the ten disjoint decile bands, their boundary convention (a value exactly
  on a cut belongs to the higher band, the minimum to ``P0-P10``, the maximum
  to ``P90-P100``) and the explicit ``-`` for an empty band;
- the band means (hand-computed), including the ``n/a`` exclusion of a
  session with no final-turn token evidence;
- the per-band top-two fallback reasons with an ``other`` remainder that
  makes each row read as shares of 100 %;
- ``--json`` parity with the rendered table, and the additive-key guarantee;
- the omitted section on an empty window (an unattended 05:00 cron must still
  exit 0) and the doc-sync contract for the heading and the JSON key.

The derivations under test are the shared ones the neighbouring section uses:
``local_turn_rows`` (cohort), ``_fallback_reason_label`` (attribution),
``_percentile`` (linear interpolation) and, for the new column,
``aggregation``'s final-turn derivation read through ``SessionStats``.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import aggregation  # noqa: E402
import bucketing  # noqa: E402
import log_parser  # noqa: E402
import reporting  # noqa: E402

WINDOW_START = datetime(2026, 10, 23, 10, 0, 0)
WINDOW_END = datetime(2026, 10, 23, 12, 0, 0)
SCHEDULE = bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])
T0 = WINDOW_START + timedelta(minutes=5)

HEADING = reporting.CONTEXT_AT_FALLBACK_SECTION_TITLE
JSON_KEY = reporting.CONTEXT_AT_FALLBACK_JSON_KEY
REASONS_HEADING = "### Top two fallback reasons per band"


# ---------------------------------------------------------------------------
# Log-line fixtures
# ---------------------------------------------------------------------------


def _ts(offset_s: float) -> str:
    return (T0 + timedelta(seconds=offset_s)).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]


def _started(session: str, provider: str, offset_s: float) -> str:
    model = "Qwen3" if provider == "local" else "deepseek-v4-flash"
    return (
        f"{_ts(offset_s)} - INFO - Stream started: provider={provider} model={model} "
        f"session={session} request=[{{'type': 'text', 'text': 'hi'}}]"
    )


def _finished(
    session: str,
    provider: str,
    offset_s: float,
    *,
    prompt: int | None = 1000,
) -> str:
    model = "Qwen3" if provider == "local" else "deepseek-v4-flash"
    tokens = f"tokens={prompt}/10/{prompt + 10} " if prompt is not None else ""
    return (
        f"{_ts(offset_s)} - INFO - Stream finished: reason=tool_calls {tokens}"
        f"session={session} provider={provider} model={model} request=[{{'text': 'hi'}}]"
    )


def _skip(
    session: str,
    offset_s: float,
    reason: str = "provider_budget_exhausted",
    *,
    estimated_tokens: int | None = 40000,
    new_tokens: int | None = None,
    cached_ratio: float | None = None,
) -> str:
    """A ``routing_skip_local`` line.

    ``new_tokens``/``cached_ratio`` are optional so the pre-``new_tokens=`` log
    era (estimated tokens only) is expressible — the provenance the report must
    distinguish.
    """
    parts = ["routing_skip_local provider=local-qwen3 model=Qwen3"]
    if estimated_tokens is not None:
        parts.append(f"estimated_tokens={estimated_tokens}")
    parts.append("cold_threshold=38000 warm_threshold=120000")
    if new_tokens is not None:
        parts.append(f"new_tokens={new_tokens}")
    if cached_ratio is not None:
        parts.append(f"cached_ratio={cached_ratio}")
    parts.append("reason=" + reason + " → skipping local, routing to next remote provider")
    return f"{_ts(offset_s)} - INFO - {' '.join(parts)} session={session}"


def _guard(session: str, offset_s: float, *, new_tokens: int) -> str:
    """A ``local_admission_guard`` line: the measured uncached prompt cost."""
    return (
        f"{_ts(offset_s)} - INFO - local_admission_guard decision=admit "
        "reason=eta_within_budget enforced=false eta_prefill_s=12.0 "
        "eta_occupancy_s=0.0 projected_wait_s=12.0 budget_s=60.0 "
        f"new_tokens={new_tokens} budget_tokens=25000 rate_bucket=solitary "
        "contended_samples=0 busy_slots=0 in_flight_prefills=0 free_slot=true "
        "priority=medium policy=observe model=Qwen3 samples=0 provider=local-qwen3 "
        f"session={session}"
    )


def _local_turn(session: str, offset_s: float, *, prompt: int | None = 1000) -> list[str]:
    return [
        _started(session, "local", offset_s),
        _finished(session, "local", offset_s + 1, prompt=prompt),
    ]


def _remote_turn(session: str, offset_s: float, *, prompt: int | None = 1000) -> list[str]:
    return [
        _started(session, "opencode-go", offset_s),
        _finished(session, "opencode-go", offset_s + 1, prompt=prompt),
    ]


def _session(
    session: str,
    offset_s: float,
    *,
    context: int | None = 12000,
    new_tokens: int | None = 800,
    cached_ratio: float | None = None,
    reason: str = "provider_budget_exhausted",
    prompt: int | None = 1000,
) -> list[str]:
    """One local-using session that fell back to remote.

    ``context`` becomes the session's ``context_at_fallback`` (the skip line's
    ``estimated_tokens=``) and ``new_tokens`` its ``final_turn_new_tokens``.
    ``context=None`` with ``prompt=None`` yields a session with no token
    evidence at all — the excluded case.
    """
    return (
        _local_turn(session, offset_s, prompt=prompt)
        + [
            _skip(
                session,
                offset_s + 2,
                reason,
                estimated_tokens=context,
                new_tokens=new_tokens,
                cached_ratio=cached_ratio,
            )
        ]
        + _remote_turn(session, offset_s + 3, prompt=prompt)
    )


def _aggregate(lines: list[str]) -> aggregation.AnalysisResult:
    events = [
        ev
        for ev in (log_parser.parse_log_line(line) for line in lines)
        if ev is not None
    ]
    return aggregation.aggregate(events, WINDOW_START, WINDOW_END, SCHEDULE)


def _report(lines: list[str]) -> str:
    return reporting.build_report(_aggregate(lines), None)


def _stats(
    session_id: str,
    *,
    context_at_fallback: int | None,
    final_turn_new_tokens: int | None = None,
    reason: str = "provider_budget_exhausted",
) -> aggregation.SessionStats:
    """A minimal local-using session for the pure band-derivation tests."""
    return aggregation.SessionStats(
        session_id=session_id,
        start=T0,
        end=T0,
        duration_seconds=0.0,
        messages=2,
        local_requests=1,
        remote_requests=1,
        start_context_size=context_at_fallback,
        avg_context_size=None,
        max_context_size=context_at_fallback,
        avg_response_size=None,
        max_response_size=None,
        initial_provider="local",
        initial_model="Qwen3",
        remote_provider="opencode-go",
        remote_model="deepseek-v4-flash",
        remote_move_time=None,
        fallback_reason=reason,
        bucket="fast",
        slots=6,
        dispatch_denied=0,
        routing_skips=1,
        context_at_fallback=context_at_fallback,
        context_at_fallback_source=(
            "routing_skip" if context_at_fallback is not None else None
        ),
        final_turn_new_tokens=final_turn_new_tokens,
        final_turn_new_tokens_source=(
            "measured_new_tokens" if final_turn_new_tokens is not None else None
        ),
    )


def _bands(sessions: list[aggregation.SessionStats]) -> dict:
    return reporting.context_at_fallback_by_percentile(
        SimpleNamespace(sessions={s.session_id: s for s in sessions})
    )


def _populated_band(data: dict) -> dict:
    """The single non-empty band of a one-value cohort."""
    populated = [band for band in data["bands"] if band["sessions"]]
    assert len(populated) == 1, data["bands"]
    return populated[0]


def _band_table(section: str) -> list[list[str]]:
    """The four-column band rows of the rendered section, as cell lists."""
    rows: list[list[str]] = []
    for line in section.splitlines():
        if not line.startswith("| P") or line.count("|") != 5:
            continue
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


# ---------------------------------------------------------------------------
# Parsing: the pricing line carries the uncached remainder
# ---------------------------------------------------------------------------


class TestPricingLineParsing:
    def test_routing_skip_carries_new_tokens_and_cached_ratio(self):
        event = log_parser.parse_log_line(
            _skip("sess-a", 0, estimated_tokens=40000, new_tokens=3100, cached_ratio=0.92)
        )
        assert event is not None
        assert event.kind == "routing_skip"
        assert event.estimated_tokens == 40000
        assert event.new_tokens == 3100
        assert event.cached_ratio == pytest.approx(0.92)

    def test_absent_tokens_stay_none_never_zero(self):
        event = log_parser.parse_log_line(
            _skip("sess-a", 0, estimated_tokens=None)
        )
        assert event is not None
        assert event.new_tokens is None
        assert event.cached_ratio is None


# ---------------------------------------------------------------------------
# Final-turn derivation (aggregation)
# ---------------------------------------------------------------------------


class TestFinalTurnNewTokensDerivation:
    def test_measured_new_tokens_on_the_final_admission_line_wins(self):
        stats = _aggregate(
            _session(
                "sess-a", 0, context=40000, new_tokens=3100, cached_ratio=0.92
            )
        ).sessions["sess-a"]
        assert stats.final_turn_new_tokens == 3100
        assert (
            stats.final_turn_new_tokens_source
            == aggregation.FINAL_TURN_NEW_TOKENS_MEASURED
        )

    def test_estimated_tokens_times_uncached_share_when_measured_is_absent(self):
        stats = _aggregate(
            _session("sess-a", 0, context=40000, new_tokens=None, cached_ratio=0.9)
        ).sessions["sess-a"]
        assert stats.final_turn_new_tokens == 4000
        assert (
            stats.final_turn_new_tokens_source
            == aggregation.FINAL_TURN_NEW_TOKENS_ESTIMATED
        )

    def test_admission_guard_supplies_the_measured_figure(self):
        lines = _local_turn("sess-a", 0)
        lines.append(_skip("sess-a", 2, estimated_tokens=40000))
        lines.append(_guard("sess-a", 3, new_tokens=2400))
        lines += _remote_turn("sess-a", 4)
        stats = _aggregate(lines).sessions["sess-a"]
        assert stats.final_turn_new_tokens == 2400
        assert (
            stats.final_turn_new_tokens_source
            == aggregation.FINAL_TURN_NEW_TOKENS_MEASURED
        )

    def test_final_turn_without_evidence_is_none_never_zero(self):
        stats = _aggregate(
            _session("sess-a", 0, context=40000, new_tokens=None)
        ).sessions["sess-a"]
        assert stats.final_turn_new_tokens is None
        assert stats.final_turn_new_tokens_source is None
        # The context figure still exists: the two derivations are independent.
        assert stats.context_at_fallback == 40000

    def test_earlier_turns_evidence_is_never_borrowed(self):
        lines = _session("sess-a", 0, context=20000, new_tokens=900)
        lines += _local_turn("sess-a", 10)
        # The final admission line carries no uncached-token evidence.
        lines.append(_skip("sess-a", 12, estimated_tokens=60000))
        lines += _remote_turn("sess-a", 13)
        stats = _aggregate(lines).sessions["sess-a"]
        assert stats.final_turn_new_tokens is None


# ---------------------------------------------------------------------------
# Cohort and exclusion accounting
# ---------------------------------------------------------------------------


class TestCohortAndExclusions:
    def _fixture(self) -> list[str]:
        lines = _session("sess-a", 0, context=12000, new_tokens=800)
        # No token evidence at the fallback and none on the streams either.
        lines += _session("sess-b", 10, context=None, new_tokens=None, prompt=None)
        # Context evidence, but no final-turn uncached evidence.
        lines += _session("sess-c", 20, context=30000, new_tokens=None)
        # A remote-only session is outside the cohort entirely.
        lines += _remote_turn("sess-remote", 30)
        return lines

    def test_cohort_counts_include_exclude_and_state_the_na_tally(self):
        data = reporting.context_at_fallback_by_percentile(
            _aggregate(self._fixture())
        )
        assert data["sessions_in_window"] == 4
        assert data["sessions_with_local"] == 3
        assert data["sessions_included"] == 2
        assert data["sessions_excluded_no_context"] == 1
        assert data["sessions_without_final_turn_tokens"] == 1

    def test_excluded_session_is_not_counted_as_zero_in_the_report(self):
        section = _report(self._fixture()).split(f"## {HEADING}", 1)[1].split(
            REASONS_HEADING, 1
        )[0]
        assert "1 session(s) excluded for missing token evidence" in section
        # sess-c carries context 30000 but no final-turn evidence: its band
        # renders `-` for that column, never a fabricated 0.
        assert "| 1 | 30,000 | - |" in section


# ---------------------------------------------------------------------------
# Bands and their boundaries
# ---------------------------------------------------------------------------


class TestBands:
    def test_exactly_ten_bands_are_rendered_for_a_populated_cohort(self):
        lines: list[str] = []
        offset = 0.0
        for i in range(5):
            lines += _session(
                f"sess-{i}",
                offset,
                context=(i + 1) * 1000,
                new_tokens=(i + 1) * 100,
            )
            offset += 10
        data = reporting.context_at_fallback_by_percentile(_aggregate(lines))
        assert [band["label"] for band in data["bands"]] == [
            "P0–P10",
            "P10–P20",
            "P20–P30",
            "P30–P40",
            "P40–P50",
            "P50–P60",
            "P60–P70",
            "P70–P80",
            "P80–P90",
            "P90–P100",
        ]
        assert sum(band["sessions"] for band in data["bands"]) == 5

    def test_minimum_lands_in_the_first_band_and_maximum_in_the_last(self):
        sessions = [
            _stats(f"sess-{i}", context_at_fallback=value)
            for i, value in enumerate(range(0, 1000, 100))
        ]
        data = _bands(sessions)
        assert data["bands"][0]["sessions"] == 1
        assert data["bands"][9]["sessions"] == 1

    def test_value_exactly_on_a_decile_cut_lands_in_the_higher_band(self):
        # Eleven sessions, 0..1000 in steps of 100: P10 lands exactly on a
        # member value (100), which must therefore join the higher band.
        sessions = [
            _stats(f"sess-{i:02d}", context_at_fallback=i * 100)
            for i in range(11)
        ]
        values = sorted(float(i * 100) for i in range(11))
        assert reporting._percentile(values, 10) == pytest.approx(100.0)
        data = _bands(sessions)
        membership = {band["label"]: band["sessions"] for band in data["bands"]}
        assert membership["P0–P10"] == 1
        assert membership["P10–P20"] == 1
        # 900 sits exactly on P90, so it joins the highest band with the max.
        assert membership["P90–P100"] == 2
        assert sum(membership.values()) == 11

    def test_a_cohort_clustered_at_its_minimum_still_anchors_the_first_band(self):
        # Nine sessions share the cohort minimum (so every interpolated decile
        # cut equals it). The cluster must still read as the smallest contexts,
        # not be pushed into the top band by the "on a cut -> higher band" rule.
        sessions = [
            _stats(f"sess-{i}", context_at_fallback=1000) for i in range(9)
        ]
        sessions.append(_stats("sess-high", context_at_fallback=9000))
        data = _bands(sessions)
        assert data["bands"][0]["sessions"] == 9
        assert data["bands"][9]["sessions"] == 1

    def test_empty_band_renders_a_dash_not_zero_or_a_fabricated_mean(self):
        lines = _session("sess-a", 0, context=10000, new_tokens=500)
        lines += _session("sess-b", 10, context=20000, new_tokens=700)
        section = _report(lines).split(f"## {HEADING}", 1)[1].split(
            REASONS_HEADING, 1
        )[0]
        rows = _band_table(section)
        assert len(rows) == 10
        empty = [row for row in rows if row[1] == "-"]
        assert len(empty) == 8
        for row in empty:
            assert row[2:] == ["-", "-"], row

    def test_bands_are_disjoint_and_cover_every_sampled_session_once(self):
        lines: list[str] = []
        offset = 0.0
        for i in range(12):
            lines += _session(
                f"sess-{i:02d}",
                offset,
                context=1000 * (i + 1),
                new_tokens=100 * (i + 1),
            )
            offset += 10
        data = reporting.context_at_fallback_by_percentile(_aggregate(lines))
        assert (
            sum(band["sessions"] for band in data["bands"])
            == data["sessions_included"]
        )


# ---------------------------------------------------------------------------
# Averages
# ---------------------------------------------------------------------------


class TestBandAverages:
    def test_band_means_match_hand_computed_figures(self):
        # Ten sessions, one per band: values 1000..10000 in steps of 1000.
        sessions = [
            _stats(
                f"sess-{i}",
                context_at_fallback=(i + 1) * 1000,
                final_turn_new_tokens=(i + 1) * 100,
            )
            for i in range(10)
        ]
        data = _bands(sessions)
        assert all(band["sessions"] == 1 for band in data["bands"])
        assert [band["avg_context_at_fallback"] for band in data["bands"]] == [
            1000,
            2000,
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
        ]
        assert [band["avg_final_turn_new_tokens"] for band in data["bands"]] == [
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
        ]

    def test_band_means_are_the_mean_of_that_bands_members(self):
        # Nine sessions at 1000 and one at 9000: the nine cluster in the band
        # whose lower cut they meet, so one band carries a real multi-member
        # mean instead of a single sample.
        sessions = [
            _stats(
                f"sess-{i}",
                context_at_fallback=1000,
                final_turn_new_tokens=(i + 1) * 100,
            )
            for i in range(9)
        ]
        sessions.append(
            _stats("sess-high", context_at_fallback=9000, final_turn_new_tokens=50)
        )
        data = _bands(sessions)
        cluster = next(band for band in data["bands"] if band["sessions"] == 9)
        assert cluster["avg_context_at_fallback"] == 1000
        assert cluster["avg_final_turn_new_tokens"] == 500

    def test_session_without_final_turn_evidence_is_excluded_from_the_mean(self):
        sessions = [
            _stats(
                f"sess-{i}",
                context_at_fallback=1000,
                # The last of the cluster carries no final-turn evidence.
                final_turn_new_tokens=None if i == 8 else (i + 1) * 100,
            )
            for i in range(9)
        ]
        sessions.append(
            _stats("sess-high", context_at_fallback=9000, final_turn_new_tokens=50)
        )
        data = _bands(sessions)
        cluster = next(band for band in data["bands"] if band["sessions"] == 9)
        assert cluster["final_turn_new_tokens_samples"] == 8
        assert cluster["final_turn_new_tokens_missing"] == 1
        # Mean of the sessions WITH evidence (100..800), never (3600 + 0) / 9.
        assert cluster["avg_final_turn_new_tokens"] == 450

    def test_band_without_new_token_evidence_renders_a_dash(self):
        sessions = [
            _stats("sess-a", context_at_fallback=1000, final_turn_new_tokens=None),
            _stats("sess-b", context_at_fallback=9000, final_turn_new_tokens=700),
        ]
        data = _bands(sessions)
        first = data["bands"][0]
        assert first["sessions"] == 1
        assert first["avg_final_turn_new_tokens"] is None
        assert first["final_turn_new_tokens_missing"] == 1


# ---------------------------------------------------------------------------
# Top-two fallback reasons per band
# ---------------------------------------------------------------------------


class TestTopTwoReasons:
    def _bands_for(self, reason_counts: dict[str, int]) -> dict:
        sessions: list[aggregation.SessionStats] = []
        for i, (reason, count) in enumerate(reason_counts.items()):
            for j in range(count):
                sessions.append(
                    _stats(
                        f"sess-{i}-{j}",
                        # One shared context value puts the whole cohort in one
                        # band, so the per-band shares are readable directly.
                        context_at_fallback=1000,
                        final_turn_new_tokens=100,
                        reason=reason,
                    )
                )
        return _bands(sessions)

    def test_two_most_common_reasons_with_an_other_remainder(self):
        data = self._bands_for(
            {"context_too_large": 4, "empty_response": 3, "unattributed": 1}
        )
        band = _populated_band(data)
        assert band["sessions"] == 8
        assert [row["reason"] for row in band["reasons"]] == [
            "context_too_large",
            "empty_response",
        ]
        assert [row["sessions"] for row in band["reasons"]] == [4, 3]
        assert [row["pct_of_band_sessions"] for row in band["reasons"]] == [50.0, 37.5]
        assert band["other"]["sessions"] == 1
        assert band["other"]["pct_of_band_sessions"] == pytest.approx(12.5)
        # The row reads as shares of 100 %.
        total = sum(row["pct_of_band_sessions"] for row in band["reasons"])
        total += band["other"]["pct_of_band_sessions"]
        assert total == pytest.approx(100.0)

    def test_ties_break_alphabetically(self):
        data = self._bands_for({"zebra_reason": 2, "alpha_reason": 2})
        band = _populated_band(data)
        assert [row["reason"] for row in band["reasons"]] == [
            "alpha_reason",
            "zebra_reason",
        ]

    def test_band_with_no_shared_reason_has_an_empty_remainder(self):
        data = self._bands_for({"context_too_large": 1})
        band = _populated_band(data)
        assert band["other"]["sessions"] == 0
        assert band["other"]["pct_of_band_sessions"] == 0.0

    def test_reason_table_renders_counts_and_shares_per_band(self):
        lines: list[str] = []
        offset = 0.0
        for i, reason in enumerate(
            ["context_too_large", "context_too_large", "empty_response"]
        ):
            lines += _session(
                f"sess-{i}",
                offset,
                context=10000,
                new_tokens=500,
                reason=reason,
            )
            offset += 10
        section = _report(lines).split(f"## {HEADING}", 1)[1].split(
            REASONS_HEADING, 1
        )[1]
        populated = [
            line
            for line in section.splitlines()
            if line.startswith("| P") and "context_too_large" in line
        ]
        assert populated, section
        row = populated[0]
        assert "context_too_large 2 (66.7%)" in row
        assert "empty_response 1 (33.3%)" in row


# ---------------------------------------------------------------------------
# Placement and empty-window behaviour
# ---------------------------------------------------------------------------


class TestPlacement:
    def test_section_sits_between_local_model_usage_and_fallback_reasons(self):
        lines = _session("sess-a", 0, context=12000, new_tokens=800)
        lines.append(
            "2026-10-23 10:05:30,000 - INFO - Fallback triggered for "
            "model=v1/chat/completions, from=local-qwen3, to=opencode-go, "
            "reason=provider_budget_exhausted session=sess-a attempt=1"
        )
        report = _report(lines)
        assert (
            report.index("## Local model usage by session")
            < report.index(f"## {HEADING}")
            < report.index("## Fallback reasons")
        )

    def test_section_is_omitted_when_no_session_used_local(self):
        report = _report(_remote_turn("sess-remote", 0))
        assert f"## {HEADING}" not in report

    def test_section_is_absent_from_a_log_directory_with_no_logs(self):
        report = reporting.build_report(_aggregate([]), None)
        assert f"## {HEADING}" not in report


# ---------------------------------------------------------------------------
# --json parity and the additive-key guarantee
# ---------------------------------------------------------------------------


class TestJsonExposure:
    def _fixture(self) -> list[str]:
        lines: list[str] = []
        offset = 0.0
        for i, (context, new_tokens) in enumerate(
            [(1000, 100), (3000, 300), (5000, None), (7000, 700)]
        ):
            lines += _session(
                f"sess-{i}", offset, context=context, new_tokens=new_tokens
            )
            offset += 10
        return lines

    def test_json_key_is_present_and_matches_the_rendered_table(self):
        summary = _aggregate(self._fixture())
        data = reporting.summary_to_json(summary)
        block = data[JSON_KEY]
        assert block["sessions_with_local"] == 4
        assert block["sessions_included"] == 4

        section = reporting.build_report(summary, None).split(f"## {HEADING}", 1)[1]
        table = section.split(REASONS_HEADING, 1)[0]
        rows = _band_table(table)
        assert len(rows) == len(block["bands"]) == 10
        for cells, band in zip(rows, block["bands"]):
            assert cells[0].startswith(band["label"])
            assert cells[1] == (
                str(band["sessions"]) if band["sessions"] else "-"
            )
            if band["avg_context_at_fallback"] is not None:
                assert cells[2] == f"{band['avg_context_at_fallback']:,}"
            else:
                assert cells[2] == "-"
            if band["avg_final_turn_new_tokens"] is not None:
                assert cells[3] == f"{band['avg_final_turn_new_tokens']:,}"
            else:
                assert cells[3] == "-"

    def test_reason_shares_are_exposed_per_band(self):
        data = reporting.summary_to_json(_aggregate(self._fixture()))[JSON_KEY]
        for band in data["bands"]:
            assert "reasons" in band and "other" in band
            if band["sessions"]:
                total = sum(
                    row["pct_of_band_sessions"] for row in band["reasons"]
                ) + band["other"]["pct_of_band_sessions"]
                assert total == pytest.approx(100.0)

    def test_empty_window_reports_an_empty_band_list_and_zero_counts(self):
        data = reporting.summary_to_json(_aggregate([]))
        assert JSON_KEY in data
        assert data[JSON_KEY]["bands"] == []
        assert data[JSON_KEY]["sessions_with_local"] == 0
        assert data[JSON_KEY]["sessions_included"] == 0
        assert data[JSON_KEY]["sessions_excluded_no_context"] == 0

    def test_preexisting_json_keys_are_unchanged(self):
        summary = _aggregate(self._fixture())
        data = reporting.summary_to_json(summary)
        # The sibling derivations this section reuses must survive untouched.
        for key in (
            "sessions",
            "fallback_sessions",
            "fallback_events",
            "local_turn_coverage",
            "fallback_attribution",
            "performance_metrics",
            "capacity_signals",
            "prompt_eval_speed",
            "ttft",
            "wasted_compute",
            "hourly_activity",
        ):
            assert key in data
        coverage = data["local_turn_coverage"]
        assert coverage["sessions_with_local"] == 4
        assert set(coverage) >= {
            "sessions_in_window",
            "sessions_with_local",
            "percentiles",
            "total",
            "reasons",
            "sessions",
            "notes",
        }
        assert set(coverage["total"]) >= {
            "local_turns",
            "remote_turns",
            "local_turn_pct",
            "context_at_fallback",
            "answered_turns",
            "unanswered_turns",
            "context_samples",
        }


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_rerunning_over_the_same_window_produces_an_identical_section(self):
        lines: list[str] = []
        offset = 0.0
        for i in range(6):
            lines += _session(
                f"sess-{i}",
                offset,
                context=1000 * (i + 1),
                new_tokens=90 * (i + 1),
            )
            offset += 10
        first = _report(lines).split(f"## {HEADING}", 1)[1]
        second = _report(lines).split(f"## {HEADING}", 1)[1]
        assert first == second


# ---------------------------------------------------------------------------
# Documentation contract
# ---------------------------------------------------------------------------


class TestDocumentationSync:
    def test_docs_document_the_heading_and_the_json_key(self):
        for doc in ("SKILL.md", "REFERENCE.md"):
            text = (SKILL_ROOT / doc).read_text(encoding="utf-8")
            assert HEADING in text, f"{doc} does not document the section heading"
            assert JSON_KEY in text, f"{doc} does not document the JSON key"

    def test_reference_documents_the_banding_rules(self):
        reference = (SKILL_ROOT / "REFERENCE.md").read_text(encoding="utf-8")
        assert "### Context size at fallback by percentile band" in reference
        section = reference.split(
            "### Context size at fallback by percentile band", 1
        )[1].split("\n### Archival", 1)[0]
        for fact in (
            "n/a",
            "P0–P10",
            "P90–P100",
            "linear interpolation",
            "final turn",
            "excluded",
        ):
            assert fact in section, f"REFERENCE.md banding subsection omits {fact!r}"
