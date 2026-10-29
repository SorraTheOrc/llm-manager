"""Local-turn coverage report (LP-0MVM93DE6009JGBU).

The 24-hour report must answer "which sessions actually used the local model,
how many turns did it answer, what share of the session's turns were local,
and why did each session move to remote" without an analyst re-reading logs.
This module pins the deterministic derivation behind the report's
``## Local model usage by session`` section:

- session selection (a session with >=1 local turn, answered or not);
- per-session ``% local turns`` = local / (local + remote), 100 % when the
  session never used remote;
- P10/P50/P90 with linear interpolation over the sessions with >=1 local turn;
- the fallback-reason counts-and-percent table;
- the per-session table's sort order (most local turns first);
- ``context size at fallback`` derivation: the session's own
  ``routing_skip_local estimated_tokens`` at/near its fallback, then the first
  remote stream's prompt tokens, then the session's own context columns, and
  an explicit ``n/a`` (never a fabricated estimate) when none exist.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aggregation
import bucketing
import log_parser
import reporting

WINDOW_START = datetime(2026, 10, 23, 10, 0, 0)
WINDOW_END = datetime(2026, 10, 23, 12, 0, 0)
SCHEDULE = bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])
T0 = WINDOW_START + timedelta(minutes=5)


def _ts(offset_s: float) -> str:
    """Log timestamp ``offset_s`` seconds after ``T0``."""
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
    reason: str = "tool_calls",
) -> str:
    model = "Qwen3" if provider == "local" else "deepseek-v4-flash"
    tokens = f"tokens={prompt}/10/{prompt + 10} " if prompt is not None else ""
    return (
        f"{_ts(offset_s)} - INFO - Stream finished: reason={reason} {tokens}"
        f"session={session} provider={provider} model={model} request=[{{'text': 'hi'}}]"
    )


def _skip(
    session: str,
    offset_s: float,
    reason: str,
    *,
    estimated_tokens: int | None = 40000,
    chain_hold_cycle: int | None = None,
) -> str:
    tokens = f"estimated_tokens={estimated_tokens} " if estimated_tokens else ""
    cycle = f" chain_hold_cycle={chain_hold_cycle}" if chain_hold_cycle else ""
    return (
        f"{_ts(offset_s)} - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 "
        f"{tokens}cold_threshold=38000 warm_threshold=120000 new_tokens=323 "
        f"cached_ratio=0.99 reason={reason} → skipping local, routing to next remote "
        f"provider session={session}{cycle}"
    )


def _aggregate(lines: list[str]) -> aggregation.AnalysisResult:
    events = [
        ev
        for ev in (log_parser.parse_log_line(line) for line in lines)
        if ev is not None
    ]
    return aggregation.aggregate(events, WINDOW_START, WINDOW_END, SCHEDULE)


def _turns(session: str, local: int, remote: int, *, start_s: float = 0.0) -> list[str]:
    """``local`` local turns and ``remote`` remote turns for one session."""
    lines: list[str] = []
    offset = start_s
    for i in range(local):
        lines.append(_started(session, "local", offset))
        lines.append(_finished(session, "local", offset + 1, prompt=1000 + i))
        offset += 2
    for i in range(remote):
        lines.append(_started(session, "opencode-go", offset))
        lines.append(_finished(session, "opencode-go", offset + 1, prompt=2000 + i))
        offset += 2
    return lines


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestRoutingSkipCarriesEstimatedTokens:
    def test_estimated_tokens_is_parsed(self):
        ev = log_parser.parse_log_line(_skip("sess-a", 0, "context_too_large"))
        assert ev is not None
        assert ev.kind == "routing_skip"
        assert ev.estimated_tokens == 40000

    def test_line_without_estimated_tokens_stays_none(self):
        ev = log_parser.parse_log_line(
            _skip("sess-a", 0, "context_too_large", estimated_tokens=None)
        )
        assert ev is not None
        assert ev.estimated_tokens is None


# ---------------------------------------------------------------------------
# Per-session derivation
# ---------------------------------------------------------------------------


class TestSessionSelectionAndTurnShare:
    def test_session_with_local_turns_is_selected(self):
        result = _aggregate(_turns("sess-local", 2, 1))
        rows = reporting.local_turn_rows(result.sessions)
        assert [r.session_id for r in rows] == ["sess-local"]
        assert rows[0].local_turns == 2
        assert rows[0].remote_turns == 1

    def test_remote_only_session_is_excluded(self):
        result = _aggregate(_turns("sess-remote", 0, 3))
        assert reporting.local_turn_rows(result.sessions) == []

    def test_unanswered_local_turn_is_counted_and_flagged(self):
        lines = [
            _started("sess-a", "local", 0),
            # An errored local turn: routed, but never answered.
            _finished("sess-a", "local", 1, prompt=None, reason="error"),
        ]
        result = _aggregate(lines)
        row = reporting.local_turn_rows(result.sessions)[0]
        assert row.local_turns == 1
        assert row.unanswered_turns == 1
        assert row.answered_turns == 0

    def test_answered_local_turns_are_paired_fifo(self):
        lines = [
            _started("sess-a", "local", 0),
            _finished("sess-a", "local", 1, prompt=None, reason="error"),
            _started("sess-a", "local", 2),
            _finished("sess-a", "local", 3),
        ]
        result = _aggregate(lines)
        row = reporting.local_turn_rows(result.sessions)[0]
        assert row.local_turns == 2
        assert row.unanswered_turns == 1


class TestLocalTurnShare:
    def test_share_is_local_fraction_of_all_turns(self):
        result = _aggregate(_turns("sess-a", 1, 3))
        row = reporting.local_turn_rows(result.sessions)[0]
        assert row.local_turn_pct == 25.0

    def test_local_only_session_is_one_hundred_percent(self):
        result = _aggregate(_turns("sess-a", 3, 0))
        row = reporting.local_turn_rows(result.sessions)[0]
        assert row.local_turn_pct == 100.0
        assert row.reason == "no_remote_fallback"

    def test_session_with_no_turns_has_no_share(self):
        stats = aggregation.SessionStats(
            session_id="sess-empty",
            start=T0,
            end=T0,
            duration_seconds=0.0,
            messages=0,
            local_requests=0,
            remote_requests=0,
            start_context_size=None,
            avg_context_size=None,
            max_context_size=None,
            avg_response_size=None,
            max_response_size=None,
            initial_provider="local",
            initial_model="Qwen3",
            remote_provider=None,
            remote_model=None,
            remote_move_time=None,
            fallback_reason=None,
            bucket="fast",
            slots=6,
            dispatch_denied=0,
            routing_skips=0,
        )
        assert stats.local_turn_pct is None


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------


class TestPercentileTable:
    def _summary(self, local_counts: list[int]) -> aggregation.AnalysisResult:
        lines: list[str] = []
        offset = 0.0
        for i, count in enumerate(local_counts):
            lines += _turns(f"sess-{i:02d}", count, 4, start_s=offset)
            offset += 40
        return _aggregate(lines)

    def test_percentiles_use_linear_interpolation(self):
        coverage = reporting.local_turn_coverage(self._summary([1, 2, 3, 4]))
        local = {row["label"]: row["local_turns"] for row in coverage["percentiles"]}
        # sorted [1, 2, 3, 4]; pos = p/100 * (n-1)
        assert local["P10"] == pytest.approx(1.3)
        assert local["P50"] == pytest.approx(2.5)
        assert local["P90"] == pytest.approx(3.7)

    def test_total_row_sums_turns_and_recomputes_the_share(self):
        coverage = reporting.local_turn_coverage(self._summary([1, 3]))
        assert coverage["total"]["local_turns"] == 4
        assert coverage["total"]["remote_turns"] == 8
        assert coverage["total"]["local_turn_pct"] == pytest.approx(33.3)

    def test_context_percentiles_ignore_sessions_without_token_data(self):
        lines = _turns("sess-a", 1, 1) + [
            _skip("sess-a", 1.5, "provider_budget_exhausted", estimated_tokens=52000)
        ]
        # sess-b fell back with no token evidence at all (no ``estimated_tokens``
        # on its skip line, no usage on either stream).
        lines += [
            _started("sess-b", "local", 40),
            _finished("sess-b", "local", 41, prompt=None),
            _skip("sess-b", 42, "empty_response", estimated_tokens=None),
            _started("sess-b", "opencode-go", 43),
            _finished("sess-b", "opencode-go", 44, prompt=None),
        ]
        result = _aggregate(lines)
        coverage = reporting.local_turn_coverage(result)
        assert coverage["sessions_with_local"] == 2
        assert coverage["total"]["context_samples"] == 1
        assert coverage["percentiles"][1]["context_at_fallback"] == 52000


# ---------------------------------------------------------------------------
# Fallback reason + context size at fallback
# ---------------------------------------------------------------------------


class TestFallbackAttribution:
    def test_reason_comes_from_the_sessions_own_skip_line(self):
        lines = _turns("sess-a", 1, 1) + [
            _skip("sess-a", 1.5, "provider_budget_exhausted", estimated_tokens=52310)
        ]
        result = _aggregate(lines)
        row = reporting.local_turn_rows(result.sessions)[0]
        assert row.reason == "provider_budget_exhausted"
        assert row.context_at_fallback == 52310
        assert row.context_source == aggregation.CONTEXT_AT_FALLBACK_ROUTING_SKIP

    def test_context_falls_back_to_first_remote_stream_prompt(self):
        lines = _turns("sess-a", 1, 0) + [
            _skip("sess-a", 3.0, "provider_budget_exhausted", estimated_tokens=None),
            _started("sess-a", "opencode-go", 4),
            _finished("sess-a", "opencode-go", 5, prompt=45678),
        ]
        result = _aggregate(lines)
        row = reporting.local_turn_rows(result.sessions)[0]
        assert row.context_at_fallback == 45678
        assert row.context_source == aggregation.CONTEXT_AT_FALLBACK_REMOTE_STREAM

    def test_context_falls_back_to_session_context_columns(self):
        # No routing_skip line, and the remote stream reports no usage; the
        # session's own measured local prompt is the last resort.
        lines = _turns("sess-a", 1, 0) + [
            "2026-10-23 10:05:02,000 - INFO - Fallback triggered for "
            "model=v1/chat/completions, from=local-qwen3, to=opencode-go, "
            "reason=provider_budget_exhausted session=sess-a attempt=1",
            _started("sess-a", "opencode-go", 4),
            _finished("sess-a", "opencode-go", 5, prompt=None),
        ]
        result = _aggregate(lines)
        row = reporting.local_turn_rows(result.sessions)[0]
        assert row.reason == "provider_budget_exhausted"
        assert row.context_at_fallback == 1000
        assert row.context_source == aggregation.CONTEXT_AT_FALLBACK_SESSION_CONTEXT

    def test_no_token_data_renders_na_never_a_guess(self):
        lines = [
            _started("sess-a", "local", 0),
            _finished("sess-a", "local", 1, prompt=None),
            _skip("sess-a", 2, "empty_response", estimated_tokens=None),
            _started("sess-a", "opencode-go", 3),
            _finished("sess-a", "opencode-go", 4, prompt=None),
        ]
        result = _aggregate(lines)
        row = reporting.local_turn_rows(result.sessions)[0]
        assert row.context_at_fallback is None
        assert row.context_source is None
        report = reporting.build_report(result, None)
        section = report.split("## Local model usage by session", 1)[1]
        assert "| sess-a | 1 | 1 | 50.0% | empty_response | n/a |" in section


class TestFallbackReasonTable:
    def test_reason_counts_and_percent_of_sessions(self):
        lines: list[str] = []
        offset = 0.0
        for i, reason in enumerate(
            ["provider_budget_exhausted", "provider_budget_exhausted", "empty_response"]
        ):
            lines += _turns(f"sess-{i}", 1, 1, start_s=offset)
            lines.append(_skip(f"sess-{i}", offset + 1.5, reason, estimated_tokens=None))
            offset += 40
        lines += _turns("sess-local-only", 2, 0, start_s=offset)
        coverage = reporting.local_turn_coverage(_aggregate(lines))
        counts = {row["reason"]: row for row in coverage["reasons"]}
        assert counts["provider_budget_exhausted"]["sessions"] == 2
        assert counts["provider_budget_exhausted"]["pct_of_sessions_with_local"] == pytest.approx(
            50.0
        )
        assert counts["empty_response"]["sessions"] == 1
        assert counts["no_remote_fallback"]["sessions"] == 1

    def test_legacy_spelling_folds_into_one_row(self):
        """A window spanning the rename (rotated logs) must report ONE reason
        row, not one per spelling: the parser normalises the pre-rename
        ``rate_limited`` spelling to ``provider_budget_exhausted``
        (LP-0MVMBM8YS003Y2JG AC3/AC4)."""
        # Pre-rename line (legacy spelling) for sess-old, current for sess-new.
        lines = _turns("sess-old", 1, 1) + [
            _skip("sess-old", 1.5, "rate_limited", estimated_tokens=None)
        ]
        lines += _turns("sess-new", 1, 1, start_s=40) + [
            _skip("sess-new", 41.5, "provider_budget_exhausted", estimated_tokens=None)
        ]
        coverage = reporting.local_turn_coverage(_aggregate(lines))
        reasons = {row["reason"]: row["sessions"] for row in coverage["reasons"]}
        assert reasons == {"provider_budget_exhausted": 2}
        section = reporting.build_report(_aggregate(lines), None).split(
            "## Local model usage by session", 1
        )[1]
        reasons_table = section.split("### Fallback reason distribution", 1)[1].split(
            "### Per-session coverage", 1
        )[0]
        # One row for the reason, and no second row under the old spelling.
        assert reasons_table.count("provider_budget_exhausted") == 1
        assert "rate_limited" not in reasons_table


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


class TestSortOrder:
    def test_sessions_sorted_by_local_turns_descending(self):
        lines: list[str] = []
        offset = 0.0
        for sid, local in (("sess-few", 1), ("sess-most", 5), ("sess-mid", 3)):
            lines += _turns(sid, local, 1, start_s=offset)
            offset += 60
        rows = reporting.local_turn_rows(_aggregate(lines).sessions)
        assert [r.session_id for r in rows] == ["sess-most", "sess-mid", "sess-few"]

    def test_ties_are_broken_by_session_id_for_determinism(self):
        lines = _turns("sess-b", 2, 1) + _turns("sess-a", 2, 1, start_s=40)
        rows = reporting.local_turn_rows(_aggregate(lines).sessions)
        assert [r.session_id for r in rows] == ["sess-a", "sess-b"]


# ---------------------------------------------------------------------------
# Report rendering + machine-readable exposure
# ---------------------------------------------------------------------------


class TestReportSection:
    def _report(self, lines: list[str]) -> str:
        return reporting.build_report(_aggregate(lines), None)

    def test_section_renders_the_three_tables(self):
        lines = _turns("sess-a", 2, 1) + [_skip("sess-a", 3.5, "provider_budget_exhausted")]
        report = self._report(lines)
        assert "## Local model usage by session" in report
        section = report.split("## Local model usage by session", 1)[1]
        summary_table = section.split("### Fallback reason distribution", 1)[0]
        for label in ("P10", "P50", "P90", "Total"):
            assert f"| {label} |" in summary_table
        assert "# turns local" in summary_table
        assert "# turns remote" in summary_table
        assert "% local turns" in summary_table
        assert "Context size at fallback" in summary_table
        reasons_table = section.split("### Fallback reason distribution", 1)[1].split(
            "### Per-session coverage", 1
        )[0]
        assert "| provider_budget_exhausted | 1 | 100.0% |" in reasons_table
        sessions_table = section.split("### Per-session coverage", 1)[1]
        assert "| sess-a | 2 | 1 | 66.7% | provider_budget_exhausted |" in sessions_table

    def test_section_renders_without_any_local_traffic(self):
        report = self._report(_turns("sess-remote", 0, 2))
        section = report.split("## Local model usage by session", 1)[1]
        assert "No session in the window used the local model." in section

    def test_no_token_data_is_documented_as_a_limitation(self):
        lines = [
            _started("sess-a", "local", 0),
            _finished("sess-a", "local", 1, prompt=None),
            _started("sess-a", "opencode-go", 2),
            _finished("sess-a", "opencode-go", 3, prompt=None),
        ]
        section = self._report(lines).split("## Local model usage by session", 1)[1]
        assert "n/a" in section
        assert "no token evidence at their fallback" in section

    def test_provider_budget_exhausted_is_explained_as_budget_pacing(self):
        lines = _turns("sess-a", 1, 1) + [_skip("sess-a", 1.5, "provider_budget_exhausted")]
        section = self._report(lines).split("## Local model usage by session", 1)[1]
        assert "per-provider RPM budget" in section
        assert "not an upstream HTTP 429" in section


class TestInterpretationNotes:
    def test_decision_first_note_counts_remote_only_sessions(self):
        lines = _turns("sess-local", 2, 1) + [
            _skip("sess-local", 3.5, "provider_budget_exhausted")
        ]
        lines += _turns("sess-remote-only", 0, 3, start_s=40)
        result = _aggregate(lines)
        coverage = reporting.local_turn_coverage(result)
        assert coverage["notes"]["sessions_reaching_remote"] == 2
        assert coverage["notes"]["declined_before_local_turn"] == 1
        section = reporting.build_report(result, None).split(
            "## Local model usage by session", 1
        )[1]
        assert (
            "1 of 2 sessions that reached remote never served a local turn (50.0%)"
            in section
        )

    def test_context_size_note_is_data_gated_and_counts_before_local(self):
        lines = _turns("sess-local", 1, 1) + [
            _skip("sess-local", 1.5, "context_too_large")
        ]
        lines += _turns("sess-remote-only", 0, 2, start_s=40) + [
            _skip("sess-remote-only", 39.5, "large_context_bypass")
        ]
        result = _aggregate(lines)
        coverage = reporting.local_turn_coverage(result)
        assert coverage["notes"]["context_fallback_sessions"] == 2
        assert coverage["notes"]["context_fallback_before_local_turn"] == 1
        section = reporting.build_report(result, None).split(
            "## Local model usage by session", 1
        )[1]
        assert "1 of 2 were declined before the session's first local turn" in section
        assert "never on a session's first message" in section

    def test_provider_budget_exhausted_note_is_omitted_without_that_reason(self):
        lines = _turns("sess-local", 1, 1) + [
            _skip("sess-local", 1.5, "empty_response")
        ]
        section = reporting.build_report(_aggregate(lines), None).split(
            "## Local model usage by session", 1
        )[1]
        assert "per-provider RPM budget" not in section

    def test_no_notes_for_local_only_traffic(self):
        section = reporting.build_report(
            _aggregate(_turns("sess-local", 2, 0)), None
        ).split("## Local model usage by session", 1)[1]
        assert "Interpretation:" not in section


class TestMachineReadableExposure:
    def test_json_summary_exposes_the_coverage(self):
        lines = _turns("sess-a", 2, 1) + [_skip("sess-a", 3.5, "provider_budget_exhausted")]
        data = reporting.summary_to_json(_aggregate(lines))
        coverage = data["local_turn_coverage"]
        assert coverage["sessions_with_local"] == 1
        assert coverage["total"]["local_turns"] == 2
        assert coverage["total"]["remote_turns"] == 1
        assert coverage["reasons"][0]["reason"] == "provider_budget_exhausted"

    def test_session_csv_carries_the_per_session_columns(self, tmp_path):
        lines = _turns("sess-a", 2, 1) + [_skip("sess-a", 3.5, "provider_budget_exhausted")]
        summary = _aggregate(lines)
        standard, _low = reporting.write_csvs(summary, tmp_path)
        with standard.open() as fh:
            row = list(csv.DictReader(fh))[0]
        assert row["local_turns_answered"] == "2"
        assert row["local_turns_unanswered"] == "0"
        assert row["local_turn_pct"] == "66.7"
        assert row["context_at_fallback"] == "40000"
        assert row["context_at_fallback_source"] == "routing_skip"
