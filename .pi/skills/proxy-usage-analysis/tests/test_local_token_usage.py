"""Local stream token usage in the usage-analysis pipeline (LP-0MVMHNVKT003PRIK).

Local ``Stream finished`` lines now carry llama-server's measured
``tokens=prompt/completion/total`` (the proxy requests
``stream_options.include_usage``), so the report must:

- parse local token counts and the ``stream=false`` buffered-reply marker;
- attribute tokens to the provider/model that actually SERVED each stream, not
  to the session's initial provider;
- count terminal lines without usage ("streams without usage") instead of
  silently treating them as zero;
- render real local prompt sizes and the usage-coverage line in the report.

Fixtures mirror the exact log format the proxy emits (see tests/fixtures.py).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aggregation
import bucketing
import log_parser
import reporting

from tests import fixtures

WINDOW_START = datetime(2026, 8, 2, 14, 0, 0)
WINDOW_END = datetime(2026, 8, 2, 15, 0, 0)
SCHEDULE = bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])


def _parse(lines: list[str]) -> list[log_parser.LogEvent]:
    return [
        ev
        for ev in (log_parser.parse_log_line(line) for line in lines)
        if ev is not None
    ]


def _aggregate(lines: list[str]) -> aggregation.AnalysisResult:
    return aggregation.aggregate(
        _parse(lines), WINDOW_START, WINDOW_END, SCHEDULE
    )


class TestParsingMeasuredLocalUsage:
    def test_local_stream_finished_carries_measured_tokens(self):
        ev = log_parser.parse_log_line(fixtures.STREAM_FINISHED_LOCAL_MEASURED)
        assert ev is not None
        assert ev.kind == "stream_finished"
        assert ev.provider == "local"
        assert ev.model == "Qwen3"
        assert (ev.prompt, ev.completion, ev.total) == (30000, 200, 30200)
        # Streaming line: no buffered-reply marker.
        assert ev.streamed is None

    def test_non_stream_line_is_marked_and_keeps_tokens(self):
        ev = log_parser.parse_log_line(fixtures.STREAM_FINISHED_LOCAL_NON_STREAM)
        assert ev is not None
        assert ev.kind == "stream_finished"
        assert (ev.prompt, ev.completion, ev.total) == (31000, 150, 31150)
        assert ev.streamed is False

    def test_legacy_line_without_tokens_is_unchanged(self):
        ev = log_parser.parse_log_line(fixtures.STREAM_FINISHED_NO_TOKENS)
        assert ev is not None
        assert ev.kind == "stream_finished"
        assert ev.prompt is None
        assert ev.streamed is None


class TestAggregationUsesServingProvider:
    """tokens are attributed to whoever served the stream (AC4)."""

    def test_session_that_fell_back_splits_tokens_by_provider(self):
        result = _aggregate(fixtures.E2E_LINES)
        by_model = result.token_by_model
        # S2 served one local turn (900/40) then one remote turn (950/200).
        assert by_model[("local", "Qwen3")] == (900, 40, 940)
        assert by_model[("opencode-go", "deepseek-v4-flash")] == (950, 200, 1150)
        # S1 is local-only and keeps its own row.
        assert by_model[("local", "Qwen3")][0] == 900 + 1200 + 1300

    def test_totals_match_by_model_totals(self):
        result = _aggregate(fixtures.E2E_LINES)
        grand_in, grand_out, grand_total = result.total_tokens
        by_model = result.token_by_model
        assert sum(v[0] for v in by_model.values()) == grand_in
        assert sum(v[1] for v in by_model.values()) == grand_out
        assert sum(v[2] for v in by_model.values()) == grand_total

    def test_non_stream_reply_tokens_reach_the_session(self):
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local "
            "model=Qwen3 session=sess-ns request=[{'type': 'text', 'text': 'hi'}]",
            fixtures.STREAM_FINISHED_LOCAL_NON_STREAM.replace(
                "019fc284-dcb8-74ca-9a64-9306b6f9d286", "sess-ns"
            ),
        ]
        result = _aggregate(lines)
        session = result.sessions["sess-ns"]
        assert (session.input_tokens, session.output_tokens) == (31000, 150)
        assert session.max_context_size == 31000
        assert result.token_by_model[("local", "Qwen3")] == (31000, 150, 31150)


class TestStreamsWithoutUsage:
    def test_counts_terminal_lines_without_tokens_by_provider(self):
        lines = [
            f"2026-08-02 14:00:00,000 - INFO - Stream started: {rest}"
            for rest in (
                "provider=local model=Qwen3 session=sess-a request=x",
                "provider=deepseek model=deepseek-v4-flash session=sess-b request=x",
            )
        ] + [
            "2026-08-02 14:00:05,000 - INFO - Stream finished: reason=stop "
            "tokens=100/10/110 session=sess-a provider=local model=Qwen3",
            "2026-08-02 14:00:06,000 - INFO - Stream finished: reason=stop "
            "session=sess-b provider=deepseek model=deepseek-v4-flash",
        ]
        result = _aggregate(lines)
        assert result.streams_finished == 1
        assert result.streams_without_usage == 1
        assert dict(result.streams_without_usage_by_provider) == {"deepseek": 1}

    def test_measured_local_streams_are_not_counted_as_without_usage(self):
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local "
            "model=Qwen3 session=sess-m request=x",
            fixtures.STREAM_FINISHED_LOCAL_MEASURED.replace(
                "019fc284-dcb8-74ca-9a64-9306b6f9d286", "sess-m"
            ),
        ]
        result = _aggregate(lines)
        assert result.streams_finished == 1
        assert result.streams_without_usage == 0
        assert dict(result.streams_without_usage_by_provider) == {}


class TestReportRendering:
    def _report(self, lines: list[str]) -> str:
        summary = _aggregate(lines)
        return reporting.build_report(summary, None)

    def test_total_tokens_section_reports_usage_coverage(self):
        report = self._report(fixtures.E2E_LINES)
        section = report.split("## Total tokens", 1)[1]
        assert "Measured usage:" in section
        assert "Streams without usage:" in section

    def test_by_model_table_names_local_measured_tokens(self):
        report = self._report(fixtures.E2E_LINES)
        section = report.split("### Tokens — by model", 1)[1]
        assert "| local | Qwen3 |" in section
        # 900 + 1200 + 1300 prompt tokens measured on local streams.
        assert "| 3,400 |" in section

    def test_without_usage_streams_are_named_in_the_limitations(self):
        report = self._report(fixtures.E2E_LINES)
        assert "include_usage" in report
        assert "streams without usage" in report.lower()

    @pytest.mark.parametrize("lines", [[], fixtures.E2E_LINES])
    def test_report_renders_with_and_without_traffic(self, lines):
        report = self._report(lines)
        assert "## Total tokens" in report
