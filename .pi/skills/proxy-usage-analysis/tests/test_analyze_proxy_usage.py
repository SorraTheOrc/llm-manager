"""Unit tests for the proxy-usage-analysis skill.

Covers: log-line parsing, session aggregation, day/night bucketing from the
slot schedule, recommendation rules, config loading, and an end-to-end run
over fixture log files.

Fixtures are derived from real lines in /var/log/llama-proxy/proxy.log
(see tests/fixtures.py).
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aggregation  # noqa: E402
import bucketing  # noqa: E402
import config_loader  # noqa: E402
import log_parser  # noqa: E402
import recommendations  # noqa: E402
import reporting  # noqa: E402
from tests import fixtures  # noqa: E402

WINDOW_START = datetime(2026, 8, 2, 14, 0, 0)
WINDOW_END = datetime(2026, 8, 2, 15, 0, 0)


def _schedule() -> bucketing.SlotSchedule:
    return bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])


def _events(lines: list[str]) -> list[object]:
    """Parse raw lines into events (no window filtering)."""
    return [e for e in (log_parser.parse_log_line(ln) for ln in lines) if e is not None]


# ---------------------------------------------------------------------------
# Log-line parsing
# ---------------------------------------------------------------------------


class TestLogLineParsing:
    def test_stream_started_local(self):
        ev = log_parser.parse_log_line(fixtures.STREAM_STARTED_LOCAL)
        assert ev is not None
        assert ev.kind == "stream_started"
        assert ev.ts == datetime(2026, 8, 2, 13, 58, 32, 260000)
        assert ev.provider == "local"
        assert ev.model == "Qwen3"
        assert ev.session == "019fc284-dcb8-74ca-9a64-9306b6f9d286"

    def test_stream_started_remote(self):
        ev = log_parser.parse_log_line(fixtures.STREAM_STARTED_REMOTE)
        assert ev.kind == "stream_started"
        assert ev.provider == "opencode-go"
        assert ev.model == "deepseek-v4-flash"

    def test_stream_finished_tokens_before_session(self):
        ev = log_parser.parse_log_line(fixtures.STREAM_FINISHED_TOKENS_SESSION)
        assert ev.kind == "stream_finished"
        assert ev.reason == "tool_calls"
        assert (ev.prompt, ev.completion, ev.total) == (43550, 460, 44010)
        assert ev.session == "019fc27d-3a46-7e5c-871e-57ab32f875f3"
        assert ev.provider == "opencode-go"
        assert ev.model == "deepseek-v4-flash"

    def test_stream_finished_session_before_tokens(self):
        ev = log_parser.parse_log_line(fixtures.STREAM_FINISHED_SESSION_TOKENS)
        assert (ev.prompt, ev.completion, ev.total) == (52031, 56, 52087)

    def test_stream_finished_tokens_no_session(self):
        ev = log_parser.parse_log_line(fixtures.STREAM_FINISHED_TOKENS_NO_SESSION)
        assert ev.kind == "stream_finished"
        assert ev.session is None
        assert (ev.prompt, ev.completion) == (49364, 6444)

    def test_stream_finished_no_tokens(self):
        ev = log_parser.parse_log_line(fixtures.STREAM_FINISHED_NO_TOKENS)
        assert ev.kind == "stream_finished"
        assert ev.prompt is None
        assert ev.session == "019fc27d-3a46-7e5c-871e-57ab32f875f3"
        assert ev.provider == "local"

    def test_fallback_triggered(self):
        ev = log_parser.parse_log_line(fixtures.FALLBACK_CONCURRENCY)
        assert ev.kind == "fallback"
        assert ev.src == "local-qwen3"
        assert ev.dst == "opencode-go-deepseek"
        assert ev.reason == "local_concurrency_limit"
        assert ev.session is None

    def test_routing_skip_local(self):
        ev = log_parser.parse_log_line(fixtures.ROUTING_SKIP_WARM)
        assert ev.kind == "routing_skip"
        assert ev.reason == "warm_cache_bypass"
        assert ev.session == "019fc27d-3a46-7e5c-871e-57ab32f875f3"

    def test_routing_skip_large_context(self):
        ev = log_parser.parse_log_line(fixtures.ROUTING_SKIP_LARGE_CONTEXT)
        assert ev.kind == "routing_skip"
        assert ev.reason == "large_context_bypass"
        assert ev.session == "aaaaaaaa-1111-2222-3333-444444444444"

    def test_dispatch_denied(self):
        ev = log_parser.parse_log_line(fixtures.DISPATCH_DENIED)
        assert ev.kind == "dispatch_denied"
        assert ev.session == "019fc245"
        assert ev.owner == "019fc27d"
        assert ev.active == 4

    @pytest.mark.parametrize(
        "line",
        [
            fixtures.ROUTING_CHECK_IGNORED,
            fixtures.SESSION_HEADER_IGNORED,
            fixtures.LEASE_RENEWED_IGNORED,
            fixtures.REQUEST_ROUTING_IGNORED,
            fixtures.WARNING_LINE_IGNORED,
            fixtures.MALFORMED_LINE,
            "",
        ],
    )
    def test_ignored_lines(self, line):
        assert log_parser.parse_log_line(line) is None

    def test_truncated_payload_is_tolerated(self):
        # Payload is cut off mid-JSON; parsing must not fail.
        line = (
            "2026-08-02 15:00:00,000 - INFO - Stream started: provider=local model=Qwen3 "
            "session=abc-123 request=[{'type': 'text', 'text': '<conversation>\\n[User]: "
        )
        ev = log_parser.parse_log_line(line)
        assert ev is not None
        assert ev.session == "abc-123"

    def test_session_unknown_is_unattributed(self):
        ev = log_parser.parse_log_line(fixtures.STREAM_STARTED_SESSION_UNKNOWN)
        assert ev.session is None


# ---------------------------------------------------------------------------
# Session aggregation
# ---------------------------------------------------------------------------


class TestSessionAggregation:
    def test_local_only_session(self):
        lines = [
            fixtures.STREAM_STARTED_LOCAL,
            fixtures.STREAM_FINISHED_LOCAL,
        ]
        # Fixture lines are at 13:58-13:59; use a window that includes them.
        res = aggregation.aggregate(
            _events(lines),
            datetime(2026, 8, 2, 13, 0),
            datetime(2026, 8, 2, 15, 0),
            _schedule(),
        )
        assert len(res.sessions) == 1
        s = res.sessions["019fc284-dcb8-74ca-9a64-9306b6f9d286"]
        assert s.messages == 1
        assert s.initial_provider == "local"
        assert s.initial_model == "Qwen3"
        assert s.start_context_size == 2430
        assert s.avg_context_size == 2430
        assert s.max_context_size == 2430
        assert s.avg_response_size == 120
        assert s.max_response_size == 120
        assert s.remote_move_time is None
        assert s.fallback_reason is None
        assert s.duration_seconds == pytest.approx(35.1, abs=0.01)

    def test_context_stats_multiple_requests(self):
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s1 request=[]",
            "2026-08-02 14:00:01,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s1 provider=local model=Qwen3 request=[]",
            "2026-08-02 14:00:02,000 - INFO - Stream started: provider=local model=Qwen3 session=s1 request=[]",
            "2026-08-02 14:00:03,000 - INFO - Stream finished: reason=stop tokens=300/30/330 session=s1 provider=local model=Qwen3 request=[]",
            "2026-08-02 14:00:04,000 - INFO - Stream started: provider=local model=Qwen3 session=s1 request=[]",
            "2026-08-02 14:00:05,000 - INFO - Stream finished: reason=stop tokens=200/20/220 session=s1 provider=local model=Qwen3 request=[]",
        ]
        res = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, _schedule())
        s = res.sessions["s1"]
        assert s.messages == 3
        assert s.start_context_size == 100
        assert s.avg_context_size == pytest.approx(200.0)
        assert s.max_context_size == 300
        assert s.avg_response_size == pytest.approx(20.0)
        assert s.max_response_size == 30
        assert s.local_requests == 3
        assert s.remote_requests == 0

    def test_fallback_attributed_via_routing_skip(self):
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s2 request=[]",
            "2026-08-02 14:00:05,000 - INFO - Stream finished: reason=stop tokens=900/40/940 session=s2 provider=local model=Qwen3 request=[]",
            "2026-08-02 14:01:06,100 - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 "
            "estimated_tokens=5000 cold_threshold=39594 warm_threshold=39594 new_tokens=50 "
            "cached_ratio=0.50 reason=local_concurrency_limit → skipping local, routing to next remote "
            "provider session=s2",
            "2026-08-02 14:01:06,200 - INFO - Stream started: provider=opencode-go model=deepseek-v4-flash session=s2 request=[]",
            "2026-08-02 14:01:09,000 - INFO - Stream finished: reason=stop tokens=950/200/1150 session=s2 provider=opencode-go model=deepseek-v4-flash request=[]",
        ]
        res = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, _schedule())
        s = res.sessions["s2"]
        assert s.remote_move_time == datetime(2026, 8, 2, 14, 1, 6, 100000)
        assert s.fallback_reason == "local_concurrency_limit"
        assert s.remote_provider == "opencode-go"
        assert s.remote_model == "deepseek-v4-flash"
        assert s.local_requests == 1
        assert s.remote_requests == 1

    def test_fallback_attributed_via_proximity(self):
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s3 request=[]",
            "2026-08-02 14:00:05,000 - INFO - Stream finished: reason=stop tokens=900/40/940 session=s3 provider=local model=Qwen3 request=[]",
            "2026-08-02 14:00:50,000 - INFO - Fallback triggered for model=v1/chat/completions, "
            "from=local-qwen3, to=opencode-go-deepseek, reason=HTTP 400",
            "2026-08-02 14:00:51,000 - INFO - Stream started: provider=deepseek model=deepseek-v4-flash session=s3 request=[]",
            "2026-08-02 14:00:55,000 - INFO - Stream finished: reason=stop tokens=950/200/1150 session=s3 provider=deepseek model=deepseek-v4-flash request=[]",
        ]
        res = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, _schedule())
        s = res.sessions["s3"]
        assert s.remote_move_time == datetime(2026, 8, 2, 14, 0, 50)
        assert s.fallback_reason == "HTTP 400"

    def test_remote_only_session(self):
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Fallback triggered for model=v1/chat/completions, "
            "from=local-qwen3, to=opencode-go-deepseek, reason=warm_cache_bypass",
            "2026-08-02 14:00:02,000 - INFO - Stream started: provider=opencode-go model=deepseek-v4-flash session=s4 request=[]",
            "2026-08-02 14:00:05,000 - INFO - Stream finished: reason=stop tokens=950/200/1150 session=s4 provider=opencode-go model=deepseek-v4-flash request=[]",
        ]
        res = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, _schedule())
        s = res.sessions["s4"]
        assert s.initial_provider == "opencode-go"
        assert s.initial_model == "deepseek-v4-flash"
        assert s.remote_move_time == datetime(2026, 8, 2, 14, 0, 0)
        assert s.fallback_reason == "warm_cache_bypass"
        assert s.local_requests == 0
        assert s.remote_requests == 1

    def test_session_without_fallback_has_empty_move(self):
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s5 request=[]",
            "2026-08-02 14:00:05,000 - INFO - Stream finished: reason=stop tokens=900/40/940 session=s5 provider=local model=Qwen3 request=[]",
        ]
        res = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, _schedule())
        s = res.sessions["s5"]
        assert s.remote_move_time is None
        assert s.fallback_reason is None

    def test_window_filtering_excludes_early_events(self):
        lines = [
            "2026-08-02 13:30:00,000 - INFO - Stream started: provider=local model=Qwen3 session=early request=[]",
            "2026-08-02 13:30:05,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=early provider=local model=Qwen3 request=[]",
            "2026-08-02 14:00:10,000 - INFO - Stream started: provider=local model=Qwen3 session=late request=[]",
        ]
        res = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, _schedule())
        assert "early" not in res.sessions
        assert "late" in res.sessions

    def test_session_starting_before_window_but_active_in_window_is_included(self):
        lines = [
            "2026-08-02 13:30:00,000 - INFO - Stream started: provider=local model=Qwen3 session=span request=[]",
            "2026-08-02 13:30:05,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=span provider=local model=Qwen3 request=[]",
            "2026-08-02 14:00:10,000 - INFO - Stream started: provider=local model=Qwen3 session=span request=[]",
            "2026-08-02 14:00:12,000 - INFO - Stream finished: reason=stop tokens=1300/60/1360 session=span provider=local model=Qwen3 request=[]",
        ]
        res = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, _schedule())
        assert "span" in res.sessions
        s = res.sessions["span"]
        # Only in-window events count: start = first in-window stream.
        assert s.start == datetime(2026, 8, 2, 14, 0, 10)
        assert s.messages == 1
        assert s.start_context_size == 1300

    def test_unattributed_streams_counted(self):
        lines = [
            fixtures.STREAM_FINISHED_TOKENS_NO_SESSION,
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s6 request=[]",
        ]
        res = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, _schedule())
        assert res.unattributed_events == 1
        assert "s6" in res.sessions

    def test_fallback_events_collected_globally(self):
        lines = [fixtures.FALLBACK_CONCURRENCY, fixtures.FALLBACK_WARM_CACHE]
        res = aggregation.aggregate(
            _events(lines),
            datetime(2026, 8, 2, 13, 0),
            datetime(2026, 8, 2, 15, 0),
            _schedule(),
        )
        assert len(res.fallback_events) == 2

    def test_empty_input(self):
        res = aggregation.aggregate([], WINDOW_START, WINDOW_END, _schedule())
        assert res.sessions == {}
        assert res.fallback_events == []
        assert res.dispatch_denied_count == 0

    def test_aggregation_across_two_files_merges_session(self):
        # In-window events for the same session split across a rotated-file
        # boundary merge into one session; pre-window events are ignored.
        ev1 = _events(
            [
                "2026-08-02 13:59:00,000 - INFO - Stream started: provider=local model=Qwen3 session=multi request=[]",
                "2026-08-02 13:59:05,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=multi provider=local model=Qwen3 request=[]",
                "2026-08-02 14:00:05,000 - INFO - Stream started: provider=local model=Qwen3 session=multi request=[]",
            ]
        )
        ev2 = _events(
            [
                "2026-08-02 14:00:10,000 - INFO - Stream finished: reason=stop tokens=200/20/220 session=multi provider=local model=Qwen3 request=[]",
            ]
        )
        res = aggregation.aggregate(ev1 + ev2, WINDOW_START, WINDOW_END, _schedule())
        s = res.sessions["multi"]
        assert s.messages == 1
        assert s.start == datetime(2026, 8, 2, 14, 0, 5)
        assert s.end == datetime(2026, 8, 2, 14, 0, 10)
        assert s.start_context_size == 200
        assert s.max_context_size == 200

    def test_day_night_bucket_assigned_by_start_time(self):
        day_line = "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=day1 request=[]"
        night_line = "2026-08-02 23:59:30,000 - INFO - Stream started: provider=local model=Qwen3 session=night1 request=[]"
        res = aggregation.aggregate(
            _events([day_line, night_line]),
            datetime(2026, 8, 2, 0, 0),
            datetime(2026, 8, 3, 0, 0),
            _schedule(),
        )
        assert res.sessions["day1"].bucket == "day"
        assert res.sessions["day1"].slots == 6
        assert res.sessions["night1"].bucket == "night"
        assert res.sessions["night1"].slots == 8


# ---------------------------------------------------------------------------
# Day/night bucketing from the slot schedule
# ---------------------------------------------------------------------------


class TestBucketing:
    def test_periods_from_schedule(self):
        sch = _schedule()
        periods = sch.periods
        assert len(periods) == 3
        by_start = {p.start_minutes: p for p in periods}
        # [00:00, 10:00) = 8 slots (night)
        assert by_start[0].slots == 8 and by_start[0].label == "night"
        # [10:00, 23:59) = 6 slots (day)
        assert by_start[600].slots == 6 and by_start[600].label == "day"
        # [23:59, 24:00) = 8 slots (night)
        assert by_start[1439].slots == 8 and by_start[1439].label == "night"
        assert sch.day_slots == 6
        assert sch.night_slots == 8

    @pytest.mark.parametrize(
        "ts,expected_label",
        [
            (datetime(2026, 8, 2, 0, 0, 0), "night"),
            (datetime(2026, 8, 2, 9, 59, 59), "night"),
            (datetime(2026, 8, 2, 10, 0, 0), "day"),
            (datetime(2026, 8, 2, 23, 58, 59), "day"),
            (datetime(2026, 8, 2, 23, 59, 0), "night"),
            (datetime(2026, 8, 2, 23, 59, 59), "night"),
        ],
    )
    def test_bucket_boundaries(self, ts, expected_label):
        sch = _schedule()
        period = bucketing.bucket_for_time(sch, ts)
        assert period.label == expected_label

    def test_schedule_disabled_falls_back_to_single_day_bucket(self):
        config = {"slot_schedule": {"enabled": False, "entries": [("23:59", 8), ("10:00", 6)]}}
        sch = bucketing.schedule_from_config(config, default_slots=6)
        assert len(sch.periods) == 1
        assert sch.periods[0].label == "day"
        assert sch.periods[0].slots == 6
        assert sch.night_slots is None

    def test_missing_schedule_falls_back(self):
        sch = bucketing.schedule_from_config(None, default_slots=6)
        assert len(sch.periods) == 1
        assert sch.periods[0].slots == 6

    def test_three_entry_schedule(self):
        sch = bucketing.schedule_from_entries([("12:00", 8), ("10:00", 4), ("14:00", 12)])
        assert sch.day_slots == 4
        assert sch.night_slots == 12
        # 10:00-12:00 is the only period with 4 slots -> day
        assert bucketing.bucket_for_time(sch, datetime(2026, 8, 2, 11, 0)).label == "day"
        assert bucketing.bucket_for_time(sch, datetime(2026, 8, 2, 13, 0)).label == "night"
        assert bucketing.bucket_for_time(sch, datetime(2026, 8, 2, 1, 0)).label == "night"

    def test_equal_slot_counts_all_day(self):
        sch = bucketing.schedule_from_entries([("10:00", 6), ("23:59", 6)])
        assert all(p.label == "day" for p in sch.periods)
        assert sch.night_slots is None

    def test_minute_of_day_uses_fractional_minutes(self):
        assert bucketing.minute_of_day(datetime(2026, 8, 2, 23, 58, 30)) == pytest.approx(1438.5)


# ---------------------------------------------------------------------------
# Recommendation rules
# ---------------------------------------------------------------------------


def _result_with_sessions(sessions: list[dict], fallback_events: list | None = None) -> aggregation.AnalysisResult:
    """Build an AnalysisResult from lightweight session dicts."""
    res = aggregation.AnalysisResult(
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        sessions={},
        fallback_events=fallback_events or [],
        routing_skip_events=[],
        dispatch_denied_count=0,
        unattributed_events=0,
        lines_skipped=0,
        total_lines=0,
    )
    for d in sessions:
        s = aggregation.SessionStats(**d)
        res.sessions[s.session_id] = s
    return res


def _session(
    sid: str,
    messages: int = 1,
    max_context: int | None = 1000,
    remote_move: bool = False,
    local_req: int = 1,
    remote_req: int = 0,
    bucket: str = "day",
    slots: int = 6,
    reason: str | None = None,
) -> dict:
    return {
        "session_id": sid,
        "start": WINDOW_START,
        "end": WINDOW_START + timedelta(minutes=1),
        "duration_seconds": 60.0,
        "messages": messages,
        "local_requests": local_req,
        "remote_requests": remote_req,
        "start_context_size": max_context,
        "avg_context_size": float(max_context) if max_context else None,
        "max_context_size": max_context,
        "avg_response_size": 50.0,
        "max_response_size": 60,
        "initial_provider": "local",
        "initial_model": "Qwen3",
        "remote_provider": "opencode-go" if remote_move else None,
        "remote_model": "deepseek-v4-flash" if remote_move else None,
        "remote_move_time": WINDOW_START + timedelta(seconds=30) if remote_move else None,
        "fallback_reason": reason,
        "bucket": bucket,
        "slots": slots,
        "dispatch_denied": 0,
        "routing_skips": 1 if remote_move else 0,
    }


class TestRecommendations:
    def test_concurrency_limit_dominant_suggests_slot_pool(self):
        res = _result_with_sessions(
            [_session("a", remote_move=True, reason="local_concurrency_limit")],
            fallback_events=[
                log_parser.LogEvent("fallback", WINDOW_START, reason="local_concurrency_limit"),
                log_parser.LogEvent("fallback", WINDOW_START, reason="local_concurrency_limit"),
            ],
        )
        recs = recommendations.generate_recommendations(res, config=None)
        titles = " | ".join(r.title for r in recs)
        assert "session_slot_pool_size" in titles.lower()

    def test_large_context_bypass_suggests_ctx(self):
        res = _result_with_sessions(
            [_session("a", remote_move=True, reason="large_context_bypass")],
        )
        recs = recommendations.generate_recommendations(res, config=None)
        titles = " | ".join(r.title for r in recs)
        assert "ctx" in titles.lower() or "context" in titles.lower()

    def test_warm_cache_bypass_suggests_warm_cache(self):
        res = _result_with_sessions(
            [_session("a", remote_move=True, reason="warm_cache_bypass")],
        )
        recs = recommendations.generate_recommendations(res, config=None)
        titles = " | ".join(r.title for r in recs)
        assert "warm" in titles.lower()

    def test_context_pressure_near_per_slot_limit(self):
        # 80% of per-slot ctx (262144/6 = 43690) is 34952.
        res = _result_with_sessions([_session("a", max_context=40000)])
        recs = recommendations.generate_recommendations(res, config={"local_model_ctx_size": 262144})
        assert any("context" in r.title.lower() for r in recs)

    def test_day_night_imbalance(self):
        sessions = [_session(f"d{i}", bucket="day", remote_move=True, reason="local_concurrency_limit") for i in range(5)]
        sessions += [_session(f"n{i}", bucket="night") for i in range(10)]
        res = _result_with_sessions(sessions)
        recs = recommendations.generate_recommendations(res, config=None)
        assert any("slot schedule" in r.title.lower() for r in recs)

    def test_remote_errors_are_informational(self):
        res = _result_with_sessions(
            [_session("a", remote_move=True, reason="HTTP 400")],
            fallback_events=[log_parser.LogEvent("fallback", WINDOW_START, reason="HTTP 400")],
        )
        recs = recommendations.generate_recommendations(res, config=None)
        assert any(r.severity == "info" and "remote" in r.title.lower() for r in recs)

    def test_low_fallback_reports_no_change_needed(self):
        res = _result_with_sessions([_session("a"), _session("b"), _session("c")])
        recs = recommendations.generate_recommendations(res, config=None)
        assert any("no configuration changes" in r.title.lower() for r in recs)

    def test_recommendations_cite_evidence(self):
        res = _result_with_sessions(
            [_session("a", remote_move=True, reason="local_concurrency_limit")],
            fallback_events=[log_parser.LogEvent("fallback", WINDOW_START, reason="local_concurrency_limit")],
        )
        recs = recommendations.generate_recommendations(res, config=None)
        for r in recs:
            assert r.evidence, f"recommendation {r.title} must cite evidence"
            assert r.title and r.detail

    def test_recommendation_evidence_has_day_night_breakdown(self):
        sessions = [
            _session(f"d{i}", bucket="day", remote_move=True, reason="local_concurrency_limit")
            for i in range(3)
        ] + [
            _session(f"n{i}", bucket="night", remote_move=True, reason="local_concurrency_limit")
            for i in range(2)
        ]
        res = _result_with_sessions(sessions)
        recs = recommendations.generate_recommendations(res, config=None)
        slot = [r for r in recs if "session_slot_pool_size" in r.title.lower()]
        assert slot, "expected slot-contention recommendation"
        assert "Day 3 (60.0%) / Night 2 (40.0%)" in slot[0].evidence

    def test_all_recommendation_evidence_includes_day_night(self):
        sessions = (
            [
                _session(f"d{i}", bucket="day", remote_move=True, reason="local_concurrency_limit", max_context=40000)
                for i in range(5)
            ]
            + [
                _session(f"n{i}", bucket="night", remote_move=True, reason="warm_cache_bypass", max_context=40000)
                for i in range(5)
            ]
            + [_session(f"r{i}", bucket="day", remote_move=True, reason="HTTP 400") for i in range(2)]
        )
        res = _result_with_sessions(sessions)
        recs = recommendations.generate_recommendations(res, config={"local_model_ctx_size": 262144})
        assert recs, "expected recommendations"
        for r in recs:
            assert "Day" in r.evidence and "Night" in r.evidence, f"{r.title}: {r.evidence}"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestConfigLoader:
    def test_yaml_parse(self):
        cfg = config_loader.parse_config_text(fixtures.CONFIG_FRAGMENT)
        assert cfg["session_slot_pool_size"] == 6
        assert cfg["local_model_ctx_size"] == 262144
        assert cfg["local_large_context_warm_cache_threshold"] == 100000
        schedule = cfg["slot_schedule"]
        assert schedule["enabled"] is True
        entries = schedule["entries"]
        assert ("10:00", 6) in entries
        assert ("23:59", 8) in entries

    def test_regex_fallback_parse(self):
        cfg = config_loader._parse_config_text_regex(fixtures.CONFIG_FRAGMENT)
        assert cfg["session_slot_pool_size"] == 6
        assert cfg["local_model_ctx_size"] == 262144
        schedule = cfg["slot_schedule"]
        assert schedule["enabled"] is True
        assert ("10:00", 6) in schedule["entries"]
        assert ("23:59", 8) in schedule["entries"]

    def test_regex_and_yaml_agree(self):
        yaml_cfg = config_loader.parse_config_text(fixtures.CONFIG_FRAGMENT)
        regex_cfg = config_loader._parse_config_text_regex(fixtures.CONFIG_FRAGMENT)
        assert regex_cfg["session_slot_pool_size"] == yaml_cfg["session_slot_pool_size"]
        assert regex_cfg["slot_schedule"]["entries"] == yaml_cfg["slot_schedule"]["entries"]
        assert regex_cfg["local_large_context_warm_cache_threshold"] == yaml_cfg["local_large_context_warm_cache_threshold"]

    def test_missing_file_returns_none(self, tmp_path):
        assert config_loader.load_proxy_config(tmp_path / "nope.yaml") is None

    def test_find_config_path_from_project_root(self, tmp_path):
        (tmp_path / "proxy").mkdir()
        (tmp_path / "proxy" / "config.yaml").write_text("default_model: code\n")
        found = config_loader.find_config_path(start=tmp_path)
        assert found == tmp_path / "proxy" / "config.yaml"

    def test_find_config_path_explicit_wins(self, tmp_path):
        p = tmp_path / "custom.yaml"
        p.write_text("x: 1\n")
        assert config_loader.find_config_path(explicit=str(p)) == p


# ---------------------------------------------------------------------------
# End-to-end run over fixture log files
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def _write_logs(self, tmp_path: Path) -> Path:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        # Rotated file: rotation at 14:00 (name encodes the rotation time).
        (log_dir / "proxy.log.2026-08-02_14").write_text("\n".join(fixtures.E2E_LINES[:3]) + "\n")
        (log_dir / "proxy.log").write_text("\n".join(fixtures.E2E_LINES[3:]) + "\n")
        return log_dir

    def test_end_to_end(self, tmp_path):
        log_dir = self._write_logs(tmp_path)
        out_dir = tmp_path / "usage-reports"

        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=out_dir,
            config=None,
        )

        # Two sessions in the window.
        assert len(result.summary.sessions) == 2

        day_csv = out_dir / "daytime_sessions.csv"
        night_csv = out_dir / "nighttime_sessions.csv"
        report_md = out_dir / "report.md"
        assert day_csv.exists()
        assert night_csv.exists()
        assert report_md.exists()

        with day_csv.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2  # both sessions start at 14:00+ (day bucket)

        header = set(rows[0].keys())
        for col in [
            "session_id",
            "start_time",
            "end_time",
            "duration_seconds",
            "messages",
            "start_context_size",
            "avg_context_size",
            "max_context_size",
            "avg_response_size",
            "max_response_size",
            "initial_provider",
            "initial_model",
            "remote_move_time",
            "fallback_reason",
        ]:
            assert col in header, f"missing CSV column {col}"

        by_id = {r["session_id"]: r for r in rows}
        # S2 fell back: move time + reason populated; S1 local-only: empty.
        assert by_id[fixtures.S2]["fallback_reason"] == "local_concurrency_limit"
        assert by_id[fixtures.S2]["remote_move_time"] != ""
        assert by_id[fixtures.S1]["fallback_reason"] == ""
        assert by_id[fixtures.S1]["remote_move_time"] == ""
        # Only in-window requests count (the 13:30 pre-window event is excluded).
        assert by_id[fixtures.S1]["messages"] == "1"

        # Report contains the key sections.
        md = report_md.read_text()
        for section in ["# Proxy Usage Analysis", "## Recommendations", "local_concurrency_limit"]:
            assert section in md

        # Night CSV has no rows (all sessions start during day hours).
        with night_csv.open() as f:
            assert len(list(csv.DictReader(f))) == 0

    def test_dispatch_denied_attributed_to_session(self, tmp_path):
        log_dir = tmp_path / "logs_d"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text(
            f"2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session={fixtures.S1} request=[]\n"
            f"2026-08-02 14:01:00,000 - INFO - local_dispatch_denied session={fixtures.S1} owner={fixtures.S2} active=4\n"
            f"2026-08-02 14:30:00,000 - INFO - local_dispatch_denied session=33333333-3333-3333-3333-333333333333 owner={fixtures.S2} active=6\n"
        )
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=tmp_path / "out_d",
            config=None,
        )
        assert result.summary.dispatch_denied_count == 2
        # Session-attributed where the UUID matches a stream session...
        assert result.summary.sessions[fixtures.S1].dispatch_denied == 1
        # ...and the report row buckets ALL dispatch events by their timestamp.
        md = (tmp_path / "out_d" / "report.md").read_text()
        section = md.split("## Session summary", 1)[1].split("## ", 1)[0]
        assert "| Dispatch denied | 2 | 2 (100.0%) | 0 (0.0%) |" in section

    def test_json_summary(self, tmp_path):
        log_dir = self._write_logs(tmp_path)
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=tmp_path / "out2",
            config=None,
        )
        data = reporting.summary_to_json(result.summary)
        assert isinstance(data, dict)
        assert data["sessions"] == 2
        assert data["fallback_events"] >= 1
        # Round-trips through json.
        json.dumps(data)

    def test_empty_log_dir(self, tmp_path):
        log_dir = tmp_path / "empty"
        log_dir.mkdir()
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=tmp_path / "out3",
            config=None,
        )
        assert result.summary.sessions == {}
        assert (tmp_path / "out3" / "report.md").exists()

    def test_rotated_file_outside_window_is_not_parsed(self, tmp_path):
        log_dir = tmp_path / "logs2"
        log_dir.mkdir()
        # Rotation time 13:00 < window_start 14:00 → excluded by discovery.
        (log_dir / "proxy.log.2026-08-02_13").write_text(
            "2026-08-02 12:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=old request=[]\n"
        )
        (log_dir / "proxy.log").write_text(
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=new request=[]\n"
        )
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=tmp_path / "out4",
            config=None,
        )
        assert list(result.summary.sessions) == ["new"]


class TestDefaultOutputDir:
    """The default output directory is ~/proxy-usage-reports (expanded)."""

    def test_constant_and_cli_default(self):
        import analyze_proxy_usage as cli

        assert cli.DEFAULT_OUTPUT_DIR == "~/proxy-usage-reports"
        assert cli.parse_args([]).output_dir == "~/proxy-usage-reports"

    def test_main_writes_to_home_dir_by_default(self, tmp_path, monkeypatch):
        import analyze_proxy_usage as cli

        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        log_dir = tmp_path / "logs3"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text("\n".join(fixtures.E2E_LINES) + "\n")

        rc = cli.main(
            [
                "--log-dir",
                str(log_dir),
                "--start",
                "2026-08-02 14:00:00",
                "--end",
                "2026-08-02 15:00:00",
                "--quiet",
            ]
        )
        assert rc == 0
        out = home / "proxy-usage-reports"
        assert out.is_dir()
        assert (out / "report.md").exists()
        assert (out / "daytime_sessions.csv").exists()
        assert (out / "nighttime_sessions.csv").exists()


class TestReportRestructure:
    """Report layout: consolidated tables with total/day/night columns, and
    no per-session fallback list."""

    def _run(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        # Rotated file (rotation 14:00, inside window) + live log.
        (log_dir / "proxy.log.2026-08-02_14").write_text("\n".join(fixtures.E2E_LINES[:3]) + "\n")
        (log_dir / "proxy.log").write_text("\n".join(fixtures.E2E_LINES[3:]) + "\n")
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=tmp_path / "out",
            config=None,
        )
        return (tmp_path / "out" / "report.md").read_text()

    def test_old_sections_removed(self, tmp_path):
        md = self._run(tmp_path)
        for section in [
            "## Session classification",
            "## Local vs remote",
            "## Sessions that fell back",
            "## Daytime vs nighttime",
            "## Context usage",
        ]:
            assert section not in md, f"section should be removed: {section}"

    def test_session_summary_has_total_day_night(self, tmp_path):
        md = self._run(tmp_path)
        section = md.split("## Session summary", 1)[1].split("## ", 1)[0]
        assert "| Metric | Total | Day | Night |" in section
        # Both fixture sessions start during day hours (14:00-15:00 window);
        # day/night cells carry the share of the metric's total.
        assert "| Sessions | 2 | 2 (100.0%) | 0 (0.0%) |" in section
        assert "| Requests |" in section
        assert "| Local requests |" in section
        assert "| Remote requests |" in section
        assert "| Local-only sessions |" in section
        assert "| Fell back (local → remote) |" in section
        assert "| Fallback events |" in section

    def test_fallback_reasons_have_day_night(self, tmp_path):
        md = self._run(tmp_path)
        section = md.split("## Fallback reasons", 1)[1].split("## ", 1)[0]
        assert "| Reason | Total | % of fallbacks | Day | Night |" in section
        assert "| local_concurrency_limit | 1 | 100.0% | 1 (100.0%) | 0 (0.0%) |" in section

    def test_routing_skip_reasons_have_day_night(self, tmp_path):
        md = self._run(tmp_path)
        section = md.split("## routing_skip_local reasons", 1)[1].split("## ", 1)[0]
        assert "| Reason | Total | % of skips | Day | Night |" in section
        assert "| local_concurrency_limit | 1 | 100.0% | 1 (100.0%) | 0 (0.0%) |" in section

    def test_per_model_breakdown_has_day_night(self, tmp_path):
        md = self._run(tmp_path)
        section = md.split("## Per-model breakdown", 1)[1].split("## ", 1)[0]
        assert "| Provider | Model | Sessions | Day | Night | Requests | Fell back |" in section
        assert "| local | Qwen3 | 2 | 2 (100.0%) | 0 (0.0%) |" in section
