"""Unit tests for the proxy-usage-analysis skill.

Covers: log-line parsing, session aggregation, fast/cheap bucketing from the
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

import aggregation
import bucketing
import config_loader
import log_parser
import recommendations
import reporting

from tests import fixtures

WINDOW_START = datetime(2026, 8, 2, 14, 0, 0)
WINDOW_END = datetime(2026, 8, 2, 15, 0, 0)

# Error fixtures live on Aug 3 (10:00-14:00 window covers all five error types).
ERROR_WINDOW_START = datetime(2026, 8, 3, 10, 0, 0)
ERROR_WINDOW_END = datetime(2026, 8, 3, 15, 0, 0)

ERROR_LINES = [
    fixtures.STREAM_FINISHED_ERROR,
    fixtures.STREAM_ERROR_LINE,
    fixtures.SLOT_SAVE_FAILED,
    fixtures.BACKEND_RETRY_TIMEOUT,
    fixtures.UPSTREAM_429,
]


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
        ev = log_parser.parse_log_line(fixtures.ROUTING_SKIP_CONTEXT_TOO_LARGE)
        assert ev.kind == "routing_skip"
        assert ev.reason == "context_too_large"
        assert ev.session == "019fc27d-3a46-7e5c-871e-57ab32f875f3"

    def test_routing_skip_legacy_warm_cache_bypass_normalized(self):
        # Legacy rotated logs carry ``warm_cache_bypass``; the parser maps it
        # to the current ``context_too_large`` reason (LP-0MSF8XDG7000PERM).
        ev = log_parser.parse_log_line(fixtures.ROUTING_SKIP_WARM)
        assert ev.kind == "routing_skip"
        assert ev.reason == "context_too_large"
        assert ev.session == "019fc27d-3a46-7e5c-871e-57ab32f875f3"

    def test_fallback_context_too_large(self):
        ev = log_parser.parse_log_line(fixtures.FALLBACK_CONTEXT_TOO_LARGE)
        assert ev.kind == "fallback"
        assert ev.reason == "context_too_large"

    def test_fallback_legacy_warm_cache_bypass_normalized(self):
        ev = log_parser.parse_log_line(fixtures.FALLBACK_WARM_CACHE)
        assert ev.kind == "fallback"
        assert ev.reason == "context_too_large"
        assert ev.session is None

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


class TestErrorLineParsing:
    """Error events are parsed into distinct error kinds with the fields the
    taxonomy needs (error type, provider/model, session, entry, evidence)."""

    def test_stream_finished_reason_error(self):
        ev = log_parser.parse_log_line(fixtures.STREAM_FINISHED_ERROR)
        assert ev is not None
        assert ev.kind == "stream_finish_error"
        assert ev.error == "finish_reason:error"
        assert ev.session == "019fc52e-05a0-78d5-b59d-bcb91055b787"
        assert ev.provider == "opencode"
        assert ev.model == "deepseek-v4-flash-free"
        assert ev.entry == "opencode-deepseek-free"
        assert ev.raw and "Stream finished: reason=error" in ev.raw

    def test_stream_error_line(self):
        ev = log_parser.parse_log_line(fixtures.STREAM_ERROR_LINE)
        assert ev is not None
        assert ev.kind == "stream_error"
        assert ev.error == "NameError"
        assert ev.provider == "local"
        assert ev.model == "Qwen3"
        assert ev.session == "019fc754-d847-75af-86ea-991480e799d0"

    def test_slot_save_failed(self):
        ev = log_parser.parse_log_line(fixtures.SLOT_SAVE_FAILED)
        assert ev is not None
        assert ev.kind == "slot_save_error"
        assert ev.error == "ReadTimeout/ReadTimeout"
        # Slot persistence always targets the local llama-server.
        assert ev.provider == "local"
        assert ev.model is None

    def test_backend_retry(self):
        ev = log_parser.parse_log_line(fixtures.BACKEND_RETRY_TIMEOUT)
        assert ev is not None
        assert ev.kind == "backend_retry"
        assert ev.error == "ConnectTimeout"
        assert ev.attempt == "1/8"
        assert ev.signal == "connect_failures"

    def test_upstream_429(self):
        ev = log_parser.parse_log_line(fixtures.UPSTREAM_429)
        assert ev is not None
        assert ev.kind == "upstream_http_error"
        assert ev.status == 429
        assert ev.error == "FreeUsageLimitError"
        # Provider is inferred from the target URL; model is not in the line.
        assert ev.provider == "opencode"
        assert ev.model is None

    def test_upstream_url_provider_mapping(self):
        cases = [
            ("url=https://opencode.ai/zen/go/v1/chat/completions", "opencode-go"),
            ("url=https://opencode.ai/zen/v1/chat/completions", "opencode"),
            ("url=https://api.deepseek.com/v1/chat/completions", "deepseek"),
            ("url=https://models.inference.ai.azure.com/v1/chat/completions", "github"),
            # Unknown endpoints fall back to the bare hostname.
            ("url=https://other.example.com/v1/chat/completions", "other.example.com"),
        ]
        for url, expected in cases:
            line = (
                f"2026-08-03 13:58:04,053 - WARNING - [remote] upstream error status=429 "
                f"{url} body={{\"type\":\"error\"}}"
            )
            ev = log_parser.parse_log_line(line)
            assert ev is not None and ev.kind == "upstream_http_error"
            assert ev.provider == expected, f"{url} → {ev.provider}, expected {expected}"

    def test_upstream_error_without_url(self):
        line = "2026-08-03 13:58:04,053 - WARNING - [remote] upstream error status=503 body={}"
        ev = log_parser.parse_log_line(line)
        assert ev is not None and ev.kind == "upstream_http_error"
        assert ev.provider is None
        assert ev.status == 503

    def test_slot_save_success_is_ignored(self):
        assert log_parser.parse_log_line(fixtures.SLOT_SAVE_SUCCESS) is None

    def test_stream_finished_stop_is_not_error(self):
        ev = log_parser.parse_log_line(fixtures.STREAM_FINISHED_STOP)
        assert ev is not None
        assert ev.kind == "stream_finished"
        assert ev.reason == "stop"

    def test_iter_events_attaches_source_file(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "proxy.log.2026-08-03_13").write_text(fixtures.STREAM_ERROR_LINE + "\n")
        events = list(
            log_parser.iter_events(
                log_dir / "proxy.log.2026-08-03_13",
                ERROR_WINDOW_START,
                ERROR_WINDOW_END,
            )
        )
        assert len(events) == 1
        assert events[0].src_file == "proxy.log.2026-08-03_13"


class TestDiscoverLogFiles:
    """Discovery of proxy log files for an analysis window.

    Rotated files (``proxy.log.YYYY-MM-DD_HH``) are included regardless of
    their name-encoded timestamp: in this deployment a rotated file routinely
    holds data well past its encoded rotation time (e.g. ``proxy.log.2026-08-07_03``
    contains data until 09:03), so discovery must never exclude a file based on
    its name. ``iter_events`` per-line timestamp filtering is the only boundary.
    """

    def test_live_log_always_included(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text("garbage\n")
        files = log_parser.discover_log_files(log_dir, WINDOW_START)
        assert files == [log_dir / "proxy.log"]

    def test_rotated_file_with_pre_window_encoded_time_is_included(self, tmp_path):
        # Regression: name-encoded rotation time (03:00 on 08-07) precedes
        # window start (04:00 on 08-07), but the file may still hold in-window
        # data — it must be discovered and filtered per line, never excluded by
        # name. Mirrors the real deployment case (proxy.log.2026-08-07_03).
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "proxy.log.2026-08-07_03").write_text("garbage\n")
        (log_dir / "proxy.log").write_text("garbage\n")
        window_start = datetime(2026, 8, 7, 4, 0)
        files = log_parser.discover_log_files(log_dir, window_start)
        assert log_dir / "proxy.log.2026-08-07_03" in files
        assert log_dir / "proxy.log" in files

    def test_all_rotated_files_included_regardless_of_encoded_time(self, tmp_path):
        # Encoded times before, inside, and after the window are all included;
        # discovery never excludes a rotated file based on its name timestamp.
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        names = [
            "proxy.log.2026-07-31_10",  # long before the window
            "proxy.log.2026-08-02_14",  # inside the window
            "proxy.log.2026-08-03_20",  # after the window
        ]
        for name in names:
            (log_dir / name).write_text("garbage\n")
        files = log_parser.discover_log_files(log_dir, WINDOW_START)
        assert {p.name for p in files} == set(names)

    def test_unrecognized_rotated_names_included(self, tmp_path):
        # Names that do not match the YYYY-MM-DD_HH convention are also kept.
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "proxy.log.1").write_text("garbage\n")
        (log_dir / "proxy.log.gz").write_text("garbage\n")
        files = log_parser.discover_log_files(log_dir, WINDOW_START)
        assert {p.name for p in files} == {"proxy.log.1", "proxy.log.gz"}

    def test_non_log_files_excluded(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "llama-server.log").write_text("garbage\n")
        (log_dir / "notes.txt").write_text("garbage\n")
        (log_dir / "proxy.log").write_text("garbage\n")
        files = log_parser.discover_log_files(log_dir, WINDOW_START)
        assert files == [log_dir / "proxy.log"]

    def test_missing_dir_returns_empty(self, tmp_path):
        assert log_parser.discover_log_files(tmp_path / "nope", WINDOW_START) == []

    def test_sorted_by_name(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        for name in ["proxy.log.2026-08-02_14", "proxy.log.2026-08-01_10", "proxy.log"]:
            (log_dir / name).write_text("garbage\n")
        files = log_parser.discover_log_files(log_dir, WINDOW_START)
        assert [p.name for p in files] == [
            "proxy.log",
            "proxy.log.2026-08-01_10",
            "proxy.log.2026-08-02_14",
        ]


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
            # Legacy reason value in a rotated log: still classified correctly
            # as ``context_too_large`` (LP-0MSF8XDG7000PERM backward compat).
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
        assert s.fallback_reason == "context_too_large"
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

    def test_fast_cheap_bucket_assigned_by_start_time(self):
        day_line = "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=fast1 request=[]"
        night_line = "2026-08-02 23:59:30,000 - INFO - Stream started: provider=local model=Qwen3 session=cheap1 request=[]"
        res = aggregation.aggregate(
            _events([day_line, night_line]),
            datetime(2026, 8, 2, 0, 0),
            datetime(2026, 8, 3, 0, 0),
            _schedule(),
        )
        assert res.sessions["fast1"].bucket == "fast"
        assert res.sessions["fast1"].slots == 6
        assert res.sessions["cheap1"].bucket == "cheap"
        assert res.sessions["cheap1"].slots == 8


class TestErrorAggregation:
    """Error events are collected into ``AnalysisResult.error_events`` and
    countable by error type / provider / model."""

    def test_error_events_collected(self):
        res = aggregation.aggregate(
            _events(ERROR_LINES), ERROR_WINDOW_START, ERROR_WINDOW_END, _schedule()
        )
        kinds = sorted(e.kind for e in res.error_events)
        assert kinds == [
            "backend_retry",
            "slot_save_error",
            "stream_error",
            "stream_finish_error",
            "upstream_http_error",
        ]
        assert res.error_events[0].raw
        assert res.error_events[0].ts >= ERROR_WINDOW_START
        assert res.error_events[0].ts <= ERROR_WINDOW_END

    def test_error_counts_by_type(self):
        res = aggregation.aggregate(
            _events(ERROR_LINES * 2), ERROR_WINDOW_START, ERROR_WINDOW_END, _schedule()
        )
        counts = res.error_counts
        assert counts["stream_finish_error"] == 2
        assert counts["stream_error"] == 2
        assert counts["slot_save_error"] == 2
        assert counts["backend_retry"] == 2
        assert counts["upstream_http_error"] == 2

    def test_error_provider_model_breakdown(self):
        res = aggregation.aggregate(
            _events(ERROR_LINES), ERROR_WINDOW_START, ERROR_WINDOW_END, _schedule()
        )
        breakdown = res.error_provider_model_counts
        # stream_finish_error: opencode / deepseek-v4-flash-free; stream_error: local / Qwen3.
        assert breakdown[("stream_finish_error", "opencode", "deepseek-v4-flash-free")] == 1
        assert breakdown[("stream_error", "local", "Qwen3")] == 1
        # slot_save is attributed to the local provider (model not in the line);
        # backend_retry carries no provider/model; upstream HTTP errors get their
        # provider from the target URL (opencode.ai/zen → opencode).
        assert breakdown[("slot_save_error", "local", None)] == 1
        assert breakdown[("backend_retry", None, None)] == 1
        assert breakdown[("upstream_http_error", "opencode", None)] == 1

    def test_error_events_outside_window_excluded(self):
        _ = aggregation.aggregate(
            _events(ERROR_LINES), ERROR_WINDOW_START, ERROR_WINDOW_END, _schedule()
        )
        # All fixtures are inside the Aug 3 10:00-15:00 window; also verify a
        # window that excludes them yields no error events.
        res2 = aggregation.aggregate(
            _events(ERROR_LINES),
            datetime(2026, 8, 1, 0, 0),
            datetime(2026, 8, 1, 23, 59),
            _schedule(),
        )
        assert res2.error_events == []

    def test_error_events_not_counted_as_lines_skipped(self):
        res = aggregation.aggregate(
            _events(ERROR_LINES), ERROR_WINDOW_START, ERROR_WINDOW_END, _schedule()
        )
        assert res.lines_skipped == 0


# ---------------------------------------------------------------------------
# Fast/cheap bucketing from the slot schedule
# ---------------------------------------------------------------------------


class TestBucketing:
    def test_periods_from_schedule(self):
        sch = _schedule()
        periods = sch.periods
        assert len(periods) == 3
        by_start = {p.start_minutes: p for p in periods}
        # [00:00, 10:00) = 8 slots (cheap)
        assert by_start[0].slots == 8 and by_start[0].label == "cheap"
        # [10:00, 23:59) = 6 slots (fast)
        assert by_start[600].slots == 6 and by_start[600].label == "fast"
        # [23:59, 24:00) = 8 slots (cheap)
        assert by_start[1439].slots == 8 and by_start[1439].label == "cheap"
        assert sch.fast_slots == 6
        assert sch.cheap_slots == 8

    @pytest.mark.parametrize(
        "ts,expected_label",
        [
            (datetime(2026, 8, 2, 0, 0, 0), "cheap"),
            (datetime(2026, 8, 2, 9, 59, 59), "cheap"),
            (datetime(2026, 8, 2, 10, 0, 0), "fast"),
            (datetime(2026, 8, 2, 23, 58, 59), "fast"),
            (datetime(2026, 8, 2, 23, 59, 0), "cheap"),
            (datetime(2026, 8, 2, 23, 59, 59), "cheap"),
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
        assert sch.periods[0].label == "fast"
        assert sch.periods[0].slots == 6
        assert sch.cheap_slots is None

    def test_missing_schedule_falls_back(self):
        sch = bucketing.schedule_from_config(None, default_slots=6)
        assert len(sch.periods) == 1
        assert sch.periods[0].slots == 6

    def test_three_entry_schedule(self):
        sch = bucketing.schedule_from_entries([("12:00", 8), ("10:00", 4), ("14:00", 12)])
        assert sch.fast_slots == 4
        assert sch.cheap_slots == 12
        # 10:00-12:00 is the only period with 4 slots -> fast
        assert bucketing.bucket_for_time(sch, datetime(2026, 8, 2, 11, 0)).label == "fast"
        assert bucketing.bucket_for_time(sch, datetime(2026, 8, 2, 13, 0)).label == "cheap"
        assert bucketing.bucket_for_time(sch, datetime(2026, 8, 2, 1, 0)).label == "cheap"

    def test_equal_slot_counts_all_day(self):
        sch = bucketing.schedule_from_entries([("10:00", 6), ("23:59", 6)])
        assert all(p.label == "fast" for p in sch.periods)
        assert sch.cheap_slots is None

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
    bucket: str = "fast",
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

    def test_context_too_large_suggests_context(self):
        res = _result_with_sessions(
            [_session("a", remote_move=True, reason="context_too_large")],
        )
        recs = recommendations.generate_recommendations(res, config=None)
        titles = " | ".join(r.title for r in recs)
        assert "context" in titles.lower()

    def test_context_pressure_near_per_slot_limit(self):
        # 80% of per-slot ctx (262144/6 = 43690) is 34952.
        res = _result_with_sessions([_session("a", max_context=40000)])
        recs = recommendations.generate_recommendations(res, config={"local_model_ctx_size": 262144})
        assert any("context" in r.title.lower() for r in recs)

    def test_fast_cheap_imbalance(self):
        sessions = [_session(f"f{i}", bucket="fast", remote_move=True, reason="local_concurrency_limit") for i in range(5)]
        sessions += [_session(f"c{i}", bucket="cheap") for i in range(10)]
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


class TestErrorRecommendations:
    """Error-driven remediation recommendations are quantified from the
    parsed error events and link to the known follow-up work items."""

    @staticmethod
    def _res_with_errors(errors: list[log_parser.LogEvent]) -> aggregation.AnalysisResult:
        res = aggregation.AnalysisResult(
            window_start=ERROR_WINDOW_START,
            window_end=ERROR_WINDOW_END,
            sessions={},
            fallback_events=[],
            routing_skip_events=[],
            dispatch_denied_count=0,
            unattributed_events=0,
            lines_skipped=0,
            total_lines=0,
            error_events=errors,
        )
        return res

    def test_stream_errors_trigger_recovery_first_recommendation(self):
        errors = [
            log_parser.LogEvent("stream_finish_error", ERROR_WINDOW_START, provider="opencode-go", model="deepseek-v4-flash")
            for _ in range(3)
        ]
        res = self._res_with_errors(errors)
        recs = recommendations.generate_recommendations(res, config=None)
        titles = " | ".join(r.title.lower() for r in recs)
        assert "recovery-first" in titles, f"expected recovery-first recommendation, got: {titles}"
        assert "LP-0MSDP2PDB004GV86" in " | ".join(r.detail for r in recs)

    def test_slot_save_errors_trigger_ctx_pressure_recommendation(self):
        errors = [
            log_parser.LogEvent("slot_save_error", ERROR_WINDOW_START, error="ReadTimeout/ReadTimeout")
            for _ in range(3)
        ]
        res = self._res_with_errors(errors)
        recs = recommendations.generate_recommendations(res, config=None)
        titles = " | ".join(r.title.lower() for r in recs)
        assert "slot_save" in titles, f"expected slot_save recommendation, got: {titles}"
        assert "LP-0MSAOQTJS000FFVM" in " | ".join(r.detail for r in recs)

    def test_upstream_429_triggers_cooldown_recommendation(self):
        errors = [
            log_parser.LogEvent("upstream_http_error", ERROR_WINDOW_START, error="FreeUsageLimitError", status=429)
            for _ in range(3)
        ]
        res = self._res_with_errors(errors)
        recs = recommendations.generate_recommendations(res, config=None)
        titles = " | ".join(r.title.lower() for r in recs)
        assert "429" in titles, f"expected 429/cooldown recommendation, got: {titles}"
        assert "LP-0MRGU0I91006ODFD" in " | ".join(r.detail for r in recs)

    def test_backend_retry_errors_are_informational(self):
        errors = [
            log_parser.LogEvent("backend_retry", ERROR_WINDOW_START, error="ConnectTimeout")
            for _ in range(3)
        ]
        res = self._res_with_errors(errors)
        recs = recommendations.generate_recommendations(res, config=None)
        assert any(r.severity == "info" and "backend_retry" in r.title.lower() for r in recs)

    def test_error_recommendations_cite_evidence_with_counts(self):
        errors = [
            log_parser.LogEvent("stream_finish_error", ERROR_WINDOW_START, provider="opencode-go", model="deepseek-v4-flash")
            for _ in range(5)
        ]
        res = self._res_with_errors(errors)
        recs = recommendations.generate_recommendations(res, config=None)
        stream_recs = [r for r in recs if "recovery-first" in r.title.lower()]
        assert stream_recs, "expected a recovery-first recommendation"
        assert "5" in stream_recs[0].evidence
        assert stream_recs[0].evidence  # evidence non-empty

    def test_no_error_events_do_not_trigger_error_recommendations(self):
        res = self._res_with_errors([])
        recs = recommendations.generate_recommendations(res, config=None)
        for r in recs:
            assert "recovery-first" not in r.title.lower()
            assert "slot_save" not in r.title.lower()
            assert "429" not in r.title.lower()

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

    def test_recommendation_evidence_has_fast_cheap_breakdown(self):
        sessions = [
            _session(f"f{i}", bucket="fast", remote_move=True, reason="local_concurrency_limit")
            for i in range(3)
        ] + [
            _session(f"c{i}", bucket="cheap", remote_move=True, reason="local_concurrency_limit")
            for i in range(2)
        ]
        res = _result_with_sessions(sessions)
        recs = recommendations.generate_recommendations(res, config=None)
        slot = [r for r in recs if "session_slot_pool_size" in r.title.lower()]
        assert slot, "expected slot-contention recommendation"
        assert "Fast 3 (60.0%) / Cheap 2 (40.0%)" in slot[0].evidence

    def test_all_recommendation_evidence_includes_fast_cheap(self):
        sessions = (
            [
                _session(f"f{i}", bucket="fast", remote_move=True, reason="local_concurrency_limit", max_context=40000)
                for i in range(5)
            ]
            + [
                _session(f"c{i}", bucket="cheap", remote_move=True, reason="context_too_large", max_context=40000)
                for i in range(5)
            ]
            + [_session(f"r{i}", bucket="fast", remote_move=True, reason="HTTP 400") for i in range(2)]
        )
        res = _result_with_sessions(sessions)
        recs = recommendations.generate_recommendations(res, config={"local_model_ctx_size": 262144})
        assert recs, "expected recommendations"
        for r in recs:
            assert "Fast" in r.evidence and "Cheap" in r.evidence, f"{r.title}: {r.evidence}"


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

    def test_find_config_path_prefers_mode_selected_profile(self, tmp_path):
        """A persisted cheap mode selects config-cheap.yaml (LP-0MSLMYEEU002IBH6)."""
        (tmp_path / "proxy").mkdir()
        (tmp_path / "proxy" / "config.yaml").write_text("default_model: code\n")
        (tmp_path / "proxy" / "config-cheap.yaml").write_text("default_model: code\n")
        (tmp_path / "proxy" / "config-fast.yaml").write_text("default_model: code\n")
        (tmp_path / "proxy" / ".mode").write_text("cheap\n")
        found = config_loader.find_config_path(start=tmp_path)
        assert found == tmp_path / "proxy" / "config-cheap.yaml"

    def test_find_config_path_mode_fast_selects_fast_profile(self, tmp_path):
        """A persisted fast mode selects config-fast.yaml."""
        (tmp_path / "proxy").mkdir()
        (tmp_path / "proxy" / "config.yaml").write_text("default_model: code\n")
        (tmp_path / "proxy" / "config-fast.yaml").write_text("default_model: code\n")
        (tmp_path / "proxy" / ".mode").write_text("fast\n")
        found = config_loader.find_config_path(start=tmp_path)
        assert found == tmp_path / "proxy" / "config-fast.yaml"

    def test_find_config_path_missing_mode_defaults_to_config_yaml(self, tmp_path):
        """No .mode file -> the default config.yaml is used."""
        (tmp_path / "proxy").mkdir()
        (tmp_path / "proxy" / "config.yaml").write_text("default_model: code\n")
        (tmp_path / "proxy" / "config-cheap.yaml").write_text("default_model: code\n")
        found = config_loader.find_config_path(start=tmp_path)
        assert found == tmp_path / "proxy" / "config.yaml"

    def test_find_config_path_invalid_mode_ignored(self, tmp_path):
        """A garbage .mode value falls back to config.yaml (fail-open)."""
        (tmp_path / "proxy").mkdir()
        (tmp_path / "proxy" / "config.yaml").write_text("default_model: code\n")
        (tmp_path / "proxy" / ".mode").write_text("garbage\n")
        found = config_loader.find_config_path(start=tmp_path)
        assert found == tmp_path / "proxy" / "config.yaml"

    def test_find_config_path_mode_config_missing_falls_back(self, tmp_path):
        """Mode selects a profile that does not exist -> config.yaml."""
        (tmp_path / "proxy").mkdir()
        (tmp_path / "proxy" / "config.yaml").write_text("default_model: code\n")
        (tmp_path / "proxy" / ".mode").write_text("cheap\n")
        found = config_loader.find_config_path(start=tmp_path)
        assert found == tmp_path / "proxy" / "config.yaml"


# ---------------------------------------------------------------------------
# End-to-end run over fixture log files
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Local model utilization (busy-time) stats
# ---------------------------------------------------------------------------


def _local_event(kind: str, ts: datetime, session: str = "s1") -> log_parser.LogEvent:
    """A local stream start/finish event with the fields busy-time needs."""
    return log_parser.LogEvent(
        kind,
        ts,
        provider="local",
        model="Qwen3",
        session=session,
    )


class TestBusyStats:
    def test_sequential_streams_single_session(self):
        events = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 10), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 15), "s1"),
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 20), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 25), "s1"),
        ]
        busy = aggregation.compute_busy_stats(events, WINDOW_START, WINDOW_END, _schedule())
        assert busy is not None
        assert busy.streams == 2
        assert busy.busy_seconds == 10.0
        assert busy.total_compute_seconds == 10.0
        assert busy.peak_concurrency == 1
        assert busy.avg_concurrency == 1.0
        assert busy.avg_stream_duration == 5.0
        assert busy.unfinished_streams == 0
        assert busy.busy_pct == pytest.approx(10.0 / 3600 * 100)

    def test_overlapping_streams_two_sessions(self):
        events = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 0), "s1"),
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 5), "s2"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 10), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 20), "s2"),
        ]
        busy = aggregation.compute_busy_stats(events, WINDOW_START, WINDOW_END, _schedule())
        assert busy is not None
        # Union is 14:00:00-14:00:20; the two streams overlap for 5s.
        assert busy.busy_seconds == 20.0
        assert busy.total_compute_seconds == 25.0
        assert busy.peak_concurrency == 2
        assert busy.avg_concurrency == pytest.approx(25.0 / 20.0)

    def test_streams_crossing_window_boundaries_are_clipped(self):
        events = [
            # Started before the window, finished inside.
            _local_event("stream_started", datetime(2026, 8, 2, 13, 59, 50), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 5), "s1"),
            # Started inside, finished after the window.
            _local_event("stream_started", datetime(2026, 8, 2, 14, 59, 55), "s2"),
            _local_event("stream_finished", datetime(2026, 8, 2, 15, 0, 5), "s2"),
        ]
        busy = aggregation.compute_busy_stats(events, WINDOW_START, WINDOW_END, _schedule())
        assert busy is not None
        assert busy.busy_seconds == 10.0  # 5s clipped at the start + 5s clipped at the end
        assert busy.streams == 2
        assert busy.peak_concurrency == 1

    def test_unfinished_streams_excluded_with_caveat_count(self):
        events = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 0), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 10), "s1"),
            _local_event("stream_started", datetime(2026, 8, 2, 14, 30, 0), "s2"),
        ]
        busy = aggregation.compute_busy_stats(events, WINDOW_START, WINDOW_END, _schedule())
        assert busy is not None
        assert busy.streams == 1
        assert busy.busy_seconds == 10.0
        assert busy.unfinished_streams == 1

    def test_no_local_traffic_returns_none(self):
        events = [
            log_parser.LogEvent("stream_started", datetime(2026, 8, 2, 14, 0, 0), provider="opencode-go"),
            log_parser.LogEvent("stream_finished", datetime(2026, 8, 2, 14, 0, 5), provider="opencode-go"),
        ]
        assert aggregation.compute_busy_stats(events, WINDOW_START, WINDOW_END, _schedule()) is None

    def test_remote_streams_do_not_count_towards_busy(self):
        events = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 0), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 10), "s1"),
            log_parser.LogEvent("stream_started", datetime(2026, 8, 2, 14, 0, 20), provider="opencode-go"),
            log_parser.LogEvent("stream_finished", datetime(2026, 8, 2, 14, 0, 25), provider="opencode-go"),
        ]
        busy = aggregation.compute_busy_stats(events, WINDOW_START, WINDOW_END, _schedule())
        assert busy is not None
        assert busy.streams == 1
        assert busy.busy_seconds == 10.0

    def test_hourly_and_day_night_attribution(self):
        # Window 09:00-11:00; schedule 10:00 -> 6 slots (fast), else 8 (cheap).
        start = datetime(2026, 8, 2, 9, 0, 0)
        end = datetime(2026, 8, 2, 11, 0, 0)
        events = [
            _local_event("stream_started", datetime(2026, 8, 2, 9, 30, 0), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 9, 30, 10), "s1"),
            _local_event("stream_started", datetime(2026, 8, 2, 10, 0, 0), "s2"),
            _local_event("stream_finished", datetime(2026, 8, 2, 10, 0, 15), "s2"),
        ]
        busy = aggregation.compute_busy_stats(events, start, end, _schedule())
        assert busy is not None
        assert busy.busy_seconds == 25.0
        assert busy.cheap_busy_seconds == 10.0
        assert busy.fast_busy_seconds == 15.0
        assert busy.cheap_window_seconds == 3600.0
        assert busy.fast_window_seconds == 3600.0
        hourly = dict(busy.hourly_busy)
        assert hourly[9] == 10.0
        assert hourly[10] == 15.0

    def test_aggregate_populates_busy(self):
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s1 request=[]",
            "2026-08-02 14:00:10,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s1 provider=local model=Qwen3 request=[]",
            "2026-08-02 14:00:05,000 - INFO - Stream started: provider=local model=Qwen3 session=s2 request=[]",
            "2026-08-02 14:00:20,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s2 provider=local model=Qwen3 request=[]",
        ]
        res = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, _schedule())
        assert res.busy is not None
        assert res.busy.streams == 2
        assert res.busy.busy_seconds == 20.0
        assert res.busy.peak_concurrency == 2


class TestIterEventsMargin:
    def test_margin_yields_events_just_outside_window(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_file = log_dir / "proxy.log"
        log_file.write_text(
            "2026-08-02 13:59:50,000 - INFO - Stream started: provider=local model=Qwen3 session=s1 request=[]\n"
            "2026-08-02 15:00:05,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s1 provider=local model=Qwen3 request=[]\n"
        )
        # Without a margin both lines fall outside the window.
        events = list(log_parser.iter_events(log_file, WINDOW_START, WINDOW_END))
        assert len(events) == 0
        # With a margin both boundary-crossing events are yielded.
        events = list(
            log_parser.iter_events(log_file, WINDOW_START, WINDOW_END, margin=timedelta(minutes=1))
        )
        assert len(events) == 2
        assert {e.kind for e in events} == {"stream_started", "stream_finished"}


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

        fast_csv = out_dir / "fast_sessions.csv"
        cheap_csv = out_dir / "cheap_sessions.csv"
        report_md = out_dir / "report.md"
        assert fast_csv.exists()
        assert cheap_csv.exists()
        assert report_md.exists()

        with fast_csv.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2  # both sessions start at 14:00+ (fast bucket)

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
            "decode_tok_s",
        ]:
            assert col in header, f"missing CSV column {col}"

        # decode_tok_s is derivable for S1/S2 (local completion tokens +
        # local active span) and empty for sessions with no local traffic.
        by_id = {r["session_id"]: r for r in rows}
        assert by_id[fixtures.S1]["decode_tok_s"] != ""
        assert by_id[fixtures.S2]["decode_tok_s"] != ""
        # S2 fell back: move time + reason populated; S1 local-only: empty.
        assert by_id[fixtures.S2]["fallback_reason"] == "local_concurrency_limit"
        assert by_id[fixtures.S2]["remote_move_time"] != ""
        assert by_id[fixtures.S1]["fallback_reason"] == ""
        assert by_id[fixtures.S1]["remote_move_time"] == ""
        # Only in-window requests count (the 13:30 pre-window event is excluded).
        assert by_id[fixtures.S1]["messages"] == "1"

        # Report contains the key sections.
        md = report_md.read_text()
        for section in ["# Proxy Usage Analysis", "## Recommendations", "local_concurrency_limit", "## Local model utilization"]:
            assert section in md
        # Busy-time section reflects the fixture streams: S1 2s (14:00:10-12)
        # + S2 5s (14:01:00-05) = 7s busy over a 1h window (0.2%).
        util = md.split("## Local model utilization", 1)[1].split("## ", 1)[0]
        assert "7s" in util and "0.2%" in util
        # No llama-server logs in this fixture dir → speed sections render empty.
        assert "## Decode speed" in md
        assert "## Prompt eval speed" in md
        assert "No llama-server eval timing samples" in md

        # Cheap CSV has no rows (all sessions start during fast hours).
        with cheap_csv.open() as f:
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
        # Local-model utilization stats are exposed.
        busy = data["local_busy"]
        assert busy["streams"] == 2
        assert busy["busy_seconds"] == 7.0
        assert busy["busy_pct"] == pytest.approx(7.0 / 3600 * 100, abs=0.01)
        assert busy["peak_concurrency"] == 1
        # Round-trips through json.
        json.dumps(data)

    def test_error_report_section_and_artifacts(self, tmp_path):
        log_dir = tmp_path / "logs_err"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text("\n".join(ERROR_LINES) + "\n")
        out_dir = tmp_path / "out_err"
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=ERROR_WINDOW_START,
            window_end=ERROR_WINDOW_END,
            output_dir=out_dir,
            config=None,
        )
        assert len(result.summary.error_events) == 5

        report_md = (out_dir / "report.md").read_text()
        assert "## Error analysis" in report_md
        assert "stream_finish_error" in report_md
        assert "Provider/model breakdown" in report_md
        # The breakdown table carries provider and model cells per error type.
        assert "| opencode | deepseek-v4-flash-free |" in report_md
        assert "| local | Qwen3 |" in report_md
        assert "slot_save_error" in report_md
        assert "upstream_http_error" in report_md
        assert "FreeUsageLimitError" in report_md

        errors_csv = out_dir / "errors.csv"
        assert errors_csv.exists()
        with errors_csv.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5
        header = set(rows[0].keys())
        for col in ["error_type", "timestamp", "provider", "model", "session", "entry", "evidence"]:
            assert col in header, f"missing errors.csv column {col}"
        by_type = {r["error_type"]: r for r in rows}
        assert by_type["stream_finish_error"]["provider"] == "opencode"
        assert by_type["upstream_http_error"]["status"] == "429"
        assert by_type["upstream_http_error"]["provider"] == "opencode"
        assert by_type["slot_save_error"]["provider"] == "local"

        errors_json = out_dir / "errors.json"
        assert errors_json.exists()
        data = json.loads(errors_json.read_text())
        assert data["total"] == 5
        assert data["by_type"]["stream_finish_error"] == 1
        assert data["by_type"]["upstream_http_error"] == 1
        # Provider/model breakdown mirrors the report table.
        bpm = data["by_provider_model"]
        assert bpm["stream_finish_error"]["opencode"]["deepseek-v4-flash-free"] == 1
        assert bpm["stream_error"]["local"]["Qwen3"] == 1
        assert bpm["slot_save_error"]["local"]["(unknown)"] == 1
        assert bpm["upstream_http_error"]["opencode"]["(unknown)"] == 1
        assert bpm["backend_retry"]["(unknown)"]["(unknown)"] == 1

    def test_error_json_summary_counts(self, tmp_path):
        log_dir = tmp_path / "logs_err2"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text("\n".join(ERROR_LINES) + "\n")
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=ERROR_WINDOW_START,
            window_end=ERROR_WINDOW_END,
            output_dir=tmp_path / "out_err2",
            config=None,
        )
        data = reporting.summary_to_json(result.summary)
        assert data["errors"] == 5
        assert data["errors_by_type"]["stream_finish_error"] == 1
        # Provider/model breakdown is exposed in the JSON summary.
        bpm = data["errors_by_provider_model"]
        assert bpm["stream_error"]["local"]["Qwen3"] == 1
        assert bpm["upstream_http_error"]["opencode"]["(unknown)"] == 1
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

    def test_rotated_file_outside_window_is_filtered_per_line(self, tmp_path):
        log_dir = tmp_path / "logs2"
        log_dir.mkdir()
        # Rotation time 13:00 < window_start 14:00: the file is still discovered
        # (name timestamps are not authoritative), but its 12:00 content lies
        # outside the margin-widened window and is dropped per line by iter_events.
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
        # The rotated file is discovered, but its pre-window event is filtered
        # by the authoritative per-line timestamp check in iter_events.
        assert log_dir / "proxy.log.2026-08-02_13" in result.files
        assert list(result.summary.sessions) == ["new"]

    def test_rotated_file_with_earlier_rotation_reports_overlapping_session(self, tmp_path):
        # Regression: a rotated file whose name-encoded rotation time precedes
        # window start may still hold in-window data (deployment files span far
        # past their encoded time). Discovery must include it so its in-window
        # session is reported — never silently dropped.
        log_dir = tmp_path / "logs3"
        log_dir.mkdir()
        (log_dir / "proxy.log.2026-08-02_13").write_text(
            "2026-08-02 14:00:10,000 - INFO - Stream started: provider=local model=Qwen3 session=carried request=[]\n"
            "2026-08-02 14:00:12,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=carried provider=local model=Qwen3 request=[]\n"
        )
        (log_dir / "proxy.log").write_text(
            "2026-08-02 14:30:00,000 - INFO - Stream started: provider=local model=Qwen3 session=live request=[]\n"
            "2026-08-02 14:30:02,000 - INFO - Stream finished: reason=stop tokens=200/20/220 session=live provider=local model=Qwen3 request=[]\n"
        )
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=tmp_path / "out5",
            config=None,
        )
        assert log_dir / "proxy.log.2026-08-02_13" in result.files
        assert set(result.summary.sessions) == {"carried", "live"}


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
        assert (out / "fast_sessions.csv").exists()
        assert (out / "cheap_sessions.csv").exists()


class TestReportRestructure:
    """Report layout: consolidated tables with total/fast/cheap columns, and
    no per-session fallback list."""

    def _run(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        # Rotated file (rotation 14:00, inside window) + live log.
        (log_dir / "proxy.log.2026-08-02_14").write_text("\n".join(fixtures.E2E_LINES[:3]) + "\n")
        (log_dir / "proxy.log").write_text("\n".join(fixtures.E2E_LINES[3:]) + "\n")
        _ = reporting.run_analysis(
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
            "## Fast vs cheap",
            "## Context usage",
        ]:
            assert section not in md, f"section should be removed: {section}"

    def test_session_summary_has_total_day_night(self, tmp_path):
        md = self._run(tmp_path)
        section = md.split("## Session summary", 1)[1].split("## ", 1)[0]
        assert "| Metric | Total | Fast | Cheap |" in section
        # Both fixture sessions start during day hours (14:00-15:00 window);
        # fast/cheap cells carry the share of the metric's total.
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
        assert "| Reason | Total | % of fallbacks | Fast | Cheap |" in section
        assert "| local_concurrency_limit | 1 | 100.0% | 1 (100.0%) | 0 (0.0%) |" in section

    def test_routing_skip_reasons_have_day_night(self, tmp_path):
        md = self._run(tmp_path)
        section = md.split("## routing_skip_local reasons", 1)[1].split("## ", 1)[0]
        assert "| Reason | Total | % of skips | Fast | Cheap |" in section
        assert "| local_concurrency_limit | 1 | 100.0% | 1 (100.0%) | 0 (0.0%) |" in section

    def test_per_model_breakdown_has_day_night(self, tmp_path):
        md = self._run(tmp_path)
        section = md.split("## Per-model breakdown", 1)[1].split("## ", 1)[0]
        assert "| Provider | Model | Sessions | Fast | Cheap | Requests | Fell back |" in section
        assert "| local | Qwen3 | 2 | 2 (100.0%) | 0 (0.0%) |" in section


class TestArchiveOutputs:
    """Existing output artifacts are moved into dated archive subdirectories
    before a fresh run overwrites them, so historical reports are kept.

    Covers AC1 (pre-existing artifacts archived), AC2 (same-day runs get
    suffixed dirs, never overwriting), AC3 (pristine dir untouched).
    """

    # Frozen clock so the dated archive dirs are deterministic in tests.
    NOW = datetime(2026, 8, 7, 5, 0, 0)

    class _FrozenClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return TestArchiveOutputs.NOW

    def _write_logs(self, tmp_path: Path) -> Path:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text("\n".join(fixtures.E2E_LINES) + "\n")
        return log_dir

    def _run(self, tmp_path: Path, out_name: str = "out", monkeypatch=None):
        if monkeypatch is not None:
            monkeypatch.setattr(reporting, "datetime", self._FrozenClock)
        return reporting.run_analysis(
            log_dir=self._write_logs(tmp_path),
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=tmp_path / out_name,
            config=None,
        )

    def test_archives_existing_artifacts_before_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.setattr(reporting, "datetime", self._FrozenClock)
        out = tmp_path / "out"
        out.mkdir()
        # Pre-existing artifacts, as if a previous run had written them.
        (out / "report.md").write_text("OLD REPORT")
        (out / "fast_sessions.csv").write_text("old fast")
        (out / "cheap_sessions.csv").write_text("old cheap")
        (out / "errors.csv").write_text("old errors")
        (out / "errors.json").write_text("{\"old\": true}")

        run = self._run(tmp_path, monkeypatch=monkeypatch)

        # All five artifacts moved into the dated archive dir, verbatim.
        archive = out / "2026-08-07"
        assert archive.is_dir()
        assert run.archived_to == archive
        assert (archive / "report.md").read_text() == "OLD REPORT"
        assert (archive / "fast_sessions.csv").read_text() == "old fast"
        assert (archive / "cheap_sessions.csv").read_text() == "old cheap"
        assert (archive / "errors.csv").read_text() == "old errors"
        assert (archive / "errors.json").read_text() == "{\"old\": true}"
        # Fresh outputs written at the root.
        assert (out / "report.md").read_text().startswith("# Proxy Usage Analysis")
        assert (out / "fast_sessions.csv").exists()
        assert (out / "cheap_sessions.csv").exists()
        assert (out / "errors.csv").exists()
        assert (out / "errors.json").exists()

    def test_same_day_runs_get_suffixed_archive_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(reporting, "datetime", self._FrozenClock)
        out = tmp_path / "out"
        log_dir = self._write_logs(tmp_path)
        # Run 1: pristine dir -> nothing archived, first archives appear later.
        reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=out,
            config=None,
        )
        assert list(out.iterdir())  # fresh outputs present
        first_md = (out / "report.md").read_text()

        # Run 2 (same day): previous outputs move into the plain dated dir.
        reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=out,
            config=None,
        )
        archive1 = out / "2026-08-07"
        assert (archive1 / "report.md").read_text() == first_md

        # Run 3 (same day): run-2 outputs move into the _2 suffix dir, and the
        # run-1 archive is never overwritten.
        second_md = (out / "report.md").read_text()
        reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=out,
            config=None,
        )
        archive2 = out / "2026-08-07_2"
        assert (archive2 / "report.md").read_text() == second_md
        assert (archive1 / "report.md").read_text() == first_md
        # A fresh report still sits at the root after every run.
        assert (out / "report.md").exists()
        assert (out / "fast_sessions.csv").exists()

    def test_same_day_archive_dir_collides_with_existing_dir(self, tmp_path):
        # A dated dir already exists (e.g. a manual archive) -> suffix, not overwrite.
        out = tmp_path / "out"
        out.mkdir()
        (out / "2026-08-07").mkdir()
        (out / "2026-08-07" / "report.md").write_text("MANUAL")
        (out / "report.md").write_text("CURRENT")

        archived = reporting._archive_existing_outputs(out, now=self.NOW)

        assert archived.name == "2026-08-07_2"
        assert (archived / "report.md").read_text() == "CURRENT"
        assert (out / "2026-08-07" / "report.md").read_text() == "MANUAL"

    def test_pristine_output_dir_left_untouched(self, tmp_path, monkeypatch):
        monkeypatch.setattr(reporting, "datetime", self._FrozenClock)
        self._run(tmp_path, out_name="fresh", monkeypatch=monkeypatch)
        out = tmp_path / "fresh"
        # No archive dirs are created when there was nothing to archive.
        dirs = [p for p in out.iterdir() if p.is_dir()]
        assert dirs == []

    def test_unrelated_files_in_output_dir_are_not_moved(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "report.md").write_text("CURRENT")
        (out / "cron.log").write_text("2026-08-07 05:00 ok\n")

        archived = reporting._archive_existing_outputs(out, now=self.NOW)

        assert (archived / "report.md").read_text() == "CURRENT"
        # Non-artifact files stay put (cron log lives at the root).
        assert (out / "cron.log").read_text() == "2026-08-07 05:00 ok\n"
        assert not (archived / "cron.log").exists()
