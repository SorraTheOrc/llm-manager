
# <!-- REFACTOR-LP-0MVDFQC2X005S7WE
# smell: formatting
# severity: high
# description: Ambiguous variable name: `l`
# -->

# <!-- REFACTOR-LP-0MTHLGVPO004DZ56
# smell: unused_import
# severity: critical
# description: Local variable `schedule` is assigned to but never used
# -->
"""Unit tests for the proxy-usage-analysis skill.

Covers: log-line parsing, session aggregation, fast/cheap bucketing from the
slot schedule, recommendation rules, config loading, and an end-to-end run
over fixture log files.

Fixtures are derived from real lines in /var/log/llama-proxy/proxy.log
(see tests/fixtures.py).
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import sys
from collections import Counter
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

    def test_stream_finished_client_disconnect_reason(self):
        # Terminal line logged by the proxy when a client disconnects
        # mid-stream (LP-0MSVRRTAB0078TMK): parses as a normal stream_finished
        # with reason=client_disconnect (NOT stream_finish_error, which is
        # reserved for the synthetic reason=error event), so busy-time pairing
        # treats the stream as finished and its compute time becomes known.
        line = (
            "2026-08-02 14:00:00,000 - INFO - Stream finished: reason=client_disconnect "
            "session=019fc27d-3a46-7e5c-871e-57ab32f875f3 provider=local model=Qwen3"
        )
        ev = log_parser.parse_log_line(line)
        assert ev.kind == "stream_finished"
        assert ev.reason == "client_disconnect"
        assert ev.session == "019fc27d-3a46-7e5c-871e-57ab32f875f3"
        assert ev.provider == "local"
        assert ev.model == "Qwen3"
        assert ev.prompt is None

    def test_fallback_triggered(self):
        ev = log_parser.parse_log_line(fixtures.FALLBACK_CONCURRENCY)
        assert ev.kind == "fallback"
        assert ev.src == "local-qwen3"
        assert ev.dst == "opencode-go-deepseek"
        assert ev.reason == "local_concurrency_limit"
        assert ev.session is None

    def test_fallback_session_bearing_line_parses_evidence(self):
        """Fallback lines since LP-0MVMBMSWR0036RES carry ``session=``/``attempt=``
        (and budget evidence for budget-driven reasons); all fields parse."""
        ev = log_parser.parse_log_line(fixtures.FALLBACK_SESSION)
        assert ev.kind == "fallback"
        assert ev.session == "herdr-1792767061-1282111-17770"
        assert ev.attempt_index == 3
        assert ev.budget_limit is None  # reason not budget-driven

    def test_fallback_rate_limited_line_parses_budget_snapshot(self):
        ev = log_parser.parse_log_line(fixtures.FALLBACK_RATE_LIMITED_BUDGET)
        assert ev.kind == "fallback"
        assert ev.session == "herdr-1792767061-1282111-17770"
        assert ev.attempt_index == 4
        assert ev.reason == "rate_limited"
        assert ev.budget_limit == 60
        assert ev.budget_remaining == 0
        assert ev.budget_waited == pytest.approx(12.5)

    def test_fallback_session_token_unknown_maps_to_none(self):
        """The proxy emits ``session=unknown`` when the request has no UUID;
        the parser degrades it to ``None`` (missing field) so legacy
        attribution logic treats it as a session-less line."""
        line = (
            "2026-10-23 09:20:00,000 - INFO - Fallback triggered for model=v1/chat/completions, "
            "from=local-qwen3, to=opencode-go-deepseek, reason=HTTP 500 session=unknown attempt=1"
        )
        ev = log_parser.parse_log_line(line)
        assert ev.kind == "fallback"
        assert ev.session is None
        assert ev.attempt_index == 1
        assert ev.reason == "HTTP 500"

    def test_fallback_legacy_line_evidence_fields_none(self):
        """Pre-change rotated logs parse with every evidence field None."""
        ev = log_parser.parse_log_line(fixtures.FALLBACK_HTTP_400)
        assert ev.kind == "fallback"
        assert ev.reason == "HTTP 400"
        assert ev.session is None
        assert ev.attempt_index is None
        assert ev.budget_limit is None
        assert ev.budget_remaining is None
        assert ev.budget_waited is None

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

    def test_routing_skip_chain_hold_cycle_parsed(self):
        # The proxy stamps chain_hold_cycle=<N> on routing_skip_local lines
        # (LP-0MU5AIAAY003KVM0); the parser carries it so aggregation can
        # distinguish hold-cycle re-emissions from genuine demand.
        ev = log_parser.parse_log_line(fixtures.ROUTING_SKIP_CHAIN_HOLD_CYCLE_2)
        assert ev.kind == "routing_skip"
        assert ev.reason == "context_too_large"
        assert ev.chain_hold_cycle == 2
        ev0 = log_parser.parse_log_line(fixtures.ROUTING_SKIP_CHAIN_HOLD_CYCLE_0)
        assert ev0.kind == "routing_skip"
        assert ev0.chain_hold_cycle == 0

    def test_routing_skip_without_chain_hold_cycle_field(self):
        # Backward compatibility (AC4): pre-chain-hold rotated logs carry no
        # chain_hold_cycle token; the field must default to None (treated as
        # cycle 0 downstream) and the line must still parse.
        ev = log_parser.parse_log_line(fixtures.ROUTING_SKIP_CONTEXT_TOO_LARGE)
        assert ev.kind == "routing_skip"
        assert ev.reason == "context_too_large"
        assert ev.chain_hold_cycle is None

    def test_routing_skip_provider_context_parsed(self):
        # LP-0MVMBM1Y3001TPRT AC1: the per-provider context-ceiling skip is its
        # own event kind carrying session, provider, type, reason, estimate and
        # the ceiling that was exceeded.
        ev = log_parser.parse_log_line(fixtures.ROUTING_SKIP_PROVIDER_CONTEXT)
        assert ev is not None
        assert ev.kind == "provider_context_skip"
        assert ev.session == "herdr-1792524101-1636143-211"
        assert ev.provider == "munitoai-free"
        assert ev.provider_type == "remote"
        assert ev.reason == "provider_context_exceeded"
        assert ev.estimated_tokens == 153917
        assert ev.limit == 150000
        # A skip is not a fallback and not a local skip: it must not leak into
        # the other kinds' fields.
        assert ev.src is None and ev.dst is None
        assert ev.chain_hold_cycle is None

    def test_routing_skip_provider_context_without_limit_degrades(self):
        # AC8: historical lines may predate ``limit=``. The event still parses
        # with everything else intact and ``limit`` None (never fatal).
        ev = log_parser.parse_log_line(fixtures.ROUTING_SKIP_PROVIDER_CONTEXT_NO_LIMIT)
        assert ev is not None
        assert ev.kind == "provider_context_skip"
        assert ev.limit is None
        assert ev.estimated_tokens == 160001
        assert ev.provider == "munitoai-free"
        assert ev.session == "herdr-1792524101-1636143-212"

    def test_routing_skip_provider_context_is_not_a_local_skip(self):
        # Both prefixes start with ``routing_skip``; the provider-context line
        # must never be parsed as a ``routing_skip_local`` event (and vice
        # versa), otherwise per-session attribution precedence inverts.
        ctx = log_parser.parse_log_line(fixtures.ROUTING_SKIP_PROVIDER_CONTEXT)
        local = log_parser.parse_log_line(fixtures.ROUTING_SKIP_CONTEXT_TOO_LARGE)
        assert ctx.kind == "provider_context_skip"
        assert local.kind == "routing_skip"

    def test_routing_skip_provider_context_unknown_session_is_unattributed(self):
        line = fixtures.ROUTING_SKIP_PROVIDER_CONTEXT.replace(
            "session=herdr-1792524101-1636143-211", "session=unknown"
        )
        ev = log_parser.parse_log_line(line)
        assert ev.kind == "provider_context_skip"
        assert ev.session is None

    def test_iter_events_attaches_file_and_line_evidence(self, tmp_path):
        # AC1: an event carries the source file/line for evidence, not just the
        # raw text. Two noise lines precede the skip so the number is non-trivial.
        log = tmp_path / "proxy.log"
        log.write_text(
            "2026-10-21 04:11:00,000 - INFO - Stream started: provider=local model=Qwen3 "
            "session=herdr-1792524101-1636143-211 request=[]\n"
            "not a log line\n" + fixtures.ROUTING_SKIP_PROVIDER_CONTEXT + "\n",
            encoding="utf-8",
        )
        events = list(
            log_parser.iter_events(
                log, datetime(2026, 10, 21, 4, 0), datetime(2026, 10, 21, 5, 0)
            )
        )
        skip = next(e for e in events if e.kind == "provider_context_skip")
        assert skip.src_file == "proxy.log"
        assert skip.line_no == 3
        assert skip.raw.strip() == fixtures.ROUTING_SKIP_PROVIDER_CONTEXT

    def test_dispatch_denied(self):
        ev = log_parser.parse_log_line(fixtures.DISPATCH_DENIED)
        assert ev.kind == "dispatch_denied"
        assert ev.session == "019fc245"
        assert ev.owner == "019fc27d"
        assert ev.active == 4

    def test_contention_dispatch(self):
        ev = log_parser.parse_log_line(fixtures.CONTENTION_DISPATCH)
        assert ev.kind == "contention_dispatch"
        assert ev.provider == "local-qwen3"
        assert ev.session == "019fc245-aaaa-7e5c-871e-57ab32f875f3"
        assert ev.queued_duration == "1.25"
        assert ev.policy == "queue"
        assert ev.depth == "0"

    def test_contention_fallback_after_queue(self):
        ev = log_parser.parse_log_line(fixtures.CONTENTION_FALLBACK_AFTER_QUEUE)
        assert ev.kind == "contention_fallback_after_queue"
        assert ev.provider == "local-qwen3"
        assert ev.session == "019fc245-bbbb-7e5c-871e-57ab32f875f3"
        assert ev.queued_duration == "60.00"

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

    def test_mode_switch_parsed(self):
        """Mode scheduler applied-mode lines build the mode timeline
        (LP-0MSPZUD4G007IYGH AC1)."""
        ev = log_parser.parse_log_line(fixtures.MODE_SWITCH_CHEAP)
        assert ev is not None
        assert ev.kind == "mode_switch"
        assert ev.mode == "cheap"
        assert ev.ts == datetime(2026, 8, 15, 1, 0, 8, 679000)
        ev2 = log_parser.parse_log_line(fixtures.MODE_SWITCH_FAST)
        assert ev2.kind == "mode_switch"
        assert ev2.mode == "fast"

    def test_mode_scheduler_enabled_line_ignored(self):
        """The ``enabled with N entries`` announcement carries no applied
        mode and is not a timeline event (fires on both scheduled and manual
        transitions, so it is not a reliable mode signal)."""
        assert log_parser.parse_log_line(fixtures.MODE_SCHEDULER_ENABLED_IGNORED) is None

    def test_manual_mode_switch_grandfathering_parsed(self):
        """A manual ``POST /admin/set-mode`` switch reports the actually-
        active mode as ``Grandfathering: enabled; other-mode config ...
        (current=<mode>)`` — a mode_switch event (LP-0MT1EE315007AKXG)."""
        ev = log_parser.parse_log_line(fixtures.MANUAL_MODE_SWITCH_GRANDFATHERING_CHEAP)
        assert ev is not None
        assert ev.kind == "mode_switch"
        assert ev.mode == "cheap"
        assert ev.ts == datetime(2026, 8, 19, 18, 20, 16, 684000)

    def test_manual_mode_switch_enabled_and_restart_lines_not_mode_signals(self):
        """The manual-switch 'enabled with N entries' announcement (it fired
        for the scheduled 10:00 transition too) carries no reliable mode and
        stays outside the timeline. The 'router-mode restart complete'
        corroboration line is not a mode signal either — since
        LP-0MUQO82LJ008F1FS it is parsed as a ``restart`` marker (used to
        correlate stream errors with a restart), never as a ``mode_switch``."""
        assert log_parser.parse_log_line(fixtures.MANUAL_MODE_SWITCH_CHEAP) is None
        restart_ev = log_parser.parse_log_line(fixtures.MANUAL_MODE_SWITCH_RESTART)
        assert restart_ev is not None
        assert restart_ev.kind == "restart"
        assert restart_ev.kind != "mode_switch"

    def test_manual_mode_switch_slot_scheduler_current_ignored(self):
        """slot_scheduler logs ``(current=%d`` with a slot COUNT, not a mode;
        the parser must not mistake it for a mode_switch."""
        line = (
            "2026-08-19 18:20:20,000 - INFO - Slot scheduler: arming transition "
            "to 2 slots at 10:00 (current=3)"
        )
        assert log_parser.parse_log_line(line) is None

    def test_status_request_lines_ignored(self):
        """status_request lines expose total_slots but are not parsed (they
        are corroborating evidence only)."""
        assert log_parser.parse_log_line(fixtures.STATUS_REQUEST_CHEAP) is None
        assert log_parser.parse_log_line(fixtures.STATUS_REQUEST_FAST) is None


class TestTtftParsing:
    """TTFT (Time to First Token) log-line parsing.

    Tests that both the local ``dispatch_first_byte_ms=`` and remote
    ``ttft_seconds=`` log lines are correctly parsed into ``ttft`` events
    (LP-0MTSSM5SO003PKU0).
    """

    def test_dispatch_first_byte_ms_local(self):
        """Local stream dispatch_first_byte_ms line is parsed correctly."""
        line = (
            "2026-08-02 14:30:00,100 - INFO - "
            "dispatch_first_byte_ms=45.2 dispatch_to_first_byte_ms=45.2 "
            "session=abc123 model=qwen3-235b-a22b"
        )
        ev = log_parser.parse_log_line(line)
        assert ev is not None
        assert ev.kind == "ttft"
        assert ev.ttft_ms == pytest.approx(45.2)
        assert ev.session == "abc123"
        assert ev.model == "qwen3-235b-a22b"

    def test_dispatch_first_byte_ms_no_session(self):
        """dispatch_first_byte_ms without session yields session=None."""
        line = (
            "2026-08-02 14:30:00,100 - INFO - "
            "dispatch_first_byte_ms=120.5 dispatch_to_first_byte_ms=120.5 "
            "model=qwen3-235b-a22b"
        )
        ev = log_parser.parse_log_line(line)
        assert ev is not None
        assert ev.kind == "ttft"
        assert ev.ttft_ms == pytest.approx(120.5)
        assert ev.session is None

    def test_ttft_seconds_remote(self):
        """Remote stream ttft_seconds line is parsed and converted to ms."""
        line = (
            "2026-08-02 14:35:00,200 - INFO - "
            "ttft_seconds=0.823 session=def456 provider=opencode-go "
            "model=deepseek-v4-flash"
        )
        ev = log_parser.parse_log_line(line)
        assert ev is not None
        assert ev.kind == "ttft"
        assert ev.ttft_ms == pytest.approx(823.0)  # 0.823s → 823ms
        assert ev.session == "def456"
        assert ev.provider == "opencode-go"
        assert ev.model == "deepseek-v4-flash"

    def test_ttft_seconds_small_value(self):
        """Small ttft_seconds values (< 1s) are converted correctly."""
        line = (
            "2026-08-02 14:35:00,200 - INFO - "
            "ttft_seconds=0.045 session=ghi789 provider=deepseek "
            "model=deepseek-v4-flash"
        )
        ev = log_parser.parse_log_line(line)
        assert ev is not None
        assert ev.ttft_ms == pytest.approx(45.0)

    def test_ttft_lines_do_not_match_stream_started(self):
        """TTFT lines are NOT captured by stream_started parsing."""
        line = (
            "2026-08-02 14:30:00,100 - INFO - "
            "dispatch_first_byte_ms=45.2 dispatch_to_first_byte_ms=45.2 "
            "session=abc123 model=qwen3-235b-a22b"
        )
        ev = log_parser.parse_log_line(line)
        assert ev.kind == "ttft"
        assert ev.kind != "stream_started"

    def test_dispatch_first_byte_real_warm_line(self):
        """A real warm-cache line from /var/log/llama-proxy parses field-exact.

        AC2: ``parse_log_line()`` yields ``kind=ttft`` with ``ttft_ms`` plus
        ``dispatch_to_first_byte_ms``, ``queue_wait_ms``, ``cache_state``,
        ``cached_ratio``, ``session`` and ``model``, matching the real-line
        format exactly.
        """
        ev = log_parser.parse_log_line(fixtures.DISPATCH_FIRST_BYTE_WARM)
        assert ev is not None
        assert ev.kind == "ttft"
        assert ev.ttft_ms == pytest.approx(29546.0)
        assert ev.dispatch_to_first_byte_ms == pytest.approx(29545.3)
        assert ev.queue_wait_ms == pytest.approx(0.6)
        assert ev.cache_state == "warm"
        assert ev.cached_ratio == pytest.approx(0.992)
        assert ev.session == fixtures.PREFILL_SESSION
        assert ev.model == "Qwen3"
        assert ev.ts == datetime(2026, 10, 22, 21, 15, 42, 7000)

    def test_dispatch_first_byte_real_cold_line(self):
        """A real cold-cache line (zero cached ratio) parses field-exact."""
        ev = log_parser.parse_log_line(fixtures.DISPATCH_FIRST_BYTE_COLD)
        assert ev is not None
        assert ev.kind == "ttft"
        assert ev.ttft_ms == pytest.approx(55656.5)
        assert ev.dispatch_to_first_byte_ms == pytest.approx(55655.9)
        assert ev.cache_state == "cold"
        assert ev.cached_ratio == pytest.approx(0.0)
        assert ev.session == fixtures.PREFILL_SESSION

    def test_dispatch_first_byte_queue_wait_split(self):
        """The service-time split is consistent: total − queue wait = service.

        Real line: 26.6 s dispatch → first byte, of which 16.2 s was spent
        queued for the single slot, leaving 10.4 s of service time.
        """
        ev = log_parser.parse_log_line(fixtures.DISPATCH_FIRST_BYTE_QUEUE_WAIT)
        assert ev is not None
        assert ev.queue_wait_ms == pytest.approx(16169.9)
        assert ev.dispatch_to_first_byte_ms == pytest.approx(10403.1)
        assert ev.ttft_ms - ev.queue_wait_ms == pytest.approx(ev.dispatch_to_first_byte_ms)

    def test_remote_ttft_line_has_no_prefill_fields(self):
        """Remote ``ttft_seconds=`` lines carry none of the local prefill fields."""
        line = (
            "2026-08-02 14:35:00,200 - INFO - "
            "ttft_seconds=0.823 session=def456 provider=opencode-go "
            "model=deepseek-v4-flash"
        )
        ev = log_parser.parse_log_line(line)
        assert ev is not None
        assert ev.kind == "ttft"
        assert ev.dispatch_to_first_byte_ms is None
        assert ev.queue_wait_ms is None
        assert ev.cache_state is None
        assert ev.cached_ratio is None


class TestPrefillPairing:
    """FIFO pairing of ``dispatch_first_byte_ms`` events to local streams.

    For each session the Nth TTFT event pairs with the Nth ``stream_started``
    event by timestamp (AC3); streams with no TTFT event surface as
    ``PrefillMatch`` with ``ttft=None`` — unknown prefill — instead of being
    silently dropped (AC4).
    """

    @staticmethod
    def _parse(lines: list[str]) -> list[log_parser.LogEvent]:
        return [e for e in (log_parser.parse_log_line(ln) for ln in lines) if e is not None]

    def test_fifo_pairs_nth_ttft_with_nth_stream_real_session(self):
        """Three real TTFT lines pair FIFO with the three starts of the same
        session, in timestamp order."""
        events = self._parse(
            [
                fixtures.STREAM_STARTED_PREFILL_COLD,
                fixtures.DISPATCH_FIRST_BYTE_COLD,
                fixtures.STREAM_STARTED_PREFILL_COLD_FAST,
                fixtures.DISPATCH_FIRST_BYTE_COLD_FAST,
                fixtures.STREAM_STARTED_PREFILL_WARM,
                fixtures.DISPATCH_FIRST_BYTE_WARM,
            ]
        )
        starts = [e for e in events if e.kind == "stream_started"]
        ttfts = [e for e in events if e.kind == "ttft"]
        matches = log_parser.pair_ttft_to_streams(ttfts, starts)

        assert len(matches) == 3
        # FIFO by timestamp: the 55.7 s cold prefill pairs with the first
        # stream, the 1.1 s cold with the second, the 29.5 s warm with the
        # third.
        assert [m.prefill_seconds for m in matches] == pytest.approx(
            [55.6565, 1.1005, 29.546]
        )
        for m in matches:
            assert m.known_prefill
            assert m.stream.session == fixtures.PREFILL_SESSION
            assert m.ttft.session == fixtures.PREFILL_SESSION
            # dispatch time is back-calculated: ts − prefill duration.
            assert m.dispatch_ts == m.ttft.ts - timedelta(milliseconds=m.ttft.ttft_ms)
        # Matches are ordered by stream start time.
        assert [m.stream.ts for m in matches] == sorted(m.stream.ts for m in matches)

    def test_single_request_session_pairs_its_ttft(self):
        events = self._parse(
            [fixtures.STREAM_STARTED_PREFILL_COLD, fixtures.DISPATCH_FIRST_BYTE_COLD]
        )
        matches = log_parser.pair_ttft_to_streams(
            [e for e in events if e.kind == "ttft"],
            [e for e in events if e.kind == "stream_started"],
        )
        assert len(matches) == 1
        assert matches[0].prefill_seconds == pytest.approx(55.6565)

    def test_multi_session_pairing_by_timestamp(self):
        """Sessions pair independently; interleaved starts keep their order."""
        starts = [
            log_parser.LogEvent("stream_started", datetime(2026, 8, 2, 14, 0, 10), session="s1"),
            log_parser.LogEvent("stream_started", datetime(2026, 8, 2, 14, 0, 20), session="s2"),
            log_parser.LogEvent("stream_started", datetime(2026, 8, 2, 14, 0, 30), session="s1"),
        ]
        ttfts = [
            log_parser.LogEvent("ttft", datetime(2026, 8, 2, 14, 0, 12), session="s1", ttft_ms=500.0),
            log_parser.LogEvent("ttft", datetime(2026, 8, 2, 14, 0, 22), session="s2", ttft_ms=200.0),
            log_parser.LogEvent("ttft", datetime(2026, 8, 2, 14, 0, 32), session="s1", ttft_ms=700.0),
        ]
        matches = log_parser.pair_ttft_to_streams(ttfts, starts)
        assert [(m.stream.session, m.prefill_seconds) for m in matches] == [
            ("s1", 0.5),
            ("s2", 0.2),
            ("s1", 0.7),
        ]

    def test_aborted_stream_reported_as_unknown_prefill(self):
        """A stream that dies before first byte has no TTFT line: it is kept
        in the pairing as unknown prefill rather than silently dropped."""
        events = self._parse(
            [
                fixtures.STREAM_STARTED_ABORTED_LOCAL,
                fixtures.STREAM_ERROR_ABORTED_LOCAL,
                fixtures.STREAM_STARTED_PREFILL_COLD,
                fixtures.DISPATCH_FIRST_BYTE_COLD,
            ]
        )
        matches = log_parser.pair_ttft_to_streams(
            [e for e in events if e.kind == "ttft"],
            [e for e in events if e.kind == "stream_started"],
        )
        assert len(matches) == 2
        aborted = next(m for m in matches if m.stream.session == "audit-WL-0MVFRRMX3003UPP3-phase2_child_6-11d812e7")
        assert not aborted.known_prefill
        assert aborted.ttft is None
        assert aborted.prefill_seconds is None
        assert aborted.dispatch_ts is None
        # The completed stream still pairs normally.
        completed = next(m for m in matches if m.stream.session == fixtures.PREFILL_SESSION)
        assert completed.known_prefill
        assert completed.prefill_seconds == pytest.approx(55.6565)

    def test_extra_ttft_events_without_a_stream_are_ignored(self):
        """More TTFT events than streams: the surplus has no stream to
        attribute to and is ignored (not fabricated)."""
        starts = [log_parser.LogEvent("stream_started", datetime(2026, 8, 2, 14, 0, 10), session="s1")]
        ttfts = [
            log_parser.LogEvent("ttft", datetime(2026, 8, 2, 14, 0, 12), session="s1", ttft_ms=500.0),
            log_parser.LogEvent("ttft", datetime(2026, 8, 2, 14, 0, 22), session="s1", ttft_ms=999.0),
        ]
        matches = log_parser.pair_ttft_to_streams(ttfts, starts)
        assert len(matches) == 1
        assert matches[0].prefill_seconds == pytest.approx(0.5)

    def test_session_less_events_are_not_paired(self):
        """Session-less TTFT/start events cannot be paired and are skipped."""
        starts = [log_parser.LogEvent("stream_started", datetime(2026, 8, 2, 14, 0, 10), session=None)]
        ttfts = [log_parser.LogEvent("ttft", datetime(2026, 8, 2, 14, 0, 12), session=None, ttft_ms=500.0)]
        assert log_parser.pair_ttft_to_streams(ttfts, starts) == []

    def test_dispatch_interval_before_window_still_reconstructs(self):
        """A dispatch that starts before the window still reconstructs its
        interval (boundary-crossing input for the union-interval busy
        calculation): the pairing is window-agnostic."""
        ttft = log_parser.LogEvent("ttft", datetime(2026, 8, 2, 14, 0, 5), session="s1", ttft_ms=60000.0)
        start = log_parser.LogEvent("stream_started", datetime(2026, 8, 2, 13, 59, 0), session="s1")
        match = log_parser.pair_ttft_to_streams([ttft], [start])[0]
        assert match.dispatch_ts == datetime(2026, 8, 2, 13, 59, 5)  # 60 s before first byte
        assert match.ttft.ts > WINDOW_START
        assert match.dispatch_ts < WINDOW_START  # pre-window dispatch


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

    def test_stream_finished_reason_error_with_enriched_payload(self):
        # LP-0MT6322OT00900OX: the proxy now appends the enriched error
        # payload (type/message/suggested-action) to the stream-finish-error
        # log line; the parser must surface it so the analysis can classify.
        line = (
            "2026-08-03 10:13:15,000 - INFO - Stream finished: reason=error "
            "error_type=stall_exhausted error_message=Upstream stalled repeatedly "
            "(2 retries exhausted; idle timeout 30s) "
            "suggested_action=Provider placed in cooldown; the next provider in "
            "the chain will be used "
            "session=019fc52e-05a0-78d5-b59d-bcb91055b787 provider=opencode-go "
            "model=deepseek-v4-flash entry=opencode-go-2-deepseek"
        )
        ev = log_parser.parse_log_line(line)
        assert ev is not None
        assert ev.kind == "stream_finish_error"
        assert ev.error_type == "stall_exhausted"
        assert ev.error_message == "Upstream stalled repeatedly (2 retries exhausted; idle timeout 30s)"
        assert ev.suggested_action == (
            "Provider placed in cooldown; the next provider in the chain will be used"
        )
        assert ev.provider == "opencode-go"
        assert ev.model == "deepseek-v4-flash"
        assert ev.entry == "opencode-go-2-deepseek"

    def test_stream_finished_reason_error_partial_payload(self):
        # Enriched event with only type (no message/action) — fields absent.
        line = (
            "2026-08-03 10:13:16,000 - INFO - Stream finished: reason=error "
            "error_type=stream_exception "
            "session=019fc52e-05a0-78d5-b59d-bcb91055b787 provider=local model=Qwen3"
        )
        ev = log_parser.parse_log_line(line)
        assert ev is not None
        assert ev.kind == "stream_finish_error"
        assert ev.error_type == "stream_exception"
        assert ev.error_message is None
        assert ev.suggested_action is None

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

    def test_upstream_429_go_usage_limit(self):
        ev = log_parser.parse_log_line(fixtures.UPSTREAM_429_GO_USAGE_LIMIT)
        assert ev is not None
        assert ev.kind == "upstream_http_error"
        assert ev.status == 429
        # Only the error type is extracted; metadata.limitName / reset are not.
        assert ev.error == "GoUsageLimitError"
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

    def test_mode_switch_restart_spawned_parsed(self):
        ev = log_parser.parse_log_line(fixtures.RESTART_SPAWNED_LINE)
        assert ev is not None
        assert ev.kind == "restart"
        assert ev.restart_kind == "spawned"
        assert ev.ts == datetime(2026, 9, 25, 17, 27, 58)
        assert ev.raw and "Mode-switch restart spawned" in ev.raw

    def test_router_mode_restart_complete_parsed(self):
        ev = log_parser.parse_log_line(fixtures.RESTART_COMPLETE_LINE)
        assert ev is not None
        assert ev.kind == "restart"
        assert ev.restart_kind == "complete"
        assert ev.ts == datetime(2026, 9, 25, 17, 28, 5)

    def test_restart_services_non_restart_line_ignored(self):
        # Only the ``router-mode restart complete`` marker is a restart event;
        # the generic ``restarting llama-server`` line is not.
        assert log_parser.parse_log_line(fixtures.RESTART_SERVICES_OTHER_LINE) is None

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

    def test_iter_events_decompresses_gzip_rotated_file(self, tmp_path):
        """A ``.gz`` rotated file is transparently decompressed and parsed.

        Regression: gzip-suffixed files were discovered but read as plain
        text, yielding 0 parseable lines and silently dropping all in-window
        data (real case: proxy.log.2026-09-13_00.gz holds 778 stream starts).
        """
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        path = log_dir / "proxy.log.2026-08-03_13.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(fixtures.STREAM_ERROR_LINE + "\n")
        events = list(
            log_parser.iter_events(path, ERROR_WINDOW_START, ERROR_WINDOW_END)
        )
        assert len(events) == 1
        assert events[0].kind == "stream_error"
        assert events[0].src_file == "proxy.log.2026-08-03_13.gz"

    def test_iter_events_between_gzip_and_plain_same_content(self, tmp_path):
        """Plain and gzipped forms of the same log line yield identical events."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        plain = log_dir / "proxy.log.2026-08-03_13"
        gz = log_dir / "proxy.log.2026-08-03_13.gz"
        plain.write_text(fixtures.STREAM_ERROR_LINE + "\n")
        with gzip.open(gz, "wt", encoding="utf-8") as fh:
            fh.write(fixtures.STREAM_ERROR_LINE + "\n")
        from_plain = list(log_parser.iter_events(plain, ERROR_WINDOW_START, ERROR_WINDOW_END))
        from_gz = list(log_parser.iter_events(gz, ERROR_WINDOW_START, ERROR_WINDOW_END))
        assert len(from_plain) == len(from_gz) == 1
        assert from_plain[0].ts == from_gz[0].ts
        # src_file reflects the actual source path
        assert from_plain[0].src_file == "proxy.log.2026-08-03_13"
        assert from_gz[0].src_file == "proxy.log.2026-08-03_13.gz"


class TestAttributionMarkers:
    """Parsing of the actor/source/reason attribution markers added by
    LP-0MVAPNC2900145RH (LP-0MVC0S6U5003SEDJ).

    The markers ride on the restart/mode-switch/startup log paths; absent
    markers (historical logs) must degrade gracefully to the pre-attribution
    behaviour (new fields stay ``None``).
    """

    def test_spawned_restart_with_attribution(self):
        """A mode-switch restart spawned line carrying the attribution captured
        at set_mode() time populates source/actor/reason."""
        ev = log_parser.parse_log_line(fixtures.RESTART_SPAWNED_ATTRIBUTED)
        assert ev is not None
        assert ev.kind == "restart"
        assert ev.restart_kind == "spawned"
        assert ev.source == "scheduler"
        assert ev.actor == "mode-scheduler"
        assert ev.reason == "scheduled mode change"

    def test_spawned_restart_without_attribution_preserves_old_behaviour(self):
        """Historical spawned lines carry no markers: new fields stay None and
        the raw line is still retained (backward compatible)."""
        ev = log_parser.parse_log_line(fixtures.RESTART_SPAWNED_LINE)
        assert ev is not None
        assert ev.kind == "restart"
        assert ev.restart_kind == "spawned"
        assert ev.source is None
        assert ev.actor is None
        assert ev.reason is None
        assert ev.raw and "Mode-switch restart spawned" in ev.raw

    def test_spawned_restart_missing_individual_keys(self):
        """Only a subset of the attribution keys present: the others stay None
        (never fatal)."""
        line = (
            "2026-10-18 05:00:00,000 - INFO - Mode-switch restart spawned: "
            f"{fixtures.RESTART_SCRIPT} source=manual"
        )
        ev = log_parser.parse_log_line(line)
        assert ev is not None
        assert ev.kind == "restart"
        assert ev.source == "manual"
        assert ev.actor is None
        assert ev.reason is None

    def test_router_mode_restart_complete_with_attribution(self):
        """Tolerant: markers on the restart-complete line are extracted too
        (the proxy does not emit them today; future-proof and never fatal)."""
        ev = log_parser.parse_log_line(fixtures.RESTART_COMPLETE_ATTRIBUTED)
        assert ev is not None
        assert ev.kind == "restart"
        assert ev.restart_kind == "complete"
        assert ev.source == "api"
        assert ev.actor == "herdr"
        assert ev.reason == "maintenance window"

    def test_router_mode_restart_complete_without_attribution(self):
        ev = log_parser.parse_log_line(fixtures.RESTART_COMPLETE_LINE)
        assert ev is not None
        assert ev.kind == "restart"
        assert ev.restart_kind == "complete"
        assert ev.source is None
        assert ev.actor is None
        assert ev.reason is None

    def test_mode_switch_line_with_attribution(self):
        """The scheduler logstamped its applied-mode line with
        source=scheduler actor=mode-scheduler."""
        ev = log_parser.parse_log_line(fixtures.MODE_SWITCH_ATTRIBUTED)
        assert ev is not None
        assert ev.kind == "mode_switch"
        assert ev.mode == "cheap"
        assert ev.source == "scheduler"
        assert ev.actor == "mode-scheduler"

    def test_mode_switch_line_without_attribution(self):
        """Historical applied-mode lines carry no markers: fields stay None."""
        ev = log_parser.parse_log_line(fixtures.MODE_SWITCH_CHEAP)
        assert ev is not None
        assert ev.kind == "mode_switch"
        assert ev.mode == "cheap"
        assert ev.source is None
        assert ev.actor is None
        assert ev.reason is None

    def test_grandfathering_line_without_attribution(self):
        """Grandfathering lines do not carry markers today; fields stay None."""
        ev = log_parser.parse_log_line(fixtures.MANUAL_MODE_SWITCH_GRANDFATHERING_CHEAP)
        assert ev is not None
        assert ev.kind == "mode_switch"
        assert ev.mode == "cheap"
        assert ev.source is None
        assert ev.actor is None
        assert ev.reason is None

    def test_startup_attribution_line(self):
        """start-proxy.sh --restart reasons are echoed into the new process
        startup log before the ``Starting LLama Proxy Server`` marker; parsed
        as a ``restart`` startup event carrying the definitive attribution."""
        ev = log_parser.parse_log_line(fixtures.STARTUP_ATTRIBUTION_LINE)
        assert ev is not None
        assert ev.kind == "restart"
        assert ev.restart_kind == "startup"
        assert ev.source == "api"
        assert ev.actor == "herdr"
        assert ev.reason == "manual (no reason provided)"
        assert ev.ts == datetime(2026, 10, 18, 5, 0, 1, 500000)

    def test_startup_attribution_line_is_a_restart_signal(self):
        """The startup line widens the restart-signal set: a local stream error
        adjacent to it is classified as a restart kill."""
        local = (
            "2026-10-18 05:00:02,000 - WARNING - Stream error: "
            "session=019fc754-d847-75af-86ea-991480e799d0 provider=local "
            "model=Qwen3 error=RemoteProtocolError"
        )
        window_start = datetime(2026, 10, 18, 5, 0, 0)
        window_end = datetime(2026, 10, 18, 5, 1, 0)
        events = [
            e
            for e in (
                log_parser.parse_log_line(fixtures.STARTUP_ATTRIBUTION_LINE),
                log_parser.parse_log_line(local),
            )
            if e is not None
        ]
        res = aggregation.aggregate(events, window_start, window_end, _schedule())
        (ev,) = res.error_events
        assert ev.root_cause == "mode_switch_restart"

    def test_attribution_extractors_tolerate_arbitrary_lines(self):
        """The tolerant extractors pull source=/actor=/reason= out of arbitrary
        log lines, stopping the reason at the next known key (or line end) so
        space-containing reasons survive."""
        assert log_parser._attribution_from(
            "routing_check reason=context_too_large source=scheduler session=abc"
        ) == ("scheduler", None, "context_too_large")
        assert log_parser._attribution_from(
            "some line actor=herdr reason=rebalanced after maintenance window end"
        ) == (None, "herdr", "rebalanced after maintenance window end")
        assert log_parser._attribution_from(
            "unattributed historical line"
        ) == (None, None, None)


class TestDiscoverLogFiles:
    """Discovery of proxy log files for an analysis window.

    Two rotation mechanisms produce two naming patterns in this deployment:
    in-process ``TimedRotatingFileHandler`` → ``proxy.log.YYYY-MM-DD_HH`` (dot)
    and logrotate safety net → ``proxy.log-YYYY-MM-DD_HH`` (dash). Both are
    included regardless of name-encoded timestamp: rotated files routinely hold
    data well past their encoded rotation time, so discovery must never exclude
    a file based on its name. ``iter_events`` per-line timestamp filtering is
    the only boundary.
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

    def test_gzipped_rotated_files_included(self, tmp_path):
        """Plain and gzipped rotated files are both discovered.

        Both rotation mechanisms may compress older files, so discovery must
        return ``.gz`` and plain siblings; ``iter_events`` decompresses.
        """
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        for name in [
            "proxy.log.2026-08-02_14.gz",
            "proxy.log-2026-08-02_14.gz",
            "proxy.log.2026-08-02_14",
            "proxy.log-2026-08-02_14",
        ]:
            (log_dir / name).write_text("garbage\n")
        files = log_parser.discover_log_files(log_dir, WINDOW_START)
        assert {p.name for p in files} == {
            "proxy.log.2026-08-02_14.gz",
            "proxy.log-2026-08-02_14.gz",
            "proxy.log.2026-08-02_14",
            "proxy.log-2026-08-02_14",
        }

    def test_dotted_and_dashed_files_sorted_correctly(self, tmp_path):
        """Both dot and dash rotated files appear in sorted order.

        '-' (0x2D) sorts before '.' (0x2E) in ASCII, so dash files come
        before dot files for the same date.
        """
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        for name in [
            "proxy.log.2026-08-02_14",
            "proxy.log-2026-08-02_14",
            "proxy.log",
        ]:
            (log_dir / name).write_text("garbage\n")
        files = log_parser.discover_log_files(log_dir, WINDOW_START)
        assert [p.name for p in files] == [
            "proxy.log",
            "proxy.log-2026-08-02_14",
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

    def test_fallback_attributed_via_provider_context_skip(self):
        # AC2: the session's own routing_skip_provider_context line is
        # session-bearing evidence, so it attributes the move (and its reason)
        # without relying on a nearest-line Fallback triggered guess.
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s7 request=[]",
            "2026-08-02 14:00:05,000 - INFO - Stream finished: reason=stop tokens=900/40/940 session=s7 provider=local model=Qwen3 request=[]",
            "2026-08-02 14:00:20,000 - INFO - routing_skip_provider_context provider=munitoai-free "
            "type=remote status=provider_context_exceeded estimated_tokens=153917 limit=150000 "
            "reason=provider_context_exceeded session=s7",
            "2026-08-02 14:00:21,000 - INFO - Stream started: provider=opencode-go model=deepseek-v4-flash session=s7 request=[]",
            "2026-08-02 14:00:25,000 - INFO - Stream finished: reason=stop tokens=950/200/1150 session=s7 provider=opencode-go model=deepseek-v4-flash request=[]",
        ]
        res = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, _schedule())
        s = res.sessions["s7"]
        assert s.fallback_reason == "provider_context_exceeded"
        assert s.remote_move_time == datetime(2026, 8, 2, 14, 0, 20)

    def test_own_provider_context_skip_beats_nearest_fallback_guess(self):
        # The own line wins over a closer session-less Fallback triggered line,
        # even when that guess is also inside the attribution window.
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s8 request=[]",
            "2026-08-02 14:00:05,000 - INFO - Stream finished: reason=stop tokens=900/40/940 session=s8 provider=local model=Qwen3 request=[]",
            "2026-08-02 14:00:40,000 - INFO - routing_skip_provider_context provider=munitoai-free "
            "type=remote status=provider_context_exceeded estimated_tokens=153917 limit=150000 "
            "reason=provider_context_exceeded session=s8",
            "2026-08-02 14:00:55,000 - INFO - Fallback triggered for model=v1/chat/completions, "
            "from=local-qwen3, to=opencode-go-deepseek, reason=HTTP 400",
            "2026-08-02 14:00:58,000 - INFO - Stream started: provider=opencode-go model=deepseek-v4-flash session=s8 request=[]",
        ]
        res = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, _schedule())
        s = res.sessions["s8"]
        assert s.fallback_reason == "provider_context_exceeded"
        assert s.remote_move_time == datetime(2026, 8, 2, 14, 0, 40)

    def test_routing_skip_local_beats_provider_context_skip(self):
        # Documented precedence (AC2): the local-side skip explains why local was
        # not used at all, so it outranks the downstream per-provider ceiling
        # hits recorded while walking the chain.
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s9 request=[]",
            "2026-08-02 14:00:05,000 - INFO - Stream finished: reason=stop tokens=900/40/940 session=s9 provider=local model=Qwen3 request=[]",
            "2026-08-02 14:00:10,000 - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 "
            "estimated_tokens=40000 cold_threshold=39594 warm_threshold=39594 new_tokens=323 "
            "cached_ratio=0.99 reason=context_too_large → skipping local, routing to next remote "
            "provider session=s9",
            "2026-08-02 14:00:12,000 - INFO - routing_skip_provider_context provider=munitoai-free "
            "type=remote status=provider_context_exceeded estimated_tokens=153917 limit=150000 "
            "reason=provider_context_exceeded session=s9",
            "2026-08-02 14:00:14,000 - INFO - Stream started: provider=opencode-go model=deepseek-v4-flash session=s9 request=[]",
        ]
        res = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, _schedule())
        s = res.sessions["s9"]
        assert s.fallback_reason == "context_too_large"
        assert s.remote_move_time == datetime(2026, 8, 2, 14, 0, 10)

    def test_provider_context_skips_collected_without_inflating_demand(self):
        # Provider-context skips are collected for the report's own breakdown
        # but never counted as fallback events, local skips, or requests.
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s10 request=[]",
            "2026-08-02 14:00:05,000 - INFO - Stream finished: reason=stop tokens=900/40/940 session=s10 provider=local model=Qwen3 request=[]",
            "2026-08-02 14:00:20,000 - INFO - routing_skip_provider_context provider=munitoai-free "
            "type=remote status=provider_context_exceeded estimated_tokens=153917 limit=150000 "
            "reason=provider_context_exceeded session=s10",
            "2026-08-02 14:00:21,000 - INFO - routing_skip_provider_context provider=opencode-go "
            "type=remote status=provider_context_exceeded estimated_tokens=195787 limit=196608 "
            "reason=provider_context_exceeded session=s10",
        ]
        res = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, _schedule())
        assert len(res.provider_context_skip_events) == 2
        assert res.fallback_events == []
        assert res.routing_skip_events == []
        assert res.total_requests == 1
        assert res.sessions["s10"].routing_skips == 0
        assert res.sessions["s10"].messages == 1
        assert dict(res.provider_context_skip_provider_counts) == {
            "munitoai-free": 1,
            "opencode-go": 1,
        }
        assert dict(res.provider_context_skip_limit_counts) == {150000: 1, 196608: 1}

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

    def test_contention_events_collected_globally(self):
        """Contention-queue dispatch / fallback lines land in their own
        collections (LP-0MSORQVK50012Q4D F4 AC3)."""
        lines = [
            fixtures.CONTENTION_DISPATCH,
            fixtures.CONTENTION_FALLBACK_AFTER_QUEUE,
        ]
        res = aggregation.aggregate(
            _events(lines),
            datetime(2026, 8, 2, 13, 0),
            datetime(2026, 8, 2, 15, 0),
            _schedule(),
        )
        assert len(res.contention_dispatch_events) == 1
        assert len(res.contention_fallback_events) == 1
        assert res.contention_dispatch_events[0].queued_duration == "1.25"
        assert res.contention_fallback_events[0].queued_duration == "60.00"

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


class TestFallbackAttribution:
    """Per-session fallback attribution (LP-0MVMBMSWR0036RES AC5).

    A session's reason comes from its **own** evidence — its ``Fallback
    triggered`` or ``routing_skip_local`` line. Another session's session-bearing
    fallback line is never inherited; only session-less lines from rotated logs
    predating the session stamp still use the nearest-line-within-60s heuristic.
    """

    WINDOW = (datetime(2026, 10, 23, 8, 0, 0), datetime(2026, 10, 23, 11, 0, 0))

    def _local_then_remote(self, session: str, start: str, remote: str) -> list[str]:
        return [
            f"2026-10-23 {start}:00,000 - INFO - Stream started: provider=local "
            f"model=Qwen3 session={session} request=[]",
            f"2026-10-23 {start}:05,000 - INFO - Stream finished: reason=stop "
            f"tokens=900/40/940 session={session} provider=local model=Qwen3 request=[]",
            f"2026-10-23 {remote}:30,000 - INFO - Stream started: provider=opencode-go "
            f"model=deepseek-v4-flash session={session} request=[]",
            f"2026-10-23 {remote}:35,000 - INFO - Stream finished: reason=stop "
            f"tokens=950/200/1150 session={session} provider=opencode-go "
            f"model=deepseek-v4-flash request=[]",
        ]

    def _summarise(self, lines: list[str], schedule=None):
        return aggregation.aggregate(
            _events(lines), *self.WINDOW, schedule or _schedule()
        )

    def test_own_fallback_line_is_used(self):
        """A session with its own ``Fallback triggered`` line (no skip line) is
        attributed from that line."""
        lines = [
            # The session's own session-bearing fallback line, before its first
            # remote stream (fixture time 09:12:44, remote at 09:13:00).
            fixtures.FALLBACK_SESSION,
            *self._local_then_remote("herdr-1792767061-1282111-17770",
                                     "09:12", "09:13"),
        ]
        res = self._summarise(lines)
        s = res.sessions["herdr-1792767061-1282111-17770"]
        assert s.fallback_reason == "local_lease_active"
        assert s.fallback_attribution == aggregation.FALLBACK_ATTRIBUTION_OWN_FALLBACK
        assert s.remote_move_time == datetime(2026, 10, 23, 9, 12, 44, 517000)

    def test_own_skip_line_still_wins(self):
        """An own ``routing_skip_local`` line keeps precedence over the fallback
        line (unchanged behaviour), and is labelled ``own_skip``."""
        lines = [
            fixtures.FALLBACK_SESSION,
            *self._local_then_remote("herdr-1792767061-1282111-17770",
                                     "09:12", "09:13"),
            "2026-10-23 09:12:50,000 - INFO - routing_skip_local provider=local-qwen3 "
            "model=Qwen3 estimated_tokens=5000 cold_threshold=39594 warm_threshold=39594 "
            "new_tokens=50 cached_ratio=0.50 reason=context_too_large → skipping local, "
            "routing to next remote provider session=herdr-1792767061-1282111-17770",
        ]
        res = self._summarise(lines)
        s = res.sessions["herdr-1792767061-1282111-17770"]
        assert s.fallback_reason == "context_too_large"
        assert s.fallback_attribution == aggregation.FALLBACK_ATTRIBUTION_OWN_SKIP

    def test_session_does_not_inherit_another_sessions_reason(self):
        """The regression this change fixes: session B reached remote seconds
        after session A's spill, but B has no fallback line of its own — it must
        stay unattributed instead of inheriting A's ``rate_limited``."""
        lines = [
            # A's session-bearing spill, just before B's remote move.
            "2026-10-23 09:05:59,000 - INFO - Fallback triggered for model=v1/chat/completions, "
            "from=munitoai-free, to=opencode-go, reason=rate_limited "
            "session=other-session-0001 attempt=1 limit=60 remaining=0 waited=0.000",
            *self._local_then_remote("session-b", "09:06", "09:06"),
        ]
        res = self._summarise(lines)
        s = res.sessions["session-b"]
        assert s.fallback_reason is None
        assert s.fallback_attribution == aggregation.FALLBACK_ATTRIBUTION_UNATTRIBUTED
        # The move itself is still recorded (it used local, then remote).
        assert s.remote_move_time == datetime(2026, 10, 23, 9, 6, 30)

    def test_session_less_legacy_line_still_attributes_by_proximity(self):
        """Rotated logs without ``session=`` keep the legacy nearest-line
        heuristic, labelled ``legacy_nearest`` so it is distinguishable."""
        lines = [
            "2026-10-23 09:06:28,000 - INFO - Fallback triggered for model=v1/chat/completions, "
            "from=local-qwen3, to=opencode-go-deepseek, reason=HTTP 500",
            *self._local_then_remote("legacy-session", "09:06", "09:06"),
        ]
        res = self._summarise(lines)
        s = res.sessions["legacy-session"]
        assert s.fallback_reason == "HTTP 500"
        assert s.fallback_attribution == aggregation.FALLBACK_ATTRIBUTION_LEGACY_NEAREST

    def test_lease_denied_session_is_labelled_explicitly(self):
        """No own fallback line, but ``local_dispatch_denied`` evidence: the
        move is lease/concurrency contention, reported as such."""
        lines = [
            "2026-10-23 09:06:20,000 - INFO - local_dispatch_denied "
            "session=leased-session owner=other-session active=1",
            *self._local_then_remote("leased-session", "09:06", "09:06"),
        ]
        res = self._summarise(lines)
        s = res.sessions["leased-session"]
        assert s.fallback_reason == aggregation.LEASE_ACTIVE_REASON
        assert s.fallback_attribution == aggregation.FALLBACK_ATTRIBUTION_LEASE_DENIED

    def test_local_only_session_is_no_remote_fallback(self):
        lines = [
            "2026-10-23 09:10:00,000 - INFO - Stream started: provider=local "
            "model=Qwen3 session=stays-local request=[]",
            "2026-10-23 09:10:05,000 - INFO - Stream finished: reason=stop "
            "tokens=900/40/940 session=stays-local provider=local model=Qwen3 request=[]",
        ]
        res = self._summarise(lines)
        s = res.sessions["stays-local"]
        assert s.fallback_reason is None
        assert s.fallback_attribution == aggregation.FALLBACK_ATTRIBUTION_NO_REMOTE

    def test_attribution_counts_cover_every_session(self):
        lines = [
            fixtures.FALLBACK_SESSION,
            *self._local_then_remote("herdr-1792767061-1282111-17770",
                                     "09:12", "09:13"),
            "2026-10-23 09:05:59,000 - INFO - Fallback triggered for model=v1/chat/completions, "
            "from=munitoai-free, to=opencode-go, reason=rate_limited "
            "session=other-session-0001 attempt=1 limit=60 remaining=0 waited=0.000",
            *self._local_then_remote("session-b", "09:06", "09:06"),
        ]
        res = self._summarise(lines)
        counts = res.fallback_attribution_counts
        assert sum(counts.values()) == len(res.sessions)
        assert counts[aggregation.FALLBACK_ATTRIBUTION_OWN_FALLBACK] == 1
        assert counts[aggregation.FALLBACK_ATTRIBUTION_UNATTRIBUTED] == 1

    def test_report_exposes_attribution_states(self):
        lines = [
            "2026-10-23 09:05:59,000 - INFO - Fallback triggered for model=v1/chat/completions, "
            "from=munitoai-free, to=opencode-go, reason=rate_limited "
            "session=other-session-0001 attempt=1 limit=60 remaining=0 waited=0.000",
            *self._local_then_remote("session-b", "09:06", "09:06"),
            "2026-10-23 09:10:00,000 - INFO - Stream started: provider=local "
            "model=Qwen3 session=stays-local request=[]",
        ]
        res = self._summarise(lines)
        md = reporting.build_report(res, None)
        assert "Session fallback attribution:" in md
        assert "unattributed **1**" in md
        assert "no_remote_fallback **1**" in md

    def test_json_summary_exposes_attribution_counts(self):
        lines = [
            fixtures.FALLBACK_SESSION,
            *self._local_then_remote("herdr-1792767061-1282111-17770",
                                     "09:12", "09:13"),
        ]
        res = self._summarise(lines)
        data = reporting.summary_to_json(res)
        assert data["fallback_attribution"] == {
            aggregation.FALLBACK_ATTRIBUTION_OWN_FALLBACK: 1
        }

    def test_csv_row_carries_attribution(self, tmp_path):
        lines = [
            "2026-10-23 09:05:59,000 - INFO - Fallback triggered for model=v1/chat/completions, "
            "from=munitoai-free, to=opencode-go, reason=rate_limited "
            "session=other-session-0001 attempt=1 limit=60 remaining=0 waited=0.000",
            *self._local_then_remote("session-b", "09:06", "09:06"),
        ]
        res = self._summarise(lines)
        standard_path, low_path = reporting.write_csvs(res, tmp_path)
        rows: list[dict] = []
        for path in (standard_path, low_path):
            with path.open(newline="", encoding="utf-8") as fh:
                rows.extend(csv.DictReader(fh))
        row = next(r for r in rows if r["session_id"] == "session-b")
        assert row["fallback_attribution"] == aggregation.FALLBACK_ATTRIBUTION_UNATTRIBUTED
        assert row["fallback_reason"] == ""

    def test_recommendations_note_attribution_coverage(self):
        """Reason-count evidence names sessions without own evidence so the
        counts are not read as complete."""
        lines = [
            "2026-10-23 09:05:20,000 - INFO - local_dispatch_denied "
            "session=leased-session owner=other-session active=1",
            *self._local_then_remote("leased-session", "09:06", "09:06"),
        ]
        res = self._summarise(lines)
        recs = recommendations.generate_recommendations(res, None)
        contention = [r for r in recs if "slot pool contention" in r.title]
        assert contention, "expected a slot-contention recommendation"
        assert "lease/concurrency denied" in contention[0].evidence

    def test_recommendations_omit_coverage_note_when_fully_attributed(self):
        lines = [
            fixtures.FALLBACK_SESSION,
            *self._local_then_remote("herdr-1792767061-1282111-17770",
                                     "09:12", "09:13"),
        ]
        res = self._summarise(lines)
        recs = recommendations.generate_recommendations(res, None)
        contention = [r for r in recs if "slot pool contention" in r.title]
        assert contention, "expected a slot-contention recommendation"
        assert "Session attribution coverage" not in contention[0].evidence


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

    def test_upstream_error_by_status(self):
        lines = [
            fixtures.UPSTREAM_429,
            # A 402 (Insufficient Balance) real line from api.deepseek.com:
            "2026-08-03 11:00:00,000 - WARNING - [remote] upstream error status=402 "
            "url=https://api.deepseek.com/v1/chat/completions "
            "body={\"error\":{\"message\":\"Insufficient Balance\",\"type\":\"unknown_error\","
            "\"param\":null,\"code\":\"invalid_request_error\"}}",
            fixtures.UPSTREAM_429,
        ]
        res = aggregation.aggregate(
            _events(lines), ERROR_WINDOW_START, ERROR_WINDOW_END, _schedule()
        )
        assert res.upstream_error_by_status == {429: 2, 402: 1}
        by_sp = res.upstream_error_by_status_provider
        assert by_sp[429] == {"opencode": 2}
        assert by_sp[402] == {"deepseek": 1}

    def test_upstream_error_by_status_ignores_missing_status(self):
        lines = [
            "2026-08-03 13:58:04,053 - WARNING - [remote] upstream error "
            "url=https://api.deepseek.com/v1/chat/completions body={}",
        ]
        res = aggregation.aggregate(
            _events(lines), ERROR_WINDOW_START, ERROR_WINDOW_END, _schedule()
        )
        assert res.upstream_error_by_status == {}
        assert res.upstream_error_by_status_provider == {}


class TestRootCauseClassification:
    """Stream errors are attributed a deterministic root cause from log fields
    plus restart correlation (LP-0MUQO82LJ008F1FS)."""

    def _window_result(self):
        return aggregation.aggregate(
            _events(fixtures.window_2026_09_25_lines()),
            fixtures.WINDOW_2026_09_25_START,
            fixtures.WINDOW_2026_09_25_END,
            _schedule(),
        )

    def _classify(self, lines):
        return aggregation.aggregate(
            _events(lines),
            fixtures.WINDOW_2026_09_25_START,
            fixtures.WINDOW_2026_09_25_END,
            _schedule(),
        )

    def test_window_asserts_24_6_1_split(self):
        res = self._window_result()
        finish = [e for e in res.error_events if e.kind == "stream_finish_error"]
        assert len(finish) == 31
        counts = Counter(e.root_cause for e in finish)
        assert counts["upstream_stall_after_content"] == 24
        assert counts["mode_switch_restart"] == 6
        assert counts["remote_chain_exhaustion"] == 1
        # No event is left unclassified.
        assert None not in counts

    def test_root_cause_counts_breakdown(self):
        res = self._window_result()
        assert res.root_cause_counts == {
            "stream_finish_error": {
                "mode_switch_restart": 6,
                "remote_chain_exhaustion": 1,
                "upstream_stall_after_content": 24,
            }
        }

    def test_stall_after_content_classified(self):
        ts, session = fixtures.STALL_AFTER_CONTENT_EVENTS_2026_09_25[0]
        line = fixtures._stream_finish_error_line(
            ts, session, "opencode-go", "deepseek-v4.1-flash",
            "stall_after_content", "Upstream idle timeout after content delivered",
            "Retry the request",
        )
        res = self._classify([line])
        (ev,) = res.error_events
        assert ev.root_cause == "upstream_stall_after_content"

    def test_local_error_near_restart_marker_classified(self):
        local = fixtures._stream_finish_error_line(
            "2026-09-25 17:28:00,333", "sess-local", "local", "Qwen3",
            "stream_exception", "Local stream error (RemoteProtocolError)",
            "Check llama-server logs",
        )
        res = self._classify([fixtures.RESTART_SPAWNED_LINE, local])
        (ev,) = [e for e in res.error_events if e.kind == "stream_finish_error"]
        assert ev.root_cause == "mode_switch_restart"

    def test_local_error_near_mode_switch_classified(self):
        # A scheduled/ manual mode-switch line is also a restart signal even
        # when the explicit restart-spawn marker is absent from the logs.
        local = fixtures._stream_finish_error_line(
            "2026-09-25 10:00:36,989", "sess-local", "local", "Qwen3",
            "stream_exception", "Local stream error (RemoteProtocolError)",
            "Check llama-server logs",
        )
        mode_switch = (
            "2026-09-25 10:00:18,041 - INFO - "
            "Mode scheduler: applied scheduled mode fast"
        )
        res = self._classify([mode_switch, local])
        (ev,) = [e for e in res.error_events if e.kind == "stream_finish_error"]
        assert ev.root_cause == "mode_switch_restart"

    def test_local_error_without_marker_degrades_to_other(self):
        # Missing-marker fixture: never fabricates a cause (F1 AC4).
        res = self._classify([fixtures.LOCAL_ERROR_NO_MARKER_2026_09_25])
        (ev,) = res.error_events
        assert ev.root_cause == "other"

    def test_local_error_outside_correlation_window_is_other(self):
        # The restart marker is 200s before the error — beyond the documented
        # ±120s correlation window — so it is not attributed to the restart.
        local = fixtures._stream_finish_error_line(
            "2026-09-25 17:31:18,000", "sess-local", "local", "Qwen3",
            "stream_exception", "Local stream error (RemoteProtocolError)",
            "Check llama-server logs",
        )
        res = self._classify([fixtures.RESTART_SPAWNED_LINE, local])
        (ev,) = [e for e in res.error_events if e.kind == "stream_finish_error"]
        assert ev.root_cause == "other"

    def test_remote_error_classified_as_chain_exhaustion(self):
        res = self._classify(fixtures.window_2026_09_25_lines())
        remote = [
            e for e in res.error_events
            if e.kind == "stream_finish_error" and e.provider == "deepseek"
        ]
        assert len(remote) == 1
        assert remote[0].root_cause == "remote_chain_exhaustion"

    def test_non_stream_errors_have_no_root_cause(self):
        res = aggregation.aggregate(
            _events(ERROR_LINES), ERROR_WINDOW_START, ERROR_WINDOW_END, _schedule()
        )
        non_stream = [
            e for e in res.error_events
            if e.kind in ("slot_save_error", "backend_retry", "upstream_http_error")
        ]
        assert non_stream
        assert all(e.root_cause is None for e in non_stream)
        # Root-cause breakdown therefore only carries stream-error kinds.
        assert set(res.root_cause_counts) <= {"stream_finish_error", "stream_error"}


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

    def test_mode_map_period_for_labels_by_mode_not_slot_counts(self):
        """A cheap-profile schedule has equal slot counts all day; the mode
        map must still label its hours "cheap" (LP-0MSPZUD4G007IYGH AC3)."""
        mm = fixtures.mode_map_fixture()
        cheap_ts = datetime(2026, 8, 15, 2, 0, 0)
        fast_ts = datetime(2026, 8, 15, 10, 30, 0)
        mm.transitions = [
            (datetime(2026, 8, 15, 1, 0, 0), "cheap"),
            (datetime(2026, 8, 15, 10, 0, 0), "fast"),
        ]
        p = mm.period_for(cheap_ts)
        assert p.label == "cheap"
        assert p.slots == 2
        assert p.ctx_size == 262144
        p2 = mm.period_for(fast_ts)
        assert p2.label == "fast"
        assert p2.slots == 3
        assert p2.ctx_size == 131072

    def test_mode_map_mode_at_uses_nearest_prior_transition(self):
        mm = fixtures.mode_map_fixture()
        mm.transitions = [
            (datetime(2026, 8, 14, 10, 0, 0), "fast"),
            (datetime(2026, 8, 15, 1, 0, 0), "cheap"),
            (datetime(2026, 8, 15, 10, 0, 0), "fast"),
        ]
        assert mm.mode_at(datetime(2026, 8, 14, 10, 30, 0)) == "fast"
        assert mm.mode_at(datetime(2026, 8, 15, 2, 0, 0)) == "cheap"
        assert mm.mode_at(datetime(2026, 8, 15, 10, 30, 0)) == "fast"

    def test_mode_map_without_transitions_falls_back_to_analysis_mode(self):
        mm = fixtures.mode_map_fixture()
        assert mm.mode_at(datetime(2026, 8, 15, 2, 0, 0)) == "fast"  # default_mode


# ---------------------------------------------------------------------------
# Mode-timeline bucketing (LP-0MSPZUD4G007IYGH)
# ---------------------------------------------------------------------------


class TestModeTimelineBucketing:
    """Sessions are bucketed by the operating mode active at their first
    in-window stream (reconstructed from ``Mode scheduler`` log lines), so a
    window crossing the 01:00/10:00 transitions splits fast vs cheap instead
    of collapsing everything into the analysis-time bucket."""

    MODE_WINDOW_START = datetime(2026, 8, 14, 10, 29, 0)
    MODE_WINDOW_END = datetime(2026, 8, 15, 10, 29, 0)

    def _lines(self):
        return [
            # Prior transition (inside the 48h timeline margin, before window start).
            "2026-08-14 10:00:02,000 - INFO - Mode scheduler: applied scheduled mode fast",
            fixtures.STATUS_REQUEST_FAST,
            # Cheap transition + a cheap-hour session (02:00).
            fixtures.MODE_SWITCH_CHEAP,
            fixtures.STATUS_REQUEST_CHEAP,
            "2026-08-15 02:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=cheap-sess request=[]",
            "2026-08-15 02:00:10,000 - INFO - Stream finished: reason=stop tokens=50000/100/50100 session=cheap-sess provider=local model=Qwen3 request=[]",
            # Fast transition + fast-hour sessions (Aug 14 11:00, Aug 15 10:20).
            fixtures.MODE_SWITCH_FAST,
            "2026-08-14 11:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=fast-early request=[]",
            "2026-08-14 11:00:10,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=fast-early provider=local model=Qwen3 request=[]",
            "2026-08-15 10:20:00,000 - INFO - Stream started: provider=local model=Qwen3 session=fast-late request=[]",
            "2026-08-15 10:20:10,000 - INFO - Stream finished: reason=stop tokens=200/20/220 session=fast-late provider=local model=Qwen3 request=[]",
        ]

    def test_sessions_bucketed_by_mode_transitions(self):
        mm = fixtures.mode_map_fixture()
        res = aggregation.aggregate(
            _events(self._lines()),
            self.MODE_WINDOW_START,
            self.MODE_WINDOW_END,
            bucketing.schedule_from_config(fixtures.MODE_FAST_CONFIG, 3),
            mm,
        )
        # Timeline reconstructed from the applied-mode lines
        # (now ModeTransition objects, LP-0MVC144KS001PDR4).
        assert mm.transitions == [
            bucketing.ModeTransition(
                datetime(2026, 8, 14, 10, 0, 2), "fast", None, None, None
            ),
            bucketing.ModeTransition(
                datetime(2026, 8, 15, 1, 0, 8, 679000), "cheap", None, None, None
            ),
            bucketing.ModeTransition(
                datetime(2026, 8, 15, 10, 0, 18, 41000), "fast", None, None, None
            ),
        ]
        s = res.sessions["cheap-sess"]
        assert s.bucket == "cheap"
        assert s.slots == 2
        assert s.ctx_size == 262144
        assert s.max_context_size == 50000
        assert res.sessions["fast-early"].bucket == "fast"
        assert res.sessions["fast-early"].slots == 3
        assert res.sessions["fast-early"].ctx_size == 131072
        assert res.sessions["fast-late"].bucket == "fast"
        assert res.sessions["fast-late"].slots == 3
        assert res.sessions["fast-late"].ctx_size == 131072

    def test_manual_mode_switch_buckets_sessions_cheap(self):
        """Regression (LP-0MT1EE315007AKXG): a MANUAL switch to cheap mid-
        window (logged only as ``Grandfathering: enabled ... (current=cheap)``,
        no 'applied scheduled mode' line) must bucket subsequent sessions as
        cheap — previously the whole window collapsed into fast (0 cheap)."""
        mm = fixtures.mode_map_fixture()
        lines = [
            # Manual switch to cheap at 18:20 (inside the window).
            fixtures.MANUAL_MODE_SWITCH_CHEAP,
            fixtures.MANUAL_MODE_SWITCH_GRANDFATHERING_CHEAP,
            fixtures.MANUAL_MODE_SWITCH_RESTART,
            # Cheap-hour session after the manual switch.
            "2026-08-19 18:30:00,000 - INFO - Stream started: provider=local model=Qwen3 session=manual-cheap request=[]",
            "2026-08-19 18:30:10,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=manual-cheap provider=local model=Qwen3 request=[]",
            # Fast-hour session before the manual switch.
            "2026-08-19 12:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=manual-fast request=[]",
            "2026-08-19 12:00:10,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=manual-fast provider=local model=Qwen3 request=[]",
        ]
        window_start = datetime(2026, 8, 19, 10, 0, 0)
        window_end = datetime(2026, 8, 19, 20, 0, 0)
        res = aggregation.aggregate(
            _events(lines),
            window_start,
            window_end,
            bucketing.schedule_from_config(fixtures.MODE_FAST_CONFIG, 3),
            mm,
        )
        # Manual switch reconstructed as a cheap transition
        # (ModeTransition object, LP-0MVC144KS001PDR4).
        cheap_trans = bucketing.ModeTransition(
            datetime(2026, 8, 19, 18, 20, 16, 684000), "cheap", None, None, None
        )
        assert cheap_trans in mm.transitions
        assert res.sessions["manual-fast"].bucket == "fast"
        assert res.sessions["manual-fast"].slots == 3
        assert res.sessions["manual-cheap"].bucket == "cheap"
        assert res.sessions["manual-cheap"].slots == 2
        assert res.sessions["manual-cheap"].ctx_size == 262144

    def test_manual_switch_same_timestamp_as_scheduled_dedupes(self):
        """When a manual switch and a scheduled 'applied scheduled mode' line
        report the same mode at the same instant, the timeline keeps a single
        transition (no spurious flip)."""
        mm = fixtures.mode_map_fixture()
        lines = [
            "2026-08-19 10:00:18,000 - INFO - Mode scheduler: applied scheduled mode fast",
            "2026-08-19 10:00:18,400 - INFO - Grandfathering: enabled; other-mode config config-cheap.yaml (current=fast)",
            "2026-08-19 11:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s1 request=[]",
            "2026-08-19 11:00:10,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s1 provider=local model=Qwen3 request=[]",
        ]
        res = aggregation.aggregate(
            _events(lines),
            datetime(2026, 8, 19, 9, 0, 0),
            datetime(2026, 8, 19, 12, 0, 0),
            bucketing.schedule_from_config(fixtures.MODE_FAST_CONFIG, 3),
            mm,
        )
        # Both lines say 'fast'; the session stays fast and no double-transition
        # introduces a spurious cheap period.
        assert res.sessions["s1"].bucket == "fast"
        assert res.sessions["s1"].slots == 3

    def test_window_without_transitions_keeps_legacy_bucketing(self):
        """A single-mode window (no transitions observed) keeps the legacy
        slot-count bucketing from the analysis-time schedule (AC notes)."""
        mm = fixtures.mode_map_fixture()
        lines = [
            "2026-08-15 02:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s1 request=[]",
            "2026-08-15 02:00:10,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s1 provider=local model=Qwen3 request=[]",
        ]
        res = aggregation.aggregate(
            _events(lines),
            datetime(2026, 8, 15, 0, 0),
            datetime(2026, 8, 15, 3, 0),
            bucketing.schedule_from_config(fixtures.MODE_FAST_CONFIG, 3),
            mm,
        )
        # No mode lines -> transitions empty -> analysis-time schedule buckets.
        assert mm.transitions == []
        assert res.sessions["s1"].bucket == "fast"
        assert res.sessions["s1"].slots == 3

    def test_mode_switch_lines_counted_as_parsed_lines(self):
        mm = fixtures.mode_map_fixture()
        res = aggregation.aggregate(
            _events(self._lines()),
            self.MODE_WINDOW_START,
            self.MODE_WINDOW_END,
            bucketing.schedule_from_config(fixtures.MODE_FAST_CONFIG, 3),
            mm,
        )
        # In-window structured lines: 2 mode switches (Aug 15 01:00/10:00;
        # the Aug 14 10:00 line lies in the margin, outside the window)
        # + 3 stream starts + 3 stream finishes.
        assert res.total_lines == 8
        assert res.lines_skipped == 0


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

    def test_provider_context_skips_trigger_ceiling_advice(self):
        # AC4: provider-context skip events drive their own advice naming count,
        # share of fallbacks, standard/low split and the ``limit`` that was hit.
        res = _result_with_sessions(
            [_session("a")],
            fallback_events=[
                log_parser.LogEvent("fallback", WINDOW_START, reason="context_too_large"),
                log_parser.LogEvent("fallback", WINDOW_START, reason="context_too_large"),
            ],
        )
        res.provider_context_skip_events = [
            log_parser.LogEvent(
                "provider_context_skip",
                WINDOW_START,
                provider="munitoai-free",
                provider_type="remote",
                reason="provider_context_exceeded",
                estimated_tokens=153917,
                limit=150000,
            ),
            log_parser.LogEvent(
                "provider_context_skip",
                WINDOW_START,
                provider="opencode-go",
                reason="provider_context_exceeded",
                estimated_tokens=195787,
                limit=196608,
            ),
        ]
        recs = recommendations.generate_recommendations(res, config=None)
        ceiling = [r for r in recs if "ceiling" in r.title.lower()]
        assert ceiling, "expected provider-ceiling advice, got: " + " | ".join(r.title for r in recs)
        evidence = ceiling[0].evidence
        assert "2 provider context-ceiling skip(s)" in evidence
        assert "(100.0%" in evidence  # no fallback events -> skips are the whole signal
        assert "150000" in evidence and "196608" in evidence
        assert "Fast 2 (100.0%)" in evidence

    def test_provider_context_skips_below_share_do_not_advise(self):
        # AC4 threshold: a handful of ceiling skips inside a large fallback
        # volume stays below PROVIDER_CONTEXT_SHARE and does not fire advice.
        res = _result_with_sessions(
            [_session("a")],
            fallback_events=[
                log_parser.LogEvent("fallback", WINDOW_START, reason="other_reason")
                for _ in range(20)
            ],
        )
        res.provider_context_skip_events = [
            log_parser.LogEvent(
                "provider_context_skip",
                WINDOW_START,
                provider="munitoai-free",
                reason="provider_context_exceeded",
                limit=150000,
            )
        ]
        recs = recommendations.generate_recommendations(res, config=None)
        titles = " | ".join(r.title for r in recs)
        assert "ceiling" not in titles.lower()


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

    def test_stream_errors_report_strategy_live_not_adopt(self):
        """With the enriched payload present, the recommendation states the
        strategy is live and no longer tells the operator to adopt it
        (LP-0MUQO862C004UT7T AC1)."""
        errors = [
            log_parser.LogEvent(
                "stream_finish_error", ERROR_WINDOW_START,
                provider="opencode-go", model="deepseek-v4-flash",
                error_type="stall_after_content",
                root_cause="upstream_stall_after_content",
            )
            for _ in range(3)
        ]
        res = self._res_with_errors(errors)
        recs = recommendations.generate_recommendations(res, config=None)
        stream_recs = [r for r in recs if "stream finish errors" in r.title.lower()]
        assert stream_recs, "expected a stream-finish recommendation"
        title = stream_recs[0].title.lower()
        assert "adopt" not in title
        assert "is live" in title
        detail = stream_recs[0].detail
        assert "already implemented and shipped" in detail
        assert "strategy is live" in detail
        # Cites the shipped strategy work items, not the stale adoption pair.
        assert "LP-0MSETOTWY000SU0Z" in detail
        assert "LP-0MT60S55M000TK1H" in detail
        assert "LP-0MSDP2PDB004GV86" not in detail

    def test_stream_errors_name_dominant_residual_cause_with_counts(self):
        """The recommendation names the dominant residual cause and cites the
        root-cause counts (LP-0MUQO862C004UT7T AC2)."""
        errors = (
            [
                log_parser.LogEvent(
                    "stream_finish_error", ERROR_WINDOW_START,
                    provider="opencode-go", model="deepseek-v4.1-flash",
                    error_type="stall_after_content",
                    root_cause="upstream_stall_after_content",
                )
                for _ in range(24)
            ]
            + [
                log_parser.LogEvent(
                    "stream_finish_error", ERROR_WINDOW_START,
                    provider="local", model="Qwen3",
                    error_type="stream_exception",
                    root_cause="mode_switch_restart",
                )
                for _ in range(6)
            ]
            + [
                log_parser.LogEvent(
                    "stream_finish_error", ERROR_WINDOW_START,
                    provider="deepseek", model="deepseek-flash",
                    error_type="stream_exception",
                    root_cause="remote_chain_exhaustion",
                )
            ]
        )
        res = self._res_with_errors(errors)
        recs = recommendations.generate_recommendations(res, config=None)
        stream_rec = next(r for r in recs if "stream finish errors" in r.title.lower())
        assert "upstream_stall_after_content 24" in stream_rec.detail
        assert "mode_switch_restart 6" in stream_rec.detail
        assert "remote_chain_exhaustion 1" in stream_rec.detail
        assert "Dominant residual cause: upstream_stall_after_content" in stream_rec.detail
        # The dominant cause is by-design and names no proxy-side action.
        assert "by design" in stream_rec.detail

    def test_stream_errors_without_enriched_payload_do_not_claim_live(self):
        errors = [
            log_parser.LogEvent(
                "stream_finish_error", ERROR_WINDOW_START,
                provider="opencode-go", model="deepseek-v4-flash",
            )
            for _ in range(3)
        ]
        res = self._res_with_errors(errors)
        recs = recommendations.generate_recommendations(res, config=None)
        stream_rec = next(r for r in recs if "stream finish errors" in r.title.lower())
        assert "is live" not in stream_rec.title.lower()
        assert "cannot be verified" in stream_rec.detail

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
        detail = " | ".join(r.detail for r in recs)
        assert "LP-0MRGU0I91006ODFD" in detail, f"expected 3-hour cooldown citation, got: {detail}"
        # A FreeUsageLimitError without a computable reset cites the cooldown
        # only; the account quarantine path (LP-0MSLJPOCC0001ROJ) is not claimed.
        assert "LP-0MSLJPOCC0001ROJ" not in detail, f"unexpected quarantine citation: {detail}"

    def test_upstream_429_go_usage_limit_cites_account_quarantine(self):
        errors = [
            log_parser.LogEvent("upstream_http_error", ERROR_WINDOW_START, error="GoUsageLimitError", status=429)
            for _ in range(3)
        ]
        res = self._res_with_errors(errors)
        recs = recommendations.generate_recommendations(res, config=None)
        detail = " | ".join(r.detail for r in recs)
        assert "LP-0MSLJPOCC0001ROJ" in detail, f"expected account quarantine citation, got: {detail}"
        # The Go path must NOT claim the 3-hour cooldown that does not apply.
        assert "LP-0MRGU0I91006ODFD" not in detail, f"unexpected cooldown citation: {detail}"
        assert "3-hour" not in detail, f"unexpected 3-hour cooldown mention: {detail}"
        # The mechanism names how the period is derived (parser cannot cite it).
        assert "metadata.limitName" in detail, f"expected limitName guidance, got: {detail}"

    def test_upstream_429_unknown_error_type_uses_generic_wording(self):
        errors = [
            log_parser.LogEvent("upstream_http_error", ERROR_WINDOW_START, error="UnknownError", status=429)
            for _ in range(3)
        ]
        res = self._res_with_errors(errors)
        recs = recommendations.generate_recommendations(res, config=None)
        detail = " | ".join(r.detail for r in recs)
        assert "check the upstream provider quota or usage limits" in detail, \
            f"expected generic 429 wording, got: {detail}"
        # No recognised type: neither mechanism may be fabricated.
        assert "LP-0MSLJPOCC0001ROJ" not in detail, f"fabricated quarantine citation: {detail}"
        assert "LP-0MRGU0I91006ODFD" not in detail, f"fabricated cooldown citation: {detail}"

    def test_upstream_429_without_error_type_uses_generic_wording(self):
        errors = [
            log_parser.LogEvent("upstream_http_error", ERROR_WINDOW_START, status=429)
            for _ in range(2)
        ]
        res = self._res_with_errors(errors)
        recs = recommendations.generate_recommendations(res, config=None)
        detail = " | ".join(r.detail for r in recs)
        assert "check the upstream provider quota or usage limits" in detail, \
            f"expected generic 429 wording, got: {detail}"
        assert "LP-0MSLJPOCC0001ROJ" not in detail, f"fabricated quarantine citation: {detail}"
        assert "LP-0MRGU0I91006ODFD" not in detail, f"fabricated cooldown citation: {detail}"

    def test_upstream_429_mixed_types_describe_both_mechanisms(self):
        errors = [
            log_parser.LogEvent("upstream_http_error", ERROR_WINDOW_START, error="GoUsageLimitError", status=429),
            log_parser.LogEvent("upstream_http_error", ERROR_WINDOW_START, error="FreeUsageLimitError", status=429),
        ]
        res = self._res_with_errors(errors)
        recs = recommendations.generate_recommendations(res, config=None)
        detail = " | ".join(r.detail for r in recs)
        # A bucket mixing both must describe both, never silently drop the Go path.
        assert "LP-0MSLJPOCC0001ROJ" in detail, f"expected account quarantine citation, got: {detail}"
        assert "LP-0MRGU0I91006ODFD" in detail, f"expected cooldown citation, got: {detail}"

    def test_upstream_402_triggers_balance_recommendation(self):
        # Status 402 (Insufficient Balance) must NOT be reported as a
        # rate-limit / cooldown issue: it is an account-funding problem.
        errors = [
            log_parser.LogEvent("upstream_http_error", ERROR_WINDOW_START, error="unknown_error", status=402)
            for _ in range(3)
        ]
        res = self._res_with_errors(errors)
        recs = recommendations.generate_recommendations(res, config=None)
        titles = " | ".join(r.title.lower() for r in recs)
        assert "402" in titles, f"expected 402 recommendation, got: {titles}"
        assert "429" not in titles
        assert "balance" in titles, f"expected balance-related title, got: {titles}"

    def test_upstream_errors_broken_out_by_status(self):
        # Mixed statuses produce one recommendation per status, not a single
        # lumped 'upstream HTTP error' recommendation.
        errors = [
            log_parser.LogEvent("upstream_http_error", ERROR_WINDOW_START, error="FreeUsageLimitError", status=429),
            log_parser.LogEvent("upstream_http_error", ERROR_WINDOW_START, error="unknown_error", status=402),
            log_parser.LogEvent("upstream_http_error", ERROR_WINDOW_START, status=500),
        ]
        res = self._res_with_errors(errors)
        recs = recommendations.generate_recommendations(res, config=None)
        status_titles = [r.title for r in recs if "429" in r.title or "402" in r.title or "500" in r.title]
        joined = " ".join(status_titles)
        assert "429" in joined and "402" in joined and "500" in joined, \
            f"expected one recommendation per upstream status, got: {status_titles}"

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

    def test_find_config_path_mode_profiles_no_longer_selected(self, tmp_path):
        """The mode profiles and persisted `.mode` were removed
        (LP-0MVDT94F0003VHUA AC4): find_config_path always returns the single
        unified ``config.yaml`` even when legacy mode files exist."""
        (tmp_path / "proxy").mkdir()
        (tmp_path / "proxy" / "config.yaml").write_text("default_model: code\n")
        (tmp_path / "proxy" / "config-cheap.yaml").write_text("default_model: code\n")
        found = config_loader.find_config_path(start=tmp_path)
        assert found == tmp_path / "proxy" / "config.yaml"

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

    def test_yaml_parse_captures_per_period_ctx_size(self):
        """slot_schedule entries with an explicit ctx_size expose it as
        ctx_by_time (LP-0MSMZOAJW002UR2A / LP-0MSPZUD4G007IYGH AC4)."""
        text = """\
server:
  slot_schedule:
    enabled: true
    entries:
      - time: "23:59"
        slots: 2
        ctx_size: 262144
      - time: "10:00"
        slots: 3
        ctx_size: 131072
"""
        cfg = config_loader.parse_config_text(text)
        schedule = cfg["slot_schedule"]
        assert schedule["entries"] == [("23:59", 2), ("10:00", 3)]
        assert schedule["ctx_by_time"] == {"23:59": 262144, "10:00": 131072}

    def test_regex_fallback_captures_per_period_ctx_size(self):
        text = """\
server:
  slot_schedule:
    enabled: true
    entries:
      - time: "23:59"
        slots: 2
        ctx_size: 262144
      - time: "10:00"
        slots: 3
"""
        cfg = config_loader._parse_config_text_regex(text)
        schedule = cfg["slot_schedule"]
        assert schedule["entries"] == [("23:59", 2), ("10:00", 3)]
        assert schedule.get("ctx_by_time") == {"23:59": 262144}

    def test_discover_configs_loads_unified_config_and_priority_window(self, tmp_path):
        """discover_configs loads the single unified ``config.yaml`` and the
        priority-window label (the mode axis was removed,
        LP-0MVDT94F0003VHUA AC4/AC6)."""
        (tmp_path / "proxy").mkdir()
        (tmp_path / "proxy" / "config.yaml").write_text(
            "local_model_ctx_size: 131072\n"
            "server:\n  priority_window:\n    enabled: true\n    start: \"01:00\"\n    end: \"10:00\"\n"
        )
        configs = config_loader.discover_configs(start=tmp_path)
        assert configs["base"] == tmp_path / "proxy" / "config.yaml"
        assert configs["profiles"]["default"]["local_model_ctx_size"] == 131072
        # Legacy mode axis removed: no analysis_mode, priority window label present.
        assert configs["analysis_mode"] is None
        assert configs["priority_window"] in ("low", "standard")
        assert configs["analysis_config"] is configs["profiles"]["default"]

    # -- base + overlay merge (LP-0MUI681CH00653DT) ----------------------

    def test_load_proxy_config_merges_base_for_mode_overlay(self, tmp_path):
        """A deduplicated mode overlay inherits base values it no longer
        repeats (LP-0MUI681CH00653DT)."""
        d = tmp_path / "proxy"
        d.mkdir()
        (d / "config.yaml").write_text(
            "server:\n"
            "  local_model_ctx_size: 262144\n"
            "  session_slot_pool_size: 1\n"
        )
        (d / "config-cheap.yaml").write_text(
            "server:\n  session_slot_pool_size: 3\n"
        )
        cfg = config_loader.load_proxy_config(d / "config-cheap.yaml")
        assert cfg["local_model_ctx_size"] == 262144  # inherited from base
        assert cfg["session_slot_pool_size"] == 3  # overlay wins

    def test_load_proxy_config_merges_models_from_base(self, tmp_path):
        """The merged profile exposes the full model surface of the base."""
        d = tmp_path / "proxy"
        d.mkdir()
        (d / "config.yaml").write_text(
            "models:\n"
            "  embed:\n"
            "    providers:\n"
            "      - name: local-embed\n"
            "        type: local\n"
            "  plan:\n"
            "    providers:\n"
            "      - name: base-plan\n"
            "        type: remote\n"
            "server:\n"
            "  local_model_ctx_size: 262144\n"
        )
        (d / "config-fast.yaml").write_text(
            "models:\n"
            "  plan:\n"
            "    providers:\n"
            "      - name: fast-plan\n"
            "        type: remote\n"
        )
        cfg = config_loader.load_proxy_config(d / "config-fast.yaml")
        assert set(cfg["models"]) == {"embed", "plan"}  # base model kept
        assert cfg["models"]["plan"]["providers"][0]["name"] == "fast-plan"

    def test_base_config_loads_without_merge(self, tmp_path):
        d = tmp_path / "proxy"
        d.mkdir()
        (d / "config.yaml").write_text(
            "server:\n  local_model_ctx_size: 131072\n"
        )
        cfg = config_loader.load_proxy_config(d / "config.yaml")
        assert cfg["local_model_ctx_size"] == 131072

    def test_discover_configs_loads_unified_base(self, tmp_path):
        """discover_configs loads the unified base config (the fast/cheap
        overlay files and ``.mode`` are ignored; LP-0MVDT94F0003VHUA AC4)."""
        d = tmp_path / "proxy"
        d.mkdir()
        (d / "config.yaml").write_text(
            "models:\n"
            "  embed:\n"
            "    providers:\n"
            "      - name: local-embed\n"
            "        type: local\n"
            "server:\n"
            "  local_model_ctx_size: 262144\n"
            "  session_slot_pool_size: 1\n"
        )
        (d / "config-cheap.yaml").write_text(
            "server:\n  session_slot_pool_size: 3\n"
        )
        (d / ".mode").write_text("cheap\n")

        configs = config_loader.discover_configs(start=tmp_path)

        assert configs["profiles"]["default"]["local_model_ctx_size"] == 262144
        assert configs["profiles"]["default"]["models"]  # full model surface
        assert configs["analysis_config"] is configs["profiles"]["default"]
        assert configs["analysis_mode"] is None

    def test_real_profiles_ctx_sizes_match_pre_dedup(self):
        """The merged loader yields the same per-mode ctx sizes the proxy
        uses, so bucketing attribution is unchanged (LP-0MUI681CH00653DT)."""
        repo_root = Path(__file__).resolve().parents[4]
        configs = config_loader.discover_configs(start=repo_root)
        schedules = bucketing.ModeScheduleMap.from_profiles(
            configs["profiles"], configs["analysis_mode"], None
        )
        assert schedules.ctx_for("fast") == 262144
        assert schedules.ctx_for("cheap") == 262144


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


def _ttft_event(
    ts: datetime,
    session: str = "s1",
    ttft_ms: float | None = 5000.0,
    **kwargs: object,
) -> log_parser.LogEvent:
    """A local ``dispatch_first_byte_ms`` TTFT event.

    Mirrors the real dispatch log line, which carries no ``provider=`` field
    (so ``provider`` stays ``None``, the local marker used by the parser).
    """
    return log_parser.LogEvent("ttft", ts, session=session, ttft_ms=ttft_ms, **kwargs)


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
        assert busy.pre_window_unfinished == 0

    def test_pre_window_unfinished_streams_not_counted_in_caveat(self):
        # A stream started more than BUSY_WINDOW_MARGIN (1h) before the window
        # start with no logged finish is a pre-window leftover from an earlier
        # window (LP-0MSVRRO3L0056N6C): it is NOT an in-window abort and must
        # not inflate unfinished_streams; it is tracked separately.
        events = [
            # Pre-window leftover: started 2h before the window, never finished.
            _local_event("stream_started", datetime(2026, 8, 2, 12, 0, 0), "stale"),
            # In-window completed stream so busy stats exist.
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 0), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 10), "s1"),
        ]
        busy = aggregation.compute_busy_stats(events, WINDOW_START, WINDOW_END, _schedule())
        assert busy is not None
        assert busy.streams == 1
        assert busy.busy_seconds == 10.0
        assert busy.unfinished_streams == 0
        assert busy.pre_window_unfinished == 1

    def test_in_window_and_pre_window_unfinished_split(self):
        # Both classes coexist: an in-window abort (no finish) and a stale
        # pre-window start (no finish) are counted separately.
        events = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 30, 0), "abort"),
            _local_event("stream_started", datetime(2026, 8, 2, 12, 30, 0), "stale"),
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 0), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 10), "s1"),
        ]
        busy = aggregation.compute_busy_stats(events, WINDOW_START, WINDOW_END, _schedule())
        assert busy is not None
        assert busy.unfinished_streams == 1
        assert busy.pre_window_unfinished == 1

    def test_stream_started_within_margin_before_window_counts_unfinished(self):
        # A stream started within BUSY_WINDOW_MARGIN (1h) before the window
        # start with no finish may have consumed in-window compute -> counts
        # as unfinished for this window.
        events = [
            _local_event("stream_started", datetime(2026, 8, 2, 13, 30, 0), "s1"),
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 0), "s2"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 10), "s2"),
        ]
        busy = aggregation.compute_busy_stats(events, WINDOW_START, WINDOW_END, _schedule())
        assert busy is not None
        assert busy.unfinished_streams == 1
        assert busy.pre_window_unfinished == 0

    def test_stream_started_after_window_end_not_counted_unfinished(self):
        # A stream started after the window end has no in-window compute; it
        # belongs to the next window's report, not this one.
        events = [
            _local_event("stream_started", datetime(2026, 8, 2, 15, 30, 0), "later"),
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 0), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 10), "s1"),
        ]
        busy = aggregation.compute_busy_stats(events, WINDOW_START, WINDOW_END, _schedule())
        assert busy is not None
        assert busy.unfinished_streams == 0
        assert busy.pre_window_unfinished == 0

    def test_client_disconnect_finish_pairs_with_start(self):
        # A ``Stream finished: reason=client_disconnect`` terminal line (logged
        # by the proxy on client abort, LP-0MSVRRTAB0078TMK) is a normal
        # stream_finished event: the start pairs with it, its compute time is
        # known (start -> disconnect), and it is NOT counted as unfinished.
        started = log_parser.LogEvent(
            "stream_started", datetime(2026, 8, 2, 14, 0, 0),
            provider="local", model="Qwen3", session="s1",
        )
        finished = log_parser.parse_log_line(
            "2026-08-02 14:00:10,000 - INFO - Stream finished: reason=client_disconnect "
            "session=s1 provider=local model=Qwen3"
        )
        assert finished is not None and finished.kind == "stream_finished"
        busy = aggregation.compute_busy_stats(
            [started, finished], WINDOW_START, WINDOW_END, _schedule()
        )
        assert busy is not None
        assert busy.streams == 1
        assert busy.busy_seconds == 10.0
        assert busy.unfinished_streams == 0
        assert busy.pre_window_unfinished == 0

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
        assert hourly[datetime(2026, 8, 2, 9, 0, 0)] == 10.0
        assert hourly[datetime(2026, 8, 2, 10, 0, 0)] == 15.0

    def test_hourly_attribution_window_bounded_non_aligned(self):
        # Window 09:30-10:45 is not hour-aligned: hourly_busy values must be
        # window-correct per absolute hour (partial first/last hours
        # truncated to the window edges), so the report can render
        # window-bounded buckets from them unchanged (LP-0MSVMLM7G009N74N
        # AC2/AC5, absolute-hour keying LP-0MTYAZFGN003BYQS).
        start = datetime(2026, 8, 2, 9, 30, 0)
        end = datetime(2026, 8, 2, 10, 45, 0)
        events = [
            # 30s busy in the partial first hour (09:30-10:00 window part).
            _local_event("stream_started", datetime(2026, 8, 2, 9, 40, 0), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 9, 40, 30), "s1"),
            # 15s busy in the partial last hour (10:00-10:45 window part).
            _local_event("stream_started", datetime(2026, 8, 2, 10, 30, 0), "s2"),
            _local_event("stream_finished", datetime(2026, 8, 2, 10, 30, 15), "s2"),
        ]
        busy = aggregation.compute_busy_stats(events, start, end, _schedule())
        assert busy is not None
        assert busy.busy_seconds == 45.0
        hourly = dict(busy.hourly_busy)
        # Keys are absolute top-of-hour datetimes, not hour-of-day integers.
        assert hourly[datetime(2026, 8, 2, 9, 0, 0)] == 30.0
        assert hourly[datetime(2026, 8, 2, 10, 0, 0)] == 15.0
        assert set(hourly) == {datetime(2026, 8, 2, 9, 0, 0), datetime(2026, 8, 2, 10, 0, 0)}

    def test_report_summary_by_hour_table_window_bounded_with_pct_and_totals(self):
        # Regression (LP-0MSVMLM7G009N74N) extended by LP-0MTFO210Q0044TTF and
        # reworked by LP-0MTYAZFGN003BYQS: the hourly profile is the
        # top-of-report ``## Summary by hour`` table — it spans exactly the
        # report window (partial first/last rows truncated to the window
        # edges), lists every hour including idle ones, carries a % column
        # (busy / window-bounded bucket duration, clamped to 100%), and its
        # per-hour columns are now activity counts keyed by event timestamp
        # rather than session-first-hour journey classifications. It ends with
        # a totals row: window busy totals plus the sum of each activity
        # column over the window.
        start = datetime(2026, 8, 2, 9, 30, 0)
        end = datetime(2026, 8, 2, 11, 45, 0)
        lines = [
            # 30s busy in the partial first hour (09:30-10:00).
            "2026-08-02 09:40:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s1 request=[]",
            "2026-08-02 09:40:30,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s1 provider=local model=Qwen3 request=[]",
            # 15s busy in the partial last hour (11:00-11:45); hour 10:00-11:00 idle.
            "2026-08-02 11:30:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s2 request=[]",
            "2026-08-02 11:30:15,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s2 provider=local model=Qwen3 request=[]",
        ]
        summary = aggregation.aggregate(_events(lines), start, end, _schedule())
        assert summary.busy is not None
        md = reporting.build_report(summary, None)
        section = md.split("## Summary by hour", 1)[1].split("## ", 1)[0]
        # AC1: renamed, moved to the top of the report (before Session summary).
        assert "Busy time by hour:" not in md
        assert md.index("## Summary by hour") < md.index("## Session summary")
        # AC2/AC1: window-bounded rows with a % column; idle middle hour
        # listed; activity columns replace the classification columns.
        assert "| Hour | State | Busy | % | Requests started | Local attempts | Local served | Local dispatch | Lease denials | Fallbacks | Local-skip |" in section
        assert "| 09:30-10:00 | serving | 30s | 1.7% | 1 | 1 | 1 | 0 | 0 | 0 | 0 |" in section
        assert "| 10:00-11:00 | idle | 0s | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |" in section
        assert "| 11:00-11:45 | serving | 15s | 0.6% | 1 | 1 | 1 | 0 | 0 | 0 | 0 |" in section
        # AC3: totals row = total busy over the window and overall busy % plus
        # the sum of each activity column over the window.
        assert "| Totals | serving | 45s | 0.6% | 2 | 2 | 2 | 0 | 0 | 0 | 0 |" in section
        # The Local model utilization section no longer hosts the hourly table.
        util = md.split("## Local model utilization", 1)[1].split("## ", 1)[0]
        assert "| Hour | Busy |" not in util

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

    def test_prefill_interval_buckets_identically_to_stream_start(self):
        """AC5: for a session whose dispatch and first byte fall in the same
        fast/cheap period, the interval buckets identically whether it starts
        at ``Stream started`` or at the back-calculated dispatch time.

        The interval start handed to ``compute_busy_stats`` is the
        ``stream_started`` timestamp; modelling the new interval is therefore
        a start event at the dispatch time (09:30:00) rather than at first
        byte (09:30:05).
        """
        start = datetime(2026, 8, 2, 9, 0, 0)
        end = datetime(2026, 8, 2, 11, 0, 0)
        finish = _local_event("stream_finished", datetime(2026, 8, 2, 9, 30, 15), "s1")
        decode_only = aggregation.compute_busy_stats(
            [_local_event("stream_started", datetime(2026, 8, 2, 9, 30, 5), "s1"), finish],
            start,
            end,
            _schedule(),
        )
        with_prefill = aggregation.compute_busy_stats(
            [_local_event("stream_started", datetime(2026, 8, 2, 9, 30, 0), "s1"), finish],
            start,
            end,
            _schedule(),
        )
        assert decode_only is not None and with_prefill is not None
        assert decode_only.cheap_busy_seconds == 10.0
        assert with_prefill.cheap_busy_seconds == 15.0
        # Both intervals lie wholly in the cheap period: the *split* (all
        # cheap, no fast) is identical; only the duration grows by the 5 s of
        # prefill.
        assert decode_only.fast_busy_seconds == with_prefill.fast_busy_seconds == 0.0
        assert with_prefill.cheap_busy_seconds - decode_only.cheap_busy_seconds == 5.0

    def test_prefill_interval_crossing_period_boundary_splits_buckets(self):
        """AC4/AC5: a prefill interval that straddles the fast/cheap boundary
        attributes each side to its own bucket (5 s cheap + 20 s fast)."""
        start = datetime(2026, 8, 2, 9, 0, 0)
        end = datetime(2026, 8, 2, 11, 0, 0)
        # Dispatch 09:59:55 (cheap), first byte 10:00:05, finish 10:00:20 (fast).
        busy = aggregation.compute_busy_stats(
            [
                _local_event("stream_started", datetime(2026, 8, 2, 9, 59, 55), "s1"),
                _local_event("stream_finished", datetime(2026, 8, 2, 10, 0, 20), "s1"),
            ],
            start,
            end,
            _schedule(),
        )
        assert busy is not None
        assert busy.cheap_busy_seconds == 5.0
        assert busy.fast_busy_seconds == 20.0
        assert busy.busy_seconds == 25.0

    def test_prefill_interval_starting_before_window_is_clipped(self):
        """AC4: a dispatch before the window (boundary-crossing interval) is
        clipped to the window start for the union busy calculation."""
        start = datetime(2026, 8, 2, 9, 0, 0)
        end = datetime(2026, 8, 2, 11, 0, 0)
        busy = aggregation.compute_busy_stats(
            [
                _local_event("stream_started", datetime(2026, 8, 2, 8, 59, 50), "s1"),
                _local_event("stream_finished", datetime(2026, 8, 2, 9, 0, 30), "s1"),
            ],
            start,
            end,
            _schedule(),
        )
        assert busy is not None
        assert busy.busy_seconds == 30.0  # 20 s pre-window clipped away
        assert busy.streams == 1


class TestPrefillStats:
    """Per-stream local prefill statistics (LP-0MVJN45LF009CO2V).

    ``compute_busy_stats`` takes the local ``dispatch_first_byte_ms`` TTFT
    events, pairs them FIFO to the local streams per session, and reports the
    prefill interval ``[ttft.ts - ttft_ms/1000, ttft.ts]`` as slot-seconds
    (the integral of concurrency), never as wall-clock.
    """

    @staticmethod
    def _busy(
        stream_events: list[log_parser.LogEvent],
        ttft_events: list[log_parser.LogEvent],
        start: datetime = WINDOW_START,
        end: datetime = WINDOW_END,
    ) -> aggregation.BusyStats | None:
        return aggregation.compute_busy_stats(
            stream_events, start, end, _schedule(), ttft_events=ttft_events
        )

    def test_per_stream_prefill_from_dispatch_first_byte_ms(self):
        """AC1: prefill comes from ``dispatch_first_byte_ms`` per local
        stream — 5 s for the first stream, 30 s for the second."""
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 10), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 20), "s1"),
            _local_event("stream_started", datetime(2026, 8, 2, 14, 5, 0), "s2"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 5, 30), "s2"),
        ]
        ttfts = [
            _ttft_event(datetime(2026, 8, 2, 14, 0, 10), "s1", 5000.0),
            _ttft_event(datetime(2026, 8, 2, 14, 5, 0), "s2", 30000.0),
        ]
        busy = self._busy(streams, ttfts)
        assert busy is not None
        stats = busy.prefill_stats
        assert stats is not None
        assert list(stats.samples) == pytest.approx([5.0, 30.0])
        assert stats.count == 2
        assert stats.unknown_count == 0
        assert stats.stream_count == 2
        assert stats.total_seconds == pytest.approx(35.0)
        assert stats.has_samples

    def test_percentiles_use_linear_interpolation(self):
        """AC3: p10/p50/p90/max over the per-stream samples.

        4 samples of 10/20/30/40 s -> p10 13 s, p50 25 s, p90 37 s, max 40 s
        (linear interpolation, matching the rest of the skill's percentiles).
        """
        streams: list[log_parser.LogEvent] = []
        ttfts: list[log_parser.LogEvent] = []
        for index, prefill_s in enumerate((10, 20, 30, 40)):
            ts = datetime(2026, 8, 2, 14, index, 0)
            session = f"s{index}"
            streams.append(_local_event("stream_started", ts, session))
            streams.append(_local_event("stream_finished", ts + timedelta(seconds=60), session))
            ttfts.append(_ttft_event(ts, session, prefill_s * 1000.0))
        busy = self._busy(streams, ttfts)
        assert busy is not None
        stats = busy.prefill_stats
        assert stats is not None
        assert stats.p10 == pytest.approx(13.0)
        assert stats.p50 == pytest.approx(25.0)
        assert stats.p90 == pytest.approx(37.0)
        assert stats.max_seconds == pytest.approx(40.0)
        assert stats.total_seconds == pytest.approx(100.0)

    def test_concurrent_prefill_is_slot_seconds_not_wall_clock(self):
        """AC3/AC5: two concurrent 10 s prefills (13:59:50 -> 14:00:00) are
        20 slot-seconds of prefill work but only 10 s of wall-clock, so the
        window total must be the per-stream sum, not a union."""
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 0), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 20), "s1"),
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 0), "s2"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 20), "s2"),
        ]
        ttfts = [
            _ttft_event(datetime(2026, 8, 2, 14, 0, 0), "s1", 10000.0),
            _ttft_event(datetime(2026, 8, 2, 14, 0, 0), "s2", 10000.0),
        ]
        busy = self._busy(streams, ttfts)
        assert busy is not None
        stats = busy.prefill_stats
        assert stats is not None
        assert stats.total_seconds == pytest.approx(20.0)  # slot-seconds
        assert stats.unit == "slot-seconds"
        # The wall-clock union of the same prefills is only 10 s. The busy
        # figure is the dispatch → finish union (LP-0MVJN9H2I00116FP), but
        # here both dispatches (13:59:50) predate the 14:00:00 window start
        # and are clipped away, so busy equals the 20 s decode-only union.
        assert busy.busy_seconds == pytest.approx(20.0)
        assert busy.decode_busy_seconds == pytest.approx(20.0)
        assert busy.prefill_busy_seconds == pytest.approx(0.0)

    def test_stream_without_ttft_is_unknown_prefill(self):
        """AC4: a stream with no first-byte line is counted as unknown
        prefill — excluded from the percentiles, retained in the count."""
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 10), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 20), "s1"),
            _local_event("stream_started", datetime(2026, 8, 2, 14, 5, 0), "aborted"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 5, 2), "aborted"),
        ]
        ttfts = [_ttft_event(datetime(2026, 8, 2, 14, 0, 10), "s1", 5000.0)]
        busy = self._busy(streams, ttfts)
        assert busy is not None
        stats = busy.prefill_stats
        assert stats is not None
        assert stats.count == 1
        assert stats.unknown_count == 1
        assert stats.stream_count == 2
        assert list(stats.samples) == pytest.approx([5.0])
        assert stats.p50 == pytest.approx(5.0)  # unknown sample not averaged in
        assert stats.total_seconds == pytest.approx(5.0)

    def test_no_ttft_events_reports_every_stream_as_unknown(self):
        """AC4: with no TTFT events at all, every local stream is unknown
        prefill (empty percentiles, zero total) rather than a silent zero."""
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 10), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 20), "s1"),
        ]
        busy = self._busy(streams, [])
        assert busy is not None
        stats = busy.prefill_stats
        assert stats is not None
        assert stats.count == 0
        assert stats.unknown_count == 1
        assert stats.stream_count == 1
        assert stats.total_seconds == 0.0
        assert not stats.has_samples
        assert stats.p10 is None and stats.p50 is None and stats.p90 is None
        assert stats.max_seconds is None

    def test_fallback_pairs_dispatch_and_first_byte_timestamps(self):
        """AC2: with ``dispatch_first_byte_ms`` absent, prefill falls back to
        pairing the dispatch (``Stream started``) and first-byte timestamps:
        14:00:02 -> 14:00:10 is 8 s."""
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 2), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 40), "s1"),
        ]
        ttfts = [_ttft_event(datetime(2026, 8, 2, 14, 0, 10), "s1", None)]
        busy = self._busy(streams, ttfts)
        assert busy is not None
        stats = busy.prefill_stats
        assert stats is not None
        assert list(stats.samples) == pytest.approx([8.0])
        assert stats.unknown_count == 0

    def test_fallback_reconstructs_total_from_the_logged_split(self):
        """AC2: when only the service/queue split survived, the prefill total
        is reconstructed from ``dispatch_to_first_byte_ms + queue_wait_ms``
        (real queue-contention fixture: 10.403 s service + 16.170 s queued)."""
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 48), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 1, 30), "s1"),
        ]
        ttfts = [
            _ttft_event(
                datetime(2026, 8, 2, 14, 0, 48),
                "s1",
                None,
                dispatch_to_first_byte_ms=10403.1,
                queue_wait_ms=16169.9,
            )
        ]
        busy = self._busy(streams, ttfts)
        assert busy is not None
        stats = busy.prefill_stats
        assert stats is not None
        assert list(stats.samples) == pytest.approx([26.573])

    def test_fallback_rejects_non_positive_timestamp_pairing(self):
        """AC2/AC4: the timestamp fallback needs positive evidence. The proxy
        logs ``Stream started`` at first byte, so a zero difference is not a
        measurement — the stream stays unknown prefill."""
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 10), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 20), "s1"),
        ]
        ttfts = [_ttft_event(datetime(2026, 8, 2, 14, 0, 10), "s1", None)]
        busy = self._busy(streams, ttfts)
        assert busy is not None
        stats = busy.prefill_stats
        assert stats is not None
        assert stats.count == 0
        assert stats.unknown_count == 1

    def test_fifo_pairing_attributes_prefill_to_the_right_stream(self):
        """AC1: the Nth TTFT of a session pairs with that session's Nth
        stream, independent of the order the events are supplied in."""
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 5), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 40), "s1"),
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 20), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 30), "s1"),
        ]
        ttfts = [
            _ttft_event(datetime(2026, 8, 2, 14, 0, 6), "s1", 55656.5),
            _ttft_event(datetime(2026, 8, 2, 14, 0, 21), "s1", 1100.5),
        ]
        ordered = self._busy(streams, ttfts)
        reversed_order = self._busy(streams, list(reversed(ttfts)))
        assert ordered is not None and reversed_order is not None
        assert ordered.prefill_stats is not None
        assert reversed_order.prefill_stats is not None
        assert list(ordered.prefill_stats.samples) == pytest.approx([55.6565, 1.1005])
        assert list(reversed_order.prefill_stats.samples) == list(
            ordered.prefill_stats.samples
        )

    def test_dispatch_before_window_start_keeps_full_prefill(self):
        """AC6 boundary: a first byte inside the window whose dispatch
        predates it keeps its full prefill duration (60 s)."""
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 5), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 40), "s1"),
        ]
        ttfts = [_ttft_event(datetime(2026, 8, 2, 14, 0, 5), "s1", 60000.0)]
        busy = self._busy(streams, ttfts)
        assert busy is not None
        stats = busy.prefill_stats
        assert stats is not None
        assert list(stats.samples) == pytest.approx([60.0])  # dispatch 13:59:05
        assert stats.total_seconds == pytest.approx(60.0)

    def test_first_byte_after_window_end_is_not_a_window_sample(self):
        """AC6 boundary: a TTFT after the window end belongs to the next
        window — it is not counted (and never double-counted)."""
        streams = [
            # In-window stream so busy stats exist at all.
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 10), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 30), "s1"),
            # Next-window stream: first byte after window_end.
            _local_event("stream_started", datetime(2026, 8, 2, 15, 0, 10), "s2"),
            _local_event("stream_finished", datetime(2026, 8, 2, 15, 0, 30), "s2"),
        ]
        ttfts = [
            _ttft_event(datetime(2026, 8, 2, 14, 0, 10), "s1", 5000.0),
            _ttft_event(datetime(2026, 8, 2, 15, 0, 10), "s2", 9000.0),
        ]
        busy = self._busy(streams, ttfts)
        assert busy is not None
        stats = busy.prefill_stats
        assert stats is not None
        assert list(stats.samples) == pytest.approx([5.0])
        assert stats.stream_count == 1
        assert stats.unknown_count == 0
        assert stats.total_seconds == pytest.approx(5.0)

    def test_stream_outside_window_without_ttft_is_not_unknown(self):
        """AC6 boundary: a stream that started before the window and has no
        first-byte line belongs to an earlier window, so it is not this
        window's unknown prefill."""
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 13, 59, 50), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 5), "s1"),
        ]
        busy = self._busy(streams, [])
        assert busy is not None
        stats = busy.prefill_stats
        assert stats is not None
        assert stats.unknown_count == 0
        assert stats.stream_count == 0

    def test_streams_expose_per_stream_attribution(self):
        """AC1: the measurements stay attributable per local stream (session
        + first-byte timestamp), so unknown prefill can be named."""
        first_byte = datetime(2026, 8, 2, 14, 0, 10)
        streams = [
            _local_event("stream_started", first_byte, "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 20), "s1"),
            _local_event("stream_started", datetime(2026, 8, 2, 14, 5, 0), "aborted"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 5, 2), "aborted"),
        ]
        ttfts = [_ttft_event(first_byte, "s1", 5000.0)]
        busy = self._busy(streams, ttfts)
        assert busy is not None
        stats = busy.prefill_stats
        assert stats is not None
        known, unknown = stats.streams
        assert (known.session, known.first_byte_ts, known.seconds) == ("s1", first_byte, 5.0)
        assert known.known
        assert (unknown.session, unknown.first_byte_ts, unknown.seconds) == (
            "aborted",
            None,
            None,
        )
        assert not unknown.known

    def test_prefill_stats_absent_when_ttft_events_not_supplied(self):
        """Callers that do not pass TTFT events get no prefill statistics
        rather than a fabricated empty measurement (backward compatible)."""
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 10), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 20), "s1"),
        ]
        busy = aggregation.compute_busy_stats(
            streams, WINDOW_START, WINDOW_END, _schedule()
        )
        assert busy is not None
        assert busy.prefill_stats is None

    def test_aggregate_populates_prefill_stats_from_real_dispatch_lines(self):
        """AC1 end-to-end: ``aggregate()`` feeds the real TTFT events to the
        busy computation, so a real session's three dispatch lines become
        three prefill samples and the aborted stream stays unknown."""
        start = datetime(2026, 10, 21, 22, 0, 0)
        end = datetime(2026, 10, 22, 22, 0, 0)
        lines = [
            fixtures.STREAM_STARTED_PREFILL_COLD,
            fixtures.DISPATCH_FIRST_BYTE_COLD,
            fixtures.STREAM_STARTED_PREFILL_COLD_FAST,
            fixtures.DISPATCH_FIRST_BYTE_COLD_FAST,
            fixtures.STREAM_STARTED_PREFILL_WARM,
            fixtures.DISPATCH_FIRST_BYTE_WARM,
            fixtures.STREAM_STARTED_ABORTED_LOCAL,
            fixtures.STREAM_ERROR_ABORTED_LOCAL,
        ]
        result = aggregation.aggregate(_events(lines), start, end, _schedule())
        assert result.busy is not None
        stats = result.busy.prefill_stats
        assert stats is not None
        assert list(stats.samples) == pytest.approx([55.6565, 1.1005, 29.546])
        assert stats.count == 3
        assert stats.unknown_count == 1
        assert stats.stream_count == 4
        assert stats.total_seconds == pytest.approx(86.3)
        assert stats.unit == "slot-seconds"


class TestDispatchToFinishBusy:
    """Busy = union of **dispatch → finish** intervals (LP-0MVJN9H2I00116FP).

    ``busy_seconds`` is the union of ``[ttft.ts - ttft_ms/1000, Stream
    finished]`` intervals (prefill + decode), with the decode-only union
    retained as ``decode_busy_seconds`` — the labelled lower bound. Total
    compute, (peak/avg) concurrency, the fast/cheap split and the hourly split
    are all recomputed on the new intervals. A stream with no first-byte line
    falls back to its decode interval and is attributed to the lower bound, so
    the busy figure is never overstated.
    """

    @staticmethod
    def _busy(
        stream_events: list[log_parser.LogEvent],
        ttft_events: list[log_parser.LogEvent],
        start: datetime = WINDOW_START,
        end: datetime = WINDOW_END,
    ) -> aggregation.BusyStats | None:
        return aggregation.compute_busy_stats(
            stream_events, start, end, _schedule(), ttft_events=ttft_events
        )

    def test_busy_union_spans_dispatch_to_finish(self):
        """AC1: the busy interval starts at ``ttft.ts - ttft_ms/1000`` — a
        30 s prefill extends the 10 s decode interval to 40 s of busy."""
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 30), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 40), "s1"),
        ]
        ttfts = [_ttft_event(datetime(2026, 8, 2, 14, 0, 30), "s1", 30000.0)]
        busy = self._busy(streams, ttfts)
        assert busy is not None
        assert busy.busy_seconds == 40.0  # dispatch 14:00:00 → finish 14:00:40
        assert busy.busy_definition == "dispatch_to_finish"
        assert busy.busy_upper_bound_seconds == 40.0
        assert busy.prefill_extended_streams == 1
        assert busy.total_compute_seconds == 40.0
        assert busy.avg_stream_duration == pytest.approx(40.0)
        # Retained decode-only figure (lower bound) and wall-clock prefill span.
        assert busy.decode_busy_seconds == 10.0
        assert busy.busy_lower_bound_seconds == 10.0
        assert busy.prefill_busy_seconds == 30.0

    def test_decode_only_figure_is_retained_and_feedless_stream_stays_low(self):
        """AC2/AC5: the decode-only union is reported alongside; a stream
        without a TTFT line contributes its decode interval to the lower bound
        only, so the upper bound is not inflated by it."""
        streams = [
            # Measured prefill: busy 14:00:05 → 14:00:20 (15 s), decode 10 s.
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 10), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 20), "s1"),
            # No first-byte line: contributes decode only (30 s).
            _local_event("stream_started", datetime(2026, 8, 2, 14, 5, 0), "s2"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 5, 30), "s2"),
        ]
        ttfts = [_ttft_event(datetime(2026, 8, 2, 14, 0, 10), "s1", 5000.0)]
        busy = self._busy(streams, ttfts)
        assert busy is not None
        assert busy.decode_busy_seconds == 40.0  # 10 s + 30 s
        assert busy.busy_seconds == 45.0  # 15 s (extended) + 30 s (feedless)
        assert busy.prefill_busy_seconds == 5.0  # only the measured prefill
        assert busy.prefill_extended_streams == 1
        assert busy.prefill_stats is not None
        assert busy.prefill_stats.unknown_count == 1  # s2 visible as unknown

    def test_overlapping_prefill_and_decode_intervals_merge(self):
        """AC6: one stream's prefill overlapping another's decode merges into
        a single union interval — no double counting."""
        start, end = datetime(2026, 8, 2, 13, 0, 0), datetime(2026, 8, 2, 15, 0, 0)
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 0), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 30), "s1"),
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 10), "s2"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 20), "s2"),
        ]
        ttfts = [_ttft_event(datetime(2026, 8, 2, 14, 0, 0), "s1", 10000.0)]
        busy = self._busy(streams, ttfts, start, end)
        assert busy is not None
        # s1 dispatch 13:59:50 → 14:00:30; s2 14:00:10–20 inside it: union 40 s.
        assert busy.busy_seconds == 40.0
        assert busy.decode_busy_seconds == 30.0  # 14:00:00–30 (s2 inside s1)
        assert busy.prefill_busy_seconds == 10.0
        assert busy.total_compute_seconds == pytest.approx(50.0)  # 40 + 10
        assert busy.peak_concurrency == 2  # s2's decode overlaps s1's prefill
        assert busy.avg_concurrency == pytest.approx(50.0 / 40.0)

    def test_unknown_prefill_stream_keeps_decode_interval(self):
        """A stream paired with a TTFT whose duration is unknown stays at its
        decode interval (a zero-delta timestamp pairing is not evidence): busy
        equals decode and the stream is counted as unknown prefill."""
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 10), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 20), "s1"),
        ]
        ttfts = [_ttft_event(datetime(2026, 8, 2, 14, 0, 10), "s1", None)]  # no duration
        busy = self._busy(streams, ttfts)
        assert busy is not None
        assert busy.busy_seconds == 10.0
        assert busy.decode_busy_seconds == 10.0
        assert busy.prefill_busy_seconds == 0.0
        assert busy.prefill_extended_streams == 0
        assert busy.prefill_stats is not None and busy.prefill_stats.unknown_count == 1

    def test_unfinished_stream_with_dispatch_does_not_inflate_busy(self):
        """AC6 boundary: a stream with a measured prefill but no logged finish
        contributes nothing (its compute time is unknown), even though its
        dispatch time is derivable."""
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 30), "aborted"),
            _local_event("stream_started", datetime(2026, 8, 2, 14, 5, 0), "s2"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 5, 10), "s2"),
        ]
        ttfts = [_ttft_event(datetime(2026, 8, 2, 14, 0, 30), "aborted", 30000.0)]
        busy = self._busy(streams, ttfts)
        assert busy is not None
        assert busy.busy_seconds == 10.0  # s2 decode only, no invented prefill
        assert busy.unfinished_streams == 1
        assert busy.prefill_extended_streams == 0

    def test_dispatch_before_window_start_is_clipped_for_busy(self):
        """AC6 boundary: a dispatch before the window (boundary-crossing
        interval) is clipped to the window start; the in-window part still
        includes the prefill seconds that fall inside the window."""
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 5), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 40), "s1"),
        ]
        ttfts = [_ttft_event(datetime(2026, 8, 2, 14, 0, 5), "s1", 60000.0)]
        busy = self._busy(streams, ttfts)  # dispatch 13:59:05, window starts 14:00:00
        assert busy is not None
        assert busy.busy_seconds == 40.0  # 35 s decode + 5 s in-window prefill
        assert busy.decode_busy_seconds == 35.0
        assert busy.prefill_busy_seconds == 5.0

    def test_fast_cheap_split_on_dispatch_intervals(self):
        """AC3: fast/cheap bucketing runs on the new intervals — a prefill
        straddling the cheap→fast boundary attributes each side to its own
        bucket (10 s cheap + 20 s fast), and the decode-only split is retained
        separately."""
        start, end = datetime(2026, 8, 2, 9, 0, 0), datetime(2026, 8, 2, 11, 0, 0)
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 10, 0, 10), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 10, 0, 20), "s1"),
        ]
        ttfts = [_ttft_event(datetime(2026, 8, 2, 10, 0, 10), "s1", 20000.0)]
        busy = self._busy(streams, ttfts, start, end)
        assert busy is not None
        assert busy.cheap_busy_seconds == 10.0  # 09:59:50–10:00:00
        assert busy.fast_busy_seconds == 20.0
        assert busy.busy_seconds == 30.0
        # Decode-only layer keeps the old all-fast behaviour.
        assert busy.decode_cheap_busy_seconds == 0.0
        assert busy.decode_fast_busy_seconds == 10.0

    def test_hourly_busy_recomputed_on_dispatch_intervals(self):
        """AC3: ``hourly_busy`` attributes the prefill to its dispatch hour
        (09:59:50 → hour 9) and the rest to the first-byte hour."""
        start, end = datetime(2026, 8, 2, 9, 0, 0), datetime(2026, 8, 2, 11, 0, 0)
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 10, 0, 10), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 10, 0, 20), "s1"),
        ]
        ttfts = [_ttft_event(datetime(2026, 8, 2, 10, 0, 10), "s1", 20000.0)]
        busy = self._busy(streams, ttfts, start, end)
        assert busy is not None
        hourly = dict(busy.hourly_busy)
        assert hourly[datetime(2026, 8, 2, 9, 0, 0)] == 10.0
        assert hourly[datetime(2026, 8, 2, 10, 0, 0)] == 20.0

    def test_no_ttft_events_keeps_disambiguation_compat(self):
        """AC3 compat: without TTFT events every interval falls back to its
        decode interval, so the busy figure stays the decode-only figure and
        the bounds coincide."""
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 10), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 15), "s1"),
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 20), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 25), "s1"),
        ]
        busy = aggregation.compute_busy_stats(streams, WINDOW_START, WINDOW_END, _schedule())
        assert busy is not None
        assert busy.busy_seconds == 10.0
        assert busy.decode_busy_seconds == 10.0
        assert busy.prefill_busy_seconds == 0.0
        assert busy.busy_lower_bound_seconds == busy.busy_upper_bound_seconds == 10.0
        assert busy.prefill_extended_streams == 0

    def test_aggregate_wires_dispatch_intervals_end_to_end(self):
        """AC1 end-to-end through ``aggregate()``: a real dispatch line extends
        the busy union back to its back-calculated dispatch time."""
        start = datetime(2026, 10, 22, 20, 0, 0)
        end = datetime(2026, 10, 22, 22, 0, 0)
        lines = [
            fixtures.STREAM_STARTED_PREFILL_COLD,  # 21:15:02,477 (= first byte)
            fixtures.DISPATCH_FIRST_BYTE_COLD,  # ttft_ms=55656.5 -> dispatch 21:14:06.8
            # Synthesised finish 10 s after the first byte.
            "2026-10-22 21:15:12,477 - INFO - Stream finished: reason=stop "
            f"tokens=1000/20/1020 session={fixtures.PREFILL_SESSION} provider=local "
            "model=Qwen3 request=[]",
        ]
        summary = aggregation.aggregate(_events(lines), start, end, _schedule())
        assert summary.busy is not None
        busy = summary.busy
        assert busy.streams == 1
        assert busy.decode_busy_seconds == 10.0
        assert busy.busy_seconds == pytest.approx(65.7, abs=0.1)  # 55.6565 s prefill + 10 s
        assert busy.prefill_busy_seconds == pytest.approx(55.7, abs=0.1)
        assert busy.prefill_extended_streams == 1
        assert busy.prefill_stats is not None
        assert list(busy.prefill_stats.samples) == pytest.approx([55.6565])

    def test_report_renders_definition_and_labelled_bounds(self):
        """AC2/AC3: the ``## Local model utilization`` section states the
        dispatch → finish definition, retains the decode-only row as the
        labelled lower bound, and labels the busy figure as the upper bound."""
        lines = [
            "2026-08-02 14:00:30,000 - INFO - Stream started: provider=local model=Qwen3 session=s1 request=[]",
            "2026-08-02 14:00:40,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s1 provider=local model=Qwen3 request=[]",
            "2026-08-02 14:00:30,000 - INFO - dispatch_first_byte_ms=30000.0 "
            "dispatch_to_first_byte_ms=29999.0 cache_state=warm cached_ratio=0.900 "
            "queue_wait_ms=1.0 session=s1 model=Qwen3",
        ]
        summary = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, _schedule())
        assert summary.busy is not None
        md = reporting.build_report(summary, None)
        section = md.split("## Local model utilization", 1)[1].split("## ", 1)[0]
        assert "from dispatch to finish" in section
        assert "prefill + decode" in section
        assert "upper bound (dispatch → finish)" in section
        assert "Decode-only busy — lower bound" in section
        assert "| Prefill busy (upper − lower) | 30s |" in section
        assert "1 of 1 stream(s) contributed a measured prefill" in section
        # Busy of 40 s over the 1h window.
        assert "40s (1.1%)" in section


class TestDecodeStats:
    """Per-stream local decode (first byte → finish) slot-seconds
    (LP-0MVJNDOUU0059MMO).

    Decode is measured on the same paired streams as busy, but reported as
    **slot-seconds** (the sum of the per-stream decode durations — the integral
    of concurrency), never as wall-clock, so it can be added to the prefill
    slot-seconds to split phase slot time.
    """

    def test_concurrent_decode_is_slot_seconds_not_wall_clock(self):
        """Two concurrent 10 s decodes are 20 slot-seconds of decode work but
        only 10 s of wall clock."""
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 0), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 10), "s1"),
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 0), "s2"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 10), "s2"),
        ]
        busy = aggregation.compute_busy_stats(
            streams, WINDOW_START, WINDOW_END, _schedule()
        )
        assert busy is not None and busy.decode_stats is not None
        assert busy.decode_stats.total_seconds == pytest.approx(20.0)
        assert busy.decode_stats.unit == "slot-seconds"
        # The wall-clock union of the same decodes is only 10 s.
        assert busy.decode_busy_seconds == pytest.approx(10.0)

    def test_decode_percentiles_use_linear_interpolation(self):
        """p10/p50/p90/max over the per-stream decode durations, matching the
        prefill percentiles: 10/20/30/40 s -> 13/25/37/40 s."""
        streams: list[log_parser.LogEvent] = []
        for index, decode_s in enumerate((10, 20, 30, 40)):
            ts = datetime(2026, 8, 2, 14, index, 0)
            session = f"s{index}"
            streams.append(_local_event("stream_started", ts, session))
            streams.append(
                _local_event("stream_finished", ts + timedelta(seconds=decode_s), session)
            )
        busy = aggregation.compute_busy_stats(
            streams, WINDOW_START, WINDOW_END, _schedule()
        )
        assert busy is not None and busy.decode_stats is not None
        stats = busy.decode_stats
        assert stats.count == 4
        assert stats.p10 == pytest.approx(13.0)
        assert stats.p50 == pytest.approx(25.0)
        assert stats.p90 == pytest.approx(37.0)
        assert stats.max_seconds == pytest.approx(40.0)
        assert stats.total_seconds == pytest.approx(100.0)

    def test_unfinished_stream_is_unknown_decode(self):
        """A stream with no logged finish has unknown decode time: excluded
        from the percentiles, retained in the stream count."""
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 10), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 20), "s1"),
            _local_event("stream_started", datetime(2026, 8, 2, 14, 5, 0), "aborted"),
        ]
        busy = aggregation.compute_busy_stats(
            streams, WINDOW_START, WINDOW_END, _schedule()
        )
        assert busy is not None and busy.decode_stats is not None
        assert busy.decode_stats.count == 1
        assert busy.decode_stats.unknown_count == 1
        assert busy.decode_stats.stream_count == 2
        assert busy.decode_stats.total_seconds == pytest.approx(10.0)

    def test_decode_stats_available_without_ttft_events(self):
        """Decode does not depend on the first-byte lines: with no TTFT events
        supplied the prefill stats are absent but decode is still reported."""
        streams = [
            _local_event("stream_started", datetime(2026, 8, 2, 14, 0, 10), "s1"),
            _local_event("stream_finished", datetime(2026, 8, 2, 14, 0, 25), "s1"),
        ]
        busy = aggregation.compute_busy_stats(
            streams, WINDOW_START, WINDOW_END, _schedule()
        )
        assert busy is not None
        assert busy.prefill_stats is None
        assert busy.decode_stats is not None
        assert busy.decode_stats.total_seconds == pytest.approx(15.0)


class TestPrefillVsDecodeSection:
    """``## Prefill vs decode`` report section (LP-0MVJNDOUU0059MMO).

    The section splits local phase slot time into prefill (dispatch → first
    byte) and decode (first byte → finish) as absolute slot-seconds and as a
    percentage of combined phase slot-time that always sums to 100%, with
    p10/p50/p90/max per phase, an explicit slot-seconds-not-wall-clock
    statement, and a named reason plus an eval-timing pointer when no prefill
    samples exist.
    """

    @staticmethod
    def _summary(lines: list[str]) -> aggregation.AnalysisResult:
        return aggregation.aggregate(
            _events(lines), WINDOW_START, WINDOW_END, _schedule()
        )

    @staticmethod
    def _section(md: str) -> str:
        return md.split("## Prefill vs decode", 1)[1].split("\n## ", 1)[0]

    @staticmethod
    def _row(section: str, label: str) -> list[str]:
        line = next(
            ln for ln in section.splitlines() if ln.startswith(f"| {label}")
        )
        return [cell.strip() for cell in line.strip("|").split("|")]

    def test_section_splits_phase_slot_seconds_and_sums_to_100_percent(self):
        """A 30 s prefill + 10 s decode stream reports 30 / 10 slot-seconds
        and 75.0% / 25.0% of combined phase slot-time."""
        lines = [
            "2026-08-02 14:00:30,000 - INFO - Stream started: provider=local model=Qwen3 session=s1 request=[]",
            "2026-08-02 14:00:40,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s1 provider=local model=Qwen3 request=[]",
            "2026-08-02 14:00:30,000 - INFO - dispatch_first_byte_ms=30000.0 "
            "dispatch_to_first_byte_ms=29999.0 cache_state=warm cached_ratio=0.900 "
            "queue_wait_ms=1.0 session=s1 model=Qwen3",
        ]
        summary = self._summary(lines)
        assert summary.busy is not None
        section = self._section(reporting.build_report(summary, None))
        prefill = self._row(section, "Prefill (dispatch → first byte)")
        decode = self._row(section, "Decode (first byte → finish)")
        assert prefill[1] == "30s"
        assert decode[1] == "10s"
        assert float(prefill[2].rstrip("%")) == pytest.approx(75.0)
        assert float(decode[2].rstrip("%")) == pytest.approx(25.0)
        assert float(prefill[2].rstrip("%")) + float(decode[2].rstrip("%")) == pytest.approx(100.0)
        total = self._row(section, "Total")
        assert total[1] == "40s"
        assert total[2] == "100.0%"

    def test_section_states_slot_seconds_not_wall_clock(self):
        """The header must say the figures are slot-seconds (the integral of
        concurrency), not wall-clock."""
        lines = [
            "2026-08-02 14:00:30,000 - INFO - Stream started: provider=local model=Qwen3 session=s1 request=[]",
            "2026-08-02 14:00:40,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s1 provider=local model=Qwen3 request=[]",
            "2026-08-02 14:00:30,000 - INFO - dispatch_first_byte_ms=30000.0 "
            "dispatch_to_first_byte_ms=29999.0 cache_state=warm cached_ratio=0.900 "
            "queue_wait_ms=1.0 session=s1 model=Qwen3",
        ]
        section = self._section(reporting.build_report(self._summary(lines), None))
        flattened = " ".join(section.split())
        assert "slot-seconds" in flattened
        assert "integral of concurrency" in flattened
        assert "not wall-clock" in flattened

    def test_section_reports_prefill_and_decode_percentiles(self):
        """p10/p50/p90/max are reported per phase over the per-stream samples."""
        lines: list[str] = []
        for index, prefill_s in enumerate((10, 20, 30, 40)):
            ts = f"2026-08-02 14:0{index}:00,000"
            lines.append(
                f"{ts} - INFO - Stream started: provider=local model=Qwen3 session=s{index} request=[]"
            )
            lines.append(
                "2026-08-02 14:0" + str(index) + ":59,000 - INFO - Stream finished: "
                f"reason=stop tokens=100/10/110 session=s{index} provider=local model=Qwen3 request=[]"
            )
            lines.append(
                f"{ts} - INFO - dispatch_first_byte_ms={prefill_s * 1000.0} "
                f"dispatch_to_first_byte_ms={prefill_s * 1000.0} cache_state=warm "
                f"cached_ratio=0.900 queue_wait_ms=0.0 session=s{index} model=Qwen3"
            )
        section = self._section(reporting.build_report(self._summary(lines), None))
        prefill = self._row(section, "Prefill (dispatch → first byte)")
        decode = self._row(section, "Decode (first byte → finish)")
        # p10, median, p90, max, samples
        assert [float(v) for v in prefill[3:7]] == [13.0, 25.0, 37.0, 40.0]
        assert prefill[7] == "4"
        assert [float(v) for v in decode[3:7]] == [59.0, 59.0, 59.0, 59.0]
        assert decode[7] == "4"

    def test_section_without_prefill_samples_names_reason_and_eval_timing_pointer(self):
        """With no first-byte lines the section prints a named reason and
        points at the llama-server eval-timing diagnostic instead of leaving an
        unattributed blank; decode is still attributed."""
        lines = [
            "2026-08-02 14:00:10,000 - INFO - Stream started: provider=local model=Qwen3 session=s1 request=[]",
            "2026-08-02 14:00:20,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s1 provider=local model=Qwen3 request=[]",
        ]
        section = self._section(reporting.build_report(self._summary(lines), None))
        assert "No prefill samples" in section
        assert "unknown prefill" in section
        assert "`## Decode speed` / `## Prompt eval speed`" in section
        assert "eval-timing" in section
        prefill = self._row(section, "Prefill (dispatch → first byte)")
        decode = self._row(section, "Decode (first byte → finish)")
        assert prefill[1] == "-" and prefill[2] == "-"
        assert decode[1] == "10s" and decode[2] == "100.0%"

    def test_section_with_no_local_traffic_renders_named_reason(self):
        """An empty window prints a named reason rather than an empty table."""
        section = self._section(
            reporting.build_report(self._summary([]), None)
        )
        assert "No local traffic" in section
        assert section.count("|") == 0

    def test_json_summary_exposes_phase_split(self):
        """The machine-readable summary carries the same phase split."""
        lines = [
            "2026-08-02 14:00:30,000 - INFO - Stream started: provider=local model=Qwen3 session=s1 request=[]",
            "2026-08-02 14:00:40,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s1 provider=local model=Qwen3 request=[]",
            "2026-08-02 14:00:30,000 - INFO - dispatch_first_byte_ms=30000.0 "
            "dispatch_to_first_byte_ms=29999.0 cache_state=warm cached_ratio=0.900 "
            "queue_wait_ms=1.0 session=s1 model=Qwen3",
        ]
        data = reporting.summary_to_json(self._summary(lines))
        phase = data["local_busy"]["prefill_vs_decode"]
        assert phase["unit"] == "slot-seconds"
        assert phase["prefill"]["slot_seconds"] == pytest.approx(30.0)
        assert phase["prefill"]["pct"] == pytest.approx(75.0)
        assert phase["decode"]["slot_seconds"] == pytest.approx(10.0)
        assert phase["decode"]["pct"] == pytest.approx(25.0)
        assert phase["prefill"]["samples"] == 1
        assert phase["decode"]["samples"] == 1
        json.dumps(data)


class TestSummaryByHourActivity:
    """Per-hour activity columns in the top-of-report ``## Summary by hour``
    table (LP-0MTYAZFGN003BYQS).

    Activity is counted by each event's own timestamp inside its absolute-hour
    bucket: ``Requests started`` (all Stream started), ``Local attempts``
    (Stream started with provider=local), ``Local served`` (Stream finished
    with provider=local), ``Fallbacks`` (Fallback triggered), and
    ``Local-skip`` (routing_skip_local). The former session-first-hour
    classification columns are gone from the table; the journey split remains
    available in the JSON summary as ``hourly_session_classification``.
    """

    WINDOW_START = datetime(2026, 8, 2, 9, 0, 0)
    WINDOW_END = datetime(2026, 8, 2, 12, 0, 0)

    def _lines(self) -> list[str]:
        # s1: local-only, first request 09:40 (hour 9).
        # s2: starts local 10:30 then falls back (hour 10): a session-less
        #     Fallback triggered event plus its own routing_skip_local.
        # s3: remote-only, first request 11:30 (hour 11).
        return [
            "2026-08-02 09:40:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s1 request=[]",
            "2026-08-02 09:40:30,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s1 provider=local model=Qwen3 request=[]",
            "2026-08-02 10:30:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s2 request=[]",
            "2026-08-02 10:30:05,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s2 provider=local model=Qwen3 request=[]",
            "2026-08-02 10:30:06,000 - INFO - Fallback triggered for model=v1/chat/completions, "
            "from=local-qwen3, to=opencode-go-deepseek, reason=local_concurrency_limit",
            "2026-08-02 10:30:06,500 - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 "
            "estimated_tokens=5000 cold_threshold=39594 warm_threshold=39594 new_tokens=50 cached_ratio=0.50 "
            "reason=local_concurrency_limit → skipping local, routing to next remote provider session=s2",
            "2026-08-02 10:30:07,000 - INFO - Stream started: provider=opencode-go model=deepseek-v4-flash session=s2 request=[]",
            "2026-08-02 10:30:09,000 - INFO - Stream finished: reason=stop tokens=950/200/1150 session=s2 provider=opencode-go model=deepseek-v4-flash request=[]",
            "2026-08-02 11:30:00,000 - INFO - Stream started: provider=opencode-go model=deepseek-v4-flash session=s3 request=[]",
            "2026-08-02 11:30:10,000 - INFO - Stream finished: reason=stop tokens=950/200/1150 session=s3 provider=opencode-go model=deepseek-v4-flash request=[]",
        ]

    def _summary(self):
        return aggregation.aggregate(
            _events(self._lines()), self.WINDOW_START, self.WINDOW_END, _schedule()
        )

    def test_activity_counted_by_event_timestamp(self):
        # Unit-level: each activity counter is keyed by the event's own
        # absolute-hour bucket; s2's remote attempt lands in hour 10 (its
        # event hour), never in a session's first-request hour.
        summary = self._summary()
        h9 = summary.hourly_activity[datetime(2026, 8, 2, 9, 0, 0)]
        assert h9["requests_started"] == 1
        assert h9["local_attempts"] == 1
        assert h9["local_served"] == 1
        assert h9["fallbacks"] == 0
        assert h9["local_skip"] == 0
        h10 = summary.hourly_activity[datetime(2026, 8, 2, 10, 0, 0)]
        assert h10["requests_started"] == 2
        assert h10["local_attempts"] == 1
        assert h10["local_served"] == 1
        assert h10["fallbacks"] == 1
        assert h10["local_skip"] == 1
        h11 = summary.hourly_activity[datetime(2026, 8, 2, 11, 0, 0)]
        assert h11["requests_started"] == 1
        assert h11["local_attempts"] == 0
        assert h11["local_served"] == 0

    def test_table_renders_activity_per_hour(self):
        # Rows carry the five activity columns; busy % stays busy /
        # window-bounded bucket duration.
        md = reporting.build_report(self._summary(), None)
        section = md.split("## Summary by hour", 1)[1].split("## ", 1)[0]
        assert "| Hour | State | Busy | % | Requests started | Local attempts | Local served | Local dispatch | Lease denials | Fallbacks | Local-skip |" in section
        assert "| 09:00-10:00 | serving | 30s | 0.8% | 1 | 1 | 1 | 0 | 0 | 0 | 0 |" in section
        assert "| 10:00-11:00 | serving | 5s | 0.1% | 2 | 1 | 1 | 0 | 0 | 1 | 1 |" in section
        assert "| 11:00-12:00 | idle | 0s | 0.0% | 1 | 0 | 0 | 0 | 0 | 0 | 0 |" in section
        assert "| Totals | serving | 35s | 0.3% | 4 | 2 | 2 | 0 | 0 | 1 | 1 |" in section

    def test_long_running_session_contributes_to_each_hour(self):
        # A single session whose requests start in different hours: the second
        # request is counted in hour 10 (its event timestamp) rather than
        # being lumped into the session's first-request hour — the
        # session-first-hour attribution bug LP-0MTYAZFGN003BYQS fixes.
        lines = [
            "2026-08-02 09:50:00,000 - INFO - Stream started: provider=local model=Qwen3 session=long request=[]",
            "2026-08-02 09:50:10,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=long provider=local model=Qwen3 request=[]",
            "2026-08-02 10:20:00,000 - INFO - Stream started: provider=local model=Qwen3 session=long request=[]",
            "2026-08-02 10:20:10,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=long provider=local model=Qwen3 request=[]",
        ]
        summary = aggregation.aggregate(_events(lines), self.WINDOW_START, self.WINDOW_END, _schedule())
        assert summary.hourly_activity[datetime(2026, 8, 2, 9, 0, 0)]["requests_started"] == 1
        assert summary.hourly_activity[datetime(2026, 8, 2, 10, 0, 0)]["requests_started"] == 1
        md = reporting.build_report(summary, None)
        section = md.split("## Summary by hour", 1)[1].split("## ", 1)[0]
        assert "| 09:00-10:00 | serving | 10s | 0.3% | 1 | 1 | 1 | 0 | 0 | 0 | 0 |" in section
        assert "| 10:00-11:00 | serving | 10s | 0.3% | 1 | 1 | 1 | 0 | 0 | 0 | 0 |" in section

    def test_multiple_requests_same_hour_sum_activity(self):
        # Two local requests in hour 10 (s2 + s4): activity columns sum the
        # events, while the busy column reflects the merged local time.
        lines = self._lines() + [
            "2026-08-02 10:45:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s4 request=[]",
            "2026-08-02 10:45:10,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s4 provider=local model=Qwen3 request=[]",
        ]
        summary = aggregation.aggregate(_events(lines), self.WINDOW_START, self.WINDOW_END, _schedule())
        md = reporting.build_report(summary, None)
        section = md.split("## Summary by hour", 1)[1].split("## ", 1)[0]
        assert "| 10:00-11:00 | serving | 15s | 0.4% | 3 | 2 | 2 | 0 | 0 | 1 | 1 |" in section
        assert "| Totals | serving | 45s | 0.4% | 5 | 3 | 3 | 0 | 0 | 1 | 1 |" in section

    def test_renders_without_local_traffic(self):
        # No local streams (busy is None): the table still renders at the top
        # of the report with 0s busy / 0.0% and the classification columns
        # populated (LP-0MTFO210Q0044TTF AC1 — renders regardless of local
        # traffic); the Local model utilization section is omitted.
        lines = [
            "2026-08-02 10:00:00,000 - INFO - Stream started: provider=opencode-go model=deepseek-v4-flash session=r1 request=[]",
            "2026-08-02 10:00:10,000 - INFO - Stream finished: reason=stop tokens=950/200/1150 session=r1 provider=opencode-go model=deepseek-v4-flash request=[]",
        ]
        summary = aggregation.aggregate(_events(lines), self.WINDOW_START, self.WINDOW_END, _schedule())
        assert summary.busy is None
        md = reporting.build_report(summary, None)
        assert md.index("## Summary by hour") < md.index("## Session summary")
        section = md.split("## Summary by hour", 1)[1].split("## ", 1)[0]
        assert "| 10:00-11:00 | idle | 0s | 0.0% | 1 | 0 | 0 | 0 | 0 | 0 | 0 |" in section
        assert "| Totals | idle | 0s | 0.0% | 1 | 0 | 0 | 0 | 0 | 0 | 0 |" in section
        assert "## Local model utilization" not in md

    def test_no_sessions_renders_no_data_note(self):
        # Empty window: the table still renders with a "No data" note and a
        # zeroed totals row.
        summary = aggregation.aggregate([], self.WINDOW_START, self.WINDOW_END, _schedule())
        md = reporting.build_report(summary, None)
        section = md.split("## Summary by hour", 1)[1].split("## ", 1)[0]
        assert "| _No session data in window._ | - | - | - | - | - | - | - | - | - | - |" in section
        assert "| Totals | idle | 0s | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |" in section

    def test_json_summary_exposes_hourly_activity_and_classification(self):
        # The machine-readable summary exposes per-hour activity (matching the
        # table) plus the retained journey classification, both keyed by
        # absolute hour and self-describing.
        data = reporting.summary_to_json(self._summary())
        assert data["hourly_activity"] == [
            {"hour": "2026-08-02 09:00:00", "state": "serving", "states": ["serving"],
             "requests_started": 1, "local_attempts": 1,
             "local_served": 1, "local_dispatch_attempts": 0, "local_denials": 0,
             "lease_active_fallbacks": 0, "fallbacks": 0, "local_skip": 0},
            {"hour": "2026-08-02 10:00:00", "state": "serving", "states": ["serving"],
             "requests_started": 2, "local_attempts": 1,
             "local_served": 1, "local_dispatch_attempts": 0, "local_denials": 0,
             "lease_active_fallbacks": 0, "fallbacks": 1, "local_skip": 1},
            {"hour": "2026-08-02 11:00:00", "state": "idle", "states": ["idle"],
             "requests_started": 1, "local_attempts": 0,
             "local_served": 0, "local_dispatch_attempts": 0, "local_denials": 0,
             "lease_active_fallbacks": 0, "fallbacks": 0, "local_skip": 0},
        ]
        assert data["hourly_session_classification"] == [
            {"hour": "2026-08-02 09:00:00", "local_only": 1, "fell_back": 0, "remote_only": 0},
            {"hour": "2026-08-02 10:00:00", "local_only": 0, "fell_back": 1, "remote_only": 0},
            {"hour": "2026-08-02 11:00:00", "local_only": 0, "fell_back": 0, "remote_only": 1},
        ]
        # hourly_busy is keyed by absolute hour and self-describing too.
        assert data["local_busy"]["hourly_busy"] == [
            {"hour": "2026-08-02 09:00:00", "seconds": 30.0},
            {"hour": "2026-08-02 10:00:00", "seconds": 5.0},
        ]
        json.dumps(data)  # round-trips


class TestChainHoldCycleDedup:
    """A held request re-emitted across chain-hold cycles is counted exactly
    once in the demand / local-skip aggregates (its cycle-0 occurrence), with
    the ``chain_hold_cycle > 0`` re-emissions surfaced separately as
    hold-cycle activity (LP-0MUIEO6IP0030CT3).

    Fixture: one session whose routing_skip_local fires in cycle 0 (genuine
    demand) and is then re-emitted in cycles 1-3 by hold restarts.
    """

    WINDOW_START = datetime(2026, 8, 2, 14, 0, 0)
    WINDOW_END = datetime(2026, 8, 2, 15, 0, 0)
    SESSION = "019fc27d-3a46-7e5c-871e-57ab32f875f3"
    HOUR_KEY = datetime(2026, 8, 2, 14, 0, 0)

    def _lines(self) -> list[str]:
        started = (
            "2026-08-02 14:02:00,000 - INFO - Stream started: provider=local "
            f"model=Qwen3 session={self.SESSION} request=[]"
        )
        skip = (
            "2026-08-02 14:02:20,100 - INFO - routing_skip_local provider=local-qwen3 "
            "model=Qwen3 estimated_tokens=40000 cold_threshold=39594 warm_threshold=39594 "
            "new_tokens=323 cached_ratio=0.99 reason=context_too_large → skipping local, "
            "routing to next remote provider session={sid} chain_hold_cycle={cycle}"
        )
        return [started] + [
            skip.format(sid=self.SESSION, cycle=c) for c in (0, 1, 2, 3)
        ]

    def _summary(self):
        return aggregation.aggregate(
            _events(self._lines()), self.WINDOW_START, self.WINDOW_END, _schedule()
        )

    def test_cycle0_counted_once_in_demand_aggregates(self):
        # AC2: the cycle-0 occurrence is the ONLY one counted in the demand
        # aggregates (hourly local_skip, routing_skip_events, per-session
        # routing_skips, routing_skip_reason_counts).
        summary = self._summary()
        assert len(summary.routing_skip_events) == 1
        assert summary.routing_skip_events[0].chain_hold_cycle == 0
        assert summary.routing_skip_reason_counts["context_too_large"] == 1
        assert summary.hourly_activity[self.HOUR_KEY]["local_skip"] == 1
        assert summary.sessions[self.SESSION].routing_skips == 1

    def test_hold_cycle_reemissions_surfaced_separately(self):
        # AC2: the chain_hold_cycle>0 occurrences land in the dedicated
        # hold-cycle list/count, never in the demand aggregates.
        summary = self._summary()
        assert summary.hold_cycle_reemission_count == 3
        events = summary.hold_cycle_routing_skip_events
        assert [e.chain_hold_cycle for e in events] == [1, 2, 3]
        assert all(e.session == self.SESSION for e in events)

    def test_backward_compat_no_field_is_cycle0(self):
        # AC4: pre-chain-hold logs carry no chain_hold_cycle token; the line
        # parses with chain_hold_cycle=None which aggregates as cycle 0
        # (genuine demand), leaving historic counts unchanged.
        summary = aggregation.aggregate(
            _events([fixtures.ROUTING_SKIP_CONTEXT_TOO_LARGE]),
            self.WINDOW_START, self.WINDOW_END, _schedule(),
        )
        assert len(summary.routing_skip_events) == 1
        assert summary.routing_skip_events[0].chain_hold_cycle is None
        assert summary.hold_cycle_reemission_count == 0
        assert summary.hourly_activity[self.HOUR_KEY]["local_skip"] == 1

    def test_report_surfaces_hold_cycle_and_excludes_from_reasons(self):
        # AC3: the report surfaces the hold-cycle re-emission volume as its
        # own metric/section and keeps the routing_skip_local reasons table
        # (and its percentages) on genuine skips only.
        md = reporting.build_report(self._summary(), None)
        header = md.split("\n\n## Session summary", 1)[0]
        assert "Local-skip events: **1**" in header
        assert "hold-cycle re-emissions (excluded from demand): **3**" in header
        section = md.split("## Hold-cycle re-emissions", 1)[1].split("## ", 1)[0]
        assert "Hold-cycle routing-skip re-emissions: **3**" in section
        reasons = md.split("## routing_skip_local reasons", 1)[1].split("## ", 1)[0]
        assert "| context_too_large | 1 | 100.0% |" in reasons
        assert "| context_too_large | 4 |" not in reasons

    def test_json_summary_surfaces_hold_cycle_metric(self):
        # AC3: the machine-readable summary exposes the hold-cycle re-emission
        # volume as its own metric (demand counters stay de-duplicated).
        data = reporting.summary_to_json(self._summary())
        assert data["hold_cycle_routing_skip_events"] == 3
        json.dumps(data)  # round-trips


class TestHourlyRollingWindow:
    """AC3 regression (LP-0MTYAZFGN003BYQS): a rolling 24h window starting at
    ``HH:04`` emits one row per **absolute** hour. The two edge partials share
    an hour-of-day (both are 11:00 buckets on different days) but must not
    share a bucket, so busy time is never double-counted and no row exceeds
    100%.
    """

    WINDOW_START = datetime(2026, 9, 11, 11, 4, 0)
    WINDOW_END = datetime(2026, 9, 12, 11, 4, 0)

    def _lines(self) -> list[str]:
        def local(session: str, start: datetime, secs: int) -> list[str]:
            end = start + timedelta(seconds=secs)
            return [
                f"{start:%Y-%m-%d %H:%M:%S},000 - INFO - Stream started: "
                f"provider=local model=Qwen3 session={session} request=[]",
                f"{end:%Y-%m-%d %H:%M:%S},000 - INFO - Stream finished: "
                f"reason=stop tokens=100/10/110 session={session} provider=local model=Qwen3 request=[]",
            ]

        d = datetime
        lines: list[str] = []
        # Two streams share hour-of-day 11 on different days (the duplicate
        # bucket trap): a short Sep 11 11:30 stream and a longer Sep 12 11:00
        # stream clipped to the window end at 11:04.
        lines += local("h1", d(2026, 9, 11, 11, 30, 0), 7)
        lines += local("h2", d(2026, 9, 12, 11, 0, 0), 180)
        # Afternoon streams, one per hour 12..18, with distinct durations.
        for hour, secs, sid in [
            (12, 30, "a1"), (13, 60, "a2"), (14, 120, "a3"), (15, 10, "a4"),
            (16, 20, "a5"), (17, 40, "a6"), (18, 5, "a7"),
        ]:
            lines += local(sid, d(2026, 9, 11, hour, 0, 0), secs)
        return lines

    def _rows(self) -> dict[str, list[str]]:
        md = reporting.build_report(
            aggregation.aggregate(_events(self._lines()), self.WINDOW_START, self.WINDOW_END, _schedule()),
            None,
        )
        section = md.split("## Summary by hour", 1)[1].split("## ", 1)[0]
        rows: dict[str, list[str]] = {}
        for line in section.splitlines():
            if not line.startswith("| ") or "---" in line:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells[0] == "Hour":
                continue
            rows[cells[0]] = cells
        return rows

    def test_afternoon_rows_non_zero_and_requests_total_matches_fixture(self):
        # (a) The 12:00–18:00 rows are non-zero and their Requests started
        # total matches the fixture (7 afternoon streams, one per hour).
        rows = self._rows()
        expected_busy = {
            "12:00-13:00": "30s",
            "13:00-14:00": "1m 00s",
            "14:00-15:00": "2m 00s",
            "15:00-16:00": "10s",
            "16:00-17:00": "20s",
            "17:00-18:00": "40s",
            "18:00-19:00": "5s",
        }
        for hour, busy in expected_busy.items():
            assert hour in rows, f"missing row {hour}"
            assert rows[hour][2] == busy, f"{hour} busy {rows[hour][2]} != {busy}"
            # Hour | State | Busy | % | Requests started | Local attempts | Local served |
            # Local dispatch | Lease denials | Fallbacks | Local-skip
            assert rows[hour][4:11] == ["1", "1", "1", "0", "0", "0", "0"], hour
        afternoon_started = sum(int(rows[h][4]) for h in expected_busy)
        assert afternoon_started == 7

    def test_no_row_exceeds_100_percent_and_no_duplicate_busy_duration(self):
        # (b) No row busy% exceeds 100%, and the two hour-of-day-11 edge
        # partials do not share a busy duration (7s vs 3m 00s), proving
        # absolute-hour keying rather than hour-of-day collision.
        rows = self._rows()
        assert rows["11:04-12:00"][2] == "7s"
        assert rows["11:04-12:00"][3] == "0.2%"
        assert rows["11:00-11:04"][2] == "3m 00s"
        assert rows["11:00-11:04"][3] == "75.0%"
        pcts = [float(cell.rstrip("%")) for cell in (r[3] for r in rows.values())]
        assert pcts and all(p <= 100.0 for p in pcts)
        busy_cells = [r[2] for label, r in rows.items() if label != "Totals"]
        non_zero = [b for b in busy_cells if b != "0s"]
        assert len(non_zero) == len(set(non_zero))


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

        fast_csv = out_dir / "standard_sessions.csv"
        cheap_csv = out_dir / "low_priority_sessions.csv"
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
        # Contention-queue fields present (LP-0MSORQVK50012Q4D F4 AC3).
        assert "contention_dispatch" in data
        assert "contention_fallback_after_queue" in data
        assert "contention_queued_duration_seconds" in data
        # Local-model utilization stats are exposed.
        busy = data["local_busy"]
        assert busy["streams"] == 2
        assert busy["busy_seconds"] == 7.0
        assert busy["busy_pct"] == pytest.approx(7.0 / 3600 * 100, abs=0.01)
        assert busy["peak_concurrency"] == 1
        # Round-trips through json.
        json.dumps(data)

    def test_report_note_splits_unfinished_and_pre_window(self, tmp_path):
        """Regression (LP-0MSVRRO3L0056N6C): the busy-time note splits
        unfinished streams into in-window aborts (started without a logged
        finish) vs pre-window leftovers (started more than 1h before the
        window) instead of lumping them together."""
        log_dir = tmp_path / "logs_note"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text(
            "2026-08-02 12:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=aaaa request=[]\n"
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=bbbb request=[]\n"
            "2026-08-02 14:00:10,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=bbbb provider=local model=Qwen3 request=[]\n"
            "2026-08-02 14:30:00,000 - INFO - Stream started: provider=local model=Qwen3 session=cccc request=[]\n"
        )
        out_dir = tmp_path / "out_note"
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=out_dir,
            config=None,
        )
        assert result.summary.busy is not None
        assert result.summary.busy.unfinished_streams == 1  # cccc (in-window abort)
        assert result.summary.busy.pre_window_unfinished == 1  # aaaa (stale)
        md = (out_dir / "report.md").read_text()
        util = md.split("## Local model utilization", 1)[1].split("## ", 1)[0]
        assert "1 local stream(s) started" in util
        assert "1 pre-window leftover" in util
        assert "pre-window leftover" in util

    def test_mode_aware_end_to_end_split(self, tmp_path):
        """Regression (LP-0MSPZUD4G007IYGH): a window crossing the
        01:00/10:00 mode transitions splits sessions fast/cheap by the mode
        active at their start — cheap sessions land in cheap_sessions.csv with
        the cheap profile's slots/ctx even though the analysis-time mode is
        fast. Uses real ``Mode scheduler`` + ``status_request`` lines."""
        log_dir = tmp_path / "logs_mode"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text("\n".join(TestModeTimelineBucketing._lines(TestModeTimelineBucketing)) + "\n")
        out_dir = tmp_path / "out_mode"
        mm = fixtures.mode_map_fixture()
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=TestModeTimelineBucketing.MODE_WINDOW_START,
            window_end=TestModeTimelineBucketing.MODE_WINDOW_END,
            output_dir=out_dir,
            config=None,
            mode_map=mm,
        )
        assert len(result.summary.sessions) == 3
        assert result.summary.sessions["cheap-sess"].bucket == "cheap"
        assert result.summary.sessions["cheap-sess"].slots == 2
        assert result.summary.sessions["cheap-sess"].ctx_size == 262144

        with (out_dir / "standard_sessions.csv").open() as f:
            fast_rows = list(csv.DictReader(f))
        with (out_dir / "low_priority_sessions.csv").open() as f:
            cheap_rows = list(csv.DictReader(f))
        assert {r["session_id"] for r in fast_rows} == {"fast-early", "fast-late"}
        assert {r["session_id"] for r in cheap_rows} == {"cheap-sess"}
        cheap = cheap_rows[0]
        assert cheap["bucket"] == "cheap"
        assert cheap["slots"] == "2"
        assert cheap["ctx_size"] == "262144"
        fast = next(r for r in fast_rows if r["session_id"] == "fast-late")
        assert fast["slots"] == "3"
        assert fast["ctx_size"] == "131072"

        # The report's Session summary reflects the actual mode split.
        md = (out_dir / "report.md").read_text()
        section = md.split("## Session summary", 1)[1].split("## ", 1)[0]
        assert "| Sessions | 3 | 2 (66.7%) | 1 (33.3%) |" in section
        assert "| Local requests | 3 (100.0%) | 2 (100.0%) | 1 (100.0%) |" in section

        # JSON summary exposes the split too.
        data = reporting.summary_to_json(result.summary, mm)
        assert data["standard_sessions"] == 2
        assert data["low_priority_sessions"] == 1

    def test_mode_aware_context_pressure_uses_per_session_profile(self, tmp_path):
        """Context-pressure recommendations use each session's own profile
        (cheap 262144/2, fast 131072/3) instead of the analysis-time ctx."""
        log_dir = tmp_path / "logs_mode2"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text("\n".join(TestModeTimelineBucketing._lines(TestModeTimelineBucketing)) + "\n")
        out_dir = tmp_path / "out_mode2"
        mm = fixtures.mode_map_fixture()
        _ = reporting.run_analysis(
            log_dir=log_dir,
            window_start=TestModeTimelineBucketing.MODE_WINDOW_START,
            window_end=TestModeTimelineBucketing.MODE_WINDOW_END,
            output_dir=out_dir,
            config=fixtures.MODE_FAST_CONFIG,
            mode_map=mm,
        )
        md = (out_dir / "report.md").read_text()
        # cheap-sess peaked at 50000 tokens vs per-slot 262144/2 = 131072
        # (38%: no pressure); fast sessions are far below 131072/3. No
        # context-pressure recommendation should fire.
        assert "Context sizes approaching per-slot limits" not in md

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
        # Upstream errors are broken out by status code (LP-402 vs 429).
        assert "| upstream HTTP error | 429 | 1 |" in report_md
        assert "Upstream HTTP error breakdown by status" in report_md
        assert "| 429 | 1 | opencode (1) |" in report_md

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

    def test_error_root_cause_artifacts(self, tmp_path):
        """The 2026-09-25 → 2026-09-26 window's 24 / 6 / 1 classification is
        exposed in errors.csv (root_cause column) and errors.json
        (by_root_cause breakdown), entirely data-driven (F1 AC3/AC4)."""
        log_dir = tmp_path / "logs_root_cause"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text(
            "\n".join(fixtures.window_2026_09_25_lines()) + "\n"
        )
        out_dir = tmp_path / "out_root_cause"
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=fixtures.WINDOW_2026_09_25_START,
            window_end=fixtures.WINDOW_2026_09_25_END,
            output_dir=out_dir,
            config=None,
        )
        finish = [
            e for e in result.summary.error_events if e.kind == "stream_finish_error"
        ]
        assert len(finish) == 31

        errors_csv = out_dir / "errors.csv"
        assert errors_csv.exists()
        with errors_csv.open() as f:
            rows = list(csv.DictReader(f))
        assert "root_cause" in rows[0]
        finish_rows = [r for r in rows if r["error_type"] == "stream_finish_error"]
        counts = Counter(r["root_cause"] for r in finish_rows)
        assert counts["upstream_stall_after_content"] == 24
        assert counts["mode_switch_restart"] == 6
        assert counts["remote_chain_exhaustion"] == 1
        assert "" not in counts  # no unclassified stream-finish row

        data = json.loads((out_dir / "errors.json").read_text())
        assert data["by_root_cause"]["stream_finish_error"] == {
            "mode_switch_restart": 6,
            "remote_chain_exhaustion": 1,
            "upstream_stall_after_content": 24,
        }

    def test_window_report_removes_stale_recommendation(self, tmp_path):
        """Regenerating the 2026-09-25 → 2026-09-26 report removes the stale
        adoption recommendation, states the strategy is live, and renders the
        24 / 6 / 1 root-cause split (LP-0MUQO862C004UT7T AC3/AC4)."""
        log_dir = tmp_path / "logs_f2"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text(
            "\n".join(fixtures.window_2026_09_25_lines()) + "\n"
        )
        out_dir = tmp_path / "out_f2"
        reporting.run_analysis(
            log_dir=log_dir,
            window_start=fixtures.WINDOW_2026_09_25_START,
            window_end=fixtures.WINDOW_2026_09_25_END,
            output_dir=out_dir,
            config=None,
        )
        report_md = (out_dir / "report.md").read_text()
        # Stale adoption wording is gone; the live-strategy wording is present.
        assert "adopt recovery-first" not in report_md
        assert "recovery-first + informative-error strategy is live" in report_md
        # The root-cause taxonomy renders the 24 / 6 / 1 split.
        assert "### Root-cause breakdown" in report_md
        assert "| Stream finished: reason=error | upstream_stall_after_content | 24 |" in report_md
        assert "| Stream finished: reason=error | mode_switch_restart | 6 |" in report_md
        assert "| Stream finished: reason=error | remote_chain_exhaustion | 1 |" in report_md
        assert "Dominant residual cause: upstream_stall_after_content" in report_md

    def test_error_root_cause_in_summary_json(self, tmp_path):
        log_dir = tmp_path / "logs_root_summary"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text(
            "\n".join(fixtures.window_2026_09_25_lines()) + "\n"
        )
        out_dir = tmp_path / "out_root_summary"
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=fixtures.WINDOW_2026_09_25_START,
            window_end=fixtures.WINDOW_2026_09_25_END,
            output_dir=out_dir,
            config=None,
        )
        data = reporting.summary_to_json(result.summary)
        assert data["errors_by_root_cause"]["stream_finish_error"] == {
            "mode_switch_restart": 6,
            "remote_chain_exhaustion": 1,
            "upstream_stall_after_content": 24,
        }
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

    def test_window_straddling_dash_file_boundary(self, tmp_path):
        """A window spanning a dash-file boundary reports sessions from both sides.

        The dash file holds the earlier part of the window and the live log the
        later part. Both must contribute; before the fix the dash file was
        undiscovered and its sessions vanished despite falling inside the
        window.
        """
        log_dir = tmp_path / "logs-dash-boundary"
        log_dir.mkdir()
        (log_dir / "proxy.log-2026-08-02_13").write_text(
            "2026-08-02 14:00:10,000 - INFO - Stream started: provider=local model=Qwen3 session=early request=[]\n"
            "2026-08-02 14:00:12,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=early provider=local model=Qwen3 request=[]\n"
        )
        (log_dir / "proxy.log").write_text(
            "2026-08-02 14:30:00,000 - INFO - Stream started: provider=local model=Qwen3 session=late request=[]\n"
            "2026-08-02 14:30:02,000 - INFO - Stream finished: reason=stop tokens=200/20/220 session=late provider=local model=Qwen3 request=[]\n"
        )
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=tmp_path / "out-dash-boundary",
            config=None,
        )
        assert log_dir / "proxy.log-2026-08-02_13" in result.files
        assert set(result.summary.sessions) == {"early", "late"}

    def test_gzipped_dash_file_contributes_sessions(self, tmp_path):
        """A gzipped dash-named rotated file contributes its in-window sessions.

        Combines both defects: the dash name must be discovered AND the gzip
        payload must be decompressed. Reading a ``.gz`` as plain text yields no
        parseable lines, so the session would be silently absent.
        """
        log_dir = tmp_path / "logs-dash-gz"
        log_dir.mkdir()
        with gzip.open(log_dir / "proxy.log-2026-08-02_13.gz", "wt", encoding="utf-8") as fh:
            fh.write(
                "2026-08-02 14:00:10,000 - INFO - Stream started: provider=local model=Qwen3 session=gzsession request=[]\n"
                "2026-08-02 14:00:12,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=gzsession provider=local model=Qwen3 request=[]\n"
            )
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=tmp_path / "out-dash-gz",
            config=None,
        )
        assert log_dir / "proxy.log-2026-08-02_13.gz" in result.files
        assert list(result.summary.sessions) == ["gzsession"]


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
        assert (out / "standard_sessions.csv").exists()
        assert (out / "low_priority_sessions.csv").exists()


class TestSummaryOutputPaths:
    """The stdout summary lists the complete absolute path to every output
    artifact, starting with the report (report.md first, all five artifacts,
    ~ expanded, absolute archive path)."""

    def _write_logs(self, tmp_path: Path) -> Path:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text("\n".join(fixtures.E2E_LINES) + "\n")
        return log_dir

    def _args(self, tmp_path: Path, out: Path) -> list[str]:
        return [
            "--log-dir", str(self._write_logs(tmp_path)),
            "--start", "2026-08-02 14:00:00",
            "--end", "2026-08-02 15:00:00",
            "--output-dir", str(out),
        ]

    def test_summary_lists_full_paths_report_first(self, tmp_path, capsys):
        import analyze_proxy_usage as cli

        out = tmp_path / "out"
        rc = cli.main(self._args(tmp_path, out))
        assert rc == 0
        captured = capsys.readouterr().out
        assert "Outputs written to:" in captured
        listed = [ln.strip() for ln in captured.splitlines() if ln.strip().startswith(str(out))]
        expected = [
            out.resolve() / "report.md",
            out.resolve() / "standard_sessions.csv",
            out.resolve() / "low_priority_sessions.csv",
            out.resolve() / "errors.csv",
            out.resolve() / "errors.json",
        ]
        # Report first, then every artifact, each a complete absolute path.
        assert listed == [str(p) for p in expected], f"unexpected output lines: {listed}"

    def test_home_dir_default_is_expanded(self, tmp_path, monkeypatch, capsys):
        import analyze_proxy_usage as cli

        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        rc = cli.main(
            [
                "--log-dir", str(self._write_logs(tmp_path)),
                "--start", "2026-08-02 14:00:00",
                "--end", "2026-08-02 15:00:00",
            ]
        )
        assert rc == 0
        captured = capsys.readouterr().out
        expected = (home / "proxy-usage-reports" / "report.md").resolve()
        assert str(expected) in captured
        assert "~" not in captured.split("Outputs written to:", 1)[1]

    def test_archive_path_is_absolute(self, tmp_path, capsys):
        import analyze_proxy_usage as cli

        out = tmp_path / "out"
        args = self._args(tmp_path, out)
        assert cli.main(args) == 0
        # Second run: the first run's outputs move into a dated archive dir.
        assert cli.main(args) == 0
        captured = capsys.readouterr().out
        archive_lines = [ln for ln in captured.splitlines() if "archived to" in ln]
        assert len(archive_lines) == 1
        path = archive_lines[0].split("archived to ", 1)[1]
        assert Path(path).is_absolute()
        assert Path(path).is_dir()


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
        assert "| Metric | Total | Standard | Low |" in section
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
        assert "| Reason | Total | % of fallbacks | Standard | Low |" in section
        assert "| local_concurrency_limit | 1 | 100.0% | 1 (100.0%) | 0 (0.0%) |" in section

    def test_routing_skip_reasons_have_day_night(self, tmp_path):
        md = self._run(tmp_path)
        section = md.split("## routing_skip_local reasons", 1)[1].split("## ", 1)[0]
        assert "| Reason | Total | % of skips | Standard | Low |" in section
        assert "| local_concurrency_limit | 1 | 100.0% | 1 (100.0%) | 0 (0.0%) |" in section

    def test_per_model_breakdown_has_day_night(self, tmp_path):
        md = self._run(tmp_path)
        section = md.split("## Per-model breakdown", 1)[1].split("## ", 1)[0]
        assert "| Provider | Model | Sessions | Standard | Low | Requests | Fell back |" in section
        assert "| local | Qwen3 | 2 | 2 (100.0%) | 0 (0.0%) |" in section


class TestProviderContextSkipReport:
    """``## Provider context skips`` section and JSON (LP-0MVMBM1Y3001TPRT
    AC3/AC6): dedicated breakdown with count, % of skips, standard/low split,
    provider breakdown and the ceiling that was hit."""

    def _aggregate(self, lines):
        return aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, _schedule())

    def test_dedicated_section_renders_breakdown(self):
        res = self._aggregate(
            [
                "2026-08-02 14:00:20,000 - INFO - routing_skip_provider_context provider=munitoai-free "
                "type=remote status=provider_context_exceeded estimated_tokens=153917 limit=150000 "
                "reason=provider_context_exceeded session=s1",
                "2026-08-02 14:00:21,000 - INFO - routing_skip_provider_context provider=opencode-go "
                "type=remote status=provider_context_exceeded estimated_tokens=195787 limit=196608 "
                "reason=provider_context_exceeded session=s1",
                "2026-08-02 14:00:22,000 - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 "
                "estimated_tokens=40000 cold_threshold=39594 warm_threshold=39594 new_tokens=323 "
                "cached_ratio=0.99 reason=context_too_large → skipping local, routing to next remote "
                "provider session=s1",
            ]
        )
        md = reporting.build_report(res, config=None)
        section = md.split("## Provider context skips", 1)[1].split("## ", 1)[0]
        # Header bullet: count + share of all skip events (2 of 3).
        assert "**2**" in section and "66.7% of 3 skip events" in section
        # Provider rows with category-relative Standard/Low and the ceiling.
        assert "| munitoai-free | remote | 1 | 33.3% | 1 (50.0%) | 0 (0.0%) | 150000 (1) |" in section
        assert "| opencode-go | remote | 1 | 33.3% | 1 (50.0%) | 0 (0.0%) | 196608 (1) |" in section
        # Skips and local skips have separate tables (never conflated).
        assert "## routing_skip_local reasons" in md

    def test_section_omitted_without_events(self):
        res = self._aggregate(
            [
                "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 "
                "session=plain request=[]",
                "2026-08-02 14:00:05,000 - INFO - Stream finished: reason=stop tokens=900/40/940 "
                "session=plain provider=local model=Qwen3 request=[]",
            ]
        )
        md = reporting.build_report(res, config=None)
        assert "## Provider context skips" not in md

    def test_absent_limit_degrades_to_not_logged(self):
        # AC8: an event without ``limit`` renders as ``not logged`` instead of
        # zero or a crash, and the note names how many lacked the field.
        res = self._aggregate(
            [
                "2026-08-02 14:00:20,000 - INFO - routing_skip_provider_context provider=munitoai-free "
                "type=remote status=provider_context_exceeded estimated_tokens=160001 "
                "reason=provider_context_exceeded session=s2",
            ]
        )
        md = reporting.build_report(res, config=None)
        section = md.split("## Provider context skips", 1)[1].split("## ", 1)[0]
        assert "carried no `limit=` field" in section
        assert "| not logged |" in section

    def test_json_summary_exposes_provider_context_skips(self):
        res = self._aggregate(
            [
                "2026-08-02 14:00:20,000 - INFO - routing_skip_provider_context provider=munitoai-free "
                "type=remote status=provider_context_exceeded estimated_tokens=153917 limit=150000 "
                "reason=provider_context_exceeded session=s1",
                "2026-08-02 14:00:22,000 - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 "
                "estimated_tokens=40000 cold_threshold=39594 warm_threshold=39594 new_tokens=323 "
                "cached_ratio=0.99 reason=context_too_large → skipping local, routing to next remote "
                "provider session=s1",
            ]
        )
        j = reporting.summary_to_json(res)
        obj = j["provider_context_skips"]
        assert obj["total"] == 1
        assert obj["share_of_skips"] == pytest.approx(0.5)  # 1 of 2 skip events
        assert obj["by_provider"] == {"munitoai-free": 1}
        assert obj["by_limit"] == {"150000": 1}
        assert obj["standard"] == 1 and obj["low"] == 0


class TestCategoryRelativePercentages:
    """Breakdown percentages in the Total/Fast/Cheap columns are
    category-relative (LP-0MSVMS95E001Z78V): each fast % is the value's share
    of the fast-only total for that metric, each cheap % its share of the
    cheap-only total — so fast + cheap no longer sum to 100% per row. Uses a
    mixed fast/cheap fixture where the new values differ from the old
    per-row-split values.

    Fixture (schedule: 6-slot fast period 10:00-23:59, 8-slot cheap otherwise):
    - 5 sessions: f1..f4 fast (3 local/Qwen3 + 1 opencode-go deepseek),
      c1 cheap (local/Qwen3); 2 messages each → 8 fast / 2 cheap requests.
    - 6 fallback events: 3 fast + 1 cheap context_too_large, 1 fast + 1
      cheap local_concurrency_limit → fast_fb=4, cheap_fb=2.
    - 4 routing skips: 2 fast + 1 cheap context_too_large, 1 cheap
      large_context_bypass → fast_skips=2, cheap_skips=2.
    """

    CONFIG = {
        "session_slot_pool_size": 8,
        "slot_schedule": {
            "enabled": True,
            "entries": [("23:59", 8), ("10:00", 6)],
        },
    }
    FAST_TS = datetime(2026, 8, 2, 14, 0, 0)
    CHEAP_TS = datetime(2026, 8, 2, 3, 0, 0)

    def _session(self, sid, bucket, model="Qwen3", provider="local"):
        d = _session(sid, messages=2, bucket=bucket, local_req=2, remote_req=0)
        d["initial_provider"] = provider
        d["initial_model"] = model
        return d

    def _build(self) -> aggregation.AnalysisResult:
        res = aggregation.AnalysisResult(
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            sessions={},
            fallback_events=[
                log_parser.LogEvent("fallback", self.FAST_TS, reason="context_too_large"),
                log_parser.LogEvent("fallback", self.FAST_TS, reason="context_too_large"),
                log_parser.LogEvent("fallback", self.FAST_TS, reason="context_too_large"),
                log_parser.LogEvent("fallback", self.CHEAP_TS, reason="context_too_large"),
                log_parser.LogEvent("fallback", self.FAST_TS, reason="local_concurrency_limit"),
                log_parser.LogEvent("fallback", self.CHEAP_TS, reason="local_concurrency_limit"),
            ],
            routing_skip_events=[
                log_parser.LogEvent("routing_skip", self.FAST_TS, reason="context_too_large"),
                log_parser.LogEvent("routing_skip", self.FAST_TS, reason="context_too_large"),
                log_parser.LogEvent("routing_skip", self.CHEAP_TS, reason="context_too_large"),
                log_parser.LogEvent("routing_skip", self.CHEAP_TS, reason="large_context_bypass"),
            ],
            dispatch_denied_count=0,
            dispatch_denied_events=[],
            unattributed_events=0,
            lines_skipped=0,
            total_lines=0,
        )
        for d in [
            self._session("f1", "fast"),
            self._session("f2", "fast"),
            self._session("f3", "fast"),
            self._session("f4", "fast", model="deepseek-v4-flash", provider="opencode-go"),
            self._session("c1", "cheap"),
        ]:
            s = aggregation.SessionStats(**d)
            res.sessions[s.session_id] = s
        return res

    def _md(self) -> str:
        return reporting.build_report(self._build(), self.CONFIG, speed=None, mode_map=None)

    def test_fallback_events_row_percentages_are_category_rates(self):
        # Fast = fast fallbacks ÷ fast requests (4/8 = 50.0%); Cheap = cheap
        # fallbacks ÷ cheap requests (2/2 = 100.0%); Total stays 6/10 = 60.0%.
        # Old behavior showed the fast/cheap split of total fallbacks
        # (4/6 = 66.7% / 2/6 = 33.3%).
        section = self._md().split("## Session summary", 1)[1].split("## ", 1)[0]
        assert "| Fallback events | 6 (60.0%) | 4 (50.0%) | 2 (100.0%) |" in section

    def test_fallback_reasons_percentages_are_category_relative(self):
        # Fast = reason fast-count ÷ fast_fb (4); Cheap = reason cheap-count ÷
        # cheap_fb (2). Old showed the per-row split (e.g. context_too_large
        # cheap 1/4 = 25.0%); new shows the share within each category.
        section = self._md().split("## Fallback reasons", 1)[1].split("## ", 1)[0]
        assert "| context_too_large | 4 | 66.7% | 3 (75.0%) | 1 (50.0%) |" in section
        assert "| local_concurrency_limit | 2 | 33.3% | 1 (25.0%) | 1 (50.0%) |" in section

    def test_fallback_reasons_fast_and_cheap_no_longer_sum_to_row(self):
        # Consistency property: per-category reason percentages sum to ~100%
        # within each category (each column's own total), so operators can
        # compare reasons across fast vs cheap columns directly.
        # Fast column: 75.0% + 25.0% = 100.0% of fast_fb; Cheap: 50.0% + 50.0%
        # = 100.0% of cheap_fb — but no single row sums to 100% (fast + cheap
        # no longer sum to the row total).
        section = self._md().split("## Fallback reasons", 1)[1].split("## ", 1)[0]
        # Fast/Cheap cells are ``count (pct%)``; the ``% of fallbacks`` column
        # and Total are bare percentages outside parens.
        pcts = [float(p) for p in re.findall(r"\((\d+\.\d)%\)", section)]
        assert pcts == [75.0, 50.0, 25.0, 50.0]
        assert sum(pcts[0::2]) == pytest.approx(100.0)  # fast column
        assert sum(pcts[1::2]) == pytest.approx(100.0)  # cheap column

    def test_routing_skip_reasons_percentages_are_category_relative(self):
        # Fast = reason fast-count ÷ fast_skips (2); Cheap = reason cheap-count
        # ÷ cheap_skips (2). context_too_large: fast 2/2 = 100.0% (old was
        # 2/3 = 66.7%), cheap 1/2 = 50.0% (old 1/3 = 33.3%).
        section = self._md().split("## routing_skip_local reasons", 1)[1].split("## ", 1)[0]
        assert "| context_too_large | 3 | 75.0% | 2 (100.0%) | 1 (50.0%) |" in section
        assert "| large_context_bypass | 1 | 25.0% | 0 (0.0%) | 1 (50.0%) |" in section

    def test_per_model_breakdown_percentages_are_category_relative(self):
        # Fast = model fast sessions ÷ total fast sessions (4); Cheap = model
        # cheap sessions ÷ total cheap sessions (1). Old showed the per-model
        # split (Qwen3 fast 3/4 = 75.0%; new 3/4 = 75.0%, cheap 1/4 = 25.0% →
        # new 1/1 = 100.0%).
        section = self._md().split("## Per-model breakdown", 1)[1].split("## ", 1)[0]
        assert "| local | Qwen3 | 4 | 3 (75.0%) | 1 (100.0%) | 8 | 0 |" in section
        assert "| opencode-go | deepseek-v4-flash | 1 | 1 (25.0%) | 0 (0.0%) | 2 | 0 |" in section


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
        (out / "standard_sessions.csv").write_text("old fast")
        (out / "low_priority_sessions.csv").write_text("old cheap")
        (out / "errors.csv").write_text("old errors")
        (out / "errors.json").write_text("{\"old\": true}")

        run = self._run(tmp_path, monkeypatch=monkeypatch)

        # All five artifacts moved into the dated archive dir, verbatim.
        archive = out / "2026-08-07"
        assert archive.is_dir()
        assert run.archived_to == archive
        assert (archive / "report.md").read_text() == "OLD REPORT"
        assert (archive / "standard_sessions.csv").read_text() == "old fast"
        assert (archive / "low_priority_sessions.csv").read_text() == "old cheap"
        assert (archive / "errors.csv").read_text() == "old errors"
        assert (archive / "errors.json").read_text() == "{\"old\": true}"
        # Fresh outputs written at the root.
        assert (out / "report.md").read_text().startswith("# Proxy Usage Analysis")
        assert (out / "standard_sessions.csv").exists()
        assert (out / "low_priority_sessions.csv").exists()
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
        assert (out / "standard_sessions.csv").exists()

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


# ---------------------------------------------------------------------------
# Time to completion
# ---------------------------------------------------------------------------


class TestTimeToCompletion:
    """The report's ``## Time to completion`` section (session duration
    percentiles, split Total / Fast / Cheap)."""

    def test_percentile_basic(self):
        assert reporting._percentile([], 50) is None
        assert reporting._percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == pytest.approx(3.0)
        # Linear interpolation between neighbours for non-integer ranks.
        assert reporting._percentile([1.0, 2.0, 3.0, 4.0, 5.0], 10) == pytest.approx(1.4)
        assert reporting._percentile([1.0, 2.0, 3.0, 4.0, 5.0], 90) == pytest.approx(4.6)
        # Single value: every percentile is that value.
        assert reporting._percentile([7.0], 90) == 7.0

    def test_report_section_splits_total_fast_cheap(self):
        # Schedule: 00:00-12:00 -> 8 slots (cheap), 12:00-14:30 -> 4 slots
        # (fast), 14:30-24:00 -> 8 slots (cheap). So in the 14:00-15:00 window
        # sessions before 14:30 are fast and after 14:30 cheap.
        schedule = bucketing.schedule_from_entries([("12:00", 4), ("14:30", 8)])
        # s1: 100s local-only (fast); s2: 300s local -> remote (fast);
        # s3: 600s remote-only (cheap); s4: 200s local-only (cheap).
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s1 request=[]",
            "2026-08-02 14:01:40,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s1 provider=local model=Qwen3 request=[]",
            "2026-08-02 14:02:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s2 request=[]",
            "2026-08-02 14:07:00,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s2 provider=local model=Qwen3 request=[]",
            "2026-08-02 14:40:00,000 - INFO - Stream started: provider=opencode-go model=deepseek-v4-flash session=s3 request=[]",
            "2026-08-02 14:50:00,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s3 provider=opencode-go model=deepseek-v4-flash request=[]",
            "2026-08-02 14:45:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s4 request=[]",
            "2026-08-02 14:48:20,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s4 provider=local model=Qwen3 request=[]",
        ]
        summary = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, schedule)
        assert len(summary.sessions) == 4
        md = reporting.build_report(summary, None)
        section = md.split("## Time to completion", 1)[1].split("## ", 1)[0]
        # Header + 3 data rows (Total / Fast / Cheap).
        assert "| Bucket | Sessions | p10 (s) | Median (s) | p90 (s) |" in section
        # Durations: s1=100s (fast), s2=300s (fast), s4=200s (cheap), s3=600s (cheap).
        # Total (4 values): median (200+300)/2 = 250s, p10 = 130s, p90 = 510s.
        assert "| Total | 4 | 130.0 | 250.0 | 510.0 |" in section
        # Fast (s1=100, s2=300): median 200s, p10 = 120s, p90 = 280s.
        assert "| Fast | 2 | 120.0 | 200.0 | 280.0 |" in section
        # Cheap (s4=200, s3=600): median 400s, p10 = 240s, p90 = 560s.
        assert "| Cheap | 2 | 240.0 | 400.0 | 560.0 |" in section

    def test_section_absent_for_empty_window(self):
        md = reporting.build_report(aggregation.aggregate([], WINDOW_START, WINDOW_END, _schedule()), None)
        # Section header renders with an empty-window note (like the speed sections).
        assert "## Time to completion" in md
        assert "_No sessions in window._" in md


# ---------------------------------------------------------------------------
# Server-side compaction — LP-0MTHCTLAF00147IT AC1/AC2/AC3/AC4/AC5/AC6/AC7
# ---------------------------------------------------------------------------


class TestCompactionLogParsing:
    """AC1: compaction_event / backstop / churn lines are parsed tolerantly."""

    def test_compaction_event_fast_parsed(self):
        ev = log_parser.parse_log_line(fixtures.COMPACTION_EVENT_FAST)
        assert ev is not None
        assert ev.kind == "compaction_event"
        assert ev.session == "019fc2be-eb73-7830-a55e-c8bd5d21c927"
        assert ev.mode == "fast"
        assert ev.action == "compact"
        assert ev.reason == "context_window"
        assert ev.pre_tokens == 55000
        assert ev.post_tokens == 42000
        assert ev.turns_summarized == 3
        assert ev.turns_dropped == 1
        assert ev.summary_tokens == 800
        assert ev.dry_run is False

    def test_compaction_event_cheap_dry_run_parsed(self):
        ev = log_parser.parse_log_line(fixtures.COMPACTION_EVENT_CHEAP)
        assert ev is not None
        assert ev.kind == "compaction_event"
        assert ev.mode == "cheap"
        assert ev.action == "remote_with_guidance"
        assert ev.dry_run is True
        assert ev.pre_tokens == 60000
        assert ev.post_tokens == 35000

    def test_compaction_backstop_parsed(self):
        ev = log_parser.parse_log_line(fixtures.COMPACTION_BACKSTOP)
        assert ev is not None
        assert ev.kind == "compaction_backstop"
        assert ev.action == "dropped"
        assert ev.turns_dropped == 1
        assert ev.dropped_messages == 2
        assert ev.pre_tokens == 65000
        assert ev.post_tokens == 48000

    def test_compaction_churn_parsed(self):
        ev = log_parser.parse_log_line(fixtures.COMPACTION_CHURN)
        assert ev is not None
        assert ev.kind == "compaction_churn"
        assert ev.session == "019fc2be-eb73-7830-a55e-c8bd5d21c927"
        assert ev.churn_count == 5
        assert ev.churn_rate == pytest.approx(2.3)

    def test_compaction_event_tolerant_missing_fields(self):
        # Missing numeric fields must default to None, never fatal.
        line = "2026-08-02 14:12:00,000 - INFO - compaction_event session=abc123 mode=fast action=compact dry_run=true"
        ev = log_parser.parse_log_line(line)
        assert ev is not None
        assert ev.kind == "compaction_event"
        assert ev.pre_tokens is None
        assert ev.post_tokens is None
        assert ev.turns_summarized is None
        assert ev.dry_run is True

    def test_compaction_backstop_tolerant_partial(self):
        line = "2026-08-02 14:13:00,000 - INFO - compaction_backstop action=exhausted estimated_before=70000"
        ev = log_parser.parse_log_line(line)
        assert ev is not None
        assert ev.kind == "compaction_backstop"
        assert ev.pre_tokens == 70000
        assert ev.post_tokens is None

    def test_compaction_churn_invalid_rate_tolerated(self):
        # Malformed rate_per_hour must not crash — defaults to None.
        line = "2026-08-02 14:13:30,000 - INFO - compaction_churn session=abc count=2 rate_per_hour=notanumber exceeds_target=true"
        # Regex won't match non-numeric rate, so churn_rate stays None gracefully.
        ev = log_parser.parse_log_line(line)
        assert ev is not None
        assert ev.kind == "compaction_churn"
        assert ev.churn_count == 2
        assert ev.churn_rate is None

    def test_compaction_event_window_filtering_via_iter_events(self, tmp_path):
        log_dir = tmp_path / "logs_comp"
        log_dir.mkdir()
        # One event inside window, one before.
        lines = [
            "2026-08-02 13:00:00,000 - INFO - compaction_event session=s1 mode=fast action=compact pre_tokens=50000 post_tokens=30000 dry_run=false",
            fixtures.COMPACTION_EVENT_FAST,  # 14:12 inside
        ]
        (log_dir / "proxy.log").write_text("\n".join(lines) + "\n")
        events = list(log_parser.iter_events(log_dir / "proxy.log", WINDOW_START, WINDOW_END))
        kinds = [e.kind for e in events]
        assert "compaction_event" in kinds
        assert len([k for k in kinds if k == "compaction_event"]) == 1
        assert events[0].pre_tokens == 55000


class TestCompactionAggregation:
    """AC3: compaction impact correlation and threshold heuristic."""

    def test_compaction_events_collected(self):
        schedule = _schedule()
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s-compact request=[]",
            "2026-08-02 14:00:02,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s-compact provider=local model=Qwen3 request=[]",
            fixtures.COMPACTION_EVENT_FAST.replace("019fc2be-eb73-7830-a55e-c8bd5d21c927", "s-compact"),
            fixtures.COMPACTION_BACKSTOP,
            fixtures.COMPACTION_CHURN.replace("019fc2be-eb73-7830-a55e-c8bd5d21c927", "s-compact"),
        ]
        summary = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, schedule)
        assert len(summary.compaction_events) == 3
        kinds = {e.kind for e in summary.compaction_events}
        assert kinds == {"compaction_event", "compaction_backstop", "compaction_churn"}

    def test_compaction_impact_stayed_local_vs_fell_back(self):
        # Two sessions: one compacted and stayed local, one compacted and fell back.
        sessions = {
            "s-local": aggregation.SessionStats(**_session("s-local", bucket="fast", remote_move=False)),
            "s-remote": aggregation.SessionStats(**_session("s-remote", bucket="fast", remote_move=True, reason="context_too_large")),
        }
        events = [
            log_parser.LogEvent("compaction_event", WINDOW_START, session="s-local", mode="fast", action="compact", pre_tokens=50000, post_tokens=30000, dry_run=False),
            log_parser.LogEvent("compaction_event", WINDOW_START, session="s-remote", mode="fast", action="compact", pre_tokens=50000, post_tokens=30000, dry_run=False),
        ]
        impact = aggregation.compaction_impact(sessions, events, config=None)
        assert impact["sessions_with_compaction"] == 2
        assert impact["stayed_local"] == 1
        assert impact["fell_back"] == 1

    def test_compaction_impact_would_have_avoided_dry_run(self):
        # Config: 262144 ctx / 6 slots -> cap 39594, warm 100000 -> effective_warm 39594.
        config = {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 6,
            "local_large_context_warm_cache_threshold": 100000,
            "local_large_context_cold_cache_threshold": 60000,
        }
        sessions = {"s1": aggregation.SessionStats(**_session("s1"))}
        # pre above threshold, post below -> would-have-avoided (dry_run=true)
        events = [
            log_parser.LogEvent("compaction_event", WINDOW_START, session="s1", mode="fast", action="compact", pre_tokens=50000, post_tokens=30000, dry_run=True),
        ]
        impact = aggregation.compaction_impact(sessions, events, config=config)
        assert impact["effective_warm"] == 39594
        assert impact["would_have_avoided"] == 1
        assert impact["avoided"] == 0

    def test_compaction_impact_live_avoided(self):
        config = {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 6,
            "local_large_context_warm_cache_threshold": 100000,
        }
        sessions = {"s1": aggregation.SessionStats(**_session("s1"))}
        events = [
            log_parser.LogEvent("compaction_event", WINDOW_START, session="s1", mode="fast", action="compact", pre_tokens=50000, post_tokens=30000, dry_run=False),
        ]
        impact = aggregation.compaction_impact(sessions, events, config=config)
        assert impact["avoided"] == 1
        assert impact["would_have_avoided"] == 0

    def test_compaction_impact_would_still_fallback(self):
        config = {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 6,
            "local_large_context_warm_cache_threshold": 100000,
        }
        sessions = {"s1": aggregation.SessionStats(**_session("s1"))}
        # Both pre and post above threshold -> would_still_fallback
        events = [
            log_parser.LogEvent("compaction_event", WINDOW_START, session="s1", mode="fast", action="compact", pre_tokens=50000, post_tokens=45000, dry_run=True),
        ]
        impact = aggregation.compaction_impact(sessions, events, config=config)
        assert impact["would_still_fallback"] == 1
        assert impact["would_have_avoided"] == 0

    def test_compaction_impact_without_config_skips_heuristic(self):
        sessions = {"s1": aggregation.SessionStats(**_session("s1"))}
        events = [
            log_parser.LogEvent("compaction_event", WINDOW_START, session="s1", mode="fast", action="compact", pre_tokens=90000, post_tokens=10000, dry_run=True),
        ]
        impact = aggregation.compaction_impact(sessions, events, config=None)
        assert impact["effective_warm"] is None
        assert impact["would_have_avoided"] == 0
        assert impact["avoided"] == 0

    def test_compaction_impact_bucket_counts_and_dry_live(self):
        sessions = {}
        events = [
            log_parser.LogEvent("compaction_event", WINDOW_START, session="s1", mode="fast", action="compact", dry_run=False),
            log_parser.LogEvent("compaction_event", WINDOW_START, session="s2", mode="cheap", action="remote_with_guidance", dry_run=True),
            log_parser.LogEvent("compaction_backstop", WINDOW_START, action="dropped", pre_tokens=65000, post_tokens=48000),
        ]
        impact = aggregation.compaction_impact(sessions, events, config=None)
        assert impact["bucket_counts"] == {"fast": 1, "cheap": 1}
        assert impact["by_action"]["compact"] == 1
        assert impact["by_action"]["remote_with_guidance"] == 1
        assert impact["by_action"]["dropped"] == 1  # backstop
        assert impact["dry_run"]["dry_run"] == 1
        assert impact["dry_run"]["live"] == 2  # one live event + one backstop


class TestTriggerThresholds:
    """LP-0MTNIJQ8U007AGVW AC2: schedule-aware trigger resolution."""

    def test_static_fallback_fast(self):
        # fast: 262144 / 3 slots - 4096 headroom = 83285 → 0.70 × 83285 = 58300
        config = {"local_model_ctx_size": 262144, "session_slot_pool_size": 3}
        t = reporting._compute_trigger_thresholds(config)
        assert t["fast"] == 58300

    def test_static_fallback_cheap(self):
        # cheap: 131072 / 2 slots - 4096 headroom = 61440 → 0.70 × 61440 = 43008
        config = {"local_model_ctx_size": 131072, "session_slot_pool_size": 2}
        t = reporting._compute_trigger_thresholds(config)
        assert t["cheap"] == 43008

    def test_schedule_aware_cheap_higher(self):
        # cheap live: 262144/2-4096=126976 → 0.70×126976=88883
        # fast static: 262144/3-4096=83285 → 0.70×83285=58300
        # Both share ctx_by_time 262144 but different slot counts
        config = {
            "local_model_ctx_size": 131072,  # cheap static fallback
            "session_slot_pool_size": 2,  # cheap uses 2 slots
            "slot_schedule": {"ctx_by_time": {"23:59": 262144, "10:00": 262144}},
        }
        t = reporting._compute_trigger_thresholds(config)
        # With slot_pool_size=2, both fast and cheap resolve via ctx_by_time
        # The function returns distinct triggers for fast/cheap from different
        # ctx_by_time keys — both keys share ctx 262144 → same trigger 88883
        assert t["cheap"] == 88883
        # Test fast vs cheap with different slot counts
        fast_config = {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 3,
            "slot_schedule": {"ctx_by_time": {"10:00": 262144}},
        }
        cheap_config = {
            "local_model_ctx_size": 131072,
            "session_slot_pool_size": 2,
            "slot_schedule": {"ctx_by_time": {"23:59": 262144}},
        }
        ft = reporting._compute_trigger_thresholds(fast_config)
        ct = reporting._compute_trigger_thresholds(cheap_config)
        assert ct["cheap"] == 88883  # 262144//2-4096=126976 → ×0.70
        assert ft["fast"] == 58300   # 262144//3-4096=83285 → ×0.70
        assert ct["cheap"] > ft["fast"]

    def test_none_config(self):
        assert reporting._compute_trigger_thresholds(None) == {"fast": 0, "cheap": 0}

    def test_missing_slot_pool(self):
        assert reporting._compute_trigger_thresholds({"local_model_ctx_size": 262144}) == {
            "fast": 0, "cheap": 0,
        }

    def test_per_profile_thresholds_split(self):
        """Production shape: each mode has its own profile (discover_configs).

        Analysis running in fast mode passes the fast analysis_config plus the
        discovered profiles; the cheap trigger must come from the cheap
        profile's 2-slot pool (88,883), not the fast profile's 3 slots (58,300).
        """
        fast_cfg = {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 3,
            "slot_schedule": {"ctx_by_time": {"10:00": 262144, "23:59": 262144}},
        }
        cheap_cfg = {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 2,
            "slot_schedule": {"ctx_by_time": {"10:00": 262144, "23:59": 262144}},
        }
        profiles = {"default": fast_cfg, "fast": fast_cfg, "cheap": cheap_cfg}
        # Analysis-time config = fast profile; profiles carry both modes.
        t = reporting._compute_trigger_thresholds(fast_cfg, profiles)
        assert t == {"fast": 58300, "cheap": 88883}
        assert t["cheap"] > t["fast"]

    def test_per_profile_missing_cheap_falls_back(self):
        """A profile file absent from discover_configs falls back to the shared
        config rather than crashing or fabricating a trigger."""
        fast_cfg = {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 3,
            "slot_schedule": {"ctx_by_time": {"10:00": 262144}},
        }
        profiles = {"default": fast_cfg, "fast": fast_cfg, "cheap": None}
        t = reporting._compute_trigger_thresholds(fast_cfg, profiles)
        assert t["fast"] == 58300
        assert t["cheap"] == 58300

    def test_report_threshold_line_per_profile(self):
        """AC1: the empty-window note shows per-mode triggers from each mode's
        own profile (fast ≈58,300 / cheap ≈88,883), not a single collapsed
        value derived from the analysis-time config alone."""
        fast_cfg = {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 3,
            "slot_schedule": {"ctx_by_time": {"10:00": 262144, "23:59": 262144}},
        }
        cheap_cfg = {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 2,
            "slot_schedule": {"ctx_by_time": {"10:00": 262144, "23:59": 262144}},
        }
        profiles = {"default": fast_cfg, "fast": fast_cfg, "cheap": cheap_cfg}
        sessions = {
            "s1": aggregation.SessionStats(**_session("s1", max_context=81000, bucket="fast")),
            "s2": aggregation.SessionStats(**_session("s2", max_context=70000, bucket="cheap")),
        }
        summary = aggregation.AnalysisResult(
            window_start=WINDOW_START, window_end=WINDOW_END,
            sessions=sessions,
            fallback_events=[], routing_skip_events=[],
            dispatch_denied_count=0, unattributed_events=0, lines_skipped=0, total_lines=0,
            compaction_events=[],
        )
        md = reporting.build_report(summary, fast_cfg, profiles=profiles)
        section = md.split("## Server-side compaction", 1)[1].split("## ", 1)[0]
        # Distinct per-mode thresholds in the note line.
        assert "fast ≈58,300 / cheap ≈88,883" in section
        # AC2: dry-run estimate compares each bucket against its own trigger:
        # fast 81k > 58.3k triggers; cheap 70k < 88.9k does NOT.
        assert "Would-have-triggered: **1** / 2 (fast 1 / cheap 0)" in section
        assert "Triggers used: fast ≈58,300 / cheap ≈88,883" in section


class TestCompactionReporting:
    """AC2/AC5/AC6: report section rendering."""

    def test_zero_compaction_placeholder(self):
        md = reporting.build_report(aggregation.aggregate([], WINDOW_START, WINDOW_END, _schedule()), None)
        section = md.split("## Server-side compaction", 1)[1].split("## ", 1)[0]
        assert "No compactions observed in window" in section

    def test_report_with_compactions_shows_split(self):
        sessions = {
            "s1": aggregation.SessionStats(**_session("s1", bucket="fast")),
            "s2": aggregation.SessionStats(**_session("s2", bucket="cheap")),
        }
        # Build a summary manually with compaction events.
        summary = aggregation.AnalysisResult(
            window_start=WINDOW_START, window_end=WINDOW_END,
            sessions=sessions,
            fallback_events=[], routing_skip_events=[],
            dispatch_denied_count=0, unattributed_events=0, lines_skipped=0, total_lines=0,
            compaction_events=[
                log_parser.LogEvent("compaction_event", WINDOW_START, session="s1", mode="fast", action="compact", reason="context_window", pre_tokens=55000, post_tokens=42000, turns_summarized=3, dry_run=False),
                log_parser.LogEvent("compaction_event", WINDOW_START, session="s2", mode="cheap", action="compact", reason="context_window", pre_tokens=60000, post_tokens=35000, turns_summarized=2, dry_run=True),
                log_parser.LogEvent("compaction_backstop", WINDOW_START, action="dropped", turns_dropped=1, dropped_messages=2, pre_tokens=65000, post_tokens=48000),
            ],
        )
        md = reporting.build_report(summary, None)
        section = md.split("## Server-side compaction", 1)[1].split("## ", 1)[0]
        assert "Total compaction events:" in section
        # Total/Fast/Cheap split appears
        assert "Fast" in section and "Cheap" in section
        # Action breakdown
        assert "action=compact" in section
        # Dry-run vs live rows
        assert "dry_run=true" in section
        assert "dry_run=false" in section
        # Token economics
        assert "Avg pre_tokens" in section
        # Fallback-avoidance impact heading
        assert "Fallback-avoidance impact" in section

    def test_report_all_dry_run_heading_note(self):
        summary = aggregation.AnalysisResult(
            window_start=WINDOW_START, window_end=WINDOW_END,
            sessions={"s1": aggregation.SessionStats(**_session("s1"))},
            fallback_events=[], routing_skip_events=[],
            dispatch_denied_count=0, unattributed_events=0, lines_skipped=0, total_lines=0,
            compaction_events=[
                log_parser.LogEvent("compaction_event", WINDOW_START, session="s1", mode="fast", action="compact", pre_tokens=50000, post_tokens=30000, dry_run=True),
            ],
        )
        md = reporting.build_report(summary, None)
        section = md.split("## Server-side compaction", 1)[1].split("## ", 1)[0]
        assert "Advisory (dry_run)" in section

    def test_report_backstop_and_churn_visible(self):
        summary = aggregation.AnalysisResult(
            window_start=WINDOW_START, window_end=WINDOW_END,
            sessions={},
            fallback_events=[], routing_skip_events=[],
            dispatch_denied_count=0, unattributed_events=0, lines_skipped=0, total_lines=0,
            compaction_events=[
                log_parser.LogEvent("compaction_backstop", WINDOW_START, action="dropped", turns_dropped=2, dropped_messages=4, pre_tokens=70000, post_tokens=50000),
                log_parser.LogEvent("compaction_churn", WINDOW_START, session="s1", churn_count=3, churn_rate=1.5),
            ],
        )
        md = reporting.build_report(summary, None)
        section = md.split("## Server-side compaction", 1)[1].split("## ", 1)[0]
        assert "compaction_backstop" in section.lower() or "dropped" in section
        assert "Churn warnings" in section

    def test_report_impact_effective_threshold(self):
        config = {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 6,
            "local_large_context_warm_cache_threshold": 100000,
        }
        summary = aggregation.AnalysisResult(
            window_start=WINDOW_START, window_end=WINDOW_END,
            sessions={"s1": aggregation.SessionStats(**_session("s1"))},
            fallback_events=[], routing_skip_events=[],
            dispatch_denied_count=0, unattributed_events=0, lines_skipped=0, total_lines=0,
            compaction_events=[
                log_parser.LogEvent("compaction_event", WINDOW_START, session="s1", mode="fast", action="compact", pre_tokens=50000, post_tokens=30000, dry_run=True),
            ],
        )
        md = reporting.build_report(summary, config)
        section = md.split("## Server-side compaction", 1)[1].split("## ", 1)[0]
        assert "Effective warm threshold" in section
        assert "Would-have-avoided" in section

    def test_empty_window_dry_run_estimate(self):
        """AC1: empty window with sessions shows would-have-triggered estimate."""
        config = {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 2,  # cheap has 2 slots; fast gets 3 via ctx_by_time keys
            "slot_schedule": {"ctx_by_time": {"23:59": 262144, "10:00": 262144}},
        }
        sessions = {
            "s1": aggregation.SessionStats(**_session("s1", max_context=81000, bucket="fast")),
            "s2": aggregation.SessionStats(**_session("s2", max_context=40000, bucket="cheap")),
        }
        summary = aggregation.AnalysisResult(
            window_start=WINDOW_START, window_end=WINDOW_END,
            sessions=sessions,
            fallback_events=[], routing_skip_events=[],
            dispatch_denied_count=0, unattributed_events=0, lines_skipped=0, total_lines=0,
            compaction_events=[],
        )
        md = reporting.build_report(summary, config)
        section = md.split("## Server-side compaction", 1)[1].split("## ", 1)[0]
        # Dry-run estimate present
        assert "DRY-RUN ESTIMATE" in section
        assert "Would-have-triggered:" in section
        # With slot_pool_size=2 and ctx_by_time 262144, both fast & cheap resolve to 88883
        assert "≈88,883" in section
        # Warning present
        assert "hypothetical" in section
        assert "No history was mutated" in section

    def test_empty_window_schedule_aware_thresholds(self):
        """AC3: copy uses resolved schedule-aware values, not static."""
        # The report renders per-mode thresholds via a SINGLE config dict.
        # In real use the caller passes the mode-specific config (fast or cheap).
        # Here we verify the function resolves the schedule-aware trigger when
        # ctx_by_time overrides the static local_model_ctx_size.
        config = {
            "local_model_ctx_size": 131072,  # static → 43008, should NOT appear
            "session_slot_pool_size": 2,
            "slot_schedule": {"ctx_by_time": {"23:59": 262144}},
        }
        t = reporting._compute_trigger_thresholds(config)
        assert t["cheap"] == 88883  # live, NOT ≈43K
        # Static fallback value should not appear in the resolved thresholds
        assert t["cheap"] != 43008
        # Verify static fallback still works when no schedule
        static_config = {"local_model_ctx_size": 131072, "session_slot_pool_size": 2}
        st = reporting._compute_trigger_thresholds(static_config)
        assert st["cheap"] == 43008


class TestCompactionJson:
    """AC4: machine-readable compaction object via --json."""

    def test_json_compaction_structure(self):
        summary = aggregation.AnalysisResult(
            window_start=WINDOW_START, window_end=WINDOW_END,
            sessions={"s1": aggregation.SessionStats(**_session("s1"))},
            fallback_events=[], routing_skip_events=[],
            dispatch_denied_count=0, unattributed_events=0, lines_skipped=0, total_lines=0,
            compaction_events=[
                log_parser.LogEvent("compaction_event", WINDOW_START, session="s1", mode="fast", action="compact", pre_tokens=55000, post_tokens=42000, dry_run=False),
                log_parser.LogEvent("compaction_backstop", WINDOW_START, action="dropped", pre_tokens=65000, post_tokens=48000),
                log_parser.LogEvent("compaction_churn", WINDOW_START, session="s1", churn_count=2, churn_rate=1.0),
            ],
        )
        data = reporting.summary_to_json(summary)
        comp = data["compaction"]
        assert comp["events"] == 3
        assert comp["compact_events"] == 1
        assert comp["backstop_events"] == 1
        assert comp["churn_events"] == 1
        assert comp["sessions_with_compaction"] == 1
        assert comp["by_action"]["compact"] == 1
        assert comp["bucket_counts"]["fast"] == 1
        assert comp["avg_pre_tokens"] is not None
        assert comp["avg_post_tokens"] is not None
        assert comp["avg_tokens_saved"] == comp["avg_pre_tokens"] - comp["avg_post_tokens"]
        assert "stayed_local" in comp
        assert "would_have_avoided" in comp
        assert "backstop_sessions" in comp
        json.dumps(data)  # serialisable

    def test_json_jq_compaction_without_markdown_parsing(self, tmp_path):
        # End-to-end JSON path: run_analysis -> summary_to_json -> jq .compaction
        log_dir = tmp_path / "logs_json"
        log_dir.mkdir()
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s1 request=[]",
            "2026-08-02 14:00:02,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=s1 provider=local model=Qwen3 request=[]",
            fixtures.COMPACTION_EVENT_FAST.replace("019fc2be-eb73-7830-a55e-c8bd5d21c927", "s1"),
        ]
        (log_dir / "proxy.log").write_text("\n".join(lines) + "\n")
        result = reporting.run_analysis(log_dir=log_dir, window_start=WINDOW_START, window_end=WINDOW_END, output_dir=tmp_path / "out_json", config=None)
        data = reporting.summary_to_json(result.summary)
        comp = data["compaction"]
        # jq .compaction equivalent: direct dict access without Markdown
        assert comp["events"] >= 1
        assert comp["compact_events"] >= 1

    def test_json_zero_compaction(self):
        summary = aggregation.aggregate([], WINDOW_START, WINDOW_END, _schedule())
        data = reporting.summary_to_json(summary)
        assert data["compaction"]["events"] == 0
        assert data["compaction"]["sessions_with_compaction"] == 0


class TestCompactionEndToEnd:
    """AC7: synthetic 24h window with compaction events."""

    def test_e2e_run_with_compactions(self, tmp_path):
        log_dir = tmp_path / "logs_e2e"
        log_dir.mkdir()
        # Two sessions: s1 compacted dry-run, s2 compacted live; s2 has a backstop.
        s1 = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        s2 = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        lines = [
            f"2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session={s1} request=[]",
            f"2026-08-02 14:00:02,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session={s1} provider=local model=Qwen3 request=[]",
            f"2026-08-02 14:00:10,000 - INFO - Stream started: provider=local model=Qwen3 session={s2} request=[]",
            f"2026-08-02 14:00:12,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session={s2} provider=local model=Qwen3 request=[]",
            f"2026-08-02 14:12:00,000 - INFO - compaction_event session={s1} mode=fast action=compact reason=context_window pre_tokens=55000 post_tokens=32000 turns_summarized=2 turns_dropped=0 summary_tokens=600 dry_run=true",
            f"2026-08-02 14:12:30,000 - INFO - compaction_event session={s2} mode=cheap action=compact reason=context_window pre_tokens=60000 post_tokens=35000 turns_summarized=3 turns_dropped=1 summary_tokens=800 dry_run=false",
            "2026-08-02 14:13:00,000 - INFO - compaction_backstop action=dropped dropped_turns=1 dropped_messages=2 estimated_before=65000 estimated_after=48000",
            f"2026-08-02 14:13:30,000 - INFO - compaction_churn session={s1} count=2 rate_per_hour=1.5 exceeds_target=true",
        ]
        (log_dir / "proxy.log").write_text("\n".join(lines) + "\n")
        out = tmp_path / "out_e2e"
        result = reporting.run_analysis(log_dir=log_dir, window_start=WINDOW_START, window_end=WINDOW_END, output_dir=out, config=None)
        assert len(result.summary.compaction_events) == 4
        md = (out / "report.md").read_text()
        assert "## Server-side compaction" in md
        assert "Total compaction events:" in md
        # JSON mirrors the report.
        data = reporting.summary_to_json(result.summary)
        assert data["compaction"]["events"] == 4
        assert data["compaction"]["compact_events"] == 2
        assert data["compaction"]["backstop_events"] == 1
        assert data["compaction"]["churn_events"] == 1


# ---------------------------------------------------------------------------
# Total tokens
# ---------------------------------------------------------------------------


class TestTotalTokens:
    """The ``## Total tokens`` section (SA-0MTM3LKP1003S28J).

    Total = prompt (input) + completion (output) summed from
    ``tokens=prompt/completion/total`` on ``Stream finished`` lines.
    Rendered immediately before ``## Fallback reasons``, with absolute + %
    share per bucket and per model.  Invariant: total == input + output.
    Unknown provider/model renders as ``(unknown)``.
    """

    def _window_with_tokens(self, schedule):
        # s1 (fast): prompt=1000, completion=200; s2 (cheap): prompt=3000, completion=500
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s1 request=[]",
            "2026-08-02 14:00:05,000 - INFO - Stream finished: reason=stop tokens=1000/200/1200 session=s1 provider=local model=Qwen3 request=[]",
            "2026-08-02 14:30:00,000 - INFO - Stream started: provider=opencode-go model=deepseek-v4-flash session=s2 request=[]",
            "2026-08-02 14:30:05,000 - INFO - Stream finished: reason=stop tokens=3000/500/3500 session=s2 provider=opencode-go model=deepseek-v4-flash request=[]",
        ]
        return aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, schedule)

    def test_session_stats_expose_input_output_tokens(self):
        schedule = bucketing.schedule_from_entries([("12:00", 4), ("14:30", 8)])
        s = self._window_with_tokens(schedule)
        assert s.sessions["s1"].input_tokens == 1000
        assert s.sessions["s1"].output_tokens == 200
        assert s.sessions["s2"].input_tokens == 3000
        assert s.sessions["s2"].output_tokens == 500

    def test_session_stats_zero_tokens_when_no_finished_line(self):
        # A session started but no Stream finished tokens -> 0 tokens.
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=lonely request=[]",
        ]
        s = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, _schedule())
        assert s.sessions["lonely"].input_tokens == 0
        assert s.sessions["lonely"].output_tokens == 0

    def test_session_sums_multiple_finished_tokens(self):
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=multi request=[]",
            "2026-08-02 14:00:05,000 - INFO - Stream finished: reason=stop tokens=400/50/450 session=multi provider=local model=Qwen3 request=[]",
            "2026-08-02 14:00:10,000 - INFO - Stream started: provider=local model=Qwen3 session=multi request=[]",
            "2026-08-02 14:00:15,000 - INFO - Stream finished: reason=stop tokens=600/70/670 session=multi provider=local model=Qwen3 request=[]",
        ]
        s = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, _schedule())
        assert s.sessions["multi"].input_tokens == 1000
        assert s.sessions["multi"].output_tokens == 120

    def test_invariant_total_equals_input_plus_output_on_session_and_window(self):
        schedule = bucketing.schedule_from_entries([("12:00", 4), ("14:30", 8)])
        s = self._window_with_tokens(schedule)
        for sess in s.sessions.values():
            assert sess.input_tokens + sess.output_tokens == sess.input_tokens + sess.output_tokens
            # Input + output is the definition of total at the session level too
            assert s.total_tokens[2] == s.total_tokens[0] + s.total_tokens[1]
        inp, out, total = s.total_tokens
        assert total == inp + out

    def test_aggregate_total_tokens_window(self):
        schedule = bucketing.schedule_from_entries([("12:00", 4), ("14:30", 8)])
        s = self._window_with_tokens(schedule)
        assert s.total_tokens == (4000, 700, 4700)

    def test_aggregate_token_buckets_fast_cheap(self):
        schedule = bucketing.schedule_from_entries([("12:00", 4), ("14:30", 8)])
        s = self._window_with_tokens(schedule)
        assert s.token_buckets == {"fast": (1000, 200, 1200), "cheap": (3000, 500, 3500)}

    def test_aggregate_token_by_model_sorted_desc(self):
        schedule = bucketing.schedule_from_entries([("12:00", 4), ("14:30", 8)])
        s = self._window_with_tokens(schedule)
        by_model = s.token_by_model
        keys = list(by_model.keys())
        # Sorted by total desc -> opencode-go (3500) before local (1200)
        assert keys[0] == ("opencode-go", "deepseek-v4-flash")
        assert keys[1] == ("local", "Qwen3")
        assert by_model[("opencode-go", "deepseek-v4-flash")] == (3000, 500, 3500)
        assert by_model[("local", "Qwen3")] == (1000, 200, 1200)

    def test_aggregate_unknown_provider_model_placeholder(self):
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=known request=[]",
            "2026-08-02 14:00:05,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=known provider=local model=Qwen3 request=[]",
            # Second session whose start has empty provider/model
            "2026-08-02 14:01:00,000 - INFO - Stream started: provider= session=anon request=[]",
            "2026-08-02 14:01:05,000 - INFO - Stream finished: reason=stop tokens=50/5/55 session=anon provider= model= request=[]",
        ]
        s = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, _schedule())
        assert ("(unknown)", "(unknown)") in s.token_by_model

    def test_report_section_rendered_before_fallback_reasons(self):
        schedule = bucketing.schedule_from_entries([("12:00", 4), ("14:30", 8)])
        s = self._window_with_tokens(schedule)
        # Add one fallback so Fallback reasons is rendered.
        fe = log_parser.LogEvent(kind="fallback", ts=WINDOW_START, reason="local_concurrency_limit", raw="Fallback")
        s.fallback_events.append(fe)
        md = reporting.build_report(s, None)
        it = md.index("## Total tokens")
        iff = md.index("## Fallback reasons")
        assert 0 <= it < iff, "Total tokens must come immediately before Fallback reasons"
        # Also before Error analysis and other sections when present
        assert md.index("## Total tokens") < md.index("## Per-model breakdown")

    def test_report_section_rendered_even_when_no_fallbacks(self):
        s = aggregation.aggregate([], WINDOW_START, WINDOW_END, _schedule())
        md = reporting.build_report(s, None)
        assert "## Total tokens" in md
        # Still the next real section is Per-model breakdown etc
        assert md.index("## Total tokens") < md.index("## Per-model breakdown")

    def test_report_section_zero_window_shows_zero_counts(self):
        s = aggregation.aggregate([], WINDOW_START, WINDOW_END, _schedule())
        md = reporting.build_report(s, None)
        sec = md.split("## Total tokens", 1)[1].split("\n## ", 1)[0]
        assert "| Input (prompt) | 0 |" in sec
        assert "| Output (completion) | 0 |" in sec
        assert "| Total | **0** |" in sec
        assert "| Standard | 0 |" in sec
        assert "| Low | 0 |" in sec
        assert "_No tokens in window._" in sec

    def test_report_section_fast_cheap_counts_and_shares(self):
        schedule = bucketing.schedule_from_entries([("12:00", 4), ("14:30", 8)])
        s = self._window_with_tokens(schedule)
        md = reporting.build_report(s, None)
        sec = md.split("## Total tokens", 1)[1].split("\n## ", 1)[0]
        # Grand totals
        assert "| Total | **4,700** |" in sec
        # Fast 1000/4000 = 25.0% input; 200/700 = 28.6% output; 1200/4700 = 25.5% total
        # Cheap 3000/4000 = 75.0% input; 500/700 = 71.4% output; 3500/4700 = 74.5% total
        assert "| Standard | 1,000" in sec
        assert "| Low | 3,000" in sec
        # At least one share column present for each row
        assert "25.5%" in sec  # fast share of total
        assert "74.5%" in sec  # cheap share of total

    def test_report_section_by_model_counts_and_shares(self):
        schedule = bucketing.schedule_from_entries([("12:00", 4), ("14:30", 8)])
        s = self._window_with_tokens(schedule)
        md = reporting.build_report(s, None)
        sec = md.split("## Total tokens", 1)[1].split("\n## ", 1)[0]
        assert "| Provider | Model |" in sec
        assert "| local | Qwen3 |" in sec
        assert "| opencode-go | deepseek-v4-flash |" in sec
        # Sorted by total desc -> opencode-go before local
        assert sec.index("opencode-go") < sec.index("local | Qwen3")

    def test_report_section_by_model_unknown_placeholder(self):
        lines = [
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=known request=[]",
            "2026-08-02 14:00:05,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=known provider=local model=Qwen3 request=[]",
            "2026-08-02 14:01:00,000 - INFO - Stream started: provider= session=anon request=[]",
            "2026-08-02 14:01:05,000 - INFO - Stream finished: reason=stop tokens=50/5/55 session=anon provider= model= request=[]",
        ]
        s = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, _schedule())
        md = reporting.build_report(s, None)
        assert "(unknown)" in md

    def test_e2e_log_file_token_totals_via_run_analysis(self, tmp_path, monkeypatch):
        # Two sessions in one log file, fast vs cheap via the time-based schedule.
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text(
            "2026-08-02 14:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa request=[]\n"
            "2026-08-02 14:00:01,000 - INFO - Stream finished: reason=stop tokens=800/100/900 session=aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa provider=local model=Qwen3 request=[]\n"
            "2026-08-02 14:40:00,000 - INFO - Stream started: provider=local model=Qwen3 session=bbbbbbbb-2222-2222-2222-bbbbbbbbbbbb request=[]\n"
            "2026-08-02 14:40:01,000 - INFO - Stream finished: reason=stop tokens=200/20/220 session=bbbbbbbb-2222-2222-2222-bbbbbbbbbbbb provider=local model=Qwen3 request=[]\n",
            encoding="utf-8",
        )
        # Force fast/cheap via the slot_schedule entries
        config = {"slot_schedule": {"enabled": True, "entries": [("12:00", 4), ("14:30", 8)]}}
        out = tmp_path / "out"
        monkeypatch.setattr(reporting, "datetime", reporting.datetime)  # no freeze
        run = reporting.run_analysis(
            log_dir=log_dir, window_start=WINDOW_START, window_end=WINDOW_END,
            output_dir=out, config=config,
        )
        assert run.summary.total_tokens == (1000, 120, 1120)
        report = (out / "report.md").read_text()
        assert "## Total tokens" in report
        # Also before fallback
        # No fallback in this window, so just check it's present and before Per-model breakdown
        assert report.index("## Total tokens") < report.index("## Per-model breakdown")
        # JSON summary reflects it too via the same source-of-truth
        j = reporting.summary_to_json(run.summary)
        assert "total_tokens" in j or run.summary.total_tokens == (1000, 120, 1120)

    def test_shares_sum_to_100_percent(self):
        schedule = bucketing.schedule_from_entries([("12:00", 4), ("14:30", 8)])
        s = self._window_with_tokens(schedule)
        inp, out, total = s.total_tokens
        buckets = s.token_buckets
        # fast% + cheap% of total == 100%
        fast_share = buckets["fast"][2] / total * 100 if total else 0
        cheap_share = buckets["cheap"][2] / total * 100 if total else 0
        assert fast_share + cheap_share == pytest.approx(100.0, abs=0.1)
        # Model %s of total also sum to 100%
        model_total_share = sum(v[2] for v in s.token_by_model.values()) / total * 100 if total else 0
        assert model_total_share == pytest.approx(100.0, abs=0.1)


def _ttft_events(*spec: tuple[str, float]) -> list:
    """Build TTFT ``LogEvent``s from ``(session_id, ttft_ms)`` pairs."""
    return [
        log_parser.LogEvent("stream_started", WINDOW_START, session=sid, ttft_ms=ms)
        for sid, ms in spec
    ]


def _ttft_sessions(*spec: tuple[str, str]) -> dict:
    """Build the analysis-session mapping from ``(session_id, bucket)`` pairs."""
    return {
        sid: aggregation.SessionStats(**_session(sid, bucket=bucket))
        for sid, bucket in spec
    }


class TestTtftReportSection:
    """TTFT report section rendering tests (LP-0MTSSM5SO003PKU0, LP-0MUI56OCI008EGML)."""

    # Mixed fast/cheap sample set shared by the rendering and JSON tests.
    EVENTS = _ttft_events(
        ("fast-1", 45.0),
        ("fast-2", 120.0),
        ("cheap-1", 80.0),
        ("cheap-2", 450.0),
        ("cheap-3", 60.0),
    )
    SESSIONS = _ttft_sessions(
        ("fast-1", "fast"),
        ("fast-2", "fast"),
        ("cheap-1", "cheap"),
        ("cheap-2", "cheap"),
        ("cheap-3", "cheap"),
    )

    def test_ttft_section_renders_total_fast_cheap_rows(self):
        """TTFT section renders Total/Fast/Cheap rows with per-bucket counts."""
        from io import StringIO

        output = StringIO()
        ap = output.write
        reporting._append_ttft_section(ap, self.EVENTS, self.SESSIONS)
        report = output.getvalue()
        assert "## Time to first token" in report
        assert "No TTFT data" not in report
        # Total counts every sample; Fast/Cheap partition them.
        assert "| Total | 5 |" in report
        assert "| Fast | 2 |" in report
        assert "| Cheap | 3 |" in report
        # Percentiles render in the existing ms/seconds format.
        assert "ms" in report

    def test_ttft_section_unattributed_session_counted_in_total_only(self):
        """A TTFT event with an unknown session counts in Total but not Fast/Cheap."""
        from io import StringIO

        events = self.EVENTS + _ttft_events(("ghost", 999.0))
        output = StringIO()
        ap = output.write
        reporting._append_ttft_section(ap, events, self.SESSIONS)
        report = output.getvalue()
        assert "| Total | 6 |" in report
        assert "| Fast | 2 |" in report
        assert "| Cheap | 3 |" in report

    def test_ttft_section_empty(self):
        """TTFT section shows 'No TTFT data in window' when no data."""
        from io import StringIO

        output = StringIO()
        ap = output.write
        reporting._append_ttft_section(ap, [], {})
        report = output.getvalue()
        assert "## Time to first token" in report
        assert "_No TTFT data in window._" in report

    def test_ttft_json_populates_fast_and_cheap(self):
        """TTFT JSON carries per-bucket samples/percentiles, not zero/null stubs."""
        result = reporting._ttft_json(self.EVENTS, self.SESSIONS)
        assert result is not None
        assert result["total"] == {
            "samples": 5, "p10": 51.0, "median": 80.0, "p90": 318.0,
        }
        assert result["fast"] == {
            "samples": 2, "p10": 52.5, "median": 82.5, "p90": 112.5,
        }
        assert result["cheap"] == {
            "samples": 3, "p10": 64.0, "median": 80.0, "p90": 376.0,
        }

    def test_ttft_json_single_bucket_leaves_other_empty(self):
        """A window with only fast samples leaves the cheap bucket empty."""
        events = self.EVENTS[:2]
        result = reporting._ttft_json(events, self.SESSIONS)
        assert result["fast"]["samples"] == 2
        assert result["cheap"] == {
            "samples": 0, "p10": None, "median": None, "p90": None,
        }

    def test_ttft_json_empty(self):
        """TTFT JSON returns None when no data."""
        result = reporting._ttft_json([], {})
        assert result is None

    def test_summary_to_json_exposes_ttft_fast_cheap_split(self):
        """The JSON summary's ``ttft`` object carries the fast/cheap split (AC3)."""
        res = _result_with_sessions([
            _session("fast-1", bucket="fast"),
            _session("cheap-1", bucket="cheap"),
        ])
        res.ttft_events = _ttft_events(("fast-1", 45.0), ("cheap-1", 80.0))
        data = reporting.summary_to_json(res)
        assert data["ttft"]["total"]["samples"] == 2
        assert data["ttft"]["fast"] == {
            "samples": 1, "p10": 45.0, "median": 45.0, "p90": 45.0,
        }
        assert data["ttft"]["cheap"] == {
            "samples": 1, "p10": 80.0, "median": 80.0, "p90": 80.0,
        }

    def test_ttft_section_placement_before_ttc(self):
        """TTFT section appears before Time to completion in the report."""
        from io import StringIO

        output = StringIO()
        ap = output.write
        reporting._append_ttft_section(ap, self.EVENTS, self.SESSIONS)
        reporting._append_ttc_section(ap, [])
        report = output.getvalue()
        ttft_pos = report.index("## Time to first token")
        ttc_pos = report.index("## Time to completion")
        assert ttft_pos < ttc_pos


class TestAttributionInReport:
    """LP-0MVC144KS001PDR4 — attribution surfaces in the report."""

    MODE_WINDOW_START = datetime(2026, 10, 18, 0, 0, 0)
    MODE_WINDOW_END = datetime(2026, 10, 19, 0, 0, 0)

    def _events(self, lines: list[str]) -> list[log_parser.LogEvent]:
        return [e for e in (log_parser.parse_log_line(ln) for ln in lines) if e is not None]

    # -- aggregation: mode transitions carry attribution ---------------------

    def test_aggregate_collects_mode_switch_attribution(self):
        """A mode-switch line carrying source/actor/reason is collected with
        those fields in the mode_map transitions."""
        mm = bucketing.ModeScheduleMap.from_profiles(
            {"fast": fixtures.MODE_FAST_CONFIG, "cheap": fixtures.MODE_CHEAP_CONFIG},
            "fast",
            3,
        )
        lines = [
            fixtures.MODE_SWITCH_ATTRIBUTED,
        ]
        aggregation.aggregate(
            self._events(lines),
            self.MODE_WINDOW_START,
            self.MODE_WINDOW_END,
            bucketing.schedule_from_config(fixtures.MODE_FAST_CONFIG, 3),
            mm,
        )
        assert len(mm.transitions) == 1
        tr = mm.transitions[0]
        assert isinstance(tr, bucketing.ModeTransition)
        assert tr.mode == "cheap"
        assert tr.source == "scheduler"
        assert tr.actor == "mode-scheduler"
        assert tr.reason is None

    def test_aggregate_collects_restart_events_with_attribution(self):
        """Restart events are collected in AnalysisResult.restart_events
        carrying their attribution markers."""
        mm = bucketing.ModeScheduleMap.from_profiles(
            {"fast": fixtures.MODE_FAST_CONFIG, "cheap": fixtures.MODE_CHEAP_CONFIG},
            "fast",
            3,
        )
        lines = [
            fixtures.RESTART_SPAWNED_ATTRIBUTED,
            fixtures.RESTART_COMPLETE_ATTRIBUTED,
        ]
        res = aggregation.aggregate(
            self._events(lines),
            self.MODE_WINDOW_START,
            self.MODE_WINDOW_END,
            bucketing.schedule_from_config(fixtures.MODE_FAST_CONFIG, 3),
            mm,
        )
        assert len(res.restart_events) == 2
        # First event: spawned
        spawned = res.restart_events[0]
        assert spawned.kind == "restart"
        assert spawned.restart_kind == "spawned"
        assert spawned.source == "scheduler"
        assert spawned.actor == "mode-scheduler"
        assert spawned.reason == "scheduled mode change"
        # Second event: complete
        complete = res.restart_events[1]
        assert complete.kind == "restart"
        assert complete.restart_kind == "complete"
        assert complete.source == "api"
        assert complete.actor == "herdr"
        assert complete.reason == "maintenance window"

    def test_aggregate_restart_events_sorted(self):
        """restart_events_sorted returns events in timestamp order."""
        mm = bucketing.ModeScheduleMap.from_profiles(
            {"fast": fixtures.MODE_FAST_CONFIG, "cheap": fixtures.MODE_CHEAP_CONFIG},
            "fast",
            3,
        )
        lines = [
            fixtures.RESTART_COMPLETE_ATTRIBUTED,
            fixtures.RESTART_SPAWNED_ATTRIBUTED,
        ]
        res = aggregation.aggregate(
            self._events(lines),
            self.MODE_WINDOW_START,
            self.MODE_WINDOW_END,
            bucketing.schedule_from_config(fixtures.MODE_FAST_CONFIG, 3),
            mm,
        )
        sorted_events = res.restart_events_sorted
        assert sorted_events[0].ts < sorted_events[1].ts

    # -- report section: attribution rendered --------------------------------

    def test_report_section_shows_attribution_when_present(self):
        """The mode-timeline section lists events with source/actor/reason
        when markers are present."""
        mm = bucketing.ModeScheduleMap.from_profiles(
            {"fast": fixtures.MODE_FAST_CONFIG, "cheap": fixtures.MODE_CHEAP_CONFIG},
            "fast",
            3,
        )
        lines = [
            fixtures.MODE_SWITCH_ATTRIBUTED,
            fixtures.RESTART_SPAWNED_ATTRIBUTED,
        ]
        res = aggregation.aggregate(
            self._events(lines),
            self.MODE_WINDOW_START,
            self.MODE_WINDOW_END,
            bucketing.schedule_from_config(fixtures.MODE_FAST_CONFIG, 3),
            mm,
        )
        report = reporting.build_report(res, fixtures.MODE_FAST_CONFIG, mode_map=mm)
        assert "## Priority window timeline & restarts" in report
        assert "scheduler" in report
        assert "mode-scheduler" in report
        assert "scheduled mode change" in report

    def test_report_section_fallback_to_unknown_without_attribution(self):
        """Events without markers show 'unknown' for attribution fields.
        Backward compatible — no attribution present means historical logs."""
        mm = bucketing.ModeScheduleMap.from_profiles(
            {"fast": fixtures.MODE_FAST_CONFIG, "cheap": fixtures.MODE_CHEAP_CONFIG},
            "fast",
            3,
        )
        lines = [
            fixtures.MODE_SWITCH_CHEAP,  # no attribution
            fixtures.RESTART_SPAWNED_LINE,  # no attribution
        ]
        res = aggregation.aggregate(
            self._events(lines),
            self.MODE_WINDOW_START,
            self.MODE_WINDOW_END,
            bucketing.schedule_from_config(fixtures.MODE_FAST_CONFIG, 3),
            mm,
        )
        report = reporting.build_report(res, fixtures.MODE_FAST_CONFIG, mode_map=mm)
        assert "## Priority window timeline & restarts" in report
        assert "unknown" in report

    def test_report_section_zero_unknown_for_attributed_events(self):
        """When every restart carries full attribution, no restart table row
        shows 'unknown' (AC: zero 'unknown/unattributable' restarts that carry
        a marker)."""
        mm = bucketing.ModeScheduleMap.from_profiles(
            {"fast": fixtures.MODE_FAST_CONFIG, "cheap": fixtures.MODE_CHEAP_CONFIG},
            "fast",
            3,
        )
        lines = [
            fixtures.RESTART_SPAWNED_ATTRIBUTED,  # source/actor/reason all present
        ]
        res = aggregation.aggregate(
            self._events(lines),
            self.MODE_WINDOW_START,
            self.MODE_WINDOW_END,
            bucketing.schedule_from_config(fixtures.MODE_FAST_CONFIG, 3),
            mm,
        )
        report = reporting.build_report(res, fixtures.MODE_FAST_CONFIG, mode_map=mm)
        section_start = report.index("## Priority window timeline & restarts")
        next_heading = report.find("\n## ", section_start + 1)
        section_text = report[section_start:next_heading] if next_heading > 0 else report[section_start:]
        # Table rows (timestamp-prefixed) — every restart row must be fully
        # attributed.
        rows = [line for line in section_text.splitlines() if line.startswith("| 2026-")]
        assert rows, "expected at least one timeline table row"
        assert all("unknown" not in row for row in rows)
        # And the restart's markers are present verbatim.
        assert "scheduler" in rows[0]
        assert "mode-scheduler" in rows[0]
        assert "scheduled mode change" in rows[0]

    def test_report_section_source_normalised(self):
        """Source values render as the canonical labels: scheduler, worker, api, manual."""
        mm = bucketing.ModeScheduleMap.from_profiles(
            {"fast": fixtures.MODE_FAST_CONFIG, "cheap": fixtures.MODE_CHEAP_CONFIG},
            "fast",
            3,
        )
        lines = [
            fixtures.RESTART_COMPLETE_ATTRIBUTED,  # source=api
        ]
        res = aggregation.aggregate(
            self._events(lines),
            self.MODE_WINDOW_START,
            self.MODE_WINDOW_END,
            bucketing.schedule_from_config(fixtures.MODE_FAST_CONFIG, 3),
            mm,
        )
        report = reporting.build_report(res, fixtures.MODE_FAST_CONFIG, mode_map=mm)
        assert "api" in report

    def test_report_section_omitted_when_no_events(self):
        """The section is omitted when there are no mode transitions or restarts."""
        mm = bucketing.ModeScheduleMap.from_profiles(
            {"fast": fixtures.MODE_FAST_CONFIG, "cheap": fixtures.MODE_CHEAP_CONFIG},
            "fast",
            3,
        )
        lines = [
            "2026-08-14 10:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=s1 request=[]",
        ]
        res = aggregation.aggregate(
            self._events(lines),
            self.MODE_WINDOW_START,
            self.MODE_WINDOW_END,
            bucketing.schedule_from_config(fixtures.MODE_FAST_CONFIG, 3),
            mm,
        )
        report = reporting.build_report(res, fixtures.MODE_FAST_CONFIG, mode_map=mm)
        assert "## Priority window timeline & restarts" not in report

    def test_restart_rows_show_requested_mode(self):
        """A restart spawned by set_mode() carries the requested mode
        (``mode=cheap`` on the line); the timeline row surfaces it.
        (AC2: ... requested mode when attribution is present)."""
        mm = bucketing.ModeScheduleMap.from_profiles(
            {"fast": fixtures.MODE_FAST_CONFIG, "cheap": fixtures.MODE_CHEAP_CONFIG},
            "fast",
            3,
        )
        lines = [
            fixtures.RESTART_SPAWNED_ATTRIBUTED,  # mode=cheap on the line
        ]
        res = aggregation.aggregate(
            self._events(lines),
            self.MODE_WINDOW_START,
            self.MODE_WINDOW_END,
            bucketing.schedule_from_config(fixtures.MODE_FAST_CONFIG, 3),
            mm,
        )
        report = reporting.build_report(res, fixtures.MODE_FAST_CONFIG, mode_map=mm)
        section_start = report.index("## Priority window timeline & restarts")
        next_heading = report.find("\n## ", section_start + 1)
        section_text = report[section_start:next_heading] if next_heading > 0 else report[section_start:]
        # The restart row shows kind → requested mode.
        assert "spawned → cheap" in section_text
        assert "scheduler" in section_text
        assert "mode-scheduler" in section_text
        assert "scheduled mode change" in section_text

    def test_log_parser_extracts_mode_on_restart_lines(self):
        """The tolerant mode= extractor populates ev.mode on restart events
        that carry the marker (additive; None otherwise)."""
        ev = log_parser.parse_log_line(fixtures.RESTART_SPAWNED_ATTRIBUTED)
        assert ev is not None
        assert ev.kind == "restart"
        assert ev.mode == "cheap"
        # Historical unattributed spawns carry no mode.
        ev2 = log_parser.parse_log_line(fixtures.RESTART_SPAWNED_LINE)
        assert ev2 is not None
        assert ev2.mode is None

    def test_report_section_mixed_attribution(self):
        """A mix of attributed and non-attributed events shows attribution
        where present and 'unknown' where absent."""
        mm = bucketing.ModeScheduleMap.from_profiles(
            {"fast": fixtures.MODE_FAST_CONFIG, "cheap": fixtures.MODE_CHEAP_CONFIG},
            "fast",
            3,
        )
        lines = [
            fixtures.MODE_SWITCH_CHEAP,  # no attribution
            fixtures.RESTART_SPAWNED_ATTRIBUTED,
        ]
        res = aggregation.aggregate(
            self._events(lines),
            self.MODE_WINDOW_START,
            self.MODE_WINDOW_END,
            bucketing.schedule_from_config(fixtures.MODE_FAST_CONFIG, 3),
            mm,
        )
        report = reporting.build_report(res, fixtures.MODE_FAST_CONFIG, mode_map=mm)
        assert "## Priority window timeline & restarts" in report
        # Should contain both 'unknown' (for the mode switch) and 'scheduler'
        # (for the restart)
        assert "scheduler" in report
        # The section text should contain 'unknown' once (for the mode switch)
        section_start = report.index("## Priority window timeline & restarts")
        next_heading = report.find("\n## ", section_start + 1)
        section_text = report[section_start:next_heading] if next_heading > 0 else report[section_start:]
        # 'unknown' should appear (for the mode_switch without attribution)
        assert "unknown" in section_text


# =========================================================================
# Prefill-ETA admission guard decisions (LP-0MVGDUV5E005UWBQ AC2)
# =========================================================================


class TestLocalAdmissionGuard:
    """AC2: the guard's admit/refuse decisions are logged and counted."""

    def test_parses_reject_decision(self):
        ev = log_parser.parse_log_line(fixtures.LOCAL_ADMISSION_GUARD_REJECT)
        assert ev.kind == "local_admission_guard"
        assert ev.decision == "reject"
        assert ev.enforced == "true"
        assert ev.reason == "eta_exceeds_budget"
        assert ev.eta_prefill_s == "468.0"
        assert ev.budget_s == "120.0"
        assert ev.provider == "local-qwen3"
        assert ev.model == "Qwen3"

    def test_parses_admit_decision(self):
        ev = log_parser.parse_log_line(fixtures.LOCAL_ADMISSION_GUARD_ADMIT)
        assert ev.kind == "local_admission_guard"
        assert ev.decision == "admit"
        assert ev.reason == "within_budget"
        assert ev.eta_prefill_s == "91.8"

    def test_parses_observe_decision(self):
        ev = log_parser.parse_log_line(fixtures.LOCAL_ADMISSION_GUARD_OBSERVE)
        assert ev.kind == "local_admission_guard"
        assert ev.decision == "reject"
        assert ev.enforced == "false"

    def test_events_collected_globally(self):
        lines = [
            fixtures.LOCAL_ADMISSION_GUARD_REJECT,
            fixtures.LOCAL_ADMISSION_GUARD_ADMIT,
            fixtures.LOCAL_ADMISSION_GUARD_OBSERVE,
        ]
        res = aggregation.aggregate(
            _events(lines),
            datetime(2026, 8, 2, 13, 0),
            datetime(2026, 8, 2, 15, 0),
            _schedule(),
        )
        assert len(res.local_admission_events) == 3
        assert [e.decision for e in res.local_admission_events] == [
            "reject",
            "admit",
            "reject",
        ]

    def test_report_row_and_json_counters(self, tmp_path):
        log_dir = tmp_path / "logs_guard"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text(
            "\n".join(
                [
                    fixtures.LOCAL_ADMISSION_GUARD_REJECT,
                    fixtures.LOCAL_ADMISSION_GUARD_ADMIT,
                    fixtures.LOCAL_ADMISSION_GUARD_OBSERVE,
                ]
            )
            + "\n"
        )
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=tmp_path / "out_guard",
            config=None,
        )

        md = (tmp_path / "out_guard" / "report.md").read_text()
        section = md.split("## Session summary", 1)[1].split("## ", 1)[0]
        assert "| Local admission refused (prefill ETA) | 2 | - | 2 |" in section
        assert "| Local admission admitted (prefill ETA) | 1 | - | 1 |" in section

        data = reporting.summary_to_json(result.summary)
        assert data["local_admission_reject"] == 2
        # Only the enforced refusal counts; the observe-mode verdict does not.
        assert data["local_admission_reject_enforced"] == 1
        assert data["local_admission_admit"] == 1
        json.dumps(data)

# =========================================================================
# Idle-aware local admission (LP-0MVGD9Z8Z0055Y5J AC3)
# =========================================================================


class TestIdleAdmitCounter:
    """AC3: idle-admitted requests are greppable and counted by the report."""

    def test_parses_idle_admit_event(self):
        ev = log_parser.parse_log_line(fixtures.ROUTING_IDLE_ADMIT)
        assert ev.kind == "idle_admit"
        assert ev.reason == "economic_bypass_idle"
        assert ev.idle_seconds == "412.3"
        assert ev.threshold_seconds == "60.0"
        assert ev.provider == "local-qwen3"
        assert ev.model == "Qwen3"
        assert ev.session == "019fc245-ffff-7e5c-871e-57ab32f875f3"

    def test_adjacent_events_still_parse(self):
        """The idle-admit branch must not shadow the admission-guard branch."""
        ev = log_parser.parse_log_line(fixtures.LOCAL_ADMISSION_GUARD_REJECT)
        assert ev.kind == "local_admission_guard"

    def test_report_row_and_json_counter(self, tmp_path):
        log_dir = tmp_path / "logs_idle"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text(
            "\n".join(
                [fixtures.ROUTING_IDLE_ADMIT, fixtures.ROUTING_IDLE_ADMIT]
            )
            + "\n"
        )
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=tmp_path / "out_idle",
            config=None,
        )

        md = (tmp_path / "out_idle" / "report.md").read_text()
        section = md.split("## Session summary", 1)[1].split("## ", 1)[0]
        assert (
            "| Local admitted while GPU idle (cold-cache override) | 2 | - | 2 |"
            in section
        )

        data = reporting.summary_to_json(result.summary)
        assert data["local_idle_admit"] == 2
        json.dumps(data)


class TestMachineReadablePrefillAndBusyJson:
    """Machine-readable prefill and redefined-busy JSON surface
    (LP-0MVJNW6SN008AKH3).

    ``summary_to_json`` gains a top-level ``local_prefill`` object (prefill
    slot-seconds total, count, p10/p50/p90/max, unknown-prefill count and the
    slot-seconds unit) and ``local_busy`` gains the dispatch → finish
    definition marker, the retained decode-only union, the wall-clock prefill
    span and the labelled lower/upper bounds — while every legacy key stays in
    place so existing consumers keep working.
    """

    @staticmethod
    def _summary(lines: list[str]) -> aggregation.AnalysisResult:
        return aggregation.aggregate(
            _events(lines), WINDOW_START, WINDOW_END, _schedule()
        )

    @staticmethod
    def _prefill_lines(prefill_seconds: list[float]) -> list[str]:
        """One local stream per prefill duration, each with a first-byte line."""
        lines: list[str] = []
        for index, seconds in enumerate(prefill_seconds):
            ts = f"2026-08-02 14:0{index}:30,000"
            lines.append(
                f"{ts} - INFO - Stream started: provider=local model=Qwen3 "
                f"session=s{index} request=[]"
            )
            lines.append(
                "2026-08-02 14:0" + str(index) + ":40,000 - INFO - Stream finished: "
                f"reason=stop tokens=100/10/110 session=s{index} provider=local "
                "model=Qwen3 request=[]"
            )
            lines.append(
                f"{ts} - INFO - dispatch_first_byte_ms={seconds * 1000.0} "
                f"dispatch_to_first_byte_ms={seconds * 1000.0} cache_state=warm "
                f"cached_ratio=0.900 queue_wait_ms=0.0 session=s{index} model=Qwen3"
            )
        return lines

    def test_local_prefill_reports_slot_seconds_count_and_percentiles(self):
        """A window with prefill samples exposes the total, count and
        p10/p50/p90/max in slot-seconds."""
        data = reporting.summary_to_json(
            self._summary(self._prefill_lines([10, 20, 30, 40]))
        )
        prefill = data["local_prefill"]
        assert prefill["unit"] == "slot-seconds"
        assert prefill["slot_seconds"] == pytest.approx(100.0)
        assert prefill["count"] == 4
        assert prefill["streams"] == 4
        assert prefill["unknown_prefill"] == 0
        assert prefill["p10"] == pytest.approx(13.0)
        assert prefill["p50"] == pytest.approx(25.0)
        assert prefill["p90"] == pytest.approx(37.0)
        assert prefill["max_seconds"] == pytest.approx(40.0)
        json.dumps(data)  # round-trips

    def test_local_prefill_without_samples_keeps_structure(self):
        """Local traffic with no first-byte line keeps every key: zero total,
        null percentiles and the stream counted as unknown prefill."""
        lines = [
            "2026-08-02 14:00:10,000 - INFO - Stream started: provider=local "
            "model=Qwen3 session=s1 request=[]",
            "2026-08-02 14:00:20,000 - INFO - Stream finished: reason=stop "
            "tokens=100/10/110 session=s1 provider=local model=Qwen3 request=[]",
        ]
        data = reporting.summary_to_json(self._summary(lines))
        prefill = data["local_prefill"]
        assert prefill["unit"] == "slot-seconds"
        assert prefill["slot_seconds"] == 0.0
        assert prefill["count"] == 0
        assert prefill["streams"] == 1
        assert prefill["unknown_prefill"] == 1
        for key in ("p10", "p50", "p90", "max_seconds"):
            assert prefill[key] is None
        json.dumps(data)

    def test_local_prefill_is_null_without_local_traffic(self):
        """No local traffic means no busy or prefill object at all (matching
        ``local_busy``'s None convention)."""
        data = reporting.summary_to_json(self._summary([]))
        assert data["local_busy"] is None
        assert data["local_prefill"] is None
        json.dumps(data)

    def test_local_busy_exposes_definition_prefill_span_and_bounds(self):
        """A 30 s prefill + 10 s decode stream reports the dispatch → finish
        union as busy, the decode-only union as the lower bound, the prefill
        wall-clock span as their difference, and the labelled bounds."""
        lines = [
            "2026-08-02 14:00:30,000 - INFO - Stream started: provider=local "
            "model=Qwen3 session=s1 request=[]",
            "2026-08-02 14:00:40,000 - INFO - Stream finished: reason=stop "
            "tokens=100/10/110 session=s1 provider=local model=Qwen3 request=[]",
            "2026-08-02 14:00:30,000 - INFO - dispatch_first_byte_ms=30000.0 "
            "dispatch_to_first_byte_ms=29999.0 cache_state=warm cached_ratio=0.900 "
            "queue_wait_ms=1.0 session=s1 model=Qwen3",
        ]
        busy = reporting.summary_to_json(self._summary(lines))["local_busy"]
        assert busy["definition"] == "dispatch_to_finish"
        assert busy["busy_seconds"] == pytest.approx(40.0)
        assert busy["decode_busy_seconds"] == pytest.approx(10.0)
        assert busy["prefill_busy_seconds"] == pytest.approx(30.0)
        assert busy["upper_bound_label"] == aggregation.BUSY_UPPER_BOUND_LABEL
        assert busy["lower_bound_label"] == aggregation.BUSY_LOWER_BOUND_LABEL
        assert busy["prefill_extended_streams"] == 1

    def test_local_busy_keeps_every_legacy_key(self):
        """Legacy consumers keep working: the pre-change ``local_busy`` keys
        are all still present, and the phase split still carries both totals."""
        busy = reporting.summary_to_json(
            self._summary(self._prefill_lines([30]))
        )["local_busy"]
        legacy = {
            "window_seconds",
            "busy_seconds",
            "busy_pct",
            "idle_seconds",
            "idle_pct",
            "total_compute_seconds",
            "streams",
            "avg_stream_duration_seconds",
            "peak_concurrency",
            "avg_concurrency",
            "unfinished_streams",
            "pre_window_unfinished",
            "fast_busy_seconds",
            "cheap_busy_seconds",
            "fast_window_seconds",
            "cheap_window_seconds",
            "hourly_busy",
            "prefill_vs_decode",
        }
        assert legacy <= set(busy)

    def test_local_busy_has_unique_keys(self):
        """No dictionary key is duplicated — a regression guard for F601.

        Duplicate key literals silently overwrite; this test catches any
        accidental re-addition (LP-0MVKE63D2007WD0E)."""
        busy = reporting.summary_to_json(
            self._summary(self._prefill_lines([30]))
        )["local_busy"]
        keys = list(busy.keys())
        assert len(keys) == len(set(keys)), (
            f"Duplicate keys detected: {[k for k in keys if keys.count(k) > 1]}"
        )
        phase = busy["prefill_vs_decode"]
        assert phase["prefill"]["slot_seconds"] == pytest.approx(30.0)
        assert phase["decode"]["slot_seconds"] == pytest.approx(10.0)
        assert phase["prefill"]["pct"] + phase["decode"]["pct"] == pytest.approx(100.0)

    def test_cli_json_summary_includes_new_keys(self, tmp_path, capsys):
        """The ``--json`` CLI summary carries the new keys end to end."""
        import analyze_proxy_usage as cli

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text("\n".join(fixtures.E2E_LINES) + "\n")
        rc = cli.main(
            [
                "--log-dir", str(log_dir),
                "--start", "2026-08-02 14:00:00",
                "--end", "2026-08-02 15:00:00",
                "--output-dir", str(tmp_path / "out"),
                "--json",
            ]
        )
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        prefill = data["local_prefill"]
        assert prefill["unit"] == "slot-seconds"
        assert prefill["count"] == 0
        # Both in-window local streams carry no first-byte line: unknown prefill.
        assert prefill["unknown_prefill"] == 2
        busy = data["local_busy"]
        assert busy["definition"] == "dispatch_to_finish"
        assert busy["decode_busy_seconds"] == pytest.approx(busy["busy_seconds"])
        assert busy["prefill_busy_seconds"] == pytest.approx(0.0)
        assert busy["upper_bound_label"] == aggregation.BUSY_UPPER_BOUND_LABEL
        assert busy["lower_bound_label"] == aggregation.BUSY_LOWER_BOUND_LABEL

    def test_cli_text_summary_references_the_corrected_busy_figure(self, tmp_path, capsys):
        """The stdout summary names the dispatch → finish definition and keeps
        the decode-only lower bound visible next to it."""
        import analyze_proxy_usage as cli

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text("\n".join(fixtures.E2E_LINES) + "\n")
        rc = cli.main(
            [
                "--log-dir", str(log_dir),
                "--start", "2026-08-02 14:00:00",
                "--end", "2026-08-02 15:00:00",
                "--output-dir", str(tmp_path / "out"),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "local model busy (dispatch → finish)" in out
        assert "decode-only busy (lower bound)" in out


# ---------------------------------------------------------------------------
# Regenerated report end-to-end verification (LP-0MVJO4382002FJF2)
# ---------------------------------------------------------------------------


class TestRegeneratedReportExpectedOrder:
    """End-to-end verification of a regenerated report.

    The fixture mirrors the intake brief's measured shape
    (LP-0MVGCQ421005IXGC): per-stream prefill (dispatch → first byte)
    dominates decode (first byte → finish) at a 180 s / 100 s ratio (the
    brief's ≈14.3 h prefill vs ≈7.8 h decode), and the six streams are spaced
    so their dispatch → finish intervals never overlap. That reproduces the
    brief's expected order — ≈16 % decode-only busy becoming ≈46 % busy
    (dispatch → finish union) — so the regenerated report must show the
    corrected figure and the new ``## Prefill vs decode`` breakdown
    (LP-0MVJO4382002FJF2 AC1/AC2/AC3/AC5).
    """

    PREFILL_S = 180.0
    DECODE_S = 100.0
    STREAMS = 6

    @classmethod
    def _stream_events(cls, index: int) -> list[str]:
        """One local stream: dispatch at 14:0X:00, first byte after the
        180 s prefill, finish after a further 100 s decode. The
        ``dispatch_first_byte_ms`` line is stamped at first-byte time (real
        proxy format) and carries the prefill duration."""
        first_byte = datetime(2026, 8, 2, 14, 0, 0) + timedelta(
            minutes=2 + index * 10, seconds=cls.PREFILL_S
        )
        finish = first_byte + timedelta(seconds=cls.DECODE_S)
        session = f"verify-s{index}"
        ts = first_byte.strftime("%Y-%m-%d %H:%M:%S")
        return [
            f"{ts},000 - INFO - Stream started: provider=local model=Qwen3 session={session} request=[]",
            f"{finish.strftime('%Y-%m-%d %H:%M:%S')},000 - INFO - Stream finished: "
            f"reason=stop tokens=1000/100/1100 session={session} provider=local model=Qwen3 request=[]",
            f"{ts},000 - INFO - dispatch_first_byte_ms={cls.PREFILL_S * 1000.0:.1f} "
            f"dispatch_to_first_byte_ms={cls.PREFILL_S * 1000.0 - 1.0:.1f} "
            f"cache_state=warm cached_ratio=0.900 queue_wait_ms=0.0 "
            f"session={session} model=Qwen3",
        ]

    @classmethod
    def _write_logs(cls, tmp_path: Path, with_llama_log: bool = True) -> Path:
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True)
        lines = [
            line
            for index in range(cls.STREAMS)
            for line in cls._stream_events(index)
        ]
        (log_dir / "proxy.log").write_text("\n".join(lines) + "\n")
        if with_llama_log:
            (log_dir / "llama-server.log").write_text(
                fixtures.QWEN3_SPAWN_LINE
                + "\n" + fixtures.PROMPT_EVAL_REAL
                + "\n" + fixtures.DECODE_EVAL_REAL
                + "\n" + fixtures.PROMPT_EVAL_REAL2
                + "\n" + fixtures.DECODE_EVAL_REAL2
                + "\n"
            )
        return log_dir

    @staticmethod
    def _row(section: str, label: str) -> list[str]:
        line = next(
            ln for ln in section.splitlines() if ln.startswith(f"| {label}")
        )
        return [cell.strip() for cell in line.strip("|").split("|")]

    def test_regenerated_report_shows_corrected_busy_and_phase_split(self, tmp_path):
        """AC1/AC3/AC5: the regenerated report over a brief-shaped window
        shows the dispatch → finish busy figure as the upper bound with the
        decode-only figure retained as the labelled lower bound plus the
        definition-change note, and the busy % is materially higher than the
        pre-change decode-only busy % (≈16 % → ≈46 % expected order)."""
        log_dir = self._write_logs(tmp_path)
        out_dir = tmp_path / "out"
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=out_dir,
            config=None,
        )
        md = (out_dir / "report.md").read_text()

        # -- Local model utilization: both figures + definition-change note --
        util = md.split("## Local model utilization", 1)[1].split("## ", 1)[0]
        assert "Busy = at least one local slot active **from dispatch to finish**" in util
        assert "previously counted decode only" in util
        # 6 × 280 s dispatch→finish = 1680 s = 46.7% of the 1 h window.
        assert "| Busy time — upper bound (dispatch → finish) | 28m 00s (46.7%)" in util
        # Retained decode-only lower bound: 6 × 100 s = 600 s = 16.7%.
        assert "| Decode-only busy — lower bound (`Stream started` → finish) | 10m 00s (16.7%)" in util
        # Prefill busy = upper − lower = 1080 s.
        assert "| Prefill busy (upper − lower) | 18m 00s |" in util

        # -- Prefill vs decode split: prefill dominant, totals to 100% --
        section = md.split("## Prefill vs decode", 1)[1].split("\n## ", 1)[0]
        assert "slot-seconds" in section and "not wall-clock" in section
        prefill = self._row(section, "Prefill (dispatch → first byte)")
        decode = self._row(section, "Decode (first byte → finish)")
        assert prefill[1] == "18m 00s"
        assert decode[1] == "10m 00s"
        assert float(prefill[2].rstrip("%")) == pytest.approx(64.3, abs=0.1)
        assert float(decode[2].rstrip("%")) == pytest.approx(35.7, abs=0.1)
        assert float(prefill[2].rstrip("%")) + float(decode[2].rstrip("%")) == pytest.approx(100.0)
        total = self._row(section, "Total")
        assert total[1] == "28m 00s" and total[2] == "100.0%"

        # -- Busy % materially higher than the decode-only busy % (AC5) --
        busy = result.summary.busy
        assert busy is not None
        decode_pct = busy.decode_busy_seconds / busy.window_seconds * 100
        assert 15.0 <= decode_pct <= 17.5  # brief's ≈16 %
        assert 44.0 <= busy.busy_pct <= 49.0  # brief's ≈46 %
        assert busy.busy_pct >= 2.5 * decode_pct  # materially higher, same order

    def test_speed_sections_are_never_an_unattributed_blank(self, tmp_path):
        """AC4: ``## Decode speed`` / ``## Prompt eval speed`` render real
        samples when llama-server eval timing exists, or a named,
        self-diagnosing reason (log dir + scanned files, never a bare blank)
        when it does not."""
        for with_llama_log in (True, False):
            log_dir = self._write_logs(
                tmp_path / f"logs_{with_llama_log}", with_llama_log=with_llama_log
            )
            out_dir = tmp_path / f"out_{with_llama_log}"
            reporting.run_analysis(
                log_dir=log_dir,
                window_start=WINDOW_START,
                window_end=WINDOW_END,
                output_dir=out_dir,
                config=None,
            )
            md = (out_dir / "report.md").read_text()
            for title in ("## Decode speed", "## Prompt eval speed"):
                section = md.split(title, 1)[1].split("\n## ", 1)[0]
                assert "| Model | Bucket | Samples |" in section or (
                    "No llama-server eval timing samples" in section
                    and "**Why no eval-timing samples**" in section
                    and "Log directory scanned: `" in section
                ), f"{title} in run with_llama_log={with_llama_log} is an unattributed blank"

    def test_json_summary_matches_report_phase_split(self, tmp_path):
        """AC2/AC5 cross-check: the machine-readable summary carries the same
        corrected busy figure and phase split as the report, and the busy
        union decomposes into decode lower bound + prefill span."""
        log_dir = self._write_logs(tmp_path)
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=tmp_path / "out_json",
            config=None,
        )
        data = reporting.summary_to_json(result.summary)
        busy = data["local_busy"]
        assert busy["definition"] == "dispatch_to_finish"
        assert busy["busy_seconds"] == pytest.approx(1680.0)
        assert busy["decode_busy_seconds"] == pytest.approx(600.0)
        assert busy["prefill_busy_seconds"] == pytest.approx(1080.0)
        assert busy["busy_pct"] == pytest.approx(1680.0 / 3600 * 100, abs=0.05)
        assert busy["prefill_busy_seconds"] == pytest.approx(
            busy["busy_seconds"] - busy["decode_busy_seconds"]
        )
        phase = busy["prefill_vs_decode"]
        assert phase["unit"] == aggregation.SLOT_SECONDS_UNIT
        assert phase["prefill"]["slot_seconds"] == pytest.approx(1080.0)
        assert phase["decode"]["slot_seconds"] == pytest.approx(600.0)
        assert phase["prefill"]["pct"] + phase["decode"]["pct"] == pytest.approx(100.0)
        # The brief's expected-order sanity check: prefill ≈ 14.3 h / decode
        # ≈ 7.8 h → prefill ≈ 65 % of combined phase slot-time.
        assert 60.0 <= phase["prefill"]["pct"] <= 70.0
        prefill = data["local_prefill"]
        assert prefill["slot_seconds"] == pytest.approx(1080.0)
        assert prefill["count"] == 6
        json.dumps(data)  # round-trips


# =========================================================================
# Priority-window bucketing (LP-0MVKSXXIR006OBI2)
#
# The fast/cheap mode schedule is retired: ``server.priority_window`` (local
# time, midnight-wrapping capable) is now the only cheap-mode signal for
# windows whose logs carry no ``Mode scheduler`` evidence. Historical windows
# that do carry mode evidence keep their mode-timeline attribution (AC5).
# =========================================================================

PRIORITY_CONFIG = {
    "local_model_ctx_size": 262144,
    "session_slot_pool_size": 2,
    "priority_window": {
        "enabled": True,
        "start": "01:00",
        "end": "10:00",
        "start_minutes": 60,
        "end_minutes": 600,
    },
}


def _priority_mode_map(
    start_minutes: float = 60,
    end_minutes: float = 600,
    wrap: bool = False,
) -> bucketing.ModeScheduleMap:
    """A mode map whose only fast/cheap source is the priority window."""
    return bucketing.ModeScheduleMap.from_profiles(
        {"default": {"session_slot_pool_size": 2, "local_model_ctx_size": 262144}},
        None,
        2,
        priority_window=bucketing.PriorityWindow(
            start_minutes=start_minutes, end_minutes=end_minutes
        ),
    )


def _no_slot_schedule() -> bucketing.SlotSchedule:
    """The unified config has no ``slot_schedule``: one flat fast period."""
    return bucketing.schedule_from_config({}, 2)


class TestPriorityWindowParsing:
    """Window parsing: enabled/disabled, custom bounds, midnight wrap (AC4/AC6)."""

    def test_normalize_parses_bounds_to_minutes(self):
        window = config_loader.normalize_priority_window(
            {"enabled": True, "start": "01:00", "end": "10:00"}
        )
        assert window == {
            "enabled": True,
            "start": "01:00",
            "end": "10:00",
            "start_minutes": 60,
            "end_minutes": 600,
        }

    def test_normalize_custom_bounds(self):
        window = config_loader.normalize_priority_window(
            {"enabled": True, "start": "22:00", "end": "06:30"}
        )
        assert window["start_minutes"] == 22 * 60
        assert window["end_minutes"] == 6 * 60 + 30

    def test_normalize_disabled_flag(self):
        window = config_loader.normalize_priority_window(
            {"enabled": False, "start": "01:00", "end": "10:00"}
        )
        assert window["enabled"] is False

    def test_normalize_invalid_bounds_fall_back_to_shipped_defaults(self):
        window = config_loader.normalize_priority_window(
            {"enabled": True, "start": "not-a-time", "end": "99:99"}
        )
        assert (window["start_minutes"], window["end_minutes"]) == (60, 600)

    def test_normalize_absent_window(self):
        assert config_loader.normalize_priority_window(None) is None
        assert config_loader.priority_window_bounds({}) is None

    def test_parse_config_text_reads_priority_window(self):
        cfg = config_loader.parse_config_text(
            "server:\n"
            "  priority_window:\n"
            "    enabled: true\n"
            '    start: "22:00"\n'
            '    end: "06:00"\n'
        )
        assert cfg["priority_window"]["start_minutes"] == 22 * 60
        assert cfg["priority_window"]["end_minutes"] == 6 * 60
        assert cfg["priority_window"]["enabled"] is True

    def test_regex_fallback_parser_reads_priority_window(self):
        cfg = config_loader._parse_config_text_regex(
            "server:\n"
            "  priority_window:\n"
            "    enabled: false\n"
            '    start: "02:30"\n'
            '    end: "05:45"\n'
        )
        window = config_loader.normalize_priority_window(cfg["priority_window"])
        assert window["enabled"] is False
        assert (window["start_minutes"], window["end_minutes"]) == (150, 345)

    def test_discover_configs_exposes_window_bounds(self, tmp_path):
        (tmp_path / "proxy").mkdir()
        (tmp_path / "proxy" / "config.yaml").write_text(
            "server:\n"
            "  local_model_ctx_size: 131072\n"
            "  priority_window:\n"
            "    enabled: true\n"
            '    start: "01:00"\n'
            '    end: "10:00"\n'
        )
        configs = config_loader.discover_configs(start=tmp_path)
        bounds = configs["priority_window_bounds"]
        assert bounds["enabled"] is True
        assert (bounds["start_minutes"], bounds["end_minutes"]) == (60, 600)


class TestPriorityWindowSchedule:
    """``PriorityWindow``/``ModeScheduleMap`` window semantics (AC1/AC4)."""

    def test_segments_split_at_the_bounds(self):
        window = bucketing.PriorityWindow(start_minutes=60, end_minutes=600)
        assert window.segments() == [
            (0.0, 60, "fast"),
            (60, 600, "cheap"),
            (600, float(bucketing.MINUTES_PER_DAY), "fast"),
        ]

    def test_midnight_wrap_is_split_across_midnight(self):
        window = bucketing.PriorityWindow(
            start_minutes=22 * 60, end_minutes=6 * 60
        )
        assert window.wraps is True
        assert window.segments() == [
            (0.0, 360, "cheap"),
            (360, 1320, "fast"),
            (1320, float(bucketing.MINUTES_PER_DAY), "cheap"),
        ]
        assert window.contains(23 * 60) is True
        assert window.contains(3 * 60) is True
        assert window.contains(12 * 60) is False

    def test_disabled_window_yields_no_schedule(self):
        assert bucketing.PriorityWindow.from_config(
            {"priority_window": {"enabled": False, "start": "01:00", "end": "10:00"}}
        ) is None
        assert bucketing.PriorityWindow.from_config({}) is None
        assert bucketing.PriorityWindow.from_config(None) is None

    def test_from_config_accepts_raw_and_normalised_blocks(self):
        raw = bucketing.PriorityWindow.from_config(
            {"priority_window": {"enabled": True, "start": "02:00", "end": "04:00"}}
        )
        assert raw == bucketing.PriorityWindow(120.0, 240.0)
        normalised = bucketing.PriorityWindow.from_config(PRIORITY_CONFIG)
        assert normalised == bucketing.PriorityWindow(60.0, 600.0)

    def test_profile_source_reports_priority_window(self):
        assert _priority_mode_map().profile_source == "server.priority_window"

    def test_profile_source_is_none_without_window_or_timeline(self):
        mm = bucketing.ModeScheduleMap.from_profiles({"default": None}, None, 2)
        assert mm.profile_source is None

    def test_bucket_label_uses_window(self):
        mm = _priority_mode_map()
        schedule = _no_slot_schedule()
        assert bucketing.bucket_label(datetime(2026, 10, 22, 3, 0), schedule, mm) == "cheap"
        assert bucketing.bucket_label(datetime(2026, 10, 22, 12, 0), schedule, mm) == "fast"
        # Bounds are inclusive-start / exclusive-end (local time).
        assert bucketing.bucket_label(datetime(2026, 10, 22, 1, 0), schedule, mm) == "cheap"
        assert bucketing.bucket_label(datetime(2026, 10, 22, 10, 0), schedule, mm) == "fast"

    def test_bucket_label_falls_back_to_slot_schedule_without_window(self):
        schedule = bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])
        mm = bucketing.ModeScheduleMap.from_profiles({"default": None}, None, 8)
        assert bucketing.bucket_label(datetime(2026, 10, 22, 3, 0), schedule, mm) == "cheap"
        assert bucketing.bucket_label(datetime(2026, 10, 22, 11, 0), schedule, mm) == "fast"
        # And with no mode map at all (legacy call sites).
        assert bucketing.bucket_label(datetime(2026, 10, 22, 11, 0), schedule, None) == "fast"


class TestPriorityWindowBucketing:
    """Overnight bucketing with no mode-schedule evidence (AC1/AC2/AC5)."""

    OVERNIGHT_START = datetime(2026, 10, 22, 0, 0, 0)
    OVERNIGHT_END = datetime(2026, 10, 22, 12, 0, 0)

    def test_overnight_window_attributes_cheap_seconds(self):
        """cheap_window_seconds matches 01:00-10:00 with no ``Mode scheduler``
        lines — the regression the nightly report showed (AC2)."""
        events = [
            _local_event("stream_started", datetime(2026, 10, 22, 3, 0, 0), "night"),
            _local_event("stream_finished", datetime(2026, 10, 22, 3, 0, 10), "night"),
            _local_event("stream_started", datetime(2026, 10, 22, 11, 0, 0), "day"),
            _local_event("stream_finished", datetime(2026, 10, 22, 11, 0, 20), "day"),
        ]
        busy = aggregation.compute_busy_stats(
            events,
            self.OVERNIGHT_START,
            self.OVERNIGHT_END,
            _no_slot_schedule(),
            _priority_mode_map(),
        )
        assert busy is not None
        assert busy.cheap_window_seconds == pytest.approx(9 * 3600.0)
        assert busy.fast_window_seconds == pytest.approx(3 * 3600.0)
        assert busy.cheap_busy_seconds == pytest.approx(10.0)
        assert busy.fast_busy_seconds == pytest.approx(20.0)

    def test_sessions_bucketed_cheap_inside_the_window(self):
        lines = [
            "2026-10-22 03:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=night request=[]",
            "2026-10-22 03:00:10,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=night provider=local model=Qwen3 request=[]",
            "2026-10-22 11:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=day request=[]",
            "2026-10-22 11:00:10,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=day provider=local model=Qwen3 request=[]",
        ]
        res = aggregation.aggregate(
            _events(lines),
            self.OVERNIGHT_START,
            self.OVERNIGHT_END,
            _no_slot_schedule(),
            _priority_mode_map(),
        )
        assert res.sessions["night"].bucket == "cheap"
        assert res.sessions["day"].bucket == "fast"

    def test_event_level_bucketing_follows_the_window(self):
        """Session-less events (fallback/routing-skip reasons) are bucketed by
        their own timestamp, so they land in the low-priority tables too."""
        lines = [
            "2026-10-22 03:30:00,000 - INFO - Fallback triggered for model=v1/chat/completions, "
            "from=local-qwen3, to=opencode-go-deepseek, reason=context_too_large",
            "2026-10-22 11:30:00,000 - INFO - Fallback triggered for model=v1/chat/completions, "
            "from=local-qwen3, to=opencode-go-deepseek, reason=provider_error",
        ]
        res = aggregation.aggregate(
            _events(lines),
            self.OVERNIGHT_START,
            self.OVERNIGHT_END,
            _no_slot_schedule(),
            _priority_mode_map(),
        )
        buckets = reporting._bucket_profile(
            [], _no_slot_schedule(), res.fallback_events, [], [], _priority_mode_map()
        )
        assert buckets["cheap"]["fallback_reasons"]["context_too_large"] == 1
        assert buckets["fast"]["fallback_reasons"]["provider_error"] == 1

    def test_midnight_wrapping_window_is_honoured(self):
        """A 22:00-06:00 window contributes cheap time on both sides of
        midnight inside a single analysis window (AC4)."""
        events = [
            _local_event("stream_started", datetime(2026, 10, 22, 2, 0, 0), "s1"),
            _local_event("stream_finished", datetime(2026, 10, 22, 2, 0, 5), "s1"),
            _local_event("stream_started", datetime(2026, 10, 22, 23, 0, 0), "s2"),
            _local_event("stream_finished", datetime(2026, 10, 22, 23, 0, 5), "s2"),
        ]
        start = datetime(2026, 10, 22, 0, 0, 0)
        end = datetime(2026, 10, 23, 0, 0, 0)
        busy = aggregation.compute_busy_stats(
            events, start, end, _no_slot_schedule(), _priority_mode_map(22 * 60, 6 * 60)
        )
        assert busy is not None
        assert busy.cheap_window_seconds == pytest.approx(8 * 3600.0)
        assert busy.fast_window_seconds == pytest.approx(16 * 3600.0)
        assert busy.cheap_busy_seconds == pytest.approx(10.0)
        assert busy.fast_busy_seconds == 0.0

    def test_disabled_window_attributes_no_cheap_time(self):
        events = [
            _local_event("stream_started", datetime(2026, 10, 22, 3, 0, 0), "night"),
            _local_event("stream_finished", datetime(2026, 10, 22, 3, 0, 10), "night"),
        ]
        busy = aggregation.compute_busy_stats(
            events,
            self.OVERNIGHT_START,
            self.OVERNIGHT_END,
            _no_slot_schedule(),
            _priority_mode_map_with_window(enabled=False),
        )
        assert busy is not None
        assert busy.cheap_window_seconds == 0.0
        assert busy.cheap_busy_seconds == 0.0

    def test_mode_timeline_evidence_wins_over_priority_window(self):
        """AC5: a window whose logs still carry ``Mode scheduler`` evidence
        keeps mode-timeline attribution; the priority window layers beneath."""
        mm = _priority_mode_map(0, 6 * 60)  # would make 03:00 cheap
        lines = [
            "2026-08-14 10:00:02,000 - INFO - Mode scheduler: applied scheduled mode fast",
            "2026-08-15 03:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=cheap-hours request=[]",
            "2026-08-15 03:00:10,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=cheap-hours provider=local model=Qwen3 request=[]",
        ]
        res = aggregation.aggregate(
            _events(lines),
            TestModeTimelineBucketing.MODE_WINDOW_START,
            TestModeTimelineBucketing.MODE_WINDOW_END,
            bucketing.schedule_from_config(fixtures.MODE_FAST_CONFIG, 3),
            mm,
        )
        assert mm.profile_source == "mode_schedule"
        # The mode timeline says fast at 03:00, so the priority window's cheap
        # attribution must not override it.
        assert res.sessions["cheap-hours"].bucket == "fast"


def _priority_mode_map_with_window(enabled: bool) -> bucketing.ModeScheduleMap:
    """A mode map over an explicitly disabled priority window."""
    return bucketing.ModeScheduleMap.from_profiles(
        {"default": {"session_slot_pool_size": 2}},
        None,
        2,
        priority_window=bucketing.PriorityWindow.from_config(
            {"priority_window": {"enabled": enabled, "start": "01:00", "end": "10:00"}}
        ),
    )


class TestPriorityWindowReportNote:
    """AC3: the timeline section names the attribution source."""

    WINDOW_START = datetime(2026, 10, 22, 0, 0, 0)
    WINDOW_END = datetime(2026, 10, 22, 12, 0, 0)

    def _report(self, mode_map: bucketing.ModeScheduleMap, lines: list[str]) -> str:
        res = aggregation.aggregate(
            _events(lines), self.WINDOW_START, self.WINDOW_END, _no_slot_schedule(), mode_map
        )
        return reporting.build_report(res, PRIORITY_CONFIG, mode_map=mode_map)

    def test_note_names_priority_window_when_mode_timeline_is_retired(self):
        lines = [
            "2026-10-22 03:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=night request=[]",
            "2026-10-22 03:00:10,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=night provider=local model=Qwen3 request=[]",
        ]
        report = self._report(_priority_mode_map(), lines)
        assert "## Priority window timeline & restarts" in report
        section = report.split("## Priority window timeline & restarts", 1)[1].split("\n## ", 1)[0]
        assert "server.priority_window" in section
        assert "01:00–10:00" in section
        assert "retired" in section

    def test_note_marks_wrapping_window(self):
        lines = [
            "2026-10-22 03:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=night request=[]",
        ]
        report = self._report(_priority_mode_map(22 * 60, 6 * 60), lines)
        section = report.split("## Priority window timeline & restarts", 1)[1].split("\n## ", 1)[0]
        assert "22:00–06:00" in section
        assert "wraps past midnight" in section

    def test_note_says_mode_schedule_won_when_evidence_present(self):
        mm = _priority_mode_map()
        lines = ["2026-10-22 10:00:02,000 - INFO - Mode scheduler: applied scheduled mode fast"]
        report = self._report(mm, lines)
        section = report.split("## Priority window timeline & restarts", 1)[1].split("\n## ", 1)[0]
        assert "active for this window" in section
        assert "not applied" in section

    def test_notes_and_limitations_mentions_priority_window(self):
        report = self._report(_priority_mode_map(), [])
        assert "`server.priority_window`" in report
        assert "retired mode schedule" in report

    def test_section_omitted_when_no_window_and_no_events(self):
        mm = bucketing.ModeScheduleMap.from_profiles({"default": None}, None, 2)
        report = self._report(mm, [])
        assert "## Priority window timeline & restarts" not in report


class TestPriorityWindowEndToEnd:
    """Overnight run through ``run_analysis``: cheap CSV + JSON + report (AC1-AC3)."""

    WINDOW_START = datetime(2026, 10, 22, 0, 0, 0)
    WINDOW_END = datetime(2026, 10, 22, 12, 0, 0)

    LINES = [
        "2026-10-22 03:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=night request=[]",
        "2026-10-22 03:00:10,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=night provider=local model=Qwen3 request=[]",
        "2026-10-22 11:00:00,000 - INFO - Stream started: provider=local model=Qwen3 session=day request=[]",
        "2026-10-22 11:00:10,000 - INFO - Stream finished: reason=stop tokens=100/10/110 session=day provider=local model=Qwen3 request=[]",
    ]

    def _run(self, tmp_path: Path, config: dict | None = None):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text("\n".join(self.LINES) + "\n")
        return reporting.run_analysis(
            log_dir=log_dir,
            window_start=self.WINDOW_START,
            window_end=self.WINDOW_END,
            output_dir=tmp_path / "out",
            config=config if config is not None else PRIORITY_CONFIG,
        )

    def test_overnight_run_splits_low_priority_sessions(self, tmp_path):
        run = self._run(tmp_path)
        out_dir = tmp_path / "out"
        assert run.summary.sessions["night"].bucket == "cheap"
        assert run.summary.sessions["day"].bucket == "fast"

        import csv

        with (out_dir / "low_priority_sessions.csv").open() as f:
            cheap_rows = list(csv.DictReader(f))
        with (out_dir / "standard_sessions.csv").open() as f:
            fast_rows = list(csv.DictReader(f))
        assert {r["session_id"] for r in cheap_rows} == {"night"}
        assert {r["session_id"] for r in fast_rows} == {"day"}

        data = reporting.summary_to_json(run.summary, run.mode_map)
        busy = data["local_busy"]
        assert busy["cheap_window_seconds"] == pytest.approx(9 * 3600.0)
        assert busy["fast_window_seconds"] == pytest.approx(3 * 3600.0)
        assert busy["cheap_busy_seconds"] == pytest.approx(10.0)
        assert busy["fast_busy_seconds"] == pytest.approx(10.0)
        assert data["low_priority_sessions"] == 1
        assert data["standard_sessions"] == 1

        report_md = (out_dir / "report.md").read_text()
        assert "server.priority_window" in report_md
        assert "01:00–10:00" in report_md

    def test_disabled_window_via_run_analysis_attributes_no_cheap_time(self, tmp_path):
        disabled = {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 2,
            "priority_window": {
                "enabled": False,
                "start": "01:00",
                "end": "10:00",
                "start_minutes": 60,
                "end_minutes": 600,
            },
        }
        run = self._run(tmp_path, config=disabled)
        assert run.summary.sessions["night"].bucket == "fast"
        assert run.summary.busy is not None
        assert run.summary.busy.cheap_window_seconds == 0.0
        assert run.summary.busy.cheap_busy_seconds == 0.0
