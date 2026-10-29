"""Exact per-turn correlation of parsed proxy log events (LP-0MVMBM467005BT4A).

The proxy stamps an 8-hex ``turn=`` id on every structured line one client
request emits. These tests prove the analysis:

- parses the id onto the relevant events (AC3),
- pairs TTFT and stream start/finish events **exactly** by id, even when two
  turns of the same session interleave (AC3/AC4),
- attributes a fallback from the turn's own evidence, not from the
  latest/nearest line of the same session (AC4),
- and still parses/pairs legacy lines without ids through the existing
  FIFO / nearest-line fallback paths (backwards compatibility).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import aggregation  # noqa: E402
import bucketing  # noqa: E402
import log_parser  # noqa: E402

from tests.fixtures import (  # noqa: E402
    DISPATCH_DENIED,
    DISPATCH_FIRST_BYTE_COLD,
    FALLBACK_SESSION,
    ROUTING_SKIP_CONTEXT_TOO_LARGE,
    ROUTING_SKIP_PROVIDER_CONTEXT,
    TURN_A,
    TURN_A_DISPATCH,
    TURN_A_FALLBACK,
    TURN_A_FINISHED,
    TURN_A_SKIP,
    TURN_A_STARTED,
    TURN_B,
    TURN_B_DENIED,
    TURN_B_DISPATCH,
    TURN_B_FALLBACK,
    TURN_B_FINISHED,
    TURN_B_STARTED,
    TURN_SESSION,
)

WINDOW_START = datetime(2026, 10, 25, 9, 0, 0)
WINDOW_END = datetime(2026, 10, 25, 11, 0, 0)


def _event(line: str) -> log_parser.LogEvent:
    ev = log_parser.parse_log_line(line)
    assert ev is not None, line
    return ev


def _events(lines: list[str]) -> list[log_parser.LogEvent]:
    return [ev for ev in (_ev for _ev in map(log_parser.parse_log_line, lines)) if ev]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line, kind, turn",
    [
        (TURN_A_SKIP, "routing_skip", TURN_A),
        (TURN_A_FALLBACK, "fallback", TURN_A),
        (TURN_A_STARTED, "stream_started", TURN_A),
        (TURN_A_FINISHED, "stream_finished", TURN_A),
        (TURN_A_DISPATCH, "ttft", TURN_A),
        (TURN_B_DENIED, "dispatch_denied", TURN_B),
    ],
)
def test_turn_id_is_parsed_onto_each_event_kind(line: str, kind: str, turn: str):
    ev = _event(line)

    assert ev.kind == kind
    assert ev.turn == turn


def test_turn_id_is_parsed_on_the_provider_context_skip():
    """``routing_skip_provider_context`` carries the id too (AC1/AC3)."""
    line = ROUTING_SKIP_PROVIDER_CONTEXT + f" turn={TURN_A}"

    ev = _event(line)

    assert ev.kind == "provider_context_skip"
    assert ev.turn == TURN_A


@pytest.mark.parametrize(
    "line, kind",
    [
        (ROUTING_SKIP_CONTEXT_TOO_LARGE, "routing_skip"),
        (FALLBACK_SESSION, "fallback"),
        (DISPATCH_FIRST_BYTE_COLD, "ttft"),
        (DISPATCH_DENIED, "dispatch_denied"),
    ],
)
def test_legacy_lines_without_turn_still_parse(line: str, kind: str):
    """Rotated logs predating the field parse exactly as before (AC4)."""
    ev = _event(line)

    assert ev.kind == kind
    assert ev.turn is None


# ---------------------------------------------------------------------------
# TTFT -> stream pairing
# ---------------------------------------------------------------------------


def test_ttft_pairs_with_its_own_turn_not_by_fifo_order():
    """Turn A's cold TTFT belongs to A even though B's stream started second."""
    starts = [_event(TURN_A_STARTED), _event(TURN_B_STARTED)]
    # Deliberately reversed relative to the stream order: a FIFO/positional
    # join would hand A's 55.7 s prefill to B.
    ttfts = [_event(TURN_B_DISPATCH), _event(TURN_A_DISPATCH)]

    matches = log_parser.pair_ttft_to_streams(ttfts, starts)

    by_turn = {m.stream.turn: m for m in matches}
    assert by_turn[TURN_A].ttft is not None
    assert by_turn[TURN_A].prefill_seconds == pytest.approx(55.6565)
    assert by_turn[TURN_B].ttft is not None
    assert by_turn[TURN_B].prefill_seconds == pytest.approx(1.1005)


def test_ttft_still_pairs_fifo_without_ids():
    """Legacy lines keep the per-session FIFO pairing (backwards compat)."""
    legacy_starts = [
        log_parser.LogEvent(
            "stream_started",
            WINDOW_START,
            provider="local",
            session="sess-legacy",
            turn=None,
        ),
        log_parser.LogEvent(
            "stream_started",
            WINDOW_START + timedelta(seconds=10),
            provider="local",
            session="sess-legacy",
            turn=None,
        ),
    ]
    legacy_ttfts = [
        log_parser.LogEvent(
            "ttft",
            WINDOW_START + timedelta(seconds=1),
            session="sess-legacy",
            ttft_ms=1000.0,
        ),
        log_parser.LogEvent(
            "ttft",
            WINDOW_START + timedelta(seconds=11),
            session="sess-legacy",
            ttft_ms=2000.0,
        ),
    ]

    matches = log_parser.pair_ttft_to_streams(legacy_ttfts, legacy_starts)

    assert [m.prefill_seconds for m in matches] == pytest.approx([1.0, 2.0])


def test_ttft_pairs_session_less_events_by_turn():
    """A turn id identifies a request even when no session id was logged."""
    stream = log_parser.LogEvent(
        "stream_started",
        WINDOW_START,
        provider="local",
        session=None,
        turn=TURN_A,
    )
    ttft = log_parser.LogEvent(
        "ttft", WINDOW_START + timedelta(seconds=2), session=None, ttft_ms=2000.0
    )
    ttft = log_parser.LogEvent(
        "ttft",
        WINDOW_START + timedelta(seconds=2),
        session=None,
        ttft_ms=2000.0,
        turn=TURN_A,
    )

    matches = log_parser.pair_ttft_to_streams([ttft], [stream])

    assert len(matches) == 1
    assert matches[0].stream is stream
    assert matches[0].ttft is ttft


# ---------------------------------------------------------------------------
# Stream start -> finish pairing
# ---------------------------------------------------------------------------


def test_stream_start_and_finish_pair_by_turn_when_interleaved():
    """Exact pairing: each start gets its own turn's finish."""
    starts = [_event(TURN_A_STARTED), _event(TURN_B_STARTED)]
    finishes = [_event(TURN_B_FINISHED), _event(TURN_A_FINISHED)]

    pairs, unpaired = log_parser.pair_stream_events(starts, finishes)

    assert unpaired == []
    by_turn = {pair[0].turn: pair[1] for pair in pairs}
    assert by_turn[TURN_A].total == 1200
    assert by_turn[TURN_B].total == 900


def test_stream_pairing_falls_back_to_fifo_without_ids():
    starts = [
        log_parser.LogEvent("stream_started", WINDOW_START, provider="local", session="s"),
        log_parser.LogEvent(
            "stream_started", WINDOW_START + timedelta(seconds=5), provider="local",
            session="s",
        ),
    ]
    finishes = [
        log_parser.LogEvent(
            "stream_finished", WINDOW_START + timedelta(seconds=9), provider="local",
            session="s",
        ),
    ]

    pairs, unpaired = log_parser.pair_stream_events(starts, finishes)

    assert len(pairs) == 1
    assert pairs[0][0] is starts[0]
    assert pairs[0][1] is finishes[0]
    assert unpaired == [starts[1]]


def test_busy_stats_use_exact_turn_pairing():
    """Interleaved local turns keep their own decode durations (AC3)."""
    session = "sess-busy-turn"
    starts = [
        log_parser.LogEvent(
            "stream_started", WINDOW_START, provider="local", session=session, turn=TURN_A
        ),
        log_parser.LogEvent(
            "stream_started",
            WINDOW_START + timedelta(seconds=10),
            provider="local",
            session=session,
            turn=TURN_B,
        ),
    ]
    finishes = [
        log_parser.LogEvent(
            "stream_finished",
            WINDOW_START + timedelta(seconds=20),
            provider="local",
            session=session,
            turn=TURN_B,
        ),
        log_parser.LogEvent(
            "stream_finished",
            WINDOW_START + timedelta(seconds=30),
            provider="local",
            session=session,
            turn=TURN_A,
        ),
    ]
    schedule = bucketing.schedule_from_entries([("23:59", 8)])

    stats = aggregation.compute_busy_stats(
        list(starts) + list(finishes), WINDOW_START, WINDOW_END, schedule
    )

    assert stats is not None
    # A decodes for 30 s, B for 10 s. FIFO pairing would report 20 s / 20 s and
    # thus a 20 s maximum.
    assert stats.decode_stats.max_seconds == 30.0
    assert stats.decode_stats.total_seconds == 40.0


# ---------------------------------------------------------------------------
# Fallback / skip attribution
# ---------------------------------------------------------------------------


def _iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S") + ",000"


def test_fallback_attribution_uses_the_turns_own_line():
    """Turn B's fallback wins over a later same-session line from turn A."""
    first_remote_ts = datetime(2026, 10, 25, 10, 0, 10)
    # Heuristic (latest line wins) would pick turn A's context_too_large.
    turn_a_late = _event(
        "2026-10-25 10:00:09,900 - INFO - Fallback triggered for model=v1/chat/completions, "
        f"from=local-qwen3, to=opencode-go, reason=context_too_large "
        f"session={TURN_SESSION} attempt=1 turn={TURN_A}"
    )
    turn_b = _event(
        "2026-10-25 10:00:09,000 - INFO - Fallback triggered for model=v1/chat/completions, "
        f"from=local-qwen3, to=opencode-go, reason=local_lease_active "
        f"session={TURN_SESSION} attempt=2 turn={TURN_B}"
    )

    with_turn = aggregation._attribute_fallback(
        TURN_SESSION,
        first_remote_ts,
        {},
        [turn_a_late, turn_b],
        turn=TURN_B,
    )
    without_turn = aggregation._attribute_fallback(
        TURN_SESSION, first_remote_ts, {}, [turn_a_late, turn_b]
    )

    assert with_turn[1] == "local_lease_active"
    assert with_turn[2] == aggregation.FALLBACK_ATTRIBUTION_OWN_FALLBACK
    # Documented legacy behaviour without ids: the latest own line wins.
    assert without_turn[1] == "context_too_large"


def test_own_skip_prefers_the_turns_own_skip_line():
    turn_a_skip = log_parser.LogEvent(
        "routing_skip",
        WINDOW_START,
        session=TURN_SESSION,
        reason="context_too_large",
        turn=TURN_A,
    )
    turn_b_skip = log_parser.LogEvent(
        "routing_skip",
        WINDOW_START + timedelta(seconds=2),
        session=TURN_SESSION,
        reason="large_context_bypass",
        turn=TURN_B,
    )
    first_remote_ts = datetime(2026, 10, 25, 10, 0, 5)

    reason, attribution = (
        aggregation._attribute_fallback(
            TURN_SESSION,
            first_remote_ts,
            {TURN_SESSION: [turn_a_skip, turn_b_skip]},
            [],
            turn=TURN_B,
        )[1:3]
    )

    assert reason == "large_context_bypass"
    assert attribution == aggregation.FALLBACK_ATTRIBUTION_OWN_SKIP


def test_session_stats_attribution_is_exact_with_interleaved_turns():
    """End to end: the session's first remote stream attributes by its own turn."""
    lines = [
        TURN_A_SKIP,
        TURN_A_FALLBACK,
        TURN_A_STARTED,
        TURN_A_FINISHED,
        TURN_B_FALLBACK,
        TURN_B_STARTED,
        TURN_B_FINISHED,
    ]
    schedule = bucketing.schedule_from_entries([("23:59", 8)])

    result = aggregation.aggregate(_events(lines), WINDOW_START, WINDOW_END, schedule)

    session = result.sessions[TURN_SESSION]
    assert session.fallback_reason == "context_too_large"
    assert session.fallback_attribution == aggregation.FALLBACK_ATTRIBUTION_OWN_SKIP
    assert session.remote_requests == 2
