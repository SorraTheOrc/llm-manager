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
    LATE_FALLBACK_FINISHED,
    LATE_FALLBACK_REMOTE_POST,
    LATE_FALLBACK_SESSION,
    LATE_FALLBACK_STARTED,
    LATE_FALLBACK_TRIGGERED,
    LATE_FALLBACK_TURN,
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


# ---------------------------------------------------------------------------
# Late own fallback line (pre-flight stream buffering)
# ---------------------------------------------------------------------------
#
# LP-0MW8U39FH005GWHY: the proxy logs ``Fallback triggered`` *after* the remote
# stream it describes has started **and finished** (pre-flight buffering), so a
# timestamp-first filter drops the session's own, exactly-matching line and the
# session reads ``unattributed``. The turn id is the exact join key here, so it
# must win over the timestamp.

LATE_WINDOW_START = datetime(2026, 11, 5, 15, 0, 0)
LATE_WINDOW_END = datetime(2026, 11, 5, 16, 0, 0)
# ``Stream started`` of the session's first remote stream: the move time is
# anchored here, never on the later fallback line.
LATE_FIRST_REMOTE_TS = datetime(2026, 11, 5, 15, 50, 10, 341000)


def test_late_own_fallback_is_attributed_by_its_turn_id():
    """AC1/AC2: the fallback line arrives after the remote stream, same turn."""
    lines = [
        LATE_FALLBACK_REMOTE_POST,
        LATE_FALLBACK_STARTED,
        LATE_FALLBACK_FINISHED,
        LATE_FALLBACK_TRIGGERED,
    ]
    schedule = bucketing.schedule_from_entries([("23:59", 8)])

    result = aggregation.aggregate(
        _events(lines), LATE_WINDOW_START, LATE_WINDOW_END, schedule
    )

    session = result.sessions[LATE_FALLBACK_SESSION]
    assert session.fallback_reason == "local_concurrency_limit"
    assert (
        session.fallback_attribution
        == aggregation.FALLBACK_ATTRIBUTION_OWN_FALLBACK
    )
    # The move time never moves later than the first remote stream: the matched
    # line (15:50:11,243) is reported as the stream's own start (15:50:10,341).
    assert session.remote_move_time == LATE_FIRST_REMOTE_TS


def test_late_own_fallback_keeps_context_anchored_on_the_remote_stream():
    """A late matched line does not move the ``context_at_fallback`` anchor.

    With the anchor clamped at the first remote stream, the context is the
    prompt of that remote stream (1000). An anchor left on the late fallback
    line (15:50:11,243) finds no ``Stream finished`` at/after it and silently
    falls back to the local turn's prompt (700).
    """
    local_started = (
        "2026-11-05 15:50:00,000 - INFO - Stream started: provider=local "
        "model=Qwen3 "
        f"session={LATE_FALLBACK_SESSION} request=[{{'type': 'text', 'text': 'x'}}] "
        f"turn={LATE_FALLBACK_TURN}"
    )
    local_finished = (
        "2026-11-05 15:50:02,000 - INFO - Stream finished: reason=stop "
        "tokens=700/50/750 "
        f"session={LATE_FALLBACK_SESSION} provider=local model=Qwen3 "
        f"turn={LATE_FALLBACK_TURN}"
    )
    lines = [
        local_started,
        local_finished,
        LATE_FALLBACK_REMOTE_POST,
        LATE_FALLBACK_STARTED,
        LATE_FALLBACK_FINISHED,
        LATE_FALLBACK_TRIGGERED,
    ]
    schedule = bucketing.schedule_from_entries([("23:59", 8)])

    result = aggregation.aggregate(
        _events(lines), LATE_WINDOW_START, LATE_WINDOW_END, schedule
    )

    session = result.sessions[LATE_FALLBACK_SESSION]
    assert session.remote_move_time == LATE_FIRST_REMOTE_TS
    assert session.context_at_fallback == 1000
    assert (
        session.context_at_fallback_source
        == aggregation.CONTEXT_AT_FALLBACK_REMOTE_STREAM
    )


def test_own_fallback_picker_prefers_the_turn_even_when_it_is_early():
    """The picker is timestamp-agnostic only for an exact turn match.

    An at/before line of another turn no longer wins over the turn's own line,
    and turn-less evidence keeps the historical ``e.ts <= first_remote_ts``
    rule (latest line wins).
    """
    first_remote_ts = datetime(2026, 10, 25, 10, 0, 10)
    early_other_turn = _event(
        "2026-10-25 10:00:09,000 - INFO - Fallback triggered for "
        "model=v1/chat/completions, from=local-qwen3, to=opencode-go, "
        f"reason=context_too_large session={TURN_SESSION} attempt=1 turn={TURN_A}"
    )
    late_own_turn = _event(
        "2026-10-25 10:00:15,000 - INFO - Fallback triggered for "
        "model=v1/chat/completions, from=local-qwen3, to=opencode-go, "
        f"reason=local_concurrency_limit session={TURN_SESSION} attempt=2 "
        f"turn={TURN_B}"
    )
    legacy_own = _event(
        "2026-10-25 10:00:08,000 - INFO - Fallback triggered for "
        "model=v1/chat/completions, from=local-qwen3, to=opencode-go, "
        f"reason=lease_active_legacy session={TURN_SESSION} attempt=1"
    )

    by_turn = aggregation._attribute_fallback(
        TURN_SESSION,
        first_remote_ts,
        {},
        [early_other_turn, late_own_turn, legacy_own],
        turn=TURN_B,
    )
    without_turn = aggregation._attribute_fallback(
        TURN_SESSION, first_remote_ts, {}, [early_other_turn, legacy_own]
    )

    assert by_turn[1] == "local_concurrency_limit"
    assert by_turn[2] == aggregation.FALLBACK_ATTRIBUTION_OWN_FALLBACK
    # No ids: the latest own line at/before the first remote stream still wins.
    assert without_turn[1] == "context_too_large"


# ---------------------------------------------------------------------------
# Turn-first skip evidence and precedence guards (LP-0MW8U39ZQ007MPQO)
# ---------------------------------------------------------------------------
#
# The turn-first picker from the own-fallback tier (LP-0MW8U39FH005GWHY)
# applies to the ``routing_skip_local`` and ``routing_skip_provider_context``
# tiers too: no tier keeps a timestamp filter ahead of an exact ``turn=``
# match, so late-arriving evidence of the session's own turn is attributed
# while the documented precedence order and the no-inheritance guarantee stay
# unchanged (parent LP-0MW0FFW8K002U0KQ AC3).

PRECEDENCE_SESSION = "herdr-precedence-0001"
PRECEDENCE_TURN = "aa11bb22"
PRECEDENCE_FIRST_REMOTE = datetime(2026, 11, 6, 12, 0, 10)


def _late_skip():
    return log_parser.LogEvent(
        "routing_skip",
        PRECEDENCE_FIRST_REMOTE + timedelta(seconds=1),
        session=PRECEDENCE_SESSION,
        reason="context_too_large",
        estimated_tokens=40000,
        turn=PRECEDENCE_TURN,
    )


def _late_provider_skip():
    return log_parser.LogEvent(
        "provider_context_skip",
        PRECEDENCE_FIRST_REMOTE + timedelta(seconds=2),
        session=PRECEDENCE_SESSION,
        reason="provider_context_exceeded",
        estimated_tokens=153917,
        limit=150000,
        turn=PRECEDENCE_TURN,
    )


def _late_own_fallback():
    return log_parser.LogEvent(
        "fallback",
        PRECEDENCE_FIRST_REMOTE + timedelta(seconds=3),
        session=PRECEDENCE_SESSION,
        reason="local_concurrency_limit",
        turn=PRECEDENCE_TURN,
    )


def _legacy_nearest():
    return log_parser.LogEvent(
        "fallback",
        PRECEDENCE_FIRST_REMOTE - timedelta(seconds=5),
        session=None,
        reason="legacy_stream_busy",
        turn=None,
    )


def test_own_skip_tier_picks_the_turns_late_skip_line():
    """A ``routing_skip_local`` line logged **after** the first remote stream
    still wins when it carries the same ``turn=`` (turn-first, AC1).

    The historical ``e.ts <= first_remote_ts`` filter would drop the line and
    read the session as unattributed; without a turn id the same line keeps
    the historical rule (not matched).
    """
    move_ts, reason, attribution = aggregation._attribute_fallback(
        PRECEDENCE_SESSION,
        PRECEDENCE_FIRST_REMOTE,
        {PRECEDENCE_SESSION: [_late_skip()]},
        [],
        turn=PRECEDENCE_TURN,
    )
    no_turn = aggregation._attribute_fallback(
        PRECEDENCE_SESSION,
        PRECEDENCE_FIRST_REMOTE,
        {PRECEDENCE_SESSION: [_late_skip()]},
        [],
    )

    assert reason == "context_too_large"
    assert attribution == aggregation.FALLBACK_ATTRIBUTION_OWN_SKIP
    # The move time never lands after the first remote stream.
    assert move_ts == PRECEDENCE_FIRST_REMOTE
    assert no_turn[2] == aggregation.FALLBACK_ATTRIBUTION_UNATTRIBUTED


def test_own_provider_skip_tier_picks_the_turns_late_line():
    """``routing_skip_provider_context`` uses the same turn-first rule (AC1)."""
    move_ts, reason, attribution = aggregation._attribute_fallback(
        PRECEDENCE_SESSION,
        PRECEDENCE_FIRST_REMOTE,
        {},
        [],
        provider_context_skips={PRECEDENCE_SESSION: [_late_provider_skip()]},
        turn=PRECEDENCE_TURN,
    )
    no_turn = aggregation._attribute_fallback(
        PRECEDENCE_SESSION,
        PRECEDENCE_FIRST_REMOTE,
        {},
        [],
        provider_context_skips={PRECEDENCE_SESSION: [_late_provider_skip()]},
    )

    assert reason == "provider_context_exceeded"
    assert (
        attribution == aggregation.FALLBACK_ATTRIBUTION_OWN_PROVIDER_SKIP
    )
    assert move_ts == PRECEDENCE_FIRST_REMOTE
    assert no_turn[2] == aggregation.FALLBACK_ATTRIBUTION_UNATTRIBUTED


def test_provider_skip_tier_prefers_an_exact_late_turn_over_an_early_other_turn():
    """An at/before line of another turn no longer wins the provider tier.

    The move time discriminates which line was picked: the early other-turn
    line (if chosen) reports its own at/before time, while the exact late
    match is clamped to the first remote stream. Without a ``turn=`` the
    historical rule still picks the early line.
    """
    early_other_turn = log_parser.LogEvent(
        "provider_context_skip",
        PRECEDENCE_FIRST_REMOTE - timedelta(seconds=2),
        session=PRECEDENCE_SESSION,
        reason="provider_context_exceeded",
        estimated_tokens=100000,
        limit=110000,
        turn="deadbeef",  # a *different* turn
    )
    late_own_turn = _late_provider_skip()

    move_ts, reason, attribution = aggregation._attribute_fallback(
        PRECEDENCE_SESSION,
        PRECEDENCE_FIRST_REMOTE,
        {},
        [],
        provider_context_skips={
            PRECEDENCE_SESSION: [early_other_turn, late_own_turn]
        },
        turn=PRECEDENCE_TURN,
    )
    historical = aggregation._attribute_fallback(
        PRECEDENCE_SESSION,
        PRECEDENCE_FIRST_REMOTE,
        {},
        [],
        provider_context_skips={
            PRECEDENCE_SESSION: [early_other_turn, late_own_turn]
        },
    )

    assert reason == "provider_context_exceeded"
    assert (
        attribution == aggregation.FALLBACK_ATTRIBUTION_OWN_PROVIDER_SKIP
    )
    # The exact late match was picked (clamped), not the early other-turn line.
    assert move_ts == PRECEDENCE_FIRST_REMOTE
    # Without ids the latest at/before line still wins, reporting its own time.
    assert historical[0] == PRECEDENCE_FIRST_REMOTE - timedelta(seconds=2)
    assert (
        historical[2] == aggregation.FALLBACK_ATTRIBUTION_OWN_PROVIDER_SKIP
    )


@pytest.mark.parametrize(
    "with_skip,with_provider,with_fallback,with_legacy,denied,"
    "expected_attr,expected_reason",
    [
        # Every own tier's evidence for this turn is timestamped *after* the
        # first remote stream; the documented precedence still decides (AC2).
        (True, True, True, True, True, "own_skip", "context_too_large"),
        (False, True, True, True, True, "own_provider_skip", "provider_context_exceeded"),
        (False, False, True, True, True, "own_fallback", "local_concurrency_limit"),
        (False, False, False, True, True, "legacy_nearest", "legacy_stream_busy"),
        (False, False, False, False, True, "lease_denied", "local_lease_active"),
        (False, False, False, False, False, "unattributed", None),
    ],
)
def test_precedence_order_holds_with_late_turn_evidence(
    with_skip,
    with_provider,
    with_fallback,
    with_legacy,
    denied,
    expected_attr,
    expected_reason,
):
    """own_skip > own_provider_skip > own_fallback > legacy_nearest >
    lease_denied > unattributed, unchanged while late evidence is accepted."""
    routing_skips = {PRECEDENCE_SESSION: [_late_skip()]} if with_skip else {}
    provider_ctx = (
        {PRECEDENCE_SESSION: [_late_provider_skip()]} if with_provider else None
    )
    fallbacks = []
    if with_fallback:
        fallbacks.append(_late_own_fallback())
    if with_legacy:
        fallbacks.append(_legacy_nearest())

    move_ts, reason, attribution = aggregation._attribute_fallback(
        PRECEDENCE_SESSION,
        PRECEDENCE_FIRST_REMOTE,
        routing_skips,
        fallbacks,
        dispatch_denied=1 if denied else 0,
        provider_context_skips=provider_ctx,
        turn=PRECEDENCE_TURN,
    )

    assert attribution == expected_attr
    assert reason == expected_reason
    # A move time is only reported when a timestamped line was matched; the
    # legacy tier's line is at/before the stream, so its time is its own.
    if expected_attr in ("unattributed", "lease_denied"):
        assert move_ts is None
    elif expected_attr == "legacy_nearest":
        assert move_ts == PRECEDENCE_FIRST_REMOTE - timedelta(seconds=5)
    else:
        assert move_ts == PRECEDENCE_FIRST_REMOTE  # clamped late evidence


def test_late_fallback_of_another_session_is_never_borrowed():
    """No-inheritance (AC3): a late ``Fallback triggered`` belonging to a
    different ``session=`` is never used, even with an exact ``turn=`` match."""
    other_session_late = log_parser.LogEvent(
        "fallback",
        PRECEDENCE_FIRST_REMOTE + timedelta(seconds=5),
        session="herdr-other-session",
        reason="local_concurrency_limit",
        turn=PRECEDENCE_TURN,
    )

    move_ts, reason, attribution = aggregation._attribute_fallback(
        PRECEDENCE_SESSION,
        PRECEDENCE_FIRST_REMOTE,
        {},
        [other_session_late],
        turn=PRECEDENCE_TURN,
    )

    assert move_ts is None
    assert reason is None
    assert attribution == aggregation.FALLBACK_ATTRIBUTION_UNATTRIBUTED


def test_late_fallback_of_another_session_never_shadows_own_evidence():
    """Positive control: the session's own late line still wins when a
    neighbour's late line is present."""
    own_late = _late_own_fallback()
    other_late = log_parser.LogEvent(
        "fallback",
        PRECEDENCE_FIRST_REMOTE + timedelta(seconds=6),
        session="herdr-other-session",
        reason="local_concurrency_limit",
        turn=PRECEDENCE_TURN,
    )

    _, reason, attribution = aggregation._attribute_fallback(
        PRECEDENCE_SESSION,
        PRECEDENCE_FIRST_REMOTE,
        {},
        [own_late, other_late],
        turn=PRECEDENCE_TURN,
    )

    assert reason == "local_concurrency_limit"
    assert attribution == aggregation.FALLBACK_ATTRIBUTION_OWN_FALLBACK


def test_late_fallback_of_a_later_turn_is_not_matched_to_this_turn():
    """Bounded match (AC4): a ``Fallback triggered`` for a **later** turn of
    the same session (different ``turn=``) is not matched to the first remote
    stream."""
    later_turn_late = log_parser.LogEvent(
        "fallback",
        PRECEDENCE_FIRST_REMOTE + timedelta(seconds=7),
        session=PRECEDENCE_SESSION,
        reason="local_concurrency_limit",
        turn="99ffeedd",  # a later turn of the same session
    )

    move_ts, reason, attribution = aggregation._attribute_fallback(
        PRECEDENCE_SESSION,
        PRECEDENCE_FIRST_REMOTE,
        {},
        [later_turn_late],
        turn=PRECEDENCE_TURN,
    )

    assert move_ts is None
    assert reason is None
    assert attribution == aggregation.FALLBACK_ATTRIBUTION_UNATTRIBUTED


def test_turnless_evidence_keeps_the_historical_timestamp_rule():
    """Legacy heuristic (AC5): turn-less rotated-log evidence attributes
    exactly as before — an own fallback before the stream is ``own_fallback``
    and one after the stream is not matched."""
    first_remote_ts = datetime(2026, 10, 25, 10, 0, 10)
    own_before = log_parser.LogEvent(
        "fallback",
        first_remote_ts - timedelta(seconds=3),
        session=TURN_SESSION,
        reason="context_too_large",
        turn=None,
    )
    own_after = log_parser.LogEvent(
        "fallback",
        first_remote_ts + timedelta(seconds=3),
        session=TURN_SESSION,
        reason="local_concurrency_limit",
        turn=None,
    )

    before = aggregation._attribute_fallback(
        TURN_SESSION, first_remote_ts, {}, [own_before], turn=None
    )
    after_only = aggregation._attribute_fallback(
        TURN_SESSION, first_remote_ts, {}, [own_after], turn=None
    )

    assert before[1] == "context_too_large"
    assert before[2] == aggregation.FALLBACK_ATTRIBUTION_OWN_FALLBACK
    assert before[0] == first_remote_ts - timedelta(seconds=3)
    assert after_only[2] == aggregation.FALLBACK_ATTRIBUTION_UNATTRIBUTED
