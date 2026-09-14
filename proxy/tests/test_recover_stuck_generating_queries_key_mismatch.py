"""
Regression tests for LP-0MU1RXEGB002CAL4:
Reconciliation false positives — counter reset mid-stream on a multi-endpoint backend.

Bug: when generating_sessions contains string session IDs but local_dispatch_records
uses tuple keys (endpoint_url, session_id), the reconciliation lookup fails and
active sessions are incorrectly reclaimed.

Evidence from production (2026-09-14):
- 811 "local_generating_queries counter recovered" WARNINGs in one day.
- 287/300 sampled recoveries fired while a local stream was still open.
- Concrete case: stream started at 16:16:42, counter reset at 16:16:44,
  stream finished at 16:16:45 (2 seconds later).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_srv(
    max_local: int = 3,
    generating_queries: int = 0,
    generating_sessions: set | None = None,
    dispatch_records: dict | None = None,
    active_queries: int = 0,
):
    """Build a minimal srv fixture mimicking proxy.server state."""
    generating_sessions = generating_sessions or set()
    dispatch_records = dispatch_records or {}

    return SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 60}},
        local_active_queries=active_queries,
        local_active_queries_lock=asyncio.Lock(),
        local_generating_queries=generating_queries,
        local_generating_queries_lock=asyncio.Lock(),
        local_generating_sessions=generating_sessions,
        local_dispatch_records=dispatch_records,
        local_dispatch_records_lock=asyncio.Lock(),
        logger=MagicMock(),
    )


# ---------------------------------------------------------------------------
# AC1: Regression test — tuple-keyed dispatch records cause false reclaim
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regression_tuple_keyed_records_false_reclaim():
    """A live stream with a tuple-keyed dispatch record is NOT reclaimed.

    This reproduces the production bug where:
    - generating_sessions contains string session keys (e.g., "herdr-1789398926-1173545-8175")
    - local_dispatch_records uses tuple keys (endpoint_url, session_id)
    - The lookup records.get(key) fails for string keys against tuple-record keys
    - The fallback records.get(("local", key)) also fails because the tuple
      uses the real endpoint URL, not "local"
    - Result: the active session is incorrectly marked stale and reclaimed
    """
    from proxy.router_helpers import (
        _recover_stuck_generating_queries,
    )

    session_id = "herdr-1789398926-1173545-8175"
    endpoint_url = "http://localhost:8080"

    srv = _make_srv(
        max_local=1,
        generating_queries=1,
        generating_sessions={session_id},
        dispatch_records={
            # Dispatch record keyed by tuple (endpoint, session_id) — this is the
            # actual key format when a non-"local" endpoint is used.
            (endpoint_url, session_id): {
                "backend": endpoint_url,
                "started_at": 0.0,
                "active": True,
                "expires_at": 10**12,
            }
        },
        active_queries=1,
    )

    initial_count = srv.local_generating_queries
    initial_sessions = set(srv.local_generating_sessions)

    with patch("proxy.contention_queue.wake_all", AsyncMock()) as mock_wake:
        await _recover_stuck_generating_queries(srv)

    # The active session should NOT be reclaimed
    assert srv.local_generating_queries == initial_count, (
        f"Counter should remain {initial_count} for active stream; got {srv.local_generating_queries}"
    )
    assert srv.local_generating_sessions == initial_sessions, (
        f"Session set should be unchanged for active stream; got {srv.local_generating_sessions}"
    )
    # No WARNING should be logged since no reclaim happened
    warning_calls = [
        c for c in srv.logger.warning.call_args_list
        if "recover" in str(c).lower()
    ]
    assert len(warning_calls) == 0, (
        f"No recovery WARNING should be logged for active stream; got {len(warning_calls)} warnings"
    )
    mock_wake.assert_not_called()


# ---------------------------------------------------------------------------
# AC2: Mixed tuple + string keyed records
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mixed_tuple_and_string_keyed_records():
    """When dispatch records use both tuple and string keys, active sessions
    in generating_sessions are correctly identified regardless of key format.
    """
    from proxy.router_helpers import (
        _recover_stuck_generating_queries,
    )

    session_id_tuple = "stream-1"
    session_id_string = "stream-2"
    endpoint_url = "http://localhost:8080"

    srv = _make_srv(
        max_local=3,
        generating_queries=3,
        generating_sessions={session_id_tuple, session_id_string, "leaked-1"},
        dispatch_records={
            # Tuple-keyed record
            (endpoint_url, session_id_tuple): {
                "backend": endpoint_url,
                "started_at": 0.0,
                "active": True,
                "expires_at": 10**12,
            },
            # String-keyed record (legacy "local" backend)
            session_id_string: {
                "backend": "local",
                "started_at": 0.0,
                "active": True,
                "expires_at": 10**12,
            },
        },
        active_queries=2,
    )

    initial_count = srv.local_generating_queries
    initial_active = {session_id_tuple, session_id_string}

    with patch("proxy.contention_queue.wake_all", AsyncMock()):
        await _recover_stuck_generating_queries(srv)

    # Only the leaked entry should be reclaimed; both active sessions preserved
    assert srv.local_generating_queries == 2, (
        f"Counter should be 2 (2 active); got {srv.local_generating_queries}"
    )
    assert session_id_tuple in srv.local_generating_sessions, (
        "Tuple-keyed active session should be preserved"
    )
    assert session_id_string in srv.local_generating_sessions, (
        "String-keyed active session should be preserved"
    )
    assert "leaked-1" not in srv.local_generating_sessions, (
        "Leaked session should be reclaimed"
    )


# ---------------------------------------------------------------------------
# AC3: Counter recovered WARNING does not fire while local stream is open
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_counter_recovered_warning_while_stream_open():
    """The 'counter recovered' WARNING must not fire when a stream is in flight.

    This was the key production symptom: 287/300 sampled recoveries fired while
    a local stream was still open, with concrete cases showing a 2-second window
    between stream start and spurious counter reset.
    """
    from proxy.router_helpers import (
        _recover_stuck_generating_queries,
    )

    session_id = "herdr-1789398926-1173545-8175"
    endpoint_url = "http://localhost:8080"

    srv = _make_srv(
        max_local=1,
        generating_queries=1,
        generating_sessions={session_id},
        dispatch_records={
            (endpoint_url, session_id): {
                "backend": endpoint_url,
                "started_at": 0.0,
                "active": True,
                "expires_at": 10**12,
            }
        },
        active_queries=1,
    )

    with patch("proxy.contention_queue.wake_all", AsyncMock()):
        await _recover_stuck_generating_queries(srv)

    # Verify no "counter recovered" warning
    for call in srv.logger.warning.call_args_list:
        msg = call[0][0] % call[0][1:] if isinstance(call[0][0], str) else str(call[0][0])
        assert "counter recovered" not in msg.lower(), (
            f"'counter recovered' WARNING should NOT fire while stream is open; got: {msg}"
        )


# ---------------------------------------------------------------------------
# AC4: Genuine stale recovery still works after fix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_genuine_stale_recovery_still_works():
    """True stale-state recovery still works — entries with no matching
    dispatch record are still reclaimed.
    """
    from proxy.router_helpers import (
        _recover_stuck_generating_queries,
    )

    session_id = "leaked-session"
    endpoint_url = "http://localhost:8080"

    srv = _make_srv(
        max_local=3,
        generating_queries=2,
        generating_sessions={session_id, "another-leaked"},
        dispatch_records={
            # These records are for DIFFERENT sessions — not matching our generating_sessions
            (endpoint_url, "other-session-1"): {
                "backend": endpoint_url,
                "started_at": 0.0,
                "active": True,
                "expires_at": 10**12,
            },
        },
        active_queries=1,
    )

    with patch("proxy.contention_queue.wake_all", AsyncMock()) as mock_wake:
        await _recover_stuck_generating_queries(srv)

    # Both leaked entries should be reclaimed (no matching dispatch records)
    assert srv.local_generating_queries == 0, (
        f"All leaked entries should be reclaimed; counter should be 0, got {srv.local_generating_queries}"
    )
    assert len(srv.local_generating_sessions) == 0, (
        "All leaked sessions should be removed"
    )
    mock_wake.assert_called_once()


# ---------------------------------------------------------------------------
# AC5: Existing tests still pass (no regressions)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existing_string_key_lookup_still_works():
    """Verify that the existing string-key lookup path still works correctly.

    This ensures the fix doesn't break the existing behavior where
    dispatch records use string keys (for "local" or None backends).
    """
    from proxy.router_helpers import (
        _recover_stuck_generating_queries,
    )

    srv = _make_srv(
        max_local=3,
        generating_queries=1,
        generating_sessions={"string-key-session"},
        dispatch_records={
            "string-key-session": {
                "backend": "local",
                "started_at": 0.0,
                "active": True,
                "expires_at": 10**12,
            }
        },
        active_queries=1,
    )

    await _recover_stuck_generating_queries(srv)

    assert srv.local_generating_queries == 1, (
        "String-keyed active session should be preserved"
    )
    assert "string-key-session" in srv.local_generating_sessions
