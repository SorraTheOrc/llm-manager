"""
Regression tests for generating-only counter leak self-healing.

Tests that leaked `local_generating_queries` / `local_generating_sessions`
entries (session keys stuck from an aborted decrement in the streaming
generator's finally block) are reclaimed by the dispatch cleanup loop,
allowing the local dispatch pool to self-heal without a proxy restart.

Related: LP-0MTYAWDCQ006RGYU
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_srv(
    max_local: int = 3,
    generating_queries: int = 0,
    generating_sessions: set | None = None,
    dispatch_records: dict | None = None,
    active_queries: int = 0,
    has_dispatch_records: bool = True,
):
    """Build a minimal srv fixture mimicking proxy.server state.

    Parameters set up a realistic generating-only pool that may be wedged
    by leaked session keys.
    """
    generating_sessions = generating_sessions or set()
    dispatch_records = dispatch_records or {}

    srv = SimpleNamespace(
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

    return srv


# ---------------------------------------------------------------------------
# AC1: Reproduction — leaked generating counters wedge the pool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_leaked_generating_counters_wedge_pool():
    """With max_local=3, leaking 3 session keys blocks new dispatches.

    Simulates the wedge by directly populating local_generating_queries
    and local_generating_sessions (equivalent to an aborted decrement
    in the streaming generator's finally block, router.py:1700-1801).
    """
    from proxy.router_helpers import (
        _try_acquire_local_dispatch,
    )

    srv = _make_srv(
        max_local=3,
        generating_queries=3,
        generating_sessions={"leaked-1", "leaked-2", "leaked-3"},
        dispatch_records={},  # no active dispatches
        active_queries=0,
    )

    # Pool should be wedged — no new dispatches can acquire
    acquired, owner, active, _ = await _try_acquire_local_dispatch(
        srv, max_local=3, session_key="new-session", backend="local"
    )

    assert acquired is False, (
        "New dispatch should be denied when generating count == max_local"
    )
    assert active == 3
    assert owner is not None or active == 3


# ---------------------------------------------------------------------------
# AC2: Reclaim — cleanup tick reclaims stale generating entries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_reclaims_stale_generating_entries():
    """_recover_stuck_generating_queries reclaims entries with no active dispatch records.

    Simulates 3 leaked session keys with zero active dispatches. After
    calling the recovery function, counters should be reset to 0 and
    the session set emptied.
    """
    from proxy.router_helpers import (
        _recover_stuck_generating_queries,
        _get_generating_only_count,
    )

    srv = _make_srv(
        max_local=3,
        generating_queries=3,
        generating_sessions={"leaked-1", "leaked-2", "leaked-3"},
        dispatch_records={},  # no active dispatches
        active_queries=0,
    )

    with patch("proxy.contention_queue.wake_all", AsyncMock()) as mock_wake:
        await _recover_stuck_generating_queries(srv)

    assert _get_generating_only_count(srv) == 0, (
        "Stale generating counter should be reclaimed to 0"
    )
    assert len(srv.local_generating_sessions) == 0, (
        "Stale generating sessions set should be emptied"
    )
    # Verify a WARNING was logged
    srv.logger.warning.assert_called()
    log_call = srv.logger.warning.call_args[0][0]
    assert "generating" in log_call.lower() or "recover" in log_call.lower(), (
        f"Recovery should log a WARNING; got: {log_call}"
    )
    # Verify contention queue waiters were woken
    mock_wake.assert_called()


@pytest.mark.asyncio
async def test_cleanup_reclaims_partial_leak():
    """Recovery handles partial leaks — not all entries are stale."""
    from proxy.router_helpers import (
        _recover_stuck_generating_queries,
        _get_generating_only_count,
    )

    srv = _make_srv(
        max_local=3,
        generating_queries=3,
        generating_sessions={"leaked-1", "leaked-2", "active-1"},
        dispatch_records={
            ("local", "active-1"): {
                "backend": "local",
                "started_at": 0.0,
                "active": True,
                "expires_at": 10**12,
            }
        },
        active_queries=1,
    )

    with patch("proxy.contention_queue.wake_all", AsyncMock()) as mock_wake:
        await _recover_stuck_generating_queries(srv)

    # Only the 2 leaked entries should be reclaimed; active-1 remains
    assert _get_generating_only_count(srv) == 1, (
        "Only stale entries should be reclaimed; active session preserved"
    )
    assert "active-1" in srv.local_generating_sessions, (
        "Active session should NOT be reclaimed"
    )
    assert "leaked-1" not in srv.local_generating_sessions
    assert "leaked-2" not in srv.local_generating_sessions
    # Should NOT wake waiters when no reclaim happened (stale_keys was empty)
    # Actually wait - 2 entries WERE reclaimed, so wake_all should be called
    mock_wake.assert_called()


@pytest.mark.asyncio
async def test_cleanup_no_change_when_all_entries_legitimate():
    """When all generating entries have active dispatch records, no reclaim happens."""
    from proxy.router_helpers import (
        _recover_stuck_generating_queries,
        _get_generating_only_count,
    )

    srv = _make_srv(
        max_local=3,
        generating_queries=2,
        generating_sessions={"active-1", "active-2"},
        dispatch_records={
            ("local", "active-1"): {
                "backend": "local",
                "started_at": 0.0,
                "active": True,
                "expires_at": 10**12,
            },
            ("local", "active-2"): {
                "backend": "local",
                "started_at": 0.0,
                "active": True,
                "expires_at": 10**12,
            },
        },
        active_queries=2,
    )

    await _recover_stuck_generating_queries(srv)

    assert _get_generating_only_count(srv) == 2, (
        "Legitimate generating sessions should not be reclaimed"
    )
    # No WARNING should be logged (no reclaim happened)
    # The warning call count should be 0 since no stale entries were found
    warning_calls = [c for c in srv.logger.warning.call_args_list
                     if "generating" in str(c).lower() or "recover" in str(c).lower()]
    assert len(warning_calls) == 0, (
        "No recovery log should be emitted when all entries are legitimate"
    )


# ---------------------------------------------------------------------------
# AC3: Self-healing — pool clears without restart
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pool_self_heals_after_recovery():
    """After reclaiming stale entries, new dispatches succeed.

    Reproduces the wedge (AC1), then applies recovery, and verifies
    the pool is functional again.
    """
    from proxy.router_helpers import (
        _recover_stuck_generating_queries,
        _try_acquire_local_dispatch,
    )

    srv = _make_srv(
        max_local=3,
        generating_queries=3,
        generating_sessions={"leaked-1", "leaked-2", "leaked-3"},
        dispatch_records={},
        active_queries=0,
    )

    # Step 1: Pool is wedged
    acquired, _, active, _ = await _try_acquire_local_dispatch(
        srv, max_local=3, session_key="new-session", backend="local"
    )
    assert acquired is False, "Pool should be wedged before recovery"
    assert active == 3

    # Step 2: Apply recovery
    await _recover_stuck_generating_queries(srv)

    # Step 3: New dispatch should succeed
    acquired, owner, active, _ = await _try_acquire_local_dispatch(
        srv, max_local=3, session_key="new-session", backend="local"
    )
    assert acquired is True, (
        "Pool should self-heal; new dispatch succeeds after recovery"
    )
    assert active == 1


# ---------------------------------------------------------------------------
# AC4: No false reclaim — legitimate in-flight sessions preserved
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_false_reclaim_for_active_sessions():
    """A legitimate in-flight generating session is NOT reclaimed.

    Even when the generating counter is non-zero and entries exist in
    local_generating_sessions, if every entry has an active dispatch
    record, the recovery function must leave everything intact.
    """
    from proxy.router_helpers import (
        _recover_stuck_generating_queries,
        _get_generating_only_count,
    )

    srv = _make_srv(
        max_local=3,
        generating_queries=2,
        generating_sessions={"streaming-1", "streaming-2"},
        dispatch_records={
            ("local", "streaming-1"): {
                "backend": "local",
                "started_at": 0.0,
                "active": True,
                "expires_at": 10**12,
            },
            ("local", "streaming-2"): {
                "backend": "local",
                "started_at": 0.0,
                "active": True,
                "expires_at": 10**12,
            },
        },
        active_queries=2,
    )

    initial_count = _get_generating_only_count(srv)
    initial_sessions = set(srv.local_generating_sessions)

    await _recover_stuck_generating_queries(srv)

    assert _get_generating_only_count(srv) == initial_count, (
        "Counter should be unchanged for legitimate sessions"
    )
    assert srv.local_generating_sessions == initial_sessions, (
        "Session set should be unchanged for legitimate sessions"
    )


# ---------------------------------------------------------------------------
# AC5: Legacy srv — no dispatch records attribute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovery_handles_legacy_srv_without_dispatch_records():
    """Recovery must not crash on a srv that lacks local_dispatch_records.

    Legacy state (no dispatch records system) should be handled gracefully.
    In legacy mode, all generating entries are considered stale and reclaimed
    because there is no way to distinguish legitimate from leaked.
    """
    from proxy.router_helpers import (
        _recover_stuck_generating_queries,
    )

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 60}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_generating_queries=2,
        local_generating_queries_lock=asyncio.Lock(),
        local_generating_sessions={"legacy-1", "legacy-2"},
        logger=MagicMock(),
    )
    # Note: no local_dispatch_records attribute

    # Should NOT raise
    await _recover_stuck_generating_queries(srv)

    assert srv.local_generating_queries == 0, (
        "In legacy mode, generating counter should be reset"
    )
    assert len(srv.local_generating_sessions) == 0


@pytest.mark.asyncio
async def test_recovery_handles_none_dispatch_records():
    """Recovery must not crash when local_dispatch_records is None."""
    from proxy.router_helpers import (
        _recover_stuck_generating_queries,
    )

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 60}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_generating_queries=1,
        local_generating_queries_lock=asyncio.Lock(),
        local_generating_sessions={"stale"},
        local_dispatch_records=None,
        logger=MagicMock(),
    )

    # Should NOT raise
    await _recover_stuck_generating_queries(srv)

    assert srv.local_generating_queries == 0


@pytest.mark.asyncio
async def test_recovery_reclaims_counter_only_leak():
    """A positive counter with an empty session set is reclaimed.

    Anonymous sessions increment/decrement the counter without touching
    ``local_generating_sessions``. A leak there leaves ``counter > 0``
    with an empty set — which must also self-heal.
    """
    from proxy.router_helpers import (
        _recover_stuck_generating_queries,
        _get_generating_only_count,
    )

    srv = _make_srv(
        max_local=3,
        generating_queries=3,
        generating_sessions=set(),  # counter-only leak
        dispatch_records={},  # no active dispatches
        active_queries=0,
    )

    with patch("proxy.contention_queue.wake_all", AsyncMock()) as mock_wake:
        await _recover_stuck_generating_queries(srv)

    assert _get_generating_only_count(srv) == 0, (
        "Counter-only leak should be reclaimed to 0"
    )
    srv.logger.warning.assert_called()
    mock_wake.assert_called()


@pytest.mark.asyncio
async def test_recovery_handles_missing_generating_state():
    """Recovery must gracefully handle srv without generating-only state."""
    from proxy.router_helpers import (
        _recover_stuck_generating_queries,
    )

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 60}},
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
        logger=MagicMock(),
    )
    # Note: no local_generating_queries attribute at all

    # Should NOT raise
    await _recover_stuck_generating_queries(srv)


# ---------------------------------------------------------------------------
# AC6: Recovery integrates with the dispatch cleanup loop pattern
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovery_wakes_contention_waiters():
    """Recovery must wake contention-queue waiters so they retry immediately.

    When stale entries are reclaimed, sessions waiting on the contention
    queue should be woken so they can retry the dispatch gate.
    """
    from proxy.router_helpers import (
        _recover_stuck_generating_queries,
    )

    srv = _make_srv(
        max_local=3,
        generating_queries=3,
        generating_sessions={"leaked-1", "leaked-2", "leaked-3"},
        dispatch_records={},
    )

    with patch("proxy.contention_queue.wake_all", AsyncMock()) as mock_wake:
        await _recover_stuck_generating_queries(srv)

    mock_wake.assert_called_once(), (
        "Contention queue waiters should be woken after recovery"
    )


@pytest.mark.asyncio
async def test_recovery_logs_warning_with_values():
    """Recovery log should include previous and new counter values."""
    from proxy.router_helpers import (
        _recover_stuck_generating_queries,
    )

    srv = _make_srv(
        max_local=3,
        generating_queries=3,
        generating_sessions={"leaked-1", "leaked-2", "leaked-3"},
        dispatch_records={},
    )

    with patch("proxy.contention_queue.wake_all", AsyncMock()):
        await _recover_stuck_generating_queries(srv)

    # The warning message uses %-format args; values are in call args
    warning_msg = srv.logger.warning.call_args[0][0]
    formatted = warning_msg % srv.logger.warning.call_args[0][1:]
    assert "3" in formatted, (
        f"Log should mention the previous value (3); got: {formatted}"
    )
    assert "0" in formatted, (
        f"Log should mention the new value (0); got: {formatted}"
    )


# ---------------------------------------------------------------------------
# Mixed leak + active — realistic scenario
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_reclaim_with_mixed_state():
    """Realistic scenario: some leaked entries + some active streams.

    5 session keys in generating state:
    - 3 leaked (no active dispatch record)
    - 2 active (have active dispatch records)

    After recovery: only the 3 leaked entries are reclaimed.
    """
    from proxy.router_helpers import (
        _recover_stuck_generating_queries,
        _get_generating_only_count,
    )

    srv = _make_srv(
        max_local=5,
        generating_queries=5,
        generating_sessions={"leaked-1", "leaked-2", "leaked-3", "active-1", "active-2"},
        dispatch_records={
            ("local", "active-1"): {
                "backend": "local",
                "started_at": 0.0,
                "active": True,
                "expires_at": 10**12,
            },
            ("local", "active-2"): {
                "backend": "local",
                "started_at": 0.0,
                "active": True,
                "expires_at": 10**12,
            },
        },
        active_queries=2,
    )

    with patch("proxy.contention_queue.wake_all", AsyncMock()):
        await _recover_stuck_generating_queries(srv)

    assert _get_generating_only_count(srv) == 2, (
        "Only stale entries reclaimed; active count preserved"
    )
    assert "active-1" in srv.local_generating_sessions
    assert "active-2" in srv.local_generating_sessions
    assert "leaked-1" not in srv.local_generating_sessions
    assert "leaked-2" not in srv.local_generating_sessions
    assert "leaked-3" not in srv.local_generating_sessions
    assert srv.logger.warning.called, (
        "Recovery should log when stale entries are reclaimed"
    )


# ---------------------------------------------------------------------------
# Edge: empty state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovery_noop_on_empty_state():
    """Recovery on srv with zero generating entries should be a no-op."""
    from proxy.router_helpers import (
        _recover_stuck_generating_queries,
        _get_generating_only_count,
    )

    srv = _make_srv(
        max_local=3,
        generating_queries=0,
        generating_sessions=set(),
        dispatch_records={},
    )

    await _recover_stuck_generating_queries(srv)

    assert _get_generating_only_count(srv) == 0
    assert len(srv.local_generating_sessions) == 0
