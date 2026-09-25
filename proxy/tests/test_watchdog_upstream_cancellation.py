"""
Unit tests for upstream stream cancellation on watchdog release (LP-0MUH0OXJE006ERB9 / LP-0MUGQHPJ50046LV8).

Context
-------
When the no-progress watchdog releases a local dispatch record, the in-flight
upstream httpx stream to llama-server must be cancelled so the slot and GPU are
actually freed. Previously only proxy-side bookkeeping was cleaned up, leaving
the orphaned prefill wedging the slot indefinitely.

These tests verify hermetically (no live llama-server):

1. stream_cm is stored on the dispatch record when a stream opens.
2. The watchdog closes the upstream stream when it releases a record.
3. Counter cleanliness: local_active_queries returns to 0 without reconciliation.
4. Idempotent cancellation: safe against the completion race.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

BASE_SERVER_CONFIG = {
    "server": {
        "llama_router_mode": False,
        "llama_server_port": 8080,
        "max_concurrent_queries": 4,
        "session_slot_pool_size": 3,
        "llama_request_timeout": 30,
        "local_dispatch_lease_timeout_seconds": 30,
        "local_dispatch_lease_per_token_seconds": 0.015,
        "local_dispatch_lease_max_seconds": 1500,
        "local_dispatch_lease_prefill_poll_seconds": 10,
        "local_dispatch_lease_prefill_buffer_seconds": 30,
        "local_dispatch_lease_chunk_refresh_buffer_seconds": 30,
        "local_dispatch_no_progress_timeout_seconds": 180,
        "local_dispatch_max_prefill_seconds": 900,
        "session_single_flight_mode": "bypass",
        "disconnect_cleanup_timeout": 1,
        "stream_heartbeat_interval_seconds": 0.05,
        "stream_idle_timeout_seconds": 0.3,
        "session_guardrail_max_runtime_seconds": 3600,
        "session_guardrail_max_completion_tokens": 4096,
    }
}


# ── Helper: build a fake server with dispatch-tracking state ──────────


def _make_srv(config_overrides: dict | None = None) -> SimpleNamespace:
    cfg = dict(BASE_SERVER_CONFIG)
    if config_overrides:
        cfg.update(config_overrides)
    return SimpleNamespace(
        config={"server": cfg},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
        local_generating_queries=0,
        local_generating_sessions=set(),
        local_generating_queries_lock=asyncio.Lock(),
        local_prefill_in_flight={},
        local_prefill_in_flight_lock=asyncio.Lock(),
        logger=MagicMock(),
    )


def _create_dispatch_record(
    srv,
    session_key: str,
    backend: str,
    elapsed_since_start: float,
    progress_at_start: int,
) -> dict:
    """Create a dispatch record and return a reference to it."""
    record = {
        "backend": backend,
        "started_at": time.monotonic() - elapsed_since_start,
        "active": True,
        "expires_at": time.monotonic() - elapsed_since_start + 1500,
        "model_name": "test-model",
        "last_progress": progress_at_start,
        "last_progress_ts": time.monotonic() - elapsed_since_start,
    }
    srv.local_dispatch_records[session_key] = record
    return record


# ═══════════════════════════════════════════════════════════════════════════════
# AC1: stream_cm stored on dispatch record, cancelled on watchdog release
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_stream_cm_stored_on_dispatch_record():
    """AC1: The stream context manager is pinned on the dispatch record."""
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
    )

    srv = _make_srv()

    # Create a dispatch record with a stream_cm attached (simulates what
    # router.py does after the stream opens).
    mock_cm = AsyncMock()
    mock_cm.__aexit__ = AsyncMock()
    record = _create_dispatch_record(
        srv, "session-a", "local",
        elapsed_since_start=200,
        progress_at_start=0,
    )
    record["stream_cm"] = mock_cm

    # The stream_cm is on the record
    assert srv.local_dispatch_records["session-a"]["stream_cm"] is mock_cm

    # Watchdog should flag the record as no-progress
    from proxy.router_helpers import _watchdog_no_progress
    assert _watchdog_no_progress(
        srv,
        "session-a",
        record,
    ) is True

    # Cleanup releases the record and should close the stream
    await _cleanup_stale_local_dispatch(srv)

    assert "session-a" not in srv.local_dispatch_records
    mock_cm.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_watchdog_closes_upstream_stream():
    """AC1: The watchdog cancels the in-flight upstream request."""
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
    )

    srv = _make_srv()
    aexit_called = False

    class TrackedCM:
        """A stream context manager that tracks __aexit__ invocations."""
        async def __aenter__(self):
            return None
        async def __aexit__(self, *args):
            nonlocal aexit_called
            aexit_called = True

    record = _create_dispatch_record(
        srv, "session-b", "local",
        elapsed_since_start=200,
        progress_at_start=0,
    )
    record["stream_cm"] = TrackedCM()

    removed = await _cleanup_stale_local_dispatch(srv)
    assert removed == 1
    assert aexit_called, "Stream __aexit__ should be called to cancel upstream"


# ═══════════════════════════════════════════════════════════════════════════════
# AC3: counter cleanliness on watchdog release
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_watchdog_counter_clean():
    """AC3: local_active_queries returns to 0 on watchdog release."""
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
    )

    srv = _make_srv()

    # Set up: simulate two active queries, one of which is wedged.
    srv.local_active_queries = 2
    _create_dispatch_record(
        srv, "stuck-session", "local",
        elapsed_since_start=200,
        progress_at_start=0,
    )

    removed = await _cleanup_stale_local_dispatch(srv)
    assert removed == 1

    # After watchdog release, local_active_queries should be decremented
    # (2 → 1, because we only removed one record).
    assert srv.local_active_queries == 1

    # No reconciliation warning should be emitted
    warnings = [
        call for call in srv.logger.warning.call_args_list
        if "counter recovered" in str(call).lower()
        and "no_progress_watchdog" in str(call).lower()
    ]
    # We should not see a reconciliation warning for the watchdog path
    # (the counter is correct by construction).
    for w in srv.logger.warning.call_args_list:
        assert "counter recovered" not in str(w).lower() or "no_progress" not in str(w).lower(), (
            f"Unexpected reconciliation warning for watchdog: {w}"
        )


@pytest.mark.asyncio
async def test_watchdog_counter_clean_single_query():
    """AC3: local_active_queries goes to 0 when only one query is wedged."""
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
    )

    srv = _make_srv()
    srv.local_active_queries = 1
    _create_dispatch_record(
        srv, "stuck-session", "local",
        elapsed_since_start=200,
        progress_at_start=0,
    )

    await _cleanup_stale_local_dispatch(srv)

    assert srv.local_active_queries == 0
    # Verify no reconciliation warning
    for w in srv.logger.warning.call_args_list:
        assert "counter recovered" not in str(w).lower(), (
            f"Unexpected reconciliation warning: {w}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AC5: idempotent cancellation (safe against completion race)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_watchdog_idempotent_stream_close():
    """AC5: Cancelling an already-closed stream does not raise."""
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
    )

    srv = _make_srv()

    class AlreadyClosedCM:
        """A stream context manager that simulates an already-closed stream."""
        async def __aenter__(self):
            return None
        async def __aexit__(self, *args):
            # Simulates the stream already being closed (e.g. normal completion).
            pass

    record = _create_dispatch_record(
        srv, "session-c", "local",
        elapsed_since_start=200,
        progress_at_start=0,
    )
    record["stream_cm"] = AlreadyClosedCM()

    # Should not raise
    await _cleanup_stale_local_dispatch(srv)
    assert "session-c" not in srv.local_dispatch_records


@pytest.mark.asyncio
async def test_watchdog_idempotent_no_stream_cm():
    """AC5: Watchdog handles records that never had a stream_cm."""
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
    )

    srv = _make_srv()
    record = _create_dispatch_record(
        srv, "session-d", "local",
        elapsed_since_start=200,
        progress_at_start=0,
    )
    # No stream_cm key — e.g. record created before the fix.

    # Should not raise and should still clean up
    removed = await _cleanup_stale_local_dispatch(srv)
    assert removed == 1
    assert "session-d" not in srv.local_dispatch_records


@pytest.mark.asyncio
async def test_watchdog_no_double_decrement():
    """AC5: Counter is decremented exactly once per watchdog release."""
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
    )

    srv = _make_srv()
    srv.local_active_queries = 3
    # Two wedged records
    _create_dispatch_record(
        srv, "stuck-1", "local",
        elapsed_since_start=200,
        progress_at_start=0,
    )
    _create_dispatch_record(
        srv, "stuck-2", "local",
        elapsed_since_start=210,
        progress_at_start=0,
    )

    await _cleanup_stale_local_dispatch(srv)

    # Two records removed, counter decremented twice: 3 → 1
    assert srv.local_active_queries == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Deferred cleanup: slot freeing and preflight tracking
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_watchdog_clears_preflight_tracking():
    """Deferred: preflight tracking is cleared for watchdog-released records."""
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
    )

    srv = _make_srv()
    srv.local_prefill_in_flight["stuck-session"] = True

    record = _create_dispatch_record(
        srv, "stuck-session", "local",
        elapsed_since_start=200,
        progress_at_start=0,
    )
    record["stream_cm"] = AsyncMock()

    await _cleanup_stale_local_dispatch(srv)

    assert "stuck-session" not in srv.local_prefill_in_flight


@pytest.mark.asyncio
async def test_watchdog_free_slot_assignment():
    """Deferred: _free_slot_assignment is called for watchdog-released records."""
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
    )

    srv = _make_srv()

    free_called = False

    def mock_free_slot(session_key, endpoint=None):
        nonlocal free_called
        free_called = True

    record = _create_dispatch_record(
        srv, "stuck-session", "local",
        elapsed_since_start=200,
        progress_at_start=0,
    )
    record["stream_cm"] = AsyncMock()

    with patch("proxy.session._free_slot_assignment", side_effect=mock_free_slot):
        await _cleanup_stale_local_dispatch(srv)

    assert free_called, "_free_slot_assignment should be called"


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: watchdog does not affect progressing records
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_progressing_record_not_released():
    """Regression: a record with recent progress keeps its stream_cm."""
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
    )

    srv = _make_srv()

    mock_cm = AsyncMock()
    mock_cm.__aexit__ = AsyncMock()
    record = _create_dispatch_record(
        srv, "progressing-session", "local",
        elapsed_since_start=200,
        progress_at_start=500,
    )
    record["stream_cm"] = mock_cm
    # Recent progress — 10s ago
    record["last_progress_ts"] = time.monotonic() - 10

    removed = await _cleanup_stale_local_dispatch(srv)
    assert removed == 0
    assert "progressing-session" in srv.local_dispatch_records
    assert srv.local_dispatch_records["progressing-session"]["stream_cm"] is mock_cm
    mock_cm.__aexit__.assert_not_awaited()
