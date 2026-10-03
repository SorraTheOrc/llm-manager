"""
Integration tests for watchdog recovery with stalled backend (LP-0MUH0OUP5007ZICU / LP-0MUGQHPJ50046LV8).

Context
-------
After the unit fix (LP-0MUH0OXJE006ERB9), the watchdog now closes the upstream
httpx stream when it releases a wedged dispatch. These integration tests
verify the end-to-end recovery: the slot becomes genuinely free and subsequent
dispatches succeed without a proxy restart.

Tests:

1. AC2: Stalled-fake-backend integration — a record with a stalled stream_cm
   is cancelled by the watchdog, counter is clean, and a subsequent dispatch
   succeeds.
2. AC4: Regression — reproduces the 2026-09-25 incident shape (68k+ token
   prefill, no progress, watchdog fires) and asserts bounded recovery.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Shared config: short timeouts for fast test execution ─────────────


def _make_base_config(no_progress: float = 5.0, max_prefill: float = 300.0) -> dict:
    """Return a minimal server config with short timeouts for testing."""
    return {
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
            "local_dispatch_no_progress_timeout_seconds": no_progress,
            "local_dispatch_max_prefill_seconds": max_prefill,
            "session_single_flight_mode": "bypass",
            "disconnect_cleanup_timeout": 1,
            "stream_heartbeat_interval_seconds": 0.05,
            "stream_idle_timeout_seconds": 0.3,
            "session_guardrail_max_runtime_seconds": 3600,
            "session_guardrail_max_completion_tokens": 4096,
        }
    }


def _make_srv(config: dict | None = None) -> SimpleNamespace:
    """Build a fake server object with dispatch-tracking state."""
    cfg = _make_base_config()
    if config:
        cfg["server"].update(config)
    return SimpleNamespace(
        config={"server": cfg["server"]},
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


def _create_stalled_record(
    srv,
    session_key: str,
    elapsed_since_start: float,
) -> dict:
    """Create a dispatch record with a stalled (never-progressing) stream_cm.

    Simulates the 2026-09-25 incident: a huge prompt (68k tokens) that
    stalled during prefill — no progress for the entire duration.
    """
    record = {
        "backend": "local",
        "started_at": time.monotonic() - elapsed_since_start,
        "active": True,
        "expires_at": time.monotonic() + 1500,  # adaptive lease, still valid
        "model_name": "test-model",
        "last_progress": 0,
        "last_progress_ts": time.monotonic() - elapsed_since_start,
        "stream_cm": AsyncMock(),  # stalled stream — never produces data
    }
    srv.local_dispatch_records[session_key] = record
    srv.local_prefill_in_flight[session_key] = True
    srv.local_active_queries += 1
    return record


# ═══════════════════════════════════════════════════════════════════════════════
# AC2: Stalled-fake-backend integration — watchdog releases, slot recovers
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_watchdog_releases_stalled_backend_and_slot_recovers():
    """AC2: A record with a stalled stream is cancelled; slot recovers.

    Simulates a stalled fake backend: a dispatch record exists with an
    in-flight httpx stream that produces no data (simulating the 68k-token
    prefill that wedged on 2026-09-25). The watchdog fires, closes the
    stream, and a subsequent dispatch succeeds.
    """
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
        _try_acquire_local_dispatch,
    )

    # Short timeout so the watchdog fires quickly
    srv = _make_srv({"local_dispatch_no_progress_timeout_seconds": 5})

    # Set up: one session with a stalled dispatch (like the 2026-09-25 incident)
    record = _create_stalled_record(srv, "stalled-session", elapsed_since_start=10)
    aexit_mock = record["stream_cm"].__aexit__
    aexit_mock.reset_mock()

    # Before cleanup: slot is wedged
    assert len(srv.local_dispatch_records) == 1
    assert srv.local_active_queries == 1

    # Watchdog fires during cleanup
    removed = await _cleanup_stale_local_dispatch(srv)
    assert removed == 1, "Watchdog should have released the stalled record"

    # Stream was cancelled
    aexit_mock.assert_awaited_once()

    # Slot is genuinely free
    assert len(srv.local_dispatch_records) == 0
    assert srv.local_active_queries == 0
    assert "stalled-session" not in srv.local_prefill_in_flight

    # No reconciliation warning needed — counters are correct by construction
    for call in srv.logger.warning.call_args_list:
        assert "counter recovered" not in str(call).lower()

    # Subsequent dispatch succeeds (no proxy restart needed)
    acquired, owner, active_count, retry_after = await _try_acquire_local_dispatch(
        srv,
        max_local=3,
        session_key="new-session",
        backend="local",
        model_name="test-model",
    )
    assert acquired is True, "New session should acquire after watchdog recovery"
    assert active_count == 1


@pytest.mark.asyncio
async def test_watchdog_recovery_multiple_stalled_slots():
    """AC2 (multi-slot): Multiple stalled sessions are all freed; dispatch recovers.

    Simulates the 2026-09-25 incident with 3 full slots wedged.
    """
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
        _try_acquire_local_dispatch,
    )

    srv = _make_srv({"local_dispatch_no_progress_timeout_seconds": 5})

    # 3 wedged slots (matching the incident)
    for i in range(3):
        _create_stalled_record(srv, f"stalled-{i}", elapsed_since_start=10 + i)

    assert srv.local_active_queries == 3
    assert len(srv.local_dispatch_records) == 3

    # Watchdog cleans all of them
    removed = await _cleanup_stale_local_dispatch(srv)
    assert removed == 3

    # All counters clean
    assert srv.local_active_queries == 0
    assert len(srv.local_dispatch_records) == 0

    # Dispatch recovers
    acquired, _, active_count, _ = await _try_acquire_local_dispatch(
        srv,
        max_local=3,
        session_key="recovering-session",
        backend="local",
        model_name="test-model",
    )
    assert acquired is True
    assert active_count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# AC4: Regression — 2026-09-25 incident shape
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_regression_2026_09_25_68k_prefill_watchdog_recovery():
    """AC4: Regression for the 2026-09-25 incident.

    The incident:
    - A huge 68k+ token prompt caused a long prefill that stalled at ~49%
    - The watchdog fired after 180s of no progress
    - The proxy only released bookkeeping; the upstream stream kept running
    - The slot remained wedged until a manual proxy restart

    The fix: the watchdog now closes the upstream stream, freeing the slot.
    """
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
        _try_acquire_local_dispatch,
    )

    srv = _make_srv({"local_dispatch_no_progress_timeout_seconds": 180})

    # Simulate the 2026-09-25 shape: a huge prompt (68k tokens) that
    # stalled during prefill. The prefill had progress markers up to ~49%
    # but then stopped.
    # In the real incident, this lasted from 07:44 to 07:59 (15 minutes
    # of no progress before the watchdog fired).
    record = _create_stalled_record(
        srv,
        "incident-session",
        elapsed_since_start=1000,  # 1000s since start, well past 180s watchdog
    )
    # The last progress was at ~49% (the incident log showed progress at
    # 0.253600 → 0.373463 → 0.493327 before stalling)
    record["last_progress"] = 49
    record["last_progress_ts"] = time.monotonic() - 1000  # very old

    assert srv.local_active_queries == 1
    aexit_mock = record["stream_cm"].__aexit__
    aexit_mock.reset_mock()

    # Bounded recovery: watchdog fires
    removed = await _cleanup_stale_local_dispatch(srv)
    assert removed == 1, "Watchdog should release the wedged record"

    # Stream was cancelled — slot is freed
    aexit_mock.assert_awaited_once()
    assert srv.local_active_queries == 0
    assert "incident-session" not in srv.local_dispatch_records

    # No restart required: dispatch recovers
    acquired, _, active_count, _ = await _try_acquire_local_dispatch(
        srv,
        max_local=3,
        session_key="recovered-session",
        backend="local",
        model_name="test-model",
    )
    assert acquired is True, "Dispatch should succeed without proxy restart"

    # Bounded recovery: no reconciliation warning
    reconciliation_warnings = [
        call for call in srv.logger.warning.call_args_list
        if "counter recovered" in str(call).lower()
    ]
    assert len(reconciliation_warnings) == 0, (
        "Watchdog release should be counter-clean — no reconciliation needed"
    )


@pytest.mark.asyncio
async def test_regression_2026_09_25_max_prefill_ceiling():
    """AC4 (alternate path): The 68k prefill also exceeds max-prefill ceiling.

    Even if the no-progress watchdog were disabled, the max-prefill ceiling
    (default 900s) would catch the long-running prefill.
    """
    from proxy.router_helpers import _cleanup_stale_local_dispatch

    # Max-prefill ceiling of 500s — the 1000s prefill exceeds it
    srv = _make_srv(
        {
            "local_dispatch_no_progress_timeout_seconds": 0,  # disable no-progress
            "local_dispatch_max_prefill_seconds": 500,
        }
    )

    record = _create_stalled_record(
        srv, "over-prefill-session",
        elapsed_since_start=1000,  # 1000s > 500s ceiling
    )
    record["last_progress"] = 49
    record["last_progress_ts"] = time.monotonic() - 1000

    removed = await _cleanup_stale_local_dispatch(srv)
    assert removed == 1, "Max-prefill ceiling should release the over-age record"
    assert srv.local_active_queries == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Regression: progressing records are not affected
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_progressing_records_preserved_with_stream_cm():
    """Regression: a record that is progressing keeps its stream_cm intact.

    Even with the fix, genuine long prefills that advance must not be aborted.
    """
    from proxy.router_helpers import _cleanup_stale_local_dispatch

    srv = _make_srv({"local_dispatch_no_progress_timeout_seconds": 5})

    record = _create_stalled_record(
        srv, "progressing-session",
        elapsed_since_start=10,
    )
    record["last_progress"] = 500  # had progress
    record["last_progress_ts"] = time.monotonic() - 2  # recent progress

    removed = await _cleanup_stale_local_dispatch(srv)
    assert removed == 0, "Progressing record must not be released"
    assert "progressing-session" in srv.local_dispatch_records
    assert "stream_cm" in srv.local_dispatch_records["progressing-session"]


@pytest.mark.asyncio
async def test_watchdog_does_not_weaken_lease_extension():
    """AC non-goal: The progress-aware lease extension is not weakened.

    A record that has genuine progress advance must keep its lease, even
    if its total lifetime is long.
    """
    from proxy.router_helpers import _cleanup_stale_local_dispatch

    # Long no-progress timeout but record has recent progress
    srv = _make_srv({"local_dispatch_no_progress_timeout_seconds": 300})

    record = _create_stalled_record(
        srv, "long-prefill-session",
        elapsed_since_start=100,
    )
    record["last_progress_ts"] = time.monotonic() - 10  # recent progress

    removed = await _cleanup_stale_local_dispatch(srv)
    assert removed == 0, "Record with recent progress must not be released"
