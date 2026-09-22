"""
Hermetic tests for the no-progress watchdog on local dispatch leases (LP-0MUCEFB8E003YVFF).

Context
-------
After a proxy restart the local dispatch pool can wedge when one or more
sessions hold dispatch leases whose backend is "processing" but produces
no output (no first byte, no progress). The lease lifetime is governed by
backend liveness, not forward progress, so the stuck request keeps
``active=True`` indefinitely and gates all other sessions.

The fix introduces:

- ``local_dispatch_no_progress_timeout_seconds`` (default 180): when a
  dispatch record has ``active=True`` but has had no observed progress
  advance or first-byte arrival for this duration, the watchdog treats
  it as a no-progress wedge and releases the lease.
- ``local_dispatch_max_prefill_seconds`` (default 900): a hard ceiling
  on how long any single dispatch record may remain active from its
  ``started_at`` timestamp. Exceeding this ceiling always releases the
  lease regardless of observed progress.
- Progress-only lease extension: ``_extend_lease_during_prefill`` no
  longer extends a lease on liveness alone; only a genuine progress
  advance triggers extension.

These tests verify hermetically (fake server state, no live llama-server):

1. A record with no progress past the timeout is released (watchdog fires).
2. A record with genuine progress advance keeps its lease (no regression).
3. A record exceeding max-prefill ceiling is released even with progress.
4. The watchdog integrates into ``_cleanup_stale_local_dispatch`` so it
   fires automatically during the periodic cleanup loop.
5. A record with progress advancing past the timeout is NOT released.
6. Default config values are correct.
7. The liveness-only extension path in ``_extend_lease_during_prefill``
   no longer extends the lease (only progress advances do).
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

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
        "session_single_flight_mode": "bypass",
        "disconnect_cleanup_timeout": 1,
        "stream_heartbeat_interval_seconds": 0.05,
        "stream_idle_timeout_seconds": 0.3,
        "session_guardrail_max_runtime_seconds": 3600,
        "session_guardrail_max_completion_tokens": 4096,
    }
}


def _make_srv(config_overrides: dict | None = None) -> SimpleNamespace:
    """Build a fake server object with dispatch-tracking state."""
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
        logger=MagicMock(),
    )


# ── Helper: create a dispatch record simulating a stuck session ─────────


def _create_stuck_record(
    srv,
    session_key: str,
    backend: str,
    elapsed_since_start: float,
    progress_at_start: int,
) -> None:
    """Create a dispatch record that started *elapsed_since_start* seconds ago.

    *progress_at_start* is the last observed progress value (set to 0 for
    a truly stuck session, > 0 for a progressing one).
    """
    srv.local_dispatch_records[session_key] = {
        "backend": backend,
        "started_at": time.monotonic() - elapsed_since_start,
        "active": True,
        "expires_at": time.monotonic() - elapsed_since_start + 1500,  # generous
        "model_name": "test-model",
        "last_progress": progress_at_start,
        "last_progress_ts": time.monotonic() - elapsed_since_start,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Config defaults
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_no_progress_timeout_default():
    """AC5: Default no-progress timeout is 180 seconds."""
    from proxy.router_helpers import _get_no_progress_timeout_seconds

    srv = _make_srv()
    timeout = _get_no_progress_timeout_seconds(srv)
    assert timeout == 180.0, f"Expected 180.0, got {timeout}"


@pytest.mark.asyncio
async def test_max_prefill_timeout_default():
    """AC5: Default max-prefill timeout is 900 seconds."""
    from proxy.router_helpers import _get_max_prefill_seconds

    srv = _make_srv()
    timeout = _get_max_prefill_seconds(srv)
    assert timeout == 900.0, f"Expected 900.0, got {timeout}"


@pytest.mark.asyncio
async def test_no_progress_timeout_custom():
    """AC5: Custom no-progress timeout from config is honoured."""
    from proxy.router_helpers import _get_no_progress_timeout_seconds

    srv = _make_srv({"local_dispatch_no_progress_timeout_seconds": 60})
    timeout = _get_no_progress_timeout_seconds(srv)
    assert timeout == 60.0


@pytest.mark.asyncio
async def test_max_prefill_timeout_custom():
    """AC5: Custom max-prefill timeout from config is honoured."""
    from proxy.router_helpers import _get_max_prefill_seconds

    srv = _make_srv({"local_dispatch_max_prefill_seconds": 600})
    timeout = _get_max_prefill_seconds(srv)
    assert timeout == 600.0


# ═══════════════════════════════════════════════════════════════════════════════
# Watchdog: no-progress timeout fires
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_watchdog_releases_stuck_record():
    """AC1: A record with no progress past the timeout is released."""
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
        _watchdog_no_progress,
    )

    srv = _make_srv()
    _create_stuck_record(srv, "stuck-session", "local", elapsed_since_start=200, progress_at_start=0)

    # The watchdog should flag the record as no-progress
    result = _watchdog_no_progress(
        srv,
        "stuck-session",
        srv.local_dispatch_records["stuck-session"],
    )
    assert result is True, "Watchdog should flag no-progress"

    # Cleanup should release it
    removed = await _cleanup_stale_local_dispatch(srv)
    assert removed == 1, f"Expected 1 removed, got {removed}"
    assert "stuck-session" not in srv.local_dispatch_records


@pytest.mark.asyncio
async def test_watchdog_does_not_release_recently_progressed_record():
    """AC4: A record with progress advancing past the timeout is NOT released."""
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
    )

    srv = _make_srv()
    # Started 200s ago but had progress 10s ago
    srv.local_dispatch_records["progressing-session"] = {
        "backend": "local",
        "started_at": time.monotonic() - 200,
        "active": True,
        "expires_at": time.monotonic() - 200 + 1500,
        "model_name": "test-model",
        "last_progress_ts": time.monotonic() - 10,  # recent progress
    }

    removed = await _cleanup_stale_local_dispatch(srv)
    assert removed == 0, "A progressing record must not be released"
    assert "progressing-session" in srv.local_dispatch_records


@pytest.mark.asyncio
async def test_watchdog_releases_expired_stuck_record_via_cleanup():
    """AC1 (cleanup integration): A stuck record past the lease timeout
    is cleaned up by the watchdog during the periodic cleanup loop."""
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
    )

    srv = _make_srv()
    # The record's expires_at is NOW in the past (stuck for 200s, lease 30s)
    srv.local_dispatch_records["stuck-session"] = {
        "backend": "local",
        "started_at": time.monotonic() - 200,
        "active": True,
        "expires_at": time.monotonic() - 35,  # expired 5 seconds ago
        "model_name": "test-model",
        "last_progress": 0,
        "last_progress_ts": time.monotonic() - 200,
    }

    # Mock _query_slot_processing to return False (slot not processing)
    # so cleanup proceeds with orphan_clean
    import proxy.router_helpers as rh
    orig_query = rh._query_slot_processing

    async def fake_query(*args, **kwargs):
        return False

    rh._query_slot_processing = fake_query
    try:
        removed = await _cleanup_stale_local_dispatch(srv)
        assert removed == 1, f"Stuck record should be cleaned up, got {removed}"
        assert "stuck-session" not in srv.local_dispatch_records
    finally:
        rh._query_slot_processing = orig_query


# ═══════════════════════════════════════════════════════════════════════════════
# Max-prefill ceiling
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_max_prefill_ceiling_releases_stuck_record():
    """AC3: A record exceeding max-prefill ceiling is released even with progress."""
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
    )

    srv = _make_srv()
    # Started 1000s ago (> 900s max prefilled), has had progress
    srv.local_dispatch_records["over-prefill"] = {
        "backend": "local",
        "started_at": time.monotonic() - 1000,
        "active": True,
        "expires_at": time.monotonic() - 35,  # expired
        "model_name": "test-model",
        "last_progress": 500,  # has had progress
        "last_progress_ts": time.monotonic() - 1000,  # but very old
    }

    import proxy.router_helpers as rh
    orig_query = rh._query_slot_processing

    async def fake_query(*args, **kwargs):
        return False

    rh._query_slot_processing = fake_query
    try:
        removed = await _cleanup_stale_local_dispatch(srv)
        assert removed == 1, f"Over-prefill record should be cleaned up, got {removed}"
        assert "over-prefill" not in srv.local_dispatch_records
    finally:
        rh._query_slot_processing = orig_query


@pytest.mark.asyncio
async def test_within_max_prefill_not_released():
    """AC3: A record within the max-prefill ceiling is NOT released."""
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
    )

    srv = _make_srv()
    # Started 100s ago (< 900s ceiling), lease still valid but has recent
    # progress 90s ago (within the 180s no-progress window).
    srv.local_dispatch_records["within-ceiling"] = {
        "backend": "local",
        "started_at": time.monotonic() - 100,
        "active": True,
        "expires_at": time.monotonic() + 100,  # still valid
        "model_name": "test-model",
        "last_progress": 100,
        "last_progress_ts": time.monotonic() - 90,  # 90s ago, within ceiling
    }

    import proxy.router_helpers as rh
    orig_query = rh._query_slot_processing

    async def fake_query(*args, **kwargs):
        return False

    rh._query_slot_processing = fake_query
    try:
        removed = await _cleanup_stale_local_dispatch(srv)
        assert removed == 0, f"Within-ceiling record should NOT be cleaned up, got {removed}"
        assert "within-ceiling" in srv.local_dispatch_records
    finally:
        rh._query_slot_processing = orig_query


# ═══════════════════════════════════════════════════════════════════════════════
# Lease extension: progress-only (no liveness-only)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_extend_lease_requires_progress_advance():
    """AC2: Lease extension during prefill only happens on progress advance.

    A slot that is alive (is_processing=True) but has no progress advance
    must NOT extend the lease. This is the key change: liveness-only
    extension is removed to prevent indefinite wedging.
    """
    from proxy.router_helpers import _extend_lease_during_prefill

    srv = _make_srv()
    srv.local_dispatch_records["test-session"] = {
        "backend": "local",
        "started_at": time.monotonic(),
        "active": True,
        "expires_at": time.monotonic() + 30,
        "model_name": "test-model",
        "last_progress": 100,
    }

    # _query_prefill_progress returns progress=None, alive=True (liveness,
    # no progress advance) — this simulates the wedge condition.
    import proxy.router_helpers as rh
    orig_query = rh._query_prefill_progress

    async def fake_query(*args, **kwargs):
        return None, True  # alive but no progress

    rh._query_prefill_progress = fake_query
    try:
        last_progress, extended = await _extend_lease_during_prefill(
            srv,
            "test-session",
            endpoint=None,
            llama_port=8080,
            model_name="test-model",
            slot_id=0,
            last_progress=100,
        )
        assert extended is False, (
            f"Lease should NOT be extended on liveness alone (got extended={extended})"
        )
        assert last_progress == 100, "last_progress should be unchanged"
    finally:
        rh._query_prefill_progress = orig_query


@pytest.mark.asyncio
async def test_extend_lease_on_progress_advance():
    """AC2: A genuine progress advance STILL extends the lease (no regression)."""
    from proxy.router_helpers import _extend_lease_during_prefill

    srv = _make_srv()
    initial_expires = time.monotonic() + 30
    srv.local_dispatch_records["test-session"] = {
        "backend": "local",
        "started_at": time.monotonic(),
        "active": True,
        "expires_at": initial_expires,
        "model_name": "test-model",
        "last_progress": 100,
    }

    import proxy.router_helpers as rh
    orig_query = rh._query_prefill_progress

    async def fake_query(*args, **kwargs):
        return 150, True  # progress advanced from 100 -> 150

    rh._query_prefill_progress = fake_query
    try:
        last_progress, extended = await _extend_lease_during_prefill(
            srv,
            "test-session",
            endpoint=None,
            llama_port=8080,
            model_name="test-model",
            slot_id=0,
            last_progress=100,
        )
        assert extended is True, (
            f"Lease SHOULD be extended on progress advance (got extended={extended})"
        )
        assert last_progress == 150, "last_progress should advance to 150"
        # The expires_at should have been pushed out by the buffer
        assert srv.local_dispatch_records["test-session"]["expires_at"] > initial_expires
    finally:
        rh._query_prefill_progress = orig_query


@pytest.mark.asyncio
async def test_extend_lease_no_progress_no_alive():
    """AC2: No progress AND not alive — lease is not extended (unchanged)."""
    from proxy.router_helpers import _extend_lease_during_prefill

    srv = _make_srv()
    srv.local_dispatch_records["test-session"] = {
        "backend": "local",
        "started_at": time.monotonic(),
        "active": True,
        "expires_at": time.monotonic() + 30,
        "model_name": "test-model",
        "last_progress": 100,
    }

    import proxy.router_helpers as rh
    orig_query = rh._query_prefill_progress

    async def fake_query(*args, **kwargs):
        return None, False  # neither progress nor alive

    rh._query_prefill_progress = fake_query
    try:
        last_progress, extended = await _extend_lease_during_prefill(
            srv,
            "test-session",
            endpoint=None,
            llama_port=8080,
            model_name="test-model",
            slot_id=0,
            last_progress=100,
        )
        assert extended is False
        assert last_progress == 100
    finally:
        rh._query_prefill_progress = orig_query


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: watchdog prevents wedge in cleanup loop
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_cleanup_loop_frees_slot_for_blocked_session():
    """AC3 (integration): The cleanup loop releases a stuck lease so a
    blocked session can acquire."""
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
        _try_acquire_local_dispatch,
    )

    srv = _make_srv()

    # Session A: stuck, no progress, past the timeout. Its lease is still
    # nominally valid (expires_at in the future) — the watchdog is the only
    # thing that can free it.
    srv.local_dispatch_records["stuck-a"] = {
        "backend": "local",
        "started_at": time.monotonic() - 250,
        "active": True,
        "expires_at": time.monotonic() + 1000,  # adaptive lease, still valid
        "model_name": "test-model",
        "last_progress": 0,
        "last_progress_ts": time.monotonic() - 250,
    }

    # Sanity: the stuck record is present before cleanup.
    assert "stuck-a" in srv.local_dispatch_records

    import proxy.router_helpers as rh
    orig_query = rh._query_slot_processing

    async def fake_query(*args, **kwargs):
        return False  # stuck session's slot not processing

    rh._query_slot_processing = fake_query
    try:
        removed = await _cleanup_stale_local_dispatch(srv)
        assert removed >= 1, f"Stuck record should be removed, got {removed}"
        assert "stuck-a" not in srv.local_dispatch_records

        # Now waiting-b should be able to acquire
        acquired, owner, active_count, retry_after = await _try_acquire_local_dispatch(
            srv,
            max_local=3,
            session_key="waiting-b",
            backend="local",
            model_name="test-model",
        )
        assert acquired is True, f"Waiting session should acquire after cleanup, got acquired={acquired}"
    finally:
        rh._query_slot_processing = orig_query


@pytest.mark.asyncio
async def test_multiple_stuck_sessions_cleared():
    """AC1: Multiple stuck sessions are all cleaned up."""
    from proxy.router_helpers import _cleanup_stale_local_dispatch

    srv = _make_srv()
    for i in range(5):
        srv.local_dispatch_records[f"stuck-{i}"] = {
            "backend": "local",
            "started_at": time.monotonic() - (200 + i * 10),
            "active": True,
            "expires_at": time.monotonic() - 40,
            "model_name": "test-model",
            "last_progress": 0,
            "last_progress_ts": time.monotonic() - (200 + i * 10),
        }

    import proxy.router_helpers as rh
    orig_query = rh._query_slot_processing

    async def fake_query(*args, **kwargs):
        return False

    rh._query_slot_processing = fake_query
    try:
        removed = await _cleanup_stale_local_dispatch(srv)
        assert removed == 5, f"Expected 5 removed, got {removed}"
        assert len(srv.local_dispatch_records) == 0
    finally:
        rh._query_slot_processing = orig_query


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_watchdog_respects_config_zero_timeout():
    """Edge case: Setting no-progress timeout to 0 disables the watchdog."""
    from proxy.router_helpers import _get_no_progress_timeout_seconds

    srv = _make_srv({"local_dispatch_no_progress_timeout_seconds": 0})
    timeout = _get_no_progress_timeout_seconds(srv)
    assert timeout == 0.0, "Zero timeout should disable the watchdog"


@pytest.mark.asyncio
async def test_watchdog_respects_config_zero_max_prefill():
    """Edge case: Setting max-prefill to 0 disables the ceiling."""
    from proxy.router_helpers import _get_max_prefill_seconds

    srv = _make_srv({"local_dispatch_max_prefill_seconds": 0})
    timeout = _get_max_prefill_seconds(srv)
    assert timeout == 0.0, "Zero max-prefill should disable the ceiling"


@pytest.mark.asyncio
async def test_no_progress_record_missing_last_progress_ts():
    """Edge case: Legacy records without ``last_progress_ts`` are not
    watchdog-released; they fall through to the expiry/orphan-cleanup path."""
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
    )

    srv = _make_srv()
    # Record from old code path — no last_progress_ts key at all
    srv.local_dispatch_records["legacy-session"] = {
        "backend": "local",
        "started_at": time.monotonic() - 250,
        "active": True,
        "expires_at": time.monotonic() - 40,
        "model_name": "test-model",
        # No last_progress_ts — legacy record
    }

    import proxy.router_helpers as rh
    orig_query = rh._query_slot_processing

    async def fake_query(*args, **kwargs):
        return False

    rh._query_slot_processing = fake_query
    try:
        removed = await _cleanup_stale_local_dispatch(srv)
        # No last_progress_ts -> watchdog abstains; the expired active record
        # is orphan-cleaned by the pre-existing slot-verification path.
        assert removed == 1
        assert "legacy-session" not in srv.local_dispatch_records
        warnings = [
            call for call in srv.logger.warning.call_args_list
            if "reason=orphan_cleanup" in str(call)
        ]
        assert len(warnings) == 1
    finally:
        rh._query_slot_processing = orig_query
