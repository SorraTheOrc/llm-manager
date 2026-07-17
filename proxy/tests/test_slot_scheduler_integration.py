"""
Integration tests for router + JobScheduler interaction.

Tests cover:
- Scheduler initialization from config
- Scheduler slot assignment in router flow
- Queue overflow → 503
- Job re-entry → same slot
- Session end → slot release
"""

import asyncio
from unittest.mock import MagicMock

import pytest
from proxy.router import _get_job_scheduler
from proxy.slot_scheduler import JobScheduler

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(autouse=True)
def reset_scheduler_globals():
    """Reset the global scheduler state between tests."""
    global _job_scheduler, _job_scheduler_initialized
    import proxy.router as router_mod
    router_mod._job_scheduler = None
    router_mod._job_scheduler_initialized = False
    yield


@pytest.fixture
def mock_server_config():
    """Create a mock server config with slot_management enabled."""
    return {
        "server": {
            "slot_management": {
                "slot_pool_size": 4,
                "slot_queue_max_depth": 16,
                "slot_job_timeout_seconds": 300.0,
                "slot_queue_overflow_retry_after": 900,
            }
        }
    }


# ===================================================================
# Scheduler initialization from config
# ===================================================================


class TestSchedulerInit:
    """Tests for scheduler initialization from config."""

    def test_no_slot_management_config_returns_none(self, monkeypatch):
        """When slot_management is absent, _get_job_scheduler() returns None."""
        monkeypatch.setattr(
            "proxy.router._srv",
            lambda: type("Srv", (), {
                "config": {"server": {}},
                "logger": MagicMock(),
            })()
        )
        result = _get_job_scheduler()
        assert result is None

    def test_empty_slot_management_returns_none(self, monkeypatch):
        """Empty slot_management config returns None."""
        monkeypatch.setattr(
            "proxy.router._srv",
            lambda: type("Srv", (), {
                "config": {"server": {"slot_management": {}}},
                "logger": MagicMock(),
            })()
        )
        result = _get_job_scheduler()
        assert result is None

    def test_pool_size_zero_returns_none(self, monkeypatch):
        """Pool size of 0 returns None."""
        monkeypatch.setattr(
            "proxy.router._srv",
            lambda: type("Srv", (), {
                "config": {"server": {"slot_management": {"slot_pool_size": 0}}},
                "logger": MagicMock(),
            })()
        )
        result = _get_job_scheduler()
        assert result is None

    def test_valid_config_creates_scheduler(self, monkeypatch):
        """Valid slot_management config creates a JobScheduler."""
        monkeypatch.setattr(
            "proxy.router._srv",
            lambda: type("Srv", (), {
                "config": {
                    "server": {
                        "slot_management": {
                            "slot_pool_size": 4,
                            "slot_queue_max_depth": 16,
                            "slot_job_timeout_seconds": 300.0,
                            "slot_queue_overflow_retry_after": 900,
                        }
                    }
                }
            })()
        )
        result = _get_job_scheduler()
        assert result is not None
        assert isinstance(result, JobScheduler)
        assert len(result.slots) == 4
        assert result.queue.maxsize == 16
        assert result.job_timeout == 300.0
        assert result.queue_overflow_retry_after == 900

    def test_no_slot_management_config_logs_message(self, monkeypatch):
        """When slot_management is absent, _get_job_scheduler() logs at INFO (AC 9)."""

        mock_srv = MagicMock()
        mock_srv.config = {"server": {}}
        mock_srv.logger = MagicMock()

        monkeypatch.setattr("proxy.router._srv", lambda: mock_srv)

        result = _get_job_scheduler()
        assert result is None

        # Verify the log was called
        mock_srv.logger.info.assert_called_once()
        call_args = mock_srv.logger.info.call_args[0][0]
        assert "scheduler not initialized" in call_args
        assert "slot_management" in call_args

    def test_pool_size_zero_logs_message(self, monkeypatch):
        """When pool_size < 1, _get_job_scheduler() logs at INFO (AC 9)."""

        mock_srv = MagicMock()
        mock_srv.config = {
            "server": {"slot_management": {"slot_pool_size": 0}},
        }
        mock_srv.logger = MagicMock()

        monkeypatch.setattr("proxy.router._srv", lambda: mock_srv)

        result = _get_job_scheduler()
        assert result is None

        # Verify the log was called
        mock_srv.logger.info.assert_called_once()
        call_args = mock_srv.logger.info.call_args[0][0]
        assert "scheduler not initialized" in call_args
        assert "pool_size" in call_args

    def test_scheduler_initialized_once(self, monkeypatch):
        """_get_job_scheduler() returns the same instance on subsequent calls."""
        monkeypatch.setattr(
            "proxy.router._srv",
            lambda: type("Srv", (), {
                "config": {
                    "server": {
                        "slot_management": {
                            "slot_pool_size": 2,
                            "slot_queue_max_depth": 5,
                            "slot_job_timeout_seconds": 60.0,
                            "slot_queue_overflow_retry_after": 900,
                        }
                    }
                }
            })()
        )
        first = _get_job_scheduler()
        second = _get_job_scheduler()
        assert first is second


# ===================================================================
# Router integration tests (via HTTPException simulation)
# ===================================================================


class TestRouterIntegration:
    """Tests for router + scheduler integration using real scheduler instances."""

    @pytest.mark.asyncio
    async def test_new_session_assigned_slot(self):
        """A new session calling proxy_to_local gets a slot assigned."""
        sched = JobScheduler(pool_size=2, max_queue_depth=5, job_timeout=300.0)
        await sched.start()
        try:
            # Simulate: new request arrives, session not active
            result = await sched.admit_job("new-session")
            assert result.kind == "ASSIGNED"
            assert result.slot_id is not None
            assert result.slot_id in (0, 1)

            # Verify slot ownership
            assert sched.slots[result.slot_id].state == "Owned"
            assert sched.slots[result.slot_id].job_id == "new-session"
        finally:
            await sched.stop()

    @pytest.mark.asyncio
    async def test_subsequent_request_reuses_slot(self):
        """Subsequent requests from same session get the same slot (fast path)."""
        sched = JobScheduler(pool_size=4, max_queue_depth=5, job_timeout=300.0)
        await sched.start()
        try:
            # First request: admit
            admit = await sched.admit_job("session-a")
            assert admit.kind == "ASSIGNED"

            # Second request: reenter (fast path)
            slot_id = await sched.reenter_job("session-a")
            assert slot_id == admit.slot_id

            # Third request: reenter again
            slot_id2 = await sched.reenter_job("session-a")
            assert slot_id2 == admit.slot_id
        finally:
            await sched.stop()

    @pytest.mark.asyncio
    async def test_multiple_sessions_different_slots(self):
        """Multiple concurrent sessions get different slots."""
        sched = JobScheduler(pool_size=4, max_queue_depth=5, job_timeout=300.0)
        await sched.start()
        try:
            r1 = await sched.admit_job("session-a")
            r2 = await sched.admit_job("session-b")
            r3 = await sched.admit_job("session-c")

            assert r1.kind == "ASSIGNED"
            assert r2.kind == "ASSIGNED"
            assert r3.kind == "ASSIGNED"
            assert len({r1.slot_id, r2.slot_id, r3.slot_id}) == 3
        finally:
            await sched.stop()

    @pytest.mark.asyncio
    async def test_queue_overflow_returns_503(self):
        """When queue is full, router returns 503 (via HTTPException)."""
        sched = JobScheduler(
            pool_size=1, max_queue_depth=2, job_timeout=300.0,
            queue_overflow_retry_after=900,
        )
        await sched.start()
        try:
            # Fill the slot
            await sched.admit_job("session-a")

            # Fill the queue
            await sched.admit_job("session-b")
            await sched.admit_job("session-c")

            # Queue overflow → simulate HTTP 503
            result = await sched.admit_job("session-d")
            assert result.kind == "REJECTED_503"
            assert result.retry_after == 900
        finally:
            await sched.stop()

    @pytest.mark.asyncio
    async def test_session_end_releases_slot(self):
        """On session end, slot is released and next queued job gets it."""
        sched = JobScheduler(pool_size=1, max_queue_depth=3, job_timeout=300.0)
        await sched.start()
        try:
            await sched.admit_job("session-a")
            await sched.admit_job("session-b")  # Queued

            slot_id = sched.active_jobs["session-a"]

            # Session ends → release slot
            await sched.release_slot(slot_id)

            # Next queued job gets the slot
            assert sched.active_jobs.get("session-b") == slot_id
            assert sched.slots[slot_id].job_id == "session-b"
        finally:
            await sched.stop()

    @pytest.mark.asyncio
    async def test_cascading_release_multiple_queued(self):
        """Releasing one slot triggers cascade through multiple queued jobs."""
        sched = JobScheduler(pool_size=2, max_queue_depth=5, job_timeout=300.0)
        await sched.start()
        try:
            await sched.admit_job("session-a")
            await sched.admit_job("session-b")
            await sched.admit_job("session-c")  # Queued
            await sched.admit_job("session-d")  # Queued

            # Release both slots
            await sched.release_slot(0)
            await sched.release_slot(1)

            # Both queued jobs get slots
            assert "session-c" in sched.active_jobs
            assert "session-d" in sched.active_jobs
            assert sched.queue.empty()
        finally:
            await sched.stop()

    @pytest.mark.asyncio
    async def test_job_timeout_releases_slot_for_next(self):
        """Timed-out job's slot is reassigned to next queued job."""
        sched = JobScheduler(pool_size=1, max_queue_depth=3, job_timeout=0.02)
        await sched.start()
        try:
            await sched.admit_job("session-a")
            await sched.admit_job("session-b")  # Queued

            # Wait for timeout
            await asyncio.sleep(0.05)
            await sched._check_timeouts_now()

            # Session-b should get the slot
            assert "session-a" not in sched.active_jobs
            assert sched.active_jobs.get("session-b") is not None
        finally:
            await sched.stop()

    @pytest.mark.asyncio
    async def test_admit_reenter_release_cycle(self):
        """Full lifecycle: admit → reenter → release → admit again."""
        sched = JobScheduler(pool_size=1, max_queue_depth=3, job_timeout=300.0)
        await sched.start()
        try:
            # Admit
            r1 = await sched.admit_job("session-a")
            assert r1.kind == "ASSIGNED"

            # Reenter
            slot = await sched.reenter_job("session-a")
            assert slot == r1.slot_id

            # Release
            await sched.release_slot(r1.slot_id)
            assert "session-a" not in sched.active_jobs

            # Admit again (new conversation)
            r2 = await sched.admit_job("session-a")
            assert r2.kind == "ASSIGNED"
        finally:
            await sched.stop()


# ===================================================================
# Active-request tracking (streaming protection)
# ===================================================================


class TestActiveRequestStreaming:
    """Integration tests for active-request tracking during streaming."""

    @pytest.mark.asyncio
    async def test_streaming_longer_than_timeout_no_release(self):
        """
        Simulates a streaming response longer than slot_job_timeout_seconds.

        Verifies the slot is NOT released mid-stream when active_requests > 0.
        """
        sched = JobScheduler(pool_size=1, max_queue_depth=3, job_timeout=0.02)
        await sched.start()
        try:
            await sched.admit_job("session-a")
            slot_id = sched.active_jobs["session-a"]

            # Mark request as active (simulating start of streaming)
            await sched.mark_request_start(slot_id)

            # Wait significantly longer than timeout
            await asyncio.sleep(0.05)

            # Run timeout check
            await sched._check_timeouts_now()

            # Slot should NOT have been released (active request in flight)
            assert "session-a" in sched.active_jobs, (
                "Session should still own slot during active request"
            )
            assert sched.slots[slot_id].state == "Owned"
            assert sched.slots[slot_id].active_requests == 1

            # Now end the request
            await sched.mark_request_end(slot_id)

            # Wait for timeout
            await asyncio.sleep(0.03)
            await sched._check_timeouts_now()

            # Slot should be released now (no active request)
            assert "session-a" not in sched.active_jobs, (
                "Session should be released after request ends and timeout expires"
            )
            assert sched.slots[slot_id].state == "Idle"
        finally:
            await sched.stop()

    @pytest.mark.asyncio
    async def test_queue_drain_during_active_stream(self):
        """Queued jobs should not preempt a slot with an active request."""
        sched = JobScheduler(pool_size=1, max_queue_depth=3, job_timeout=0.02)
        await sched.start()
        try:
            await sched.admit_job("session-a")
            slot_id = sched.active_jobs["session-a"]

            # Queue another job
            await sched.admit_job("session-b")
            assert sched.queue.qsize() == 1

            # Mark request active
            await sched.mark_request_start(slot_id)

            # Wait past timeout
            await asyncio.sleep(0.05)
            await sched._check_timeouts_now()

            # Session-a should still own the slot
            assert "session-a" in sched.active_jobs
            assert sched.queue.qsize() == 1  # session-b still queued

            # End request and re-check
            await sched.mark_request_end(slot_id)
            await asyncio.sleep(0.03)
            await sched._check_timeouts_now()

            # Session-b should now get the slot
            assert "session-a" not in sched.active_jobs
            assert sched.active_jobs.get("session-b") == slot_id
            assert sched.queue.empty()
        finally:
            await sched.stop()


# ===================================================================
# Session invalidation releases scheduler slot
# ===================================================================


class TestInvalidationReleasesSchedulerSlot:
    """Tests that _invalidate_session_and_slot releases the scheduler slot."""

    @pytest.mark.asyncio
    async def test_invalidation_releases_scheduler_slot(self):
        """
        When _invalidate_session_and_slot is called with a scheduler
        reference, the scheduler slot is released.
        """
        from proxy.session import _invalidate_session_and_slot

        sched = JobScheduler(pool_size=1, max_queue_depth=3, job_timeout=300.0)
        await sched.start()
        try:
            await sched.admit_job("session-a")
            slot_id = sched.active_jobs["session-a"]

            # Invalidate with scheduler reference
            await _invalidate_session_and_slot(
                "session-a",
                reason="test_invalidation",
                slot_filename=None,
                scheduler=sched,
                scheduler_slot_id=slot_id,
            )

            # Scheduler slot should be released
            assert "session-a" not in sched.active_jobs
            assert sched.slots[slot_id].state == "Idle"
        finally:
            await sched.stop()

    @pytest.mark.asyncio
    async def test_invalidation_without_scheduler_no_error(self):
        """Calling _invalidate_session_and_slot without a scheduler works."""
        from proxy.session import _invalidate_session_and_slot

        # Should not raise
        await _invalidate_session_and_slot(
            "test-session",
            reason="test_no_scheduler",
            slot_filename=None,
        )
