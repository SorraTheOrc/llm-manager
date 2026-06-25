"""
Unit tests for the JobScheduler class.

Tests cover:
- State initialization
- admit_job() — idle slot assignment, re-entry, queue, rejection
- reenter_job() — fast path for active jobs
- release_slot() — slot lifecycle, queue management
- Edge cases — timeout, lifecycle, no-op operations
"""

import asyncio
import time

import pytest

from proxy.slot_scheduler import JobScheduler, AdmitResult, SlotState


# ===================================================================
# Helpers
# ===================================================================


async def make_scheduler(pool_size=2, max_queue_depth=3, job_timeout=300.0):
    """Create and start a JobScheduler."""
    sched = JobScheduler(pool_size=pool_size, max_queue_depth=max_queue_depth,
                         job_timeout=job_timeout)
    await sched.start()
    return sched


async def cleanup_scheduler(sched):
    """Stop a scheduler."""
    await sched.stop()


# ===================================================================
# Initialization
# ===================================================================


class TestJobSchedulerInit:
    """Tests for JobScheduler initialization."""

    def test_pool_size_respected(self):
        """Scheduler creates exactly pool_size slots, all idle."""
        sched = JobScheduler(pool_size=4, max_queue_depth=10, job_timeout=300.0)
        assert len(sched.slots) == 4
        for slot_id, slot_state in sched.slots.items():
            assert isinstance(slot_state, SlotState)
            assert slot_state.state == "Idle"
            assert slot_state.slot_id == slot_id
            assert slot_state.job_id is None

    def test_min_pool_size(self):
        """Pool size of 1 is valid."""
        sched = JobScheduler(pool_size=1, max_queue_depth=5, job_timeout=60.0)
        assert len(sched.slots) == 1
        assert sched.slots[0].state == "Idle"

    def test_queue_initialized_empty(self):
        """Queue starts empty."""
        sched = JobScheduler(pool_size=2, max_queue_depth=5, job_timeout=300.0)
        assert sched.queue.empty()
        assert len(sched.active_jobs) == 0
        assert len(sched.slot_to_job) == 0


# ===================================================================
# admit_job()
# ===================================================================


class TestAdmitJob:
    """Tests for JobScheduler.admit_job()."""

    @pytest.mark.asyncio
    async def test_idle_slot_assignment(self):
        """A new job is assigned to an idle slot."""
        s = await make_scheduler()
        try:
            result = await s.admit_job("session-a")
            assert result.kind == "ASSIGNED"
            assert isinstance(result.slot_id, int)
            assert result.slot_id in (0, 1)

            slot = s.slots[result.slot_id]
            assert slot.state == "Owned"
            assert slot.job_id == "session-a"
            assert s.active_jobs["session-a"] == result.slot_id
            assert s.slot_to_job[result.slot_id] == "session-a"
        finally:
            await cleanup_scheduler(s)

    @pytest.mark.asyncio
    async def test_reentry_returns_existing_slot(self):
        """Re-admitting the same job returns its existing slot."""
        s = await make_scheduler()
        try:
            result1 = await s.admit_job("session-a")
            result2 = await s.admit_job("session-a")
            assert result2.kind == "ASSIGNED"
            assert result2.slot_id == result1.slot_id
        finally:
            await cleanup_scheduler(s)

    @pytest.mark.asyncio
    async def test_multiple_jobs_different_slots(self):
        """Two jobs get different idle slots."""
        s = await make_scheduler()
        try:
            result1 = await s.admit_job("session-a")
            result2 = await s.admit_job("session-b")
            assert result1.kind == "ASSIGNED"
            assert result2.kind == "ASSIGNED"
            assert result1.slot_id != result2.slot_id
        finally:
            await cleanup_scheduler(s)

    @pytest.mark.asyncio
    async def test_queue_when_all_slots_busy(self):
        """Third job is queued when both slots are busy."""
        s = await make_scheduler()
        try:
            await s.admit_job("session-a")
            await s.admit_job("session-b")

            result = await s.admit_job("session-c")
            assert result.kind == "QUEUED"
            assert result.position is not None
            assert s.queue.qsize() == 1
        finally:
            await cleanup_scheduler(s)

    @pytest.mark.asyncio
    async def test_queue_position_increments(self):
        """Multiple queued jobs reflect correct positions."""
        s = await make_scheduler()
        try:
            await s.admit_job("session-a")
            await s.admit_job("session-b")
            await s.admit_job("session-c")
            result_d = await s.admit_job("session-d")
            assert result_d.kind == "QUEUED"
            assert result_d.position == 1  # Second in queue (0-indexed)
        finally:
            await cleanup_scheduler(s)

    @pytest.mark.asyncio
    async def test_rejection_when_queue_full(self):
        """Sixth job (2 active + 3 queued = full) is rejected."""
        s = await make_scheduler()
        try:
            await s.admit_job("session-a")
            await s.admit_job("session-b")
            await s.admit_job("session-c")
            await s.admit_job("session-d")
            await s.admit_job("session-e")

            result = await s.admit_job("session-f")
            assert result.kind == "REJECTED_503"
            assert result.retry_after > 0
        finally:
            await cleanup_scheduler(s)

    @pytest.mark.asyncio
    async def test_queue_fifo_ordering(self):
        """Queued jobs are dequeued in FIFO order."""
        s = await make_scheduler()
        try:
            await s.admit_job("session-a")
            await s.admit_job("session-b")
            await s.admit_job("session-c")
            await s.admit_job("session-d")

            await s.release_slot(0)
            assert s.active_jobs.get("session-c") == 0
        finally:
            await cleanup_scheduler(s)


# ===================================================================
# reenter_job()
# ===================================================================


class TestReenterJob:
    """Tests for JobScheduler.reenter_job()."""

    @pytest.mark.asyncio
    async def test_active_job_returns_slot(self):
        """An active job re-entering gets its slot."""
        s = await make_scheduler()
        try:
            await s.admit_job("session-a")
            slot_id = s.active_jobs["session-a"]
            result = await s.reenter_job("session-a")
            assert result == slot_id
        finally:
            await cleanup_scheduler(s)

    @pytest.mark.asyncio
    async def test_inactive_job_returns_none(self):
        """An inactive job (never admitted) returns None."""
        s = await make_scheduler()
        try:
            result = await s.reenter_job("never-admitted")
            assert result is None
        finally:
            await cleanup_scheduler(s)

    @pytest.mark.asyncio
    async def test_released_job_returns_none(self):
        """A job that was released returns None."""
        s = await make_scheduler()
        try:
            await s.admit_job("session-a")
            slot_id = s.active_jobs["session-a"]
            await s.release_slot(slot_id)
            result = await s.reenter_job("session-a")
            assert result is None
        finally:
            await cleanup_scheduler(s)

    @pytest.mark.asyncio
    async def test_multiple_simultaneous_active_jobs(self):
        """Multiple active jobs each re-enter to their own slot."""
        s = await make_scheduler()
        try:
            await s.admit_job("session-a")
            await s.admit_job("session-b")
            result_a = await s.reenter_job("session-a")
            result_b = await s.reenter_job("session-b")
            assert result_a is not None
            assert result_b is not None
            assert result_a != result_b
        finally:
            await cleanup_scheduler(s)


# ===================================================================
# release_slot()
# ===================================================================


class TestReleaseSlot:
    """Tests for JobScheduler.release_slot()."""

    @pytest.mark.asyncio
    async def test_slot_transitions_to_idle(self):
        """Released slot returns to Idle state."""
        s = await make_scheduler()
        try:
            result = await s.admit_job("session-a")
            await s.release_slot(result.slot_id)
            assert s.slots[result.slot_id].state == "Idle"
            assert s.slots[result.slot_id].job_id is None
        finally:
            await cleanup_scheduler(s)

    @pytest.mark.asyncio
    async def test_active_jobs_cleared(self):
        """Released job is removed from active_jobs mapping."""
        s = await make_scheduler()
        try:
            result = await s.admit_job("session-a")
            await s.release_slot(result.slot_id)
            assert "session-a" not in s.active_jobs
            assert result.slot_id not in s.slot_to_job
        finally:
            await cleanup_scheduler(s)

    @pytest.mark.asyncio
    async def test_next_queued_job_assigned(self):
        """Next queued job is assigned to the freed slot."""
        s = await make_scheduler()
        try:
            await s.admit_job("session-a")
            await s.admit_job("session-b")
            await s.admit_job("session-c")

            slot_b = s.active_jobs["session-b"]
            await s.release_slot(slot_b)

            assert s.active_jobs.get("session-c") == slot_b
            assert s.slots[slot_b].job_id == "session-c"
        finally:
            await cleanup_scheduler(s)

    @pytest.mark.asyncio
    async def test_empty_queue_keeps_slot_idle(self):
        """Releasing when queue is empty leaves slot idle."""
        s = await make_scheduler()
        try:
            result = await s.admit_job("session-a")
            await s.release_slot(result.slot_id)
            assert s.slots[result.slot_id].state == "Idle"
            assert s.queue.empty()
        finally:
            await cleanup_scheduler(s)

    @pytest.mark.asyncio
    async def test_cascading_release(self):
        """Releasing both slots assigns both queued jobs."""
        s = await make_scheduler()
        try:
            await s.admit_job("session-a")
            await s.admit_job("session-b")
            await s.admit_job("session-c")
            await s.admit_job("session-d")

            await s.release_slot(0)
            await s.release_slot(1)

            assert s.active_jobs.get("session-c") is not None
            assert s.active_jobs.get("session-d") is not None
            assert s.queue.empty()
        finally:
            await cleanup_scheduler(s)


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Tests for edge cases in JobScheduler."""

    @pytest.mark.asyncio
    async def test_release_unassigned_slot(self):
        """Releasing a slot that is already idle is a no-op."""
        s = await make_scheduler()
        try:
            await s.release_slot(0)
            assert s.slots[0].state == "Idle"
        finally:
            await cleanup_scheduler(s)

    @pytest.mark.asyncio
    async def test_admit_after_release(self):
        """A job can be re-admitted after being released."""
        s = await make_scheduler()
        try:
            result1 = await s.admit_job("session-a")
            await s.release_slot(result1.slot_id)

            result2 = await s.admit_job("session-a")
            assert result2.kind == "ASSIGNED"
        finally:
            await cleanup_scheduler(s)

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        """Scheduler start creates background task; stop cancels it."""
        sched = JobScheduler(pool_size=2, max_queue_depth=5, job_timeout=300.0)
        assert sched._timeout_check_task is None

        await sched.start()
        assert sched._timeout_check_task is not None
        assert not sched._timeout_check_task.done()

        await sched.stop()
        assert sched._timeout_check_task.cancelled()

    @pytest.mark.asyncio
    async def test_job_timeout_releases_slot(self):
        """A job that exceeds timeout is released by _check_timeouts_now."""
        sched = JobScheduler(pool_size=1, max_queue_depth=3, job_timeout=0.01)
        await sched.start()
        try:
            await sched.admit_job("session-a")
            assert sched.active_jobs.get("session-a") is not None

            await asyncio.sleep(0.02)
            await sched._check_timeouts_now()

            assert "session-a" not in sched.active_jobs
            assert sched.slots[0].state == "Idle"
        finally:
            await sched.stop()

    @pytest.mark.asyncio
    async def test_reenter_job_not_blocking(self):
        """reenter_job returns None immediately for unknown sessions."""
        s = await make_scheduler()
        try:
            result = await s.reenter_job("nonexistent")
            assert result is None
        finally:
            await cleanup_scheduler(s)


# ===================================================================
# Active request tracking
# ===================================================================


class TestActiveRequestTracking:
    """Tests for mark_request_start / mark_request_end."""

    @pytest.mark.asyncio
    async def test_mark_request_start_increments_counter(self):
        """mark_request_start increments the active_requests counter."""
        s = await make_scheduler()
        try:
            await s.admit_job("session-a")
            slot_id = s.active_jobs["session-a"]

            assert s.slots[slot_id].active_requests == 0

            await s.mark_request_start(slot_id)
            assert s.slots[slot_id].active_requests == 1

            await s.mark_request_start(slot_id)
            assert s.slots[slot_id].active_requests == 2
        finally:
            await cleanup_scheduler(s)

    @pytest.mark.asyncio
    async def test_mark_request_end_decrements_counter(self):
        """mark_request_end decrements the active_requests counter."""
        s = await make_scheduler()
        try:
            await s.admit_job("session-a")
            slot_id = s.active_jobs["session-a"]

            await s.mark_request_start(slot_id)
            await s.mark_request_start(slot_id)
            assert s.slots[slot_id].active_requests == 2

            await s.mark_request_end(slot_id)
            assert s.slots[slot_id].active_requests == 1

            await s.mark_request_end(slot_id)
            assert s.slots[slot_id].active_requests == 0
        finally:
            await cleanup_scheduler(s)

    @pytest.mark.asyncio
    async def test_mark_request_end_does_not_go_below_zero(self):
        """mark_request_end does not decrement below zero."""
        s = await make_scheduler()
        try:
            await s.admit_job("session-a")
            slot_id = s.active_jobs["session-a"]

            assert s.slots[slot_id].active_requests == 0
            await s.mark_request_end(slot_id)
            assert s.slots[slot_id].active_requests == 0
        finally:
            await cleanup_scheduler(s)

    @pytest.mark.asyncio
    async def test_mark_request_start_unknown_slot_no_error(self):
        """mark_request_start on unknown slot is a no-op."""
        s = await make_scheduler()
        try:
            # Should not raise
            await s.mark_request_start(999)
        finally:
            await cleanup_scheduler(s)

    @pytest.mark.asyncio
    async def test_mark_request_end_unknown_slot_no_error(self):
        """mark_request_end on unknown slot is a no-op."""
        s = await make_scheduler()
        try:
            await s.mark_request_end(999)
        finally:
            await cleanup_scheduler(s)


# ===================================================================
# Timeout check with active requests
# ===================================================================


class TestTimeoutWithActiveRequests:
    """Tests that _check_timeouts_now skips slots with active requests."""

    @pytest.mark.asyncio
    async def test_timeout_skips_slot_with_active_request(self):
        """A slot with an active request is NOT released by timeout."""
        sched = JobScheduler(pool_size=1, max_queue_depth=3, job_timeout=0.01)
        await sched.start()
        try:
            await sched.admit_job("session-a")
            slot_id = sched.active_jobs["session-a"]

            # Mark request as active
            await sched.mark_request_start(slot_id)

            # Wait for timeout period
            await asyncio.sleep(0.02)
            await sched._check_timeouts_now()

            # Slot should still be owned (not released)
            assert "session-a" in sched.active_jobs
            assert sched.slots[slot_id].state == "Owned"
            assert sched.slots[slot_id].active_requests == 1
        finally:
            await sched.stop()

    @pytest.mark.asyncio
    async def test_timeout_releases_after_request_ends(self):
        """After request ends, idle timeout releases the slot normally."""
        sched = JobScheduler(pool_size=1, max_queue_depth=3, job_timeout=0.01)
        await sched.start()
        try:
            await sched.admit_job("session-a")
            slot_id = sched.active_jobs["session-a"]

            # Mark request active then end it
            await sched.mark_request_start(slot_id)
            await sched.mark_request_end(slot_id)

            # Wait for timeout period
            await asyncio.sleep(0.02)
            await sched._check_timeouts_now()

            # Slot should be released now
            assert "session-a" not in sched.active_jobs
            assert sched.slots[slot_id].state == "Idle"
        finally:
            await sched.stop()

    @pytest.mark.asyncio
    async def test_release_slot_resets_active_requests(self):
        """release_slot resets the active_requests counter to zero."""
        s = await make_scheduler()
        try:
            await s.admit_job("session-a")
            slot_id = s.active_jobs["session-a"]

            await s.mark_request_start(slot_id)
            assert s.slots[slot_id].active_requests == 1

            await s.release_slot(slot_id)
            assert s.slots[slot_id].active_requests == 0
        finally:
            await cleanup_scheduler(s)
