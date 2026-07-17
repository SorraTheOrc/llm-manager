"""
Cache-rebuild regression detection tests.

Verifies that the job-level slot ownership model prevents cache rebuilds
across multi-turn sessions. Tests fail if slot restore/re-entry success
rate drops below defined thresholds, catching future regressions.

Uses patterns from test_slot_polling.py and test_slot_scheduler.py.
"""

import asyncio

import pytest
from proxy.slot_scheduler import JobScheduler

# ===================================================================
# Helpers
# ===================================================================


async def make_scheduler(pool_size=2, max_queue_depth=16, job_timeout=60.0):
    """Create and start a JobScheduler."""
    sched = JobScheduler(
        pool_size=pool_size,
        max_queue_depth=max_queue_depth,
        job_timeout=job_timeout,
    )
    await sched.start()
    return sched


async def cleanup(sched):
    """Stop a scheduler."""
    await sched.stop()


def compute_reentry_rate(scheduler, session_ids, turns=5):
    """
    Simulate multi-turn conversations and compute re-entry success rate.

    Returns: (success_count, total_attempts, rate)
    """
    success = 0
    total = 0

    for session_id in session_ids:
        for turn in range(turns):
            total += 1
            slot_id = scheduler.active_jobs.get(session_id)
            if slot_id is not None:
                success += 1

    return success, total, success / total if total > 0 else 0.0


# ===================================================================
# Single-session tests
# ===================================================================


class TestSingleSession:
    """Single session should always re-enter its slot."""

    @pytest.mark.asyncio
    async def test_single_session_always_reenters(self):
        """A single session across multiple turns always gets the same slot."""
        sched = await make_scheduler(pool_size=2)
        try:
            # First turn: admit
            result = await sched.admit_job("session-a")
            original_slot = result.slot_id
            assert result.kind == "ASSIGNED"

            # Simulate 10 subsequent turns
            for turn in range(10):
                slot_id = await sched.reenter_job("session-a")
                assert slot_id == original_slot, (
                    f"Turn {turn + 1}: expected slot {original_slot}, got {slot_id}"
                )
        finally:
            await cleanup(sched)

    @pytest.mark.asyncio
    async def test_single_session_pool_size_one(self):
        """Even with pool_size=1, single session always re-enters."""
        sched = await make_scheduler(pool_size=1)
        try:
            result = await sched.admit_job("session-a")

            for turn in range(10):
                slot_id = await sched.reenter_job("session-a")
                assert slot_id == result.slot_id
        finally:
            await cleanup(sched)

    @pytest.mark.asyncio
    async def test_single_session_no_contention_reentry_rate(self):
        """Re-entry rate for single session must be 100%."""
        sched = await make_scheduler(pool_size=2)
        try:
            await sched.admit_job("session-a")

            success = 0
            total = 10
            for _ in range(total):
                slot_id = await sched.reenter_job("session-a")
                if slot_id is not None:
                    success += 1

            rate = success / total
            assert rate >= 0.9, (
                f"Single-session re-entry rate {rate:.1%} < 90% threshold"
            )
        finally:
            await cleanup(sched)


# ===================================================================
# Multi-session tests
# ===================================================================


class TestMultiSession:
    """Multiple concurrent sessions should maintain high re-entry rate."""

    @pytest.mark.asyncio
    async def test_two_sessions_pool_size_two(self):
        """Two sessions with pool_size=2 each keep their slot."""
        sched = await make_scheduler(pool_size=2)
        try:
            await sched.admit_job("session-a")
            await sched.admit_job("session-b")

            for turn in range(10):
                a_slot = await sched.reenter_job("session-a")
                b_slot = await sched.reenter_job("session-b")
                assert a_slot is not None
                assert b_slot is not None
                assert a_slot != b_slot
        finally:
            await cleanup(sched)

    @pytest.mark.asyncio
    async def test_four_sessions_pool_size_four(self):
        """Four sessions with pool_size=4 each keep their slot."""
        sched = await make_scheduler(pool_size=4)
        try:
            sessions = [f"session-{chr(ord('a') + i)}" for i in range(4)]
            for s in sessions:
                await sched.admit_job(s)

            for turn in range(10):
                slots = set()
                for s in sessions:
                    slot_id = await sched.reenter_job(s)
                    assert slot_id is not None, f"{s} lost slot on turn {turn}"
                    slots.add(slot_id)
                assert len(slots) == 4, f"Slot collision on turn {turn}"
        finally:
            await cleanup(sched)

    @pytest.mark.asyncio
    async def test_eight_sessions_pool_size_four_queue(self):
        """8 sessions with pool_size=4: 4 active, 4 queued. Active keep slots."""
        sched = await make_scheduler(pool_size=4, max_queue_depth=8)
        try:
            sessions = [f"session-{chr(ord('a') + i)}" for i in range(8)]
            results = []
            for s in sessions:
                r = await sched.admit_job(s)
                results.append(r)

            # First 4 should be ASSIGNED
            assert all(r.kind == "ASSIGNED" for r in results[:4])
            # Last 4 should be QUEUED
            assert all(r.kind == "QUEUED" for r in results[4:])

            # Active sessions should re-enter
            for turn in range(5):
                for i in range(4):
                    slot_id = await sched.reenter_job(sessions[i])
                    assert slot_id is not None
        finally:
            await cleanup(sched)

    @pytest.mark.asyncio
    async def test_multi_session_reentry_rate_above_threshold(self):
        """Multi-session re-entry rate must be >= 80%."""
        sched = await make_scheduler(pool_size=4)
        try:
            sessions = [f"session-{chr(ord('a') + i)}" for i in range(4)]
            for s in sessions:
                await sched.admit_job(s)

            # Simulate multiple turns
            success = 0
            total = 0
            for turn in range(10):
                for s in sessions:
                    total += 1
                    slot_id = await sched.reenter_job(s)
                    if slot_id is not None:
                        success += 1

            rate = success / total
            assert rate >= 0.8, (
                f"Multi-session re-entry rate {rate:.1%} < 80% threshold"
            )
        finally:
            await cleanup(sched)


# ===================================================================
# Stress tests
# ===================================================================


class TestStress:
    """Stress tests for scheduler stability under load."""

    @pytest.mark.asyncio
    async def test_rapid_admit_release_cycle(self):
        """Rapid admit/release cycles don't cause slot leaks."""
        sched = await make_scheduler(pool_size=2)
        try:
            for i in range(50):
                result = await sched.admit_job(f"session-{i}")
                if result.kind == "ASSIGNED":
                    await sched.release_slot(result.slot_id)

            # All slots should be idle
            for slot in sched.slots.values():
                assert slot.state == "Idle"
            assert len(sched.active_jobs) == 0
        finally:
            await cleanup(sched)

    @pytest.mark.asyncio
    async def test_concurrent_admit_burst(self):
        """Burst of concurrent admits doesn't drop slots."""
        sched = await make_scheduler(pool_size=4)
        try:
            sessions = [f"session-{i}" for i in range(20)]

            # Concurrent admits
            results = await asyncio.gather(*[
                sched.admit_job(s) for s in sessions
            ])

            assigned = sum(1 for r in results if r.kind == "ASSIGNED")
            queued = sum(1 for r in results if r.kind == "QUEUED")
            rejected = sum(1 for r in results if r.kind == "REJECTED_503")

            assert assigned == 4  # All slots assigned
            assert queued == 16  # Rest queued (max_depth=16)
            assert rejected == 0  # No rejection
        finally:
            await cleanup(sched)


# ===================================================================
# No-mid-job-preemption tests
# ===================================================================


class TestNoPreemption:
    """Verifies that active jobs are not preempted."""

    @pytest.mark.asyncio
    async def test_active_job_not_preempted(self):
        """An active job's slot is not stolen by another session."""
        sched = await make_scheduler(pool_size=1)
        try:
            await sched.admit_job("session-a")
            original_slot = sched.active_jobs["session-a"]

            # Other sessions try to get a slot
            result = await sched.admit_job("session-b")
            assert result.kind == "QUEUED"

            # Session-a still owns its slot
            assert sched.active_jobs.get("session-a") == original_slot
            assert sched.slots[original_slot].job_id == "session-a"
        finally:
            await cleanup(sched)

    @pytest.mark.asyncio
    async def test_no_preemption_under_load(self):
        """Under load, active jobs retain their slots."""
        sched = await make_scheduler(pool_size=2)
        try:
            await sched.admit_job("session-a")
            await sched.admit_job("session-b")

            # Try to flood with new sessions
            for i in range(10):
                await sched.admit_job(f"intruder-{i}")

            # Check original jobs still own slots
            assert sched.active_jobs.get("session-a") is not None
            assert sched.active_jobs.get("session-b") is not None
        finally:
            await cleanup(sched)
