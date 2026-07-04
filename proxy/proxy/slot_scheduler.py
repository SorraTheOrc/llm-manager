"""
Job Scheduler — Job-Level Slot Ownership

Manages job-to-slot assignment with ownership semantics for multi-turn
conversations. Each session (job) is assigned to a slot for the entire
lifetime of the conversation, preventing inter-session slot stealing.

Usage:
    scheduler = JobScheduler(pool_size=4, max_queue_depth=16, job_timeout=300.0)
    await scheduler.start()

    # New job
    result = await scheduler.admit_job("session-id")
    if result.kind == "ASSIGNED":
        slot_id = result.slot_id
    elif result.kind == "QUEUED":
        # Wait for slot
    else:
        # REJECTED_503 - queue full

    # Subsequent requests from active job
    slot_id = await scheduler.reenter_job("session-id")
    if slot_id is not None:
        # Fast path - job owns a slot

    # On session end
    await scheduler.release_slot(slot_id)
    await scheduler.stop()

    # On client disconnect (release slot or remove from queue)
    await scheduler.remove_job(session_id)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union


logger = logging.getLogger(__name__)


# ===================================================================
# Data types
# ===================================================================


@dataclass
class SlotState:
    """Per-slot state tracking."""

    slot_id: int
    state: Literal["Idle", "Owned"] = "Idle"
    job_id: Optional[str] = None
    job_assigned_at: Optional[float] = None
    job_last_request_at: Optional[float] = None
    active_requests: int = 0


@dataclass
class QueuedJob:
    """A job waiting in the queue."""

    tenant_id: str
    job_id: str
    enqueue_time: float = field(default_factory=time.monotonic)
    request_count: int = 0


@dataclass
class AdmitResult:
    """Result of admitting a job to the scheduler."""

    kind: str  # "ASSIGNED", "QUEUED", "REJECTED_503"
    slot_id: Optional[int] = None
    position: Optional[int] = None
    retry_after: Optional[float] = None


# ===================================================================
# JobScheduler
# ===================================================================


class JobScheduler:
    """
    Manages job-to-slot assignment with ownership semantics.

    State:
        slots: dict[slot_id, SlotState]       -- per-slot state
        queue: asyncio.Queue[QueuedJob]        -- waiting jobs
        active_jobs: dict[session_id, int]     -- session → slot_id mapping
        slot_to_job: dict[int, str]            -- slot_id → session_id mapping
    """

    def __init__(
        self,
        pool_size: int,
        max_queue_depth: int,
        job_timeout: float,
        queue_overflow_retry_after: float = 900.0,
    ):
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        if max_queue_depth < 1:
            raise ValueError("max_queue_depth must be >= 1")

        self.slots: Dict[int, SlotState] = {
            i: SlotState(slot_id=i) for i in range(pool_size)
        }
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_depth)
        self.active_jobs: Dict[str, int] = {}  # session_id → slot_id
        self.slot_to_job: Dict[int, str] = {}  # slot_id → session_id
        self.job_timeout = job_timeout
        self.queue_overflow_retry_after = queue_overflow_retry_after
        self._timeout_check_task: Optional[asyncio.Task] = None
        # Tracking for queued job removal (LP-0MQTHP828000JYM6)
        self._queued_jobs: Dict[str, QueuedJob] = {}  # session_id -> QueuedJob
        self._cancelled_jobs: set[str] = set()  # session_ids marked for removal

        logger.info(
            "scheduler init pool_size=%s max_queue_depth=%s job_timeout=%s",
            pool_size,
            max_queue_depth,
            job_timeout,
        )

    async def start(self) -> None:
        """Start the scheduler (including background timeout checks)."""
        self._timeout_check_task = asyncio.create_task(
            self._periodic_timeout_check()
        )

    async def stop(self) -> None:
        """Stop the scheduler and cancel the background timeout task."""
        if self._timeout_check_task and not self._timeout_check_task.done():
            self._timeout_check_task.cancel()
            try:
                await self._timeout_check_task
            except asyncio.CancelledError:
                pass

    async def admit_job(self, session_id: str) -> AdmitResult:
        """
        Admit a new job.

        Returns one of:
          - ASSIGNED(slot_id): Job assigned to an idle slot
          - QUEUED: Job added to queue, waiting for a slot
          - REJECTED_503: Queue full, job rejected with Retry-After
        """
        # Case 1: Job already owns a slot — return the slot
        if session_id in self.active_jobs:
            slot_id = self.active_jobs[session_id]
            logger.info(
                "scheduler admit_job re-entry session=%s slot=%s",
                session_id[:8], slot_id,
            )
            return AdmitResult(kind="ASSIGNED", slot_id=slot_id)

        # Case 2: Find an idle slot
        for slot_id, slot_state in self.slots.items():
            if slot_state.state == "Idle":
                now = time.monotonic()
                slot_state.state = "Owned"
                slot_state.job_id = session_id
                slot_state.job_assigned_at = now
                slot_state.job_last_request_at = now
                self.active_jobs[session_id] = slot_id
                self.slot_to_job[slot_id] = session_id
                logger.info(
                    "scheduler admit_job assign session=%s slot=%s",
                    session_id[:8], slot_id,
                )
                return AdmitResult(kind="ASSIGNED", slot_id=slot_id)

        # Case 3: All slots busy — try to enqueue
        if self.queue.full():
            logger.info(
                "scheduler queue full session=%s",
                session_id[:8],
            )
            return AdmitResult(
                kind="REJECTED_503",
                retry_after=self.queue_overflow_retry_after,
            )

        # Determine queue position before enqueueing
        position = self.queue.qsize()

        queued_job = QueuedJob(
            tenant_id=session_id,
            job_id=session_id,
            enqueue_time=time.monotonic(),
            request_count=0,
        )
        self._queued_jobs[session_id] = queued_job
        await self.queue.put(queued_job)
        logger.info(
            "scheduler queue session=%s position=%s",
            session_id[:8], position,
        )
        return AdmitResult(kind="QUEUED", position=position)

    async def reenter_job(self, session_id: str) -> Optional[int]:
        """
        Called for subsequent requests from an active job.

        Returns the slot_id if the job is still active, None otherwise.
        Caller must call admit_job() when None is returned.
        """
        if session_id in self.active_jobs:
            slot_id = self.active_jobs[session_id]
            # Update last request time
            if slot_id in self.slots:
                self.slots[slot_id].job_last_request_at = time.monotonic()
            logger.info(
                "scheduler reenter_job session=%s slot=%s",
                session_id[:8], slot_id,
            )
            return slot_id
        return None

    async def mark_request_start(self, slot_id: int) -> None:
        """
        Mark that a request has started on the given slot.

        This increments the active-request counter, preventing the
        background timeout check from releasing the slot while a
        request is actively being served.
        """
        if slot_id in self.slots:
            self.slots[slot_id].active_requests += 1

    def has_idle_slot(self) -> bool:
        """
        Check if any slot is currently idle.

        Returns True if at least one slot has state == "Idle".
        This is a cheap check with no side effects (no queuing, no assignment).
        """
        for slot_state in self.slots.values():
            if slot_state.state == "Idle":
                return True
        return False

    async def mark_request_end(self, slot_id: int) -> None:
        """
        Mark that a request has ended on the given slot.

        Decrements the active-request counter. Must be called in a
        finally block paired with mark_request_start.

        If the request was the last active one, logs the projected
        timeout time for the slot (LP-0MR5MAJNM005R905).
        """
        if slot_id in self.slots and self.slots[slot_id].active_requests > 0:
            self.slots[slot_id].active_requests -= 1
            # Log projected timeout if this was the last active request
            slot_state = self.slots[slot_id]
            if slot_state.active_requests == 0 and slot_state.job_last_request_at is not None:
                next_timeout_at = slot_state.job_last_request_at + self.job_timeout
                logger.info(
                    "scheduler request complete slot=%s job=%s next_timeout_at=%.1fs",
                    slot_id,
                    (slot_state.job_id[:8] if slot_state.job_id else "none"),
                    next_timeout_at,
                )

    async def remove_job(self, session_id: str) -> bool:
        """
        Remove a job by session_id.

        Handles two cases:
        - If the session owns a slot: releases the slot (AC4)
        - If the session is queued: marks the job as cancelled (AC3)

        Returns True if any action was taken, False if the session was unknown.
        """
        # Case 1: Session owns a slot — release it
        if session_id in self.active_jobs:
            slot_id = self.active_jobs[session_id]
            await self.release_slot(slot_id)
            logger.info(
                "scheduler remove_job slot session=%s slot=%s",
                session_id[:8], slot_id,
            )
            return True

        # Case 2: Session is queued — mark as cancelled
        if session_id in self._queued_jobs:
            self._cancelled_jobs.add(session_id)
            del self._queued_jobs[session_id]
            logger.info(
                "scheduler remove_job queued session=%s",
                session_id[:8],
            )
            return True

        return False

    async def release_slot(self, slot_id: int) -> None:
        """
        Release a slot from its owning job.

        Transitions slot to Idle and assigns the next job from the queue
        (if any) to this slot.
        """
        if slot_id not in self.slots:
            return

        slot_state = self.slots[slot_id]

        # If already idle, nothing to do
        if slot_state.state == "Idle":
            return

        job_id = slot_state.job_id

        # Log the release reason (caller should pass context, but we log generically)
        logger.info(
            "scheduler release_slot slot=%s job=%s",
            slot_id,
            (job_id[:8] if job_id else "none"),
        )

        # Clear ownership
        slot_state.state = "Idle"
        slot_state.job_id = None
        slot_state.job_assigned_at = None
        slot_state.job_last_request_at = None
        slot_state.active_requests = 0

        if job_id and job_id in self.active_jobs:
            del self.active_jobs[job_id]
        if slot_id in self.slot_to_job:
            del self.slot_to_job[slot_id]

        # Assign next job from queue (skipping cancelled jobs)
        while not self.queue.empty():
            try:
                queued_job = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            # Skip cancelled jobs
            if queued_job.tenant_id in self._cancelled_jobs:
                self._cancelled_jobs.discard(queued_job.tenant_id)
                if queued_job.tenant_id in self._queued_jobs:
                    del self._queued_jobs[queued_job.tenant_id]
                continue

            now = time.monotonic()
            slot_state.state = "Owned"
            slot_state.job_id = queued_job.tenant_id
            slot_state.job_assigned_at = now
            slot_state.job_last_request_at = now
            self.active_jobs[queued_job.tenant_id] = slot_id
            self.slot_to_job[slot_id] = queued_job.tenant_id
            if queued_job.tenant_id in self._queued_jobs:
                del self._queued_jobs[queued_job.tenant_id]
            logger.info(
                "scheduler queue_assign slot=%s session=%s",
                slot_id,
                (queued_job.tenant_id[:8] if queued_job.tenant_id else "none"),
            )
            return  # Job assigned; slot no longer idle
        # No more valid jobs; slot stays Idle

    async def _check_timeouts_now(self) -> None:
        """
        Single-pass timeout check. Releases all jobs that have been idle
        for longer than self.job_timeout.

        Skips slots that have active requests in flight to prevent
        premature release during a streaming response.
        """
        now = time.monotonic()
        for slot_id, slot_state in self.slots.items():
            if (
                slot_state.state == "Owned"
                and slot_state.job_last_request_at is not None
                and slot_state.active_requests == 0
            ):
                idle_time = now - slot_state.job_last_request_at
                if idle_time > self.job_timeout:
                    logger.warning(
                        "scheduler timeout slot=%s job=%s idle=%.1fs",
                        slot_id,
                        (slot_state.job_id[:8] if slot_state.job_id else "none"),
                        idle_time,
                    )
                    await self.release_slot(slot_id)

    async def _periodic_timeout_check(self) -> None:
        """
        Background task: run every 10s to release idle jobs that have
        timed out. A job is considered idle if no requests have been
        received for longer than self.job_timeout.
        """
        while True:
            await asyncio.sleep(10)
            await self._check_timeouts_now()
