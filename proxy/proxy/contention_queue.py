"""Bounded cross-session contention queue for local slot dispatch.

In cheap mode (``contention_queue_policy: queue``), when all local slots are
busy at the ``local_concurrency_limit`` decision point (provider.py), a
request QUEUES (bounded by ``contention_queue_max_wait_seconds`` and
``contention_queue_max_depth``) instead of immediately falling back to the
next remote provider (LP-0MSORQVK50012Q4D). When a slot frees within the
caps the request is dispatched local; when the caps are exceeded it falls
back exactly as today.

Context bypasses (``context_too_large`` / ``large_context_bypass``) are
physical capacity limits and are NEVER queued — provider.py evaluates the
skip decision before enqueueing.

Wake signals (F3 AC2): the queue wakes on BOTH ``local_active_queries``
decrement (a local stream ended → a slot freed) AND slot-persistence /
lease release (slot save/restore frees the backend during model switches).
Call sites: router_helpers._decrement_local_active_queries,
router_helpers._release_local_dispatch,
router_helpers._cleanup_stale_local_dispatch, and
router_helpers._recover_stuck_local_active_queries.

The queue is cross-session: state lives at module scope (one proxy process),
so a request from session B can wait behind a long audit stream from session
A — bounded by the wait/depth caps.
"""

import asyncio
import time
from collections import deque
from collections.abc import Callable

# ---------------------------------------------------------------------------
# Cross-session queue state (module-level = process-wide)
# ---------------------------------------------------------------------------

_waiters: deque = deque()  # deque of (enqueued_at, asyncio.Event)
_condition: asyncio.Condition | None = None


def _get_condition() -> asyncio.Condition:
    global _condition
    if _condition is None:
        _condition = asyncio.Condition()
    return _condition


def reset() -> None:
    """Reset all queue state (test helper)."""
    global _waiters, _condition
    _waiters = deque()
    _condition = None


def queue_depth() -> int:
    """Current number of waiters in the cross-session queue."""
    return len(_waiters)


# ---------------------------------------------------------------------------
# Wake signalling
# ---------------------------------------------------------------------------

async def wake(count: int = 1) -> None:
    """Wake up to *count* queued waiters (FIFO — one per freed local slot).

    Each wake call corresponds to one freed slot (one ``local_active_queries``
    decrement or one lease release), so exactly one waiter is woken per slot.
    The woken waiter re-checks slot availability before proceeding; if another
    request grabbed the slot first it keeps waiting (bounded by the caps).
    """
    cond = _get_condition()
    async with cond:
        for _ in range(max(0, count)):
            if not _waiters:
                break
            _waiters[0][1].set()
            _waiters.rotate(-1)  # FIFO: next wake targets the next waiter


async def wake_all() -> None:
    """Wake every queued waiter (used when ALL slots free at once, e.g. a
    stuck-counter recovery)."""
    cond = _get_condition()
    async with cond:
        for _, ev in _waiters:
            ev.set()


# ---------------------------------------------------------------------------
# Bounded wait
# ---------------------------------------------------------------------------

async def wait_for_local_slot(
    max_wait_seconds: float,
    max_depth: int,
    slot_free_check: Callable[[], bool],
) -> float | None:
    """Wait (bounded, cross-session) for a local slot to free.

    Returns the elapsed wait in seconds when a slot freed within the caps
    (the caller should dispatch local), or None when the caps were exceeded
    (the caller falls back to the next remote provider exactly as today).

    *slot_free_check* is a zero-arg callable returning True when a local
    slot is currently free (e.g. ``lambda: _get_local_concurrency_info(config)[0] < max_local``).
    """
    cond = _get_condition()
    started = time.monotonic()
    async with cond:
        if len(_waiters) >= max_depth:
            # Depth cap exceeded — don't even enqueue; fall back immediately.
            return None
        if slot_free_check():
            # Slot already free (race between the decision point and here).
            return 0.0
        ev = asyncio.Event()
        _waiters.append((started, ev))
    try:
        while True:
            remaining = max_wait_seconds - (time.monotonic() - started)
            if remaining <= 0:
                return None
            try:
                await asyncio.wait_for(ev.wait(), timeout=remaining)
            except TimeoutError:
                return None
            if slot_free_check():
                return time.monotonic() - started
            # Spurious wake (another waiter claimed the slot) — keep waiting
            # for the remaining budget.
            ev.clear()
    finally:
        async with cond:
            for i, (t, e) in enumerate(_waiters):
                if e is ev:
                    del _waiters[i]
                    break
