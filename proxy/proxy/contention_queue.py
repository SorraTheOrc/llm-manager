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

Lease-owner retention priority (LP-0MVI2VRT2000V1AP): a session that already
holds a local lease, or was granted one within
``contention_queue_lease_grace_seconds`` and still has local cache history,
is a *retained* waiter. Retained waiters are inserted ahead of non-retained
ones (FIFO within each class) so a returning session reclaims the slot it
just left instead of queueing behind a newcomer for the full budget and
detouring to a paid provider. A retained session without cache history (its
prefix was genuinely evicted) gets no priority.

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
import logging
import time
from collections import deque
from collections.abc import Callable

logger = logging.getLogger("llama-proxy")

# ---------------------------------------------------------------------------
# Cross-session queue state (module-level = process-wide)
# ---------------------------------------------------------------------------

# Each waiter: (enqueued_at, asyncio.Event, session_id, retained, eta_seconds).
_waiters: deque = deque()
_condition: asyncio.Condition | None = None

#: ``session_id -> monotonic`` time of its most recent local lease grant.
_local_grant_times: dict[str, float] = {}

#: Cap on the grant registry (one entry per session, pruned on insert).
_MAX_GRANT_ENTRIES = 1000


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
    _local_grant_times.clear()


def queue_depth() -> int:
    """Current number of waiters in the cross-session queue."""
    return len(_waiters)


# ---------------------------------------------------------------------------
# Lease-owner retention registry (LP-0MVI2VRT2000V1AP)
# ---------------------------------------------------------------------------

def note_local_grant(session_id: str | None, *, now: float | None = None) -> None:
    """Record that *session_id* was granted a local dispatch lease.

    Called from the dispatch-lease acquisition path. The timestamp is what
    makes a session a *returning* session after it releases its lease: within
    ``contention_queue_lease_grace_seconds`` it keeps retention priority.
    Never raises — a bookkeeping failure must not affect dispatch.
    """
    if not session_id:
        return
    try:
        timestamp = time.monotonic() if now is None else float(now)
        if session_id not in _local_grant_times and len(_local_grant_times) >= _MAX_GRANT_ENTRIES:
            try:
                _local_grant_times.pop(next(iter(_local_grant_times)), None)
            except StopIteration:
                pass
        _local_grant_times[session_id] = timestamp
    except Exception:  # pragma: no cover - defensive
        logger.debug("note_local_grant failed", exc_info=True)


def forget_local_grant(session_id: str | None) -> None:
    """Drop the grant record for *session_id* (session invalidated)."""
    if not session_id:
        return
    _local_grant_times.pop(session_id, None)


def is_recent_local_owner(
    session_id: str | None,
    grace_seconds: float,
    *,
    now: float | None = None,
) -> bool:
    """True when *session_id* held a local lease within *grace_seconds*.

    ``grace_seconds <= 0`` disables retention entirely (the rollback path).
    """
    if not session_id or grace_seconds <= 0:
        return False
    granted = _local_grant_times.get(session_id)
    if granted is None:
        return False
    current = time.monotonic() if now is None else float(now)
    return (current - granted) <= grace_seconds


def is_retained_local_owner(
    session_id: str | None,
    model_name: str | None,
    *,
    grace_seconds: float,
) -> bool:
    """Full retention predicate: recent local owner AND surviving cache history.

    The cache-history half is what keeps the priority honest: when the
    session's prefix was genuinely evicted (its ratio record was cleared and
    no measurement survives), re-admitting it is no cheaper than a newcomer,
    so it earns no priority (LP-0MVI2VRT2000V1AP AC3).
    """
    if not is_recent_local_owner(session_id, grace_seconds):
        return False
    try:
        from proxy.provider import _get_cached_ratio_source

        return _get_cached_ratio_source(model_name, session_id) is not None
    except Exception:  # pragma: no cover - defensive
        return False


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
        for _, ev, *_rest in _waiters:
            ev.set()


# ---------------------------------------------------------------------------
# Bounded wait
# ---------------------------------------------------------------------------

def _enqueue(
    entry: tuple,
    retained: bool,
    *,
    eta_ordering: bool = False,
    promote_after_seconds: float = 0.0,
) -> int:
    """Insert *entry* into ``_waiters``, retained waiters ahead of newcomers.

    Returns the number of waiters the entry jumped ahead of (0 when appended
    FIFO). Retained entries are appended after any existing retained ones, so
    ordering within a class stays FIFO unless ETA ordering is enabled.

    ETA-aware ordering (LP-0MVIQ0G2M0006QF6 AC1): when *eta_ordering* is on,
    the entry is placed inside its retention class by ascending projected
    prefill ETA, so a queued short turn is not parked behind a multi-minute
    prefill (``local_max_concurrent_prefills: 1`` makes head-of-line blocking
    the dominant tail). Retention still wins over size: ordering only applies
    *within* a retention class.

    Age-based promotion bounds starvation: an existing waiter whose age
    exceeds *promote_after_seconds* is treated as ETA 0 (front of its class),
    so a long prefill that keeps losing still gets the slot once it has
    waited long enough.
    """
    def _eta_of(waiter: tuple) -> float | None:
        return waiter[4] if len(waiter) > 4 else None

    def _key(waiter: tuple, now: float) -> float:
        if promote_after_seconds > 0:
            try:
                age = now - float(waiter[0])
            except (TypeError, ValueError):
                age = 0.0
            if age >= promote_after_seconds:
                return 0.0  # promoted: front of its class
        eta = _eta_of(waiter)
        return float(eta) if eta is not None else float("inf")

    if not retained:
        if not eta_ordering:
            _waiters.append(entry)
            return 0
        segment_start = 0
        while segment_start < len(_waiters) and _waiters[segment_start][3]:
            segment_start += 1
        return _insert_sorted(entry, segment_start, len(_waiters), _key)
    insert_at = 0
    for index, existing in enumerate(_waiters):
        if existing[3] if len(existing) > 3 else False:
            insert_at = index + 1
        else:
            break
    if not eta_ordering:
        _waiters.insert(insert_at, entry)
        return len(_waiters) - insert_at - 1
    return _insert_sorted(entry, 0, insert_at, _key)


def _insert_sorted(
    entry: tuple,
    start: int,
    end: int,
    key,
) -> int:
    """Insert *entry* into ``_waiters[start:end]`` keeping *key* ascending.

    Ties keep FIFO order (the new entry is placed after existing equal keys).
    Returns how many waiters in the segment the entry jumped ahead of.
    """
    now = time.monotonic()
    entry_key = key(entry, now)
    position = end
    for index in range(start, end):
        if key(_waiters[index], now) > entry_key:
            position = index
            break
    _waiters.insert(position, entry)
    # Waiters previously at [position, end) now sit behind the new entry.
    return max(0, end - position)


async def wait_for_local_slot(
    max_wait_seconds: float,
    max_depth: int,
    slot_free_check: Callable[[], bool],
    session_id: str | None = None,
    retained: bool = False,
    eta_seconds: float | None = None,
    eta_ordering: bool = False,
    promote_after_seconds: float = 0.0,
) -> float | None:
    """Wait (bounded, cross-session) for a local slot to free.

    Returns the elapsed wait in seconds when a slot freed within the caps
    (the caller should dispatch local), or None when the caps were exceeded
    (the caller falls back to the next remote provider exactly as today).

    *slot_free_check* is a zero-arg callable returning True when a local
    slot is currently free (e.g. ``lambda: _get_local_concurrency_info(config)[0] < max_local``).

    *retained* marks a returning session (see the module docstring): its
    waiter is placed ahead of non-retained waiters and the reorder is logged
    naming both sessions and the reason (LP-0MVI2VRT2000V1AP AC1).

    *eta_seconds* / *eta_ordering* enable shortest-ETA-first ordering within a
    retention class (LP-0MVIQ0G2M0006QF6 AC1): *eta_seconds* is this
    request's projected prefill, and *promote_after_seconds* bounds long-ETA
    starvation by treating a waiter older than that as highest priority.
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
        ahead_of = _enqueue(
            (started, ev, session_id, retained, eta_seconds),
            retained,
            eta_ordering=eta_ordering,
            promote_after_seconds=promote_after_seconds,
        )
        if retained and ahead_of > 0:
            try:
                overtaken = next(
                    (w[2] for w in _waiters if not (w[3] if len(w) > 3 else False)),
                    "unknown",
                )
                logger.info(
                    "contention_queue_retention session=%s ahead_of=%s "
                    "reason=lease_owner_retention depth=%d",
                    session_id or "unknown",
                    overtaken or "unknown",
                    len(_waiters),
                )
            except Exception:  # pragma: no cover - defensive
                pass
        elif eta_ordering and ahead_of > 0:
            # Shortest-ETA-first reorder (AC1): name the waiters it passed so
            # the scheduling decision is auditable.
            try:
                overtaken = next(
                    (
                        w[2]
                        for w in _waiters
                        if w[1] is not ev and (w[3] if len(w) > 3 else False) == retained
                    ),
                    "unknown",
                )
                logger.info(
                    "contention_queue_eta_order session=%s ahead_of=%s "
                    "eta_s=%s reason=shortest_eta_first retained=%s depth=%d",
                    session_id or "unknown",
                    overtaken or "unknown",
                    "unknown" if eta_seconds is None else f"{float(eta_seconds):.1f}",
                    retained,
                    len(_waiters),
                )
            except Exception:  # pragma: no cover - defensive
                pass
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
            for i, waiter in enumerate(_waiters):
                if waiter[1] is ev:
                    del _waiters[i]
                    break
