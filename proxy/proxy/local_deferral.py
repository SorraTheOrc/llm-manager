"""Bounded per-session deferral budget for pin-local requests.

LP-0MVUH9PL0005PW90 (child of LP-0MVU1GSDY003EJ0O; parent AC5, AC6, AC7, AC8,
AC10).

The problem this module bounds: a ``local-preferred`` client that keeps being
told "retry later" can be told that for ever. The 2026-10-25 incident
(LP-0MVNTT4X8009IO2X) is the client-side shape of it — seven retries against
``Retry-After: 1`` — and an unattended worker whose pane polls a permanently
contended local GPU gets no answer at all, only a chain of deferrals.

This store counts **consecutive retryable local-capacity refusals per session**.
When the count reaches the configured threshold
(``local_deferral_max_refusals``, default 3, ``0`` disables the escalation) the
next request from that session is no longer refused: the dispatch cycle strips
the local entry for that request and the remote tiers serve it, visibly
(``X-Local-Route: remote`` + ``X-Local-Fallback-Reason:
deferral_refusals_exhausted``).

Design constraints (from the work item):

* **One classification.** The counter is incremented where
  :func:`proxy.provider._refusal_retry_timing` already decided the refusal is
  retryable; this module never re-classifies a refusal.
* **Bounded memory.** Entries are TTL'd (``TTL_SECONDS``) and the store is
  capped (``MAX_ENTRIES``, oldest-first eviction), so a long-lived proxy cannot
  accumulate per-session state without bound.
* **No stable key, no budget.** A request whose session id cannot be resolved
  keeps today's unbounded behaviour: there is nothing to count it against.
* **Reconfiguration resets.** Each entry remembers the threshold it was counted
  under; when ``local_deferral_max_refusals`` changes the stale count is
  discarded rather than applied against a different budget.

The clock is read through :func:`_now` on purpose: the contract specs freeze it
(``local_preference_v2_support.frozen_clock``) so TTL expiry is asserted
deterministically instead of by sleeping.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

#: How long a session's refusals stay relevant (seconds). A session that has
#: not been refused for an hour has long since moved on, so its count is not a
#: budget any more.
TTL_SECONDS: float = 3600.0

#: Hard cap on tracked sessions. Oldest-first eviction keeps the store bounded
#: under a session-id-spraying client (parent AC8: no unbounded growth).
MAX_ENTRIES: int = 4096

#: Documented default threshold (``proxy/config.yaml`` mirrors this value).
DEFAULT_MAX_REFUSALS: int = 3


@dataclass
class _Entry:
    """One session's budget state."""

    count: int
    recorded_at: float
    threshold: int


#: ``session_id -> _Entry``. Insertion order is the eviction order, and a
#: re-recorded session moves to the end (most-recently-used survives).
_entries: OrderedDict[str, _Entry] = OrderedDict()


def _now() -> float:
    """The store's single clock (patched by the contract specs)."""
    return time.monotonic()


def _is_expired(entry: _Entry) -> bool:
    return (_now() - entry.recorded_at) > TTL_SECONDS


def _live_entry(session_id: str, threshold: int) -> _Entry | None:
    """Return *session_id*'s entry when it is fresh and matches *threshold*."""
    entry = _entries.get(session_id)
    if entry is None:
        return None
    if _is_expired(entry) or entry.threshold != threshold:
        # Expired (TTL) or counted under a different budget (reconfiguration):
        # the stale count must not escalate anybody.
        _entries.pop(session_id, None)
        return None
    return entry


def _evict_over_capacity() -> None:
    while len(_entries) > MAX_ENTRIES:
        _entries.popitem(last=False)


def refusal_count(session_id: str | None, *, threshold: int = DEFAULT_MAX_REFUSALS) -> int:
    """Consecutive retryable refusals recorded for *session_id*.

    Returns ``0`` for a missing session id (no stable key, no budget), for an
    unknown session, for an expired entry and for an entry counted under a
    different threshold.
    """
    if not session_id:
        return 0
    entry = _live_entry(session_id, threshold)
    return entry.count if entry is not None else 0


def record_refusal(
    session_id: str | None, *, threshold: int = DEFAULT_MAX_REFUSALS
) -> int:
    """Count one retryable refusal and return the session's new total.

    A missing session id records nothing and returns ``0``: today's unbounded
    deferral is preserved for keyless requests.
    """
    if not session_id:
        return 0
    entry = _live_entry(session_id, threshold)
    count = (entry.count if entry is not None else 0) + 1
    _entries[session_id] = _Entry(
        count=count, recorded_at=_now(), threshold=threshold
    )
    _entries.move_to_end(session_id)
    _evict_over_capacity()
    return count


def budget_spent(
    session_id: str | None, *, threshold: int = DEFAULT_MAX_REFUSALS
) -> bool:
    """True when *session_id* has used up its deferral budget.

    ``threshold <= 0`` disables the escalation entirely (the documented
    rollback), which is why that check lives here rather than at every caller.
    """
    if threshold <= 0 or not session_id:
        return False
    return refusal_count(session_id, threshold=threshold) >= threshold


def remaining(
    session_id: str | None, *, threshold: int = DEFAULT_MAX_REFUSALS
) -> int:
    """Refusals left before the escalation (``0`` once the budget is spent)."""
    if threshold <= 0:
        return 0
    return max(0, threshold - refusal_count(session_id, threshold=threshold))


def reset(session_id: str | None) -> None:
    """Drop exactly one session's budget (a local success, a promotion, ...)."""
    if session_id:
        _entries.pop(session_id, None)


def reset_all() -> None:
    """Drop every entry (test isolation and an explicit operator reset)."""
    _entries.clear()
