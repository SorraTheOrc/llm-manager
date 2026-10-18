"""Per-provider request-budget gating with session affinity (LP-0MVDQ1GMO00801PL).

Some remote providers — notably the MunitoAI free tier (``munitoai-free``) —
allow only a small number of requests per 60-second window. When many agents
share that allowance it is exhausted almost immediately, and blindly
dispatching into it produces upstream ``429 rate_limit_exceeded`` responses and
long stalls before the chain falls through to a paid tier.

This module turns the existing proactive sliding-window limiter
(:mod:`proxy.rate_limiter`) into a *capacity-aware* gate:

- **Budget** — an entry with ``rate_limit_rpm`` configured accepts at most that
  many dispatches in any rolling 60-second window (AC1).
- **New sessions pass through** — a request whose session has never been served
  by the gated entry is never parked: when the window is full it is denied
  immediately and the fallback chain moves on (AC2).
- **Reserve for in-progress sessions** — a fraction of the window
  (``server.provider_rate_limit_reserve_ratio``, default 0.3) is reserved for
  sessions that already use the entry: new sessions may consume at most
  ``rate_limit_rpm - reserve`` slots while affinity sessions may use the whole
  window (LP-0MVDUN5CY009VD20). Without the reserve, affinity sessions queue
  behind new-session traffic and wait 15–55 s per turn, which dominated
  first-token latency (p99 21 s → 53 s).
- **In-progress sessions wait, bounded** — a request whose session *has* been
  served by the entry (recorded affinity) may wait for the window to drain, so
  it keeps the same upstream provider and its prompt cache/prefix continuity.
  The wait is bounded by the remaining window and
  ``server.provider_rate_limit_wait_max_seconds`` (default 15), depth-capped by
  ``server.provider_rate_limit_wait_max_waiters`` so waiters cannot pile up, and
  aborted as soon as the client disconnects (AC3).
- **Affinity** — ``session -> entry`` records expire after
  ``server.provider_affinity_ttl_seconds`` (AC4), so a session that comes back
  much later is a *new* session again.

Fail-open: an entry without ``rate_limit_rpm`` (or with a malformed value) is
never gated, and no affinity is recorded for it.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, replace

logger = logging.getLogger("llama-proxy.provider_budget")

#: The budget window. Free tiers advertise a per-minute allowance, so the gate
#: always uses a 60-second rolling window.
RATE_LIMIT_WINDOW_SECONDS = 60

_DEFAULT_WAIT_MAX_SECONDS = 15.0
_DEFAULT_WAIT_MAX_WAITERS = 8
_DEFAULT_AFFINITY_TTL_SECONDS = 900.0
#: Fraction of a gated entry's window reserved for sessions that already use it.
_DEFAULT_RESERVE_RATIO = 0.3

#: Minimum sleep slice while waiting, so a zero-length recomputation cannot
#: spin the event loop.
_MIN_WAIT_SLICE_SECONDS = 0.05

#: Disconnect polling granularity while waiting.
_DISCONNECT_POLL_SECONDS = 0.5

#: Guard so a wait that wakes a hair before the window edge re-checks capacity
#: instead of sleeping a whole extra poll interval.
_WAIT_EPSILON_SECONDS = 0.001

# (session_id, entry_name) -> expiry (monotonic seconds)
_affinity: dict[tuple[str, str], float] = {}
_waiters = 0


@dataclass(frozen=True)
class BudgetDecision:
    """Outcome of a per-provider budget check.

    ``reason`` is one of:

    - ``unlimited`` — the entry has no ``rate_limit_rpm`` (not gated);
    - ``allowed`` — a slot was available immediately;
    - ``allowed_after_wait`` — an in-progress session waited and got a slot;
    - ``rate_limited`` — over budget, new session: passed through immediately;
    - ``rate_limited_after_wait`` — an in-progress session's wait ran out;
    - ``rate_limited_waiters`` — waiter depth cap reached: passed through;
    - ``rate_limited_disconnect`` — the client disconnected during the wait.
    """

    allowed: bool
    reason: str
    limit_rpm: int = 0
    remaining: int = 0
    waited_seconds: float = 0.0
    #: Slots held back for in-progress sessions on this entry (0 = no reserve).
    reserve_slots: int = 0
    #: Per-window cap actually applied to this request (``limit_rpm`` for an
    #: affinity session, ``limit_rpm - reserve_slots`` for a new one).
    effective_limit: int = 0

    @property
    def waited(self) -> bool:
        """True when this decision involved any waiting."""
        return self.waited_seconds > 0.0


# ---------------------------------------------------------------------------
# Config helpers (flat-first, then ``server.*`` — same pattern as provider.py)
# ---------------------------------------------------------------------------


def _config_value(config: dict | None, key: str, default):
    if not isinstance(config, dict):
        return default
    value = config.get(key)
    if value is None:
        server = config.get("server")
        if isinstance(server, dict):
            value = server.get(key)
    return default if value is None else value


def wait_max_seconds(config: dict | None) -> float:
    """Maximum time an in-progress session may wait for free-tier capacity."""
    try:
        return max(
            0.0, float(_config_value(config, "provider_rate_limit_wait_max_seconds", _DEFAULT_WAIT_MAX_SECONDS))
        )
    except (TypeError, ValueError):
        return _DEFAULT_WAIT_MAX_SECONDS


def wait_max_waiters(config: dict | None) -> int:
    """Maximum number of requests allowed to wait for capacity concurrently."""
    try:
        return max(
            0,
            int(_config_value(config, "provider_rate_limit_wait_max_waiters", _DEFAULT_WAIT_MAX_WAITERS)),
        )
    except (TypeError, ValueError):
        return _DEFAULT_WAIT_MAX_WAITERS


def affinity_ttl_seconds(config: dict | None) -> float:
    """How long a session is considered 'in progress' on a gated entry."""
    try:
        return max(
            0.0, float(_config_value(config, "provider_affinity_ttl_seconds", _DEFAULT_AFFINITY_TTL_SECONDS))
        )
    except (TypeError, ValueError):
        return _DEFAULT_AFFINITY_TTL_SECONDS


def reserve_ratio(config: dict | None) -> float:
    """Fraction of a gated entry's window reserved for in-progress sessions.

    Clamped to ``[0.0, 1.0]``; malformed values fall back to the default.
    """
    try:
        ratio = float(
            _config_value(config, "provider_rate_limit_reserve_ratio", _DEFAULT_RESERVE_RATIO)
        )
    except (TypeError, ValueError):
        ratio = _DEFAULT_RESERVE_RATIO
    return min(1.0, max(0.0, ratio))


def reserved_slots(rpm: int, config: dict | None) -> int:
    """Slots in a ``rpm``-sized window held back for in-progress sessions.

    ``0`` when the reserve is disabled, when the budget is too small to split
    (``rpm <= 1``), and always at most ``rpm - 1`` so new sessions can never be
    locked out of the tier entirely.
    """
    if rpm <= 1:
        return 0
    ratio = reserve_ratio(config)
    if ratio <= 0.0:
        return 0
    return max(0, min(int(math.floor(rpm * ratio)), rpm - 1))


def new_session_limit(rpm: int, config: dict | None) -> int:
    """Per-window cap for a session with no affinity for the entry."""
    return max(1, rpm - reserved_slots(rpm, config))


def _provider_rpm(provider_cfg: dict) -> int:
    """Read ``rate_limit_rpm`` from a provider entry (0 = not gated)."""
    try:
        return max(0, int(provider_cfg.get("rate_limit_rpm", 0) or 0))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Session affinity
# ---------------------------------------------------------------------------


def record_provider_affinity(
    session_id: str | None,
    entry_name: str,
    config: dict | None = None,
) -> None:
    """Record that *session_id* was successfully served by *entry_name* (AC4).

    No-op for anonymous requests (no session identity) and for a zero TTL.
    """
    if not session_id or not entry_name:
        return
    ttl = affinity_ttl_seconds(config)
    if ttl <= 0:
        return
    _affinity[(session_id, entry_name)] = time.monotonic() + ttl
    _prune_affinity()


def provider_affinity_active(
    session_id: str | None,
    entry_name: str,
    config: dict | None = None,
) -> bool:
    """True when *session_id* has an unexpired affinity record for *entry_name*."""
    if not session_id or not entry_name:
        return False
    expiry = _affinity.get((session_id, entry_name))
    if expiry is None:
        return False
    if time.monotonic() >= expiry:
        _affinity.pop((session_id, entry_name), None)
        return False
    return True


def _prune_affinity() -> None:
    """Drop expired affinity records (cheap, called on writes)."""
    if not _affinity:
        return
    now = time.monotonic()
    for key, expiry in list(_affinity.items()):
        if now >= expiry:
            del _affinity[key]


def reset_provider_budget_state() -> None:
    """Clear affinity records and waiter accounting (used by tests)."""
    global _waiters
    _affinity.clear()
    _waiters = 0


# ---------------------------------------------------------------------------
# Bounded wait
# ---------------------------------------------------------------------------


async def _sleep_with_disconnect(request, seconds: float) -> bool:
    """Sleep *seconds*, polling for a client disconnect.

    Returns True when the client disconnected (the caller must stop waiting),
    False when the full slice elapsed. A request without ``is_disconnected``
    (unit tests, internal callers) simply sleeps.
    """
    if seconds <= 0:
        return False
    check_disconnect = getattr(request, "is_disconnected", None)
    if check_disconnect is None:
        await asyncio.sleep(seconds)
        return False

    remaining = seconds
    while remaining > 0:
        try:
            if await check_disconnect():
                return True
        except Exception:  # pragma: no cover - defensive: a broken request object
            pass
        slice_seconds = min(_DISCONNECT_POLL_SECONDS, remaining)
        await asyncio.sleep(slice_seconds)
        remaining -= slice_seconds
    return False


async def _wait_for_capacity(
    entry_name: str,
    rpm: int,
    config: dict | None,
    request,
) -> BudgetDecision:
    """Wait (bounded, depth-capped) for a slot on *entry_name*.

    Returns an ``allowed_after_wait`` decision when a slot freed, or a denial
    naming the reason (wait cap, waiter depth, disconnect).
    """
    global _waiters

    from proxy.rate_limiter import get_rate_limiter

    limiter = get_rate_limiter()
    max_wait = wait_max_seconds(config)
    max_waiters = wait_max_waiters(config)

    if max_wait <= 0:
        # Waiting disabled by config (provider_rate_limit_wait_max_seconds: 0):
        # in-progress sessions pass through exactly like new sessions.
        logger.info(
            "Provider budget exhausted: skipping provider=%s limit=%d "
            "remaining=%d reason=rate_limited session_affinity=true wait_disabled=true",
            entry_name,
            rpm,
            limiter.remaining(entry_name, rpm, RATE_LIMIT_WINDOW_SECONDS),
        )
        return BudgetDecision(
            allowed=False,
            reason="rate_limited",
            limit_rpm=rpm,
            remaining=limiter.remaining(entry_name, rpm, RATE_LIMIT_WINDOW_SECONDS),
        )
    if _waiters >= max_waiters:
        logger.info(
            "Provider budget wait declined: provider=%s limit=%d "
            "reason=rate_limited_waiters waiters=%d max_waiters=%d",
            entry_name,
            rpm,
            _waiters,
            max_waiters,
        )
        return BudgetDecision(
            allowed=False,
            reason="rate_limited_waiters",
            limit_rpm=rpm,
            remaining=limiter.remaining(entry_name, rpm, RATE_LIMIT_WINDOW_SECONDS),
        )

    deadline = time.monotonic() + max_wait
    waited = 0.0
    _waiters += 1
    try:
        logger.info(
            "Provider budget wait start: provider=%s limit=%d "
            "remaining=%d max_wait=%.0fs session_affinity=true",
            entry_name,
            rpm,
            limiter.remaining(entry_name, rpm, RATE_LIMIT_WINDOW_SECONDS),
            max_wait,
        )
        while True:
            remaining_budget = deadline - time.monotonic()
            if remaining_budget <= 0:
                break
            delay = limiter.seconds_until_capacity(entry_name, rpm, RATE_LIMIT_WINDOW_SECONDS)
            slice_seconds = min(
                max(delay + _WAIT_EPSILON_SECONDS, _MIN_WAIT_SLICE_SECONDS),
                remaining_budget,
            )
            disconnected = await _sleep_with_disconnect(request, slice_seconds)
            if disconnected:
                logger.info(
                    "Provider budget wait aborted: provider=%s reason=client_disconnect "
                    "waited=%.2fs",
                    entry_name,
                    waited,
                )
                return BudgetDecision(
                    allowed=False,
                    reason="rate_limited_disconnect",
                    limit_rpm=rpm,
                    remaining=limiter.remaining(entry_name, rpm, RATE_LIMIT_WINDOW_SECONDS),
                    waited_seconds=waited,
                )
            waited += slice_seconds
            if await limiter.check_and_increment(
                entry_name, rpm, window_seconds=RATE_LIMIT_WINDOW_SECONDS
            ):
                logger.info(
                    "Provider budget wait satisfied: provider=%s limit=%d waited=%.2fs",
                    entry_name,
                    rpm,
                    waited,
                )
                return BudgetDecision(
                    allowed=True,
                    reason="allowed_after_wait",
                    limit_rpm=rpm,
                    remaining=limiter.remaining(entry_name, rpm, RATE_LIMIT_WINDOW_SECONDS),
                    waited_seconds=waited,
                )
    finally:
        _waiters -= 1

    logger.warning(
        "Provider budget wait exhausted: provider=%s limit=%d waited=%.2fs "
        "reason=rate_limited_after_wait",
        entry_name,
        rpm,
        waited,
    )
    return BudgetDecision(
        allowed=False,
        reason="rate_limited_after_wait",
        limit_rpm=rpm,
        remaining=limiter.remaining(entry_name, rpm, RATE_LIMIT_WINDOW_SECONDS),
        waited_seconds=waited,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def acquire_provider_budget(
    provider_cfg: dict,
    config: dict | None,
    session_id: str | None,
    request=None,
) -> BudgetDecision:
    """Decide whether *provider_cfg* may be dispatched for *session_id*.

    A successful decision reserves a slot (the limiter increments on allow).
    New sessions may only use the unreserved part of the window
    (``rpm - reserved_slots``) so the reserve is always available to sessions
    that are already using the entry:

    - an **in-progress** session (recorded affinity) may use the whole window;
      when even that is full it waits a bounded time for capacity and reserves
      a slot if one frees;
    - any other request (new session, anonymous request, wait cap reached) is
      denied immediately so the fallback chain can move on without waiting.

    Args:
        provider_cfg: Provider entry (must carry ``name``; ``rate_limit_rpm``
            opts the entry into gating).
        config: Server config (flat or ``server.*``) for the wait/TTL knobs.
        session_id: Header-resolved session identity, if any.
        request: Incoming request, used to abort a wait on client disconnect.
    """
    rpm = _provider_rpm(provider_cfg)
    if rpm <= 0:
        return BudgetDecision(allowed=True, reason="unlimited")

    entry_name = str(provider_cfg.get("name") or "")
    if not entry_name:
        return BudgetDecision(allowed=True, reason="unlimited")

    from proxy.rate_limiter import get_rate_limiter

    limiter = get_rate_limiter()
    reserve = reserved_slots(rpm, config)
    in_progress = provider_affinity_active(session_id, entry_name, config)
    # In-progress sessions may use the whole window; new sessions stop at the
    # unreserved part so slots are always waiting for established sessions.
    effective_limit = rpm if in_progress else new_session_limit(rpm, config)

    if await limiter.check_and_increment(
        entry_name, effective_limit, window_seconds=RATE_LIMIT_WINDOW_SECONDS
    ):
        return BudgetDecision(
            allowed=True,
            reason="allowed",
            limit_rpm=rpm,
            remaining=limiter.remaining(entry_name, rpm, RATE_LIMIT_WINDOW_SECONDS),
            reserve_slots=reserve,
            effective_limit=effective_limit,
        )

    if in_progress:
        decision = await _wait_for_capacity(entry_name, rpm, config, request)
        return replace(decision, reserve_slots=reserve, effective_limit=effective_limit)

    remaining = limiter.remaining(entry_name, rpm, RATE_LIMIT_WINDOW_SECONDS)

    logger.warning(
        "Provider budget exhausted: skipping provider=%s limit=%d reserve=%d "
        "remaining=%d reason=rate_limited session_affinity=false session=%s",
        entry_name,
        effective_limit,
        reserve,
        remaining,
        str(session_id or "")[:8] or "anonymous",
    )
    return BudgetDecision(
        allowed=False,
        reason="rate_limited",
        limit_rpm=rpm,
        remaining=remaining,
        reserve_slots=reserve,
        effective_limit=effective_limit,
    )
