"""
Slot Scheduler Module

Provides time-based slot scheduling for llama-server/proxy, allowing
operators to vary the number of concurrent slots (``--parallel N``)
based on the time of day.

Features:
- User-configurable schedule in ``config.yaml`` with time ranges and slot counts.
- Background scheduler that sleeps until the next transition time and triggers
  an immediate restart of llama-server with the new slot count.  There is no
  drain window and no request-rejection period (LP-0MSF9RUSQ007M346) — in-flight
  requests are terminated by the restart and clients retry.
- Enabled by default with a sensible schedule (10:00→4, 12:00→8). Disable
  by setting ``enabled: false`` or removing the ``slot_schedule`` section.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from datetime import time as dt_time
from typing import Any

logger = logging.getLogger("llama-proxy")


@dataclass
class SlotScheduleEntry:
    """A single schedule entry mapping a time-of-day to a slot count.

    ``ctx_size`` is an optional per-period context-size override: the total
    context across all slots (llama-server's ``--ctx-size``) that applies
    while this entry is active (LP-0MSLNK96T0018W4D). When ``None`` the
    global ``local_model_ctx_size`` from config applies.
    """

    time: dt_time
    slots: int
    ctx_size: int | None = None


@dataclass
class SlotScheduleConfig:
    """Parsed slot schedule configuration.

    Reads the ``slot_schedule`` section from the server config and provides
    helper methods for determining the active slot count at any given time
    of day.

    ``drain_minutes`` is parsed for backward compatibility but **ignored**:
    transitions apply immediately with no drain window (LP-0MSF9RUSQ007M346).
    """

    enabled: bool = False
    drain_minutes: int = 3  # Parsed but IGNORED (LP-0MSF9RUSQ007M346): no drain window
    entries: list[SlotScheduleEntry] = field(default_factory=list)

    def __init__(self, raw: dict[str, Any] | None):
        """Parse raw schedule config dict.

        Expects structure:
        .. code-block:: yaml

            slot_schedule:
              enabled: true
              drain_minutes: 3
              entries:
                - time: "10:00"
                  slots: 4
                - time: "12:00"
                  slots: 8

        When *raw* is ``None`` or empty, the schedule is disabled by default.
        """
        if not raw or not isinstance(raw, dict):
            self.enabled = False
            self.drain_minutes = 3  # Parsed but IGNORED (LP-0MSF9RUSQ007M346)
            self.entries = []
            return

        self.enabled = bool(raw.get("enabled", False))
        # drain_minutes is parsed but IGNORED (LP-0MSF9RUSQ007M346): transitions
        # have no drain window; kept only for backward compatibility.
        self.drain_minutes = int(raw.get("drain_minutes", 3) or 3)
        self.entries = []

        raw_entries = raw.get("entries", [])
        if not isinstance(raw_entries, list):
            raw_entries = []

        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            try:
                time_str = entry.get("time")
                slots = entry.get("slots")
                if not time_str or slots is None:
                    continue
                # Parse "HH:MM" format
                parts = str(time_str).strip().split(":")
                if len(parts) != 2:
                    continue
                hour = int(parts[0])
                minute = int(parts[1])
                if not (0 <= hour <= 23 and 0 <= minute <= 59):
                    continue
                # Optional per-period ctx_size (LP-0MSLNK96T0018W4D): total
                # context across all slots while this entry is active.
                # Invalid/absent values fall back to the global
                # ``local_model_ctx_size`` (represented as None).
                ctx_size: int | None = None
                raw_ctx = entry.get("ctx_size")
                if raw_ctx is not None:
                    try:
                        parsed_ctx = int(raw_ctx)
                        if parsed_ctx > 0:
                            ctx_size = parsed_ctx
                    except (ValueError, TypeError):
                        ctx_size = None
                self.entries.append(
                    SlotScheduleEntry(
                        time=dt_time(hour, minute),
                        slots=int(slots),
                        ctx_size=ctx_size,
                    )
                )
            except (ValueError, TypeError):
                continue

        # Sort entries by time
        self.entries.sort(key=lambda e: e.time)

    @classmethod
    def from_server_config(
        cls, server_config: dict[str, Any] | None
    ) -> "SlotScheduleConfig":
        """Extract slot schedule from the server configuration dict.

        Reads the ``slot_schedule`` key from *server_config*.  Returns a
        disabled config when the key is absent or *server_config* is None.
        """
        if not server_config or not isinstance(server_config, dict):
            return cls(None)
        raw = server_config.get("slot_schedule")
        return cls(raw)

    def get_active_entry(self, now: dt_time | None = None) -> SlotScheduleEntry | None:
        """Return the schedule entry active at *now*, or None if no entry matches.

        Returns the most recent schedule entry whose time is at or before
        *now*.  If no entry has been reached yet today, the schedule wraps
        circularly to the last entry (persisting from the previous
        fast/cheap period).

        Returns ``None`` only when:
        - The schedule is disabled.
        - No entries are configured (caller should use the static slot count).
        """
        if not self.enabled or not self.entries:
            return None

        now = now or datetime.now().time()

        # Walk entries in order; find the last one whose time <= now.
        last_matching: SlotScheduleEntry | None = None
        for entry in self.entries:
            if entry.time <= now:
                last_matching = entry
            else:
                break

        if last_matching is not None:
            return last_matching

        # Before the first entry of the day — the schedule wraps circularly,
        # so the last entry from the previous day applies.
        return self.entries[-1]

    def get_active_slot(self, now: dt_time | None = None) -> int | None:
        """Return the slot count active at *now*, or None if no entry matches.

        Returns the slot count of the most recent schedule entry whose time
        is at or before *now*.  If no entry has been reached yet today,
        the schedule wraps circularly to the last entry (persisting from
        the previous fast/cheap period).

        Returns ``None`` only when:
        - The schedule is disabled.
        - No entries are configured (caller should use the static slot count).
        """
        entry = self.get_active_entry(now)
        return entry.slots if entry is not None else None

    def get_active_ctx_size(self, now: dt_time | None = None) -> int | None:
        """Return the ctx_size of the entry active at *now*.

        ``None`` means the active entry has no per-period override (callers
        fall back to the global ``local_model_ctx_size`` from config).
        """
        entry = self.get_active_entry(now)
        return entry.ctx_size if entry is not None else None

    def _get_next_entry(self, now: dt_time) -> SlotScheduleEntry | None:
        """Return the next schedule entry after *now*.

        Handles wrapping: if no entry remains today, returns the first entry
        (interpreted as the next day). Returns None for disabled or empty schedules.
        """
        if not self.enabled or not self.entries:
            return None

        for entry in self.entries:
            if entry.time > now:
                return entry

        # All entries have passed — wrap to the first entry (next day).
        return self.entries[0]

    def _get_next_entry_time(self, now: dt_time) -> tuple[dt_time, int] | None:
        """Return the (time, slots) of the next schedule entry after *now*.

        Handles wrapping: if no entry remains today, returns the first entry
        (interpreted as the next day). Returns None for disabled or empty schedules.
        """
        entry = self._get_next_entry(now)
        if entry is None:
            return None
        return (entry.time, entry.slots)


class SlotScheduler:
    """Background scheduler for time-based slot count transitions.

    Instead of polling on a fixed interval, the scheduler calculates the
    exact time until the next relevant event (a transition deadline) and
    sleeps only until then.  This avoids hundreds of unnecessary wake-ups
    per day.

    The scheduler:
    1. Computes seconds until the next interesting event.
    2. Sleeps exactly that long (adaptive sleep).
    3. On wake, runs the evaluation cycle.
    4. Repeats — each cycle recomputes the next target.

    Events the scheduler cares about:
    - A transition deadline (the time a new slot count takes effect).
    - The next event after a pending restart was executed.

    Usage::

        scheduler = SlotScheduler(srv)
        asyncio.create_task(scheduler.run())

    The scheduler is disabled by default (no schedule → no-op).
    """

    # Maximum sleep between checks (24 hours).  In practice the next
    # transition is always within 24 h, but this cap prevents infinity.
    _MAX_SLEEP_SECONDS: float = 86400.0
    # Minimum sleep — avoid busy-waiting when events are sub-second apart.
    _MIN_SLEEP_SECONDS: float = 1.0

    def __init__(self, srv):
        """Initialize the scheduler.

        Args:
            srv: The server module (``proxy.server``) for access to config,
                 lifecycle functions, and logging.
        """
        self._srv = srv
        self._config: SlotScheduleConfig = SlotScheduleConfig.from_server_config(
            srv.config.get("server", {}) if isinstance(srv.config, dict) else None
        )
        self._pending_restart_slot: int | None = None
        self._pending_restart_ctx: int | None = None
        self._task: asyncio.Task | None = None
        # Base context size captured at construction (before any transition
        # mutates config ``local_model_ctx_size``). Schedule entries without
        # an explicit ``ctx_size`` fall back to this value
        # (LP-0MSLNK96T0018W4D).
        self._base_ctx_size = self._get_static_ctx_size()

    @property
    def enabled(self) -> bool:
        """Whether the scheduler is active (has a configured, enabled schedule)."""
        return self._config.enabled and len(self._config.entries) > 0

    @property
    def pending_restart_slot(self) -> int | None:
        """The slot count for the pending restart, or None if no restart is pending."""
        return self._pending_restart_slot

    @property
    def pending_restart_ctx(self) -> int | None:
        """The ctx_size for the pending restart (None = fall back to global)."""
        return self._pending_restart_ctx

    def get_active_slot(self, now: dt_time | None = None) -> int | None:
        """Schedule-aware active slot count (delegates to the schedule config).

        Exposed on the scheduler so routing can prefer the live schedule
        (LP-0MSLNK96T0018W4D).
        """
        return self._config.get_active_slot(now)

    def get_active_ctx_size(self, now: dt_time | None = None) -> int | None:
        """Schedule-aware active ctx_size (None = no per-period override)."""
        return self._config.get_active_ctx_size(now)

    def set_pending_restart(
        self, slot_count: int, ctx_size: int | None = None
    ) -> None:
        """Mark a restart as pending with the given slot count and ctx_size."""
        self._pending_restart_slot = slot_count
        self._pending_restart_ctx = ctx_size

    def clear_pending_restart(self) -> None:
        """Clear the pending restart flag (and its ctx_size)."""
        self._pending_restart_slot = None
        self._pending_restart_ctx = None

    def _now(self) -> dt_time:
        """Return the current time-of-day.  PATCHABLE in tests."""
        return datetime.now().time()

    async def start(self) -> None:
        """Start the background scheduler loop.

        Creates an asyncio task that runs ``_check_loop``.  The loop uses
        adaptive sleep: it calculates the exact time until the next
        transition deadline and sleeps only that long.
        """
        if not self.enabled:
            logger.info("Slot scheduler: disabled, not starting background loop")
            return

        entries_desc = ", ".join(
            f'{e.time.strftime("%H:%M")}→{e.slots}'
            for e in self._config.entries
        )
        logger.info(
            "Slot scheduler: starting (entries=%d: %s)",
            len(self._config.entries),
            entries_desc,
        )
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._check_loop())

    async def stop(self) -> None:
        """Stop the background scheduler loop."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
            logger.info("Slot scheduler: stopped")

    async def _check_loop(self) -> None:
        """Adaptive check loop that sleeps until the next interesting event.

        On each iteration:
        1. Run the evaluation cycle (transition / clear).
        2. Calculate the exact seconds until the next relevant event.
        3. Sleep for that duration.

        This avoids hundreds of unnecessary 60-second polls per day.
        """
        while True:
            try:
                await self._run_check_cycle()
                sleep_seconds = self._calculate_sleep_seconds()
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)
            except asyncio.CancelledError:
                logger.info("Slot scheduler: check loop cancelled")
                return
            except Exception:
                logger.exception("Slot scheduler: unexpected error in check loop, continuing...")
                await asyncio.sleep(self._MIN_SLEEP_SECONDS)

    async def _run_check_cycle(self) -> None:
        """Perform a single schedule evaluation cycle.

        Called periodically by the check loop.  Examines the current time
        against the schedule and decides whether to:
        - Arm the next slot-count transition (no drain window).
        - Trigger a restart (transition time arrived).
        - Clear pending state (no restart needed because the static config
          already matches).

        The "current slot count" is determined by:
        1. The schedule's ``get_active_slot()`` (last matched entry).
        2. If that returns ``None`` (before first entry), the static
           ``session_slot_pool_size`` from config.

        All exceptions are caught and logged — a single bad cycle never
        crashes the background loop.
        """
        try:
            await self._run_check_cycle_inner()
        except Exception:
            logger.exception("Slot scheduler: error in check cycle")

    def _get_static_slot_count(self) -> int:
        """Return the static ``session_slot_pool_size`` from config.

        This is the initial slot count before any schedule transition has
        been performed.  Returns 1 as a safe default.
        """
        try:
            server_cfg = getattr(self._srv, 'config', {}).get("server", {})
            return int(server_cfg.get("session_slot_pool_size", 1) or 1)
        except Exception:
            return 1

    def _get_static_ctx_size(self) -> int:
        """Return the static ``local_model_ctx_size`` from config.

        This is the initial context size before any schedule transition has
        been performed. Returns 0 when unset (clamp disabled).
        """
        try:
            server_cfg = getattr(self._srv, 'config', {}).get("server", {})
            return int(server_cfg.get("local_model_ctx_size", 0) or 0)
        except Exception:
            return 0

    def _effective_ctx_size(self, entry: SlotScheduleEntry | None) -> int:
        """Resolve the effective context size for a schedule entry.

        Per-period ``ctx_size`` when set, else the base global value captured
        at construction (LP-0MSLNK96T0018W4D).
        """
        if entry is None or entry.ctx_size is None:
            return self._base_ctx_size
        return entry.ctx_size

    async def _run_check_cycle_inner(self) -> None:
        """Inner implementation of _run_check_cycle (no exception wrapping).

        Separated to allow tests to call the inner logic directly without
        exception shielding.
        """
        if not self.enabled:
            return

        # Don't start a new restart if a model switch is already in
        # progress (prevents overlapping restarts).
        try:
            if getattr(self._srv, 'model_switch_refcount', 0) > 0:
                return
        except Exception:
            pass

        # ── Catch-up: if we started after a transition time, apply it now ──
        if self._pending_restart_slot is None:
            now = self._now()
            active_entry = self._config.get_active_entry(now)
            static_slots = self._get_static_slot_count()
            if active_entry is not None and (
                active_entry.slots != static_slots
                or self._effective_ctx_size(active_entry) != self._get_static_ctx_size()
            ):
                logger.info(
                    "Slot scheduler: catch-up detected — should be at %d slots "
                    "(currently at %d) per schedule; applying transition now",
                    active_entry.slots,
                    static_slots,
                )
                self.set_pending_restart(active_entry.slots, active_entry.ctx_size)
                await self.perform_restart()
                return

        now = self._now()

        # Determine the "current running" slot count for transition
        # comparison:
        #   - Before any transition has been performed (pending_restart is
        #     None), the system is running the static config value.
        #   - After a transition has been detected (pending_restart is set),
        #     the "current" value comes from the schedule's wrapping.
        #   - If pending_restart just executed and cleared, we start fresh.
        schedule_current = self._config.get_active_slot(now)
        schedule_current_ctx = self._config.get_active_ctx_size(now)
        static_slots = self._get_static_slot_count()
        static_ctx = self._get_static_ctx_size()

        if self._pending_restart_slot is not None:
            # We're in a transition cycle — use schedule's wrapped value
            # for comparison, since at least one transition has been
            # performed or is in progress.
            current_slots = schedule_current
            current_ctx = schedule_current_ctx
        else:
            # Before any transition — use static config as the baseline.
            # This ensures a single-entry schedule like [12:00→8] correctly
            # detects that a transition IS needed (static 4 != 8).
            current_slots = static_slots
            current_ctx = static_ctx

        # ── Phase 1: Check if a pending restart should execute now ──────────
        if self._pending_restart_slot is not None:
            now_dt = self._now_dt()

            # Find the entry that matches our pending (slots, ctx) pair.
            # Matching on BOTH values disambiguates entries that share a
            # slot count but differ in per-period ctx_size
            # (LP-0MSLNK96T0018W4D).
            matched_entry = None
            for entry in self._config.entries:
                if (
                    entry.slots == self._pending_restart_slot
                    and entry.ctx_size == self._pending_restart_ctx
                ):
                    matched_entry = entry
                    break

            if matched_entry is not None:
                entry_dt = datetime.combine(now_dt.date(), matched_entry.time)
                diff_seconds = (now_dt - entry_dt).total_seconds()

                if diff_seconds >= -1.0:
                    # Transition time reached:
                    # - diff >= 0: entry time is in the past (catch-up)
                    # - -1s < diff < 0: entry is slightly in the future
                    #   (microsecond timing jitter tolerance)
                    logger.info(
                        "Slot scheduler: transition time reached for %d slots at %s "
                        "(diff=%.3fs)",
                        matched_entry.slots,
                        matched_entry.time.strftime("%H:%M"),
                        diff_seconds,
                    )
                    await self.perform_restart()
                    return

                # ── Missed transition: entry time has passed ──────────
                # The entry time is more than 1s in the future today.
                # This means the pending restart slot's conceptual
                # transition was missed (would sleep 24h for this entry).
                # Clear pending so the scheduler falls through to normal
                # evaluation.
                logger.warning(
                    "Slot scheduler: missed transition window for %d slots "
                    "at %s — time has passed, clearing pending restart",
                    matched_entry.slots,
                    matched_entry.time.strftime("%H:%M"),
                )
                self.clear_pending_restart()
                return

            # If the pending (slots, ctx) matches the current active entry
            # without needing a restart (e.g., static config already
            # matches) and the transition time has not yet passed, just
            # clear silently.
            if (
                self._pending_restart_slot == current_slots
                and self._pending_restart_ctx == current_ctx
            ):
                self.clear_pending_restart()
                return

        # ── Phase 2: Arm the next slot-count transition (no drain window) ──
        # Find the next schedule entry whose slot count differs from the
        # current value and arm the restart.  Phase 1 executes the restart
        # as soon as the entry's transition time arrives.  Requests are
        # never rejected in the meantime — there is no drain window
        # (LP-0MSF9RUSQ007M346).
        next_entry = self._config._get_next_entry(now)
        if next_entry is not None:
            next_time = next_entry.time
            if (next_entry.slots, self._effective_ctx_size(next_entry)) != (
                current_slots,
                current_ctx,
            ):
                logger.info(
                    "Slot scheduler: arming transition to %d slots at %s (current=%s)",
                    next_entry.slots,
                    next_time.strftime("%H:%M"),
                    current_slots,
                )
                self.set_pending_restart(next_entry.slots, next_entry.ctx_size)
                return

    # ────────────────────────────────────────────────────────────────────
    # Adaptive sleep calculation
    # ────────────────────────────────────────────────────────────────────

    def _now_dt(self) -> datetime:
        """Return the current datetime.  PATCHABLE in tests."""
        return datetime.now()

    def _seconds_until(self, target: dt_time) -> float:
        """Return the number of seconds from now until the next *target* time-of-day.

        If *target* has already passed today, wraps to tomorrow.
        Returns at least ``_MIN_SLEEP_SECONDS`` and at most ``_MAX_SLEEP_SECONDS``.
        """
        now_dt = self._now_dt()
        target_dt = datetime.combine(now_dt.date(), target)
        if target_dt <= now_dt:
            target_dt += timedelta(days=1)
        seconds = (target_dt - now_dt).total_seconds()
        return max(self._MIN_SLEEP_SECONDS, min(seconds, self._MAX_SLEEP_SECONDS))

    def _calculate_sleep_seconds(self) -> float:
        """Calculate seconds until the next action is needed.

        Returns the number of seconds to sleep before the next
        ``_run_check_cycle`` should run.  The calculation depends on
        the scheduler's current state:

        **Pending restart**
            Sleep until the transition time of the matching schedule entry.

        **Normal (no pending work)**
            Find the next schedule entry where the slot count differs from
            the current value and sleep until its transition time.  There
            is no drain window (LP-0MSF9RUSQ007M346).

        If no future event differs (all entries match the current slot),
        sleep the full maximum duration (24 h).
        """
        if not self.enabled:
            return self._MAX_SLEEP_SECONDS

        now_time = self._now()

        # ── Pending restart → wake when the matching entry's time arrives ──
        if self._pending_restart_slot is not None:
            now_dt = self._now_dt()
            for entry in self._config.entries:
                if (
                    entry.slots == self._pending_restart_slot
                    and entry.ctx_size == self._pending_restart_ctx
                ):
                    entry_dt = datetime.combine(now_dt.date(), entry.time)
                    if entry_dt <= now_dt:
                        # Entry time has already passed today — fall through
                        # to normal schedule evaluation instead of sleeping
                        # until tomorrow (24h freeze bug).
                        logger.warning(
                            "Slot scheduler: pending slot %d time has passed, "
                            "falling through to normal schedule",
                            entry.slots,
                        )
                        break
                    return self._seconds_until(entry.time)
            # Pending time passed — clear and fall through
            self.clear_pending_restart()

        # ── Determine current slot count for comparison ───────────────
        static_slots = self._get_static_slot_count()
        static_ctx = self._get_static_ctx_size()
        current_slots = static_slots  # before first transition
        current_ctx = static_ctx

        # ── Find the next entry where (slots, ctx) actually changes ──
        for entry in self._config.entries:
            if entry.time > now_time:
                if (
                    entry.slots != current_slots
                    or self._effective_ctx_size(entry) != current_ctx
                ):
                    # No drain window — sleep straight through to the
                    # transition time (LP-0MSF9RUSQ007M346).
                    return self._seconds_until(entry.time)
                break  # no change → skip this transition

        # ── No transition with a different slot count exists today ────
        # All remaining entries match or are no-ops.  Sleep until the
        # first entry tomorrow (the schedule wraps).
        if self._config.entries:
            return self._seconds_until(self._config.entries[0].time)

        return self._MAX_SLEEP_SECONDS

    async def perform_restart(self) -> bool:
        """Execute the pending restart of llama-server with the new slot count.

        Calls ``restart_services(slot_count=..., reason="scheduled_slot_change")``
        on the server module to perform the restart. When the pending entry
        carries a per-period ctx_size (or falls back to the base global
        value), ``ctx_size`` is passed so llama-server restarts with the new
        context size AND the routing clamp derives thresholds from the active
        period (LP-0MSLNK96T0018W4D).

        Returns ``True`` if the restart was initiated, ``False`` if no pending
        restart slot was set or if an error occurred.
        """
        slot_count = self._pending_restart_slot
        if slot_count is None:
            return False

        # Resolve the ctx_size for this transition: the pending entry's
        # per-period override, else the base global value (None when the
        # clamp is disabled — local_model_ctx_size 0/absent).
        ctx_size = self._pending_restart_ctx
        if ctx_size is None:
            ctx_size = self._base_ctx_size

        try:
            logger.info(
                "Slot scheduler: performing restart with %d slots",
                slot_count,
            )
            kwargs: dict[str, Any] = {
                "slot_count": slot_count,
                "reason": "scheduled_slot_change",
            }
            if ctx_size is not None and ctx_size > 0:
                kwargs["ctx_size"] = ctx_size
            result = await self._srv.restart_services(**kwargs)
            self.clear_pending_restart()
            return bool(result)
        except Exception:
            logger.exception(
                "Slot scheduler: restart failed for %d slots",
                slot_count,
            )
            self.clear_pending_restart()
            return False
