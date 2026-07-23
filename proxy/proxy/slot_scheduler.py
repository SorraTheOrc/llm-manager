"""
Slot Scheduler Module

Provides time-based slot scheduling for llama-server/proxy, allowing
operators to vary the number of concurrent slots (``--parallel N``)
based on the time of day.

Features:
- User-configurable schedule in ``config.yaml`` with time ranges and slot counts.
- Automatic drain phase before transitions (configurable drain window).
- Background scheduler that periodically checks the schedule and triggers
  graceful restart of llama-server with the new slot count.
- Disabled by default (no schedule configured → current static behavior).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta
from typing import Any

logger = logging.getLogger("llama-proxy")


@dataclass
class SlotScheduleEntry:
    """A single schedule entry mapping a time-of-day to a slot count."""

    time: dt_time
    slots: int


@dataclass
class SlotScheduleConfig:
    """Parsed slot schedule configuration.

    Reads the ``slot_schedule`` section from the server config and provides
    helper methods for determining the active slot count and drain window
    at any given time of day.
    """

    enabled: bool = False
    drain_minutes: int = 15
    entries: list[SlotScheduleEntry] = field(default_factory=list)

    def __init__(self, raw: dict[str, Any] | None):
        """Parse raw schedule config dict.

        Expects structure:
        .. code-block:: yaml

            slot_schedule:
              enabled: true
              drain_minutes: 15
              entries:
                - time: "10:00"
                  slots: 4
                - time: "12:00"
                  slots: 8

        When *raw* is ``None`` or empty, the schedule is disabled by default.
        """
        if not raw or not isinstance(raw, dict):
            self.enabled = False
            self.drain_minutes = 15
            self.entries = []
            return

        self.enabled = bool(raw.get("enabled", False))
        self.drain_minutes = int(raw.get("drain_minutes", 15) or 15)
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
                self.entries.append(
                    SlotScheduleEntry(time=dt_time(hour, minute), slots=int(slots))
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

    def get_active_slot(self, now: dt_time | None = None) -> int | None:
        """Return the slot count active at *now*, or None if no entry matches.

        Returns the slot count of the most recent schedule entry whose time
        is at or before *now*.  If no entry has been reached yet today,
        the schedule wraps circularly to the last entry (persisting from
        the previous day/night).

        Returns ``None`` only when:
        - The schedule is disabled.
        - No entries are configured (caller should use the static slot count).
        """
        if not self.enabled or not self.entries:
            return None

        now = now or datetime.now().time()

        # Walk entries in order; find the last one whose time <= now.
        last_matching: int | None = None
        for entry in self.entries:
            if entry.time <= now:
                last_matching = entry.slots
            else:
                break

        if last_matching is not None:
            return last_matching

        # Before the first entry of the day — the schedule wraps circularly,
        # so the last entry from the previous day applies.
        return self.entries[-1].slots

    def _get_next_entry_time(self, now: dt_time) -> tuple[dt_time, int] | None:
        """Return the (time, slots) of the next schedule entry after *now*.

        Handles wrapping: if no entry remains today, returns the first entry
        (interpreted as the next day). Returns None for disabled or empty schedules.
        """
        if not self.enabled or not self.entries:
            return None

        for entry in self.entries:
            if entry.time > now:
                return (entry.time, entry.slots)

        # All entries have passed — wrap to the first entry (next day).
        first = self.entries[0]
        return (first.time, first.slots)

    def get_slot_at_entry(self, target_time: dt_time) -> int | None:
        """Return the slot count for the entry closest to *target_time*."""
        if not self.enabled or not self.entries:
            return None
        for entry in self.entries:
            if entry.time == target_time:
                return entry.slots
        return None

    def is_in_drain_window(
        self,
        now: dt_time | None = None,
        current_slots: int | None = None,
    ) -> bool | tuple[bool, int | None]:
        """Check if the current time is within a drain window before any transition.

        A drain window is the ``drain_minutes`` period immediately before a schedule
        entry where the slot count will change. During this window, the proxy should
        refuse new requests and drain in-flight workloads.

        The method checks whether the slot count actually changes at the next
        transition — if the count is the same as the current active slot, no drain
        is needed.

        Args:
            now: The current time-of-day. Defaults to ``datetime.now().time()``.
            current_slots: The currently active slot count (from server config or
                the last matched entry). When ``None``, uses ``get_active_slot(now)``
                which returns ``None`` before the first transition (static config
                should be compared separately if needed).

        Returns:
            ``True`` when draining should be active.  Returns ``(True, next_slot_count)``
            when migrating to new dispatch handlers.  Returns ``False`` when no drain
            is needed.
        """
        if not self.enabled or not self.entries:
            return False

        now = now or datetime.now().time()
        if current_slots is None:
            current_slots = self.get_active_slot(now)

        # Find the next transition where the slot count actually changes.
        for entry in self.entries:
            if entry.time > now:
                # Only trigger drain if slot count actually changes.
                if entry.slots != current_slots:
                    drain_start = self._drain_start_time(entry.time)
                    if drain_start <= now < entry.time:
                        return (True, entry.slots)
                break

        # No upcoming transition found today (all entries passed).
        return False

    def _drain_start_time(self, transition_time: dt_time) -> dt_time:
        """Compute the start time of the drain window for a transition.

        Subtracts ``drain_minutes`` from *transition_time*, handling
        wrapping past midnight (returns a time that may be > transition_time
        if the drain window started before midnight).
        """
        delta = timedelta(minutes=self.drain_minutes)
        transition_dt = datetime.combine(datetime.today(), transition_time)
        drain_dt = transition_dt - delta
        return drain_dt.time()


class SlotScheduler:
    """Background scheduler for time-based slot count transitions.

    Runs a periodic check (every 60 seconds) to:
    1. Check if we're inside a drain window approaching a slot transition.
    2. Set ``draining`` flag to prevent new requests during drain.
    3. When the transition time arrives, stop draining and trigger a
       graceful restart of llama-server with the new slot count.

    Usage::

        scheduler = SlotScheduler(srv)
        asyncio.create_task(scheduler.run())

    The scheduler is disabled by default (no schedule → no-op).
    """

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
        self._draining: bool = False
        self._pending_restart_slot: int | None = None
        self._task: asyncio.Task | None = None
        self._check_interval: float = 60.0  # check every 60 seconds

    @property
    def enabled(self) -> bool:
        """Whether the scheduler is active (has a configured, enabled schedule)."""
        return self._config.enabled and len(self._config.entries) > 0

    @property
    def draining(self) -> bool:
        """Whether the proxy is currently in drain mode.

        When ``True``, new requests should receive a 503 ``Service Unavailable``
        response with a ``retry-after`` header.
        """
        return self._draining

    def set_draining(self, value: bool) -> None:
        """Set the draining flag."""
        self._draining = value

    @property
    def pending_restart_slot(self) -> int | None:
        """The slot count for the pending restart, or None if no restart is pending."""
        return self._pending_restart_slot

    def set_pending_restart(self, slot_count: int) -> None:
        """Mark a restart as pending with the given slot count."""
        self._pending_restart_slot = slot_count

    def clear_pending_restart(self) -> None:
        """Clear the pending restart flag."""
        self._pending_restart_slot = None

    def _now(self) -> dt_time:
        """Return the current time-of-day.  PATCHABLE in tests."""
        return datetime.now().time()

    async def start(self) -> None:
        """Start the background scheduler loop.

        Creates an asyncio task that runs ``_check_loop``.
        """
        if not self.enabled:
            logger.info("Slot scheduler: disabled, not starting background loop")
            return

        logger.info(
            "Slot scheduler: starting background loop (interval=%ss, entries=%d)",
            self._check_interval,
            len(self._config.entries),
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
        """Periodic check loop that evaluates the schedule every 60 seconds."""
        while True:
            try:
                await asyncio.sleep(self._check_interval)
                await self._run_check_cycle()
            except asyncio.CancelledError:
                logger.info("Slot scheduler: check loop cancelled")
                return
            except Exception:
                logger.exception("Slot scheduler: unexpected error in check loop, continuing...")

    async def _run_check_cycle(self) -> None:
        """Perform a single schedule evaluation cycle.

        Called periodically by the check loop.  Examines the current time
        against the schedule and decides whether to:
        - Start draining (drain window opened for an upcoming transition).
        - Trigger a restart (transition time arrived, drain window ended).
        - Clear pending state (drain window ended without restart in simple
          cases where the static config already matches).

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

    async def _run_check_cycle_inner(self) -> None:
        """Inner implementation of _run_check_cycle (no exception wrapping).

        Separated to allow tests to call the inner logic directly without
        exception shielding.
        """
        if not self.enabled:
            return

        # Don't start a new drain/restart if a model switch is already in
        # progress (prevents overlapping restarts).
        try:
            if getattr(self._srv, 'model_switch_refcount', 0) > 0:
                return
        except Exception:
            pass

        now = self._now()

        # Determine the "current running" slot count for drain comparison:
        #   - Before any transition has been performed (pending_restart is
        #     None), the system is running the static config value.
        #   - After a transition has been detected (pending_restart is set),
        #     the "current" value comes from the schedule's wrapping.
        #   - If pending_restart just executed and cleared, we start fresh.
        schedule_current = self._config.get_active_slot(now)
        static_slots = self._get_static_slot_count()

        if self._pending_restart_slot is not None:
            # We're in a transition cycle — use schedule's wrapped value
            # for comparison, since at least one transition has been
            # performed or is in progress.
            current_slots = schedule_current
        else:
            # Before any transition — use static config as the baseline.
            # This ensures a single-entry schedule like [12:00→8] correctly
            # detects that a transition IS needed (static 4 != 8).
            current_slots = static_slots

        # ── Phase 1: Check if a pending restart should execute now ──────────
        if self._pending_restart_slot is not None:
            # Find the entry that matches our pending slot and is at or past now.
            for entry in self._config.entries:
                if (
                    entry.slots == self._pending_restart_slot
                    and entry.time <= now
                ):
                    # Transition time arrived (or passed) — execute restart.
                    logger.info(
                        "Slot scheduler: transition time reached for %d slots at %s",
                        entry.slots,
                        entry.time.strftime("%H:%M"),
                    )
                    self.set_draining(False)
                    await self.perform_restart()
                    return

            # If the pending slot matches the current active slot without
            # needing a restart (e.g., static config already matches), clear.
            if self._pending_restart_slot == current_slots:
                self.set_draining(False)
                self.clear_pending_restart()
                return

        # ── Phase 2: Check drain window for upcoming transitions ───────────
        drain_result = self._config.is_in_drain_window(
            now, current_slots=current_slots
        )
        if drain_result:
            _, next_slot_count = drain_result if isinstance(drain_result, tuple) else (True, None)
            if next_slot_count is not None:
                logger.info(
                    "Slot scheduler: drain window active for transition to %d slots "
                    "at %s (current=%s)",
                    next_slot_count,
                    self._next_entry_time_str(now),
                    current_slots,
                )
                self.set_draining(True)
                self.set_pending_restart(next_slot_count)
            return

        # ── Phase 3: If we were draining but the window passed, clear ──────
        if self._draining:
            self.set_draining(False)
            logger.info(
                "Slot scheduler: drain window ended, clearing drain flag"
            )

    def _next_entry_time_str(self, now: dt_time) -> str:
        """Return a human-readable string for the next entry time after *now*."""
        for entry in self._config.entries:
            if entry.time > now:
                return entry.time.strftime("%H:%M")
        return self._config.entries[0].time.strftime("%H:%M") + " (next day)" if self._config.entries else "(none)"

    async def perform_restart(self) -> bool:
        """Execute the pending restart of llama-server with the new slot count.

        Calls ``restart_services(slot_count=..., reason="scheduled_slot_change")``
        on the server module to perform the graceful drain and restart.

        Returns ``True`` if the restart was initiated, ``False`` if no pending
        restart slot was set or if an error occurred.
        """
        slot_count = self._pending_restart_slot
        if slot_count is None:
            return False

        try:
            logger.info(
                "Slot scheduler: performing restart with %d slots",
                slot_count,
            )
            result = await self._srv.restart_services(
                slot_count=slot_count,
                reason="scheduled_slot_change",
            )
            self.clear_pending_restart()
            return bool(result)
        except Exception:
            logger.exception(
                "Slot scheduler: restart failed for %d slots",
                slot_count,
            )
            self.clear_pending_restart()
            return False
