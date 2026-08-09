"""Operating-mode state (fast/cheap) for the LLM proxy server.

The proxy runs in one of two operator-selected operating modes:

- **fast** — cloud-backed: remote providers are eligible and the server
  behaves as before (current day settings; ``config-fast.yaml``).
- **cheap** — local-only: requests use only the local llama-server at no
  cost (1-slot pool; ``config-cheap.yaml``).

The active mode is persisted in a small runtime state file
(``proxy/.mode``); when absent the mode defaults to ``fast`` (current
behavior). ``scripts/start-proxy.sh`` reads the mode at startup and selects
the corresponding config file; ``load_config()`` (``proxy/proxy/utils.py``)
falls back to the mode-selected config when ``LLAMA_PROXY_CONFIG`` is unset.

Switching modes via ``POST /admin/set-mode`` persists the new mode and
triggers a full proxy restart (``scripts/start-proxy.sh --restart``) so the
new config profile takes effect. A mode-switch restart terminates in-flight
requests — clients retry (same semantics as slot-schedule transitions,
LP-0MSF9RUSQ007M346). This is accepted behavior, not a bug.

An automatic ``mode_schedule`` (default: cheap 01:00-10:00, fast
10:00-01:00, local server time) is enforced by a background scheduler — see
``ModeScheduleConfig`` and ``start_mode_scheduler`` (LP-0MSM5K4TX004MICX).

A mode switched via ``POST /admin/set-mode`` is a **manual override**: it is
respected until the next scheduled mode transition (``ModeScheduleConfig.next_change``)
instead of being reverted on the next scheduler tick. The override expiry is
persisted in a companion state file (``proxy/.mode.override-until``) so it
survives proxy restarts and reboots; once the next scheduled change passes,
the schedule reasserts control (LP-0MSMF25V9002AY1J).
"""

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Any

logger = logging.getLogger("llama-proxy")

MODE_FAST = "fast"
MODE_CHEAP = "cheap"
VALID_MODES = (MODE_FAST, MODE_CHEAP)

DEFAULT_MODE = MODE_FAST

# Mode -> config file name (relative to the proxy directory).
MODE_CONFIG_FILES = {
    MODE_FAST: "config-fast.yaml",
    MODE_CHEAP: "config-cheap.yaml",
}

# Delay (seconds) before the background restart spawns, so the API response
# flushes before the process is killed (LP-0MSLMYEEU002IBH6).
RESTART_DELAY_SECONDS = 1.5

# Built-in automatic mode schedule: cheap from 01:00 until 10:00, fast from
# 10:00 until 01:00 (LP-0MSM5K4TX004MICX). Used when the config has no
# ``mode_schedule`` section; entries are ``(HH:MM, mode)``.
MODE_SCHEDULE_DEFAULT_ENTRIES = [("01:00", MODE_CHEAP), ("10:00", MODE_FAST)]

# How often (seconds) the background mode-scheduler re-checks the clock. A
# short poll bounds both the transition latency at schedule boundaries and
# how long a manual override survives before the timer reverts it.
MODE_SCHEDULE_CHECK_INTERVAL_SECONDS = 30

# Sentinel override expiry used when a schedule has no future transitions
# (disabled or constant): the manual mode then persists until the next API
# call rather than being reverted by the scheduler (LP-0MSMF25V9002AY1J).
OVERRIDE_UNTIL_NEVER = datetime.max


@dataclass
class ModeScheduleEntry:
    """A single schedule entry mapping a time-of-day to an operating mode."""

    time: dt_time
    mode: str


class ModeScheduleConfig:
    """Parsed automatic mode-schedule configuration.

    Reads the ``mode_schedule`` section from the server config. An absent
    section (or absent ``entries``) falls back to the built-in schedule
    (cheap 01:00-10:00, fast 10:00-01:00) so the timer stays on unless
    explicitly disabled with ``enabled: false``. Invalid entries (bad time
    format or unknown mode) are skipped with a warning; if no valid entry
    remains, the built-in schedule is used.

    The active mode at any instant is the most recent entry whose time is
    at or before *now*; before the first entry of the day the schedule
    wraps circularly to the last entry (so ``10:00 -> fast`` also covers
    00:00-00:59, matching the slot_schedule semantics).
    """

    def __init__(self, raw: dict[str, Any] | None):
        if not raw or not isinstance(raw, dict):
            # Absent/empty section: enabled with the built-in schedule.
            self.enabled = True
            self.entries = self._parse_entries(None)
            return

        self.enabled = bool(raw.get("enabled", True))
        self.entries = self._parse_entries(raw.get("entries"))

    @classmethod
    def from_server_config(
        cls, server_config: dict[str, Any] | None
    ) -> "ModeScheduleConfig":
        """Extract the mode schedule from the server config dict."""
        if not server_config or not isinstance(server_config, dict):
            return cls(None)
        return cls(server_config.get("mode_schedule"))

    @staticmethod
    def _parse_entries(raw_entries: Any) -> list[ModeScheduleEntry]:
        """Parse ``[{time, mode}, ...]`` into sorted entries, skipping invalid."""
        entries: list[ModeScheduleEntry] = []
        if isinstance(raw_entries, list):
            for entry in raw_entries:
                if not isinstance(entry, dict):
                    continue
                parsed = _parse_schedule_entry(entry)
                if parsed is not None:
                    entries.append(parsed)
        if entries:
            entries.sort(key=lambda e: e.time)
            return entries
        # No valid entries (absent section or all invalid): use the built-in
        # schedule so the timer never silently turns off.
        logger.warning(
            "mode_schedule: no valid entries, using built-in default schedule"
        )
        return [
            ModeScheduleEntry(
                time=_parse_hhmm(time_str), mode=mode
            )
            for time_str, mode in MODE_SCHEDULE_DEFAULT_ENTRIES
        ]

    def active_mode(self, now: dt_time | None = None) -> str | None:
        """Return the mode mandated by the schedule at *now* (or None if disabled).

        None is returned only when the schedule is disabled
        (``enabled: false``).
        """
        if not self.enabled or not self.entries:
            return None
        now = now or datetime.now().time()
        last_matching: ModeScheduleEntry | None = None
        for entry in self.entries:
            if entry.time <= now:
                last_matching = entry
            else:
                break
        if last_matching is not None:
            return last_matching.mode
        # Before the first entry of the day — wrap circularly to the last
        # entry (the previous period persists until the first transition).
        return self.entries[-1].mode

    def next_change(self, now: datetime | None = None) -> datetime | None:
        """Return the next datetime at which the scheduled mode changes, or None.

        The next change is the earliest schedule boundary strictly after
        *now* whose mode differs from the mode in effect just before the
        boundary (consecutive same-mode entries are not changes). When no
        change remains today the search wraps to tomorrow's first change.

        Returns None when the schedule is disabled, has no entries, or is
        constant (no boundary ever changes the mode) — in those cases a
        manual override never expires on its own (LP-0MSMF25V9002AY1J).
        """
        if not self.enabled or not self.entries:
            return None
        now = now or datetime.now()
        entries = self.entries
        # Boundaries where the mandated mode actually changes. The segment
        # before entries[0] is the last entry of the previous day (wrap),
        # so entries[-1] is the correct predecessor for index 0.
        change_indices = [
            i for i, entry in enumerate(entries) if entry.mode != entries[i - 1].mode
        ]
        if not change_indices:
            return None  # constant schedule — no changes ever
        for i in change_indices:
            if entries[i].time > now.time():
                return datetime.combine(now.date(), entries[i].time)
        # No change remains today: the next one is tomorrow at the first
        # change boundary.
        return datetime.combine(
            now.date() + timedelta(days=1), entries[change_indices[0]].time
        )


def _parse_hhmm(time_str: str) -> dt_time | None:
    """Parse an ``HH:MM`` string into a time, or None when invalid."""
    parts = str(time_str).strip().split(":")
    if len(parts) != 2:
        return None
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return dt_time(hour, minute)


def _parse_schedule_entry(entry: dict[str, Any]) -> ModeScheduleEntry | None:
    """Parse a single ``{time, mode}`` entry, or None when invalid."""
    parsed_time = _parse_hhmm(entry.get("time") or "")
    mode = str(entry.get("mode") or "").strip().lower()
    if parsed_time is None or mode not in VALID_MODES:
        logger.warning("mode_schedule: ignoring invalid entry %r", entry)
        return None
    return ModeScheduleEntry(time=parsed_time, mode=mode)


def expected_mode_for_time(
    now: dt_time | None = None,
    schedule: ModeScheduleConfig | None = None,
) -> str | None:
    """Return the mode the schedule mandates at *now* (None when disabled)."""
    schedule = schedule or ModeScheduleConfig(None)
    return schedule.active_mode(now)


def _mode_scheduler_step(
    schedule: ModeScheduleConfig, now: dt_time | None = None
) -> bool:
    """One scheduler check: apply the scheduled mode when it diverges.

    A manual API override (persisted override-until expiry that has not yet
    passed) is respected: the scheduled mode is NOT applied before the next
    scheduled transition. Once the override expires the scheduled mode is
    applied regardless of the current setting. Returns True when a mode
    change was applied. A pending mode-switch restart (e.g. a manual switch
    in flight) is left alone and retried on the next cycle.
    """
    expected = schedule.active_mode(now)
    if expected is None:
        return False
    if read_mode() == expected:
        return False
    if manual_override_active():
        # A manual API override is in effect until the next scheduled
        # change; stand down instead of reverting it.
        return False
    try:
        set_mode(expected)
    except RuntimeError:
        logger.debug("Mode scheduler: restart pending, retrying next cycle")
        return False
    logger.info("Mode scheduler: applied scheduled mode %s", expected)
    return True


def _mode_scheduler_loop(
    schedule: ModeScheduleConfig, interval: float
) -> None:
    """Background loop: enforce the schedule, checking immediately and then
    every *interval* seconds. Runs as a daemon thread so it dies with the
    proxy process (a mode-switch restart replaces the whole process anyway).
    """
    while True:
        try:
            _mode_scheduler_step(schedule)
        except Exception:
            logger.exception("Mode scheduler: unexpected error in check cycle")
        time.sleep(interval)


def start_mode_scheduler(
    schedule: ModeScheduleConfig,
    interval: float = MODE_SCHEDULE_CHECK_INTERVAL_SECONDS,
) -> threading.Thread:
    """Start the background mode-scheduler thread and return it."""
    thread = threading.Thread(
        target=_mode_scheduler_loop,
        args=(schedule, interval),
        daemon=True,
        name="mode-scheduler",
    )
    thread.start()
    return thread


# Serializes set-mode calls and guards the pending-restart flag so a second
# switch cannot arm a second restart while one is already in flight
# (avoids restart loops).
_mode_lock = threading.Lock()
_restart_pending = False


def proxy_dir() -> Path:
    """Return the proxy directory (parent of the ``proxy`` package)."""
    return Path(__file__).parent.parent


def mode_state_file() -> Path:
    """Path to the persisted mode state file (``proxy/.mode``)."""
    return proxy_dir() / ".mode"


def override_until_file() -> Path:
    """Path to the manual-override expiry state file (``proxy/.mode.override-until``).

    Holds an ISO-format naive local datetime marking when the manual override
    expires (the next scheduled mode transition). Absent file = no override.
    """
    return proxy_dir() / ".mode.override-until"


def read_override_until() -> datetime | None:
    """Return the persisted manual-override expiry, or None when absent/invalid.

    The expiry is a naive local datetime matching the schedule's local-time
    semantics. A missing, empty, or unparsable state file yields None (no
    override). An expired value is returned as-is — callers decide whether
    the override is still active via ``manual_override_active``.
    """
    try:
        text = override_until_file().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        logger.warning(
            "Ignoring unparsable override-until state %r, treating as no override",
            text,
        )
        return None


def write_override_until(expiry: datetime | None) -> None:
    """Persist (or clear) the manual-override expiry state file.

    ``None`` removes the file (no override in effect).
    """
    path = override_until_file()
    if expiry is None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Failed to remove override-until state file")
        return
    path.write_text(expiry.isoformat() + "\n", encoding="utf-8")


def manual_override_active(now: datetime | None = None) -> bool:
    """Whether a manual mode override is currently in effect.

    True when an override expiry is persisted and has not yet passed. An
    absent/expired/invalid expiry yields False (the schedule applies).
    """
    expiry = read_override_until()
    if expiry is None:
        return False
    return (now or datetime.now()) < expiry


def read_mode() -> str:
    """Return the persisted operating mode, defaulting to ``fast``.

    A missing, empty, or invalid state file yields ``fast`` (the current
    behavior when no mode has ever been persisted).
    """
    try:
        text = mode_state_file().read_text(encoding="utf-8").strip().lower()
    except FileNotFoundError:
        return DEFAULT_MODE
    except OSError:
        logger.warning("Failed to read mode state file, defaulting to %s", DEFAULT_MODE)
        return DEFAULT_MODE
    return text if text in VALID_MODES else DEFAULT_MODE


def write_mode(mode: str) -> None:
    """Persist the operating mode to the state file.

    Raises ``ValueError`` for anything other than ``fast`` or ``cheap``.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"invalid mode: {mode!r}")
    mode_state_file().write_text(mode.strip().lower() + "\n", encoding="utf-8")


def mode_config_file(mode: str) -> Path:
    """Return the config file path for *mode*.

    ``fast`` → ``config-fast.yaml``, ``cheap`` → ``config-cheap.yaml``.
    Falls back to ``proxy/config.yaml`` when the mode-specific file is
    missing (or the mode is invalid), keeping config.yaml as the
    default/fallback profile.
    """
    name = MODE_CONFIG_FILES.get(mode)
    path = proxy_dir() / name if name else proxy_dir() / "config.yaml"
    return path if path.is_file() else proxy_dir() / "config.yaml"


def resolve_config_path() -> Path:
    """Resolve the active config path.

    Precedence:
    1. ``LLAMA_PROXY_CONFIG`` env var (explicit override — set by
       ``scripts/start-proxy.sh`` from the persisted mode).
    2. The mode-selected config file (``config-fast.yaml`` /
       ``config-cheap.yaml``) when a valid mode has been persisted.
    3. ``proxy/config.yaml`` (default/fallback, current behavior).
    """
    env = os.environ.get("LLAMA_PROXY_CONFIG")
    if env:
        return Path(env)
    return mode_config_file(read_mode())


def restart_pending() -> bool:
    """Whether a mode-switch restart is pending (thread-safe)."""
    with _mode_lock:
        return _restart_pending


def set_mode(
    mode: str,
    manual: bool = False,
    schedule: ModeScheduleConfig | None = None,
) -> tuple[str, bool]:
    """Persist *mode* and arm a background restart when it changes.

    *manual* marks the call as an explicit API/operator override: the
    override expiry is (re)computed from *schedule* and persisted, so the
    background scheduler respects the chosen mode until the next scheduled
    time change instead of reverting it on the next tick
    (LP-0MSMF25V9002AY1J). A non-manual call (the scheduler enforcing the
    schedule) persists the mode and clears any pending override.

    Returns ``(persisted_mode, restart_triggered)``:

    - Requesting the mode that is already active is a **noop**: nothing is
      persisted and no restart is armed (a manual call still refreshes the
      override expiry).
    - Requesting a different mode persists the new mode and spawns the
      restart (``scripts/start-proxy.sh --restart``) in the background.

    Raises ``RuntimeError`` when a mode-switch restart is already pending
    and the requested mode differs (rejected to avoid restart loops).
    """
    global _restart_pending
    with _mode_lock:
        if _restart_pending:
            if read_mode() == mode:
                if manual:
                    _write_override_expiry(schedule)
                return mode, False
            raise RuntimeError("A mode-switch restart is already in progress")
        if read_mode() == mode:
            if manual:
                _write_override_expiry(schedule)
            return mode, False
        write_mode(mode)
        if manual:
            _write_override_expiry(schedule)
        else:
            write_override_until(None)
        _restart_pending = True
    _spawn_restart()
    return mode, True


def _write_override_expiry(schedule: ModeScheduleConfig | None) -> None:
    """Persist the manual-override expiry derived from *schedule*.

    A disabled or constant schedule (``next_change`` returns None) yields
    the ``OVERRIDE_UNTIL_NEVER`` sentinel so the manual mode persists until
    the next API call. With no schedule available, no override is recorded
    (fail-safe: the schedule applies).
    """
    if schedule is None:
        write_override_until(None)
        return
    next_change = schedule.next_change()
    write_override_until(
        next_change if next_change is not None else OVERRIDE_UNTIL_NEVER
    )


def _spawn_restart() -> None:
    """Spawn ``scripts/start-proxy.sh --restart`` in the background.

    Runs in a daemon thread after ``RESTART_DELAY_SECONDS`` so the API
    response flushes before the process is killed. The persisted mode is
    already written, so a failed restart still applies on the next manual
    start.
    """

    def _run() -> None:
        try:
            time.sleep(RESTART_DELAY_SECONDS)
            script = proxy_dir() / "scripts" / "start-proxy.sh"
            subprocess.Popen(
                ["bash", str(script), "--restart"],
                cwd=str(proxy_dir()),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            logger.info("Mode-switch restart spawned: %s --restart", script)
        except Exception:
            logger.exception("Failed to spawn mode-switch restart")

    threading.Thread(target=_run, daemon=True).start()
