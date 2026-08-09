"""Tests for the automatic fast/cheap mode schedule (LP-0MSM5K4TX004MICX).

Covers the schedule semantics (cheap 01:00-10:00, fast 10:00-01:00 with
midnight wrap), config parsing (disabled / custom / invalid entries /
built-in fallback), and the background enforcement step (applies the
scheduled mode when the persisted mode diverges, defers while a restart is
pending).
"""

from datetime import time as dt_time

import pytest
from proxy.mode import ModeScheduleConfig, _mode_scheduler_step

from proxy import mode as mode_module

T = dt_time


def builtin():
    """The default (absent-section) schedule."""
    return ModeScheduleConfig(None)


# ---------------------------------------------------------------------------
# Schedule semantics
# ---------------------------------------------------------------------------


class TestDefaultScheduleSemantics:
    def test_cheap_period(self):
        schedule = builtin()
        assert schedule.active_mode(T(1, 0)) == "cheap"
        assert schedule.active_mode(T(1, 1)) == "cheap"
        assert schedule.active_mode(T(9, 59)) == "cheap"

    def test_fast_period(self):
        schedule = builtin()
        assert schedule.active_mode(T(10, 0)) == "fast"
        assert schedule.active_mode(T(12, 30)) == "fast"
        assert schedule.active_mode(T(23, 59)) == "fast"
        assert schedule.active_mode(T(0, 30)) == "fast"  # fast until 01:00

    def test_midnight_wrap_covers_0000_to_0100(self):
        """10:00->fast wraps circularly over midnight (00:00-00:59)."""
        schedule = builtin()
        assert schedule.active_mode(T(0, 0, 0)) == "fast"
        assert schedule.active_mode(T(0, 0, 30)) == "fast"
        assert schedule.active_mode(T(0, 59, 59)) == "fast"

    def test_boundaries_exact(self):
        schedule = builtin()
        assert schedule.active_mode(T(0, 59)) == "fast"
        assert schedule.active_mode(T(1, 0)) == "cheap"
        assert schedule.active_mode(T(10, 0)) == "fast"

    def test_expected_mode_for_time_helper(self):
        assert mode_module.expected_mode_for_time(T(3, 0)) == "cheap"
        assert mode_module.expected_mode_for_time(T(15, 0)) == "fast"
        assert mode_module.expected_mode_for_time(T(0, 30)) == "fast"
        assert mode_module.expected_mode_for_time(T(0, 0, 59)) == "fast"


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestModeScheduleConfig:
    def test_absent_section_enabled_with_builtin(self):
        schedule = ModeScheduleConfig(None)
        assert schedule.enabled is True
        assert [(e.time, e.mode) for e in schedule.entries] == [
            (T(1, 0), "cheap"),
            (T(10, 0), "fast"),
        ]

    def test_disabled_returns_none(self):
        schedule = ModeScheduleConfig({"enabled": False, "entries": [
            {"time": "00:01", "mode": "cheap"},
        ]})
        assert schedule.enabled is False
        assert schedule.active_mode(T(3, 0)) is None

    def test_custom_entries_honored(self):
        schedule = ModeScheduleConfig({"enabled": True, "entries": [
            {"time": "06:00", "mode": "cheap"},
            {"time": "18:00", "mode": "fast"},
        ]})
        assert schedule.active_mode(T(5, 59)) == "fast"  # wrap to last
        assert schedule.active_mode(T(6, 0)) == "cheap"
        assert schedule.active_mode(T(17, 59)) == "cheap"
        assert schedule.active_mode(T(18, 0)) == "fast"

    def test_invalid_entries_skipped(self):
        schedule = ModeScheduleConfig({"enabled": True, "entries": [
            {"time": "not-a-time", "mode": "cheap"},
            {"time": "10:00", "mode": "bogus-mode"},
            {"time": "25:99", "mode": "fast"},
            {"time": "10:00", "mode": "fast"},  # the only valid one
        ]})
        assert [(e.time, e.mode) for e in schedule.entries] == [(T(10, 0), "fast")]

    def test_all_invalid_entries_fall_back_to_builtin(self):
        schedule = ModeScheduleConfig({"enabled": True, "entries": [
            {"time": "bogus", "mode": "fast"},
        ]})
        assert [(e.time, e.mode) for e in schedule.entries] == [
            (T(1, 0), "cheap"),
            (T(10, 0), "fast"),
        ]

    def test_from_server_config_reads_server_section(self):
        schedule = ModeScheduleConfig.from_server_config(
            {"mode_schedule": {"enabled": False, "entries": []}}
        )
        assert schedule.enabled is False

    def test_from_server_config_absent_section(self):
        schedule = ModeScheduleConfig.from_server_config({})
        assert schedule.enabled is True
        assert schedule.active_mode(T(3, 0)) == "cheap"

    def test_entries_sorted_by_time(self):
        schedule = ModeScheduleConfig({"enabled": True, "entries": [
            {"time": "10:00", "mode": "fast"},
            {"time": "01:00", "mode": "cheap"},
        ]})
        assert [(e.time, e.mode) for e in schedule.entries] == [
            (T(1, 0), "cheap"),
            (T(10, 0), "fast"),
        ]


# ---------------------------------------------------------------------------
# Enforcement step
# ---------------------------------------------------------------------------


class TestModeSchedulerStep:
    @pytest.fixture
    def schedule(self):
        return builtin()

    def test_applies_scheduled_mode_when_diverged(self, schedule, monkeypatch):
        """A manual override (cheap at 14:00) is reverted to the scheduled fast."""
        monkeypatch.setattr(mode_module, "read_mode", lambda: "cheap")
        applied = []
        monkeypatch.setattr(
            mode_module, "set_mode", lambda m: applied.append(m) or ("fast", True)
        )
        assert _mode_scheduler_step(schedule, now=T(14, 0)) is True
        assert applied == ["fast"]

    def test_noop_when_mode_matches(self, schedule, monkeypatch):
        monkeypatch.setattr(mode_module, "read_mode", lambda: "fast")
        applied = []
        monkeypatch.setattr(mode_module, "set_mode", lambda m: applied.append(m))
        assert _mode_scheduler_step(schedule, now=T(14, 0)) is False
        assert applied == []

    def test_defers_while_restart_pending(self, schedule, monkeypatch):
        """A pending restart (RuntimeError from set_mode) is retried later."""
        monkeypatch.setattr(mode_module, "read_mode", lambda: "cheap")

        def reject(_mode):
            raise RuntimeError("A mode-switch restart is already in progress")

        monkeypatch.setattr(mode_module, "set_mode", reject)
        # Must not raise; returns False so the loop retries next cycle.
        assert _mode_scheduler_step(schedule, now=T(14, 0)) is False

    def test_noop_when_schedule_disabled(self, monkeypatch):
        schedule = ModeScheduleConfig({"enabled": False, "entries": [
            {"time": "00:01", "mode": "cheap"},
        ]})
        monkeypatch.setattr(mode_module, "read_mode", lambda: "fast")
        applied = []
        monkeypatch.setattr(mode_module, "set_mode", lambda m: applied.append(m))
        assert _mode_scheduler_step(schedule, now=T(14, 0)) is False
        assert applied == []

    def test_loop_calls_step_immediately(self, schedule, monkeypatch):
        """The first check runs before the first sleep (startup applies the
        scheduled mode right away)."""
        import threading
        import time

        calls = []
        monkeypatch.setattr(
            mode_module,
            "_mode_scheduler_step",
            lambda s, now=None: calls.append(1) or False,
        )
        thread = mode_module.start_mode_scheduler(schedule, interval=3600)
        assert isinstance(thread, threading.Thread)
        assert thread.daemon is True
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not calls:
            time.sleep(0.01)
        assert calls == [1]
