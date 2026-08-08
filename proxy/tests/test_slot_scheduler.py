"""
Tests for the slot scheduler module.

Covers schedule parsing, transition arming, and the restart-trigger
sequence.  Transitions apply immediately with no drain window
(LP-0MSF9RUSQ007M346) — requests are never rejected before a restart.
"""

import asyncio
import json
import logging
import subprocess
import time
from datetime import datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def sample_schedule():
    """A representative slot schedule with two transitions."""
    return {
        "enabled": True,
        "drain_minutes": 15,
        "entries": [
            {"time": "10:00", "slots": 4},
            {"time": "12:00", "slots": 8},
        ],
    }


@pytest.fixture
def disabled_schedule():
    """Schedule with enabled=False."""
    return {
        "enabled": False,
        "drain_minutes": 15,
        "entries": [{"time": "10:00", "slots": 4}],
    }


@pytest.fixture
def empty_schedule():
    """Schedule with no entries."""
    return {
        "enabled": True,
        "drain_minutes": 15,
        "entries": [],
    }


@pytest.fixture
def single_entry_schedule():
    """Schedule with one transition."""
    return {
        "enabled": True,
        "drain_minutes": 10,
        "entries": [{"time": "14:00", "slots": 6}],
    }


# ===================================================================
# Schedule parsing
# ===================================================================


class TestSlotScheduleConfig:
    """Tests for SlotScheduleConfig parsing and helper methods."""

    def test_parse_enabled_schedule(self, sample_schedule):
        """Verify a valid enabled schedule is parsed correctly."""
        from proxy.slot_scheduler import SlotScheduleConfig, SlotScheduleEntry

        cfg = SlotScheduleConfig(sample_schedule)
        assert cfg.enabled is True
        assert cfg.drain_minutes == 15
        assert len(cfg.entries) == 2
        assert cfg.entries[0].time == dt_time(10, 0)
        assert cfg.entries[0].slots == 4
        assert cfg.entries[1].time == dt_time(12, 0)
        assert cfg.entries[1].slots == 8

    def test_parse_disabled_schedule(self, disabled_schedule):
        """Verify disabled=False produces a disabled config."""
        from proxy.slot_scheduler import SlotScheduleConfig

        cfg = SlotScheduleConfig(disabled_schedule)
        assert cfg.enabled is False

    def test_parse_empty_entries(self, empty_schedule):
        """Verify empty entries list is handled."""
        from proxy.slot_scheduler import SlotScheduleConfig

        cfg = SlotScheduleConfig(empty_schedule)
        assert cfg.enabled is True
        assert cfg.drain_minutes == 15
        assert len(cfg.entries) == 0

    def test_parse_defaults_on_missing_keys(self):
        """Verify missing keys produce sensible defaults."""
        from proxy.slot_scheduler import SlotScheduleConfig

        cfg = SlotScheduleConfig({})
        assert cfg.enabled is False  # disabled by default
        assert cfg.drain_minutes == 3
        assert len(cfg.entries) == 0

    def test_parse_defaults_on_none(self):
        """Verify None input produces a disabled default config."""
        from proxy.slot_scheduler import SlotScheduleConfig

        cfg = SlotScheduleConfig(None)
        assert cfg.enabled is False
        assert cfg.drain_minutes == 3  # LP-0MS6OD1G90023F0A: reduced from 15 to 3
        assert len(cfg.entries) == 0

    def test_default_drain_minutes_is_3_minutes(self):
        """Verify the default drain window is 180 seconds (3 minutes).

        LP-0MS6OD1G90023F0A: reduced from 15 to 3 to speed up slot-count
        transition restarts while still allowing in-flight streams to finish.
        """
        from proxy.slot_scheduler import SlotScheduleConfig

        # No drain_minutes specified — should default to 3
        cfg = SlotScheduleConfig({"enabled": True, "entries": []})
        assert cfg.drain_minutes == 3, (
            f"Expected default drain_minutes=3, got {cfg.drain_minutes}"
        )

        # Full end-to-end: parse from server config without drain_minutes
        cfg2 = SlotScheduleConfig({
            "enabled": True,
            "entries": [{"time": "12:00", "slots": 8}]
        })
        assert cfg2.drain_minutes == 3

        # Explicit value of 15 must still work (backward compatibility)
        cfg3 = SlotScheduleConfig({
            "enabled": True,
            "drain_minutes": 15,
            "entries": [{"time": "12:00", "slots": 8}]
        })
        assert cfg3.drain_minutes == 15

    def test_from_server_config(self, sample_schedule):
        """Verify from_server_config extracts slot_schedule from server config."""
        from proxy.slot_scheduler import SlotScheduleConfig

        server_cfg = {"slot_schedule": sample_schedule}
        cfg = SlotScheduleConfig.from_server_config(server_cfg)
        assert cfg.enabled is True
        assert cfg.drain_minutes == 15  # explicit value in sample_schedule
        assert len(cfg.entries) == 2

    def test_from_server_config_no_schedule(self):
        """Verify from_server_config returns disabled config when no schedule."""
        from proxy.slot_scheduler import SlotScheduleConfig

        cfg = SlotScheduleConfig.from_server_config({})
        assert cfg.enabled is False

    def test_from_server_config_none(self):
        """Verify from_server_config handles None config gracefully."""
        from proxy.slot_scheduler import SlotScheduleConfig

        cfg = SlotScheduleConfig.from_server_config(None)
        assert cfg.enabled is False

    def test_edge_case_invalid_time_format(self):
        """Verify entries with invalid time format are skipped or raise."""
        from proxy.slot_scheduler import SlotScheduleConfig

        schedule = {
            "enabled": True,
            "drain_minutes": 3,
            "entries": [
                {"time": "not-a-time", "slots": 4},
                {"time": "25:00", "slots": 8},
            ],
        }
        cfg = SlotScheduleConfig(schedule)
        assert cfg.enabled is True
        # Invalid entries should be silently skipped
        assert len(cfg.entries) == 0

    def test_edge_case_missing_time_or_slots(self):
        """Verify entries missing 'time' or 'slots' are skipped."""
        from proxy.slot_scheduler import SlotScheduleConfig

        schedule = {
            "enabled": True,
            "drain_minutes": 15,
            "entries": [
                {"time": "10:00"},  # missing slots
                {"slots": 4},       # missing time
            ],
        }
        cfg = SlotScheduleConfig(schedule)
        assert len(cfg.entries) == 0


class TestSlotScheduleConfigActiveSlot:
    """Tests for get_active_slot() — what slot count applies now."""

    def _make_cfg(self, entries, drain_minutes=15):
        from proxy.slot_scheduler import SlotScheduleConfig

        return SlotScheduleConfig({
            "enabled": True,
            "drain_minutes": drain_minutes,
            "entries": [{"time": t, "slots": s} for t, s in entries],
        })

    def test_before_first_entry_wraps_to_last(self):
        """Before the first entry, the schedule wraps to the last entry."""
        # First entry is at 10:00, current time is 08:00
        # The schedule wraps: last entry (12:00→8) applies from midnight to 10:00
        cfg = self._make_cfg([("10:00", 4), ("12:00", 8)])
        result = cfg.get_active_slot(dt_time(8, 0))
        assert result == 8  # wraps to last entry

    def test_after_first_entry_returns_slot_count(self):
        """After first entry time, the slot count for that entry applies."""
        cfg = self._make_cfg([("10:00", 4), ("12:00", 8)])
        result = cfg.get_active_slot(dt_time(10, 30))
        assert result == 4

    def test_wraps_around_midnight(self):
        """Schedule entries can wrap around midnight."""
        cfg = self._make_cfg([("22:00", 2), ("06:00", 8)])
        # Entries sorted: [06:00→8, 22:00→2]
        # At 01:00, before 06:00 but after 22:00 on prev day → 2 slots (wrap)
        result = cfg.get_active_slot(dt_time(1, 0))
        assert result == 2
        # At 10:00, after 06:00, before 22:00 → 8 slots
        result = cfg.get_active_slot(dt_time(10, 0))
        assert result == 8

    def test_midnight_boundary(self):
        """At exact midnight, the last entry from previous day applies (wrap)."""
        cfg = self._make_cfg([("10:00", 4), ("20:00", 8)])
        result = cfg.get_active_slot(dt_time(0, 0))
        # Before 10:00 — wrap to last entry (20:00→8)
        assert result == 8

    def test_disabled_returns_none(self):
        """Disabled schedule always returns None."""
        from proxy.slot_scheduler import SlotScheduleConfig

        cfg = SlotScheduleConfig({
            "enabled": False,
            "drain_minutes": 15,
            "entries": [{"time": "10:00", "slots": 4}],
        })
        result = cfg.get_active_slot(dt_time(12, 0))
        assert result is None

    def test_exact_entry_time(self):
        """At the exact entry time, that entry's slot count should apply."""
        cfg = self._make_cfg([("14:00", 6)])
        result = cfg.get_active_slot(dt_time(14, 0))
        assert result == 6

    def test_single_entry_no_wrap(self):
        """Single entry wraps — before entry returns the entry's value."""
        cfg = self._make_cfg([("14:00", 6)])
        result_before = cfg.get_active_slot(dt_time(13, 59))
        assert result_before == 6  # wraps (schedule says 6 at 14:00, wraps for prev period)
        result_after = cfg.get_active_slot(dt_time(14, 0))
        assert result_after == 6

    def test_multiple_entries_non_wrapping(self):
        """Multiple entries with circular wrapping."""
        cfg = self._make_cfg([("08:00", 2), ("12:00", 4), ("18:00", 8)])
        assert cfg.get_active_slot(dt_time(7, 59)) == 8  # wraps from last entry (18:00→8)
        assert cfg.get_active_slot(dt_time(8, 0)) == 2
        assert cfg.get_active_slot(dt_time(12, 0)) == 4
        assert cfg.get_active_slot(dt_time(18, 0)) == 8
        assert cfg.get_active_slot(dt_time(23, 59)) == 8

    def test_empty_entries_returns_none(self, empty_schedule):
        """Empty entries always returns None."""
        from proxy.slot_scheduler import SlotScheduleConfig

        cfg = SlotScheduleConfig(empty_schedule)
        assert cfg.get_active_slot(dt_time(12, 0)) is None


class TestSlotScheduler:
    """Tests for the SlotScheduler background task."""

    @pytest.mark.asyncio
    async def test_init_disabled_by_default(self):
        """Verify scheduler starts disabled when no schedule is configured."""
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        mock_srv.config = {"server": {}}
        scheduler = SlotScheduler(mock_srv)
        assert scheduler.enabled is False

    @pytest.mark.asyncio
    async def test_init_enabled_with_schedule(self, sample_schedule):
        """Verify scheduler is enabled when schedule is provided."""
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        mock_srv.config = {"server": {"slot_schedule": sample_schedule}}
        scheduler = SlotScheduler(mock_srv)
        assert scheduler.enabled is True

    @pytest.mark.asyncio
    async def test_calculate_sleep_no_pending(self):
        """Verify calculate_sleep_seconds returns reasonable values."""
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        schedule = {
            "enabled": True,
            "drain_minutes": 15,
            "entries": [{"time": "12:00", "slots": 8}],
        }
        mock_srv.config = {"server": {"slot_schedule": schedule}}
        scheduler = SlotScheduler(mock_srv)
        # No pending restart
        assert scheduler.pending_restart_slot is None
        sleep_s = scheduler._calculate_sleep_seconds()
        assert sleep_s >= 1.0
        assert sleep_s <= 86400.0

    @pytest.mark.asyncio
    async def test_calculate_sleep_with_pending_restart(self):
        """Verify calculate_sleep_seconds targets the matching entry."""
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        now_dt = datetime(2026, 7, 23, 11, 0, 0)  # 11:00 today
        schedule = {
            "enabled": True,
            "drain_minutes": 15,
            "entries": [{"time": "12:00", "slots": 8}],
        }
        mock_srv.config = {"server": {"slot_schedule": schedule}}
        scheduler = SlotScheduler(mock_srv)
        scheduler.set_pending_restart(8)
        with patch.object(scheduler, '_now_dt', return_value=now_dt):
            with patch.object(scheduler, '_now', return_value=now_dt.time()):
                sleep_s = scheduler._calculate_sleep_seconds()
                # Should sleep until 12:00 (3600 seconds from 11:00)
                assert 3599 <= sleep_s <= 3601, f"Expected ~3600, got {sleep_s}"

    @pytest.mark.asyncio
    async def test_calculate_sleep_disabled(self, disabled_schedule):
        """Verify disabled schedule returns max sleep."""
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        mock_srv.config = {"server": {"slot_schedule": disabled_schedule}}
        scheduler = SlotScheduler(mock_srv)
        assert scheduler.enabled is False
        sleep_s = scheduler._calculate_sleep_seconds()
        assert sleep_s == scheduler._MAX_SLEEP_SECONDS

    @pytest.mark.asyncio
    async def test_pending_restart_slot(self, sample_schedule):
        """Verify pending_restart_slot is set and cleared."""
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        mock_srv.config = {"server": {"slot_schedule": sample_schedule}}
        scheduler = SlotScheduler(mock_srv)
        assert scheduler.pending_restart_slot is None
        scheduler.set_pending_restart(6)
        assert scheduler.pending_restart_slot == 6
        scheduler.clear_pending_restart()
        assert scheduler.pending_restart_slot is None

    @pytest.mark.asyncio
    async def test_perform_restart_calls_lifecycle(self, sample_schedule):
        """Verify perform_restart calls the lifecycle restart function."""
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        mock_srv.config = {"server": {"slot_schedule": sample_schedule}}
        mock_srv.restart_services = AsyncMock(return_value=True)
        scheduler = SlotScheduler(mock_srv)
        scheduler.set_pending_restart(8)
        result = await scheduler.perform_restart()
        assert result is True
        mock_srv.restart_services.assert_called_once_with(
            slot_count=8, reason="scheduled_slot_change"
        )

    @pytest.mark.asyncio
    async def test_perform_restart_no_slot_set(self, sample_schedule):
        """Verify perform_restart is a no-op when no pending slot."""
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        mock_srv.config = {"server": {"slot_schedule": sample_schedule}}
        scheduler = SlotScheduler(mock_srv)
        result = await scheduler.perform_restart()
        assert result is False  # nothing to do

    @pytest.mark.asyncio
    async def test_run_cycle_no_transition(self, sample_schedule):
        """Verify a check cycle with no pending transition does nothing."""
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        mock_srv.config = {"server": {"slot_schedule": sample_schedule}}
        scheduler = SlotScheduler(mock_srv)
        # Patch current_time to give a time far from any transition.
        # With schedule [10:00→4, 12:00→8], at 03:00 the wrapped value is
        # 8 (last entry), and the next transition at 10:00 is also 8→4
        # but current_slots=static (default 1).  So check with that context.
        with patch.object(scheduler, '_now', return_value=dt_time(3, 0)):
            await scheduler._run_check_cycle()
            # At 03:00, no transition is armed and no restart is pending
            assert scheduler.pending_restart_slot is None

    @pytest.mark.asyncio
    async def test_run_cycle_arms_transition_without_rejection_state(self):
        """Verify a check cycle arms the next transition WITHOUT any drain state.

        LP-0MSF9RUSQ007M346: at a time that would previously have been inside
        the drain window (11:55 for a 12:00 transition), the scheduler arms
        the restart but never marks the proxy as draining — new requests are
        not rejected.
        """
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        # Two entries: 10:00→4 and 12:00→8. At 11:55, the active slot from
        # the schedule is 4 (second entry hasn't arrived yet). We set static
        # slots to 4 so catch-up doesn't fire; the 12:00→8 transition (slot
        # count changes) is armed.
        schedule = {
            "enabled": True,
            "drain_minutes": 15,
            "entries": [
                {"time": "10:00", "slots": 4},
                {"time": "12:00", "slots": 8},
            ],
        }
        mock_srv.config = {
            "server": {
                "session_slot_pool_size": 4,
                "slot_schedule": schedule,
            }
        }
        mock_srv.restart_services = AsyncMock(return_value=True)
        scheduler = SlotScheduler(mock_srv)
        # At 11:55 — inside the old 15-min drain window (11:45–12:00).
        with patch.object(scheduler, '_now', return_value=dt_time(11, 55)):
            await scheduler._run_check_cycle()
            # The 12:00 restart is armed…
            assert scheduler.pending_restart_slot == 8
            # …but the proxy is never placed in a rejecting drain state.
            assert getattr(mock_srv, 'draining', None) is not True

    @pytest.mark.asyncio
    async def test_run_cycle_restarts_at_transition_time(self):
        """Verify the restart happens when the transition time is reached."""
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        schedule = {
            "enabled": True,
            "drain_minutes": 15,
            "entries": [{"time": "12:00", "slots": 8}],
        }
        mock_srv.config = {"server": {"slot_schedule": schedule}}
        mock_srv.restart_services = AsyncMock(return_value=True)
        scheduler = SlotScheduler(mock_srv)
        scheduler.set_pending_restart(8)
        # At 12:00 exactly, transition time
        now_dt = datetime(2026, 7, 23, 12, 0, 0)
        with patch.object(scheduler, '_now_dt', return_value=now_dt):
            with patch.object(scheduler, '_now', return_value=now_dt.time()):
                await scheduler._run_check_cycle()
                # Restart should have happened
                mock_srv.restart_services.assert_called_once_with(
                    slot_count=8, reason="scheduled_slot_change"
                )

    @pytest.mark.asyncio
    async def test_run_cycle_disabled_schedule_noop(self, disabled_schedule):
        """Verify disabled schedule never triggers any action."""
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        mock_srv.config = {"server": {"slot_schedule": disabled_schedule}}
        scheduler = SlotScheduler(mock_srv)
        # Disabled schedule → enabled=False → _run_check_cycle returns early
        assert scheduler.enabled is False
        with patch.object(scheduler, '_now', return_value=dt_time(9, 45)):
            await scheduler._run_check_cycle()
            assert scheduler.pending_restart_slot is None

    @pytest.mark.asyncio
    async def test_run_cycle_returns_early_when_already_restarting(self):
        """Verify a cycle does nothing when llama_process is already being restarted."""
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        schedule = {
            "enabled": True,
            "drain_minutes": 15,
            "entries": [{"time": "12:00", "slots": 8}],
        }
        mock_srv.config = {"server": {"slot_schedule": schedule}}
        # Simulate model_switch in progress
        mock_srv.model_switch_refcount = 1
        scheduler = SlotScheduler(mock_srv)
        with patch.object(scheduler, '_now', return_value=dt_time(11, 55)):
            await scheduler._run_check_cycle()
            # Should NOT arm a restart because a switch is in progress
            assert scheduler.pending_restart_slot is None

    @pytest.mark.asyncio
    async def test_run_cycle_same_slot_skips_transition(self):
        """Verify no restart is armed when the slot count doesn't change."""
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        schedule = {
            "enabled": True,
            "drain_minutes": 15,
            "entries": [{"time": "12:00", "slots": 4}],
        }
        mock_srv.config = {"server": {
            "slot_schedule": schedule,
            "session_slot_pool_size": 4,
        }}
        # Static config is 4, next transition at 12:00 is also 4.
        # Since 4 == 4, no drain is needed.
        scheduler = SlotScheduler(mock_srv)
        with patch.object(scheduler, '_now', return_value=dt_time(11, 55)):
            await scheduler._run_check_cycle()
            # No restart armed when slot count is same (wrapping matches)
            assert scheduler.pending_restart_slot is None


# ===================================================================
# Edge cases and error handling
# ===================================================================


class TestSlotSchedulerEdgeCases:

    @pytest.mark.asyncio
    async def test_midnight_transition(self):
        """Verify a midnight transition is armed without draining and restarts."""
        from proxy.slot_scheduler import SlotScheduleConfig, SlotScheduler

        mock_srv = MagicMock()
        schedule = {
            "enabled": True,
            "drain_minutes": 15,
            "entries": [
                {"time": "23:50", "slots": 2},
                {"time": "00:00", "slots": 8},
            ],
        }
        # Set static slot count to match the wrapped schedule value (8, from the
        # last entry at 00:00) so the catch-up logic does not fire at 23:45.
        mock_srv.config = {
            "server": {
                "session_slot_pool_size": 8,
                "slot_schedule": schedule,
            }
        }
        mock_srv.restart_services = AsyncMock(return_value=True)
        scheduler = SlotScheduler(mock_srv)

        # At 23:45 — inside the old drain window for the 23:50 transition.
        now_dt_2345 = datetime(2026, 7, 23, 23, 45, 0)
        with patch.object(scheduler, '_now_dt', return_value=now_dt_2345):
            with patch.object(scheduler, '_now', return_value=now_dt_2345.time()):
                await scheduler._run_check_cycle()
                # Transition is armed but the proxy is never rejecting.
                assert scheduler.pending_restart_slot == 2
                assert getattr(mock_srv, 'draining', None) is not True

        # At 23:50, the restart fires at the transition time.
        now_dt_2350 = datetime(2026, 7, 23, 23, 50, 0)
        with patch.object(scheduler, '_now_dt', return_value=now_dt_2350):
            with patch.object(scheduler, '_now', return_value=now_dt_2350.time()):
                await scheduler._run_check_cycle()
                mock_srv.restart_services.assert_called_with(
                    slot_count=2, reason="scheduled_slot_change"
                )

    @pytest.mark.asyncio
    async def test_restart_exception_handled_gracefully(self, sample_schedule):
        """Verify an exception during restart doesn't crash the scheduler."""
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        mock_srv.config = {"server": {"slot_schedule": sample_schedule}}
        mock_srv.restart_services = AsyncMock(side_effect=RuntimeError("boom"))
        scheduler = SlotScheduler(mock_srv)
        scheduler.set_pending_restart(8)
        # Should not raise
        result = await scheduler.perform_restart()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_cycle_with_exception(self, sample_schedule):
        """Verify an exception in _run_check_cycle is caught and logged."""
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        mock_srv.config = {"server": {"slot_schedule": sample_schedule}}
        scheduler = SlotScheduler(mock_srv)
        # Patch get_active_slot on _config to raise
        with patch.object(
            scheduler._config, 'get_active_slot', side_effect=ValueError("bad time")
        ):
            # Should not raise
            await scheduler._run_check_cycle()


# ===================================================================
# Transition safety: microsecond tolerance & missed-transition fallback
# ===================================================================


class TestSlotSchedulerTransitionSafety:
    """Tests for transition safety: restart microsecond tolerance,
    missed-transition fallback, and the absence of a rejection window
    (LP-0MSF9RUSQ007M346)."""

    # ── Microsecond tolerance ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_phase1_fires_with_microsecond_tolerance_before(self):
        """Phase 1 fires when now is just under the entry time (within 1s tolerance)."""
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        schedule = {
            "enabled": True,
            "drain_minutes": 15,
            "entries": [{"time": "23:59", "slots": 8}],
        }
        mock_srv.config = {"server": {"slot_schedule": schedule}}
        mock_srv.restart_services = AsyncMock(return_value=True)
        scheduler = SlotScheduler(mock_srv)

        # Set pending restart for 23:59→8 transition
        scheduler.set_pending_restart(8)

        # now is 23:58:59.999999 — microseconds before the entry time
        # With the tolerance, entry.time (23:59:00) <= now + 1s should pass
        now_dt = datetime(2026, 7, 23, 23, 58, 59, 999999)
        with patch.object(scheduler, '_now_dt', return_value=now_dt):
            with patch.object(scheduler, '_now', return_value=now_dt.time()):
                await scheduler._run_check_cycle_inner()

        # Restart should have fired (tolerance allowed the match)
        mock_srv.restart_services.assert_called_once_with(
            slot_count=8, reason="scheduled_slot_change"
        )

    @pytest.mark.asyncio
    async def test_phase1_fires_with_microsecond_tolerance_after(self):
        """Phase 1 fires when now is just past entry time (well within 1s tolerance)."""
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        schedule = {
            "enabled": True,
            "drain_minutes": 15,
            "entries": [{"time": "23:59", "slots": 8}],
        }
        mock_srv.config = {"server": {"slot_schedule": schedule}}
        mock_srv.restart_services = AsyncMock(return_value=True)
        scheduler = SlotScheduler(mock_srv)

        # Set pending restart for 23:59→8 transition
        scheduler.set_pending_restart(8)

        # now is 23:59:00.000001 — 1 microsecond past entry time
        now_dt = datetime(2026, 7, 23, 23, 59, 0, 1)
        with patch.object(scheduler, '_now_dt', return_value=now_dt):
            with patch.object(scheduler, '_now', return_value=now_dt.time()):
                await scheduler._run_check_cycle_inner()

        # Restart should have fired (entry.time <= now, directly)
        mock_srv.restart_services.assert_called_once_with(
            slot_count=8, reason="scheduled_slot_change"
        )

    # ── Missed transition fallback ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_missed_transition_clears_pending(self):
        """When pending_restart_slot is set but its time has passed by >1s,
        the scheduler clears pending, falling through."""
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        schedule = {
            "enabled": True,
            "drain_minutes": 15,
            "entries": [
                {"time": "10:00", "slots": 4},
                {"time": "23:59", "slots": 8},
            ],
        }
        mock_srv.config = {"server": {"slot_schedule": schedule}}
        mock_srv.restart_services = AsyncMock(return_value=True)
        scheduler = SlotScheduler(mock_srv)

        # We had a pending restart for 23:59→8 but it's now 07:00 next day
        scheduler.set_pending_restart(8)

        with patch.object(scheduler, '_now', return_value=dt_time(7, 0)):
            await scheduler._run_check_cycle_inner()

        # Pending should be cleared
        assert scheduler.pending_restart_slot is None

    @pytest.mark.asyncio
    async def test_calculate_sleep_falls_through_on_missed_pending(self):
        """When pending_restart_slot's time has passed, _calculate_sleep_seconds
        falls through to normal schedule instead of sleeping 24h."""
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        schedule = {
            "enabled": True,
            "drain_minutes": 15,
            "entries": [
                {"time": "10:00", "slots": 4},
                {"time": "23:59", "slots": 8},
            ],
        }
        mock_srv.config = {
            "server": {
                "session_slot_pool_size": 4,
                "slot_schedule": schedule,
            }
        }
        scheduler = SlotScheduler(mock_srv)

        # Simulate missed transition: pending_restart=8 for 23:59→8
        # which has now passed (it's 07:00 next day).
        scheduler.set_pending_restart(8)

        # At 07:00, pending slot 8's entry at 23:59 has passed.
        # _calculate_sleep_seconds should detect this and fall through.
        now_dt = datetime(2026, 7, 23, 7, 0, 0)
        with patch.object(scheduler, '_now_dt', return_value=now_dt):
            with patch.object(scheduler, '_now', return_value=now_dt.time()):
                sleep_s = scheduler._calculate_sleep_seconds()

        # Should NOT sleep 24h (the bug behavior).
        # Sleep should be targeted to a normal schedule event.
        assert sleep_s < 86400.0, f"Expected <24h sleep, got {sleep_s}"

    @pytest.mark.asyncio
    async def test_missed_transition_logs_warning(self, caplog):
        """Verify a WARNING is logged when a missed transition is detected.

        Uses a cross-midnight scenario: pending_restart was set for 23:59→8
        from the previous day, now is 07:00 the next day. The entry at 23:59
        appears in the future on the same day, so the diff < -1.0 triggers
        the missed-transition warning.
        """
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        schedule = {
            "enabled": True,
            "drain_minutes": 15,
            "entries": [
                {"time": "10:00", "slots": 4},
                {"time": "23:59", "slots": 8},
            ],
        }
        mock_srv.config = {"server": {"slot_schedule": schedule}}
        mock_srv.restart_services = AsyncMock(return_value=True)
        scheduler = SlotScheduler(mock_srv)

        scheduler.set_pending_restart(8)
        now_dt = datetime(2026, 7, 24, 7, 0, 0)

        with caplog.at_level(logging.WARNING):
            with patch.object(scheduler, '_now_dt', return_value=now_dt):
                with patch.object(scheduler, '_now', return_value=now_dt.time()):
                    await scheduler._run_check_cycle_inner()

        # Should contain a warning about missed transition
        assert any(
            "missed transition window" in msg or "Missed transition" in msg
            for msg in caplog.messages
        ), f"Expected warning about missed transition, got: {caplog.messages}"

    # ── LP-0MSF9RUSQ007M346: no rejection window before transitions ──

    @pytest.mark.asyncio
    async def test_calculate_sleep_targets_transition_time_not_drain_start(self):
        """Sleep goes straight to the transition time — never to a drain start.

        With a 12:00 transition and drain_minutes=15, at 11:30 the old code
        woke at 11:45 (drain start, 900s away).  The new code sleeps until
        12:00 (1800s away): there is no drain window (LP-0MSF9RUSQ007M346).
        """
        from proxy.slot_scheduler import SlotScheduler

        mock_srv = MagicMock()
        schedule = {
            "enabled": True,
            "drain_minutes": 15,
            "entries": [{"time": "12:00", "slots": 8}],
        }
        mock_srv.config = {
            "server": {
                "session_slot_pool_size": 4,
                "slot_schedule": schedule,
            }
        }
        scheduler = SlotScheduler(mock_srv)
        now_dt = datetime(2026, 7, 23, 11, 30, 0)
        with patch.object(scheduler, '_now_dt', return_value=now_dt):
            with patch.object(scheduler, '_now', return_value=now_dt.time()):
                sleep_s = scheduler._calculate_sleep_seconds()
        # 30 min until the 12:00 transition (1800s), not 15 min to drain start.
        assert 1799 <= sleep_s <= 1801, f"Expected ~1800s to transition, got {sleep_s}"

    @pytest.mark.asyncio
    async def test_calculate_sleep_ignores_drain_minutes_value(self):
        """drain_minutes no longer affects when the scheduler wakes.

        Whether drain_minutes is 15 or 3, the scheduler sleeps until the
        transition time (12:00) — the drain value is parsed but ignored.
        """
        from proxy.slot_scheduler import SlotScheduler

        for drain_minutes in (3, 15):
            mock_srv = MagicMock()
            schedule = {
                "enabled": True,
                "drain_minutes": drain_minutes,
                "entries": [{"time": "12:00", "slots": 8}],
            }
            mock_srv.config = {
                "server": {
                    "session_slot_pool_size": 4,
                    "slot_schedule": schedule,
                }
            }
            scheduler = SlotScheduler(mock_srv)
            now_dt = datetime(2026, 7, 23, 11, 30, 0)
            with patch.object(scheduler, '_now_dt', return_value=now_dt):
                with patch.object(scheduler, '_now', return_value=now_dt.time()):
                    sleep_s = scheduler._calculate_sleep_seconds()
            assert 1799 <= sleep_s <= 1801, (
                f"drain_minutes={drain_minutes}: expected ~1800s to transition, "
                f"got {sleep_s}"
            )
