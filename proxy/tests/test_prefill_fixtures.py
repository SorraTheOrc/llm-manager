"""
Unit tests for prefill-scenario fixtures (LP-0MW7V2DK400365ER / LP-0MW8VMJYL009K9UB AC1-AC6).

Verifies that each fixture produces correct, deterministic state that can be
used by downstream feature tests.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Fixture modules are on sys.path via conftest.py:
#   proxy/           (so ``import proxy.*`` works)
#   proxy/tests/     (so bare ``from test_* import ...`` works)
#   proxy/tests/fixtures/ (so fixtures import without package prefix)
from test_prefill_scenarios import (
    STDERR_MIXED,
    STDERR_WITH_PROGRESS,
    STDERR_WITHOUT_PROGRESS,
    ConcurrentSlotsState,
    ProgressingSlotState,
    SlotsTimeoutError,
    StalledSlotState,
    StderrFixture,
    concurrent_slots_fixture,
    make_test_srv,
    patch_llama_status,
    patch_query_slots_progress,
    progressing_task_fixture,
    slots_timeout_fixture,
    slots_timeout_fixture_async,
    slots_timeout_fixture_detail,
    stalled_task_fixture,
    stderr_mixed,
    stderr_with_progress,
    stderr_without_progress,
    stubbed_slots_payload_multi_slot,
    stubbed_slots_payload_timeout,
    stubbed_slots_payload_with_progress,
    stubbed_slots_payload_without_progress,
)

# ===================================================================
# AC1 — Two concurrent local tasks fixture
# ===================================================================


class TestConcurrentSlotsFixture:
    """AC1 — Two concurrent local tasks fixture."""

    def test_fixture_produces_two_slots(self):
        """Fixture yields exactly two slots."""
        with concurrent_slots_fixture() as state:
            slots = state.to_slots_list()
            assert len(slots) == 2
            assert slots[0]["id"] == 0
            assert slots[1]["id"] == 1

    def test_independent_progress_counters(self):
        """Each slot tracks progress independently."""
        with concurrent_slots_fixture(progress_a=1000, progress_b=500) as state:
            slots = state.to_slots_list()
            assert slots[0]["n_past"] == 1000
            assert slots[1]["n_past"] == 500

    def test_advance_mutates_both(self):
        """Advance() increments both slots' progress."""
        with concurrent_slots_fixture(progress_a=100, progress_b=50) as state:
            state.advance(delta_a=500, delta_b=200)
            assert state.progress_a == 600
            assert state.progress_b == 250
            slots = state.to_slots_list()
            assert slots[0]["n_past"] == 600
            assert slots[1]["n_past"] == 250

    def test_stall_slot_halts_one(self):
        """stall_slot(0) stops the first slot's progress."""
        with concurrent_slots_fixture() as state:
            state.advance(delta_a=100)  # A:1000+100=1100, B:500+50=550
            assert state.progress_a == 1100
            state.stall_slot(0)
            assert not state.processing_a
            assert state.progress_a == 1100  # hasn't changed
            state.advance(delta_a=100)  # only B advances (A stalled)
            assert state.progress_a == 1100  # still stalled
            assert state.progress_b == 600  # B:550+50=600

    def test_default_values(self):
        """Default fixture: both slots processing with standard progress."""
        with concurrent_slots_fixture() as state:
            assert state.progress_a == 1000
            assert state.progress_b == 500
            assert state.processing_a is True
            assert state.processing_b is True


# ===================================================================
# AC2 — Stalled task fixture
# ===================================================================


class TestStalledTaskFixture:
    """AC2 — Stalled task fixture."""

    def test_fixture_produces_single_slot(self):
        """Fixture yields exactly one slot."""
        with stalled_task_fixture() as state:
            slots = state.to_slots_list()
            assert len(slots) == 1
            assert slots[0]["n_past"] == 5000

    def test_elapsed_tracks_time(self):
        """Elapsed time increases from fixture start."""
        with stalled_task_fixture() as state:
            t0 = state.elapsed()
            time.sleep(0.05)
            t1 = state.elapsed()
            assert t1 > t0

    def test_is_stalled_returns_true_initially(self):
        """While within stall_duration, is_stalled() is True."""
        with stalled_task_fixture(stall_duration=30) as state:
            assert state.is_stalled() is True

    def test_stall_duration_configurable(self):
        """stall_duration controls how long the slot appears stalled."""
        with stalled_task_fixture(stall_duration=0.1) as state:
            assert state.is_stalled() is True
            time.sleep(0.15)
            assert state.is_stalled() is False

    def test_processing_flag_preserved(self):
        """The processing flag is preserved in slot output."""
        with stalled_task_fixture(processing=True) as state:
            slots = state.to_slots_list()
            assert slots[0]["is_processing"] is True

    def test_b8782_shape_no_context_tokens(self):
        """b8782 build shape: context_tokens absent."""
        with stalled_task_fixture() as state:
            slots = state.to_slots_list()
            assert slots[0].get("context_tokens") is None


# ===================================================================
# AC3 — Progressing task fixture
# ===================================================================


class TestProgressingTaskFixture:
    """AC3 — Progressing task fixture."""

    def test_fixture_produces_single_slot(self):
        """Fixture yields exactly one slot."""
        with progressing_task_fixture() as state:
            slots = state.to_slots_list()
            assert len(slots) == 1
            assert slots[0]["n_past"] == 0

    def test_tick_advances_progress(self):
        """Each tick() advances progress by advance_per_tick."""
        with progressing_task_fixture(advance_per_tick=200) as state:
            slots = state.tick()
            assert state.progress == 200
            assert slots[0]["n_past"] == 200

    def test_multiple_ticks_accumulate(self):
        """Progress accumulates across multiple ticks."""
        with progressing_task_fixture(progress=100, advance_per_tick=100) as state:
            state.tick()  # 200
            state.tick()  # 300
            state.tick()  # 400
            assert state.progress == 400
            assert state.total_progress == 400

    def test_tick_count_tracks_ticks(self):
        """tick_count increments each call to tick()."""
        with progressing_task_fixture() as state:
            assert state.tick_count == 0
            state.tick()
            assert state.tick_count == 1
            state.tick()
            assert state.tick_count == 2

    def test_configurable_initial_progress(self):
        """Fixture can start at a non-zero progress value."""
        with progressing_task_fixture(progress=5000, advance_per_tick=500) as state:
            assert state.progress == 5000
            state.tick()
            assert state.progress == 5500

    def test_processing_flag_in_output(self):
        """processing flag is preserved in slot output."""
        with progressing_task_fixture(processing=True) as state:
            slots = state.tick()
            assert slots[0]["is_processing"] is True


# ===================================================================
# AC4 — llama-server stderr fixture
# ===================================================================


class TestStderrFixture:
    """AC4 — llama-server stderr fixture (with and without progress lines)."""

    def test_with_progress_contains_progress_lines(self):
        """stderr_with_progress yields lines with progress markers."""
        with stderr_with_progress() as fixture:
            assert fixture.with_progress is True
            assert fixture.next_line() is not None
            assert "n_tokens" in fixture.lines[0]

    def test_without_progress_no_progress_markers(self):
        """stderr_without_progress yields lines without progress markers."""
        with stderr_without_progress() as fixture:
            assert fixture.with_progress is False
            assert "n_tokens" not in fixture.lines[0]

    def test_next_line_returns_in_order(self):
        """next_line() yields lines sequentially."""
        lines = ["line A", "line B", "line C"]
        with stderr_with_progress(lines=lines) as fixture:
            assert fixture.next_line() == "line A"
            assert fixture.next_line() == "line B"
            assert fixture.next_line() == "line C"
            assert fixture.next_line() is None

    def test_remaining_counts_lines(self):
        """remaining() returns the number of unread lines."""
        lines = ["A", "B"]
        with stderr_with_progress(lines=lines) as fixture:
            assert fixture.remaining() == 2
            fixture.next_line()
            assert fixture.remaining() == 1

    def test_reset_clears_index(self):
        """reset() allows replaying from the beginning."""
        lines = ["A", "B"]
        with stderr_with_progress(lines=lines) as fixture:
            fixture.next_line()
            fixture.next_line()
            assert fixture.remaining() == 0
            fixture.reset()
            assert fixture.remaining() == 2
            assert fixture.next_line() == "A"

    def test_mixed_stderr(self):
        """stderr_mixed contains both progress and non-progress lines."""
        with stderr_mixed() as fixture:
            lines = fixture.lines
            assert any("progress" in line for line in lines)
            assert len(lines) == 3


# ===================================================================
# AC5 — /slots timeout fixture
# ===================================================================


class TestSlotsTimeoutFixture:
    """AC5 — /slots timeout fixture."""

    def test_sync_fixture_raises_timeout(self):
        """slots_timeout_fixture patches /slots to raise SlotsTimeoutError."""
        with slots_timeout_fixture():
            with pytest.raises(SlotsTimeoutError):
                from proxy.observability import _query_slots
                asyncio.run(_query_slots(8080, timeout=0.01))

    @pytest.mark.asyncio
    async def test_async_fixture_raises_timeout(self):
        """slots_timeout_fixture_async patches /slots to raise."""
        async with slots_timeout_fixture_async():
            with pytest.raises(SlotsTimeoutError):
                from proxy.observability import _query_slots
                await _query_slots(8080, timeout=0.01)

    def test_detail_fixture_raises_timeout(self):
        """slots_timeout_fixture_detail patches _query_slots_detail."""
        with slots_timeout_fixture_detail():
            with pytest.raises(SlotsTimeoutError):
                from proxy.observability import _query_slots_detail
                asyncio.run(_query_slots_detail(8080, timeout=0.01))


# ===================================================================
# AC6 — Stubbed /slots payloads
# ===================================================================


class TestStubbedSlotsPayloads:
    """AC6 — Stubbed /slots payloads."""

    def test_payload_with_progress_has_numeric_progress(self):
        """stubbed_slots_payload_with_progress includes n_past and progress."""
        payload = stubbed_slots_payload_with_progress(slot_id=0, progress=5000)
        slot = payload[0]
        assert slot["n_past"] == 5000
        assert slot["n_prompt_tokens_processed"] == 5000
        assert slot["context_tokens"] == 5000
        assert slot["is_processing"] is True

    def test_payload_without_progress_missing_numeric_fields(self):
        """stubbed_slots_payload_without_progress omits n_past (b8782 shape)."""
        payload = stubbed_slots_payload_without_progress()
        slot = payload[0]
        assert "n_past" not in slot
        assert "n_prompt_tokens_processed" not in slot
        assert "context_tokens" not in slot
        assert slot["is_processing"] is False

    def test_multi_slot_payload_has_two_slots(self):
        """stubbed_slots_payload_multi_slot yields two slots."""
        payload = stubbed_slots_payload_multi_slot()
        assert len(payload) == 2
        # Slot 0: active
        assert payload[0]["n_past"] == 3000
        assert payload[0]["is_processing"] is True
        # Slot 1: idle
        assert payload[1]["n_past"] == 0
        assert payload[1]["is_processing"] is False

    def test_timeout_payload_is_mock_response(self):
        """stubbed_slots_payload_timeout returns a mock httpx.Response."""
        resp = stubbed_slots_payload_timeout()
        assert isinstance(resp, MagicMock)
        assert resp.status_code == 200


# ===================================================================
# patch_query_slots_progress / patch_llama_status
# ===================================================================


class TestPatchHelpers:
    """Test the patch helper fixtures."""

    def test_patch_query_slots_progress_returns_result(self):
        """patch_query_slots_progress returns the specified dict."""
        expected = {0: {"progress": 3000, "processing": True}}
        with patch_query_slots_progress(expected):
            from proxy.observability import _query_slots_progress
            result = asyncio.run(_query_slots_progress(8080))
            assert result == expected

    def test_patch_query_slots_progress_raises_side_effect(self):
        """patch_query_slots_progress raises on side_effect."""
        with patch_query_slots_progress({}, side_effect=ValueError("test")):
            from proxy.observability import _query_slots_progress
            with pytest.raises(ValueError, match="test"):
                asyncio.run(_query_slots_progress(8080))

    def test_patch_llama_status_returns_result(self):
        """patch_llama_status returns the specified dict."""
        expected = {"kv_cache_tokens": 5000, "n_past": None}
        with patch_llama_status(expected):
            from proxy.observability import query_llama_status
            result = asyncio.run(query_llama_status())
            assert result == expected

    def test_patch_llama_status_raises_side_effect(self):
        """patch_llama_status raises on side_effect."""
        with patch_llama_status({}, side_effect=ConnectionError("no server")):
            from proxy.observability import query_llama_status
            with pytest.raises(ConnectionError, match="no server"):
                asyncio.run(query_llama_status())


# ===================================================================
# make_test_srv
# ===================================================================


class TestMakeTestSrv:
    """Test the minimal server factory."""

    def test_srv_has_required_attributes(self):
        """Server object has all attributes needed by router_helpers."""
        srv = make_test_srv()
        assert hasattr(srv, "config")
        assert hasattr(srv, "local_dispatch_records")
        assert hasattr(srv, "local_dispatch_records_lock")
        assert hasattr(srv, "logger")
        assert hasattr(srv, "local_prefill_in_flight")

    def test_srv_config_defaults(self):
        """Server config has sensible defaults."""
        srv = make_test_srv()
        assert srv.config["server"]["llama_server_port"] == 8080
        assert srv.config["server"]["local_dispatch_lease_prefill_poll_seconds"] > 0
        assert srv.config["server"]["local_dispatch_lease_prefill_buffer_seconds"] > 0

    def test_srv_accepts_custom_config(self):
        """Server accepts custom configuration."""
        custom = {"server": {"llama_server_port": 9999}}
        srv = make_test_srv(config=custom)
        assert srv.config["server"]["llama_server_port"] == 9999

    def test_srv_accepts_custom_records(self):
        """Server accepts custom dispatch records."""
        records = {"key1": {"active": True, "expires_at": time.monotonic() + 10}}
        srv = make_test_srv(records=records)
        assert "key1" in srv.local_dispatch_records


# ===================================================================
# pytest fixture integration
# ===================================================================


class TestPytestFixtures:
    """Verify pytest fixtures work correctly (via context managers)."""

    def test_test_srv_fixture(self):
        """make_test_srv provides a server object."""
        srv = make_test_srv()
        assert srv is not None
        assert hasattr(srv, "config")

    def test_concurrent_slots_fixture(self):
        """concurrent_slots_fixture provides a ConcurrentSlotsState."""
        with concurrent_slots_fixture() as state:
            assert isinstance(state, ConcurrentSlotsState)
            assert len(state.to_slots_list()) == 2

    def test_stalled_slot_fixture(self):
        """stalled_task_fixture provides a StalledSlotState."""
        with stalled_task_fixture() as state:
            assert isinstance(state, StalledSlotState)
            assert state.is_stalled()

    def test_progressing_slot_fixture(self):
        """progressing_task_fixture provides a ProgressingSlotState."""
        with progressing_task_fixture() as state:
            assert isinstance(state, ProgressingSlotState)

    def test_slots_with_progress_fixture(self):
        """slots_with_progress fixture provides payload with progress."""
        payload = stubbed_slots_payload_with_progress()
        assert payload[0]["n_past"] == 5000

    def test_slots_without_progress_fixture(self):
        """slots_without_progress fixture provides b8782 payload."""
        payload = stubbed_slots_payload_without_progress()
        assert "n_past" not in payload[0]

    def test_multi_slot_payload_fixture(self):
        """multi_slot_payload fixture provides two-slot payload."""
        payload = stubbed_slots_payload_multi_slot()
        assert len(payload) == 2

    def test_stderr_with_progress_fixture(self):
        """stderr_with_progress fixture yields progress lines."""
        with stderr_with_progress() as fixture:
            assert fixture.with_progress is True

    def test_stderr_without_progress_fixture(self):
        """stderr_without_progress fixture yields non-progress lines."""
        with stderr_without_progress() as fixture:
            assert fixture.with_progress is False


# ===================================================================
# Integration-style: fixtures work with _query_slots_progress
# ===================================================================


class TestFixtureIntegration:
    """Integration: fixtures produce state that _query_slots_progress can consume."""

    def test_concurrent_slots_match_slots_progress_output(self):
        """ConcurrentSlotsState.to_slots_list() produces data _query_slots_progress expects."""
        with concurrent_slots_fixture(progress_a=2000, progress_b=800) as state:
            slots = state.to_slots_list()
            # _query_slots_progress iterates the list and builds {slot_id: {...}}
            # We verify the slot dicts have the expected keys
            for slot in slots:
                assert "id" in slot
                assert "is_processing" in slot
                assert "n_past" in slot
                assert "context_tokens" in slot

    def test_progressing_slot_matches_progress_output(self):
        """ProgressingSlotState produces state with incrementing progress."""
        with progressing_task_fixture(progress=100, advance_per_tick=100) as state:
            # Initial state
            slots = state.to_slots_list()
            assert slots[0]["n_past"] == 100
            # After tick
            slots = state.tick()
            assert slots[0]["n_past"] == 200

    def test_payload_with_progress_asserts_measured_path(self):
        """A payload with progress should assert the measured path."""
        payload = stubbed_slots_payload_with_progress(progress=7500)
        slot = payload[0]
        # _slot_context_tokens would return 7500 from this slot
        candidates = [
            c
            for c in (slot.get("n_past"), slot.get("n_prompt_tokens_processed"))
            if isinstance(c, (int, float)) and c >= 0
        ]
        assert max(candidates) == 7500

    def test_payload_without_progress_asserts_failsafe_path(self):
        """A payload without progress should assert the fail-safe path."""
        payload = stubbed_slots_payload_without_progress()
        slot = payload[0]
        candidates = [
            c
            for c in (slot.get("n_past"), slot.get("n_prompt_tokens_processed"))
            if isinstance(c, (int, float)) and c >= 0
        ]
        assert candidates == []  # No numeric progress — fail-safe path
