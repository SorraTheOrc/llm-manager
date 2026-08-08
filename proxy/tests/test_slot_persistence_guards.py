"""
Save/restore failure-path test harness (LP-0MSI1RWLM007N367 F2).

Deterministic coverage of the slot persistence guards exercised by F3's
load-aware save gating + timeout rebalance, using a mocked llama-server
(httpx MockTransport) so no real processes or GPU are needed:

- timeout scaling: base + per-token coefficient, capped at max timeout
- circuit breaker under concurrency: 3 consecutive failures -> cooldown
  (no save/restore issued) -> retry after expiry; state resets on restart
- load gate: busy slot -> save/restore skipped, no HTTP request issued
- stalled-save no-wedge: a never-responding transport proves the breaker
  trips and the slot lock releases within a bounded window
- context gating: oversized context -> persistence skipped
"""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from proxy.session import (
    _build_slot_context,
    _call_slot_endpoint,
    _record_slot_failure,
    _slot_failure_state,
    _slot_persistence_skip_when_busy,
    slot_lock_coordinator,
)

pytestmark = pytest.mark.refactor_parity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Base slot-persistence config for non-gate tests (gate disabled by
    default so tests don't depend on ambient global proxy state)."""
    config = {
        "session_slot_save_path": "/tmp/slot-cache",
        "session_slot_pool_size": 3,
        "session_slot_timeout_seconds": 3.0,
        "session_slot_timeout_per_token_seconds": 0.0015,
        "session_slot_max_timeout_seconds": 60.0,
        "session_slot_max_prompt_tokens": 0,
        "session_slot_max_consecutive_failures": 3,
        "session_slot_failure_cooldown_seconds": 300,
        "session_slot_skip_when_busy": False,
    }
    config.update(overrides)
    return config


def _make_gate_config(**overrides):
    """Config with the load-aware gate explicitly enabled (F3 behavior)."""
    config = _make_config(session_slot_skip_when_busy=True)
    config.update(overrides)
    return config


def _body_for_tokens(n_tokens: int) -> dict:
    """A request body whose tiktoken estimate is ≈ *n_tokens*."""
    return {"messages": [{"role": "user", "content": "x" * (n_tokens * 8)}]}


def _make_mock_srv(logger=None, **attrs):
    """Minimal server object; load state attributes are explicit ints so
    MagicMock's __int__ (which returns 1) can't leak into counters."""
    srv = MagicMock()
    srv.logger = logger or logging.getLogger("test_logger")
    srv._http_client = None
    srv.active_queries = 0
    srv.local_active_queries = 0
    srv.local_dispatch_records = {}
    for key, value in attrs.items():
        setattr(srv, key, value)
    return srv


@pytest.fixture(autouse=True)
def _clear_slot_state():
    from proxy.session import _slot_owners
    _slot_owners.clear()
    _slot_failure_state.clear()
    yield
    _slot_owners.clear()
    _slot_failure_state.clear()


# ---------------------------------------------------------------------------
# AC1: Timeout scaling (base + per-token coefficient, capped at max)
# ---------------------------------------------------------------------------

class TestTimeoutScaling:
    def test_timeout_scales_with_context(self):
        """~20K tokens at 0.0015 s/token -> 3.0 + 0.0015*20000 = 33s."""
        config = _make_config()
        _, _, timeout = _build_slot_context(config, "scale-session", _body_for_tokens(20000))
        assert timeout == pytest.approx(3.0 + 0.0015 * 20000, abs=1.0)

    def test_timeout_capped_at_max(self):
        """A huge context is capped at session_slot_max_timeout_seconds (60)."""
        config = _make_config()
        _, _, timeout = _build_slot_context(config, "cap-session", _body_for_tokens(200000))
        assert timeout == 60.0

    def test_timeout_uses_configured_coefficient(self):
        """A larger per-token coefficient yields a longer window for the
        same context (F3 rebalance 0.001 -> 0.0015)."""
        config = _make_config(session_slot_timeout_per_token_seconds=0.0015)
        _, _, t1 = _build_slot_context(config, "coef-a", _body_for_tokens(20000))
        config2 = _make_config(session_slot_timeout_per_token_seconds=0.001)
        _, _, t2 = _build_slot_context(config2, "coef-b", _body_for_tokens(20000))
        assert t1 > t2

    def test_fixed_timeout_when_per_token_disabled(self):
        """per_token=0 disables the add-on (fixed base timeout only)."""
        config = _make_config(session_slot_timeout_per_token_seconds=0.0)
        _, _, timeout = _build_slot_context(config, "fixed-session", _body_for_tokens(20000))
        assert timeout == 3.0


# ---------------------------------------------------------------------------
# AC2: Circuit breaker under concurrency
# ---------------------------------------------------------------------------

class TestCircuitBreakerConcurrency:
    def test_three_failures_trip_breaker_no_request_issued(self):
        """After 3 consecutive failures the gate returns None, so the router
        issues no save/restore HTTP request for that slot."""
        config = _make_config()
        slot_id, _, _ = _build_slot_context(config, "breaker-concurrent")
        assert slot_id is not None
        for _ in range(3):
            _record_slot_failure(slot_id)
        slot_id2, filename2, _ = _build_slot_context(config, "breaker-concurrent")
        assert slot_id2 is None
        assert filename2 is None

    def test_cooldown_expiry_allows_retry(self):
        """After the cooldown window expires, persistence is allowed again."""
        config = _make_config()
        slot_id, _, _ = _build_slot_context(config, "breaker-cooldown")
        assert slot_id is not None
        _slot_failure_state[slot_id] = (3, time.time() - 400)  # expired
        slot_id2, filename2, _ = _build_slot_context(config, "breaker-cooldown")
        assert slot_id2 is not None
        assert filename2 is not None

    def test_breaker_state_resets_on_restart(self):
        """The in-memory breaker state is cleared by a restart (module state
        wipe), so persistence resumes immediately after restart."""
        config = _make_config()
        slot_id, _, _ = _build_slot_context(config, "breaker-restart")
        assert slot_id is not None
        for _ in range(3):
            _record_slot_failure(slot_id)
        # Simulate restart: module-level state is reinitialized.
        _slot_failure_state.clear()
        slot_id2, filename2, _ = _build_slot_context(config, "breaker-restart")
        assert slot_id2 is not None
        assert filename2 is not None

    @pytest.mark.asyncio
    async def test_concurrent_failures_trip_breaker(self):
        """3 concurrent failing saves trip the breaker; the gate then
        disables persistence (no further requests issued)."""
        from proxy.session import _slot_owners
        # Force the session onto slot 2 (pool 0-2) by occupying slots 0,1.
        _slot_owners[0] = "filler-0"
        _slot_owners[1] = "filler-1"
        config = _make_config()
        slot_id, _, _ = _build_slot_context(config, "breaker-concurrent-2")
        assert slot_id == 2

        mock_srv = _make_mock_srv()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("stalled"))
        mock_srv._http_client = mock_client

        with patch("proxy.session._srv", return_value=mock_srv):
            results = await asyncio.gather(*[
                _call_slot_endpoint(8080, 2, "save", "/tmp/s.bin", timeout=0.5)
                for _ in range(3)
            ])
        assert results == [False, False, False]
        count, _ = _slot_failure_state.get(2, (0, 0.0))
        assert count == 3
        # The slot is in cooldown -> persistence disabled.
        slot_id2, filename2, _ = _build_slot_context(config, "breaker-concurrent-2")
        assert slot_id2 is None
        assert filename2 is None


# ---------------------------------------------------------------------------
# AC3: Load gate (busy slot -> save/restore skipped, no HTTP request)
# ---------------------------------------------------------------------------

class TestLoadGate:
    def test_busy_other_session_skips_persistence(self):
        """When another local session is actively streaming, persistence is
        skipped (gate returns None -> no save/restore HTTP request)."""
        config = _make_gate_config()
        mock_srv = _make_mock_srv(
            local_dispatch_records={"other-session": {"active": True}},
        )
        with patch("proxy.session._srv", return_value=mock_srv):
            slot_id, filename, _ = _build_slot_context(
                config, "busy-session", _body_for_tokens(5000)
            )
        assert slot_id is None
        assert filename is None

    def test_idle_persistence_allowed(self):
        """No active local sessions -> persistence proceeds."""
        config = _make_gate_config()
        mock_srv = _make_mock_srv(local_dispatch_records={})
        with patch("proxy.session._srv", return_value=mock_srv):
            slot_id, filename, _ = _build_slot_context(
                config, "idle-session", _body_for_tokens(5000)
            )
        assert slot_id is not None
        assert filename is not None

    def test_gate_disabled_when_config_false(self):
        """session_slot_skip_when_busy=false disables the gate even under load."""
        config = _make_config(session_slot_skip_when_busy=False)
        mock_srv = _make_mock_srv(
            local_dispatch_records={"other-session": {"active": True}},
        )
        with patch("proxy.session._srv", return_value=mock_srv):
            slot_id, filename, _ = _build_slot_context(
                config, "gate-off-session", _body_for_tokens(5000)
            )
        assert slot_id is not None
        assert filename is not None

    def test_own_session_not_counted_as_busy(self):
        """The requesting session's own active record does not make the slot
        busy (the gate excludes the current session)."""
        config = _make_gate_config()
        mock_srv = _make_mock_srv(
            local_dispatch_records={"self-session": {"active": True}},
        )
        with patch("proxy.session._srv", return_value=mock_srv):
            slot_id, filename, _ = _build_slot_context(
                config, "self-session", _body_for_tokens(5000)
            )
        assert slot_id is not None
        assert filename is not None

    def test_skip_when_busy_helper_true_with_other_session(self):
        """_slot_persistence_skip_when_busy is True when another session is
        active and False when only the current session is active."""
        mock_srv = _make_mock_srv(
            local_dispatch_records={"other": {"active": True}},
        )
        with patch("proxy.session._srv", return_value=mock_srv):
            assert _slot_persistence_skip_when_busy(
                _make_gate_config(), slot_id=0, session_id="self"
            ) is True
            # Only the current session active -> not busy.
            mock_srv.local_dispatch_records = {"self": {"active": True}}
            assert _slot_persistence_skip_when_busy(
                _make_gate_config(), slot_id=0, session_id="self"
            ) is False

    def test_skip_when_busy_helper_respects_config_off(self):
        """The helper returns False when the gate is configured off."""
        mock_srv = _make_mock_srv(
            local_dispatch_records={"other": {"active": True}},
        )
        with patch("proxy.session._srv", return_value=mock_srv):
            assert _slot_persistence_skip_when_busy(
                _make_config(),
                slot_id=0,
                session_id="self",
            ) is False


# ---------------------------------------------------------------------------
# AC4: Stalled-save no-wedge invariant
# ---------------------------------------------------------------------------

class _HungLlamaServer:
    """A TCP server that accepts connections but never responds — simulates a
    wedged llama-server so httpx's genuine read timeout fires (MockTransport
    does NOT enforce client timeouts, so a real socket is required)."""

    def __init__(self):
        self._server = None
        self._writers: list = []
        self.port = 0

    async def __aenter__(self):
        self._server = await asyncio.start_server(
            self._handle, "localhost", 0
        )
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def _handle(self, reader, writer):
        self._writers.append(writer)
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass

    async def __aexit__(self, *exc):
        for writer in self._writers:
            writer.close()
        self._server.close()
        await self._server.wait_closed()


class TestStalledSaveNoWedge:
    @pytest.mark.asyncio
    async def test_stalled_save_returns_within_bounded_window(self):
        """A save to a never-responding backend returns False (not hangs)
        within a bounded window, and feeds the circuit breaker."""
        async with _HungLlamaServer() as hung:
            mock_srv = _make_mock_srv()
            with patch("proxy.session._srv", return_value=mock_srv):
                result = await asyncio.wait_for(
                    _call_slot_endpoint(hung.port, 4, "save", "/tmp/s.bin", timeout=0.3),
                    timeout=5.0,
                )
        assert result is False
        count, _ = _slot_failure_state.get(4, (0, 0.0))
        assert count == 1

    @pytest.mark.asyncio
    async def test_repeated_stalls_trip_breaker(self):
        """3 stalled saves trip the breaker: the gate then returns None so no
        further save/restore request is issued (no pile-up on the GPU)."""
        from proxy.session import _slot_owners
        _slot_owners[0] = "filler-0"
        _slot_owners[1] = "filler-1"
        config = _make_config()
        slot_id, _, _ = _build_slot_context(config, "stall-trip-session")
        assert slot_id == 2

        async with _HungLlamaServer() as hung:
            mock_srv = _make_mock_srv()
            with patch("proxy.session._srv", return_value=mock_srv):
                for _ in range(3):
                    result = await asyncio.wait_for(
                        _call_slot_endpoint(hung.port, 2, "save", "/tmp/s.bin", timeout=0.3),
                        timeout=5.0,
                    )
                    assert result is False
        slot_id2, filename2, _ = _build_slot_context(config, "stall-trip-session")
        # The stalled slot is now in cooldown -> persistence disabled.
        assert slot_id2 is None
        assert filename2 is None

    @pytest.mark.asyncio
    async def test_slot_lock_releases_after_stalled_save(self):
        """The per-slot lock held around a stalled save is released promptly,
        so the router can proceed (no GPU wedge / deadlock)."""
        async with _HungLlamaServer() as hung:
            mock_srv = _make_mock_srv()
            with patch("proxy.session._srv", return_value=mock_srv):
                guard = slot_lock_coordinator.acquire(2)
                async with guard:
                    result = await asyncio.wait_for(
                        _call_slot_endpoint(hung.port, 2, "save", "/tmp/s.bin", timeout=0.3),
                        timeout=5.0,
                    )
                    assert result is False
                # After the guarded block exits, the lock is free: re-acquiring
                # with a bound proves it released (a wedge would time out).
                guard2 = slot_lock_coordinator.acquire(2)
                async with asyncio.timeout(2.0):
                    async with guard2:
                        pass


# ---------------------------------------------------------------------------
# AC5: Context gating still applies (oversized -> skipped)
# ---------------------------------------------------------------------------

class TestContextGating:
    def test_oversized_context_skips_persistence(self):
        """A context above the derived cap disables persistence regardless of
        load state (existing clamp-derived gate preserved)."""
        config = _make_config(
            session_slot_max_prompt_tokens=0,  # dynamic derivation
            local_model_ctx_size=131072,
        )
        # ~50K tokens > derived cap (39594 for 3-slot) -> skipped.
        with patch("proxy.session._srv", return_value=_make_mock_srv()):
            slot_id, filename, _ = _build_slot_context(
                config, "oversize-session", _body_for_tokens(50000)
            )
        assert slot_id is None
        assert filename is None

    def test_undersized_context_persists(self):
        """A context within the cap persists under idle load."""
        config = _make_config(
            session_slot_max_prompt_tokens=0,
            local_model_ctx_size=131072,
        )
        with patch("proxy.session._srv", return_value=_make_mock_srv()):
            slot_id, filename, _ = _build_slot_context(
                config, "normal-session", _body_for_tokens(5000)
            )
        assert slot_id is not None
        assert filename is not None
