"""Regression tests reproducing the RCA's abandoned-stream stuck state.

Parent: LP-0MSU72I2V009YN79 (RCA: llama-proxy reported busy for the entire
7h09m idle window 2026-08-07/08; zero herdr downtime dispatch).
Child:  LP-0MSU92G4G003U4AP (F5: Tests: reproduce abandoned-stream stuck state).

The RCA proposed three proxy-side mechanisms that could make
``GET /llama/local/status`` report busy while the local model was idle:

- **H6a** — stuck global ``active_queries`` counter: an abandoned local
  stream (started, never "Stream finished") leaves the counter > 0 forever,
  so the status endpoint reports ``active_query=true``.
- **H6b** — lingering local dispatch lease: the abandoned stream's lease
  record stays ``active`` past the stream end, so the endpoint reports the
  owner session until idle-timeout.
- **H6c** — ``/slots`` failures during model reload (covered by F3, not here).

These tests reproduce the stuck state through the REAL production paths —
the counter/lease helpers in ``router_helpers.py`` and the real
``/llama/local/status`` handler (ASGI transport) — and assert exact counter
and status values. They mirror the RCA scenario: increment → abandon
without cleanup → busy status; the periodic recovery
(``_recover_stuck_global_active_queries``) must NOT reset while the
abandoned stream's lease record is still active (the ``has_active`` guard);
after idle-timeout, ``_cleanup_stale_local_dispatch`` + the recovery loop
restore idle; and repeated herdr-style polls then pass the 4-minute
continuous-idle dispatch gate.
"""

import asyncio
import time
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import httpx
import pytest


class _RcaStuckStateHarness:
    """Drive the real status endpoint + lifecycle helpers against a controlled state.

    Patches ``proxy.server`` module attributes (the state object the real
    handler reads via ``_srv()``) into the RCA's idle-window shape: llama
    server running, Qwen3 loaded (3/3 slots free), no model switch, no
    background loads — so the ONLY busy signal comes from the counters /
    lease records under test.
    """

    SESSION = "019fdf5f-abandoned"

    FREE_SLOTS = [
        {"slot_id": 0, "is_processing": False, "n_decoded": None},
        {"slot_id": 1, "is_processing": False, "n_decoded": None},
        {"slot_id": 2, "is_processing": False, "n_decoded": None},
    ]

    class _FakeLock:
        def locked(self):
            return False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

    def __init__(self):
        import proxy.server as server

        self._stack = ExitStack()
        s = self._stack
        s.enter_context(patch("proxy.server.query_llama_status", side_effect=self._fake_query))
        s.enter_context(patch.object(server, "active_queries", 0))
        s.enter_context(patch.object(server, "active_queries_lock", asyncio.Lock()))
        s.enter_context(patch.object(server, "local_active_queries", 0))
        s.enter_context(patch.object(server, "local_active_queries_lock", asyncio.Lock()))
        s.enter_context(patch.object(server, "local_dispatch_records", {}))
        s.enter_context(patch.object(server, "local_dispatch_records_lock", asyncio.Lock()))
        s.enter_context(patch.object(server, "model_switch_refcount", 0))
        s.enter_context(patch.object(server, "model_switch_lock", self._FakeLock()))
        s.enter_context(patch.object(server, "background_loads", {}))
        s.enter_context(patch.object(server, "current_model", "test-model"))
        s.enter_context(
            patch.object(
                server,
                "config",
                {"server": {"llama_server_port": 8080, "local_dispatch_lease_timeout_seconds": 60}},
            )
        )
        # Slots are always stubbed: 3/3 free so the test never touches a real
        # llama-server on localhost, and "slots free" is explicit (AC4).
        s.enter_context(patch("proxy.observability._query_slots", AsyncMock(return_value=(3, 3))))
        s.enter_context(
            patch("proxy.observability._query_slots_detail", AsyncMock(return_value=self.FREE_SLOTS))
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._stack.close()

    async def _fake_query(self):
        return {"llama_server_running": True}

    async def status(self) -> dict:
        """GET /llama/local/status through the real ASGI app and return JSON."""
        from proxy.server import app

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/llama/local/status")
        assert resp.status_code == 200
        return resp.json()


async def _recovery_tick():
    """One dispatch-cleanup-loop tick, mirroring server._dispatch_cleanup_loop."""
    import proxy.server as server
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
        _recover_stuck_global_active_queries,
        _recover_stuck_local_active_queries,
    )

    await _cleanup_stale_local_dispatch(server)
    await _recover_stuck_local_active_queries(server)
    await _recover_stuck_global_active_queries(server)


# ======================================================================
# AC1: abandoned stream leaves exact stuck counters + busy status
# ======================================================================


@pytest.mark.asyncio
async def test_abandoned_stream_leaves_exact_stuck_counters_and_busy_status():
    """An abandoned stream (no cleanup / no "Stream finished") leaves the
    counters > 0 and the status endpoint reports active_query=true.

    Mirrors RCA H6a: the two local streams (sessions 019fdea8, 019fdebf)
    started but never logged "Stream finished", so the global counter was
    never decremented. Asserts the EXACT counter/status values, not just
    truthiness.
    """
    import proxy.server as server
    from proxy.router_helpers import (
        _increment_active_queries,
        _increment_local_active_queries,
        _recover_stuck_global_active_queries,
        _recover_stuck_local_active_queries,
    )

    with _RcaStuckStateHarness() as h:
        # Stream start: global counter incremented on the routing path,
        # local counter + dispatch lease acquired for the session.
        await _increment_active_queries(server)
        await _increment_local_active_queries(server, session_key=h.SESSION, backend="local")

        # Exact counter/lease values after the stream started.
        assert server.active_queries == 1
        assert server.local_active_queries == 1
        rec = server.local_dispatch_records[h.SESSION]
        assert rec["active"] is True
        assert rec["expires_at"] > time.monotonic()

        # The stream is abandoned: cleanup/"Stream finished" never runs, so
        # nothing decrements the counters. The status endpoint must report
        # busy with the exact owner fields.
        j = await h.status()
        assert j["active_query"] is True
        assert j["local_active_query"] is True
        assert j["local_owner_session_id"] == h.SESSION
        assert j["local_owner_lease_remaining_seconds"] is not None
        assert 0 < j["local_owner_lease_remaining_seconds"] <= 60.0
        # Slots are free — the busy signal is the counters, not capacity.
        assert j["available_slots"] == 3
        assert j["total_slots"] == 3

        # The periodic recovery must NOT reset the counters while the
        # abandoned stream's lease record is still active (the `has_active`
        # guard) — the busy state is held until idle-timeout. This is the
        # residual-gap probe flagged by the plan for F4/F6.
        await _recover_stuck_local_active_queries(server)
        await _recover_stuck_global_active_queries(server)
        assert server.local_active_queries == 1
        assert server.active_queries == 1
        j = await h.status()
        assert j["active_query"] is True


# ======================================================================
# AC2: lease held past stream end keeps status busy until idle-timeout
# ======================================================================


@pytest.mark.asyncio
async def test_lease_held_past_stream_end_busy_until_idle_timeout_then_idle():
    """A dispatch lease held past stream end keeps the status busy until
    idle-timeout; after expiry + _cleanup_stale_local_dispatch the status
    reports idle.

    Mirrors RCA H6b: the last audit-wave pane (session 019fdf5f) held a
    local dispatch lease past its stream end (release-on-finish never ran),
    so the lease stayed active until idle-timeout and the status stayed
    busy. Uses the router's real acquisition path
    (_try_acquire_local_dispatch) to create the lease.
    """
    import proxy.server as server
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
        _increment_active_queries,
        _try_acquire_local_dispatch,
    )

    with _RcaStuckStateHarness() as h:
        # Router acquisition path: stream start acquires the dispatch lease
        # and increments both counters.
        await _increment_active_queries(server)
        acquired, owner, active_count, retry_after = await _try_acquire_local_dispatch(
            server, max_local=1, session_key=h.SESSION, backend="local"
        )
        assert acquired is True
        assert owner is None
        assert active_count == 1
        rec = server.local_dispatch_records[h.SESSION]
        assert rec["active"] is True
        assert rec["expires_at"] > time.monotonic() + 30  # 60s lease window

        # Stream end: the stream's bytes stopped flowing but the release path
        # never ran — the lease is still active and the status stays busy.
        j = await h.status()
        assert j["active_query"] is True
        assert j["local_active_query"] is True
        assert j["local_owner_session_id"] == h.SESSION
        assert 0 < j["local_owner_lease_remaining_seconds"] <= 60.0

        # Still busy on a later poll before idle-timeout.
        j2 = await h.status()
        assert j2["active_query"] is True
        assert j2["local_owner_session_id"] == h.SESSION

        # Idle-timeout: expires_at passes. The periodic cleanup removes the
        # orphan lease (decrementing the local counter); the recovery loop
        # then resets the stuck global counter.
        server.local_dispatch_records[h.SESSION]["expires_at"] = time.monotonic() - 1.0
        removed = await _cleanup_stale_local_dispatch(server)
        assert removed == 1
        assert h.SESSION not in server.local_dispatch_records
        assert server.local_active_queries == 0

        await _recovery_tick()
        assert server.active_queries == 0
        assert server.local_active_queries == 0

        # Status reports idle: no active query, no lease owner, slots free.
        j3 = await h.status()
        assert j3["active_query"] is False
        assert j3["local_active_query"] is False
        assert j3["local_owner_session_id"] is None
        assert j3["local_owner_lease_remaining_seconds"] is None
        assert j3["available_slots"] == 3
        assert j3["total_slots"] == 3


# ======================================================================
# AC3: recovery leaves the global counter intact while local work exists
# ======================================================================


@pytest.mark.asyncio
async def test_recovery_keeps_global_counter_while_anonymous_local_request_in_flight():
    """_recover_stuck_global_active_queries leaves the global counter intact
    while a legitimate anonymous local request is in flight (local counter > 0).

    Anonymous sessions (no session affinity) increment the local counter
    without creating a dispatch record (AC3's "local count > 0" branch).
    The global counter must NOT be reset — an in-flight request is real
    work, not a leak. Exercises the real increment helpers + recovery +
    status endpoint end-to-end.
    """
    import proxy.server as server
    from proxy.router_helpers import (
        _increment_active_queries,
        _increment_local_active_queries,
        _recover_stuck_global_active_queries,
    )

    with _RcaStuckStateHarness() as h:
        # Anonymous stream: counters incremented, NO lease record created.
        await _increment_active_queries(server)
        await _increment_local_active_queries(server)
        assert server.active_queries == 1
        assert server.local_active_queries == 1
        assert server.local_dispatch_records == {}

        # Recovery must NOT reset while a local request is in flight.
        await _recover_stuck_global_active_queries(server)
        assert server.active_queries == 1

        # Status still reports the in-flight local work.
        j = await h.status()
        assert j["active_query"] is True
        assert j["local_active_query"] is True


# ======================================================================
# AC4: herdr-polling simulation — repeated polls idle + 4-min gate passes
# ======================================================================

# herdr's DEFAULT_DOWNTIME_IDLE_THRESHOLD_MS = 240_000 (4 minutes):
# ContextHub packages/herdr/src/downtime-worker.ts. Pollers run at a ~30s
# cadence (per the RCA's inter-poll gap histogram, mean gap 3.78s across
# ~8 pollers → ~30s per poller).
_POLL_EVERY_S = 30
_IDLE_GATE_SECONDS = 4 * 60


def _herdr_is_idle(status: dict) -> bool:
    """Mirror herdr's isIdleStatus against a /llama/local/status payload.

    Busy (fail-closed) when: ``local_active_query`` (falling back to the
    global ``active_query`` for pre-observability proxies) is true, total
    slots are unknown/zero, or fewer than all slots are free (default
    requiredFreeSlots=0 → all slots must be free).
    """
    busy_query = status.get("local_active_query", status.get("active_query", False))
    if busy_query:
        return False
    total = status.get("total_slots", 0)
    available = status.get("available_slots", 0)
    if not isinstance(total, (int, float)) or total <= 0:
        return False
    return bool(available >= total)


def _longest_continuous_idle_seconds(polls: list[tuple[float, bool]]) -> float:
    """Longest run of consecutive idle polls (seconds), the basis of herdr's
    ≥4-minute continuous-idle dispatch gate."""
    best = 0.0
    run = 0.0
    last_t = None
    for t, idle in polls:
        if idle:
            if last_t is not None:
                run += t - last_t
                best = max(best, run)
        else:
            run = 0.0
        last_t = t
    return best


@pytest.mark.asyncio
async def test_herdr_polling_after_recovery_passes_continuous_idle_gate():
    """After recovery, repeated status polls see active_query=false with all
    slots free, and the ≥4-minute continuous-idle dispatch gate passes (the
    dispatch would occur). A busy poll inside the window must break the gate.
    """
    import proxy.server as server
    from proxy.router_helpers import _increment_active_queries, _increment_local_active_queries

    with _RcaStuckStateHarness() as h:
        # Reproduce the abandoned stream, then let idle-timeout + the
        # recovery loop restore the idle state.
        await _increment_active_queries(server)
        await _increment_local_active_queries(server, session_key=h.SESSION, backend="local")
        server.local_dispatch_records[h.SESSION]["expires_at"] = time.monotonic() - 1.0
        await _recovery_tick()
        assert server.active_queries == 0
        assert server.local_active_queries == 0

        # Phase 1: a 240s window containing ONE busy poll (a new local request
        # at t=180) — the continuous-idle gate must NOT pass.
        polls: list[tuple[float, bool]] = []
        for i in range(9):
            t = float(i * _POLL_EVERY_S)
            if i == 6:
                # A new local request starts at t=180 — real handler reports busy.
                server.active_queries = 1
                server.local_active_queries = 1
            j = await h.status()
            if i == 6:
                # The request finishes immediately; the idle window resumes.
                server.active_queries = 0
                server.local_active_queries = 0
            # Every poll carries countable slots (fail-closed consumer never
            # sees total_slots=0) and, when idle, all slots free.
            assert j["total_slots"] == 3
            assert j["available_slots"] == 3
            polls.append((t, _herdr_is_idle(j)))

        assert any(not idle for _, idle in polls), "control: the busy poll must be present"
        assert _longest_continuous_idle_seconds(polls) < _IDLE_GATE_SECONDS, (
            "A busy poll inside the window must break the ≥4-min idle gate"
        )

        # Phase 2: a fresh ≥4-min window with NO busy polls — the gate passes
        # and a downtime dispatch would occur.
        polls2: list[tuple[float, bool]] = []
        for i in range(9):
            t = 300.0 + i * _POLL_EVERY_S
            j = await h.status()
            assert j["active_query"] is False
            assert j["local_active_query"] is False
            assert j["available_slots"] == j["total_slots"] == 3
            # Per-slot evidence: the same free slots stay free across polls
            # (LP-0MSORPUMX002LLIA — herdr tracks same-slot idleness).
            assert all(not slot["is_processing"] for slot in j["slots"])
            polls2.append((t, _herdr_is_idle(j)))

        assert all(idle for _, idle in polls2)
        assert _longest_continuous_idle_seconds(polls2) >= _IDLE_GATE_SECONDS, (
            "A fully idle ≥4-min window must pass the continuous-idle gate"
        )
