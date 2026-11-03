"""Restart-churn bounding for llama-server self-heal (LP-0MVZVZLCS000LQNB).

Context: on 2026-11-02 the proxy logged 22 ``Attempting host-first startup``
lines in ~2 h and 14 distinct ``llama-server*.log`` files. Two independent
defects produced that: a llama-server SIGSEGV in ``libggml-cpu.so.0.9.11``
(diagnosis: ``docs/segfault-libggml-cpu-LP-0MVZVZLCS000LQNB.md``) and a
self-heal loop that turned one crash into a burst.

This module pins the loop-bounding half:

* **AC2** — every restart attempt carries an explicit ``restart_reason``, and a
  backend that is already reachable on the port is adopted instead of being
  spawned over (which is what produced the port-conflict cascade).
* **AC3** — a deliberate restart records the dispatches it abandoned as
  ``backend_restart`` evictions, so a restart is visible in the eviction
  accounting instead of looking like a capacity problem.
* **AC4** — hermetic regression: a crashed backend is **not** respawned while
  an unadopted spawn/backend holds the port.

No test spawns a process or opens a socket.
"""

import logging

import proxy.server as server
import pytest

from proxy import admission_events

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, pid: int = 5101) -> None:
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.returncode = -15


def _config(**overrides) -> dict:
    cfg = {
        "llama_router_mode": True,
        "llama_server_port": 8080,
        "llama_self_heal_max_attempts": 2,
        "llama_self_heal_window_seconds": 300,
        "llama_self_heal_backoff_base_seconds": 0,
        "llama_self_heal_retry_after_seconds": 30,
        "llama_startup_timeout": 1,
        "session_slot_pool_size": 3,
    }
    cfg.update(overrides)
    return {"server": cfg}


def _async_return(value):
    async def _inner(*_a, **_kw):
        return value

    return _inner


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.setattr(server, "llama_process", None)
    monkeypatch.setattr(server, "backend_ready", False)
    monkeypatch.setattr(server, "current_model", None)
    monkeypatch.setattr(
        server,
        "backend_recovery_state",
        {
            "in_progress": False,
            "attempt_timestamps": [],
            "max_attempts": 2,
            "window_seconds": 300,
            "retry_after_seconds": 30,
            "last_failure": None,
        },
    )
    monkeypatch.setattr(server, "config", _config())
    monkeypatch.setattr(server, "_kill_process_group", lambda proc, logger, **kw: True)
    monkeypatch.setattr(server, "_record_backend_signal", lambda *_a, **_kw: None)
    admission_events.reset_eviction_after_acceptance_counters()
    yield
    admission_events.reset_eviction_after_acceptance_counters()


# ---------------------------------------------------------------------------
# AC4 / AC2 — never respawn over a backend that already holds the port
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_respawn_when_unadopted_backend_holds_the_port(monkeypatch):
    """AC4: the retry adopts a port that is already held — never respawns over it.

    Shape of the 2026-11-02 21:39-21:41Z cascade: attempt 1 spawns, its
    readiness wait expires with nothing reachable yet; attempt 2 then found a
    backend holding :8080 and died on ``couldn't bind HTTP server socket``.
    One death must yield at most one spawn, so the retry adopts the backend
    that is already serving.
    """
    spawned: list = []
    probe_results = [False, True]  # attempt 1 settle: free; attempt 2 guard: held

    async def _probe(_port):
        return probe_results.pop(0) if probe_results else True

    def fake_start(_model):
        proc = _FakeProc()
        spawned.append(proc)
        return proc

    monkeypatch.setattr(server, "start_llama_server", fake_start)
    monkeypatch.setattr(server, "wait_for_llama_server", _async_return(False))
    monkeypatch.setattr(server, "_probe_backend_reachable", _probe)

    recovered = await server._attempt_router_self_heal(
        reason="llama_server_exit_code_137"
    )

    assert recovered is True
    assert len(spawned) == 1, (
        "one death must yield at most one spawn — a backend already answering "
        "on the port must never be spawned over (the 21:40:16Z cascade)"
    )
    adopted = server.llama_process
    assert adopted is not None and getattr(adopted, "adopted", False) is True
    assert server.backend_ready is True


@pytest.mark.asyncio
async def test_port_held_restart_records_the_reason(monkeypatch, caplog):
    """AC2: the no-respawn decision carries an explicit restart_reason."""
    spawned: list = []
    probe_results = [False, True]

    async def _probe(_port):
        return probe_results.pop(0) if probe_results else True

    def fake_start(_model):
        proc = _FakeProc()
        spawned.append(proc)
        return proc

    monkeypatch.setattr(server, "start_llama_server", fake_start)
    monkeypatch.setattr(server, "wait_for_llama_server", _async_return(False))
    monkeypatch.setattr(server, "_probe_backend_reachable", _probe)

    with caplog.at_level(logging.WARNING, logger="llama-proxy"):
        await server._attempt_router_self_heal(reason="llama_server_exit_code_137")

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "restart_reason=port_held_by_unmanaged_backend" in messages
    assert "restart_reason=llama_server_exit_code_137" in messages


@pytest.mark.asyncio
async def test_restart_attempt_logs_the_death_reason(monkeypatch, caplog):
    """AC2: the attempt line names why the restart happened."""
    proc = _FakeProc()
    monkeypatch.setattr(server, "start_llama_server", lambda _model: proc)
    monkeypatch.setattr(server, "wait_for_llama_server", _async_return(False))
    monkeypatch.setattr(server, "_probe_backend_reachable", _async_return(False))

    with caplog.at_level(logging.WARNING, logger="llama-proxy"):
        recovered = await server._attempt_router_self_heal(
            reason="llama_server_exit_code_137"
        )

    assert recovered is False
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "restart_reason=llama_server_exit_code_137" in messages
    assert server.backend_recovery_state.get("last_restart_reason") == (
        "llama_server_exit_code_137"
    )


@pytest.mark.asyncio
async def test_retry_after_a_settled_failure_may_spawn_again(monkeypatch):
    """AC2 guard: a *settled* failure frees the port, so a bounded retry spawns."""
    spawned: list = []

    def fake_start(_model):
        proc = _FakeProc()
        spawned.append(proc)
        return proc

    monkeypatch.setattr(server, "start_llama_server", fake_start)
    monkeypatch.setattr(server, "wait_for_llama_server", _async_return(False))
    # Port stays free: the previous spawn was terminated, so nothing holds it.
    monkeypatch.setattr(server, "_probe_backend_reachable", _async_return(False))

    recovered = await server._attempt_router_self_heal(reason="backend_unreachable")

    assert recovered is False
    assert len(spawned) == 2, "the bounded retry budget is still used when settled"


# ---------------------------------------------------------------------------
# AC3 — a restart is visible in the eviction accounting
# ---------------------------------------------------------------------------


def _dispatch_record(session_id: str, *, first_byte_seen: bool = False) -> dict:
    return {
        "session_id": session_id,
        "active": True,
        "first_byte_seen": first_byte_seen,
        "started_at": 0.0,
        "estimated_tokens": 12000,
        "new_tokens": 9000,
        "eta_prefill_s": 40.0,
        "slot_id": 1,
    }


def test_backend_restart_reason_is_in_the_vocabulary():
    """AC3: the new reason is part of the documented closed vocabulary."""
    assert admission_events.EVICTION_BACKEND_RESTART == "backend_restart"
    assert (
        admission_events.EVICTION_BACKEND_RESTART
        in admission_events.EVICTION_REASONS_AFTER_ACCEPTANCE
    )
    assert admission_events.EVICTION_BACKEND_RESTART in admission_events.EVICTION_REASONS


def test_backend_restart_records_abandoned_prefills(monkeypatch):
    """AC3: in-flight dispatches are recorded as backend_restart evictions."""
    records = {
        "sess-a": _dispatch_record("sess-a"),
        "sess-b": _dispatch_record("sess-b"),
    }
    monkeypatch.setattr(server, "local_dispatch_records", records)

    recorded = admission_events.record_backend_restart_evictions(
        server, restart_reason="llama_server_exit_code_137"
    )

    assert recorded == 2
    counters = admission_events.eviction_after_acceptance_counters()
    assert counters["by_reason"][admission_events.EVICTION_BACKEND_RESTART] == 2


def test_backend_restart_skips_dispatches_that_already_served(monkeypatch):
    """AC3 guard: a served response is not an abandoned prefill."""
    records = {
        "served": _dispatch_record("served", first_byte_seen=True),
        "abandoned": _dispatch_record("abandoned"),
    }
    monkeypatch.setattr(server, "local_dispatch_records", records)

    recorded = admission_events.record_backend_restart_evictions(
        server, restart_reason="manual_restart"
    )

    assert recorded == 1
    counters = admission_events.eviction_after_acceptance_counters()
    assert counters["by_reason"][admission_events.EVICTION_BACKEND_RESTART] == 1


def test_backend_restart_is_a_noop_without_a_registry(monkeypatch):
    """AC3 guard: legacy servers without dispatch records are unaffected."""
    monkeypatch.setattr(server, "local_dispatch_records", None)

    assert (
        admission_events.record_backend_restart_evictions(server, restart_reason="x") == 0
    )
