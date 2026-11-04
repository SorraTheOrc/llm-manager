"""Regression tests for LP-0MVZT9L3N001XABB — backend self-heal orphan adoption.

Evidence (2026-11-02 21:38-22:00Z, ``/var/log/llama-proxy/proxy.log``): a
host-first router spawn that outlived ``wait_for_llama_server``'s readiness
budget was declared failed and left running but unadopted, holding port 8080.
The next self-heal attempt then died on ``couldn't bind HTTP server socket ...
port: 8080``, self-heal exhausted, and the proxy reported
``llama_server_running: false`` / ``total_slots: 0`` for ~20 minutes while the
orphan served the local route perfectly.

Acceptance criteria covered here:

* **AC1** — a spawn whose readiness wait expires is *adopted* when it is
  reachable and *terminated* when it is not: it is never left unadopted
  holding the port.
* **AC2** — ``GET /health`` reports ``llama_server_running: true`` and
  ``backend_reachable: true`` for a reachable backend while
  ``llama_process is None`` (router mode).
* **AC3** — ``GET /llama/local/status`` reports non-zero ``total_slots`` in
  that same state.
* **AC4** — a port-bind conflict names the conflicting listener pid in the
  log, so the orphan is identifiable without ``ss``.

Every test is hermetic: no real spawn, no real socket, no real ``ss``.
"""

import logging

import httpx
import proxy.lifecycle as lifecycle
import proxy.server as server
import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _FakeProc:
    """Minimal Popen stand-in: alive (poll() -> None) unless killed."""

    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15


def _self_heal_config(**overrides) -> dict:
    cfg = {
        "llama_router_mode": True,
        "llama_server_port": 8080,
        "llama_self_heal_max_attempts": 1,
        "llama_self_heal_window_seconds": 300,
        "llama_self_heal_backoff_base_seconds": 0,
        "llama_self_heal_retry_after_seconds": 30,
        "llama_startup_timeout": 1,
        "session_slot_pool_size": 3,
    }
    cfg.update(overrides)
    return {"server": cfg}


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Pin the process globals these tests mutate (mirrors test_backend_resilience)."""
    monkeypatch.setattr(server, "llama_process", None)
    monkeypatch.setattr(server, "backend_ready", False)
    monkeypatch.setattr(server, "current_model", None)
    monkeypatch.setattr(
        server,
        "backend_recovery_state",
        {
            "in_progress": False,
            "attempt_timestamps": [],
            "max_attempts": 1,
            "window_seconds": 300,
            "retry_after_seconds": 30,
            "last_failure": None,
        },
    )
    monkeypatch.setattr(server, "config", _self_heal_config())
    # Never let a test reach a real socket or a real process group.
    monkeypatch.setattr(
        server, "_kill_process_group", lambda proc, logger, **kw: True
    )
    yield


@pytest.fixture
def app():
    from proxy.server import app as fastapi_app

    return fastapi_app


@pytest.fixture
def transport(app):
    return httpx.ASGITransport(app=app)


@pytest.fixture(autouse=True)
def _stub_slot_queries(monkeypatch):
    """Keep the status endpoint hermetic (no live ``/slots`` HTTP calls)."""
    import proxy.observability as obs

    async def _fake_query_slots(client, port, timeout=2.0, model=None, status_out=None):
        if isinstance(status_out, dict):
            status_out.update({"status": 200, "reason": None})
        return 1, 3

    async def _fake_query_detail(port, timeout=2.0, model=None, _client=None):
        return [{"slot_id": 0, "is_processing": False, "n_decoded": 0}]

    monkeypatch.setattr(obs, "_query_slots", _fake_query_slots)
    monkeypatch.setattr(obs, "_query_slots_detail", _fake_query_detail)


# ---------------------------------------------------------------------------
# AC1 — the readiness-timeout spawn is settled, never orphaned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_heal_adopts_spawn_that_becomes_reachable_after_readiness_timeout(
    monkeypatch,
):
    """AC1: a late-ready spawn is adopted instead of being discarded."""
    proc = _FakeProc()
    killed: list = []

    monkeypatch.setattr(server, "start_llama_server", lambda _model: proc)
    monkeypatch.setattr(
        server, "wait_for_llama_server", _async_return(False)
    )
    monkeypatch.setattr(server, "_probe_backend_reachable", _async_return(True))
    monkeypatch.setattr(
        server, "_kill_process_group", lambda p, logger, **kw: killed.append(p) or True
    )

    recovered = await server._attempt_router_self_heal()

    assert recovered is True, "a reachable late-ready spawn must count as recovery"
    assert server.llama_process is proc, "the live spawn handle must be adopted"
    assert server.backend_ready is True
    assert killed == [], "an adopted spawn must never be killed"
    assert server.backend_recovery_state["last_failure"] is None


@pytest.mark.asyncio
async def test_self_heal_adopts_by_port_when_only_a_descendant_survived(monkeypatch):
    """AC1: reachable with a dead wrapper handle -> adopt by port sentinel."""
    proc = _FakeProc()
    proc.returncode = 0  # the start-llama.sh wrapper exited; the backend did not

    monkeypatch.setattr(server, "start_llama_server", lambda _model: proc)
    monkeypatch.setattr(server, "wait_for_llama_server", _async_return(False))
    monkeypatch.setattr(server, "_probe_backend_reachable", _async_return(True))

    recovered = await server._attempt_router_self_heal()

    assert recovered is True
    adopted = server.llama_process
    assert adopted is not None and adopted is not proc
    assert getattr(adopted, "adopted", False) is True, (
        "a surviving descendant must be represented by an adopted-port sentinel"
    )
    assert int(getattr(adopted, "port", 0)) == 8080
    assert server.backend_ready is True


@pytest.mark.asyncio
async def test_self_heal_terminates_spawn_that_never_becomes_reachable(monkeypatch):
    """AC1: an unreachable spawn is terminated so the port is free for the retry."""
    proc = _FakeProc()
    killed: list = []

    monkeypatch.setattr(server, "start_llama_server", lambda _model: proc)
    monkeypatch.setattr(server, "wait_for_llama_server", _async_return(False))
    monkeypatch.setattr(server, "_probe_backend_reachable", _async_return(False))
    monkeypatch.setattr(
        server, "_kill_process_group", lambda p, logger, **kw: killed.append(p) or True
    )

    recovered = await server._attempt_router_self_heal()

    assert recovered is False
    assert killed == [proc], (
        "the unadopted spawn must be terminated — leaving it alive is what "
        "orphaned port 8080 on 2026-11-02"
    )
    assert server.llama_process is None
    assert server.backend_ready is False


@pytest.mark.asyncio
async def test_self_heal_still_recovers_when_readiness_succeeds(monkeypatch):
    """AC1 guard: the normal happy path is unchanged."""
    proc = _FakeProc()
    killed: list = []

    monkeypatch.setattr(server, "start_llama_server", lambda _model: proc)
    monkeypatch.setattr(server, "wait_for_llama_server", _async_return(True))
    monkeypatch.setattr(
        server, "_kill_process_group", lambda p, logger, **kw: killed.append(p) or True
    )

    assert await server._attempt_router_self_heal() is True
    assert server.llama_process is proc
    assert killed == []


@pytest.mark.asyncio
async def test_self_heal_terminates_spawn_when_readiness_wait_raises(monkeypatch):
    """AC1: an exceptional attempt failure still settles the live spawn."""
    proc = _FakeProc()
    killed: list = []

    async def _raising_wait(_timeout):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(server, "start_llama_server", lambda _model: proc)
    monkeypatch.setattr(server, "wait_for_llama_server", _raising_wait)
    monkeypatch.setattr(server, "_probe_backend_reachable", _async_return(False))
    monkeypatch.setattr(
        server, "_kill_process_group", lambda p, logger, **kw: killed.append(p) or True
    )

    recovered = await server._attempt_router_self_heal()

    assert recovered is False
    assert killed == [proc], "the live spawn must be settled on the failure path too"
    assert server.llama_process is None


@pytest.mark.asyncio
async def test_watchdog_adopts_reachable_unmanaged_backend(monkeypatch):
    """AC2: the watchdog adopts a reachable backend instead of only flipping ready."""
    import asyncio as _asyncio

    monkeypatch.setattr(
        server,
        "config",
        _self_heal_config(llama_watchdog_interval_seconds=0.01),
    )
    monkeypatch.setattr(server, "_probe_backend_reachable", _async_return(True))
    monkeypatch.setattr(server, "_record_backend_signal", lambda *_a: None)

    task = _asyncio.create_task(server._backend_watchdog_loop())
    await _asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except _asyncio.CancelledError:
        pass

    adopted = server.llama_process
    assert adopted is not None, "the watchdog must adopt the reachable backend"
    assert getattr(adopted, "adopted", False) is True
    assert int(getattr(adopted, "port", 0)) == 8080
    assert server.backend_ready is True


@pytest.mark.asyncio
async def test_watchdog_adopts_reachable_backend_after_process_exit(monkeypatch):
    """AC2: a crashed handle with a reachable backend is adopted, not dropped."""
    import asyncio as _asyncio

    crashed = _FakeProc()
    crashed.returncode = 137
    monkeypatch.setattr(server, "llama_process", crashed)
    monkeypatch.setattr(
        server,
        "config",
        _self_heal_config(llama_watchdog_interval_seconds=0.01),
    )
    monkeypatch.setattr(server, "_probe_backend_reachable", _async_return(True))
    monkeypatch.setattr(server, "_record_backend_signal", lambda *_a: None)
    monkeypatch.setattr(
        server, "_worker_process_unhealthy", lambda _proc: False
    )

    task = _asyncio.create_task(server._backend_watchdog_loop())
    await _asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except _asyncio.CancelledError:
        pass

    adopted = server.llama_process
    assert adopted is not None, "the crashed handle must be replaced by an adoption"
    assert getattr(adopted, "adopted", False) is True
    assert server.backend_ready is True


# ---------------------------------------------------------------------------
# AC2 — /health reflects a reachable, unmanaged backend
# ---------------------------------------------------------------------------


def _stub_router_models(monkeypatch):
    monkeypatch.setattr(
        server, "router_list_models", _async_return({"data": []})
    )


@pytest.mark.asyncio
async def test_health_reports_running_for_reachable_unmanaged_backend(monkeypatch):
    """AC2: reachable backend + ``llama_process is None`` -> running/reachable."""
    _stub_router_models(monkeypatch)
    monkeypatch.setattr(server, "_probe_backend_reachable", _async_return(True))
    monkeypatch.setattr(server, "backend_ready", True)

    health = await server.health_check()

    assert health["llama_server_running"] is True
    assert health["backend_reachable"] is True
    assert health["llama_server_managed"] is False, (
        "an unmanaged-but-reachable backend must stay distinguishable"
    )


@pytest.mark.asyncio
async def test_health_reports_not_running_for_unreachable_unmanaged_backend(
    monkeypatch,
):
    """AC2 guard: no false 'running' when nothing answers on the port."""
    _stub_router_models(monkeypatch)
    monkeypatch.setattr(server, "_probe_backend_reachable", _async_return(False))

    health = await server.health_check()

    assert health["llama_server_running"] is False
    assert health["backend_reachable"] is False
    assert health["llama_server_managed"] is False
    assert health["ready"] is False


@pytest.mark.asyncio
async def test_health_reports_managed_backend_as_before(monkeypatch):
    """AC2 guard: a live process handle is still reported as managed/running."""
    _stub_router_models(monkeypatch)
    monkeypatch.setattr(server, "llama_process", _FakeProc())
    monkeypatch.setattr(server, "_probe_backend_reachable", _async_return(True))
    monkeypatch.setattr(server, "backend_ready", True)

    health = await server.health_check()

    assert health["llama_server_running"] is True
    assert health["backend_reachable"] is True
    assert health["llama_server_managed"] is True


# ---------------------------------------------------------------------------
# AC3 — /llama/local/status reports capacity for that backend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_status_reports_nonzero_total_slots_for_reachable_unmanaged_backend(
    monkeypatch, transport
):
    """AC3: a reachable router-mode backend must not report ``total_slots: 0``."""
    monkeypatch.setattr(server, "_probe_backend_reachable", _async_return(True))
    monkeypatch.setattr(server, "backend_ready", True)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/llama/local/status")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["llama_server_running"] is True
    assert payload["total_slots"] > 0, (
        "total_slots: 0 while the backend answers is what blocked the slot sweep"
    )


# ---------------------------------------------------------------------------
# AC4 — a port-bind conflict names the conflicting listener
# ---------------------------------------------------------------------------


class _DummySrv:
    """Minimal server stand-in for ``start_llama_server``."""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("test_self_heal_orphan_adoption")
        self.log_dir = None
        self.llama_log_file = None
        self.last_start_failure = None
        self.llama_log_source_unhealthy = False

    def rotate_llama_logs(self, *a, **kw):
        pass

    def broadcast_status_sync(self, *a, **kw):
        pass


def test_bind_conflict_names_the_conflicting_listener_pid(monkeypatch, caplog):
    """AC4: the bind-conflict log line carries the port holder's pid."""
    dummy = _DummySrv({"server": {}})
    monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        lifecycle,
        "spawn_and_capture",
        lambda *a, **kw: (
            None,
            "start: couldn't bind HTTP server socket, hostname: 0.0.0.0, port: 8080",
        ),
    )
    monkeypatch.setattr(lifecycle, "_find_pid_on_port", lambda port: 4242)

    with caplog.at_level(logging.ERROR):
        proc = lifecycle.start_llama_server(None)

    assert proc is None
    bind_lines = [r.getMessage() for r in caplog.records if "4242" in r.getMessage()]
    assert bind_lines, (
        "a bind conflict must name the conflicting listener pid; got: "
        f"{[r.getMessage() for r in caplog.records]}"
    )
    assert any("could not bind" in msg.lower() for msg in bind_lines)


def test_non_bind_failure_does_not_probe_the_port(monkeypatch, caplog):
    """AC4 guard: only bind conflicts pay for the port-holder lookup."""
    dummy = _DummySrv({"server": {}})
    probed: list = []
    monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        lifecycle, "spawn_and_capture", lambda *a, **kw: (None, "boom: bad config")
    )
    monkeypatch.setattr(
        lifecycle, "_find_pid_on_port", lambda port: probed.append(port) or None
    )

    with caplog.at_level(logging.ERROR):
        assert lifecycle.start_llama_server(None) is None

    assert probed == []


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------


def _async_return(value):
    async def _inner(*_a, **_kw):
        return value

    return _inner
