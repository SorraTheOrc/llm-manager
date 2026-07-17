"""Tests for LP-0MRCQW0HC000J4F9: Proxy backend_ready flag recovery.

When the backend (llama-server) is alive and reachable but backend_ready has
been set to False (e.g., by stop_llama_server or a transient failure), the
watchdog loop must detect reachability and reset backend_ready to True
without requiring a full process restart.
"""

import asyncio

import pytest

import proxy.server as server

pytestmark = pytest.mark.refactor_parity


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Reset backend state before each test."""
    monkeypatch.setattr(server, "backend_ready", False)
    monkeypatch.setattr(server, "llama_process", None)
    monkeypatch.setattr(server, "current_model", None)
    monkeypatch.setattr(
        server,
        "backend_recovery_state",
        {
            "in_progress": False,
            "attempt_timestamps": [],
            "max_attempts": 3,
            "window_seconds": 300,
            "retry_after_seconds": 30,
            "last_failure": None,
        },
    )
    monkeypatch.setattr(
        server,
        "config",
        {
            "server": {
                "llama_router_mode": True,
                "llama_watchdog_interval_seconds": 0.01,
                "llama_server_port": 8080,
                "llama_self_heal_max_attempts": 3,
                "llama_self_heal_window_seconds": 300,
                "llama_self_heal_backoff_base_seconds": 1,
                "llama_self_heal_retry_after_seconds": 30,
                "llama_startup_timeout": 5,
            }
        },
    )
    monkeypatch.setattr(server, "logger", type("L", (), {
        "error": lambda *a, **kw: None,
        "warning": lambda *a, **kw: None,
        "info": lambda *a, **kw: None,
        "debug": lambda *a, **kw: None,
        "exception": lambda *a, **kw: None,
    })())
    monkeypatch.setattr(server, "_record_backend_signal", lambda *a: None)


@pytest.mark.asyncio
async def test_backend_ready_recovers_from_stale_false_when_proc_none_and_backend_reachable(monkeypatch):
    """AC 1 & 5: When llama_process is None and backend_ready is False,
    if the backend is actually reachable on its port, the watchdog should
    set backend_ready to True without attempting a restart."""
    restart_attempts = []
    probe_results = []  # Results for successive probe calls

    def fake_start(model=None):
        restart_attempts.append(model)
        return None  # Simulates podman start failure

    async def fake_wait(timeout):
        return False

    async def fake_probe_backend_reachable(port):
        """Return a result from the probe_results list, defaulting to True."""
        if probe_results:
            return probe_results.pop(0)
        return True  # Default: backend is reachable

    monkeypatch.setattr(server, "start_llama_server", fake_start)
    monkeypatch.setattr(server, "wait_for_llama_server", fake_wait)
    monkeypatch.setattr(server, "_probe_backend_reachable", fake_probe_backend_reachable)

    # Run watchdog for a few cycles
    task = asyncio.create_task(server._backend_watchdog_loop())
    await asyncio.sleep(0.2)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Watchdog should NOT have attempted a restart because probe succeeded
    assert len(restart_attempts) == 0, (
        f"Expected 0 restart attempts when backend is reachable, "
        f"got {len(restart_attempts)}"
    )
    # backend_ready should have been set to True
    assert server.backend_ready is True, (
        "backend_ready should be True when backend is reachable"
    )


@pytest.mark.asyncio
async def test_watchdog_probes_before_restart_when_proc_none(monkeypatch):
    """AC 2: Watchdog should probe the backend for reachability before
    attempting a full restart when llama_process is None in router mode."""
    call_order = []
    probe_results = [True]  # Backend is reachable

    async def fake_probe_backend_reachable(port):
        call_order.append("probe")
        if probe_results:
            return probe_results.pop(0)
        return False

    def fake_start(model=None):
        call_order.append("start")
        return None

    async def fake_wait(timeout):
        return False

    monkeypatch.setattr(server, "_probe_backend_reachable", fake_probe_backend_reachable)
    monkeypatch.setattr(server, "start_llama_server", fake_start)
    monkeypatch.setattr(server, "wait_for_llama_server", fake_wait)

    task = asyncio.create_task(server._backend_watchdog_loop())
    await asyncio.sleep(0.2)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Probe should have been called before any restart
    assert "probe" in call_order, "Watchdog should probe backend reachability"
    # No restart should have been attempted since probe succeeded
    assert "start" not in call_order, (
        "Watchdog should not restart when backend is reachable"
    )


@pytest.mark.asyncio
async def test_watchdog_restarts_when_proc_none_and_backend_unreachable(monkeypatch):
    """AC 2 (fallback): When backend is NOT reachable, watchdog should
    fall through to the existing restart logic."""
    restart_attempts = []

    def fake_start(model=None):
        restart_attempts.append(model)
        # Return a fake process so the heal can succeed
        return type("P", (), {"poll": lambda self: None, "pid": 999})()

    async def fake_wait(timeout):
        return True

    async def fake_probe_backend_reachable(port):
        return False  # Backend is not reachable

    monkeypatch.setattr(server, "start_llama_server", fake_start)
    monkeypatch.setattr(server, "wait_for_llama_server", fake_wait)
    monkeypatch.setattr(server, "_probe_backend_reachable", fake_probe_backend_reachable)

    task = asyncio.create_task(server._backend_watchdog_loop())
    await asyncio.sleep(0.2)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(restart_attempts) >= 1, (
        f"Expected at least 1 restart when backend is unreachable, "
        f"got {len(restart_attempts)}"
    )


@pytest.mark.asyncio
async def test_backend_ready_recovers_from_stale_false_with_process_exited(monkeypatch):
    """AC 3: When llama_process has exited but backend is reachable,
    watchdog should set backend_ready to True without restart."""
    restart_attempts = []

    class ExitedProc:
        def poll(self):
            return 1  # Process has exited

    async def fake_probe_backend_reachable(port):
        return True  # Backend still reachable

    def fake_start(model=None):
        restart_attempts.append(model)
        return type("P", (), {"poll": lambda self: None, "pid": 999})()

    async def fake_wait(timeout):
        return True

    monkeypatch.setattr(server, "llama_process", ExitedProc())
    monkeypatch.setattr(server, "backend_ready", False)
    monkeypatch.setattr(server, "current_model", "qwen3")
    monkeypatch.setattr(server, "_probe_backend_reachable", fake_probe_backend_reachable)
    monkeypatch.setattr(server, "start_llama_server", fake_start)
    monkeypatch.setattr(server, "wait_for_llama_server", fake_wait)

    task = asyncio.create_task(server._backend_watchdog_loop())
    await asyncio.sleep(0.2)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Should have probed the backend and set backend_ready to True
    assert server.backend_ready is True, (
        "backend_ready should be True when backend is reachable "
        "even if llama_process exited"
    )
    # Should NOT have attempted a restart since backend is reachable
    assert len(restart_attempts) == 0, (
        f"Expected 0 restarts when backend reachable but process exited, "
        f"got {len(restart_attempts)}"
    )


@pytest.mark.asyncio
async def test_no_excessive_log_spam_when_healthy(monkeypatch):
    """AC 6: When backend is reachable and healthy, recovery should produce
    no more than one log line per watch cycle."""
    log_lines = []

    class CountingLogger:
        def error(self, *a, **kw):
            log_lines.append(("error", a, kw))

        def warning(self, *a, **kw):
            log_lines.append(("warning", a, kw))

        def info(self, *a, **kw):
            log_lines.append(("info", a, kw))

        def debug(self, *a, **kw):
            log_lines.append(("debug", a, kw))

        def exception(self, *a, **kw):
            log_lines.append(("exception", a, kw))

    monkeypatch.setattr(server, "logger", CountingLogger())
    monkeypatch.setattr(server, "backend_ready", False)
    monkeypatch.setattr(server, "llama_process", None)
    monkeypatch.setattr(server, "current_model", None)

    async def fake_probe_backend_reachable(port):
        return True

    monkeypatch.setattr(server, "_probe_backend_reachable", fake_probe_backend_reachable)

    task = asyncio.create_task(server._backend_watchdog_loop())
    await asyncio.sleep(0.2)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Count the relevant log lines (info about recovery, not asyncio noise)
    recovery_logs = [line for line in log_lines if "recover" in str(line).lower() or "ready" in str(line).lower()]

    # There should be at most 1-2 log lines per cycle for recovery
    assert len(recovery_logs) <= 3, (
        f"Expected at most 3 recovery-related log lines, got {len(recovery_logs)}"
    )


@pytest.mark.asyncio
async def test_watchdog_probe_skipped_in_non_router_mode(monkeypatch):
    """In non-router mode with proc=None and not backend_ready, watchdog
    should skip the probe (since there's no independent backend to check)."""
    monkeypatch.setattr(
        server,
        "config",
        {
            "server": {
                "llama_router_mode": False,
                "llama_watchdog_interval_seconds": 0.01,
                "llama_server_port": 8080,
            }
        },
    )

    probe_calls = []

    async def fake_probe_backend_reachable(port):
        probe_calls.append(port)
        return True

    monkeypatch.setattr(server, "_probe_backend_reachable", fake_probe_backend_reachable)
    monkeypatch.setattr(server, "backend_ready", False)
    monkeypatch.setattr(server, "llama_process", None)

    task = asyncio.create_task(server._backend_watchdog_loop())
    await asyncio.sleep(0.2)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Probe should NOT have been called because router_mode is False
    assert len(probe_calls) == 0, (
        f"Expected 0 probe calls in non-router mode, got {len(probe_calls)}"
    )
    # backend_ready should remain False
    assert server.backend_ready is False


@pytest.mark.asyncio
async def test_health_endpoint_ready_after_watchdog_recovery(monkeypatch):
    """AC 4: After watchdog recovers backend_ready, /health should
    return ready:true."""
    class Proc:
        def poll(self):
            return None

    async def fake_probe(_port):
        return True

    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "llama_process", Proc())
    monkeypatch.setattr(server, "current_model", "qwen3")
    monkeypatch.setattr(server, "config", {"server": {"llama_router_mode": True, "llama_server_port": 8080}})
    monkeypatch.setattr(server, "_probe_backend_reachable", fake_probe)

    health = await server.health_check()

    assert health["ready"] is True
    assert health["status"] in ("healthy", "degraded")
