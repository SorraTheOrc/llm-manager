"""Adopted-backend death detection (LP-0MW3MOMD9007DC26).

When the proxy adopts an already-running llama-server it installs the
``_AdoptedLlamaProcess`` sentinel (``pid=0``, ``poll()`` always ``None``) as
``llama_process`` — by design it can never trip the watchdog's
poll()-based death checks. The 2026-11-05 observations (35-minute silent
local-route outage; ``/health`` reporting ``llama_server_running: true`` while
``backend_reachable: false``; a second 6.5 h outage at 21:36 with
``backend_recovery.attempt_count=0`` despite 600 connect failures) pin the
failure: an adopted backend's death is invisible to the watchdog **and** to
the health consumers.

These tests pin the fix:

* **AC1** — an adopted backend's death is detected within one watchdog
  interval, with a logged death line carrying the true signal (regression
  test with a fake adopted handle whose process has exited);
* **AC2** — ``/health`` never reports ``llama_server_running: true`` while
  ``backend_reachable`` is false in this state;
* **AC3** — the adopted-backend death → self-heal path fires with no ``Popen``
  handle available.
"""

from __future__ import annotations

import asyncio
import subprocess
from unittest.mock import AsyncMock

import proxy.lifecycle as lifecycle
import proxy.server as server
import pytest

from proxy import death_capture as dc


class _CaptureLogger:
    """Logger stand-in that keeps formatted lines for assertions."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def _add(self, msg, *args) -> None:
        try:
            self.lines.append(msg % args if args else str(msg))
        except Exception:  # pragma: no cover - defensive
            self.lines.append(str(msg))

    def error(self, msg, *args, **_kwargs):
        self._add(msg, *args)

    def warning(self, msg, *args, **_kwargs):
        self._add(msg, *args)

    def info(self, msg, *args, **_kwargs):
        self._add(msg, *args)

    def debug(self, msg, *args, **_kwargs):
        self._add(msg, *args)

    def exception(self, msg, *args, **_kwargs):  # pragma: no cover - failure path
        self._add(msg, *args)


@pytest.fixture(autouse=True)
def _reset_capture():
    dc.reset_death_capture()
    yield
    dc.reset_death_capture()


@pytest.fixture
def logger(monkeypatch) -> _CaptureLogger:
    capture_logger = _CaptureLogger()
    monkeypatch.setattr(server, "logger", capture_logger)
    return capture_logger


@pytest.fixture
def adopted_env(monkeypatch, tmp_path):
    """Router-mode adopted-backend state: sentinel installed, ready True."""
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "current_model", "qwen-local")
    monkeypatch.setattr(
        server, "llama_process", lifecycle._AdoptedLlamaProcess(port=8080)
    )
    monkeypatch.setattr(server, "log_dir", tmp_path)
    monkeypatch.setattr(
        server,
        "backend_recovery_state",
        {
            "in_progress": False,
            "attempt_timestamps": [],
            "max_attempts": 3,
            "window_seconds": 300,
            "last_failure": None,
            "last_restart_reason": None,
        },
    )
    monkeypatch.setattr(server, "_record_backend_signal", lambda *a: None)
    monkeypatch.setattr(server, "_probe_backend_reachable", AsyncMock(return_value=False))
    monkeypatch.setattr(
        server,
        "config",
        {
            "server": {
                "llama_router_mode": True,
                "llama_watchdog_interval_seconds": 0.01,
                "llama_server_port": 8080,
                "llama_adopted_unreachable_probes": 1,
            }
        },
    )
    heal = AsyncMock(return_value=True)
    monkeypatch.setattr(server, "_attempt_router_self_heal", heal)
    # Hermetic: the OOM ranker and isolation guard are real modules that must
    # not touch live processes (or the capture logger) inside tests.
    import proxy.backend_isolation as _backend_isolation
    import proxy.lifecycle as _lifecycle

    monkeypatch.setattr(_lifecycle, "_reapply_backend_oom_protection", lambda *a, **k: None)
    monkeypatch.setattr(_backend_isolation, "assert_backend_isolation", lambda *a, **k: None)
    return server


async def _run_loop(loop_factory, seconds: float = 0.15) -> None:
    task = asyncio.create_task(loop_factory())
    await asyncio.sleep(seconds)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# AC3 — adopted death (pid-0 sentinel) detected via reachability, self-heals
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watchdog_declares_adopted_backend_dead_and_self_heals(
    adopted_env, logger, monkeypatch
):
    """AC3: no ``Popen`` available → reachability makes the death visible and
    the router self-heal fires with a distinct ``restart_reason``."""
    await _run_loop(server._backend_watchdog_loop)

    death_lines = [line for line in logger.lines if "adopted llama-server death" in line]
    assert (
        death_lines
    ), f"no adopted death logged; got {logger.lines}"
    assert "restart_reason=llama_server_adopted_unreachable" in death_lines[0]

    # The sentinel is cleared so every downstream poll()-based consumer
    # (ui.py, observability, local_route_starvation) stops reporting running.
    assert server.llama_process is None
    assert server.backend_ready is False
    assert server.current_model is None

    # The self-heal path fires with the adopted reason (AC3).
    reasons = [call.kwargs.get("reason") for call in adopted_env._attempt_router_self_heal.await_args_list]
    assert "llama_server_adopted_unreachable" in reasons

    # The stored record carries the same reason (observability for audits).
    assert (
        adopted_env.backend_recovery_state["last_restart_reason"]
        == "llama_server_adopted_unreachable"
    )
    assert (
        adopted_env.backend_recovery_state["last_death"]["extra"]["restart_reason"]
        == "llama_server_adopted_unreachable"
    )


@pytest.mark.asyncio
async def test_watchdog_keeps_adopted_backend_ready_while_reachable(monkeypatch, logger):
    """The reachability probe must not restart a healthy adopted backend."""
    monkeypatch.setattr(server, "backend_ready", False)
    monkeypatch.setattr(
        server, "llama_process", lifecycle._AdoptedLlamaProcess(port=8080)
    )
    monkeypatch.setattr(
        server, "backend_recovery_state",
        {
            "in_progress": False,
            "attempt_timestamps": [],
            "max_attempts": 3,
            "window_seconds": 300,
            "last_failure": None,
            "last_restart_reason": None,
        },
    )
    monkeypatch.setattr(server, "_record_backend_signal", lambda *a: None)
    monkeypatch.setattr(server, "_probe_backend_reachable", AsyncMock(return_value=True))
    monkeypatch.setattr(server, "_attempt_router_self_heal", AsyncMock(return_value=True))
    monkeypatch.setattr(
        server,
        "config",
        {
            "server": {
                "llama_router_mode": True,
                "llama_watchdog_interval_seconds": 0.01,
                "llama_server_port": 8080,
            }
        },
    )

    await _run_loop(server._backend_watchdog_loop)

    assert server.backend_ready is True, "reachable adopted backend must recover readiness"
    assert server.llama_process is not None, "adopted backend must not be cleared"
    assert not any("adopted llama-server death" in line for line in logger.lines), (
        f"no death expected; got {logger.lines}"
    )


# ---------------------------------------------------------------------------
# AC1 — the death line carries the true signal when the handle can decode it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adopted_death_line_carries_true_signal_for_exited_handle(
    adopted_env, logger, monkeypatch
):
    """AC1: a fake adopted handle whose underlying process has exited yields
    the true signal (SIGKILL via ``peek_death_status``), not a blanket 137."""
    child = subprocess.Popen(["sleep", "60"])
    try:
        child.kill()  # SIGKILL; keep un-reaped so waitid(WNOWAIT) decodes it
        # The sentinel carries the real pid, so probe_death can decode it.
        monkeypatch.setattr(
            server, "llama_process", lifecycle._AdoptedLlamaProcess(pid=child.pid, port=8080)
        )
        # The adopted backend is gone from the port too.
        monkeypatch.setattr(
            server, "_probe_backend_reachable", AsyncMock(return_value=False)
        )

        await _run_loop(server._backend_watchdog_loop)

        # The watchdog logs a death line that names the signal.
        death_lines = [line for line in logger.lines if "llama-server death" in line]
        assert death_lines, f"no death logged; got {logger.lines}"
        assert "SIGKILL(9)/137" in death_lines[0], death_lines[0]
        assert "restart_reason=llama_server_signal_SIGKILL" in death_lines[0]
        assert server.llama_process is None
    finally:
        try:
            child.wait(timeout=5)
        except Exception:  # pragma: no cover - cleanup must not hide the result
            pass
        try:
            child.kill()
        except ProcessLookupError:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# AC2 — /health never reports running:true while backend_unreachable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_reports_not_running_when_adopted_backend_unreachable(
    adopted_env, monkeypatch
):
    """AC2: the sentinel alone must not make ``llama_server_running`` true."""
    monkeypatch.setattr(
        server, "router_list_models", AsyncMock(return_value={"data": []})
    )
    health = await server.health_check()

    assert health["llama_server_running"] is False, (
        f"adopted-but-unreachable backend must report not running; got {health}"
    )
    assert health["backend_reachable"] is False
    assert health["ready"] is False


@pytest.mark.asyncio
async def test_health_reports_running_when_adopted_backend_reachable(
    adopted_env, monkeypatch
):
    """AC2 guard: a reachable adopted backend still reports running."""
    monkeypatch.setattr(
        server, "router_list_models", AsyncMock(return_value={"data": []})
    )
    monkeypatch.setattr(
        server, "_probe_backend_reachable", AsyncMock(return_value=True)
    )
    health = await server.health_check()

    assert health["llama_server_running"] is True
    assert health["backend_reachable"] is True


# ---------------------------------------------------------------------------
# /llama/local/status also derives running from the sentinel (AC2 companion):
# ``query_llama_status`` must probe reachability for an adopted handle instead
# of trusting the pid-0 sentinel's perpetual ``poll() is None``.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_llama_status_probes_adopted_backend_before_reporting_running(
    adopted_env, monkeypatch
):
    """An adopted backend (sentinel only) must be probed, not trusted."""
    from proxy.observability import query_llama_status

    probe = AsyncMock(return_value=False)
    monkeypatch.setattr(server, "_probe_backend_reachable", probe)
    status = await query_llama_status()

    assert status["llama_server_running"] is False, (
        f"adopted-but-unreachable backend must not report running; got {status}"
    )
    probe.assert_awaited()


@pytest.mark.asyncio
async def test_query_llama_status_reachable_adopted_backend_reports_running(
    adopted_env, monkeypatch
):
    """An adopted backend that still answers the port reports running."""
    from proxy.observability import query_llama_status

    monkeypatch.setattr(
        server, "_probe_backend_reachable", AsyncMock(return_value=True)
    )
    status = await query_llama_status()

    assert status["llama_server_running"] is True
    assert status["router_mode"] is True
