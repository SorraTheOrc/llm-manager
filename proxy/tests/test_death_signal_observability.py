"""Watchdog wiring for the true death signal (LP-0MW3K7WD6003TJKZ).

``decode_wait_status``/``decode_returncode`` name the cause; these tests pin
that the **watchdogs actually use it**:

* **AC1** — a llama-server death from a synthetic ``SIGSEGV`` wait status is
  logged as ``SIGSEGV(11)/139`` and stored with
  ``restart_reason=llama_server_signal_SIGSEGV`` — never the blanket
  ``exit code=137`` the operator saw 133 times;
* **AC3** — a TTS death carries its ``restart_reason`` in the watchdog log and
  in ``tts_recovery_state`` (previously only ``code=-9``), and the reason is
  handed to the self-heal path that logs and stores it.
"""

from __future__ import annotations

import asyncio
import signal
from unittest.mock import AsyncMock

import proxy.backend_health as backend_health
import proxy.death_capture as dc
import proxy.server as server
import pytest
from proxy.backends import tts as tts_backend


class _CaptureLogger:
    """Logger stand-in that keeps the formatted lines for assertions."""

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


class _FakeProc:
    """A ``Popen``-shaped handle whose real status is irrelevant here."""

    pid = None

    def __init__(self, returncode=137) -> None:
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def terminate(self):  # pragma: no cover - only on the unhealthy-worker path
        self.returncode = -signal.SIGTERM


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
def llama_env(monkeypatch, tmp_path):
    """Router-mode llama watchdog state with a dead process handle."""
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "current_model", "qwen-local")
    monkeypatch.setattr(server, "llama_process", _FakeProc())
    monkeypatch.setattr(server, "log_dir", tmp_path)
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
            "last_restart_reason": None,
        },
    )
    monkeypatch.setattr(server, "_record_backend_signal", lambda *a: None)
    monkeypatch.setattr(
        server,
        "config",
        {
            "server": {
                "llama_router_mode": True,
                "llama_watchdog_interval_seconds": 0.02,
                "llama_server_port": 8080,
            }
        },
    )
    return server


async def _run_loop(loop_factory, seconds: float = 0.2) -> None:
    task = asyncio.create_task(loop_factory())
    await asyncio.sleep(seconds)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# AC1 — llama-server watchdog reports the true signal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llama_watchdog_reports_sigsegv_not_blanket_137(
    llama_env, logger, monkeypatch
):
    """AC1: a synthetic SIGSEGV status is logged and stored as SIGSEGV/139."""
    sigsegv = dc.decode_wait_status(0x80 | signal.SIGSEGV)
    monkeypatch.setattr(dc, "peek_death_status", lambda pid: sigsegv)
    monkeypatch.setattr(
        server, "_probe_backend_reachable", AsyncMock(return_value=False)
    )
    heal = AsyncMock(return_value=True)
    monkeypatch.setattr(server, "_attempt_router_self_heal", heal)

    await _run_loop(server._backend_watchdog_loop)

    death_lines = [line for line in logger.lines if "llama-server death" in line]
    assert death_lines, f"no death logged; got {logger.lines}"
    assert "SIGSEGV(11)/139" in death_lines[0]
    assert "restart_reason=llama_server_signal_SIGSEGV" in death_lines[0]
    assert not any("exit code=137" in line for line in logger.lines)

    # Stored, not just logged: the recovery state carries the decoded cause.
    stored = llama_env.backend_recovery_state["last_death"]
    assert stored["restart_reason"] == "llama_server_signal_SIGSEGV"
    assert stored["death"]["signal_name"] == "SIGSEGV"
    assert stored["death"]["code"] == 139
    assert stored["kind"] == "death_window"

    # ... and the self-heal is triggered with the decoded reason.
    reasons = [call.kwargs.get("reason") for call in heal.await_args_list]
    assert "llama_server_signal_SIGSEGV" in reasons


@pytest.mark.asyncio
async def test_llama_watchdog_reports_a_normal_exit_distinctly(
    llama_env, logger, monkeypatch
):
    """AC2: a normal exit is named as an exit code, not as a signal."""
    monkeypatch.setattr(
        dc, "peek_death_status", lambda pid: dc.decode_wait_status(1 << 8)
    )
    monkeypatch.setattr(
        server, "_probe_backend_reachable", AsyncMock(return_value=False)
    )
    heal = AsyncMock(return_value=True)
    monkeypatch.setattr(server, "_attempt_router_self_heal", heal)

    await _run_loop(server._backend_watchdog_loop)

    death_lines = [line for line in logger.lines if "llama-server death" in line]
    assert death_lines, f"no death logged; got {logger.lines}"
    assert "exit code 1" in death_lines[0]
    assert "restart_reason=llama_server_exit_code_1" in death_lines[0]
    assert (
        llama_env.backend_recovery_state["last_death"]["restart_reason"]
        == "llama_server_exit_code_1"
    )


# ---------------------------------------------------------------------------
# AC3 — TTS deaths carry a restart_reason
# ---------------------------------------------------------------------------


@pytest.fixture
def tts_env(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "tts_process", _FakeProc())
    monkeypatch.setattr(server, "log_dir", tmp_path)
    monkeypatch.setattr(
        server,
        "tts_recovery_state",
        {
            "in_progress": False,
            "attempt_timestamps": [],
            "max_attempts": 3,
            "window_seconds": 120,
            "last_failure": None,
        },
    )
    monkeypatch.setattr(
        server,
        "config",
        {
            "server": {
                "tts_enabled": True,
                "tts_server_port": 8081,
                "tts_self_heal_interval_seconds": 0.02,
                "tts_self_heal_max_attempts": 3,
                "tts_self_heal_window_seconds": 120,
            }
        },
    )
    return server


@pytest.mark.asyncio
async def test_tts_watchdog_reports_sigkill_and_passes_restart_reason(
    tts_env, logger, monkeypatch
):
    """AC3: ``code=-9`` becomes ``SIGKILL(9)/137`` + a stored restart_reason."""
    monkeypatch.setattr(
        dc, "peek_death_status", lambda pid: dc.decode_wait_status(signal.SIGKILL)
    )
    heal = AsyncMock(return_value=True)
    monkeypatch.setattr(backend_health, "_attempt_tts_self_heal", heal)

    await _run_loop(backend_health._tts_watchdog_loop)

    death_lines = [line for line in logger.lines if "TTS server died" in line]
    assert death_lines, f"no TTS death logged; got {logger.lines}"
    assert "SIGKILL(9)/137" in death_lines[0]
    assert "restart_reason=tts_server_signal_SIGKILL" in death_lines[0]
    assert not any("exited with code=-9" in line for line in logger.lines)

    assert heal.await_args_list
    assert heal.await_args.kwargs["restart_reason"] == "tts_server_signal_SIGKILL"

    stored = tts_env.tts_recovery_state["last_death"]
    assert stored["restart_reason"] == "tts_server_signal_SIGKILL"
    assert stored["death"]["signal_name"] == "SIGKILL"


@pytest.mark.asyncio
async def test_tts_self_heal_logs_and_stores_its_restart_reason(
    tts_env, logger, monkeypatch
):
    """AC3: the self-heal path records the reason it was given."""
    monkeypatch.setattr(
        "proxy.lifecycle.start_tts_server", lambda: _FakeProc(returncode=None)
    )
    monkeypatch.setattr(
        "proxy.lifecycle.wait_for_tts_server", AsyncMock(return_value=True)
    )

    healed = await tts_backend._attempt_tts_self_heal(
        restart_reason="tts_server_signal_SIGKILL"
    )

    assert healed is True
    assert (
        tts_env.tts_recovery_state["last_restart_reason"]
        == "tts_server_signal_SIGKILL"
    )
    attempt_lines = [line for line in logger.lines if "TTS self-heal attempt" in line]
    assert attempt_lines
    assert "restart_reason=tts_server_signal_SIGKILL" in attempt_lines[0]


@pytest.mark.asyncio
async def test_tts_self_heal_default_reason_is_explicit(tts_env, logger, monkeypatch):
    """A restart with no decoded cause is labelled, never left anonymous."""
    monkeypatch.setattr(
        "proxy.lifecycle.start_tts_server", lambda: _FakeProc(returncode=None)
    )
    monkeypatch.setattr(
        "proxy.lifecycle.wait_for_tts_server", AsyncMock(return_value=True)
    )

    await tts_backend._attempt_tts_self_heal()

    assert tts_env.tts_recovery_state["last_restart_reason"] == "unspecified"
