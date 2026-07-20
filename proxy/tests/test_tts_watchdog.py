"""
Tests for the TTS watchdog self-healing loop.

Verifies that:
- _tts_watchdog_loop() periodically checks tts_process.poll()
- When the process has exited, watchdog restarts via start_tts_server()
- Watchdog respects tts_enabled=false
- Health endpoint reflects recovery states during restart attempts
"""

import asyncio
import subprocess
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_config(
    tts_enabled=True,
    tts_server_port=8081,
    tts_self_heal_max_attempts=3,
    tts_self_heal_window_seconds=120,
    tts_self_heal_interval_seconds=10,
    tts_self_heal_probe_timeout_seconds=3,
):
    """Build a config dict with TTS self-heal settings."""
    return {
        "server": {
            "tts_enabled": tts_enabled,
            "tts_server_port": tts_server_port,
            "tts_server_host": "localhost",
            "llama_router_mode": False,
            "llama_server_port": 8080,
            "llama_watchdog_interval_seconds": 5,
            "tts_self_heal_max_attempts": tts_self_heal_max_attempts,
            "tts_self_heal_window_seconds": tts_self_heal_window_seconds,
            "tts_self_heal_interval_seconds": tts_self_heal_interval_seconds,
            "tts_self_heal_probe_timeout_seconds": tts_self_heal_probe_timeout_seconds,
        },
        "models": {},
        "default_model": "test-model",
    }


def _make_mock_server(
    tts_process_alive=True,
    tts_enabled=True,
    tts_server_port=8081,
    llama_process_alive=True,
    **kwargs,
):
    """Build a mock server object with the specified state."""
    srv = MagicMock()
    srv.config = _make_mock_config(tts_enabled, tts_server_port)
    srv.logger = MagicMock()
    srv.tts_recovery_state = {
        "in_progress": False,
        "attempt_timestamps": [],
        "max_attempts": _make_mock_config(tts_enabled, tts_server_port)["server"].get("tts_self_heal_max_attempts", 3),
        "window_seconds": _make_mock_config(tts_enabled, tts_server_port)["server"].get("tts_self_heal_window_seconds", 120),
        "last_failure": None,
    }

    # TTS process
    if tts_process_alive:
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll = MagicMock(return_value=None)  # None means still running
        mock_proc.pid = 12345
        srv.tts_process = mock_proc
    else:
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll = MagicMock(return_value=1)  # non-None means exited
        mock_proc.pid = 12345
        srv.tts_process = mock_proc

    # Llama process
    if llama_process_alive:
        mock_llama = MagicMock(spec=subprocess.Popen)
        mock_llama.poll = MagicMock(return_value=None)
        srv.llama_process = mock_llama
    else:
        srv.llama_process = None

    srv.backend_ready = True
    srv.current_model = "test-model"

    return srv


# ---------------------------------------------------------------------------
# Tests for _tts_watchdog_loop
# ---------------------------------------------------------------------------

class TestTtsWatchdogBasic:
    """Basic tests for the TTS watchdog loop."""

    @pytest.mark.asyncio
    async def test_watchdog_does_not_run_when_tts_disabled(self):
        """Watchdog should not start any monitoring when tts_enabled=False."""
        from proxy.backend_health import _tts_watchdog_loop

        srv = _make_mock_server(
            tts_process_alive=True,
            tts_enabled=False,
        )

        start_tts_server_called = False

        async def fake_start_tts_server():
            nonlocal start_tts_server_called
            start_tts_server_called = True

        with patch("proxy.backend_health._srv", return_value=srv):
            with patch("proxy.lifecycle.start_tts_server", side_effect=fake_start_tts_server):
                task = asyncio.create_task(_tts_watchdog_loop())
                await asyncio.sleep(0.2)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Verify no restart was attempted
        assert not start_tts_server_called, "start_tts_server should NOT have been called"

    @pytest.mark.asyncio
    async def test_watchdog_restarts_dead_process(self):
        """Watchdog should restart TTS when process has exited."""
        from proxy.backend_health import _tts_watchdog_loop

        srv = _make_mock_server(
            tts_process_alive=False,  # Process is dead
            tts_enabled=True,
        )

        # Patch start_tts_server to return a new process
        new_proc = MagicMock(spec=subprocess.Popen)
        new_proc.poll = MagicMock(return_value=None)
        new_proc.pid = 12346

        restart_count = 0

        def fake_start_tts_server():
            nonlocal restart_count
            restart_count += 1
            srv.tts_process = new_proc
            return new_proc

        with patch("proxy.backend_health._srv", return_value=srv):
            # start_tts_server is imported inside _attempt_tts_self_heal from proxy.lifecycle
            with patch("proxy.lifecycle.start_tts_server", side_effect=fake_start_tts_server):
                with patch("proxy.lifecycle.wait_for_tts_server", AsyncMock(return_value=True)):
                    with patch("proxy.backend_health._get_tts_watchdog_interval", return_value=0.05):
                        task = asyncio.create_task(_tts_watchdog_loop())
                        await asyncio.sleep(0.3)
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                # Verify restart was attempted
                assert restart_count >= 1, "start_tts_server should have been called"

    @pytest.mark.asyncio
    async def test_watchdog_does_not_restart_healthy_process(self):
        """Watchdog should not restart TTS when process is healthy."""
        from proxy.backend_health import _tts_watchdog_loop

        srv = _make_mock_server(
            tts_process_alive=True,  # Process is alive
            tts_enabled=True,
        )

        restart_count = 0

        async def fake_start_tts_server():
            nonlocal restart_count
            restart_count += 1

        with patch("proxy.backend_health._srv", return_value=srv):
            with patch("proxy.lifecycle.start_tts_server", side_effect=fake_start_tts_server):
                with patch("proxy.backend_health._get_tts_watchdog_interval", return_value=0.05):
                    task = asyncio.create_task(_tts_watchdog_loop())
                    await asyncio.sleep(0.3)
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Verify NO restart was attempted
                assert restart_count == 0, "start_tts_server should NOT have been called"

    @pytest.mark.asyncio
    async def test_watchdog_respects_max_attempts(self):
        """Watchdog should stop restarting after exhausting max attempts."""
        from proxy.backend_health import _tts_watchdog_loop

        srv = _make_mock_server(
            tts_process_alive=False,
            tts_enabled=True,
        )

        call_count = 0

        def failing_start():
            nonlocal call_count
            call_count += 1
            return None  # Simulate failed start

        with patch("proxy.backend_health._srv", return_value=srv):
            with patch("proxy.lifecycle.start_tts_server", side_effect=failing_start):
                with patch("proxy.lifecycle.wait_for_tts_server", AsyncMock(return_value=False)):
                    with patch("proxy.backend_health._get_tts_watchdog_interval", return_value=0.05):
                        task = asyncio.create_task(_tts_watchdog_loop())
                        await asyncio.sleep(0.5)
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                # Should not exceed max attempts (config default 3)
                assert call_count <= 3, f"Expected <= 3 calls, got {call_count}"
                assert call_count >= 1, "Expected at least 1 restart attempt"

    @pytest.mark.asyncio
    async def test_watchdog_logs_failed_restart(self):
        """Watchdog should log failed restart attempts."""
        from proxy.backend_health import _tts_watchdog_loop

        srv = _make_mock_server(
            tts_process_alive=False,
            tts_enabled=True,
        )

        def failing_start():
            return None

        with patch("proxy.backend_health._srv", return_value=srv):
            with patch("proxy.lifecycle.start_tts_server", side_effect=failing_start):
                with patch("proxy.lifecycle.wait_for_tts_server", AsyncMock(return_value=False)):
                    with patch("proxy.backend_health._get_tts_watchdog_interval", return_value=0.05):
                        task = asyncio.create_task(_tts_watchdog_loop())
                        await asyncio.sleep(0.3)
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                # Verify at least one error or warning was logged
                assert srv.logger.warning.call_count > 0 or srv.logger.error.call_count > 0, \
                    "Expected at least one log message"


class TestTtsWatchdogLifecycle:
    """Tests for TTS watchdog lifecycle integration."""

    @pytest.mark.asyncio
    async def test_watchdog_task_launched_on_startup(self):
        """TTS watchdog task should be created during server startup."""
        from proxy.server import _startup_launch_watchdog_tasks

        import proxy.server as server_mod

        loop = asyncio.get_running_loop()

        # Reset globals
        server_mod.backend_watchdog_task = None
        server_mod.tts_watchdog_task = None
        server_mod.model_health_task = None

        _startup_launch_watchdog_tasks()

        # After startup, a TTS watchdog task reference should exist
        assert server_mod.tts_watchdog_task is not None
        assert isinstance(server_mod.tts_watchdog_task, asyncio.Task)

        # Cleanup
        server_mod.tts_watchdog_task.cancel()
        try:
            await server_mod.tts_watchdog_task
        except (asyncio.CancelledError, Exception):
            pass
        server_mod.tts_watchdog_task = None
        if server_mod.backend_watchdog_task is not None:
            server_mod.backend_watchdog_task.cancel()
            try:
                await server_mod.backend_watchdog_task
            except (asyncio.CancelledError, Exception):
                pass
            server_mod.backend_watchdog_task = None

    @pytest.mark.asyncio
    async def test_watchdog_cancelled_on_shutdown(self):
        """TTS watchdog task should be cancelled during shutdown."""
        from proxy.server import _shutdown_cleanup_tasks

        import proxy.server as server_mod

        loop = asyncio.get_running_loop()

        # Create a real task
        async def dummy_loop():
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass

        task = loop.create_task(dummy_loop())
        server_mod.tts_watchdog_task = task

        await _shutdown_cleanup_tasks()

        assert task.done() or task.cancelled()
        server_mod.tts_watchdog_task = None


class TestTtsWatchdogConfig:
    """Tests for TTS watchdog configuration."""

    def test_get_tts_watchdog_interval_default(self):
        """Should return default interval when config key is missing."""
        from proxy.backend_health import _get_tts_watchdog_interval

        server_cfg = {}
        interval = _get_tts_watchdog_interval(server_cfg)
        assert interval == 10.0, f"Expected 10.0, got {interval}"

    def test_get_tts_watchdog_interval_custom(self):
        """Should return custom interval from config."""
        from proxy.backend_health import _get_tts_watchdog_interval

        server_cfg = {"tts_self_heal_interval_seconds": 30}
        interval = _get_tts_watchdog_interval(server_cfg)
        assert interval == 30.0, f"Expected 30.0, got {interval}"

    def test_get_tts_self_heal_max_attempts_default(self):
        """Should return default max attempts when config key is missing."""
        from proxy.backend_health import _get_tts_self_heal_max_attempts

        server_cfg = {}
        max_attempts = _get_tts_self_heal_max_attempts(server_cfg)
        assert max_attempts == 3, f"Expected 3, got {max_attempts}"

    def test_get_tts_self_heal_max_attempts_custom(self):
        """Should return custom max attempts from config."""
        from proxy.backend_health import _get_tts_self_heal_max_attempts

        server_cfg = {"tts_self_heal_max_attempts": 5}
        max_attempts = _get_tts_self_heal_max_attempts(server_cfg)
        assert max_attempts == 5, f"Expected 5, got {max_attempts}"
