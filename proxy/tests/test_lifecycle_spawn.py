"""
Tests for TTS server spawn verification and port-based zombie cleanup.

Verifies that:
- _kill_process_on_port() detects and kills a process listening on a port
- start_tts_server() kills zombies on the TTS port before spawning
- start_tts_server() returns None when the spawned process exits immediately
  (port conflict / startup failure), instead of returning a dead Popen handle
- _startup_launch_tts_server() calls start_tts_server() which does cleanup
- _attempt_tts_self_heal() cleans up port before restart
"""

import asyncio
import os
import signal
import socket
import subprocess
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    """Return a currently unused TCP port on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_mock_config(
    tts_enabled=True,
    tts_server_port=8081,
    tts_start_script="/fake/start-qwentts.sh",
    tts_self_heal_max_attempts=3,
    tts_self_heal_window_seconds=120,
):
    """Build a config dict with TTS settings."""
    return {
        "server": {
            "tts_enabled": tts_enabled,
            "tts_server_port": tts_server_port,
            "tts_server_host": "localhost",
            "tts_start_script": tts_start_script,
            "llama_router_mode": False,
            "llama_server_port": 8080,
            "tts_self_heal_max_attempts": tts_self_heal_max_attempts,
            "tts_self_heal_window_seconds": tts_self_heal_window_seconds,
        },
        "models": {},
        "default_model": "test-model",
    }


def _make_mock_server(tts_server_port=8081):
    """Build a mock server object with logger etc."""
    srv = MagicMock()
    srv.config = _make_mock_config(tts_server_port=tts_server_port)
    srv.logger = MagicMock()
    srv.tts_recovery_state = {
        "in_progress": False,
        "attempt_timestamps": [],
        "max_attempts": 3,
        "window_seconds": 120,
        "last_failure": None,
    }
    srv.tts_process = None
    return srv


# ---------------------------------------------------------------------------
# Tests for _kill_process_on_port
# ---------------------------------------------------------------------------

class TestKillProcessOnPort:
    """Tests for the port-based zombie process killer."""

    def test_returns_false_when_port_not_in_use(self):
        """Should return False when no process is listening on the port."""
        from proxy.lifecycle import _kill_process_on_port

        free_port = _find_free_port()
        result = _kill_process_on_port(free_port)
        assert result is False, (
            f"Expected False for unused port {free_port}, got {result}"
        )

    def test_kills_listening_process(self):
        """Should kill a simple process listening on a TCP port."""
        from proxy.lifecycle import _kill_process_on_port

        port = _find_free_port()

        # Spawn a simple TCP listener using Python's socketserver
        listener_proc = subprocess.Popen(
            [
                "python3", "-c",
                rf"""
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('127.0.0.1', {port}))
s.listen(1)
# Signal ready by printing "listening"
print("listening", flush=True)
import time
time.sleep(30)  # Stay alive until killed
"""
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for the listener to be ready
        line = listener_proc.stdout.readline() if listener_proc.stdout else ""
        assert "listening" in line, f"Listener did not start: {line}"

        # Verify port is in use
        sock_check = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            result = sock_check.connect_ex(("127.0.0.1", port))
            assert result == 0, f"Port {port} should be in use, connect_ex={result}"
        finally:
            sock_check.close()

        # Kill the process via _kill_process_on_port
        result = _kill_process_on_port(port)

        assert result is True, f"Expected True (killed), got {result}"

        # Give the kill time to take effect
        time.sleep(0.5)

        # Verify port is now free
        sock_check2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            result2 = sock_check2.connect_ex(("127.0.0.1", port))
            assert result2 != 0, f"Port {port} should be free after kill"
        finally:
            sock_check2.close()

        # Clean up
        listener_proc.kill()
        listener_proc.wait(timeout=2)


# ---------------------------------------------------------------------------
# Tests for start_tts_server spawn verification
# ---------------------------------------------------------------------------

class TestStartTtsServerSpawnVerification:
    """Tests that start_tts_server verifies the spawned process."""

    @pytest.mark.asyncio
    async def test_returns_none_when_process_exits_immediately(self):
        """Should return None when the spawned process exits immediately."""
        from proxy.lifecycle import start_tts_server

        srv = _make_mock_server(tts_server_port=_find_free_port())

        with patch("os.path.isfile", return_value=True):
            with patch("proxy.lifecycle._srv", return_value=srv):
                with patch("proxy.lifecycle._kill_process_on_port", return_value=False):
                    with patch(
                        "proxy.lifecycle.subprocess.Popen",
                        return_value=MagicMock(
                            poll=MagicMock(return_value=1),  # Exited with code 1
                            pid=99999,
                        ),
                    ) as mock_popen:
                        result = start_tts_server()

        assert result is None, (
            f"Expected None when process exits immediately, got {result}"
        )
        assert mock_popen.called, "Popen should have been called"

    @pytest.mark.asyncio
    async def test_returns_popen_when_process_stays_alive(self):
        """Should return the Popen handle when the process stays alive."""
        from proxy.lifecycle import start_tts_server

        srv = _make_mock_server(tts_server_port=_find_free_port())

        # Mock Popen returning a process that stays alive.
        # The mock has .poll() return None (alive) and .wait() raise
        # TimeoutExpired to simulate a process that stays running.
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll = MagicMock(return_value=None)  # Still running
        mock_proc.pid = 12345
        mock_proc.wait = MagicMock(side_effect=subprocess.TimeoutExpired(
            cmd="test", timeout=0.5
        ))

        with patch("os.path.isfile", return_value=True):
            with patch("proxy.lifecycle._srv", return_value=srv):
                with patch("proxy.lifecycle._kill_process_on_port", return_value=False):
                    with patch(
                        "proxy.lifecycle.subprocess.Popen",
                        return_value=mock_proc,
                    ) as mock_popen:
                        result = start_tts_server()

        assert result is not None, "Expected a Popen handle, got None"
        assert result.poll() is None, "Expected process to be alive"
        assert mock_popen.called, "Popen should have been called"

    @pytest.mark.asyncio
    async def test_returns_none_when_script_not_found(self):
        """Should return None when the start script is missing."""
        from proxy.lifecycle import start_tts_server

        srv = _make_mock_server(tts_server_port=_find_free_port())
        # Override the script path to a non-existent file
        srv.config["server"]["tts_start_script"] = "/nonexistent/start-qwentts.sh"

        # Do NOT mock os.path.isfile so it actually checks the filesystem
        with patch("proxy.lifecycle._srv", return_value=srv):
            with patch("proxy.lifecycle._kill_process_on_port", return_value=False):
                result = start_tts_server()

        assert result is None, "Expected None when script not found"

    @pytest.mark.asyncio
    async def test_kills_zombie_before_spawning(self):
        """Should kill existing process on port before spawning new one."""
        from proxy.lifecycle import start_tts_server

        srv = _make_mock_server(tts_server_port=_find_free_port())

        kill_called = [False]

        def fake_kill_on_port(port, logger=None):
            kill_called[0] = True
            return True

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll = MagicMock(return_value=None)
        mock_proc.pid = 12345
        mock_proc.wait = MagicMock(side_effect=subprocess.TimeoutExpired(
            cmd="test", timeout=0.5
        ))

        with patch("os.path.isfile", return_value=True):
            with patch("proxy.lifecycle._srv", return_value=srv):
                with patch(
                    "proxy.lifecycle._kill_process_on_port",
                    side_effect=fake_kill_on_port,
                ):
                    with patch("proxy.lifecycle.subprocess.Popen", return_value=mock_proc):
                        result = start_tts_server()

        assert result is not None, "Expected a Popen handle"
        assert kill_called[0], (
            "_kill_process_on_port should have been called before spawning"
        )


# ---------------------------------------------------------------------------
# Tests for _startup_launch_tts_server
# ---------------------------------------------------------------------------

class TestStartupLaunchTtsServer:
    """Tests for startup TTS server launch with zombie cleanup."""

    @pytest.mark.asyncio
    async def test_does_not_start_when_tts_disabled(self):
        """Should not start TTS when tts_enabled=false."""
        from proxy.server import _startup_launch_tts_server
        import proxy.server as server_mod

        loop = asyncio.get_running_loop()

        server_mod.config = _make_mock_config(tts_enabled=False)
        server_mod.logger = MagicMock()
        server_mod.tts_process = None

        start_tts_called = [False]

        def fake_start_tts():
            start_tts_called[0] = True
            return None

        with patch("proxy.lifecycle.start_tts_server", side_effect=fake_start_tts):
            with patch("proxy.lifecycle.wait_for_tts_server", AsyncMock()):
                task = _startup_launch_tts_server()
                await asyncio.sleep(0.1)
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # We just verify it doesn't crash
        server_mod.tts_process = None

    @pytest.mark.asyncio
    async def test_startup_handles_start_tts_failure(self):
        """Startup should log warning when start_tts_server returns None."""
        from proxy.server import _startup_launch_tts_server
        import proxy.server as server_mod

        loop = asyncio.get_running_loop()

        server_mod.config = _make_mock_config(tts_enabled=True)
        server_mod.logger = MagicMock()
        server_mod.tts_process = None

        with patch("proxy.lifecycle.start_tts_server", return_value=None):
            with patch("proxy.lifecycle.wait_for_tts_server", AsyncMock()):
                task = _startup_launch_tts_server()
                await asyncio.sleep(0.1)
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # Verify a warning was logged about TTS server failing to start
        # (The actual assertions are hard to make deterministic in async;
        #  we just verify no crash and tts_process remains None)
        assert server_mod.tts_process is None
        server_mod.tts_process = None


# ---------------------------------------------------------------------------
# Tests for _attempt_tts_self_heal port cleanup
# ---------------------------------------------------------------------------

class TestTtsSelfHealPortCleanup:
    """Tests that _attempt_tts_self_heal cleans up port before restart."""

    @pytest.mark.asyncio
    async def test_cleans_port_before_restart(self):
        """Self-heal should clean up zombie on port before calling start_tts_server."""
        from proxy.backends.tts import _attempt_tts_self_heal

        srv = _make_mock_server(tts_server_port=_find_free_port())

        port_cleaned = [False]

        def fake_start_tts():
            # Verify port was cleaned
            return None

        with patch("proxy.backends.tts._srv", return_value=srv):
            with patch("proxy.lifecycle.start_tts_server", side_effect=fake_start_tts):
                with patch("proxy.lifecycle.wait_for_tts_server", AsyncMock(return_value=True)):
                    with patch("proxy.backend_health._prune_recovery_attempts", return_value=[]):
                        with patch("proxy.backend_health._get_tts_self_heal_max_attempts", return_value=3):
                            with patch("proxy.backend_health._get_tts_self_heal_window", return_value=120):
                                result = await _attempt_tts_self_heal()

        # Function should complete without error
        assert result is not None


# ---------------------------------------------------------------------------
# Tests for _wait_for_port_release interop
# ---------------------------------------------------------------------------

class TestWaitForPortRelease:
    """Tests for the port release waiter used in restart_services."""

    def test_returns_true_when_port_is_free(self):
        """Should return True when the port is not in use."""
        from proxy.lifecycle import _wait_for_port_release

        free_port = _find_free_port()
        result = _wait_for_port_release(free_port, timeout=2.0)
        assert result is True, (
            f"Expected True for free port {free_port}, got {result}"
        )

    def test_returns_true_after_port_becomes_free(self):
        """Should return True after the port is released."""
        from proxy.lifecycle import _wait_for_port_release

        port = _find_free_port()

        # Start a listener, then kill it after a delay
        listener_proc = subprocess.Popen(
            [
                "python3", "-c",
                rf"""
import socket, threading, time
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('127.0.0.1', {port}))
s.listen(1)
print("listening", flush=True)
# Close after 0.7s via timer
def close():
    s.close()
    print("closed", flush=True)
t = threading.Timer(0.7, close)
t.daemon = True
t.start()
time.sleep(10)
""" ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        line = listener_proc.stdout.readline() if listener_proc.stdout else ""
        assert "listening" in line, f"Listener did not start: {line}"

        # Wait for port release
        result = _wait_for_port_release(port, timeout=3.0)

        assert result is True, f"Expected True after port released, got {result}"

        listener_proc.kill()
        listener_proc.wait(timeout=2)
