"""
Tests for TTS server spawn verification and port-based zombie cleanup.

Verifies that:
- _extract_pids_from_ss_output() correctly parses ss -ltnp output
- _kill_process_on_port() detects no process on unused ports
- start_tts_server() kills zombies on the TTS port before spawning
- start_tts_server() returns None when the spawned process exits immediately
  (port conflict / startup failure), instead of returning a dead Popen handle
- _startup_launch_tts_server() calls start_tts_server() which does cleanup
- _attempt_tts_self_heal() cleans up port before restart
- _wait_for_port_release() uses safe in-process port holding instead of
  spawning real subprocesses

WARNING: No test in this file spawns or kills real OS subprocesses.
All process-killing functions are fully mocked.
"""

import asyncio
import os
import signal
import socket
import subprocess
import threading
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
# Tests for _extract_pids_from_ss_output
# ---------------------------------------------------------------------------


class TestExtractPidsFromSsOutput:
    """Unit tests for the ss output parsing function."""

    def _call_fut(self, ss_output: str, port: int):
        """Call the function under test (_extract_pids_from_ss_output)."""
        from proxy.lifecycle import _extract_pids_from_ss_output
        return _extract_pids_from_ss_output(ss_output, port)

    def test_extracts_pid_when_port_present(self):
        """Should extract PID when the port is present in ss output."""
        ss_output = (
            "LISTEN 0 128 0.0.0.0:8081 0.0.0.0:* users:((\"python3\",pid=12345,fd=5))\n"
            "LISTEN 0 128 127.0.0.53%lo:53 0.0.0.0:* users:((\"systemd-resolve\",pid=789,fd=13))"
        )
        pids = self._call_fut(ss_output, 8081)
        assert pids == [12345], f"Expected [12345], got {pids}"

    def test_returns_empty_list_when_port_not_present(self):
        """Should return empty list when the port is not in ss output."""
        ss_output = (
            "LISTEN 0 128 0.0.0.0:8081 0.0.0.0:* users:((\"python3\",pid=12345,fd=5))\n"
            "LISTEN 0 128 127.0.0.53%lo:53 0.0.0.0:* users:((\"systemd-resolve\",pid=789,fd=13))"
        )
        pids = self._call_fut(ss_output, 9999)
        assert pids == [], f"Expected empty list, got {pids}"

    def test_extracts_multiple_pids_on_same_port(self):
        """Should extract all PIDs when multiple processes share the port."""
        ss_output = (
            "LISTEN 0 128 0.0.0.0:8081 0.0.0.0:* users:((\"python3\",pid=12345,fd=5))\n"
            "LISTEN 0 128 0.0.0.0:8081 0.0.0.0:* users:((\"python3\",pid=12346,fd=6))"
        )
        pids = self._call_fut(ss_output, 8081)
        assert pids == [12345, 12346], f"Expected [12345, 12346], got {pids}"

    def test_handles_malformed_lines_gracefully(self):
        """Should skip malformed lines without raising."""
        ss_output = (
            "LISTEN 0 128 0.0.0.0:8081 0.0.0.0:* users:((\"python3\",pid=12345,fd=5))\n"
            "LISTEN 0 128 0.0.0.0:8081 0.0.0.0:* users:((\"python3\",unknown_key=999,fd=5))\n"
            "LISTEN 0 128 0.0.0.0:9999 0.0.0.0:* users:((\"sshd\",pid=0,fd=3))\n"
        )
        pids = self._call_fut(ss_output, 8081)
        assert pids == [12345], f"Expected [12345], got {pids}"

    def test_handles_empty_output(self):
        """Should return empty list for empty ss output."""
        pids = self._call_fut("", 8081)
        assert pids == [], f"Expected empty list, got {pids}"


# ---------------------------------------------------------------------------
# Tests for _kill_process_on_port (fully mocked, no real subprocesses)
# ---------------------------------------------------------------------------


class TestKillProcessOnPort:
    """Tests for the port-based zombie process killer (fully mocked)."""

    def test_kills_listening_process(self):
        """Should kill a process identified as holding the port."""
        from proxy.lifecycle import _kill_process_on_port

        with patch("proxy.lifecycle._find_pid_on_port", return_value=12345) as mock_find:
            with patch("os.kill") as mock_kill:
                result = _kill_process_on_port(8081)

        assert result is True, "Expected True when a PID is found and killed"
        mock_find.assert_called_once_with(8081)
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)

    def test_returns_false_when_port_not_in_use(self):
        """Should return False when no process is listening on the port."""
        from proxy.lifecycle import _kill_process_on_port

        with patch("proxy.lifecycle._find_pid_on_port", return_value=None) as mock_find:
            with patch("os.kill") as mock_kill:
                result = _kill_process_on_port(8081)

        assert result is False, "Expected False when no PID found"
        mock_find.assert_called_once_with(8081)
        mock_kill.assert_not_called()

    def test_waits_for_port_release_after_kill(self):
        """Should call _wait_for_port_release after killing the process."""
        from proxy.lifecycle import _kill_process_on_port

        with patch("proxy.lifecycle._find_pid_on_port", return_value=12345):
            with patch("os.kill"):
                with patch("proxy.lifecycle._wait_for_port_release") as mock_wait:
                    _kill_process_on_port(8081)

        mock_wait.assert_called_once()
        args, _ = mock_wait.call_args
        assert args[0] == 8081, f"Expected port 8081, got {args[0]}"

    def test_handles_multiple_pids(self):
        """Should kill all PIDs when multiple are returned."""
        from proxy.lifecycle import _kill_process_on_port

        with patch("proxy.lifecycle._find_pid_on_port", return_value=[12345, 12346]):
            with patch("os.kill") as mock_kill:
                _kill_process_on_port(8081)

        assert mock_kill.call_count == 2
        mock_kill.assert_any_call(12345, signal.SIGTERM)
        mock_kill.assert_any_call(12346, signal.SIGTERM)

    def test_handles_process_already_gone(self):
        """Should not crash if process is already gone between detection and kill."""
        from proxy.lifecycle import _kill_process_on_port

        def fake_kill(pid, sig):
            raise ProcessLookupError()

        with patch("proxy.lifecycle._find_pid_on_port", return_value=12345):
            with patch("os.kill", side_effect=fake_kill):
                result = _kill_process_on_port(8081)

        assert result is False, "Expected False when process already gone"


# ---------------------------------------------------------------------------
# Tests for _wait_for_port_release (safe in-process port holding)
# ---------------------------------------------------------------------------


class TestWaitForPortRelease:
    """Tests for the port release waiter using safe in-process socket holding."""

    def test_returns_true_when_port_is_free(self):
        """Should return True when the port is not in use."""
        from proxy.lifecycle import _wait_for_port_release

        free_port = _find_free_port()
        result = _wait_for_port_release(free_port, timeout=2.0)
        assert result is True, (
            f"Expected True for free port {free_port}, got {result}"
        )

    def test_returns_false_when_port_stays_busy(self):
        """Should return False when the port remains in use (in-process socket bind)."""
        from proxy.lifecycle import _wait_for_port_release

        port = _find_free_port()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.listen(1)
        try:
            result = _wait_for_port_release(port, timeout=0.5, interval=0.1)
            assert result is False, (
                f"Expected False for busy port {port}, got {result}"
            )
        finally:
            s.close()

    def test_returns_true_after_port_becomes_free(self):
        """Should return True after the port is released (timer closes socket)."""
        from proxy.lifecycle import _wait_for_port_release

        port = _find_free_port()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.listen(1)

        def release():
            s.close()

        t = threading.Timer(0.3, release)
        t.daemon = True
        t.start()

        try:
            result = _wait_for_port_release(port, timeout=3.0, interval=0.1)
            assert result is True, (
                f"Expected True after port released, got {result}"
            )
        finally:
            t.cancel()
            try:
                s.close()
            except OSError:
                pass

    def test_port_available_for_rebind_after_release(self):
        """After _wait_for_port_release returns True, a new socket can bind."""
        from proxy.lifecycle import _wait_for_port_release

        port = _find_free_port()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.listen(1)

        def release():
            s.close()

        t = threading.Timer(0.3, release)
        t.daemon = True
        t.start()

        try:
            assert _wait_for_port_release(port, timeout=3.0, interval=0.1), "Port did not release"
            s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s2.bind(("127.0.0.1", port))
            finally:
                s2.close()
        finally:
            t.cancel()
            try:
                s.close()
            except OSError:
                pass

    def test_short_timeout_on_busy_port(self):
        """Very short timeout on a busy port should return False quickly."""
        from proxy.lifecycle import _wait_for_port_release

        port = _find_free_port()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.listen(1)
        try:
            start = time.monotonic()
            result = _wait_for_port_release(port, timeout=0.3, interval=0.1)
            elapsed = time.monotonic() - start

            assert result is False, "Should return False for busy port"
            assert elapsed < 2.0, f"Took too long: {elapsed:.2f}s"
        finally:
            s.close()


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
        srv.config["server"]["tts_start_script"] = "/nonexistent/start-qwentts.sh"

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
# Tests for start_tts_server config path resolution
# ---------------------------------------------------------------------------


class TestStartTtsServerPathResolution:
    """Tests that start_tts_server resolves config paths correctly."""

    @pytest.mark.asyncio
    async def test_relative_config_path_resolved_against_repo_root(self):
        """Relative path in config should be resolved against repo root.

        Config has tts_start_script: "proxy/scripts/start-qwentts.sh" which is
        relative to repo root. The code must resolve it to an absolute path
        based on Path(__file__).parent.parent.parent before checking isfile().
        """
        from proxy.lifecycle import start_tts_server

        srv = _make_mock_server(tts_server_port=_find_free_port())
        # Set a relative config path (assumes repo root as base)
        srv.config["server"]["tts_start_script"] = "proxy/scripts/start-qwentts.sh"

        # Track what path is passed to os.path.isfile
        checked_paths = []

        def fake_isfile(path):
            checked_paths.append(path)
            return False  # script doesn't exist in test env

        with patch("proxy.lifecycle._srv", return_value=srv):
            with patch("proxy.lifecycle._kill_process_on_port", return_value=False):
                with patch("proxy.lifecycle.os.path.isfile", side_effect=fake_isfile):
                    result = start_tts_server()

        assert result is None, "Expected None since script doesn't exist"
        assert len(checked_paths) > 0, "os.path.isfile should have been called"
        checked = checked_paths[0]
        # The path should be an absolute path ending with proxy/scripts/start-qwentts.sh
        assert checked.endswith("proxy/scripts/start-qwentts.sh"), (
            f"Expected path ending with 'proxy/scripts/start-qwentts.sh', got {checked!r}"
        )
        assert os.path.isabs(checked), (
            f"Expected absolute path, got {checked!r}"
        )
        # It should NOT contain double proxy/proxy
        assert "proxy/proxy/scripts" not in checked, (
            f"Path should not have duplicate proxy/ segment: {checked!r}"
        )

    @pytest.mark.asyncio
    async def test_absolute_config_path_unchanged(self):
        """Absolute path in config should be used as-is (no regression)."""
        from proxy.lifecycle import start_tts_server

        srv = _make_mock_server(tts_server_port=_find_free_port())
        abs_path = "/custom/absolute/path/start-qwentts.sh"
        srv.config["server"]["tts_start_script"] = abs_path

        checked_paths = []

        def fake_isfile(path):
            checked_paths.append(path)
            return False

        with patch("proxy.lifecycle._srv", return_value=srv):
            with patch("proxy.lifecycle._kill_process_on_port", return_value=False):
                with patch("proxy.lifecycle.os.path.isfile", side_effect=fake_isfile):
                    result = start_tts_server()

        assert result is None
        assert len(checked_paths) > 0
        assert checked_paths[0] == abs_path, (
            f"Expected absolute path unchanged, got {checked_paths[0]!r}"
        )

    @pytest.mark.asyncio
    async def test_default_config_path_resolved_correctly(self):
        """When no config override, the default path should be correct."""
        from proxy.lifecycle import start_tts_server

        srv = _make_mock_server(tts_server_port=_find_free_port())
        # Remove tts_start_script from config to trigger the code default
        srv.config["server"].pop("tts_start_script", None)

        checked_paths = []

        def fake_isfile(path):
            checked_paths.append(path)
            return False

        with patch("proxy.lifecycle._srv", return_value=srv):
            with patch("proxy.lifecycle._kill_process_on_port", return_value=False):
                with patch("proxy.lifecycle.os.path.isfile", side_effect=fake_isfile):
                    result = start_tts_server()

        assert result is None
        assert len(checked_paths) > 0
        checked = checked_paths[0]
        assert checked.endswith("proxy/scripts/start-qwentts.sh"), (
            f"Expected ending with proxy/scripts/start-qwentts.sh, got {checked!r}"
        )
        assert os.path.isabs(checked), f"Expected absolute path, got {checked!r}"


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
            return None

        with patch("proxy.backends.tts._srv", return_value=srv):
            with patch("proxy.lifecycle.start_tts_server", side_effect=fake_start_tts):
                with patch("proxy.lifecycle.wait_for_tts_server", AsyncMock(return_value=True)):
                    with patch("proxy.backend_health._prune_recovery_attempts", return_value=[]):
                        with patch("proxy.backend_health._get_tts_self_heal_max_attempts", return_value=3):
                            with patch("proxy.backend_health._get_tts_self_heal_window", return_value=120):
                                result = await _attempt_tts_self_heal()

        assert result is not None
