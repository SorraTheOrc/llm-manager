"""
Tests for port release verification after process killing.

Verifies that:
- _wait_for_port_release() detects when a port is free or becomes free
- Port cleanup waits until the port is actually released (not just sleep)
- Port-based process killing works (port freed after kill)
"""
import asyncio
import os
import signal
import socket
import subprocess
import time
from unittest.mock import MagicMock, patch

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


def _start_listener(port: int, close_delay: float = 0) -> subprocess.Popen:
    """Start a background TCP listener on the given port.

    The listener binds and listens until killed. If ``close_delay`` > 0 the
    listener self-closes after that many seconds.

    Returns the Popen handle.
    """
    if close_delay > 0:
        code = rf"""
import socket, threading, time
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('127.0.0.1', {port}))
s.listen(1)
print("listening", flush=True)
def close():
    s.close()
    print("closed", flush=True)
t = threading.Timer({close_delay}, close)
t.daemon = True
t.start()
time.sleep(10)
"""
    else:
        code = rf"""
import socket, time
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('127.0.0.1', {port}))
s.listen(1)
print("listening", flush=True)
time.sleep(30)
"""

    proc = subprocess.Popen(
        ["python3", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc


def _wait_for_listener_ready(proc: subprocess.Popen, timeout: float = 3.0) -> None:
    """Wait for the listener subprocess to print 'listening'."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.stdout and "listening" in proc.stdout.readline():
            return
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    raise TimeoutError("Listener did not start within timeout")


# ---------------------------------------------------------------------------
# Tests for _wait_for_port_release
# ---------------------------------------------------------------------------

class TestWaitForPortRelease:
    """Tests for the port release waiter."""

    def test_returns_true_when_port_is_free(self):
        """Should return True when the port is not in use."""
        from proxy.lifecycle import _wait_for_port_release

        free_port = _find_free_port()
        result = _wait_for_port_release(free_port, timeout=2.0)
        assert result is True, (
            f"Expected True for free port {free_port}, got {result}"
        )

    def test_returns_false_when_port_stays_busy(self):
        """Should return False when the port remains in use."""
        from proxy.lifecycle import _wait_for_port_release

        port = _find_free_port()
        proc = _start_listener(port)
        _wait_for_listener_ready(proc)

        try:
            result = _wait_for_port_release(port, timeout=1.0)
            assert result is False, (
                f"Expected False for busy port {port}, got {result}"
            )
        finally:
            proc.kill()
            proc.wait(timeout=2)

    def test_returns_true_after_port_becomes_free(self):
        """Should return True after the port is released."""
        from proxy.lifecycle import _wait_for_port_release

        port = _find_free_port()
        # Listener auto-closes after 0.7s
        proc = _start_listener(port, close_delay=0.7)
        _wait_for_listener_ready(proc)

        try:
            result = _wait_for_port_release(port, timeout=3.0)
            assert result is True, (
                f"Expected True after port released, got {result}"
            )
        finally:
            proc.kill()
            proc.wait(timeout=2)

    def test_port_available_for_rebind_after_release(self):
        """After _wait_for_port_release returns True, a new socket should be able to bind."""
        from proxy.lifecycle import _wait_for_port_release

        port = _find_free_port()
        proc = _start_listener(port, close_delay=0.5)
        _wait_for_listener_ready(proc)

        try:
            assert _wait_for_port_release(port, timeout=3.0), "Port did not release"

            # Verify we can bind to the port
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
            finally:
                s.close()
        finally:
            proc.kill()
            proc.wait(timeout=2)


# ---------------------------------------------------------------------------
# Tests for port-based process killing
# ---------------------------------------------------------------------------

class TestPortBasedKilling:
    """Tests for killing processes by port and verifying release."""

    def test_kill_process_frees_port(self):
        """Killing a process listening on a port should free the port."""
        from proxy.lifecycle import _wait_for_port_release

        port = _find_free_port()
        proc = _start_listener(port)
        _wait_for_listener_ready(proc)

        # Verify port is in use
        assert not _wait_for_port_release(port, timeout=0.3), (
            "Port should be in use before kill"
        )

        # Kill the process
        proc.kill()
        proc.wait(timeout=2)

        # Verify port is freed
        assert _wait_for_port_release(port, timeout=2.0), (
            "Port should be free after kill"
        )

    def test_multiple_kills_leave_port_free(self):
        """Killing multiple times (idempotent) should leave port free."""
        from proxy.lifecycle import _wait_for_port_release

        port = _find_free_port()
        proc = _start_listener(port)
        _wait_for_listener_ready(proc)

        # Kill twice (first succeeds, second is no-op)
        proc.kill()
        proc.wait(timeout=2)

        try:
            proc.kill()
        except ProcessLookupError:
            pass

        assert _wait_for_port_release(port, timeout=2.0), (
            "Port should be free after idempotent kill"
        )


# ---------------------------------------------------------------------------
# Tests for shell-level port cleanup (start-proxy.sh --restart)
# ---------------------------------------------------------------------------

class TestShellPortCleanup:
    """Tests for shell-script port cleanup that start-proxy.sh uses."""

    def test_fuser_frees_port(self):
        """Using fuser -k on a port should free it (fallback mechanism)."""
        if not self._has_fuser():
            pytest.skip("fuser not available on this system")

        from proxy.lifecycle import _wait_for_port_release

        port = _find_free_port()
        proc = _start_listener(port)
        _wait_for_listener_ready(proc)

        try:
            result = subprocess.run(
                ["fuser", "-k", f"{port}/tcp"],
                capture_output=True,
                text=True,
            )
            # fuser may return non-zero even when it kills successfully
            # e.g., exit code 1 when no process found, code 0 when kill succeeds
            # We verify by checking port release

            assert _wait_for_port_release(port, timeout=3.0), (
                f"Port {port} should be free after fuser -k (exit={result.returncode})"
            )
        finally:
            proc.kill()
            proc.wait(timeout=2)

    def test_fuser_on_already_free_port(self):
        """fuser -k on a free port should fail gracefully (no crash)."""
        if not self._has_fuser():
            pytest.skip("fuser not available on this system")

        port = _find_free_port()
        result = subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True,
            text=True,
        )
        # fuser returns 1 when no process is using the port
        assert result.returncode in (0, 1), (
            f"fuser on free port should not fail hard, got exit={result.returncode}, "
            f"stderr={result.stderr}"
        )

    @staticmethod
    def _has_fuser() -> bool:
        """Check if fuser is available on PATH."""
        try:
            result = subprocess.run(
                ["which", "fuser"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False


# ---------------------------------------------------------------------------
# Tests for wait-loop polling behavior
# ---------------------------------------------------------------------------

class TestPortReleasePolling:
    """Tests for polling interval and timeout behavior."""

    def test_short_timeout_on_busy_port(self):
        """Very short timeout on a busy port should return False quickly."""
        from proxy.lifecycle import _wait_for_port_release

        port = _find_free_port()
        proc = _start_listener(port)
        _wait_for_listener_ready(proc)

        try:
            start = time.monotonic()
            result = _wait_for_port_release(port, timeout=0.3, interval=0.1)
            elapsed = time.monotonic() - start

            assert result is False, "Should return False for busy port"
            # Should complete within ~2x the timeout + overhead
            assert elapsed < 2.0, f"Took too long: {elapsed:.2f}s"
        finally:
            proc.kill()
            proc.wait(timeout=2)
