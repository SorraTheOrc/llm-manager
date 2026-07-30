"""
Tests for port release verification after process killing.

Verifies that:
- _wait_for_port_release() detects when a port is free or becomes free
- Port cleanup waits until the port is actually released (not just sleep)

All tests use safe in-process socket.bind() to hold ports. No real OS
subprocesses are spawned or killed.
"""

import os
import signal
import socket
import threading
import time
from unittest.mock import patch

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
        """Should return False when the port remains in use."""
        from proxy.lifecycle import _wait_for_port_release

        port = _find_free_port()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.listen(1)
        try:
            result = _wait_for_port_release(port, timeout=1.0, interval=0.1)
            assert result is False, (
                f"Expected False for busy port {port}, got {result}"
            )
        finally:
            s.close()

    def test_returns_true_after_port_becomes_free(self):
        """Should return True after the port is released."""
        from proxy.lifecycle import _wait_for_port_release

        port = _find_free_port()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.listen(1)

        def release():
            s.close()

        t = threading.Timer(0.5, release)
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

        t = threading.Timer(0.5, release)
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
