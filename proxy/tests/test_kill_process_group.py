"""
Tests for ``_kill_process_group`` signal-safety (LP-0MS9LJIAE008Q3AK).

``os.killpg(pgid, sig)`` NEGATES the pgid: ``os.killpg(1, SIGTERM)`` issues
the syscall ``kill(-1, SIGTERM)`` which broadcasts the signal to EVERY
process the caller can signal. A fake/mock ``llama_process`` with ``pid=1``
left in the pytest process's module state caused a full-suite run to
SIGTERM the entire running proxy stack (RCA: LP-0MS9LJIAE008Q3AK / the
``kill(-1, SIGTERM)`` syscall captured under strace).

These tests guarantee the killpg path never runs with a pid <= 1
(0 = caller's own process group, 1 = broadcast via negation).
"""

import signal
from unittest.mock import MagicMock, patch

from proxy.lifecycle import _kill_process_group


class TestKillProcessGroup:
    def test_none_proc_returns_false(self):
        assert _kill_process_group(None, None) is False

    def test_valid_pid_kills_process_group(self):
        proc = MagicMock()
        proc.pid = 12345
        proc.poll.return_value = None
        with patch("proxy.lifecycle.os.killpg") as mock_killpg:
            result = _kill_process_group(proc, MagicMock())
        mock_killpg.assert_called_once_with(12345, signal.SIGTERM)
        assert result is True

    def test_pid_one_never_calls_killpg(self):
        # Regression case: os.killpg(1, SIGTERM) -> kill(-1, SIGTERM)
        # broadcasts to ALL processes (the proxy-stack killer).
        proc = MagicMock()
        proc.pid = 1
        proc.poll.return_value = None
        with patch("proxy.lifecycle.os.killpg") as mock_killpg:
            with patch("proxy.lifecycle.os.kill") as mock_kill:
                result = _kill_process_group(proc, MagicMock())
        mock_killpg.assert_not_called()
        mock_kill.assert_not_called()
        assert result is False

    def test_pid_zero_never_calls_killpg(self):
        # os.killpg(0, SIGTERM) targets the caller's own process group.
        proc = MagicMock()
        proc.pid = 0
        proc.poll.return_value = None
        with patch("proxy.lifecycle.os.killpg") as mock_killpg:
            with patch("proxy.lifecycle.os.kill") as mock_kill:
                result = _kill_process_group(proc, MagicMock())
        mock_killpg.assert_not_called()
        mock_kill.assert_not_called()
        assert result is False

    def test_bool_pid_one_never_calls_killpg(self):
        # bool is a subclass of int; True == 1 must not reach os.killpg.
        proc = MagicMock()
        proc.pid = True
        proc.poll.return_value = None
        with patch("proxy.lifecycle.os.killpg") as mock_killpg:
            result = _kill_process_group(proc, MagicMock())
        mock_killpg.assert_not_called()
        assert result is False

    def test_none_pid_never_calls_killpg(self):
        proc = MagicMock()
        proc.pid = None
        with patch("proxy.lifecycle.os.killpg") as mock_killpg:
            result = _kill_process_group(proc, None)
        mock_killpg.assert_not_called()
        assert result is False

    def test_non_int_pid_never_calls_killpg(self):
        # A MagicMock pid (fake process object) must not reach os.killpg.
        proc = MagicMock()
        proc.pid = MagicMock()
        proc.poll.return_value = None
        with patch("proxy.lifecycle.os.killpg") as mock_killpg:
            result = _kill_process_group(proc, None)
        mock_killpg.assert_not_called()
        assert result is False
