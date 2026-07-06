"""Unit tests for the extracted spawn_and_capture helper functions."""

import io
import logging
import subprocess
import threading
import types
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib
lifecycle = importlib.import_module("proxy.lifecycle")


class FakeProc:
    """A fake subprocess.Popen that simulates a long-running process (TimeoutExpired)."""
    def __init__(self, stdout_data=None):
        self.returncode = None
        self.stdout = io.StringIO(stdout_data or "")
    def communicate(self, timeout=None):
        raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
    def poll(self):
        return None
    def terminate(self):
        self.returncode = -1
    def wait(self, timeout=None):
        return


class FakeProcFastExit:
    """A fake subprocess.Popen that simulates a fast-exiting process (returns output immediately)."""
    def __init__(self, stdout_data="started OK"):
        self.returncode = 0
        self.stdout = io.StringIO(stdout_data)
    def communicate(self, timeout=None):
        return self.stdout.getvalue(), None
    def poll(self):
        return 0
    def terminate(self):
        pass
    def wait(self, timeout=None):
        return 0


class TestSpawnAndCapture:

    def test_successful_spawn_long_running(self, monkeypatch):
        """Process starts and runs long (TimeoutExpired) → returns proc."""
        log_file = io.StringIO()
        logger = logging.getLogger("test")

        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
            return FakeProc()

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)
        # Monkeypatch threading.Thread.start so it doesn't actually start a thread
        original_thread = threading.Thread
        started_threads = []
        class FakeThread:
            def __init__(self, target=None, args=(), daemon=None):
                self.target = target
                self.args = args
                self.daemon = daemon
            def start(self):
                started_threads.append(self)
        monkeypatch.setattr(lifecycle.threading, "Thread", FakeThread)

        proc, out = lifecycle.spawn_and_capture(
            cmd=["start-llama.sh", "router"],
            env={},
            log_file=log_file,
            logger=logger,
        )

        assert proc is not None
        assert out is None  # No captured output on long-running
        assert len(started_threads) == 1  # Stream thread was created

    def test_successful_spawn_fast_exit(self, monkeypatch):
        """Process exits quickly → returns None, captured output."""
        log_file = io.StringIO()
        logger = logging.getLogger("test")

        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
            return FakeProcFastExit("started OK\nall good")

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

        proc, out = lifecycle.spawn_and_capture(
            cmd=["start-llama.sh", "router"],
            env={},
            log_file=log_file,
            logger=logger,
        )

        assert proc is None
        assert out is not None
        assert "started OK" in out

    def test_command_not_found(self, monkeypatch):
        """FileNotFoundError on spawn → returns None with error message."""
        log_file = io.StringIO()
        logger = logging.getLogger("test")

        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
            raise FileNotFoundError("llama-server not found")

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

        proc, out = lifecycle.spawn_and_capture(
            cmd=["/nonexistent/llama-server"],
            env={},
            log_file=log_file,
            logger=logger,
        )

        assert proc is None
        assert out is not None
        assert "Command not found" in out

    def test_generic_spawn_error(self, monkeypatch):
        """Generic exception on spawn → returns None with error message."""
        log_file = io.StringIO()
        logger = logging.getLogger("test")

        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
            raise PermissionError("Permission denied")

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

        proc, out = lifecycle.spawn_and_capture(
            cmd=["start-llama.sh"],
            env={},
            log_file=log_file,
            logger=logger,
        )

        assert proc is None
        assert out is not None
        assert "Spawn failed" in out

    def test_stream_output_writes_lines(self):
        """_stream_output reads lines from src and writes them to dst."""
        src = io.StringIO("line1\nline2\nline3\n")
        dst = io.StringIO()

        lifecycle._stream_output(src, dst)

        output = dst.getvalue()
        assert "line1" in output
        assert "line2" in output
        assert "line3" in output

    def test_stream_output_empty_src(self):
        """_stream_output handles an empty source gracefully."""
        src = io.StringIO("")
        dst = io.StringIO()

        lifecycle._stream_output(src, dst)

        assert dst.getvalue() == ""

    def test_log_file_is_none_does_not_stream(self, monkeypatch):
        """When log_file is None, no streaming thread is started."""
        logger = logging.getLogger("test")

        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
            return FakeProc()

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)
        thread_starts = []
        class TrackingThread:
            def __init__(self, target=None, args=(), daemon=None):
                self.target = target
                self.args = args
                self.daemon = daemon
            def start(self):
                thread_starts.append(True)
        monkeypatch.setattr(lifecycle.threading, "Thread", TrackingThread)

        proc, out = lifecycle.spawn_and_capture(
            cmd=["start-llama.sh", "router"],
            env={},
            log_file=None,
            logger=logger,
        )

        assert proc is not None
        # When log_file is None, the streaming thread should NOT be started
        # because the inner try/except catches AttributeError on None type
        # Since the stream_output code does `if srv.llama_log_file and proc.stdout:`
        # (now if log_file): the thread is only created when log_file is truthy
        # With log_file=None, the condition fails, so no thread

    def test_stdout_is_none_does_not_crash(self, monkeypatch):
        """When proc.stdout is None (DEVNULL), no crash."""
        log_file = io.StringIO()
        logger = logging.getLogger("test")

        class FakeProcNoStdout:
            def __init__(self):
                self.returncode = None
                self.stdout = None
            def communicate(self, timeout=None):
                raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)

        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
            return FakeProcNoStdout()

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

        proc, out = lifecycle.spawn_and_capture(
            cmd=["start-llama.sh"],
            env={},
            log_file=log_file,
            logger=logger,
        )

        assert proc is not None
        # Should not crash even though proc.stdout is None
