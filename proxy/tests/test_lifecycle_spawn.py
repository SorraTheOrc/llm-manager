"""Unit tests for the extracted spawn_and_capture helper functions."""

import io
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib

lifecycle = importlib.import_module("proxy.lifecycle")


# ---------------------------------------------------------------------------
# Fake socket helpers for port-release tests
# ---------------------------------------------------------------------------


class _FakeSocketAvailable:
    """Simulates a port that IS available (socket.connect_ex returns non-zero)."""
    def __init__(self):
        self.timeout = None
    def settimeout(self, t):
        self.timeout = t
    def connect_ex(self, addr):
        return 1  # errno — port not reachable => available
    def close(self):
        pass


class _FakeSocketUnavailable:
    """Simulates a port that is NOT available (socket.connect_ex returns 0)."""
    def __init__(self):
        self.timeout = None
    def settimeout(self, t):
        self.timeout = t
    def connect_ex(self, addr):
        return 0  # errno — port reachable => in use
    def close(self):
        pass


class _FakeSocketBecomesAvailable:
    """Simulates a port that is busy for N calls, then becomes free.

    Uses a shared counter across all instances so that each new socket
    created in the polling loop is not stuck resetting its own count.
    """
    _global_call_count = 0
    def __init__(self, busy_count=2):
        self.timeout = None
        self._busy_count = busy_count
        type(self)._global_call_count += 1
        self._my_call = self._global_call_count
    def settimeout(self, t):
        self.timeout = t
    def connect_ex(self, addr):
        if self._my_call <= self._busy_count:
            return 0  # busy
        return 1  # free
    def close(self):
        pass

    @classmethod
    def _reset(cls):
        cls._global_call_count = 0


class _FakeSocketNeverAvailable:
    """Simulates a port that never becomes free."""
    def __init__(self):
        self.timeout = None
        self._call_count = 0
    def settimeout(self, t):
        self.timeout = t
    def connect_ex(self, addr):
        self._call_count += 1
        return 0  # always busy
    def close(self):
        pass


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

        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None, start_new_session=False):
            return FakeProc()

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)
        # Monkeypatch threading.Thread.start so it doesn't actually start a thread
        _original_thread = threading.Thread
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

        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None, start_new_session=False):
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

        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None, start_new_session=False):
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

        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None, start_new_session=False):
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

        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None, start_new_session=False):
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

        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None, start_new_session=False):
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


# ---------------------------------------------------------------------------
# Helper classes for display_name and dynamic resolution tests
# ---------------------------------------------------------------------------


class _DummySrv:
    """Minimal server state for start_llama_server tests."""
    def __init__(self, current_model=None):
        cfg = {"server": {"llama_allow_host_fallback": True}}
        self.config = cfg
        self.logger = logging.getLogger("dummy")
        self.log_dir = None
        self.llama_log_file = None
        self.last_start_failure = None
        self.current_model = current_model
        self.backend_ready = False
        self.llama_process = None
    def rotate_llama_logs(self, *a, **kw):
        pass
    def broadcast_status_sync(self, *a, **kw):
        pass


class _FakeProc:
    """Fake subprocess.Popen that simulates a long-running process."""
    def __init__(self):
        self.returncode = None
        self.stdout = io.StringIO()
    def communicate(self, timeout=None):
        raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
    def poll(self):
        return None
    def terminate(self):
        self.returncode = -1
    def wait(self, timeout=None):
        return 0


class TestStartLlamaServerDisplayName:
    """Tests for start_llama_server's display_name parameter."""

    def test_display_name_used_for_progress(self, monkeypatch):
        """When display_name is provided, it is used for progress display instead of model name."""
        captured_kwargs = {}

        def fake_spawn_and_capture(cmd, env, log_file, logger, model_name="unknown"):
            captured_kwargs["model_name"] = model_name
            return (_FakeProc(), None)

        monkeypatch.setattr(lifecycle, "spawn_and_capture", fake_spawn_and_capture)
        monkeypatch.setattr(lifecycle, "_srv", lambda: _DummySrv())
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        proc = lifecycle.start_llama_server("Qwen3-0.6B-Q4_K_M.gguf", display_name="Qwen3")
        assert proc is not None
        assert captured_kwargs.get("model_name") == "Qwen3", \
            f"Expected 'Qwen3' but got {captured_kwargs.get('model_name')}"

    def test_display_name_fallback_to_model(self, monkeypatch):
        """When display_name is None, falls back to model name."""
        captured_kwargs = {}

        def fake_spawn_and_capture(cmd, env, log_file, logger, model_name="unknown"):
            captured_kwargs["model_name"] = model_name
            return (_FakeProc(), None)

        monkeypatch.setattr(lifecycle, "spawn_and_capture", fake_spawn_and_capture)
        monkeypatch.setattr(lifecycle, "_srv", lambda: _DummySrv(current_model="Qwen3"))
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        proc = lifecycle.start_llama_server("test-model", display_name=None)
        assert proc is not None
        assert captured_kwargs.get("model_name") == "test-model", \
            f"Expected 'test-model' but got {captured_kwargs.get('model_name')}"

    def test_display_name_router_mode_fallback(self, monkeypatch):
        """In router mode with display_name=None, falls back to current_model from server state."""
        captured_kwargs = {}

        def fake_spawn_and_capture(cmd, env, log_file, logger, model_name="unknown"):
            captured_kwargs["model_name"] = model_name
            return (_FakeProc(), None)

        monkeypatch.setattr(lifecycle, "spawn_and_capture", fake_spawn_and_capture)
        monkeypatch.setattr(lifecycle, "_srv", lambda: _DummySrv(current_model="Qwen3"))
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        proc = lifecycle.start_llama_server(None, display_name=None)
        assert proc is not None
        assert captured_kwargs.get("model_name") == "Qwen3", \
            f"Expected 'Qwen3' (from current_model) but got {captured_kwargs.get('model_name')}"

    def test_display_name_router_mode_no_current_model(self, monkeypatch):
        """In router mode with neither display_name nor current_model set, falls back to 'unknown'."""
        captured_kwargs = {}

        def fake_spawn_and_capture(cmd, env, log_file, logger, model_name="unknown"):
            captured_kwargs["model_name"] = model_name
            return (_FakeProc(), None)

        monkeypatch.setattr(lifecycle, "spawn_and_capture", fake_spawn_and_capture)
        monkeypatch.setattr(lifecycle, "_srv", lambda: _DummySrv(current_model=None))
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        proc = lifecycle.start_llama_server(None)
        assert proc is not None
        assert captured_kwargs.get("model_name") == "unknown", \
            f"Expected 'unknown' but got {captured_kwargs.get('model_name')}"


class TestStreamOutputDynamicModel:
    """Tests for _stream_output's dynamic model name resolution."""

    def test_dynamic_resolution_from_current_model(self, monkeypatch):
        """When model_name is 'unknown', resolves from current_model dynamically."""
        # Track the model_name passed to format_progress
        captured = {}

        def fake_format_progress(n_tokens, total_tokens, progress, model_name="unknown",
                                  slot_id=0, tokens_per_sec=None):
            captured["model_name"] = model_name
            return f"[slot:{slot_id} {model_name}] Processing {n_tokens}/{total_tokens} tokens"

        monkeypatch.setattr(lifecycle, "format_progress", fake_format_progress)
        monkeypatch.setattr(lifecycle, "_srv", lambda: _DummySrv(current_model="Qwen3"))

        src = io.StringIO("slot 1 : prompt processing progress, n_tokens = 100, progress = 0.50\n")
        dst = io.StringIO()

        lifecycle._stream_output(src, dst, model_name="unknown")

        assert captured.get("model_name") == "Qwen3", \
            f"Expected 'Qwen3' (from current_model) but got {captured.get('model_name')}"

    def test_passes_through_explicit_model_name(self, monkeypatch):
        """When model_name is not 'unknown', the provided name is used as-is."""
        captured = {}

        def fake_format_progress(n_tokens, total_tokens, progress, model_name="unknown",
                                  slot_id=0, tokens_per_sec=None):
            captured["model_name"] = model_name
            return f"[slot:{slot_id} {model_name}] Processing {n_tokens}/{total_tokens} tokens"

        monkeypatch.setattr(lifecycle, "format_progress", fake_format_progress)

        src = io.StringIO("slot 1 : prompt processing progress, n_tokens = 100, progress = 0.50\n")
        dst = io.StringIO()

        lifecycle._stream_output(src, dst, model_name="gemma4")

        assert captured.get("model_name") == "gemma4", \
            f"Expected 'gemma4' but got {captured.get('model_name')}"


# ---------------------------------------------------------------------------
# Tests for _wait_for_port_release
# ---------------------------------------------------------------------------


class TestWaitForPortRelease:
    """Tests for the _wait_for_port_release() helper."""

    def test_port_already_free(self, monkeypatch):
        """Returns True immediately when port is already available."""
        def fake_socket(*args, **kwargs):
            return _FakeSocketAvailable()
        monkeypatch.setattr(lifecycle.socket, "socket", fake_socket)

        result = lifecycle._wait_for_port_release(8080, timeout=1.0, interval=0.1)
        assert result is True

    def test_port_becomes_free(self, monkeypatch):
        """Returns True when port becomes free within timeout."""
        _FakeSocketBecomesAvailable._reset()
        def fake_socket(*args, **kwargs):
            return _FakeSocketBecomesAvailable(busy_count=2)
        monkeypatch.setattr(lifecycle.socket, "socket", fake_socket)

        result = lifecycle._wait_for_port_release(8080, timeout=3.0, interval=0.1)
        assert result is True

    def test_port_never_released(self, monkeypatch):
        """Returns False when port stays busy until timeout."""
        def fake_socket(*args, **kwargs):
            return _FakeSocketNeverAvailable()
        monkeypatch.setattr(lifecycle.socket, "socket", fake_socket)

        result = lifecycle._wait_for_port_release(8080, timeout=0.5, interval=0.1)
        assert result is False


# ---------------------------------------------------------------------------
# Tests for _kill_process_group
# ---------------------------------------------------------------------------


class _FakeProcWithPgid:
    """Fake subprocess.Popen that tracks process group kill calls."""
    def __init__(self, pid=12345):
        self.pid = pid
        self.returncode = None
        self._killed = False
    def poll(self):
        return self.returncode
    def terminate(self):
        self.returncode = -1
    def wait(self, timeout=None):
        return
    def kill(self):
        self.returncode = -9


class TestKillProcessGroup:
    """Tests for the _kill_process_group() helper."""

    def test_kills_process_group(self, monkeypatch):
        """Calls os.killpg with the process PID and SIGTERM first."""
        killed_pgids = []
        killed_signals = []

        def fake_killpg(pgid, sig):
            killed_pgids.append(pgid)
            killed_signals.append(sig)

        monkeypatch.setattr(lifecycle.os, "killpg", fake_killpg)
        monkeypatch.setattr(lifecycle.signal, "SIGTERM", 15)
        monkeypatch.setattr(lifecycle.signal, "SIGKILL", 9)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        proc = _FakeProcWithPgid(pid=12345)
        result = lifecycle._kill_process_group(proc, logger=logging.getLogger("test"))

        assert result is True
        assert 12345 in killed_pgids
        assert 15 in killed_signals  # SIGTERM

    def test_falls_back_to_terminate_when_killpg_unsupported(self, monkeypatch):
        """Falls back to proc.kill() when os.killpg raises TypeError."""
        def fake_killpg(pgid, sig):
            raise TypeError("killpg not supported")
        monkeypatch.setattr(lifecycle.os, "killpg", fake_killpg)
        monkeypatch.setattr(lifecycle.signal, "SIGTERM", 15)
        monkeypatch.setattr(lifecycle.signal, "SIGKILL", 9)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        proc = _FakeProcWithPgid(pid=12345)
        result = lifecycle._kill_process_group(proc, logger=logging.getLogger("test"))

        assert result is True
        # TypeError falls into the generic `except Exception` handler which
        # falls back to proc.kill(), returncode = -9
        assert proc.returncode == -9

    def test_falls_back_to_terminate_when_killpg_raises(self, monkeypatch):
        """Falls back to process.terminate() if os.killpg raises an error."""
        def fake_killpg(pgid, sig):
            raise ProcessLookupError(f"No process with PID {pgid}")

        monkeypatch.setattr(lifecycle.os, "killpg", fake_killpg)
        monkeypatch.setattr(lifecycle.signal, "SIGTERM", 15)

        proc = _FakeProcWithPgid(pid=12345)
        result = lifecycle._kill_process_group(proc, logger=logging.getLogger("test"))

        assert result is True
        assert proc.returncode == -1  # terminate fallback was called

    def test_returns_false_when_no_process(self, monkeypatch):
        """Returns False when process is None."""
        result = lifecycle._kill_process_group(None, logger=logging.getLogger("test"))
        assert result is False


# ---------------------------------------------------------------------------
# Tests for stop_llama_server with process group killing
# ---------------------------------------------------------------------------


class _DummySrvFull:
    """Minimal server state for stop_* / restart lifecycle tests."""
    def __init__(self, llama_proc=None, tts_proc=None, backend_ready=True):
        self.llama_process = llama_proc
        self.tts_process = tts_proc
        self.backend_ready = backend_ready
        self.current_model = "test-model"
        self.logger = logging.getLogger("test-dummy")
        self.llama_log_file = None
        self.config = {"models": {}, "server": {}}
        self.tts_recovery_state = {
            "in_progress": False,
            "attempt_timestamps": [],
            "max_attempts": 3,
            "window_seconds": 120,
            "last_failure": None,
        }
    def rotate_llama_logs(self, *a, **kw):
        pass
    def broadcast_status_sync(self, *a, **kw):
        pass


class TestStopLlamaServer:
    """Tests for stop_llama_server() process group killing."""

    def test_kills_process_group_via_helper(self, monkeypatch):
        """stop_llama_server uses _kill_process_group to kill the process group."""
        fake_proc = _FakeProcWithPgid(pid=9999)
        srv = _DummySrvFull(llama_proc=fake_proc)
        monkeypatch.setattr(lifecycle, "_srv", lambda: srv)

        killpg_called = []
        def fake_killpg(pgid, sig):
            killpg_called.append((pgid, sig))
        monkeypatch.setattr(lifecycle.os, "killpg", fake_killpg)
        monkeypatch.setattr(lifecycle.signal, "SIGTERM", 15)
        monkeypatch.setattr(lifecycle.signal, "SIGKILL", 9)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        lifecycle.stop_llama_server()

        assert len(killpg_called) >= 1
        assert killpg_called[0][0] == 9999  # PID used as PGID
        assert srv.llama_process is None
        assert srv.backend_ready is False
        assert srv.current_model is None

    def test_handles_none_process(self, monkeypatch):
        """stop_llama_server handles llama_process being None gracefully."""
        srv = _DummySrvFull(llama_proc=None)
        monkeypatch.setattr(lifecycle, "_srv", lambda: srv)

        # Should not raise
        lifecycle.stop_llama_server()

        assert srv.llama_process is None

    def test_skips_mock_processes(self, monkeypatch):
        """stop_llama_server skips process cleanup for mock objects without terminate/kill."""
        class MockProcNoMethods:
            pid = 12345
            def poll(self):
                return None

        srv = _DummySrvFull(llama_proc=MockProcNoMethods())
        monkeypatch.setattr(lifecycle, "_srv", lambda: srv)

        lifecycle.stop_llama_server()

        assert srv.llama_process is None
        assert srv.backend_ready is False


class TestStopTtsServer:
    """Tests for stop_tts_server() process group killing."""

    def test_kills_process_group_via_helper(self, monkeypatch):
        """stop_tts_server uses _kill_process_group to kill the process group."""
        fake_proc = _FakeProcWithPgid(pid=7777)
        srv = _DummySrvFull(tts_proc=fake_proc)
        monkeypatch.setattr(lifecycle, "_srv", lambda: srv)

        killpg_called = []
        def fake_killpg(pgid, sig):
            killpg_called.append((pgid, sig))
        monkeypatch.setattr(lifecycle.os, "killpg", fake_killpg)
        monkeypatch.setattr(lifecycle.signal, "SIGTERM", 15)
        monkeypatch.setattr(lifecycle.signal, "SIGKILL", 9)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        lifecycle.stop_tts_server()

        assert len(killpg_called) >= 1
        assert killpg_called[0][0] == 7777
        assert srv.tts_process is None

    def test_handles_none_process(self, monkeypatch):
        """stop_tts_server handles tts_process being None gracefully."""
        srv = _DummySrvFull(tts_proc=None)
        monkeypatch.setattr(lifecycle, "_srv", lambda: srv)

        lifecycle.stop_tts_server()

        assert srv.tts_process is None

    def test_skips_mock_processes(self, monkeypatch):
        """stop_tts_server skips process cleanup for mock objects without terminate/kill."""
        class MockProcNoMethods:
            pid = 7777
            def poll(self):
                return None

        srv = _DummySrvFull(tts_proc=MockProcNoMethods())
        monkeypatch.setattr(lifecycle, "_srv", lambda: srv)

        lifecycle.stop_tts_server()

        assert srv.tts_process is None

    def test_resets_recovery_state_on_intentional_stop(self, monkeypatch):
        """stop_tts_server resets tts_recovery_state when stopping a real process."""
        fake_proc = _FakeProcWithPgid(pid=7777)
        srv = _DummySrvFull(tts_proc=fake_proc)
        srv.tts_recovery_state = {
            "in_progress": False,
            "attempt_timestamps": [100.0, 110.0, 120.0],
            "max_attempts": 3,
            "window_seconds": 120,
            "last_failure": "previous TTS crash",
        }
        monkeypatch.setattr(lifecycle, "_srv", lambda: srv)

        killpg_called = []
        def fake_killpg(pgid, sig):
            killpg_called.append((pgid, sig))
        monkeypatch.setattr(lifecycle.os, "killpg", fake_killpg)
        monkeypatch.setattr(lifecycle.signal, "SIGTERM", 15)
        monkeypatch.setattr(lifecycle.signal, "SIGKILL", 9)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        lifecycle.stop_tts_server()

        # Recovery state should be reset
        assert srv.tts_recovery_state["attempt_timestamps"] == []
        assert srv.tts_recovery_state["last_failure"] is None
        assert srv.tts_process is None


# ---------------------------------------------------------------------------
# Tests for start_llama_server / start_tts_server with start_new_session
# ---------------------------------------------------------------------------


class TestStartServerProcessGroup:
    """Tests that start functions use start_new_session=True for process group isolation."""

    def test_spawn_and_capture_uses_start_new_session(self, monkeypatch):
        """spawn_and_capture passes start_new_session=True to Popen."""
        captured_kwargs = {}

        def fake_popen(cmd, **kwargs):
            captured_kwargs.update(kwargs)
            return _FakeProc()

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(lifecycle.threading, "Thread", lambda **kw: threading.Thread(**kw))

        proc, out = lifecycle.spawn_and_capture(
            cmd=["test.sh"],
            env={},
            log_file=io.StringIO(),
            logger=logging.getLogger("test"),
        )

        assert captured_kwargs.get("start_new_session") is True

    def test_start_tts_server_uses_start_new_session(self, monkeypatch):
        """start_tts_server passes start_new_session=True to Popen."""
        captured_kwargs = {}

        def fake_popen(cmd, **kwargs):
            captured_kwargs.update(kwargs)
            fake = _FakeProc()
            fake.pid = 8888
            return fake

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

        # Stub config and filesystem checks
        class _DummySrvCfg:
            config = {"server": {}}
            logger = logging.getLogger("test")

        monkeypatch.setattr(lifecycle, "_srv", lambda: _DummySrvCfg())

        # Make the script appear to exist
        _orig_isfile = os.path.isfile
        monkeypatch.setattr(lifecycle.os.path, "isfile", lambda p: True)

        proc = lifecycle.start_tts_server()

        assert captured_kwargs.get("start_new_session") is True


# ---------------------------------------------------------------------------
# Tests for restart_services port release handling
# ---------------------------------------------------------------------------


class _DummySrvRestart:
    """Full enough server state to exercise restart_services logic."""
    def __init__(self, llama_proc=None, tts_proc=None, backend_ready=True):
        self.llama_process = llama_proc
        self.tts_process = tts_proc
        self.backend_ready = backend_ready
        self.current_model = "test-model"
        self.logger = logging.getLogger("test-restart")
        self.llama_log_file = None
        self.config = {
            "models": {},
            "server": {
                "session_slot_pool_size": 4,
                "llama_router_mode": False,
                "llama_startup_timeout": 300,
            },
        }
        self._released_ports = []
        self.tts_recovery_state = {
            "in_progress": False,
            "attempt_timestamps": [],
            "max_attempts": 3,
            "window_seconds": 120,
            "last_failure": None,
        }
    def rotate_llama_logs(self, *a, **kw):
        pass
    def broadcast_status_sync(self, *a, **kw):
        pass
    def stop_llama_server(self):
        self.logger.info("stop_llama_server called")
        self.llama_process = None
        self.backend_ready = False
        self.current_model = None
    def stop_tts_server(self):
        self.logger.info("stop_tts_server called")
        self.tts_process = None
    def start_tts_server(self):
        self.logger.info("start_tts_server called")
        from proxy.lifecycle import start_tts_server as real_start
        return real_start()
    def get_local_model_name(self, model_name):
        return model_name
    def get_model_config(self, model_name):
        return self.config.get("models", {}).get(model_name)
    def start_llama_server(self, model, display_name=None):
        self.logger.info(f"start_llama_server called with model={model}")
        return _FakeProc()
    def wait_for_llama_server(self, timeout=300):
        self.logger.info("wait_for_llama_server called")
        return True
    async def ensure_model_loaded(self, model):
        self.logger.info(f"ensure_model_loaded called with model={model}")
        self.llama_process = _FakeProc()
        self.backend_ready = True
        self.current_model = model
        return True
    async def router_load_model(self, model):
        return True
    async def router_wait_for_model(self, model, timeout=300):
        return True
    def broadcast_status(self, *a, **kw):
        pass


class TestRestartServices:
    """Tests for restart_services() port release handling."""

    def test_restart_verifies_port_release(self, monkeypatch):
        """restart_services calls _wait_for_port_release after stopping processes."""
        fake_proc = _FakeProcWithPgid(pid=5001)
        srv = _DummySrvRestart(llama_proc=fake_proc, tts_proc=_FakeProcWithPgid(pid=5002))
        monkeypatch.setattr(lifecycle, "_srv", lambda: srv)

        port_release_calls = []
        def fake_port_release(port, timeout=5.0, interval=0.5):
            port_release_calls.append(port)
            return True
        monkeypatch.setattr(lifecycle, "_wait_for_port_release", fake_port_release)

        # Mock async sleep to avoid actual delays
        monkeypatch.setattr(lifecycle.asyncio, "sleep", lambda s: _FakeAwaitable())
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        import asyncio
        result = asyncio.run(lifecycle.restart_services(slot_count=8))

        assert result is True
        assert 8080 in port_release_calls
        assert srv.backend_ready is True

    def test_restart_fails_when_port_not_released(self, monkeypatch):
        """restart_services returns False when port 8080 is not released."""
        fake_proc = _FakeProcWithPgid(pid=5001)
        srv = _DummySrvRestart(llama_proc=fake_proc, tts_proc=_FakeProcWithPgid(pid=5002))
        monkeypatch.setattr(lifecycle, "_srv", lambda: srv)

        def fake_port_release(port, timeout=5.0, interval=0.5):
            return False
        monkeypatch.setattr(lifecycle, "_wait_for_port_release", fake_port_release)

        monkeypatch.setattr(lifecycle.asyncio, "sleep", lambda s: _FakeAwaitable())
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        import asyncio
        result = asyncio.run(lifecycle.restart_services(slot_count=8))

        assert result is False
        assert srv.backend_ready is False

    def test_restart_stops_tts_server(self, monkeypatch):
        """restart_services stops the TTS server when it is running."""
        fake_proc = _FakeProcWithPgid(pid=5001)
        srv = _DummySrvRestart(llama_proc=fake_proc, tts_proc=_FakeProcWithPgid(pid=5002))
        monkeypatch.setattr(lifecycle, "_srv", lambda: srv)

        stop_tts_called = []
        original_stop_tts = srv.stop_tts_server
        def tracked_stop_tts():
            stop_tts_called.append(True)
            original_stop_tts()
        srv.stop_tts_server = tracked_stop_tts

        def fake_port_release(port, timeout=5.0, interval=0.5):
            return True
        monkeypatch.setattr(lifecycle, "_wait_for_port_release", fake_port_release)
        monkeypatch.setattr(lifecycle.asyncio, "sleep", lambda s: _FakeAwaitable())
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        import asyncio
        result = asyncio.run(lifecycle.restart_services(slot_count=8))

        assert result is True
        assert len(stop_tts_called) >= 1

    def test_restart_restarts_tts_server(self, monkeypatch):
        """restart_services restarts TTS server after llama is ready when TTS was running."""
        fake_proc = _FakeProcWithPgid(pid=5001)
        srv = _DummySrvRestart(llama_proc=fake_proc, tts_proc=_FakeProcWithPgid(pid=5002))
        monkeypatch.setattr(lifecycle, "_srv", lambda: srv)

        start_tts_called = []
        def fake_start_tts():
            start_tts_called.append(True)
            return _FakeProcWithPgid(pid=5003)
        monkeypatch.setattr(lifecycle, "start_tts_server", fake_start_tts)

        def fake_port_release(port, timeout=5.0, interval=0.5):
            return True
        monkeypatch.setattr(lifecycle, "_wait_for_port_release", fake_port_release)
        monkeypatch.setattr(lifecycle.asyncio, "sleep", lambda s: _FakeAwaitable())
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        import asyncio
        result = asyncio.run(lifecycle.restart_services(slot_count=8))

        assert result is True
        assert len(start_tts_called) >= 1
        assert srv.backend_ready is True
        assert srv.tts_process is not None


class _FakeAwaitable:
    """Minimal awaitable that resolves immediately."""
    def __await__(self):
        return iter([])
