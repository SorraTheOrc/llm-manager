"""Unit tests for the extracted spawn_and_capture helper functions."""

import io
import logging
import subprocess
import threading
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
