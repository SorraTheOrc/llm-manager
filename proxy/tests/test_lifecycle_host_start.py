import io
import subprocess
import threading
import types
import logging
import sys
from pathlib import Path

# Ensure the package path is on sys.path so `import proxy.proxy.lifecycle` works
# when running tests from the project root. We add the `proxy/` directory
# (the project package root) so Python finds the nested `proxy` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

# Import the lifecycle module via the package path. Add the containing
# `proxy/` directory to sys.path so the `proxy` package resolves correctly.
import importlib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Import as package.module so relative imports inside the package work.
lifecycle = importlib.import_module("proxy.lifecycle")


class DummySrv:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("dummy")
        self.log_dir = None
        self.llama_log_file = None
        self.last_start_failure = None
    def rotate_llama_logs(self, *a, **kw):
        pass
    def broadcast_status_sync(self, *a, **kw):
        pass


class FakeProc:
    def __init__(self, *a, **kw):
        self.returncode = None
        self.stdout = io.StringIO()
    def communicate(self, timeout=None):
        raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
    def poll(self):
        return None
    def terminate(self):
        self.returncode = -1
    def wait(self, timeout=None):
        return


def test_start_script_success(monkeypatch):
    """Start script spawns a long-running process successfully."""
    dummy = DummySrv({"server": {}})
    monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)  # No-op sleep

    def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
        if isinstance(cmd, list) and len(cmd) > 0 and cmd[0].endswith("start-llama.sh"):
            return FakeProc()
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

    proc = lifecycle.start_llama_server("mymodel")
    assert proc is not None
    assert hasattr(proc, "communicate")


def test_start_script_fails_after_retries(monkeypatch):
    """When the start script fails repeatedly, start_llama_server returns None and sets last_start_failure."""
    dummy = DummySrv({"server": {}})
    monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)  # No-op sleep

    def failing_popen(cmd, env=None, stdout=None, stderr=None, text=None):
        raise FileNotFoundError(f"{cmd[0]} not found")

    monkeypatch.setattr(lifecycle.subprocess, "Popen", failing_popen)

    proc = lifecycle.start_llama_server("mymodel")
    assert proc is None
    assert dummy.last_start_failure is not None
    assert "Failed to start llama-server" in dummy.last_start_failure
