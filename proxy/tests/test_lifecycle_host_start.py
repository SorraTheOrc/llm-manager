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


def test_host_start_success(monkeypatch, tmp_path):
    # Host-start allowed and host command spawns a long-running process
    dummy = DummySrv({"server": {"llama_allow_host_fallback": True}})
    monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)

    def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
        # Simulate a long-running host process when start-llama.sh is invoked
        if isinstance(cmd, list) and cmd[0].endswith("start-llama.sh"):
            return FakeProc()
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)
    # Ensure distrobox exists to avoid early failure
    monkeypatch.setattr(lifecycle.shutil, "which", lambda name: "/usr/bin/distrobox" if name == "distrobox" else None)

    proc = lifecycle.start_llama_server("mymodel")
    assert proc is not None
    assert hasattr(proc, "communicate")


def test_host_start_fails_fallback_to_distrobox(monkeypatch):
    # Host-start attempted but FileNotFoundError; distrobox path succeeds
    dummy = DummySrv({"server": {"llama_allow_host_fallback": True}})
    monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)

    def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
        # If distrobox is invoked, return a running process, otherwise raise
        if isinstance(cmd, list) and cmd[0] == "distrobox":
            return FakeProc()
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(lifecycle.shutil, "which", lambda name: "/usr/bin/distrobox" if name == "distrobox" else None)

    proc = lifecycle.start_llama_server("mymodel")
    assert proc is not None
    assert hasattr(proc, "communicate")


def test_distrobox_missing_and_no_host_fallback(monkeypatch):
    # No distrobox installed and host-fallback disabled -> start fails
    dummy = DummySrv({"server": {"llama_allow_host_fallback": False}})
    monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)

    # Ensure distrobox which() returns None
    monkeypatch.setattr(lifecycle.shutil, "which", lambda name: None)

    # Popen should not be called; but if it is, raise to fail loudly
    def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
        raise RuntimeError("unexpected popen")

    monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

    proc = lifecycle.start_llama_server("mymodel")
    assert proc is None
