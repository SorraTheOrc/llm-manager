import io
import subprocess
import logging
import sys
from pathlib import Path

# Ensure the package path is on sys.path so `import proxy.proxy.lifecycle` works
# when running tests from the project root. We add the `proxy/` directory
# (the project package root) so Python finds the nested `proxy` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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


# ---- Host-fallback tests ----

def test_host_fallback_default_disabled(monkeypatch):
    """When llama_allow_host_fallback is false (default), only the configured script is used."""
    # Configure via llama_start_script pointing to a podman wrapper, but the popen
    # only accepts the configured script path (not start-llama.sh in repo root).
    dummy = DummySrv({"server": {
        "llama_start_script": "/custom/podman_start_llama.sh",
        "llama_allow_host_fallback": False,
    }})
    monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

    started_cmds = []
    def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
        if isinstance(cmd, list) and cmd[0] == "/custom/podman_start_llama.sh":
            started_cmds.append(cmd)
            return FakeProc()
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

    proc = lifecycle.start_llama_server("mymodel")
    assert proc is not None
    # Should only have started the configured script, not start-llama.sh
    assert len(started_cmds) == 1
    assert started_cmds[0][0] == "/custom/podman_start_llama.sh"


def test_host_fallback_enabled_succeeds(monkeypatch):
    """When host-fallback is enabled and host-start succeeds, return immediately."""
    dummy = DummySrv({"server": {
        "llama_start_script": "/custom/podman_start_llama.sh",
        "llama_allow_host_fallback": True,
    }})
    monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

    started_cmds = []
    def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
        if isinstance(cmd, list) and cmd[0].endswith("start-llama.sh"):
            started_cmds.append(cmd)
            return FakeProc()
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

    proc = lifecycle.start_llama_server("mymodel")
    assert proc is not None
    assert len(started_cmds) == 1
    assert "start-llama.sh" in started_cmds[0][0]


def test_host_fallback_enabled_fails_then_fallback(monkeypatch):
    """When host-fallback is enabled but host-start fails, fall through to configured script."""
    dummy = DummySrv({"server": {
        "llama_start_script": "/custom/podman_start_llama.sh",
        "llama_allow_host_fallback": True,
    }})
    monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

    started_cmds = []
    def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
        started_cmds.append(cmd[0])
        if isinstance(cmd, list) and cmd[0] == "/custom/podman_start_llama.sh":
            return FakeProc()
        raise FileNotFoundError(f"{cmd[0]} not found")

    monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

    proc = lifecycle.start_llama_server("mymodel")
    assert proc is not None
    # Should have tried host-start (start-llama.sh) first, then fallen through
    # to the configured script. The retry loop should find the configured script.
    host_attempts = [c for c in started_cmds if "start-llama.sh" in c]
    configured_attempts = [c for c in started_cmds if c == "/custom/podman_start_llama.sh"]
    assert len(host_attempts) >= 1, "Should have attempted host-start first"
    assert len(configured_attempts) >= 1, "Should have fallen back to configured script"


def test_host_fallback_router_mode(monkeypatch):
    """Host-fallback in router mode (model=None) passes 'router' argument."""
    dummy = DummySrv({"server": {
        "llama_start_script": "/custom/podman_start_llama.sh",
        "llama_allow_host_fallback": True,
    }})
    monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

    host_cmds = []
    def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
        if isinstance(cmd, list) and cmd[0].endswith("start-llama.sh"):
            host_cmds.append(cmd)
            return FakeProc()
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

    proc = lifecycle.start_llama_server(None)  # router mode
    assert proc is not None
    assert len(host_cmds) == 1
    assert host_cmds[0][1] == "router"
