"""Integration tests for the host-first startup flow.

These tests validate the proxy startup lifecycle using mocked subprocess
calls — no GPU or real llama-server required.

Live end-to-end tests are in test_host_flow_live_e2e.py (run with
RUN_LIVE_HOST_FLOW=1).

Run:
    pytest tests/test_host_flow_integration.py -v
"""

import io
import logging
import subprocess
import sys
from pathlib import Path


# Ensure the package path is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib
lifecycle = importlib.import_module("proxy.lifecycle")


# ── Helpers ──────────────────────────────────────────────────────────────────

class FakeProc:
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


class DummySrv:
    """Minimal server state for start_llama_server tests."""
    def __init__(self, config_overrides=None):
        cfg = {"server": {
            "llama_allow_host_fallback": True,
            **(config_overrides or {}),
        }}
        self.config = cfg
        self.logger = logging.getLogger("dummy")
        self.log_dir = None
        self.llama_log_file = None
        self.last_start_failure = None
        self.current_model = None
        self.backend_ready = False
        self.llama_process = None
    def rotate_llama_logs(self, *a, **kw):
        pass
    def broadcast_status_sync(self, *a, **kw):
        pass


# ── Mocked Tests ─────────────────────────────────────────────────────────────

class TestHostFlowMocked:
    """Mocked integration tests that validate startup logic without GPU."""

    def test_host_startup_success(self, monkeypatch):
        """Host-first startup: host-start succeeds → returns proc immediately."""
        dummy = DummySrv()
        monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        call_count = 0
        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
            nonlocal call_count
            call_count += 1
            if "start-llama.sh" in cmd[0]:
                return FakeProc()
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

        proc = lifecycle.start_llama_server("test-model")
        assert proc is not None
        assert call_count == 1

    def test_host_fallback_to_container(self, monkeypatch):
        """Host-first startup: host-start fails → falls back to configured script."""
        dummy = DummySrv()
        monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        call_count = 0
        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
            nonlocal call_count
            call_count += 1
            if "podman" in cmd[0] or call_count > 1:
                return FakeProc()
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

        proc = lifecycle.start_llama_server("test-model")
        assert proc is not None
        assert call_count >= 2

    def test_host_fallback_all_fail(self, monkeypatch):
        """Both host-start and configured script fail → returns None with error."""
        dummy = DummySrv()
        monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        def failing_popen(cmd, env=None, stdout=None, stderr=None, text=None):
            raise FileNotFoundError(f"{cmd[0]} not found")

        monkeypatch.setattr(lifecycle.subprocess, "Popen", failing_popen)

        proc = lifecycle.start_llama_server("test-model")
        assert proc is None
        assert dummy.last_start_failure is not None
        assert "Failed to start llama-server" in dummy.last_start_failure

    def test_router_mode_startup(self, monkeypatch):
        """Router mode (model=None) uses 'router' argument for host-start."""
        dummy = DummySrv()
        monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        host_cmds = []
        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
            if "start-llama.sh" in cmd[0]:
                host_cmds.append(cmd)
                return FakeProc()
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

        proc = lifecycle.start_llama_server(None)  # router mode
        assert proc is not None
        assert len(host_cmds) == 1
        assert host_cmds[0][1] == "router"

    def test_llama_allow_host_fallback_disabled(self, monkeypatch):
        """When llama_allow_host_fallback is false, no separate host-start attempt."""
        dummy = DummySrv({
            "llama_allow_host_fallback": False,
            "llama_start_script": "/custom/podman_start_llama.sh",
        })
        monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        cmds = []
        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
            cmds.append(cmd[0])
            return FakeProc()

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

        proc = lifecycle.start_llama_server("test-model")
        assert proc is not None
        host_attempts = [c for c in cmds if "start-llama.sh" in c]
        assert len(host_attempts) == 0

    def test_model_load_after_startup(self, monkeypatch):
        """After successful start, model loading state transitions are consistent."""
        dummy = DummySrv()
        monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
            if "start-llama.sh" in cmd[0]:
                return FakeProc()
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

        proc = lifecycle.start_llama_server("test-model")
        assert proc is not None
        assert not dummy.backend_ready
        assert dummy.llama_process is None

    def test_progress_logging_output(self):
        """Verify progress parsing produces expected output format with model/slot prefix and TPS."""
        from proxy.handlers import extract_progress_data, format_progress

        log_line = "slot 3 : prompt processing, n_tokens=100, progress=0.50"
        parsed = extract_progress_data(log_line)
        assert parsed is not None
        slot_id, n_tokens, progress = parsed
        assert slot_id == 3
        assert n_tokens == 100
        assert progress == 0.50

        formatted = format_progress(n_tokens, 200, progress,
                                     model_name="Qwen3", slot_id=slot_id, tokens_per_sec=45.2)
        assert "100" in formatted
        assert "200" in formatted
        assert "[slot:3 Qwen3]" in formatted
        assert "@ 45.2 tok/s" in formatted
