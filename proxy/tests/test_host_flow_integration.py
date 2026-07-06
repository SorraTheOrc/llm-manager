"""Integration tests for the host-first startup flow.

These tests validate the proxy startup lifecycle: spawning llama-server,
loading models (router mode), session reuse, and progress logging.

Default mode (no env var): runs mocked tests that validate startup logic
without requiring a GPU or actual llama-server.

Live mode (env var set): starts host llama-server and proxy, performs
real e2e checks against a running backend.

Run mocked tests (default):
    pytest tests/test_host_flow_integration.py -v

Run live tests (requires GPU + llama-server):
    RUN_LIVE_HOST_FLOW=1 pytest tests/test_host_flow_integration.py -v -m live
"""

import asyncio
import io
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


# ── Mocked Tests (run by default) ────────────────────────────────────────────

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
            # Only the host-start script (start-llama.sh) should succeed
            if "start-llama.sh" in cmd[0]:
                return FakeProc()
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

        proc = lifecycle.start_llama_server("test-model")
        assert proc is not None
        # Only 1 call — host-start succeeded, no fallback
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
            # Host-start fails (start-llama.sh not found), configured script succeeds
            if "podman" in cmd[0] or call_count > 1:
                return FakeProc()
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

        proc = lifecycle.start_llama_server("test-model")
        assert proc is not None
        # At least 2 calls: 1 host-start attempt + 1+ retry attempts
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
        # Use a non-default configured script so the host-start path is distinct
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
        # Should NOT have attempted start-llama.sh (host-start) since fallback is false
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
        # After startup, backend_ready should still be False (wait_for_llama_server
        # hasn't been called yet in the mocked path — that's the caller's responsibility)
        assert not dummy.backend_ready
        assert dummy.llama_process is None  # start_llama_server returns proc but doesn't store it

    def test_progress_logging_output(self):
        """Verify progress parsing produces expected output format."""
        from proxy.handlers import extract_progress_data, format_progress

        # Simulate a progress log line from llama-server
        log_line = "prompt processing, n_tokens=100, progress=0.50"
        parsed = extract_progress_data(log_line)
        assert parsed is not None
        n_tokens, progress = parsed
        assert n_tokens == 100
        assert progress == 0.50

        # Format the progress
        formatted = format_progress(n_tokens, 200, progress)
        assert "100" in formatted
        assert "200" in formatted
        assert "50" in formatted or "50.0" in formatted or "50%" in formatted


# ── Live Integration Tests (run with RUN_LIVE_HOST_FLOW=1) ───────────────────

LIVE_FLAG = os.environ.get("RUN_LIVE_HOST_FLOW", "").strip() in ("1", "true", "yes")
LIVE_PROXY_URL = os.environ.get("LIVE_PROXY_BASE_URL", "http://localhost:8000")
LIVE_LLAMA_URL = os.environ.get("LIVE_LLAMA_BASE_URL", "http://localhost:8080")


@pytest.mark.skipif(not LIVE_FLAG, reason="Set RUN_LIVE_HOST_FLOW=1 to run live tests")
@pytest.mark.live
class TestHostFlowLive:
    """Live integration tests against a running llama-server and proxy.

    These tests require:
    - A running llama-server on port 8080 (or LIVE_LLAMA_BASE_URL)
    - A running proxy on port 8000 (or LIVE_PROXY_BASE_URL)
    - GPU with ROCm drivers
    """

    def test_llama_server_health(self):
        """llama-server health endpoint is reachable."""
        import requests
        resp = requests.get(f"{LIVE_LLAMA_URL}/health", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_proxy_health(self):
        """Proxy health endpoint is reachable."""
        import requests
        resp = requests.get(f"{LIVE_PROXY_URL}/health", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        # Proxy should report it can reach the backend
        # (actual key depends on proxy implementation, just check it responds)

    def test_embedding_request(self):
        """A simple embedding request returns 200."""
        import requests
        payload = {
            "model": "embeddings",
            "input": "Hello, world!"
        }
        resp = requests.post(
            f"{LIVE_PROXY_URL}/v1/embeddings",
            json=payload,
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) > 0
        assert "embedding" in data["data"][0]

    def test_chat_completion(self):
        """A simple chat completion request returns 200."""
        import requests
        payload = {
            "model": "Qwen3",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 50,
        }
        resp = requests.post(
            f"{LIVE_PROXY_URL}/v1/chat/completions",
            json=payload,
            timeout=60,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"] is not None
