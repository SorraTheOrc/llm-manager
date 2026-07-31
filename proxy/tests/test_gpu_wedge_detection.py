"""
Tests for GPU-wedge detection and self-healing (LP-0MS91DHQ9003EGB0).

A wedged llama-server is still HTTP-reachable but its GPU kernels never
complete: gpu_busy_percent reads ~100 while every slot is idle and no
tokens are produced. These tests cover the pure decision helpers and the
/health endpoint reporting; the model-health loop integration is exercised
by the helpers it composes.
"""

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from proxy.backends.llama import (
    _gpu_busy_percent_rocm_smi,
    _gpu_wedge_signature,
    _probe_gpu_wedge,
    _read_gpu_busy_percent,
    _slots_all_idle,
)


# ===================================================================
# _slots_all_idle
# ===================================================================

class TestSlotsAllIdle:
    def test_all_idle(self):
        payload = [
            {"id": 0, "is_processing": False},
            {"id": 1, "is_processing": False},
        ]
        assert _slots_all_idle(payload) is True

    def test_any_processing(self):
        payload = [
            {"id": 0, "is_processing": False},
            {"id": 1, "is_processing": True},
        ]
        assert _slots_all_idle(payload) is False

    def test_empty_list(self):
        assert _slots_all_idle([]) is True

    def test_not_a_list(self):
        assert _slots_all_idle({}) is False
        assert _slots_all_idle(None) is False

    def test_missing_processing_field_counts_as_idle(self):
        payload = [{"id": 0}, {"id": 1, "is_processing": False}]
        assert _slots_all_idle(payload) is True


# ===================================================================
# _gpu_wedge_signature
# ===================================================================

class TestGpuWedgeSignature:
    def test_busy_and_idle_is_wedge(self):
        assert _gpu_wedge_signature(100.0, True, 90.0) is True

    def test_busy_but_processing_not_wedge(self):
        assert _gpu_wedge_signature(100.0, False, 90.0) is False

    def test_idle_but_gpu_not_busy_not_wedge(self):
        assert _gpu_wedge_signature(5.0, True, 90.0) is False

    def test_below_threshold_not_wedge(self):
        assert _gpu_wedge_signature(89.0, True, 90.0) is False

    def test_at_threshold_is_wedge(self):
        assert _gpu_wedge_signature(90.0, True, 90.0) is True

    def test_unknown_busy_not_wedge(self):
        assert _gpu_wedge_signature(None, True, 90.0) is False


# ===================================================================
# GPU busy sources
# ===================================================================

class TestGpuBusyPercent:
    def test_sysfs_used_when_available(self, monkeypatch):
        monkeypatch.setattr(
            "proxy.backends.llama._gpu_busy_percent_sysfs", lambda: 100.0
        )
        monkeypatch.setattr(
            "proxy.backends.llama._gpu_busy_percent_rocm_smi", lambda: 0.0
        )
        assert _read_gpu_busy_percent() == 100.0

    def test_rocm_smi_fallback_when_sysfs_missing(self, monkeypatch):
        monkeypatch.setattr(
            "proxy.backends.llama._gpu_busy_percent_sysfs", lambda: None
        )
        monkeypatch.setattr(
            "proxy.backends.llama._gpu_busy_percent_rocm_smi", lambda: 42.0
        )
        assert _read_gpu_busy_percent() == 42.0

    def test_none_when_both_sources_missing(self, monkeypatch):
        monkeypatch.setattr(
            "proxy.backends.llama._gpu_busy_percent_sysfs", lambda: None
        )
        monkeypatch.setattr(
            "proxy.backends.llama._gpu_busy_percent_rocm_smi", lambda: None
        )
        assert _read_gpu_busy_percent() is None

    def test_rocm_smi_parses_showuse_output(self, monkeypatch):
        fake_run = MagicMock()
        fake_run.stdout = (
            "GPU[0]\t\t: GPU use (%): 100\n"
            "GPU[1]\t\t: GPU use (%): 42\n"
        )
        monkeypatch.setattr(
            "proxy.backends.llama.subprocess.run",
            MagicMock(return_value=fake_run),
        )
        assert _gpu_busy_percent_rocm_smi() == 100.0

    def test_rocm_smi_missing_binary_returns_none(self, monkeypatch):
        def raise_fnf(*args, **kwargs):
            raise FileNotFoundError("rocm-smi")

        monkeypatch.setattr(
            "proxy.backends.llama.subprocess.run", raise_fnf
        )
        assert _gpu_busy_percent_rocm_smi() is None


# ===================================================================
# _probe_gpu_wedge
# ===================================================================

class TestProbeGpuWedge:
    @pytest.mark.asyncio
    async def test_returns_busy_and_slots_state(self, monkeypatch):
        monkeypatch.setattr(
            "proxy.backends.llama._read_gpu_busy_percent", lambda: 100.0
        )

        class FakeResp:
            status_code = 200

            def json(self):
                return [{"id": 0, "is_processing": False}]

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, timeout=None):
                assert "/slots" in url
                return FakeResp()

        monkeypatch.setattr("proxy.backends.llama.httpx.AsyncClient", FakeClient)
        busy, idle = await _probe_gpu_wedge("127.0.0.1", 8080)
        assert busy == 100.0
        assert idle is True

    @pytest.mark.asyncio
    async def test_slots_failure_keeps_idle_false(self, monkeypatch):
        monkeypatch.setattr(
            "proxy.backends.llama._read_gpu_busy_percent", lambda: 100.0
        )

        class BoomClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, timeout=None):
                raise RuntimeError("connection refused")

        monkeypatch.setattr("proxy.backends.llama.httpx.AsyncClient", BoomClient)
        busy, idle = await _probe_gpu_wedge("127.0.0.1", 8080)
        assert busy == 100.0
        assert idle is False


# ===================================================================
# /health endpoint reporting
# ===================================================================

def _make_server_mock(gpu_wedge_detected=False, gpu_wedge_signals=0):
    srv = MagicMock()
    srv.config = {
        "server": {
            "llama_router_mode": False,
            "llama_server_port": 8080,
            "tts_enabled": False,
        },
        "models": {},
    }
    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll = MagicMock(return_value=None)
    srv.llama_process = mock_proc
    srv.backend_ready = True
    srv.current_model = "Qwen3"
    srv._probe_backend_reachable = AsyncMock(return_value=True)
    srv._is_self_healing_active = MagicMock(return_value=False)
    srv._backend_recovery_snapshot = MagicMock(
        return_value={"in_progress": False, "attempt_count": 0}
    )
    srv.backend_signal_counts = {
        "connect_failures": 0,
        "read_failures": 0,
        "timeout_failures": 0,
        "other_failures": 0,
        "concurrency_rejects": 0,
        "gpu_wedge": gpu_wedge_signals,
    }
    srv.gpu_wedge_detected = gpu_wedge_detected
    srv.tts_process = None
    return srv


def _call_health_check(srv):
    import asyncio

    from proxy.handlers import health_check

    with patch("proxy.handlers._srv", return_value=srv):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(health_check())
        finally:
            loop.close()


def test_health_ready_when_no_wedge():
    srv = _make_server_mock(gpu_wedge_detected=False)
    result = _call_health_check(srv)
    assert result["status"] == "healthy"
    assert result["ready"] is True
    assert result["gpu_wedge_detected"] is False
    assert result["gpu_wedge_signal_count"] == 0


def test_health_degraded_while_wedge_detected():
    srv = _make_server_mock(gpu_wedge_detected=True, gpu_wedge_signals=1)
    result = _call_health_check(srv)
    assert result["status"] == "degraded"
    assert result["ready"] is False
    assert result["gpu_wedge_detected"] is True
    assert result["gpu_wedge_signal_count"] == 1
