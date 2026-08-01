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
    _gpu_power_draw_rocm_smi,
    _gpu_power_draw_sysfs,
    _gpu_wedge_discriminate,
    _probe_gpu_wedge,
    _read_gpu_busy_percent,
    _read_gpu_power_draw,
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
# _gpu_wedge_discriminate
#
# LP-0MS9GA6QL007XTBP: on the Strix Halo APU (Radeon 8060S iGPU, gfx1151)
# gpu_busy_percent is pinned at 100% regardless of actual GPU state
# (30/30 samples read 100 while power draw showed the GPU idle at ~54W vs
# ~115W when genuinely computing). The old busy+idle signature therefore
# fired on every idle period, causing constant unload/reload churn.
#
# Power draw is the discriminating signal: a busy counter with LOW power
# means the GPU is NOT genuinely computing - either the counter is pinned/
# unreliable (APU idle) or the engine is stalled at idle power (the original
# LP-0MS91D782006XIR6 wedge). The two are indistinguishable from GPU
# sensors, so this state is classified "pinned" and never triggers an
# unload/reload; only busy+idle with compute-level power (or no power
# source, preserving legacy behavior) is a genuine "wedge".
# ===================================================================

class TestGpuWedgeDiscriminate:
    def test_busy_and_idle_is_wedge(self):
        # Legacy signature, no power source available.
        assert _gpu_wedge_discriminate(100.0, True, None, 90.0, 90.0) == "wedge"

    def test_busy_but_processing_not_wedge(self):
        assert _gpu_wedge_discriminate(100.0, False, 54.0, 90.0, 90.0) == "none"

    def test_idle_but_gpu_not_busy_not_wedge(self):
        assert _gpu_wedge_discriminate(5.0, True, 54.0, 90.0, 90.0) == "none"

    def test_below_threshold_not_wedge(self):
        assert _gpu_wedge_discriminate(89.0, True, 110.0, 90.0, 90.0) == "none"

    def test_at_threshold_is_wedge(self):
        assert _gpu_wedge_discriminate(90.0, True, 110.0, 90.0, 90.0) == "wedge"

    def test_unknown_busy_not_wedge(self):
        assert _gpu_wedge_discriminate(None, True, 54.0, 90.0, 90.0) == "none"

    def test_pinned_at_100_busy_low_power_is_pinned_not_wedge(self):
        # Critical APU regression case: busy counter pinned at 100% with idle
        # slots and low power draw must NOT be treated as a wedge.
        assert _gpu_wedge_discriminate(100.0, True, 54.0, 90.0, 90.0) == "pinned"
        assert _gpu_wedge_discriminate(100.0, True, 53.0, 90.0, 90.0) == "pinned"

    def test_busy_idle_compute_level_power_is_wedge(self):
        # GPU drawing real compute power (110-115W during genuine work) while
        # all slots are idle and busy is high -> genuinely anomalous.
        assert _gpu_wedge_discriminate(100.0, True, 110.0, 90.0, 90.0) == "wedge"
        assert _gpu_wedge_discriminate(100.0, True, 115.0, 90.0, 90.0) == "wedge"

    def test_power_at_threshold_counts_as_compute(self):
        assert _gpu_wedge_discriminate(95.0, True, 90.0, 90.0, 90.0) == "wedge"

    def test_power_just_below_threshold_is_pinned(self):
        assert _gpu_wedge_discriminate(95.0, True, 89.9, 90.0, 90.0) == "pinned"


# ===================================================================
# GPU power draw sources
# ===================================================================

class TestGpuPowerDraw:
    def test_sysfs_reads_microwatts_as_watts(self, tmp_path):
        fake_hwmon = tmp_path / "hwmon" / "hwmon5"
        fake_hwmon.mkdir(parents=True)
        (fake_hwmon / "name").write_text("amdgpu\n")
        (fake_hwmon / "power1_average").write_text("55069000\n")
        val = _gpu_power_draw_sysfs(base=str(tmp_path))
        assert val == pytest.approx(55.069)

    def test_sysfs_missing_power_node_returns_none(self, tmp_path):
        fake_hwmon = tmp_path / "hwmon" / "hwmon5"
        fake_hwmon.mkdir(parents=True)
        (fake_hwmon / "name").write_text("amdgpu\n")
        assert _gpu_power_draw_sysfs(base=str(tmp_path)) is None

    def test_sysfs_missing_hwmon_dir_returns_none(self, tmp_path):
        assert _gpu_power_draw_sysfs(base=str(tmp_path / "does-not-exist")) is None

    def test_sysfs_skips_non_power_hwmon_nodes(self, tmp_path):
        (tmp_path / "hwmon" / "hwmon1").mkdir(parents=True)
        (tmp_path / "hwmon" / "hwmon1" / "name").write_text("npu\n")
        (tmp_path / "hwmon" / "hwmon1" / "temp1_input").write_text("54000\n")
        fake_hwmon = tmp_path / "hwmon" / "hwmon2"
        fake_hwmon.mkdir()
        (fake_hwmon / "name").write_text("amdgpu\n")
        (fake_hwmon / "power1_average").write_text("115000000\n")
        assert _gpu_power_draw_sysfs(base=str(tmp_path)) == pytest.approx(115.0)

    def test_rocm_smi_parses_showpower_output(self, monkeypatch):
        fake_run = MagicMock()
        fake_run.stdout = (
            "GPU[0]\t\t: Current Socket Graphics Package Power (W): 58.023\n"
        )
        monkeypatch.setattr(
            "proxy.backends.llama.subprocess.run",
            MagicMock(return_value=fake_run),
        )
        assert _gpu_power_draw_rocm_smi() == pytest.approx(58.023)

    def test_rocm_smi_missing_binary_returns_none(self, monkeypatch):
        def raise_fnf(*args, **kwargs):
            raise FileNotFoundError("rocm-smi")

        monkeypatch.setattr(
            "proxy.backends.llama.subprocess.run", raise_fnf
        )
        assert _gpu_power_draw_rocm_smi() is None

    def test_sysfs_used_when_available(self, monkeypatch, tmp_path):
        fake_hwmon = tmp_path / "hwmon" / "hwmon5"
        fake_hwmon.mkdir(parents=True)
        (fake_hwmon / "power1_average").write_text("115000000\n")
        monkeypatch.setattr(
            "proxy.backends.llama._gpu_power_draw_sysfs",
            lambda base=None: _gpu_power_draw_sysfs(base=str(tmp_path)),
        )
        monkeypatch.setattr(
            "proxy.backends.llama._gpu_power_draw_rocm_smi", lambda: 0.0
        )
        assert _read_gpu_power_draw() == pytest.approx(115.0)

    def test_rocm_smi_fallback_when_sysfs_missing(self, monkeypatch):
        monkeypatch.setattr(
            "proxy.backends.llama._gpu_power_draw_sysfs", lambda base=None: None
        )
        monkeypatch.setattr(
            "proxy.backends.llama._gpu_power_draw_rocm_smi", lambda: 42.0
        )
        assert _read_gpu_power_draw() == 42.0

    def test_none_when_both_sources_missing(self, monkeypatch):
        monkeypatch.setattr(
            "proxy.backends.llama._gpu_power_draw_sysfs", lambda base=None: None
        )
        monkeypatch.setattr(
            "proxy.backends.llama._gpu_power_draw_rocm_smi", lambda: None
        )
        assert _read_gpu_power_draw() is None


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
    async def test_returns_busy_power_and_slots_state(self, monkeypatch):
        monkeypatch.setattr(
            "proxy.backends.llama._read_gpu_busy_percent", lambda: 100.0
        )
        monkeypatch.setattr(
            "proxy.backends.llama._read_gpu_power_draw", lambda: 54.0
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
        busy, idle, power = await _probe_gpu_wedge("127.0.0.1", 8080)
        assert busy == 100.0
        assert idle is True
        assert power == 54.0

    @pytest.mark.asyncio
    async def test_slots_failure_keeps_idle_false(self, monkeypatch):
        monkeypatch.setattr(
            "proxy.backends.llama._read_gpu_busy_percent", lambda: 100.0
        )
        monkeypatch.setattr(
            "proxy.backends.llama._read_gpu_power_draw", lambda: None
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
        busy, idle, power = await _probe_gpu_wedge("127.0.0.1", 8080)
        assert busy == 100.0
        assert idle is False
        assert power is None


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
    # MagicMock auto-creates truthy attributes, which would flip the /health
    # gpu_wedge_detection_disabled field; default it to False explicitly.
    srv.gpu_wedge_detection_disabled = False
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


def test_health_reports_wedge_detection_disabled():
    srv = _make_server_mock(gpu_wedge_detected=False)
    srv.gpu_wedge_detection_disabled = True
    result = _call_health_check(srv)
    assert result["gpu_wedge_detection_disabled"] is True
    assert result["status"] == "healthy"


def test_health_wedge_detection_not_disabled_by_default():
    srv = _make_server_mock(gpu_wedge_detected=False)
    result = _call_health_check(srv)
    assert result["gpu_wedge_detection_disabled"] is False
