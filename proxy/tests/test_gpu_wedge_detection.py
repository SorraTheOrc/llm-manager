"""
Tests for GPU-wedge detection and self-healing (LP-0MS91DHQ9003EGB0).

A wedged llama-server is still HTTP-reachable but its GPU kernels never
complete: the GPU draws compute-level power while every slot is idle and no
tokens are produced. These tests cover the pure decision helpers and the
/health endpoint reporting; the model-health loop integration is exercised
by the helpers it composes.

The wedge signature is power-only (LP-0MW5K8ZW3005SEA8):

  - compute-level power + all slots idle -> wedge
  - idle power + all slots idle          -> no wedge (the GPU is not computing)
  - missing power source                 -> no verdict (fail closed)

The gpu_busy_percent counter is not read anywhere in the module; a source
guard below keeps it that way.
"""

import logging
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from proxy.backends import llama as llama_backend
from proxy.backends.llama import (
    _gpu_power_draw_rocm_smi,
    _gpu_power_draw_sysfs,
    _gpu_wedge_discriminate,
    _probe_gpu_wedge,
    _read_gpu_power_draw,
    _slots_all_idle,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

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
# LP-0MW5K8ZW3005SEA8: the wedge signature is decided by power and slot
# state only. On the Strix Halo APU (Radeon 8060S iGPU, gfx1151) the old
# gpu_busy_percent counter is a two-state detector that inverts across the
# working band (99.2% idle-with-display vs 98.7% computing on kernel
# 7.0.0-38, LP-0MW5K7WKZ005VJXA), so it was removed and the power
# discriminator became the whole signal:
#
#   - compute-level power (>= power_low_threshold) + all slots idle -> wedge
#   - power below the compute threshold                              -> no wedge
#   - missing power source                                           -> no verdict
#
# A missing power source never yields a wedge verdict (fail closed).
# ===================================================================

class TestGpuWedgeDiscriminate:
    def test_compute_power_with_idle_slots_is_wedge(self):
        # Genuine compute (~115 W) with nothing processing: GPU kernels
        # active, no work to show for them.
        assert _gpu_wedge_discriminate(110.0, True, 90.0) == "wedge"
        assert _gpu_wedge_discriminate(115.0, True, 90.0) == "wedge"

    def test_power_at_threshold_counts_as_compute(self):
        assert _gpu_wedge_discriminate(90.0, True, 90.0) == "wedge"

    def test_idle_power_with_idle_slots_is_not_wedge(self):
        # Idle-with-display band (~33-54 W, p50 45 W): the GPU is simply not
        # computing, so there is no wedge signature.
        assert _gpu_wedge_discriminate(54.0, True, 90.0) == "none"
        assert _gpu_wedge_discriminate(45.0, True, 90.0) == "none"

    def test_power_just_below_threshold_is_not_wedge(self):
        assert _gpu_wedge_discriminate(89.9, True, 90.0) == "none"

    def test_processing_slot_is_never_a_wedge(self):
        # A slot is processing tokens: normal compute, not a wedge.
        assert _gpu_wedge_discriminate(110.0, False, 90.0) == "none"

    def test_missing_power_source_fails_closed(self):
        # No power source: no verdict, never a wedge from missing data.
        assert _gpu_wedge_discriminate(None, True, 90.0) == "none"
        assert _gpu_wedge_discriminate(None, False, 90.0) == "none"


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
# _probe_gpu_wedge
# ===================================================================

class TestProbeGpuWedge:
    @pytest.mark.asyncio
    async def test_returns_power_and_slots_state(self, monkeypatch):
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

            async def aclose(self):
                return None

            async def get(self, url, timeout=None):
                assert "/slots" in url
                assert "?model=" in url
                return FakeResp()

        monkeypatch.setattr("proxy.backends.llama.httpx.AsyncClient", FakeClient)
        power, idle = await _probe_gpu_wedge(
            "127.0.0.1", 8080, model="Qwen3"
        )
        assert power == 54.0
        assert idle is True

    @pytest.mark.asyncio
    async def test_slots_failure_fails_open_with_idle_false(self, monkeypatch):
        monkeypatch.setattr(
            "proxy.backends.llama._read_gpu_power_draw", lambda: 54.0
        )

        class BoomClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def aclose(self):
                return None

            async def get(self, url, timeout=None):
                raise RuntimeError("connection refused")

        monkeypatch.setattr("proxy.backends.llama.httpx.AsyncClient", BoomClient)
        power, idle = await _probe_gpu_wedge(
            "127.0.0.1", 8080, model="Qwen3"
        )
        assert power == 54.0
        assert idle is False

    @pytest.mark.asyncio
    async def test_slot_query_failure_streak_is_logged(self, monkeypatch, caplog):
        """AC: consecutive slot-query failures are visible to the probe's logging.

        The probe fails open (a failed slot query never flips idle True), but
        the failures must not be silent: each one is logged with the running
        streak so a slot-query outage is observable.
        """
        monkeypatch.setattr(
            "proxy.backends.llama._read_gpu_power_draw", lambda: 54.0
        )

        class BoomClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def aclose(self):
                return None

            async def get(self, url, timeout=None):
                raise TimeoutError("slots query timed out")

        monkeypatch.setattr("proxy.backends.llama.httpx.AsyncClient", BoomClient)
        llama_backend._slot_query_failure_streak = 0
        with caplog.at_level(logging.WARNING, logger="llama-proxy.wedge"):
            await _probe_gpu_wedge("127.0.0.1", 8080, model="Qwen3")
            await _probe_gpu_wedge("127.0.0.1", 8080, model="Qwen3")
        streak_messages = [r.getMessage() for r in caplog.records]
        assert any("1 consecutive failure" in m for m in streak_messages)
        assert any("2 consecutive failure" in m for m in streak_messages)

    @pytest.mark.asyncio
    async def test_slot_query_failure_streak_resets_after_success(
        self, monkeypatch, caplog
    ):
        """A successful query breaks the streak: the next failure logs 1 again."""

        class OkClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def aclose(self):
                return None

            async def get(self, url, timeout=None):
                class FakeResp:
                    status_code = 200

                    def json(self):
                        return []

                return FakeResp()

        class BoomClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def aclose(self):
                return None

            async def get(self, url, timeout=None):
                raise TimeoutError("slots query timed out")

        monkeypatch.setattr(
            "proxy.backends.llama._read_gpu_power_draw", lambda: 54.0
        )
        llama_backend._slot_query_failure_streak = 0

        def make_client(factory):
            monkeypatch.setattr(
                "proxy.backends.llama.httpx.AsyncClient", factory
            )

        with caplog.at_level(logging.WARNING, logger="llama-proxy.wedge"):
            make_client(BoomClient)
            await _probe_gpu_wedge("127.0.0.1", 8080, model="Qwen3")
            make_client(OkClient)
            await _probe_gpu_wedge("127.0.0.1", 8080, model="Qwen3")
            make_client(BoomClient)
            await _probe_gpu_wedge("127.0.0.1", 8080, model="Qwen3")
        last_failure = [
            r.getMessage()
            for r in caplog.records
            if "consecutive failure" in r.getMessage()
        ][-1]
        assert "1 consecutive failure" in last_failure


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


def test_health_no_wedge_detection_disabled_field_after_pinned_removal():
    """The PINNED auto-disable state is gone (LP-0MW5K8ZW3005SEA8 AC2), so
    /health no longer carries a gpu_wedge_detection_disabled field."""
    srv = _make_server_mock(gpu_wedge_detected=False)
    result = _call_health_check(srv)
    assert "gpu_wedge_detection_disabled" not in result


# ===================================================================
# Source guard (LP-0MW5K8ZW3005SEA8 AC6)
# ===================================================================

def test_module_contains_no_gpu_busy_percent_reference():
    """No code path may read gpu_busy_percent: the counter is not referenced
    anywhere in the wedge module, comments included (the wedge signature is
    described by power and slot state only)."""
    source = (
        PROJECT_ROOT / "proxy" / "proxy" / "backends" / "llama.py"
    ).read_text()
    assert "gpu_busy_percent" not in source
