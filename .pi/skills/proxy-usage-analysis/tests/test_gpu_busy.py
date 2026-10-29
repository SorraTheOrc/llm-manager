"""Tests for gpu_busy module — Prometheus-based GPU busy-time measurement."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import gpu_busy

# A mock Prometheus response for avg_over_time(gpu:computing:bool[1h]).
_MOCK_BOOL_RESPONSE = {
    "status": "success",
    "data": {
        "result": [
            {
                "metric": {"__name__": "gpu:computing:bool"},
                "value": [1698766800, "0.50"],  # ts, value
            }
        ]
    },
}

# A mock Prometheus response for gpu:power_watts:avg5m range query.
_MOCK_POWER_RESPONSE = {
    "status": "success",
    "data": {
        "result": [
            {
                "metric": {
                    "__name__": "gpu:power_watts:avg5m",
                    "chip_name": "amdgpu",
                },
                "values": [
                    [1698766200, "93.20"],
                    [1698766260, "10.80"],
                    [1698766320, "95.50"],
                ],
            }
        ]
    },
}


class TestComputeGpuBusy:
    """compute_gpu_busy with mocked Prometheus responses."""

    def test_returns_gpubusyresult_on_success(self):
        ws = datetime(2024, 10, 31, 14, 0, 0).timestamp()
        we = datetime(2024, 10, 31, 15, 0, 0).timestamp()  # 1 h window
        with patch("gpu_busy.urllib.request.urlopen") as mock_urlopen:
            mock_resp = type("MockResp", (), {
                "read": lambda: json.dumps(_MOCK_BOOL_RESPONSE).encode(),
                "__enter__": lambda self: self,
                "__exit__": lambda self, *a: None,
            })()
            mock_urlopen.return_value = mock_resp
            with patch("gpu_busy.query_prometheus") as mock_q:
                # First call is the bool query (step=None, end-only)
                # Second call is the power trace
                mock_q.side_effect = [_MOCK_BOOL_RESPONSE, _MOCK_POWER_RESPONSE]
                result = gpu_busy.compute_gpu_busy(ws, we)
        assert result is not None
        assert result.computing_fraction == 0.5
        assert result.busy_seconds == pytest.approx(1800.0)
        assert result.idle_seconds == pytest.approx(1800.0)
        assert result.busy_pct == pytest.approx(50.0)
        assert len(result.smoothed_power_values) == 3

    def test_returns_none_on_prometheus_failure(self):
        ws = datetime(2024, 10, 31, 14, 0, 0).timestamp()
        we = datetime(2024, 10, 31, 15, 0, 0).timestamp()
        with patch("gpu_busy.query_prometheus", return_value=None):
            result = gpu_busy.compute_gpu_busy(ws, we)
        assert result is None

    def test_returns_none_on_empty_result(self):
        ws = datetime(2024, 10, 31, 14, 0, 0).timestamp()
        we = datetime(2024, 10, 31, 15, 0, 0).timestamp()
        with patch("gpu_busy.query_prometheus", return_value={"data": {"result": []}}):
            result = gpu_busy.compute_gpu_busy(ws, we)
        assert result is None

    def test_returns_none_on_invalid_value(self):
        ws = datetime(2024, 10, 31, 14, 0, 0).timestamp()
        we = datetime(2024, 10, 31, 15, 0, 0).timestamp()
        bad_response = {
            "status": "success",
            "data": {"result": [{"value": [0, "not-a-number"]}]},
        }
        with patch("gpu_busy.query_prometheus", return_value=bad_response):
            result = gpu_busy.compute_gpu_busy(ws, we)
        assert result is None

    def test_returns_none_on_zero_or_negative_window(self):
        ws = datetime(2024, 10, 31, 15, 0, 0).timestamp()
        we = datetime(2024, 10, 31, 14, 0, 0).timestamp()  # negative
        with patch("gpu_busy.query_prometheus", return_value=_MOCK_BOOL_RESPONSE):
            result = gpu_busy.compute_gpu_busy(ws, we)
        assert result is None
        result2 = gpu_busy.compute_gpu_busy(ws, ws)  # zero
        assert result2 is None


class TestGpuBusyResult:
    """GpuBusyResult dataclass convenience properties."""

    def test_busy_and_idle_pct_sum_to_100(self):
        result = gpu_busy.GpuBusyResult(
            computing_fraction=0.35,
            total_seconds=3600,
            busy_seconds=1260,
            idle_seconds=2340,
            smoothed_power_values=[],
        )
        assert result.busy_pct == pytest.approx(35.0)
        assert result.idle_pct == pytest.approx(65.0)

    def test_busy_seconds_equals_fraction_times_total(self):
        result = gpu_busy.GpuBusyResult(
            computing_fraction=0.75,
            total_seconds=7200,
            busy_seconds=5400.0,
            idle_seconds=1800.0,
            smoothed_power_values=[],
        )
        assert result.busy_seconds == pytest.approx(5400.0)
        assert result.idle_seconds == pytest.approx(1800.0)


class TestQueryPrometheus:
    """query_prometheus — network boundary, tested via mock."""

    def test_returns_dict_on_success(self):
        with patch("gpu_busy.urllib.request.urlopen") as mock_urlopen:
            mock_resp = type("MockResp", (), {
                "read": lambda self: json.dumps({"status": "success"}).encode(),
                "__enter__": lambda self: self,
                "__exit__": lambda self, *a: None,
            })()
            mock_urlopen.return_value = mock_resp
            result = gpu_busy.query_prometheus("avg_over_time(test[1h])", 0, 3600)
        assert result == {"status": "success"}

    def test_returns_none_on_exception(self):
        with patch("gpu_busy.urllib.request.urlopen", side_effect=OSError("network error")):
            result = gpu_busy.query_prometheus("test", 0, 3600)
        assert result is None


class TestGpuBusyJson:
    """_gpu_busy_json rendering in reporting.py."""

    def test_gpu_busy_json_from_result(self):
        import reporting

        gpu = gpu_busy.GpuBusyResult(
            computing_fraction=0.492,
            total_seconds=21600,
            busy_seconds=10627.2,
            idle_seconds=10972.8,
            smoothed_power_values=[(1698766200, 93.2), (1698766260, 10.8)],
        )
        data = reporting._gpu_busy_json(gpu)
        assert data["computing_fraction"] == 0.492
        assert data["busy_pct"] == 49.2
        assert data["idle_pct"] == 50.8
        assert data["total_seconds"] == 21600
        assert data["busy_seconds"] == 10627.2
        assert len(data["smoothed_power"]) == 2
        assert data["smoothed_power"][0] == {"timestamp": 1698766200, "power_watts": 93.2}

    def test_gpu_busy_json_returns_none_for_none(self):
        import reporting

        assert reporting._gpu_busy_json(None) is None


class TestComputeGpuBusyMatrixFormat:
    """Regression: the bool fraction may arrive in matrix (values[]) form.

    Guards against the bug where the range endpoint returned ``values`` but
    the code read ``value`` — silently yielding ``None``.
    """

    def test_reads_matrix_values_format(self):
        matrix_response = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"chip_name": "amdgpu"},
                        "values": [
                            [1698766200, "0.40"],
                            [1698766800, "0.60"],
                        ],
                    }
                ],
            },
        }
        ws = datetime(2024, 10, 31, 14, 0, 0).timestamp()
        we = datetime(2024, 10, 31, 15, 0, 0).timestamp()
        with patch("gpu_busy.query_prometheus") as mock_q:
            mock_q.side_effect = [matrix_response, _MOCK_POWER_RESPONSE]
            result = gpu_busy.compute_gpu_busy(ws, we)
        assert result is not None
        # Uses the LAST value of the matrix series.
        assert result.computing_fraction == 0.60

    def test_busy_plus_idle_equals_window(self):
        ws = datetime(2024, 10, 31, 14, 0, 0).timestamp()
        we = datetime(2024, 10, 31, 15, 0, 0).timestamp()
        with patch("gpu_busy.query_prometheus") as mock_q:
            mock_q.side_effect = [_MOCK_BOOL_RESPONSE, _MOCK_POWER_RESPONSE]
            result = gpu_busy.compute_gpu_busy(ws, we)
        assert result.busy_seconds + result.idle_seconds == pytest.approx(result.total_seconds)
        assert result.busy_pct + result.idle_pct == pytest.approx(100.0)

    def test_rejects_out_of_range_fraction(self):
        bad = {
            "status": "success",
            "data": {"result": [{"value": [0, "1.5"]}]},
        }
        ws = datetime(2024, 10, 31, 14, 0, 0).timestamp()
        we = datetime(2024, 10, 31, 15, 0, 0).timestamp()
        with patch("gpu_busy.query_prometheus", return_value=bad):
            result = gpu_busy.compute_gpu_busy(ws, we)
        assert result is None
