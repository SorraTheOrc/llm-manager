"""Tests for gpu_busy module — Prometheus-based GPU busy-time measurement."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aggregation
import bucketing
import gpu_busy
import log_parser
import reporting

from tests import fixtures

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


class TestLocalBusyIsDecisionFigure:
    """AC3/AC4 (LP-0MVUIM7A50053EFN): the decision-facing busy figure is the
    log-derived ``local_busy`` (``dispatch → finish``); the power-based
    ``gpu_busy`` stays a separate, clearly-labelled hardware key. Fixtures
    come from real log lines (``tests/fixtures.py``); the Prometheus source
    is stubbed — never live.
    """

    WINDOW_START = datetime(2026, 8, 2, 14, 0, 0)
    WINDOW_END = datetime(2026, 8, 2, 15, 0, 0)

    @staticmethod
    def _summary() -> aggregation.AnalysisResult:
        """Summary aggregated from the real-log ``fixtures.E2E_LINES``."""
        schedule = bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])
        events = [
            e
            for e in (
                log_parser.parse_log_line(ln) for ln in fixtures.E2E_LINES
            )
            if e is not None
        ]
        return aggregation.aggregate(
            events, TestLocalBusyIsDecisionFigure.WINDOW_START,
            TestLocalBusyIsDecisionFigure.WINDOW_END, schedule,
        )

    @staticmethod
    def _stub_gpu_busy() -> gpu_busy.GpuBusyResult:
        """A deterministic hardware figure that diverges from the log data."""
        return gpu_busy.GpuBusyResult(
            computing_fraction=0.50,
            total_seconds=3600.0,
            busy_seconds=1800.0,
            idle_seconds=1800.0,
            smoothed_power_values=[],
        )

    def test_json_decision_figure_is_log_derived(self):
        """``summary_to_json``'s decision-facing busy is the log-derived
        ``local_busy`` carrying the ``dispatch_to_finish`` definition marker;
        ``gpu_busy`` is a separate key and ``None`` without Prometheus."""
        data = reporting.summary_to_json(self._summary(), None)
        local_busy = data["local_busy"]
        assert local_busy is not None
        assert local_busy["definition"] == "dispatch_to_finish"
        # The dispatch → finish union measured from the real-log fixtures
        # (14:01:00 → 14:01:05 local stream plus the sibling pair).
        assert local_busy["busy_seconds"] == pytest.approx(7.0)
        assert local_busy["busy_pct"] == pytest.approx(0.2, abs=0.05)
        assert data["gpu_busy"] is None

    def test_gpu_busy_stays_separate_and_never_presented_as_local_utilisation(self):
        """With a stubbed GPU result the two JSON keys stay separate and the
        hardware figure never replaces the log-derived decision figure."""
        summary = self._summary()
        summary.gpu_busy = self._stub_gpu_busy()
        data = reporting.summary_to_json(summary, None)
        local_busy = data["local_busy"]
        gpu = data["gpu_busy"]
        assert local_busy["definition"] == "dispatch_to_finish"
        assert gpu is not None
        assert gpu["busy_pct"] == 50.0
        assert gpu["busy_seconds"] == 1800.0
        # The log-derived decision figure is untouched by the hardware block
        # and the hardware and log figures diverge by design; a consumer
        # reading either as the local utilisation number would be misled.
        assert local_busy["busy_seconds"] == pytest.approx(7.0)
        assert local_busy["busy_pct"] != gpu["busy_pct"]

    def test_report_note_states_not_local_model_utilisation(self):
        """AC4: the report note carries the explicit \"not local model
        utilisation\" statement naming ``local_busy`` as the decision figure,
        and the hardware row label stays ``GPU busy — power-based
        (hardware)``."""
        summary = self._summary()
        summary.gpu_busy = self._stub_gpu_busy()
        report = reporting.build_report(summary, None)
        section = report.split("## Local model utilization", 1)[1].split("## ", 1)[0]
        assert "GPU busy — power-based (hardware)" in section
        assert "not local model utilisation" in section
        assert "local_busy" in section
