"""Tests for gpu_busy module — coverage-honest GPU busy-time measurement.

The module classifies every expected instant on a 15 s grid across the window
as ``computing`` / ``idle`` / ``unknown`` (LP-0MW5K8YE9001QO39). These tests
pin the coverage arithmetic against the authored gapped-window fixture and the
small synthetic cases, and guard against the renormalising rolling-average
query returning.
"""

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

#: The authored two-hour store bundle with its deliberate 10-minute hole.
#: ``monitoring/tests/fixtures/window_with_gap.json`` is shared with the
#: ``gpu_utilisation`` CLI tests, so both readers measure the same store.
WINDOW_WITH_GAP = (
    Path(__file__).resolve().parents[4]
    / "monitoring"
    / "tests"
    / "fixtures"
    / "window_with_gap.json"
)
#: The fixture's own documented arithmetic: 481 expected grid instants, 441
#: observed, 40 missing (8.316 % unknown), of which 68 computing and 373 idle.
EXPECTED_SAMPLES = 481
OBSERVED_SAMPLES = 441
COMPUTING_SAMPLES = 68
IDLE_SAMPLES = 373
UNKNOWN_SAMPLES = 40


def _matrix(points: dict[int, float]) -> dict:
    """A Prometheus ``query_range`` matrix payload for one series."""
    return {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {},
                    "values": [[ts, str(value)] for ts, value in sorted(points.items())],
                }
            ],
        },
    }


def _stub_series(
    *,
    computing_bool: dict[int, float],
    up_llama: dict[int, float] | None = None,
    up_node: dict[int, float] | None = None,
    heartbeat: dict[int, float] | None = None,
    power: dict[int, float] | None = None,
):
    """A ``query_prometheus`` stand-in answering from explicit series."""

    def _query(query: str, start: int, end: int, step: str = "60") -> dict:
        if query == gpu_busy.BOOL_QUERY:
            return _matrix(computing_bool)
        if query == gpu_busy.UP_LLAMA_PROXY_QUERY:
            return _matrix(up_llama or {})
        if query == gpu_busy.UP_NODE_EXPORTER_QUERY:
            return _matrix(up_node or {})
        if query == gpu_busy.HEARTBEAT_QUERY:
            return _matrix(heartbeat or {})
        if query == gpu_busy.POWER_QUERY:
            return _matrix(power or {})
        raise AssertionError(f"unexpected query: {query}")

    return _query


def _fixture_bundle() -> dict:
    return json.loads(WINDOW_WITH_GAP.read_text())


def _fixture_query(query: str, start: int, end: int, step: str = "60") -> dict | None:
    """Answer from the recorded bundle (the bundle predates the power alias)."""
    responses = _fixture_bundle()["responses"]
    if query == gpu_busy.POWER_QUERY:
        return responses["gpu:power_watts:amdgpu"]
    return responses.get(query)


class TestGappedWindowFixture:
    """The authored 2 h store bundle: three shares + the seconds identity."""

    START = _fixture_bundle()["capture"]["window"]["start_unix"]
    END = _fixture_bundle()["capture"]["window"]["end_unix"]

    def _result(self) -> gpu_busy.GpuBusyResult:
        with patch("gpu_busy.query_prometheus", side_effect=_fixture_query):
            result = gpu_busy.compute_gpu_busy(self.START, self.END)
        assert result is not None
        return result

    def test_reports_the_documented_three_shares(self):
        result = self._result()

        assert result.expected_samples == EXPECTED_SAMPLES
        assert result.observed_samples == OBSERVED_SAMPLES
        assert result.computing_samples == COMPUTING_SAMPLES
        assert result.idle_samples == IDLE_SAMPLES
        assert result.unknown_samples == UNKNOWN_SAMPLES

        assert result.shares_pct["computing"] == pytest.approx(
            100.0 * COMPUTING_SAMPLES / EXPECTED_SAMPLES
        )
        assert result.shares_pct["idle"] == pytest.approx(
            100.0 * IDLE_SAMPLES / EXPECTED_SAMPLES
        )
        assert result.shares_pct["unknown"] == pytest.approx(8.316, abs=1e-3)
        assert sum(result.shares_pct.values()) == pytest.approx(100.0)

    def test_seconds_cover_the_window_exactly(self):
        result = self._result()

        assert result.busy_seconds + result.idle_seconds + result.unknown_seconds == (
            pytest.approx(result.total_seconds)
        )
        assert result.unknown_seconds == pytest.approx(
            UNKNOWN_SAMPLES / EXPECTED_SAMPLES * result.total_seconds
        )

    def test_the_gap_is_never_counted_as_idle_or_in_the_numerator(self):
        result = self._result()

        # Absence is not idle: the two observed shares exclude the 40 missing
        # instants, and computing cannot include what nobody measured.
        assert result.idle_samples == IDLE_SAMPLES
        assert result.idle_samples != EXPECTED_SAMPLES - COMPUTING_SAMPLES
        assert result.computing_fraction == pytest.approx(
            COMPUTING_SAMPLES / EXPECTED_SAMPLES
        )

    def test_smoothed_power_trace_is_carried(self):
        result = self._result()
        assert len(result.smoothed_power_values) == OBSERVED_SAMPLES


class TestAbsenceIsUnknown:
    """Small synthetic grids pin the three-way classification."""

    def test_absent_samples_are_unknown_not_idle_or_numerator(self):
        # Four instants (0, 15, 30, 45): one idle, one computing, two absent.
        stub = _stub_series(
            computing_bool={0: 0.0, 15: 1.0},
            up_llama={0: 1.0, 15: 1.0},
        )
        with patch("gpu_busy.query_prometheus", side_effect=stub):
            result = gpu_busy.compute_gpu_busy(0, 45)

        assert result is not None
        assert (result.computing_samples, result.idle_samples, result.unknown_samples) == (
            1,
            1,
            2,
        )
        assert result.shares_pct == pytest.approx(
            {"computing": 25.0, "idle": 25.0, "unknown": 50.0}
        )

    def test_a_missing_sentinel_makes_only_that_instant_unknown(self):
        # The rule reports at all four instants, but the pipeline proves itself
        # alive at only three of them.
        stub = _stub_series(
            computing_bool={0: 1.0, 15: 0.0, 30: 0.0, 45: 0.0},
            up_llama={0: 1.0, 15: 1.0, 30: 1.0},
        )
        with patch("gpu_busy.query_prometheus", side_effect=stub):
            result = gpu_busy.compute_gpu_busy(0, 45)

        assert result is not None
        assert (result.computing_samples, result.idle_samples, result.unknown_samples) == (
            1,
            2,
            1,
        )

    def test_a_stale_heartbeat_does_not_prove_the_instant_was_observed(self):
        # Instant 0 has a heartbeat 100 s old (stale); instant 15 is fresh.
        stub = _stub_series(
            computing_bool={0: 1.0, 15: 0.0},
            heartbeat={0: -100.0, 15: 15.0},
        )
        with patch("gpu_busy.query_prometheus", side_effect=stub):
            result = gpu_busy.compute_gpu_busy(0, 15)

        assert result is not None
        assert (result.computing_samples, result.idle_samples, result.unknown_samples) == (
            0,
            1,
            1,
        )

    def test_a_fully_absent_window_keeps_the_no_row_behaviour(self):
        # Documented choice (module docstring): no observed instants -> None,
        # the report omits the row, and nothing is reported as a fabricated 0 %.
        stub = _stub_series(computing_bool={})
        with patch("gpu_busy.query_prometheus", side_effect=stub):
            assert gpu_busy.compute_gpu_busy(0, 45) is None

    def test_unreachable_prometheus_returns_none(self):
        with patch("gpu_busy.query_prometheus", return_value=None):
            assert gpu_busy.compute_gpu_busy(0, 45) is None

    def test_zero_or_negative_window_returns_none(self):
        with patch("gpu_busy.query_prometheus", return_value=_matrix({0: 1.0})):
            assert gpu_busy.compute_gpu_busy(45, 0) is None
            assert gpu_busy.compute_gpu_busy(0, 0) is None


class TestNoRenormalisingQuery:
    """The renormalising rolling average must not come back (AC3/AC6)."""

    def test_the_module_issues_a_plain_selector(self):
        seen: list[str] = []

        def _capturing(query, start, end, step="60"):
            seen.append(query)
            return _stub_series(
                computing_bool={0: 1.0}, up_llama={0: 1.0}
            )(query, start, end, step)

        with patch("gpu_busy.query_prometheus", side_effect=_capturing):
            gpu_busy.compute_gpu_busy(0, 15)

        assert gpu_busy.BOOL_QUERY == "gpu:computing:bool"
        assert seen, "the module must query Prometheus"
        assert all("avg_over_time" not in query for query in seen)

    def test_the_module_source_holds_no_renormalising_average(self):
        # Explicit regression pin required by the work item: the renormalised
        # form must fail loudly if it is reintroduced.
        source = Path(gpu_busy.__file__).read_text(encoding="utf-8")
        assert "avg_over_time" not in source


class TestGridHelpers:
    """Grid alignment: jitter is snapped, out-of-window samples are dropped."""

    def test_align_to_grid_snaps_jitter_and_drops_out_of_window_samples(self):
        aligned = gpu_busy._align_to_grid(
            {7: 1.0, 22: 2.0, 300: 9.0}, start=0, step=15, expected=2
        )
        assert aligned == {0: 1.0, 15: 2.0}

    def test_expected_sample_count_covers_both_window_ends(self):
        assert gpu_busy._expected_sample_count(0, 7200, 15) == 481
        assert gpu_busy._expected_sample_count(0, 0, 15) == 1


class TestGpuBusyResultShares:
    """GpuBusyResult convenience properties stay coverage-consistent."""

    def test_busy_and_idle_pct_sum_to_100_without_unknown(self):
        result = gpu_busy.GpuBusyResult(
            computing_fraction=0.35,
            total_seconds=3600,
            busy_seconds=1260,
            idle_seconds=2340,
            smoothed_power_values=[],
        )
        assert result.busy_pct == pytest.approx(35.0)
        assert result.idle_pct == pytest.approx(65.0)
        assert result.unknown_pct == pytest.approx(0.0)
        assert sum(result.shares_pct.values()) == pytest.approx(100.0)

    def test_unknown_share_is_carried_alongside_computing_and_idle(self):
        result = gpu_busy.GpuBusyResult(
            computing_fraction=0.5,
            total_seconds=100.0,
            busy_seconds=50.0,
            idle_seconds=40.0,
            unknown_seconds=10.0,
            smoothed_power_values=[],
        )
        assert result.busy_pct == pytest.approx(50.0)
        assert result.idle_pct == pytest.approx(40.0)
        assert result.unknown_pct == pytest.approx(10.0)
        assert sum(result.shares_pct.values()) == pytest.approx(100.0)


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
            result = gpu_busy.query_prometheus("test", 0, 3600)
        assert result == {"status": "success"}

    def test_returns_none_on_exception(self):
        with patch("gpu_busy.urllib.request.urlopen", side_effect=OSError("network error")):
            result = gpu_busy.query_prometheus("test", 0, 3600)
        assert result is None


class TestGpuBusyJson:
    """_gpu_busy_json rendering in reporting.py."""

    def test_gpu_busy_json_carries_coverage_and_the_three_shares(self):
        gpu = gpu_busy.GpuBusyResult(
            computing_fraction=0.5,
            total_seconds=21600,
            busy_seconds=10800.0,
            idle_seconds=7200.0,
            unknown_seconds=3600.0,
            smoothed_power_values=[(1698766200, 93.2), (1698766260, 10.8)],
            expected_samples=60,
            observed_samples=50,
            computing_samples=30,
            idle_samples=20,
            unknown_samples=10,
        )
        data = reporting._gpu_busy_json(gpu)
        assert data["computing_fraction"] == 0.5
        assert data["busy_pct"] == 50.0
        assert data["idle_pct"] == pytest.approx(33.3, abs=0.1)
        assert data["unknown_pct"] == pytest.approx(16.7, abs=0.1)
        assert data["unknown_seconds"] == 3600.0
        assert data["total_seconds"] == 21600.0
        assert data["coverage"] == {
            "expected_samples": 60,
            "observed_samples": 50,
            "computing_samples": 30,
            "idle_samples": 20,
            "unknown_samples": 10,
        }
        assert data["shares_pct"] == {
            "computing": 50.0,
            "idle": pytest.approx(33.3, abs=0.1),
            "unknown": pytest.approx(16.7, abs=0.1),
        }
        assert len(data["smoothed_power"]) == 2
        assert data["smoothed_power"][0] == {"timestamp": 1698766200, "power_watts": 93.2}

    def test_gpu_busy_json_returns_none_for_none(self):
        assert reporting._gpu_busy_json(None) is None


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
    def _stub_gpu_busy(unknown_seconds: float = 0.0) -> gpu_busy.GpuBusyResult:
        """A deterministic hardware figure that diverges from the log data."""
        return gpu_busy.GpuBusyResult(
            computing_fraction=0.50,
            total_seconds=3600.0,
            busy_seconds=1800.0,
            idle_seconds=1800.0 - unknown_seconds,
            unknown_seconds=unknown_seconds,
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
        assert "any GPU consumer" in section

    def test_report_row_names_the_unknown_share(self):
        """A partially observed window states its unobserved share in the row."""
        summary = self._summary()
        summary.gpu_busy = self._stub_gpu_busy(unknown_seconds=360.0)
        report = reporting.build_report(summary, None)
        section = report.split("## Local model utilization", 1)[1].split("## ", 1)[0]
        assert "unknown 10.0%" in section
