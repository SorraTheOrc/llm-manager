
# <!-- REFACTOR-LP-0MWIGY7V10086FB5
# smell: formatting
# severity: high
# description: Module level import not at top of file
# -->
"""Tests for the attributable GPU-activity share (:mod:`gpu_busy`).

LP-0MWG7KQVZ008DBZM: the report's GPU-usage figure is the attributable
``gpu:computing_workload:bool`` share (power ≥ 90 W **and** ≥ 1 busy local
slot) produced only by ``scripts/gpu_utilisation.py``. These tests pin:

- the attributable arithmetic (computing / idle / unknown over the expected
  grid) against the shared gapped-window fixture;
- the per-hour bucketing (same definition, absolute hours, ``unknown`` for an
  unobserved hour — never a fabricated ``0 %``);
- that the report consumes the producer rather than re-deriving the share and
  never re-introduces the retired power-only ``gpu:computing:bool`` figure.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
# The producer lives at the repository root; make it importable for the
# single-producer assertions.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import aggregation
import bucketing
import gpu_busy
import gpu_utilisation
import log_parser
import reporting

from tests import fixtures

#: The authored two-hour store bundle with its deliberate 10-minute hole.
#: ``monitoring/tests/fixtures/window_with_gap.json`` is shared with the
#: ``gpu_utilisation`` CLI tests, so both readers measure the same store.
WINDOW_WITH_GAP = (
    PROJECT_ROOT / "monitoring" / "tests" / "fixtures" / "window_with_gap.json"
)
#: The fixture's own documented arithmetic: 481 expected grid instants, 441
#: observed, 40 missing (8.316 % unknown), of which 68 attributable computing
#: and 373 idle.
EXPECTED_SAMPLES = 481
OBSERVED_SAMPLES = 441
COMPUTING_SAMPLES = 68
IDLE_SAMPLES = 373
UNKNOWN_SAMPLES = 40


def _bundle_series(bundle: dict) -> dict[str, dict[int, float]]:
    """Build the producer's ``{query_key: {unix_ts: value}}`` series mapping."""
    window = bundle["capture"]["window"]
    start, end = window["start_unix"], window["end_unix"]
    expected = gpu_utilisation.expected_sample_count(start, end, 15)
    series: dict[str, dict[int, float]] = {}
    for key, query in gpu_utilisation.QUERIES.items():
        series[key] = gpu_utilisation.align_to_grid(
            gpu_utilisation.parse_matrix(bundle["responses"][query]),
            start=start,
            step=15,
            expected=expected,
        )
    return series


class TestGappedWindowFixture:
    """The authored 2 h store bundle: attributable shares + the seconds identity."""

    BUNDLE = json.loads(WINDOW_WITH_GAP.read_text())
    START = BUNDLE["capture"]["window"]["start_unix"]
    END = BUNDLE["capture"]["window"]["end_unix"]

    def _result(self) -> gpu_busy.GpuBusyResult:
        result = gpu_busy.compute_gpu_busy(
            self.START, self.END, series=_bundle_series(self.BUNDLE)
        )
        assert result is not None
        return result

    def test_reports_the_attributable_three_shares(self):
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

    def test_the_attributable_share_matches_the_producer(self):
        """The report's figure is the producer's figure, not a re-derivation."""
        result = self._result()
        series = _bundle_series(self.BUNDLE)
        window = gpu_utilisation.Window(start=self.START, end=self.END)
        produced = gpu_utilisation.compute_utilisation(
            series, window=window, step_seconds=15, scrape_interval_seconds=15
        )
        assert result.busy_pct == pytest.approx(produced.computing_pct)
        assert result.unknown_pct == pytest.approx(produced.unknown_pct)

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

    def test_smoothed_power_trace_is_a_one_minute_mean(self):
        result = self._result()
        # 2 h of 15 s samples collapse to 111 one-minute means for this bundle
        # (the hole trims a few minutes); never the raw 441 samples.
        assert 100 <= len(result.smoothed_power_values) <= 120
        assert len(result.smoothed_power_values) < OBSERVED_SAMPLES

    def test_hourly_shares_reconcile_with_the_window(self):
        result = self._result()
        assert len(result.hourly_shares) == 3  # partial 02h, full 03h, partial 04h
        assert sum(s.expected_samples for s in result.hourly_shares) == EXPECTED_SAMPLES
        assert sum(s.computing_samples for s in result.hourly_shares) == COMPUTING_SAMPLES
        # The per-hour bucket (03:00) carries the whole deliberate hole.
        assert [s.unknown_samples for s in result.hourly_shares] == [0, 40, 0]


class TestAttributableClassification:
    """Small synthetic grids pin the three-way classification."""

    @staticmethod
    def _series(
        *,
        workload: dict[int, float],
        up_llama: dict[int, float] | None = None,
        up_node: dict[int, float] | None = None,
        heartbeat: dict[int, float] | None = None,
    ) -> dict[str, dict[int, float]]:
        return {
            "computing_workload_bool": workload,
            "up_llama_proxy": up_llama or {},
            "up_node_exporter": up_node or {},
            "slot_heartbeat": heartbeat or {},
            "power_watts": {},
        }

    def test_absent_samples_are_unknown_not_idle_or_numerator(self):
        # Four instants (0, 15, 30, 45): one idle, one computing, two absent.
        series = self._series(
            workload={0: 0.0, 15: 1.0},
            up_llama={0: 1.0, 15: 1.0},
        )
        result = gpu_busy.compute_gpu_busy(0, 45, series=series)

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
        series = self._series(
            workload={0: 1.0, 15: 0.0, 30: 0.0, 45: 0.0},
            up_llama={0: 1.0, 15: 1.0, 30: 1.0},
        )
        result = gpu_busy.compute_gpu_busy(0, 45, series=series)

        assert result is not None
        assert (result.computing_samples, result.idle_samples, result.unknown_samples) == (
            1,
            2,
            1,
        )

    def test_a_stale_heartbeat_does_not_prove_the_instant_was_observed(self):
        series = self._series(
            workload={0: 1.0, 15: 0.0},
            heartbeat={0: -100.0, 15: 15.0},
        )
        result = gpu_busy.compute_gpu_busy(0, 15, series=series)

        assert result is not None
        assert (result.computing_samples, result.idle_samples, result.unknown_samples) == (
            0,
            1,
            1,
        )

    def test_a_fully_absent_window_keeps_the_no_row_behaviour(self):
        result = gpu_busy.compute_gpu_busy(
            0, 45, series=self._series(workload={})
        )
        assert result is None

    def test_unreachable_prometheus_returns_none(self):
        producer = gpu_busy._producer()

        def _boom(url):  # pragma: no cover - exercised by the call below
            raise producer.PrometheusError("down")

        assert gpu_busy.compute_gpu_busy(0, 45, getter=_boom) is None

    def test_zero_or_negative_window_returns_none(self):
        series = self._series(workload={0: 1.0}, up_llama={0: 1.0})
        assert gpu_busy.compute_gpu_busy(45, 0, series=series) is None
        assert gpu_busy.compute_gpu_busy(0, 0, series=series) is None


class TestSingleProducerConsumption:
    """The report consumes ``scripts/gpu_utilisation.py``; it never re-derives."""

    def test_the_module_imports_the_named_producer(self):
        producer = gpu_busy._producer()
        assert producer.__file__ == str(
            PROJECT_ROOT / "scripts" / "gpu_utilisation.py"
        )
        # Same implementation: the report's arithmetic *is* the producer's
        # (loaded once by path into its own module namespace, so identity is
        # proven on the compiled code, not on the object).
        for name in ("compute_utilisation", "hourly_attributable_shares"):
            assert (
                getattr(producer, name).__code__.co_code
                == getattr(gpu_utilisation, name).__code__.co_code
            ), name
            assert (
                getattr(producer, name).__code__.co_filename
                == getattr(gpu_utilisation, name).__code__.co_filename
            ), name

    def test_the_module_holds_no_promql_of_its_own(self):
        source = Path(gpu_busy.__file__).read_text(encoding="utf-8")
        # No query constants and no PromQL aggregation: any querying here would
        # be a second producer surface the guard forbids.
        assert "BOOL_QUERY" not in source
        assert "avg_over_time" not in source
        assert "sum_over_time" not in source
        assert "query_range" not in source

    def test_no_power_only_figure_is_produced(self):
        """The retired ``gpu:computing:bool`` share cannot return as a figure.

        The module holds no power-only query constant and exposes no
        ``power_only_computing`` share; the only figures it produces are the
        attributable ones.
        """
        source = Path(gpu_busy.__file__).read_text(encoding="utf-8")
        assert "BOOL_QUERY" not in source
        assert "power_only_computing" not in source
        result = gpu_busy.compute_gpu_busy(
            0,
            45,
            series={
                "computing_workload_bool": {0: 0.0, 15: 1.0},
                "up_llama_proxy": {0: 1.0, 15: 1.0},
                "power_watts": {},
            },
        )
        block = reporting._gpu_busy_json(result)
        assert "power_only_computing" not in block
        assert "power_only_computing" not in block["shares_pct"]


class TestHourlyAttributableShares:
    """Per-hour bucketing: same measure, ``unknown`` for unobserved hours."""

    @staticmethod
    def _hourly(window_start: int, window_end: int, series) -> list:
        return gpu_utilisation.hourly_attributable_shares(
            series,
            window=gpu_utilisation.Window(start=window_start, end=window_end),
            step_seconds=15,
            scrape_interval_seconds=15,
        )

    def test_an_unobserved_hour_reads_none_not_zero(self):
        # Hour 0 is densely observed (every 15 s instant computing); hour 1's
        # four instants have no samples, so they are unknown — never a
        # fabricated 0 %.
        start, end = 0, 3600 + 45
        hour0_instants = list(range(0, 3600, 15))  # 0 .. 3585
        series = {
            "computing_workload_bool": {ts: 1.0 for ts in hour0_instants},
            "up_llama_proxy": {ts: 1.0 for ts in hour0_instants},
            "power_watts": {},
        }
        shares = {
            share.hour_start: share for share in self._hourly(start, end, series)
        }
        assert shares[0].attributable_pct == pytest.approx(100.0)
        assert shares[0].observed_samples == len(hour0_instants)
        assert shares[3600].observed_samples == 0
        assert shares[3600].attributable_pct is None, "never a fabricated 0 %"
        assert shares[3600].unknown_pct == pytest.approx(100.0)

    def test_partial_edge_hours_are_bounded_by_the_window(self):
        shares = self._hourly(
            1793412780,  # 02:13 offset, mirrors the shared fixture
            1793419980,
            {
                "computing_workload_bool": {},
                "up_llama_proxy": {},
                "power_watts": {},
            },
        )
        assert [s.duration_seconds for s in shares] == [2820, 3600, 780]
        assert sum(s.duration_seconds for s in shares) == 7200

    def test_shares_never_claim_more_than_the_hour(self):
        bundle = json.loads(WINDOW_WITH_GAP.read_text())
        series = _bundle_series(bundle)
        window = bundle["capture"]["window"]
        shares = self._hourly(window["start_unix"], window["end_unix"], series)
        for share in shares:
            assert 0.0 <= share.unknown_pct <= 100.0
            assert share.attributable_pct is not None
            assert 0.0 <= share.attributable_pct <= 100.0


class TestGpuBusyJson:
    """_gpu_busy_json rendering in reporting.py."""

    @staticmethod
    def _result(**kwargs) -> gpu_busy.GpuBusyResult:
        base = dict(
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
        base.update(kwargs)
        return gpu_busy.GpuBusyResult(**base)

    def test_carries_coverage_the_three_shares_and_hourly(self):
        producer = gpu_busy._producer()
        hourly = [
            producer.HourlyAttributable(
                hour_start=1698766200,
                duration_seconds=3600,
                expected_samples=240,
                observed_samples=240,
                computing_samples=60,
                idle_samples=180,
                unknown_samples=0,
            ),
            producer.HourlyAttributable(
                hour_start=1698769800,
                duration_seconds=3600,
                expected_samples=240,
                observed_samples=0,
                computing_samples=0,
                idle_samples=0,
                unknown_samples=240,
            ),
        ]
        data = reporting._gpu_busy_json(self._result(hourly_shares=hourly))

        assert data["computing_fraction"] == 0.5
        assert data["busy_pct"] == 50.0
        assert data["idle_pct"] == pytest.approx(33.3, abs=0.1)
        assert data["unknown_pct"] == pytest.approx(16.7, abs=0.1)
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
        assert "power_only_computing" not in data["shares_pct"]
        # Hourly twin of the report column: observed hour has a share, the
        # unobserved hour is explicitly null.
        assert data["hourly"][0]["attributable_pct"] == pytest.approx(25.0)
        assert data["hourly"][1]["attributable_pct"] is None
        assert data["smoothed_power"][0] == {
            "timestamp": 1698766200,
            "power_watts": 93.2,
        }

    def test_returns_none_for_none(self):
        assert reporting._gpu_busy_json(None) is None


class TestReportRendering:
    """The rendered report reports the attributable measure on both surfaces."""

    WINDOW_START = datetime(2026, 8, 2, 14, 0, 0)
    WINDOW_END = datetime(2026, 8, 2, 16, 0, 0)

    @classmethod
    def _summary(cls) -> aggregation.AnalysisResult:
        schedule = bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])
        events = [
            e
            for e in (log_parser.parse_log_line(ln) for ln in fixtures.E2E_LINES)
            if e is not None
        ]
        return aggregation.aggregate(
            events, cls.WINDOW_START, cls.WINDOW_END, schedule
        )

    @staticmethod
    def _gpu(unknown_seconds: float = 0.0) -> gpu_busy.GpuBusyResult:
        return gpu_busy.GpuBusyResult(
            computing_fraction=0.50,
            total_seconds=7200.0,
            busy_seconds=3600.0,
            idle_seconds=3600.0 - unknown_seconds,
            unknown_seconds=unknown_seconds,
            smoothed_power_values=[],
        )

    def test_local_utilisation_row_is_attributable_and_scoped(self):
        summary = self._summary()
        summary.gpu_busy = self._gpu()
        report = reporting.build_report(summary, None)
        section = report.split("## Local model utilization", 1)[1].split("## ", 1)[0]
        assert reporting.GPU_ATTRIBUTABLE_ROW_LABEL in section
        assert "gpu:computing_workload:bool" in section
        assert "scripts/gpu_utilisation.py" in section
        assert "not any GPU consumer" in section
        # The retired power-only figure is gone from the rendered report.
        assert "GPU busy — power-based" not in report
        assert "gpu:computing:bool" not in section

    def test_row_names_the_unknown_share(self):
        summary = self._summary()
        summary.gpu_busy = self._gpu(unknown_seconds=720.0)
        report = reporting.build_report(summary, None)
        section = report.split("## Local model utilization", 1)[1].split("## ", 1)[0]
        assert "unknown 10.0%" in section

    def test_hourly_table_reports_the_attributable_share_and_unknown_hours(self):
        producer = gpu_busy._producer()
        summary = self._summary()
        start = int(self.WINDOW_START.timestamp())
        summary.gpu_busy = self._gpu()
        summary.gpu_busy.hourly_shares = [
            producer.HourlyAttributable(
                hour_start=start - start % 3600,
                duration_seconds=7200,
                expected_samples=480,
                observed_samples=480,
                computing_samples=120,
                idle_samples=360,
                unknown_samples=0,
            )
        ]
        report = reporting.build_report(summary, None)
        section = report.split("## Summary by hour", 1)[1].split("## ", 1)[0]
        assert "| Hour | State | GPU computing (attributable) |" in section
        assert "25.0%" in section
        # The old log-derived Busy/% columns are gone.
        assert "| Hour | State | Busy | % |" not in report

    def test_hourly_table_reads_unknown_without_gpu_telemetry(self):
        summary = self._summary()
        summary.gpu_busy = None
        report = reporting.build_report(summary, None)
        section = report.split("## Summary by hour", 1)[1].split("## ", 1)[0]
        assert "| Hour | State | GPU computing (attributable) |" in section
        # No GPU telemetry: every hour's attributable cell is ``unknown``, never
        # a fabricated 0 %.
        assert section.count("| unknown |") >= 2  # each hour row + the Totals row

    def test_local_busy_stays_a_separate_scoped_figure(self):
        summary = self._summary()
        summary.gpu_busy = self._gpu()
        data = reporting.summary_to_json(summary, None)
        assert data["local_busy"]["definition"] == "dispatch_to_finish"
        assert data["gpu_busy"] is not None
        # The two remain different questions and different keys.
        assert data["local_busy"]["busy_seconds"] != data["gpu_busy"]["busy_seconds"]
