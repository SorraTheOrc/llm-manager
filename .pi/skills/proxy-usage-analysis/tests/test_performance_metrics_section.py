"""Behaviour of the rendered scorecard section (LP-0MW1BUA2M0005QRV).

The integration contract in ``test_performance_metrics_report.py`` pins the
section's position, columns and row set. This module pins what that contract
does not: that a run records the window's measurement through the engine
(and only when asked), that the rendered baseline/target reflect the history
the run produced, that the committed defaults file is never a write target,
that a metric with no measurement is never recorded as ``0``, and that a
cross-era target is flagged with the fingerprints it mixes.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SKILL_ROOT = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import aggregation  # noqa: E402
import bucketing  # noqa: E402
import log_parser  # noqa: E402
import performance_metrics as pm  # noqa: E402
import reporting  # noqa: E402

from tests import fixtures  # noqa: E402

DEFAULTS_PATH = SKILL_ROOT / "metrics" / "performance_metrics.yaml"
SUMMARY_HEADING = "## Performance metrics summary"
STATE_FILENAME = pm.STATE_FILENAME

WINDOW_START = datetime(2026, 8, 2, 14, 0, 0)
WINDOW_END = datetime(2026, 8, 2, 15, 0, 0)
WINDOW_ID = "2026-08-02"

#: The slot geometry the analysed proxy config was running: two slots of
#: 131072 tokens each out of a 262144-token total.
CONFIG = {"session_slot_pool_size": 2, "local_model_ctx_size": 262144}
FINGERPRINT = "slots=2,ctx=131072"

#: The E2E fixture window's measurable metrics. ``decode_speed``,
#: ``prompt_eval_speed``, ``time_to_first_token``, ``gpu_busy``,
#: ``prompt_cache_hit_ratio`` and ``prefill_eta_accuracy`` have no sample in it
#: (no speed stats, no GPU telemetry, no matched ETA pair) and must stay
#: unrecorded. ``context_pressure`` is measurable here only because the slot
#: geometry is known from the analysed config: no config means no per-slot
#: limit, and therefore no measured share.
MEASURABLE = (
    "fallback_rate",
    "stream_error_rate",
    "contention_denials_per_hour",
    "context_pressure",
)
UNMEASURABLE = (
    "gpu_busy",
    "decode_speed",
    "prompt_eval_speed",
    "time_to_first_token",
    "prompt_cache_hit_ratio",
    "prefill_eta_accuracy",
)


def _summary() -> aggregation.AnalysisResult:
    events = [
        event
        for event in (log_parser.parse_log_line(line) for line in fixtures.E2E_LINES)
        if event is not None
    ]
    schedule = bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])
    return aggregation.aggregate(events, WINDOW_START, WINDOW_END, schedule)


def _report(
    *,
    metrics_state: Path | None = None,
    metrics_config: Path | None = None,
    update: bool = False,
) -> str:
    return reporting.build_report(
        _summary(),
        CONFIG,
        metrics_state=metrics_state,
        metrics_config=metrics_config,
        update_metrics=update,
    )


def _measurements(state: dict, metric_id: str) -> list[dict]:
    return (state.get("metrics") or {}).get(metric_id, {}).get("measurements", [])


def _cells(report: str, label: str) -> list[str]:
    """The summary table row for *label*, as its six cells."""
    section = report.split(SUMMARY_HEADING, 1)[1].split("\n## ", 1)[0]
    rows = [
        line.strip().strip("|").split("|")
        for line in section.splitlines()
        if line.startswith("|")
    ]
    for row in rows[2:]:
        if row[0].strip() == label:
            return [cell.strip() for cell in row]
    raise AssertionError(f"no summary row for {label!r}")


#: Cell indices once the Tolerance column sits between Target and Actual.
BASELINE, TARGET, TOLERANCE, ACTUAL, RESULT = 1, 2, 3, 4, 5


def _seed(state_path: Path, metric_id: str, entries: list[tuple[str, float, str]]) -> None:
    """Write a live state file holding *entries* as (window, value, fingerprint)."""
    state = pm.new_state()
    for window, value, fingerprint in entries:
        pm.record_measurement(
            state,
            metric_id,
            window=window,
            value=value,
            date=window,
            fingerprint=fingerprint,
        )
    pm.save_state(state, state_path)


# ---------------------------------------------------------------------------
# Recording the window's measurement
# ---------------------------------------------------------------------------


class TestRecording:
    def test_updates_record_every_measurable_metric_and_skip_the_unmeasured(
        self, tmp_path
    ):
        state_path = tmp_path / STATE_FILENAME

        _report(metrics_state=state_path, update=True)

        assert state_path.exists()
        state = pm.load_state(state_path)
        for metric_id in MEASURABLE:
            measurements = _measurements(state, metric_id)
            assert len(measurements) == 1, f"{metric_id} not recorded"
            measurement = measurements[0]
            assert measurement["window"] == WINDOW_ID
            assert measurement["date"] == WINDOW_ID
            assert measurement["fingerprint"] == FINGERPRINT
        # A metric with no sample is absent, never recorded as 0.
        for metric_id in UNMEASURABLE:
            assert _measurements(state, metric_id) == [], metric_id

    def test_a_measured_zero_is_recorded_as_a_real_measurement(self, tmp_path):
        # The E2E window has streams but no stream errors: a measured 0 %, not
        # a missing figure, so it must be recorded rather than dropped.
        state_path = tmp_path / STATE_FILENAME

        _report(metrics_state=state_path, update=True)

        recorded = _measurements(pm.load_state(state_path), "stream_error_rate")
        assert [measurement["value"] for measurement in recorded] == [0.0]

    def test_rerunning_the_same_window_replaces_its_measurement(self, tmp_path):
        state_path = tmp_path / STATE_FILENAME

        _report(metrics_state=state_path, update=True)
        first = pm.load_state(state_path)
        _report(metrics_state=state_path, update=True)
        second = pm.load_state(state_path)

        for metric_id in MEASURABLE:
            assert len(_measurements(second, metric_id)) == 1, metric_id
            assert _measurements(second, metric_id)[0]["window"] == WINDOW_ID
        assert {
            metric_id: len(_measurements(second, metric_id)) for metric_id in MEASURABLE
        } == {
            metric_id: len(_measurements(first, metric_id)) for metric_id in MEASURABLE
        }

    def test_history_is_extended_without_losing_prior_windows(self, tmp_path):
        state_path = tmp_path / STATE_FILENAME
        _seed(
            state_path,
            "fallback_rate",
            [("2026-07-01", 20.0, FINGERPRINT), ("2026-07-15", 22.0, FINGERPRINT)],
        )

        _report(metrics_state=state_path, update=True)

        windows = [
            measurement["window"]
            for measurement in _measurements(pm.load_state(state_path), "fallback_rate")
        ]
        assert windows == ["2026-07-01", "2026-07-15", WINDOW_ID]

    def test_no_update_leaves_the_live_state_file_untouched(self, tmp_path):
        state_path = tmp_path / STATE_FILENAME
        _seed(state_path, "fallback_rate", [("2026-07-01", 20.0, FINGERPRINT)])
        before = state_path.read_bytes()

        _report(metrics_state=state_path, update=False)

        assert state_path.read_bytes() == before

    def test_without_a_state_path_nothing_is_written(self, tmp_path, monkeypatch):
        written: list[Path] = []
        real_save = pm.save_state

        def _record_save(state, path):
            written.append(Path(path))
            real_save(state, path)

        monkeypatch.setattr(pm, "save_state", _record_save)

        report = _report(update=True)

        assert written == []
        assert SUMMARY_HEADING in report

    def test_the_committed_defaults_file_is_never_written(self, tmp_path):
        before = DEFAULTS_PATH.read_bytes()
        state_path = tmp_path / STATE_FILENAME

        _report(metrics_state=state_path, update=True)

        assert DEFAULTS_PATH.read_bytes() == before


# ---------------------------------------------------------------------------
# Rendered baseline/target reflect the run's history
# ---------------------------------------------------------------------------


class TestRenderedHistory:
    def test_target_and_baseline_reflect_the_measurement_this_run_recorded(
        self, tmp_path
    ):
        # fallback_rate is lower-is-better: baseline = lowest value, target =
        # mean of the three lowest. Two prior windows (20.0, 22.0) plus this
        # window's 33.3 make three, so the row stops being "Insufficient
        # history" and its target is the mean of exactly those three.
        state_path = tmp_path / STATE_FILENAME
        _seed(
            state_path,
            "fallback_rate",
            [("2026-07-01", 20.0, FINGERPRINT), ("2026-07-15", 22.0, FINGERPRINT)],
        )

        report = _report(metrics_state=state_path, update=True)
        cells = _cells(report, "Fallback rate")
        state = pm.load_state(state_path)
        leading = [
            _measurements(state, "fallback_rate")[index]["value"] for index in (0, 1)
        ]

        assert cells[BASELINE] == "20.0 % of requests (2026-07-01)"
        assert cells[TARGET] == f"{(20.0 + 22.0 + 33.3) / 3:.1f} % of requests"
        assert cells[ACTUAL] == "33.3 % of requests"
        assert cells[RESULT] == "Unmet", "33.3 % is above the best-3 target and its band"
        assert leading == [20.0, 22.0], "prior windows are retained"

    def test_a_three_measurement_history_met_verdict_matches_the_engine(
        self, tmp_path
    ):
        state_path = tmp_path / STATE_FILENAME
        _seed(
            state_path,
            "fallback_rate",
            [("2026-07-01", 60.0, FINGERPRINT), ("2026-07-15", 64.0, FINGERPRINT)],
        )

        report = _report(metrics_state=state_path, update=True)
        cells = _cells(report, "Fallback rate")
        state = pm.load_state(state_path)
        expected = pm.derive_verdict(
            _measurements(state, "fallback_rate")[-1]["value"],
            _measurements(state, "fallback_rate"),
            pm.LOWER,
        )

        assert cells[RESULT] == expected == "Met"
        assert cells[TARGET] != "n/a", "a three-measurement target must render"

    def test_an_unknown_configuration_fingerprint_is_recorded_as_unknown(
        self, tmp_path
    ):
        state_path = tmp_path / STATE_FILENAME

        reporting.build_report(
            _summary(), None, metrics_state=state_path, update_metrics=True
        )

        state = pm.load_state(state_path)
        assert {
            measurement["fingerprint"]
            for measurement in _measurements(state, "fallback_rate")
        } == {pm.UNKNOWN_FINGERPRINT}


# ---------------------------------------------------------------------------
# Cross-era note
# ---------------------------------------------------------------------------


class TestCrossEraNote:
    def test_note_names_every_fingerprint_the_best_three_span(self, tmp_path):
        state_path = tmp_path / STATE_FILENAME
        other = pm.configuration_fingerprint(1, 262144)
        _seed(
            state_path,
            "decode_speed",
            [
                ("2026-07-01", 41.2, FINGERPRINT),
                ("2026-07-02", 42.0, FINGERPRINT),
                ("2026-07-03", 39.0, other),
            ],
        )

        report = _report(metrics_state=state_path)

        section = report.split(SUMMARY_HEADING, 1)[1].split("\n## ", 1)[0]
        assert "Cross-era note" in section
        assert FINGERPRINT in section.split("Cross-era note", 1)[1]
        assert other in section.split("Cross-era note", 1)[1]
        assert "Decode speed" in section.split("Cross-era note", 1)[1]

    def test_no_note_when_one_era_or_no_history(self, tmp_path):
        same_era = tmp_path / "same.yaml"
        _seed(
            same_era,
            "decode_speed",
            [
                ("2026-07-01", 41.2, FINGERPRINT),
                ("2026-07-02", 42.0, FINGERPRINT),
                ("2026-07-03", 39.0, FINGERPRINT),
            ],
        )
        backfilled = tmp_path / "backfilled.yaml"
        _seed(
            backfilled,
            "decode_speed",
            [
                ("2026-07-01", 41.2, pm.UNKNOWN_FINGERPRINT),
                ("2026-07-02", 42.0, pm.UNKNOWN_FINGERPRINT),
                ("2026-07-03", 39.0, pm.UNKNOWN_FINGERPRINT),
            ],
        )

        for state_path in (same_era, backfilled, tmp_path / "absent.yaml"):
            report = _report(metrics_state=state_path)
            assert "Cross-era note" not in report, state_path.name


# ---------------------------------------------------------------------------
# Direction conversion
# ---------------------------------------------------------------------------


class TestPrefillEtaAccuracyDirection:
    @pytest.mark.parametrize(
        ("error", "expected"),
        [(0.0, 100.0), (40.0, 60.0), (100.0, 0.0), (140.0, 0.0)],
    )
    def test_the_recorded_actual_is_accuracy_not_the_error(
        self, tmp_path, monkeypatch, error, expected
    ):
        # The aggregation helper reports a median absolute *relative error*
        # (lower is better) while the registry declares the metric
        # higher-is-better; recording the error would invert best-3 retention,
        # so the scorecard records 100 - error (clamped at 0).
        monkeypatch.setattr(
            aggregation,
            "performance_metric_inputs",
            lambda summary, config, mode_map: {"prefill_eta_accuracy": error},
        )
        state_path = tmp_path / STATE_FILENAME

        report = _report(metrics_state=state_path, update=True)

        state = pm.load_state(state_path)
        assert [
            measurement["value"]
            for measurement in _measurements(state, "prefill_eta_accuracy")
        ] == [expected]
        assert _cells(report, "Prefill ETA accuracy")[ACTUAL] == (
            f"{expected:.1f} %"
        )


# ---------------------------------------------------------------------------
# Tolerance column (LP-0MWADHS83006QFQS)
# ---------------------------------------------------------------------------


class TestToleranceColumn:
    def test_every_row_renders_the_default_ten_percent_band(self):
        report = _report()

        for metric in pm.METRIC_REGISTRY:
            assert _cells(report, metric.label)[TOLERANCE] == "±10 %", metric.id

    def test_the_header_names_target_tolerance_actual_and_result(self):
        section = _report().split(SUMMARY_HEADING, 1)[1].split("\n## ", 1)[0]
        header = next(line for line in section.splitlines() if line.startswith("|"))

        assert [cell.strip() for cell in header.strip("|").split("|")] == [
            "Metric",
            "Baseline",
            "Target",
            "Tolerance",
            "Actual",
            "Result",
        ]

    def test_the_paragraph_explains_the_tolerance_band(self):
        section = _report().split(SUMMARY_HEADING, 1)[1].split("\n## ", 1)[0]

        assert "within the tolerance band of the target" in section
        assert "±10 %" in section

    def test_a_per_metric_override_changes_the_rendered_band(self, tmp_path):
        config = tmp_path / "custom_metrics.yaml"
        config.write_text(
            "metrics:\n  fallback_rate:\n    tolerance_pct: 25\n", encoding="utf-8"
        )

        report = _report(metrics_config=config)

        assert _cells(report, "Fallback rate")[TOLERANCE] == "±25 %"
        # Every other metric keeps the default band.
        assert _cells(report, "Decode speed")[TOLERANCE] == "±10 %"

    def test_the_verdict_uses_the_band_not_an_exact_comparison(self, tmp_path):
        # fallback_rate is lower-is-better: three seeded 30.5 % windows give a
        # 30.5 % target whose band edge is 33.55 %. This window's 33.3 % is
        # worse than the target yet inside the band, so it must read Met. A
        # 30.0 % seed makes the edge 33.0 %, which the same actual misses.
        inside = tmp_path / "inside.yaml"
        _seed(
            inside,
            "fallback_rate",
            [
                ("2026-07-01", 30.5, FINGERPRINT),
                ("2026-07-02", 30.5, FINGERPRINT),
                ("2026-07-03", 30.5, FINGERPRINT),
            ],
        )
        outside = tmp_path / "outside.yaml"
        _seed(
            outside,
            "fallback_rate",
            [
                ("2026-07-01", 30.0, FINGERPRINT),
                ("2026-07-02", 30.0, FINGERPRINT),
                ("2026-07-03", 30.0, FINGERPRINT),
            ],
        )

        inside_cells = _cells(_report(metrics_state=inside), "Fallback rate")
        outside_cells = _cells(_report(metrics_state=outside), "Fallback rate")

        assert inside_cells[ACTUAL] == "33.3 % of requests"
        assert inside_cells[RESULT] == "Met", (
            "33.3 % is worse than the 30.5 % target but inside its ±10 % band"
        )
        assert outside_cells[RESULT] == "Unmet", (
            "33.3 % is outside the 30.0 % target's ±10 % band"
        )
