"""Unit specification for the performance-metrics engine and backfill parser.

Fail-first contract for ``scripts/performance_metrics.py``, the module delivery
of LP-0MW0FU5Z6006A4P3 (child LP-0MW1BU9F8007SE0U) and its backfill parser
(child LP-0MW1BUB1Z004AK21). It fixes the behaviour the engine must satisfy so
the implementation is driven by these tests rather than the other way round.

Required public API
-------------------

Registry (ordered, exactly the ten parent-specified metrics)::

    METRIC_REGISTRY: Sequence[Metric]
        Metric.id         -> str   one of the ten ids below
        Metric.label      -> str   human-readable label (non-empty)
        Metric.unit       -> str   unit of measurement (non-empty)
        Metric.direction  -> str   "higher" (higher is better) or "lower"
        Metric.source     -> str   data source (non-empty)

State (a plain mapping, YAML round-trippable)::

    {"metrics": {"<metric_id>": {"measurements": [measurement, ...]}}}

    measurement = {"window": str, "value": float, "date": str,
                   "fingerprint": str}

Functions::

    new_state() -> dict
    load_state(path, *, defaults_path=None) -> dict      # never writes
    save_state(state, path) -> None
    record_measurement(state, metric_id, *, window, value, date=None,
                       fingerprint=None) -> None         # in place
    retained(measurements, direction, limit=3) -> list[dict]  # best first
    target_from(measurements, direction) -> float | None      # None below 3
    baseline_from(measurements, direction) -> dict | None     # best value+date
    derive_verdict(actual, measurements, direction) -> str
    configuration_fingerprint(slot_count, per_slot_context) -> str
    is_cross_era(measurements) -> bool
    parse_archived_report(report_text) -> dict[str, float]    # omit unparseable
    archived_window_id(report_text, fallback) -> str          # Generated date,
        # else the archive directory name
    seed_history(state, reports) -> dict                      # reports:
        # iterable of (window_id, report_text); returns at least
        # {"seeded": int, "already_present": int, "skipped_metrics": dict}

``target_from`` returns ``None`` below three measurements; the summary renderer
prints that as ``n/a`` (pinned by the report integration item
LP-0MW1BU93N0022IYJ). ``derive_verdict`` returns the literals ``"Met"``,
``"Unmet"``, ``"Insufficient history"`` and ``n/a``.

Fail-first window
-----------------

The engine lands in LP-0MW1BU9F8007SE0U, which is blocked by this item. Until
``scripts/performance_metrics.py`` exists every test in this module is marked
skipped (the module still *collects*, so the suite exits clean, not with
pytest's "no tests collected" status), and no test has to be edited for the
specification to start running once the engine lands. The guard is on the
*file*, not an ``ImportError``, so a present-but-broken module fails loudly at
collection instead of silently skipping.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from tests.fixtures import (  # noqa: E402
    ARCHIVED_REPORT_ERA_A,
    ARCHIVED_REPORT_ERA_B,
    ARCHIVED_REPORT_ERA_SECONDS,
)

ENGINE_PATH = SCRIPTS / "performance_metrics.py"
ENGINE_AVAILABLE = ENGINE_PATH.exists()

if ENGINE_AVAILABLE:  # pragma: no cover - fail-first window
    import performance_metrics as pm
else:  # pragma: no cover - fail-first window
    pm = None

# Skip per test rather than per module: a module-level skip collects nothing and
# pytest then exits 5 ("no tests collected"), which the deterministic test gate
# treats as a red suite.
pytestmark = pytest.mark.skipif(
    not ENGINE_AVAILABLE,
    reason="performance_metrics engine not implemented yet (LP-0MW1BU9F8007SE0U)",
)

HIGHER = "higher"
LOWER = "lower"

#: The ten metrics from LP-0MW0FU5Z6006A4P3 with their direction. Higher-is-
#: better metrics improve as the number grows; lower-is-better metrics improve
#: as it falls.
TEN_METRICS = {
    "gpu_busy": HIGHER,
    "fallback_rate": LOWER,
    "decode_speed": HIGHER,
    "prompt_eval_speed": HIGHER,
    "time_to_first_token": LOWER,
    "stream_error_rate": LOWER,
    "context_pressure": LOWER,
    "prompt_cache_hit_ratio": HIGHER,
    "contention_denials_per_hour": LOWER,
    "prefill_eta_accuracy": HIGHER,
}

FP = "slots=2,ctx=131072"


def measurement(value: float, window: str, *, fingerprint: str = FP, date: str | None = None) -> dict:
    """Build a stored measurement in the schema the engine persists."""
    return {
        "window": window,
        "value": value,
        "date": date or window,
        "fingerprint": fingerprint,
    }


def registry_entry(metric_id: str):
    for metric in pm.METRIC_REGISTRY:
        if metric.id == metric_id:
            return metric
    raise AssertionError(f"{metric_id} missing from METRIC_REGISTRY")


# Five measurements per direction. The three best are deliberately not the three
# most recent, so retention cannot pass by keeping the newest three.
HIGHER_MEASUREMENTS = [
    measurement(25.8, "2026-08-07"),
    measurement(41.2, "2026-09-02"),
    measurement(33.3, "2026-09-14"),
    measurement(47.9, "2026-10-06"),
    measurement(30.1, "2026-10-21"),
]
HIGHER_BEST_THREE = [(47.9, "2026-10-06"), (41.2, "2026-09-02"), (33.3, "2026-09-14")]
HIGHER_TARGET = (47.9 + 41.2 + 33.3) / 3


LOWER_MEASUREMENTS = [
    measurement(670.0, "2026-08-07"),
    measurement(412.0, "2026-09-02"),
    measurement(900.0, "2026-09-14"),
    measurement(355.0, "2026-10-06"),
    measurement(800.0, "2026-10-21"),
]
LOWER_BEST_THREE = [(355.0, "2026-10-06"), (412.0, "2026-09-02"), (670.0, "2026-08-07")]
LOWER_TARGET = (355.0 + 412.0 + 670.0) / 3

DEFAULTS_YAML = """\
# Committed performance-metrics defaults: read-only at runtime.
metrics:
  decode_speed:
    measurements:
      - window: "2026-06-01"
        value: 31.5
        date: "2026-06-01"
        fingerprint: "slots=2,ctx=131072"
"""


# --- Registry ---------------------------------------------------------------


def test_registry_exposes_exactly_the_ten_parent_metrics():
    ids = [metric.id for metric in pm.METRIC_REGISTRY]

    assert len(ids) == 10
    assert len(set(ids)) == 10, "metric ids must be unique"
    assert set(ids) == set(TEN_METRICS)


@pytest.mark.parametrize("metric_id", sorted(TEN_METRICS))
def test_registry_entry_declares_its_contract_fields(metric_id):
    metric = registry_entry(metric_id)

    assert metric.label.strip(), "every metric needs a label"
    assert metric.unit.strip(), "every metric needs a unit"
    assert metric.source.strip(), "every metric needs a data source"
    assert metric.direction in {HIGHER, LOWER}


@pytest.mark.parametrize(("metric_id", "direction"), sorted(TEN_METRICS.items()))
def test_registry_direction_matches_the_parent_definition(metric_id, direction):
    assert registry_entry(metric_id).direction == direction


def test_gpu_busy_names_its_hardware_scope_and_keeps_a_comparable_history():
    """AC4 (LP-0MW5K8YE9001QO39): the scorecard label names the hardware
    scope (power-based, any GPU consumer) rather than reading as local model
    utilisation, while the unit and direction stay unchanged so the recorded
    history remains comparable."""
    metric = registry_entry("gpu_busy")

    assert "power-based" in metric.label
    assert "any consumer" in metric.label
    assert "any GPU consumer" in metric.source
    assert metric.unit == "%"
    assert metric.direction == HIGHER


# --- Direction-aware retention ---------------------------------------------


def test_retention_keeps_the_three_best_when_higher_is_better():
    retained = pm.retained(HIGHER_MEASUREMENTS, HIGHER)

    assert [(entry["value"], entry["date"]) for entry in retained] == HIGHER_BEST_THREE


def test_retention_keeps_the_three_best_when_lower_is_better():
    retained = pm.retained(LOWER_MEASUREMENTS, LOWER)

    assert [(entry["value"], entry["date"]) for entry in retained] == LOWER_BEST_THREE


def test_retention_keeps_every_measurement_below_the_limit():
    two = HIGHER_MEASUREMENTS[:2]

    assert len(pm.retained(two, HIGHER)) == 2
    assert len(pm.retained([], HIGHER)) == 0


# --- Target maths -----------------------------------------------------------


@pytest.mark.parametrize(
    ("measurements", "direction", "expected"),
    [
        (HIGHER_MEASUREMENTS, HIGHER, HIGHER_TARGET),
        (LOWER_MEASUREMENTS, LOWER, LOWER_TARGET),
    ],
)
def test_target_is_the_mean_of_the_retained_three(measurements, direction, expected):
    assert pm.target_from(measurements, direction) == pytest.approx(expected)


@pytest.mark.parametrize("direction", [HIGHER, LOWER])
@pytest.mark.parametrize("count", [0, 1, 2])
def test_target_is_unavailable_below_three_measurements(direction, count):
    measurements = HIGHER_MEASUREMENTS[:count]

    assert pm.target_from(measurements, direction) is None


@pytest.mark.parametrize(
    ("measurements", "direction", "value", "date"),
    [
        (HIGHER_MEASUREMENTS, HIGHER, 47.9, "2026-10-06"),
        (LOWER_MEASUREMENTS, LOWER, 355.0, "2026-10-06"),
    ],
)
def test_baseline_is_the_single_best_value_with_its_date(measurements, direction, value, date):
    assert pm.baseline_from(measurements, direction) == {"value": value, "date": date}


# --- Verdict derivation -----------------------------------------------------


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (45.0, "Met"),
        (30.0, "Unmet"),
    ],
)
def test_higher_is_better_is_met_when_actual_reaches_the_target(actual, expected):
    assert pm.derive_verdict(actual, HIGHER_MEASUREMENTS, HIGHER) == expected


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (400.0, "Met"),
        (480.0, "Unmet"),
    ],
)
def test_lower_is_better_is_met_when_actual_stays_under_the_target(actual, expected):
    assert pm.derive_verdict(actual, LOWER_MEASUREMENTS, LOWER) == expected


@pytest.mark.parametrize(
    ("measurements", "direction"),
    [(HIGHER_MEASUREMENTS, HIGHER), (LOWER_MEASUREMENTS, LOWER)],
)
def test_equality_with_the_target_counts_as_met(measurements, direction):
    # Compare against the value the engine itself derives: re-summing the best
    # three in a test constant would make the verdict a float-associativity
    # problem rather than an equality-semantics one.
    target = pm.target_from(measurements, direction)
    worse = target - 0.1 if direction == HIGHER else target + 0.1
    better = target + 0.1 if direction == HIGHER else target - 0.1

    assert pm.derive_verdict(target, measurements, direction) == "Met"
    assert pm.derive_verdict(better, measurements, direction) == "Met"
    assert pm.derive_verdict(worse, measurements, direction) == "Unmet"


@pytest.mark.parametrize("direction", [HIGHER, LOWER])
@pytest.mark.parametrize("count", [0, 1, 2])
def test_fewer_than_three_measurements_never_yields_a_verdict(direction, count):
    measurements = HIGHER_MEASUREMENTS[:count]

    # Even a perfect-looking actual must not produce a Met/Unmet verdict.
    assert pm.derive_verdict(0.0, measurements, direction) == "Insufficient history"
    assert pm.derive_verdict(1_000_000.0, measurements, direction) == "Insufficient history"


def test_a_missing_actual_renders_not_available():
    assert pm.derive_verdict(None, HIGHER_MEASUREMENTS, HIGHER) == "n/a"


# --- State ------------------------------------------------------------------


def test_new_state_covers_every_registered_metric():
    state = pm.new_state()

    assert set(state["metrics"]) == set(TEN_METRICS)
    assert all(state["metrics"][metric_id]["measurements"] == [] for metric_id in TEN_METRICS)


def test_state_round_trips_through_save_and_load(tmp_path):
    state = pm.new_state()
    pm.record_measurement(state, "decode_speed", window="2026-11-01", value=41.2, date="2026-11-01", fingerprint=FP)
    pm.record_measurement(
        state, "time_to_first_token", window="2026-11-01", value=410.0, date="2026-11-01", fingerprint=FP
    )
    path = tmp_path / "performance_metrics_state.yaml"

    pm.save_state(state, path)

    assert path.exists()
    assert pm.load_state(path) == state


def test_committed_defaults_file_is_never_mutated(tmp_path):
    defaults = tmp_path / "performance_metrics_defaults.yaml"
    defaults.write_text(DEFAULTS_YAML, encoding="utf-8")
    before = defaults.read_bytes()
    state_path = tmp_path / "missing_state.yaml"

    state = pm.load_state(state_path, defaults_path=defaults)

    assert [entry["value"] for entry in state["metrics"]["decode_speed"]["measurements"]] == [31.5]
    assert not state_path.exists(), "loading must never create or write the state file"

    pm.record_measurement(state, "decode_speed", window="2026-11-01", value=37.8, date="2026-11-01", fingerprint=FP)
    output_state = tmp_path / "out" / "performance_metrics_state.yaml"
    pm.save_state(state, output_state)

    assert output_state.exists()
    assert defaults.read_bytes() == before


def test_measurement_records_its_window_and_configuration_fingerprint():
    state = pm.new_state()

    pm.record_measurement(
        state, "decode_speed", window="2026-11-02", value=41.2, date="2026-11-02", fingerprint=FP
    )

    stored = state["metrics"]["decode_speed"]["measurements"][-1]
    assert stored["window"] == "2026-11-02"
    assert stored["value"] == pytest.approx(41.2)
    assert stored["date"] == "2026-11-02"
    assert stored["fingerprint"] == FP


def test_duplicate_window_is_not_counted_twice():
    state = pm.new_state()

    pm.record_measurement(state, "decode_speed", window="2026-11-02", value=41.2, date="2026-11-02", fingerprint=FP)
    pm.record_measurement(state, "decode_speed", window="2026-11-02", value=43.0, date="2026-11-02", fingerprint=FP)

    retained = pm.retained(state["metrics"]["decode_speed"]["measurements"], HIGHER)
    assert [entry["window"] for entry in retained] == ["2026-11-02"]


# --- Configuration fingerprint ---------------------------------------------


def test_configuration_fingerprint_distinguishes_slot_geometry():
    base = pm.configuration_fingerprint(2, 131072)

    assert isinstance(base, str)
    assert base.strip()
    assert pm.configuration_fingerprint(2, 131072) == base
    assert pm.configuration_fingerprint(1, 262144) != base
    assert pm.configuration_fingerprint(2, 65536) != base


def test_best_three_spanning_two_fingerprints_is_flagged_cross_era():
    same_era = [
        measurement(30.0, "2026-10-01"),
        measurement(31.0, "2026-10-02"),
        measurement(32.0, "2026-10-03"),
    ]
    mixed = [
        measurement(31.0, "2026-10-02"),
        measurement(32.0, "2026-10-03"),
        measurement(33.0, "2026-09-01", fingerprint="slots=1,ctx=262144"),
    ]

    assert not pm.is_cross_era(same_era)
    assert pm.is_cross_era(mixed)


# --- Backfill parser --------------------------------------------------------


def test_backfill_parser_reads_the_era_a_day_night_report():
    parsed = pm.parse_archived_report(ARCHIVED_REPORT_ERA_A)

    assert parsed["fallback_rate"] == pytest.approx(79.7)
    assert parsed["decode_speed"] == pytest.approx(25.8)
    assert parsed["prompt_eval_speed"] == pytest.approx(137.2)


def test_backfill_parser_reads_the_era_b_standard_low_report():
    parsed = pm.parse_archived_report(ARCHIVED_REPORT_ERA_B)

    assert parsed["fallback_rate"] == pytest.approx(48.1)
    assert parsed["decode_speed"] == pytest.approx(37.8)
    assert parsed["prompt_eval_speed"] == pytest.approx(193.3)
    assert parsed["time_to_first_token"] == pytest.approx(670.0)
    assert parsed["contention_denials_per_hour"] == pytest.approx(12.8)


#: The era-B-style TTFT table with its p90 replaced, so the unit-conversion
#: contract can be exercised cell by cell without another full report fixture.
_TTFT_TOTAL_ROW = "| Total | 3294 | 358ms | 4.4s | 43.7s |"


def _report_with_ttft_p90(cell: str) -> str:
    assert _TTFT_TOTAL_ROW in ARCHIVED_REPORT_ERA_SECONDS
    return ARCHIVED_REPORT_ERA_SECONDS.replace(
        _TTFT_TOTAL_ROW, f"| Total | 3294 | 358ms | 4.4s | {cell} |"
    )


def test_backfill_parser_converts_the_seconds_era_ttft_to_milliseconds():
    """LP-0MW20RKLE006TG4M: the 2026-09-17-era report renders the p90 as
    ``43.7s``. Blindly taking the number seeds 43.7 ms — a sub-second value
    that permanently wins the lower-is-better best-3 — so the cell must be
    converted to the registry's milliseconds unit."""
    parsed = pm.parse_archived_report(ARCHIVED_REPORT_ERA_SECONDS)

    assert parsed["time_to_first_token"] == pytest.approx(43700.0)
    assert parsed["time_to_first_token"] > 1000.0, "a seconds-era value must never seed sub-second"


@pytest.mark.parametrize(
    ("cell", "expected_ms"),
    [
        ("43.7s", 43700.0),
        ("12.5 sec", 12500.0),
        ("670ms", 670.0),
        ("48971.5 ms", 48971.5),
        ("670", 670.0),
    ],
)
def test_backfill_parser_converts_ttft_duration_units_to_milliseconds(cell, expected_ms):
    parsed = pm.parse_archived_report(_report_with_ttft_p90(cell))

    assert parsed["time_to_first_token"] == pytest.approx(expected_ms)


@pytest.mark.parametrize("cell", ["43.7x", "43.7 ms/req", "~43.7", "fast"])
def test_backfill_parser_refuses_an_unrecognised_ttft_unit(cell):
    """An unrecognised suffix is skipped, never guessed: a wrong unit must not
    seed history at all rather than seed a mis-scaled value."""
    parsed = pm.parse_archived_report(_report_with_ttft_p90(cell))

    assert "time_to_first_token" not in parsed
    assert parsed["decode_speed"] == pytest.approx(29.4), "other metrics still parse"


def test_backfill_parser_returns_no_measurement_for_metrics_it_cannot_parse():
    parsed = pm.parse_archived_report(ARCHIVED_REPORT_ERA_A)

    # Era A renders neither a time-to-first-token nor a contention section, so
    # those metrics are skipped rather than guessed.
    assert "time_to_first_token" not in parsed
    assert "contention_denials_per_hour" not in parsed
    assert set(parsed) <= set(TEN_METRICS)
    assert all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in parsed.values())


# --- Backfill seeding -------------------------------------------------------


def _measurement_count(state, metric_id: str) -> int:
    return len(state["metrics"][metric_id]["measurements"])


def test_backfill_is_idempotent_over_the_same_window():
    state = pm.new_state()
    reports = [("2026-10-31", ARCHIVED_REPORT_ERA_B)]

    first = pm.seed_history(state, reports)
    counts_after_first = {metric_id: _measurement_count(state, metric_id) for metric_id in TEN_METRICS}
    second = pm.seed_history(state, reports)

    assert first["seeded"] > 0
    assert first["already_present"] == 0
    assert second["seeded"] == 0
    assert second["already_present"] == first["seeded"]
    assert {metric_id: _measurement_count(state, metric_id) for metric_id in TEN_METRICS} == counts_after_first


def test_backfill_records_the_archived_window_and_date():
    state = pm.new_state()

    pm.seed_history(state, [("2026-10-31", ARCHIVED_REPORT_ERA_B)])

    stored = state["metrics"]["decode_speed"]["measurements"][0]
    assert stored["window"] == "2026-10-31"
    assert stored["date"]
    assert stored["value"] == pytest.approx(37.8)


def test_backfill_reports_the_metrics_it_could_not_parse():
    state = pm.new_state()

    result = pm.seed_history(state, [("2026-08-07", ARCHIVED_REPORT_ERA_A)])

    skipped = result["skipped_metrics"]
    assert "time_to_first_token" in skipped
    assert "contention_denials_per_hour" in skipped
    assert _measurement_count(state, "time_to_first_token") == 0
    assert _measurement_count(state, "decode_speed") == 1


def test_archived_window_id_uses_the_generated_date():
    # Era A's header renders "Generated: 2026-08-07 02:10:21", so the report's
    # own date names the window even when the archive directory carries a
    # run-to-run suffix.
    assert pm.archived_window_id(ARCHIVED_REPORT_ERA_A, "2026-08-07_2") == "2026-08-07"


def test_archived_window_id_falls_back_to_the_directory_name_when_generated_is_missing():
    without_header = ARCHIVED_REPORT_ERA_A.replace(
        "- Generated: 2026-08-07 02:10:21\n", ""
    )

    assert pm.archived_window_id(without_header, "2026-08-07") == "2026-08-07"
    assert (
        pm.archived_window_id(without_header, "2026-08-07_2") == "2026-08-07_2"
    )
