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
    derive_verdict(actual, measurements, direction,
                   tolerance_pct=DEFAULT_TOLERANCE_PCT) -> str
    tolerance_for(metric_id, defaults=None) -> float          # band, 10 % default
    DEFAULT_TOLERANCE_PCT -> float                            # 10.0
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
``"Unmet"``, ``"Insufficient history"`` and ``n/a``, and judges the actual
against the target inside the metric's tolerance band (LP-0MWADHS83006QFQS).

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
    ARCHIVED_REPORT_ERA_PREFILL,
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


def test_gpu_busy_names_its_attributable_scope():
    """LP-0MWG7KQVZ008DBZM: the scorecard label names the attributable scope
    (power ≥ 90 W **and** a busy local slot, our local model's work) rather
    than the retired power-only ``gpu:computing:bool`` figure. Unit and
    direction are unchanged so the reset history stays comparable within one
    definition."""
    metric = registry_entry("gpu_busy")

    assert "attributable" in metric.label
    assert "power" in metric.label and "busy local slot" in metric.label
    assert "gpu:computing_workload:bool" in metric.source
    assert "scripts/gpu_utilisation.py" in metric.source
    assert "any consumer" not in metric.label
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
def test_higher_is_better_is_met_within_the_tolerance_band(actual, expected):
    assert pm.derive_verdict(actual, HIGHER_MEASUREMENTS, HIGHER) == expected


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (400.0, "Met"),
        (600.0, "Unmet"),
    ],
)
def test_lower_is_better_is_met_within_the_tolerance_band(actual, expected):
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
    better = target + 0.1 if direction == HIGHER else target - 0.1

    assert pm.derive_verdict(target, measurements, direction) == "Met"
    assert pm.derive_verdict(better, measurements, direction) == "Met"


# --- Tolerance band (LP-0MWADHS83006QFQS) -----------------------------------
#
# A target is met when the actual is inside a relative band of it (default
# ±10 %), not only when it beats it exactly. The band is direction-aware and
# both edges count as met, so run-to-run measurement noise is not reported as a
# regression.


def test_the_default_tolerance_is_ten_percent():
    assert pm.DEFAULT_TOLERANCE_PCT == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("measurements", "direction"),
    [(HIGHER_MEASUREMENTS, HIGHER), (LOWER_MEASUREMENTS, LOWER)],
)
def test_an_actual_just_inside_the_band_is_met(measurements, direction):
    target = pm.target_from(measurements, direction)
    band = pm.DEFAULT_TOLERANCE_PCT / 100.0
    inside = target * (1 - band) if direction == HIGHER else target * (1 + band)

    assert pm.derive_verdict(inside, measurements, direction) == "Met"


@pytest.mark.parametrize(
    ("measurements", "direction"),
    [(HIGHER_MEASUREMENTS, HIGHER), (LOWER_MEASUREMENTS, LOWER)],
)
def test_the_band_edge_counts_as_met(measurements, direction):
    target = pm.target_from(measurements, direction)
    band = pm.DEFAULT_TOLERANCE_PCT / 100.0
    # Derived from the engine's own target so the assertion is about band
    # semantics, not float associativity.
    edge = target * (1 - band) if direction == HIGHER else target * (1 + band)

    assert pm.derive_verdict(edge, measurements, direction) == "Met"


@pytest.mark.parametrize(
    ("measurements", "direction"),
    [(HIGHER_MEASUREMENTS, HIGHER), (LOWER_MEASUREMENTS, LOWER)],
)
def test_an_actual_just_outside_the_band_is_unmet(measurements, direction):
    target = pm.target_from(measurements, direction)
    band = pm.DEFAULT_TOLERANCE_PCT / 100.0
    outside = (
        target * (1 - band) - 1e-6
        if direction == HIGHER
        else target * (1 + band) + 1e-6
    )

    assert pm.derive_verdict(outside, measurements, direction) == "Unmet"


@pytest.mark.parametrize(
    ("measurements", "direction"),
    [(HIGHER_MEASUREMENTS, HIGHER), (LOWER_MEASUREMENTS, LOWER)],
)
def test_an_actual_better_than_the_target_is_met(measurements, direction):
    target = pm.target_from(measurements, direction)
    better = target * 2 if direction == HIGHER else target / 2

    assert pm.derive_verdict(better, measurements, direction) == "Met"


def test_a_zero_target_has_a_zero_width_band():
    # A lower-is-better metric whose target is 0: 10 % of 0 is 0, so only an
    # actual of 0 is inside the band. Documented, never replaced by an invented
    # absolute epsilon.
    measurements = [
        measurement(0.0, "2026-07-01"),
        measurement(0.0, "2026-07-02"),
        measurement(0.0, "2026-07-03"),
    ]

    assert pm.target_from(measurements, LOWER) == pytest.approx(0.0)
    assert pm.derive_verdict(0.0, measurements, LOWER) == "Met"
    assert pm.derive_verdict(0.1, measurements, LOWER) == "Unmet"


# --- Strict Fallback-rate rule (LP-0MWJOUTOV000G701) ------------------------
#
# The producer approved a strict verdict for the lower-is-better Fallback-rate
# metric only: `Met` if and only if the window's actual is at or below the
# target (`actual <= target`), no +tolerance band above the target, while every
# other metric keeps its band (parent LP-0MWEOGIUK006B1DD Q1 answer (a)). The
# strictness is expressed through the same per-metric tolerance machinery the
# scorecard uses — the committed defaults' ``tolerance_pct`` read by
# :func:`tolerance_for` and handed to :func:`derive_verdict` by
# ``reporting._scorecard_rows`` — not a new comparison surface, so the
# scorecard and the ``--json`` block cannot diverge.
#
# Red-first: the above-target assertion fails against the current ±10 % band
# and is marked ``xfail(strict=True)`` naming the implementation child
# (LP-0MWJOUY4E002FNHP) that narrows ``fallback_rate.tolerance_pct``; the tree
# stays green for the dev-push gate. The at/below-target and per-metric-band
# assertions pass under both bands and run unmarked — they pin the sides of
# the boundary that must not move either way.

PENDING_STRICT_FALLBACK_VERDICT = (
    "pending LP-0MWJOUY4E002FNHP (strict Fallback-rate verdict: committed "
    "defaults narrow fallback_rate.tolerance_pct so only actual <= target "
    "reads Met)"
)

#: The decision example from the parent item: a 64.9 % target whose former
#: ±10 % band edge is 71.39 % with the window's 66.3 % inside that band.
FALLBACK_TARGET = 64.9
FALLBACK_TARGET_MEASUREMENTS = [
    measurement(FALLBACK_TARGET, "2026-09-01"),
    measurement(FALLBACK_TARGET, "2026-09-02"),
    measurement(FALLBACK_TARGET, "2026-09-03"),
]


@pytest.mark.xfail(strict=True, reason=PENDING_STRICT_FALLBACK_VERDICT)
def test_fallback_rate_is_unmet_above_the_target_inside_the_former_band():
    """The boundary the current ±10 % band gets wrong: 66.3 % against a
    64.9 % target is below the 71.39 % band edge, so the lax rule reads `Met`;
    the approved strict rule (at or below the target) reads `Unmet`."""
    assert (
        pm.derive_verdict(
            66.3,
            FALLBACK_TARGET_MEASUREMENTS,
            LOWER,
            tolerance_pct=pm.tolerance_for("fallback_rate"),
        )
        == "Unmet"
    )


def test_fallback_rate_is_met_when_the_actual_equals_the_target():
    """Boundary equality stays `Met` under the band and under the strict rule.

    Compared against the value the engine itself derives, re-summing the best
    three in a test constant would make the verdict a float-associativity
    problem rather than an equality-semantics one.
    """
    target = pm.target_from(FALLBACK_TARGET_MEASUREMENTS, LOWER)

    assert (
        pm.derive_verdict(
            target,
            FALLBACK_TARGET_MEASUREMENTS,
            LOWER,
            tolerance_pct=pm.tolerance_for("fallback_rate"),
        )
        == "Met"
    )


def test_fallback_rate_is_met_when_the_actual_is_below_the_target():
    assert (
        pm.derive_verdict(
            FALLBACK_TARGET - 5.0,
            FALLBACK_TARGET_MEASUREMENTS,
            LOWER,
            tolerance_pct=pm.tolerance_for("fallback_rate"),
        )
        == "Met"
    )


LOWER_IS_BETTER_IDS = [
    metric.id
    for metric in pm.METRIC_REGISTRY
    if metric.direction == LOWER and metric.id != "fallback_rate"
]


@pytest.mark.parametrize("metric_id", LOWER_IS_BETTER_IDS)
def test_every_other_lower_is_better_metric_keeps_its_tolerance_band(metric_id):
    """The strictness is per-metric: 5 % above the target still reads `Met`
    for every other lower-is-better metric under its own committed band."""
    target = pm.target_from(LOWER_MEASUREMENTS, LOWER)

    assert (
        pm.derive_verdict(
            target * 1.05,
            LOWER_MEASUREMENTS,
            LOWER,
            tolerance_pct=pm.tolerance_for(metric_id),
        )
        == "Met"
    )


# --- Prefill-ETA raw error metric (LP-0MWJOUV5P007R799) ---------------------
#
# Parent LP-0MWEOGIUK006B1DD Q3 answer (a): the producer approved reporting the
# median absolute relative **error** for `prefill_eta_accuracy` directly and
# registering the metric lower-is-better, so a smaller error is better. The
# aggregation helper already returns that error (p50 ≈ 1041 % on recent
# windows); the scorecard instead converts it to `max(0, 100 − error)` for a
# higher-is-better entry, so every window reads ~0 % and best-3 retention keeps
# the **worst** windows.
#
# Red-first: these assertions fail against the current higher-is-better entry
# and are marked ``xfail(strict=True)`` naming the implementation child
# (LP-0MWJOUZG9008Q4RG); the tree stays green for the dev-push gate, and once
# the direction flips each strict marker turns red so it cannot be left behind.
#
# ``TEN_METRICS`` above still records the *parent* definition, which declared
# this metric higher-is-better. The approved redefinition supersedes it, so the
# implementation child flips that entry to ``LOWER`` alongside the registry
# change (and ``test_registry_direction_matches_the_parent_definition`` follows
# it).

PENDING_RAW_PREFILL_ETA_ERROR = (
    "pending LP-0MWJOUZG9008Q4RG (report the raw median absolute relative "
    "error for prefill_eta_accuracy, registered lower-is-better)"
)

#: Median absolute relative errors of the matched prefill-ETA pairs: lower is
#: better, so the three best measurements are the three smallest errors. Their
#: windows are deliberately not the three most recent, so retention cannot pass
#: by keeping the newest three.
PREFILL_ERROR_MEASUREMENTS = [
    measurement(1041.0, "2026-11-10"),
    measurement(980.0, "2026-11-11"),
    measurement(1200.0, "2026-11-12"),
    measurement(1010.0, "2026-11-13"),
    measurement(1100.0, "2026-11-14"),
]
PREFILL_ERROR_BEST_THREE = [(980.0, "2026-11-11"), (1010.0, "2026-11-13"), (1041.0, "2026-11-10")]
PREFILL_ERROR_TARGET = (980.0 + 1010.0 + 1041.0) / 3


def prefill_direction() -> str:
    """The direction the registry itself declares for the prefill metric."""
    return registry_entry("prefill_eta_accuracy").direction


@pytest.mark.xfail(strict=True, reason=PENDING_RAW_PREFILL_ETA_ERROR)
def test_prefill_eta_accuracy_is_registered_lower_is_better():
    """The reported figure is an error, so a smaller value must be better."""
    assert prefill_direction() == LOWER


@pytest.mark.xfail(strict=True, reason=PENDING_RAW_PREFILL_ETA_ERROR)
def test_prefill_eta_best_three_keeps_the_smallest_errors():
    """A higher-is-better entry retains the three largest errors — the worst
    windows — which is exactly the inversion the redefinition removes."""
    retained = pm.retained(PREFILL_ERROR_MEASUREMENTS, prefill_direction())

    assert [(entry["value"], entry["date"]) for entry in retained] == PREFILL_ERROR_BEST_THREE


@pytest.mark.xfail(strict=True, reason=PENDING_RAW_PREFILL_ETA_ERROR)
def test_prefill_eta_target_is_the_mean_of_the_three_smallest_errors():
    assert pm.target_from(PREFILL_ERROR_MEASUREMENTS, prefill_direction()) == pytest.approx(
        PREFILL_ERROR_TARGET
    )


@pytest.mark.xfail(strict=True, reason=PENDING_RAW_PREFILL_ETA_ERROR)
def test_a_smaller_error_than_the_target_reads_met():
    """Under the redefinition the best window is the one whose error is
    smallest; judged higher-is-better a better-than-target error reads
    `Unmet`, which is the misleading verdict this pins against."""
    target = pm.target_from(PREFILL_ERROR_MEASUREMENTS, LOWER)

    assert (
        pm.derive_verdict(
            target - 100.0,
            PREFILL_ERROR_MEASUREMENTS,
            prefill_direction(),
            tolerance_pct=pm.tolerance_for("prefill_eta_accuracy"),
        )
        == "Met"
    )


@pytest.mark.xfail(strict=True, reason=PENDING_RAW_PREFILL_ETA_ERROR)
def test_a_larger_error_than_the_target_reads_unmet():
    target = pm.target_from(PREFILL_ERROR_MEASUREMENTS, LOWER)

    assert (
        pm.derive_verdict(
            target * 2.0,
            PREFILL_ERROR_MEASUREMENTS,
            prefill_direction(),
            tolerance_pct=pm.tolerance_for("prefill_eta_accuracy"),
        )
        == "Unmet"
    )


# --- Per-metric tolerance override -----------------------------------------


def test_tolerance_defaults_to_ten_percent_when_the_key_is_absent():
    defaults = {"metrics": {"decode_speed": {"measurements": []}}}

    assert pm.tolerance_for("decode_speed", defaults) == pytest.approx(
        pm.DEFAULT_TOLERANCE_PCT
    )


def test_tolerance_reads_the_per_metric_override():
    defaults = {"metrics": {"decode_speed": {"tolerance_pct": 25}}}

    assert pm.tolerance_for("decode_speed", defaults) == pytest.approx(25.0)


@pytest.mark.parametrize("value", ["wide", None, -5, -0.1, True, float("nan")])
def test_an_invalid_tolerance_falls_back_to_the_default(value):
    defaults = {"metrics": {"decode_speed": {"tolerance_pct": value}}}

    assert pm.tolerance_for("decode_speed", defaults) == pytest.approx(
        pm.DEFAULT_TOLERANCE_PCT
    )


def test_the_committed_defaults_carry_a_ten_percent_tolerance_per_metric():
    defaults = pm.load_defaults()

    for metric in pm.METRIC_REGISTRY:
        assert pm.tolerance_for(metric.id, defaults) == pytest.approx(10.0), metric.id


def test_an_explicit_tolerance_widens_the_band_the_verdict_uses():
    target = pm.target_from(LOWER_MEASUREMENTS, LOWER)
    actual = target * 1.15

    assert pm.derive_verdict(actual, LOWER_MEASUREMENTS, LOWER) == "Unmet"
    assert (
        pm.derive_verdict(actual, LOWER_MEASUREMENTS, LOWER, tolerance_pct=20) == "Met"
    )
    # A metric that demands exactness can override to a zero band, in which
    # case only a value at least as good as the target qualifies.
    assert (
        pm.derive_verdict(
            target, LOWER_MEASUREMENTS, LOWER, tolerance_pct=0
        )
        == "Met"
    )
    assert (
        pm.derive_verdict(
            target * 1.01, LOWER_MEASUREMENTS, LOWER, tolerance_pct=0
        )
        == "Unmet"
    )


@pytest.mark.parametrize("direction", [HIGHER, LOWER])
@pytest.mark.parametrize("count", [0, 1, 2])
def test_fewer_than_three_measurements_never_yields_a_verdict(direction, count):
    measurements = HIGHER_MEASUREMENTS[:count]

    # Even a perfect-looking actual must not produce a Met/Unmet verdict.
    assert pm.derive_verdict(0.0, measurements, direction) == "Insufficient history"
    assert pm.derive_verdict(1_000_000.0, measurements, direction) == "Insufficient history"


def test_a_missing_actual_renders_not_available():
    assert pm.derive_verdict(None, HIGHER_MEASUREMENTS, HIGHER) == "n/a"
    # A missing actual stays n/a whatever tolerance is applied.
    assert pm.derive_verdict(None, HIGHER_MEASUREMENTS, HIGHER, tolerance_pct=25) == "n/a"


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
    assert parsed["contention_denials_per_hour"] == pytest.approx(12.8)


# --- Scorecard parser: the attributable GPU row (LP-0MWG7KQVZ008DBZM) -------

_ATTRIBUTABLE_REPORT = """
## Local model utilization

| Metric | Total | Standard | Low |
|---|---|---|---|
| GPU computing — attributable (power ≥ 90 W and a busy local slot) | 6h 45m (28.1%) | - | - |
"""

_POWER_ONLY_REPORT = """
## Local model utilization

| Metric | Total | Standard | Low |
|---|---|---|---|
| GPU busy — power-based (hardware) | 6h 45m (28.1%) | - | - |
"""


def test_scorecard_parser_reads_the_attributable_row():
    """The scorer parses the attributable figure the report now renders."""
    assert pm._parse_gpu_busy(_ATTRIBUTABLE_REPORT) == pytest.approx(28.1)


def test_scorecard_parser_ignores_the_retired_power_only_row():
    """A pre-reset archived report carries the power-only figure; it must seed
    **nothing**, so history is never compared across two definitions."""
    assert pm._parse_gpu_busy(_POWER_ONLY_REPORT) is None
    assert "gpu_busy" not in pm.parse_archived_report(_POWER_ONLY_REPORT)


# --- History reset (definition change) --------------------------------------


def test_reset_metric_history_clears_measurements_and_records_the_reason():
    """A definition change resets the metric's history, documenting the reset."""
    state = pm.new_state()
    pm.record_measurement(state, "gpu_busy", window="2026-11-03", value=23.1)
    pm.record_measurement(state, "gpu_busy", window="2026-11-04", value=52.3)

    removed = pm.reset_metric_history(
        state,
        "gpu_busy",
        note="definition changed to the attributable gpu:computing_workload:bool share",
        reset_at="2026-11-15",
    )

    assert removed == 2
    assert state["metrics"]["gpu_busy"]["measurements"] == []
    assert pm.metric_notes(state, "gpu_busy") == [
        {
            "event": "definition_change",
            "note": "definition changed to the attributable gpu:computing_workload:bool share",
            "at": "2026-11-15",
        }
    ]
    # Baseline/target are derived only from the post-reset history.
    assert pm.baseline_from(state["metrics"]["gpu_busy"]["measurements"], HIGHER) is None


def test_reset_metric_history_round_trips_through_the_state_file(tmp_path):
    """The reset note survives save/load, so it is not a silent discontinuity."""
    path = tmp_path / "state.yaml"
    state = pm.new_state()
    pm.record_measurement(state, "gpu_busy", window="2026-11-03", value=23.1)
    pm.reset_metric_history(
        state,
        "gpu_busy",
        note="definition changed to the attributable share",
        reset_at="2026-11-15",
    )
    pm.save_state(state, path)

    loaded = pm.load_state(path)
    assert loaded["metrics"]["gpu_busy"]["measurements"] == []
    assert pm.metric_notes(loaded, "gpu_busy")[0]["event"] == "definition_change"


def test_reset_metric_history_rejects_an_unknown_metric():
    with pytest.raises(ValueError):
        pm.reset_metric_history(pm.new_state(), "nope", note="x")


# --- Time-to-first-token population (LP-0MWBLZ8HP005F6KO) -------------------
#
# The archive carries two first-token p90s over different populations. The
# ``## Time to first token`` ``| Total |`` row is the **combined local +
# remote** p90 (remote ``ttft_seconds=`` samples dominate its sample count and
# drag it to single-digit seconds), whereas the scorecard metric is the
# **local-only** p90 (``aggregation.ttft_p90_ms(..., include_remote=False)``).
# Only the ``## Prefill vs decode`` ``Prefill (dispatch → first byte)``
# ``p90 (s)`` column is local-only, so the backfill derives the seed from it
# and never falls back to the combined row (which seeded an unreachable
# 558-646 ms best-3 target for a metric whose actual is 30-60 s).

#: The ``## Prefill vs decode`` prefill row of the recent-era fixture.
_PREFILL_P90_ROW = (
    "| Prefill (dispatch → first byte) | 11h 07m | 60.4% | 0.6 | 3.9 | 33.5 | 198.0 | 3081 |"
)


def _report_with_prefill_p90(cell: str) -> str:
    assert _PREFILL_P90_ROW in ARCHIVED_REPORT_ERA_PREFILL
    rows = ARCHIVED_REPORT_ERA_PREFILL.replace(
        _PREFILL_P90_ROW, _PREFILL_P90_ROW.replace("| 33.5 |", f"| {cell} |")
    )
    assert cell in rows
    return rows


def test_backfill_parser_seeds_the_local_only_ttft_p90_not_the_combined_one():
    """The fixture renders both figures: a combined 9.0 s p90 and a local-only
    33.5 s p90. Only the local-only one is comparable with the metric."""
    parsed = pm.parse_archived_report(ARCHIVED_REPORT_ERA_PREFILL)

    assert parsed["time_to_first_token"] == pytest.approx(33_500.0)
    assert parsed["time_to_first_token"] != pytest.approx(9_000.0)


def test_backfill_parser_never_seeds_the_combined_local_plus_remote_ttft():
    """Era B renders only the combined ``## Time to first token`` p90
    (``670ms``): seeding it as the metric's value would install a target the
    local-only actual can never meet."""
    parsed = pm.parse_archived_report(ARCHIVED_REPORT_ERA_B)

    assert "time_to_first_token" not in parsed
    assert parsed["decode_speed"] == pytest.approx(37.8), "other metrics still parse"


def test_backfill_parser_ignores_the_seconds_era_combined_ttft_row():
    """The middle-era report renders its combined p90 as ``43.7s``. The unit
    suffix no longer rescues the value: it is the wrong population."""
    parsed = pm.parse_archived_report(ARCHIVED_REPORT_ERA_SECONDS)

    assert "time_to_first_token" not in parsed
    assert parsed["decode_speed"] == pytest.approx(29.4), "other metrics still parse"


@pytest.mark.parametrize(
    ("cell", "expected_ms"),
    [
        ("33.5", 33_500.0),  # the column is declared in seconds
        ("33.5s", 33_500.0),
        ("12.5 sec", 12_500.0),
        ("33500ms", 33_500.0),
        ("33500 ms", 33_500.0),
    ],
)
def test_backfill_parser_converts_the_prefill_p90_to_milliseconds(cell, expected_ms):
    """The local-only column is in seconds while the registry unit is ms, so a
    unit-blind read would seed 33.5 ms — a sub-second value that would win the
    lower-is-better best-3 forever."""
    parsed = pm.parse_archived_report(_report_with_prefill_p90(cell))

    assert parsed["time_to_first_token"] == pytest.approx(expected_ms)


@pytest.mark.parametrize("cell", ["33.5x", "33.5 s/req", "~33.5", "fast"])
def test_backfill_parser_refuses_an_unrecognised_prefill_unit(cell):
    """An unrecognised suffix is skipped, never guessed: a wrong unit must not
    seed history at all rather than seed a mis-scaled value."""
    parsed = pm.parse_archived_report(_report_with_prefill_p90(cell))

    assert "time_to_first_token" not in parsed
    assert parsed["decode_speed"] == pytest.approx(33.9), "other metrics still parse"


def test_backfill_parser_omits_ttft_when_the_prefill_row_has_no_samples():
    """A window with no local prefill samples renders ``-`` in every phase
    cell. That is not a value, so the metric is skipped rather than seeded as
    0 — and the combined row is still not consulted."""
    report = ARCHIVED_REPORT_ERA_PREFILL.replace(
        _PREFILL_P90_ROW,
        "| Prefill (dispatch → first byte) | - | - | - | - | - | - | 0 |",
    )

    parsed = pm.parse_archived_report(report)

    assert "time_to_first_token" not in parsed
    assert parsed["decode_speed"] == pytest.approx(33.9)


def test_backfill_parser_returns_no_measurement_for_metrics_it_cannot_parse():
    parsed = pm.parse_archived_report(ARCHIVED_REPORT_ERA_A)

    # Era A renders neither a time-to-first-token nor a local-only prefill
    # section, so those metrics are skipped rather than guessed.
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


def test_backfill_skips_ttft_when_only_the_combined_row_is_rendered():
    """A report with no local-only figure contributes nothing for the metric,
    so it is reported as skipped rather than seeded from the combined row."""
    state = pm.new_state()

    result = pm.seed_history(state, [("2026-10-31", ARCHIVED_REPORT_ERA_B)])

    assert "time_to_first_token" in result["skipped_metrics"]
    assert _measurement_count(state, "time_to_first_token") == 0


def test_backfill_seeds_a_ttft_in_the_scorecard_actual_range():
    """The seeded value is comparable with the metric's own windows (the
    live 2026-11-08/09 actuals are 33.5 s and 34.0 s), not with the combined
    p90 (single-digit seconds) of the same report."""
    state = pm.new_state()

    pm.seed_history(state, [("2026-11-09", ARCHIVED_REPORT_ERA_PREFILL)])

    stored = state["metrics"]["time_to_first_token"]["measurements"][0]
    assert stored["window"] == "2026-11-09"
    assert 25_000.0 <= stored["value"] <= 60_000.0


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
