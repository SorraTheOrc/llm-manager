"""Defaults-file and optional-PyYAML contract for the metrics engine.

Companion to ``test_performance_metrics.py`` (the engine and backfill behaviour
specification). This module pins the data-layer promises behind the report's
metric details and the state store (LP-0MW1BU9F8007SE0U):

- the committed defaults carry complete per-metric detail copy, and each
  ``calculation`` names the metric's registered data source, so the renderer
  can stay prose-free;
- the committed defaults and the live state file survive without PyYAML (the
  guarded-import fallback in ``config_loader``), and a fallback-written state
  file is still ordinary YAML for later readers;
- the live state file wins only for the metrics it actually defines, so a
  partially written file never discards the defaults' seeded history;
- a measurement whose configuration era is unknown (a backfill) does not
  manufacture a false cross-era warning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import config_loader  # noqa: E402
import performance_metrics as pm  # noqa: E402

from tests import fixtures  # noqa: E402


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _measurements(state: dict, metric_id: str) -> list[dict]:
    return (state.get("metrics") or {}).get(metric_id, {}).get("measurements", [])


# ---------------------------------------------------------------------------
# Committed defaults: detail copy
# ---------------------------------------------------------------------------


def test_committed_defaults_carry_complete_detail_copy_for_every_metric():
    defaults = pm.load_defaults()

    for metric in pm.METRIC_REGISTRY:
        copy = pm.detail_copy(metric.id, defaults)
        assert set(pm.DETAIL_FIELDS) <= set(copy), f"{metric.id}: missing detail fields"
        for field in pm.DETAIL_FIELDS:
            assert copy[field].strip(), f"{metric.id}: empty {field}"


def test_each_calculation_copy_names_the_registered_data_source():
    defaults = pm.load_defaults()

    for metric in pm.METRIC_REGISTRY:
        calculation = pm.detail_copy(metric.id, defaults)["calculation"]
        assert _norm(metric.source) in _norm(calculation), (
            f"{metric.id}: calculation does not name its source {metric.source!r}"
        )


def test_gpu_busy_detail_copy_describes_coverage_and_hardware_scope():
    """AC4/AC5 (LP-0MW5K8YE9001QO39): the why/calculation copy states the
    coverage-honest grid and that the figure counts any GPU consumer."""
    copy = pm.detail_copy("gpu_busy", pm.load_defaults())
    text = _norm(f"{copy['why']} {copy['calculation']}")

    assert "any gpu consumer" in text
    assert "15-second instant" in text
    assert "unknown" in text


def test_each_detail_copy_anchors_an_existing_report_section_heading():
    defaults = pm.load_defaults()

    for metric in pm.METRIC_REGISTRY:
        anchor = pm.detail_copy(metric.id, defaults)["data_section_anchor"]
        assert anchor.startswith("## "), f"{metric.id}: anchor {anchor!r} is not a heading"
        assert len(anchor) > 3


# ---------------------------------------------------------------------------
# Optional PyYAML
# ---------------------------------------------------------------------------


def test_committed_defaults_parse_identically_without_pyyaml(monkeypatch):
    text = pm.DEFAULTS_PATH.read_text(encoding="utf-8")
    real_import_yaml = config_loader.import_yaml
    with_yaml = config_loader.parse_yaml_mapping(text)

    monkeypatch.setattr(config_loader, "import_yaml", lambda: None)
    without_yaml = config_loader.parse_yaml_mapping(text)
    monkeypatch.setattr(config_loader, "import_yaml", real_import_yaml)

    assert without_yaml == with_yaml
    assert set(without_yaml["metrics"]) == {metric.id for metric in pm.METRIC_REGISTRY}


def test_state_round_trips_without_pyyaml_and_stays_valid_yaml(monkeypatch, tmp_path):
    fingerprint = pm.configuration_fingerprint(2, 131072)
    state = pm.new_state()
    pm.record_measurement(
        state,
        "decode_speed",
        window="2026-11-01",
        value=41.2,
        date="2026-11-01",
        fingerprint=fingerprint,
    )
    pm.record_measurement(
        state,
        "time_to_first_token",
        window="2026-11-01",
        value=410.0,
        date="2026-11-01",
        fingerprint=fingerprint,
    )
    path = tmp_path / "state.yaml"
    real_import_yaml = config_loader.import_yaml

    monkeypatch.setattr(config_loader, "import_yaml", lambda: None)
    pm.save_state(state, path)
    assert pm.load_state(path) == state

    monkeypatch.setattr(config_loader, "import_yaml", real_import_yaml)
    assert pm.load_state(path) == state, "the fallback output must stay valid YAML"


# ---------------------------------------------------------------------------
# Load precedence and configuration era
# ---------------------------------------------------------------------------


_DEFAULTS_WITH_HISTORY = """\
metrics:
  decode_speed:
    measurements:
      - window: "2026-06-01"
        value: 31.5
        date: "2026-06-01"
        fingerprint: "slots=2,ctx=131072"
  fallback_rate:
    measurements:
      - window: "2026-06-01"
        value: 60.0
        date: "2026-06-01"
        fingerprint: "slots=2,ctx=131072"
"""

_PARTIAL_LIVE_STATE = """\
metrics:
  fallback_rate:
    measurements:
      - window: "2026-11-01"
        value: 12.0
        date: "2026-11-01"
        fingerprint: "slots=2,ctx=131072"
"""


def test_live_state_overrides_defaults_only_where_it_defines_the_metric(tmp_path):
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text(_DEFAULTS_WITH_HISTORY, encoding="utf-8")
    live = tmp_path / "state.yaml"
    live.write_text(_PARTIAL_LIVE_STATE, encoding="utf-8")

    state = pm.load_state(live, defaults_path=defaults)

    assert [m["value"] for m in _measurements(state, "decode_speed")] == [31.5]
    assert [m["value"] for m in _measurements(state, "fallback_rate")] == [12.0]


def test_backfilled_measurements_do_not_manufacture_a_cross_era_warning():
    state = pm.new_state()
    pm.seed_history(state, [("2026-10-31", fixtures.ARCHIVED_REPORT_ERA_B)])
    pm.record_measurement(
        state,
        "decode_speed",
        window="2026-11-01",
        value=39.0,
        date="2026-11-01",
        fingerprint=pm.configuration_fingerprint(2, 131072),
    )

    measurements = _measurements(state, "decode_speed")

    assert any(m["fingerprint"] == "" for m in measurements), "backfill has no known era"
    assert pm.is_cross_era(measurements) is False, "an unknown era is not a second era"


@pytest.mark.parametrize("direction", [pm.HIGHER, pm.LOWER])
def test_retention_ignores_repeated_windows_and_derives_from_one_per_window(direction):
    repeated = [
        {"window": "2026-10-01", "value": 10.0, "date": "2026-10-01", "fingerprint": ""},
        {"window": "2026-10-01", "value": 20.0, "date": "2026-10-01", "fingerprint": ""},
        {"window": "2026-10-02", "value": 30.0, "date": "2026-10-02", "fingerprint": ""},
        {"window": "2026-10-03", "value": 40.0, "date": "2026-10-03", "fingerprint": ""},
    ]

    kept = pm.retained(repeated, direction)

    assert len(kept) == 3
    assert len({m["window"] for m in kept}) == 3
