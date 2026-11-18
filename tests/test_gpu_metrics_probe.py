#!/usr/bin/env python3
"""Tests for ``scripts/gpu_metrics_probe.py`` (LP-0MVZEW6UL0054D8W).

The probe is the evidence tool for the parent investigation
(LP-0MVSQT7YC0045HFW): one command that samples every candidate
``gpu_metrics`` activity field next to amdgpu hwmon ``power1_average`` and
``gpu_busy_percent``, and (with ``--verdict``) prints the discrimination
statistics that settle whether any field tracks GPU activity.

Nothing here touches the live host. Every capture runs against a synthetic
sysfs tree built in ``tmp_path`` (a fake ``class/drm/card0/device`` with a real
264-byte ``gpu_metrics`` record and a fake hwmon directory), and the clock is a
fake, so the schedule is asserted exactly without sleeping.

Coverage:

* CLI defaults are documented in ``--help`` and the module docstring (AC1),
* rows carry a timestamp, every candidate activity field, power and
  ``gpu_busy_percent``; row count and monotonic timestamps match the documented
  ``floor(duration/interval)+1`` grid (AC1, AC2, AC5),
* CSV is the default, ``--json`` emits the document, ``--verdict`` prints the
  per-field statistics, per-band counts and pass/fail decisions, and a
  one-sided capture is visibly ``insufficient_data`` (AC3),
* sysfs root and hwmon directory are overridable by flag and env (AC4),
* a missing, unreadable, truncated or unknown-revision ``gpu_metrics`` exits
  non-zero with nothing on stdout and never yields a partial row (AC6),
* every output path states that the values are SMU time-filtered with an
  undocumented window (AC7),
* graded, saturated and one-sided series produce the exact pass/fail/
  insufficient decisions through this module's verdict wiring (AC3, AC5).
"""

from __future__ import annotations

import csv
import io
import json
import struct
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gpu_metrics_probe as probe  # noqa: E402
from lib.gpu_metrics_decoder import FIELD_TABLE, decode  # noqa: E402

FIXTURE_METRICS = REPO_ROOT / "tests" / "fixtures" / "gpu_metrics_v3_0.bin"
FIXTURE_META = json.loads((REPO_ROOT / "tests" / "fixtures" / "gpu_metrics_v3_0.meta.json").read_text())
FIXTURE_POWER_UW = FIXTURE_META["hwmon"]["power1_average_uw"]
FIXTURE_GFX_POWER_UW = 95_060_000

GFX = "average_gfx_activity"
VCN = "average_vcn_activity"
C0 = "average_core_c0_activity"


# ---------------------------------------------------------------------------
# Synthetic sysfs trees and a fake clock
# ---------------------------------------------------------------------------


def _set_u16(blob: bytearray, field: str, value: int, index: int | None = None) -> None:
    spec = next(spec for spec in FIELD_TABLE if spec.name == field)
    offset = spec.offset if index is None else spec.offset + index * spec.element_size
    struct.pack_into("<H", blob, offset, value)


def metrics_blob(
    *,
    gfx_activity: int = 100,
    vcn_activity: int = 0,
    ipu_activity: int = 0,
    c0_activity: int = 0,
) -> bytes:
    """A valid v3.0 record with chosen activity values, built from the fixture."""
    blob = bytearray(FIXTURE_METRICS.read_bytes())
    _set_u16(blob, GFX, gfx_activity)
    _set_u16(blob, VCN, vcn_activity)
    for index in range(8):
        _set_u16(blob, "average_ipu_activity", ipu_activity, index)
    for index in range(16):
        _set_u16(blob, C0, c0_activity, index)
    return bytes(blob)


def build_tree(
    tmp_path: Path,
    *,
    metrics: bytes | None | bool = None,
    power_uw: int = FIXTURE_POWER_UW,
    busy: int | None = 100,
    hwmon_name: str = "hwmon5",
) -> Path:
    """Create a fake sysfs tree and return its root.

    ``metrics=False`` omits the ``gpu_metrics`` file entirely; ``metrics=None``
    (the default) writes the committed real capture.
    """
    root = tmp_path / "sys"
    device = root / probe.DEVICE_RELATIVE_PATH
    device.mkdir(parents=True)
    if metrics is not False:
        (device / probe.METRICS_FILENAME).write_bytes(FIXTURE_METRICS.read_bytes() if metrics is None else metrics)
    hwmon = device / "hwmon" / hwmon_name
    hwmon.mkdir(parents=True)
    (hwmon / probe.POWER_FILENAME).write_text(f"{power_uw}\n")
    if busy is not None:
        (device / probe.BUSY_FILENAME).write_text(f"{busy}\n")
    return root


class FakeClock:
    """A clock whose ``sleep`` advances both the wall clock and monotonic time."""

    def __init__(self, start: datetime | None = None) -> None:
        self.now_value = start or datetime(2026, 11, 18, 12, 0, 0, tzinfo=UTC)
        self.monotonic_value = 0.0
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self.now_value

    def monotonic(self) -> float:
        return self.monotonic_value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.monotonic_value += seconds
        self.now_value += timedelta(seconds=seconds)


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(probe, "_utc_now", fake.now)
    monkeypatch.setattr(probe, "_monotonic", fake.monotonic)
    monkeypatch.setattr(probe, "_sleep", fake.sleep)
    return fake


def make_probe(root: Path, *, hwmon_dir: Path | None = None) -> probe.GpuMetricsProbe:
    return probe.GpuMetricsProbe(probe.resolve_paths(root, hwmon_dir))


def _csv_output(stdout: str) -> tuple[list[str], list[list[str]]]:
    lines = stdout.splitlines()
    assert lines and lines[0].startswith("#"), "CSV must lead with the time-filter note"
    parsed = list(csv.reader(lines[1:]))
    return parsed[0], parsed[1:]


# ---------------------------------------------------------------------------
# AC1: defaults documented in --help and the module docstring
# ---------------------------------------------------------------------------


def test_parser_defaults_are_documented():
    parser = probe.build_parser()
    args = parser.parse_args([])
    assert args.duration == 1800.0
    assert args.interval == 1.0

    help_text = parser.format_help()
    assert "--duration" in help_text and "1800" in help_text and "30 min" in help_text
    assert "--interval" in help_text and "1.0" in help_text

    doc = probe.__doc__ or ""
    assert "--duration 1800" in doc
    assert "--interval 1.0" in doc
    assert "floor(duration / interval) + 1" in doc


def test_duration_and_interval_must_be_positive():
    with pytest.raises(SystemExit):
        probe.build_parser().parse_args(["--duration", "0"])
    with pytest.raises(SystemExit):
        probe.build_parser().parse_args(["--interval", "-1"])


# ---------------------------------------------------------------------------
# AC2: every candidate activity field plus both reference signals
# ---------------------------------------------------------------------------


def test_activity_columns_are_every_percentage_field_in_the_layout():
    percentage_fields = {spec.name for spec in FIELD_TABLE if spec.unit == probe.ACTIVITY_UNIT}
    assert {column.field for column in probe.ACTIVITY_COLUMNS} == percentage_fields
    # 1 gfx + 1 vcn + 8 IPU columns + 16 core C0 columns.
    assert len(probe.ACTIVITY_COLUMNS) == 26
    names = [column.column for column in probe.ACTIVITY_COLUMNS]
    assert names[0] == GFX
    assert names[1] == VCN
    assert names[-1] == f"{C0}[15]"
    assert probe.COLUMNS[:2] == ("timestamp", "elapsed_seconds")
    assert probe.COLUMNS[-2:] == ("power1_average_watts", "gpu_busy_percent")


def test_row_carries_every_activity_field_and_both_references(tmp_path, clock):
    blob = metrics_blob(gfx_activity=100)
    root = build_tree(tmp_path, metrics=blob)
    rows = make_probe(root).run(duration=0.0 + 1, interval=1)

    assert len(rows) == 2
    row = rows[0]
    assert set(row) == set(probe.COLUMNS)
    decoded = decode(blob)
    for column in probe.ACTIVITY_COLUMNS:
        expected = column.value(decoded)
        assert row[column.column] == expected, column.column
    assert row["power1_average_watts"] == pytest.approx(42.011)
    assert row["gpu_busy_percent"] == 100


def test_activity_values_are_fixture_derived_not_live(tmp_path, clock):
    root = build_tree(
        tmp_path,
        metrics=metrics_blob(gfx_activity=37, vcn_activity=12, ipu_activity=55, c0_activity=88),
    )
    row = make_probe(root).run(duration=1, interval=1)[0]
    assert row[GFX] == 37
    assert row[VCN] == 12
    assert row["average_ipu_activity[3]"] == 55
    assert row["average_core_c0_activity[15]"] == 88


def test_power_is_converted_from_microwatts_to_watts(tmp_path, clock):
    root = build_tree(tmp_path, power_uw=FIXTURE_GFX_POWER_UW)
    row = make_probe(root).run(duration=1, interval=1)[0]
    assert row["power1_average_watts"] == pytest.approx(95.06)


def test_missing_busy_percent_keeps_the_column_and_reads_null(tmp_path, clock):
    root = build_tree(tmp_path, busy=None)
    row = make_probe(root).run(duration=1, interval=1)[0]
    assert "gpu_busy_percent" in row
    assert row["gpu_busy_percent"] is None


# ---------------------------------------------------------------------------
# AC5: row count and monotonic timestamps, without sleeping
# ---------------------------------------------------------------------------


def test_row_count_follows_the_documented_grid(tmp_path, clock):
    root = build_tree(tmp_path)
    rows = make_probe(root).run(duration=2, interval=1)
    assert [row["elapsed_seconds"] for row in rows] == [0.0, 1.0, 2.0]
    assert clock.sleeps == [1.0, 1.0]

    clock.now_value = FakeClock().now()
    clock.monotonic_value = 0.0
    clock.sleeps.clear()
    fractional = make_probe(root).run(duration=2.5, interval=1)
    # floor(2.5 / 1) + 1 == 3
    assert len(fractional) == 3


def test_timestamps_are_strictly_monotonic(tmp_path, clock):
    root = build_tree(tmp_path)
    rows = make_probe(root).run(duration=3, interval=1)
    stamps = [datetime.fromisoformat(str(row["timestamp"])) for row in rows]
    assert len(stamps) == 4
    assert all(later > earlier for earlier, later in zip(stamps, stamps[1:]))


# ---------------------------------------------------------------------------
# AC3 + AC4: CLI output (CSV default, --json, --verdict) and overrides
# ---------------------------------------------------------------------------


def test_csv_is_the_default_with_a_time_filter_note(tmp_path, clock, capsys):
    root = build_tree(tmp_path, metrics=metrics_blob(gfx_activity=100))
    assert probe.main(["--sysfs-root", str(root), "--duration", "1"]) == 0

    captured = capsys.readouterr()
    header, rows = _csv_output(captured.out)
    assert header == list(probe.COLUMNS)
    assert len(rows) == 2
    assert "SMU time-filtered" in captured.out.splitlines()[0]
    assert "undocumented window" in captured.out.splitlines()[0]

    row = dict(zip(header, rows[0]))
    assert row[GFX] == "100"
    assert float(row["power1_average_watts"]) == pytest.approx(42.011)
    assert row["gpu_busy_percent"] == "100"


def test_json_document_carries_meta_samples_and_the_note(tmp_path, clock, capsys):
    root = build_tree(tmp_path)
    assert probe.main(["--sysfs-root", str(root), "--duration", "1", "--interval", "1", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema"] == "gpu_metrics_probe.v1"
    meta = payload["meta"]
    assert "SMU time-filtered" in meta["note"]
    assert "undocumented window" in meta["note"]
    assert meta["capture"]["sample_count"] == 2
    assert meta["capture"]["interval_seconds"] == 1.0
    assert meta["paths"]["metrics"].endswith("class/drm/card0/device/gpu_metrics")
    assert meta["activity_fields"][0] == GFX
    assert len(payload["samples"]) == 2
    assert payload["samples"][0]["power1_average_watts"] == pytest.approx(42.011)
    assert "verdict" not in payload


def test_json_verdict_reports_stats_band_counts_and_decisions(tmp_path, clock, capsys):
    root = build_tree(tmp_path)
    assert (
        probe.main(
            [
                "--sysfs-root",
                str(root),
                "--duration",
                "1",
                "--json",
                "--verdict",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    verdict = payload["verdict"]

    # A constant-power fixture capture is one-sided: no compute band, so the
    # capture cannot discriminate and says so instead of emitting a fake verdict.
    assert verdict["band_counts"] == {
        "idle": 2,
        "middle": 0,
        "compute": 0,
        "invalid": 0,
        "total": 2,
    }
    assert verdict["status"] == "insufficient_data"
    assert verdict["passed_fields"] == []
    assert set(verdict["fields"]) == set(probe.VERDICT_FIELDS)
    gfx_stats = verdict["fields"][GFX]
    assert gfx_stats["idle_band_mean"] == pytest.approx(100.0)
    assert gfx_stats["compute_band_mean"] is None
    assert gfx_stats["separation_pp"] is None
    assert gfx_stats["decision"] == "insufficient_data"
    assert gfx_stats["idle_band_count"] == 2
    assert gfx_stats["compute_band_count"] == 0
    # The rejected counter is evaluated as the baseline next to the candidates.
    assert verdict["fields"]["gpu_busy_percent"]["decision"] == "insufficient_data"


def test_verdict_in_csv_mode_goes_to_stderr_and_keeps_stdout_pipeable(tmp_path, clock, capsys):
    root = build_tree(tmp_path)
    assert probe.main(["--sysfs-root", str(root), "--duration", "1", "--verdict"]) == 0
    captured = capsys.readouterr()

    header, rows = _csv_output(captured.out)  # stdout is still a clean CSV stream
    assert header == list(probe.COLUMNS)
    assert len(rows) == 2

    text = captured.err
    assert "gpu_metrics discrimination verdict" in text
    assert "bar: separation >= 30 pp" in text
    assert "bands: idle=2 middle=0 compute=0 invalid=0 total=2" in text
    assert "status: insufficient_data" in text
    assert "note: gpu_metrics values are SMU time-filtered with an undocumented window" in text
    assert "passed: none" in text
    gfx_line = next(line for line in text.splitlines() if line.startswith(f"{GFX} "))
    assert gfx_line.endswith("insufficient_data")
    assert any(line.startswith("gpu_busy_percent ") for line in text.splitlines())


def test_sysfs_root_and_hwmon_dir_env_overrides(tmp_path, clock, monkeypatch, capsys):
    root = build_tree(tmp_path)
    other_hwmon = root / probe.DEVICE_RELATIVE_PATH / "hwmon" / "hwmon9"
    other_hwmon.mkdir()
    (other_hwmon / probe.POWER_FILENAME).write_text(f"{FIXTURE_GFX_POWER_UW}\n")

    monkeypatch.setenv(probe.SYSFS_ROOT_ENV, str(root))
    monkeypatch.setenv(probe.HWMON_DIR_ENV, str(other_hwmon))
    assert probe.main(["--duration", "1", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["meta"]["paths"]["hwmon_dir"] == str(other_hwmon)
    assert payload["samples"][0]["power1_average_watts"] == pytest.approx(95.06)


def test_hwmon_dir_flag_overrides_auto_discovery(tmp_path, clock):
    root = build_tree(tmp_path)
    device = root / probe.DEVICE_RELATIVE_PATH
    other_hwmon = device / "hwmon" / "hwmon9"
    other_hwmon.mkdir()
    (other_hwmon / probe.POWER_FILENAME).write_text(f"{FIXTURE_GFX_POWER_UW}\n")

    discovered = make_probe(root).run(duration=1, interval=1)[0]
    assert discovered["power1_average_watts"] == pytest.approx(42.011)

    overridden = make_probe(root, hwmon_dir=other_hwmon).run(duration=1, interval=1)[0]
    assert overridden["power1_average_watts"] == pytest.approx(95.06)


# ---------------------------------------------------------------------------
# AC6: missing / unreadable / malformed records fail closed, never partially
# ---------------------------------------------------------------------------


def _assert_fails_closed(capsys, root: Path, *args: str, expected: str) -> None:
    code = probe.main(["--sysfs-root", str(root), "--duration", "1", *args])
    assert code == probe.EXIT_PROBE_ERROR
    captured = capsys.readouterr()
    assert captured.out == ""  # no header, no comment line, no partial row
    assert expected in captured.err
    assert "gpu_metrics" in captured.err


def test_missing_gpu_metrics_exits_nonzero_without_output(tmp_path, clock, capsys):
    root = build_tree(tmp_path, metrics=False)
    _assert_fails_closed(capsys, root, expected="cannot read gpu_metrics")


def test_unreadable_gpu_metrics_exits_nonzero_without_output(tmp_path, clock, capsys):
    root = build_tree(tmp_path)
    metrics_path = root / probe.DEVICE_RELATIVE_PATH / probe.METRICS_FILENAME
    metrics_path.chmod(0o000)
    try:
        _assert_fails_closed(capsys, root, expected="cannot read gpu_metrics")
    finally:
        metrics_path.chmod(0o644)


def test_truncated_blob_fails_closed_with_observed_and_expected_sizes(tmp_path, clock, capsys):
    root = build_tree(tmp_path, metrics=FIXTURE_METRICS.read_bytes()[:100])
    code = probe.main(["--sysfs-root", str(root), "--duration", "1"])
    captured = capsys.readouterr()
    assert code == probe.EXIT_PROBE_ERROR
    assert captured.out == ""
    assert "cannot decode gpu_metrics" in captured.err
    assert "100" in captured.err and "264" in captured.err


def test_unknown_revision_fails_closed(tmp_path, clock, capsys):
    blob = bytearray(FIXTURE_METRICS.read_bytes())
    blob[2] = 4  # format_revision
    root = build_tree(tmp_path, metrics=bytes(blob))
    _assert_fails_closed(capsys, root, expected="format_revision")


def test_no_partial_row_when_a_read_fails_mid_capture(tmp_path, clock):
    root = build_tree(tmp_path)
    metrics_path = root / probe.DEVICE_RELATIVE_PATH / probe.METRICS_FILENAME
    emitted: list[dict[str, object]] = []

    def sleep_then_remove(seconds: float) -> None:
        clock.sleep(seconds)
        metrics_path.unlink()

    probe_runner = probe.GpuMetricsProbe(
        probe.resolve_paths(root),
        clock=clock.now,
        monotonic=clock.monotonic,
        sleep=sleep_then_remove,
    )
    with pytest.raises(probe.ProbeError):
        probe_runner.run(duration=5, interval=1, on_row=emitted.append)

    # The first sample was complete before the file vanished; the failed second
    # sample produced nothing at all.
    assert len(emitted) == 1
    assert set(emitted[0]) == set(probe.COLUMNS)
    assert all(emitted[0][column] is not None for column in probe.COLUMNS)


def test_missing_hwmon_power_is_an_actionable_error(tmp_path, clock):
    root = tmp_path / "sys"
    device = root / probe.DEVICE_RELATIVE_PATH
    device.mkdir(parents=True)
    (device / probe.METRICS_FILENAME).write_bytes(FIXTURE_METRICS.read_bytes())

    with pytest.raises(probe.ProbeError) as excinfo:
        probe.resolve_paths(root)
    message = str(excinfo.value)
    assert "power1_average" in message
    assert "--hwmon-dir" in message


# ---------------------------------------------------------------------------
# AC3: verdict wiring passes graded series and fails saturated ones
# ---------------------------------------------------------------------------


def _activity_rows(pairs: list[tuple[float, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (power_watts, gfx_activity) in enumerate(pairs):
        row: dict[str, object] = {column: 0 for column in probe.COLUMNS}
        row[probe.TIMESTAMP_COLUMN] = f"2026-11-18T12:00:{index:02d}.000+00:00"
        row["power1_average_watts"] = power_watts
        row[GFX] = gfx_activity
        row["gpu_busy_percent"] = 100
        rows.append(row)
    return rows


def test_graded_series_passes_and_saturated_series_fails():
    graded = _activity_rows([(40.0, 5.0), (50.0, 6.0), (55.0, 4.0), (95.0, 95.0), (100.0, 97.0), (110.0, 94.0)])
    verdict = probe.build_verdict(graded)
    assert verdict.status == "ok"
    assert verdict.passed_fields == (GFX,)
    stats = verdict.fields[GFX].to_dict()
    assert stats["separation_pp"] > 30.0
    assert stats["pearson_r"] >= 0.6
    assert stats["idle_false_busy_share"] == 0.0
    assert stats["decision"] == "pass"

    saturated = _activity_rows([(40.0, 100.0), (55.0, 100.0), (95.0, 100.0), (120.0, 100.0)])
    verdict = probe.build_verdict(saturated)
    assert verdict.passed_fields == ()
    stats = verdict.fields[GFX].to_dict()
    assert stats["decision"] == "fail"
    assert any("separation" in reason for reason in stats["reasons"])
    assert any("pearson r undefined" in reason for reason in stats["reasons"])
    assert any("idle false-busy 100.0%" in reason for reason in stats["reasons"])
    # The rejected counter is judged by the same bar, next to the candidates.
    assert verdict.fields["gpu_busy_percent"].decision == "fail"


def test_one_sided_series_is_insufficient_and_render_shows_band_counts():
    idle_only = _activity_rows([(40.0, 5.0), (50.0, 6.0), (55.0, 4.0)])
    verdict = probe.build_verdict(idle_only)
    assert verdict.status == "insufficient_data"
    stats = verdict.fields[GFX]
    assert stats.decision == "insufficient_data"
    assert stats.idle_band_count == 3
    assert stats.compute_band_count == 0
    assert stats.separation_pp is None

    text = probe.render_verdict(verdict, sample_count=3, interval=1.0, duration=2.0)
    assert "bands: idle=3 middle=0 compute=0 invalid=0 total=3" in text
    assert "status: insufficient_data" in text
    assert "no compute-band samples" in text
    assert "note: gpu_metrics values are SMU time-filtered with an undocumented window" in text


def test_render_verdict_marks_a_passing_field():
    graded = _activity_rows([(40.0, 5.0), (50.0, 6.0), (55.0, 4.0), (95.0, 95.0), (100.0, 97.0), (110.0, 94.0)])
    verdict = probe.build_verdict(graded)
    text = probe.render_verdict(verdict, sample_count=6, interval=1.0, duration=5.0)
    gfx_line = next(line for line in text.splitlines() if line.startswith(f"{GFX} "))
    assert gfx_line.endswith(" pass")
    assert "passed: average_gfx_activity" in text
    assert "status: ok (both power bands present)" in text


def test_build_verdict_samples_map_power_and_skip_unreadable_busy():
    rows = _activity_rows([(40.0, 5.0), (95.0, 95.0)])
    rows[0]["gpu_busy_percent"] = None
    samples = probe.build_verdict_samples(rows)
    assert samples[0]["power_watts"] == 40.0
    assert "gpu_busy_percent" not in samples[0]
    assert samples[1]["gpu_busy_percent"] == 100

    verdict = probe.build_verdict(rows)
    assert verdict.fields["gpu_busy_percent"].idle_band_count == 0
    assert verdict.fields[GFX].idle_band_count == 1
