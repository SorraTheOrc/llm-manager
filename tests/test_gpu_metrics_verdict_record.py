#!/usr/bin/env python3
"""Guard the recorded ``gpu_metrics`` verdict against the documentation.

LP-0MVZEW7R1004GLJN (parent LP-0MVSQT7YC0045HFW).

The probe, the decoder and the guard shipped in sibling items; this module makes
the *verdict* citable. ``tests/fixtures/gpu_metrics_verdict_capture_2026-11-18.
csv`` is the raw capture the documents quote, and its sidecar ``.meta.json``
records the probe command, the capture window, the kernel and the file hash.

The tests recompute the verdict from that capture with the shipped engine, then
assert the three documents quote *those* numbers:

* ``docs/systemd/gpu-metrics.service`` — the verdict, the pinned v3.0 decode
  layout and the re-measure recipe (AC1, AC4, AC5),
* ``monitoring/README.md`` — the ``gpu_metrics`` row in *Why not ``amd-smi``,
  ``rocm-smi`` or ``amdtop``?* with the band statistics and the recipe pointer
  (AC2),
* ``monitoring/gpu_compute_rules.yaml`` — the header comment (AC3).

The bar quoted by the documents is compared with the constants in
``scripts/lib/gpu_metrics_verdict.py``, so the prose cannot drift from the code;
the probe and decoder docstrings must point at the recorded verdict (AC6).

A re-measurement after a kernel, firmware or driver change updates the capture,
its sidecar and the documents together: this module is what fails when only one
of them moves.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gpu_metrics_probe as probe  # noqa: E402
from lib import gpu_metrics_decoder as decoder  # noqa: E402
from lib import gpu_metrics_verdict as verdict_engine  # noqa: E402

FIXTURES = REPO_ROOT / "tests" / "fixtures"
CAPTURE_CSV = FIXTURES / "gpu_metrics_verdict_capture_2026-11-18.csv"
CAPTURE_META = FIXTURES / "gpu_metrics_verdict_capture_2026-11-18.meta.json"
GPU_METRICS_DOC = REPO_ROOT / "docs" / "systemd" / "gpu-metrics.service"
MONITORING_README = REPO_ROOT / "monitoring" / "README.md"
RULES_YAML = REPO_ROOT / "monitoring" / "gpu_compute_rules.yaml"

#: Heading of the decision record inside ``docs/systemd/gpu-metrics.service``.
VERDICT_HEADING = "no `gpu_metrics` activity series is published"

#: Heading of the re-measure recipe inside the same document.
REMEASURE_HEADING = "Re-measuring the `gpu_metrics` verdict"

#: The candidate field the verdict is about, and the rejected baseline it is
#: compared with.
GFX_FIELD = "average_gfx_activity"
BUSY_FIELD = "gpu_busy_percent"


def _section(text: str, heading: str, *, level: int = 2) -> str:
    """Return the body of the section whose heading contains *heading*.

    Deeper headings are included, so a ``##`` section keeps its ``###``
    subsections (the verdict record and its re-measure recipe).
    """
    marker = "#" * level + " "
    collected: list[str] = []
    capturing = False
    for line in text.splitlines():
        if line.startswith(marker):
            if capturing:
                break
            capturing = heading.lower() in line.lower()
        elif capturing:
            collected.append(line)
    assert collected, f"no '{marker}{heading}' section found"
    return "\n".join(collected)


def _prose(text: str) -> str:
    """Collapse markdown line wrapping so a phrase can be asserted byte-safely."""
    return " ".join(text.split())


def _yaml_header(path: Path) -> str:
    """Return the leading comment block of a YAML file."""
    lines: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            lines.append(line)
        elif line.strip():
            break
    assert lines, f"{path} starts with no comment block"
    return "\n".join(lines)


@lru_cache(maxsize=1)
def _capture() -> dict:
    """Parse the committed capture and recompute its verdict."""
    lines = CAPTURE_CSV.read_text().splitlines()
    assert lines and lines[0].startswith("# "), (
        "the capture must keep the probe's SMU time-filter caveat line"
    )
    rows: list[dict[str, object]] = []
    for raw in csv.DictReader(lines[1:]):
        row: dict[str, object] = {
            "timestamp": raw["timestamp"],
            "elapsed_seconds": float(raw["elapsed_seconds"]),
        }
        for column in probe.ACTIVITY_COLUMNS:
            row[column.column] = float(raw[column.column])
        row[probe.POWER_COLUMN] = float(raw[probe.POWER_COLUMN])
        busy = raw[probe.BUSY_COLUMN]
        row[probe.BUSY_COLUMN] = float(busy) if busy != "" else None
        rows.append(row)

    meta = json.loads(CAPTURE_META.read_text())
    verdict = probe.build_verdict(rows).to_dict()
    powers = [row[probe.POWER_COLUMN] for row in rows]
    gfx_values = [row[GFX_FIELD] for row in rows]
    busy_values = [row[BUSY_FIELD] for row in rows]
    return {
        "rows": rows,
        "meta": meta,
        "verdict": verdict,
        "gfx": verdict["fields"][GFX_FIELD],
        "busy": verdict["fields"][BUSY_FIELD],
        "powers": powers,
        "gfx_values": gfx_values,
        "busy_values": busy_values,
        "distinct_gfx": sorted({float(value) for value in gfx_values}),
    }


def _mean(value: float) -> str:
    return f"{value:.2f} %"


def _separation(value: float) -> str:
    return f"{value:.2f} pp"


def _false_busy(stats: dict) -> str:
    return f"{stats['idle_false_busy_pct']:.1f} %"


def _distinct_phrase(values: list[float]) -> str:
    numbers = [f"{value:.0f}" for value in values]
    return ", ".join(numbers[:-1]) + " and " + numbers[-1]


def _assert_bar_stated(text: str, *, where: str) -> None:
    """The pass/fail bar in *text* must equal the engine's constants (AC4)."""
    for fragment in (
        f"separation >= {verdict_engine.PASS_SEPARATION_PP:.0f} pp",
        f"r >= {verdict_engine.PASS_PEARSON_R:.1f}",
        f"false-busy <= {verdict_engine.PASS_MAX_IDLE_FALSE_BUSY_SHARE * 100.0:.0f} %",
        f"idle band (< {verdict_engine.IDLE_POWER_CEILING_W:.0f} W)",
        f"compute band (>= {verdict_engine.COMPUTE_POWER_FLOOR_W:.0f} W)",
    ):
        assert fragment in text, f"{where} must state the bar element {fragment!r}"


# --- The committed evidence -------------------------------------------------


def test_capture_fixture_matches_its_sidecar() -> None:
    """The capture, its hash and its metadata are one record, not three."""
    capture = _capture()
    meta = capture["meta"]
    rows = capture["rows"]

    digest = hashlib.sha256(CAPTURE_CSV.read_bytes()).hexdigest()
    assert digest == meta["csv_sha256"], "the capture changed without its sidecar"

    assert len(rows) == meta["capture"]["sample_count"] == 181
    assert meta["capture"]["interval_seconds"] == probe.DEFAULT_INTERVAL_SECONDS
    assert meta["capture"]["duration_seconds"] == 180
    assert rows[0]["timestamp"] == meta["capture_window"]["started_at"]
    assert rows[-1]["timestamp"] == meta["capture_window"]["finished_at"]
    assert meta["host_kernel"].count(".") >= 2, "the sidecar must name the kernel"
    assert meta["probe_command"].startswith("python3 scripts/gpu_metrics_probe.py")
    assert "SMU time-filtered" in meta["note"]


def test_capture_holds_both_power_bands() -> None:
    """A one-sided capture could not discriminate, so the record must hold both."""
    counts = _capture()["verdict"]["band_counts"]

    assert counts["total"] == 181
    assert counts["invalid"] == 0
    assert counts["idle"] >= 20 and counts["compute"] >= 20, counts
    assert counts["idle"] + counts["middle"] + counts["compute"] == counts["total"]


def test_average_gfx_activity_is_gpu_busy_percent_on_every_sample() -> None:
    """The mechanism the documents state: two names, one saturated counter."""
    capture = _capture()

    assert capture["gfx_values"] == capture["busy_values"], (
        "the recorded claim that average_gfx_activity is identical to "
        "gpu_busy_percent no longer holds for this capture"
    )
    assert capture["distinct_gfx"] == [98.0, 99.0, 100.0], (
        "the record says the field only ever reads 98, 99 or 100"
    )


def test_gfx_activity_fails_every_element_of_the_bar() -> None:
    """The negative verdict is all three bar elements failing, not one."""
    capture = _capture()
    gfx = capture["gfx"]

    assert capture["verdict"]["status"] == verdict_engine.STATUS_OK
    assert gfx["decision"] == verdict_engine.DECISION_FAIL
    assert gfx["separation_pp"] < verdict_engine.PASS_SEPARATION_PP
    assert gfx["pearson_r"] < verdict_engine.PASS_PEARSON_R
    assert gfx["idle_false_busy_share"] > verdict_engine.PASS_MAX_IDLE_FALSE_BUSY_SHARE
    assert len(gfx["reasons"]) == 3
    assert GFX_FIELD not in capture["verdict"]["passed_fields"]


def test_only_cpu_core_residency_columns_pass() -> None:
    """The fields that pass are CPU C0 residency, not a GPU utilisation input."""
    passed = _capture()["verdict"]["passed_fields"]

    assert passed == ["average_core_c0_activity[5]", "average_core_c0_activity[7]"]
    assert not any(name.startswith(("average_gfx", "average_vcn", "average_ipu")) for name in passed)


# --- The documents ----------------------------------------------------------


def test_unit_document_records_the_measured_verdict() -> None:
    """AC1/AC5: verdict, capture identity, band statistics and the consequence."""
    section = _prose(_section(GPU_METRICS_DOC.read_text(), VERDICT_HEADING))
    capture = _capture()
    meta = capture["meta"]
    gfx = capture["gfx"]

    for fact in (
        "negative verdict",
        meta["captured_at"][:10],
        meta["host_kernel"],
        f"{meta['capture']['sample_count']} samples",
        f"{capture['verdict']['band_counts']['idle']} idle",
        f"{capture['verdict']['band_counts']['compute']} compute",
        f"{capture['verdict']['band_counts']['middle']} middle",
        _mean(gfx["idle_band_mean"]),
        _mean(gfx["compute_band_mean"]),
        _separation(gfx["separation_pp"]),
        f"{gfx['pearson_r']:.3f}",
        _false_busy(gfx),
        f"{min(capture['powers']):.1f} W",
        f"{max(capture['powers']):.1f} W",
        _distinct_phrase(capture["distinct_gfx"]),
        "activity series is published",
    ):
        assert fact in section, f"the verdict section must record {fact!r}"

    assert f"identical to `{BUSY_FIELD}`" in section
    assert f"`{GFX_FIELD}`" in section


def test_unit_document_pins_the_v3_0_decode_layout() -> None:
    """AC1: the layout is recorded, so the record is decodable without the code."""
    section = _prose(_section(GPU_METRICS_DOC.read_text(), VERDICT_HEADING))

    for fragment in (
        "gpu_metrics_v3_0",
        "kgd_pp_interface.h",
        "pack(4)",
        f"`structure_size` | {decoder.STRUCTURE_SIZE_BYTES}",
        f"`format_revision` | {decoder.FORMAT_REVISION}",
        f"`content_revision` | {decoder.CONTENT_REVISION}",
        f"{decoder.FIELD_REGION_BYTES}-byte field region",
        f"{decoder.TAIL_PADDING_BYTES} bytes of tail padding",
    ):
        assert fragment in section, f"the layout must be recorded as {fragment!r}"

    by_name = {spec.name: spec for spec in decoder.FIELD_TABLE}
    for name in (
        GFX_FIELD,
        "average_vcn_activity",
        "average_ipu_activity",
        "average_core_c0_activity",
        "time_filter_alphavalue",
    ):
        spec = by_name[name]
        expected = f"| `{name}` | {spec.offset} | {spec.count} |"
        assert expected in section, f"the layout table must pin {name!r} at {spec.offset}"


def test_re_measure_recipe_is_documented() -> None:
    """AC4: the command, the bands, the bar and the trigger are spelled out."""
    recipe = _prose(
        _section(GPU_METRICS_DOC.read_text(), REMEASURE_HEADING, level=3)
    )

    for fragment in (
        "scripts/gpu_metrics_probe.py",
        "--duration",
        "--interval",
        "--verdict",
        "kernel",
        "firmware",
        "driver",
        "time_filter_alphavalue",
        "undocumented",
    ):
        assert fragment in recipe, f"the re-measure recipe must mention {fragment!r}"

    _assert_bar_stated(recipe, where="the re-measure recipe")
    assert "both bands" in recipe.lower()


def test_readme_gpu_metrics_row_quotes_the_band_statistics() -> None:
    """AC2: the comparison table gains a ``gpu_metrics`` row with the numbers."""
    section = _section(MONITORING_README.read_text(), "Why not `amd-smi`", level=3)
    capture = _capture()
    gfx = capture["gfx"]

    rows = [line for line in section.splitlines() if line.startswith("| `gpu_metrics")]
    assert rows, "the comparison table has no `gpu_metrics` row"
    row = " ".join(rows)

    for fragment in (
        _mean(gfx["idle_band_mean"]),
        _mean(gfx["compute_band_mean"]),
        _separation(gfx["separation_pp"]),
        f"{gfx['pearson_r']:.3f}",
        _false_busy(gfx),
        f"identical to `{BUSY_FIELD}`",
        capture["meta"]["host_kernel"],
        str(capture["meta"]["captured_at"][:10]),
    ):
        assert fragment in row, f"the gpu_metrics row must quote {fragment!r}"

    assert f"`{GFX_FIELD}`" in row


def test_readme_notes_the_re_measure_recipe() -> None:
    """AC2/AC4: the README points at the recipe and states the bar."""
    section = _prose(
        _section(MONITORING_README.read_text(), "Why not `amd-smi`", level=3)
    )

    assert "scripts/gpu_metrics_probe.py" in section
    assert "re-measur" in section.lower()
    assert "docs/systemd/gpu-metrics.service" in section
    _assert_bar_stated(section, where="the monitoring README")


def test_rules_header_records_the_verdict() -> None:
    """AC3: the rule file says why no ``gpu_metrics`` field feeds a rule."""
    header = _yaml_header(RULES_YAML)
    capture = _capture()
    gfx = capture["gfx"]

    for fragment in (
        "gpu_metrics",
        f"`{GFX_FIELD}`",
        capture["meta"]["captured_at"][:10],
        capture["meta"]["host_kernel"],
        f"{capture['meta']['capture']['sample_count']} samples",
        _mean(gfx["idle_band_mean"]),
        _mean(gfx["compute_band_mean"]),
        _separation(gfx["separation_pp"]),
        "identical to gpu_busy_percent",
    ):
        assert fragment in header, f"the rule header must record {fragment!r}"

    assert "no rule" in header.lower()


def test_probe_and_decoder_point_at_the_recorded_verdict() -> None:
    """AC6: a future reader of either module lands on the record, not a guess."""
    capture = _capture()
    meta = capture["meta"]

    for module in (probe, decoder):
        docstring = module.__doc__ or ""
        for fragment in (
            "negative",
            "docs/systemd/gpu-metrics.service",
            VERDICT_HEADING,
            meta["captured_at"][:10],
            meta["host_kernel"],
        ):
            assert fragment in docstring, (
                f"{module.__name__}'s docstring must point at the recorded "
                f"verdict ({fragment!r} missing)"
            )
