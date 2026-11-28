#!/usr/bin/env python3
"""Drift guard between the documented and emitted GPU sampler series.

LP-0MVZEW7G7002CLDX (parent LP-0MVSQT7YC0045HFW).

The ``gpu_metrics`` (``amd_gpu_metrics_v3_0``) verdict is **negative**: the
2026-11-18 capture on kernel 7.0.0-38 (61 samples at 1 s: 23 idle below 60 W,
25 compute at or above 90 W) read ``average_gfx_activity`` = 100 % in both
bands — 0 pp separation, Pearson r undefined (zero variance) and a 100 %
idle-band false-busy share against a <= 10 % bar. It reproduces the 2026-10-29
intake capture (120 samples, 100 % across 42–123 W). No ``gpu_metrics``
activity series is therefore published: the 10 s textfile sampler's series set
is unchanged.

The series table in ``docs/systemd/gpu-metrics.service`` is the reproduction
source for a clean clone, so it must equal the set the sampler actually emits.
These tests execute the fenced bash block from that document against a fixture
sysfs tree and compare the emitted set with the table in the same document.
Adding a series to the sampler without documenting it, or documenting a series
the sampler does not emit, fails here.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_SYSTEMD_DIR = PROJECT_ROOT / "docs" / "systemd"
GPU_METRICS_DOC = DOCS_SYSTEMD_DIR / "gpu-metrics.service"

FENCE = "```"

#: Fixture readings for a fully readable amdgpu device. The three memory
#: attributes are the documented series; ``gpu_busy_percent`` and the binary
#: ``gpu_metrics`` blob stay readable on purpose, so the sampler's output proves
#: it ignores them (the retired counter and the negative-verdict activity blob).
FULL_GPU_SYSFS = {
    "gpu_busy_percent": "100",
    "mem_info_vram_used": "309604352",
    "mem_info_vram_total": "1073741824",
    "mem_info_gtt_used": "36573331456",
}

GPU_METRICS_BLOB = PROJECT_ROOT / "tests" / "fixtures" / "gpu_metrics_v3_0.bin"

#: The series a positive ``gpu_metrics`` verdict would have published, and the
#: decoded field name the guard in ``proxy/tests/test_monitoring_deployment.py``
#: forbids. Neither may appear in the sampler or its documentation table.
GPUMETRICS_ACTIVITY_SERIES = frozenset({"average_gfx_activity", "gpu_gfx_activity"})


def _fenced_blocks(path: Path) -> list[tuple[str, str]]:
    """Return ``(language, body)`` for every fenced code block in *path*."""
    blocks: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    language = ""
    for line in path.read_text().splitlines():
        if line.startswith(FENCE):
            if current is None:
                language = line[len(FENCE) :].strip()
                current = []
            else:
                blocks.append((language, "\n".join(current) + "\n"))
                current = None
            continue
        if current is not None:
            current.append(line)
    return blocks


def _sampler_script() -> str:
    """Return the sampler script documented with ``gpu-metrics.service``."""
    for language, body in _fenced_blocks(GPU_METRICS_DOC):
        if language == "bash" and body.lstrip().startswith("#!"):
            return body
    raise AssertionError(f"{GPU_METRICS_DOC} does not document the sampler in a fenced bash block")


def _section(text: str, heading: str) -> str:
    """Return the body of the ``##`` section whose heading contains *heading*."""
    collected: list[str] = []
    capturing = False
    for line in text.splitlines():
        if line.startswith("## "):
            if capturing:
                break
            capturing = heading.lower() in line.lower()
        elif capturing:
            collected.append(line)
    assert collected, f"{GPU_METRICS_DOC} has no '## {heading}' section"
    return "\n".join(collected)


def _documented_series() -> set[str]:
    """Return the series named in the document's ``Metric names`` table.

    Only the first cell of a table row counts, and it must be a backticked
    lower-case series name; the header row and the ``|---|`` separator do not
    match, so prose and other tables cannot be mistaken for published series.
    """
    table = _section(GPU_METRICS_DOC.read_text(), "Metric names")
    names = set(re.findall(r"^\|\s*`([a-z_][a-z0-9_]*)`\s*\|", table, re.MULTILINE))
    assert names, "the Metric names table names no series"
    return names


def _fake_gpu_sysfs(tmp_path: Path, **attributes: str) -> Path:
    """Create a fixture sysfs tree holding the amdgpu attributes given."""
    device = tmp_path / "sys" / "class" / "drm" / "card0" / "device"
    device.mkdir(parents=True)
    for name, value in attributes.items():
        (device / name).write_text(f"{value}\n")
    return device


def _run_documented_sampler(tmp_path: Path, device: Path) -> Path:
    """Run the documented sampler against *device* and return its textfile dir."""
    sampler = tmp_path / "gpu-metrics-textfile.sh"
    sampler.write_text(_sampler_script())
    out_dir = tmp_path / "textfile"
    out_dir.mkdir()
    result = subprocess.run(
        ["bash", str(sampler), str(out_dir)],
        capture_output=True,
        text=True,
        env={**os.environ, "GPU_SYSFS": str(device)},
    )
    assert result.returncode == 0, f"sampler failed: {result.stderr}"
    return out_dir


def _emitted_series(out_dir: Path) -> dict[str, str]:
    """Return the ``{series: value}`` pairs the sampler wrote to ``gpu.prom``."""
    return dict(
        re.findall(
            r'^(gpu_\w+)\{gpu="card0"\} (\S+)$',
            (out_dir / "gpu.prom").read_text(),
            re.MULTILINE,
        )
    )


def test_documented_series_table_is_parseable() -> None:
    """The documented set is non-empty and holds only real series names."""
    documented = _documented_series()
    assert documented == {
        "gpu_vram_used_bytes",
        "gpu_vram_total_bytes",
        "gpu_gtt_used_bytes",
    }


def test_emitted_series_set_matches_documented_table(tmp_path: Path) -> None:
    """AC3: behaviour and documentation cannot drift, in either direction.

    The sampler is executed exactly as documented (the fenced bash block), so
    the assertion covers the reproduction source a clean clone installs, not a
    host-local copy.
    """
    device = _fake_gpu_sysfs(tmp_path, **FULL_GPU_SYSFS)
    emitted = set(_emitted_series(_run_documented_sampler(tmp_path, device)))

    documented = _documented_series()
    assert emitted == documented, (
        "the sampler's emitted series and the documented series table diverged; "
        f"emitted-only={sorted(emitted - documented)}, "
        f"documented-only={sorted(documented - emitted)}"
    )


def test_no_gpu_metrics_activity_series_is_published(tmp_path: Path) -> None:
    """AC2: the negative verdict publishes nothing, even with a readable blob.

    ``gpu_metrics`` and ``gpu_busy_percent`` are both present in the fixture,
    so the emitted set proves the sampler ignores them rather than merely
    failing to find them.
    """
    device = _fake_gpu_sysfs(tmp_path, **FULL_GPU_SYSFS)
    (device / "gpu_metrics").write_bytes(GPU_METRICS_BLOB.read_bytes())

    emitted = set(_emitted_series(_run_documented_sampler(tmp_path, device)))

    assert emitted == _documented_series()
    assert not (emitted & GPUMETRICS_ACTIVITY_SERIES), (
        "the sampler published a gpu_metrics activity series although the "
        "verdict is negative and the guard forbids it"
    )
    assert not any("gpu_metrics" in name for name in emitted)
    assert "average_gfx_activity" not in _sampler_script()


def test_sampler_writes_nothing_when_read_fails(tmp_path: Path) -> None:
    """AC1: the write-nothing-on-read-failure behaviour survives unchanged."""
    device = _fake_gpu_sysfs(tmp_path, gpu_busy_percent="100")
    (device / "gpu_metrics").write_bytes(GPU_METRICS_BLOB.read_bytes())

    out_dir = _run_documented_sampler(tmp_path, device)

    assert not (out_dir / "gpu.prom").exists(), (
        "the sampler must not publish fabricated values when the memory "
        "attributes are unreadable"
    )


def test_documentation_records_the_negative_verdict_and_resolution_floor() -> None:
    """AC5: the document says why nothing is published, and at what resolution.

    The values are SMU time-filtered, so the 10 s sampler cadence and the 15 s
    Prometheus scrape remain the resolution floor; the section must say so
    rather than implying per-second fidelity.
    """
    text = " ".join(GPU_METRICS_DOC.read_text().split()).lower()

    for fact in (
        "no `gpu_metrics` activity series is published",
        "2026-11-18",
        "7.0.0-38",
        "average_gfx_activity",
        "100 % idle-band false-busy",
        "resolution floor",
        "10 s",
        "15 s",
    ):
        assert fact in text, f"the negative-verdict section must state {fact!r}"
