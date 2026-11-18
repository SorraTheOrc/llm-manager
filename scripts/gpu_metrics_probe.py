#!/usr/bin/env python3
"""Paired sampler and verdict probe for the ``gpu_metrics`` activity fields.

Why this exists (LP-0MVZEW6UL0054D8W, parent LP-0MVSQT7YC0045HFW)
-----------------------------------------------------------------
The usage report's GPU busy figure is a *power proxy*
(``gpu:computing:bool >= 90`` W), ``gpu_busy_percent`` was rejected for
saturating, and the last candidate hardware activity source is the
``amd_gpu_metrics_v3_0`` blob at ``/sys/class/drm/card0/device/gpu_metrics``.
Settling whether any field in that blob carries a *graded* activity signal
needs one reproducible command that samples every candidate activity field
next to the two reference signals and prints the per-field discrimination
statistics. This module is that command.

The three collaborators it does **not** duplicate:

* :mod:`lib.gpu_metrics_decoder` — the pinned v3.0 layout and header validation;
* :mod:`lib.gpu_metrics_verdict` — the pass bar and the per-field statistics;
* :mod:`gpu_utilisation` (LP-0MVNWUEY8008XM6W) — slot attribution and the
  Prometheus-side utilisation CLI. This probe samples sysfs only: no daemon, no
  port, no scrape job, one 264-byte read per sample.

Sampling model
--------------
One sample reads ``gpu_metrics`` (decoded), the amdgpu hwmon
``power1_average`` (microwatts → watts) and, when the driver still publishes
it, ``gpu_busy_percent``. A row is built only after all three reads succeed, so
an unreadable file aborts the capture with a non-zero exit rather than
emitting a partially written row. Samples are taken at ``elapsed = 0, interval,
2 × interval, …`` while ``elapsed <= duration`` — exactly
``floor(duration / interval) + 1`` rows, the same grid convention the
utilisation CLI documents. A
``--duration 1800`` (30 min) ``--interval 1`` capture therefore yields 1 801
rows.

Defaults (also shown by ``--help``)
-----------------------------------
* ``--duration 1800`` — seconds to sample (30 minutes);
* ``--interval 1.0`` — seconds between samples.

The capture is deliberately long enough for both power bands (the intake saw
idle at 42–66 W and compute at 90–123 W within ~15 minutes); longer captures are
supported by the same flags.

Output
------
CSV on stdout by default, one row per sample, preceded by one ``#`` comment
line stating the time-filter caveat. ``--json`` emits a single JSON document
with a ``meta`` block, the ``samples`` array and (with ``--verdict``) the
verdict object. ``--verdict`` adds the human-readable per-field statistics,
band counts and pass/fail decision; in CSV mode it is written to **stderr** so
stdout remains a valid CSV stream.

Overrides (for tests and re-measurement)
----------------------------------------
``--sysfs-root`` (env ``GPU_METRICS_SYSFS_ROOT``) replaces ``/sys``; the device
directory is then ``<root>/class/drm/card0/device``. ``--hwmon-dir`` (env
``GPU_METRICS_HWMON_DIR``) replaces the auto-discovered
``<device>/hwmon/hwmon*`` directory that holds ``power1_average``.

No instantaneous claims
-----------------------
Every value in the record is SMU **time filtered**
(``time_filter_alphavalue`` reads 1 000 000 µs on this host) with an
undocumented window, so no claim finer than the sample interval is implied.
The CSV comment line, the JSON ``meta.note`` and the verdict header all say so.

Exit status
-----------
``0`` — the capture completed (a negative verdict is a complete outcome, not a
failure). ``2`` — the capture could not be read or decoded; nothing partial is
written.

Usage
-----
::

    python3 scripts/gpu_metrics_probe.py                     # 30 min, 1 s, CSV
    python3 scripts/gpu_metrics_probe.py --duration 60 --interval 0.5 --json
    python3 scripts/gpu_metrics_probe.py --duration 1800 --verdict > capture.csv
    GPU_METRICS_SYSFS_ROOT=/tmp/fixture python3 scripts/gpu_metrics_probe.py

Example
-------
>>> columns = activity_columns()
>>> columns[0].column
'average_gfx_activity'
>>> len(columns)  # gfx + vcn + 8 IPU columns + 16 core C0 columns
26
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:  # pragma: no cover - import-layout helper
    sys.path.insert(0, str(SCRIPT_DIR))

from lib.gpu_metrics_decoder import (  # noqa: E402
    FIELD_TABLE,
    GpuMetricsError,
    decode,
)
from lib.gpu_metrics_verdict import (  # noqa: E402
    DECISION_INSUFFICIENT,
    PASS_MAX_IDLE_FALSE_BUSY_SHARE,
    PASS_PEARSON_R,
    PASS_SEPARATION_PP,
    STATUS_OK,
    compute_verdict,
)

# --- Defaults and environment overrides -------------------------------------

DEFAULT_DURATION_SECONDS = 1800.0
"""30 minutes: long enough for both power bands on this host (see module doc)."""

DEFAULT_INTERVAL_SECONDS = 1.0
"""One second between samples; the finest claim the time-filter supports."""

DEFAULT_SYSFS_ROOT = Path("/sys")
SYSFS_ROOT_ENV = "GPU_METRICS_SYSFS_ROOT"
HWMON_DIR_ENV = "GPU_METRICS_HWMON_DIR"

DEVICE_RELATIVE_PATH = Path("class") / "drm" / "card0" / "device"
"""Card-0 device directory under the sysfs root."""

METRICS_FILENAME = "gpu_metrics"
POWER_FILENAME = "power1_average"
BUSY_FILENAME = "gpu_busy_percent"

POWER_MICROWATTS_PER_WATT = 1_000_000.0
POWER_WATTS_DECIMALS = 3
ELAPSED_DECIMALS = 3
DURATION_EPSILON_SECONDS = 1e-9
"""Float slack so ``elapsed <= duration`` does not lose the last sample."""

ACTIVITY_UNIT = "%"
"""Decoder fields carrying this unit are the candidate activity fields."""

VERDICT_BASELINE_FIELD = BUSY_FILENAME
"""The rejected counter is evaluated next to the candidates as the baseline."""

POWER_WATTS_KEY = "power_watts"
"""Key :mod:`lib.gpu_metrics_verdict` uses to define the power bands."""

TIMESTAMP_COLUMN = "timestamp"
ELAPSED_COLUMN = "elapsed_seconds"
POWER_COLUMN = "power1_average_watts"
BUSY_COLUMN = "gpu_busy_percent"

TIME_FILTER_NOTE = (
    "gpu_metrics values are SMU time-filtered with an undocumented window "
    "(time_filter_alphavalue); no claim finer than the sample interval is implied."
)

EXIT_OK = 0
EXIT_PROBE_ERROR = 2

# --- Errors -----------------------------------------------------------------


class ProbeError(RuntimeError):
    """A capture could not be read, decoded or parsed.

    Carries the offending ``path`` when there is one, so the caller can print an
    actionable message (which file, what to check) instead of a traceback.
    """

    def __init__(self, message: str, *, path: Path | None = None) -> None:
        super().__init__(message)
        self.path = path


# --- Candidate activity fields ---------------------------------------------


@dataclass(frozen=True, slots=True)
class ActivityColumn:
    """One CSV/JSON column projected from a decoded activity field.

    Array members (``average_ipu_activity``, ``average_core_c0_activity``) are
    expanded per element — ``average_ipu_activity[0]`` — because a series mean
    would hide the per-column behaviour the verdict is judging.
    """

    column: str
    field: str
    index: int | None = None

    def value(self, decoded: dict[str, object]) -> int | float:
        raw = decoded[self.field]
        return raw if self.index is None else raw[self.index]  # type: ignore[index]


def activity_columns() -> tuple[ActivityColumn, ...]:
    """Every decoder field whose unit is a percentage, in field-table order."""
    columns: list[ActivityColumn] = []
    for spec in FIELD_TABLE:
        if spec.unit != ACTIVITY_UNIT:
            continue
        if spec.is_array:
            columns.extend(ActivityColumn(f"{spec.name}[{index}]", spec.name, index) for index in range(spec.count))
        else:
            columns.append(ActivityColumn(spec.name, spec.name))
    return tuple(columns)


ACTIVITY_COLUMNS: tuple[ActivityColumn, ...] = activity_columns()

COLUMNS: tuple[str, ...] = (
    TIMESTAMP_COLUMN,
    ELAPSED_COLUMN,
    *(column.column for column in ACTIVITY_COLUMNS),
    POWER_COLUMN,
    BUSY_COLUMN,
)
"""Row layout shared by the CSV header and the JSON sample objects."""

VERDICT_FIELDS: tuple[str, ...] = (
    *(column.column for column in ACTIVITY_COLUMNS),
    VERDICT_BASELINE_FIELD,
)
"""Candidate fields plus the rejected counter as the comparison baseline."""


# --- Clock indirection (tests drive the schedule without sleeping) ----------


def _utc_now() -> datetime:
    """Wall-clock ``now``; indirection so tests can drive the schedule."""
    return datetime.now(UTC)


def _monotonic() -> float:
    return time.monotonic()


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


# --- Paths ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProbePaths:
    """The three sysfs files one sample reads."""

    metrics: Path
    power: Path
    busy: Path
    hwmon_dir: Path


def _hwmon_candidates(device_dir: Path) -> tuple[Path, ...]:
    hwmon_root = device_dir / "hwmon"
    if not hwmon_root.is_dir():
        return ()
    return tuple(sorted(path for path in hwmon_root.iterdir() if path.is_dir()))


def resolve_paths(sysfs_root: Path, hwmon_dir: Path | None = None) -> ProbePaths:
    """Resolve the metrics, power and busy paths under a sysfs tree.

    ``hwmon_dir`` overrides auto-discovery; otherwise the first
    ``<device>/hwmon/hwmon*`` directory that carries ``power1_average`` wins.
    A tree with no hwmon power attribute is an error: the reference signal is
    what the verdict compares against, so a capture without it proves nothing.
    """
    device_dir = Path(sysfs_root) / DEVICE_RELATIVE_PATH
    if hwmon_dir is None:
        candidates = _hwmon_candidates(device_dir)
        for candidate in candidates:
            if (candidate / POWER_FILENAME).is_file():
                hwmon_dir = candidate
                break
        else:
            tried = ", ".join(str(path) for path in candidates) or "none"
            raise ProbeError(
                f"no {POWER_FILENAME} under {device_dir / 'hwmon'} "
                f"(hwmon directories tried: {tried}); pass --hwmon-dir or "
                f"set {HWMON_DIR_ENV}",
                path=device_dir / "hwmon",
            )
    resolved = Path(hwmon_dir)
    return ProbePaths(
        metrics=device_dir / METRICS_FILENAME,
        power=resolved / POWER_FILENAME,
        busy=device_dir / BUSY_FILENAME,
        hwmon_dir=resolved,
    )


# --- Signal reads -----------------------------------------------------------


def _read_int(path: Path, *, label: str) -> int:
    try:
        text = path.read_text().strip()
    except OSError as exc:
        raise ProbeError(
            f"cannot read {label} at {path}: {exc.strerror or exc}",
            path=path,
        ) from exc
    try:
        return int(text)
    except ValueError as exc:
        raise ProbeError(
            f"{label} at {path} is not an integer: {text!r}",
            path=path,
        ) from exc


def read_metrics(path: Path) -> dict[str, object]:
    """Read and decode one ``gpu_metrics`` record, failing closed."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ProbeError(
            f"cannot read gpu_metrics at {path}: {exc.strerror or exc}. "
            "Check that the amdgpu driver publishes it on this host, or point "
            f"--sysfs-root / {SYSFS_ROOT_ENV} at a fixture tree",
            path=path,
        ) from exc
    try:
        return decode(raw)
    except GpuMetricsError as exc:
        raise ProbeError(f"cannot decode gpu_metrics at {path}: {exc}", path=path) from exc


def read_optional_int(path: Path) -> int | None:
    """Read an integer that may legitimately be absent.

    ``gpu_busy_percent`` is retired on newer drivers; a missing or unparsable
    file yields ``None`` (an empty CSV cell / JSON ``null``) rather than failing
    the capture, because it is a comparison baseline and not the signal under
    test.
    """
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


# --- Sampler ----------------------------------------------------------------


class GpuMetricsProbe:
    """Reads one paired sample and schedules a capture at a fixed interval."""

    def __init__(
        self,
        paths: ProbePaths,
        *,
        columns: Sequence[ActivityColumn] = ACTIVITY_COLUMNS,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.paths = paths
        self.columns = tuple(columns)
        self._clock = clock or _utc_now
        self._monotonic = monotonic or _monotonic
        self._sleep = sleep or _sleep

    def read_sample(self, *, elapsed_seconds: float = 0.0) -> dict[str, object]:
        """Read all three signals, then build one complete row.

        The row is returned only after every read succeeded, so callers never
        receive (and therefore never print) a partially populated row.
        """
        decoded = read_metrics(self.paths.metrics)
        power_microwatts = _read_int(self.paths.power, label="hwmon power1_average")
        busy_percent = read_optional_int(self.paths.busy)

        row: dict[str, object] = {
            TIMESTAMP_COLUMN: self._clock().isoformat(timespec="milliseconds"),
            ELAPSED_COLUMN: round(elapsed_seconds, ELAPSED_DECIMALS),
        }
        for column in self.columns:
            row[column.column] = column.value(decoded)
        row[POWER_COLUMN] = round(power_microwatts / POWER_MICROWATTS_PER_WATT, POWER_WATTS_DECIMALS)
        row[BUSY_COLUMN] = busy_percent
        return row

    def preflight(self) -> dict[str, object]:
        """Prove the paths are readable/decodable before any output is emitted."""
        return self.read_sample(elapsed_seconds=0.0)

    def run(
        self,
        duration: float,
        interval: float,
        *,
        on_row: Callable[[dict[str, object]], None] | None = None,
    ) -> list[dict[str, object]]:
        """Sample at ``0, interval, 2 × interval, …`` until ``duration`` elapses.

        Returns every row; ``on_row`` is invoked per sample so a caller can
        stream output. The schedule is ``floor(duration / interval) + 1``
        samples, matching the utilisation CLI's documented grid.
        """
        if duration <= 0:
            raise ValueError("duration must be positive")
        if interval <= 0:
            raise ValueError("interval must be positive")

        start_monotonic = self._monotonic()
        rows: list[dict[str, object]] = []
        index = 0
        while True:
            elapsed = self._monotonic() - start_monotonic
            row = self.read_sample(elapsed_seconds=elapsed)
            rows.append(row)
            if on_row is not None:
                on_row(row)

            index += 1
            next_elapsed = index * interval
            if next_elapsed > duration + DURATION_EPSILON_SECONDS:
                break
            delay = next_elapsed - (self._monotonic() - start_monotonic)
            if delay > 0:
                self._sleep(delay)
        return rows


# --- Verdict ----------------------------------------------------------------


def build_verdict_samples(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Project captured rows into the shape :func:`compute_verdict` expects.

    The verdict engine defines its bands from ``power_watts``
    (:data:`POWER_WATTS_KEY`), while the capture records
    ``power1_average_watts``; fields the capture could not read are dropped so
    a sparse sample shrinks the denominator instead of crashing.
    """
    samples: list[dict[str, object]] = []
    for row in rows:
        sample: dict[str, object] = {POWER_WATTS_KEY: row[POWER_COLUMN]}
        for column in ACTIVITY_COLUMNS:
            sample[column.column] = row[column.column]
        busy = row[BUSY_COLUMN]
        if busy is not None:
            sample[VERDICT_BASELINE_FIELD] = busy
        samples.append(sample)
    return samples


def build_verdict(rows: Iterable[dict[str, object]]):
    """Compute the per-field verdict over captured rows."""
    return compute_verdict(build_verdict_samples(rows), fields=VERDICT_FIELDS)


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    assert isinstance(value, (int, float))
    return f"{value:.{digits}f}"


def render_verdict(
    verdict,
    *,
    sample_count: int,
    interval: float,
    duration: float,
) -> str:
    """Human-readable verdict: bar, band counts, per-field stats, decisions.

    Rendered from :meth:`Verdict.to_dict` so the printed numbers are exactly
    the numbers recorded in ``--json`` output.
    """
    data = verdict.to_dict()
    band_counts = data["band_counts"]
    lines = [
        "gpu_metrics discrimination verdict",
        (
            "bar: separation >= "
            f"{PASS_SEPARATION_PP:.0f} pp AND pearson r >= {PASS_PEARSON_R:.1f} "
            "AND idle false-busy <= "
            f"{PASS_MAX_IDLE_FALSE_BUSY_SHARE * 100.0:.0f}%"
        ),
        (f"capture: {sample_count} sample(s), duration {duration:g} s, interval {interval:g} s"),
        ("bands: idle={idle} middle={middle} compute={compute} invalid={invalid} total={total}".format(**band_counts)),
    ]
    status = data["status"]
    if status == STATUS_OK:
        lines.append("status: ok (both power bands present)")
    else:
        lines.append(
            f"status: {DECISION_INSUFFICIENT} — a power band is missing "
            "(idle < 60 W or compute >= 90 W), so the capture cannot discriminate"
        )
    lines.append(f"note: {TIME_FILTER_NOTE}")
    lines.append("")
    lines.append(
        f"{'field':32} {'idle_n':>6} {'compute_n':>9} {'idle_mean':>10} "
        f"{'compute_mean':>13} {'sep_pp':>8} {'pearson_r':>9} "
        f"{'false_busy':>10} decision"
    )
    for name, stats in data["fields"].items():
        false_busy = "n/a" if stats["idle_false_busy_pct"] is None else f"{stats['idle_false_busy_pct']:.1f}%"
        lines.append(
            f"{name:32} {stats['idle_band_count']:>6} "
            f"{stats['compute_band_count']:>9} "
            f"{_fmt(stats['idle_band_mean']):>10} "
            f"{_fmt(stats['compute_band_mean']):>13} "
            f"{_fmt(stats['separation_pp']):>8} "
            f"{_fmt(stats['pearson_r'], 3):>9} "
            f"{false_busy:>10} {stats['decision']}"
        )
        for reason in stats["reasons"]:
            lines.append(f"  - {reason}")

    passed = data["passed_fields"]
    lines.append("")
    lines.append(f"passed: {', '.join(passed) if passed else 'none'}")
    return "\n".join(lines)


# --- CLI --------------------------------------------------------------------


def _finite_positive_seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} must be a positive, finite number of seconds")
    return value


def _path_argument(text: str) -> Path:
    return Path(text).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu_metrics_probe.py",
        description=(
            "Sample every candidate gpu_metrics activity field alongside amdgpu "
            "hwmon power1_average and gpu_busy_percent, and print the "
            "discrimination statistics that decide whether any field tracks GPU "
            "activity. Values are SMU time-filtered, so no claim finer than the "
            "sample interval is implied."
        ),
        epilog=(
            "Defaults: --duration 1800 (30 min) and --interval 1.0 s, yielding "
            "floor(duration/interval)+1 = 1801 rows. CSV goes to stdout; "
            "--verdict in CSV mode goes to stderr so stdout stays a valid CSV "
            f"stream. Overrides: --sysfs-root / ${SYSFS_ROOT_ENV}, "
            f"--hwmon-dir / ${HWMON_DIR_ENV}."
        ),
    )
    parser.add_argument(
        "--duration",
        type=_finite_positive_seconds,
        default=DEFAULT_DURATION_SECONDS,
        metavar="SECONDS",
        help=f"how long to sample (default: {DEFAULT_DURATION_SECONDS:g} = 30 min)",
    )
    parser.add_argument(
        "--interval",
        type=_finite_positive_seconds,
        default=DEFAULT_INTERVAL_SECONDS,
        metavar="SECONDS",
        help=f"seconds between samples (default: {DEFAULT_INTERVAL_SECONDS:g})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON document (meta, samples, and the verdict with --verdict)",
    )
    parser.add_argument(
        "--verdict",
        action="store_true",
        help=(
            "print the per-field discrimination statistics and pass/fail "
            "decision (stderr when the samples are CSV on stdout)"
        ),
    )
    parser.add_argument(
        "--sysfs-root",
        type=_path_argument,
        default=_path_argument(os.environ.get(SYSFS_ROOT_ENV, str(DEFAULT_SYSFS_ROOT))),
        metavar="PATH",
        help=(f"sysfs root holding class/drm/card0/device (default: ${SYSFS_ROOT_ENV} or {DEFAULT_SYSFS_ROOT})"),
    )
    parser.add_argument(
        "--hwmon-dir",
        type=_path_argument,
        default=(_path_argument(os.environ[HWMON_DIR_ENV]) if os.environ.get(HWMON_DIR_ENV) else None),
        metavar="PATH",
        help=(f"directory holding power1_average (default: ${HWMON_DIR_ENV} or auto-discovery under the device dir)"),
    )
    return parser


def _csv_cell(value: object) -> object:
    return "" if value is None else value


def _json_payload(
    *,
    args: argparse.Namespace,
    paths: ProbePaths,
    rows: Sequence[dict[str, object]],
    verdict,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "gpu_metrics_probe.v1",
        "meta": {
            "note": TIME_FILTER_NOTE,
            "kernel": platform.release(),
            "columns": list(COLUMNS),
            "activity_fields": [column.column for column in ACTIVITY_COLUMNS],
            "capture": {
                "started_at": rows[0][TIMESTAMP_COLUMN] if rows else None,
                "finished_at": rows[-1][TIMESTAMP_COLUMN] if rows else None,
                "duration_seconds": args.duration,
                "interval_seconds": args.interval,
                "sample_count": len(rows),
            },
            "paths": {
                "metrics": str(paths.metrics),
                "power": str(paths.power),
                "busy": str(paths.busy),
                "hwmon_dir": str(paths.hwmon_dir),
            },
        },
        "samples": list(rows),
    }
    if verdict is not None:
        payload["verdict"] = verdict.to_dict()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        paths = resolve_paths(args.sysfs_root, args.hwmon_dir)
        probe = GpuMetricsProbe(paths)
        # Fail fast: a missing, unreadable or undecodable gpu_metrics must not
        # emit a header, a comment line or a partially written row.
        probe.preflight()
    except ProbeError as exc:
        print(f"gpu_metrics_probe: error: {exc}", file=sys.stderr)
        return EXIT_PROBE_ERROR

    csv_writer = None
    if not args.json:
        print(f"# {TIME_FILTER_NOTE}")
        csv_writer = csv.writer(sys.stdout)
        csv_writer.writerow(COLUMNS)

        def emit(row: dict[str, object]) -> None:
            csv_writer.writerow([_csv_cell(row[column]) for column in COLUMNS])
            sys.stdout.flush()

    else:
        emit = None  # JSON is a single document, written once the capture ends.

    try:
        rows = probe.run(args.duration, args.interval, on_row=emit)
    except ProbeError as exc:
        print(
            f"gpu_metrics_probe: error: {exc}; capture aborted after "
            f"{'no rows' if not csv_writer else 'the rows already written'}",
            file=sys.stderr,
        )
        return EXIT_PROBE_ERROR

    verdict = build_verdict(rows) if args.verdict else None

    if args.json:
        print(
            json.dumps(
                _json_payload(args=args, paths=paths, rows=rows, verdict=verdict),
                indent=2,
            )
        )
    elif verdict is not None:
        print(
            render_verdict(
                verdict,
                sample_count=len(rows),
                interval=args.interval,
                duration=args.duration,
            ),
            file=sys.stderr,
        )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
