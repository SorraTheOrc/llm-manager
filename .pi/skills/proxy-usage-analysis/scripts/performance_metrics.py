"""Performance-metric registry, rolling history state and verdict engine.

Ten metrics are the scorecard the daily usage report is held to (parent
LP-0MW0FU5Z6006A4P3). This module is the single source of truth for their
definitions, for the best-3 history behind each target and for the
direction-aware verdict a window earns against that target. Rendering lives in
``reporting.py``; this module never writes outside an explicit ``save_state``
call.

Registry
--------

``METRIC_REGISTRY`` is an ordered sequence of :class:`Metric` with ``id``,
``label``, ``unit``, ``direction`` (``"higher"`` or ``"lower"`` — which way
improvement points) and ``source`` (the log/telemetry input the metric is
measured from). Registry order is the report's row and detail-section order.

Retention, target and verdict rules
-----------------------------------

- **Direction** — ``Metric.direction`` decides which way improvement points:
  ``higher`` keeps the largest values, ``lower`` the smallest. Every rule
  below is direction-aware.
- **Retention** — :func:`retained` returns the best ``RETENTION_LIMIT`` (3)
  measurements, best first, at most one per window
  (:func:`_dedupe_by_window` keeps that window's better value), so recording
  the same window twice can never inflate a target.
- **Baseline** — :func:`baseline_from` is the single best retained value plus
  its measurement date. It is the report's context column, not the target.
- **Target** — :func:`target_from` is the mean of the retained best three, or
  ``None`` below three. With less evidence there is no target, and the report
  says ``Insufficient history`` rather than inventing one.
- **Verdict** — :func:`derive_verdict` compares this window's actual against
  that target (equality counts as met) and returns ``Met``/``Unmet``; fewer
  than three measurements always returns ``Insufficient history``, and an
  unmeasurable actual (``None``) returns ``n/a`` and is never recorded as
  ``0``.

State schema
------------

The state is a plain mapping, YAML round-trippable::

    {"metrics": {"<metric_id>": {"measurements": [measurement, ...]}}}

    measurement = {"window": str,   # window id, e.g. "2026-11-03"
                   "value": float,  # the measured value
                   "date": str,     # the measurement date
                   "fingerprint": str}  # configuration era, see below

Every registry metric has a ``measurements`` list, ordered by first recording.
All recorded measurements are kept (they are the evidence behind the target);
the best three are *derived* per direction by :func:`retained`. A repeat
recording for a window replaces that window's measurement, so a re-run never
double-counts history.

Two files exist, with different ownership:

- ``metrics/performance_metrics.yaml`` — committed defaults: empty history plus
  the per-metric detail copy (``why``, ``calculation``, ``levers``,
  ``trade_offs``, ``data_section_anchor``). Read-only at runtime.
- ``<output-dir>/performance_metrics_state.yaml`` — the live, mutable history
  written by :func:`save_state`.

Loading never writes either file: :func:`load_state` returns a fresh mapping
that starts from the defaults file's history and lets an existing state file
override it per metric.

Configuration era
-----------------

Measurements are not comparable across configuration changes, so each records
:func:`configuration_fingerprint` (slot count and per-slot context geometry).
:func:`is_cross_era` reports when a retained set spans more than one
fingerprint so callers can flag it; a measurement whose era is unknown (for
example a value backfilled from an archived report) carries an empty
fingerprint and is ignored by that check.

YAML access follows ``config_loader``'s guarded-import pattern, so PyYAML stays
optional and no new mandatory dependency is introduced.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import config_loader

HIGHER = "higher"
LOWER = "lower"

MET = "Met"
UNMET = "Unmet"
INSUFFICIENT = "Insufficient history"
NOT_AVAILABLE = "n/a"

#: Measurements retained behind a target (best 3, direction-aware).
RETENTION_LIMIT = 3

#: Fingerprint recorded when the configuration era is unknown (backfill).
UNKNOWN_FINGERPRINT = ""

#: Committed defaults: registry detail copy, never written at runtime.
DEFAULTS_PATH = Path(__file__).resolve().parents[1] / "metrics" / "performance_metrics.yaml"

#: Live, mutable history file name under the report output directory.
STATE_FILENAME = "performance_metrics_state.yaml"

#: Detail-copy fields every metric may declare in the committed defaults file.
DETAIL_FIELDS = (
    "why",
    "calculation",
    "levers",
    "trade_offs",
    "data_section_anchor",
)


@dataclass(frozen=True)
class Metric:
    """One scorecard metric: identity, presentation and its data source."""

    id: str
    label: str
    unit: str
    direction: str
    source: str

    @property
    def higher_is_better(self) -> bool:
        return self.direction == HIGHER


#: Ordered registry: this order is the report's table and details order.
METRIC_REGISTRY: tuple[Metric, ...] = (
    Metric(
        id="gpu_busy",
        label="GPU activity — power-based, any consumer",
        unit="%",
        direction=HIGHER,
        source="GPU telemetry sampler power samples (gpu:computing:bool, any GPU consumer)",
    ),
    Metric(
        id="fallback_rate",
        label="Fallback rate",
        unit="% of requests",
        direction=LOWER,
        source="Fallback triggered log events",
    ),
    Metric(
        id="decode_speed",
        label="Decode speed",
        unit="tok/s",
        direction=HIGHER,
        source="Stream finished token counts",
    ),
    Metric(
        id="prompt_eval_speed",
        label="Prompt eval speed",
        unit="tok/s",
        direction=HIGHER,
        source="llama-server prompt eval timing",
    ),
    Metric(
        id="time_to_first_token",
        label="Time to first token",
        unit="ms",
        direction=LOWER,
        source="dispatch_first_byte_ms and ttft_seconds log lines",
    ),
    Metric(
        id="stream_error_rate",
        label="Stream error rate",
        unit="%",
        direction=LOWER,
        source="Stream error and Stream finished reason=error lines",
    ),
    Metric(
        id="context_pressure",
        label="Context-size pressure",
        unit="%",
        direction=LOWER,
        source="routing_skip_local estimated_tokens lines",
    ),
    Metric(
        id="prompt_cache_hit_ratio",
        label="Prompt cache hit ratio",
        unit="ratio",
        direction=HIGHER,
        source="cached_ratio and cache_state log fields",
    ),
    Metric(
        id="contention_denials_per_hour",
        label="Contention denials per hour",
        unit="denials/h",
        direction=LOWER,
        source="local_dispatch_denied log events",
    ),
    Metric(
        id="prefill_eta_accuracy",
        label="Prefill ETA accuracy",
        unit="%",
        direction=HIGHER,
        source="local_admission_guard prefill ETA and dispatch first-byte pairs",
    ),
)

_METRICS_BY_ID = {metric.id: metric for metric in METRIC_REGISTRY}


def metric(metric_id: str) -> Metric | None:
    """Return the registered :class:`Metric` for *metric_id* (``None`` if unknown)."""
    return _METRICS_BY_ID.get(metric_id)


# ---------------------------------------------------------------------------
# Direction-aware retention and derived values
# ---------------------------------------------------------------------------


def _validate_direction(direction: str) -> str:
    if direction not in (HIGHER, LOWER):
        raise ValueError(f"direction must be {HIGHER!r} or {LOWER!r}, got {direction!r}")
    return direction


def _measurement_value(measurement: Mapping) -> float:
    return float(measurement.get("value"))


def _is_better(candidate: Mapping, current: Mapping, direction: str) -> bool:
    if direction == HIGHER:
        return _measurement_value(candidate) > _measurement_value(current)
    return _measurement_value(candidate) < _measurement_value(current)


def _dedupe_by_window(measurements: Iterable[Mapping], direction: str) -> list[Mapping]:
    """Keep one (the best) measurement per window, preserving first-seen order."""
    best: dict[str, Mapping] = {}
    order: list[str] = []
    for measurement in measurements:
        window = str(measurement.get("window"))
        if window not in best:
            best[window] = measurement
            order.append(window)
        elif _is_better(measurement, best[window], direction):
            best[window] = measurement
    return [best[window] for window in order]


def retained(
    measurements: Iterable[Mapping], direction: str, limit: int = RETENTION_LIMIT
) -> list[dict]:
    """Return the ``limit`` best measurements, best first, one per window.

    Direction-aware: ``higher`` keeps the largest values, ``lower`` the
    smallest. A window is never retained twice, so a repeated recording cannot
    inflate the target.
    """
    _validate_direction(direction)
    unique = _dedupe_by_window(measurements, direction)
    reverse = direction == HIGHER
    ordered = sorted(unique, key=_measurement_value, reverse=reverse)
    return [dict(measurement) for measurement in ordered[: max(limit, 0)]]


def target_from(measurements: Iterable[Mapping], direction: str) -> float | None:
    """Mean of the best three measurements, or ``None`` below three."""
    best = retained(measurements, direction)
    if len(best) < RETENTION_LIMIT:
        return None
    return sum(_measurement_value(measurement) for measurement in best) / len(best)


def baseline_from(measurements: Iterable[Mapping], direction: str) -> dict | None:
    """The single best value with its measurement date (``None`` when empty)."""
    best = retained(measurements, direction, limit=1)
    if not best:
        return None
    measurement = best[0]
    return {"value": _measurement_value(measurement), "date": measurement.get("date")}


def derive_verdict(
    actual: float | None, measurements: Iterable[Mapping], direction: str
) -> str:
    """Direction-aware verdict of *actual* against the best-3 target.

    Returns ``"n/a"`` when the window has no actual value, ``"Insufficient
    history"`` when fewer than three measurements are recorded (never a
    misleading Met/Unmet), and otherwise ``"Met"``/``"Unmet"``. Equality with
    the target counts as met.
    """
    if actual is None:
        return NOT_AVAILABLE
    _validate_direction(direction)
    target = target_from(measurements, direction)
    if target is None:
        return INSUFFICIENT
    actual_value = float(actual)
    met = actual_value >= target if direction == HIGHER else actual_value <= target
    return MET if met else UNMET


# ---------------------------------------------------------------------------
# Configuration fingerprint
# ---------------------------------------------------------------------------


def configuration_fingerprint(slot_count: int, per_slot_context: int) -> str:
    """Stable descriptor of the slot geometry a measurement was taken under."""
    return f"slots={int(slot_count)},ctx={int(per_slot_context)}"


def is_cross_era(measurements: Iterable[Mapping]) -> bool:
    """True when *measurements* span more than one known configuration era.

    Callers pass the retained best-3 set. Measurements with an unknown era
    (empty fingerprint, e.g. backfilled from archived reports) are ignored so a
    missing era never manufactures a false cross-era warning.
    """
    fingerprints = {str(measurement.get("fingerprint") or "") for measurement in measurements}
    fingerprints.discard(UNKNOWN_FINGERPRINT)
    return len(fingerprints) > 1


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def new_state() -> dict:
    """An empty state covering every registered metric."""
    return {"metrics": {metric.id: {"measurements": []} for metric in METRIC_REGISTRY}}


def _normalise_measurement(raw: Mapping) -> dict:
    """Validate and normalise one stored measurement (raises on malformed input)."""
    window = raw.get("window")
    if window is None:
        raise ValueError(f"measurement is missing its window: {raw!r}")
    try:
        value = float(raw.get("value"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"measurement has a non-numeric value: {raw!r}") from exc
    return {
        "window": str(window),
        "value": value,
        "date": str(raw.get("date") or window),
        "fingerprint": str(raw.get("fingerprint") or UNKNOWN_FINGERPRINT),
    }


def _history_entries(data: Mapping) -> dict[str, list[dict]]:
    """Return normalised ``{metric_id: measurements}`` for registered metrics."""
    metrics = data.get("metrics") if isinstance(data, Mapping) else None
    if not isinstance(metrics, Mapping):
        return {}
    entries: dict[str, list[dict]] = {}
    for metric_id, entry in metrics.items():
        if metric_id not in _METRICS_BY_ID or not isinstance(entry, Mapping):
            continue
        measurements = entry.get("measurements")
        if not isinstance(measurements, list):
            continue
        entries[metric_id] = [
            _normalise_measurement(measurement)
            for measurement in measurements
            if isinstance(measurement, Mapping)
        ]
    return entries


def load_state(path: str | Path | None, *, defaults_path: str | Path | None = None) -> dict:
    """Load the rolling history, never writing either file.

    History starts from *defaults_path* (the committed defaults) and is then
    overridden per metric by an existing *path* state file, so the live file
    always wins where it has something to say. Missing files simply leave the
    corresponding metrics empty.
    """
    state = new_state()
    if defaults_path is not None:
        for metric_id, measurements in _history_entries(
            config_loader.load_yaml_mapping(defaults_path)
        ).items():
            state["metrics"][metric_id]["measurements"] = measurements
    if path is not None and Path(path).is_file():
        for metric_id, measurements in _history_entries(
            config_loader.load_yaml_mapping(path)
        ).items():
            state["metrics"][metric_id]["measurements"] = measurements
    return state


def save_state(state: Mapping, path: str | Path) -> None:
    """Write the history to *path* (the only function that writes state)."""
    payload: dict = {"metrics": {}}
    metrics = state.get("metrics") if isinstance(state, Mapping) else None
    for registered in METRIC_REGISTRY:
        entry = metrics.get(registered.id) if isinstance(metrics, Mapping) else None
        raw = entry.get("measurements") if isinstance(entry, Mapping) else None
        measurements = [
            _normalise_measurement(measurement)
            for measurement in (raw or [])
            if isinstance(measurement, Mapping)
        ]
        payload["metrics"][registered.id] = {"measurements": measurements}
    config_loader.dump_yaml_mapping(payload, path)


def record_measurement(
    state: dict,
    metric_id: str,
    *,
    window: str,
    value: float,
    date: str | None = None,
    fingerprint: str | None = None,
) -> None:
    """Record one window's measurement in place (replacing that window's entry)."""
    if metric_id not in _METRICS_BY_ID:
        raise ValueError(f"unknown metric id {metric_id!r}")
    measurements = state["metrics"][metric_id]["measurements"]
    measurement = _normalise_measurement(
        {
            "window": window,
            "value": value,
            "date": date,
            "fingerprint": fingerprint,
        }
    )
    for index, existing in enumerate(measurements):
        if str(existing.get("window")) == measurement["window"]:
            measurements[index] = measurement
            return
    measurements.append(measurement)


def _has_window(state: Mapping, metric_id: str, window: str) -> bool:
    metrics = state.get("metrics") if isinstance(state, Mapping) else None
    entry = metrics.get(metric_id) if isinstance(metrics, Mapping) else None
    measurements = entry.get("measurements") if isinstance(entry, Mapping) else None
    return any(str(measurement.get("window")) == window for measurement in (measurements or []))


# ---------------------------------------------------------------------------
# Committed defaults and detail copy
# ---------------------------------------------------------------------------


def load_defaults(path: str | Path | None = None) -> dict:
    """Load the committed metric defaults (copy + optional seeded history)."""
    return config_loader.load_yaml_mapping(DEFAULTS_PATH if path is None else path)


def detail_copy(metric_id: str, defaults: Mapping | None = None) -> dict:
    """Return the per-metric detail copy declared in the committed defaults.

    Only the fields actually present are returned, so a renderer can render
    what the defaults provide without hard-coding prose. An unregistered metric
    id raises, matching :func:`record_measurement`.
    """
    if metric_id not in _METRICS_BY_ID:
        raise ValueError(f"unknown metric id {metric_id!r}")
    data = load_defaults() if defaults is None else defaults
    metrics = data.get("metrics") if isinstance(data, Mapping) else None
    entry = metrics.get(metric_id) if isinstance(metrics, Mapping) else None
    if not isinstance(entry, Mapping):
        return {}
    return {
        field: str(entry[field])
        for field in DETAIL_FIELDS
        if entry.get(field) is not None
    }


# ---------------------------------------------------------------------------
# Archived-report parsing and backfill
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_GENERATED_DATE_RE = re.compile(r"(?m)^-\s*Generated:\s*(\d{4}-\d{2}-\d{2})")
_FALLBACK_RATE_RE = re.compile(
    r"Fallback events:\s*\*\*[\d,]+\*\*\s*\(([\d.]+)% of requests\)"
)
_PERCENT_IN_TEXT_RE = re.compile(r"\(([\d.]+)%\)")


def _number(text: object) -> float | None:
    """First numeric token in *text* as a float (``None`` when there is none)."""
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    match = _NUMBER_RE.search(str(text).replace(",", ""))
    return float(match.group(0)) if match else None


def _section(report_text: str, title: str) -> str | None:
    """Body of ``## <title>`` up to the next ``## `` heading (``None`` if absent)."""
    marker = f"## {title}"
    start = report_text.find(marker)
    if start < 0:
        return None
    body = report_text[start + len(marker) :]
    end = body.find("\n## ")
    return body if end < 0 else body[:end]


def _table_rows(section: str) -> list[list[str]]:
    rows = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        rows.append([cell.strip() for cell in stripped.strip("|").split("|")])
    return rows


def _column_index(header: Sequence[str], needle: str) -> int | None:
    for index, cell in enumerate(header):
        if needle in cell.lower():
            return index
    return None


def _row_for_bucket(rows: Sequence[Sequence[str]], bucket: str) -> Sequence[str] | None:
    """The table's all-buckets row, located by header rather than position."""
    if not rows:
        return None
    header = rows[0]
    bucket_column = _column_index(header, "bucket")
    if bucket_column is None:
        bucket_column = 0
    for row in rows[1:]:
        if bucket_column < len(row) and row[bucket_column].strip().lower() == bucket:
            return row
    return None


def _parse_fallback_rate(report_text: str) -> float | None:
    match = _FALLBACK_RATE_RE.search(report_text)
    return float(match.group(1)) if match else None


def _parse_speed(report_text: str, section_title: str) -> float | None:
    section = _section(report_text, section_title)
    if section is None:
        return None
    rows = _table_rows(section)
    if not rows:
        return None
    median_column = _column_index(rows[0], "median")
    row = _row_for_bucket(rows, "total")
    if median_column is None or row is None or median_column >= len(row):
        return None
    return _number(row[median_column])


def _parse_time_to_first_token(report_text: str) -> float | None:
    section = _section(report_text, "Time to first token")
    if section is None:
        return None
    rows = _table_rows(section)
    if not rows:
        return None
    p90_column = _column_index(rows[0], "p90")
    row = _row_for_bucket(rows, "total")
    if p90_column is None or row is None or p90_column >= len(row):
        return None
    return _number(row[p90_column])


def _parse_gpu_busy(report_text: str) -> float | None:
    section = _section(report_text, "Local model utilization")
    if section is None:
        return None
    for row in _table_rows(section):
        if not row or not row[0].strip().lower().startswith("gpu busy"):
            continue
        for cell in row[1:]:
            percent = _PERCENT_IN_TEXT_RE.search(cell)
            if percent:
                return float(percent.group(1))
        return None
    return None


def _parse_denials_per_hour(report_text: str) -> float | None:
    section = _section(report_text, "Local contention prediction vs reality")
    if section is None:
        return None
    for row in _table_rows(section):
        if row and row[0].strip().lower() == "denials per hour" and len(row) > 1:
            return _number(row[1])
    return None


#: Metric id -> archived-report parser. Only metrics whose value is rendered
#: directly in the archived report are parsed; a metric whose rendering is
#: absent (or changed era) is skipped rather than guessed.
_ARCHIVE_PARSERS = (
    ("fallback_rate", _parse_fallback_rate),
    ("decode_speed", lambda text: _parse_speed(text, "Decode speed")),
    ("prompt_eval_speed", lambda text: _parse_speed(text, "Prompt eval speed")),
    ("time_to_first_token", _parse_time_to_first_token),
    ("gpu_busy", _parse_gpu_busy),
    ("contention_denials_per_hour", _parse_denials_per_hour),
)


def archived_window_id(report_text: str, fallback: str) -> str:
    """Window id for an archived report: its ``Generated`` date, else *fallback*.

    The archived daily report's header records when it was generated; its date
    names the window the report covers, so run-to-run directory suffixes
    (``2026-08-07_2``) collapse onto one window and a re-run can never record
    the same day twice. A report whose ``Generated`` line is missing or
    unparseable falls back to *fallback* (the archive directory name) rather
    than guessing a date.
    """
    if isinstance(report_text, str):
        match = _GENERATED_DATE_RE.search(report_text)
        if match:
            return match.group(1)
    return str(fallback)


def parse_archived_report(report_text: str) -> dict[str, float]:
    """Extract the metric values an archived ``report.md`` renders reliably.

    Metrics whose figure is missing from that report era are omitted, never
    estimated: the caller seeds only what was actually rendered.
    """
    if not isinstance(report_text, str) or not report_text.strip():
        return {}
    parsed: dict[str, float] = {}
    for metric_id, parser in _ARCHIVE_PARSERS:
        value = parser(report_text)
        if value is not None:
            parsed[metric_id] = float(value)
    return parsed


def seed_history(state: dict, reports: Iterable[tuple[str, str]]) -> dict:
    """Seed the rolling history from archived reports (idempotent per window).

    *reports* is an iterable of ``(window_id, report_text)``. A metric already
    recorded for a window is left untouched, so re-running the backfill never
    duplicates history. Backfilled measurements carry the unknown-era
    fingerprint because an archived report does not record its configuration.
    Returns ``{"seeded", "already_present", "skipped_metrics"}`` where
    ``skipped_metrics`` maps each metric that no report could provide to the
    reason it was skipped.
    """
    seeded = 0
    already_present = 0
    reasons: dict[str, str] = {}
    provided: set[str] = set()
    for window_id, report_text in reports:
        window = str(window_id)
        parsed = parse_archived_report(report_text)
        for registered in METRIC_REGISTRY:
            if registered.id not in parsed:
                reasons.setdefault(
                    registered.id, "not rendered in the archived report"
                )
                continue
            provided.add(registered.id)
            if _has_window(state, registered.id, window):
                already_present += 1
                continue
            record_measurement(
                state,
                registered.id,
                window=window,
                value=parsed[registered.id],
                date=window,
                fingerprint=UNKNOWN_FINGERPRINT,
            )
            seeded += 1
    skipped = {
        registered.id: reasons[registered.id]
        for registered in METRIC_REGISTRY
        if registered.id not in provided and registered.id in reasons
    }
    return {
        "seeded": seeded,
        "already_present": already_present,
        "skipped_metrics": skipped,
    }
