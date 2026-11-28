#!/usr/bin/env python3
"""Discrimination statistics for candidate ``gpu_metrics`` activity fields.

Why this exists (LP-0MVZEVAVT009TPT1, parent LP-0MVSQT7YC0045HFW)
-----------------------------------------------------------------
The usage report cannot tell GFX work from any other power draw: its busy figure
is a *power proxy* (``gpu:computing:bool >= 90``), ``gpu_busy_percent`` was
rejected for saturating (45.2 % false-busy, r = 0.28 against power), and the
last candidate — the ``amd_gpu_metrics_v3_0`` activity blob — must be judged
with numbers rather than eyeballing. This module is the arithmetic half of that
judgement; the decoder and the paired sampler live in ``scripts/gpu_metrics_
probe.py`` (LP-0MVZEW6UL0054D8W) and the v3.0 layout in LP-0MVZEW6K6007BPY0.

The module is deliberately **format-agnostic**: it takes a sequence of sample
dicts carrying ``power_watts`` plus one numeric value per candidate activity
field, and knows nothing about sysfs, struct revisions or file layouts. That
keeps the bar testable against synthetic series.

The bar (all three, recorded here so a producer can tighten or relax it once)
------------------------------------------------------------------------------
A field passes only when:

* **separation** — the compute-band mean exceeds the idle-band mean by at least
  :data:`PASS_SEPARATION_PP` percentage points;
* **correlation** — Pearson r against package power is at least
  :data:`PASS_PEARSON_R`;
* **idle false-busy** — at most :data:`PASS_MAX_IDLE_FALSE_BUSY_SHARE` of
  idle-band samples read at or above :data:`FALSE_BUSY_ACTIVITY_PCT` % activity.

Band definitions
----------------
* **idle** — ``power_watts < 60`` (:data:`IDLE_POWER_CEILING_W`, strict);
* **compute** — ``power_watts >= 90`` (:data:`COMPUTE_POWER_FLOOR_W`,
  inclusive);
* **middle** — ``60 <= power_watts < 90``. The middle band is excluded from the
  band means (it is deliberately neither idle nor compute) but every usable
  sample, middle included, contributes to Pearson r.

Exactly 60 W is therefore *middle*, exactly 90 W is *compute*, and activity of
exactly 90 % in the idle band counts as false-busy.

One-sided captures
------------------
A capture that produced only idle (or only compute) samples cannot discriminate
and must not emit a verdict that looks like evidence. Such a field is reported
as ``insufficient_data`` with the available band mean, null separation and null
correlation, and :attr:`Verdict.status` is ``insufficient_data``.

Pure standard library: Pearson r is implemented directly, so this module adds no
``numpy`` dependency (the constraint recorded on the work item).

Example
-------
>>> samples = [
...     {"power_watts": 40.0, "average_gfx_activity": 5.0},
...     {"power_watts": 100.0, "average_gfx_activity": 95.0},
... ]
>>> verdict = compute_verdict(samples)
>>> verdict.fields["average_gfx_activity"].decision
'pass'
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

# --- Band definitions -------------------------------------------------------

IDLE_POWER_CEILING_W = 60.0
"""Idle band is strictly below this package power."""

COMPUTE_POWER_FLOOR_W = 90.0
"""Compute band is at or above this package power."""

FALSE_BUSY_ACTIVITY_PCT = 90.0
"""Idle samples reading at or above this activity are false-busy."""

# --- The pass bar -----------------------------------------------------------

PASS_SEPARATION_PP = 30.0
"""Minimum compute-minus-idle band mean separation, in percentage points."""

PASS_PEARSON_R = 0.6
"""Minimum Pearson correlation against package power."""

PASS_MAX_IDLE_FALSE_BUSY_SHARE = 0.10
"""Maximum share of idle-band samples that may read busy."""

# --- Vocabulary -------------------------------------------------------------

BAND_IDLE = "idle"
BAND_MIDDLE = "middle"
BAND_COMPUTE = "compute"

DECISION_PASS = "pass"
DECISION_FAIL = "fail"
DECISION_INSUFFICIENT = "insufficient_data"

STATUS_OK = "ok"

POWER_KEY = "power_watts"
"""The one key a sample must always carry; it defines the bands."""

NON_ACTIVITY_KEYS = frozenset(
    {
        POWER_KEY,
        "timestamp",
        "time",
        "t",
        "captured_at",
        "sample_index",
        "index",
        "seq",
        "elapsed_seconds",
        # The probe records hwmon power under both names; neither is a candidate
        # activity field, and treating power as one would be circular.
        "power1_average_watts",
    }
)
"""Keys excluded from default candidate-field inference.

Every other numeric key is treated as a candidate activity field. That
deliberately includes ``gpu_busy_percent`` when a capture carries it, so the
verdict shows the rejected counter failing the same bar next to the
``gpu_metrics`` fields.
"""


def _is_number(value: object) -> bool:
    """True for real numbers, excluding ``bool`` (an ``int`` subclass)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def classify_band(power_watts: float) -> str:
    """Return ``"idle"``, ``"middle"`` or ``"compute"`` for a package power."""
    if power_watts < IDLE_POWER_CEILING_W:
        return BAND_IDLE
    if power_watts >= COMPUTE_POWER_FLOOR_W:
        return BAND_COMPUTE
    return BAND_MIDDLE


def pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation coefficient, or ``None`` when it is undefined.

    Undefined means fewer than two paired points or zero variance in either
    series — the saturated and flat series the bar exists to reject. A
    correlation that is numerically undefined must never be read as "no
    correlation" *or* as a pass, so callers receive ``None`` explicitly.
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    covariance = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    variance_x = sum((x - mx) ** 2 for x in xs)
    variance_y = sum((y - my) ** 2 for y in ys)
    if variance_x <= 0.0 or variance_y <= 0.0:
        return None
    denominator = math.sqrt(variance_x * variance_y)
    if denominator == 0.0:  # pragma: no cover - guard against underflow
        return None
    coefficient = covariance / denominator
    # Clamp tiny floating-point overshoot, e.g. 1.0000000000000002.
    return max(-1.0, min(1.0, coefficient))


def passes_bar(
    separation_pp: float | None,
    pearson_r: float | None,
    idle_false_busy_share: float | None,
) -> bool:
    """Apply the three-part bar. ``None`` (undefined) always fails.

    The comparisons are inclusive at the bar: separation ``>=`` 30 pp,
    ``r >=`` 0.6 and false-busy share ``<=`` 0.10 all pass.
    """
    if separation_pp is None or pearson_r is None or idle_false_busy_share is None:
        return False
    return (
        separation_pp >= PASS_SEPARATION_PP
        and pearson_r >= PASS_PEARSON_R
        and idle_false_busy_share <= PASS_MAX_IDLE_FALSE_BUSY_SHARE
    )


@dataclass(frozen=True)
class BandCounts:
    """How many samples fell into each power band."""

    idle: int
    compute: int
    middle: int
    invalid: int
    total: int

    def to_dict(self) -> dict[str, int]:
        return {
            "idle": self.idle,
            "compute": self.compute,
            "middle": self.middle,
            "invalid": self.invalid,
            "total": self.total,
        }


@dataclass(frozen=True)
class FieldVerdict:
    """The statistics and explicit decision for one candidate field."""

    field: str
    idle_band_mean: float | None
    compute_band_mean: float | None
    separation_pp: float | None
    pearson_r: float | None
    idle_false_busy_share: float | None
    idle_band_count: int
    compute_band_count: int
    valid_count: int
    decision: str
    reasons: tuple[str, ...] = ()

    @property
    def passes(self) -> bool:
        return self.decision == DECISION_PASS

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "idle_band_mean": _round(self.idle_band_mean),
            "compute_band_mean": _round(self.compute_band_mean),
            "separation_pp": _round(self.separation_pp),
            "pearson_r": _round(self.pearson_r),
            "idle_false_busy_share": _round(self.idle_false_busy_share),
            "idle_false_busy_pct": _round(
                None
                if self.idle_false_busy_share is None
                else self.idle_false_busy_share * 100.0
            ),
            "idle_band_count": self.idle_band_count,
            "compute_band_count": self.compute_band_count,
            "valid_count": self.valid_count,
            "decision": self.decision,
            "passes": self.passes,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class Verdict:
    """Per-field statistics plus the capture's overall usability status."""

    fields: Mapping[str, FieldVerdict]
    band_counts: BandCounts

    @property
    def status(self) -> str:
        """``"insufficient_data"`` when the capture misses a power band."""
        if self.band_counts.idle == 0 or self.band_counts.compute == 0:
            return DECISION_INSUFFICIENT
        return STATUS_OK

    @property
    def passed_fields(self) -> tuple[str, ...]:
        """Names of fields that cleared the bar, in evaluation order."""
        return tuple(name for name, stat in self.fields.items() if stat.passes)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "passed_fields": list(self.passed_fields),
            "bar": {
                "min_separation_pp": PASS_SEPARATION_PP,
                "min_pearson_r": PASS_PEARSON_R,
                "max_idle_false_busy_share": PASS_MAX_IDLE_FALSE_BUSY_SHARE,
                "idle_power_ceiling_w": IDLE_POWER_CEILING_W,
                "compute_power_floor_w": COMPUTE_POWER_FLOOR_W,
                "false_busy_activity_pct": FALSE_BUSY_ACTIVITY_PCT,
            },
            "band_counts": self.band_counts.to_dict(),
            "fields": {name: stat.to_dict() for name, stat in self.fields.items()},
        }


def _infer_fields(samples: Sequence[Mapping[str, object]]) -> list[str]:
    """Candidate fields are every numeric key except power and known metadata.

    Sorted so a caller that does not name the fields gets stable output.
    """
    discovered: set[str] = set()
    for sample in samples:
        for key, value in sample.items():
            if key in NON_ACTIVITY_KEYS:
                continue
            if _is_number(value):
                discovered.add(str(key))
    return sorted(discovered)


def _band_counts(samples: Sequence[Mapping[str, object]]) -> BandCounts:
    idle = compute = middle = invalid = 0
    for sample in samples:
        power = sample.get(POWER_KEY)
        if not _is_number(power):
            invalid += 1
            continue
        band = classify_band(float(power))  # type: ignore[arg-type]
        if band == BAND_IDLE:
            idle += 1
        elif band == BAND_COMPUTE:
            compute += 1
        else:
            middle += 1
    return BandCounts(
        idle=idle,
        compute=compute,
        middle=middle,
        invalid=invalid,
        total=len(samples),
    )


def _evaluate_field(
    samples: Sequence[Mapping[str, object]], field: str
) -> FieldVerdict:
    idle_activities: list[float] = []
    compute_activities: list[float] = []
    powers: list[float] = []
    activities: list[float] = []

    for sample in samples:
        power = sample.get(POWER_KEY)
        activity = sample.get(field)
        if not _is_number(power) or not _is_number(activity):
            continue
        power_f = float(power)  # type: ignore[arg-type]
        activity_f = float(activity)  # type: ignore[arg-type]
        powers.append(power_f)
        activities.append(activity_f)
        band = classify_band(power_f)
        if band == BAND_IDLE:
            idle_activities.append(activity_f)
        elif band == BAND_COMPUTE:
            compute_activities.append(activity_f)

    idle_mean = _mean(idle_activities)
    compute_mean = _mean(compute_activities)

    # Separation needs both band means; a one-sided capture has no separation to
    # report, so it stays null rather than defaulting to zero.
    if idle_mean is not None and compute_mean is not None:
        separation: float | None = compute_mean - idle_mean
    else:
        separation = None

    if idle_activities:
        busy = sum(
            1 for value in idle_activities if value >= FALSE_BUSY_ACTIVITY_PCT
        )
        false_busy: float | None = busy / len(idle_activities)
    else:
        false_busy = None

    # Pearson r is only meaningful when both bands are represented. Computing it
    # over a single band would produce a correlation that looks like evidence for
    # a discrimination that was never captured.
    if idle_activities and compute_activities:
        correlation = pearson_r(powers, activities)
    else:
        correlation = None

    reasons: list[str] = []
    if not idle_activities and not compute_activities:
        decision = DECISION_INSUFFICIENT
        reasons.append("no usable samples carrying power and this field")
    elif not idle_activities:
        decision = DECISION_INSUFFICIENT
        reasons.append(
            f"no idle-band samples (power < {IDLE_POWER_CEILING_W:.0f} W)"
        )
    elif not compute_activities:
        decision = DECISION_INSUFFICIENT
        reasons.append(
            f"no compute-band samples (power >= {COMPUTE_POWER_FLOOR_W:.0f} W)"
        )
    elif passes_bar(separation, correlation, false_busy):
        decision = DECISION_PASS
    else:
        decision = DECISION_FAIL
        if separation is None or separation < PASS_SEPARATION_PP:
            rendered = "n/a" if separation is None else f"{separation:.2f}"
            reasons.append(
                f"separation {rendered} pp < {PASS_SEPARATION_PP:.0f} pp"
            )
        if correlation is None:
            reasons.append("pearson r undefined (zero variance)")
        elif correlation < PASS_PEARSON_R:
            reasons.append(
                f"pearson r {correlation:.3f} < {PASS_PEARSON_R:.1f}"
            )
        if false_busy is None or false_busy > PASS_MAX_IDLE_FALSE_BUSY_SHARE:
            rendered = "n/a" if false_busy is None else f"{false_busy * 100.0:.1f}%"
            reasons.append(
                f"idle false-busy {rendered} > "
                f"{PASS_MAX_IDLE_FALSE_BUSY_SHARE * 100.0:.0f}%"
            )

    return FieldVerdict(
        field=field,
        idle_band_mean=idle_mean,
        compute_band_mean=compute_mean,
        separation_pp=separation,
        pearson_r=correlation,
        idle_false_busy_share=false_busy,
        idle_band_count=len(idle_activities),
        compute_band_count=len(compute_activities),
        valid_count=len(powers),
        decision=decision,
        reasons=tuple(reasons),
    )


def compute_verdict(
    samples: Iterable[Mapping[str, object]],
    fields: Iterable[str] | None = None,
) -> Verdict:
    """Compute per-field discrimination statistics against the pass bar.

    Parameters
    ----------
    samples:
        Sequence of mappings. Each must carry ``power_watts``; each candidate
        activity field is read from the same mapping. Samples missing either a
        numeric power or a numeric field value are skipped for that field, so a
        sparse capture degrades to a smaller denominator rather than an error.
    fields:
        Candidate activity field names to evaluate. When omitted, every numeric
        key except :data:`NON_ACTIVITY_KEYS` is evaluated (sorted), which
        includes ``gpu_busy_percent`` if the capture carries it — useful as the
        rejected-counter baseline.

    Returns
    -------
    Verdict
        ``verdict.fields[name]`` holds the band means, separation in percentage
        points, Pearson r, idle false-busy share, per-band counts and an
        explicit ``"pass"`` / ``"fail"`` / ``"insufficient_data"`` decision.
        ``verdict.to_dict()`` is JSON-serialisable and repeats the bar so a
        recorded verdict is reproducible from its output alone.
    """
    sample_list = list(samples)
    if fields is None:
        field_names = _infer_fields(sample_list)
    else:
        # Preserve the caller's order but drop duplicates.
        field_names = list(dict.fromkeys(str(name) for name in fields))

    verdicts = {
        name: _evaluate_field(sample_list, name) for name in field_names
    }
    return Verdict(fields=verdicts, band_counts=_band_counts(sample_list))
