#!/usr/bin/env python3
"""Tests for the ``gpu_metrics`` verdict statistics engine.

``scripts/lib/gpu_metrics_verdict.py`` answers one question with numbers: does a
candidate ``gpu_metrics`` activity field separate compute from idle better than
the already-rejected ``gpu_busy_percent``? It is the arithmetic half of
LP-0MVZEVAVT009TPT1 (parent LP-0MVSQT7YC0045HFW), deliberately free of sysfs or
file-format knowledge so the probe CLI can feed it decoded samples.

The bar (AC2) is all three of:

* separation of the compute-band mean over the idle-band mean >= 30 pp,
* Pearson r against package power >= 0.6,
* idle-band false-busy share (activity >= 90 % while power < 60 W) <= 10 %.

Coverage:

* synthetic saturated (all 100 %), graded, flat-low and noisy series with exact
  pass/fail decisions and stats (AC3),
* band boundaries at exactly 60 W / 90 W, the activity-90 % false-busy edge and
  the separation-30 pp / r-0.6 pass edges (AC4),
* one-sided captures reporting ``insufficient_data`` rather than a verdict (AC5).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lib import gpu_metrics_verdict as gv  # noqa: E402
from lib.gpu_metrics_verdict import (  # noqa: E402
    COMPUTE_POWER_FLOOR_W,
    IDLE_POWER_CEILING_W,
    PASS_MAX_IDLE_FALSE_BUSY_SHARE,
    PASS_PEARSON_R,
    PASS_SEPARATION_PP,
    classify_band,
    compute_verdict,
    passes_bar,
    pearson_r,
)

FIELD = "average_gfx_activity"
OTHER_FIELD = "average_vcn_activity"


def _samples(pairs: list[tuple[float, float]], field: str = FIELD) -> list[dict]:
    """Build samples carrying ``power_watts`` and one candidate activity field."""
    return [{"power_watts": power, field: activity} for power, activity in pairs]


def _stats(verdict, field: str = FIELD):
    return verdict.fields[field]


# ---------------------------------------------------------------------------
# Pearson r (implemented directly, stdlib only)
# ---------------------------------------------------------------------------


def test_pearson_r_perfect_positive_correlation():
    assert pearson_r([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


def test_pearson_r_perfect_negative_correlation():
    assert pearson_r([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == pytest.approx(-1.0)


def test_pearson_r_zero_variance_is_undefined():
    assert pearson_r([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]) is None
    assert pearson_r([1.0, 2.0], [5.0, 5.0]) is None


def test_pearson_r_needs_two_points():
    assert pearson_r([1.0], [2.0]) is None
    assert pearson_r([], []) is None


# ---------------------------------------------------------------------------
# Band assignment boundaries (AC4)
# ---------------------------------------------------------------------------


def test_idle_ceiling_is_strict_and_compute_floor_is_inclusive():
    assert IDLE_POWER_CEILING_W == 60.0
    assert COMPUTE_POWER_FLOOR_W == 90.0
    assert classify_band(59.999) == gv.BAND_IDLE
    # Exactly 60 W is neither idle (strict <) nor compute (>= 90): middle band.
    assert classify_band(60.0) == gv.BAND_MIDDLE
    assert classify_band(89.999) == gv.BAND_MIDDLE
    # Exactly 90 W is compute (inclusive >=).
    assert classify_band(90.0) == gv.BAND_COMPUTE
    assert classify_band(120.0) == gv.BAND_COMPUTE


def test_band_counts_assign_60w_to_middle_and_90w_to_compute():
    verdict = compute_verdict(
        _samples([(60.0, 50.0), (90.0, 100.0), (40.0, 5.0), (100.0, 95.0)])
    )
    assert verdict.band_counts.idle == 1  # 40 W
    assert verdict.band_counts.middle == 1  # 60 W
    assert verdict.band_counts.compute == 2  # 90 W and 100 W
    assert verdict.band_counts.total == 4
    # 60 W is excluded from both band means.
    assert _stats(verdict).idle_band_mean == pytest.approx(5.0)
    assert _stats(verdict).compute_band_mean == pytest.approx(97.5)


# ---------------------------------------------------------------------------
# Synthetic series (AC3)
# ---------------------------------------------------------------------------


def test_saturated_series_fails():
    """All-100 % activity cannot separate anything: zero pp, r undefined, 100 % false-busy."""
    samples = _samples(
        [(40.0, 100.0), (50.0, 100.0), (55.0, 100.0),
         (95.0, 100.0), (100.0, 100.0), (110.0, 100.0), (120.0, 100.0)]
    )
    stats = _stats(compute_verdict(samples))
    assert stats.idle_band_mean == pytest.approx(100.0)
    assert stats.compute_band_mean == pytest.approx(100.0)
    assert stats.separation_pp == pytest.approx(0.0)
    assert stats.pearson_r is None  # zero activity variance
    assert stats.idle_false_busy_share == pytest.approx(1.0)
    assert stats.idle_band_count == 3
    assert stats.compute_band_count == 4
    assert stats.decision == "fail"
    assert stats.passes is False


def test_graded_series_passes():
    """Low activity in idle, high in compute: separation and r clear the bar."""
    samples = _samples(
        [(40.0, 5.0), (45.0, 6.0), (50.0, 7.0), (55.0, 8.0),
         (95.0, 92.0), (100.0, 94.0), (110.0, 96.0), (120.0, 98.0)]
    )
    verdict = compute_verdict(samples)
    stats = _stats(verdict)
    assert stats.idle_band_mean == pytest.approx(6.5)
    assert stats.compute_band_mean == pytest.approx(95.0)
    assert stats.separation_pp == pytest.approx(88.5)
    assert stats.pearson_r == pytest.approx(0.9755, abs=1e-3)
    assert stats.idle_false_busy_share == pytest.approx(0.0)
    assert stats.idle_band_count == 4
    assert stats.compute_band_count == 4
    assert stats.decision == "pass"
    assert stats.passes is True
    assert verdict.passed_fields == (FIELD,)


def test_flat_low_series_fails():
    """A field that never moves cannot discriminate, even though it is never false-busy."""
    samples = _samples(
        [(40.0, 5.0), (50.0, 5.0), (55.0, 5.0),
         (95.0, 5.0), (100.0, 5.0), (110.0, 5.0), (120.0, 5.0)]
    )
    stats = _stats(compute_verdict(samples))
    assert stats.separation_pp == pytest.approx(0.0)
    assert stats.pearson_r is None
    assert stats.idle_false_busy_share == pytest.approx(0.0)
    assert stats.decision == "fail"


def test_noisy_series_fails():
    """Uncorrelated, idle-saturated activity fails on separation and false-busy."""
    samples = _samples(
        [(40.0, 95.0), (45.0, 20.0), (50.0, 94.0), (55.0, 10.0),
         (95.0, 30.0), (100.0, 85.0), (110.0, 40.0), (120.0, 90.0)]
    )
    stats = _stats(compute_verdict(samples))
    assert stats.idle_band_mean == pytest.approx(54.75)
    assert stats.compute_band_mean == pytest.approx(61.25)
    assert stats.separation_pp == pytest.approx(6.5)
    assert stats.idle_false_busy_share == pytest.approx(0.5)
    assert stats.pearson_r is not None
    assert stats.pearson_r < PASS_PEARSON_R
    assert stats.decision == "fail"


# ---------------------------------------------------------------------------
# Pass-bar edges (AC4)
# ---------------------------------------------------------------------------


def test_separation_exactly_30pp_passes_the_edge():
    """A hand-built series with separation exactly 30.0 pp and r = 1.0 passes.

    Activity is the affine map ``0.6 * power - 19``: idle powers 40/50 W give
    5/11 %, compute powers 90/100 W give 35/41 %, so the band means differ by
    exactly 30.0 pp and the correlation is exactly 1.0.
    """
    samples = _samples(
        [(40.0, 5.0), (50.0, 11.0), (90.0, 35.0), (100.0, 41.0)]
    )
    stats = _stats(compute_verdict(samples))
    assert stats.separation_pp == pytest.approx(30.0)
    assert stats.separation_pp >= PASS_SEPARATION_PP
    assert stats.pearson_r == pytest.approx(1.0)
    assert stats.decision == "pass"


def test_passes_bar_edges():
    """Exact bar values pass; anything below (or above the false-busy cap) fails."""
    assert PASS_SEPARATION_PP == 30.0
    assert PASS_PEARSON_R == 0.6
    assert PASS_MAX_IDLE_FALSE_BUSY_SHARE == 0.10

    assert passes_bar(30.0, 0.6, 0.10) is True
    assert passes_bar(29.999, 0.6, 0.10) is False
    assert passes_bar(30.0, 0.5999, 0.10) is False
    assert passes_bar(30.0, 0.6, 0.1001) is False
    assert passes_bar(80.0, 0.9, 0.0) is True


def test_activity_exactly_90_in_idle_is_false_busy():
    """The false-busy count is inclusive at 90 % activity.

    Ten idle samples with exactly one at 90 % gives a 0.10 share, which passes;
    a second one gives 0.20 and fails. The compute band is clean in both runs so
    only the false-busy term changes.
    """
    idle_base = [10.0] * 9
    compute = [(100.0, 100.0)] * 10

    one_busy = [(50.0, a) for a in [90.0] + idle_base] + compute
    one_stats = _stats(compute_verdict(_samples(one_busy)))
    assert one_stats.idle_false_busy_share == pytest.approx(0.10)
    assert one_stats.decision == "pass"

    two_busy = [(50.0, a) for a in [90.0, 90.0] + [10.0] * 8] + compute
    two_stats = _stats(compute_verdict(_samples(two_busy)))
    assert two_stats.idle_false_busy_share == pytest.approx(0.20)
    assert two_stats.decision == "fail"


# ---------------------------------------------------------------------------
# One-sided captures (AC5)
# ---------------------------------------------------------------------------


def test_idle_only_capture_reports_insufficient_data():
    samples = _samples([(40.0, 5.0), (45.0, 6.0), (50.0, 7.0), (55.0, 8.0)])
    verdict = compute_verdict(samples)
    stats = _stats(verdict)
    assert verdict.status == "insufficient_data"
    assert stats.idle_band_count == 4
    assert stats.compute_band_count == 0
    assert stats.compute_band_mean is None
    assert stats.separation_pp is None
    assert stats.pearson_r is None
    assert stats.decision == "insufficient_data"
    assert stats.passes is False
    assert verdict.passed_fields == ()


def test_compute_only_capture_reports_insufficient_data():
    samples = _samples([(95.0, 95.0), (100.0, 96.0), (110.0, 97.0), (120.0, 98.0)])
    verdict = compute_verdict(samples)
    stats = _stats(verdict)
    assert verdict.status == "insufficient_data"
    assert stats.idle_band_count == 0
    assert stats.compute_band_count == 4
    assert stats.idle_band_mean is None
    assert stats.separation_pp is None
    assert stats.decision == "insufficient_data"


def test_middle_band_only_capture_reports_insufficient_data():
    samples = _samples([(70.0, 50.0), (75.0, 55.0), (80.0, 60.0)])
    verdict = compute_verdict(samples)
    assert verdict.status == "insufficient_data"
    assert _stats(verdict).decision == "insufficient_data"


# ---------------------------------------------------------------------------
# Multi-field evaluation and rendering
# ---------------------------------------------------------------------------


def test_multiple_fields_are_decided_independently():
    samples = []
    for power, gfx, vcn in [
        (40.0, 100.0, 4.0),
        (50.0, 100.0, 5.0),
        (55.0, 100.0, 6.0),
        (95.0, 100.0, 94.0),
        (100.0, 100.0, 96.0),
        (110.0, 100.0, 97.0),
        (120.0, 100.0, 98.0),
    ]:
        samples.append({"power_watts": power, FIELD: gfx, OTHER_FIELD: vcn})

    verdict = compute_verdict(samples)
    assert set(verdict.fields) == {FIELD, OTHER_FIELD}
    assert verdict.fields[FIELD].decision == "fail"
    assert verdict.fields[OTHER_FIELD].decision == "pass"
    assert verdict.passed_fields == (OTHER_FIELD,)
    assert verdict.status == "ok"


def test_explicit_fields_restricts_evaluation():
    samples = [
        {"power_watts": 40.0, FIELD: 5.0, OTHER_FIELD: 100.0},
        {"power_watts": 50.0, FIELD: 6.0, OTHER_FIELD: 100.0},
        {"power_watts": 100.0, FIELD: 95.0, OTHER_FIELD: 100.0},
        {"power_watts": 110.0, FIELD: 96.0, OTHER_FIELD: 100.0},
    ]
    verdict = compute_verdict(samples, fields=[FIELD])
    assert list(verdict.fields) == [FIELD]
    assert verdict.fields[FIELD].decision == "pass"


def test_to_dict_is_json_serialisable_and_carries_the_bar():
    samples = _samples(
        [(40.0, 5.0), (50.0, 7.0), (100.0, 95.0), (110.0, 97.0)]
    )
    payload = compute_verdict(samples).to_dict()
    # Must round-trip through json without a custom encoder.
    rendered = json.dumps(payload)
    assert FIELD in rendered
    field_payload = payload["fields"][FIELD]
    for key in (
        "idle_band_mean",
        "compute_band_mean",
        "separation_pp",
        "pearson_r",
        "idle_false_busy_share",
        "idle_band_count",
        "compute_band_count",
        "decision",
    ):
        assert key in field_payload
    assert payload["status"] == "ok"
    assert payload["passed_fields"] == [FIELD]
    # The bar is part of the output so a verdict is reproducible from the JSON.
    assert payload["bar"]["min_separation_pp"] == PASS_SEPARATION_PP
    assert payload["bar"]["min_pearson_r"] == PASS_PEARSON_R
    assert payload["bar"]["max_idle_false_busy_share"] == PASS_MAX_IDLE_FALSE_BUSY_SHARE
    assert payload["band_counts"]["total"] == 4


def test_pearson_r_matches_numpy_free_reference_formula():
    """Guard the hand-rolled correlation against a direct covariance formula."""
    xs = [3.0, 5.0, 7.0, 9.0, 11.0]
    ys = [2.0, 6.0, 5.0, 11.0, 12.0]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    assert pearson_r(xs, ys) == pytest.approx(cov / denom)
