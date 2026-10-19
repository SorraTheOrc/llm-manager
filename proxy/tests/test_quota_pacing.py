"""Tests for the pure quota-surge pacing model (LP-0MVF7KHI5000J7BL).

The module under test (``proxy.quota_pacing``) is deliberately pure — no I/O,
no clock, no globals — so the whole acceptance-criteria surface is covered by
ordinary table-driven assertions against its public API:

- AC7a: ``urgency_score`` is monotonically non-decreasing in headroom at a
  fixed time-to-reset and non-decreasing as the reset approaches at fixed
  headroom.
- AC7b: outside the surge window the baseline preference is returned, and a
  large residual with a near reset outscores a small residual with a distant
  reset.
- AC7c: ``ramp_started`` fires before the final week when the residual is
  large, and no returned score implies spending more than the rolling (20% /
  5 h) or weekly (50% / 7 d) cap in one window.
- AC7d: peak weighting prefers an already off-peak-eligible window and only
  escalates when the residual is otherwise unspendable, with the 2x peak burn
  rate represented explicitly.
- AC7e: degenerate inputs (``seconds_to_reset <= 0``, negative/NaN/inf
  headroom, absent config) return the baseline and never raise, and an
  out-of-range ``quota_pacing_target`` is clamped rather than trusted.

Frozen inputs are expressed in seconds and percentages — the same units the
quota poller (``proxy.quota_state``) derives — so the tables read like the
fork's documented cap relationship (https://opencode.ai/docs/go).
"""

from __future__ import annotations

import math

import pytest
from proxy.quota_pacing import (
    BASELINE_PREFERENCE,
    DEFAULT_PACING_TARGET,
    MIN_PACING_TARGET,
    PEAK_BURN_MULTIPLIER,
    REASON_BASELINE,
    REASON_DEGENERATE,
    REASON_DISABLED,
    REASON_RAMP,
    REASON_SURGE_WINDOW,
    ROLLING_CAP_FRACTION,
    ROLLING_WINDOW_SECONDS,
    WEEKLY_CAP_FRACTION,
    WEEKLY_WINDOW_SECONDS,
    achievable_fraction,
    assess_pacing,
    burn_multiplier,
    cap_rate_per_second,
    cap_spend_per_window,
    pacing_target,
    peak_escalation_allowed,
    ramp_started,
    surge_days,
    surge_dry_run,
    surge_enabled,
    surge_window_active,
    surge_window_seconds,
    urgency_score,
)

HOUR = 3600.0
DAY = 86400.0

#: Representative fixed time-to-reset values spanning "long before reset" to
#: "about to reset", used by the monotonicity sweeps.
RESET_HORIZONS = (30 * DAY, 8 * DAY, 7 * DAY, 3 * DAY, 1 * DAY, 6 * HOUR, 300.0)

#: Representative fixed headroom values (percent of the monthly allowance).
HEADROOMS = (0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 99.5, 100.0)


# ---------------------------------------------------------------------------
# Config helpers: documented defaults and clamping
# ---------------------------------------------------------------------------


def test_config_defaults():
    """Absent config yields the documented defaults (epic spend profile 1a)."""
    assert surge_enabled({}) is True
    assert surge_dry_run({}) is False
    assert surge_days({}) == 7.0
    assert surge_window_seconds({}) == 7 * DAY
    assert pacing_target({}) == DEFAULT_PACING_TARGET == 0.95


def test_config_reads_flat_and_server_scoped_keys():
    """Both the flat and the ``server.``-scoped spelling are honoured."""
    flat = {"quota_surge_days": 3, "quota_pacing_target": 0.5}
    nested = {"server": {"quota_surge_days": 3, "quota_pacing_target": 0.5}}
    assert surge_days(flat) == surge_days(nested) == 3.0
    assert pacing_target(flat) == pacing_target(nested) == 0.5
    assert surge_enabled({"server": {"quota_surge_enabled": False}}) is False
    assert surge_dry_run({"server": {"quota_surge_dry_run": "true"}}) is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (2.0, 1.0),
        (1.0, 1.0),
        (0.95, 0.95),
        (0.0, MIN_PACING_TARGET),
        (-0.5, MIN_PACING_TARGET),
        (math.nan, DEFAULT_PACING_TARGET),
        (math.inf, DEFAULT_PACING_TARGET),
        (-math.inf, DEFAULT_PACING_TARGET),
        ("nonsense", DEFAULT_PACING_TARGET),
        (None, DEFAULT_PACING_TARGET),
    ],
)
def test_pacing_target_is_clamped_not_trusted(raw, expected):
    """AC7e: ``quota_pacing_target`` outside ``(0, 1]`` is clamped."""
    config = {"server": {"quota_pacing_target": raw}}
    assert pacing_target(config) == pytest.approx(expected)


def test_surge_days_never_negative():
    """A negative ``quota_surge_days`` clamps to 0 (surge window disabled)."""
    assert surge_days({"server": {"quota_surge_days": -4}}) == 0.0
    assert surge_window_seconds({"server": {"quota_surge_days": -4}}) == 0.0
    assert surge_window_active(60, {"server": {"quota_surge_days": -4}}) is False


# ---------------------------------------------------------------------------
# AC7a — monotonicity sweeps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", RESET_HORIZONS)
def test_urgency_score_is_non_decreasing_in_headroom(horizon):
    """AC7a(a): fixed time-to-reset, headroom swept 0 -> 100."""
    scores = [urgency_score(h, horizon) for h in [i / 2 for i in range(201)]]
    assert all(later >= earlier for earlier, later in zip(scores, scores[1:]))
    # Inside the surge window the model must actually express a preference.
    if horizon <= 7 * DAY:
        assert scores[-1] > BASELINE_PREFERENCE


@pytest.mark.parametrize("headroom", HEADROOMS)
def test_urgency_score_is_non_decreasing_as_reset_approaches(headroom):
    """AC7a(b): fixed headroom, reset swept 30 days -> 5 minutes."""
    horizons = [30 * DAY * (1 - i / 200) for i in range(200)] + [300.0]
    scores = [urgency_score(headroom, horizon) for horizon in horizons]
    assert all(later >= earlier for earlier, later in zip(scores, scores[1:]))


def test_urgency_score_inside_window_equals_headroom_over_time_with_target():
    """The core producer decision: score = headroom * target / seconds_to_reset."""
    config = {"server": {"quota_pacing_target": 0.95}}
    headroom, horizon = 10.0, 2 * DAY
    expected = headroom * 0.95 / horizon
    assert urgency_score(headroom, horizon, config) == pytest.approx(expected)
    # A lower pacing target scales the planned spend down, never the ordering.
    assert urgency_score(headroom, horizon, {"server": {"quota_pacing_target": 0.5}}) == (
        pytest.approx(headroom * 0.5 / horizon)
    )


# ---------------------------------------------------------------------------
# AC7b — baseline outside the surge window; residual-vs-horizon preference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("headroom", "horizon"),
    [
        (10.0, 30 * DAY),  # small residual, far reset: spendable, no surge
        (40.0, 8 * DAY),  # just outside the window, still comfortably spendable
        (5.0, 20 * DAY),
        (100.0, 29 * DAY),
    ],
)
def test_baseline_outside_surge_window(headroom, horizon):
    """AC7b: outside the surge window the baseline (no-surge) preference wins."""
    assert surge_window_active(horizon) is False
    assert ramp_started(headroom, horizon) is False
    assert urgency_score(headroom, horizon) == BASELINE_PREFERENCE
    assessment = assess_pacing(headroom, horizon)
    assert assessment.reason == REASON_BASELINE
    assert assessment.baseline is True
    assert assessment.surge_active is False


def test_large_residual_near_reset_beats_small_residual_distant_reset():
    """AC7b: headroom/time favours the account whose allowance is at risk."""
    at_risk = urgency_score(90.0, 2 * DAY)
    distant = urgency_score(10.0, 20 * DAY)
    assert at_risk > distant == BASELINE_PREFERENCE

    # Even against a small residual that *is* inside the window, the large
    # residual with the near reset still wins.
    small_in_window = urgency_score(10.0, 6 * DAY)
    assert at_risk > small_in_window > BASELINE_PREFERENCE


def test_surge_window_boundaries():
    """``0 < seconds_to_reset <= quota_surge_days * 86400`` exactly."""
    assert surge_window_active(7 * DAY) is True
    assert surge_window_active(7 * DAY + 1) is False
    assert surge_window_active(1.0) is True
    assert surge_window_active(0.0) is False
    assert surge_window_active(-1.0) is False
    # A custom window moves the boundary.
    config = {"server": {"quota_surge_days": 3}}
    assert surge_window_active(3 * DAY, config) is True
    assert surge_window_active(3 * DAY + 1, config) is False


# ---------------------------------------------------------------------------
# AC7c — achievability under the rolling (20%) / weekly (50%) caps
# ---------------------------------------------------------------------------


def test_achievable_fraction_follows_the_documented_cap_relationship():
    """AC7c: the achievable fraction is the binding cap at that horizon."""
    # 5-hour horizon: the weekly cap (50% per 7 d) binds, not the rolling one.
    assert achievable_fraction(ROLLING_WINDOW_SECONDS) == pytest.approx(
        WEEKLY_CAP_FRACTION * ROLLING_WINDOW_SECONDS / WEEKLY_WINDOW_SECONDS
    )
    # One day: 50% * 1/7.
    assert achievable_fraction(DAY) == pytest.approx(0.5 / 7)
    # A full week: exactly the weekly cap.
    assert achievable_fraction(WEEKLY_WINDOW_SECONDS) == pytest.approx(WEEKLY_CAP_FRACTION)
    # Two weeks and beyond: never more than the whole residual.
    assert achievable_fraction(14 * DAY) == 1.0
    assert achievable_fraction(60 * DAY) == 1.0
    # No time left means nothing more can be discharged.
    assert achievable_fraction(0) == 0.0
    assert achievable_fraction(-10) == 0.0


def test_cap_rate_is_the_binding_sustainable_rate():
    """AC7c: the cap rate is the smaller of the rolling and weekly rates."""
    rolling_rate = ROLLING_CAP_FRACTION * 100 / ROLLING_WINDOW_SECONDS
    weekly_rate = WEEKLY_CAP_FRACTION * 100 / WEEKLY_WINDOW_SECONDS
    assert weekly_rate < rolling_rate
    assert cap_rate_per_second() == pytest.approx(weekly_rate)


@pytest.mark.parametrize(
    ("headroom", "horizon"),
    [
        (90.0, 2 * DAY),
        (95.0, 1 * DAY),
        (100.0, 1.0),
        (60.0, 5 * DAY),
        (100.0, 5 * HOUR),
        (10.0, 6 * DAY),
        (75.0, 3 * DAY),
    ],
)
def test_no_score_exceeds_a_window_cap(headroom, horizon):
    """AC7c: no returned score implies more spend than a window cap allows."""
    score = urgency_score(headroom, horizon)
    assert cap_spend_per_window(score, ROLLING_WINDOW_SECONDS) <= (ROLLING_CAP_FRACTION * 100 + 1e-9)
    assert cap_spend_per_window(score, WEEKLY_WINDOW_SECONDS) <= (WEEKLY_CAP_FRACTION * 100 + 1e-9)


def test_score_sweep_never_breaches_a_window_cap():
    """AC7c over a grid: the cap bound holds for every combination."""
    for headroom in (0.5, 5.0, 20.0, 50.0, 90.0, 99.9, 100.0):
        for horizon in (1.0, 300.0, 5 * HOUR, DAY, 2 * DAY, 6 * DAY, 7 * DAY):
            score = urgency_score(headroom, horizon)
            assert cap_spend_per_window(score, ROLLING_WINDOW_SECONDS) <= (ROLLING_CAP_FRACTION * 100 + 1e-9)
            assert cap_spend_per_window(score, WEEKLY_WINDOW_SECONDS) <= (WEEKLY_CAP_FRACTION * 100 + 1e-9)


@pytest.mark.parametrize(
    ("headroom", "horizon", "expected"),
    [
        # The epic's worked example: 90% left with 2 days cannot be discharged
        # (at most ~14% is spendable), so the ramp must already have started.
        (90.0, 2 * DAY, True),
        # 10% left with 6 days is comfortably spendable (~43%).
        (10.0, 6 * DAY, False),
        (95.0, 1 * DAY, True),
        (60.0, 5 * DAY, True),
        (20.0, 6 * DAY, False),
        (5.0, 7 * DAY, False),
        # No residual -> nothing to ramp for.
        (0.0, 2 * DAY, False),
        # A month out, the whole residual is still spendable.
        (90.0, 30 * DAY, False),
        # Just outside the final week a 100% residual is still at risk.
        (100.0, 8 * DAY, True),
    ],
)
def test_ramp_started_achievability_table(headroom, horizon, expected):
    """AC7c: the ramp starts before the final week only when it must."""
    assert ramp_started(headroom, horizon) is expected


def test_ramp_started_is_reported_outside_the_surge_window():
    """AC7c: a very large residual ramps early, flagged as ``ramp``."""
    horizon = 30 * DAY
    assert surge_window_active(horizon) is False
    # 100% residual is spendable in 30 days (achievable == 1.0): baseline.
    assert assess_pacing(100.0, horizon).reason == REASON_BASELINE

    near = 8 * DAY
    assessment = assess_pacing(100.0, near)
    assert assessment.reason == REASON_RAMP
    assert assessment.ramp_started is True
    assert assessment.surge_active is True
    assert assessment.score > BASELINE_PREFERENCE


# ---------------------------------------------------------------------------
# AC7d — peak/off-peak weighting (R6)
# ---------------------------------------------------------------------------


def test_burn_multiplier_represents_the_2x_peak_rate():
    """AC7d: peak hours burn the dollar allowance 2x faster per token."""
    assert PEAK_BURN_MULTIPLIER == 2.0
    assert burn_multiplier(off_peak_eligible=True) == 1.0
    assert burn_multiplier(off_peak_eligible=False) == PEAK_BURN_MULTIPLIER


@pytest.mark.parametrize(
    ("headroom", "horizon", "off_peak", "expected"),
    [
        # Already inside its available_times window -> never escalate.
        (90.0, 2 * DAY, True, False),
        # Peak window with an unspendable residual -> escalation allowed.
        (90.0, 2 * DAY, False, True),
        # Peak window, residual comfortably spendable off-peak -> no escalation.
        (10.0, 6 * DAY, False, False),
        (0.0, 2 * DAY, False, False),
        # Off-peak eligibility wins even when escalation would be allowed.
        (99.0, DAY, True, False),
    ],
)
def test_peak_escalation_only_when_otherwise_unspendable(headroom, horizon, off_peak, expected):
    """AC7d: off-peak preferred; peak only for a projected-unspendable residual."""
    assert peak_escalation_allowed(headroom, horizon, off_peak_eligible=off_peak) is expected


def test_assessment_exposes_peak_weighting():
    """AC7d: the assessment carries the off-peak verdict and 2x burn factor."""
    off_peak = assess_pacing(90.0, 2 * DAY, off_peak_eligible=True)
    assert off_peak.peak_escalation is False
    assert off_peak.burn_multiplier == 1.0
    assert off_peak.off_peak_eligible is True

    peak = assess_pacing(90.0, 2 * DAY, off_peak_eligible=False)
    assert peak.peak_escalation is True
    assert peak.burn_multiplier == PEAK_BURN_MULTIPLIER
    assert peak.off_peak_eligible is False


# ---------------------------------------------------------------------------
# AC7e — degenerate inputs never raise and return the baseline
# ---------------------------------------------------------------------------


DEGENERATE_INPUTS = [
    (0.0, DAY),
    (-10.0, DAY),
    (50.0, 0.0),
    (50.0, -1.0),
    (math.nan, DAY),
    (50.0, math.nan),
    (math.inf, DAY),
    (50.0, math.inf),
    (-math.inf, DAY),
    (50.0, -math.inf),
    (None, DAY),
    (50.0, None),
    (None, None),
    ("not-a-number", "also-not"),
    ({}, []),
]


@pytest.mark.parametrize(("headroom", "horizon"), DEGENERATE_INPUTS)
def test_degenerate_inputs_return_baseline_and_never_raise(headroom, horizon):
    """AC7e: every degenerate pair returns the baseline preference quietly."""
    assert urgency_score(headroom, horizon) == BASELINE_PREFERENCE
    assert ramp_started(headroom, horizon) is False
    assert peak_escalation_allowed(headroom, horizon, off_peak_eligible=False) is False

    assessment = assess_pacing(headroom, horizon)
    assert assessment.score == BASELINE_PREFERENCE
    assert assessment.reason == REASON_DEGENERATE
    assert assessment.baseline is True
    assert assessment.surge_active is False


@pytest.mark.parametrize("horizon", [0.0, -1.0, math.nan, math.inf, -math.inf, None, "x", {}])
def test_surge_window_inactive_for_degenerate_horizons(horizon):
    """AC7e: a missing/elapsed/unknown reset never activates the surge."""
    assert surge_window_active(horizon) is False
    assert achievable_fraction(horizon) == 0.0


def test_headroom_is_clamped_to_the_allowance_range():
    """A nonsensical >100% headroom can never imply more than 100% allowance."""
    assert urgency_score(150.0, DAY) == pytest.approx(urgency_score(100.0, DAY))


# ---------------------------------------------------------------------------
# Kill switch, dry-run and assessment composition
# ---------------------------------------------------------------------------


def test_disabled_feature_returns_the_baseline_everywhere():
    """The kill switch silences the model: no surge, no ramp, no escalation."""
    config = {"server": {"quota_surge_enabled": False}}
    assert urgency_score(95.0, DAY, config) == BASELINE_PREFERENCE
    assert ramp_started(95.0, DAY, config) is False
    # The surge *window* is still a factual time window; only the policy is off.
    assert surge_window_active(DAY, config) is True
    assert peak_escalation_allowed(95.0, DAY, off_peak_eligible=False, config=config) is False
    assessment = assess_pacing(95.0, DAY, off_peak_eligible=False, config=config)
    assert assessment.reason == REASON_DISABLED
    assert assessment.surge_active is False
    assert assessment.peak_escalation is False


def test_dry_run_still_computes_the_model():
    """Dry-run computes the real decision (so it can be logged) but flags it."""
    live = {"server": {"quota_surge_dry_run": False}}
    dry = {"server": {"quota_surge_dry_run": True}}
    assert urgency_score(90.0, 2 * DAY, dry) == pytest.approx(urgency_score(90.0, 2 * DAY, live))
    assessment = assess_pacing(90.0, 2 * DAY, config=dry)
    assert assessment.dry_run is True
    assert assessment.reason == REASON_SURGE_WINDOW
    assert assess_pacing(90.0, 2 * DAY, config=live).dry_run is False


@pytest.mark.parametrize("horizon", (30 * DAY, 8 * DAY, 7 * DAY, 2 * DAY, DAY, 60.0))
@pytest.mark.parametrize("headroom", (1.0, 10.0, 50.0, 90.0, 99.5, 100.0))
@pytest.mark.parametrize("off_peak", (True, False))
def test_assessment_agrees_with_the_individual_model_functions(headroom, horizon, off_peak):
    """The single-object assessment composes the public functions unchanged."""
    assessment = assess_pacing(headroom, horizon, off_peak_eligible=off_peak)
    assert assessment.score == pytest.approx(urgency_score(headroom, horizon))
    assert assessment.surge_window is surge_window_active(horizon)
    assert assessment.ramp_started is ramp_started(headroom, horizon)
    assert assessment.achievable_fraction == pytest.approx(achievable_fraction(horizon))
    assert assessment.peak_escalation is peak_escalation_allowed(headroom, horizon, off_peak_eligible=off_peak)
    assert assessment.burn_multiplier == burn_multiplier(off_peak)
    expected_reason = (
        REASON_SURGE_WINDOW
        if assessment.surge_window
        else (REASON_RAMP if assessment.ramp_started else REASON_BASELINE)
    )
    assert assessment.reason == expected_reason
    assert assessment.surge_active is (assessment.score > BASELINE_PREFERENCE)


def test_assessment_log_fields_are_flat_and_serialisable():
    """R8: one JSON-safe field set per routing decision."""
    fields = assess_pacing(90.0, 2 * DAY, off_peak_eligible=False).as_log_fields()
    assert set(fields) == {
        "score",
        "reason",
        "headroom",
        "seconds_to_reset",
        "residual_fraction",
        "achievable_fraction",
        "surge_window",
        "ramp_started",
        "surge_active",
        "off_peak_eligible",
        "peak_escalation",
        "burn_multiplier",
        "dry_run",
    }
    assert fields["reason"] == REASON_SURGE_WINDOW
    assert fields["residual_fraction"] == pytest.approx(0.9)
    assert fields["burn_multiplier"] == PEAK_BURN_MULTIPLIER
    import json

    json.dumps(fields)  # never raises: no datetimes / NaN in the log payload
