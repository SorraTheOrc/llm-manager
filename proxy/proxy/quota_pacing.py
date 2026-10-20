"""Pure pacing model for the quota-aware ``opencode-*`` surge.

This module is the *decision* half of the quota-surge epic
(LP-0MVF4FIP1001RD8P). :mod:`proxy.quota_state` answers "how much allowance
does this account have left and when does it reset?"; this module answers
"given that headroom and reset, how strongly should this entry be preferred —
and can the residual still be spent at all before it expires?".

Everything here is a **pure function**: no I/O, no globals, no caching, no
clock reads. All configuration arrives through an explicit ``config`` argument
with documented defaults, so the model can be unit-tested exhaustively and
tuned independently of ``proxy/provider.py``.

The model
---------

The producer decision (epic Q2 → option 2c) is that surge ordering uses a
**per-account urgency score**::

    score = effective_headroom / seconds_to_reset

i.e. "percent of the monthly allowance at risk per second until reset". A
large residual with a near reset scores higher than a small residual with a
distant reset, and equal (or missing/degenerate) inputs collapse to
:data:`BASELINE_PREFERENCE` (``0.0``) so the caller's stable config order is
preserved.

Why a ramp before the final week
--------------------------------

OpenCode Go allowances are use-it-or-lose-it, and the allowance is capped by a
rolling 5-hour window at **20%** of the monthly cap and a weekly window at
**50%** (https://opencode.ai/docs/go). Those caps bound how fast the residual
can physically be spent, so a large residual cannot be discharged in a
last-minute burst: :func:`ramp_started` returns ``True`` whenever the residual
exceeds :func:`achievable_fraction` — the fraction still spendable before the
reset given both caps — even when the reset is further away than
``quota_surge_days``. That is what makes the last day *finish* the quota
instead of attempting all of it at once.

Peak vs off-peak
----------------

Peak hours (``01:00-04:00`` and ``06:00-10:00`` UTC, Mon–Fri) price DeepSeek
V4.1 Flash at 2x the off-peak rate, so the same tokens burn the monthly
dollar allowance twice as fast (epic discovery finding 7,
:data:`PEAK_BURN_MULTIPLIER`). Surge therefore **prefers an already
``available_times``-eligible (off-peak) window** and only escalates to peak
hours when the residual is projected unspendable off-peak
(:func:`peak_escalation_allowed`). Peak escalation never relaxes any other
gate — it is a pricing/urgency decision only.

Degenerate inputs
-----------------

Every function is total: ``seconds_to_reset <= 0``, negative/NaN/infinite
headroom, absent or malformed config values all return the baseline
preference and never raise. ``server.quota_pacing_target`` outside ``(0, 1]``
is clamped, never trusted. Outside the surge window the model returns the
baseline preference, so a caller that sorts by the score behaves exactly as
today.

Scope
-----

No routing changes live here. Consumers:

- ``proxy/provider.py`` headroom-aware ordering (LP-0MVF7KIKE002DZVD) sorts by
  :func:`urgency_score`.
- the final-week ``available_times`` bypass (LP-0MVF7KJN2000ASE8) uses
  :func:`surge_window_active` / :func:`ramp_started`.
- the observability surface (LP-0MVF7KQED005TG9Z) logs
  :func:`assess_pacing` (a single :class:`PacingAssessment` worth of fields,
  including the human-readable ``reason``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Documented constants — encoded in exactly one place
# ---------------------------------------------------------------------------

# OpenCode Go allowance caps (https://opencode.ai/docs/go): the monthly
# allowance is a dollar amount, with a rolling-5-hour window capped at 20% of
# the monthly cap and a weekly window capped at 50% of it. The two windows
# nest, so the *binding* sustainable rate at any horizon is the smaller of
# (20% per 5 h) and (50% per week).
ROLLING_WINDOW_SECONDS = 5 * 3600
WEEKLY_WINDOW_SECONDS = 7 * 86400
ROLLING_CAP_FRACTION = 0.20
WEEKLY_CAP_FRACTION = 0.50

#: Peak hours burn the monthly dollar allowance twice as fast per token
#: (epic discovery finding 7: DeepSeek V4.1 Flash $0.15/$0.60 off-peak vs
#: $0.30/$1.20 peak per 1M tokens).
PEAK_BURN_MULTIPLIER = 2.0

#: ``server.quota_surge_days`` default: the final week before ``resetsAt``.
DEFAULT_SURGE_DAYS = 7.0

#: ``server.quota_pacing_target`` default: aim to spend 95% of the residual by
#: ``resetsAt``, leaving a 5% safety margin for the rolling/weekly caps.
DEFAULT_PACING_TARGET = 0.95

#: Smallest pacing target a clamped ``quota_pacing_target`` may take. The
#: value must stay strictly positive so the model never divides by zero or
#: silently drops the surge to a no-op.
MIN_PACING_TARGET = 0.01

#: Preference returned whenever the model has no surge opinion (feature
#: disabled, outside the surge window, or degenerate inputs). ``0.0`` means
#: "no quota preference" so callers fall back to their existing (config) order.
BASELINE_PREFERENCE = 0.0

#: Seconds in a day; every time-to-reset input is expressed in seconds.
SECONDS_PER_DAY = 86400.0

#: Reason codes reported by :func:`assess_pacing` (stable, log-friendly).
REASON_DISABLED = "disabled"
REASON_DEGENERATE = "degenerate"
REASON_BASELINE = "baseline"
REASON_SURGE_WINDOW = "surge_window"
REASON_RAMP = "ramp"


# ---------------------------------------------------------------------------
# Config helpers (flat-first, then ``server.*`` — same pattern as provider.py)
# ---------------------------------------------------------------------------


def _config_value(config: dict | None, key: str, default: Any) -> Any:
    if not isinstance(config, dict):
        return default
    value = config.get(key)
    if value is None:
        server = config.get("server")
        if isinstance(server, dict):
            value = server.get(key)
    return default if value is None else value


def _finite(value: Any) -> float | None:
    """Coerce *value* to a finite float, or ``None`` when that is impossible."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _config_bool(config: dict | None, key: str, default: bool) -> bool:
    value = _config_value(config, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def surge_enabled(config: dict | None = None) -> bool:
    """``server.quota_surge_enabled`` (default ``True``, the epic default)."""
    return _config_bool(config, "quota_surge_enabled", True)


def surge_dry_run(config: dict | None = None) -> bool:
    """``server.quota_surge_dry_run`` (default ``False``).

    Dry-run still *computes* the model (so the decisions can be logged) — it
    only tells the caller not to apply them. The pacing model therefore keeps
    returning real scores in dry-run; applications decide.
    """
    return _config_bool(config, "quota_surge_dry_run", False)


def surge_days(config: dict | None = None) -> float:
    """``server.quota_surge_days`` in days (default 7, clamped to ``>= 0``)."""
    value = _finite(_config_value(config, "quota_surge_days", DEFAULT_SURGE_DAYS))
    if value is None:
        return DEFAULT_SURGE_DAYS
    return max(0.0, value)


def surge_window_seconds(config: dict | None = None) -> float:
    """The final-week surge window in seconds (``surge_days * 86400``)."""
    return surge_days(config) * SECONDS_PER_DAY


def pacing_target(config: dict | None = None) -> float:
    """``server.quota_pacing_target``, clamped into ``(0, 1]``.

    An absent or malformed value falls back to
    :data:`DEFAULT_PACING_TARGET`. A value outside ``(0, 1]`` is clamped, never
    trusted: ``> 1`` becomes ``1.0`` (spend everything) and ``<= 0`` becomes
    :data:`MIN_PACING_TARGET` (never a zero/negative spend target).
    """
    raw = _finite(_config_value(config, "quota_pacing_target", DEFAULT_PACING_TARGET))
    if raw is None:
        return DEFAULT_PACING_TARGET
    return min(1.0, max(MIN_PACING_TARGET, raw))


# ---------------------------------------------------------------------------
# Achievability: how much of the residual can still be spent before reset
# ---------------------------------------------------------------------------


def achievable_fraction(seconds_to_reset: float, config: dict | None = None) -> float:
    """Fraction (``0.0``–``1.0``) of the monthly allowance still spendable.

    Derived from the documented cap relationship — a rolling 5-hour window of
    at most :data:`ROLLING_CAP_FRACTION` (20%) of the monthly cap and a weekly
    window of at most :data:`WEEKLY_CAP_FRACTION` (50%). Over ``T`` seconds the
    caps permit::

        rolling: 0.20 * T / (5 h)
        weekly:  0.50 * T / (7 d)

    The achievable fraction is the smaller of the two, capped at ``1.0`` (you
    can never spend more than the whole residual, however long the horizon).

    Degenerate inputs (``seconds_to_reset <= 0``, NaN/inf) yield ``0.0``: no
    time left means nothing more can be discharged. ``config`` is accepted for
    API symmetry with the other functions and to allow a future cap override
    without changing the call sites.
    """
    horizon = _finite(seconds_to_reset)
    if horizon is None or horizon <= 0:
        return 0.0
    rolling = ROLLING_CAP_FRACTION * horizon / ROLLING_WINDOW_SECONDS
    weekly = WEEKLY_CAP_FRACTION * horizon / WEEKLY_WINDOW_SECONDS
    return max(0.0, min(1.0, min(rolling, weekly)))


def cap_rate_per_second() -> float:
    """Highest allowance-consumption rate the caps allow (percent per second).

    The urgency score must never imply a spend rate above this, otherwise a
    "spend it all now" ramp would demand more than the rolling (20% / 5 h) or
    weekly (50% / 7 d) window permits. The weekly window is always the binding
    one here::

        50 / (7 d) = 8.27e-5 %/s   vs   20 / (5 h) = 1.11e-3 %/s
    """
    return min(
        ROLLING_CAP_FRACTION * 100.0 / ROLLING_WINDOW_SECONDS,
        WEEKLY_CAP_FRACTION * 100.0 / WEEKLY_WINDOW_SECONDS,
    )


def cap_spend_per_window(score: float, window_seconds: float) -> float:
    """Allowance percentage *score* would spend over *window_seconds*.

    Exposed so callers (and tests) can assert AC7c's bound directly: for any
    score returned by :func:`urgency_score`, spending at that rate for one
    rolling (5 h) or weekly (7 d) window never exceeds that window's cap.
    """
    rate = _finite(score)
    span = _finite(window_seconds)
    if rate is None or span is None or rate <= 0 or span <= 0:
        return 0.0
    return rate * span


# ---------------------------------------------------------------------------
# Surge window and ramp
# ---------------------------------------------------------------------------


def surge_window_active(seconds_to_reset: float, config: dict | None = None) -> bool:
    """True inside the final surge window (``0 < s <= quota_surge_days * 86400``).

    Degenerate values (``<= 0``, NaN, inf) are outside the window: an elapsed
    or unknown reset never activates the surge.
    """
    horizon = _finite(seconds_to_reset)
    if horizon is None or horizon <= 0:
        return False
    window = surge_window_seconds(config)
    return 0.0 < window and horizon <= window


def ramp_started(headroom: float, seconds_to_reset: float, config: dict | None = None) -> bool:
    """True when the residual can no longer be discharged before the reset.

    The cap relationship makes a large residual unspendable in a last-minute
    burst: with 90% of the monthly allowance left and 2 days to reset, the
    weekly cap (50% per week) allows at most ~14% more, so the ramp must have
    started already — even though the reset is *before* the final week. With
    only 10% left and 6 days to reset, ~43% is still spendable, so no ramp is
    needed.

    Returns ``False`` when the surge feature is disabled (nothing may ramp) or
    on any degenerate input.
    """
    if not surge_enabled(config):
        return False
    value = _finite(headroom)
    horizon = _finite(seconds_to_reset)
    if value is None or horizon is None or value <= 0 or horizon <= 0:
        return False
    residual = min(100.0, value) / 100.0
    return residual > achievable_fraction(horizon, config)


# ---------------------------------------------------------------------------
# Urgency score
# ---------------------------------------------------------------------------


def urgency_score(headroom: float, seconds_to_reset: float, config: dict | None = None) -> float:
    """Per-account surge preference (``effective_headroom / seconds_to_reset``).

    Returns :data:`BASELINE_PREFERENCE` (``0.0``) whenever the model has no
    surge opinion:

    - the feature is disabled (``server.quota_surge_enabled: false``);
    - any input is degenerate (non-finite, negative/zero headroom, or a reset
      that is now/past);
    - the entry is outside the surge window *and* :func:`ramp_started` is
      ``False`` (the residual is comfortably spendable in the remaining time).

    Otherwise the score is ``headroom * quota_pacing_target / seconds_to_reset``
    (the pacing target scales the planned spend, so the default 0.95 aims to
    finish at 95% consumed rather than exactly 100%), capped at
    :func:`cap_rate_per_second` so the implied rate never demands more than the
    rolling/weekly caps allow.

    The result is monotonically non-decreasing in ``headroom`` at a fixed
    reset, and non-decreasing as the reset approaches at a fixed headroom. In
    dry-run mode the real score is still returned (the caller logs it and does
    not apply it).
    """
    if not surge_enabled(config):
        return BASELINE_PREFERENCE
    value = _finite(headroom)
    horizon = _finite(seconds_to_reset)
    if value is None or horizon is None or value <= 0 or horizon <= 0:
        return BASELINE_PREFERENCE
    residual = min(100.0, value)
    if not (surge_window_active(horizon, config) or ramp_started(residual, horizon, config)):
        return BASELINE_PREFERENCE
    target = pacing_target(config)
    return min(residual * target / horizon, cap_rate_per_second())


# ---------------------------------------------------------------------------
# Peak vs off-peak weighting (R6)
# ---------------------------------------------------------------------------


def burn_multiplier(off_peak_eligible: bool) -> float:
    """Allowance burn multiplier for the current window.

    ``1.0`` in an already ``available_times``-eligible (off-peak) window,
    :data:`PEAK_BURN_MULTIPLIER` (2x) during peak hours — the same tokens
    deplete the monthly dollar allowance twice as fast.
    """
    return 1.0 if off_peak_eligible else PEAK_BURN_MULTIPLIER


def peak_escalation_allowed(
    headroom: float,
    seconds_to_reset: float,
    *,
    off_peak_eligible: bool,
    config: dict | None = None,
) -> bool:
    """Whether surge may spend in peak hours for this entry.

    Cheaper tokens come first (epic R6): a window that is already eligible
    under ``available_times`` is always preferred, so peak escalation is
    ``False`` there. Peak hours are used only when waiting for off-peak would
    strand the residual — i.e. when :func:`ramp_started` is ``True`` (the
    residual exceeds what the rolling/weekly caps allow off-peak). During peak
    the allowance burns at :data:`PEAK_BURN_MULTIPLIER` (2x), which is what
    makes peak hours effective at discharging an otherwise-unspendable
    residual.
    """
    if off_peak_eligible:
        return False
    return ramp_started(headroom, seconds_to_reset, config)


# ---------------------------------------------------------------------------
# Single assessment object (what consumers log / branch on)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PacingAssessment:
    """One immutable pacing verdict for an entry at a point in time.

    ``reason`` is a stable, log-friendly code: :data:`REASON_DISABLED`,
    :data:`REASON_DEGENERATE`, :data:`REASON_BASELINE`,
    :data:`REASON_SURGE_WINDOW` or :data:`REASON_RAMP`.
    """

    score: float
    reason: str
    headroom: float
    seconds_to_reset: float
    residual_fraction: float
    achievable_fraction: float
    surge_window: bool
    ramp_started: bool
    surge_active: bool
    off_peak_eligible: bool
    peak_escalation: bool
    burn_multiplier: float
    dry_run: bool

    @property
    def baseline(self) -> bool:
        """True when the model has no surge opinion (plain config order)."""
        return self.score == BASELINE_PREFERENCE

    def as_log_fields(self) -> dict[str, Any]:
        """Flat, JSON-serialisable fields for a routing/log line."""
        return {
            "score": self.score,
            "reason": self.reason,
            "headroom": self.headroom,
            "seconds_to_reset": self.seconds_to_reset,
            "residual_fraction": self.residual_fraction,
            "achievable_fraction": self.achievable_fraction,
            "surge_window": self.surge_window,
            "ramp_started": self.ramp_started,
            "surge_active": self.surge_active,
            "off_peak_eligible": self.off_peak_eligible,
            "peak_escalation": self.peak_escalation,
            "burn_multiplier": self.burn_multiplier,
            "dry_run": self.dry_run,
        }


def assess_pacing(
    headroom: float,
    seconds_to_reset: float,
    *,
    off_peak_eligible: bool = False,
    config: dict | None = None,
) -> PacingAssessment:
    """Assess an entry in one call, never raising on degenerate input.

    Composes the individual model functions so consumers (ordering, bypass,
    observability) branch on one object and log one coherent set of fields
    instead of re-deriving each value. ``off_peak_eligible`` is the caller's
    ``available_times`` verdict for the current instant; it only affects the
    peak/off-peak weighting, never the surge or ramp decision.
    """
    raw_headroom = _finite(headroom)
    raw_horizon = _finite(seconds_to_reset)
    enabled = surge_enabled(config)
    off_peak = bool(off_peak_eligible)

    if raw_headroom is None or raw_horizon is None:
        degenerate = True
        residual = 0.0
        horizon = 0.0
    else:
        degenerate = raw_headroom <= 0 or raw_horizon <= 0
        residual = max(0.0, min(100.0, raw_headroom))
        horizon = max(0.0, raw_horizon)

    window = False if degenerate or not enabled else surge_window_active(horizon, config)
    ramping = False if degenerate or not enabled else ramp_started(residual, horizon, config)
    score = urgency_score(
        residual if not degenerate else raw_headroom,
        horizon if not degenerate else raw_horizon,
        config,
    )

    if not enabled:
        reason = REASON_DISABLED
    elif degenerate:
        reason = REASON_DEGENERATE
    elif window:
        reason = REASON_SURGE_WINDOW
    elif ramping:
        reason = REASON_RAMP
    else:
        reason = REASON_BASELINE

    escalation = (
        False
        if degenerate or not enabled
        else peak_escalation_allowed(
            residual,
            horizon,
            off_peak_eligible=off_peak,
            config=config,
        )
    )

    return PacingAssessment(
        score=score,
        reason=reason,
        headroom=residual,
        seconds_to_reset=horizon,
        residual_fraction=residual / 100.0,
        achievable_fraction=achievable_fraction(horizon, config),
        surge_window=window,
        ramp_started=ramping,
        surge_active=score > BASELINE_PREFERENCE,
        off_peak_eligible=off_peak,
        peak_escalation=escalation,
        burn_multiplier=burn_multiplier(off_peak),
        dry_run=surge_dry_run(config),
    )
