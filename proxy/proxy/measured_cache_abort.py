"""Option D early-abort decision over *measured* prefill progress (F3).

LP-0MVR7HJD0007RFEM / parent LP-0MVQIT9QY007QU27.

Why measurement, not admission
------------------------------

Every admission-time warmth predicate fails for the "fake-warm" population:
``routing_resident_prefix`` decisions carry ``n_past_actual=none``, so
``resolve_resident_prefix`` reports ``reuse_ratio=1.0`` for a slot-owner
session and the prompt is admitted on the *prediction* that it is cached —
even though llama-server is about to re-prefill it from scratch (measured
2026-10-26: 29 of 116 deadline evictions were prompts ~56k with priced
``new_tokens`` 17–182, cancelled at 120 s).

llama.cpp b8782 emits ``prompt_progress`` inside the streaming response when
the request carries ``return_progress: true`` (contract pinned by
:mod:`proxy.prompt_progress`, F1). The first event is sent right after
``n_past`` is assigned, so its ``cache`` value is the *ground truth* reused
prefix for that dispatch; ``total - cache`` is the work actually being
computed. A genuinely warm dispatch shows ``cache ≈ 0.999 × total`` ⇒
``total - cache ≈ 0``; a fake-warm one shows a tiny ``cache`` against a large
``total``.

The decision
------------

Release the dispatch early (``reason=measured_cache_deficit``) when **all**
of the following hold:

1. the feature is enabled (config rollback: set the enable key to ``0``);
2. a valid measurement exists (``total > 0``, ``0 <= cache <= total``);
3. the measured deficit ratio ``(total - cache) / total`` is at least
   ``min_deficit_ratio`` (default 0.25) — this is what makes a genuinely warm
   dispatch *unreachable* for **any** value of the token bound, including
   ``0``, at any prompt size;
4. the *measured* actual new-token work exceeds the admission-priced
   ``new_tokens`` by more than ``bound_tokens`` (default 2 000);
5. **the benefit gate**: the measured deficit genuinely cannot complete inside
   the request's own prefill budget (LP-0MVWDZ1GG004UGIK, parent
   LP-0MVWBD91Z004U135).

Condition 4 alone is the parent's spec; condition 3 is the ratio floor that
AC2 (warm never aborts) and AC6 (documented bound + monotonicity) require:
with ``cache ≥ 0.99 × total`` the deficit ratio is ``≤ 0.01``, so no prompt
size and no bound value can trip the abort.

Conditions 3 and 4 are monotone in the safe direction: raising ``bound_tokens``
or raising ``min_deficit_ratio`` can only make an abort *less* likely.

Condition 5 (the benefit gate)
------------------------------

Conditions 3 and 4 are necessary but not sufficient. A mid-size cold prompt
(measured 2026-10-31: ``total=14 380``, ``cache=0``, priced ``new_tokens=48``)
satisfies both and was still cancelled at the 120 s prefill deadline — into a
600 s thrash cooldown and an endless ``429 Local admission refused`` loop — even
though its real cold prefill costs 38.8 s at the host median rate (371 tok/s)
and 84.6 s at p25 (170 tok/s). Waiting could therefore have succeeded.

The gate prices the **measured** deficit ``total - cache`` with the same
calibrated model the admission guard uses
(:func:`proxy.prefill_eta.estimate_prefill_seconds`, ``cached_ratio=0.0``
because a deficit is by definition uncached — never the admission-priced
``new_tokens``) and aborts only when that ETA **exceeds** the request's own
prefill-deadline budget::

    budget_s = budget_safety_fraction × local_prefill_deadline_seconds
               (0.95 × 120 s = 114 s shipped)

falling back to the bare ``local_prefill_deadline_seconds`` when the fraction
is explicitly ``0`` (the documented "no safety margin" rollback) or omitted.
The admission **wait** budget (``budget_seconds`` /
``contention_queue_max_wait_seconds``, 60 s) is never the comparator: it bounds
queueing, and the recorded incident carried ``deadline_s=120`` /
``new_tokens=49`` / ``eta_prefill_s=8.2``. Budget resolution lives in
:func:`proxy.prefill_eta.prefill_deadline_budget_seconds`, so the advertised
threshold can never drift from the enforced one.

``budget_seconds is None`` means "the benefit gate is not configured" (every
hand-built :class:`AbortSettings` row) and preserves the pre-gate verdict
byte-for-byte; ``budget_seconds <= 0`` means "configured, but no usable
budget" and fails open (continue).

Configuration keys (``server`` section of ``proxy/config.yaml``)
----------------------------------------------------------------

``local_measured_cache_abort_enabled``
    Feature switch. ``0`` / ``false`` / ``off`` restores pre-change
    behaviour: the decision always answers *continue* (no early release).
    Default: enabled.
``local_measured_cache_abort_bound_tokens``
    Absolute token slack above the admission-priced ``new_tokens``. Default
    ``2000``; ``0`` means "abort as soon as measurement exceeds the price".
``local_measured_cache_abort_min_deficit_ratio``
    Minimum measured ``(total - cache) / total`` for an abort. Default
    ``0.25``. Must sit far above the warm-path ceiling (``0.01``).

The benefit gate reads the **existing** admission-guard deadline keys (no new
config key): ``local_prefill_deadline_seconds`` and
``local_admission_guard_budget_safety_fraction``, resolved through
:func:`proxy.prefill_eta.prefill_deadline_budget_seconds`. The admission wait
budget (``contention_queue_max_wait_seconds``) is deliberately **not** read.

Fail-open contract (AC5)
------------------------

Nothing in this module can raise: a missing/partial/non-numeric measurement,
an unreadable progress object, an invalid priced value, an unreadable or
non-positive deadline/budget, an unusable ETA estimator, unparseable config,
or an internal exception all resolve to *continue* — i.e. today's behaviour.
The decision is a pure function of its arguments; wiring it into
``proxy_to_local``'s reframed stream loop is F4 (LP-0MVR7IN0O000DPXN).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

#: Eviction reason emitted when the decision fires (parent AC2/AC6).
MEASURED_CACHE_DEFICIT_REASON = "measured_cache_deficit"

#: ``server`` config keys (documented in the module docstring; F4 adds them to
#: ``proxy/config.yaml`` with the same wording).
ENABLED_KEY = "local_measured_cache_abort_enabled"
BOUND_TOKENS_KEY = "local_measured_cache_abort_bound_tokens"
MIN_DEFICIT_RATIO_KEY = "local_measured_cache_abort_min_deficit_ratio"

#: Feature default. The shipped ``config.yaml`` key is the operator rollback
#: switch; absent keys keep the target behaviour enabled so a partially
#: configured deploy does not silently leave the defect in place.
DEFAULT_ENABLED = True
DEFAULT_BOUND_TOKENS = 2_000
DEFAULT_MIN_DEFICIT_RATIO = 0.25

#: ``detail`` values on the returned decision (observability / tests).
DETAIL_ABORT = "deficit_over_bound"
DETAIL_DISABLED = "disabled"
DETAIL_NO_MEASUREMENT = "no_measurement"
DETAIL_INVALID_MEASUREMENT = "invalid_measurement"
DETAIL_INVALID_PRICED = "invalid_priced_new_tokens"
DETAIL_BELOW_RATIO_FLOOR = "below_deficit_ratio_floor"
DETAIL_WITHIN_BOUND = "within_bound"
DETAIL_ERROR = "decision_error"

#: The benefit gate's single new ``detail`` token (LP-0MVWDZ1GG004UGIK AC1b):
#: the deficit passed the ratio floor and the token bound, and waiting fits the
#: request's own prefill-deadline budget. No new reason token, no new header and
#: no refusal-body change; every fail-open path reuses ``DETAIL_ERROR``.
DETAIL_WITHIN_ETA_BUDGET = "within_eta_budget"


def _as_token_count(value: Any) -> int | None:
    """Coerce a b8782 token-count field to ``int``, or ``None`` if invalid.

    b8782 emits ``int32_t``, but an integral ``float`` (``56000.0``) is
    accepted for tolerance; ``bool``, strings, NaN/inf and fractional floats
    are rejected so a malformed event can never be mistaken for a measurement.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        return int(value)
    return value


def _as_non_negative_number(value: Any) -> float | None:
    """Coerce a (possibly fractional) non-negative number, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _as_positive_number(value: Any) -> float | None:
    """Coerce a strictly positive finite number, else ``None``."""
    number = _as_non_negative_number(value)
    if number is None or number <= 0.0:
        return None
    return number


def _field(source: Any, name: str) -> Any:
    """Read ``name`` from a mapping or object, returning ``None`` when absent.

    Both shapes are supported because F4's dispatch record may expose the
    measurement either as a :class:`proxy.prompt_progress.PromptProgress`
    (attributes) or as a plain dict from the stream loop.
    """
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


@dataclass(frozen=True)
class AbortSettings:
    """Resolved early-abort configuration.

    ``from_config`` is fail-open: any value it cannot understand resolves to
    ``enabled=False`` (pre-change behaviour), never to a *more* aggressive
    abort.

    The benefit gate's inputs (LP-0MVWDZ1GG004UGIK):

    ``model``
        Calibration key the measured ETA is priced with (the model the
        dispatch targets).
    ``config``
        The config the settings were resolved from — handed back to
        :func:`proxy.prefill_eta.estimate_prefill_seconds` so the gate prices
        with the same calibrated/percentile knobs the admission guard uses.
        Excluded from equality (runtime wiring, not a knob).
    ``budget_seconds``
        The comparator, ``prefill_deadline_budget_seconds(config)`` (114 s
        shipped). ``None`` = the benefit gate is not configured (pre-gate
        verdict, byte-for-byte); ``<= 0`` = configured but unusable (fail open).
        Excluded from equality so a fully-resolved settings object still
        compares equal to the hand-built rows whose verdicts it preserves.
    ``eta_estimator``
        One-argument callable pricing the **measured** deficit (tokens) into
        seconds; ``None`` uses the production estimator. Injected only for
        deterministic tests.
    """

    enabled: bool = DEFAULT_ENABLED
    bound_tokens: float = DEFAULT_BOUND_TOKENS
    min_deficit_ratio: float = DEFAULT_MIN_DEFICIT_RATIO
    model: str | None = None
    config: Mapping[str, Any] | None = field(
        default=None, compare=False, repr=False
    )
    budget_seconds: float | None = field(default=None, compare=False)
    eta_estimator: Callable[[int], float] | None = field(
        default=None, compare=False, repr=False
    )

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> AbortSettings:
        """Read the ``server`` keys, tolerating a flat dict and ``None``.

        An absent config (``None``) resolves to the documented defaults
        (feature enabled, benefit gate not configured); a *malformed* config
        resolves to ``enabled=False`` — fail open rather than guess at a bound.
        A readable config whose deadline budget is unusable resolves the benefit
        gate as *configured but unusable* (``budget_seconds=0``), which is a
        per-verdict fail-open rather than a disabled feature.
        """
        if config is None:
            return cls()
        if not isinstance(config, Mapping):
            return cls(enabled=False)
        server_cfg = config.get("server", config)
        if not isinstance(server_cfg, Mapping):
            return cls(enabled=False)

        enabled = _coerce_enabled(server_cfg.get(ENABLED_KEY, DEFAULT_ENABLED))
        if enabled is None:
            return cls(enabled=False)
        if not enabled:
            return cls(enabled=False)

        bound = _as_non_negative_number(
            server_cfg.get(BOUND_TOKENS_KEY, DEFAULT_BOUND_TOKENS)
        )
        if bound is None:
            return cls(enabled=False)

        ratio = _as_non_negative_number(
            server_cfg.get(MIN_DEFICIT_RATIO_KEY, DEFAULT_MIN_DEFICIT_RATIO)
        )
        if ratio is None or ratio > 1.0:
            return cls(enabled=False)

        budget = _resolve_budget_seconds(config)
        return cls(
            enabled=True,
            bound_tokens=bound,
            min_deficit_ratio=ratio,
            config=config,
            # ``None`` from the helper means "configured, but unusable": 0 is
            # the falsy budget the decision fails open on (never an abort).
            budget_seconds=0.0 if budget is None else budget,
        )


def _resolve_budget_seconds(config: Mapping[str, Any]) -> float | None:
    """The deadline-derived comparator, or ``None`` when unusable (AC3).

    Delegates to the public
    :func:`proxy.prefill_eta.prefill_deadline_budget_seconds` so the threshold
    can never drift from the one the admission guard enforces. Fail-open: an
    import error or any exception here reads as "no usable budget".
    """
    try:
        from proxy.prefill_eta import prefill_deadline_budget_seconds

        return prefill_deadline_budget_seconds(dict(config))
    except Exception:  # AC5: resolution trouble is never an abort
        return None


def _coerce_enabled(value: Any) -> bool | None:
    """Interpret an enable switch; ``None`` when the value is unusable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value != 0
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalised in {"0", "false", "no", "off", "disabled", ""}:
            return False
    return None


@dataclass(frozen=True)
class AbortDecision:
    """Outcome of :func:`decide_measured_cache_abort`.

    ``abort`` is the only field wiring code (F4) must act on; the remaining
    fields exist so the eviction snapshot and the post-deploy window can
    report *why* (parent AC6) without re-deriving the numbers.

    ``measured_eta_s`` / ``budget_s`` / ``eta_rate_bucket`` are the benefit
    gate's provenance (LP-0MVWDZ1GG004UGIK): the ETA the measured deficit was
    priced at, the comparator it was compared against, and the calibration
    bucket that priced it — so the dispatch log line can report why a deficit
    was tolerated (parent AC5). All ``None`` when the gate did not price.
    """

    abort: bool
    reason: str | None
    detail: str
    measured_new_tokens: int | None = None
    deficit_ratio: float | None = None
    priced_new_tokens: float | None = None
    bound_tokens: float = DEFAULT_BOUND_TOKENS
    min_deficit_ratio: float = DEFAULT_MIN_DEFICIT_RATIO
    measured_eta_s: float | None = None
    budget_s: float | None = None
    eta_rate_bucket: str | None = None

    @property
    def continue_verdict(self) -> bool:
        """``True`` when the dispatch must proceed (today's behaviour)."""
        return not self.abort


def _continue(
    detail: str,
    *,
    measured_new_tokens: int | None = None,
    deficit_ratio: float | None = None,
    priced_new_tokens: float | None = None,
    measured_eta_s: float | None = None,
    budget_s: float | None = None,
    eta_rate_bucket: str | None = None,
    settings: AbortSettings | None = None,
) -> AbortDecision:
    """Build a *continue* verdict carrying whatever context was resolved."""
    effective = settings if isinstance(settings, AbortSettings) else AbortSettings()
    return AbortDecision(
        abort=False,
        reason=None,
        detail=detail,
        measured_new_tokens=measured_new_tokens,
        deficit_ratio=deficit_ratio,
        priced_new_tokens=priced_new_tokens,
        bound_tokens=effective.bound_tokens,
        min_deficit_ratio=effective.min_deficit_ratio,
        measured_eta_s=measured_eta_s,
        budget_s=budget_s,
        eta_rate_bucket=eta_rate_bucket,
    )


def _price_measured_deficit(
    deficit_tokens: int, settings: AbortSettings
) -> tuple[float | None, str | None]:
    """Price the **measured** deficit into ``(eta_seconds, rate_bucket)``.

    Uses the admission guard's own calibrated model
    (:func:`proxy.prefill_eta.estimate_prefill_seconds` with
    ``cached_ratio=0.0``: a deficit is by definition uncached) unless an
    estimator was injected. Any failure — an import error, a raising estimator,
    a non-numeric/NaN/negative result — reads as "unpriceable" (``None``), so
    the gate fails open instead of raising (AC6).
    """
    estimator = settings.eta_estimator
    if estimator is None:
        try:
            from proxy.prefill_eta import estimate_prefill_seconds

            seconds, coefficients = estimate_prefill_seconds(
                settings.model, deficit_tokens, 0.0, settings.config
            )
        except Exception:
            return None, None
        bucket = getattr(coefficients, "rate_bucket", None)
        return _as_non_negative_number(seconds), (
            bucket if isinstance(bucket, str) else None
        )
    if not callable(estimator):
        return None, None
    try:
        seconds = estimator(deficit_tokens)
    except Exception:
        return None, None
    return _as_non_negative_number(seconds), None


def decide_measured_cache_abort(
    progress: Any | None,
    priced_new_tokens: Any,
    settings: AbortSettings | None = None,
) -> AbortDecision:
    """Decide whether to release a local dispatch on measured cache deficit.

    Args:
        progress: The first/latest ``prompt_progress`` measurement — a
            :class:`proxy.prompt_progress.PromptProgress`, any object with
            ``total``/``cache`` attributes, or a mapping with those keys.
            ``None`` (no measurement) always continues.
        priced_new_tokens: The admission-time priced compute for this
            dispatch (``estimated_tokens * (1 - cached_ratio)``). Fractions
            are accepted.
        settings: Resolved configuration; ``None`` uses
            :class:`AbortSettings` defaults. A non-``AbortSettings`` value is
            itself a fail-open condition (continue).

    Returns:
        An :class:`AbortDecision`. Never raises (AC5).
    """
    effective = AbortSettings() if settings is None else settings
    if not isinstance(effective, AbortSettings):
        return _continue(DETAIL_ERROR)

    try:
        if not effective.enabled:
            return _continue(DETAIL_DISABLED, settings=effective)

        priced = _as_non_negative_number(priced_new_tokens)
        if priced is None:
            return _continue(DETAIL_INVALID_PRICED, settings=effective)

        total = _as_token_count(_field(progress, "total"))
        cache = _as_token_count(_field(progress, "cache"))
        if total is None or cache is None:
            return _continue(
                DETAIL_NO_MEASUREMENT, priced_new_tokens=priced, settings=effective
            )
        if total <= 0 or cache < 0 or cache > total:
            return _continue(
                DETAIL_INVALID_MEASUREMENT,
                priced_new_tokens=priced,
                settings=effective,
            )

        measured_new = total - cache
        deficit_ratio = measured_new / total

        if deficit_ratio < effective.min_deficit_ratio:
            return _continue(
                DETAIL_BELOW_RATIO_FLOOR,
                measured_new_tokens=measured_new,
                deficit_ratio=deficit_ratio,
                priced_new_tokens=priced,
                settings=effective,
            )

        if measured_new <= priced + effective.bound_tokens:
            return _continue(
                DETAIL_WITHIN_BOUND,
                measured_new_tokens=measured_new,
                deficit_ratio=deficit_ratio,
                priced_new_tokens=priced,
                settings=effective,
            )

        # Enforced order: ratio floor -> token bound -> ETA budget (AC1).
        return _decide_on_eta_budget(
            measured_new=measured_new,
            deficit_ratio=deficit_ratio,
            priced=priced,
            settings=effective,
        )
    except Exception:  # AC5: a decision error must never abort
        return _continue(DETAIL_ERROR, settings=effective)


def _decide_on_eta_budget(
    *,
    measured_new: int,
    deficit_ratio: float,
    priced: float,
    settings: AbortSettings,
) -> AbortDecision:
    """The benefit gate: abort only when waiting cannot fit the budget (AC1).

    Called only after the ratio floor and the token bound have both been
    passed, so an abort here requires all three conditions.

    * ``budget_seconds is None`` — the gate is not configured (hand-built
      settings): the pre-gate verdict is preserved byte-for-byte.
    * ``budget_seconds`` unusable (``<= 0``/NaN) — configured but no comparator:
      fail open (continue), never an abort.
    * the measured deficit cannot be priced — fail open (continue).
    * ``measured_eta_s > budget_s`` — abort; equal is *within* budget.
    """
    budget_s = settings.budget_seconds
    context: dict[str, Any] = {
        "measured_new_tokens": measured_new,
        "deficit_ratio": deficit_ratio,
        "priced_new_tokens": priced,
    }

    if budget_s is None:
        return AbortDecision(
            abort=True,
            reason=MEASURED_CACHE_DEFICIT_REASON,
            detail=DETAIL_ABORT,
            bound_tokens=settings.bound_tokens,
            min_deficit_ratio=settings.min_deficit_ratio,
            **context,
        )

    usable_budget = _as_positive_number(budget_s)
    if usable_budget is None:
        # Configured, but no usable comparator: fail open. ``DETAIL_ERROR`` is
        # reused (AC1b: the gate adds exactly one ``detail`` token); the
        # ``budget_s=0`` / ``measured_eta_s=None`` pair identifies the case.
        return _continue(
            DETAIL_ERROR, budget_s=0.0, settings=settings, **context
        )

    measured_eta_s, rate_bucket = _price_measured_deficit(measured_new, settings)
    if measured_eta_s is None:
        # Unpriceable measured deficit: fail open (never an abort).
        return _continue(
            DETAIL_ERROR,
            budget_s=usable_budget,
            eta_rate_bucket=rate_bucket,
            settings=settings,
            **context,
        )

    if measured_eta_s > usable_budget:
        return AbortDecision(
            abort=True,
            reason=MEASURED_CACHE_DEFICIT_REASON,
            detail=DETAIL_ABORT,
            bound_tokens=settings.bound_tokens,
            min_deficit_ratio=settings.min_deficit_ratio,
            measured_eta_s=measured_eta_s,
            budget_s=usable_budget,
            eta_rate_bucket=rate_bucket,
            **context,
        )

    return _continue(
        DETAIL_WITHIN_ETA_BUDGET,
        measured_eta_s=measured_eta_s,
        budget_s=usable_budget,
        eta_rate_bucket=rate_bucket,
        settings=settings,
        **context,
    )
