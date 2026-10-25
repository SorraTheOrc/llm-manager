"""Prefill-ETA calibration and the local admission guard (LP-0MVGDUV5E005UWBQ).

Problem
-------

When every local slot is busy, a request enters the bounded contention queue
(bounded by ``contention_queue_max_wait_seconds``, 60 s in the shipped
config — halved from 120 in LP-0MVIQ0EXV00673HO) and waits. For a large
cache-cold prompt the local prefill cannot finish inside that budget, so the
wait is a *guaranteed loss*: the request burns up to the full wait and then
falls back to a remote provider anyway (measured 2026-10-19/20: 5,805 waits
→ 855 dispatched, 4,947 = 85 % abandoned after up to the previous 120 s).

The guard
---------

Before entering the queue (and before a dispatch that would start another
concurrent prefill), the proxy costs the projected prefill from *observed*
local dispatches and compares it with the caller's wait budget::

    eta_prefill = coded_tokens * coded_ms_per_token
                + new_tokens   * prefill_ms_per_token

where ``coded_tokens = est_tokens * cached_ratio`` (the prompt prefix served
from the prompt cache) and ``new_tokens = est_tokens * (1 - cached_ratio)``
(the tokens that must actually be prefilled). A cache-warm prompt therefore
stays admissible — its expensive ``new_tokens`` term is small — while a large
cache-cold prompt is refused *immediately* instead of parking for the full
queue budget.

Occupancy (LP-0MVICD2YW004DJPV)
-------------------------------

The own-prefill term alone is not the wait a newcomer faces. With prefills
serialised (``local_max_concurrent_prefills: 1``) an in-flight prefill
occupies the gate first, so the true wait is its own prefill **plus** the
remaining service of the gate ahead of it::

    projected_wait = eta_prefill * (busy_slots + 1)

``projected_wait_seconds`` is the single source of truth for that number: the
admission guard compares it against the budget, and ``status_estimate``
publishes it as ``queue_eta_seconds``, so the verdict a dispatcher paces
against can never drift from the verdict the guard enforces. Measuring the
own term alone admitted waits that the proxy's own published
``queue_eta_seconds`` already flagged as unservable (2026-10-21 window:
302 queued waits -> 261 fell back after queueing, 215 at the full 120 s
budget, 25,804 s of aggregate wasted wait).

The occupancy term is also the floor used when per-slot prefill progress is
unobservable (``prefill_progress_unobservable``, ``alive=False``): an
in-flight prefill is treated as occupying the gate for at least the
calibrated estimate, never as "free soon". The change is fail-open by
construction — it can only refuse admits that were already guaranteed to
lose, never admit more than before.

Occupant-evidence pricing (LP-0MVOFVCA2008DQVE)
-----------------------------------------------

The symmetric ``eta * busy_slots`` term prices an occupant at the
*newcomer's* own prefill, which is structurally wrong for heterogeneous
traffic: a small warm prompt behind a huge cold prefill was priced at a few
seconds and admitted, then cancelled by ``local_prefill_deadline_seconds``
with zero prefill progress (measured 2026-10-25: own ETA 7.1 s, elapsed
120.1 s, ``progress=None alive=False in_flight=True``). The occupancy term
therefore also accepts the **measured remaining service** of the in-flight
prefills (each occupant's calibrated ETA minus its elapsed run time, from its
live dispatch record) and both bounds — the queue wait budget *and* the
prefill deadline — are evaluated against the projected time to first byte.
Measured evidence can only raise the symmetric floor, never lower it, so a
missing record keeps today's behaviour byte-for-byte.

Calibration is data-driven: every local dispatch records a sample
``(coded_tokens, new_tokens, dispatch_to_first_byte_ms)`` from the existing
``dispatch_first_byte_ms`` log line (service time, i.e. excluding the queue
wait), keyed by ``(model, context bucket)``, in a bounded rolling window. A
two-coefficient least-squares fit produces the per-bucket coefficients. When
a bucket has too few samples the guard is **inert** — it admits, i.e. today's
queueing behaviour is unchanged — so the guard can never reject on a guess.
Fitted coefficients pass through a conservatism factor (an over-estimate
prefers the cheap-but-fast remote call over a futile 120 s wait) and are
clamped to plausible bounds.

Conservative pricing for large prompts (LP-0MVKUKF65008D5UN)
---------------------------------------------------------------

The point fit is *optimistic* for large prompts: observed first-byte times for
this workload span p50 16 s … p90 189 s … max 591 s, and a 19,940-token
prompt was once modelled at 19.9 s while a 41,682-token prompt actually took
~86 s to first byte. For prompts >= ``CONSERVATIVE_MIN_TOKENS`` (32k) the
projected prefill is therefore priced from a conservative percentile of the
observed ``dispatch_first_byte_ms`` in the ``(model, bucket)`` window
(default p75, ``local_admission_guard_eta_percentile``) instead of the point
estimate alone, and never below the window's measured p50. Combined with the
C2 deadline bound (``local_prefill_deadline_seconds``, LP-0MVKUKEU80022C3U)
an infeasible large prompt is refused at admission instead of starting a
prefill the deadline will cancel. Setting the percentile to 0 restores the
point estimate byte-for-byte (documented rollback).

Rollout control
---------------

``server.local_admission_guard_policy``:

- ``reject`` (default) — refuse admission when ETA exceeds the budget.
- ``observe`` — log the decision (``decision=admit|reject``) but never
  refuse; use this to validate calibration on live traffic before enforcing.
- ``off`` — do not evaluate the guard at all (today's behaviour).

The published estimate (``prefill_eta_seconds`` / ``queue_eta_seconds`` on
``/llama/local/status``) is computed from the same coefficients so dispatchers
can pace themselves *before* launching a session.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger("llama-proxy")

# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

POLICY_REJECT = "reject"
POLICY_OBSERVE = "observe"
POLICY_OFF = "off"
VALID_POLICIES = (POLICY_REJECT, POLICY_OBSERVE, POLICY_OFF)
DEFAULT_POLICY = POLICY_REJECT

# ---------------------------------------------------------------------------
# Defaults (evidence 2026-10-19/20 window: 254 new tok/s solitary ≈ 3.9 ms per
# fresh token; a cache-served prefix costs a small fraction of that, 0.5 ms).
#
# NOTE ON THE 1.1 ms FIGURE in the intake brief: applying 1.1 ms to *cached*
# tokens rejects the AC5 case (a 90 %-cached 100k prompt costs
# 0.9*100k*1.1 ms + 0.1*100k*3.9 ms = 138 s > the 120 s budget). The cached
# term is therefore calibrated from warm dispatches with a cheap default,
# which keeps cache-warm prompts admissible while still rejecting the
# cache-cold 100k prompt (390 s > 120 s).
# ---------------------------------------------------------------------------

DEFAULT_CODED_MS_PER_TOKEN = 0.5
DEFAULT_PREFILL_MS_PER_TOKEN = 3.9

#: Conservative ETA pricing (LP-0MVKUKF65008D5UN): percentile of the observed
#: ``dispatch_first_byte_ms`` window that large prompts are priced at.
#: 0 disables the percentile pricing (documented rollback — the estimator is
#: byte-for-byte the point estimate).
DEFAULT_ETA_PERCENTILE = 75.0
ETA_PERCENTILE_KEY = "local_admission_guard_eta_percentile"

#: Prompts >= this many tokens ("large prompt") are priced at the configured
#: percentile of observed first-byte times, never below the bucket's measured
#: p50 (AC2). Smaller prompts keep the point estimate exactly (small-prompt
#: behaviour unchanged, AC3).
CONSERVATIVE_MIN_TOKENS = 32_768

#: Minimum samples in a ``(model, bucket)`` window before the guard may
#: reject. Below this the guard is inert (today's behaviour is unchanged).
MIN_CALIBRATION_SAMPLES = 5

#: Rolling-window size per ``(model, bucket)``.
WINDOW_SAMPLES = 200

#: Reference prompt size used for the published status estimate when no
#: dispatch has been observed yet for the current model.
DEFAULT_REFERENCE_TOKENS = 32_000

#: Applied to *fitted* coefficients: over-estimating the prefill makes the
#: guard prefer the fast remote call, which is the safe direction.
CONSERVATISM_FACTOR = 1.2

# Clamps applied to fitted coefficients (fail-safe against outliers).
MIN_CODED_MS_PER_TOKEN = 0.05
MAX_CODED_MS_PER_TOKEN = 3.0
MIN_PREFILL_MS_PER_TOKEN = 1.0
MAX_PREFILL_MS_PER_TOKEN = 30.0

# Default local-prefill deadline (seconds).  0 disables the bound.
#: When the projected first-byte ETA is >= this value the guard refuses
#: local dispatch so we never start a prefill the deadline will later
#: cancel (LP-0MVKUKEU80022C3U).
DEFAULT_PREFILL_DEADLINE_SECONDS = 90
PREFILL_DEADLINE_KEY = "local_prefill_deadline_seconds"

#: Upper bounds of the context buckets: <16k, 16k-64k, 64k-128k, >=128k.
CONTEXT_BUCKET_BOUNDS = (16_384, 65_536, 131_072)

# ---------------------------------------------------------------------------
# Cold-prefill admission budget (LP-0MVN3KMGI009BZ0D / LP-0MVN4LBEQ005QRVL)
# ---------------------------------------------------------------------------

#: Hard cap on the NEW (to-be-prefilled) tokens admitted for a local dispatch.
#: 0 disables the cap (the documented rollback / legacy default), so a config
#: without the key behaves exactly as before.
COLD_PREFILL_BUDGET_KEY = "local_cold_prefill_budget_tokens"
DEFAULT_COLD_PREFILL_BUDGET_TOKENS = 0

#: Prompt size from which conservative percentile pricing engages. Defaults to
#: the pre-existing hard-coded floor (``CONSERVATIVE_MIN_TOKENS``, 32 768) so
#: an absent key is byte-for-byte today's behaviour.
ETA_PERCENTILE_MIN_TOKENS_KEY = "local_admission_guard_eta_percentile_min_tokens"
DEFAULT_ETA_PERCENTILE_MIN_TOKENS = CONSERVATIVE_MIN_TOKENS

#: Fraction of the prefill deadline a budgeted prompt's conservative ETA must
#: fit inside (0.95 of 120 s = 114 s). 0.0 disables the bound and leaves the
#: bare token budget (legacy/rollback).
BUDGET_SAFETY_FRACTION_KEY = "local_admission_guard_budget_safety_fraction"
DEFAULT_BUDGET_SAFETY_FRACTION = 0.0

#: Client-visible queue deadline (seconds) the prefill deadline must stay
#: strictly below, so the proxy answers before the client gives up.
CLIENT_QUEUE_DEADLINE_SECONDS = 900.0

# ---------------------------------------------------------------------------
# Rolling calibration window (module scope = one proxy process)
# ---------------------------------------------------------------------------

#: ``(model, bucket) -> deque[(coded_tokens, new_tokens, service_ms)]``
_samples: dict[tuple[str, int], deque] = {}
#: ``model -> prompt_tokens`` of the most recent observed dispatch.
_last_prompt_tokens: dict[str, int] = {}


def reset() -> None:
    """Clear all calibration state (test helper / config reload)."""
    _samples.clear()
    _last_prompt_tokens.clear()


def bucket_for(prompt_tokens: int) -> int:
    """Return the context-bucket index for *prompt_tokens*.

    Buckets are the intervals delimited by :data:`CONTEXT_BUCKET_BOUNDS`;
    prefill cost per token grows with context length, so coefficients are
    fitted per bucket rather than per model alone.
    """
    try:
        tokens = int(prompt_tokens or 0)
    except (TypeError, ValueError):
        tokens = 0
    for index, bound in enumerate(CONTEXT_BUCKET_BOUNDS):
        if tokens < bound:
            return index
    return len(CONTEXT_BUCKET_BOUNDS)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _server_config(config: dict | None) -> dict:
    """Return the ``server`` sub-config (tolerating a flat dict)."""
    if not isinstance(config, dict):
        return {}
    server_cfg = config.get("server", config)
    return server_cfg if isinstance(server_cfg, dict) else {}


def _float_setting(server_cfg: dict, key: str, default: float) -> float:
    raw = server_cfg.get(key, default)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value) or value < 0:
        return float(default)
    return value


def _int_setting(server_cfg: dict, key: str, default: int) -> int:
    return int(_float_setting(server_cfg, key, float(default)))


def _warn_malformed(server_cfg: dict, key: str, default: object) -> None:
    """Warn that a present-but-unusable setting fell back to *default*.

    ``_float_setting`` is deliberately silent (it is on the hot path); the
    budget knobs are startup-visible so a typo must be diagnosable rather
    than silently inert (LP-0MVN4LBEQ005QRVL AC2).
    """
    raw = server_cfg.get(key)
    if raw is None:
        return
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float("nan")
    if not math.isfinite(value) or value < 0:
        logger.warning(
            "Invalid %s=%r — using %r (expected a finite number >= 0)",
            key,
            raw,
            default,
        )


@dataclass(frozen=True)
class GuardConfig:
    """Resolved ``server.local_admission_guard_*`` settings."""

    policy: str
    min_samples: int
    window_samples: int
    reference_tokens: int
    #: Projected-first-byte deadline (seconds); 0 disables the bound.
    prefill_deadline_seconds: float = float(DEFAULT_PREFILL_DEADLINE_SECONDS)
    #: Percentile of the observed first-byte window large prompts are priced
    #: at (LP-0MVKUKF65008D5UN); 0 disables conservative pricing.
    eta_percentile: float = DEFAULT_ETA_PERCENTILE
    #: Hard cap on NEW (to-be-prefilled) tokens admitted locally; 0 disables
    #: (LP-0MVN4LBEQ005QRVL, enforced by LP-0MVN4LBZ500281Z7).
    cold_prefill_budget_tokens: int = DEFAULT_COLD_PREFILL_BUDGET_TOKENS
    #: Prompt size from which percentile pricing engages (default 32 768).
    eta_percentile_min_tokens: int = DEFAULT_ETA_PERCENTILE_MIN_TOKENS
    #: Fraction of the deadline a budgeted prompt's conservative ETA must fit
    #: inside (0.95 x 120 s = 114 s); 0.0 disables the bound.
    budget_safety_fraction: float = DEFAULT_BUDGET_SAFETY_FRACTION


def resolve_config(config: dict | None) -> GuardConfig:
    """Resolve the guard settings from the merged server config (fail-open)."""
    server_cfg = _server_config(config)
    raw_policy = str(
        server_cfg.get("local_admission_guard_policy", DEFAULT_POLICY)
        or DEFAULT_POLICY
    ).strip().lower()
    if raw_policy not in VALID_POLICIES:
        logger.warning(
            "Invalid local_admission_guard_policy=%r — using %r (valid: %s)",
            server_cfg.get("local_admission_guard_policy"),
            DEFAULT_POLICY,
            ", ".join(VALID_POLICIES),
        )
        raw_policy = DEFAULT_POLICY
    min_samples = max(1, _int_setting(
        server_cfg, "local_admission_guard_min_samples", MIN_CALIBRATION_SAMPLES,
    ))
    window = max(
        min_samples,
        _int_setting(
            server_cfg, "local_admission_guard_window_samples", WINDOW_SAMPLES,
        ),
    )
    reference = max(1, _int_setting(
        server_cfg, "local_admission_guard_reference_tokens",
        DEFAULT_REFERENCE_TOKENS,
    ))
    deadline = _float_setting(
        server_cfg, PREFILL_DEADLINE_KEY, float(DEFAULT_PREFILL_DEADLINE_SECONDS),
    )
    percentile = _float_setting(
        server_cfg, ETA_PERCENTILE_KEY, float(DEFAULT_ETA_PERCENTILE),
    )
    if not 0.0 <= percentile <= 100.0:
        logger.warning(
            "Invalid %s=%r — using %r (valid: 0.0-100.0; 0 disables)",
            ETA_PERCENTILE_KEY,
            server_cfg.get(ETA_PERCENTILE_KEY),
            DEFAULT_ETA_PERCENTILE,
        )
        percentile = DEFAULT_ETA_PERCENTILE
    _warn_malformed(
        server_cfg, COLD_PREFILL_BUDGET_KEY, DEFAULT_COLD_PREFILL_BUDGET_TOKENS
    )
    _warn_malformed(
        server_cfg,
        ETA_PERCENTILE_MIN_TOKENS_KEY,
        DEFAULT_ETA_PERCENTILE_MIN_TOKENS,
    )
    _warn_malformed(
        server_cfg, BUDGET_SAFETY_FRACTION_KEY, DEFAULT_BUDGET_SAFETY_FRACTION
    )
    budget_tokens = max(0, _int_setting(
        server_cfg, COLD_PREFILL_BUDGET_KEY, DEFAULT_COLD_PREFILL_BUDGET_TOKENS,
    ))
    percentile_min_tokens = max(1, _int_setting(
        server_cfg,
        ETA_PERCENTILE_MIN_TOKENS_KEY,
        DEFAULT_ETA_PERCENTILE_MIN_TOKENS,
    ))
    safety_fraction = _float_setting(
        server_cfg, BUDGET_SAFETY_FRACTION_KEY, DEFAULT_BUDGET_SAFETY_FRACTION,
    )
    if safety_fraction > 1.0:
        logger.warning(
            "Invalid %s=%r — using %r (valid: 0.0-1.0; 0 disables)",
            BUDGET_SAFETY_FRACTION_KEY,
            server_cfg.get(BUDGET_SAFETY_FRACTION_KEY),
            DEFAULT_BUDGET_SAFETY_FRACTION,
        )
        safety_fraction = DEFAULT_BUDGET_SAFETY_FRACTION
    if budget_tokens > 0 and percentile_min_tokens > budget_tokens:
        logger.warning(
            "%s=%d exceeds %s=%d — percentile pricing would never engage, so "
            "the band guard is silently dead; lower the percentile floor or "
            "raise the budget",
            ETA_PERCENTILE_MIN_TOKENS_KEY,
            percentile_min_tokens,
            COLD_PREFILL_BUDGET_KEY,
            budget_tokens,
        )
    return GuardConfig(
        policy=raw_policy,
        min_samples=min_samples,
        window_samples=window,
        reference_tokens=reference,
        prefill_deadline_seconds=deadline,
        eta_percentile=percentile,
        cold_prefill_budget_tokens=budget_tokens,
        eta_percentile_min_tokens=percentile_min_tokens,
        budget_safety_fraction=safety_fraction,
    )


def validate_prefill_timeout_ordering(config: dict | None) -> list[str]:
    """Return FATAL problems with the local prefill timeout ordering.

The admitted-request lifetime must be strictly increasing so each bound fires
before the next one and all of them fire before the client gives up
(LP-0MVN4LBEQ005QRVL AC3)::

        contention_queue_max_wait_seconds (60)
          < local_prefill_deadline_seconds (120)
          < local_dispatch_first_byte_timeout_seconds (840)
          < local_dispatch_max_prefill_seconds (900)
          < client queue deadline (900 s)

A violating configuration is a startup FATAL: a deadline below the admission
budget cuts off requests the guard legitimately admitted, and a deadline at or
above the client's own deadline loses the race it exists to win. A zero
(disabled) value is skipped rather than ordered — ``0`` means "no bound".
"""
    server_cfg = _server_config(config)
    queue_budget = _float_setting(
        server_cfg, "contention_queue_max_wait_seconds", 60.0
    )
    deadline = _float_setting(
        server_cfg, PREFILL_DEADLINE_KEY, float(DEFAULT_PREFILL_DEADLINE_SECONDS)
    )
    first_byte = _float_setting(
        server_cfg, "local_dispatch_first_byte_timeout_seconds", 840.0
    )
    max_prefill = _float_setting(
        server_cfg, "local_dispatch_max_prefill_seconds", 900.0
    )

    problems: list[str] = []
    if deadline > 0 and queue_budget > 0 and queue_budget >= deadline:
        problems.append(
            "FATAL: contention_queue_max_wait_seconds="
            f"{queue_budget:g} must be < local_prefill_deadline_seconds="
            f"{deadline:g} (an admitted request must not be cut off before "
            "its wait budget expires)"
        )
    if deadline > 0 and first_byte > 0 and deadline >= first_byte:
        problems.append(
            "FATAL: local_prefill_deadline_seconds="
            f"{deadline:g} must be < "
            "local_dispatch_first_byte_timeout_seconds="
            f"{first_byte:g} (the deadline must be the bound that fires)"
        )
    if first_byte > 0 and max_prefill > 0 and first_byte > max_prefill:
        problems.append(
            "FATAL: local_dispatch_first_byte_timeout_seconds="
            f"{first_byte:g} must be <= local_dispatch_max_prefill_seconds="
            f"{max_prefill:g} (a record cannot outlive its lifetime wedge)"
        )
    if deadline > 0 and deadline >= CLIENT_QUEUE_DEADLINE_SECONDS:
        problems.append(
            "FATAL: local_prefill_deadline_seconds="
            f"{deadline:g} must be < the client queue deadline "
            f"{CLIENT_QUEUE_DEADLINE_SECONDS:g} s (otherwise the proxy races "
            "the client instead of answering first)"
        )
    return problems


# ---------------------------------------------------------------------------
# Sample recording + calibration
# ---------------------------------------------------------------------------

def record_dispatch_sample(
    model: str | None,
    prompt_tokens: int,
    cached_ratio: float,
    service_ms: float,
) -> None:
    """Record one observed local prefill (rolling calibration window).

    Called from the local streaming loop where ``dispatch_first_byte_ms`` is
    logged. *service_ms* is ``dispatch_to_first_byte_ms`` (queue wait already
    removed) so the fit describes prefill cost, not contention.

    Never raises: a metric path must not break a stream. Invalid inputs
    (missing model, non-positive tokens/time, non-finite values) are ignored.
    """
    try:
        model_key = str(model or "").strip()
        if not model_key:
            return
        tokens = int(prompt_tokens or 0)
        if tokens <= 0:
            return
        ms = float(service_ms)
        if not math.isfinite(ms) or ms <= 0:
            return
        try:
            ratio = float(cached_ratio)
        except (TypeError, ValueError):
            ratio = 0.0
        if not math.isfinite(ratio):
            ratio = 0.0
        ratio = min(1.0, max(0.0, ratio))
        coded = int(tokens * ratio)
        new = max(0, tokens - coded)
        bucket = bucket_for(tokens)
        window = _samples.get((model_key, bucket))
        if window is None:
            window = deque(maxlen=WINDOW_SAMPLES)
            _samples[(model_key, bucket)] = window
        window.append((coded, new, ms))
        _last_prompt_tokens[model_key] = tokens
    except Exception:  # pragma: no cover - defensive
        logger.debug("record_dispatch_sample failed", exc_info=True)


@dataclass(frozen=True)
class Coefficients:
    """Fitted (or default) per-token prefill coefficients, in milliseconds.

    ``eta_percentile`` / ``percentile_ms`` / ``floor_ms`` / ``uplift`` are
    the conservative-pricing diagnostics (LP-0MVKUKF65008D5UN): for large
    prompts (>= ``CONSERVATIVE_MIN_TOKENS``) the coefficients are uplifted so
    the reference prompt prices at the configured percentile (never below the
    window's p50); ``percentile_ms``/``floor_ms`` are those observed
    first-byte thresholds (``None`` when conservative pricing did not apply)
    and ``uplift`` is 1.0 when the point estimate already dominates.
    """

    coded_ms_per_token: float
    prefill_ms_per_token: float
    samples: int
    calibrated: bool
    eta_percentile: float = 0.0
    percentile_ms: float | None = None
    floor_ms: float | None = None
    uplift: float = 1.0


def _solve(
    sum_cc: float, sum_cn: float, sum_nn: float, sum_cm: float, sum_nm: float
) -> tuple[float, float] | None:
    """Least-squares solve of ``a*coded + b*new ~ ms``; None when singular."""
    det = sum_cc * sum_nn - sum_cn * sum_cn
    scale = max(sum_cc * sum_nn, 1e-9)
    if abs(det) <= 1e-6 * scale:
        return None
    a = (sum_cm * sum_nn - sum_nm * sum_cn) / det
    b = (sum_cc * sum_nm - sum_cn * sum_cm) / det
    return a, b


def _fit(samples: list[tuple[int, int, float]]) -> tuple[float | None, float | None] | None:
    """Fit ``(coded_ms, prefill_ms)`` from *samples*, or None when impossible.

    A coefficient that the window cannot identify (every sample cache-cold, or
    every sample fully cache-warm) is returned as ``None`` so the caller keeps
    the sane default *unscaled* for it.
    """
    sum_cc = sum_cn = sum_nn = sum_cm = sum_nm = 0.0
    for coded, new, ms in samples:
        sum_cc += coded * coded
        sum_cn += coded * new
        sum_nn += new * new
        sum_cm += coded * ms
        sum_nm += new * ms
    solved = _solve(sum_cc, sum_cn, sum_nn, sum_cm, sum_nm)
    if solved is not None:
        return solved
    # Degenerate window: only one coefficient is identifiable.
    if sum_nn > 0 and sum_cc <= 0:
        return (None, sum_nm / sum_nn)
    if sum_cc > 0 and sum_nn <= 0:
        return (sum_cm / sum_cc, None)
    return None


def _clamp(value: float, low: float, high: float) -> float:
    if not math.isfinite(value):
        return low
    return min(high, max(low, value))


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated *pct* (0-100) of *values* (sorted ascending)."""
    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return float(values[0])
    idx = (pct / 100.0) * (n - 1)
    lower = min(n - 1, int(math.floor(idx)))
    upper = min(n - 1, lower + 1)
    frac = max(0.0, min(1.0, idx - lower))
    return values[lower] * (1.0 - frac) + values[upper] * frac


def _median(values: list[float]) -> float:
    """Median of *values* (the p50 percentile of the sorted window)."""
    return _percentile(sorted(values), 50.0)


def coefficients(model: str | None, prompt_tokens: int, config: dict | None = None
                 ) -> Coefficients:
    """Return the coefficients to use for *model* at *prompt_tokens*.

    Falls back to the sane defaults (``calibrated=False``) when the bucket has
    fewer than ``min_samples`` observations or the fit is unusable — in which
    case the guard stays inert and today's behaviour is preserved.

    For large prompts (``prompt_tokens >= CONSERVATIVE_MIN_TOKENS``) with a
    configured ``eta_percentile > 0`` the fitted coefficients are uplifted so
    the bucket's reference prompt prices at that percentile of the observed
    first-byte times (never below its measured p50) — the conservative-ETA
    change (LP-0MVKUKF65008D5UN). Smaller prompts and ``eta_percentile: 0``
    (rollback) return the un-uplifted point estimate byte-for-byte.
    """
    guard_cfg = resolve_config(config)
    model_key = str(model or "").strip()
    bucket = bucket_for(prompt_tokens)
    window = _samples.get((model_key, bucket))
    samples = list(window) if window else []
    if len(samples) < guard_cfg.min_samples:
        return Coefficients(
            coded_ms_per_token=DEFAULT_CODED_MS_PER_TOKEN,
            prefill_ms_per_token=DEFAULT_PREFILL_MS_PER_TOKEN,
            samples=len(samples),
            calibrated=False,
            eta_percentile=guard_cfg.eta_percentile,
        )
    fitted = _fit(samples)
    if fitted is None:
        return Coefficients(
            coded_ms_per_token=DEFAULT_CODED_MS_PER_TOKEN,
            prefill_ms_per_token=DEFAULT_PREFILL_MS_PER_TOKEN,
            samples=len(samples),
            calibrated=False,
            eta_percentile=guard_cfg.eta_percentile,
        )
    fitted_coded, fitted_prefill = fitted
    coded_ms = (
        fitted_coded * CONSERVATISM_FACTOR
        if fitted_coded is not None
        else DEFAULT_CODED_MS_PER_TOKEN
    )
    prefill_ms = (
        fitted_prefill * CONSERVATISM_FACTOR
        if fitted_prefill is not None
        else DEFAULT_PREFILL_MS_PER_TOKEN
    )
    coded_ms = _clamp(
        coded_ms,
        MIN_CODED_MS_PER_TOKEN,
        MAX_CODED_MS_PER_TOKEN,
    )
    prefill_ms = _clamp(
        prefill_ms,
        MIN_PREFILL_MS_PER_TOKEN,
        MAX_PREFILL_MS_PER_TOKEN,
    )
    # Conservative pricing for large prompts (LP-0MVKUKF65008D5UN): cost the
    # projected prefill from a conservative percentile of the observed
    # first-byte times rather than the optimistic point estimate. The uplift is
    # anchored to the bucket's MEDIAN sample (``ref_coded``/``ref_new``): the
    # median prompt priced this way lands exactly on the percentile threshold,
    # and a prompt at that size never prices below the window's p50. An
    # identical-sample window (p50 == p75 == point estimate) yields uplift 1.0,
    # so calibration windows without variance are byte-for-byte unchanged.
    percentile_ms = None
    floor_ms = None
    uplift = 1.0
    if (
        guard_cfg.eta_percentile > 0.0
        and prompt_tokens >= guard_cfg.eta_percentile_min_tokens
    ):
        ref_coded = _median([float(c) for c, _, _ in samples])
        ref_new = _median([float(n) for _, n, _ in samples])
        pred_ref_ms = ref_coded * coded_ms + ref_new * prefill_ms
        service = sorted(float(ms) for _, _, ms in samples)
        percentile_ms = _percentile(service, guard_cfg.eta_percentile)
        floor_ms = _percentile(service, 50.0)
        target_ms = max(percentile_ms, floor_ms)
        if pred_ref_ms > 0.0 and target_ms > pred_ref_ms:
            uplift = target_ms / pred_ref_ms
            coded_ms *= uplift
            prefill_ms *= uplift
            coded_ms = _clamp(
                coded_ms,
                MIN_CODED_MS_PER_TOKEN,
                MAX_CODED_MS_PER_TOKEN,
            )
            prefill_ms = _clamp(
                prefill_ms,
                MIN_PREFILL_MS_PER_TOKEN,
                MAX_PREFILL_MS_PER_TOKEN,
            )
    return Coefficients(
        coded_ms_per_token=coded_ms,
        prefill_ms_per_token=prefill_ms,
        samples=len(samples),
        calibrated=True,
        eta_percentile=guard_cfg.eta_percentile,
        percentile_ms=percentile_ms,
        floor_ms=floor_ms,
        uplift=uplift,
    )


def estimate_prefill_seconds(
    model: str | None,
    prompt_tokens: int,
    cached_ratio: float = 0.0,
    config: dict | None = None,
) -> tuple[float, Coefficients]:
    """Estimated prefill seconds for *prompt_tokens* and its cache state.

    Returns ``(seconds, coefficients)``. Coefficients come from the rolling
    per-``(model, bucket)`` calibration when available, otherwise from the
    defaults (``coefficients.calibrated is False``). For large prompts the
    estimate is priced conservatively (see :func:`coefficients`): at or above
    the configured percentile of observed first-byte times, and never below
    the window's p50 (LP-0MVKUKF65008D5UN).
    """
    try:
        tokens = max(0, int(prompt_tokens or 0))
    except (TypeError, ValueError):
        tokens = 0
    try:
        ratio = min(1.0, max(0.0, float(cached_ratio)))
    except (TypeError, ValueError):
        ratio = 0.0
    coeffs = coefficients(model, tokens, config)
    coded = tokens * ratio
    new = tokens - coded
    seconds = (
        coded * coeffs.coded_ms_per_token + new * coeffs.prefill_ms_per_token
    ) / 1000.0
    return max(0.0, seconds), coeffs


# ---------------------------------------------------------------------------
# Occupancy-aware projected wait (LP-0MVICD2YW004DJPV)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectedWait:
    """The settlement of one newcomer's wait against the occupied gate.

    ``own_prefill_seconds`` is the newcomer's own calibrated prefill;
    ``occupancy_seconds`` is the service the already-occupied prefill gate
    owes before the newcomer can start; ``projected_seconds`` is their sum —
    the quantity ``/llama/local/status`` publishes as ``queue_eta_seconds``
    (``eta * (busy_slots + 1)``) and the *only* number the admission guard may
    compare against a wait budget (LP-0MVICD2YW004DJPV).
    """

    own_prefill_seconds: float
    occupancy_seconds: float
    projected_seconds: float
    busy_slots: int
    coefficients: Coefficients


def projected_wait_seconds(
    model: str | None,
    prompt_tokens: int,
    cached_ratio: float = 0.0,
    busy_slots: int = 0,
    config: dict | None = None,
    occupant_prefill_seconds: float | None = None,
) -> ProjectedWait:
    """Projected pre-gate wait for a newcomer behind *busy_slots* occupants.

    Single source of truth for the occupancy-aware wait: the admission guard
    (this module) and the published ``queue_eta_seconds`` (``status_estimate``)
    both call it, so the verdict a client is paced against can never drift
    from the verdict the guard enforces (LP-0MVICD2YW004DJPV AC2).

    The occupancy term is the calibrated prefill repeated once per occupied
    gate slot. That is also the *floor* used when per-slot prefill progress is
    unobservable (``prefill_progress_unobservable``, ``alive=False``): an
    in-flight prefill must be treated as occupying the gate for at least the
    calibrated estimate, never as "free soon" (AC4).

    *occupant_prefill_seconds* is the **measured** remaining service owed by
    the in-flight prefills, projected from each occupant's own admission
    evidence (LP-0MVOFVCA2008DQVE). It can only raise the occupancy term above
    the symmetric floor, never lower it: pricing the occupant at the
    newcomer's own ETA is structurally wrong when a small warm prompt queues
    behind a huge cold prefill (measured 2026-10-25: an own ETA of 7.1 s
    waited 120 s behind a single occupant and was cancelled at the deadline).
    ``None``/unparseable values leave the floor verdict byte-for-byte
    unchanged, so every caller that does not probe the live records keeps
    today's behaviour.
    """
    eta, coeffs = estimate_prefill_seconds(model, prompt_tokens, cached_ratio, config)
    try:
        busy = max(0, int(busy_slots or 0))
    except (TypeError, ValueError):
        busy = 0
    occupancy = eta * busy
    try:
        measured_occupancy = max(0.0, float(occupant_prefill_seconds or 0.0))
    except (TypeError, ValueError):
        measured_occupancy = 0.0
    occupancy = max(occupancy, measured_occupancy)
    return ProjectedWait(
        own_prefill_seconds=eta,
        occupancy_seconds=occupancy,
        projected_seconds=eta + occupancy,
        busy_slots=busy,
        coefficients=coeffs,
    )


# ---------------------------------------------------------------------------
# Admission decision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdmissionDecision:
    """Outcome of the prefill-ETA admission check for one request."""

    admit: bool
    reason: str
    eta_prefill_seconds: float | None
    budget_seconds: float | None
    policy: str
    calibrated: bool
    samples: int
    #: Occupancy-aware projection (LP-0MVICD2YW004DJPV). ``None`` on an inert
    #: decision that never costed the prompt (``policy_off`` / ``no_budget``).
    projected_wait_seconds: float | None = None
    eta_occupancy_seconds: float | None = None
    busy_slots: int = 0
    in_flight_prefills: int = 0
    #: Projected first-byte deadline (seconds) used by this decision.
    prefill_deadline_seconds: float = 0.0
    #: New (uncached, to-be-prefilled) tokens priced by this decision
    #: (LP-0MVN4LBZ500281Z7).
    new_tokens: int = 0
    #: Configured cold-prefill budget in new tokens (0 = cap disabled).
    budget_tokens: int = 0
    #: ``budget_safety_fraction * prefill_deadline_seconds`` — the ETA a
    #: budgeted prompt must fit inside (0 = bound disabled).
    deadline_budget_seconds: float = 0.0

    @property
    def active(self) -> bool:
        """True when the guard actually made a decision (not inert)."""
        return self.reason in (
            "within_budget",
            "eta_exceeds_budget",
            "prefill_eta_exceeds_deadline",
            "prefill_budget_exceeded",
            "prefill_eta_exceeds_safety_budget",
        )


def wait_budget_seconds(config: dict | None, priority: str | None = None) -> float:
    """The caller's local wait budget (seconds).

    ``high`` priority never queues, so its budget is 0 (the guard is inert for
    it — the pre-existing "never queue" behaviour is preserved). Otherwise the
    budget is ``contention_queue_max_wait_seconds`` resolved through the
    canonical queue config (so clamps and warnings match the queue).
    """
    try:
        from proxy.priority import PRIORITY_HIGH, parse_priority

        if parse_priority(priority) == PRIORITY_HIGH:
            return 0.0
    except Exception:
        pass
    try:
        from proxy.router import _get_contention_queue_config

        return float(
            _get_contention_queue_config(_server_config(config))["max_wait_seconds"]
        )
    except Exception:
        return 0.0


def evaluate_admission(
    config: dict | None,
    model: str | None,
    prompt_tokens: int,
    cached_ratio: float,
    priority: str | None = None,
    *,
    busy_slots: int = 0,
    in_flight_prefills: int = 0,
    occupant_prefill_seconds: float | None = None,
) -> AdmissionDecision:
    """Decide whether a local dispatch can finish inside the wait budget.

    Two independent bounds, both enforced when the policy is ``reject``:

    1. **Budget bound (LP-0MVGDUV5E005UWBQ).** The *projected wait*
       (own prefill plus the remaining service of the occupied prefill
       gate, :func:`projected_wait_seconds`) is compared against the wait
       budget (``contention_queue_max_wait_seconds``).

    2. **Deadline bound (LP-0MVKUKEU80022C3U).** The projected time to first
       byte — own-prefill ETA plus the occupied gate — is compared against
       ``local_prefill_deadline_seconds`` (default 120 s). When it would
       exceed the deadline the guard refuses so we never start a dispatch
       that the deadline will later cancel. Including the occupancy term is
       the LP-0MVOFVCA2008DQVE residual fix: a warm 7 s prompt admitted
       behind a 120 s occupant was cancelled with zero prefill progress.

    *busy_slots* is the number of occupied gate slots ahead of this request
    (``max(generating concurrency, prefills in flight)``); *in_flight_prefills*
    is carried for the decision log so an unobservable-progress gate is
    diagnosable (AC4/AC6). *occupant_prefill_seconds* is the measured
    remaining prefill service read from the occupants' own dispatch records
    (raises, never lowers, the symmetric occupancy floor).

    Fail-open by construction: the guard is inert (``admit=True``) when the
    policy is ``off``, the priority never queues, the bucket is not yet
    calibrated, or no budget can be resolved. Only a calibrated projected
    wait that exceeds a bound refuses admission, and only when the policy is
    ``reject`` (``observe`` logs the same verdict without refusing).
    """
    guard_cfg = resolve_config(config)

    try:
        ratio = min(1.0, max(0.0, float(cached_ratio or 0.0)))
        new_tokens = max(0, int(round(float(prompt_tokens or 0) * (1.0 - ratio))))
    except (TypeError, ValueError):
        new_tokens = 0

    try:
        occupancy = max(0, int(busy_slots or 0))
    except (TypeError, ValueError):
        occupancy = 0
    try:
        in_flight = max(0, int(in_flight_prefills or 0))
    except (TypeError, ValueError):
        in_flight = 0

    def _inert(reason: str, eta: float | None = None, budget: float | None = None,
               coeffs: Coefficients | None = None,
               projected: ProjectedWait | None = None) -> AdmissionDecision:
        return AdmissionDecision(
            admit=True,
            reason=reason,
            eta_prefill_seconds=eta,
            budget_seconds=budget,
            policy=guard_cfg.policy,
            calibrated=bool(coeffs.calibrated) if coeffs else False,
            samples=coeffs.samples if coeffs else 0,
            projected_wait_seconds=(
                projected.projected_seconds if projected else None
            ),
            eta_occupancy_seconds=(
                projected.occupancy_seconds if projected else None
            ),
            busy_slots=projected.busy_slots if projected else occupancy,
            in_flight_prefills=in_flight,
            prefill_deadline_seconds=guard_cfg.prefill_deadline_seconds,
            new_tokens=new_tokens,
            budget_tokens=guard_cfg.cold_prefill_budget_tokens,
            deadline_budget_seconds=_deadline_budget_seconds(guard_cfg),
        )

    if guard_cfg.policy == POLICY_OFF:
        return _inert("policy_off")
    # Cold-prefill budget (LP-0MVN4LBZ500281Z7, parent AC1): a FLAT cap on the
    # new (uncached) tokens, checked BEFORE every other bound so it can never
    # be skipped — not by an uncalibrated bucket, not by a zero wait budget
    # (``high`` priority), not by the idle/economic bypass. It prices GPU
    # compute, not queue patience, so it is calibration-independent.
    if (
        guard_cfg.cold_prefill_budget_tokens > 0
        and new_tokens > guard_cfg.cold_prefill_budget_tokens
    ):
        return AdmissionDecision(
            admit=guard_cfg.policy == POLICY_OBSERVE,
            reason="prefill_budget_exceeded",
            eta_prefill_seconds=None,
            budget_seconds=wait_budget_seconds(config, priority),
            policy=guard_cfg.policy,
            calibrated=False,
            samples=0,
            projected_wait_seconds=None,
            eta_occupancy_seconds=None,
            busy_slots=occupancy,
            in_flight_prefills=in_flight,
            prefill_deadline_seconds=guard_cfg.prefill_deadline_seconds,
            new_tokens=new_tokens,
            budget_tokens=guard_cfg.cold_prefill_budget_tokens,
            deadline_budget_seconds=_deadline_budget_seconds(guard_cfg),
        )
    budget = wait_budget_seconds(config, priority)
    if budget <= 0:
        return _inert("no_budget")
    projected = projected_wait_seconds(
        model,
        prompt_tokens,
        cached_ratio,
        occupancy,
        config,
        occupant_prefill_seconds=occupant_prefill_seconds,
    )
    coeffs = projected.coefficients
    eta = projected.own_prefill_seconds
    if not coeffs.calibrated:
        return _inert("insufficient_samples", eta, budget, coeffs, projected)
    within = projected.projected_seconds <= budget
    reason = "within_budget" if within else "eta_exceeds_budget"
    admit = within or guard_cfg.policy == POLICY_OBSERVE
    # Second independent bound: deadline check (LP-0MVKUKEU80022C3U).
    # If the projected TIME TO FIRST BYTE (own prefill + the occupied gate
    # ahead of it) reaches the prefill deadline we refuse *now* so we never
    # start a dispatch the deadline will later cancel. The own-prefill term
    # alone is not the wait a newcomer faces: with prefills serialised the
    # occupant's remaining prefill is spent inside the same deadline window
    # (LP-0MVOFVCA2008DQVE residual, measured 2026-10-25).
    # The deadline is the dominant reason when it fires (AC1) and, like
    # the budget bound, honours ``observe`` (log but never refuse).
    deadline = guard_cfg.prefill_deadline_seconds
    if deadline > 0 and projected.projected_seconds >= deadline:
        reason = "prefill_eta_exceeds_deadline"
        admit = guard_cfg.policy == POLICY_OBSERVE
    # Third bound, deadline-RELATIVE (LP-0MVN4LBZ500281Z7 / parent AC3): when a
    # budget is configured, a prompt must fit inside
    # ``budget_safety_fraction * deadline`` (0.95 x 120 s = 114 s), not merely
    # inside the deadline. This is what stops a prompt whose BAND is slow from
    # slipping past the flat token cap priced at the fast band's rate.
    deadline_budget = _deadline_budget_seconds(guard_cfg)
    if deadline_budget > 0.0 and eta > deadline_budget:
        reason = "prefill_eta_exceeds_safety_budget"
        admit = guard_cfg.policy == POLICY_OBSERVE
    return AdmissionDecision(
        admit=admit,
        reason=reason,
        eta_prefill_seconds=eta,
        budget_seconds=budget,
        policy=guard_cfg.policy,
        calibrated=True,
        samples=coeffs.samples,
        projected_wait_seconds=projected.projected_seconds,
        eta_occupancy_seconds=projected.occupancy_seconds,
        busy_slots=projected.busy_slots,
        in_flight_prefills=in_flight,
        prefill_deadline_seconds=deadline,
        new_tokens=new_tokens,
        budget_tokens=guard_cfg.cold_prefill_budget_tokens,
        deadline_budget_seconds=deadline_budget,
    )


def _deadline_budget_seconds(guard_cfg: GuardConfig) -> float:
    """``budget_safety_fraction * prefill_deadline_seconds`` (0 = disabled).

    Single source of truth for the deadline-relative budget: the admission
    guard, the decision log and the rejection headers all read it, so the
    advertised threshold can never drift from the enforced one.
    """
    if guard_cfg.budget_safety_fraction <= 0.0:
        return 0.0
    if guard_cfg.prefill_deadline_seconds <= 0.0:
        return 0.0
    return guard_cfg.budget_safety_fraction * guard_cfg.prefill_deadline_seconds


# ---------------------------------------------------------------------------
# Published estimate for /llama/local/status
# ---------------------------------------------------------------------------

def reference_tokens(model: str | None, config: dict | None = None) -> int:
    """Prompt size to cost for the published estimate.

    The most recent dispatch observed for *model* keeps the published number
    truthful for the traffic actually being served; without one, the
    configured reference (default 32k) is used.
    """
    guard_cfg = resolve_config(config)
    try:
        observed = int(_last_prompt_tokens.get(str(model or "").strip(), 0) or 0)
    except (TypeError, ValueError):
        observed = 0
    return observed if observed > 0 else guard_cfg.reference_tokens


def per_slot_prefill_eta_seconds(
    model: str | None,
    context_tokens: int | None,
    config: dict | None = None,
) -> float | None:
    """Calibrated prefill cost of one slot's loaded context (AC3).

    Published per-slot on ``/llama/local/status`` (LP-0MVGDUVGE0024YAI AC3):
    the single source of truth is the shared calibration this module owns, so
    the status endpoint never grows a second estimator. Returns ``None``
    rather than a misleading ``0`` when the slot's prompt progress is unknown
    (modern llama.cpp builds dropped ``n_past``/``n_prompt_tokens_processed``)
    or not positive — a prefill whose size cannot be observed must not be
    presented as free.
    """
    try:
        tokens = int(context_tokens or 0)
    except (TypeError, ValueError):
        return None
    if tokens <= 0:
        return None
    try:
        seconds, _coeffs = estimate_prefill_seconds(model, tokens, 0.0, config)
    except Exception:  # pragma: no cover - defensive
        return None
    return round(seconds, 1)


def status_estimate(
    config: dict | None,
    model: str | None,
    total_slots: int = 0,
    available_slots: int = 0,
    active_queries: int = 0,
) -> dict:
    """Published prefill/queue ETA for ``/llama/local/status``.

    ``prefill_eta_seconds`` is the cost of prefilling the reference prompt
    (assumed cache-cold — the conservative case). ``queue_eta_seconds`` is
    what a *newcomer* would wait before its first byte: the reference prefill
    repeated for every currently occupied slot, plus its own prefill. It is
    computed by the same :func:`projected_wait_seconds` the admission guard
    uses, so the published number and the enforced verdict cannot drift apart
    (LP-0MVICD2YW004DJPV AC2).

    Fail-open: any error yields ``None`` fields rather than a broken payload.
    """
    try:
        tokens = reference_tokens(model, config)
        try:
            total = int(total_slots or 0)
            available = int(available_slots or 0)
            active = int(active_queries or 0)
        except (TypeError, ValueError):
            total, available, active = 0, 0, 0
        if total > 0:
            busy = max(0, total - available)
        else:
            busy = max(0, active)
        queue_eta = projected_wait_seconds(model, tokens, 0.0, busy, config)
        eta = queue_eta.own_prefill_seconds
        coeffs = queue_eta.coefficients
        # The admission guard's own budget (the contention-queue max wait, 0
        # for high-priority that never queues): the same number a dispatcher
        # must pace a session against (LP-0MVGDUVGE0024YAI AC3). ``None``
        # when no budget is resolvable, never a fabricated ``0``.
        try:
            budget = wait_budget_seconds(config)
        except Exception:  # pragma: no cover - defensive
            budget = 0.0
        budget_published = round(float(budget), 1) if float(budget) > 0 else None
        return {
            "prefill_eta_seconds": round(eta, 1),
            "queue_eta_seconds": round(queue_eta.projected_seconds, 1),
            "queue_eta_occupancy_seconds": round(queue_eta.occupancy_seconds, 1),
            "queue_eta_busy_slots": queue_eta.busy_slots,
            "prefill_eta_reference_tokens": tokens,
            "prefill_eta_samples": coeffs.samples,
            "prefill_eta_calibrated": coeffs.calibrated,
            "prefill_budget_seconds": budget_published,
        }
    except Exception:  # pragma: no cover - defensive
        logger.debug("status_estimate failed", exc_info=True)
        return {
            "prefill_eta_seconds": None,
            "queue_eta_seconds": None,
            "queue_eta_occupancy_seconds": None,
            "queue_eta_busy_slots": None,
            "prefill_eta_reference_tokens": None,
            "prefill_eta_samples": 0,
            "prefill_eta_calibrated": False,
            "prefill_budget_seconds": None,
        }


# ---------------------------------------------------------------------------
# Suggested retry delay for a refused request
# ---------------------------------------------------------------------------

def suggested_retry_after_seconds(
    eta_prefill_seconds: float | None,
    budget_seconds: float | None,
    maximum: int = 60,
) -> int:
    """``Retry-After`` hint (whole seconds, >= 1) for a refused request.

    A refusal means the projected prefill cannot fit the caller's budget; the
    useful hint is how much *longer* the local prefill would take than the
    caller was prepared to wait (rounded up, capped at *maximum*).
    """
    try:
        eta = float(eta_prefill_seconds or 0.0)
        budget = float(budget_seconds or 0.0)
    except (TypeError, ValueError):
        return 1
    delay = int(math.ceil(max(0.0, eta - budget)))
    return max(1, min(int(maximum or 1), delay))


def record_guard_decision(
    decision: AdmissionDecision,
    *,
    provider: str | None = None,
    session_id: str | None = None,
    model: str | None = None,
    priority: str | None = None,
    reason_label: str | None = None,
    free_slot: bool = False,
) -> None:
    """Emit the ``local_admission_guard`` log line for *decision*.

    Only active decisions are logged (an inert guard made no decision, and
    logging every inert request would flood the log). The line is parsed by
    the proxy-usage-analysis skill, which counts admits and refusals
    (AC2).

    The occupancy evidence is part of the line (LP-0MVICD2YW004DJPV AC6): own
    ETA, occupancy term, projected wait, budget, busy slots and in-flight
    prefills, then the verdict — so a refusal (or a surprising admit) is
    diagnosable from the log alone. ``eta_prefill_s`` keeps its historical
    meaning (the own-prefill term) so existing parsers keep working.

    *reason_label* overrides the logged reason so the direct (free-slot)
    dispatch guard (LP-0MVIQ0F8T002FS42) is distinguishable from the
    queue-entry guard: the same decision emitted one step earlier is logged
    ``reason=eta_exceeds_budget_direct`` (with ``free_slot=true``), and the
    line is emitted exactly once — the caller must not log a second copy.
    The deadline refusal (LP-0MVKUKEU80022C3U) keeps its own reason
    (``prefill_eta_exceeds_deadline``) on BOTH paths; ``free_slot=true``
    already identifies the direct path, so the override does not apply.
    """
    # A cold-prefill budget refusal is logged even though it carries no ETA:
    # the cap is calibration-independent, so its evidence is the token counts
    # (AC7, LP-0MVN4LBZ500281Z7), not a priced estimate.
    if not decision.active or (
        decision.eta_prefill_seconds is None
        and decision.reason != "prefill_budget_exceeded"
    ):
        return
    # The logged ``decision`` is the guard's *verdict* (admit/reject); the
    # ``enforced`` flag distinguishes an enforced refusal from an ``observe``
    # rollout, so the usage report can count would-be refusals during a
    # validation window.
    verdict = (
        "reject"
        if str(decision.reason).startswith("eta_exceeds_budget")
        or str(decision.reason).startswith("prefill_eta_exceeds_deadline")
        or str(decision.reason).startswith("prefill_eta_exceeds_safety_budget")
        or str(decision.reason).startswith("prefill_budget_exceeded")
        else "admit"
    )
    projected = (
        decision.projected_wait_seconds
        if decision.projected_wait_seconds is not None
        else decision.eta_prefill_seconds
    )
    if projected is None:
        # A cold-prefill budget refusal carries no priced ETA (the cap is
        # calibration-independent), so the projection falls back to 0.0 and
        # the token evidence carries the decision (LP-0MVN4LBZ500281Z7 AC7).
        projected = 0.0
    # The deadline refusal keeps its own reason on both paths so AC1 holds
    # literally (``reason=prefill_eta_exceeds_deadline``); ``free_slot``
    # already distinguishes the direct dispatch.
    effective_label = reason_label
    if reason_label and str(decision.reason).startswith(
        ("prefill_eta_exceeds_deadline", "prefill_eta_exceeds_safety_budget",
         "prefill_budget_exceeded")
    ):
        effective_label = decision.reason
    logger.info(
        "local_admission_guard decision=%s reason=%s enforced=%s "
        "eta_prefill_s=%.1f eta_occupancy_s=%.1f projected_wait_s=%.1f "
        "budget_s=%.1f deadline_s=%.1f deadline_budget_s=%.1f "
        "new_tokens=%d budget_tokens=%d "
        "busy_slots=%d in_flight_prefills=%d "
        "free_slot=%s priority=%s policy=%s model=%s samples=%d provider=%s "
        "session=%s",
        verdict,
        effective_label or decision.reason,
        "false" if decision.admit else "true",
        float(decision.eta_prefill_seconds or 0.0),
        float(decision.eta_occupancy_seconds or 0.0),
        float(projected),
        float(decision.budget_seconds or 0.0),
        float(decision.prefill_deadline_seconds or 0.0),
        float(decision.deadline_budget_seconds or 0.0),
        int(decision.new_tokens or 0),
        int(decision.budget_tokens or 0),
        int(decision.busy_slots or 0),
        int(decision.in_flight_prefills or 0),
        "true" if free_slot else "false",
        priority or "unknown",
        decision.policy,
        model or "unknown",
        decision.samples,
        provider or "unknown",
        session_id or "unknown",
    )
