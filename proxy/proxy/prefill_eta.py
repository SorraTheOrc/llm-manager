"""Prefill-ETA calibration and the local admission guard (LP-0MVGDUV5E005UWBQ).

Problem
-------

When every local slot is busy, a request enters the bounded contention queue
(bounded by ``contention_queue_max_wait_seconds``, 120 s in the shipped
config) and waits. For a large cache-cold prompt the local prefill cannot
finish inside that budget, so the wait is a *guaranteed loss*: the request
burns up to the full wait and then falls back to a remote provider anyway
(measured 2026-10-19/20: 5,805 waits → 855 dispatched, 4,947 = 85 % abandoned
after up to 120 s).

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

#: Upper bounds of the context buckets: <16k, 16k-64k, 64k-128k, >=128k.
CONTEXT_BUCKET_BOUNDS = (16_384, 65_536, 131_072)

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


@dataclass(frozen=True)
class GuardConfig:
    """Resolved ``server.local_admission_guard_*`` settings."""

    policy: str
    min_samples: int
    window_samples: int
    reference_tokens: int


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
    return GuardConfig(
        policy=raw_policy,
        min_samples=min_samples,
        window_samples=window,
        reference_tokens=reference,
    )


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
    """Fitted (or default) per-token prefill coefficients, in milliseconds."""

    coded_ms_per_token: float
    prefill_ms_per_token: float
    samples: int
    calibrated: bool


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


def coefficients(model: str | None, prompt_tokens: int, config: dict | None = None
                 ) -> Coefficients:
    """Return the coefficients to use for *model* at *prompt_tokens*.

    Falls back to the sane defaults (``calibrated=False``) when the bucket has
    fewer than ``min_samples`` observations or the fit is unusable — in which
    case the guard stays inert and today's behaviour is preserved.
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
        )
    fitted = _fit(samples)
    if fitted is None:
        return Coefficients(
            coded_ms_per_token=DEFAULT_CODED_MS_PER_TOKEN,
            prefill_ms_per_token=DEFAULT_PREFILL_MS_PER_TOKEN,
            samples=len(samples),
            calibrated=False,
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
    return Coefficients(
        coded_ms_per_token=_clamp(
            coded_ms,
            MIN_CODED_MS_PER_TOKEN,
            MAX_CODED_MS_PER_TOKEN,
        ),
        prefill_ms_per_token=_clamp(
            prefill_ms,
            MIN_PREFILL_MS_PER_TOKEN,
            MAX_PREFILL_MS_PER_TOKEN,
        ),
        samples=len(samples),
        calibrated=True,
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
    defaults (``coefficients.calibrated is False``).
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

    @property
    def active(self) -> bool:
        """True when the guard actually made a decision (not inert)."""
        return self.reason in ("within_budget", "eta_exceeds_budget")


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
) -> AdmissionDecision:
    """Decide whether a local dispatch can finish inside the wait budget.

    Fail-open by construction: the guard is inert (``admit=True``) when the
    policy is ``off``, the priority never queues, the bucket is not yet
    calibrated, or no budget can be resolved. Only a calibrated ETA that
    exceeds the budget refuses admission, and only when the policy is
    ``reject`` (``observe`` logs the same verdict without refusing).
    """
    guard_cfg = resolve_config(config)

    def _inert(reason: str, eta: float | None = None, budget: float | None = None,
               coeffs: Coefficients | None = None) -> AdmissionDecision:
        return AdmissionDecision(
            admit=True,
            reason=reason,
            eta_prefill_seconds=eta,
            budget_seconds=budget,
            policy=guard_cfg.policy,
            calibrated=bool(coeffs.calibrated) if coeffs else False,
            samples=coeffs.samples if coeffs else 0,
        )

    if guard_cfg.policy == POLICY_OFF:
        return _inert("policy_off")
    budget = wait_budget_seconds(config, priority)
    if budget <= 0:
        return _inert("no_budget")
    eta, coeffs = estimate_prefill_seconds(model, prompt_tokens, cached_ratio, config)
    if not coeffs.calibrated:
        return _inert("insufficient_samples", eta, budget, coeffs)
    within = eta <= budget
    reason = "within_budget" if within else "eta_exceeds_budget"
    admit = within or guard_cfg.policy == POLICY_OBSERVE
    return AdmissionDecision(
        admit=admit,
        reason=reason,
        eta_prefill_seconds=eta,
        budget_seconds=budget,
        policy=guard_cfg.policy,
        calibrated=True,
        samples=coeffs.samples,
    )


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
    repeated for every currently occupied slot, plus its own prefill.

    Fail-open: any error yields ``None`` fields rather than a broken payload.
    """
    try:
        tokens = reference_tokens(model, config)
        eta, coeffs = estimate_prefill_seconds(model, tokens, 0.0, config)
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
            "queue_eta_seconds": round(eta * (busy + 1), 1),
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
) -> None:
    """Emit the ``local_admission_guard`` log line for *decision*.

    Only active decisions are logged (an inert guard made no decision, and
    logging every inert request would flood the log). The line is parsed by
    the proxy-usage-analysis skill, which counts admits and refusals
    (AC2).
    """
    if not decision.active or decision.eta_prefill_seconds is None:
        return
    # The logged ``decision`` is the guard's *verdict* (admit/reject); the
    # ``enforced`` flag distinguishes an enforced refusal from an ``observe``
    # rollout, so the usage report can count would-be refusals during a
    # validation window.
    verdict = "reject" if decision.reason == "eta_exceeds_budget" else "admit"
    logger.info(
        "local_admission_guard decision=%s reason=%s enforced=%s "
        "eta_prefill_s=%.1f budget_s=%.1f priority=%s policy=%s model=%s "
        "samples=%d provider=%s session=%s",
        verdict,
        decision.reason,
        "false" if decision.admit else "true",
        float(decision.eta_prefill_seconds),
        float(decision.budget_seconds or 0.0),
        priority or "unknown",
        decision.policy,
        model or "unknown",
        decision.samples,
        provider or "unknown",
        session_id or "unknown",
    )
