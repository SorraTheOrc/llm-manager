"""Capacity-based local admission for the economic bypass band.

LP-0MVGDUVS6003T2IX. The local size gates were fixed token thresholds
(38k cold / ~124k warm): every prompt in the 38k–124k band that is cache-cold
was bypassed to a paid remote provider, yet the GPU is idle more than half of
every 24 h window. The naive fix — raising the thresholds — was rejected
twice (LP-0MVGCOZMZ003I6PF, LP-0MVGD9YLM0055SFF) because those bands are ~4×
over capacity when cold; they only fit with warm caches, admission control and
serialised prefills.

This module adds the missing piece: an admission rule expressed in **slot
time**, layered on the existing thresholds. The economic band is admitted
locally only while the projected slot-seconds of admissions inside a rolling
horizon stay within a configured share of the *measured idle* slot time:

    projected_s = projected prefill (prefill_eta, from observed dispatches)
                + decode allowance
    budget_s    = budget_ratio * idle_slot_s
    idle_slot_s = slots * horizon - union(busy intervals in the horizon)

    admit iff admitted_s + projected_s <= budget_s

Both terms are recomputed from rolling observed data (no fixed capacity
constant): ``busy`` comes from recorded local dispatch intervals and the
prefill cost from :mod:`proxy.prefill_eta`, which is calibrated from
``dispatch_first_byte_ms`` samples.

Disabled by default (``local_capacity_admission_enabled: false``) — this is a
deliberate follow-on: it must only be enabled once cache warmth
(LP-0MVGD9YXH008O304), the prefill-ETA guard (LP-0MVGDUV5E005UWBQ) and prefill
serialisation (LP-0MVGDUUV3001XFRJ) are live in the running proxy.

Fail-open: any error, a disabled policy or an uncalibrated ETA keeps today's
behaviour (the bypass stands).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger("llama-proxy.provider")

DEFAULT_BUDGET_RATIO = 0.8
DEFAULT_HORIZON_SECONDS = 3600.0
DEFAULT_DECODE_SECONDS = 30.0
DEFAULT_SLOTS = 1

_MAX_INTERVALS = 4096
"""Bound on retained busy intervals / admission records (memory safety)."""


@dataclass(frozen=True)
class CapacityPolicy:
    """Resolved capacity-admission policy."""

    enabled: bool
    budget_ratio: float
    horizon_seconds: float
    decode_seconds: float


_busy_intervals: deque = deque(maxlen=_MAX_INTERVALS)
"""``(start, end)`` monotonic intervals during which a local slot was busy."""

_admitted: deque = deque(maxlen=_MAX_INTERVALS)
"""``(admitted_at, projected_seconds)`` for requests admitted by this gate."""


def _server_config(config: dict | None) -> dict:
    if not isinstance(config, dict):
        return {}
    return config.get("server", config)


def _float(raw, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def resolve_policy(config: dict | None) -> CapacityPolicy:
    """Resolve the capacity-admission policy from *config*.

    ``local_capacity_admission_enabled`` defaults to **false** (the rollback
    switch — clearing it restores plain threshold routing).
    ``local_capacity_admission_budget_ratio`` defaults to 0.8 and is clamped to
    ``(0, 1]``; a value outside that range falls back to the default.
    ``local_capacity_admission_horizon_seconds`` defaults to 3600 and a
    non-positive value falls back to the default.
    ``local_capacity_admission_decode_seconds`` defaults to 30 s and must be
    non-negative (a negative value falls back to the default).
    """
    server = _server_config(config)
    ratio = _float(
        server.get("local_capacity_admission_budget_ratio", DEFAULT_BUDGET_RATIO),
        DEFAULT_BUDGET_RATIO,
    )
    if not 0.0 < ratio <= 1.0:
        ratio = DEFAULT_BUDGET_RATIO
    horizon = _float(
        server.get("local_capacity_admission_horizon_seconds", DEFAULT_HORIZON_SECONDS),
        DEFAULT_HORIZON_SECONDS,
    )
    if horizon <= 0:
        horizon = DEFAULT_HORIZON_SECONDS
    decode = _float(
        server.get("local_capacity_admission_decode_seconds", DEFAULT_DECODE_SECONDS),
        DEFAULT_DECODE_SECONDS,
    )
    if decode < 0:
        decode = DEFAULT_DECODE_SECONDS
    return CapacityPolicy(
        enabled=bool(server.get("local_capacity_admission_enabled", False)),
        budget_ratio=ratio,
        horizon_seconds=horizon,
        decode_seconds=decode,
    )


def _slots(config: dict | None) -> int:
    """Slot-pool size from ``server.session_slot_pool_size`` (min 1)."""
    server = _server_config(config)
    try:
        return max(1, int(server.get("session_slot_pool_size", DEFAULT_SLOTS) or 1))
    except (TypeError, ValueError):
        return DEFAULT_SLOTS


def record_slot_busy(start: float | None, end: float | None) -> None:
    """Record one local dispatch interval for utilisation accounting."""
    if start is None or end is None:
        return
    try:
        s = float(start)
        e = float(end)
    except (TypeError, ValueError):
        return
    if e <= s:
        return
    _busy_intervals.append((s, e))


def busy_slot_seconds(
    horizon_seconds: float, now: float | None = None
) -> float:
    """Union of busy slot-seconds inside the last *horizon_seconds*.

    Overlapping dispatches (two slots busy at once) are merged, so the result
    is slot-seconds of *occupancy*, not the sum of request durations.
    """
    reference = time.monotonic() if now is None else float(now)
    window_start = reference - max(0.0, float(horizon_seconds))
    intervals = []
    for start, end in list(_busy_intervals):
        if end <= window_start or start >= reference:
            continue
        intervals.append((max(start, window_start), min(end, reference)))
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return max(0.0, total)


def idle_slot_seconds(
    config: dict | None,
    horizon_seconds: float,
    now: float | None = None,
) -> float:
    """Measured idle slot-seconds in the horizon (never negative)."""
    capacity = _slots(config) * max(0.0, float(horizon_seconds))
    return max(0.0, capacity - busy_slot_seconds(horizon_seconds, now=now))


def _prune_admissions(reference: float, horizon_seconds: float) -> None:
    cutoff = reference - max(0.0, float(horizon_seconds))
    while _admitted and _admitted[0][0] < cutoff:
        _admitted.popleft()


def admitted_slot_seconds(
    horizon_seconds: float, now: float | None = None
) -> float:
    """Projected slot-seconds admitted by this gate inside the horizon."""
    reference = time.monotonic() if now is None else float(now)
    _prune_admissions(reference, horizon_seconds)
    return sum(seconds for _ts, seconds in _admitted)


def project_slot_seconds(
    config: dict | None,
    model: str | None,
    estimated_tokens: int,
    cached_ratio: float,
) -> float:
    """Projected slot-seconds for admitting one request.

    ``prefill_eta`` is calibrated from observed ``dispatch_first_byte_ms``
    samples (per model and context bucket) and already accounts for the cached
    prefix; the decode allowance is a configured constant because decode time
    is dominated by the client's output budget, not by prompt size.
    """
    policy = resolve_policy(config)
    eta = None
    try:
        from proxy.prefill_eta import estimate_prefill_seconds

        eta, _coeffs = estimate_prefill_seconds(
            model, int(estimated_tokens or 0), float(cached_ratio or 0.0), config
        )
    except Exception:
        eta = None
    if eta is None or eta < 0:
        return 0.0
    return float(eta) + float(policy.decode_seconds)


def evaluate_capacity_admission(
    config: dict | None,
    model: str | None,
    estimated_tokens: int,
    cached_ratio: float,
    now: float | None = None,
) -> dict | None:
    """Decide whether one economic-band request fits the idle capacity budget.

    Returns a decision dict (recorded in the log line) when the request should
    be admitted locally, else ``None`` — meaning today's bypass stands.

    The decision is recorded in the rolling admission ledger, so a burst of
    band requests cannot collectively over-commit the horizon.
    """
    policy = resolve_policy(config)
    if not policy.enabled:
        return None
    reference = time.monotonic() if now is None else float(now)
    projected = project_slot_seconds(
        config, model, estimated_tokens, cached_ratio
    )
    if projected <= 0.0:
        return None
    idle = idle_slot_seconds(config, policy.horizon_seconds, now=reference)
    budget = policy.budget_ratio * idle
    admitted = admitted_slot_seconds(policy.horizon_seconds, now=reference)
    if admitted + projected > budget:
        return None
    _admitted.append((reference, projected))
    return {
        "reason": "capacity_admission",
        "projected_seconds": projected,
        "admitted_seconds": admitted + projected,
        "budget_seconds": budget,
        "idle_seconds": idle,
        "budget_ratio": policy.budget_ratio,
        "horizon_seconds": policy.horizon_seconds,
    }


def reset_capacity_state() -> None:
    """Clear the rolling state (test helper / operator reset)."""
    _busy_intervals.clear()
    _admitted.clear()
