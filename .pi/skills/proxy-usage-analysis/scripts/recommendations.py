"""Rule-based, data-backed recommendations for the usage analysis report.

Each recommendation cites the aggregates that support it (fallback counts and
reasons, context sizes vs configured limits, day/night fallback rates), so an
operator can judge whether a change is warranted. The rules encode the proxy
operator's domain knowledge:

- ``local_concurrency_limit`` / ``local_lease_active`` / ``slot_exhaustion``
  fallbacks indicate slot pool contention → raise the slot pool / schedule.
- ``large_context_bypass`` (or any reason containing ``large_context``)
  indicates prompts that cannot fit or would contend the KV cache → raise
  local ctx-size / routing thresholds.
- ``warm_cache_bypass`` indicates the session cache was not warm at routing
  decision time → improve cache warm-up or raise the warm threshold.
- Context pressure: sessions whose max context approaches the per-slot
  context limit (``local_model_ctx_size / slots``) → raise ctx-size.
- Day vs night fallback-rate imbalance → adjust ``slot_schedule`` entries.
- Remote-side errors (HTTP 4xx/5xx, empty responses, timeouts) are
  informational: check the remote provider configuration.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


import bucketing
from aggregation import AnalysisResult

# Reasons that point at local slot pool contention.
SLOT_CONTENTION_REASONS = {
    "local_concurrency_limit",
    "local_lease_active",
    "slot_exhaustion",
    "slot_saturated",
    "local_dispatch_denied",
}

# Reasons that point at remote provider issues (informational).
REMOTE_ERROR_REASONS = (
    "http",
    "empty_response",
    "timeout",
    "exhausted",
    "unavailable",
    "500",
    "502",
    "503",
    "504",
    "429",
    "400",
    "401",
    "403",
)

MIN_EVENTS = 1
REASON_SHARE = 0.20
FALLBACK_RATE_LOW = 0.10
FALLBACK_RATE_HIGH = 0.30
CONTEXT_PRESSURE_RATIO = 0.80
CONTEXT_CRITICAL_RATIO = 0.95
IMBALANCE_RATIO = 1.5
IMBALANCE_MIN_SESSIONS = 3


@dataclass
class Recommendation:
    severity: str  # "high" | "medium" | "info"
    title: str
    detail: str
    evidence: str


def _pct(part: int, total: int) -> float:
    return (part / total * 100.0) if total else 0.0


def _bucket_key(bucket: str | None) -> str:
    return "night" if bucket == "night" else "day"


def _dn(total: int, day: int, night: int) -> str:
    """Format a total as a day/night split with shares of the total."""
    return f"Day {day} ({_pct(day, total):.1f}%) / Night {night} ({_pct(night, total):.1f}%)"


def _reason_counts_by_bucket(result: AnalysisResult, schedule) -> dict[str, Counter]:
    """Per-bucket fallback-reason counts (combined global + per-session).

    Mirrors ``_combined_reason_counts``: per-session reasons are bucketed by
    the session's bucket, global fallback events by their own timestamp.
    """
    buckets: dict[str, Counter] = {"day": Counter(), "night": Counter()}
    for s in result.sessions.values():
        if s.fallback_reason:
            buckets[_bucket_key(s.bucket)][s.fallback_reason] += 1
    for ev in result.fallback_events:
        if ev.reason:
            label = schedule.period_for(ev.ts).label if schedule.periods else "day"
            buckets[_bucket_key(label)][ev.reason] += 1
    return buckets


def _combined_reason_counts(result: AnalysisResult) -> Counter:
    """Global fallback-event reasons + per-session attributed reasons.

    Per-session reasons are often (but not always) the same event as a global
    fallback line; the heuristic deliberately counts both so a single clear
    signal is never missed. Shares are computed against the combined total.
    """
    counts: Counter = Counter()
    for s in result.sessions.values():
        if s.fallback_reason:
            counts[s.fallback_reason] += 1
    for ev in result.fallback_events:
        if ev.reason:
            counts[ev.reason] += 1
    return counts


def _slot_counts(config: dict | None, result: AnalysisResult) -> tuple[int | None, int | None]:
    if config:
        schedule = bucketing.schedule_from_config(config, config.get("session_slot_pool_size"))
        return schedule.day_slots, schedule.night_slots
    # Fall back to the slot counts observed per bucket in the data.
    day_slots = {s.slots for s in result.sessions.values() if s.bucket == "day" and s.slots}
    night_slots = {s.slots for s in result.sessions.values() if s.bucket == "night" and s.slots}
    return (sorted(day_slots)[-1] if day_slots else None), (
        sorted(night_slots)[-1] if night_slots else None
    )


def _bucket_stats(result: AnalysisResult) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for s in result.sessions.values():
        b = stats.setdefault(s.bucket or "day", {"sessions": 0, "fell_back": 0, "requests": 0})
        b["sessions"] += 1
        b["requests"] += s.messages
        if s.fell_back:
            b["fell_back"] += 1
    for b in stats.values():
        b["fallback_rate"] = (b["fell_back"] / b["sessions"]) if b["sessions"] else 0.0
    return stats


def generate_recommendations(result: AnalysisResult, config: dict | None) -> list[Recommendation]:
    recs: list[Recommendation] = []

    sessions = list(result.sessions.values())
    total_requests = result.total_requests
    reason_counts = _combined_reason_counts(result)
    total_fallbacks = sum(reason_counts.values())
    fallback_rate = (total_fallbacks / total_requests) if total_requests else 0.0
    day_slots, night_slots = _slot_counts(config, result)
    bucket_stats = _bucket_stats(result)
    schedule = bucketing.schedule_from_config(
        config, (config or {}).get("session_slot_pool_size")
    )
    bucket_reasons = _reason_counts_by_bucket(result, schedule)

    slot_counts_str = _slot_counts_str(day_slots, night_slots)
    cfg_ctx = (config or {}).get("local_model_ctx_size")

    # --- 1. Slot pool contention ------------------------------------------
    contention = sum(reason_counts[r] for r in SLOT_CONTENTION_REASONS)
    if contention >= MIN_EVENTS and _pct(contention, total_fallbacks) >= REASON_SHARE * 100:
        breakdown = ", ".join(
            f"{r}: {reason_counts[r]}" for r in sorted(reason_counts) if r in SLOT_CONTENTION_REASONS and reason_counts[r]
        )
        contention_day = sum(bucket_reasons["day"][r] for r in SLOT_CONTENTION_REASONS)
        contention_night = sum(bucket_reasons["night"][r] for r in SLOT_CONTENTION_REASONS)
        recs.append(
            Recommendation(
                severity="high",
                title="Raise session_slot_pool_size (slot pool contention)",
                detail=(
                    "Local dispatch is frequently denied or deferred because all slots are busy. "
                    "Increase the slot pool (`session_slot_pool_size` in proxy/config.yaml) or the "
                    "slot counts in the `slot_schedule` entries; the pool must stay aligned with "
                    "llama-server's `--parallel` flag."
                ),
                evidence=(
                    f"{contention} of {total_fallbacks} fallback events ({_pct(contention, total_fallbacks):.1f}%) "
                    f"were slot-contention related ({breakdown}). "
                    f"{_dn(contention, contention_day, contention_night)}. "
                    f"Current slot counts: {slot_counts_str}."
                ),
            )
        )

    # --- 2. Large-context bypass -------------------------------------------
    large_ctx = sum(
        reason_counts[r] for r in reason_counts if r and "large_context" in r.lower()
    )
    if large_ctx >= MIN_EVENTS and _pct(large_ctx, total_fallbacks) >= REASON_SHARE * 100:
        large_ctx_day = sum(
            bucket_reasons["day"][r]
            for r in bucket_reasons["day"]
            if r and "large_context" in r.lower()
        )
        large_ctx_night = sum(
            bucket_reasons["night"][r]
            for r in bucket_reasons["night"]
            if r and "large_context" in r.lower()
        )
        thresholds = (
            f"Configured thresholds: cold={config.get('local_large_context_cold_cache_threshold')}, "
            f"warm={config.get('local_large_context_warm_cache_threshold')}, "
            f"session_slot_max_prompt_tokens={config.get('session_slot_max_prompt_tokens')}."
            if config
            else "See proxy/config.yaml for current routing thresholds."
        )
        recs.append(
            Recommendation(
                severity="high",
                title="Raise local ctx-size / large-context routing thresholds",
                detail=(
                    "Requests are being bypassed to remote providers because their contexts exceed the "
                    "large-context routing thresholds. Consider raising the local model's ctx-size "
                    "(models.ini), `local_large_context_cold_cache_threshold`, "
                    "`local_large_context_warm_cache_threshold`, or `session_slot_max_prompt_tokens`. "
                    "See also work item LP-0MSAOQTJS000FFVM (evaluating an increase of the local ctx-size)."
                ),
                evidence=(
                    f"{large_ctx} of {total_fallbacks} fallback events ({_pct(large_ctx, total_fallbacks):.1f}%) "
                    f"were `large_context_bypass`. {_dn(large_ctx, large_ctx_day, large_ctx_night)}. {thresholds}"
                ),
            )
        )

    # --- 3. Warm-cache bypass ----------------------------------------------
    warm = reason_counts.get("warm_cache_bypass", 0)
    if warm >= MIN_EVENTS and _pct(warm, total_fallbacks) >= REASON_SHARE * 100:
        warm_day = bucket_reasons["day"].get("warm_cache_bypass", 0)
        warm_night = bucket_reasons["night"].get("warm_cache_bypass", 0)
        recs.append(
            Recommendation(
                severity="medium",
                title="Warm-cache bypass is the dominant fallback reason",
                detail=(
                    "The router skipped local because the session's cache was not warm at decision time "
                    "(high estimated tokens with a cold cache). If this is frequent, consider raising "
                    "`local_large_context_warm_cache_threshold` or improving slot-cache warm-up / session "
                    "affinity so warm sessions stay on the local model."
                ),
                evidence=(
                    f"{warm} of {total_fallbacks} fallback events ({_pct(warm, total_fallbacks):.1f}%) "
                    f"had reason `warm_cache_bypass`. {_dn(warm, warm_day, warm_night)}."
                ),
            )
        )

    # --- 4. Context pressure ------------------------------------------------
    if cfg_ctx:
        pressured = []
        for s in sessions:
            if not s.max_context_size or not s.slots:
                continue
            per_slot = cfg_ctx / s.slots
            ratio = s.max_context_size / per_slot
            if ratio >= CONTEXT_PRESSURE_RATIO:
                pressured.append((s.session_id, s.max_context_size, per_slot, ratio, s.bucket))
        if pressured:
            worst = max(pressured, key=lambda t: t[3])
            critical = any(t[3] >= CONTEXT_CRITICAL_RATIO for t in pressured)
            pressured_day = sum(1 for t in pressured if _bucket_key(t[4]) == "day")
            pressured_night = len(pressured) - pressured_day
            recs.append(
                Recommendation(
                    severity="high" if critical else "medium",
                    title="Context sizes approaching per-slot limits",
                    detail=(
                        "Some sessions reach contexts near the effective per-slot context "
                        "(local_model_ctx_size / active slots), which can force `large_context_bypass` "
                        "or degrade local performance. Raising the local model ctx-size "
                        "(models.ini) and/or the large-context thresholds reduces remote fallback."
                    ),
                    evidence=(
                        f"{len(pressured)} session(s) peaked at >= {int(CONTEXT_PRESSURE_RATIO * 100)}% of "
                        "per-slot context; worst: session " + worst[0][:8] + f" at {worst[1]} tokens "
                        f"(per-slot {worst[2]:.0f}, {worst[3] * 100:.0f}%). "
                        f"{_dn(len(pressured), pressured_day, pressured_night)}. "
                        f"Configured local_model_ctx_size={cfg_ctx}."
                    ),
                )
            )

    # --- 5. Day/night imbalance ---------------------------------------------
    buckets = [b for b in bucket_stats.values() if b["sessions"] >= IMBALANCE_MIN_SESSIONS]
    if len(buckets) == 2:
        low, high = sorted(buckets, key=lambda b: b["fallback_rate"])
        ratio = (high["fallback_rate"] / low["fallback_rate"]) if low["fallback_rate"] > 0 else float("inf")
        if ratio >= IMBALANCE_RATIO:
            low_name = next(n for n, b in bucket_stats.items() if b is low)
            high_name = next(n for n, b in bucket_stats.items() if b is high)
            total_fb = sum(b["fell_back"] for b in bucket_stats.values())
            total_sess = sum(b["sessions"] for b in bucket_stats.values())
            recs.append(
                Recommendation(
                    severity="medium",
                    title="Day/night fallback-rate imbalance in the slot schedule",
                    detail=(
                        f"{high_name.capitalize()}time sessions fall back at a much higher rate than "
                        f"{low_name}time sessions. Consider raising the slot count for the "
                        f"{high_name}time `slot_schedule` entry (and keep `--parallel` aligned)."
                    ),
                    evidence=(
                        f"{high_name.capitalize()}time fallback rate {high['fallback_rate'] * 100:.1f}% "
                        f"({high['fell_back']}/{high['sessions']} sessions) vs {low_name}time "
                        f"{low['fallback_rate'] * 100:.1f}% ({low['fell_back']}/{low['sessions']}); "
                        f"overall {_pct(total_fb, total_sess):.1f}% ({total_fb}/{total_sess}). "
                        f"Current slot counts: {slot_counts_str}."
                    ),
                )
            )

    # --- 6. Remote provider errors (informational) ---------------------------
    remote_errors = [
        r for r in reason_counts if r and any(tok in r.lower() for tok in REMOTE_ERROR_REASONS)
    ]
    if remote_errors:
        counts = ", ".join(f"{r}: {reason_counts[r]}" for r in sorted(remote_errors, key=lambda r: -reason_counts[r]))
        remote_day = sum(bucket_reasons["day"][r] for r in remote_errors)
        remote_night = sum(bucket_reasons["night"][r] for r in remote_errors)
        remote_total = sum(reason_counts[r] for r in remote_errors)
        recs.append(
            Recommendation(
                severity="info",
                title="Remote provider errors observed",
                detail=(
                    "These fallback reasons originate from remote providers (HTTP errors, empty "
                    "responses, timeouts). They are not slot-related; check the remote provider "
                    "configuration, credentials, and rate limits."
                ),
                evidence=(
                    f"Fallback events with remote-error reasons: {counts}. "
                    f"{_dn(remote_total, remote_day, remote_night)}."
                ),
            )
        )

    # --- 7. No change needed --------------------------------------------------
    if not recs and fallback_rate < FALLBACK_RATE_LOW:
        day_fb_total = sum(bucket_reasons["day"].values())
        night_fb_total = sum(bucket_reasons["night"].values())
        recs.append(
            Recommendation(
                severity="info",
                title="No configuration changes needed",
                detail=(
                    "Local models handled the window with a low fallback rate and no concerning "
                    "patterns were detected. Re-run this skill after any configuration change to "
                    "verify the effect."
                ),
                evidence=(
                    f"Fallback rate {fallback_rate * 100:.1f}% ({total_fallbacks}/{total_requests} "
                    f"events per request); {_dn(total_fallbacks, day_fb_total, night_fb_total)}. "
                    "No slot contention, large-context, or warm-cache issues detected."
                ),
            )
        )

    return recs


def _slot_counts_str(day_slots: int | None, night_slots: int | None) -> str:
    if day_slots is not None and night_slots is not None:
        return f"{day_slots} day / {night_slots} night"
    if day_slots is not None:
        return f"{day_slots} (single bucket)"
    return "see proxy/config.yaml slot_schedule"
