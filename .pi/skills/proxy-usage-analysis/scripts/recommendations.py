"""Rule-based, data-backed recommendations for the usage analysis report.

Each recommendation cites the aggregates that support it (fallback counts and
reasons, context sizes vs configured limits, fast/cheap fallback rates), so an
operator can judge whether a change is warranted. The rules encode the proxy
operator's domain knowledge:

- ``local_concurrency_limit`` / ``local_lease_active`` / ``slot_exhaustion``
  fallbacks indicate slot pool contention → raise the slot pool / schedule.
- ``large_context_bypass`` (or any reason containing ``large_context``)
  indicates prompts that cannot fit or would contend the KV cache → raise
  local ctx-size / routing thresholds.
- ``context_too_large`` (legacy ``warm_cache_bypass`` in rotated logs)
  indicates the prompt context exceeded the per-slot hard cap → raise
  local ctx-size / routing thresholds.
- Context pressure: sessions whose max context approaches the per-slot
  context limit (``local_model_ctx_size / slots``) → raise ctx-size.
- Fast vs cheap fallback-rate imbalance → adjust ``slot_schedule`` entries.
- Remote-side errors (HTTP 4xx/5xx, empty responses, timeouts) are
  informational: check the remote provider configuration.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import bucketing
from aggregation import AnalysisResult
from log_parser import CONTEXT_TOO_LARGE

# Reasons that point at local slot pool contention.
SLOT_CONTENTION_REASONS = {
    "local_concurrency_limit",
    "local_lease_active",
    "slot_exhaustion",
    "slot_saturated",
    "local_dispatch_denied",
    # LP-0MSORQVK50012Q4D: after the bounded contention queue, only
    # requests that exceeded the wait/depth caps still fall back — a
    # (hopefully reduced) subset of the old immediate-fallback behavior.
    "fallback_after_queue",
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
    return "cheap" if bucket == "cheap" else "fast"


def _dn(total: int, fast: int, cheap: int) -> str:
    """Format a total as a fast/cheap split with shares of the total."""
    return f"Fast {fast} ({_pct(fast, total):.1f}%) / Cheap {cheap} ({_pct(cheap, total):.1f}%)"


def _reason_counts_by_bucket(
    result: AnalysisResult,
    schedule,
    mode_map: bucketing.ModeScheduleMap | None = None,
) -> dict[str, Counter]:
    """Per-bucket fallback-reason counts (combined global + per-session).

    Mirrors ``_combined_reason_counts``: per-session reasons are bucketed by
    the session's bucket, global fallback events by their own timestamp
    (mode-aware when the logs show mode transitions, LP-0MSPZUD4G007IYGH).
    """
    buckets: dict[str, Counter] = {"fast": Counter(), "cheap": Counter()}
    for s in result.sessions.values():
        if s.fallback_reason:
            buckets[_bucket_key(s.bucket)][s.fallback_reason] += 1
    for ev in result.fallback_events:
        if ev.reason:
            if mode_map is not None and mode_map.transitions:
                label = mode_map.period_for(ev.ts).label
            elif schedule.periods:
                label = schedule.period_for(ev.ts).label
            else:
                label = "fast"
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


def _slot_counts(
    config: dict | None,
    result: AnalysisResult,
    mode_map: bucketing.ModeScheduleMap | None = None,
) -> tuple[int | None, int | None]:
    if mode_map is not None:
        fast = mode_map.schedules.get("fast")
        cheap = mode_map.schedules.get("cheap")
        return (
            fast.fast_slots if fast is not None else None,
            cheap.fast_slots if cheap is not None else None,
        )
    if config:
        schedule = bucketing.schedule_from_config(config, config.get("session_slot_pool_size"))
        return schedule.fast_slots, schedule.cheap_slots
    # Fall back to the slot counts observed per bucket in the data.
    fast_slots = {s.slots for s in result.sessions.values() if s.bucket == "fast" and s.slots}
    cheap_slots = {s.slots for s in result.sessions.values() if s.bucket == "cheap" and s.slots}
    return (sorted(fast_slots)[-1] if fast_slots else None), (
        sorted(cheap_slots)[-1] if cheap_slots else None
    )


def _bucket_stats(result: AnalysisResult) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for s in result.sessions.values():
        b = stats.setdefault(s.bucket or "fast", {"sessions": 0, "fell_back": 0, "requests": 0})
        b["sessions"] += 1
        b["requests"] += s.messages
        if s.fell_back:
            b["fell_back"] += 1
    for b in stats.values():
        b["fallback_rate"] = (b["fell_back"] / b["sessions"]) if b["sessions"] else 0.0
    return stats


def generate_recommendations(
    result: AnalysisResult,
    config: dict | None,
    mode_map: bucketing.ModeScheduleMap | None = None,
) -> list[Recommendation]:
    recs: list[Recommendation] = []

    sessions = list(result.sessions.values())
    total_requests = result.total_requests
    reason_counts = _combined_reason_counts(result)
    total_fallbacks = sum(reason_counts.values())
    fallback_rate = (total_fallbacks / total_requests) if total_requests else 0.0
    fast_slots, cheap_slots = _slot_counts(config, result, mode_map)
    bucket_stats = _bucket_stats(result)
    schedule = bucketing.schedule_from_config(
        config, (config or {}).get("session_slot_pool_size")
    )
    bucket_reasons = _reason_counts_by_bucket(result, schedule, mode_map)

    slot_counts_str = _slot_counts_str(fast_slots, cheap_slots)
    cfg_ctx = (config or {}).get("local_model_ctx_size")

    recs.extend(_error_recommendations(result))

    # --- 1. Slot pool contention ------------------------------------------
    contention = sum(reason_counts[r] for r in SLOT_CONTENTION_REASONS)
    if contention >= MIN_EVENTS and _pct(contention, total_fallbacks) >= REASON_SHARE * 100:
        breakdown = ", ".join(
            f"{r}: {reason_counts[r]}" for r in sorted(reason_counts) if r in SLOT_CONTENTION_REASONS and reason_counts[r]
        )
        contention_fast = sum(bucket_reasons["fast"][r] for r in SLOT_CONTENTION_REASONS)
        contention_cheap = sum(bucket_reasons["cheap"][r] for r in SLOT_CONTENTION_REASONS)
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
                    f"{_dn(contention, contention_fast, contention_cheap)}. "
                    f"Current slot counts: {slot_counts_str}."
                ),
            )
        )

    # --- 2. Large-context bypass -------------------------------------------
    large_ctx = sum(
        reason_counts[r] for r in reason_counts if r and "large_context" in r.lower()
    )
    if large_ctx >= MIN_EVENTS and _pct(large_ctx, total_fallbacks) >= REASON_SHARE * 100:
        large_ctx_fast = sum(
            bucket_reasons["fast"][r]
            for r in bucket_reasons["fast"]
            if r and "large_context" in r.lower()
        )
        large_ctx_cheap = sum(
            bucket_reasons["cheap"][r]
            for r in bucket_reasons["cheap"]
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
                    f"were `large_context_bypass`. {_dn(large_ctx, large_ctx_fast, large_ctx_cheap)}. {thresholds}"
                ),
            )
        )

    # --- 3. Context-too-large bypass -----------------------------------------
    # ``context_too_large`` is the current name for the warm-cache hard-cap
    # skip (renamed from ``warm_cache_bypass``, LP-0MSF8XDG7000PERM); the log
    # parser normalizes the legacy value, so only the current name is counted.
    warm = reason_counts.get(CONTEXT_TOO_LARGE, 0)
    if warm >= MIN_EVENTS and _pct(warm, total_fallbacks) >= REASON_SHARE * 100:
        warm_fast = bucket_reasons["fast"].get(CONTEXT_TOO_LARGE, 0)
        warm_cheap = bucket_reasons["cheap"].get(CONTEXT_TOO_LARGE, 0)
        recs.append(
            Recommendation(
                severity="medium",
                title="Context-too-large bypass is the dominant fallback reason",
                detail=(
                    "The router skipped local because the estimated prompt context exceeded the "
                    "hard per-slot capacity (`local_large_context_warm_cache_threshold`, clamped to "
                    "local_model_ctx_size / slots). This is a context-size signal, not a cache-warmth "
                    "problem. If it is frequent, consider raising the local ctx-size (models.ini) or "
                    "the large-context routing thresholds."
                ),
                evidence=(
                    f"{warm} of {total_fallbacks} fallback events ({_pct(warm, total_fallbacks):.1f}%) "
                    f"had reason `{CONTEXT_TOO_LARGE}`. {_dn(warm, warm_fast, warm_cheap)}."
                ),
            )
        )

    # --- 4. Context pressure ------------------------------------------------
    if cfg_ctx is not None or mode_map is not None:
        pressured = []
        for s in sessions:
            if not s.max_context_size or not s.slots:
                continue
            if mode_map is not None:
                # Per-session context comes from the profile active for that
                # session (per-period ctx_size when pinned, else the profile's
                # global local_model_ctx_size): 262144 for cheap (2 slots),
                # 131072 for fast (3 slots).
                per_slot_ctx = s.ctx_size if s.ctx_size is not None else mode_map.ctx_for(s.bucket)
            else:
                per_slot_ctx = cfg_ctx
            if per_slot_ctx is None:
                continue
            per_slot = per_slot_ctx / s.slots
            ratio = s.max_context_size / per_slot
            if ratio >= CONTEXT_PRESSURE_RATIO:
                pressured.append((s.session_id, s.max_context_size, per_slot, ratio, s.bucket))
        if pressured:
            worst = max(pressured, key=lambda t: t[3])
            critical = any(t[3] >= CONTEXT_CRITICAL_RATIO for t in pressured)
            pressured_fast = sum(1 for t in pressured if _bucket_key(t[4]) == "fast")
            pressured_cheap = len(pressured) - pressured_fast
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
                        f"{_dn(len(pressured), pressured_fast, pressured_cheap)}. "
                        f"Configured local_model_ctx_size={cfg_ctx}."
                    ),
                )
            )

    # --- 5. Fast/cheap imbalance -------------------------------------------
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
                    title="Fast/cheap fallback-rate imbalance in the slot schedule",
                    detail=(
                        f"{high_name} mode sessions fall back at a much higher rate than "
                        f"{low_name} mode sessions. Consider raising the slot count for the "
                        f"{high_name} mode `slot_schedule` entry (and keep `--parallel` aligned)."
                    ),
                    evidence=(
                        f"{high_name} mode fallback rate {high['fallback_rate'] * 100:.1f}% "
                        f"({high['fell_back']}/{high['sessions']} sessions) vs {low_name} mode "
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
        remote_fast = sum(bucket_reasons["fast"][r] for r in remote_errors)
        remote_cheap = sum(bucket_reasons["cheap"][r] for r in remote_errors)
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
                    f"{_dn(remote_total, remote_fast, remote_cheap)}."
                ),
            )
        )

    # --- 7. No change needed --------------------------------------------------
    if not recs and fallback_rate < FALLBACK_RATE_LOW:
        fast_fb_total = sum(bucket_reasons["fast"].values())
        cheap_fb_total = sum(bucket_reasons["cheap"].values())
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
                    f"events per request); {_dn(total_fallbacks, fast_fb_total, cheap_fb_total)}. "
                    "No slot contention, large-context, or warm-cache issues detected."
                ),
            )
        )

    return recs


def _error_recommendations(result: AnalysisResult) -> list[Recommendation]:
    """Remediation recommendations driven by the parsed error events.

    Mirrors the Aug 3 error-analysis plan (LP-0MSDFKCK4007CPMY): stream
    finish errors point at recovery-first silent continue and informative-
    error fallback; slot_save ReadTimeouts point at local ctx-size pressure;
    upstream 429s point at the FreeUsageLimitError cooldown; backend_retry
    timeouts are informational (upstream instability).
    """
    recs: list[Recommendation] = []
    if not result.error_events:
        return recs

    counts = result.error_counts
    total = len(result.error_events)

    stream_finish = counts.get("stream_finish_error", 0)
    stream_err = counts.get("stream_error", 0)
    slot_save = counts.get("slot_save_error", 0)
    backend_retry = counts.get("backend_retry", 0)
    if stream_finish > 0:
        provider_detail = ""
        pm = Counter(
            (e.provider, e.model)
            for e in result.error_events
            if e.kind == "stream_finish_error" and (e.provider or e.model)
        )
        if pm:
            top = ", ".join(f"{p}/{m}" if p and m else (p or m) for (p, m), _ in pm.most_common(3))
            provider_detail = f" Affected: {top}."
        # Informative-error coverage (LP-0MT6322OT00900OX): when the stream
        # finish errors carry the enriched payload, note the coverage.
        enriched = [
            e for e in result.error_events
            if e.kind == "stream_finish_error" and e.error_type
        ]
        if enriched:
            payload_note = (
                f"{len(enriched)} of {stream_finish} carry the enriched error payload "
                "(type/message/suggested-action) in the log and client-visible SSE event."
            )
        else:
            payload_note = "Log lines carry no error payload; enriched coverage cannot be verified."
        recs.append(
            Recommendation(
                severity="high",
                title="Stream finish errors: adopt recovery-first + informative-error strategy",
                detail=(
                    f"{stream_finish} stream(s) ended with the synthetic `finish_reason: error` event. "
                    f"{payload_note} Recommended "
                    "proxy-side remediation: (1) recovery-first silent continue — re-route to the next "
                    "healthy provider before content is delivered (see LP-0MSDP2PDB004GV86); (2) when "
                    "recovery is impossible, emit an informative error (type/message/provider/suggested "
                    "action) in the synthetic SSE event (see LP-0MSDP2PH20079WQ7). No client-side change "
                    "required."
                ),
                evidence=(
                    f"{stream_finish} of {total} error events ({_pct(stream_finish, total):.1f}%) were "
                    f"`stream_finish_error` ({stream_err} `stream_error` proxy exceptions)."
                    f"{provider_detail} Follow-ups: LP-0MSDP2PDB004GV86 (recovery-first), "
                    "LP-0MSDP2PH20079WQ7 (informative error)."
                ),
            )
        )

    if slot_save > 0:
        recs.append(
            Recommendation(
                severity="medium",
                title="slot_save failures (ReadTimeout): local context pressure",
                detail=(
                    f"{slot_save} `slot_save failed` event(s) (typically ReadTimeout/ReadTimeout) indicate "
                    "the local llama-server slot persistence is struggling, often under large-context "
                    "pressure. Consider raising the local model ctx-size (models.ini) or the routing "
                    "thresholds so fewer oversized prompts hit local slots. See LP-0MSAOQTJS000FFVM "
                    "(evaluate increasing local ctx-size)."
                ),
                evidence=f"{slot_save} of {total} error events ({_pct(slot_save, total):.1f}%) were `slot_save_error`.",
            )
        )

    # Break out upstream errors by status code and error type.
    upstream_by_status: dict[int, dict] = {}
    for e in result.error_events:
        if e.kind == "upstream_http_error" and e.status is not None:
            upstream_by_status.setdefault(e.status, {"count": 0, "error_types": Counter(), "providers": set()})
            upstream_by_status[e.status]["count"] += 1
            if e.error:
                upstream_by_status[e.status]["error_types"][e.error] += 1
            if e.provider:
                upstream_by_status[e.status]["providers"].add(e.provider)

    if upstream_by_status:
        # Build one recommendation per status code bucket.
        for status, info in sorted(upstream_by_status.items()):
            count = info["count"]
            error_types = info["error_types"]
            providers = info["providers"]
            et_str = ", ".join(
                f"{et} ({c})" for et, c in error_types.most_common(5)
            ) if error_types else "none extracted"
            prov_str = ", ".join(sorted(providers)) if providers else "unknown"

            # Status-specific remediation.
            if status == 429:
                severity = "medium"
                title = "Upstream HTTP 429: rate limiting active"
                detail = (
                    f"{count} upstream HTTP 429 event(s) from {prov_str} "
                    f"(error types: {et_str}). The proxy's 3-hour per-model cooldown "
                    f"(LP-0MRGU0I91006ODFD) should suppress repeat fallbacks to the affected model; "
                    f"if 429s persist, check the upstream provider quota or usage limits."
                )
            elif status == 402:
                severity = "high"
                title = "Upstream HTTP 402: payment/balance required"
                detail = (
                    f"{count} upstream HTTP 402 event(s) from {prov_str} "
                    f"(error types: {et_str}). This indicates an account balance or subscription issue "
                    f"with the upstream provider — not a transient error. Top up the account, switch "
                    f"to a different provider tier, or adjust routing to avoid this provider."
                )
            elif status == 400:
                severity = "medium"
                title = "Upstream HTTP 400: bad request"
                detail = (
                    f"{count} upstream HTTP 400 event(s) from {prov_str} "
                    f"(error types: {et_str}). These are client-side errors — check the request "
                    f"payload, credentials, or model identifier being sent."
                )
            elif status in (500, 502, 503, 504):
                severity = "medium"
                title = f"Upstream HTTP {status}: server error"
                detail = (
                    f"{count} upstream HTTP {status} event(s) from {prov_str} "
                    f"(error types: {et_str}). These are server-side errors from the upstream provider. "
                    f"Monitor for clustering; persistent errors suggest a provider outage."
                )
            else:
                severity = "medium"
                title = f"Upstream HTTP {status}: unexpected status"
                detail = (
                    f"{count} upstream HTTP {status} event(s) from {prov_str} "
                    f"(error types: {et_str}). Check upstream provider docs for this status code."
                )

            recs.append(
                Recommendation(
                    severity=severity,
                    title=title,
                    detail=detail,
                    evidence=f"{count} of {total} error events ({_pct(count, total):.1f}%) were `upstream_http_error` status={status} ({et_str}). Providers: {prov_str}.",
                )
            )

    if backend_retry > 0:
        recs.append(
            Recommendation(
                severity="info",
                title="backend_retry timeouts: upstream instability",
                detail=(
                    f"{backend_retry} `backend_retry` event(s) (ConnectTimeout/ReadError) show upstream "
                    "connectivity issues during the retry backoff. These are transient unless they "
                    "cluster; monitor the next window and check upstream health if they persist."
                ),
                evidence=f"{backend_retry} of {total} error events ({_pct(backend_retry, total):.1f}%) were `backend_retry`.",
            )
        )

    return recs


def _slot_counts_str(fast_slots: int | None, cheap_slots: int | None) -> str:
    if fast_slots is not None and cheap_slots is not None:
        return f"{fast_slots} fast / {cheap_slots} cheap"
    if fast_slots is not None:
        return f"{fast_slots} (single bucket)"
    return "see proxy/config.yaml slot_schedule"
