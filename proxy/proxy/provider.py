
# <!-- REFACTOR-LP-0MSI4ZIF60064EOR
# smell: modernization
# severity: medium
# description: Use format specifiers instead of percent format
# -->
"""
Provider Module

Provider resolution and fallback logic for model requests.

Provides:
- `resolve_provider()`: Select the next available provider for a model config
- `proxy_with_remote_fallback()`: Remote provider fallback loop
- Cooldown tracking: Mark providers as temporarily unavailable after failures
- Timed access: Skip providers outside their configured `available_times` UTC
  windows (LP-0MS4ETBNO0022QAC)
"""

import asyncio
import json
import logging
import re
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import HTTPException, Response
from fastapi.responses import StreamingResponse

from proxy.utils import _is_empty_response

logger = logging.getLogger("llama-proxy.provider")

# ---------------------------------------------------------------------------
# Cooldown / circuit-breaker state
# ---------------------------------------------------------------------------

# In-memory cooldown tracking: provider_name -> expiry_timestamp (seconds since epoch)
_provider_unavailable_until: dict[str, float] = {}

# Consecutive failure count for exponential backoff: provider_name -> count
# Incremented on each failure, reset to 0 on success.
_provider_failure_count: dict[str, int] = {}

# Exponential backoff constants (remote providers only)
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 45.0

# FreeUsageLimitError cooldown: 3 hours (10800 seconds)
# Applied when upstream returns HTTP 429 with error.type = "FreeUsageLimitError"
# See LP-0MRGU0I91006ODFD for details.
_FREE_USAGE_LIMIT_COOLDOWN_SECONDS = 10800

# Usage-limit reset tracking (LP-0MSLJPOCC0001ROJ): failure-domain key ->
# absolute epoch timestamp when the usage limit resets (including the
# 2-minute safety margin). Providers in a domain with a pending usage reset
# are skipped by every routing decision until the reset time passes.
_usage_reset_at: dict[str, float] = {}

# 2-minute safety margin added to the computed usage-limit reset time so a
# clock-skewed upstream does not start re-serving 429s the moment the limit
# nominally resets.
_USAGE_LIMIT_RESET_MARGIN_SECONDS = 120

# Fallback durations per metadata.limitName when the upstream message carries
# no explicit "Resets in ..." duration (daily/weekly/monthly periods).
_PERIOD_DEFAULT_SECONDS = {
    "daily": 24 * 3600,
    "weekly": 7 * 24 * 3600,
    "monthly": 30 * 24 * 3600,
}

# ---------------------------------------------------------------------------
# Timed access to models (LP-0MS4ETBNO0022QAC)
#
# Provider entries may carry an optional ``available_times`` list of
# "HH:MM-HH:MM" windows (interpreted in UTC, end exclusive, overnight ranges
# wrap past midnight). A provider whose current UTC time is outside all of its
# windows is skipped during fallback resolution exactly like a provider in
# cooldown. Providers without ``available_times`` remain unrestricted
# (backward compatible).
#
# Malformed window strings are logged and treated as unrestricted (fail-open)
# so a config typo never breaks proxy startup.
# ---------------------------------------------------------------------------

# Lazy parse cache: tuple(raw window strings) -> tuple((start_min, end_min), ...)
# or None when the provider is unrestricted. Keyed by the raw strings so the
# unhashable provider dict is never used as a key.
_NOT_CACHED = object()
_WINDOW_PARSE_CACHE: dict[tuple[str, ...], tuple[tuple[int, int], ...] | None] = {}

# ---------------------------------------------------------------------------
# Three-tier retry system (LP-0MRE8G94H005ZBLV, LP-0MRFEXXVC001RYKB)
#
# The proxy has three layers of retry for remote upstream requests:
#
# Tier 1 — Per-stream retries (proxy_remote.py:_handle_remote_streaming)
#   - Fires on upstream stall (idle timeout) or httpx ReadTimeout
#   - Bounded exponential backoff: base_delay * 2^attempt, capped at max_delay
#   - Configurable via server.upstream_retry_* config keys
#   - After max_attempts exhausted, yields finish_reason:error and the
#     caller (provider.py fallback chain) routes to the next provider
#   - Total max wait time approximates: sum of capped backoff delays
#
# Tier 2 — Provider-level cooldown (provider.py)
#   - Applied after a provider fails (Tier 1 exhausted or other failure)
#   - Provider is marked unavailable for cooldown_seconds
#   - Uses its own exponential backoff via _provider_failure_count
#   - Configurable via server.provider_cooldown_seconds
#
# Tier 3 — Cross-request stall circuit breaker
#           (stall_circuit_breaker.py:_check_stall_circuit_breaker)
#   - Tracks stall frequency per provider within a sliding time window
#   - When stall count exceeds threshold within window, marks provider
#     unavailable via the same Tier 2 cooldown mechanism
#   - Configurable via server.upstream_stall_window_seconds,
#     server.upstream_stall_threshold, server.upstream_stall_cooldown_seconds
#   - Default: 3 stalls within 300s window triggers 180s cooldown
#
# Interaction:
#   Tier 1 retries fire first for streaming connections that stall.
#   If Tier 1 exhausts (max retries reached), the stall circuit breaker
#   (Tier 3) records the stall. If the threshold is exceeded across
#   requests, the provider is marked unavailable via Tier 2 cooldown.
#   For non-streaming errors (e.g., HTTP 4xx/5xx), Tier 1 is bypassed
#   and Tier 2 cooldown applies directly.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cached-tokens-based routing state for smart routing
# (LP-0MRP44W7I0085I6N)
# ---------------------------------------------------------------------------

# Per-(model, session) cached-tokens ratio: (model_name, session_id) -> ratio.
# The ratio is cached_tokens / prompt_tokens from the last local response.
# A ratio of 1.0 means ALL tokens were cached (fully warm).
# A ratio < 1 means at least some tokens needed recomputation (cache partially
# or fully cold).
# When no entry exists, defaults to 0.0 (cold - conservative).
_last_cached_ratio: dict[tuple[str, str], float] = {}

# Maximum number of entries in _last_cached_ratio to prevent unbounded growth.
_MAX_CACHED_RATIO_ENTRIES = 1000


def update_cached_ratio(
    model_name: str,
    session_id: str | None,
    cached_tokens: int,
    prompt_tokens: int,
) -> None:
    """Update the cached-tokens ratio for a (model, session) pair.

    The ratio is computed as cached_tokens / prompt_tokens, with safe
    handling for zero prompt_tokens (returns 0.0).

    To prevent unbounded memory growth, the dict is capped at
    ``_MAX_CACHED_RATIO_ENTRIES``. When the cap is reached, the oldest
    entry (first inserted) is evicted.
    """
    if not model_name or not session_id:
        return
    if prompt_tokens <= 0:
        ratio = 0.0
    else:
        ratio = min(1.0, cached_tokens / prompt_tokens)
    key = (model_name, session_id)
    # Enforce cap: if adding a new entry would exceed the limit, evict oldest
    if key not in _last_cached_ratio and len(_last_cached_ratio) >= _MAX_CACHED_RATIO_ENTRIES:
        try:
            # Evict oldest entry (dict maintains insertion order in Python 3.7+)
            _last_cached_ratio.pop(next(iter(_last_cached_ratio)))
        except (StopIteration, KeyError):
            pass
    _last_cached_ratio[key] = ratio


def _get_cached_ratio(model_name: str, session_id: str | None) -> float:
    """Get the cached-tokens ratio for a (model, session) pair.

    Returns 0.0 (cold) when no entry exists (conservative default).
    """
    if not model_name or not session_id:
        return 0.0
    return _last_cached_ratio.get((model_name, session_id), 0.0)


def _extract_cached_tokens_from_usage(usage: dict | None) -> int:
    """Extract cached_tokens from a usage dict.

    Reads ``prompt_tokens_details.cached_tokens`` from the usage data
    returned by llama.cpp in the SSE response. Returns 0 when the field
    is missing or the dict is None.
    """
    if not isinstance(usage, dict):
        return 0
    try:
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            return int(details.get("cached_tokens", 0) or 0)
    except (ValueError, TypeError):
        pass
    return 0


def _extract_cached_tokens_from_sse_text(sse_text: str) -> int:
    """Extract cached_tokens from full SSE response text.

    Parses each ``data:`` line looking for a ``usage`` field with
    ``prompt_tokens_details.cached_tokens``. The usage event is typically
    carried in the final chunk of an SSE stream alongside ``finish_reason``.

    Returns 0 when no usage data is found.
    """
    usage = _extract_usage_from_sse_text(sse_text)
    return _extract_cached_tokens_from_usage(usage)


def _extract_usage_from_sse_text(sse_text: str) -> dict | None:
    """Extract the full usage dict from SSE response text.

    Parses each ``data:`` line looking for a ``usage`` field. The usage
    event is typically carried in the final chunk of an SSE stream
    alongside ``finish_reason``. Returns the LAST usage dict found (the
    final chunk is authoritative) or None when absent.

    (LP-0MS9GAN2P009KK6G: wire real cached_tokens from local responses)
    """
    if not sse_text:
        return None
    import json
    last_usage: dict | None = None
    for line in sse_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
            if isinstance(data, dict) and isinstance(data.get("usage"), dict):
                last_usage = data["usage"]
        except (json.JSONDecodeError, Exception):
            continue
    return last_usage


def _estimate_prompt_tokens_for_routing(body_json: dict, tokenizer=None) -> int:
    """Estimate prompt token count for smart-routing decisions.

    Concatenates all message content (including reasoning_content
    and tool_calls) and counts tokens using the native tokenizer when one
    is supplied (LP-0MSEQ71IF0003FRT), otherwise tiktoken (via
    ``count_text_tokens``) for accurate token counting.  Falls back to a
    conservative byte-based heuristic (1 byte per token) when tiktoken is
    unavailable.

    Using tiktoken guarantees correct routing decisions regardless of
    content density — the previous byte heuristic with ``// 2`` could
    underestimate by 2× for very dense content (hex, compact JSON with
    bpt ~1.2), causing the cache-cold bypass to miss requests with 70K+
    actual tokens.

    Args:
        body_json: Parsed request body.
        tokenizer: Optional native tokenizer (e.g. from
            ``_get_tokenizer_for_model``). When provided, counts with it
            directly (exact Qwen3-native counts); falls back to tiktoken
            on any encode error.
    """
    if not isinstance(body_json, dict):
        return 0
    messages = body_json.get("messages", [])
    if not messages:
        return 0

    parts: list[str] = []
    total_bytes = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        _role = msg.get("role", "")
        # Count content field (system messages included — LP-0MRGT35H1003D1PM)
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
            total_bytes += len(content.encode("utf-8"))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text = str(item["text"])
                    parts.append(text)
                    total_bytes += len(text.encode("utf-8"))
        # Count reasoning_content (assistant messages with long reasoning)
        reasoning = msg.get("reasoning_content")
        if isinstance(reasoning, str):
            parts.append(reasoning)
            total_bytes += len(reasoning.encode("utf-8"))
        # Count tool_calls (function names and arguments)
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_func = tc.get("function")
                if isinstance(tc_func, dict):
                    name = tc_func.get("name", "")
                    if isinstance(name, str):
                        parts.append(name)
                        total_bytes += len(name.encode("utf-8"))
                    args = tc_func.get("arguments", "")
                    if isinstance(args, str):
                        parts.append(args)
                        total_bytes += len(args.encode("utf-8"))

    if not parts:
        return 0

    # Primary path: native tokenizer when provided (exact Qwen3 counts).
    if tokenizer is not None:
        try:
            text = " ".join(parts)
            return len(tokenizer.encode(text).ids)
        except Exception:
            # Fall through to tiktoken on any native-encode error.
            pass

    # Primary path: use tiktoken for accurate token counting.
    # This correctly handles all content densities (code, JSON, text).
    try:
        from proxy.utils import count_text_tokens
        text = " ".join(parts)
        return count_text_tokens(text)
    except Exception:
        # Fallback: ultra-conservative byte-based heuristic (~1 byte per
        # token).  This overestimates for all content types (code bpt ~3,
        # JSON ~4, text ~4) guaranteeing the bypass always fires for large
        # requests even when tiktoken is unavailable.
        return max(1, total_bytes // 1)


def _get_tokenizer_for_model(
    model_config: dict | None,
    server_config: dict | None = None,
) -> tuple[object | None, float]:
    """Resolve the native tokenizer + effective multiplier for a model.

    Single ordered resolution chain (LP-0MSEQ71IF0003FRT):

    1. Model entry has ``tokenizer: <name>`` that loads -> use that
       tokenizer, **multiplier forced to 1.0** (a native tokenizer is
       exact; applying the ~1.69x cl100k multiplier on top would
       over-count ~69% and wrongly push requests remote).
    2. No tokenizer named -> ``cl100k_base`` **+ multiplier** (per-model
       override wins, else server-level, else 1.0) — today's behavior,
       unchanged.
    3. Named tokenizer exists but fails to load (missing file / import
       error) -> warn + fall back to step 2.

    Shared by BOTH the routing estimate (provider.py
    ``_estimate_prompt_tokens_for_routing``) and the persistence estimate
    (session.py ``_estimate_slot_prompt_tokens``) so the routing clamp and
    the persistence cap can never disagree (AC3).

    Returns:
        ``(tokenizer_or_None, multiplier)``.
    """
    multiplier = _get_token_estimate_multiplier(server_config or {}, model_config)
    if not isinstance(model_config, dict):
        return None, multiplier
    tokenizer_name = model_config.get("tokenizer")
    if not tokenizer_name:
        return None, multiplier
    try:
        from proxy.tokenizers import get_tokenizer

        tokenizer = get_tokenizer(tokenizer_name)
    except Exception:
        tokenizer = None
    if tokenizer is None:
        logger.warning(
            "tokenizer %r unavailable; falling back to tiktoken + multiplier %.3f",
            tokenizer_name,
            multiplier,
        )
        return None, multiplier
    return tokenizer, 1.0


def _get_token_estimate_multiplier(config: dict, model_config: dict | None = None) -> float:
    """Server-level ``token_estimate_multiplier`` with per-model override.

    Accounts for tokenizer mismatch between tiktoken (cl100k) and the local
    model's native tokenizer. The ctx-size evaluation (LP-0MSAOQTJS000FFVM)
    found cl100k undercounts Qwen3 native tokens ~1.69x for dense prose, so
    routing clamps and the persistence cap must compare Qwen3-native token
    counts (LP-0MSEGPO77005CYCQ F2).

    Resolution order:
    1. Per-model ``token_estimate_multiplier`` on the model entry (wins).
    2. Server-level ``token_estimate_multiplier`` in the server config.
    3. 1.0 (no adjustment) when neither is set.
    """
    server_cfg = config.get("server", config) if isinstance(config, dict) else {}
    try:
        server_mult = float(server_cfg.get("token_estimate_multiplier", 1.0) or 1.0)
    except (ValueError, TypeError):
        server_mult = 1.0
    if model_config:
        try:
            model_mult = float(model_config.get("token_estimate_multiplier", 1.0) or 1.0)
        except (ValueError, TypeError):
            model_mult = 1.0
        if model_mult != 1.0:
            return model_mult
    return server_mult


def _get_large_context_threshold(config: dict) -> int:
    """Read the large-context fallback threshold from config.

    Supports both nested and flat config keys for production and test
    compatibility.  Also supports the legacy key name
    ``local_large_context_fallback_threshold`` for backward compatibility.

    Returns 0 when not configured (disabled).
    """
    # New key first (preferred)
    val = config.get("local_large_context_cold_cache_threshold")
    if val is None:
        val = config.get("server", {}).get("local_large_context_cold_cache_threshold")
    # Fall back to legacy key
    if val is None:
        val = config.get("local_large_context_fallback_threshold")
    if val is None:
        val = config.get("server", {}).get("local_large_context_fallback_threshold", 0)
    try:
        return max(0, int(val or 0))
    except (ValueError, TypeError):
        return 0


def _get_warm_cache_threshold(config: dict) -> int:
    """Read the context-too-large (warm-cache) total-context threshold.

    When ``estimated_tokens`` exceeds this value, local is bypassed with
    skip reason ``context_too_large`` (LP-0MSF8XDG7000PERM) regardless of
    cache state — total context is too large for the local model slot.

    Supports both nested and flat config keys for production and test
    compatibility.  Config default: 100000.

    Returns 0 when not configured (disabled).
    """
    val = config.get("local_large_context_warm_cache_threshold")
    if val is None:
        val = config.get("server", {}).get(
            "local_large_context_warm_cache_threshold", 0
        )
    try:
        return max(0, int(val or 0))
    except (ValueError, TypeError):
        return 0


def _get_local_model_ctx_size(config: dict) -> int:
    """Read the local model's total context size (across all slots).

    Mirrors the per-model ``ctx-size`` in models.ini for the local model
    (LP-0MSAZXXDY005AWA1). Used to compute the actual per-slot context
    (ctx_size / active_slots) which clamps the large-context routing
    thresholds. 0 disables the clamp.
    """
    val = config.get("local_model_ctx_size")
    if val is None:
        val = config.get("server", {}).get("local_model_ctx_size", 0)
    try:
        return max(0, int(val or 0))
    except (ValueError, TypeError):
        return 0


def _get_active_local_slots(config: dict) -> int:
    """Return the number of currently active local slots.

    Prefers the live slot scheduler (schedule-aware) when available;
    otherwise falls back to ``session_slot_pool_size`` from config.
    Returns 1 as a safe default when nothing is configured.
    """
    # Live slot scheduler (schedule-aware; e.g. 6 day / 8 night).
    try:
        import proxy.server as _srv

        scheduler = getattr(_srv, "slot_scheduler", None)
        if scheduler is not None and hasattr(scheduler, "get_active_slot"):
            slots = scheduler.get_active_slot()
            if slots and int(slots) > 0:
                return int(slots)
    except Exception:
        pass

    server_cfg = config.get("server", config)
    try:
        val = server_cfg.get("session_slot_pool_size")
        if val is not None and int(val) > 0:
            return int(val)
    except (ValueError, TypeError):
        pass
    return 1


# Default fraction of the effective per-slot context at which the proxy
# emits a session-context-pressure warning suggesting compaction
# (LP-0MSDCLQ2W001LGWC). Configurable via ``context_pressure_warn_ratio``
# on the server config; 0 disables the warning.
_DEFAULT_CONTEXT_PRESSURE_WARN_RATIO = 0.8


def _get_active_local_ctx_size(config: dict) -> int:
    """Return the currently active local context size (schedule-aware).

    Prefers the live slot scheduler's per-period ``ctx_size`` (the ACTIVE
    schedule entry's override, LP-0MSLNK96T0018W4D); falls back to the
    static ``local_model_ctx_size`` from config, which ``restart_services``
    keeps in sync with the last applied transition. Mirrors
    ``_get_active_local_slots``.
    """
    try:
        import proxy.server as _srv

        scheduler = getattr(_srv, "slot_scheduler", None)
        if scheduler is not None and hasattr(scheduler, "get_active_ctx_size"):
            ctx = scheduler.get_active_ctx_size()
            if ctx and int(ctx) > 0:
                return int(ctx)
    except Exception:
        pass
    return _get_local_model_ctx_size(config)


def _get_context_pressure_warn_ratio(config: dict) -> float:
    """Read the context-pressure warning ratio from config.

    Supports both nested (server.*) and flat keys. 0 disables the warning.
    Defaults to ``_DEFAULT_CONTEXT_PRESSURE_WARN_RATIO`` (0.8).
    """
    val = config.get("context_pressure_warn_ratio")
    if val is None:
        val = config.get("server", {}).get(
            "context_pressure_warn_ratio", _DEFAULT_CONTEXT_PRESSURE_WARN_RATIO
        )
    try:
        ratio = float(val or 0)
    except (ValueError, TypeError):
        return _DEFAULT_CONTEXT_PRESSURE_WARN_RATIO
    return max(0.0, ratio)


def context_pressure_ratio(estimated_tokens: int, ctx_size: int, slots: int) -> float:
    """Fraction of the effective per-slot context consumed by a session.

    Uses the same per-slot computation as the routing clamp
    (``ctx_size // slots - _LOCAL_ROUTING_OUTPUT_HEADROOM``); returns 0.0
    when the computation is not meaningful (ctx_size <= 0 or per-slot leaves
    no room for output tokens).

    Args:
        estimated_tokens: Session estimated prompt/context tokens.
        ctx_size: Total local model context (``local_model_ctx_size``).
        slots: Active local slot count.

    Returns:
        A float ratio (0.0 when disabled; > 1.0 when the session exceeds
        the effective per-slot context).
    """
    cap = effective_per_slot_threshold(ctx_size, slots)
    if cap <= 0 or estimated_tokens <= 0:
        return 0.0
    return estimated_tokens / cap


def should_warn_context_pressure(estimated_tokens: int, config: dict) -> bool:
    """Whether a session at ``estimated_tokens`` should trigger the
    context-pressure compaction warning.

    Uses the effective per-slot context (clamped with output headroom) and
    the configured ``context_pressure_warn_ratio`` (default 0.8). Returns
    False when the clamp is disabled (ctx_size 0), the ratio is 0, or the
    session is below the ratio.

    Args:
        estimated_tokens: Session estimated prompt/context tokens.
        config: Proxy configuration (flat or nested ``server`` dict).

    Returns:
        True when the session should be flagged for compaction.
    """
    ctx_size = _get_local_model_ctx_size(config)
    if ctx_size <= 0 or estimated_tokens <= 0:
        return False
    slots = _get_active_local_slots(config)
    ratio = _get_context_pressure_warn_ratio(config)
    if ratio <= 0:
        return False
    return context_pressure_ratio(estimated_tokens, ctx_size, slots) >= ratio


# Output-token headroom reserved below the per-slot context when clamping
# routing thresholds (LP-0MSAZXXDY005AWA1). Ensures prompts routed local
# leave room for the model's completion tokens in the KV slot.
_LOCAL_ROUTING_OUTPUT_HEADROOM = 4096

# Default minimum effective per-slot large-context routing threshold.
# Below this, every realistic agent session exceeds the clamp and ALL local
# traffic silently bypasses to remote (LP-0MSAOQTJS000FFVM failure mode).
# Configurable via ``min_local_routing_threshold`` on the server config.
_DEFAULT_MIN_LOCAL_ROUTING_THRESHOLD = 10000


def effective_per_slot_threshold(ctx_size: int, slots: int) -> int:
    """Compute the effective per-slot large-context routing threshold.

    Mirrors the clamp applied by ``_effective_large_context_thresholds``:
    ``ctx_size // slots - _LOCAL_ROUTING_OUTPUT_HEADROOM`` (LP-0MSAZXXDY005AWA1).

    Returns 0 when the computation is not meaningful:
    - ``ctx_size`` <= 0 (clamp disabled)
    - ``slots`` <= 0
    - per-slot context does not leave room for output tokens
      (per_slot <= headroom)
    """
    if ctx_size <= 0 or slots <= 0:
        return 0
    per_slot = ctx_size // slots
    if per_slot <= _LOCAL_ROUTING_OUTPUT_HEADROOM:
        return 0
    return per_slot - _LOCAL_ROUTING_OUTPUT_HEADROOM


def _effective_large_context_thresholds(config: dict) -> tuple[int, int]:
    """Return (cold, warm) thresholds clamped to the actual per-slot context.

    The configured cold/warm thresholds were tuned assuming a fixed per-slot
    context (e.g. 65000). When the real per-slot context is smaller
    (``local_model_ctx_size`` / active slots, minus output headroom), prompts
    larger than the slot capacity would be routed local and truncate with
    ``finish_reason=length`` (context exhaustion) — surfaced by pi as the
    misleading "maximum output token limit" error (LP-0MSAZXXDY005AWA1).

    The warm threshold is clamped to the actual per-slot context because it
    represents a **hard capacity limit**: total context exceeding the slot
    must be routed remote to prevent context exhaustion.

    The cold threshold is an **economic new-token threshold** and must NOT be
    clamped — it defines the band (cold, warm] where the cached_ratio routing
    check (Check 2 in ``_should_skip_local``) operates. Clamping cold to the
    same cap as warm collapses the band to zero width, making the ratio check
    unreachable dead code (LP-0MSI2M5BT004BCDP).

    Returns the configured thresholds unchanged when ``local_model_ctx_size``
    is 0 (clamp disabled) or per-slot context cannot be computed.
    """
    cold = _get_large_context_threshold(config)
    warm = _get_warm_cache_threshold(config)
    ctx_size = _get_active_local_ctx_size(config)
    slots = _get_active_local_slots(config)

    cap = effective_per_slot_threshold(ctx_size, slots)
    if cap <= 0:
        return cold, warm

    # Only clamp the WARM threshold to the per-slot cap (hard capacity limit).
    # COLD stays as the economic new-token threshold so the (cold, warm] band
    # remains non-empty for Check 2 (cached_ratio routing) to operate in.
    if warm > 0:
        warm = min(warm, cap)
    return cold, warm


def _collect_local_ctx_pairs(config: dict) -> list[tuple[int, int]]:
    """All (ctx_size, slots) pairs the proxy may run with.

    The static ``local_model_ctx_size``/``session_slot_pool_size`` pair plus
    every ``slot_schedule`` entry, where an entry's per-period ``ctx_size``
    overrides the global value (falling back to it when unset)
    (LP-0MSLNK96T0018W4D). Pairs are de-duplicated while preserving order.
    """
    server_cfg = config.get("server", config)
    ctx_size = _get_local_model_ctx_size(config)
    pairs: list[tuple[int, int]] = []
    try:
        pool = int(server_cfg.get("session_slot_pool_size", 0) or 0)
    except (ValueError, TypeError):
        pool = 0
    if pool > 0:
        pairs.append((ctx_size, pool))

    try:
        from proxy.slot_scheduler import SlotScheduleConfig

        schedule = SlotScheduleConfig.from_server_config(server_cfg)
    except Exception:
        schedule = None
    if schedule is not None and schedule.enabled:
        for entry in schedule.entries:
            entry_ctx = entry.ctx_size if entry.ctx_size is not None else ctx_size
            if entry.slots > 0 and (entry_ctx, entry.slots) not in pairs:
                pairs.append((entry_ctx, entry.slots))
    return pairs


def _get_min_local_routing_threshold(config: dict) -> int:
    """Read the minimum effective per-slot routing threshold from config.

    Supports both nested and flat config keys. Defaults to
    ``_DEFAULT_MIN_LOCAL_ROUTING_THRESHOLD`` (10000). An explicit 0 disables
    the minimum check (consistent with ``local_model_ctx_size: 0`` disabling
    the clamp).
    """
    val = config.get("min_local_routing_threshold")
    if val is None:
        val = config.get("server", {}).get(
            "min_local_routing_threshold", _DEFAULT_MIN_LOCAL_ROUTING_THRESHOLD
        )
    try:
        return max(0, int(val or 0))
    except (ValueError, TypeError):
        return _DEFAULT_MIN_LOCAL_ROUTING_THRESHOLD


def validate_local_routing_config(config: dict) -> list[str]:
    """Validate the ctx-size / slot-count routing clamp configuration.

    Computes the effective per-slot large-context routing threshold
    (``ctx_size // slots - _LOCAL_ROUTING_OUTPUT_HEADROOM``, mirroring
    ``_effective_large_context_thresholds``) for EVERY (ctx_size, slots)
    pair the proxy may run with — the static
    ``local_model_ctx_size``/``session_slot_pool_size`` AND all
    ``slot_schedule`` entries (each entry's per-period ``ctx_size``
    overriding the global when set, LP-0MSLNK96T0018W4D) — and reports a
    problem when the threshold falls below the configured minimum (default
    10000 tokens).

    A threshold this small means every realistic agent session exceeds the
    clamp, silently bypassing ALL local traffic to remote providers (the
    2026-08-02 failure mode, LP-0MSAOQTJS000FFVM).

    Returns a list of human-readable problem descriptions (empty when the
    config is consistent). When ``min_local_routing_threshold_fatal`` is
    set, each problem is prefixed with ``FATAL: `` so callers can fail
    startup; otherwise callers log a WARNING.
    """
    min_threshold = _get_min_local_routing_threshold(config)
    if min_threshold <= 0:
        return []  # minimum check disabled (min_local_routing_threshold: 0)

    fatal = bool(
        config.get("server", {}).get("min_local_routing_threshold_fatal", False)
        or config.get("min_local_routing_threshold_fatal", False)
    )

    problems: list[str] = []
    for ctx, slots in _collect_local_ctx_pairs(config):
        if ctx <= 0:
            continue  # clamp disabled for this pair
        threshold = effective_per_slot_threshold(ctx, slots)
        if threshold <= 0:
            # Per-slot context leaves no room for output tokens; the clamp
            # leaves thresholds unchanged rather than clamping, so nothing
            # below the minimum can be derived here.
            continue
        if threshold < min_threshold:
            msg = (
                f"local_model_ctx_size={ctx} with {slots} slots yields "
                f"effective per-slot large-context routing threshold "
                f"{threshold} ({ctx}//{slots} - "
                f"{_LOCAL_ROUTING_OUTPUT_HEADROOM} headroom), below the "
                f"minimum of {min_threshold}. Every prompt above {threshold} "
                f"tokens bypasses local to remote, silently disabling local "
                f"routing (LP-0MSAOQTJS000FFVM). Increase ctx-size or reduce "
                f"slots."
            )
            problems.append(f"FATAL: {msg}" if fatal else msg)
    return problems


async def _estimate_effective_prompt_tokens_for_routing(
    request,
    body_json: dict,
    tokenizer=None,
) -> int:
    """Estimate prompt tokens for routing, including active session history.

    ``proxy_with_fallback`` runs before local session handling has a chance to
    compute/forward deltas. For long-running sessions, the incoming request
    body may be small while the active session history is very large.

    To avoid missing cache-cold large-context bypass in that case, this helper
    takes the max of:

    - incoming request token estimate
    - existing session history token estimate (if a session header resolves)

    ``tokenizer`` (optional native tokenizer, LP-0MSEQ71IF0003FRT) is passed
    through to both estimates so routing clamps compare Qwen3-native counts
    consistently with the persistence cap.
    """
    estimated_tokens = _estimate_prompt_tokens_for_routing(body_json, tokenizer=tokenizer)

    try:
        from proxy.session import _resolve_session_id_header

        session_id, _ = _resolve_session_id_header(getattr(request, "headers", {}))
        if not session_id:
            return estimated_tokens

        import proxy.server as _srv

        session_manager = getattr(_srv, "session_manager", None)
        if session_manager is None:
            return estimated_tokens

        session = await session_manager.get(session_id)
        if session is None:
            return estimated_tokens

        session_messages = getattr(session, "messages", None)
        if not isinstance(session_messages, list) or not session_messages:
            return estimated_tokens

        session_tokens = _estimate_prompt_tokens_for_routing(
            {"messages": session_messages}, tokenizer=tokenizer
        )
        if session_tokens > estimated_tokens:
            logger.info(
                "routing_estimate_session session=%s request_tokens=%d session_tokens=%d",
                session_id[:8],
                estimated_tokens,
                session_tokens,
            )

        return max(estimated_tokens, session_tokens)
    except Exception:
        return estimated_tokens


def _should_skip_local(
    model_name: str,
    session_id: str | None,
    body_json: dict,
    cold_cache_threshold: int,
    estimated_tokens: int | None = None,
    warm_cache_threshold: int = 0,
) -> bool:
    """Determine whether a local provider should be skipped due to large context.

    Implements a two-tier check:

    1. **Context-too-large threshold (hard cap):** If ``estimated_tokens``
       exceeds ``warm_cache_threshold``, bypass local regardless of cache
       state (skip reason ``context_too_large``). This prevents routing
       excessively large total contexts to local even when the cache is
       warm.

    2. **Cold-cache new-token check:** Calculates the number of uncached
       tokens: ``new_tokens = int(estimated_tokens * (1 - cached_ratio))``.
       If ``new_tokens > cold_cache_threshold``, the prefill is too
       expensive, so bypass local.  Otherwise route local.

    This replaces the old binary ``ratio < 1.0`` check which was effectively
    always true (there is always new content in a conversation), causing
    every request above the threshold to bypass local even when the actual
    prefill cost was trivial.

    A ratio of 0.0 (default for unknown sessions) means conservative
    behavior: bypass if new_tokens exceed threshold.

    A ratio of 1.0 means the cache was fully warm, so new_tokens = 0 and
    local is always used (unless warm_cache_threshold blocks it).

    A threshold of 0 disables the bypass entirely.

    Args:
        model_name: The llama_model name of the local provider.
        session_id: The session ID (for per-session ratio tracking).
        body_json: The parsed request body.
        cold_cache_threshold: Token threshold for bypass. 0 = disabled.
        estimated_tokens: Optional pre-computed estimate.
        warm_cache_threshold: Hard cap on total context. 0 = disabled.

    Returns:
        True if local should be skipped, False for normal local routing.
    """
    if cold_cache_threshold <= 0:
        return False
    if estimated_tokens is None:
        estimated_tokens = _estimate_prompt_tokens_for_routing(body_json)

    # Check 1: Warm-cache threshold (hard cap on total context)
    if warm_cache_threshold > 0 and estimated_tokens > warm_cache_threshold:
        return True

    # If estimated tokens are below cold cache threshold, always route local
    if estimated_tokens <= cold_cache_threshold:
        return False

    # Check 2: Dynamic new-token calculation
    ratio = _get_cached_ratio(model_name, session_id)
    new_tokens = int(estimated_tokens * (1 - ratio))
    return new_tokens > cold_cache_threshold


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_provider(
    model_config: dict,
    failed_provider: str | None = None,
) -> dict | None:
    """Get the next available provider for a model config.

    Iterates through the model's ordered `providers` list and returns the
    first provider that:
    - Is not the `failed_provider` (if specified)
    - Does not share the failed provider's failure domain (same endpoint /
      brand — LP-0MSG45I8Q0020N1F)
    - Is not in cooldown (entry name OR provider brand —
      LP-0MSG45LOO007K236)
    - Does not belong to a usage-limit ACCOUNT with a pending usage-limit
      reset (LP-0MSLJPOCC0001ROJ / LP-0MSMBWB23009XYPW)

    Args:
        model_config: Model configuration dict. Must contain a ``providers``
                      key whose value is an ordered list of provider configs.
        failed_provider: Optional name of a provider that just failed and
                         should be skipped.

    Returns:
        A provider config dict (with keys ``name``, ``type``, etc.), or
        ``None`` if no provider is available.
    """
    providers: list[dict[str, Any]] | None = model_config.get("providers")
    if not providers:
        return None

    # When a provider failed, also skip any OTHER entry sharing its failure
    # domain (normalized endpoint, or brand for local providers) so the chain
    # hops to a genuinely different gateway rather than retrying the same one
    # through a second API key (LP-0MSG45I8Q0020N1F).
    failed_domain: str | None = None
    if failed_provider:
        for candidate in providers:
            if isinstance(candidate, dict) and candidate.get("name") == failed_provider:
                failed_domain = _failure_domain_key(candidate)
                break

    for provider_cfg in providers:
        name = provider_cfg.get("name", "")
        if failed_provider and name == failed_provider:
            continue
        domain = _failure_domain_key(provider_cfg)
        if failed_domain is not None and domain == failed_domain:
            logger.info(
                "Skipping provider=%s: same failure domain as %s (%s)",
                name,
                failed_provider,
                domain,
            )
            continue
        # Usage-limit reset pending (LP-0MSLJPOCC0001ROJ). Quarantine is keyed
        # on the API-key ACCOUNT, not the endpoint: distinct api_key_env
        # entries on the same gateway have independent limits
        # (LP-0MSMBWB23009XYPW).
        usage_key = _usage_limit_account_key(provider_cfg)
        reset_remaining = _usage_reset_remaining(usage_key)
        if reset_remaining > 0:
            logger.info(
                "Skipping provider=%s: usage_limit_reset_pending "
                "(account=%s, reset_at=%s, reset_in=%ds)",
                name,
                usage_key,
                datetime.fromtimestamp(_usage_reset_at[usage_key], tz=UTC).isoformat(),
                int(reset_remaining),
            )
            continue
        cooldown_key = _entry_cooldown_key(provider_cfg)
        if cooldown_key is not None:
            remaining = _provider_cooldown_remaining(cooldown_key)
            logger.info(
                "Skipping provider=%s: %s in cooldown (%ds remaining)",
                name,
                cooldown_key,
                remaining,
            )
            continue
        if not _is_within_allowed_window(provider_cfg):
            logger.info(
                "Skipping provider=%s: outside its available_times window (UTC)",
                name,
            )
            continue
        return provider_cfg

    return None


def get_model_type(model_config: dict) -> str | None:
    """Determine the model type from the providers list.

    Returns ``"local"`` if the first provider is a local provider,
    ``"remote"`` if it is remote, or ``None`` if no providers are defined.

    This replaces the legacy ``model_config["type"]`` field.
    """
    providers: list[dict[str, Any]] | None = model_config.get("providers")
    if not providers:
        return None
    first = providers[0]
    ptype = first.get("type")
    if ptype in ("local", "remote"):
        return ptype
    return None


def get_local_model_name_from_providers(model_config: dict) -> str | None:
    """Extract the llama_model name from the providers list.

    Searches the ordered ``providers`` list for a local provider and
    returns its ``llama_model`` value.

    This replaces the legacy ``model_config["llama_model"]`` field.

    Returns ``None`` if no local provider is found.
    """
    providers: list[dict[str, Any]] | None = model_config.get("providers")
    if not providers:
        return None
    for p in providers:
        if isinstance(p, dict) and p.get("type") == "local":
            return p.get("llama_model")
    return None


def get_remote_endpoint(model_config: dict) -> str | None:
    """Extract the endpoint URL from the providers list.

    Searches the ordered ``providers`` list for a remote provider and
    returns its ``endpoint`` value.

    This replaces the legacy ``model_config["endpoint"]`` field.

    Returns ``None`` if no remote provider is found.
    """
    providers: list[dict[str, Any]] | None = model_config.get("providers")
    if not providers:
        return None
    for p in providers:
        if isinstance(p, dict) and p.get("type") == "remote":
            return p.get("endpoint")
    return None


def mark_provider_unavailable(
    provider_name: str,
    cooldown_seconds: float,
    use_exponential_backoff: bool = False,
) -> None:
    """Mark a provider as unavailable for the given cooldown duration.

    When *use_exponential_backoff* is ``True``, the actual cooldown is
    computed via exponential backoff based on consecutive failure count
    instead of using *cooldown_seconds* directly:

        cooldown = min(BACKOFF_BASE * 2^failure_count, BACKOFF_MAX)

    The failure count is incremented on each call and reset to 0 on
    successful provider response via ``_reset_provider_failure_count()``.

    Args:
        provider_name: Name of the provider to mark.
        cooldown_seconds: Number of seconds the provider should be
                          considered unavailable (used as max when
                          *use_exponential_backoff* is ``True``).
        use_exponential_backoff: If ``True``, apply exponential backoff.
                                 Default is ``False``.
    """
    if use_exponential_backoff:
        count = _provider_failure_count.get(provider_name, 0)
        backoff = min(
            _BACKOFF_BASE_SECONDS * (2 ** count),
            _BACKOFF_MAX_SECONDS,
        )
        cooldown_seconds = min(backoff, cooldown_seconds)
        _provider_failure_count[provider_name] = count + 1

    _provider_unavailable_until[provider_name] = time.time() + cooldown_seconds


def _reset_provider_failure_count(provider_name: str) -> None:
    """Reset the consecutive failure count for a provider on success.

    Removes the provider from the failure count dict so that the next
    failure starts with the base backoff interval.
    """
    _provider_failure_count.pop(provider_name, None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _provider_cooldown_remaining(provider_name: str) -> int:
    """Return the remaining cooldown seconds for a provider key.

    Returns 0 when the provider is not in cooldown or its cooldown has
    expired.  Expired entries are cleaned up lazily.

    A cooldown with less than one second remaining still reports ``1`` so
    entries are not treated as expired early (the exponential-backoff base
    cooldown is 1.0s).

    This check is global — it reads from the shared module-level dict, so
    any session calling this function sees the same cooldown state. There is
    no per-session isolation of cooldown state.
    """
    expiry = _provider_unavailable_until.get(provider_name)
    if expiry is None:
        return 0
    remaining = expiry - time.time()
    if remaining <= 0:
        # Cooldown expired — clean up entry
        del _provider_unavailable_until[provider_name]
        return 0
    return int(remaining) if remaining >= 1 else 1


def _is_provider_unavailable(provider_name: str) -> bool:
    """Check if a provider is currently in cooldown.

    Returns ``True`` if the provider is marked unavailable and its cooldown
    has not yet expired.  Expired entries are cleaned up lazily.

    This check is global — it reads from the shared module-level dict, so
    any session calling this function sees the same cooldown state. There is
    no per-session isolation of cooldown state.
    """
    return _provider_cooldown_remaining(provider_name) > 0


def _entry_cooldown_key(provider_cfg: dict) -> str | None:
    """Return the cooldown key under which a provider entry is unavailable.

    Checks BOTH the entry name and the provider brand (LP-0MSG45LOO007K236):
    the Tier-3 stall circuit breaker marks the provider BRAND (e.g.
    ``opencode-go``) unavailable via ``mark_provider_unavailable()``, but the
    fallback resolvers previously only checked the ENTRY name
    (``opencode-go-2-deepseek``).  The brand key was never consulted, so the
    breaker cooldown never blocked the entries pointing at the broken gateway.

    Returns the cooldown key (entry name or brand) that is currently cooling
    down, or ``None`` if the entry is eligible.
    """
    name = provider_cfg.get("name", "")
    if _is_provider_unavailable(name):
        return name
    brand = provider_cfg.get("provider")
    if brand and _is_provider_unavailable(brand):
        return brand
    return None


def _parse_window(window_str: str) -> tuple[int, int] | None:
    """Parse a single ``"HH:MM-HH:MM"`` window string (UTC).

    Returns ``(start_minutes, end_minutes)`` since midnight, or ``None`` for
    malformed input. End times are exclusive; overnight ranges (``end < start``)
    wrap past midnight.
    """
    try:
        start_str, end_str = window_str.split("-", 1)
        start_h, start_m = start_str.split(":")
        end_h, end_m = end_str.split(":")
        start_h, start_m, end_h, end_m = int(start_h), int(start_m), int(end_h), int(end_m)
        if not (0 <= start_h <= 23 and 0 <= end_h <= 23):
            return None
        if not (0 <= start_m <= 59 and 0 <= end_m <= 59):
            return None
        return (start_h * 60 + start_m, end_h * 60 + end_m)
    except Exception:
        return None


def _parse_available_times(provider_cfg: dict) -> tuple[tuple[int, int], ...] | None:
    """Parse a provider's ``available_times`` into window tuples.

    Returns ``None`` when the provider has no usable windows (unrestricted),
    otherwise a tuple of ``(start_minutes, end_minutes)`` windows in config
    order. Malformed entries are logged and skipped; if *every* entry is
    malformed the provider is treated as unrestricted (fail-open). Parsed
    results are cached per unique list of raw strings.
    """
    raw = provider_cfg.get("available_times")
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        logger.warning(
            "available_times for provider %r must be a list of \"HH:MM-HH:MM\" "
            "windows; ignoring (fail-open)",
            provider_cfg.get("name", "?"),
        )
        return None

    key = tuple(str(w) for w in raw)
    cached = _WINDOW_PARSE_CACHE.get(key, _NOT_CACHED)
    if cached is not _NOT_CACHED:
        return cached

    windows: list[tuple[int, int]] = []
    for entry in raw:
        parsed = _parse_window(str(entry))
        if parsed is None:
            logger.warning(
                "Invalid available_times entry %r for provider %r; ignoring "
                "(fail-open)",
                entry, provider_cfg.get("name", "?"),
            )
            continue
        windows.append(parsed)

    result: tuple[tuple[int, int], ...] | None = tuple(windows) if windows else None
    _WINDOW_PARSE_CACHE[key] = result
    return result


def _is_within_allowed_window(provider_cfg: dict, now_utc: datetime | None = None) -> bool:
    """Return ``True`` when the provider may be used at the given UTC time.

    Providers without ``available_times`` are always allowed (backward
    compatible). Windows are ``"HH:MM-HH:MM"`` interpreted in **UTC**; window
    start is inclusive, end is exclusive, and overnight ranges
    (``"22:00-02:00"``) wrap past midnight. When *now_utc* is omitted the
    current UTC wall-clock time is used.
    """
    windows = _parse_available_times(provider_cfg)
    if windows is None:
        return True
    if now_utc is None:
        now_utc = datetime.now(UTC)
    current_min = now_utc.hour * 60 + now_utc.minute
    for start_min, end_min in windows:
        if start_min < end_min:
            if start_min <= current_min < end_min:
                return True
        else:
            # Overnight window wraps past midnight
            if current_min >= start_min or current_min < end_min:
                return True
    return False


def _providers_outside_window(model_config: dict) -> list[dict[str, str]]:
    """Return ``{name, type}`` pairs for providers whose ``available_times``
    window excludes the current UTC time.

    Used to record ``outside_time_window`` diagnostics when a fallback chain is
    exhausted. Providers actually attempted this request cannot be outside
    their window (they were selected), so this set is exactly the providers
    skipped solely due to time windows.
    """
    result: list[dict[str, str]] = []
    for p in model_config.get("providers") or []:
        if isinstance(p, dict) and not _is_within_allowed_window(p):
            result.append({
                "name": p.get("name", "unknown"),
                "type": p.get("type", "remote"),
            })
    return result


def _parse_retry_after(response: Response) -> float | None:
    """Parse Retry-After header from a response.

    Supports both integer seconds and HTTP-date formats.

    Returns:
        Number of seconds to wait, or ``None`` if no Retry-After header.
    """
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return None
    # Try integer seconds first
    try:
        return float(retry_after)
    except ValueError:
        pass
    # Try HTTP-date format
    try:
        dt = parsedate_to_datetime(retry_after)
        now = time.time()
        dt_ts = dt.timestamp()
        if dt_ts > now:
            return dt_ts - now
        return 0.0
    except Exception:
        return None





def _get_cooldown_seconds(config: dict) -> float:
    """Read ``provider_cooldown_seconds`` from config, supporting both flat and nested formats.

    Checks ``config["provider_cooldown_seconds"]`` (flat) first for backward
    compatibility with unit tests, then falls back to
    ``config["server"]["provider_cooldown_seconds"]`` (nested) for production
    configs loaded from ``config.yaml``.  Defaults to 60.
    """
    val = config.get("provider_cooldown_seconds")
    if val is None:
        val = config.get("server", {}).get("provider_cooldown_seconds", 60)
    return float(val)


def _chain_hold_enabled(config: dict) -> bool:
    """Return True when the chain-hold feature is explicitly configured.

    The feature is enabled when either ``chain_hold_seconds`` or
    ``chain_hold_max_cycles`` is present (flat or ``server.*``). Production
    config.yaml ships both (300s / 3 cycles). When neither is present the
    fallback chain runs single-pass with no hold — exactly the legacy
    behavior — so existing unit tests that pass minimal config dicts are
    unaffected.
    """
    server_cfg = config.get("server", {}) if isinstance(config, dict) else {}
    return (
        "chain_hold_seconds" in config
        or "chain_hold_seconds" in server_cfg
        or "chain_hold_max_cycles" in config
        or "chain_hold_max_cycles" in server_cfg
    )


def _get_chain_hold_seconds(config: dict) -> float:
    """Read ``chain_hold_seconds`` from config (flat or ``server.*``).

    Supports the same flat-first / nested-fallback pattern as
    ``_get_cooldown_seconds`` for production configs loaded from
    ``config.yaml``. Defaults to 300 (LP-0MSH94Z7K007VKC9 AC1/AC5).
    """
    val = config.get("chain_hold_seconds")
    if val is None:
        val = config.get("server", {}).get("chain_hold_seconds", 300)
    try:
        return max(0.0, float(val or 0))
    except (ValueError, TypeError):
        return 300.0


def _get_chain_hold_max_cycles(config: dict) -> int:
    """Read ``chain_hold_max_cycles`` from config (flat or ``server.*``).

    0 = infinite (keep holding/retrying until the client disconnects).
    Defaults to 3 (LP-0MSH94Z7K007VKC9 AC2/AC5).
    """
    val = config.get("chain_hold_max_cycles")
    if val is None:
        val = config.get("server", {}).get("chain_hold_max_cycles", 3)
    try:
        return max(0, int(val or 0))
    except (ValueError, TypeError):
        return 3


def validate_chain_hold_config(config: dict) -> list[str]:
    """Validate the chain-hold configuration (LP-0MSH94Z7K007VKC9 AC5).

    Checks that ``chain_hold_seconds`` and ``chain_hold_max_cycles`` (flat or
    ``server.*``) are non-negative numbers when present, and flags the
    pathological combination of a zero hold interval with an unlimited cycle
    count (an unbounded busy-retry loop).

    Returns a list of human-readable problem descriptions (empty when the
    config is consistent).
    """
    problems: list[str] = []
    server_cfg = config.get("server", {}) if isinstance(config, dict) else {}

    seconds = config.get("chain_hold_seconds")
    if seconds is None:
        seconds = server_cfg.get("chain_hold_seconds")
    if seconds is not None:
        try:
            if float(seconds) < 0:
                problems.append(
                    f"server.chain_hold_seconds must be >= 0 (got {seconds!r})"
                )
        except (ValueError, TypeError):
            problems.append(
                f"server.chain_hold_seconds must be a number (got {seconds!r})"
            )

    max_cycles = config.get("chain_hold_max_cycles")
    if max_cycles is None:
        max_cycles = server_cfg.get("chain_hold_max_cycles")
    if max_cycles is not None:
        try:
            if int(max_cycles) < 0:
                problems.append(
                    f"server.chain_hold_max_cycles must be >= 0 (got {max_cycles!r})"
                )
        except (ValueError, TypeError):
            problems.append(
                f"server.chain_hold_max_cycles must be an integer (got {max_cycles!r})"
            )

    # Unbounded busy-retry: zero hold interval with unlimited cycles.
    try:
        seconds_f = float(seconds) if seconds is not None else None
        cycles_i = int(max_cycles) if max_cycles is not None else None
        if seconds_f is not None and cycles_i is not None and seconds_f == 0 and cycles_i == 0:
            problems.append(
                "server.chain_hold_seconds=0 with server.chain_hold_max_cycles=0 "
                "(infinite) creates an unbounded retry loop; set a positive hold "
                "interval or a finite cycle bound"
            )
    except (ValueError, TypeError):
        pass

    return problems


def _get_local_slot_retry_attempts(config: dict) -> int:
    """Read local slot-exhaustion retry attempts from config.

    Supports flat key (tests) and nested server key (runtime).
    """
    val = config.get("local_slot_exhaustion_retry_attempts")
    if val is None:
        val = config.get("server", {}).get("local_slot_exhaustion_retry_attempts", 0)
    try:
        return max(0, int(val or 0))
    except Exception:
        return 0


def _get_local_slot_retry_delay_seconds(config: dict) -> float:
    """Read local slot-exhaustion retry delay (seconds) from config."""
    val = config.get("local_slot_exhaustion_retry_delay_seconds")
    if val is None:
        val = config.get("server", {}).get("local_slot_exhaustion_retry_delay_seconds", 0.2)
    try:
        return max(0.0, float(val or 0.0))
    except Exception:
        return 0.2


def _get_slot_unavailable_retry_after(config: dict) -> float:
    """Read the short cooldown for slot-exhaustion (slot busy), default 5s.

    Distinct from provider_cooldown_seconds: a busy slot frees quickly (when
    the in-flight request finishes), so we use a short cooldown so the next
    request can retry local soon instead of waiting the full provider cooldown.
    """
    val = config.get("slot_unavailable_retry_after")
    if val is None:
        val = config.get("server", {}).get("slot_unavailable_retry_after", 5)
    try:
        return max(1.0, float(val or 5))
    except Exception:
        return 5.0


def _is_streaming_response(response: Response) -> bool:
    """Return True when response is a StreamingResponse (body is a generator).

    Such responses cannot be inspected for emptiness and should be treated as
    success when their status is 2xx.
    """
    return isinstance(response, StreamingResponse)


def _response_body_text(response: Response) -> str:
    """Best-effort extraction of text body for diagnostics/classification."""
    try:
        if hasattr(response, 'content'):
            b = response.content
        elif hasattr(response, 'body'):
            b = response.body
        else:
            b = None
        if b:
            return b.decode('utf-8', errors='replace') if isinstance(b, (bytes, bytearray)) else str(b)
    except Exception:
        return ""
    return ""


def _add_provider_header(response: Response, provider_name: str) -> Response:
    """Add X-Provider header to a response.

    Uses set (not append) to prevent duplicate headers on fallback responses.
    """
    response.headers["X-Provider"] = provider_name
    return response


def _build_resolved_model_value(provider_cfg: dict) -> str | None:
    """Build the X-Resolved-Model header value from a provider config.

    Returns ``<provider-name>/<model-id>`` or ``None`` if the config
    doesn't have the required fields.

    For local providers, uses ``llama_model`` as the model ID.
    For remote providers, uses ``model`` (upstream model ID).

    The provider name is taken from the ``provider`` field first (actual
    provider brand name), falling back to ``name`` (provider entry name)
    for backward compatibility.  A warning is logged when a remote provider
    entry lacks the ``provider`` field.
    """
    provider_name = provider_cfg.get("provider") or provider_cfg.get("name")
    if not provider_name:
        return None
    model_id = provider_cfg.get("llama_model") or provider_cfg.get("model")
    if not model_id:
        return None
    # Warn when a remote provider entry is missing the ``provider`` field
    if not provider_cfg.get("provider") and provider_cfg.get("type") == "remote":
        logger.warning(
            "Remote provider entry %r is missing the 'provider' field; "
            "X-Resolved-Model header will use 'name' (%r) instead of the "
            "actual provider brand name. Add 'provider: <brand>' to the "
            "provider config to fix this.",
            provider_cfg.get("name"),
            provider_name,
        )
    return f"{provider_name}/{model_id}"


def _add_resolved_model_header(response: Response, provider_cfg: dict) -> Response:
    """Add X-Resolved-Model header to a response based on provider config.

    Sets the header using ``_build_resolved_model_value()``. Overwrites
    any existing value so the fallback's resolved provider takes priority.
    """
    value = _build_resolved_model_value(provider_cfg)
    if value:
        response.headers["X-Resolved-Model"] = value
    return response


def _is_reasoning_content_roundtrip_error(response: Response) -> bool:
    """True when a response is the upstream thinking-mode reasoning_content 400.

    Remote thinking-mode providers (Console opencode.ai/zen, Console Go
    opencode.ai/zen/go, api.deepseek.com) reject multi-turn requests with this
    HTTP 400 when any assistant message lacks the ``reasoning_content`` field
    (LP-0MSGU3JNU0092AFQ). The message appears both wrapped by the Console
    gateway ("Error from provider (Console Go): Upstream request failed:
    [invalid_request_error] The `reasoning_content` ...") and verbatim from
    api.deepseek.com. Match on the invariant substring rather than the exact
    body so both variants are caught.
    """
    try:
        if int(getattr(response, "status_code", 0) or 0) != 400:
            return False
    except Exception:
        return False
    body_l = _response_body_text(response).lower()
    return (
        "reasoning_content" in body_l
        and "must be passed back" in body_l
        and "thinking mode" in body_l
    )


def _build_reasoning_content_roundtrip_error() -> Response:
    """Build the synthetic error returned instead of the raw upstream 400.

    AC3 (LP-0MSGU3JNU0092AFQ): when the reasoning_content round-trip 400 still
    occurs after the sanitizer repaired the payload (edge case) and all
    fallback providers are exhausted, the proxy must not surface the raw
    upstream body to the client. This synthetic 400 carries the cause and
    remediation guidance instead.
    """
    payload = {
        "error": {
            "type": "proxy_error",
            "code": "reasoning_content_roundtrip",
            "message": (
                "Upstream thinking-mode provider rejected the request: an "
                "assistant message in the conversation history is missing "
                "`reasoning_content`. The proxy has repaired the payload; if "
                "this error persists, the conversation history cannot be "
                "replayed to the thinking-mode provider."
            ),
            "suggested_action": (
                "Retry the request. If it persists, compact the conversation "
                "history or start a new session."
            ),
        }
    }
    return Response(
        content=json.dumps(payload).encode("utf-8"),
        status_code=400,
        media_type="application/json",
    )


def _build_exhausted_response(all_local_slot_exhaustion: bool = False, total_slots: int = 0, unavailable_providers: dict | None = None, diagnostics: list[dict[str, Any]] | None = None) -> Response:
    """Build the response when all providers are exhausted.

    Args:
        all_local_slot_exhaustion: If ``True``, all providers exhausted due to
                                   slot exhaustion (returns HTTP 429).
                                   Otherwise, returns HTTP 503 with JSON body.
        total_slots: Total number of slots across local providers (used only
                     for the slot-exhaustion 429 text body).
        unavailable_providers: Optional mapping of provider -> remaining cooldown seconds
                               to include in the 503 JSON payload for diagnostics.
        diagnostics: Optional list of per-provider attempt diagnostics (order-preserving)
    """
    if all_local_slot_exhaustion:
        # total_slots may be 0 if unknown; still format per acceptance criteria
        return Response(
            content=(f"Model server busy: 0/{int(total_slots)} slots available. Retry later.").encode(),
            status_code=429,
            media_type="text/plain",
        )

    payload = {"error": "All providers exhausted", "retry_after": 60}
    if unavailable_providers:
        # Attach diagnostic info about which providers are in cooldown
        try:
            payload["unavailable_providers"] = unavailable_providers
        except Exception:
            pass

    if diagnostics:
        try:
            # Include a sanitized diagnostics list to aid troubleshooting
            payload["diagnostics"] = diagnostics
        except Exception:
            pass

    return Response(
        content=json.dumps(payload).encode("utf-8"),
        status_code=503,
        media_type="application/json",
    )


def _build_time_window_exhausted_response(
    attempts: list[dict[str, Any]],
    unavailable: dict[str, int],
    any_provider_tried: bool,
) -> Response | None:
    """Return a distinguishable 503 when every provider was skipped solely due
    to its configured ``available_times`` window.

    The distinguishable response is used only when time windows are the *only*
    reason nothing could be used: no provider was actually tried (no errors, no
    cooldown recorded this request) and no provider is currently in cooldown.
    Otherwise ``None`` is returned and the caller falls through to the generic
    exhausted response (whose diagnostics still expose any
    ``outside_time_window`` skips).
    """
    if any_provider_tried or unavailable:
        return None
    if not any(a.get("status") == "outside_time_window" for a in attempts):
        return None

    payload = {
        "error": "All providers unavailable: no provider is available during the current scheduled time window",
        "retry_after": 60,
    }
    if attempts:
        try:
            payload["diagnostics"] = attempts
        except Exception:
            pass
    return Response(
        content=json.dumps(payload).encode("utf-8"),
        status_code=503,
        media_type="application/json",
    )


def _is_connection_error(exc: Exception) -> bool:
    """Check if an exception is a connection-related error."""
    return isinstance(exc, (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.TimeoutException,
        httpx.RemoteProtocolError,
        httpx.NetworkError,
    ))


def _is_http_error_status(status_code: int) -> bool:
    """Check if an HTTP status code indicates a provider failure."""
    return status_code >= 400


# Lazy import to avoid circular dependency
# NOTE: We do NOT cache the result so that tests can patch
# proxy.proxy_remote.proxy_to_remote between calls.


def _get_proxy_to_remote():
    """Lazily import proxy_to_remote.

    Uses ``proxy.server`` as the source so that any monkeypatches
    applied to the server module (e.g. in tests) are picked up.
    """
    from proxy.server import proxy_to_remote
    return proxy_to_remote


def _get_proxy_to_local():
    """Lazily import proxy_to_local.

    Select the best available implementation:
    - If `proxy.server.proxy_to_local` has been monkeypatched (differs from
      the router implementation), prefer that so server-level patches are used.
    - Otherwise, prefer `proxy.router.proxy_to_local` when available (so tests
      that patch the router are respected).
    - Fallback to whichever one is importable.
    """
    try:
        import proxy.router as _router
    except Exception:
        _router = None

    try:
        import proxy.server as _server
    except Exception:
        _server = None

    # If both router and server provide proxy_to_local, try to detect which
    # one has been monkeypatched by tests.  Prefer the implementation that
    # appears to come from outside the package (i.e., its __module__ is not
    # the original module name), otherwise prefer router by default.
    router_fn = getattr(_router, 'proxy_to_local', None) if _router is not None else None
    server_fn = getattr(_server, 'proxy_to_local', None) if _server is not None else None

    if router_fn is not None and server_fn is None:
        return router_fn
    if server_fn is not None and router_fn is None:
        return server_fn

    if router_fn is not None and server_fn is not None:
        router_modname = getattr(router_fn, '__module__', '')
        server_modname = getattr(server_fn, '__module__', '')
        # If router function appears to be patched (not from proxy.router), prefer it
        if not router_modname.startswith('proxy.router') and server_modname.startswith('proxy.server'):
            return router_fn
        # If server function appears to be patched, prefer it
        if not server_modname.startswith('proxy.server') and router_modname.startswith('proxy.router'):
            return server_fn
        # Fallback: prefer router implementation
        return router_fn

    raise ImportError('No proxy_to_local implementation available')


def _get_local_concurrency_info(config: dict) -> tuple:
    """Lazily import and return (current_local_active, max_local) from config.

    Returns the current local active query count and the configured
    local concurrency limit.  Uses ``session_slot_pool_size`` as the
    primary config key (same value that controls ``--parallel`` in
    llama-server). Falls back to the legacy ``local_max_concurrent_queries``
    key for backward compatibility.  Defaults to (0, 1) on error.
    """
    cur_active = 0
    max_local = 1
    try:
        import proxy.server as _srv
        cur_active = max(0, int(getattr(_srv, 'local_active_queries', 0) or 0))
    except Exception:
        pass
    try:
        server_cfg = config.get("server", config)
        # Primary: session_slot_pool_size (same as router._get_local_max_concurrent_queries)
        val = server_cfg.get("session_slot_pool_size", None)
        if val is None:
            # Fallback: local_max_concurrent_queries for backward compatibility
            val = server_cfg.get("local_max_concurrent_queries", 1)
        max_local = max(1, int(val or 1))
    except (ValueError, TypeError):
        pass
    return (cur_active, max_local)


# ---------------------------------------------------------------------------
# Bounded cross-session contention queue (LP-0MSORQVK50012Q4D)
# ---------------------------------------------------------------------------

def _contention_queue_enabled(config: dict) -> bool:
    """True when the contention queue should engage: queue policy AND cheap mode.

    The per-mode policy comes from the active mode config
    (``contention_queue_policy``; config-cheap.yaml declares ``queue``,
    config-fast.yaml declares ``fallback``). Belt-and-braces: the queue also
    requires ``proxy.mode.read_mode() == "cheap"`` so an operator override of
    LLAMA_PROXY_CONFIG cannot enable queueing in fast mode (LP-0MSORQVK50012Q4D
    constraint 5). Fail-open: any error → queue disabled (today's behavior).
    """
    try:
        from proxy.router import _get_contention_queue_config

        server_cfg = config.get("server", config) if isinstance(config, dict) else {}
        cq = _get_contention_queue_config(server_cfg)
        if cq["policy"] != "queue":
            return False
        from proxy.mode import read_mode

        return read_mode() == "cheap"
    except Exception:
        return False


async def _queue_context_bypass(
    config: dict,
    model_config: dict,
    provider_cfg: dict,
    request,
    body_json: dict,
    session_id: str | None,
) -> tuple[bool, str | None]:
    """Mirror of the smart-routing large-context skip decision.

    Used ONLY at the contention-queue decision point so context bypasses
    (``context_too_large`` / ``large_context_bypass``) are NEVER queued — they
    fall back exactly as today (LP-0MSORQVK50012Q4D AC4). Keeps the same
    thresholds/tokenizer/estimate/``_should_skip_local`` pipeline as the main
    smart-routing block below; returns ``(skip_local, skip_reason)``.
    """
    _llama_model = provider_cfg.get("llama_model", "")
    _cold_threshold, _warm_threshold = _effective_large_context_thresholds(config)
    _tokenizer, _multiplier = _get_tokenizer_for_model(model_config, config)
    _estimated_tokens = await _estimate_effective_prompt_tokens_for_routing(
        request, body_json, tokenizer=_tokenizer,
    )
    if _multiplier != 1.0:
        _estimated_tokens = int(_estimated_tokens * _multiplier)
    _skip_local = _should_skip_local(
        _llama_model, session_id, body_json, _cold_threshold,
        estimated_tokens=_estimated_tokens,
        warm_cache_threshold=_warm_threshold,
    )
    if _skip_local:
        if _warm_threshold > 0 and _estimated_tokens > _warm_threshold:
            _skip_reason = "context_too_large"
        else:
            _skip_reason = "large_context_bypass"
    else:
        _skip_reason = None
    return _skip_local, _skip_reason


def _set_queue_wait_on_request(request, elapsed: float) -> None:
    """Record the elapsed contention-queue wait on the request so
    ``proxy_to_local`` can subtract it from the adaptive timeout budget
    (Q2=a: total wait + serve stays within ``llama_adaptive_timeout_*``)."""
    try:
        setattr(request, "_contention_queue_wait_seconds", float(elapsed or 0.0))
    except Exception:
        pass


async def _maybe_queue_for_local_slot(
    config: dict,
    cur_local: int,
    max_local: int,
    request,
    body_json: dict,
    model_config: dict,
    provider_cfg: dict,
    session_id: str | None,
) -> tuple[str, str | None, float | None]:
    """Queue-wait for a local slot (cheap mode, queue policy) instead of
    immediately falling back to the next remote provider.

    Returns ``(action, reason, elapsed_seconds)`` where action is one of:

      - ``"dispatch"`` — a slot freed within the caps; the caller should
        dispatch local. *elapsed_seconds* is the queue wait (for the Q2=a
        budget subtraction).
      - ``"fallback"`` — caps exceeded; the caller falls back to the next
        remote provider exactly as today.
      - ``"context_bypass"`` — the request cannot fit a KV slot; it was
        never queued; the caller records ``cached_tokens_skip`` with the
        returned *reason*.
      - ``"fallback_policy"`` — policy is fallback / not cheap mode; the
        caller keeps today's ``local_concurrency_limit`` behavior.
    """
    if not _contention_queue_enabled(config):
        return ("fallback_policy", None, None)
    # Context bypasses never queue (AC4): a request that cannot fit the KV
    # slot must fall back exactly as today.
    skip_local, skip_reason = await _queue_context_bypass(
        config, model_config, provider_cfg, request, body_json, session_id,
    )
    if skip_local:
        return ("context_bypass", skip_reason, None)

    from proxy.router import _get_contention_queue_config

    server_cfg = config.get("server", config) if isinstance(config, dict) else {}
    cq = _get_contention_queue_config(server_cfg)

    import time as _time

    from proxy import contention_queue

    _wait_started = _time.monotonic()
    elapsed = await contention_queue.wait_for_local_slot(
        max_wait_seconds=cq["max_wait_seconds"],
        max_depth=cq["max_depth"],
        slot_free_check=lambda: _get_local_concurrency_info(config)[0] < max_local,
    )
    if elapsed is None:
        # Caps exceeded — the caller falls back. Surface the measured wait so
        # the fallback-after-queue event can record elapsed wait time (F4 AC2).
        return ("fallback", None, _time.monotonic() - _wait_started)
    return ("dispatch", None, elapsed)





def _parse_slot_exhaustion(response):
    """Parse a slot-exhaustion response and return slot info.

    Returns a dict with keys:
      - total_slots
      - available_slots
      - reason (optional)
      - local_owner_session_id (optional)

    when the response indicates slot exhaustion, otherwise returns None.

    Handles two response formats:

    1. Proxy-generated (``_build_slot_exhaustion_response``):

           {"error": {"code": "no_slots_available", ...}, "total_slots": 1}

    2. Llama-server native (flat):

           {"type": "server_busy", "code": "no_slots_available", ...}
    """
    try:
        if response.status_code != 503:
            return None
        import json
        body = json.loads(response.body)

        # Format 1: nested error.code
        error = body.get("error", {})
        if isinstance(error, dict) and error.get("code") == "no_slots_available":
            total = int(body.get("total_slots", 0) or 0)
            avail = int(body.get("available_slots", 0) or 0)
            reason = body.get("reason") or error.get("reason")
            owner = body.get("local_owner_session_id")
            return {
                "total_slots": total,
                "available_slots": avail,
                "reason": reason,
                "local_owner_session_id": owner,
            }

        # Format 2: flat top-level code (llama-server native)
        if body.get("code") == "no_slots_available":
            total = int(body.get("total_slots", 0) or 0)
            avail = int(body.get("available_slots", 0) or 0)
            reason = body.get("reason")
            owner = body.get("local_owner_session_id")
            return {
                "total_slots": total,
                "available_slots": avail,
                "reason": reason,
                "local_owner_session_id": owner,
            }
    except Exception:
        pass
    return None


def _is_slot_exhaustion_response(response) -> bool:
    """Backward-compatible boolean check for slot exhaustion."""
    return _parse_slot_exhaustion(response) is not None


def _is_local_lease_active_response(response) -> bool:
    """Return True when response indicates local lease-active contention."""
    try:
        slot_info = _parse_slot_exhaustion(response)
        if isinstance(slot_info, dict):
            reason = str(slot_info.get("reason") or "").strip().lower()
            if reason == "local_lease_active":
                return True
    except Exception:
        pass

    # Fallback heuristic in case payload shape is unexpected.
    try:
        body_text = _response_body_text(response).lower()
        return "local_lease_active" in body_text
    except Exception:
        return False


def _has_next_provider(
    model_config: dict,
    attempted_provider_names: set[str],
    excluded_domains: set[str] | None = None,
) -> bool:
    """True if at least one more provider remains untried (and not in cooldown).

    Used to gate pre-content streaming pre-flight: re-route is only worthwhile
    when the chain has a remaining provider to fall back to (LP-0MSETOTWY000SU0Z).
    Accepts already-failed failure domains so same-gateway siblings are not
    counted as a usable fallback (LP-0MSG45I8Q0020N1F).
    """
    return (
        _resolve_provider_with_exclusions(
            model_config,
            attempted_provider_names,
            excluded_domains,
        )
        is not None
    )


def _failure_domain_key(provider_cfg: dict) -> str:
    """Return a canonical failure-domain key for a provider entry.

    Remote entries key on the normalized ``endpoint`` URL: scheme and host
    lowercased, default ports dropped, trailing slash and fragment stripped,
    path case and query strings preserved. Local / no-endpoint entries fall
    back to the ``provider`` brand, then to the entry name (last resort so
    entries without either never share a key).

    Entries that share a failure-domain key are treated as ONE failure domain:
    a stall/terminal error on one entry excludes the whole domain from the
    fallback chain / mid-stream re-route (LP-0MSG45I8Q0020N1F).
    """
    endpoint = provider_cfg.get("endpoint")
    if endpoint:
        normalized = _normalize_endpoint_for_failure_domain(str(endpoint))
        if normalized:
            return normalized
    brand = provider_cfg.get("provider")
    if brand:
        return str(brand)
    return str(provider_cfg.get("name") or "unknown")


def _normalize_endpoint_for_failure_domain(endpoint: str) -> str | None:
    """Normalize an endpoint URL into a canonical failure-domain key.

    Lowercases scheme+host, drops default ports (80/443), strips a trailing
    slash and any fragment, preserves path case and query strings. Returns
    ``None`` when the URL cannot be parsed.
    """
    try:
        parsed = urlsplit(endpoint)
    except Exception:
        return None
    if not parsed.scheme or not parsed.hostname:
        return None
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    default_port = {"http": 80, "https": 443}.get(scheme)
    port = parsed.port
    if port and port == default_port:
        netloc = host
    elif port:
        netloc = f"{host}:{port}"
    else:
        netloc = host
    path = parsed.path.rstrip("/")
    # Keep query strings; drop the fragment.
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def _usage_limit_account_key(provider_cfg: dict) -> str:
    """Return a canonical key identifying the upstream ACCOUNT for usage-limit
    quarantine (LP-0MSMBWB23009XYPW).

    Usage limits are per-account (per API key), NOT per endpoint: entries that
    share a gateway but use different ``api_key_env`` values (e.g.
    ``opencode-go`` and ``opencode-go-2`` on https://opencode.ai/zen/go) have
    independent limits and must be quarantined independently. The key combines
    ``api_key_env`` with the normalized endpoint so distinct accounts on the
    same gateway stay separate. Entries without an ``api_key_env`` cannot be
    distinguished by account and fall back to the failure-domain key.
    """
    api_key_env = provider_cfg.get("api_key_env")
    if api_key_env:
        return f"{api_key_env}@{_failure_domain_key(provider_cfg)}"
    return _failure_domain_key(provider_cfg)


def _resolve_provider_with_exclusions(
    model_config: dict,
    excluded_provider_names: set[str],
    excluded_domains: set[str] | None = None,
) -> dict | None:
    """Resolve next available provider while excluding names tried this request
    and any entry sharing an already-failed failure domain.

    ``excluded_domains`` holds failure-domain keys (see ``_failure_domain_key``)
    that already stalled / terminally failed during THIS request, so the chain
    skips straight past same-gateway API-key siblings (LP-0MSG45I8Q0020N1F).
    Usage-limit account keys (see ``_usage_limit_account_key``) are checked
    against the same set so an account exhausted by a usage-limit error is not
    retried via another entry sharing that account (LP-0MSMBWB23009XYPW).
    """
    providers: list[dict[str, Any]] | None = model_config.get("providers")
    if not providers:
        return None

    excluded_domains = excluded_domains or set()

    for provider_cfg in providers:
        name = provider_cfg.get("name", "")
        if name in excluded_provider_names:
            continue
        domain = _failure_domain_key(provider_cfg)
        if domain in excluded_domains:
            logger.info(
                "Skipping provider=%s: same failure domain as an already-failed "
                "entry (domain=%s)",
                name,
                domain,
            )
            continue
        # Usage-limit reset pending (LP-0MSLJPOCC0001ROJ): the ACCOUNT hit an
        # upstream usage-limit error (GoUsageLimitError etc.) and is
        # quarantined until the computed reset time + margin passes. Keyed on
        # the API-key account, not the endpoint: distinct api_key_env entries
        # on the same gateway have independent limits (LP-0MSMBWB23009XYPW).
        usage_key = _usage_limit_account_key(provider_cfg)
        if usage_key in excluded_domains:
            logger.info(
                "Skipping provider=%s: same usage-limit account as an "
                "already-exhausted entry (account=%s)",
                name,
                usage_key,
            )
            continue
        reset_remaining = _usage_reset_remaining(usage_key)
        if reset_remaining > 0:
            logger.info(
                "Skipping provider=%s: usage_limit_reset_pending "
                "(account=%s, reset_at=%s, reset_in=%ds)",
                name,
                usage_key,
                datetime.fromtimestamp(_usage_reset_at[usage_key], tz=UTC).isoformat(),
                int(reset_remaining),
            )
            continue
        cooldown_key = _entry_cooldown_key(provider_cfg)
        if cooldown_key is not None:
            remaining = _provider_cooldown_remaining(cooldown_key)
            logger.info(
                "Skipping provider=%s: %s in cooldown (%ds remaining)",
                name,
                cooldown_key,
                remaining,
            )
            continue
        if not _is_within_allowed_window(provider_cfg):
            logger.info(
                "Skipping provider=%s: outside its available_times window (UTC)",
                name,
            )
            continue
        return provider_cfg
    return None


def _is_model_loading_response(response: Response, body_text: str) -> bool:
    """Return True when a 503 response represents transient model loading."""
    if int(getattr(response, "status_code", 0) or 0) != 503:
        return False

    try:
        payload = json.loads(body_text) if body_text else None
    except Exception:
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            code = str(error.get("code", "")).strip().lower()
            err_type = str(error.get("type", "")).strip().lower()
            message = str(error.get("message", "")).strip().lower()
            if code == "model_loading" or err_type == "model_loading":
                return True
            if "model" in message and "loading" in message:
                return True
        elif isinstance(error, str):
            lowered = error.strip().lower()
            if "model_loading" in lowered or ("model" in lowered and "loading" in lowered):
                return True

    lowered_body = (body_text or "").strip().lower()
    if "model_loading" in lowered_body:
        return True
    if "model" in lowered_body and "loading" in lowered_body:
        return True

    return False


def _is_free_usage_limit_error(response: Response, body_text: str) -> bool:
    """Return True when a 429 response is a FreeUsageLimitError.

    Detects upstream quota-exhaustion responses where the JSON body contains
    ``error.type = "FreeUsageLimitError"`` (case-insensitive).  When detected,
    the proxy applies a 3-hour cooldown on the affected provider entry so the
    fallback chain routes to paid alternatives instead of repeatedly retrying
    the exhausted free tier.

    Expected upstream format (observed from opencode.ai/zen):
        HTTP 429
        Body: {"type": "error", "error": {"type": "FreeUsageLimitError", ...}}
    """
    status = int(getattr(response, "status_code", 0) or 0)
    if status != 429:
        return False

    try:
        payload = json.loads(body_text) if body_text else None
    except Exception:
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            err_type = str(error.get("type", "")).strip().lower()
            if err_type == "freeusagelimiterror":
                return True

    # Fallback: check raw body text for the error type string
    lowered_body = (body_text or "").strip().lower()
    if "freeusagelimiterror" in lowered_body:
        return True

    return False


def _parse_resets_in(message: str) -> float | None:
    """Parse a ``Resets in ...`` duration from an upstream error message.

    Handles the observed opencode format (``Resets in 22hr 43min.``) plus
    plural/full-word variants: hours, minutes, seconds, and days (used by
    monthly limits). Returns the duration in seconds, or ``None`` when the
    message carries no parseable reset duration.
    """
    match = re.search(r"resets?\s+in\s+(.+?)(?:\.|$)", message or "", re.IGNORECASE)
    if not match:
        return None
    total = 0.0
    found = False
    for num_str, unit in re.findall(
        r"(\d+(?:\.\d+)?)\s*(days?|d|hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)",
        match.group(1),
        re.IGNORECASE,
    ):
        value = float(num_str)
        u = unit.lower()
        if u in ("d", "day", "days"):
            total += value * 86400
        elif u in ("h", "hr", "hrs", "hour", "hours"):
            total += value * 3600
        elif u in ("m", "min", "mins", "minute", "minutes"):
            total += value * 60
        elif u in ("s", "sec", "secs", "second", "seconds"):
            total += value
        found = True
    return total if found else None


def _usage_limit_reset_seconds(response: Response, body_text: str) -> float | None:
    """Return seconds until the usage limit resets for a 429 usage-limit error.

    Recognizes ``GoUsageLimitError`` (LP-0MSLJPOCC0001ROJ) and any usage-limit
    error variant that carries a reset duration in its message (including
    ``FreeUsageLimitError`` responses that include one). The reset duration is
    parsed from the provider message (``Resets in 22hr 43min``); when the
    message has no explicit duration, ``metadata.limitName``
    (daily/weekly/monthly) supplies the period. The 2-minute safety margin is
    added to the returned duration.

    Returns ``None`` when the response is not a 429 usage-limit error or no
    reset duration can be computed — callers then fall back to the existing
    ``FreeUsageLimitError`` 3-hour cooldown / generic rate-limit handling.

    Expected upstream format (observed from opencode.ai/zen):
        HTTP 429
        Body: {"type": "error", "error": {"type": "GoUsageLimitError",
              "message": "Weekly usage limit reached. Resets in 22hr 43min."},
              "metadata": {"limitName": "weekly"}}
    """
    status = int(getattr(response, "status_code", 0) or 0)
    if status != 429:
        return None

    try:
        payload = json.loads(body_text) if body_text else None
    except Exception:
        payload = None

    err_type = None
    message = None
    limit_name = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            err_type = str(error.get("type", "")).strip().lower()
            message = error.get("message")
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            limit_name = str(metadata.get("limitName", "")).strip().lower()

    if err_type not in ("gousagelimiterror", "freeusagelimiterror"):
        return None

    seconds = _parse_resets_in(message) if message else None
    if seconds is None and limit_name in _PERIOD_DEFAULT_SECONDS:
        seconds = _PERIOD_DEFAULT_SECONDS[limit_name]
    if seconds is None:
        return None
    return float(seconds) + _USAGE_LIMIT_RESET_MARGIN_SECONDS


def _usage_reset_remaining(failure_domain: str) -> float:
    """Return the remaining seconds before a usage-limit reset for an account
    / failure-domain key (see ``_usage_limit_account_key``).

    Returns 0.0 when the domain has no pending usage reset or its reset time
    has passed (expired entries are cleaned up lazily, mirroring the cooldown
    dict behaviour).
    """
    expiry = _usage_reset_at.get(failure_domain)
    if expiry is None:
        return 0.0
    remaining = expiry - time.time()
    if remaining <= 0:
        del _usage_reset_at[failure_domain]
        return 0.0
    return remaining


# ---------------------------------------------------------------------------
# Shared fallback primitives (extracted from proxy_with_remote_fallback and
# proxy_with_fallback to eliminate duplicated state-machine logic)
# ---------------------------------------------------------------------------


def _record_attempt(attempts: list[dict[str, Any]], **fields) -> None:
    """Append a diagnostic attempt entry to the attempts list.

    Each entry records which provider was tried, the outcome, and optional
    diagnostic payload (status code, body snippet, cooldown, etc.).
    """
    attempts.append(dict(fields))


class StreamingPreContentError(Exception):
    """Raised when a remote streaming response fails before delivering any
    content-bearing chunk (stall-exhausted, empty response, or stream error).

    The fallback chain catches this, marks the provider unavailable (Tier-2
    cooldown), and routes to the next provider in the chain
    (LP-0MSETOTWY000SU0Z / proxy/docs/error-analysis-2026-08-03.md
    Recommendation 1).
    """

    def __init__(self, provider_name: str, reason: str):
        self.provider_name = provider_name
        self.reason = reason
        super().__init__(f"pre-content stream failure for {provider_name}: {reason}")


class StreamingRecoverableAfterReasoningError(Exception):
    """Raised when a remote streaming response stalls AFTER delivering
    reasoning_content but BEFORE any final-answer content (and with zero
    tool_calls) was delivered.

    This is the mid-stream re-route signal (LP-0MSF1PUM90099ZSW): the client
    has not committed to a final answer (no final content chunk forwarded, no
    tool-result round-trip), so the fallback chain can re-route the SAME
    request to the next provider instead of surfacing an error that tells the
    client to retry.

    Tool-call-only stalls do NOT use this exception: once tool_calls are
    delivered, re-routing would make the model re-plan the request, so they
    terminate via the existing enriched-error path (operator decision Q1).
    """

    def __init__(self, provider_name: str, reason: str):
        self.provider_name = provider_name
        self.reason = reason
        super().__init__(f"mid-stream stall after reasoning for {provider_name}: {reason}")


class ChainExhaustedError(Exception):
    """Raised when a fallback chain CYCLE exhausts every provider.

    This is the distinguishable exhaustion signal at the chain-hold wrapper
    boundary (LP-0MSH94Z7K007VKC9): the fallback cycle functions raise it from
    their exhaustion tail instead of returning an error response, so the
    cycle-hold wrapper can decide whether to hold + restart a new cycle from
    the first provider, or return the carried response verbatim once the hold
    bound is reached. Genuine mid-stream failures NEVER raise this — they
    re-route inside the cycle or return a normal (non-exhaustion) response.

    Attributes:
        response: The response the exhausted chain would have returned
            (503 "All providers exhausted", first-provider error, time-window
            exhaustion, etc.).
    """

    def __init__(self, response: Response):
        self.response = response
        super().__init__(f"provider chain exhausted: status={getattr(response, 'status_code', '?')}")


def _classify_stream_chunk(chunk: bytes) -> tuple[bool, bool, bool, bool, bool]:
    """Classify one raw SSE chunk for pre-content recovery pre-flight.

    Returns ``(has_final_content, has_tool_calls, has_reasoning,
    is_terminal_error, is_done)`` where:
    - ``has_final_content``: the chunk carries a non-empty final-answer
      ``content`` delta — the commit point for mid-stream re-route
      (LP-0MSF1PUM90099ZSW).
    - ``has_tool_calls``: the chunk carries a non-empty ``tool_calls`` delta
      (intermediate; terminate-eligible).
    - ``has_reasoning``: the chunk carries a non-empty ``reasoning_content``
      delta (intermediate; re-route-eligible).
    - ``is_terminal_error``: a choice carries ``finish_reason: "error"``.
    - ``is_done``: a ``[DONE]`` marker or ``finish_reason: "stop"``.

    Keep-alive comments (``: keep-alive``) and non-``data:`` lines are
    ignored, so a stalled-but-alive stream yields all-False until either
    content or a terminal event arrives.
    """
    has_final_content = False
    has_tool_calls = False
    has_reasoning = False
    is_terminal_error = False
    is_done = False
    try:
        text = chunk.decode("utf-8", errors="replace")
    except Exception:
        return False, False, False, False, False
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            is_done = True
            continue
        try:
            j = json.loads(payload)
        except Exception:
            continue
        for choice in j.get("choices", []):
            if not isinstance(choice, dict):
                continue
            fr = choice.get("finish_reason")
            if fr == "error":
                is_terminal_error = True
            elif fr == "stop":
                is_done = True
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            c = delta.get("content")
            if isinstance(c, str) and c.strip():
                has_final_content = True
            tc = delta.get("tool_calls")
            if isinstance(tc, list) and tc:
                has_tool_calls = True
            rc = delta.get("reasoning_content")
            if isinstance(rc, str) and rc.strip():
                has_reasoning = True
    return has_final_content, has_tool_calls, has_reasoning, is_terminal_error, is_done


async def _preflight_streaming_response(
    response: Response,
    request,
    provider_name: str,
) -> Response:
    """Pre-consume a 2xx streaming response up to the commit point or a
    terminal event, so a mid-stream failure can re-route.

    Reads the response body iterator chunk-by-chunk, buffering intermediate
    chunks (reasoning_content / tool_calls) WITHOUT forwarding them, until one
    of:
    - the commit point: a final-answer ``content`` chunk arrives → returns a
      wrapped response that replays the buffered chunks in order then
      continues the original iterator (LP-0MSF1PUM90099ZSW commit point);
    - a terminal ``finish_reason: error`` event arrives with only
      reasoning_content delivered (zero tool_calls, zero final content) →
      raises :class:`StreamingRecoverableAfterReasoningError` so the fallback
      chain re-routes to the next provider (buffer discarded);
    - a terminal ``finish_reason: error`` event arrives with zero content of
      any kind → raises :class:`StreamingPreContentError` (existing behavior,
      LP-0MSETOTWY000SU0Z);
    - a terminal ``finish_reason: error`` event arrives with tool_calls
      delivered → returns the wrapped response as-is (replay buffer +
      continue): the enriched error event reaches the client and the chain
      does NOT re-route (operator decision Q1, tool-call-only stalls
      terminate);
    - the stream ends with zero content (empty response) → raises
      :class:`StreamingPreContentError`;
    - a stream exception occurs with zero content → raises
      :class:`StreamingPreContentError`.

    ``[DONE]`` / ``finish_reason: stop`` markers are consumed but do NOT
    terminate the pre-flight: the upstream generator retries empty responses
    internally (LP-0MRF77A0E0026B9T) and only yields a terminal error event
    once those retries are exhausted, so the pre-flight keeps consuming until
    content or that terminal error.

    Args:
        response: The candidate response from the remote provider.
        request: The client request (for disconnect checks).
        provider_name: Config entry name, used for the error signal.

    Returns:
        The (possibly wrapped) response to stream to the client.
    """
    if not _is_streaming_response(response) or int(getattr(response, "status_code", 0) or 0) >= 400:
        return response

    original = response.body_iterator
    buffered: list[bytes] = []
    saw_final_content = False
    saw_tool_calls = False
    saw_reasoning = False
    disconnected = False
    try:
        while True:
            try:
                if await request.is_disconnected():
                    disconnected = True
                    break
            except Exception:
                pass
            try:
                chunk = await original.__anext__()
            except StopAsyncIteration:
                break
            except Exception as exc:
                if not (saw_final_content or saw_tool_calls or saw_reasoning):
                    raise StreamingPreContentError(
                        provider_name,
                        f"stream_exception:{type(exc).__name__}",
                    ) from exc
                break
            buffered.append(chunk)
            (
                has_final_content,
                has_tool_calls,
                has_reasoning,
                is_terminal_error,
                _is_done,
            ) = _classify_stream_chunk(chunk)
            # Accumulate what has been delivered so far (buffered intermediate
            # chunks count toward the stall classification decision).
            saw_final_content = saw_final_content or has_final_content
            saw_tool_calls = saw_tool_calls or has_tool_calls
            saw_reasoning = saw_reasoning or has_reasoning
            if has_final_content:
                saw_final_content = True
                break
            if is_terminal_error:
                if saw_tool_calls:
                    # Tool-call-only stall: terminate via the enriched error
                    # (replay buffer + continue; the error event is in the
                    # stream). No re-route (operator decision Q1).
                    break
                if saw_reasoning:
                    raise StreamingRecoverableAfterReasoningError(
                        provider_name,
                        "stall_after_reasoning",
                    )
                raise StreamingPreContentError(provider_name, "finish_reason:error")
            # [DONE] / finish_reason: stop with zero content is NOT terminal
            # here — the generator retries empty responses internally and only
            # yields a terminal error event once those retries are exhausted
            # (LP-0MRF77A0E0026B9T). Keep consuming.
    except GeneratorExit:
        raise

    if disconnected:
        # Client is gone — hand the (partially consumed) response back as-is;
        # re-routing to another provider would be wasteful.
        return response

    if not (saw_final_content or saw_tool_calls or saw_reasoning):
        raise StreamingPreContentError(provider_name, "empty_response")

    async def _replay_and_continue():
        for c in buffered:
            yield c
        async for c in original:
            yield c

    return StreamingResponse(
        _replay_and_continue(),
        media_type=response.media_type,
        headers=dict(response.headers),
        status_code=response.status_code,
    )


def _handle_streaming_success(
    response: Response,
    provider_name: str,
    provider_type: str,
    attempts: list[dict[str, Any]],
    prev_provider: str | None,
    fallback_reason: str | None,
    path: str,
) -> Response | None:
    """If *response* is a 2xx StreamingResponse, record the attempt, add
    the ``X-Provider`` header, log the fallback (if one occurred), and
    return the augmented response.

    Returns ``None`` if *response* is **not** a streaming success (caller
    should continue normal processing).
    """
    if _is_streaming_response(response) and int(getattr(response, "status_code", 0) or 0) < 400:
        _record_attempt(
            attempts,
            provider=provider_name,
            type=provider_type,
            status="streaming_success",
            status_code=int(getattr(response, "status_code", 0) or 0),
        )
        # Reset exponential-backoff failure count on success
        _reset_provider_failure_count(provider_name)
        result = _add_provider_header(response, provider_name)
        if prev_provider:
            logger.info(
                "Fallback triggered for model=%s, from=%s, to=%s, reason=%s",
                path, prev_provider, provider_name, fallback_reason or "streaming",
            )
        return result
    return None


def _prepend_sse_comment(response: Response, comment: str) -> Response:
    """Return a StreamingResponse whose body emits *comment* (an SSE comment
    line, e.g. ``: re-route provider=a->b reason=stall_after_reasoning``)
    before the original body bytes.

    If *response* is not a streaming response, the comment is dropped (the
    client gets the JSON/other body as-is) — the marker is best-effort.
    """
    if not _is_streaming_response(response):
        return response
    original = response.body_iterator
    comment_bytes = (comment if comment.endswith("\n\n") else comment + "\n\n").encode()

    async def _with_comment():
        yield comment_bytes
        async for c in original:
            yield c

    return StreamingResponse(
        _with_comment(),
        media_type=response.media_type,
        headers=dict(response.headers),
        status_code=response.status_code,
    )


def _build_fallback_success_response(
    response: Response,
    provider_name: str,
    provider_type: str,
    attempts: list[dict[str, Any]],
    prev_provider: str | None,
    fallback_reason: str | None,
    path: str,
    body_text: str = "",
    status_override: str = "success",
) -> Response:
    """Record a successful provider attempt, add the ``X-Provider`` header,
    log the fallback (if one occurred), and return the augmented response.

    This is the normal (non-streaming) success path used by both fallback
    entrypoints when a provider returns a successful response.
    """
    _record_attempt(
        attempts,
        provider=provider_name,
        type=provider_type,
        status=status_override,
        status_code=int(getattr(response, "status_code", 0) or 0),
        body_snippet=(body_text[:512] if body_text else None),
    )
    # Reset exponential-backoff failure count on success
    _reset_provider_failure_count(provider_name)
    result = _add_provider_header(response, provider_name)
    if prev_provider:
        logger.info(
            "Fallback triggered for model=%s, from=%s, to=%s, reason=%s",
            path,
            prev_provider,
            provider_name,
            fallback_reason or "unknown",
        )
    return result


def _handle_connection_error_in_fallback(
    exc: Exception,
    provider_name: str,
    provider_type: str,
    cooldown_seconds: float,
    attempts: list[dict[str, Any]],
) -> bool:
    """If *exc* is a connection error, mark the provider unavailable, record
    a diagnostic attempt entry, and return ``True`` (caller should ``continue``
    to the next provider).

    Applies exponential backoff for remote providers (capped at configured
    *cooldown_seconds*, so setting it to 0 disables backoff entirely).

    Returns ``False`` if *exc* is **not** a connection error (caller should
    re-raise or handle differently).
    """
    if _is_connection_error(exc):
        cooldown = cooldown_seconds
        if provider_type == "remote" and cooldown_seconds > 0:
            count = _provider_failure_count.get(provider_name, 0)
            backoff = min(
                _BACKOFF_BASE_SECONDS * (2 ** count),
                _BACKOFF_MAX_SECONDS,
            )
            cooldown = min(backoff, cooldown_seconds)
            _provider_failure_count[provider_name] = count + 1
        mark_provider_unavailable(provider_name, cooldown)
        _record_attempt(
            attempts,
            provider=provider_name,
            type=provider_type,
            status="connection_error",
            error=str(type(exc).__name__),
        )
        return True
    return False


def _handle_http_error_with_cooldown(
    response: Response,
    provider_name: str,
    provider_type: str,
    cooldown_seconds: float,
    attempts: list[dict[str, Any]],
    body_text: str,
) -> float:
    """Handle an HTTP error response: compute effective cooldown, mark the
    provider unavailable, record a diagnostic attempt entry, and return the
    effective cooldown duration.

    Applies exponential backoff for remote providers.

    The caller is responsible for setting ``fallback_reason``, ``prev_provider``,
    and ``all_slot_exhaustion`` after calling this function, and for issuing
    ``continue``.
    """
    # Parse Retry-After separately so we can respect it alongside backoff
    retry_after = _parse_retry_after(response)

    cooldown = cooldown_seconds
    if provider_type == "remote" and cooldown_seconds > 0:
        count = _provider_failure_count.get(provider_name, 0)
        backoff = min(
            _BACKOFF_BASE_SECONDS * (2 ** count),
            _BACKOFF_MAX_SECONDS,
        )
        cooldown = min(backoff, cooldown_seconds)
        _provider_failure_count[provider_name] = count + 1

    # Respect Retry-After header regardless of backoff
    if retry_after is not None:
        cooldown = max(cooldown, retry_after)

    mark_provider_unavailable(provider_name, cooldown)
    _record_attempt(
        attempts,
        provider=provider_name,
        type=provider_type,
        status="http_error",
        status_code=int(response.status_code),
        body_snippet=(body_text[:512] if body_text else None),
        cooldown_seconds=cooldown,
    )
    return cooldown


def _observe_http_error_400(
    response: Response,
    provider_name: str,
    provider_type: str,
    path: str,
    body_text: str,
    fallback_reason: str,
) -> None:
    """Observability for remote HTTP 400 rejections (LP-0MSC1BNP90017L9K).

    Emits a per-fallback INFO log line containing the response body snippet
    (first 512 chars) and increments ``proxy_http_errors_total{status=400}``
    with the fallback reason, so rejection causes are discoverable without
    code changes. Best-effort: never raises.
    """
    try:
        if int(response.status_code) != 400 or provider_type != "remote":
            return
        snippet = (body_text or "")[:512]
        logger.info(
            "Remote HTTP 400 from provider=%s model=%s reason=%s body_snippet=%s",
            provider_name,
            path,
            fallback_reason,
            snippet,
        )
        try:
            from proxy.metrics import record_http_error

            record_http_error(path, "400", fallback_reason)
        except Exception:
            pass
    except Exception:
        pass


def _handle_empty_response_with_cooldown(
    response: Response,
    provider_name: str,
    provider_type: str,
    cooldown_seconds: float,
    attempts: list[dict[str, Any]],
    body_text: str,
) -> float:
    """Handle an empty (non-reasoning) successful response: compute effective
    cooldown, mark the provider unavailable, record a diagnostic attempt entry,
    and return the effective cooldown duration.

    Applies exponential backoff for remote providers.

    The caller is responsible for setting ``fallback_reason``, ``prev_provider``,
    and ``all_slot_exhaustion`` after calling this function, and for issuing
    ``continue``.
    """
    # Parse Retry-After separately so we can respect it alongside backoff
    retry_after = _parse_retry_after(response)

    cooldown = cooldown_seconds
    if provider_type == "remote" and cooldown_seconds > 0:
        count = _provider_failure_count.get(provider_name, 0)
        backoff = min(
            _BACKOFF_BASE_SECONDS * (2 ** count),
            _BACKOFF_MAX_SECONDS,
        )
        cooldown = min(backoff, cooldown_seconds)
        _provider_failure_count[provider_name] = count + 1

    # Respect Retry-After header regardless of backoff
    if retry_after is not None:
        cooldown = max(cooldown, retry_after)

    mark_provider_unavailable(provider_name, cooldown)
    _record_attempt(
        attempts,
        provider=provider_name,
        type=provider_type,
        status="empty_response",
        status_code=int(getattr(response, "status_code", 0) or 0),
        body_snippet=(body_text[:512] if body_text else None),
        cooldown_seconds=cooldown,
    )
    return cooldown


def _resolve_reasoning_content_promotion(
    response: Response,
    provider_name: str,
    provider_type: str,
    attempts: list[dict[str, Any]],
    prev_provider: str | None,
    fallback_reason: str | None,
    path: str,
    body_text: str,
) -> Response | None:
    """If the response body contains ``reasoning_content``, treat this
    empty-but-meaningful response as a success (promote it).  Records the
    attempt, adds the provider header, logs the fallback, and returns the
    augmented response.

    Returns ``None`` if the body does **not** contain ``reasoning_content``
    (caller should continue with empty-response cooldown logic).

    Consistency note (LP-0MSEHOE7B005DE08): since the placeholder change,
    thinking-only responses (``reasoning_content`` present, no tool call)
    are extracted as the non-empty placeholder ``"Thinking..."`` by
    ``_extract_assistant_content``, so ``_is_empty_response`` returns False
    and the fallback chain treats them as plain successes without reaching
    this function. This function remains the safety net for bodies whose raw
    text contains the ``reasoning_content`` key but where extraction found
    nothing usable (e.g. an empty ``reasoning_content`` value) - those still
    count as promoted successes rather than triggering cooldown/fallback.
    """
    body_l = (body_text or "").lower()
    if "reasoning_content" in body_l:
        _record_attempt(
            attempts,
            provider=provider_name,
            type=provider_type,
            status="promoted_reasoning",
            status_code=int(getattr(response, "status_code", 0) or 0),
            body_snippet=(body_text[:512] if body_text else None),
        )
        # Reset exponential-backoff failure count on success
        _reset_provider_failure_count(provider_name)
        result = _add_provider_header(response, provider_name)
        if prev_provider:
            logger.info(
                "Fallback triggered for model=%s, from=%s, to=%s, reason=%s",
                path,
                prev_provider,
                provider_name,
                fallback_reason or "promoted_reasoning",
            )
        return result
    return None


def _log_exhausted_providers(model_config: dict, path: str) -> dict[str, int]:
    """Log diagnostic details about which providers are in cooldown and return
    the mapping of provider key to remaining cooldown seconds.

    Includes BOTH entry-name and provider-brand cooldown keys so a Tier-3
    stall circuit breaker trip (which marks the brand unavailable) is visible
    when all providers are exhausted (LP-0MSG45LOO007K236).
    """
    unavailable: dict[str, int] = {}
    try:
        providers = model_config.get("providers", []) or []
        for p in providers:
            if not isinstance(p, dict):
                continue
            for key in (p.get("name"), p.get("provider")):
                if not key:
                    continue
                remaining = _provider_cooldown_remaining(key)
                if remaining > 0:
                    unavailable[key] = remaining
        logger.warning("All providers exhausted for model=%s; unavailable=%s", path, unavailable)
    except Exception:
        pass
    return unavailable


# ---------------------------------------------------------------------------
# Chain-hold retry (LP-0MSH94Z7K007VKC9)
#
# When a fallback chain CYCLE exhausts every provider, the request is held for
# ``server.chain_hold_seconds`` (default 300) — giving short cooldowns
# (provider cooldown, stall circuit breaker, time-window edges) time to
# expire — then a NEW cycle starts from the FIRST provider with fresh
# per-request state. The number of hold-retry cycles is bounded by
# ``server.chain_hold_max_cycles`` (default 3; 0 = infinite).
#
# The hold only defers the exhaustion verdict: successful responses, provider
# ordering, and existing cooldown/circuit-breaker behavior are unchanged.
# Streaming requests receive periodic SSE comment lines
# (``: chain exhausted (...); retrying from <first> in <Ns>``) so the client
# can surface live progress; non-streaming requests are held silently
# (deferred). A client disconnect aborts the hold promptly.
#
# The feature is enabled when either config key is present (flat or
# ``server.*``); production config.yaml ships both (300s / 3 cycles).
# ---------------------------------------------------------------------------


def _first_provider_name(model_config: dict) -> str:
    """Return the name of the first provider in the chain.

    Used for the ``retrying from <first-model>`` hold-feedback comment: a new
    cycle always restarts from the first provider.
    """
    providers = model_config.get("providers") or []
    for p in providers:
        if isinstance(p, dict):
            return str(p.get("name") or p.get("provider") or "unknown")
    return "unknown"


def _exhaustion_diagnostics(response: Response) -> str:
    """Extract a compact provider-diagnostics string from an exhaustion response.

    Reads the ``error``, ``unavailable_providers`` and ``diagnostics`` fields
    of the exhausted 503 payload (built by ``_build_exhausted_response``) so
    the hold-feedback comment can tell the client which providers were
    unavailable and why. Returns "" when the body is not a JSON payload.
    """
    try:
        body_text = _response_body_text(response)
        payload = json.loads(body_text) if body_text else None
        if isinstance(payload, dict):
            bits: list[str] = []
            err = payload.get("error")
            if isinstance(err, str) and err:
                bits.append(f"error={err}")
            unavail = payload.get("unavailable_providers")
            if isinstance(unavail, dict) and unavail:
                bits.append(f"unavailable={unavail}")
            diag = payload.get("diagnostics")
            if isinstance(diag, list) and diag:
                bits.append(f"attempts={len(diag)}")
            if bits:
                return ", ".join(bits)[:512]
    except Exception:
        pass
    return ""


def _build_chain_hold_comment(first_provider: str, hold_seconds: float, diagnostics: str) -> str:
    """Build the SSE hold-feedback comment line.

    Mirrors the existing ``: re-route provider=a->b reason=...`` comment
    format (LP-0MSF1PUM90099ZSW)::

        : chain exhausted (<provider diagnostics>); retrying from <first> in <Ns>

    Clients may surface these comment lines as live progress while the
    request is held (companion item SA-0MSHAKSEA001LQ6T).
    """
    return (
        f": chain exhausted ({diagnostics}); retrying from {first_provider} "
        f"in {int(hold_seconds)}s"
    )


async def _request_is_streaming(request) -> bool:
    """Return True when the request asks for an SSE stream (``stream: true``).

    Best-effort: reads the (Starlette-cached) request body; defaults to False
    when the body cannot be read or parsed.
    """
    try:
        raw = await request.body()
        payload = json.loads(raw) if raw else {}
        return bool(payload.get("stream", False))
    except Exception:
        return False


async def _hold_sleep(request, seconds: float) -> bool:
    """Silent (non-streaming) hold: sleep *seconds* with periodic disconnect
    checks.

    Returns True when the client disconnected during the hold (caller should
    abort the hold, AC4), False when the full hold elapsed.
    """
    if seconds <= 0:
        return False
    interval = min(1.0, max(0.05, seconds / 20))
    remaining = seconds
    while remaining > 0:
        try:
            if await request.is_disconnected():
                return True
        except Exception:
            pass
        slice_ = min(interval, remaining)
        await asyncio.sleep(slice_)
        remaining -= slice_
    return False


async def _hold_feedback(request, hold_seconds: float, comment: str):
    """Async generator: emit *comment* as an SSE line, periodically, while
    sleeping *hold_seconds*.

    Aborts (stops yielding) as soon as the client disconnects (AC4). A
    zero-length hold still emits one comment so the client sees the hold
    start.
    """
    comment_bytes = (comment + "\n\n").encode("utf-8")
    if hold_seconds <= 0:
        yield comment_bytes
        return
    interval = min(30.0, max(0.1, hold_seconds / 10))
    remaining = hold_seconds
    while remaining > 0:
        try:
            if await request.is_disconnected():
                return
        except Exception:
            pass
        yield comment_bytes
        slice_ = min(interval, remaining)
        await asyncio.sleep(slice_)
        remaining -= slice_


def _response_body_bytes(response: Response) -> bytes:
    """Best-effort extraction of the raw body bytes of a plain Response."""
    try:
        if hasattr(response, "body"):
            b = response.body
            return b if isinstance(b, bytes) else str(b).encode("utf-8")
        if hasattr(response, "content"):
            b = response.content
            return b if isinstance(b, bytes) else str(b).encode("utf-8")
    except Exception:
        pass
    return b""


def _response_to_sse_bytes(response: Response) -> bytes:
    """Serialize a plain Response body as one SSE ``data:`` chunk.

    Used to deliver the terminal exhaustion/error response inside a streaming
    hold, where the stream already started with 200 + hold comments.
    """
    body = _response_body_bytes(response)
    if not body:
        return b"data: {}\n\n"
    if body.lstrip().startswith(b"data:"):
        return body
    return b"data: " + body + b"\n\n"


def _log_chain_hold_start(path: str, hold_seconds: float, cycle: int, max_cycles: int) -> None:
    """Log the start of a chain hold (operator-visible, mirrors the existing
    re-route / exhausted log lines)."""
    logger.warning(
        "Chain exhausted for model=%s; holding %.0fs before restarting cycle "
        "from the first provider (cycle=%d, max_cycles=%s)",
        path,
        hold_seconds,
        cycle,
        max_cycles if max_cycles else "infinite",
    )


def _build_streaming_hold_response(
    request,
    path: str,
    model_config: dict,
    config: dict,
    cycle_fn,
    next_cycle: int,
    max_cycles: int,
    hold_seconds: float,
    exhaustion_response: Response,
) -> StreamingResponse:
    """Build the streaming response that delivers the chain hold (AC3/AC4).

    The generator:
    1. emits ``: chain exhausted (...)`` SSE comment lines periodically while
       sleeping *hold_seconds* (disconnect-aware — AC4);
    2. runs the next chain cycle from the FIRST provider;
    3. on success, replays the cycle's response body after the comments;
    4. on exhaustion, repeats from step 1 until the cycle bound is reached,
       then emits the terminal exhaustion body as an SSE ``data:`` chunk.
    """
    first_provider = _first_provider_name(model_config)

    async def _gen():
        cycle = next_cycle
        current_exhaustion = exhaustion_response
        while True:
            _log_chain_hold_start(path, hold_seconds, cycle, max_cycles)
            comment = _build_chain_hold_comment(
                first_provider,
                hold_seconds,
                _exhaustion_diagnostics(current_exhaustion),
            )
            async for c in _hold_feedback(request, hold_seconds, comment):
                yield c
            try:
                if await request.is_disconnected():
                    return
            except Exception:
                pass
            try:
                result = await cycle_fn(request, path, model_config, config)
            except ChainExhaustedError as exc:
                if max_cycles != 0 and cycle >= max_cycles:
                    yield _response_to_sse_bytes(exc.response)
                    return
                current_exhaustion = exc.response
                cycle += 1
                continue
            # Success — replay the cycle's response body (streaming or plain).
            if _is_streaming_response(result):
                async for chunk in result.body_iterator:
                    yield chunk
            else:
                yield _response_to_sse_bytes(result)
            return

    return StreamingResponse(_gen(), media_type="text/event-stream")


async def _run_chain_cycles(
    request,
    path: str,
    model_config: dict,
    config: dict,
    cycle_fn,
) -> Response:
    """Run the provider chain with cycle-hold semantics.

    Each call to *cycle_fn* runs one full fallback cycle from the FIRST
    provider with fresh per-request state; on exhaustion it raises
    :class:`ChainExhaustedError` carrying the exhaustion response.

    When the chain-hold feature is enabled (``chain_hold_seconds`` and/or
    ``chain_hold_max_cycles`` configured), an exhausted cycle holds the
    request for ``chain_hold_seconds`` (default 300) before starting a NEW
    cycle, so short cooldowns (provider cooldown, stall circuit breaker,
    time-window edges) can expire. The number of hold-retry cycles is bounded
    by ``chain_hold_max_cycles`` (default 3; 0 = infinite). After the bound
    the exhaustion response is returned unchanged.

    Streaming requests receive periodic SSE hold-feedback comments (AC3);
    non-streaming requests are held silently (deferred). A client disconnect
    aborts the hold promptly (AC4).

    When the feature is not configured, *cycle_fn* is run exactly once —
    legacy single-pass behavior with no hold.
    """
    if not _chain_hold_enabled(config):
        try:
            return await cycle_fn(request, path, model_config, config)
        except ChainExhaustedError as exc:
            return exc.response

    hold_seconds = _get_chain_hold_seconds(config)
    max_cycles = _get_chain_hold_max_cycles(config)

    cycle = 0
    while True:
        try:
            return await cycle_fn(request, path, model_config, config)
        except ChainExhaustedError as exc:
            if max_cycles != 0 and cycle >= max_cycles:
                return exc.response
            _log_chain_hold_start(path, hold_seconds, cycle, max_cycles)
            if await _request_is_streaming(request):
                # Streaming: return a streaming response that emits SSE hold
                # comments, sleeps, then runs the remaining cycles inside the
                # generator (AC3). Cycle 0 already ran; resume at cycle + 1.
                return _build_streaming_hold_response(
                    request,
                    path,
                    model_config,
                    config,
                    cycle_fn,
                    next_cycle=cycle + 1,
                    max_cycles=max_cycles,
                    hold_seconds=hold_seconds,
                    exhaustion_response=exc.response,
                )
            # Non-streaming: silent (deferred) hold, then a new cycle from
            # the first provider.
            if await _hold_sleep(request, hold_seconds):
                # Client disconnected during the hold — abort (AC4).
                return exc.response
            cycle += 1


async def _proxy_with_remote_fallback_cycle(
    request,
    path: str,
    model_config: dict,
    config: dict,
) -> Response:
    """Proxy a request to a remote model with provider fallback (one cycle).

    Iterates through the model's configured providers (in order) and
    returns the first successful response.  On failure (connection error
    or HTTP status >= 400), the provider is marked with a cooldown and
    the next provider is tried.

    Runs ONE cycle: when every provider is exhausted, raises
    :class:`ChainExhaustedError` carrying the exhaustion response so the
    cycle-hold wrapper (``proxy_with_remote_fallback``) can hold + restart
    from the first provider (LP-0MSH94Z7K007VKC9).

    Args:
        request: The incoming FastAPI Request.
        path: The API path to proxy (e.g., ``v1/chat/completions``).
        model_config: Model configuration dict with a ``providers`` list.
        config: Server configuration dict (for ``provider_cooldown_seconds``).

    Returns:
        A ``Response`` from a successful provider.

    Raises:
        ChainExhaustedError: when all providers are exhausted (the carried
            response is the 503/429/error response the cycle would return).
    """
    cooldown_seconds = _get_cooldown_seconds(config)
    all_slot_exhaustion = True
    any_provider_tried = False
    prev_provider: str | None = None
    fallback_reason: str | None = None
    # SSE comment emitted on the next successful streaming response after a
    # mid-stream re-route (LP-0MSF1PUM90099ZSW). ``{to}`` is filled in once
    # the next provider is resolved.
    _pending_reroute: str | None = None
    # AC3 (LP-0MSGU3JNU0092AFQ): remember the first reasoning_content
    # round-trip 400 so the exhausted path returns a synthetic error with
    # remediation guidance instead of the raw upstream body (or a 503 whose
    # diagnostics leak it).
    _reasoning_roundtrip_response: Response | None = None

    ptr = _get_proxy_to_remote()

    # Diagnostics: record attempts (ordered) for inclusion in exhausted responses
    attempts: list[dict[str, Any]] = []
    attempted_provider_names: set[str] = set()
    # Failure-domain keys already failed/stalled THIS request — same-gateway
    # API-key siblings are skipped so re-route hops to a different gateway
    # (LP-0MSG45I8Q0020N1F). Per-request only; cross-request quarantine stays
    # with the Tier-3 stall circuit breaker (LP-0MSG45LOO007K236).
    attempted_domains: set[str] = set()

    # Preserve first model-loading response so single-provider models
    # do not collapse into generic "All providers exhausted".
    first_model_loading_response: Response | None = None

    while True:
        provider_cfg = _resolve_provider_with_exclusions(
            model_config, attempted_provider_names, attempted_domains,
        )
        if provider_cfg is None:
            break

        provider_name = provider_cfg.get("name", "unknown")
        attempted_provider_names.add(provider_name)
        provider_type = provider_cfg.get("type", "remote")
        try:
            # Mark that we attempted this provider
            any_provider_tried = True

            # Proactive rate-limit check for remote providers
            provider_rpm = int(provider_cfg.get("rate_limit_rpm", 0) or 0)
            if provider_rpm > 0:
                from proxy.rate_limiter import get_rate_limiter
                allowed = await get_rate_limiter().check_and_increment(
                    provider_name, provider_rpm, window_seconds=60
                )
                if not allowed:
                    logger.warning(
                        "Rate limited: skipping provider=%s model=%s (limit=%d rpm)",
                        provider_name, path, provider_rpm,
                    )
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type="remote",
                        status="rate_limited",
                        status_code=429,
                    )
                    continue

            response = await ptr(request, path, provider_cfg)

            # Pre-flight remote streaming responses: detect a pre-content
            # finish_reason: error / empty / stream-exception so the fallback
            # chain re-routes to the next provider instead of surfacing a
            # bare error to the client (LP-0MSETOTWY000SU0Z, Recommendation 1).
            if provider_type == "remote" and _has_next_provider(
                model_config, attempted_provider_names, attempted_domains,
            ):
                try:
                    response = await _preflight_streaming_response(
                        response, request, provider_name
                    )
                except StreamingPreContentError as exc:
                    mark_provider_unavailable(provider_name, cooldown_seconds)
                    attempted_domains.add(_failure_domain_key(provider_cfg))
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="stream_error",
                        error=exc.reason,
                    )
                    fallback_reason = exc.reason
                    prev_provider = provider_name
                    all_slot_exhaustion = False
                    continue
                except StreamingRecoverableAfterReasoningError as exc:
                    # Mid-stream re-route (LP-0MSF1PUM90099ZSW): reasoning was
                    # delivered but no final content / no tool_calls, so the
                    # client has not committed to an answer. Re-route the SAME
                    # request to the next provider; the buffered intermediate
                    # output is discarded (never reaches the client).
                    mark_provider_unavailable(provider_name, cooldown_seconds)
                    # Same-gateway exclusion (LP-0MSG45I8Q0020N1F): the stalled
                    # entry's failure domain is skipped for the rest of this
                    # request so the re-route hops straight to a different
                    # gateway instead of retrying via another API-key sibling.
                    attempted_domains.add(_failure_domain_key(provider_cfg))
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="stream_reroute",
                        error=exc.reason,
                    )
                    fallback_reason = exc.reason
                    prev_provider = provider_name
                    all_slot_exhaustion = False
                    _pending_reroute = f": re-route provider={provider_name}->{{to}} reason={exc.reason}"
                    logger.warning(
                        "Mid-stream re-route: provider=%s model=%s reason=%s "
                        "(no final content committed; routing to next provider)",
                        provider_name, path, exc.reason,
                    )
                    continue

            # Shared primitive: handle streaming success
            stream_result = _handle_streaming_success(
                response, provider_name, provider_type, attempts,
                prev_provider, fallback_reason, path,
            )
            if stream_result is not None:
                if _pending_reroute:
                    stream_result = _prepend_sse_comment(
                        stream_result,
                        _pending_reroute.format(to=provider_name),
                    )
                    _pending_reroute = None
                return stream_result

            # Safely extract a small body snippet for diagnostics
            body_text = _response_body_text(response)

            # Check for HTTP error status
            if _is_http_error_status(response.status_code):
                if _is_model_loading_response(response, body_text):
                    fallback_reason = "model_loading"
                    prev_provider = provider_name
                    if first_model_loading_response is None:
                        first_model_loading_response = response
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="model_loading",
                        status_code=int(response.status_code),
                        body_snippet=(body_text[:512] if body_text else None),
                    )
                    all_slot_exhaustion = False
                    continue

                # Usage-limit reset (LP-0MSLJPOCC0001ROJ): GoUsageLimitError /
                # FreeUsageLimitError-with-reset-time quarantines the failing
                # API-key ACCOUNT until the computed reset time + 2m margin.
                # Not the whole endpoint: distinct api_key_env entries on the
                # same gateway have independent limits (LP-0MSMBWB23009XYPW).
                _reset_seconds = _usage_limit_reset_seconds(response, body_text)
                if _reset_seconds is not None:
                    _reset_account = _usage_limit_account_key(provider_cfg)
                    _usage_reset_at[_reset_account] = time.time() + _reset_seconds
                    fallback_reason = "usage_limit_reset"
                    prev_provider = provider_name
                    attempted_domains.add(_reset_account)
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="usage_limit_reset",
                        status_code=int(response.status_code),
                        body_snippet=(body_text[:512] if body_text else None),
                        reset_in_seconds=int(_reset_seconds),
                    )
                    all_slot_exhaustion = False
                    continue

                # FreeUsageLimitError: apply 3-hour cooldown on affected provider
                # so the fallback chain routes to paid alternatives instead of
                # repeatedly retrying the exhausted free tier.
                if _is_free_usage_limit_error(response, body_text):
                    fallback_reason = "free_usage_limit"
                    prev_provider = provider_name
                    mark_provider_unavailable(provider_name, _FREE_USAGE_LIMIT_COOLDOWN_SECONDS)
                    attempted_domains.add(_failure_domain_key(provider_cfg))
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="free_usage_limit",
                        status_code=int(response.status_code),
                        body_snippet=(body_text[:512] if body_text else None),
                        cooldown_seconds=_FREE_USAGE_LIMIT_COOLDOWN_SECONDS,
                    )
                    all_slot_exhaustion = False
                    continue

                # Shared primitive: HTTP error with cooldown
                _handle_http_error_with_cooldown(
                    response, provider_name, provider_type,
                    cooldown_seconds, attempts, body_text,
                )
                # AC3 (LP-0MSGU3JNU0092AFQ): remember this specific 400 so the
                # exhausted path can return a synthetic error instead of the
                # raw upstream body / exhausted-503-with-diagnostics.
                if _reasoning_roundtrip_response is None and _is_reasoning_content_roundtrip_error(response):
                    _reasoning_roundtrip_response = response
                attempted_domains.add(_failure_domain_key(provider_cfg))
                fallback_reason = f"HTTP {response.status_code}"
                prev_provider = provider_name
                if response.status_code != 429:
                    all_slot_exhaustion = False
                _observe_http_error_400(
                    response, provider_name, provider_type, path, body_text, fallback_reason,
                )
                continue

            # Treat empty successful responses as failures to allow fallback
            try:
                resp_json = None
                try:
                    resp_json = json.loads(body_text) if body_text else None
                except Exception:
                    resp_json = None
                if _is_empty_response(body_text or '', resp_json):
                    # Shared primitive: check for reasoning_content promotion
                    promoted = _resolve_reasoning_content_promotion(
                        response, provider_name, provider_type, attempts,
                        prev_provider, fallback_reason, path, body_text,
                    )
                    if promoted is not None:
                        # LP-0MR4ZIGDT004A3E1: Surface resolved provider/model for Pi extension
                        _add_resolved_model_header(promoted, provider_cfg)
                        return promoted

                    # Shared primitive: empty response with cooldown
                    _handle_empty_response_with_cooldown(
                        response, provider_name, provider_type,
                        cooldown_seconds, attempts, body_text,
                    )
                    attempted_domains.add(_failure_domain_key(provider_cfg))
                    fallback_reason = "empty_response"
                    prev_provider = provider_name
                    all_slot_exhaustion = False
                    continue
            except Exception:
                pass

            # Shared primitive: success path
            result = _build_fallback_success_response(
                response, provider_name, provider_type, attempts,
                prev_provider, fallback_reason, path, body_text,
            )
            # LP-0MR4ZIGDT004A3E1: Surface resolved provider/model for Pi extension
            _add_resolved_model_header(result, provider_cfg)
            return result

        except Exception as exc:
            # Shared primitive: handle connection errors
            if _handle_connection_error_in_fallback(
                exc, provider_name, provider_type, cooldown_seconds, attempts,
            ):
                any_provider_tried = True
                attempted_domains.add(_failure_domain_key(provider_cfg))
                fallback_reason = str(type(exc).__name__)
                prev_provider = provider_name
                all_slot_exhaustion = False
                continue
            # Non-connection error — propagate
            raise

    # All providers exhausted — log diagnostic details
    unavailable = _log_exhausted_providers(model_config, path)

    # Record time-window skips as distinct diagnostics so operators can tell
    # whether providers were excluded by their available_times windows rather
    # than by cooldown/errors (LP-0MS4ETBNO0022QAC).
    for skipped in _providers_outside_window(model_config):
        _record_attempt(
            attempts,
            provider=skipped["name"],
            type=skipped["type"],
            status="outside_time_window",
        )

    # Distinguishable exhaustion: when time windows are the *only* reason no
    # provider could be used, surface a specific message instead of the generic
    # "All providers exhausted" (LP-0MS4ETBNO0022QAC).
    time_window_exhausted = _build_time_window_exhausted_response(
        attempts, unavailable, any_provider_tried,
    )
    if time_window_exhausted is not None:
        raise ChainExhaustedError(time_window_exhausted)

    if not any_provider_tried:
        raise ChainExhaustedError(
            _build_exhausted_response(all_local_slot_exhaustion=False, unavailable_providers=unavailable, diagnostics=attempts)
        )

    if first_model_loading_response is not None:
        logger.info(
            "Returning model_loading response instead of generic exhausted message for model=%s",
            path,
        )
        raise ChainExhaustedError(first_model_loading_response)

    # AC3 (LP-0MSGU3JNU0092AFQ): never surface the raw upstream
    # reasoning_content round-trip 400 (or a 503 whose diagnostics leak it);
    # return a synthetic error with remediation guidance instead.
    if _reasoning_roundtrip_response is not None:
        logger.warning(
            "Intercepting reasoning_content round-trip 400 for model=%s; "
            "returning synthetic error instead of raw upstream body",
            path,
        )
        raise ChainExhaustedError(_build_reasoning_content_roundtrip_error())

    raise ChainExhaustedError(
        _build_exhausted_response(all_local_slot_exhaustion=all_slot_exhaustion, unavailable_providers=unavailable, diagnostics=attempts)
    )


async def proxy_with_remote_fallback(
    request,
    path: str,
    model_config: dict,
    config: dict,
) -> Response:
    """Proxy a request to a remote model with provider fallback and chain-hold
    retry semantics (LP-0MSH94Z7K007VKC9).

    Runs the provider chain; when every provider in the fallback chain is
    exhausted (final model unavailable), the request is HELD for
    ``server.chain_hold_seconds`` (default 300) and a NEW cycle starts from
    the FIRST provider — giving short cooldowns time to expire instead of
    erroring immediately. Streaming requests receive periodic SSE feedback
    comments (``: chain exhausted (...); retrying from <first> in <Ns>``);
    non-streaming requests are held silently. The number of hold-retry cycles
    is bounded by ``server.chain_hold_max_cycles`` (default 3; 0 = infinite);
    after the bound the exhaustion response is returned unchanged. A client
    disconnect aborts the hold promptly.

    Args:
        request: The incoming FastAPI Request.
        path: The API path to proxy (e.g., ``v1/chat/completions``).
        model_config: Model configuration dict with a ``providers`` list.
        config: Server configuration dict (for ``provider_cooldown_seconds``).

    Returns:
        A ``Response`` from a successful provider, or the 503/429 exhaustion
        response once the hold-retry bound is reached.
    """
    return await _run_chain_cycles(
        request, path, model_config, config, _proxy_with_remote_fallback_cycle,
    )


async def _proxy_with_fallback_cycle(
    request,
    path: str,
    model_config: dict,
    config: dict,
) -> Response:
    """Proxy a request with fallback across both local and remote providers (one cycle).

    Iterates through the model's configured providers (in order) and
    tries each one.  Local providers are dispatched via ``proxy_to_local``,
    remote providers via ``proxy_to_remote``.  On failure (connection error,
    HTTP status >= 400 for remote, slot exhaustion for local), the provider
    enters cooldown and the next provider is tried.

    Runs ONE cycle: when every provider is exhausted, raises
    :class:`ChainExhaustedError` carrying the exhaustion response so the
    cycle-hold wrapper (``proxy_with_fallback``) can hold + restart from the
    first provider (LP-0MSH94Z7K007VKC9).

    Args:
        request: The incoming FastAPI Request.
        path: The API path to proxy.
        model_config: Model configuration dict with a ``providers`` list.
        config: Server configuration dict.

    Returns:
        A ``Response`` from a successful provider.

    Raises:
        ChainExhaustedError: when all providers are exhausted (the carried
            response is the 503/429/error response the cycle would return).
    """
    cooldown_seconds = _get_cooldown_seconds(config)
    local_slot_retry_attempts = _get_local_slot_retry_attempts(config)
    local_slot_retry_delay_seconds = _get_local_slot_retry_delay_seconds(config)
    slot_unavailable_cooldown = _get_slot_unavailable_retry_after(config)
    all_slot_exhaustion = True
    any_provider_tried = False
    prev_provider: str | None = None
    fallback_reason: str | None = None
    # SSE comment emitted on the next successful streaming response after a
    # mid-stream re-route (LP-0MSF1PUM90099ZSW). ``{to}`` is filled in once
    # the next provider is resolved.
    _pending_reroute: str | None = None

    # Accumulate slot counts when local providers report slot exhaustion
    total_slots_sum = 0
    available_slots_sum = 0

    # Track the first error response so we can return it when all providers
    # are exhausted, instead of the generic "All providers exhausted" message.
    # This preserves the actual error (e.g. backend_unavailable, concurrency)
    # that a single-provider model would have returned directly.
    _first_error_response = None

    ptr_remote = _get_proxy_to_remote()
    ptr_local = _get_proxy_to_local()

    # Read request body for smart-routing decisions (LP-0MRCSSBTM002NK3B).
    # The body is cached by Starlette/FastAPI after the first read, so this
    # is safe to call before dispatching to local/remote.
    _raw_body = await request.body()
    try:
        body_json = json.loads(_raw_body) if _raw_body else {}
    except Exception:
        body_json = {}

    # Resolve session_id for per-session cache state tracking (LP-0MRMMBZ7T007ER59)
    _session_id: str | None = None
    try:
        from proxy.session import _resolve_session_id_header
        _session_id, _ = _resolve_session_id_header(getattr(request, "headers", {}))
    except Exception:
        pass

    # Diagnostics: record attempts (ordered) for inclusion in exhausted responses
    attempts: list[dict[str, Any]] = []
    attempted_provider_names: set[str] = set()
    # Failure-domain keys already failed/stalled THIS request — same-gateway
    # API-key siblings are skipped so re-route hops to a different gateway
    # (LP-0MSG45I8Q0020N1F). Per-request only; cross-request quarantine stays
    # with the Tier-3 stall circuit breaker (LP-0MSG45LOO007K236).
    attempted_domains: set[str] = set()

    while True:
        provider_cfg = _resolve_provider_with_exclusions(
            model_config, attempted_provider_names, attempted_domains,
        )
        if provider_cfg is None and fallback_reason == "local_lease_active":
            # Local lease-active is expected contention, not provider failure.
            # For transparent fallback, allow trying the next remote provider
            # even if it is currently in cooldown.
            providers = model_config.get("providers") or []
            for candidate in providers:
                if not isinstance(candidate, dict):
                    continue
                candidate_name = candidate.get("name", "")
                if candidate_name in attempted_provider_names:
                    continue
                if _failure_domain_key(candidate) in attempted_domains:
                    continue
                if candidate.get("type") != "remote":
                    continue
                if not _is_within_allowed_window(candidate):
                    continue
                provider_cfg = candidate
                break

        if provider_cfg is None:
            break

        provider_name = provider_cfg.get("name", "unknown")
        attempted_provider_names.add(provider_name)
        provider_type = provider_cfg.get("type", "remote")

        try:
            # Mark attempt
            any_provider_tried = True
            if provider_type == "local":
                # Local concurrency limit check (LP-0MR5MAJNM005R905):
                # if local concurrency limit (session_slot_pool_size) is exceeded, skip to next
                # provider without marking local as unavailable.
                cur_local, max_local = _get_local_concurrency_info(config)
                if cur_local >= max_local:
                    # Per-mode contention queue (LP-0MSORQVK50012Q4D): in cheap
                    # mode with policy=queue, a request that finds local slots
                    # exhausted QUEUES (bounded by caps) instead of immediately
                    # falling back to the next remote provider. When a slot
                    # frees within the caps it is dispatched local; when the
                    # caps are exceeded it falls back exactly as today.
                    _cq_action, _cq_reason, _cq_elapsed = (
                        await _maybe_queue_for_local_slot(
                            config, cur_local, max_local, request, body_json,
                            model_config, provider_cfg, _session_id,
                        )
                    )
                    if _cq_action == "dispatch":
                        # A slot freed within the caps — dispatch local below.
                        _set_queue_wait_on_request(request, _cq_elapsed)
                        try:
                            from proxy import contention_queue

                            _cq_m = contention_queue.metrics()
                        except Exception:
                            _cq_m = {}
                        logger.info(
                            "contention_queue_dispatch provider=%s session=%s "
                            "queued_duration=%.2fs policy=queue depth=%d",
                            provider_name, _session_id or "unknown",
                            _cq_elapsed or 0.0,
                            _cq_m.get("contention_queue_depth", 0),
                        )
                        try:
                            from proxy.metrics import record_contention_queued

                            record_contention_queued(_cq_elapsed or 0.0)
                        except Exception:
                            pass
                    elif _cq_action == "fallback":
                        # Caps exceeded — fall back to the next remote provider
                        # exactly as today (fallback-after-queue recorded with
                        # the elapsed queue wait, F4 AC2).
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="fallback_after_queue",
                            active=cur_local,
                            max=max_local,
                            elapsed_wait_seconds=round(_cq_elapsed or 0.0, 3),
                        )
                        fallback_reason = "fallback_after_queue"
                        prev_provider = provider_name
                        all_slot_exhaustion = False
                        logger.info(
                            "contention_queue_fallback_after_queue provider=%s "
                            "session=%s queued_duration=%.2fs",
                            provider_name, _session_id or "unknown",
                            _cq_elapsed or 0.0,
                        )
                        try:
                            from proxy.metrics import record_contention_fallback_after_queue

                            record_contention_fallback_after_queue()
                        except Exception:
                            pass
                        continue
                    elif _cq_action == "context_bypass":
                        # Context bypasses never queue (AC4): fall back exactly
                        # as today with the context skip reason.
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="cached_tokens_skip",
                            reason=_cq_reason,
                        )
                        fallback_reason = _cq_reason
                        prev_provider = provider_name
                        all_slot_exhaustion = False
                        continue
                    else:  # fallback_policy — fast mode unchanged
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="local_concurrency_limit",
                            active=cur_local,
                            max=max_local,
                        )
                        fallback_reason = "local_concurrency_limit"
                        prev_provider = provider_name
                        all_slot_exhaustion = False
                        continue

                # Smart routing: skip local when cache is cold and request is large
                # (LP-0MRCSSBTM002NK3B). This avoids expensive full re-prefill of
                # large contexts when the slot cache is invalidated.
                # Smart routing: skip local when cache is cold and request is large
                # (LP-0MRP44W7I0085I6N). Uses cached_tokens ratio from the last
                # local response instead of inferred cache-cold state.
                _llama_model = provider_cfg.get("llama_model", "")
                # Clamp the WARM threshold to the actual per-slot context so
                # prompts that cannot fit the KV slot fall through to remote
                # BEFORE context exhaustion (LP-0MSAZXXDY005AWA1). COLD stays
                # as the economic new-token threshold so the (cold, warm] band
                # keeps the cached_ratio check reachable (LP-0MSI2M5BT004BCDP).
                _cold_threshold, _warm_threshold = _effective_large_context_thresholds(
                    config
                )
                # Resolve the native tokenizer + multiplier via the shared
                # helper (LP-0MSEQ71IF0003FRT): a named tokenizer (e.g.
                # ``tokenizer: qwen3`` on the model entry) replaces the
                # tiktoken-vs-Qwen3 mismatch (~1.69x undercount,
                # LP-0MSAOQTJS000FFVM F2/F3) with exact counts and forces
                # the multiplier to 1.0; without a tokenizer, the server-
                # level / per-model token_estimate_multiplier still applies
                # (LP-0MSEGPO77005CYCQ F2).
                _tokenizer, _multiplier = _get_tokenizer_for_model(
                    model_config, config
                )
                _estimated_tokens = await _estimate_effective_prompt_tokens_for_routing(
                    request,
                    body_json,
                    tokenizer=_tokenizer,
                )
                if _multiplier != 1.0:
                    _estimated_tokens = int(_estimated_tokens * _multiplier)
                # Compute once for reuse in logs and reason detection
                _routing_cached_ratio = _get_cached_ratio(_llama_model, _session_id)
                _routing_new_tokens = int(
                    _estimated_tokens * (1 - _routing_cached_ratio)
                )
                logger.info(
                    "routing_check provider=%s model=%s "
                    "estimated_tokens=%d cold_threshold=%d warm_threshold=%d "
                    "new_tokens=%d cached_ratio=%.2f messages=%d session=%s",
                    provider_name,
                    _llama_model or "unknown",
                    _estimated_tokens,
                    _cold_threshold,
                    _warm_threshold,
                    _routing_new_tokens,
                    _routing_cached_ratio,
                    len(body_json.get("messages", [])) if isinstance(body_json, dict) else -1,
                    _session_id or "unknown",
                )
                # Context-pressure compaction signal (LP-0MSDCLQ2W001LGWC):
                # KV reads scale linearly with context (20 KB/token at f16),
                # so sessions approaching the per-slot limit decode at a
                # fraction of their earlier speed. Warn so agents/operators
                # compact before decode degrades.
                if should_warn_context_pressure(_estimated_tokens, config):
                    _active_ctx = _get_active_local_ctx_size(config)
                    _per_slot = effective_per_slot_threshold(
                        _active_ctx,
                        _get_active_local_slots(config),
                    )
                    logger.warning(
                        "context_pressure session=%s estimated_tokens=%d "
                        "per_slot_ctx=%d ratio=%.2f >= %.2f; consider "
                        "compacting the session history to reduce local decode "
                        "cost (KV read scales with context)",
                        _session_id or "unknown",
                        _estimated_tokens,
                        _per_slot,
                        context_pressure_ratio(
                            _estimated_tokens,
                            _active_ctx,
                            _get_active_local_slots(config),
                        ),
                        _get_context_pressure_warn_ratio(config),
                    )
                _skip_local = _should_skip_local(
                    _llama_model,
                    _session_id,
                    body_json,
                    _cold_threshold,
                    estimated_tokens=_estimated_tokens,
                    warm_cache_threshold=_warm_threshold,
                )
                if _skip_local:
                    # Determine reason: the context-too-large hard cap fires
                    # when estimated_tokens > warm_cache_threshold (total
                    # context too large regardless of cache state), logged as
                    # ``context_too_large`` (LP-0MSF8XDG7000PERM).  Otherwise
                    # the cold-cache new-token check triggered.
                    if _warm_threshold > 0 and _estimated_tokens > _warm_threshold:
                        _skip_reason = "context_too_large"
                    else:
                        _skip_reason = "large_context_bypass"
                    logger.info(
                        "routing_skip_local provider=%s model=%s "
                        "estimated_tokens=%d cold_threshold=%d warm_threshold=%d "
                        "new_tokens=%d cached_ratio=%.2f "
                        "reason=%s → skipping local, routing to next remote provider "
                        "session=%s",
                        provider_name,
                        _llama_model or "unknown",
                        _estimated_tokens,
                        _cold_threshold,
                        _warm_threshold,
                        _routing_new_tokens,
                        _routing_cached_ratio,
                        _skip_reason,
                        _session_id or "unknown",
                    )
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="cached_tokens_skip",
                        estimated_tokens=_estimated_tokens,
                        cold_threshold=_cold_threshold,
                        warm_threshold=_warm_threshold,
                        new_tokens=_routing_new_tokens,
                        cached_ratio=_routing_cached_ratio,
                        reason=_skip_reason,
                    )
                    fallback_reason = _skip_reason
                    prev_provider = provider_name
                    all_slot_exhaustion = False
                    continue

                response = await ptr_local(request, path)
            else:
                # Proactive rate-limit check for remote providers
                # (LP-0MQNRDUP4008KT6T: rate limiter for remote models)
                provider_rpm = int(provider_cfg.get("rate_limit_rpm", 0) or 0)
                if provider_rpm > 0:
                    from proxy.rate_limiter import get_rate_limiter
                    allowed = await get_rate_limiter().check_and_increment(
                        provider_name, provider_rpm, window_seconds=60
                    )
                    if not allowed:
                        logger.warning(
                            "Rate limited: skipping provider=%s model=%s (limit=%d rpm)",
                            provider_name, path, provider_rpm,
                        )
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type="remote",
                            status="rate_limited",
                            status_code=429,
                        )
                        fallback_reason = "rate_limited"
                        prev_provider = provider_name
                        all_slot_exhaustion = False
                        continue

                response = await ptr_remote(request, path, provider_cfg)

            # Pre-flight remote streaming responses: detect a pre-content
            # finish_reason: error / empty / stream-exception so the fallback
            # chain re-routes to the next provider instead of surfacing a
            # bare error to the client (LP-0MSETOTWY000SU0Z, Recommendation 1).
            if provider_type == "remote" and _has_next_provider(
                model_config, attempted_provider_names, attempted_domains,
            ):
                try:
                    response = await _preflight_streaming_response(
                        response, request, provider_name
                    )
                except StreamingPreContentError as exc:
                    mark_provider_unavailable(provider_name, cooldown_seconds)
                    attempted_domains.add(_failure_domain_key(provider_cfg))
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="stream_error",
                        error=exc.reason,
                    )
                    fallback_reason = exc.reason
                    prev_provider = provider_name
                    all_slot_exhaustion = False
                    continue
                except StreamingRecoverableAfterReasoningError as exc:
                    # Mid-stream re-route (LP-0MSF1PUM90099ZSW): reasoning was
                    # delivered but no final content / no tool_calls, so the
                    # client has not committed to an answer. Re-route the SAME
                    # request to the next provider; the buffered intermediate
                    # output is discarded (never reaches the client).
                    mark_provider_unavailable(provider_name, cooldown_seconds)
                    # Same-gateway exclusion (LP-0MSG45I8Q0020N1F): the stalled
                    # entry's failure domain is skipped for the rest of this
                    # request so the re-route hops straight to a different
                    # gateway instead of retrying via another API-key sibling.
                    attempted_domains.add(_failure_domain_key(provider_cfg))
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="stream_reroute",
                        error=exc.reason,
                    )
                    fallback_reason = exc.reason
                    prev_provider = provider_name
                    all_slot_exhaustion = False
                    _pending_reroute = f": re-route provider={provider_name}->{{to}} reason={exc.reason}"
                    logger.warning(
                        "Mid-stream re-route: provider=%s model=%s reason=%s "
                        "(no final content committed; routing to next provider)",
                        provider_name, path, exc.reason,
                    )
                    continue

            # Shared primitive: handle streaming success
            stream_result = _handle_streaming_success(
                response, provider_name, provider_type, attempts,
                prev_provider, fallback_reason, path,
            )
            if stream_result is not None:
                # LP-0MR4ZIGDT004A3E1: Surface resolved provider/model for Pi extension
                _add_resolved_model_header(stream_result, provider_cfg)
                if _pending_reroute:
                    stream_result = _prepend_sse_comment(
                        stream_result,
                        _pending_reroute.format(to=provider_name),
                    )
                    _pending_reroute = None
                return stream_result

            # Capture the first non-success response so we can return it when
            # all providers are exhausted (instead of the generic exhausted message).
            # Do not capture local slot-exhaustion responses here — they are
            # routing signals (busy/lease-active), not terminal provider errors.
            if _first_error_response is None and response.status_code >= 400:
                _first_slot_info = _parse_slot_exhaustion(response)
                if not (provider_type == "local" and _first_slot_info is not None):
                    _first_error_response = response

            # Extract small response snippet for diagnostics
            body_text = _response_body_text(response)

            # Check for slot exhaustion (local model)
            slot_info = _parse_slot_exhaustion(response)
            if slot_info:
                slot_reason = str(slot_info.get("reason") or "").strip().lower()

                # Lease-aware behavior: when local is reserved for another
                # session, do not retry local and do not put local in cooldown.
                # Route to the next provider in the chain immediately.
                if provider_type == "local" and slot_reason == "local_lease_active":
                    fallback_reason = "local_lease_active"
                    prev_provider = provider_name
                    total_slots_sum += int(slot_info.get("total_slots", 0) or 0)
                    available_slots_sum += int(slot_info.get("available_slots", 0) or 0)
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="local_lease_active",
                        slot_info=slot_info,
                    )
                    all_slot_exhaustion = False
                    continue

                # Optional local retry window for startup races where router/model
                # is loaded but slot probes briefly report 0 available.
                if provider_type == "local" and local_slot_retry_attempts > 0:
                    resolved_after_retry = False
                    for retry_idx in range(1, local_slot_retry_attempts + 1):
                        if local_slot_retry_delay_seconds > 0:
                            await asyncio.sleep(local_slot_retry_delay_seconds)

                        retry_response = await ptr_local(request, path)
                        retry_body_text = _response_body_text(retry_response)
                        retry_slot_info = _parse_slot_exhaustion(retry_response)

                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="slot_exhaustion_retry",
                            retry_attempt=retry_idx,
                            status_code=int(getattr(retry_response, "status_code", 0) or 0),
                            slot_info=retry_slot_info,
                            body_snippet=(retry_body_text[:512] if retry_body_text else None),
                        )

                        if retry_slot_info:
                            # Preserve lease-aware semantics during retry loop.
                            retry_reason = str(retry_slot_info.get("reason") or "").strip().lower()
                            if retry_reason == "local_lease_active":
                                slot_info = retry_slot_info
                                break
                            slot_info = retry_slot_info
                            continue

                        response = retry_response
                        body_text = retry_body_text
                        slot_info = None
                        resolved_after_retry = True
                        break

                    if resolved_after_retry and slot_info is None:
                        # Continue evaluating updated response below.
                        pass
                    else:
                        # If retries ended with lease-active, skip cooldown and
                        # route to next provider immediately.
                        final_reason = str((slot_info or {}).get("reason") or "").strip().lower()
                        if provider_type == "local" and final_reason == "local_lease_active":
                            fallback_reason = "local_lease_active"
                            prev_provider = provider_name
                            total_slots_sum += int((slot_info or {}).get("total_slots", 0) or 0)
                            available_slots_sum += int((slot_info or {}).get("available_slots", 0) or 0)
                            _record_attempt(
                                attempts,
                                provider=provider_name,
                                type=provider_type,
                                status="local_lease_active",
                                slot_info=slot_info,
                            )
                            all_slot_exhaustion = False
                            continue

                        mark_provider_unavailable(provider_name, slot_unavailable_cooldown)
                        attempted_domains.add(_failure_domain_key(provider_cfg))
                        fallback_reason = "slot_exhaustion"
                        prev_provider = provider_name
                        total_slots_sum += int(slot_info.get("total_slots", 0) or 0)
                        available_slots_sum += int(slot_info.get("available_slots", 0) or 0)
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="slot_exhaustion",
                            slot_info=slot_info,
                        )
                        continue
                else:
                    mark_provider_unavailable(provider_name, slot_unavailable_cooldown)
                    attempted_domains.add(_failure_domain_key(provider_cfg))
                    fallback_reason = "slot_exhaustion"
                    prev_provider = provider_name
                    total_slots_sum += int(slot_info.get("total_slots", 0) or 0)
                    available_slots_sum += int(slot_info.get("available_slots", 0) or 0)
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="slot_exhaustion",
                        slot_info=slot_info,
                    )
                    continue

            # Check for HTTP error status
            if _is_http_error_status(response.status_code):
                if _is_model_loading_response(response, body_text):
                    fallback_reason = "model_loading"
                    prev_provider = provider_name
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="model_loading",
                        status_code=int(response.status_code),
                        body_snippet=(body_text[:512] if body_text else None),
                    )
                    all_slot_exhaustion = False
                    continue

                # Local 5xx can be transient right after startup (slot routing,
                # backend warm-up). Retry local a few times before falling back.
                if provider_type == "local" and int(response.status_code) >= 500 and local_slot_retry_attempts > 0:
                    for retry_idx in range(1, local_slot_retry_attempts + 1):
                        if local_slot_retry_delay_seconds > 0:
                            await asyncio.sleep(local_slot_retry_delay_seconds)

                        retry_response = await ptr_local(request, path)
                        retry_body_text = _response_body_text(retry_response)
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="local_http_retry",
                            retry_attempt=retry_idx,
                            status_code=int(getattr(retry_response, "status_code", 0) or 0),
                            body_snippet=(retry_body_text[:512] if retry_body_text else None),
                        )

                        response = retry_response
                        body_text = retry_body_text
                        if not _is_http_error_status(response.status_code):
                            break

                    if _is_model_loading_response(response, body_text):
                        fallback_reason = "model_loading"
                        prev_provider = provider_name
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="model_loading",
                            status_code=int(response.status_code),
                            body_snippet=(body_text[:512] if body_text else None),
                        )
                        all_slot_exhaustion = False
                        continue

                if _is_http_error_status(response.status_code):
                    # Local 4xx responses are typically request-shape
                    # incompatibilities (e.g. optional OpenAI fields unsupported
                    # by llama-server), not provider health failures. Allow
                    # same-request fallback, but do not poison local provider
                    # cooldown across subsequent requests.
                    if provider_type == "local" and 400 <= int(response.status_code) < 500:
                        fallback_reason = f"HTTP {response.status_code}"
                        prev_provider = provider_name
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="http_error_no_cooldown",
                            status_code=int(response.status_code),
                            body_snippet=(body_text[:512] if body_text else None),
                        )
                        all_slot_exhaustion = False
                        continue

                    # Usage-limit reset (LP-0MSLJPOCC0001ROJ): GoUsageLimitError /
                    # FreeUsageLimitError-with-reset-time quarantines the failing
                    # API-key ACCOUNT until the computed reset time + 2m margin.
                    # Not the whole endpoint: distinct api_key_env entries on the
                    # same gateway have independent limits (LP-0MSMBWB23009XYPW).
                    _reset_seconds = _usage_limit_reset_seconds(response, body_text)
                    if _reset_seconds is not None:
                        _reset_account = _usage_limit_account_key(provider_cfg)
                        _usage_reset_at[_reset_account] = time.time() + _reset_seconds
                        fallback_reason = "usage_limit_reset"
                        prev_provider = provider_name
                        attempted_domains.add(_reset_account)
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="usage_limit_reset",
                            status_code=int(response.status_code),
                            body_snippet=(body_text[:512] if body_text else None),
                            reset_in_seconds=int(_reset_seconds),
                        )
                        all_slot_exhaustion = False
                        continue

                    # FreeUsageLimitError: apply 3-hour cooldown on affected provider
                    # so the fallback chain routes to paid alternatives instead of
                    # repeatedly retrying the exhausted free tier.
                    if _is_free_usage_limit_error(response, body_text):
                        fallback_reason = "free_usage_limit"
                        prev_provider = provider_name
                        mark_provider_unavailable(provider_name, _FREE_USAGE_LIMIT_COOLDOWN_SECONDS)
                        attempted_domains.add(_failure_domain_key(provider_cfg))
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="free_usage_limit",
                            status_code=int(response.status_code),
                            body_snippet=(body_text[:512] if body_text else None),
                            cooldown_seconds=_FREE_USAGE_LIMIT_COOLDOWN_SECONDS,
                        )
                        all_slot_exhaustion = False
                        continue

                    # Shared primitive: HTTP error with cooldown
                    _handle_http_error_with_cooldown(
                        response, provider_name, provider_type,
                        cooldown_seconds, attempts, body_text,
                    )
                    attempted_domains.add(_failure_domain_key(provider_cfg))
                    fallback_reason = f"HTTP {response.status_code}"
                    prev_provider = provider_name
                    if response.status_code != 429:
                        all_slot_exhaustion = False
                    _observe_http_error_400(
                        response, provider_name, provider_type, path, body_text, fallback_reason,
                    )
                    continue

            # Treat empty successful responses as failures to allow fallback
            try:
                resp_json = None
                try:
                    resp_json = json.loads(body_text) if body_text else None
                except Exception:
                    resp_json = None
                if _is_empty_response(body_text or '', resp_json):
                    # Shared primitive: check for reasoning_content promotion
                    promoted = _resolve_reasoning_content_promotion(
                        response, provider_name, provider_type, attempts,
                        prev_provider, fallback_reason, path, body_text,
                    )
                    if promoted is not None:
                        # LP-0MR4ZIGDT004A3E1: Surface resolved provider/model for Pi extension
                        _add_resolved_model_header(promoted, provider_cfg)
                        return promoted

                    # Local empty 200 can be transient (slot busy/cancelled
                    # right after a previous request). Retry locally before
                    # falling back to remote providers.
                    if provider_type == "local" and local_slot_retry_attempts > 0:
                        resolved_after_empty_retry = False
                        for retry_idx in range(1, local_slot_retry_attempts + 1):
                            if local_slot_retry_delay_seconds > 0:
                                await asyncio.sleep(local_slot_retry_delay_seconds)
                            retry_response = await ptr_local(request, path)
                            retry_body_text = _response_body_text(retry_response)
                            _record_attempt(
                                attempts,
                                provider=provider_name,
                                type=provider_type,
                                status="local_empty_retry",
                                retry_attempt=retry_idx,
                                status_code=int(getattr(retry_response, "status_code", 0) or 0),
                                body_snippet=(retry_body_text[:512] if retry_body_text else None),
                            )
                            try:
                                retry_resp_json = json.loads(retry_body_text) if retry_body_text else None
                            except Exception:
                                retry_resp_json = None
                            if not _is_empty_response(retry_body_text or "", retry_resp_json):
                                response = retry_response
                                body_text = retry_body_text
                                resolved_after_empty_retry = True
                                break

                        if resolved_after_empty_retry:
                            # Shared primitive: check reasoning_content after retry
                            promoted2 = _resolve_reasoning_content_promotion(
                                response, provider_name, provider_type, attempts,
                                prev_provider, fallback_reason, path, body_text,
                            )
                            if promoted2 is not None:
                                # LP-0MR4ZIGDT004A3E1: Surface resolved provider/model
                                _add_resolved_model_header(promoted2, provider_cfg)
                                return promoted2
                            # Fall through to success path below.
                            pass
                        else:
                            # Shared primitive: empty response with cooldown
                            _handle_empty_response_with_cooldown(
                                response, provider_name, provider_type,
                                cooldown_seconds, attempts, body_text,
                            )
                            attempted_domains.add(_failure_domain_key(provider_cfg))
                            fallback_reason = "empty_response"
                            prev_provider = provider_name
                            all_slot_exhaustion = False
                            continue
                    else:
                        # Shared primitive: empty response with cooldown
                        _handle_empty_response_with_cooldown(
                            response, provider_name, provider_type,
                            cooldown_seconds, attempts, body_text,
                        )
                        attempted_domains.add(_failure_domain_key(provider_cfg))
                        fallback_reason = "empty_response"
                        prev_provider = provider_name
                        all_slot_exhaustion = False
                        continue
            except Exception:
                pass

            # Shared primitive: success path
            result = _build_fallback_success_response(
                response, provider_name, provider_type, attempts,
                prev_provider, fallback_reason, path, body_text,
            )
            # LP-0MR4ZIGDT004A3E1: Surface resolved provider/model for Pi extension
            _add_resolved_model_header(result, provider_cfg)
            return result

        except Exception as exc:
            # Shared primitive: handle connection errors
            if _handle_connection_error_in_fallback(
                exc, provider_name, provider_type, cooldown_seconds, attempts,
            ):
                any_provider_tried = True
                attempted_domains.add(_failure_domain_key(provider_cfg))
                fallback_reason = str(type(exc).__name__)
                prev_provider = provider_name
                all_slot_exhaustion = False
                continue
            # HTTPException from the local provider (e.g., backend busy, slot
            # queue full, concurrency limit) should also trigger fallback.
            # Only 5xx responses are retryable via fallback; 4xx errors are
            # client errors that should propagate.
            if isinstance(exc, HTTPException) and exc.status_code >= 500:
                any_provider_tried = True

                # Local 5xx HTTPException can be transient (slot warm-up,
                # brief concurrency spike right after restart). Retry local a
                # few times before falling back to remote providers.
                if provider_type == "local" and local_slot_retry_attempts > 0:
                    retry_exc: Exception | None = exc
                    resolved_response: Response | None = None
                    for retry_idx in range(1, local_slot_retry_attempts + 1):
                        if local_slot_retry_delay_seconds > 0:
                            await asyncio.sleep(local_slot_retry_delay_seconds)
                        try:
                            retry_response = await ptr_local(request, path)
                        except Exception as inner_exc:
                            retry_exc = inner_exc
                            _record_attempt(
                                attempts,
                                provider=provider_name,
                                type=provider_type,
                                status="local_http_exception_retry",
                                retry_attempt=retry_idx,
                                error=str(type(inner_exc).__name__),
                            )
                            continue
                        retry_exc = None
                        resolved_response = retry_response
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="local_http_exception_retry",
                            retry_attempt=retry_idx,
                            status_code=int(getattr(retry_response, "status_code", 0) or 0),
                        )
                        break

                    if retry_exc is None and resolved_response is not None:
                        # Local retry succeeded; re-enter success-path checks by
                        # re-processing the resolved response.
                        response = resolved_response
                        body_text = _response_body_text(response)
                        slot_info = _parse_slot_exhaustion(response)
                        if slot_info is None and not _is_http_error_status(response.status_code):
                            # Success — record and return below via normal path.
                            result = _build_fallback_success_response(
                                response, provider_name, provider_type, attempts,
                                prev_provider, fallback_reason, path, body_text,
                                status_override="success_after_http_exception_retry",
                            )
                            # LP-0MR4ZIGDT004A3E1: Surface resolved provider/model
                            _add_resolved_model_header(result, provider_cfg)
                            return result
                        # Retry produced a response but still slot-exhaustion/error;
                        # fall through to normal handling by continuing the loop.
                        continue

                if _first_error_response is None:
                    # Capture the first HTTPException as an error response so
                    # the actual error (e.g. concurrency limit, slot queue) is
                    # preserved instead of replaced by the generic exhausted message.
                    _first_error_response = Response(
                        content=json.dumps({
                            "error": {
                                "type": "backend_error",
                                "code": "backend_error",
                                "message": str(exc.detail),
                            },
                            "status": exc.status_code,
                        }).encode("utf-8"),
                        status_code=exc.status_code,
                        media_type="application/json",
                    )
                mark_provider_unavailable(provider_name, cooldown_seconds)
                attempted_domains.add(_failure_domain_key(provider_cfg))
                fallback_reason = f"HTTPException {exc.status_code}"
                prev_provider = provider_name
                _record_attempt(
                    attempts,
                    provider=provider_name,
                    type=provider_type,
                    status="http_exception",
                    status_code=exc.status_code,
                    error=str(exc),
                )
                all_slot_exhaustion = False
                continue
            # Non-connection error — propagate
            raise

    # All providers exhausted — log diagnostic details
    unavailable = _log_exhausted_providers(model_config, path)

    # Record time-window skips as distinct diagnostics so operators can tell
    # whether providers were excluded by their available_times windows rather
    # than by cooldown/errors (LP-0MS4ETBNO0022QAC).
    for skipped in _providers_outside_window(model_config):
        _record_attempt(
            attempts,
            provider=skipped["name"],
            type=skipped["type"],
            status="outside_time_window",
        )

    # Distinguishable exhaustion: when time windows are the *only* reason no
    # provider could be used, surface a specific message instead of the generic
    # "All providers exhausted" (LP-0MS4ETBNO0022QAC).
    time_window_exhausted = _build_time_window_exhausted_response(
        attempts, unavailable, any_provider_tried,
    )
    if time_window_exhausted is not None:
        raise ChainExhaustedError(time_window_exhausted)

    if not any_provider_tried:
        raise ChainExhaustedError(
            _build_exhausted_response(all_local_slot_exhaustion=False, unavailable_providers=unavailable, diagnostics=attempts)
        )

    # If all failures were slot exhaustion, include total slots in message
    if all_slot_exhaustion:
        raise ChainExhaustedError(
            _build_exhausted_response(all_local_slot_exhaustion=True, total_slots=total_slots_sum, unavailable_providers=unavailable, diagnostics=attempts)
        )

    # When all providers are exhausted, return the first provider's actual
    # error response instead of the generic "All providers exhausted"
    # message.  This preserves the real error (e.g. backend_unavailable,
    # concurrency limit, slot exhaustion, backend error) that the client
    # would have received from a single-provider model or direct call.
    #
    # Exception: if the first error is local lease-active contention and the
    # model has remote providers, do not return that local routing signal to
    # clients; prefer generic exhausted/remote error semantics.
    if _first_error_response is not None:
        has_remote_provider = any(
            isinstance(p, dict) and p.get("type") == "remote"
            for p in (model_config.get("providers") or [])
        )
        if has_remote_provider and _is_local_lease_active_response(_first_error_response):
            logger.info(
                "Suppressing local_lease_active first error response for model=%s; "
                "remote fallback chain present",
                path,
            )
        elif _is_reasoning_content_roundtrip_error(_first_error_response):
            # AC3 (LP-0MSGU3JNU0092AFQ): never surface the raw upstream
            # reasoning_content round-trip 400 to the client; return a
            # synthetic error with remediation guidance instead.
            logger.warning(
                "Intercepting reasoning_content round-trip 400 for model=%s; "
                "returning synthetic error instead of raw upstream body",
                path,
            )
            raise ChainExhaustedError(_build_reasoning_content_roundtrip_error())
        else:
            logger.info(
                "Returning first provider error response instead of generic exhausted "
                "message for model=%s (status=%s)",
                path, _first_error_response.status_code,
            )
            raise ChainExhaustedError(_first_error_response)

    raise ChainExhaustedError(
        _build_exhausted_response(all_local_slot_exhaustion=False, unavailable_providers=unavailable, diagnostics=attempts)
    )


async def proxy_with_fallback(
    request,
    path: str,
    model_config: dict,
    config: dict,
) -> Response:
    """Proxy a request with fallback across both local and remote providers,
    with chain-hold retry semantics (LP-0MSH94Z7K007VKC9).

    Runs the provider chain; when every provider in the fallback chain is
    exhausted (final model unavailable), the request is HELD for
    ``server.chain_hold_seconds`` (default 300) and a NEW cycle starts from
    the FIRST provider — giving short cooldowns time to expire instead of
    erroring immediately. Streaming requests receive periodic SSE feedback
    comments (``: chain exhausted (...); retrying from <first> in <Ns>``);
    non-streaming requests are held silently. The number of hold-retry cycles
    is bounded by ``server.chain_hold_max_cycles`` (default 3; 0 = infinite);
    after the bound the exhaustion response is returned unchanged. A client
    disconnect aborts the hold promptly.

    Args:
        request: The incoming FastAPI Request.
        path: The API path to proxy.
        model_config: Model configuration dict with a ``providers`` list.
        config: Server configuration dict.

    Returns:
        A ``Response`` from a successful provider, or the 503/429 exhaustion
        response once the hold-retry bound is reached.
    """
    return await _run_chain_cycles(
        request, path, model_config, config, _proxy_with_fallback_cycle,
    )
