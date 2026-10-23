
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
- Availability persistence: provider cooldowns and usage-limit account
  quarantine survive proxy restarts via `proxy/provider-state.json`
  (LP-0MUI6KB67005X44B)
"""

import asyncio
import inspect
import json
import logging
import math
import os
import re
import tempfile
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse

from proxy import cold_start, quota_pacing, quota_state
from proxy.observability import _record_backend_signal
from proxy.priority import (
    PRIORITY_HIGH,
    PRIORITY_LOW,
    effective_chain_for_request,
    priority_routing_profile,
    request_priority,
)
from proxy.provider_budget import (
    RATE_LIMIT_WINDOW_SECONDS,
    acquire_provider_budget,
    affinity_ttl_seconds,
    record_provider_affinity,
)
from proxy.utils import _is_empty_response

logger = logging.getLogger("llama-proxy.provider")


def _maybe_record_provider_affinity(
    provider_cfg: dict,
    session_id: str | None,
    response,
    config: dict | None = None,
    model_config: dict | None = None,
) -> None:
    """Record session -> gated-entry affinity after a successful dispatch.

    The free-tier budget gate (LP-0MVDQ1GMO00801PL) only lets a session *wait*
    for capacity if that session was already served by the entry, so the record
    must reflect a real success: a 4xx/5xx response (e.g. the upstream 429 that
    proves the allowance is spent) must never create affinity. Entries without
    ``rate_limit_rpm`` are not gated and record nothing.

    The TTL resolves through the same precedence as the gate itself
    (entry ``budget`` > ``models.<name>.budget`` > ``server.*`` > default,
    LP-0MVDVK5TM004TQQV), so the recorded affinity lifetime always matches the
    model whose request created it.
    """
    try:
        if int(getattr(response, "status_code", 0) or 0) >= 400:
            return
        if int(provider_cfg.get("rate_limit_rpm", 0) or 0) <= 0:
            return
        record_provider_affinity(
            session_id,
            str(provider_cfg.get("name") or ""),
            config,
            ttl_seconds=affinity_ttl_seconds(
                config, model_config=model_config, provider_cfg=provider_cfg
            ),
        )
    except Exception:  # pragma: no cover - affinity is best-effort telemetry
        pass


def _budget_diagnostics(decision) -> dict[str, str]:
    """Budget knob provenance for an attempt diagnostic entry.

    Returns ``{"budget_source": "reserve_ratio@model"}`` when a
    ``models.<name>.budget`` or entry ``budget`` block supplied any effective
    knob, and ``{}`` when the server config / built-in defaults decided — so a
    config without ``budget`` blocks records exactly the same diagnostics as
    before (LP-0MVDVK5TM004TQQV AC3/AC6).
    """
    label = getattr(decision, "source_label", "")
    return {"budget_source": label} if label else {}

# ---------------------------------------------------------------------------
# Cooldown / circuit-breaker state
# ---------------------------------------------------------------------------

# In-memory cooldown tracking: provider_name -> expiry_timestamp (seconds since epoch)
_provider_unavailable_until: dict[str, float] = {}

# Consecutive failure count for exponential backoff: provider_name -> count
# Incremented on each failure, reset to 0 on success.
_provider_failure_count: dict[str, int] = {}

# ---------------------------------------------------------------------------
# Sibling-fallback circuit breaker (LP-0MTPMF03P0046MFG)
#
# Tracks consecutive empty_response/stall failures per provider across
# retry cycles. When the count exceeds the configured threshold within
# the sliding window, the provider is marked unavailable for an extended
# cooldown so the retry cycle permanently skips it and tries siblings.
#
# State is in-memory (no persistence), consistent with the existing
# cooldown mechanism.
# ---------------------------------------------------------------------------

# Consecutive empty/stall failure count: provider_name -> count
_sibling_failure_count: dict[str, int] = {}

# Timestamp (monotonic) when the current consecutive-failure streak began
# for a provider. Reset when the streak clears.
_sibling_failure_streak_start: dict[str, float] = {}

# Exponential backoff constants (remote providers only)
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 45.0

# Empty-response cooldown cap (LP-0MTVPJBF0001SQF4): empty responses are
# typically transient gateway quirks, not hard endpoint failures, so cap
# their effective cooldown at a short bound (default 10s). Retry-After
# headers still override upward.
_EMPTY_RESPONSE_MAX_COOLDOWN_SECONDS = 10.0

# FreeUsageLimitError cooldown: 3 hours (10800 seconds) by default.
# Applied when upstream returns HTTP 429 with error.type = "FreeUsageLimitError"
# See LP-0MRGU0I91006ODFD for details. The per-provider overrides
# (LP-0MSMCM5UG00378G8) were retired when the last free-tier provider
# was removed from all chains (LP-0MT652JRM004ZLSI).
_FREE_USAGE_LIMIT_COOLDOWN_SECONDS = 10800

# Usage-limit reset tracking (LP-0MSLJPOCC0001ROJ): failure-domain key ->
# absolute epoch timestamp when the usage limit resets (including the
# 2-minute safety margin). Providers in a domain with a pending usage reset
# are skipped by every routing decision until the reset time passes.
_usage_reset_at: dict[str, float] = {}

# Accounts whose quarantine duration was GUESSED from ``metadata.limitName``
# alone — the upstream 429 carried no explicit "Resets in ..." duration. A
# period name is a weak signal (all observed opencode 429s are duration-less),
# so guessed quarantines are (a) capped to a short window and (b) never
# persisted: an account that recovered must not stay locked out for weeks
# after a restart (LP-0MUQTCMW2001VXH8).
_usage_reset_guessed: set[str] = set()

# 2-minute safety margin added to the computed usage-limit reset time so a
# clock-skewed upstream does not start re-serving 429s the moment the limit
# nominally resets.
_USAGE_LIMIT_RESET_MARGIN_SECONDS = 120

# Fallback durations per metadata.limitName when the upstream message carries
# no explicit "Resets in ..." duration (daily/weekly/monthly periods). These
# are the FALLBACK-ONLY guess for an ``opencode-*`` account whose fresh quota
# snapshot is unavailable (stale/absent/disabled): with a fresh snapshot the
# authoritative ``/v1/usage`` ``seconds_to_reset`` wins (LP-0MVF7MPZN009GOJB).
_PERIOD_DEFAULT_SECONDS = {
    "daily": 24 * 3600,
    "weekly": 7 * 24 * 3600,
    "monthly": 30 * 24 * 3600,
}

# Upper bound applied to a GUESSED (period-name-only) quarantine. The full
# period (up to 30 days) is never used without an explicit duration from the
# gateway; a guess is capped at this short window. Override with
# ``server.usage_limit_guess_cap_seconds`` or the env var below
# (LP-0MUQTCMW2001VXH8).
_USAGE_LIMIT_GUESS_MAX_SECONDS = 24 * 3600
_USAGE_LIMIT_GUESS_CAP_ENV = "LLAMA_PROXY_USAGE_LIMIT_GUESS_CAP_SECONDS"

# Self-healing probe: re-validate a guessed quarantine at most once per this
# interval per account, so a recovered account is reintroduced promptly
# without hammering the upstream on every request.
_USAGE_LIMIT_PROBE_INTERVAL_SECONDS = 300.0
_usage_limit_last_probe: dict[str, float] = {}

# Per-probe request timeout (seconds). A probe must never hang a request.
_USAGE_LIMIT_PROBE_TIMEOUT_SECONDS = 15.0

# Injectable probe transport: an async ``(provider_cfg) -> bool`` returning
# True when the account is usable again. Left ``None`` in production (the
# httpx default is used); tests replace it to avoid real network calls.
_usage_limit_probe_transport: Callable[[dict], Awaitable[bool]] | None = None

# ---------------------------------------------------------------------------
# Provider availability persistence (LP-0MUI6KB67005X44B)
#
# Provider availability is process state: upstream cooldowns
# (_provider_unavailable_until) and usage-limit account quarantine
# (_usage_reset_at) previously lived only in memory, so every proxy restart
# wiped them and re-exposed an already-exhausted upstream. Both maps are
# persisted to a small versioned JSON document beside proxy/.mode and
# restored on startup.
#
# Scope boundary: _provider_failure_count (the exponential-backoff counter)
# and the sibling-failure / local-400 streak state are deliberately NOT
# persisted — backoff restarts at its base interval after a restart and those
# counters are cheap to re-learn.
#
# The write path is atomic (temp file + os.replace) and runs only on the cold
# path (a provider failing / an upstream 429). The read path is
# corruption-tolerant: any error leaves the maps empty and never blocks
# startup.
# ---------------------------------------------------------------------------

# On-disk schema version for the persisted provider-availability document.
_PROVIDER_STATE_VERSION = 1

# Environment override for the state-file path (used by tests and non-default
# deployments). When unset, the file lives beside proxy/.mode.
_PROVIDER_STATE_FILE_ENV = "LLAMA_PROXY_PROVIDER_STATE_FILE"

# Top-level keys of the persisted document.
_PROVIDER_STATE_COOLDOWNS_KEY = "provider_unavailable_until"
_PROVIDER_STATE_QUARANTINE_KEY = "usage_reset_at"


def default_provider_state_file() -> Path:
    """Return the persisted provider-availability state-file path.

    Defaults to ``proxy/provider-state.json`` (beside ``proxy/.mode`` and
    ``proxy/grandfathering-state.json``). Override with the
    ``LLAMA_PROXY_PROVIDER_STATE_FILE`` environment variable.
    """
    override = os.environ.get(_PROVIDER_STATE_FILE_ENV)
    if override:
        return Path(override)
    return Path(__file__).parent.parent / "provider-state.json"


def _prune_expired_expiries(
    mapping: dict[str, float], now: float
) -> dict[str, float]:
    """Return a copy of *mapping* without entries expired at *now*."""
    return {key: expiry for key, expiry in mapping.items() if expiry > now}


def save_provider_state(path: str | Path | None = None) -> None:
    """Persist provider cooldowns and usage-limit quarantine atomically.

    Serialises both availability maps to a versioned JSON document and writes
    it via a temp file in the target directory plus ``os.replace`` so a
    concurrent reader never observes a partial file. Already-expired entries
    are excluded from the payload so the file does not grow with stale keys.

    The write is synchronous: callers invoke it on the cold path (a provider
    failing or an upstream 429), where a small atomic write is inexpensive.
    A failure is raised to the caller, which is expected to persist
    best-effort without letting the error reach the routing path.

    Args:
        path: Target state-file path (defaults to
            :func:`default_provider_state_file`).
    """
    target = Path(path) if path else default_provider_state_file()
    now = time.time()
    # Guessed (period-name-only) quarantines are deliberately NOT persisted:
    # they are weak evidence and must not outlive a restart as a long-lived
    # lockout. Only explicit-duration quarantines are written
    # (LP-0MUQTCMW2001VXH8).
    persistable_quarantine = {
        key: expiry
        for key, expiry in _usage_reset_at.items()
        if key not in _usage_reset_guessed
    }
    payload = {
        "version": _PROVIDER_STATE_VERSION,
        _PROVIDER_STATE_COOLDOWNS_KEY: _prune_expired_expiries(
            _provider_unavailable_until, now
        ),
        _PROVIDER_STATE_QUARANTINE_KEY: _prune_expired_expiries(
            persistable_quarantine, now
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _coerce_expiry_map(raw: Any) -> dict[str, float]:
    """Coerce a persisted mapping to ``{str: float}``, dropping bad entries.

    A non-dict payload, a non-string key, or a non-numeric expiry is skipped
    rather than raising, so a single malformed entry never discards the rest
    of the map.
    """
    if not isinstance(raw, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        result[key] = float(value)
    return result


def load_provider_state(path: str | Path | None = None) -> tuple[int, int]:
    """Restore provider cooldowns and usage-limit quarantine from disk.

    Reads the state file written by :func:`save_provider_state`, drops any
    entry whose absolute epoch expiry is at or before ``time.time()``, and
    installs the remainder into the module-level maps. Both maps are replaced
    (not merged) so a reload is deterministic.

    Corruption tolerance: a missing, unreadable, malformed, or non-dict file
    leaves the maps empty and logs a warning; a malformed entry is skipped
    while valid siblings survive. This function never raises.

    Args:
        path: Source state-file path (defaults to
            :func:`default_provider_state_file`).

    Returns:
        ``(restored_cooldowns, restored_quarantine)`` — the number of entries
        restored into each map.
    """
    target = Path(path) if path else default_provider_state_file()
    now = time.time()
    cooldowns: dict[str, float] = {}
    quarantine: dict[str, float] = {}
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning(
            "provider-state: no state file at %s; starting with empty "
            "cooldown/quarantine state",
            target,
        )
        raw = None
    except (OSError, ValueError) as exc:
        logger.warning(
            "provider-state: ignoring unreadable state file %s: %s",
            target,
            exc,
        )
        raw = None
    else:
        if not isinstance(raw, dict):
            logger.warning(
                "provider-state: ignoring malformed state file %s "
                "(top-level payload is not an object)",
                target,
            )
            raw = None

    if isinstance(raw, dict):
        cooldowns = _prune_expired_expiries(
            _coerce_expiry_map(raw.get(_PROVIDER_STATE_COOLDOWNS_KEY)), now
        )
        quarantine = _prune_expired_expiries(
            _coerce_expiry_map(raw.get(_PROVIDER_STATE_QUARANTINE_KEY)), now
        )

    _provider_unavailable_until.clear()
    _provider_unavailable_until.update(cooldowns)
    _usage_reset_at.clear()
    _usage_reset_at.update(quarantine)
    # Loaded entries are always explicit/hard (guessed quarantines are never
    # persisted), so no stale guessed markers survive a reload.
    _usage_reset_guessed.clear()
    _usage_limit_last_probe.clear()

    logger.info(
        "provider-state: restored %d cooldown entries and %d usage-limit "
        "quarantine entries from %s",
        len(cooldowns),
        len(quarantine),
        target,
    )
    return len(cooldowns), len(quarantine)


def _persist_provider_state_best_effort() -> None:
    """Persist the availability maps, swallowing (but logging) any failure.

    Called on the cold path — a provider failing or an upstream 429. The write
    must never raise into the routing/fallback path: a failed persist only
    means the cooldown/quarantine is lost on the next restart, which is no
    worse than the pre-persistence behaviour.
    """
    try:
        save_provider_state()
    except Exception:
        logger.warning(
            "provider-state: failed to persist availability state",
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Default sibling-fallback constants
# ---------------------------------------------------------------------------

# Default number of consecutive empty/stall failures before triggering
# an extended cooldown that forces the retry cycle to skip this provider.
_DEFAULT_SIBLING_FALLBACK_THRESHOLD = 2

# Default extended cooldown duration (seconds) applied when the sibling-
# fallback threshold is exceeded. This is much longer than the standard
# provider cooldown so the retry cycle permanently skips the failing
# provider and tries siblings instead of repeatedly hitting the same one.
_DEFAULT_SIBLING_FALLBACK_COOLDOWN_SECONDS = 600  # 10 minutes

# Sliding window (seconds) for the consecutive-failure streak. If a
# provider has no failures for longer than this window, the streak
# is considered stale and resets.
_DEFAULT_SIBLING_FALLBACK_WINDOW_SECONDS = 600  # 10 minutes

# ---------------------------------------------------------------------------
# Deterministic local HTTP 400 escalation (LP-0MTXEBQ4E001BMI2)
#
# A local 4xx is treated as a request-shape incompatibility: same-request
# fallback to the next provider WITHOUT local cooldown (transient one-off
# noise must stay cheap — AC4). But a 400 that repeats turn after turn for
# the SAME session on the SAME history is deterministic — compacted history
# that llama-server rejects every time — and would otherwise route remote
# silently forever. These streaks escalate to a visible WARNING + dedicated
# metric once they cross the threshold.
#
# State is in-memory (no persistence), consistent with the sibling-failure
# and cooldown mechanisms, and bounded by a stale-entry prune.
# ---------------------------------------------------------------------------

# Consecutive local-400 count per (session_id, provider): the number of
# consecutive same-session local 400s within the sliding window.
_local_http_400_streaks: dict[tuple[str, str], dict[str, float | int]] = {}

# Repeat threshold: the 2nd same-session local 400 within the window is
# already 'turn after turn' — escalate.
_DETERMINISTIC_LOCAL_400_THRESHOLD = 2

# Sliding window (seconds) for the local-400 streak before it resets.
_DETERMINISTIC_LOCAL_400_WINDOW_SECONDS = 600.0  # 10 minutes

# Upper bound on tracked (session, provider) streak entries before pruning
# stale ones (avoids unbounded growth on many short sessions).
_MAX_LOCAL_HTTP_400_STREAK_ENTRIES = 1000

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

# ---------------------------------------------------------------------------
# Routing-hop counter (LP-0MVI2VRT2000V1AP AC5)
# ---------------------------------------------------------------------------
# Per-session count of provider *transitions* (local↔remote, remote↔remote).
# One routing-thrash case (LP-0MVI2VRT2000V1AP) alternated local↔remote seven
# times in twelve minutes; without a counter that is only visible by reading
# raw log lines by hand. ``note_routing_hop`` returns the session's new hop
# count (0 = same provider as last time) and emits a ``routing_hop`` log line
# only when the provider actually changes, so the count is a direct aggregate.
#
# ``_last_routing_hop``: (model, session) -> (last_provider, hops).
_last_routing_hop: dict[tuple[str, str], tuple[str, int]] = {}
_MAX_ROUTING_HOP_ENTRIES = 1000


def note_routing_hop(
    session_id: str | None,
    provider_name: str | None,
    *,
    model_name: str | None = None,
    **context,
) -> int:
    """Record the provider a request was served by; return its hop count.

    Returns the number of provider transitions observed for the session so
    far (0 for the first sighting). A ``routing_hop`` log line is emitted only
    when the provider changes, carrying ``from``/``to``/``hops`` plus any
    keyword *context* (e.g. ``reason=``), so the usage analysis can count
    thrash per session without re-deriving it from stream lines.
    """
    if not session_id or not provider_name:
        return 0
    key = (str(model_name or ""), session_id)
    previous, hops = _last_routing_hop.get(key, ("", 0))
    if previous == provider_name:
        return hops
    hops = hops + 1 if previous else 0
    if key not in _last_routing_hop and len(_last_routing_hop) >= _MAX_ROUTING_HOP_ENTRIES:
        try:
            _last_routing_hop.pop(next(iter(_last_routing_hop)), None)
        except StopIteration:
            pass
    _last_routing_hop[key] = (provider_name, hops)
    try:
        extra = " ".join(f"{k}={v}" for k, v in context.items() if v is not None)
        logger.info(
            "routing_hop session=%s from=%s to=%s hops=%d%s",
            session_id,
            previous or "none",
            provider_name,
            hops,
            f" {extra}" if extra else "",
        )
    except Exception:  # pragma: no cover - defensive
        pass
    return hops


# Per-(model, session) cached-tokens ratio: (model_name, session_id) -> ratio.
# The ratio is cached_tokens / prompt_tokens from the last local response.
# A ratio of 1.0 means ALL tokens were cached (fully warm).
# A ratio < 1 means at least some tokens needed recomputation (cache partially
# or fully cold).
# When no entry exists, defaults to 0.0 (cold - conservative).
_last_cached_ratio: dict[tuple[str, str], float] = {}

# Maximum number of entries in _last_cached_ratio to prevent unbounded growth.
_MAX_CACHED_RATIO_ENTRIES = 1000

# Where each ratio came from — recorded alongside the ratio so the proxy can
# label restored-prefix wins separately from genuine prompt-cache hits
# (LP-0MVGD9YXH008O304 AC4).  Values: ``"usage"`` (measured on a previous local
# response), ``"save"`` (slot snapshot saved, no usage data), ``"restored"``
# (seeded from a persisted slot snapshot — see ``proxy/slot_cache_ratio.py``).
_CACHED_RATIO_SOURCE_USAGE = "usage"
_CACHED_RATIO_SOURCE_SAVE = "save"
_cached_ratio_source: dict[tuple[str, str], str] = {}


def set_cached_ratio(
    model_name: str,
    session_id: str | None,
    ratio: float,
    source: str = _CACHED_RATIO_SOURCE_USAGE,
) -> None:
    """Record a pre-computed cached-tokens ratio for a (model, session) pair.

    The ratio is clamped to ``[0.0, 1.0]``; the *source* label is stored so
    observability can distinguish a restored slot-prefix credit
    (``"restored"``) from a measured prompt-cache hit (``"usage"``).

    To prevent unbounded memory growth, both dicts are capped at
    ``_MAX_CACHED_RATIO_ENTRIES``. When the cap is reached, the oldest entry
    (first inserted) is evicted.
    """
    if not model_name or not session_id:
        return
    try:
        value = float(ratio)
    except (TypeError, ValueError):
        return
    if math.isnan(value):
        return
    value = min(1.0, max(0.0, value))
    key = (model_name, session_id)
    # Enforce cap: if adding a new entry would exceed the limit, evict oldest
    if key not in _last_cached_ratio and len(_last_cached_ratio) >= _MAX_CACHED_RATIO_ENTRIES:
        try:
            # Evict oldest entry (dict maintains insertion order in Python 3.7+)
            oldest = next(iter(_last_cached_ratio))
            _last_cached_ratio.pop(oldest)
            _cached_ratio_source.pop(oldest, None)
        except (StopIteration, KeyError):
            pass
    _last_cached_ratio[key] = value
    _cached_ratio_source[key] = source or _CACHED_RATIO_SOURCE_USAGE


def update_cached_ratio(
    model_name: str,
    session_id: str | None,
    cached_tokens: int,
    prompt_tokens: int,
    source: str = _CACHED_RATIO_SOURCE_USAGE,
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
    set_cached_ratio(model_name, session_id, ratio, source=source)


def _get_cached_ratio(model_name: str, session_id: str | None) -> float:
    """Get the cached-tokens ratio for a (model, session) pair.

    Returns 0.0 (cold) when no entry exists (conservative default).
    """
    if not model_name or not session_id:
        return 0.0
    return _last_cached_ratio.get((model_name, session_id), 0.0)


def _get_cached_ratio_source(model_name: str, session_id: str | None) -> str | None:
    """Return the provenance label for a (model, session) ratio.

    ``None`` when no ratio has been recorded yet (unknown → treated as cold).
    """
    if not model_name or not session_id:
        return None
    return _cached_ratio_source.get((model_name, session_id))


# ---------------------------------------------------------------------------
# Expected-reuse credit from the previous prompt (LP-0MVI2XOTY008Q4UV)
# ---------------------------------------------------------------------------
#
# The measured ratio above describes the *previous* prompt's hit fraction:
# a session's first local turn is genuinely cold (nothing cached), so it
# records ``cached_ratio=0.0`` and every later request inherits that zero
# until a later response happens to report a hit.  Turn 2 of a growing
# session was therefore always costed as fully cache-cold even though turn
# 1's tokens are resident in the KV slot.
#
# The prefix record closes that gap: after each successful local dispatch
# (measured prompt_tokens from usage), we store the prompt size together
# with a fingerprint of the message list.  At routing time, if the recorded
# message list is a strict prefix of the incoming one (proven by
# ``slot_cache_ratio.prefix_matches`` — the same check the persisted
# snapshot sidecar uses), the expected reusable prefix is the recorded
# prompt size, safety-margined so we never over-credit a restored prefix
# that llama.cpp may have had to erase (hybrid-arch checkpoint invalidation).
# The measured ratio remains a floor: a warm measured ratio always wins.

#: Per-(model, session) recorded prompt prefix: (prompt_tokens, fingerprints).
#: Only written when a local response reported real usage prompt_tokens, so
#: today's behaviour is unchanged when no usage data is present.
_last_prompt_prefix: dict[tuple[str, str], tuple[int, list[str]]] = {}

#: Fraction of the recorded prompt credited at routing time (matches the
#: sidecar's safety margin; llama.cpp may invalidate its own checkpoints).
_DEFAULT_PREFIX_CREDIT_MARGIN = 0.9


def record_prompt_prefix(
    model_name: str,
    session_id: str | None,
    prompt_tokens: int,
    messages: list | None,
) -> None:
    """Record the prompt size and message fingerprints of one local dispatch.

    Called from the local streaming loop alongside ``update_cached_ratio``
    (usage path).  *prompt_tokens* is the measured prompt size of the request
    the KV slot now holds, so a later request whose message list extends this
    one can be credited for them.  Never raises: a metric path must not break
    a stream.  Also re-caps the dict like ``set_cached_ratio``.
    """
    if not model_name or not session_id:
        return
    try:
        tokens = int(prompt_tokens or 0)
        if tokens <= 0:
            return
        from proxy.slot_cache_ratio import message_fingerprints

        fingerprints = message_fingerprints(messages)
        if not fingerprints:
            return
    except Exception:
        logger.debug("record_prompt_prefix failed", exc_info=True)
        return
    key = (model_name, session_id)
    if key not in _last_prompt_prefix and len(_last_prompt_prefix) >= _MAX_CACHED_RATIO_ENTRIES:
        try:
            oldest = next(iter(_last_prompt_prefix))
            _last_prompt_prefix.pop(oldest, None)
        except (StopIteration, KeyError):
            pass
    _last_prompt_prefix[key] = (tokens, fingerprints)


def expected_cached_ratio(
    model_name: str | None,
    session_id: str | None,
    estimated_tokens: int,
    messages: list | None,
    server_config: dict | None = None,
) -> float:
    """Routing ratio for *estimated_tokens* of the current prompt.

    Returns the measured ratio plus any prefix credit from the previous
    local dispatch:

    - no recorded prefix, no prefix match, or no estimate → measured ratio
      (today's behaviour — cold 0.0 for a fresh session);
    - a matching recorded prefix → ``max(measured, min(1.0, margin *
      prev_prompt / estimate))`` so the credit is proportionate to the
      incoming prompt and never over-credits.
    """
    measured = _get_cached_ratio(model_name, session_id)
    if not model_name or not session_id:
        return measured
    try:
        estimate = max(0, int(estimated_tokens or 0))
        if estimate <= 0:
            return measured
    except (TypeError, ValueError):
        return measured
    record = _last_prompt_prefix.get((model_name, session_id))
    if record is None:
        return measured
    prev_tokens, fingerprints = record
    try:
        margin = float(
            (server_config or {}).get("cached_prefix_credit_margin", _DEFAULT_PREFIX_CREDIT_MARGIN)
        )
        if not 0.0 <= margin <= 1.0:
            margin = _DEFAULT_PREFIX_CREDIT_MARGIN
    except (TypeError, ValueError):
        margin = _DEFAULT_PREFIX_CREDIT_MARGIN
    try:
        from proxy.slot_cache_ratio import message_fingerprints, prefix_matches

        if not prefix_matches(fingerprints, message_fingerprints(messages)):
            return measured
    except Exception:
        logger.debug("prefix credit check failed", exc_info=True)
        return measured
    credit = min(1.0, margin * min(prev_tokens, estimate) / estimate)
    return max(measured, credit)


def clear_cached_ratio_for_session(session_id: str | None) -> None:
    """Forget every ratio (and its source) recorded for *session_id*.

    Used by session invalidation: the slot snapshot and its sidecar are being
    discarded, so any restored credit for that session is no longer valid.
    """
    if not session_id:
        return
    keys = [k for k in _last_cached_ratio if k[1] == session_id]
    for key in keys:
        _last_cached_ratio.pop(key, None)
        _cached_ratio_source.pop(key, None)
    for key in [k for k in _cached_ratio_source if k[1] == session_id]:
        _cached_ratio_source.pop(key, None)
    # The recorded prompt prefix is equally invalid: the slot snapshot and
    # sidecar are being discarded, so no credit may survive (LP-0MVI2XOTY008Q4UV).
    for key in [k for k in _last_prompt_prefix if k[1] == session_id]:
        _last_prompt_prefix.pop(key, None)
    # A discarded session starts a fresh hop count (LP-0MVI2VRT2000V1AP).
    for key in [k for k in _last_routing_hop if k[1] == session_id]:
        _last_routing_hop.pop(key, None)


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


def _economic_bypass_serves_local(config: dict, priority: str | None = None) -> bool:
    """Whether the economic cold-cache bypass should be recovered locally.

    Priority-gated (AC3, LP-0MVDT94F0003VHUA): the flag replaces the retired
    mode-profile gate (``config-cheap.yaml`` set
    ``local_large_context_economic_bypass_serves_local: true``):

    - ``low``: always True — economic bypasses are served locally (entering
      the contention queue when all slots are busy) instead of falling back
      to a paid remote (cheap-mode semantics in both windows).
    - ``high``: always False — economic bypass routes remote (fast-mode
      semantics).
    - ``medium``: the merged base config flag
      (``server.local_large_context_economic_bypass_serves_local``; the
      merged base omits it, preserving today's remote-fallback behaviour).

    Supports both nested (``server:``) and flat config keys for test
    compatibility.  Default: false.
    """
    profile = (
        priority_routing_profile(
            priority,
            config.get("server", config) if isinstance(config, dict) else None,
        )
        if priority is not None
        else None
    )
    if profile is not None:
        return bool(profile.get("economic_bypass_serves_local"))
    val = config.get("local_large_context_economic_bypass_serves_local")
    if val is None:
        val = config.get("server", {}).get(
            "local_large_context_economic_bypass_serves_local", False
        )
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(val)


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

    The slot count is a single per-mode definition: the active mode-profile
    config's ``session_slot_pool_size`` (default/fast: 3, cheap: 2). The
    time-based slot scheduler was removed (operator-directed simplification
    LP-0MTZRM5HV0007S0V) — there is no schedule-aware live slot count.
    Returns 1 as a safe default when nothing is configured.
    """
    server_cfg = config.get("server", config)
    try:
        val = server_cfg.get("session_slot_pool_size")
        if val is not None and int(val) > 0:
            return int(val)
    except (ValueError, TypeError):
        pass
    return 1


def _get_active_local_ctx_size(config: dict) -> int:
    """Return the currently active local context size.

    Static ``local_model_ctx_size`` from the active mode-profile config; the
    time-based slot scheduler was removed (operator-directed simplification
    LP-0MTZRM5HV0007S0V) so there is no per-period override.
    """
    return _get_local_model_ctx_size(config)


def _get_context_pressure_warn_ratio(config: dict) -> float:
    """Return the unified session-detection ratio.

    LP-0MTVXP7DG00613ZB AC3: ``context_pressure_warn_ratio`` is no longer a
    separate operator knob — the advisory warning fires at the single
    ``compaction_trigger_ratio`` detection threshold. The legacy key is
    ignored; this function remains the single read point for the routing-time
    advisory log.
    """
    try:
        ratio = float(compaction_config(config)["trigger_ratio"])
    except Exception:
        ratio = _DEFAULT_COMPACTION_TRIGGER_RATIO
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
    context-pressure compaction advisory.

    LP-0MTVXP7DG00613ZB AC3: the advisory fires at the SAME threshold as
    prompt-assembly compaction (``compaction_trigger_ratio`` × effective
    per-slot clamp, strict ``>``) — one detection knob drives both the
    advisory log and the live compaction path. Returns False when compaction
    is disabled (trigger 0) or the session is at/below the trigger.

    Args:
        estimated_tokens: Session estimated prompt/context tokens.
        config: Proxy configuration (flat or nested ``server`` dict).

    Returns:
        True when the session should be flagged for compaction.
    """
    if estimated_tokens <= 0:
        return False
    try:
        from proxy.compaction import compaction_trigger_tokens

        trigger = compaction_trigger_tokens("fast", config)
    except Exception:
        return False
    if trigger <= 0:
        return False
    return int(estimated_tokens) > trigger


# ===================================================================
# Compaction config — LP-0MTG6RW3L003X122
# ===================================================================

# Default compaction trigger ratio (0.70 × effective per-slot context).
# LP-0MTVXP7DG00613ZB AC3: this is the SINGLE detection knob — the
# context_pressure advisory, the live compaction path and the 429 gate all
# fire from it (the legacy 0.8 context_pressure_warn_ratio is retired).
_DEFAULT_COMPACTION_TRIGGER_RATIO = 0.70

# Default summarizer context size (tokens). 8192 is enough for a short
# summarisation prompt + middle turns; avoids unnecessary KV cache footprint.
_DEFAULT_SUMMARIZER_CTX_SIZE = 8192

# Default summariser output budget (tokens). 512 tokens of summary text
# is enough for a concise middle-turn condensation without wasting GPU.
_DEFAULT_SUMMARIZER_MAX_TOKENS = 512

# Default HTTP timeout for the local summarizer call to llama-server.
# 600 seconds (10 minutes) — allows the summariser to wait for the local
# slot to become free when it is occupied by a long generating stream.
# This is critical for 1-slot deployments where a single generating request
# can hold the slot for many minutes (observed max dispatch_first_byte_ms
# was 713 s; sessions blocked 17-18 min before the 900 s upstream timeout
# fired).  See LP-0MU1RXEY10075TUU.
# An operator may override via ``server.compaction_summarizer_timeout``.
_DEFAULT_SUMMARIZER_TIMEOUT_SECONDS = 600.0

# Retry policy for transient summarizer failures (R3, LP-0MTTPXIAB0031AC4).
# The summarizer retries connection errors / read timeouts / 5xx / 429 up to
# ``summarizer_retries`` times (default 2) with a fixed inter-attempt delay
# before failing open (returns ""). Other 4xx responses (auth, model-not-
# found, …) are permanent and never retried. Configurable via
# ``server.compaction_summarizer_retries`` and
# ``server.compaction_summarizer_retry_delay_seconds``; 0 disables retries.
_DEFAULT_SUMMARIZER_RETRIES = 2
_DEFAULT_SUMMARIZER_RETRY_DELAY_SECONDS = 0.5

# Reasoning suppression for the compaction summarizer (LP-0MU58PBRD004OV1I).
# Reasoning models (Muse via opencode-go, DeepSeek) share the
# ``summarizer_max_tokens`` output budget with their reasoning tokens. With
# the upstream default effort (Muse: "high") they can spend the entire budget
# on reasoning and return an empty completion — which fails compaction and
# forces remote-only dispatch. "minimal" leaves room for the summary text.
# Configurable via ``server.summarizer_reasoning_effort``; an explicit
# ``null`` disables the override (use the upstream default).
_DEFAULT_SUMMARIZER_REASONING_EFFORT = "minimal"

# Disable Qwen3 "thinking" for the local summarizer via the llama-server
# ``chat_template_kwargs.enable_thinking`` request flag (LP-0MU58PBRD004OV1I).
# Configurable via ``server.summarizer_disable_thinking``.
_DEFAULT_SUMMARIZER_DISABLE_THINKING = True

# Dedicated system prompt for the proxy-side compaction summariser.
#
# Rationale (LP-0MTTSL2AW000A5OG — R1 companion): Pi delivers summarization
# instructions through TWO separate messages — SUMMARIZATION_SYSTEM_PROMPT
# (role + guard rails, dist/core/compaction/utils.js) and SUMMARIZATION_PROMPT
# (the structured format template, dist/core/compaction/compaction.js).
# R1 (LP-0MTTPXHTI0081WIR) merged both into a single system-prompt constant;
# this companion restores Pi's exact split so the model treats the transcript
# as material to summarise, not a conversation to continue:
#   - _SUMMARIZER_SYSTEM_PROMPT — role + "Do NOT continue / Do NOT respond /
#     ONLY output" guard rails (verbatim Pi SUMMARIZATION_SYSTEM_PROMPT).
#   - _SUMMARIZATION_PROMPT — the structured format template (Goal,
#     Constraints, Progress, Decisions, Next Steps, Critical Context),
#     appended AFTER the serialised <conversation> in the summarizer's USER
#     message (see proxy/compaction_summarizer.py:build_local_summarizer).
# Keeping Pi's wording verbatim closes the prompt-quality gap from the
# compaction comparison (docs/session-compaction-pi-comparison.md §5.1) and
# avoids format drift across the local Qwen3 and future remote `compact`
# (Muse → DeepSeek, LP-0MTT0O74N009E7N2) paths. Operators may override this
# system prompt per-installation by dropping a file at
# .sorraAgents/prompts/compaction-summarizer.txt (see
# proxy/compaction_summarizer.py:_resolve_system_prompt_override).
_SUMMARIZER_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a "
    "conversation between a user and an AI coding assistant, then produce "
    "a structured summary following the exact format specified.\n"
    "\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the "
    "conversation. ONLY output the structured summary."
)

# Structured summarization format template — verbatim Pi SUMMARIZATION_PROMPT
# (dist/core/compaction/compaction.js). Delivered in the summarizer's USER
# message directly after the serialised <conversation> so the model follows
# the exact checkpoint format below for the condensed middle turns. File
# operations (<read-files>/<modified-files>) are appended to the OUTPUT by
# the file-ops tracker (R2, LP-0MTTPXI1Y003YFOU), matching Pi's
# utils.js:formatFileOperations — they are not part of the template.
_SUMMARIZATION_PROMPT = (
    "The messages above are a conversation to summarize. Create a structured "
    "context checkpoint summary that another LLM will use to continue the work.\n"
    "\n"
    "Use this EXACT format:\n"
    "\n"
    "## Goal\n"
    "[What is the user trying to accomplish? Can be multiple items if the "
    "session covers different tasks.]\n"
    "\n"
    "## Constraints & Preferences\n"
    "- [Any constraints, preferences, or requirements mentioned by user]\n"
    "- [Or \"(none)\" if none were mentioned]\n"
    "\n"
    "## Progress\n"
    "### Done\n"
    "- [x] [Completed tasks/changes]\n"
    "\n"
    "### In Progress\n"
    "- [ ] [Current work]\n"
    "\n"
    "### Blocked\n"
    "- [Issues preventing progress, if any]\n"
    "\n"
    "## Key Decisions\n"
    "- **[Decision]**: [Brief rationale]\n"
    "\n"
    "## Next Steps\n"
    "1. [Ordered list of what should happen next]\n"
    "\n"
    "## Critical Context\n"
    "- [Any data, examples, or references needed to continue]\n"
    "- [Or \"(none)\" if not applicable]\n"
    "\n"
    "Keep each section concise. Preserve exact file paths, function names, "
    "and error messages."
)

# Incremental summarization template — verbatim Pi UPDATE_SUMMARIZATION_PROMPT
# (dist/core/compaction/compaction.js). Used (R5, LP-0MTTPXIIX005Y0Z9) when the
# session already carries a compaction summary: the summarizer merges the NEW
# middle turns into the existing summary rather than regenerating from scratch,
# preserving all prior sections and updating only what changed. The existing
# summary is passed in the ``<previous-summary>`` block that precedes this
# template in the USER message (see proxy/compaction_summarizer.py).
_UPDATE_SUMMARIZATION_PROMPT = (
    "The messages above are NEW conversation messages to incorporate into the "
    "existing summary provided in <previous-summary> tags.\n"
    "\n"
    "Update the existing structured summary with new information. RULES:\n"
    "- PRESERVE all existing information from the previous summary\n"
    "- ADD new progress, decisions, and context from the new messages\n"
    "- UPDATE the Progress section: move items from \"In Progress\" to \"Done\" "
    "when completed\n"
    "- UPDATE \"Next Steps\" based on what was accomplished\n"
    "- PRESERVE exact file paths, function names, and error messages\n"
    "- If something is no longer relevant, you may remove it\n"
    "\n"
    "Use this EXACT format:\n"
    "\n"
    "## Goal\n"
    "[Preserve existing goals, add new ones if the task expanded]\n"
    "\n"
    "## Constraints & Preferences\n"
    "- [Preserve existing, add new ones discovered]\n"
    "\n"
    "## Progress\n"
    "### Done\n"
    "- [x] [Include previously done items AND newly completed items]\n"
    "\n"
    "### In Progress\n"
    "- [ ] [Current work - update based on progress]\n"
    "\n"
    "### Blocked\n"
    "- [Current blockers - remove if resolved]\n"
    "\n"
    "## Key Decisions\n"
    "- **[Decision]**: [Brief rationale] (preserve all previous, add new)\n"
    "\n"
    "## Next Steps\n"
    "1. [Update based on current state]\n"
    "\n"
    "## Critical Context\n"
    "- [Preserve important context, add new if needed]\n"
    "\n"
    "Keep each section concise. Preserve exact file paths, function names, "
    "and error messages."
)


def compaction_config(config: dict) -> dict:
    """Resolve the compaction configuration from the proxy config.

    Reads from the nested ``server.compaction_*`` keys (or flat keys for
    backward compatibility) and applies defaults where absent.

    Args:
        config: Proxy configuration (flat or nested ``server`` dict).

    Returns:
        A dict with keys:
            - trigger_ratio (float): Compaction trigger ratio.
            - summarizer_model_type (str): "local" or "remote".
            - summarizer_model_name (str): Model identifier (e.g. "Qwen3").
            - summarizer_ctx_size (int): Context size for the summariser.
            - summarizer_max_tokens (int): Max output tokens for summary.
            - summarizer_system_prompt (str): Dedicated system prompt.
            - summarizer_retries (int): Retries on transient failures
              (0 disables).
            - summarizer_retry_delay_seconds (float): Inter-attempt delay.
    """
    server = config.get("server", {})

    # Trigger ratio
    trigger_ratio = server.get("compaction_trigger_ratio")
    if trigger_ratio is None:
        trigger_ratio = config.get("compaction_trigger_ratio")
    if trigger_ratio is None:
        trigger_ratio = _DEFAULT_COMPACTION_TRIGGER_RATIO
    try:
        trigger_ratio = float(trigger_ratio)
    except (ValueError, TypeError):
        trigger_ratio = _DEFAULT_COMPACTION_TRIGGER_RATIO

    # Summariser model
    model_cfg = server.get("summarizer_model", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    summarizer_model_type = model_cfg.get("type", "local")
    summarizer_model_name = model_cfg.get("llama_model", "Qwen2.5-7B")

    # Summariser context size
    ctx_size = server.get("summarizer_ctx_size")
    if ctx_size is None:
        ctx_size = _DEFAULT_SUMMARIZER_CTX_SIZE
    try:
        ctx_size = int(ctx_size)
    except (ValueError, TypeError):
        ctx_size = _DEFAULT_SUMMARIZER_CTX_SIZE

    # Summariser max output tokens
    max_tokens = server.get("summarizer_max_tokens")
    if max_tokens is None:
        max_tokens = _DEFAULT_SUMMARIZER_MAX_TOKENS
    try:
        max_tokens = int(max_tokens)
    except (ValueError, TypeError):
        max_tokens = _DEFAULT_SUMMARIZER_MAX_TOKENS

    # Summariser reasoning effort (reasoning models only; explicit null disables)
    if "summarizer_reasoning_effort" in server:
        reasoning_effort = server.get("summarizer_reasoning_effort")
    elif "summarizer_reasoning_effort" in config:
        reasoning_effort = config.get("summarizer_reasoning_effort")
    else:
        reasoning_effort = _DEFAULT_SUMMARIZER_REASONING_EFFORT
    if isinstance(reasoning_effort, str):
        reasoning_effort = reasoning_effort.strip() or None

    # Disable local Qwen3 thinking for the summarizer (chat-template flag)
    disable_thinking = server.get("summarizer_disable_thinking")
    if disable_thinking is None:
        disable_thinking = config.get("summarizer_disable_thinking")
    if not isinstance(disable_thinking, bool):
        disable_thinking = _DEFAULT_SUMMARIZER_DISABLE_THINKING

    # Summarizer retries (transient failures only; 0 disables)
    retries = server.get("compaction_summarizer_retries")
    if retries is None:
        retries = config.get("compaction_summarizer_retries")
    try:
        retries = int(retries)
        retries = max(0, retries)
    except (ValueError, TypeError):
        retries = _DEFAULT_SUMMARIZER_RETRIES

    # Summarizer retry inter-attempt delay (seconds; 0 = no wait)
    retry_delay = server.get("compaction_summarizer_retry_delay_seconds")
    if retry_delay is None:
        retry_delay = config.get("compaction_summarizer_retry_delay_seconds")
    try:
        retry_delay = float(retry_delay)
        retry_delay = max(0.0, retry_delay)
    except (ValueError, TypeError):
        retry_delay = _DEFAULT_SUMMARIZER_RETRY_DELAY_SECONDS

    # Summarizer HTTP timeout (seconds; allows waiting for slot to free)
    timeout = server.get("compaction_summarizer_timeout")
    if timeout is None:
        timeout = config.get("compaction_summarizer_timeout")
    try:
        timeout = float(timeout)
        timeout = max(1.0, timeout)
    except (ValueError, TypeError):
        timeout = _DEFAULT_SUMMARIZER_TIMEOUT_SECONDS

    return {
        "trigger_ratio": trigger_ratio,
        "summarizer_model_type": summarizer_model_type,
        "summarizer_model_name": summarizer_model_name,
        "summarizer_ctx_size": ctx_size,
        "summarizer_max_tokens": max_tokens,
        "summarizer_reasoning_effort": reasoning_effort,
        "summarizer_disable_thinking": disable_thinking,
        "summarizer_system_prompt": _SUMMARIZER_SYSTEM_PROMPT,
        "summarizer_retries": retries,
        "summarizer_retry_delay_seconds": retry_delay,
        "summarizer_timeout_seconds": timeout,
    }


def validate_compaction_config(config: dict) -> list[str]:
    """Validate the compaction configuration at startup.

    Checks that all required fields are present and within valid ranges.
    Returns a list of problem strings (empty when config is valid).

    Problem strings are prefixed with ``FATAL: `` when the server must not
    start, or ``WARNING: `` when the config is questionable but salvageable.

    Args:
        config: Proxy configuration (flat or nested ``server`` dict).

    Returns:
        List of problem strings (empty when config is valid).
    """
    problems: list[str] = []
    cfg = compaction_config(config)

    # trigger_ratio must be in [0, 1]
    tr = cfg["trigger_ratio"]
    if tr < 0 or tr > 1:
        problems.append(
            f"FATAL: compaction_trigger_ratio={tr!r} is outside "
            "valid range [0.0, 1.0]"
        )

    # summarizer_model must have a llama_model name
    model_name = cfg["summarizer_model_name"]
    if not model_name or not isinstance(model_name, str):
        problems.append(
            "FATAL: summarizer_model.llama_model must be a non-empty string"
        )

    # summarizer_ctx_size must be positive
    ctx_size = cfg["summarizer_ctx_size"]
    if ctx_size <= 0:
        problems.append(
            f"FATAL: summarizer_ctx_size={ctx_size} must be positive"
        )

    # summarizer_max_tokens must be positive
    max_tokens = cfg["summarizer_max_tokens"]
    if max_tokens <= 0:
        problems.append(
            f"FATAL: summarizer_max_tokens={max_tokens} must be positive"
        )

    return problems


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


# Default large-context warm economic-ceiling ratio
# (LP-0MU466L1X003RTHN). The ratio scales the warm routing threshold
# linearly with the local model's total context size so a profile change
# (smaller model, different slot count) needs no manual threshold
# recalculation. 0 disables the ratio — the absolute
# ``local_large_context_warm_cache_threshold`` is then the only upper bound.
# The shipped ``config.yaml`` sets 0.3815 (= 100000 / 262144), the
# latency-derived ceiling calibrated to the ~37 tok/s @100K prefill floor.
_DEFAULT_ECONOMIC_CEILING_RATIO = 0.0


def _get_economic_ceiling_ratio(config: dict) -> float:
    """Read the large-context warm economic-ceiling ratio.

    Supports both nested (``server.local_large_context_economic_ceiling_ratio``)
    and flat config keys for production/test compatibility.

    Returns 0.0 when unset/disabled (the absolute
    ``local_large_context_warm_cache_threshold`` is then the only upper
    bound). Negative or non-numeric values are treated as 0.0.
    """
    val = config.get("local_large_context_economic_ceiling_ratio")
    if val is None:
        val = config.get("server", {}).get(
            "local_large_context_economic_ceiling_ratio",
            _DEFAULT_ECONOMIC_CEILING_RATIO,
        )
    try:
        return max(0.0, float(val or 0))
    except (ValueError, TypeError):
        return 0.0


def effective_large_context_economic_ceiling(config: dict) -> int:
    """Compute the latency-derived economic ceiling for the warm threshold.

    The ceiling is ``round(ratio × local_model_ctx_size)`` capped by the
    absolute ``local_large_context_warm_cache_threshold`` when that is > 0.
    It scales with the **model** context size (not the slot count): the
    physical per-slot clamp is the capacity term, this is the economic term
    (LP-0MU466L1X003RTHN).

    Returns 0 (warm cap disabled) when both the ratio and the absolute warm
    override are 0. When the ratio is 0 the absolute value is used verbatim
    (legacy behaviour). When ``local_model_ctx_size`` is 0 (scaling disabled)
    the absolute value is also returned.
    """
    ratio = _get_economic_ceiling_ratio(config)
    warm_override = _get_warm_cache_threshold(config)

    # Both inputs 0 -> warm cap explicitly disabled.
    if ratio <= 0 and warm_override <= 0:
        return 0
    # Ratio disabled or no ctx to scale with -> fall back to the absolute
    # (legacy) warm override.
    if ratio <= 0:
        return warm_override
    ctx_size = _get_local_model_ctx_size(config)
    if ctx_size <= 0:
        return warm_override
    ceiling = round(ratio * ctx_size)
    if warm_override > 0:
        ceiling = min(ceiling, warm_override)
    return ceiling


def warm_cap_ceiling_shortfall(config: dict) -> int:
    """Tokens of a configured warm cap that the economic ceiling shadows.

    ``local_large_context_warm_cache_threshold`` only supplies an UPPER
    bound on the economic ceiling (``round(ratio x local_model_ctx_size)``),
    so raising the absolute value above that ceiling does not move the
    effective warm cap at all — the economic term still binds. That silent
    no-op is what makes the two knobs a co-change (LP-0MVHSET0A008VU95).

    Returns:
        ``configured_absolute - economic_ceiling`` when the economic term is
        the smaller of the two, else 0. Also 0 when either knob is disabled
        (a ratio of 0 uses the absolute verbatim) or when there is no model
        ctx size to scale with — in both cases nothing is shadowed.
    """
    ratio = _get_economic_ceiling_ratio(config)
    warm_override = _get_warm_cache_threshold(config)
    if ratio <= 0 or warm_override <= 0:
        return 0
    ctx_size = _get_local_model_ctx_size(config)
    if ctx_size <= 0:
        return 0
    ceiling = round(ratio * ctx_size)
    return max(0, warm_override - ceiling)


def _effective_large_context_thresholds(config: dict) -> tuple[int, int]:
    """Return (cold, warm) thresholds under the 3-term warm model.

    WARM resolves through the 3-term model (LP-0MU466L1X003RTHN):

        warm = min(physical per-slot clamp, latency-derived economic ceiling)

    - **Physical term** — ``effective_per_slot_threshold(ctx_size, slots)``
      (= ``ctx_size // slots - _LOCAL_ROUTING_OUTPUT_HEADROOM``): a hard
      capacity limit. Prompts above it cannot fit the KV slot and must be
      routed remote to prevent context exhaustion (``finish_reason=length``,
      surfaced by pi as the misleading "maximum output token limit" error,
      LP-0MSAZXXDY005AWA1).
    - **Economic term** — ``effective_large_context_economic_ceiling(config)``:
      ``round(ratio × local_model_ctx_size)`` capped by the absolute
      ``local_large_context_warm_cache_threshold``. A latency/utilisation
      limit that scales with the model context size, so a profile change
      needs no manual threshold recalculation.

    The per-slot clamp dominates when ``slots >= 3``; the economic ceiling
    dominates when ``slots == 1``. The absolute warm config, when set, is
    retained as an upper bound inside the ceiling helper; a ratio of 0 falls
    back to the absolute value, and both 0 disables the warm cap.

    COLD is an **economic new-token threshold** and is NOT clamped or scaled:
    it defines the band (cold, warm] where the cached_ratio routing check
    (Check 2 in ``_should_skip_local``) operates. Clamping cold to the same
    cap as warm collapses the band to zero width, making the ratio check
    unreachable dead code (LP-0MSI2M5BT004BCDP).

    Returns the configured thresholds unchanged when ``local_model_ctx_size``
    is 0 (clamp disabled) or per-slot context cannot be computed: the
    economic ceiling then falls back to the absolute warm override.
    """
    cold = _get_large_context_threshold(config)
    ctx_size = _get_active_local_ctx_size(config)
    slots = _get_active_local_slots(config)

    # WARM: 3-term model (LP-0MU466L1X003RTHN). The economic ceiling replaces
    # the raw warm config as the economic term and already folds in the
    # absolute ``local_large_context_warm_cache_threshold`` as an upper bound.
    warm = effective_large_context_economic_ceiling(config)

    cap = effective_per_slot_threshold(ctx_size, slots)
    if cap <= 0:
        return cold, warm

    # The physical per-slot clamp is a hard capacity limit: warm is capped by
    # it, but only when the warm cap is enabled (> 0). COLD stays as the
    # economic new-token threshold so the (cold, warm] band remains non-empty
    # for Check 2 (cached_ratio routing) to operate in.
    if warm > 0:
        warm = min(warm, cap)
        # LP-0MTBOX45O005LD1S AC4: also clamp warm to the hard-routing cap
        # (the ratio keys were mode-aware; with the mode axis removed
        # LP-0MVDT94F0003VHUA the cap is a single unified ratio) so
        # ``context_too_large`` fires exactly at the hard cap — the SAME
        # single source as ``session_slot_max_prompt_tokens`` persistence.
        # No circularity: ``compute_hard_routing_cap`` derives its denominator
        # from the RAW warm config + per-slot threshold, not this clamped warm.
        # The per-mode ratios are retired (0 → disabled) and the slot pool is
        # fixed at 2, so the unified profile is used unconditionally.
        try:
            _hard_cap = compute_hard_routing_cap("fast", config)
        except Exception:
            _hard_cap = 0
        if _hard_cap > 0:
            warm = min(warm, _hard_cap)
    return cold, warm


def _collect_local_ctx_pairs(config: dict) -> list[tuple[int, int]]:
    """The single (ctx_size, slots) pair the active mode profile runs with.

    Derived from the static ``local_model_ctx_size`` and
    ``session_slot_pool_size`` of the active profile. The time-based slot
    scheduler (whose entries could add per-period pairs) was removed
    (operator-directed simplification LP-0MTZRM5HV0007S0V), so there is
    exactly one pair.
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


# ---------------------------------------------------------------------------
# Hard local-routing caps (LP-0MTBOX45O005LD1S)
# ---------------------------------------------------------------------------
# Per-mode ratio of the effective per-slot context below which local routing
# is allowed.  Ratios are applied against the *effective per-slot threshold*
# (``ctx_size // slots - _LOCAL_ROUTING_OUTPUT_HEADROOM``) so they
# automatically rescale when the mode profile's slot count changes.
#
# Resolved caps (operator-approved 2026-08-27):
#   fast:  70 000 tokens  → ratio 70000/83285 ≈ 0.84049
#   cheap: 61 440 tokens  → ratio 61440/100000 = 0.6144
#
# Denominators are the effective warm/routing clamp (decision #4):
# ``min(warm_config, effective_per_slot_threshold)`` — the SAME value the
# warm-context routing check is clamped to (see
# ``_effective_large_context_thresholds``):
#   fast  schedule (3 slots x 262144 ctx):
#       per_slot = 262144//3 - 4096 = 83285; min(100000, 83285) = 83285
#       → 0.84049 x 83285 = 70000 EXACT
#   cheap schedule (2 slots x 262144 ctx):
#       per_slot = 262144//2 - 4096 = 126976; min(100000, 126976) = 100000
#       → 0.6144 x 100000 = 61440 EXACT
#
# A ratio of 0 (or absent) disables the hard cap.
_DEFAULT_HARD_ROUTING_CAP_RATIO_FAST = 0.0
_DEFAULT_HARD_ROUTING_CAP_RATIO_CHEAP = 0.0


def _get_hard_routing_cap_ratio(config: dict, mode: str) -> float:
    """Read the per-mode hard-routing-cap ratio from config.

    Supports both nested (``server.local_hard_routing_cap_ratio_<mode>``) and
    flat keys.  Mode must be ``"fast"`` or ``"cheap"``.
    A ratio of 0 (or absent) disables the hard cap.

    Args:
        config: Proxy configuration dict.
        mode: Operating mode ("fast" or "cheap").

    Returns:
        The ratio to apply (0.0 = disabled).
    """
    key = f"local_hard_routing_cap_ratio_{mode}"
    val = config.get(key)
    if val is None:
        val = config.get("server", {}).get(key, 0)
    try:
        ratio = float(val or 0)
    except (ValueError, TypeError):
        ratio = 0.0
    return max(0.0, ratio)


def compute_hard_routing_cap(mode: str, config: dict) -> int:
    """Resolve the hard-routing cap for *mode* against the active per-slot context.

    Returns the absolute token count at which local is hard-skipped (fast) or
    gated with a compaction response (cheap).  Computed as:
        ``ratio × effective_warm_clamp``
    rounded to the nearest integer, where ``effective_warm_clamp`` is
    ``min(warm_cache_threshold, effective_per_slot_threshold(ctx_size, slots))``
    — the SAME clamp the warm-context routing threshold resolves to
    (LP-0MTBOX45O005LD1S decision #4): the approved ratios are expressed
    against this clamp so the caps reproduce the approved absolutes (~70000 /
    61440) under the current fast/cheap schedules.  Falls back to the raw
    per-slot threshold when no warm threshold is configured (warm=0 disables
    the clamp).

    Rounding uses ``round()`` (not floor) to absorb float representation error
    (0.6144 × 100000 = 61439.999… in IEEE-754; floor would yield 61439).

    When the slot scheduler is available, the function uses the schedule-aware
    slot count and ctx_size so the cap rescales automatically on schedule
    transitions.  NOTE (cheap boot-static): when the slot scheduler is NOT
    available, the cheap profile falls back to its static pair
    (local_model_ctx_size=131072, 2 slots) → per-slot clamp
    min(100000, 61440)=61440 → 0.6144 × 61440 ≈ 37749, NOT the approved
    61440.  Production always runs the active mode profile's static pair
    (session_slot_pool_size set in the profile, per-slot ctx derived from
    local_model_ctx_size), so the approved absolute holds; the transient
    boot-static value is strictly more conservative and is accepted per the
    item's "caps rescale automatically when the profile's slot count
    changes" caveat.

    Args:
        mode: Operating mode ("fast" or "cheap").
        config: Proxy configuration dict.

    Returns:
        Absolute token cap (0 = disabled).
    """
    ratio = _get_hard_routing_cap_ratio(config, mode)
    if ratio <= 0:
        return 0
    ctx_size = _get_active_local_ctx_size(config)
    slots = _get_active_local_slots(config)
    per_slot = effective_per_slot_threshold(ctx_size, slots)
    if per_slot <= 0:
        return 0
    warm = _get_warm_cache_threshold(config)
    if warm > 0:
        per_slot = min(per_slot, warm)
    if per_slot <= 0:
        return 0
    return round(ratio * per_slot)


def check_hard_routing_cap(
    estimated_tokens: int, mode: str, config: dict
) -> bool:
    """Whether ``estimated_tokens`` exceeds the hard-routing cap for *mode*.

    Args:
        estimated_tokens: Estimated prompt/context tokens for the request.
        mode: Operating mode ("fast" or "cheap").
        config: Proxy configuration dict.

    Returns:
        ``True`` when the estimate is above the cap (local must be skipped).
    """
    cap = compute_hard_routing_cap(mode, config)
    if cap <= 0:
        return False  # cap disabled
    return estimated_tokens > cap


def _build_compaction_gate_response(
    estimated_tokens: int, cap: int, priority: str, session_id: str | None,
    model_name: str | None,
) -> Response:
    """Build a 429 response that gates the request for client-side compaction.

    Used when the hard-routing cap is exceeded by a low-priority request
    (cheap semantics, AC3 — the client compacts instead of spending on a
    paid remote).  The response includes:
    - HTTP 429 status code
    - ``X-Compaction-Gate: true`` header
    - ``X-Compaction-Estimated-Tokens`` header
    - ``X-Compaction-Cap`` header
    - JSON body with ``error`` and compaction guidance

    This is NOT a remote fallback — it tells the client to compact its
    session before retrying.

    Args:
        estimated_tokens: The estimated token count that triggered the gate.
        cap: The hard-routing cap that was exceeded.
        priority: Effective priority ("low") that triggered the gate.
        session_id: Session ID for the request (may be None).
        model_name: Model name for the request.

    Returns:
        A FastAPI ``Response`` to return to the client.
    """
    headers = {
        "Content-Type": "application/json",
        "X-Compaction-Gate": "true",
        "X-Compaction-Estimated-Tokens": str(estimated_tokens),
        "X-Compaction-Cap": str(cap),
        "X-Compaction-Priority": priority,
    }
    if session_id:
        headers["X-Session-Id"] = session_id
    headers["X-Resolved-Model"] = f"local/{model_name}" if model_name else "local/unknown"
    payload = {
        "error": {
            "type": "compaction_gate",
            "code": "context_too_large_for_local",
            "message": (
                f"Context ({estimated_tokens} tokens) exceeds the local "
                f"routing cap ({cap} tokens) at priority {priority!r}. "
                "Compact the session before retrying (see the "
                "context_pressure guidance signal)."
            ),
        },
        "status": 429,
        "estimated_tokens": estimated_tokens,
        "cap": cap,
        "priority": priority,
    }
    return JSONResponse(status_code=429, content=payload, headers=headers)


def _is_compaction_gate_response(response) -> bool:
    """Whether *response* is the low-priority compaction gate (429).

    The gate is a TERMINAL response (LP-0MTBOX45O005LD1S AC2): it must
    reach the client as-is and must never trigger the fallback cycle's
    local-4xx branch, which would silently re-route to the next remote
    provider.  Detected via the ``X-Compaction-Gate`` header emitted by
    :func:`_build_compaction_gate_response`.
    """
    try:
        headers = getattr(response, "headers", {}) or {}
        return str(headers.get("X-Compaction-Gate", "")).lower() == "true"
    except Exception:
        return False


def validate_local_routing_config(config: dict) -> list[str]:
    """Validate the ctx-size / slot-count routing clamp configuration.

    Computes the effective per-slot large-context routing threshold
    (``ctx_size // slots - _LOCAL_ROUTING_OUTPUT_HEADROOM``, mirroring
    ``_effective_large_context_thresholds``) for the (ctx_size, slots) pair
    the active mode profile runs with — the static
    ``local_model_ctx_size``/``session_slot_pool_size`` — and reports a
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
    fatal = bool(
        config.get("server", {}).get("min_local_routing_threshold_fatal", False)
        or config.get("min_local_routing_threshold_fatal", False)
    )

    problems: list[str] = []

    # --- session_slot_pool_size is REQUIRED (LP-0MTCZ35X7009IZKE) ---------
    # The legacy ``local_max_concurrent_queries`` fallback was removed; the
    # slot pool is the single source of truth for both llama-server's
    # --parallel and the local dispatch lease pool. Refuse to start when it
    # is missing or invalid so a stale config cannot silently drop to a
    # 1-slot pool (the original single-slot regression). Unconditional FATAL
    # — independent of min_local_routing_threshold_fatal.
    server_cfg = config.get("server", config)
    raw_pool = server_cfg.get("session_slot_pool_size", None)
    try:
        pool_val = int(raw_pool or 0)
    except (ValueError, TypeError):
        pool_val = 0
    if pool_val <= 0:
        problems.append(
            "FATAL: session_slot_pool_size is required in the server config "
            "(controls llama-server --parallel and the local dispatch lease "
            f"pool); found {raw_pool!r}. Add session_slot_pool_size to the "
            "server section (LP-0MTCZ35X7009IZKE)."
        )

    # --- warm cap shadowed by the economic ceiling (LP-0MVHSET0A008VU95) ---
    # effective_warm = min(physical clamp, economic ceiling, absolute), where
    # the economic ceiling is round(ratio x local_model_ctx_size). An absolute
    # value ABOVE that ceiling never binds, so raising it alone is a silent
    # no-op. Reported independently of min_local_routing_threshold so the two
    # knobs stay a co-change.
    shortfall = warm_cap_ceiling_shortfall(config)
    if shortfall > 0:
        warm_abs = _get_warm_cache_threshold(config)
        ratio = _get_economic_ceiling_ratio(config)
        ctx = _get_local_model_ctx_size(config)
        problems.append(
            f"local_large_context_warm_cache_threshold={warm_abs} is shadowed "
            f"by the economic ceiling round({ratio} x {ctx}) = {warm_abs - shortfall}"
            f": the effective warm cap will be {warm_abs - shortfall}, not "
            f"{warm_abs} ({shortfall} tokens lost). Raise "
            "local_large_context_economic_ceiling_ratio to at least "
            f"{warm_abs}/{ctx} (the two knobs must move together, "
            "LP-0MVHSET0A008VU95)."
        )

    if min_threshold <= 0:
        return problems  # minimum check disabled (min_local_routing_threshold: 0)

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


class _RoutingEstimateCache:
    """Lazily-computed, per-cycle routing token estimate.

    The authoritative routing estimate (``_get_tokenizer_for_model`` +
    ``_estimate_effective_prompt_tokens_for_routing`` +
    ``token_estimate_multiplier``) is **model-scoped**, not provider-scoped, so
    every per-entry consumer in a fallback cycle — the local
    ``_should_skip_local`` gate and the per-entry ``max_context_tokens``
    ceiling — must share ONE computation (LP-0MVF9N2QL001CDA7 AC4).

    Lazy: a cycle that never consults the estimate (e.g. an ungated remote
    chain) pays no tokenizer pass at all. ``recompute()`` refreshes it after
    the request body has been rewritten in place (compaction).
    """

    def __init__(self, request, body_json: dict, model_config: dict, config: dict):
        self._request = request
        self._body_json = body_json
        self._model_config = model_config
        self._config = config
        self._value: int | None = None
        self.tokenizer = None
        self.multiplier = 1.0

    async def get(self) -> int:
        """Return the cached estimate, computing it on first use."""
        if self._value is None:
            await self.recompute()
        return int(self._value or 0)

    async def recompute(self) -> int:
        """Re-estimate from the current body (after in-place compaction)."""
        self.tokenizer, self.multiplier = _get_tokenizer_for_model(
            self._model_config, self._config
        )
        tokens = await _estimate_effective_prompt_tokens_for_routing(
            self._request, self._body_json, tokenizer=self.tokenizer
        )
        if self.multiplier != 1.0:
            tokens = int(tokens * self.multiplier)
        self._value = int(tokens or 0)
        return self._value


def _should_skip_local(
    model_name: str,
    session_id: str | None,
    body_json: dict,
    cold_cache_threshold: int,
    estimated_tokens: int | None = None,
    warm_cache_threshold: int = 0,
    economic_bypass_serves_local: bool = False,
) -> tuple[bool, str | None]:
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

    Config-driven mode behaviour (LP-0MU5A4QBR003YJM0):

    - **``economic_bypass_serves_local is True``** (set by the active mode
      config — ``config-cheap.yaml``): when the *only* trigger is the
      economic cold-cache check (``large_context_bypass``), the request is
      served locally instead of being skipped.  This preserves cheap mode's
      intent to avoid paid remote calls by waiting in the contention queue
      rather than falling back.  The physical-capacity check
      (``context_too_large``) still applies in both modes.
    - **``economic_bypass_serves_local is False``** (default; fast profile):
      behaviour is unchanged — the economic check still causes a skip
      (``large_context_bypass``).

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
        economic_bypass_serves_local: Config-driven flag (from the active
            mode profile) enabling cheap-mode economic-bypass recovery.

    Returns:
        A tuple ``(skip, reason)`` where:

        - ``skip`` is True if local should be skipped, False for normal
          local routing.
        - ``reason`` is the skip reason when ``skip`` is True
          (``"context_too_large"`` or ``"large_context_bypass"``), or None
          when ``skip`` is False.  When ``economic_bypass_serves_local`` is
          True and the economic check would normally trigger but physical
          capacity is available, ``(False, None)`` is returned so the
          request proceeds to local (with contention-queue wait if slots
          are busy).
    """
    if cold_cache_threshold <= 0:
        return (False, None)
    if estimated_tokens is None:
        estimated_tokens = _estimate_prompt_tokens_for_routing(body_json)

    # Check 1: Warm-cache threshold (hard cap on total context)
    if warm_cache_threshold > 0 and estimated_tokens > warm_cache_threshold:
        return (True, "context_too_large")

    # If estimated tokens are below cold cache threshold, always route local
    if estimated_tokens <= cold_cache_threshold:
        return (False, None)

    # Check 2: Dynamic new-token calculation
    ratio = expected_cached_ratio(
        model_name,
        session_id,
        estimated_tokens,
        (body_json or {}).get("messages") if isinstance(body_json, dict) else None,
    )
    new_tokens = int(estimated_tokens * (1 - ratio))
    if new_tokens <= cold_cache_threshold:
        return (False, None)

    # Economic cold-cache bypass (large_context_bypass)
    if economic_bypass_serves_local:
        # Active mode profile recovers economic bypasses locally. The
        # contention queue (if enabled) will wait for a slot when all are
        # busy.
        return (False, None)

    return (True, "large_context_bypass")


# ---------------------------------------------------------------------------
# Idle-aware local admission (LP-0MVGD9Z8Z0055Y5J)
#
# Routing is size/economics-driven and ignores whether the backend is doing
# anything, so a large or cache-cold prompt is bypassed while both slots sit
# unused (measured: 187 ``local_dispatch_denied`` in an hour where local busy
# was 2.5 %). When the local model has demonstrably been idle, the economic
# cold-cache bypass is overridden and the request is served locally.
# ---------------------------------------------------------------------------

IDLE_ADMIT_REASON = "economic_bypass_idle"
"""Skip-admission reason recorded when the idle override admits a request."""

DEFAULT_IDLE_ADMIT_SECONDS = 60.0

def _idle_admit_config(server_config: dict | None) -> tuple[bool, float]:
    """Return ``(enabled, idle_seconds_threshold)`` for the idle override.

    Disabled by default (``local_idle_admit_enabled: false``). A missing,
    malformed, or non-positive ``local_idle_admit_seconds`` (default 60)
    disables the override rather than admitting on a guess — the guard must
    never widen admission because of a config typo.
    """
    if not isinstance(server_config, dict):
        return False, DEFAULT_IDLE_ADMIT_SECONDS
    raw_seconds = server_config.get(
        "local_idle_admit_seconds", DEFAULT_IDLE_ADMIT_SECONDS
    )
    try:
        seconds = float(raw_seconds)
    except (TypeError, ValueError):
        return False, DEFAULT_IDLE_ADMIT_SECONDS
    if seconds <= 0:
        return False, seconds
    return bool(server_config.get("local_idle_admit_enabled", False)), seconds


def _local_idle_blocker(endpoint: str | None) -> str | None:
    """Return why idle admission must not fire, or ``None`` when it may.

    Fail-safe checks (AC2/AC3): nothing may be admitted while the backend is
    unhealthy, self-healing, switching/loading a model, holding a lease, or
    prefilling — or while any request is queued for a slot.
    """
    try:
        import proxy.server as srv
    except Exception:
        return "server_unavailable"

    try:
        if srv.backend_recovery_state.get("in_progress"):
            return "self_healing"
    except Exception:
        pass

    # Model switch / background load in progress.
    try:
        if int(getattr(srv, "model_switch_refcount", 0) or 0) > 0:
            return "model_switch"
    except Exception:
        pass
    try:
        lock = getattr(srv, "model_switch_lock", None)
        if lock is not None and lock.locked():
            return "model_switch"
    except Exception:
        pass
    try:
        if bool(getattr(srv, "background_loads", None)):
            return "model_loading"
    except Exception:
        pass

    # Local backend health. An explicit provider ``endpoint`` points at an
    # externally managed llama-server, so the single-process ready gate does
    # not apply to it (LP-0MRPILSMW004T4H8).
    if not endpoint:
        if not bool(getattr(srv, "backend_ready", False)):
            return "backend_not_ready"
        if getattr(srv, "llama_process", None) is None:
            return "backend_not_running"

    # Lease / prefill / generation in flight: the GPU is not idle.
    try:
        if int(getattr(srv, "local_active_queries", 0) or 0) > 0:
            return "local_active"
    except Exception:
        pass
    try:
        if int(getattr(srv, "local_generating_queries", 0) or 0) > 0:
            return "generating"
    except Exception:
        pass
    try:
        if getattr(srv, "local_prefill_in_flight", None):
            return "prefill_in_flight"
    except Exception:
        pass

    # A queued request means the next slot must go to interactive traffic.
    try:
        from proxy import contention_queue

        if contention_queue.queue_depth() > 0:
            return "queue_not_empty"
    except Exception:
        pass

    return None


def _evaluate_idle_admission_override(
    config: dict,
    provider_cfg: dict,
    estimated_tokens: int,
) -> dict | None:
    """Decide whether idle local capacity should serve a bypassed request.

    Only the **economic** band (``large_context_bypass``) is eligible: the
    physical ``context_too_large`` cap is a capacity limit and is never
    overridden. When the local backend has been idle for at least
    ``local_idle_admit_seconds`` and :func:`_local_idle_blocker` reports no
    blocker, a decision dict is returned; otherwise ``None`` (today's
    behaviour, byte-for-byte).

    Returns:
        ``{"reason", "idle_seconds", "threshold_seconds"}`` when the request
        should be admitted locally, else ``None``.
    """
    try:
        server_config = config.get("server", config) if isinstance(config, dict) else {}
        enabled, threshold = _idle_admit_config(server_config)
        if not enabled:
            return None
        endpoint = _get_local_provider_endpoint(provider_cfg, config)
        blocker = _local_idle_blocker(endpoint)
        if blocker is not None:
            logger.debug(
                "idle_admit_blocked reason=%s endpoint=%s estimated_tokens=%d",
                blocker,
                endpoint or "default",
                max(0, int(estimated_tokens or 0)),
            )
            return None
        from proxy.local_activity import local_idle_seconds

        idle_seconds = local_idle_seconds()
        if idle_seconds < threshold:
            return None
        return {
            "reason": IDLE_ADMIT_REASON,
            "idle_seconds": idle_seconds,
            "threshold_seconds": threshold,
        }
    except Exception:
        logger.debug("idle admission evaluation failed", exc_info=True)
        return None


def _provider_context_limit(provider_cfg: dict) -> int:
    """Read ``max_context_tokens`` from a provider entry (0 = ungated).

    LP-0MVF9N2QL001CDA7: the optional per-entry ceiling on the estimated prompt
    context the entry may serve. Parsing mirrors ``rate_limit_rpm``
    (``provider_budget._provider_rpm``): a malformed value fails open (the entry
    is ungated) rather than raising at dispatch time, and any non-positive
    value — ``0``, a negative number, ``None`` — disables the gate so an absent
    field is byte-for-byte today's behaviour.

    Args:
        provider_cfg: A single provider entry from the model's ``providers``.

    Returns:
        The positive integer limit, or ``0`` when the entry is ungated.
    """
    if not isinstance(provider_cfg, dict):
        return 0
    raw = provider_cfg.get("max_context_tokens", 0)
    # ``bool`` is an ``int`` subclass: ``True`` would silently mean a 1-token
    # ceiling, so treat it as malformed (ungated).
    if isinstance(raw, bool):
        return 0
    try:
        value = int(raw or 0)
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


def _should_skip_provider_context(
    provider_cfg: dict,
    estimated_tokens: int | None,
) -> tuple[bool, int]:
    """Decide whether an entry's ``max_context_tokens`` ceiling is exceeded.

    The ceiling is **inclusive**: an estimate at or below the limit is
    dispatched; only a strictly larger estimate is skipped. Fail-open in both
    directions — an ungated entry never skips, and a missing/non-positive
    estimate (nothing to compare) never skips either, mirroring the local
    gate's behaviour when it has no usable estimate.

    Args:
        provider_cfg: A single provider entry from the model's ``providers``.
        estimated_tokens: The authoritative per-request routing estimate.

    Returns:
        ``(skip, limit)``; ``limit`` is ``0`` when the entry is ungated.
    """
    limit = _provider_context_limit(provider_cfg)
    if limit <= 0:
        return (False, 0)
    try:
        estimate = int(estimated_tokens or 0)
    except (TypeError, ValueError):
        return (False, limit)
    if estimate <= 0:
        return (False, limit)
    return (estimate > limit, limit)


def _record_provider_context_skip(
    attempts: list[dict[str, Any]],
    provider_name: str,
    provider_type: str,
    estimated_tokens: int,
    limit: int,
    session_id: str | None = None,
) -> None:
    """Record + log a per-entry context-ceiling skip (AC3).

    A context skip is a per-request size mismatch, NOT a provider failure
    (AC5): this helper deliberately records nothing beyond the diagnostic
    attempt and the cumulative ``provider_context_exceeded`` backend signal —
    no cooldown, no usage-limit reset, no failure-domain exclusion and no
    affinity — so the entry stays eligible for later, smaller requests. The
    counter mirrors ``context_too_large`` (LP-0MVF9N3BV00155RI) and is exposed
    as ``backend_signals`` on ``/health`` and ``/admin/metrics``.
    """
    _record_attempt(
        attempts,
        provider=provider_name,
        type=provider_type,
        status="provider_context_exceeded",
        estimated_tokens=estimated_tokens,
        limit=limit,
        reason="provider_context_exceeded",
    )
    _record_backend_signal("provider_context_exceeded")
    logger.info(
        "routing_skip_provider_context provider=%s type=%s "
        "status=provider_context_exceeded estimated_tokens=%d limit=%d "
        "reason=provider_context_exceeded session=%s",
        provider_name,
        provider_type,
        estimated_tokens,
        limit,
        str(session_id or "unknown"),
    )


# ---------------------------------------------------------------------------
# Headroom-aware quota ordering (LP-0MVF7KIKE002DZVD)
# ---------------------------------------------------------------------------


def _quota_ordering_active(config: dict | None) -> bool:
    """Whether headroom-aware ordering may re-order the provider chain.

    The config-level predicate guarding the whole quota-ordering feature
    (LP-0MVF7KIKE002DZVD): whenever it returns ``False`` every resolution path
    behaves exactly as it did before the feature existed. Ordering is active
    only while the quota-surge feature is enabled *and* not in dry-run.
    Dry-run still computes and logs the ranking
    (:func:`_quota_ranked_providers`) but never applies it.

    Delegates to :func:`_quota_surge_applies` for the config-level gate, and to
    :func:`_quota_behaviour` for the per-account / stale-state part of the
    decision (which also gates the final-week bypass,
    LP-0MVF7KPB9009U7CG).
    """
    return _quota_surge_applies(config)


def _quota_behaviour(
    config: dict | None, now: datetime | None = None
) -> quota_state.QuotaBehaviour:
    """The single central predicate every quota decision site consults.

    Returns :class:`proxy.quota_state.QuotaBehaviour` (enabled / dry-run /
    suspended-on-stale-state / disabled by the kill switch). Ordering, the
    ``available_times`` bypass and the quota-driven skip all branch on this
    one object, so a test can prove "all quota behaviour off" by construction
    (LP-0MVF7KPB9009U7CG). ``now`` defaults to the injectable quota clock.
    """
    return quota_state.quota_behaviour(config, now=now)


def _quota_provider_score(
    provider_cfg: dict, config: dict | None
) -> tuple[float, float, float] | None:
    """Return ``(score, effective_headroom, seconds_to_reset)`` or ``None``.

    A pure cache read through :func:`proxy.quota_state.get_quota_snapshot`:
    it never polls, never blocks and never raises. ``None`` means "no quota
    opinion" — the entry keeps its configured position with score 0.
    """
    snapshot = quota_state.get_quota_snapshot(provider_cfg, config)
    if snapshot is None:
        return None
    headroom = snapshot.effective_headroom
    seconds_to_reset = snapshot.seconds_to_reset
    score = quota_pacing.urgency_score(headroom, seconds_to_reset, config)
    return score, headroom, seconds_to_reset


def _quota_ranked_providers(
    model_config: dict, config: dict | None = None
) -> list[dict[str, Any]]:
    """Return the model's provider list with quota entries headroom-ordered.

    Only entries matched by :func:`proxy.quota_state.is_quota_provider`
    (``opencode-*``, or the ``server.quota_surge_accounts`` allow-list) are
    re-ordered: they are ranked by descending
    :func:`proxy.quota_pacing.urgency_score` (``effective_headroom /
    seconds_to_reset``) and spliced back into the *same positions* they
    occupy in config order, so every other provider keeps its configured
    position. The sort is stable, so equal scores tie-break on config order
    and repeated calls are deterministic.

    A copy is always returned — the caller's list is never mutated. An entry
    with no snapshot scores 0 and keeps config order among its peers; when no
    quota entry has been polled the configured order is preserved exactly.

    Ordering is gated by the central :func:`_quota_behaviour` predicate: it is
    suspended entirely when any snapshot is stale (older than
    ``server.quota_state_max_age_seconds``), so the chain reverts to config
    order and a stale reading can never re-rank or strand a candidate (child
    AC9). Disabled and suspended modes therefore both return the configured
    order unchanged.
    """
    providers = model_config.get("providers")
    if not isinstance(providers, list) or len(providers) < 2:
        return list(providers) if isinstance(providers, list) else []
    if not quota_pacing.surge_enabled(config):
        return list(providers)

    behaviour = _quota_behaviour(config)
    if not behaviour.computes:
        # Stale state: fail-safe — no ranking, no log, plain config order.
        return list(providers)

    active = _quota_ordering_active(config)
    quota_positions: list[int] = []
    ranked: list[dict[str, Any]] = []
    for index, provider_cfg in enumerate(providers):
        if not isinstance(provider_cfg, dict):
            continue
        if not quota_state.is_quota_provider(provider_cfg, config):
            continue
        quota_positions.append(index)
        scored = _quota_provider_score(provider_cfg, config)
        if scored is None:
            ranked.append(
                {
                    "score": 0.0,
                    "index": index,
                    "provider": provider_cfg,
                    "headroom": 0.0,
                    "seconds_to_reset": 0.0,
                    "polled": False,
                }
            )
            continue
        score, headroom, seconds_to_reset = scored
        ranked.append(
            {
                "score": score,
                "index": index,
                "provider": provider_cfg,
                "headroom": headroom,
                "seconds_to_reset": seconds_to_reset,
                "polled": True,
            }
        )

    if not quota_positions:
        return list(providers)

    # ``reverse=True`` keeps the sort stable, so equal scores retain config order.
    ranked.sort(key=lambda entry: entry["score"], reverse=True)

    if any(entry["polled"] for entry in ranked):
        for rank, entry in enumerate(ranked, start=1):
            logger.debug(
                "quota_ordering rank=%d provider=%s config_index=%d headroom=%.1f "
                "seconds_to_reset=%.0f score=%.6g applied=%s behaviour=%s reason=%s",
                rank,
                entry["provider"].get("name", "unknown"),
                entry["index"],
                entry["headroom"],
                entry["seconds_to_reset"],
                entry["score"],
                active,
                behaviour.mode,
                behaviour.log_reason(),
            )

    if not active:
        return list(providers)

    ordered = [entry["provider"] for entry in ranked]
    result = list(providers)
    for position, provider_cfg in zip(quota_positions, ordered):
        result[position] = provider_cfg
    return result


def _quota_unavailable_reason(
    provider_cfg: dict, config: dict | None
) -> str | None:
    """Return a log-friendly reason when quota state marks the entry spent.

    Ordering never drops a candidate that still has headroom (AC3c), but an
    entry whose snapshot is
    :attr:`~proxy.quota_state.QuotaSnapshot.exhausted` (any window blocking,
    or no effective headroom left) is treated as unavailable — the same
    conservative stance the snapshot itself takes.

    The decision is gated on the central behaviour predicate
    (:func:`proxy.quota_state.quota_behaviour`): it returns ``None`` (fail
    open) when the feature is disabled, when the state is stale (suspended —
    a stale exhausted reading must not strand the account, child AC9b), when
    the entry is not quota-managed, or when no snapshot has been polled yet.
    In dry-run mode the would-be skip is logged once per account with
    ``reason=quota_surge_dry_run`` but ``None`` is returned, so eligibility is
    byte-identical to the pre-feature behaviour (child AC8b).
    """
    if not quota_state.is_quota_provider(provider_cfg, config):
        return None
    behaviour = _quota_behaviour(config)
    if behaviour.disabled or behaviour.suspended:
        return None
    snapshot = quota_state.get_quota_snapshot(provider_cfg, config)
    if snapshot is None or not snapshot.exhausted:
        return None
    reason = (
        f"quota_exhausted (account={snapshot.account}, "
        f"effective_headroom={snapshot.effective_headroom:.1f}%, "
        f"resets_in={int(snapshot.seconds_to_reset)}s)"
    )
    if behaviour.dry_run:
        _log_quota_skip_decision(provider_cfg, snapshot)
        return None
    return reason


def _quota_exhausted(
    provider_cfg: dict, config: dict | None
) -> tuple[bool, str]:
    """Return ``(True, reason)`` when the quota hard stop skips *provider_cfg*.

    The cost guard (LP-0MVF7KO8C005NB59): an entry is skipped before any
    upstream request is attempted when its monthly consumption is at or above
    ``server.quota_exhaustion_percent`` (default 99.5) **or** its
    ``effective_headroom`` is at or below the equivalent headroom
    (``100 - threshold``) — so a spent rolling or weekly window blocks the
    entry even when the monthly percent is still low. The decision is a pure
    cache read: it never polls, never blocks and never raises.

    Unlike the ordering-gated :func:`_quota_unavailable_reason`, this guard is
    **not** conditional on the surge feature: an exhausted account is skipped
    whether or not surge is enabled, and the final-week surge bypass can
    never re-enable it (the skip runs before the ``available_times``/surge
    gate). It is also deliberately **not** suspended by stale state
    (``server.quota_state_max_age_seconds``): it is a money guard, so a stale
    "nearly spent" reading fails in the safe direction (keep blocking) rather
    than spending real pay-as-you-go money on an old observation. Returns
    ``(False, "")`` (fail open) when the entry is not
    quota-managed or no snapshot has been polled yet — the guard only acts on
    authoritative quota state.
    """
    if not quota_state.is_quota_provider(provider_cfg, config):
        return False, ""
    snapshot = quota_state.get_quota_snapshot(provider_cfg, config)
    if snapshot is None:
        return False, ""
    threshold = quota_state.quota_exhaustion_percent(config)
    monthly_percent = snapshot.monthly.percent
    effective_headroom = snapshot.effective_headroom
    if (
        monthly_percent < threshold
        and effective_headroom > 100.0 - threshold
    ):
        return False, ""
    return True, (
        f"reason=quota_exhausted (account={snapshot.account}, "
        f"monthly_percent={monthly_percent:.1f}%, "
        f"effective_headroom={effective_headroom:.1f}%, "
        f"threshold={threshold:g}%)"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_provider(
    model_config: dict,
    failed_provider: str | None = None,
    *,
    config: dict | None = None,
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
    - Is not marked spent by proactive quota state (LP-0MVF7KIKE002DZVD)
    - Is not at or near its monthly quota cap (the cost-guard hard stop —
      ``server.quota_exhaustion_percent``, LP-0MVF7KO8C005NB59)

    Quota-managed (``opencode-*``) entries are considered in descending
    headroom urgency instead of config order — see
    :func:`_quota_ranked_providers`.

    Args:
        model_config: Model configuration dict. Must contain a ``providers``
                      key whose value is an ordered list of provider configs.
        failed_provider: Optional name of a provider that just failed and
                         should be skipped.
        config: Optional proxy config, used for the quota-ordering gates
                (``server.quota_surge_*``). ``None`` falls back to the
                documented defaults.

    Returns:
        A provider config dict (with keys ``name``, ``type``, etc.), or
        ``None`` if no provider is available.
    """
    providers = _quota_ranked_providers(model_config, config)
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
        # Proactive quota state says this account has nothing left to spend
        # (LP-0MVF7KIKE002DZVD): treat it as unavailable instead of burning a
        # dispatch on a guaranteed limit error.
        quota_reason = _quota_unavailable_reason(provider_cfg, config)
        if quota_reason is not None:
            logger.info("Skipping provider=%s: %s", name, quota_reason)
            continue
        # Quota hard stop (LP-0MVF7KO8C005NB59): at-or-near the monthly cap the
        # entry is skipped before any upstream request is attempted. Runs after
        # the usage-limit-quarantine and cooldown checks so those reasons keep
        # their precedence, and before the available_times/surge gate so the
        # final-week surge can never re-enable an exhausted account. The
        # guard is independent of the surge kill switch/dry-run.
        quota_hard_stop, quota_stop_reason = _quota_exhausted(
            provider_cfg, config
        )
        if quota_hard_stop:
            logger.info("Skipping provider=%s: %s", name, quota_stop_reason)
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
        if not _provider_in_allowed_window(provider_cfg, config):
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


def _get_local_provider_endpoint(provider_cfg: dict, config: dict) -> str:
    """Resolve the endpoint URL for a ``type: local`` provider entry.

    When the provider config declares an explicit ``endpoint`` field, that
    URL is returned unchanged (LP-0MRPILSMW004T4H8). Otherwise the legacy
    single-server default ``http://localhost:{llama_server_port}`` (default
    port 8080) is used for backward compatibility.

    Args:
        provider_cfg: The provider config dict (must be ``type: local``).
        config: Server configuration dict (for ``server.llama_server_port``).

    Returns:
        The resolved endpoint URL string (no trailing slash).
    """
    endpoint = provider_cfg.get("endpoint")
    if endpoint:
        return str(endpoint).rstrip("/")
    try:
        server_cfg = config.get("server", config) if isinstance(config, dict) else {}
        port = int(server_cfg.get("llama_server_port", 8080) or 8080)
    except (TypeError, ValueError):
        port = 8080
    return f"http://localhost:{port}"


def _check_local_backend_gpu_oom(endpoint: str, body_text: str) -> bool:
    """Detect GPU OOM / allocation-failure patterns in a local backend response.

    Scope of this helper is the 5GB/8GB-class single-GPU case: llama-server
    surfaces memory exhaustion as 500 responses whose body contains patterns
    like ``out of memory`` (LP-0MRPILSMW004T4H8 AC5b). When detected, the
    requesting code should mark the provider unhealthy so the fallback chain
    skips the wedged server instead of hammering it.

    Detection is best-effort: llama-server may return generic 500s without a
    distinguishable body. Slot-capacity probes (HTTP health + available-slot
    count) supplement this in ``probe_local_backend``.
    """
    try:
        from proxy.backend_health import _detect_gpu_oom_in_text

        return _detect_gpu_oom_in_text(body_text or "")
    except Exception:
        return False


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
    _persist_provider_state_best_effort()


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
    # Model-scoped failure-domain quarantine: the sibling-fallback circuit
    # breaker marks ``normalized_endpoint:model`` so same-model siblings on a
    # shared gateway are skipped WITHOUT over-quarantining other models on the
    # same gateway (LP-0MTVMB8DW0067H9A AC4).
    domain = _failure_domain_key(provider_cfg)
    if domain != name and _is_provider_unavailable(domain):
        return domain
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


# ---------------------------------------------------------------------------
# Final-week available_times surge bypass (LP-0MVF7KJN2000ASE8)
#
# During the final ``server.quota_surge_days`` (default 7) before an
# ``opencode-*`` account's monthly reset its remaining allowance is
# use-it-or-lose-it, so the ``available_times`` restriction (a pricing-only
# off-peak gate, LP-0MS4ETBNO0022QAC) is lifted for THAT account: the residual
# may be spent at any UTC hour. Nothing else is bypassed — cooldown,
# ``rate_limit_rpm`` budget, the sibling-failure circuit breaker, failure-domain
# grouping, usage-limit quarantine and stall gates all keep running unchanged
# (child AC6 / epic R7).
#
# The window is computed per account from its OWN ``resetsAt`` (reset dates are
# not calendar-aligned) and the bypass clears automatically when the account
# becomes exhausted or its reset passes (child AC5b).
# ---------------------------------------------------------------------------

#: account key -> last observed bypass state, for once-per-transition logging.
_SURGE_BYPASS_LOGGED: dict[str, bool] = {}

#: ``(decision, account)`` pairs already reported in dry-run, so a dry run
#: names each would-be decision once per account instead of on every request.
_QUOTA_DRY_RUN_LOGGED: set[tuple[str, str]] = set()


def reset_surge_bypass_state() -> None:
    """Forget per-account transition / dry-run decision log state (tests)."""
    _SURGE_BYPASS_LOGGED.clear()
    _QUOTA_DRY_RUN_LOGGED.clear()


def _log_quota_skip_decision(
    provider_cfg: dict, snapshot: quota_state.QuotaSnapshot
) -> None:
    """INFO once per account that a dry run *would* have skipped the entry.

    Dry-run must not change eligibility (child AC8b), so the decision is
    reported and the caller returns ``None`` (fail open) instead of skipping.
    The ``reason=quota_surge_dry_run`` code keeps these lines greppable apart
    from the applied ``reason=quota_exhausted`` skips.
    """
    key = ("skip", snapshot.account)
    if key in _QUOTA_DRY_RUN_LOGGED:
        return
    _QUOTA_DRY_RUN_LOGGED.add(key)
    logger.info(
        "quota_skip dry_run would_skip=true provider=%s account=%s "
        "effective_headroom=%.1f%% resets_in=%ds reason=quota_surge_dry_run",
        str(provider_cfg.get("name") or "unknown"),
        snapshot.account,
        snapshot.effective_headroom,
        int(snapshot.seconds_to_reset),
    )


def _log_surge_bypass_dry_run(
    provider_cfg: dict, snapshot: quota_state.QuotaSnapshot
) -> None:
    """INFO once per account that a dry run *would* have lifted the window.

    The would-be bypass is exactly the decision that differs from today's
    routing (the configured ``available_times`` window still applies in
    dry-run), so it is logged with ``reason=quota_surge_dry_run`` while the
    eligibility decision itself stays unchanged (child AC8b).
    """
    key = ("bypass", snapshot.account)
    if key in _QUOTA_DRY_RUN_LOGGED:
        return
    _QUOTA_DRY_RUN_LOGGED.add(key)
    resets_at = snapshot.monthly.resets_at
    resets_at_text = (
        resets_at.isoformat().replace("+00:00", "Z") if resets_at else "unknown"
    )
    logger.info(
        "quota_surge_bypass dry_run would_bypass=true provider=%s account=%s "
        "resets_at=%s seconds_to_reset=%ds reason=quota_surge_dry_run",
        str(provider_cfg.get("name") or "unknown"),
        snapshot.account,
        resets_at_text,
        int(snapshot.seconds_to_reset),
    )


def _quota_surge_applies(config: dict | None = None) -> bool:
    """Config-level gate: surge is enabled and not in dry-run.

    The single review point for the kill switch (``quota_surge_enabled``) and
    dry-run (``quota_surge_dry_run``). The per-account decision (including the
    stale-state suspension) is made by
    :func:`proxy.quota_state.quota_behaviour`; this helper only answers
    "would surge be allowed at all for this config?".
    """
    return quota_pacing.surge_enabled(config) and not quota_pacing.surge_dry_run(
        config
    )


def _log_surge_bypass_transition(
    provider_cfg: dict,
    snapshot: quota_state.QuotaSnapshot,
    active: bool,
) -> None:
    """Log one INFO line per account state change (enter / leave) only.

    Per-request logging would flood the log; the transition is remembered per
    quota account (``api_key_env``, so two entries on independent keys track
    independently). The line carries ``reason=quota_surge`` so operators can
    grep the bypass, plus the account and its ``resetsAt``.
    """
    account = quota_state.account_key(provider_cfg)
    previous = _SURGE_BYPASS_LOGGED.get(account, False)
    if previous is active:
        return
    _SURGE_BYPASS_LOGGED[account] = active
    resets_at = snapshot.monthly.resets_at
    resets_at_text = (
        resets_at.isoformat().replace("+00:00", "Z") if resets_at else "unknown"
    )
    logger.info(
        "quota_surge_bypass %s provider=%s account=%s resets_at=%s "
        "seconds_to_reset=%ds reason=quota_surge",
        "entered" if active else "left",
        str(provider_cfg.get("name") or "unknown"),
        account,
        resets_at_text,
        int(snapshot.seconds_to_reset),
    )


def _quota_surge_bypass_active(
    provider_cfg: dict, config: dict | None = None
) -> bool:
    """True when the final-week surge removes this entry's time restriction.

    Active only when **all** hold (child AC5 / epic R4):

    - the entry is quota-managed (:func:`proxy.quota_state.is_quota_provider`);
    - :func:`_quota_behaviour` reports ``enabled`` — the central predicate, so
      the kill switch, dry-run and stale-state suspension all switch the
      bypass off (children AC8 / AC8b / AC9);
    - a snapshot exists and is not
      :attr:`~proxy.quota_state.QuotaSnapshot.exhausted`;
    - ``0 < seconds_to_reset <= server.quota_surge_days * 86400``, measured
      from the account's own monthly ``resetsAt``.

    A missing snapshot fails closed (no bypass): without authoritative quota
    state the configured ``available_times`` gate stands. A **stale** snapshot
    suspends the bypass (fail-safe revert to the configured window), and the
    live transition state is dropped so a later recovery logs a fresh
    ``entered`` line. In dry-run the would-be bypass is logged once per
    account with ``reason=quota_surge_dry_run`` and ``False`` is returned, so
    eligibility, retry-after and next-window values stay unchanged.
    The module clock is injectable through
    :func:`proxy.quota_state.set_clock`, so the reset horizon is deterministic
    in tests.
    """
    if not quota_state.is_quota_provider(provider_cfg, config):
        return False
    behaviour = _quota_behaviour(config)
    if behaviour.disabled:
        return False
    if behaviour.suspended:
        # Fail-safe: stale state must not lift a configured restriction. Drop
        # the live transition flag so a later recovery re-logs ``entered``.
        _SURGE_BYPASS_LOGGED[quota_state.account_key(provider_cfg)] = False
        return False
    snapshot = quota_state.get_quota_snapshot(provider_cfg, config)
    if snapshot is None:
        return False
    active = not snapshot.exhausted and quota_pacing.surge_window_active(
        snapshot.seconds_to_reset, config
    )
    if behaviour.dry_run:
        if active:
            _log_surge_bypass_dry_run(provider_cfg, snapshot)
        return False
    _log_surge_bypass_transition(provider_cfg, snapshot, active)
    return active


def _provider_in_allowed_window(
    provider_cfg: dict,
    config: dict | None = None,
    now_utc: datetime | None = None,
) -> bool:
    """True when *provider_cfg* may be used now: window **or** surge bypass.

    The single eligibility predicate every resolution path (and the
    retry-after / diagnostics derivations) uses for ``available_times``, so
    they can never disagree: :func:`_is_within_allowed_window` is unchanged and
    the final-week bypass is layered on top of it (LP-0MVF7KJN2000ASE8).
    Nothing else is relaxed — callers keep checking cooldown, budget, circuit
    breaker, failure domains, quarantine and stall gates themselves.
    The bypass itself is off whenever surge is in dry-run, suspended (stale
    quota state) or disabled, so eligibility, ``retry_after`` and
    next-window values are byte-identical to the pre-feature behaviour in
    those modes (LP-0MVF7KPB9009U7CG).
    """
    if _is_within_allowed_window(provider_cfg, now_utc=now_utc):
        return True
    return _quota_surge_bypass_active(provider_cfg, config)


def _seconds_until_next_window(
    provider_cfg: dict,
    now_utc: datetime | None = None,
    config: dict | None = None,
) -> float | None:
    """Return seconds until the provider's next available_times window opens.

    Returns ``None`` when the provider is currently inside its window or has
    no ``available_times`` restriction. Otherwise returns the positive number
    of seconds until the next window start (UTC).

    A surged entry (LP-0MVF7KJN2000ASE8) is already usable, so it contributes
    no phantom "next window" delay: the surge bypass is part of the
    eligibility check via :func:`_provider_in_allowed_window` (child AC5b).
    """
    windows = _parse_available_times(provider_cfg)
    if windows is None:
        return None
    if _provider_in_allowed_window(provider_cfg, config, now_utc=now_utc):
        return None
    if now_utc is None:
        now_utc = datetime.now(UTC)
    current_min = now_utc.hour * 60 + now_utc.minute
    current_sec = now_utc.second
    # Find the soonest upcoming window start
    best: int | None = None
    for start_min, _end_min in windows:
        if start_min > current_min:
            delta_min = start_min - current_min
        elif start_min == current_min and current_sec == 0:
            delta_min = 0
        else:
            # Window is tomorrow
            delta_min = (24 * 60 - current_min) + start_min
        if best is None or delta_min < best:
            best = delta_min
    if best is None:
        return None
    # Subtract elapsed seconds within the current minute for sub-minute precision
    return float(best * 60 - current_sec)


def _compute_retry_after(
    unavailable_providers: dict | None = None,
    model_config: dict | None = None,
    now_utc: datetime | None = None,
    config: dict | None = None,
) -> int:
    """Compute an honest ``retry_after`` (seconds) from real availability data.

    Takes the maximum of:
    - cooldown durations (``unavailable_providers`` values)
    - available_times window edges (seconds until next window for providers
      currently outside their window)
    - usage-limit reset times (``_usage_reset_at`` remaining seconds)

    ``config`` enables the final-week surge bypass for the window edges: a
    surged entry is usable now, so it contributes no delay
    (LP-0MVF7KJN2000ASE8, child AC5b).

    Returns 0 when every provider is actually available (edge case).
    """
    candidates: list[float] = []

    if unavailable_providers:
        for v in unavailable_providers.values():
            try:
                candidates.append(float(v))
            except Exception:
                pass

    if model_config is not None:
        for p in (model_config.get("providers") or []):
            if not isinstance(p, dict):
                continue
            secs = _seconds_until_next_window(p, now_utc=now_utc, config=config)
            if secs is not None and secs > 0:
                candidates.append(secs)

    # Usage-limit resets: consider every account with a pending reset
    for _key, expiry in _usage_reset_at.items():
        remaining = expiry - time.time()
        if remaining > 0:
            candidates.append(remaining)

    if not candidates:
        return 0
    return int(max(candidates))


#: Attempt statuses recorded when the proactive free-tier request-budget gate
#: (``proxy.provider_budget.acquire_provider_budget``) denied a dispatch. A
#: budget skip applies NO cooldown and NO usage-limit quarantine, so the
#: generic exhaustion payload carries ``unavailable={}`` and
#: ``_compute_retry_after`` finds no candidate (returns 0) even though the
#: provider comes back as soon as its rolling window drains
#: (LP-0MVF0N9SA0071LWM).
_BUDGET_SKIP_STATUSES = frozenset({"rate_limited"})


def _budget_window_retry_after(
    diagnostics: list[dict[str, Any]] | None = None,
) -> int:
    """Derive a ``retry_after`` from a free-tier budget window rollover.

    A budget skip is the one exhaustion cause that records no cooldown and no
    reset time: the provider is healthy and simply out of allowance for the
    rest of its sliding window. Without this derivation the exhaustion payload
    advertises ``retry_after: 0`` — which the chain-hold reads as "instant
    recovery" and retries immediately, replaying every cycle inside the same
    still-full window (the measured 13/11 ``munitoai-free`` failure,
    LP-0MVF0N9SA0071LWM).

    For every diagnostics entry that IS a budget skip (``status`` in
    :data:`_BUDGET_SKIP_STATUSES`, entry carrying a window cap), ask the rate
    limiter how long until that entry's window frees a slot and take the
    EARLIEST positive rollover: the next cycle can succeed as soon as ANY
    provider regains capacity. ``0`` is returned when no budget-skipped entry
    is waiting on its window (capacity is available now, or the blockers are
    not budget-related) so genuine instant recovery keeps zero added latency.

    The derived value can never exceed the limiter's window
    (:data:`RATE_LIMIT_WINDOW_SECONDS`), so it is inherently within the
    chain-hold budgets; the caller additionally caps the advertised value with
    ``max_retry_after_seconds`` and ``_bounded_hold_seconds`` bounds the hold
    by ``chain_hold_seconds`` / ``chain_hold_total_seconds``.
    """
    if not diagnostics:
        return 0
    try:
        from proxy.rate_limiter import get_rate_limiter

        limiter = get_rate_limiter()
    except Exception:  # pragma: no cover - defensive: limiter always importable
        return 0

    rollovers: list[float] = []
    for attempt in diagnostics:
        if not isinstance(attempt, dict):
            continue
        if attempt.get("status") not in _BUDGET_SKIP_STATUSES:
            continue
        entry_name = str(attempt.get("provider") or "")
        if not entry_name:
            continue
        # ``effective_limit`` is the per-window cap actually applied to this
        # request (``limit_rpm - reserve`` for a new session, ``limit_rpm``
        # for an affinity session); fall back to the entry's own limit.
        cap = 0
        for key in ("effective_limit", "limit_rpm"):
            try:
                cap = int(attempt.get(key) or 0)
            except (TypeError, ValueError):
                cap = 0
            if cap > 0:
                break
        if cap <= 0:
            continue
        try:
            seconds = limiter.seconds_until_capacity(
                entry_name, cap, RATE_LIMIT_WINDOW_SECONDS
            )
        except Exception:  # pragma: no cover - defensive: limiter is sync/cheap
            continue
        if seconds > 0:
            rollovers.append(seconds)

    if not rollovers:
        return 0
    return int(math.ceil(min(rollovers)))


# Client-facing exhaustion retry-timer cap (LP-0MV0MM5R60025XZE). The honest
# ``_compute_retry_after`` value can be hours away (the next ``available_times``
# window edge or a pending usage-limit reset). That is truthful, but parking a
# client for up to ~6 h for capacity that routinely returns sooner wastes time.
# The advertised wait (JSON ``retry_after`` + ``Retry-After`` header) is therefore
# bounded; the true reset remains visible in ``diagnostics`` /
# ``unavailable_providers`` for operators. Configurable via
# ``server.max_retry_after_seconds``; a missing, malformed or non-positive value
# falls back to the default so the cap can never be accidentally disabled.
_DEFAULT_MAX_RETRY_AFTER_SECONDS = 3600


def _max_retry_after_seconds(config: dict | None) -> int:
    """Return the client-facing exhaustion retry-timer cap in seconds.

    Reads ``max_retry_after_seconds`` from config, supporting both the
    production nested form (``server.max_retry_after_seconds``) and a flat
    key for backward compatibility with unit tests — matching
    ``_get_cooldown_seconds``. Returns
    :data:`_DEFAULT_MAX_RETRY_AFTER_SECONDS` (3600 s) when the key is absent
    or its value is not a positive number; a malformed value never disables
    the cap (LP-0MV0MM5R60025XZE AC4).
    """
    default = _DEFAULT_MAX_RETRY_AFTER_SECONDS
    if not isinstance(config, dict):
        return default
    raw = config.get("max_retry_after_seconds")
    if raw is None:
        server_cfg = config.get("server")
        if isinstance(server_cfg, dict):
            raw = server_cfg.get("max_retry_after_seconds")
    if isinstance(raw, bool):
        return default
    try:
        seconds = float(raw)
    except (ValueError, TypeError):
        return default
    if seconds <= 0:
        return default
    return int(seconds)


def _quarantined_providers(model_config: dict) -> list[dict[str, Any]]:
    """Return provider entries quarantined by a pending usage-limit reset.

    Usage-limit quarantine (``_usage_reset_at``, keyed by
    ``_usage_limit_account_key``) does not populate the cooldown map
    (``_provider_unavailable_until``), so it was previously invisible to
    exhaustion diagnostics. This helper exposes it with the remaining reset
    seconds and the true reason (quarantine before window/cooldown),
    (LP-0MU56ZFVD001LP0H).

    Each entry contains ``name``, ``type``, ``account``, ``reset_in`` (whole
    seconds) and ``reset_at`` (ISO-8601 UTC). Entries sharing a quarantined
    account are each returned; expired entries are pruned by
    ``_usage_reset_remaining``.
    """
    result: list[dict[str, Any]] = []
    for p in model_config.get("providers") or []:
        if not isinstance(p, dict):
            continue
        account = _usage_limit_account_key(p)
        remaining = _usage_reset_remaining(account)
        if remaining <= 0:
            continue
        result.append({
            "name": p.get("name", "unknown"),
            "type": p.get("type", "remote"),
            "account": account,
            "reset_in": int(remaining),
            "reset_at": datetime.fromtimestamp(
                _usage_reset_at[account], tz=UTC
            ).isoformat(),
        })
    return result


def _record_quarantine_attempts(
    attempts: list[dict[str, Any]], model_config: dict
) -> list[dict[str, Any]]:
    """Append ``usage_limit_reset`` diagnostics for quarantined providers.

    Called before window-skip diagnostics so a provider that is both
    quarantined and outside its window is represented once, as a quota reset
    (quarantine takes precedence over the window reason). Returns the
    quarantined entries for callers that want to log them
    (LP-0MU56ZFVD001LP0H).
    """
    quarantined = _quarantined_providers(model_config)
    for q in quarantined:
        _record_attempt(
            attempts,
            provider=q["name"],
            type=q["type"],
            status="usage_limit_reset",
            reset_in=q["reset_in"],
            reset_at=q["reset_at"],
        )
    return quarantined


def _providers_outside_window(
    model_config: dict, config: dict | None = None
) -> list[dict[str, str]]:
    """Return ``{name, type}`` pairs for providers whose ``available_times``
    window excludes the current UTC time.

    Used to record ``outside_time_window`` diagnostics when a fallback chain is
    exhausted. Providers actually attempted this request cannot be outside
    their window (they were selected), so this set is (absent quarantine)
    exactly the providers skipped solely due to time windows. A provider that
    is also usage-limit quarantined is excluded: quarantine is the true reason
    and is reported once as ``usage_limit_reset`` by
    ``_record_quarantine_attempts`` (LP-0MU56ZFVD001LP0H).

    A surged entry (:func:`_provider_in_allowed_window`) is not restricted, so
    it is never reported as a window skip (LP-0MVF7KJN2000ASE8).
    """
    result: list[dict[str, str]] = []
    for p in model_config.get("providers") or []:
        if not isinstance(p, dict):
            continue
        if _provider_in_allowed_window(p, config):
            continue
        if _usage_reset_remaining(_usage_limit_account_key(p)) > 0:
            continue  # quarantine takes precedence over the window reason
        result.append({
            "name": p.get("name", "unknown"),
            "type": p.get("type", "remote"),
        })
    return result


def format_available_times(provider_cfg: dict) -> str:
    """Format a provider's ``available_times`` for display.

    Returns a human-readable string of ``"HH:MM-HH:MM"`` windows in config
    order with a ``(UTC)`` suffix, or ``"Always"`` when the provider is
    unrestricted (no ``available_times`` or all entries malformed).
    """
    windows = _parse_available_times(provider_cfg)
    if windows is None:
        return "Always"
    parts = []
    for start_min, end_min in windows:
        sh, sm = divmod(start_min, 60)
        eh, em = divmod(end_min, 60)
        parts.append(f"{sh:02d}:{sm:02d}-{eh:02d}:{em:02d}")
    return ", ".join(parts) + " (UTC)"


def format_budget_status(provider_cfg: dict) -> str:
    """Format a provider entry's remaining request budget for display.

    Returns ``"budget 7/10 per 60s"`` for an entry with ``rate_limit_rpm``
    (remaining/limit within the current rolling window), or an empty string
    when the entry is not budget-gated (LP-0MVDQ1GMO00801PL). Used by the
    provider diagnostics table so operators can see how much free-tier
    capacity is left instead of inferring it from skip logs.
    """
    try:
        rpm = int(provider_cfg.get("rate_limit_rpm", 0) or 0)
    except (TypeError, ValueError):
        return ""
    if rpm <= 0:
        return ""
    from proxy.rate_limiter import get_rate_limiter

    remaining = get_rate_limiter().remaining(
        str(provider_cfg.get("name") or ""), rpm, RATE_LIMIT_WINDOW_SECONDS
    )
    return f"budget {remaining}/{rpm} per {RATE_LIMIT_WINDOW_SECONDS}s"


def format_active_status(provider_cfg: dict, now_utc: datetime | None = None) -> str:
    """Format a provider's active status for display.

    Returns ``"Active"`` when the provider is within its allowed window (or
    unrestricted), ``"Inactive"`` otherwise. Uses the same UTC window
    semantics as routing (start-inclusive, end-exclusive, overnight wrap).
    """
    if _is_within_allowed_window(provider_cfg, now_utc=now_utc):
        return "Active"
    return "Inactive"


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


# ---------------------------------------------------------------------------
# Upstream x-ratelimit-* header support (LP-0MVBHDNLB007ALLJ)
#
# Some gateways (notably MunitoAI) answer HTTP 429 with custom
# ``x-ratelimit-*`` headers instead of the standard ``Retry-After``.  Those
# headers carry the upstream's own reset timing, so honouring them yields a
# truthful cooldown instead of a blind exponential-backoff guess that keeps
# probing an endpoint which is guaranteed to reject for the whole window.
#
# Support is opt-in: either a provider entry sets
# ``respect_ratelimit_headers: true`` or the behaviour is enabled globally
# under ``server.respect_ratelimit_headers``.  When neither is set the new
# code paths are never reached and every existing provider behaves exactly as
# before (backward compatibility).
# ---------------------------------------------------------------------------

# (label, reset header, remaining header) for each rate-limit dimension an
# upstream may report.  The label is only used in diagnostics.
_RATELIMIT_RESET_HEADERS: tuple[tuple[str, str, str], ...] = (
    ("requests", "x-ratelimit-reset-requests", "x-ratelimit-remaining-requests"),
    ("tokens", "x-ratelimit-reset-tokens", "x-ratelimit-remaining-tokens"),
    (
        "input_tokens",
        "x-ratelimit-reset-input-tokens",
        "x-ratelimit-remaining-input-tokens",
    ),
    (
        "output_tokens",
        "x-ratelimit-reset-output-tokens",
        "x-ratelimit-remaining-output-tokens",
    ),
)

# One ``<number><unit>`` segment.  ``ms`` is matched before ``m`` so the
# millisecond suffix is not misread as minutes.
_RATELIMIT_DURATION_SEGMENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)")

_RATELIMIT_UNIT_SECONDS: dict[str, float] = {
    "ms": 0.001,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}

# Extra seconds added on top of the upstream reset so a cooldown never expires
# exactly at the boundary the upstream reported (clock skew / queueing).
_RATELIMIT_COOLDOWN_MARGIN_SECONDS = 2.0


def _parse_ratelimit_duration(value: Any) -> float | None:
    """Parse a single ``x-ratelimit-reset-*`` value into seconds.

    Accepts a bare number (seconds), or one or more concatenated
    ``<number><unit>`` segments such as ``52s``, ``1m``, ``1h30m`` or
    ``250ms``.  Returns ``None`` for a missing, empty or unparseable value so
    the caller falls back to the existing backoff untouched.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    # Bare number -> seconds (the Retry-After convention).
    try:
        return float(text)
    except ValueError:
        pass
    total = 0.0
    position = 0
    matched = False
    for match in _RATELIMIT_DURATION_SEGMENT_RE.finditer(text):
        # Reject gaps between segments (e.g. ``1x2s``) so a malformed header
        # is not partially honoured.
        if text[position:match.start()].strip():
            return None
        total += float(match.group(1)) * _RATELIMIT_UNIT_SECONDS[match.group(2)]
        position = match.end()
        matched = True
    if not matched or text[position:].strip():
        return None
    return total


def _ratelimit_over_limit(headers) -> bool:
    """Return ``True`` when the upstream flags the account as over-limit.

    MunitoAI sends ``x-ratelimit-over-limit: yes`` alongside the reset headers.
    Parsing it keeps the flag available for diagnostics and lets callers
    distinguish a rate-limited 429 from an unrelated error.
    """
    value = headers.get("x-ratelimit-over-limit") if headers is not None else None
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _parse_ratelimit_reset_seconds(headers) -> float | None:
    """Return the longest most-relevant ``x-ratelimit-reset-*`` duration.

    Preference order:

    1. When one or more dimensions report ``x-ratelimit-remaining-*`` at or
       below zero, use the longest reset among those exhausted dimensions —
       that is the wait the upstream actually demands before the next request
       can succeed.
    2. Otherwise use the longest parseable reset across all dimensions (a 429
       with reset headers is evidence of *a* limit, so the most conservative
       reported reset is honoured).

    Returns ``None`` when no dimension carries a parseable reset, preserving
    the caller's existing exponential-backoff behaviour.
    """
    if headers is None:
        return None
    exhausted: list[float] = []
    all_resets: list[float] = []
    for _label, reset_header, remaining_header in _RATELIMIT_RESET_HEADERS:
        reset = _parse_ratelimit_duration(headers.get(reset_header))
        if reset is None:
            continue
        all_resets.append(reset)
        remaining = _parse_ratelimit_duration(headers.get(remaining_header))
        if remaining is not None and remaining <= 0:
            exhausted.append(reset)
    if exhausted:
        return max(exhausted)
    if all_resets:
        return max(all_resets)
    return None


def _coerce_optional_bool(value: Any) -> bool:
    """Coerce a config value to a bool, tolerating YAML/env string forms."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _get_respect_ratelimit_headers(
    config: dict | None, provider_cfg: dict | None = None
) -> bool:
    """Return whether ``x-ratelimit-*`` headers should drive 429 cooldowns.

    The per-provider entry flag takes precedence over the global
    ``server.respect_ratelimit_headers`` setting (a flat top-level key is also
    accepted for unit-test convenience).  Defaults to ``False`` so existing
    providers are unaffected (LP-0MVBHDNLB007ALLJ AC6).
    """
    if isinstance(provider_cfg, dict):
        value = provider_cfg.get("respect_ratelimit_headers")
        if value is not None:
            return _coerce_optional_bool(value)
    if isinstance(config, dict):
        value = config.get("respect_ratelimit_headers")
        if value is not None:
            return _coerce_optional_bool(value)
        server_cfg = config.get("server")
        if isinstance(server_cfg, dict):
            value = server_cfg.get("respect_ratelimit_headers")
            if value is not None:
                return _coerce_optional_bool(value)
    return False


def _get_ratelimit_cooldown_margin_seconds(config: dict | None) -> float:
    """Read the safety margin added to a header-derived cooldown.

    Supports both the nested production form
    (``server.ratelimit_cooldown_margin_seconds``) and a flat key for unit
    tests.  Defaults to :data:`_RATELIMIT_COOLDOWN_MARGIN_SECONDS`; a missing,
    malformed or negative value falls back to the default.
    """
    default = _RATELIMIT_COOLDOWN_MARGIN_SECONDS
    if not isinstance(config, dict):
        return default
    raw = config.get("ratelimit_cooldown_margin_seconds")
    if raw is None:
        server_cfg = config.get("server")
        if isinstance(server_cfg, dict):
            raw = server_cfg.get("ratelimit_cooldown_margin_seconds")
    if raw is None or isinstance(raw, bool):
        return default
    try:
        margin = float(raw)
    except (TypeError, ValueError):
        return default
    if margin < 0:
        return default
    return margin


def _apply_ratelimit_account_quarantine(
    response: Response,
    body_text: str,
    provider_cfg: dict,
    ratelimit_reset: float | None,
) -> float | None:
    """Quarantine the failing API-key ACCOUNT for an upstream reset window.

    A quota 429 (a per-account rate-limit error or an explicit
    ``x-ratelimit-over-limit`` flag) that also carries a parseable
    ``x-ratelimit-reset-*`` duration is evidence the upstream will not serve
    that ACCOUNT until the reported reset. The ENTRY-level cooldown applied by
    :func:`_handle_http_error_with_cooldown` (reset + the 2 s
    ``_RATELIMIT_COOLDOWN_MARGIN_SECONDS``) is only in-memory for that entry;
    recording the reset as an explicit ACCOUNT quarantine here makes the wait
    survive the request, feeds :func:`_compute_retry_after` and the chain-hold,
    and is persisted to ``provider-state.json`` (LP-0MVFA1VRT007T9HT).

    The quarantine is keyed on :func:`_usage_limit_account_key`
    (``api_key_env@endpoint:model``), so a paid sibling on the same gateway
    with an independent quota is never poisoned. ``explicit=True`` keeps it
    exempt from the guessed-quarantine 24 h cap and the self-healing probe.

    Returns the applied quarantine duration (parsed reset plus the existing
    2-minute :data:`_USAGE_LIMIT_RESET_MARGIN_SECONDS`) when a quarantine was
    recorded, otherwise ``None`` (the caller then keeps the pre-existing
    behaviour untouched).
    """
    if ratelimit_reset is None:
        return None
    if int(getattr(response, "status_code", 0) or 0) != 429:
        return None
    if not (
        _is_account_rate_limit_error(response, body_text)
        or _ratelimit_over_limit(response.headers)
    ):
        return None
    quarantine_seconds = ratelimit_reset + _USAGE_LIMIT_RESET_MARGIN_SECONDS
    _set_usage_limit_quarantine(
        _usage_limit_account_key(provider_cfg),
        _UsageLimitReset(quarantine_seconds, True, "x-ratelimit-headers"),
    )
    return quarantine_seconds


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


def _get_empty_response_max_cooldown_seconds(config: dict) -> float:
    """Read ``empty_response_max_cooldown_seconds`` from config.

    Checks ``config[\"empty_response_max_cooldown_seconds\"]`` (flat) first
    for backward compatibility with unit tests, then falls back to
    ``config[\"server\"][\"empty_response_max_cooldown_seconds\"]`` (nested)
    for production configs.  Defaults to 10.
    """
    val = config.get("empty_response_max_cooldown_seconds")
    if val is None:
        val = config.get("server", {}).get(
            "empty_response_max_cooldown_seconds",
            _EMPTY_RESPONSE_MAX_COOLDOWN_SECONDS,
        )
    return float(val)


def _get_sibling_fallback_threshold(config: dict) -> int:
    """Read the sibling-fallback failure threshold from config.

    After this many consecutive ``empty_response`` or ``stall_after_content``
    failures on the same provider, the provider is marked unavailable for an
    extended cooldown (``sibling_fallback_cooldown_seconds``) so the retry
    cycle permanently skips it and tries siblings.

    Supports both nested (``server.sibling_fallback_*``) and flat keys.
    Defaults: threshold=2.
    """
    server_cfg = config.get("server", {}) if isinstance(config, dict) else {}
    val = server_cfg.get("sibling_fallback_threshold")
    if val is None:
        val = config.get("sibling_fallback_threshold", _DEFAULT_SIBLING_FALLBACK_THRESHOLD)
    try:
        return max(1, int(val or _DEFAULT_SIBLING_FALLBACK_THRESHOLD))
    except (ValueError, TypeError):
        return _DEFAULT_SIBLING_FALLBACK_THRESHOLD


def _get_sibling_fallback_cooldown_seconds(config: dict) -> float:
    """Read the sibling-fallback extended cooldown from config.

    Applied when the consecutive-failure threshold is exceeded.
    Supports both nested (``server.sibling_fallback_cooldown_seconds``)
    and flat keys. Defaults to 600 (10 minutes).
    """
    server_cfg = config.get("server", {}) if isinstance(config, dict) else {}
    val = server_cfg.get("sibling_fallback_cooldown_seconds")
    if val is None:
        val = config.get(
            "sibling_fallback_cooldown_seconds",
            _DEFAULT_SIBLING_FALLBACK_COOLDOWN_SECONDS,
        )
    try:
        return max(60.0, float(val or _DEFAULT_SIBLING_FALLBACK_COOLDOWN_SECONDS))
    except (ValueError, TypeError):
        return _DEFAULT_SIBLING_FALLBACK_COOLDOWN_SECONDS


def _get_sibling_fallback_window_seconds(config: dict) -> int:
    """Read the sibling-fallback streak window from config.

    If a provider has no failures for longer than this window, the
    consecutive-failure streak resets.
    Supports both nested (``server.sibling_fallback_window_seconds``)
    and flat keys. Defaults to 600 (10 minutes).
    """
    server_cfg = config.get("server", {}) if isinstance(config, dict) else {}
    val = server_cfg.get("sibling_fallback_window_seconds")
    if val is None:
        val = config.get(
            "sibling_fallback_window_seconds",
            _DEFAULT_SIBLING_FALLBACK_WINDOW_SECONDS,
        )
    try:
        return max(60, int(val or _DEFAULT_SIBLING_FALLBACK_WINDOW_SECONDS))
    except (ValueError, TypeError):
        return _DEFAULT_SIBLING_FALLBACK_WINDOW_SECONDS


def _record_sibling_failure(
    provider_name: str,
    config: dict,
    brand: str | None = None,
    domain_key: str | None = None,
    provider_type: str | None = None,
) -> bool:
    """Record a consecutive empty_response/stall failure for *provider_name*.

    Increments the streak counter. If the streak exceeds the configured
    threshold within the sliding window, marks the provider unavailable
    for an extended cooldown and returns ``True``.  On any other case
    returns ``False``.

    The streak resets when a successful response occurs (via
    ``_reset_sibling_failure_count()``) or when the window expires
    without failures.

    When *domain_key* is provided (the entry's model-scoped failure domain,
    e.g. ``https://opencode.ai/zen/go:deepseek-v4-flash``), the extended
    cooldown is applied to the entry name AND that domain. ``_entry_cooldown_key``
    checks the domain of every entry, so quarantining the MODEL-scoped domain
    blocks same-model same-gateway siblings without over-quarantining other
    upstream models on the same gateway (LP-0MTVMB8DW0067H9A AC4).

    When *brand* is provided and no *domain_key* is available (entries with no
    endpoint/model), the extended cooldown is applied to BOTH the entry name
    and the brand (legacy behavior, LP-0MTPMF03P0046MFG).

    Args:
        provider_name: Provider entry name (e.g. ``"opencode-go"``).
        config: Server configuration (to read threshold/window).
        brand: Optional provider brand (``provider_cfg.get("provider")``)
            shared by same-gateway sibling entries.
        domain_key: Optional model-scoped failure-domain key
            (``_failure_domain_key(provider_cfg)``). When present, it is
            quarantined instead of *brand* so other models on the same
            gateway stay eligible (LP-0MTVMB8DW0067H9A AC4).
        provider_type: Optional provider type (``"local"`` / ``"remote"``).
            ``"local"`` exempts the provider from the sibling-fallback
            breaker entirely (no streak increment, no quarantine); ``None``
            and ``"remote"`` both keep the remote behaviour so existing
            callers are unchanged (LP-0MUNTOGLB005LUKY AC1/AC2).

    Returns:
        ``True`` if the extended cooldown threshold was exceeded.
    """
    # AC4 (LP-0MUNTOGLB005LUKY): local providers are exempt from the
    # sibling-fallback circuit breaker.  The 600s quarantine was designed
    # for remote siblings sharing a dead gateway — it has no meaning for
    # a local entry (no sibling keys), and when it fires it removes the
    # only healthy local capacity, forcing requests onto exhausted remote
    # providers.  Local providers keep their existing normal handling:
    # empty_response_max_cooldown_seconds cap (10s), slot-exhaustion
    # retry, no-progress watchdog, and dispatch lease.
    if provider_type == "local":
        return False

    now = time.monotonic()
    window = _get_sibling_fallback_window_seconds(config)

    # Get or initialize the streak count
    count = _sibling_failure_count.get(provider_name, 0)

    # Initialize streak start if this is the first failure
    if provider_name not in _sibling_failure_streak_start:
        _sibling_failure_streak_start[provider_name] = now

    # Check if the previous streak has expired (no failures for > window)
    streak_start = _sibling_failure_streak_start.get(provider_name, 0)
    if now - streak_start > window:
        # Streak expired — start fresh
        count = 0
        _sibling_failure_streak_start[provider_name] = now

    # Increment the count
    count += 1
    _sibling_failure_count[provider_name] = count

    # Check against threshold
    threshold = _get_sibling_fallback_threshold(config)
    if count >= threshold:
        cooldown = _get_sibling_fallback_cooldown_seconds(config)
        mark_provider_unavailable(provider_name, cooldown)
        # Also quarantine the shared failure domain so same-model siblings are
        # not retried via other API keys. Prefer the MODEL-scoped domain
        # (``endpoint:model``) over the whole brand so healthy models on the
        # same gateway are not over-quarantined (LP-0MTVMB8DW0067H9A AC4).
        # Entries without an endpoint/model fall back to the brand (legacy
        # behavior, LP-0MTPMF03P0046MFG).
        quarantine_key = domain_key or brand
        if quarantine_key and str(quarantine_key) != provider_name:
            mark_provider_unavailable(str(quarantine_key), cooldown)
        logger.warning(
            "Sibling-fallback circuit breaker triggered: "
            "provider=%s brand=%s consecutive_failures=%d "
            "threshold=%d extended_cooldown=%ds",
            provider_name,
            brand or provider_name,
            count,
            threshold,
            cooldown,
        )
        return True

    return False


def _reset_sibling_failure_count(provider_name: str) -> None:
    """Reset the consecutive sibling-failure count for a provider on success.

    Removes both the count and streak start from their dicts so that
    the next failure starts a fresh streak.

    Args:
        provider_name: Provider entry name.
    """
    _sibling_failure_count.pop(provider_name, None)
    _sibling_failure_streak_start.pop(provider_name, None)


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


# Cumulative chain-hold budget across all cycles (see
# ``_get_chain_hold_total_seconds``): strictly below the client's 900 s
# queue-timeout deadline so an exhausted chain can never hold a request to
# (or past) the client limit.
_CHAIN_HOLD_TOTAL_DEFAULT = 720.0


def _get_chain_hold_total_seconds(config: dict) -> float:
    """Read ``chain_hold_total_seconds`` from config (flat or ``server.*``).

    Cumulative wall-clock budget for ALL chain-hold waits across cycles
    (streaming + non-streaming). ``chain_hold_seconds * chain_hold_max_cycles``
    can otherwise reach exactly the client's 900-second queue-timeout deadline
    (300 s x 3 = 900 s) — the AC1 gap from the LP-0MU1S617L002Q9PR audit. The
    default 720 s leaves a 180 s margin below that deadline so the proxy always
    responds before the client gives up. 0 disables the cap (legacy
    cycle-bounded behaviour).
    """
    val = config.get("chain_hold_total_seconds")
    if val is None:
        val = config.get("server", {}).get(
            "chain_hold_total_seconds", _CHAIN_HOLD_TOTAL_DEFAULT
        )
    try:
        return max(0.0, float(val or 0))
    except (ValueError, TypeError):
        return _CHAIN_HOLD_TOTAL_DEFAULT


def validate_chain_hold_config(config: dict) -> list[str]:
    """Validate the chain-hold configuration (LP-0MSH94Z7K007VKC9 AC5).

    Checks that ``chain_hold_seconds`` and ``chain_hold_max_cycles`` (flat or
    ``server.*``) are non-negative numbers when present, and flags the
    pathological combination of a zero hold interval with an unlimited cycle
    count (an unbounded busy-retry loop).

    ``chain_hold_total_seconds`` (LP-0MV6BR3HI00957PF) must be a non-negative
    number and, when non-zero, must sit strictly below the client's 900-second
    queue-timeout deadline — a budget at or above 900 s recreates the exact
    race the bound exists to prevent.

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

    total = config.get("chain_hold_total_seconds")
    if total is None:
        total = server_cfg.get("chain_hold_total_seconds")
    if total is not None:
        try:
            total_f = float(total)
            if total_f < 0:
                problems.append(
                    f"server.chain_hold_total_seconds must be >= 0 (got {total!r})"
                )
            elif total_f > 0 and total_f >= 900:
                problems.append(
                    "server.chain_hold_total_seconds must be strictly below "
                    f"the client's 900 s queue-timeout deadline (got {total!r})"
                )
        except (ValueError, TypeError):
            problems.append(
                f"server.chain_hold_total_seconds must be a number (got {total!r})"
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


def _resolve_display_provider_name(provider_cfg: dict) -> str | None:
    """Resolve the human-readable provider component used in X-Resolved-Model.

    Priority: ``display_name`` (opt-in tier-aware label, e.g. ``MunitoAI
    Free`` / ``OpenCode Go 3``) -> ``provider`` (actual provider brand name)
    -> ``name`` (provider entry name). The ``display_name`` field lets
    operators distinguish provider tiers (free vs paid; same-brand accounts
    such as the OpenCode Go tiers -- LP-0MVCQ36QV009QX8K, LP-0MVDUW7OB006UT62)
    that share a brand + model, without touching failure-domain, cooldown,
    or usage-limit keys (which key on endpoint, model, api_key_env, brand
    and entry name upstream of this display string).
    """
    return (
        provider_cfg.get("display_name")
        or provider_cfg.get("provider")
        or provider_cfg.get("name")
    )


def _build_resolved_model_value(provider_cfg: dict) -> str | None:
    """Build the X-Resolved-Model header value from a provider config.

    Returns ``<provider-name>/<model-id>`` or ``None`` if the config
    doesn't have the required fields.

    For local providers, uses ``llama_model`` as the model ID.
    For remote providers, uses ``model`` (upstream model ID).

    The provider name is resolved by ``_resolve_display_provider_name()``:
    ``display_name`` first (opt-in tier-aware label, e.g. ``MunitoAI Free``),
    then the ``provider`` field (actual provider brand name), falling back
    to ``name`` (provider entry name) for backward compatibility.  A warning
    is logged when a remote provider entry lacks the ``provider`` field.
    """
    provider_name = _resolve_display_provider_name(provider_cfg)
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


# Stable machine-readable exhaustion codes (LP-0MU56ZM0K005F69G). Clients must
# discriminate on ``code`` (never on the human-readable ``error``/``detail``
# prose); the existing ``error`` text is retained for older clients.
EXHAUSTION_CODE_ALL = "all_exhausted"
EXHAUSTION_CODE_TIME_WINDOW = "outside_time_window"
EXHAUSTION_CODE_SLOTS = "all_slots_exhausted"
EXHAUSTION_CODE_LOCAL_ADMISSION = "local_admission_reject"


def _build_local_admission_reject_response(
    attempt: dict[str, Any], config: dict | None = None
) -> Response:
    """Retryable response for a request the prefill-ETA guard refused.

    The guard refuses local admission *immediately* when the projected prefill
    cannot finish inside the caller's wait budget, and the request normally
    falls through to a remote provider. When the rest of the chain is
    exhausted as well, the client gets this distinguishable 429 instead of the
    generic exhaustion body: ``Retry-After`` carries how much longer the local
    prefill would need than the caller was prepared to wait, and the
    ``X-Local-Admission*`` headers expose the estimate so a dispatcher can pace
    its next attempt (LP-0MVGDUV5E005UWBQ AC3).
    """
    from proxy.prefill_eta import suggested_retry_after_seconds

    eta = float(attempt.get("eta_prefill_seconds") or 0.0)
    budget = float(attempt.get("budget_seconds") or 0.0)
    retry_after = suggested_retry_after_seconds(
        eta, budget, maximum=_max_retry_after_seconds(config)
    )
    payload: dict[str, Any] = {
        "error": "All providers exhausted",
        "code": EXHAUSTION_CODE_LOCAL_ADMISSION,
        "detail": (
            "local admission refused: projected prefill exceeds the wait budget "
            "and no remote provider is available"
        ),
        "eta_prefill_seconds": eta,
        "budget_seconds": budget,
        "retry_after": retry_after,
        "suggested_action": (
            "Retry after the advertised delay, reduce the prompt size (compact "
            "the session) so the prefill fits the local wait budget, or route "
            "the request to a remote provider."
        ),
    }
    try:
        payload["diagnostics"] = [attempt]
    except Exception:
        pass
    headers = {
        "Retry-After": str(retry_after),
        "Cache-Control": "no-store",
        "X-Local-Admission": "reject",
        "X-Local-Admission-Eta": f"{eta:.1f}",
        "X-Local-Admission-Budget": f"{budget:.1f}",
        # LP-0MVIKPCSG002B0NL AC6: the refusal is a routing outcome, not a
        # remote serve — say so on the response itself.
        "X-Local-Route": "refused",
        "X-Local-Fallback-Reason": str(
            attempt.get("reason") or "eta_exceeds_budget"
        ),
    }
    return Response(
        content=json.dumps(payload).encode("utf-8"),
        status_code=429,
        media_type="application/json",
        headers=headers,
    )


def _build_exhausted_response(all_local_slot_exhaustion: bool = False, total_slots: int = 0, unavailable_providers: dict | None = None, diagnostics: list[dict[str, Any]] | None = None, model_config: dict | None = None, config: dict | None = None) -> Response:
    """Build the response when all providers are exhausted.

    Args:
        all_local_slot_exhaustion: If ``True``, all providers exhausted due to
                                   slot exhaustion (returns HTTP 429).
                                   Otherwise, returns HTTP 503 with JSON body.
        total_slots: Total number of slots across local providers (used only
                     for the slot-exhaustion 429 body).
        unavailable_providers: Optional mapping of provider -> remaining cooldown seconds
                               to include in the 503 JSON payload for diagnostics.
        diagnostics: Optional list of per-provider attempt diagnostics (order-preserving)
        model_config: Optional model config (used to derive window-edge waits).
        config: Optional server config; its ``max_retry_after_seconds`` bounds
                the advertised wait (LP-0MV0MM5R60025XZE).

    The JSON body carries a stable ``code`` (``all_exhausted`` or
    ``all_slots_exhausted``) so clients can classify exhaustion without relying
    on the ``error`` prose, whose meaning is preserved for older clients
    (LP-0MU56ZM0K005F69G).

    A request refused by the prefill-ETA admission guard
    (LP-0MVGDUV5E005UWBQ) that then finds no remote provider yields a
    retryable 429 carrying ``Retry-After`` plus the ETA guidance headers
    instead of the generic exhaustion body, so the refusal is actionable.

    Verified end-to-end by ``proxy/tests/test_overnight_exhaustion_e2e.py``
    (LP-0MV728XBH00546FB): ``Retry-After`` / ``retry_after`` are identical,
    the stable ``code`` discriminator is present, the advertisement cap is
    applied, and the hold is bounded by the real ``retry_after``.
    """
    if all_local_slot_exhaustion:
        # total_slots may be 0 if unknown; still format per acceptance criteria.
        # JSON keeps the machine-readable ``code`` alongside the retained prose.
        slot_payload: dict[str, Any] = {
            "error": "All providers exhausted",
            "code": EXHAUSTION_CODE_SLOTS,
            "message": f"Model server busy: 0/{int(total_slots)} slots available. Retry later.",
        }
        return Response(
            content=json.dumps(slot_payload).encode("utf-8"),
            status_code=429,
            media_type="application/json",
        )

    admission_reject = _local_admission_reject_payload(diagnostics)
    if admission_reject is not None:
        return _build_local_admission_reject_response(admission_reject, config)

    retry_after = _compute_retry_after(
        unavailable_providers, model_config=model_config, config=config
    )
    if retry_after <= 0:
        # Budget-skip-only exhaustion (LP-0MVF0N9SA0071LWM): a free-tier
        # rolling-window skip records no cooldown and no usage-limit reset, so
        # ``_compute_retry_after`` has no candidate and would advertise 0.
        # Derive the wait from the window rollover instead, otherwise the
        # chain-hold treats the exhaustion as instant recovery and replays
        # every cycle inside the still-full window.
        retry_after = _budget_window_retry_after(diagnostics)
    client_retry_after = min(retry_after, _max_retry_after_seconds(config))
    payload: dict[str, Any] = {
        "error": "All providers exhausted",
        "code": EXHAUSTION_CODE_ALL,
        "retry_after": client_retry_after,
    }
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
        headers={"Retry-After": str(client_retry_after)},
    )


def _build_time_window_exhausted_response(
    attempts: list[dict[str, Any]],
    unavailable: dict[str, int],
    any_provider_tried: bool,
    model_config: dict | None = None,
    config: dict | None = None,
) -> Response | None:
    """Return a distinguishable 503 when every provider was skipped solely due
    to its configured ``available_times`` window.

    The distinguishable response is used only when time windows are the *only*
    reason nothing could be used: no provider was actually tried (no errors, no
    cooldown recorded this request), no provider is currently in cooldown, and
    every recorded skip is an ``outside_time_window`` skip. A usage-limit
    quarantine, a local/compaction skip, an error or any other non-window
    reason means the window is not the sole cause (LP-0MU56ZFII004L7OO).
    Otherwise ``None`` is returned and the caller falls through to the generic
    exhausted response (whose diagnostics still expose any
    ``outside_time_window`` skips).

    The ``outside_time_window`` variant is verified end-to-end by
    ``proxy/tests/test_overnight_exhaustion_e2e.py`` (LP-0MV728XBH00546FB):
    truthful ``retry_after`` + matching header, capped advertisement, and the
    stable ``code`` discriminator.
    """
    if any_provider_tried or unavailable:
        return None
    # Sole-cause check (LP-0MU56ZFII004L7OO): require at least one window skip
    # AND that every recorded attempt is a window skip. A single
    # ``usage_limit_reset`` / local-skip / error diagnostic means the window is
    # not the only blocker and the generic exhausted response must be used.
    if not attempts or any(
        a.get("status") != "outside_time_window" for a in attempts
    ):
        return None
    # Defensive completeness check: when the chain is known, every provider
    # must have been window-skipped. A provider skipped for an unrecorded
    # reason (e.g. a router-level local/compaction skip) must not yield the
    # window-specific message either.
    if model_config is not None:
        window_names = {
            a.get("provider") for a in attempts
            if a.get("status") == "outside_time_window"
        }
        for p in (model_config.get("providers") or []):
            if isinstance(p, dict) and p.get("name") not in window_names:
                return None

    retry_after = _compute_retry_after(
        unavailable, model_config=model_config, config=config
    )
    client_retry_after = min(retry_after, _max_retry_after_seconds(config))
    payload: dict[str, Any] = {
        "error": "All providers exhausted",
        "code": EXHAUSTION_CODE_TIME_WINDOW,
        "detail": "no provider is available during the current scheduled time window",
        "retry_after": client_retry_after,
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
        headers={"Retry-After": str(client_retry_after)},
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


# Connection-ESTABLISHMENT failures poison the whole gateway: the TCP/TLS
# connection could not be made (DNS resolution, refused connection, connect
# timeout) or the wire protocol failed before a usable HTTP exchange.
# ``ReadTimeout``/``WriteTimeout`` are deliberately excluded: they can occur
# after the connection is established and may be specific to the upstream
# model, so they stay model-scoped (LP-0MTVMB8DW0067H9A).
_GATEWAY_LEVEL_ERROR_TYPES: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.RemoteProtocolError,
)


def _is_gateway_level_error(exc: Exception) -> bool:
    """Return True when *exc* indicates the gateway itself is unreachable.

    A ``True`` result means failure-domain exclusion must use the
    endpoint-wide scope (``_gateway_domain_key``): every provider entry
    sharing the endpoint is skipped for the rest of the request, regardless
    of upstream model (LP-0MTVMB8DW0067H9A AC1).

    ``ConnectError`` covers DNS resolution failures and refused connections;
    ``ConnectTimeout`` covers TCP/TLS connect timeouts; ``RemoteProtocolError``
    covers malformed/closes-before-response wire failures. ``ReadTimeout`` is
    intentionally NOT included — it is ambiguous (connect vs post-connect) and
    is treated as request-level/model-scoped (AC2).
    """
    return isinstance(exc, _GATEWAY_LEVEL_ERROR_TYPES)


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


def _local_concurrency_info(config: dict, endpoint: str | None = None) -> tuple:
    """Per-endpoint-aware wrapper around ``_get_local_concurrency_info``.

    Tests monkeypatch ``_get_local_concurrency_info`` with callables that only
    accept ``(config)``; passing the endpoint kwarg would break them. This
    helper inspects the callable signature and only forwards the endpoint when
    it is accepted (LP-0MRPILSMW004T4H8).
    """
    fn = _get_local_concurrency_info
    try:
        sig = inspect.signature(fn)
        accepts_endpoint = (
            "endpoint" in sig.parameters
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        )
    except Exception:
        accepts_endpoint = False
    if accepts_endpoint and endpoint is not None:
        return fn(config, endpoint=endpoint)
    return fn(config)


def _dispatch_local(ptr_local, request, path, endpoint: str | None):
    """Dispatch to a local backend, passing the endpoint when supported.

    The endpoint argument was added for multi-backend support
    (LP-0MRPILSMW004T4H8). Legacy implementations / test mocks that accept
    only ``(request, path)`` keep working by falling back to the default
    endpoint resolution inside ``proxy_to_local``.
    """
    try:
        sig = inspect.signature(ptr_local)
        accepts_endpoint = "endpoint" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except Exception:
        accepts_endpoint = False
    if accepts_endpoint:
        return ptr_local(request, path, endpoint=endpoint)
    return ptr_local(request, path)


def _get_local_concurrency_info(config: dict, endpoint: str | None = None) -> tuple:
    """Lazily import and return (current_generating_active, max_local) from config.

    Generating-only pool (LP-0MTH7JX82000YS5N): occupancy is measured as the
    number of sessions currently in the generating phase (first-byte onward).
    Prefill time does not count against session_slot_pool_size.

    When *endpoint* is provided (a full ``http://host:port`` URL), the active
    count is scoped to that specific llama-server instance via its
    per-endpoint dispatch records, so each server's slot pool is evaluated
    independently (LP-0MRPILSMW004T4H8). When it is None (legacy
    single-server mode), the global ``local_generating_queries`` counter is
    used.

    Returns the current generating-only count and the configured
    local concurrency limit.  Reads ``session_slot_pool_size`` (same value
    that controls ``--parallel`` in llama-server); the legacy
    ``local_max_concurrent_queries`` fallback was removed (LP-0MTCZ35X7009IZKE)
    and a missing value is caught at launch by
    ``validate_local_routing_config``.  Defaults to (0, 1) on error.

    """
    cur_active = 0
    max_local = 1
    try:
        import proxy.server as _srv
        if endpoint:
            # Per-endpoint active count: count active/unexpired per-endpoint
            # dispatch records for this endpoint.
            records = getattr(_srv, "local_dispatch_records", None)
            # The legacy single-server default URL (used when a ``type:
            # local`` provider declares no explicit endpoint). Plain-key
            # records (created by the pre-multi-backend code paths) occupy
            # this default server (LP-0MRPILSMW004T4H8 backward compat).
            try:
                _server_cfg = config.get("server", config) if isinstance(config, dict) else {}
                default_ep = "http://localhost:{}".format(
                    int(_server_cfg.get("llama_server_port", 8080) or 8080)
                )
            except (TypeError, ValueError):
                default_ep = "http://localhost:8080"
            if isinstance(records, dict):
                if endpoint == default_ep and not any(
                    isinstance(k, tuple) for k in records
                ):
                    # Pure legacy state (records keyed by plain session ids):
                    # the global generating counter is authoritative and must
                    # gate exactly as it did before multi-backend support.
                    try:
                        cur_active = max(0, int(getattr(_srv, 'local_generating_queries', 0) or 0))
                    except Exception:
                        cur_active = max(0, int(getattr(_srv, 'local_active_queries', 0) or 0))
                else:
                    now = time.monotonic()
                    for key, record in records.items():
                        if isinstance(key, tuple) and len(key) == 2 and key[0] == endpoint:
                            if record.get("active") or record.get("expires_at", 0) > now:
                                cur_active += 1
                        elif not isinstance(key, tuple) and endpoint == default_ep:
                            # Legacy plain-key record occupies the default
                            # server's pool alongside tuple-keyed records.
                            if record.get("active") or record.get("expires_at", 0) > now:
                                cur_active += 1
            else:
                try:
                    cur_active = max(0, int(getattr(_srv, 'local_generating_queries', 0) or 0))
                except Exception:
                    cur_active = max(0, int(getattr(_srv, 'local_active_queries', 0) or 0))
        else:
            try:
                cur_active = max(0, int(getattr(_srv, 'local_generating_queries', 0) or 0))
            except Exception:
                cur_active = max(0, int(getattr(_srv, 'local_active_queries', 0) or 0))

    except Exception:
        pass
    try:
        server_cfg = config.get("server", config)
        max_local = max(1, int(server_cfg.get("session_slot_pool_size", 1) or 1))
    except (ValueError, TypeError):
        pass
    return (cur_active, max_local)


def _local_slot_saturation(
    config: dict, max_local: int, endpoint: str | None = None
) -> str | None:
    """Which local occupancy gate is saturated, or None when a slot is free.

    LP-0MU498TWH008DBCI: the contention queue wake predicate must account for
    BOTH occupancy gates that ``_try_acquire_local_dispatch`` enforces:

    - ``"generating"`` — the generating-only pool is at capacity
      (LP-0MTH7JX82000YS5N).
    - ``"prefill"`` — the prefill-in-flight count is at the slot-pool cap
      (the router's prefill-aware guard, LP-0MTJET4I5009EHNX).
    - ``None`` — at least one slot is free for a new dispatch.

    Generating is checked first so a busy generating pool (the common case)
    is labelled correctly. The prefill counter is global (keyed by session),
    so it is not endpoint-scoped.
    """
    try:
        cur_local, _ = _local_concurrency_info(config, endpoint=endpoint)
        if cur_local >= max_local:
            return "generating"
    except Exception:
        pass
    # Prefill-aware guard (LP-0MTJET4I5009EHNX): a prefill in flight holds a
    # slot; never admit more concurrent prefills than the prefill cap.
    # Serialised prefills (LP-0MVGDUUV3001XFRJ): the cap is
    # ``local_max_concurrent_prefills`` (default 1) rather than the slot pool,
    # because concurrent prefills halve per-request throughput at flat
    # aggregate tok/s.
    try:
        import proxy.server as _srv
        from proxy.router_helpers import (
            _effective_prefill_cap,
            _get_prefill_in_flight_count,
        )

        server_cfg = config.get("server", config) if isinstance(config, dict) else {}
        if _get_prefill_in_flight_count(_srv) >= _effective_prefill_cap(
            server_cfg, max_local
        ):
            return "prefill"
    except Exception:
        pass
    return None


def _local_slot_free(
    config: dict, max_local: int, endpoint: str | None = None
) -> bool:
    """Whether a local slot is free, accounting for prefill-in-flight.

    A predicate that only watches the generating count would wake a queued
    request immediately while a prefill is in flight (generating == 0),
    re-dispatch it, be denied by the router's prefill guard and fall through
    to a remote provider without ever waiting for the prefill to finish
    (LP-0MU498TWH008DBCI).
    """
    return _local_slot_saturation(config, max_local, endpoint=endpoint) is None


# ---------------------------------------------------------------------------
# Bounded cross-session contention queue (LP-0MSORQVK50012Q4D)
# ---------------------------------------------------------------------------

def _contention_queue_enabled(config: dict, priority: str | None = None) -> bool:
    """True when the contention queue should engage for *priority* (AC3).

    The queue policy was a per-mode knob known to the active mode config
    (``contention_queue_policy``; the retired config-fast.yaml /
    config-cheap.yaml both declared ``queue`` depth 8 / wait 120s). With the
    mode axis removed the policy is a server-level knob and priority gates
    entry:

    - ``high``  — never queues (immediate remote fallback when local slots
      are busy).
    - ``low``   — always queues (bounded by the configured caps): prefer a
      local slot over a paid remote call.
    - ``medium`` — today's behaviour: engages when the merged base config's
      ``contention_queue_policy`` is ``queue``.

    The configured depth/wait caps always bound the queue. Fail-open: any
    error → queue disabled (a broken policy never parks a request).
    """
    try:
        from proxy.router import _get_contention_queue_config

        if priority is not None:
            profile = priority_routing_profile(priority, config.get("server", config))
            if priority in (PRIORITY_HIGH, PRIORITY_LOW):
                return bool(profile.get("queue"))
        server_cfg = config.get("server", config) if isinstance(config, dict) else {}
        cq = _get_contention_queue_config(server_cfg)
        if cq["policy"] != "queue":
            return False
        return True
    except Exception:
        return False


async def _effective_routing_tokens(
    config: dict,
    model_config: dict,
    request,
    body_json: dict,
) -> int:
    """Authoritative prompt-token estimate shared by the local routing gates.

    Uses the same pipeline as the smart-routing block below — native tokenizer
    (``_get_tokenizer_for_model``) + session history + ``token_estimate_multiplier``
    — so the contention-queue context bypass and the prefill-ETA admission
    guard cost the prompt identically (LP-0MSEQ71IF0003FRT).

    Fail-open: an estimate failure yields 0 ("no prompt to measure"), which
    keeps every gate open — an estimate error must never drop local from the
    chain.
    """
    try:
        _tokenizer, _multiplier = _get_tokenizer_for_model(model_config, config)
        _tokens = await _estimate_effective_prompt_tokens_for_routing(
            request, body_json, tokenizer=_tokenizer,
        )
        if _multiplier != 1.0:
            _tokens = int(_tokens * _multiplier)
        return max(0, int(_tokens or 0))
    except Exception:
        logger.debug(
            "routing token estimate failed; gates fail open", exc_info=True
        )
        return 0


async def _queue_context_bypass(
    config: dict,
    model_config: dict,
    provider_cfg: dict,
    request,
    body_json: dict,
    session_id: str | None,
    priority: str | None = None,
    estimated_tokens: int | None = None,
) -> tuple[bool, str | None]:
    """Mirror of the smart-routing large-context skip decision.

    Used ONLY at the contention-queue decision point.  Priority-gated (AC3):
    for ``low`` (cheap semantics) ``large_context_bypass`` is recovered (the
    request proceeds to local and may queue); ``high`` and ``medium`` keep
    today's behaviour where only ``context_too_large`` is a hard bypass
    (LP-0MU5A4QBR003YJM0).  Keeps the same
    thresholds/tokenizer/estimate/``_should_skip_local`` pipeline as the main
    smart-routing block below; returns ``(skip_local, skip_reason)``.

    *estimated_tokens* is the caller's already-computed estimate
    (``_effective_routing_tokens``) so the same queue decision point does not
    pay for the tokenizer twice; ``None`` recomputes it here.
    """
    _llama_model = provider_cfg.get("llama_model", "")
    _cold_threshold, _warm_threshold = _effective_large_context_thresholds(config)
    # A LOCAL entry's ``max_context_tokens`` (LP-0MVF9N2QL001CDA7 AC7) composes
    # as an additional hard cap here too, so a request the local gate will skip
    # is never queued for a local slot (AC6 queue-safety).
    _local_context_limit = _provider_context_limit(provider_cfg)
    if _local_context_limit > 0:
        _warm_threshold = (
            min(_warm_threshold, _local_context_limit)
            if _warm_threshold > 0
            else _local_context_limit
        )
    if estimated_tokens is None:
        _estimated_tokens = await _effective_routing_tokens(
            config, model_config, request, body_json
        )
    else:
        _estimated_tokens = max(0, int(estimated_tokens))
    # Pass the priority-driven recovery flag (AC3): low priority recovers
    # economic bypasses locally even when the base config flag is off.
    _recover_economic = _economic_bypass_serves_local(config, priority)
    _skip_local, _skip_reason = _should_skip_local(
        _llama_model, session_id, body_json, _cold_threshold,
        estimated_tokens=_estimated_tokens,
        warm_cache_threshold=_warm_threshold,
        economic_bypass_serves_local=_recover_economic,
    )
    return _skip_local, _skip_reason


def _local_gate_occupancy(
    config: dict, max_local: int, endpoint: str | None = None
) -> tuple[int, int]:
    """Return ``(busy_slots, in_flight_prefills)`` for this request's gate.

    LP-0MVICD2YW004DJPV: the admission guard must compare the wait *including
the occupied gate ahead of it* against the budget, so it needs the number of
    occupants — not just the newcomer's own prefill. Two occupancy sources are
    combined, taking the larger (a conservative over-estimate is the safe
    direction, and both can be true at once):

    - the generating concurrency from ``_local_concurrency_info`` (slots that
      will hold the gate while they decode), and
    - the prefill-in-flight count (``_get_live_prefill_in_flight_count``), which is
      the signal that remains when per-slot prefill progress is unobservable
      (``prefill_progress_unobservable``, ``alive=False``). An in-flight
      prefill must never be read as "free soon".

    LP-0MVKUKFSO008GI6O AC3: only in-flight entries *backed by a live
    dispatch record* are counted — an orphaned marker (record released by the
    prefill-deadline release, a non-explicit cleanup, an orphan sweep or a
    session eviction) must not raise ``busy_slots`` for the rest of the pool.
    The 2026-10-23 incident shows a single stuck marker reporting
    ``busy_slots=1 in_flight_prefills=1; available_slots=0`` for every other
    session.

    Never raises: a routing decision must not break on an observability gap.
    """
    in_flight = 0
    try:
        import proxy.server as _srv
        from proxy.router_helpers import _get_live_prefill_in_flight_count

        in_flight = max(0, int(_get_live_prefill_in_flight_count(_srv) or 0))
    except Exception:
        in_flight = 0
    cur_local = 0
    try:
        cur_local, _ = _local_concurrency_info(config, endpoint=endpoint)
        cur_local = max(0, int(cur_local or 0))
    except Exception:
        cur_local = 0
    try:
        pool = max(1, int(max_local or 1))
    except (TypeError, ValueError):
        pool = 1
    # Never claim more occupants than the gate can physically hold: a stale
    # counter must not multiply the occupancy term without bound.
    busy = min(pool, max(cur_local, in_flight))
    return busy, in_flight


def _note_local_fallback_reason(request, reason: str | None) -> None:
    """Record why local was not used, for ``X-Local-Fallback-Reason``.

    LP-0MVIKPCSG002B0NL AC6: a client must be able to observe its loss at the
    moment it happens. The reason is stamped on the request as the routing
    cycle falls back and surfaced by ``_stamp_local_route_headers`` at the
    top-level return, so every fallback path is covered by one choke point.
    Never raises: header bookkeeping must not break routing.
    """
    try:
        setattr(request, "_local_fallback_reason", reason)
    except Exception:
        pass


_PREF_LOCAL_ONLY = "local-only"
_PREF_REMOTE_OK = "remote-ok"


def _request_local_preference(request) -> str:
    """Return the request's ``X-Local-Preference`` (default ``prefer-local``).

    Unknown values fall back to today's behaviour (AC8). The parsed value is
    cached on the request so both routing cycles and the refusal path agree.
    """
    cached = getattr(request, "_local_preference", None)
    if cached:
        return cached
    try:
        from proxy import admission_reserve

        preference = admission_reserve.parse_preference(
            getattr(request, "headers", {}) or {}
        )
    except Exception:
        preference = "prefer-local"
    try:
        setattr(request, "_local_preference", preference)
    except Exception:
        pass
    return preference


def _build_local_only_refusal_response(
    attempt: dict[str, Any] | None, config: dict | None = None
) -> Response:
    """429 for a ``local-only`` request the admission guard refused (AC4).

    Unlike the exhaustion path, this response is produced **while remote
    providers are still available**: the client said local-only, so moving it
    remote would silently violate its intent. It carries the same retryable
    contract as the exhaustion refusal (``Retry-After``, ``X-Local-Admission*``)
    plus ``X-Local-Route: refused``.
    """
    from proxy.prefill_eta import suggested_retry_after_seconds

    attempt = attempt or {}
    eta = float(attempt.get("eta_prefill_seconds") or 0.0)
    budget = float(attempt.get("budget_seconds") or 0.0)
    projected = float(attempt.get("projected_wait_seconds") or eta)
    reason = str(attempt.get("reason") or "eta_exceeds_budget")
    retry_after = attempt.get("retry_after_s")
    if retry_after is None:
        retry_after = suggested_retry_after_seconds(
            eta, budget, maximum=_max_retry_after_seconds(config)
        )
    payload: dict[str, Any] = {
        "error": "Local admission refused",
        "code": EXHAUSTION_CODE_LOCAL_ADMISSION,
        "detail": (
            "local admission refused: the projected wait exceeds the wait "
            "budget and X-Local-Preference: local-only forbids remote fallback"
        ),
        "reason": reason,
        "eta_prefill_seconds": eta,
        "projected_wait_seconds": projected,
        "budget_seconds": budget,
        "retry_after": retry_after,
        "suggested_action": (
            "Retry after the advertised delay, reduce the prompt size so the "
            "prefill fits the local wait budget, or send "
            "X-Local-Preference: prefer-local to allow a remote fallback."
        ),
    }
    headers = {
        "Retry-After": str(retry_after),
        "Cache-Control": "no-store",
        "X-Local-Admission": "reject",
        "X-Local-Admission-Eta": f"{eta:.1f}",
        "X-Local-Admission-Budget": f"{budget:.1f}",
        "X-Local-Route": "refused",
        "X-Local-Fallback-Reason": reason,
    }
    return Response(
        content=json.dumps(payload).encode("utf-8"),
        status_code=429,
        media_type="application/json",
        headers=headers,
    )


def _stamp_local_route_headers(
    response: Response, request, model_config: dict
) -> Response:
    """Add ``X-Local-Route`` / ``X-Local-Fallback-Reason`` to *response*.

    Single choke point on the top-level return path so local success, remote
    success, refusal and exhaustion are all covered (LP-0MVIKPCSG002B0NL AC6).
    A response that already carries the route header (a local-only refusal) is
    left untouched; every other response is classified by the ``X-Provider``
    that served it.
    """
    try:
        from proxy import admission_reserve

        return admission_reserve.stamp_route_headers(
            response,
            model_config,
            fallback_reason=getattr(request, "_local_fallback_reason", None),
        )
    except Exception:  # pragma: no cover - header stamping is best-effort
        logger.debug("route header stamping failed", exc_info=True)
        return response


async def _prefill_admission_guard(
    config: dict,
    provider_cfg: dict,
    estimated_tokens: int,
    session_id: str | None,
    priority: str | None = None,
    messages: list | None = None,
    max_local: int = 1,
    endpoint: str | None = None,
    reason_label: str | None = None,
    free_slot: bool = False,
):
    """Evaluate (and log) the prefill-ETA admission guard.

    Returns the :class:`proxy.prefill_eta.AdmissionDecision` when the guard
    actually decided (a calibrated ETA against a real wait budget), else
    ``None`` — an inert guard leaves today's queueing behaviour untouched
    (LP-0MVGDUV5E005UWBQ AC1/AC3). Never raises: the guard must not be able to
    break local routing.  *messages* (the request message list) lets the guard
    cost the *expected* reusable prefix (LP-0MVI2XOTY008Q4UV) rather than the
    raw measured ratio.

    The verdict is occupancy-aware (LP-0MVICD2YW004DJPV): *max_local* and
    *endpoint* let it read the occupied gate ahead of this request, so the
    projected wait it compares against the budget includes the in-flight
    prefill rather than the own-prefill term alone.

    *reason_label* / *free_slot* let the direct (free-slot) dispatch guard
    (LP-0MVIQ0F8T002FS42) reuse this one implementation and one budget source
    while logging the decision as ``eta_exceeds_budget_direct`` — the log line
    is emitted exactly once, by this helper.
    """
    try:
        from proxy import prefill_eta

        if estimated_tokens <= 0:
            return None
        server_cfg = config.get("server", config) if isinstance(config, dict) else {}
        guard_cfg = prefill_eta.resolve_config(server_cfg)
        if guard_cfg.policy == prefill_eta.POLICY_OFF:
            return None
        if prefill_eta.wait_budget_seconds(config, priority) <= 0:
            # ``high`` priority never queues, so there is no budget to guard.
            return None
        model_name = (
            provider_cfg.get("llama_model", "")
            if isinstance(provider_cfg, dict)
            else ""
        )
        server_cfg = config.get("server", config) if isinstance(config, dict) else {}
        ratio = expected_cached_ratio(
            model_name,
            session_id,
            estimated_tokens,
            messages,
            server_config=server_cfg,
        )
        # Occupancy-aware verdict (LP-0MVICD2YW004DJPV): the wait a newcomer
        # actually faces is its own prefill *plus* the occupied gate ahead of
        # it, so the guard needs the occupants before it can compare against
        # the budget.
        occupancy, in_flight_prefills = _local_gate_occupancy(
            config, max_local, endpoint=endpoint
        )
        decision = prefill_eta.evaluate_admission(
            config,
            model_name,
            estimated_tokens,
            ratio,
            priority,
            busy_slots=occupancy,
            in_flight_prefills=in_flight_prefills,
        )
        if not decision.active:
            return None
        # For the deadline refusal keep the decision's own reason so the
        # log reason is ``prefill_eta_exceeds_deadline`` on both paths
        # (LP-0MVKUKEU80022C3U AC1).  The budget refusal still gets the
        # queue/direct suffix.
        log_label = reason_label
        if reason_label and str(decision.reason).startswith(
            "prefill_eta_exceeds_deadline"
        ):
            log_label = decision.reason
        prefill_eta.record_guard_decision(
            decision,
            provider=(provider_cfg.get("name") if isinstance(provider_cfg, dict) else None),
            session_id=session_id,
            model=model_name,
            priority=priority,
            reason_label=log_label,
            free_slot=free_slot,
        )
        return decision
    except Exception:
        logger.debug(
            "prefill-ETA admission guard failed; admitting as today", exc_info=True
        )
        return None


def _local_admission_reject_payload(attempts: list | None) -> dict | None:
    """Return the ETA-guard rejection diagnostic from *attempts*, if any.

    A request refused by the prefill-ETA admission guard falls through to the
    remote chain; when that chain is exhausted too, the client must see a
    retryable status (with ``Retry-After`` and the ETA guidance headers)
    rather than a generic exhaustion error, so the refusal is recoverable
    (LP-0MVGDUV5E005UWBQ AC3).
    """
    if not attempts:
        return None
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        if attempt.get("status") == "local_admission_reject":
            return attempt
    return None


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
    endpoint: str | None = None,
    priority: str | None = None,
) -> tuple[str, str | None, float | None]:
    """Queue-wait for a local slot (low/medium priority, queue policy)
    instead of immediately falling back to the next remote provider.

    Returns ``(action, reason, elapsed_seconds)`` where action is one of:

      - ``"dispatch"`` — a slot freed within the caps; the caller should
        dispatch local. *elapsed_seconds* is the queue wait (for the Q2=a
        budget subtraction).
      - ``"fallback"`` — caps exceeded; the caller falls back to the next
        remote provider exactly as today.
      - ``"context_bypass"`` — the request cannot fit a KV slot; it was
        never queued; the caller records ``cached_tokens_skip`` with the
        returned *reason*.
      - ``"fallback_policy"`` — policy is fallback / high priority never
        queues; the caller keeps today's ``local_concurrency_limit``
        behavior.
      - ``"admission_reject"`` — the prefill-ETA admission guard refused
        local admission because the projected prefill cannot finish inside
        the wait budget; the caller falls through to the next provider
        immediately (no queueing) and records the ETA diagnostic
        (LP-0MVGDUV5E005UWBQ).
      - ``"refused_local_only"`` — the guard refused and the caller set
        ``X-Local-Preference: local-only``: the caller must return the
        retryable 429 admission response instead of falling back to a remote
        provider (LP-0MVIKPCSG002B0NL AC4).

    A valid ``X-Local-Reservation`` ticket is consumed first: a reservation is
    a pre-granted admission, so the follow-up request dispatches without
    re-checking or queueing (LP-0MVIKPCSG002B0NL AC1).
    """
    # Atomic local-admission reservation (LP-0MVIKPCSG002B0NL): a ticket
    # claimed via POST /llama/local/reserve is a pre-granted admission. It is
    # single-use and TTL-bound, so consuming it here is what makes the
    # follow-up dispatch skip both the admission re-check and the queue. A
    # stale / missing / already-consumed ticket simply falls through to the
    # ordinary path — never a hard failure (AC3).
    try:
        from proxy import admission_reserve as _reservations

        _ticket_id = _reservations.reservation_id_from_headers(
            getattr(request, "headers", {}) or {}
        )
        _ticket = _reservations.peek(_ticket_id)
        if _ticket is not None:
            # Reservations are scoped per provider/model: a ticket for another
            # local model must not fast-path (nor consume) this endpoint's
            # gate (the work item's "per-gate, scoped per provider endpoint"
            # constraint).
            _ticket_local_model = (
                provider_cfg.get("llama_model", "")
                if isinstance(provider_cfg, dict)
                else ""
            )
            if _ticket_local_model and _ticket.provider and (
                _ticket.provider != _ticket_local_model
            ):
                _ticket = None
            else:
                _ticket = _reservations.claim(_ticket_id)
    except Exception:
        _ticket = None
    if _ticket is not None:
        _note_local_fallback_reason(request, None)
        logger.info(
            "local_admission_reserve_claimed ticket=%s session=%s tokens=%d",
            _ticket.ticket_id,
            _ticket.session_id or "unknown",
            _ticket.estimated_tokens,
        )
        return ("dispatch", "reserved", 0.0)

    if not _contention_queue_enabled(config, priority):
        return ("fallback_policy", None, None)
    # One authoritative prompt estimate, shared by the context-bypass check,
    # the smart-routing mirror and the prefill-ETA guard below.
    _estimated_tokens = await _effective_routing_tokens(
        config, model_config, request, body_json
    )
    # Context bypasses never queue (AC4): a request that cannot fit the KV
    # slot must fall back exactly as today.
    skip_local, skip_reason = await _queue_context_bypass(
        config, model_config, provider_cfg, request, body_json, session_id,
        priority=priority, estimated_tokens=_estimated_tokens,
    )
    if skip_local:
        return ("context_bypass", skip_reason, None)

    # Prefill-ETA admission guard (LP-0MVGDUV5E005UWBQ): a wait that cannot
    # finish the prefill inside the budget is a guaranteed loss (measured 85 %
    # of queued waits were abandoned after up to 120 s and went remote). Refuse
    # immediately and let the caller fall through instead. Inert (``None``)
    # when uncalibrated, so today's queueing behaviour is unchanged.
    _admission = await _prefill_admission_guard(
        config,
        provider_cfg,
        _estimated_tokens,
        session_id,
        priority=priority,
        messages=body_json.get("messages") if isinstance(body_json, dict) else None,
        max_local=max_local,
        endpoint=endpoint,
    )
    if _admission is not None and not _admission.admit:
        try:
            setattr(request, "_local_admission_decision", _admission)
        except Exception:
            pass
        # Explicit client intent (LP-0MVIKPCSG002B0NL AC4): with
        # ``X-Local-Preference: local-only`` the client must never be moved
        # remote. The caller turns this into the retryable 429 admission
        # response instead of falling through to the next provider.
        _preference = _request_local_preference(request)
        if _preference == _PREF_LOCAL_ONLY:
            return ("refused_local_only", _admission.reason, None)
        return ("admission_reject", _admission.reason, None)

    from proxy.router import _get_contention_queue_config

    server_cfg = config.get("server", config) if isinstance(config, dict) else {}
    cq = _get_contention_queue_config(server_cfg)

    import time as _time

    from proxy import contention_queue

    # LP-0MU498TWH008DBCI AC5: record which gate the request waits on so
    # generating-saturation and prefill-saturation waits are distinguishable.
    # LP-0MVGDUUV3001XFRJ AC4: the saturation is stamped on the request so the
    # local-dispatch path can emit ``prefill_gate_wait_ms`` (the time this
    # request waited specifically for the prefill gate).
    _saturation = _local_slot_saturation(config, max_local, endpoint=endpoint)
    try:
        setattr(request, "_local_queue_saturation", _saturation)
    except Exception:
        pass
    logger.info(
        "contention_queue_wait max_wait=%.1fs max_depth=%d saturation=%s",
        cq["max_wait_seconds"], cq["max_depth"], _saturation or "none",
    )

    _wait_started = _time.monotonic()
    # Lease-owner retention priority (LP-0MVI2VRT2000V1AP): a session that
    # holds, or has just released, a local lease and still has cache history
    # queues ahead of newcomers, so it reclaims the slot it just left instead
    # of burning the full budget and detouring to a paid provider.
    _retained = False
    try:
        _grace = float(cq.get("lease_grace_seconds", 300) or 0)
        _retained = contention_queue.is_retained_local_owner(
            session_id,
            (
                provider_cfg.get("llama_model", "")
                if isinstance(provider_cfg, dict)
                else None
            ),
            grace_seconds=_grace,
        )
    except Exception:
        logger.debug("retention check failed; queueing as a newcomer", exc_info=True)
    elapsed = await contention_queue.wait_for_local_slot(
        max_wait_seconds=cq["max_wait_seconds"],
        max_depth=cq["max_depth"],
        slot_free_check=lambda: _local_slot_free(
            config, max_local, endpoint=endpoint
        ),
        session_id=session_id,
        retained=_retained,
        # ETA-aware ordering (LP-0MVIQ0G2M0006QF6 AC1): the admission guard
        # already costed this request's prefill, so the queue can admit the
        # shortest projected prefill first within its retention class.
        eta_seconds=(
            float(getattr(_admission, "eta_prefill_seconds", 0.0) or 0.0)
            if _admission is not None
            else None
        ),
        eta_ordering=bool(cq.get("eta_ordering", False)),
        promote_after_seconds=float(cq.get("eta_promotion_seconds", 0) or 0),
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
    *,
    config: dict | None = None,
) -> bool:
    """True if at least one more provider remains untried (and not in cooldown).

    Used to gate pre-content streaming pre-flight: re-route is only worthwhile
    when the chain has a remaining provider to fall back to (LP-0MSETOTWY000SU0Z).
    Accepts already-failed failure domains so same-gateway siblings are not
    counted as a usable fallback (LP-0MSG45I8Q0020N1F).

    Delegates to :func:`_resolve_provider_with_exclusions`, so it walks the
    same headroom-ranked chain as resolution (LP-0MVF7KIKE002DZVD AC3b) and
    must be given the same ``config``.
    """
    return (
        _resolve_provider_with_exclusions(
            model_config,
            attempted_provider_names,
            excluded_domains,
            config=config,
        )
        is not None
    )


def _failure_domain_key(provider_cfg: dict) -> str:
    """Return a canonical failure-domain key for a provider entry.

    Remote entries (with an ``endpoint``) key on the normalized endpoint URL
    PLUS the upstream ``model``: ``normalized_endpoint:model``. This allows
    router endpoints (e.g. ``https://opencode.ai/zen/go``) to distinguish
    between different upstream models — a failure for model A does not exclude
    model B on the same gateway.

    When no ``model`` is provided for a remote entry, a wildcard ``*`` is used
    so all such entries on the same endpoint still share one domain.

    Local / no-endpoint entries fall back to the ``provider`` brand, then to
    the entry name (last resort so entries without either never share a key).

    Entries that share a failure-domain key are treated as ONE failure domain:
    a stall/terminal error on one entry excludes the whole domain from the
    fallback chain / mid-stream re-route (LP-0MSG45I8Q0020N1F).
    """
    endpoint = provider_cfg.get("endpoint")
    if endpoint:
        normalized = _normalize_endpoint_for_failure_domain(str(endpoint))
        if normalized:
            model = provider_cfg.get("model") or "*"
            return f"{normalized}:{model}"
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


def _gateway_domain_key(provider_cfg: dict) -> str:
    """Return the endpoint-only failure-domain key for a provider entry.

    Unlike ``_failure_domain_key`` (which appends ``:model``), this key
    identifies the whole gateway: a connection-level failure poisons it so
    ALL entries sharing the endpoint are excluded for the rest of the request,
    regardless of upstream model (LP-0MTVMB8DW0067H9A AC1).

    Remote entries key on the normalized endpoint; local/no-endpoint entries
    fall back to the brand, then the entry name (mirroring
    ``_failure_domain_key`` so local domains stay unchanged).
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


def _is_domain_excluded(provider_cfg: dict, excluded_domains: set[str]) -> str | None:
    """Return the excluded key matching *provider_cfg*, or ``None``.

    Checks BOTH the model-scoped failure domain (``endpoint:model``) and the
    endpoint-wide gateway domain (``endpoint``): a connection-level failure
    records the latter so every model on the dead gateway is skipped
    (LP-0MTVMB8DW0067H9A AC1), while request-level failures record the former
    (AC2).
    """
    domain = _failure_domain_key(provider_cfg)
    if domain in excluded_domains:
        return domain
    gateway = _gateway_domain_key(provider_cfg)
    if gateway != domain and gateway in excluded_domains:
        return gateway
    return None


def _connection_error_domain_key(provider_cfg: dict, exc: Exception) -> str:
    """Return the failure-domain key to record for a connection exception.

    Connection-establishment failures exclude the whole gateway
    (``_gateway_domain_key``); ambiguous connection-ish failures such as
    ``ReadTimeout`` stay model-scoped (``_failure_domain_key``)
    (LP-0MTVMB8DW0067H9A AC1/AC2).
    """
    if _is_gateway_level_error(exc):
        return _gateway_domain_key(provider_cfg)
    return _failure_domain_key(provider_cfg)


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
    *,
    config: dict | None = None,
) -> dict | None:
    """Resolve next available provider while excluding names tried this request
    and any entry sharing an already-failed failure domain.

    ``excluded_domains`` holds failure-domain keys (see ``_failure_domain_key``)
    that already stalled / terminally failed during THIS request, so the chain
    skips straight past same-gateway API-key siblings (LP-0MSG45I8Q0020N1F).
    Usage-limit account keys (see ``_usage_limit_account_key``) are checked
    against the same set so an account exhausted by a usage-limit error is not
    retried via another entry sharing that account (LP-0MSMBWB23009XYPW).

    Quota-managed entries are walked in descending headroom urgency order
    (LP-0MVF7KIKE002DZVD); ``config`` supplies the ordering gates and defaults
    to the documented behaviour when omitted.
    """
    providers = _quota_ranked_providers(model_config, config)
    if not providers:
        return None

    excluded_domains = excluded_domains or set()

    for provider_cfg in providers:
        name = provider_cfg.get("name", "")
        if name in excluded_provider_names:
            continue
        matched_domain = _is_domain_excluded(provider_cfg, excluded_domains)
        if matched_domain is not None:
            logger.info(
                "Skipping provider=%s: same failure domain as an already-failed "
                "entry (domain=%s)",
                name,
                matched_domain,
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
        # Proactive quota state says this account has nothing left to spend
        # (LP-0MVF7KIKE002DZVD): treat it as unavailable.
        quota_reason = _quota_unavailable_reason(provider_cfg, config)
        if quota_reason is not None:
            logger.info("Skipping provider=%s: %s", name, quota_reason)
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
        # Quota hard stop (LP-0MVF7KO8C005NB59): at-or-near the monthly cap the
        # entry is skipped before any upstream request is attempted. Runs after
        # the usage-limit-quarantine and cooldown checks so those reasons keep
        # their precedence, and before the available_times/surge gate so the
        # final-week surge can never re-enable an exhausted account. The
        # guard is independent of the surge kill switch/dry-run.
        quota_hard_stop, quota_stop_reason = _quota_exhausted(
            provider_cfg, config
        )
        if quota_hard_stop:
            logger.info("Skipping provider=%s: %s", name, quota_stop_reason)
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
        if not _provider_in_allowed_window(provider_cfg, config):
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


def _is_account_rate_limit_error(response: Response, body_text: str) -> bool:
    """Return True when a 429 response is a per-ACCOUNT rate-limit error.

    Detects upstream responses whose JSON body identifies the rate limit as
    bound to a specific API-key account -- ``type == "rate_limit_error"`` or
    ``code == "rate_limit_exceeded"`` (case-insensitive), either top-level
    (observed api.munito.ai body) or nested under an ``error`` dict.

    A per-account rate limit is consumed by ONE API key: entries that share a
    gateway but use different ``api_key_env`` values have independent quota
    (LP-0MSMBWB23009XYPW), so the failing entry must poison only its ACCOUNT
    key in the per-request exclusion set, NOT the shared endpoint failure
    domain -- otherwise paid sibling keys on the same gateway are skipped for
    the rest of the request despite having independent quota
    (LP-0MVCX0E2V001VB5N).

    Expected upstream format (observed from api.munito.ai):
        HTTP 429
        Body: {"code": "rate_limit_exceeded", "type": "rate_limit_error"}
    """
    status = int(getattr(response, "status_code", 0) or 0)
    if status != 429:
        return False

    try:
        payload = json.loads(body_text) if body_text else None
    except Exception:
        payload = None

    if isinstance(payload, dict):
        # Check the top level first, then a nested ``error`` dict (OpenAI-style
        # bodies nest the markers under ``error``; the observed api.munito.ai
        # body is top-level).
        candidates = [payload]
        error = payload.get("error")
        if isinstance(error, dict):
            candidates.append(error)
        for cand in candidates:
            err_type = str(cand.get("type", "")).strip().lower()
            code = str(cand.get("code", "")).strip().lower()
            if err_type == "rate_limit_error" or code == "rate_limit_exceeded":
                return True

    # Fallback: check raw body text for the identifying markers
    lowered_body = (body_text or "").strip().lower()
    return "rate_limit_error" in lowered_body or "rate_limit_exceeded" in lowered_body


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


class _UsageLimitReset(NamedTuple):
    """Resolved usage-limit reset for a 429 response.

    ``explicit`` is True only when the gateway message carried an actual
    ``Resets in ...`` duration; a duration guessed from ``metadata.limitName``
    is ``explicit=False`` and is capped and treated as soft/re-checkable
    (LP-0MUQTCMW2001VXH8).
    """

    seconds: float
    explicit: bool
    limit_name: str | None


def _usage_limit_guess_cap_seconds(config: dict | None = None) -> float:
    """Return the cap applied to a guessed (period-name-only) quarantine.

    FALLBACK-ONLY (LP-0MVF7MPZN009GOJB): this cap is applied only when an
    ``opencode-*`` account has no fresh quota snapshot (stale/absent/disabled
    quota state) or the entry is not quota-managed at all; with a fresh
    snapshot the authoritative ``/v1/usage`` reset is used instead. See
    :func:`_usage_limit_reset_info` for the resolution order.

    Resolution order: ``server.usage_limit_guess_cap_seconds`` (accepting a
    flat or nested ``server:`` config), then the
    ``LLAMA_PROXY_USAGE_LIMIT_GUESS_CAP_SECONDS`` env override, then the
    24-hour default. Invalid/non-positive values fall through to the default.
    """
    candidates: list[Any] = []
    if isinstance(config, dict):
        server_cfg = config.get("server", config)
        if isinstance(server_cfg, dict):
            candidates.append(server_cfg.get("usage_limit_guess_cap_seconds"))
    candidates.append(os.environ.get(_USAGE_LIMIT_GUESS_CAP_ENV))
    for value in candidates:
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return float(_USAGE_LIMIT_GUESS_MAX_SECONDS)


def _quota_snapshot_reset_seconds(
    provider_cfg: dict | None, config: dict | None = None
) -> float | None:
    """Return the authoritative reset for *provider_cfg* from quota state.

    Only ``opencode-*`` entries are quota-managed, and the value is honoured
    only while the quota source is strictly **enabled** for routing
    (:func:`proxy.quota_state.quota_behaviour`): a disabled (kill switch),
    stale (suspended) or dry-run quota mode returns ``None`` so the reactive
    usage-limit handling keeps its pre-feature guessed/message behaviour for
    every other case (LP-0MVF7MPZN009GOJB).

    The returned value is the account-level
    :attr:`~proxy.quota_state.QuotaSnapshot.seconds_to_reset` (the monthly
    allowance window, resolved per account from ``monthly.resetsAt`` — the
    same value surge and ``available_times`` treat as authoritative) rather
    than the guessed cap from :func:`_usage_limit_guess_cap_seconds`. A
    missing/unparseable reset, an absent snapshot, a stale snapshot or a
    non-positive value all return ``None`` (fail open to the existing
    message/guess cascade). Never blocks, never polls and never raises: a
    pure cache read on the request path with a fail-safe exception guard.
    """
    if not isinstance(provider_cfg, dict):
        return None
    try:
        if not quota_state.is_quota_provider(provider_cfg, config):
            return None
        if not quota_state.quota_behaviour(config).applies:
            return None
        snapshot = quota_state.get_quota_snapshot(provider_cfg, config)
        if snapshot is None:
            return None
        seconds = snapshot.seconds_to_reset
    except Exception:
        logger.info(
            "usage-limit snapshot reset unavailable; falling back to the "
            "message/guess path",
            exc_info=True,
        )
        return None
    if seconds <= 0:
        return None
    return float(seconds)


def _usage_limit_reset_info(
    response: Response,
    body_text: str,
    config: dict | None = None,
    provider_cfg: dict | None = None,
) -> _UsageLimitReset | None:
    """Return the resolved usage-limit reset for a 429 error response.

    Recognizes ``GoUsageLimitError`` (LP-0MSLJPOCC0001ROJ) and any usage-limit
    error variant that carries a reset duration in its message (including
    ``FreeUsageLimitError`` responses that include one).

    Resolution order (LP-0MVF7MPZN009GOJB): for an ``opencode-*`` entry with a
    **fresh** quota snapshot the snapshot's authoritative ``seconds_to_reset``
    is used (monthly window, + the 2-minute margin) and returned as a hard
    ``explicit=True`` quarantine -- the guessed cap path is unreachable for
    that case. Otherwise (non-``opencode-*`` entry, or stale/absent/disabled
    quota state) the previous cascade is unchanged: an explicit
    ``Resets in 22hr 43min`` duration is honoured in full; when the message
    has no explicit duration, ``metadata.limitName`` (daily/weekly/monthly)
    supplies a GUESSED period capped at
    :func:`_usage_limit_guess_cap_seconds` (24h by default) — the full 30-day
    period is never used without explicit evidence (LP-0MUQTCMW2001VXH8). The
    2-minute safety margin is added to the returned duration. The guess path
    is therefore retained purely as the stale/absent fallback, never for a
    fresh-snapshot ``opencode-*`` account.

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

    if provider_cfg is not None:
        snapshot_seconds = _quota_snapshot_reset_seconds(provider_cfg, config)
        if snapshot_seconds is not None:
            logger.info(
                "usage-limit reset source=snapshot provider=%s account=%s "
                "seconds_to_reset=%.0f limit=%s",
                provider_cfg.get("name", ""),
                _usage_limit_account_key(provider_cfg),
                snapshot_seconds,
                limit_name or "",
            )
            return _UsageLimitReset(
                snapshot_seconds + _USAGE_LIMIT_RESET_MARGIN_SECONDS,
                True,
                limit_name,
            )

    account = (
        _usage_limit_account_key(provider_cfg)
        if isinstance(provider_cfg, dict)
        else None
    )
    seconds = _parse_resets_in(message) if message else None
    if seconds is not None:
        if account is not None:
            logger.info(
                "usage-limit reset source=message provider=%s account=%s "
                "seconds_to_reset=%.0f limit=%s",
                provider_cfg.get("name", ""),
                account,
                float(seconds),
                limit_name or "",
            )
        return _UsageLimitReset(
            float(seconds) + _USAGE_LIMIT_RESET_MARGIN_SECONDS, True, limit_name
        )
    if limit_name in _PERIOD_DEFAULT_SECONDS:
        guessed = min(
            float(_PERIOD_DEFAULT_SECONDS[limit_name]),
            _usage_limit_guess_cap_seconds(config),
        )
        if account is not None:
            logger.info(
                "usage-limit reset source=guess provider=%s account=%s "
                "seconds_to_reset=%.0f limit=%s",
                provider_cfg.get("name", ""),
                account,
                guessed + _USAGE_LIMIT_RESET_MARGIN_SECONDS,
                limit_name,
            )
        return _UsageLimitReset(
            guessed + _USAGE_LIMIT_RESET_MARGIN_SECONDS, False, limit_name
        )
    return None


def _usage_limit_reset_seconds(
    response: Response,
    body_text: str,
    config: dict | None = None,
    provider_cfg: dict | None = None,
) -> float | None:
    """Backwards-compatible wrapper returning only the reset duration.

    See :func:`_usage_limit_reset_info` for the snapshot-vs-message-vs-guessed
    detail and the optional ``provider_cfg`` snapshot preference.
    """
    info = _usage_limit_reset_info(response, body_text, config, provider_cfg)
    return info.seconds if info is not None else None


def _set_usage_limit_quarantine(
    account: str, reset: _UsageLimitReset
) -> None:
    """Record a usage-limit quarantine and persist it (best effort).

    Guessed (period-name-only) quarantines are tagged in
    ``_usage_reset_guessed`` so they are capped at read time, skipped by the
    persistence layer, and eligible for self-healing probes
    (LP-0MUQTCMW2001VXH8).
    """
    _usage_reset_at[account] = time.time() + reset.seconds
    if reset.explicit:
        _usage_reset_guessed.discard(account)
    else:
        _usage_reset_guessed.add(account)
    _persist_provider_state_best_effort()


def clear_usage_limit_quarantine(account: str | None = None) -> list[str]:
    """Clear pending usage-limit quarantine entries at runtime.

    With ``account`` set, clears just that account (a no-op when absent).
    With ``account`` ``None``, clears every pending entry. Returns the list of
    cleared account keys and persists the change (best effort). This backs the
    ``POST /admin/clear-usage-limit`` escape hatch so an operator can
    reintroduce a healthy account without restarting the proxy
    (LP-0MUQTCMW2001VXH8).
    """
    if account is None:
        cleared = list(_usage_reset_at)
        _usage_reset_at.clear()
        _usage_reset_guessed.clear()
        _usage_limit_last_probe.clear()
    elif account in _usage_reset_at:
        del _usage_reset_at[account]
        _usage_reset_guessed.discard(account)
        _usage_limit_last_probe.pop(account, None)
        cleared = [account]
    else:
        cleared = []
    if cleared:
        _persist_provider_state_best_effort()
    return cleared


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
        _usage_reset_guessed.discard(failure_domain)
        _usage_limit_last_probe.pop(failure_domain, None)
        return 0.0
    return remaining


# ---------------------------------------------------------------------------
# Usage-limit self-healing probe (LP-0MUQTCMW2001VXH8)
# ---------------------------------------------------------------------------


async def _default_usage_limit_probe_transport(provider_cfg: dict) -> bool:
    """Send a cheap probe request to the provider's upstream.

    Returns True on any 2xx response (the account can serve again). A missing
    endpoint/api key, a non-2xx status, or any transport error returns False —
    the caller then keeps the quarantine. Never raises.
    """
    endpoint = str(provider_cfg.get("endpoint", "") or "").rstrip("/")
    if not endpoint:
        return False
    api_key = None
    api_key_env = provider_cfg.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env)
    if not api_key:
        api_key = provider_cfg.get("api_key")
    if not api_key:
        return False
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        # opencode gateways reject requests without a session id header.
        from .proxy_remote import _is_opencode_upstream

        if _is_opencode_upstream(endpoint):
            headers["x-opencode-session"] = "usage-limit-probe"
    except Exception:
        pass
    model = provider_cfg.get("model") or provider_cfg.get("name") or "probe"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=_USAGE_LIMIT_PROBE_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{endpoint}/v1/chat/completions",
                headers=headers,
                json=payload,
            )
    except Exception:
        logger.info(
            "usage-limit probe transport error for account=%s",
            _usage_limit_account_key(provider_cfg),
            exc_info=True,
        )
        return False
    return 200 <= int(response.status_code) < 300


async def _probe_usage_limit_account(provider_cfg: dict) -> bool:
    """Return True when a quarantined account is usable again.

    Uses the injectable ``_usage_limit_probe_transport`` when set (tests),
    otherwise :func:`_default_usage_limit_probe_transport`. Never raises: any
    error is treated as "still unavailable" (False).
    """
    transport = _usage_limit_probe_transport
    try:
        if transport is not None:
            return bool(await transport(provider_cfg))
        return bool(await _default_usage_limit_probe_transport(provider_cfg))
    except Exception:
        logger.info(
            "usage-limit probe failed for account=%s (keeping quarantine)",
            _usage_limit_account_key(provider_cfg),
            exc_info=True,
        )
        return False


async def _maybe_probe_quarantined_accounts(model_config: dict) -> None:
    """Self-heal guessed usage-limit quarantines before routing.

    For each provider whose account carries a GUESSED quarantine that is due
    for a probe (interval-gated), issue a cheap probe and clear the
    quarantine on success, so the recovered account is routed to within the
    same request cycle. Hard (explicit-duration) quarantines are left alone.
    Never raises.
    """
    if not _usage_reset_guessed:
        return
    now = time.time()
    seen: set[str] = set()
    for provider_cfg in (model_config.get("providers") or []):
        if not isinstance(provider_cfg, dict):
            continue
        account = _usage_limit_account_key(provider_cfg)
        if account in seen or account not in _usage_reset_guessed:
            continue
        seen.add(account)
        if account not in _usage_reset_at:
            _usage_reset_guessed.discard(account)
            continue
        last = _usage_limit_last_probe.get(account, 0.0)
        if now - last < _USAGE_LIMIT_PROBE_INTERVAL_SECONDS:
            continue
        _usage_limit_last_probe[account] = now
        if await _probe_usage_limit_account(provider_cfg):
            logger.info(
                "usage-limit probe succeeded; clearing quarantine for "
                "account=%s",
                account,
            )
            clear_usage_limit_quarantine(account)


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
        # Reset sibling-failure count on success (LP-0MTPMF03P0046MFG)
        _reset_sibling_failure_count(provider_name)
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
    # Reset sibling-failure count on success (LP-0MTPMF03P0046MFG)
    _reset_sibling_failure_count(provider_name)
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
    config: dict | None = None,
) -> bool:
    """If *exc* is a connection error, mark the provider unavailable, record
    a diagnostic attempt entry, and return ``True`` (caller should ``continue``
    to the next provider).

    Applies exponential backoff for remote providers (capped at configured
    *cooldown_seconds*, so setting it to 0 disables backoff entirely).

    Grace-period immunity (LP-0MV9UNIOZ009RIAN): while ``cold_start.is_cold()``
    is True and *provider_type* is ``"local"``, a connection error reflects the
    backend still warming up after a restart, not a provider fault — so it is
    DEFERRED instead of quarantined.  The attempt is recorded as
    ``cold_start_deferral``, no cooldown is applied and the failure counter is
    not bumped, letting the next chain cycle try local again immediately.

    Returns ``False`` if *exc* is **not** a connection error (caller should
    re-raise or handle differently).
    """
    if _is_connection_error(exc):
        if provider_type == "local" and cold_start.is_cold(config):
            _record_attempt(
                attempts,
                provider=provider_name,
                type=provider_type,
                status="cold_start_deferral",
                error=str(type(exc).__name__),
            )
            return True
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
    config: dict | None = None,
    provider_cfg: dict | None = None,
) -> float:
    """Handle an HTTP error response: compute effective cooldown, mark the
    provider unavailable, record a diagnostic attempt entry, and return the
    effective cooldown duration.

    Applies exponential backoff for remote providers.

    When the provider entry opts into ``respect_ratelimit_headers`` (or the
    behaviour is enabled globally under ``server``), a 429 carrying parseable
    ``x-ratelimit-reset-*`` headers sets the cooldown to at least the upstream
    reset duration plus a small safety margin.  Exponential backoff still
    applies, so a cooldown is the *longer* of the two and repeated failures
    keep extending it sensibly (LP-0MVBHDNLB007ALLJ).

    Grace-period immunity (LP-0MV9UNIOZ009RIAN): while ``cold_start.is_cold()``
    is True and *provider_type* is ``"local"``, an HTTP 503 reflects llama-server
    still warming up after a restart, not a provider fault — so it is DEFERRED
    instead of quarantined.  The attempt is recorded as ``cold_start_deferral``
    (with the configured Retry-After hint), no cooldown is applied, the failure
    counter is not bumped, and the retry hint is returned so the caller can
    surface Retry-After.  Any other status code during the window still
    quarantines normally so genuine faults are not masked.

    The caller is responsible for setting ``fallback_reason``, ``prev_provider``,
    and ``all_slot_exhaustion`` after calling this function, and for issuing
    ``continue``.
    """
    if (
        provider_type == "local"
        and cold_start.is_cold(config)
        and int(response.status_code) == 503
    ):
        retry_hint = cold_start.retry_after_seconds(config)
        _record_attempt(
            attempts,
            provider=provider_name,
            type=provider_type,
            status="cold_start_deferral",
            status_code=int(response.status_code),
            retry_after_seconds=retry_hint,
        )
        return float(retry_hint)

    # Parse Retry-After separately so we can respect it alongside backoff
    retry_after = _parse_retry_after(response)

    # Upstream x-ratelimit-* headers (LP-0MVBHDNLB007ALLJ): opt-in per entry or
    # globally; only consulted for a 429 so unrelated HTTP errors keep the
    # pre-existing behaviour exactly.  A missing/parseable-less reset leaves
    # the exponential-backoff guess untouched (AC3).
    ratelimit_reset: float | None = None
    ratelimit_over_limit = False
    if (
        provider_cfg is not None
        and int(response.status_code) == 429
        and _get_respect_ratelimit_headers(config, provider_cfg)
    ):
        ratelimit_over_limit = _ratelimit_over_limit(response.headers)
        ratelimit_reset = _parse_ratelimit_reset_seconds(response.headers)

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

    # Honour the upstream's own reset timing when it is longer than the
    # backoff guess: at least the reset duration plus a safety margin.
    if ratelimit_reset is not None:
        margin = _get_ratelimit_cooldown_margin_seconds(config)
        cooldown = max(cooldown, ratelimit_reset + margin)

    # Account-scoped application (LP-0MVFA1VRT007T9HT): the reset also poisons
    # the failing API-key ACCOUNT (reset + 2-minute margin) so the wait outlives
    # the entry cooldown and is consumed by the exclusion / retry-after / chain
    # hold machinery unchanged. Only applied to a quota 429 that parsed a reset
    # header; a plain 429 with no reset signal keeps the pre-existing behaviour.
    account_quarantine_seconds: float | None = None
    if ratelimit_reset is not None and provider_cfg is not None:
        account_quarantine_seconds = _apply_ratelimit_account_quarantine(
            response, body_text, provider_cfg, ratelimit_reset
        )

    mark_provider_unavailable(provider_name, cooldown)
    attempt_fields: dict[str, Any] = {
        "provider": provider_name,
        "type": provider_type,
        "status": "http_error",
        "status_code": int(response.status_code),
        "body_snippet": (body_text[:512] if body_text else None),
        "cooldown_seconds": cooldown,
    }
    if ratelimit_reset is not None:
        attempt_fields["ratelimit_reset_seconds"] = ratelimit_reset
        attempt_fields["ratelimit_over_limit"] = ratelimit_over_limit
    if account_quarantine_seconds is not None:
        attempt_fields["account_quarantine_seconds"] = account_quarantine_seconds
    _record_attempt(attempts, **attempt_fields)
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


def _reset_local_http_400_streak(session_id: str | None, provider_name: str) -> None:
    """Reset the deterministic local-400 streak after a successful dispatch.

    A session whose local 400s are interspersed with successful local
    dispatches is NOT deterministic — only an unbroken same-session run of
    400s escalates (LP-0MTXEBQ4E001BMI2, mirroring the sibling-failure
    streak semantics). Called from the local success paths of
    ``_proxy_with_fallback_cycle``.
    """
    key = (session_id or "unknown", provider_name)
    _local_http_400_streaks.pop(key, None)


def _prune_local_http_400_streaks(now: float) -> None:
    """Drop stale (session, provider) local-400 streak entries.

    Keeps the in-memory streak dict bounded (LP-0MTXEBQ4E001BMI2); entries
    whose window has fully elapsed are removed on the prune path.
    """
    for key in list(_local_http_400_streaks):
        entry = _local_http_400_streaks[key]
        if now - float(entry.get("start", 0.0)) > _DETERMINISTIC_LOCAL_400_WINDOW_SECONDS:
            del _local_http_400_streaks[key]


def _observe_local_http_error_400(
    response: Response,
    provider_name: str,
    path: str,
    body_text: str,
    session_id: str | None = None,
) -> None:
    """Observability for local HTTP 400 fallbacks (LP-0MTXEBQ4E001BMI2).

    A local 400 is otherwise silent: the local-4xx branch falls back to the
    next provider without cooldown and without any log/metric, so a
    request-shape rejection that is deterministic for a session (repeats
    turn after turn on the same history) invisibly routes remote forever.

    This helper makes every occurrence visible:

    1. **Per occurrence (AC1):** INFO log line with the upstream body
       snippet plus ``proxy_http_errors_total{status=400, reason="local_http_400"}``
       so the rejection cause is discoverable (the snippet is the first
       diagnostic an operator needs to root-cause a compacted-pointer or
       request-shape 400).
    2. **Deterministic repeat (AC2):** when the same session repeats the
       400 within the sliding window (threshold ``_DETERMINISTIC_LOCAL_400_THRESHOLD``),
       a WARNING + ``local_http_400_deterministic`` metric escalates the
       failure instead of silently routing remote each turn.

    Transient one-off 400s never cooldown the local provider (the caller's
    ``http_error_no_cooldown`` path is unchanged — AC4). Best-effort: never
    raises.
    """
    try:
        if int(getattr(response, "status_code", 0) or 0) != 400:
            return
        snippet = (body_text or "")[:512]
        logger.info(
            "Local HTTP 400 from provider=%s model=%s session=%s body_snippet=%s",
            provider_name,
            path,
            session_id or "unknown",
            snippet,
        )
        try:
            from proxy.metrics import record_http_error

            record_http_error(path, "400", "local_http_400")
        except Exception:
            pass

        # Deterministic-repeat escalation (AC2): same session + provider
        # rejecting turn after turn is a request-shape incompatibility that
        # will recur every turn; make it visible instead of silent routing.
        key = (session_id or "unknown", provider_name)
        now = time.monotonic()
        entry = _local_http_400_streaks.get(key)
        if entry is None or now - float(entry.get("start", 0.0)) > _DETERMINISTIC_LOCAL_400_WINDOW_SECONDS:
            entry = {"count": 0, "start": now}
            _local_http_400_streaks[key] = entry
            if len(_local_http_400_streaks) > _MAX_LOCAL_HTTP_400_STREAK_ENTRIES:
                _prune_local_http_400_streaks(now)
        entry["count"] = int(entry.get("count", 0) or 0) + 1
        if int(entry["count"]) >= _DETERMINISTIC_LOCAL_400_THRESHOLD:
            logger.warning(
                "Deterministic local HTTP 400: provider=%s model=%s session=%s "
                "consecutive=%d threshold=%d body_snippet=%s — local rejects this "
                "history every turn; inspect llama-server for the request-shape cause",
                provider_name,
                path,
                session_id or "unknown",
                int(entry["count"]),
                _DETERMINISTIC_LOCAL_400_THRESHOLD,
                snippet,
            )
            try:
                from proxy.metrics import record_http_error

                record_http_error(path, "400", "local_http_400_deterministic")
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
    config: dict | None = None,
) -> float:
    """Handle an empty (non-reasoning) successful response: compute effective
    cooldown, mark the provider unavailable, record a diagnostic attempt entry,
    and return the effective cooldown duration.

    Applies exponential backoff for remote providers, capped at
    ``empty_response_max_cooldown_seconds`` (default 10s).  Retry-After
    headers still override upward.

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

    # Cap for empty responses (LP-0MTVPJBF0001SQF4): transient quirks deserve
    # a lighter penalty than hard failures.  Retry-After still overrides
    # upward (checked below).
    if config is not None:
        max_cd = _get_empty_response_max_cooldown_seconds(config)
        cooldown = min(cooldown, max_cd)

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

    Consistency note (LP-0MTTSBT0R004HC6B): thinking-only responses
    (``reasoning_content`` present, no tool call) are now treated as
    non-empty directly by ``_is_empty_response`` (which checks
    ``reasoning_content`` independently), so the fallback chain treats
    them as plain successes without reaching this function. This function
    remains the safety net for bodies whose raw text contains the
    ``reasoning_content`` key but where extraction found nothing usable
    (e.g. an empty ``reasoning_content`` value) - those still count as
    promoted successes rather than triggering cooldown/fallback.
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
    quarantined: list[dict[str, Any]] = []
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
        # Usage-limit quarantine does not populate the cooldown map; surface
        # it explicitly so operators can distinguish a quota block from a
        # transient cooldown. Quarantine overrides a cooldown on the same
        # provider because the resolver checks it first
        # (LP-0MU56ZFVD001LP0H).
        quarantined = _quarantined_providers(model_config)
        for q in quarantined:
            unavailable[q["name"]] = q["reset_in"]
        logger.warning(
            "All providers exhausted for model=%s; unavailable=%s; "
            "usage_limit_reset=%s",
            path,
            unavailable,
            {q["name"]: q["reset_in"] for q in quarantined} or None,
        )
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
    abort the hold, AC4), False when the full hold elapsed. A zero-length hold
    still checks for a disconnect so an instant-retry (``retry_after == 0``)
    loop with ``chain_hold_max_cycles == 0`` cannot spin forever on a gone
    client (LP-0MU56ZKQD005SX08).
    """
    try:
        if await request.is_disconnected():
            return True
    except Exception:
        pass
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
    try:
        if await request.is_disconnected():
            return
    except Exception:
        pass
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
        "from the first provider (chain_hold_cycle=%d, cycle=%d, max_cycles=%s)",
        path,
        hold_seconds,
        cycle,
        cycle,
        max_cycles if max_cycles else "infinite",
    )


def _set_chain_hold_cycle(request, cycle: int) -> None:
    """Record the current chain-hold cycle index on the request.

    Cycle 0 is the first (pre-hold) cycle; every hold-restart increments it.
    The value is surfaced on routing / skip / compaction / hold log lines so a
    single held request re-emitted across cycles is distinguishable from
    genuine new client demand (LP-0MU5AIAAY003KVM0 AC4). Fail-open: requests
    without a ``state`` (test stubs) simply do not carry the field.
    """
    try:
        request.state.chain_hold_cycle = int(cycle)
    except Exception:
        pass


def _get_chain_hold_cycle(request) -> int:
    """Return the current chain-hold cycle index (0 when unset)."""
    try:
        return int(getattr(request.state, "chain_hold_cycle", 0))
    except Exception:
        return 0


def _set_chain_hold_consumed(request, consumed: float) -> None:
    """Record the cumulative chain-hold budget consumed on the request.

    Mirrors ``_set_chain_hold_cycle``: the streaming hold generator and the
    non-streaming hold loop share the running total via ``request.state`` so
    the cumulative budget (``chain_hold_total_seconds``) is never exceeded
    across the two code paths (LP-0MV6BR3HI00957PF). Fail-open: test stubs
    without a ``state`` do not carry the field.
    """
    try:
        request.state.chain_hold_consumed = float(consumed)
    except Exception:
        pass


def _get_chain_hold_consumed(request, default: float = 0.0) -> float:
    """Return the cumulative chain-hold budget consumed (0.0 when unset)."""
    try:
        return float(getattr(request.state, "chain_hold_consumed", default))
    except Exception:
        return default


def _exhaustion_retry_after(response: Response) -> int | None:
    """Return an exhaustion response's ``Retry-After`` in seconds.

    Prefers the ``Retry-After`` header; falls back to the JSON
    ``retry_after`` field. Returns ``None`` when neither is present/parsable
    (unknown wait — the caller then preserves the legacy full hold).
    """
    try:
        header = response.headers.get("Retry-After")
        if header is not None:
            return max(0, int(float(header)))
    except Exception:
        pass
    try:
        body = _response_body_text(response)
        payload = json.loads(body) if body else None
        if isinstance(payload, dict) and "retry_after" in payload:
            return max(0, int(float(payload["retry_after"])))
    except Exception:
        pass
    return None


def _bounded_hold_seconds(
    hold_seconds: float, response: Response, max_cycles: int
) -> float | None:
    """Bound a chain hold by the exhaustion response's real ``retry_after``.

    Semantics (LP-0MU56ZKQD005SX08): a hold only helps when a provider can
    become available within the configured hold budget. Returns:

    - ``None`` when the chain must NOT hold — ``retry_after`` exceeds the
      total budget ``hold_seconds * max_cycles`` (a window edge / usage-limit
      reset hours away), so holding would only add latency before an
      inevitable exhaustion. The response is returned immediately with its
      accurate ``Retry-After``.
    - ``min(hold_seconds, retry_after)`` when ``retry_after`` is within the
      budget, so the next cycle runs as soon as a provider can serve. A
      ``retry_after`` of 0 (instant cooldown expiry) returns ``0.0`` — retry
      immediately without latency.
    - ``hold_seconds`` (legacy behaviour) when ``retry_after`` is unknown
      (non-JSON error responses), so existing short-cooldown recovery is
      preserved.
    """
    retry_after = _exhaustion_retry_after(response)
    if retry_after is None:
        return hold_seconds
    if retry_after <= 0:
        # No known wait (e.g. instant cooldown expiry): retry immediately
        # without adding latency.
        return 0.0
    total_budget = (
        hold_seconds * max_cycles if max_cycles and max_cycles > 0 else hold_seconds
    )
    if retry_after > total_budget:
        return None
    return min(hold_seconds, float(retry_after))


def _build_streaming_hold_response(
    request,
    path: str,
    model_config: dict,
    config: dict,
    cycle_fn,
    next_cycle: int,
    max_cycles: int,
    hold_seconds: float,
    total_budget: float,
    consumed: float,
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

    *total_budget* (``chain_hold_total_seconds``) caps the CUMULATIVE hold
    time across all cycles so the request can never be held to (or past) the
    client's 900-second queue-timeout deadline; *consumed* carries the budget
    already spent before this streaming response began (LP-0MV6BR3HI00957PF).
    A zero/disabled budget preserves the legacy cycle-bounded behaviour.
    """
    first_provider = _first_provider_name(model_config)

    async def _gen():
        nonlocal consumed
        cycle = next_cycle
        current_exhaustion = exhaustion_response
        while True:
            if total_budget > 0 and consumed >= total_budget:
                # Cumulative hold budget spent — surface the exhaustion now
                # instead of holding through doomed cycles.
                yield _response_to_sse_bytes(current_exhaustion)
                return
            effective_hold = _bounded_hold_seconds(
                hold_seconds, current_exhaustion, max_cycles
            )
            if effective_hold is None:
                # retry_after exceeds the budget — surface the exhaustion
                # immediately instead of holding through doomed cycles.
                yield _response_to_sse_bytes(current_exhaustion)
                return
            if total_budget > 0 and consumed + effective_hold > total_budget:
                effective_hold = max(0.0, total_budget - consumed)
            consumed += effective_hold
            _set_chain_hold_consumed(request, consumed)
            _log_chain_hold_start(path, effective_hold, cycle, max_cycles)
            comment = _build_chain_hold_comment(
                first_provider,
                effective_hold,
                _exhaustion_diagnostics(current_exhaustion),
            )
            async for c in _hold_feedback(request, effective_hold, comment):
                yield c
            try:
                if await request.is_disconnected():
                    return
            except Exception:
                pass
            try:
                _set_chain_hold_cycle(request, cycle)
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
    total_budget = _get_chain_hold_total_seconds(config)
    consumed = 0.0
    _set_chain_hold_consumed(request, 0.0)

    # Semantics (LP-0MU5AIAAY003KVM0, option (a) — decided): every hold cycle
    # re-runs the FULL fallback cycle, including local routing and compaction
    # evaluation. Local session handling (``_handle_session``) runs whenever a
    # cycle dispatches local; when local is bypassed, the bypass path still
    # re-evaluates compaction (``_evaluate_compaction_for_bypass``). A
    # summarizer failure in one cycle therefore does NOT pin the request: a
    # later cycle re-evaluates compaction, and a success refreshes the request
    # body before dispatch. ``_set_chain_hold_cycle`` stamps each cycle so the
    # repeated routing_check / compaction log lines are attributable to one
    # held request (AC4/AC5).
    cycle = 0
    while True:
        try:
            _set_chain_hold_cycle(request, cycle)
            return await cycle_fn(request, path, model_config, config)
        except ChainExhaustedError as exc:
            if max_cycles != 0 and cycle >= max_cycles:
                return exc.response
            effective_hold = _bounded_hold_seconds(
                hold_seconds, exc.response, max_cycles
            )
            if effective_hold is None:
                # retry_after exceeds the hold budget (or is 0): holding could
                # not make the next cycle succeed, so return the exhaustion
                # response immediately with its accurate Retry-After
                # (LP-0MU56ZKQD005SX08).
                return exc.response
            # Cumulative hold budget (LP-0MV6BR3HI00957PF): ``chain_hold_seconds
            # * chain_hold_max_cycles`` can reach exactly the client's 900 s
            # deadline; the total budget caps ALL holds across cycles so the
            # proxy always responds first. Zero disables the cap.
            if total_budget > 0:
                if consumed >= total_budget:
                    return exc.response
                if consumed + effective_hold > total_budget:
                    effective_hold = max(0.0, total_budget - consumed)
            consumed += effective_hold
            _set_chain_hold_consumed(request, consumed)
            _log_chain_hold_start(path, effective_hold, cycle, max_cycles)
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
                    total_budget=total_budget,
                    consumed=consumed,
                    exhaustion_response=exc.response,
                )
            # Non-streaming: silent (deferred) hold, then a new cycle from
            # the first provider.
            if await _hold_sleep(request, effective_hold):
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
    # Self-healing: re-validate guessed (period-name-only) usage-limit
    # quarantines so a recovered account is reintroduced within this request
    # cycle (LP-0MUQTCMW2001VXH8).
    await _maybe_probe_quarantined_accounts(model_config)
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

    # Session identity for the free-tier budget gate's in-progress check
    # (LP-0MVDQ1GMO00801PL). Resolved once; per-request affinity never leaks
    # between sessions.
    _budget_session_id: str | None = None
    try:
        from proxy.session import _resolve_session_id_header
        _budget_session_id, _ = _resolve_session_id_header(
            getattr(request, "headers", {}) or {}
        )
    except Exception:
        pass

    # Effective priority (AC1/AC2): cheap-window capped header value, stamped
    # on the request; the remote-only chain is selected by it (priority_routes)
    # just like the local+remote cycle, so a ``low`` request on a remote-only
    # model uses its low chain (e.g. free-tier-first).
    server_cfg = config.get("server", config) if isinstance(config, dict) else {}
    _priority = request_priority(request, server_cfg)
    from proxy.priority import set_request_priority

    set_request_priority(request, _priority)
    _preference = _request_local_preference(request)
    _chain = effective_chain_for_request(request, model_config, server_cfg)
    if _chain:
        model_config = {**model_config, "providers": _chain}
    elif _preference == _PREF_REMOTE_OK:
        # ``remote-ok`` with no remote-capable provider in the chain: the
        # client asked for no local route, so the chain is genuinely empty
        # (AC5).
        model_config = {**model_config, "providers": []}
    logger.info(
        "priority_routing model=%s priority=%s providers=%d preference=%s",
        model_config.get("name") or "unknown",
        _priority,
        len(_chain or model_config.get("providers") or []),
        _preference,
    )

    # Per-entry context ceiling (LP-0MVF9N2QL001CDA7): parse the body once so
    # the authoritative routing estimate can be computed lazily (once per
    # cycle) for entries that declare ``max_context_tokens``. Fail-open on a
    # body read error (e.g. a synthetic request without a ``body()`` method):
    # routing must never break because the estimate cannot be produced, so the
    # estimate degrades to whatever the session history yields (usually 0).
    _raw_body = None
    try:
        _raw_body = await request.body()
    except Exception:
        logger.debug("max_context_tokens body read failed; gate disabled for this request")
    try:
        body_json = json.loads(_raw_body) if _raw_body else {}
    except Exception:
        body_json = {}
    _routing_estimate = _RoutingEstimateCache(
        request, body_json, model_config, config,
    )

    while True:
        provider_cfg = _resolve_provider_with_exclusions(
            model_config, attempted_provider_names, attempted_domains,
            config=config,
        )
        if provider_cfg is None:
            break

        provider_name = provider_cfg.get("name", "unknown")
        attempted_provider_names.add(provider_name)
        provider_type = provider_cfg.get("type", "remote")
        try:
            # Mark that we attempted this provider
            any_provider_tried = True

            # Per-entry context ceiling (LP-0MVF9N2QL001CDA7 AC2/AC5/AC6): a
            # request whose estimate exceeds the entry's ``max_context_tokens``
            # falls through to the next provider BEFORE the budget gate and
            # dispatch. A skip is a size mismatch, not a failure — no cooldown,
            # usage-limit reset, failure-domain exclusion or affinity — and it
            # never consumes a rate-limit slot.
            _entry_limit = _provider_context_limit(provider_cfg)
            if _entry_limit > 0:
                _entry_estimate = await _routing_estimate.get()
                _context_skip, _context_limit = _should_skip_provider_context(
                    provider_cfg, _entry_estimate,
                )
                if _context_skip:
                    _record_provider_context_skip(
                        attempts, provider_name, provider_type,
                        _entry_estimate, _context_limit, _budget_session_id,
                    )
                    fallback_reason = "provider_context_exceeded"
                    prev_provider = provider_name
                    all_slot_exhaustion = False
                    continue

            # Capacity-aware request budget for free-tier style entries
            # (LP-0MVDQ1GMO00801PL): dispatches only while the rolling window
            # has capacity. A NEW session is never parked — it passes straight
            # through to the next provider — while a session already using the
            # entry waits a bounded time for the window to reset so it keeps
            # its prompt cache/prefix continuity.
            _budget = await acquire_provider_budget(
                provider_cfg, config, _budget_session_id, request,
                model_config=model_config,
            )
            if not _budget.allowed:
                _record_attempt(
                    attempts,
                    provider=provider_name,
                    type="remote",
                    status="rate_limited",
                    status_code=429,
                    reason=_budget.reason,
                    limit_rpm=_budget.limit_rpm,
                    effective_limit=_budget.effective_limit,
                    reserve_slots=_budget.reserve_slots,
                    remaining=_budget.remaining,
                    elapsed_wait_seconds=round(_budget.waited_seconds, 3),
                    **_budget_diagnostics(_budget),
                )
                fallback_reason = "rate_limited"
                prev_provider = provider_name
                all_slot_exhaustion = False
                continue
            if _budget.waited:
                _record_attempt(
                    attempts,
                    provider=provider_name,
                    type="remote",
                    status="rate_limited_waited",
                    reason=_budget.reason,
                    limit_rpm=_budget.limit_rpm,
                    effective_limit=_budget.effective_limit,
                    reserve_slots=_budget.reserve_slots,
                    remaining=_budget.remaining,
                    elapsed_wait_seconds=round(_budget.waited_seconds, 3),
                    **_budget_diagnostics(_budget),
                )

            response = await ptr(request, path, provider_cfg)
            _maybe_record_provider_affinity(
                provider_cfg, _budget_session_id, response, config,
                model_config=model_config,
            )

            # Pre-flight remote streaming responses: detect a pre-content
            # finish_reason: error / empty / stream-exception so the fallback
            # chain re-routes to the next provider instead of surfacing a
            # bare error to the client (LP-0MSETOTWY000SU0Z, Recommendation 1).
            if provider_type == "remote" and _has_next_provider(
                model_config, attempted_provider_names, attempted_domains,
                config=config,
            ):
                try:
                    response = await _preflight_streaming_response(
                        response, request, provider_name
                    )
                except StreamingPreContentError as exc:
                    mark_provider_unavailable(provider_name, cooldown_seconds)
                    # Sibling-fallback circuit breaker (LP-0MTPMF03P0046MFG):
                    # a pre-content stream error (empty_response / finish_reason:
                    # error with zero content) counts toward the consecutive-
                    # failure streak so the provider gets an extended cooldown
                    # after the threshold and the retry cycle skips to a sibling.
                    _record_sibling_failure(
                        provider_name, config, provider_cfg.get("provider"),
                        domain_key=_failure_domain_key(provider_cfg)
                    )
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
                    # Sibling-fallback circuit breaker (LP-0MTPMF03P0046MFG):
                    # reasoning was delivered but the provider stalled before
                    # usable final content; repeated stalls on the same provider
                    # count toward the streak so it is skipped for a sibling
                    # after the threshold.
                    _record_sibling_failure(
                        provider_name, config, provider_cfg.get("provider"),
                        domain_key=_failure_domain_key(provider_cfg)
                    )
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
                _reset_info = _usage_limit_reset_info(
                    response, body_text, config, provider_cfg=provider_cfg
                )
                if _reset_info is not None:
                    _reset_account = _usage_limit_account_key(provider_cfg)
                    _set_usage_limit_quarantine(_reset_account, _reset_info)
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
                        reset_in_seconds=int(_reset_info.seconds),
                    )
                    all_slot_exhaustion = False
                    continue

                # FreeUsageLimitError: apply cooldown on affected provider
                # so the fallback chain routes to paid alternatives instead of
                # repeatedly retrying the exhausted free tier (LP-0MRGU0I91006ODFD).
                if _is_free_usage_limit_error(response, body_text):
                    fallback_reason = "free_usage_limit"
                    prev_provider = provider_name
                    cooldown_seconds = _FREE_USAGE_LIMIT_COOLDOWN_SECONDS
                    mark_provider_unavailable(provider_name, cooldown_seconds)
                    attempted_domains.add(_failure_domain_key(provider_cfg))
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="free_usage_limit",
                        status_code=int(response.status_code),
                        body_snippet=(body_text[:512] if body_text else None),
                        cooldown_seconds=cooldown_seconds,
                    )
                    all_slot_exhaustion = False
                    continue

                # Shared primitive: HTTP error with cooldown
                _handle_http_error_with_cooldown(
                    response, provider_name, provider_type,
                    cooldown_seconds, attempts, body_text, config,
                    provider_cfg=provider_cfg,
                )
                # AC3 (LP-0MSGU3JNU0092AFQ): remember this specific 400 so the
                # exhausted path can return a synthetic error instead of the
                # raw upstream body / exhausted-503-with-diagnostics.
                if _reasoning_roundtrip_response is None and _is_reasoning_content_roundtrip_error(response):
                    _reasoning_roundtrip_response = response
                # Per-account rate limit (LP-0MVCX0E2V001VB5N): a 429 with
                # body type=rate_limit_error / code=rate_limit_exceeded is a
                # per-ACCOUNT quota event (independent per api_key_env), NOT an
                # endpoint-level failure. Poison only the failing account key in
                # the per-request exclusion set so paid siblings on the same
                # gateway stay eligible (e.g. munitoai-free 429 must not skip
                # the paid munitoai key). Other HTTP errors keep poisoning the
                # shared failure domain (AC2).
                if _is_account_rate_limit_error(response, body_text):
                    attempted_domains.add(_usage_limit_account_key(provider_cfg))
                else:
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
                        cooldown_seconds, attempts, body_text, config,
                    )
                    # Sibling-fallback circuit breaker (LP-0MTPMF03P0046MFG):
                    # track consecutive empty/stall failures across cycles.
                    # Only poison the failure domain (skip same-endpoint
                    # siblings) once the streak threshold is exceeded so a
                    # single empty response does not block sibling API keys
                    # on the same endpoint (LP-0MTVPJQ6T004EZ75).
                    # LP-0MUNTOGLB005LUKY: pass provider_type so the breaker
                    # can skip local providers entirely.
                    _threshold_exceeded = _record_sibling_failure(
                        provider_name, config, provider_cfg.get("provider"),
                        domain_key=_failure_domain_key(provider_cfg),
                        provider_type=provider_type,
                    )
                    if _threshold_exceeded:
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
                exc, provider_name, provider_type, cooldown_seconds, attempts, config,
            ):
                any_provider_tried = True
                attempted_domains.add(_connection_error_domain_key(provider_cfg, exc))
                fallback_reason = str(type(exc).__name__)
                prev_provider = provider_name
                all_slot_exhaustion = False
                continue
            # Non-connection error — propagate
            raise

    # All providers exhausted — log diagnostic details
    unavailable = _log_exhausted_providers(model_config, path)

    # Usage-limit quarantine takes precedence over the window reason, so it
    # is recorded first; the window loop below skips providers already
    # quarantined (LP-0MU56ZFVD001LP0H).
    _record_quarantine_attempts(attempts, model_config)

    # Record time-window skips as distinct diagnostics so operators can tell
    # whether providers were excluded by their available_times windows rather
    # than by cooldown/errors (LP-0MS4ETBNO0022QAC).
    for skipped in _providers_outside_window(model_config, config):
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
        attempts, unavailable, any_provider_tried, model_config=model_config,
        config=config,
    )
    if time_window_exhausted is not None:
        raise ChainExhaustedError(time_window_exhausted)

    if not any_provider_tried:
        raise ChainExhaustedError(
            _build_exhausted_response(all_local_slot_exhaustion=False, unavailable_providers=unavailable, diagnostics=attempts, model_config=model_config, config=config)
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
        _build_exhausted_response(all_local_slot_exhaustion=all_slot_exhaustion, unavailable_providers=unavailable, diagnostics=attempts, model_config=model_config, config=config)
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
    response = await _run_chain_cycles(
        request, path, model_config, config, _proxy_with_remote_fallback_cycle,
    )
    return _stamp_local_route_headers(response, request, model_config)


async def _evaluate_compaction_for_bypass(
    config: dict,
    provider_name: str,
    body_json: dict,
    session_id: str | None,
) -> dict:
    """Evaluate server-side session compaction in the local-bypass path.

    LP-0MU5ARWSP001BYYB: the bypass path used to ``continue`` before any
    session handling, so oversized sessions — the ones that need compaction
    most — were never evaluated. This delegates to the shared
    :func:`proxy.router_helpers.evaluate_and_apply_compaction` so the bypass
    path and the local-dispatch path share one implementation.

    The helper mutates ``body_json`` in place when a live compaction is
    applied (rewriting ``messages`` to the compacted history) and also
    updates the persisted session history.

    Fail-open: any error is logged and a noop outcome is returned so routing
    continues unchanged.
    """
    outcome: dict[str, Any] = {
        "evaluated": False,
        "action": "noop",
        "applied": False,
        "estimated_before": 0,
        "estimated_after": 0,
        "reason": None,
        "messages": None,
    }
    try:
        import proxy.server as _server
        from proxy.router_helpers import evaluate_and_apply_compaction

        server_config = (
            config.get("server", config) if isinstance(config, dict) else {}
        )
        result = {"session_id": session_id}
        return await evaluate_and_apply_compaction(
            _server,
            result,
            body_json,
            server_config,
        )
    except Exception:
        logger.warning(
            "Bypass compaction evaluation failed; continuing unchanged "
            "(provider=%s session=%s)",
            provider_name,
            str(session_id or "")[:8],
            exc_info=True,
        )
        return outcome


def _refresh_request_body(
    request,
    body_json: dict,
    provider_name: str,
    session_id: str | None,
) -> None:
    """Replace the cached request body with the compacted ``body_json``.

    ``proxy_to_local`` reads ``await request.body()``; Starlette caches the
    parsed bytes in ``Request._body``. Re-serialising the compacted body into
    that cache lets the ordinary local-dispatch path below forward the
    compacted history without a second ``_dispatch_local`` call
    (LP-0MU5ARWSP001BYYB).
    """
    try:
        request._body = json.dumps(body_json).encode("utf-8")
    except Exception:
        logger.warning(
            "compaction_local_redispatch body refresh failed provider=%s "
            "session=%s",
            provider_name,
            str(session_id or "")[:8],
            exc_info=True,
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
    # Self-healing: re-validate guessed (period-name-only) usage-limit
    # quarantines so a recovered account is reintroduced within this request
    # cycle (LP-0MUQTCMW2001VXH8).
    await _maybe_probe_quarantined_accounts(model_config)
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

    # Effective priority (AC1/AC2): cheap-window capped header value, stamped
    # on the request; the provider chain is selected by it (priority_routes).
    # Rebound local ``model_config`` so the whole cycle (including the
    # lease-active fallback below) iterates the priority-selected chain.
    server_cfg = config.get("server", config) if isinstance(config, dict) else {}
    _priority = request_priority(request, server_cfg)
    from proxy.priority import set_request_priority

    set_request_priority(request, _priority)
    _preference = _request_local_preference(request)
    _chain = effective_chain_for_request(request, model_config, server_cfg)
    if _chain:
        model_config = {**model_config, "providers": _chain}
    elif _preference == _PREF_REMOTE_OK:
        # ``remote-ok`` with no remote-capable provider in the chain: the
        # client asked for no local route, so the chain is genuinely empty
        # (AC5).
        model_config = {**model_config, "providers": []}
    logger.info(
        "priority_routing model=%s priority=%s providers=%d preference=%s",
        model_config.get("name") or "unknown",
        _priority,
        len(_chain or model_config.get("providers") or []),
        _preference,
    )

    # Shared per-cycle routing estimate (LP-0MVF9N2QL001CDA7 AC4): computed
    # once on first use — by the local smart-routing gate and/or the per-entry
    # ``max_context_tokens`` ceiling — and reused for every provider in the
    # chain, so the gate never adds a tokenizer pass per entry.
    _routing_estimate = _RoutingEstimateCache(
        request, body_json, model_config, config,
    )

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
            config=config,
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
                if _is_domain_excluded(candidate, attempted_domains) is not None:
                    continue
                if candidate.get("type") != "remote":
                    continue
                if not _provider_in_allowed_window(candidate, config):
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
                # Resolve the per-endpoint target for this local provider
                # (LP-0MRPILSMW004T4H8). When the provider config has no
                # explicit ``endpoint`` the legacy default is used, keeping
                # single-server deployments working without changes.
                local_endpoint = _get_local_provider_endpoint(provider_cfg, config)
                # LP-0MVKUKFHH005UDWL: post-deadline prefill-thrash cooldown.
                # A session whose last local dispatch was cancelled by the
                # prefill deadline must not immediately re-dispatch the
                # identical prompt and burn the whole prefill again (the
                # 2026-10-23 incident: 41 local POSTs, 35 inside 02:00-03:00,
                # re-prefilling near-duplicates of the same prompt). While the
                # session cools down, local is skipped here — BEFORE the
                # concurrency / queue / smart-routing gates, so the request
                # never queues for a slot it would then be forbidden to use —
                # and the chain reaches the next provider (AC1/AC4). The
                # window is armed by router.py on the deadline event, cleared
                # on expiry or a successful local first byte, and per-session
                # (AC2 cross-session isolation).
                _thrash_cooldown_remaining = 0.0
                try:
                    from proxy.router_helpers import (
                        _prefill_thrash_cooldown_remaining,
                    )

                    _thrash_cooldown_remaining = (
                        _prefill_thrash_cooldown_remaining(
                            _session_id,
                            config,
                        )
                        or 0.0
                    )
                except Exception:
                    _thrash_cooldown_remaining = 0.0
                if _thrash_cooldown_remaining > 0:
                    logger.info(
                        "routing_skip_local provider=%s model=%s "
                        "session=%s cooldown_remaining=%.0fs "
                        "reason=prefill_thrash_cooldown → skipping local, "
                        "routing to next remote provider",
                        provider_name,
                        provider_cfg.get("llama_model") or "unknown",
                        _session_id or "unknown",
                        _thrash_cooldown_remaining,
                    )
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="cached_tokens_skip",
                        reason="prefill_thrash_cooldown",
                        cooldown_remaining_seconds=round(
                            _thrash_cooldown_remaining, 1
                        ),
                    )
                    fallback_reason = "prefill_thrash_cooldown"
                    prev_provider = provider_name
                    all_slot_exhaustion = False
                    _note_local_fallback_reason(
                        request, "prefill_thrash_cooldown"
                    )
                    continue
                # Local concurrency limit check (LP-0MR5MAJNM005R905):
                # if local concurrency limit (session_slot_pool_size) is exceeded, skip to next
                # provider without marking local as unavailable. The check is
                # scoped per-endpoint so each llama-server's own slot pool is
                # evaluated independently (LP-0MRPILSMW004T4H8).
                cur_local, max_local = _local_concurrency_info(

                    config, endpoint=local_endpoint
                )
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
                            endpoint=local_endpoint,
                            priority=_priority,
                        )
                    )
                    if _cq_action == "dispatch":
                        # A slot freed within the caps — dispatch local below.
                        _set_queue_wait_on_request(request, _cq_elapsed)
                        try:
                            from proxy import contention_queue

                            _cq_depth = contention_queue.queue_depth()
                        except Exception:
                            _cq_depth = 0
                        logger.info(
                            "contention_queue_dispatch provider=%s session=%s "
                            "queued_duration=%.2fs policy=queue depth=%d",
                            provider_name, _session_id or "unknown",
                            _cq_elapsed or 0.0,
                            _cq_depth,
                        )
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
                        _note_local_fallback_reason(request, fallback_reason)
                        logger.info(
                            "contention_queue_fallback_after_queue provider=%s "
                            "session=%s queued_duration=%.2fs",
                            provider_name, _session_id or "unknown",
                            _cq_elapsed or 0.0,
                        )
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
                        _note_local_fallback_reason(request, _cq_reason)
                        continue
                    elif _cq_action == "admission_reject":
                        # Prefill-ETA admission guard (LP-0MVGDUV5E005UWBQ):
                        # the projected prefill cannot finish inside the wait
                        # budget, so refuse local admission NOW and fall
                        # through to the next provider instead of parking in
                        # the contention queue for the full budget.
                        _admission = getattr(
                            request, "_local_admission_decision", None
                        )
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="local_admission_reject",
                            reason=_cq_reason,
                            eta_prefill_seconds=round(
                                float(getattr(_admission, "eta_prefill_seconds", 0.0) or 0.0), 1
                            ),
                            budget_seconds=round(
                                float(getattr(_admission, "budget_seconds", 0.0) or 0.0), 1
                            ),
                        )
                        fallback_reason = "local_admission_guard"
                        prev_provider = provider_name
                        all_slot_exhaustion = False
                        _note_local_fallback_reason(request, fallback_reason)
                        continue
                    elif _cq_action == "refused_local_only":
                        # AC4 (LP-0MVIKPCSG002B0NL): the client set
                        # X-Local-Preference: local-only and the admission
                        # guard refused. Unlike admission_reject, the request
                        # must NOT fall through to a remote provider — return
                        # the retryable 429 with X-Local-Route: refused now.
                        _admission = getattr(
                            request, "_local_admission_decision", None
                        )
                        _note_local_fallback_reason(request, _cq_reason)
                        _refusal_attempt = {
                            "status": "local_admission_reject",
                            "reason": _cq_reason or "eta_exceeds_budget",
                            "eta_prefill_seconds": round(
                                float(getattr(_admission, "eta_prefill_seconds", 0.0) or 0.0), 1
                            ),
                            "budget_seconds": round(
                                float(getattr(_admission, "budget_seconds", 0.0) or 0.0), 1
                            ),
                            "projected_wait_seconds": round(
                                float(getattr(_admission, "projected_wait_seconds", 0.0) or 0.0), 1
                            ),
                        }
                        return _build_local_only_refusal_response(
                            _refusal_attempt, config
                        )
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
                        _note_local_fallback_reason(request, fallback_reason)
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
                # Per-entry context ceiling on a LOCAL entry
                # (LP-0MVF9N2QL001CDA7 AC7): ``max_context_tokens`` composes as
                # an ADDITIONAL hard cap on the existing warm threshold —
                # effective = ``min(existing, limit)`` — so it can only tighten
                # the local gate, never loosen it.
                _local_context_limit = _provider_context_limit(provider_cfg)
                if _local_context_limit > 0:
                    _warm_threshold = (
                        min(_warm_threshold, _local_context_limit)
                        if _warm_threshold > 0
                        else _local_context_limit
                    )
                # Resolve the native tokenizer + multiplier via the shared
                # helper (LP-0MSEQ71IF0003FRT): a named tokenizer (e.g.
                # ``tokenizer: qwen3`` on the model entry) replaces the
                # tiktoken-vs-Qwen3 mismatch (~1.69x undercount,
                # LP-0MSAOQTJS000FFVM F2/F3) with exact counts and forces
                # the multiplier to 1.0; without a tokenizer, the server-
                # level / per-model token_estimate_multiplier still applies
                # (LP-0MSEGPO77005CYCQ F2). The estimate itself is shared per
                # cycle (_RoutingEstimateCache) with the per-entry ceiling.
                _estimated_tokens = await _routing_estimate.get()
                # Seed the routing ratio from the persisted local slot snapshot
                # (LP-0MVGD9YXH008O304): a session whose KV prefix is already on
                # disk is warm even though this process has no in-memory record
                # of it (fresh session, or any request after a proxy restart).
                # Fail-open: any problem keeps today's cached_ratio (cold).
                try:
                    from proxy.slot_cache_ratio import seed_cached_ratio

                    seed_cached_ratio(
                        config.get("server", config)
                        if isinstance(config, dict)
                        else {},
                        body_json,
                        _llama_model,
                        _session_id,
                        _estimated_tokens,
                        endpoint=_get_local_provider_endpoint(provider_cfg, config),
                    )
                except Exception:
                    logger.debug(
                        "slot snapshot ratio seeding failed", exc_info=True
                    )
                # Compute once for reuse in logs and reason detection. The
                # expected-reuse credit (LP-0MVI2XOTY008Q4UV) makes the first
                # turn after a cold start non-zero when the prompt extends the
                # previous local prompt.
                _routing_cached_ratio = expected_cached_ratio(
                    _llama_model,
                    _session_id,
                    _estimated_tokens,
                    body_json.get("messages") if isinstance(body_json, dict) else None,
                    server_config=(
                        config.get("server", config) if isinstance(config, dict) else None
                    ),
                )
                try:
                    from proxy.router_helpers import _classify_cache_state

                    _routing_cache_state = _classify_cache_state(
                        _routing_cached_ratio,
                        _get_cached_ratio_source(_llama_model, _session_id),
                    )
                except Exception:
                    _routing_cache_state = "cold"
                _routing_new_tokens = int(
                    _estimated_tokens * (1 - _routing_cached_ratio)
                )
                logger.info(
                    "routing_check provider=%s model=%s "
                    "estimated_tokens=%d cold_threshold=%d warm_threshold=%d "
                    "new_tokens=%d cached_ratio=%.2f cache_state=%s "
                    "messages=%d session=%s "
                    "chain_hold_cycle=%d",
                    provider_name,
                    _llama_model or "unknown",
                    _estimated_tokens,
                    _cold_threshold,
                    _warm_threshold,
                    _routing_new_tokens,
                    _routing_cached_ratio,
                    _routing_cache_state,
                    len(body_json.get("messages", [])) if isinstance(body_json, dict) else -1,
                    _session_id or "unknown",
                    _get_chain_hold_cycle(request),
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
                # Resolve the priority-driven economic-bypass recovery flag
                # (AC3, LP-0MVDT94F0003VHUA): low priority recovers economic
                # bypasses locally (cheap semantics); high routes remote;
                # medium uses the merged base config flag (absent — false).
                _recover_economic = _economic_bypass_serves_local(config, _priority)
                _skip_local, _skip_reason = _should_skip_local(
                    _llama_model,
                    _session_id,
                    body_json,
                    _cold_threshold,
                    estimated_tokens=_estimated_tokens,
                    warm_cache_threshold=_warm_threshold,
                    economic_bypass_serves_local=_recover_economic,
                )
                # Idle-aware local admission (LP-0MVGD9Z8Z0055Y5J): spend
                # obviously idle GPU capacity instead of paying for a remote
                # call. Only the ECONOMIC bypass is overridable — the physical
                # ``context_too_large`` cap still applies (the request cannot
                # fit the KV slot).
                if _skip_local and _skip_reason == "large_context_bypass":
                    _idle_admit = _evaluate_idle_admission_override(
                        config, provider_cfg, _estimated_tokens
                    )
                    if _idle_admit is not None:
                        logger.info(
                            "routing_idle_admit provider=%s model=%s "
                            "reason=%s idle_seconds=%.1f threshold_seconds=%.1f "
                            "estimated_tokens=%d new_tokens=%d cached_ratio=%.2f "
                            "cache_state=%s session=%s",
                            provider_name,
                            _llama_model or "unknown",
                            _idle_admit["reason"],
                            _idle_admit["idle_seconds"],
                            _idle_admit["threshold_seconds"],
                            _estimated_tokens,
                            _routing_new_tokens,
                            _routing_cached_ratio,
                            _routing_cache_state,
                            _session_id or "unknown",
                        )
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="idle_admit",
                            reason=_idle_admit["reason"],
                            estimated_tokens=_estimated_tokens,
                            idle_seconds=round(_idle_admit["idle_seconds"], 1),
                            threshold_seconds=_idle_admit["threshold_seconds"],
                        )
                        _skip_local = False
                        _skip_reason = None
                # Capacity-based admission (LP-0MVGDUVS6003T2IX): admit the
                # economic band locally while the projected slot-seconds of
                # admissions stay inside a share of MEASURED idle slot time.
                # The physical ``context_too_large`` cap is never overridden.
                # Default-off: the rollback switch is
                # ``local_capacity_admission_enabled``.
                if _skip_local and _skip_reason == "large_context_bypass":
                    try:
                        from proxy.capacity_admission import (
                            evaluate_capacity_admission,
                        )

                        _capacity_admit = evaluate_capacity_admission(
                            config,
                            _llama_model,
                            _estimated_tokens,
                            _routing_cached_ratio,
                        )
                    except Exception:
                        _capacity_admit = None
                    if _capacity_admit is not None:
                        logger.info(
                            "routing_capacity_admit provider=%s model=%s "
                            "reason=%s projected_s=%.1f admitted_s=%.1f "
                            "budget_s=%.1f idle_s=%.1f budget_ratio=%.2f "
                            "horizon_s=%.0f estimated_tokens=%d "
                            "cached_ratio=%.2f cache_state=%s warm_threshold=%d "
                            "session=%s",
                            provider_name,
                            _llama_model or "unknown",
                            _capacity_admit["reason"],
                            _capacity_admit["projected_seconds"],
                            _capacity_admit["admitted_seconds"],
                            _capacity_admit["budget_seconds"],
                            _capacity_admit["idle_seconds"],
                            _capacity_admit["budget_ratio"],
                            _capacity_admit["horizon_seconds"],
                            _estimated_tokens,
                            _routing_cached_ratio,
                            _routing_cache_state,
                            _warm_threshold,
                            _session_id or "unknown",
                        )
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="capacity_admit",
                            reason=_capacity_admit["reason"],
                            estimated_tokens=_estimated_tokens,
                            projected_seconds=round(
                                _capacity_admit["projected_seconds"], 1
                            ),
                            budget_seconds=round(
                                _capacity_admit["budget_seconds"], 1
                            ),
                        )
                        _skip_local = False
                        _skip_reason = None
                if _skip_local:
                    logger.info(
                        "routing_skip_local provider=%s model=%s "
                        "estimated_tokens=%d cold_threshold=%d warm_threshold=%d "
                        "new_tokens=%d cached_ratio=%.2f cache_state=%s "
                        "reason=%s → skipping local, routing to next remote provider "
                        "session=%s chain_hold_cycle=%d",
                        provider_name,
                        _llama_model or "unknown",
                        _estimated_tokens,
                        _cold_threshold,
                        _warm_threshold,
                        _routing_new_tokens,
                        _routing_cached_ratio,
                        _routing_cache_state,
                        _skip_reason,
                        _session_id or "unknown",
                        _get_chain_hold_cycle(request),
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
                        cache_state=_routing_cache_state,
                        reason=_skip_reason,
                    )
                    fallback_reason = _skip_reason
                    prev_provider = provider_name
                    all_slot_exhaustion = False
                    _note_local_fallback_reason(request, _skip_reason)

                    # Server-side compaction rescue (LP-0MU5ARWSP001BYYB).
                    # The bypass used to ``continue`` here, before any session
                    # handling, so oversized sessions were never evaluated for
                    # compaction. Always evaluate now; a successful compaction
                    # may bring the session back under the local thresholds.
                    _bypass_compaction = await _evaluate_compaction_for_bypass(
                        config, provider_name, body_json, _session_id,
                    )
                    _rescued_local = False
                    if _bypass_compaction["applied"]:
                        # Re-run local eligibility against the compacted size so
                        # a session bypassed purely for size can return to
                        # local on this same request (AC2). ``recompute()`` also
                        # refreshes the per-cycle cache for any later entry.
                        _compacted_tokens = await _routing_estimate.recompute()
                        _skip_after, _skip_after_reason = _should_skip_local(
                            _llama_model,
                            _session_id,
                            body_json,
                            _cold_threshold,
                            estimated_tokens=_compacted_tokens,
                            warm_cache_threshold=_warm_threshold,
                            economic_bypass_serves_local=_recover_economic,
                        )
                        if _skip_after:
                            logger.info(
                                "routing_compaction_still_oversized provider=%s "
                                "model=%s est_before=%d est_after=%d "
                                "reason=%s session=%s",
                                provider_name,
                                _llama_model or "unknown",
                                _bypass_compaction["estimated_before"],
                                _bypass_compaction["estimated_after"],
                                _skip_after_reason,
                                _session_id or "unknown",
                            )
                        else:
                            logger.info(
                                "routing_compaction_local_redispatch provider=%s "
                                "model=%s est_before=%d est_after=%d "
                                "skip_reason=%s session=%s",
                                provider_name,
                                _llama_model or "unknown",
                                _bypass_compaction["estimated_before"],
                                _bypass_compaction["estimated_after"],
                                _skip_reason,
                                _session_id or "unknown",
                            )
                            _record_attempt(
                                attempts,
                                provider=provider_name,
                                type=provider_type,
                                status="compaction_local_redispatch",
                                estimated_tokens=_compacted_tokens,
                                reason="compaction_reenabled_local",
                            )
                            # Refresh the cached body so the ordinary local
                            # dispatch below forwards the compacted history.
                            # Its own session handling re-evaluates compaction,
                            # but the compacted history is below the trigger so
                            # that evaluation is a cheap no-op — no second
                            # summarisation and only one local dispatch.
                            _refresh_request_body(
                                request, body_json, provider_name, _session_id,
                            )
                            _rescued_local = True
                    else:
                        logger.info(
                            "compaction_bypass_eval provider=%s model=%s "
                            "evaluated=%s action=%s reason=%s est_before=%d "
                            "est_after=%d session=%s chain_hold_cycle=%d",
                            provider_name,
                            _llama_model or "unknown",
                            _bypass_compaction["evaluated"],
                            _bypass_compaction["action"],
                            _bypass_compaction.get("reason"),
                            _bypass_compaction["estimated_before"],
                            _bypass_compaction["estimated_after"],
                            _session_id or "unknown",
                            _get_chain_hold_cycle(request),
                        )
                    if not _rescued_local:
                        # No rescue — preserve the original bypass behaviour
                        # and continue to the next (remote) provider.
                        continue
                    # Rescued: fall through to the ordinary local dispatch
                    # below with the compacted body.

                # Cheap-mode economic-bypass recovery: log when a request
                # that would have been cold-cache-bypassed proceeds to local
                # (LP-0MU5A4QBR003YJM0).  The `_recover_economic` /
                # `_cold_threshold` variables are available from the scope
                # above.
                if (
                    _recover_economic
                    and _cold_threshold > 0
                    and _routing_new_tokens > _cold_threshold
                ):
                    logger.info(
                        "routing_economic_bypass_local provider=%s model=%s "
                        "estimated_tokens=%d cold_threshold=%d warm_threshold=%d "
                        "new_tokens=%d cached_ratio=%.2f cache_state=%s session=%s",
                        provider_name,
                        _llama_model or "unknown",
                        _estimated_tokens,
                        _cold_threshold,
                        _warm_threshold,
                        _routing_new_tokens,
                        _routing_cached_ratio,
                        _routing_cache_state,
                        _session_id or "unknown",
                    )
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="economic_bypass_local_serve",
                        estimated_tokens=_estimated_tokens,
                        cold_threshold=_cold_threshold,
                        warm_threshold=_warm_threshold,
                        new_tokens=_routing_new_tokens,
                        cached_ratio=_routing_cached_ratio,
                        reason="economic_bypass_local_serve",
                    )

                # Prefill-gate wait accounting (LP-0MVGDUUV3001XFRJ AC4): how
                # long this request waited specifically for the prefill gate.
                # ``_local_queue_saturation`` is stamped by
                # ``_maybe_queue_for_local_slot`` before the wait and
                # ``_contention_queue_wait_seconds`` records the measured wait,
                # so both are read here (0.0 when the request was never
                # prefill-gated).
                _prefill_gate_wait_ms = 0.0
                try:
                    if (
                        getattr(request, "_local_queue_saturation", None)
                        == "prefill"
                    ):
                        _prefill_gate_wait_ms = max(
                            0.0,
                            float(
                                getattr(
                                    request,
                                    "_contention_queue_wait_seconds",
                                    0.0,
                                )
                                or 0.0
                            ),
                        ) * 1000.0
                except Exception:
                    _prefill_gate_wait_ms = 0.0
                try:
                    logger.info(
                        "prefill_gate_wait_ms=%.1f saturation=%s session=%s "
                        "model=%s provider=%s",
                        _prefill_gate_wait_ms,
                        getattr(request, "_local_queue_saturation", None) or "none",
                        _session_id or "unknown",
                        _llama_model or "unknown",
                        provider_name,
                    )
                except Exception:
                    pass

                # Prefill-ETA guard on the DIRECT (free-slot) dispatch path
                # (LP-0MVIQ0F8T002FS42). The queue-entry guard only runs when
                # every local slot is busy; without this a request that finds a
                # free slot starts a 40-80k-token prefill that cannot finish
                # inside the caller's wait budget and is abandoned (baseline: 48
                # of 606 local dispatches had a first byte after >120 s). Same
                # estimator, same single budget source
                # (``contention_queue_max_wait_seconds``) — this is the same
                # decision applied one step earlier. Inert (``None``) when the
                # bucket is uncalibrated, so cold start behaves exactly as
                # today.
                _direct_admission = await _prefill_admission_guard(
                    config,
                    provider_cfg,
                    _estimated_tokens,
                    _session_id,
                    priority=_priority,
                    messages=(
                        body_json.get("messages")
                        if isinstance(body_json, dict)
                        else None
                    ),
                    max_local=max_local,
                    endpoint=local_endpoint,
                    reason_label="eta_exceeds_budget_direct",
                    free_slot=True,
                )
                if _direct_admission is not None and not _direct_admission.admit:
                    # Deadline refusal (LP-0MVKUKEU80022C3U) keeps its own
                    # reason so the diagnostic says *why* local was refused.
                    if str(_direct_admission.reason).startswith(
                        "prefill_eta_exceeds_deadline"
                    ):
                        _direct_reason = "prefill_eta_exceeds_deadline"
                    else:
                        _direct_reason = "eta_exceeds_budget_direct"
                    try:
                        setattr(request, "_local_admission_decision", _direct_admission)
                    except Exception:
                        pass
                    # The guard helper already emitted the single
                    # ``local_admission_guard ... reason=eta_exceeds_budget_direct``
                    # line (ETA, budget, free_slot=true). Record the attempt so
                    # the fallback provider and the diagnostics carry it too
                    # (AC4).
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="local_admission_reject",
                        reason=_direct_reason,
                        eta_prefill_seconds=round(
                            float(
                                getattr(
                                    _direct_admission,
                                    "eta_prefill_seconds",
                                    0.0,
                                )
                                or 0.0
                            ),
                            1,
                        ),
                        budget_seconds=round(
                            float(
                                getattr(_direct_admission, "budget_seconds", 0.0)
                                or 0.0
                            ),
                            1,
                        ),
                    )
                    try:
                        logger.info(
                            "local_admission_guard_direct_fallback "
                            "provider=%s model=%s session=%s "
                            "reason=%s free_slot=true → routing to next provider",
                            provider_name,
                            _llama_model or "unknown",
                            _session_id or "unknown",
                            _direct_reason,
                        )
                    except Exception:
                        pass
                    # Explicit client intent (LP-0MVIKPCSG002B0NL AC4): a
                    # local-only client must never be moved remote — the
                    # caller turns the decision into the retryable 429.
                    if _preference == _PREF_LOCAL_ONLY:
                        fallback_reason = "local_admission_guard"
                        prev_provider = provider_name
                        all_slot_exhaustion = False
                        _note_local_fallback_reason(request, fallback_reason)
                        return _build_exhausted_response(
                            all_local_slot_exhaustion=False,
                            unavailable_providers={},
                            diagnostics=attempts,
                            config=config,
                        )
                    fallback_reason = "local_admission_guard"
                    prev_provider = provider_name
                    all_slot_exhaustion = False
                    _note_local_fallback_reason(request, fallback_reason)
                    continue

                response = await _dispatch_local(ptr_local, request, path, local_endpoint)

                # LP-0MTBOX45O005LD1S AC2: the cheap-mode compaction gate
                # (429) is TERMINAL.  Without this, the local-4xx branch
                # below would ``continue`` to the next provider → a silent
                # remote fallback for a request the gate exists to reject.
                if _is_compaction_gate_response(response):
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="compaction_gate",
                        status_code=int(getattr(response, "status_code", 0) or 0),
                    )
                    return response

            else:
                # Per-entry context ceiling (LP-0MVF9N2QL001CDA7 AC2/AC5/AC6):
                # same gate as the remote-only cycle — an over-limit entry is
                # skipped BEFORE the budget gate and dispatch, records no
                # failure-side state, and the chain proceeds to the next
                # eligible provider. The routing estimate is the shared
                # per-cycle one, so a local probe above already paid for it.
                _entry_limit = _provider_context_limit(provider_cfg)
                if _entry_limit > 0:
                    _entry_estimate = await _routing_estimate.get()
                    _context_skip, _context_limit = _should_skip_provider_context(
                        provider_cfg, _entry_estimate,
                    )
                    if _context_skip:
                        _record_provider_context_skip(
                            attempts, provider_name, provider_type,
                            _entry_estimate, _context_limit, _session_id,
                        )
                        fallback_reason = "provider_context_exceeded"
                        prev_provider = provider_name
                        all_slot_exhaustion = False
                        continue

                # Capacity-aware request budget for free-tier style entries
                # (LP-0MVDQ1GMO00801PL). See _proxy_with_remote_fallback_cycle:
                # new sessions pass straight through, in-progress sessions wait
                # a bounded time for the window to reset.
                _budget = await acquire_provider_budget(
                    provider_cfg, config, _session_id, request,
                    model_config=model_config,
                )
                if not _budget.allowed:
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type="remote",
                        status="rate_limited",
                        status_code=429,
                        reason=_budget.reason,
                        limit_rpm=_budget.limit_rpm,
                        effective_limit=_budget.effective_limit,
                        reserve_slots=_budget.reserve_slots,
                        remaining=_budget.remaining,
                        elapsed_wait_seconds=round(_budget.waited_seconds, 3),
                        **_budget_diagnostics(_budget),
                    )
                    fallback_reason = "rate_limited"
                    prev_provider = provider_name
                    all_slot_exhaustion = False
                    continue
                if _budget.waited:
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type="remote",
                        status="rate_limited_waited",
                        reason=_budget.reason,
                        limit_rpm=_budget.limit_rpm,
                        effective_limit=_budget.effective_limit,
                        reserve_slots=_budget.reserve_slots,
                        remaining=_budget.remaining,
                        elapsed_wait_seconds=round(_budget.waited_seconds, 3),
                        **_budget_diagnostics(_budget),
                    )

                response = await ptr_remote(request, path, provider_cfg)
                _maybe_record_provider_affinity(
                    provider_cfg, _session_id, response, config,
                    model_config=model_config,
                )

            # Pre-flight remote streaming responses: detect a pre-content
            # finish_reason: error / empty / stream-exception so the fallback
            # chain re-routes to the next provider instead of surfacing a
            # bare error to the client (LP-0MSETOTWY000SU0Z, Recommendation 1).
            if provider_type == "remote" and _has_next_provider(
                model_config, attempted_provider_names, attempted_domains,
                config=config,
            ):
                try:
                    response = await _preflight_streaming_response(
                        response, request, provider_name
                    )
                except StreamingPreContentError as exc:
                    mark_provider_unavailable(provider_name, cooldown_seconds)
                    # Sibling-fallback circuit breaker (LP-0MTPMF03P0046MFG):
                    # a pre-content stream error (empty_response / finish_reason:
                    # error with zero content) counts toward the consecutive-
                    # failure streak so the provider gets an extended cooldown
                    # after the threshold and the retry cycle skips to a sibling.
                    _record_sibling_failure(
                        provider_name, config, provider_cfg.get("provider"),
                        domain_key=_failure_domain_key(provider_cfg)
                    )
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
                    # Sibling-fallback circuit breaker (LP-0MTPMF03P0046MFG):
                    # reasoning was delivered but the provider stalled before
                    # usable final content; repeated stalls on the same provider
                    # count toward the streak so it is skipped for a sibling
                    # after the threshold.
                    _record_sibling_failure(
                        provider_name, config, provider_cfg.get("provider"),
                        domain_key=_failure_domain_key(provider_cfg)
                    )
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
                # LP-0MTXEBQ4E001BMI2: local success resets the deterministic
                # local-400 streak (only unbroken same-session 400 runs escalate).
                if provider_type == "local":
                    _reset_local_http_400_streak(_session_id, provider_name)
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
                    # A lease denial is a LOCAL CAPACITY condition, not a
                    # provider failure. With the queue policy, wait for the
                    # protected local slot (bounded by
                    # contention_queue_max_wait_seconds) before falling
                    # through to a remote chain that may be entirely out of
                    # window (LP-0MU5BOMJM008F17C).
                    _lease_cur, _lease_max = _local_concurrency_info(
                        config, endpoint=local_endpoint
                    )
                    _lease_action, _lease_reason, _lease_elapsed = (
                        await _maybe_queue_for_local_slot(
                            config, _lease_cur, _lease_max, request, body_json,
                            model_config, provider_cfg, _session_id,
                            endpoint=local_endpoint,
                        )
                    )
                    if _lease_action == "dispatch":
                        # A slot freed within the budget: re-dispatch local
                        # with the queue wait subtracted from the adaptive
                        # timeout budget.
                        _set_queue_wait_on_request(request, _lease_elapsed)
                        try:
                            from proxy import contention_queue as _cq_mod
                            _lease_depth = _cq_mod.queue_depth()
                        except Exception:
                            _lease_depth = 0
                        logger.info(
                            "contention_queue_dispatch provider=%s session=%s "
                            "queued_duration=%.2fs policy=queue depth=%d "
                            "reason=local_lease_active",
                            provider_name,
                            _session_id or "unknown",
                            _lease_elapsed or 0.0,
                            _lease_depth,
                        )
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="local_lease_queued",
                            elapsed_wait_seconds=round(_lease_elapsed or 0.0, 3),
                        )
                        response = await _dispatch_local(
                            ptr_local, request, path, local_endpoint
                        )
                        body_text = _response_body_text(response)
                        slot_info = _parse_slot_exhaustion(response)
                        if slot_info is not None and _is_local_lease_active_response(
                            response
                        ):
                            # Still denied after the wait: fall through to the
                            # remote chain (bounded, no further waiting).
                            all_slot_exhaustion = False
                            continue
                        # Slot freed — evaluate the fresh response below.
                    elif _lease_action == "fallback":
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="fallback_after_queue",
                            elapsed_wait_seconds=round(_lease_elapsed or 0.0, 3),
                        )
                        logger.info(
                            "contention_queue_fallback_after_queue provider=%s "
                            "session=%s queued_duration=%.2fs "
                            "reason=local_lease_active",
                            provider_name,
                            _session_id or "unknown",
                            _lease_elapsed or 0.0,
                        )
                        all_slot_exhaustion = False
                        continue
                    elif _lease_action == "context_bypass":
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="cached_tokens_skip",
                            reason=_lease_reason,
                        )
                        fallback_reason = _lease_reason
                        all_slot_exhaustion = False
                        continue
                    elif _lease_action == "admission_reject":
                        # Prefill-ETA admission guard (LP-0MVGDUV5E005UWBQ) on
                        # the lease-denial path: refuse the wait immediately.
                        _admission = getattr(
                            request, "_local_admission_decision", None
                        )
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="local_admission_reject",
                            reason=_lease_reason,
                            eta_prefill_seconds=round(
                                float(getattr(_admission, "eta_prefill_seconds", 0.0) or 0.0), 1
                            ),
                            budget_seconds=round(
                                float(getattr(_admission, "budget_seconds", 0.0) or 0.0), 1
                            ),
                        )
                        fallback_reason = "local_admission_guard"
                        all_slot_exhaustion = False
                        _note_local_fallback_reason(request, fallback_reason)
                        continue
                    elif _lease_action == "refused_local_only":
                        # AC4 (LP-0MVIKPCSG002B0NL): local-only + guard refusal
                        # on the lease path — return the 429 instead of
                        # falling through to a remote provider.
                        _admission = getattr(
                            request, "_local_admission_decision", None
                        )
                        _note_local_fallback_reason(request, _lease_reason)
                        _refusal_attempt = {
                            "status": "local_admission_reject",
                            "reason": _lease_reason or "eta_exceeds_budget",
                            "eta_prefill_seconds": round(
                                float(getattr(_admission, "eta_prefill_seconds", 0.0) or 0.0), 1
                            ),
                            "budget_seconds": round(
                                float(getattr(_admission, "budget_seconds", 0.0) or 0.0), 1
                            ),
                            "projected_wait_seconds": round(
                                float(getattr(_admission, "projected_wait_seconds", 0.0) or 0.0), 1
                            ),
                        }
                        return _build_local_only_refusal_response(
                            _refusal_attempt, config
                        )
                    else:
                        # Queue disabled / fallback policy: preserve the
                        # pre-existing immediate remote fallback.
                        all_slot_exhaustion = False
                        continue

                # A successful post-dispatch lease re-dispatch clears
                # ``slot_info``; fall through to the common response handling
                # instead of the slot-retry paths (LP-0MU5BOMJM008F17C).
                if slot_info is None:
                    pass
                # Optional local retry window for startup races where router/model
                # is loaded but slot probes briefly report 0 available.
                elif provider_type == "local" and local_slot_retry_attempts > 0:
                    resolved_after_retry = False
                    for retry_idx in range(1, local_slot_retry_attempts + 1):
                        if local_slot_retry_delay_seconds > 0:
                            await asyncio.sleep(local_slot_retry_delay_seconds)

                        retry_response = await _dispatch_local(ptr_local, request, path, local_endpoint)
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
                # GPU OOM detection for local backends (LP-0MRPILSMW004T4H8
                # AC5b): a wedged server surfaces memory-exhaustion patterns in
                # its error body. When detected, put the provider in cooldown
                # so the fallback chain skips it instead of retrying a server
                # that is OOM-thrashing.
                if provider_type == "local" and _check_local_backend_gpu_oom(
                    local_endpoint, body_text
                ):
                    logger.warning(
                        "local_gpu_oom provider=%s endpoint=%s status=%s",
                        provider_name, local_endpoint, response.status_code,
                    )
                    mark_provider_unavailable(provider_name, slot_unavailable_cooldown)
                    attempted_domains.add(_failure_domain_key(provider_cfg))
                    _record_attempt(
                        attempts,
                        provider=provider_name,
                        type=provider_type,
                        status="gpu_oom",
                        status_code=int(response.status_code),
                        body_snippet=(body_text[:512] if body_text else None),
                    )
                    fallback_reason = "gpu_oom"
                    prev_provider = provider_name
                    all_slot_exhaustion = False
                    continue

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

                        retry_response = await _dispatch_local(ptr_local, request, path, local_endpoint)
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
                        # LP-0MTXEBQ4E001BMI2: local 400s are otherwise silent —
                        # log the upstream body snippet (AC1) and escalate when
                        # the same session repeats the 400 (AC2), so a
                        # deterministic compacted-history rejection never routes
                        # remote invisibly turn after turn.
                        if int(response.status_code) == 400:
                            _observe_local_http_error_400(
                                response,
                                provider_name,
                                path,
                                body_text,
                                _session_id,
                            )
                        continue

                    # Usage-limit reset (LP-0MSLJPOCC0001ROJ): GoUsageLimitError /
                    # FreeUsageLimitError-with-reset-time quarantines the failing
                    # API-key ACCOUNT until the computed reset time + 2m margin.
                    # Not the whole endpoint: distinct api_key_env entries on the
                    # same gateway have independent limits (LP-0MSMBWB23009XYPW).
                    _reset_info = _usage_limit_reset_info(
                        response, body_text, config, provider_cfg=provider_cfg
                    )
                    if _reset_info is not None:
                        _reset_account = _usage_limit_account_key(provider_cfg)
                        _set_usage_limit_quarantine(_reset_account, _reset_info)
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
                            reset_in_seconds=int(_reset_info.seconds),
                        )
                        all_slot_exhaustion = False
                        continue

                    # FreeUsageLimitError: apply cooldown on affected provider
                    # so the fallback chain routes to paid alternatives instead of
                    # repeatedly retrying the exhausted free tier (LP-0MRGU0I91006ODFD).
                    if _is_free_usage_limit_error(response, body_text):
                        fallback_reason = "free_usage_limit"
                        prev_provider = provider_name
                        cooldown_seconds = _FREE_USAGE_LIMIT_COOLDOWN_SECONDS
                        mark_provider_unavailable(provider_name, cooldown_seconds)
                        attempted_domains.add(_failure_domain_key(provider_cfg))
                        _record_attempt(
                            attempts,
                            provider=provider_name,
                            type=provider_type,
                            status="free_usage_limit",
                            status_code=int(response.status_code),
                            body_snippet=(body_text[:512] if body_text else None),
                            cooldown_seconds=cooldown_seconds,
                        )
                        all_slot_exhaustion = False
                        continue

                    # Shared primitive: HTTP error with cooldown
                    _handle_http_error_with_cooldown(
                        response, provider_name, provider_type,
                        cooldown_seconds, attempts, body_text, config,
                        provider_cfg=provider_cfg,
                    )
                    # Per-account rate limit (LP-0MVCX0E2V001VB5N): a 429 with
                    # body type=rate_limit_error / code=rate_limit_exceeded is a
                    # per-ACCOUNT quota event (independent per api_key_env), NOT
                    # an endpoint-level failure. Poison only the failing account
                    # key in the per-request exclusion set so paid siblings on
                    # the same gateway stay eligible (e.g. munitoai-free 429
                    # must not skip the paid munitoai key). Other HTTP errors
                    # keep poisoning the shared failure domain (AC2).
                    if _is_account_rate_limit_error(response, body_text):
                        attempted_domains.add(_usage_limit_account_key(provider_cfg))
                    else:
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
                            retry_response = await _dispatch_local(ptr_local, request, path, local_endpoint)
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
                                cooldown_seconds, attempts, body_text, config,
                            )
                            # LP-0MUNTOGLB005LUKY: pass provider_type so the
                            # breaker can skip local providers entirely.
                            _threshold_exceeded = _record_sibling_failure(
                                provider_name, config, provider_cfg.get("provider"),
                                domain_key=_failure_domain_key(provider_cfg),
                                provider_type=provider_type,
                            )
                            if _threshold_exceeded:
                                attempted_domains.add(_failure_domain_key(provider_cfg))
                            fallback_reason = "empty_response"
                            prev_provider = provider_name
                            all_slot_exhaustion = False
                            continue
                    else:
                        # Shared primitive: empty response with cooldown
                        _handle_empty_response_with_cooldown(
                            response, provider_name, provider_type,
                            cooldown_seconds, attempts, body_text, config,
                        )
                        # LP-0MUNTOGLB005LUKY: pass provider_type so the
                        # breaker can skip local providers entirely.
                        _threshold_exceeded = _record_sibling_failure(
                            provider_name, config, provider_cfg.get("provider"),
                            domain_key=_failure_domain_key(provider_cfg),
                            provider_type=provider_type,
                        )
                        if _threshold_exceeded:
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
            # LP-0MTXEBQ4E001BMI2: local success resets the deterministic
            # local-400 streak (only unbroken same-session 400 runs escalate).
            if provider_type == "local":
                _reset_local_http_400_streak(_session_id, provider_name)
            # LP-0MR4ZIGDT004A3E1: Surface resolved provider/model for Pi extension
            _add_resolved_model_header(result, provider_cfg)
            return result

        except Exception as exc:
            # Shared primitive: handle connection errors
            if _handle_connection_error_in_fallback(
                exc, provider_name, provider_type, cooldown_seconds, attempts, config,
            ):
                any_provider_tried = True
                attempted_domains.add(_connection_error_domain_key(provider_cfg, exc))
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
                            retry_response = await _dispatch_local(ptr_local, request, path, local_endpoint)
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

    # Usage-limit quarantine takes precedence over the window reason, so it
    # is recorded first; the window loop below skips providers already
    # quarantined (LP-0MU56ZFVD001LP0H).
    _record_quarantine_attempts(attempts, model_config)

    # Record time-window skips as distinct diagnostics so operators can tell
    # whether providers were excluded by their available_times windows rather
    # than by cooldown/errors (LP-0MS4ETBNO0022QAC).
    for skipped in _providers_outside_window(model_config, config):
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
        attempts, unavailable, any_provider_tried, model_config=model_config,
        config=config,
    )
    if time_window_exhausted is not None:
        raise ChainExhaustedError(time_window_exhausted)

    if not any_provider_tried:
        raise ChainExhaustedError(
            _build_exhausted_response(all_local_slot_exhaustion=False, unavailable_providers=unavailable, diagnostics=attempts, model_config=model_config, config=config)
        )

    # If all failures were slot exhaustion, include total slots in message
    if all_slot_exhaustion:
        raise ChainExhaustedError(
            _build_exhausted_response(all_local_slot_exhaustion=True, total_slots=total_slots_sum, unavailable_providers=unavailable, diagnostics=attempts, model_config=model_config, config=config)
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
        _build_exhausted_response(all_local_slot_exhaustion=False, unavailable_providers=unavailable, diagnostics=attempts, model_config=model_config, config=config)
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
    response = await _run_chain_cycles(
        request, path, model_config, config, _proxy_with_fallback_cycle,
    )
    return _stamp_local_route_headers(response, request, model_config)
