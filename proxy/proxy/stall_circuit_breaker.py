"""
Stall Circuit Breaker Module

A sliding-window stall circuit breaker that tracks stall frequency across
requests per provider. When a provider exceeds the stall threshold within
the sliding time window, it is marked unavailable for a cooldown period
via the existing Tier 2 cooldown mechanism (``mark_provider_unavailable()``).

This is Tier 3 of the proxy's retry system:

- Tier 1: Per-stream retry (proxy_remote.py) — bounded exponential backoff
          on idle timeout or ReadTimeout, max 3 retries.
- Tier 2: Provider-level cooldown (provider.py) — after Tier 1 exhausts,
          the provider is marked unavailable for a cooldown period.
- Tier 3 (this module): Cross-request stall circuit breaker — tracks stall
          frequency across requests so unreliable providers are quarantined
          faster on subsequent requests, rather than starting from scratch
          on each new request.

The circuit breaker uses a sliding time window implemented as a deque of
timestamps per provider. Expired timestamps are pruned lazily on each
``record_stall()`` call. State is in-memory (no persistence), consistent
with the existing cooldown mechanism.

Thread-safety: Uses asyncio.Lock around shared state access to handle
concurrent stall recordings from multiple requests.
"""

import asyncio
import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional

import proxy.provider as provider_mod

logger = logging.getLogger("llama-proxy.stall_circuit_breaker")

# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------

_DEFAULT_WINDOW_SECONDS = 300
_DEFAULT_THRESHOLD = 3
_DEFAULT_COOLDOWN_SECONDS = 180

# ---------------------------------------------------------------------------
# Config reader helpers
# ---------------------------------------------------------------------------


def _get_circuit_breaker_config(config: dict) -> dict:
    """Read circuit breaker config keys from the server config.

    Supports both nested (``config["server"]["upstream_stall_*"]``) and flat
    formats for production and test compatibility.

    Returns a dict with keys: ``window_seconds``, ``threshold``, ``cooldown_seconds``.
    """
    server_cfg = config.get("server", {}) if isinstance(config, dict) else {}

    window_seconds = server_cfg.get("upstream_stall_window_seconds")
    if window_seconds is None:
        window_seconds = config.get("upstream_stall_window_seconds", _DEFAULT_WINDOW_SECONDS)

    threshold = server_cfg.get("upstream_stall_threshold")
    if threshold is None:
        threshold = config.get("upstream_stall_threshold", _DEFAULT_THRESHOLD)

    cooldown_seconds = server_cfg.get("upstream_stall_cooldown_seconds")
    if cooldown_seconds is None:
        cooldown_seconds = config.get("upstream_stall_cooldown_seconds", _DEFAULT_COOLDOWN_SECONDS)

    try:
        window_seconds = max(1, int(window_seconds or _DEFAULT_WINDOW_SECONDS))
    except (ValueError, TypeError):
        window_seconds = _DEFAULT_WINDOW_SECONDS

    try:
        threshold = max(1, int(threshold or _DEFAULT_THRESHOLD))
    except (ValueError, TypeError):
        threshold = _DEFAULT_THRESHOLD

    try:
        cooldown_seconds = max(1, int(cooldown_seconds or _DEFAULT_COOLDOWN_SECONDS))
    except (ValueError, TypeError):
        cooldown_seconds = _DEFAULT_COOLDOWN_SECONDS

    return {
        "window_seconds": window_seconds,
        "threshold": threshold,
        "cooldown_seconds": cooldown_seconds,
    }


# ---------------------------------------------------------------------------
# StallCircuitBreaker class
# ---------------------------------------------------------------------------


class StallCircuitBreaker:
    """Sliding-window stall circuit breaker for provider health tracking.

    Tracks stall events per provider within a sliding time window. When the
    number of stalls in the window exceeds the configured threshold, the
    provider is marked unavailable via the existing Tier 2 cooldown mechanism
    (``mark_provider_unavailable()``).

    The sliding window is implemented as a deque of timestamps per provider.
    Expired timestamps are pruned lazily on each ``record_stall()`` call.

    Thread-safe: Uses ``asyncio.Lock`` to protect shared state access.

    Args:
        config: Dict with keys ``window_seconds``, ``threshold``,
                ``cooldown_seconds`` (from ``_get_circuit_breaker_config()``).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or _get_circuit_breaker_config({})
        self._window_seconds: int = cfg.get("window_seconds", _DEFAULT_WINDOW_SECONDS)
        self._threshold: int = cfg.get("threshold", _DEFAULT_THRESHOLD)
        self._cooldown_seconds: int = cfg.get("cooldown_seconds", _DEFAULT_COOLDOWN_SECONDS)

        # Per-provider deque of stall timestamps (seconds since epoch)
        # Key: provider_name (str), Value: deque of float timestamps
        self._stall_timestamps: Dict[str, deque] = {}

        # asyncio lock for concurrent access safety
        self._lock = asyncio.Lock()

    def record_stall(self, provider_name: str) -> bool:
        """Record a stall event for the given provider.

        If the provider does not have a timestamp deque, one is created.
        Expired timestamps (older than window_seconds from now) are pruned
        lazily on each call. If the number of non-expired stalls equals or
        exceeds the threshold, the provider is marked unavailable via
        ``mark_provider_unavailable()``.

        Args:
            provider_name: Name of the provider that stalled.

        Returns:
            True if the circuit breaker triggered cooldown, False otherwise.
            Returns False if the provider is already in cooldown.
        """
        # Check if already in cooldown — if so, still record the stall
        # but do NOT extend cooldown (the existing cooldown mechanism
        # handles duration). Record the timestamp for post-cooldown tracking.
        already_in_cooldown = provider_mod._is_provider_unavailable(provider_name)

        now = time.time()

        # Get or create the deque for this provider
        if provider_name not in self._stall_timestamps:
            self._stall_timestamps[provider_name] = deque()

        timestamps = self._stall_timestamps[provider_name]

        # Prune expired timestamps (older than window_seconds)
        cutoff = now - self._window_seconds
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

        # Add the new stall timestamp
        timestamps.append(now)

        # Check if threshold exceeded
        if len(timestamps) >= self._threshold and not already_in_cooldown:
            provider_mod.mark_provider_unavailable(
                provider_name, float(self._cooldown_seconds)
            )
            logger.warning(
                "Stall circuit breaker triggered: provider=%s stalls=%d "
                "window=%ds threshold=%d cooldown=%ds",
                provider_name,
                len(timestamps),
                self._window_seconds,
                self._threshold,
                self._cooldown_seconds,
            )
            return True

        return False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

# Created once at import time with default config.
# The singleton is lazily re-initialized by _check_stall_circuit_breaker()
# which reads the live config on each call.
stall_circuit_breaker = StallCircuitBreaker()


# ---------------------------------------------------------------------------
# Integration function (called from proxy_remote.py)
# ---------------------------------------------------------------------------

# Sentinel to detect first-call initialization and avoid repeated config
# re-reads on every stall circuit breaker call.
_initialized = False


def _check_stall_circuit_breaker(provider_name: str, config: dict) -> bool:
    """Record a stall event and trigger cooldown if threshold exceeded.

    This is the integration point called from proxy_remote.py when per-stream
    retries exhaust.

    Args:
        provider_name: Name of the provider that stalled.
        config: Server configuration dict (to read circuit breaker config).

    Returns:
        True if the circuit breaker triggered cooldown, False otherwise.
    """
    global stall_circuit_breaker, _initialized

    # Re-initialize the singleton with live config on first call
    if not _initialized:
        cb_cfg = _get_circuit_breaker_config(config)
        stall_circuit_breaker = StallCircuitBreaker(cb_cfg)
        _initialized = True

    return stall_circuit_breaker.record_stall(provider_name)
