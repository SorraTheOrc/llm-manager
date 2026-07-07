"""
Sliding-window rate limiter for remote providers.

Tracks request counts per provider (or other key) over a configurable
time window and rejects requests that exceed the configured maximum.

The limiter is designed for proactive throttling — it checks the rate
limit *before* a request is dispatched, preventing the proxy from ever
sending more than X requests per minute to a given upstream endpoint.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Dict

logger = logging.getLogger("llama-proxy.rate_limiter")


class SlidingWindowRateLimiter:
    """Sliding-window rate limiter for per-provider request throttling.

    Each key (typically a provider name) has an independent sliding window
    counter. Old entries are pruned on each check so the window reflects
    only the most recent ``window_seconds`` of activity.

    Thread-safe via ``asyncio.Lock``.
    """

    def __init__(self) -> None:
        # key (str) -> deque of monotonic timestamps
        self._windows: Dict[str, "deque[float]"] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check_and_increment(
        self,
        key: str,
        max_rpm: int,
        window_seconds: int = 60,
    ) -> bool:
        """Check if a request is allowed and record it if so.

        Args:
            key: Provider name being rate-limited.
            max_rpm: Maximum requests allowed per window. 0 means unlimited.
            window_seconds: Sliding-window duration in seconds (default 60).

        Returns:
            ``True`` if the request is within the limit (counter incremented),
            ``False`` if the request should be rejected (rate limited).
        """
        if max_rpm <= 0:
            return True  # No limit

        now = time.monotonic()
        cutoff = now - window_seconds

        async with self._lock:
            window = self._windows[key]

            # Prune entries outside the window
            while window and window[0] < cutoff:
                window.popleft()

            if len(window) >= max_rpm:
                logger.warning(
                    "Rate limit exceeded for %s: %d/%d requests in %ds window",
                    key,
                    len(window),
                    max_rpm,
                    window_seconds,
                )
                return False

            window.append(now)
            return True

    def remaining(self, key: str, max_rpm: int, window_seconds: int = 60) -> int:
        """Return the number of requests still available in the current window.

        This is a best-effort snapshot (no lock) useful for diagnostics.

        Args:
            key: Provider name.
            max_rpm: Maximum requests per window.
            window_seconds: Window duration in seconds.

        Returns:
            Remaining request capacity (``max_rpm - current_count``).
        """
        if max_rpm <= 0:
            return 0  # Unlimited — 0 remaining is misleading; return 0 for display

        cutoff = time.monotonic() - window_seconds
        window = self._windows.get(key, deque())
        # Count entries still inside the window (pruning expired on read)
        while window and window[0] < cutoff:
            window.popleft()
        used = len(window)
        return max(0, max_rpm - used)

    def reset(self, key: str) -> None:
        """Remove all tracked data for *key* (useful in tests)."""
        self._windows.pop(key, None)


# ---------------------------------------------------------------------------
# Global instance
# ---------------------------------------------------------------------------

_rate_limiter = SlidingWindowRateLimiter()


def get_rate_limiter() -> SlidingWindowRateLimiter:
    """Return the global ``SlidingWindowRateLimiter`` singleton."""
    return _rate_limiter
