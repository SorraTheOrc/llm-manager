"""Cold-start admission control for local dispatch (LP-0MUCEFCAT005NFNN).

Context
-------
After a proxy restart the co-located llama-server reloads the model cold:
``slots_stale=true``, no prompt-cache entries, and the KV cache must be
re-prefilled. If several large-context sessions dispatch at the same time
they each issue a full prefill concurrently on one GPU; first bytes then
take minutes, clients time out and retry, and the local pool wedges (see
the parent epic LP-0MUCEEZ6B003NZH5).

This module tracks a bounded *cold-start grace window* that begins when the
backend transitions from not-ready to ready (model just loaded). During that
window local dispatch concurrency is capped (default 1) so only one large
prefill runs at a time; the cache warms after the first request completes
prefill and the cap lifts immediately (``mark_warm``).

The state machine is intentionally tiny and fail-open:

- ``note_model_loaded()`` — called by the lifecycle load paths when a model
  becomes ready; arms the cold window with a fresh grace clock.
- ``is_cold(config)`` — True only while cold **and** within the configured
  grace window; self-expiring so a stuck flag cannot throttle forever.
- ``mark_warm()`` — lifts the cap as soon as a local prefill completes
  (first data byte), since the prompt cache is warm from then on.
- ``note_backend_down()`` — clears the cold state on a restart/crash so the
  next load re-arms it.

All functions are cheap (no I/O) and thread-safe enough for the asyncio
event loop (single-threaded; the dict update is atomic under the GIL).
"""

from __future__ import annotations

import time

# Defaults (overridable via the ``server`` config section).
COLD_START_DEFAULT_GRACE_SECONDS = 120.0
COLD_START_DEFAULT_MAX_CONCURRENT = 1
COLD_START_DEFAULT_RETRY_AFTER_SECONDS = 15


class _ColdStartState:
    """Tiny mutable state holder for the cold-start window."""

    __slots__ = ("cold", "loaded_at")

    def __init__(self) -> None:
        self.cold: bool = False
        self.loaded_at: float | None = None


_state = _ColdStartState()


def reset() -> None:
    """Reset all cold-start state (tests / process restart)."""
    _state.cold = False
    _state.loaded_at = None


def mark_warm() -> None:
    """Lift the cold-start cap (the local prompt cache is now warm)."""
    _state.cold = False


def note_model_loaded(now: float | None = None) -> None:
    """Arm the cold-start window because a model was just loaded/switched.

    Called by the lifecycle load paths (``ensure_model_loaded``,
    ``restart_services``, startup default-model load) at the exact point the
    model becomes ready — not from the request path — so a request that
    arrives long after the load is not falsely treated as cold.
    """
    _state.cold = True
    _state.loaded_at = time.monotonic() if now is None else now


def note_backend_down() -> None:
    """Record that the backend is no longer ready (restart/crash).

    Clears the cold state so the next ``note_model_loaded`` re-arms it with
    a fresh grace clock.
    """
    _state.cold = False
    _state.loaded_at = None


def _server_section(config: dict | None) -> dict:
    if not isinstance(config, dict):
        return {}
    section = config.get("server")
    return section if isinstance(section, dict) else config


def grace_seconds(config: dict | None) -> float:
    """Return the configured cold-start grace window (default 120s)."""
    section = _server_section(config)
    try:
        value = section.get(
            "local_cold_start_grace_seconds", COLD_START_DEFAULT_GRACE_SECONDS
        )
        return float(value) if value is not None else COLD_START_DEFAULT_GRACE_SECONDS
    except (TypeError, ValueError):
        return COLD_START_DEFAULT_GRACE_SECONDS


def max_concurrent(config: dict | None) -> int:
    """Return the concurrent-local-dispatch cap while cold (default 1, min 1)."""
    section = _server_section(config)
    try:
        value = section.get(
            "local_cold_start_max_concurrent", COLD_START_DEFAULT_MAX_CONCURRENT
        )
        return max(1, int(value)) if value is not None else COLD_START_DEFAULT_MAX_CONCURRENT
    except (TypeError, ValueError):
        return COLD_START_DEFAULT_MAX_CONCURRENT


def retry_after_seconds(config: dict | None) -> int:
    """Return the Retry-After hint for a cold-start deferral (default 15s)."""
    section = _server_section(config)
    try:
        value = section.get(
            "local_cold_start_retry_after_seconds",
            COLD_START_DEFAULT_RETRY_AFTER_SECONDS,
        )
        return max(1, int(value)) if value is not None else COLD_START_DEFAULT_RETRY_AFTER_SECONDS
    except (TypeError, ValueError):
        return COLD_START_DEFAULT_RETRY_AFTER_SECONDS


def is_cold(config: dict | None = None, now: float | None = None) -> bool:
    """Whether the local backend is in its post-load cold window.

    Returns False when the cold window is disabled (grace <= 0), the backend
    is not cold, or the grace window has elapsed (self-expiring).
    """
    if not _state.cold:
        return False
    grace = grace_seconds(config)
    if grace <= 0:
        return False
    loaded_at = _state.loaded_at
    if loaded_at is None:
        return False
    current = time.monotonic() if now is None else now
    return (current - loaded_at) < grace


def effective_max_concurrent(config: dict | None, configured_max: int) -> int:
    """Clamp *configured_max* to the cold-start cap while cold.

    Returns *configured_max* unchanged when warm or when the cold cap is not
    lower — so warm operation has no added latency or restriction.
    """
    if not is_cold(config):
        return configured_max
    return min(configured_max, max_concurrent(config))
