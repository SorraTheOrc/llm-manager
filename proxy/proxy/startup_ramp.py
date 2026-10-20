"""Post-restart startup ramp (LP-0MU9ZXFQS0023DXT).

After ANY proxy restart (operator restart, model switch), competing
herdr/agent workers reconnect simultaneously, creating a thundering herd
that overwhelms the freshly-started proxy. The startup ramp gates new chat
requests with a short 503 + random Retry-After so reconnects spread across
the ramp window instead of hitting the server at once. Config lives in
``server.startup_ramp``:

    enabled: true, max_seconds: 30, jitter: [5, 15]

Set ``enabled: false`` or ``max_seconds: 0`` to disable entirely.

This module was extracted from the retired ``proxy.mode`` package
(LP-0MVDT94F0003VHUA AC4 — the fast/cheap mode machinery, including the
mode-switch-triggered ramp motivation, was removed; the ramp itself is a
generic restart mitigation and is retained).
"""

import logging
import time

logger = logging.getLogger("llama-proxy")

STARTUP_RAMP_DEFAULT_ENABLED = True
STARTUP_RAMP_DEFAULT_MAX_SECONDS = 30.0
STARTUP_RAMP_DEFAULT_JITTER_MIN = 5.0
STARTUP_RAMP_DEFAULT_JITTER_MAX = 15.0

_startup_ramp_config: dict | None = None  # resolved from config


def set_startup_ramp_config(cfg: dict | None) -> None:
    """Store the resolved startup_ramp config (called once at server startup).

    *cfg* is a dict with ``enabled``, ``max_seconds``, ``jitter_min``,
    ``jitter_max`` — or None to reset to defaults.
    """
    global _startup_ramp_config
    if cfg is None:
        _startup_ramp_config = {
            "enabled": STARTUP_RAMP_DEFAULT_ENABLED,
            "max_seconds": STARTUP_RAMP_DEFAULT_MAX_SECONDS,
            "jitter_min": STARTUP_RAMP_DEFAULT_JITTER_MIN,
            "jitter_max": STARTUP_RAMP_DEFAULT_JITTER_MAX,
        }
        return
    _startup_ramp_config = cfg


def _startup_ramp_config_section(server_config: dict | None) -> dict:
    """Resolve the ``server.startup_ramp`` config with defaults.

    Args:
        server_config: The ``server`` section of the server config dict,
            or None to read it from the live server module lazily.

    Returns:
        A dict with ``enabled``, ``max_seconds``, ``jitter_min``,
        ``jitter_max`` keys.
    """
    if server_config is None:
        server_config = {}
    section = server_config.get("startup_ramp") or {}
    enabled = bool(section.get("enabled", STARTUP_RAMP_DEFAULT_ENABLED))
    try:
        max_seconds = float(section.get("max_seconds", STARTUP_RAMP_DEFAULT_MAX_SECONDS) or 0)
    except (TypeError, ValueError):
        max_seconds = STARTUP_RAMP_DEFAULT_MAX_SECONDS
    try:
        jitter_min = float(section.get("jitter_min", STARTUP_RAMP_DEFAULT_JITTER_MIN) or 0)
    except (TypeError, ValueError):
        jitter_min = STARTUP_RAMP_DEFAULT_JITTER_MIN
    try:
        jitter_max = float(section.get("jitter_max", STARTUP_RAMP_DEFAULT_JITTER_MAX) or 0)
    except (TypeError, ValueError):
        jitter_max = STARTUP_RAMP_DEFAULT_JITTER_MAX
    return {
        "enabled": enabled,
        "max_seconds": max(0.0, max_seconds),
        "jitter_min": max(0.0, min(jitter_min, jitter_max)),
        "jitter_max": max(0.0, max(jitter_min, jitter_max)),
    }


def startup_ramp_active() -> bool:
    """Whether the configured ramp window is currently active.

    True when the ramp config is enabled, has a positive ``max_seconds``,
    and the process is within the ramp window (``PROXY_START_TIME`` set by
    server.py's lifespan handler). Returns False on any error (fail-open:
    a missing config or clock never blocks requests).
    """
    cfg = _startup_ramp_config
    if not cfg or not cfg.get("enabled") or cfg.get("max_seconds", 0) <= 0:
        return False
    try:
        import proxy.server as _srv

        start = float(getattr(_srv, "PROXY_START_TIME", 0.0) or 0.0)
        if start <= 0:
            return False
        return (time.monotonic() - start) < cfg["max_seconds"]
    except Exception:
        return False
