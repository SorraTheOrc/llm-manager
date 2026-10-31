"""Shared helpers for tests that consume the proxy config.

``config.yaml`` is the single authoritative config since the mode-profile
overlay system (``config-fast.yaml`` / ``config-cheap.yaml``) was removed with
the fast/cheap mode axis (LP-0MVDT94F0003VHUA AC4). ``load_config()`` and
``load_merged_config()`` therefore resolve to the base alone; these helpers
retain the legacy ``get_merged_config(mode)`` / ``load_profile(name)``
signatures so existing tests keep working against the unified config.
"""

import yaml
from proxy.priority import proxy_dir
from proxy.utils import _load_merged_config


def get_merged_config(mode: str) -> dict:
    """Return the runtime config for *mode* (``fast`` / ``cheap``).

    Legacy mode argument retained for backward compatibility: with the mode
    axis removed there is a single unified config, so both values resolve to
    the base ``config.yaml`` merged with itself (identical to
    ``load_config()``).
    """
    return _load_merged_config(proxy_dir() / "config.yaml")


#: Server-block routing geometry the shipped ``config.yaml`` runs with, as a
#: fixture mirror for the single-slot threshold derivations
#: (LP-0MVTV6F65005HKQZ). The values are literals on purpose: the derivation
#: tests pin absolute token counts against them, and
#: ``test_ctx_slot_validation.TestShippedGeometryMatchesPinnedFixture`` fails
#: loudly (naming the key and both values) when the shipped block drifts.
SHIPPED_ROUTING_GEOMETRY: dict[str, object] = {
    "local_large_context_cold_cache_threshold": 38000,
    "local_large_context_warm_cache_threshold": 120000,
    "local_large_context_economic_ceiling_ratio": 0.4578,
    "local_model_ctx_size": 262144,
    "session_slot_pool_size": 2,
}


def geometry_config(slots: int) -> dict:
    """Return a config dict mirroring the shipped server block at *slots* slots.

    ``local_model_ctx_size`` (262144) is held fixed so the slot count is the
    only variable — exactly the geometry a single-slot switch changes
    (LP-0MVTV6F65005HKQZ). The retired hard-routing-cap ratio keys
    (``local_hard_routing_cap_ratio_fast`` / ``_cheap``) are intentionally
    absent, matching the shipped config where the cap is disabled (0).

    Args:
        slots: Number of local slots in the geometry (>= 1).

    Returns:
        A ``{"server": {...}}`` config dict.

    Raises:
        ValueError: when *slots* is not a positive integer.
    """
    if slots < 1:
        raise ValueError(f"slots must be >= 1, got {slots}")
    return {
        "server": {
            **SHIPPED_ROUTING_GEOMETRY,
            "session_slot_pool_size": slots,
        }
    }


def load_profile(name: str) -> dict:
    """Return the config for a profile file name, as seen at runtime.

    ``config.yaml`` is returned raw (it is the authoritative base). A legacy
    mode file name (``config-fast.yaml`` / ``config-cheap.yaml``) is treated
    as the unified config (the overlays no longer exist).
    """
    if name == "config.yaml":
        with open(proxy_dir() / name) as fh:
            return yaml.safe_load(fh)
    return get_merged_config("fast")
