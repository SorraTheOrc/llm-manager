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
