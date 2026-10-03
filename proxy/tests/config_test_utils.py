"""Shared helpers for tests that consume the proxy config.

``config.yaml`` is the authoritative base config; the mode profiles
(``config-fast.yaml`` / ``config-cheap.yaml``) are overlays merged on top of
it by ``load_config()``. Tests must assert against the *merged* surface rather
than parsing the raw overlay files, otherwise they break whenever a value is
deduplicated into the base.
"""

import yaml
from proxy.mode import mode_config_file, proxy_dir
from proxy.utils import _load_merged_config


def get_merged_config(mode: str) -> dict:
    """Return the runtime config for *mode* (``fast`` / ``cheap``).

    Uses the production merge path (``proxy.utils._load_merged_config`` with
    the mode's overlay file) so tests and runtime cannot drift.
    """
    return _load_merged_config(mode_config_file(mode))


def load_profile(name: str) -> dict:
    """Return the config for a profile file name, as seen at runtime.

    ``config.yaml`` is returned raw (it is the authoritative base); a mode
    file name (``config-fast.yaml`` / ``config-cheap.yaml``) is returned with
    the base merged in, because the mode files are overlays that no longer
    repeat inherited values.
    """
    if name == "config.yaml":
        with open(proxy_dir() / name) as fh:
            return yaml.safe_load(fh)
    return get_merged_config("fast" if "fast" in name else "cheap")
