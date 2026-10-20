"""Deep-merge helper for the layered proxy config.

``config.yaml`` is the single authoritative config since the mode overlays
(``config-fast.yaml`` / ``config-cheap.yaml``) were removed
(LP-0MVDT94F0003VHUA AC4). :func:`deep_merge` remains for consumers that
supply an optional overlay (backward compatibility) and for the restart-
signature helper.

This module is intentionally dependency-light — stdlib plus ``yaml`` only — so
the shell entry point ``proxy/scripts/start-proxy.sh`` can reuse the merge
semantics from a tiny inline Python snippet without importing the full
``proxy`` package (which pulls in FastAPI, httpx, tiktoken, …).

Merge semantics
---------------
* ``dict`` merged with ``dict`` → recursive merge; keys unique to either side
  are preserved.
* Any other value (scalars, lists, ``None``) → the overlay value replaces the
  base value **wholesale**. Lists are never merged element-wise; an overlay
  provider list replaces the base provider list entirely.
* The inputs are never mutated; a new dict is returned.

Restart signature
-----------------
The shared restart-signature helper :func:`merged_config_sha256` hashes the
effective (merged) config as canonical JSON — deterministic, key-sorted,
formatting-agnostic — so both the shell read path and the Python write path
produce byte-identical hashes for the same config inputs (LP-0MUJRVYQH002IEHW).

Typical use::

    from proxy.config_merge import deep_merge

    base = yaml.safe_load(open("config.yaml"))
    overlay = yaml.safe_load(open("config.yaml"))  # no overlay: base alone
    cfg = deep_merge(base, overlay)
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, overlay: dict) -> dict:
    """Return a new dict with *overlay* merged on top of *base*.

    Args:
        base: The authoritative base mapping (``config.yaml``).
        overlay: The mode overlay mapping (``config-fast.yaml`` /
            ``config-cheap.yaml``).

    Returns:
        A newly allocated ``dict``. Nested mappings are merged recursively;
        every other overlay value (including ``None``) replaces the
        corresponding base value. ``base`` and ``overlay`` are not mutated,
        even transitively.

    Raises:
        TypeError: If *base* or *overlay* is not a ``dict``. Callers loading
            YAML should normalise an empty document (``None``) to ``{}``
            before merging.
    """
    if not isinstance(base, dict):
        raise TypeError(
            f"deep_merge base must be a dict, got {type(base).__name__}"
        )
    if not isinstance(overlay, dict):
        raise TypeError(
            f"deep_merge overlay must be a dict, got {type(overlay).__name__}"
        )

    result: dict[str, Any] = deepcopy(base)
    for key, value in overlay.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml_config(path) -> dict:
    """Load a YAML config file as a ``dict`` (empty document -> ``{}``)."""
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def load_merged(base_path, overlay_path) -> dict:
    """Load *base_path* and deep-merge *overlay_path* on top of it.

    A dependency-light convenience for shell consumers (``start-proxy.sh``
    and its test) that must resolve the effective config without importing
    the full ``proxy`` package. Mirrors ``proxy.utils._load_merged_config``
    for the common case where the overlay exists; callers that need the
    missing-overlay fallback and warning should use that loader instead.
    """
    return deep_merge(
        load_yaml_config(Path(base_path)),
        load_yaml_config(Path(overlay_path)),
    )


def merged_config_sha256(base_path, overlay_path) -> str:
    """Canonical sha256 of the merged (base + overlay) config.

    The restart signature (LP-0MUJRVYQH002IEHW) must reflect the *effective*
    config: after the mode files became overlays (LP-0MUI67ZNM008627E),
    base-inherited local-backend keys (e.g. ``local_model_ctx_size``,
    ``session_slot_pool_size``, ``llama_server_port``) no longer appear in
    them, so a base-only change would not change the hash of the raw overlay.
    Hashing the merged dict instead makes a base-only change alter the
    signature and forces llama-server to restart when a ``--restart`` is
    issued.

    The canonical serialisation is ``json.dumps(merged, sort_keys=True,
    separators=(",", ":"))`` — deterministic key ordering makes the hash
    stable across Python runs and independent of YAML formatting/comments
    (only effective values matter). Both ``start-proxy.sh`` (shell read
    path) and ``lifecycle._config_file_sha256`` (Python write path) must use
    this exact helper so their hashes are byte-identical for the same config
    inputs; do not change the serialisation on one side only.

    Args:
        base_path: Path of the authoritative base ``config.yaml``.
        overlay_path: Path of the mode overlay (``config-<mode>.yaml``). When
            the overlay file is absent, the base is merged with itself
            (equivalent to an empty overlay).

    Returns:
        Hex sha256 of the canonical JSON serialisation of the merged dict.
    """
    base_path = Path(base_path)
    overlay_path = Path(overlay_path)
    if overlay_path.is_file():
        merged = load_merged(base_path, overlay_path)
    else:
        # Missing overlay -> the effective config is the base alone.
        merged = load_yaml_config(base_path)
    canonical = json.dumps(merged, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
