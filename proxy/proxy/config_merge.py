"""Deep-merge helper for the layered proxy config (base + mode overlay).

``config.yaml`` is the authoritative base config. The mode profiles
(``config-fast.yaml`` / ``config-cheap.yaml``) are overlays that override only
the values that genuinely differ; :func:`deep_merge` composes the two.

This module is intentionally dependency-light — stdlib plus nothing else — so
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

Typical use::

    from proxy.config_merge import deep_merge

    base = yaml.safe_load(open("config.yaml"))
    overlay = yaml.safe_load(open("config-cheap.yaml"))
    cfg = deep_merge(base, overlay)
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


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
