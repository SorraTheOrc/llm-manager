"""
Provider Resolver Module

Resolves audit model short names to provider-prefixed model identifiers
(e.g., ``deepseek-v4-flash-free`` → ``opencode/deepseek-v4-flash-free``).

Provides:
- ``resolve_audit_model(name, fallbacks=[])`` — Resolve a short model name
  to one or more provider-prefixed model IDs.
- ``validate_audit_models(config, strict=False)`` — Startup validation that
  attempts to resolve the primary audit model and optional fallbacks,
  logging warnings for unresolved names.
- Lightweight structured logging and a counter metric for unresolved names.

Configuration (see ``config.yaml``):
  audit_model: "deepseek-v4-flash-free"
  audit_model_fallbacks:
    - "openrouter/free"
    - "deepseek-v4-flash"

Canonical model IDs (discovered via ``pi --list-models``):
  opencode/deepseek-v4-flash-free
  opencode-go/deepseek-v4-flash
  openrouter/openrouter/free
  openrouter/deepseek/deepseek-v4-flash
  opencode/deepseek-v4-flash
"""

import logging
from typing import Any

logger = logging.getLogger("llama-proxy.provider-resolver")

# ---------------------------------------------------------------------------
# Metrics (lightweight — no external deps)
# ---------------------------------------------------------------------------

# Counter for unresolved model name lookups.
# Format: <short_name> -> count
_unresolved_counts: dict[str, int] = {}


def _record_unresolved(name: str) -> None:
    """Increment the unresolved counter for *name*."""
    _unresolved_counts[name] = _unresolved_counts.get(name, 0) + 1


def get_unresolved_counts() -> dict[str, int]:
    """Return a copy of the unresolved name counter map."""
    return dict(_unresolved_counts)


# ---------------------------------------------------------------------------
# Static mapping table
# ---------------------------------------------------------------------------

# Mapping of short / well-known model name patterns → provider-prefixed IDs.
# Keys are matched in order; the first match wins.
#
# Pattern types:
#   - Exact match: "deepseek-v4-flash-free"
#   - Prefix match: "deepseek-*"  (any name starting with "deepseek-")
#
# The mapping is intentionally static — no runtime ``pi --list-models``
# dependency at resolve time.  Update this table when new provider/model
# combinations are added.


def _resolve_opencode_go_candidates(name: str) -> list[str]:
    """Resolve opencode-go specific names."""
    if "/" in name:
        return ["opencode-go/" + name.split("/", 1)[-1]]
    return ["opencode-go/" + name]


def _resolve_opencode_candidates(name: str) -> list[str]:
    """Resolve opencode specific names."""
    stripped = name.replace("opencode-", "").replace("opencode/", "")
    return ["opencode/" + stripped]


_MODEL_RESOLVERS: list[tuple] = [
    # DeepSeek v4 flash — free / no-key tier
    (
        lambda name: name == "deepseek-v4-flash-free" or name == "deepseek-v4-flash-free*",
        [
            "opencode/deepseek-v4-flash-free",
            "openrouter/openrouter/free",
            "opencode-go/deepseek-v4-flash",
        ],
    ),
    # DeepSeek v4 flash — standard (paid) tier
    (
        lambda name: name == "deepseek-v4-flash" or name == "deepseek-v4-flash*",
        [
            "opencode/deepseek-v4-flash",
            "opencode-go/deepseek-v4-flash",
            "openrouter/deepseek/deepseek-v4-flash",
        ],
    ),
    # Generic "deepseek" prefix → deepseek-v4-flash
    (
        lambda name: name.startswith("deepseek") and name != "deepseek-v4-flash-free",
        [
            "opencode/deepseek-v4-flash",
            "opencode-go/deepseek-v4-flash",
            "openrouter/deepseek/deepseek-v4-flash",
        ],
    ),
    # Generic "deepseek-v4" prefix
    (
        lambda name: name.startswith("deepseek-v4"),
        [
            "opencode/deepseek-v4-flash",
            "opencode-go/deepseek-v4-flash",
            "openrouter/deepseek/deepseek-v4-flash",
        ],
    ),
    # OpenRouter free / no-key tier
    (
        lambda name: name == "openrouter/free" or name == "openrouter-free" or name == "free-model",
        [
            "openrouter/openrouter/free",
            "opencode/deepseek-v4-flash-free",
            "opencode-go/deepseek-v4-flash",
        ],
    ),
    # OpenRouter deepseek
    (
        lambda name: name.startswith("openrouter/deepseek"),
        [
            "openrouter/deepseek/deepseek-v4-flash",
            "opencode/deepseek-v4-flash",
            "opencode-go/deepseek-v4-flash",
        ],
    ),
    # Generic "openrouter" prefix → openrouter/free
    (
        lambda name: name.startswith("openrouter") and not name.startswith("openrouter/deepseek"),
        [
            "openrouter/openrouter/free",
            "opencode/deepseek-v4-flash-free",
            "opencode-go/deepseek-v4-flash",
        ],
    ),
    # opencode-go specific
    (
        lambda name: name.startswith("opencode-go"),
        _resolve_opencode_go_candidates,
    ),
    # opencode specific
    (
        lambda name: name.startswith("opencode/") or name.startswith("opencode-"),
        _resolve_opencode_candidates,
    ),
    # Fallback: pass through already-qualified names (provider/model-id format)
    (
        lambda name: "/" in name,
        lambda name: [name],
    ),
    # Last resort: opencode as default provider
    (
        lambda name: True,
        lambda name: [f"opencode/{name}", f"opencode-go/{name}", f"openrouter/{name}"],
    ),
]


def resolve_name_to_ids(name: str) -> list[str]:
    """Resolve a single short model name to a list of provider-prefixed IDs.

    Iterates through the static mapping table and returns the candidate
    list for the first matching entry.  If no entry matches, returns an
    empty list.

    Args:
        name: A short model name (e.g. ``deepseek-v4-flash-free``).

    Returns:
        An ordered list of provider-prefixed model IDs, or ``[]`` if
        the name cannot be resolved.
    """
    for matcher, candidates in _MODEL_RESOLVERS:
        if matcher(name):
            # candidates may be a static list or a callable that returns a list
            if callable(candidates):
                resolved = candidates(name)
            else:
                resolved = list(candidates)
            logger.debug(
                "Resolved '%s' → %s",
                name,
                ", ".join(resolved),
            )
            return resolved

    # Should not be reached given the catch-all entry above
    logger.warning("No resolver matched name '%s'", name)
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_audit_model(
    name: str,
    fallbacks: list[str] | None = None,
) -> list[str]:
    """Resolve an audit model name (with fallbacks) to provider-prefixed IDs.

    The primary *name* is resolved first.  If it resolves to zero candidates,
    each *fallback* is attempted in order until at least one candidate is
    found.

    Args:
        name: The primary short model name.
        fallbacks: Optional ordered list of alternative short names.

    Returns:
        An ordered list of provider-prefixed model IDs.

    Logs:
        - INFO: successful resolution (with candidates).
        - WARNING: each failed resolution attempt.
    """
    results = resolve_name_to_ids(name)
    if results:
        logger.info(
            "Audit model '%s' resolved → %s",
            name,
            ", ".join(results),
        )
        return results

    logger.warning(
        "Audit model '%s' not resolved; trying fallbacks",
        name,
    )
    _record_unresolved(name)

    if fallbacks:
        for fb in fallbacks:
            fb_results = resolve_name_to_ids(fb)
            if fb_results:
                logger.info(
                    "Audit model fallback '%s' resolved → %s",
                    fb,
                    ", ".join(fb_results),
                )
                return fb_results
            logger.warning(
                "Audit model fallback '%s' also not resolved",
                fb,
            )
            _record_unresolved(fb)

    logger.warning(
        "Audit model '%s' and all %d fallback(s) failed to resolve",
        name,
        len(fallbacks or []),
    )
    return []


def validate_audit_models(
    config: dict,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate audit model configuration at startup.

    Attempts to resolve the primary ``audit_model`` and all entries in
    ``audit_model_fallbacks``.  Logs warnings for unresolvable names and,
    when *strict* is ``True``, returns an error if no valid models exist.

    Args:
        config: The full server config dict (must contain
                ``audit_model`` and optionally ``audit_model_fallbacks``).
        strict: If ``True``, returns ``{"ok": False, "reason": ...}`` when
                no valid models exist.  Default is ``False`` (lenient).

    Returns:
        A dict with keys:
          - ``ok`` (bool): Whether at least one valid model was resolved.
          - ``primary`` (str): The primary model name from config.
          - ``resolved_ids`` (list): List of resolved provider-prefixed IDs
            (primary + fallbacks).
          - ``warnings`` (list): Warnings for each unresolved name.
    """
    primary = config.get("audit_model", "")
    fallbacks = config.get("audit_model_fallbacks", []) or []
    warnings: list[str] = []
    resolved_ids: list[str] = []

    if not primary:
        warnings.append("No audit_model configured in config.yaml")
        return {
            "ok": False if strict else True,
            "primary": primary,
            "resolved_ids": [],
            "warnings": warnings,
        }

    resolved_ids = resolve_audit_model(primary, fallbacks)

    if not resolved_ids:
        reason = (
            f"audit_model '{primary}' and all {len(fallbacks)} fallback(s) "
            f"failed to resolve"
        )
        warnings.append(reason)
        if strict:
            logger.error("Audit validation FAILED: %s", reason)
            return {
                "ok": False,
                "primary": primary,
                "resolved_ids": [],
                "warnings": warnings,
            }
        logger.warning("Audit validation lenient: %s", reason)

    logger.info(
        "Audit model validation OK: primary='%s', %d resolved IDs",
        primary,
        len(resolved_ids),
    )
    return {
        "ok": True,
        "primary": primary,
        "resolved_ids": resolved_ids,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# CLI demo (standalone)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys

    # Configure basic logging for CLI usage
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python provider_resolver.py <name> [fallback1 fallback2 ...]")
        sys.exit(1)

    name = sys.argv[1]
    fallbacks = sys.argv[2:] if len(sys.argv) > 2 else []

    results = resolve_audit_model(name, fallbacks or None)
    if results:
        print(f"Resolved to: {', '.join(results)}")
    else:
        print(f"Failed to resolve '{name}'", file=sys.stderr)
        sys.exit(1)
