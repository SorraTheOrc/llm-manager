"""API-key resolution from ``~/.pi/agent/auth.json`` for ``start-proxy.sh``.

Extracted from the inline Python block previously embedded in
``start-proxy.sh``'s ``resolve_from_auth_json`` shell function so the mapping
logic is unit-testable. The blocker LP-0MVBJPQE50017BGL identified a mapping
gap: ``MUNITOAI_FREE_API_KEY`` resolves to the auth.json stem ``munitoai_free``
(underscore), but the documented/free-tier key is stored under the hyphenated
name ``munitoai-free``. This module therefore falls back to the hyphenated
stem so a fresh ``start-proxy.sh`` can start unattended.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

API_KEY_ERR_MSG = (
    "could not be resolved from ~/.pi/agent/auth.json — add it under the "
    "matching provider name or export it in the environment"
)


def load_auth(auth_file: str | Path) -> dict:
    """Load the auth.json mapping (``{key-name: {"type": ..., "key": ...}}``)."""
    with Path(auth_file).open() as f:
        return json.load(f)


def resolve_from_auth_json(auth: dict, env_var: str) -> str | None:
    """Resolve *env_var* to an API key from the *auth* mapping, or None.

    Mirrors the start-proxy.sh mapping rules:

    - ``OPENCODE_API_KEY`` prefers ``opencode-go`` over ``opencode`` (the
      historical shortcut for the default Go-backend deployment).
    - An exact lowercase match on the env-var name wins.
    - Otherwise the ``_API_KEY`` suffix is stripped and the stem is looked
      up; both the underscore stem (``munitoai_free``) and the hyphenated
      stem (``munitoai-free``) are tried, so documented auth.json names such
      as ``munitoai-free`` resolve correctly (LP-0MVBJPQE50017BGL).

    Returns the key string, or None when nothing matches.
    """
    key = env_var.lower()

    # Prefer opencode-go over opencode for OPENCODE_API_KEY
    if key == "opencode_api_key":
        for preferred in ("opencode-go", "opencode"):
            entry = auth.get(preferred)
            if entry and entry.get("type") == "api_key" and entry.get("key"):
                return entry["key"]

    # Exact lowercase match
    if _api_key_entry(auth, key):
        return auth[key]["key"]

    # Strip _API_KEY suffix, then try underscore + hyphenated stems.
    if key.endswith("_api_key"):
        stem = key[:-8]
        for candidate in (stem, stem.replace("_", "-")):
            if _api_key_entry(auth, candidate):
                return auth[candidate]["key"]

    return None


def _api_key_entry(auth: dict, name: str) -> bool:
    entry = auth.get(name)
    return bool(entry and entry.get("type") == "api_key" and entry.get("key"))


def _candidate_key_names(env_var: str) -> list[str]:
    """Return the auth.json key names a resolution of *env_var* would try.

    Used by the failure message so the mismatch is self-diagnosing: the
    resolver strips ``_API_KEY`` and tries both the underscore and the
    hyphenated stem (LP-0MVBJXZY5002UW10 AC).
    """
    key = env_var.lower()
    candidates = [key]
    if key == "opencode_api_key":
        candidates += ["opencode-go", "opencode"]
    if key.endswith("_api_key"):
        stem = key[:-8]
        candidates += [stem, stem.replace("_", "-")]
    # Deduplicate (e.g. OPENCODE_API_KEY's stem equals the preferred name).
    return list(dict.fromkeys(candidates))


def main(argv: list[str] | None = None) -> int:
    """CLI used by ``start-proxy.sh``: print the resolved key or exit 1."""
    parser = argparse.ArgumentParser(
        description="Resolve an api_key_env var from ~/.pi/agent/auth.json."
    )
    parser.add_argument("auth_file", help="Path to auth.json")
    parser.add_argument("env_var", help="api_key_env name, e.g. MUNITOAI_FREE_API_KEY")
    args = parser.parse_args(argv)

    def _fail() -> int:
        names = ", ".join(_candidate_key_names(args.env_var))
        print(
            f"ERROR: {args.env_var} {API_KEY_ERR_MSG} — tried auth.json "
            f"key(s): {names}",
            file=sys.stderr,
        )
        return 1

    try:
        auth = load_auth(args.auth_file)
    except (FileNotFoundError, json.JSONDecodeError):
        return _fail()

    resolved = resolve_from_auth_json(auth, args.env_var)
    if resolved:
        print(resolved)
        return 0
    return _fail()


if __name__ == "__main__":
    sys.exit(main())
