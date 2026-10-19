"""API-key resolution from ``~/.pi/agent/auth.json`` for ``start-proxy.sh``.

Extracted from the inline Python block previously embedded in
``start-proxy.sh``'s ``resolve_from_auth_json`` shell function so the mapping
logic is unit-testable. The blocker LP-0MVBJPQE50017BGL identified a mapping
gap: ``MUNITOAI_FREE_API_KEY`` resolves to the auth.json stem ``munitoai_free``
(underscore), but the documented/free-tier key is stored under the hyphenated
name ``munitoai-free``. This module therefore falls back to the hyphenated
stem so a fresh ``start-proxy.sh`` can start unattended.

The stale-key contract (parent LP-0MVD2RF7H008B7OT) adds a diagnosis-only
fingerprint API (:func:`key_fingerprint`, :func:`compare_env_key`, the
:class:`EnvKeyCheck` record) and a ``--check`` CLI mode: :func:`compare_env_key`
compares an ``api_key_env`` value against the ``auth.json`` resolution, exits 0
for match/resolved/env_only, 3 for a stale mismatch and 4 when no key is
usable, and never emits a plaintext key through the env-set path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
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


# Fingerprint width for diagnosis-only key logging (never authorisation).
FINGERPRINT_LEN = 8

# ``EnvKeyCheck.status`` values. ``match`` and ``mismatch`` compare an
# environment value against the auth.json resolution; ``resolved`` is the
# env-unset path that start-proxy.sh exports; ``env_only`` means auth.json is
# absent so staleness cannot be verified; ``auth_missing`` means neither source
# supplied a key; ``unresolved`` means auth.json is present but has no entry.
STATUS_MATCH = "match"
STATUS_MISMATCH = "mismatch"
STATUS_RESOLVED = "resolved"
STATUS_ENV_ONLY = "env_only"
STATUS_AUTH_MISSING = "auth_missing"
STATUS_UNRESOLVED = "unresolved"

# ``--check`` exit codes consumed by start-proxy.sh's stale-key branch.
CHECK_EXIT_OK = 0
CHECK_EXIT_MISMATCH = 3
CHECK_EXIT_UNRESOLVED = 4


@dataclass(frozen=True)
class EnvKeyCheck:
    """Outcome of comparing an ``api_key_env`` value with ``auth.json``.

    ``resolved_key`` is populated only when the environment variable is unset
    (the export path); whenever the env var supplies the key only fingerprints
    are carried, so a plaintext secret is never emitted for diagnosis.
    """

    env_var: str
    status: str
    env_fingerprint: str = ""
    auth_fingerprint: str = ""
    resolved_key: str = ""


def key_fingerprint(key: str | None) -> str:
    """Return at most ``FINGERPRINT_LEN`` leading characters of *key*.

    ``None`` and empty strings yield ``""``. Keys longer than
    ``FINGERPRINT_LEN`` are truncated (never returned whole); shorter keys are
    returned as-is. The result is for log/diagnosis only, never authorisation.
    """
    if not key:
        return ""
    return key[:FINGERPRINT_LEN]


def compare_env_key(
    env_var: str, env_value: str | None, auth: dict | None
) -> EnvKeyCheck:
    """Compare an environment key with the ``auth.json`` resolution.

    Resolution semantics are exactly those of :func:`resolve_from_auth_json`
    (hyphenated + underscore stems, ``opencode-go`` preference), so the
    comparison can never disagree with what the resolver would export.

    * ``auth is None`` (no auth.json) + env set -> ``env_only``: the env key is
      kept and staleness cannot be verified.
    * ``auth is None`` + env unset -> ``auth_missing``.
    * no resolvable entry -> ``unresolved`` (never raises).
    * env unset -> ``resolved`` with ``resolved_key`` for the caller to export.
    * env equal to the resolution -> ``match``; differing -> ``mismatch``.
    """
    env_fingerprint = key_fingerprint(env_value)

    if auth is None:
        if env_value:
            return EnvKeyCheck(
                env_var=env_var,
                status=STATUS_ENV_ONLY,
                env_fingerprint=env_fingerprint,
            )
        return EnvKeyCheck(env_var=env_var, status=STATUS_AUTH_MISSING)

    resolved = resolve_from_auth_json(auth, env_var)
    if not resolved:
        return EnvKeyCheck(
            env_var=env_var,
            status=STATUS_UNRESOLVED,
            env_fingerprint=env_fingerprint,
        )

    auth_fingerprint = key_fingerprint(resolved)
    if not env_value:
        return EnvKeyCheck(
            env_var=env_var,
            status=STATUS_RESOLVED,
            auth_fingerprint=auth_fingerprint,
            resolved_key=resolved,
        )
    if env_value == resolved:
        return EnvKeyCheck(
            env_var=env_var,
            status=STATUS_MATCH,
            env_fingerprint=env_fingerprint,
            auth_fingerprint=auth_fingerprint,
        )
    return EnvKeyCheck(
        env_var=env_var,
        status=STATUS_MISMATCH,
        env_fingerprint=env_fingerprint,
        auth_fingerprint=auth_fingerprint,
    )


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


def _check(env_var: str, auth_file: str) -> int:
    """Run the ``--check`` comparison and emit exactly one JSON line.

    Exit codes: 0 for ``match`` / ``resolved`` / ``env_only``, 3 for a stale
    ``mismatch``, 4 for ``unresolved`` / ``auth_missing`` (nothing usable).
    """
    env_value = os.environ.get(env_var) or None
    try:
        auth = load_auth(auth_file)
    except (OSError, json.JSONDecodeError):
        auth = None

    check = compare_env_key(env_var, env_value, auth)
    payload = {
        "env_var": check.env_var,
        "status": check.status,
        "env_fingerprint": check.env_fingerprint,
        "auth_fingerprint": check.auth_fingerprint,
        "resolved_key": check.resolved_key,
    }
    print(json.dumps(payload))

    if check.status in (STATUS_MATCH, STATUS_RESOLVED, STATUS_ENV_ONLY):
        return CHECK_EXIT_OK
    if check.status == STATUS_MISMATCH:
        return CHECK_EXIT_MISMATCH
    return CHECK_EXIT_UNRESOLVED


def main(argv: list[str] | None = None) -> int:
    """CLI used by ``start-proxy.sh``.

    Default form (``<auth_file> <env_var>``) prints the resolved plaintext key
    and exits 1 when unresolved — unchanged for backwards compatibility.
    ``--check <auth_file> <env_var>`` prints one JSON line of fingerprints
    (plus ``resolved_key`` on the env-unset export path) and exits 0/3/4 so
    the shell can decide whether to fail closed on a stale env key.
    """
    parser = argparse.ArgumentParser(
        description="Resolve an api_key_env var from ~/.pi/agent/auth.json."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Compare the environment value with auth.json and print JSON "
            "(0=match/resolved/env_only, 3=stale mismatch, 4=unresolved). "
            "Emits fingerprints only when the env var is set."
        ),
    )
    parser.add_argument("auth_file", help="Path to auth.json")
    parser.add_argument("env_var", help="api_key_env name, e.g. MUNITOAI_FREE_API_KEY")
    args = parser.parse_args(argv)

    if args.check:
        return _check(args.env_var, args.auth_file)

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
