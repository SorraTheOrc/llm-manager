"""
Hermetic tests for restart-lifecycle preservation (LP-0MUCEFCL6001ZXYN).

Context
-------
Proxy-only restarts previously killed the co-located llama-server, discarding
the warm KV / prompt cache and triggering a cold-prefill thundering-herd.
The mode-switch drain also waited only on ``local_active_queries``, which
could read 0 while real streams were still running, so the restart dropped
them with a client-visible ``finish_reason: error``.

These tests verify hermetically (no live llama-server):

1. The drain counts real in-flight streams via active dispatch records even
   when the counter is 0, and waits until they clear (bounded).
2. The llama-server signature is recorded/read and drives adoption.
3. A new proxy process adopts an already-running server whose signature
   matches (preserving the warm cache) and refuses mismatches.
4. The adopted process sentinel behaves like a running process and a real
   stop still terminates the server by port.
5. ``start-proxy.sh`` is syntactically valid and understands
   ``--keep-llama-server``.
"""

import asyncio
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import proxy.lifecycle as lifecycle
import pytest

from proxy import cold_start


def _make_srv(**overrides) -> SimpleNamespace:
    base = dict(
        config={"server": {"llama_server_port": 8080, "llama_router_mode": True}},
        llama_process=None,
        current_model=None,
        backend_ready=False,
        llama_log_file=None,
        logger=MagicMock(),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ═══════════════════════════════════════════════════════════════════════════════
# Drain: real in-flight streams
# ═══════════════════════════════════════════════════════════════════════════════


def test_mode_switch_drain_removed():
    """The bounded mode-switch drain was removed with the fast/cheap mode
    axis (LP-0MVDT94F0003VHUA AC4): the proxy never restarts for a routing
    policy change, so no drain exists. The drain helpers lived in
    ``proxy.mode``, which is gone.
    """
    import importlib

    with pytest.raises(ImportError):
        importlib.import_module("proxy.mode")



@pytest.fixture
def sig_path(tmp_path, monkeypatch):
    path = tmp_path / ".llama_server_signature.json"
    monkeypatch.setattr(lifecycle, "_llama_server_signature_path", lambda: path)
    return path


def test_write_and_read_signature(sig_path, monkeypatch):
    monkeypatch.setattr(lifecycle, "_read_mode_label", lambda: "unified")
    monkeypatch.setattr(lifecycle, "_config_file_sha256", lambda: "abc123")
    lifecycle._write_llama_server_signature("Qwen3", 1, router_mode=True)

    sig = lifecycle._read_llama_server_signature()
    assert sig is not None
    assert sig["mode"] == "unified"
    assert sig["model"] == "router"
    assert sig["parallel"] == 1
    assert sig["router_mode"] is True
    assert sig["config_sha256"] == "abc123"


def test_read_signature_absent_returns_none(sig_path):
    assert lifecycle._read_llama_server_signature() is None


def test_read_signature_corrupt_returns_none(sig_path):
    sig_path.write_text("not json", encoding="utf-8")
    assert lifecycle._read_llama_server_signature() is None


def test_signature_file_is_gitignored():
    """The runtime signature file must be untracked (LP-0MUD2G0V8004EXXZ).

    It is written by ``_write_llama_server_signature`` on every llama-server
    start and must never be committed (it dirtied the tree on every mode
    switch when accidentally tracked by LP-0MUCEFCL6001ZXYN).
    """
    proxy_dir = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["git", "check-ignore", "proxy/.llama_server_signature.json"],
        cwd=str(proxy_dir.parent),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "proxy/.llama_server_signature.json must be gitignored"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Adoption
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_adopt_router_mode_when_signature_matches(sig_path, monkeypatch):
    lifecycle._write_llama_server_signature("router", 1, router_mode=True)
    monkeypatch.setattr(
        lifecycle, "_probe_backend_reachable", AsyncMock(return_value=True)
    )
    srv = _make_srv()
    cold_start.reset()
    cold_start.note_model_loaded()
    assert cold_start.is_cold({"server": {}})

    adopted = await lifecycle._try_adopt_running_llama_server(
        srv, "Qwen3", router_mode=True
    )
    assert adopted is True
    assert srv.backend_ready is True
    assert getattr(srv.llama_process, "adopted", False) is True
    # Preserved cache is warm: the cold window is lifted.
    assert not cold_start.is_cold({"server": {}})


@pytest.mark.asyncio
async def test_adopt_single_model_requires_model_match(sig_path, monkeypatch):
    lifecycle._write_llama_server_signature("Qwen3", 3, router_mode=False)
    monkeypatch.setattr(
        lifecycle, "_probe_backend_reachable", AsyncMock(return_value=True)
    )
    srv = _make_srv(config={"server": {"llama_server_port": 8080}})

    assert (
        await lifecycle._try_adopt_running_llama_server(srv, "Qwen3", router_mode=False)
        is True
    )
    assert srv.current_model == "Qwen3"

    srv2 = _make_srv(config={"server": {"llama_server_port": 8080}})
    assert (
        await lifecycle._try_adopt_running_llama_server(srv2, "Other", router_mode=False)
        is False
    )


@pytest.mark.asyncio
async def test_adopt_rejects_when_unreachable(sig_path, monkeypatch):
    lifecycle._write_llama_server_signature("router", 1, router_mode=True)
    monkeypatch.setattr(
        lifecycle, "_probe_backend_reachable", AsyncMock(return_value=False)
    )
    srv = _make_srv()
    assert (
        await lifecycle._try_adopt_running_llama_server(srv, "Qwen3", router_mode=True)
        is False
    )


@pytest.mark.asyncio
async def test_adopt_rejects_router_mode_mismatch(sig_path, monkeypatch):
    lifecycle._write_llama_server_signature("Qwen3", 1, router_mode=False)
    monkeypatch.setattr(
        lifecycle, "_probe_backend_reachable", AsyncMock(return_value=True)
    )
    srv = _make_srv()
    assert (
        await lifecycle._try_adopt_running_llama_server(srv, "Qwen3", router_mode=True)
        is False
    )


@pytest.mark.asyncio
async def test_adopt_rejects_config_hash_mismatch(sig_path, monkeypatch):
    """LP-0MUPIIDPW009RMLB: adoption is rejected when the recorded
    merged-config hash differs from the current effective config (same
    stale-config bug class as the restart gate)."""
    lifecycle._write_llama_server_signature("Qwen3", 1, router_mode=False)
    sig = json.loads(sig_path.read_text(encoding="utf-8"))
    sig["config_sha256"] = "stale-hash"
    sig_path.write_text(json.dumps(sig), encoding="utf-8")
    monkeypatch.setattr(lifecycle, "_config_file_sha256", lambda: "current-hash")
    monkeypatch.setattr(
        lifecycle, "_probe_backend_reachable", AsyncMock(return_value=True)
    )
    srv = _make_srv(config={"server": {"llama_server_port": 8080}})

    assert (
        await lifecycle._try_adopt_running_llama_server(srv, "Qwen3", router_mode=False)
        is False
    )
    # Not adopted: no sentinel, backend not marked ready.
    assert srv.llama_process is None
    assert srv.backend_ready is False


@pytest.mark.asyncio
async def test_adopt_succeeds_when_config_hash_matches(sig_path, monkeypatch):
    """LP-0MUPIIDPW009RMLB: adoption still succeeds when the recorded
    merged-config hash equals the current effective config."""
    lifecycle._write_llama_server_signature("Qwen3", 1, router_mode=False)
    sig = json.loads(sig_path.read_text(encoding="utf-8"))
    sig["config_sha256"] = "matching-hash"
    sig_path.write_text(json.dumps(sig), encoding="utf-8")
    monkeypatch.setattr(lifecycle, "_config_file_sha256", lambda: "matching-hash")
    monkeypatch.setattr(
        lifecycle, "_probe_backend_reachable", AsyncMock(return_value=True)
    )
    srv = _make_srv(config={"server": {"llama_server_port": 8080}})

    assert (
        await lifecycle._try_adopt_running_llama_server(srv, "Qwen3", router_mode=False)
        is True
    )
    assert getattr(srv.llama_process, "adopted", False) is True
    assert srv.backend_ready is True


@pytest.mark.asyncio
async def test_adopt_tolerates_missing_config_hash(sig_path, monkeypatch):
    """LP-0MUPIIDPW009RMLB: pre-upgrade signatures without a
    ``config_sha256`` key are still adopted (backward-compatible, fail-safe
    in the preserve direction)."""
    sig_path.write_text(
        json.dumps(
            {
                "mode": "fast",
                "model": "Qwen3",
                "parallel": 1,
                "router_mode": False,
            }
        ),
        encoding="utf-8",
    )
    assert "config_sha256" not in json.loads(sig_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(lifecycle, "_config_file_sha256", lambda: "whatever")
    monkeypatch.setattr(
        lifecycle, "_probe_backend_reachable", AsyncMock(return_value=True)
    )
    srv = _make_srv(config={"server": {"llama_server_port": 8080}})

    assert (
        await lifecycle._try_adopt_running_llama_server(srv, "Qwen3", router_mode=False)
        is True
    )
    assert getattr(srv.llama_process, "adopted", False) is True


def test_adopted_process_behaves_like_running_process():
    proc = lifecycle._AdoptedLlamaProcess(port=8080)
    assert proc.poll() is None
    assert proc.adopted is True
    assert proc.port == 8080


def test_stop_adopted_server_kills_by_port(sig_path, monkeypatch):
    """A genuine stop of an adopted server terminates it by port."""
    killed = []
    monkeypatch.setattr(
        lifecycle, "_kill_process_on_port", lambda port, logger=None: killed.append(port)
    )
    monkeypatch.setattr(lifecycle, "_close_and_recreate_http_client", lambda srv=None: None)
    srv = _make_srv()
    srv.llama_process = lifecycle._AdoptedLlamaProcess(port=8080)
    srv.current_model = "Qwen3"
    srv.backend_ready = True
    monkeypatch.setattr(lifecycle, "_srv", lambda: srv)

    lifecycle.stop_llama_server()
    assert killed == [8080]
    assert srv.llama_process is None
    assert srv.backend_ready is False


# ═══════════════════════════════════════════════════════════════════════════════
# Merged-config restart signature (LP-0MUJRVYQH002IEHW)
# ═══════════════════════════════════════════════════════════════════════════════


def test_config_file_sha256_matches_shared_merged_helper(monkeypatch, tmp_path):
    """Write-side parity (parent AC1): the value persisted by
    ``_config_file_sha256()`` (base config.yaml + ``$LLAMA_PROXY_CONFIG``
    overlay) is byte-identical to the shared ``merged_config_sha256`` helper
    on the same inputs."""
    from proxy.config_merge import merged_config_sha256
    from proxy.priority import proxy_dir

    overlay = tmp_path / "custom-overlay.yaml"
    overlay.write_text("server:\n  session_slot_pool_size: 2\n", encoding="utf-8")
    monkeypatch.setenv("LLAMA_PROXY_CONFIG", str(overlay))

    base_path = proxy_dir() / "config.yaml"
    expected = merged_config_sha256(base_path, overlay)
    assert lifecycle._config_file_sha256() == expected


def test_config_file_sha256_hashes_merged_not_raw_overlay(monkeypatch, tmp_path):
    """Sensitivity guard: a base-only change (which cannot be seen in the
    raw overlay bytes) must change the write-side hash — proxied through the
    shared helper which reads the merged surface."""
    from proxy.config_merge import merged_config_sha256
    from proxy.priority import proxy_dir

    overlay = tmp_path / "custom-overlay.yaml"
    overlay.write_text("server:\n  session_slot_pool_size: 2\n", encoding="utf-8")
    monkeypatch.setenv("LLAMA_PROXY_CONFIG", str(overlay))

    base_path = proxy_dir() / "config.yaml"
    base_hash = merged_config_sha256(base_path, overlay)

    # The helper is shared, so the write side is covered by construction;
    # assert the base-sensitivity property end-to-end via the helper.
    import proxy.config_merge as cm

    base_text = base_path.read_text(encoding="utf-8")
    assert "local_model_ctx_size" in base_text  # base carries local-backend keys

    alt_base = tmp_path / "alt-base.yaml"
    alt_base.write_text(
        base_text.replace("local_model_ctx_size: 262144", "local_model_ctx_size: 131072"),
        encoding="utf-8",
    )
    alt_hash = merged_config_sha256(alt_base, overlay)
    assert alt_hash != base_hash


def test_config_file_sha256_falls_back_to_base_without_env(monkeypatch):
    """With ``$LLAMA_PROXY_CONFIG`` unset the write side hashes the base
    merged with itself (base still exists), not the empty string."""
    from proxy.config_merge import merged_config_sha256
    from proxy.priority import proxy_dir

    monkeypatch.delenv("LLAMA_PROXY_CONFIG", raising=False)
    base_path = proxy_dir() / "config.yaml"
    assert lifecycle._config_file_sha256() == merged_config_sha256(
        base_path, base_path
    )


def test_config_file_sha256_tolerates_missing_overlay(monkeypatch, tmp_path):
    """A ``$LLAMA_PROXY_CONFIG`` pointing at a missing file falls back to
    the base rather than returning an error/empty string."""
    from proxy.config_merge import merged_config_sha256
    from proxy.priority import proxy_dir

    missing = tmp_path / "missing-overlay.yaml"
    monkeypatch.setenv("LLAMA_PROXY_CONFIG", str(missing))
    base_path = proxy_dir() / "config.yaml"

    assert lifecycle._config_file_sha256() == merged_config_sha256(
        base_path, base_path
    )


# ═══════════════════════════════════════════════════════════════════════════════
# start-proxy.sh
# ═══════════════════════════════════════════════════════════════════════════════


def _script_path() -> Path:
    return Path(__file__).resolve().parent.parent / "scripts" / "start-proxy.sh"


def test_start_proxy_script_syntax():
    """The restart-lifecycle edits keep the script syntactically valid."""
    result = subprocess.run(
        ["bash", "-n", str(_script_path())],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"


def test_start_proxy_help_documents_keep_llama_server():
    """The new flag is documented in the script's usage header."""
    header = _script_path().read_text(encoding="utf-8").splitlines()[:12]
    joined = "\n".join(header)
    assert "--keep-llama-server" in joined


def test_start_proxy_documents_reason_flag():
    """The --reason restart flag is documented in the usage header."""
    header = _script_path().read_text(encoding="utf-8").splitlines()[:8]
    joined = "\n".join(header)
    assert "--reason" in joined


def test_start_proxy_reason_resolution_precedence():
    """The restart-reason resolution is FLAG > ENV > default (LP-0MVAPNC2900145RH).

    Runs the script's own resolution expression in a hermetic bash subshell
    (no proxy processes are touched).
    """
    expr = "${x:-${LLAMA_PROXY_RESTART_REASON:-manual (no reason provided)}}"
    code = (
        f'x="flag reason"; echo "flag: {expr}"; '
        f'unset x; unset LLAMA_PROXY_RESTART_REASON; echo "env: {expr}"; '
        f'unset x; unset LLAMA_PROXY_RESTART_REASON; echo "default: {expr}"; '
        f"unset x; LLAMA_PROXY_RESTART_REASON='env reason'; echo \"env-wins: {expr}\";"
    )
    result = subprocess.run(["bash", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    out = dict(
        (k.strip(), v.strip())
        for line in result.stdout.strip().splitlines()
        for k, v in [line.split(":", 1)]
    )
    assert out["flag"] == "flag reason"
    assert out["env"] == "manual (no reason provided)"
    assert out["default"] == "manual (no reason provided)"
    assert out["env-wins"] == "env reason"


# ═══════════════════════════════════════════════════════════════════════════════
# api_key_resolver (LP-0MVBJPQE50017BGL)
# ═══════════════════════════════════════════════════════════════════════════════


def _auth_fixture(**keys) -> dict:
    """Build an auth.json-style mapping from name -> key-value pairs."""
    return {k: {"type": "api_key", "key": v} for k, v in keys.items()}


def test_resolver_maps_hyphenated_stem():
    """MUNITOAI_FREE_API_KEY must resolve the hyphenated auth.json stem
    ``munitoai-free`` (blocker AC1/AC3: documented name has a hyphen while
    the api_key_env stem is underscore-separated)."""
    from proxy.api_key_resolver import resolve_from_auth_json

    auth = _auth_fixture(**{"munitoai-free": "FREE_KEY", "munitoai": "PAID_KEY"})
    assert resolve_from_auth_json(auth, "MUNITOAI_FREE_API_KEY") == "FREE_KEY"


def test_resolver_underscore_stem_still_matches():
    """A plain underscore stem (e.g. ``opencode_2``) keeps resolving."""
    from proxy.api_key_resolver import resolve_from_auth_json

    auth = _auth_fixture(opencode_2="O2", opencode="O")
    assert resolve_from_auth_json(auth, "OPENCODE_2_API_KEY") == "O2"


def test_resolver_prefers_opencode_go():
    """OPENCODE_API_KEY preference order is opencode-go then opencode."""
    from proxy.api_key_resolver import resolve_from_auth_json

    both = _auth_fixture(**{"opencode-go": "GO", "opencode": "BASE"})
    assert resolve_from_auth_json(both, "OPENCODE_API_KEY") == "GO"
    only_base = _auth_fixture(opencode="BASE")
    assert resolve_from_auth_json(only_base, "OPENCODE_API_KEY") == "BASE"


def test_resolver_missing_key_returns_none():
    from proxy.api_key_resolver import resolve_from_auth_json

    auth = _auth_fixture(munitoai="PAID")
    assert resolve_from_auth_json(auth, "SOME_UNKNOWN_KEY") is None


def test_resolver_cli_roundtrip(tmp_path):
    """CLI prints the key for a resolvable env var and exits 0/1."""
    import subprocess
    import sys

    from proxy.api_key_resolver import main

    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps(_auth_fixture(**{"munitoai-free": "FREE_KEY"})))

    import io
    from contextlib import redirect_stderr, redirect_stdout

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main([str(auth_file), "MUNITOAI_FREE_API_KEY"])
    assert rc == 0
    assert out.getvalue().strip() == "FREE_KEY"

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main([str(auth_file), "MISSING_ENV_KEY"])
    assert rc == 1
    assert "could not be resolved" in err.getvalue()


def test_resolver_failure_message_names_candidate_keys():
    """An unresolved key must name the auth.json key(s) that would have
    satisfied it (LP-0MVBJXZY5002UW10 AC: self-diagnosing mismatch)."""
    from proxy.api_key_resolver import _candidate_key_names

    assert _candidate_key_names("MUNITOAI_FREE_API_KEY") == [
        "munitoai_free_api_key",
        "munitoai_free",
        "munitoai-free",
    ]
    assert _candidate_key_names("OPENCODE_API_KEY") == [
        "opencode_api_key",
        "opencode-go",
        "opencode",
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# api_key_resolver stale-key comparison / fingerprint (LP-0MVF7DO29009T8MH)
#
# AC-pinning tests for the fingerprint + ``compare_env_key`` helper and the
# ``--check`` CLI contract that start-proxy.sh will call per ``api_key_env``
# (parent LP-0MVD2RF7H008B7OT AC1–AC4). Authored test-first (RED) before the
# helper child (LP-0MVF7DRBM004N90W) lands, so the contract is executable,
# not prose. Fixture keys are deliberately >= 12 chars so fingerprint
# redaction is genuinely exercised (an <= 8-char key would be returned whole).
#
# TDD red phase completed: the helper and ``--check`` mode landed in
# LP-0MVF7DRBM004N90W, which removed the ``xfail(strict=True)`` markers that
# kept ``dev`` green while the contract was unimplemented. The assertions below
# are now enforced directly (same convention as
# ``test_compaction_headers_contract.py``, LP-0MU44KR5F0015D7A).
# ═══════════════════════════════════════════════════════════════════════════════

# Test names carry the parent AC they pin (audit AC6 traceability):
#   ac1 — env key compared against auth.json (match / mismatch / cannot verify)
#   ac2 — short fingerprint only, never the full key
#   ac3 — fail-closed decision surface (--check exit 0/3/4)
#   ac4_caseN — the four required AC4 cases (match, mismatch, resolved, auth.json
#               missing) plus the derived env_only/auth_missing/unresolved cases

# Env var name uses a synthetic provider stem so the tests never read or write
# a real provider key in the ambient environment (fixtures only, via tmp_path).
_ENV_VAR = "TEST_PROVIDER_API_KEY"
_ENV_KEY = "ENV_KEY_MATCHING_AAAA"
_AUTH_KEY = "AUTH_KEY_ROTATED_AAAA"


def _auth_json(tmp_path, **keys) -> Path:
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(_auth_fixture(**keys)))
    return path


# ── key_fingerprint (AC2: short fingerprint, never the whole key) ─────────────


def test_key_fingerprint_is_first_8_chars_ac2():
    """Pin the fingerprint width and that it never returns the whole key."""
    from proxy.api_key_resolver import FINGERPRINT_LEN, key_fingerprint

    assert FINGERPRINT_LEN == 8
    fp = key_fingerprint(_AUTH_KEY)
    assert fp == _AUTH_KEY[:8]
    assert len(fp) <= FINGERPRINT_LEN
    # longer-than-8 keys must never be returned whole (diagnosis, not auth)
    assert fp != _AUTH_KEY


def test_key_fingerprint_empty_and_none_yield_empty_string_ac2():
    from proxy.api_key_resolver import key_fingerprint

    assert key_fingerprint(None) == ""
    assert key_fingerprint("") == ""


# ── compare_env_key: the four AC4 cases ──────────────────────────────────────


def test_compare_env_key_match_when_env_equals_auth_json_ac1_ac4_case1():
    """AC4 case 1: env set and matching auth.json -> ``match``."""
    from proxy.api_key_resolver import EnvKeyCheck, compare_env_key, key_fingerprint

    auth = _auth_fixture(test_provider=_AUTH_KEY)
    check = compare_env_key(_ENV_VAR, _AUTH_KEY, auth)

    assert isinstance(check, EnvKeyCheck)
    assert check.env_var == _ENV_VAR
    assert check.status == "match"
    assert check.env_fingerprint == key_fingerprint(_AUTH_KEY)
    assert check.env_fingerprint == check.auth_fingerprint
    # env already supplies the key, so nothing is exported
    assert not check.resolved_key


def test_compare_env_key_mismatch_when_env_differs_from_auth_json_ac1_ac3_ac4_case2():
    """AC4 case 2: env set and differing -> ``mismatch`` (stale key)."""
    from proxy.api_key_resolver import compare_env_key, key_fingerprint

    auth = _auth_fixture(test_provider=_AUTH_KEY)
    check = compare_env_key(_ENV_VAR, _ENV_KEY, auth)

    assert check.status == "mismatch"
    assert check.env_fingerprint == key_fingerprint(_ENV_KEY)
    assert check.auth_fingerprint == key_fingerprint(_AUTH_KEY)
    assert check.env_fingerprint != check.auth_fingerprint
    # fingerprints must be redacted, never the full key
    assert check.env_fingerprint != _ENV_KEY
    assert check.auth_fingerprint != _AUTH_KEY


def test_compare_env_key_resolved_from_auth_json_when_env_unset_ac4_case3():
    """AC4 case 3: env unset -> ``resolved``, returning the auth.json key
    so start-proxy.sh can export it."""
    from proxy.api_key_resolver import compare_env_key, key_fingerprint

    auth = _auth_fixture(test_provider=_AUTH_KEY)
    for unset in (None, ""):
        check = compare_env_key(_ENV_VAR, unset, auth)
        assert check.status == "resolved"
        assert check.resolved_key == _AUTH_KEY
        assert check.env_fingerprint == ""
        assert check.auth_fingerprint == key_fingerprint(_AUTH_KEY)


def test_compare_env_key_auth_json_missing_with_env_is_env_only_ac1_ac4_case4():
    """AC4 case 4a: auth.json absent (``auth=None``) + env set -> ``env_only``
    (staleness cannot be verified; keep the env key, warn, do not fail)."""
    from proxy.api_key_resolver import compare_env_key

    check = compare_env_key(_ENV_VAR, _ENV_KEY, None)
    assert check.status == "env_only"
    assert check.env_fingerprint == _ENV_KEY[:8]
    assert check.auth_fingerprint == ""
    assert not check.resolved_key


def test_compare_env_key_auth_json_missing_with_env_unset_is_auth_missing_ac3_ac4_case4():
    """AC4 case 4b: auth.json absent + env unset -> ``auth_missing``, and
    never raises."""
    from proxy.api_key_resolver import compare_env_key

    check = compare_env_key(_ENV_VAR, None, None)
    assert check.status == "auth_missing"
    assert not check.resolved_key


def test_compare_env_key_unresolvable_entry_is_unresolved_ac4_case4():
    """auth.json present but no resolvable entry cannot be compared ->
    ``unresolved`` (never raises)."""
    from proxy.api_key_resolver import compare_env_key

    auth = _auth_fixture(munitoai="PAID_KEY_AAAA")
    assert compare_env_key(_ENV_VAR, _ENV_KEY, auth).status == "unresolved"
    assert compare_env_key(_ENV_VAR, None, auth).status == "unresolved"


def test_compare_env_key_prefers_opencode_go_ac1_ac4():
    """OPENCODE_API_KEY resolution follows the opencode-go preference, so the
    comparison is against the Go key, not the base ``opencode`` key."""
    from proxy.api_key_resolver import compare_env_key

    go_key = "OPENCODE_GO_KEY_AAAA"
    base_key = "OPENCODE_BASE_KEY_AAAA"
    auth = _auth_fixture(**{"opencode-go": go_key, "opencode": base_key})

    resolved = compare_env_key("OPENCODE_API_KEY", None, auth)
    assert resolved.status == "resolved"
    assert resolved.resolved_key == go_key

    assert compare_env_key("OPENCODE_API_KEY", go_key, auth).status == "match"
    assert (
        compare_env_key("OPENCODE_API_KEY", base_key, auth).status == "mismatch"
    )


# ── --check CLI contract (exit codes 0/3/4, fingerprints-only stdout) ────────


def test_check_cli_exit_0_with_fingerprints_only_on_match_ac2_ac3_ac4_case1(
    tmp_path, monkeypatch, capsys
):
    from proxy.api_key_resolver import key_fingerprint, main

    auth_file = _auth_json(tmp_path, test_provider=_AUTH_KEY)
    monkeypatch.setenv(_ENV_VAR, _AUTH_KEY)

    rc = main(["--check", str(auth_file), _ENV_VAR])
    out = capsys.readouterr().out

    assert rc == 0
    payload = json.loads(out.strip())  # exactly one line of JSON
    assert payload["status"] == "match"
    assert payload["env_var"] == _ENV_VAR
    assert payload["env_fingerprint"] == key_fingerprint(_AUTH_KEY)
    # no plaintext key may appear when the env var is set
    assert _AUTH_KEY not in out
    assert _ENV_KEY not in out


def test_check_cli_exit_0_and_exports_resolved_key_when_env_unset_ac3_ac4_case3(
    tmp_path, monkeypatch, capsys
):
    from proxy.api_key_resolver import main

    auth_file = _auth_json(tmp_path, test_provider=_AUTH_KEY)
    monkeypatch.delenv(_ENV_VAR, raising=False)

    rc = main(["--check", str(auth_file), _ENV_VAR])
    out = capsys.readouterr().out

    assert rc == 0
    payload = json.loads(out.strip())
    assert payload["status"] == "resolved"
    # the env-unset path is the export path, so the full key is expected
    assert payload["resolved_key"] == _AUTH_KEY


def test_check_cli_exit_3_on_mismatch_without_plaintext_ac1_ac3_ac4_case2(
    tmp_path, monkeypatch, capsys
):
    from proxy.api_key_resolver import key_fingerprint, main

    auth_file = _auth_json(tmp_path, test_provider=_AUTH_KEY)
    monkeypatch.setenv(_ENV_VAR, _ENV_KEY)

    rc = main(["--check", str(auth_file), _ENV_VAR])
    out = capsys.readouterr().out

    assert rc == 3
    payload = json.loads(out.strip())
    assert payload["status"] == "mismatch"
    assert payload["env_fingerprint"] == key_fingerprint(_ENV_KEY)
    assert payload["auth_fingerprint"] == key_fingerprint(_AUTH_KEY)
    assert _ENV_KEY not in out
    assert _AUTH_KEY not in out


def test_check_cli_exit_4_when_key_unresolvable_ac3_ac4_case4(tmp_path, monkeypatch, capsys):
    from proxy.api_key_resolver import main

    auth_file = _auth_json(tmp_path, munitoai="PAID_KEY_AAAA")
    monkeypatch.delenv(_ENV_VAR, raising=False)

    rc = main(["--check", str(auth_file), _ENV_VAR])
    out = capsys.readouterr().out

    assert rc == 4
    assert json.loads(out.strip())["status"] == "unresolved"


def test_check_cli_exit_0_env_only_when_auth_json_absent_ac1_ac4_case4(
    tmp_path, monkeypatch, capsys
):
    from proxy.api_key_resolver import main

    missing = tmp_path / "auth-does-not-exist.json"
    monkeypatch.setenv(_ENV_VAR, _ENV_KEY)

    rc = main(["--check", str(missing), _ENV_VAR])
    out = capsys.readouterr().out

    assert rc == 0  # env_only keeps the key, warns, does not fail
    assert json.loads(out.strip())["status"] == "env_only"
    assert _ENV_KEY not in out


def test_check_cli_non_zero_when_auth_json_absent_and_env_unset_ac3_ac4_case4(
    tmp_path, monkeypatch, capsys
):
    from proxy.api_key_resolver import main

    missing = tmp_path / "auth-does-not-exist.json"
    monkeypatch.delenv(_ENV_VAR, raising=False)

    rc = main(["--check", str(missing), _ENV_VAR])
    out = capsys.readouterr().out

    # no key at all: fail closed rather than silently exporting nothing
    assert rc != 0
    assert json.loads(out.strip())["status"] == "auth_missing"
