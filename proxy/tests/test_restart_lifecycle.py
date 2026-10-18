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


def test_count_in_flight_counts_active_records_when_counter_zero(monkeypatch):
    """The drain sees real streams even when ``local_active_queries`` is 0."""
    import proxy.server as srv

    monkeypatch.setattr(srv, "local_active_queries", 0)
    monkeypatch.setattr(
        srv,
        "local_dispatch_records",
        {
            "sess-a": {"active": True},
            "sess-b": {"active": False},
            "sess-c": {"active": True},
        },
    )
    from proxy import mode

    assert mode._count_in_flight_local_streams() == 2


def test_count_in_flight_uses_counter_when_higher(monkeypatch):
    """The counter wins when it exceeds the active-record count."""
    import proxy.server as srv

    monkeypatch.setattr(srv, "local_active_queries", 5)
    monkeypatch.setattr(srv, "local_dispatch_records", {"sess-a": {"active": True}})
    from proxy import mode

    assert mode._count_in_flight_local_streams() == 5


def test_count_in_flight_fail_open(monkeypatch):
    """A missing/invalid records dict falls back to the counter."""
    import proxy.server as srv

    monkeypatch.setattr(srv, "local_active_queries", 3)
    monkeypatch.setattr(srv, "local_dispatch_records", None)
    from proxy import mode

    assert mode._count_in_flight_local_streams() == 3


def test_wait_for_in_flight_waits_for_active_records(monkeypatch):
    """A drain deadline holds while an active dispatch record is present."""
    import proxy.server as srv

    from proxy import mode

    # Counter is 0 but a real stream is active; then it clears.
    state = {"calls": 0}

    def _count():
        state["calls"] += 1
        return 1 if state["calls"] < 3 else 0

    monkeypatch.setattr(mode, "_count_in_flight_local_streams", _count)
    monkeypatch.setattr(mode.time, "sleep", lambda _s: None)
    monkeypatch.setattr(mode, "_end_drain", lambda: None)

    deadline = time.monotonic() + 5
    mode._wait_for_in_flight_local_streams(deadline=deadline)
    assert state["calls"] >= 3, "Drain must poll until the real stream clears"


def test_wait_for_in_flight_bounded_by_deadline(monkeypatch):
    """A stuck stream cannot hold the restart past the deadline."""
    from proxy import mode

    monkeypatch.setattr(mode, "_count_in_flight_local_streams", lambda: 99)
    monkeypatch.setattr(mode.time, "sleep", lambda _s: None)
    monkeypatch.setattr(mode, "_end_drain", lambda: None)

    # Already-expired deadline returns immediately.
    mode._wait_for_in_flight_local_streams(deadline=time.monotonic() - 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Signature
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sig_path(tmp_path, monkeypatch):
    path = tmp_path / ".llama_server_signature.json"
    monkeypatch.setattr(lifecycle, "_llama_server_signature_path", lambda: path)
    return path


def test_write_and_read_signature(sig_path, monkeypatch):
    monkeypatch.setattr(lifecycle, "_read_mode_label", lambda: "fast")
    monkeypatch.setattr(lifecycle, "_config_file_sha256", lambda: "abc123")
    lifecycle._write_llama_server_signature("Qwen3", 1, router_mode=True)

    sig = lifecycle._read_llama_server_signature()
    assert sig is not None
    assert sig["mode"] == "fast"
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
    from proxy.mode import proxy_dir

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
    from proxy.mode import proxy_dir

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
    from proxy.mode import proxy_dir

    monkeypatch.delenv("LLAMA_PROXY_CONFIG", raising=False)
    base_path = proxy_dir() / "config.yaml"
    assert lifecycle._config_file_sha256() == merged_config_sha256(
        base_path, base_path
    )


def test_config_file_sha256_tolerates_missing_overlay(monkeypatch, tmp_path):
    """A ``$LLAMA_PROXY_CONFIG`` pointing at a missing file falls back to
    the base rather than returning an error/empty string."""
    from proxy.config_merge import merged_config_sha256
    from proxy.mode import proxy_dir

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
