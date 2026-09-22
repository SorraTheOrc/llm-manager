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
