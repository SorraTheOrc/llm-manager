"""Log-sink relay survives adoption of a running llama-server (LP-0MVECX47Z004LJAQ).

Context
-------
``spawn_and_capture()`` used to start llama-server with
``stdout=subprocess.PIPE`` plus a **proxy-owned** reader thread that copied the
stream into ``llama-server.log``. On a proxy-only restart the new proxy ADOPTS
the still-running backend, but the reader thread died with the old proxy: the
pipe was left with writers only, so ``llama-server.log`` froze forever. That
silently disabled log-based diagnostics, froze ``GET /logs/tail?source=llama``
and fed stale slot progress to the Web UI (trigger for LP-0MVECWYWK000U3E1).

These tests pin the fix:

1. AC1/AC5 — llama-server's stdout is redirected **directly** into the log
   file, so the relay is owned by the backend process, not the proxy; a
   proxy-only restart that adopts the backend needs no re-attachment and the
   warm prompt cache is preserved (no restart is forced).
2. AC2 — when re-attachment is impossible (pre-fix pipe-based backend, nobody
   holds the log for writing) the proxy logs a WARNING and marks the llama log
   source unhealthy rather than freezing silently.
3. AC3 — ``/health`` exposes ``llama_log_age_seconds`` and
   ``llama_log_source_healthy`` so staleness is observable without shell access.
4. AC4 — adopt-then-reattach: adopting a running backend restarts the log
   tailer, and prefill progress lines from the adopted backend's log are again
   logged.

All tests are hermetic: no live llama-server, no GPU.
"""

import asyncio
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import proxy.lifecycle as lifecycle
import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_srv(log_dir: Path | None = None, **overrides) -> SimpleNamespace:
    base = dict(
        config={"server": {"llama_server_port": 8080, "llama_router_mode": True}},
        llama_process=None,
        current_model=None,
        backend_ready=False,
        llama_log_file=None,
        log_dir=log_dir,
        logger=MagicMock(),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def sig_path(tmp_path, monkeypatch):
    path = tmp_path / ".llama_server_signature.json"
    monkeypatch.setattr(lifecycle, "_llama_server_signature_path", lambda: path)
    return path


@pytest.fixture(autouse=True)
def _stop_tailer_between_tests():
    yield
    lifecycle.stop_llama_log_tailer()


# ═══════════════════════════════════════════════════════════════════════════════
# AC1/AC5: direct file redirection — relay owned by the backend
# ═══════════════════════════════════════════════════════════════════════════════


def test_spawn_redirects_stdout_directly_to_log_file(tmp_path):
    """The child's stdout/stderr are the log file itself — no proxy-owned pipe.

    This is the core of the fix: because the write fd belongs to the backend,
    killing the proxy (adoption) cannot break the relay.
    """
    log_path = tmp_path / "llama-server.log"
    log_file = open(log_path, "w", encoding="utf-8")
    try:
        proc, captured = lifecycle.spawn_and_capture(
            [
                sys.executable,
                "-c",
                "import sys, time; print('backend line', flush=True); time.sleep(30)",
            ],
            env={},
            log_file=log_file,
            logger=MagicMock(),
            model_name="Qwen3",
        )
        assert proc is not None, "long-running backend must be returned"
        assert captured is None

        # The backend writes into the file through its OWN fd.
        deadline = time.time() + 10
        content = ""
        while time.time() < deadline:
            log_file.flush()
            content = log_path.read_text(encoding="utf-8")
            if "backend line" in content:
                break
            time.sleep(0.05)
        assert "backend line" in content, f"backend stdout must land in the log file directly; got {content!r}"
        # No reader thread is needed / started in this design.
        assert not any(t.name.startswith("Thread-") and "stream" in (t.name or "") for t in threading.enumerate())
        proc.terminate()
        proc.wait(timeout=10)
    finally:
        log_file.close()


def test_spawn_reads_startup_diagnostics_from_log_on_quick_exit(tmp_path):
    """A backend that exits within the probe window returns its log tail.

    With direct redirection there is no pipe to drain, so the quick-exit
    diagnostics must be recovered from the log file (otherwise failed starts
    would report an empty reason).
    """
    log_path = tmp_path / "llama-server.log"
    log_file = open(log_path, "w", encoding="utf-8")
    try:
        proc, captured = lifecycle.spawn_and_capture(
            [
                sys.executable,
                "-c",
                "import sys; print('fatal: model not found', file=sys.stderr); sys.exit(2)",
            ],
            env={},
            log_file=log_file,
            logger=MagicMock(),
            model_name="Qwen3",
        )
        assert proc is None
        assert captured is not None
        assert "fatal: model not found" in captured
    finally:
        log_file.close()


def test_spawn_without_log_file_still_uses_pipe_relay():
    """DEVNULL/None keeps the legacy pipe relay (no durable sink to protect)."""
    proc, captured = lifecycle.spawn_and_capture(
        [sys.executable, "-c", "print('quick')"],
        env={},
        log_file=subprocess.DEVNULL,
        logger=MagicMock(),
        model_name="Qwen3",
    )
    assert proc is None
    assert captured is not None and "quick" in captured


# ═══════════════════════════════════════════════════════════════════════════════
# AC2: impossible re-attachment is loud, not silent
# ═══════════════════════════════════════════════════════════════════════════════


def test_log_has_live_writer_detects_child_writer(tmp_path):
    log_path = tmp_path / "llama-server.log"
    log_file = open(log_path, "w", encoding="utf-8")
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    # Parent must not hold a write fd, otherwise it counts as a writer too.
    log_file.close()
    try:
        assert lifecycle._log_has_live_writer(log_path) is True
        # Excluding the only writer proves no OTHER process holds it. This is
        # deterministic (no dependence on unrelated machine processes).
        assert (
            lifecycle._log_has_live_writer(log_path, exclude_pid=child.pid) is False
        )
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_log_has_live_writer_false_when_never_opened(tmp_path):
    """A log file nobody holds open for writing reports no live writer."""
    log_path = tmp_path / "llama-server.log"
    log_path.write_text("frozen\n", encoding="utf-8")
    assert lifecycle._log_has_live_writer(log_path) is False


def test_log_has_live_writer_ignores_read_only_holders(tmp_path):
    """A read-only holder (log viewer / tailer) must not mask a broken relay."""
    log_path = tmp_path / "llama-server.log"
    log_path.write_text("x\n", encoding="utf-8")
    with open(log_path, encoding="utf-8") as reader:
        assert lifecycle._log_has_live_writer(log_path) is False
        assert reader.read() == "x\n"


def test_adoption_warns_and_marks_unhealthy_when_relay_broken(sig_path, tmp_path, monkeypatch):
    """Pre-fix pipe-based backend: adoption cannot re-attach — WARNING + unhealthy."""
    log_path = tmp_path / "llama-server.log"
    log_path.write_text("frozen\n", encoding="utf-8")  # exists, nobody holds it

    lifecycle._write_llama_server_signature("Qwen3", 1, router_mode=False)
    sig = json.loads(sig_path.read_text(encoding="utf-8"))
    sig["config_sha256"] = "hash"
    sig_path.write_text(json.dumps(sig), encoding="utf-8")
    monkeypatch.setattr(lifecycle, "_config_file_sha256", lambda: "hash")
    monkeypatch.setattr(lifecycle, "_probe_backend_reachable", AsyncMock(return_value=True))
    # Hermetic: no real /proc scan.
    monkeypatch.setattr(lifecycle, "_log_has_live_writer", lambda *_a, **_k: False)
    monkeypatch.setattr(lifecycle, "start_llama_log_tailer", lambda *_a, **_k: None)

    srv = _make_srv(log_dir=tmp_path, config={"server": {"llama_server_port": 8080}})
    assert asyncio.run(lifecycle._try_adopt_running_llama_server(srv, "Qwen3", router_mode=False)) is True

    # AC2: loud, not silent.
    warnings = [str(c.args[0]) for c in srv.logger.warning.call_args_list]
    assert any("relay is NOT attached" in w for w in warnings), warnings
    assert getattr(srv, "llama_log_source_unhealthy", False) is True
    # AC5: adoption still preserves the backend (no forced restart).
    assert getattr(srv.llama_process, "adopted", False) is True
    assert srv.backend_ready is True


def test_adoption_reports_attached_when_live_writer_present(sig_path, tmp_path, monkeypatch):
    log_path = tmp_path / "llama-server.log"
    log_path.write_text("flowing\n", encoding="utf-8")

    lifecycle._write_llama_server_signature("Qwen3", 1, router_mode=False)
    sig = json.loads(sig_path.read_text(encoding="utf-8"))
    sig["config_sha256"] = "hash"
    sig_path.write_text(json.dumps(sig), encoding="utf-8")
    monkeypatch.setattr(lifecycle, "_config_file_sha256", lambda: "hash")
    monkeypatch.setattr(lifecycle, "_probe_backend_reachable", AsyncMock(return_value=True))
    monkeypatch.setattr(lifecycle, "_log_has_live_writer", lambda *_a, **_k: True)
    monkeypatch.setattr(lifecycle, "start_llama_log_tailer", lambda *_a, **_k: None)

    srv = _make_srv(log_dir=tmp_path, config={"server": {"llama_server_port": 8080}})
    assert asyncio.run(lifecycle._try_adopt_running_llama_server(srv, "Qwen3", router_mode=False)) is True

    assert not getattr(srv, "llama_log_source_unhealthy", False)
    infos = [str(c.args[0]) for c in srv.logger.info.call_args_list]
    assert any("relay attached" in i for i in infos), infos


# ═══════════════════════════════════════════════════════════════════════════════
# AC4: adopt-then-reattach — progress logging resumes on the adopted backend
# ═══════════════════════════════════════════════════════════════════════════════


def test_adoption_restarts_progress_tailer(sig_path, tmp_path, monkeypatch):
    """Adoption re-attaches the log sink by restarting the progress tailer."""
    log_path = tmp_path / "llama-server.log"
    log_path.write_text("", encoding="utf-8")

    lifecycle._write_llama_server_signature("Qwen3", 1, router_mode=False)
    sig = json.loads(sig_path.read_text(encoding="utf-8"))
    sig["config_sha256"] = "hash"
    sig_path.write_text(json.dumps(sig), encoding="utf-8")
    monkeypatch.setattr(lifecycle, "_config_file_sha256", lambda: "hash")
    monkeypatch.setattr(lifecycle, "_probe_backend_reachable", AsyncMock(return_value=True))
    monkeypatch.setattr(lifecycle, "_log_has_live_writer", lambda *_a, **_k: True)

    started = []
    monkeypatch.setattr(
        lifecycle,
        "start_llama_log_tailer",
        lambda srv, model_name="unknown", **kw: started.append(model_name),
    )

    srv = _make_srv(log_dir=tmp_path, config={"server": {"llama_server_port": 8080}})
    assert asyncio.run(lifecycle._try_adopt_running_llama_server(srv, "Qwen3", router_mode=False)) is True
    assert started == ["Qwen3"], "adoption must re-attach the log tailer"


def _progress_line(slot_id: int, n_tokens: int, progress: float) -> str:
    return f"slot {slot_id} : prompt processing, n_tokens={n_tokens}, progress={progress}\n"


def test_tailer_logs_progress_from_appended_lines(tmp_path):
    """The tailer picks up lines appended after it starts (adopted backend)."""
    log_path = tmp_path / "llama-server.log"
    log_path.write_text("", encoding="utf-8")

    logger = MagicMock()
    stop = threading.Event()
    thread = threading.Thread(
        target=lifecycle._llama_log_progress_tailer,
        args=(log_path, "Qwen3", logger, stop),
        kwargs={"poll_interval": 0.02},
        daemon=True,
    )
    thread.start()
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(_progress_line(1, 100, 0.5))
            fh.flush()
        deadline = time.time() + 10
        while time.time() < deadline and not logger.info.called:
            time.sleep(0.02)
        assert logger.info.called, "tailer must log prefill progress lines"
        logged = str(logger.info.call_args_list[0].args[0])
        assert "[slot:1 Qwen3]" in logged
        assert "(50%)" in logged
    finally:
        stop.set()
        thread.join(timeout=5)


def test_tailer_follows_log_rotation(tmp_path):
    """Rotation (rename + new file) is followed, so logging never silently stops."""
    log_path = tmp_path / "llama-server.log"
    log_path.write_text("", encoding="utf-8")

    logger = MagicMock()
    stop = threading.Event()
    thread = threading.Thread(
        target=lifecycle._llama_log_progress_tailer,
        args=(log_path, "Qwen3", logger, stop),
        kwargs={"poll_interval": 0.02},
        daemon=True,
    )
    thread.start()
    try:
        # Simulate rotation: rename away and create a fresh log file.
        time.sleep(0.1)
        rotated = tmp_path / "llama-server.log.1"
        log_path.rename(rotated)
        log_path.write_text(_progress_line(2, 200, 1.0), encoding="utf-8")

        deadline = time.time() + 10
        while time.time() < deadline and not logger.info.called:
            time.sleep(0.02)
        assert logger.info.called, "tailer must follow the rotated log file"
        assert "[slot:2 Qwen3]" in str(logger.info.call_args_list[0].args[0])
    finally:
        stop.set()
        thread.join(timeout=5)


def test_stop_tailer_ends_thread_and_is_idempotent(tmp_path):
    log_path = tmp_path / "llama-server.log"
    log_path.write_text("", encoding="utf-8")
    srv = _make_srv(log_dir=tmp_path)

    lifecycle.start_llama_log_tailer(srv, model_name="Qwen3")
    thread = lifecycle._log_tailer_thread
    assert thread is not None and thread.is_alive()

    lifecycle.stop_llama_log_tailer()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert lifecycle._log_tailer_thread is None

    # Idempotent.
    lifecycle.stop_llama_log_tailer()


def test_start_tailer_replaces_existing_tailer(tmp_path):
    """Re-attachment restarts rather than stacking tailers."""
    log_path = tmp_path / "llama-server.log"
    log_path.write_text("", encoding="utf-8")
    srv = _make_srv(log_dir=tmp_path)

    lifecycle.start_llama_log_tailer(srv, model_name="Qwen3")
    first = lifecycle._log_tailer_thread
    lifecycle.start_llama_log_tailer(srv, model_name="Qwen3")
    second = lifecycle._log_tailer_thread

    assert second is not None and second.is_alive()
    assert first is not second
    first.join(timeout=5)
    assert not first.is_alive()


def test_spawn_clears_stale_unhealthy_flag(tmp_path, monkeypatch):
    """A fresh spawn re-establishes the relay, clearing a stale unhealthy verdict."""
    class FakeProc:
        def poll(self):
            return None

    srv = _make_srv(log_dir=tmp_path, llama_log_source_unhealthy=True)
    srv.rotate_llama_logs = lambda *a, **k: None
    srv.broadcast_status_sync = lambda *a, **k: None
    srv.last_start_failure = None
    monkeypatch.setattr(lifecycle, "_srv", lambda: srv)
    monkeypatch.setattr(lifecycle, "spawn_and_capture", lambda *a, **k: (FakeProc(), None))
    monkeypatch.setattr(lifecycle, "start_llama_log_tailer", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_write_llama_server_signature", lambda *a, **k: None)

    assert lifecycle.start_llama_server("Qwen3") is not None
    assert srv.llama_log_source_unhealthy is False


# ═══════════════════════════════════════════════════════════════════════════════
# AC3: freshness observable via /health
# ═══════════════════════════════════════════════════════════════════════════════


def test_llama_log_age_seconds(tmp_path):
    log_path = tmp_path / "llama-server.log"
    log_path.write_text("x\n", encoding="utf-8")
    old = time.time() - 120
    import os as _os

    _os.utime(log_path, (old, old))
    srv = _make_srv(log_dir=tmp_path)
    age = lifecycle._llama_log_age_seconds(srv)
    assert age is not None and 110 < age < 200


def test_llama_log_age_seconds_absent_returns_none(tmp_path):
    srv = _make_srv(log_dir=tmp_path)
    assert lifecycle._llama_log_age_seconds(srv) is None


@pytest.mark.asyncio
async def test_health_exposes_log_freshness_fields(tmp_path, monkeypatch):
    """AC3: /health exposes log age + health so staleness is observable."""
    import proxy.handlers as handlers
    import proxy.server as server

    log_path = tmp_path / "llama-server.log"
    log_path.write_text("x\n", encoding="utf-8")

    class Proc:
        def poll(self):
            return None

    monkeypatch.setattr(server, "llama_process", Proc())
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "current_model", "qwen3")
    monkeypatch.setattr(server, "config", {"server": {"llama_router_mode": False}})
    monkeypatch.setattr(server, "log_dir", tmp_path)
    monkeypatch.setattr(server, "llama_log_source_unhealthy", False, raising=False)
    monkeypatch.setattr(server, "_probe_backend_reachable", AsyncMock(return_value=True))
    monkeypatch.setattr(server, "_is_self_healing_active", lambda: False)
    monkeypatch.setattr(server, "backend_signal_counts", {}, raising=False)
    monkeypatch.setattr(server, "_backend_recovery_snapshot", lambda: {}, raising=False)

    health = await handlers.health_check()
    assert "llama_log_age_seconds" in health
    assert health["llama_log_age_seconds"] is not None
    assert health["llama_log_age_seconds"] >= 0
    assert health["llama_log_source_healthy"] is True


@pytest.mark.asyncio
async def test_health_reports_unhealthy_after_broken_relay(tmp_path, monkeypatch):
    import proxy.handlers as handlers
    import proxy.server as server

    log_path = tmp_path / "llama-server.log"
    log_path.write_text("frozen\n", encoding="utf-8")
    old = time.time() - 600
    import os as _os

    _os.utime(log_path, (old, old))

    class Proc:
        def poll(self):
            return None

    monkeypatch.setattr(server, "llama_process", Proc())
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "current_model", "qwen3")
    monkeypatch.setattr(server, "config", {"server": {"llama_router_mode": False}})
    monkeypatch.setattr(server, "log_dir", tmp_path)
    monkeypatch.setattr(server, "llama_log_source_unhealthy", True, raising=False)
    monkeypatch.setattr(server, "_probe_backend_reachable", AsyncMock(return_value=True))
    monkeypatch.setattr(server, "_is_self_healing_active", lambda: False)
    monkeypatch.setattr(server, "backend_signal_counts", {}, raising=False)
    monkeypatch.setattr(server, "_backend_recovery_snapshot", lambda: {}, raising=False)

    health = await handlers.health_check()
    assert health["llama_log_source_healthy"] is False
