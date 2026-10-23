#!/usr/bin/env python3
"""Single-instance proxy guard (LP-0MVLHMAWV008EKTI AC2).

The 2026-10-23 incident had **two** ``uvicorn proxy.server:app`` processes
alive at once, both with ``proxy.log`` open for write; the one without a
listener was a wedged twin that could still rotate/truncate the shared log.
Port binding alone cannot prevent that — the twin was bound-less but alive.

The guard is an exclusive, process-lifetime ``flock`` on ``proxy.lock`` in the
log directory: the first process to acquire it owns the log; a second one
fails fast with the holder's pid. ``flock`` is released automatically when the
holding process exits (even on SIGKILL), so there is no stale-lock recovery
problem.

Contract under test (observable behaviour):

* the first acquire succeeds and records the holder pid;
* a second, independently-opened lock fails until the first is released;
* the lock is free again after the holder process exits (subprocess test);
* the server startup guard returns the lock when free, is idempotent, and
  raises with the holder's pid when another live instance owns the log.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROXY_DIR = Path(__file__).resolve().parents[1]

_HOLDER = r"""
import sys
import time

from proxy.single_instance import acquire_instance_lock

lock = acquire_instance_lock(sys.argv[1])
if lock is None:
    sys.exit(3)
print(lock.holder_pid, flush=True)
time.sleep(30)
"""


@pytest.fixture
def lock_path(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "proxy.lock"


@pytest.fixture(autouse=True)
def _clean_server_lock():
    """Reset the server-module lock so tests never share a held acquire."""
    import proxy.server as srv_mod

    srv_mod._startup_release_instance_lock()
    yield
    srv_mod._startup_release_instance_lock()


def test_first_acquire_succeeds_and_records_pid(lock_path: Path):
    from proxy.single_instance import acquire_instance_lock, read_holder_pid

    lock = acquire_instance_lock(lock_path)
    assert lock is not None
    try:
        assert read_holder_pid(lock_path) == os.getpid()
        assert lock.holder_pid == os.getpid()
    finally:
        lock.release()
    assert read_holder_pid(lock_path) == os.getpid(), (
        "releasing must not clear the last-held pid (diagnostics only)"
    )


def test_second_acquire_fails_until_release(lock_path: Path):
    from proxy.single_instance import acquire_instance_lock

    first = acquire_instance_lock(lock_path)
    assert first is not None
    try:
        second = acquire_instance_lock(lock_path)
        assert second is None, (
            "a second writer for the same log directory must be refused"
        )
    finally:
        first.release()

    # Released: a fresh acquire works again.
    again = acquire_instance_lock(lock_path)
    assert again is not None
    again.release()


def test_lock_freed_when_holder_process_exits(lock_path: Path):
    """The flock dies with the process — a crashed twin never wedges the lock."""
    from proxy.single_instance import acquire_instance_lock

    child = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(lock_path)],
        cwd=str(PROXY_DIR),
        env={**os.environ, "PYTHONPATH": str(PROXY_DIR)},
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        holder_pid = child.stdout.readline().strip()
        assert holder_pid == str(child.pid), (
            f"holder did not report its pid (got {holder_pid!r})"
        )

        # Alive: lock is held by the subprocess.
        assert acquire_instance_lock(lock_path) is None
    finally:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            child.kill()
            child.wait(timeout=10)
        child.stdout.close()

    # The kernel released the flock when the holder died.
    reclaimed = acquire_instance_lock(lock_path)
    assert reclaimed is not None, "the lock must be re-acquirable after the holder dies"
    reclaimed.release()


def test_acquire_creates_parent_directories(tmp_path: Path):
    from proxy.single_instance import acquire_instance_lock

    lock_path = tmp_path / "a" / "b" / "proxy.lock"
    lock = acquire_instance_lock(lock_path)
    assert lock is not None
    assert lock_path.is_file()
    lock.release()


def test_startup_guard_raises_with_holder_pid_when_taken(lock_path: Path):
    """The server refuses to start when another live instance owns the log."""
    from proxy.single_instance import acquire_instance_lock

    from proxy import server as srv_mod

    holder = acquire_instance_lock(lock_path)
    assert holder is not None
    try:
        with pytest.raises(RuntimeError) as excinfo:
            srv_mod._startup_acquire_instance_lock(lock_path=lock_path)
        assert "already running" in str(excinfo.value)
        assert str(os.getpid()) in str(excinfo.value), (
            "the error must name the holder pid so the operator can kill the "
            "specific stale process"
        )
    finally:
        holder.release()


def test_startup_guard_acquires_when_free_and_is_idempotent(lock_path: Path):
    from proxy import server as srv_mod

    lock = srv_mod._startup_acquire_instance_lock(lock_path=lock_path)
    assert lock is not None
    # A second call in the same process returns the held lock (idempotent).
    assert srv_mod._startup_acquire_instance_lock(lock_path=lock_path) is lock
    srv_mod._startup_release_instance_lock()
    assert srv_mod._instance_lock is None


def test_startup_guard_skip_env(lock_path: Path, monkeypatch):
    """LLAMA_PROXY_SKIP_INSTANCE_LOCK=1 disables the guard (dev/tests)."""
    from proxy import server as srv_mod

    monkeypatch.setenv("LLAMA_PROXY_SKIP_INSTANCE_LOCK", "1")
    result = srv_mod._startup_acquire_instance_lock(lock_path=lock_path)
    assert result is None


# ---------------------------------------------------------------------------
# Ordering: the lock is taken BEFORE the live log is archived
# (LP-0MVLHMAWV008EKTI AC1+AC2)
# ---------------------------------------------------------------------------


def test_startup_guard_lock_lives_in_the_configured_log_directory(
    tmp_path: Path, monkeypatch
):
    """The guard locks the same directory setup_logging() writes into.

    If the lock path and the log directory ever diverge, the guard would protect
    a directory nobody writes to and two processes could share a log again.
    """
    from proxy import server as srv_mod

    log_dir = tmp_path / "logs"
    config = {"logging": {"directory": str(log_dir)}}
    monkeypatch.setattr(srv_mod, "config", config)

    lock = srv_mod._startup_acquire_instance_lock()

    assert lock is not None
    assert lock.path == log_dir / "proxy.lock"
    srv_mod._startup_release_instance_lock()


def test_rejected_second_instance_leaves_the_live_log_in_place(
    tmp_path: Path, monkeypatch
):
    """A refused second instance must not rename the owning process's log.

    setup_logging() archives a non-empty ``proxy.log`` aside for restart
    safety. If the single-instance guard ran *after* that archive, a rejected
    second instance would rename the live log out from under the process that
    owns it — the file path would go stale and that writer's output would land
    in an archived inode. The guard therefore runs first: the rejection leaves
    every log file exactly where it was.
    """
    from proxy.single_instance import acquire_instance_lock

    from proxy import server as srv_mod

    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    live = log_dir / "proxy.log"
    live.write_text(
        "2026-10-23 22:04:16,001 - INFO - pre-restart window line\n",
        encoding="utf-8",
    )

    holder = acquire_instance_lock(log_dir / "proxy.lock")
    assert holder is not None, "test setup: the first instance must hold the lock"
    config = {"logging": {"directory": str(log_dir)}}
    monkeypatch.setattr(srv_mod, "load_config", lambda: config)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            srv_mod._startup_config_logging()
    finally:
        holder.release()

    assert str(os.getpid()) in str(excinfo.value)
    assert live.is_file(), "the owning process's live log must not be renamed away"
    assert "pre-restart window line" in live.read_text(encoding="utf-8")
    assert sorted(p.name for p in log_dir.glob("proxy.log.*")) == [], (
        "a rejected instance must not create an archive of the live log"
    )
