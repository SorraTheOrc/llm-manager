"""Single-instance guard for the proxy process (LP-0MVLHMAWV008EKTI AC2).

Why a lock and not just the port: on 2026-10-23 two
``python3 -m uvicorn proxy.server:app`` processes were alive at once. The
newest owned the ``:8000`` listener; its predecessor lingered with no listener
but with ``proxy.log`` open for write, so it could still rotate/truncate the
log the live process was writing — a second writer on the same log is how the
3 h 35 m verification window disappeared. A bound-less process never fails a
port check, so port binding alone cannot enforce "exactly one proxy".

The guard is an exclusive, **process-lifetime** ``flock`` on
``<log_dir>/proxy.lock``:

* the first process owns the log and records its pid in the lock file;
* a second process fails fast, naming the holder pid, instead of silently
  becoming a second writer;
* the kernel releases the lock when the holder exits (including ``SIGKILL``),
  so a crashed process can never leave a stale lock behind — no timeouts, no
  pid-liveness heuristics.

The lock is advisory and per log directory: a dev instance
(``LLAMA_PROXY_DEV=1``, separate log dir) and the production instance do not
contend.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

#: Lock file name inside the proxy log directory.
LOCK_FILENAME = "proxy.lock"


class InstanceLock:
    """A held exclusive lock on the proxy single-instance lock file."""

    def __init__(self, path: Path, handle):
        self.path = Path(path)
        self._handle = handle
        self.holder_pid = os.getpid()

    def release(self) -> None:
        """Release the lock (idempotent). The kernel also releases it on exit."""
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            handle.close()
        except OSError:
            pass

    def __enter__(self) -> InstanceLock:
        return self

    def __exit__(self, *_exc) -> None:
        self.release()

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"InstanceLock(path={self.path}, holder_pid={self.holder_pid})"


def acquire_instance_lock(lock_path: Path | str) -> InstanceLock | None:
    """Acquire the exclusive instance lock, or return ``None`` when held.

    Creates the parent directory when missing, so a first run in a fresh log
    directory works. Non-blocking: this is a fail-fast guard, never a wait.
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    lock = InstanceLock(lock_path, handle)
    _record_holder(handle)
    return lock


def _record_holder(handle) -> None:
    """Write this process's pid into the lock file (diagnostics only)."""
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        os.fsync(handle.fileno())
    except OSError:
        pass


def read_holder_pid(lock_path: Path | str) -> int | None:
    """Return the pid recorded by the current/last holder, or ``None``."""
    try:
        text = Path(lock_path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(text.split()[0])
    except (ValueError, IndexError):
        return None
