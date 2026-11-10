"""
GPU-backend process-group isolation guard (LP-0MW3K7YO1005C41X).

The `libamdhip64.so.7.2.70204` general-protection fault (deterministic offset
``0x2b931f``, a wild-pointer deref inside ROCm's HIP launch path) kills only
the faulting thread group — the llama-server tree. The bound chosen for it is
that the other GPU process (the TTS server tree) must never share a process
group with llama-server, so the restart/stop that follows a fault in one
backend can never signal the other.

Both backend spawners use ``start_new_session=True`` (each backend is its own
session and process-group leader), so the invariant holds by construction.
This module makes the invariant explicit, verifiable on live processes
(:func:`check_backend_isolation`), and guarded:

- the watchdogs re-assert it periodically via :func:`assert_backend_isolation`
  and log a loud ERROR on a violation instead of a silent cascade;
- :mod:`proxy.lifecycle`'s ``_kill_process_group`` accepts ``forbidden_pids``
  (the other backend's tree) and refuses — fails closed — when a forbidden
  pid is a member of the target process group;
- ``/health`` surfaces the report (``backend_isolation``) so a violation is
  observable without shell access.

Everything here is fail-open for *resolution* (a pid whose PGID cannot be
resolved is skipped) and fail-closed only on *positive evidence* of sharing
(the kill is refused, the watchdogs log, /health reports ``violation``).
"""

from __future__ import annotations

import logging
import os
import time

try:  # module-level so tests can patch it; best-effort like every resolver here
    from proxy.backend_rss import discover_llama_pids
except Exception:  # pragma: no cover - defensive

    def discover_llama_pids(port=None):  # type: ignore[misc]
        return []


#: Throttle for the watchdog re-assertion (seconds). The /proc scan is cheap,
#: but the watchdogs run every few seconds and nothing needs faster feedback.
ISOLATION_ASSERT_THROTTLE_SECONDS = 30.0

_last_assert_at: float = 0.0


# ---------------------------------------------------------------------------
# /proc helpers
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    """True when a pid exists (signal 0 probe)."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _pgid_of(pid: int) -> int | None:
    """Live process-group id of *pid*, or None when unresolvable."""
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        return None


def tracked_backend_pids(srv) -> dict[str, list[int]]:
    """The tracked (handle + discovered) pid lists of both backends.

    Returns ``{"llama": [...], "tts": [...]}`` of live, distinct, ``> 1``
    pids. The llama list uses the process handle when this proxy spawned it;
    an adopted backend (handle pid 0) is resolved by port discovery just like
    :mod:`proxy.backend_rss`, so the isolation verdict still has something to
    assert. The TTS list is the tracked handle pid (the TTS wrapper owns the
    session; the ``tts-server`` child inherits its PGID).
    """
    llama: list[int] = []
    tts: list[int] = []

    llama_proc = getattr(srv, "llama_process", None)
    llama_pid = _coerce_pid(getattr(llama_proc, "pid", None))
    if llama_pid and llama_pid > 1:
        llama.append(llama_pid)
    else:
        # Adopted / no handle: find the live backend by its port.
        try:
            server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}
            port = int(server_cfg.get("llama_server_port", 8080) or 8080)
            llama.extend(p for p in discover_llama_pids(port) if p > 1)
        except Exception:
            pass

    tts_proc = getattr(srv, "tts_process", None)
    tts_pid = _coerce_pid(getattr(tts_proc, "pid", None))
    if tts_pid and tts_pid > 1:
        tts.append(tts_pid)

    return {"llama": sorted(set(llama)), "tts": sorted(set(tts))}


def _coerce_pid(value) -> int | None:
    """Coerce an object to a positive int pid, None when not one."""
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    if pid <= 1:
        return None
    return pid


# ---------------------------------------------------------------------------
# The isolation check
# ---------------------------------------------------------------------------


def check_backend_isolation(srv) -> dict:
    """Report whether the two GPU backends live in disjoint process groups.

    The verdict:

    - ``state: "isolated"`` — both backends are running and their process
      groups are disjoint: a restart/stop of one cannot signal the other.
    - ``state: "violation"`` — the two backends share a pid or a process
      group; a fault in one could cascade to the other. This is the
      fail-closed signal the watchdogs log and /health surfaces.
    - ``state: "unknown"`` — one backend is not running (or unresolvable);
      there is nothing to assert (fail-open, e.g. TTS disabled).

    ``violation`` is always a bool; ``shared_pids`` / ``shared_pgids`` name
    the offenders. Never raises: every resolution failure degrades to
    ``unknown`` or skips that pid.
    """
    report = {
        "guard": "process-group isolation (start_new_session + disjoint PGIDs)",
        "state": "unknown",
        "violation": False,
        "llama": {"pids": [], "pgids": []},
        "tts": {"pids": [], "pgids": []},
        "shared_pids": [],
        "shared_pgids": [],
        "reason": None,
    }
    try:
        tracked = tracked_backend_pids(srv)
    except Exception:
        return report

    llama_pids = [p for p in tracked["llama"] if _pid_alive(p)]
    tts_pids = [p for p in tracked["tts"] if _pid_alive(p)]

    for key, pids in (("llama", llama_pids), ("tts", tts_pids)):
        pgids = sorted({p for p in (_pgid_of(pid) for pid in pids) if p})
        report[key] = {"pids": pids, "pgids": pgids}

    if not llama_pids or not tts_pids:
        # One backend not running: nothing to protect against.
        return report

    shared_pids = sorted(set(llama_pids) & set(tts_pids))
    llama_pgids = set(report["llama"]["pgids"])
    tts_pgids = set(report["tts"]["pgids"])
    shared_pgids = sorted(llama_pgids & tts_pgids)

    report["shared_pids"] = shared_pids
    report["shared_pgids"] = shared_pgids

    if shared_pids or shared_pgids:
        report["state"] = "violation"
        report["violation"] = True
        report["reason"] = _violation_reason(shared_pids, shared_pgids)
    else:
        report["state"] = "isolated"
    return report


def _violation_reason(shared_pids: list[int], shared_pgids: list[int]) -> str:
    parts = []
    if shared_pids:
        parts.append(f"shared pids {shared_pids}")
    if shared_pgids:
        parts.append(f"shared process groups {shared_pgids}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def assert_backend_isolation(srv, logger: logging.Logger) -> bool:
    """Re-assert the process-group isolation invariant (watchdog guard).

    Logs an ERROR naming the shared process groups/pids when the two GPU
    backends have collapsed into one process group — the exact condition under
    which a fault in one backend would signal the other. Throttled (see
    :data:`ISOLATION_ASSERT_THROTTLE_SECONDS`) and fail-open: an unexpected
    error never turns into a false alarm or a crash.

    Returns True when isolation holds (or cannot be judged), False on a
    verified violation.
    """
    global _last_assert_at
    now = time.time()
    if now - _last_assert_at < ISOLATION_ASSERT_THROTTLE_SECONDS:
        return True
    _last_assert_at = now
    try:
        report = check_backend_isolation(srv)
    except Exception:
        return True
    if report.get("state") != "violation":
        return True
    try:
        reason = report.get("reason") or "shared process group"
        logger.error(
            "backend isolation VIOLATION: llama-server and the TTS server share "
            "%s — a fault in one GPU backend could now signal the other; the "
            "isolation guard (start_new_session + disjoint PGIDs) has been "
            "defeated. Restart supervision is independent, but the kill paths "
            "fail closed only while no pid is shared. (LP-0MW3K7YO1005C41X)",
            reason,
        )
    except Exception:
        pass
    return False
