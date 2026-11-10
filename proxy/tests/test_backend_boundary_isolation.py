"""
Tests for the proxy↔backend supervision-boundary isolation (LP-0MW3PLK5M002YPOA).

The supervised ``llama-server`` router must be immune to a *stray* SIGKILL
sweep. The decoy experiment in the death-attribution doc proved the (then
unknown) sender does not match ``argv[0]``, so name-based hardening is not
enough: the durable answer is a **boundary**. The router runs under its own
supervision boundary — a systemd user unit with its own cgroup
(``user@1000.service/llama-server.service``, disjoint from the proxy's login
session scope) and its own session/process group — so that:

- a signal addressed to the **proxy's** process group or session cannot
  reach the backend (disjoint PGID/SID; the unit's cgroup extends the same
  boundary to systemd-level kills);
- the proxy's kill escalation (``proxy.lifecycle._kill_process_group``)
  targets only the process group of the handle the proxy owns — a backend
  outside that group is outside its kill set;
- ``Restart=always`` heals even a direct kill of the backend, so a stray
  sweep cannot strand it (on 2026-11-05 the proxy's own spawn failed with a
  missing shared library and the router stayed down for hours — a supervised
  backend would have been back within ``RestartSec``).

Signal delivery is per-pid/pgid, so process-group disjointness is the
mechanism the cgroup boundary builds on and it is fully reproducible without
root. These tests spawn real ``sleep`` trees in their own sessions — exactly
as ``test_backend_isolation.py`` does — and never signal anything outside
their own spawned groups.

A *coupled* backend (spawned inside the proxy's process group, the
pre-boundary coupling the unit removes) is the negative control: it dies
with the proxy group, proving the tests measure a real signal-delivery
difference rather than a no-op.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest
from proxy.lifecycle import _kill_process_group

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spawn_tree(
    lifetime: int = 120,
    new_session: bool = True,
    argv0: str | None = None,
) -> subprocess.Popen:
    """Spawn a fake process tree.

    ``new_session=True`` mirrors the boundary spawners (the systemd unit /
    ``start_new_session=True``): the process is its own session and
    process-group leader. ``argv0`` sets ``argv[0]`` via ``exec -a`` so a
    name-based sweep can select (or avoid) it.
    """
    if argv0 is not None:
        cmd = ["bash", "-c", f"exec -a {argv0} sleep {lifetime}"]
    else:
        cmd = ["bash", "-c", f"exec sleep {lifetime}"]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=new_session,
    )


def _find_child_of(ppid: int, timeout: float = 3.0) -> int | None:
    """Find a direct child of *ppid* (bounded retry: the child may not have
    forked yet when the parent returns from ``Popen``)."""
    deadline = time.monotonic() + timeout
    while True:
        child = _scan_for_child(ppid)
        if child is not None:
            return child
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.05)


def _scan_for_child(ppid: int) -> int | None:
    """One /proc scan for a direct child of *ppid*."""
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as fh:
                fields = fh.read().split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if len(fields) < 4:
            continue
        try:
            stat_ppid = int(fields[3])
        except ValueError:
            continue
        if stat_ppid == ppid:
            return int(name)
    return None


def _pid_gone(pid: int) -> bool:
    """True when *pid* is gone or an unreaped zombie.

    A killed child becomes a zombie (``/proc/<pid>`` still exists and
    ``os.kill(pid, 0)`` succeeds) until its parent reaps it. For liveness
    assertions the zombie state == dead: the process was terminated.
    """
    try:
        stat = open(f"/proc/{pid}/stat").read()
    except (FileNotFoundError, ProcessLookupError):
        return True
    try:
        state = stat.rsplit(")", 1)[1].split()[0]
    except (IndexError, ValueError):
        return True
    return state == "Z"  # zombie: dead, awaiting wait()


def _wait_pid_dead(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _pid_gone(pid):
            return True
        time.sleep(0.05)
    return _pid_gone(pid)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill a spawned tree by PGID (best-effort cleanup)."""
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _alive(proc) -> bool:
    return proc is not None and proc.poll() is None


@pytest.fixture()
def proxy_and_backend():
    """A fake proxy (own session/group) plus a supervised backend (own
    session/group), disjoint like the systemd-unit deployment."""
    proxy = _spawn_tree()
    backend = _spawn_tree()
    try:
        assert _alive(proxy) and _alive(backend)
        assert os.getpgid(proxy.pid) == proxy.pid
        assert os.getpgid(backend.pid) == backend.pid
        assert os.getpgid(proxy.pid) != os.getpgid(backend.pid)
        yield proxy, backend
    finally:
        _kill_tree(proxy)
        _kill_tree(backend)


# ---------------------------------------------------------------------------
# AC2: a signal to the proxy's group (or the _kill_process_group path)
# does not reach the backend and does not restart it
# ---------------------------------------------------------------------------


class TestProxyGroupSweepSeparation:
    def test_backend_pgid_sid_disjoint_from_proxy(self, proxy_and_backend):
        """The boundary spawners give each side its own session + PGID."""
        proxy, backend = proxy_and_backend
        assert os.getpgid(proxy.pid) == proxy.pid  # session leader
        assert os.getpgid(backend.pid) == backend.pid
        assert os.getsid(backend.pid) != os.getsid(proxy.pid)

    def test_signal_to_proxy_process_group_does_not_reach_backend(
        self, proxy_and_backend
    ):
        """AC2: `kill -TERM -<proxy-pgid>` (a sweep of the proxy's whole
        group) takes the proxy down and leaves the backend untouched."""
        proxy, backend = proxy_and_backend
        backend_pid = backend.pid

        os.killpg(proxy.pid, signal.SIGTERM)

        assert proxy.wait(timeout=5) is not None  # proxy group died
        assert _alive(backend)  # the backend never felt the signal
        assert backend.pid == backend_pid  # and was not restarted
        assert os.getpgid(backend.pid) == backend_pid

    def test_kill_process_group_on_proxy_owned_handle_does_not_reach_backend(
        self, proxy_and_backend
    ):
        """AC2: the proxy's escalation (_kill_process_group) targets only the
        process group of the handle the proxy owns; a backend in its own
        group is outside the kill set."""
        proxy, backend = proxy_and_backend
        logger = logging.getLogger("test-boundary")

        killed = _kill_process_group(proxy, logger)

        assert killed is True  # proxy-owned pid + group were stopped
        assert proxy.poll() is not None
        assert _alive(backend)  # supervised backend untouched

    def test_coupled_backend_without_boundary_dies_with_proxy_group(self):
        """Negative control: without the boundary (the pre-unit coupling) the
        same sweep reaches the backend. Proves the positive tests measure a
        real signal-delivery difference, not a no-op."""
        proxy = subprocess.Popen(
            ["bash", "-c", "sleep 120 & wait"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            child = _find_child_of(proxy.pid)
            assert child is not None
            assert os.getpgid(child) == os.getpgid(proxy.pid)

            os.killpg(proxy.pid, signal.SIGKILL)
            assert proxy.wait(timeout=5) is not None
            assert _wait_pid_dead(child)  # the coupled child died with it
        finally:
            _kill_tree(proxy)

    def test_neighbour_name_sweep_does_not_match_backend(self):
        """AC3 plane, deterministic: a `pkill -f` sweep against a
        neighbouring process's name/argv does not match the supervised
        backend (the decoy experiment: the sweep does not select the
        backend's argv[0])."""
        marker = f"BOUNDARY-DECOY-{os.getpid()}"
        neighbour = _spawn_tree(argv0=marker)
        backend = _spawn_tree()
        try:
            result = subprocess.run(
                ["pkill", "-TERM", "-f", marker],
                capture_output=True,
            )
            assert result.returncode in (0, 1)  # 0=matched,1=nothing left
            assert _wait_pid_dead(neighbour.pid)  # the neighbour died
            neighbour.wait(timeout=5)  # reap the zombie
            assert neighbour.poll() is not None
            assert _alive(backend)  # the backend matched nothing
            assert os.getpgid(backend.pid) == backend.pid
        finally:
            _kill_tree(neighbour)
            _kill_tree(backend)


# ---------------------------------------------------------------------------
# The boundary report (/health surface, LP-0MW3PLK5M002YPOA AC1/AC4)
# ---------------------------------------------------------------------------


class TestCheckSupervisionBoundary:
    """``check_supervision_boundary`` classifies the live boundary honestly."""

    @patch("proxy.backend_boundary._unit_scope_report", return_value={"available": False})
    def test_unknown_when_backend_not_running(self, _unit):
        from proxy.backend_boundary import check_supervision_boundary

        srv = MagicMock()
        srv.config = {"server": {"llama_server_port": 8080}}
        srv.llama_process = None
        with patch("proxy.backend_boundary._discover_llama_pids", return_value=[]):
            report = check_supervision_boundary(srv)
        assert report["state"] == "unknown"
        assert report["violation"] is False
        assert report["backend"]["pids"] == []

    @patch("proxy.backend_boundary._unit_scope_report", return_value={"available": False})
    def test_isolated_for_a_disjoint_pgid(self, _unit):
        """The boundary spawner's disjoint session/PGID is 'isolated'."""
        from proxy.backend_boundary import check_supervision_boundary

        backend = _spawn_tree()
        try:
            srv = MagicMock()
            srv.config = {"server": {"llama_server_port": 8080}}
            srv.llama_process = MagicMock(pid=backend.pid)
            report = check_supervision_boundary(srv)
            assert report["state"] == "isolated"
            assert report["violation"] is False
            assert report["shared_pgids"] == []
            assert backend.pid in report["backend"]["pids"]
            assert report["proxy"]["pgid"] != os.getpgid(backend.pid)
        finally:
            _kill_tree(backend)

    @patch("proxy.backend_boundary._unit_scope_report", return_value={"available": False})
    def test_coupled_when_pgid_and_cgroup_are_shared(self, _unit):
        """No axis separates (inline spawn) => 'coupled' with a reason."""
        from proxy.backend_boundary import check_supervision_boundary

        backend = _spawn_tree()
        try:
            srv = MagicMock()
            srv.config = {"server": {"llama_server_port": 8080}}
            srv.llama_process = MagicMock(pid=backend.pid)
            # Pretend both live in one process group and one cgroup: the
            # historical inline coupling.
            with patch(
                "proxy.backend_boundary._pgid_of", return_value=os.getpid()
            ), patch(
                "proxy.backend_boundary._sid_of", return_value=os.getsid(0)
            ), patch(
                "proxy.backend_boundary._read_cgroup", return_value="/same.scope"
            ):
                report = check_supervision_boundary(srv)
            assert report["state"] == "coupled"
            assert report["violation"] is True
            assert report["shared_pgids"]
            assert report["shared_cgroups"] == ["/same.scope"]
            assert "process group" in report["reason"]
        finally:
            _kill_tree(backend)

    def test_health_wiring_reports_the_boundary_verdict(self):
        """/health surfaces the boundary; a check failure degrades to unknown."""
        from proxy.handlers import _supervision_boundary_health

        srv = MagicMock()
        report = _supervision_boundary_health(srv)
        assert set(report) >= {"guard", "state", "violation"}
        assert report["state"] in {"isolated", "coupled", "unknown"}

        with patch(
            "proxy.backend_boundary.check_supervision_boundary",
            side_effect=RuntimeError("boom"),
        ):
            degraded = _supervision_boundary_health(srv)
        assert degraded["state"] == "unknown"
        assert degraded["violation"] is False
