"""
Tests for the GPU-backend process-group isolation guard (LP-0MW3K7YO1005C41X).

A `libamdhip64.so.7.2.70204` general-protection fault kills only the faulting
thread group (the llama-server tree). The bound for that GPF is that the other
GPU process (the TTS server tree) must never share a process group with
llama-server, so a restart/stop of one backend cannot signal the other.

Both backends are spawned with ``start_new_session=True`` (separate session and
process group), so the invariant holds by construction; the guard makes it
explicit, verifiable on live processes, and regression-tested here:

- ``proxy.backend_isolation.check_backend_isolation`` resolves the live PGIDs
  of both backend trees and reports a ``violation`` when any process group is
  shared (a death in one backend would then signal the other).
- ``proxy.backend_isolation.assert_backend_isolation`` logs the violation
  loudly from the watchdogs (the "guard").
- ``proxy.lifecycle._kill_process_group`` accepts ``forbidden_pids`` and
  **fails closed**: it refuses to kill a process group that a forbidden pid
  (the other backend) is a member of, so a cross-backend kill can never
  cascade a fault.

The end-to-end tests use real processes and the real ``/proc`` (spawning
``sleep`` trees in their own sessions, as the spawners do); they never signal
anything outside their own spawned process groups.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from proxy.backend_isolation import (
    assert_backend_isolation,
    check_backend_isolation,
    tracked_backend_pids,
)
from proxy.lifecycle import _kill_process_group

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spawn_backend_tree(lifetime: int = 120) -> subprocess.Popen:
    """Spawn a fake GPU backend: a session leader with one child.

    Mirrors the real spawners (``spawn_and_capture`` / ``start_tts_server``):
    ``start_new_session=True`` gives the wrapper its own session + PGID and the
    child inherits that PGID, exactly like ``start-llama.sh`` ->
    ``llama-server`` and ``start-qwentts.sh`` -> ``tts-server``.
    """
    proc = subprocess.Popen(
        ["bash", "-c", f"exec sleep {lifetime}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill a spawned tree by PGID (best-effort cleanup)."""
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, 9)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _fake_srv(llama_proc=None, tts_proc=None, llama_port: int = 8080):
    """Build a minimal fake server object exposing the used attrs."""
    srv = MagicMock()
    srv.config = {"server": {"llama_server_port": llama_port}}
    srv.llama_process = llama_proc
    srv.tts_process = tts_proc
    return srv


def _alive(proc) -> bool:
    return proc is not None and proc.poll() is None


# ---------------------------------------------------------------------------
# End-to-end: real processes, the AC2 regression
# ---------------------------------------------------------------------------


class TestRealProcessIsolation:
    """A death in one backend tree must not signal the other (AC2)."""

    @pytest.fixture()
    def two_trees(self):
        llama = _spawn_backend_tree()
        tts = _spawn_backend_tree()
        try:
            assert _alive(llama) and _alive(tts)
            yield llama, tts
        finally:
            _kill_tree(llama)
            _kill_tree(tts)

    def test_spawned_backends_have_disjoint_process_groups(self, two_trees):
        """start_new_session=True gives each backend its own PGID/SID."""
        llama, tts = two_trees
        llama_pgid = os.getpgid(llama.pid)
        tts_pgid = os.getpgid(tts.pid)
        assert llama_pgid != tts_pgid
        assert llama_pgid == llama.pid  # session leader => PGID == own PID
        assert tts_pgid == tts.pid

    def test_isolation_check_reports_isolated_for_disjoint_trees(self, two_trees):
        """Live PGIDs of the two trees are disjoint => state 'isolated'."""
        llama, tts = two_trees
        srv = _fake_srv(llama_proc=llama, tts_proc=tts)
        report = check_backend_isolation(srv)
        assert report["state"] == "isolated"
        assert report["violation"] is False
        assert llama.pid in report["llama"]["pgids"]
        assert tts.pid in report["tts"]["pgids"]
        assert report["shared_pgids"] == []
        assert report["shared_pids"] == []

    def test_killing_one_backend_does_not_signal_the_other(self, two_trees):
        """The kill path of one backend leaves the other's tree alive.

        This is the AC2 regression: a fault in llama-server triggers an
        proxy-side restart that kills llama's process group; the TTS tree must
        survive. The guard's ``forbidden_pids`` makes the cross-backend kill
        impossible even if the PGIDs ever collapsed.
        """
        llama, tts = two_trees
        logger = logging.getLogger("test-backend-isolation")

        killed = _kill_process_group(llama, logger, forbidden_pids=[tts.pid])
        assert killed is True
        assert llama.poll() is not None  # llama tree died
        assert _alive(tts)  # TTS tree untouched

    def test_kill_with_forbidden_overlap_refuses_without_killing(self, two_trees):
        """Failed-closed: a forbidden pid in the target PG stops the kill.

        Simulates the catastrophic misconfiguration (both backends collapsed
        into one process group): the kill must be refused, not executed.
        """
        llama, tts = two_trees
        logger = logging.getLogger("test-backend-isolation")

        # Force the guard's view of the world: make the TTS pid a member of
        # llama's PG by resolving llama's own pgid as the forbidden pgid.
        # The guard resolves each forbidden pid's PGID; if we pass the llama
        # pid itself as 'forbidden', its PGID equals the target PGID and the
        # kill must be refused while the process stays alive.
        refused = _kill_process_group(llama, logger, forbidden_pids=[llama.pid])
        assert refused is False
        assert _alive(llama)  # nothing was signalled


# ---------------------------------------------------------------------------
# Unit: the forbidden-pid kill guard (_kill_process_group)
# ---------------------------------------------------------------------------


class TestKillProcessGroupForbidden:
    def test_refuses_when_forbidden_pid_shares_target_pgid(self):
        proc = MagicMock()
        proc.pid = 12345
        proc.poll.return_value = None
        with patch("proxy.lifecycle.os.getpgid", side_effect=lambda p: 111):
            with patch("proxy.lifecycle.os.killpg") as mock_killpg:
                result = _kill_process_group(proc, MagicMock(), forbidden_pids=[9999])
        mock_killpg.assert_not_called()
        assert result is False

    def test_proceeds_when_forbidden_pid_in_disjoint_pgid(self):
        proc = MagicMock()
        proc.pid = 12345
        proc.poll.return_value = None
        real_pgids = {12345: 111, 9999: 222}
        with patch("proxy.lifecycle.os.getpgid", side_effect=lambda p: real_pgids[p]):
            with patch("proxy.lifecycle.os.killpg") as mock_killpg:
                result = _kill_process_group(proc, MagicMock(), forbidden_pids=[9999])
        mock_killpg.assert_called_once_with(12345, 15)  # SIGTERM
        assert result is True

    def test_proceeds_when_forbidden_pid_is_dead(self):
        """A dead forbidden pid cannot be resolved => no refusal (fail-open on
        resolution, fail-closed only on positive evidence of sharing)."""
        proc = MagicMock()
        proc.pid = 12345
        proc.poll.return_value = None

        def getpgid(pid):
            if pid == 12345:
                return 111
            raise ProcessLookupError(3, "No such process")

        with patch("proxy.lifecycle.os.getpgid", side_effect=getpgid):
            with patch("proxy.lifecycle.os.killpg") as mock_killpg:
                result = _kill_process_group(proc, MagicMock(), forbidden_pids=[9999])
        mock_killpg.assert_called_once_with(12345, signal.SIGTERM)
        assert result is True

    def test_no_forbidden_pids_behaves_unchanged(self):
        proc = MagicMock()
        proc.pid = 12345
        proc.poll.return_value = None
        with patch("proxy.lifecycle.os.killpg") as mock_killpg:
            result = _kill_process_group(proc, MagicMock())
        mock_killpg.assert_called_once_with(12345, signal.SIGTERM)
        assert result is True


# ---------------------------------------------------------------------------
# Unit: check_backend_isolation
# ---------------------------------------------------------------------------


class TestCheckBackendIsolation:
    def test_unknown_when_no_backend(self):
        srv = _fake_srv(llama_proc=None, tts_proc=None)
        report = check_backend_isolation(srv)
        assert report["state"] == "unknown"
        assert report["violation"] is False

    def test_unknown_when_tts_absent(self):
        llama = _spawn_backend_tree()
        try:
            srv = _fake_srv(llama_proc=llama, tts_proc=None)
            report = check_backend_isolation(srv)
            assert report["state"] == "unknown"
            assert report["violation"] is False
        finally:
            _kill_tree(llama)

    @patch("proxy.backend_isolation.discover_llama_pids", return_value=[])
    def test_violation_when_shared_pid(self, _discover):
        llama = _spawn_backend_tree()
        try:
            # Same live pid registered for both backends: a kill of one
            # obviously reaches the other -- must be a violation.
            srv = _fake_srv(llama_proc=llama, tts_proc=MagicMock(pid=llama.pid))
            report = check_backend_isolation(srv)
            assert report["state"] == "violation"
            assert llama.pid in report["shared_pids"]
        finally:
            _kill_tree(llama)

    @patch("proxy.backend_isolation.discover_llama_pids", return_value=[])
    def test_violation_when_shared_pgid(self, _discover):
        """PIDs differ but both trees resolve to one PG => violation.

        A process that joins the other backend's process group (an inherited
        session, a stray ``setpgid``) puts one tree inside the kill range of the
        other. The verdict is derived from the resolved PGIDs, so the shared
        group is the violation regardless of how it arose.
        """
        llama = _spawn_backend_tree()
        tts = _spawn_backend_tree()
        try:
            with patch("proxy.backend_isolation._pgid_of", return_value=7777):
                srv = _fake_srv(llama_proc=llama, tts_proc=tts)
                report = check_backend_isolation(srv)
            assert report["state"] == "violation"
            assert report["shared_pgids"] == [7777]
            assert report["shared_pids"] == []
        finally:
            _kill_tree(llama)
            _kill_tree(tts)


# ---------------------------------------------------------------------------
# Unit: tracked_backend_pids
# ---------------------------------------------------------------------------


class TestTrackedBackendPids:
    def test_tracks_handle_pids(self):
        srv = _fake_srv(llama_proc=MagicMock(pid=100), tts_proc=MagicMock(pid=200))
        assert tracked_backend_pids(srv) == {"llama": [100], "tts": [200]}

    def test_skips_non_int_and_dead_handle_pids(self):
        llama_proc = _spawn_backend_tree()
        try:
            srv = _fake_srv(llama_proc=llama_proc, tts_proc=MagicMock(pid=MagicMock()))
            tracked = tracked_backend_pids(srv)
            assert llama_proc.pid in tracked["llama"]
            assert tracked["tts"] == []
        finally:
            _kill_tree(llama_proc)

    @patch("proxy.backend_isolation.discover_llama_pids", return_value=[300, 301])
    def test_adopted_llama_uses_port_discovery(self, _discover):
        """Adopted backends carry pid 0; the live tree is found by port."""
        adopted = MagicMock(pid=0, port=8080)
        srv = _fake_srv(llama_proc=adopted, tts_proc=MagicMock(pid=200))
        tracked = tracked_backend_pids(srv)
        assert tracked["llama"] == [300, 301]
        assert tracked["tts"] == [200]


# ---------------------------------------------------------------------------
# Unit: assert_backend_isolation (the watchdog guard)
# ---------------------------------------------------------------------------


class TestAssertBackendIsolation:
    @pytest.fixture(autouse=True)
    def _reset_throttle(self):
        """Determinism: the throttle is module-global; reset it per test."""
        import proxy.backend_isolation as bi

        bi._last_assert_at = 0.0
        yield

    def test_logs_error_on_violation(self):
        srv = MagicMock()
        with patch(
            "proxy.backend_isolation.check_backend_isolation",
            return_value={
                "state": "violation",
                "violation": True,
                "shared_pgids": [42],
                "shared_pids": [],
                "llama": {"pids": [1], "pgids": [42]},
                "tts": {"pids": [2], "pgids": [42]},
                "reason": "process group 42 shared",
            },
        ):
            logger = MagicMock()
            ok = assert_backend_isolation(srv, logger)
        assert ok is False
        logger.error.assert_called_once()
        logged = " ".join(str(a) for a in logger.error.call_args[0])
        assert "42" in logged

    def test_no_error_when_isolated(self):
        srv = MagicMock()
        with patch(
            "proxy.backend_isolation.check_backend_isolation",
            return_value={
                "state": "isolated",
                "violation": False,
                "shared_pgids": [],
                "shared_pids": [],
                "llama": {"pids": [1], "pgids": [11]},
                "tts": {"pids": [2], "pgids": [22]},
                "reason": None,
            },
        ):
            logger = MagicMock()
            ok = assert_backend_isolation(srv, logger)
        assert ok is True
        logger.error.assert_not_called()

    def test_fail_open_on_unexpected_error(self):
        srv = MagicMock()
        with patch(
            "proxy.backend_isolation.check_backend_isolation",
            side_effect=RuntimeError("boom"),
        ):
            logger = MagicMock()
            ok = assert_backend_isolation(srv, logger)
        assert ok is True
