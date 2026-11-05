"""The backend's ``oom_score_adj`` must reach llama-server, not just its wrapper.

Measured live on 2026-11-05 during the LP-0MW3K7WYV001XLL1 attribution run::

    $ for p in $(pgrep -x llama-server); do echo "$p $(cat /proc/$p/oom_score_adj)"; done
    2089209 0     # router parent
    2089312 0     # its per-model child

``server.llama_server_oom_score_adj: 800`` and
:func:`proxy.host_pressure.protect_backend_process` both existed, yet every
``llama-server`` still scored ``0``. The proxy spawns ``start-llama.sh``, which
runs ``llama-server`` as a foreground **child** — so the score was written to the
bash wrapper *after* the backend had already forked, and the kernel copies
``oom_score_adj`` into a child at fork time only. The router's own children
(one per loaded model) inherit whatever the router has, so the whole backend
tree stayed unranked.

These tests use real processes and the real ``/proc`` (raising a score is always
permitted, so no privilege is needed); a live global OOM is never provoked.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import proxy.host_pressure as hp
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_oom_protection.py"


def _wait_for(predicate, timeout: float = 15.0):
    """Poll *predicate* until it returns a truthy value (real processes settle)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(f"condition not met within {timeout}s")


def _spawn(script: str) -> subprocess.Popen:
    """A bash script in its own session, so cleanup cannot signal the test run."""
    return subprocess.Popen(["bash", "-c", script], start_new_session=True)


def _stop(proc: subprocess.Popen) -> None:
    """Kill this specific process group — never a broad process-name kill."""
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive
        pass


@pytest.fixture(autouse=True)
def _reset_warn_state():
    """Each test starts with an empty ``warn once`` memory."""
    hp._REPORTED.clear()
    yield
    hp._REPORTED.clear()


# ---------------------------------------------------------------------------
# Process-tree resolution
# ---------------------------------------------------------------------------


def test_process_tree_lists_roots_then_descendants():
    """The tree walk finds the backend a start script forked, not just the root."""
    proc = _spawn('sleep 60 & sleep 60 & wait')
    try:
        tree = _wait_for(
            lambda: (lambda pids: pids if len(pids) >= 3 else None)(
                hp.process_tree([proc.pid])
            )
        )
        assert tree[0] == proc.pid
        assert proc.pid in tree
        # Both background sleeps are descendants of the root.
        assert len(tree) >= 3
        assert all(pid > 0 for pid in tree)
        # No duplicates, and the terminated shell's own children are real.
        assert len(tree) == len(set(tree))
    finally:
        _stop(proc)


def test_process_table_reads_parent_and_command_line_of_a_real_process():
    proc = _spawn("sleep 60")
    try:
        table = _wait_for(lambda: hp.read_process_table() or None)
        entry = table[proc.pid]
        assert entry.pid == proc.pid
        assert entry.ppid == os.getpid()
        assert "sleep" in entry.cmdline
        assert entry.comm
        assert not entry.is_llama_server
    finally:
        _stop(proc)


def test_parse_stat_accepts_a_comm_containing_spaces_and_parentheses():
    """The kernel does not escape ``comm``, so ``)`` alone cannot delimit it."""
    parsed = hp._parse_stat("4242 (llama-server worker) 1) S 17 4242 4242 0 -1\n")
    assert parsed == ("llama-server worker) 1", 17)
    assert hp._parse_stat("garbage") is None


def test_backend_pids_matches_the_command_line_of_a_renamed_process():
    proc = _spawn('(exec -a llama-server sleep 60) & wait')
    try:
        pids = _wait_for(lambda: hp.backend_pids([proc.pid]) or None)
        assert pids, "the llama-server-named child must be discovered"
        assert all(pid != proc.pid for pid in pids)
    finally:
        _stop(proc)


# ---------------------------------------------------------------------------
# The live failure: a child forked before the write stayed at 0
# ---------------------------------------------------------------------------


def test_tree_walk_scores_a_backend_child_that_forked_before_the_write():
    """Regression: the forked backend was left at 0 while the wrapper got 800.

    ``start-llama.sh`` forks ``llama-server`` within milliseconds of the
    ``Popen``, long before the proxy writes to ``/proc``. The score must
    therefore be written to the backend pid the tree walk finds, not only to the
    wrapper.
    """
    proc = _spawn('(exec -a llama-server sleep 60) & wait')
    try:
        child = _wait_for(lambda: (hp.backend_pids([proc.pid]) or [None])[0])
        assert hp.read_oom_score_adj(child) == 0, "precondition: unprotected child"

        applied = hp.protect_backend_process(proc, {"server": {}}, None)

        assert applied == hp.LLAMA_SERVER_OOM_SCORE_ADJ
        assert hp.read_oom_score_adj(proc.pid) == hp.LLAMA_SERVER_OOM_SCORE_ADJ
        assert hp.read_oom_score_adj(child) == hp.LLAMA_SERVER_OOM_SCORE_ADJ
    finally:
        _stop(proc)


def test_a_child_forked_after_the_write_inherits_the_score():
    """The wrapper's score is what a later-forked router child inherits."""
    proc = _spawn('sleep 0.7; (exec -a llama-server sleep 60) & wait')
    try:
        # Score the wrapper before the backend forks (as the spawn hook does).
        assert hp.protect_backend_process(proc, {"server": {}}, None) == (
            hp.LLAMA_SERVER_OOM_SCORE_ADJ
        )
        child = _wait_for(lambda: (hp.backend_pids([proc.pid]) or [None])[0])
        assert hp.read_oom_score_adj(child) == hp.LLAMA_SERVER_OOM_SCORE_ADJ
    finally:
        _stop(proc)


# ---------------------------------------------------------------------------
# Re-apply path and gap recording (AC2/AC3)
# ---------------------------------------------------------------------------


def test_reapply_protects_the_whole_tree_and_reports_no_gap():
    proc = _spawn('(exec -a llama-server sleep 60) & wait')
    try:
        child = _wait_for(lambda: (hp.backend_pids([proc.pid]) or [None])[0])

        report = hp.reapply_backend_oom_protection([proc.pid], {"server": {}}, None)

        assert report.protected
        assert report.gap is None
        assert report.configured == hp.LLAMA_SERVER_OOM_SCORE_ADJ
        assert {row.pid for row in report.rows} >= {proc.pid, child}
        scored = {row.pid: row.applied for row in report.rows}
        assert scored[proc.pid] == hp.LLAMA_SERVER_OOM_SCORE_ADJ
        assert scored[child] == hp.LLAMA_SERVER_OOM_SCORE_ADJ
        assert [row.pid for row in report.rows if row.is_backend] == [child]
    finally:
        _stop(proc)


def test_reapply_records_the_gap_with_a_reason_and_an_operator_action(caplog):
    """No backend in the tree is a *documented gap*, never a silent pass."""
    proc = _spawn("sleep 60")
    logger = logging.getLogger("test-backend-oom-gap")
    try:
        with caplog.at_level(logging.WARNING, logger="test-backend-oom-gap"):
            report = hp.reapply_backend_oom_protection(
                [proc.pid], {"server": {}}, logger, require_backend=True
            )
        assert not report.protected
        assert report.gap is not None
        assert "no llama-server process" in report.gap
        message = " ".join(record.getMessage() for record in caplog.records)
        assert "verify_oom_protection.py --apply" in message
        assert "oom-recovery-runbook" in message
        # A repeated reconcile does not spam the log.
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="test-backend-oom-gap"):
            hp.reapply_backend_oom_protection(
                [proc.pid], {"server": {}}, logger, require_backend=True
            )
        assert caplog.records == []
    finally:
        _stop(proc)


def test_inspect_reads_the_live_values_without_writing_them():
    proc = _spawn('(exec -a llama-server sleep 60) & wait')
    try:
        child = _wait_for(lambda: (hp.backend_pids([proc.pid]) or [None])[0])

        report = hp.inspect_backend_oom_protection([proc.pid], {"server": {}})

        assert not report.protected
        assert report.gap is not None
        values = {row.pid: row.applied for row in report.rows}
        assert values[proc.pid] == 0
        assert values[child] == 0
        assert hp.read_oom_score_adj(child) == 0, "inspect must not write"
    finally:
        _stop(proc)


# ---------------------------------------------------------------------------
# Spawn path: the write must happen before the quick-exit wait
# ---------------------------------------------------------------------------


def test_spawn_and_capture_runs_on_spawn_before_the_quick_exit_wait():
    """The hook fires immediately after ``Popen``, not after the 3 s probe.

    ``start-llama.sh`` forks ``llama-server`` during that probe window, and a
    score written afterwards is never inherited — hence the ordering contract.
    """
    import proxy.lifecycle as lifecycle

    seen: dict[str, float] = {}

    def _on_spawn(proc):
        seen["pid"] = float(proc.pid)
        seen["at"] = time.monotonic()

    started = time.monotonic()
    proc, captured = lifecycle.spawn_and_capture(
        ["sleep", "60"],
        env=os.environ.copy(),
        log_file=subprocess.DEVNULL,
        logger=logging.getLogger("test-spawn-on-spawn"),
        on_spawn=_on_spawn,
    )
    try:
        assert proc is not None
        assert captured is None
        assert seen["pid"] == float(proc.pid)
        # The legacy quick-exit probe waits 3 s; the hook must precede it.
        assert seen["at"] - started < 1.5
    finally:
        _stop(proc)


def test_start_llama_server_passes_the_oom_hook_into_the_spawner(monkeypatch):
    """The spawn path wires ``_apply_backend_oom_protection`` as ``on_spawn``."""
    import proxy.lifecycle as lifecycle

    captured: dict[str, object] = {}

    class _Srv:
        config = {"server": {"llama_start_script": "/bin/true"}}
        logger = logging.getLogger("test-start-llama-server")
        log_dir = None
        last_start_failure = None
        llama_log_file = None
        llama_log_source_unhealthy = False

        def broadcast_status_sync(self, *_a, **_k):
            pass

        def rotate_llama_logs(self, *_a, **_k):
            return True

    monkeypatch.setattr(lifecycle, "_srv", lambda: _Srv())

    def _fake_spawn(cmd, env, log_file, logger, model_name="unknown", on_spawn=None):
        captured["on_spawn"] = on_spawn
        captured["cmd"] = cmd
        return None, "spawn refused"

    monkeypatch.setattr(lifecycle, "spawn_and_capture", _fake_spawn)
    monkeypatch.setattr(lifecycle, "_write_llama_server_signature", lambda *_a, **_k: None)
    monkeypatch.setattr(lifecycle, "start_llama_log_tailer", lambda *_a, **_k: None)

    lifecycle.start_llama_server("Qwen3")

    assert captured["on_spawn"] is lifecycle._apply_backend_oom_protection


# ---------------------------------------------------------------------------
# Lifecycle reconcile helper (the watchdog's periodic re-apply)
# ---------------------------------------------------------------------------


def test_reapply_helper_collects_the_tracked_pid_and_scores_it(monkeypatch):
    import proxy.backend_rss as backend_rss
    import proxy.lifecycle as lifecycle

    proc = _spawn("sleep 60")
    try:
        class _Srv:
            config = {"server": {"llama_server_port": 8080}}
            logger = logging.getLogger("test-lifecycle-reapply")
            backend_ready = True
            llama_process = type("_P", (), {"pid": proc.pid})()

        monkeypatch.setattr(lifecycle, "_srv", lambda: _Srv())
        monkeypatch.setattr(backend_rss, "discover_llama_pids", lambda port=None: [])
        monkeypatch.setattr(lifecycle, "_backend_oom_reconcile_at", 0.0)

        report = lifecycle._reapply_backend_oom_protection(force=True)

        assert report is not None
        assert hp.read_oom_score_adj(proc.pid) == hp.LLAMA_SERVER_OOM_SCORE_ADJ
        assert report["configured"] == hp.LLAMA_SERVER_OOM_SCORE_ADJ
        assert proc.pid in [row["pid"] for row in report["rows"]]
        # The throttle suppresses an immediate repeat.
        assert lifecycle._reapply_backend_oom_protection() is None
    finally:
        _stop(proc)


def test_reapply_helper_never_raises_when_the_backend_lookup_fails(monkeypatch):
    import proxy.lifecycle as lifecycle

    def _boom():
        raise RuntimeError("no server module")

    monkeypatch.setattr(lifecycle, "_srv", _boom)
    monkeypatch.setattr(lifecycle, "_backend_oom_reconcile_at", 0.0)

    assert lifecycle._reapply_backend_oom_protection(force=True) is None


@pytest.mark.asyncio
async def test_backend_watchdog_runs_the_reconcile_on_every_pass(monkeypatch):
    """The watchdog is the periodic re-apply: a router child may appear later."""
    import asyncio

    import proxy.lifecycle as lifecycle
    import proxy.server as server

    calls: list[int] = []
    monkeypatch.setattr(
        lifecycle, "_reapply_backend_oom_protection", lambda *a, **k: calls.append(1)
    )
    monkeypatch.setattr(server, "llama_process", None)
    monkeypatch.setattr(server, "backend_ready", False)
    monkeypatch.setattr(
        server,
        "config",
        {"server": {"llama_router_mode": False, "llama_watchdog_interval_seconds": 0}},
    )

    sleeps = {"n": 0}

    async def fake_sleep(_delay):
        sleeps["n"] += 1
        if sleeps["n"] > 1:
            raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await server._backend_watchdog_loop()

    assert calls == [1]


# ---------------------------------------------------------------------------
# Operator tool (runbook, AC4)
# ---------------------------------------------------------------------------


def test_verify_script_reports_raw_proc_values_and_exits_non_zero(tmp_path):
    proc = _spawn('(exec -a llama-server sleep 60) & wait')
    try:
        child = _wait_for(lambda: (hp.backend_pids([proc.pid]) or [None])[0])

        result = subprocess.run(
            [sys.executable, str(VERIFY_SCRIPT), "--pid", str(proc.pid), "--json"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        payload = json.loads(result.stdout)
        assert payload["configured"] == hp.LLAMA_SERVER_OOM_SCORE_ADJ
        assert payload["protected"] is False
        values = {row["pid"]: row["applied"] for row in payload["rows"]}
        assert values[proc.pid] == 0
        assert values[child] == 0
        assert payload["gap"]
        assert result.returncode == 1
    finally:
        _stop(proc)


def test_verify_script_applies_the_score_on_request():
    proc = _spawn('(exec -a llama-server sleep 60) & wait')
    try:
        child = _wait_for(lambda: (hp.backend_pids([proc.pid]) or [None])[0])

        result = subprocess.run(
            [sys.executable, str(VERIFY_SCRIPT), "--pid", str(proc.pid), "--apply", "--json"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        payload = json.loads(result.stdout)
        assert payload["protected"] is True
        assert result.returncode == 0
        assert hp.read_oom_score_adj(proc.pid) == hp.LLAMA_SERVER_OOM_SCORE_ADJ
        assert hp.read_oom_score_adj(child) == hp.LLAMA_SERVER_OOM_SCORE_ADJ
    finally:
        _stop(proc)
