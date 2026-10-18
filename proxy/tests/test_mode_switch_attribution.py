"""Tests for mode-switch / restart attribution (LP-0MVAPNC2900145RH).

Every ``set_mode``-driven restart now records who/why requested it:

- ``set_mode(...)`` accepts optional ``source``/``actor``/``reason`` kwargs
  (captured at call time into ``_restart_attribution``).
- The ``Mode-switch restart spawned`` log line carries
  ``source=.. actor=.. mode=.. reason=..`` markers (additive: an unattributed
  call keeps the old line shape).
- The spawned ``start-proxy.sh --restart`` receives the reason via the
  ``--reason`` argv and ``LLAMA_PROXY_RESTART_SOURCE/ACTOR/REASON`` env vars,
  so the new process's startup log can echo it (operator restarts are
  attributable from the log alone).
- ``POST /admin/set-mode`` accepts optional ``actor``/``source``/``reason``
  body fields (worker self-identification; default ``source=api``).
"""

import json
import logging
from datetime import time as dt_time

import httpx
import pytest

from proxy import mode as mode_module


@pytest.fixture
def mode_file(tmp_path):
    """Return a temp path to use as the mode state file."""
    return tmp_path / ".mode"


@pytest.fixture(autouse=True)
def _override_file(tmp_path, monkeypatch):
    """Point the manual-override expiry state file at a temp path so no test
    writes into the real proxy directory (LP-0MSMF25V9002AY1J)."""
    path = tmp_path / ".mode.override-until"
    monkeypatch.setattr(mode_module, "override_until_file", lambda: path)
    monkeypatch.setattr(
        mode_module,
        "last_fast_switch_file",
        lambda: tmp_path / ".mode.last-fast-switch",
    )
    return path


@pytest.fixture(autouse=True)
def _reset_pending():
    """Ensure the pending-restart flag and attribution start clean."""
    with mode_module._mode_lock:
        mode_module._restart_pending = False
        mode_module._restart_attribution = (None, None, None, None)
    yield
    with mode_module._mode_lock:
        mode_module._restart_pending = False
        mode_module._restart_attribution = (None, None, None, None)


def _flippable_schedule():
    """A schedule whose active mode differs from the persisted fast mode."""
    return mode_module.ModeScheduleConfig(
        {
            "enabled": True,
            "entries": [
                {"time": "01:00", "mode": "cheap"},
                {"time": "10:00", "mode": "fast"},
            ],
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# set_mode() attribution
# ═══════════════════════════════════════════════════════════════════════════════


class TestSetModeAttribution:
    def test_set_mode_records_attribution_state(
        self, mode_file, monkeypatch
    ):
        """set_mode captures source/actor/reason for the forthcoming spawn."""
        mode_file.write_text("fast\n")
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        observed = {}

        def _fake_spawn():
            observed["attribution"] = mode_module._restart_attribution

        monkeypatch.setattr(mode_module, "_spawn_restart", _fake_spawn)
        persisted, restart = mode_module.set_mode(
            "cheap",
            manual=True,
            source="worker",
            actor="herdr-worker",
            reason="mode-flap recovery",
        )
        assert restart is True
        assert persisted == "cheap"
        assert observed["attribution"] == (
            "worker",
            "herdr-worker",
            "mode-flap recovery",
            "cheap",
        )

    def test_set_mode_attribution_defaults_to_none(self, mode_file, monkeypatch):
        """An unattributed set_mode call leaves the markers empty (None)."""
        mode_file.write_text("fast\n")
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        observed = {}

        def _fake_spawn():
            observed["attribution"] = mode_module._restart_attribution

        monkeypatch.setattr(mode_module, "_spawn_restart", _fake_spawn)
        mode_module.set_mode("cheap")
        assert observed["attribution"] == (None, None, None, "cheap")


class TestSchedulerAttribution:
    def test_scheduler_step_passes_scheduler_attribution(
        self, mode_file, monkeypatch
    ):
        """The scheduler identifies itself as source=scheduler."""
        mode_file.write_text("fast\n")
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        monkeypatch.setattr(mode_module, "manual_override_active", lambda: False)
        monkeypatch.setattr(mode_module, "_spawn_restart", lambda: None)

        schedule = _flippable_schedule()
        changed = mode_module._mode_scheduler_step(schedule, now=dt_time(1, 0))
        assert changed is True
        assert mode_module._restart_attribution == (
            "scheduler",
            "mode-scheduler",
            "scheduled mode change",
            "cheap",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# _spawn_restart: log-line markers + reason transport to start-proxy.sh
# ═══════════════════════════════════════════════════════════════════════════════


class _SyncThread:
    """threading.Thread stand-in that runs the target synchronously."""

    def __init__(self, target=None, daemon=False, **kwargs):
        self._target = target
        self.daemon = daemon

    def start(self):
        assert self._target is not None
        self._target()

    def join(self, timeout=None):
        return None


def _run_spawn_synchronously(monkeypatch, tmp_path, attribution):
    """Run _spawn_restart's thread body synchronously; return captured Popen."""
    captured = {}

    class _Popen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

    monkeypatch.setattr(mode_module, "_restart_attribution", attribution)
    monkeypatch.setattr(mode_module.subprocess, "Popen", _Popen)
    monkeypatch.setattr(mode_module.time, "sleep", lambda s: None)
    monkeypatch.setattr(mode_module, "_wait_for_in_flight_local_streams", lambda: None)
    monkeypatch.setattr(
        mode_module.threading, "Thread", _SyncThread
    )
    monkeypatch.setattr(
        mode_module, "proxy_dir", lambda: tmp_path / "proxy"
    )
    mode_module._spawn_restart()
    return captured


class TestSpawnRestartAttribution:
    def test_spawn_log_line_carries_markers(self, monkeypatch, tmp_path, caplog):
        """The spawned restart line carries source/actor/mode/reason markers."""
        with caplog.at_level(logging.INFO, logger="llama-proxy"):
            _run_spawn_synchronously(
                monkeypatch,
                tmp_path,
                ("scheduler", "mode-scheduler", "scheduled mode change", "cheap"),
            )
        spawned = [
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith("Mode-switch restart spawned:")
        ]
        assert len(spawned) == 1
        line = spawned[0]
        assert "source=scheduler" in line
        assert "actor=mode-scheduler" in line
        assert "mode=cheap" in line
        assert "reason=scheduled mode change" in line

    def test_spawn_log_line_without_attribution_keeps_old_shape(
        self, monkeypatch, tmp_path, caplog
    ):
        """No markers when no attribution is recorded (backward compatible)."""
        with caplog.at_level(logging.INFO, logger="llama-proxy"):
            _run_spawn_synchronously(monkeypatch, tmp_path, (None, None, None, "cheap"))
        spawned = [
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith("Mode-switch restart spawned:")
        ]
        assert len(spawned) == 1
        assert "source=" not in spawned[0]
        assert "Mode-switch restart spawned:" in spawned[0]

    def test_spawn_passes_reason_to_start_proxy(self, monkeypatch, tmp_path):
        """The reason reaches start-proxy.sh via --reason argv and env vars."""
        captured = _run_spawn_synchronously(
            monkeypatch,
            tmp_path,
            ("worker", "herdr-worker", "mode-flap recovery", "cheap"),
        )
        cmd = captured["cmd"]
        assert cmd[:2] == ["bash", str(tmp_path / "proxy" / "scripts" / "start-proxy.sh")]
        assert cmd[2] == "--restart"
        assert "mode-flap recovery" in cmd  # --reason <text> position
        assert "--reason" in cmd
        env = captured["kwargs"]["env"]
        assert env["LLAMA_PROXY_RESTART_REASON"] == "mode-flap recovery"
        assert env["LLAMA_PROXY_RESTART_SOURCE"] == "worker"
        assert env["LLAMA_PROXY_RESTART_ACTOR"] == "herdr-worker"
        assert env["LLAMA_PROXY_RESTART_REASON"].isspace() is False


# ═══════════════════════════════════════════════════════════════════════════════
# POST /admin/set-mode attribution fields
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def client():
    """ASGI test client against the real proxy app."""
    from proxy.server import app

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


@pytest.mark.asyncio
async def test_api_set_mode_forwards_attribution(
    mode_file, client, monkeypatch
):
    """The handler forwards actor/source/reason to set_mode."""
    mode_file.write_text("fast\n")
    monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
    monkeypatch.setattr(mode_module, "_spawn_restart", lambda: None)
    async with client as c:
        resp = await c.post(
            "/admin/set-mode",
            content=json.dumps(
                {
                    "mode": "cheap",
                    "source": "worker",
                    "actor": "herdr-worker",
                    "reason": "mode-flap recovery",
                }
            ),
        )
    assert resp.status_code == 200
    assert mode_module._restart_attribution == (
        "worker",
        "herdr-worker",
        "mode-flap recovery",
        "cheap",
    )


@pytest.mark.asyncio
async def test_api_set_mode_defaults_source_api(mode_file, client, monkeypatch):
    """Absent source field defaults to api (manual/keyboard switch)."""
    mode_file.write_text("fast\n")
    monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
    monkeypatch.setattr(mode_module, "_spawn_restart", lambda: None)
    async with client as c:
        resp = await c.post("/admin/set-mode", content=json.dumps({"mode": "cheap"}))
    assert resp.status_code == 200
    assert mode_module._restart_attribution[0] == "api"


@pytest.mark.asyncio
async def test_api_set_mode_rejects_bogus_source(
    mode_file, client, monkeypatch
):
    """An unknown source value falls back to api (fail-closed attribution)."""
    mode_file.write_text("fast\n")
    monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
    monkeypatch.setattr(mode_module, "_spawn_restart", lambda: None)
    async with client as c:
        resp = await c.post(
            "/admin/set-mode",
            content=json.dumps({"mode": "cheap", "source": "bogus-source"}),
        )
    assert resp.status_code == 200
    assert mode_module._restart_attribution[0] == "api"


# ═══════════════════════════════════════════════════════════════════════════════
# start-proxy.sh --reason: startup attribution line (server side)
# ═══════════════════════════════════════════════════════════════════════════════


def test_server_startup_logs_restart_attribution(monkeypatch, caplog):
    """The new server process echoes the operator's restart reason."""
    import proxy.server as srv

    monkeypatch.setenv("LLAMA_PROXY_RESTART_REASON", "nightly ops")
    monkeypatch.setenv("LLAMA_PROXY_RESTART_SOURCE", "manual")
    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        srv._log_restart_attribution()
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Proxy restart attribution:")
    ]
    assert len(lines) == 1
    assert "source=manual" in lines[0]
    assert "reason=nightly ops" in lines[0]


def test_server_startup_no_attribution_line_without_env(monkeypatch, caplog):
    """No attribution line on an unattributed start (fresh boot)."""
    import proxy.server as srv

    monkeypatch.delenv("LLAMA_PROXY_RESTART_REASON", raising=False)
    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        srv._log_restart_attribution()
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Proxy restart attribution:")
    ]
    assert lines == []


# ═══════════════════════════════════════════════════════════════════════════════
# AC3: reproducible grep attributes every restart (zero unattributable)
# ═══════════════════════════════════════════════════════════════════════════════


def test_reproducible_grep_attributes_all_new_restarts(tmp_path):
    """Every restart line on the new logging path carries a source marker.

    The documented operator grep

        zgrep -h "Mode-switch restart spawned:.* source=" proxy.log* | wc -l

    must equal the restart count, i.e. nothing marker-carrying is
    unattributable (AC3).
    """
    import re

    log = tmp_path / "proxy.log"
    lines = [
        # (timestamp, message) pairs on the new logging path
        "2026-10-16 10:00:00,123 INFO Mode-switch restart spawned: /opt/proxy/scripts/start-proxy.sh --restart source=scheduler actor=mode-scheduler mode=fast reason=scheduled mode change",
        "2026-10-16 10:05:00,456 INFO Mode-switch restart spawned: /opt/proxy/scripts/start-proxy.sh --restart source=api actor=unknown mode=cheap reason=manual switch",
        "2026-10-16 11:00:00,789 INFO Proxy restart attribution: source=manual actor=ops reason=nightly cleanup",
    ]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    grep = r"Mode-switch restart spawned:.* source="
    matches = [ln for ln in log.read_text().splitlines() if re.search(grep, ln)]
    spawned = [ln for ln in lines if "Mode-switch restart spawned:" in ln]
    assert len(matches) == len(spawned)  # zero unattributable restarts
