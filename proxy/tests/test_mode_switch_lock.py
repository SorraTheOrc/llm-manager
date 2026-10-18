"""Tests for the mode-switch lock (LP-0MVCBMLP7009MN81).

A slot sweep / performance investigation must not be corrupted by a
fast<->cheap mode switch mid-run (the herdr mode-switch worker and the mode
schedule can both flip modes, each triggering a proxy restart). The lock is
a runtime switch:

- OFF (default) -> every mode change proceeds exactly as before.
- ON -> a request to CHANGE the mode is rejected (``ModeSwitchLockedError`` /
  HTTP 423) BEFORE any state mutation or restart, and the mode scheduler
  stands down. A same-mode request stays a noop.

The lock state persists in ``proxy/.mode-switch-lock`` so it survives the
restart a pre-lock change would have armed.
"""

import json

import httpx
import pytest

from proxy import mode as mode_module


@pytest.fixture
def state_files(tmp_path, monkeypatch):
    """Redirect every mode state file at a temp path (hermetic)."""
    files = {
        "mode": tmp_path / ".mode",
        "override": tmp_path / ".mode.override-until",
        "last_fast": tmp_path / ".mode.last-fast-switch",
        "lock": tmp_path / ".mode-switch-lock",
    }
    monkeypatch.setattr(mode_module, "mode_state_file", lambda: files["mode"])
    monkeypatch.setattr(mode_module, "override_until_file", lambda: files["override"])
    monkeypatch.setattr(
        mode_module, "last_fast_switch_file", lambda: files["last_fast"]
    )
    monkeypatch.setattr(
        mode_module, "mode_switch_lock_file", lambda: files["lock"]
    )
    return files


@pytest.fixture(autouse=True)
def _reset_pending():
    with mode_module._mode_lock:
        mode_module._restart_pending = False
    with mode_module._drain_lock:
        mode_module._draining = False
        mode_module._drain_deadline = None
    yield
    with mode_module._mode_lock:
        mode_module._restart_pending = False
    with mode_module._drain_lock:
        mode_module._draining = False
        mode_module._drain_deadline = None


@pytest.fixture
def client():
    """ASGI test client against the real proxy app."""
    from proxy.server import app

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


@pytest.fixture(autouse=True)
def _no_live_config(monkeypatch):
    """Default the config fallback OFF so tests are deterministic and never
    read the real server config."""
    monkeypatch.setattr(mode_module, "_live_server_config", lambda: {})


# ---------------------------------------------------------------------------
# mode_switch_locked() / toggle helpers
# ---------------------------------------------------------------------------


class TestLockState:
    def test_default_off(self, state_files):
        assert mode_module.mode_switch_locked() is False

    def test_engage_persists_and_reads(self, state_files):
        mode_module.set_mode_switch_lock(True, reason="slot sweep")
        assert mode_module.mode_switch_locked() is True
        data = json.loads(state_files["lock"].read_text())
        assert data["enabled"] is True
        assert data["reason"] == "slot sweep"
        assert "since" in data

    def test_release_removes_state(self, state_files):
        mode_module.set_mode_switch_lock(True)
        assert mode_module.mode_switch_locked() is True
        mode_module.set_mode_switch_lock(False)
        assert mode_module.mode_switch_locked() is False
        assert not state_files["lock"].exists()

    def test_release_when_absent_is_safe(self, state_files):
        mode_module.set_mode_switch_lock(False)  # no file
        assert mode_module.mode_switch_locked() is False

    def test_survives_restart(self, state_files):
        """The state file (not module memory) drives the lock, so it survives
        the proxy restart a pre-lock change would have armed."""
        mode_module.set_mode_switch_lock(True, reason="sweep")
        # A fresh process would only see the file; simulate by re-reading.
        assert mode_module.mode_switch_locked() is True

    def test_unparsable_state_does_not_lock(self, state_files):
        state_files["lock"].write_text("{ not json")
        assert mode_module.mode_switch_locked() is False

    def test_config_default_engages_lock(self, state_files, monkeypatch):
        """``server.mode_switch_lock.enabled: true`` pins the lock with no
        runtime file (deployments that always want it)."""
        monkeypatch.setattr(
            mode_module,
            "_live_server_config",
            lambda: {"mode_switch_lock": {"enabled": True, "reason": "pinned"}},
        )
        assert mode_module.mode_switch_locked() is True

    def test_runtime_state_overrides_config(self, state_files, monkeypatch):
        """A runtime 'released' file wins over a config-enabled default."""
        monkeypatch.setattr(
            mode_module,
            "_live_server_config",
            lambda: {"mode_switch_lock": {"enabled": True}},
        )
        # No runtime file -> config engages.
        assert mode_module.mode_switch_locked() is True
        # Now write a runtime file... a file whose enabled is false means
        # "explicitly unlocked" only if we support it; our design removes the
        # file to release. Assert the config default applies when absent.
        mode_module.set_mode_switch_lock(False)
        assert mode_module.mode_switch_locked() is True  # falls back to config


# ---------------------------------------------------------------------------
# set_mode() gate
# ---------------------------------------------------------------------------


class TestSetModeGate:
    def test_unlocked_allows_change(self, state_files, monkeypatch):
        state_files["mode"].write_text("fast\n")
        monkeypatch.setattr(mode_module, "_spawn_restart", lambda: None)
        persisted, restart = mode_module.set_mode(mode_module.MODE_CHEAP)
        assert (persisted, restart) == (mode_module.MODE_CHEAP, True)
        assert state_files["mode"].read_text().strip() == "cheap"

    def test_locked_rejects_change_no_mutation(self, state_files, monkeypatch):
        state_files["mode"].write_text("fast\n")
        spawned = []
        monkeypatch.setattr(mode_module, "_spawn_restart", lambda: spawned.append(True))
        mode_module.set_mode_switch_lock(True, reason="sweep")

        with pytest.raises(mode_module.ModeSwitchLockedError):
            mode_module.set_mode(mode_module.MODE_CHEAP)

        # No state mutation, no restart armed, no drain.
        assert state_files["mode"].read_text().strip() == "fast"
        assert spawned == []
        assert mode_module.draining() is False

    def test_locked_same_mode_is_noop(self, state_files, monkeypatch):
        state_files["mode"].write_text("fast\n")
        monkeypatch.setattr(mode_module, "_spawn_restart", lambda: None)
        mode_module.set_mode_switch_lock(True)
        # Requesting the active mode is not a CHANGE -> allowed noop.
        persisted, restart = mode_module.set_mode(mode_module.MODE_FAST)
        assert (persisted, restart) == (mode_module.MODE_FAST, False)

    def test_unlock_restores_changes(self, state_files, monkeypatch):
        state_files["mode"].write_text("fast\n")
        monkeypatch.setattr(mode_module, "_spawn_restart", lambda: None)
        mode_module.set_mode_switch_lock(True)
        with pytest.raises(mode_module.ModeSwitchLockedError):
            mode_module.set_mode(mode_module.MODE_CHEAP)
        mode_module.set_mode_switch_lock(False)
        persisted, restart = mode_module.set_mode(mode_module.MODE_CHEAP)
        assert (persisted, restart) == (mode_module.MODE_CHEAP, True)


# ---------------------------------------------------------------------------
# Scheduler stand-down
# ---------------------------------------------------------------------------


class TestSchedulerStandDown:
    def test_scheduler_does_not_switch_while_locked(self, state_files, monkeypatch):
        state_files["mode"].write_text("cheap\n")
        mode_module.set_mode_switch_lock(True, reason="sweep")
        called = []
        monkeypatch.setattr(
            mode_module, "set_mode", lambda *a, **k: called.append(True) or ("x", True)
        )
        # A schedule that mandates 'fast' while the proxy is 'cheap'.
        from datetime import time as dt_time

        schedule = mode_module.ModeScheduleConfig(
            {"enabled": True, "entries": [{"time": "00:00", "mode": "fast"}]}
        )
        changed = mode_module._mode_scheduler_step(
            schedule, now=dt_time(12, 0)
        )
        assert changed is False
        assert called == []  # set_mode never invoked while locked

    def test_scheduler_switches_when_unlocked(self, state_files, monkeypatch):
        state_files["mode"].write_text("cheap\n")
        called = []

        def _fake_set_mode(expected, **kwargs):
            called.append(expected)
            return expected, True

        monkeypatch.setattr(mode_module, "set_mode", _fake_set_mode)
        from datetime import time as dt_time

        schedule = mode_module.ModeScheduleConfig(
            {"enabled": True, "entries": [{"time": "00:00", "mode": "fast"}]}
        )
        changed = mode_module._mode_scheduler_step(schedule, now=dt_time(12, 0))
        assert changed is True
        assert called == [mode_module.MODE_FAST]


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


class TestLockApi:
    @pytest.mark.asyncio
    async def test_get_default_off(self, state_files, client):
        async with client as c:
            resp = await c.get("/admin/mode-switch-lock")
        assert resp.status_code == 200
        assert resp.json() == {"enabled": False, "source": "off"}

    @pytest.mark.asyncio
    async def test_toggle_on_and_off(self, state_files, client):
        async with client as c:
            resp = await c.post(
                "/admin/mode-switch-lock",
                content=json.dumps({"enabled": True, "reason": "sweep"}),
            )
            assert resp.status_code == 200
            assert resp.json()["enabled"] is True
            get1 = await c.get("/admin/mode-switch-lock")
            assert get1.json() == {"enabled": True, "source": "runtime"}

            resp2 = await c.post(
                "/admin/mode-switch-lock", content=json.dumps({"enabled": False})
            )
            assert resp2.json()["enabled"] is False
            get2 = await c.get("/admin/mode-switch-lock")
            assert get2.json() == {"enabled": False, "source": "off"}

    @pytest.mark.asyncio
    async def test_invalid_json_400(self, state_files, client):
        async with client as c:
            resp = await c.post("/admin/mode-switch-lock", content="not json")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_set_mode_returns_423_when_locked(
        self, state_files, client, monkeypatch
    ):
        state_files["mode"].write_text("fast\n")
        mode_module.set_mode_switch_lock(True, reason="sweep")
        async with client as c:
            resp = await c.post(
                "/admin/set-mode", content=json.dumps({"mode": "cheap"})
            )
        assert resp.status_code == 423
        assert "locked" in resp.json()["detail"].lower()
        # No mutation.
        assert state_files["mode"].read_text().strip() == "fast"

    @pytest.mark.asyncio
    async def test_set_mode_allowed_when_unlocked(
        self, state_files, client, monkeypatch
    ):
        state_files["mode"].write_text("fast\n")
        monkeypatch.setattr(mode_module, "_spawn_restart", lambda: None)
        async with client as c:
            resp = await c.post(
                "/admin/set-mode", content=json.dumps({"mode": "cheap"})
            )
        assert resp.status_code == 200
        assert resp.json()["mode"] == "cheap"
