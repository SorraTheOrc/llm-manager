"""Tests for the operating-mode admin API (fast/cheap switching).

Covers (LP-0MSLMYEEU002IBH6):
- GET /admin/mode returns the persisted mode (defaults to fast)
- POST /admin/set-mode validates the mode parameter (400 for invalid/missing)
- Requesting the active mode is a noop (200, no restart)
- Requesting a different mode persists it and spawns a background restart
- A second switch while a restart is pending is a noop if the mode matches,
  otherwise rejected with 409 (avoids restart loops)
- Mode state file round-trip (write/read)
"""

import json

import httpx
import pytest

from proxy import mode as mode_module


@pytest.fixture
def mode_file(tmp_path):
    """Return a temp path to use as the mode state file."""
    return tmp_path / ".mode"


@pytest.fixture(autouse=True)
def _reset_pending():
    """Ensure the pending-restart flag starts clean for every test."""
    with mode_module._mode_lock:
        mode_module._restart_pending = False
    yield
    with mode_module._mode_lock:
        mode_module._restart_pending = False


@pytest.fixture
def client():
    """ASGI test client against the real proxy app."""
    from proxy.server import app

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


# ---------------------------------------------------------------------------
# GET /admin/mode
# ---------------------------------------------------------------------------


class TestGetMode:
    @pytest.mark.asyncio
    async def test_defaults_to_fast_when_no_mode_persisted(self, mode_file, client, monkeypatch):
        """No .mode file -> GET /admin/mode returns fast (current behavior)."""
        # Point the mode module at the (empty) temp file path.
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        async with client as c:
            resp = await c.get("/admin/mode")
        assert resp.status_code == 200
        assert resp.json() == {"mode": "fast"}

    @pytest.mark.asyncio
    async def test_returns_persisted_mode(self, mode_file, client, monkeypatch):
        """.mode=cheap -> GET /admin/mode returns cheap."""
        mode_file.write_text("cheap\n")
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        async with client as c:
            resp = await c.get("/admin/mode")
        assert resp.status_code == 200
        assert resp.json() == {"mode": "cheap"}

    @pytest.mark.asyncio
    async def test_invalid_mode_file_defaults_to_fast(self, mode_file, client, monkeypatch):
        """A garbage .mode file -> fast (fail-open, current behavior)."""
        mode_file.write_text("gibberish\n")
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        assert mode_module.read_mode() == "fast"


# ---------------------------------------------------------------------------
# POST /admin/set-mode
# ---------------------------------------------------------------------------


class TestSetMode:
    @pytest.mark.asyncio
    async def test_invalid_mode_returns_400(self, mode_file, client, monkeypatch):
        """A mode other than fast/cheap -> 400."""
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        async with client as c:
            resp = await c.post(
                "/admin/set-mode", content=json.dumps({"mode": "turbo"})
            )
        assert resp.status_code == 400
        assert "fast" in resp.json()["detail"] and "cheap" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_missing_mode_returns_400(self, mode_file, client, monkeypatch):
        """A body without a mode parameter -> 400."""
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        async with client as c:
            resp = await c.post("/admin/set-mode", content=json.dumps({}))
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, mode_file, client, monkeypatch):
        """A non-JSON body -> 400."""
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        async with client as c:
            resp = await c.post("/admin/set-mode", content="not json")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_same_mode_is_noop_no_restart(self, mode_file, client, monkeypatch):
        """Requesting the active mode returns success WITHOUT restarting."""
        mode_file.write_text("fast\n")
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        spawned = []
        monkeypatch.setattr(mode_module, "_spawn_restart", lambda: spawned.append(True))

        async with client as c:
            resp = await c.post(
                "/admin/set-mode", content=json.dumps({"mode": "fast"})
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["mode"] == "fast"
        assert data["restart"] is False
        assert spawned == []  # no restart triggered
        assert mode_file.read_text().strip() == "fast"

    @pytest.mark.asyncio
    async def test_different_mode_persists_and_restarts(self, mode_file, client, monkeypatch):
        """Switching to a different mode persists it and spawns a restart."""
        mode_file.write_text("fast\n")
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        spawned = []
        monkeypatch.setattr(mode_module, "_spawn_restart", lambda: spawned.append(True))

        async with client as c:
            resp = await c.post(
                "/admin/set-mode", content=json.dumps({"mode": "cheap"})
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["mode"] == "cheap"
        assert data["restart"] is True
        assert spawned == [True]  # background restart armed
        assert mode_file.read_text().strip() == "cheap"  # persisted

    @pytest.mark.asyncio
    async def test_switch_back_to_fast_persists(self, mode_file, client, monkeypatch):
        """Switching from cheap back to fast persists the fast mode."""
        mode_file.write_text("cheap\n")
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        monkeypatch.setattr(mode_module, "_spawn_restart", lambda: None)
        async with client as c:
            resp = await c.post(
                "/admin/set-mode", content=json.dumps({"mode": "fast"})
            )
        assert resp.status_code == 200
        assert resp.json()["mode"] == "fast"
        assert resp.json()["restart"] is True
        assert mode_file.read_text().strip() == "fast"

    @pytest.mark.asyncio
    async def test_restart_pending_same_mode_is_noop(self, mode_file, client, monkeypatch):
        """While a restart is pending, requesting the persisted mode is a noop."""
        mode_file.write_text("cheap\n")
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        monkeypatch.setattr(mode_module, "_spawn_restart", lambda: None)
        with mode_module._mode_lock:
            mode_module._restart_pending = True
        try:
            async with client as c:
                resp = await c.post(
                    "/admin/set-mode", content=json.dumps({"mode": "cheap"})
                )
            assert resp.status_code == 200
            assert resp.json()["restart"] is False
        finally:
            with mode_module._mode_lock:
                mode_module._restart_pending = False

    @pytest.mark.asyncio
    async def test_restart_pending_different_mode_rejected_409(self, mode_file, client, monkeypatch):
        """While a restart is pending, a different mode is rejected with 409."""
        mode_file.write_text("cheap\n")
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        monkeypatch.setattr(mode_module, "_spawn_restart", lambda: None)
        with mode_module._mode_lock:
            mode_module._restart_pending = True
        try:
            async with client as c:
                resp = await c.post(
                    "/admin/set-mode", content=json.dumps({"mode": "fast"})
                )
            assert resp.status_code == 409
            assert "restart" in resp.json()["detail"]
        finally:
            with mode_module._mode_lock:
                mode_module._restart_pending = False


# ---------------------------------------------------------------------------
# Mode state file round-trip (module-level)
# ---------------------------------------------------------------------------


class TestModeStateFile:
    def test_write_then_read_round_trip(self, mode_file, monkeypatch):
        """write_mode persists and read_mode recovers the value."""
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        mode_module.write_mode("cheap")
        assert mode_module.read_mode() == "cheap"
        mode_module.write_mode("fast")
        assert mode_module.read_mode() == "fast"

    def test_write_invalid_mode_raises(self, mode_file, monkeypatch):
        """write_mode rejects anything other than fast/cheap."""
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        with pytest.raises(ValueError):
            mode_module.write_mode("turbo")

    def test_read_missing_file_defaults_to_fast(self, mode_file, monkeypatch):
        """read_mode with no file returns fast (no exception)."""
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        assert mode_module.read_mode() == "fast"

    def test_mode_config_file_mapping(self):
        """fast/cheap map to the two profile files; unknown falls back."""
        assert mode_module.MODE_CONFIG_FILES == {
            "fast": "config-fast.yaml",
            "cheap": "config-cheap.yaml",
        }
        # config files exist on disk
        proxy = mode_module.proxy_dir()
        for name in mode_module.MODE_CONFIG_FILES.values():
            assert (proxy / name).is_file()
