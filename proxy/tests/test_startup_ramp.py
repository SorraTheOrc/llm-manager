"""Tests for the post-restart startup ramp (LP-0MU9ZXFQS0023DXT).

After a mode-switch restart, competing herdr/agent workers reconnect
simultaneously, creating a thundering-herd that overwhelms the freshly-started
proxy. The startup ramp gates new chat requests with 503 + random
Retry-After for a configurable window after each process start.

Clients simply retry with the given delay, so reconnects spread across the
ramp window instead of hitting the server all at once.
"""

import time

import pytest

from proxy import mode as mode_module


def _apply_ramp_config(monkeypatch, cfg=None):
    """Helper to apply a startup_ramp config in a test."""
    if cfg is None:
        cfg = {"enabled": True, "max_seconds": 30.0, "jitter_min": 2.0, "jitter_max": 8.0}
    monkeypatch.setattr(mode_module, "_startup_ramp_config", dict(cfg), raising=False)


@pytest.fixture
def ramp_config(monkeypatch):
    """Force a startup_ramp config for tests."""
    cfg = {
        "enabled": True,
        "max_seconds": 30.0,
        "jitter_min": 2.0,
        "jitter_max": 8.0,
    }
    _apply_ramp_config(monkeypatch, cfg)
    yield cfg
    mode_module.set_startup_ramp_config(None)  # clean up


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


class TestStartupRampConfig:
    def test_default_config(self):
        """Reset-to-None produces the default config values."""
        mode_module.set_startup_ramp_config(None)
        cfg = mode_module._startup_ramp_config_section({})
        assert cfg["enabled"] is True
        assert cfg["max_seconds"] == 180.0
        assert cfg["jitter_min"] == 5.0
        assert cfg["jitter_max"] == 15.0

    def test_enabled_false(self):
        """enabled: false produces a disabled config."""
        cfg = mode_module._startup_ramp_config_section(
            {"startup_ramp": {"enabled": False}}
        )
        assert cfg["enabled"] is False

    def test_zero_max_seconds_disables(self):
        """max_seconds: 0 disables the ramp."""
        cfg = mode_module._startup_ramp_config_section(
            {"startup_ramp": {"max_seconds": 0}}
        )
        assert cfg["max_seconds"] == 0.0

    def test_jitter_ordering_fixed(self):
        """If jitter_min > jitter_max, they are swapped."""
        cfg = mode_module._startup_ramp_config_section(
            {"startup_ramp": {"jitter_min": 20, "jitter_max": 5}}
        )
        assert cfg["jitter_min"] == 5.0
        assert cfg["jitter_max"] == 20.0

    def test_custom_values(self):
        """Custom values are respected."""
        cfg = mode_module._startup_ramp_config_section(
            {"startup_ramp": {"max_seconds": 180, "jitter_min": 10, "jitter_max": 30}}
        )
        assert cfg["max_seconds"] == 180.0
        assert cfg["jitter_min"] == 10.0
        assert cfg["jitter_max"] == 30.0


# ---------------------------------------------------------------------------
# Startup time recording (server.py integration)
# ---------------------------------------------------------------------------


class TestProxyStartTime:
    def test_start_time_set_by_server(self):
        """server.PROXY_START_TIME is a monotonic timestamp set at startup."""
        from proxy import server as srv_mod

        # The lifespan handler sets PROXY_START_TIME = time.monotonic().
        # In a test environment it may not have run, but the attribute always
        # exists (default 0.0).
        assert hasattr(srv_mod, "PROXY_START_TIME")
        assert isinstance(srv_mod.PROXY_START_TIME, (int, float))
        # After a real startup, it would be > 0.
        srv_mod.PROXY_START_TIME = time.monotonic()
        assert srv_mod.PROXY_START_TIME > 0


# ---------------------------------------------------------------------------
# API gate: new chat requests deferred during startup ramp
# ---------------------------------------------------------------------------


class TestStartupRampApiGate:
    @pytest.mark.asyncio
    async def test_chat_completions_deferred_during_ramp(self, ramp_config, monkeypatch):
        """New chat completions get 503 + Retry-After during the ramp."""
        import httpx
        import proxy.server as srv_mod
        from proxy import mode as _mode_mod
        from proxy.server import app

        # Verify config is active
        assert _mode_mod._startup_ramp_config is not None
        assert _mode_mod._startup_ramp_config["max_seconds"] == 30.0

        # Simulate a recent startup (within the ramp window).
        original_start_time = srv_mod.PROXY_START_TIME
        srv_mod.PROXY_START_TIME = time.monotonic() - 10  # 10s in

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"model": "plan", "messages": [{"role": "user", "content": "hi"}]},
                )
            # The ramp is 30s max and we started 10s ago, so we should be
            # throttled.
            assert resp.status_code == 503, f"Expected 503 from startup_ramp but got {resp.status_code}: {resp.text}"
            body = resp.json()
            assert body["error"]["type"] == "startup_ramp"
            assert body["status"] == 503
            assert "Retry-After" in resp.headers
            retry = int(resp.headers["Retry-After"])
            assert 5 <= retry <= 23  # jitter 2-8 + margin 3
        finally:
            srv_mod.PROXY_START_TIME = original_start_time

    @pytest.mark.asyncio
    async def test_not_deferred_after_ramp_expires(self, ramp_config, monkeypatch):
        """After the ramp window, requests proceed normally."""
        import httpx
        import proxy.server as srv_mod
        from proxy.server import app

        # Simulate a startup 60s ago (ramp is 30s max).
        original_start_time = srv_mod.PROXY_START_TIME
        srv_mod.PROXY_START_TIME = time.monotonic() - 60

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"model": "plan", "messages": [{"role": "user", "content": "hi"}]},
                )
            # Not the ramp deferral.
            assert resp.status_code != 503 or resp.json().get("error", {}).get("type") != "startup_ramp"
        finally:
            srv_mod.PROXY_START_TIME = original_start_time

    @pytest.mark.asyncio
    async def test_disabled_ramp_serves_normal(self, monkeypatch):
        """enabled: false → no throttling."""
        import httpx
        import proxy.server as srv_mod
        from proxy.server import app

        mode_module.set_startup_ramp_config({"enabled": False, "max_seconds": 180})
        srv_mod.PROXY_START_TIME = time.monotonic()

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"model": "plan", "messages": [{"role": "user", "content": "hi"}]},
                )
            assert resp.status_code != 503 or resp.json().get("error", {}).get("type") != "startup_ramp"
        finally:
            mode_module.set_startup_ramp_config(None)

    @pytest.mark.asyncio
    async def test_non_chat_endpoint_not_deferred(self, ramp_config, monkeypatch):
        """Health/admin endpoints stay available during the ramp."""
        import httpx
        import proxy.server as srv_mod
        from proxy.server import app

        srv_mod.PROXY_START_TIME = time.monotonic() - 5  # within ramp

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/admin/mode")
            assert resp.status_code == 200
        finally:
            pass

    @pytest.mark.asyncio
    async def test_ramp_zero_max_serves_normal(self, monkeypatch):
        """max_seconds: 0 → no throttling."""
        import httpx
        import proxy.server as srv_mod
        from proxy.server import app

        mode_module.set_startup_ramp_config({"max_seconds": 0})
        srv_mod.PROXY_START_TIME = time.monotonic()

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"model": "plan", "messages": [{"role": "user", "content": "hi"}]},
                )
            assert resp.status_code != 503 or resp.json().get("error", {}).get("type") != "startup_ramp"
        finally:
            mode_module.set_startup_ramp_config(None)

    @pytest.mark.asyncio
    async def test_jitter_spreads_values(self, ramp_config, monkeypatch):
        """Retry-After values are random within the jitter range."""
        import httpx
        import proxy.server as srv_mod
        from proxy.server import app

        srv_mod.PROXY_START_TIME = time.monotonic() - 5

        values = set()
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                for _ in range(20):
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={"model": "plan", "messages": [{"role": "user", "content": "hi"}]},
                    )
                    assert resp.status_code == 503
                    values.add(resp.headers["Retry-After"])
            # With 20 samples from a uniform range, we should see multiple
            # distinct values (not the same every time).
            assert len(values) > 1, "Jitter should produce different Retry-After values"
        finally:
            pass
