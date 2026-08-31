"""Tests for the bounded mode-switch drain (LP-0MT631JKW008WAKE / AC2).

A mode switch (scheduled 01:00/10:00 or manual ``POST /admin/set-mode``)
restarts the proxy. Previously the restart killed in-flight local streams
mid-generation, surfacing a synthetic ``finish_reason: error`` to clients
(4 events in the LP-0MT60S55M000TK1H analysis). The drain fixes this with a
**bounded** window:

- ``set_mode`` arms the drain synchronously when a restart is triggered.
- While draining, NEW chat requests are deferred (short 503 + Retry-After).
- The restart spawn is delayed until in-flight local streams
  (``local_active_queries``) finish, bounded by
  ``server.mode_switch_drain.max_seconds`` (default 30s).

The window is short and bounded so LP-0MSF9RUSQ007M346's "no long rejection
window" property is preserved (new requests are refused only during the
drain, with a Retry-After, and ``enabled: false`` restores the old
"just restart" behavior).
"""

import math
import threading
import time

import pytest

from proxy import mode as mode_module


@pytest.fixture(autouse=True)
def _reset_drain_state(monkeypatch):
    """Reset module drain state before/after every test."""
    with mode_module._drain_lock:
        mode_module._draining = False
        mode_module._drain_deadline = None
    yield
    with mode_module._drain_lock:
        mode_module._draining = False
        mode_module._drain_deadline = None
    with mode_module._mode_lock:
        mode_module._restart_pending = False


@pytest.fixture
def drain_config(monkeypatch):
    """Force the drain config section used by mode.py."""
    cfg = {
        "enabled": True,
        "max_seconds": 5.0,
        "retry_after_margin_seconds": 1.0,
    }
    monkeypatch.setattr(
        mode_module,
        "_mode_switch_drain_config",
        lambda server_config: dict(cfg),
    )
    return cfg


# ---------------------------------------------------------------------------
# Drain-arming behavior
# ---------------------------------------------------------------------------


class TestDrainArming:
    def test_begin_drain_sets_state(self, drain_config):
        """_begin_drain arms draining with a future deadline."""
        mode_module._begin_drain()
        assert mode_module.draining() is True
        with mode_module._drain_lock:
            deadline = mode_module._drain_deadline
        assert deadline is not None
        assert deadline > time.monotonic()

    def test_draining_false_when_disabled(self, drain_config):
        """enabled: false leaves draining inactive."""
        drain_config["enabled"] = False
        mode_module._begin_drain()
        assert mode_module.draining() is False

    def test_draining_false_when_zero_max_seconds(self, drain_config):
        """max_seconds <= 0 leaves draining inactive (opt-out)."""
        drain_config["max_seconds"] = 0
        mode_module._begin_drain()
        assert mode_module.draining() is False

    def test_set_mode_arms_drain_on_restart(self, drain_config, monkeypatch, tmp_path):
        """set_mode arming a restart begins the drain synchronously."""
        from proxy import mode as m

        monkeypatch.setattr(m, "_spawn_restart", lambda: None)
        # Ensure the switch is a real change (fast -> cheap).
        monkeypatch.setattr(m, "read_mode", lambda: m.MODE_FAST)
        monkeypatch.setattr(m, "write_mode", lambda mode: None)
        # Keep override-until state out of the real proxy dir.
        monkeypatch.setattr(m, "override_until_file", lambda: tmp_path / ".mode.override-until")

        persisted, restart = m.set_mode(m.MODE_CHEAP)
        assert restart is True
        assert persisted == m.MODE_CHEAP
        assert m.draining() is True

    def test_set_mode_does_not_arm_when_same_mode(self, drain_config, monkeypatch):
        """A noop set_mode (same mode) must not arm the drain."""
        from proxy import mode as m

        monkeypatch.setattr(m, "read_mode", lambda: m.MODE_FAST)
        monkeypatch.setattr(m, "_spawn_restart", lambda: None)
        persisted, restart = m.set_mode(m.MODE_FAST)
        assert restart is False
        assert m.draining() is False

    def test_drain_retry_after_zero_when_idle(self, drain_config):
        """No drain -> Retry-After of 0."""
        assert mode_module.drain_retry_after() == 0

    def test_drain_retry_after_positive_when_draining(self, drain_config):
        """Draining -> Retry-After covers remaining drain + margin."""
        mode_module._begin_drain()
        retry = mode_module.drain_retry_after()
        assert retry >= 1
        assert retry == math.ceil(5.0 + 1.0)


# ---------------------------------------------------------------------------
# Drain-wait (in-flight local streams)
# ---------------------------------------------------------------------------


class TestDrainWait:
    def test_wait_returns_when_streams_finish(self, drain_config, monkeypatch):
        """The wait completes as soon as local_active_queries hits 0."""
        import proxy.server as srv_mod

        monkeypatch.setattr(srv_mod, "local_active_queries", 2)

        def _drain_quickly():
            time.sleep(0.15)
            srv_mod.local_active_queries = 0

        t = threading.Thread(target=_drain_quickly, daemon=True)
        t.start()
        mode_module._begin_drain()  # arms a deadline (now + 5.0s)
        start = time.monotonic()
        mode_module._wait_for_in_flight_local_streams()
        elapsed = time.monotonic() - start
        assert elapsed < 5.0  # well under the bounded window
        assert mode_module.draining() is False  # state cleared after wait

    def test_wait_is_bounded(self, drain_config, monkeypatch):
        """A stuck in-flight stream times out at the configured bound."""
        import proxy.server as srv_mod

        monkeypatch.setattr(srv_mod, "local_active_queries", 1)
        drain_config["max_seconds"] = 0.2  # short bounded window
        mode_module._begin_drain()
        start = time.monotonic()
        mode_module._wait_for_in_flight_local_streams()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.1  # waited most of the bound
        assert elapsed < 2.0  # but bounded
        assert mode_module.draining() is False

    def test_wait_noop_when_idle(self, drain_config, monkeypatch):
        """No in-flight streams -> instant return, drain cleared."""
        import proxy.server as srv_mod

        monkeypatch.setattr(srv_mod, "local_active_queries", 0)
        mode_module._begin_drain()
        start = time.monotonic()
        mode_module._wait_for_in_flight_local_streams()
        assert time.monotonic() - start < 0.5
        assert mode_module.draining() is False

    def test_wait_noop_when_disabled(self, drain_config, monkeypatch):
        """enabled: false -> no deadline is armed, so the wait returns instantly."""
        import proxy.server as srv_mod

        monkeypatch.setattr(srv_mod, "local_active_queries", 3)
        drain_config["enabled"] = False
        mode_module._begin_drain()
        start = time.monotonic()
        mode_module._wait_for_in_flight_local_streams()
        assert time.monotonic() - start < 0.5
        assert mode_module.draining() is False


# ---------------------------------------------------------------------------
# API gate: new chat requests deferred while draining
# ---------------------------------------------------------------------------


class TestDrainApiGate:
    @pytest.mark.asyncio
    async def test_chat_completions_deferred_while_draining(self, drain_config, monkeypatch):
        """New chat completions get 503 + Retry-After during the drain."""
        import httpx
        from proxy.server import app

        with mode_module._mode_lock:
            mode_module._restart_pending = True
        mode_module._begin_drain()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "plan", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 503
        body = resp.json()
        assert body["error"]["type"] == "mode_switch_drain"
        assert body["status"] == 503
        assert "Retry-After" in resp.headers
        assert resp.headers["Retry-After"] == str(mode_module.drain_retry_after())

    def test_gate_requires_restart_pending(self, drain_config):
        """Draining alone must not defer: the gate also requires a pending
        restart (set by a real mode switch). This guards full-suite pollution
        from an earlier set_mode arming the drain with _spawn_restart mocked
        away and the fixture resetting only _restart_pending."""
        mode_module._begin_drain()
        assert mode_module.draining() is True
        # The fixture resets _restart_pending between tests -> gate inert.
        assert mode_module.restart_pending() is False

    @pytest.mark.asyncio
    async def test_chat_completions_not_deferred_when_idle(self, drain_config):
        """Without a drain, the normal path is not intercepted."""
        import httpx
        from proxy.server import app

        assert mode_module.draining() is False
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "plan", "messages": [{"role": "user", "content": "hi"}]},
            )
        # Not the drain deferral: no mode_switch_drain error type, and the
        # request proceeds through normal routing (which may 400/503 for
        # other reasons, but never the drain gate).
        assert resp.status_code != 503 or resp.json().get("error", {}).get("type") != "mode_switch_drain"

    @pytest.mark.asyncio
    async def test_non_chat_endpoint_not_deferred(self, drain_config):
        """Health/admin endpoints stay available during the drain."""
        import httpx
        from proxy.server import app

        mode_module._begin_drain()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/admin/mode")
        assert resp.status_code == 200
        assert "mode" in resp.json()

    @pytest.mark.asyncio
    async def test_deferral_expires_with_deadline(self, drain_config, monkeypatch):
        """After the drain deadline passes, chat requests are no longer
        deferred even if restart_pending is still set (self-expiring drain
        prevents an indefinite rejection window)."""
        import httpx
        from proxy.server import app

        with mode_module._mode_lock:
            mode_module._restart_pending = True
        # Arm the drain with an already-expired deadline.
        drain_config["max_seconds"] = -1
        mode_module._begin_drain()
        assert mode_module.draining() is False

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "plan", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code != 503 or resp.json().get("error", {}).get("type") != "mode_switch_drain"
