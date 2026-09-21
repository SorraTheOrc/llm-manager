"""Tests for the post-restart startup ramp (LP-0MU9ZXFQS0023DXT, LP-0MUAY98ZR002JBAA).

After a mode-switch restart, competing herdr/agent workers reconnect
simultaneously, creating a thundering-herd that overwhelms the freshly-started
proxy. The startup ramp gates new chat requests with 503 + random
Retry-After only while local backends are not yet ready
(``backend_ready=False``).  Once backends are ready the gate clears
immediately, regardless of elapsed time.  ``max_seconds`` acts as a short
safety ceiling so requests are never blocked indefinitely.

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
        assert cfg["max_seconds"] == 30.0
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
        from proxy.server import app

        from proxy import mode as _mode_mod

        # Verify config is active
        assert _mode_mod._startup_ramp_config is not None
        assert _mode_mod._startup_ramp_config["max_seconds"] == 30.0

        # Simulate a recent startup (within the ramp window).
        original_start_time = srv_mod.PROXY_START_TIME
        srv_mod.PROXY_START_TIME = time.monotonic() - 10  # 10s in
        # Ensure backends are NOT ready — gate should defer.
        original_backend_ready = srv_mod.backend_ready
        srv_mod.backend_ready = False

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"model": "plan", "messages": [{"role": "user", "content": "hi"}]},
                )
            # The ramp is 30s max and we started 10s ago, so we should be
            # throttled (backends not ready).
            assert resp.status_code == 503, f"Expected 503 from startup_ramp but got {resp.status_code}: {resp.text}"
            body = resp.json()
            assert body["error"]["type"] == "startup_ramp"
            assert body["status"] == 503
            assert "Retry-After" in resp.headers
            retry = int(resp.headers["Retry-After"])
            assert 5 <= retry <= 23  # jitter 2-8 + margin 3
        finally:
            srv_mod.PROXY_START_TIME = original_start_time
            srv_mod.backend_ready = original_backend_ready

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
        # Ensure backends are NOT ready — gate should defer.
        original_backend_ready = srv_mod.backend_ready
        srv_mod.backend_ready = False

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
            srv_mod.backend_ready = original_backend_ready

    # -----------------------------------------------------------------------
    # Startup-ramp readiness-driven clear (LP-0MUAY98ZR002JBAA)
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_ready_early_clears_gate(self, monkeypatch):
        """When backends are ready before the ramp expires, the gate clears
        immediately — requests are served normally (AC1, AC4)."""
        import httpx
        import proxy.server as srv_mod
        from proxy.server import app

        mode_module.set_startup_ramp_config(
            {"enabled": True, "max_seconds": 180.0, "jitter_min": 5.0, "jitter_max": 15.0}
        )
        # Simulate startup 120s ago — well past the old 180s window but
        # backends are ready early at t=3.5s.
        original_start_time = srv_mod.PROXY_START_TIME
        srv_mod.PROXY_START_TIME = time.monotonic() - 120
        # Simulate that backends are ready
        original_backend_ready = srv_mod.backend_ready
        srv_mod.backend_ready = True

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"model": "plan", "messages": [{"role": "user", "content": "hi"}]},
                )
            # Not blocked by the ramp because backend_ready is True.
            assert resp.status_code != 503 or resp.json().get("error", {}).get("type") != "startup_ramp"
        finally:
            srv_mod.PROXY_START_TIME = original_start_time
            srv_mod.backend_ready = original_backend_ready
            mode_module.set_startup_ramp_config(None)

    @pytest.mark.asyncio
    async def test_never_ready_holds_until_ceiling(self, monkeypatch):
        """When backends never become ready, the gate still lifts at the
        max_seconds ceiling so chat requests are never blocked
        indefinitely (AC2, AC4)."""
        import httpx
        import proxy.server as srv_mod
        from proxy.server import app

        # Ceiling is 10s — short for test speed.
        mode_module.set_startup_ramp_config(
            {"enabled": True, "max_seconds": 10.0, "jitter_min": 2.0, "jitter_max": 5.0}
        )
        original_start_time = srv_mod.PROXY_START_TIME
        srv_mod.PROXY_START_TIME = time.monotonic() - 5  # 5s in
        # Simulate backends NOT ready
        original_backend_ready = srv_mod.backend_ready
        srv_mod.backend_ready = False

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"model": "plan", "messages": [{"role": "user", "content": "hi"}]},
                )
            # Still within the 10s ceiling and backends not ready → 503.
            assert resp.status_code == 503
            assert resp.json()["error"]["type"] == "startup_ramp"
        finally:
            srv_mod.PROXY_START_TIME = original_start_time
            srv_mod.backend_ready = original_backend_ready
            mode_module.set_startup_ramp_config(None)

    @pytest.mark.asyncio
    async def test_ready_beyond_ceiling_serves_normal(self, monkeypatch):
        """When max_seconds ceiling expires, the gate lifts even if
        backends are still not ready (safety net — AC2)."""
        import httpx
        import proxy.server as srv_mod
        from proxy.server import app

        mode_module.set_startup_ramp_config(
            {"enabled": True, "max_seconds": 5.0, "jitter_min": 1.0, "jitter_max": 3.0}
        )
        original_start_time = srv_mod.PROXY_START_TIME
        srv_mod.PROXY_START_TIME = time.monotonic() - 20  # well past 5s ceiling
        original_backend_ready = srv_mod.backend_ready
        srv_mod.backend_ready = False  # backends never ready

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"model": "plan", "messages": [{"role": "user", "content": "hi"}]},
                )
            # Ceiling expired → gate is lifted.
            assert resp.status_code != 503 or resp.json().get("error", {}).get("type") != "startup_ramp"
        finally:
            srv_mod.PROXY_START_TIME = original_start_time
            srv_mod.backend_ready = original_backend_ready
            mode_module.set_startup_ramp_config(None)

    @pytest.mark.asyncio
    async def test_disabled_ramp_ignores_backend_ready(self, monkeypatch):
        """enabled: false → no throttling regardless of backend_ready state.
        (AC5)."""
        import httpx
        import proxy.server as srv_mod
        from proxy.server import app

        mode_module.set_startup_ramp_config({"enabled": False, "max_seconds": 180})
        srv_mod.PROXY_START_TIME = time.monotonic()
        original_backend_ready = srv_mod.backend_ready
        srv_mod.backend_ready = False

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
            srv_mod.backend_ready = original_backend_ready
            mode_module.set_startup_ramp_config(None)

    @pytest.mark.asyncio
    async def test_zero_max_seconds_ignores_backend_ready(self, monkeypatch):
        """max_seconds: 0 → no throttling regardless of backend_ready.
        (AC5)."""
        import httpx
        import proxy.server as srv_mod
        from proxy.server import app

        mode_module.set_startup_ramp_config({"max_seconds": 0})
        srv_mod.PROXY_START_TIME = time.monotonic()
        original_backend_ready = srv_mod.backend_ready
        srv_mod.backend_ready = False

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
            srv_mod.backend_ready = original_backend_ready
            mode_module.set_startup_ramp_config(None)


# ---------------------------------------------------------------------------
# LP-0MUAY9AMS002O2ZO: ramp gates only local-bound chat requests
# ---------------------------------------------------------------------------
#
# The startup ramp runs before any routing decision.  Before this change it
# deferred *every* chat request while the gate was active — including requests
# the router would have sent to a remote provider.  The gate now inspects the
# model's provider chain and only defers requests that can *only* be served
# by the local backend (a chain with no remote provider).

LOCAL_ONLY_MODEL = {
    "providers": [
        {"name": "local-qwen", "type": "local", "llama_model": "Qwen3"},
    ],
}

# Multiple LOCAL backends are still local-only: there is no remote escape.
MULTI_LOCAL_MODEL = {
    "providers": [
        {"name": "local-a", "type": "local", "llama_model": "Qwen3"},
        {"name": "local-b", "type": "local", "llama_model": "Qwen3"},
    ],
}

LOCAL_WITH_REMOTE_FALLBACK_MODEL = {
    "providers": [
        {"name": "local-qwen", "type": "local", "llama_model": "Qwen3"},
        {"name": "opencode-go", "type": "remote", "endpoint": "https://opencode.ai/zen/go"},
    ],
}

REMOTE_ONLY_MODEL = {
    "providers": [
        {"name": "opencode-go", "type": "remote", "endpoint": "https://opencode.ai/zen/go"},
    ],
}


def _set_models(monkeypatch, models, default_remote=None):
    """Install a minimal server config with the given ``models`` mapping."""
    import proxy.server as srv_mod

    cfg = {"server": {}, "models": models}
    if default_remote is not None:
        cfg["default_remote"] = default_remote
    monkeypatch.setattr(srv_mod, "config", cfg, raising=True)


def _within_ramp(monkeypatch, seconds_in=10):
    """Enter the ramp window with backends not yet ready.

    ``seconds_in`` is how far into the window the process is; the default
    (10 s) is inside the 30 s ``ramp_config`` window.  ``backend_ready`` is
    forced False so the readiness-driven early clear does not short-circuit
    the gate.
    """
    import proxy.server as srv_mod

    monkeypatch.setattr(
        srv_mod, "PROXY_START_TIME", time.monotonic() - seconds_in, raising=True
    )
    monkeypatch.setattr(srv_mod, "backend_ready", False, raising=True)


@pytest.fixture
def stub_dispatch(monkeypatch):
    """Replace every downstream dispatch entry point with a 200 sentinel.

    A gated request returns the ``startup_ramp`` 503 *before* dispatch; a
    request that passes the gate reaches the sentinel (marked with the
    ``X-Test-Dispatch`` header).  Any accidental reach of a real backend is
    caught because the sentinel header would be absent.
    """
    from starlette.responses import JSONResponse

    import proxy.provider as provider_mod
    import proxy.server as srv_mod
    import proxy.ui as ui_mod

    async def _sentinel(*args, **kwargs):
        return JSONResponse(
            status_code=200,
            content={"served": True},
            headers={"X-Test-Dispatch": "stub"},
        )

    monkeypatch.setattr(ui_mod, "_dispatch_local_model_load", _sentinel, raising=True)
    monkeypatch.setattr(
        provider_mod, "proxy_with_remote_fallback", _sentinel, raising=True
    )
    monkeypatch.setattr(provider_mod, "proxy_with_fallback", _sentinel, raising=True)
    monkeypatch.setattr(srv_mod, "proxy_to_remote", _sentinel, raising=True)
    monkeypatch.setattr(srv_mod, "proxy_to_local", _sentinel, raising=True)
    return _sentinel


async def _post_chat(model):
    """POST a minimal chat/completions request through the ASGI app."""
    import httpx
    from proxy.server import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
        )


def _is_ramp_503(resp) -> bool:
    if resp.status_code != 503:
        return False
    try:
        return resp.json().get("error", {}).get("type") == "startup_ramp"
    except Exception:
        return False


class TestStartupRampLocalBoundGating:
    """AC1/AC2: the ramp only defers requests that must dispatch locally."""

    @pytest.mark.asyncio
    async def test_local_only_model_deferred_during_ramp(
        self, ramp_config, stub_dispatch, monkeypatch
    ):
        """AC2: a local-only chain is deferred with the unchanged 503 body."""
        _set_models(monkeypatch, {"local-model": LOCAL_ONLY_MODEL})
        _within_ramp(monkeypatch)

        resp = await _post_chat("local-model")

        assert _is_ramp_503(resp), f"expected startup_ramp 503: {resp.text}"
        body = resp.json()
        assert body["error"]["type"] == "startup_ramp"
        assert body["error"]["code"] == "startup_ramp"
        assert body["error"]["message"] == "Server is starting up; retry shortly."
        assert body["status"] == 503
        assert "Retry-After" in resp.headers
        assert resp.headers["Cache-Control"] == "no-store"
        # Gate blocked before dispatch.
        assert resp.headers.get("X-Test-Dispatch") != "stub"

    @pytest.mark.asyncio
    async def test_multi_local_backend_deferred_during_ramp(
        self, ramp_config, stub_dispatch, monkeypatch
    ):
        """AC3: multiple LOCAL providers have no remote escape → still gated."""
        _set_models(monkeypatch, {"multi-local": MULTI_LOCAL_MODEL})
        _within_ramp(monkeypatch)

        resp = await _post_chat("multi-local")

        assert _is_ramp_503(resp), f"expected startup_ramp 503: {resp.text}"

    @pytest.mark.asyncio
    async def test_local_with_remote_fallback_served_during_ramp(
        self, ramp_config, stub_dispatch, monkeypatch
    ):
        """AC1: a local+remote chain escapes the gate (matches the evidence)."""
        _set_models(monkeypatch, {"hybrid-model": LOCAL_WITH_REMOTE_FALLBACK_MODEL})
        _within_ramp(monkeypatch)

        resp = await _post_chat("hybrid-model")

        assert not _is_ramp_503(resp), f"request was ramp-gated: {resp.text}"
        assert resp.headers.get("X-Test-Dispatch") == "stub"

    @pytest.mark.asyncio
    async def test_remote_only_model_served_during_ramp(
        self, ramp_config, stub_dispatch, monkeypatch
    ):
        """AC1: a remote-first chain never dispatches locally → served."""
        _set_models(monkeypatch, {"remote-model": REMOTE_ONLY_MODEL})
        _within_ramp(monkeypatch)

        resp = await _post_chat("remote-model")

        assert not _is_ramp_503(resp), f"request was ramp-gated: {resp.text}"
        assert resp.headers.get("X-Test-Dispatch") == "stub"

    @pytest.mark.asyncio
    async def test_unknown_model_with_default_remote_served(
        self, ramp_config, stub_dispatch, monkeypatch
    ):
        """No model config + default_remote enabled → dispatch goes remote."""
        _set_models(
            monkeypatch,
            {},
            default_remote={
                "enabled": True,
                "endpoint": "https://opencode.ai/zen/go",
                "model": "some-remote-model",
            },
        )
        _within_ramp(monkeypatch)

        resp = await _post_chat("unknown-model-xyz")

        assert not _is_ramp_503(resp), f"request was ramp-gated: {resp.text}"
        assert resp.headers.get("X-Test-Dispatch") == "stub"


class TestStartupRampWindowStillBounded:
    """AC4: ramp-expired and ramp-disabled serve both local- and remote-bound."""

    @pytest.mark.asyncio
    async def test_local_only_served_after_ramp_expires(
        self, ramp_config, stub_dispatch, monkeypatch
    ):
        _set_models(monkeypatch, {"local-model": LOCAL_ONLY_MODEL})
        _within_ramp(monkeypatch, seconds_in=60)

        resp = await _post_chat("local-model")

        assert not _is_ramp_503(resp), f"ramp-expired request gated: {resp.text}"
        assert resp.headers.get("X-Test-Dispatch") == "stub"

    @pytest.mark.asyncio
    async def test_remote_bound_served_after_ramp_expires(
        self, ramp_config, stub_dispatch, monkeypatch
    ):
        _set_models(monkeypatch, {"remote-model": REMOTE_ONLY_MODEL})
        _within_ramp(monkeypatch, seconds_in=60)

        resp = await _post_chat("remote-model")

        assert not _is_ramp_503(resp), f"ramp-expired request gated: {resp.text}"
        assert resp.headers.get("X-Test-Dispatch") == "stub"

    @pytest.mark.asyncio
    async def test_disabled_ramp_serves_local_and_remote(
        self, stub_dispatch, monkeypatch
    ):
        import proxy.server as srv_mod

        _set_models(
            monkeypatch,
            {
                "local-model": LOCAL_ONLY_MODEL,
                "remote-model": REMOTE_ONLY_MODEL,
            },
        )
        monkeypatch.setattr(
            srv_mod, "PROXY_START_TIME", time.monotonic(), raising=True
        )
        # Use monkeypatch (not ``set_startup_ramp_config``) so the prior
        # ``_startup_ramp_config`` value is restored on teardown and this test
        # does not leak an enabled ramp into subsequent test modules.
        monkeypatch.setattr(
            mode_module,
            "_startup_ramp_config",
            {"enabled": False, "max_seconds": 180},
            raising=True,
        )

        resp_local = await _post_chat("local-model")
        resp_remote = await _post_chat("remote-model")

        assert not _is_ramp_503(resp_local), f"disabled ramp gated local: {resp_local.text}"
        assert not _is_ramp_503(resp_remote), f"disabled ramp gated remote: {resp_remote.text}"
        assert resp_local.headers.get("X-Test-Dispatch") == "stub"
        assert resp_remote.headers.get("X-Test-Dispatch") == "stub"


class TestStartupRampBackendReadyClearsGate:
    """The readiness-driven clear (LP-0MUAY98ZR002JBAA) is orthogonal: when
    backends are ready the gate clears for *every* model, local- or remote-bound.
    """

    @pytest.mark.asyncio
    async def test_local_only_served_when_backends_ready(
        self, ramp_config, stub_dispatch, monkeypatch
    ):
        import proxy.server as srv_mod

        _set_models(monkeypatch, {"local-model": LOCAL_ONLY_MODEL})
        monkeypatch.setattr(
            srv_mod, "PROXY_START_TIME", time.monotonic() - 5, raising=True
        )
        monkeypatch.setattr(srv_mod, "backend_ready", True, raising=True)

        resp = await _post_chat("local-model")

        assert not _is_ramp_503(resp), f"ready backends should clear the gate: {resp.text}"
        assert resp.headers.get("X-Test-Dispatch") == "stub"

    @pytest.mark.asyncio
    async def test_remote_bound_served_when_backends_ready(
        self, ramp_config, stub_dispatch, monkeypatch
    ):
        import proxy.server as srv_mod

        _set_models(monkeypatch, {"remote-model": REMOTE_ONLY_MODEL})
        monkeypatch.setattr(
            srv_mod, "PROXY_START_TIME", time.monotonic() - 5, raising=True
        )
        monkeypatch.setattr(srv_mod, "backend_ready", True, raising=True)

        resp = await _post_chat("remote-model")

        assert not _is_ramp_503(resp), f"ready backends should clear the gate: {resp.text}"
        assert resp.headers.get("X-Test-Dispatch") == "stub"


class TestStartupRampNonChatUnaffected:
    """AC5: non-chat endpoints are untouched by the refined gate."""

    @pytest.mark.asyncio
    async def test_non_chat_endpoint_not_gated(self, ramp_config, monkeypatch):
        import httpx
        from proxy.server import app

        _set_models(monkeypatch, {"local-model": LOCAL_ONLY_MODEL})
        _within_ramp(monkeypatch)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/admin/mode")

        assert resp.status_code == 200


class TestModelCanRouteRemotePredicate:
    """AC3: the gate predicate mirrors dispatch provider selection."""

    def test_local_only_is_false(self):
        from proxy.ui import _model_can_route_remote

        assert _model_can_route_remote(LOCAL_ONLY_MODEL) is False

    def test_multi_local_is_false(self):
        from proxy.ui import _model_can_route_remote

        assert _model_can_route_remote(MULTI_LOCAL_MODEL) is False

    def test_local_with_remote_is_true(self):
        from proxy.ui import _model_can_route_remote

        assert _model_can_route_remote(LOCAL_WITH_REMOTE_FALLBACK_MODEL) is True

    def test_remote_only_is_true(self):
        from proxy.ui import _model_can_route_remote

        assert _model_can_route_remote(REMOTE_ONLY_MODEL) is True

    def test_missing_or_malformed_is_false(self):
        from proxy.ui import _model_can_route_remote

        assert _model_can_route_remote(None) is False
        assert _model_can_route_remote({}) is False
        assert _model_can_route_remote({"providers": []}) is False
        assert _model_can_route_remote({"providers": "bad"}) is False
