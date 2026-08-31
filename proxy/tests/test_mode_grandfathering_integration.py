"""Integration tests: grandfathered routing across a fast→cheap mode switch.

F2 of LP-0MSMF5LXO006YDNM (allow in-flight sessions to keep using their
models across fast↔cheap mode switches). These tests exercise the real
routing path (``_do_proxy_openai_api``) with monkeypatched stub configs and
stub remote providers:

- A session that made a request before the switch keeps being served on
  ``github`` via the fast-config remote provider after the switch.
- A pre-existing ``plan`` session keeps its full fast-mode provider chain
  (local first, then opencode → opencode-go → deepseek).
- Bindings persisted before a restart are recognized after it.
- A brand-new session (or a request with no session id) that appears after
  the switch is NOT grandfathered: cheap-mode handling only.

The synthetic configs below deliberately make cheap mode "local-only" so the
restricted-model machinery is observable, mirroring the intake brief's
scenario (the shipped cheap/fast profiles are identical since
LP-0MSMIPPJI007GU9N, which makes grandfathering inert in practice — these
tests pin the machinery's behavior when the configs DO differ).
"""

import json

import httpx
import proxy.server as server
import pytest
from proxy.grandfathering import GrandfatheringRegistry
from proxy.lifecycle import lookup_model_config

# ---------------------------------------------------------------------------
# Synthetic fast / cheap configs (cheap = local-only, github removed)
# ---------------------------------------------------------------------------

FAST_MODELS = {
    "github": {
        "providers": [
            {
                "name": "github-primary",
                "type": "remote",
                "endpoint": "https://models.inference.ai.azure.com",
            }
        ],
        "aliases": ["github-*"],
    },
    "plan": {
        "providers": [
            {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"},
            {
                "name": "opencode-deepseek",
                "type": "remote",
                "provider": "opencode",
                "endpoint": "https://opencode.ai/zen",
            },
            {
                "name": "opencode-go-2-deepseek",
                "type": "remote",
                "provider": "opencode-go",
                "endpoint": "https://opencode.ai/zen/go",
            },
            {
                "name": "opencode-go-deepseek",
                "type": "remote",
                "provider": "opencode-go",
                "endpoint": "https://opencode.ai/zen/go",
            },
            {
                "name": "deepseek-v4-flash",
                "type": "remote",
                "provider": "deepseek",
                "endpoint": "https://api.deepseek.com",
            },
        ],
        "aliases": ["plan"],
    },
    "author": {
        "providers": [
            {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"},
            {
                "name": "opencode-go-deepseek",
                "type": "remote",
                "endpoint": "https://opencode.ai/zen/go",
            },
        ],
        "aliases": ["author"],
    },
    "code": {
        "providers": [
            {"name": "local-qwen3-next", "type": "local", "llama_model": "Qwen3-Next"},
            {
                "name": "opencode-go-deepseek",
                "type": "remote",
                "endpoint": "https://opencode.ai/zen/go",
            },
        ],
        "aliases": ["code"],
    },
}

# Cheap mode strips github entirely and keeps only the local provider on the
# hybrid models (the intake-brief scenario; the shipped configs are now
# identical — see module docstring).
CHEAP_MODELS = {
    "plan": {
        "providers": [
            {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"},
        ],
        "aliases": ["plan"],
    },
    "author": {
        "providers": [
            {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"},
        ],
        "aliases": ["author"],
    },
    "code": {
        "providers": [
            {"name": "local-qwen3-next", "type": "local", "llama_model": "Qwen3-Next"},
        ],
        "aliases": ["code"],
    },
}


def cheap_config():
    return {"models": dict(CHEAP_MODELS), "server": {"max_concurrent_queries": 16}}


def fast_config():
    return {"models": dict(FAST_MODELS), "server": {"max_concurrent_queries": 16}}


# ---------------------------------------------------------------------------
# Request stub / fixtures
# ---------------------------------------------------------------------------


class _StubRequest:
    """Minimal FastAPI-Request-like stub for the routing path."""

    def __init__(self, session_id: str | None = None, model: str = "github"):
        self.headers = {}
        if session_id:
            self.headers["x-session-id"] = session_id
        self._body = json.dumps({"model": model}).encode("utf-8")
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


@pytest.fixture
def cheap_cheap_state(tmp_path, monkeypatch):
    """Wire server state as if the proxy is running CHEAP mode.

    Sets ``server.config`` (cheap), ``server.other_mode_config`` (fast), a
    registry backed by a temp state file, and the persisted mode file to
    ``cheap``.
    """
    from proxy import mode as mode_module

    mode_file = tmp_path / ".mode"
    mode_file.write_text("cheap\n")
    monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)

    registry = GrandfatheringRegistry(
        tmp_path / "grandfathering-state.json",
        mode_schedule=None,
    )
    monkeypatch.setattr(server, "config", cheap_config())
    monkeypatch.setattr(server, "other_mode_config", fast_config())
    monkeypatch.setattr(server, "grandfathering_registry", registry)
    monkeypatch.setattr(server, "current_model", None)
    monkeypatch.setattr(server, "llama_process", None)
    # Deterministic observability assertions: start the signal counter clean.
    server.backend_signal_counts["grandfathered"] = 0
    return {"registry": registry, "tmp_path": tmp_path}


async def _route(cheap_cheap_state, request):
    """Run the real routing path for a chat-completions request."""
    from proxy.ui import _do_proxy_openai_api

    return await _do_proxy_openai_api(
        request, "chat/completions", await request.body(), server
    )


# ---------------------------------------------------------------------------
# AC 1a: remote-only github keeps its fast-config provider after the switch
# ---------------------------------------------------------------------------


class TestGithubContinuation:
    @pytest.mark.asyncio
    async def test_grandfathered_github_session_served_via_fast_config(
        self, cheap_cheap_state, monkeypatch
    ):
        """A session that used github before the switch keeps the fast-config
        github remote provider after the switch (stub provider)."""
        registry = cheap_cheap_state["registry"]
        registry.record("sess-before", "github", mode="fast")

        captured = {}

        async def _stub_remote_fallback(request, path, model_cfg, config):
            captured["model_cfg"] = model_cfg
            return httpx.Response(
                status_code=200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )

        import proxy.provider as provider

        monkeypatch.setattr(
            provider, "proxy_with_remote_fallback", _stub_remote_fallback
        )

        request = _StubRequest(session_id="sess-before", model="github")
        resp = await _route(cheap_cheap_state, request)

        assert resp.status_code == 200
        assert captured.get("model_cfg") is not None, (
            "grandfathered github request must reach the remote fallback stub"
        )
        assert (
            captured["model_cfg"]["providers"][0]["endpoint"]
            == "https://models.inference.ai.azure.com"
        ), "must route with the FAST-config github provider chain"

        # Observability (AC 5): grandfathered routes are marked in metrics.
        assert server.backend_signal_counts.get("grandfathered", 0) == 1, (
            "grandfathered backend signal must be incremented"
        )

    @pytest.mark.asyncio
    async def test_github_alias_session_grandfathered(
        self, cheap_cheap_state, monkeypatch
    ):
        """A github-* alias session is grandfathered and resolved from the
        fast config via alias matching."""
        registry = cheap_cheap_state["registry"]
        registry.record("sess-alias", "github", mode="fast")

        captured = {}

        async def _stub_remote_fallback(request, path, model_cfg, config):
            captured["model_cfg"] = model_cfg
            return httpx.Response(
                status_code=200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )

        import proxy.provider as provider

        monkeypatch.setattr(
            provider, "proxy_with_remote_fallback", _stub_remote_fallback
        )

        request = _StubRequest(session_id="sess-alias", model="github-session")
        resp = await _route(cheap_cheap_state, request)

        assert resp.status_code == 200
        assert captured.get("model_cfg") is not None
        assert (
            captured["model_cfg"]["providers"][0]["endpoint"]
            == "https://models.inference.ai.azure.com"
        )


# ---------------------------------------------------------------------------
# AC 1b: hybrid plan keeps its full fast-mode provider chain
# ---------------------------------------------------------------------------


class TestPlanChainContinuation:
    @pytest.mark.asyncio
    async def test_grandfathered_plan_session_keeps_full_fast_chain(
        self, cheap_cheap_state, monkeypatch
    ):
        """A plan session active before the switch keeps the full fast-mode
        chain: local first, then opencode → opencode-go → deepseek (stubs;
        the local dispatch receives the full-chain config)."""
        registry = cheap_cheap_state["registry"]
        registry.record("plan-sess", "plan", mode="fast")

        captured = {}

        async def _fake_dispatch(request, srv, model_cfg, model_name, endpoint_path, enable_grace_window=True):
            captured["model_cfg"] = model_cfg
            return httpx.Response(
                status_code=200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )

        import proxy.ui as ui

        monkeypatch.setattr(ui, "_dispatch_local_model_load", _fake_dispatch)

        request = _StubRequest(session_id="plan-sess", model="plan")
        resp = await _route(cheap_cheap_state, request)

        assert resp.status_code == 200
        cfg = captured.get("model_cfg")
        assert cfg is not None, "grandfathered plan request must reach dispatch"
        types = [p["type"] for p in cfg["providers"]]
        assert types[0] == "local", "chain must stay local-first"
        assert "remote" in types, "remote fallback tiers must be present"
        brands = [p.get("provider") for p in cfg["providers"]]
        assert any("opencode" in (b or "") for b in brands), (
            "opencode fallback tier must be in the chain"
        )
        assert any("deepseek" in (b or "") for b in brands), (
            "deepseek fallback tier must be in the chain"
        )
        # The full fast-mode chain must include every remote tier (opencode,
        # opencode-go, deepseek) — not the degraded cheap-mode local-only set.
        assert len(cfg["providers"]) == len(FAST_MODELS["plan"]["providers"]), (
            "must use the full fast-mode provider chain, not the local-only one"
        )


# ---------------------------------------------------------------------------
# AC 2: bindings survive a restart
# ---------------------------------------------------------------------------


class TestRestartPersistence:
    @pytest.mark.asyncio
    async def test_binding_persisted_before_restart_is_recognized_after(
        self, cheap_cheap_state, monkeypatch
    ):
        """A binding saved to disk before a simulated restart is recognized
        by a freshly-constructed registry afterwards: the same X-Session-Id
        is grandfathered and served without reconfiguration."""
        registry = cheap_cheap_state["registry"]
        registry.record("restart-sess", "github", mode="fast")
        registry.save()

        # Simulate restart: a NEW registry loads the persisted state file.
        fresh = GrandfatheringRegistry(
            cheap_cheap_state["tmp_path"] / "grandfathering-state.json",
            mode_schedule=None,
        )
        monkeypatch.setattr(server, "grandfathering_registry", fresh)

        captured = {}

        async def _stub_remote_fallback(request, path, model_cfg, config):
            captured["model_cfg"] = model_cfg
            return httpx.Response(
                status_code=200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )

        import proxy.provider as provider

        monkeypatch.setattr(
            provider, "proxy_with_remote_fallback", _stub_remote_fallback
        )

        request = _StubRequest(session_id="restart-sess", model="github")
        resp = await _route(cheap_cheap_state, request)

        assert resp.status_code == 200
        assert captured.get("model_cfg") is not None, (
            "restored binding must be grandfathered and served"
        )
        assert (
            captured["model_cfg"]["providers"][0]["endpoint"]
            == "https://models.inference.ai.azure.com"
        )


# ---------------------------------------------------------------------------
# AC 4: no grandfathering without prior activity
# ---------------------------------------------------------------------------


class TestNoGrandfathering:
    @pytest.mark.asyncio
    async def test_new_session_after_switch_not_grandfathered_github(
        self, cheap_cheap_state, monkeypatch
    ):
        """A brand-new session after the switch requesting github gets
        current-mode handling: github is unknown in cheap config → the
        request falls through (default_remote disabled) and is rejected —
        never reaching the fast remote stub."""
        called = []

        async def _stub_remote_fallback(request, path, model_cfg, config):
            called.append(model_cfg)
            return httpx.Response(status_code=200, json={})

        import proxy.provider as provider

        monkeypatch.setattr(
            provider, "proxy_with_remote_fallback", _stub_remote_fallback
        )

        request = _StubRequest(session_id="brand-new", model="github")
        with pytest.raises(Exception) as exc_info:
            await _route(cheap_cheap_state, request)

        # github is unknown in cheap mode (no default_remote, no current
        # model) → the routing path raises an HTTPException (400).
        from fastapi import HTTPException

        assert isinstance(exc_info.value, HTTPException)
        assert exc_info.value.status_code == 400
        assert called == [], "new session must NOT reach the fast remote provider"

    @pytest.mark.asyncio
    async def test_new_session_after_switch_plan_gets_cheap_chain(
        self, cheap_cheap_state, monkeypatch
    ):
        """A brand-new plan session after the switch is routed with the
        cheap-mode local-only chain (no remote tiers)."""
        captured = {}

        async def _fake_dispatch(request, srv, model_cfg, model_name, endpoint_path, enable_grace_window=True):
            captured["model_cfg"] = model_cfg
            return httpx.Response(
                status_code=200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )

        import proxy.ui as ui

        monkeypatch.setattr(ui, "_dispatch_local_model_load", _fake_dispatch)

        request = _StubRequest(session_id="brand-new", model="plan")
        resp = await _route(cheap_cheap_state, request)

        assert resp.status_code == 200
        cfg = captured.get("model_cfg")
        assert cfg is not None
        types = [p["type"] for p in cfg["providers"]]
        assert types == ["local"], (
            "new session must be served by the cheap-mode local-only chain, "
            f"got provider types {types}"
        )

    @pytest.mark.asyncio
    async def test_request_without_session_id_not_grandfathered(
        self, cheap_cheap_state, monkeypatch
    ):
        """A request with no session id is never grandfathered, even if the
        model would be restricted."""
        registry = cheap_cheap_state["registry"]
        registry.record("anonymous-sess", "github", mode="fast")

        called = []

        async def _stub_remote_fallback(request, path, model_cfg, config):
            called.append(model_cfg)
            return httpx.Response(status_code=200, json={})

        import proxy.provider as provider

        monkeypatch.setattr(
            provider, "proxy_with_remote_fallback", _stub_remote_fallback
        )

        request = _StubRequest(session_id=None, model="github")
        with pytest.raises(Exception) as exc_info:
            await _route(cheap_cheap_state, request)

        from fastapi import HTTPException

        assert isinstance(exc_info.value, HTTPException)
        assert exc_info.value.status_code == 400
        assert called == []

    @pytest.mark.asyncio
    async def test_request_records_binding_for_explicit_session(
        self, cheap_cheap_state, monkeypatch
    ):
        """An explicit session's request records/refreshes the binding."""
        registry = cheap_cheap_state["registry"]

        async def _fake_dispatch(request, srv, model_cfg, model_name, endpoint_path, enable_grace_window=True):
            return httpx.Response(
                status_code=200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )

        import proxy.ui as ui

        monkeypatch.setattr(ui, "_dispatch_local_model_load", _fake_dispatch)

        request = _StubRequest(session_id="record-me", model="plan")
        await _route(cheap_cheap_state, request)

        binding = registry.get("record-me")
        assert binding is not None, "explicit session must be recorded"
        assert binding.model == "plan"
        assert binding.recorded_mode == "cheap", (
            "binding records the mode active when the session was first seen"
        )


# ---------------------------------------------------------------------------
# lookup_model_config helper (used by the routing override)
# ---------------------------------------------------------------------------


class TestLookupModelConfig:
    def test_direct_match(self):
        assert lookup_model_config(FAST_MODELS, "github") is not None

    def test_wildcard_alias_match(self):
        cfg = lookup_model_config(FAST_MODELS, "github-session")
        assert cfg is not None
        assert cfg["providers"][0]["endpoint"] == "https://models.inference.ai.azure.com"

    def test_missing_model_returns_none(self):
        assert lookup_model_config(FAST_MODELS, "nope") is None

    def test_none_returns_none(self):
        assert lookup_model_config(FAST_MODELS, None) is None
