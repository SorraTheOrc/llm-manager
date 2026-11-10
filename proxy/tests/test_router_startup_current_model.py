"""Router startup must not pin an embed preset as ``current_model``.

Regression for LP-0MVWBYAGL004BV8X: the startup preload path previously set
``current_model`` to ``resolved[0]`` which could be ``mxbai-embed`` — an
embedding preset that should never be "current" for chat dispatches.

The fix prefers the first non-embed model from the resolved preload list,
falling back to ``resolved[0]`` only in the degenerate all-embed case.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from proxy import router_helpers as rh
from proxy import server


@pytest.fixture(autouse=True)
def reset_server_state(monkeypatch):
    """Reset mutable global server state between tests."""
    monkeypatch.setattr(server, 'background_loads', {})
    monkeypatch.setattr(server, 'model_last_used', {})
    monkeypatch.setattr(server, 'llama_process', None)
    monkeypatch.setattr(server, '_http_client', None)
    monkeypatch.setattr(server, '_remote_http_client', None)
    server.current_model = None
    server.last_start_failure = None
    server.backend_ready = False
    try:
        server.model_switch_refcount = 0
    except Exception:
        pass
    yield


def _models_payload(*ids: str) -> dict:
    return {
        "object": "list",
        "data": [
            {"id": model_id, "status": {"value": "loaded"}} for model_id in ids
        ],
    }


def _install_router_mock(monkeypatch, loaded: set[str]) -> httpx.AsyncClient:
    """Install a MockTransport that emulates router /models and /models/load."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        if path == '/health' and method == 'GET':
            return httpx.Response(200)
        if path == '/models/load' and method == 'POST':
            try:
                payload = json.loads(
                    request.content.decode('utf-8') if request.content else '{}'
                )
            except Exception:
                payload = {}
            model = payload.get('model')
            if model:
                loaded.add(model)
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(400, json={"error": "missing model"})
        if path == '/models' and method == 'GET':
            return httpx.Response(200, json=_models_payload(*loaded))
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(server, '_http_client', client)
    return client


def _router_config(preload: list[str]) -> dict:
    return {
        "default_model": "qwen3",
        "models": {
            "embed": {
                "aliases": ["embeddings", "embed"],
                "llama_model": "mxbai-embed",
                "type": "local",
            },
            "qwen3": {
                "aliases": ["qwen3", "chat"],
                "llama_model": "Qwen3",
                "type": "local",
            },
        },
        "server": {
            "llama_router_mode": True,
            "llama_router_preload": preload,
            "embeddings_model": "mxbai-embed",
            "llama_server_port": 8080,
            "llama_embed_load_timeout": 5,
            "llama_model_load_timeout": 10,
            "llama_startup_timeout": 5,
        },
    }


async def _run_startup_loader(monkeypatch, preload: list[str]) -> None:
    """Run the real ``_startup_launch_default_model_loader`` task to completion."""
    monkeypatch.setattr(server, 'config', _router_config(preload))
    # Restore the real alias→llama_model resolver: some tests replace
    # ``server.get_local_model_name`` directly (not via monkeypatch) and leave
    # it stubbed, which would otherwise make this test order-dependent.
    from proxy.lifecycle import get_local_model_name as _real_resolver

    monkeypatch.setattr(server, 'get_local_model_name', _real_resolver)
    loaded: set[str] = set()
    client = _install_router_mock(monkeypatch, loaded)

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    monkeypatch.setattr(server, 'start_llama_server', lambda model, **kwargs: fake_proc)
    monkeypatch.setattr(
        server, 'wait_for_llama_server', AsyncMock(return_value=True)
    )
    # Pre-set a live process so the adoption branch (llama_process is None)
    # is skipped entirely.
    monkeypatch.setattr(server, 'llama_process', fake_proc)

    try:
        task = server._startup_launch_default_model_loader()
        await task
    finally:
        await client.aclose()


class TestStartupLoaderEndToEnd:
    """End-to-end tests through the real startup loader task."""

    @pytest.mark.asyncio
    async def test_embed_first_preload_does_not_pin_embed(self, monkeypatch):
        """AC2: an embed-first preload list does not leave the embed preset current."""
        # 'embeddings' (alias of embed) is appended by the loader; 'qwen3'
        # (default_model) too. The embed preset lands first.
        await _run_startup_loader(monkeypatch, ["mxbai-embed"])

        assert server.current_model == 'Qwen3', (
            f"Expected the chat model as current_model, got {server.current_model!r}"
        )
        assert server.current_model != 'mxbai-embed'

    @pytest.mark.asyncio
    async def test_model_scoped_paths_target_chat_child(self, monkeypatch):
        """AC3: model-scoped slot resolution targets the chat child after startup."""
        await _run_startup_loader(monkeypatch, ["mxbai-embed"])

        model_id, source = rh.resolve_local_model_id(server)
        assert model_id == 'Qwen3'
        assert source == 'current_model'
        assert model_id != 'mxbai-embed'


class TestSelectCurrentModelFromPreload:
    """Unit tests for the _select_current_model_from_preload helper.

    These tests cover the core selection logic used at startup to pick
    ``current_model`` from the resolved preload list.
    """

    def test_embed_first_preload_selects_non_embed(self):
        """Embed-first preload list picks the first non-embed model.

        AC1 + AC2: mxbai-embed first → current_model = Qwen3, not mxbai-embed.
        """
        resolved = ["mxbai-embed", "Qwen3"]
        result = server._select_current_model_from_preload(resolved)
        assert result == "Qwen3"

    def test_all_embed_preload_falls_back_to_first(self):
        """When every preload entry is embed, fall back to resolved[0]."""
        resolved = ["mxbai-embed", "all-minilm"]
        result = server._select_current_model_from_preload(resolved)
        # "all-minilm" does NOT contain "embed" substring, so it's not filtered
        # → it's selected as the non-embed model
        assert result == "all-minilm"

    def test_non_embed_first_selects_first_non_embed(self):
        """First non-embed entry in preload is chosen."""
        resolved = ["Qwen3-32B", "mxbai-embed", "Qwen3"]
        result = server._select_current_model_from_preload(resolved)
        assert result == "Qwen3-32B"

    def test_empty_resolved_returns_none(self):
        """Empty resolved list → current_model stays None."""
        result = server._select_current_model_from_preload([])
        assert result is None

    def test_single_chat_model(self):
        """Single chat model is selected as-is."""
        resolved = ["Qwen3"]
        result = server._select_current_model_from_preload(resolved)
        assert result == "Qwen3"

    def test_embed_only_all_embed(self):
        """When truly all entries contain 'embed', fall back to resolved[0]."""
        resolved = ["mxbai-embed", "llm-embed-v2"]
        result = server._select_current_model_from_preload(resolved)
        assert result == "mxbai-embed"


class TestNonEmbedModelIds:
    """Unit tests for the _non_embed_model_ids helper from router_helpers."""

    def test_filters_embed_models(self):
        """Only models containing 'embed' (case-insensitive) are filtered."""
        ids = ["mxbai-embed", "Qwen3", "llm-embed-v2", "gpt-4"]
        result = rh._non_embed_model_ids(ids)
        assert result == ["Qwen3", "gpt-4"]

    def test_all_embed_returns_empty(self):
        """When all ids contain 'embed', return empty list."""
        ids = ["mxbai-embed", "llm-embed-v2"]
        result = rh._non_embed_model_ids(ids)
        assert result == []

    def test_no_embed_returns_all(self):
        """When no ids contain 'embed', return all."""
        ids = ["Qwen3", "gpt-4", "llama3"]
        result = rh._non_embed_model_ids(ids)
        assert result == ids

    def test_case_insensitive(self):
        """'Embed' in any case is treated as embed."""
        ids = ["MXBai-EMBED", "qwen3", "LLM-EMBED-V2"]
        result = rh._non_embed_model_ids(ids)
        assert result == ["qwen3"]

    def test_embed_in_middle_of_name(self):
        """'embed' substring anywhere triggers the filter."""
        ids = ["my-embed-model", "Qwen3"]
        result = rh._non_embed_model_ids(ids)
        assert result == ["Qwen3"]
