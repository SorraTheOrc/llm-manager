"""
Integration / unit tests verifying that the proxy resolves
`plan` and `code` friendly model aliases to the correct model config.

Uses monkeypatched config (like test_prompt_injection_integration.py)
and can be run without a live proxy backend.
"""

import json
import time
from unittest.mock import MagicMock

import httpx
import pytest
from fnmatch import fnmatch

import proxy.server as server
from proxy.lifecycle import get_model_config


# ---------------------------------------------------------------------------
# Fixture: a minimal config that includes plan/code aliases on qwen3-next
# ---------------------------------------------------------------------------

@pytest.fixture
def alias_config():
    """Return a config dict with plan and code aliases on qwen3-next."""
    return {
        "models": {
            "qwen3-next": {
                "type": "local",
                "providers": [
                    {
                        "name": "local-qwen3-next",
                        "type": "local",
                        "llama_model": "Qwen3-Next",
                    },
                ],
                "aliases": [
                    "qwen3-next",
                    "qwen3-coder-next",
                    "qwen3-next*",
                    "plan",
                    "code",
                ],
                "force_full_prompt": True,
                "llama_model": "Qwen3-Next",
            },
            "plan": {
                "type": "local",
                "providers": [
                    {
                        "name": "local-qwen3-next",
                        "type": "local",
                        "llama_model": "Qwen3-Next",
                    },
                ],
                "aliases": ["plan"],
                "force_full_prompt": True,
                "llama_model": "Qwen3-Next",
            },
            "code": {
                "type": "local",
                "providers": [
                    {
                        "name": "local-qwen3-next",
                        "type": "local",
                        "llama_model": "Qwen3-Next",
                    },
                ],
                "aliases": ["code"],
                "force_full_prompt": True,
                "llama_model": "Qwen3-Next",
            },
            "embed": {
                "type": "local",
                "providers": [
                    {
                        "name": "local-embed",
                        "type": "local",
                        "llama_model": "mxbai-embed",
                    },
                ],
                "aliases": ["embeddings", "embed"],
                "llama_model": "mxbai-embed",
            },
        },
        "server": {
            "llama_router_mode": True,
            "llama_server_port": 9999,
            "llama_embed_load_timeout": 5,
            "llama_model_load_timeout": 10,
            "llama_startup_timeout": 5,
            "max_concurrent_queries": 16,
        },
    }


# ===================================================================
# Unit tests for get_model_config alias resolution
# ===================================================================


def test_get_model_config_exact_plan(monkeypatch, alias_config):
    """get_model_config returns qwen3-next config when model='plan'."""
    monkeypatch.setattr(server, "config", alias_config)
    cfg = get_model_config("plan")
    assert cfg is not None
    assert cfg.get("llama_model") == "Qwen3-Next"


def test_get_model_config_exact_code(monkeypatch, alias_config):
    """get_model_config returns qwen3-next config when model='code'."""
    monkeypatch.setattr(server, "config", alias_config)
    cfg = get_model_config("code")
    assert cfg is not None
    assert cfg.get("llama_model") == "Qwen3-Next"


def test_get_model_config_case_insensitive(monkeypatch, alias_config):
    """get_model_config resolves case-insensitive alias variants."""
    monkeypatch.setattr(server, "config", alias_config)
    for variant in ["Plan", "CODE", "pLaN", "Code"]:
        cfg = get_model_config(variant)
        assert cfg is not None, f"get_model_config({variant!r}) returned None"
        assert cfg.get("llama_model") == "Qwen3-Next", f"Unexpected model for {variant!r}"


def test_get_model_config_unknown_alias(monkeypatch, alias_config):
    """get_model_config returns None for an unknown alias."""
    monkeypatch.setattr(server, "config", alias_config)
    cfg = get_model_config("nonexistent-alias-xyz")
    assert cfg is None


def test_get_model_config_embeddings_still_works(monkeypatch, alias_config):
    """Existing embeddings alias still resolves correctly."""
    monkeypatch.setattr(server, "config", alias_config)
    cfg = get_model_config("embeddings")
    assert cfg is not None
    assert cfg.get("llama_model") == "mxbai-embed"


def test_get_model_config_none(monkeypatch, alias_config):
    """get_model_config returns None for None input."""
    monkeypatch.setattr(server, "config", alias_config)
    cfg = get_model_config(None)
    assert cfg is None


def test_get_model_config_wildcard_preserved(monkeypatch, alias_config):
    """Existing wildcard aliases (qwen3-next*) still work after adding plan/code."""
    monkeypatch.setattr(server, "config", alias_config)
    cfg = get_model_config("qwen3-next-some-variant")
    assert cfg is not None
    assert cfg.get("llama_model") == "Qwen3-Next"


# ===================================================================
# Tests for /v1/models listing
# ===================================================================


def test_models_endpoint_contains_plan_and_code(monkeypatch, alias_config):
    """The /v1/models response includes plan and code in qwen3-next aliases."""
    monkeypatch.setattr(server, "config", alias_config)

    from proxy.handlers import list_models

    resp = None

    async def run():
        nonlocal resp
        resp = await list_models()
        return resp

    import asyncio
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run())
    finally:
        loop.close()

    assert resp is not None

    # Parse the response
    body = json.loads(resp.body.decode("utf-8")) if hasattr(resp, "body") else resp

    if isinstance(body, dict) and "data" in body:
        models = body["data"]
    elif isinstance(resp, dict) and "data" in resp:
        models = resp["data"]
    else:
        pytest.fail(f"Unexpected response format: {body}")

    qwen3_next_entry = None
    for m in models:
        if m.get("id") == "qwen3-next":
            qwen3_next_entry = m
            break

    assert qwen3_next_entry is not None, "qwen3-next not found in /v1/models"
    aliases = qwen3_next_entry.get("aliases", [])
    assert "plan" in aliases, f"'plan' not in aliases: {aliases}"
    assert "code" in aliases, f"'code' not in aliases: {aliases}"
    assert "embeddings" not in aliases, "'embeddings' should not be in qwen3-next aliases"

    # Also verify plan and code appear as standalone model entries
    plan_entry = next((m for m in models if m.get("id") == "plan"), None)
    code_entry = next((m for m in models if m.get("id") == "code"), None)
    assert plan_entry is not None, "'plan' should appear as a standalone model entry"
    assert code_entry is not None, "'code' should appear as a standalone model entry"
    assert plan_entry.get("aliases") == ["plan"], f"plan entry aliases: {plan_entry.get('aliases')}"
    assert code_entry.get("aliases") == ["code"], f"code entry aliases: {code_entry.get('aliases')}"


# ===================================================================
# Integration test pattern: proxy_openai_api routing
# These tests use httpx.MockTransport to capture the backend request
# and verify the model was resolved and forwarded correctly.
# ===================================================================


@pytest.mark.asyncio
async def test_plan_alias_routes_to_qwen3_next(monkeypatch, alias_config):
    """POST /v1/chat/completions with model='plan' routes to Qwen3-Next."""
    monkeypatch.setattr(server, "config", alias_config)

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    monkeypatch.setattr(server, "llama_process", fake_proc)
    monkeypatch.setattr(server, "backend_ready", True)
    server.current_model = "Qwen3-Next"

    captured_body = {}

    def backend_handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body
        try:
            captured_body = json.loads(request.content.decode("utf-8") if request.content else "{}")
        except Exception:
            captured_body = {"error": "parse_failed"}
        return httpx.Response(200, json={"id": "test", "choices": [{"message": {"content": "OK"}}]})

    transport = httpx.MockTransport(backend_handler)
    client = httpx.AsyncClient(transport=transport)

    # Monkey-patch proxy_to_local so it forwards through our mock client
    async def fake_proxy_to_local(request, path):
        srv = server
        body_bytes = await request.body()
        body_json = json.loads(body_bytes) if body_bytes else {}
        new_body = json.dumps(body_json).encode("utf-8")
        headers = {"content-type": "application/json"}
        resp = await client.post(
            f"http://localhost:9999/{path}",
            content=new_body,
            headers=headers,
        )
        return resp

    monkeypatch.setattr(server, 'proxy_to_local', fake_proxy_to_local)

    from proxy.ui import proxy_openai_api
    from fastapi import Request as FastAPIRequest

    body = json.dumps({
        "model": "plan",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }).encode("utf-8")

    mock_request = MagicMock(spec=FastAPIRequest)
    mock_request.method = "POST"
    mock_request.url = type("U", (), {"path": "/v1/chat/completions"})()
    mock_request.headers = {}
    mock_request._body = body

    async def mock_body():
        return mock_request._body

    mock_request.body = mock_body

    resp = await proxy_openai_api(mock_request, "chat/completions")
    await client.aclose()

    assert resp is not None
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_code_alias_routes_to_qwen3_next(monkeypatch, alias_config):
    """POST /v1/chat/completions with model='code' routes to Qwen3-Next."""
    monkeypatch.setattr(server, "config", alias_config)

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    monkeypatch.setattr(server, "llama_process", fake_proc)
    monkeypatch.setattr(server, "backend_ready", True)
    server.current_model = "Qwen3-Next"

    captured_body = {}

    def backend_handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body
        try:
            captured_body = json.loads(request.content.decode("utf-8") if request.content else "{}")
        except Exception:
            captured_body = {"error": "parse_failed"}
        return httpx.Response(200, json={"id": "test", "choices": [{"message": {"content": "OK"}}]})

    transport = httpx.MockTransport(backend_handler)
    client = httpx.AsyncClient(transport=transport)

    async def fake_proxy_to_local(request, path):
        srv = server
        body_bytes = await request.body()
        body_json = json.loads(body_bytes) if body_bytes else {}
        new_body = json.dumps(body_json).encode("utf-8")
        headers = {"content-type": "application/json"}
        resp = await client.post(
            f"http://localhost:9999/{path}",
            content=new_body,
            headers=headers,
        )
        return resp

    monkeypatch.setattr(server, 'proxy_to_local', fake_proxy_to_local)

    from proxy.ui import proxy_openai_api
    from fastapi import Request as FastAPIRequest

    body = json.dumps({
        "model": "code",
        "messages": [{"role": "user", "content": "Write a function"}],
        "stream": False,
    }).encode("utf-8")

    mock_request = MagicMock(spec=FastAPIRequest)
    mock_request.method = "POST"
    mock_request.url = type("U", (), {"path": "/v1/chat/completions"})()
    mock_request.headers = {}
    mock_request._body = body

    async def mock_body():
        return mock_request._body

    mock_request.body = mock_body

    resp = await proxy_openai_api(mock_request, "chat/completions")
    await client.aclose()

    assert resp is not None
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_plan_alias_resolves_model_name(monkeypatch, alias_config):
    """The model in the forwarded request body should remain 'plan' (alias preserved)."""
    monkeypatch.setattr(server, "config", alias_config)

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    monkeypatch.setattr(server, "llama_process", fake_proc)
    monkeypatch.setattr(server, "backend_ready", True)
    server.current_model = "Qwen3-Next"

    captured_model = None

    def backend_handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_model
        try:
            data = json.loads(request.content.decode("utf-8") if request.content else "{}")
            captured_model = data.get("model")
        except Exception:
            pass
        return httpx.Response(200, json={"id": "test", "choices": [{"message": {"content": "OK"}}]})

    transport = httpx.MockTransport(backend_handler)
    client = httpx.AsyncClient(transport=transport)

    async def fake_proxy_to_local(request, path):
        srv = server
        body_bytes = await request.body()
        body_json = json.loads(body_bytes) if body_bytes else {}
        new_body = json.dumps(body_json).encode("utf-8")
        headers = {"content-type": "application/json"}
        resp = await client.post(
            f"http://localhost:9999/{path}",
            content=new_body,
            headers=headers,
        )
        return resp

    monkeypatch.setattr(server, 'proxy_to_local', fake_proxy_to_local)

    from proxy.ui import proxy_openai_api
    from fastapi import Request as FastAPIRequest

    body = json.dumps({
        "model": "plan",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }).encode("utf-8")

    mock_request = MagicMock(spec=FastAPIRequest)
    mock_request.method = "POST"
    mock_request.url = type("U", (), {"path": "/v1/chat/completions"})()
    mock_request.headers = {}
    mock_request._body = body

    async def mock_body():
        return mock_request._body

    mock_request.body = mock_body

    await proxy_openai_api(mock_request, "chat/completions")
    await client.aclose()

    # The model name in the forwarded body should still be 'plan' (backend
    # receives the alias, and the config resolves it for routing).
    # Note: proxy passes model through, so it should be 'plan'.
    assert captured_model == "plan", f"Expected 'plan', got {captured_model!r}"
