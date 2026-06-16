"""
Prompt injection integration test.

Integration test that exercises the full proxy path:
incoming request → prompt resolution → message composition → forwarded call to backend.
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest

import proxy.server as server
from proxy.prompt_resolver import resolve_system_prompt


# ---------------------------------------------------------------------------
# Integration test: override mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_override_mode_injects_system_prompt(monkeypatch, tmp_path):
    """A request to an alias with system_prompt.override should replace system messages."""
    # Setup: create an override prompt file
    override_dir = tmp_path / ".sorraAgents" / "prompts"
    override_dir.mkdir(parents=True, exist_ok=True)
    (override_dir / "assistant.txt").write_text("You are an AI assistant specialized in coding.", encoding="utf-8")

    # Point the resolver at our temp dir
    monkeypatch.setattr("proxy.prompt_resolver.OVERRIDE_DIR", override_dir)
    monkeypatch.setattr("proxy.prompt_resolver.REPO_ROOT", tmp_path)

    # --- Setup mock config ---
    mock_config = {
        "models": {
            "assistant-model": {
                "aliases": ["assistant", "asst"],
                "llama_model": "test-llama",
                "type": "local",
                "system_prompt": {
                    "mode": "override",
                    "file": "proxy/prompts/assistant.txt",
                },
            },
        },
        "server": {
            "llama_router_mode": False,
            "llama_server_port": 9999,
            "llama_embed_load_timeout": 5,
            "llama_model_load_timeout": 10,
            "llama_startup_timeout": 5,
            "max_concurrent_queries": 16,
        },
    }
    monkeypatch.setattr(server, 'config', mock_config)

    # Simulate a running local process
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    monkeypatch.setattr(server, 'llama_process', fake_proc)
    monkeypatch.setattr(server, 'backend_ready', True)
    server.current_model = "test-llama"

    # Use httpx.MockTransport to capture the request to the backend
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
    monkeypatch.setattr(server, '_http_client', client)

    from proxy.ui import proxy_openai_api
    from fastapi import Request as FastAPIRequest

    body = json.dumps({
        "model": "assistant",
        "messages": [
            {"role": "system", "content": "Original system message"},
            {"role": "user", "content": "Hello!"},
        ],
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

    async def fake_proxy_to_local(request, path):
        srv = server
        body_bytes = await request.body()
        body_json = json.loads(body_bytes) if body_bytes else {}

        # Forward to mock backend (prompt already applied by proxy_openai_api)
        new_body = json.dumps(body_json).encode("utf-8")
        headers = {"content-type": "application/json"}
        resp = await client.post(
            f"http://localhost:9999/{path}",
            content=new_body,
            headers=headers,
        )
        return resp

    monkeypatch.setattr(server, 'proxy_to_local', fake_proxy_to_local)

    resp = await proxy_openai_api(mock_request, "chat/completions")

    await client.aclose()

    # Verify the backend received the message with system prompt applied
    assert resp is not None
    assert resp.status_code == 200

    # Check captured body
    assert "messages" in captured_body
    messages = captured_body["messages"]

    # In override mode, system messages should be replaced
    system_msgs = [m for m in messages if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == "You are an AI assistant specialized in coding."

    # Non-system messages preserved
    user_msgs = [m for m in messages if m["role"] == "user"]
    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == "Hello!"


# ---------------------------------------------------------------------------
# Integration test: prepend mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prepend_mode_injects_system_prompt(monkeypatch, tmp_path):
    """A request to an alias with system_prompt.prepend should prepend system messages."""
    # Setup: create a repo prompt file
    repo_prompts = tmp_path / "proxy" / "prompts"
    repo_prompts.mkdir(parents=True, exist_ok=True)
    (repo_prompts / "code.txt").write_text("PREFIX: Always write tests.", encoding="utf-8")

    monkeypatch.setattr("proxy.prompt_resolver.OVERRIDE_DIR", tmp_path / ".sorraAgents" / "prompts")
    monkeypatch.setattr("proxy.prompt_resolver.REPO_ROOT", tmp_path)

    mock_config = {
        "models": {
            "code-model": {
                "aliases": ["code", "coder"],
                "llama_model": "test-llama",
                "type": "local",
                "system_prompt": {
                    "mode": "prepend",
                    "file": "proxy/prompts/code.txt",
                },
            },
        },
        "server": {
            "llama_router_mode": False,
            "llama_server_port": 9999,
            "llama_embed_load_timeout": 5,
            "llama_model_load_timeout": 10,
            "llama_startup_timeout": 5,
            "max_concurrent_queries": 16,
        },
    }
    monkeypatch.setattr(server, 'config', mock_config)

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    monkeypatch.setattr(server, 'llama_process', fake_proc)
    monkeypatch.setattr(server, 'backend_ready', True)
    server.current_model = "test-llama"

    captured_body = {}

    def backend_handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body
        try:
            captured_body = json.loads(request.content.decode("utf-8") if request.content else "{}")
        except Exception:
            captured_body = {"error": "parse_failed"}
        return httpx.Response(200, json={"id": "test", "message": {"content": "OK"}})

    transport = httpx.MockTransport(backend_handler)
    client = httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(server, '_http_client', client)

    from proxy.ui import proxy_openai_api
    from fastapi import Request as FastAPIRequest

    body = json.dumps({
        "model": "code",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a function"},
        ],
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

    async def fake_proxy_to_local(request, path):
        srv = server
        body_bytes = await request.body()
        body_json = json.loads(body_bytes) if body_bytes else {}

        # Forward to mock backend (prompt already applied by proxy_openai_api)
        new_body = json.dumps(body_json).encode("utf-8")
        headers = {"content-type": "application/json"}
        resp = await client.post(
            f"http://localhost:9999/{path}",
            content=new_body,
            headers=headers,
        )
        return resp

    monkeypatch.setattr(server, 'proxy_to_local', fake_proxy_to_local)

    resp = await proxy_openai_api(mock_request, "chat/completions")
    await client.aclose()

    assert resp is not None
    assert resp.status_code == 200
    assert "messages" in captured_body
    messages = captured_body["messages"]

    # In prepend mode, the prompt should be first, original system message preserved
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "PREFIX: Always write tests."

    system_msgs = [m for m in messages if m["role"] == "system"]
    assert len(system_msgs) == 2
    assert system_msgs[1]["content"] == "You are a helpful assistant."


# ---------------------------------------------------------------------------
# Integration test: no system_prompt configured passes through unchanged
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_system_prompt_passes_through(monkeypatch):
    """Without system_prompt config, messages should pass through unchanged."""
    mock_config = {
        "models": {
            "basic-model": {
                "aliases": ["basic"],
                "llama_model": "test-llama",
                "type": "local",
            },
        },
        "server": {
            "llama_router_mode": False,
            "llama_server_port": 9999,
            "llama_embed_load_timeout": 5,
            "llama_model_load_timeout": 10,
            "llama_startup_timeout": 5,
            "max_concurrent_queries": 16,
        },
    }
    monkeypatch.setattr(server, 'config', mock_config)

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    monkeypatch.setattr(server, 'llama_process', fake_proc)
    monkeypatch.setattr(server, 'backend_ready', True)
    server.current_model = "test-llama"

    captured_body = {}

    def backend_handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body
        try:
            captured_body = json.loads(request.content.decode("utf-8") if request.content else "{}")
        except Exception:
            captured_body = {"error": "parse_failed"}
        return httpx.Response(200, json={"id": "test", "message": {"content": "OK"}})

    transport = httpx.MockTransport(backend_handler)
    client = httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(server, '_http_client', client)

    from proxy.ui import proxy_openai_api
    from fastapi import Request as FastAPIRequest

    original_messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
    ]
    body = json.dumps({
        "model": "basic",
        "messages": original_messages,
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

    resp = await proxy_openai_api(mock_request, "chat/completions")
    await client.aclose()

    assert resp is not None
    assert resp.status_code == 200
    assert captured_body.get("messages") == original_messages
