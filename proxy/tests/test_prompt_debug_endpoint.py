"""
Tests for the prompt debug endpoint (`GET /debug/prompt?alias=<alias>`).

Verifies gating, response format, and content preview behavior.
"""

import json
from unittest.mock import MagicMock

import proxy.server as server
import pytest

# ---------------------------------------------------------------------------
# Helper to build a mock Request
# ---------------------------------------------------------------------------

def _make_mock_request(client_host: str = "127.0.0.1"):
    """Create a minimal mock Request for calling the debug endpoint handler."""
    mock_request = MagicMock()
    mock_request.client = MagicMock()
    mock_request.client.host = client_host
    return mock_request


# ---------------------------------------------------------------------------
# Test: Endpoint gating
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_debug_endpoint_rejected_when_not_localhost(monkeypatch):
    """When not accessed from localhost and debug is off, reject."""
    monkeypatch.setattr(server, 'config', {"server": {"debug": False}})

    from proxy.server import debug_prompt

    mock_req = _make_mock_request(client_host="192.168.1.1")
    with pytest.raises(Exception):
        await debug_prompt(mock_req, alias="test")


@pytest.mark.asyncio
async def test_debug_endpoint_accepted_when_debug_true(monkeypatch):
    """When server.debug is True, access from any host is allowed."""
    monkeypatch.setattr(server, 'config', {"server": {"debug": True}, "models": {}})

    from proxy.server import debug_prompt

    mock_req = _make_mock_request(client_host="192.168.1.1")
    resp = await debug_prompt(mock_req, alias="test")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_debug_endpoint_accepted_from_localhost(monkeypatch):
    """Access from localhost is always allowed."""
    monkeypatch.setattr(server, 'config', {"server": {"debug": False}, "models": {}})

    from proxy.server import debug_prompt

    mock_req = _make_mock_request(client_host="127.0.0.1")
    resp = await debug_prompt(mock_req, alias="test")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Test: Missing alias parameter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_debug_endpoint_requires_alias(monkeypatch):
    """Request without alias should return 400."""
    monkeypatch.setattr(server, 'config', {"server": {"debug": True}})

    from proxy.server import debug_prompt

    mock_req = _make_mock_request()
    with pytest.raises(Exception) as _exc_info:
        await debug_prompt(mock_req, alias="")
    # Should be HTTPException with 400


# ---------------------------------------------------------------------------
# Test: Unknown model returns resolved=False
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_debug_endpoint_unknown_model(monkeypatch):
    """Unknown alias should return resolved=False."""
    monkeypatch.setattr(server, 'config', {"server": {"debug": True}, "models": {}})

    from proxy.server import debug_prompt

    mock_req = _make_mock_request()
    resp = await debug_prompt(mock_req, alias="nonexistent")
    data = json.loads(resp.body.decode("utf-8"))
    assert data["resolved"] is False
    assert data["alias"] == "nonexistent"


# ---------------------------------------------------------------------------
# Test: Model with no system_prompt returns resolved=False
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_debug_endpoint_no_system_prompt(monkeypatch):
    """Model without system_prompt should return resolved=False."""
    monkeypatch.setattr(
        server, 'config',
        {
            "server": {"debug": True},
            "models": {
                "basic": {
                    "type": "local",
                    "aliases": ["basic"],
                    "llama_model": "test-model",
                },
            },
        }
    )

    from proxy.server import debug_prompt

    mock_req = _make_mock_request()
    resp = await debug_prompt(mock_req, alias="basic")
    data = json.loads(resp.body.decode("utf-8"))
    assert data["resolved"] is False
    assert data["alias"] == "basic"


# ---------------------------------------------------------------------------
# Test: Resolved prompt returns expected fields
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_debug_endpoint_resolved_prompt(monkeypatch, tmp_path):
    """When a prompt is resolved, return expected fields with preview."""
    # Create a prompt file
    repo_prompts = tmp_path / "proxy" / "prompts"
    repo_prompts.mkdir(parents=True, exist_ok=True)
    prompt_file = repo_prompts / "test.txt"
    prompt_file.write_text("You are a test assistant.", encoding="utf-8")

    monkeypatch.setattr("proxy.prompt_resolver.OVERRIDE_DIR", tmp_path / ".sorraAgents" / "prompts")
    monkeypatch.setattr("proxy.prompt_resolver.REPO_ROOT", tmp_path)

    monkeypatch.setattr(
        server, 'config',
        {
            "server": {"debug": True},
            "models": {
                "test-model": {
                    "type": "local",
                    "aliases": ["test-model", "test"],
                    "llama_model": "test",
                    "system_prompt": {
                        "mode": "prepend",
                        "file": str(prompt_file),
                    },
                },
            },
        }
    )

    from proxy.server import debug_prompt

    mock_req = _make_mock_request()
    resp = await debug_prompt(mock_req, alias="test-model")
    data = json.loads(resp.body.decode("utf-8"))
    assert data["resolved"] is True
    assert data["alias"] == "test-model"
    assert data["mode"] == "prepend"
    assert data["source_path"] == str(prompt_file.resolve())
    assert data["content_preview"] == "You are a test assistant."
    assert data["size_bytes"] == len("You are a test assistant.")


# ---------------------------------------------------------------------------
# Test: Content preview is truncated to 200 chars
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_debug_endpoint_content_preview_truncated(monkeypatch, tmp_path):
    """Content preview should be truncated to 200 characters by default."""
    repo_prompts = tmp_path / "proxy" / "prompts"
    repo_prompts.mkdir(parents=True, exist_ok=True)
    prompt_file = repo_prompts / "long.txt"
    long_content = "Hello. " * 50  # ~300 chars
    prompt_file.write_text(long_content, encoding="utf-8")

    monkeypatch.setattr("proxy.prompt_resolver.OVERRIDE_DIR", tmp_path / ".sorraAgents" / "prompts")
    monkeypatch.setattr("proxy.prompt_resolver.REPO_ROOT", tmp_path)

    monkeypatch.setattr(
        server, 'config',
        {
            "server": {"debug": True},
            "models": {
                "long-model": {
                    "type": "local",
                    "aliases": ["long"],
                    "system_prompt": {
                        "mode": "override",
                        "file": str(prompt_file),
                    },
                },
            },
        }
    )

    from proxy.server import debug_prompt

    mock_req = _make_mock_request()
    resp = await debug_prompt(mock_req, alias="long-model")
    data = json.loads(resp.body.decode("utf-8"))
    assert data["resolved"] is True
    preview = data["content_preview"]
    assert len(preview) <= 203  # 200 chars + "..." = 203
    assert preview.endswith("...") or len(preview) == len(long_content)


# ---------------------------------------------------------------------------
# Test: Full content when full=true and debug enabled
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_debug_endpoint_full_content(monkeypatch, tmp_path):
    """When full=true and debug mode, return the full content."""
    repo_prompts = tmp_path / "proxy" / "prompts"
    repo_prompts.mkdir(parents=True, exist_ok=True)
    prompt_file = repo_prompts / "full.txt"
    long_content = "Word. " * 60  # ~360 chars
    prompt_file.write_text(long_content, encoding="utf-8")

    monkeypatch.setattr("proxy.prompt_resolver.OVERRIDE_DIR", tmp_path / ".sorraAgents" / "prompts")
    monkeypatch.setattr("proxy.prompt_resolver.REPO_ROOT", tmp_path)

    monkeypatch.setattr(
        server, 'config',
        {
            "server": {"debug": True},
            "models": {
                "full-model": {
                    "type": "local",
                    "aliases": ["full"],
                    "system_prompt": {
                        "mode": "override",
                        "file": str(prompt_file),
                    },
                },
            },
        }
    )

    from proxy.server import debug_prompt

    mock_req = _make_mock_request()
    resp = await debug_prompt(mock_req, alias="full-model", full=True)
    data = json.loads(resp.body.decode("utf-8"))
    assert data["resolved"] is True
    assert data["content_preview"] == long_content
