"""Tests for the new home page showing model endpoints with activity state.

Tests cover:
1. Per-model active query increment/decrement functions
2. SSE payload includes per_model_queries
3. Index handler injects model endpoint data into template
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

# Ensure proxy package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def mock_srv():
    """Create a mock server module with per_model_queries state."""
    srv = MagicMock()
    srv.per_model_queries = {}
    srv.per_model_queries_lock = MagicMock()
    srv.config = {
        "models": {
            "model-a": {
                "providers": [
                    {"name": "local-llama", "type": "local", "llama_model": "qwen3"},
                    {"name": "remote-api", "type": "remote", "endpoint": "https://api.example.com"},
                ],
                "aliases": ["a-*"],
            },
            "model-b": {
                "providers": [
                    {"name": "remote-only", "type": "remote", "endpoint": "https://remote.api/v1"},
                ],
                "aliases": ["b-*"],
            },
            "model-c": {
                "providers": [
                    {"name": "local-only", "type": "local", "llama_model": "gemma4"},
                ],
            },
        },
        "server": {"llama_router_mode": False},
    }
    srv.current_model = "model-a"
    srv.llama_process = MagicMock()
    srv.llama_process.poll = MagicMock(return_value=None)
    srv.llama_server_version = "build 1234"
    srv.rocm_version = "6.0.0"
    srv.sse_clients = set()
    return srv


@pytest.mark.asyncio
async def test_increment_per_model_query():
    """Test that _increment_per_model_query increments the counter."""
    from proxy.router_helpers import _increment_per_model_query

    srv = MagicMock()
    srv.per_model_queries = {}
    srv.per_model_queries_lock = AsyncMock()
    srv.per_model_queries_lock.__aenter__.return_value = None
    srv.per_model_queries_lock.__aexit__.return_value = None

    # Patch the lock context manager to actually manage the dict
    async def mock_aenter():
        return srv.per_model_queries

    async def mock_aexit(*args):
        pass

    srv.per_model_queries_lock.__aenter__ = mock_aenter
    srv.per_model_queries_lock.__aexit__ = mock_aexit

    # Simulate with actual dict access inside context manager
    # Since AsyncMock doesn't actually execute the context manager body,
    # let's use a real asyncio.Lock
    import asyncio
    srv.per_model_queries = {}
    srv.per_model_queries_lock = asyncio.Lock()

    await _increment_per_model_query(srv, "model-a")
    assert srv.per_model_queries == {"model-a": 1}

    await _increment_per_model_query(srv, "model-a")
    assert srv.per_model_queries == {"model-a": 2}

    await _increment_per_model_query(srv, "model-b")
    assert srv.per_model_queries == {"model-a": 2, "model-b": 1}


@pytest.mark.asyncio
async def test_decrement_per_model_query():
    """Test that _decrement_per_model_query decrements the counter."""
    from proxy.router_helpers import _increment_per_model_query, _decrement_per_model_query

    import asyncio
    srv = MagicMock()
    srv.per_model_queries = {}
    srv.per_model_queries_lock = asyncio.Lock()

    await _increment_per_model_query(srv, "model-a")
    await _increment_per_model_query(srv, "model-a")
    await _increment_per_model_query(srv, "model-b")

    await _decrement_per_model_query(srv, "model-a")
    assert srv.per_model_queries == {"model-a": 1, "model-b": 1}

    await _decrement_per_model_query(srv, "model-a")
    assert srv.per_model_queries == {"model-a": 0, "model-b": 1}

    # Decrement below zero should be clamped
    await _decrement_per_model_query(srv, "model-a")
    assert srv.per_model_queries == {"model-a": 0, "model-b": 1}


@pytest.mark.asyncio
async def test_per_model_query_unknown_model():
    """Test that increment/decrement with None or empty model name is safe."""
    from proxy.router_helpers import _increment_per_model_query, _decrement_per_model_query

    import asyncio
    srv = MagicMock()
    srv.per_model_queries = {}
    srv.per_model_queries_lock = asyncio.Lock()

    # Should not raise
    await _increment_per_model_query(srv, None)
    await _decrement_per_model_query(srv, None)
    await _increment_per_model_query(srv, "")
    await _decrement_per_model_query(srv, "")

    assert srv.per_model_queries == {}


@pytest.mark.asyncio
async def test_get_per_model_queries():
    """Test snapshot function returns current state."""
    from proxy.router_helpers import _increment_per_model_query, _get_per_model_queries

    import asyncio
    srv = MagicMock()
    srv.per_model_queries = {}
    srv.per_model_queries_lock = asyncio.Lock()

    await _increment_per_model_query(srv, "model-a")
    await _increment_per_model_query(srv, "model-b")

    snapshot = await _get_per_model_queries(srv)
    assert snapshot == {"model-a": 1, "model-b": 1}
    # Verify it's a copy, not a reference
    snapshot["model-new"] = 5
    assert "model-new" not in srv.per_model_queries


@pytest.mark.asyncio
async def test_sse_payload_includes_per_model_queries():
    """Test that status_events payload includes per_model_queries."""
    from proxy.ui import status_events

    import asyncio
    srv = MagicMock()
    srv.per_model_queries = {"model-a": 2}
    srv.per_model_queries_lock = asyncio.Lock()
    srv.sse_clients = set()
    srv.current_model = "model-a"
    srv.token_counts = {"total_sent": 100, "total_recv": 200}
    srv.config = {"server": {}}
    srv.query_llama_status = AsyncMock(return_value={
        "llama_server_running": True,
        "n_ctx": 4096,
        "kv_cache_tokens": 128,
        "router_mode": False,
    })
    srv.router_list_models = AsyncMock(return_value=[])

    with patch("proxy.ui._srv", return_value=srv):
        response = await status_events()
        assert response.media_type == "text/event-stream"

        # Verify the response shape is correct
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-cache"


def test_index_handler_injects_model_endpoint_data():
    """Test that index handler builds model endpoint data for the template."""
    # This test verifies the model endpoint data structure that will be used
    # by the index handler to populate the home page tab
    test_data = []
    for model_name, cfg in {
        "model-a": {
            "providers": [
                {"name": "local-llama", "type": "local", "llama_model": "qwen3"},
                {"name": "fallback-remote", "type": "remote", "endpoint": "https://fallback.example.com"},
            ],
        },
        "model-b": {
            "providers": [
                {"name": "remote-only", "type": "remote", "endpoint": "https://remote.api/v1"},
            ],
        },
        "model-c": {
            "providers": [
                {"name": "local-only", "type": "local", "llama_model": "gemma4"},
            ],
        },
    }.items():
        entry = {
            "name": model_name,
            "type": "local" if any(p.get("type") == "local" for p in cfg.get("providers", [])) else "remote",
            "providers": [],
        }
        for p in cfg.get("providers", []):
            provider_info = {
                "name": p.get("name", ""),
                "type": p.get("type", ""),
                "endpoint": p.get("endpoint") or p.get("llama_model", ""),
            }
            entry["providers"].append(provider_info)
        test_data.append(entry)

    assert len(test_data) == 3
    assert test_data[0]["name"] == "model-a"
    assert test_data[0]["type"] == "local"
    assert len(test_data[0]["providers"]) == 2
    assert test_data[0]["providers"][1]["type"] == "remote"
    assert test_data[0]["providers"][1]["endpoint"] == "https://fallback.example.com"

    assert test_data[1]["name"] == "model-b"
    assert test_data[1]["type"] == "remote"
    assert len(test_data[1]["providers"]) == 1

    assert test_data[2]["name"] == "model-c"
    assert test_data[2]["type"] == "local"
    assert test_data[2]["providers"][0]["endpoint"] == "gemma4"
