import asyncio
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_ensure_model_loaded_emits_switching_then_ready():
    from proxy import server

    events = []

    async def fake_broadcast(event_type, data):
        events.append((event_type, data))

    class DummyLock:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    with patch.object(server, "config", {"server": {"llama_router_mode": False}}):
        with patch.object(server, "model_switch_lock", DummyLock()):
            with patch.object(server, "broadcast_status", side_effect=fake_broadcast):
                with patch.object(server, "start_llama_server", return_value=object()):
                    with patch.object(server, "wait_for_llama_server", new=AsyncMock(return_value=True)):
                        with patch.object(server, "get_local_model_name", return_value="llama-7b"):
                            server.current_model = "old-model"
                            ok = await server.ensure_model_loaded("new-model")

    assert ok is True
    assert events[0][0] == "switching"
    assert events[0][1]["target_model"] == "llama-7b"
    assert events[0][1]["previous_model"] == "old-model"
    assert events[-1][0] == "ready"
    assert events[-1][1]["current_model"] == "llama-7b"
    assert server.current_model == "llama-7b"


@pytest.mark.asyncio
async def test_ensure_model_loaded_does_not_update_current_model_on_failure():
    from proxy import server

    events = []

    async def fake_broadcast(event_type, data):
        events.append((event_type, data))

    class DummyLock:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    with patch.object(server, "config", {"server": {"llama_router_mode": False}}):
        with patch.object(server, "model_switch_lock", DummyLock()):
            with patch.object(server, "broadcast_status", side_effect=fake_broadcast):
                with patch.object(server, "start_llama_server", return_value=object()):
                    with patch.object(server, "wait_for_llama_server", new=AsyncMock(return_value=False)):
                        with patch.object(server, "get_local_model_name", return_value="llama-7b"):
                            server.current_model = "old-model"
                            ok = await server.ensure_model_loaded("new-model")

    assert ok is False
    assert events[0][0] == "switching"
    assert events[-1][0] == "error"
    assert server.current_model == "old-model"
