import pytest
from unittest.mock import patch
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_llama_local_status_not_running():
    from proxy.server import app

    async def fake_query():
        return {"llama_server_running": False}

    with patch("proxy.server.query_llama_status", side_effect=fake_query):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            resp = await ac.get("/llama/local/status")
            assert resp.status_code == 200
            j = resp.json()
            assert isinstance(j.get("active_query"), bool)
            assert isinstance(j.get("model_switch_in_progress"), bool)
            assert j.get("current_model") is None
            assert j.get("llama_server_running") is False


@pytest.mark.asyncio
async def test_llama_local_status_running_and_switch():
    from proxy import server
    from proxy.server import app

    async def fake_query():
        return {"llama_server_running": True}

    # simulate model switch lock and background loads
    class DummyLock:
        def locked(self):
            return True

    with patch("proxy.server.query_llama_status", side_effect=fake_query):
        # patch the model_switch_lock and background_loads
        with patch.object(server, "model_switch_lock", DummyLock()):
            with patch.object(server, "background_loads", {"m": True}):
                # also set a current_model value
                with patch.object(server, "current_model", "test-model"):
                    async with AsyncClient(app=app, base_url="http://test") as ac:
                        resp = await ac.get("/llama/local/status")
                        assert resp.status_code == 200
                        j = resp.json()
                        assert isinstance(j.get("active_query"), bool)
                        assert j.get("model_switch_in_progress") is True
                        assert j.get("current_model") == "test-model"
                        assert j.get("llama_server_running") is True
