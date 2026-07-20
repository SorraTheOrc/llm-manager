"""Integration tests for GET /llama/local/status.

These tests exercise the actual handler (via ASGI transport) with real
server state, simulating model-switch and active-query scenarios.
"""
import asyncio
import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest

pytestmark = pytest.mark.refactor_parity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class UncontestedLock:
    """An async context manager that does not block — simulates an unlocked lock."""

    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """Import the FastAPI app (lazy to avoid circular imports)."""
    from proxy.server import app
    return app


@pytest.fixture
def transport(app):
    """ASGI transport for the proxy app."""
    return httpx.ASGITransport(app=app)


# ---------------------------------------------------------------------------
# Model-switch integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_shows_model_switch_in_progress(transport):
    """When a background model load is scheduled, status reports switch in progress."""
    from proxy import server as srv_module

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as ac:
        mock_lock = AsyncMock()
        mock_lock.locked.return_value = False

        with patch.object(srv_module, "model_switch_refcount", 1):
            with patch.object(srv_module, "model_switch_lock", mock_lock):
                with patch.object(srv_module, "background_loads", {"dummy_model": True}):
                    with patch.object(srv_module, "current_model", "test-model"):
                        with patch.object(
                            srv_module, "query_llama_status", new_callable=AsyncMock
                        ) as mock_qls:
                            mock_qls.return_value = {"llama_server_running": True}
                            resp = await ac.get("/llama/local/status")

        assert resp.status_code == 200
        j = resp.json()
        assert j["model_switch_in_progress"] is True
        assert j["llama_server_running"] is True
        assert j["current_model"] == "test-model"


@pytest.mark.asyncio
async def test_status_shows_model_switch_via_lock(transport):
    """When model_switch_lock is locked (but refcount is 0), status reports switch in progress."""
    from proxy import server as srv_module

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as ac:
        mock_lock = AsyncMock()
        mock_lock.locked.return_value = True

        with patch.object(srv_module, "model_switch_refcount", 0):
            with patch.object(srv_module, "model_switch_lock", mock_lock):
                with patch.object(srv_module, "background_loads", {}):
                    with patch.object(srv_module, "current_model", "switching-model"):
                        with patch.object(
                            srv_module, "query_llama_status", new_callable=AsyncMock
                        ) as mock_qls:
                            mock_qls.return_value = {"llama_server_running": True}
                            resp = await ac.get("/llama/local/status")

        assert resp.status_code == 200
        j = resp.json()
        assert j["model_switch_in_progress"] is True


# ---------------------------------------------------------------------------
# Active-query integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_shows_active_query(transport):
    """While a request is in-flight, status reports active_query=True."""
    from proxy import server as srv_module

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as ac:
        with patch.object(srv_module, "active_queries", 1):
            with patch.object(srv_module, "active_queries_lock", UncontestedLock()):
                with patch.object(srv_module, "model_switch_refcount", 0):
                    with patch.object(srv_module, "model_switch_lock", AsyncMock()):
                        with patch.object(srv_module, "background_loads", {}):
                            with patch.object(srv_module, "current_model", "active-model"):
                                with patch.object(
                                    srv_module, "query_llama_status", new_callable=AsyncMock
                                ) as mock_qls:
                                    mock_qls.return_value = {"llama_server_running": True}
                                    resp = await ac.get("/llama/local/status")

        assert resp.status_code == 200
        j = resp.json()
        assert j["active_query"] is True


@pytest.mark.asyncio
async def test_status_no_active_query(transport):
    """When no queries are active, status reports active_query=False."""
    from proxy import server as srv_module

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as ac:
        with patch.object(srv_module, "active_queries", 0):
            with patch.object(srv_module, "active_queries_lock", UncontestedLock()):
                with patch.object(srv_module, "model_switch_refcount", 0):
                    with patch.object(srv_module, "model_switch_lock", AsyncMock()):
                        with patch.object(srv_module, "background_loads", {}):
                            with patch.object(srv_module, "current_model", None):
                                with patch.object(
                                    srv_module, "query_llama_status", new_callable=AsyncMock
                                ) as mock_qls:
                                    mock_qls.return_value = {"llama_server_running": True}
                                    resp = await ac.get("/llama/local/status")

        assert resp.status_code == 200
        j = resp.json()
        assert j["active_query"] is False


# ---------------------------------------------------------------------------
# Server-not-running integration test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_server_not_running(transport):
    """When llama-server is not running, status reflects that."""
    from proxy import server as srv_module

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as ac:
        _mock_lock = AsyncMock()

        # The handler tries 'async with active_queries_lock'
        # and 'model_switch_lock.locked()'
        with patch.object(srv_module, "query_llama_status", new_callable=AsyncMock) as mock_qls:
            mock_qls.return_value = {"llama_server_running": False}
            resp = await ac.get("/llama/local/status")

        assert resp.status_code == 200
        j = resp.json()
        assert j["llama_server_running"] is False
        assert j["active_query"] is False
        assert j["model_switch_in_progress"] is False
        assert j["current_model"] is None


# ---------------------------------------------------------------------------
# Timeout/fallback integration test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_query_timeout_fallback(transport):
    """When query_llama_status times out, status still returns with safe defaults."""
    from proxy import server as srv_module

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as ac:
        with patch.object(
            srv_module, "query_llama_status", new_callable=AsyncMock
        ) as mock_qls:
            mock_qls.side_effect = TimeoutError("slow")
            resp = await ac.get("/llama/local/status")

        assert resp.status_code == 200
        j = resp.json()
        assert j["llama_server_running"] is False
        # Other fields should be safe defaults
        assert isinstance(j["active_query"], bool)
        assert isinstance(j["model_switch_in_progress"], bool)
        assert j["current_model"] is None


# ---------------------------------------------------------------------------
# Concurrency / non-blocking tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_status_requests(transport):
    """Multiple concurrent GET /llama/local/status requests must all complete within 5 s."""
    from proxy import server as srv_module

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as ac:
        with patch.object(
            srv_module, "query_llama_status", new_callable=AsyncMock
        ) as mock_qls:
            mock_qls.return_value = {"llama_server_running": True}

            # Launch N concurrent requests
            N = 8
            start = asyncio.get_event_loop().time()
            results = await asyncio.wait_for(
                asyncio.gather(*[ac.get("/llama/local/status") for _ in range(N)]),
                timeout=5.0,
            )
            elapsed = asyncio.get_event_loop().time() - start

            assert len(results) == N
            for resp in results:
                assert resp.status_code == 200
                j = resp.json()
                assert "llama_server_running" in j
                assert "active_query" in j
                assert "model_switch_in_progress" in j
                assert "current_model" in j

            assert elapsed < 5.0, (
                f"Concurrent requests took {elapsed:.2f}s, expected < 5.0s"
            )


@pytest.mark.asyncio
async def test_status_response_under_query_load(transport):
    """When query_llama_status is slow, the endpoint still responds within 5 s."""
    from proxy import server as srv_module

    async def slow_query():
        await asyncio.sleep(0.5)
        return {"llama_server_running": True}

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as ac:
        with patch.object(
            srv_module, "query_llama_status", side_effect=slow_query
        ):
            with patch.object(
                srv_module, "active_queries_lock", UncontestedLock()
            ):
                start = asyncio.get_event_loop().time()
                resp = await asyncio.wait_for(
                    ac.get("/llama/local/status"), timeout=5.0
                )
                elapsed = asyncio.get_event_loop().time() - start

                assert resp.status_code == 200
                j = resp.json()
                assert j["llama_server_running"] is True
                assert elapsed < 5.0, (
                    f"Response took {elapsed:.2f}s, expected < 5.0s"
                )


@pytest.mark.asyncio
async def test_status_request_logged(caplog, transport):
    """Each call to /llama/local/status emits a structured log entry with latency."""
    from proxy import server as srv_module

    caplog.set_level(logging.INFO, logger="llama-proxy")

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as ac:
        with patch.object(
            srv_module, "query_llama_status", new_callable=AsyncMock
        ) as mock_qls:
            mock_qls.return_value = {"llama_server_running": True}
            resp = await ac.get("/llama/local/status")

        assert resp.status_code == 200

    # Verify a structured log entry was emitted
    assert any("status_request" in record.getMessage() for record in caplog.records), (
        "Expected 'status_request' in log output"
    )
    # Verify at least one record has the latency_ms extra field
    assert any(
        hasattr(record, "latency_ms") for record in caplog.records
    ), "Expected log record with latency_ms attribute"


@pytest.mark.asyncio
async def test_status_returns_slot_fields(transport):
    """Response includes available_slots and total_slots (integers)."""
    from proxy import server as srv_module

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as ac:
        with patch.object(
            srv_module, "query_llama_status", new_callable=AsyncMock
        ) as mock_qls:
            # When server is running the slots query is attempted (may fail in test)
            mock_qls.return_value = {"llama_server_running": True}
            resp = await ac.get("/llama/local/status")

        assert resp.status_code == 200
        j = resp.json()
        assert isinstance(j.get("available_slots"), int)
        assert isinstance(j.get("total_slots"), int)
        assert j["available_slots"] >= 0
        assert j["total_slots"] >= 0
