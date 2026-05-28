import asyncio

import httpx
import pytest
from fastapi import HTTPException

import proxy.server as server


@pytest.fixture(autouse=True)
def reset_backend_state(monkeypatch):
    monkeypatch.setattr(
        server,
        "backend_signal_counts",
        {
            "connect_failures": 0,
            "read_failures": 0,
            "timeout_failures": 0,
            "other_failures": 0,
            "concurrency_rejects": 0,
        },
    )
    monkeypatch.setattr(server, "backend_ready", False)
    monkeypatch.setattr(
        server,
        "config",
        {
            "server": {
                "backend_retry_attempts": 3,
                "backend_retry_base_delay_seconds": 0,
                "backend_retry_max_delay_seconds": 0,
                "backend_retry_jitter_ratio": 0,
            }
        },
    )


@pytest.mark.asyncio
async def test_call_with_backend_retries_retries_connect_and_succeeds(monkeypatch):
    attempts = {"n": 0}

    async def fake_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def call_factory():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("connect failed", request=httpx.Request("GET", "http://test"))
        return "ok"

    result = await server._call_with_backend_retries(call_factory, path="v1/chat/completions", stream=False)

    assert result == "ok"
    assert attempts["n"] == 3
    assert server.backend_signal_counts["connect_failures"] == 2


@pytest.mark.asyncio
async def test_call_with_backend_retries_raises_after_max_attempts(monkeypatch):
    async def fake_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def call_factory():
        raise httpx.ReadError("read failed", request=httpx.Request("POST", "http://test"))

    with pytest.raises(httpx.ReadError):
        await server._call_with_backend_retries(call_factory, path="v1/chat/completions", stream=True)

    # Two retries + final failed attempt all count as read failures
    assert server.backend_signal_counts["read_failures"] == 3
    assert server.backend_ready is False


@pytest.mark.asyncio
async def test_health_endpoint_reports_degraded_until_ready(monkeypatch):
    class Proc:
        def poll(self):
            return None

    monkeypatch.setattr(server, "llama_process", Proc())
    monkeypatch.setattr(server, "backend_ready", False)
    monkeypatch.setattr(server, "current_model", "qwen3")
    monkeypatch.setattr(server, "config", {"server": {"llama_router_mode": False}})

    health = await server.health_check()

    assert health["status"] == "degraded"
    assert health["ready"] is False
    assert health["llama_server_running"] is True
    assert "backend_signals" in health


@pytest.mark.asyncio
async def test_proxy_to_local_concurrency_reject_increments_signal(monkeypatch):
    class DummyRequest:
        headers = {}
        method = "POST"
        url = type("U", (), {"path": "/v1/chat/completions"})

        async def body(self):
            return b'{"model":"qwen3","messages":[{"role":"user","content":"hello"}]}'

    monkeypatch.setattr(
        server,
        "config",
        {
            "server": {
                "llama_router_mode": False,
                "llama_server_port": 8080,
                "max_concurrent_queries": 1,
            }
        },
    )
    monkeypatch.setattr(server, "active_queries", 1)

    with pytest.raises(HTTPException) as excinfo:
        await server.proxy_to_local(DummyRequest(), "v1/chat/completions")

    assert excinfo.value.status_code == 503
    assert server.backend_signal_counts["concurrency_rejects"] == 1
