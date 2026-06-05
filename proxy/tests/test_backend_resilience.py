import asyncio
import json

import httpx
import pytest
from fastapi import HTTPException

import proxy.server as server

pytestmark = pytest.mark.refactor_parity


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
        "backend_recovery_state",
        {
            "in_progress": False,
            "attempt_timestamps": [],
            "max_attempts": 3,
            "window_seconds": 300,
            "retry_after_seconds": 30,
            "last_failure": None,
        },
    )
    monkeypatch.setattr(
        server,
        "config",
        {
            "server": {
                "backend_retry_attempts": 3,
                "backend_retry_base_delay_seconds": 0,
                "backend_retry_max_delay_seconds": 0,
                "backend_retry_jitter_ratio": 0,
                "llama_self_heal_max_attempts": 3,
                "llama_self_heal_window_seconds": 300,
                "llama_self_heal_backoff_base_seconds": 1,
                "llama_self_heal_retry_after_seconds": 30,
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


@pytest.mark.asyncio
async def test_model_loading_response_is_machine_readable(monkeypatch):
    monkeypatch.setattr(server, "config", {"server": {"model_loading_retry_after": 17}})
    monkeypatch.setattr(server, "current_model", None)
    monkeypatch.setattr(server, "llama_process", None)

    resp = server._model_loading_response(
        requested_model="gemma4",
        target_model="gemma4",
        scheduled=True,
        endpoint="/v1/chat/completions",
    )

    body = resp.body.decode("utf-8")
    data = json.loads(body)

    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "17"
    assert data["error"]["type"] == "model_loading"
    assert data["error"]["code"] == "model_loading"
    assert data["status"] == 503
    assert data["requested_model"] == "gemma4"
    assert data["target_model"] == "gemma4"
    assert data["scheduled"] is True
    assert data["endpoint"] == "/v1/chat/completions"
    assert data["retry_after"] == 17


@pytest.mark.asyncio
async def test_backend_watchdog_marks_backend_degraded_on_exit(monkeypatch):
    class Proc:
        def poll(self):
            return 7

    monkeypatch.setattr(server, "llama_process", Proc())
    monkeypatch.setattr(server, "current_model", "qwen3")
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "config", {"server": {"llama_router_mode": False, "llama_watchdog_interval_seconds": 0}})

    sleep_calls = {"n": 0}

    async def fake_sleep(_delay):
        sleep_calls["n"] += 1
        if sleep_calls["n"] > 1:
            raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await server._backend_watchdog_loop()

    assert server.backend_ready is False
    assert server.llama_process is None
    assert server.current_model is None
    assert server.backend_signal_counts["other_failures"] == 1


@pytest.mark.asyncio
async def test_health_endpoint_degraded_when_backend_probe_fails(monkeypatch):
    class Proc:
        def poll(self):
            return None

    async def fake_probe(_port):
        return False

    monkeypatch.setattr(server, "llama_process", Proc())
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "current_model", "qwen3")
    monkeypatch.setattr(server, "config", {"server": {"llama_router_mode": False, "llama_server_port": 8080}})
    monkeypatch.setattr(server, "_probe_backend_reachable", fake_probe)

    health = await server.health_check()

    assert health["status"] == "degraded"
    assert health["backend_reachable"] is False
    assert health["ready"] is False


@pytest.mark.asyncio
async def test_proxy_returns_503_during_active_self_healing(monkeypatch):
    class DummyRequest:
        headers = {}
        method = "POST"
        url = type("U", (), {"path": "/v1/completions"})

        async def body(self):
            return b'{"model":"qwen3","prompt":"hello"}'

    async def fail_call(*_args, **_kwargs):
        raise httpx.ConnectError("connect failed", request=httpx.Request("POST", "http://test"))

    monkeypatch.setattr(
        server,
        "config",
        {
            "server": {
                "llama_router_mode": False,
                "llama_server_port": 8080,
                "max_concurrent_queries": 4,
                "llama_request_timeout": 1,
                "llama_self_heal_retry_after_seconds": 30,
            }
        },
    )
    monkeypatch.setattr(server, "active_queries", 0)
    monkeypatch.setattr(server, "backend_recovery_state", {
        "in_progress": True,
        "attempt_timestamps": [],
        "max_attempts": 3,
        "window_seconds": 300,
        "retry_after_seconds": 30,
        "last_failure": "backend crashed",
    })
    monkeypatch.setattr(server, "_call_with_backend_retries", fail_call)

    response = await server.proxy_to_local(DummyRequest(), "v1/completions")

    payload = json.loads(response.body.decode("utf-8"))
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "30"
    assert payload["error"]["message"] == "Backend error detected, team is working on recovery. Please retry after 30 seconds."


@pytest.mark.asyncio
async def test_proxy_preserves_backend_error_when_not_self_healing(monkeypatch):
    class DummyRequest:
        headers = {}
        method = "POST"
        url = type("U", (), {"path": "/v1/completions"})

        async def body(self):
            return b'{"model":"qwen3","prompt":"hello"}'

    async def fail_call(*_args, **_kwargs):
        raise httpx.ConnectError("connect failed", request=httpx.Request("POST", "http://test"))

    monkeypatch.setattr(
        server,
        "config",
        {
            "server": {
                "llama_router_mode": False,
                "llama_server_port": 8080,
                "max_concurrent_queries": 4,
                "llama_request_timeout": 1,
            }
        },
    )
    monkeypatch.setattr(server, "active_queries", 0)
    monkeypatch.setattr(server, "_call_with_backend_retries", fail_call)

    response = await server.proxy_to_local(DummyRequest(), "v1/completions")
    payload = json.loads(response.body.decode("utf-8"))
    assert response.status_code == 503
    assert payload["status"] == 503
    assert payload["error"]["code"] == "backend_error"


@pytest.mark.asyncio
async def test_watchdog_detects_worker_failure_and_triggers_router_recovery(monkeypatch):
    class Proc:
        pid = 1234

        def poll(self):
            return None

    calls = {"recover": 0}

    async def fake_recover():
        calls["recover"] += 1
        return False

    sleep_calls = {"n": 0}

    async def fake_sleep(_delay):
        sleep_calls["n"] += 1
        if sleep_calls["n"] > 1:
            raise asyncio.CancelledError()

    monkeypatch.setattr(server, "llama_process", Proc())
    monkeypatch.setattr(server, "current_model", "qwen3")
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "config", {"server": {"llama_router_mode": True, "llama_watchdog_interval_seconds": 0}})
    monkeypatch.setattr(server, "_worker_process_unhealthy", lambda _proc: True)
    monkeypatch.setattr(server, "_attempt_router_self_heal", fake_recover)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await server._backend_watchdog_loop()

    assert calls["recover"] == 1
    assert server.backend_signal_counts["other_failures"] == 1


@pytest.mark.asyncio
async def test_router_self_heal_uses_exponential_backoff_and_attempt_cap(monkeypatch):
    class Proc:
        pid = 999

        def poll(self):
            return None

    start_calls = {"n": 0}
    sleep_calls = []

    def fake_start(_model):
        start_calls["n"] += 1
        return Proc()

    async def fake_wait(_timeout):
        return False

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr(server, "start_llama_server", fake_start)
    monkeypatch.setattr(server, "wait_for_llama_server", fake_wait)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(server, "config", {"server": {
        "llama_self_heal_max_attempts": 3,
        "llama_self_heal_window_seconds": 300,
        "llama_self_heal_backoff_base_seconds": 1,
        "llama_self_heal_retry_after_seconds": 30,
        "llama_startup_timeout": 1,
    }})
    monkeypatch.setattr(server, "backend_recovery_state", {
        "in_progress": False,
        "attempt_timestamps": [],
        "max_attempts": 3,
        "window_seconds": 300,
        "retry_after_seconds": 30,
        "last_failure": None,
    })

    recovered = await server._attempt_router_self_heal()

    assert recovered is False
    assert start_calls["n"] == 3
    assert sleep_calls == [1, 2]
    assert len(server.backend_recovery_state["attempt_timestamps"]) == 3
    assert server.backend_recovery_state["in_progress"] is False


@pytest.mark.asyncio
async def test_router_self_heal_recovers_backend_on_success(monkeypatch):
    class Proc:
        pid = 1001

        def poll(self):
            return None

    start_calls = {"n": 0}

    def fake_start(_model):
        start_calls["n"] += 1
        return Proc()

    async def fake_wait(_timeout):
        return True

    monkeypatch.setattr(server, "start_llama_server", fake_start)
    monkeypatch.setattr(server, "wait_for_llama_server", fake_wait)
    monkeypatch.setattr(server, "config", {"server": {
        "llama_self_heal_max_attempts": 3,
        "llama_self_heal_window_seconds": 300,
        "llama_self_heal_backoff_base_seconds": 1,
        "llama_self_heal_retry_after_seconds": 30,
        "llama_startup_timeout": 1,
    }})

    recovered = await server._attempt_router_self_heal()

    assert recovered is True
    assert start_calls["n"] == 1
    assert server.backend_ready is True
    assert server.backend_recovery_state["in_progress"] is False


@pytest.mark.asyncio
async def test_health_endpoint_healthy_when_probe_succeeds(monkeypatch):
    class Proc:
        def poll(self):
            return None

    async def fake_probe(_port):
        return True

    monkeypatch.setattr(server, "llama_process", Proc())
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "current_model", "qwen3")
    monkeypatch.setattr(server, "config", {"server": {"llama_router_mode": False, "llama_server_port": 8080}})
    monkeypatch.setattr(server, "_probe_backend_reachable", fake_probe)

    health = await server.health_check()

    assert health["status"] == "healthy"
    assert health["backend_reachable"] is True
    assert health["ready"] is True


@pytest.mark.asyncio
async def test_proxy_handles_history_mismatch_and_backend_failure(monkeypatch):
    # Prepare a session that will mismatch incoming history
    sid = "sess-abc"
    # Create an existing session with different messages
    session, created = await server.session_manager.get_or_create(sid)
    await server.session_manager.update_messages(sid, [{"role": "user", "content": "old message"}])

    class DummyRequest:
        headers = {"x-session-id": sid}
        method = "POST"
        url = type("U", (), {"path": "/v1/chat/completions"})

        async def body(self):
            return b'{"model":"qwen3","messages":[{"role":"user","content":"new message"}]}'

    async def fail_call(*_args, **_kwargs):
        raise httpx.ConnectError("connect failed", request=httpx.Request("POST", "http://test"))

    monkeypatch.setattr(
        server,
        "config",
        {
            "server": {
                "llama_router_mode": False,
                "llama_server_port": 8080,
                "max_concurrent_queries": 4,
                "llama_request_timeout": 1,
            }
        },
    )
    monkeypatch.setattr(server, "active_queries", 0)
    monkeypatch.setattr(server, "_call_with_backend_retries", fail_call)

    response = await server.proxy_to_local(DummyRequest(), "v1/chat/completions")
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 503
    assert payload["status"] == 503
    # Ensure session headers are present and show the session was re-created
    assert response.headers.get("X-Session-Id") == sid
    assert response.headers.get("X-Session-Created") == "true"
