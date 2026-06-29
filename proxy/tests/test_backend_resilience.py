import asyncio
import json

import httpx
import pytest
from fastapi import HTTPException

import proxy.server as server
import proxy.backend_health as backend_health

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
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "llama_process", type("P", (), {"poll": lambda s: None, "pid": 1})())

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
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "llama_process", type("P", (), {"poll": lambda s: None, "pid": 1})())
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
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "llama_process", type("P", (), {"poll": lambda s: None, "pid": 1})())
    monkeypatch.setattr(server, "_call_with_backend_retries", fail_call)

    response = await server.proxy_to_local(DummyRequest(), "v1/chat/completions")
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 503
    assert payload["status"] == 503
    # Ensure session headers are present and show the session was re-created
    assert response.headers.get("X-Session-Id") == sid
    assert response.headers.get("X-Session-Created") == "true"


# ---------------------------------------------------------------------------
# LP-0MQ4GQ2LO005PZPY: Eliminate 500s during backend recovery window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_returns_503_when_backend_ready_is_false(monkeypatch):
    """proxy_to_local returns 503 immediately when backend_ready is False
    without attempting any backend connection."""
    connect_attempted = []

    class DummyRequest:
        headers = {}
        method = "POST"
        url = type("U", (), {"path": "/v1/chat/completions"})

        async def body(self):
            return b'{"model":"qwen3","messages":[{"role":"user","content":"hi"}]}'

    async def should_not_be_called(*_args, **_kwargs):
        connect_attempted.append(True)
        raise RuntimeError("should not connect to backend")

    monkeypatch.setattr(server, "backend_ready", False)
    monkeypatch.setattr(server, "llama_process", None)
    monkeypatch.setattr(server, "active_queries", 0)
    monkeypatch.setattr(server, "_call_with_backend_retries", should_not_be_called)
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

    response = await server.proxy_to_local(DummyRequest(), "v1/chat/completions")
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 503
    assert payload["status"] == 503
    assert "backend" in payload["error"]["type"].lower() or "unavailable" in payload["error"]["message"].lower()
    assert connect_attempted == [], "backend should not have been contacted"


@pytest.mark.asyncio
async def test_proxy_returns_503_when_llama_process_is_none(monkeypatch):
    """proxy_to_local returns 503 immediately when llama_process is None
    even if backend_ready hasn't been set to False yet."""
    connect_attempted = []

    class DummyRequest:
        headers = {}
        method = "POST"
        url = type("U", (), {"path": "/v1/chat/completions"})

        async def body(self):
            return b'{"model":"qwen3","messages":[{"role":"user","content":"hi"}]}'

    async def should_not_be_called(*_args, **_kwargs):
        connect_attempted.append(True)
        raise RuntimeError("should not connect to backend")

    monkeypatch.setattr(server, "backend_ready", True)  # not yet updated by watchdog
    monkeypatch.setattr(server, "llama_process", None)
    monkeypatch.setattr(server, "active_queries", 0)
    monkeypatch.setattr(server, "_call_with_backend_retries", should_not_be_called)
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

    response = await server.proxy_to_local(DummyRequest(), "v1/chat/completions")
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 503
    assert payload["status"] == 503
    assert connect_attempted == [], "backend should not have been contacted"


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_watchdog_retries_restart_when_process_is_none(monkeypatch):
    """Watchdog attempts restart even when llama_process is None
    (i.e. after a failed self-heal or initial crash)."""
    restart_calls = []

    def fake_start(model=None):
        restart_calls.append(model)
        fake_proc = type("P", (), {"poll": lambda self: None, "pid": 999})()
        return fake_proc

    async def fake_wait(timeout):
        return True

    monkeypatch.setattr(server, "start_llama_server", fake_start)
    monkeypatch.setattr(server, "wait_for_llama_server", fake_wait)
    monkeypatch.setattr(server, "llama_process", None)
    monkeypatch.setattr(server, "current_model", None)
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
                "llama_router_mode": True,
                "llama_watchdog_interval_seconds": 0.01,
                "llama_self_heal_max_attempts": 2,
                "llama_self_heal_window_seconds": 300,
                "llama_self_heal_backoff_base_seconds": 0,
                "llama_startup_timeout": 5,
            }
        },
    )
    monkeypatch.setattr(server, "logger", type("L", (), {
        "error": lambda *a, **kw: None,
        "warning": lambda *a, **kw: None,
        "info": lambda *a, **kw: None,
        "exception": lambda *a, **kw: None,
    })())
    monkeypatch.setattr(server, "_record_backend_signal", lambda *a: None)

    # Run watchdog for a short time
    task = asyncio.create_task(server._backend_watchdog_loop())
    await asyncio.sleep(0.15)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Watchdog should have attempted restart even though process was None
    assert len(restart_calls) >= 1, f"Expected at least 1 restart attempt, got {len(restart_calls)}"


@pytest.mark.asyncio
async def test_router_model_health_loop_uses_legacy_interval_key(monkeypatch):
    """Model health loop should honor legacy llama_health_check_interval when
    llama_model_health_interval_seconds is not set."""

    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)
        raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        server,
        "config",
        {"server": {"llama_router_mode": True, "llama_health_check_interval": 12}},
    )

    await server._router_model_health_loop()

    assert sleep_calls, "expected at least one sleep call"
    assert sleep_calls[0] == 12.0


@pytest.mark.asyncio
async def test_router_model_health_requires_consecutive_failures_before_recovery(monkeypatch):
    """Single failed probe should not immediately unload/reload a model."""

    sleep_calls = {"n": 0}

    async def fake_sleep(_delay):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 2:
            raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        server,
        "config",
        {
            "server": {
                "llama_router_mode": True,
                "llama_model_health_interval_seconds": 0,
                "llama_model_health_failures_before_recovery": 2,
                "llama_model_health_grace_period_seconds": 0,
                "llama_model_health_probe_attempts": 1,
            }
        },
    )
    monkeypatch.setattr(server, "backend_recovery_state", {"in_progress": False})

    async def fake_router_list_models():
        return {
            "data": [
                {
                    "id": "Qwen3",
                    "status": {"value": "loaded", "args": ["--port", "7777"]},
                }
            ]
        }

    probe_calls = {"n": 0}

    async def fake_probe_with_retries(*_args, **_kwargs):
        probe_calls["n"] += 1
        return False

    unload_calls = {"n": 0}

    class FakeClient:
        async def post(self, *args, **kwargs):
            unload_calls["n"] += 1
            return type("Resp", (), {"status_code": 200})()

    load_calls = {"n": 0}

    async def fake_router_load_model(_model):
        load_calls["n"] += 1
        return True

    monkeypatch.setattr(server, "router_list_models", fake_router_list_models)
    monkeypatch.setattr(backend_health, "_probe_model_instance_with_retries", fake_probe_with_retries)
    monkeypatch.setattr(server, "_http_client", FakeClient())
    monkeypatch.setattr(server, "router_load_model", fake_router_load_model)

    await server._router_model_health_loop()

    assert probe_calls["n"] == 1
    assert unload_calls["n"] == 0
    assert load_calls["n"] == 0


@pytest.mark.asyncio
async def test_router_model_health_recovers_after_failure_threshold(monkeypatch):
    """Recovery should trigger once the consecutive failure threshold is met."""

    sleep_calls = {"n": 0}

    async def fake_sleep(_delay):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 3:
            raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        server,
        "config",
        {
            "server": {
                "llama_router_mode": True,
                "llama_model_health_interval_seconds": 0,
                "llama_model_health_failures_before_recovery": 2,
                "llama_model_health_grace_period_seconds": 0,
                "llama_model_health_probe_attempts": 1,
            }
        },
    )
    monkeypatch.setattr(server, "backend_recovery_state", {"in_progress": False})

    async def fake_router_list_models():
        return {
            "data": [
                {
                    "id": "Qwen3",
                    "status": {"value": "loaded", "args": ["--port", "8888"]},
                }
            ]
        }

    async def fake_probe_with_retries(*_args, **_kwargs):
        return False

    unload_calls = {"n": 0}

    class FakeClient:
        async def post(self, *args, **kwargs):
            unload_calls["n"] += 1
            return type("Resp", (), {"status_code": 200})()

    load_calls = {"n": 0}

    async def fake_router_load_model(_model):
        load_calls["n"] += 1
        return True

    monkeypatch.setattr(server, "router_list_models", fake_router_list_models)
    monkeypatch.setattr(backend_health, "_probe_model_instance_with_retries", fake_probe_with_retries)
    monkeypatch.setattr(server, "_http_client", FakeClient())
    monkeypatch.setattr(server, "router_load_model", fake_router_load_model)

    await server._router_model_health_loop()

    assert unload_calls["n"] == 1
    assert load_calls["n"] == 1


# ---------------------------------------------------------------------------
# LP-0MQPUCOFJ000QIRF: Release session/slot when upstream returns error (429/5xx)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_429_decrements_active_queries(monkeypatch):
    """When a streaming request receives an HTTP 429 from upstream,
    _decrement_active_queries must be called before returning the error,
    so that subsequent requests are not blocked by a leaked counter."""

    class MockResponse:
        status_code = 429
        headers = {"content-type": "text/plain"}

        async def aread(self):
            return b'{"error":"rate limited"}'

    class MockCM:
        async def __aexit__(self, *args):
            pass

    async def mock_call(*_args, **_kwargs):
        return MockCM(), MockResponse()

    class DummyRequest:
        headers = {}
        method = "POST"
        url = type("U", (), {"path": "/v1/chat/completions"})

        async def body(self):
            return b'{"model":"qwen3","messages":[{"role":"user","content":"hi"}],"stream":true}'

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
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "llama_process", type("P", (), {"poll": lambda s: None, "pid": 1})())
    monkeypatch.setattr(server, "_call_with_backend_retries", mock_call)

    response = await server.proxy_to_local(DummyRequest(), "v1/chat/completions")

    assert response.status_code == 429
    # Verify active_queries was decremented back to 0 after increment + decrement
    assert server.active_queries == 0


@pytest.mark.asyncio
async def test_streaming_500_decrements_active_queries(monkeypatch):
    """When a streaming request receives an HTTP 500 from upstream,
    _decrement_active_queries must be called before returning the error."""

    class MockResponse:
        status_code = 500
        headers = {"content-type": "text/plain"}

        async def aread(self):
            return b'{"error":"internal error"}'

    class MockCM:
        async def __aexit__(self, *args):
            pass

    async def mock_call(*_args, **_kwargs):
        return MockCM(), MockResponse()

    class DummyRequest:
        headers = {}
        method = "POST"
        url = type("U", (), {"path": "/v1/chat/completions"})

        async def body(self):
            return b'{"model":"qwen3","messages":[{"role":"user","content":"hi"}],"stream":true}'

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
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "llama_process", type("P", (), {"poll": lambda s: None, "pid": 1})())
    monkeypatch.setattr(server, "_call_with_backend_retries", mock_call)

    response = await server.proxy_to_local(DummyRequest(), "v1/chat/completions")

    assert response.status_code == 500
    # Verify active_queries was decremented back to 0
    assert server.active_queries == 0


@pytest.mark.asyncio
async def test_streaming_429_does_not_block_concurrent_request(monkeypatch):
    """When the first streaming request gets a 429 from upstream,
    a second concurrent request must NOT be rejected with 503
    due to the leaked active_queries counter."""

    class MockResponse:
        status_code = 429
        headers = {"content-type": "text/plain"}

        async def aread(self):
            return b'{"error":"rate limited"}'

    class MockCM:
        async def __aexit__(self, *args):
            pass

    async def mock_call(*_args, **_kwargs):
        return MockCM(), MockResponse()

    class DummyRequest:
        headers = {}
        method = "POST"
        url = type("U", (), {"path": "/v1/chat/completions"})

        async def body(self):
            return b'{"model":"qwen3","messages":[{"role":"user","content":"hi"}],"stream":true}'

    monkeypatch.setattr(
        server,
        "config",
        {
            "server": {
                "llama_router_mode": False,
                "llama_server_port": 8080,
                "max_concurrent_queries": 1,
                "llama_request_timeout": 1,
            }
        },
    )
    monkeypatch.setattr(server, "active_queries", 0)
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "llama_process", type("P", (), {"poll": lambda s: None, "pid": 1})())
    monkeypatch.setattr(server, "_call_with_backend_retries", mock_call)

    # First request should get 429 but NOT leak the counter
    response1 = await server.proxy_to_local(DummyRequest(), "v1/chat/completions")
    assert response1.status_code == 429
    assert server.active_queries == 0, "active_queries leaked after 429"

    # Second request should succeed (not get 503) because counter was released
    response2 = await server.proxy_to_local(DummyRequest(), "v1/chat/completions")
    assert response2.status_code == 429  # Still 429 from upstream, but NOT 503
    assert server.active_queries == 0, "active_queries leaked after second 429"
