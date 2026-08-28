"""Tests for bypassing the slow router /slots?model= availability check.

LP-0MTDGBRPU003Z7KU: the llama-server ROUTER (port 8080) serializes
``GET /slots?model=Qwen3`` behind the busy Qwen3 child (port 58113)
generation loop (measured 5-7s vs 0.17s direct). The proxy made this call on
EVERY local request via the shared _http_client pool, so under multi-session
load the pool exhausted -> slot_save PoolTimeout (up to 60s) -> client-visible
"Request timed out".

The fix (LP-0MTDH2Q8C001WEGU):
- ``_discover_local_child_port()`` parses the llama-server log spawn line
  (``spawning server instance with name=Qwen3 on port <port>``), cached per
  llama-server process, so availability checks can target the child directly.
- ``_check_slot_availability()`` queries the child port when discovered
  (falling back to the router port) and is skipped entirely when the caller
  already holds a dispatch lease.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from proxy.router_helpers import (
    _build_slot_exhaustion_response,
    _check_slot_availability,
    _discover_local_child_port,
)


@pytest.fixture(autouse=True)
def _clear_child_port_cache():
    """Clear the module-level child-port cache between tests."""
    from proxy import router_helpers as rh

    rh._child_port_cache.clear()
    yield
    rh._child_port_cache.clear()


# ======================================================================
# Child-port discovery
# ======================================================================


class TestDiscoverLocalChildPort:
    """_discover_local_child_port parses the llama-server log spawn line."""

    def test_parses_qwen3_spawn_line(self, tmp_path):
        """The Qwen3 spawn line yields the child port."""
        log = tmp_path / "llama-server.log"
        log.write_text(
            "srv          load: spawning server instance with name=Qwen3 on port 58113\n"
            "[58113] main: model loaded\n"
        )
        srv = SimpleNamespace(log_dir=tmp_path, llama_process=SimpleNamespace(pid=123))
        port = _discover_local_child_port(srv)
        assert port == 58113

    def test_returns_none_when_log_missing(self, tmp_path):
        """Missing log file -> None (caller falls back to the router port)."""
        srv = SimpleNamespace(log_dir=tmp_path, llama_process=SimpleNamespace(pid=123))
        assert _discover_local_child_port(srv) is None

    def test_returns_none_when_no_spawn_line(self, tmp_path):
        """Log without a spawn line -> None."""
        log = tmp_path / "llama-server.log"
        log.write_text("[58113] main: model loaded\n")
        srv = SimpleNamespace(log_dir=tmp_path, llama_process=SimpleNamespace(pid=123))
        assert _discover_local_child_port(srv) is None

    def test_cached_per_process(self, tmp_path):
        """The port is cached per llama-server pid and not re-parsed."""
        log = tmp_path / "llama-server.log"
        log.write_text("srv          load: spawning server instance with name=Qwen3 on port 58113\n")
        srv = SimpleNamespace(log_dir=tmp_path, llama_process=SimpleNamespace(pid=123))
        assert _discover_local_child_port(srv) == 58113
        # Mutate the log; cache should still return the original port for the
        # same pid (the port is stable for the lifetime of the process).
        log.write_text("garbage\n")
        assert _discover_local_child_port(srv) == 58113

    def test_new_pid_reparses(self, tmp_path):
        """A new llama-server pid invalidates the cache and re-parses."""
        log = tmp_path / "llama-server.log"
        log.write_text("srv          load: spawning server instance with name=Qwen3 on port 58113\n")
        srv = SimpleNamespace(log_dir=tmp_path, llama_process=SimpleNamespace(pid=123))
        assert _discover_local_child_port(srv) == 58113
        log.write_text("srv          load: spawning server instance with name=Qwen3 on port 59000\n")
        srv.llama_process = SimpleNamespace(pid=456)
        assert _discover_local_child_port(srv) == 59000

    def test_falls_back_to_log_dir_default(self, tmp_path):
        """When srv.log_dir is unset, the helper uses the default logs dir."""
        from proxy import server as srv_module

        default_dir = Path(__file__).parent.parent / "proxy" / "logs"
        default_dir.mkdir(parents=True, exist_ok=True)
        marker = default_dir / "llama-server.log"
        original = marker.read_bytes() if marker.exists() else None
        try:
            marker.write_text(
                "srv          load: spawning server instance with name=Qwen3 on port 58113\n"
            )
            srv = SimpleNamespace(log_dir=None, llama_process=SimpleNamespace(pid=999))
            assert _discover_local_child_port(srv) == 58113
        finally:
            if original is None:
                marker.unlink(missing_ok=True)
            else:
                marker.write_bytes(original)


# ======================================================================
# _check_slot_availability targets the child port and honours lease-hold
# ======================================================================


class TestCheckSlotAvailability:
    """_check_slot_availability queries the child port and skips on lease hold."""

    def _make_srv(self, http_client=None):
        proc = SimpleNamespace(pid=123)
        return SimpleNamespace(
            log_dir=Path("/nonexistent"),
            llama_process=proc,
            current_model="Qwen3",
            _http_client=http_client,
        )

    def _mock_resp(self, slots):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = slots
        return resp

    @pytest.mark.asyncio
    async def test_queries_child_port_when_discovered(self, monkeypatch):
        """With a discovered child port, the /slots call targets the child."""
        from proxy import router_helpers as rh

        captured = {}

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, timeout=None):
                captured["url"] = url
                return self._resp

        client = _Client()
        client._resp = self._mock_resp(
            [{"id": 0, "is_processing": False}, {"id": 1, "is_processing": False}]
        )
        srv = self._make_srv()
        monkeypatch.setattr(
            rh, "httpx",
            SimpleNamespace(AsyncClient=lambda timeout: client, Timeout=lambda t: t),
        )
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda s: 58113)

        result = await _check_slot_availability(
            srv, {"llama_server_port": 8080}, 8080, "Qwen3", "Qwen3",
            "v1/chat/completions",
        )
        assert result is None  # slots available -> no 503
        assert captured["url"] == "http://localhost:58113/slots?model=Qwen3"

    @pytest.mark.asyncio
    async def test_returns_503_when_all_slots_busy_via_child_port(self, monkeypatch):
        """All slots busy on the child -> 503 slot-exhaustion response."""
        from proxy import router_helpers as rh

        captured = {}

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, timeout=None):
                captured["url"] = url
                return self._resp

        client = _Client()
        client._resp = self._mock_resp(
            [
                {"id": 0, "is_processing": True},
                {"id": 1, "is_processing": True},
                {"id": 2, "is_processing": True},
            ]
        )
        srv = self._make_srv()
        monkeypatch.setattr(
            rh, "httpx",
            SimpleNamespace(AsyncClient=lambda timeout: client, Timeout=lambda t: t),
        )
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda s: 58113)

        result = await _check_slot_availability(
            srv, {"llama_server_port": 8080}, 8080, "Qwen3", "Qwen3",
            "v1/chat/completions",
        )
        assert result is not None
        assert result.status_code == 503
        assert captured["url"] == "http://localhost:58113/slots?model=Qwen3"

    @pytest.mark.asyncio
    async def test_falls_back_to_router_port_without_discovery(self, monkeypatch):
        """No child-port discovery -> falls back to the router port."""
        from proxy import router_helpers as rh

        captured = {}

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, timeout=None):
                captured["url"] = url
                return self._resp

        client = _Client()
        client._resp = self._mock_resp(
            [{"id": 0, "is_processing": False}]
        )
        srv = self._make_srv()
        monkeypatch.setattr(
            rh, "httpx",
            SimpleNamespace(AsyncClient=lambda timeout: client, Timeout=lambda t: t),
        )
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda s: None)

        result = await _check_slot_availability(
            srv, {"llama_server_port": 8080}, 8080, "Qwen3", "Qwen3",
            "v1/chat/completions",
        )
        assert result is None
        assert captured["url"] == "http://localhost:8080/slots?model=Qwen3"

    @pytest.mark.asyncio
    async def test_skip_when_lease_held_does_not_query(self):
        """lease_held=True short-circuits without any HTTP call."""
        srv = self._make_srv()
        result = await _check_slot_availability(
            srv, {"llama_server_port": 8080}, 8080, "Qwen3", "Qwen3",
            "v1/chat/completions", lease_held=True,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_non_chat_completions_path_returns_none(self):
        """Non chat/completions paths are never checked."""
        srv = self._make_srv()
        result = await _check_slot_availability(
            srv, {"llama_server_port": 8080}, 8080, "Qwen3", "Qwen3",
            "v1/models", lease_held=False,
        )
        assert result is None


# ======================================================================
# proxy_to_local wiring: lease-holding explicit sessions skip the check
# ======================================================================


class TestProxyToLocalLeaseWiring:
    """proxy_to_local passes lease_held=True only after lease acquisition."""

    def _make_request(self):
        body = json.dumps({
            "model": "Qwen3",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }).encode()
        req = SimpleNamespace(
            method="POST",
            headers={
                "content-type": "application/json",
                "x-session-id": "test-sess-1",
            },
            url=SimpleNamespace(path="/v1/chat/completions"),
        )
        req.body = AsyncMock(return_value=body)
        return req

    def _setup_common(self, monkeypatch, explicit: bool):
        """Shared server/global state setup; returns the fake slot-check recorder.

        All ``proxy.server`` globals are patched via ``monkeypatch.setattr`` so
        pytest auto-restores them after each test — direct assignment would
        leak state into later tests in the same session (e.g. logging
        verbosity and config-dependent tests).
        """
        from proxy import router as router_mod
        from proxy import server as srv_module

        proc = MagicMock()
        proc.poll.return_value = None
        monkeypatch.setattr(srv_module, "config", {
            "server": {
                "llama_server_port": 8080,
                "session_slot_pool_size": 1,
                "max_concurrent_queries": 16,
            }
        })
        monkeypatch.setattr(srv_module, "llama_process", proc)
        monkeypatch.setattr(srv_module, "backend_ready", True)
        monkeypatch.setattr(srv_module, "current_model", "Qwen3")
        monkeypatch.setattr(srv_module, "logger", MagicMock())
        monkeypatch.setattr(srv_module, "_http_client", None)

        session = SimpleNamespace(session_id="test-sess-1", message_count=0)
        monkeypatch.setattr(
            srv_module,
            "session_manager",
            SimpleNamespace(get_or_create=AsyncMock(return_value=(session, True))),
        )

        async def _fake_handle_session(srv, body_json, server_config, headers):
            return {
                "session_id": "test-sess-1",
                "session_created": True,
                "is_delta_request": False,
                "session_fallback_reason": None,
                "delta_messages": None,
                "original_message_count": 1,
                "body_json": body_json,
                "body_override": None,
                "session_explicit": explicit,
            }

        monkeypatch.setattr(router_mod, "_is_self_healing_active", lambda: False)
        monkeypatch.setattr(router_mod, "_handle_session", _fake_handle_session)
        monkeypatch.setattr(router_mod, "_build_slot_context", lambda *_: (None, None, 3.0))
        monkeypatch.setattr(router_mod, "_resolve_slot_model_name", lambda model, *_: model)

        async def _fake_acquire(*args, **kwargs):
            return (True, "test-sess-1", 1, 0)

        monkeypatch.setattr(router_mod, "_try_acquire_local_dispatch", _fake_acquire)

        slot_check_calls = []

        async def _fake_check(*args, **kwargs):
            slot_check_calls.append(kwargs)
            return None

        monkeypatch.setattr(router_mod, "_check_slot_availability", _fake_check)
        return slot_check_calls

    @pytest.mark.asyncio
    async def test_explicit_session_with_lease_skips_slot_check(self, monkeypatch):
        """An explicit session that acquired the lease skips the /slots check.

        Regression for LP-0MTDGBRPU003Z7KU: the router /slots?model= check
        serializes behind the busy child (5-7s) and starved the shared pool.
        With a lease held the check is redundant (the lease gates concurrency)
        and must not run.
        """
        from proxy import router as router_mod

        slot_check_calls = self._setup_common(monkeypatch, explicit=True)

        # Intercept before the streaming path runs: raise so we only assert
        # the slot-check wiring.
        async def _boom(*args, **kwargs):
            raise RuntimeError("stop here")

        monkeypatch.setattr(router_mod, "_call_with_backend_retries", _boom)

        try:
            await router_mod.proxy_to_local(self._make_request(), "v1/chat/completions")
        except RuntimeError:
            pass

        assert len(slot_check_calls) == 1, "slot check should be called once"
        assert slot_check_calls[0].get("lease_held") is True, (
            "lease_held must be True for an explicit session that acquired "
            "the dispatch lease (the helper then short-circuits without an "
            "HTTP call — LP-0MTDGBRPU003Z7KU)"
        )

    @pytest.mark.asyncio
    async def test_anonymous_session_still_checks_slots(self, monkeypatch):
        """Anonymous sessions (no lease) still run the /slots availability check."""
        from proxy import router as router_mod

        slot_check_calls = self._setup_common(monkeypatch, explicit=False)

        async def _boom(*args, **kwargs):
            raise RuntimeError("stop here")

        monkeypatch.setattr(router_mod, "_call_with_backend_retries", _boom)

        try:
            await router_mod.proxy_to_local(self._make_request(), "v1/chat/completions")
        except RuntimeError:
            pass

        assert len(slot_check_calls) == 1, "slot check should be called once"
        assert slot_check_calls[0].get("lease_held") is False, (
            "lease_held must be False for anonymous sessions (no lease), so "
            "the availability check still runs"
        )
