"""Every proxy-initiated /slots query targets the discovered child port
(LP-0MVP7525P008RXMF AC2/AC3).

The router llama-server answers /slots with 500 whenever the poll arrives while
the Qwen3 child is mid-giant-prefill, so no proxy /slots query may traverse the
router. The rank-1 fix patched only the status handler and the SSE broadcast
loop, and even those silently fell back to the router because discovery was
inert. These tests assert that *every* call site — status handler, SSE initial
payload, slot-availability check, progress/liveness probes, and the per-model
slot poller — resolves through the shared ``_resolve_local_slots_target()`` path
and therefore uses the discovered child port, and that the fallback is loud
(throttled warning + counter).
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from proxy import router_helpers as rh


@pytest.fixture(autouse=True)
def _clear_child_port_state(monkeypatch):
    rh._child_port_cache.clear()
    rh._child_port_cache_at.clear()
    rh._child_port_miss_cache.clear()
    rh._last_child_port_fallback_warn_ts = 0.0
    yield
    rh._child_port_cache.clear()
    rh._child_port_cache_at.clear()
    rh._child_port_miss_cache.clear()
    rh._last_child_port_fallback_warn_ts = 0.0


def _srv(pid=41395, port=8080, router_mode=True, logger=None):
    return SimpleNamespace(
        log_dir=Path("/nonexistent"),
        llama_process=SimpleNamespace(pid=pid),
        current_model="Qwen3",
        config={
            "server": {
                "llama_server_port": port,
                "llama_router_mode": router_mode,
            }
        },
        logger=logger or SimpleNamespace(
            warning=lambda *a, **k: None, debug=lambda *a, **k: None
        ),
    )


class TestResolveLocalSlotsTarget:
    """The single shared resolution path (AC2)."""

    def test_local_router_endpoint_is_rewritten_to_child(self, monkeypatch):
        srv = _srv()
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)
        endpoint, port = rh._resolve_local_slots_target(
            srv, model="Qwen3", endpoint="http://localhost:8080"
        )
        assert endpoint is None
        assert port == 58113

    def test_no_endpoint_resolves_child(self, monkeypatch):
        srv = _srv()
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)
        endpoint, port = rh._resolve_local_slots_target(srv, model="Qwen3")
        assert endpoint is None
        assert port == 58113

    def test_external_endpoint_is_untouched(self, monkeypatch):
        """A genuinely different llama-server instance is never rewritten."""
        srv = _srv()
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)
        endpoint, port = rh._resolve_local_slots_target(
            srv, model="Qwen3", endpoint="http://192.168.1.50:8080"
        )
        assert endpoint == "http://192.168.1.50:8080"
        assert port == 8080

    def test_loopback_child_port_endpoint_is_untouched(self, monkeypatch):
        """http://localhost:<child> is already the child — not rewritten."""
        srv = _srv()
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 9999)
        endpoint, port = rh._resolve_local_slots_target(
            srv, model="Qwen3", endpoint="http://localhost:58113"
        )
        assert endpoint == "http://localhost:58113"
        assert port == 58113

    def test_fallback_warns_and_counts(self, monkeypatch):
        """AC3: no child port -> throttled warning + counter + router port."""
        log = []
        srv = _srv(logger=SimpleNamespace(warning=lambda *a: log.append(str(a[0]))))
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: None)
        import proxy.metrics as metrics

        reasons = []
        monkeypatch.setattr(metrics, "record_slots_child_port_fallback", reasons.append)

        endpoint, port = rh._resolve_local_slots_target(
            srv, model="Qwen3", endpoint="http://localhost:8080"
        )
        assert endpoint is None
        assert port == 8080
        assert reasons == ["no_child_port_router_mode"]
        assert any("slots_child_port_unavailable" in m for m in log)

    def test_fallback_warning_is_throttled(self, monkeypatch):
        """Repetitive fallbacks log one warning per interval, not per call."""
        log = []
        srv = _srv(logger=SimpleNamespace(warning=lambda *a: log.append(str(a[0]))))
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: None)
        import proxy.metrics as metrics

        monkeypatch.setattr(metrics, "record_slots_child_port_fallback", lambda r: None)

        for _ in range(3):
            rh._resolve_local_slots_target(srv, model="Qwen3")
        assert len(log) == 1

    def test_single_server_mode_counts_but_does_not_warn(self, monkeypatch):
        """No router -> no child exists; counted, logged at debug level only."""
        warn, dbg = [], []
        logger = SimpleNamespace(
            warning=lambda *a: warn.append(str(a[0])),
            debug=lambda *a: dbg.append(str(a[0])),
        )
        srv = _srv(router_mode=False, logger=logger)
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: None)
        import proxy.metrics as metrics

        reasons = []
        monkeypatch.setattr(metrics, "record_slots_child_port_fallback", reasons.append)

        endpoint, port = rh._resolve_local_slots_target(srv, model="Qwen3")
        assert (endpoint, port) == (None, 8080)
        assert reasons == ["no_child_port_single_server"]
        assert warn == []
        assert any("slots_child_port_unavailable" in m for m in dbg)


class TestCheckSlotAvailabilityCallSite:
    """_check_slot_availability must not use the router port for local mode."""

    def _mock_resp(self, slots):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = slots
        return resp

    @pytest.mark.asyncio
    async def test_local_endpoint_is_rewritten_to_child_port(self, monkeypatch):
        """AC2: endpoint=http://localhost:8080 (the local provider endpoint /
        the router) is rewritten to the discovered child port."""
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
        srv = _srv()
        monkeypatch.setattr(
            rh, "httpx",
            SimpleNamespace(AsyncClient=lambda timeout: client, Timeout=lambda t: t),
        )
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)

        result = await rh._check_slot_availability(
            srv, {"llama_server_port": 8080}, 8080, "Qwen3", "Qwen3",
            "v1/chat/completions", endpoint="http://localhost:8080",
        )
        assert result is None  # slots available -> no 503
        assert captured["url"] == "http://localhost:58113/slots?model=Qwen3", (
            "no /slots URL in this path may use server.llama_server_port"
        )

    @pytest.mark.asyncio
    async def test_fallback_warns_and_targets_router(self, monkeypatch):
        """AC3: no child discovered -> warning + router port (observable)."""
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
        client._resp = self._mock_resp([{"id": 0, "is_processing": True}])
        log = []
        srv = _srv(logger=SimpleNamespace(warning=lambda *a: log.append(str(a[0]))))
        monkeypatch.setattr(
            rh, "httpx",
            SimpleNamespace(AsyncClient=lambda timeout: client, Timeout=lambda t: t),
        )
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: None)
        import proxy.metrics as metrics

        reasons = []
        monkeypatch.setattr(metrics, "record_slots_child_port_fallback", reasons.append)

        result = await rh._check_slot_availability(
            srv, {"llama_server_port": 8080}, 8080, "Qwen3", "Qwen3",
            "v1/chat/completions",
        )
        assert result.status_code == 503
        assert captured["url"] == "http://localhost:8080/slots?model=Qwen3"
        assert reasons == ["no_child_port_router_mode"]
        assert log, "fallback must be observable in proxy.log"


class TestStatusHandlerCallSite:
    """/llama/local/status queries must target the child port."""

    @pytest.mark.asyncio
    async def test_status_queries_child_port(self, monkeypatch):
        import proxy.server as server
        from proxy.handlers import get_llama_local_status

        with monkeypatch.context() as m:
            m.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)
            m.setattr(
                server, "query_llama_status",
                AsyncMock(return_value={"llama_server_running": True}),
            )
            server.current_model = "Qwen3"
            server.llama_process = MagicMock(poll=lambda: None)
            server._http_client = None
            server.config = {
                "server": {"llama_server_port": 8080, "session_slot_pool_size": 3}
            }
            seen = {}

            async def fake_query_slots(client, port, timeout=2.0, model=None, status_out=None):
                seen["counts_port"] = port
                if isinstance(status_out, dict):
                    status_out.update({"status": 200, "reason": None})
                return (1, 3)

            async def fake_query_detail(port, timeout=2.0, model=None, _client=None):
                seen["detail_port"] = port
                return [{"slot_id": 0, "is_processing": False, "n_decoded": 10}]

            m.setattr("proxy.observability._query_slots", fake_query_slots)
            m.setattr("proxy.observability._query_slots_detail", fake_query_detail)

            req = MagicMock()
            req.headers = {}
            req.client = MagicMock(host="127.0.0.1", port=54321)
            j = await get_llama_local_status(req)
            assert j["total_slots"] == 3
            assert seen["counts_port"] == 58113
            assert seen["detail_port"] == 58113


class TestSseInitialPayloadCallSite:
    """The SSE initial payload (ui.py) must target the child port."""

    @pytest.mark.asyncio
    async def test_initial_payload_queries_child_port(self, monkeypatch):
        import proxy.ui as ui

        srv = SimpleNamespace(
            sse_clients=set(),
            query_llama_status=AsyncMock(
                return_value={
                    "llama_server_running": True,
                    "router_mode": False,
                    "n_ctx": 32768,
                    "kv_cache_tokens": 0,
                }
            ),
            token_counts={"total_sent": 0, "total_recv": 0},
            current_model="Qwen3",
            config={"server": {"llama_server_port": 8080}},
            logger=SimpleNamespace(warning=lambda *a, **k: None, debug=lambda *a, **k: None),
        )
        seen = {}

        async def spy_detail(port, timeout=2.0, model=None, _client=None):
            seen["port"] = port
            return [{"slot_id": 0, "is_processing": False, "n_decoded": 0}]

        monkeypatch.setattr(ui, "_srv", lambda: srv)
        monkeypatch.setattr(ui, "_query_slots_detail", spy_detail)
        monkeypatch.setattr(ui, "_get_per_model_queries", AsyncMock(return_value={}))
        monkeypatch.setattr(ui, "_quota_status_payload", lambda cfg: None)
        import proxy.observability as obs

        monkeypatch.setattr(obs, "_update_slot_progress_from_log", lambda: None)
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)

        resp = await ui.status_events()
        async for chunk in resp.body_iterator:
            break
        assert seen["port"] == 58113, (
            "the SSE initial payload slot query must use the discovered child port"
        )
        assert '"slots"' in chunk or 'slots' in chunk


class TestProgressProbeCallSites:
    """Prefill progress and lease-liveness probes must use the child port."""

    @pytest.mark.asyncio
    async def test_query_prefill_progress_targets_child_port(self, monkeypatch):
        import proxy.observability as obs

        seen = {}

        async def spy_progress(port, timeout=2.0, model=None, endpoint=None, _client=None):
            seen["port"] = port
            seen["endpoint"] = endpoint
            return {0: {"progress": 100, "processing": True}}

        monkeypatch.setattr(obs, "_query_slots_progress", spy_progress)
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)
        srv = _srv()

        progress, alive = await rh._query_prefill_progress(
            srv, llama_port=8080, model_name="Qwen3", slot_id=0,
        )
        assert (progress, alive) == (100, True)
        assert seen["port"] == 58113
        assert seen["endpoint"] is None

    @pytest.mark.asyncio
    async def test_query_slot_processing_targets_child_port(self, monkeypatch):
        import proxy.observability as obs

        seen = {}

        async def spy_progress(port, timeout=2.0, model=None, endpoint=None, _client=None):
            seen["port"] = port
            seen["endpoint"] = endpoint
            return {7: {"progress": None, "processing": True}}

        monkeypatch.setattr(obs, "_query_slots_progress", spy_progress)
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)
        import proxy.session as session_mod

        monkeypatch.setattr(
            session_mod, "_assigned_slot_for_session", lambda session_id, endpoint=None: 7
        )
        srv = _srv()

        processing = await rh._query_slot_processing(
            srv, "sess-1", model_name="Qwen3", endpoint="http://localhost:8080"
        )
        assert processing is True
        assert seen["port"] == 58113
        assert seen["endpoint"] is None
        """Probes against a non-local instance keep their endpoint (no rewrite)."""
        import proxy.observability as obs

        seen = {}

        async def spy_progress(port, timeout=2.0, model=None, endpoint=None, _client=None):
            seen["port"] = port
            seen["endpoint"] = endpoint
            return {7: {"progress": None, "processing": True}}

        monkeypatch.setattr(obs, "_query_slots_progress", spy_progress)
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 9999)
        import proxy.session as session_mod

        monkeypatch.setattr(
            session_mod, "_assigned_slot_for_session", lambda session_id, endpoint=None: 7
        )
        srv = _srv()

        processing = await rh._query_slot_processing(
            srv, "sess-1", model_name="Qwen3", endpoint="http://192.168.1.50:8080"
        )
        assert processing is True
        assert seen["endpoint"] == "http://192.168.1.50:8080"


class TestSlotPollerCallSite:
    """poll_slots_for_model must not traverse the router."""

    @pytest.mark.asyncio
    async def test_poller_uses_child_port(self, monkeypatch):
        from proxy import handlers as handlers_mod
        from proxy import server as srv_module

        captured = {}

        class _FakeResp:
            status_code = 200

            def json(self):
                return [{"id": 0, "is_processing": False}]

        class _FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, timeout=None):
                captured["url"] = url
                return _FakeResp()

        fake_client = _FakeClient()
        proc = MagicMock()
        monkeypatch.setattr(srv_module, "config", {"server": {"llama_server_port": 8080}})
        monkeypatch.setattr(srv_module, "llama_process", proc)
        monkeypatch.setattr(srv_module, "_http_client", None)
        monkeypatch.setattr(srv_module, "slot_polling_state", {})
        monkeypatch.setattr(srv_module, "_slot_polling_tasks", {})
        monkeypatch.setattr(srv_module, "logger", MagicMock())
        monkeypatch.setattr(
            srv_module, "_srv", lambda: SimpleNamespace(
                config=srv_module.config,
                _http_client=None,
                llama_process=proc,
                _slot_polling_tasks=srv_module._slot_polling_tasks,
                slot_polling_state=srv_module.slot_polling_state,
            )
        )
        # handlers.poll_slots_for_model calls _srv() — patch the module's import
        # so the fake state is visible inside the function body.
        monkeypatch.setattr(handlers_mod, "_srv", srv_module._srv)
        monkeypatch.setattr(
            handlers_mod, "httpx",
            SimpleNamespace(AsyncClient=lambda timeout: fake_client, Timeout=lambda t: t),
        )
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)

        await handlers_mod.poll_slots_for_model("Qwen3", llama_port=8080, max_polls=1)
        assert captured["url"] == "http://localhost:58113/slots?model=Qwen3"
        assert srv_module.slot_polling_state["Qwen3"]["is_processing"] is False
