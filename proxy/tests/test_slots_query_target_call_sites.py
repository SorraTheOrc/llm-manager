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

Slice 3 of LP-0MVVCY3HQ0009F0K (LP-0MVWBXEAI008YZ7B) extends the same shared
path with the skip sentinel contract: when ``_resolve_local_slots_target()``
returns ``(None, None)`` the model is **definitively not loaded** and the call
site must issue **no** ``/slots`` request. The sentinel cases below pin that
contract per call site — a future call site that ignores the sentinel and
passes a ``None`` port into a URL builder fails loudly in its own tests (a
``None`` port can never be a real llama-server port), and its missing skip
case here is the deliberate gap the suite documents.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from proxy import router_helpers as rh

#: Production-shaped /models payload: Qwen3-32B is loaded, gpt120 is
#: definitively unloaded (the router's own answer).
LOADED_PAYLOAD = {
    "object": "list",
    "data": [
        {"id": "Qwen3-32B", "status": {"value": "loaded"}},
    ],
}

UNLOADED_PAYLOAD = {
    "object": "list",
    "data": [
        {"id": "gpt120", "status": {"value": "unloaded"}},
    ],
}


async def _warm_models(srv, payload: dict) -> None:
    """Warm the shared ``/models`` snapshot for *srv* (async dispatch path).

    Mirrors ``ensure_router_model_ids()``'s producer: the resolver reads this
    same snapshot synchronously, so warming here makes the skip guard and its
    loaded-set view assertable without an HTTP round trip.
    """

    async def _lister():
        return payload

    srv.router_list_models = _lister
    await rh.ensure_router_model_ids(srv)


@pytest.fixture(autouse=True)
def _clear_child_port_state(monkeypatch):
    rh._child_port_cache.clear()
    rh._child_port_cache_at.clear()
    rh._child_port_miss_cache.clear()
    rh._child_model_signal_cache.clear()
    rh._router_model_ids_snapshot.clear()
    rh._last_child_port_fallback_warn_ts = 0.0
    rh._last_child_port_skip_debug_ts = 0.0
    yield
    rh._child_port_cache.clear()
    rh._child_port_cache_at.clear()
    rh._child_port_miss_cache.clear()
    rh._child_model_signal_cache.clear()
    rh._router_model_ids_snapshot.clear()
    rh._last_child_port_fallback_warn_ts = 0.0
    rh._last_child_port_skip_debug_ts = 0.0


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

    # ------------------------------------------------------------------
    # Skip sentinel contract (LP-0MVWBXEAI008YZ7B): the loaded-set guard
    # sits in this single shared path, so every call site inherits it. A
    # right guard must never swallow the loaded-model fallback signal.
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_unloaded_model_returns_skip_sentinel(self, monkeypatch):
        """A model the router reports as not loaded resolves to (None, None)
        and emits the AC6 skip signal once."""
        srv = _srv()
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)
        await _warm_models(srv, UNLOADED_PAYLOAD)
        import proxy.metrics as metrics

        skipped = []
        monkeypatch.setattr(metrics, "record_slots_child_port_skipped", skipped.append)

        endpoint, port = rh._resolve_local_slots_target(srv, model="gpt120")
        assert (endpoint, port) == (None, None), (
            "the (None, None) skip sentinel is returned for a definitively "
            "unloaded model; a None port can never be a real llama-server port"
        )
        assert skipped == ["model_not_loaded"]

    @pytest.mark.asyncio
    async def test_skip_debug_log_is_throttled_and_names_model(self, monkeypatch):
        """Every skip counts; the debug line is throttled and names the model."""
        dbg: list[str] = []
        srv = _srv(
            logger=SimpleNamespace(
                warning=lambda *a: None, debug=lambda *a: dbg.append(str(a[0]))
            )
        )
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: None)
        await _warm_models(srv, UNLOADED_PAYLOAD)

        rh._resolve_local_slots_target(srv, model="gpt120")
        rh._resolve_local_slots_target(srv, model="gpt120")

        lines = [m for m in dbg if "slots_child_port_skipped" in m]
        assert len(lines) == 1, "repetitive skips stay throttled"
        assert "gpt120" in lines[0]

    @pytest.mark.asyncio
    async def test_loaded_model_never_skipped(self, monkeypatch):
        """A loaded model resolves to the discovered child port as today."""
        srv = _srv()
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)
        await _warm_models(srv, LOADED_PAYLOAD)

        endpoint, port = rh._resolve_local_slots_target(srv, model="Qwen3-32B")
        assert (endpoint, port) == (None, 58113)

    @pytest.mark.asyncio
    async def test_alias_of_loaded_model_is_not_skipped(self, monkeypatch):
        """An alias/preset spelling of a loaded id is canonicalised against the
        router's own ids before the membership check — never misread as
        unloaded (a false skip would swallow the AC2 fallback signal)."""
        srv = _srv()
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)
        await _warm_models(srv, LOADED_PAYLOAD)

        endpoint, port = rh._resolve_local_slots_target(srv, model="Qwen3")
        assert (endpoint, port) == (None, 58113)

    @pytest.mark.asyncio
    async def test_indeterminate_loadedness_fails_open(self, monkeypatch):
        """No warm snapshot (the common boot case) -> the guard is
        indeterminate (None) and discovery proceeds exactly as today."""
        srv = _srv()
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)

        endpoint, port = rh._resolve_local_slots_target(srv, model="gpt120")
        assert (endpoint, port) == (None, 58113)

    @pytest.mark.asyncio
    async def test_external_endpoint_never_skipped(self, monkeypatch):
        """An endpoint pointing at a genuinely different instance is returned
        untouched even when the local snapshot reports the model unloaded —
        the local loaded-set is only authoritative for the local router."""
        srv = _srv()
        await _warm_models(srv, UNLOADED_PAYLOAD)
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)

        endpoint, port = rh._resolve_local_slots_target(
            srv, model="gpt120", endpoint="http://192.168.1.50:8080"
        )
        assert endpoint == "http://192.168.1.50:8080"
        assert port == 8080


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

    @pytest.mark.asyncio
    async def test_unloaded_model_is_not_queried(self, monkeypatch):
        """LP-0MVWBXEAI008YZ7B AC7: an unloaded model skips the availability
        check entirely — same admission result (None) as the definitive-400
        fall-through, and no /slots request is ever issued."""
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
        client._resp = self._mock_resp([{"id": 0, "is_processing": False}])

        def _async_client_factory(timeout):
            captured["created"] = True
            return client

        srv = _srv()
        await _warm_models(srv, UNLOADED_PAYLOAD)
        monkeypatch.setattr(
            rh, "httpx",
            SimpleNamespace(AsyncClient=_async_client_factory, Timeout=lambda t: t),
        )
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)

        result = await rh._check_slot_availability(
            srv,
            {"llama_server_port": 8080,
             "session_slot_availability_timeout_seconds": 1.0},
            8080, "gpt120", "gpt120",
            "v1/chat/completions",
        )
        assert result is None, "an unloaded model must never be refused (AC7)"
        assert "created" not in captured, "no /slots client may be created"
        assert "url" not in captured, "no /slots request may be issued"


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

            async def fake_query_slots(
                client, port, timeout=2.0, model=None, status_out=None,
                provenance_out=None,
            ):
                seen["counts_port"] = port
                if isinstance(status_out, dict):
                    status_out.update({"status": 200, "reason": None})
                return (1, 3)

            async def fake_query_detail(
                port, timeout=2.0, model=None, _client=None, provenance_out=None,
            ):
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

    @pytest.mark.asyncio
    async def test_status_skips_slots_for_unloaded_model(self, monkeypatch):
        """LP-0MVWBXEAI008YZ7B AC3: an unloaded model's /status issues **no**
        /slots counts/detail query and fails open on the configured pool —
        never a fossil snapshot."""
        import proxy.server as server
        from proxy.handlers import get_llama_local_status

        queries: list[tuple[str, int]] = []

        async def fake_query_slots(
            client, port, timeout=2.0, model=None, status_out=None,
            provenance_out=None,
        ):
            queries.append(("counts", port))
            return (0, 0)

        async def fake_query_detail(
            port, timeout=2.0, model=None, _client=None, provenance_out=None,
        ):
            queries.append(("detail", port))
            return []

        with monkeypatch.context() as m:
            m.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)
            m.setattr(
                server, "query_llama_status",
                AsyncMock(return_value={"llama_server_running": True}),
            )
            server.current_model = "gpt120"
            server.llama_process = MagicMock(poll=lambda: None)
            server._http_client = None
            server.config = {
                "server": {"llama_server_port": 8080, "session_slot_pool_size": 3}
            }

            async def _list_models():
                return UNLOADED_PAYLOAD

            m.setattr(server, "router_list_models", _list_models)
            await rh.ensure_router_model_ids(server)

            m.setattr("proxy.observability._query_slots", fake_query_slots)
            m.setattr("proxy.observability._query_slots_detail", fake_query_detail)

            req = MagicMock()
            req.headers = {}
            req.client = MagicMock(host="127.0.0.1", port=54321)
            j = await get_llama_local_status(req)

        assert j["total_slots"] == 3, "fail open on the configured pool (AC3)"
        assert j["available_slots"] == 3
        assert queries == [], "no /slots request for a definitively unloaded model"


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

    @pytest.mark.asyncio
    async def test_initial_payload_skips_slots_for_unloaded_model(self, monkeypatch):
        """LP-0MVWBXEAI008YZ7B: the SSE initial payload issues **no** /slots
        detail query for a definitively unloaded model; the payload still
        streams (degraded empty slot list)."""
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
            current_model="gpt120",
            config={"server": {"llama_server_port": 8080}},
            logger=SimpleNamespace(
                warning=lambda *a, **k: None, debug=lambda *a, **k: None
            ),
        )
        seen = {}

        async def spy_detail(port, timeout=2.0, model=None, _client=None):
            seen["port"] = port
            return [{"slot_id": 0, "is_processing": False, "n_decoded": 0}]

        await _warm_models(srv, UNLOADED_PAYLOAD)
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
        assert "port" not in seen, (
            "the SSE initial payload must issue no /slots query for an unloaded model"
        )


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

    @pytest.mark.asyncio
    async def test_prefill_probe_skips_unloaded_model(self, monkeypatch):
        """LP-0MVWBXEAI008YZ7B: a definitively unloaded model issues **no**
        /slots probe; the progress call falls through to the /status
        aggregation (which is not a /slots request)."""
        import proxy.observability as obs

        seen = {}

        async def spy_progress(
            port, timeout=2.0, model=None, endpoint=None, _client=None
        ):
            seen["port"] = port
            return {0: {"progress": 100, "processing": True}}

        monkeypatch.setattr(obs, "_query_slots_progress", spy_progress)
        monkeypatch.setattr(
            obs, "query_llama_status", AsyncMock(return_value={"kv_cache_tokens": 0})
        )
        srv = _srv()
        await _warm_models(srv, UNLOADED_PAYLOAD)

        progress, alive = await rh._query_prefill_progress(
            srv, llama_port=8080, model_name="gpt120", slot_id=0,
        )
        assert progress is None and alive is False
        assert "port" not in seen, "no /slots probe for an unloaded model"

    @pytest.mark.asyncio
    async def test_lease_liveness_probe_skips_unloaded_model(self, monkeypatch):
        """LP-0MVWBXEAI008YZ7B: the lease-liveness probe issues **no** /slots
        probe for a definitively unloaded model and reports not-processing
        (same outcome as the probe-failure path today)."""
        import proxy.observability as obs
        import proxy.session as session_mod

        seen = {}

        async def spy_progress(
            port, timeout=2.0, model=None, endpoint=None, _client=None
        ):
            seen["port"] = port
            return {7: {"progress": None, "processing": True}}

        monkeypatch.setattr(obs, "_query_slots_progress", spy_progress)
        monkeypatch.setattr(
            session_mod,
            "_assigned_slot_for_session",
            lambda session_id, endpoint=None: 7,
        )
        srv = _srv()
        await _warm_models(srv, UNLOADED_PAYLOAD)

        processing = await rh._query_slot_processing(
            srv, "sess-1", model_name="gpt120", endpoint="http://localhost:8080"
        )
        assert processing is False
        assert "port" not in seen, "no /slots probe for an unloaded model"


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

    @pytest.mark.asyncio
    async def test_poller_skips_unloaded_model(self, monkeypatch):
        """LP-0MVWBXEAI008YZ7B AC1: the per-model poller issues **no** /slots
        request for a definitively unloaded model and leaves the polling
        state untouched."""
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
        fake_srv = SimpleNamespace(
            config={"server": {"llama_server_port": 8080}},
            _http_client=fake_client,
            llama_process=proc,
            _slot_polling_tasks={},
            slot_polling_state={},
        )
        monkeypatch.setattr(srv_module, "config", fake_srv.config)
        monkeypatch.setattr(srv_module, "llama_process", proc)
        monkeypatch.setattr(srv_module, "_http_client", None)
        monkeypatch.setattr(srv_module, "slot_polling_state", {})
        monkeypatch.setattr(srv_module, "_slot_polling_tasks", {})
        monkeypatch.setattr(srv_module, "logger", MagicMock())
        monkeypatch.setattr(srv_module, "_srv", lambda: fake_srv)
        monkeypatch.setattr(handlers_mod, "_srv", srv_module._srv)
        monkeypatch.setattr(
            handlers_mod, "httpx",
            SimpleNamespace(AsyncClient=lambda timeout: fake_client, Timeout=lambda t: t),
        )
        await _warm_models(fake_srv, UNLOADED_PAYLOAD)

        await handlers_mod.poll_slots_for_model("gpt120", llama_port=8080, max_polls=1)
        assert "url" not in captured, "no /slots request for an unloaded model"
        assert "gpt120" not in fake_srv.slot_polling_state


class TestSseBroadcastLoopCallSite:
    """The SSE broadcast loop must honour the skip sentinel.

    The loop is an infinite background task, so the assertion drives exactly
    one iteration: the second ``asyncio.sleep`` raises ``CancelledError``,
    which the loop's own ``except asyncio.CancelledError: break`` turns into
    a normal return after one full round.
    """

    @pytest.mark.asyncio
    async def test_broadcast_loop_skips_slots_for_unloaded_model(self, monkeypatch):
        import proxy.observability as obs

        class _ALock:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        srv = SimpleNamespace(
            config={"server": {"llama_server_port": 8080}},
            current_model="gpt120",
            request_counts={},
            token_counts={"total_sent": 0, "total_recv": 0},
            counts_lock=_ALock(),
            token_lock=_ALock(),
            per_model_queries_lock=_ALock(),
            per_model_queries={},
            sse_clients=set(),
            periodic_broadcast_task=None,
            llama_process=MagicMock(pid=41395),
            logger=SimpleNamespace(
                warning=lambda *a, **k: None, debug=lambda *a, **k: None
            ),
        )
        detail_calls: list[int] = []

        async def spy_detail(port, timeout=5.0, model=None, _client=None, **kw):
            detail_calls.append(port)
            return [{"slot_id": 0}]

        await _warm_models(srv, UNLOADED_PAYLOAD)

        sleep_calls = {"n": 0}
        real_sleep = asyncio.sleep

        async def _bounded_sleep(sec):
            sleep_calls["n"] += 1
            if sleep_calls["n"] >= 2:
                raise asyncio.CancelledError()
            await real_sleep(0)

        with monkeypatch.context() as m:
            m.setattr(obs, "_srv", lambda: srv)
            m.setattr(
                obs, "query_llama_status",
                AsyncMock(return_value={"llama_server_running": True}),
            )
            m.setattr(obs, "_query_slots_detail", spy_detail)
            m.setattr(obs, "_update_slot_progress_from_log", lambda: None)
            m.setattr(
                obs, "_enrich_slot_details_with_progress",
                lambda details, srv=None: None,
            )
            m.setattr(obs, "log_tail_clients", [])
            m.setattr(asyncio, "sleep", _bounded_sleep)
            await obs._periodic_broadcast_loop()

        assert sleep_calls["n"] >= 2, "one full broadcast iteration must have run"
        assert detail_calls == [], (
            "the SSE broadcast loop must issue no /slots request for an "
            "unloaded model (LP-0MVWBXEAI008YZ7B)"
        )
