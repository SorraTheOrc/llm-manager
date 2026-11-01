"""Thread the resolved model through the model-less ``/slots`` probes.

AC2 (part 1) and AC3(b)/(c) of LP-0MVP7HV7Q001AJF8 (router ``/slots`` HTTP 400s
from a model-less query), delivered by slice 3 LP-0MVU7ZZ7M002D9GT.

Every probe covered here used to issue a bare ``GET /slots``. The router
answers that with HTTP 400 because it cannot resolve a model, so a model-less
probe was indistinguishable in ``llama-server.log`` from a genuine capacity or
health problem. These tests drive each probe against a **router-mode responder
stub** built with ``httpx.MockTransport``: it answers 400 when the request
omits ``?model=`` (or names a model the router does not know) and 200 with a
slot payload when the model is resolvable — exactly llama.cpp's router
contract.

Three behaviours are pinned per call site:

* **AC3(c)** a resolved model is sent as ``?model=<model>``;
* a site with no resolved model issues **no** ``/slots`` request at all
  (the query is skipped, preserving today's fail-open verdict);
* **AC3(b)** an unknown / alias-mismatched model produces **one** request and
  a fail-open answer — never a 400-retry loop.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from proxy.backends import llama as llama_backend

from proxy import backend_health
from proxy import compaction_summarizer as summarizer
from proxy import router_helpers as rh

ROUTER_PORT = 8080
CHILD_PORT = 58113
KNOWN_MODEL = "Qwen3"
#: An alias/preset the router cannot resolve (the Q2=A "unknown id" defect class).
UNKNOWN_MODEL = "Qwen3-Coder-Next-GGUF:Q5_K_M"


class RouterResponder:
    """llama.cpp router-mode ``/slots`` stub for ``httpx.MockTransport``.

    * ``GET /health`` → 404 (a llama-server build without ``/health``, so the
      ``/slots`` fallback inside ``_probe_local_endpoint_http`` is exercised).
    * ``GET /slots?model=<known>`` → 200 with the configured slot payload.
    * ``GET /slots`` (no model) or ``?model=<unknown>`` → 400.
    """

    def __init__(self, known_models=(KNOWN_MODEL,), slots=None):
        self.known_models = set(known_models)
        self.slots = (
            slots
            if slots is not None
            else [
                {"id": 0, "is_processing": False},
                {"id": 1, "is_processing": True},
            ]
        )
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(404)
        if request.url.path != "/slots":
            return httpx.Response(404)
        model = request.url.params.get("model")
        if not model or model not in self.known_models:
            return httpx.Response(400, json={"error": {"message": "model required"}})
        return httpx.Response(200, json=self.slots)

    @property
    def slots_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path == "/slots"]

    @property
    def models_sent(self) -> list[str | None]:
        return [r.url.params.get("model") for r in self.slots_requests]


def _install_async_transport(monkeypatch, responder: RouterResponder, module) -> None:
    """Route *module*'s ``httpx.AsyncClient`` through the responder stub."""
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(responder.handler)

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", _factory)


def _install_sync_transport(monkeypatch, responder: RouterResponder, module) -> None:
    """Route *module*'s synchronous ``httpx.Client`` through the stub."""
    real_client = httpx.Client
    transport = httpx.MockTransport(responder.handler)

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "Client", _factory)


def _fake_srv(port: int = ROUTER_PORT) -> SimpleNamespace:
    """Minimal server handle accepted by ``_resolve_local_slots_target``."""
    return SimpleNamespace(
        config={"server": {"llama_server_port": port}},
        current_model=KNOWN_MODEL,
        llama_process=SimpleNamespace(pid=41395),
        log_dir=Path("/nonexistent"),
        logger=SimpleNamespace(warning=lambda *a, **k: None, debug=lambda *a, **k: None),
    )


# ---------------------------------------------------------------------------
# backend_health._probe_local_endpoint_http (/health -> /slots fallback)
# ---------------------------------------------------------------------------


class TestProbeLocalEndpointHttp:
    @pytest.mark.asyncio
    async def test_slots_fallback_sends_the_resolved_model(self, monkeypatch):
        responder = RouterResponder()
        _install_async_transport(monkeypatch, responder, backend_health)

        ok = await backend_health._probe_local_endpoint_http(
            f"http://127.0.0.1:{ROUTER_PORT}", model_name=KNOWN_MODEL
        )

        assert ok is True
        assert responder.models_sent == [KNOWN_MODEL]
        assert all(r.url.params.get("model") for r in responder.slots_requests)

    @pytest.mark.asyncio
    async def test_no_model_skips_the_slots_query_entirely(self, monkeypatch):
        responder = RouterResponder()
        _install_async_transport(monkeypatch, responder, backend_health)

        ok = await backend_health._probe_local_endpoint_http(
            f"http://127.0.0.1:{ROUTER_PORT}"
        )

        assert ok is False
        assert responder.slots_requests == []
        assert [r.url.path for r in responder.requests] == ["/health"]

    @pytest.mark.asyncio
    async def test_unknown_model_fails_open_once_without_a_retry_loop(self, monkeypatch):
        responder = RouterResponder()
        _install_async_transport(monkeypatch, responder, backend_health)

        ok = await backend_health._probe_local_endpoint_http(
            f"http://127.0.0.1:{ROUTER_PORT}", model_name=UNKNOWN_MODEL
        )

        assert ok is False
        assert responder.models_sent == [UNKNOWN_MODEL]
        assert len(responder.slots_requests) == 1

    @pytest.mark.asyncio
    async def test_healthy_health_endpoint_short_circuits_without_slots(self, monkeypatch):
        """A 200 from /health answers the probe; no /slots request is made."""
        responder = RouterResponder()

        def handler(request: httpx.Request) -> httpx.Response:
            responder.requests.append(request)
            return httpx.Response(200)

        real_client = httpx.AsyncClient
        transport = httpx.MockTransport(handler)

        def _factory(*args, **kwargs):
            kwargs["transport"] = transport
            return real_client(*args, **kwargs)

        monkeypatch.setattr(backend_health.httpx, "AsyncClient", _factory)

        ok = await backend_health._probe_local_endpoint_http(
            f"http://127.0.0.1:{ROUTER_PORT}", model_name=KNOWN_MODEL
        )

        assert ok is True
        assert responder.slots_requests == []


# ---------------------------------------------------------------------------
# backend_health._probe_local_slot_capacity
# ---------------------------------------------------------------------------


class TestProbeLocalSlotCapacity:
    @pytest.mark.asyncio
    async def test_capacity_probe_sends_the_resolved_model(self, monkeypatch):
        responder = RouterResponder()
        _install_async_transport(monkeypatch, responder, backend_health)

        available, total = await backend_health._probe_local_slot_capacity(
            f"http://127.0.0.1:{ROUTER_PORT}", model_name=KNOWN_MODEL
        )

        assert (available, total) == (1, 2)
        assert responder.models_sent == [KNOWN_MODEL]
        assert all(r.url.params.get("model") for r in responder.slots_requests)

    @pytest.mark.asyncio
    async def test_no_model_skips_the_capacity_query(self, monkeypatch):
        responder = RouterResponder()
        _install_async_transport(monkeypatch, responder, backend_health)

        available, total = await backend_health._probe_local_slot_capacity(
            f"http://127.0.0.1:{ROUTER_PORT}"
        )

        assert (available, total) == (0, 0)
        assert responder.requests == []

    @pytest.mark.asyncio
    async def test_unknown_model_fails_open_once_without_a_retry_loop(self, monkeypatch):
        responder = RouterResponder()
        _install_async_transport(monkeypatch, responder, backend_health)

        available, total = await backend_health._probe_local_slot_capacity(
            f"http://127.0.0.1:{ROUTER_PORT}", model_name=UNKNOWN_MODEL
        )

        assert (available, total) == (0, 0)
        assert responder.models_sent == [UNKNOWN_MODEL]
        assert len(responder.slots_requests) == 1


# ---------------------------------------------------------------------------
# backends/llama.py::_probe_gpu_wedge
# ---------------------------------------------------------------------------


class TestProbeGpuWedge:
    @pytest.fixture(autouse=True)
    def _gpu_sensors(self, monkeypatch):
        monkeypatch.setattr(llama_backend, "_read_gpu_busy_percent", lambda: 100.0)
        monkeypatch.setattr(llama_backend, "_read_gpu_power_draw", lambda: 54.0)

    @pytest.mark.asyncio
    async def test_wedge_probe_sends_the_target_model(self, monkeypatch):
        responder = RouterResponder(slots=[{"id": 0, "is_processing": False}])
        _install_async_transport(monkeypatch, responder, llama_backend)

        busy, slots_idle, power = await llama_backend._probe_gpu_wedge(
            "127.0.0.1", CHILD_PORT, model=KNOWN_MODEL
        )

        assert (busy, slots_idle, power) == (100.0, True, 54.0)
        assert responder.models_sent == [KNOWN_MODEL]

    @pytest.mark.asyncio
    async def test_no_model_skips_the_query_and_fails_open(self, monkeypatch):
        responder = RouterResponder()
        _install_async_transport(monkeypatch, responder, llama_backend)

        busy, slots_idle, power = await llama_backend._probe_gpu_wedge(
            "127.0.0.1", CHILD_PORT
        )

        assert (busy, slots_idle, power) == (100.0, False, 54.0)
        assert responder.requests == []

    @pytest.mark.asyncio
    async def test_unknown_model_fails_open_once_without_a_retry_loop(self, monkeypatch):
        responder = RouterResponder()
        _install_async_transport(monkeypatch, responder, llama_backend)

        busy, slots_idle, power = await llama_backend._probe_gpu_wedge(
            "127.0.0.1", CHILD_PORT, model=UNKNOWN_MODEL
        )

        assert (busy, slots_idle, power) == (100.0, False, 54.0)
        assert responder.models_sent == [UNKNOWN_MODEL]
        assert len(responder.slots_requests) == 1


# ---------------------------------------------------------------------------
# compaction_summarizer._probe_slot_capacity_sync
# ---------------------------------------------------------------------------


class TestProbeSlotCapacitySync:
    def test_probe_targets_the_discovered_child_port_with_the_model(self, monkeypatch):
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: CHILD_PORT)
        monkeypatch.setattr(rh, "_child_port_cache", {})
        monkeypatch.setattr(rh, "_child_port_cache_at", {})
        monkeypatch.setattr(rh, "_child_port_miss_cache", {})
        responder = RouterResponder()
        _install_sync_transport(monkeypatch, responder, summarizer)

        available, total = summarizer._probe_slot_capacity_sync(
            ROUTER_PORT, model_name=KNOWN_MODEL, srv=_fake_srv()
        )

        assert (available, total) == (1, 2)
        assert responder.models_sent == [KNOWN_MODEL]
        request = responder.slots_requests[0]
        assert request.url.port == CHILD_PORT
        assert request.url.port != ROUTER_PORT

    def test_probe_falls_back_to_the_given_port_when_no_child_is_discovered(
        self, monkeypatch
    ):
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: None)
        monkeypatch.setattr(rh, "_child_port_cache", {})
        monkeypatch.setattr(rh, "_child_port_cache_at", {})
        monkeypatch.setattr(rh, "_child_port_miss_cache", {})
        monkeypatch.setattr(rh, "_last_child_port_fallback_warn_ts", 0.0)
        responder = RouterResponder()
        _install_sync_transport(monkeypatch, responder, summarizer)

        available, total = summarizer._probe_slot_capacity_sync(
            ROUTER_PORT, model_name=KNOWN_MODEL, srv=_fake_srv()
        )

        assert (available, total) == (1, 2)
        assert responder.slots_requests[0].url.port == ROUTER_PORT

    def test_no_model_skips_the_query(self, monkeypatch):
        responder = RouterResponder()
        _install_sync_transport(monkeypatch, responder, summarizer)

        available, total = summarizer._probe_slot_capacity_sync(
            ROUTER_PORT, srv=_fake_srv()
        )

        assert (available, total) == (0, 0)
        assert responder.requests == []

    def test_unknown_model_fails_open_once_without_a_retry_loop(self, monkeypatch):
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: CHILD_PORT)
        monkeypatch.setattr(rh, "_child_port_cache", {})
        monkeypatch.setattr(rh, "_child_port_cache_at", {})
        monkeypatch.setattr(rh, "_child_port_miss_cache", {})
        responder = RouterResponder()
        _install_sync_transport(monkeypatch, responder, summarizer)

        available, total = summarizer._probe_slot_capacity_sync(
            ROUTER_PORT, model_name=UNKNOWN_MODEL, srv=_fake_srv()
        )

        assert (available, total) == (0, 0)
        assert responder.models_sent == [UNKNOWN_MODEL]
        assert len(responder.slots_requests) == 1
