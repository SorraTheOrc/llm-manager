"""``/models``-derived model id for proxy ``/slots`` queries.

Slice 4 of LP-0MVP7HV7Q001AJF8 (router ``/slots`` HTTP 400s), delivered by
LP-0MVU7ZZIA0092AXW. Covers AC2 (part 2) and AC3(b).

Before this slice ``_discover_local_child_port`` and ``_check_slot_availability``
fell back to the literal ``"Qwen3"``, and the three ``/slots`` query helpers
appended ``?model=`` only behind ``if model:``. The router's ids are *presets*,
not one universal name (``Qwen3``, ``Qwen3-32B``, ``mxbai-embed``,
``Qwen/Qwen3-Coder-Next-GGUF:Q5_K_M``, …), so a hard-coded default sends an
unresolvable ``?model=`` — a router HTTP 400 by construction.

Pinned here:

* resolution order — explicit model → ``srv.current_model`` → the sole
  non-embed ``/models`` preset → the live child-port log signal canonicalised
  against the ``/models`` ids → the log signal verbatim → the documented,
  logged ``"Qwen3"`` last resort (never a silent hard-coded default);
* the emitted ``?model=`` is the ``/models`` preset id while the child **port**
  still comes from the log signal (even when the log only names an alias);
* a warm ``/models`` snapshot costs **zero** HTTP requests; the snapshot is
  refreshed on a llama-server restart, a model switch, and TTL expiry;
* an unknown / alias-mismatched model produces exactly one fail-open result
  (never a 400-retry loop), with ``http_400`` still the definitive
  "model absent" signal.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from proxy import router_helpers as rh

ROUTER_PORT = 8080
CHILD_PORT = 41395

#: The production log shape (LP-0MVP7525P008RXMF AC1): the *live* signal is
#: present, the startup spawn line has been rotated away, and the first
#: proxied request was the embeddings child.
PRODUCTION_SHAPE = (
    "[59947] main: model loaded\n"
    "srv  proxy_reques: proxying request to model mxbai-embed on port 59947\n"
    "srv  proxy_reques: proxying request to model Qwen3 on port 41395\n"
    "srv  proxy_reques: proxying request to model Qwen3 on port 41395\n"
)


def _models_payload(*ids: str) -> dict:
    return {
        "object": "list",
        "data": [
            {"id": model_id, "status": {"value": "loaded"}} for model_id in ids
        ],
    }


def _srv(
    tmp_path,
    *,
    pid: int = CHILD_PORT,
    current_model: str | None = None,
    models: tuple[str, ...] = ("Qwen3-32B", "mxbai-embed"),
    log: str | None = None,
    warnings: list[str] | None = None,
):
    """Minimal router-mode server handle with a stubbed ``/models`` fetch."""
    calls = {"models": 0}

    def _warning(message, *args, **kwargs):
        if warnings is not None:
            warnings.append(str(message))

    srv = SimpleNamespace(
        log_dir=Path(tmp_path),
        llama_process=SimpleNamespace(pid=pid),
        current_model=current_model,
        config={
            "server": {
                "llama_server_port": ROUTER_PORT,
                "llama_router_mode": True,
            }
        },
        logger=SimpleNamespace(warning=_warning, debug=lambda *a, **k: None),
        models_calls=calls,
    )
    srv.models = list(models)

    async def _router_list_models():
        calls["models"] += 1
        return _models_payload(*srv.models)

    srv.router_list_models = _router_list_models
    if log is not None:
        (Path(tmp_path) / "llama-server.log").write_text(log)
    return srv


@pytest.fixture(autouse=True)
def _clear_resolution_caches():
    """Reset the module-level child-port / model-id caches between tests."""
    rh._child_port_cache.clear()
    rh._child_port_cache_at.clear()
    rh._child_port_miss_cache.clear()
    rh._child_model_signal_cache.clear()
    rh._router_model_ids_snapshot.clear()
    rh._last_child_port_fallback_warn_ts = 0.0
    rh._last_model_last_resort_warn_ts = 0.0
    rh._last_model_resolution.clear()
    yield
    rh._child_port_cache.clear()
    rh._child_port_cache_at.clear()
    rh._child_port_miss_cache.clear()
    rh._child_model_signal_cache.clear()
    rh._router_model_ids_snapshot.clear()
    rh._last_child_port_fallback_warn_ts = 0.0
    rh._last_model_last_resort_warn_ts = 0.0
    rh._last_model_resolution.clear()


# ---------------------------------------------------------------------------
# Resolution order: /models presets replace the hard-coded "Qwen3"
# ---------------------------------------------------------------------------


class TestModelsDerivedResolution:
    def test_models_preset_replaces_the_hardcoded_default(self, tmp_path):
        """A different router preset id is emitted, not ``"Qwen3"``."""
        srv = _srv(tmp_path, current_model=None, log=PRODUCTION_SHAPE)
        asyncio.run(rh.ensure_router_model_ids(srv))

        model, source = rh.resolve_local_model_id(srv)

        assert model == "Qwen3-32B"
        assert model != rh.LAST_RESORT_MODEL_ID
        assert source == "models_snapshot"
        # ... and the child port still comes from the live log signal.
        assert rh._discover_local_child_port(srv) == CHILD_PORT

    def test_child_port_survives_a_preset_alias_mismatch(self, tmp_path):
        """The log signal says ``Qwen3``; the emitted id is the preset."""
        srv = _srv(
            tmp_path,
            current_model=None,
            models=("Qwen3-32B", "Qwen2.5-7B", "mxbai-embed"),
            log=PRODUCTION_SHAPE,
        )
        asyncio.run(rh.ensure_router_model_ids(srv))

        model, source = rh.resolve_local_model_id(srv)
        assert model == "Qwen3-32B"
        # Two non-embed presets are ambiguous, so the live log signal is
        # canonicalised to the preset's id/casing.
        assert source == "log_signal+models_snapshot"

        # Filtering the log by the preset id finds nothing (the log only ever
        # names ``Qwen3``), yet discovery still resolves the child port.
        log_path = Path(tmp_path) / "llama-server.log"
        assert rh._scan_child_port_signals(log_path, "Qwen3-32B") is None
        assert rh._discover_local_child_port(srv) == CHILD_PORT

    def test_log_signal_is_canonicalised_against_a_loaded_preset(self, tmp_path):
        """With several non-embed presets the live signal is matched, not guessed."""
        srv = _srv(
            tmp_path,
            current_model=None,
            models=("Qwen3-32B", "Qwen2.5-7B", "mxbai-embed"),
            log="srv  proxy_reques: proxying request to model qwen3 on port 41395\n",
        )
        asyncio.run(rh.ensure_router_model_ids(srv))

        model, source = rh.resolve_local_model_id(srv)

        assert model == "Qwen3-32B"
        assert source == "log_signal+models_snapshot"

    def test_unknown_signal_id_is_used_as_the_router_reported_it(self, tmp_path):
        """The router's own id is resolvable by definition — never rewritten."""
        srv = _srv(
            tmp_path,
            current_model=None,
            models=("Qwen3-32B", "Qwen2.5-7B"),
            log="srv  proxy_reques: proxying request to model Qwen3-Coder on port 41395\n",
        )
        asyncio.run(rh.ensure_router_model_ids(srv))

        assert rh.resolve_local_model_id(srv) == ("Qwen3-Coder", "log_signal")

    def test_current_model_wins_over_the_snapshot(self, tmp_path):
        """An explicitly known current model is never replaced by a preset."""
        srv = _srv(
            tmp_path,
            current_model="Qwen2.5-7B",
            models=("Qwen3-32B", "Qwen2.5-7B", "mxbai-embed"),
            log=PRODUCTION_SHAPE,
        )
        asyncio.run(rh.ensure_router_model_ids(srv))

        assert rh.resolve_local_model_id(srv) == ("Qwen2.5-7B", "current_model")

    def test_explicit_model_still_wins(self, tmp_path):
        """A caller-supplied model (e.g. the embed child) is honoured."""
        srv = _srv(tmp_path, current_model="Qwen3-32B", log=PRODUCTION_SHAPE)
        asyncio.run(rh.ensure_router_model_ids(srv))

        assert rh.resolve_local_model_id(srv, "mxbai-embed") == (
            "mxbai-embed",
            "explicit",
        )

    def test_last_resort_is_documented_and_logged(self, tmp_path):
        """``"Qwen3"`` survives only as a *logged* last resort."""
        warnings: list[str] = []
        srv = _srv(
            tmp_path, current_model=None, models=(), warnings=warnings
        )
        asyncio.run(rh.ensure_router_model_ids(srv))

        model, source = rh.resolve_local_model_id(srv)

        assert (model, source) == (rh.LAST_RESORT_MODEL_ID, "last_resort")
        assert any("last_resort" in warning for warning in warnings), warnings

    def test_discovery_uses_the_log_signal_not_the_default(self, tmp_path):
        """Without ``/models`` the router-reported log id is used verbatim."""
        srv = _srv(
            tmp_path,
            current_model=None,
            models=(),
            log="srv  proxy_reques: proxying request to model Qwen3-Coder on port 41395\n",
        )

        assert rh._discover_local_child_port(srv) == CHILD_PORT


# ---------------------------------------------------------------------------
# The call sites: /models preset id + log-signal child port
# ---------------------------------------------------------------------------


class _DedicatedClient:
    """Stand-in for the per-call ``httpx.AsyncClient`` used by the check."""

    def __init__(self, response: httpx.Response, captured: dict):
        self._response = response
        self._captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, timeout=None):
        self._captured["url"] = url
        self._captured.setdefault("attempts", 0)
        self._captured["attempts"] += 1
        return self._response


def _install_dedicated_client(monkeypatch, response, captured) -> None:
    client = _DedicatedClient(response, captured)
    monkeypatch.setattr(
        rh,
        "httpx",
        SimpleNamespace(
            AsyncClient=lambda timeout: client,
            Timeout=lambda value: value,
        ),
    )


class TestNoHardCodedDefaultAtCallSites:
    @pytest.mark.asyncio
    async def test_check_slot_availability_sends_the_models_preset(
        self, tmp_path, monkeypatch
    ):
        """AC2: the emitted ``?model=`` is the preset id, the port is the child."""
        srv = _srv(tmp_path, current_model=None, log=PRODUCTION_SHAPE)
        await rh.ensure_router_model_ids(srv)

        captured: dict = {}
        response = httpx.Response(
            200, json=[{"id": 0, "is_processing": False}]
        )
        _install_dedicated_client(monkeypatch, response, captured)

        result = await rh._check_slot_availability(
            srv,
            {"session_slot_pool_size": 3},
            ROUTER_PORT,
            None,
            None,
            "v1/chat/completions",
        )

        assert result is None  # a free slot -> dispatch may proceed
        assert captured["url"] == (
            f"http://localhost:{CHILD_PORT}/slots?model=Qwen3-32B"
        )

    @pytest.mark.asyncio
    async def test_check_slot_availability_never_sends_the_last_resort_when_known(
        self, tmp_path, monkeypatch
    ):
        """A known ``current_model`` is used verbatim (no ``"Qwen3"`` guess)."""
        srv = _srv(tmp_path, current_model="Qwen2.5-7B", models=())
        await rh.ensure_router_model_ids(srv)

        captured: dict = {}
        response = httpx.Response(
            200, json=[{"id": 0, "is_processing": False}]
        )
        _install_dedicated_client(monkeypatch, response, captured)

        await rh._check_slot_availability(
            srv, {}, ROUTER_PORT, None, None, "v1/chat/completions"
        )

        assert captured["url"].endswith("/slots?model=Qwen2.5-7B")


# ---------------------------------------------------------------------------
# The /models snapshot costs no extra round trip on the hot path
# ---------------------------------------------------------------------------


class TestModelsSnapshotCaching:
    @pytest.mark.asyncio
    async def test_warm_snapshot_costs_zero_models_requests(self, tmp_path):
        """AC: a warm cache resolves the id with no ``/models`` HTTP request."""
        srv = _srv(tmp_path, current_model=None, log=PRODUCTION_SHAPE)
        await rh.ensure_router_model_ids(srv)
        assert srv.models_calls["models"] == 1

        for _ in range(25):
            rh.resolve_local_model_id(srv)
            await rh.ensure_router_model_ids(srv)

        assert srv.models_calls["models"] == 1, (
            "resolution must reuse the cached /models id snapshot"
        )
        assert rh.resolve_local_model_id(srv)[0] == "Qwen3-32B"

    @pytest.mark.asyncio
    async def test_ttl_expiry_refreshes_the_snapshot(self, tmp_path, monkeypatch):
        """The snapshot TTL mirrors ``_child_port_cache_ttl()``."""
        srv = _srv(tmp_path, current_model=None, log=PRODUCTION_SHAPE)
        await rh.ensure_router_model_ids(srv)
        assert rh.resolve_local_model_id(srv)[0] == "Qwen3-32B"

        srv.models = ["Qwen2.5-7B", "mxbai-embed"]
        monkeypatch.setenv("CHILD_PORT_CACHE_TTL", "0")
        await rh.ensure_router_model_ids(srv)

        assert srv.models_calls["models"] == 2
        assert rh.resolve_local_model_id(srv)[0] == "Qwen2.5-7B"

    @pytest.mark.asyncio
    async def test_llama_server_restart_invalidates_the_snapshot(self, tmp_path):
        """A new parent pid re-fetches the ids instead of trusting the old ones."""
        srv = _srv(tmp_path, current_model=None, log=PRODUCTION_SHAPE)
        await rh.ensure_router_model_ids(srv)

        srv.models = ["Qwen2.5-7B", "mxbai-embed"]
        srv.llama_process = SimpleNamespace(pid=999)
        assert rh._cached_router_model_ids(srv) == []

        await rh.ensure_router_model_ids(srv)
        assert srv.models_calls["models"] == 2
        assert rh.resolve_local_model_id(srv)[0] == "Qwen2.5-7B"

    @pytest.mark.asyncio
    async def test_model_switch_invalidates_the_snapshot(self, tmp_path):
        """A changed ``current_model`` re-fetches the ids immediately."""
        srv = _srv(tmp_path, current_model="Qwen3-32B", models=())
        await rh.ensure_router_model_ids(srv)

        srv.current_model = "Qwen2.5-7B"
        srv.models = ["Qwen2.5-7B", "mxbai-embed"]
        assert rh._cached_router_model_ids(srv) == []

        await rh.ensure_router_model_ids(srv)
        assert srv.models_calls["models"] == 2
        assert rh.resolve_local_model_id(srv) == ("Qwen2.5-7B", "current_model")


# ---------------------------------------------------------------------------
# AC3(b): one fail-open result, never a 400-retry loop
# ---------------------------------------------------------------------------


class TestFailOpenLoopGuard:
    @pytest.mark.asyncio
    async def test_unknown_model_yields_one_fail_open_result(
        self, tmp_path, monkeypatch
    ):
        """An unknown id makes exactly one ``/slots`` attempt, then fails open."""
        from proxy import observability as obs

        srv = _srv(tmp_path, current_model=None, log=PRODUCTION_SHAPE)
        monkeypatch.setattr(obs, "_srv", lambda: srv)

        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                400, json={"error": {"message": "model required"}}
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            status_out: dict = {}
            available, total = await obs._query_slots(
                client,
                CHILD_PORT,
                timeout=1.0,
                model="Qwen3-Coder-Next-GGUF:Q5_K_M",
                status_out=status_out,
            )

        assert (available, total) == (0, 0)
        assert len(requests) == 1, "a 400 must never trigger a retry loop"
        assert requests[0].url.params.get("model") == (
            "Qwen3-Coder-Next-GGUF:Q5_K_M"
        )
        assert status_out["reason"] == "http_400"
        # http_400 stays the definitive "this model is not loaded" signal.
        assert obs.slots_failure_is_definitive_model_absent(status_out) is True

    @pytest.mark.asyncio
    async def test_check_slot_availability_fails_open_once(
        self, tmp_path, monkeypatch
    ):
        """The dispatch-path check makes one attempt and never raises/loops."""
        srv = _srv(tmp_path, current_model=None, log=PRODUCTION_SHAPE)
        await rh.ensure_router_model_ids(srv)

        captured: dict = {}
        response = httpx.Response(400, json={"error": {"message": "no model"}})
        _install_dedicated_client(monkeypatch, response, captured)

        result = await rh._check_slot_availability(
            srv, {}, ROUTER_PORT, None, None, "v1/chat/completions"
        )

        assert result is None  # fail-open: no 503, request proceeds
        assert captured["attempts"] == 1
        assert "model=" in captured["url"]


# ---------------------------------------------------------------------------
# AC2: the observability helpers always carry a model id
# ---------------------------------------------------------------------------


class _RecordingClient:
    def __init__(self, captured: dict, payload: list):
        self._captured = captured
        self._payload = payload

    async def get(self, url):
        self._captured["url"] = url
        return httpx.Response(200, json=self._payload)


class TestObservabilityAlwaysSendsAModel:
    @pytest.mark.asyncio
    async def test_slots_detail_resolves_the_model_when_none_is_passed(
        self, tmp_path, monkeypatch
    ):
        from proxy import observability as obs

        srv = _srv(tmp_path, current_model="Qwen3", models=())
        monkeypatch.setattr(obs, "_srv", lambda: srv)
        captured: dict = {}

        await obs._query_slots_detail(
            8080, model=None, _client=_RecordingClient(captured, [])
        )

        assert captured["url"] == "http://localhost:8080/slots?model=Qwen3"

    @pytest.mark.asyncio
    async def test_slots_progress_resolves_the_model_when_none_is_passed(
        self, tmp_path, monkeypatch
    ):
        from proxy import observability as obs

        srv = _srv(tmp_path, current_model=None, log=PRODUCTION_SHAPE)
        await rh.ensure_router_model_ids(srv)
        monkeypatch.setattr(obs, "_srv", lambda: srv)
        captured: dict = {}

        await obs._query_slots_progress(
            8080, model=None, _client=_RecordingClient(captured, [])
        )

        assert captured["url"].endswith("?model=Qwen3-32B")

    @pytest.mark.asyncio
    async def test_slots_counts_resolves_the_model_when_none_is_passed(
        self, tmp_path, monkeypatch
    ):
        from proxy import observability as obs

        srv = _srv(tmp_path, current_model="Qwen2.5-7B", models=())
        monkeypatch.setattr(obs, "_srv", lambda: srv)
        captured: dict = {}

        await obs._query_slots(
            _RecordingClient(captured, [{"id": 0, "is_processing": False}]),
            8080,
            model=None,
        )

        assert captured["url"] == "http://localhost:8080/slots?model=Qwen2.5-7B"
