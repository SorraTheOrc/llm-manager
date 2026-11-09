"""Baseline pins: the loaded-model child-port fallback signal.

Slice 1 of LP-0MVVCY3HQ0009F0K (chat-model child-port fallback for unloaded
local models), delivered by LP-0MVWBYZKB007H0CA. The parent's biggest risk
is the unloaded-model skip guard (slice 3) "helpfully" suppressing the
genuine loaded-model regression signal that LP-0MVP7525P008RXMF AC3 made
observable (counter + throttled warning). These tests pin **today's**
behaviour **before** the resolver changes, asserting only observable
outcomes — counter values, warning/debug messages, and the requested URL —
so after the guard lands they become the parent AC2 regression pin: a guard
that swallows the signal, or turns an *indeterminate* loadedness into a
skip, fails the suite immediately.

Pinned here:

* a ``/slots`` poll for a model the router's ``/models`` reports as
  **loaded**, with no discoverable child port, still increments
  ``llama_slots_child_port_fallbacks_total{reason="no_child_port_router_mode"}``
  exactly once per poll and emits the throttled
  ``slots_child_port_unavailable`` warning naming that model, and
  ``is_model_known_unloaded()`` says the model is **not** unloaded — so a
  skip-unloaded guard must not skip it;
* single-server mode (``llama_router_mode: false``): the fallback is counted
  under ``no_child_port_single_server`` and logged at debug level only —
  unchanged behaviour;
* **fail-open characterisation**: when loadedness is *indeterminate* (the
  ``/models`` probe errors, returns a non-200 payload, or the model name is
  absent/unresolvable) today's behaviour is to resolve the poll target and
  count the discovery fallback — an unknown answer must never become a skip.

The module is deliberately named ``test_slots_unloaded_model_skip`` (the
future home of the guard's own AC1/AC6/AC7 tests) so the baseline signal
pins live next to the skip tests that must not regress them.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import proxy.metrics as metrics
import pytest

from proxy import router_helpers as rh

ROUTER_PORT = 8080
"""The router llama-server port every fake srv uses (no child discovered)."""

#: Production-shaped /models payload: Qwen3-32B is the loaded chat model.
LOADED_PAYLOAD = {
    "object": "list",
    "data": [
        {"id": "Qwen3-32B", "status": {"value": "loaded"}},
    ],
}

#: Production-shaped /models payload: gpt120 is *definitively* unloaded — the
#: router reports it, so every proxy /slots poll for it must be skipped
#: (LP-0MVWBXEAI008YZ7B AC1).
UNLOADED_PAYLOAD = {
    "object": "list",
    "data": [
        {"id": "gpt120", "status": {"value": "unloaded"}},
    ],
}


def _fallback_counter(reason: str) -> float:
    """Read the real Prometheus counter for one fallback reason label.

    The suite requires prometheus_client (``test_metrics`` asserts
    ``metrics._enabled``), so ``llama_slots_child_port_fallbacks_total`` is a
    real labelled Counter and its value is the metric the AC2 regression pin
    must protect. Reading the label directly also makes a renamed/added reason
    string fail loudly instead of being silently ignored.
    """
    counter = metrics.llama_slots_child_port_fallbacks_total
    assert counter is not None, (
        "prometheus_client must be available for the child-port fallback "
        "counter (the test suite requires metrics._enabled)"
    )
    return float(counter.labels(reason=reason)._value.get())


def _skip_counter(reason: str) -> float:
    """Read the real Prometheus skip counter for one reason label.

    Same reasoning as :func:`_fallback_counter`: reading the actual counter
    makes the AC6 skip signal observable in ``/metrics`` exactly as operators
    would, and a renamed/removed reason string fails loudly.
    """
    counter = metrics.llama_slots_child_port_skipped_total
    assert counter is not None, (
        "prometheus_client must be available for the slots skip counter "
        "(the test suite requires metrics._enabled)"
    )
    return float(counter.labels(reason=reason)._value.get())


@pytest.fixture(autouse=True)
def _clear_signal_state():
    """Reset child-port caches, the /models snapshot and the warning throttle.

    Same pattern as ``test_slots_query_target_call_sites.py`` (child-port
    discovery + fallback-warning throttle) extended with the ``/models``
    snapshot caches from ``test_slots_loaded_set_snapshot.py``, so a warm
    snapshot or a cached miss can never leak between cases.
    """
    for cache in (
        rh._child_port_cache,
        rh._child_port_cache_at,
        rh._child_port_miss_cache,
        rh._child_model_signal_cache,
        rh._router_model_ids_snapshot,
    ):
        cache.clear()
    rh._last_child_port_fallback_warn_ts = 0.0
    rh._last_child_port_skip_debug_ts = 0.0
    yield
    for cache in (
        rh._child_port_cache,
        rh._child_port_cache_at,
        rh._child_port_miss_cache,
        rh._child_model_signal_cache,
        rh._router_model_ids_snapshot,
    ):
        cache.clear()
    rh._last_child_port_fallback_warn_ts = 0.0
    rh._last_child_port_skip_debug_ts = 0.0


def _make_srv(
    tmp_path,
    *,
    router_mode: bool = True,
    payload=LOADED_PAYLOAD,
    raise_on_list: bool = False,
    requests: list | None = None,
):
    """Minimal router-mode server handle for a ``/slots`` poll.

    * ``router_list_models()`` is a counting stub returning *payload*
      (``None`` models the non-200 shape; ``raise_on_list`` models a probe
      error).
    * ``log_dir=tmp_path`` holds no ``llama-server.log``, so child-port
      discovery always fails and every poll exercises the fallback path.
    * When *requests* is given, ``_http_client`` is a request-counting fake
      appending each polled URL — the observable "the poll did happen"
      assertion.
    * ``warn_msgs`` / ``debug_msgs`` collect the logger output so the
      throttled ``slots_child_port_unavailable`` signal is assertable.
    """
    warn_msgs: list[str] = []
    debug_msgs: list[str] = []
    calls = {"models": 0}

    def _warning(message, *args, **kwargs):
        warn_msgs.append(str(message))

    def _debug(message, *args, **kwargs):
        debug_msgs.append(str(message))

    srv = SimpleNamespace(
        log_dir=Path(tmp_path),
        llama_process=SimpleNamespace(pid=41395),
        current_model="Qwen3-32B",
        config={
            "server": {
                "llama_server_port": ROUTER_PORT,
                "llama_router_mode": router_mode,
            }
        },
        logger=SimpleNamespace(warning=_warning, debug=_debug),
        models_calls=calls,
        warn_msgs=warn_msgs,
        debug_msgs=debug_msgs,
        slot_polling_state={},
        _http_client=None,
    )
    srv.payload = payload
    srv.raise_on_list = raise_on_list

    async def _router_list_models():
        calls["models"] += 1
        if srv.raise_on_list:
            raise RuntimeError("probe failed")
        return srv.payload

    srv.router_list_models = _router_list_models

    if requests is not None:

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
                requests.append(url)
                return _FakeResp()

        srv._http_client = _FakeClient()
    return srv


def _run_poll(monkeypatch, srv, model: str, *, max_polls: int = 1):
    """Run the real per-model ``/slots`` poller against a fake srv."""
    from proxy import handlers as handlers_mod

    monkeypatch.setattr(handlers_mod, "_srv", lambda: srv)
    return handlers_mod.poll_slots_for_model(model, llama_port=ROUTER_PORT, max_polls=max_polls)


# ---------------------------------------------------------------------------
# AC2 baseline: a *loaded* model's no-child-port signal must stay observable
# ---------------------------------------------------------------------------


class TestLoadedModelFallbackSignal:
    """The genuine loaded-model regression signal the guard must not swallow."""

    @pytest.mark.asyncio
    async def test_loaded_model_poll_counts_and_warns(self, tmp_path, monkeypatch):
        """A /slots poll for a /models-loaded model with no child port fires
        the fallback exactly once and warns (throttled) naming the model."""
        requests: list[str] = []
        srv = _make_srv(tmp_path, requests=requests)
        await rh.ensure_router_model_ids(srv)
        # The guard's loadedness view says LOADED — a skip-unloaded guard must
        # never skip this model, so the poll below must keep happening.
        assert rh.is_model_known_unloaded(srv, "Qwen3-32B") is False
        assert rh._loaded_models_snapshot(srv) == {"Qwen3-32B"}

        before_router = _fallback_counter("no_child_port_router_mode")
        before_single = _fallback_counter("no_child_port_single_server")
        await _run_poll(monkeypatch, srv, "Qwen3-32B")

        # Exactly one fallback per poll, under the router-mode reason.
        assert _fallback_counter("no_child_port_router_mode") == before_router + 1
        # The other reason label is untouched (per-reason isolation).
        assert _fallback_counter("no_child_port_single_server") == before_single
        # The poll really happened, at the fallback target (the router port).
        assert requests == ["http://localhost:8080/slots?model=Qwen3-32B"]
        # Throttled warning: exactly one, naming the model.
        warns = [m for m in srv.warn_msgs if "slots_child_port_unavailable" in m]
        assert len(warns) == 1
        assert "Qwen3-32B" in warns[0]

    @pytest.mark.asyncio
    async def test_each_poll_counts_once_warning_stays_throttled(self, tmp_path, monkeypatch):
        """Two consecutive polls fire two increments but only one warning.

        The per-poll counter must be exact (a double-count or a swallowed
        poll breaks the regression pin) while the warning stays throttled
        to the 300s interval.
        """
        requests: list[str] = []
        srv = _make_srv(tmp_path, requests=requests)
        await rh.ensure_router_model_ids(srv)
        assert rh.is_model_known_unloaded(srv, "Qwen3-32B") is False

        before = _fallback_counter("no_child_port_router_mode")
        await _run_poll(monkeypatch, srv, "Qwen3-32B")
        await _run_poll(monkeypatch, srv, "Qwen3-32B")

        assert _fallback_counter("no_child_port_router_mode") == before + 2
        assert len(requests) == 2
        warns = [m for m in srv.warn_msgs if "slots_child_port_unavailable" in m]
        assert len(warns) == 1, "repetitive fallbacks stay throttled"

    @pytest.mark.asyncio
    async def test_loaded_poll_needs_no_forced_probe(self, tmp_path, monkeypatch):
        """The pin holds even when no /models snapshot was warmed first.

        The health check for the guard is the loadedness read; the fallback
        signal itself must not depend on any prior /models warm-up — the
        poll path resolves the model from ``current_model`` and fires the
        fallback regardless (LP-0MVP7525P008RXMF AC3 behaviour).
        """
        requests: list[str] = []
        srv = _make_srv(tmp_path, requests=requests)
        # deliberately no ensure_router_model_ids — cold start like a boot

        before = _fallback_counter("no_child_port_router_mode")
        await _run_poll(monkeypatch, srv, "Qwen3-32B")

        assert _fallback_counter("no_child_port_router_mode") == before + 1
        assert requests == ["http://localhost:8080/slots?model=Qwen3-32B"]
        warns = [m for m in srv.warn_msgs if "slots_child_port_unavailable" in m]
        assert len(warns) == 1
        assert "Qwen3-32B" in warns[0]


# ---------------------------------------------------------------------------
# Unchanged single-server mode: counted, debug-only
# ---------------------------------------------------------------------------


class TestSingleServerModeSignal:
    """Single-server mode must keep counting under its own reason, quietly."""

    @pytest.mark.asyncio
    async def test_single_server_poll_counts_but_only_debugs(self, tmp_path, monkeypatch):
        """llama_router_mode: false -> no_child_port_single_server, no warn."""
        requests: list[str] = []
        srv = _make_srv(tmp_path, router_mode=False, requests=requests)

        before_single = _fallback_counter("no_child_port_single_server")
        before_router = _fallback_counter("no_child_port_router_mode")
        await _run_poll(monkeypatch, srv, "Qwen3-32B")

        assert _fallback_counter("no_child_port_single_server") == before_single + 1
        assert _fallback_counter("no_child_port_router_mode") == before_router
        assert requests == ["http://localhost:8080/slots?model=Qwen3-32B"]
        assert [m for m in srv.warn_msgs if "slots_child_port_unavailable" in m] == []
        debugs = [m for m in srv.debug_msgs if "slots_child_port_unavailable" in m]
        assert len(debugs) == 1
        assert "Qwen3-32B" in debugs[0]


# ---------------------------------------------------------------------------
# Fail-open characterisation: indeterminate loadedness must not become a skip
# ---------------------------------------------------------------------------


class TestFailOpenCharacterisation:
    """An unknown loadedness answer keeps today's poll-and-count behaviour."""

    @pytest.mark.asyncio
    async def test_errored_probe_fails_open(self, tmp_path, monkeypatch):
        """A /models probe error -> indeterminate; the poll still happens."""
        requests: list[str] = []
        srv = _make_srv(tmp_path, raise_on_list=True, requests=requests)
        await rh.ensure_router_model_ids(srv)
        assert rh._loaded_models_snapshot(srv) is None
        assert rh.is_model_known_unloaded(srv, "Qwen3-32B") is None, (
            "an errored probe must be indeterminate, never a skip"
        )

        before = _fallback_counter("no_child_port_router_mode")
        await _run_poll(monkeypatch, srv, "Qwen3-32B")

        assert _fallback_counter("no_child_port_router_mode") == before + 1
        assert requests, "an indeterminate probe must not suppress the poll"
        assert [m for m in srv.warn_msgs if "slots_child_port_unavailable" in m]

    @pytest.mark.asyncio
    async def test_non_200_probe_fails_open(self, tmp_path, monkeypatch):
        """A non-200 /models response -> indeterminate; poll + count."""
        requests: list[str] = []
        srv = _make_srv(tmp_path, payload=None, requests=requests)
        await rh.ensure_router_model_ids(srv)
        assert rh._loaded_models_snapshot(srv) is None
        assert rh.is_model_known_unloaded(srv, "Qwen3-32B") is None

        before = _fallback_counter("no_child_port_router_mode")
        await _run_poll(monkeypatch, srv, "Qwen3-32B")

        assert _fallback_counter("no_child_port_router_mode") == before + 1
        assert requests, "a non-200 probe must not suppress the poll"
        assert [m for m in srv.warn_msgs if "slots_child_port_unavailable" in m]

    @pytest.mark.asyncio
    async def test_unparseable_probe_fails_open(self, tmp_path, monkeypatch):
        """An unparseable /models payload (not a dict of lists) is unknown."""
        requests: list[str] = []
        srv = _make_srv(tmp_path, payload=["not", "a", "dict"], requests=requests)
        await rh.ensure_router_model_ids(srv)
        assert rh._loaded_models_snapshot(srv) is None
        assert rh.is_model_known_unloaded(srv, "Qwen3-32B") is None

        before = _fallback_counter("no_child_port_router_mode")
        await _run_poll(monkeypatch, srv, "Qwen3-32B")

        assert _fallback_counter("no_child_port_router_mode") == before + 1
        assert requests
        assert [m for m in srv.warn_msgs if "slots_child_port_unavailable" in m]

    @pytest.mark.asyncio
    async def test_missing_model_name_fails_open(self, tmp_path, monkeypatch):
        """An absent/unresolvable model name is indeterminate, still counted.

        ``is_model_known_unloaded`` returns ``None`` for an empty/missing
        name (never ``True``), and the resolution path with no explicit model
        falls back to ``srv.current_model`` and fires the discovery fallback
        — an unknown name must not silently become a skip.
        """
        srv = _make_srv(tmp_path)
        await rh.ensure_router_model_ids(srv)

        assert rh.is_model_known_unloaded(srv, None) is None
        assert rh.is_model_known_unloaded(srv, "") is None

        before = _fallback_counter("no_child_port_router_mode")
        before_single = _fallback_counter("no_child_port_single_server")
        endpoint, port = rh._resolve_local_slots_target(srv, model=None, router_port=ROUTER_PORT)
        assert (endpoint, port) == (None, ROUTER_PORT), "no child port + router mode -> fall back to the router port"
        assert _fallback_counter("no_child_port_router_mode") == before + 1
        assert _fallback_counter("no_child_port_single_server") == before_single
        warns = [m for m in srv.warn_msgs if "slots_child_port_unavailable" in m]
        assert len(warns) == 1
        # The model is resolved from current_model and named in the warning.
        assert "Qwen3-32B" in warns[0]
        assert "source=current_model" in warns[0]


# ---------------------------------------------------------------------------
# Slice 3 (LP-0MVWBXEAI008YZ7B): AC1/AC6 — the unloaded-model skip itself
# ---------------------------------------------------------------------------


class TestUnloadedModelSkipSignal:
    """AC1/AC6: a definitively unloaded model's poll is skipped, observably.

    The router that unloads idle models reports them via ``/models``; a poll
    for such a model must issue **no** ``/slots`` request and must not touch
    the genuine loaded-model fallback signal
    (``no_child_port_router_mode`` counter, ``slots_child_port_unavailable``
    warning). The deliberate skip must itself be observable
    (``llama_slots_child_port_skipped_total{reason="model_not_loaded"}`` plus
    a throttled debug line naming the model) so AC1 is verifiable post-deploy.
    """

    @pytest.mark.asyncio
    async def test_unloaded_poll_issues_no_request_and_no_fallback(
        self, tmp_path, monkeypatch
    ):
        """AC1: no router request, no fallback counter, no warning.

        A ``/slots`` poll for ``gpt120`` (the router /models snapshot reports
        it as unloaded) resolves to the ``(None, None)`` skip sentinel, so
        ``poll_slots_for_model`` returns before building any request.
        """
        requests: list[str] = []
        srv = _make_srv(tmp_path, payload=UNLOADED_PAYLOAD, requests=requests)
        await rh.ensure_router_model_ids(srv)
        assert rh.is_model_known_unloaded(srv, "gpt120") is True

        before_fb = _fallback_counter("no_child_port_router_mode")
        before_skip = _skip_counter("model_not_loaded")
        await _run_poll(monkeypatch, srv, "gpt120")

        # The genuine fallback signal is untouched (AC1).
        assert _fallback_counter("no_child_port_router_mode") == before_fb
        assert (
            [m for m in srv.warn_msgs if "slots_child_port_unavailable" in m] == []
        )
        # The router is never contacted: no /slots request was issued.
        assert requests == [], "an unloaded model's poll must issue no request"
        # The skip itself is observable in /metrics (AC6).
        assert _skip_counter("model_not_loaded") == before_skip + 1

    @pytest.mark.asyncio
    async def test_skip_debug_log_names_model_and_is_throttled(
        self, tmp_path, monkeypatch
    ):
        """AC6: throttled debug line names the skipped model; counter is exact.

        Two resolver calls for the same unloaded model increment the skip
        counter twice (once per skip) but emit only one (throttled) debug
        line inside the 300 s interval.
        """
        srv = _make_srv(tmp_path, payload=UNLOADED_PAYLOAD)
        await rh.ensure_router_model_ids(srv)

        before = _skip_counter("model_not_loaded")
        rh._resolve_local_slots_target(srv, model="gpt120", router_port=ROUTER_PORT)
        rh._resolve_local_slots_target(srv, model="gpt120", router_port=ROUTER_PORT)

        assert _skip_counter("model_not_loaded") == before + 2
        debugs = [m for m in srv.debug_msgs if "slots_child_port_skipped" in m]
        assert len(debugs) == 1, "repetitive skips stay throttled"
        assert "gpt120" in debugs[0]

    @pytest.mark.asyncio
    async def test_loaded_model_is_never_skipped(self, tmp_path, monkeypatch):
        """AC2 guard: a loaded model keeps polling (no false skip).

        Direct resolver check against the AC2 baseline fixture: the loaded
        chat model resolves to a real target and fires the fallback (visible
        in ``_run_poll``'s request), never ``(None, None)``.
        """
        srv = _make_srv(tmp_path, payload=LOADED_PAYLOAD)
        await rh.ensure_router_model_ids(srv)
        assert rh.is_model_known_unloaded(srv, "Qwen3-32B") is False

        endpoint, port = rh._resolve_local_slots_target(
            srv, model="Qwen3-32B", router_port=ROUTER_PORT
        )
        assert (endpoint, port) == (None, ROUTER_PORT), (
            "a loaded model must never be skipped — it falls back to the "
            "router port (kept observable via the AC2 fallback signal)"
        )


class TestAdmissionUnchanged:
    """AC7: skipping the availability check never refuses or blocks."""

    @pytest.mark.asyncio
    async def test_availability_check_for_unloaded_model_returns_none(
        self, tmp_path, monkeypatch
    ):
        """The skip yields the same admission result (None) as the current
        definitive-400 fall-through: no 503, no /slots request at all."""
        created: list[str] = []

        class _SpyClient:
            def __init__(self, *a, **k):
                created.append("client")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, timeout=None):
                created.append(url)
                return None

        srv = _make_srv(tmp_path, payload=UNLOADED_PAYLOAD)
        await rh.ensure_router_model_ids(srv)
        assert rh.is_model_known_unloaded(srv, "gpt120") is True
        monkeypatch.setattr(
            rh, "httpx",
            SimpleNamespace(
                AsyncClient=lambda timeout: _SpyClient(),
                Timeout=lambda t: t,
            ),
        )

        before = _skip_counter("model_not_loaded")
        result = await rh._check_slot_availability(
            srv,
            {"llama_server_port": ROUTER_PORT,
             "session_slot_availability_timeout_seconds": 1.0},
            ROUTER_PORT,
            "gpt120", "gpt120",
            "v1/chat/completions",
        )

        assert result is None, "an unloaded model must never be refused (AC7)"
        assert created == [], "no /slots request may be issued for the skip"
        assert _skip_counter("model_not_loaded") == before + 1


class TestStatusCapacityTruth:
    """AC3: /status keeps reporting truthful capacity for an unloaded model."""

    @pytest.mark.asyncio
    async def test_status_fails_open_on_pool_without_querying(
        self, tmp_path, monkeypatch
    ):
        """The skip preserves the definitive-absent behaviour: fail open on
        the configured pool, no /slots counts/detail fetch, no fossils."""
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
            port, timeout=2.0, model=None, _client=None, provenance_out=None, status_out=None,
        ):
            queries.append(("detail", port))
            return []

        with monkeypatch.context() as m:
            m.setattr(rh, "_discover_local_child_port", lambda *a, **k: None)
            m.setattr(
                server, "query_llama_status",
                AsyncMock(return_value={"llama_server_running": True}),
            )
            server.current_model = "gpt120"
            server.llama_process = MagicMock(poll=lambda: None)
            server._http_client = None
            server.config = {
                "server": {
                    "llama_server_port": ROUTER_PORT,
                    "session_slot_pool_size": 3,
                }
            }

            async def _list_models():
                return UNLOADED_PAYLOAD

            m.setattr(server, "router_list_models", _list_models)
            await rh.ensure_router_model_ids(server)
            assert rh.is_model_known_unloaded(server, "gpt120") is True, (
                "fixture: gpt120 must be definitively unloaded for the /status skip"
            )

            m.setattr("proxy.observability._query_slots", fake_query_slots)
            m.setattr("proxy.observability._query_slots_detail", fake_query_detail)

            req = MagicMock()
            req.headers = {}
            req.client = MagicMock(host="127.0.0.1", port=54321)
            j = await get_llama_local_status(req)

        # Truthful capacity: fail open on the configured pool (AC3), same as
        # today's definitive-400 path — never fossil counts.
        assert j["total_slots"] == 3
        assert j["available_slots"] == 3
        # The skip means zero upstream /slots requests were issued.
        assert queries == [], "an unloaded model's /status must skip /slots"
