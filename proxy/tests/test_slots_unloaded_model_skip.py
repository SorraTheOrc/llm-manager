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
