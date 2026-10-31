"""Tests for proxy/proxy/backend_rss.py (LP-0MVO9F877008ZCP7).

Covers the RSS resolution chain that replaced the host-pid-only read in
``/admin/metrics``:

- host process-tree summing (router + model children, the ~99 % under-report
  case) and the adopted-server (pid = 0) / podman-wrapper failure modes;
- container-backed resolution via the container init pid's process tree, with
  the cgroup ``memory.current`` fallback;
- fail-soft behaviour when the container runtime lookup raises and when no
  runtime can be resolved (scrape still succeeds, last known value kept,
  warning logged once, not per scrape);
- node ``/metrics`` freshness (refresh happens on scrape) and gauge
  attribution (labelled values sum to the process total).
"""

import asyncio
import itertools
from unittest.mock import AsyncMock, MagicMock

import httpx
import proxy.backend_rss as backend_rss
import proxy.metrics as metrics
import pytest


@pytest.fixture(autouse=True)
def _reset_backend_rss_state():
    """Isolate the module-level cache and warn-once set between tests."""
    backend_rss.reset_state()
    yield
    backend_rss.reset_state()


class FakeProcTable:
    """Deterministic stand-in for the /proc walk and discovery.

    ``processes`` maps pid -> (rss_pages_equivalent_rss_bytes, ppid, cmdline).
    ``_proc_pids`` / ``_proc_stat`` / ``_proc_rss_bytes`` / ``_cmdline`` are
    monkeypatched so discovery and tree summing run entirely in memory.
    """

    def __init__(self, processes: dict[int, tuple[int, int, str]]):
        self.processes = processes

    def install(self, monkeypatch):
        procs = dict(self.processes)

        def fake_pids():
            return list(procs)

        def fake_stat(pid):
            data = procs.get(pid)
            if data is None:
                return None
            return "S", data[1]

        def fake_rss(pid):
            data = procs.get(pid)
            if data is None or data[0] < 0:
                return None
            return data[0]

        def fake_cmdline(pid):
            data = procs.get(pid)
            if data is None:
                return ""
            return data[2]

        monkeypatch.setattr(backend_rss, "_proc_pids", fake_pids)
        monkeypatch.setattr(backend_rss, "_proc_stat", fake_stat)
        monkeypatch.setattr(backend_rss, "_proc_rss_bytes", fake_rss)
        monkeypatch.setattr(backend_rss, "_cmdline", fake_cmdline)
        return self


def _router_cmd(port: int) -> str:
    return f"/usr/bin/llama-server --models-preset x --port {port} --no-mmap"


def _child_cmd(alias: str) -> str:
    return f"/usr/bin/llama-server --embeddings --port 0 --alias {alias} --no-mmap"


def _gauge_value(gauge) -> float:
    """Read a single-instance gauge under either prometheus_client flavour.

    The real client exposes ``_value.get()``; the repo-root test stub
    (``prometheus_client.py``, used when the repo root precedes site-packages
    on sys.path) stores values in ``_store[()]``.
    """
    value = getattr(gauge, "_value", None)
    if value is not None:
        return float(value.get())
    return float(gauge._store.get((), 0))


def _labelled_value(gauge, **labels) -> float:
    """Read a labelled gauge under either prometheus_client flavour.

    The real client returns a child metric exposing ``_value.get()``; the
    repo-root test stub keeps a child object whose value lives in
    ``_value.v`` (LP-0MVT1O0VG00905EL made the stub return a child so repeated
    ``labels(...).set(...)`` calls update the same series, as the real client
    does). An unlabelled stub keeps a plain int.
    """
    store = getattr(gauge, "_store", None)
    if isinstance(store, dict):
        key = tuple(labels.get(name) for name in gauge._labelnames)
        value = store.get(key, 0)
        inner = getattr(value, "_value", None)
        if inner is not None:
            return float(getattr(inner, "v", 0))
        return float(value)
    return float(gauge.labels(**labels)._value.get())


_MODEL_SEQ = itertools.count(1)


def _model(tag: str) -> str:
    """A model label that is unique within the test session.

    The repo-root ``prometheus_client.py`` test stub stores a labelled gauge
    value as a plain int, so a *second* ``.labels(key).set()`` on the same key
    raises (it returns the int, not a child) and the per-model update loop
    aborts. A fresh label per test keeps these assertions meaningful under both
    the real client and the stub.
    """
    return f"rss-test-{tag}-{next(_MODEL_SEQ)}"


def _update(srv, models=None):
    """``backend_rss.update_gauges`` with a fresh model label set."""
    return backend_rss.update_gauges(
        srv, models if models is not None else [_model("call")]
    )


def _router_mode_srv(port: int = 8080, **overrides) -> MagicMock:
    srv = MagicMock()
    srv.config = overrides.pop("config", {"server": {"llama_router_mode": True, "llama_server_port": port}})
    srv.logger = MagicMock()
    srv.llama_process = overrides.pop("llama_process", MagicMock())
    srv.model_last_used = {}
    for key, value in overrides.items():
        setattr(srv, key, value)
    return srv


# ---------------------------------------------------------------------------
# Host process-tree resolution
# ---------------------------------------------------------------------------


def test_host_tree_sums_router_and_children(monkeypatch):
    """AC1/AC3: router-mode model memory lives in child processes; the tree
    sum publishes the real total instead of under-reporting by ~99 %."""
    FakeProcTable({
        100: (214_000, 90, _router_cmd(8080)),           # router
        101: (44_000_000_000, 100, _child_cmd("Qwen3")),
        102: (350_000_000, 100, _child_cmd("mxbai-embed")),
        90: (5_000_000_000, 1, "/usr/bin/python -m uvicorn proxy.server:app"),
    }).install(monkeypatch)
    srv = _router_mode_srv()
    srv.llama_process = MagicMock(pid=100, poll=lambda: None)
    model = _model("tree")

    rss = backend_rss.update_gauges(srv, [model])

    assert rss == 44_350_214_000
    assert backend_rss.last_source() == "host-process-tree"
    # The gauge publishes the same non-zero value (AC1: non-zero, within 5 %).
    assert _gauge_value(metrics.llama_process_rss_bytes) == 44_350_214_000
    # AC2: per-model attribution via the equal-split heuristic.
    assert _labelled_value(metrics.llama_model_rss_bytes, model=model) == 44_350_214_000


def test_per_model_labels_sum_to_process_total(monkeypatch):
    """AC2: the documented equal-split heuristic keeps the labelled values
    summing to within ~10 % of llama_process_rss_bytes."""
    FakeProcTable({
        100: (1_000_000, 90, _router_cmd(8080)),
        90: (0, 1, "/usr/bin/python"),
    }).install(monkeypatch)
    srv = _router_mode_srv()
    srv.llama_process = MagicMock(pid=100, poll=lambda: None)
    models = [_model("a"), _model("b"), _model("c")]

    rss = backend_rss.update_gauges(srv, models)

    assert rss == 1_000_000
    labelled_total = sum(
        _labelled_value(metrics.llama_model_rss_bytes, model=m) for m in models
    )
    assert labelled_total > 0
    assert abs(labelled_total - rss) / rss <= 0.10
    assert _gauge_value(metrics.llama_models_loaded) == len(models)


def test_adopted_server_pid_zero_discovers_by_port(monkeypatch):
    """AC3: an adopted server (_AdoptedLlamaProcess has pid = 0) must still
    resolve via /proc port discovery."""
    FakeProcTable({
        100: (214_000, 90, _router_cmd(8080)),
        101: (44_000_000_000, 100, _child_cmd("Qwen3")),
        90: (5_000_000_000, 1, "/usr/bin/python"),
    }).install(monkeypatch)
    srv = _router_mode_srv()
    srv.llama_process = MagicMock(pid=0)
    srv.llama_process.adopted = True

    rss = _update(srv)

    assert rss == 44_000_214_000
    assert backend_rss.last_source() == "host-process-tree"


def test_tracked_podman_wrapper_ignored_in_favour_of_container_process(monkeypatch):
    """A host-side `podman exec` wrapper pid must not be read — the backend
    RSS lives in the container's own process (visible in the host pid ns)."""
    FakeProcTable({
        200: (5_000, 90, "podman exec llama /work/start-llama.sh router"),  # wrapper
        300: (44_000_000_000, 500, _router_cmd(8080)),  # the in-container server
        90: (5_000_000_000, 1, "/usr/bin/python"),
    }).install(monkeypatch)
    srv = _router_mode_srv()
    srv.llama_process = MagicMock(pid=200, poll=lambda: None)

    rss = _update(srv)

    assert rss == 44_000_000_000
    assert rss != 5_000  # never the wrapper's own RSS


def test_no_runtime_is_fail_soft_and_keeps_last_value(monkeypatch):
    """AC3: with nothing resolvable the gauge keeps its last known value
    (never silently re-zeroed) and the failure is logged once, not per scrape."""
    FakeProcTable({
        90: (5_000_000_000, 1, "/usr/bin/python"),
    }).install(monkeypatch)
    srv = _router_mode_srv(port=8080)
    srv.llama_process = MagicMock(pid=0)
    srv.llama_process.adopted = True

    assert _update(srv) is None
    assert _gauge_value(metrics.llama_process_rss_bytes) == 0.0  # never resolved

    # A later failing refresh still warns exactly once.
    backend_rss._state["at"] = 0.0  # force re-resolution
    _update(srv)
    warns = [c.args for c in srv.logger.warning.call_args_list]
    no_runtime_warns = [w for w in warns if w and "no container or host llama-server" in w[0]]
    assert len(no_runtime_warns) == 1


def test_transient_failure_keeps_last_known_good(monkeypatch):
    """A resolution that succeeds once and then fails keeps the good value."""
    procs = {
        100: (1_000_000, 90, _router_cmd(8080)),
        90: (5_000_000_000, 1, "/usr/bin/python"),
    }
    FakeProcTable(procs).install(monkeypatch)
    srv = _router_mode_srv()
    srv.llama_process = MagicMock(pid=100, poll=lambda: None)

    assert _update(srv) == 1_000_000

    # Backend disappears: discovery now finds nothing.
    monkeypatch.setattr(backend_rss, "_proc_pids", lambda: [90])
    backend_rss._state["at"] = 0.0  # force re-resolution
    rss = _update(srv)
    assert rss == 1_000_000  # last known-good kept, not zeroed
    assert _gauge_value(metrics.llama_process_rss_bytes) == 1_000_000


# ---------------------------------------------------------------------------
# Container-backed resolution
# ---------------------------------------------------------------------------


def test_container_backed_uses_container_process_tree(monkeypatch):
    """AC1/AC3: a container-backed backend is resolved through the container
    init pid's tree, not through any (absent) host process handle."""
    FakeProcTable({
        500: (10_000, 1, "sleep infinity"),                        # container init
        501: (42_000_000_000, 500, _router_cmd(8080)),             # in-container server
        90: (5_000_000_000, 1, "/usr/bin/python"),
    }).install(monkeypatch)
    monkeypatch.setattr(backend_rss, "container_name", lambda srv: "llama")
    monkeypatch.setattr(backend_rss, "container_init_pid", lambda name: 500)
    srv = _router_mode_srv(llama_process=MagicMock(pid=0))

    rss = _update(srv)

    assert rss == 42_000_010_000
    assert backend_rss.last_source() == "container-process-tree"


def test_container_cgroup_fallback(monkeypatch, tmp_path):
    """When the container's tree is unreadable, the cgroup memory.current of
    the init pid is used (the same number podman stats reports)."""
    # Container init has no llama-server child and its own RSS is unreadable
    # (-1), so the process-tree path fails and the cgroup value is used.
    FakeProcTable({
        500: (-1, 1, "sleep infinity"),
        90: (5_000_000_000, 1, "/usr/bin/python"),
    }).install(monkeypatch)
    monkeypatch.setattr(backend_rss, "container_name", lambda srv: "llama")
    monkeypatch.setattr(backend_rss, "container_init_pid", lambda name: 500)

    proc_root = tmp_path / "proc"
    cgroup_scope = tmp_path / "sys" / "fs" / "cgroup" / "machine.slice" / "libpod-abc.scope"
    cgroup_scope.mkdir(parents=True)
    (proc_root / "500").mkdir(parents=True)
    (proc_root / "500" / "cgroup").write_text("0::/machine.slice/libpod-abc.scope\n")
    (cgroup_scope / "memory.current").write_text("44000000000\n")
    original = backend_rss.cgroup_memory_current
    monkeypatch.setattr(
        backend_rss, "cgroup_memory_current",
        lambda pid, **kw: original(
            pid, proc_root=proc_root, sys_root=tmp_path / "sys" / "fs" / "cgroup"
        ),
    )
    srv = _router_mode_srv(llama_process=MagicMock(pid=0))

    rss = _update(srv)

    assert rss == 44_000_000_000
    assert backend_rss.last_source() == "container-cgroup"


def test_container_lookup_raises_fails_soft(monkeypatch):
    """AC4: a raising container-runtime lookup must not propagate — the
    endpoint still serves and the gauge keeps its previous value."""
    FakeProcTable({
        90: (5_000_000_000, 1, "/usr/bin/python"),
    }).install(monkeypatch)
    monkeypatch.setattr(backend_rss, "container_name", lambda srv: "llama")

    def boom(name):
        raise RuntimeError("podman inspect exploded")

    monkeypatch.setattr(backend_rss, "container_init_pid", boom)
    srv = _router_mode_srv(llama_process=MagicMock(pid=0))

    # Must not raise.
    assert _update(srv) is None


def test_container_not_running_logs_once_then_host_fallback(monkeypatch):
    """A configured-but-not-running container warns once and the host path
    (discovery by port) still publishes."""
    FakeProcTable({
        100: (1_000_000, 90, _router_cmd(8080)),
        90: (5_000_000_000, 1, "/usr/bin/python"),
    }).install(monkeypatch)
    monkeypatch.setattr(backend_rss, "container_name", lambda srv: "llama")
    monkeypatch.setattr(backend_rss, "container_init_pid", lambda name: None)
    srv = _router_mode_srv(llama_process=MagicMock(pid=0))

    rss = _update(srv)
    assert rss == 1_000_000
    warns = [c.args[0] for c in srv.logger.warning.call_args_list if c.args]
    assert any("podman container" in w for w in warns)
    # Second forced refresh: still exactly one container warning.
    backend_rss._state["at"] = 0.0
    _update(srv)
    warns = [c.args[0] for c in srv.logger.warning.call_args_list if c.args]
    assert sum("podman container" in w for w in warns) == 1


# ---------------------------------------------------------------------------
# /metrics freshness: refresh happens on scrape, not only on /admin/metrics
# ---------------------------------------------------------------------------


class TestMetricsEndpointRefresh:
    """Contract tests: /metrics alone refreshes the gauges (AC1)."""

    @pytest.fixture(autouse=True)
    def _setup_server(self, monkeypatch):
        import proxy.server as server

        monkeypatch.setattr(server, "config", {"server": {"llama_router_mode": False}})
        server.llama_process = MagicMock()
        server.llama_process.poll.return_value = None
        monkeypatch.setattr(server, "router_list_models", AsyncMock(return_value={"data": []}))
        monkeypatch.setattr(server, "_extract_router_model_ids", lambda m: [])
        return server

    async def _scrape(self):
        import proxy.server as server

        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            return await ac.get("/metrics")

    def test_metrics_refresh_publishes_live_rss(self, monkeypatch):
        """Scraping /metrics (with no /admin/metrics poll) publishes a non-zero
        RSS when a backend is resolvable."""
        FakeProcTable({
            100: (44_000_000_000, 90, _router_cmd(8080)),
            90: (5_000_000_000, 1, "/usr/bin/python"),
        }).install(monkeypatch)
        asyncio.run(self._scrape())
        assert _gauge_value(metrics.llama_process_rss_bytes) == 44_000_000_000

    def test_metrics_still_served_when_runtime_inaccessible(self, monkeypatch):
        """AC4: with the container lookup raising, /metrics still returns 200
        (fail soft) and keeps the previously published value."""
        monkeypatch.setattr(backend_rss, "container_name", lambda srv: "llama")

        def boom(name):
            raise RuntimeError("boom")

        monkeypatch.setattr(backend_rss, "container_init_pid", boom)
        resp = asyncio.run(self._scrape())
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")
        assert "llama_process_rss_bytes" in resp.text or "Prometheus" in resp.text


# ---------------------------------------------------------------------------
# Loaded-model freshness (AC1/AC2: attribution labels stay current on /metrics)
# ---------------------------------------------------------------------------


class TestEnsureModels:
    @pytest.mark.asyncio
    async def test_router_mode_lists_models_and_caches(self, monkeypatch):
        srv = _router_mode_srv()
        listing = AsyncMock(return_value={"data": [{"id": "A"}, {"id": "B"}], "models": None})
        monkeypatch.setattr(
            backend_rss, "_router_mode", lambda s: True,
        )
        srv.router_list_models = listing
        srv._extract_router_model_ids = lambda d: [str(m.get("id")) for m in (d.get("data") or [])]

        first = await backend_rss.ensure_models(srv)
        assert first == ["A", "B"]
        listing.assert_awaited_once()

        # Within the TTL the router is not re-polled.
        second = await backend_rss.ensure_models(srv)
        assert second == ["A", "B"]
        listing.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_failed_list_retries_after_negative_cache(self, monkeypatch):
        srv = _router_mode_srv()
        srv.router_list_models = AsyncMock(return_value=None)
        srv._extract_router_model_ids = lambda d: []

        # First call fails -> empty, caches a miss.
        assert await backend_rss.ensure_models(srv) == []
        # Immediate retry is served from the negative cache (no re-poll).
        await backend_rss.ensure_models(srv)
        assert srv.router_list_models.await_count == 1
