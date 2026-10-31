"""Slot-occupancy sampler for the proxy's Prometheus surface.

LP-0MVT1O0VG00905EL (parent LP-0MVNWUEY8008XM6W). The workload half of the GPU
telemetry epic: power/VRAM/GTT already arrive through node_exporter, but nothing
published llama-server **slot occupancy**, so "the GPU drew 115 W" could not be
attributed to llama-server work.

What is pinned here, against the real gauges and the real coalesced ``/slots``
path (the payload builders live in ``gpu_fixtures.py``):

- interval configurability — shipped default 10 s, override, invalid → default,
  ``<= 0`` → sampler disabled;
- healthy publication of every gauge from a lean (deployed) payload and from a
  rich payload carrying ``id_task``;
- the parent brief's trap: **a failed ``/slots`` query writes no gauge at all**,
  never ``llama_slots_busy 0``, and the failure stays visible through
  ``llama_slots_query_failures_total{reason=...}``;
- the heartbeat ``llama_slots_sample_timestamp_seconds`` advances only on a
  successful sample, so a stalled sampler is distinguishable from idle slots;
- per-slot ``slot=`` label cleanup when a slot disappears from the payload (a
  restarted llama-server must not leave ghost series);
- the optional-``prometheus_client`` no-op path: the proxy still serves
  ``/metrics`` and the sampler raises nothing when the library is absent;
- the sampler's reuse of child-port discovery and the coalesced fetch with a
  ``/slots`` budget no larger than the existing 2 s default.

Deliberately not asserted here: the full ``lifespan`` startup body (it spawns
the container, TTS and podman work). The two launcher helpers it delegates to
are tested directly instead, which is where the "only when metrics are enabled"
gate actually lives.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import proxy.metrics as metrics
import proxy.observability as obs
import proxy.server as server
import pytest
import yaml
from gpu_fixtures import (
    empty_slots_payload,
    expected_busy_slots,
    expected_in_flight_tasks,
    mixed_slots_payload,
    rich_slots_payload,
)

_CONFIG_YAML = Path(__file__).resolve().parents[1] / "config.yaml"
_URL = "http://localhost:58113/slots?model=Qwen3"

#: Snapshot used by the pure-publication cases.
SNAPSHOT = {
    "total_slots": 4,
    "busy_slots": 2,
    "in_flight_tasks": 2,
    "slots": [
        {"slot_id": 0, "busy": True},
        {"slot_id": 1, "busy": False},
        {"slot_id": 2, "busy": True},
        {"slot_id": 3, "busy": False},
    ],
    "sampled_at": 1_700_000_000.0,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_slot_occupancy_gauges():
    """Start and end every case without ghost per-slot series.

    The per-slot label set tracked by ``record_slot_occupancy()`` is
    process-global; without a reset an earlier case's slots would look like a
    disappeared slot in a later case and vice-versa.
    """
    metrics.reset_slot_occupancy_gauges()
    yield
    metrics.reset_slot_occupancy_gauges()


@pytest.fixture(autouse=True)
def _clear_child_port_state():
    """Keep child-port discovery deterministic across cases."""
    from proxy import router_helpers as rh

    rh._child_port_cache.clear()
    rh._child_port_cache_at.clear()
    rh._child_port_miss_cache.clear()
    yield
    rh._child_port_cache.clear()
    rh._child_port_cache_at.clear()
    rh._child_port_miss_cache.clear()


@pytest.fixture
def srv():
    """A minimal server stand-in carrying only what the sampler reads."""
    return SimpleNamespace(
        config={"server": {"llama_server_port": 8080}},
        current_model="Qwen3",
        llama_process=SimpleNamespace(pid=41395),
        log_dir=Path("/nonexistent"),
        logger=SimpleNamespace(
            warning=lambda *a, **k: None,
            debug=lambda *a, **k: None,
            info=lambda *a, **k: None,
        ),
    )


def _require_client() -> None:
    if not metrics._enabled:
        pytest.skip("prometheus_client not installed: gauges are no-ops")


def _gauge_value(gauge, **labels):
    """Read a gauge value from either prometheus_client implementation.

    Mirrors ``test_quota_observability._gauge_value``: the repo-root
    ``prometheus_client.py`` test stub returns the stored int (or a small store
    object) instead of a real labelled child, and pytest runs from the repo root
    under the dev-push gate, so the reader adapts to whichever library loaded.
    """
    if labels:
        child = gauge.labels(**labels)
        if isinstance(child, (int, float)):
            return child
        value = getattr(child, "_value", None)
        if value is not None and hasattr(value, "get"):
            return value.get()
        store = getattr(child, "_store", None)
        key = getattr(child, "_key", None)
        assert store is not None and key in store, "unreadable gauge implementation"
        return store[key]
    value = getattr(gauge, "_value", None)
    if value is not None and hasattr(value, "get"):
        return value.get()
    store = getattr(gauge, "_store", None)
    assert store is not None and () in store, "unreadable gauge implementation"
    return store[()]


def _slot_series(gauge) -> set[str]:
    """Return the ``slot=`` label values that currently exist on *gauge*.

    Read from the client's own child-series map (``_metrics`` on the real
    library, ``_store`` on the repo-root stub) rather than by string-matching
    ``generate_latest()``, so the assertion means "the series is gone" and not
    "the exposition text did not contain that substring".
    """
    children = getattr(gauge, "_metrics", None)
    if children is None:
        children = getattr(gauge, "_store", None)
    assert children is not None, "unreadable gauge implementation"
    return {str(key[0]) for key in children if key}


async def _wait_until(predicate, *, timeout: float = 3.0) -> bool:
    """Poll *predicate* until true or *timeout* elapses (bounded, no fixed sleep)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return predicate()


# ---------------------------------------------------------------------------
# Interval configurability (AC1, AC7)
# ---------------------------------------------------------------------------


def test_sampler_default_interval_is_ten_seconds():
    """The documented default is 10 s, and it is what an absent key yields."""
    assert obs.DEFAULT_SLOT_METRICS_INTERVAL_SECONDS == 10.0
    assert obs._slot_metrics_interval_seconds({}) == 10.0
    assert obs._slot_metrics_interval_seconds(None) == 10.0


def test_sampler_interval_is_overridable_from_server_config():
    """``server.llama_slot_metrics_interval_seconds`` overrides the default."""
    assert obs._slot_metrics_interval_seconds(
        {"llama_slot_metrics_interval_seconds": 3.5}
    ) == 3.5
    # A YAML string (quoted value) is coerced rather than silently ignored.
    assert obs._slot_metrics_interval_seconds(
        {"llama_slot_metrics_interval_seconds": "2"}
    ) == 2.0


@pytest.mark.parametrize("raw", ["not-a-number", None, [], {}])
def test_invalid_sampler_interval_falls_back_to_the_default(raw):
    """A malformed value never disables the sampler by accident."""
    assert obs._slot_metrics_interval_seconds(
        {"llama_slot_metrics_interval_seconds": raw}
    ) == 10.0


@pytest.mark.parametrize("raw", [0, 0.0, -1])
def test_non_positive_interval_disables_the_sampler(raw):
    """``0``/negative is an explicit opt-out, distinguishable from "invalid"."""
    assert obs._slot_metrics_interval_seconds(
        {"llama_slot_metrics_interval_seconds": raw}
    ) == 0.0


def test_shipped_config_declares_the_documented_default():
    """``config.yaml`` ships the default the reader and the docs claim."""
    with _CONFIG_YAML.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    server_cfg = config["server"]
    assert "llama_slot_metrics_interval_seconds" in server_cfg, (
        "the sampler interval is documented next to the wedge-detector keys and "
        "must be present in proxy/config.yaml"
    )
    assert server_cfg["llama_slot_metrics_interval_seconds"] == 10
    assert obs._slot_metrics_interval_seconds(server_cfg) == (
        obs.DEFAULT_SLOT_METRICS_INTERVAL_SECONDS
    )


# ---------------------------------------------------------------------------
# Payload → snapshot derivation
# ---------------------------------------------------------------------------


def test_snapshot_from_lean_payload_matches_the_consumer_contract():
    """A lean (deployed) payload yields total/busy/in-flight/per-slot state."""
    payload = mixed_slots_payload(total=4, busy_ids=(0, 2))

    snapshot = metrics.summarise_slot_occupancy(payload, sampled_at=12.5)

    assert snapshot["total_slots"] == len(payload) == 4
    assert snapshot["busy_slots"] == expected_busy_slots(payload) == 2
    # This build publishes no id_task, so the documented fallback counts
    # is_processing; it must agree with the busy count rather than invent tasks.
    assert snapshot["in_flight_tasks"] == expected_in_flight_tasks(payload) == 2
    assert snapshot["slots"] == [
        {"slot_id": 0, "busy": True},
        {"slot_id": 1, "busy": False},
        {"slot_id": 2, "busy": True},
        {"slot_id": 3, "busy": False},
    ]
    assert snapshot["sampled_at"] == 12.5


def test_snapshot_prefers_id_task_for_in_flight_when_published():
    """A rich build's ``id_task`` is the in-flight count when the slot works.

    The rich fixture marks two slots processing *and* assigns exactly those two
    real task ids.
    """
    payload = rich_slots_payload()

    snapshot = metrics.summarise_slot_occupancy(payload, sampled_at=0.0)

    assert snapshot["total_slots"] == 4
    assert snapshot["busy_slots"] == 2
    assert snapshot["in_flight_tasks"] == expected_in_flight_tasks(payload) == 2


def test_snapshot_does_not_count_a_stale_id_task_on_an_idle_slot():
    """Measured live 2026-10-31: an idle slot keeps its previous ``id_task``.

    The deployed build publishes ``id_task`` for the Qwen3 child and slot 1 was
    observed as ``{"is_processing": false, "id_task": 20026}`` beside a
    processing slot with ``id_task: 24207``. Counting ``id_task >= 0`` alone
    would publish ``llama_slots_in_flight_tasks 2`` while one slot worked, so
    the rule requires ``is_processing`` too. The ``-1`` sentinel is excluded on
    both paths.
    """
    payload = [
        {"id": 0, "is_processing": True, "id_task": 24207, "n_ctx": 131072},
        {"id": 1, "is_processing": False, "id_task": 20026, "n_ctx": 131072},
        {"id": 2, "is_processing": False, "id_task": -1, "n_ctx": 131072},
    ]

    snapshot = metrics.summarise_slot_occupancy(payload, sampled_at=0.0)

    assert snapshot["total_slots"] == 3
    assert snapshot["busy_slots"] == 1
    assert snapshot["in_flight_tasks"] == 1, (
        "only the processing slot carries an in-flight task; the idle slot's "
        "retained id_task must not be counted"
    )


def test_snapshot_of_an_empty_backend_is_a_truthful_zero():
    """``[]`` with HTTP 200 is a real answer: zero slots, zero busy."""
    snapshot = metrics.summarise_slot_occupancy(empty_slots_payload(), sampled_at=1.0)

    assert snapshot["total_slots"] == 0
    assert snapshot["busy_slots"] == 0
    assert snapshot["in_flight_tasks"] == 0
    assert snapshot["slots"] == []


def test_snapshot_rejects_a_non_list_payload():
    """A dict/None/error body is a caller bug, never a zero-occupancy sample."""
    for bad in (None, {}, {"slots": []}, "[]"):
        with pytest.raises(TypeError):
            metrics.summarise_slot_occupancy(bad, sampled_at=0.0)


# ---------------------------------------------------------------------------
# Publication (AC1, AC2)
# ---------------------------------------------------------------------------


def test_record_publishes_every_gauge():
    """One successful snapshot sets all five series the card names."""
    _require_client()

    assert metrics.record_slot_occupancy(SNAPSHOT) is True

    assert _gauge_value(metrics.llama_slots_total) == 4
    assert _gauge_value(metrics.llama_slots_busy) == 2
    assert _gauge_value(metrics.llama_slots_in_flight_tasks) == 2
    assert _gauge_value(metrics.llama_slots_sample_timestamp_seconds) == int(
        SNAPSHOT["sampled_at"]
    )
    assert {
        slot: _gauge_value(metrics.llama_slot_busy, slot=slot)
        for slot in ("0", "1", "2", "3")
    } == {"0": 1.0, "1": 0.0, "2": 1.0, "3": 0.0}


def test_record_none_writes_nothing_at_all():
    """A failed sample must not publish ``0`` busy slots (the card's trap).

    Every gauge keeps its previous value — including the heartbeat, which is
    what makes staleness queryable.
    """
    _require_client()
    metrics.record_slot_occupancy(SNAPSHOT)

    assert metrics.record_slot_occupancy(None) is False

    assert _gauge_value(metrics.llama_slots_total) == 4
    assert _gauge_value(metrics.llama_slots_busy) == 2
    assert _gauge_value(metrics.llama_slots_in_flight_tasks) == 2
    assert _gauge_value(metrics.llama_slots_sample_timestamp_seconds) == int(
        SNAPSHOT["sampled_at"]
    )
    assert _gauge_value(metrics.llama_slot_busy, slot="0") == 1.0
    assert _gauge_value(metrics.llama_slot_busy, slot="1") == 0.0


def test_heartbeat_advances_only_on_a_successful_sample():
    """The timestamp is a heartbeat: success advances it, failure does not."""
    _require_client()

    metrics.record_slot_occupancy({**SNAPSHOT, "sampled_at": 1000.0})
    assert _gauge_value(metrics.llama_slots_sample_timestamp_seconds) == 1000.0

    metrics.record_slot_occupancy(None)
    assert _gauge_value(metrics.llama_slots_sample_timestamp_seconds) == 1000.0

    metrics.record_slot_occupancy({**SNAPSHOT, "sampled_at": 1060.0})
    assert _gauge_value(metrics.llama_slots_sample_timestamp_seconds) == 1060.0


def test_record_removes_per_slot_labels_for_disappeared_slots():
    """A restarted backend with fewer slots leaves no ghost ``slot=`` series."""
    _require_client()
    metrics.record_slot_occupancy(SNAPSHOT)  # slots 0..3
    assert _slot_series(metrics.llama_slot_busy) == {"0", "1", "2", "3"}

    shrunk = {
        "total_slots": 2,
        "busy_slots": 0,
        "in_flight_tasks": 0,
        "slots": [{"slot_id": 0, "busy": False}, {"slot_id": 1, "busy": False}],
        "sampled_at": SNAPSHOT["sampled_at"] + 10,
    }
    assert metrics.record_slot_occupancy(shrunk) is True

    assert _slot_series(metrics.llama_slot_busy) == {"0", "1"}, (
        "slot=2/slot=3 must be dropped, not left reporting their final value"
    )
    assert metrics._slot_busy_labelvalues == {"0", "1"}
    assert _gauge_value(metrics.llama_slots_total) == 2
    assert _gauge_value(metrics.llama_slots_busy) == 0
    assert _gauge_value(metrics.llama_slot_busy, slot="1") == 0.0


# ---------------------------------------------------------------------------
# Sampler fetch path (AC1, AC4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sampler_publishes_a_healthy_sample(srv, monkeypatch):
    """A 200 payload reaches every gauge, heartbeat included."""
    _require_client()
    payload = mixed_slots_payload(total=4, busy_ids=(0, 2))
    seen = {}

    async def fake_fetch(url, timeout, client=None):
        seen["url"] = url
        seen["timeout"] = timeout
        return 200, payload

    monkeypatch.setattr(obs, "_fetch_slots_payload", fake_fetch)
    from proxy import router_helpers as rh

    monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)

    snapshot = await obs._sample_slot_occupancy(srv)
    assert metrics.record_slot_occupancy(snapshot) is True

    assert seen["url"] == _URL
    assert seen["timeout"] <= 2.0, "/slots budget must stay within the 2 s default"
    assert _gauge_value(metrics.llama_slots_total) == 4
    assert _gauge_value(metrics.llama_slots_busy) == 2
    assert _gauge_value(metrics.llama_slots_in_flight_tasks) == 2
    assert abs(
        _gauge_value(metrics.llama_slots_sample_timestamp_seconds) - time.time()
    ) < 5
    assert _gauge_value(metrics.llama_slot_busy, slot="0") == 1.0
    assert _gauge_value(metrics.llama_slot_busy, slot="1") == 0.0


@pytest.mark.asyncio
async def test_sampler_uses_the_coalesced_fetch_and_child_port_discovery(srv, monkeypatch):
    """Load on ``/slots`` stays bounded: the shared coalesced path, child port.

    A dedicated poller that bypasses ``_fetch_slots_payload`` would multiply the
    upstream request rate on an endpoint that has already produced 1 392 timeout
    failures on this host, so the call is asserted, not assumed.
    """
    _require_client()
    calls = []

    async def fake_fetch(url, timeout, client=None):
        calls.append((url, timeout))
        return 200, mixed_slots_payload(total=1, busy_ids=())

    monkeypatch.setattr(obs, "_fetch_slots_payload", fake_fetch)
    from proxy import router_helpers as rh

    monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)

    await obs._sample_slot_occupancy(srv)

    assert calls == [(_URL, obs._SLOT_METRICS_TIMEOUT_SECONDS)]
    assert calls[0][1] <= 2.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,expected_reason",
    [
        ("timeout", "timeout"),
        ("connection_error", "connection_error"),
        ("http_500", "http_500"),
        ("invalid_payload", "invalid_payload"),
    ],
)
async def test_failed_sample_publishes_no_gauge_and_counts_the_failure(
    srv, monkeypatch, failure, expected_reason
):
    """Failure ⇒ no gauge write, but the existing failure counter still moves."""
    _require_client()
    metrics.record_slot_occupancy(SNAPSHOT)  # a known-good prior sample
    before_ts = _gauge_value(metrics.llama_slots_sample_timestamp_seconds)
    before_busy = _gauge_value(metrics.llama_slots_busy)
    before_failures = _gauge_value(
        metrics.llama_slots_query_failures_total.labels(reason=expected_reason)
    )

    if failure == "timeout":
        async def fake_fetch(url, timeout, client=None):
            raise TimeoutError("budget exhausted")
    elif failure == "connection_error":
        async def fake_fetch(url, timeout, client=None):
            raise ConnectionError("connection refused")
    elif failure == "http_500":
        async def fake_fetch(url, timeout, client=None):
            return 500, None
    else:
        async def fake_fetch(url, timeout, client=None):
            return 200, {"not": "a list"}

    monkeypatch.setattr(obs, "_fetch_slots_payload", fake_fetch)
    from proxy import router_helpers as rh

    monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)

    snapshot = await obs._sample_slot_occupancy(srv)
    assert snapshot is None
    assert metrics.record_slot_occupancy(snapshot) is False
    assert _gauge_value(
        metrics.llama_slots_query_failures_total.labels(reason=expected_reason)
    ) == before_failures + 1
    # The trap: never ``llama_slots_busy 0`` for a query that did not answer.
    assert _gauge_value(metrics.llama_slots_busy) == before_busy
    assert _gauge_value(metrics.llama_slots_sample_timestamp_seconds) == before_ts


@pytest.mark.asyncio
async def test_heartbeat_does_not_advance_across_a_recovered_failure(srv, monkeypatch):
    """success → failure → success: the gap stays visible in the heartbeat."""
    _require_client()
    payload = mixed_slots_payload(total=2, busy_ids=(1,))
    from proxy import router_helpers as rh

    monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)

    async def ok(url, timeout, client=None):
        return 200, payload

    async def broken(url, timeout, client=None):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(obs, "_fetch_slots_payload", ok)
    first = await obs._sample_slot_occupancy(srv)
    metrics.record_slot_occupancy(first)
    ts_after_success = _gauge_value(metrics.llama_slots_sample_timestamp_seconds)

    monkeypatch.setattr(obs, "_fetch_slots_payload", broken)
    failed = await obs._sample_slot_occupancy(srv)
    metrics.record_slot_occupancy(failed)
    assert _gauge_value(metrics.llama_slots_sample_timestamp_seconds) == ts_after_success
    assert _gauge_value(metrics.llama_slots_busy) == 1

    monkeypatch.setattr(obs, "_fetch_slots_payload", ok)
    # Let the wall clock move so the recovered heartbeat is strictly newer.
    await asyncio.sleep(0.01)
    recovered = await obs._sample_slot_occupancy(srv)
    metrics.record_slot_occupancy(recovered)
    # Compared on the snapshots (float) and on the gauge at epoch-second
    # resolution: the repo-root prometheus_client test stub stores gauge values
    # as ints, so a sub-second delta is not representable there.
    assert recovered["sampled_at"] > first["sampled_at"]
    assert abs(
        _gauge_value(metrics.llama_slots_sample_timestamp_seconds)
        - recovered["sampled_at"]
    ) < 1, "the gauge must carry the recovered sample's timestamp"


# ---------------------------------------------------------------------------
# Sampler loop (AC1, AC5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_samples_immediately_then_waits_the_interval(monkeypatch, srv):
    """The first sample is taken at startup, not one interval later."""
    calls = []

    async def fake_sample(_srv):
        calls.append(time.monotonic())
        return None

    monkeypatch.setattr(obs, "_sample_slot_occupancy", fake_sample)
    task = asyncio.create_task(obs._slot_metrics_loop(interval_seconds=30.0, srv=srv))
    try:
        assert await _wait_until(lambda: len(calls) >= 1)
        await asyncio.sleep(0.05)
        assert len(calls) == 1, "a 30 s interval must not sample twice in 50 ms"
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_loop_repeats_on_the_configured_interval(monkeypatch, srv):
    """A short interval yields repeated samples (the loop actually loops)."""
    calls = []

    async def fake_sample(_srv):
        calls.append(1)
        return None

    monkeypatch.setattr(obs, "_sample_slot_occupancy", fake_sample)
    task = asyncio.create_task(obs._slot_metrics_loop(interval_seconds=0.01, srv=srv))
    try:
        assert await _wait_until(lambda: len(calls) >= 3)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_loop_is_a_noop_when_the_interval_disables_it(monkeypatch, srv):
    """``<= 0`` stops the loop before it samples anything."""
    calls = []

    async def fake_sample(_srv):
        calls.append(1)
        return None

    monkeypatch.setattr(obs, "_sample_slot_occupancy", fake_sample)

    await asyncio.wait_for(obs._slot_metrics_loop(interval_seconds=0, srv=srv), 1.0)

    assert calls == []


@pytest.mark.asyncio
async def test_loop_survives_a_raising_sample(monkeypatch, srv):
    """An exception in one iteration must not kill the sampler (or the proxy)."""
    _require_client()
    payload = mixed_slots_payload(total=2, busy_ids=(0,))
    attempts = []

    async def flaky_sample(_srv):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("boom")
        return metrics.summarise_slot_occupancy(payload, sampled_at=42.0)

    monkeypatch.setattr(obs, "_sample_slot_occupancy", flaky_sample)
    metrics.record_slot_occupancy(SNAPSHOT)  # a prior good sample exists

    task = asyncio.create_task(obs._slot_metrics_loop(interval_seconds=0.01, srv=srv))
    try:
        assert await _wait_until(
            lambda: _gauge_value(metrics.llama_slots_sample_timestamp_seconds) == 42.0
        )
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert len(attempts) >= 2
    assert _gauge_value(metrics.llama_slots_busy) == 1


# ---------------------------------------------------------------------------
# Lifespan launchers — "only when metrics are enabled"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_launch_is_skipped_when_prometheus_client_is_absent(monkeypatch):
    """No sampler task is created when the gauges are unavailable."""
    monkeypatch.setattr(metrics, "_enabled", False)
    monkeypatch.setattr(server, "slot_metrics_task", None)

    server._startup_launch_slot_metrics_sampler()

    assert server.slot_metrics_task is None


@pytest.mark.asyncio
async def test_launch_creates_exactly_one_sampler_task(monkeypatch):
    """Idempotent: a second startup call does not spawn a second sampler."""
    _require_client()
    monkeypatch.setattr(server, "slot_metrics_task", None)

    server._startup_launch_slot_metrics_sampler()
    task = server.slot_metrics_task
    try:
        assert task is not None and not task.done()
        server._startup_launch_slot_metrics_sampler()
        assert server.slot_metrics_task is task
    finally:
        server._shutdown_stop_slot_metrics_sampler()
    assert server.slot_metrics_task is None


# ---------------------------------------------------------------------------
# Optional prometheus_client (AC3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_absent_client_keeps_metrics_served_and_the_sampler_silent(
    monkeypatch, srv
):
    """Without prometheus_client: ``/metrics`` still answers, sampler no-ops."""
    payload = mixed_slots_payload(total=2, busy_ids=(0,))

    async def fake_fetch(url, timeout, client=None):
        return 200, payload

    monkeypatch.setattr(obs, "_fetch_slots_payload", fake_fetch)
    from proxy import router_helpers as rh

    monkeypatch.setattr(rh, "_discover_local_child_port", lambda *a, **k: 58113)

    with monkeypatch.context() as ctx:
        ctx.setattr(metrics, "_enabled", False)
        for gauge_name in (
            "llama_slots_total",
            "llama_slots_busy",
            "llama_slot_busy",
            "llama_slots_in_flight_tasks",
            "llama_slots_sample_timestamp_seconds",
        ):
            ctx.setattr(metrics, gauge_name, None)

        # The sampler must still resolve a snapshot (it is transport-only) …
        snapshot = await obs._sample_slot_occupancy(srv)
        assert snapshot is not None and snapshot["busy_slots"] == 1
        # … and publication must be a silent no-op rather than an exception.
        assert metrics.record_slot_occupancy(snapshot) is False

        monkeypatch.setattr(server, "config", {"server": {}})
        monkeypatch.setattr(server, "llama_process", SimpleNamespace(poll=lambda: None))
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.get("/metrics")

    assert resp.status_code == 200
    assert "text/plain" in resp.headers.get("content-type", "")
    assert "metrics disabled" in resp.text
