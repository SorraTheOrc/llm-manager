"""Prefill serialisation (LP-0MVGDUUV3001XFRJ).

Prefill does not scale with concurrency on this backend: 47.8 % of local
prefills shared the GPU with at least one other, per-request first-byte p50 was
85.0 s with one overlap vs 23.4 s alone, while aggregate prefill throughput
stayed flat (~254 tok/s). At most ``local_max_concurrent_prefills`` (default 1)
prefills now run at once; decoding in another slot is unaffected.

Covers the ACs:

- AC1: ``server.local_max_concurrent_prefills`` (default 1, ``0`` disables)
  with validation;
- AC2: a second concurrent prefill is denied, a decoding slot does not block,
  a request that cannot be admitted falls back exactly as today, and disabling
  restores the previous behaviour;
- AC4: ``prefill_gate_wait_ms`` is emitted per request and the gate state is
  published on ``/llama/local/status``;
- AC5: a lone request is never delayed by the gate.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from proxy.router_helpers import (
    DEFAULT_MAX_CONCURRENT_PREFILLS,
    _effective_prefill_cap,
    _max_concurrent_prefills,
    _try_acquire_local_dispatch,
)

from proxy import contention_queue

POOL = 2


@pytest.fixture
def srv(monkeypatch):
    """A minimal proxy.server stand-in with empty dispatch + prefill state."""
    import proxy.server as server

    monkeypatch.setattr(server, "local_dispatch_records", {}, raising=False)
    monkeypatch.setattr(
        server, "local_dispatch_records_lock", asyncio.Lock(), raising=False
    )
    monkeypatch.setattr(server, "local_active_queries", 0, raising=False)
    monkeypatch.setattr(
        server, "local_active_queries_lock", asyncio.Lock(), raising=False
    )
    monkeypatch.setattr(server, "local_generating_queries", 0, raising=False)
    monkeypatch.setattr(server, "local_generating_sessions", set(), raising=False)
    monkeypatch.setattr(
        server, "local_generating_queries_lock", asyncio.Lock(), raising=False
    )
    monkeypatch.setattr(server, "local_prefill_in_flight", {}, raising=False)
    monkeypatch.setattr(
        server, "local_prefill_in_flight_lock", asyncio.Lock(), raising=False
    )
    monkeypatch.setattr(server, "logger", MagicMock(), raising=False)
    monkeypatch.setattr(
        server,
        "config",
        {
            "server": {
                "local_dispatch_lease_timeout_seconds": 60,
                "session_slot_pool_size": POOL,
            }
        },
        raising=False,
    )
    contention_queue.reset()
    yield server
    contention_queue.reset()


def _config(**server_overrides) -> dict:
    server = {"session_slot_pool_size": POOL}
    server.update(server_overrides)
    return {"server": server}


def _set_prefills_cap(srv, value) -> None:
    srv.config = {
        "server": {
            "local_dispatch_lease_timeout_seconds": 60,
            "session_slot_pool_size": POOL,
            "local_max_concurrent_prefills": value,
        }
    }


class TestPrefillCapConfig:
    """AC1 — the knob is defaulted, clamped and validated."""

    def test_default_is_one(self):
        assert _max_concurrent_prefills({}) == DEFAULT_MAX_CONCURRENT_PREFILLS == 1

    def test_zero_disables(self):
        assert _max_concurrent_prefills({"local_max_concurrent_prefills": 0}) == 0

    @pytest.mark.parametrize("bad", ["not-a-number", None, -3, [1]])
    def test_malformed_or_negative_falls_back_to_default(self, bad):
        assert _max_concurrent_prefills({"local_max_concurrent_prefills": bad}) == 1

    def test_explicit_value(self):
        assert _max_concurrent_prefills({"local_max_concurrent_prefills": 3}) == 3

    def test_effective_cap_clamps_to_pool(self):
        assert _effective_prefill_cap({}, POOL) == 1
        assert _effective_prefill_cap({"local_max_concurrent_prefills": 5}, POOL) == POOL
        assert _effective_prefill_cap({"local_max_concurrent_prefills": 0}, POOL) == POOL


class TestPrefillGate:
    """AC2/AC5 — the gate's decision matrix."""

    @pytest.mark.asyncio
    async def test_second_concurrent_prefill_is_denied(self, srv):
        first = await _try_acquire_local_dispatch(
            srv, max_local=POOL, session_key="sess-a", backend="local"
        )
        assert first[0] is True
        assert srv.local_prefill_in_flight == {"sess-a": True}

        second = await _try_acquire_local_dispatch(
            srv, max_local=POOL, session_key="sess-b", backend="local"
        )

        assert second[0] is False
        assert second[1] == "sess-a"  # owner reported for diagnostics
        assert "sess-b" not in srv.local_prefill_in_flight

    @pytest.mark.asyncio
    async def test_prefill_is_admitted_after_the_gate_opens(self, srv):
        from proxy.router_helpers import _increment_generating_only_slot

        await _try_acquire_local_dispatch(
            srv, max_local=POOL, session_key="sess-a", backend="local"
        )
        # First byte arrived: the prefill hold is released and A starts decoding.
        await _increment_generating_only_slot(srv, "sess-a")
        assert srv.local_prefill_in_flight == {}

        second = await _try_acquire_local_dispatch(
            srv, max_local=POOL, session_key="sess-b", backend="local"
        )

        assert second[0] is True

    @pytest.mark.asyncio
    async def test_decoding_slot_does_not_block_a_prefill(self, srv):
        from proxy.router_helpers import _increment_generating_only_slot

        await _try_acquire_local_dispatch(
            srv, max_local=POOL, session_key="sess-a", backend="local"
        )
        await _increment_generating_only_slot(srv, "sess-a")  # A is decoding
        assert srv.local_generating_queries == 1
        assert srv.local_prefill_in_flight == {}

        second = await _try_acquire_local_dispatch(
            srv, max_local=POOL, session_key="sess-b", backend="local"
        )

        assert second[0] is True

    @pytest.mark.asyncio
    async def test_disabled_gate_restores_pool_sized_cap(self, srv):
        _set_prefills_cap(srv, 0)

        first = await _try_acquire_local_dispatch(
            srv, max_local=POOL, session_key="sess-a", backend="local"
        )
        second = await _try_acquire_local_dispatch(
            srv, max_local=POOL, session_key="sess-b", backend="local"
        )

        assert first[0] is True
        assert second[0] is True
        assert len(srv.local_prefill_in_flight) == POOL

    @pytest.mark.asyncio
    async def test_lone_request_is_not_delayed(self, srv):
        """AC5: nothing else prefilling → the gate is open immediately."""
        started = time.monotonic()

        acquired = await _try_acquire_local_dispatch(
            srv, max_local=POOL, session_key="solo", backend="local"
        )

        assert acquired[0] is True
        assert time.monotonic() - started < 0.5


class TestSaturationLabel:
    """AC2 — the queue/fallback path labels the gate that blocked it."""

    def test_saturation_reports_prefill_at_cap(self, srv, monkeypatch):
        from proxy.provider import _local_slot_saturation

        srv.local_prefill_in_flight = {"sess-a": True}

        assert _local_slot_saturation(_config(), POOL) == "prefill"

    def test_saturation_open_below_cap(self, srv):
        from proxy.provider import _local_slot_saturation

        assert _local_slot_saturation(_config(), POOL) is None

    def test_saturation_open_when_disabled(self, srv):
        """With the gate disabled the cap is the pool, so 1 in flight passes."""
        from proxy.provider import _local_slot_saturation

        srv.local_prefill_in_flight = {"sess-a": True}

        assert (
            _local_slot_saturation(
                _config(local_max_concurrent_prefills=0), POOL
            )
            is None
        )

    def test_saturation_reports_generating_first(self, srv):
        from proxy.provider import _local_slot_saturation

        srv.local_generating_queries = POOL

        assert _local_slot_saturation(_config(), POOL) == "generating"


class TestQueueStampsPrefillSaturation:
    """AC4 — the wait accounting is available to the dispatch path."""

    @pytest.mark.asyncio
    async def test_prefill_saturation_is_stamped_on_the_request(
        self, srv, monkeypatch
    ):
        from proxy import provider

        srv.local_prefill_in_flight = {"sess-a": True}

        async def _no_dispatch(**_kwargs):
            return None  # caps exceeded → caller falls back (today's behaviour)

        monkeypatch.setattr(
            contention_queue, "wait_for_local_slot", _no_dispatch, raising=False
        )

        request = SimpleNamespace()
        action, reason, _elapsed = await provider._maybe_queue_for_local_slot(
            _config(contention_queue_policy="queue"),
            0,
            POOL,
            request,
            {"messages": [{"role": "user", "content": "hi"}]},
            {"name": "local", "providers": []},
            {"name": "local", "type": "local", "llama_model": "Qwen3"},
            "sess-b",
            priority="low",
        )

        assert action == "fallback"
        assert request._local_queue_saturation == "prefill"


class TestStatusPayload:
    """AC4 — gate state is published on /llama/local/status."""

    def test_status_exposes_prefill_gate_state(self, srv, monkeypatch):
        import proxy.server as server

        async def _fake_query():
            return {"llama_server_running": False, "n_ctx": None}

        monkeypatch.setattr(server, "query_llama_status", _fake_query, raising=False)
        from proxy.handlers import get_llama_local_status
        from starlette.requests import Request

        srv.local_prefill_in_flight = {"sess-a": True}
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/llama/local/status",
                "headers": [],
                "query_string": b"",
                "client": ("127.0.0.1", 4321),
                "server": ("testserver", 80),
                "scheme": "http",
                "root_path": "",
            }
        )

        payload = asyncio.run(get_llama_local_status(request))

        assert payload["local_prefills_in_flight"] == 1
        assert payload["local_max_concurrent_prefills"] == 1
        assert payload["prefill_gate_waiters"] == 0
