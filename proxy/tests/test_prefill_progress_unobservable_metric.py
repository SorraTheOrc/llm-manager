"""Fail-safe fallback and the unobservable-progress metric
(LP-0MW8VPH2Q007XPHQ, parent AC6/AC11).

When no numeric prefill progress can be observed the behaviour must stay the
same as today (liveness-only extension on the adaptive acquisition estimate —
never an unbounded extension), the throttled WARNING must still fire, and the
event must now be *counted* via ``proxy_prefill_progress_unobservable_total``
so a persistently blind progress source is visible in monitoring, not only in
the log.
"""

from unittest.mock import AsyncMock, patch

import proxy.metrics as metrics
import pytest
from test_prefill_lease_extension import _make_srv


def _counter_value() -> float:
    counter = metrics.prefill_progress_unobservable_total
    assert counter is not None, (
        "prometheus_client must be available for the unobservable counter "
        "(the suite requires metrics._enabled)"
    )
    return float(counter.labels(reason="no_source")._value.get())


@pytest.fixture(autouse=True)
def _reset_warn_throttle(monkeypatch):
    """Reset the module-level throttle so each test sees its own warning."""
    import proxy.router_helpers as rh

    monkeypatch.setattr(rh, "_last_prefill_progress_warn_ts", 0.0)


class TestUnobservableMetric:
    def test_counter_helper_increments(self):
        """record_prefill_progress_unobservable increments by one."""
        before = _counter_value()
        metrics.record_prefill_progress_unobservable()
        assert _counter_value() == before + 1

    @pytest.mark.asyncio
    async def test_unobservable_poll_increments_metric(self):
        """AC2: an unobservable lease poll increments the counter."""
        from proxy.router_helpers import _query_prefill_progress

        srv = _make_srv()
        before = _counter_value()
        with patch(
            "proxy.observability._query_slots_progress", AsyncMock(return_value={})
        ), patch(
            "proxy.observability.query_llama_status",
            AsyncMock(return_value={"kv_cache_tokens": None}),
        ), patch(
            "proxy.observability.get_slot_prefill_progress",
            return_value=None,
        ):
            progress, alive = await _query_prefill_progress(
                srv, 8080, model_name="Qwen3", slot_id=3
            )
        assert (progress, alive) == (None, False)
        assert _counter_value() == before + 1

    @pytest.mark.asyncio
    async def test_unobservable_counts_every_poll_but_warns_once(self):
        """AC3: the metric counts every poll; the WARNING is throttled."""
        from proxy.router_helpers import _query_prefill_progress

        srv = _make_srv()
        before = _counter_value()
        with patch(
            "proxy.observability._query_slots_progress", AsyncMock(return_value={})
        ), patch(
            "proxy.observability.query_llama_status",
            AsyncMock(return_value={"kv_cache_tokens": None}),
        ), patch(
            "proxy.observability.get_slot_prefill_progress",
            return_value=None,
        ):
            for _ in range(3):
                await _query_prefill_progress(
                    srv, 8080, model_name="Qwen3", slot_id=3
                )
        assert _counter_value() == before + 3
        # The throttled warning fired only once in the 60 s window.
        warning_calls = [
            c
            for c in srv.logger.warning.call_args_list
            if "prefill_progress_unobservable" in str(c)
        ]
        assert len(warning_calls) == 1

    @pytest.mark.asyncio
    async def test_measured_progress_does_not_increment_metric(self):
        """AC4: when a measured source works, no spurious increment/warning."""
        from proxy.router_helpers import _query_prefill_progress

        srv = _make_srv()
        before = _counter_value()
        with patch(
            "proxy.observability._query_slots_progress",
            AsyncMock(return_value={3: {"progress": 5000, "processing": True}}),
        ):
            progress, alive = await _query_prefill_progress(
                srv, 8080, model_name="Qwen3", slot_id=3
            )
        assert (progress, alive) == (5000, True)
        assert _counter_value() == before
        assert not any(
            "prefill_progress_unobservable" in str(c)
            for c in srv.logger.warning.call_args_list
        )

    @pytest.mark.asyncio
    async def test_log_progress_does_not_increment_metric(self):
        """A log-derived measured progress is not 'unobservable'."""
        from proxy.router_helpers import _query_prefill_progress

        srv = _make_srv()
        before = _counter_value()
        with patch(
            "proxy.observability._query_slots_progress", AsyncMock(return_value={})
        ), patch(
            "proxy.observability.get_slot_prefill_progress",
            return_value={"n_tokens": 12345, "progress": 0.7, "timestamp": 0.0},
        ):
            progress, alive = await _query_prefill_progress(
                srv, 8080, model_name="Qwen3", slot_id=3
            )
        assert (progress, alive) == (12345, True)
        assert _counter_value() == before


class TestFailSafeExtension:
    @pytest.mark.asyncio
    async def test_unobservable_leaves_lease_untouched(self):
        """AC1/AC6: no progress and no liveness → no extension (fail-safe)."""
        from proxy.router_helpers import _extend_lease_during_prefill

        srv = _make_srv(
            records={
                "sess-1": {
                    "backend": "local",
                    "active": True,
                    "expires_at": 1.0,
                    "last_progress": 0,
                    "last_progress_ts": 1.0,
                }
            }
        )
        with patch(
            "proxy.router_helpers._query_prefill_progress",
            AsyncMock(return_value=(None, False)),
        ):
            last_progress, extended = await _extend_lease_during_prefill(
                srv, "sess-1", llama_port=8080, slot_id=0, last_progress=0
            )
        assert extended is False
        assert last_progress == 0
        # The lease expires_at is NOT pushed out — never an unbounded extension.
        assert srv.local_dispatch_records["sess-1"]["expires_at"] == 1.0

    @pytest.mark.asyncio
    async def test_liveness_only_still_extends(self):
        """AC1: liveness-only behaviour is preserved when numeric progress is
        unobservable (the b8782 fail-safe)."""
        from proxy.router_helpers import _extend_lease_during_prefill

        srv = _make_srv(
            records={
                "sess-1": {
                    "backend": "local",
                    "active": True,
                    "expires_at": 1.0,
                    "last_progress": 0,
                    "last_progress_ts": 1.0,
                }
            }
        )
        with patch(
            "proxy.router_helpers._query_prefill_progress",
            AsyncMock(return_value=(None, True)),
        ):
            last_progress, extended = await _extend_lease_during_prefill(
                srv, "sess-1", llama_port=8080, slot_id=0, last_progress=0
            )
        assert extended is True
        assert last_progress == 0
