"""Prefix-reuse and slot-steal metrics for slot thrash (LP-0MVWPK2VH008YWBD).

E7 of ``docs/dev/single-slot-impact.md`` predicts that at one slot every session
maps to slot 0, so each session switch evicts the previous prefix (the recorded
95 % steal rate the report demands be re-measured). Before this change the
*cause* — prefix reuse vs steal — had no metric at all; only the *cost* was
visible, as a rising ``prefill_slot_seconds`` per served turn, which a
decode-side or prompt-size change could mimic.

What is pinned here, against the real counters and the real production hooks:

- AC1: ``llama_prefix_cache_reuse_total{outcome}`` moves on a reused prefix and
  on a cold one, and counts nothing for an unmeasured stream (a missing prompt
  size must not be read as thrash); ``llama_slot_steal_total{reason}`` moves on
  a steal and reuses the eviction reason the save/restore path already receives.
- AC2/AC3: both counters share the slot-metrics lifecycle — the reset helper
  that stands in for a restart clears their tracked label sets and removes the
  published series, so no ghost series survives.
- Wiring: the served-local-stream accounting path (``router._update_session_and_slot``)
  feeds the reuse counter, and the single eviction chokepoint
  (``router_helpers._record_local_eviction``) feeds the steal counter.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import proxy.metrics as metrics
import proxy.router as router
import proxy.router_helpers as router_helpers

MODEL = "Qwen3"
SESSION = "sess-reuse-steal-1"


# ---------------------------------------------------------------------------
# Counter readers (adapt to the real client and the repo-root test stub)
# ---------------------------------------------------------------------------


def _require_client() -> None:
    if not metrics._enabled:
        pytest.skip("prometheus_client not installed: counters are no-ops")


def _counter_value(counter, **labels) -> int:
    child = counter.labels(**labels)
    if isinstance(child, (int, float)):
        return int(child)
    value = getattr(child, "_value", None)
    if value is not None and hasattr(value, "get"):
        return int(value.get())
    store = getattr(child, "_store", None)
    assert store is not None, "unreadable counter implementation"
    return int(store[()])


def _counter_series(counter) -> set[tuple]:
    children = getattr(counter, "_metrics", None)
    if children is None:
        children = getattr(counter, "_store", None)
    assert children is not None, "unreadable counter implementation"
    return set(children.keys())


@pytest.fixture(autouse=True)
def _clean_slot_metric_lifecycle():
    """Start and end every case without ghost labels from an earlier case."""
    metrics.reset_slot_occupancy_gauges()
    yield
    metrics.reset_slot_occupancy_gauges()


# ---------------------------------------------------------------------------
# AC1: classification + recorder semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cached,prompt,expected",
    [
        (1200, 30000, metrics.PREFIX_CACHE_REUSE_REUSED),
        (1, 1, metrics.PREFIX_CACHE_REUSE_REUSED),
        (0, 30000, metrics.PREFIX_CACHE_REUSE_COLD),
        (None, 30000, metrics.PREFIX_CACHE_REUSE_COLD),
        (0, 0, None),  # unmeasured: no prompt size, must not be counted
        (500, 0, None),
        (0, None, None),
    ],
)
def test_classify_prefix_cache_reuse(cached, prompt, expected):
    assert metrics.classify_prefix_cache_reuse(cached, prompt) == expected


def test_reuse_counter_moves_on_reused_and_cold_streams():
    _require_client()
    assert metrics.record_prefix_cache_reuse(1200, 30000) == "reused"
    assert metrics.record_prefix_cache_reuse(0, 30000) == "cold"
    assert metrics.record_prefix_cache_reuse(2000, 40000) == "reused"

    assert _counter_value(
        metrics.llama_prefix_cache_reuse_total, outcome="reused"
    ) == 2
    assert _counter_value(
        metrics.llama_prefix_cache_reuse_total, outcome="cold"
    ) == 1


def test_reuse_counter_ignores_unmeasured_streams():
    _require_client()
    assert metrics.record_prefix_cache_reuse(0, 0) is None
    assert metrics.record_prefix_cache_reuse(500, 0) is None
    # Nothing published: an unmeasured stream is not evidence of a cold prefix.
    assert _counter_series(metrics.llama_prefix_cache_reuse_total) == set()


def test_steal_counter_moves_by_reason_and_keeps_unknown_visible():
    _require_client()
    metrics.record_slot_steal("lease_evicted")
    metrics.record_slot_steal("lease_evicted")
    metrics.record_slot_steal("resident_prefix_evicted_by_tenant")
    metrics.record_slot_steal("")

    assert _counter_value(
        metrics.llama_slot_steal_total, reason="lease_evicted"
    ) == 2
    assert _counter_value(
        metrics.llama_slot_steal_total, reason="resident_prefix_evicted_by_tenant"
    ) == 1
    # An empty reason is recorded as ``unknown`` rather than silently dropped.
    assert _counter_value(metrics.llama_slot_steal_total, reason="unknown") == 1


# ---------------------------------------------------------------------------
# AC2/AC3: shared slot-metrics lifecycle — label set cleaned up on restart
# ---------------------------------------------------------------------------


def test_reset_clears_reuse_and_steal_label_sets():
    """A restart (reset) must leave no reuse/steal ghost series behind (AC2/AC3)."""
    _require_client()
    metrics.record_prefix_cache_reuse(1200, 30000)
    metrics.record_prefix_cache_reuse(0, 30000)
    metrics.record_slot_steal("lease_evicted")
    metrics.record_slot_steal("slot_assignment_freed")

    assert metrics._prefix_reuse_labelvalues == {"reused", "cold"}
    assert metrics._slot_steal_labelvalues == {"lease_evicted", "slot_assignment_freed"}
    assert _counter_series(metrics.llama_prefix_cache_reuse_total)
    assert _counter_series(metrics.llama_slot_steal_total)

    metrics.reset_slot_occupancy_gauges()  # the restart-lifecycle reset

    # Tracked label sets are emptied and the published series are gone.
    assert metrics._prefix_reuse_labelvalues == set()
    assert metrics._slot_steal_labelvalues == set()
    assert _counter_series(metrics.llama_prefix_cache_reuse_total) == set()
    assert _counter_series(metrics.llama_slot_steal_total) == set()


# ---------------------------------------------------------------------------
# Wiring: the production hooks actually feed the counters
# ---------------------------------------------------------------------------


def _usage_chunk(prompt: int, completion: int, cached: int) -> bytes:
    payload = {
        "choices": [],
        "created": 0,
        "id": "chatcmpl-usage",
        "model": MODEL,
        "object": "chat.completion.chunk",
        "usage": {
            "completion_tokens": completion,
            "prompt_tokens": prompt,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


def _finish_chunk() -> bytes:
    payload = {
        "choices": [{"finish_reason": "stop", "index": 0, "delta": {}}],
        "created": 0,
        "id": "chatcmpl-finish",
        "model": MODEL,
        "object": "chat.completion.chunk",
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


_CONTENT_CHUNK = (
    b'data: {"choices":[{"delta":{"content":"hi"},"index":0}],'
    b'"created":0,"id":"chatcmpl-1","model":"Qwen3",'
    b'"object":"chat.completion.chunk"}\n\n'
)


class _StreamResponse:
    def __init__(self, chunks):
        self._chunks = chunks
        self.headers = {}

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


def _mock_server():
    srv = MagicMock()
    srv.config = {"server": {"max_concurrent_queries": 4}}
    srv.current_model = MODEL
    srv.logger = MagicMock()
    srv.session_manager = AsyncMock()
    srv.local_dispatch_records = {}
    return srv


async def _serve_local_stream(cached_tokens: int) -> None:
    sse = b"".join(
        [_CONTENT_CHUNK, _finish_chunk(), _usage_chunk(30000, 200, cached_tokens)]
    )
    with patch(
        "proxy.router._detect_restore_signal_from_llama_log", return_value=False
    ), patch("proxy.router._has_explicit_restore_signal", return_value=False):
        await router._update_session_and_slot(
            _mock_server(),
            SESSION,
            {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "summarise the module layout"},
                ],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            False,
            None,
            1,
            _StreamResponse([sse]),
            8080,
            None,
            None,
            3.0,
            None,
            False,
            upstream_status=200,
            slot_save_allowed=False,
            collected_content=[sse.decode("utf-8")],
        )


@pytest.mark.asyncio
async def test_router_counts_a_reused_prefix():
    _require_client()
    await _serve_local_stream(cached_tokens=24000)
    assert _counter_value(
        metrics.llama_prefix_cache_reuse_total, outcome="reused"
    ) == 1
    assert _counter_value(
        metrics.llama_prefix_cache_reuse_total, outcome="cold"
    ) == 0


@pytest.mark.asyncio
async def test_router_counts_a_cold_stream():
    _require_client()
    await _serve_local_stream(cached_tokens=0)
    assert _counter_value(
        metrics.llama_prefix_cache_reuse_total, outcome="cold"
    ) == 1
    assert _counter_value(
        metrics.llama_prefix_cache_reuse_total, outcome="reused"
    ) == 0


def test_eviction_chokepoint_counts_a_steal():
    """``_record_local_eviction`` — the save/restore chokepoint — feeds the steal."""
    _require_client()
    srv = MagicMock()
    srv.config = {"server": {}}
    srv.logger = MagicMock()

    planned = router_helpers._record_local_eviction(
        srv,
        SESSION,
        endpoint=None,
        reason="resident_prefix_evicted_by_tenant",
    )

    assert planned is False  # no slot assignment, so no snapshot was planned
    assert _counter_value(
        metrics.llama_slot_steal_total,
        reason="resident_prefix_evicted_by_tenant",
    ) == 1
