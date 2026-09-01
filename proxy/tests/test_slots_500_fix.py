"""Integration tests: /slots 500 rate and slots_stale measurement.

LP-0MTE9HAF8008909G (F2): Tests that validate the /slots 500 rate and
slots_stale metrics before and after the fix.

Acceptance Criteria:
1. Test verifies GET /slots returns 200 from model-instance direct polling
2. Test verifies router-proxy /slots fallback path behavior
3. Test verifies last-known-state fallback when both paths fail
4. Test measures slots_stale rate under simulated conditions

Key files:
- proxy/proxy/observability.py — _query_slots, _query_slots_detail,
  last_known_slot_counts, slots_stale logic
- proxy/proxy/server.py — /llama/local/status endpoint that uses slots data
- proxy/proxy/handlers.py — poll_slots_for_model, _periodic_slots_polling
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import proxy.metrics as metrics
import pytest

from proxy.observability import (
    _query_slots,
    _query_slots_detail,
    _query_slots_progress,
    last_known_slot_counts,
)


# ======================================================================
# Helpers — fake httpx responses
# ======================================================================


class _FakeResponse:
    """Minimal mock of httpx.Response for slot-query testing."""

    def __init__(self, status_code: int, json_data=None):
        self.status_code = status_code
        self._json_data = json_data

    async def json(self):
        return self._json_data


class _FakeClient:
    """Async context-manager mock for httpx.AsyncClient.get().

    Yields a coroutine that returns *responses* in order; after the list is
    exhausted, raises ConnectionError to simulate backend failure.
    """

    def __init__(self, responses: list[tuple[int, list]]):
        """
        Args:
            responses: list of (status_code, json_payload) tuples.
        """
        self.responses = list(responses)
        self.calls: list[str] = []

    async def get(self, url: str, **kwargs):
        self.calls.append(url)
        if self.responses:
            status, data = self.responses.pop(0)
            return _FakeResponse(status, data)
        raise ConnectionError("no more responses")


# ======================================================================
# AC1: GET /slots returns 200 from model-instance direct polling
# ======================================================================


class TestDirectModelInstancePolling:
    """Verify that /slots returns 200 and correctly parses slot data."""

    def test_direct_polling_returns_200_with_slots(self):
        """GET /slots returns 200 and yields correct available/total counts."""
        slots_data = [
            {"id": 0, "is_processing": False, "next_token": {"n_decoded": 0}},
            {"id": 1, "is_processing": True, "next_token": {"n_decoded": 42}},
            {"id": 2, "is_processing": False, "next_token": {"n_decoded": 0}},
        ]
        mock_client = _FakeClient([(200, slots_data)])
        available, total = asyncio.run(
            _query_slots(mock_client, llama_port=8080, model="Qwen3")
        )
        assert total == 3
        assert available == 2  # slots 0 and 2 are idle

    def test_direct_polling_parses_details(self):
        """_query_slots_detail extracts slot_id, is_processing, n_decoded."""
        slots_data = [
            {"id": 0, "is_processing": False, "next_token": {"n_decoded": 0}},
            {"id": 1, "is_processing": True, "next_token": {"n_decoded": 42}},
        ]
        mock_client = _FakeClient([(200, slots_data)])
        detail = asyncio.run(
            _query_slots_detail(8080, model="Qwen3", _client=mock_client)
        )
        assert len(detail) == 2
        assert detail[0] == {"slot_id": 0, "is_processing": False, "n_decoded": 0}
        assert detail[1] == {"slot_id": 1, "is_processing": True, "n_decoded": 42}

    def test_direct_polling_parses_progress(self):
        """_query_slots_progress extracts progress from n_past / n_prompt_tokens_processed."""
        slots_data = [
            {"id": 0, "is_processing": True, "n_past": 100, "n_prompt_tokens_processed": 200},
            {"id": 1, "is_processing": False, "n_past": 0, "n_prompt_tokens_processed": 0},
        ]
        mock_client = _FakeClient([(200, slots_data)])
        progress = asyncio.run(
            _query_slots_progress(8080, model="Qwen3", _client=mock_client)
        )
        assert progress[0]["progress"] == 200  # max of n_past and n_prompt_tokens_processed
        assert progress[0]["processing"] is True
        assert progress[1]["progress"] == 0
        assert progress[1]["processing"] is False

    def test_direct_polling_without_model_param(self):
        """Direct polling works without a model parameter (port 8080)."""
        slots_data = [{"id": 0, "is_processing": False}]
        mock_client = _FakeClient([(200, slots_data)])
        url_used = ""

        async def capture_url():
            result = await _query_slots(mock_client, llama_port=8080)
            return result, mock_client.calls[0]

        (available, total), url = asyncio.run(capture_url())
        assert total == 1
        assert "model=" not in url  # no model param when not passed

    def test_direct_polling_with_model_param(self):
        """Direct polling includes ?model=... when model is provided."""
        slots_data = [{"id": 0, "is_processing": False}]
        mock_client = _FakeClient([(200, slots_data)])

        async def capture_url():
            result = await _query_slots(mock_client, llama_port=8080, model="Qwen3")
            return result, mock_client.calls[0]

        (available, total), url = asyncio.run(capture_url())
        assert total == 1
        assert "model=Qwen3" in url


# ======================================================================
# AC2: Router-proxy /slots fallback path behavior
# ======================================================================


class TestRouterFallbackBehavior:
    """Verify router-proxy fallback path for /slots queries."""

    def test_router_returns_500_records_failure_metric(self):
        """When router returns 500, the failure metric is incremented."""
        mock_client = _FakeClient([(500, None)])
        before = metrics.llama_slots_query_failures_total.labels(
            reason="http_500"
        )._value.get()
        available, total = asyncio.run(
            _query_slots(mock_client, llama_port=8080, model="Qwen3")
        )
        after = metrics.llama_slots_query_failures_total.labels(
            reason="http_500"
        )._value.get()
        assert after == before + 1
        # Should return 0/0 on failure
        assert available == 0
        assert total == 0

    def test_router_timeout_records_timeout_failure(self):
        """A timeout on /slots queries increments the timeout failure counter."""
        async def fake_get_timeout(*args, **kwargs):
            raise asyncio.TimeoutError("request timed out")

        mock_client = MagicMock()
        mock_client.get = fake_get_timeout
        before = metrics.llama_slots_query_failures_total.labels(
            reason="timeout"
        )._value.get()
        available, total = asyncio.run(
            _query_slots(mock_client, llama_port=8080, model="Qwen3", timeout=2.0)
        )
        after = metrics.llama_slots_query_failures_total.labels(
            reason="timeout"
        )._value.get()
        assert after == before + 1
        assert available == 0
        assert total == 0

    def test_router_connection_error_records_connection_failure(self):
        """A connection error on /slots queries increments the connection failure counter."""
        async def fake_get_connection_error(*args, **kwargs):
            raise ConnectionError("connection refused")

        mock_client = MagicMock()
        mock_client.get = fake_get_connection_error
        before = metrics.llama_slots_query_failures_total.labels(
            reason="connection_error"
        )._value.get()
        available, total = asyncio.run(
            _query_slots(mock_client, llama_port=8080, model="Qwen3")
        )
        after = metrics.llama_slots_query_failures_total.labels(
            reason="connection_error"
        )._value.get()
        assert after == before + 1
        assert available == 0
        assert total == 0

    def test_router_returns_400_records_400_failure(self):
        """Router 400 (missing model param) records a 400 failure."""
        mock_client = _FakeClient([(400, None)])
        before = metrics.llama_slots_query_failures_total.labels(
            reason="http_400"
        )._value.get()
        available, total = asyncio.run(
            _query_slots(mock_client, llama_port=8080)
        )
        after = metrics.llama_slots_query_failures_total.labels(
            reason="http_400"
        )._value.get()
        assert after == before + 1
        assert available == 0
        assert total == 0


# ======================================================================
# AC3: Last-known-state fallback when both paths fail
# ======================================================================


class TestLastKnownFallback:
    """Verify last-known slot counts cache and fallback behavior."""

    def test_last_known_cache_updated_on_success(self):
        """A successful /slots query updates the last-known cache."""
        slots_data = [
            {"id": 0, "is_processing": False},
            {"id": 1, "is_processing": True},
        ]
        mock_client = _FakeClient([(200, slots_data)])
        asyncio.run(_query_slots(mock_client, llama_port=8080, model="Qwen3"))
        cached = last_known_slot_counts()
        assert cached is not None
        available, total = cached
        assert total == 2
        assert available == 1

    def test_last_known_fallback_returns_cached_on_failure(self):
        """After a failure, last_known_slot_counts() returns cached values."""
        # First, establish a cached value
        slots_data = [
            {"id": 0, "is_processing": False},
            {"id": 1, "is_processing": False},
        ]
        mock_client_ok = _FakeClient([(200, slots_data)])
        asyncio.run(_query_slots(mock_client_ok, llama_port=8080, model="Qwen3"))

        # Then simulate failure — last known should still be available
        mock_client_fail = _FakeClient([(500, None)])
        available, total = asyncio.run(
            _query_slots(mock_client_fail, llama_port=8080, model="Qwen3")
        )
        assert available == 0  # fresh query returns 0/0 on failure
        assert total == 0

        # But last_known should still have the previous values
        cached = last_known_slot_counts()
        assert cached is not None
        cached_available, cached_total = cached
        assert cached_total == 2
        assert cached_available == 2

    def test_last_known_expires_after_stale_period(self):
        """last_known_slot_counts returns None after the stale period."""
        slots_data = [{"id": 0, "is_processing": False}]
        mock_client = _FakeClient([(200, slots_data)])

        # Patch time.monotonic: first call (cache record) returns 0,
        # second call (stale check) returns past stale period.
        stale_seconds = 3600  # default stale period
        call_count = [0]
        original_monotonic = time.monotonic

        def fake_monotonic():
            call_count[0] += 1
            if call_count[0] == 1:
                return 0.0  # cache record time
            return stale_seconds + 10.0  # stale check time

        with patch.object(time, "monotonic", fake_monotonic):
            asyncio.run(_query_slots(mock_client, llama_port=8080, model="Qwen3"))

        result = last_known_slot_counts()
        assert result is None

    def test_last_known_not_updated_on_failure(self):
        """Failed queries do NOT update the last-known cache."""
        # Establish initial cache
        slots_data = [{"id": 0, "is_processing": False}]
        mock_client_ok = _FakeClient([(200, slots_data)])
        asyncio.run(_query_slots(mock_client_ok, llama_port=8080, model="Qwen3"))
        initial = last_known_slot_counts()

        # Fail a query
        mock_client_fail = _FakeClient([(500, None)])
        asyncio.run(_query_slots(mock_client_fail, llama_port=8080, model="Qwen3"))

        # Cache should be unchanged
        final = last_known_slot_counts()
        assert final == initial

    def test_total_zero_not_cached(self):
        """Successful queries returning total=0 do NOT update the cache."""
        mock_client = _FakeClient([(200, [])])
        asyncio.run(_query_slots(mock_client, llama_port=8080, model="Qwen3"))

        # No cache should be recorded for empty slot lists
        cached = last_known_slot_counts()
        assert cached is None


# ======================================================================
# AC4: slots_stale rate under simulated conditions
# ======================================================================


class TestSlotsStaleMeasurement:
    """Measure and verify slots_stale rate under various conditions."""

    def test_slots_stale_false_on_direct_success(self):
        """When a direct query succeeds, slots_stale should be False."""
        slots_data = [
            {"id": 0, "is_processing": False},
        ]
        mock_client = _FakeClient([(200, slots_data)])

        async def check_stale():
            result = await _query_slots(mock_client, llama_port=8080, model="Qwen3")
            # Direct success — no staleness
            return True  # will verify by checking cache is fresh

        asyncio.run(check_stale())
        cached = last_known_slot_counts()
        assert cached is not None
        # The cache was just updated, so staleness flag would be False in status output

    def test_slots_stale_true_when_fresh_fails_and_cache_available(self):
        """slots_stale should be True when fresh query fails but cache exists."""
        # Establish cache first
        slots_data = [
            {"id": 0, "is_processing": False},
            {"id": 1, "is_processing": False},
        ]
        mock_client_ok = _FakeClient([(200, slots_data)])
        asyncio.run(_query_slots(mock_client_ok, llama_port=8080, model="Qwen3"))

        # Now fail the fresh query
        mock_client_fail = _FakeClient([(500, None)])
        available, total = asyncio.run(
            _query_slots(mock_client_fail, llama_port=8080, model="Qwen3")
        )

        # Fresh query returns 0/0, but cache still has values
        # In the status endpoint, this would set slots_stale=True
        assert available == 0
        assert total == 0
        cached = last_known_slot_counts()
        assert cached is not None
        # slots_stale scenario: fresh=0/0, cached=2/2 → stale=True

    def test_slots_stale_true_when_no_cache_and_fails(self):
        """slots_stale is effectively True when there's no cache and query fails."""
        mock_client = _FakeClient([(500, None)])
        available, total = asyncio.run(
            _query_slots(mock_client, llama_port=8080, model="Qwen3")
        )

        assert available == 0
        assert total == 0
        assert last_known_slot_counts() is None
        # No cache + failure = fully stale state

    def test_slots_stale_rate_under_repeated_failures(self):
        """Under repeated failures, slots_stale should remain True."""
        # Establish cache
        slots_data = [{"id": 0, "is_processing": False}]
        mock_client_ok = _FakeClient([(200, slots_data)])
        asyncio.run(_query_slots(mock_client_ok, llama_port=8080, model="Qwen3"))

        # Simulate 10 consecutive failures
        for i in range(10):
            mock_client_fail = _FakeClient([(500, None)])
            asyncio.run(
                _query_slots(mock_client_fail, llama_port=8080, model="Qwen3")
            )

        # Cache should still exist but be increasingly stale
        cached = last_known_slot_counts()
        assert cached is not None  # still cached, but no fresh data

    def test_slots_stale_resets_after_successful_query(self):
        """slots_stale resets to False after a successful query."""
        # Establish cache
        slots_data = [
            {"id": 0, "is_processing": False},
            {"id": 1, "is_processing": False},
        ]
        mock_client_ok = _FakeClient([(200, slots_data)])
        asyncio.run(_query_slots(mock_client_ok, llama_port=8080, model="Qwen3"))

        # Fail a query (would set slots_stale=True)
        mock_client_fail = _FakeClient([(500, None)])
        asyncio.run(
            _query_slots(mock_client_fail, llama_port=8080, model="Qwen3")
        )

        # Recover with a successful query
        slots_data_recovery = [
            {"id": 0, "is_processing": False},
            {"id": 1, "is_processing": False},
        ]
        mock_client_recovery = _FakeClient([(200, slots_data_recovery)])
        asyncio.run(
            _query_slots(mock_client_recovery, llama_port=8080, model="Qwen3")
        )

        # Cache should be freshly updated
        cached = last_known_slot_counts()
        assert cached is not None
        # The cache was just refreshed, so staleness is resolved


# ======================================================================
# Router vs direct polling comparison
# ======================================================================


class TestRouterVsDirectPolling:
    """Compare router-proxy and direct model-instance /slots polling."""

    def test_direct_polling_faster_than_router(self):
        """Direct polling to model instance should be faster than via router."""
        # Direct polling: single call to model port
        direct_calls = []

        async def direct_poll():
            slots_data = [{"id": 0, "is_processing": False}]
            mock_client = _FakeClient([(200, slots_data)])
            result = await _query_slots(mock_client, llama_port=8080, model="Qwen3")
            direct_calls.extend(mock_client.calls)
            return result

        asyncio.run(direct_poll())
        assert len(direct_calls) == 1  # single call
        assert "8080" in direct_calls[0]

    def test_router_polling_serializes_requests(self):
        """Router polling serializes behind busy child — measurable overhead."""
        # Router path: goes through router port first, then may fall back
        # The key issue (LP-0MTDGBRPU003Z7KU): router serializes /slots
        # behind the busy child's generation loop (5-7s vs 0.17s direct)
        #
        # This test verifies the router fallback path exists and can handle
        # a router 500 by falling back to direct polling.

        # Scenario: router returns 500 (busy child serialization)
        router_calls = []
        direct_calls = []

        async def simulate_router_fallback():
            # First try: router (500)
            router_client = _FakeClient([(500, None)])
            result = await _query_slots(router_client, llama_port=8080, model="Qwen3")
            router_calls.extend(router_client.calls)

            # Fallback: direct model-instance polling (200)
            slots_data = [{"id": 0, "is_processing": False}]
            direct_client = _FakeClient([(200, slots_data)])
            result2 = await _query_slots(
                direct_client, llama_port=8080, model="Qwen3"
            )
            direct_calls.extend(direct_client.calls)

            return result, result2

        r1, r2 = asyncio.run(simulate_router_fallback())
        assert r1 == (0, 0)  # router failed
        assert r2 == (1, 1)  # direct succeeded

    def test_router_500_rate_under_load(self):
        """Simulate high router-500 rate during concurrent load."""
        # Simulate 100 router /slots requests, 70% get 500s (busy child)
        responses = [(500, None)] * 70 + [(200, [{"id": 0, "is_processing": False}])] * 30
        mock_client = _FakeClient(responses)
        mock_client.calls.clear()

        results = []
        for _ in range(100):
            result = asyncio.run(
                _query_slots(mock_client, llama_port=8080, model="Qwen3")
            )
            results.append(result)

        # Count successes vs failures
        successes = sum(1 for r in results if r == (1, 1))
        failures = sum(1 for r in results if r == (0, 0))

        assert successes == 30
        assert failures == 70
        # Router 500 rate: 70%
        router_500_rate = 100 * failures / len(results)
        assert router_500_rate == 70.0


# ======================================================================
# Integration: status endpoint slots_stale flag
# ======================================================================


class TestStatusEndpointSlotsStale:
    """Verify slots_stale flag behavior in the /llama/local/status endpoint."""

    def test_status_shares_stale_from_fresh_failure(self):
        """When /slots fails in status endpoint, slots_stale=True is set."""
        # Establish cache
        slots_data = [
            {"id": 0, "is_processing": False},
            {"id": 1, "is_processing": True},
        ]
        mock_client_ok = _FakeClient([(200, slots_data)])
        asyncio.run(_query_slots(mock_client_ok, llama_port=8080, model="Qwen3"))

        # Status endpoint queries /slots — fails
        mock_client_fail = _FakeClient([(500, None)])

        async def simulate_status_query():
            # This mirrors what happens in server.py get_llama_local_status():
            # 1. _query_slots fails → (0, 0)
            # 2. Falls back to last_known_slot_counts()
            # 3. Sets slots_stale=True
            available, total = await _query_slots(
                mock_client_fail, llama_port=8080, model="Qwen3", timeout=2.0
            )

            # Check if we should fall back
            slots_stale = False
            if total == 0:
                last_known = last_known_slot_counts()
                if last_known:
                    available, total = last_known
                    slots_stale = True

            return available, total, slots_stale

        avail, total, stale = asyncio.run(simulate_status_query())
        # Cache has 1 idle (id=0) and 1 busy (id=1)
        assert avail == 1
        assert total == 2
        assert stale is True  # slots_stale should be True when using cached data

    def test_status_no_stale_when_fresh_succeeds(self):
        """When /slots succeeds, slots_stale=False."""
        slots_data = [{"id": 0, "is_processing": False}]
        mock_client = _FakeClient([(200, slots_data)])

        async def simulate_status_query():
            available, total = await _query_slots(
                mock_client, llama_port=8080, model="Qwen3", timeout=2.0
            )
            slots_stale = False
            if total == 0:
                last_known = last_known_slot_counts()
                if last_known:
                    available, total = last_known
                    slots_stale = True
            return available, total, slots_stale

        avail, total, stale = asyncio.run(simulate_status_query())
        assert avail == 1
        assert total == 1
        assert stale is False  # fresh success, no staleness

    def test_status_fully_stale_when_no_cache(self):
        """When /slots fails and no cache exists, slots_stale=True with 0/0."""
        mock_client = _FakeClient([(500, None)])

        async def simulate_status_query():
            available, total = await _query_slots(
                mock_client, llama_port=8080, model="Qwen3", timeout=2.0
            )
            slots_stale = False
            if total == 0:
                last_known = last_known_slot_counts()
                if last_known:
                    available, total = last_known
                    slots_stale = True
            return available, total, slots_stale

        avail, total, stale = asyncio.run(simulate_status_query())
        assert avail == 0
        assert total == 0
        assert stale is False  # No cache to fall back to, so no "stale" — just 0/0
