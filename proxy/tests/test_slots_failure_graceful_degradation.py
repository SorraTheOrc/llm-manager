"""Tests for graceful degradation of /slots failures in /llama/local/status.

LP-0MSVP7XJ6008QPKX: after the scheduled cheap-mode restart, llama-server's
``/slots`` endpoint returned HTTP 500 during the model reload (11,011× in the
01:00–10:00 proxy log). ``_query_slots()`` swallowed the failure and returned
``(0, 0)``, so the status endpoint reported ``total_slots=0`` and the Herdr
downtime worker's ``isIdleStatus`` fail-closed → zero dispatches overnight.

The fix:
- ``_query_slots()`` records the last successful ``(available, total)`` counts
  and a failure metric on every failed query (``llama_slots_query_failures_total``).
- The status endpoint serves the last-known counts (instead of 0/0) when the
  fresh query fails, bounded by ``SLOT_COUNTS_STALE_AFTER_SECONDS`` (default
  3600s) so a genuinely-unavailable llama-server eventually fail-closes.
- A ``slots_stale`` boolean in the payload/log makes the degraded state
  observable so a future silent failure surfaces (monitoring AC).
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import proxy.metrics as metrics
import pytest

# ======================================================================
# Failure metric
# ======================================================================


class TestSlotsQueryFailureMetric:
    """llama_slots_query_failures_total counter records /slots failures."""

    def test_counter_exists_with_reason_label(self):
        """The counter is registered with a reason label when prometheus is enabled."""
        assert metrics._enabled, (
            "prometheus_client should be available; if not, check test environment"
        )
        assert metrics.llama_slots_query_failures_total is not None
        from prometheus_client import Counter

        assert isinstance(metrics.llama_slots_query_failures_total, Counter)
        assert "reason" in metrics.llama_slots_query_failures_total._labelnames

    def test_record_slots_query_failure_increments_counter(self):
        """Calling record_slots_query_failure() increments the matching label set."""
        before = metrics.llama_slots_query_failures_total.labels(
            reason="http_500"
        )._value.get()
        metrics.record_slots_query_failure("http_500")
        after = metrics.llama_slots_query_failures_total.labels(
            reason="http_500"
        )._value.get()
        assert after == before + 1

    def test_record_slots_query_failure_separate_reasons(self):
        """Different reasons produce separate label combinations (no cross-talk)."""
        before_500 = metrics.llama_slots_query_failures_total.labels(
            reason="http_500"
        )._value.get()
        before_to = metrics.llama_slots_query_failures_total.labels(
            reason="timeout"
        )._value.get()
        metrics.record_slots_query_failure("http_500")
        assert (
            metrics.llama_slots_query_failures_total.labels(reason="http_500")._value.get()
            == before_500 + 1
        )
        assert (
            metrics.llama_slots_query_failures_total.labels(reason="timeout")._value.get()
            == before_to
        )


# ======================================================================
# _query_slots: last-known cache + failure recording
# ======================================================================


class _MockClient:
    """Minimal async client returning a scripted sequence of responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def get(self, url, **kwargs):
        self.calls.append(url)
        if not self.responses:
            raise ConnectionError("connection refused")
        return self.responses.pop(0)


class _Resp:
    def __init__(self, status_code, payload=None, exc=None):
        self.status_code = status_code
        self._payload = payload
        self._exc = exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        if self._exc:
            raise self._exc
        return self._payload

    def __await__(self):
        async def _wrap():
            return self
        return _wrap().__await__()


class TestQuerySlotsRecordsLastKnown:
    """_query_slots() maintains a last-known counts cache for graceful degradation."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        import proxy.observability as obs

        obs._last_slot_counts_cache = None
        yield
        obs._last_slot_counts_cache = None

    @pytest.mark.asyncio
    async def test_success_updates_last_known_cache(self):
        """A successful query with total>0 updates the module-level cache."""
        from proxy.observability import _query_slots

        client = _MockClient([
            _Resp(200, [
                {"is_processing": False},
                {"is_processing": True},
                {"is_processing": False},
            ]),
        ])
        available, total = await _query_slots(client, 8080, timeout=2.0, model="Qwen3")
        assert (available, total) == (2, 3)
        from proxy.observability import _last_slot_counts_cache as cache

        assert cache is not None
        assert cache[:2] == (2, 3)

    @pytest.mark.asyncio
    async def test_empty_slot_list_does_not_poison_cache(self):
        """A 200 with an empty list is treated as failure; cache is not overwritten."""
        import proxy.observability as obs

        obs._last_slot_counts_cache = (2, 3, time.monotonic())
        from proxy.observability import _query_slots

        client = _MockClient([_Resp(200, [])])
        available, total = await _query_slots(client, 8080, timeout=2.0)
        assert (available, total) == (0, 0)
        # The previous last-known value must survive an empty/transient response.
        assert obs._last_slot_counts_cache is not None
        assert obs._last_slot_counts_cache[:2] == (2, 3)

    @pytest.mark.asyncio
    async def test_http_500_records_failure_metric(self):
        """A 500 response increments the failure counter with reason http_500."""
        from proxy.observability import _query_slots

        before = metrics.llama_slots_query_failures_total.labels(
            reason="http_500"
        )._value.get()
        client = _MockClient([_Resp(500, {"error": "model is loading"})])
        available, total = await _query_slots(client, 8080, timeout=2.0, model="Qwen3")
        assert (available, total) == (0, 0)
        after = metrics.llama_slots_query_failures_total.labels(
            reason="http_500"
        )._value.get()
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_connection_error_records_failure_metric(self):
        """Connection failures increment the counter with reason connection_error."""
        from proxy.observability import _query_slots

        before = metrics.llama_slots_query_failures_total.labels(
            reason="connection_error"
        )._value.get()
        client = _MockClient([])  # raises ConnectionError
        available, total = await _query_slots(client, 8080, timeout=2.0)
        assert (available, total) == (0, 0)
        after = metrics.llama_slots_query_failures_total.labels(
            reason="connection_error"
        )._value.get()
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_timeout_records_failure_metric(self):
        """A timeout increments the counter with reason timeout."""
        from proxy.observability import _query_slots

        before = metrics.llama_slots_query_failures_total.labels(
            reason="timeout"
        )._value.get()

        async def slow(*args, **kwargs):
            await asyncio.sleep(10)

        client = MagicMock()
        client.get = slow
        available, total = await asyncio.wait_for(
            _query_slots(client, 8080, timeout=0.05), timeout=1.0
        )
        assert (available, total) == (0, 0)
        after = metrics.llama_slots_query_failures_total.labels(
            reason="timeout"
        )._value.get()
        assert after == before + 1


# ======================================================================
# Status endpoint: graceful degradation (slots_stale)
# ======================================================================


class _FakeLock:
    def locked(self):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass


class TestHandlerGracefulDegradation:
    """GET /llama/local/status serves last-known counts on /slots failure."""

    async def _get(
        self, slots_counts, cache_value=None, stale_after=None, counts_failed=False
    ):
        """Issue /llama/local/status with controllable counts + last-known cache.

        ``counts_failed`` marks the fresh counts read as a transport failure
        (HTTP 500), which is what the graceful-degradation fixtures model. Its
        outcome matters to the age-derived ``slots_stale``: a failed read that
        served nothing is a *missing snapshot* and reports stale
        (LP-0MVVXGM54009TOXD), while a successful empty read is a live answer.

        Returns the parsed JSON body.
        """
        import httpx
        from proxy.server import app

        from proxy import server

        async def fake_query():
            return {"llama_server_running": True}

        def _stamp_provenance(kwargs):
            """Publish the fetch stamp the real helpers always record."""
            prov = kwargs.get("provenance_out")
            if isinstance(prov, dict):
                prov.setdefault("source", "upstream")
                prov.setdefault("fetched_at", time.monotonic())
                prov.setdefault("upstream_calls", 1)

        async def _counts(*_args, **kwargs):
            _stamp_provenance(kwargs)
            if counts_failed:
                out = kwargs.get("status_out")
                if isinstance(out, dict):
                    out.update({"status": 500, "reason": "http_500"})
                # The real helper records the failure reason for the payload.
                from proxy import observability

                observability._record_slots_query_failure("http_500")
            return slots_counts

        async def _detail(*_args, **kwargs):
            _stamp_provenance(kwargs)
            return []

        counts_mock = AsyncMock(side_effect=_counts)
        counts_mock.return_value = slots_counts
        detail_mock = AsyncMock(side_effect=_detail)
        transport = httpx.ASGITransport(app=app)
        with patch("proxy.server.query_llama_status", side_effect=fake_query):
            with patch.object(server, "current_model", "test-model"):
                with patch.object(server, "model_switch_refcount", 0):
                    with patch.object(server, "model_switch_lock", _FakeLock()):
                        with patch.object(server, "background_loads", {}):
                            with patch.object(server, "local_dispatch_records", {}):
                                with patch.object(
                                    server, "local_dispatch_records_lock", _FakeLock()
                                ):
                                    with patch.object(
                                        server,
                                        "config",
                                        {"server": {"llama_server_port": 8080}},
                                    ):
                                        with patch(
                                            "proxy.observability._query_slots_detail",
                                            detail_mock,
                                        ):
                                            with patch(
                                                "proxy.observability._query_slots",
                                                counts_mock,
                                            ):
                                                with patch(
                                                    "proxy.observability._last_slot_counts_cache",
                                                    cache_value,
                                                ):
                                                    if stale_after is not None:
                                                        with patch(
                                                            "proxy.observability._slots_counts_stale_after_seconds",
                                                            return_value=stale_after,
                                                        ):
                                                            async with httpx.AsyncClient(
                                                                transport=transport,
                                                                base_url="http://test",
                                                            ) as ac:
                                                                resp = await ac.get(
                                                                    "/llama/local/status"
                                                                )
                                                    else:
                                                        async with httpx.AsyncClient(
                                                            transport=transport,
                                                            base_url="http://test",
                                                        ) as ac:
                                                            resp = await ac.get(
                                                                "/llama/local/status"
                                                            )
        assert resp.status_code == 200
        return resp.json()

    @pytest.mark.asyncio
    async def test_slots_failure_serves_last_known_counts(self):
        """A last-known counts cache is served on a /slots failure.

        The cached snapshot is seconds old, so under the age-derived
        ``slots_stale`` (LP-0MVVXGM54009TOXD) the poll is *not* stale even
        though the counts phase fell back — ``slot_counts_stale`` discloses
        that. Since LP-0MW1VDZIX0011D8M a cache-served counts pair asserts no
        availability: ``available_slots`` is 0 with ``availability_unknown``
        true, while the cached pool size is still published as ``total_slots``.
        """
        cache = (2, 3, time.monotonic() - 2.0)  # inside the 300 s ceiling
        j = await self._get(
            slots_counts=(0, 0), cache_value=cache, counts_failed=True
        )
        assert j["total_slots"] == 3
        assert j["available_slots"] == 0
        assert j["availability_unknown"] is True
        assert j["slot_counts_stale"] is True
        assert j["slots_stale"] is False

    @pytest.mark.asyncio
    async def test_slots_failure_with_no_last_known_stays_closed(self):
        """A failed /slots read with no cache → 0/0 capacity and stale.

        Nothing was served and nothing dated can be read from the backend, so
        the surface is a *missing snapshot* under the age-derived definition
        (LP-0MVVXGM54009TOXD): ``slots_stale`` is true (no live evidence) while
        the counts stay fail-closed at 0/0.
        """
        j = await self._get(
            slots_counts=(0, 0), cache_value=None, counts_failed=True
        )
        assert j["total_slots"] == 0
        assert j["available_slots"] == 0
        assert j["slots_stale"] is True
        assert j["slots_stale_reason"] == "query_error"

    @pytest.mark.asyncio
    async def test_fresh_counts_not_stale(self):
        """A healthy (2,3) counts response → real counts, slots_stale=false."""
        j = await self._get(slots_counts=(2, 3), cache_value=None)
        assert j["total_slots"] == 3
        assert j["available_slots"] == 2
        assert j["slots_stale"] is False

    @pytest.mark.asyncio
    async def test_stale_cache_expires_after_ttl(self):
        """An expired last-known cache (>SLOT_COUNTS_STALE_AFTER_SECONDS) is not served."""
        expired = (2, 3, time.monotonic() - 7200)  # 2h old, TTL forced to 3600
        j = await self._get(slots_counts=(0, 0), cache_value=expired, stale_after=3600)
        assert j["total_slots"] == 0
        assert j["available_slots"] == 0
        assert j["slots_stale"] is False
