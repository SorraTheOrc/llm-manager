"""Tests for cached per-slot detail fallback in /llama/local/status.

LP-0MUFSVXID0039ZAQ: when the fresh per-slot detail query fails
(e.g. HTTP 500 during a model reload), the status endpoint now falls
back to ``_last_slot_details_cache`` so consumers (herdr) retain per-slot
identity instead of receiving ``slots: []``.

``slots_stale`` is set to true when served slot data came from cache,
mirroring the count-based ``slots_stale`` behaviour (LP-0MSVP7XJ6008QPKX).
"""

import os
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

# ======================================================================
# Cached per-slot detail fallback (LP-0MUFSVXID0039ZAQ)
# ======================================================================


class _FakeLock:
    def locked(self):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass


class TestSlotsDetailCacheFallback:
    """GET /llama/local/status serves cached slot detail when fresh query fails."""

    _FAKE_SLOTS = [
        {"slot_id": 0, "is_processing": False, "n_decoded": None},
        {"slot_id": 1, "is_processing": True, "n_decoded": 42},
        {"slot_id": 2, "is_processing": False, "n_decoded": None},
    ]

    async def _get(
        self,
        slots_counts,
        slots_detail=None,
        slots_detail_side_effect=None,
        cache_value=None,
        current_model="test-model",
        config=None,
    ):
        """Issue /llama/local/status with controllable /slots detail + cache.

        Returns the parsed JSON body.
        """
        from proxy.server import app

        from proxy import server

        async def fake_query():
            return {"llama_server_running": True}

        counts_mock = AsyncMock(return_value=slots_counts)
        detail_mock = AsyncMock(
            return_value=slots_detail, side_effect=slots_detail_side_effect
        )
        transport = httpx.ASGITransport(app=app)
        with patch("proxy.server.query_llama_status", side_effect=fake_query):
            with patch.object(server, "current_model", current_model):
                with patch.object(server, "model_switch_refcount", 0):
                    with patch.object(server, "model_switch_lock", _FakeLock()):
                        with patch.object(server, "background_loads", {}):
                            with patch.object(server, "local_dispatch_records", {}):
                                with patch.object(
                                    server,
                                    "local_dispatch_records_lock",
                                    _FakeLock(),
                                ):
                                    with patch.object(
                                        server,
                                        "config",
                                        config
                                        if config is not None
                                        else {"server": {"llama_server_port": 8080}},
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
                                                    "proxy.observability._last_slot_details_cache",
                                                    cache_value,
                                                ):
                                                    async with httpx.AsyncClient(
                                                        transport=transport,
                                                        base_url="http://test",
                                                    ) as ac:
                                                        resp = await ac.get(
                                                            "/llama/local/status"
                                                        )
        assert resp.status_code == 200
        return resp.json(), detail_mock, counts_mock

    @pytest.mark.asyncio
    async def test_fresh_slots_detail_served_directly(self):
        """When /slots detail succeeds, the fresh data is used (no cache needed)."""
        j, detail_mock, counts_mock = await self._get(
            slots_counts=(2, 3),
            slots_detail=self._FAKE_SLOTS,
        )
        assert j["slots"] == self._FAKE_SLOTS
        assert j["slots_stale"] is False
        detail_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fresh_detail_non_empty_slots_stale_false(self):
        """When the fresh detail query returns data, slots_stale stays false."""
        j, _, counts_mock = await self._get(
            slots_counts=(2, 3),
            slots_detail=self._FAKE_SLOTS,
            cache_value=[
                {"slot_id": 99, "is_processing": True, "n_decoded": 0},
            ],
        )
        assert j["slots"] == self._FAKE_SLOTS
        assert j["slots_stale"] is False

    @pytest.mark.asyncio
    async def test_slots_detail_failure_serves_cached_detail(self):
        """AC1: when /slots detail fails and cache is non-empty, cached detail is served."""
        j, detail_mock, counts_mock = await self._get(
            slots_counts=(0, 0),
            slots_detail=[],
            cache_value=self._FAKE_SLOTS,
        )
        assert j["slots"] == self._FAKE_SLOTS
        assert j["slots_stale"] is True

    @pytest.mark.asyncio
    async def test_slots_detail_failure_serves_cached_with_fresh_counts(self):
        """Cached slot detail is served even when counts are fresh."""
        j, detail_mock, counts_mock = await self._get(
            slots_counts=(2, 3),
            slots_detail=[],
            cache_value=self._FAKE_SLOTS,
        )
        assert j["slots"] == self._FAKE_SLOTS
        assert j["slots_stale"] is True

    @pytest.mark.asyncio
    async def test_slots_detail_failure_empty_cache_returns_empty(self):
        """AC3: empty/missing cache yields slots: [], never malformed."""
        j, detail_mock, counts_mock = await self._get(
            slots_counts=(0, 0),
            slots_detail=[],
            cache_value=None,
        )
        assert j["slots"] == []
        assert j["slots_stale"] is False

    @pytest.mark.asyncio
    async def test_slots_detail_failure_empty_cache_list_returns_empty(self):
        """Cache is an empty list (not None) — still returns slots: []."""
        j, detail_mock, counts_mock = await self._get(
            slots_counts=(0, 0),
            slots_detail=[],
            cache_value=[],
        )
        assert j["slots"] == []
        assert j["slots_stale"] is False

    @pytest.mark.asyncio
    async def test_slots_detail_timeout_serves_cached(self):
        """When the detail query raises TimeoutError, cached detail is served."""
        j, detail_mock, counts_mock = await self._get(
            slots_counts=(0, 0),
            slots_detail_side_effect=TimeoutError("slots detail timed out"),
            cache_value=self._FAKE_SLOTS,
        )
        assert j["slots"] == self._FAKE_SLOTS
        assert j["slots_stale"] is True

    @pytest.mark.asyncio
    async def test_slots_detail_exception_serves_cached(self):
        """When the detail query raises any exception, cached detail is served."""
        j, detail_mock, counts_mock = await self._get(
            slots_counts=(0, 0),
            slots_detail_side_effect=ConnectionError("connection refused"),
            cache_value=self._FAKE_SLOTS,
        )
        assert j["slots"] == self._FAKE_SLOTS
        assert j["slots_stale"] is True

    @pytest.mark.asyncio
    async def test_slots_detail_preserves_cache_content(self):
        """AC4: cached slot data is passed through unchanged — no re-projection."""
        cached_slots = [
            {"slot_id": 0, "is_processing": True, "n_decoded": 100},
        ]
        j, _, _ = await self._get(
            slots_counts=(0, 0),
            slots_detail=[],
            cache_value=cached_slots,
        )
        assert j["slots"] == cached_slots

    @pytest.mark.asyncio
    async def test_slots_stale_true_when_both_counts_and_detail_from_cache(self):
        """When both counts AND detail come from cache, slots_stale is True."""
        j, detail_mock, counts_mock = await self._get(
            slots_counts=(0, 0),
            slots_detail=[],
            cache_value=self._FAKE_SLOTS,
        )
        assert j["slots_stale"] is True
        assert j["slots"] == self._FAKE_SLOTS
        # Counts were from cache too (total_slots=0 from fresh query)
        assert counts_mock.return_value == (0, 0)

    @pytest.mark.asyncio
    async def test_slots_detail_cached_when_counts_fresh(self):
        """When counts are fresh but detail fails, slots_stale is True."""
        j, detail_mock, counts_mock = await self._get(
            slots_counts=(2, 3),
            slots_detail=[],
            cache_value=self._FAKE_SLOTS,
        )
        assert j["slots"] == self._FAKE_SLOTS
        assert j["slots_stale"] is True
        # Counts were fresh — the stale flag is driven by detail cache hit
        assert counts_mock.return_value == (2, 3)


# ======================================================================
# AC4: response budget (STATUS_QUERY_TIMEOUT)
# ======================================================================


class TestSlotsDetailResponseBudget:
    """The cached fallback never adds I/O; the endpoint stays within budget."""

    _CACHED_SLOTS = [
        {"slot_id": 0, "is_processing": True, "n_decoded": 7},
    ]

    async def _get(self, detail_side_effect, timeout_env, cache_value):
        from proxy.server import app

        from proxy import server

        async def fake_query():
            return {"llama_server_running": True}

        detail_mock = AsyncMock(side_effect=detail_side_effect)
        counts_mock = AsyncMock(return_value=(0, 0))
        transport = httpx.ASGITransport(app=app)
        with patch.dict(os.environ, {"STATUS_QUERY_TIMEOUT": timeout_env}):
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
                                                        "proxy.observability._last_slot_details_cache",
                                                        cache_value,
                                                    ):
                                                        started = time.monotonic()
                                                        async with httpx.AsyncClient(
                                                            transport=transport,
                                                            base_url="http://test",
                                                        ) as ac:
                                                            resp = await ac.get(
                                                                "/llama/local/status"
                                                            )
                                                        elapsed = time.monotonic() - started
        return resp, elapsed, detail_mock

    @pytest.mark.asyncio
    async def test_detail_timeout_uses_status_query_timeout_budget(self):
        """The STATUS_QUERY_TIMEOUT value is passed through to the detail query."""
        resp, _, detail_mock = await self._get(
            detail_side_effect=TimeoutError("slow /slots"),
            timeout_env="0.25",
            cache_value=self._CACHED_SLOTS,
        )
        assert resp.status_code == 200
        _, kwargs = detail_mock.await_args
        assert kwargs["timeout"] == 0.25
        assert resp.json()["slots"] == self._CACHED_SLOTS

    @pytest.mark.asyncio
    async def test_cached_fallback_response_within_budget(self):
        """A detail timeout with a cache hit returns promptly (in-memory fallback)."""
        resp, elapsed, _ = await self._get(
            detail_side_effect=TimeoutError("slow /slots"),
            timeout_env="1.0",
            cache_value=self._CACHED_SLOTS,
        )
        assert resp.status_code == 200
        assert resp.json()["slots_stale"] is True
        # No new I/O on the fallback path; allow generous CI headroom.
        assert elapsed < 1.0
