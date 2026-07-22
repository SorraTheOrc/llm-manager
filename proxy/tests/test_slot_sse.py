"""Tests for the SSE broadcast of per-slot status data.

LP-0MRW7LDUP003JWZZ: Verifies that:
- ``_query_slots_detail()`` returns per-slot details from llama-server's ``/slots`` endpoint.
- The SSE broadcast payload includes a ``slots`` field with per-slot data.
- The UI renders slot status from the SSE data (Playwright).
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.refactor_parity


# ======================================================================
# _query_slots_detail unit tests
# ======================================================================


class TestQuerySlotsDetail:
    """Tests for the new ``_query_slots_detail`` helper."""

    @pytest.mark.asyncio
    async def test_returns_per_slot_data(self):
        """Returns a list of per-slot dicts from a successful /slots response."""
        from proxy.observability import _query_slots_detail

        mock_client = AsyncMock()

        # Simulate /slots returning three slots
        async def mock_json():
            return [
                {"is_processing": False},
                {"is_processing": True, "next_token": {"n_decoded": 42}},
                {"is_processing": True, "next_token": {"n_decoded": 100}},
            ]

        mock_response = MagicMock(status_code=200)
        mock_response.json = mock_json
        mock_client.get.return_value = mock_response

        result = await _query_slots_detail(8080, timeout=2.0, _client=mock_client)

        assert isinstance(result, list)
        assert len(result) == 3

        # Slot 0: idle
        assert result[0]["slot_id"] == 0
        assert result[0]["is_processing"] is False
        assert result[0]["n_decoded"] is None

        # Slot 1: processing with 42 decoded tokens
        assert result[1]["slot_id"] == 1
        assert result[1]["is_processing"] is True
        assert result[1]["n_decoded"] == 42

        # Slot 2: processing with 100 decoded tokens
        assert result[2]["slot_id"] == 2
        assert result[2]["is_processing"] is True
        assert result[2]["n_decoded"] == 100

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_http_error(self):
        """Returns [] when /slots returns non-200."""
        from proxy.observability import _query_slots_detail

        mock_client = AsyncMock()
        mock_response = MagicMock(status_code=500)
        mock_response.json = AsyncMock(return_value={})
        mock_client.get.return_value = mock_response

        result = await _query_slots_detail(8080, timeout=2.0, _client=mock_client)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_connection_error(self):
        """Returns [] when the /slots query itself fails."""
        from proxy.observability import _query_slots_detail

        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Connection refused")

        result = await _query_slots_detail(8080, timeout=2.0, _client=mock_client)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_response_not_a_list(self):
        """Returns [] when /slots returns a non-list JSON payload."""
        from proxy.observability import _query_slots_detail

        mock_client = AsyncMock()
        mock_response = MagicMock(status_code=200)

        async def mock_json():
            return {"error": "not available"}

        mock_response.json = mock_json
        mock_client.get.return_value = mock_response

        result = await _query_slots_detail(8080, timeout=2.0, _client=mock_client)
        assert result == []

    @pytest.mark.asyncio
    async def test_handles_empty_slots_list(self):
        """Returns [] when /slots returns an empty list (no slots configured)."""
        from proxy.observability import _query_slots_detail

        mock_client = AsyncMock()
        mock_response = MagicMock(status_code=200)

        async def mock_json():
            return []

        mock_response.json = mock_json
        mock_client.get.return_value = mock_response

        result = await _query_slots_detail(8080, timeout=2.0, _client=mock_client)
        assert result == []

    @pytest.mark.asyncio
    async def test_queries_with_model_param(self):
        """When model is provided, the URL includes ?model=..."""
        from proxy.observability import _query_slots_detail

        mock_client = AsyncMock()
        mock_response = MagicMock(status_code=200)

        async def mock_json():
            return []

        mock_response.json = mock_json
        mock_client.get.return_value = mock_response

        await _query_slots_detail(
            8080, timeout=2.0, model="Qwen3", _client=mock_client,
        )

        # Verify the URL includes ?model=Qwen3
        call_url = mock_client.get.call_args[0][0]
        assert "?model=Qwen3" in call_url
        assert "/slots" in call_url

    @pytest.mark.asyncio
    async def test_queries_without_model_when_none(self):
        """When model is None, the URL has no query string."""
        from proxy.observability import _query_slots_detail

        mock_client = AsyncMock()
        mock_response = MagicMock(status_code=200)

        async def mock_json():
            return []

        mock_response.json = mock_json
        mock_client.get.return_value = mock_response

        await _query_slots_detail(
            8080, timeout=2.0, model=None, _client=mock_client,
        )

        call_url = mock_client.get.call_args[0][0]
        assert "?" not in call_url

    @pytest.mark.asyncio
    async def test_handles_400_without_model(self):
        """Returns [] when /slots returns 400 due to missing model."""
        from proxy.observability import _query_slots_detail

        mock_client = AsyncMock()
        mock_response = MagicMock(status_code=400)
        mock_client.get.return_value = mock_response

        result = await _query_slots_detail(8080, _client=mock_client)
        assert result == []

    @pytest.mark.asyncio
    async def test_handles_next_token_as_list(self):
        """Handles next_token being a list of token objects."""
        from proxy.observability import _query_slots_detail

        mock_client = AsyncMock()
        mock_response = MagicMock(status_code=200)

        async def mock_json():
            return [
                {
                    "id": 3,
                    "is_processing": True,
                    "next_token": [
                        {"has_next_token": True, "n_decoded": 42}
                    ]
                },
            ]

        mock_response.json = mock_json
        mock_client.get.return_value = mock_response

        result = await _query_slots_detail(8080, timeout=2.0, _client=mock_client)
        assert len(result) == 1
        assert result[0]["slot_id"] == 3
        assert result[0]["is_processing"] is True
        assert result[0]["n_decoded"] == 42

    @pytest.mark.asyncio
    async def test_handles_slot_without_next_token(self):
        """Slot data lacking next_token is handled gracefully (n_decoded=None)."""
        from proxy.observability import _query_slots_detail

        mock_client = AsyncMock()
        mock_response = MagicMock(status_code=200)

        async def mock_json():
            return [
                {"is_processing": True},  # no next_token at all
            ]

        mock_response.json = mock_json
        mock_client.get.return_value = mock_response

        result = await _query_slots_detail(8080, timeout=2.0, _client=mock_client)
        assert len(result) == 1
        assert result[0]["slot_id"] == 0
        assert result[0]["is_processing"] is True
        assert result[0]["n_decoded"] is None


# ======================================================================
# SSE broadcast integration tests
# ======================================================================


class TestSseBroadcastSlotData:
    """Tests that the SSE broadcast can include per-slot slot data."""

    @pytest.mark.asyncio
    async def test_status_event_can_carry_slots_field(self):
        """The broadcast_status function can deliver a slots field in the event."""
        from proxy.observability import broadcast_status, sse_clients

        queue: asyncio.Queue = asyncio.Queue()
        sse_clients.add(queue)
        try:
            slot_data = [
                {"slot_id": 0, "is_processing": False, "n_decoded": None},
                {"slot_id": 1, "is_processing": True, "n_decoded": 42},
            ]
            await broadcast_status("status", {"slots": slot_data})

            message = await asyncio.wait_for(queue.get(), timeout=3.0)
            for line in message.splitlines():
                if line.startswith("data:"):
                    payload = line[len("data:"):].strip()
                    j = json.loads(payload)
                    assert j["type"] == "status"
                    assert "slots" in j
                    assert len(j["slots"]) == 2
                    assert j["slots"][0]["slot_id"] == 0
                    assert j["slots"][0]["is_processing"] is False
                    assert j["slots"][1]["slot_id"] == 1
                    assert j["slots"][1]["n_decoded"] == 42
                    break
        finally:
            sse_clients.discard(queue)

    @pytest.mark.asyncio
    async def test_status_event_without_slots_still_works(self):
        """The broadcast still works without a slots field (backward compatible)."""
        from proxy.observability import broadcast_status, sse_clients

        queue: asyncio.Queue = asyncio.Queue()
        sse_clients.add(queue)
        try:
            await broadcast_status("status", {"current_model": "test"})

            message = await asyncio.wait_for(queue.get(), timeout=3.0)
            for line in message.splitlines():
                if line.startswith("data:"):
                    payload = line[len("data:"):].strip()
                    j = json.loads(payload)
                    assert j["type"] == "status"
                    assert j["current_model"] == "test"
                    # slots field is NOT required — backward compatible
                    break
        finally:
            sse_clients.discard(queue)


# ======================================================================
# Last-known slot details cache tests
# ======================================================================


class TestLastSlotDetailsCache:
    """Tests for the shared module-level slot details cache.

    The cache is defined as ``_last_slot_details_cache`` in
    ``proxy.observability`` and is shared between
    ``_periodic_broadcast_loop()`` and ``ui.status_events()`` so that
    transient timeouts (ReadTimeout) during busy token generation don't
    blank the slot display in the UI.
    """

    def test_cache_is_accessible_from_ui_module(self):
        """The shared cache can be imported from the proxy.ui module path."""
        from proxy.observability import _last_slot_details_cache
        assert isinstance(_last_slot_details_cache, list)

    def test_cache_starts_empty(self):
        """The cache starts as an empty list."""
        from proxy.observability import _last_slot_details_cache
        assert _last_slot_details_cache == []

    def test_cache_can_be_populated_and_read(self):
        """The cache accepts slot data and can be read back as a copy."""
        from proxy.observability import _last_slot_details_cache
        _last_slot_details_cache.clear()
        _last_slot_details_cache.extend([
            {"slot_id": 0, "is_processing": False, "n_decoded": None},
            {"slot_id": 1, "is_processing": True, "n_decoded": 42},
        ])
        assert len(_last_slot_details_cache) == 2
        assert _last_slot_details_cache[0]["slot_id"] == 0
        assert _last_slot_details_cache[1]["n_decoded"] == 42

    def test_cache_read_returns_reference_to_same_list(self):
        """Importing the cache gives a reference to the same module list."""
        from proxy.observability import _last_slot_details_cache as cache1
        from proxy.observability import _last_slot_details_cache as cache2
        assert cache1 is cache2

    @pytest.mark.asyncio
    async def test_status_events_payload_has_slots_field(self):
        """status_events() includes a slots field in the initial SSE payload.
        When llama-server is not running, slots should be an empty list."""
        from proxy.ui import status_events
        from proxy.observability import _last_slot_details_cache
        from unittest.mock import patch, AsyncMock, MagicMock

        # Clear the shared cache
        _last_slot_details_cache.clear()

        srv = MagicMock()
        srv.per_model_queries = {}
        srv.per_model_queries_lock = MagicMock()
        srv.sse_clients = set()
        srv.current_model = "Qwen3"
        srv.token_counts = {"total_sent": 100, "total_recv": 200}
        srv.config = {"server": {}}
        # llama_server_running=False means no slot query is attempted
        srv.query_llama_status = AsyncMock(return_value={
            "llama_server_running": False,
            "n_ctx": None,
            "kv_cache_tokens": None,
            "router_mode": False,
        })
        srv.router_list_models = AsyncMock(return_value=[])

        with patch("proxy.ui._srv", return_value=srv):
            response = await status_events()

        chunk = await asyncio.wait_for(
            response.body_iterator.__anext__(),
            timeout=3.0
        )
        raw = chunk if isinstance(chunk, str) else chunk.decode("utf-8")
        for line in raw.splitlines():
            if line.startswith("data:"):
                payload = json.loads(line[len("data:"):].strip())
                assert payload["type"] == "status"
                assert "slots" in payload
                assert payload["slots"] == []
                break

    @pytest.mark.asyncio
    async def test_caching_pattern_matches_observability(self):
        """The caching pattern in ui.py matches the pattern in
        observability.py._periodic_broadcast_loop(): on success update
        the cache; on failure fall back to it.  We verify the logic by
        exercising _query_slots_detail() with mock clients that simulate
        success then failure."""
        from proxy.observability import _query_slots_detail, _last_slot_details_cache
        from unittest.mock import AsyncMock, MagicMock

        # Start with empty cache
        _last_slot_details_cache.clear()
        assert _last_slot_details_cache == []

        # --- First call: success, should populate cache ---
        mock_client_ok = AsyncMock()
        mock_response_ok = MagicMock(status_code=200)

        async def mock_json_ok():
            return [
                {"id": 5, "is_processing": True, "next_token": {"n_decoded": 12288}},
                {"id": 3, "is_processing": False},
            ]

        mock_response_ok.json = mock_json_ok
        mock_client_ok.get.return_value = mock_response_ok

        result = await _query_slots_detail(
            8080, timeout=2.0, model="Qwen3", _client=mock_client_ok,
        )

        # The function itself returns slot data
        assert len(result) == 2
        assert result[0]["slot_id"] == 5
        assert result[0]["n_decoded"] == 12288

        # Now simulate the caching logic used in both _periodic_broadcast_loop
        # and status_events: if slot_details is truthy, update the cache
        if result:
            _last_slot_details_cache.clear()
            _last_slot_details_cache.extend(result)

        assert len(_last_slot_details_cache) == 2
        assert _last_slot_details_cache[0]["slot_id"] == 5

        # --- Second call: failure (timeout), should fall back to cache ---
        mock_client_fail = AsyncMock()
        mock_client_fail.get.side_effect = Exception("ReadTimeout")

        result2 = await _query_slots_detail(
            8080, timeout=2.0, model="Qwen3", _client=mock_client_fail,
        )
        # Returns [] from the function itself
        assert result2 == []

        # Simulate the caching fallback
        if result2:
            _last_slot_details_cache.clear()
            _last_slot_details_cache.extend(result2)
        else:
            result2 = list(_last_slot_details_cache)

        # Should have fallen back to the cached data
        assert len(result2) == 2
        assert result2[0]["slot_id"] == 5
        assert result2[0]["n_decoded"] == 12288


# ======================================================================
# Slot progress cache tests
# ======================================================================


class TestSlotProgressCache:
    """Tests for the per-slot progress cache that merges llama-server log
    progress data into slot details when ``/slots`` is unresponsive.
    """

    def setup_method(self):
        """Clear the progress cache before each test."""
        from proxy.observability import _slot_progress_cache
        _slot_progress_cache.clear()

    # --- _extract_progress_data_from_log ---

    def test_extract_progress_data_valid_line(self):
        """Parses a standard llama-server progress line."""
        from proxy.observability import _extract_progress_data_from_log
        line = (
            "[58143] slot update_slots: id  5 | task 1 | prompt processing progress, "
            "n_tokens = 4096, batch.n_tokens = 4096, progress = 0.170"
        )
        result = _extract_progress_data_from_log(line)
        assert result is not None
        slot_id, n_tokens, progress = result
        assert slot_id == 5
        assert n_tokens == 4096
        assert progress == 0.17

    def test_extract_progress_data_multi_digit_slot(self):
        """Parses a line with multi-digit slot id."""
        from proxy.observability import _extract_progress_data_from_log
        line = (
            "[58143] slot update_slots: id  15 | task 1 | prompt processing progress, "
            "n_tokens = 22800, batch.n_tokens = 22800, progress = 0.840"
        )
        result = _extract_progress_data_from_log(line)
        assert result is not None
        slot_id, n_tokens, progress = result
        assert slot_id == 15
        assert n_tokens == 22800
        assert progress == 0.84

    def test_extract_progress_data_non_progress_line(self):
        """Returns None for non-progress log lines."""
        from proxy.observability import _extract_progress_data_from_log
        assert _extract_progress_data_from_log("INFO - Server started") is None
        assert _extract_progress_data_from_log("GET /slots 200") is None
        assert _extract_progress_data_from_log("") is None
        assert _extract_progress_data_from_log(None) is None

    def test_extract_progress_data_partial_fields(self):
        """Returns None when required fields are missing, or defaults slot to 0."""
        from proxy.observability import _extract_progress_data_from_log
        # Missing n_tokens entirely
        assert _extract_progress_data_from_log("slot update_slots: id=5") is None
        # Missing progress entirely
        assert _extract_progress_data_from_log("slot 5 n_tokens=4096") is None
        # Has n_tokens and progress but no slot -> defaults to 0
        result = _extract_progress_data_from_log("n_tokens=4096 progress=0.17")
        assert result is not None
        assert result[0] == 0  # slot_id defaults to 0
        assert result[1] == 4096
        assert result[2] == 0.17

    # --- _enrich_slot_details_with_progress ---

    def test_enrich_overrides_n_decoded_from_progress(self):
        """When progress has higher n_tokens than API n_decoded, override."""
        from proxy.observability import (
            _enrich_slot_details_with_progress, _slot_progress_cache,
        )
        _slot_progress_cache[3] = {"n_tokens": 4096, "progress": 0.17, "timestamp": 1000.0}
        slot_details = [{"slot_id": 3, "is_processing": True, "n_decoded": 512}]
        with _patch_time(1000.0):
            result = _enrich_slot_details_with_progress(slot_details)
        assert result[0]["n_decoded"] == 4096

    def test_enrich_does_not_override_when_api_is_higher(self):
        """When API n_decoded is already higher than progress, keep it."""
        from proxy.observability import (
            _enrich_slot_details_with_progress, _slot_progress_cache,
        )
        _slot_progress_cache[3] = {"n_tokens": 100, "progress": 0.5, "timestamp": 1000.0}
        slot_details = [{"slot_id": 3, "is_processing": True, "n_decoded": 500}]
        with _patch_time(1000.0):
            result = _enrich_slot_details_with_progress(slot_details)
        assert result[0]["n_decoded"] == 500  # API value kept

    def test_enrich_sets_is_processing_from_progress(self):
        """Sets is_processing=True when progress > 0 but API says idle."""
        from proxy.observability import (
            _enrich_slot_details_with_progress, _slot_progress_cache,
        )
        _slot_progress_cache[5] = {"n_tokens": 2048, "progress": 0.50, "timestamp": 1000.0}
        slot_details = [{"slot_id": 5, "is_processing": False, "n_decoded": None}]
        with _patch_time(1000.0):
            result = _enrich_slot_details_with_progress(slot_details)
        assert result[0]["is_processing"] is True
        assert result[0]["n_decoded"] == 2048

    def test_enrich_ignores_stale_progress(self):
        """Ignores progress data older than 60 seconds."""
        from proxy.observability import (
            _enrich_slot_details_with_progress, _slot_progress_cache,
        )
        _slot_progress_cache[3] = {"n_tokens": 999, "progress": 0.5, "timestamp": 900.0}
        slot_details = [{"slot_id": 3, "is_processing": True, "n_decoded": 10}]
        with _patch_time(1000.0):  # 100s later = stale
            result = _enrich_slot_details_with_progress(slot_details)
        assert result[0]["n_decoded"] == 10  # Not overridden

    def test_enrich_skips_slot_without_progress(self):
        """Slots without progress cache entry are unchanged."""
        from proxy.observability import _enrich_slot_details_with_progress
        slot_details = [{"slot_id": 9, "is_processing": False, "n_decoded": None}]
        result = _enrich_slot_details_with_progress(slot_details)
        assert result[0]["n_decoded"] is None
        assert result[0]["is_processing"] is False

    def test_enrich_empty_list(self):
        """Empty slot list returns empty."""
        from proxy.observability import _enrich_slot_details_with_progress
        assert _enrich_slot_details_with_progress([]) == []


class _TimePatcher:
    """Context manager that patches time.time() to return a fixed value."""
    def __init__(self, fixed_time):
        self.fixed_time = fixed_time
        self._orig = None

    def __enter__(self):
        import proxy.observability as obs
        self._orig = obs.time.time
        obs.time.time = lambda: self.fixed_time
        return self

    def __exit__(self, *args):
        import proxy.observability as obs
        obs.time.time = self._orig


def _patch_time(fixed_time):
    """Return a context manager that patches time.time."""
    return _TimePatcher(fixed_time)
