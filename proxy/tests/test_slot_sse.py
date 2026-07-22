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

        result = await _query_slots_detail(mock_client, llama_port=8080, timeout=2.0)

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

        result = await _query_slots_detail(mock_client, llama_port=8080, timeout=2.0)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_connection_error(self):
        """Returns [] when the /slots query itself fails."""
        from proxy.observability import _query_slots_detail

        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Connection refused")

        result = await _query_slots_detail(mock_client, llama_port=8080, timeout=2.0)
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

        result = await _query_slots_detail(mock_client, llama_port=8080, timeout=2.0)
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

        result = await _query_slots_detail(mock_client, llama_port=8080, timeout=2.0)
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
            mock_client, llama_port=8080, timeout=2.0, model="Qwen3",
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
            mock_client, llama_port=8080, timeout=2.0, model=None,
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

        result = await _query_slots_detail(mock_client, llama_port=8080)
        assert result == []

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

        result = await _query_slots_detail(mock_client, llama_port=8080, timeout=2.0)
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
