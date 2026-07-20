"""Parity tests for the llama-status consolidation refactor (LP-0MR6Y11OP005UHIH).

Extracted helpers
-----------------
- ``_safe_parse_json_response(response)`` — defensive JSON parsing with
  async/sync ``.json()`` → ``.text`` → ``json.loads()`` fallback.
- ``_build_llama_url(llama_port, endpoint)`` — URL builder for llama-server.
- ``_query_slots(client, llama_port, timeout)`` — query /slots endpoint.

These helpers were extracted from duplicated code in ``query_llama_status()``
(observability.py) and ``get_llama_local_status()`` (handlers.py).  The parity
tests below lock their behaviour so the extraction is safe.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.refactor_parity


# ─── helpers for mocking responses ───────────────────────────────────────────


class AsyncJsonResponse:
    """Mock response where ``.json()`` returns a coroutine (like mock objects)."""

    def __init__(self, status_code=200, json_data=None, text_data=None):
        self.status_code = status_code
        self._json_data = json_data
        self._text_data = text_data

    async def json(self):
        return self._json_data

    @property
    def text(self):
        return self._text_data or json.dumps(self._json_data) if self._json_data else ""


class SyncJsonResponse:
    """Mock response where ``.json()`` is synchronous (like httpx.Response)."""

    def __init__(self, status_code=200, json_data=None, text_data=None):
        self.status_code = status_code
        self._json_data = json_data
        self._text_data = text_data

    def json(self):
        return self._json_data

    @property
    def text(self):
        return self._text_data or json.dumps(self._json_data) if self._json_data else ""


class NoJsonResponse:
    """Mock response without ``.json()`` method (rare edge case)."""

    def __init__(self, status_code=200, text_data=None):
        self.status_code = status_code
        self._text_data = text_data

    @property
    def text(self):
        return self._text_data or ""


class InvalidJsonResponse:
    """Mock response that raises on ``.json()`` but has valid text fallback."""

    def __init__(self, status_code=200, text_data=None):
        self.status_code = status_code
        self._text_data = text_data

    async def json(self):
        raise ValueError("invalid json from .json()")

    @property
    def text(self):
        return self._text_data or "{}"


# ═══════════════════════════════════════════════════════════════════════════
# _safe_parse_json_response
# ═══════════════════════════════════════════════════════════════════════════


class TestSafeParseJsonResponse:
    """Contract tests for ``_safe_parse_json_response``.

    The extracted helper must handle all response variants that the original
    duplicated code handled:
    - Async ``.json()`` (coroutine) — used by test mocks
    - Sync ``.json()`` (returns dict directly) — used by httpx.Response
    - ``.json()`` failure → ``.text`` fallback → ``json.loads()``
    - No ``.json()`` method → ``.text`` fallback
    - No viable parsing → returns ``None``
    """

    @pytest.mark.asyncio
    async def test_async_json_returns_dict(self):
        """Async .json() returns parsed dict directly."""
        from proxy.observability import _safe_parse_json_response

        resp = AsyncJsonResponse(json_data={"n_ctx": 4096})
        result = await _safe_parse_json_response(resp)
        assert result == {"n_ctx": 4096}

    @pytest.mark.asyncio
    async def test_sync_json_returns_dict(self):
        """Sync .json() (httpx style) returns parsed dict."""
        from proxy.observability import _safe_parse_json_response

        resp = SyncJsonResponse(json_data={"n_ctx": 4096})
        result = await _safe_parse_json_response(resp)
        assert result == {"n_ctx": 4096}

    @pytest.mark.asyncio
    async def test_json_falls_back_to_text(self):
        """When .json() raises, falls back to .text + json.loads."""
        from proxy.observability import _safe_parse_json_response

        resp = InvalidJsonResponse(text_data='{"n_ctx": 8192}')
        result = await _safe_parse_json_response(resp)
        assert result == {"n_ctx": 8192}

    @pytest.mark.asyncio
    async def test_no_json_method_uses_text(self):
        """Response without .json() method uses .text fallback."""
        from proxy.observability import _safe_parse_json_response

        resp = NoJsonResponse(text_data='{"llama_server_running": true}')
        result = await _safe_parse_json_response(resp)
        assert result == {"llama_server_running": True}

    @pytest.mark.asyncio
    async def test_both_parse_paths_fail_returns_none(self):
        """When both .json() and .text parsing fail, returns None."""
        from proxy.observability import _safe_parse_json_response

        resp = InvalidJsonResponse(text_data="not-json")
        result = await _safe_parse_json_response(resp)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_text_method_returns_none(self):
        """Response with no .text method and no .json returns None."""
        from proxy.observability import _safe_parse_json_response

        class BareResponse:
            def __init__(self):
                self.status_code = 200

        resp = BareResponse()
        result = await _safe_parse_json_response(resp)
        assert result is None

    @pytest.mark.asyncio
    async def test_json_returns_non_dict_uses_as_is(self):
        """If .json() returns a list, it passes through (for /slots)."""
        from proxy.observability import _safe_parse_json_response

        resp = AsyncJsonResponse(json_data=[{"is_processing": False}])
        result = await _safe_parse_json_response(resp)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["is_processing"] is False


# ═══════════════════════════════════════════════════════════════════════════
# _build_llama_url
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildLlamaUrl:
    """Contract tests for ``_build_llama_url``."""

    def test_builds_url_with_port_and_endpoint(self):
        from proxy.observability import _build_llama_url
        url = _build_llama_url(8080, "/slots")
        assert url == "http://localhost:8080/slots"

    def test_builds_url_with_custom_port(self):
        from proxy.observability import _build_llama_url
        url = _build_llama_url(9090, "/status")
        assert url == "http://localhost:9090/status"

    def test_builds_url_with_model_endpoint(self):
        from proxy.observability import _build_llama_url
        url = _build_llama_url(8080, "/model")
        assert url == "http://localhost:8080/model"

    def test_endpoint_has_leading_slash(self):
        from proxy.observability import _build_llama_url
        url = _build_llama_url(8080, "slots")
        assert url == "http://localhost:8080/slots"


# ═══════════════════════════════════════════════════════════════════════════
# _query_slots
# ═══════════════════════════════════════════════════════════════════════════


class MockAsyncClient:
    """Minimal async HTTP client for testing."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []

    async def get(self, url, **kwargs):
        self.calls.append(url)
        if self.responses:
            return self.responses.pop(0)
        return AsyncJsonResponse(404)


class TestQuerySlots:
    """Contract tests for ``_query_slots``."""

    @pytest.mark.asyncio
    async def test_returns_available_and_total_slots(self):
        from proxy.observability import _query_slots

        slots_data = [
            {"is_processing": False},
            {"is_processing": True},
            {"is_processing": False},
        ]
        client = MockAsyncClient([AsyncJsonResponse(200, slots_data)])
        available, total = await _query_slots(client, 8080)
        assert total == 3
        assert available == 2

    @pytest.mark.asyncio
    async def test_all_slots_processing(self):
        from proxy.observability import _query_slots

        slots_data = [
            {"is_processing": True},
            {"is_processing": True},
        ]
        client = MockAsyncClient([AsyncJsonResponse(200, slots_data)])
        available, total = await _query_slots(client, 8080)
        assert total == 2
        assert available == 0

    @pytest.mark.asyncio
    async def test_empty_slots_list(self):
        from proxy.observability import _query_slots

        client = MockAsyncClient([AsyncJsonResponse(200, [])])
        available, total = await _query_slots(client, 8080)
        assert total == 0
        assert available == 0

    @pytest.mark.asyncio
    async def test_slots_default_is_processing_to_true(self):
        from proxy.observability import _query_slots

        # If is_processing is missing, it defaults to True
        slots_data = [
            {},
            {"is_processing": False},
        ]
        client = MockAsyncClient([AsyncJsonResponse(200, slots_data)])
        available, total = await _query_slots(client, 8080)
        assert total == 2
        assert available == 1

    @pytest.mark.asyncio
    async def test_returns_zeroes_on_non_list_response(self):
        from proxy.observability import _query_slots

        # /slots returns a dict (unexpected) — should return (0, 0)
        client = MockAsyncClient([AsyncJsonResponse(200, {"error": "not a list"})])
        available, total = await _query_slots(client, 8080)
        assert available == 0
        assert total == 0

    @pytest.mark.asyncio
    async def test_returns_zeroes_on_http_error(self):
        from proxy.observability import _query_slots

        client = MockAsyncClient([AsyncJsonResponse(404)])
        available, total = await _query_slots(client, 8080)
        assert available == 0
        assert total == 0

    @pytest.mark.asyncio
    async def test_returns_zeroes_on_connection_error(self):
        from proxy.observability import _query_slots

        async def raise_error(*args, **kwargs):
            raise ConnectionError("connection refused")

        client = MagicMock()
        client.get = raise_error
        available, total = await _query_slots(client, 8080)
        assert available == 0
        assert total == 0

    @pytest.mark.asyncio
    async def test_returns_zeroes_on_timeout(self):
        from proxy.observability import _query_slots

        async def slow(*args, **kwargs):
            await asyncio.sleep(10)

        client = MagicMock()
        client.get = slow
        available, total = await asyncio.wait_for(
            _query_slots(client, 8080, timeout=0.05), timeout=1.0
        )
        assert available == 0
        assert total == 0

    @pytest.mark.asyncio
    async def test_correct_url_and_port(self):
        from proxy.observability import _query_slots

        client = MockAsyncClient([AsyncJsonResponse(200, [])])
        await _query_slots(client, 9090)
        assert any("localhost:9090/slots" in call for call in client.calls)


# ═══════════════════════════════════════════════════════════════════════════
# Integration-level: slots query via handler (mock query_llama_status)
# ═══════════════════════════════════════════════════════════════════════════


class TestSlotsInHandler:
    """Verify that the handler returns slot fields correctly when
    the slots query goes through the extracted helper."""

    @pytest.mark.asyncio
    async def test_handler_includes_slot_fields_when_server_running(self):
        """When llama is running, status includes slot fields (default 0 in test)."""
        from unittest.mock import patch

        import httpx

        from proxy import server as srv_module

        app = srv_module.app
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            with patch.object(
                srv_module, "query_llama_status", new_callable=AsyncMock
            ) as mock_qls:
                mock_qls.return_value = {"llama_server_running": True}
                resp = await ac.get("/llama/local/status")

        assert resp.status_code == 200
        j = resp.json()
        assert isinstance(j.get("available_slots"), int)
        assert isinstance(j.get("total_slots"), int)
        assert j["available_slots"] >= 0
        assert j["total_slots"] >= 0
