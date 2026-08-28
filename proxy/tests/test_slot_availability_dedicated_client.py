"""Tests for _check_slot_availability using a dedicated short-timeout client.

LP-0MTDH2U6V0062TUF (fix #2 for LP-0MTDGBRPU003Z7KU): the availability check
previously borrowed the shared ``_http_client`` (max 100 connections, 5s
default timeout). When the router ``/slots?model=`` call was slow (5-7s under
load), each call held a shared-pool connection; under multi-session load the
pool exhausted -> slot_save PoolTimeout (up to 60s) -> client-visible
"Request timed out".

Fix: ``_check_slot_availability`` creates its own per-call ``httpx.AsyncClient``
with a short timeout (config ``session_slot_availability_timeout_seconds``,
default 2.0), so a slow router/child response fails fast and can never starve
the shared pool. The check is best-effort (exception -> pass), so a fast
timeout degrades gracefully.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ======================================================================
# Dedicated-client behaviour
# ======================================================================


class TestCheckSlotAvailabilityDedicatedClient:
    """The availability check uses its own short-timeout client, not the shared pool."""

    def _make_srv(self):
        proc = SimpleNamespace(pid=123)
        return SimpleNamespace(
            log_dir=Path("/nonexistent"),
            llama_process=proc,
            current_model="Qwen3",
            _http_client=MagicMock(),  # shared client MUST NOT be used
        )

    def _mock_resp(self, slots):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = slots
        return resp

    @pytest.mark.asyncio
    async def test_uses_dedicated_client_not_shared(self, monkeypatch):
        """The shared _http_client is never used for the availability check."""
        from proxy import router_helpers as rh

        srv = self._make_srv()
        captured = {}

        class _Dedicated:
            def __init__(self, timeout):
                captured["timeout"] = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, timeout=None):
                captured["url"] = url
                return self._resp

        client = _Dedicated(timeout=None)
        client._resp = self._mock_resp(
            [{"id": 0, "is_processing": False}, {"id": 1, "is_processing": False}]
        )
        monkeypatch.setattr(
            rh, "httpx",
            SimpleNamespace(
                AsyncClient=lambda timeout: client,
                Timeout=lambda t: t,
            ),
        )
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda s: None)

        result = await rh._check_slot_availability(
            srv, {"llama_server_port": 8080}, 8080, "Qwen3", "Qwen3",
            "v1/chat/completions",
        )
        assert result is None
        # The shared client must NOT have been touched.
        srv._http_client.get.assert_not_called()
        assert captured["url"] == "http://localhost:8080/slots?model=Qwen3"

    @pytest.mark.asyncio
    async def test_dedicated_client_timeout_from_config(self, monkeypatch):
        """The dedicated client timeout comes from session_slot_availability_timeout_seconds."""
        from proxy import router_helpers as rh

        srv = self._make_srv()
        captured = {}

        class _Dedicated:
            def __init__(self, timeout):
                captured["timeout"] = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, timeout=None):
                return self._resp

        client = _Dedicated(timeout=None)
        client._resp = self._mock_resp([{"id": 0, "is_processing": False}])
        monkeypatch.setattr(
            rh, "httpx",
            SimpleNamespace(
                AsyncClient=lambda timeout: (captured.__setitem__("timeout", timeout), client)[1],
                Timeout=lambda t: t,
            ),
        )
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda s: None)

        result = await rh._check_slot_availability(
            srv,
            {"llama_server_port": 8080, "session_slot_availability_timeout_seconds": 1.5},
            8080, "Qwen3", "Qwen3", "v1/chat/completions",
        )
        assert result is None
        assert captured["timeout"] == 1.5

    @pytest.mark.asyncio
    async def test_default_timeout_is_two_seconds(self, monkeypatch):
        """Without config, the dedicated client timeout defaults to 2.0s."""
        from proxy import router_helpers as rh

        srv = self._make_srv()
        captured = {}

        class _Dedicated:
            def __init__(self, timeout):
                captured["timeout"] = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, timeout=None):
                return self._resp

        client = _Dedicated(timeout=None)
        client._resp = self._mock_resp([{"id": 0, "is_processing": False}])
        monkeypatch.setattr(
            rh, "httpx",
            SimpleNamespace(
                AsyncClient=lambda timeout: (captured.__setitem__("timeout", timeout), client)[1],
                Timeout=lambda t: t,
            ),
        )
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda s: None)

        result = await rh._check_slot_availability(
            srv, {"llama_server_port": 8080}, 8080, "Qwen3", "Qwen3",
            "v1/chat/completions",
        )
        assert result is None
        assert captured["timeout"] == 2.0

    @pytest.mark.asyncio
    async def test_slow_response_times_out_fast_and_returns_none(self, monkeypatch):
        """A /slots response slower than the short timeout fails fast -> None (best effort)."""
        from proxy import router_helpers as rh

        srv = self._make_srv()
        captured = {}

        class _Slow:
            def __init__(self, timeout):
                captured["timeout"] = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, timeout=None):
                captured["started"] = True
                await asyncio.sleep(10)  # exceeds the 2s timeout
                return MagicMock(status_code=200)

        client = _Slow(timeout=None)
        monkeypatch.setattr(rh, "httpx", SimpleNamespace(AsyncClient=lambda timeout: client))
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda s: None)

        start = asyncio.get_event_loop().time()
        result = await rh._check_slot_availability(
            srv, {"llama_server_port": 8080}, 8080, "Qwen3", "Qwen3",
            "v1/chat/completions",
        )
        elapsed = asyncio.get_event_loop().time() - start
        assert result is None  # best-effort: timeout swallowed
        assert elapsed < 5, f"check must fail fast, took {elapsed:.1f}s"

    @pytest.mark.asyncio
    async def test_503_still_returned_when_all_slots_busy(self, monkeypatch):
        """All slots busy via the dedicated client -> 503 slot-exhaustion response."""
        from proxy import router_helpers as rh

        srv = self._make_srv()
        captured = {}

        class _Dedicated:
            def __init__(self, timeout):
                captured["timeout"] = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, timeout=None):
                return self._resp

        client = _Dedicated(timeout=None)
        client._resp = self._mock_resp(
            [
                {"id": 0, "is_processing": True},
                {"id": 1, "is_processing": True},
                {"id": 2, "is_processing": True},
            ]
        )
        monkeypatch.setattr(
            rh, "httpx",
            SimpleNamespace(
                AsyncClient=lambda timeout: client,
                Timeout=lambda t: t,
            ),
        )
        monkeypatch.setattr(rh, "_discover_local_child_port", lambda s: None)

        result = await rh._check_slot_availability(
            srv, {"llama_server_port": 8080}, 8080, "Qwen3", "Qwen3",
            "v1/chat/completions",
        )
        assert result is not None
        assert result.status_code == 503
