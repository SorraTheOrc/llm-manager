import asyncio
from unittest.mock import patch

import httpx
import pytest

pytestmark = pytest.mark.refactor_parity


@pytest.mark.asyncio
async def test_llama_local_status_not_running():
    from proxy.server import app

    async def fake_query():
        return {"llama_server_running": False}

    transport = httpx.ASGITransport(app=app)
    with patch("proxy.server.query_llama_status", side_effect=fake_query):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/llama/local/status")
            assert resp.status_code == 200
            j = resp.json()
            assert isinstance(j.get("active_query"), bool)
            assert isinstance(j.get("model_switch_in_progress"), bool)
            assert j.get("current_model") is None
            assert j.get("llama_server_running") is False
            assert isinstance(j.get("available_slots"), int)
            assert isinstance(j.get("total_slots"), int)
            assert j["available_slots"] == 0
            assert j["total_slots"] == 0


@pytest.mark.asyncio
async def test_llama_local_status_shows_local_owner_when_lease_active():
    """When a local dispatch lease is active, status returns the owner session and remaining time."""
    import time

    from proxy.server import app

    from proxy import server

    async def fake_query():
        return {"llama_server_running": True}

    transport = httpx.ASGITransport(app=app)

    # Pre-seed an active lease
    lease_expires_at = time.monotonic() + 120.0
    records = {
        "owner-session-abc": {
            "backend": "local",
            "started_at": time.monotonic(),
            "active": True,
            "expires_at": lease_expires_at,
        }
    }

    class FakeLock:
        def locked(self):
            return False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

    with patch("proxy.server.query_llama_status", side_effect=fake_query):
        with patch.object(server, "local_dispatch_records", records):
            with patch.object(server, "local_dispatch_records_lock", FakeLock()):
                with patch.object(server, "model_switch_refcount", 0):
                    with patch.object(server, "model_switch_lock", FakeLock()):
                        with patch.object(server, "background_loads", {}):
                            with patch.object(server, "current_model", "test-model"):
                                async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                                    resp = await ac.get("/llama/local/status")

    assert resp.status_code == 200
    j = resp.json()
    assert j.get("local_owner_session_id") == "owner-session-abc"
    assert j.get("local_owner_lease_remaining_seconds") is not None
    assert isinstance(j.get("local_owner_lease_remaining_seconds"), (int, float))
    assert j["local_owner_lease_remaining_seconds"] > 0
    assert j["local_owner_lease_remaining_seconds"] <= 120.0


@pytest.mark.asyncio
async def test_llama_local_status_shows_no_local_owner_when_no_lease():
    """When no local dispatch lease is active, status returns null for owner fields."""
    from proxy.server import app

    from proxy import server

    async def fake_query():
        return {"llama_server_running": True}

    transport = httpx.ASGITransport(app=app)

    # Empty records — no lease
    records = {}

    class FakeLock:
        def locked(self):
            return False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

    with patch("proxy.server.query_llama_status", side_effect=fake_query):
        with patch.object(server, "local_dispatch_records", records):
            with patch.object(server, "local_dispatch_records_lock", FakeLock()):
                with patch.object(server, "model_switch_refcount", 0):
                    with patch.object(server, "model_switch_lock", FakeLock()):
                        with patch.object(server, "background_loads", {}):
                            with patch.object(server, "current_model", "test-model"):
                                async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                                    resp = await ac.get("/llama/local/status")

    assert resp.status_code == 200
    j = resp.json()
    assert j.get("local_owner_session_id") is None
    assert j.get("local_owner_lease_remaining_seconds") is None


@pytest.mark.asyncio
async def test_llama_local_status_running_and_switch():
    from proxy.server import app

    from proxy import server

    async def fake_query():
        return {"llama_server_running": True}

    # simulate model switch lock and background loads
    class DummyLock:
        def locked(self):
            return True

    transport = httpx.ASGITransport(app=app)
    with patch("proxy.server.query_llama_status", side_effect=fake_query):
        # patch the model_switch_lock and background_loads
        with patch.object(server, "model_switch_lock", DummyLock()):
            with patch.object(server, "background_loads", {"m": True}):
                # also set a current_model value
                with patch.object(server, "current_model", "test-model"):
                    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                        resp = await ac.get("/llama/local/status")
                        assert resp.status_code == 200
                        j = resp.json()
                        assert isinstance(j.get("active_query"), bool)
                        assert j.get("model_switch_in_progress") is True
                        assert j.get("current_model") == "test-model"
                        assert j.get("llama_server_running") is True
                        assert isinstance(j.get("available_slots"), int)
                        assert isinstance(j.get("total_slots"), int)


# ======================================================================
# _query_slots model-param regression tests (LP-0MSHFGO0M003Q5BL)
# ======================================================================


class TestQuerySlotsModelParam:
    """Regression tests for /llama/local/status slot counts with ?model=.

    LP-0MSHFGO0M003Q5BL: the status endpoint reported total_slots=0 because
    ``_query_slots`` queried llama-server ``/slots`` without the required
    ``?model=`` parameter (llama-server returns HTTP 400 without it).
    """

    @pytest.mark.asyncio
    async def test_passes_model_param_in_url(self):
        """The model name is appended to the /slots URL as ?model=..."""
        from unittest.mock import AsyncMock, MagicMock

        from proxy.observability import _query_slots

        mock_client = AsyncMock()
        mock_response = MagicMock(status_code=200)
        mock_response.json = MagicMock(return_value=[{"is_processing": False}])
        mock_client.get.return_value = mock_response

        await _query_slots(mock_client, 8080, timeout=2.0, model="Qwen3")

        # The URL must carry ?model=Qwen3 (llama-server 400s without it)
        call_url = mock_client.get.call_args.args[0]
        assert call_url == "http://localhost:8080/slots?model=Qwen3"

    @pytest.mark.asyncio
    async def test_counts_available_and_total_slots(self):
        """Available/total slots reflect the real slot array when model is passed."""
        from unittest.mock import AsyncMock, MagicMock

        from proxy.observability import _query_slots

        mock_client = AsyncMock()
        mock_response = MagicMock(status_code=200)
        mock_response.json = MagicMock(
            return_value=[
                {"is_processing": False},
                {"is_processing": True},
                {"is_processing": False},
            ]
        )
        mock_client.get.return_value = mock_response

        available, total = await _query_slots(mock_client, 8080, timeout=2.0, model="Qwen3")
        assert (available, total) == (2, 3)

    @pytest.mark.asyncio
    async def test_defaults_to_zero_on_http_400(self):
        """Without a model param a 400 response yields (0, 0) — the original bug."""
        from unittest.mock import AsyncMock, MagicMock

        from proxy.observability import _query_slots

        mock_client = AsyncMock()
        mock_response = MagicMock(status_code=400)
        mock_response.json = MagicMock(return_value={"error": {"message": "model name is missing"}})
        mock_client.get.return_value = mock_response

        available, total = await _query_slots(mock_client, 8080, timeout=2.0)
        assert (available, total) == (0, 0)


# ======================================================================
# Fail-open slot capacity when no model is loaded (LP-0MSI06HPB0043MV1)
# ======================================================================


class TestFailOpenSlotCapacity:
    """When llama-server is up but no model is loaded, report configured capacity.

    LP-0MSI06HPB0043MV1: right after a restart, router-mode has no model
    loaded. Querying /slots without ?model= returns HTTP 400 -> 0/0, which
    wedges orchestrators (Herdr downtime worker) into "no capacity" mode.
    The status endpoint now reports the configured session_slot_pool_size
    instead of querying /slots when no model is loaded.
    """

    class _FakeLock:
        def locked(self):
            return False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

    @pytest.mark.asyncio
    async def test_running_no_model_reports_configured_capacity(self):
        """(a) running + no model -> total/available = configured pool size."""
        from unittest.mock import AsyncMock

        from proxy.server import app

        from proxy import server

        async def fake_query():
            return {"llama_server_running": True}

        transport = httpx.ASGITransport(app=app)
        slots_mock = AsyncMock(return_value=(0, 0))
        with patch("proxy.server.query_llama_status", side_effect=fake_query):
            with patch.object(server, "current_model", None):
                with patch.object(server, "model_switch_refcount", 0):
                    with patch.object(server, "model_switch_lock", self._FakeLock()):
                        with patch.object(server, "background_loads", {}):
                            with patch.object(
                                server,
                                "config",
                                {"server": {"session_slot_pool_size": 5, "llama_server_port": 8080}},
                            ):
                                with patch("proxy.observability._query_slots", slots_mock):
                                    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                                        resp = await ac.get("/llama/local/status")

        assert resp.status_code == 200
        j = resp.json()
        assert j["llama_server_running"] is True
        assert j["current_model"] is None
        assert j["total_slots"] == 5
        assert j["available_slots"] == 5
        # AC4: /slots must never be queried without a loaded model name
        slots_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_running_no_model_defaults_to_three_when_unconfigured(self):
        """Without session_slot_pool_size in config the default (3) is used."""
        from unittest.mock import AsyncMock

        from proxy.server import app

        from proxy import server

        async def fake_query():
            return {"llama_server_running": True}

        transport = httpx.ASGITransport(app=app)
        slots_mock = AsyncMock(return_value=(0, 0))
        with patch("proxy.server.query_llama_status", side_effect=fake_query):
            with patch.object(server, "current_model", None):
                with patch.object(server, "model_switch_refcount", 0):
                    with patch.object(server, "model_switch_lock", self._FakeLock()):
                        with patch.object(server, "background_loads", {}):
                            with patch.object(server, "config", {"server": {}}):
                                with patch("proxy.observability._query_slots", slots_mock):
                                    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                                        resp = await ac.get("/llama/local/status")

        assert resp.status_code == 200
        j = resp.json()
        assert j["total_slots"] == 3
        assert j["available_slots"] == 3
        slots_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_running_with_model_queries_slots_with_model(self):
        """(b) running + model loaded -> real counts from /slots (unchanged path)."""
        from unittest.mock import AsyncMock

        from proxy.server import app

        from proxy import server

        async def fake_query():
            return {"llama_server_running": True}

        transport = httpx.ASGITransport(app=app)
        slots_mock = AsyncMock(return_value=(2, 3))
        with patch("proxy.server.query_llama_status", side_effect=fake_query):
            with patch.object(server, "current_model", "test-model"):
                with patch.object(server, "model_switch_refcount", 0):
                    with patch.object(server, "model_switch_lock", self._FakeLock()):
                        with patch.object(server, "background_loads", {}):
                            with patch("proxy.observability._query_slots", slots_mock):
                                async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                                    resp = await ac.get("/llama/local/status")

        assert resp.status_code == 200
        j = resp.json()
        assert j["current_model"] == "test-model"
        assert j["total_slots"] == 3
        assert j["available_slots"] == 2
        slots_mock.assert_awaited_once()
        # the loaded model name must be passed through (AC2 / LP-0MSHFGO0M003Q5BL)
        _, kwargs = slots_mock.await_args
        assert kwargs["model"] == "test-model"


# ======================================================================
# local_active_query (LP-0MSL2ZLLS009RVKR)
# ======================================================================


class _LocalStatusHarness:
    """Build a /llama/local/status call with controllable counters."""

    class _FakeLock:
        def locked(self):
            return False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

    async def get(self, active_queries, local_active_queries):
        """Issue a status request with the given global/local counter values."""
        from proxy.server import app

        from proxy import server

        async def fake_query():
            return {"llama_server_running": True}

        transport = httpx.ASGITransport(app=app)
        with patch("proxy.server.query_llama_status", side_effect=fake_query):
            with patch.object(server, "active_queries", active_queries):
                with patch.object(server, "active_queries_lock", asyncio.Lock()):
                    with patch.object(server, "local_active_queries", local_active_queries):
                        with patch.object(server, "local_active_queries_lock", asyncio.Lock()):
                            with patch.object(server, "local_dispatch_records", {}):
                                with patch.object(server, "local_dispatch_records_lock", self._FakeLock()):
                                    with patch.object(server, "model_switch_refcount", 0):
                                        with patch.object(server, "model_switch_lock", self._FakeLock()):
                                            with patch.object(server, "background_loads", {}):
                                                with patch.object(server, "current_model", "test-model"):
                                                    async with httpx.AsyncClient(
                                                        transport=transport, base_url="http://test"
                                                    ) as ac:
                                                        resp = await ac.get("/llama/local/status")
        assert resp.status_code == 200
        return resp.json()


@pytest.mark.asyncio
async def test_local_active_query_remote_only_streams():
    """Remote provider streams drive global active but NOT local_active_query.

    LP-0MSL2ZLLS009RVKR: opencode-go → deepseek-v4-flash streams consume zero
    local Qwen3 slots yet keep active_query=true. With no local activity the
    status must report local_active_query=false so herdr sees idle.
    """
    j = await _LocalStatusHarness().get(active_queries=2, local_active_queries=0)
    assert j["active_query"] is True
    assert j["local_active_query"] is False


@pytest.mark.asyncio
async def test_local_active_local_stream_sets_both():
    """A local stream in flight sets both active_query and local_active_query."""
    j = await _LocalStatusHarness().get(active_queries=3, local_active_queries=1)
    assert j["active_query"] is True
    assert j["local_active_query"] is True


@pytest.mark.asyncio
async def test_local_active_no_activity_both_false():
    """No activity at all: both counters zero → both fields false."""
    j = await _LocalStatusHarness().get(active_queries=0, local_active_queries=0)
    assert j["active_query"] is False
    assert j["local_active_query"] is False


# ======================================================================
# Global active_queries recovery → status reports active_query=false
# (LP-0MSL1OX51003DOP4)
# ======================================================================


@pytest.mark.asyncio
async def test_llama_local_status_active_query_false_after_recovery():
    """A stuck global active_queries counter is recovered so status reports active_query=false.

    LP-0MSL1OX51003DOP4: the global counter had no recovery mechanism, so an
    abandoned stream left active_query=true forever, blocking herdr downtime
    dispatch. After the periodic in-process recovery
    (_recover_stuck_global_active_queries, run every 10s from
    _dispatch_cleanup_loop), the status endpoint must report
    active_query=false again — without a proxy restart.
    """
    import asyncio

    from proxy.router_helpers import _recover_stuck_global_active_queries
    from proxy.server import app

    from proxy import server

    async def fake_query():
        return {"llama_server_running": True}

    class FakeLock:
        def locked(self):
            return False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

    transport = httpx.ASGITransport(app=app)

    # Stuck global counter: >0 but no active local work (the RCA scenario:
    # active_query=true with slots free, no lease, no requests in flight).
    with patch("proxy.server.query_llama_status", side_effect=fake_query):
        with patch.object(server, "active_queries", 3):
            with patch.object(server, "active_queries_lock", asyncio.Lock()):
                with patch.object(server, "local_active_queries", 0):
                    with patch.object(server, "local_active_queries_lock", asyncio.Lock()):
                        with patch.object(server, "local_dispatch_records", {}):
                            with patch.object(server, "local_dispatch_records_lock", FakeLock()):
                                with patch.object(server, "model_switch_refcount", 0):
                                    with patch.object(server, "model_switch_lock", FakeLock()):
                                        with patch.object(server, "background_loads", {}):
                                            with patch.object(server, "current_model", "test-model"):
                                                # Before recovery: stuck counter reports active_query=true
                                                async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                                                    resp = await ac.get("/llama/local/status")
                                                    assert resp.status_code == 200
                                                    assert resp.json()["active_query"] is True, (
                                                        "Stuck counter should report active_query=true"
                                                    )

                                                # In-process periodic recovery (no restart)
                                                await _recover_stuck_global_active_queries(server)

                                                # After recovery: active_query=false
                                                async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                                                    resp = await ac.get("/llama/local/status")
                                                    assert resp.status_code == 200
                                                    assert resp.json()["active_query"] is False, (
                                                        "Recovered counter should report active_query=false"
                                                    )


# ======================================================================
# Per-slot details in /llama/local/status (LP-0MSORPUMX002LLIA)
# ======================================================================


class TestPerSlotDetails:
    """GET /llama/local/status exposes per-slot details from /slots.

    LP-0MSORPUMX002LLIA: herdr's downtime worker needs the SAME N slots to
    stay free for the whole idle threshold (WL-0MSG7P9N8009PCKG). The status
    endpoint now includes a ``slots`` array with per-slot identity
    (``slot_id``, ``is_processing``, ``n_decoded``) so consumers can track
    individual slots instead of just counts.
    """

    _FAKE_SLOTS = [
        {"slot_id": 0, "is_processing": False, "n_decoded": None},
        {"slot_id": 1, "is_processing": True, "n_decoded": 42},
        {"slot_id": 2, "is_processing": False, "n_decoded": None},
    ]

    class _FakeLock:
        def locked(self):
            return False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

    async def _get(
        self,
        query_result,
        slots_detail=None,
        slots_detail_side_effect=None,
        current_model="test-model",
        config=None,
    ):
        """Issue /llama/local/status with controllable /slots detail responses.

        Returns the parsed JSON body. ``_query_slots`` is always stubbed so
        the test never touches a real llama-server on localhost.
        """
        from unittest.mock import AsyncMock

        from proxy.server import app

        from proxy import server

        async def fake_query():
            return query_result

        slots_detail_mock = AsyncMock(
            return_value=slots_detail, side_effect=slots_detail_side_effect
        )
        slots_counts_mock = AsyncMock(return_value=(0, 0))
        transport = httpx.ASGITransport(app=app)
        with patch("proxy.server.query_llama_status", side_effect=fake_query):
            with patch.object(server, "current_model", current_model):
                with patch.object(server, "model_switch_refcount", 0):
                    with patch.object(server, "model_switch_lock", self._FakeLock()):
                        with patch.object(server, "background_loads", {}):
                            with patch.object(server, "local_dispatch_records", {}):
                                with patch.object(
                                    server,
                                    "local_dispatch_records_lock",
                                    self._FakeLock(),
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
                                            slots_detail_mock,
                                        ):
                                            with patch(
                                                "proxy.observability._query_slots",
                                                slots_counts_mock,
                                            ):
                                                async with httpx.AsyncClient(
                                                    transport=transport,
                                                    base_url="http://test",
                                                ) as ac:
                                                    resp = await ac.get(
                                                        "/llama/local/status"
                                                    )
        assert resp.status_code == 200
        return resp.json(), slots_detail_mock

    @pytest.mark.asyncio
    async def test_slots_present_when_running_and_model_loaded(self):
        """AC1: slots array reflects per-slot state when llama-server is up."""
        j, detail_mock = await self._get(
            query_result={"llama_server_running": True},
            slots_detail=self._FAKE_SLOTS,
        )
        assert j["slots"] == self._FAKE_SLOTS
        assert j["total_slots"] == 0  # counts path unchanged (stubbed)
        detail_mock.assert_awaited_once()
        _, kwargs = detail_mock.await_args
        assert kwargs["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_slots_empty_when_server_not_running(self):
        """AC2: slots is an empty array when llama-server is down."""
        j, detail_mock = await self._get(
            query_result={"llama_server_running": False},
        )
        assert j["llama_server_running"] is False
        assert j["slots"] == []
        detail_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_slots_empty_on_slots_query_timeout(self):
        """AC2/AC3: a slow /slots detail query must not blow the response budget.

        The handler wraps the detail fetch in its own error handling, so a
        timeout from the helper surfaces as an empty ``slots`` array while
        the endpoint still returns HTTP 200.
        """
        j, _ = await self._get(
            query_result={"llama_server_running": True},
            slots_detail_side_effect=TimeoutError("slots timed out"),
        )
        assert j["slots"] == []

    @pytest.mark.asyncio
    async def test_slots_empty_when_no_model_loaded(self):
        """AC2: no per-slot data when llama-server is up but no model is loaded.

        Without a model name the /slots endpoint 400s, so the fail-open path
        reports configured capacity and an empty slots array (explicit about
        having no per-slot data).
        """
        j, detail_mock = await self._get(
            query_result={"llama_server_running": True},
            current_model=None,
            config={"server": {"session_slot_pool_size": 5}},
        )
        assert j["total_slots"] == 5
        assert j["available_slots"] == 5
        assert j["slots"] == []
        detail_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_slot_dicts_are_compact_and_typed(self):
        """AC4: each slot dict carries only the compact per-slot fields.

        The endpoint passes through the helper's compact projection — no
        streaming state or session identifiers leak into the payload.
        """
        j, _ = await self._get(
            query_result={"llama_server_running": True},
            slots_detail=self._FAKE_SLOTS,
        )
        assert isinstance(j["slots"], list)
        assert len(j["slots"]) == 3
        for slot in j["slots"]:
            assert set(slot.keys()) == {"slot_id", "is_processing", "n_decoded"}
            assert isinstance(slot["slot_id"], int)
            assert isinstance(slot["is_processing"], bool)
            assert slot["n_decoded"] is None or isinstance(slot["n_decoded"], int)
