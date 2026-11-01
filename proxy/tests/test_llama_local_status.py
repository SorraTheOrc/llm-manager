import asyncio
import logging
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

pytestmark = pytest.mark.refactor_parity

#: Prompt-size history must be clear of test-local calibration, so this is the
#: main-checkout repo root for docs assertions (LP-0MVSD5RJW008F1IB AC2).
REPO_ROOT = Path(__file__).resolve().parents[2]


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
async def test_llama_local_status_normalises_tuple_lease_owner():
    """A per-endpoint (endpoint, session_id) lease key is exposed as the bare
    session-id string, never as a JSON array (WL-0MUKY4MAX0008I0L)."""
    import time

    from proxy.server import app

    from proxy import server

    async def fake_query():
        return {"llama_server_running": True}

    transport = httpx.ASGITransport(app=app)

    lease_expires_at = time.monotonic() + 60.0
    session_id = "01a0e6f0-38eb-71f2-9961-6d186bac2ded"
    records = {
        ("http://localhost:8080", session_id): {
            "backend": "http://localhost:8080",
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
    assert j.get("local_owner_session_id") == session_id
    assert isinstance(j.get("local_owner_session_id"), str)


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

    async def get(self, active_queries, local_active_queries, dispatch_records=None):
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
                            with patch.object(
                                server, "local_dispatch_records", dispatch_records or {}
                            ):
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
    import time

    j = await _LocalStatusHarness().get(
        active_queries=3,
        local_active_queries=1,
        dispatch_records={
            "live-session": {"active": True, "expires_at": time.monotonic() + 60}
        },
    )
    assert j["active_query"] is True
    assert j["local_active_query"] is True


@pytest.mark.asyncio
async def test_local_active_query_ignores_retained_lease_and_leaked_counter():
    """AC1 (LP-0MVKSXR4P001V0CM): a retained post-stream lease plus a leaked
    counter must not report local work in flight — no dispatch record is
    ``active``, so the truthful in-flight count is zero."""
    import time

    j = await _LocalStatusHarness().get(
        active_queries=2,
        local_active_queries=4,
        dispatch_records={
            "idle-holder": {"active": False, "expires_at": time.monotonic() + 60}
        },
    )
    assert j["local_active_query"] is False
    assert j["local_in_flight_requests"] == 0


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
        """AC1: slots array reflects per-slot state when llama-server is up.

        The base per-slot fields are preserved (additive contract); the new
        capacity fields are added alongside (LP-0MVGDUVGE0024YAI).
        """
        j, detail_mock = await self._get(
            query_result={"llama_server_running": True},
            slots_detail=self._FAKE_SLOTS,
        )
        assert len(j["slots"]) == len(self._FAKE_SLOTS)
        for served, base in zip(j["slots"], self._FAKE_SLOTS):
            for key in ("slot_id", "is_processing", "n_decoded"):
                assert served[key] == base[key]
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
    async def test_slot_dicts_carry_capacity_fields(self):
        """AC1/AC2/AC5: each slot dict is compact but carries the additive
        capacity fields (owner_session_id, lease_state, phase, context_tokens,
        prefill_eta_seconds) on top of the legacy base fields, plus the
        abandonment annotation (abandoned / abandonment_reason /
        abandoned_session_id / abandoned_at_seconds_ago, LP-0MVRMI5HK00558TB).
        """
        j, _ = await self._get(
            query_result={"llama_server_running": True},
            slots_detail=self._FAKE_SLOTS,
        )
        assert isinstance(j["slots"], list)
        assert len(j["slots"]) == 3
        for slot in j["slots"]:
            assert set(slot.keys()) == {
                "slot_id",
                "is_processing",
                "n_decoded",
                "context_tokens",
                "owner_session_id",
                "lease_state",
                "phase",
                "prefill_eta_seconds",
                "prefill_eta_kind",
                "prefill_eta_assumed_cached_ratio",
                "abandoned",
                "abandonment_reason",
                "abandoned_session_id",
                "abandoned_at_seconds_ago",
                "abandoned_scope",
            }
            # LP-0MVSD5RJW008F1IB AC2: the per-slot ETA is labelled as the
            # cache-cold estimate it is, on every slot.
            assert slot["prefill_eta_kind"] == "reference_cold_bound"
            assert slot["prefill_eta_assumed_cached_ratio"] == 0.0
            # The abandonment annotation is always present and always
            # attributable: a boolean alone is not enough. ``abandoned_scope``
            # is ``None`` unless the slot is abandoned (LP-0MVUVHA55009EU4B) and
            # then names which registry entry matched (``"slot"`` |
            # ``"endpoint"``).
            assert slot["abandoned"] is False
            assert slot["abandonment_reason"] is None
            assert slot["abandoned_session_id"] is None
            assert slot["abandoned_scope"] is None
            assert isinstance(slot["slot_id"], int)
            assert isinstance(slot["is_processing"], bool)
            assert slot["n_decoded"] is None or isinstance(slot["n_decoded"], int)
            assert isinstance(slot["lease_state"], str)
            assert slot["phase"] in ("prefill", "decode", None)
            assert slot["prefill_eta_seconds"] is None or isinstance(
                slot["prefill_eta_seconds"], (int, float)
            )
        # v5 (LP-0MVWFKIKJ005XHAY): the three ``local_route_*`` fields.
        assert j["status_schema_version"] == 5  # AC1: additive + versioned


# ======================================================================
# Prefill / queue ETA fields (LP-0MVGDUV5E005UWBQ)
# ======================================================================


class _EtaStatusHarness:
    """Status call with a controllable calibration window + server config."""

    class _FakeLock:
        def locked(self):
            return False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

    async def get(self, server_config, local_active_queries=0):
        from proxy.server import app

        from proxy import server

        async def fake_query():
            return {"llama_server_running": True}

        transport = httpx.ASGITransport(app=app)
        with patch("proxy.server.query_llama_status", side_effect=fake_query):
            with patch.object(server, "active_queries", local_active_queries):
                with patch.object(server, "active_queries_lock", asyncio.Lock()):
                    with patch.object(server, "local_active_queries", local_active_queries):
                        with patch.object(server, "local_active_queries_lock", asyncio.Lock()):
                            with patch.object(server, "local_dispatch_records", {}):
                                with patch.object(server, "local_dispatch_records_lock", self._FakeLock()):
                                    with patch.object(server, "model_switch_refcount", 0):
                                        with patch.object(server, "model_switch_lock", self._FakeLock()):
                                            with patch.object(server, "background_loads", {}):
                                                with patch.object(server, "current_model", "test-model"):
                                                    with patch.object(server, "config", server_config):
                                                        async with httpx.AsyncClient(
                                                            transport=transport, base_url="http://test"
                                                        ) as ac:
                                                            resp = await ac.get("/llama/local/status")
        assert resp.status_code == 200
        return resp.json()


@pytest.fixture
def _eta_calibration():
    from proxy import prefill_eta

    prefill_eta.reset()
    for _ in range(6):
        prefill_eta.record_dispatch_sample("test-model", 100_000, 0.0, 390_000.0)
    yield
    prefill_eta.reset()


@pytest.mark.asyncio
async def test_status_publishes_prefill_and_queue_eta(_eta_calibration):
    """The calibrated estimate is published so dispatchers can pace before
    launching a session (LP-0MVGDUV5E005UWBQ)."""
    server_config = {
        "server": {
            "llama_server_port": 8080,
            "contention_queue_max_wait_seconds": 60,
        }
    }
    j = await _EtaStatusHarness().get(server_config, local_active_queries=1)

    assert j["prefill_eta_seconds"] > 0
    assert j["queue_eta_seconds"] >= j["prefill_eta_seconds"]
    assert j["prefill_eta_calibrated"] is True
    assert j["prefill_eta_samples"] == 6
    assert j["prefill_eta_reference_tokens"] == 100_000


@pytest.mark.asyncio
async def test_status_publishes_the_shipped_swept_budget(_eta_calibration):
    """AC2 (LP-0MVT1146V007WXA4): the published budget is the shipped 30 s.

    A dispatcher paces its session against ``prefill_budget_seconds``; it must
    be the same number the admission guard and ``/reserve`` enforce, resolved
    from ``proxy/config.yaml`` through the one resolver — not a stale literal.
    """
    from config_test_utils import get_merged_config

    server_config = {"server": dict(get_merged_config("fast")["server"])}
    j = await _EtaStatusHarness().get(server_config)

    assert j["prefill_budget_seconds"] == 30.0


@pytest.mark.asyncio
async def test_status_labels_the_prefill_eta_as_a_cache_cold_reference_bound():
    """AC2 (LP-0MVSD5RJW008F1IB): the published bound is unmistakable.

    A dispatcher must be able to tell, from the payload alone, that
    ``prefill_eta_seconds`` is a cache-cold *reference-prompt* worst-case bound
    (not its own request's expected wait) and which reference size produced
    it. The labelling is asserted against the value it describes: an idle pool
    publishes exactly the cache-cold price of ``prefill_eta_reference_tokens``.
    """
    from proxy import prefill_eta

    prefill_eta.reset()
    for _ in range(6):
        prefill_eta.record_dispatch_sample("test-model", 8_000, 0.0, 8_000 * 3.9)
    server_config = {
        "server": {
            "llama_server_port": 8080,
            "contention_queue_max_wait_seconds": 60,
        }
    }
    try:
        j = await _EtaStatusHarness().get(server_config)
        expected, _coeffs = prefill_eta.estimate_prefill_seconds(
            "test-model",
            j["prefill_eta_reference_tokens"],
            j["prefill_eta_assumed_cached_ratio"],
            server_config,
        )
    finally:
        prefill_eta.reset()

    assert j["prefill_eta_kind"] == "reference_cold_bound"
    assert j["prefill_eta_assumed_cached_ratio"] == 0.0
    assert j["prefill_eta_reference_tokens"] == 8_000
    assert j["prefill_eta_seconds"] == pytest.approx(round(expected, 1))


@pytest.mark.asyncio
async def test_status_request_log_line_carries_the_calibration_context(caplog):
    """AC6 (LP-0MVSD5RJW008F1IB): the poll line says which reference sized it.

    The M3 metric could not band or correct anything because the
    ``status_request`` line published only the ETA; the calibration context
    ``status_estimate`` already computes must be on the line too.
    """
    from proxy import prefill_eta

    prefill_eta.reset()
    for _ in range(6):
        prefill_eta.record_dispatch_sample("test-model", 8_000, 0.0, 8_000 * 3.9)
    server_config = {
        "server": {
            "llama_server_port": 8080,
            "contention_queue_max_wait_seconds": 60,
        }
    }
    try:
        with caplog.at_level(logging.INFO, logger="llama-proxy"):
            await _EtaStatusHarness().get(server_config)
    finally:
        prefill_eta.reset()

    record = next(
        r for r in caplog.records if r.getMessage() == "status_request"
    )
    assert record.prefill_eta_reference_tokens == 8_000
    assert record.prefill_eta_calibrated is True
    assert record.prefill_eta_rate_bucket == "solitary"
    assert record.prefill_eta_samples == 6
    assert record.prefill_eta_kind == "reference_cold_bound"
    assert record.prefill_eta_assumed_cached_ratio == 0.0
    # The pre-existing fields are unchanged on the enriched line.
    assert record.prefill_eta_seconds is not None
    assert record.queue_eta_seconds is not None


def test_docs_state_the_published_field_is_a_cache_cold_reference_bound():
    """AC2: the human-facing docs make the same one-sentence statement.

    The payload labelling only helps a consumer that reads it; the router docs
    are where an operator/dispatcher author learns the semantics, so the
    statement is pinned here rather than left to prose drift.
    """
    docs = (REPO_ROOT / "docs" / "llama-router.md").read_text(encoding="utf-8")

    assert "cache-cold" in docs
    assert "reference-prompt worst-case bound" in docs
    assert "not your request's cost" in docs
    assert "prefill_eta_kind" in docs
    assert "prefill_eta_assumed_cached_ratio" in docs
    assert "prefill_eta_reference_tokens" in docs


@pytest.mark.asyncio
async def test_status_eta_fields_are_well_formed_without_calibration():
    from proxy import prefill_eta

    prefill_eta.reset()
    server_config = {
        "server": {
            "llama_server_port": 8080,
            "local_admission_guard_reference_tokens": 8_000,
        }
    }
    j = await _EtaStatusHarness().get(server_config)

    assert j["prefill_eta_seconds"] > 0
    assert j["queue_eta_seconds"] > 0
    assert j["prefill_eta_calibrated"] is False
    assert j["prefill_eta_reference_tokens"] == 8_000


@pytest.mark.asyncio
async def test_status_eta_failure_is_fail_open():
    """An estimate failure must not break the status payload."""
    server_config = {"server": {"llama_server_port": 8080}}
    with patch(
        "proxy.prefill_eta.status_estimate", side_effect=RuntimeError("boom")
    ):
        j = await _EtaStatusHarness().get(server_config)

    assert j["prefill_eta_seconds"] is None
    assert j["queue_eta_seconds"] is None
    assert j["prefill_eta_calibrated"] is False


# ======================================================================
# Truthful local capacity: slot phase + prefill ETA (LP-0MVGDUVGE0024YAI)
#
# The endpoint must distinguish a slot that is *generating* from one that is
# merely *leased but idle* (the reserved-but-idle false-busy case) and publish
# a calibrated prefill ETA so dispatchers pace before launching a session.
# ======================================================================


class _CapacityStatusHarness:
    """Status call with controllable slots, lease registries and calibration."""

    class _FakeLock:
        def locked(self):
            return False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

    async def get(
        self,
        slots_detail=None,
        *,
        stale_cache=None,
        current_model="test-model",
        config=None,
        slot_owners=None,
        dispatch_records=None,
        counts_outcome=None,
        counts=None,
        llama_running=True,
    ):
        """Issue /llama/local/status and return the parsed JSON body.

        ``stale_cache`` exercises the degraded path: the fresh detail query
        returns nothing and the last-known per-slot cache is served with
        ``slots_stale=true``. ``counts`` is the ``(available, total)`` tuple the
        fresh counts query returns (default ``(0, 0)``); it is deliberately
        independent of ``slots_detail`` because the two are separate ``/slots``
        reads. ``counts_outcome`` simulates the fresh counts
        query's ``status_out`` (e.g. ``{"status": 400, "reason": "http_400"}``
        for an unloaded model) while still returning ``(0, 0)``.
        ``llama_running`` selects the engine-health state the fake status query
        reports (``False`` exercises the payload's absent-safe paths).
        """
        from unittest.mock import AsyncMock

        from proxy.server import app

        from proxy import server

        async def fake_query():
            return {"llama_server_running": llama_running}

        detail_mock = AsyncMock(return_value=slots_detail or [])
        if counts_outcome is None:
            counts_mock = AsyncMock(return_value=counts or (0, 0))
        else:
            async def _counts(*_args, **kwargs):
                out = kwargs.get("status_out")
                if isinstance(out, dict):
                    out.update(counts_outcome)
                return (0, 0)

            counts_mock = AsyncMock(side_effect=_counts)
        transport = httpx.ASGITransport(app=app)
        resolved_config = config if config is not None else {
            "server": {
                "llama_server_port": 8080,
                "contention_queue_max_wait_seconds": 60,
            }
        }
        with patch("proxy.server.query_llama_status", side_effect=fake_query):
            with patch.object(server, "active_queries", 0):
                with patch.object(server, "active_queries_lock", asyncio.Lock()):
                    with patch.object(server, "local_active_queries", 0):
                        with patch.object(
                            server, "local_active_queries_lock", asyncio.Lock()
                        ):
                            with patch.object(server, "current_model", current_model):
                                with patch.object(server, "model_switch_refcount", 0):
                                    with patch.object(
                                        server, "model_switch_lock", self._FakeLock()
                                    ):
                                        with patch.object(server, "background_loads", {}):
                                            with patch.object(
                                                server,
                                                "local_dispatch_records",
                                                dispatch_records or {},
                                            ):
                                                with patch.object(
                                                    server,
                                                    "local_dispatch_records_lock",
                                                    self._FakeLock(),
                                                ):
                                                    with patch.object(
                                                        server, "config", resolved_config
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
                                                                    stale_cache,
                                                                ):
                                                                    with patch.dict(
                                                                        "proxy.session._slot_owners",
                                                                        slot_owners or {},
                                                                        clear=True,
                                                                    ):
                                                                        async with httpx.AsyncClient(
                                                                            transport=transport,
                                                                            base_url="http://test",
                                                                        ) as ac:
                                                                            resp = await ac.get(
                                                                                "/llama/local/status"
                                                                            )
        assert resp.status_code == 200
        return resp.json()


def _slot(slot_id, is_processing=False, n_decoded=None, context_tokens=None):
    return {
        "slot_id": slot_id,
        "is_processing": is_processing,
        "n_decoded": n_decoded,
        "context_tokens": context_tokens,
    }


# -- AC1: additive, versioned, old-shaped consumers unaffected ---------------


@pytest.mark.asyncio
async def test_new_capacity_fields_are_additive_and_versioned():
    """AC1: new fields are additive and versioned; an old-shaped consumer
    (reads only the pre-change fields) behaves exactly as today."""
    j = await _CapacityStatusHarness().get(
        slots_detail=[_slot(0, is_processing=False), _slot(1, is_processing=True, n_decoded=9)],
    )

    # v4 (LP-0MVRMI5HK00558TB): per-slot abandonment annotation, the additive
    # ``abandoned_busy`` hint value and ``local_abandoned_slots``.
    # v5 (LP-0MVWFKIKJ005XHAY): ``local_route_available`` /
    # ``local_route_reason`` / ``local_route_cooldown_remaining_s``.
    assert j["status_schema_version"] == 5
    assert j["capacity_hint"] in (
        "idle",
        "prefill_bound",
        "saturated",
        "abandoned_busy",
    )
    assert j["local_abandoned_slots"] == 0
    assert "prefill_budget_seconds" in j
    for slot in j["slots"]:
        assert {"lease_state", "phase", "owner_session_id", "prefill_eta_seconds"} <= set(slot)
    # ...and the legacy view is unchanged in value and type.
    assert j["available_slots"] == 0 and j["total_slots"] == 0
    assert isinstance(j["llama_server_running"], bool)
    legacy = [
        {k: s[k] for k in ("slot_id", "is_processing", "n_decoded")}
        for s in j["slots"]
    ]
    assert legacy == [
        {"slot_id": 0, "is_processing": False, "n_decoded": None},
        {"slot_id": 1, "is_processing": True, "n_decoded": 9},
    ]


# -- AC2: lease_state ----------------------------------------------------------


@pytest.mark.asyncio
async def test_lease_state_generating_while_processing():
    """AC2: a processing slot is ``generating`` — never ``free`` while a
    prefill or decode is in flight, even with no lease owner."""
    j = await _CapacityStatusHarness().get(
        slots_detail=[_slot(0, is_processing=True, n_decoded=0)],
    )
    assert j["slots"][0]["lease_state"] == "generating"


@pytest.mark.asyncio
async def test_lease_state_idle_leased_after_stream_finished():
    """AC2: a session holding an inactive (post-stream) lease marks the slot
    ``idle_leased`` — the reserved-but-idle case, not a working slot."""
    import time

    j = await _CapacityStatusHarness().get(
        slots_detail=[_slot(0, is_processing=False)],
        slot_owners={("", 0): "idle-holder"},
        dispatch_records={
            "idle-holder": {
                "active": False,
                "expires_at": time.monotonic() + 60.0,
            }
        },
    )
    slot = j["slots"][0]
    assert slot["lease_state"] == "idle_leased"
    assert slot["owner_session_id"] == "idle-holder"


@pytest.mark.asyncio
async def test_available_slots_free_while_hint_saturated_for_leased_slot():
    """LP-0MVUVHCAP004V0JI triage: ``available_slots`` and ``capacity_hint``
    answer different questions, so the pair can legitimately read
    ``available_slots: 1`` with ``capacity_hint: saturated``.

    ``available_slots`` counts the slots llama-server reports as *not
    processing*; the hint classifies admission capacity, and a post-stream
    lease still reserves a slot (``lease_state="idle_leased"``), so it is not
    capacity for a newcomer while another slot decodes. This reproduces the
    *shape* of the 2026-10-29 residual row 01:43:11 (the streaming dispatch
    finished at 01:43:10, its 30 s lease was renewed 0.5 s before the poll, and
    one other slot was decoding) — documented behaviour, not a defect.
    """
    import time

    j = await _CapacityStatusHarness().get(
        slots_detail=[
            _slot(0, is_processing=True, n_decoded=41),
            _slot(1, is_processing=False),
        ],
        counts=(1, 2),
        slot_owners={("", 1): "post-stream-holder"},
        dispatch_records={
            "post-stream-holder": {
                "active": False,
                "expires_at": time.monotonic() + 30.0,
            }
        },
    )
    assert j["available_slots"] == 1
    assert [s["lease_state"] for s in j["slots"]] == ["generating", "idle_leased"]
    assert j["capacity_hint"] == "saturated"


@pytest.mark.asyncio
async def test_lease_state_free_when_no_lease_and_not_processing():
    """AC2: not processing and no lease holder → ``free``."""
    j = await _CapacityStatusHarness().get(
        slots_detail=[_slot(0, is_processing=False)],
    )
    slot = j["slots"][0]
    assert slot["lease_state"] == "free"
    assert slot["owner_session_id"] is None


@pytest.mark.asyncio
async def test_lease_state_ignores_expired_lease():
    """AC2: an expired inactive lease does not keep the slot ``idle_leased``."""
    import time

    j = await _CapacityStatusHarness().get(
        slots_detail=[_slot(0, is_processing=False)],
        slot_owners={("", 0): "stale-holder"},
        dispatch_records={
            "stale-holder": {"active": False, "expires_at": time.monotonic() - 1.0}
        },
    )
    assert j["slots"][0]["lease_state"] == "free"


# -- AC5: phase inference ------------------------------------------------------


def test_infer_slot_phase_null_when_not_processing():
    from proxy.observability import infer_slot_phase

    assert infer_slot_phase(False, 42) is None
    assert infer_slot_phase(False, None) is None


def test_infer_slot_phase_prefill_without_decoded_tokens():
    from proxy.observability import infer_slot_phase

    # ``is_processing`` with n_decoded ≈ 0 (or absent) is prefill; prompt
    # progress corroborates but does not change the verdict.
    assert infer_slot_phase(True, 0) == "prefill"
    assert infer_slot_phase(True, None) == "prefill"
    assert infer_slot_phase(True, 0, context_tokens=4_096) == "prefill"


def test_infer_slot_phase_decode_with_decoded_tokens():
    from proxy.observability import infer_slot_phase

    assert infer_slot_phase(True, 42) == "decode"
    assert infer_slot_phase(True, 1, context_tokens=100_000) == "decode"


# -- AC3: shared calibration, nulls when not computable -----------------------


@pytest.mark.asyncio
async def test_per_slot_prefill_eta_sourced_from_shared_calibration(_eta_calibration):
    """AC3: per-slot ETA and the budget come from ``prefill_eta.py`` — the same
    calibration the admission guard uses, never a second estimator."""
    from proxy import prefill_eta

    config = {
        "server": {
            "llama_server_port": 8080,
            "contention_queue_max_wait_seconds": 60,
        }
    }
    j = await _CapacityStatusHarness().get(
        slots_detail=[_slot(0, is_processing=True, n_decoded=0, context_tokens=50_000)],
        config=config,
    )
    slot = j["slots"][0]
    expected, _coeffs = prefill_eta.estimate_prefill_seconds(
        "test-model", 50_000, 0.0, config
    )
    assert slot["phase"] == "prefill"
    assert slot["prefill_eta_seconds"] == round(expected, 1)
    assert slot["prefill_eta_seconds"] > 0
    # The budget is the admission guard's own wait budget, not a second source.
    assert j["prefill_budget_seconds"] == 60.0


@pytest.mark.asyncio
async def test_per_slot_prefill_eta_null_when_context_unknown(_eta_calibration):
    """AC3: unknown prompt progress → ``null`` ETA, never a misleading ``0``."""
    j = await _CapacityStatusHarness().get(
        slots_detail=[_slot(0, is_processing=True, n_decoded=0, context_tokens=None)],
    )
    slot = j["slots"][0]
    assert slot["phase"] == "prefill"
    assert slot["prefill_eta_seconds"] is None


@pytest.mark.asyncio
async def test_per_slot_prefill_eta_null_for_decoding_slot(_eta_calibration):
    """AC3: a slot past prefill has no prefill ETA (null, not zero)."""
    j = await _CapacityStatusHarness().get(
        slots_detail=[_slot(0, is_processing=True, n_decoded=42, context_tokens=1_000)],
    )
    slot = j["slots"][0]
    assert slot["phase"] == "decode"
    assert slot["prefill_eta_seconds"] is None


# -- AC4: capacity_hint decision table ---------------------------------------


def _capacity_state(lease_state, phase=None):
    return {"lease_state": lease_state, "phase": phase}


def test_capacity_hint_idle_when_a_slot_is_free():
    from proxy.observability import compute_capacity_hint

    slots = [
        _capacity_state("generating", "decode"),
        _capacity_state("free"),
    ]
    assert compute_capacity_hint(slots) == "idle"


def test_capacity_hint_idle_when_every_occupied_slot_is_idle_leased():
    from proxy.observability import compute_capacity_hint

    slots = [
        _capacity_state("idle_leased"),
        _capacity_state("idle_leased"),
    ]
    assert compute_capacity_hint(slots) == "idle"


def test_capacity_hint_prefill_bound_when_no_free_and_any_prefill():
    from proxy.observability import compute_capacity_hint

    slots = [
        _capacity_state("generating", "prefill"),
        _capacity_state("generating", "decode"),
    ]
    assert compute_capacity_hint(slots) == "prefill_bound"


def test_capacity_hint_saturated_when_no_free_and_all_decode():
    from proxy.observability import compute_capacity_hint

    slots = [
        _capacity_state("generating", "decode"),
        _capacity_state("generating", "decode"),
    ]
    assert compute_capacity_hint(slots) == "saturated"


def test_capacity_hint_idle_for_empty_slots():
    from proxy.observability import compute_capacity_hint

    assert compute_capacity_hint([]) == "idle"


@pytest.mark.asyncio
async def test_endpoint_publishes_capacity_hint():
    """AC4: the endpoint's capacity_hint follows the same decision table."""
    j = await _CapacityStatusHarness().get(
        slots_detail=[
            _slot(0, is_processing=True, n_decoded=0),  # prefill
            _slot(1, is_processing=True, n_decoded=12),  # decode
        ],
    )
    assert j["capacity_hint"] == "prefill_bound"


# -- AC7: staleness honesty ----------------------------------------------------


@pytest.mark.asyncio
async def test_stale_slots_null_eta_fields(_eta_calibration):
    """AC7: when the served slot data is stale, per-slot AND top-level ETA
    fields are null (never stale numbers); slots_stale stays true."""
    cached = [
        {
            "slot_id": 0,
            "is_processing": True,
            "n_decoded": 0,
            "context_tokens": 50_000,
        }
    ]
    j = await _CapacityStatusHarness().get(
        slots_detail=[], stale_cache=cached
    )

    assert j["slots_stale"] is True
    assert j["slots"][0]["prefill_eta_seconds"] is None
    assert j["prefill_eta_seconds"] is None
    assert j["queue_eta_seconds"] is None


# ======================================================================
# Stale capacity honesty (LP-0MVKSXR4P001V0CM AC1/AC3)
#
# A `capacity_hint` computed from cached `/slots` detail must never claim
# saturation without live evidence: stale `is_processing` is a fossil of a
# previous request, and retained leases / a leaked counter are not work.
# ======================================================================


def _stale_decode_slot():
    return {
        "slot_id": 0,
        "is_processing": True,
        "n_decoded": 5,
        "context_tokens": None,
    }


@pytest.mark.asyncio
async def test_stale_detail_without_live_work_reports_idle_capacity():
    """AC3: a stale snapshot that looks saturated, with no in-flight local
    request, degrades to ``idle`` and reconciles ``available_slots`` upward so
    a dispatcher polling both fields sees an open gate."""
    j = await _CapacityStatusHarness().get(
        slots_detail=[], stale_cache=[_stale_decode_slot()]
    )

    assert j["slots_stale"] is True
    assert j["slots_detail_stale"] is True
    assert j["capacity_hint"] == "idle"
    # total_slots (0) is reconciled to the configured pool when no current
    # counts are known, and the whole pool is reported available.
    assert j["total_slots"] == 3
    assert j["available_slots"] == 3


@pytest.mark.asyncio
async def test_stale_detail_with_live_work_never_reports_saturated():
    """AC1/AC3: live in-flight work makes the pool busy, but a stale snapshot
    alone is not evidence of decode — the hint is ``prefill_bound``, not
    ``saturated``."""
    import time
    from unittest.mock import patch as _patch

    with _patch(
        "proxy.local_activity.activity_snapshot",
        return_value={
            "idle_seconds": 300.0,
            "last_activity_at": None,
            "last_activity_source": "startup",
        },
    ):
        j = await _CapacityStatusHarness().get(
            slots_detail=[],
            stale_cache=[_stale_decode_slot()],
            dispatch_records={
                "live-session": {"active": True, "expires_at": time.monotonic() + 60}
            },
        )

    assert j["slots_detail_stale"] is True
    assert j["capacity_hint"] == "prefill_bound"
    assert j["local_in_flight_requests"] == 1


@pytest.mark.asyncio
async def test_stale_detail_keeps_saturated_with_fresh_decode_evidence():
    """AC3: fresh decode evidence (chunks arriving now) keeps the saturation
    claim — the exception the AC carves out."""
    import time
    from unittest.mock import patch as _patch

    with _patch(
        "proxy.local_activity.activity_snapshot",
        return_value={
            "idle_seconds": 2.0,
            "last_activity_at": None,
            "last_activity_source": "chunk",
        },
    ):
        j = await _CapacityStatusHarness().get(
            slots_detail=[],
            stale_cache=[_stale_decode_slot()],
            dispatch_records={
                "live-session": {"active": True, "expires_at": time.monotonic() + 60}
            },
        )

    assert j["slots_detail_stale"] is True
    assert j["capacity_hint"] == "saturated"


@pytest.mark.asyncio
async def test_stale_detail_ignores_leaked_counter_but_flags_in_flight_truthfully():
    """AC1: a leaked counter with an inactive (lease-retained) record does not
    hold the pool busy; ``local_active_query`` reflects live records only."""
    import time

    j = await _CapacityStatusHarness().get(
        slots_detail=[],
        stale_cache=[_stale_decode_slot()],
        dispatch_records={
            "idle-holder": {"active": False, "expires_at": time.monotonic() + 60}
        },
    )

    assert j["capacity_hint"] == "idle"
    assert j["local_active_query"] is False
    assert j["local_in_flight_requests"] == 0


def test_downgrade_stale_capacity_hint_leaves_fresh_hints_alone():
    from proxy.observability import downgrade_stale_capacity_hint

    for hint in ("idle", "prefill_bound", "saturated"):
        assert (
            downgrade_stale_capacity_hint(
                hint, stale=False, live_local_requests=0
            )
            == hint
        )


@pytest.mark.parametrize(
    "live,source,idle,expected",
    [
        (0, None, None, "idle"),
        (0, "chunk", 1.0, "idle"),  # no live request: a counter is not work
        (1, None, None, "prefill_bound"),
        (1, "dispatch", 1.0, "prefill_bound"),
        (1, "chunk", 2.0, "saturated"),
        (1, "chunk", 300.0, "prefill_bound"),  # evidence is stale
    ],
)
def test_downgrade_stale_capacity_hint_decision_table(
    live, source, idle, expected
):
    from proxy.observability import downgrade_stale_capacity_hint

    assert (
        downgrade_stale_capacity_hint(
            "saturated",
            stale=True,
            live_local_requests=live,
            activity_source=source,
            idle_seconds=idle,
        )
        == expected
    )


@pytest.mark.asyncio
async def test_status_publishes_live_prefill_count_only():
    """AC1/AC3: an orphaned prefill marker (no live dispatch record) is not
    published as an in-flight prefill; a marker backed by an active record is."""
    import time
    from unittest.mock import patch as _patch

    from proxy import server

    with _patch.object(server, "local_prefill_in_flight", {"orphan": True}):
        j = await _CapacityStatusHarness().get(
            slots_detail=[_slot(0, is_processing=False)],
            dispatch_records={},
        )
    assert j["local_prefills_in_flight"] == 0

    with _patch.object(server, "local_prefill_in_flight", {"live": True}):
        j = await _CapacityStatusHarness().get(
            slots_detail=[_slot(0, is_processing=True, n_decoded=0)],
            dispatch_records={
                "live": {"active": True, "expires_at": time.monotonic() + 60}
            },
        )
    assert j["local_prefills_in_flight"] == 1


@pytest.mark.asyncio
async def test_unloaded_model_is_fresh_absence_not_stale_saturation():
    """AC3: llama-server's router answers /slots?model=<name> with HTTP 400
    when it has unloaded the idle model. That is a definitive, fresh "no
    slots yet" — the handler must fail open on the configured pool and must
    not resurrect the cached (fossil, `is_processing=true`) snapshot as
    capacity."""
    j = await _CapacityStatusHarness().get(
        slots_detail=[],
        stale_cache=[_stale_decode_slot()],
        counts_outcome={"status": 400, "reason": "http_400"},
    )

    assert j["slots_stale"] is False
    assert j["slot_counts_stale"] is False
    assert j["slots_detail_stale"] is False
    assert j["slots"] == []
    assert j["capacity_hint"] == "idle"
    assert j["total_slots"] == 3
    assert j["available_slots"] == 3


@pytest.mark.asyncio
async def test_server_error_still_uses_graceful_degradation():
    """A transient 500 keeps the existing cache fallback, so the fossil is
    still served (with slots_stale) — but the hint is reconciled to idle."""
    j = await _CapacityStatusHarness().get(
        slots_detail=[],
        stale_cache=[_stale_decode_slot()],
        counts_outcome={"status": 500, "reason": "http_500"},
    )

    assert j["slots_stale"] is True
    assert j["slots_detail_stale"] is True
    assert j["capacity_hint"] == "idle"


@pytest.mark.asyncio
async def test_query_slots_reports_outcome_for_definitive_absence():
    """``_query_slots(status_out=...)`` exposes the fresh query outcome so a
    caller can tell a definitive model-absent answer from a transient fault."""
    from unittest.mock import AsyncMock
    from unittest.mock import patch as _patch

    from proxy.observability import _query_slots

    outcome: dict = {}
    with _patch(
        "proxy.observability._fetch_slots_payload",
        AsyncMock(return_value=(400, None)),
    ):
        available, total = await _query_slots(None, 8080, model="m", status_out=outcome)
    assert (available, total) == (0, 0)
    assert outcome == {"status": 400, "reason": "http_400"}

    outcome = {}
    with _patch(
        "proxy.observability._fetch_slots_payload",
        AsyncMock(
            return_value=(
                200,
                [
                    {"id": 0, "is_processing": False},
                    {"id": 1, "is_processing": True},
                ],
            )
        ),
    ):
        available, total = await _query_slots(None, 8080, model="m", status_out=outcome)
    assert (available, total) == (1, 2)
    assert outcome == {"status": 200, "reason": None}


# ======================================================================
# Endpoint-wide abandonment annotation (LP-0MVUVHA55009EU4B, test-first)
#
# The residual post-cancel release lag has two shapes: a recorded slot id that
# no longer matches the draining slot (shape 1) and a release with no slot
# identity at all (shape 2). Both must be annotated from an endpoint-wide
# registry entry, so `/llama/local/status` reports `abandoned_busy` rather than
# `saturated` while a ghost drains — without ever annotating live work.
#
# These are the payload-contract assertions; the registry and annotation
# mechanics are pinned in `test_abandoned_local_dispatch.py`. Every assertion
# here now runs unmarked: LP-0MVUVHA55009EU4B pinned them red-first with
# `xfail(strict=True)` and LP-0MVUVHB1V006HA5I removed those markers as it
# turned them green (strict xfail would otherwise fail the suite).
# ======================================================================

_ENDPOINT = "local-llama"
_ENDPOINT_WIDE_REASON = "measured_cache_deficit"

@pytest.fixture
def endpoint_wide_abandonment():
    """Record one slot-less abandonment, isolated to the test that uses it."""
    import time as _time

    from proxy.router_helpers import _record_abandoned_slot, _reset_abandoned_slots

    _reset_abandoned_slots()
    entry = _record_abandoned_slot(
        None,
        endpoint=_ENDPOINT,
        reason=_ENDPOINT_WIDE_REASON,
        session_id="herdr-ghost",
        now=_time.monotonic(),
    )
    assert entry is not None
    try:
        yield entry
    finally:
        _reset_abandoned_slots()


@pytest.mark.asyncio
async def test_endpoint_wide_entry_annotates_the_draining_slot(
    endpoint_wide_abandonment,
):
    """Shape 2 end-to-end: fresh detail, one processing slot, no live dispatch
    — ``abandoned_busy`` with one annotated slot; the annotation stays additive
    within the current schema version (5 since LP-0MVWFKIKJ005XHAY)."""
    j = await _CapacityStatusHarness().get(
        slots_detail=[_slot(1, is_processing=True, n_decoded=7)],
    )

    assert j["status_schema_version"] == 5
    assert j["capacity_hint"] == "abandoned_busy"
    assert j["capacity_hint"] != "saturated"
    assert j["capacity_hint"] != "idle"
    assert j["local_abandoned_slots"] == 1
    slot = j["slots"][0]
    assert slot["abandoned"] is True
    assert slot["abandonment_reason"] == _ENDPOINT_WIDE_REASON
    assert slot["abandoned_session_id"] == "herdr-ghost"
    assert slot["abandoned_scope"] == "endpoint"


@pytest.mark.asyncio
async def test_local_abandoned_slots_counts_annotated_slots_not_entries(
    endpoint_wide_abandonment,
):
    """One endpoint-wide entry matching two draining slots is reported as
    **two** abandoned slots: the count mirrors the ``slots`` array a consumer
    reads, not the raw registry size."""
    j = await _CapacityStatusHarness().get(
        slots_detail=[
            _slot(0, is_processing=True, n_decoded=3),
            _slot(1, is_processing=True, n_decoded=4),
        ],
    )

    assert [s["abandoned"] for s in j["slots"]] == [True, True]
    assert [s["abandoned_scope"] for s in j["slots"]] == ["endpoint", "endpoint"]
    assert j["local_abandoned_slots"] == 2
    assert j["capacity_hint"] == "abandoned_busy"


@pytest.mark.asyncio
async def test_endpoint_wide_entry_never_annotates_live_owned_work(
    endpoint_wide_abandonment,
):
    """Truthful status (LP-0MVKSXR4P001V0CM): the entry is present, but the
    processing slot is owned by a live dispatch lease, so the pool must still
    read ``saturated`` with nothing annotated."""
    import time as _time

    from proxy.session import _slot_registry_key

    dispatch_records = {
        "live-session": {
            "active": True,
            "expires_at": _time.monotonic() + 60.0,
            "first_byte_seen": True,
        }
    }
    j = await _CapacityStatusHarness().get(
        slots_detail=[_slot(1, is_processing=True, n_decoded=99)],
        slot_owners={_slot_registry_key(None, 1): "live-session"},
        dispatch_records=dispatch_records,
    )

    assert j["capacity_hint"] == "saturated"
    assert j["local_abandoned_slots"] == 0
    slot = j["slots"][0]
    assert slot.get("abandoned") is not True
    assert slot.get("abandoned_scope") is None


# ======================================================================
# Local-route availability on /llama/local/status
# (LP-0MVWFKIKJ005XHAY, parent LP-0MVWEZCB0006WPBQ AC10)
#
# Every capacity field above describes the ENGINE. A dispatcher also needs the
# ROUTE: on 2026-10-31 the local entry was quarantined (100 % of local attempts
# refused, 217 arms / 5.5 h) while this endpoint kept serving
# ``available_slots: 2, capacity_hint: idle``, so herdr's downtime dispatcher
# launched work into the poisoned route. The three additive fields are
# READ-ONLY observability: the routing decision never reads them.
# ======================================================================

_LOCAL_ENTRY_NAME = "local-qwen3"


def _route_config(*, local_entry=None, with_models=True) -> dict:
    """Server config carrying a local + remote provider chain.

    The same local entry is referenced by two models (as ``local-qwen3`` is in
    ``proxy/config.yaml``), which also exercises the entry de-duplication.
    """
    config = {"server": {"llama_server_port": 8080}}
    if not with_models:
        return config
    entry = local_entry or {
        "name": _LOCAL_ENTRY_NAME,
        "type": "local",
        "llama_model": "Qwen3",
    }
    config["models"] = {
        "plan": {
            "providers": [
                entry,
                {
                    "name": "remote-fallback",
                    "type": "remote",
                    "endpoint": "https://example.invalid/v1",
                },
            ]
        },
        "private": {"providers": [dict(entry)]},
    }
    return config


def _window_excluding_now() -> str:
    """A non-current ``HH:MM-HH:MM`` window (deterministic, no clock patching)."""
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    ahead = now + timedelta(hours=3)
    if ahead.date() == now.date():
        start, end = ahead, ahead + timedelta(minutes=1)
    else:
        start = now - timedelta(hours=3)
        end = start + timedelta(minutes=1)
    return f"{start:%H:%M}-{end:%H:%M}"


def _deficit_marked_503():
    """The retryable 503 the router returns for a measured cache-deficit abort."""
    from fastapi import Response
    from proxy.admission_reserve import HEADER_LOCAL_DISPATCH_REASON
    from proxy.measured_cache_abort import MEASURED_CACHE_DEFICIT_REASON

    return Response(
        content=b'{"error": {"message": "Backend unavailable, please retry later"}}',
        status_code=503,
        media_type="application/json",
        headers={
            "Retry-After": "11",
            HEADER_LOCAL_DISPATCH_REASON: MEASURED_CACHE_DEFICIT_REASON,
        },
    )


def _unmarked_503():
    """A plain local 503: a genuine fault, which still quarantines the entry."""
    from fastapi import Response

    return Response(
        content=b'{"error": {"message": "boom"}}',
        status_code=503,
        media_type="application/json",
        headers={"Retry-After": "11"},
    )


class TestLocalRouteAvailability:
    """AC10/AC10b/AC10c/AC10d: the route's armed state beside the engine's."""

    @pytest.fixture(autouse=True)
    def _isolated_armed_state(self):
        """Isolate the shared in-process cooldown/failure state per test."""
        import proxy.provider as provider

        from proxy import cold_start

        def _clear():
            provider._provider_unavailable_until.clear()
            provider._provider_failure_count.clear()
            cold_start.reset()

        _clear()
        yield
        _clear()

    @pytest.mark.asyncio
    async def test_idle_route_publishes_absent_safe_defaults(self):
        """AC10b/AC10d: nothing armed reads true / null / null (never a 0)."""
        j = await _CapacityStatusHarness().get(
            config=_route_config(), current_model="Qwen3"
        )

        assert j["status_schema_version"] == 5
        assert j["local_route_available"] is True
        assert j["local_route_reason"] is None
        # ``None`` — a fabricated ``0`` would read as "armed and expiring now".
        assert j["local_route_cooldown_remaining_s"] is None

    @pytest.mark.asyncio
    async def test_armed_cooldown_is_published_with_a_canonical_token(self):
        """AC10/AC10d: a real armed cooldown is reflected in the payload.

        The engine view is deliberately unchanged (``capacity_hint`` stays
        ``idle``): that disagreement is exactly the 2026-10-31 misread.
        """
        import proxy.provider as provider
        from proxy.admission_reserve import LOCAL_FALLBACK_REASON_TOKENS

        provider.mark_provider_unavailable(_LOCAL_ENTRY_NAME, 60)

        j = await _CapacityStatusHarness().get(config=_route_config())

        assert j["local_route_available"] is False
        assert j["local_route_reason"] == "local_provider_cooldown"
        assert j["local_route_reason"] in LOCAL_FALLBACK_REASON_TOKENS
        assert isinstance(j["local_route_cooldown_remaining_s"], int)
        assert 0 < j["local_route_cooldown_remaining_s"] <= 60
        # The engine still reports itself healthy: the two axes are independent.
        assert j["capacity_hint"] == "idle"
        assert j["local_abandoned_slots"] == 0

    @pytest.mark.asyncio
    async def test_brand_or_entry_arm_is_attributed_to_the_route(self):
        """AC10: the arm may live on the entry name OR its brand — either way
        the route is unavailable and the reason is the canonical cooldown token.

        ``_entry_cooldown_key`` is the same predicate chain resolution uses, so
        the published field cannot disagree with the skip line.
        """
        import proxy.provider as provider

        provider.mark_provider_unavailable("Qwen3-Brand", 45)
        entry = {
            "name": _LOCAL_ENTRY_NAME,
            "type": "local",
            "llama_model": "Qwen3",
            "provider": "Qwen3-Brand",
        }

        j = await _CapacityStatusHarness().get(
            config=_route_config(local_entry=entry), current_model="Qwen3"
        )

        assert j["local_route_available"] is False
        assert j["local_route_reason"] == "local_provider_cooldown"
        assert 0 < j["local_route_cooldown_remaining_s"] <= 45

    @pytest.mark.asyncio
    async def test_outside_available_window_reports_its_own_token(self):
        """AC10: an entry outside ``available_times`` is unavailable with the
        window token and **no** fabricated cooldown value."""
        from proxy.admission_reserve import LOCAL_FALLBACK_REASON_TOKENS

        entry = {
            "name": _LOCAL_ENTRY_NAME,
            "type": "local",
            "llama_model": "Qwen3",
            "available_times": [_window_excluding_now()],
        }

        j = await _CapacityStatusHarness().get(
            config=_route_config(local_entry=entry), current_model="Qwen3"
        )

        assert j["local_route_available"] is False
        assert j["local_route_reason"] == "outside_available_window"
        assert j["local_route_reason"] in LOCAL_FALLBACK_REASON_TOKENS
        assert j["local_route_cooldown_remaining_s"] is None

    @pytest.mark.asyncio
    async def test_deficit_abort_leaves_the_route_available_while_a_fault_arms_it(
        self,
    ):
        """AC10d: the post-fix deficit abort is NOT an entry fault.

        The router's deficit 503 is deferred by
        ``_handle_http_error_with_cooldown`` (LP-0MVWFKESK0005G61), so the
        published route stays available; an unmarked 503 from the same path is
        a genuine fault and still arms it. This is the cross-child contract: the
        status field must agree with the skip-reason decision.
        """
        import proxy.provider as provider

        attempts: list[dict] = []
        provider._handle_http_error_with_cooldown(
            _deficit_marked_503(),
            _LOCAL_ENTRY_NAME,
            "local",
            60,
            attempts,
            "measured cache deficit abort",
            config={"server": {"provider_cooldown_seconds": 60}},
        )
        assert attempts and attempts[0]["status"] == "local_dispatch_deferral"

        after_deficit = await _CapacityStatusHarness().get(
            config=_route_config(), current_model="Qwen3"
        )
        assert after_deficit["local_route_available"] is True
        assert after_deficit["local_route_reason"] is None
        assert after_deficit["local_route_cooldown_remaining_s"] is None

        provider._handle_http_error_with_cooldown(
            _unmarked_503(),
            _LOCAL_ENTRY_NAME,
            "local",
            60,
            attempts,
            "genuine local fault",
            config={"server": {"provider_cooldown_seconds": 60}},
        )

        after_fault = await _CapacityStatusHarness().get(
            config=_route_config(), current_model="Qwen3"
        )
        assert after_fault["local_route_available"] is False
        assert after_fault["local_route_reason"] == "local_provider_cooldown"
        assert 0 < after_fault["local_route_cooldown_remaining_s"] <= 60

    @pytest.mark.asyncio
    async def test_payload_is_absent_safe_without_a_local_entry(self):
        """AC10b: a config with no resolvable local entry stays well-formed."""
        j = await _CapacityStatusHarness().get(config=_route_config(with_models=False))

        assert j["status_schema_version"] == 5
        assert j["local_route_available"] is True
        assert j["local_route_reason"] is None
        assert j["local_route_cooldown_remaining_s"] is None

    @pytest.mark.asyncio
    async def test_route_state_is_published_while_the_engine_is_unavailable(self):
        """AC10b: route armed + engine state unavailable is still well-formed.

        The two axes are independent, so an unavailable ``llama_server_running``
        must not swallow the route evidence.
        """
        import proxy.provider as provider

        provider.mark_provider_unavailable(_LOCAL_ENTRY_NAME, 60)

        j = await _CapacityStatusHarness().get(
            config=_route_config(), llama_running=False
        )

        assert j["llama_server_running"] is False
        assert j["local_route_available"] is False
        assert j["local_route_reason"] == "local_provider_cooldown"
        assert 0 < j["local_route_cooldown_remaining_s"] <= 60

    @pytest.mark.asyncio
    async def test_reading_the_route_state_never_mutates_it(self):
        """AC10c: read-only — polling twice leaves the armed state intact.

        A handler that consumed or extended the cooldown would make this
        endpoint a routing participant; it must only observe.
        """
        import proxy.provider as provider

        provider.mark_provider_unavailable(_LOCAL_ENTRY_NAME, 60)
        before_until = dict(provider._provider_unavailable_until)
        before_failures = dict(provider._provider_failure_count)

        first = await _CapacityStatusHarness().get(
            config=_route_config(), current_model="Qwen3"
        )
        second = await _CapacityStatusHarness().get(
            config=_route_config(), current_model="Qwen3"
        )

        assert provider._provider_unavailable_until == before_until
        assert provider._provider_failure_count == before_failures
        assert first["local_route_available"] is second["local_route_available"]
        assert first["local_route_reason"] == second["local_route_reason"]
        # The armed window is still running (not consumed by the first poll).
        assert second["local_route_cooldown_remaining_s"] > 0

    @pytest.mark.asyncio
    async def test_status_request_log_line_carries_the_route_state(self, caplog):
        """AC10: the poll line answers "armed?" as well as "busy?".

        A skip reason that existed only in routing log lines forced log
        archaeology; the ``status_request`` line now carries the same verdict
        (plus the failure count that explains why it armed).
        """
        import proxy.provider as provider

        provider.mark_provider_unavailable(_LOCAL_ENTRY_NAME, 60)
        provider._provider_failure_count[_LOCAL_ENTRY_NAME] = 3

        with caplog.at_level(logging.INFO, logger="llama-proxy"):
            await _CapacityStatusHarness().get(
                config=_route_config(), current_model="Qwen3"
            )

        record = next(r for r in caplog.records if r.getMessage() == "status_request")
        assert record.local_route_available is False
        assert record.local_route_reason == "local_provider_cooldown"
        assert record.local_route_cooldown_remaining_s > 0
        assert record.local_route_failure_count == 3
        # Pre-existing fields are unchanged on the enriched line.
        assert isinstance(record.available_slots, int)
        assert record.capacity_hint == "idle"

    @pytest.mark.asyncio
    async def test_one_eligible_backend_keeps_the_route_available(self):
        """AC10: a multi-backend local model is available while any sibling is.

        Reporting the whole route unavailable because ONE backend is armed
        would hold back work that a healthy local backend could serve.
        """
        import proxy.provider as provider

        config = {
            "server": {"llama_server_port": 8080},
            "models": {
                "plan": {
                    "providers": [
                        {"name": "local-qwen3-a", "type": "local", "llama_model": "Qwen3"},
                        {"name": "local-qwen3-b", "type": "local", "llama_model": "Qwen3"},
                    ]
                }
            },
        }

        provider.mark_provider_unavailable("local-qwen3-a", 60)
        one_armed = await _CapacityStatusHarness().get(
            config=config, current_model="Qwen3"
        )
        assert one_armed["local_route_available"] is True
        assert one_armed["local_route_reason"] is None
        assert one_armed["local_route_cooldown_remaining_s"] is None

        # Both armed: the route is unavailable and the published delay is the
        # SHORTEST arm — how long until a local backend is usable again.
        provider.mark_provider_unavailable("local-qwen3-b", 20)
        both_armed = await _CapacityStatusHarness().get(
            config=config, current_model="Qwen3"
        )
        assert both_armed["local_route_available"] is False
        assert both_armed["local_route_reason"] == "local_provider_cooldown"
        assert 0 < both_armed["local_route_cooldown_remaining_s"] <= 20

    @pytest.mark.asyncio
    async def test_embed_and_chat_routes_are_judged_separately(self):
        """AC10: an armed embed entry must not declare the chat route armed.

        Both are ``type: local`` entries, so without model scoping the embed
        route would mask (or fake) the chat route's availability — the false
        positive this field exists to prevent.
        """
        import proxy.provider as provider

        config = {
            "server": {"llama_server_port": 8080},
            "models": {
                "embed": {
                    "providers": [
                        {
                            "name": "local-embed",
                            "type": "local",
                            "llama_model": "mxbai-embed",
                        }
                    ]
                },
                "plan": {
                    "providers": [
                        {"name": _LOCAL_ENTRY_NAME, "type": "local", "llama_model": "Qwen3"}
                    ]
                },
            },
        }

        provider.mark_provider_unavailable("local-embed", 60)

        chat = await _CapacityStatusHarness().get(
            config=config, current_model="Qwen3"
        )
        assert chat["local_route_available"] is True
        assert chat["local_route_reason"] is None

        embed = await _CapacityStatusHarness().get(
            config=config, current_model="mxbai-embed"
        )
        assert embed["local_route_available"] is False
        assert embed["local_route_reason"] == "local_provider_cooldown"
