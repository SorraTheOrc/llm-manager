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
        prefill_eta_seconds) on top of the legacy base fields.
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
            }
            assert isinstance(slot["slot_id"], int)
            assert isinstance(slot["is_processing"], bool)
            assert slot["n_decoded"] is None or isinstance(slot["n_decoded"], int)
            assert isinstance(slot["lease_state"], str)
            assert slot["phase"] in ("prefill", "decode", None)
            assert slot["prefill_eta_seconds"] is None or isinstance(
                slot["prefill_eta_seconds"], (int, float)
            )
        assert j["status_schema_version"] == 2  # AC1: additive + versioned


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
            "contention_queue_max_wait_seconds": 120,
        }
    }
    j = await _EtaStatusHarness().get(server_config, local_active_queries=1)

    assert j["prefill_eta_seconds"] > 0
    assert j["queue_eta_seconds"] >= j["prefill_eta_seconds"]
    assert j["prefill_eta_calibrated"] is True
    assert j["prefill_eta_samples"] == 6
    assert j["prefill_eta_reference_tokens"] == 100_000


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
    ):
        """Issue /llama/local/status and return the parsed JSON body.

        ``stale_cache`` exercises the degraded path: the fresh detail query
        returns nothing and the last-known per-slot cache is served with
        ``slots_stale=true``.
        """
        from unittest.mock import AsyncMock

        from proxy.server import app

        from proxy import server

        async def fake_query():
            return {"llama_server_running": True}

        detail_mock = AsyncMock(return_value=slots_detail or [])
        counts_mock = AsyncMock(return_value=(0, 0))
        transport = httpx.ASGITransport(app=app)
        resolved_config = config if config is not None else {
            "server": {
                "llama_server_port": 8080,
                "contention_queue_max_wait_seconds": 120,
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

    assert j["status_schema_version"] == 2
    # New fields exist...
    assert j["capacity_hint"] in ("idle", "prefill_bound", "saturated")
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
            "contention_queue_max_wait_seconds": 120,
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
    assert j["prefill_budget_seconds"] == 120.0


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
