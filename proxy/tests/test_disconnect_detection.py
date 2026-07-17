"""
Tests for client disconnect detection and cleanup.

Verifies that:
1. request.is_disconnected() is called periodically during streaming (AC5)
2. On disconnect, cleanup functions execute correctly (AC5)
3. Queued jobs are removed when the client disconnects (AC3/AC5)
4. Active queries counter does not leak (AC5)
5. Scheduler slots are released on disconnect (AC4)
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import pytest
from fastapi.responses import StreamingResponse
from proxy.slot_scheduler import JobScheduler


class AsyncIterator:
    """Helper to turn a list into an async iterator."""

    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        return self._iterator()

    async def _iterator(self):
        for item in self.items:
            yield item


# ===================================================================
# JobScheduler.remove_job() tests
# ===================================================================


class TestRemoveJob:
    """Tests for JobScheduler.remove_job()."""

    async def _make_scheduler(self, pool_size=2, max_queue_depth=4, job_timeout=300.0):
        sched = JobScheduler(
            pool_size=pool_size,
            max_queue_depth=max_queue_depth,
            job_timeout=job_timeout,
        )
        await sched.start()
        return sched

    async def _cleanup(self, sched):
        await sched.stop()

    @pytest.mark.asyncio
    async def test_remove_owned_slot_releases_it(self):
        """remove_job releases the slot when session owns one (AC4)."""
        s = await self._make_scheduler()
        try:
            result = await s.admit_job("session-a")
            assert result.kind == "ASSIGNED"
            slot_id = result.slot_id

            assert s.active_jobs["session-a"] == slot_id
            assert s.slots[slot_id].state == "Owned"

            await s.remove_job("session-a")

            assert "session-a" not in s.active_jobs
            assert s.slots[slot_id].state == "Idle"
        finally:
            await self._cleanup(s)

    @pytest.mark.asyncio
    async def test_remove_queued_job_clears_it(self):
        """remove_job removes a queued job from the queue (AC3)."""
        s = await self._make_scheduler()
        try:
            await s.admit_job("session-a")
            await s.admit_job("session-b")

            result = await s.admit_job("session-c")
            assert result.kind == "QUEUED"

            removed = await s.remove_job("session-c")
            assert removed is True

            assert "session-c" not in s._queued_jobs
            assert "session-c" in s._cancelled_jobs

            await s.release_slot(0)
            slot0 = s.slots[0]
            if slot0.state == "Owned":
                assert slot0.job_id != "session-c"
        finally:
            await self._cleanup(s)

    @pytest.mark.asyncio
    async def test_remove_unknown_session_noop(self):
        """remove_job on unknown session is a no-op."""
        s = await self._make_scheduler()
        try:
            result = await s.remove_job("nonexistent")
            assert result is False
        finally:
            await self._cleanup(s)

    @pytest.mark.asyncio
    async def test_remove_owned_releases_and_dequeues(self):
        """remove_job of an owner releases slot and assigns next queued job."""
        s = await self._make_scheduler()
        try:
            await s.admit_job("session-a")
            await s.admit_job("session-b")
            await s.admit_job("session-c")
            await s.admit_job("session-d")

            slot_b = s.active_jobs["session-b"]
            await s.remove_job("session-b")

            assert s.active_jobs.get("session-c") == slot_b
            assert s.slots[slot_b].job_id == "session-c"
        finally:
            await self._cleanup(s)

    @pytest.mark.asyncio
    async def test_remove_queued_does_not_affect_slots(self):
        """remove_job on a queued session does not affect active slots."""
        s = await self._make_scheduler()
        try:
            await s.admit_job("session-a")
            await s.admit_job("session-b")
            await s.admit_job("session-c")

            await s.remove_job("session-c")

            assert "session-a" in s.active_jobs
            assert "session-b" in s.active_jobs
            assert s.slots[0].state == "Owned"
            assert s.slots[1].state == "Owned"
        finally:
            await self._cleanup(s)

    @pytest.mark.asyncio
    async def test_remove_then_admit_same_session(self):
        """After remove_job, same session can be re-admitted."""
        s = await self._make_scheduler()
        try:
            _result1 = await s.admit_job("session-a")

            await s.remove_job("session-a")

            result2 = await s.admit_job("session-a")
            assert result2.kind == "ASSIGNED"
        finally:
            await self._cleanup(s)

    @pytest.mark.asyncio
    async def test_single_job_pool_remove(self):
        """With a single-slot scheduler, remove_job prevents timeout race."""
        sched = JobScheduler(pool_size=1, max_queue_depth=3, job_timeout=300.0)
        await sched.start()
        try:
            await sched.admit_job("session-a")
            await sched.remove_job("session-a")
            assert "session-a" not in sched.active_jobs
            assert sched.slots[0].state == "Idle"
        finally:
            await sched.stop()

    @pytest.mark.asyncio
    async def test_all_queued_cancelled_slot_stays_idle(self):
        """When all queued jobs are cancelled, released slot stays idle."""
        s = await self._make_scheduler()
        try:
            await s.admit_job("session-a")
            await s.admit_job("session-b")
            await s.admit_job("session-c")

            await s.remove_job("session-c")

            await s.release_slot(0)

            assert s.slots[0].state == "Idle"
        finally:
            await self._cleanup(s)


# ===================================================================
# Proxy remote streaming disconnect tests
# ===================================================================


def _make_mock_remote_stream_response(chunks, status=200):
    """Create a mock httpx response with streaming bytes."""
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.headers = {"content-type": "text/event-stream"}
    mock_resp.aiter_bytes = lambda: AsyncIterator(chunks)
    return mock_resp


@pytest.mark.asyncio
async def test_remote_stream_disconnect_detected():
    """Verify disconnect detection in proxy_remote streaming (AC5)."""
    from proxy.proxy_remote import _handle_remote_streaming

    mock_request = MagicMock(spec=["method", "url", "headers", "is_disconnected"])
    mock_request.method = "POST"
    type(mock_request.url).path = PropertyMock(return_value="/v1/chat/completions")
    mock_request.headers = {}
    mock_request.is_disconnected = AsyncMock(return_value=False)

    # Need 11+ chunks to trigger the disconnect check counter (check every 10)
    chunks = [
        b'data: {"choices": [{"delta": {"content": "A"}, "index": 0}]}\n\n',
    ] * 12

    with patch("proxy.proxy_remote.httpx.AsyncClient") as mock_client_cls:
        mock_cm = AsyncMock()
        mock_resp = _make_mock_remote_stream_response(chunks)
        mock_cm.__aenter__.return_value = mock_resp

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client

        with patch("proxy.proxy_remote.log_response_chunk"):
            result = await _handle_remote_streaming(
                mock_request,
                "http://fake.api/v1/chat/completions",
                {"Authorization": "Bearer test-key"},
                json.dumps({"model": "test-model", "stream": True}).encode(),
                {"model": "test-model", "stream": True},
                "test-model",
                httpx.Timeout(300),
            )

            assert isinstance(result, StreamingResponse)

            collected = b""
            async for chunk in result.body_iterator:
                collected += chunk

            assert mock_request.is_disconnected.called, (
                "request.is_disconnected() should be checked during remote streaming"
            )


@pytest.mark.asyncio
async def test_remote_stream_disconnect_cleanup():
    """Verify cleanup runs when disconnect detected in remote streaming."""
    from proxy.proxy_remote import _handle_remote_streaming

    mock_request = MagicMock(spec=["method", "url", "headers", "is_disconnected"])
    mock_request.method = "POST"
    type(mock_request.url).path = PropertyMock(return_value="/v1/chat/completions")
    mock_request.headers = {}
    mock_request.is_disconnected = AsyncMock(return_value=True)

    chunks = [
        b'data: {"choices": [{"delta": {"content": "Hi"}, "index": 0}]}\n\n',
    ]

    with patch("proxy.proxy_remote.httpx.AsyncClient") as mock_client_cls:
        mock_cm = AsyncMock()
        mock_resp = _make_mock_remote_stream_response(chunks)
        mock_cm.__aenter__.return_value = mock_resp

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client

        with patch("proxy.proxy_remote.log_response_chunk"):
            result = await _handle_remote_streaming(
                mock_request,
                "http://fake.api/v1/chat/completions",
                {"Authorization": "Bearer test-key"},
                json.dumps({"model": "test-model", "stream": True}).encode(),
                {"model": "test-model", "stream": True},
                "test-model",
                httpx.Timeout(300),
            )

            async for chunk in result.body_iterator:
                pass

            assert mock_cm.__aexit__.called, (
                "Context manager __aexit__ should be called on disconnect"
            )
            assert mock_client.aclose.called, (
                "httpx client aclose should be called on disconnect"
            )


# ===================================================================
# Router streaming disconnect detection
# ===================================================================

def _make_mock_request(is_disconnected=False):
    """Create a mock Starlette Request."""
    mock_req = MagicMock(spec=["is_disconnected", "headers", "method", "url"])
    mock_req.is_disconnected = AsyncMock(return_value=is_disconnected)
    mock_req.method = "POST"
    type(mock_req.url).path = PropertyMock(return_value="/v1/chat/completions")
    mock_req.headers = {}
    return mock_req


def _make_mock_stream_response(chunks, status=200):
    """Create a mock httpx response with streaming bytes."""
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.headers = {"content-type": "text/event-stream"}
    mock_resp.aiter_bytes = lambda: AsyncIterator(chunks)
    return mock_resp


@pytest.fixture
def mock_server():
    """Create a minimal mock server object."""
    srv = MagicMock()
    srv.active_queries = 0
    srv.active_queries_lock = asyncio.Lock()
    srv.backend_ready = True
    srv.llama_process = MagicMock()
    srv.current_model = "test-model"
    srv.logger = MagicMock()
    srv.session_manager = AsyncMock()
    srv.config = {
        "server": {
            "max_concurrent_queries": 4,
            "session_single_flight_mode": "queue",
            "session_single_flight_max_queue_depth": 1,
        }
    }
    return srv


@pytest.mark.asyncio
async def test_router_stream_disconnect_calls_is_disconnected(mock_server):
    """Verify request.is_disconnected() is called during streaming (AC5)."""
    from proxy.router import proxy_to_local

    mock_req = _make_mock_request(is_disconnected=False)

    chunks = [b'data: test\n\n'] * 15  # 15 chunks to ensure counter reaches 10

    mock_resp = _make_mock_stream_response(chunks)

    with (
        patch("proxy.router._srv", return_value=mock_server),
        patch("proxy.router._get_job_scheduler", return_value=None),
        patch("proxy.router._is_self_healing_active", return_value=False),
        patch("proxy.router._check_slot_availability", return_value=None),
        patch("proxy.router._increment_active_queries"),
        patch("proxy.router._decrement_active_queries"),
        patch("proxy.router._handle_session") as mock_session,
        patch("proxy.router._build_slot_context", return_value=(None, None, 3.0)),
        patch("proxy.router._call_with_backend_retries", new_callable=AsyncMock) as mock_retry,
        patch("proxy.router.httpx.AsyncClient") as mock_client_cls,
        patch("proxy.router._schedule_token_increment"),
        patch("proxy.router._schedule_recv_token_increment"),
        patch("proxy.router.count_text_tokens", return_value=5),
        patch("proxy.router._extract_delta_text_from_sse_chunk", return_value=""),
        patch("proxy.router.evaluate_stream_guardrail", return_value=None),
        patch("proxy.router.log_request"),
        patch("proxy.router.log_response_chunk"),
    ):
        mock_session.return_value = {
            "session_id": "test-session-123",
            "session_created": True,
            "is_delta_request": False,
            "session_fallback_reason": None,
            "delta_messages": None,
            "body_override": None,
            "body_json": {"messages": [], "model": "test-model", "stream": True},
            "original_message_count": 0,
        }

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_resp
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client
        mock_retry.return_value = (mock_cm, mock_resp)

        mock_req.body = AsyncMock(return_value=json.dumps(
            {"messages": [], "model": "test-model", "stream": True}
        ).encode())

        result = await proxy_to_local(mock_req, "v1/chat/completions")
        assert isinstance(result, StreamingResponse), (
            f"Expected StreamingResponse, got {type(result).__name__}"
        )

        collected = b""
        async for chunk in result.body_iterator:
            collected += chunk

        assert mock_req.is_disconnected.called, (
            "request.is_disconnected() should have been called during streaming"
        )


@pytest.mark.asyncio
async def test_router_stream_disconnect_triggers_cleanup(mock_server):
    """On disconnect, cleanup functions execute correctly (AC5)."""
    from proxy.router import proxy_to_local

    mock_req = _make_mock_request(is_disconnected=True)

    chunks = [b'data: test\n\n']

    mock_resp = _make_mock_stream_response(chunks)

    with (
        patch("proxy.router._srv", return_value=mock_server),
        patch("proxy.router._get_job_scheduler", return_value=None),
        patch("proxy.router._is_self_healing_active", return_value=False),
        patch("proxy.router._check_slot_availability", return_value=None),
        patch("proxy.router._increment_active_queries"),
        patch("proxy.router._decrement_active_queries") as mock_decrement,
        patch("proxy.router._handle_session") as mock_session,
        patch("proxy.router._build_slot_context", return_value=(None, None, 3.0)),
        patch("proxy.router._call_with_backend_retries", new_callable=AsyncMock) as mock_retry,
        patch("proxy.router.httpx.AsyncClient") as mock_client_cls,
        patch("proxy.router._schedule_token_increment"),
        patch("proxy.router._schedule_recv_token_increment"),
        patch("proxy.router.count_text_tokens", return_value=5),
        patch("proxy.router._extract_delta_text_from_sse_chunk", return_value=""),
        patch("proxy.router.evaluate_stream_guardrail", return_value=None),
        patch("proxy.router.log_request"),
        patch("proxy.router.log_response_chunk"),
    ):
        mock_session.return_value = {
            "session_id": "test-session-123",
            "session_created": True,
            "is_delta_request": False,
            "session_fallback_reason": None,
            "delta_messages": None,
            "body_override": None,
            "body_json": {"messages": [], "model": "test-model", "stream": True},
            "original_message_count": 0,
        }

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_resp
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client
        mock_retry.return_value = (mock_cm, mock_resp)

        mock_req.body = AsyncMock(return_value=json.dumps(
            {"messages": [], "model": "test-model", "stream": True}
        ).encode())

        result = await proxy_to_local(mock_req, "v1/chat/completions")
        assert isinstance(result, StreamingResponse), (
            f"Expected StreamingResponse, got {type(result).__name__}"
        )

        collected = b""
        async for chunk in result.body_iterator:
            collected += chunk

        assert mock_decrement.called, (
            "_decrement_active_queries should have been called on cleanup"
        )
        assert mock_client.aclose.called, (
            "httpx client aclose should be called on cleanup"
        )
        assert mock_cm.__aexit__.called, (
            "stream context manager __aexit__ should be called on cleanup"
        )


@pytest.mark.asyncio
async def test_router_stream_disconnect_active_queries_not_leaked(mock_server):
    """Active queries counter does not leak on disconnect (AC5)."""
    from proxy.router import proxy_to_local

    mock_req = _make_mock_request(is_disconnected=True)
    session_id = "test-session-456"

    chunks = [b'data: test\n\n']

    mock_resp = _make_mock_stream_response(chunks)

    with (
        patch("proxy.router._srv", return_value=mock_server),
        patch("proxy.router._get_job_scheduler", return_value=None),
        patch("proxy.router._is_self_healing_active", return_value=False),
        patch("proxy.router._check_slot_availability", return_value=None),
        patch("proxy.router._increment_active_queries"),
        patch("proxy.router._handle_session") as mock_session,
        patch("proxy.router._build_slot_context", return_value=(None, None, 3.0)),
        patch("proxy.router._call_with_backend_retries", new_callable=AsyncMock) as mock_retry,
        patch("proxy.router.httpx.AsyncClient") as mock_client_cls,
        patch("proxy.router._schedule_token_increment"),
        patch("proxy.router._schedule_recv_token_increment"),
        patch("proxy.router.count_text_tokens", return_value=5),
        patch("proxy.router._extract_delta_text_from_sse_chunk", return_value=""),
        patch("proxy.router.evaluate_stream_guardrail", return_value=None),
        patch("proxy.router.log_request"),
        patch("proxy.router.log_response_chunk"),
        patch("proxy.router._decrement_active_queries") as mock_decrement,
    ):
        mock_session.return_value = {
            "session_id": session_id,
            "session_created": True,
            "is_delta_request": False,
            "session_fallback_reason": None,
            "delta_messages": None,
            "body_override": None,
            "body_json": {"messages": [], "model": "test-model", "stream": True},
            "original_message_count": 0,
        }

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_resp
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client
        mock_retry.return_value = (mock_cm, mock_resp)

        mock_req.body = AsyncMock(return_value=json.dumps(
            {"messages": [], "model": "test-model", "stream": True}
        ).encode())

        result = await proxy_to_local(mock_req, "v1/chat/completions")
        assert isinstance(result, StreamingResponse), (
            f"Expected StreamingResponse, got {type(result).__name__}"
        )
        async for chunk in result.body_iterator:
            pass

        assert mock_decrement.called, (
            "active_queries must be decremented on disconnect"
        )


@pytest.mark.asyncio
async def test_router_stream_disconnect_releases_scheduler_slot(mock_server):
    """On disconnect, scheduler slot is released (AC4)."""
    from proxy.router import proxy_to_local

    mock_req = _make_mock_request(is_disconnected=True)
    session_id = "test-session-789"

    # Need 11+ chunks to trigger disconnect check (check every 10 iterations)
    chunks = [b'data: test\n\n'] * 15

    mock_resp = _make_mock_stream_response(chunks)
    mock_scheduler = MagicMock()

    with (
        patch("proxy.router._srv", return_value=mock_server),
        patch("proxy.router._get_job_scheduler", return_value=mock_scheduler),
        patch("proxy.router._is_self_healing_active", return_value=False),
        patch("proxy.router._check_slot_availability", return_value=None),
        patch("proxy.router._increment_active_queries"),
        patch("proxy.router._decrement_active_queries"),
        patch("proxy.router._handle_session") as mock_session,
        patch("proxy.router._build_slot_context", return_value=(None, None, 3.0)),
        patch("proxy.router._call_with_backend_retries", new_callable=AsyncMock) as mock_retry,
        patch("proxy.router.httpx.AsyncClient") as mock_client_cls,
        patch("proxy.router._schedule_token_increment"),
        patch("proxy.router._schedule_recv_token_increment"),
        patch("proxy.router.count_text_tokens", return_value=5),
        patch("proxy.router._extract_delta_text_from_sse_chunk", return_value=""),
        patch("proxy.router.evaluate_stream_guardrail", return_value=None),
        patch("proxy.router.log_request"),
        patch("proxy.router.log_response_chunk"),
    ):
        mock_session.return_value = {
            "session_id": session_id,
            "session_created": True,
            "is_delta_request": False,
            "session_fallback_reason": None,
            "delta_messages": None,
            "body_override": None,
            "body_json": {"messages": [], "model": "test-model", "stream": True},
            "original_message_count": 0,
        }

        # Mock scheduler admit_job to return ASSIGNED
        result = MagicMock()
        result.kind = "ASSIGNED"
        result.slot_id = 0
        mock_scheduler.admit_job = AsyncMock(return_value=result)
        mock_scheduler.reenter_job = AsyncMock(return_value=None)
        mock_scheduler.remove_job = AsyncMock()
        mock_scheduler.mark_request_start = AsyncMock()
        mock_scheduler.mark_request_end = AsyncMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_resp
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client
        mock_retry.return_value = (mock_cm, mock_resp)

        mock_req.body = AsyncMock(return_value=json.dumps(
            {"messages": [], "model": "test-model", "stream": True}
        ).encode())

        result = await proxy_to_local(mock_req, "v1/chat/completions")
        assert isinstance(result, StreamingResponse), (
            f"Expected StreamingResponse, got {type(result).__name__}"
        )
        async for chunk in result.body_iterator:
            pass

        assert mock_scheduler.remove_job.called, (
            "scheduler.remove_job should be called on client disconnect"
        )


@pytest.mark.asyncio
async def test_stream_disconnect_with_scheduler_cleanup_integration(mock_server):
    """Integration: streaming disconnect triggers scheduler.remove_job."""
    from proxy.router import proxy_to_local

    mock_req = _make_mock_request(is_disconnected=True)
    session_id = "test-integration-001"

    # Need 11+ chunks to trigger disconnect check (check every 10 iterations)
    chunks = [b'data: test\n\n'] * 15

    mock_resp = _make_mock_stream_response(chunks)
    mock_scheduler = MagicMock()

    with (
        patch("proxy.router._srv", return_value=mock_server),
        patch("proxy.router._get_job_scheduler", return_value=mock_scheduler),
        patch("proxy.router._is_self_healing_active", return_value=False),
        patch("proxy.router._check_slot_availability", return_value=None),
        patch("proxy.router._increment_active_queries"),
        patch("proxy.router._decrement_active_queries"),
        patch("proxy.router._handle_session") as mock_session,
        patch("proxy.router._build_slot_context", return_value=(None, None, 3.0)),
        patch("proxy.router._call_with_backend_retries", new_callable=AsyncMock) as mock_retry,
        patch("proxy.router.httpx.AsyncClient") as mock_client_cls,
        patch("proxy.router._schedule_token_increment"),
        patch("proxy.router._schedule_recv_token_increment"),
        patch("proxy.router.count_text_tokens", return_value=5),
        patch("proxy.router._extract_delta_text_from_sse_chunk", return_value=""),
        patch("proxy.router.evaluate_stream_guardrail", return_value=None),
        patch("proxy.router.log_request"),
        patch("proxy.router.log_response_chunk"),
    ):
        mock_session.return_value = {
            "session_id": session_id,
            "session_created": False,
            "is_delta_request": False,
            "session_fallback_reason": None,
            "delta_messages": None,
            "body_override": None,
            "body_json": {"messages": [], "model": "test-model", "stream": True},
            "original_message_count": 0,
        }

        result = MagicMock()
        result.kind = "ASSIGNED"
        result.slot_id = 0
        mock_scheduler.admit_job = AsyncMock(return_value=result)
        mock_scheduler.reenter_job = AsyncMock(return_value=None)
        mock_scheduler.remove_job = AsyncMock()
        mock_scheduler.mark_request_start = AsyncMock()
        mock_scheduler.mark_request_end = AsyncMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_resp
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client
        mock_retry.return_value = (mock_cm, mock_resp)

        mock_req.body = AsyncMock(return_value=json.dumps(
            {"messages": [], "model": "test-model", "stream": True}
        ).encode())

        result = await proxy_to_local(mock_req, "v1/chat/completions")
        assert isinstance(result, StreamingResponse), (
            f"Expected StreamingResponse, got {type(result).__name__}"
        )
        async for chunk in result.body_iterator:
            pass

        mock_scheduler.remove_job.assert_called_once_with(session_id)


# ===================================================================
# Non-streaming disconnect tests
# ===================================================================

@pytest.mark.asyncio
async def test_non_streaming_disconnect_before_request(mock_server):
    """Non-streaming path detects disconnect before sending to backend."""
    from proxy.router import proxy_to_local

    mock_req = _make_mock_request(is_disconnected=True)
    mock_req.body = AsyncMock(return_value=json.dumps(
        {"messages": [{"role": "user", "content": "Hello"}], "model": "test-model"}
    ).encode())

    with (
        patch("proxy.router._srv", return_value=mock_server),
        patch("proxy.router._get_job_scheduler", return_value=None),
        patch("proxy.router._is_self_healing_active", return_value=False),
        patch("proxy.router._check_slot_availability", return_value=None),
        patch("proxy.router._increment_active_queries"),
        patch("proxy.router._decrement_active_queries"),
        patch("proxy.router._handle_session") as mock_session,
        patch("proxy.router._build_slot_context", return_value=(None, None, 3.0)),
        patch("proxy.router._call_with_backend_retries", new_callable=AsyncMock),
        patch("proxy.router._call_with_empty_retry", new_callable=AsyncMock) as mock_empty_retry,
        patch("proxy.router._estimate_tokens_sent", return_value=10),
        patch("proxy.router._schedule_token_increment"),
        patch("proxy.router._schedule_recv_token_increment"),
        patch("proxy.router.log_request"),
        patch("proxy.router.log_response"),
    ):
        mock_resp_for_nonstream = MagicMock()
        mock_resp_for_nonstream.status_code = 200
        mock_resp_for_nonstream.headers = {"content-type": "application/json"}
        mock_resp_for_nonstream.content = b'{"choices": [{"message": {"content": "Hi"}}]}'
        mock_empty_retry.return_value = mock_resp_for_nonstream

        mock_session.return_value = {
            "session_id": "test-session-ns",
            "session_created": False,
            "is_delta_request": False,
            "session_fallback_reason": None,
            "delta_messages": None,
            "body_override": None,
            "body_json": {"messages": [{"role": "user", "content": "Hello"}], "model": "test-model"},
            "original_message_count": 1,
        }

        result = await proxy_to_local(mock_req, "v1/chat/completions")

        assert result is not None
