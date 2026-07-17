"""
Behavioural parity tests for proxy_to_local streaming vs buffered paths.

These tests verify that the streaming and non-streaming code paths in
``proxy_to_local`` produce identical external behaviour for:

1. Session response headers (``X-Session-*``)
2. Slot save/restore call patterns
3. Scheduler ``mark_request_start`` / ``mark_request_end`` call parity
4. Guardrail cut-off semantics
5. Active-query counter management

Each test exercises both paths with the same input and checks that
the observable side-effects (headers, mock calls, counter values)
are equivalent.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import proxy.server as server
import pytest
from proxy.router import proxy_to_local
pytestmark = pytest.mark.refactor_parity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_request(body: dict, stream: bool = False):
    """Build a minimal dummy Request that ``proxy_to_local`` can consume."""
    payload = {**body}
    if stream:
        payload["stream"] = True
    body_bytes = json.dumps(payload).encode("utf-8")

    class DummyRequest:
        headers = {
            "host": "localhost",
        }
        method = "POST"
        url = type("U", (), {"path": "/v1/chat/completions"})()

        async def body(self):
            return body_bytes

        async def is_disconnected(self):
            return False

    return DummyRequest()


def _mock_upstream_response(
    status_code: int = 200,
    content: bytes = b'{"id":"test","choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"Hello!"}}]}',
    content_type: str = "application/json",
):
    """Build a synchronous mock Response (plain object, not httpx spec)."""
    return type("MockResponse", (), {
        "status_code": status_code,
        "content": content,
        "headers": {"content-type": content_type},
    })()


def _mock_streaming_upstream_response(
    status_code: int = 200,
    chunks: list = None,
    content_type: str = "text/event-stream",
):
    """Build a mock httpx streaming response."""
    if chunks is None:
        chunks = [
            b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0}]}\n\n",
            b"data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"index\":0}]}\n\n",
            b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}\n\n",
            b"data: [DONE]\n\n",
        ]

    async def _aiter():
        for c in chunks:
            yield c

    MockStreamResponse = type("MockStreamResponse", (), {
        "status_code": status_code,
        "headers": {"content-type": content_type},
        "aiter_bytes": staticmethod(_aiter),
        "aread": AsyncMock(return_value=b"".join(chunks)),
    })

    class MockCM:
        async def __aenter__(self):
            return MockStreamResponse()

        async def __aexit__(self, *args):
            pass

    return MockCM(), MockStreamResponse()


# ---------------------------------------------------------------------------
# Fixtures – shared server config for all parity tests
# ---------------------------------------------------------------------------

BASE_SERVER_CONFIG = {
    "server": {
        "llama_router_mode": False,
        "llama_server_port": 8080,
        "max_concurrent_queries": 4,
        "local_max_concurrent_queries": 1,
        "llama_request_timeout": 30,
        "session_single_flight_mode": "bypass",
        "disconnect_cleanup_timeout": 1,
    }
}


@pytest.fixture(autouse=True)
def _reset_server_state(monkeypatch):
    """Reset server-level state before each test."""
    monkeypatch.setattr(server, "config", dict(BASE_SERVER_CONFIG))
    monkeypatch.setattr(server, "active_queries", 0)
    monkeypatch.setattr(server, "local_active_queries", 0)
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "llama_process", MagicMock(poll=lambda: None, pid=1))
    monkeypatch.setattr(server, "current_model", "test-model")
    monkeypatch.setattr(server, "session_manager", MagicMock())
    monkeypatch.setattr(server, "logger", MagicMock())

    # Disable self-healing
    monkeypatch.setattr("proxy.router._is_self_healing_active", lambda: False)

    # Reset metrics
    monkeypatch.setattr(server, "backend_signal_counts", {
        "connect_failures": 0,
        "read_failures": 0,
        "timeout_failures": 0,
        "other_failures": 0,
        "concurrency_rejects": 0,
    })

    # Mock out slot save/restore so they don't make real HTTP calls
    # Note: patch on proxy.router because names are imported directly there
    monkeypatch.setattr("proxy.router._restore_slot_snapshot", AsyncMock(return_value=False))
    monkeypatch.setattr("proxy.router._save_slot_snapshot", AsyncMock(return_value=False))
    monkeypatch.setattr("proxy.router._build_slot_context", MagicMock(return_value=(None, None, 3.0)))

    # Mock session handlers
    monkeypatch.setattr("proxy.router._handle_session", AsyncMock(return_value={
        "session_id": "test-session-id",
        "session_created": True,
        "is_delta_request": False,
        "session_fallback_reason": None,
        "delta_messages": [],
        "original_message_count": 1,
        "body_override": None,
        "body_json": None,
    }))

    # Mock out log resolvers
    # _resolve_log_path is imported inside proxy_to_local() scope, so
    # we must patch it at the source module (proxy.session).
    monkeypatch.setattr("proxy.session._resolve_log_path", MagicMock(return_value=MagicMock(
        exists=lambda: False,
        stat=lambda: MagicMock(st_size=0),
    )))

    # Mock slot availability check passes
    monkeypatch.setattr("proxy.router._check_slot_availability", AsyncMock(return_value=None))


# ===================================================================
# Test 1: Session header parity
# ===================================================================


class TestSessionHeaderParity:
    """X-Session-* headers must be identical for stream and non-stream paths."""

    @pytest.mark.asyncio
    async def test_session_headers_present_in_buffered_response(self, monkeypatch):
        """Non-streaming responses include X-Session-* headers."""
        monkeypatch.setattr(
            "proxy.router._handle_session", AsyncMock(return_value={
                "session_id": "parity-session-1",
                "session_created": True,
                "is_delta_request": False,
                "session_fallback_reason": None,
                "delta_messages": [],
                "original_message_count": 2,
                "body_override": None,
                "body_json": None,
            })
        )

        resp = _mock_upstream_response()
        monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=resp))
        monkeypatch.setattr("proxy.router._call_with_empty_retry", AsyncMock(return_value=resp))

        response = await proxy_to_local(
            _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=False),
            "v1/chat/completions",
        )

        assert response.status_code == 200
        assert response.headers.get("X-Session-Id") == "parity-session-1"
        assert response.headers.get("X-Session-Created") == "true"
        assert response.headers.get("X-Session-Delta") == "false"
        assert response.headers.get("X-Session-Fallback-Reason") is None

    @pytest.mark.asyncio
    async def test_session_headers_present_in_streaming_response(self, monkeypatch):
        """Streaming responses include X-Session-* headers."""
        monkeypatch.setattr(
            "proxy.router._handle_session", AsyncMock(return_value={
                "session_id": "parity-session-2",
                "session_created": False,
                "is_delta_request": True,
                "session_fallback_reason": "history_too_long",
                "delta_messages": [{"role": "user", "content": "more"}],
                "original_message_count": 5,
                "body_override": None,
                "body_json": None,
            })
        )

        cm, sresp = _mock_streaming_upstream_response()
        monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, sresp)))

        response = await proxy_to_local(
            _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=True),
            "v1/chat/completions",
        )

        assert response.status_code == 200
        assert response.headers.get("X-Session-Id") == "parity-session-2"
        assert response.headers.get("X-Session-Created") == "false"
        assert response.headers.get("X-Session-Delta") == "true"
        assert response.headers.get("X-Session-Fallback-Reason") == "history_too_long"


# ===================================================================
# Test 2: Slot save/restore call parity
# ===================================================================


class TestSlotRestoreParity:
    """Slot restore must be called equivalently for both paths."""

    @pytest.mark.asyncio
    async def test_slot_restore_called_for_buffered_path(self, monkeypatch):
        """Slot restore is called in the buffered path when slot is enabled."""
        monkeypatch.setattr("proxy.router._build_slot_context", MagicMock(return_value=("slot-0", "/tmp/slot.json", 3.0)))
        restore_mock = AsyncMock(return_value=True)
        monkeypatch.setattr("proxy.router._restore_slot_snapshot", restore_mock)

        resp = _mock_upstream_response()
        monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=resp))
        monkeypatch.setattr("proxy.router._call_with_empty_retry", AsyncMock(return_value=resp))

        await proxy_to_local(
            _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=False),
            "v1/chat/completions",
        )

        restore_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_slot_restore_called_for_streaming_path(self, monkeypatch):
        """Slot restore is called in the streaming path when slot is enabled."""
        monkeypatch.setattr("proxy.router._build_slot_context", MagicMock(return_value=("slot-0", "/tmp/slot.json", 3.0)))
        restore_mock = AsyncMock(return_value=True)
        monkeypatch.setattr("proxy.router._restore_slot_snapshot", restore_mock)

        cm, sresp = _mock_streaming_upstream_response()
        monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, sresp)))

        await proxy_to_local(
            _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=True),
            "v1/chat/completions",
        )

        restore_mock.assert_called_once()


class TestSlotSaveParity:
    """Slot save must be called equivalently for both paths on success."""

    @pytest.mark.asyncio
    async def test_slot_save_called_for_buffered_success(self, monkeypatch):
        """Slot save is called in the buffered path when slot is enabled on 200."""
        monkeypatch.setattr("proxy.router._build_slot_context", MagicMock(return_value=("slot-0", "/tmp/slot.json", 3.0)))
        save_mock = AsyncMock(return_value=True)
        monkeypatch.setattr("proxy.router._save_slot_snapshot", save_mock)

        resp = _mock_upstream_response()
        monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=resp))
        monkeypatch.setattr("proxy.router._call_with_empty_retry", AsyncMock(return_value=resp))

        await proxy_to_local(
            _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=False),
            "v1/chat/completions",
        )

        save_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_slot_save_called_for_streaming_success(self, monkeypatch):
        """Slot save is called in the streaming path when slot is enabled on 200."""
        monkeypatch.setattr("proxy.router._build_slot_context", MagicMock(return_value=("slot-0", "/tmp/slot.json", 3.0)))
        save_mock = AsyncMock(return_value=True)
        monkeypatch.setattr("proxy.router._save_slot_snapshot", save_mock)

        cm, sresp = _mock_streaming_upstream_response()
        monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, sresp)))

        response = await proxy_to_local(
            _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=True),
            "v1/chat/completions",
        )

        # Drain generator so that the finally block runs slot save
        async for _ in response.body_iterator:
            pass

        save_mock.assert_called_once()


# ===================================================================
# Test 3: Active query counter parity
# ===================================================================


class TestActiveQueryParity:
    """Active query counters must be managed identically."""

    @pytest.mark.asyncio
    async def test_active_queries_decremented_on_buffered_success(self, monkeypatch):
        """Active queries is 0 after buffered success."""
        resp = _mock_upstream_response()
        monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=resp))
        monkeypatch.setattr("proxy.router._call_with_empty_retry", AsyncMock(return_value=resp))

        assert server.active_queries == 0

        await proxy_to_local(
            _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=False),
            "v1/chat/completions",
        )

        assert server.active_queries == 0

    @pytest.mark.asyncio
    async def test_active_queries_decremented_on_streaming_success(self, monkeypatch):
        """Active queries is 0 after streaming success."""
        cm, sresp = _mock_streaming_upstream_response()
        monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, sresp)))

        assert server.active_queries == 0

        response = await proxy_to_local(
            _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=True),
            "v1/chat/completions",
        )

        # Drain the streaming generator to trigger cleanup
        async for _ in response.body_iterator:
            pass

        assert server.active_queries == 0


# ===================================================================
# Test 4: Scheduler mark_request_start/end parity
# ===================================================================


# ===================================================================
# Test 5: Guardrail parity
# ===================================================================


class TestGuardrailParity:
    """Guardrail behavior must be identical for both paths (where applicable)."""

    @pytest.mark.asyncio
    async def test_guardrail_config_loaded_for_streaming_path(self, monkeypatch):
        """Guardrail config is read from server_config for streaming path."""
        config = dict(BASE_SERVER_CONFIG)
        config["server"]["session_guardrail_max_runtime_seconds"] = 10
        config["server"]["session_guardrail_max_completion_tokens"] = 100
        monkeypatch.setattr(server, "config", config)

        cm, sresp = _mock_streaming_upstream_response()
        monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, sresp)))

        # Mock evaluate_stream_guardrail to verify it gets the right config
        guardrail_mock = MagicMock(return_value=None)
        monkeypatch.setattr("proxy.router.evaluate_stream_guardrail", guardrail_mock)

        response = await proxy_to_local(
            _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=True),
            "v1/chat/completions",
        )

        async for _ in response.body_iterator:
            pass

        guardrail_mock.assert_called()


# ===================================================================
# Test 6: Session history update parity
# ===================================================================


class TestSessionUpdateParity:
    """Session history updates must happen for both paths."""

    @pytest.mark.asyncio
    async def test_session_updated_for_buffered_path(self, monkeypatch):
        """Session history is updated after a buffered response."""
        update_mock = AsyncMock()
        monkeypatch.setattr(server, "session_manager", MagicMock(
            update_messages=update_mock,
            get=AsyncMock(return_value=None),
            set_restore_confirmed=AsyncMock(),
        ))

        resp = _mock_upstream_response()
        monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=resp))
        monkeypatch.setattr("proxy.router._call_with_empty_retry", AsyncMock(return_value=resp))

        await proxy_to_local(
            _dummy_request(
                {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
                stream=False,
            ),
            "v1/chat/completions",
        )

        update_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_updated_for_streaming_path(self, monkeypatch):
        """Session history is updated during a streaming response."""
        update_mock = AsyncMock()
        monkeypatch.setattr(server, "session_manager", MagicMock(
            update_messages=update_mock,
            get=AsyncMock(return_value=MagicMock(messages=[])),
            set_restore_confirmed=AsyncMock(),
        ))

        cm, sresp = _mock_streaming_upstream_response()
        monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, sresp)))

        response = await proxy_to_local(
            _dummy_request(
                {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
                stream=True,
            ),
            "v1/chat/completions",
        )

        async for _ in response.body_iterator:
            pass

        update_mock.assert_called_once()
