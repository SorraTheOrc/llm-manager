"""Tests for log_request enhancement (LP-0MQQSM1V7004QOGL).

Covers:
- Logging with a session_id present
- Logging when no session_id is resolved
- Slot_id present and absent
- Body preview with system prompt excluded/redacted
"""

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_request(method="POST", url="http://localhost:8080/v1/chat/completions"):
    """Build a minimal mock FastAPI Request for log_request tests."""
    mock_request = MagicMock()
    mock_request.method = method
    mock_request.url = MagicMock()
    mock_request.url.__str__ = MagicMock(return_value=url)
    mock_request.headers = {}
    return mock_request


def _build_body(messages):
    """Build a JSON request body (bytes)."""
    return json.dumps({"messages": messages}).encode("utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_srv():
    """Mock the server module so log_request calls don't hit real I/O."""
    with patch("proxy.router_helpers._srv") as mock_srv:
        mock_logger = MagicMock()
        mock_srv.return_value.logger = mock_logger
        yield mock_srv, mock_logger


# ---------------------------------------------------------------------------
# Tests — session_id
# ---------------------------------------------------------------------------


class TestLogRequestSessionId:
    """Acceptance criteria: INFO log line includes the resolved session ID."""

    def test_log_request_includes_session_id_when_present(self, _mock_srv):
        """When session_id is provided, it appears in the log line."""
        from proxy.router_helpers import log_request

        mock_srv, mock_logger = _mock_srv
        request = _make_mock_request()
        body = _build_body([{"role": "user", "content": "Hello"}])

        log_request(request, body, "local", session_id="sess-abc-123")

        # Verify info() was called with session_id in the message
        call_args = mock_logger.info.call_args
        assert call_args is not None
        message = call_args[0][0]
        assert "session_id=sess-abc-123" in message

    def test_log_request_excludes_session_id_field_when_none(self, _mock_srv):
        """When session_id is None (default), no session_id= field appears."""
        from proxy.router_helpers import log_request

        mock_srv, mock_logger = _mock_srv
        request = _make_mock_request()
        body = _build_body([{"role": "user", "content": "Hello"}])

        log_request(request, body, "local")

        call_args = mock_logger.info.call_args
        assert call_args is not None
        message = call_args[0][0]
        assert "session_id=" not in message

    def test_log_request_remote_includes_session_id(self, _mock_srv):
        """Remote path should also include session_id when provided."""
        from proxy.router_helpers import log_request

        mock_srv, mock_logger = _mock_srv
        request = _make_mock_request()
        body = _build_body([{"role": "user", "content": "Hello"}])

        log_request(request, body, "remote", endpoint="http://remote.example.com", session_id="sess-remote-456")

        call_args = mock_logger.info.call_args
        assert call_args is not None
        message = call_args[0][0]
        assert "session_id=sess-remote-456" in message
        assert "http://remote.example.com" in message


# ---------------------------------------------------------------------------
# Tests — slot_id
# ---------------------------------------------------------------------------


class TestLogRequestSlotId:
    """Acceptance criteria: INFO log line includes assigned slot ID or placeholder."""

    def test_log_request_includes_slot_id_when_present(self, _mock_srv):
        """When slot_id is provided, it appears in the log line."""
        from proxy.router_helpers import log_request

        mock_srv, mock_logger = _mock_srv
        request = _make_mock_request()
        body = _build_body([{"role": "user", "content": "Hello"}])

        log_request(request, body, "local", slot_id="slot-7")

        call_args = mock_logger.info.call_args
        assert call_args is not None
        message = call_args[0][0]
        assert "slot=slot-7" in message

    def test_log_request_slot_none_when_not_provided(self, _mock_srv):
        """When slot_id is not provided, slot=none appears."""
        from proxy.router_helpers import log_request

        mock_srv, mock_logger = _mock_srv
        request = _make_mock_request()
        body = _build_body([{"role": "user", "content": "Hello"}])

        log_request(request, body, "local")

        call_args = mock_logger.info.call_args
        assert call_args is not None
        message = call_args[0][0]
        assert "slot=none" in message

    def test_log_request_slot_queued_placeholder(self, _mock_srv):
        """When slot_id is 'queued', log slot=queued."""
        from proxy.router_helpers import log_request

        mock_srv, mock_logger = _mock_srv
        request = _make_mock_request()
        body = _build_body([{"role": "user", "content": "Hello"}])

        log_request(request, body, "local", slot_id="queued")

        call_args = mock_logger.info.call_args
        assert call_args is not None
        message = call_args[0][0]
        assert "slot=queued" in message


# ---------------------------------------------------------------------------
# Tests — body preview (system prompt exclusion)
# ---------------------------------------------------------------------------


class TestLogRequestBodyPreview:
    """Acceptance criteria: body preview excludes redacted system prompt content."""

    def test_log_request_excludes_system_prompt_content(self, _mock_srv):
        """System message content should not appear in the body preview."""
        from proxy.router_helpers import log_request

        mock_srv, mock_logger = _mock_srv
        request = _make_mock_request()
        body = _build_body([
            {"role": "system", "content": "You are a helpful assistant. Do not leak this."},
            {"role": "user", "content": "Tell me a joke"},
        ])

        log_request(request, body, "local")

        call_args = mock_logger.info.call_args
        assert call_args is not None
        message = call_args[0][0]
        # System prompt content should be excluded
        assert "helpful assistant" not in message
        assert "Do not leak this" not in message
        # User content should be present
        assert "Tell me a joke" in message

    def test_log_request_with_only_system_messages(self, _mock_srv):
        """When body contains only system messages, preview should not contain their content."""
        from proxy.router_helpers import log_request

        mock_srv, mock_logger = _mock_srv
        request = _make_mock_request()
        body = _build_body([
            {"role": "system", "content": "Secret system instructions"},
        ])

        log_request(request, body, "local")

        call_args = mock_logger.info.call_args
        assert call_args is not None
        message = call_args[0][0]
        assert "Secret system instructions" not in message

    def test_log_request_preserves_user_content_with_system_messages(self, _mock_srv):
        """Body preview should include user message content even when system messages are present."""
        from proxy.router_helpers import log_request

        mock_srv, mock_logger = _mock_srv
        request = _make_mock_request()
        body = _build_body([
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris"},
        ])

        log_request(request, body, "local")

        call_args = mock_logger.info.call_args
        assert call_args is not None
        message = call_args[0][0]
        # User and assistant content should be present
        assert "capital of France" in message
        assert "Paris" in message
        # System content should be excluded
        assert "Be helpful" not in message

    def test_log_request_handles_non_list_messages(self, _mock_srv):
        """When messages is not a list, body preview should still work."""
        from proxy.router_helpers import log_request

        mock_srv, mock_logger = _mock_srv
        request = _make_mock_request()
        body = _build_body("not a list")

        log_request(request, body, "local")

        # Should not raise
        mock_srv.return_value.logger.info.assert_called_once()


# ---------------------------------------------------------------------------
# Tests — backward compatibility
# ---------------------------------------------------------------------------


class TestLogRequestBackwardCompatibility:
    """Ensure the enhanced log_request is backward compatible with existing callers."""

    def test_log_request_works_without_new_params(self, _mock_srv):
        """Calling log_request without session_id/slot_id should still work (existing callers)."""
        from proxy.router_helpers import log_request

        mock_srv, mock_logger = _mock_srv
        request = _make_mock_request()
        body = _build_body([{"role": "user", "content": "Hello"}])

        # Call with only the original 3 required args (plus endpoint for remote)
        log_request(request, body, "local")

        mock_srv.return_value.logger.info.assert_called_once()

    def test_log_request_remote_signature_compat(self, _mock_srv):
        """Remote caller signature (4 args) should still work."""
        from proxy.router_helpers import log_request

        mock_srv, mock_logger = _mock_srv
        request = _make_mock_request()
        body = _build_body([{"role": "user", "content": "Hello"}])

        log_request(request, body, "remote", "http://remote.example.com")

        mock_srv.return_value.logger.info.assert_called_once()
