"""Tests for slot endpoint logging improvements in _call_slot_endpoint.

Verifies that:
- Exception type names are logged in warning messages (not empty errors)
- Non-200 HTTP responses are logged at WARNING with status code and truncated body
- Debug-level logs with exc_info=True are emitted for exceptions
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.refactor_parity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_srv(logger: logging.Logger):
    """Build a minimal mock server object with a logger and optional http_client."""
    srv = MagicMock()
    srv.logger = logger
    srv._http_client = None
    return srv


# ---------------------------------------------------------------------------
# Exception type name is logged
# ---------------------------------------------------------------------------

class TestSlotEndpointExceptionLogging:
    @pytest.mark.asyncio
    async def test_logs_exception_type_on_failure(self, caplog):
        """AC1: Exception type name appears in warning message — no empty error."""
        from proxy.session import _call_slot_endpoint

        caplog.set_level(logging.WARNING)

        mock_srv = _make_mock_srv(logging.getLogger("test_logger"))
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ValueError("something broke"))
        mock_srv._http_client = mock_client

        with patch("proxy.session._srv", return_value=mock_srv):
            result = await _call_slot_endpoint(
                1234, 0, "save", "/tmp/test.bin", timeout=1.0
            )

        assert result is False
        # Must contain the exception type name, not an empty error
        assert len(caplog.records) >= 1
        warning_record = caplog.records[0]
        assert warning_record.levelname == "WARNING"
        assert "ValueError" in warning_record.message
        assert "something broke" in warning_record.message
        assert "slot_save failed" in warning_record.message
        assert "slot=0" in warning_record.message

    @pytest.mark.asyncio
    async def test_logs_exception_type_with_empty_str(self, caplog):
        """AC1: When exc.__str__() is empty, the type name provides context."""
        from proxy.session import _call_slot_endpoint

        caplog.set_level(logging.WARNING)

        # An exception whose string representation is empty
        class EmptyStrError(Exception):
            def __str__(self):
                return ""

        mock_srv = _make_mock_srv(logging.getLogger("test_logger"))
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=EmptyStrError("detail"))
        mock_srv._http_client = mock_client

        with patch("proxy.session._srv", return_value=mock_srv):
            result = await _call_slot_endpoint(
                1234, 3, "restore", "/tmp/test.bin", timeout=1.0
            )

        assert result is False
        warning_record = caplog.records[0]
        assert "EmptyStrError" in warning_record.message
        # Even with empty __str__, the error field must not be empty
        assert "error=EmptyStrError" in warning_record.message

    @pytest.mark.asyncio
    async def test_logs_debug_with_exc_info_on_failure(self, caplog):
        """AC3: Debug-level log with exc_info=True is emitted for every exception."""
        from proxy.session import _call_slot_endpoint

        caplog.set_level(logging.DEBUG)

        mock_srv = _make_mock_srv(logging.getLogger("test_logger"))
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=TimeoutError("connection timed out"))
        mock_srv._http_client = mock_client

        with patch("proxy.session._srv", return_value=mock_srv):
            result = await _call_slot_endpoint(
                1234, 5, "save", "/tmp/test.bin", timeout=1.0
            )

        assert result is False
        # Find the debug record with exc_info
        debug_records = [r for r in caplog.records if r.levelname == "DEBUG"]
        assert len(debug_records) >= 1
        # Verify exc_info is set (it will be a tuple if True was passed)
        assert debug_records[0].exc_info is not None
        assert debug_records[0].exc_info[0] is TimeoutError


# ---------------------------------------------------------------------------
# Non-200 HTTP responses are logged
# ---------------------------------------------------------------------------

class TestSlotEndpointNon200Logging:
    @pytest.mark.asyncio
    async def test_logs_warning_for_non_200_response(self, caplog):
        """AC2: Non-200 status code is logged at WARNING level."""
        from proxy.session import _call_slot_endpoint

        caplog.set_level(logging.WARNING)

        response = MagicMock()
        response.status_code = 500
        # Simulate a response body
        response.text = "Internal Server Error: cannot process slot"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=response)

        mock_srv = _make_mock_srv(logging.getLogger("test_logger"))
        mock_srv._http_client = mock_client

        with patch("proxy.session._srv", return_value=mock_srv):
            result = await _call_slot_endpoint(
                1234, 2, "save", "/tmp/test.bin", timeout=1.0
            )

        assert result is False
        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) >= 1
        assert "500" in warning_records[0].message
        assert "slot_save failed" in warning_records[0].message

    @pytest.mark.asyncio
    async def test_logs_truncated_body_for_long_response(self, caplog):
        """AC2: Response body is truncated to 500 chars in warning."""
        from proxy.session import _call_slot_endpoint

        caplog.set_level(logging.WARNING)

        response = MagicMock()
        response.status_code = 400
        # Create a very long body
        response.text = "x" * 2000

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=response)

        mock_srv = _make_mock_srv(logging.getLogger("test_logger"))
        mock_srv._http_client = mock_client

        with patch("proxy.session._srv", return_value=mock_srv):
            result = await _call_slot_endpoint(
                1234, 7, "save", "/tmp/test.bin", timeout=1.0
            )

        assert result is False
        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) >= 1
        message = warning_records[0].message

        # Verify truncation: the body in the message should be <= 503 chars (500 + "...")
        # Extract the body portion after "body="
        assert "body=" in message
        # Check it's not the full 2000 chars
        assert "xxxxx" in message
        # The full 2000 chars should NOT appear
        assert "x" * 501 not in message  # Should be truncated

    @pytest.mark.asyncio
    async def test_200_response_returns_true_no_warning(self, caplog):
        """AC2: Successful 200 responses should not log a warning."""
        from proxy.session import _call_slot_endpoint

        caplog.set_level(logging.WARNING)

        response = MagicMock()
        response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=response)

        mock_srv = _make_mock_srv(logging.getLogger("test_logger"))
        mock_srv._http_client = mock_client

        with patch("proxy.session._srv", return_value=mock_srv):
            result = await _call_slot_endpoint(
                1234, 1, "save", "/tmp/test.bin", timeout=1.0
            )

        assert result is True
        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestSlotEndpointEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_filename_returns_false(self):
        """When filename is empty/falsy, return False without making a request."""
        from proxy.session import _call_slot_endpoint

        # No mock needed — the early return happens before any HTTP call
        result = await _call_slot_endpoint(
            1234, 0, "save", "", timeout=1.0
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_httpx_exception_logs_type(self, caplog):
        """AC1: httpx-specific exceptions log their type name."""
        from proxy.session import _call_slot_endpoint

        caplog.set_level(logging.WARNING)

        mock_srv = _make_mock_srv(logging.getLogger("test_logger"))
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=OSError("connection refused"))
        mock_srv._http_client = mock_client

        with patch("proxy.session._srv", return_value=mock_srv):
            result = await _call_slot_endpoint(
                1234, 9, "save", "/tmp/test.bin", timeout=1.0
            )

        assert result is False
        warning_record = caplog.records[0]
        assert "OSError" in warning_record.message
