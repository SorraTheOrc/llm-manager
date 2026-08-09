"""Tests for status_request structured logging (client IP + response payload).

Covers the client-IP resolution helper and the KeyValueFormatter that
renders structured ``extra`` fields into the plain-text proxy.log
(LP-0MSK9XXCN0077CMA).
"""
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

pytestmark = pytest.mark.refactor_parity


# ---------------------------------------------------------------------------
# _resolve_client_ip unit tests
# ---------------------------------------------------------------------------


def _make_request(headers=None, client_host="127.0.0.1", has_client=True):
    """Build a minimal mock Request for calling _resolve_client_ip."""
    req = MagicMock()
    req.headers = headers or {}
    if has_client:
        req.client = MagicMock()
        req.client.host = client_host
    else:
        req.client = None
    return req


def test_resolve_client_ip_direct():
    """A direct connection resolves to request.client.host with source 'direct'."""
    from proxy.handlers import _resolve_client_ip

    req = _make_request(headers={}, client_host="192.168.0.191")
    assert _resolve_client_ip(req) == ("192.168.0.191", "direct")


def test_resolve_client_ip_x_forwarded_for():
    """X-Forwarded-For is honored with source 'header'."""
    from proxy.handlers import _resolve_client_ip

    req = _make_request(headers={"x-forwarded-for": "203.0.113.7"})
    assert _resolve_client_ip(req) == ("203.0.113.7", "header")


def test_resolve_client_ip_x_forwarded_for_chain_takes_first():
    """A proxy chain 'client, proxy1' resolves to the first entry."""
    from proxy.handlers import _resolve_client_ip

    req = _make_request(headers={"x-forwarded-for": "203.0.113.7, 10.0.0.1"})
    assert _resolve_client_ip(req) == ("203.0.113.7", "header")


def test_resolve_client_ip_x_real_ip():
    """X-Real-IP is honored with source 'header'."""
    from proxy.handlers import _resolve_client_ip

    req = _make_request(headers={"x-real-ip": "198.51.100.9"})
    assert _resolve_client_ip(req) == ("198.51.100.9", "header")


def test_resolve_client_ip_prefers_xff_over_x_real_ip():
    """X-Forwarded-For takes precedence when both headers are present."""
    from proxy.handlers import _resolve_client_ip

    req = _make_request(
        headers={"x-forwarded-for": "203.0.113.7", "x-real-ip": "198.51.100.9"}
    )
    assert _resolve_client_ip(req) == ("203.0.113.7", "header")


def test_resolve_client_ip_unknown_when_no_client():
    """Without a client address or headers, resolves to ('unknown', 'direct')."""
    from proxy.handlers import _resolve_client_ip

    req = _make_request(headers={}, has_client=False)
    assert _resolve_client_ip(req) == ("unknown", "direct")


# ---------------------------------------------------------------------------
# Endpoint logging tests (via ASGI transport)
# ---------------------------------------------------------------------------


def _status_records(caplog):
    return [r for r in caplog.records if "status_request" in r.getMessage()]


@pytest.mark.asyncio
async def test_status_request_logs_local_active_query(caplog):
    """The status_request log line carries local_active_query (LP-0MSL2ZLLS009RVKR)."""
    from proxy.server import app

    from proxy import server as srv_module

    caplog.set_level(logging.INFO, logger="llama-proxy")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        with patch.object(
            srv_module, "query_llama_status", new_callable=AsyncMock
        ) as mock_qls:
            mock_qls.return_value = {"llama_server_running": True}
            resp = await ac.get("/llama/local/status")

    assert resp.status_code == 200
    records = _status_records(caplog)
    assert records, "Expected status_request log record"
    assert hasattr(records[-1], "local_active_query")
    assert records[-1].local_active_query is False


@pytest.mark.asyncio
async def test_status_request_logs_client_ip_direct(caplog):
    """Direct pollers are attributable: client_ip comes from request.client."""
    from proxy.server import app

    from proxy import server as srv_module

    caplog.set_level(logging.INFO, logger="llama-proxy")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        with patch.object(
            srv_module, "query_llama_status", new_callable=AsyncMock
        ) as mock_qls:
            mock_qls.return_value = {"llama_server_running": True}
            resp = await ac.get("/llama/local/status")

    assert resp.status_code == 200
    records = _status_records(caplog)
    assert records, "Expected status_request log record"
    assert records[-1].client_ip == "127.0.0.1"
    assert records[-1].client_ip_source == "direct"


@pytest.mark.asyncio
async def test_status_request_logs_client_ip_from_xff_header(caplog):
    """Reverse-proxy pollers are attributable via X-Forwarded-For."""
    from proxy.server import app

    from proxy import server as srv_module

    caplog.set_level(logging.INFO, logger="llama-proxy")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        with patch.object(
            srv_module, "query_llama_status", new_callable=AsyncMock
        ) as mock_qls:
            mock_qls.return_value = {"llama_server_running": True}
            resp = await ac.get(
                "/llama/local/status",
                headers={"X-Forwarded-For": "203.0.113.7"},
            )

    assert resp.status_code == 200
    records = _status_records(caplog)
    assert records, "Expected status_request log record"
    assert records[-1].client_ip == "203.0.113.7"
    assert records[-1].client_ip_source == "header"


@pytest.mark.asyncio
async def test_status_request_logs_all_response_fields(caplog):
    """The log record carries every response field plus client_ip."""
    from proxy.server import app

    from proxy import server as srv_module

    caplog.set_level(logging.INFO, logger="llama-proxy")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        with patch.object(
            srv_module, "query_llama_status", new_callable=AsyncMock
        ) as mock_qls:
            mock_qls.return_value = {"llama_server_running": True}
            resp = await ac.get("/llama/local/status")

    assert resp.status_code == 200
    records = _status_records(caplog)
    assert records, "Expected status_request log record"
    rec = records[-1]
    for field in (
        "client_ip",
        "client_ip_source",
        "latency_ms",
        "llama_server_running",
        "active_query",
        "local_active_query",
        "model_switch_in_progress",
        "current_model",
        "available_slots",
        "total_slots",
        "local_owner_session_id",
        "local_owner_lease_remaining_seconds",
    ):
        assert hasattr(rec, field), f"Missing log field {field}"

    assert rec.llama_server_running is True
    assert isinstance(rec.latency_ms, int)
    assert isinstance(rec.available_slots, int)
    assert isinstance(rec.total_slots, int)


# ---------------------------------------------------------------------------
# KeyValueFormatter tests
# ---------------------------------------------------------------------------


def _make_status_record():
    """Build a LogRecord mimicking the status_request extra payload."""
    record = logging.LogRecord(
        name="llama-proxy",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="status_request",
        args=(),
        exc_info=None,
    )
    record.client_ip = "192.168.0.191"
    record.client_ip_source = "direct"
    record.latency_ms = 12
    record.llama_server_running = True
    record.active_query = False
    record.local_active_query = False
    record.model_switch_in_progress = False
    record.current_model = "Qwen3"
    record.available_slots = 3
    record.total_slots = 3
    record.local_owner_session_id = None
    record.local_owner_lease_remaining_seconds = None
    return record


def test_key_value_formatter_renders_status_request_payload():
    """Plain-text log line includes every extra field as key=value."""
    from proxy.utils import KeyValueFormatter

    formatter = KeyValueFormatter("%(asctime)s - %(levelname)s - %(message)s")
    text = formatter.format(_make_status_record())

    assert "status_request" in text
    assert "client_ip=192.168.0.191" in text
    assert "client_ip_source=direct" in text
    assert "latency_ms=12" in text
    assert "llama_server_running=true" in text
    assert "active_query=false" in text
    assert "local_active_query=false" in text
    assert "model_switch_in_progress=false" in text
    assert "current_model=Qwen3" in text
    assert "available_slots=3" in text
    assert "total_slots=3" in text
    assert "local_owner_session_id=None" in text
    assert "local_owner_lease_remaining_seconds=None" in text


def test_key_value_formatter_skips_reserved_attributes():
    """LogRecord machinery attributes are not rendered as key=value pairs."""
    from proxy.utils import KeyValueFormatter

    formatter = KeyValueFormatter("%(asctime)s - %(levelname)s - %(message)s")
    text = formatter.format(_make_status_record())

    for reserved in (
        "name=",
        "levelname=",
        "levelno=",
        "pathname=",
        "lineno=",
        "module=",
        "funcName=",
        "created=",
        "thread=",
        "process=",
        "message=",
        "msg=",
        "asctime=",
        "taskName=",
    ):
        assert reserved not in text, f"Reserved attribute rendered: {reserved}"


def test_key_value_formatter_plain_record_unchanged():
    """Records without extras render exactly as before (message only)."""
    from proxy.utils import KeyValueFormatter

    record = logging.LogRecord(
        name="llama-proxy",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Stream started: session-1",
        args=(),
        exc_info=None,
    )
    formatter = KeyValueFormatter("%(asctime)s - %(levelname)s - %(message)s")
    text = formatter.format(record)
    assert "Stream started: session-1" in text
    assert text.rstrip().endswith("Stream started: session-1")


def test_key_value_formatter_handles_non_scalar_values():
    """Non-scalar extra values are JSON-encoded so they stay readable."""
    from proxy.utils import KeyValueFormatter

    record = _make_status_record()
    record.current_model = {"name": "Qwen3", "tokens": 4096}
    formatter = KeyValueFormatter("%(asctime)s - %(levelname)s - %(message)s")
    text = formatter.format(record)
    assert 'current_model={"name": "Qwen3", "tokens": 4096}' in text
