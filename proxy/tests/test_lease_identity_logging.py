"""Tests for client identity on lease_released structured log events.

Covers the explicit-release path (``POST /v1/leases/release`` — a Request is
in scope, so ``client_ip`` / ``client_ip_source`` / ``client_port`` are
attributed) and the background cleanup paths (idle timeout / orphan cleanup —
no Request in scope, identity omitted, event still logged)
(LP-0MSKV3IEQ004ZV88).
"""
import asyncio
import logging
import time
from types import SimpleNamespace

import httpx
import pytest

pytestmark = pytest.mark.refactor_parity


def _lease_records(caplog):
    return [r for r in caplog.records if "lease_released" in r.getMessage()]


def _seed_lease(srv_module, session_id: str) -> dict:
    """Insert a dispatch lease record so the release path has something to log."""
    record = {
        "backend": "local",
        "started_at": 1.0,
        "active": True,
        "expires_at": time.monotonic() + 60,
    }
    srv_module.local_dispatch_records[session_id] = record
    return record


# ---------------------------------------------------------------------------
# Explicit release path (Request in scope)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_release_logs_client_identity_direct(caplog):
    """Direct release requests are attributed: client_ip + client_port logged."""
    from proxy.server import app

    from proxy import server as srv_module

    caplog.set_level(logging.INFO, logger="llama-proxy")

    sid = "sess-lease-identity"
    _seed_lease(srv_module, sid)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/leases/release", json={"session_id": sid})
    finally:
        srv_module.local_dispatch_records.pop(sid, None)

    assert resp.status_code == 200
    records = _lease_records(caplog)
    assert records, "Expected lease_released record"
    rec = records[-1]
    assert rec.client_ip == "127.0.0.1"
    assert rec.client_ip_source == "direct"
    assert rec.client_port == 123  # httpx ASGITransport default client port


@pytest.mark.asyncio
async def test_explicit_release_logs_client_identity_from_header(caplog):
    """Reverse-proxy release requests attribute the header IP; port unknown."""
    from proxy.server import app

    from proxy import server as srv_module

    caplog.set_level(logging.INFO, logger="llama-proxy")

    sid = "sess-lease-identity-header"
    _seed_lease(srv_module, sid)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.post(
                "/v1/leases/release",
                json={"session_id": sid},
                headers={"X-Forwarded-For": "203.0.113.7"},
            )
    finally:
        srv_module.local_dispatch_records.pop(sid, None)

    assert resp.status_code == 200
    records = _lease_records(caplog)
    assert records, "Expected lease_released record"
    rec = records[-1]
    assert rec.client_ip == "203.0.113.7"
    assert rec.client_ip_source == "header"
    assert rec.client_port == "unknown"


# ---------------------------------------------------------------------------
# Background cleanup paths (no Request in scope)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_cleanup_omits_client_identity(caplog):
    """Idle-timeout cleanup logs the event but omits client identity."""
    from proxy.router_helpers import _cleanup_stale_local_dispatch

    caplog.set_level(logging.INFO, logger="llama-proxy")
    logger = logging.getLogger("llama-proxy")

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 180}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={
            "sess-idle-identity": {
                "backend": "local",
                "started_at": 1.0,
                "active": False,
                "expires_at": 0.0,
            },
        },
        local_dispatch_records_lock=asyncio.Lock(),
        logger=logger,
    )

    await _cleanup_stale_local_dispatch(srv)

    records = _lease_records(caplog)
    assert records, "Expected lease_released record"
    rec = records[-1]
    assert "reason=idle_timeout" in rec.getMessage()
    assert not hasattr(rec, "client_ip"), (
        "Background path has no Request; identity must be omitted"
    )


# ---------------------------------------------------------------------------
# KeyValueFormatter rendering
# ---------------------------------------------------------------------------


def test_lease_released_renders_message_and_identity_additively():
    """KeyValueFormatter preserves the message and appends identity pairs."""
    from proxy.utils import KeyValueFormatter

    record = logging.LogRecord(
        name="llama-proxy",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="lease_released session=sess-123 reason=explicit_release",
        args=(),
        exc_info=None,
    )
    record.client_ip = "192.168.0.191"
    record.client_ip_source = "direct"
    record.client_port = 51842
    formatter = KeyValueFormatter("%(asctime)s - %(levelname)s - %(message)s")
    text = formatter.format(record)

    assert "lease_released session=sess-123 reason=explicit_release" in text
    assert "client_ip=192.168.0.191" in text
    assert "client_ip_source=direct" in text
    assert "client_port=51842" in text
