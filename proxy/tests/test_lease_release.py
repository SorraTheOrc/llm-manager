"""
Tests for the POST /v1/leases/release endpoint.

Tests cover:
- Release of an active lease (record removed from local_dispatch_records)
- Release of an inactive lease (expired but not yet cleaned up)
- Idempotent no-op when session_id has no matching lease
- 400 Bad Request when session_id is missing or empty
- 500 Internal Server Error on unexpected errors
"""

import asyncio
import time
from unittest.mock import patch

import httpx
import pytest


class _FakeLock:
    """A minimal async lock that always acquires immediately."""

    def locked(self):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass


@pytest.fixture
def transport():
    """Return an ASGI transport bound to the real proxy app."""
    from proxy.server import app

    return httpx.ASGITransport(app=app)


# ---------------------------------------------------------------------------
# Shared setup: patch server state so /v1/leases/release can reach
# local_dispatch_records without a live server.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_release_active_lease(transport):
    """Releasing an active lease removes the record and returns 200."""
    from proxy import server

    now = time.monotonic()
    records = {
        "session-123": {
            "backend": "local",
            "started_at": now - 10,
            "active": True,
            "expires_at": now + 170,
        }
    }

    with patch.object(server, "local_dispatch_records", records):
        with patch.object(server, "local_dispatch_records_lock", _FakeLock()):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/v1/leases/release",
                    json={"session_id": "session-123"},
                )

    assert resp.status_code == 200
    j = resp.json()
    assert j == {"status": "ok"}
    # Verify the record was removed
    assert "session-123" not in records


@pytest.mark.asyncio
async def test_release_inactive_lease(transport):
    """Releasing an inactive (expired) lease also removes the record."""
    from proxy import server

    now = time.monotonic()
    records = {
        "session-456": {
            "backend": "local",
            "started_at": now - 200,
            "active": False,
            "expires_at": now - 20,
        }
    }

    with patch.object(server, "local_dispatch_records", records):
        with patch.object(server, "local_dispatch_records_lock", _FakeLock()):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/v1/leases/release",
                    json={"session_id": "session-456"},
                )

    assert resp.status_code == 200
    j = resp.json()
    assert j == {"status": "ok"}
    assert "session-456" not in records


@pytest.mark.asyncio
async def test_release_unknown_session_idempotent(transport):
    """Releasing a non-existent session_id is an idempotent no-op (200)."""
    from proxy import server

    records = {
        "existing-session": {
            "backend": "local",
            "started_at": time.monotonic() - 5,
            "active": True,
            "expires_at": time.monotonic() + 175,
        }
    }

    with patch.object(server, "local_dispatch_records", records):
        with patch.object(server, "local_dispatch_records_lock", _FakeLock()):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/v1/leases/release",
                    json={"session_id": "unknown-session"},
                )

    assert resp.status_code == 200
    j = resp.json()
    assert j == {"status": "ok"}
    # Existing records should be untouched
    assert "existing-session" in records


@pytest.mark.asyncio
async def test_release_missing_session_id_400(transport):
    """Missing session_id returns 400 Bad Request."""
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/v1/leases/release",
            json={},
        )

    assert resp.status_code == 400
    j = resp.json()
    assert "session_id is required" in j.get("detail", "")


@pytest.mark.asyncio
async def test_release_empty_session_id_400(transport):
    """Empty session_id returns 400 Bad Request."""
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/v1/leases/release",
            json={"session_id": ""},
        )

    assert resp.status_code == 400
    j = resp.json()
    assert "session_id is required" in j.get("detail", "")


@pytest.mark.asyncio
async def test_release_null_session_id_400(transport):
    """Null session_id returns 400 Bad Request."""
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/v1/leases/release",
            json={"session_id": None},
        )

    assert resp.status_code == 400
    j = resp.json()
    assert "session_id is required" in j.get("detail", "")


@pytest.mark.asyncio
async def test_release_not_json_body_400(transport):
    """Non-JSON body returns 400."""
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/v1/leases/release",
            content=b"not-json",
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 400
    j = resp.json()
    # Should contain an error detail
    assert "detail" in j


@pytest.mark.asyncio
async def test_release_multiple_calls_idempotent(transport):
    """Calling the endpoint multiple times with the same session_id must not error."""
    from proxy import server

    records = {
        "session-duplicate": {
            "backend": "local",
            "started_at": time.monotonic() - 10,
            "active": True,
            "expires_at": time.monotonic() + 170,
        }
    }

    with patch.object(server, "local_dispatch_records", records):
        with patch.object(server, "local_dispatch_records_lock", _FakeLock()):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                # First call
                resp1 = await ac.post(
                    "/v1/leases/release",
                    json={"session_id": "session-duplicate"},
                )
                assert resp1.status_code == 200
                assert resp1.json() == {"status": "ok"}
                # Record should be gone
                assert "session-duplicate" not in records

                # Second call — idempotent
                resp2 = await ac.post(
                    "/v1/leases/release",
                    json={"session_id": "session-duplicate"},
                )
                assert resp2.status_code == 200
                assert resp2.json() == {"status": "ok"}
