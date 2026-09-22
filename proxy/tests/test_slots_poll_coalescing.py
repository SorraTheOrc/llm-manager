"""
Hermetic tests for /slots poll coalescing (LP-0MUCEFD5B003TURT).

Context
-------
Every herdr client polls ``/llama/local/status``, and each poll queries
llama-server ``/slots``. Under N concurrent pollers that is N upstream
requests per interval, amplifying load on an already-saturated backend.
A short-TTL cache plus single-flight coalescing bounds upstream /slots
volume by the TTL instead of by N; failures are cached for a backoff
window and callers fall back to last-known counts with the stale flag.

These tests verify hermetically (mock HTTP client, no live llama-server).
"""

import asyncio
import time

import pytest

from proxy import observability as obs


class _Resp:
    def __init__(self, status_code: int, data):
        self.status_code = status_code
        self._data = data
        self.text = "[]"

    async def json(self):
        return self._data


class _CountingClient:
    """Mock httpx client that counts upstream /slots requests."""

    def __init__(self, status_code: int = 200, data=None, delay: float = 0.0):
        self.status_code = status_code
        self.data = data if data is not None else [{"id": 0, "is_processing": False}]
        self.delay = delay
        self.calls: list[str] = []

    async def get(self, url):
        self.calls.append(url)
        if self.delay:
            await asyncio.sleep(self.delay)
        return _Resp(self.status_code, self.data)


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    obs.reset_slots_poll_cache()
    monkeypatch.delenv("SLOTS_POLL_TTL_SECONDS", raising=False)
    monkeypatch.delenv("SLOTS_POLL_FAILURE_BACKOFF_SECONDS", raising=False)
    yield
    obs.reset_slots_poll_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# Coalescing / TTL
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_concurrent_polls_share_one_upstream_request():
    """AC: N concurrent pollers produce one upstream /slots request."""
    client = _CountingClient(delay=0.05)
    url = "http://localhost:8080/slots?model=m"

    results = await asyncio.gather(
        *[obs._fetch_slots_payload(url, 2.0, client) for _ in range(10)]
    )

    assert len(client.calls) == 1, (
        f"10 concurrent pollers must coalesce to 1 upstream request, got {len(client.calls)}"
    )
    assert all(status == 200 for status, _ in results)


@pytest.mark.asyncio
async def test_second_poll_within_ttl_is_cached(monkeypatch):
    """A repeat poll within the TTL does not touch the backend."""
    monkeypatch.setenv("SLOTS_POLL_TTL_SECONDS", "30")
    client = _CountingClient()
    url = "http://localhost:8080/slots"

    await obs._fetch_slots_payload(url, 2.0, client)
    await obs._fetch_slots_payload(url, 2.0, client)
    await obs._fetch_slots_payload(url, 2.0, client)

    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_poll_after_ttl_refetches(monkeypatch):
    """Once the TTL elapses, the next poll hits the backend again."""
    monkeypatch.setenv("SLOTS_POLL_TTL_SECONDS", "0")
    client = _CountingClient()
    url = "http://localhost:8080/slots"

    await obs._fetch_slots_payload(url, 2.0, client)
    await obs._fetch_slots_payload(url, 2.0, client)

    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_distinct_urls_do_not_share_cache():
    """Different models/endpoints have independent cache entries."""
    client = _CountingClient()
    await obs._fetch_slots_payload("http://localhost:8080/slots?model=a", 2.0, client)
    await obs._fetch_slots_payload("http://localhost:8080/slots?model=b", 2.0, client)
    assert len(client.calls) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Failure backoff
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_failure_is_cached_for_backoff_window(monkeypatch):
    """A failed /slots fetch is cached for the backoff window."""
    monkeypatch.setenv("SLOTS_POLL_TTL_SECONDS", "0")
    monkeypatch.setenv("SLOTS_POLL_FAILURE_BACKOFF_SECONDS", "30")
    client = _CountingClient(status_code=500)
    url = "http://localhost:8080/slots"

    status1, _ = await obs._fetch_slots_payload(url, 2.0, client)
    status2, _ = await obs._fetch_slots_payload(url, 2.0, client)

    assert status1 == 500 and status2 == 500
    assert len(client.calls) == 1, "Failure must be cached for the backoff window"


@pytest.mark.asyncio
async def test_success_not_held_by_failure_backoff(monkeypatch):
    """A success uses the short TTL, not the longer failure backoff."""
    monkeypatch.setenv("SLOTS_POLL_TTL_SECONDS", "0")
    monkeypatch.setenv("SLOTS_POLL_FAILURE_BACKOFF_SECONDS", "30")
    client = _CountingClient(status_code=200)
    url = "http://localhost:8080/slots"

    await obs._fetch_slots_payload(url, 2.0, client)
    await obs._fetch_slots_payload(url, 2.0, client)
    assert len(client.calls) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# _query_slots routes through the coalescer
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_query_slots_coalesces_repeat_calls(monkeypatch):
    """The status-path ``_query_slots`` benefits from the coalescer."""
    monkeypatch.setenv("SLOTS_POLL_TTL_SECONDS", "30")
    client = _CountingClient(
        data=[
            {"id": 0, "is_processing": False},
            {"id": 1, "is_processing": True},
        ]
    )

    available1, total1 = await obs._query_slots(client, 8080, model="m")
    available2, total2 = await obs._query_slots(client, 8080, model="m")

    assert (available1, total1) == (1, 2)
    assert (available2, total2) == (1, 2)
    assert len(client.calls) == 1, "Repeat status polls within TTL must coalesce"


@pytest.mark.asyncio
async def test_query_slots_updates_last_known_counts(monkeypatch):
    """A successful coalesced query still updates the last-known counts."""
    monkeypatch.setenv("SLOTS_POLL_TTL_SECONDS", "30")
    client = _CountingClient(
        data=[{"id": 0, "is_processing": False}, {"id": 1, "is_processing": False}]
    )
    await obs._query_slots(client, 8080, model="m")
    assert obs.last_known_slot_counts() == (2, 2)
