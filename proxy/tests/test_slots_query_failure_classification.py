"""``/slots`` transport-failure classification (LP-0MWCOMI160011F6C AC3).

The 2026-11-11 starvation was undiagnosable from the metrics: ``/slots``
failed on every call for 93 minutes while
``llama_slots_query_failures_total{reason="unknown"}`` was the only signal,
because ``_query_slots``'s generic ``except Exception`` branch caught every
httpx transport error (``httpx.ConnectError`` / ``ConnectTimeout`` /
``ReadTimeout`` are neither the builtin ``TimeoutError`` nor the builtin
``ConnectionError``/``OSError``).

These tests pin the mapping through the published label vocabulary
(``monitoring/slots_query_alerts.yaml`` matches those labels):

- ``httpx.ConnectError`` (and the other ``httpx.NetworkError`` members, plus
  builtin ``ConnectionError``/``OSError``) → ``connection_error``;
- ``httpx.ConnectTimeout`` / ``httpx.ReadTimeout`` (any
  ``httpx.TimeoutException``) → ``timeout``;
- anything else → ``unknown``.

Every assertion goes through the observable behaviour of ``_query_slots`` (the
recorded metric label, the ``status_out`` reason and the published
``slots_stale_reason``), never by re-implementing the mapping.
"""

from __future__ import annotations

import httpx
import proxy.observability as observability
import pytest


class _FailingClient:
    """Minimal httpx-AsyncClient stand-in whose ``get`` raises *exc*."""

    def __init__(self, exc: BaseException):
        self._exc = exc

    async def get(self, *_args, **_kwargs):
        raise self._exc


def _request() -> httpx.Request:
    return httpx.Request("GET", "http://localhost:8080/slots?model=Qwen3")


@pytest.fixture(autouse=True)
def _isolate_slots_state(monkeypatch):
    """Reset module caches/state and capture recorded failure labels."""
    import proxy.metrics as metrics

    recorded: list[str] = []
    monkeypatch.setattr(
        metrics,
        "record_slots_query_failure",
        lambda reason: recorded.append(reason),
    )
    observability.reset_slots_stale_reason()
    observability.reset_slots_poll_cache()
    yield recorded
    observability.reset_slots_stale_reason()
    observability.reset_slots_poll_cache()


async def _query_with(exc: BaseException) -> tuple[tuple, dict]:
    """Run one ``_query_slots`` call whose transport layer raises *exc*."""
    status_out: dict = {}
    result = await observability._query_slots(
        _FailingClient(exc),
        8080,
        timeout=0.01,
        model="Qwen3",
        status_out=status_out,
    )
    return result, status_out


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("connection refused", request=_request()),
        httpx.ReadError("read failed", request=_request()),
        httpx.WriteError("write failed", request=_request()),
        httpx.CloseError("close failed", request=_request()),
        ConnectionError("builtin connection error"),
        OSError("socket error"),
    ],
)
@pytest.mark.asyncio
async def test_network_errors_classify_as_connection_error(exc, _isolate_slots_state):
    """``httpx.NetworkError`` and the builtin socket errors are connection
    errors, not ``unknown`` — the label the incident could not show."""
    result, status_out = await _query_with(exc)

    assert result == (0, 0)
    assert status_out["reason"] == "connection_error"
    assert _isolate_slots_state == ["connection_error"]
    assert (
        observability.last_slots_stale_reason()
        == observability.SLOTS_STALE_REASON_QUERY_ERROR
    )


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectTimeout("connect timed out", request=_request()),
        httpx.ReadTimeout("read timed out", request=_request()),
        httpx.WriteTimeout("write timed out", request=_request()),
        httpx.PoolTimeout("pool timed out"),
        TimeoutError("builtin/asyncio timeout"),
    ],
)
@pytest.mark.asyncio
async def test_timeouts_classify_as_timeout(exc, _isolate_slots_state):
    """Every timeout shape — httpx's ``TimeoutException`` family and the
    builtin ``TimeoutError`` — is reported as ``timeout``.

    ``ConnectTimeout`` is both a connect error and a timeout; the timeout
    reading must win, because the alert distinguishes the two labels.
    """
    result, status_out = await _query_with(exc)

    assert result == (0, 0)
    assert status_out["reason"] == "timeout"
    assert _isolate_slots_state == ["timeout"]
    assert (
        observability.last_slots_stale_reason()
        == observability.SLOTS_STALE_REASON_QUERY_TIMEOUT
    )


@pytest.mark.asyncio
async def test_non_transport_error_stays_unknown(_isolate_slots_state):
    """A non-transport exception keeps the ``unknown`` label — the fix widens
    no other part of the vocabulary."""
    result, status_out = await _query_with(ValueError("unexpected"))

    assert result == (0, 0)
    assert status_out["reason"] == "unknown"
    assert _isolate_slots_state == ["unknown"]


@pytest.mark.asyncio
async def test_sampler_path_shares_the_classification():
    """The slot-occupancy sampler's ``/slots`` fetch delegates to the same
    mapping, so the label vocabulary cannot drift between callers."""
    exc = httpx.ConnectError("connection refused", request=_request())

    async def _failing_fetch(_url, _timeout, _client):
        raise exc

    orig_fetch = observability._fetch_slots_payload
    observability._fetch_slots_payload = _failing_fetch
    try:
        payload, reason = await observability._fetch_slots_payload_for_sampler(
            "http://localhost:8080/slots?model=Qwen3", 0.01
        )
    finally:
        observability._fetch_slots_payload = orig_fetch

    assert payload is None
    assert reason == "connection_error"
