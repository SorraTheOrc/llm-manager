"""Regression tests for the bounded remote-provider retry failover budget.

LP-0MU1RXEPL005CK97: a stalled remote gateway used to sit in the request's
critical path for the per-chunk idle timeout once per Tier-1 retry
(default 4 x 240s) plus empty-response retries, pushing clients past their
900s queue-timeout. ``upstream_retry_failover_budget_seconds`` hard-bounds
the wall-clock time a single provider may spend retrying a stall/empty
response before yielding a terminal error so the fallback chain fails over.

These tests assert observable stream behaviour via the public
``_handle_remote_streaming`` entry point:
- AC1: time-to-failover on a stalling gateway is bounded (no repeated
  idle-timeout waits).
- AC2: the retry budget is configurable and the exhaustion is logged.
- AC4: bounded failover for a stalling-then-empty upstream.
- Content-bearing streams are unaffected by the pre-content budget.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse
from proxy.proxy_remote import _handle_remote_streaming

# ===================================================================
# Async iterator / response / client helpers
# ===================================================================


class _ChunkIterator:
    """Async iterator that yields byte chunks, optionally hanging after."""

    def __init__(self, chunks, hang_after=False, chunk_delay=0.0):
        self._chunks = list(chunks)
        self._hang_after = hang_after
        self._chunk_delay = chunk_delay

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for chunk in self._chunks:
            if self._chunk_delay > 0:
                await asyncio.sleep(self._chunk_delay)
            yield chunk
        if self._hang_after:
            await asyncio.Event().wait()


class _DelayedAfterFirstIterator:
    """Yields the first chunk immediately, then delays before each later chunk."""

    def __init__(self, chunks, delay_between=0.0):
        self._chunks = list(chunks)
        self._delay = delay_between
        self._first = True

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for chunk in self._chunks:
            if not self._first and self._delay > 0:
                await asyncio.sleep(self._delay)
            self._first = False
            yield chunk


def _make_response(status_code=200, headers=None, aiter_chunks=None, hang_after=False):
    resp = MagicMock(spec=httpx.Response)
    type(resp).status_code = PropertyMock(return_value=status_code)
    resp.headers = headers or {"content-type": "text/event-stream"}
    if aiter_chunks is not None:
        resp.aiter_bytes = MagicMock(
            return_value=_ChunkIterator(
                aiter_chunks, hang_after=hang_after
            )
        )
    return resp


def _make_client(responses):
    """Build a mock client whose stream() returns each response in order."""
    cms = []
    for resp in responses:
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=None)
        cms.append(cm)
    client = MagicMock(spec=httpx.AsyncClient)
    client.stream = MagicMock(side_effect=cms)
    client.aclose = AsyncMock(return_value=None)
    return client


def _make_request():
    req = MagicMock(spec=Request)
    req.method = "POST"
    req.url.path = "/v1/chat/completions"
    req.is_disconnected = AsyncMock(return_value=False)
    req.headers = {}
    return req


def _make_srv(config):
    srv = MagicMock()
    srv.config = config
    srv.logger = MagicMock()
    return srv


_EMPTY_STREAM = [
    b'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}\n\n',
    b'data: [DONE]\n\n',
]


async def _run_stream(client, srv, **kwargs):
    request = _make_request()
    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client):
        with patch("proxy.proxy_remote._srv", return_value=srv):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        with patch(
                            "proxy.proxy_remote._schedule_recv_token_increment",
                            AsyncMock(),
                        ):
                            result = await _handle_remote_streaming(
                                request=request,
                                target_url="https://api.example.com/v1/chat/completions",
                                headers={"Authorization": "Bearer test"},
                                body=b'{"stream": true, "model": "test"}',
                                body_json={"stream": True, "model": "test"},
                                model_name="deepseek-v4-flash",
                                remote_timeout=httpx.Timeout(30.0),
                                provider="opencode-go",
                                entry="opencode-go-3",
                                **kwargs,
                            )
                            collected = [
                                chunk async for chunk in result.body_iterator
                            ]
    return result, collected


def _last_error_type(collected):
    text = b"".join(collected).decode("utf-8", errors="replace")
    assert '"finish_reason":"error"' in text.replace(" ", ""), (
        f"expected terminal finish_reason:error, got: {text[:300]}"
    )
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:") and '"finish_reason"' in line:
            try:
                payload = json.loads(line[5:].strip())
            except Exception:
                continue
            err = payload.get("choices", [{}])[0].get("error")
            if err:
                return err.get("type")
    return None


# ===================================================================
# AC1 / AC4: bounded failover for a stalling upstream
# ===================================================================


@pytest.mark.asyncio
async def test_failover_budget_bounds_stalling_gateway():
    """A stalling gateway fails over after a bounded number of attempts.

    With a small failover budget the proxy must NOT repeat the idle-timeout
    stall once per Tier-1 retry; it yields a terminal error after ~budget
    seconds so the fallback chain advances.
    """
    client = _make_client([
        _make_response(aiter_chunks=[], hang_after=True),
        _make_response(aiter_chunks=[], hang_after=True),
        _make_response(aiter_chunks=[], hang_after=True),
        _make_response(aiter_chunks=[], hang_after=True),
    ])
    srv = _make_srv({
        "server": {
            "upstream_retry_max_attempts": 3,
            "upstream_retry_base_delay_seconds": 0.02,
            "upstream_retry_max_delay_seconds": 0.02,
        }
    })

    start = time.monotonic()
    result, collected = await _run_stream(
        client,
        srv,
        upstream_idle_timeout_seconds=0.1,
        upstream_retry_connect_timeout_seconds=0.05,
        upstream_retry_failover_budget_seconds=0.2,
    )
    elapsed = time.monotonic() - start

    assert isinstance(result, StreamingResponse)
    # Initial attempt + one bounded retry (0.1s stall + 0.02s backoff +
    # remaining 0.08s capped read = ~0.2s budget), not the full 1 + 3
    # retries (4 x idle timeout).
    assert client.stream.call_count == 2, (
        f"expected 2 stream() calls (bounded failover), got {client.stream.call_count}"
    )
    assert elapsed < 1.0, f"failover took {elapsed:.2f}s (budget 0.2s)"
    assert _last_error_type(collected) == "retry_budget_exhausted"


@pytest.mark.asyncio
async def test_stalling_then_empty_fails_over_within_budget():
    """Stalling then empty upstream is bounded by the retry failover budget.

    AC4: attempt 1 stalls (consuming part of the budget), attempt 2 returns an
    empty stream. The empty-response retries must not extend the critical
    path past the budget.
    """
    # Attempt 1: silent stall. Attempts 2..: empty streams.
    client = _make_client([
        _make_response(aiter_chunks=[], hang_after=True),
        _make_response(aiter_chunks=_EMPTY_STREAM),
        _make_response(aiter_chunks=_EMPTY_STREAM),
        _make_response(aiter_chunks=_EMPTY_STREAM),
    ])
    srv = _make_srv({
        "server": {
            "upstream_retry_max_attempts": 3,
            "upstream_retry_base_delay_seconds": 0.02,
            "upstream_retry_max_delay_seconds": 0.02,
            # Long empty delay: without the budget this would add ~3s of
            # bounded-but-slow empty retries; the budget caps it.
            "upstream_empty_retry_max_attempts": 3,
            "upstream_empty_retry_base_delay_seconds": 1.0,
        }
    })

    start = time.monotonic()
    result, collected = await _run_stream(
        client,
        srv,
        upstream_idle_timeout_seconds=0.05,
        upstream_retry_connect_timeout_seconds=0.05,
        upstream_retry_failover_budget_seconds=0.2,
    )
    elapsed = time.monotonic() - start

    assert isinstance(result, StreamingResponse)
    # Unbounded worst case would be 1 (stall) + 3 (stall retries) +
    # 3 (empty retries) = 7 stream calls and several seconds. The budget
    # bounds it to the stall + a couple of attempts.
    assert client.stream.call_count <= 3, (
        f"expected bounded attempts, got {client.stream.call_count}"
    )
    assert elapsed < 1.0, f"failover took {elapsed:.2f}s (budget 0.2s)"
    assert _last_error_type(collected) == "retry_budget_exhausted"


# ===================================================================
# AC2: configurable + logged
# ===================================================================


@pytest.mark.asyncio
async def test_failover_budget_read_from_config_and_logged():
    """The budget is read from server config and its exhaustion is logged.

    AC2: the retry budget is configurable (here via ``server`` config) and
    the terminal failover logs the configured budget.
    """
    client = _make_client([
        _make_response(aiter_chunks=[], hang_after=True),
        _make_response(aiter_chunks=[], hang_after=True),
    ])
    srv = _make_srv({
        "server": {
            "upstream_retry_max_attempts": 3,
            "upstream_retry_base_delay_seconds": 0.02,
            "upstream_retry_max_delay_seconds": 0.02,
            "upstream_retry_failover_budget_seconds": 0.1,
        }
    })

    _result, collected = await _run_stream(
        client,
        srv,
        upstream_idle_timeout_seconds=0.05,
        upstream_retry_connect_timeout_seconds=0.05,
    )

    assert client.stream.call_count == 2
    assert _last_error_type(collected) == "retry_budget_exhausted"

    budget_logs = [
        call for call in srv.logger.warning.call_args_list
        if "failover budget exhausted" in str(call.args[0])
    ]
    assert len(budget_logs) == 1, (
        f"expected exactly one failover-budget warning, got {len(budget_logs)}"
    )
    fmt, *args = budget_logs[0].args
    assert "budget=%.2fs" in fmt and "elapsed=%.2fs" in fmt, (
        f"budget log must include budget/elapsed placeholders: {fmt}"
    )
    # The configured budget value is passed through to the log.
    assert 0.1 in args, f"expected budget 0.1 in log args, got {args}"


@pytest.mark.asyncio
async def test_failover_budget_param_overrides_config():
    """An explicit parameter overrides the config value.

    AC2: the budget is configurable; an explicit argument wins over config.
    """
    client = _make_client([
        _make_response(aiter_chunks=[], hang_after=True),
        _make_response(aiter_chunks=[], hang_after=True),
    ])
    srv = _make_srv({
        "server": {
            "upstream_retry_max_attempts": 3,
            "upstream_retry_base_delay_seconds": 0.02,
            "upstream_retry_max_delay_seconds": 0.02,
            # Config says 300s, but the explicit param must win.
            "upstream_retry_failover_budget_seconds": 300,
        }
    })

    start = time.monotonic()
    _result, collected = await _run_stream(
        client,
        srv,
        upstream_idle_timeout_seconds=0.05,
        upstream_retry_connect_timeout_seconds=0.05,
        upstream_retry_failover_budget_seconds=0.1,
    )
    elapsed = time.monotonic() - start

    assert client.stream.call_count == 2
    assert elapsed < 1.0, f"explicit param did not override config: {elapsed:.2f}s"
    assert _last_error_type(collected) == "retry_budget_exhausted"


# ===================================================================
# Content-bearing streams are not affected by the pre-content budget
# ===================================================================


@pytest.mark.asyncio
async def test_content_streaming_unaffected_by_tiny_budget():
    """Once content is delivered the failover budget no longer applies.

    A content-bearing stream that pauses between chunks for longer than the
    (tiny) failover budget must still complete: the budget only bounds
    pre-content retries, so it cannot cut off a productive stream.
    """
    chunks = [
        b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"},"index":0}]}\n\n',
        b'data: [DONE]\n\n',
    ]
    # The first (content-bearing) chunk arrives immediately so the budget
    # does not pre-empt it; the SECOND chunk arrives 0.06s later, which is
    # LONGER than the 0.05s budget — that pause must not fail over because
    # content was already delivered (the budget only bounds pre-content
    # retries).
    resp = MagicMock(spec=httpx.Response)
    type(resp).status_code = PropertyMock(return_value=200)
    resp.headers = {"content-type": "text/event-stream"}
    resp.aiter_bytes = MagicMock(
        return_value=_DelayedAfterFirstIterator(chunks, delay_between=0.06)
    )
    client = _make_client([resp])
    srv = _make_srv({"server": {}})

    result, collected = await _run_stream(
        client,
        srv,
        upstream_idle_timeout_seconds=1.0,
        upstream_retry_connect_timeout_seconds=0.05,
        upstream_retry_failover_budget_seconds=0.05,
    )

    assert isinstance(result, StreamingResponse)
    assert client.stream.call_count == 1, "content stream must not retry/fail over"
    assert collected == chunks, "content stream must pass through unchanged"
