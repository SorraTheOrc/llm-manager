"""Tests for remote stream maximum-duration / activity watchdog.

LP-0MSVP7ZML003XZTJ: a pi-agent audit pane held a remote deepseek stream
for 13+ hours in a "connected but idle" state — the per-chunk idle timeout
(upstream_idle_timeout_seconds) only fires on SILENCE, so a stream that
keeps receiving heartbeats/keep-alives (or trickles empty deltas) never
terminates, holding proxy state (local_active_query, slots) indefinitely.

The fix adds two bounded deadlines to ``_handle_remote_streaming``:

- ``upstream_max_stream_duration_seconds`` (default 14400 = 4h): hard cap on
  total remote stream lifetime. On expiry the stream is terminated with a
  synthetic ``finish_reason: error`` event (error.type ``stream_max_duration``)
  and NO retry — restarting a stuck stream just re-sticks it.
- ``upstream_activity_timeout_seconds`` (default 1800 = 30 min): max time
  since the last CONTENT-bearing chunk (heartbeats/empty deltas do not count
  as progress). On expiry the stream terminates (error.type
  ``stream_activity_timeout``), no retry.

Both are best-effort watchdog bounds inside the stream generator: the
per-read ``asyncio.wait_for`` budget is the minimum of the idle timeout and
the remaining duration/activity budgets, so a deadline firing classifies as
its own reason instead of the ordinary idle-stall retry path.

Tests cover: max-duration expiry, activity expiry (heartbeats flowing but no
content), normal streams unaffected, idle-stall retry still retries when
within budgets, and the termination metric.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import proxy.metrics as metrics
import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse
from proxy.proxy_remote import _handle_remote_streaming

# ===================================================================
# Async iterator helpers
# ===================================================================


class HeartbeatThenHangIterator:
    """Yields content, then keeps yielding non-content keep-alive chunks
    every *hb_interval* seconds forever (simulates a connected-but-idle
    upstream that never goes SILENT — so the per-chunk idle timeout never
    fires)."""

    def __init__(self, content_chunks, hb_chunk, hb_interval=0.01):
        self._content = list(content_chunks)
        self._hb = hb_chunk
        self._interval = hb_interval

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._content:
            return self._content.pop(0)
        await asyncio.sleep(self._interval)
        return self._hb


class HangAfterChunksIterator:
    """Yields the given chunks then hangs forever (true silence)."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            await asyncio.Event().wait()
        return self._chunks.pop(0)


# ===================================================================
# Mock response / client factories (mirror test_upstream_stall_detection)
# ===================================================================


def _make_mock_response(status_code=200, headers=None, iterator=None, error_body=None):
    mock_resp = MagicMock(spec=httpx.Response)
    type(mock_resp).status_code = PropertyMock(return_value=status_code)
    mock_resp.headers = headers or {"content-type": "text/event-stream"}
    if iterator is not None:
        mock_resp.aiter_bytes = MagicMock(return_value=iterator)
    if error_body is not None:
        mock_resp.aread = AsyncMock(return_value=error_body)
    return mock_resp


def _make_streaming_mock_client(mock_response, stream_call_count=1):
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_response)
    cm.__aexit__ = AsyncMock(return_value=None)

    client_instance = MagicMock(spec=httpx.AsyncClient)
    client_instance.stream = MagicMock(return_value=cm)
    client_instance.aclose = AsyncMock(return_value=None)
    return client_instance


async def _collect_stream(result):
    return [chunk async for chunk in result.body_iterator]


def _synthetic_error_from(chunks):
    """Return the error dict of the first synthetic finish_reason:error SSE
    chunk in *chunks* (or None)."""
    for chunk in chunks:
        text = chunk.decode("utf-8", errors="replace")
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            try:
                j = json.loads(payload)
            except Exception:
                continue
            for choice in j.get("choices", []):
                if choice.get("finish_reason") == "error":
                    return choice.get("error") or {}
    return None


@pytest.fixture
def mock_request():
    req = MagicMock(spec=Request)
    req.method = "POST"
    req.url.path = "/v1/chat/completions"
    req.is_disconnected = AsyncMock(return_value=False)
    return req


async def _call_streaming(mock_request, client, **kwargs):
    """Run _handle_remote_streaming with standard patches; return result + chunks.

    Returns (result, chunks) — chunks already collected from the stream.
    """
    defaults = dict(
        target_url="https://api.example.com/v1/chat/completions",
        headers={"Authorization": "Bearer test"},
        body=b'{"stream": true, "model": "test"}',
        body_json={"stream": True, "model": "test"},
        model_name="test-model",
        remote_timeout=httpx.Timeout(30.0),
        upstream_idle_timeout_seconds=0.5,
    )
    defaults.update(kwargs)
    with patch("proxy.proxy_remote.httpx.AsyncClient", return_value=client):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        with patch("proxy.proxy_remote._srv") as mock_srv:
                            mock_srv.return_value.config = {}
                            mock_srv.return_value.logger = MagicMock()
                            result = await _handle_remote_streaming(
                                request=mock_request, **defaults
                            )
                            chunks = await _collect_stream(result)
    return result, chunks


# ===================================================================
# Max stream duration watchdog
# ===================================================================


class TestMaxStreamDuration:
    """upstream_max_stream_duration_seconds caps total remote stream lifetime."""

    @pytest.mark.asyncio
    async def test_max_duration_expiry_terminates_stream(self, mock_request):
        """A stream that stays alive past the duration cap terminates with a
        synthetic error and does NOT retry."""
        content = [
            b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
            b'data: {"choices":[{"delta":{"content":" world"},"index":0}]}\n\n',
        ]
        # After content, keep sending keep-alives forever (never silent).
        hb = b": keep-alive\n\n"
        iterator = HeartbeatThenHangIterator(content, hb, hb_interval=0.01)
        mock_resp = _make_mock_response(iterator=iterator)
        client = _make_streaming_mock_client(mock_resp)

        _, chunks = await _call_streaming(
            mock_request,
            client,
            upstream_max_stream_duration_seconds=0.2,
            upstream_activity_timeout_seconds=9999,  # duration should fire first
        )

        err = _synthetic_error_from(chunks)
        assert err is not None, f"expected synthetic error, got chunks: {chunks!r}"
        assert err["type"] == "stream_max_duration"
        # No retry: the stuck stream is not restarted.
        assert client.stream.call_count == 1, (
            f"expected no retry after max-duration expiry, got {client.stream.call_count} stream() calls"
        )

    @pytest.mark.asyncio
    async def test_normal_short_stream_unaffected_by_duration_cap(self, mock_request):
        """A stream that finishes well within the cap passes through unchanged."""
        chunks_in = [
            b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
            b'data: {"choices":[{"delta":{"content":" world"},"index":0}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        mock_resp = _make_mock_response(iterator=HangAfterChunksIterator(chunks_in))
        client = _make_streaming_mock_client(mock_resp)

        _, chunks = await _call_streaming(
            mock_request,
            client,
            upstream_max_stream_duration_seconds=60,
            upstream_activity_timeout_seconds=60,
        )

        assert _synthetic_error_from(chunks) is None
        assert len(chunks) >= 3


# ===================================================================
# Activity watchdog (heartbeats flow but no content progress)
# ===================================================================


class TestActivityTimeout:
    """upstream_activity_timeout_seconds terminates a connected-but-idle stream."""

    @pytest.mark.asyncio
    async def test_heartbeats_without_content_trigger_activity_timeout(self, mock_request):
        """Keep-alives flowing with zero content progress for longer than the
        activity budget → terminate with stream_activity_timeout, no retry."""
        hb = b": keep-alive\n\n"
        iterator = HeartbeatThenHangIterator([], hb, hb_interval=0.01)
        mock_resp = _make_mock_response(iterator=iterator)
        client = _make_streaming_mock_client(mock_resp)

        _, chunks = await _call_streaming(
            mock_request,
            client,
            upstream_max_stream_duration_seconds=9999,  # activity should fire first
            upstream_activity_timeout_seconds=0.2,
        )

        err = _synthetic_error_from(chunks)
        assert err is not None, f"expected synthetic error, got chunks: {chunks!r}"
        assert err["type"] == "stream_activity_timeout"
        assert client.stream.call_count == 1, (
            f"expected no retry after activity timeout, got {client.stream.call_count} stream() calls"
        )

    @pytest.mark.asyncio
    async def test_content_progress_resets_activity_timer(self, mock_request):
        """A stream that keeps producing content-bearing chunks is never
        killed by the activity watchdog."""
        content = [
            b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
            b'data: {"choices":[{"delta":{"content":" world"},"index":0}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"!"},"index":0}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        mock_resp = _make_mock_response(iterator=HangAfterChunksIterator(content))
        client = _make_streaming_mock_client(mock_resp)

        _, chunks = await _call_streaming(
            mock_request,
            client,
            upstream_max_stream_duration_seconds=9999,
            upstream_activity_timeout_seconds=0.2,
        )

        assert _synthetic_error_from(chunks) is None
        assert len(chunks) >= 4


# ===================================================================
# Idle-stall retry still works when within budgets
# ===================================================================


class TestIdleStallRetryPreserved:
    """The existing per-chunk idle-stall retry path is unchanged by the
    watchdog when neither budget is exhausted."""

    @pytest.mark.asyncio
    async def test_idle_stall_retries_within_budgets(self, mock_request):
        """True silence (no chunks, no heartbeats) still triggers the
        ordinary idle-stall retry when the duration/activity budgets are
        large."""

        class SilentHang:
            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.Event().wait()

        first_resp = _make_mock_response(iterator=SilentHang())
        second_chunks = [
            b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        second_resp = _make_mock_response(iterator=HangAfterChunksIterator(second_chunks))

        cm1 = MagicMock()
        cm1.__aenter__ = AsyncMock(return_value=first_resp)
        cm1.__aexit__ = AsyncMock(return_value=None)
        cm2 = MagicMock()
        cm2.__aenter__ = AsyncMock(return_value=second_resp)
        cm2.__aexit__ = AsyncMock(return_value=None)

        client_instance = MagicMock(spec=httpx.AsyncClient)
        client_instance.stream = MagicMock(side_effect=[cm1, cm2])
        client_instance.aclose = AsyncMock(return_value=None)

        _, chunks = await _call_streaming(
            mock_request,
            client_instance,
            upstream_idle_timeout_seconds=0.05,
            upstream_max_stream_duration_seconds=60,
            upstream_activity_timeout_seconds=60,
        )

        # The retry succeeded → content chunks present, no synthetic error.
        assert _synthetic_error_from(chunks) is None
        assert any(b"Hello" in c for c in chunks), (
            f"expected retry content, got {chunks!r}"
        )
        assert client_instance.stream.call_count == 2


# ===================================================================
# Termination metric
# ===================================================================


class TestStreamTerminationMetric:
    """llama_remote_stream_terminated_total records watchdog terminations."""

    def test_counter_exists_with_reason_label(self):
        assert metrics._enabled
        assert metrics.llama_remote_stream_terminated_total is not None
        from prometheus_client import Counter

        assert isinstance(metrics.llama_remote_stream_terminated_total, Counter)
        assert "reason" in metrics.llama_remote_stream_terminated_total._labelnames

    def test_record_increments_counter(self):
        before = metrics.llama_remote_stream_terminated_total.labels(
            reason="stream_max_duration"
        )._value.get()
        metrics.record_remote_stream_terminated("stream_max_duration")
        after = metrics.llama_remote_stream_terminated_total.labels(
            reason="stream_max_duration"
        )._value.get()
        assert after == before + 1

    def test_record_separate_reasons(self):
        before_max = metrics.llama_remote_stream_terminated_total.labels(
            reason="stream_max_duration"
        )._value.get()
        before_act = metrics.llama_remote_stream_terminated_total.labels(
            reason="stream_activity_timeout"
        )._value.get()
        metrics.record_remote_stream_terminated("stream_activity_timeout")
        assert (
            metrics.llama_remote_stream_terminated_total.labels(
                reason="stream_max_duration"
            )._value.get()
            == before_max
        )
        assert (
            metrics.llama_remote_stream_terminated_total.labels(
                reason="stream_activity_timeout"
            )._value.get()
            == before_act + 1
        )

    @pytest.mark.asyncio
    async def test_max_duration_expiry_records_metric(self, mock_request):
        before = metrics.llama_remote_stream_terminated_total.labels(
            reason="stream_max_duration"
        )._value.get()
        content = [b'data: {"choices":[{"delta":{"content":"x"},"index":0}]}\n\n']
        hb = b": keep-alive\n\n"
        mock_resp = _make_mock_response(
            iterator=HeartbeatThenHangIterator(content, hb, hb_interval=0.01)
        )
        client = _make_streaming_mock_client(mock_resp)
        await _call_streaming(
            mock_request,
            client,
            upstream_max_stream_duration_seconds=0.2,
            upstream_activity_timeout_seconds=9999,
        )
        after = metrics.llama_remote_stream_terminated_total.labels(
            reason="stream_max_duration"
        )._value.get()
        assert after == before + 1
