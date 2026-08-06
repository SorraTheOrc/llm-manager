"""
Tests for logging the config entry name in stream-level proxy log lines
(LP-0MSC7F7BG0043TE1).

Verifies that:
1. ``Stream started`` lines for remote streams include ``entry=<config-entry-name>``.
2. ``Stream finished`` lines include ``entry=<name>`` (via ``log_response_chunk``).
3. ``Stream error`` lines include ``entry=<name>``.
4. The entry field is omitted (not empty) when no config entry name is configured.
5. ``proxy_to_remote`` derives the entry name from ``model_config["name"]``.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import proxy.server as server
import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse


def _configure_logging(tmp_path) -> None:
    """Set up the llama-proxy logger with a minimal config in tmp_path."""
    cfg = {
        "logging": {
            "directory": str(tmp_path / "logs"),
            "rotation_hours": 1,
            "retention_days": 1,
            "level": "INFO",
        }
    }
    logger = logging.getLogger("llama-proxy")
    for h in list(logger.handlers):
        logger.removeHandler(h)
    server.setup_logging(cfg)


def _records_with(caplog, substring: str) -> list:
    """Return log records whose message contains *substring*."""
    return [
        r for r in caplog.records
        if substring in str(r.getMessage())
    ]


# ═══════════════════════════════════════════════════════════════════════════
# log_response_chunk: Stream finished lines
# ═══════════════════════════════════════════════════════════════════════════


def test_stream_finished_includes_entry(caplog, tmp_path):
    """Stream finished lines include entry=<name> when entry is provided."""
    _configure_logging(tmp_path)
    caplog.set_level(logging.INFO, logger="llama-proxy")

    chunk = (
        b'data: {"choices":[{"delta":{"content":"hello"},'
        b'"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}\n\n'
    )
    server.log_response_chunk(
        chunk,
        session_id="sess123",
        model="deepseek-v4-flash",
        provider="opencode-go",
        entry="opencode-go-2-deepseek",
    )

    finished = _records_with(caplog, "Stream finished: reason=stop")
    assert finished, "expected a Stream finished record"
    assert any(
        "entry=opencode-go-2-deepseek" in str(r.getMessage())
        for r in finished
    ), "Stream finished line must include entry=<config-entry-name>"


def test_stream_finished_includes_entry_with_tokens(caplog, tmp_path):
    """Per-entry token usage is attributable when entry is present."""
    _configure_logging(tmp_path)
    caplog.set_level(logging.INFO, logger="llama-proxy")

    chunk = (
        b'data: {"choices":[{"delta":{"content":"hello"},'
        b'"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}\n\n'
    )
    server.log_response_chunk(
        chunk,
        session_id="sess123",
        model="deepseek-v4-flash",
        provider="opencode-go",
        entry="opencode-go-2-deepseek",
    )

    finished = _records_with(caplog, "Stream finished: reason=stop")
    msg = " ".join(str(r.getMessage()) for r in finished)
    assert "tokens=10/20/30" in msg
    assert "entry=opencode-go-2-deepseek" in msg


def test_stream_finished_omits_entry_when_missing(caplog, tmp_path):
    """Stream finished lines omit entry= (not empty) when no name is configured."""
    _configure_logging(tmp_path)
    caplog.set_level(logging.INFO, logger="llama-proxy")

    chunk = (
        b'data: {"choices":[{"delta":{"content":"hello"},'
        b'"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n\n'
    )
    server.log_response_chunk(
        chunk,
        session_id="sess123",
        model="deepseek-v4-flash",
        provider="opencode-go",
    )

    finished = _records_with(caplog, "Stream finished: reason=stop")
    assert finished, "expected a Stream finished record"
    for r in finished:
        msg = str(r.getMessage())
        assert "entry=" not in msg, (
            f"entry= must be omitted when no entry name is configured, got: {msg}"
        )


def test_stream_finished_omits_entry_when_empty_string(caplog, tmp_path):
    """An empty entry name is treated as absent (entry= not emitted)."""
    _configure_logging(tmp_path)
    caplog.set_level(logging.INFO, logger="llama-proxy")

    chunk = (
        b'data: {"choices":[{"delta":{"content":"hello"},'
        b'"finish_reason":"stop"}]}\n\n'
    )
    server.log_response_chunk(
        chunk,
        session_id="sess123",
        model="deepseek-v4-flash",
        provider="opencode-go",
        entry="",
    )

    finished = _records_with(caplog, "Stream finished: reason=stop")
    assert finished, "expected a Stream finished record"
    for r in finished:
        msg = str(r.getMessage())
        assert "entry=" not in msg, (
            f"entry= must be omitted for empty entry name, got: {msg}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# _handle_remote_streaming: Stream started / error lines and entry threading
# ═══════════════════════════════════════════════════════════════════════════


class AsyncIterator:
    """Helper to turn a list into an async iterator."""

    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        return self._iterator()

    async def _iterator(self):
        for item in self.items:
            yield item


class ErrorAsyncIterator:
    """Async iterator that yields chunks then raises on the next anext call."""

    def __init__(self, chunks, exc):
        self.chunks = list(chunks)
        self.exc = exc
        self.idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.idx < len(self.chunks):
            val = self.chunks[self.idx]
            self.idx += 1
            return val
        raise self.exc


def _make_mock_response(chunks, status=200, content_type="text/event-stream"):
    """Create a mock httpx response with controlled streaming chunks."""
    mock = AsyncMock()
    mock.status_code = status
    mock.headers = {"content-type": content_type}
    mock.aiter_bytes = lambda: AsyncIterator(chunks)
    return mock


def _make_error_response(chunks, exc_cls=httpx.RemoteProtocolError):
    """Create a mock httpx response whose aiter_bytes raises after chunks."""
    mock = AsyncMock()
    mock.status_code = 200
    mock.headers = {"content-type": "text/event-stream"}
    exc = exc_cls("Simulated stream error for testing")
    mock.aiter_bytes = lambda: ErrorAsyncIterator(chunks, exc)
    return mock


def _make_mock_client(mock_response):
    """Create a mock httpx.AsyncClient that returns mock_response on stream()."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_response)
    cm.__aexit__ = AsyncMock(return_value=None)

    client_instance = MagicMock()
    client_instance.stream = MagicMock(return_value=cm)
    client_instance.aclose = AsyncMock()

    mock_client_cls = MagicMock(return_value=client_instance)
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=client_instance)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

    return mock_client_cls


@pytest.fixture
def mock_request():
    """Create a mock Request object."""
    req = MagicMock(spec=Request)
    req.method = "POST"
    req.url.path = "/v1/chat/completions"
    req.is_disconnected = AsyncMock(return_value=False)
    return req


_STARTED_CHUNKS = [
    b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
    b'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}\n\n',
    b"data: [DONE]\n\n",
]


@pytest.mark.asyncio
async def test_stream_started_includes_entry(mock_request):
    """Stream started lines include entry=<config-entry-name> (AC1)."""
    from proxy.proxy_remote import _handle_remote_streaming

    info_messages = []

    mock_resp = _make_mock_response(_STARTED_CHUNKS)
    mock_client_cls = _make_mock_client(mock_resp)

    with patch("proxy.proxy_remote.httpx.AsyncClient", mock_client_cls):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        with patch("proxy.proxy_remote._srv") as mock_srv:
                            mock_srv.return_value = MagicMock(
                                logger=MagicMock(
                                    info=lambda msg, *a, **k: info_messages.append(str(msg) % a if a else str(msg)),
                                    warning=MagicMock(),
                                )
                            )
                            result = await _handle_remote_streaming(
                                request=mock_request,
                                target_url="https://api.example.com/v1/chat/completions",
                                headers={"Authorization": "Bearer test"},
                                body=b'{"stream": true, "model": "test"}',
                                body_json={"stream": True, "model": "test"},
                                model_name="deepseek-v4-flash",
                                remote_timeout=httpx.Timeout(30.0),
                                session_id="test-session-123",
                                provider="opencode-go",
                                entry="opencode-go-2-deepseek",
                            )
                            _ = [chunk async for chunk in result.body_iterator]

    started = [m for m in info_messages if "Stream started:" in m]
    assert started, "expected a Stream started log line"
    assert any(
        "entry=opencode-go-2-deepseek" in m for m in started
    ), f"Stream started line must include entry=, got: {started}"
    assert any(
        "provider=opencode-go" in m and "model=deepseek-v4-flash" in m
        for m in started
    ), "existing provider/model fields must be preserved"


@pytest.mark.asyncio
async def test_stream_started_omits_entry_when_missing(mock_request):
    """Stream started lines omit entry= when no config entry name exists."""
    from proxy.proxy_remote import _handle_remote_streaming

    info_messages = []

    mock_resp = _make_mock_response(_STARTED_CHUNKS)
    mock_client_cls = _make_mock_client(mock_resp)

    with patch("proxy.proxy_remote.httpx.AsyncClient", mock_client_cls):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        with patch("proxy.proxy_remote._srv") as mock_srv:
                            mock_srv.return_value = MagicMock(
                                logger=MagicMock(
                                    info=lambda msg, *a, **k: info_messages.append(str(msg) % a if a else str(msg)),
                                    warning=MagicMock(),
                                )
                            )
                            result = await _handle_remote_streaming(
                                request=mock_request,
                                target_url="https://api.example.com/v1/chat/completions",
                                headers={"Authorization": "Bearer test"},
                                body=b'{"stream": true, "model": "test"}',
                                body_json={"stream": True, "model": "test"},
                                model_name="deepseek-v4-flash",
                                remote_timeout=httpx.Timeout(30.0),
                                session_id="test-session-123",
                                provider="opencode-go",
                            )
                            _ = [chunk async for chunk in result.body_iterator]

    started = [m for m in info_messages if "Stream started:" in m]
    assert started, "expected a Stream started log line"
    for m in started:
        assert "entry=" not in m, (
            f"entry= must be omitted when no entry name is configured, got: {m}"
        )


@pytest.mark.asyncio
async def test_log_response_chunk_receives_entry(mock_request):
    """_handle_remote_streaming threads entry= into log_response_chunk (AC2)."""
    from proxy.proxy_remote import _handle_remote_streaming

    mock_resp = _make_mock_response(_STARTED_CHUNKS)
    mock_client_cls = _make_mock_client(mock_resp)

    with patch("proxy.proxy_remote.httpx.AsyncClient", mock_client_cls):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk") as mock_lrc:
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        with patch("proxy.proxy_remote._srv") as mock_srv:
                            mock_srv.return_value = MagicMock(
                                logger=MagicMock(info=MagicMock(), warning=MagicMock())
                            )
                            result = await _handle_remote_streaming(
                                request=mock_request,
                                target_url="https://api.example.com/v1/chat/completions",
                                headers={"Authorization": "Bearer test"},
                                body=b'{"stream": true, "model": "test"}',
                                body_json={"stream": True, "model": "test"},
                                model_name="deepseek-v4-flash",
                                remote_timeout=httpx.Timeout(30.0),
                                session_id="test-session-123",
                                provider="opencode-go",
                                entry="opencode-go-2-deepseek",
                            )
                            _ = [chunk async for chunk in result.body_iterator]

    assert mock_lrc.called, "log_response_chunk should have been called"
    assert all(
        call.kwargs.get("entry") == "opencode-go-2-deepseek"
        for call in mock_lrc.call_args_list
    ), "all log_response_chunk calls must receive entry=opencode-go-2-deepseek"


@pytest.mark.asyncio
async def test_log_response_chunk_entry_none_when_missing(mock_request):
    """Without a config entry name, entry= is not passed to log_response_chunk."""
    from proxy.proxy_remote import _handle_remote_streaming

    mock_resp = _make_mock_response(_STARTED_CHUNKS)
    mock_client_cls = _make_mock_client(mock_resp)

    with patch("proxy.proxy_remote.httpx.AsyncClient", mock_client_cls):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk") as mock_lrc:
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        with patch("proxy.proxy_remote._srv") as mock_srv:
                            mock_srv.return_value = MagicMock(
                                logger=MagicMock(info=MagicMock(), warning=MagicMock())
                            )
                            result = await _handle_remote_streaming(
                                request=mock_request,
                                target_url="https://api.example.com/v1/chat/completions",
                                headers={"Authorization": "Bearer test"},
                                body=b'{"stream": true, "model": "test"}',
                                body_json={"stream": True, "model": "test"},
                                model_name="deepseek-v4-flash",
                                remote_timeout=httpx.Timeout(30.0),
                                session_id="test-session-123",
                                provider="opencode-go",
                            )
                            _ = [chunk async for chunk in result.body_iterator]

    assert mock_lrc.called, "log_response_chunk should have been called"
    for call in mock_lrc.call_args_list:
        assert call.kwargs.get("entry", None) is None, (
            "entry must be omitted (None) when no config entry name is configured"
        )


@pytest.mark.asyncio
async def test_stream_error_includes_entry(mock_request):
    """Stream error lines include entry=<config-entry-name> (AC3)."""
    from proxy.proxy_remote import _handle_remote_streaming

    warning_messages = []

    chunks = [
        b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n\n',
    ]
    mock_resp = _make_error_response(chunks, httpx.RemoteProtocolError)
    mock_client_cls = _make_mock_client(mock_resp)

    with patch("proxy.proxy_remote.httpx.AsyncClient", mock_client_cls):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response_chunk"):
                with patch("proxy.proxy_remote.log_response"):
                    with patch("proxy.proxy_remote.log_request"):
                        with patch("proxy.proxy_remote._srv") as mock_srv:
                            mock_srv.return_value = MagicMock(
                                logger=MagicMock(
                                    info=MagicMock(),
                                    warning=lambda msg, *a, **k: warning_messages.append(str(msg) % a if a else str(msg)),
                                )
                            )
                            result = await _handle_remote_streaming(
                                request=mock_request,
                                target_url="https://api.example.com/v1/chat/completions",
                                headers={"Authorization": "Bearer test"},
                                body=b'{"stream": true, "model": "test"}',
                                body_json={"stream": True, "model": "test"},
                                model_name="deepseek-v4-flash",
                                remote_timeout=httpx.Timeout(30.0),
                                session_id="test-session-123",
                                provider="opencode-go",
                                entry="opencode-go-2-deepseek",
                            )
                            _ = [chunk async for chunk in result.body_iterator]

    errors = [m for m in warning_messages if "Stream error:" in m]
    assert errors, "expected a Stream error log line"
    assert any(
        "entry=opencode-go-2-deepseek" in m for m in errors
    ), f"Stream error line must include entry=, got: {errors}"


# ═══════════════════════════════════════════════════════════════════════════
# proxy_to_remote: entry name derived from model_config
# ═══════════════════════════════════════════════════════════════════════════


def _make_streaming_request():
    """Create a mock Request for the streaming path of proxy_to_remote."""
    req = MagicMock(spec=Request)
    req.method = "POST"
    req.url.path = "/v1/chat/completions"
    req.headers = {"content-type": "application/json"}
    req.body = AsyncMock(
        return_value=json.dumps({"model": "deepseek-v4-flash", "stream": True}).encode()
    )
    return req


@pytest.mark.asyncio
async def test_proxy_to_remote_passes_entry_from_model_config():
    """proxy_to_remote passes model_config['name'] as entry= (AC1 integration)."""
    import proxy.proxy_remote as pr

    mock_req = _make_streaming_request()
    model_config = {
        "endpoint": "https://api.example.com/v1",
        "type": "remote",
        "name": "opencode-go-2-deepseek",
        "provider": "opencode-go",
        "model": "deepseek-v4-flash",
    }

    with patch.object(pr, "_srv") as mock_srv:
        mock_srv.return_value.config = {"server": {}}
        mock_srv.return_value.logger = MagicMock()
        mock_srv.return_value.current_model = "deepseek-v4-flash"
        mock_srv.return_value._remote_http_client = None

        with patch.object(pr, "normalize_upstream_request_headers", return_value=dict(mock_req.headers)):
            with patch.object(pr, "log_request"):
                with patch.object(pr, "_handle_remote_streaming", AsyncMock()) as mock_handle:
                    with patch.object(pr, "_handle_remote_non_streaming", AsyncMock()):
                        await pr.proxy_to_remote(
                            request=mock_req,
                            path="v1/chat/completions",
                            model_config=model_config,
                        )

    assert mock_handle.called, "_handle_remote_streaming should be called for streaming requests"
    call = mock_handle.call_args
    assert call.kwargs.get("entry") == "opencode-go-2-deepseek", (
        f"entry must be model_config['name'], got: {call.kwargs.get('entry')}"
    )


@pytest.mark.asyncio
async def test_proxy_to_remote_entry_none_when_config_has_no_name():
    """Without a name key in model_config, entry is None (field omitted)."""
    import proxy.proxy_remote as pr

    mock_req = _make_streaming_request()
    model_config = {
        "endpoint": "https://api.example.com/v1",
        "type": "remote",
        "provider": "opencode-go",
        "model": "deepseek-v4-flash",
    }

    with patch.object(pr, "_srv") as mock_srv:
        mock_srv.return_value.config = {"server": {}}
        mock_srv.return_value.logger = MagicMock()
        mock_srv.return_value.current_model = "deepseek-v4-flash"
        with patch.object(pr, "normalize_upstream_request_headers", return_value=dict(mock_req.headers)):
            with patch.object(pr, "log_request"):
                with patch.object(pr, "_handle_remote_streaming", AsyncMock()) as mock_handle:
                    with patch.object(pr, "_handle_remote_non_streaming", AsyncMock()):
                        await pr.proxy_to_remote(
                            request=mock_req,
                            path="v1/chat/completions",
                            model_config=model_config,
                        )

    assert mock_handle.called, "_handle_remote_streaming should be called for streaming requests"
    call = mock_handle.call_args
    assert call.kwargs.get("entry", None) is None, (
        "entry must be None when model_config has no name key"
    )
