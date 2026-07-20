import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

pytestmark = pytest.mark.refactor_parity


@pytest.fixture
def app():
    """Import the FastAPI app (lazy to avoid circular imports)."""
    from proxy.server import app
    return app


@pytest.fixture
def transport(app):
    """ASGI transport for the proxy app."""
    return httpx.ASGITransport(app=app)


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory with dummy log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "proxy.log"
        log_path.write_text("line1\nline2\nline3\n")
        llama_path = Path(tmpdir) / "llama-server.log"
        llama_path.write_text("llama line 1\nllama line 2\n")
        yield tmpdir


@pytest.mark.asyncio
async def test_resolve_log_path():
    """Test the _resolve_log_path helper function."""
    from proxy.server import _resolve_log_path

    proxy_path = _resolve_log_path("proxy")
    assert "proxy.log" in str(proxy_path)

    llama_path = _resolve_log_path("llama")
    assert "llama-server.log" in str(llama_path)


@pytest.mark.asyncio
async def test_resolve_log_path_default():
    """Test that default source is proxy."""
    from proxy.server import _resolve_log_path

    # Invalid source should default to proxy
    default_path = _resolve_log_path("invalid")
    assert "proxy.log" in str(default_path)


# ---------------------------------------------------------------------------
# Integration tests via ASGI transport (non-streaming endpoints)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_logs_endpoint_returns_200(transport):
    """Smoke test: GET /logs returns HTTP 200."""
    from proxy import server as srv_module

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as ac:
        with patch.object(srv_module, "config", {"server": {"llama_router_mode": False}}):
            with patch.object(srv_module, "request_counts", {}):
                with patch.object(srv_module, "token_counts", {"total_sent": 0}):
                    with patch.object(
                        srv_module, "counts_lock", AsyncMock()
                    ):
                        with patch.object(
                            srv_module, "token_lock", AsyncMock()
                        ):
                            resp = await ac.get("/logs")

    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_log_tail_missing_file_returns_error(transport):
    """GET /logs/tail for a non-existent file returns an error SSE message (stream closes)."""
    from proxy import server as srv_module

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as ac:
        with patch.object(srv_module, "log_dir", Path("/nonexistent/logs")):
            with patch.object(srv_module, "log_tail_clients", set()):
                resp = await ac.get("/logs/tail?lines=5&source=proxy")

    assert resp.status_code == 200
    body = resp.text
    assert '"error"' in body
    assert '"log_not_found"' in body


# ---------------------------------------------------------------------------
# Direct handler tests for open-ended SSE streams
# The httpx ASGI transport cannot handle open-ended StreamingResponses
# because it buffers the entire response. These tests call the handler
# directly to verify the SSE message format.
# ---------------------------------------------------------------------------


async def _collect_sse_first_chunk(handler, request, lines=2, source="proxy"):
    """Call an SSE handler directly and collect the first chunk."""

    response = await handler(request, lines=lines, source=source)
    try:
        async for chunk in response.body_iterator:
            return chunk
    finally:
        # Ensure we cancel the generator to prevent pending task warnings
        await response.body_iterator.aclose()
    return ""


async def _make_starlette_request():
    """Create a minimal Starlette Request for direct handler calls."""
    from starlette.requests import Request as StarletteRequest
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/logs/tail",
        "query_string": b"",
        "headers": [],
        "server": ("testserver", 80),
    }
    return StarletteRequest(scope)


@pytest.mark.asyncio
async def test_log_tail_proxy_source_returns_initial(temp_log_dir):
    """GET /logs/tail?lines=2&source=proxy returns initial SSE with proxy log lines."""
    from proxy.ui import tail_logs

    from proxy import server as srv_module

    request = await _make_starlette_request()

    with patch.object(srv_module, "log_dir", Path(temp_log_dir)):
        with patch.object(srv_module, "log_tail_clients", set()):
            chunk = await _collect_sse_first_chunk(tail_logs, request, lines=2, source="proxy")

    assert chunk is not None
    assert '"initial"' in chunk
    assert '"source": "proxy"' in chunk
    # Should include the last 2 lines
    assert "line2" in chunk
    assert "line3" in chunk


@pytest.mark.asyncio
async def test_log_tail_llama_source_returns_initial(temp_log_dir):
    """GET /logs/tail?lines=1&source=llama returns initial SSE with llama log lines."""
    from proxy.ui import tail_logs

    from proxy import server as srv_module

    request = await _make_starlette_request()

    with patch.object(srv_module, "log_dir", Path(temp_log_dir)):
        with patch.object(srv_module, "log_tail_clients", set()):
            chunk = await _collect_sse_first_chunk(tail_logs, request, lines=1, source="llama")

    assert chunk is not None
    assert '"initial"' in chunk
    assert '"source": "llama"' in chunk
    # Should include the last line of llama-server.log
    assert "llama line 2" in chunk


@pytest.mark.asyncio
async def test_log_tail_invalid_source_defaults_to_proxy(temp_log_dir):
    """Invalid source parameter defaults to 'proxy'."""
    from proxy.ui import tail_logs

    from proxy import server as srv_module

    request = await _make_starlette_request()

    with patch.object(srv_module, "log_dir", Path(temp_log_dir)):
        with patch.object(srv_module, "log_tail_clients", set()):
            chunk = await _collect_sse_first_chunk(tail_logs, request, lines=1, source="unknown")

    assert chunk is not None
    # Invalid source falls back to proxy, so source should be "proxy"
    assert '"source": "proxy"' in chunk
