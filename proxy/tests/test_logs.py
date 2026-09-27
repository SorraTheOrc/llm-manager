import asyncio
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


async def _collect_sse_first_chunk(handler, request, lines=2, source="proxy", **kwargs):
    """Call an SSE handler directly and collect the first chunk."""

    response = await handler(request, lines=lines, source=source, **kwargs)
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


# ---------------------------------------------------------------------------
# Slot-aware /logs/tail (LP-0MSHET5SI000LYSK)
# Optional `slot` / `session` query params filter the stream per slot without
# changing the existing source=proxy|llama behaviour.
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_slot_log_dir():
    """Temporary directory with slot-marked llama + proxy log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        llama_path = Path(tmpdir) / "llama-server.log"
        llama_path.write_text(
            "[57463] slot update_slots: id  2 | task 209403 | n_tokens = 16750, ...\n"
            "[57463] slot update_slots: id  3 | task 209410 | n_tokens = 9000, ...\n"
            "srv  log_server_r: server listening on port 8080\n"
            "[57463] slot      release: id  2 | task 209403\n"
        )
        proxy_path = Path(tmpdir) / "proxy.log"
        proxy_path.write_text(
            "slot_save success session=11111111-2222-3333-4444-555555555555 slot=2\n"
            "lease_renewed session=11111111-2222-3333-4444-555555555555\n"
            "dispatch line session=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee slot=none\n"
            "generic line without markers\n"
        )
        yield tmpdir


@pytest.mark.asyncio
async def test_log_tail_llama_slot_filter_initial(temp_slot_log_dir):
    """source=llama&slot=2 returns only the slot-2 llama lines initially."""
    from proxy.ui import tail_logs

    from proxy import server as srv_module

    request = await _make_starlette_request()

    with patch.object(srv_module, "log_dir", Path(temp_slot_log_dir)):
        with patch.object(srv_module, "log_tail_clients", set()):
            chunk = await _collect_sse_first_chunk(tail_logs, request, lines=10, source="llama", slot=2)

    assert chunk is not None
    assert '"initial"' in chunk
    assert '"source": "llama"' in chunk
    assert '"slot": 2' in chunk
    assert "id  2" in chunk
    # Slot 3 and non-slot lines must be filtered out
    assert "id  3" not in chunk
    assert "log_server_r" not in chunk


@pytest.mark.asyncio
async def test_log_tail_llama_slot_filter_excludes_other_slot(temp_slot_log_dir):
    """source=llama&slot=3 keeps only the slot-3 lines."""
    from proxy.ui import tail_logs

    from proxy import server as srv_module

    request = await _make_starlette_request()

    with patch.object(srv_module, "log_dir", Path(temp_slot_log_dir)):
        with patch.object(srv_module, "log_tail_clients", set()):
            chunk = await _collect_sse_first_chunk(tail_logs, request, lines=10, source="llama", slot=3)

    assert chunk is not None
    assert "id  3" in chunk
    assert "id  2" not in chunk


@pytest.mark.asyncio
async def test_log_tail_proxy_slot_filter_by_session(temp_slot_log_dir):
    """source=proxy&slot=2&session=<uuid> keeps only lines for that session."""
    from proxy.ui import tail_logs

    from proxy import server as srv_module

    request = await _make_starlette_request()

    with patch.object(srv_module, "log_dir", Path(temp_slot_log_dir)):
        with patch.object(srv_module, "log_tail_clients", set()):
            chunk = await _collect_sse_first_chunk(
                tail_logs, request, lines=10, source="proxy", slot=2,
                session="11111111-2222-3333-4444-555555555555",
            )

    assert chunk is not None
    assert '"source": "proxy"' in chunk
    assert "slot_save success" in chunk
    assert "lease_renewed" in chunk
    # Unmapped (slot=none) and marker-less lines must be excluded
    assert "slot=none" not in chunk
    assert "generic line" not in chunk


@pytest.mark.asyncio
async def test_log_tail_proxy_slot_filter_fallback_without_session(temp_slot_log_dir):
    """source=proxy&slot=2 without session uses the slot=<n> fallback."""
    from proxy.ui import tail_logs

    from proxy import server as srv_module

    request = await _make_starlette_request()

    with patch.object(srv_module, "log_dir", Path(temp_slot_log_dir)):
        with patch.object(srv_module, "log_tail_clients", set()):
            chunk = await _collect_sse_first_chunk(tail_logs, request, lines=10, source="proxy", slot=2)

    assert chunk is not None
    assert "slot_save success" in chunk
    # lease_renewed has no slot= marker and no session filter -> excluded
    assert "lease_renewed" not in chunk


@pytest.mark.asyncio
async def test_log_tail_without_slot_param_is_unfiltered(temp_slot_log_dir):
    """Omitting the slot param preserves the existing unfiltered behaviour."""
    from proxy.ui import tail_logs

    from proxy import server as srv_module

    request = await _make_starlette_request()

    with patch.object(srv_module, "log_dir", Path(temp_slot_log_dir)):
        with patch.object(srv_module, "log_tail_clients", set()):
            chunk = await _collect_sse_first_chunk(tail_logs, request, lines=10, source="llama")

    assert chunk is not None
    assert "id  2" in chunk
    assert "id  3" in chunk
    assert "log_server_r" in chunk
    assert '"slot"' not in chunk
# ---------------------------------------------------------------------------
# Combined per-slot tail + follow-loop starvation (LP-0MUJZHTCN006IZRW)
#
# The View Logs Slots tab used to open two SSE connections per slot (proxy and
# llama). Combined with the two raw-pane streams and the /events status stream
# that exceeded the browser's per-origin HTTP/1.1 connection budget, so most
# slot panes received no data. tail_slot_logs() now carries both sources for a
# slot over one connection.
#
# Separately, the follow loop used to `continue` after forwarding a
# counts/tokens update, starving the file-follow check while the queue was
# continuously non-empty (i.e. throughout active generation), so newly appended
# lines were never streamed.
# ---------------------------------------------------------------------------


async def _collect_sse_chunks(agen, count, timeout=2.0):
    """Collect up to *count* chunks from an async iterator (with timeout)."""
    chunks = []

    async def _collect():
        async for chunk in agen:
            chunks.append(chunk)
            if len(chunks) >= count:
                return

    try:
        await asyncio.wait_for(_collect(), timeout=timeout)
    except (TimeoutError, StopAsyncIteration):
        pass
    return chunks


@pytest.mark.asyncio
async def test_tail_slot_logs_returns_both_sources(temp_slot_log_dir):
    """A single connection carries the slot's proxy AND llama initial blocks."""
    from proxy.ui import tail_slot_logs

    from proxy import server as srv_module

    request = await _make_starlette_request()
    with patch.object(srv_module, "log_dir", Path(temp_slot_log_dir)):
        with patch.object(srv_module, "log_tail_clients", set()):
            response = await tail_slot_logs(request, lines=10, slot=2)
            chunks = await _collect_sse_chunks(response.body_iterator, 2)
            await response.body_iterator.aclose()

    assert len(chunks) == 2
    proxy_chunk = next(c for c in chunks if '"source": "proxy"' in c)
    llama_chunk = next(c for c in chunks if '"source": "llama"' in c)
    assert '"slot": 2' in proxy_chunk
    assert '"slot": 2' in llama_chunk
    assert "slot_save success" in proxy_chunk
    assert "id  2" in llama_chunk
    # Other slots must not leak into either pane.
    assert "id  3" not in llama_chunk
    assert "lease_renewed" not in proxy_chunk


@pytest.mark.asyncio
async def test_tail_slot_logs_streams_appended_lines(temp_slot_log_dir):
    """Newly appended lines from both sources are streamed after the initial block."""
    from proxy.ui import tail_slot_logs

    from proxy import server as srv_module

    request = await _make_starlette_request()
    llama_path = Path(temp_slot_log_dir) / "llama-server.log"
    with patch.object(srv_module, "log_dir", Path(temp_slot_log_dir)):
        with patch.object(srv_module, "log_tail_clients", set()):
            response = await tail_slot_logs(request, lines=10, slot=2)
            iterator = response.body_iterator
            # Drain the initial blocks (lets the follow loop record end offsets).
            await _collect_sse_chunks(iterator, 2, timeout=1.0)
            with open(llama_path, "a") as f:
                f.write("[57463] slot update_slots: id  2 | task 99 | n_tokens = 424242\n")
            chunks = await _collect_sse_chunks(iterator, 5, timeout=2.0)
            await iterator.aclose()

    assert any("424242" in c for c in chunks), chunks


@pytest.mark.asyncio
async def test_tail_logs_streams_while_counts_updates_continue(temp_log_dir):
    """The counts/tokens fast path must not starve the log-file follow check.

    Regression test for LP-0MUJZHTCN006IZRW: previously the loop `continue`d
    immediately after every counts update, so with a continuously non-empty
    queue (active generation) the file was never re-read.
    """
    from proxy.ui import tail_logs

    from proxy import server as srv_module

    request = await _make_starlette_request()
    log_path = Path(temp_log_dir) / "proxy.log"
    stop = asyncio.Event()

    with patch.object(srv_module, "log_dir", Path(temp_log_dir)):
        with patch.object(srv_module, "log_tail_clients", set()):
            response = await tail_logs(request, lines=1, source="proxy")
            iterator = response.body_iterator
            await _collect_sse_chunks(iterator, 1, timeout=1.0)

            async def _feeder():
                while not stop.is_set():
                    for q in list(srv_module.log_tail_clients):
                        try:
                            q.put_nowait({"counts": {"x": 1}})
                        except asyncio.QueueFull:
                            pass
                    await asyncio.sleep(0.02)

            feeder_task = asyncio.create_task(_feeder())
            try:
                await asyncio.sleep(0.4)
                with open(log_path, "a") as f:
                    f.write("fresh line 424242\n")
                chunks = await _collect_sse_chunks(iterator, 5, timeout=2.0)
            finally:
                stop.set()
                feeder_task.cancel()
                await iterator.aclose()

    assert any("424242" in c for c in chunks), chunks
