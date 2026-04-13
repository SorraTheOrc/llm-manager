"""
Tests verifying that the real httpx AsyncClient does NOT block the event loop.

These tests prove the root cause fix: the local httpx/ shim used synchronous
requests.get() inside async functions, which blocked the entire asyncio event
loop. With the real httpx package, all HTTP calls are truly async and the
event loop remains responsive.

The key test starts a slow HTTP server and verifies that a status request
completes promptly even while a long-running streaming request is in progress.
If httpx were using synchronous blocking calls, the status request would hang
until the streaming request finished.
"""
import asyncio
import time
import pytest
import httpx


async def _run_slow_server(ready_event: asyncio.Event, port_holder: list):
    """Start an asyncio TCP server that simulates a slow llama-server.

    /status  -> responds instantly with JSON
    /stream  -> delays 3 seconds before responding (simulates LLM inference)
    """

    async def handle_client(reader, writer):
        data = await reader.read(4096)
        request_line = data.decode(errors="replace").split("\r\n")[0]

        if "/status" in request_line:
            body = b'{"n_ctx": 4096, "status": "ok"}'
            resp = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"Connection: close\r\n"
                f"\r\n"
            ).encode() + body
            writer.write(resp)
            await writer.drain()
        elif "/stream" in request_line:
            # Simulate a long-running LLM inference request
            await asyncio.sleep(3.0)
            body = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'
            resp = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: text/event-stream\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"Connection: close\r\n"
                f"\r\n"
            ).encode() + body
            writer.write(resp)
            await writer.drain()
        else:
            writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()

        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    port_holder.append(port)
    ready_event.set()

    async with server:
        await server.serve_forever()


@pytest.mark.asyncio
async def test_status_not_blocked_by_streaming_with_real_httpx():
    """Prove that a status request completes in <1s while a 3s stream is in flight.

    This is the exact scenario that was broken: the proxy would call
    query_llama_status() while llama-server was busy with a streaming request.
    With the old httpx shim (synchronous requests.get()), the status call
    would block the event loop for the full 3 seconds. With real httpx,
    the status call completes independently.
    """
    ready = asyncio.Event()
    port_holder = []
    server_task = asyncio.create_task(_run_slow_server(ready, port_holder))

    try:
        await asyncio.wait_for(ready.wait(), timeout=5.0)
        port = port_holder[0]
        base_url = f"http://127.0.0.1:{port}"

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        ) as client:

            # Start a slow streaming request (takes 3s on the server side)
            stream_task = asyncio.create_task(client.get(f"{base_url}/stream"))

            # Give a moment for the stream request to reach the server
            await asyncio.sleep(0.1)

            # Now make a status request - this MUST complete quickly
            t0 = time.monotonic()
            status_resp = await client.get(f"{base_url}/status")
            elapsed = time.monotonic() - t0

            assert status_resp.status_code == 200
            data = status_resp.json()
            assert data["status"] == "ok"
            assert data["n_ctx"] == 4096

            # The critical assertion: status completed in well under the 3s
            # stream delay, proving the event loop was NOT blocked.
            assert elapsed < 1.0, (
                f"Status request took {elapsed:.2f}s - event loop appears blocked! "
                f"Expected <1s since status endpoint responds instantly."
            )

            # Clean up: cancel or await the streaming task
            stream_task.cancel()
            try:
                await stream_task
            except (asyncio.CancelledError, Exception):
                pass

    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_concurrent_status_requests_complete_promptly():
    """Verify that 5 concurrent status requests all complete within 2 seconds.

    With the old blocking shim, concurrent requests would serialize on
    the event loop, each blocking the next. With real async httpx, they
    all run concurrently.
    """
    ready = asyncio.Event()
    port_holder = []
    server_task = asyncio.create_task(_run_slow_server(ready, port_holder))

    try:
        await asyncio.wait_for(ready.wait(), timeout=5.0)
        port = port_holder[0]
        base_url = f"http://127.0.0.1:{port}"

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(5.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        ) as client:

            t0 = time.monotonic()
            results = await asyncio.gather(*[
                client.get(f"{base_url}/status") for _ in range(5)
            ])
            elapsed = time.monotonic() - t0

            for resp in results:
                assert resp.status_code == 200
                assert resp.json()["n_ctx"] == 4096

            # All 5 should complete almost simultaneously since they're async
            assert elapsed < 2.0, (
                f"5 concurrent status requests took {elapsed:.2f}s - "
                f"expected <2s with true async HTTP."
            )

    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_real_httpx_is_not_the_shim():
    """Verify that the imported httpx is the real package, not the local shim.

    This test guards against regression: if someone reintroduces an httpx/
    directory at the project root, this test will catch it.
    """
    assert hasattr(httpx, "Limits"), "httpx.Limits missing - are you using the local shim?"
    assert hasattr(httpx, "Timeout"), "httpx.Timeout missing - are you using the local shim?"
    assert hasattr(httpx, "ASGITransport"), "httpx.ASGITransport missing - are you using the local shim?"
    assert "test_httpx_shim" not in httpx.__file__, (
        f"httpx resolved to the test shim: {httpx.__file__}"
    )
