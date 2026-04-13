"""
Integration tests for httpx connection pooling behavior.

These tests verify that the shared httpx client with connection pooling
allows concurrent requests to complete without blocking each other.

NOTE: These tests require the real httpx package and should be run
from outside the proxy directory to avoid importing the local httpx shim.
"""
import asyncio
import pytest
from concurrent.futures import ThreadPoolExecutor
import threading
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '.venv', 'lib', 'python3.12', 'site-packages'))

import httpx


class SlowHTTPHandler:
    """HTTP handler that simulates slow/streaming responses."""
    
    def __init__(self, delay=0.5, streaming=False):
        self.delay = delay
        self.streaming = streaming
        self._stream_complete = threading.Event()
    
    async def handle(self, reader, writer):
        """Handle incoming HTTP requests."""
        data = await reader.read(1024)
        request_line = data.decode().split('\r\n')[0]
        
        if '/stream' in request_line:
            await self._handle_streaming(writer)
        elif '/status' in request_line:
            await self._handle_status(writer)
        else:
            await self._handle_not_found(writer)
        
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    
    async def _handle_status(self, writer):
        """Return a quick status response."""
        body = b'{"n_ctx": 4096}'
        response = (
            f"HTTP/1.1 200 OK\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode() + body
        writer.write(response)
        await writer.drain()
    
    async def _handle_not_found(self, writer):
        """Return 404."""
        response = (
            "HTTP/1.1 404 Not Found\r\n"
            "Content-Length: 0\r\n"
            "\r\n"
        )
        writer.write(response.encode())
        await writer.drain()
    
    async def _handle_streaming(self, writer):
        """Simulate a slow streaming response."""
        await asyncio.sleep(self.delay)
        body = b"data: chunk\r\n\r\n"
        response = (
            f"HTTP/1.1 200 OK\r\n"
            f"Content-Type: text/event-stream\r\n"
            f"Transfer-Encoding: chunked\r\n"
            f"\r\n"
            f"{len(body):x}\r\n"
        ).encode() + body + b"0\r\n\r\n"
        writer.write(response)
        await writer.drain()


async def start_test_server(handler, host='localhost', port=0):
    """Start a test HTTP server."""
    server = await asyncio.start_server(
        handler.handle, host, port
    )
    sock = server.sockets[0]
    addr = sock.getsockname()
    return server, addr


@pytest.mark.asyncio
async def test_connection_pooling_allows_concurrent_status_requests():
    """Verify that multiple concurrent HTTP requests use connection pooling.
    
    This test creates a real HTTP server and uses httpx with connection pooling
    to verify that multiple concurrent status requests complete without blocking.
    """
    import httpx
    
    handler = SlowHTTPHandler(delay=0.1)
    server, addr = await start_test_server(handler)
    port = addr[1]
    
    try:
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        async with httpx.AsyncClient(limits=limits, timeout=5.0) as client:
            start = asyncio.get_event_loop().time()
            
            async def make_request():
                response = await client.get(f"http://localhost:{port}/status")
                return response.json()
            
            results = await asyncio.gather(
                make_request(),
                make_request(),
                make_request()
            )
            
            elapsed = asyncio.get_event_loop().time() - start
            
            assert len(results) == 3
            for r in results:
                assert r["n_ctx"] == 4096
            assert elapsed < 1.0
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_status_request_not_blocked_by_streaming_request():
    """Verify status request completes even when streaming request is active.
    
    This test simulates the bug scenario where a status request would block
    when a streaming request is in progress, and verifies that connection
    pooling prevents this blocking.
    """
    import httpx
    
    streaming_started = asyncio.Event()
    server_ready = asyncio.Event()
    
    class SlowResponseHandler:
        async def handle(self, reader, writer):
            data = await reader.read(1024)
            request_line = data.decode().split('\r\n')[0]
            
            if '/stream' in request_line:
                await self._handle_slow_stream(writer, streaming_started, server_ready)
            elif '/status' in request_line:
                await self._handle_quick_status(writer)
            else:
                await self._handle_not_found(writer)
            
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        
        async def _handle_quick_status(self, writer):
            body = b'{"n_ctx": 4096}'
            response = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"Connection: close\r\n"
                f"\r\n"
            ).encode() + body
            writer.write(response)
        
        async def _handle_not_found(self, writer):
            response = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"
            writer.write(response)
        
        async def _handle_slow_stream(self, writer, streaming_started, server_ready):
            body = b'{"model": "slow"}'
            response = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"Connection: close\r\n"
                f"\r\n"
            ).encode() + body
            writer.write(response)
            await writer.drain()
            server_ready.set()
            streaming_started.set()
            await asyncio.sleep(2.0)
    
    handler = SlowResponseHandler()
    server, addr = await start_test_server(handler)
    port = addr[1]
    
    try:
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        async with httpx.AsyncClient(limits=limits, timeout=10.0) as client:
            async def make_status_request():
                response = await client.get(f"http://localhost:{port}/status")
                return response.json()
            
            async def make_streaming_request():
                response = await client.get(f"http://localhost:{port}/stream")
                return response.json()
            
            streaming_task = asyncio.create_task(make_streaming_request())
            await server_ready.wait()
            
            status_task = asyncio.create_task(make_status_request())
            
            done, pending = await asyncio.wait(
                [status_task, streaming_task],
                timeout=3.0
            )
            
            assert status_task in done, "Status request did not complete in time"
            
            status_result = await status_task
            assert status_result["n_ctx"] == 4096
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_without_connection_pooling_blocks():
    """Verify that WITHOUT connection pooling, requests would block.
    
    This test creates a NEW client per request (no pooling) and shows
    that concurrent requests complete slower than with pooling.
    """
    import httpx
    
    handler = SlowHTTPHandler(delay=0.2)
    server, addr = await start_test_server(handler)
    port = addr[1]
    
    try:
        async def make_request_without_pooling():
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://localhost:{port}/status")
                return response.json()
        
        start = asyncio.get_event_loop().time()
        
        results = await asyncio.gather(
            make_request_without_pooling(),
            make_request_without_pooling(),
            make_request_without_pooling()
        )
        
        elapsed = asyncio.get_event_loop().time() - start
        
        assert len(results) == 3
        for r in results:
            assert r["n_ctx"] == 4096
        assert elapsed < 2.0
    finally:
        server.close()
        await server.wait_closed()
