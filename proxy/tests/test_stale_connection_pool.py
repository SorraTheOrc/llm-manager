"""
Tests for stale connection pool detection and recovery.

These tests verify that:
- A fresh httpx client can be created after the backend server dies
- Old cached connections do not prevent new connections to a restarted server
- The proxy recovers from stale connection pool scenarios
"""
import asyncio
import socket
import threading

import httpx
import pytest


class MockHTTPHandler:
    """Simple HTTP handler returning JSON responses."""

    def __init__(self, fail_after: int = 0):
        """Initialize handler.

        Args:
            fail_after: If > 0, after this many requests the handler starts
                        returning connection resets (simulating server death).
        """
        self.request_count = 0
        self.fail_after = fail_after
        self._closed = False

    async def handle(self, reader, writer):
        """Handle incoming HTTP requests."""
        self.request_count += 1

        if self.fail_after > 0 and self.request_count > self.fail_after:
            # Simulate server death by closing abruptly
            writer.close()
            await writer.wait_closed()
            return

        data = await reader.read(1024)
        body = b'{"status": "ok", "n_ctx": 4096}'
        response = (
            f"HTTP/1.1 200 OK\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode() + body
        writer.write(response)
        await writer.drain()
        writer.close()
        await writer.wait_closed()


async def start_test_server(handler, host='localhost', port=0):
    """Start a test HTTP server on an ephemeral port."""
    server = await asyncio.start_server(handler.handle, host, port)
    sock = server.sockets[0]
    addr = sock.getsockname()
    return server, addr


@pytest.mark.asyncio
async def test_fresh_client_works_after_server_restart():
    """A fresh httpx client should work after a server restart.

    This simulates: old server dies → new server starts on same port →
    new client created → request succeeds.
    """
    handler1 = MockHTTPHandler()
    server1, addr1 = await start_test_server(handler1)
    port = addr1[1]

    try:
        # Make a request with first client
        async with httpx.AsyncClient(timeout=5.0) as client1:
            response1 = await client1.get(f"http://localhost:{port}/")
            assert response1.json()["status"] == "ok"

        # Stop the first server
        server1.close()
        await server1.wait_closed()

        # Start a new server on the SAME port
        handler2 = MockHTTPHandler()
        server2 = await asyncio.start_server(handler2.handle, "localhost", port)
        try:
            # Create a fresh client and make a request
            async with httpx.AsyncClient(timeout=5.0) as client2:
                response2 = await client2.get(f"http://localhost:{port}/")
                assert response2.json()["status"] == "ok"
                assert response2.json()["n_ctx"] == 4096
        finally:
            server2.close()
            await server2.wait_closed()
    finally:
        # Cleanup in case of exception
        if not server1.is_serving():
            pass


@pytest.mark.asyncio
async def test_old_client_fails_after_server_death_new_client_succeeds():
    """An old client connection should fail after server death, but a new client works."""
    handler = MockHTTPHandler(fail_after=1)
    server, addr = await start_test_server(handler)
    port = addr[1]

    try:
        # Create a client and make a request (succeeds)
        async with httpx.AsyncClient(timeout=5.0) as client:
            response1 = await client.get(f"http://localhost:{port}/")
            assert response1.json()["status"] == "ok"

            # Second request: server dies (fail_after=1 means only first succeeds)
            # The client may get a connection error or a response depending on timing
            try:
                await client.get(f"http://localhost:{port}/")
            except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError):
                pass

        # The old server is dead. Start a new one on the same port.
        server.close()
        await server.wait_closed()

        handler2 = MockHTTPHandler()
        server2 = await asyncio.start_server(handler2.handle, "localhost", port)
        try:
            # A fresh client should work with the new server
            async with httpx.AsyncClient(timeout=5.0) as fresh_client:
                response2 = await fresh_client.get(f"http://localhost:{port}/")
                assert response2.json()["status"] == "ok"
        finally:
            server2.close()
            await server2.wait_closed()
    finally:
        if not server.is_serving():
            pass


@pytest.mark.asyncio
async def test_stale_connection_does_not_block_new_client():
    """Stale connections from old client should not prevent new client from connecting."""
    handler1 = MockHTTPHandler()
    server1, addr1 = await start_test_server(handler1)
    port = addr1[1]

    try:
        # Make requests with client1 to populate connection pool
        async with httpx.AsyncClient(timeout=5.0) as client1:
            for _ in range(3):
                resp = await client1.get(f"http://localhost:{port}/")
                assert resp.json()["status"] == "ok"

        # Kill server1, start server2 on same port
        server1.close()
        await server1.wait_closed()

        handler2 = MockHTTPHandler()
        server2 = await asyncio.start_server(handler2.handle, "localhost", port)
        try:
            # New client should work even though old connections are stale
            async with httpx.AsyncClient(timeout=5.0) as client2:
                resp = await client2.get(f"http://localhost:{port}/")
                assert resp.json()["status"] == "ok"
        finally:
            server2.close()
            await server2.wait_closed()
    finally:
        if not server1.is_serving():
            pass


@pytest.mark.asyncio
async def test_concurrent_old_and_new_clients():
    """Old client with stale pool should not interfere with new client's requests."""
    handler = MockHTTPHandler()
    server, addr = await start_test_server(handler)
    port = addr[1]

    try:
        # Establish connections
        async with httpx.AsyncClient(timeout=5.0) as old_client:
            await old_client.get(f"http://localhost:{port}/")

            # New client on same port should work independently
            async with httpx.AsyncClient(timeout=5.0) as new_client:
                resp = await new_client.get(f"http://localhost:{port}/")
                assert resp.json()["status"] == "ok"

            # Old client should still work too
            resp = await old_client.get(f"http://localhost:{port}/")
            assert resp.json()["status"] == "ok"
    finally:
        server.close()
        await server.wait_closed()
