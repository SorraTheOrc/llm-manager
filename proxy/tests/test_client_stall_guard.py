"""Downstream write-stall guard (LP-0MVWZ9CH6006NW1H).

A client that stops reading does not look disconnected: TCP stays ESTABLISHED,
no ``http.disconnect`` is ever delivered, and ``is_disconnected()`` keeps
returning False (observed: two streams held 1.1 MB / 0.6 MB unsent for 30+
minutes). The ASGI ``send`` the streaming response awaits then never completes,
so the request, its task and its handler are held forever.

These tests pin the contract:

- AC1: a ``send`` that makes no progress within the deadline aborts the
  response (the request unwinds instead of hanging), bounded by the configured
  window plus one tick.
- AC2: the abort is reported with a distinguishable ``reason=client_stall``.
- AC3: writes after the abort are dropped instead of blocking a second time on
  the same full socket.
- AC4: ``/health`` exposes the ``client_stall`` counter so a wedged reader is
  visible without ``ss``.
- Fail-open: a non-HTTP scope, a disabled timeout and unparsable config all
  leave the untouched path exactly as it was.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from proxy.client_stall_guard import (
    CLIENT_STALL_SIGNAL,
    CLIENT_STALL_TIMEOUT_CONFIG_KEY,
    DEFAULT_CLIENT_STALL_TIMEOUT_SECONDS,
    STALL_SCOPE_KEY,
    STREAM_IDLE_TIMEOUT_CONFIG_KEY,
    ClientStallAbort,
    ClientStallGuardMiddleware,
    ClientStallState,
    client_stall_observed,
    client_stall_state,
    resolve_client_stall_timeout,
)
from proxy.observability import backend_signal_counts

STALL_TIMEOUT = 0.05
BODY = b"x" * 4096


class _Recorder:
    """Capture the guard's stall report instead of logging it."""

    def __init__(self) -> None:
        self.reports: list[tuple[ClientStallState, dict]] = []

    def __call__(self, state: ClientStallState, scope: dict) -> None:
        self.reports.append((state, scope))


def _http_scope(path: str = "/v1/chat/completions") -> dict:
    return {"type": "http", "path": path, "method": "POST"}


async def _receive() -> dict:
    await asyncio.Event().wait()  # never yields a disconnect
    return {}  # pragma: no cover


def _stalling_send(seen: list[dict]):
    """Accept the response start; block forever on any body write.

    This is the wedged-socket shape: the headers fit in the buffer, the body
    does not, and the reader never drains.
    """

    async def send(message: dict) -> None:
        seen.append(message)
        if message.get("type") == "http.response.body":
            await asyncio.Event().wait()

    return send


class TestConfigResolution:
    """The deadline is a config knob with a documented fallback chain."""

    def test_explicit_override_wins(self):
        config = {
            "server": {
                CLIENT_STALL_TIMEOUT_CONFIG_KEY: 45,
                STREAM_IDLE_TIMEOUT_CONFIG_KEY: 120,
            }
        }
        assert resolve_client_stall_timeout(config) == 45.0

    def test_falls_back_to_stream_idle_timeout(self):
        config = {"server": {STREAM_IDLE_TIMEOUT_CONFIG_KEY: 90}}
        assert resolve_client_stall_timeout(config) == 90.0

    def test_default_when_absent(self):
        assert (
            resolve_client_stall_timeout({"server": {}})
            == DEFAULT_CLIENT_STALL_TIMEOUT_SECONDS
        )
        assert resolve_client_stall_timeout(None) == DEFAULT_CLIENT_STALL_TIMEOUT_SECONDS
        assert resolve_client_stall_timeout({}) == DEFAULT_CLIENT_STALL_TIMEOUT_SECONDS

    def test_zero_disables_and_negative_clamps_to_zero(self):
        assert resolve_client_stall_timeout(
            {"server": {CLIENT_STALL_TIMEOUT_CONFIG_KEY: 0}}
        ) == 0.0
        assert resolve_client_stall_timeout(
            {"server": {CLIENT_STALL_TIMEOUT_CONFIG_KEY: -5}}
        ) == 0.0

    def test_unparsable_value_falls_through_to_the_fallback(self):
        config = {
            "server": {
                CLIENT_STALL_TIMEOUT_CONFIG_KEY: "not-a-number",
                STREAM_IDLE_TIMEOUT_CONFIG_KEY: 77,
            }
        }
        assert resolve_client_stall_timeout(config) == 77.0

    def test_health_exposes_the_counter(self):
        """AC4: the signal is present in the /health payload's source dict."""
        assert CLIENT_STALL_SIGNAL in backend_signal_counts


class TestStallAbort:
    """A stalled write abandons the response instead of hanging forever."""

    @pytest.mark.asyncio
    async def test_stalled_body_write_aborts_the_request(self):
        """AC1/AC2: the request unwinds within the window, reported as a stall."""
        recorder = _Recorder()
        seen: list[dict] = []
        scope = _http_scope()
        middleware = ClientStallGuardMiddleware(
            app=_stalling_app(seen), stall_timeout=STALL_TIMEOUT, record=recorder
        )

        # The assertion is the absence of a hang: this returns in ~STALL_TIMEOUT.
        await asyncio.wait_for(
            middleware(scope, _receive, _stalling_send(seen)), timeout=5.0
        )

        assert len(recorder.reports) == 1, "stall not reported"
        state, reported_scope = recorder.reports[0]
        assert state.aborted is True
        assert state.stall_seconds >= STALL_TIMEOUT * 0.9
        assert state.blocked_body_bytes == len(BODY)
        assert reported_scope is scope
        assert client_stall_observed(_TypeWithScope(scope)) is True

    @pytest.mark.asyncio
    async def test_stall_is_logged_with_a_distinguishable_reason(self, caplog):
        """AC2: the log line carries reason=client_stall for operator greps."""
        seen: list[dict] = []
        middleware = ClientStallGuardMiddleware(
            app=_stalling_app(seen), stall_timeout=STALL_TIMEOUT
        )
        with caplog.at_level(logging.WARNING, logger="llama-proxy.client_stall"):
            await asyncio.wait_for(
                middleware(_http_scope(), _receive, _stalling_send(seen)), timeout=5.0
            )

        stall_lines = [r for r in caplog.records if "client_stall" in r.getMessage()]
        assert stall_lines, "no client_stall log line emitted"
        message = stall_lines[0].getMessage()
        assert "reason=client_stall" in message
        assert "/v1/chat/completions" in message
        assert f"blocked_body_bytes={len(BODY)}" in message

    @pytest.mark.asyncio
    async def test_stall_increments_the_health_counter(self, monkeypatch):
        """AC4: one stall == one ``client_stall`` signal (production recorder)."""
        monkeypatch.setitem(backend_signal_counts, CLIENT_STALL_SIGNAL, 0)
        seen: list[dict] = []
        middleware = ClientStallGuardMiddleware(
            app=_stalling_app(seen), stall_timeout=STALL_TIMEOUT
        )
        await asyncio.wait_for(
            middleware(_http_scope(), _receive, _stalling_send(seen)), timeout=5.0
        )
        assert backend_signal_counts[CLIENT_STALL_SIGNAL] == 1

    @pytest.mark.asyncio
    async def test_writes_after_the_abort_are_dropped_not_blocked(self):
        """AC3: the unwind cannot block on the same full socket again."""
        recorder = _Recorder()
        seen: list[dict] = []

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            try:
                await send({"type": "http.response.body", "body": BODY, "more_body": True})
            except ClientStallAbort:
                for _ in range(3):
                    await send(
                        {"type": "http.response.body", "body": BODY, "more_body": True}
                    )

        middleware = ClientStallGuardMiddleware(
            app=app, stall_timeout=STALL_TIMEOUT, record=recorder
        )
        await asyncio.wait_for(
            middleware(_http_scope(), _receive, _stalling_send(seen)), timeout=5.0
        )

        state, _ = recorder.reports[0]
        assert state.dropped_writes == 3
        # Only the response start and the first body write ever reached the
        # socket layer; the three post-abort writes were dropped before it.
        assert [m["type"] for m in seen] == [
            "http.response.start",
            "http.response.body",
        ]


class TestFailOpenPaths:
    """Everything that is not a stalled HTTP write behaves exactly as before."""

    @pytest.mark.asyncio
    async def test_non_http_scope_passes_through(self):
        seen = []
        calls = []

        async def app(scope, receive, send):
            calls.append(scope["type"])
            await send({"type": "websocket.close"})

        middleware = ClientStallGuardMiddleware(app=app, stall_timeout=STALL_TIMEOUT)
        await middleware({"type": "websocket"}, _receive, _async_sink(seen))

        assert calls == ["websocket"]
        assert seen == [{"type": "websocket.close"}]

    @pytest.mark.asyncio
    async def test_zero_timeout_disables_the_guard(self):
        """A disabled guard must not arm a deadline (or attach state)."""
        seen: list[dict] = []
        scope = _http_scope()

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})

        middleware = ClientStallGuardMiddleware(app=app, stall_timeout=0)
        await middleware(scope, _receive, _async_sink(seen))

        assert STALL_SCOPE_KEY not in scope
        assert [m["type"] for m in seen] == ["http.response.start"]

    @pytest.mark.asyncio
    async def test_healthy_send_is_untouched_and_state_is_exposed(self):
        """A normal response completes unchanged; state is readable mid-request."""
        seen: list[dict] = []
        scope = _http_scope()
        observed = {}

        async def app(scope, receive, send):
            observed["state"] = client_stall_state(_TypeWithScope(scope))
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": BODY, "more_body": False})

        middleware = ClientStallGuardMiddleware(app=app, stall_timeout=STALL_TIMEOUT)
        await middleware(scope, _receive, _async_sink(seen))

        assert isinstance(observed["state"], ClientStallState)
        assert observed["state"].aborted is False
        assert [m["type"] for m in seen] == [
            "http.response.start",
            "http.response.body",
        ]
        assert client_stall_observed(_TypeWithScope(scope)) is False


class TestMiddlewareRegistration:
    """The guard must be the outermost wrapper to see every ``send``."""

    def test_guard_is_registered_outside_the_disconnect_reaper(self):
        from proxy.disconnect_reaper import DisconnectReaperMiddleware
        from proxy.server import app

        classes = [entry.cls for entry in app.user_middleware]
        assert ClientStallGuardMiddleware in classes
        assert classes.index(ClientStallGuardMiddleware) < classes.index(
            DisconnectReaperMiddleware
        ), "the stall guard must wrap the reaper to see every send"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TypeWithScope:
    """Minimal ``Request``-shaped object (the helpers only touch ``.scope``)."""

    def __init__(self, scope: dict) -> None:
        self.scope = scope


def _stalling_app(seen: list[dict]):
    """A streaming app: response start, then one body write that stalls."""

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": BODY, "more_body": True})

    return app


def _async_sink(seen: list[dict]):
    async def send(message: dict) -> None:
        seen.append(message)

    return send


class TestRealTransport:
    """End-to-end over a real uvicorn transport (not a fake ``send``).

    The unit tests above prove the deadline logic against a coroutine that
    never returns. This proves the *transport* shape that was actually observed
    on 2026-10-31: a kernel-level socket whose reader has stopped, which pauses
    uvicorn's flow control and blocks the streaming write. A real uvicorn server
    is started on an ephemeral port with the guard in front of a streaming app;
    the client reads the headers and then stops reading entirely.

    This is the live-pair verification for LP-0MVWZ9CH6006NW1H AC5, run against
    a scratch server so the production proxy (and the ~44 sessions streaming
    through it) is never disturbed.
    """

    @pytest.mark.asyncio
    async def test_a_reader_that_stops_reading_is_disconnected(self, caplog):
        import contextlib
        import socket
        import uvicorn

        stall_seconds = 0.3
        chunk = b"y" * 65536
        chunks = 512  # 32 MiB: far beyond any kernel socket buffer

        async def app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")],
                }
            )
            for _ in range(chunks):
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        # Default recorder: this test asserts the production log line (the
        # counter path is covered above).
        guarded = ClientStallGuardMiddleware(app=app, stall_timeout=stall_seconds)

        # Pick a free port, then hand it to uvicorn.
        with contextlib.closing(socket.socket()) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        config = uvicorn.Config(
            guarded,
            host="127.0.0.1",
            port=port,
            log_config=None,
            access_log=False,
            lifespan="off",
        )
        server = uvicorn.Server(config)
        serve_task = asyncio.create_task(server.serve())
        try:
            for _ in range(200):
                if server.started:
                    break
                await asyncio.sleep(0.02)
            assert server.started, "scratch uvicorn never started"

            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            try:
                # Capture from before the request: the stall may fire the moment
                # the client stops reading.
                with caplog.at_level(logging.WARNING, logger="llama-proxy.client_stall"):
                    writer.write(
                        b"GET /stream HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                        b"Accept: text/event-stream\r\n\r\n"
                    )
                    await writer.drain()
                    # Read the headers only, then stop reading: the socket fills
                    # and uvicorn pauses the flow, exactly like the wedged sockets.
                    head = await asyncio.wait_for(
                        reader.readuntil(b"\r\n\r\n"), timeout=10
                    )
                    assert b"200" in head

                    for _ in range(200):
                        if any(
                            "reason=client_stall" in r.getMessage()
                            for r in caplog.records
                        ):
                            break
                        await asyncio.sleep(0.05)
                stall_lines = [
                    r for r in caplog.records if "reason=client_stall" in r.getMessage()
                ]
                assert stall_lines, (
                    "a reader that stopped reading was never disconnected: "
                    f"{[r.getMessage() for r in caplog.records][-5:]}"
                )
            finally:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
        finally:
            server.should_exit = True
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(serve_task, timeout=10)
