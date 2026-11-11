"""Abort a response when the reader stops accepting bytes (LP-0MVWZ9CH6006NW1H).

Why this exists
---------------
``disconnect_reaper`` covers a client that has *gone*: it polls
``request.is_disconnected()`` and cancels the request task. It cannot cover a
client that is still connected but **stalled** — a reader that stops draining
its socket while TCP stays ESTABLISHED (observed 2026-10-31: two streams from a
remote host sat with 1.1 MB / 0.6 MB unsent, 95% of their lifetime
receiver-window limited, no ACK for 35 minutes; ``is_disconnected()`` is False
throughout, because no ``http.disconnect`` ever arrives).

In that state the ASGI ``send`` awaited by ``StreamingResponse`` simply never
completes: uvicorn's write buffer is full, the flow is paused, and the
handler blocks inside ``send``. The request, its task and its proxy handler are
held indefinitely and the session on the other end hangs with no error
anywhere.

What this does
--------------
:class:`ClientStallGuardMiddleware` wraps the ASGI ``send`` with a **write
progress deadline**: each individual ``send`` gets ``client_stall_timeout``
seconds to complete. A send still in flight after that deadline means the
reader is not accepting bytes, so the response is abandoned:

- the stall is recorded (``WARN`` with ``reason=client_stall``, stall duration
  and the size of the message that could not be written),
- the ``client_stall`` backend-signal counter is incremented (surfaced by
  ``/health``, so a wedged remote host is visible without ``ss``), and
- the request unwinds; the connection is closed by the server, which releases
  the handler and makes the client fail fast instead of hanging.

Every later ``send`` is dropped without waiting, so the unwind cannot block on
the same full socket a second time. uvicorn logs its own
``ASGI callable returned without completing response`` ERROR for the abandoned
response: that line is the expected consequence of the abort — grep
``client_stall`` for the cause.

The guard is fail-open by construction: a non-positive timeout disables it, and
any error while resolving configuration falls back to the documented default.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("llama-proxy.client_stall")

STALL_SCOPE_KEY = "llama_proxy.client_stall"
"""ASGI scope key holding the :class:`ClientStallState` for a request.

Present for every HTTP request the guard sees, so a handler (or its tests) can
report a stall verdict without importing the middleware instance.
"""

#: Default downstream write deadline (seconds). Matches the upstream
#: ``server.stream_idle_timeout_seconds`` so both directions of a stream are
#: bounded by the same budget by default.
DEFAULT_CLIENT_STALL_TIMEOUT_SECONDS = 120.0

#: ``server.<key>`` config override for the deadline.
CLIENT_STALL_TIMEOUT_CONFIG_KEY = "client_stall_timeout_seconds"

#: Upstream knob used as the fallback when the explicit override is absent.
STREAM_IDLE_TIMEOUT_CONFIG_KEY = "stream_idle_timeout_seconds"

#: Backend-signal counter name (surfaced by ``/health`` → ``backend_signals``).
CLIENT_STALL_SIGNAL = "client_stall"


class ClientStallError(Exception):
    """Raised internally when a stalled write aborts the response."""


@dataclass
class ClientStallState:
    """Per-request record of downstream write progress.

    ``aborted`` is the terminal verdict: once set, every later ``send`` is
    dropped (nothing can be delivered to a reader that is not reading) and no
    further deadline is ever armed.
    """

    stall_timeout: float
    aborted: bool = False
    stall_seconds: float = 0.0
    blocked_body_bytes: int = 0
    dropped_writes: int = 0

    def mark_aborted(self, stall_seconds: float, blocked_body_bytes: int) -> None:
        """Record a confirmed stall (idempotent)."""
        if self.aborted:
            return
        self.aborted = True
        self.stall_seconds = stall_seconds
        self.blocked_body_bytes = blocked_body_bytes

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable view (used by logs and tests)."""
        return {
            "aborted": self.aborted,
            "stall_seconds": round(self.stall_seconds, 3),
            "blocked_body_bytes": self.blocked_body_bytes,
            "dropped_writes": self.dropped_writes,
            "stall_timeout_seconds": self.stall_timeout,
        }


def client_stall_state(request: Any) -> ClientStallState | None:
    """Return the guard state attached to *request*, if any (fail-soft)."""
    try:
        state = request.scope.get(STALL_SCOPE_KEY)
    except Exception:
        return None
    return state if isinstance(state, ClientStallState) else None


def client_stall_observed(request: Any) -> bool:
    """True only when the guard confirmed a downstream stall for *request*.

    Fail-closed like ``client_disconnect_observed``: anything other than an
    explicit ``True`` reads as "not stalled".
    """
    state = client_stall_state(request)
    return bool(state is not None and state.aborted)


def resolve_client_stall_timeout(config: Any) -> float:
    """Resolve the downstream write deadline from *config*.

    Precedence: ``server.client_stall_timeout_seconds`` >
    ``server.stream_idle_timeout_seconds`` > :data:`DEFAULT_CLIENT_STALL_TIMEOUT_SECONDS`.
    ``0`` or a negative value disables the guard; an unparsable value falls
    back to the default (fail-open, never crash a request over config).
    """
    server_cfg = config.get("server", {}) if isinstance(config, dict) else {}
    for key in (CLIENT_STALL_TIMEOUT_CONFIG_KEY, STREAM_IDLE_TIMEOUT_CONFIG_KEY):
        if not isinstance(server_cfg, dict) or key not in server_cfg:
            continue
        value = server_cfg.get(key)
        try:
            timeout = float(value)
        except (TypeError, ValueError):
            continue
        return max(0.0, timeout)
    return DEFAULT_CLIENT_STALL_TIMEOUT_SECONDS


def _resolve_from_server() -> float:
    """Resolve the deadline from the running server's config (fail-open)."""
    try:
        from proxy.observability import _srv

        return resolve_client_stall_timeout(getattr(_srv(), "config", None))
    except Exception:
        return DEFAULT_CLIENT_STALL_TIMEOUT_SECONDS


def _record_stall(state: ClientStallState, scope: dict[str, Any]) -> None:
    """Log the abort and bump the health counter (both fail-soft)."""
    try:
        from proxy.observability import _record_backend_signal

        _record_backend_signal(CLIENT_STALL_SIGNAL)
    except Exception:
        pass
    try:
        logger.warning(
            "client_stall: aborting a response whose reader stopped accepting "
            "bytes reason=client_stall path=%s method=%s stall_seconds=%.1f "
            "blocked_body_bytes=%d dropped_writes=%d",
            scope.get("path", "?"),
            scope.get("method", "?"),
            state.stall_seconds,
            state.blocked_body_bytes,
            state.dropped_writes,
        )
    except Exception:
        pass


def _message_body_bytes(message: Any) -> int:
    """Return the payload size of an ASGI message (0 when unknown)."""
    if not isinstance(message, dict):
        return 0
    body = message.get("body")
    if isinstance(body, (bytes, bytearray, memoryview)):
        return len(body)
    return 0


class ClientStallGuardMiddleware:
    """Abandon a response whose ASGI ``send`` cannot complete in time.

    Pure ASGI (not ``BaseHTTPMiddleware``) because the whole point is to wrap
    the ``send`` callable that the response actually writes through.
    """

    def __init__(
        self,
        app: Any,
        stall_timeout: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        record: Callable[[ClientStallState, dict[str, Any]], None] | None = None,
    ) -> None:
        self.app = app
        # ``None`` defers resolution to the first request (the server config is
        # not necessarily loaded when the middleware is constructed).
        self._stall_timeout = stall_timeout
        self._clock = clock
        self._record = record or _record_stall

    def _timeout(self) -> float:
        if self._stall_timeout is None:
            self._stall_timeout = _resolve_from_server()
        return self._stall_timeout

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        timeout = self._timeout()
        if timeout <= 0:
            await self.app(scope, receive, send)
            return

        state = ClientStallState(stall_timeout=timeout)
        scope[STALL_SCOPE_KEY] = state

        async def guarded_send(message: dict[str, Any]) -> None:
            if state.aborted:
                # The reader is not accepting bytes: dropping is the only
                # non-blocking option, and the response is already forfeit.
                state.dropped_writes += 1
                return
            started = self._clock()
            try:
                await asyncio.wait_for(send(message), timeout)
            except TimeoutError:
                already_aborted = state.aborted
                state.mark_aborted(
                    stall_seconds=max(0.0, self._clock() - started),
                    blocked_body_bytes=_message_body_bytes(message),
                )
                if not already_aborted:
                    # Report at detection time, not on unwind: an application
                    # that swallows the abort (and keeps "sending") must still
                    # produce exactly one stall record.
                    self._record(state, scope)
                raise ClientStallError() from None

        try:
            await self.app(scope, receive, guarded_send)
        except ClientStallError:
            # The response is abandoned: returning releases the request and the
            # server closes the connection (uvicorn logs its own "returned
            # without completing response" for the forfeited body).
            pass
