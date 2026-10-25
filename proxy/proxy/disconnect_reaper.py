"""CLOSE-WAIT mitigation for abandoned in-flight requests.

When a client abandons a connection mid-request (disconnects without
finishing), the server-side socket can linger in CLOSE-WAIT until the
handler finishes. Long-running requests — e.g. the historical full-tree
``/admin/sessions`` scan — left dozens of sockets stuck in CLOSE-WAIT
(LP-0MSNKMZCP003T8OG / LP-0MSNM9UCC002CHYU).

This module provides:

- ``DisconnectReaperMiddleware`` — a Starlette ``BaseHTTPMiddleware`` that
  tracks in-flight request tasks without touching the request body. It does
  NOT probe ``request.is_disconnected()`` pre-dispatch (that can deadlock on
  buffered single-message bodies); instead the background ``DisconnectReaper``
  polls disconnect state and cancels abandoned tasks.
- ``DisconnectReaper`` — a background reaper that periodically cancels
  registered tasks whose client disconnected while running.

The streaming chat path already polls ``request.is_disconnected()`` between
chunks; this module covers the non-streaming / long-running cases where the
handler does not self-check.

It also carries the *verdict*: before cancelling a task the reaper records
``client_disconnect_observed`` on the request's ASGI scope, because a
cancelled task cannot reliably re-probe the connection. The local dispatch
release path reads that flag synchronously to free a dispatch whose client
died during prefill (LP-0MVLFVY6K003VYEL).
"""

import asyncio
import logging
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger("llama-proxy.disconnect_reaper")

# Registered in-flight request tasks → their request objects. Guarded by
# ``_REGISTRY_LOCK``; entries are removed on completion or cancellation.
reaper_registry: dict[asyncio.Task, Any] = {}
_REGISTRY_LOCK = asyncio.Lock()

# How often the background reaper scans the registry (seconds).
REAP_INTERVAL_SECONDS = 5.0

CLIENT_DISCONNECTED_SCOPE_KEY = "llama_proxy.client_disconnected"
"""ASGI scope key the reaper sets when it confirms a client disconnect.

A cancelled request task cannot reliably re-probe ``request.is_disconnected()``
(awaiting inside a cancellation handler can be cancelled again), so the
reaper — which *did* observe the disconnect before cancelling — records the
verdict on the request's ASGI ``scope``. The scope dict is shared by the
middleware request and the handler request, so the local dispatch release
path reads the same flag synchronously
(LP-0MVLFVY6K003VYEL AC3).
"""


def mark_client_disconnected(request: Any) -> None:
    """Record a confirmed client disconnect on the request's ASGI scope."""
    try:
        request.scope[CLIENT_DISCONNECTED_SCOPE_KEY] = True
    except Exception:
        pass


def client_disconnect_observed(request: Any) -> bool:
    """True only when the reaper confirmed this request's client is gone.

    Fail-closed: anything other than an explicit ``True`` (missing scope,
    scope without the key, a mock) reads as "not confirmed", so a request is
    never released as client-dead on an inference.
    """
    if request is None:
        return False
    try:
        return request.scope.get(CLIENT_DISCONNECTED_SCOPE_KEY) is True
    except Exception:
        return False


class DisconnectReaperMiddleware(BaseHTTPMiddleware):
    """Tracks in-flight request tasks for the CLOSE-WAIT reaper.

    Registers the current task + request in ``reaper_registry`` while the
    request is being processed, and unregisters in a ``finally``. The
    background ``DisconnectReaper`` cancels tasks whose client has
    disconnected. Does not read the request body, so streaming and
    single-message bodies are unaffected.
    """

    async def dispatch(self, request, call_next):
        current_task = asyncio.current_task()
        if current_task is not None:
            async with _REGISTRY_LOCK:
                reaper_registry[current_task] = request
        try:
            return await call_next(request)
        except asyncio.CancelledError:
            # The DisconnectReaper background loop may have cancelled this
            # task because the client disconnected. This is expected and
            # benign — it happened for one of two reasons:
            #
            #   1. Response already sent: the handler finished and Starlette
            #      rendered the response, but the reaper won a race and
            #      cancelled us while `call_next` was still unwinding. The
            #      ASGI `send()` became a no-op (uvicorn swallows sends to
            #      a disconnected client), so accessing uvicorn's "error"
            #      logging is fine but the explicit ERROR traceback is noise.
            #
            #   2. Abandoned request cancelled mid-processing: there is no
            #      client left to read a response, so sending a new one is
            #      pointless — return a body that will be discarded as a
            #      no-op send.
            #
            # Either way, swallow the CancelledError so it does not surface
            # as an "ERROR: Exception in ASGI application" in the logs.
            # A body-less 499 (client closed request) is never shown to a
            # client; the send is a no-op for a genuinely gone client.
            logger.debug(
                "task cancelled (likely by DisconnectReaper); request=%s",
                request.url.path,
            )
            return Response(status_code=499, content=b"")
        finally:
            if current_task is not None:
                async with _REGISTRY_LOCK:
                    reaper_registry.pop(current_task, None)


class DisconnectReaper:
    """Cancels in-flight request tasks whose client has disconnected."""

    def __init__(self, interval: float = REAP_INTERVAL_SECONDS):
        self._interval = interval
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the background reaping loop (idempotent)."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the background loop and clear the registry."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        async with _REGISTRY_LOCK:
            reaper_registry.clear()

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self.reap_once()
            except Exception:
                logger.exception("disconnect reaper iteration failed")

    async def reap_once(self) -> None:
        """Cancel registered tasks whose client has disconnected."""
        async with _REGISTRY_LOCK:
            snapshot = list(reaper_registry.items())

        done_tasks: list[asyncio.Task] = []
        cancel_tasks: list[asyncio.Task] = []
        cancel_requests: dict[asyncio.Task, Any] = {}
        for task, request in snapshot:
            if task.done():
                done_tasks.append(task)
                continue
            try:
                disconnected = await request.is_disconnected()
            except Exception:
                disconnected = False
            if disconnected:
                cancel_tasks.append(task)
                cancel_requests[task] = request

        async with _REGISTRY_LOCK:
            for task in done_tasks:
                reaper_registry.pop(task, None)
            for task in cancel_tasks:
                reaper_registry.pop(task, None)

        for task in cancel_tasks:
            # Record the verdict on the request scope *before* cancelling: the
            # cancelled task cannot re-probe the connection, and the local
            # dispatch release path reads this flag synchronously
            # (LP-0MVLFVY6K003VYEL).
            mark_client_disconnected(cancel_requests.get(task))
            task.cancel()
            logger.info(
                "client_disconnect reaped in-flight request task=%s",
                task.get_name(),
            )
