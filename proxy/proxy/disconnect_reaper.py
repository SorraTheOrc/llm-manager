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
"""

import asyncio
import logging
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("llama-proxy.disconnect_reaper")

# Registered in-flight request tasks → their request objects. Guarded by
# ``_REGISTRY_LOCK``; entries are removed on completion or cancellation.
reaper_registry: dict[asyncio.Task, Any] = {}
_REGISTRY_LOCK = asyncio.Lock()

# How often the background reaper scans the registry (seconds).
REAP_INTERVAL_SECONDS = 5.0


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

        async with _REGISTRY_LOCK:
            for task in done_tasks:
                reaper_registry.pop(task, None)
            for task in cancel_tasks:
                reaper_registry.pop(task, None)

        for task in cancel_tasks:
            task.cancel()
            logger.info(
                "client_disconnect reaped in-flight request task=%s",
                task.get_name(),
            )
