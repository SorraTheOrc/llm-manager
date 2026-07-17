"""
Request Coalescing Module

Provides in-flight request deduplication to coalesce duplicate HTTP requests.
When multiple identical requests arrive concurrently, only the first is
processed; the rest await the result of the first.

This prevents retry cascades caused by client timeouts creating multiple
concurrent processing chains for the same logical request.
"""

import asyncio
import hashlib
import json
from typing import Any, Callable, Coroutine, Dict, Optional, Tuple

from fastapi import Response


class RequestCoalescer:
    """Coalesce duplicate in-flight requests.

    Maps ``(path, body_hash)`` to an ``asyncio.Future``. When a request
    arrives with the same ``(path, body_hash)`` as an already-in-flight
    request, the caller awaits the stored future instead of initiating a
    new processing chain.

    Responses are buffered so that multiple waiters can receive a copy
    without consuming the original stream.  Only non-streaming responses
    (where ``body`` is available as plain bytes) are coalesced; streaming
    responses are forwarded to the first caller and subsequent duplicates
    are allowed to proceed independently.
    """

    def __init__(self, max_retained: int = 256) -> None:
        self._in_flight: Dict[Tuple[str, str], asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._max_retained = max_retained

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def coalesce_or_execute(
        self,
        path: str,
        body: Optional[bytes],
        coro_factory: Callable[[], Coroutine[Any, Any, Response]],
    ) -> Response:
        """Execute ``coro_factory()``, coalescing in-flight duplicates.

        If a request with the same ``(path, body_hash)`` is already in
        flight, the result of that in-flight request is returned instead
        of invoking ``coro_factory`` again.

        Args:
            path: The API path (e.g., ``v1/chat/completions``).
            body: The raw request body (bytes or ``None``).
            coro_factory: A zero-argument callable that returns an
                awaitable ``Response``.

        Returns:
            A ``Response`` — either from the in-flight request or from
            a new processing chain.
        """
        body_hash = self._hash_body(body)
        key = (path, body_hash)

        # Phase 1: Check for existing in-flight request (brief lock)
        async with self._lock:
            existing = self._in_flight.get(key)
            if existing is not None:
                # Duplicate found — release lock BEFORE awaiting.
                # We hold a reference to the future; the lock is only
                # needed for map access, not for the wait.
                pass

            if existing is None:
                # Become the leader — create a future and register it.
                future = asyncio.get_running_loop().create_future()
                self._in_flight[key] = future
                # Evict oldest entries if over limit
                if len(self._in_flight) > self._max_retained:
                    self._trim()

        # Phase 2: Either await the existing future or execute the factory.
        if existing is not None:
            try:
                return await self._await_duplicate(existing)
            except _NonCoalesceableError:
                # The leader produced a streaming (non-coalesceable) response.
                # Remove the leader's stale entry and proceed independently.
                async with self._lock:
                    self._in_flight.pop(key, None)

        return await self._execute_leader(key, coro_factory)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _await_duplicate(
        self,
        future: asyncio.Future,
    ) -> Response:
        """Await the leader's future and reconstruct a response.

        The lock is NOT held during this wait, avoiding the deadlock
        where the leader needs the lock to store its result.
        """
        try:
            status_code, headers_list, body_bytes = await future
        except Exception as exc:
            # Re-raise the original exception so the caller sees the
            # same error the first request encountered.
            raise exc from None

        if body_bytes is None:
            # Streaming response (or empty) — we cannot safely re-serve
            # it.  Signal the caller to proceed independently by raising
            # a dedicated sentinel.
            raise _NonCoalesceableError()

        return self._reconstruct_response(status_code, headers_list, body_bytes)

    async def _execute_leader(
        self,
        key: Tuple[str, str],
        coro_factory: Callable[[], Coroutine[Any, Any, Response]],
    ) -> Response:
        """Execute the factory, store the result, and clean up."""
        try:
            response = await coro_factory()

            # Determine whether the response is coalesceable (non-streaming).
            try:
                body_bytes: bytes = response.body
            except Exception:
                body_bytes = b""

            async with self._lock:
                future = self._in_flight.get(key)
                if future is not None and not future.done():
                    if body_bytes:
                        # Non-streaming — store for duplicate waiters.
                        status_code = response.status_code
                        headers_list = list(response.headers.items())
                        future.set_result((status_code, headers_list, body_bytes))
                    else:
                        # Streaming (or empty body) — mark as
                        # non-coalesceable so twins proceed independently.
                        future.set_result((response.status_code, [], None))

            return response

        except Exception as exc:
            async with self._lock:
                future = self._in_flight.get(key)
                if future is not None and not future.done():
                    future.set_exception(exc)
            raise

        finally:
            async with self._lock:
                self._in_flight.pop(key, None)

    @staticmethod
    def _hash_body(body: Optional[bytes]) -> str:
        """Deterministic hash of request body.

        For JSON bodies, we parse and re-serialize with sorted keys so that
        semantically identical requests with different field ordering produce
        the same hash.  For non-JSON or empty bodies, we hash the raw bytes.
        """
        if not body:
            return hashlib.sha256(b"").hexdigest()
        try:
            obj = json.loads(body)
            normalized = json.dumps(obj, sort_keys=True, ensure_ascii=False)
            return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        except (json.JSONDecodeError, ValueError):
            return hashlib.sha256(body).hexdigest()

    @staticmethod
    def _reconstruct_response(
        status_code: int,
        headers_list: list,
        body_bytes: bytes,
    ) -> Response:
        """Reconstruct a ``Response`` from stored components."""
        resp = Response(
            content=body_bytes,
            status_code=status_code,
        )
        for k, v in headers_list:
            resp.headers[k] = v
        return resp

    def _trim(self) -> None:
        """Evict oldest entries when the map exceeds ``_max_retained``.

        Iterates in insertion order (Python 3.7+) and removes the first
        entries until the map is back within the limit.  Only removes
        **done** futures — in-flight futures are never removed here.
        """
        keys_to_remove: list = []
        for key, fut in list(self._in_flight.items()):
            if len(self._in_flight) - len(keys_to_remove) <= self._max_retained:
                break
            if fut.done():
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self._in_flight.pop(key, None)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_coalescer: Optional[RequestCoalescer] = None


def get_coalescer() -> RequestCoalescer:
    """Return the module-level ``RequestCoalescer`` singleton."""
    global _coalescer
    if _coalescer is None:
        _coalescer = RequestCoalescer()
    return _coalescer


# ---------------------------------------------------------------------------
# Sentinel exception
# ---------------------------------------------------------------------------

class _NonCoalesceableError(Exception):
    """Raised when a streaming response cannot be coalesced.

    The duplicate caller should fall through to independent execution.
    """
    pass
