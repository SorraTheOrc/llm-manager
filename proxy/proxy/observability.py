"""
Observability Module

Centralizes backend signal tracking (connect/read/timeout/other failures),
SSE client management for real-time status and log tail broadcasts, and
exception classification for observability.

Uses a lazy server import (_srv()) to access module-level state without
circular import issues.

Shared HTTP helpers for llama-server queries
--------------------------------------------

The module exposes several helpers that were extracted from the duplicated
HTTP JSON parsing and URL building patterns found in both
``query_llama_status()`` (here) and ``get_llama_local_status()``
(``handlers.py``):

- ``_safe_parse_json_response(response)`` — Defensive JSON parsing with
  async/sync ``.json()`` → ``.text`` → ``json.loads()`` fallback.
- ``_build_llama_url(llama_port, endpoint)`` — URL builder for llama-server
  endpoints.
- ``_query_slots(client, llama_port, timeout)`` — Query the ``/slots``
  endpoint, returning ``(available_slots, total_slots)``. Records the last
  successful counts (``_last_slot_counts_cache``) and increments
  ``llama_slots_query_failures_total`` on failure (graceful degradation,
  LP-0MSVP7XJ6008QPKX).
- ``last_known_slot_counts()`` — Last successful ``(available, total)`` when
  fresh (bounded by ``SLOT_COUNTS_STALE_AFTER_SECONDS``), served by the
  status endpoint when a fresh ``/slots`` query fails so orchestrators see
  known capacity instead of ``total_slots=0``.
- ``_query_slots_detail(llama_port, timeout, model)`` — Query the ``/slots``
  endpoint with an optional ``model`` query parameter, returning a list of
  per-slot dicts with keys ``slot_id``, ``is_processing``, and ``n_decoded``.
- ``_query_slots_progress(llama_port, timeout, model)`` — Query the ``/slots``
  endpoint and return a dict mapping slot id -> prefill state
  ``{"progress": int|None, "processing": bool}`` where *progress* is the max
  of ``n_past`` / ``n_prompt_tokens_processed`` (older llama.cpp builds) and
  *processing* is the slot's ``is_processing`` flag (the only per-slot
  liveness signal llama.cpp b8782 still exposes), used by the
  prefill-progress dispatch-lease extension (LP-0MSE05J53004C6EL,
  LP-0MSUO5Z0K007HBSS).

See LP-0MR6Y11OP005UHIH for the consolidation rationale.
"""

import asyncio
import contextlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Lazy server import — avoids circular imports when server.py imports us
# ---------------------------------------------------------------------------
def _srv():
    import proxy.server as _m
    return _m


# ===================================================================
# Backend signal counters (connect/read/timeout/other/concurrency)
# ===================================================================

backend_signal_counts: dict = {
    "connect_failures": 0,
    "read_failures": 0,
    "timeout_failures": 0,
    "other_failures": 0,
    "concurrency_rejects": 0,
    "gpu_wedge": 0,
    "grandfathered": 0,
    "context_too_large": 0,  # LP-0MTBOX45O005LD1S fast-mode cap skip
    # Per-entry ``max_context_tokens`` ceiling skip (LP-0MVF9N3BV00155RI):
    # mirrors ``context_too_large``; a size mismatch, never a failure signal.
    "provider_context_exceeded": 0,
}


def _record_backend_signal(signal_name: str) -> None:
    """Increment a backend signal counter for observability."""
    srv = _srv()
    try:
        if signal_name in srv.backend_signal_counts:
            srv.backend_signal_counts[signal_name] = (
                int(srv.backend_signal_counts.get(signal_name, 0)) + 1
            )
    except Exception:
        pass


def _classify_backend_exception(exc: Exception) -> str:
    """Map backend transport exceptions to signal buckets."""
    import httpx

    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return "connect_failures"
    if isinstance(exc, (httpx.ReadError,)):
        return "read_failures"
    if isinstance(exc, (httpx.ReadTimeout, httpx.TimeoutException)):
        return "timeout_failures"
    return "other_failures"


# ===================================================================
# Shared HTTP helpers for llama-server queries
# ===================================================================

async def _safe_parse_json_response(response) -> Any:
    """Parse JSON from an HTTP response, handling async/sync variants and text fallback.

    Uses the pattern from ``query_llama_status()``:
    ``.json()`` (async/sync) → ``.text`` (async/sync) → ``json.loads()``
    with full exception safety.

    Returns the parsed data (dict, list, etc.) or ``None`` if all parse paths
    fail.
    """
    data = None
    if hasattr(response, "json"):
        try:
            maybe = response.json()
            data = await maybe if asyncio.iscoroutine(maybe) else maybe
        except Exception:
            data = None
    if data is None and hasattr(response, "text"):
        try:
            txt = response.text if not asyncio.iscoroutine(response.text) else await response.text
            data = json.loads(txt)
        except Exception:
            data = None
    return data


def _build_llama_url(llama_port: int, endpoint: str, base_url: str | None = None) -> str:
    """Build a URL for a llama-server endpoint.

    Args:
        llama_port: The port llama-server is listening on. Ignored when
            *base_url* is provided (LP-0MRPILSMW004T4H8).
        endpoint: The API path (e.g. ``"/slots"``). A leading slash is
            added if missing.
        base_url: Optional full server base URL (e.g.
            ``http://192.168.0.199:8080``) for multi-backend deployments.
            When provided, the URL is built from it instead of
            ``http://localhost:{llama_port}``.
    """
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    if base_url:
        return f"{base_url.rstrip('/')}{endpoint}"
    return f"http://localhost:{llama_port}{endpoint}"


# ===================================================================
# Last-known slot counts cache (graceful degradation, LP-0MSVP7XJ6008QPKX)
#
# After a cheap-mode restart, llama-server's /slots endpoint returns HTTP
# 500 while the model reloads. Reporting total_slots=0 during that window
# fail-closed the herdr downtime worker (isIdleStatus false → zero
# dispatches overnight). We keep the last successful (available, total)
# counts and serve them on transient /slots failure, bounded by
# SLOT_COUNTS_STALE_AFTER_SECONDS so a genuinely-unavailable llama-server
# eventually fail-closes instead of serving stale capacity forever.
# ===================================================================

_last_slot_counts_cache: tuple[int, int, float] | None = None
"""Last successful (available, total, monotonic_ts) from ``_query_slots()``.

Served by ``last_known_slot_counts()`` when a fresh /slots query fails so
orchestrators (herdr downtime worker) still see known capacity instead of
total_slots=0 (LP-0MSVP7XJ6008QPKX).
"""


def _slots_counts_stale_after_seconds() -> float:
    """Staleness bound (seconds) for the last-known slot counts cache.

    Configurable via ``SLOT_COUNTS_STALE_AFTER_SECONDS`` (default 3600s = 1h).
    The bound exists so a llama-server that is genuinely down for a long
    period eventually fail-closes (returns 0/0) instead of serving stale
    capacity; the /slots failure-rate alert (monitoring/slots_query_alerts.yaml)
    surfaces sustained failures.
    """
    import os

    try:
        return float(os.environ.get("SLOT_COUNTS_STALE_AFTER_SECONDS", "3600"))
    except (TypeError, ValueError):
        return 3600.0


def last_known_slot_counts() -> tuple[int, int] | None:
    """Return the last known (available, total) slot counts when fresh.

    Returns ``None`` when no successful query has been recorded or the
    cached counts are older than ``SLOT_COUNTS_STALE_AFTER_SECONDS``.
    """
    cache = _last_slot_counts_cache
    if cache is None:
        return None
    available, total, ts = cache
    if time.monotonic() - ts > _slots_counts_stale_after_seconds():
        return None
    return (available, total)


# ===================================================================
# /slots poll coalescing (LP-0MUCEFD5B003TURT)
#
# Every herdr client polls /llama/local/status, and each poll queries
# llama-server /slots. Under N concurrent pollers that is N upstream
# requests per poll interval, which amplifies load on an already-saturated
# backend (4554 cancel-task warnings in the wedge window). A short-TTL
# cache plus single-flight coalescing bounds upstream /slots volume by the
# TTL instead of by N. A failed fetch is cached for a longer backoff window
# so an unavailable backend is not hammered, and callers fall back to the
# last-known counts (with the stale flag) via ``last_known_slot_counts``.
#
# Invariant (LP-0MVJML600003CQ6Y, AC1-AC3, AC5):
#
#   1. The shared single-flight future is ALWAYS settled before its owner
#      stops owning it: ``set_result``/``set_exception`` on a completed fetch,
#      or ``cancel()`` when the leader itself is cancelled. A failure nobody is
#      following is marked as retrieved by its owner (``_consume_slots_inflight_outcome``)
#      so asyncio logs no "Future exception was never retrieved" at arbitrary
#      GC times for the usual lone-failure case (LP-0MVK6QUAQ002KFHU). The second case is
#      the 2026-10-22 Web UI freeze: ``asyncio.CancelledError`` derives from
#      ``BaseException``, so a browser disconnect cancelling the SSE handler
#      bypassed the old ``except Exception`` and left the future pending with
#      no owner while ``finally`` removed the registry entry.
#   2. A follower NEVER waits on that future unboundedly. It waits at most its
#      own caller timeout using ``asyncio.wait`` — which, unlike
#      ``asyncio.wait_for``/``asyncio.timeout``, does not cancel the awaited
#      future, so one impatient follower cannot break the wait of the others.
#      Only once that bound elapses does it promote itself: the abandoned
#      future is cancelled (releasing any remaining follower of it) and a fresh
#      one is installed under the lock, with this caller as the new leader.
#   3. A registry entry is only removed by the caller that still owns it, so a
#      promoted leader's future is never evicted by the task it replaced.
#
# Together these guarantee ``_periodic_broadcast_loop()`` — which reaches this
# helper through ``query_llama_status()`` and ``_query_slots_detail()`` — can
# never stall on an abandoned future, so one cancelled request can no longer
# freeze status updates for every connected Web UI client.
# ===================================================================

_slots_poll_cache: dict[str, tuple[float, int, object]] = {}
_slots_poll_inflight: dict[str, "asyncio.Future"] = {}
_slots_poll_lock: "asyncio.Lock | None" = None

# How many bounded waits a follower performs before it degrades instead of
# retrying: one to notice the original owner is cancelled/stuck, one to notice
# that the promotion race handed it a future that is itself stuck.
_SLOTS_POLL_FOLLOWER_ATTEMPTS = 2


def _get_slots_poll_lock() -> "asyncio.Lock":
    global _slots_poll_lock
    if _slots_poll_lock is None:
        _slots_poll_lock = asyncio.Lock()
    return _slots_poll_lock


def _slots_poll_ttl_seconds() -> float:
    """Freshness window (seconds) for the coalesced /slots snapshot."""
    import os

    try:
        return max(0.0, float(os.environ.get("SLOTS_POLL_TTL_SECONDS", "1.0")))
    except (TypeError, ValueError):
        return 1.0


def _slots_poll_failure_backoff_seconds() -> float:
    """Backoff window (seconds) cached after a failed /slots fetch."""
    import os

    try:
        return max(
            0.0,
            float(os.environ.get("SLOTS_POLL_FAILURE_BACKOFF_SECONDS", "5.0")),
        )
    except (TypeError, ValueError):
        return 5.0


def _slots_poll_follower_timeout_seconds(caller_timeout) -> float:
    """Upper bound (seconds) on a follower's wait for the shared future.

    Defaults to the caller's own *timeout* — the budget it would have spent
    fetching upstream itself — so a follower degrades no later than it would
    have failed on its own. ``SLOTS_POLL_FOLLOWER_TIMEOUT_SECONDS`` may lower
    that bound, never raise it. A follower that takes over leadership gets a
    fresh caller-sized fetch budget, so its total work is bounded by two
    caller timeouts (LP-0MVJML600003CQ6Y, AC2).
    """
    import os

    try:
        budget = max(0.0, float(caller_timeout))
    except (TypeError, ValueError):
        budget = 5.0
    try:
        configured = float(os.environ.get("SLOTS_POLL_FOLLOWER_TIMEOUT_SECONDS", ""))
    except (TypeError, ValueError):
        return budget
    if configured <= 0:
        return budget
    return min(budget, configured)


def reset_slots_poll_cache() -> None:
    """Clear the coalesced /slots cache (tests / config reload)."""
    _slots_poll_cache.clear()


def _cached_slots_payload(url: str) -> tuple[int, object] | None:
    """Return the cached (status, data) for *url* when still fresh."""
    cached = _slots_poll_cache.get(url)
    if cached is None:
        return None
    ts, status, data = cached
    ttl = _slots_poll_ttl_seconds()
    if status != 200:
        ttl = max(ttl, _slots_poll_failure_backoff_seconds())
    if time.monotonic() - ts < ttl:
        return status, data
    return None


async def _do_slots_fetch(
    url: str, timeout: float, client=None
) -> tuple[int, object]:
    """Perform one raw /slots fetch, returning ``(status_code, parsed)``."""
    if client is not None:
        resp = await asyncio.wait_for(client.get(url), timeout=timeout)
    else:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as _c:
            resp = await asyncio.wait_for(_c.get(url), timeout=timeout)
    status = int(getattr(resp, "status_code", 0) or 0)
    if status == 200:
        return status, await _safe_parse_json_response(resp)
    return status, None


def _consume_slots_inflight_outcome(fut: "asyncio.Future") -> None:
    """Retrieve a settled shared future's outcome so asyncio logs no late warning.

    The single-flight future is a broadcast channel that is usually
    *unobserved*: the common case is a lone poller whose ``/slots`` fetch fails
    (e.g. ConnectionError while llama-server restarts) and no follower ever
    calls ``fut.result()``. A settled exception nobody retrieves makes
    ``Future.__del__`` log "Future exception was never retrieved" whenever the
    cyclic GC later reclaims the traceback cycle -- production log noise whose
    timestamp is pure GC timing, and a cross-test flake source in the suite
    (LP-0MVK6QUAQ002KFHU). Calling ``exception()`` here marks the outcome as
    observed without changing semantics: a follower that does call
    ``fut.result()`` still re-raises the leader's failure.
    """
    if fut.cancelled():
        return
    with contextlib.suppress(BaseException):
        fut.exception()


def _new_slots_inflight_future() -> "asyncio.Future":
    """Create the shared single-flight future on the running loop."""
    fut = asyncio.get_running_loop().create_future()
    fut.add_done_callback(_consume_slots_inflight_outcome)
    return fut


async def _release_slots_inflight(url: str, fut: "asyncio.Future") -> None:
    """Drop *url*'s registry entry, but only when *fut* is still the entry.

    A promoted leader may already have replaced the entry for *url*; popping it
    blindly would let a third caller start a duplicate upstream fetch
    (LP-0MVJML600003CQ6Y, invariant 3).
    """
    async with _get_slots_poll_lock():
        if _slots_poll_inflight.get(url) is fut:
            _slots_poll_inflight.pop(url, None)


async def _promote_slots_follower(
    url: str, stale: "asyncio.Future"
) -> tuple["asyncio.Future", bool]:
    """Promote a follower whose leader never resolved the shared future.

    The abandoned *stale* future is cancelled (releasing any other follower of
    it) and evicted, and a fresh future is installed under the lock. Returns
    ``(future, is_leader)``; ``is_leader`` is ``False`` when another caller won
    the promotion race and already installed a newer future, in which case that
    future is returned for this caller to follow.
    """
    async with _get_slots_poll_lock():
        current = _slots_poll_inflight.get(url)
        if current is not None and current is not stale:
            return current, False
        if not stale.done():
            stale.cancel()
        fut = _new_slots_inflight_future()
        _slots_poll_inflight[url] = fut
        return fut, True


async def _await_slots_inflight(fut: "asyncio.Future", wait_seconds: float) -> bool:
    """Wait at most *wait_seconds* for *fut*; return whether it is done.

    Uses ``asyncio.wait`` rather than ``asyncio.wait_for``/``asyncio.timeout``
    because those cancel the awaited future on timeout, which would break the
    wait for every *other* follower sharing it (LP-0MVJML600003CQ6Y, AC2).
    """
    if fut.done():
        return True
    if wait_seconds <= 0:
        return fut.done()
    await asyncio.wait({fut}, timeout=wait_seconds)
    return fut.done()


async def _run_slots_fetch_as_leader(
    url: str, timeout: float, client, fut: "asyncio.Future"
) -> tuple[int, object]:
    """Fetch /slots and settle the shared future for every follower."""
    try:
        status, data = await _do_slots_fetch(url, timeout, client)
        _slots_poll_cache[url] = (time.monotonic(), status, data)
        if not fut.done():
            fut.set_result((status, data))
        return status, data
    except BaseException as exc:
        # ``CancelledError`` derives from ``BaseException``, so a leader
        # cancelled by a client disconnect used to skip this path entirely and
        # leave the future pending with no owner (LP-0MVJML600003CQ6Y, AC1).
        # Always settle the future before re-raising. Cancelling — rather than
        # setting an exception — also keeps asyncio's "Future exception was
        # never retrieved" warning out of the logs when nobody is following.
        if not fut.done():
            if isinstance(exc, Exception):
                fut.set_exception(exc)
            else:
                fut.cancel()
        raise
    finally:
        await _release_slots_inflight(url, fut)


async def _fetch_slots_payload(
    url: str, timeout: float, client=None
) -> tuple[int, object]:
    """Fetch /slots with TTL caching and single-flight coalescing.

    Concurrent callers for the same *url* share one upstream request; a
    fresh cached result (within ``SLOTS_POLL_TTL_SECONDS``) is returned
    without touching the backend. A non-200 result is cached for
    ``SLOTS_POLL_FAILURE_BACKOFF_SECONDS`` so a failing backend is not
    hammered. Returns ``(status_code, parsed_data)``.

    Cancellation safety (LP-0MVJML600003CQ6Y): the shared future is always
    settled by its owner, and a follower waits for it only for a bounded time
    before promoting itself to leader — see the invariant block above.
    """
    cached = _cached_slots_payload(url)
    if cached is not None:
        return cached

    lock = _get_slots_poll_lock()
    leader = False
    async with lock:
        cached = _cached_slots_payload(url)
        if cached is not None:
            return cached
        inflight = _slots_poll_inflight.get(url)
        if inflight is not None:
            fut = inflight
        else:
            fut = _new_slots_inflight_future()
            _slots_poll_inflight[url] = fut
            leader = True

    if leader:
        return await _run_slots_fetch_as_leader(url, timeout, client, fut)

    # Follower: bounded wait, then take over leadership (AC1/AC2).
    wait_seconds = _slots_poll_follower_timeout_seconds(timeout)
    for attempt in range(_SLOTS_POLL_FOLLOWER_ATTEMPTS):
        if await _await_slots_inflight(fut, wait_seconds) and not fut.cancelled():
            # Re-raises the leader's exception, preserving the old shared-await
            # semantics for genuine upstream failures.
            return fut.result()
        # The leader was cancelled or never resolved: this caller makes its own
        # progress rather than waiting forever.
        fut, leader = await _promote_slots_follower(url, fut)
        if leader:
            return await _run_slots_fetch_as_leader(url, timeout, client, fut)
        if attempt + 1 >= _SLOTS_POLL_FOLLOWER_ATTEMPTS:
            break

    # Every shared future this caller followed was abandoned: degrade to a
    # still-fresh cached payload, else report the timeout to the caller (which
    # records a /slots failure and falls back to last-known counts).
    cached = _cached_slots_payload(url)
    if cached is not None:
        return cached
    raise TimeoutError(
        f"/slots single-flight follower waited {_SLOTS_POLL_FOLLOWER_ATTEMPTS} "
        f"bounded attempts for {url} without a payload"
    )


async def _query_slots(
    client, llama_port: int, timeout: float = 2.0, model: str | None = None,
    base_url: str | None = None,

) -> tuple:
    """Query the llama-server ``/slots`` endpoint.

    Returns a ``(available_slots, total_slots)`` tuple.  Both default to
    ``0`` on any failure (HTTP error, connection error, timeout, or
    unexpected response shape).

    Many llama-server instances require ``?model=...`` on the ``/slots``
    endpoint and return HTTP 400 without it (LP-0MSHFGO0M003Q5BL), so
    pass *model* when the current model is known.

    When *base_url* is provided (a full ``http://host:port`` URL), the
    query targets that specific llama-server instance (LP-0MRPILSMW004T4H8).
    Otherwise ``http://localhost:{llama_port}`` is used.


    The 2.0-second default timeout matches the original inline query in
    ``get_llama_local_status()``.

    On a successful query with total > 0 the last-known counts cache is
    updated (graceful degradation, LP-0MSVP7XJ6008QPKX); on failure the
    ``llama_slots_query_failures_total`` metric is incremented so a
    sustained /slots outage surfaces as an alert.
    """
    try:
        url = _build_llama_url(llama_port, "/slots", base_url=base_url)

        if model:
            url = f"{url}?model={model}"
        # LP-0MUCEFD5B003TURT: coalesce concurrent /slots polls behind a
        # short TTL so N status pollers produce a bounded number of upstream
        # requests (not N per interval).
        status, slots_data = await _fetch_slots_payload(url, timeout, client)
        if status == 200:
            if isinstance(slots_data, list):
                total_slots = len(slots_data)
                available_slots = sum(
                    1 for s in slots_data if not s.get("is_processing", True)
                )
                if total_slots > 0:
                    _record_last_slot_counts(available_slots, total_slots)
                return available_slots, total_slots
            _record_slots_query_failure("invalid_payload")
            return 0, 0
        _record_slots_query_failure(f"http_{status}")
        return 0, 0
    except TimeoutError:
        _record_slots_query_failure("timeout")
    except Exception as exc:
        if isinstance(exc, (ConnectionError, OSError)):
            _record_slots_query_failure("connection_error")
        else:
            _record_slots_query_failure("unknown")
    return 0, 0


def _record_last_slot_counts(available_slots: int, total_slots: int) -> None:
    """Store the last successful slot counts with a monotonic timestamp."""
    global _last_slot_counts_cache
    _last_slot_counts_cache = (available_slots, total_slots, time.monotonic())


def _record_slots_query_failure(reason: str) -> None:
    """Increment the /slots query failure metric (best-effort)."""
    try:
        from proxy.metrics import record_slots_query_failure

        record_slots_query_failure(reason)
    except Exception:
        pass


def _slot_context_tokens(slot: dict) -> int | None:
    """Return the slot's current context-token usage (prompt progress).

    The maximum of the slot's ``n_past`` and ``n_prompt_tokens_processed``
    fields when present (older llama.cpp builds that still report per-slot
    prefill progress; LP-0MSE05J53004C6EL), else ``None``. llama.cpp b8782
    removed both fields from the ``/slots`` response
    (LP-0MSUO5Z0K007HBSS), so ``None`` is the *honest* value on modern
    builds — the caller must not fabricate a context size (LP-0MVGDUVGE0024YAI
    AC3/AC7: null ETA rather than a misleading estimate).
    """
    candidates = [
        c
        for c in (slot.get("n_past"), slot.get("n_prompt_tokens_processed"))
        if isinstance(c, (int, float)) and c >= 0
    ]
    return int(max(candidates)) if candidates else None


async def _query_slots_detail(
    llama_port: int,
    timeout: float = 2.0,
    model: str | None = None,
    endpoint: str | None = None,
    _client: httpx.AsyncClient | None = None,
) -> list[dict]:
    """Query the llama-server ``/slots`` endpoint and return per-slot details.

    By default creates its own ``httpx.AsyncClient`` for each call,
    avoiding shared-client state issues.  Uses httpx's built-in timeout
    (not ``asyncio.wait_for``) to avoid cancellation-related bugs with
    httpx's internal connection management (Python 3.12+).

    Args:
        llama_port: Port llama-server is listening on.
        timeout: Request timeout in seconds.
        model: Optional model name to filter slots.  Many llama-server
            instances require ``?model=...`` on the ``/slots`` endpoint
            and return HTTP 400 without it.
        _client: Internal — for testing only.  Pass a mock client to
            avoid real HTTP calls in unit tests.

    Returns a list of dicts, one per slot, with keys:

    - ``slot_id`` (int) — slot identifier from the response.
    - ``is_processing`` (bool) — whether the slot is actively processing.
    - ``n_decoded`` (int or ``None``) — the number of decoded tokens so far.
    - ``context_tokens`` (int or ``None``) — current prompt progress
      (``max(n_past, n_prompt_tokens_processed)``) when the build reports it,
      else ``None``; feeds per-slot ``phase``/``prefill_eta_seconds``
      (LP-0MVGDUVGE0024YAI AC5/AC3).

    Returns an empty list on any failure (HTTP error, connection error,
    timeout, or unexpected response shape).
    """
    try:
        url = _build_llama_url(llama_port, "/slots", base_url=endpoint)
        if model:
            url = f"{url}?model={model}"
        if _client is not None:
            slots_resp = await _client.get(url)
            if slots_resp.status_code == 200:
                slots_data = await _safe_parse_json_response(slots_resp)
            else:
                slots_data = None
        else:
            # LP-0MUCEFD5B003TURT: coalesce concurrent /slots polls.
            status, slots_data = await _fetch_slots_payload(url, timeout, None)
            if status != 200:
                slots_data = None
        if isinstance(slots_data, list):
            result = []
            for i, slot in enumerate(slots_data):
                is_processing = bool(slot.get("is_processing", False))
                next_token = slot.get("next_token")
                n_decoded = None
                if isinstance(next_token, dict):
                    n_decoded = next_token.get("n_decoded")
                elif isinstance(next_token, list) and len(next_token) > 0:
                    first = next_token[0]
                    if isinstance(first, dict):
                        n_decoded = first.get("n_decoded")
                result.append({
                    "slot_id": slot.get("id", i),
                    "is_processing": is_processing,
                    "n_decoded": n_decoded,
                    # Per-slot prompt progress (LP-0MVGDUVGE0024YAI AC3/AC5):
                    # the single source for context_tokens, sourced from the
                    # same raw fields ``_query_slots_progress`` reads.
                    "context_tokens": _slot_context_tokens(slot),
                })
            return result
    except Exception as exc:
        _srv().logger.debug(
            "Slot detail query failed [%s] for %s?model=%s: %s",
            type(exc).__name__, _build_llama_url(llama_port, "/slots"),
            model or "(none)", exc,
        )
    return []


async def _query_slots_progress(
    llama_port: int,
    timeout: float = 2.0,
    model: str | None = None,
    endpoint: str | None = None,
    _client: httpx.AsyncClient | None = None,
) -> dict[int, dict]:
    """Query the llama-server ``/slots`` endpoint for per-slot prefill state.

    Returns a dict mapping slot id -> ``{"progress": int|None,
    "processing": bool}``:

    - *progress* — the maximum of the slot's ``n_past`` and
      ``n_prompt_tokens_processed`` fields when present (older llama.cpp
      builds that still report per-slot prefill progress;
      LP-0MSE05J53004C6EL), else ``None``. llama.cpp b8782 removed both
      fields from the ``/slots`` response (LP-0MSUO5Z0K007HBSS).
    - *processing* — the slot's ``is_processing`` flag, the only per-slot
      liveness signal b8782 still exposes. While True the slot is actively
      working a request (prefill or generation), so the dispatch lease can
      be kept alive on liveness even when no numeric progress is reported.

    When *endpoint* is provided (a full ``http://host:port`` URL), the
    query targets that specific llama-server instance
    (LP-0MRPILSMW004T4H8). Otherwise ``http://localhost:{llama_port}``
    is used.

    Returns an empty dict on any failure (HTTP error, connection error,
    timeout, or unexpected response shape), mirroring ``_query_slots_detail``.

    Pass ``_client`` only in tests to avoid real HTTP.
    """
    try:
        url = _build_llama_url(llama_port, "/slots", base_url=endpoint)
        if model:
            url = f"{url}?model={model}"
        if _client is not None:
            slots_resp = await _client.get(url)
            if slots_resp.status_code == 200:
                slots_data = await _safe_parse_json_response(slots_resp)
            else:
                slots_data = None
        else:
            # LP-0MUCEFD5B003TURT: coalesce concurrent /slots polls.
            status, slots_data = await _fetch_slots_payload(url, timeout, None)
            if status != 200:
                slots_data = None
        if isinstance(slots_data, list):
            result: dict[int, dict] = {}
            for i, slot in enumerate(slots_data):
                if not isinstance(slot, dict):
                    continue
                sid = slot.get("id", i)
                progress = _slot_context_tokens(slot)
                result[sid] = {
                    "progress": progress,
                    "processing": bool(slot.get("is_processing", False)),
                }
            return result
    except Exception as exc:
        _srv().logger.debug(
            "Slot progress query failed [%s] for %s: %s",
            type(exc).__name__, _build_llama_url(llama_port, "/slots", base_url=endpoint), exc,
        )
    return {}


# ===================================================================
# SSE clients for real-time broadcasts
# ===================================================================

sse_clients: set[asyncio.Queue] = set()
log_tail_clients: set[asyncio.Queue] = set()


# ===================================================================
# Last-known slot details cache
#
# Shared module-level cache so that both ``_periodic_broadcast_loop()`` and
# per-connection SSE handlers (``ui.status_events()``) can survive transient
# timeouts when llama-server is busy generating tokens.
# ===================================================================

_last_slot_details_cache: list[dict] = []
"""Last successful result from ``_query_slots_detail()``.

Updated by ``_periodic_broadcast_loop()`` on each successful query.
Read by ``status_events()`` in ``proxy.ui`` and by the
``/llama/local/status`` handler (``proxy.handlers``) as a fallback when
the initial /slots query fails or times out (LP-0MUFSVXID0039ZAQ).
"""


_slot_progress_cache: dict[int, dict] = {}
"""Per-slot progress data parsed from llama-server log lines.

Maps ``slot_id`` to ``{"n_tokens": int, "progress": float,"timestamp": float}``.
Used as a supplemental source of token counts when the ``/slots`` HTTP
endpoint is unresponsive during busy token generation.

``timestamp`` is the *source* time of the data — the log file's mtime at the
read that produced the entry — not the read time.  A frozen log therefore
ages out of the cache instead of being re-stamped fresh on every cycle
(LP-0MVECWYWK000U3E1).

Populated by ``_update_slot_progress_from_log()`` which is called by
``_periodic_broadcast_loop()`` on each iteration.
"""


SLOT_PROGRESS_MAX_AGE_SECONDS: float = 60.0
"""Maximum age (seconds) for log-derived progress data to be trusted.

Entries older than this are pruned from ``_slot_progress_cache`` and
rejected by ``_enrich_slot_details_with_progress()`` / ``_slot_progress_is_fresh()``.
The window is measured against the log's real mtime, so a frozen
``llama-server.log`` self-expires (LP-0MVECWYWK000U3E1).
"""


_slot_stable_tracker: dict[int, dict] = {}
"""Tracks n_tokens stability to detect when generation is complete.

Maps ``slot_id`` to ``{"n_tokens": int, "stable_since": float}``.
When ``n_tokens`` stops increasing for > 3 seconds, generation is
considered complete (streaming response to client).  Used to set
``generation_done`` in enriched slot details.
"""


GENERATION_COMPLETE_TOKEN_MARGIN: int = 20
"""Tokens within ``total_tokens`` that count as prefill-complete.

A stable ``n_tokens`` value is not sufficient evidence that generation
has finished: during prompt prefill a batch can pause for more than the
stability window, which previously flipped the Web UI to its 'Working'
badge and back to 'Processed x of y...' on the next increment
(WL-0MSZVB902004AZZN).  Only treat stability as completion when prefill
is also essentially done, i.e. ``n_tokens`` is within this margin of
``total_tokens``.
"""


_processing_slot_assignments: dict[int, str] = {}
"""Mapping from llama-server processing slot_id to session_id.

Populated by ``_update_processing_slot_assignments()`` which matches
progress data (llama-server slot_id) against active dispatch leases
(session_id).  This is a separate namespace from ``_slot_owners`` in
``proxy.session``, which tracks *persistence* slot numbers for session
state save/restore.

The value is always the bare session-id **string**.  Per-endpoint dispatch
leases are keyed by ``(endpoint, session_id)`` tuples; the tuple key is
normalised through ``_dispatch_key_session_id()`` before it is stored here,
so the SSE/status payload never exposes a composite array as a session id
(WL-0MUKY4MAX0008I0L).

Cleaned up when dispatch leases expire.
"""


# ===================================================================
# Counters, tokens, status broadcast and persistence
# ===================================================================

async def broadcast_status(event_type: str, data: dict):
    """Broadcast a status event to all connected SSE clients.

    All events include a ``type`` field, an ISO8601 ``timestamp``,
    and any additional fields provided in *data*.
    """
    now = datetime.now(UTC).isoformat(timespec="seconds")
    event_data = json.dumps({"type": event_type, "timestamp": now, **data})
    message = f"data: {event_data}\n\n"

    # Send to all connected clients
    dead_clients = set()
    for queue in sse_clients:
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            dead_clients.add(queue)

    # Clean up dead clients
    for client in dead_clients:
        sse_clients.discard(client)



def broadcast_status_sync(event_type: str, data: dict):
    """Synchronous wrapper to broadcast status (for use in sync code)."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(broadcast_status(event_type, data))
    except RuntimeError:
        # No running loop, skip broadcast
        pass


async def broadcast_session_event(
    session_id: str,
    event: str,
    model: str = "",
    details: str = "",
) -> None:
    """Broadcast a session lifecycle event (created, ended, restored).

    Args:
        session_id: The session identifier.
        event: The event type (e.g. ``"session_created"``, ``"session_ended"``).
        model: Optional model name associated with the session.
        details: Optional human-readable detail message.
    """
    await broadcast_status(
        "session_activity",
        {
            "session_id": session_id,
            "event": event,
            "message": f"Session {event}: {session_id[:16] if session_id else 'unknown'}",
        },
    )
    # Increment session activity counter
    try:
        srv = _srv()
        srv.session_observability["session_activity_total"] = (
            int(srv.session_observability.get("session_activity_total", 0)) + 1
        )
    except Exception:
        pass


async def broadcast_provider_event(
    provider_name: str,
    event: str,
    model: str = "",
    details: str = "",
) -> None:
    """Broadcast a provider lifecycle event (fallback, cooldown, recovery).

    Args:
        provider_name: The provider identifier.
        event: The event type (e.g. ``"provider_fallback"``, ``"provider_cooldown"``).
        model: Optional model name associated with the event.
        details: Optional human-readable detail message.
    """
    data = {
        "provider": provider_name,
        "event": event,
        "message": f"Provider {event}: {provider_name}",
    }
    if model:
        data["model"] = model
    if details:
        data["details"] = details
    await broadcast_status("provider_fallback", data)
    # Increment provider fallback counter
    try:
        srv = _srv()
        if event in ("provider_fallback", "provider_cooldown"):
            bucket = srv.provider_fallback_count.get(provider_name, 0)
            srv.provider_fallback_count[provider_name] = bucket + 1
    except Exception:
        pass



def _counts_file_path() -> Path:
    """Return path to the persisted counts file inside log_dir (or local logs)."""
    srv = _srv()
    if srv.log_dir:
        return srv.log_dir / srv.counts_filename
    return Path(__file__).parent / "logs" / srv.counts_filename



def load_counts():
    srv = _srv()
    try:
        path = _counts_file_path()
        if path.exists():
            with open(path, encoding="utf-8") as f:
                srv.request_counts = json.load(f)
        else:
            srv.request_counts = {}
    except Exception:
        srv.request_counts = {}



def save_counts_sync():
    srv = _srv()
    try:
        path = _counts_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(srv.request_counts, f, indent=2)
        tmp.replace(path)
    except Exception as e:
        srv.logger.error(f"Failed to persist request counts: {e}")



def _token_file_path() -> Path:
    srv = _srv()
    if srv.log_dir:
        return srv.log_dir / srv.token_counts_filename
    return Path(__file__).parent / "logs" / srv.token_counts_filename



def load_token_counts():
    srv = _srv()
    try:
        path = _token_file_path()
        if path.exists():
            with open(path, encoding="utf-8") as f:
                srv.token_counts = json.load(f)
        else:
            srv.token_counts = {}
    except Exception:
        srv.token_counts = {}



def save_token_counts_sync():
    srv = _srv()
    try:
        path = _token_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(srv.token_counts, f, indent=2)
        tmp.replace(path)
    except Exception as e:
        srv.logger.error(f"Failed to persist token counts: {e}")



async def save_token_counts():
    await asyncio.to_thread(save_token_counts_sync)



async def save_counts():
    await asyncio.to_thread(save_counts_sync)



async def _counts_persist_loop():
    srv = _srv()
    try:
        while True:
            await asyncio.sleep(2.0)
            if srv.counts_dirty:
                try:
                    await save_counts()
                    srv.counts_dirty = False
                except Exception:
                    pass
    finally:
        srv.counts_persist_task = None



async def _tokens_persist_loop():
    srv = _srv()
    try:
        while True:
            await asyncio.sleep(2.0)
            if srv.tokens_dirty:
                try:
                    await save_token_counts()
                    srv.tokens_dirty = False
                except Exception:
                    pass
    finally:
        srv.tokens_persist_task = None



async def query_llama_status() -> dict:
    """
    Query llama-server HTTP endpoints for model metadata.

    Attempts HTTP GET to /model then /status and returns parsed JSON if successful.
    If those endpoints are not present or do not include n_ctx / KV cache values,
    returns null for those fields.

    Returns dict with:
      - n_ctx: max context size (int) or None
      - kv_cache_tokens: KV cache token count (int) or None
      - llama_server_running: bool
      - router_mode: bool
    """
    # allow updating/reading endpoint cache/failures
    srv = _srv()

    result = {
        "n_ctx": None,
        "kv_cache_tokens": None,
        "llama_server_running": srv.llama_process is not None and srv.llama_process.poll() is None,
        "router_mode": False
    }

    if not result["llama_server_running"]:
        return result

    server_config = srv.config.get("server", {})
    llama_port = server_config.get("llama_server_port", 8080)

    # One-time discovery: probe metadata endpoints once per llama-server process
    client = srv._http_client if srv._http_client else httpx.AsyncClient(timeout=5.0)
    try:
        async def _do_discovery_if_needed():
            """Discover a working metadata endpoint once per process.

            Returns a (n_ctx, kv_cache_tokens) tuple discovered during probing
            so callers can use the data without re-requesting mocked responses
            (important for tests that provide a sequence of responses).
            """
            # If discovery was already done for this pid, skip.
            try:
                current_pid = getattr(srv.llama_process, 'pid', None)
            except Exception:
                current_pid = None
            if srv._llama_status_discovered and srv._llama_status_discovered_pid == current_pid:
                return None, None

            endpoints = ["/model", "/status", "/models", "/v1/models"]
            found_n = None
            found_kv = None
            for endpoint in endpoints:
                try:
                    url = _build_llama_url(llama_port, endpoint)
                    resp = await client.get(url, timeout=5.0)
                    if getattr(resp, 'status_code', None) == 200:
                        # remember endpoint
                        if not srv._llama_status_endpoint_cache:
                            srv._llama_status_endpoint_cache = endpoint
                        srv._llama_status_endpoint_failures.pop(endpoint, None)

                        # attempt to parse JSON/text for n_ctx / kv fields
                        data = await _safe_parse_json_response(resp)

                        if isinstance(data, dict):
                            if found_n is None:
                                found_n = data.get("n_ctx") or data.get("n_ctx_total")
                            if found_kv is None:
                                found_kv = data.get("kv_cache_tokens") or data.get("kv_cache_token_count")

                        # If we've discovered both values, stop probing
                        if found_n is not None and found_kv is not None:
                            break
                except Exception:
                    # ignore and try next
                    continue

            # Mark discovery done for this pid even if none found
            srv._llama_status_discovered = True
            try:
                srv._llama_status_discovered_pid = getattr(srv.llama_process, 'pid', None)
            except Exception:
                srv._llama_status_discovered_pid = None

            return found_n, found_kv

        # Run discovery and capture any discovered metadata so we can
        # return parsed values immediately (important for tests that
        # provide a sequence of mocked responses consumed during discovery).
        found_n, found_kv = await _do_discovery_if_needed()
        if found_n is not None:
            result["n_ctx"] = found_n
        if found_kv is not None:
            result["kv_cache_tokens"] = found_kv

        # If we have a cached endpoint, try it first
        if srv._llama_status_endpoint_cache:
            try:
                url = _build_llama_url(llama_port, srv._llama_status_endpoint_cache)
                response = await client.get(url, timeout=5.0)
                if getattr(response, 'status_code', None) == 200:
                    data = await _safe_parse_json_response(response)

                    if isinstance(data, dict):
                        if result["n_ctx"] is None:
                            result["n_ctx"] = data.get("n_ctx") or data.get("n_ctx_total")
                        if result["kv_cache_tokens"] is None:
                            result["kv_cache_tokens"] = (
                                data.get("kv_cache_tokens") or data.get("kv_cache_token_count")
                            )
                else:
                    # cache no longer valid; clear and allow future rediscovery
                    srv._llama_status_endpoint_failures[srv._llama_status_endpoint_cache] = time.time()
                    srv._llama_status_endpoint_cache = None
                    srv._llama_status_discovered = False
                    srv._llama_status_discovered_pid = None
            except Exception:
                # ignore and fallthrough to props check
                pass

        # As a fallback, check /props to detect router mode
        try:
            props_url = _build_llama_url(llama_port, "/props")
            response = await client.get(props_url, timeout=5.0)
            if getattr(response, "status_code", None) == 200:
                props = await _safe_parse_json_response(response)
                if isinstance(props, dict):
                    result["router_mode"] = True
        except Exception:
            pass
    finally:
        if not srv._http_client:
            try:
                await client.aclose()
            except Exception:
                pass

    return result


# ===================================================================
# Slot progress cache – supplemental token counts from llama-server log
# ===================================================================


def _update_slot_progress_from_log() -> None:
    """Read the tail of llama-server.log and extract per-slot progress data.

    llama-server writes progress lines like::

        slot update_slots: id=5 n_tokens=4096 progress=0.17

    These are parsed by ``extract_progress_data()`` (from ``handlers``)
    and stored in the module-level ``_slot_progress_cache`` keyed by
    ``slot_id``.

    When the log file cannot be read (e.g. not yet created, missing
    permissions) the cache is simply not updated — no exception is raised.
    """
    srv = _srv()
    try:
        log_dir = getattr(srv, "log_dir", None)
        if log_dir is None:
            log_path = Path(__file__).parent / "logs" / "llama-server.log"
        else:
            log_path = Path(log_dir) / "llama-server.log"

        if not log_path.exists():
            return

        # Read the last 64KB to find progress lines
        stat = log_path.stat()
        read_size = min(65536, stat.st_size)
        with open(log_path, "rb") as f:
            if read_size < stat.st_size:
                f.seek(stat.st_size - read_size)
            # Skip to first newline to avoid partial-line reads
            if read_size < stat.st_size:
                f.readline()
            tail = f.read().decode("utf-8", errors="replace")

        now = time.time()
        # Stamp each parsed entry with the log's real modification time, not
        # the read time (LP-0MVECWYWK000U3E1).  A frozen log (e.g. an adopted
        # llama-server whose stdout pipe was never re-attached) is otherwise
        # re-parsed every cycle and re-stamped as "now", pinning the UI at
        # stale token counts forever.  Clamp a future mtime (clock skew) to
        # now so it cannot keep entries alive indefinitely either.
        source_time = stat.st_mtime
        if source_time > now:
            source_time = now
        for line in tail.splitlines():
            result = _extract_progress_data_from_log(line)
            if result is not None:
                slot_id, n_tokens, progress = result
                _slot_progress_cache[slot_id] = {
                    "n_tokens": n_tokens,
                    "progress": progress,
                    "timestamp": source_time,
                }

        # Prune entries older than the trust window (generations don't last
        # that long, and stale entries would show bogus token counts).
        stale = [sid for sid, data in _slot_progress_cache.items()
                 if not _slot_progress_is_fresh(data, now)]
        for sid in stale:
            _slot_progress_cache.pop(sid, None)
    except Exception:
        pass


def _extract_progress_data_from_log(line: str) -> tuple | None:
    """Lightweight inline parser for llama-server progress log lines.

    Extracts ``(slot_id, n_tokens, progress)`` from lines like::

        slot update_slots: id=5 n_tokens=4096 progress=0.17

    Returns ``None`` if the line does not contain valid progress data.
    """
    if not isinstance(line, str):
        return None
    text = line.strip()
    if not text or "n_tokens" not in text or "progress" not in text:
        return None
    try:
        m_slot = re.search(
            r'slot\s+update_slots:.*?id\s+(\d+)|slot\s+(\d+)',
            text, flags=re.IGNORECASE,
        )
        m_tokens = re.search(r'\bn_tokens\s*=\s*(\d+)\b', text, flags=re.IGNORECASE)
        m_progress = re.search(
            r'\bprogress\s*=\s*([0-9]+(?:\.[0-9]+)?)\b',
            text, flags=re.IGNORECASE,
        )
        if not m_tokens or not m_progress:
            return None
        slot_id = int(m_slot.group(1)) if m_slot and m_slot.group(1) is not None else (
            int(m_slot.group(2)) if m_slot else 0
        )
        return (slot_id, int(m_tokens.group(1)), float(m_progress.group(1)))
    except Exception:
        return None


def _slot_progress_is_fresh(prog: dict | None, now: float | None = None) -> bool:
    """Return True when *prog* is within the log-progress trust window.

    ``prog`` is an entry from ``_slot_progress_cache`` whose ``timestamp``
    is the source log's mtime at read time (LP-0MVECWYWK000U3E1).  Missing
    or non-numeric timestamps are treated as stale so malformed data can
    never be served as live.
    """
    if not isinstance(prog, dict):
        return False
    ts = prog.get("timestamp")
    if not isinstance(ts, (int, float)) or isinstance(ts, bool):
        return False
    if now is None:
        now = time.time()
    return (now - ts) <= SLOT_PROGRESS_MAX_AGE_SECONDS


def _update_processing_slot_assignments(srv, slot_details=None) -> None:
    """Update ``_processing_slot_assignments`` by matching active dispatch
    leases against slots that are processing (from progress cache OR
    ``/slots`` API).

    Each active dispatch session is paired with a processing slot that
    has recent progress data or is reported as processing by the
    ``/slots`` endpoint, in order of dispatch start time.  This gives
    us a working ``{llama_server_slot_id: session_id}`` mapping for the
    UI without relying on the persistence slot registry.

    Args:
        srv: The server module (from ``_srv()``).
        slot_details: Current slot details from ``_query_slots_detail()``
            or ``_last_slot_details_cache``.  When provided, slots that
            are ``is_processing == True`` but lack progress cache entries
            are included as candidates for pairing.
    """
    global _processing_slot_assignments
    try:
        now = time.time()
        # Get active dispatch sessions sorted by start time
        records = getattr(srv, "local_dispatch_records", {})

        # Normalise every dispatch-lease key to its bare session-id string
        # before it is exposed as a session id (WL-0MUKY4MAX0008I0L).
        # Per-endpoint leases are keyed by ``(endpoint, session_id)`` tuples;
        # storing the raw key leaked the tuple into the SSE payload as a JSON
        # array, which the Web UI could not treat as a string. The records
        # themselves stay tuple-keyed — only the emitted value changes.
        from proxy.router_helpers import _dispatch_key_session_id

        active_sessions: list[tuple] = []
        for sid_key, info in list(records.items()):
            if info.get("active"):
                session_id = _dispatch_key_session_id(sid_key)
                if not isinstance(session_id, str):
                    continue
                active_sessions.append((info.get("started_at", 0), session_id))
        active_sessions.sort(key=lambda x: x[0])  # oldest first

        # Collect candidate slots from TWO sources:
        # 1. Slots with recent progress data in the cache
        # 2. Slots that /slots reports as processing (may not have progress yet)
        candidate_slots: set[int] = set()

        # Source 1: progress cache (recent, within the trust window)
        for sid, data in _slot_progress_cache.items():
            if _slot_progress_is_fresh(data, now):
                candidate_slots.add(sid)

        # Source 2: /slots API reports as processing
        if slot_details:
            for sd in slot_details:
                if sd.get("is_processing"):
                    sid = sd.get("slot_id")
                    if sid is not None:
                        candidate_slots.add(sid)

        candidate_sorted = sorted(candidate_slots)

        # Build reverse: session_id -> slot_id for already-mapped entries
        session_to_slot: dict[str, int] = {
            s: p for p, s in _processing_slot_assignments.items()
        }

        # Assign unmatched sessions to unmatched candidate slots
        i = 0
        for _, session_id in active_sessions:
            if session_id in session_to_slot:
                continue  # already mapped
            # Find next candidate slot not yet assigned
            while i < len(candidate_sorted):
                ps = candidate_sorted[i]
                i += 1
                if ps not in _processing_slot_assignments:
                    _processing_slot_assignments[ps] = session_id
                    session_to_slot[session_id] = ps
                    break

        # Clean up: remove assignments for sessions no longer active
        session_ids = {s for _, s in active_sessions}
        stale = [
            ps for ps, s in _processing_slot_assignments.items()
            if s not in session_ids
        ]
        for ps in stale:
            _processing_slot_assignments.pop(ps, None)

        # Clean up: remove assignments for slots no longer in candidate set
        stale_candidates = [
            ps for ps in _processing_slot_assignments
            if ps not in candidate_sorted
        ]
        for ps in stale_candidates:
            _processing_slot_assignments.pop(ps, None)
    except Exception:
        pass


def _build_slot_to_session_map(srv, slot_details=None) -> dict:
    """Return the current ``_processing_slot_assignments`` mapping from
    llama-server processing slot_id to session_id.

    This is populated by ``_update_processing_slot_assignments()`` which
    matches progress data (llama-server slot) against active dispatch
    leases (session_id).  It is separate from ``_slot_owners`` in
    ``proxy.session``, which tracks *persistence* slot numbers for
    session state save/restore.

    Args:
        srv: The server module (from ``_srv()``).
        slot_details: Optional list of slot dicts from ``/slots`` API.
            Passed through to ``_update_processing_slot_assignments``.
    """
    _update_processing_slot_assignments(srv, slot_details=slot_details)
    return dict(_processing_slot_assignments)


def _build_slot_to_lease_map(srv) -> dict[int, dict]:
    """Map each llama-server slot to the dispatch lease holding it.

    Slot persistence registry (``proxy.session._slot_owners``) binds a
    llama-server slot id to the session that owns it; each owning session's
    dispatch record (``srv.local_dispatch_records``) carries the lease
    ``active`` flag and ``expires_at`` (monotonic). This merges the two so
    the WebUI can show, per slot, which work item/session holds the lease
    and how long it remains (LP-0MTCZ35X7009IZKE AC2).

    A slot is included only when its owning session has a dispatch record
    (active or inactive/unexpired) — i.e. when a lease actually reserves it.

    Args:
        srv: The server module (from ``_srv()``).

    Returns:
        ``{slot_id: {"session_id": str, "active": bool,
        "remaining_seconds": float}}``.
    """
    try:
        from proxy.session import _slot_owners
    except Exception:
        _slot_owners = {}
    records = getattr(srv, "local_dispatch_records", {}) or {}
    result: dict[int, dict] = {}
    now = time.monotonic()
    for slot_id, session_id in list(_slot_owners.items()):
        rec = records.get(session_id)
        if not rec:
            continue
        expires_at = rec.get("expires_at", 0) or 0
        remaining = max(0.0, expires_at - now)
        result[slot_id] = {
            "session_id": session_id,
            "active": bool(rec.get("active")),
            "remaining_seconds": round(remaining, 1),
        }
    return result


# ===================================================================
# Truthful local capacity (LP-0MVGDUVGE0024YAI)
#
# ``/llama/local/status`` publishes a per-slot ``lease_state`` and ``phase``
# plus a top-level ``capacity_hint`` so a dispatcher can distinguish a slot
# that is *generating* from one that is merely *leased but idle* (the
# "reserved-but-idle GPU" false-busy case: 2,633 of 2,641 local dispatch
# denials had ``active=2`` lease records while local decoded only 29.3 % of
# wall clock). The helpers here are pure so they can be unit-tested against
# the decision tables without an HTTP request; the annotation below wires
# them to live dispatch-lease state.
# ===================================================================


LEASE_STATE_GENERATING = "generating"
LEASE_STATE_IDLE_LEASED = "idle_leased"
LEASE_STATE_FREE = "free"

PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"

CAPACITY_IDLE = "idle"
CAPACITY_PREFILL_BOUND = "prefill_bound"
CAPACITY_SATURATED = "saturated"


def slot_lease_state(is_processing: bool, owner_session_id: str | None) -> str:
    """Classify a slot's lease state (LP-0MVGDUVGE0024YAI AC2).

    - ``generating`` — the slot is processing (prefill or decode in flight).
      This takes precedence so a working slot is NEVER reported ``free``.
    - ``idle_leased`` — not processing, but a session still holds a live
      (active or post-stream cooldown) lease on the slot: the reserved-but-idle
      case a dispatcher must not confuse with genuine GPU work.
    - ``free`` — not processing and no lease holder.
    """
    if is_processing:
        return LEASE_STATE_GENERATING
    if owner_session_id:
        return LEASE_STATE_IDLE_LEASED
    return LEASE_STATE_FREE


def infer_slot_phase(
    is_processing: bool,
    n_decoded: int | None,
    context_tokens: int | None = None,
) -> str | None:
    """Infer a slot's generation phase from ``/slots`` signals (AC5).

    llama-server exposes no phase field, so it is derived proxy-side from
    ``is_processing`` and ``next_token.n_decoded`` (with prompt progress —
    ``context_tokens`` — as corroboration on builds that still report it):

    - ``None`` when the slot is not processing.
    - ``prefill`` while processing has not produced any decoded token
      (``n_decoded`` is ``None`` or ``0``) — the prompt is still being
      consumed, including the transition where prompt progress advances while
      ``n_decoded`` stays at zero.
    - ``decode`` once tokens are being generated (``n_decoded > 0``).

    The heuristic is deliberately conservative for very short prefills: a
    processing slot with no decode signal at all is reported ``prefill``
    rather than guessed as ``decode``.
    """
    if not is_processing:
        return None
    try:
        decoded = int(n_decoded) if n_decoded is not None else 0
    except (TypeError, ValueError):
        decoded = 0
    if decoded > 0:
        return PHASE_DECODE
    return PHASE_PREFILL


def compute_capacity_hint(slots: list[dict]) -> str:
    """Summarise whether the local slot pool is genuinely working (AC4).

    Decision table, evaluated **in order**:

    1. ``idle`` — at least one slot is ``free`` **or** every occupied slot is
       ``idle_leased`` (nothing is actually being generated).
    2. ``prefill_bound`` — no slot is free and at least one occupied slot is
       in ``prefill``.
    3. ``saturated`` — no slot is free and every occupied slot is in
       ``decode``.

    Two documented extensions cover states the table leaves open: an empty
    ``slots`` array (server down / no model / no cached detail) is ``idle``
    (fail-open — never claim saturation with no evidence), and a pool with no
    free slot, no prefill and mixed ``decode``/``idle_leased`` occupancy is
    ``saturated`` (genuine decode work with nothing available to a newcomer).
    """
    if not slots:
        return CAPACITY_IDLE
    states = [str(s.get("lease_state") or LEASE_STATE_FREE) for s in slots]
    if LEASE_STATE_FREE in states:
        return CAPACITY_IDLE
    occupied = [
        s
        for s in slots
        if str(s.get("lease_state") or LEASE_STATE_FREE) != LEASE_STATE_FREE
    ]
    if not occupied:
        return CAPACITY_IDLE
    if all(str(s.get("lease_state")) == LEASE_STATE_IDLE_LEASED for s in occupied):
        return CAPACITY_IDLE
    if any(s.get("phase") == PHASE_PREFILL for s in occupied):
        return CAPACITY_PREFILL_BOUND
    if all(s.get("phase") == PHASE_DECODE for s in occupied):
        return CAPACITY_SATURATED
    # No free slot, no prefill: either every occupied slot is decoding or the
    # remaining holders are idle leases alongside a decoding slot. Both mean
    # the pool is genuinely working with no capacity for a newcomer.
    if any(str(s.get("lease_state")) == LEASE_STATE_GENERATING for s in occupied):
        return CAPACITY_SATURATED
    return CAPACITY_IDLE


def _build_status_slot_owners(srv) -> dict[int, dict]:
    """Map llama-server slot_id -> the dispatch lease reserving it.

    Used by ``/llama/local/status`` to publish a truthful per-slot
    ``owner_session_id`` / ``lease_state`` (LP-0MVGDUVGE0024YAI AC2). The
    slot registry (``proxy.session._slot_owners``) binds a session to a
    llama-server slot id; each owning session's dispatch record carries the
    lease ``active`` flag and ``expires_at``. Records are normalised through
    ``_dispatch_key_session_id`` so both legacy bare string keys and
    per-endpoint ``(endpoint, session_id)`` tuple keys resolve
    (WL-0MUKY4MAX0008I0L).

    A slot is included only while its lease is live: active, or inactive but
    still inside its post-stream cooldown (``expires_at`` in the future) — the
    exact window in which an idle lease holder looks busy to a naive consumer.

    Returns ``{slot_id: {"session_id", "active", "remaining_seconds"}}``;
    best-effort (never raises), empty when nothing is leased.
    """
    try:
        from proxy.router_helpers import _dispatch_key_session_id
        from proxy.session import (
            _slot_key_endpoint,
            _slot_key_slot_id,
            _slot_owners,
        )

        records = dict(getattr(srv, "local_dispatch_records", {}) or {})
        now = time.monotonic()

        # Normalise the dispatch records to ``session_id -> record``. A
        # session may carry both a bare and a tuple-keyed record; an active
        # record wins over an inactive one for the same session.
        by_session: dict[str, dict] = {}
        for key, rec in records.items():
            if not isinstance(rec, dict):
                continue
            session_id = _dispatch_key_session_id(key)
            if not isinstance(session_id, str) or not session_id:
                continue
            previous = by_session.get(session_id)
            if previous is None or (not previous.get("active") and rec.get("active")):
                by_session[session_id] = rec

        result: dict[int, dict] = {}
        for key, session_id in list(_slot_owners.items()):
            if not isinstance(session_id, str) or not session_id:
                continue
            slot_id = _slot_key_slot_id(key)
            if not isinstance(slot_id, int):
                continue
            rec = by_session.get(session_id)
            if not rec:
                # A slot binding without a dispatch record is a stale
                # registry entry, not a lease: do not report it as leased.
                continue
            remaining = max(0.0, float(rec.get("expires_at", 0) or 0) - now)
            active = bool(rec.get("active"))
            if not active and remaining <= 0:
                continue  # expired lease: the slot is genuinely free
            endpoint = _slot_key_endpoint(key)
            # A slot id can be registered on several backends; prefer the
            # local (empty-endpoint) registry entry for the local status.
            if slot_id in result and endpoint:
                continue
            result[slot_id] = {
                "session_id": session_id,
                "active": active,
                "remaining_seconds": round(remaining, 1),
            }
        return result
    except Exception:
        return {}


def annotate_slots_for_status(
    slot_details: list[dict],
    srv,
    model: str | None,
    config: dict | None = None,
    stale: bool = False,
) -> list[dict]:
    """Annotate compact ``/slots`` dicts with capacity fields (AC2/AC3/AC5).

    Adds, per slot: ``owner_session_id``, ``lease_state``, ``phase``,
    ``context_tokens`` and ``prefill_eta_seconds``. Never mutates the input
    dicts (the per-slot detail cache must stay pristine) and never raises —
    a failure leaves the base fields intact. Per-slot ETAs are ``None`` when
    the underlying ``/slots`` poll is stale (AC7) rather than echoing a stale
    number, and the estimate comes from the shared ``prefill_eta`` calibration
    (AC3), never a second estimator.
    """
    if not slot_details:
        return []
    try:
        from proxy.prefill_eta import per_slot_prefill_eta_seconds
    except Exception:  # pragma: no cover - defensive
        per_slot_prefill_eta_seconds = None

    try:
        owners = _build_status_slot_owners(srv)
    except Exception:
        owners = {}
    try:
        # Secondary binding: dispatch leases matched to processing slots
        # (covers a processing slot whose persistence binding is absent).
        processing_map = _build_slot_to_session_map(srv, slot_details=slot_details)
    except Exception:
        processing_map = {}

    annotated: list[dict] = []
    for slot in slot_details:
        entry = dict(slot) if isinstance(slot, dict) else {}
        slot_id = entry.get("slot_id")
        is_processing = bool(entry.get("is_processing", False))

        owner_session_id = None
        lease = owners.get(slot_id) if slot_id is not None else None
        if lease:
            owner_session_id = lease.get("session_id")
        elif slot_id in processing_map:
            owner_session_id = processing_map.get(slot_id)

        phase = infer_slot_phase(
            is_processing, entry.get("n_decoded"), entry.get("context_tokens")
        )

        prefill_eta = None
        if (
            not stale
            and per_slot_prefill_eta_seconds is not None
            and phase == PHASE_PREFILL
        ):
            prefill_eta = per_slot_prefill_eta_seconds(
                model, entry.get("context_tokens"), config
            )

        entry["owner_session_id"] = owner_session_id
        entry["lease_state"] = slot_lease_state(is_processing, owner_session_id)
        entry["phase"] = phase
        # Always present (defaults to None for pre-change cache entries / builds
        # that no longer report prompt progress) so the payload contract is stable.
        entry["context_tokens"] = entry.get("context_tokens")
        entry["prefill_eta_seconds"] = prefill_eta
        annotated.append(entry)
    return annotated


def _resolve_slot_details_for_enrichment(slot_details: list[dict] | None) -> list[dict]:
    """Return a detached working copy of slot details, updating the cache.

    On a successful fresh ``/slots`` query (*slot_details* truthy) a pristine
    copy of the raw values is stored in the module-level
    ``_last_slot_details_cache``; on a failed query the last-known cache is
    used.  Either way, a *copy* is returned so that
    ``_enrich_slot_details_with_progress()`` mutating the working list can
    never bake log-derived fields into the fallback cache
    (LP-0MVECWYWK000U3E1, AC2).

    The cache list object is preserved (slice assignment) so any module
    holding a reference to it keeps observing updates.
    """
    if slot_details:
        _last_slot_details_cache[:] = [dict(s) for s in slot_details]
    return [dict(s) for s in _last_slot_details_cache]


def _enrich_slot_details_with_progress(slot_details: list[dict],
                                        srv=None) -> list[dict]:
    """Merge log-parsed progress n_tokens into slot_details.

    For each slot in *slot_details*, if the progress cache has a
    fresher *n_tokens* value than the API's *n_decoded*, override
    *n_decoded* with the progress value (and mark *is_processing* if
    progress is > 0).

    Also injects ``n_tokens``, ``progress``, ``total_tokens``, and
    ``session_id`` into each slot dict so the frontend can display
    "Processed x of y (z%)" with the owning session.

    Args:
        slot_details: List of slot dicts from ``_query_slots_detail()``
            or ``_last_slot_details_cache``.
        srv: Optional server module. If provided, ``session_id`` is
            resolved from dispatch records.

    Returns:
        The same list (modified in-place) with enriched token counts.

    Note:
        Enrichment mutates the list it is given.  Callers serving the
        fallback cache must pass a copy from
        ``_resolve_slot_details_for_enrichment()`` so the pristine
        ``/slots`` values in ``_last_slot_details_cache`` are never poisoned
        by log-derived fields (LP-0MVECWYWK000U3E1, AC2).
    """
    if not slot_details:
        return slot_details

    # Build slot→session mapping once for this batch
    slot_to_session: dict = {}
    if srv is not None:
        try:
            slot_to_session = _build_slot_to_session_map(srv, slot_details=slot_details)
        except Exception:
            pass

    now = time.time()
    # Resolve dispatch-lease metadata once for this batch (LP-0MTCZ35X7009IZKE
    # AC2): maps each llama-server slot to the session holding an active or
    # inactive (post-request cooldown) dispatch lease, plus its remaining time.
    slot_to_lease: dict = {}
    if srv is not None:
        try:
            slot_to_lease = _build_slot_to_lease_map(srv)
        except Exception:
            pass
    for slot in slot_details:
        sid = slot.get("slot_id")
        if sid is None:
            continue

        # Attach session_id if we found an active dispatch for this slot
        if sid in slot_to_session:
            slot["session_id"] = slot_to_session[sid]

        # Attach lease metadata (owner session + remaining seconds + active flag)
        lease = slot_to_lease.get(sid)
        if lease:
            slot["lease_session_id"] = lease["session_id"]
            slot["lease_active"] = lease["active"]
            slot["lease_remaining_seconds"] = lease["remaining_seconds"]

        prog = _slot_progress_cache.get(sid)
        if prog is None:
            continue
        # Only use progress data whose *source* is still fresh.  The
        # timestamp is the log's mtime, so a frozen log is rejected here even
        # if a caller injected it directly (LP-0MVECWYWK000U3E1, AC3).
        if not _slot_progress_is_fresh(prog, now):
            continue
        n_tokens = prog["n_tokens"]
        pct = prog["progress"]

        # Inject progress fields for frontend display
        slot["n_tokens"] = n_tokens
        slot["progress"] = round(pct, 3)
        if pct > 0:
            slot["total_tokens"] = int(round(n_tokens / pct))
        else:
            slot["total_tokens"] = None

        # Use the larger of API n_decoded and log n_tokens
        api_n_decoded = slot.get("n_decoded")
        if n_tokens is not None and (
            api_n_decoded is None or n_tokens > api_n_decoded
        ):
            slot["n_decoded"] = n_tokens

        # --- Generation-complete detection via n_tokens stability ---
        # When n_tokens stops increasing for > 3s *and* prefill is complete,
        # generation is done and the session is streaming the response to
        # the client.  Requiring prefill completion prevents a prefill-phase
        # pause from being mistaken for completion, which made the Web UI
        # flip to 'Working' and back to a prefill increment
        # (WL-0MSZVB902004AZZN).
        total_tokens = slot.get("total_tokens")
        prefill_complete = (
            total_tokens is not None
            and n_tokens >= total_tokens - GENERATION_COMPLETE_TOKEN_MARGIN
        )
        stable = _slot_stable_tracker.get(sid)
        if stable and stable["n_tokens"] == n_tokens:
            # n_tokens hasn't changed — check how long it's been stable
            if now - stable["stable_since"] > 3.0 and prefill_complete:
                slot["generation_done"] = True
                slot["is_processing"] = False
            else:
                # Still prefilling (or within the stability window): keep
                # reporting progress instead of the 'Working' state.
                slot["generation_done"] = False
                slot["is_processing"] = True
        else:
            # n_tokens increased (or first sighting) — reset tracker
            _slot_stable_tracker[sid] = {"n_tokens": n_tokens, "stable_since": now}
            slot["generation_done"] = False
            # Mark as processing if progress > 0
            if pct > 0:
                slot["is_processing"] = True
    return slot_details



async def _periodic_broadcast_loop():
    """Periodically broadcast current counts/tokens to connected log-tail clients.

    This ensures UI updates even if no direct increment message races occur.
    Also queries llama-server for status and broadcasts stats.
    """
    srv = _srv()
    # Cache the last successful slot query result so that transient
    # timeouts (e.g. during busy token generation) don't blank the UI.
    # Use the module-level cache so that ``status_events()`` in ``proxy.ui``
    # can also survive timeouts on initial SSE connection.  The cache is
    # updated by ``_resolve_slot_details_for_enrichment()``.
    try:
        while True:
            try:
                await asyncio.sleep(1.0)
                snap_c = {}
                snap_t = {}
                async with srv.counts_lock:
                    snap_c = dict(srv.request_counts)
                async with srv.token_lock:
                    snap_t = dict(srv.token_counts)

                llama_status = await query_llama_status()

                total_sent = srv.token_counts.get("total_sent", 0)
                total_recv = srv.token_counts.get("total_recv", 0)

                if log_tail_clients:
                    for q in list(log_tail_clients):
                        try:
                            q.put_nowait({
                                "counts": snap_c,
                                "tokens": snap_t,
                                "llama_status": llama_status,
                                "total_sent": total_sent,
                                "total_recv": total_recv
                            })
                        except asyncio.QueueFull:
                            continue

                # --- Per-slot data query (best-effort) ---
                slot_details: list[dict] = []
                server_running = llama_status.get("llama_server_running")
                if server_running:
                    try:
                        server_cfg = srv.config.get("server", {})
                        llama_port = int(server_cfg.get("llama_server_port", 8080) or 8080)
                        # Prefer the discovered local child port: the router
                        # serializes /slots?model=... behind the busy child
                        # (LP-0MTDGBRPU003Z7KU, 5-7s vs 0.17s direct), so the
                        # broadcast SSE loop must query the model instance
                        # directly to avoid the 6,865/day router 500 storm
                        # (LP-0MTIHZ8M5005ZAU8 / F3 triage rank 1).
                        # Use current_model as the model param for /slots
                        model_name = srv.current_model or None
                        try:
                            from proxy.router_helpers import _discover_local_child_port
                            # LP-0MTP1FQXH004JYEF: filter to the current model
                            # so the SSE broadcast does not query the embed
                            # child (idle 256-ctx slots) when Qwen3 is busy.
                            _child = _discover_local_child_port(srv, model=model_name)
                            if _child is not None:
                                llama_port = _child
                        except Exception:
                            pass
                        # 5s timeout: llama-server may be slow to respond
                        # to /slots when busy generating tokens. The single-flight
                        # helper bounds a follower's wait too, so a cancelled leader
                        # can no longer stall this loop for every SSE client
                        # (LP-0MVJML600003CQ6Y, AC5).
                        slot_details = await _query_slots_detail(
                            llama_port, timeout=5.0, model=model_name,
                        )
                    except Exception:
                        pass

                # Read llama-server log for supplemental progress data
                _update_slot_progress_from_log()

                # Preserve last known slot data when query fails or llama-server
                # is too busy to respond (ReadTimeout during token generation).
                # Enrich a *copy* so the fallback cache keeps the pristine
                # /slots values instead of the log-derived ones
                # (LP-0MVECWYWK000U3E1, AC2).
                slot_details = _resolve_slot_details_for_enrichment(slot_details)

                # Merge log-parsed progress token counts and session IDs into
                # slot details. This ensures the web UI shows real token counts
                # even when the /slots endpoint is unresponsive during busy
                # generation, and identifies the owning session.
                _enrich_slot_details_with_progress(slot_details, srv=srv)

                if sse_clients:
                    # Snapshot per-model and per-provider queries for SSE broadcast
                    try:
                        async with srv.per_model_queries_lock:
                            per_model_snapshot = dict(srv.per_model_queries)
                    except Exception:
                        per_model_snapshot = {}
                    status_data = {
                        "type": "status",
                        "current_model": srv.current_model,
                        "llama_server_running": llama_status["llama_server_running"],
                        "n_ctx": llama_status["n_ctx"],
                        "kv_cache_tokens": llama_status["kv_cache_tokens"],
                        "total_sent": total_sent,
                        "total_recv": total_recv,
                        "per_model_queries": per_model_snapshot,
                        "slots": slot_details,
                    }
                    event_data = json.dumps(status_data)
                    message = f"data: {event_data}\n\n"
                    dead_clients = set()
                    for q in sse_clients:
                        try:
                            q.put_nowait(message)
                        except asyncio.QueueFull:
                            dead_clients.add(q)
                    for client in dead_clients:
                        sse_clients.discard(client)

            except asyncio.CancelledError:
                break
            except Exception:
                # ignore transient errors
                pass
    finally:
        srv.periodic_broadcast_task = None



async def _increment_count(key: str):
    """Increment the in-memory counter for a request key and persist."""
    srv = _srv()
    try:
        async with srv.counts_lock:
            prev = srv.request_counts.get(key, 0)
            srv.request_counts[key] = prev + 1
            # mark dirty but don't persist immediately; background task will persist
            srv.counts_dirty = True
            srv.logger.debug(f"_increment_count: key={key} prev={prev} new={srv.request_counts[key]}")
    except Exception as e:
        srv.logger.error(f"Error incrementing request count: {e}")
    # Broadcast updated counts to connected log tail clients
    try:
        snapshot = None
        async with srv.counts_lock:
            snapshot = dict(srv.request_counts)

        # Send to all connected log-tail queues
        for q in list(log_tail_clients):
            try:
                q.put_nowait({"counts": snapshot})
            except asyncio.QueueFull:
                # skip slow listeners
                continue
    except Exception:
        pass



async def _increment_count_multi(keys: list[str]):
    """Increment multiple request count keys in one go and broadcast snapshot."""
    srv = _srv()
    try:
        async with srv.counts_lock:
            # log previous values for debugging
            prevs = {k: srv.request_counts.get(k, 0) for k in keys}
            for key in keys:
                srv.request_counts[key] = prevs.get(key, 0) + 1
            srv.counts_dirty = True
            srv.logger.debug(f"_increment_count_multi: keys={keys} prevs={prevs} new_vals={{k: srv.request_counts[k] for k in keys}}")
    except Exception as e:
        srv.logger.error(f"Error incrementing request counts: {e}")

    # Broadcast snapshot
    try:
        snapshot = None
        async with srv.counts_lock:
            snapshot = dict(srv.request_counts)

        for q in list(log_tail_clients):
            try:
                q.put_nowait({"counts": snapshot})
            except asyncio.QueueFull:
                continue
    except Exception:
        pass



async def _increment_tokens(key_prefix: str, key: str, n: int):
    """Increment token counts and persist; key_prefix is 'sent' or 'recv'."""
    srv = _srv()
    try:
        async with srv.token_lock:
            pk = key_prefix + ':' + key
            prev = srv.token_counts.get(pk, 0)
            srv.token_counts[pk] = prev + n
            total_key = 'total_sent' if key_prefix == 'sent' else 'total_recv'
            prev_total = srv.token_counts.get(total_key, 0)
            srv.token_counts[total_key] = prev_total + n
            srv.tokens_dirty = True
            srv.logger.debug(f"_increment_tokens: prefix={key_prefix} key={key} n={n} prev={prev} new={srv.token_counts[pk]} total_prev={prev_total} total_new={srv.token_counts[total_key]}")
    except Exception as e:
        srv.logger.error(f"Error incrementing token counts: {e}")
    # Broadcast token snapshot
    try:
        snap = None
        async with srv.token_lock:
            snap = dict(srv.token_counts)
        for q in list(log_tail_clients):
            try:
                q.put_nowait({"tokens": snap})
            except asyncio.QueueFull:
                continue
    except Exception:
        pass



