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
  endpoint, returning ``(available_slots, total_slots)``.
- ``_query_slots_detail(llama_port, timeout, model)`` — Query the ``/slots``
  endpoint with an optional ``model`` query parameter, returning a list of
  per-slot dicts with keys ``slot_id``, ``is_processing``, and ``n_decoded``.

See LP-0MR6Y11OP005UHIH for the consolidation rationale.
"""

import asyncio
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


def _build_llama_url(llama_port: int, endpoint: str) -> str:
    """Build a URL for a llama-server endpoint.

    Args:
        llama_port: The port llama-server is listening on.
        endpoint: The API path (e.g. ``"/slots"``). A leading slash is
            added if missing.
    """
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    return f"http://localhost:{llama_port}{endpoint}"


async def _query_slots(client, llama_port: int, timeout: float = 2.0) -> tuple:
    """Query the llama-server ``/slots`` endpoint.

    Returns a ``(available_slots, total_slots)`` tuple.  Both default to
    ``0`` on any failure (HTTP error, connection error, timeout, or
    unexpected response shape).

    The 2.0-second default timeout matches the original inline query in
    ``get_llama_local_status()``.
    """
    try:
        url = _build_llama_url(llama_port, "/slots")
        slots_resp = await asyncio.wait_for(client.get(url), timeout=timeout)
        if slots_resp.status_code == 200:
            slots_data = await _safe_parse_json_response(slots_resp)
            if isinstance(slots_data, list):
                total_slots = len(slots_data)
                available_slots = sum(
                    1 for s in slots_data if not s.get("is_processing", True)
                )
                return available_slots, total_slots
    except Exception:
        pass
    return 0, 0


async def _query_slots_detail(
    llama_port: int,
    timeout: float = 2.0,
    model: str | None = None,
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

    Returns an empty list on any failure (HTTP error, connection error,
    timeout, or unexpected response shape).
    """
    try:
        url = _build_llama_url(llama_port, "/slots")
        if model:
            url = f"{url}?model={model}"
        if _client is not None:
            slots_resp = await _client.get(url)
        else:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                slots_resp = await client.get(url)
        if slots_resp.status_code == 200:
            slots_data = await _safe_parse_json_response(slots_resp)
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
                    })
                return result
    except Exception as exc:
        _srv().logger.debug(
            "Slot detail query failed [%s] for %s?model=%s: %s",
            type(exc).__name__, _build_llama_url(llama_port, "/slots"),
            model or "(none)", exc,
        )
    return []


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
Read by ``status_events()`` in ``proxy.ui`` as a fallback when the
initial /slots query times out.
"""


_slot_progress_cache: dict[int, dict] = {}
"""Per-slot progress data parsed from llama-server log lines.

Maps ``slot_id`` to ``{"n_tokens": int, "progress": float,"timestamp": float}``.
Used as a supplemental source of token counts when the ``/slots`` HTTP
endpoint is unresponsive during busy token generation.

Populated by ``_update_slot_progress_from_log()`` which is called by
``_periodic_broadcast_loop()`` on each iteration.
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
        for line in tail.splitlines():
            result = _extract_progress_data_from_log(line)
            if result is not None:
                slot_id, n_tokens, progress = result
                _slot_progress_cache[slot_id] = {
                    "n_tokens": n_tokens,
                    "progress": progress,
                    "timestamp": now,
                }

        # Prune entries older than 60 seconds (generations don't last that
        # long, and stale entries would show bogus token counts).
        stale = [sid for sid, data in _slot_progress_cache.items()
                 if now - data["timestamp"] > 60.0]
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


def _enrich_slot_details_with_progress(slot_details: list[dict]) -> list[dict]:
    """Merge log-parsed progress n_tokens into slot_details.

    For each slot in *slot_details*, if the progress cache has a
    fresher *n_tokens* value than the API's *n_decoded*, override
    *n_decoded* with the progress value (and mark *is_processing* if
    progress is > 0).

    Also injects ``n_tokens``, ``progress``, and ``total_tokens`` into
    each slot dict so the frontend can display "Processed x of y (z%)".

    Args:
        slot_details: List of slot dicts from ``_query_slots_detail()``
            or ``_last_slot_details_cache``.

    Returns:
        The same list (modified in-place) with enriched token counts.
    """
    if not slot_details:
        return slot_details
    now = time.time()
    for slot in slot_details:
        sid = slot.get("slot_id")
        if sid is None:
            continue
        prog = _slot_progress_cache.get(sid)
        if prog is None:
            continue
        # Only use progress data if it's recent (within 60 seconds)
        if now - prog["timestamp"] > 60.0:
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
        # If progress > 0, the slot is actively processing
        if pct > 0 and not slot.get("is_processing"):
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
    # can also survive timeouts on initial SSE connection.
    global _last_slot_details_cache
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
                        # Use current_model as the model param for /slots
                        model_name = srv.current_model or None
                        # 5s timeout: llama-server may be slow to respond
                        # to /slots when busy generating tokens.
                        slot_details = await _query_slots_detail(
                            llama_port, timeout=5.0, model=model_name,
                        )
                    except Exception:
                        pass

                # Read llama-server log for supplemental progress data
                _update_slot_progress_from_log()

                # Preserve last known slot data when query fails or llama-server
                # is too busy to respond (ReadTimeout during token generation).
                if slot_details:
                    _last_slot_details_cache = slot_details
                else:
                    slot_details = _last_slot_details_cache

                # Merge log-parsed progress token counts into slot details.
                # This ensures the web UI shows real token counts even when
                # the /slots endpoint is unresponsive during busy generation.
                _enrich_slot_details_with_progress(slot_details)

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



