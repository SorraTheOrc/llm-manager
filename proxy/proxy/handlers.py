"""
HTTP request handlers for the proxy server, extracted from the monolithic
server.py module.

Each handler is a plain async function that receives a FastAPI ``Request``
object (and optional path/query parameters) and returns a Response.

Module-level access to server globals (config, current_model, etc.) is
performed via lazy import of ``proxy.proxy.server`` at function-call time
to avoid circular-import problems during the transitional refactor period.
"""

import asyncio
import json
import logging
import os
import re
import threading
import time

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from proxy.provider import clear_usage_limit_quarantine, get_model_type

logger = logging.getLogger("llama-proxy")


def _resolve_client_ip(request: Request) -> tuple[str, str]:
    """Resolve the client IP from the request.

    Prefers reverse-proxy headers (X-Forwarded-For, X-Real-IP) when present
    (reverse-proxy safe), falling back to the direct connection address.
    Returns a ``(ip, source)`` tuple where source is ``"header"`` or
    ``"direct"``.
    """
    # X-Forwarded-For: may contain "client, proxy1, proxy2" — take the first
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip(), "header"

    # X-Real-IP: single address set by reverse proxy
    xri = request.headers.get("x-real-ip")
    if xri:
        return xri.strip(), "header"

    # Direct connection
    if request.client:
        return request.client.host, "direct"

    return "unknown", "direct"


def _resolve_client_port(request: Request) -> int | str:
    """Resolve the client source port for structured logging.

    Returns ``request.client.port`` for direct connections, or the string
    ``"unknown"`` when the client identity came from a reverse-proxy header
    (X-Forwarded-For / X-Real-IP — these carry no source port) or when no
    client address/port is present (LP-0MSKV3IEQ004ZV88).

    The header check mirrors ``_resolve_client_ip`` precedence: when a header
    is present the effective client identity is the header value, so the
    direct connection's port must not be attributed to it.
    """
    if request.headers.get("x-forwarded-for") or request.headers.get("x-real-ip"):
        return "unknown"
    if request.client and request.client.port is not None:
        return request.client.port
    return "unknown"


def _resolve_client_id(request: Request) -> str | None:
    """Resolve a stable client identifier from the session headers.

    Returns the first present of ``x-session-id`` / ``session_id`` /
    ``x-client-request-id`` (matching the session-header convention in
    ``proxy/proxy/session.py``), or ``None`` when none are present so the
    field can be omitted — not rendered as ``unknown`` — from the structured
    log (LP-0MSKV3IEQ004ZV88).
    """
    for header_name in ("x-session-id", "session_id", "x-client-request-id"):
        value = request.headers.get(header_name)
        if value:
            return value
    return None

# APIRouter for use by the main server app
router = APIRouter()


# ---------------------------------------------------------------------------
# Voice validation — cache of valid TTS voices
# ---------------------------------------------------------------------------

_valid_voices: list[str] | None = None
"""Cached list of valid TTS voice names from the TTS server. ``None`` means
not yet fetched."""

_voice_cache_time: float = 0.0
"""Timestamp (``time.monotonic()``) when ``_valid_voices`` was last fetched."""

_VOICE_CACHE_TTL: float = 300.0
"""How long (seconds) the voice list is considered fresh."""

_DEFAULT_VOICE = "vivian"
"""Default voice name used when an unrecognized voice is provided."""


async def _ensure_voices_cached() -> list[str] | None:
    """Fetch and cache the valid TTS voice list from ``/v1/voices``.

    Returns the list of valid voice names (strings), or ``None`` if the TTS
    server could not be reached.  The cache has a TTL of ``_VOICE_CACHE_TTL``
    seconds (default 5 minutes).
    """
    global _valid_voices, _voice_cache_time

    now = time.monotonic()
    if _valid_voices is not None and (now - _voice_cache_time) < _VOICE_CACHE_TTL:
        return _valid_voices

    srv = _srv()
    server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}
    tts_host = server_cfg.get("tts_server_host", "localhost")
    tts_port = server_cfg.get("tts_server_port", 8081)
    voices_url = f"http://{tts_host}:{tts_port}/v1/voices"

    try:
        client = srv._http_client if srv._http_client else None
        if client is None:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as c:
                resp = await c.get(voices_url, timeout=10.0)
        else:
            try:
                resp = await client.get(voices_url, timeout=10.0)
            except RuntimeError:
                async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as c:
                    resp = await c.get(voices_url, timeout=10.0)
    except Exception:
        logger.warning("Failed to fetch voices from %s, using cached list", voices_url)
        # Return stale cache if available, otherwise None
        return _valid_voices

    if resp.status_code != 200:
        logger.warning("TTS server returned %d fetching voices, using cached list", resp.status_code)
        return _valid_voices

    try:
        data = resp.json()
    except Exception:
        logger.warning("Invalid JSON from TTS /v1/voices, using cached list")
        return _valid_voices

    # Support both {"voices": [...]} and plain [...] response formats
    voices_raw = data.get("voices", data) if isinstance(data, dict) else data

    voice_names: list[str] = []
    for entry in voices_raw if isinstance(voices_raw, list) else []:
        if isinstance(entry, str):
            voice_names.append(entry)
        elif isinstance(entry, dict):
            name = entry.get("name", "")
            if name:
                voice_names.append(name)

    if not voice_names:
        logger.warning("No voice names found in TTS /v1/voices response")
        return _valid_voices

    _valid_voices = voice_names
    _voice_cache_time = now
    return _valid_voices


# ---------------------------------------------------------------------------
# Internal helpers  (lazy server import)
# ---------------------------------------------------------------------------

def _srv():
    """Lazy import of the server module to avoid circular imports."""
    import proxy.server as _m
    return _m


# ---------------------------------------------------------------------------
# Progress parsing helpers
# ---------------------------------------------------------------------------

def extract_progress_data(line: str | None) -> tuple[int, int, float] | None:
    """Extract ``slot_id``, ``n_tokens`` and ``progress`` from llama-server stdout progress lines.

    Returns a tuple (slot_id: int, n_tokens: int, progress: float) or None if the line does
    not contain valid progress data. When slot ID is not found, defaults to 0.
    """
    if not isinstance(line, str):
        return None
    text = line.strip()
    if not text:
        return None
    if 'n_tokens' not in text or 'progress' not in text:
        return None
    if ',' not in text:
        return None
    try:
        m_slot = re.search(
            r'slot\s+update_slots:.*?id\s+(\d+)|slot\s+(\d+)',
            text, flags=re.IGNORECASE
        )
        m_tokens = re.search(r'\bn_tokens\s*=\s*(\d+)\b', text, flags=re.IGNORECASE)
        m_progress = re.search(r'\bprogress\s*=\s*([0-9]+(?:\.[0-9]+)?)\b', text, flags=re.IGNORECASE)
        if not m_tokens or not m_progress:
            return None
        slot_id = int(m_slot.group(1)) if m_slot and m_slot.group(1) is not None else (
            int(m_slot.group(2)) if m_slot else 0
        )
        n_tokens = int(m_tokens.group(1))
        progress = float(m_progress.group(1))
        return (slot_id, n_tokens, progress)
    except Exception:
        return None


# NOTE: slot_polling_state and _slot_polling_tasks remain defined in server.py
# because tests access them via the server module. Handlers access them
# through the lazy _srv() helper.


async def poll_slots_for_model(
    model: str,
    llama_port: int = 0,
    interval: float = 0.5,
    max_polls: int | None = None,
) -> None:
    """Poll the llama-server `/slots` endpoint for a given model and update
    ``slot_polling_state[model]``.
    """
    polls = 0
    if not model:
        return None
    slots_url = f"http://localhost:{llama_port}/slots?model={model}"
    while True:
        try:
            srv = _srv()
            client = srv._http_client if srv._http_client else httpx.AsyncClient(timeout=httpx.Timeout(5.0))
            if getattr(client, '__aenter__', None):
                async with client as c:
                    resp = await c.get(slots_url, timeout=5.0)
            else:
                resp = await client.get(slots_url, timeout=5.0)

            n_decoded = None
            is_processing = False
            if resp is not None and getattr(resp, 'status_code', None) == 200:
                try:
                    data = resp.json()
                except Exception:
                    data = None
                if isinstance(data, list):
                    if data:
                        # Aggregate across ALL slots, not just data[0]
                        is_processing = any(
                            bool(slot.get('is_processing', False)) for slot in data
                        )
                        n_decoded = None
                        for slot in data:
                            next_token = slot.get('next_token') if isinstance(slot.get('next_token'), dict) else None
                            if next_token is not None and 'n_decoded' in next_token:
                                slot_n = next_token.get('n_decoded')
                            else:
                                slot_n = slot.get('n_decoded')
                            if slot_n is not None:
                                if n_decoded is None or slot_n > n_decoded:
                                    n_decoded = slot_n
                elif isinstance(data, dict):
                    is_processing = bool(data.get('is_processing', False))
                    next_token = data.get('next_token')
                    if isinstance(next_token, dict) and 'n_decoded' in next_token:
                        n_decoded = next_token.get('n_decoded')
                    else:
                        n_decoded = data.get('n_decoded')
                # Access server's slot_polling_state
                srv.slot_polling_state[model] = {'is_processing': is_processing, 'n_decoded': n_decoded}
            else:
                srv.slot_polling_state[model] = {'is_processing': False, 'n_decoded': None}
        except Exception:
            try:
                _srv().slot_polling_state[model] = {'is_processing': False, 'n_decoded': None}
            except Exception:
                pass

        polls += 1
        if max_polls is not None and polls >= max_polls:
            break
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break


def start_slot_polling(model: str, llama_port: int, interval: float = 0.5) -> None:
    """Start an asyncio task to poll the slots endpoint for ``model``."""
    srv = _srv()
    if not model:
        return None
    if model in srv._slot_polling_tasks and not srv._slot_polling_tasks[model].done():
        return None
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(poll_slots_for_model(model, llama_port, interval, None))
        srv._slot_polling_tasks[model] = task
    except RuntimeError:
        def _run():
            try:
                asyncio.run(poll_slots_for_model(model, llama_port, interval, None))
            except Exception:
                pass
        t = threading.Thread(target=_run, daemon=True)
        t.start()


def format_progress(
    n_tokens: int,
    total_tokens: int,
    progress: float,
    model_name: str = "unknown",
    slot_id: int = 0,
    tokens_per_sec: float | None = None,
) -> str:
    """Return a clean, log-friendly progress string without terminal control characters.

    Includes model/slot prefix and tokens-per-second rate when available.
    Output is suitable for logging via the Python logging system (no ANSI
    escape codes, no carriage returns).
    """
    try:
        pct = int(max(0, min(100, int(progress * 100))))
    except Exception:
        pct = 0

    # Build TPS suffix
    if tokens_per_sec is not None:
        tps_str = f" @ {tokens_per_sec:.1f} tok/s"
    else:
        tps_str = " @ --.- tok/s"

    return f"[slot:{slot_id} {model_name}] Processing {n_tokens}/{total_tokens} tokens ({pct}%){tps_str}"


# ---------------------------------------------------------------------------
# /llama/local/status
# ---------------------------------------------------------------------------

@router.get("/llama/local/status")
async def get_llama_local_status(request: Request):
    """Return a small JSON object describing local llama-server status.

    The endpoint is designed to be non-blocking even when the underlying
    llama-server is busy. ``query_llama_status()`` is wrapped with a short
    timeout so the endpoint itself remains responsive under load (target
    response time < 5 s).

    Fields returned::

        {"active_query": bool,
         "local_active_query": bool,
         "model_switch_in_progress": bool,
         "current_model": str | None,
         "llama_server_running": bool,
         "available_slots": int,
         "total_slots": int,
         "slots": [{"slot_id": int, "is_processing": bool, "n_decoded": int | None,
                    "context_tokens": int | None, "owner_session_id": str | None,
                    "lease_state": str, "phase": str | None,
                    "prefill_eta_seconds": float | None}],
         "local_owner_session_id": str | None,
         "local_owner_lease_remaining_seconds": float | None,
         "local_idle_seconds": float | None,
         "local_last_activity_at": str | None,
         "local_last_activity_source": str | None,
         "prefill_eta_seconds": float | None,
         "queue_eta_seconds": float | None,
         "prefill_budget_seconds": float | None,
         "capacity_hint": str,
         "status_schema_version": int}

    ``local_idle_seconds`` reports how long the local backend has been idle
    (LP-0MVGD9Z8Z0055Y5J) — the same monotonic signal the idle-aware routing
    override reads — with ``local_last_activity_at`` (wall clock) and
    ``local_last_activity_source`` (dispatch/chunk/finish/...).

    ``prefill_eta_seconds`` and ``queue_eta_seconds`` publish the calibrated
    prefill-ETA estimate (LP-0MVGDUV5E005UWBQ): the projected prefill cost of
    the reference prompt (the most recent dispatch for the current model, else
    ``server.local_admission_guard_reference_tokens``, assumed cache-cold) and
    the wait a *newcomer* would face (the prefills of the currently occupied
    slots plus its own). Dispatchers can pace themselves from these instead of
    discovering a contended 120 s wait. ``prefill_eta_calibrated`` reports
    whether the rolling calibration has enough samples; with too few samples
    the guard is inert and the wait behaviour is unchanged.

    ``available_slots`` and ``total_slots`` reflect the model-serving slot
    state reported by llama-server's ``/slots`` endpoint. When the server is
    not running or the query fails both default to 0. When the server is
    running but no model is loaded yet, the configured
    ``session_slot_pool_size`` is reported instead (fail-open, so
    orchestrators see available capacity during idle windows —
    LP-0MSI06HPB0043MV1).

    When llama-server is running with a model loaded but the ``/slots``
    query fails (e.g. HTTP 500 while the model reloads after a cheap-mode
    restart), the last known slot counts are served instead of 0/0
    (graceful degradation, LP-0MSVP7XJ6008QPKX), bounded by
    ``SLOT_COUNTS_STALE_AFTER_SECONDS`` (default 1h). ``slots_stale`` is
    ``true`` whenever the served slot data came from that cache rather than
    a fresh query — counts or per-slot detail — so a future silent failure
    (worker alive but zero dispatches across an idle window) is observable.
    The two sources are also published separately:
    ``slot_counts_stale`` and ``slots_detail_stale``. Only the detail matters
    for capacity: a cached ``is_processing``/``phase`` is a fossil of a past
    request, so a hint derived from it is reconciled with live proxy state
    (see ``capacity_hint`` below, LP-0MVKSXR4P001V0CM AC3).

    ``slots`` carries the per-slot details (``slot_id``, ``is_processing``,
    ``n_decoded``) from llama-server's ``/slots`` endpoint so consumers such
    as herdr's downtime worker can track the SAME slots staying free across
    polls (LP-0MSORPUMX002LLIA). When the fresh per-slot query fails or
    returns nothing, the last-known per-slot detail cache is served instead
    so slot identity survives transient failures (LP-0MUFSVXID0039ZAQ); when
    that cache is also empty the field is an empty array. It is an empty
    array when llama-server is not running or no model is loaded yet — never
    a malformed payload. The fetch reuses ``_query_slots_detail()``'s own
    httpx timeout bounded by the ``STATUS_QUERY_TIMEOUT`` window so a slow
    /slots response cannot blow the endpoint's response budget.

    Per-slot capacity (LP-0MVGDUVGE0024YAI, ``status_schema_version: 3``):
    each slot also carries ``owner_session_id`` and ``lease_state``
    (``generating`` | ``idle_leased`` | ``free``) so a dispatcher can tell a
    slot that is actually working from one that is merely reserved-but-idle
    (the case behind 2,633 of 2,641 local dispatch denials while local decoded
    29.3 % of wall clock). ``phase`` (``prefill`` | ``decode`` | ``null``) is
    inferred proxy-side from ``is_processing``/``n_decoded`` (AC5);
    ``context_tokens`` is the slot's prompt progress when the build reports it
    and ``prefill_eta_seconds`` is the shared calibrated prefill cost for it
    (null when not computable). ``lease_state`` is never ``free`` while a
    prefill or decode is in flight.

    The top level publishes the same calibration's ``prefill_budget_seconds``
    (the admission guard's wait budget) and a ``capacity_hint`` summarising
    the pool, evaluated in order: ``idle`` when a slot is free or every
    occupied slot is ``idle_leased``; ``prefill_bound`` when nothing is free
    and at least one slot is prefilling; ``saturated`` when nothing is free
    and every occupied slot is decoding. All new fields are additive — a
    client that ignores them behaves exactly as before (AC1), and the ETA
    fields are ``null`` (never stale numbers) when the slot detail is stale
    (AC7).

    When the served slot detail is stale, ``capacity_hint`` is reconciled
    with live state (LP-0MVKSXR4P001V0CM AC1/AC3) so a fossil snapshot can
    never starve a dispatcher: with no in-flight local request (counted from
    active dispatch records in ``local_in_flight_requests``) every verdict
    degrades to ``idle`` and ``available_slots`` is reconciled upward;
    in-flight work with fresh chunk activity within
    ``STALE_DECODE_EVIDENCE_MAX_IDLE_SECONDS`` keeps ``saturated``; any other
    in-flight work is reported ``prefill_bound``, never ``saturated``.

    ``local_active_query`` reports whether local model work is genuinely in
    flight, derived from active dispatch records rather than the
    ``local_active_queries`` counter, so retained post-stream leases and a
    leaked counter do not make it true (LP-0MVKSXR4P001V0CM AC1). Remote
    provider streams (e.g. opencode-go → deepseek-v4-flash) never make it
    true. Local activity is a subset of all activity, so
    ``local_active_query=true`` implies ``active_query=true``
    (LP-0MSL2ZLLS009RVKR).

    Timeout is configurable via the ``STATUS_QUERY_TIMEOUT`` env var
    (seconds, default 1.0).

    Each call is logged with a ``status_request`` structured message that
    includes the client IP (with source: header vs direct), the client source
    port (direct connections only; ``unknown`` for reverse-proxy header
    paths), a stable ``client_id`` from the session headers when present
    (``x-session-id`` → ``session_id`` → ``x-client-request-id``; omitted
    when none are sent), the response fields and request latency (ms). The
    client IP resolves via X-Forwarded-For / X-Real-IP headers when present
    (reverse-proxy safe), falling back to the direct connection address.
    """
    import os  # noqa: local import for config access

    _start = time.monotonic()

    # -- query_llama_status with timeout (non-blocking guarantee) ---------
    srv = _srv()
    timeout = float(os.environ.get("STATUS_QUERY_TIMEOUT", "1.0"))
    try:
        status = await asyncio.wait_for(srv.query_llama_status(), timeout=timeout)
    except (TimeoutError, Exception):
        status = {"llama_server_running": False, "n_ctx": None, "kv_cache_tokens": None, "router_mode": False}

    llama_running = bool(status.get("llama_server_running", False))

    # -- model_switch_in_progress (thread-safe read of refcount) ----------
    switch_in_progress = False
    try:
        if hasattr(srv, "model_switch_refcount"):
            with srv.model_switch_refcount_lock:  # type: ignore[arg-type]
                refcount = srv.model_switch_refcount
            switch_in_progress = refcount > 0
    except Exception:
        switch_in_progress = False

    # -- secondary indicators ---------------------------------------------
    if not switch_in_progress:
        try:
            switch_in_progress = srv.model_switch_lock.locked() if srv.model_switch_lock is not None else False
        except Exception:
            pass

    if not switch_in_progress:
        try:
            switch_in_progress = bool(srv.background_loads)
        except Exception:
            pass

    cm = srv.current_model if llama_running else None

    # -- active queries (non-blocking snapshot) ---------------------------
    active = False
    try:
        async with srv.active_queries_lock:
            active = srv.active_queries > 0
    except Exception:
        active = False

    # -- local active queries (non-blocking snapshot, LP-0MSL2ZLLS009RVKR) --
    # The global counter above also counts remote provider streams, so herdr
    # dispatch needs a local-only signal. local_active_query=true implies
    # active_query=true (local activity is a subset of all activity).
    #
    # LP-0MVKSXR4P001V0CM AC1: derive the answer from the live dispatch
    # records (a request is in flight only while its record is ``active``),
    # not from ``local_active_queries``, whose retained post-stream leases no
    # longer correspond to work. The counter remains the legacy fallback when
    # the server has no dispatch-record registry at all.
    live_local_requests: int | None = None
    local_active_query = False
    try:
        from proxy.router_helpers import (
            _count_active_local_dispatch_records as _count_active_records,
        )

        live_local_requests = _count_active_records(srv)
    except Exception:
        live_local_requests = None
    if live_local_requests is not None:
        local_active_query = live_local_requests > 0
    else:
        try:
            async with srv.local_active_queries_lock:
                local_active_query = srv.local_active_queries > 0
        except Exception:
            local_active_query = False

    # -- slots query (lightweight, short timeout) -------------------------
    available_slots = 0
    total_slots = 0
    slots: list[dict] = []
    # ``counts_stale`` / ``detail_stale`` track the two independent /slots
    # fallbacks separately: the counts cache (LP-0MSVP7XJ6008QPKX) and the
    # per-slot detail cache (LP-0MUFSVXID0039ZAQ). Only ``detail_stale``
    # matters for capacity: ``is_processing``/``phase`` from a cached
    # snapshot are a fossil of a past request, so a hint derived from them
    # must be reconciled with live proxy state (LP-0MVKSXR4P001V0CM AC3).
    counts_stale = False
    detail_stale = False
    if llama_running:
        try:
            server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}
        except Exception:
            server_cfg = {}
        if cm:
            try:
                llama_port = int(server_cfg.get("llama_server_port", 8080) or 8080)
                # Prefer the discovered local child port: the router serializes
                # /slots?model=... behind the busy child (LP-0MTDGBRPU003Z7KU,
                # measured 5-7s vs 0.17s direct), which starved the shared
                # client pool and made status polls report llama_server_running
                # = false on ~76% of polls.
                from proxy.router_helpers import _discover_local_child_port
                # LP-0MTP1FQXH004JYEF: pass current_model so the lookup
                # filters to the Qwen3 child, not the first spawn line
                # (mxbai-embed) which yields idle n_ctx 256 slots.
                child_port = _discover_local_child_port(srv, model=cm)
                if child_port is not None:
                    llama_port = child_port
                client = srv._http_client if srv._http_client else httpx.AsyncClient(timeout=5.0)
                from proxy.observability import (
                    _last_slot_details_cache,
                    _query_slots,
                    _query_slots_detail,
                    last_known_slot_counts,
                    slots_failure_is_definitive_model_absent,
                )
                # The outcome of the fresh counts query (status + reason) lets
                # the handler tell a definitive "this model is not loaded"
                # (HTTP 400/404 from llama-server's router) from a transient
                # failure, so a fossil cache snapshot is never served as
                # current capacity (LP-0MVKSXR4P001V0CM AC3).
                counts_outcome: dict = {}
                available_slots, total_slots = await _query_slots(
                    client,
                    llama_port,
                    timeout=2.0,
                    model=cm,
                    status_out=counts_outcome,
                )
                model_absent = slots_failure_is_definitive_model_absent(
                    counts_outcome
                )
                # Graceful degradation (LP-0MSVP7XJ6008QPKX): a /slots failure
                # (e.g. HTTP 500 during a model reload after cheap-mode restart)
                # previously surfaced as total_slots=0, fail-closing the herdr
                # downtime worker. Serve the last known counts when fresh, and
                # flag the degraded state so monitoring sees it.
                if total_slots == 0:
                    if model_absent:
                        # Definitive: the idle model instance is not loaded
                        # (llama-server's router unloads idle models). That is
                        # a fresh, truthful "no slots yet", not staleness:
                        # fail open on the configured pool — the first
                        # dispatched job loads the model (LP-0MSI06HPB0043MV1)
                        # — and do not resurrect cached occupancy.
                        try:
                            total_slots = int(
                                server_cfg.get("session_slot_pool_size", 3) or 3
                            )
                        except (TypeError, ValueError):
                            total_slots = 3
                        available_slots = total_slots
                    else:
                        last_known = last_known_slot_counts()
                        if last_known is not None:
                            available_slots, total_slots = last_known
                            counts_stale = True
                # Per-slot details (LP-0MSORPUMX002LLIA): best-effort, empty
                # list on any failure (helper catches HTTP/connect/timeout
                # errors internally; the handler guard keeps the response
                # well-formed even on unexpected exceptions). Bounded by the
                # STATUS_QUERY_TIMEOUT window via the helper's own httpx
                # timeout, so a slow /slots response cannot blow the
                # endpoint's response budget (AC3).
                try:
                    detail = await _query_slots_detail(llama_port, timeout=timeout, model=cm)
                except Exception:
                    detail = None
                if isinstance(detail, list) and detail:
                    slots = detail
                else:
                    # Graceful degradation for per-slot detail
                    # (LP-0MUFSVXID0039ZAQ): the fresh query failed or
                    # returned nothing, so serve the last-known per-slot
                    # detail (updated by _periodic_broadcast_loop() and the
                    # SSE handlers) instead of blanking slot identity. Mirrors
                    # the counts fallback above; when the cache is empty the
                    # payload stays a well-formed empty array (AC3).
                    cached_detail = _last_slot_details_cache
                    if cached_detail and not model_absent:
                        slots = list(cached_detail)
                        detail_stale = True
            except Exception:
                # slots query is best-effort; default to 0 on failure
                pass
        else:
            # Fail-open (LP-0MSI06HPB0043MV1): server is up but no model is
            # loaded yet (e.g. right after restart while preload runs).
            # Querying /slots without ?model= yields HTTP 400 -> 0/0, which
            # wedges orchestrators (Herdr downtime worker) into "no capacity".
            # Report the configured pool size instead; all slots are idle.
            # The first dispatched job triggers ensure_model_loaded().
            try:
                total_slots = int(server_cfg.get("session_slot_pool_size", 3) or 3)
            except (TypeError, ValueError):
                total_slots = 3
            available_slots = total_slots

    # -- per-slot capacity annotation (LP-0MVGDUVGE0024YAI) -----------------
    # Add owner_session_id / lease_state / phase / context_tokens /
    # prefill_eta_seconds to each slot so a dispatcher can distinguish a slot
    # that is *generating* from one that is merely *leased but idle* (the
    # reserved-but-idle false-busy case) and can read the shared calibrated
    # prefill ETA. ``slots_stale`` nulls the per-slot ETA: a cached /slots
    # snapshot must never be presented with a fresh-looking number (AC7).
    try:
        from proxy.observability import (
            annotate_slots_for_status,
        )

        slots = annotate_slots_for_status(
            slots,
            srv,
            cm,
            srv.config if isinstance(getattr(srv, "config", None), dict) else {},
            stale=detail_stale,
        )
    except Exception:
        # Annotation is best-effort; the base slot fields remain valid.
        pass

    # -- local dispatch lease info (LP-0MR9G183O004SJLO) --------------------
    # The owner id is normalised to its bare session-id string: per-endpoint
    # leases are keyed by ``(endpoint, session_id)`` tuples and the raw key
    # must never be serialised into the status payload as an array
    # (WL-0MUKY4MAX0008I0L).
    local_owner_session_id = None
    local_owner_lease_remaining_seconds = None
    try:
        from proxy.router_helpers import _dispatch_key_session_id

        lock = getattr(srv, "local_dispatch_records_lock", None)
        records = getattr(srv, "local_dispatch_records", {})
        if lock is not None:
            async with lock:
                for sid, rec in list(records.items()):
                    if rec.get("active"):
                        session_id = _dispatch_key_session_id(sid)
                        if not isinstance(session_id, str):
                            continue
                        local_owner_session_id = session_id
                        remaining = rec.get("expires_at", 0) - time.monotonic()
                        local_owner_lease_remaining_seconds = max(0.0, remaining)
                        break
    except Exception:
        pass

    # -- prefill / queue ETA estimate (LP-0MVGDUV5E005UWBQ) -----------------
    # Published so a dispatcher can pace BEFORE launching a session instead of
    # discovering a 120 s contended wait. Both fields come from the calibrated
    # prefill cost of the reference prompt (most recent dispatch for the
    # current model, else ``local_admission_guard_reference_tokens``); they are
    # ``None`` only if the estimate itself failed (fail-open payload).
    prefill_eta: dict = {}
    try:
        from proxy.prefill_eta import status_estimate

        prefill_eta = status_estimate(
            srv.config if isinstance(srv.config, dict) else {},
            cm,
            total_slots=total_slots,
            available_slots=available_slots,
            active_queries=int(getattr(srv, "local_active_queries", 0) or 0),
        )
    except Exception:
        prefill_eta = {}

    # AC7 (LP-0MVGDUVGE0024YAI): when the served slot data came from the
    # last-known cache, a poll-derived ETA is not a fresh number. Null it
    # rather than let a dispatcher act on a stale estimate. The budget is a
    # configuration constant (not poll-derived) and stays truthful.
    if detail_stale:
        prefill_eta = dict(prefill_eta)
        prefill_eta["prefill_eta_seconds"] = None
        prefill_eta["queue_eta_seconds"] = None

    # Idle-aware local admission (LP-0MVGD9Z8Z0055Y5J): idle age of the local
    # backend, the signal the routing override reads. Also feeds the stale
    # capacity reconciliation below (recent chunk activity is the only fresh
    # evidence that a cached ``decode`` phase is still true).
    try:
        from proxy.local_activity import activity_snapshot as _activity_snapshot

        _idle_snapshot = _activity_snapshot()
    except Exception:
        _idle_snapshot = {}

    # AC4 (LP-0MVGDUVGE0024YAI): summarise whether the pool is genuinely
    # working — ``idle`` (a free slot, or only idle leases), ``prefill_bound``
    # (no free slot, at least one prefill) or ``saturated`` (no free slot, all
    # decode). Computed from the annotated slots by the pure decision table.
    try:
        from proxy.observability import (
            compute_capacity_hint as _capacity_hint,
        )
        from proxy.observability import (
            downgrade_stale_capacity_hint as _downgrade_stale_hint,
        )

        capacity_hint = _capacity_hint(slots)
        # LP-0MVKSXR4P001V0CM AC3: a hint derived from cached ``/slots``
        # detail must never claim the pool is saturated without fresh
        # evidence. Reconcile it with the live dispatch/activity state so a
        # fossil snapshot cannot starve a dispatcher through an idle window.
        capacity_hint = _downgrade_stale_hint(
            capacity_hint,
            stale=detail_stale,
            live_local_requests=live_local_requests or 0,
            activity_source=_idle_snapshot.get("last_activity_source"),
            idle_seconds=_idle_snapshot.get("idle_seconds"),
        )
        if detail_stale and capacity_hint == "idle":
            # The proxy sees no in-flight local work, so the cached counts
            # must not report a full pool either: reconcile
            # ``available_slots`` with the same view (fail-open, mirroring the
            # no-model-loaded branch LP-0MSI06HPB0043MV1). A dispatcher polls
            # this pair together and acts on both.
            if total_slots <= 0:
                try:
                    total_slots = int(
                        server_cfg.get("session_slot_pool_size", 3) or 3
                    )
                except (TypeError, ValueError):
                    total_slots = 3
            available_slots = max(
                available_slots, total_slots - (live_local_requests or 0)
            )
    except Exception:
        capacity_hint = "idle"

    # -- prefill gate state (LP-0MVGDUUV3001XFRJ AC4) ----------------------
    # Prefills are serialised by default (``local_max_concurrent_prefills``),
    # so a dispatcher needs to see how many prefills are in flight, what the
    # cap is, and how many requests are waiting behind the gate.
    try:
        from proxy.router_helpers import (
            _effective_prefill_cap,
            _get_live_prefill_in_flight_count,
        )

        _server_cfg_for_prefill = (
            srv.config.get("server", srv.config)
            if isinstance(getattr(srv, "config", None), dict)
            else {}
        )
        # LP-0MVKUKFSO008GI6O AC3: publish only in-flight prefills backed by
        # a live dispatch record. An orphaned marker (LP-0MVKSXR4P001V0CM
        # evidence: ``local_prefills_in_flight: 1`` while the GPU was idle)
        # must not make the pool look busy to a dispatcher.
        prefill_in_flight = _get_live_prefill_in_flight_count(srv)
        prefill_cap = _effective_prefill_cap(
            _server_cfg_for_prefill, int(total_slots or 1)
        )
    except Exception:
        prefill_in_flight = 0
        prefill_cap = 0
    try:
        from proxy.contention_queue import queue_depth as _queue_depth

        prefill_gate_waiters = _queue_depth()
    except Exception:
        prefill_gate_waiters = 0

    # -- structured log entry with latency --------------------------------
    _latency_ms = int((time.monotonic() - _start) * 1000)
    # ``slots_stale`` keeps its documented meaning (counts OR detail served
    # from cache) for existing consumers; the per-source flags are published
    # alongside it so an operator can tell a count fallback from a capacity
    # view built on cached slot detail.
    slots_stale = counts_stale or detail_stale
    client_ip, client_ip_source = _resolve_client_ip(request)
    status_extra = {
        "client_ip": client_ip,
        "client_ip_source": client_ip_source,
        "client_port": _resolve_client_port(request),
        "latency_ms": _latency_ms,
        "llama_server_running": llama_running,
        "active_query": active,
        "local_active_query": local_active_query,
        "model_switch_in_progress": switch_in_progress,
        "current_model": cm,
        "available_slots": available_slots,
        "total_slots": total_slots,
        "slots_stale": slots_stale,
        "prefill_eta_seconds": prefill_eta.get("prefill_eta_seconds"),
        "queue_eta_seconds": prefill_eta.get("queue_eta_seconds"),
        "local_owner_session_id": local_owner_session_id,
        "local_owner_lease_remaining_seconds": local_owner_lease_remaining_seconds,
        "local_idle_seconds": _idle_snapshot.get("idle_seconds"),
        "capacity_hint": capacity_hint,
        "local_in_flight_requests": live_local_requests,
        "slots_detail_stale": detail_stale,
        "status_schema_version": 3,
    }
    # Stable client identifier from session headers — omitted (not "unknown")
    # when absent so the log stays additive (LP-0MSKV3IEQ004ZV88).
    client_id = _resolve_client_id(request)
    if client_id:
        status_extra["client_id"] = client_id
    logger.info("status_request", extra=status_extra)

    return {
        "active_query": bool(active),
        "local_active_query": bool(local_active_query),
        "model_switch_in_progress": bool(switch_in_progress),
        "current_model": cm,
        "llama_server_running": bool(llama_running),
        "available_slots": available_slots,
        "total_slots": total_slots,
        # True when the served slot data (counts or per-slot detail) came from
        # the last-known cache because the fresh /slots query failed
        # (LP-0MSVP7XJ6008QPKX, LP-0MUFSVXID0039ZAQ). Additive field;
        # consumers that ignore unknown fields are unaffected.
        "slots_stale": bool(slots_stale),
        # Per-source freshness (LP-0MVKSXR4P001V0CM AC3): ``slots_stale`` is
        # the union below; ``slots_detail_stale`` is the one that matters for
        # ``capacity_hint`` (cached ``is_processing``/``phase`` are not live
        # evidence) and ``slot_counts_stale`` only affects the counts.
        "slots_detail_stale": bool(detail_stale),
        "slot_counts_stale": bool(counts_stale),
        # Truthful in-flight local request count, derived from live dispatch
        # records (``None`` in legacy mode with no record registry).
        # ``local_active_query`` mirrors ``> 0``; ``Null`` is never emitted.
        "local_in_flight_requests": live_local_requests,
        # Per-slot details (LP-0MSORPUMX002LLIA): compact slot dicts from
        # llama-server's /slots endpoint, falling back to the last-known
        # detail cache on a transient failure (LP-0MUFSVXID0039ZAQ); empty
        # when not running / no model / no cached data, so herdr can track
        # same-slot idleness.
        "slots": slots,
        # Prefill / queue ETA (LP-0MVGDUV5E005UWBQ): the projected prefill cost
        # of the reference prompt and the wait a newcomer would face (the
        # prefills of the occupied slots plus its own). Additive fields;
        # ``None`` when the estimate is unavailable.
        "prefill_eta_seconds": prefill_eta.get("prefill_eta_seconds"),
        "queue_eta_seconds": prefill_eta.get("queue_eta_seconds"),
        "prefill_eta_reference_tokens": prefill_eta.get(
            "prefill_eta_reference_tokens"
        ),
        "prefill_eta_samples": prefill_eta.get("prefill_eta_samples", 0),
        "prefill_eta_calibrated": bool(
            prefill_eta.get("prefill_eta_calibrated", False)
        ),
        # Admission-guard budget (LP-0MVGDUVGE0024YAI AC3), sourced from the
        # shared prefill_eta calibration: the wait a prefill must fit inside
        # (``None`` when unresolvable — never a fabricated 0).
        "prefill_budget_seconds": prefill_eta.get("prefill_budget_seconds"),
        # Whether the local pool is genuinely working (LP-0MVGDUVGE0024YAI
        # AC4): ``idle`` | ``prefill_bound`` | ``saturated``.
        "capacity_hint": capacity_hint,
        # Additive payload version (AC1): bumped when per-slot/top-level
        # capacity fields change meaning. Consumers that ignore it behave
        # exactly as before. v3 (LP-0MVKSXR4P001V0CM): ``capacity_hint`` is
        # reconciled with live dispatch state when slot detail is stale, and
        # ``slots_detail_stale`` / ``local_in_flight_requests`` are added.
        "status_schema_version": 3,
        "local_owner_session_id": local_owner_session_id,
        "local_owner_lease_remaining_seconds": local_owner_lease_remaining_seconds,
        # Idle-aware local admission (LP-0MVGD9Z8Z0055Y5J): how long the local
        # backend has been idle (monotonic-derived) and when it last worked.
        # ``idle_seconds`` is the signal the routing override reads; the
        # timestamps are for operators.
        "local_idle_seconds": _idle_snapshot.get("idle_seconds"),
        "local_last_activity_at": _idle_snapshot.get("last_activity_at"),
        "local_last_activity_source": _idle_snapshot.get("last_activity_source"),
        # Prefill serialisation (LP-0MVGDUUV3001XFRJ AC4): in-flight prefills,
        # the effective cap (0 when the gate is disabled) and the requests
        # waiting for it (the contention queue depth).
        "local_prefills_in_flight": prefill_in_flight,
        "local_max_concurrent_prefills": prefill_cap,
        "prefill_gate_waiters": prefill_gate_waiters,
        # Local-admission reservations (LP-0MVIKPCSG002B0NL): outstanding /
        # issued / used / cancelled / expired tickets. ``reserved_issued -
        # reserved_used - reserved_cancelled - reserved_expired`` is the
        # reserved-but-unused signal the work item asks to monitor.
        **(admission_reserve_stats()),
    }


# ---------------------------------------------------------------------------
# /llama/local/reserve — atomic local-admission reservations
# (LP-0MVIKPCSG002B0NL)
# ---------------------------------------------------------------------------

def admission_reserve_stats() -> dict:
    """Reservation counters for the status payload (fail-open)."""
    try:
        from proxy.admission_reserve import stats

        return stats()
    except Exception:
        return {}


@router.post("/llama/local/reserve")
async def reserve_local_slot(request: Request):
    """Claim a local-admission ticket for a specific request, atomically.

    The client sends the prompt size it is about to send and receives either a
    ticket (``200 {"admit": true, "reservation_id", ...}``) that a follow-up
    request honours via ``X-Local-Reservation: <id>``, or an immediate refusal
    (``429 {"admit": false, "reason", "queue_eta_s", "budget_s",
    "retry_after_s"}``) telling it to defer. The verdict shares the occupancy-
    aware estimator the admission guard uses, so the two are identical for
    identical inputs (AC7).

    ``dry_run=true`` returns the verdict without claiming a ticket (this is
    also the ``local-only`` preamble). Tickets are TTL-bound, single-use and
    cancellable via ``DELETE /llama/local/reserve/<id>``.
    """
    from proxy import admission_reserve
    from proxy import provider as provider_mod

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    srv = _srv()
    config = getattr(srv, "config", {})
    if not isinstance(config, dict):
        config = {}
    server_cfg = config.get("server", config) if isinstance(config, dict) else {}

    # The current local model (the same one the admission guard would cost)
    # and the occupancy probe the guard uses. Sharing the probe is what keeps
    # ``/reserve`` and the in-flight guard identical (AC7).
    cm = getattr(srv, "current_model", None) or server_cfg.get("llama_model")
    current_concurrency = (0, 1)
    busy_slots = 0
    in_flight_prefills = 0
    try:
        current_concurrency = provider_mod._local_concurrency_info(
            config, endpoint=None
        )
        busy_slots, in_flight_prefills = provider_mod._local_gate_occupancy(
            config, current_concurrency[1] or 1, endpoint=None
        )
        occupant_prefill_seconds = provider_mod._local_occupant_prefill_seconds(
            endpoint=None, session_id=body.get("session_id")
        )
    except Exception:
        busy_slots, in_flight_prefills = 0, 0
        occupant_prefill_seconds = 0.0

    status, payload = admission_reserve.reserve(
        body,
        config,
        model=cm,
        busy_slots=busy_slots,
        in_flight_prefills=in_flight_prefills,
        occupant_prefill_seconds=occupant_prefill_seconds,
    )
    headers: dict[str, str] = {
        "Cache-Control": "no-store",
    }
    if status == 429:
        headers.update(
            admission_reserve.headers_for_route(
                "refused", payload.get("reason")
            )
        )
        headers["Retry-After"] = str(payload.get("retry_after_s", 1))
        headers["X-Local-Admission"] = "reject"
        headers["X-Local-Admission-Eta"] = str(payload.get("queue_eta_s", 0))
        headers["X-Local-Admission-Budget"] = str(payload.get("budget_s", 0))
    return JSONResponse(status_code=status, content=payload, headers=headers)


@router.delete("/llama/local/reserve/{reservation_id}")
async def cancel_reservation(reservation_id: str):
    """Cancel an outstanding reservation, releasing its gate claim."""
    from proxy import admission_reserve

    status, payload = admission_reserve.cancel(reservation_id)
    return JSONResponse(status_code=status, content=payload)


# ---------------------------------------------------------------------------
# TTS health probe helper
# ---------------------------------------------------------------------------

async def _probe_tts_health(tts_host: str, tts_port: int, timeout: float = 2.0) -> bool:
    """Probe the TTS server's /health endpoint.

    Performs a simple GET to ``http://<host>:<port>/health``.
    Returns True if the endpoint responds with HTTP 200 and
    ``{"status": "ok"}``.
    """
    if not tts_host or not tts_port or tts_port <= 0:
        return False
    try:
        url = f"http://{tts_host}:{tts_port}/health"
        async with httpx.AsyncClient(timeout=httpx.Timeout(max(timeout, 1.0))) as client:
            response = await client.get(url, timeout=timeout)
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, dict) and data.get("status") == "ok":
                        return True
                except Exception:
                    pass
            return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@router.get("/health")
async def health_check():
    """Health check endpoint with readiness gating.

    Includes TTS server status fields:
    - ``tts_enabled``: Whether TTS is enabled in config.
    - ``tts_server_running``: Whether the TTS process is alive (poll() is None).
    - ``tts_server_healthy``: Whether the TTS server responds to its /health endpoint.
    """
    srv = _srv()
    server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}
    router_mode = bool(server_cfg.get("llama_router_mode", False))
    loaded_models = None
    if router_mode:
        router_models = await srv.router_list_models()
        loaded_models = srv._extract_router_model_ids(router_models)

    llama_running = srv.llama_process is not None and srv.llama_process.poll() is None
    llama_port = int(server_cfg.get("llama_server_port", 8080) or 8080)
    backend_reachable = bool(llama_running and await srv._probe_backend_reachable(llama_port))
    self_healing = srv._is_self_healing_active()
    gpu_wedge_detected = bool(getattr(srv, "gpu_wedge_detected", False))
    gpu_wedge_detection_disabled = bool(
        getattr(srv, "gpu_wedge_detection_disabled", False)
    )
    ready = bool(
        llama_running
        and srv.backend_ready
        and backend_reachable
        and not self_healing
        and not gpu_wedge_detected
    )

    # -- TTS health fields -------------------------------------------------
    tts_enabled = bool(server_cfg.get("tts_enabled", True))
    tts_host = str(server_cfg.get("tts_server_host", "localhost"))
    tts_port = int(server_cfg.get("tts_server_port", 8081) or 8081)

    tts_process_alive = (
        srv.tts_process is not None
        and callable(getattr(srv.tts_process, "poll", None))
        and srv.tts_process.poll() is None
    )

    tts_server_healthy = False
    if tts_enabled and tts_process_alive:
        tts_server_healthy = await _probe_tts_health(tts_host, tts_port, timeout=2.0)

    return {
        "status": "healthy" if ready else "degraded",
        "ready": ready,
        "current_model": srv.current_model,
        "loaded_models": loaded_models,
        "llama_server_running": llama_running,
        "backend_reachable": backend_reachable,
        "self_healing_in_progress": self_healing,
        "gpu_wedge_detected": gpu_wedge_detected,
        "gpu_wedge_detection_disabled": gpu_wedge_detection_disabled,
        "gpu_wedge_signal_count": int(
            srv.backend_signal_counts.get("gpu_wedge", 0) or 0
        ),
        "backend_recovery": srv._backend_recovery_snapshot(),
        "backend_signals": dict(srv.backend_signal_counts),
        "tts_enabled": tts_enabled,
        "tts_server_running": tts_process_alive,
        "tts_server_healthy": tts_server_healthy,
        # LP-0MVECX47Z004LJAQ: llama-server.log freshness / relay health so
        # log staleness is observable without shell access (AC3).
        "llama_log_age_seconds": _llama_log_age_seconds_health(srv),
        "llama_log_source_healthy": _llama_log_source_healthy(srv),
    }


# ---------------------------------------------------------------------------
# llama-server.log freshness helpers (LP-0MVECX47Z004LJAQ, AC3)
# ---------------------------------------------------------------------------


def _llama_log_age_seconds_health(srv) -> float | None:
    """Age of llama-server.log in seconds, or None when absent/unreadable."""
    try:
        from proxy.lifecycle import _llama_log_age_seconds

        return _llama_log_age_seconds(srv)
    except Exception:
        return None


def _llama_log_source_healthy(srv) -> bool:
    """True when llama-server.log is expected to keep receiving lines.

    Combines the adoption-time relay verdict (``llama_log_source_unhealthy``)
    with a live-writer check: a fresh log file with no live writer would
    already be frozen, so it is reported unhealthy rather than silently stale.
    """
    try:
        if getattr(srv, "llama_log_source_unhealthy", False):
            return False
        from proxy.lifecycle import (
            _llama_log_age_seconds,
            _log_has_live_writer,
            _resolve_llama_log_path,
        )

        age = _llama_log_age_seconds(srv)
        if age is None:
            # No log yet (fresh startup before first backend write) — nothing
            # to flag.
            return True
        log_path = _resolve_llama_log_path(srv)
        if age <= 60.0:
            # Recently written — relay clearly alive.
            return True
        # Stale: healthy only if a live writer (other than this proxy's own
        # log handle) still holds the file.
        if not log_path.exists():
            return False
        return _log_has_live_writer(log_path, exclude_pid=os.getpid())
    except Exception:
        return True


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

@router.get("/v1/models")
async def list_models():
    """List available models from proxy configuration."""
    srv = _srv()
    models_list = []
    for name, cfg in srv.config.get("models", {}).items():
        mtype = get_model_type(cfg)
        models_list.append({
            "id": name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local" if mtype == "local" else ("remote" if mtype == "remote" else "unknown"),
            "type": mtype or "unknown",
            "aliases": cfg.get("aliases", []),
        })
    return {"object": "list", "data": models_list}


# ---------------------------------------------------------------------------
# /metrics  (Prometheus)
# ---------------------------------------------------------------------------

@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus scrape endpoint (text/plain, exposition format)."""
    srv = _srv()
    try:
        payload, content_type = srv.metrics.generate_metrics_payload()
        return Response(content=payload, media_type=content_type)
    except Exception:
        raise HTTPException(status_code=503, detail="Prometheus metrics unavailable")


# ---------------------------------------------------------------------------
# /admin/metrics
# ---------------------------------------------------------------------------

def _local_eviction_metrics() -> dict:
    """Post-acceptance eviction counters (LP-0MVN4LCED00661MX AC5).

    ``total`` plus a per-reason breakdown. Monotonic; reset only on process
    restart (matching the in-process Prometheus counter). Never raises: an
    observability endpoint must not fail because a counter is unavailable.
    """
    try:
        from proxy.admission_events import eviction_after_acceptance_counters

        return eviction_after_acceptance_counters()
    except Exception:
        return {"total": 0, "by_reason": {}}


@router.get("/admin/metrics")
async def admin_metrics():
    """Return router/memory/metrics for observability."""
    srv = _srv()
    server_config = srv.config.get("server", {})
    models_max = server_config.get("llama_models_max")
    router_mode = server_config.get("llama_router_mode", False)
    loaded_models = None
    if router_mode:
        router_models = await srv.router_list_models()
        loaded_models = srv._extract_router_model_ids(router_models)

    per_model = {}
    for m in loaded_models or []:
        per_model[m] = {"last_used": srv.model_last_used.get(m), "rss_bytes": None}

    process_rss = None
    try:
        if 'psutil' in dir(srv) and srv.psutil and srv.llama_process is not None:
            pid = getattr(srv.llama_process, 'pid', None)
            if pid:
                p = srv.psutil.Process(pid)
                mem = p.memory_info()
                process_rss = getattr(mem, 'rss', None)
    except Exception:
        process_rss = None

    if process_rss is not None and loaded_models:
        try:
            per = int(process_rss // len(loaded_models))
            for m in loaded_models:
                per_model[m]['rss_bytes'] = per
        except Exception:
            for m in loaded_models:
                per_model[m]['rss_bytes'] = None

    try:
        srv.metrics.update_metrics(process_rss, loaded_models)
    except Exception:
        pass

    # Session-list index observability (LP-0MSNM9IAC000GVXT): index size,
    # index-hit vs cold-scan counts, last scan duration.
    index_obs = {}
    try:
        recorder = getattr(srv, "session_recorder", None)
        if recorder is None:
            from proxy.session_recorder import SessionRecorder
            recorder = SessionRecorder.from_config(srv.config)
            srv.session_recorder = recorder
        index_obs = recorder.get_index_observability()
    except Exception:
        index_obs = {}

    return {
        "models_max": models_max,
        "loaded_models": loaded_models,
        "per_model": per_model,
        "process_rss_bytes": process_rss,
        "index_observability": index_obs,
        "session_metrics": srv.session_manager.get_metrics(),
        "restore_success_total": int(srv.session_restore_observability.get("restore_success_total", 0)),
        "restore_fallback_total": dict(srv.session_restore_observability.get("restore_fallback_total", {})),
        "delta_payload_bytes_total": int(srv.session_restore_observability.get("delta_payload_bytes_total", 0)),
        "single_flight_metrics": dict(srv.session_single_flight_observability),
        # LP-0MVN4LCED00661MX AC5: post-acceptance local evictions (an accepted
        # dispatch released before its first byte) with a per-reason
        # breakdown, beside the guardrail block. Alert on any non-zero count
        # for a within-budget prompt — that is a pricing/budget defect.
        "local_evictions_after_acceptance": _local_eviction_metrics(),
        "guardrail_metrics": {
            "guardrail_cutoff_total": int(srv.session_guardrail_observability.get("guardrail_cutoff_total", 0)),
            "guardrail_cutoff_reasons": dict(srv.session_guardrail_observability.get("guardrail_cutoff_reasons", {})),
            "session_invalidation_total": int(srv.session_guardrail_observability.get("session_invalidation_total", 0)),
            "session_invalidation_reasons": dict(srv.session_guardrail_observability.get("session_invalidation_reasons", {})),
        },
        "backend_ready": bool(srv.backend_ready),
        "backend_recovery": srv._backend_recovery_snapshot(),
        "backend_signals": dict(srv.backend_signal_counts),
    }


# ---------------------------------------------------------------------------
# /admin/dump-counts
# ---------------------------------------------------------------------------

@router.get("/admin/dump-counts")
async def admin_dump_counts():
    """Return in-memory request and token counts for debugging."""
    srv = _srv()
    snap_c = {}
    snap_t = {}
    async with srv.counts_lock:
        snap_c = dict(srv.request_counts)
    async with srv.token_lock:
        snap_t = dict(srv.token_counts)
    return {"counts": snap_c, "tokens": snap_t}


# ---------------------------------------------------------------------------
# /admin/stop-server
# ---------------------------------------------------------------------------

@router.post("/admin/stop-server")
async def admin_stop_server():
    """Stop the llama-server."""
    srv = _srv()
    srv.stop_llama_server()
    return {"status": "success", "message": "llama-server stopped"}


# ---------------------------------------------------------------------------
# /admin/reset-counts
# ---------------------------------------------------------------------------

@router.post("/admin/reset-counts")
async def admin_reset_counts():
    """Reset in-memory and persisted request/token counts to empty."""
    srv = _srv()
    async with srv.counts_lock:
        srv.request_counts = {}
        srv.counts_dirty = True
    async with srv.token_lock:
        srv.token_counts = {}
        srv.tokens_dirty = True
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(srv.save_counts())
        loop.create_task(srv.save_token_counts())
    except RuntimeError:
        await srv.save_counts()
        await srv.save_token_counts()
    for q in list(srv.log_tail_clients):
        try:
            q.put_nowait({"counts": {}, "tokens": {}})
        except Exception:
            continue
    return {"status": "success", "message": "Counts reset"}


# ---------------------------------------------------------------------------
# /admin/sessions — session DELETE only; GET is served by ui.py list_all_sessions
# ---------------------------------------------------------------------------

@router.delete("/admin/sessions/{session_id}")
async def admin_delete_session(session_id: str):
    """Delete a specific session by ID."""
    srv = _srv()
    removed = await srv.session_manager.remove(session_id)
    if removed:
        return {"status": "success", "message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


# ---------------------------------------------------------------------------
# /admin/reload-config
# ---------------------------------------------------------------------------

@router.post("/admin/reload-config")
async def reload_config():
    """Reload the configuration from disk."""
    srv = _srv()
    try:
        srv.config = srv.load_config()
        logger.info("Configuration reloaded successfully via admin endpoint")
        return {"status": "success", "message": "Configuration reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# /admin/clear-usage-limit  —  runtime quarantine escape hatch
# ---------------------------------------------------------------------------

@router.post("/admin/clear-usage-limit")
async def admin_clear_usage_limit(request: Request):
    """Clear usage-limit quarantine entries at runtime (LP-0MUQTCMW2001VXH8).

    Body (optional JSON): ``{"account": "<account-key>"}``. When the body is
    empty or ``account`` is omitted, every pending quarantine entry is
    cleared. Returns the cleared account keys so an operator can reintroduce
    a recovered account without restarting the proxy.
    """
    account = None
    try:
        body = await request.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        account = body.get("account")
    if account is not None and not isinstance(account, str):
        raise HTTPException(
            status_code=400,
            detail="'account' must be a string when provided",
        )
    cleared = clear_usage_limit_quarantine(account or None)
    logger.info(
        "admin: cleared %d usage-limit quarantine entr%s (account=%s)",
        len(cleared),
        "y" if len(cleared) == 1 else "ies",
        account or "<all>",
    )
    return {"status": "success", "cleared": cleared, "count": len(cleared)}


# ---------------------------------------------------------------------------
# /v1/leases/release  —  Proactive lease release (LP-0MRFOF7XO003T7CT)
# ---------------------------------------------------------------------------

@router.post("/v1/leases/release")
async def release_lease(request: Request):
    """Explicitly release the dispatch lease for the caller's session.

    Accepts a JSON body with a ``session_id`` field and removes the
    corresponding dispatch record from ``local_dispatch_records``,
    allowing other sessions to acquire the local backend immediately.

    The endpoint is idempotent: calling it with a session_id that has
    no matching lease returns ``200 OK`` with ``{"status": "ok"}``.

    Returns ``400 Bad Request`` when ``session_id`` is missing, empty,
    or ``null``.
    """
    srv = _srv()

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON body",
        )

    session_id = body.get("session_id")
    if not session_id or not isinstance(session_id, str) or not session_id.strip():
        raise HTTPException(
            status_code=400,
            detail="session_id is required",
        )

    session_id = session_id.strip()

    try:
        from proxy.router_helpers import _release_local_dispatch
        await _release_local_dispatch(srv, session_id, request=request)
    except Exception as e:
        logger.exception(
            "Failed to release dispatch lease for session %s",
            session_id if session_id else "unknown",
        )
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# /v1/voices  —  List available TTS voices
# ---------------------------------------------------------------------------

@router.get("/v1/voices")
async def list_voices():
    """List available TTS voices from the tts-server.

    Forwards a GET request to the tts-server ``/v1/voices`` endpoint
    and returns the JSON response with available speaker voices.

    Returns ``200 OK`` with the voices list on success.
    Returns ``502 Bad Gateway`` when the tts-server is unreachable.
    """
    srv = _srv()
    server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}

    tts_host = server_cfg.get("tts_server_host", "localhost")
    tts_port = server_cfg.get("tts_server_port", 8081)
    tts_url = f"http://{tts_host}:{tts_port}/v1/voices"

    try:
        client = srv._http_client if srv._http_client else None
        if client is None:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as c:
                resp = await c.get(tts_url, timeout=10.0)
        else:
            try:
                resp = await client.get(tts_url, timeout=10.0)
            except RuntimeError:
                async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as c:
                    resp = await c.get(tts_url, timeout=10.0)
    except httpx.ConnectError as e:
        logger.error("TTS server unreachable at %s: %s", tts_url, e)
        raise HTTPException(status_code=502, detail=f"TTS server unreachable: {e}")
    except httpx.TimeoutException as e:
        logger.error("TTS server timeout at %s: %s", tts_url, e)
        raise HTTPException(status_code=502, detail=f"TTS server timeout: {e}")
    except Exception as e:
        logger.exception("Unexpected error forwarding TTS request to %s", tts_url)
        raise HTTPException(status_code=502, detail=f"TTS server error: {e}")

    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ---------------------------------------------------------------------------
# /v1/audio/speech  —  TTS (Speech Synthesis)
# ---------------------------------------------------------------------------

@router.post("/v1/audio/speech")
async def create_speech(request: Request):
    """Generate speech from text using the local TTS server backend.

    OpenAI-compatible endpoint that forwards requests to the qwentts.cpp
    ``tts-server`` (which should be running alongside the proxy).

    Request body (JSON)::

        {
            "model": "qwen3-tts",
            "input": "Text to convert to speech",
            "voice": "default",
            "response_format": "wav",
            "lang": "english"
        }

    Returns audio content with ``Content-Type: audio/wav`` on success.
    Returns ``400 Bad Request`` for invalid/missing parameters.
    Returns ``502 Bad Gateway`` when the tts-server is unreachable.
    """
    srv = _srv()
    server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}

    # Parse request body
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Validate required parameters
    model = body.get("model")
    input_text = body.get("input")

    if not model or not isinstance(model, str):
        raise HTTPException(status_code=400, detail="model is required and must be a string")

    if not isinstance(input_text, str) or not input_text.strip():
        raise HTTPException(status_code=400, detail="input is required and must be a non-empty string")

    # Optional parameters
    voice = body.get("voice", "")
    response_format = body.get("response_format", "wav")
    instructions = body.get("instructions", "")
    lang = body.get("lang", "")

    # -- Voice validation: ensure the voice name is recognised -----------
    voice_substituted = False
    original_voice = voice
    if voice:
        valid_voices = await _ensure_voices_cached()
        if valid_voices is not None and voice not in valid_voices:
            logger.warning(
                "Voice '%s' not recognized, using default '%s'",
                voice, _DEFAULT_VOICE,
            )
            voice = _DEFAULT_VOICE
            voice_substituted = True

    # Input length guard
    if len(input_text) > 10000:
        raise HTTPException(status_code=400, detail="input exceeds maximum length of 10000 characters")

    # Build tts-server URL from config
    tts_host = server_cfg.get("tts_server_host", "localhost")
    tts_port = server_cfg.get("tts_server_port", 8081)
    tts_url = f"http://{tts_host}:{tts_port}/v1/audio/speech"

    # Forward request to tts-server (omit empty optional fields)
    forward_body = {
        "model": model,
        "input": input_text,
        "response_format": response_format,
    }
    if voice:
        forward_body["voice"] = voice
    if instructions:
        forward_body["instructions"] = instructions
    if lang:
        forward_body["lang"] = lang

    try:
        client = srv._http_client if srv._http_client else None
        if client is None:
            # No shared client — create a temporary one
            async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as c:
                resp = await c.post(tts_url, json=forward_body, timeout=180.0)
        else:
            try:
                # Use shared client directly without entering/closing it
                resp = await client.post(tts_url, json=forward_body, timeout=180.0)
            except RuntimeError:
                # Shared client was closed by another handler — fall back to temp
                async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as c:
                    resp = await c.post(tts_url, json=forward_body, timeout=180.0)
    except httpx.ConnectError as e:
        logger.error("TTS server unreachable at %s: %s", tts_url, e)
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "tts_error",
                    "code": "tts_server_unreachable",
                    "message": f"TTS server unreachable: {e}",
                },
                "status": 502,
                "path": "/v1/audio/speech",
            },
        )
    except httpx.TimeoutException as e:
        logger.error("TTS server timeout at %s: %s", tts_url, e)
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "tts_error",
                    "code": "tts_server_timeout",
                    "message": f"TTS server timeout: {e}",
                },
                "status": 502,
                "path": "/v1/audio/speech",
            },
        )
    except Exception as e:
        logger.exception("Unexpected error forwarding TTS request to %s", tts_url)
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "tts_error",
                    "code": "tts_server_error",
                    "message": f"TTS server error: {e}",
                },
                "status": 502,
                "path": "/v1/audio/speech",
            },
        )

    # Check for backend HTTP errors (>=400) and return structured 502
    if resp.status_code >= 400:
        try:
            root_body = json.loads(resp.content.decode("utf-8", errors="replace"))
        except Exception:
            root_body = resp.content.decode("utf-8", errors="replace")
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "tts_error",
                    "code": "tts_server_error",
                    "message": f"TTS server returned HTTP {resp.status_code}",
                },
                "status": 502,
                "path": "/v1/audio/speech",
                "root_cause": {
                    "backend_status_code": resp.status_code,
                    "backend_body": root_body,
                },
            },
        )

    # Return the audio response from tts-server
    content_type = resp.headers.get("content-type", "audio/wav")
    headers = {}
    if voice_substituted:
        headers["Warning"] = (
            f'299 proxy "voice=\'{original_voice}\' not recognized, '
            f"using default '{_DEFAULT_VOICE}'"
        )
    return Response(
        content=resp.content,
        media_type=content_type,
        status_code=resp.status_code,
        headers=headers or None,
    )
