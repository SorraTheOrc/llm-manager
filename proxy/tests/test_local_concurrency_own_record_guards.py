"""Guard tests: the own-record exclusion must not change anything else.

Parent LP-0MWMSQR3O002YNPW (AC2, AC3, AC4); guard-test item LP-0MWNHBC8E009QW8U.

The finished-turn window fix is read-side only: the routing concurrency gate
stops counting the requesting session's own record while that record's turn has
already emitted its terminal chunk, without granting any capacity. This module
pins everything that must **not** move:

- **AC2 — genuine concurrency still counts.** Two overlapping dispatches of one
  session keep the pool saturated at the routing gate, a second dispatch is
  still denied by the lease gate when the generating pool is full, and a
  partially finished session (two credits, one dispatch still live) still
  counts, so a third concurrent stream cannot be admitted.
- **AC3 — anonymous readers exclude nothing.** ``session_id=None`` (and the
  argument omitted) counts every record exactly as today, and the
  pool-description readers behind ``/llama/local/status`` and
  ``/llama/local/reserve`` never filter the requesting session's own record.
- **AC4 — the routing gate stays a read.** ``_try_acquire_local_dispatch``
  remains the capacity authority: a refused-then-admitted dispatch never lifts
  the number of live local streams above ``session_slot_pool_size``.
- **Risk mitigation — a client disconnect is not "finished".** A stream that
  ends by client disconnect keeps its credit (and its live sibling's), so the
  record keeps counting and a third stream is not admitted.

Every assertion here holds on the pre-change tree *and* after
LP-0MWNHBFN6000GH2J (terminal-chunk marking) and LP-0MWNHBG7D002NH5S (the
own-record predicate), so none of them is ``xfail``-marked: a regression is
immediately red. The red-first twin of this module is
``test_local_concurrency_finished_own_record.py``.

Harness idioms are reused from ``test_local_concurrency_own_lease.py`` (the
``proxy_with_fallback`` routing harness), the dispatch credit-accounting tests
(the real lease/credit/cleanup helpers) and ``test_llama_local_status.py`` (the
status ASGI harness).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import proxy.provider as provider
import proxy.router as router
import proxy.router_helpers as helpers
import proxy.server as server
import pytest
from starlette.responses import Response

ENDPOINT = "http://localhost:8080"
POOL = 2
ME = "session-me"
OTHER = "session-other"
THIRD = "session-third"

#: Per-dispatch credit tokens (LP-0MVYZEYZY003IQAH) held by the records below.
ME_TOKEN = 8101
ME_TOKEN_SIBLING = 8102
OTHER_TOKEN = 8111
THIRD_TOKEN = 8121


# ---------------------------------------------------------------------------
# Records and routing harness
# ---------------------------------------------------------------------------


def _config(**server_overrides) -> dict:
    server_cfg = {
        "llama_server_port": 8080,
        "session_slot_pool_size": POOL,
        # Two different sessions must be able to hold a prefill hold at once,
        # otherwise every two-session fixture below would be denied by the
        # serialised-prefill cap rather than by the generating pool.
        "local_max_concurrent_prefills": POOL,
        "local_dispatch_lease_timeout_seconds": 30,
        "max_concurrent_queries": 8,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": 30,
        "contention_queue_max_depth": 4,
        "local_prefill_deadline_seconds": 0,
        "local_cold_prefill_budget_tokens": 25000,
    }
    server_cfg.update(server_overrides)
    return {"server": server_cfg, "provider_cooldown_seconds": 60}


def _dispatch_record(
    session: str,
    *,
    credits: list[int],
    streaming: list[int] | None = None,
    active: bool = True,
    expires_in: float = 25.0,
) -> dict:
    """A per-endpoint dispatch record, in the production post-acquire shape.

    ``dispatch_credits`` holds one credit per in-flight dispatch of this
    session (LP-0MVWF78QK0060TZH) and ``streaming_dispatch_tokens`` the subset
    whose stream has not yet emitted its terminal chunk (the key the fix adds).
    ``streaming=None`` models a legacy record that predates both keys, which
    must keep today's ``active``-only behaviour.
    """
    now = time.monotonic()
    record = {
        "backend": "local",
        "started_at": now,
        "active": active,
        "expires_at": now + expires_in,
        "model_name": "Qwen3",
        "endpoint": ENDPOINT,
        "session": session,
        "dispatch_token": credits[0] if credits else None,
        "dispatch_credits": list(credits),
    }
    if streaming is not None:
        record["streaming_dispatch_tokens"] = list(streaming)
    return record


@pytest.fixture
def install_records(monkeypatch):
    """Install dispatch records and the local counters on ``proxy.server``."""

    def _install(records: dict, *, generating: int = 0) -> dict:
        active = sum(
            1
            for rec in records.values()
            if isinstance(rec, dict) and rec.get("active")
        )
        monkeypatch.setattr(server, "local_dispatch_records", records, raising=False)
        monkeypatch.setattr(
            server, "local_dispatch_records_lock", asyncio.Lock(), raising=False
        )
        monkeypatch.setattr(
            server, "local_generating_queries", generating, raising=False
        )
        monkeypatch.setattr(server, "local_active_queries", active, raising=False)
        monkeypatch.setattr(server, "active_queries", active, raising=False)
        monkeypatch.setattr(server, "local_generating_sessions", set(), raising=False)
        return records

    return _install


class _Request:
    """Minimal request stub carrying the session header the gate reads."""

    def __init__(self, session_id: str | None = ME):
        headers = {"host": "localhost"}
        if session_id:
            headers["x-session-id"] = session_id
        self.headers = headers
        self._body = json.dumps(
            {"model": "code", "messages": [{"role": "user", "content": "hi"}]}
        ).encode("utf-8")
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


def _model_config() -> dict:
    return {
        "providers": [
            {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"},
            {
                "name": "opencode-go-2",
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
            },
        ]
    }


def _ok() -> Response:
    return Response(
        content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
        status_code=200,
        media_type="application/json",
    )


async def _route(calls: list[str], *, session_id: str | None = ME) -> Response:
    """Drive ``proxy_with_fallback`` with only occupancy able to divert it."""

    async def _local(_req, _path):
        calls.append("local")
        return _ok()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok()

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider.request_priority", return_value="low"),
        patch(
            "proxy.provider._estimate_effective_prompt_tokens_for_routing",
            new=AsyncMock(return_value=2_000),
        ),
        patch("proxy.provider._get_tokenizer_for_model", lambda *_: (None, 1.0)),
        # Warm cache + small prompt: nothing but occupancy may divert the request.
        patch("proxy.provider._get_cached_ratio", lambda *_: 0.95),
        patch("proxy.provider._should_skip_local", lambda *a, **k: (False, None)),
    ):
        return await provider.proxy_with_fallback(
            _Request(session_id), "v1/chat/completions", _model_config(), _config()
        )


# ---------------------------------------------------------------------------
# Real lease / credit harness
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_prefill_thrash_cooldowns():
    """Start every row with no armed post-deadline cooldown."""
    helpers._reset_prefill_thrash_cooldowns()
    yield
    helpers._reset_prefill_thrash_cooldowns()


def _lease_server(**server_overrides) -> SimpleNamespace:
    """Minimal but real server state for the counter/lease helpers."""
    return SimpleNamespace(
        config=_config(**server_overrides),
        logger=MagicMock(),
        active_queries=0,
        active_queries_lock=asyncio.Lock(),
        per_model_queries={},
        per_model_queries_lock=asyncio.Lock(),
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
        local_generating_queries=0,
        local_generating_queries_lock=asyncio.Lock(),
        local_generating_sessions=set(),
        local_prefill_in_flight={},
        local_prefill_in_flight_lock=asyncio.Lock(),
    )


async def _acquire(
    srv, session: str, *, max_local: int = POOL
) -> tuple[bool, str | None, SimpleNamespace]:
    """Take (or be refused) one dispatch lease through the real choke point."""
    request = SimpleNamespace()
    acquired, owner, _count, _retry = await helpers._try_acquire_local_dispatch(
        srv,
        max_local,
        session,
        ENDPOINT,
        model_name="Qwen3",
        request=request,
    )
    return acquired, owner, request


async def _start_generating(srv, session: str) -> None:
    """Take the generating-only slot the streaming router takes on first byte."""
    await helpers._increment_generating_only_slot(srv, session_key=session)


async def _release(
    srv,
    session: str,
    *,
    request: SimpleNamespace,
    disconnected: bool = False,
) -> None:
    """Release one dispatch through the router's real teardown pair."""
    await helpers._decrement_generating_only_slot(srv, session_key=session)
    await router._cleanup_after_request(
        srv,
        session,
        disconnected=disconnected,
        decrement_local=True,
        session_explicit=True,
        model_name="Qwen3",
        request=request,
        backend=ENDPOINT,
    )


def _install_registry(monkeypatch, srv) -> None:
    """Point the routing read at the lease harness's live registry."""
    monkeypatch.setattr(
        server, "local_dispatch_records", srv.local_dispatch_records, raising=False
    )
    monkeypatch.setattr(
        server,
        "local_generating_queries",
        srv.local_generating_queries,
        raising=False,
    )


# ======================================================================
# AC2 — same-session concurrency is still gated
# ======================================================================


@pytest.mark.asyncio
async def test_two_genuinely_overlapping_same_session_streams_are_gated(
    monkeypatch, install_records, caplog
):
    """AC2: both records still stream upstream ⇒ the routing gate still refuses.

    The requester's own record is not the finished-turn shape: it holds its
    streaming token, so it is genuine in-flight work. Excluding it would hide a
    second concurrent stream of one session, so the pool must read 2/2 and the
    request must fall through with ``local_concurrency_limit`` exactly as today.
    """
    caplog.set_level(logging.INFO, logger="llama-proxy")
    install_records(
        {
            (ENDPOINT, ME): _dispatch_record(
                ME, credits=[ME_TOKEN], streaming=[ME_TOKEN]
            ),
            (ENDPOINT, OTHER): _dispatch_record(
                OTHER, credits=[OTHER_TOKEN], streaming=[OTHER_TOKEN]
            ),
        },
        generating=2,
    )

    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    ) == (POOL, POOL), "a live own dispatch must keep counting toward the pool"

    calls: list[str] = []
    monkeypatch.setattr(
        provider, "_get_local_provider_endpoint", lambda *a, **k: ENDPOINT
    )
    result = await _route(calls)

    assert result.status_code == 200
    assert calls == ["remote"], (
        "two overlapping streams of one session must both be gated at the "
        f"routing gate, got {calls}"
    )
    assert "reason=local_concurrency_limit" in caplog.text


@pytest.mark.asyncio
async def test_same_session_second_dispatch_is_denied_when_the_pool_generates():
    """AC2: the dispatch denial is unchanged — the lease gate stays strict.

    Same-session re-acquisition is permitted by ``own_has_lease``, but not when
    the generating-only pool is already at ``session_slot_pool_size``: the
    second dispatch must be denied without taking a credit.
    """
    srv = _lease_server()
    acquired_me, _owner, _req_me = await _acquire(srv, ME)
    acquired_other, _owner, _req_other = await _acquire(srv, OTHER)
    assert acquired_me is True and acquired_other is True
    await _start_generating(srv, ME)
    await _start_generating(srv, OTHER)
    assert srv.local_active_queries == POOL

    acquired, owner, _request = await _acquire(srv, ME)

    assert acquired is False, (
        "a second same-session dispatch must be denied while the generating "
        "pool is full"
    )
    assert owner in {ME, OTHER}
    assert helpers._outstanding_dispatch_credits(srv) == POOL, (
        "a denied dispatch must not take a credit: "
        f"{srv.local_dispatch_records!r}"
    )
    assert srv.local_active_queries == POOL


# ======================================================================
# AC2 + risk mitigation — a partially finished session still counts
# ======================================================================


@pytest.mark.asyncio
async def test_partially_finished_own_record_still_counts(
    monkeypatch, install_records, caplog
):
    """AC2: two dispatches, only the first finished ⇒ the record still counts.

    The parent's risk mitigation keys on live dispatch work: a dispatch that
    has emitted its terminal chunk leaves the streaming set, but the sibling
    dispatch is still streaming, so the record remains occupancy and a third
    concurrent stream from the same session cannot be admitted.
    """
    caplog.set_level(logging.INFO, logger="llama-proxy")
    records = install_records(
        {
            (ENDPOINT, ME): _dispatch_record(
                ME,
                credits=[ME_TOKEN, ME_TOKEN_SIBLING],
                # The first dispatch's terminal chunk was emitted; the sibling
                # is still streaming.
                streaming=[ME_TOKEN_SIBLING],
            ),
            (ENDPOINT, OTHER): _dispatch_record(
                OTHER, credits=[OTHER_TOKEN], streaming=[OTHER_TOKEN]
            ),
        },
        generating=2,
    )

    own = records[(ENDPOINT, ME)]
    assert len(own["dispatch_credits"]) == 2
    assert len(own["streaming_dispatch_tokens"]) == 1, (
        "one dispatch is still live, so exactly one streaming token remains"
    )
    assert set(own["streaming_dispatch_tokens"]) <= set(own["dispatch_credits"]), (
        "the streaming set must stay a subset of the held credits"
    )

    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    ) == (POOL, POOL), (
        "a partially finished own record still holds a live dispatch and must "
        "count toward the pool"
    )

    calls: list[str] = []
    monkeypatch.setattr(
        provider, "_get_local_provider_endpoint", lambda *a, **k: ENDPOINT
    )
    result = await _route(calls)

    assert result.status_code == 200
    assert calls == ["remote"], (
        "a third concurrent stream must not be admitted while one dispatch of "
        f"the session is still live, got {calls}"
    )
    assert "reason=local_concurrency_limit" in caplog.text


# ======================================================================
# AC3 — anonymous readers exclude nothing
# ======================================================================


def test_anonymous_readers_exclude_nothing(install_records):
    """AC3: ``session_id=None`` (and omitted) counts every record as today.

    ``None`` is the anonymous/legacy contract (and the status/reserve shape):
    the own-record exclusion is a session-scoped read, so anonymous callers and
    an unrelated requester must both see every active-or-unexpired record —
    including a finished-but-not-yet-released own record.
    """
    install_records(
        {
            # The requester's own finished-but-not-released record: no
            # streaming tokens, credit still held until teardown.
            (ENDPOINT, ME): _dispatch_record(ME, credits=[ME_TOKEN], streaming=[]),
            # Another session's idle (no-preemption) lease.
            (ENDPOINT, OTHER): _dispatch_record(
                OTHER, credits=[OTHER_TOKEN], active=False
            ),
            # A genuinely streaming third session.
            (ENDPOINT, THIRD): _dispatch_record(
                THIRD, credits=[THIRD_TOKEN], streaming=[THIRD_TOKEN]
            ),
        },
        generating=1,
    )
    config = _config(session_slot_pool_size=3)

    omitted = provider._get_local_concurrency_info(config, endpoint=ENDPOINT)
    explicit_none = provider._get_local_concurrency_info(
        config, endpoint=ENDPOINT, session_id=None
    )
    unrelated = provider._get_local_concurrency_info(
        config, endpoint=ENDPOINT, session_id="session-unrelated"
    )

    assert omitted == (3, 3)
    assert explicit_none == omitted, "session_id=None must equal the omitted form"
    assert unrelated == omitted, (
        "the exclusion applies only to the requester's own key; another "
        "session must never filter this record away"
    )


def test_only_the_matching_session_is_excluded(install_records):
    """AC3: the session-scoped read never widens into a global exclusion."""
    install_records(
        {
            # Both are finished-but-unreleased shapes; `ME`'s is the
            # requester's own only when `ME` is the requester, so a *different*
            # session must see both (the exclusion is session-scoped, it never
            # leaks into a global filter).
            (ENDPOINT, ME): _dispatch_record(ME, credits=[ME_TOKEN], streaming=[]),
            (ENDPOINT, OTHER): _dispatch_record(
                OTHER, credits=[OTHER_TOKEN], streaming=[]
            ),
        },
        generating=0,
    )

    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=THIRD
    ) == (POOL, POOL)


# ======================================================================
# AC3 — the status / reserve pool-description readers
# ======================================================================


class _UnlockedLock:
    """Async + sync context manager that never blocks (test stand-in)."""

    def locked(self) -> bool:
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _ReserveRequest:
    """Minimal request stub for ``POST /llama/local/reserve``."""

    def __init__(self, body: dict):
        self._body = body

    async def json(self) -> dict:
        return self._body


@pytest.mark.asyncio
async def test_reserve_reader_counts_every_session_and_forwards_no_session(
    monkeypatch, install_records
):
    """AC3: the reserve occupancy reader excludes nothing and is anonymous.

    ``/llama/local/reserve`` describes the pool with the same helper, called
    with no ``session_id`` — even when the request body names a session. The
    reader therefore counts the reserving session's own record too.
    """
    import proxy.admission_reserve as admission_reserve
    import proxy.handlers as handlers

    install_records({}, generating=2)
    monkeypatch.setattr(server, "config", _config())
    monkeypatch.setattr(server, "current_model", "Qwen3")

    # The helper is reached through the module attribute (the wrapper resolves
    # it at call time), so a spy records exactly what the reserve path asked for.
    seen_kwargs: list[dict] = []
    real_info = provider._get_local_concurrency_info

    def _spy(config, **kwargs):
        seen_kwargs.append(dict(kwargs))
        return real_info(config, **kwargs)

    monkeypatch.setattr(provider, "_get_local_concurrency_info", _spy)

    captured: dict = {}

    def _fake_reserve(body, config, **kwargs):
        captured["body"] = body
        captured.update(kwargs)
        return 200, {"admit": True, "reservation_id": "res-test"}

    monkeypatch.setattr(admission_reserve, "reserve", _fake_reserve)

    response = await handlers.reserve_local_slot(
        _ReserveRequest({"session_id": ME, "prompt_tokens": 100})
    )

    assert response.status_code == 200
    assert captured["busy_slots"] == POOL, (
        "the reserve reader must describe the whole pool (nothing excluded); "
        f"got busy_slots={captured.get('busy_slots')!r}"
    )
    assert seen_kwargs, "the reserve path must read occupancy through the helper"
    assert all("session_id" not in kwargs for kwargs in seen_kwargs), (
        "the reserve reader must stay anonymous: "
        f"{seen_kwargs}"
    )
    # The same helper with no session_id excludes nothing.
    assert provider._get_local_concurrency_info(
        _config(), endpoint=None, session_id=None
    ) == (POOL, POOL)


@pytest.mark.asyncio
async def test_status_live_work_attribution_excludes_nothing(monkeypatch):
    """AC3: ``/llama/local/status`` describes the session's own record too.

    The status surface derives its live-work attribution from the dispatch
    registry with no requester in scope, so an own finished-but-not-yet-released
    record is still described (owner + in-flight count) rather than filtered
    out as "mine". An idle lease is described but is not live work.
    """
    import httpx
    from proxy.server import app

    lock = _UnlockedLock()
    records = {
        (ENDPOINT, ME): _dispatch_record(ME, credits=[ME_TOKEN], streaming=[]),
        (ENDPOINT, OTHER): _dispatch_record(
            OTHER, credits=[OTHER_TOKEN], active=False
        ),
    }

    async def _fake_status():
        return {"llama_server_running": True}

    async def _no_counts(*_args, **_kwargs):
        return (0, 0)

    async def _no_detail(*_args, **_kwargs):
        return []

    monkeypatch.setattr(server, "query_llama_status", _fake_status)
    monkeypatch.setattr(server, "local_dispatch_records", records)
    monkeypatch.setattr(server, "local_dispatch_records_lock", lock)
    monkeypatch.setattr(server, "local_active_queries", 1)
    monkeypatch.setattr(server, "local_active_queries_lock", lock)
    monkeypatch.setattr(server, "active_queries", 1)
    monkeypatch.setattr(server, "active_queries_lock", lock)
    monkeypatch.setattr(server, "model_switch_refcount", 0)
    monkeypatch.setattr(server, "model_switch_lock", lock)
    monkeypatch.setattr(server, "model_switch_refcount_lock", lock)
    monkeypatch.setattr(server, "background_loads", {})
    monkeypatch.setattr(server, "current_model", "Qwen3")
    monkeypatch.setattr(server, "config", _config())
    monkeypatch.setattr("proxy.observability._query_slots", _no_counts)
    monkeypatch.setattr("proxy.observability._query_slots_detail", _no_detail)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/llama/local/status")

    assert resp.status_code == 200
    body = resp.json()
    assert body["local_in_flight_requests"] == 1, (
        "the status reader must count the session's own active record: "
        f"{body.get('local_in_flight_requests')!r}"
    )
    assert body["local_owner_session_id"] == ME, (
        "the status owner must not filter out the session's own record: "
        f"{body.get('local_owner_session_id')!r}"
    )
    assert body["local_active_query"] is True


# ======================================================================
# AC4 — no false admission: the lease gate stays the capacity authority
# ======================================================================


@pytest.mark.asyncio
async def test_refused_then_admitted_dispatch_never_exceeds_the_pool(monkeypatch):
    """AC4: the routing read never grants capacity; the lease gate bounds it.

    Two genuinely live streams fill a 2-slot pool, so a third dispatch is
    refused at the routing gate *and* denied at the lease choke point (the
    authority). Once one stream is released the previously refused dispatch is
    admitted, and the number of live local streams never exceeds the pool.
    """
    srv = _lease_server()
    admitted_me, _owner, request_me = await _acquire(srv, ME)
    admitted_other, _owner, request_other = await _acquire(srv, OTHER)
    assert admitted_me is True and admitted_other is True
    await _start_generating(srv, ME)
    await _start_generating(srv, OTHER)

    # The routing view and the lease view agree: the pool is full.
    _install_registry(monkeypatch, srv)
    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=THIRD
    ) == (POOL, POOL)

    refused, _owner, _refused_request = await _acquire(srv, THIRD)
    assert refused is False, (
        "the lease choke point must refuse the third stream even though the "
        "routing gate already refused it"
    )
    assert helpers._outstanding_dispatch_credits(srv) == POOL
    assert srv.local_active_queries == POOL, (
        "a refused dispatch must not lift local_active_queries above the pool"
    )

    # One stream finishes and its slot is released.
    await _release(srv, OTHER, request=request_other)
    assert helpers._outstanding_dispatch_credits(srv) == 1
    assert srv.local_active_queries == 1

    # The refusal was not permanent: the same dispatch is now admitted.
    admitted, _owner, _third_request = await _acquire(srv, THIRD)
    assert admitted is True, "a freed slot must admit the previously refused stream"
    assert helpers._outstanding_dispatch_credits(srv) == POOL, (
        "the refused-then-admitted dispatch must not overshoot the pool: "
        f"{srv.local_dispatch_records!r}"
    )
    assert srv.local_active_queries == POOL

    await _release(srv, ME, request=request_me)
    assert helpers._outstanding_dispatch_credits(srv) <= POOL
    assert srv.local_active_queries <= POOL


# ======================================================================
# Risk mitigation — a client disconnect is not "finished"
# ======================================================================


@pytest.mark.asyncio
async def test_client_disconnect_keeps_the_live_sibling_counted(monkeypatch):
    """Risk: a stream ending by client disconnect must not free its slot.

    A session holding two overlapping dispatches loses the first one to a
    client disconnect. The synchronous disconnect release must not delete the
    shared record (the sibling still holds its credit), and the routing read
    must keep counting that record: a third stream is gated at the routing gate
    and denied at the lease choke point.
    """
    srv = _lease_server()
    admitted_a, _owner, request_a = await _acquire(srv, ME)
    admitted_b, _owner, request_b = await _acquire(srv, ME)
    assert admitted_a is True and admitted_b is True
    assert helpers._outstanding_dispatch_credits(srv) == 2

    # The first dispatch's client vanishes mid-stream: the synchronous
    # pre-teardown release runs, the credit is still held until cleanup.
    helpers._release_client_disconnect_dispatch(
        srv, ME, endpoint=ENDPOINT, request=request_a
    )
    assert helpers._outstanding_dispatch_credits(srv) == 2, (
        "a client disconnect must not release a credit the teardown owns: "
        f"{srv.local_dispatch_records!r}"
    )
    _install_registry(monkeypatch, srv)
    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    ) == (1, POOL), (
        "a disconnected-but-credit-holding own record must still count"
    )

    # Teardown of the disconnected dispatch releases exactly its own credit.
    await _release(srv, ME, request=request_a, disconnected=True)
    assert helpers._outstanding_dispatch_credits(srv) == 1
    assert srv.local_dispatch_records.get((ENDPOINT, ME)) is not None, (
        "the disconnect must not delete the record the live sibling still owns"
    )
    _install_registry(monkeypatch, srv)
    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    ) == (1, POOL), (
        "the surviving sibling dispatch keeps the record as occupancy"
    )

    # A second session starts streaming: the pool is genuinely full.
    admitted_other, _owner, request_other = await _acquire(srv, OTHER)
    assert admitted_other is True
    await _start_generating(srv, ME)
    await _start_generating(srv, OTHER)
    _install_registry(monkeypatch, srv)

    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    ) == (POOL, POOL), (
        "a client-disconnect-terminated stream in this session must not be "
        "treated as 'finished' while its sibling is live"
    )
    refused, _owner, _third_request = await _acquire(srv, THIRD)
    assert refused is False, (
        "a third concurrent stream must not be admitted after a client "
        "disconnect left a sibling dispatch live"
    )

    # Everything settles exactly once.
    await _release(srv, ME, request=request_b)
    await _release(srv, OTHER, request=request_other)
    assert helpers._outstanding_dispatch_credits(srv) == 0
    assert srv.local_active_queries == 0
