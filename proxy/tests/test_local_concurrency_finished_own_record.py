"""Red regression: a sequential turn refused by its own finished record.

Parent LP-0MWMSQR3O002YNPW (AC1); test item LP-0MWNHBBMS002WBLQ.

The production ordering pinned here (incident trace 2026-11-18 22:38:52,
``herdr-1795041420-110182-25266``) — no wall-clock sleeps anywhere:

1. the session acquires a local dispatch: the record is ``active: True`` and
   holds one ``dispatch_credits`` token;
2. the stream reaches its **terminal chunk**: that dispatch's token leaves the
   record's streaming set (the write side is LP-0MWNHBFN6000GH2J), while
   ``active`` and ``dispatch_credits`` stay untouched — the credit is still
   held until teardown;
3. the **same session issues its next request ~145 ms later**, i.e. before
   ``request._cleanup_after_request`` (``router.py``) has run
   ``_decrement_local_active_queries`` (measured 1.41 s after
   ``Stream finished``);
4. ``proxy_with_fallback`` evaluates the routing concurrency gate.

On the current tree ``_is_own_idle_lease`` returns ``False`` whenever
``record["active"]`` is truthy, so the requester's own finished-but-not-released
record counts toward ``cur_local``. With ``session_slot_pool_size: 2`` and one
*other* session genuinely streaming, the pool reads 2/2, the gate records
``status=local_concurrency_limit`` and exiles the request remote — even though
the dispatch-lease gate (``own_has_lease``) would have admitted it.

Red-first encoding: the two contract assertions that only hold once
LP-0MWNHBG7D002NH5S (the own-record predicate keyed on
``streaming_dispatch_tokens``) lands carry ``xfail(strict=True)`` — the
repository's red-first pattern (cf. ``test_local_active_queries_invariant.py``
and ``test_abandoned_dispatch_starvation.py``). ``strict`` makes the marker
self-cleaning: the implementation item must remove it, because an unexpected
pass is a failure. The guards that already hold on both trees (a streaming own
dispatch still counts; the gate is a read) are deliberately **not** xfail.

The record shape asserted here is the one the plan fixed
(LP-0MWMSQR3O002YNPW §"Mechanism decision"): ``streaming_dispatch_tokens`` is a
strict subset of ``dispatch_credits``, keyed by the existing per-dispatch
credit token (LP-0MVYZEYZY003IQAH). A record without that key is a legacy
record and keeps today's ``active``-only behaviour.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from unittest.mock import AsyncMock, patch

import proxy.provider as provider
import proxy.server as server
import pytest
from starlette.responses import Response

ENDPOINT = "http://localhost:8080"
POOL = 2
ME = "session-me"
OTHER = "session-other"

#: Per-dispatch credit tokens (LP-0MVYZEYZY003IQAH) held by the records below.
ME_TOKEN = 9001
OTHER_TOKEN = 9002

#: The implementation item that turns this module's contract assertions green.
#: It removes the ``xfail(strict=True)`` markers (kept here for provenance).
IMPL_ITEM = "LP-0MWNHBG7D002NH5S"


def _config(**server_overrides) -> dict:
    server_cfg = {
        "llama_server_port": 8080,
        "session_slot_pool_size": POOL,
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


def _streaming_record(token: int, *, model: str = "Qwen3") -> dict:
    """A live dispatch record: streaming, credit held (production post-acquire).

    ``dispatch_credits`` and ``streaming_dispatch_tokens`` both hold the same
    token while the dispatch is genuinely in flight.
    """
    now = time.monotonic()
    return {
        "backend": "local",
        "started_at": now,
        "active": True,
        "expires_at": now + 30.0,
        "model_name": model,
        "endpoint": ENDPOINT,
        "session": ME,
        "dispatch_token": token,
        "dispatch_credits": [token],
        "streaming_dispatch_tokens": [token],
    }


def _mark_terminal_chunk(record: dict, token: int) -> dict:
    """Step 2 of the ordering: the dispatch's terminal chunk released its stream.

    This is the *only* mutation the fix introduces at the terminal chunk
    (LP-0MWNHBFN6000GH2J): the token leaves the streaming set and nothing else
    changes. ``active`` is still ``True`` and the credit is still held, so the
    record is indistinguishable from live work today — which is the defect.
    """
    record["streaming_dispatch_tokens"].remove(token)
    return record


@pytest.fixture
def install_state(monkeypatch):
    """Install the dispatch registry and the four local counters."""

    def _install(records: dict, *, generating: int = 1, active: int = 1) -> dict:
        monkeypatch.setattr(server, "local_dispatch_records", records, raising=False)
        monkeypatch.setattr(server, "local_generating_queries", generating, raising=False)
        monkeypatch.setattr(server, "local_active_queries", active, raising=False)
        monkeypatch.setattr(server, "active_queries", active, raising=False)
        monkeypatch.setattr(server, "local_generating_sessions", set(), raising=False)
        return records

    return _install


def _snapshot() -> dict:
    """Deep copy of everything the routing gate must treat as read-only."""
    return {
        "records": copy.deepcopy(getattr(server, "local_dispatch_records", {})),
        "local_active_queries": getattr(server, "local_active_queries", None),
        "local_generating_queries": getattr(server, "local_generating_queries", None),
        "active_queries": getattr(server, "active_queries", None),
        "local_generating_sessions": set(
            getattr(server, "local_generating_sessions", set())
        ),
    }


# ======================================================================
# The count itself
# ======================================================================


def test_own_streaming_dispatch_still_counts(install_state):
    """AC2 guard (live on both trees): genuine in-flight work keeps counting.

    The requester's own record still holds its streaming token, so it is not
    the finished-but-not-released shape: with one other session streaming, the
    pool is genuinely full and today's count must not move.
    """
    install_state(
        {
            (ENDPOINT, ME): _streaming_record(ME_TOKEN),
            (ENDPOINT, OTHER): _streaming_record(OTHER_TOKEN),
        }
    )

    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    ) == (POOL, POOL)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "red until the own-record predicate keys on streaming_dispatch_tokens "
        f"({IMPL_ITEM}): a finished-but-not-released own record still counts"
    ),
)
def test_own_finished_record_is_not_occupancy(install_state):
    """AC1/AC4: ME's own finished record must not make the pool read saturated.

    Only *another* session is genuinely streaming; the second occupied slot is
    ME's own record whose terminal chunk has been emitted but whose teardown
    (``_cleanup_after_request``) has not run yet. The pool is under-full from
    the requester's point of view, so the routing count must be ``1 < 2``.
    """
    records = install_state(
        {
            (ENDPOINT, ME): _streaming_record(ME_TOKEN),
            (ENDPOINT, OTHER): _streaming_record(OTHER_TOKEN),
        }
    )

    # Step 2 of the production ordering: the own stream hits its terminal chunk.
    record = _mark_terminal_chunk(records[(ENDPOINT, ME)], ME_TOKEN)
    # The terminal-chunk marker must not release the credit or clear active:
    # teardown owns that, and doing it here would break exactly-once release.
    assert record["active"] is True
    assert record["dispatch_credits"] == [ME_TOKEN]
    assert record["streaming_dispatch_tokens"] == []

    # Step 3: the next request from the same session, before teardown.
    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    ) == (1, POOL)


# ======================================================================
# End-to-end through the routing gate
# ======================================================================


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


async def _route(calls: list[str]) -> Response:
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
            _Request(), "v1/chat/completions", _model_config(), _config()
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "red until the own-record predicate keys on streaming_dispatch_tokens "
        f"({IMPL_ITEM}): the sequential turn is exiled with local_concurrency_limit"
    ),
)
@pytest.mark.asyncio
async def test_sequential_turn_after_terminal_chunk_routes_locally(
    monkeypatch, install_state, caplog
):
    """AC1: terminal chunk → next turn before teardown ⇒ served locally.

    This is the incident ordering: ME's previous turn is finished upstream but
    its record is still ``active`` because teardown has not released the credit;
    one other session is genuinely streaming. The returning request must be
    dispatched locally and must not be recorded as ``local_concurrency_limit``.
    """
    caplog.set_level(logging.INFO, logger="llama-proxy")
    records = install_state(
        {
            (ENDPOINT, ME): _streaming_record(ME_TOKEN),
            (ENDPOINT, OTHER): _streaming_record(OTHER_TOKEN),
        }
    )
    _mark_terminal_chunk(records[(ENDPOINT, ME)], ME_TOKEN)
    before = _snapshot()
    calls: list[str] = []
    monkeypatch.setattr(provider, "_get_local_provider_endpoint", lambda *a, **k: ENDPOINT)

    result = await _route(calls)

    assert result.status_code == 200
    assert calls == [
        "local"
    ], f"the sequential turn must be served locally, got {calls}"
    assert "local_concurrency_limit" not in caplog.text
    # AC5: the routing gate is a read; dispatching through it (proxy_to_local
    # is stubbed here) must leave the registry and counters untouched.
    assert _snapshot() == before


# ======================================================================
# Read-only invariant
# ======================================================================


def test_routing_gate_read_is_free_of_mutation(install_state):
    """AC5: reading occupancy never mutates records, counters or the registry.

    The exactly-once release invariant (LP-0MVWF78QK0060TZH) depends on the
    routing gate staying a pure read: ``_try_acquire_local_dispatch`` remains
    the only capacity authority. The gate is called with and without a
    ``session_id`` (the anonymous/status shape must stay untouched too).
    """
    records = install_state(
        {
            (ENDPOINT, ME): _streaming_record(ME_TOKEN),
            (ENDPOINT, OTHER): _streaming_record(OTHER_TOKEN),
        }
    )
    _mark_terminal_chunk(records[(ENDPOINT, ME)], ME_TOKEN)
    before = _snapshot()

    provider._get_local_concurrency_info(_config(), endpoint=ENDPOINT, session_id=ME)
    provider._get_local_concurrency_info(_config(), endpoint=ENDPOINT)
    provider._get_local_concurrency_info(_config())

    assert server.local_dispatch_records is records, "the registry must not be rebound"
    assert _snapshot() == before
    # The requester's own record is still the untouched finished shape.
    assert records[(ENDPOINT, ME)]["active"] is True
    assert records[(ENDPOINT, ME)]["dispatch_credits"] == [ME_TOKEN]
    assert records[(ENDPOINT, ME)]["streaming_dispatch_tokens"] == []
