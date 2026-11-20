"""Terminal-chunk finished marking on the local dispatch record.

Parent LP-0MWMSQR3O002YNPW (AC1, AC2, AC4); implementation item
LP-0MWNHBFN6000GH2J.

The parent's routing gate needs to tell a **finished-but-not-yet-released**
turn from a genuinely in-flight one. Teardown
(``request._cleanup_after_request`` → ``_decrement_local_active_queries``) runs
**after** the stream ends (measured 1.41 s in the incident), so the record is
still ``active`` when the same client issues its next turn ~145 ms after the
last chunk. This module pins the write side of the mechanism the parent fixed
(LP-0MWMSQR3O002YNPW §"Mechanism decision"):

``streaming_dispatch_tokens`` is a set of the record's existing per-dispatch
credit tokens (LP-0MVYZEYZY003IQAH) whose stream has **not** emitted its
terminal chunk. The token is added when the dispatch takes its credit and
removed at the terminal chunk. ``active`` and ``dispatch_credits`` are never
touched by the marker, so exactly-once release
(LP-0MVWF78QK0060TZH) is unchanged and the credit still belongs to teardown.

Acceptance-criterion map
------------------------

- **AC1** — ``test_terminal_chunk_removes_the_token_exactly_once`` and
  ``test_streaming_terminal_chunk_marks_finished_before_teardown`` (the real
  ``proxy_to_local`` stream) pin the removal and the per-request guard that
  makes a repeated detection a no-op.
- **AC2** — ``test_streaming_set_stays_a_subset_of_credits_across_the_lifecycle``
  asserts the subset invariant after acquisition, after one of two overlapping
  dispatches finishes and after teardown.
- **AC3** — ``test_marking_does_not_release_the_credit_or_clear_active`` and
  ``test_teardown_still_claims_and_releases_the_credit_exactly_once`` pin that
  marking changes no counter and that teardown's exactly-once claim is
  untouched.
- **AC4** — ``test_non_streaming_cleanup_marks_finished_before_the_credit_claim``
  drives ``router._cleanup_after_request`` (the non-streaming path's teardown)
  and observes the marker at the moment the credit is claimed.
- **AC5** — ``test_client_disconnect_release_keeps_the_record_counting`` and
  ``test_watchdog_eviction_counts_until_the_credit_is_released`` pin that the
  disconnect and watchdog release paths never mark a live stream finished.
- **AC6** — ``test_marking_is_best_effort_and_never_raises`` pins that a
  failure logs and leaves the record counting (the conservative outcome).

Follow-up LP-0MWOZFF7T001YA87 (the synthesised error chunk is a terminal chunk):

- ``test_stream_error_terminal_chunk_marks_finished_before_teardown`` — a local
  stream that fails (``Local stream error``) has its token removed as the
  synthesised ``finish_reason=error`` chunk is forwarded, while the credit is
  still held;
- ``test_stream_error_finished_window_admits_the_next_turn`` — with one other
  session holding a slot the pool reads 1/2 at that moment, so the retry is not
  refused ``local_concurrency_limit``.

Blocks LP-0MWNHBG7D002NH5S (the own-record predicate that reads the marker):
the read side is exercised by ``test_local_concurrency_finished_own_record.py``
and is deliberately not changed here.

Harness idioms are reused from ``test_local_active_queries_own_record_guards``
(the real lease/credit helpers) and ``test_local_active_queries_invariant``
(the in-process ``proxy_to_local`` streaming harness).
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import proxy.provider as provider
import proxy.router as router
import proxy.router_helpers as helpers
import proxy.server as server
import pytest
from fastapi.responses import StreamingResponse
from test_local_active_queries_invariant import (
    _CONTENT_CHUNK,
    _DONE,
    _FINISH_CHUNK,
    _checkpoint,
    _drive,
    _install_server,
    _Observations,
    _PacedUpstream,
    _patch_router,
    _wait,
)
from test_local_active_queries_invariant import (
    _Request as _StreamRequest,
)

ENDPOINT = "http://localhost:8080"
POOL = 2
ME = "session-me"
MODEL = "Qwen3"


# ---------------------------------------------------------------------------
# Real lease / credit harness (helper level)
# ---------------------------------------------------------------------------


def _config(**server_overrides) -> dict:
    server_cfg = {
        "llama_server_port": 8080,
        "session_slot_pool_size": POOL,
        # Two sessions must be able to hold a prefill marker at once, otherwise
        # a fixture could be denied by the serialised-prefill cap rather than by
        # the surface under test.
        "local_max_concurrent_prefills": POOL,
        "local_dispatch_lease_timeout_seconds": 30,
        "max_concurrent_queries": 8,
        "contention_queue_max_wait_seconds": 30,
        "contention_queue_max_depth": 4,
        "local_prefill_deadline_seconds": 0,
        "local_cold_prefill_budget_tokens": 25000,
        "disconnect_cleanup_timeout": 0.1,
    }
    server_cfg.update(server_overrides)
    return {"server": server_cfg, "provider_cooldown_seconds": 60}


def _lease_server(**server_overrides) -> SimpleNamespace:
    """Minimal but real server state for the counter/lease/cleanup helpers."""
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


@pytest.fixture(autouse=True)
def _reset_prefill_thrash_cooldowns():
    """Start every row with no armed post-deadline cooldown."""
    helpers._reset_prefill_thrash_cooldowns()
    yield
    helpers._reset_prefill_thrash_cooldowns()


async def _acquire(srv, session: str = ME):
    """Take one dispatch through the production choke point."""
    request = SimpleNamespace()
    acquired, owner, _count, _retry = await helpers._try_acquire_local_dispatch(
        srv, POOL, session, ENDPOINT, model_name=MODEL, request=request
    )
    assert acquired is True, f"the harness must acquire a real lease (owner={owner})"
    return request


def _record(srv, session: str = ME) -> dict:
    return srv.local_dispatch_records[(ENDPOINT, session)]


def _assert_subset(record: dict, where: str) -> None:
    """AC2: the streaming marker is always a subset of the held credits."""
    streaming = set(record["streaming_dispatch_tokens"])
    credits = set(record["dispatch_credits"])
    assert streaming <= credits, (
        f"[{where}] streaming tokens {sorted(streaming)} are not a subset of "
        f"the held credits {sorted(credits)}"
    )


# ---------------------------------------------------------------------------
# AC1 — acquisition records the dispatch as streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acquisition_marks_the_dispatch_as_streaming():
    """AC1: a fresh dispatch is streaming, keyed by its own credit token."""
    srv = _lease_server()
    request = await _acquire(srv)

    record = _record(srv)
    token = record["dispatch_token"]
    assert token == getattr(request, "_local_dispatch_token")
    assert record["dispatch_credits"] == [token]
    assert record["streaming_dispatch_tokens"] == [token]
    assert record["active"] is True
    _assert_subset(record, "after acquire")


@pytest.mark.asyncio
async def test_anonymous_acquisition_marks_the_dispatch_as_streaming():
    """AC1: the anonymous increment path stamps the marker too.

    ``_increment_local_active_queries`` is the second (anonymous) acquisition
    site; a record it creates without the marker would be a permanent
    finished-but-counting shape.
    """
    srv = _lease_server()
    request = SimpleNamespace()
    await helpers._increment_local_active_queries(
        srv, session_key=ME, backend=ENDPOINT, request=request
    )

    record = _record(srv)
    token = record["dispatch_token"]
    assert record["dispatch_credits"] == [token]
    assert record["streaming_dispatch_tokens"] == [token]
    _assert_subset(record, "after anonymous acquire")

    # A second, overlapping anonymous dispatch appends its own token.
    await helpers._increment_local_active_queries(
        srv, session_key=ME, backend=ENDPOINT, request=SimpleNamespace()
    )
    sibling = record["dispatch_token"]
    assert sibling != token
    assert record["dispatch_credits"] == [token, sibling]
    assert record["streaming_dispatch_tokens"] == [token, sibling]
    _assert_subset(record, "after overlapping anonymous acquire")


# ---------------------------------------------------------------------------
# AC1 / AC3 — the terminal chunk removes the token, nothing else
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminal_chunk_removes_the_token_exactly_once():
    """AC1/AC3: remove once, never raise the counter or clear ``active``."""
    srv = _lease_server()
    request = await _acquire(srv)
    record = _record(srv)
    token = record["dispatch_token"]
    counter_before = srv.local_active_queries

    marked = await helpers._mark_local_dispatch_stream_finished(
        srv, ME, endpoint=ENDPOINT, request=request
    )

    assert marked is True
    assert record["streaming_dispatch_tokens"] == []
    # AC3: marking is a marker, not a release — the credit is still held and
    # teardown is still the only owner of the decrement.
    assert record["dispatch_credits"] == [token]
    assert record["active"] is True
    assert srv.local_active_queries == counter_before
    assert helpers._outstanding_dispatch_credits(srv) == 1

    # The per-request guard makes a repeated detection a no-op.
    assert (
        await helpers._mark_local_dispatch_stream_finished(
            srv, ME, endpoint=ENDPOINT, request=request
        )
        is False
    )
    assert record["streaming_dispatch_tokens"] == []
    assert record["dispatch_credits"] == [token]
    assert srv.local_active_queries == counter_before


@pytest.mark.asyncio
async def test_teardown_still_claims_and_releases_the_credit_exactly_once():
    """AC3: exactly-once release is unchanged by the marker."""
    srv = _lease_server()
    request = await _acquire(srv)
    record = _record(srv)
    token = record["dispatch_token"]

    await helpers._mark_local_dispatch_stream_finished(
        srv, ME, endpoint=ENDPOINT, request=request
    )

    released = await helpers._decrement_local_active_queries(
        srv, session_key=ME, backend=ENDPOINT, dispatch_token=token
    )
    assert released is True
    assert srv.local_active_queries == 0
    assert record["dispatch_credits"] == []
    assert record["active"] is False
    # The teardown prunes the marker too, so the subset holds after release.
    assert record["streaming_dispatch_tokens"] == []
    _assert_subset(record, "after teardown")

    # A second teardown (the disconnect release racing the cleanup) is a no-op.
    assert (
        await helpers._decrement_local_active_queries(
            srv, session_key=ME, backend=ENDPOINT, dispatch_token=token
        )
        is False
    )
    assert srv.local_active_queries == 0


# ---------------------------------------------------------------------------
# AC2 — the subset invariant across the lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_set_stays_a_subset_of_credits_across_the_lifecycle():
    """AC2: after acquire, after one of two finishes, and after teardown."""
    srv = _lease_server()
    request_a = await _acquire(srv)
    token_a = _record(srv)["dispatch_token"]
    await _acquire(srv)
    record = _record(srv)
    token_b = record["dispatch_token"]

    # After acquire: two overlapping dispatches, two credits, two streams.
    assert token_b != token_a
    assert record["dispatch_credits"] == [token_a, token_b]
    assert record["streaming_dispatch_tokens"] == [token_a, token_b]
    _assert_subset(record, "after two acquires")

    # After one of two overlapping dispatches finishes: its token leaves the
    # streaming set while its credit is still held for teardown.
    await helpers._mark_local_dispatch_stream_finished(
        srv, ME, endpoint=ENDPOINT, request=request_a
    )
    assert record["streaming_dispatch_tokens"] == [token_b]
    assert record["dispatch_credits"] == [token_a, token_b]
    _assert_subset(record, "after one of two finishes")

    # After that dispatch's teardown: its credit and its marker are both gone.
    await helpers._decrement_local_active_queries(
        srv, session_key=ME, backend=ENDPOINT, dispatch_token=token_a
    )
    assert record["dispatch_credits"] == [token_b]
    assert record["streaming_dispatch_tokens"] == [token_b]
    _assert_subset(record, "after one teardown")

    # After the last teardown: no credit, no marker, the record is inactive.
    await helpers._decrement_local_active_queries(
        srv, session_key=ME, backend=ENDPOINT, dispatch_token=token_b
    )
    assert record["dispatch_credits"] == []
    assert record["streaming_dispatch_tokens"] == []
    assert record["active"] is False
    assert srv.local_active_queries == 0
    _assert_subset(record, "after full teardown")


@pytest.mark.asyncio
async def test_teardown_without_a_mark_prunes_the_marker():
    """AC2: a credit claimed from a still-streaming dispatch prunes its token.

    The teardown can arrive without the terminal-chunk mark (a buffered
    request, a cancelled stream): the streaming set must still never retain a
    token whose credit is gone.
    """
    srv = _lease_server()
    await _acquire(srv)
    record = _record(srv)
    token = record["dispatch_token"]
    assert record["streaming_dispatch_tokens"] == [token]

    await helpers._decrement_local_active_queries(
        srv, session_key=ME, backend=ENDPOINT, dispatch_token=token
    )

    assert record["dispatch_credits"] == []
    assert record["streaming_dispatch_tokens"] == []
    _assert_subset(record, "teardown without a mark")


# ---------------------------------------------------------------------------
# AC4 — the non-streaming path marks at teardown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_streaming_cleanup_marks_finished_before_the_credit_claim(
    monkeypatch,
):
    """AC4: the buffered path marks before the credit is claimed.

    The non-streaming response has no terminal chunk of its own, so its
    dispatch is marked before/at teardown — and, crucially, *before* the credit
    claim, so the record's end state is the same consistent shape (no streaming
    token left behind, no credit left behind).
    """
    srv = _lease_server()
    request = await _acquire(srv)
    record = _record(srv)
    token = record["dispatch_token"]

    streaming_at_claim: list[list[int]] = []
    real_claim = helpers._claim_local_dispatch_credit

    async def _spy_claim(*args, **kwargs):
        streaming_at_claim.append(list(record["streaming_dispatch_tokens"]))
        return await real_claim(*args, **kwargs)

    monkeypatch.setattr(helpers, "_claim_local_dispatch_credit", _spy_claim)

    await router._cleanup_after_request(
        srv,
        ME,
        decrement_local=True,
        session_explicit=True,
        model_name=MODEL,
        request=request,
        backend=ENDPOINT,
    )

    assert streaming_at_claim == [[]], (
        "the buffered path must mark the dispatch finished before the credit "
        f"claim; observed {streaming_at_claim}"
    )
    assert record["dispatch_credits"] == []
    assert record["streaming_dispatch_tokens"] == []
    assert record["active"] is False
    assert srv.local_active_queries == 0
    assert token == getattr(request, "_local_dispatch_token", token)


# ---------------------------------------------------------------------------
# AC5 — release paths that are not a terminal chunk
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_disconnect_release_keeps_the_record_counting():
    """AC5: a disconnected dispatch keeps counting while its credit is held.

    The client-disconnect release must not mark the live sibling's stream
    finished: the shared record keeps its remaining credit and the sibling's
    streaming token, so it is still occupancy for the routing gate.
    """
    srv = _lease_server()
    request_a = await _acquire(srv)
    token_a = request_a._local_dispatch_token
    request_b = await _acquire(srv)
    token_b = request_b._local_dispatch_token
    record = _record(srv)

    released = helpers._release_client_disconnect_dispatch(
        srv, ME, endpoint=ENDPOINT, request=request_a
    )
    assert released is True

    record = _record(srv)
    assert record["dispatch_credits"] == [token_a, token_b], (
        "the disconnect release must not consume the sibling's credit"
    )
    assert token_b in record["streaming_dispatch_tokens"], (
        "the live sibling must keep streaming: a disconnect is not a terminal "
        "chunk"
    )
    _assert_subset(record, "after client disconnect")
    assert helpers._outstanding_dispatch_credits(srv) == 2

    # The paired end-of-request cleanup releases only this request's credit.
    await router._cleanup_after_request(
        srv,
        ME,
        disconnected=True,
        decrement_local=True,
        session_explicit=True,
        model_name=MODEL,
        request=request_a,
        backend=ENDPOINT,
    )
    record = _record(srv)
    assert record["dispatch_credits"] == [token_b]
    assert record["streaming_dispatch_tokens"] == [token_b]
    assert helpers._outstanding_dispatch_credits(srv) == 1
    _assert_subset(record, "after disconnect cleanup")


@pytest.mark.asyncio
async def test_watchdog_eviction_counts_until_the_credit_is_released(
    monkeypatch,
):
    """AC5: the watchdog counts a live-but-wedged dispatch until it evicts it.

    The no-progress watchdog is a *release* path, not a terminal chunk: while
    it is still counting, the record keeps its streaming token and credit, and
    the eviction then reclaims both.
    """
    srv = _lease_server(local_dispatch_no_progress_timeout_seconds=1)
    request = await _acquire(srv)
    record = _record(srv)
    token = record["dispatch_token"]

    # Still streaming → still counting, even though the dispatch is wedged.
    assert record["streaming_dispatch_tokens"] == [token]
    assert helpers._outstanding_dispatch_credits(srv) == 1
    assert srv.local_active_queries == 1

    record["last_progress_ts"] = time.monotonic() - 100.0
    record["last_progress"] = 0
    monkeypatch.setattr(
        helpers, "_query_slot_processing", AsyncMock(return_value=False)
    )

    await helpers._cleanup_stale_local_dispatch(srv)

    assert (ENDPOINT, ME) not in srv.local_dispatch_records, (
        "the no-progress watchdog must evict the wedged dispatch record"
    )
    assert srv.local_active_queries == 0, (
        "the eviction must reclaim the wedged dispatch's counter credit"
    )
    assert getattr(request, "_local_dispatch_stream_finished", False) is False, (
        "the eviction path must not mark a stream finished — it releases the "
        "dispatch instead"
    )


# ---------------------------------------------------------------------------
# AC6 — best-effort
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_marking_is_best_effort_and_never_raises():
    """AC6: a failure logs and leaves the record counting."""
    # No registry at all (legacy state): no marker to maintain, no raise.
    bare = SimpleNamespace(config=_config(), logger=MagicMock())
    assert (
        await helpers._mark_local_dispatch_stream_finished(
            bare, ME, endpoint=ENDPOINT, request=SimpleNamespace()
        )
        is False
    )

    # A legacy record without the marker key keeps its ``active``-only shape.
    srv = _lease_server()
    legacy = {
        "active": True,
        "expires_at": time.monotonic() + 30.0,
        "dispatch_token": 4242,
    }
    srv.local_dispatch_records[(ENDPOINT, ME)] = legacy
    assert (
        await helpers._mark_local_dispatch_stream_finished(
            srv, ME, endpoint=ENDPOINT, request=SimpleNamespace()
        )
        is False
    )
    assert "streaming_dispatch_tokens" not in legacy

    # An untracked caller cannot name its dispatch: it must not guess, so the
    # record keeps counting.
    srv = _lease_server()
    request = await _acquire(srv)
    record = _record(srv)
    token = record["dispatch_token"]
    assert (
        await helpers._mark_local_dispatch_stream_finished(
            srv, ME, endpoint=ENDPOINT, request=SimpleNamespace()
        )
        is False
    )
    assert record["streaming_dispatch_tokens"] == [token]

    # An unexpected error is logged and swallowed, leaving the record counting.
    class _Boom(dict):
        def get(self, key, default=None):
            raise RuntimeError("boom")

    srv = _lease_server()
    srv.local_dispatch_records[(ENDPOINT, ME)] = _Boom()
    assert (
        await helpers._mark_local_dispatch_stream_finished(
            srv, ME, endpoint=ENDPOINT, request=request
        )
        is False
    )
    srv.logger.warning.assert_called()
    warning_lines = [
        c.args[0] for c in srv.logger.warning.call_args_list if c.args
    ]
    assert any(
        "Failed to mark dispatch stream finished" in line
        for line in warning_lines
    ), warning_lines


# ---------------------------------------------------------------------------
# AC1 / AC4 — the real streaming router marks at its terminal chunk
# ---------------------------------------------------------------------------

SESSION_STREAM = "sess-stream-finished-marking"


@pytest.mark.asyncio
async def test_streaming_terminal_chunk_marks_finished_before_teardown(monkeypatch):
    """AC1: the real streaming path marks at the terminal chunk.

    The dispatch is still ``active`` with its credit held when the terminal
    chunk has been forwarded to the client — that is exactly the
    finished-but-not-yet-released window the routing gate keys on — and the
    credit is released only at teardown.
    """
    helpers._reset_prefill_thrash_cooldowns()
    _install_server(monkeypatch)
    upstream = _PacedUpstream()
    plans = {"stream-finished": upstream}
    sent: list[tuple[str, dict]] = []
    _patch_router(monkeypatch, plans, sent)

    observations = _Observations()
    request = _StreamRequest(SESSION_STREAM, "stream-finished")
    task = asyncio.create_task(_drive(request))

    await _wait(upstream.parked, "first-byte park")
    key = next(iter(server.local_dispatch_records))
    record = server.local_dispatch_records[key]
    token = record["dispatch_token"]
    assert record["streaming_dispatch_tokens"] == [token]
    assert server.local_active_queries == 1
    await _checkpoint(server, observations, "admission")

    # A content chunk is forwarded; the dispatch is still streaming.
    upstream.push(_CONTENT_CHUNK)
    upstream.parked.clear()
    await _wait(upstream.parked, "content chunk consumed")
    assert record["streaming_dispatch_tokens"] == [token], (
        "a non-terminal chunk must not mark the stream finished"
    )
    await _checkpoint(server, observations, "per_chunk")

    # The terminal chunk: the marker clears while the credit is still held.
    upstream.push(_FINISH_CHUNK)
    upstream.parked.clear()
    await _wait(upstream.parked, "finish chunk consumed")
    assert record["streaming_dispatch_tokens"] == [], (
        "the terminal chunk must remove this dispatch's streaming token"
    )
    assert record["dispatch_credits"] == [token], (
        "the terminal chunk must not release the credit — teardown owns that"
    )
    assert record["active"] is True
    assert server.local_active_queries == 1
    await _checkpoint(server, observations, "terminal_chunk")

    # A second terminal marker ([DONE]) must not remove anything twice.
    upstream.push(_DONE)
    upstream.parked.clear()
    await _wait(upstream.parked, "[DONE] consumed")
    assert record["streaming_dispatch_tokens"] == []
    assert record["dispatch_credits"] == [token]

    # Teardown releases the credit and leaves the consistent end state.
    upstream.end()
    result, received = await asyncio.wait_for(task, timeout=10)
    assert isinstance(result, StreamingResponse)
    assert result.status_code == 200
    assert _CONTENT_CHUNK in received, (
        f"the client-visible bytes must be forwarded; got {received[:120]!r}"
    )
    assert record["dispatch_credits"] == []
    assert record["streaming_dispatch_tokens"] == []
    assert record.get("active") is False
    assert server.local_active_queries == 0
    await _checkpoint(server, observations, "settled")
    assert not observations.clamped, (
        "the marker must not disturb exactly-once counter accounting: "
        f"{observations.clamped}"
    )


# ---------------------------------------------------------------------------
# AC1 / AC2 — the synthesised error terminal chunk is a terminal chunk too
# ---------------------------------------------------------------------------


class _ErroringUpstream(_PacedUpstream):
    """A fake SSE upstream that raises a transport error instead of ending.

    Models the incident's local failure: the llama-server connection died
    (``ReadError``) after the watchdog evicted the slot, so the router's
    ``except Exception`` branch synthesises the client's terminal chunk.
    """

    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self.exc = exc

    async def aiter_bytes(self):
        self.entered.set()
        while True:
            self.parked.set()
            item = await self._queue.get()
            self.parked.clear()
            if item is None:
                raise self.exc
            yield item


SESSION_ERROR = "sess-stream-error-marking"
OTHER_SESSION = "sess-stream-error-other"


def _record_key(session: str):
    """Return the dispatch-record key for *session* (plain or per-endpoint)."""
    for key in server.local_dispatch_records:
        if isinstance(key, tuple) and len(key) == 2:
            if key[1] == session:
                return key
        elif key == session:
            return key
    raise AssertionError(
        f"no dispatch record for {session}: {list(server.local_dispatch_records)}"
    )


class _ErrorWindowObservation:
    """Snapshot of the record at the moment the error chunk reaches the client."""

    def __init__(self, record: dict, counter: int) -> None:
        self.streaming = list(record.get("streaming_dispatch_tokens") or [])
        self.credits = list(record.get("dispatch_credits") or [])
        self.active = record.get("active")
        self.counter = counter


async def _drive_observing_error_chunk(request, session: str):
    """Drain one streaming local dispatch, sampling its record at the error chunk."""
    response = await router.proxy_to_local(request, "v1/chat/completions")
    assert isinstance(response, StreamingResponse), type(response)
    received = b""
    observations: list[_ErrorWindowObservation] = []
    async for chunk in response.body_iterator:
        received += chunk
        if b"stream_exception" in chunk:
            key = _record_key(session)
            observations.append(
                _ErrorWindowObservation(
                    server.local_dispatch_records[key],
                    server.local_active_queries,
                )
            )
    return received, observations


def _install_error_scenario(monkeypatch) -> _ErroringUpstream:
    """Install the real lease bookkeeping with a pool of ``POOL`` slots."""
    helpers._reset_prefill_thrash_cooldowns()
    _install_server(monkeypatch)
    server.config["server"]["session_slot_pool_size"] = POOL
    server.config["server"]["local_max_concurrent_prefills"] = POOL
    upstream = _ErroringUpstream(_ReadError("llama-server went away"))
    sent: list[tuple[str, dict]] = []
    _patch_router(monkeypatch, {SESSION_ERROR: upstream}, sent)
    return upstream


class _ReadError(Exception):
    """Stand-in for the httpx transport error the incident logged."""


@pytest.mark.asyncio
async def test_stream_error_terminal_chunk_marks_finished_before_teardown(
    monkeypatch,
):
    """AC1: the synthesised error chunk is a terminal chunk and clears the marker.

    The client has its complete response (``finish_reason=error`` + ``[DONE]``)
    the moment this chunk is forwarded, so the dispatch is no longer streaming:
    the record's streaming token must already be gone while the credit is still
    held — exactly the finished-but-not-yet-released shape the routing gate
    keyed on for a clean finish.
    """
    upstream = _install_error_scenario(monkeypatch)
    request = _StreamRequest(SESSION_ERROR, SESSION_ERROR)

    task = asyncio.create_task(
        _drive_observing_error_chunk(request, SESSION_ERROR)
    )

    await _wait(upstream.parked, "first-byte park")
    record = server.local_dispatch_records[_record_key(SESSION_ERROR)]
    token = record["dispatch_token"]
    assert record["streaming_dispatch_tokens"] == [token]

    upstream.push(_CONTENT_CHUNK)
    await _wait(upstream.parked, "content chunk consumed")
    assert record["streaming_dispatch_tokens"] == [token], (
        "a non-terminal chunk must not mark the stream finished"
    )

    upstream.end()  # raises inside the upstream iterator
    received, observations = await asyncio.wait_for(task, timeout=10)

    assert b"stream_exception" in received, received[:200]
    assert len(observations) == 1, (
        "exactly one synthetic error chunk must be observed: "
        f"{[o.streaming for o in observations]}"
    )
    seen = observations[0]
    assert seen.streaming == [], (
        "the synthesised error terminal chunk must remove this dispatch's "
        f"streaming token; it still shows {seen.streaming}"
    )
    assert seen.credits == [token], (
        "marking must not release the credit — teardown owns that: "
        f"{seen.credits}"
    )
    assert seen.active is True
    assert seen.counter == 1

    # Teardown still releases exactly the one credit and settles the record.
    assert record["dispatch_credits"] == []
    assert record["streaming_dispatch_tokens"] == []
    assert record.get("active") is False
    assert server.local_active_queries == 0


@pytest.mark.asyncio
async def test_stream_error_finished_window_admits_the_next_turn(monkeypatch):
    """AC2: after the error the session's own record stops gating its retry.

    One *other* session occupies one of the two slots. With the own record
    excluded the pool reads 1/2 and a retry is admitted; before the marker was
    removed on this path it read 2/2 and the retry was exiled with
    ``local_concurrency_limit`` (the incident's 10-25 s tail).
    """
    upstream = _install_error_scenario(monkeypatch)
    other_request = SimpleNamespace()
    acquired, owner, _count, _retry = await helpers._try_acquire_local_dispatch(
        server, POOL, OTHER_SESSION, ENDPOINT, model_name=MODEL,
        request=other_request,
    )
    assert acquired is True, f"the other session must hold a real lease ({owner})"

    request = _StreamRequest(SESSION_ERROR, SESSION_ERROR)
    task = asyncio.create_task(
        _drive_observing_error_chunk(request, SESSION_ERROR)
    )

    await _wait(upstream.parked, "first-byte park")
    upstream.push(_CONTENT_CHUNK)
    await _wait(upstream.parked, "content chunk consumed")
    upstream.end()
    _received, observations = await asyncio.wait_for(task, timeout=10)

    seen = observations[0]
    assert seen.streaming == [], (
        "the error terminal chunk must clear the own streaming token: "
        f"{seen.streaming}"
    )
    cur, max_local = provider._get_local_concurrency_info(
        server.config, endpoint=ENDPOINT, session_id=SESSION_ERROR
    )
    assert max_local == POOL
    assert cur == 1, (
        "only the other session may occupy a slot once this session's stream "
        f"has errored terminally; got {cur}/{max_local} "
        f"(records={server.local_dispatch_records})"
    )
    assert cur < max_local, "the retry would still be refused local_concurrency_limit"

    # The other session's credit is untouched by this session's teardown.
    other = server.local_dispatch_records[_record_key(OTHER_SESSION)]
    assert other["active"] is True
    assert other["dispatch_credits"] == [other["dispatch_token"]]
    assert server.local_active_queries == 1
