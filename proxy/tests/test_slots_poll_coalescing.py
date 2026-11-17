"""
Hermetic tests for /slots poll coalescing (LP-0MUCEFD5B003TURT).

Context
-------
Every herdr client polls ``/llama/local/status``, and each poll queries
llama-server ``/slots``. Under N concurrent pollers that is N upstream
requests per interval, amplifying load on an already-saturated backend.
A short-TTL cache plus single-flight coalescing bounds upstream /slots
volume by the TTL instead of by N; failures are cached for a backoff
window and callers fall back to last-known counts with the stale flag.

These tests verify hermetically (mock HTTP client, no live llama-server).
"""

import asyncio
import contextlib
import gc
import logging
import time

import httpx
import pytest

from proxy import observability as obs


class _Resp:
    def __init__(self, status_code: int, data):
        self.status_code = status_code
        self._data = data
        self.text = "[]"

    async def json(self):
        return self._data


class _CountingClient:
    """Mock httpx client that counts upstream /slots requests."""

    def __init__(self, status_code: int = 200, data=None, delay: float = 0.0):
        self.status_code = status_code
        self.data = data if data is not None else [{"id": 0, "is_processing": False}]
        self.delay = delay
        self.calls: list[str] = []

    async def get(self, url):
        self.calls.append(url)
        if self.delay:
            await asyncio.sleep(self.delay)
        return _Resp(self.status_code, self.data)


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    obs.reset_slots_poll_cache()
    # The single-flight registry is process-global too: a future left behind
    # by one test (e.g. the orphan-injection test below) must not make a
    # later test a follower of a dead leader.
    obs._slots_poll_inflight.clear()
    monkeypatch.delenv("SLOTS_POLL_TTL_SECONDS", raising=False)
    monkeypatch.delenv("SLOTS_POLL_FAILURE_BACKOFF_SECONDS", raising=False)
    yield
    obs.reset_slots_poll_cache()
    obs._slots_poll_inflight.clear()
    obs.reset_slots_snapshot_state()


# ═══════════════════════════════════════════════════════════════════════════════
# Coalescing / TTL
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_concurrent_polls_share_one_upstream_request():
    """AC: N concurrent pollers produce one upstream /slots request."""
    client = _CountingClient(delay=0.05)
    url = "http://localhost:8080/slots?model=m"

    results = await asyncio.gather(
        *[obs._fetch_slots_payload(url, 2.0, client) for _ in range(10)]
    )

    assert len(client.calls) == 1, (
        f"10 concurrent pollers must coalesce to 1 upstream request, got {len(client.calls)}"
    )
    assert all(status == 200 for status, _ in results)


@pytest.mark.asyncio
async def test_second_poll_within_ttl_is_cached(monkeypatch):
    """A repeat poll within the TTL does not touch the backend."""
    monkeypatch.setenv("SLOTS_POLL_TTL_SECONDS", "30")
    client = _CountingClient()
    url = "http://localhost:8080/slots"

    await obs._fetch_slots_payload(url, 2.0, client)
    await obs._fetch_slots_payload(url, 2.0, client)
    await obs._fetch_slots_payload(url, 2.0, client)

    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_poll_after_ttl_refetches(monkeypatch):
    """Once the TTL elapses, the next poll hits the backend again."""
    monkeypatch.setenv("SLOTS_POLL_TTL_SECONDS", "0")
    client = _CountingClient()
    url = "http://localhost:8080/slots"

    await obs._fetch_slots_payload(url, 2.0, client)
    await obs._fetch_slots_payload(url, 2.0, client)

    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_distinct_urls_do_not_share_cache():
    """Different models/endpoints have independent cache entries."""
    client = _CountingClient()
    await obs._fetch_slots_payload("http://localhost:8080/slots?model=a", 2.0, client)
    await obs._fetch_slots_payload("http://localhost:8080/slots?model=b", 2.0, client)
    assert len(client.calls) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Failure backoff
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_failure_is_cached_for_backoff_window(monkeypatch):
    """A failed /slots fetch is cached for the backoff window."""
    monkeypatch.setenv("SLOTS_POLL_TTL_SECONDS", "0")
    monkeypatch.setenv("SLOTS_POLL_FAILURE_BACKOFF_SECONDS", "30")
    client = _CountingClient(status_code=500)
    url = "http://localhost:8080/slots"

    status1, _ = await obs._fetch_slots_payload(url, 2.0, client)
    status2, _ = await obs._fetch_slots_payload(url, 2.0, client)

    assert status1 == 500 and status2 == 500
    assert len(client.calls) == 1, "Failure must be cached for the backoff window"


@pytest.mark.asyncio
async def test_success_not_held_by_failure_backoff(monkeypatch):
    """A success uses the short TTL, not the longer failure backoff."""
    monkeypatch.setenv("SLOTS_POLL_TTL_SECONDS", "0")
    monkeypatch.setenv("SLOTS_POLL_FAILURE_BACKOFF_SECONDS", "30")
    client = _CountingClient(status_code=200)
    url = "http://localhost:8080/slots"

    await obs._fetch_slots_payload(url, 2.0, client)
    await obs._fetch_slots_payload(url, 2.0, client)
    assert len(client.calls) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# _query_slots routes through the coalescer
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_query_slots_coalesces_repeat_calls(monkeypatch):
    """The status-path ``_query_slots`` benefits from the coalescer."""
    monkeypatch.setenv("SLOTS_POLL_TTL_SECONDS", "30")
    client = _CountingClient(
        data=[
            {"id": 0, "is_processing": False},
            {"id": 1, "is_processing": True},
        ]
    )

    available1, total1 = await obs._query_slots(client, 8080, model="m")
    available2, total2 = await obs._query_slots(client, 8080, model="m")

    assert (available1, total1) == (1, 2)
    assert (available2, total2) == (1, 2)
    assert len(client.calls) == 1, "Repeat status polls within TTL must coalesce"


@pytest.mark.asyncio
async def test_query_slots_updates_last_known_counts(monkeypatch):
    """A successful coalesced query still updates the last-known counts."""
    monkeypatch.setenv("SLOTS_POLL_TTL_SECONDS", "30")
    client = _CountingClient(
        data=[{"id": 0, "is_processing": False}, {"id": 1, "is_processing": False}]
    )
    await obs._query_slots(client, 8080, model="m")
    assert obs.last_known_slot_counts() == (2, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Leader cancellation / orphaned single-flight future (LP-0MVJML600003CQ6Y)
#
# 2026-10-22 "Web UI is not responding": a ``/slots`` single-flight LEADER is
# cancelled mid-fetch (the browser tab closes, Starlette cancels the SSE
# handler while it awaits the 5 s ``/slots`` fetch). ``CancelledError`` is a
# ``BaseException``, so the helper's ``except Exception`` never resolves the
# shared future — yet the ``finally`` block removes the registry entry. Every
# follower already awaiting ``await fut`` then waits forever, and because
# ``ui.status_events()`` awaits that before its keepalive loop, the stream
# yields zero bytes and the UI freezes silently.
#
# The eight tests above are happy-path / TTL / backoff only: neither
# cancellation path (leader cancelled, leader vanished) is exercised, which is
# what these three cancellation regression tests pin (the fourth test here
# covers the unobserved-failure warning, LP-0MVK6QUAQ002KFHU).
#
# These tests were written failing-first under ``xfail(strict=True)`` on
# LP-0MVK1YCVV007HNAG (pre-fix RED output recorded there).
# LP-0MVK1YDKB0009HTM resolved/cancelled the shared future on leader
# cancellation and bounded the follower wait, and removed the markers so each
# assertion is now enforced (RED-before / GREEN-after evidence is recorded on
# that item).
#
# Leak-warning assertions and the ``asyncio`` logger (LP-0MVK6QUAQ002KFHU)
# ---------------------------------------------------------------------------
# "Future exception was never retrieved" is emitted from ``Future.__del__``,
# not when the future settles, so it lands in whichever test happens to run the
# cyclic ``gc.collect()`` that reclaims the traceback cycle. Because the
# ``asyncio`` logger is process-global, a warning caused by an *earlier* test
# would otherwise fail a leak assertion here (the 7f7785db flake: the warning
# came from ``test_slots_500_fix``'s stubbed ConnectionError). Two rules follow:
#
#   1. Collect *before* opening a leak-warning window (``gc.collect()`` then
#      ``caplog.clear()``) so the window covers only objects the test creates.
#   2. Production code must not leave an unobserved future exception behind at
#      all — a leader's failure is marked as retrieved by its owner, since the
#      usual case is a lone poller that nothing follows
#      (``_consume_slots_inflight_outcome``).
# ═══════════════════════════════════════════════════════════════════════════════

_URL = "http://localhost:8080/slots?model=m"

# In-test deadlines. They are generous relative to the follower's own caller
# timeout (0.5 s): a slow host cannot flake the bounded-wait assertion, while a
# genuinely unbounded ``await fut`` (the defect) still fails well inside them.
_CANCEL_RELEASE_DEADLINE = 5.0    # AC1: followers released inside 5 s
_ORPHAN_RECOVERY_DEADLINE = 6.0   # AC2: bounded follower wait + fresh fetch

_LEAK_WARNING_FRAGMENTS = ("never retrieved", "destroyed but it is pending")


class _StallingClient(_CountingClient):
    """Client whose upstream ``get()`` stalls until the caller is cancelled.

    Models the real trigger: llama-server answers ``/slots`` in ~23 s while the
    local model is generating, so the SSE handler is cancelled with the fetch
    still in flight.
    """

    def __init__(self, delay: float = 30.0):
        super().__init__(delay=delay)
        self.started = asyncio.Event()

    async def get(self, url):
        self.calls.append(url)
        self.started.set()
        await asyncio.sleep(self.delay)
        return _Resp(self.status_code, self.data)


async def _wait_until_fetching(client: _StallingClient, timeout: float = 2.0) -> None:
    """Wait until the leader has registered the future and issued the fetch."""
    try:
        await asyncio.wait_for(client.started.wait(), timeout=timeout)
    except TimeoutError:  # pragma: no cover - broken harness only
        raise AssertionError("leader never issued an upstream /slots request")


def _harvest(task: "asyncio.Task") -> None:
    """Retrieve a finished task's outcome so asyncio logs no late warning."""
    if task.cancelled():
        return
    with contextlib.suppress(BaseException):
        task.result()


async def _bounded_wait(tasks, deadline: float):
    """Wait up to *deadline*; return ``(finished, still_pending)`` after reaping."""
    done, pending = await asyncio.wait(set(tasks), timeout=deadline)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    for task in tasks:
        _harvest(task)
    return done, pending


@pytest.mark.asyncio
async def test_cancelled_leader_releases_followers():
    """AC1: cancelling the owner of the fetch must release its followers.

    The leader is cancelled while two followers already await the shared
    future; both must be released inside a bounded in-test deadline instead of
    hanging on a future nobody can resolve, and the registry must be clean
    afterwards.
    """
    client = _StallingClient(delay=30.0)
    leader = asyncio.create_task(obs._fetch_slots_payload(_URL, 30.0, client))
    followers: list[asyncio.Task] = []
    try:
        await _wait_until_fetching(client)
        followers = [
            asyncio.create_task(obs._fetch_slots_payload(_URL, 0.5, client))
            for _ in range(2)
        ]
        # Let both followers reach ``await fut`` before the leader dies.
        for _ in range(20):
            await asyncio.sleep(0)

        leader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader

        _done, pending = await _bounded_wait(followers, _CANCEL_RELEASE_DEADLINE)
        assert not pending, (
            f"{len(pending)} follower(s) were still awaiting the orphaned single-flight "
            f"future {_CANCEL_RELEASE_DEADLINE}s after the leader was cancelled; a "
            "cancelled leader must resolve or cancel the shared future (AC1)"
        )
        assert obs._slots_poll_inflight.get(_URL) is None, (
            "the single-flight registry entry must not leak after the leader is cancelled"
        )
    finally:
        if not leader.done():
            leader.cancel()
            with contextlib.suppress(BaseException):
                await leader
        if followers:
            await _bounded_wait(followers, 0.0)
        obs._slots_poll_inflight.pop(_URL, None)


@pytest.mark.asyncio
async def test_follower_recovers_when_leader_disappears_without_resolving(monkeypatch):
    """AC2: a follower wait is bounded even when the leader simply vanishes.

    A pending future with no live owner (the pre-fix outcome of a cancelled
    leader, injected directly here) must not hold a follower forever: with no
    cached payload to fall back on, the follower has to re-fetch as the new
    leader, which shows up as a second upstream ``/slots`` call.
    """
    monkeypatch.setenv("SLOTS_POLL_TTL_SECONDS", "0")
    client = _CountingClient()
    orphan = asyncio.get_running_loop().create_future()
    obs._slots_poll_inflight[_URL] = orphan
    follower = asyncio.create_task(obs._fetch_slots_payload(_URL, 0.5, client))
    try:
        _done, pending = await _bounded_wait([follower], _ORPHAN_RECOVERY_DEADLINE)
        assert not pending, (
            f"follower waited more than {_ORPHAN_RECOVERY_DEADLINE}s on a single-flight "
            "future whose leader disappeared; the follower wait must be bounded and fall "
            "back to a fresh fetch (or cached values) within its own timeout (AC2)"
        )
        status, _data = follower.result()
        assert status == 200, (
            f"follower fallback must produce the fresh payload (status 200), got {status}"
        )
        assert client.calls, (
            "no cached payload exists, so the follower must re-fetch as the new leader "
            "(a second upstream /slots call is expected)"
        )
    finally:
        if not follower.done():
            follower.cancel()
        _harvest(follower)
        obs._slots_poll_inflight.pop(_URL, None)
        if not orphan.done():
            orphan.cancel()


@pytest.mark.asyncio
async def test_cancelled_leader_leaves_no_pending_future_or_leak_warnings(caplog):
    """AC3: cancel-owning request must not leak a pending future or warnings.

    Asserts the shared future is resolved/cancelled (not left pending with no
    owner), that no task is leaked, and that finalising the future does not
    emit "Future exception was never retrieved" / "Task was destroyed but it is
    pending" on the ``asyncio`` logger.
    """
    loop = asyncio.get_running_loop()
    previous_debug = loop.get_debug()
    loop.set_debug(True)
    # The ``asyncio`` logger is process-global: "Future exception was never
    # retrieved" is emitted from ``Future.__del__``, i.e. whenever the cyclic
    # GC happens to reclaim the future's traceback cycle. Collect first, then
    # drop those records, so this test's verdict covers only warnings emitted
    # for objects it creates itself rather than leftovers from any earlier test
    # in the same process (LP-0MVK6QUAQ002KFHU).
    gc.collect()
    caplog.clear()
    caplog.set_level(logging.ERROR, logger="asyncio")
    baseline_tasks = asyncio.all_tasks()
    client = _StallingClient(delay=30.0)
    leader = asyncio.create_task(obs._fetch_slots_payload(_URL, 30.0, client))
    shared = None
    try:
        await _wait_until_fetching(client)
        shared = obs._slots_poll_inflight.get(_URL)
        assert shared is not None, (
            "the leader must register the shared future before fetching"
        )

        leader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader
        await asyncio.sleep(0)  # let the helper's cancellation path settle

        assert shared.done(), (
            "cancelling the leader must resolve or cancel the shared future; leaving it "
            "pending with no owner is the follower-hang root cause (AC3)"
        )

        # Drop every reference and finalise the future now: an unretrieved
        # exception (or a destroyed pending task) only logs at GC / loop close,
        # where it would surface as post-test noise instead of a test verdict.
        obs._slots_poll_inflight.pop(_URL, None)
        shared = None
        gc.collect()
        await asyncio.sleep(0)
        gc.collect()

        leaked = [
            task
            for task in asyncio.all_tasks()
            if task not in baseline_tasks
            and task is not asyncio.current_task()
            and not task.done()
        ]
        assert not leaked, f"cancelling the leader leaked pending task(s): {leaked!r}"

        noisy = [
            record.getMessage()
            for record in caplog.records
            if any(fragment in record.getMessage() for fragment in _LEAK_WARNING_FRAGMENTS)
        ]
        assert not noisy, f"cancelling the leader emitted asyncio leak warnings: {noisy}"
    finally:
        if not leader.done():
            leader.cancel()
            with contextlib.suppress(BaseException):
                await leader
        obs._slots_poll_inflight.pop(_URL, None)
        if shared is not None and not shared.done():
            shared.cancel()
        loop.set_debug(previous_debug)


@pytest.mark.asyncio
async def test_leader_failure_without_follower_logs_no_never_retrieved(caplog):
    """A failed leader with no follower must not log a late asyncio warning.

    LP-0MVK6QUAQ002KFHU: the shared future is a broadcast channel that is
    *usually* unobserved — the common case is a single poller whose fetch
    fails (ConnectionError on a restarting llama-server). Setting the failure
    on a future nobody retrieves makes ``Future.__del__`` log "Future exception
    was never retrieved" whenever the GC later reclaims the traceback cycle,
    which is both production log noise and a cross-test flake source: that
    message lands in whichever *later* test happens to run ``gc.collect()``.

    The leader re-raises to its own caller (unchanged), while the shared
    future's outage is marked as retrieved by its owner.
    """

    class _RefusingClient:
        async def get(self, url):
            raise ConnectionError("connection refused")

    # Collect first so warnings from earlier tests land outside this window.
    gc.collect()
    caplog.clear()
    caplog.set_level(logging.ERROR, logger="asyncio")

    with pytest.raises(ConnectionError):
        await obs._fetch_slots_payload(_URL, 2.0, _RefusingClient())

    # The failure's traceback holds a reference cycle back to the shared
    # future, so only a cyclic collection finalises it — exactly the delayed
    # window in which the warning used to surface.
    gc.collect()
    await asyncio.sleep(0)
    gc.collect()

    noisy = [
        record.getMessage()
        for record in caplog.records
        if any(fragment in record.getMessage() for fragment in _LEAK_WARNING_FRAGMENTS)
    ]
    assert not noisy, (
        f"a failed /slots leader left an unretrieved future exception: {noisy}"
    )
    assert obs._slots_poll_inflight.get(_URL) is None, (
        "the single-flight registry entry must not leak after a failed leader "
        "with no followers"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Status-path fan-out (LP-0MVVXGHHB0073W1B AC2 / F3 AC1)
#
# Every herdr client polls ``GET /llama/local/status``; on the pre-fix path each
# poll performs its OWN counts query (2 s budget) and its OWN detail query
# (``STATUS_QUERY_TIMEOUT``) in series. When llama-server answers slower than
# the per-query budgets, every bounded-await expiry promotes the follower to a
# fresh upstream ``/slots`` leader — measured: 5 concurrent polls issue 5
# upstream fetches. The contract the shared-snapshot read path (F3) implements:
# N concurrent status polls issued inside the freshness window share exactly
# one upstream ``/slots`` request. The count is taken at
# ``obs._do_slots_fetch`` — the single primitive every coalesced upstream fetch
# reaches (``_run_slots_fetch_as_leader``) — so the exact-count assertion
# measures real upstream amplification, not which client object performed it.
# ═══════════════════════════════════════════════════════════════════════════════

_SLOTS_PAYLOAD = [
    {"id": 0, "is_processing": False, "next_token": None, "n_past": 0},
    {
        "id": 1,
        "is_processing": True,
        "next_token": {"n_decoded": 7},
        "n_past": 128,
    },
]


class _FanoutBackend:
    """Raw /slots fetch stub that answers AFTER the per-query budgets.

    llama-server answers ``/slots`` in ~3 s under load — after the counts
    (2 s) and detail (``STATUS_QUERY_TIMEOUT``, 1 s) budgets expire — which is
    exactly the condition that makes the pre-fix follower-promotion cascade
    amplify one poll cycle into one upstream fetch per poll. The stub records
    every upstream attempt so the test can count them.
    """

    def __init__(self):
        self.calls: list[str] = []

    async def __call__(self, url: str, timeout: float, client=None):
        self.calls.append(url)
        await asyncio.sleep(3.0)
        return (200, _SLOTS_PAYLOAD)


class _FakeLock:
    def locked(self):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass


@pytest.mark.asyncio
async def test_status_path_fanout_exactly_one_upstream_fetch_for_5_concurrent_polls():
    """AC2: 5 concurrent status polls inside the freshness window issue exactly
    one upstream /slots request (exact call count, not a ``<=`` bound).
    """
    from unittest.mock import patch as _patch

    from proxy.server import app

    from proxy import server

    backend = _FanoutBackend()

    async def fake_query():
        return {"llama_server_running": True}

    transport = httpx.ASGITransport(app=app)
    with _patch("proxy.server.query_llama_status", side_effect=fake_query):
        with _patch.object(server, "current_model", "test-model"):
            with _patch.object(server, "model_switch_refcount", 0):
                with _patch.object(server, "model_switch_lock", _FakeLock()):
                    with _patch.object(server, "background_loads", {}):
                        with _patch.object(server, "local_dispatch_records", {}):
                            with _patch.object(
                                server, "local_dispatch_records_lock", _FakeLock()
                            ):
                                with _patch.object(
                                    server,
                                    "config",
                                    {"server": {"llama_server_port": 8080}},
                                ):
                                    with _patch(
                                        "proxy.observability._do_slots_fetch", backend
                                    ):
                                        async with httpx.AsyncClient(
                                            transport=transport, base_url="http://test"
                                        ) as ac:
                                            responses = await asyncio.gather(
                                                *[
                                                    ac.get("/llama/local/status")
                                                    for _ in range(5)
                                                ]
                                            )

    assert [r.status_code for r in responses] == [200] * 5
    assert all(
        "/slots" in url for url in backend.calls
    ), f"upstream calls must target /slots: {backend.calls}"
    assert len(backend.calls) == 1, (
        f"5 concurrent status polls inside the freshness window must issue exactly "
        f"one upstream /slots request, got {len(backend.calls)}: {backend.calls}"
    )
    # The shared fetch outlives every poll's budget by design (that is what
    # stops the polls from promoting a second fetch); reap it so no task leaks
    # into the next test.
    await obs.acancel_slots_snapshot_tasks()


# ═══════════════════════════════════════════════════════════════════════════════
# Shared snapshot read path (LP-0MVVXGLTM008ROJ9)
#
# One poll ⇒ one snapshot ⇒ one bounded upstream fetch. The counts pair and the
# per-slot detail are derived from that single payload (AC3), the poll reuses
# the snapshot while it is within the coalescer's freshness window (AC1), and a
# poll whose budget expires degrades without cancelling the shared fetch, which
# lands for later polls instead of extending this one (AC2).
# ═══════════════════════════════════════════════════════════════════════════════


class _ScriptedBackend:
    """Raw ``/slots`` fetch stub: records attempts, answers from a script."""

    def __init__(self, status: int = 200, payload=None, delay: float = 0.0, error=None):
        self.status = status
        self.payload = payload if payload is not None else list(_SLOTS_PAYLOAD)
        self.delay = delay
        self.error = error
        self.calls: list[str] = []

    async def __call__(self, url: str, timeout: float, client=None):
        self.calls.append(url)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.status, self.payload


def _with_default_slots_model(monkeypatch):
    """Pin the model id so the shared fetch URL is deterministic."""
    monkeypatch.setattr(obs, "_resolve_slots_model_id", lambda model: model or "m")


@pytest.mark.asyncio
async def test_read_scope_derives_counts_and_detail_from_one_fetch(monkeypatch):
    """AC1/AC3: one poll's counts and detail share one upstream fetch."""
    _with_default_slots_model(monkeypatch)
    backend = _ScriptedBackend()
    monkeypatch.setattr(obs, "_do_slots_fetch", backend)

    with obs.slots_read_scope(budget_seconds=1.0):
        available, total = await obs._query_slots(None, 8080, model="m", timeout=1.0)
        details = await obs._query_slots_detail(8080, timeout=1.0, model="m")

    assert len(backend.calls) == 1, (
        "a poll's counts and detail reads must share ONE upstream /slots fetch, "
        f"got {len(backend.calls)}"
    )
    assert (available, total) == (1, 2)
    assert [slot["is_processing"] for slot in details] == [False, True]
    # AC3: the summary is counted from the SAME list that is served.
    assert available == sum(1 for slot in details if not slot["is_processing"])


@pytest.mark.asyncio
async def test_read_scope_dedupes_a_failed_attempt_across_phases(monkeypatch):
    """AC1: a failed fetch is not retried by the detail phase of the same poll."""
    _with_default_slots_model(monkeypatch)
    backend = _ScriptedBackend(error=TimeoutError("backend hung"))
    monkeypatch.setattr(obs, "_do_slots_fetch", backend)
    counts_out: dict = {}
    detail_out: dict = {}

    with obs.slots_read_scope(budget_seconds=1.0):
        counts = await obs._query_slots(
            None, 8080, model="m", timeout=1.0, status_out=counts_out
        )
        details = await obs._query_slots_detail(
            8080, timeout=1.0, model="m", status_out=detail_out
        )

    assert len(backend.calls) == 1, (
        "the detail phase must reuse the counts phase's failed attempt instead "
        f"of issuing a second fetch, got {len(backend.calls)}"
    )
    assert counts == (0, 0)
    assert details == []
    assert counts_out.get("reason") == "timeout"
    assert detail_out.get("reason") == "timeout"
    # The detail read reused the shared attempt, so the handler must not record
    # the same failure a second time.
    assert detail_out.get("shared_read") is True


@pytest.mark.asyncio
async def test_pending_snapshot_degrades_without_cancelling_the_shared_fetch(monkeypatch):
    """AC2: a fetch slower than the poll budget neither blocks nor dies."""
    _with_default_slots_model(monkeypatch)
    backend = _ScriptedBackend(delay=0.3)
    monkeypatch.setattr(obs, "_do_slots_fetch", backend)

    started = time.monotonic()
    with obs.slots_read_scope(budget_seconds=0.05):
        counts = await obs._query_slots(None, 8080, model="m", timeout=5.0)
        details = await obs._query_slots_detail(8080, timeout=5.0, model="m")
    elapsed = time.monotonic() - started

    assert len(backend.calls) == 1
    assert counts == (0, 0), "a pending fetch asserts no counts"
    assert details == [], "a pending fetch serves no detail"
    assert elapsed < 0.25, (
        f"the poll must degrade at its budget instead of waiting out the fetch "
        f"(took {elapsed:.2f}s)"
    )
    # The shared fetch survived the poll's budget.
    assert obs._slots_snapshot_tasks, (
        "the shared fetch must keep running after the poll's budget expires"
    )

    # ... and its payload is available to the next poll from the coalesced
    # cache (the fetch lands at ~0.3 s; the default TTL then serves it).
    await asyncio.sleep(0.4)
    available, total = await obs._query_slots(None, 8080, model="m", timeout=1.0)
    assert (available, total) == (1, 2)
    assert len(backend.calls) == 1, (
        "the completed shared fetch must serve the next poll from the cache, "
        f"not trigger another fetch (got {len(backend.calls)})"
    )


def test_derive_slots_from_payload_is_the_single_source_of_counts():
    """AC3: counts and detail come from one derivation, never two readings."""
    payload = [
        {"id": 0, "is_processing": False, "next_token": {"n_decoded": 3}, "n_past": 12},
        {"id": 1, "is_processing": True, "next_token": None, "id_task": 7},
        # No ``is_processing`` field at all: the counts and the detail must
        # still agree on the same interpretation (not processing).
        {"id": 2},
    ]
    available, total, slots = obs.derive_slots_from_payload(payload)

    assert total == 3
    assert [slot["slot_id"] for slot in slots] == [0, 1, 2]
    assert [slot["is_processing"] for slot in slots] == [False, True, False]
    assert available == sum(1 for slot in slots if not slot["is_processing"]) == 2
    assert slots[0]["n_decoded"] == 3
    assert slots[0]["context_tokens"] == 12
    assert slots[1]["id_task"] == 7
    # A malformed payload is rejected rather than half-derived.
    assert obs.derive_slots_from_payload("not-a-list") is None
    assert obs.derive_slots_from_payload([1, 2]) is None
    assert obs.derive_slots_from_payload([]) == (0, 0, ())
