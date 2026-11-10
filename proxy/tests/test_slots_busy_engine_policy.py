
"""Busy-engine ``/slots`` regression harness for ``/llama/local/status``.

LP-0MW3ED3XN003TPXI (child of LP-0MW10EZXX004M7BI).

Why this module exists
----------------------
The intake measured the defect on a live engine: while a local generation ran,
the model child's ``/slots`` endpoint did not answer inside either proxy poll
budget, so essentially every status poll degraded to a fossil snapshot
(``slots_stale=true``, ``slots_detail_stale=true``, ``null`` ETAs). Without a
deterministic reproduction every later fix is unverifiable and the "< 5 % of
polls stale" target cannot be measured in CI.

This module is the **single reusable busy-engine harness** for the whole
``/slots`` policy family. Later features (fossil-``generating`` reproduction,
ETA availability, authoritative-occupancy policy, documentation) assert against
these helpers instead of inventing a second stub mechanism:

- :class:`BusySlotsStub` — one configurable fake ``/slots`` upstream
  (``delay``, ``status``, ``payload``) patched onto the coalesced fetch choke
  point ``proxy.observability._do_slots_fetch``. It faithfully emulates
  ``asyncio.wait_for(client.get(url), timeout)``: it waits at most the caller's
  budget and raises :class:`TimeoutError` when the simulated latency exceeds it,
  so the *real* degradation path (timeout → last-known cache) is exercised.
- :func:`call_local_status` — drives ``GET /llama/local/status`` through the
  same FastAPI ``ASGITransport`` pattern the existing slots tests use, under a
  stub, with optional seeded last-known caches.

Reuse from a sibling test module (``proxy/tests`` is on ``sys.path`` via
``proxy/conftest.py``)::

    from test_slots_busy_engine_policy import BusySlotsStub, call_local_status

The two proxy poll budgets the stub must exceed to reproduce the defect are
:data:`COUNTS_BUDGET_SECONDS` (2.0 s, hard-coded in the status handler) and
:data:`DEFAULT_DETAIL_BUDGET_SECONDS` (1.0 s, ``STATUS_QUERY_TIMEOUT``).
"""

import asyncio
import contextlib
import os
import time
from unittest.mock import AsyncMock, patch

import httpx
import proxy.metrics as metrics
import pytest

# Documented proxy poll budgets. The stub latency must exceed BOTH to
# reproduce the measured busy-engine degradation.
COUNTS_BUDGET_SECONDS = 2.0
"""Counts-phase ``/slots`` budget, hard-coded in ``get_llama_local_status``."""

DEFAULT_DETAIL_BUDGET_SECONDS = 1.0
"""Detail-phase ``/slots`` budget (``STATUS_QUERY_TIMEOUT`` default)."""

# Extra headroom granted to the wall-clock assertion so ordinary CI scheduling
# jitter cannot make a correctly-bounded poll look unbounded.
_BUDGET_SLACK_SECONDS = 1.0

_DEFAULT_SEED_SLOTS = [
    {"slot_id": 0, "is_processing": False, "n_decoded": None},
    {"slot_id": 1, "is_processing": True, "n_decoded": 42},
]
"""Last-known per-slot detail seeded so the detail fallback has something to
serve (a cache-served list is what makes ``slots_detail_stale`` observable)."""

_DEFAULT_SEED_COUNTS = (1, 2)
"""Last-known ``(available, total)`` counts seeded so ``slot_counts_stale`` has
a last-known pair to serve after the fresh counts query fails."""


class BusySlotsStub:
    """A configurable fake ``/slots`` upstream shared by every policy test.

    Parameters
    ----------
    delay:
        Simulated upstream latency in seconds. Use a value greater than
        :data:`COUNTS_BUDGET_SECONDS` / :data:`DEFAULT_DETAIL_BUDGET_SECONDS` to
        reproduce the busy-engine timeout path; ``0.0`` for a healthy engine.
    status:
        HTTP status the request would carry if it returned before timing out.
        ``200`` with a ``payload`` models a healthy-but-slow engine; ``500``
        models the measured ``proxying request … 500`` shape.
    payload:
        Parsed JSON body returned when ``status == 200``. Entries use
        llama.cpp's raw ``/slots`` shape (``id``, ``is_processing``,
        ``next_token``).

    ``calls`` records every fetch as ``{url, timeout, delay}`` so a test can
    assert which budget each phase applied.
    """

    def __init__(self, *, delay: float = 0.0, status: int = 200, payload=None):
        self.delay = float(delay)
        self.status = int(status)
        self.payload = list(payload) if payload is not None else []
        self.calls: list[dict] = []

    async def fetch(self, url, timeout, client=None):
        """Stand-in for ``proxy.observability._do_slots_fetch``.

        Faithfully mirrors the real helper's ``asyncio.wait_for`` semantics:
        waits at most ``timeout`` seconds and only raises ``TimeoutError`` when
        the simulated latency exceeds that budget.
        """
        budget = float(timeout)
        self.calls.append({"url": url, "timeout": budget, "delay": self.delay})
        await asyncio.sleep(min(self.delay, budget))
        if self.delay > budget:
            raise TimeoutError(
                f"busy /slots latency {self.delay}s exceeded {budget}s budget"
            )
        if self.status != 200:
            return self.status, None
        return self.status, self.payload

    def patch(self):
        """Patch the coalesced fetch choke point used by both query helpers."""
        return patch("proxy.observability._do_slots_fetch", self.fetch)

    def budget_exceeding_calls(self) -> list[dict]:
        """Fetches whose simulated latency exceeded the budget applied to them."""
        return [c for c in self.calls if c["delay"] > c["timeout"]]


class _FakeLock:
    """Non-blocking async lock stand-in for the server's asyncio locks."""

    def locked(self):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass


@contextlib.contextmanager
def _status_endpoint(stub: BusySlotsStub, *, seed_counts, seed_slots, config, counts_query=None):
    """Context manager installing every patch a status poll needs.

    Mirrors the established slots tests' patch set (server state + the
    last-known caches) and adds the busy-engine stub at the fetch choke point.

    ``counts_query`` optionally replaces the counts phase (``_query_slots``) so
    a feature can isolate the *detail* phase (e.g. LP-0MW3ED73W007JYGI, which
    must prove the failure it records is the detail phase's, not the counts').
    """
    from proxy.server import app

    from proxy import server as srv

    async def fake_query():
        return {"llama_server_running": True}

    # A stale coalesced entry from an earlier test on the same URL would serve
    # the next poll from cache and bypass the stub; start each poll clean.
    from proxy.observability import reset_slots_poll_cache

    reset_slots_poll_cache()

    counts_cache = None
    if seed_counts is not None:
        counts_cache = (seed_counts[0], seed_counts[1], time.monotonic())

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch("proxy.server.query_llama_status", side_effect=fake_query)
        )
        stack.enter_context(patch.object(srv, "current_model", "test-model"))
        stack.enter_context(patch.object(srv, "model_switch_refcount", 0))
        stack.enter_context(patch.object(srv, "model_switch_lock", _FakeLock()))
        stack.enter_context(patch.object(srv, "background_loads", {}))
        stack.enter_context(patch.object(srv, "local_dispatch_records", {}))
        stack.enter_context(
            patch.object(srv, "local_dispatch_records_lock", _FakeLock())
        )
        stack.enter_context(
            patch.object(
                srv,
                "config",
                config
                if config is not None
                else {"server": {"llama_server_port": 8080}},
            )
        )
        stack.enter_context(
            patch("proxy.observability._last_slot_counts_cache", counts_cache)
        )
        stack.enter_context(
            patch(
                "proxy.observability._last_slot_details_cache",
                list(seed_slots) if seed_slots is not None else [],
            )
        )
        # A fresh-but-not-zero stamp so a degraded poll that serves the detail
        # cache publishes a real ``slots_snapshot_age_seconds`` (LP-0MW3ED73W007JYGI
        # asserts the provenance is visible), while staying inside the ceiling.
        stack.enter_context(
            patch("proxy.observability._last_slot_details_at", time.monotonic() - 2.0)
        )
        stack.enter_context(stub.patch())
        if counts_query is not None:
            stack.enter_context(
                patch("proxy.observability._query_slots", counts_query)
            )
        yield app


async def call_local_status(
    *,
    stub: BusySlotsStub | None = None,
    seed_counts=_DEFAULT_SEED_COUNTS,
    seed_slots=None,
    detail_timeout: float | None = None,
    config: dict | None = None,
    counts_query=None,
):
    """Drive ``GET /llama/local/status`` under a :class:`BusySlotsStub`.

    Args:
        stub: The fake ``/slots`` upstream. Defaults to a healthy, instant,
            empty-payload stub.
        seed_counts: Last-known ``(available, total)`` pair, or ``None`` to
            leave the counts cache empty.
        seed_slots: Last-known per-slot detail list, or ``None`` for the
            module default (:data:`_DEFAULT_SEED_SLOTS`).
        detail_timeout: Override ``STATUS_QUERY_TIMEOUT`` for the detail phase.
        config: Server config override.
        counts_query: Optional awaitable replacing the counts phase (a mock
            returning ``(available, total)``), used to isolate the detail phase.

    Returns:
        ``(payload, elapsed_seconds)`` where *payload* is the parsed JSON body.
    """
    stub = stub if stub is not None else BusySlotsStub(delay=0.0, payload=[])
    seed_slots = _DEFAULT_SEED_SLOTS if seed_slots is None else seed_slots

    with _status_endpoint(
        stub,
        seed_counts=seed_counts,
        seed_slots=seed_slots,
        config=config,
        counts_query=counts_query,
    ) as app:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as ac:
            env = (
                {"STATUS_QUERY_TIMEOUT": str(detail_timeout)}
                if detail_timeout is not None
                else {}
            )
            with patch.dict(os.environ, env):
                started = time.monotonic()
                resp = await ac.get("/llama/local/status")
                elapsed = time.monotonic() - started

    assert resp.status_code == 200, (
        f"/llama/local/status returned {resp.status_code} under a busy-engine "
        f"stub; a degraded poll must stay 200 (graceful degradation)"
    )
    return resp.json(), elapsed


# ======================================================================
# Baseline: the harness provably reproduces the measured degradation
# ======================================================================


class TestBusyEngineBaseline:
    """A slow ``/slots`` reproduces the intake's degraded status payload."""

    @pytest.mark.asyncio
    async def test_slow_slots_exceeding_budgets_reproduces_degradation(self):
        """AC1: latency beyond both poll budgets → the three stale flags.

        The stub's latency exceeds both documented budgets, so *both* the counts
        phase (2.0 s) and the detail phase (``STATUS_QUERY_TIMEOUT``, 1.0 s)
        time out. Each phase then serves its last-known cache, which is exactly
        the shape the intake measured on the live engine.
        """
        stub = BusySlotsStub(
            delay=COUNTS_BUDGET_SECONDS + 0.5,
            status=200,
            payload=[
                {"id": 0, "is_processing": False, "next_token": {"n_decoded": None}},
                {"id": 1, "is_processing": True, "next_token": {"n_decoded": 7}},
            ],
        )
        # Recorded baseline: the harness only reproduces the defect while the
        # stub's latency exceeds the real budgets. If either budget grows past
        # the stub delay, this test fails *here* naming the invariant instead
        # of silently passing on a poll that never timed out.
        assert stub.delay > COUNTS_BUDGET_SECONDS, (
            "busy-engine harness must exceed the counts budget "
            f"({COUNTS_BUDGET_SECONDS}s) or the counts phase cannot time out"
        )
        assert stub.delay > DEFAULT_DETAIL_BUDGET_SECONDS, (
            "busy-engine harness must exceed the detail budget "
            f"({DEFAULT_DETAIL_BUDGET_SECONDS}s) or the detail phase cannot time out"
        )

        payload, elapsed = await call_local_status(stub=stub)

        assert payload["slot_counts_stale"] is True, (
            "a timed-out counts phase with a fresh last-known cache must report "
            "slot_counts_stale=true"
        )
        assert payload["slots_detail_stale"] is True, (
            "a timed-out detail phase with a cached detail list must report "
            "slots_detail_stale=true"
        )
        assert payload["slots_stale"] is True, (
            "slots_stale is the union of both fallbacks; a degraded poll must "
            "never read fresh"
        )
        # Both phases actually hit the (too-slow) upstream, and both exceeded
        # the budget they were given.
        assert len(stub.budget_exceeding_calls()) >= 2, (
            "both the counts and the detail phase must have exceeded their "
            f"budgets; observed calls: {stub.calls}"
        )

    @pytest.mark.asyncio
    async def test_degraded_poll_returns_well_formed_payload_within_budget(self):
        """The degraded poll is well-formed and bounded by counts + detail."""
        stub = BusySlotsStub(delay=COUNTS_BUDGET_SECONDS + 0.5)
        payload, elapsed = await call_local_status(stub=stub)

        # Well-formed: the documented fields are present with their types.
        assert isinstance(payload["slots"], list), "slots must always be a list"
        assert isinstance(payload["slot_counts_stale"], bool)
        assert isinstance(payload["slots_detail_stale"], bool)
        assert isinstance(payload["slots_stale"], bool)
        assert isinstance(payload["status_schema_version"], int)
        assert payload["status_schema_version"] >= 1
        # The cached detail is still served so slot identity survives (the
        # capability the fossil-serving path deliberately preserves).
        assert len(payload["slots"]) == len(_DEFAULT_SEED_SLOTS)
        for slot in payload["slots"]:
            assert "slot_id" in slot
            assert "is_processing" in slot

        # Bounded wall clock: the endpoint may wait for counts + detail
        # serially, but never beyond that documented budget (+ CI slack).
        budget = COUNTS_BUDGET_SECONDS + DEFAULT_DETAIL_BUDGET_SECONDS
        assert elapsed < budget + _BUDGET_SLACK_SECONDS, (
            f"degraded poll took {elapsed:.2f}s, beyond the documented "
            f"{budget:.1f}s budget (+{_BUDGET_SLACK_SECONDS:.1f}s slack)"
        )

    @pytest.mark.asyncio
    async def test_healthy_fast_slots_does_not_degrade(self):
        """Control: a fast, healthy ``/slots`` must NOT report degraded.

        This proves the harness actually controls latency — the stale flags in
        the baseline test come from the slow stub, not from the fixture setup.
        """
        stub = BusySlotsStub(
            delay=0.0,
            status=200,
            payload=[
                {"id": 0, "is_processing": False, "next_token": {"n_decoded": None}},
                {"id": 1, "is_processing": True, "next_token": {"n_decoded": 5}},
            ],
        )
        payload, elapsed = await call_local_status(stub=stub)

        assert payload["slot_counts_stale"] is False
        assert payload["slots_detail_stale"] is False
        assert payload["slots_stale"] is False
        assert len(payload["slots"]) == 2
        # A healthy poll is not held for either poll budget.
        assert elapsed < DEFAULT_DETAIL_BUDGET_SECONDS, (
            f"a healthy poll should answer promptly, took {elapsed:.2f}s"
        )


# ======================================================================
# HTTP 500 variant: the measured `proxying request … 500` shape
# ======================================================================


class TestBusyEngineHttp500:
    """An immediate HTTP 500 degrades gracefully (the measured 36 % shape)."""

    @pytest.mark.asyncio
    async def test_http_500_serves_last_known_counts_and_detail(self):
        """A 500 on ``/slots`` still serves the last-known snapshot."""
        stub = BusySlotsStub(delay=0.0, status=500)
        payload, _ = await call_local_status(stub=stub)

        assert payload["slots_stale"] is True
        # Last-known pool size survives the failure; availability is withheld
        # because a stale list cannot prove capacity.
        assert payload["total_slots"] == _DEFAULT_SEED_COUNTS[1]
        assert payload["available_slots"] == 0
        assert payload["availability_unknown"] is True
        assert len(payload["slots"]) == len(_DEFAULT_SEED_SLOTS)


# ======================================================================
# Harness reusability: budgets are threaded through, not hard-coded in tests
# ======================================================================


class TestHarnessReuse:
    """The harness exposes one configurable stub for every later feature."""

    @pytest.mark.asyncio
    async def test_detail_budget_override_is_applied(self):
        """``detail_timeout`` is threaded into the detail phase's fetch."""
        stub = BusySlotsStub(delay=COUNTS_BUDGET_SECONDS + 0.5)
        _, _ = await call_local_status(stub=stub, detail_timeout=0.25)

        observed_budgets = [c["timeout"] for c in stub.calls]
        assert 0.25 in observed_budgets, (
            "the detail phase must have applied the overridden "
            f"STATUS_QUERY_TIMEOUT=0.25; observed budgets: {observed_budgets}"
        )


# ======================================================================
# Detail-phase failure reason observability (LP-0MW3ED73W007JYGI)
#
# The status detail path used to swallow a /slots failure into a debug log, so
# the measured ~36 % failure rate was invisible in the metrics surface and a
# timeout could not be told from an HTTP 500. These tests reuse the single
# busy-engine harness above (`BusySlotsStub`) rather than adding a second stub.
# ======================================================================


def _failure_count(reason: str) -> float:
    """Current value of ``llama_slots_query_failures_total{reason}``."""
    return metrics.llama_slots_query_failures_total.labels(reason=reason)._value.get()


class TestDetailPhaseFailureReason:
    """A status detail failure records a labelled ``timeout`` / ``http_<code>``."""

    @pytest.mark.asyncio
    async def test_http_500_detail_failure_records_http_500(self):
        """A 500 on the detail read records ``http_500`` and degrades gracefully."""
        # Counts succeeds (mocked) so the recorded failure is provably the
        # DETAIL phase's — otherwise the counts phase would record too and the
        # increment could not be attributed to the detail path.
        counts_query = AsyncMock(return_value=(1, 2))
        before = _failure_count("http_500")
        stub = BusySlotsStub(delay=0.0, status=500)
        payload, _ = await call_local_status(stub=stub, counts_query=counts_query)
        after = _failure_count("http_500")

        assert after > before, "a detail-phase HTTP 500 must increment http_500"
        # AC1: the graceful-degradation contract is unchanged.
        assert payload["slots_detail_stale"] is True
        assert payload["slots_stale"] is True
        assert len(payload["slots"]) == len(_DEFAULT_SEED_SLOTS)
        # AC3: provenance names the served snapshot and its age.
        assert payload["slots_snapshot_source"] == "snapshot"
        assert payload["slots_snapshot_age_seconds"] is not None

    @pytest.mark.asyncio
    async def test_detail_timeout_records_timeout_not_http_500(self):
        """A detail timeout records ``timeout`` (distinct from ``http_500``)."""
        counts_query = AsyncMock(return_value=(1, 2))
        before_timeout = _failure_count("timeout")
        before_500 = _failure_count("http_500")
        stub = BusySlotsStub(delay=COUNTS_BUDGET_SECONDS + 0.5)  # > detail budget
        payload, _ = await call_local_status(
            stub=stub, counts_query=counts_query, detail_timeout=0.2
        )
        after_timeout = _failure_count("timeout")
        after_500 = _failure_count("http_500")

        assert after_timeout > before_timeout, "a detail timeout must increment timeout"
        assert after_500 == before_500, "a timeout must NOT be labelled http_500"
        # Same graceful-degradation payload shape as the 500 case.
        assert payload["slots_detail_stale"] is True
        assert payload["slots_stale"] is True
        assert payload["slots_snapshot_source"] == "snapshot"

    @pytest.mark.asyncio
    async def test_query_slots_detail_fills_status_out_but_never_records(self):
        """The query helper publishes the outcome only via ``status_out``.

        Pins the *scope*: recording lives at the status call site (the handler),
        so a bare ``_query_slots_detail`` call never touches the metric — the
        general caller-tagged call-site instrumentation remains
        LP-0MVTDMXXH005D5D5's.
        """
        from proxy.observability import _query_slots_detail, reset_slots_poll_cache

        # 500 → http_500 surfaced through status_out.
        before_500 = _failure_count("http_500")
        out: dict = {}
        stub = BusySlotsStub(delay=0.0, status=500)
        with stub.patch():
            result = await _query_slots_detail(
                8080, timeout=1.0, model="Qwen3", status_out=out
            )
        assert result == []
        assert out.get("reason") == "http_500", out
        assert _failure_count("http_500") == before_500, (
            "_query_slots_detail must not record — only the status handler does"
        )

        # Timeout → timeout surfaced through status_out, still no recording.
        # A failed fetch is cached for the failure-backoff window, so clear the
        # coalesced cache or this call would replay the cached 500.
        reset_slots_poll_cache()
        before_timeout = _failure_count("timeout")
        out2: dict = {}
        stub2 = BusySlotsStub(delay=5.0)
        with stub2.patch():
            result2 = await _query_slots_detail(
                8080, timeout=0.1, model="Qwen3", status_out=out2
            )
        assert result2 == []
        assert out2.get("reason") == "timeout", out2
        assert _failure_count("timeout") == before_timeout


# ======================================================================
# ETA availability contract (LP-0MW3ED4Z30036L4E)
#
# Routing reads ``prefill_eta_seconds`` / ``queue_eta_seconds``, but the status
# builder nulls both whenever the detail phase is stale — so they are ``null``
# for the whole of a busy-engine generation. These tests pin the intended
# contract so LP-0MW3ED6K5000RT5R implements it without guessing:
#
#   * available (a live/derived estimate) when the detail is fresh;
#   * otherwise the **last measured** value with an explicit ``<field>_stale``
#     marker and ``<field>_age_seconds`` — never a silent ``null``;
#   * with no measurement ever, both stay ``null`` (never a fabricated ``0``);
#   * the markers/ages are additive; existing fields keep their names/meaning.
#
# The repo runs a green suite, so the not-yet-satisfied case is pinned with
# ``xfail(strict=True)``: it fails today (the ETA is silently nulled on stale
# detail) and turns into an unexpected pass — forcing the xfail mark off — once
# the contract lands.
# ======================================================================

#: A ``status_estimate``-shaped result with a live ETA available.
_ETA_AVAILABLE = {
    "prefill_eta_seconds": 12.5,
    "queue_eta_seconds": 25.0,
    "queue_eta_occupancy_seconds": 13.0,
    "queue_eta_busy_slots": 1,
    "prefill_eta_kind": "reference_cold_bound",
    "prefill_eta_assumed_cached_ratio": 0.0,
    "prefill_eta_reference_tokens": 32000,
    "prefill_eta_samples": 5,
    "prefill_eta_calibrated": True,
    "prefill_eta_rate_bucket": "solitary",
    "prefill_eta_contended_samples": 2,
    "prefill_budget_seconds": 60.0,
}

#: A ``status_estimate``-shaped result where no measurement exists at all.
_ETA_UNAVAILABLE = {
    "prefill_eta_seconds": None,
    "queue_eta_seconds": None,
    "queue_eta_occupancy_seconds": None,
    "queue_eta_busy_slots": None,
    "prefill_eta_kind": "reference_cold_bound",
    "prefill_eta_assumed_cached_ratio": 0.0,
    "prefill_eta_reference_tokens": None,
    "prefill_eta_samples": 0,
    "prefill_eta_calibrated": False,
    "prefill_eta_rate_bucket": "solitary",
    "prefill_eta_contended_samples": 0,
    "prefill_budget_seconds": None,
}


def _eta_patch(estimate):
    return patch("proxy.prefill_eta.status_estimate", return_value=estimate)


def _stale_detail_kwargs():
    """Drive a poll whose detail read times out (stale detail served)."""
    return dict(
        stub=BusySlotsStub(delay=5.0),
        counts_query=AsyncMock(return_value=(1, 2)),
        detail_timeout=0.2,
    )


class TestEtaAvailabilityContract:
    """The ETA is available, or last-measured with a stale marker — never null."""

    @pytest.mark.asyncio
    async def test_fresh_detail_publishes_live_eta(self):
        """A fresh detail poll publishes the live/derived ETA and its labelling."""
        stub = BusySlotsStub(
            delay=0.0,
            status=200,
            payload=[
                {"id": 0, "is_processing": False, "next_token": {"n_decoded": None}},
                {"id": 1, "is_processing": True, "next_token": {"n_decoded": 5}},
            ],
        )
        with _eta_patch(_ETA_AVAILABLE):
            payload, _ = await call_local_status(stub=stub)

        assert payload["prefill_eta_seconds"] == 12.5
        assert payload["queue_eta_seconds"] == 25.0
        # Additive labelling is untouched and present.
        assert payload["prefill_eta_kind"] == "reference_cold_bound"
        assert payload["prefill_eta_assumed_cached_ratio"] == 0.0
        # No stale marker on a fresh poll (additive, so ``not True`` is enough).
        assert payload.get("prefill_eta_stale") is not True
        assert payload.get("queue_eta_stale") is not True

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "LP-0MW3ED4Z30036L4E red-first (AC1/AC2): today the handler nulls "
            "prefill_eta_seconds and queue_eta_seconds unconditionally when the "
            "detail is stale, so the last measured ETA vanishes for the whole "
            "generation. LP-0MW3ED6K5000RT5R must publish the last measured "
            "value with a stale marker + age — remove the xfail when it lands."
        ),
    )
    async def test_stale_detail_publishes_last_measured_eta_with_marker(self):
        """After a measurement, a stale poll republishes it with a marker + age."""
        # Poll 1: fresh detail → a live measurement is recorded.
        fresh_stub = BusySlotsStub(
            delay=0.0,
            status=200,
            payload=[
                {"id": 0, "is_processing": False, "next_token": {"n_decoded": None}},
                {"id": 1, "is_processing": True, "next_token": {"n_decoded": 5}},
            ],
        )
        with _eta_patch(_ETA_AVAILABLE):
            fresh, _ = await call_local_status(stub=fresh_stub)
            assert fresh["prefill_eta_seconds"] == 12.5  # measurement recorded
            # Poll 2: detail times out → a stale snapshot is served.
            stale, _ = await call_local_status(**_stale_detail_kwargs())

        assert stale["slots_detail_stale"] is True
        # The last measured value survives, with an explicit marker and age.
        assert stale["prefill_eta_seconds"] == 12.5, (
            "the last measured prefill ETA must not be silently nulled on a "
            "stale poll"
        )
        assert stale.get("prefill_eta_stale") is True
        age = stale.get("prefill_eta_age_seconds")
        assert isinstance(age, (int, float)) and age >= 0, age
        # Same contract for the queue ETA.
        assert stale["queue_eta_seconds"] == 25.0, (
            "the last measured queue ETA must not be silently nulled on a "
            "stale poll"
        )
        assert stale.get("queue_eta_stale") is True
        assert stale.get("queue_eta_age_seconds") is not None

    @pytest.mark.asyncio
    async def test_no_measurement_ever_stays_null(self):
        """With no measurement recorded, both ETAs stay null (never a 0)."""
        with _eta_patch(_ETA_UNAVAILABLE):
            payload, _ = await call_local_status(**_stale_detail_kwargs())

        assert payload["slots_detail_stale"] is True
        assert payload["prefill_eta_seconds"] is None
        assert payload["queue_eta_seconds"] is None
        # Never a fabricated zero.
        assert payload["prefill_eta_seconds"] != 0
        assert payload["queue_eta_seconds"] != 0
