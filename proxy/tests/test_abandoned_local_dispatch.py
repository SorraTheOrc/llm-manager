"""Abandoned local dispatches must not read as live saturated work.

Regression harness for LP-0MVRMI5HK00558TB AC1 (observable behaviour: an
abandoned local dispatch stops being reported as *live* saturated work, and is
never folded into ``idle`` either) and AC3 (deterministic coverage — mocked
llama-server, no network, no real sleeps).

Parent work item: LP-0MVRMI5HK00558TB — *Abandoned local dispatches leave
llama-server busy and start a 600 s prefill_thrash_cooldown*.
Implementation item that turns the defect assertions green:
LP-0MVT2KSN3007QYFK (abandoned-slot registry + upstream cancel + status
annotation).

Measured defect (LP-0MVOG7IP80003MLV, 2026-10-24 20:00 → 2026-10-25 07:00):
332 local dispatches were abandoned by the proxy while llama-server kept
computing them, so 247 fresh ``/slots`` polls reported
``capacity_hint=saturated`` with ``local_in_flight_requests=0``
(``slots_detail_stale=false``) — work nobody was waiting for, truthfully
reported from a ``/slots`` snapshot that cannot tell who abandoned the slot.

Assertion map (parent AC → test class):

- **AC1 — abandoned work is not live saturated work**
  - ``TestAbandonedHintContract`` — the ``capacity_hint`` decision table
    extension: abandoned-only occupancy is a distinct hint, never
    ``saturated``, never ``idle``, for **every** abandonment reason in
    ``{prefill_deadline, session_evicted, no_progress_watchdog,
    client_disconnect}``.
  - ``TestAbandonedSlotRegistry`` — the attributable slot identity (endpoint,
    slot id, session, reason) with a bounded TTL.
  - ``TestStatusAnnotation`` — the per-slot annotation and its attribution.
  - ``TestStatusPayload`` — end-to-end ``/llama/local/status``: the ghost poll
    the measurement counted, and the additive top-level count.
- **AC1 truthfulness control** (LP-0MVKSXR4P001V0CM must not weaken):
  ``TestTruthfulnessControls`` — genuinely live, proxy-owned decode work still
  reports ``saturated``; an idle or free pool still reports ``idle``; a live
  lease holder alongside abandoned work still reports ``saturated``.

**Red-first encoding (noted deviation).** The defect assertions below are
marked ``xfail(strict=True)`` so this test-only item can land on ``dev`` with a
green suite: the assertion is present and pinned, and the paired
implementation item removes the marker as it turns each one green (``strict``
means an unexpectedly passing test fails the suite, so the marker cannot be
forgotten). Control cases run unmarked and pass on today's code and after the
fix.

Harness idioms are reused from ``test_llama_local_status.py`` (the
``httpx.ASGITransport`` status harness and slot-snapshot builders) and
``test_prefill_thrash_wedge.py`` (injected monotonic clock) rather than
inventing a second one.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

#: Every abandonment reason the parent AC names. The fix must cover all four;
#: a reason missing from this tuple is a coverage hole, not a pass.
ABANDONMENT_REASONS = (
    "prefill_deadline",
    "session_evicted",
    "no_progress_watchdog",
    "client_disconnect",
)

REASON_ITEM = "LP-0MVT2KSN3007QYFK"

"""Note on the red-first markers.

The defect assertions were pinned with ``pytest.mark.xfail(strict=True)`` by
the prerequisite test item (LP-0MVT2KS1R006P8E2) so a test-only change could
land on ``dev`` with a green suite. The paired implementation
(LP-0MVT2KSN3007QYFK) removed every marker as it turned the assertions green —
these tests now run unmarked, which is the only state that proves the fix.
"""


# ---------------------------------------------------------------------------
# Injected clock (mirrors test_prefill_thrash_cooldown.py / _wedge.py)
# ---------------------------------------------------------------------------


class _InjectedClock:
    """Monotonic clock that only advances when the test drives it."""

    def __init__(self) -> None:
        self._now = 0.0

    @property
    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class _TimeShim:
    """Replace ``router_helpers.time`` so only ``monotonic`` is injected."""

    def __init__(self, clock: _InjectedClock) -> None:
        self._clock = clock

    def monotonic(self) -> float:
        return self._clock.now

    def __getattr__(self, name: str):
        return getattr(time, name)


@pytest.fixture
def clock() -> _InjectedClock:
    return _InjectedClock()


@pytest.fixture(autouse=True)
def _isolate_abandoned_state(monkeypatch, clock: _InjectedClock):
    """Isolate the abandoned-slot registry and the injected clock."""
    from proxy import router_helpers as helpers

    monkeypatch.setattr(helpers, "time", _TimeShim(clock))
    reset = getattr(helpers, "_reset_abandoned_slots", None)
    if reset is not None:
        reset()
    yield
    if reset is not None:
        reset()


# ---------------------------------------------------------------------------
# Snapshot builders
# ---------------------------------------------------------------------------


def _raw_slot(
    slot_id: int = 0,
    *,
    is_processing: bool = True,
    n_decoded: int | None = None,
    context_tokens: int | None = None,
) -> dict:
    """A raw llama-server ``/slots`` entry (pre-annotation shape)."""
    return {
        "slot_id": slot_id,
        "is_processing": is_processing,
        "n_decoded": n_decoded,
        "context_tokens": context_tokens,
    }


def _annotated_slot(
    slot_id: int = 0,
    *,
    lease_state: str = "generating",
    phase: str | None = "decode",
    owner_session_id: str | None = None,
    abandoned: bool | None = None,
    abandonment_reason: str | None = None,
    abandoned_session_id: str | None = None,
) -> dict:
    """A slot in the shape ``annotate_slots_for_status`` emits (post-AC2)."""
    slot = {
        "slot_id": slot_id,
        "is_processing": lease_state == "generating",
        "n_decoded": 32 if phase == "decode" else None,
        "context_tokens": 4096 if phase == "prefill" else None,
        "owner_session_id": owner_session_id,
        "lease_state": lease_state,
        "phase": phase,
        "prefill_eta_seconds": None,
    }
    if abandoned is not None:
        slot["abandoned"] = abandoned
        slot["abandonment_reason"] = abandonment_reason
        slot["abandoned_session_id"] = abandoned_session_id
    return slot


def _ghost_slot(
    *, slot_id: int = 0, reason: str = "prefill_deadline", phase: str = "decode"
) -> dict:
    """The measured defect in one slot: proxy says nobody is waiting, the
    llama-server snapshot says the slot is still processing."""
    return _annotated_slot(
        slot_id,
        phase=phase,
        owner_session_id=None,  # no live dispatch record → nobody is waiting
        abandoned=True,
        abandonment_reason=reason,
        abandoned_session_id="herdr-ghost-session",
    )


# ═════════════════════════════════════════════════════════════════════════════
# AC1 — capacity_hint decision-table extension
# ═════════════════════════════════════════════════════════════════════════════


class TestAbandonedHintContract:
    """AC1: abandoned-only occupancy gets its own hint value."""

    def test_abandoned_busy_hint_value_exists(self):
        """The new value is an additive extension of the decision table."""
        from proxy.observability import CAPACITY_ABANDONED_BUSY

        assert CAPACITY_ABANDONED_BUSY == "abandoned_busy"

    @pytest.mark.parametrize("reason", ABANDONMENT_REASONS)
    def test_abandoned_decode_pool_is_not_saturated(self, reason):
        """AC1: for every abandonment reason, a pool whose only work is
        abandoned is neither ``saturated`` (today's defect) nor ``idle``."""
        from proxy.observability import (
            CAPACITY_ABANDONED_BUSY,
            CAPACITY_IDLE,
            CAPACITY_SATURATED,
            compute_capacity_hint,
        )

        hint = compute_capacity_hint([_ghost_slot(reason=reason)])

        assert hint == CAPACITY_ABANDONED_BUSY, (
            f"abandoned ({reason}) work must be reported as abandoned_busy, "
            f"not {hint}"
        )
        assert hint != CAPACITY_SATURATED
        assert hint != CAPACITY_IDLE

    def test_abandoned_prefill_pool_is_abandoned_busy(self):
        """A ghost still prefilling is busy-but-not-ours, not prefill_bound
        (nothing live is prefilling) and never ``idle``."""
        from proxy.observability import (
            CAPACITY_ABANDONED_BUSY,
            CAPACITY_IDLE,
            compute_capacity_hint,
        )

        hint = compute_capacity_hint([_ghost_slot(phase="prefill")])

        assert hint == CAPACITY_ABANDONED_BUSY
        assert hint != CAPACITY_IDLE

    def test_whole_pool_abandoned_is_abandoned_busy(self):
        """Both slots of a 2-slot pool abandoned → one abandoned_busy verdict."""
        from proxy.observability import CAPACITY_ABANDONED_BUSY, compute_capacity_hint

        hint = compute_capacity_hint(
            [
                _ghost_slot(slot_id=0, reason="prefill_deadline"),
                _ghost_slot(slot_id=1, reason="session_evicted"),
            ]
        )

        assert hint == CAPACITY_ABANDONED_BUSY


# ═════════════════════════════════════════════════════════════════════════════
# AC1 truthfulness control (LP-0MVKSXR4P001V0CM must not weaken)
# ═════════════════════════════════════════════════════════════════════════════


class TestTruthfulnessControls:
    """The pre-existing decision table must keep its meanings."""

    def test_live_decode_pool_is_still_saturated(self):
        """A genuinely busy GPU (live, proxy-owned decode) is ``saturated``."""
        from proxy.observability import CAPACITY_SATURATED, compute_capacity_hint

        hint = compute_capacity_hint(
            [_annotated_slot(0, owner_session_id="live-session")]
        )

        assert hint == CAPACITY_SATURATED

    def test_free_slot_is_still_idle(self):
        """A pool with a free slot is still ``idle``."""
        from proxy.observability import CAPACITY_IDLE, compute_capacity_hint

        assert compute_capacity_hint([_annotated_slot(0, lease_state="free", phase=None)]) == CAPACITY_IDLE

    def test_only_idle_leases_is_still_idle(self):
        """Reserved-but-idle leases are not work: still ``idle``."""
        from proxy.observability import CAPACITY_IDLE, compute_capacity_hint

        hint = compute_capacity_hint(
            [
                _annotated_slot(0, lease_state="idle_leased", phase=None),
                _annotated_slot(1, lease_state="idle_leased", phase=None),
            ]
        )

        assert hint == CAPACITY_IDLE

    def test_empty_snapshot_is_still_idle(self):
        """No evidence must never be read as saturation."""
        from proxy.observability import CAPACITY_IDLE, compute_capacity_hint

        assert compute_capacity_hint([]) == CAPACITY_IDLE

    def test_live_prefill_pool_is_still_prefill_bound(self):
        """A live prefill with no free slot stays ``prefill_bound``."""
        from proxy.observability import CAPACITY_PREFILL_BOUND, compute_capacity_hint

        hint = compute_capacity_hint(
            [
                _annotated_slot(
                    0,
                    phase="prefill",
                    owner_session_id="live-session",
                )
            ]
        )

        assert hint == CAPACITY_PREFILL_BOUND

    def test_live_work_alongside_abandoned_work_stays_saturated(self):
        """Abandoned work never discounts live work: a live decode slot
        alongside a ghost still reports the pool as working."""
        from proxy.observability import CAPACITY_SATURATED, compute_capacity_hint

        hint = compute_capacity_hint(
            [
                _ghost_slot(slot_id=0),
                _annotated_slot(1, owner_session_id="live-session"),
            ]
        )

        assert hint == CAPACITY_SATURATED

    def test_unannotated_slots_keep_todays_verdict(self):
        """Slots without an ``abandoned`` annotation behave exactly as today."""
        from proxy.observability import CAPACITY_SATURATED, compute_capacity_hint

        hint = compute_capacity_hint([_annotated_slot(0, owner_session_id="live")])

        assert hint == CAPACITY_SATURATED


# ═════════════════════════════════════════════════════════════════════════════
# AC1 — abandoned-slot registry (attributable identity + bounded TTL)
# ═════════════════════════════════════════════════════════════════════════════


class TestAbandonedSlotRegistry:
    """AC1: the abandoned slot identity must be recorded and attributable."""

    def test_reasons_constant_covers_every_abandonment_reason(self):
        from proxy.router_helpers import ABANDONED_SLOT_REASONS

        assert set(ABANDONMENT_REASONS) <= set(ABANDONED_SLOT_REASONS)

    @pytest.mark.parametrize("reason", ABANDONMENT_REASONS)
    def test_recorded_entry_is_attributable(self, reason):
        """The record carries endpoint, slot id, session and reason — never a
        bare boolean."""
        from proxy.router_helpers import _abandoned_slot_for, _record_abandoned_slot

        _record_abandoned_slot(
            2,
            session_id="herdr-abc",
            endpoint="local-llama",
            reason=reason,
            now=0.0,
        )

        entry = _abandoned_slot_for(2, endpoint="local-llama", now=0.0)

        assert entry is not None
        assert entry["slot_id"] == 2
        assert entry["endpoint"] == "local-llama"
        assert entry["session_id"] == "herdr-abc"
        assert entry["reason"] == reason
        assert entry["abandoned_at"] == 0.0
        assert entry["ttl_seconds"] > 0

    def test_entry_is_unknown_endpoint_fallback(self):
        """A legacy single-backend entry (no endpoint) still matches a lookup
        with an explicit endpoint, so the annotation cannot silently vanish
        when endpoint resolution differs between write and read."""
        from proxy.router_helpers import _abandoned_slot_for, _record_abandoned_slot

        _record_abandoned_slot(
            1, session_id="s", endpoint=None, reason="client_disconnect", now=0.0
        )

        entry = _abandoned_slot_for(1, endpoint="local-llama", now=0.0)

        assert entry is not None
        assert entry["slot_id"] == 1

    def test_entry_expires_after_the_bounded_ttl(self, clock):
        """The registry is bounded: an entry stops matching after its TTL so a
        recycled slot id cannot stay 'abandoned' forever."""
        from proxy.router_helpers import _abandoned_slot_for, _record_abandoned_slot

        entry = _record_abandoned_slot(
            0, session_id="s", reason="prefill_deadline", now=clock.now
        )
        ttl = entry["ttl_seconds"]

        clock.advance(ttl - 1)
        assert _abandoned_slot_for(0, now=clock.now) is not None

        clock.advance(2)
        assert _abandoned_slot_for(0, now=clock.now) is None

    def test_prune_reports_and_drops_only_expired_entries(self, clock):
        from proxy.router_helpers import (
            _abandoned_slot_for,
            _prune_abandoned_slots,
            _record_abandoned_slot,
        )

        stale = _record_abandoned_slot(
            0, session_id="a", reason="prefill_deadline", now=clock.now
        )
        _record_abandoned_slot(
            1, session_id="b", reason="session_evicted", now=clock.now
        )

        clock.advance(stale["ttl_seconds"] + 1)
        # Re-arm slot 1 so it is fresh at prune time.
        _record_abandoned_slot(
            1, session_id="b", reason="session_evicted", now=clock.now
        )

        assert _prune_abandoned_slots(now=clock.now) == 1
        assert _abandoned_slot_for(0, now=clock.now) is None
        assert _abandoned_slot_for(1, now=clock.now) is not None

    def test_reset_clears_the_registry(self):
        from proxy.router_helpers import (
            _abandoned_slot_for,
            _record_abandoned_slot,
            _reset_abandoned_slots,
        )

        _record_abandoned_slot(0, session_id="a", reason="prefill_deadline", now=0.0)
        _reset_abandoned_slots()

        assert _abandoned_slot_for(0, now=0.0) is None

    def test_rerecording_the_same_slot_refreshes_the_entry(self, clock):
        """A slot abandoned twice (a re-dispatch that also died) keeps a fresh
        window instead of expiring from the first event."""
        from proxy.router_helpers import _abandoned_slot_for, _record_abandoned_slot

        first = _record_abandoned_slot(
            0, session_id="a", reason="prefill_deadline", now=clock.now
        )
        clock.advance(first["ttl_seconds"] - 1)
        _record_abandoned_slot(
            0, session_id="a", reason="no_progress_watchdog", now=clock.now
        )

        clock.advance(2)
        entry = _abandoned_slot_for(0, now=clock.now)
        assert entry is not None
        assert entry["reason"] == "no_progress_watchdog"


# ═════════════════════════════════════════════════════════════════════════════
# AC1 — per-slot status annotation
# ═════════════════════════════════════════════════════════════════════════════


def _annotate(
    slots: list[dict], srv=None, *, model: str = "Qwen3", stale: bool = False
) -> list[dict]:
    from proxy.observability import annotate_slots_for_status

    return annotate_slots_for_status(slots, srv, model, config={}, stale=stale)


class TestStatusAnnotation:
    """AC1: the status payload annotates the ghost, with attribution."""

    @pytest.mark.parametrize("reason", ABANDONMENT_REASONS)
    def test_slot_is_annotated_with_reason_and_identity(self, reason):
        from proxy.router_helpers import _record_abandoned_slot

        _record_abandoned_slot(
            1,
            session_id="herdr-ghost",
            endpoint=None,
            reason=reason,
            now=0.0,
        )

        annotated = _annotate([_raw_slot(1, is_processing=True, n_decoded=12)])

        assert len(annotated) == 1
        slot = annotated[0]
        assert slot["abandoned"] is True
        assert slot["abandonment_reason"] == reason
        assert slot["slot_id"] == 1  # attributable identity, not a bare flag
        assert slot["abandoned_session_id"] == "herdr-ghost"

    def test_recency_ordering_prefers_abandoned_over_a_stale_binding(self):
        """An abandoned slot with no live lease must not be reported as
        ``free``/``idle``: ``lease_state`` stays ``generating`` because the
        backend is genuinely processing."""
        from proxy.router_helpers import _record_abandoned_slot

        _record_abandoned_slot(0, session_id="s", reason="prefill_deadline", now=0.0)

        slot = _annotate([_raw_slot(0, is_processing=True, n_decoded=3)])[0]

        assert slot["lease_state"] == "generating"
        assert slot["abandoned"] is True


class TestAnnotationControls:
    """Control: annotation is additive and never guesses."""

    def test_live_slot_is_not_annotated_abandoned(self):
        """Control: a slot with no registry entry never claims abandonment."""
        slot = _annotate([_raw_slot(0, is_processing=True, n_decoded=3)])[0]

        assert slot.get("abandoned") is not True
        assert not slot.get("abandonment_reason")

    def test_base_fields_are_preserved(self):
        """Control: the pre-existing per-slot fields are unchanged."""
        slot = _annotate([_raw_slot(2, is_processing=False)])[0]

        assert {
            "slot_id",
            "is_processing",
            "owner_session_id",
            "lease_state",
            "phase",
        } <= set(slot)
        assert slot["slot_id"] == 2


# ═════════════════════════════════════════════════════════════════════════════
# AC1 — end-to-end /llama/local/status payload
# ═════════════════════════════════════════════════════════════════════════════


class _StatusHarness:
    """Issue ``/llama/local/status`` with a mocked llama-server (no network).

    Mirrors ``test_llama_local_status._CapacityStatusHarness``: the fresh
    ``/slots`` counts query returns ``(0, 0)`` (as in the measured window's
    ghost polls) while the per-slot detail query returns the supplied
    snapshot.
    """

    async def get(self, slots_detail, *, dispatch_records=None, config=None):
        import httpx
        from proxy.server import app

        from proxy import server

        async def _fake_query():
            return {"llama_server_running": True}

        async def _counts(*_args, **_kwargs):
            return (0, 0)

        async def _detail(*_args, **_kwargs):
            return slots_detail

        resolved_config = config or {
            "server": {
                "llama_server_port": 8080,
                "session_slot_pool_size": 2,
            }
        }
        with patch("proxy.server.query_llama_status", side_effect=_fake_query):
            with patch.object(server, "active_queries", 0):
                with patch.object(server, "active_queries_lock", asyncio.Lock()):
                    with patch.object(server, "local_active_queries", 0):
                        with patch.object(
                            server, "local_active_queries_lock", asyncio.Lock()
                        ):
                            with patch.object(
                                server, "current_model", "Qwen3-local"
                            ):
                                with patch.object(server, "model_switch_refcount", 0):
                                    with patch.object(
                                        server, "model_switch_lock", _UnlockedLock()
                                    ):
                                        with patch.object(
                                            server, "background_loads", {}
                                        ):
                                            with patch.object(
                                                server,
                                                "local_dispatch_records",
                                                dispatch_records or {},
                                            ):
                                                with patch.object(
                                                    server,
                                                    "local_dispatch_records_lock",
                                                    _UnlockedLock(),
                                                ):
                                                    with patch.object(
                                                        server, "config", resolved_config
                                                    ):
                                                        with patch(
                                                            "proxy.observability._query_slots_detail",
                                                            _detail,
                                                        ):
                                                            with patch(
                                                                "proxy.observability._query_slots",
                                                                _counts,
                                                            ):
                                                                with patch.dict(
                                                                    "proxy.session._slot_owners",
                                                                    {},
                                                                    clear=True,
                                                                ):
                                                                    transport = (
                                                                        httpx.ASGITransport(
                                                                            app=app
                                                                        )
                                                                    )
                                                                    async with httpx.AsyncClient(
                                                                        transport=transport,
                                                                        base_url="http://test",
                                                                    ) as ac:
                                                                        resp = await ac.get(
                                                                            "/llama/local/status"
                                                                        )
        assert resp.status_code == 200
        return resp.json()


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


class TestStatusPayload:
    """AC1: the poll the measurement counted must stop saying ``saturated``."""

    @pytest.mark.parametrize("reason", ABANDONMENT_REASONS)
    def test_ghost_poll_is_not_saturated(self, reason):
        """Baseline defect (2.12 % of polls): fresh detail, no live dispatch,
        slot ``is_processing`` → ``saturated`` with
        ``local_in_flight_requests=0``. After the fix the same poll reports
        ``abandoned_busy`` and an attributable per-slot annotation."""
        from proxy.observability import CAPACITY_ABANDONED_BUSY
        from proxy.router_helpers import _record_abandoned_slot

        _record_abandoned_slot(
            0,
            session_id="herdr-ghost",
            endpoint=None,
            reason=reason,
            now=time.monotonic(),
        )

        payload = asyncio.run(
            _StatusHarness().get([_raw_slot(0, is_processing=True, n_decoded=17)])
        )

        assert payload["local_in_flight_requests"] == 0
        assert payload["slots_detail_stale"] is False
        assert payload["capacity_hint"] == CAPACITY_ABANDONED_BUSY, (
            "abandoned work must not be reported as live saturated work: "
            f"capacity_hint={payload['capacity_hint']!r}"
        )
        assert payload["local_abandoned_slots"] == 1
        slot = payload["slots"][0]
        assert slot["abandoned"] is True
        assert slot["abandonment_reason"] == reason
        assert slot["slot_id"] == 0

    def test_live_occupancy_still_reports_saturated(self):
        """Control: a pool genuinely busy with live, proxy-owned decode work
        is still ``saturated`` and none of its slots is annotated abandoned."""
        import time as _time

        dispatch_records = {
            "live-session": {
                "active": True,
                "expires_at": _time.monotonic() + 60.0,
                "first_byte_seen": True,
            }
        }
        from proxy.session import _slot_registry_key

        with patch.dict(
            "proxy.session._slot_owners",
            {_slot_registry_key(None, 0): "live-session"},
            clear=True,
        ):
            payload = asyncio.run(
                _StatusHarness().get(
                    [_raw_slot(0, is_processing=True, n_decoded=99)],
                    dispatch_records=dispatch_records,
                )
            )

        assert payload["capacity_hint"] == "saturated"
        assert payload["local_in_flight_requests"] == 1
        assert payload["slots"][0].get("abandoned") is not True

    def test_live_occupancy_is_not_counted_abandoned(self):
        """AC1: the additive top-level count only counts abandoned slots."""
        import time as _time

        from proxy.session import _slot_registry_key

        dispatch_records = {
            "live-session": {
                "active": True,
                "expires_at": _time.monotonic() + 60.0,
                "first_byte_seen": True,
            }
        }
        with patch.dict(
            "proxy.session._slot_owners",
            {_slot_registry_key(None, 0): "live-session"},
            clear=True,
        ):
            payload = asyncio.run(
                _StatusHarness().get(
                    [_raw_slot(0, is_processing=True, n_decoded=99)],
                    dispatch_records=dispatch_records,
                )
            )

        assert payload["local_abandoned_slots"] == 0


# ═════════════════════════════════════════════════════════════════════════════
# AC1 — upstream cancel + abandoned-slot recording on every abandonment path
#
# "Upstream cancel is applied on every path that holds a handle" and "a
# bounded, read-only investigation decides whether a real abort is achievable
# on this llama-server build". The investigation (2026-10-28 llama-server log,
# 82 ``Connection handling canceled`` events) found the abort IS achievable and
# observable during prefill: 42 of the 82 cancelled a task that never reached
# ``prompt processing done``, because ``return_progress`` makes llama-server
# write to the socket throughout the prefill and it acts on the closed socket
# at that write. So every path that owns a handle must close it.
# ═════════════════════════════════════════════════════════════════════════════


def _fake_stream_cm() -> AsyncMock:  # noqa: F821 - annotation is documentation
    """A stand-in upstream stream whose ``__aexit__`` is observable."""
    from unittest.mock import AsyncMock

    cm = AsyncMock()
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


def _server_stub(records: dict | None = None, config: dict | None = None):
    """Minimal server stub with the dispatch-record state the releases touch."""
    import asyncio as _asyncio
    from unittest.mock import MagicMock

    return SimpleNamespace(
        config=config
        or {
            "server": {
                "session_slot_pool_size": 2,
                "local_dispatch_lease_timeout_seconds": 60,
                "local_dispatch_no_progress_timeout_seconds": 180,
                "local_dispatch_max_prefill_seconds": 900,
            }
        },
        local_active_queries=0,
        local_active_queries_lock=_asyncio.Lock(),
        local_dispatch_records=records if records is not None else {},
        local_dispatch_records_lock=_asyncio.Lock(),
        local_prefill_in_flight={},
        local_prefill_in_flight_lock=_asyncio.Lock(),
        logger=MagicMock(),
    )


def _bind_slot(session_id: str, slot_id: int) -> None:
    """Bind *session_id* to a llama-server slot in the ownership registry."""
    from proxy.session import _slot_registry_key

    with patch.dict(
        "proxy.session._slot_owners",
        {_slot_registry_key(None, slot_id): session_id},
        clear=True,
    ):
        pass  # patch.dict restores on exit, so callers re-enter it themselves


class TestUpstreamCancel:
    """AC1: the deadline path cancels the real upstream task."""

    def test_deadline_release_closes_the_passed_stream(self, clock):
        """``prefill_deadline`` gained the mechanism the watchdog already had:
        the captured upstream stream is closed, aborting the prefill."""
        from proxy.router_helpers import _release_prefill_deadline_dispatch

        srv = _server_stub(
            records={
                "herdr-ghost": {
                    "active": True,
                    "expires_at": time.monotonic() + 60.0,
                    "first_byte_seen": False,
                }
            }
        )
        srv.local_prefill_in_flight = {"herdr-ghost": True}
        stream_cm = _fake_stream_cm()

        asyncio.run(
            _release_prefill_deadline_dispatch(
                srv,
                "herdr-ghost",
                abandoned_slot_id=1,
                upstream_stream_cm=stream_cm,
            )
        )

        stream_cm.__aexit__.assert_awaited_once()
        from proxy.router_helpers import _abandoned_slot_for

        entry = _abandoned_slot_for(1, now=clock.now)
        assert entry is not None
        assert entry["reason"] == "prefill_deadline"
        assert entry["session_id"] == "herdr-ghost"

    def test_deadline_release_closes_the_records_pinned_stream(self, clock):
        """When the caller cannot pass a handle, the record's pinned
        ``stream_cm`` (set after the SSE headers) is closed instead."""
        from proxy.router_helpers import _release_prefill_deadline_dispatch

        stream_cm = _fake_stream_cm()
        srv = _server_stub(
            records={
                "herdr-ghost": {
                    "active": True,
                    "expires_at": time.monotonic() + 60.0,
                    "stream_cm": stream_cm,
                }
            }
        )

        asyncio.run(_release_prefill_deadline_dispatch(srv, "herdr-ghost"))

        stream_cm.__aexit__.assert_awaited_once()

    def test_measured_cache_abort_release_records_the_abandoned_slot(self, clock):
        """The shared core (measured cache-deficit early abort) also records
        the ghost, so the annotation covers every abandonment, not just the
        four reasons the parent AC names."""
        from proxy.router_helpers import (
            _abandoned_slot_for,
            _release_measured_cache_abort_dispatch,
        )

        srv = _server_stub(
            records={
                "sess-abort": {
                    "active": True,
                    "expires_at": time.monotonic() + 60.0,
                }
            }
        )
        with patch.dict(
            "proxy.session._slot_owners",
            {("", 0): "sess-abort"},
            clear=True,
        ):
            asyncio.run(_release_measured_cache_abort_dispatch(srv, "sess-abort"))

        entry = _abandoned_slot_for(0, now=clock.now)
        assert entry is not None
        assert entry["session_id"] == "sess-abort"


class TestOtherAbandonmentPaths:
    """AC1: client disconnect, watchdog and session eviction are covered too."""

    def test_client_disconnect_records_the_abandoned_slot(self, clock):
        from proxy.router_helpers import (
            _abandoned_slot_for,
            _release_client_disconnect_dispatch,
        )

        stream_cm = _fake_stream_cm()
        srv = _server_stub(
            records={
                "sess-gone": {
                    "active": True,
                    "expires_at": time.monotonic() + 60.0,
                    "stream_cm": stream_cm,
                }
            }
        )
        with patch.dict(
            "proxy.session._slot_owners",
            {("", 0): "sess-gone"},
            clear=True,
        ):
            released = _release_client_disconnect_dispatch(srv, "sess-gone")

        assert released is True
        entry = _abandoned_slot_for(0, now=clock.now)
        assert entry is not None
        assert entry["reason"] == "client_disconnect"
        assert entry["session_id"] == "sess-gone"

    def test_no_progress_watchdog_records_the_abandoned_slot(self, monkeypatch, clock):
        from proxy.router_helpers import _abandoned_slot_for

        from proxy import router_helpers as helpers

        # The autouse isolation fixture injects the clock into
        # ``router_helpers.time``, so the watchdog's "now" and the record's
        # timestamps must come from the same clock domain.
        now = clock.now
        stream_cm = _fake_stream_cm()
        srv = _server_stub(
            records={
                "sess-wedged": {
                    "backend": "local",
                    "started_at": now - 300.0,
                    "active": True,
                    "expires_at": now + 600.0,
                    "last_progress": 0,
                    "last_progress_ts": now - 300.0,
                    "stream_cm": stream_cm,
                }
            }
        )
        srv.local_prefill_in_flight = {"sess-wedged": True}
        srv.local_active_queries = 1

        # /slots unobservable: the watchdog falls back to the no-progress clock.
        async def _no_slots(*_a, **_k):
            return None

        monkeypatch.setattr(
            helpers, "_query_slots_progress", _no_slots, raising=False
        )

        with patch.dict(
            "proxy.session._slot_owners",
            {("", 1): "sess-wedged"},
            clear=True,
        ):
            asyncio.run(helpers._cleanup_stale_local_dispatch(srv))

        stream_cm.__aexit__.assert_awaited_once()
        entry = _abandoned_slot_for(1, now=clock.now)
        assert entry is not None
        assert entry["reason"] == "no_progress_watchdog"

    def test_session_eviction_cancels_and_records(self, clock):
        """``session_evicted`` holds no handle through ``_release_local_dispatch``;
        the eviction callback must cancel the pinned stream itself and record
        the abandoned slot."""
        from proxy.router_helpers import _abandoned_slot_for

        from proxy import server as server_mod

        stream_cm = _fake_stream_cm()
        records = {
            "sess-evicted": {
                "active": True,
                "expires_at": time.monotonic() + 60.0,
                "stream_cm": stream_cm,
                "endpoint": "local-llama",
            }
        }
        with (
            patch.object(server_mod, "local_dispatch_records", records),
            patch.object(server_mod, "local_dispatch_records_lock", asyncio.Lock()),
            patch.object(server_mod, "local_prefill_in_flight", {"sess-evicted": True}),
            patch.object(
                server_mod, "local_prefill_in_flight_lock", asyncio.Lock()
            ),
            patch.dict(
                "proxy.session._slot_owners",
                {("local-llama", 0): "sess-evicted"},
                clear=True,
            ),
        ):
            asyncio.run(server_mod._release_lease_on_session_eviction("sess-evicted"))
            # Detached tasks close the stream on the loop the release ran on;
            # ``asyncio.run`` drains them before returning.
            entry = _abandoned_slot_for(0, endpoint="local-llama", now=clock.now)

        stream_cm.__aexit__.assert_awaited_once()
        assert entry is not None
        assert entry["reason"] == "session_evicted"
        assert entry["session_id"] == "sess-evicted"


# ═════════════════════════════════════════════════════════════════════════════
# Endpoint-wide (slot-less) abandonment entries — LP-0MVUVHA55009EU4B
#
# Two shapes of the remaining post-cancel release lag
# (LP-0MVT2KW5A008X037, 12 polls / 0.11 % of a comparable night):
#
# 1. **Attributable, but unmatched** — the release recorded an identity that no
#    longer matches the slot llama-server is draining (the exact lookup misses
#    and ``local_abandoned_slots`` stays 0).
# 2. **Unattributable** — the release had no slot identity at all
#    (``measured_cache_deficit``), so ``_record_abandoned_slot`` dropped it.
#
# Both become annotatable only if an *endpoint-wide* entry exists and the status
# builder matches it to a processing slot that **no** live dispatch record owns.
# Shape 1 additionally needs the endpoint-wide entry recorded alongside the
# mismatching exact one, so a recorded-vs-observed mismatch still annotates.
#
# Red-first encoding (the repository convention — see the module docstring and
# ``test_prefill_thrash_cooldown.py``): every assertion pinning behaviour that
# did not exist yet was marked ``pytest.mark.xfail(strict=True)`` so the
# test-only item landed on ``dev`` with a green suite. ``strict`` makes the
# assertion fail the suite the moment it starts passing, so the paired
# implementation items cannot forget to remove the marker.
#
# Marker status — **no markers remain**, each was removed by the item that
# turned its assertions green:
#
# - ``TestEndpointWideRegistry`` — LP-0MVUVHAQP001IRKK made
#   ``_record_abandoned_slot`` store the endpoint-wide entry (sentinel key,
#   ``scope`` field, TTL, refresh, per-endpoint bound) instead of dropping it.
# - the annotation / clear-on-free classes below — LP-0MVUVHB1V006HA5I
#   matched the entry in the status builder (``abandoned_scope="endpoint"``,
#   guarded by "no lease owner and no live dispatch record") and clears it on
#   a fresh snapshot that reports the slot — or the whole endpoint — free.
#
# ``ENDPOINT`` is the label the release paths carry for the local backend (the
# dispatch record's ``endpoint`` field). The status payload describes that same
# local backend without a label of its own, so the endpoint-wide lookup must
# fall back across endpoint labels exactly as the exact-match lookup already
# does (``TestAbandonedSlotRegistry.test_entry_is_unknown_endpoint_fallback``).
# ═════════════════════════════════════════════════════════════════════════════

#: Endpoint label the local backend's dispatch records carry.
ENDPOINT = "local-llama"

#: Reason a release with no slot identity carries today
#: (``proxy/proxy/measured_cache_abort.py`` ``MEASURED_CACHE_DEFICIT_REASON``).
ENDPOINT_WIDE_REASON = "measured_cache_deficit"

def _record_endpoint_wide(
    *,
    endpoint: str | None = ENDPOINT,
    reason: str = ENDPOINT_WIDE_REASON,
    session_id: str | None = "herdr-ghost",
    now: float | None = None,
) -> dict | None:
    """Record an abandonment with no slot identity (shape 2)."""
    from proxy.router_helpers import _record_abandoned_slot

    return _record_abandoned_slot(
        None,
        endpoint=endpoint,
        reason=reason,
        session_id=session_id,
        now=now,
    )


def _live_dispatch_records(session_id: str = "live-session") -> dict:
    """A genuinely active dispatch lease (live proxy-owned work)."""
    return {
        session_id: {
            "active": True,
            "started_at": 0.0,
            "expires_at": time.monotonic() + 60.0,
            "first_byte_seen": True,
        }
    }


class TestEndpointWideRegistry:
    """AC: a slot-less abandonment is stored, bounded and never collides."""

    def test_sentinel_cannot_collide_with_a_real_slot_id(self):
        """The "endpoint-wide" key is a documented sentinel that is not a
        non-negative integer, so it can never be mistaken for a real slot."""
        from proxy.router_helpers import ABANDONED_SLOT_ENDPOINT_WIDE

        assert isinstance(ABANDONED_SLOT_ENDPOINT_WIDE, int)
        assert ABANDONED_SLOT_ENDPOINT_WIDE < 0

    def test_recording_without_a_slot_yields_one_endpoint_wide_entry(self):
        """``_record_abandoned_slot(None, ...)`` records instead of silently
        dropping: one entry, attributable, keyed by the sentinel."""
        from proxy.router_helpers import (
            ABANDONED_SLOT_ENDPOINT_WIDE,
            DEFAULT_ABANDONED_SLOT_TTL_SECONDS,
        )

        from proxy import router_helpers as helpers

        entry = _record_endpoint_wide(now=0.0)

        assert entry is not None
        assert entry["scope"] == "endpoint"
        assert entry["slot_id"] is None
        assert entry["endpoint"] == ENDPOINT
        assert entry["session_id"] == "herdr-ghost"
        assert entry["reason"] == ENDPOINT_WIDE_REASON
        assert entry["abandoned_at"] == 0.0
        assert entry["ttl_seconds"] == DEFAULT_ABANDONED_SLOT_TTL_SECONDS
        # Exactly one entry, keyed by the sentinel under the recorded endpoint.
        assert list(helpers._abandoned_slots) == [
            (ENDPOINT, ABANDONED_SLOT_ENDPOINT_WIDE)
        ]

    def test_endpoint_wide_entry_never_matches_a_real_slot(self):
        """The fallback must be an explicit endpoint-wide lookup, never a
        match on a real slot id: an exact lookup for any slot stays empty, so
        the endpoint-wide entry can only be applied by the guarded path."""
        from proxy.router_helpers import _abandoned_slot_for

        assert _record_endpoint_wide(now=0.0) is not None

        for slot_id in (0, 1, 2, 7):
            assert _abandoned_slot_for(slot_id, endpoint=ENDPOINT, now=0.0) is None
        # ...including the legacy/default endpoint label.
        assert _abandoned_slot_for(0, now=0.0) is None

    def test_rerecording_refreshes_rather_than_duplicates(self, clock):
        """A second slot-less abandonment refreshes the single entry (no
        growth, latest reason) instead of appending a duplicate."""
        from proxy.router_helpers import _abandoned_slot_count

        first = _record_endpoint_wide(
            reason=ENDPOINT_WIDE_REASON, now=clock.now
        )
        assert first is not None
        clock.advance(first["ttl_seconds"] - 1.0)
        _record_endpoint_wide(reason="client_disconnect", now=clock.now)

        assert _abandoned_slot_count(endpoint=ENDPOINT, now=clock.now) == 1
        refreshed = _record_endpoint_wide(
            reason="prefill_deadline", now=clock.now
        )
        assert refreshed["reason"] == "prefill_deadline"
        assert refreshed["abandoned_at"] == clock.now

    def test_entry_is_scoped_per_endpoint(self):
        """One entry per endpoint — the bound stays
        ``(endpoint, slot)`` pairs plus one endpoint-wide entry per endpoint."""
        from proxy.router_helpers import _abandoned_slot_count

        _record_endpoint_wide(endpoint=ENDPOINT, now=0.0)
        _record_endpoint_wide(endpoint="remote-backend", now=0.0)

        assert _abandoned_slot_count(endpoint=ENDPOINT, now=0.0) == 1
        assert _abandoned_slot_count(endpoint="remote-backend", now=0.0) == 1

    def test_entry_expires_at_the_default_ttl(self, clock):
        """The endpoint-wide entry is TTL-bounded like every other entry."""
        from proxy.router_helpers import (
            DEFAULT_ABANDONED_SLOT_TTL_SECONDS,
            _abandoned_slot_count,
        )

        entry = _record_endpoint_wide(now=clock.now)
        assert entry is not None
        assert entry["ttl_seconds"] == DEFAULT_ABANDONED_SLOT_TTL_SECONDS

        clock.advance(DEFAULT_ABANDONED_SLOT_TTL_SECONDS - 1.0)
        assert _abandoned_slot_count(endpoint=ENDPOINT, now=clock.now) == 1

        clock.advance(2.0)
        assert _abandoned_slot_count(endpoint=ENDPOINT, now=clock.now) == 0


class TestEndpointWideAnnotation:
    """AC: the draining slot is annotated, truthfully, from the entry."""

    def test_shape2_no_identity_slot_is_annotated(self):
        """Shape 2 (no slot identity): a fresh slot is ``is_processing`` with
        no lease owner and no live dispatch record, so the endpoint-wide entry
        annotates it — attributable, with an explicit endpoint scope."""
        _record_endpoint_wide(now=0.0)

        slot = _annotate([_raw_slot(1, is_processing=True, n_decoded=5)])[0]

        assert slot["abandoned"] is True
        assert slot["abandonment_reason"] == ENDPOINT_WIDE_REASON
        assert slot["abandoned_session_id"] == "herdr-ghost"
        assert slot["abandoned_scope"] == "endpoint"

    def test_shape1_identity_mismatch_resolves_through_endpoint_wide(self):
        """Shape 1: the recorded id (7) differs from the observed slot id (1),
        so the exact lookup misses; the endpoint-wide entry recorded alongside
        it still annotates the draining slot."""
        from proxy.router_helpers import _abandoned_slot_for, _record_abandoned_slot

        assert (
            _record_abandoned_slot(
                7, endpoint=ENDPOINT, reason="session_evicted", session_id="herdr-ghost", now=0.0
            )
            is not None
        )
        assert _record_endpoint_wide(
            endpoint=ENDPOINT, reason="session_evicted", now=0.0
        ) is not None
        # The exact lookup genuinely misses for the observed slot.
        assert _abandoned_slot_for(1, endpoint=ENDPOINT, now=0.0) is None

        slot = _annotate([_raw_slot(1, is_processing=True, n_decoded=5)])[0]

        assert slot["abandoned"] is True
        assert slot["abandonment_reason"] == "session_evicted"
        assert slot["abandoned_scope"] == "endpoint"

    def test_exact_entry_reports_slot_scope(self):
        """Control on the new additive field: an exact match reports
        ``abandoned_scope="slot"``, not ``"endpoint"``."""
        from proxy.router_helpers import _record_abandoned_slot

        assert (
            _record_abandoned_slot(
                1, endpoint=ENDPOINT, reason="prefill_deadline", session_id="s", now=0.0
            )
            is not None
        )

        slot = _annotate([_raw_slot(1, is_processing=True, n_decoded=5)])[0]

        assert slot["abandoned"] is True
        assert slot["abandoned_scope"] == "slot"

    def test_live_lease_owner_is_never_annotated(self):
        """Truthful status (LP-0MVKSXR4P001V0CM): an endpoint-wide entry must
        not annotate a processing slot a live dispatch lease owns."""
        from proxy.observability import CAPACITY_SATURATED, compute_capacity_hint
        from proxy.session import _slot_registry_key

        assert _record_endpoint_wide(now=0.0) is not None
        srv = _server_stub(records=_live_dispatch_records())

        with patch.dict(
            "proxy.session._slot_owners",
            {_slot_registry_key(None, 1): "live-session"},
            clear=True,
        ):
            annotated = _annotate(
                [_raw_slot(1, is_processing=True, n_decoded=42)], srv
            )

        assert annotated[0].get("abandoned") is not True
        assert annotated[0].get("abandoned_scope") is None
        assert compute_capacity_hint(annotated) == CAPACITY_SATURATED

    def test_live_dispatch_record_without_a_slot_binding_is_never_annotated(self):
        """The guard covers the dispatch-record binding too: an active record
        matched to the processing slot (no persistence binding) is live work,
        so the endpoint-wide entry must not annotate it."""
        from proxy.observability import CAPACITY_SATURATED, compute_capacity_hint

        from proxy import observability as obs

        assert _record_endpoint_wide(now=0.0) is not None
        srv = _server_stub(records=_live_dispatch_records())

        with patch.object(obs, "_processing_slot_assignments", {}):
            with patch.dict("proxy.session._slot_owners", {}, clear=True):
                annotated = _annotate(
                    [_raw_slot(0, is_processing=True, n_decoded=42)], srv
                )

        assert annotated[0]["owner_session_id"] == "live-session"
        assert annotated[0].get("abandoned") is not True
        assert compute_capacity_hint(annotated) == CAPACITY_SATURATED

    def test_all_draining_pool_is_abandoned_busy(self):
        """Every generating slot draining (annotated from the endpoint-wide
        entry) → ``abandoned_busy``, never ``saturated``, never ``idle``."""
        from proxy.observability import (
            CAPACITY_ABANDONED_BUSY,
            CAPACITY_IDLE,
            CAPACITY_SATURATED,
            compute_capacity_hint,
        )

        _record_endpoint_wide(now=0.0)

        annotated = _annotate(
            [
                _raw_slot(0, is_processing=True, n_decoded=1),
                _raw_slot(1, is_processing=True, n_decoded=2),
            ]
        )

        assert [s["abandoned_scope"] for s in annotated] == ["endpoint", "endpoint"]
        hint = compute_capacity_hint(annotated)
        assert hint == CAPACITY_ABANDONED_BUSY
        assert hint != CAPACITY_SATURATED
        assert hint != CAPACITY_IDLE


class TestEndpointWideClearOnObservedFree:
    """AC: the marker survives until llama-server reports the slot free."""

    def test_fresh_snapshot_reporting_the_slot_free_clears_the_entry(self):
        """A *fresh* ``is_processing=False`` reading is trustworthy, so it
        drops the entry: a later poll cannot resurrect the annotation."""
        _record_endpoint_wide(now=0.0)

        draining = _annotate([_raw_slot(1, is_processing=True, n_decoded=9)])
        assert draining[0]["abandoned"] is True

        freed = _annotate([_raw_slot(1, is_processing=False)])
        assert freed[0]["abandoned"] is False
        assert freed[0]["abandoned_scope"] is None

        again = _annotate([_raw_slot(1, is_processing=True, n_decoded=9)])
        assert again[0]["abandoned"] is False

    def test_fresh_free_reading_also_clears_the_exact_entry(self):
        """The clear applies to the exact entry for the observed slot as well,
        so a freed slot cannot keep a stale exact annotation."""
        from proxy.router_helpers import _abandoned_slot_for, _record_abandoned_slot

        assert (
            _record_abandoned_slot(
                1, endpoint=ENDPOINT, reason="client_disconnect", session_id="s", now=0.0
            )
            is not None
        )

        assert _annotate([_raw_slot(1, is_processing=False)])[0]["abandoned"] is False
        assert _abandoned_slot_for(1, endpoint=ENDPOINT, now=0.0) is None

    def test_stale_snapshot_never_clears_the_entry(self):
        """Stale detail is not evidence of freedom: a cached "free" reading
        must not drop a live annotation (only fresh readings clear)."""
        _record_endpoint_wide(now=0.0)

        stale_free = _annotate([_raw_slot(1, is_processing=False)], stale=True)
        assert stale_free[0]["abandoned"] is False  # nothing to annotate free

        still_draining = _annotate([_raw_slot(1, is_processing=True, n_decoded=9)])
        assert still_draining[0]["abandoned"] is True

    def test_one_free_slot_does_not_clear_a_draining_sibling(self):
        """Multi-slot pool guard: the entry is cleared only when the fresh
        snapshot reports *nothing* processing — a free sibling must not drop
        the annotation of the slot that is still draining."""
        _record_endpoint_wide(now=0.0)

        annotated = _annotate(
            [
                _raw_slot(0, is_processing=True, n_decoded=4),
                _raw_slot(1, is_processing=False),
            ]
        )

        assert annotated[0]["abandoned"] is True
        assert annotated[0]["abandoned_scope"] == "endpoint"
        assert annotated[1]["abandoned"] is False


class TestClearAbandonedSlot:
    """AC: ``_clear_abandoned_slot`` drops only what the backend disproved.

    The status builder calls this on **fresh** evidence only; these pin the
    helper's own contract so the call site cannot silently over- or
    under-clear (dropping a sibling's legitimate annotation, or leaving a
    freed slot annotated forever).
    """

    def test_clearing_one_slot_leaves_a_sibling_entry(self, clock):
        """Per-slot clear: a free sibling must not disarm the draining slot."""
        from proxy.router_helpers import (
            _abandoned_slot_count,
            _abandoned_slot_for,
            _clear_abandoned_slot,
            _record_abandoned_slot,
        )

        _record_abandoned_slot(
            0, endpoint=ENDPOINT, reason="prefill_deadline", session_id="a", now=clock.now
        )
        _record_abandoned_slot(
            1, endpoint=ENDPOINT, reason="client_disconnect", session_id="b", now=clock.now
        )

        assert _clear_abandoned_slot(0) == 1
        assert _abandoned_slot_for(0, endpoint=ENDPOINT, now=clock.now) is None
        assert _abandoned_slot_for(1, endpoint=ENDPOINT, now=clock.now) is not None
        assert _abandoned_slot_count(endpoint=ENDPOINT, now=clock.now) == 1

    def test_endpoint_scope_drops_only_the_endpoint_wide_entry(self, clock):
        """``scope="endpoint"`` never touches a real slot's exact entry."""
        from proxy.router_helpers import (
            _abandoned_slot_endpoint_wide,
            _abandoned_slot_for,
            _clear_abandoned_slot,
            _record_abandoned_slot,
        )

        _record_abandoned_slot(
            1, endpoint=ENDPOINT, reason="session_evicted", session_id="a", now=clock.now
        )
        assert _record_endpoint_wide(now=clock.now) is not None

        assert _clear_abandoned_slot(scope="endpoint") == 1
        assert _abandoned_slot_endpoint_wide(now=clock.now) is None
        assert _abandoned_slot_for(1, endpoint=ENDPOINT, now=clock.now) is not None

    def test_unresolvable_identity_is_a_no_op(self, clock):
        """An unknown slot id names nothing, so it can never clear the
        endpoint-wide entry by accident — that needs ``scope="endpoint"``."""
        from proxy.router_helpers import (
            _abandoned_slot_endpoint_wide,
            _clear_abandoned_slot,
        )

        assert _record_endpoint_wide(now=clock.now) is not None

        for unresolvable in (None, -1, "1", True):
            assert _clear_abandoned_slot(unresolvable) == 0
        assert _abandoned_slot_endpoint_wide(now=clock.now) is not None
        # A real, unrecorded slot id is likewise a harmless no-op.
        assert _clear_abandoned_slot(7) == 0

    def test_clear_matches_across_endpoint_labels(self, clock):
        """Symmetric with the lookup's label fallback: an entry written under
        one endpoint spelling is still removable by the label-less status
        builder, otherwise a freed slot would stay annotated for its TTL."""
        from proxy.router_helpers import (
            _abandoned_slot_for,
            _clear_abandoned_slot,
            _record_abandoned_slot,
        )

        _record_abandoned_slot(
            1, endpoint=ENDPOINT, reason="no_progress_watchdog", session_id="a", now=clock.now
        )

        assert _clear_abandoned_slot(1) == 1
        assert _abandoned_slot_for(1, endpoint=ENDPOINT, now=clock.now) is None


class TestEndpointWideAnnotationIsReadOnly:
    """AC: annotating and clearing is decoration — never capacity mutation."""

    def test_annotation_never_touches_dispatch_or_lease_state(self):
        from proxy.router_helpers import _abandoned_slot_count
        from proxy.session import _slot_registry_key

        from proxy import session as session_mod

        assert _record_endpoint_wide(now=0.0) is not None
        assert _abandoned_slot_count() == 1

        srv = _server_stub(records=_live_dispatch_records())
        records_before = {k: dict(v) for k, v in srv.local_dispatch_records.items()}
        prefill_before = dict(srv.local_prefill_in_flight)

        with patch.dict(
            "proxy.session._slot_owners",
            {_slot_registry_key(None, 1): "live-session"},
            clear=True,
        ):
            owners_before = dict(session_mod._slot_owners)
            _annotate([_raw_slot(0, is_processing=True, n_decoded=1)], srv)
            _annotate([_raw_slot(0, is_processing=False)], srv)
            owners_after = dict(session_mod._slot_owners)

        assert srv.local_dispatch_records == records_before
        assert srv.local_prefill_in_flight == prefill_before
        assert srv.local_active_queries == 0
        assert owners_after == owners_before
        # Clearing dropped the registry entry without re-recording anything.
        assert _abandoned_slot_count() == 0


# ═════════════════════════════════════════════════════════════════════════════
# Recording and re-verifying the abandoned slot identity on every release path
# — LP-0MVUVHAQP001IRKK
#
# The registry half of the endpoint-wide contract: every abandonment path must
# record *something* attributable (the slot the backend is still draining, or
# the endpoint-wide entry when that slot cannot be resolved), and the resolved
# identity must be re-verified against what llama-server actually reports so a
# shape-1 mismatch is diagnosable instead of silent.
# ═════════════════════════════════════════════════════════════════════════════


def _rendered_log_calls(mock, needle: str) -> list[str]:
    """Render the ``%``-formatted calls a mocked logger received.

    Keeps the assertions on the *rendered* line (the shape an operator greps),
    never on the raw format string plus argument tuple.
    """
    rendered: list[str] = []
    for logged in getattr(mock, "call_args_list", []):
        args = logged.args
        if not args or not isinstance(args[0], str) or needle not in args[0]:
            continue
        try:
            rendered.append(args[0] % tuple(args[1:]))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            rendered.append(args[0])
    return rendered


def _assert_endpoint_wide(reason: str, session_id: str, clock) -> dict:
    """Assert the endpoint-wide entry exists, is attributable and matches nothing."""
    from proxy.router_helpers import (
        _abandoned_slot_endpoint_wide,
        _abandoned_slot_for,
    )

    entry = _abandoned_slot_endpoint_wide(now=clock.now)
    assert entry is not None, (
        f"{reason} dropped an unresolvable abandonment instead of recording "
        f"the endpoint-wide entry"
    )
    assert entry["scope"] == "endpoint"
    assert entry["slot_id"] is None
    assert entry["reason"] == reason
    assert entry["session_id"] == session_id
    # The endpoint-wide entry must never be returned to a slot-exact lookup.
    for slot_id in (0, 1, 2, 7):
        assert _abandoned_slot_for(slot_id, now=clock.now) is None
    return entry


class TestEveryPathRecordsAnEntryWhenIdentityIsUnresolvable:
    """AC: no abandonment path silently drops an unresolvable identity.

    One case per recording path named by the work item: prefill deadline,
    client disconnect, no-progress watchdog, orphan sweep, session eviction and
    the measured cache-deficit early abort.
    """

    def test_prefill_deadline(self, clock):
        from proxy import router_helpers as helpers

        srv = _server_stub(
            records={
                "sess-deadline": {
                    "active": True,
                    "expires_at": time.monotonic() + 60.0,
                }
            }
        )
        with patch.dict("proxy.session._slot_owners", {}, clear=True):
            asyncio.run(
                helpers._release_prefill_deadline_dispatch(srv, "sess-deadline")
            )

        _assert_endpoint_wide("prefill_deadline", "sess-deadline", clock)

    def test_client_disconnect(self, clock):
        from proxy import router_helpers as helpers

        srv = _server_stub(
            records={
                "sess-gone": {
                    "active": True,
                    "expires_at": time.monotonic() + 60.0,
                }
            }
        )
        with patch.dict("proxy.session._slot_owners", {}, clear=True):
            released = helpers._release_client_disconnect_dispatch(
                srv, "sess-gone"
            )

        assert released is True
        _assert_endpoint_wide("client_disconnect", "sess-gone", clock)

    def test_no_progress_watchdog(self, clock):
        from proxy import router_helpers as helpers

        now = clock.now
        srv = _server_stub(
            records={
                "sess-wedged": {
                    "backend": "local",
                    "active": True,
                    "started_at": now - 300.0,
                    "expires_at": now + 600.0,
                    "last_progress": 0,
                    "last_progress_ts": now - 300.0,
                }
            }
        )
        srv.local_prefill_in_flight = {"sess-wedged": True}
        srv.local_active_queries = 1

        async def _no_slots(*_a, **_k):
            return None

        with (
            patch.dict("proxy.session._slot_owners", {}, clear=True),
            patch.object(
                helpers, "_query_slots_progress", _no_slots, create=True
            ),
        ):
            asyncio.run(helpers._cleanup_stale_local_dispatch(srv))

        _assert_endpoint_wide("no_progress_watchdog", "sess-wedged", clock)

    def test_orphan_sweep(self, clock):
        from proxy import router_helpers as helpers

        now = clock.now
        srv = _server_stub(
            records={
                # No ``last_progress_ts`` baseline → the no-progress watchdog
                # abstains and the record reaches the orphan-cleanup branch.
                "sess-orphan": {
                    "backend": "local",
                    "active": True,
                    "started_at": now - 400.0,
                    "expires_at": now - 10.0,
                }
            }
        )
        srv.local_active_queries = 1

        with patch.dict("proxy.session._slot_owners", {}, clear=True):
            removed = asyncio.run(helpers._cleanup_stale_local_dispatch(srv))

        assert removed == 1
        _assert_endpoint_wide("orphan_cleanup", "sess-orphan", clock)

    def test_session_eviction(self, clock):
        from proxy import server as server_mod

        records = {
            "sess-evicted": {
                "active": True,
                "expires_at": time.monotonic() + 60.0,
                "endpoint": "local-llama",
            }
        }
        with (
            patch.object(server_mod, "local_dispatch_records", records),
            patch.object(
                server_mod, "local_dispatch_records_lock", asyncio.Lock()
            ),
            patch.object(
                server_mod, "local_prefill_in_flight", {"sess-evicted": True}
            ),
            patch.object(
                server_mod, "local_prefill_in_flight_lock", asyncio.Lock()
            ),
            patch.dict("proxy.session._slot_owners", {}, clear=True),
        ):
            asyncio.run(
                server_mod._release_lease_on_session_eviction("sess-evicted")
            )

        _assert_endpoint_wide("session_evicted", "sess-evicted", clock)

    def test_measured_cache_deficit_abort(self, clock):
        from proxy import router_helpers as helpers

        srv = _server_stub(
            records={
                "sess-abort": {
                    "active": True,
                    "expires_at": time.monotonic() + 60.0,
                }
            }
        )
        with patch.dict("proxy.session._slot_owners", {}, clear=True):
            asyncio.run(
                helpers._release_measured_cache_abort_dispatch(srv, "sess-abort")
            )

        _assert_endpoint_wide(
            "measured_cache_deficit", "sess-abort", clock
        )


class TestMeasuredCacheAbortForwardsThePinnedIdentity:
    """AC: the earliest-abort path forwards the record's slot and stream."""

    def test_record_pin_is_recorded_and_its_stream_is_cancelled(self, clock):
        """No caller arguments and no session assignment: the identity the
        dispatch record pinned at admission and the stream it pinned after the
        headers both come from the record, so the cancel and the annotation
        fire on the earliest-abort path."""
        from proxy.router_helpers import (
            _abandoned_slot_for,
            _release_measured_cache_abort_dispatch,
        )

        stream_cm = _fake_stream_cm()
        srv = _server_stub(
            records={
                "sess-abort": {
                    "active": True,
                    "expires_at": time.monotonic() + 60.0,
                    "slot_id": 3,
                    "stream_cm": stream_cm,
                }
            }
        )
        with patch.dict("proxy.session._slot_owners", {}, clear=True):
            asyncio.run(
                _release_measured_cache_abort_dispatch(srv, "sess-abort")
            )

        stream_cm.__aexit__.assert_awaited_once()
        entry = _abandoned_slot_for(3, now=clock.now)
        assert entry is not None
        assert entry["scope"] == "slot"
        assert entry["reason"] == "measured_cache_deficit"
        assert entry["session_id"] == "sess-abort"

    def test_caller_forwarded_identity_wins_over_the_pin(self, clock):
        """The explicit ``abandoned_slot_id`` / ``upstream_stream_cm`` the
        router forwards (the deadline-twin shape) take precedence, so the
        annotation names the slot the dispatch actually used."""
        from proxy.router_helpers import (
            _abandoned_slot_for,
            _release_measured_cache_abort_dispatch,
        )

        pinned_stream = _fake_stream_cm()
        forwarded_stream = _fake_stream_cm()
        srv = _server_stub(
            records={
                "sess-abort": {
                    "active": True,
                    "expires_at": time.monotonic() + 60.0,
                    "slot_id": 3,
                    "stream_cm": pinned_stream,
                }
            }
        )
        with patch.dict("proxy.session._slot_owners", {}, clear=True):
            asyncio.run(
                _release_measured_cache_abort_dispatch(
                    srv,
                    "sess-abort",
                    abandoned_slot_id=5,
                    upstream_stream_cm=forwarded_stream,
                )
            )

        forwarded_stream.__aexit__.assert_awaited_once()
        assert _abandoned_slot_for(5, now=clock.now) is not None
        assert _abandoned_slot_for(3, now=clock.now) is None

    def test_the_dispatch_record_pins_the_slot_at_admission(self, clock):
        """``_try_acquire_local_dispatch`` pins the slot identity on the record,
        so a later release can name it even after the assignment is freed."""
        from proxy.router_helpers import _try_acquire_local_dispatch
        from proxy.session import _slot_registry_key

        srv = _server_stub()
        with patch.dict(
            "proxy.session._slot_owners",
            {_slot_registry_key(None, 2): "sess-pinned"},
            clear=True,
        ):
            asyncio.run(
                _try_acquire_local_dispatch(
                    srv, 2, "sess-pinned", backend="local", body_json=None
                )
            )

        assert srv.local_dispatch_records["sess-pinned"]["slot_id"] == 2


class TestAbandonedSlotIdentityReverification:
    """AC: the recorded id is re-verified against what the backend reports."""

    def _release_drifted(self, srv, session_id: str, explicit=None):
        from proxy import router_helpers as helpers

        return asyncio.run(
            helpers._release_prefill_deadline_dispatch(
                srv, session_id, abandoned_slot_id=explicit
            )
        )

    def test_mismatch_logs_both_slots_and_records_endpoint_wide(self, clock):
        """Shape 1: the recorded id no longer matches the draining slot. The
        release must say so (``abandoned_slot_mismatch`` with both ids) and
        still record the endpoint-wide entry, so the ghost is annotatable."""
        from proxy.router_helpers import (
            _abandoned_slot_endpoint_wide,
            _abandoned_slot_for,
        )

        from proxy import observability as obs

        srv = _server_stub(
            records={
                "sess-drift": {
                    "active": True,
                    "expires_at": time.monotonic() + 60.0,
                    "slot_id": 7,
                }
            }
        )
        with (
            patch.dict("proxy.session._slot_owners", {}, clear=True),
            patch.object(obs, "_processing_slot_assignments", {1: "sess-drift"}),
        ):
            self._release_drifted(srv, "sess-drift")

        mismatches = _rendered_log_calls(srv.logger.warning, "abandoned_slot_mismatch")
        assert len(mismatches) == 1, mismatches
        assert "recorded_slot=7" in mismatches[0]
        assert "observed_slot=1" in mismatches[0]
        assert "reason=prefill_deadline" in mismatches[0]
        # Both entries: the recorded id (exact evidence) and the endpoint-wide
        # fallback the guarded status matcher can apply to the draining slot.
        exact = _abandoned_slot_for(7, now=clock.now)
        assert exact is not None and exact["scope"] == "slot"
        wide = _abandoned_slot_endpoint_wide(now=clock.now)
        assert wide is not None and wide["scope"] == "endpoint"
        assert wide["reason"] == "prefill_deadline"

    def test_no_mismatch_when_the_backend_agrees(self, clock):
        """Control: an agreeing observation records only the exact entry."""
        from proxy.router_helpers import (
            _abandoned_slot_endpoint_wide,
            _abandoned_slot_for,
        )

        from proxy import observability as obs

        srv = _server_stub(
            records={
                "sess-agreed": {
                    "active": True,
                    "expires_at": time.monotonic() + 60.0,
                    "slot_id": 1,
                }
            }
        )
        with (
            patch.dict("proxy.session._slot_owners", {}, clear=True),
            patch.object(obs, "_processing_slot_assignments", {1: "sess-agreed"}),
        ):
            self._release_drifted(srv, "sess-agreed")

        assert not _rendered_log_calls(
            srv.logger.warning, "abandoned_slot_mismatch"
        )
        assert _abandoned_slot_for(1, now=clock.now) is not None
        assert _abandoned_slot_endpoint_wide(now=clock.now) is None

    def test_unobserved_backend_is_not_a_mismatch(self, clock):
        """No ``/slots``-derived observation yet is not evidence of drift."""
        from proxy.router_helpers import (
            _abandoned_slot_endpoint_wide,
            _abandoned_slot_for,
        )

        from proxy import observability as obs

        srv = _server_stub(
            records={
                "sess-unobserved": {
                    "active": True,
                    "expires_at": time.monotonic() + 60.0,
                    "slot_id": 4,
                }
            }
        )
        with (
            patch.dict("proxy.session._slot_owners", {}, clear=True),
            patch.object(obs, "_processing_slot_assignments", {}),
        ):
            self._release_drifted(srv, "sess-unobserved")

        assert not _rendered_log_calls(
            srv.logger.warning, "abandoned_slot_mismatch"
        )
        assert _abandoned_slot_for(4, now=clock.now) is not None
        assert _abandoned_slot_endpoint_wide(now=clock.now) is None


class TestAbandonedSlotSourceProvenance:
    """AC: ``slot_source=explicit|assignment|endpoint_wide`` on the log line."""

    def _release(self, srv, session_id: str, **kwargs):
        from proxy import router_helpers as helpers

        asyncio.run(
            helpers._release_prefill_deadline_dispatch(
                srv, session_id, **kwargs
            )
        )
        return _rendered_log_calls(srv.logger.info, "abandoned_slot_recorded")

    def test_explicit_slot_id(self):
        srv = _server_stub(
            records={
                "sess-explicit": {
                    "active": True,
                    "expires_at": time.monotonic() + 60.0,
                }
            }
        )
        with patch.dict("proxy.session._slot_owners", {}, clear=True):
            lines = self._release(srv, "sess-explicit", abandoned_slot_id=6)

        assert lines and "slot=6" in lines[0]
        assert "slot_source=explicit" in lines[0]

    def test_assignment_from_the_dispatch_record(self):
        srv = _server_stub(
            records={
                "sess-assigned": {
                    "active": True,
                    "expires_at": time.monotonic() + 60.0,
                    "slot_id": 2,
                }
            }
        )
        with patch.dict("proxy.session._slot_owners", {}, clear=True):
            lines = self._release(srv, "sess-assigned")

        assert lines and "slot=2" in lines[0]
        assert "slot_source=assignment" in lines[0]

    def test_endpoint_wide_when_unresolvable(self):
        srv = _server_stub(
            records={
                "sess-unknown": {
                    "active": True,
                    "expires_at": time.monotonic() + 60.0,
                }
            }
        )
        with patch.dict("proxy.session._slot_owners", {}, clear=True):
            lines = self._release(srv, "sess-unknown")

        assert lines and "slot_source=endpoint_wide" in lines[0]
        assert "slot=None" in lines[0]


class TestAbandonedSlotEntryScope:
    """AC: every entry carries its ``scope`` and an unusable id is unresolvable."""

    def test_exact_entry_scope_is_slot(self, clock):
        from proxy.router_helpers import _record_abandoned_slot

        entry = _record_abandoned_slot(
            1, endpoint=ENDPOINT, reason="prefill_deadline", now=clock.now
        )
        assert entry["scope"] == "slot"
        assert entry["slot_id"] == 1

    def test_non_integer_identity_records_endpoint_wide(self, clock):
        """A non-int identity (e.g. a stringified id) is unresolvable, not a
        slot: it must record the endpoint-wide entry instead of dropping."""
        from proxy.router_helpers import _record_abandoned_slot

        entry = _record_abandoned_slot(
            "3",  # type: ignore[arg-type]
            endpoint=ENDPOINT,
            reason="client_disconnect",
            now=clock.now,
        )
        assert entry["scope"] == "endpoint"
        assert entry["slot_id"] is None

    def test_endpoint_wide_lookup_falls_back_across_endpoint_labels(self, clock):
        from proxy.router_helpers import _abandoned_slot_endpoint_wide

        _record_endpoint_wide(endpoint=None, now=clock.now)
        assert _abandoned_slot_endpoint_wide(
            endpoint=ENDPOINT, now=clock.now
        ) is not None
