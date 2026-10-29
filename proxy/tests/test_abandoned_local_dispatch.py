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


def _annotate(slots: list[dict], srv=None, *, model: str = "Qwen3") -> list[dict]:
    from proxy.observability import annotate_slots_for_status

    return annotate_slots_for_status(slots, srv, model, config={}, stale=False)


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
