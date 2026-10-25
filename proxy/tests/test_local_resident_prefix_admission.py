"""Residency-aware local admission ladder (LP-0MVN1IQTF0037WTG).

The local pool is scheduled **per request**: a request is admitted to one of
two llama-server slots without any session→slot tenancy and without consulting
the *actual* reusable prefix, so another session's turn evicts a resident
prefix and the resident re-prefills from scratch (2026-10-24: 27 sessions
through 2 slots, aggregate prefix reuse 26.1 %, median 15.6 % for prompts
>20k tokens).

This module pins the intended behaviour of the three-part fix before any of it
exists (test-first, per the parent item's plan):

- **AC1 — resident-prefix admission gate.** A large prompt is admitted locally
  only when the session's *live* measured reusable prefix is at or above
  ``local_resident_prefix_min_ratio`` (default 0.5, ``0`` disables). Denials
  skip local with reason ``resident_prefix_insufficient`` and fall through to
  the normal chain.
- **AC2 — session tenancy.** At most ``local_session_tenants`` distinct
  sessions hold a local reservation at once; a further session's large prompt
  falls through instead of evicting a resident (denial reason ``tenant_limit``).
  Same-session re-entry is always allowed and an idle-lapsed tenancy is free.
- **AC3 — live-prefix pricing.** ``new_tokens`` is priced from the same live
  measurement, so the cold-token cap is evaluated against real prefill work
  rather than the (structurally over-optimistic) predicted sidecar ratio.
- **AC6 — tenancy is a lease.** An observed client disconnect frees the
  tenancy immediately; a silent half-open drop is bounded by the no-progress
  watchdog window, never only by the 900 s lifetime ceiling.

Store of truth: only the two documented sources are stubbed — the in-memory
``/slots`` snapshot (``observability._last_slot_details_cache`` +
``_last_slot_details_at``) and the in-process ownership registry
(``session._slot_owners``). No HTTP, no ``/slots`` call on the admission hot
path (LP-0MUXTW4MD003TVIC), no sleeps: every matrix cell is a pure call.
"""

from __future__ import annotations

import time

import proxy.observability as observability
import proxy.provider as provider
import proxy.router_helpers as router_helpers
import proxy.session as session_registry
import pytest

# The 2026-10-24 divergence fixture: a 49,103-token prompt whose slot reported
# n_past=738 (≈1.5 % reusable) while the sidecar prediction said 0.99.
EST_TOKENS = 49_103
N_PAST_COLD = 738
COLD_THRESHOLD = 25_000

# Skip reasons the ladder emits (parent AC1/AC2).
SKIP_RESIDENT_PREFIX = "resident_prefix_insufficient"
SKIP_TENANT_LIMIT = "tenant_limit"

PENDING_VIEW = "pending LP-0MVN4L7AG003PC80 (shared residency view)"
PENDING_GATE = "pending LP-0MVN4L7KT003XUBA (resident-prefix gate / live pricing)"


# ---------------------------------------------------------------------------
# Fixture factory — stubs only the two documented residency sources
# ---------------------------------------------------------------------------

def _install_slot_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    slots: list[dict],
    owners: dict[tuple, str],
    age: float = 0.5,
) -> None:
    """Install a slot snapshot + ownership registry and return the view.

    *slots* entries use the cache's own shape (``slot_id``, ``is_processing``,
    ``context_tokens``); *owners* is keyed by ``(endpoint, slot_id)`` exactly
    like ``session._slot_owners``.
    """
    observability._last_slot_details_cache[:] = [dict(s) for s in slots]
    observability._last_slot_details_at = time.monotonic() - age
    session_registry._slot_owners.clear()
    session_registry._slot_owners.update(owners)


@pytest.fixture(autouse=True)
def _reset_residency_state():
    """Keep the two documented sources (and the tenancy registry) isolated."""
    observability._last_slot_details_cache[:] = []
    observability._last_slot_details_at = 0.0
    session_registry._slot_owners.clear()
    yield
    observability._last_slot_details_cache[:] = []
    observability._last_slot_details_at = 0.0
    session_registry._slot_owners.clear()


def _slot(slot_id: int, context_tokens: int | None, *, processing: bool = True) -> dict:
    return {
        "slot_id": slot_id,
        "is_processing": processing,
        "n_decoded": None,
        "context_tokens": context_tokens,
    }


def _view(*, slots=None, owners=None, age: float = 0.5, monkeypatch=None):
    """Build a residency view from a stubbed snapshot (fails until C2 lands)."""
    from proxy.residency import get_residency_view

    _install_slot_state(
        monkeypatch,
        slots=slots if slots is not None else [_slot(0, EST_TOKENS)],
        owners=owners if owners is not None else {("", 0): "sess-a"},
        age=age,
    )
    return get_residency_view()


def _gate(
    *,
    session_id: str = "sess-a",
    estimated_tokens: int = EST_TOKENS,
    min_ratio: float = 0.5,
    predicted_ratio: float = 0.0,
    view=None,
    cold_threshold: int = COLD_THRESHOLD,
    warm_threshold: int = 0,
    monkeypatch: pytest.MonkeyPatch | None = None,
):
    """Run the local admission ladder as the router does for a large prompt."""
    if monkeypatch is not None:
        monkeypatch.setattr(
            provider, "expected_cached_ratio", lambda *a, **k: predicted_ratio
        )
    return provider._should_skip_local(
        "Qwen3",
        session_id,
        {"messages": []},
        cold_threshold,
        estimated_tokens=estimated_tokens,
        warm_cache_threshold=warm_threshold,
        resident_prefix_min_ratio=min_ratio,
        residency_view=view,
    )


# ---------------------------------------------------------------------------
# AC1 — resident accept / non-resident reject
# ---------------------------------------------------------------------------

def test_resident_session_admitted(monkeypatch):
    """A session whose slot holds its full prompt is admitted locally."""
    view = _view(slots=[_slot(0, EST_TOKENS)], owners={("", 0): "sess-a"}, monkeypatch=monkeypatch)

    skip, reason = _gate(view=view, predicted_ratio=0.0, monkeypatch=monkeypatch)

    assert (skip, reason) == (False, None)


def test_non_resident_session_rejected(monkeypatch):
    """A cold prompt from a session that owns no prefix is skipped locally."""
    view = _view(
        slots=[_slot(0, EST_TOKENS)],
        owners={("", 0): "sess-other"},
        monkeypatch=monkeypatch,
    )

    skip, reason = _gate(view=view, predicted_ratio=0.99, monkeypatch=monkeypatch)

    assert skip is True
    assert reason == SKIP_RESIDENT_PREFIX


# ---------------------------------------------------------------------------
# AC1 — ratio boundary
# ---------------------------------------------------------------------------

def test_ratio_at_boundary_admits(monkeypatch):
    """Exactly ``local_resident_prefix_min_ratio`` of the prompt is enough."""
    view = _view(
        slots=[_slot(0, 50_000)],
        owners={("", 0): "sess-a"},
        monkeypatch=monkeypatch,
    )

    # The live new-token cap is raised here so the boundary under test is the
    # ratio, not the (separately covered) live-pricing cap.
    skip, reason = _gate(
        view=view, estimated_tokens=100_000, min_ratio=0.5,
        cold_threshold=60_000, monkeypatch=monkeypatch,
    )

    assert (skip, reason) == (False, None)


def test_one_token_below_boundary_rejects(monkeypatch):
    """One token below the boundary is not resident enough."""
    view = _view(
        slots=[_slot(0, 49_999)],
        owners={("", 0): "sess-a"},
        monkeypatch=monkeypatch,
    )

    # The live new-token cap is raised here so the boundary under test is the
    # ratio, not the (separately covered) live-pricing cap.
    skip, reason = _gate(
        view=view, estimated_tokens=100_000, min_ratio=0.5,
        cold_threshold=60_000, monkeypatch=monkeypatch,
    )

    assert skip is True
    assert reason == SKIP_RESIDENT_PREFIX


def test_zero_min_ratio_disables_gate(monkeypatch):
    """``local_resident_prefix_min_ratio: 0`` restores the predicted-ratio ladder."""
    view = _view(
        slots=[_slot(0, EST_TOKENS)],
        owners={("", 0): "sess-other"},
        monkeypatch=monkeypatch,
    )

    # Non-resident, but the prediction says the prompt is already cached.
    skip, reason = _gate(view=view, min_ratio=0.0, predicted_ratio=0.99,
                         monkeypatch=monkeypatch)

    assert (skip, reason) == (False, None)


# ---------------------------------------------------------------------------
# AC1/AC7 — unknown ``context_tokens`` fallback ladder
# ---------------------------------------------------------------------------

def test_unknown_context_tokens_with_ownership_is_resident(monkeypatch):
    """Unreported ``context_tokens`` + slot ownership ⇒ treat as resident."""
    view = _view(
        slots=[_slot(0, None)],
        owners={("", 0): "sess-a"},
        monkeypatch=monkeypatch,
    )

    skip, reason = _gate(view=view, monkeypatch=monkeypatch)

    assert (skip, reason) == (False, None)


def test_unknown_context_tokens_without_ownership_with_fresh_view_rejects(monkeypatch):
    """A fresh view showing no slot of its own is a measured zero prefix.

    With a live snapshot that appears to own nothing, the session's reusable
    prefix is measurably zero, so *no* prediction may readmit it (this is the
    2026-10-24 eviction shape: the prediction says 0.99, the pool holds
    another session's prefix).
    """
    view = _view(
        slots=[_slot(0, None)],
        owners={},
        monkeypatch=monkeypatch,
    )

    skip_cached, reason_cached = _gate(
        view=view, predicted_ratio=0.99, monkeypatch=monkeypatch
    )

    assert skip_cached is True
    assert reason_cached == SKIP_RESIDENT_PREFIX


def test_unknown_state_falls_back_to_prediction(monkeypatch):
    """No fresh view and no ownership ⇒ today's predicted-ratio decision.

    Fail-open: a stale/missing snapshot must never become a hard local
    outage for a session the ownership registry knows nothing about.
    """
    from proxy.residency import RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS

    view = _view(
        slots=[_slot(0, None)],
        owners={},
        age=RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS + 1.0,
        monkeypatch=monkeypatch,
    )

    skip_cached, reason_cached = _gate(
        view=view, predicted_ratio=0.99, monkeypatch=monkeypatch
    )
    skip_cold, reason_cold = _gate(
        view=view, predicted_ratio=0.0, monkeypatch=monkeypatch
    )

    assert (skip_cached, reason_cached) == (False, None)
    assert skip_cold is True
    assert reason_cold == "large_context_bypass"


def test_stale_snapshot_is_unknown(monkeypatch):
    """A snapshot older than the freshness bound is ``unknown``, not trusted."""
    from proxy.residency import RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS

    view = _view(
        slots=[_slot(0, N_PAST_COLD)],
        owners={("", 0): "sess-a"},
        age=RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS + 1.0,
        monkeypatch=monkeypatch,
    )

    assert view.known is False
    # The (untrusted) measurement is still readable for logging, but the
    # admission ladder must not act on it.
    assert view.context_tokens_for_session("sess-a") == N_PAST_COLD


def test_stale_snapshot_falls_open_to_ownership(monkeypatch):
    """A stale snapshot must not become a hard local outage."""
    from proxy.residency import RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS

    view = _view(
        slots=[_slot(0, N_PAST_COLD)],
        owners={("", 0): "sess-a"},
        age=RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS + 1.0,
        monkeypatch=monkeypatch,
    )

    # Session owns the slot (authoritative in-process registry) ⇒ resident.
    skip, reason = _gate(view=view, monkeypatch=monkeypatch)
    assert (skip, reason) == (False, None)


def test_missing_snapshot_reports_unknown_not_error(monkeypatch):
    """No snapshot at all is an explicit ``unknown`` view, never an exception."""
    _install_slot_state(monkeypatch, slots=[], owners={})
    from proxy.residency import get_residency_view

    view = get_residency_view()

    assert view.known is False
    assert view.slots == ()


# ---------------------------------------------------------------------------
# C2 — shared residency view projection
# ---------------------------------------------------------------------------

def test_view_maps_snapshot_and_owner_registry(monkeypatch):
    """Per-slot fields come from the snapshot, ownership from the registry."""
    _install_slot_state(
        monkeypatch,
        slots=[_slot(0, EST_TOKENS, processing=True), _slot(1, None, processing=False)],
        owners={("", 0): "sess-a", ("http://192.0.2.9:8080", 1): "sess-b"},
    )
    from proxy.residency import get_residency_view

    view = get_residency_view()

    assert view.known is True
    slot0, slot1 = view.slots
    assert (slot0.endpoint, slot0.slot_id, slot0.owner_session) == ("", 0, "sess-a")
    assert slot0.context_tokens == EST_TOKENS
    assert slot0.is_processing is True
    # Multi-endpoint key shape is preserved for the second server.
    assert (slot1.endpoint, slot1.slot_id, slot1.owner_session) == (
        "http://192.0.2.9:8080",
        1,
        "sess-b",
    )
    assert slot1.context_tokens is None
    assert view.owner_of_slot(1) == "sess-b"
    assert view.context_tokens_for_session("sess-a") == EST_TOKENS
    assert view.context_tokens_for_session("sess-unknown") is None


def test_view_issues_no_upstream_slots_query(monkeypatch):
    """Building the view must never call ``/slots`` on the hot path."""
    _install_slot_state(
        monkeypatch, slots=[_slot(0, EST_TOKENS)], owners={("", 0): "sess-a"}
    )

    def _boom(*args, **kwargs):
        raise AssertionError("/slots must not be queried on the admission path")

    monkeypatch.setattr(observability, "_query_slots_detail", _boom)
    monkeypatch.setattr(observability, "_query_slots_progress", _boom)
    from proxy.residency import get_residency_view

    view = get_residency_view()

    assert view.known is True
    assert view.context_tokens_for_session("sess-a") == EST_TOKENS


def test_view_is_a_detached_snapshot(monkeypatch):
    """A concurrent refresh cannot mutate a view that is mid-decision."""
    _install_slot_state(
        monkeypatch, slots=[_slot(0, EST_TOKENS)], owners={("", 0): "sess-a"}
    )
    from proxy.residency import get_residency_view

    view = get_residency_view()
    # Simulate the ~1 s poller landing while the decision is in flight.
    observability._last_slot_details_cache[:] = [_slot(0, 1)]
    session_registry._slot_owners[("", 0)] = "sess-other"

    assert view.context_tokens_for_session("sess-a") == EST_TOKENS
    assert view.owner_of_slot(0) == "sess-a"
    assert view.occupied_sessions() == ("sess-a",)


def test_view_exposes_decision_logging_fields(monkeypatch):
    """AC5 fields (n_past_actual, slot owner, snapshot age) come from here."""
    _install_slot_state(
        monkeypatch,
        slots=[_slot(0, N_PAST_COLD), _slot(1, 0, processing=False)],
        owners={("", 0): "sess-a", ("", 1): "sess-b"},
        age=1.5,
    )
    from proxy.residency import get_residency_view

    view = get_residency_view()

    assert view.snapshot_age is not None
    assert 1.4 <= view.snapshot_age <= 2.5
    assert view.context_tokens_for_session("sess-a") == N_PAST_COLD
    assert view.owner_of_slot(0) == "sess-a"
    assert view.processing_sessions() == ("sess-a",)
    assert view.occupied_sessions() == ("sess-a", "sess-b")


# ---------------------------------------------------------------------------
# AC7 — predicted-vs-actual divergence (the 2026-10-24 failure shape)
# ---------------------------------------------------------------------------

def test_predicted_high_actual_low_rejects(monkeypatch):
    """Predicted 0.99 with a measured 738/49,103 (≈1.5 %) prefix is refused."""
    view = _view(
        slots=[_slot(0, N_PAST_COLD)],
        owners={("", 0): "sess-a"},
        monkeypatch=monkeypatch,
    )

    skip, reason = _gate(view=view, predicted_ratio=0.99, monkeypatch=monkeypatch)

    assert skip is True
    assert reason == SKIP_RESIDENT_PREFIX


# ---------------------------------------------------------------------------
# AC3 — live-prefix pricing
# ---------------------------------------------------------------------------

def test_live_new_tokens_over_budget_rejected(monkeypatch):
    """A resident session whose *live* new_tokens exceed the cap is refused.

    The predicted ratio (0.99) would have admitted it (1,000 new tokens); the
    live measurement (60,000 of 100,000 reusable ⇒ 40,000 new) must not.
    """
    view = _view(
        slots=[_slot(0, 60_000)],
        owners={("", 0): "sess-a"},
        monkeypatch=monkeypatch,
    )

    skip, reason = _gate(
        view=view,
        estimated_tokens=100_000,
        predicted_ratio=0.99,
        monkeypatch=monkeypatch,
    )

    assert skip is True
    assert reason == "large_context_bypass"


def test_live_new_tokens_within_budget_admitted(monkeypatch):
    """A resident session with a small *live* new-token term is admitted."""
    view = _view(
        slots=[_slot(0, EST_TOKENS - 1_000)],
        owners={("", 0): "sess-a"},
        monkeypatch=monkeypatch,
    )

    skip, reason = _gate(view=view, predicted_ratio=0.0, monkeypatch=monkeypatch)

    assert (skip, reason) == (False, None)


def test_hard_warm_cap_still_wins(monkeypatch):
    """The physical warm-cache cap is unchanged and evaluated first."""
    view = _view(slots=[_slot(0, EST_TOKENS)], owners={("", 0): "sess-a"},
                 monkeypatch=monkeypatch)

    skip, reason = _gate(
        view=view,
        estimated_tokens=200_000,
        warm_threshold=120_000,
        monkeypatch=monkeypatch,
    )

    assert skip is True
    assert reason == "context_too_large"


# ---------------------------------------------------------------------------
# Idle-exempt override stays last in the ladder
# ---------------------------------------------------------------------------

def test_resident_prefix_denial_is_idle_overridable(monkeypatch):
    """A resident-prefix denial may still be served by an idle GPU.

    The ladder itself refuses cold work from a non-resident session; the
    idle-aware override (``local_idle_admit_*``) is applied *after* it, so the
    new gate must not silently disable that behaviour.
    """
    view = _view(slots=[_slot(0, EST_TOKENS)], owners={("", 0): "sess-other"},
                 monkeypatch=monkeypatch)

    skip, reason = _gate(view=view, monkeypatch=monkeypatch)

    assert skip is True
    assert reason == SKIP_RESIDENT_PREFIX
    assert reason in provider.IDLE_OVERRIDABLE_SKIP_REASONS
    assert "context_too_large" not in provider.IDLE_OVERRIDABLE_SKIP_REASONS


# ---------------------------------------------------------------------------
# AC2 — session tenancy bound
# ---------------------------------------------------------------------------

class _SrvStub:
    """Minimal server stub: only the dispatch registry and a logger."""

    def __init__(self, records: dict | None = None, *, now: float | None = None):
        import asyncio

        self.local_dispatch_records = dict(records or {})
        self.local_prefill_in_flight: dict = {}
        self.local_active_queries = 0
        self.config = {"server": {}}
        self.logger = _NullLogger()
        # Locks exist so the async release helpers take their production path.
        self.local_dispatch_records_lock = asyncio.Lock()
        self.local_active_queries_lock = asyncio.Lock()
        self.local_prefill_in_flight_lock = asyncio.Lock()

    def record(self, session_id: str, *, now: float, active: bool = True,
               ttl: float = 30.0) -> str:
        key = router_helpers._dispatch_lease_key(None, session_id)
        self.local_dispatch_records[key] = {
            "backend": "local",
            "started_at": now,
            "active": active,
            "expires_at": now + ttl,
            "model_name": "Qwen3",
            "last_progress": 0,
            "last_progress_ts": now,
        }
        return key


class _NullLogger:
    def __getattr__(self, _name):
        return lambda *a, **k: None


def test_tenancy_bound_denies_new_session():
    """A third session's large prompt falls through instead of evicting."""
    now = 1000.0
    srv = _SrvStub()
    srv.record("sess-a", now=now)
    srv.record("sess-b", now=now)

    decision = router_helpers.evaluate_session_tenancy(
        srv, "sess-c", tenants=2, idle_seconds=30.0, now=now
    )

    assert decision.admitted is False
    assert decision.reason == SKIP_TENANT_LIMIT
    assert set(decision.occupants) == {"sess-a", "sess-b"}


def test_tenancy_same_session_reentry_allowed():
    """A tenant's own later turn is never blocked by the bound."""
    now = 1000.0
    srv = _SrvStub()
    srv.record("sess-a", now=now)
    srv.record("sess-b", now=now)

    decision = router_helpers.evaluate_session_tenancy(
        srv, "sess-a", tenants=2, idle_seconds=30.0, now=now
    )

    assert decision.admitted is True
    assert decision.reason is None


def test_tenancy_idle_lapsed_resident_is_not_counted():
    """A resident whose tenancy idle-lapsed no longer blocks a newcomer."""
    now = 1000.0
    srv = _SrvStub()
    srv.record("sess-a", now=now)
    srv.record("sess-b", now=now)

    decision = router_helpers.evaluate_session_tenancy(
        srv, "sess-c", tenants=2, idle_seconds=30.0, now=now + 31.0
    )

    assert decision.admitted is True


def test_tenancy_zero_disables_bound():
    """``local_session_tenants: 0`` disables the bound."""
    now = 1000.0
    srv = _SrvStub()
    srv.record("sess-a", now=now)
    srv.record("sess-b", now=now)
    srv.record("sess-c", now=now)

    decision = router_helpers.evaluate_session_tenancy(
        srv, "sess-d", tenants=0, idle_seconds=30.0, now=now
    )

    assert decision.admitted is True


def test_tenancy_default_bound_is_the_slot_count():
    """The resolved default is the slot count, i.e. two tenants for two slots."""
    assert router_helpers.resolve_session_tenants({"server": {}}, max_local=2) == 2
    assert router_helpers.resolve_session_tenants(
        {"server": {"local_session_tenants": 1}}, max_local=2
    ) == 1
    assert router_helpers.resolve_session_tenants(
        {"server": {"local_session_tenants": 0}}, max_local=2
    ) == 0


# ---------------------------------------------------------------------------
# AC6 — tenancy is a lease, not a lock (both reclaim bounds)
# ---------------------------------------------------------------------------

def test_observed_disconnect_frees_tenancy_immediately(monkeypatch):
    """An observed disconnect releases the reservation without waiting."""
    now = 1000.0
    srv = _SrvStub()
    srv.record("sess-a", now=now)
    srv.record("sess-b", now=now)
    session_registry._slot_owners[("", 0)] = "sess-a"
    session_registry._slot_owners[("", 1)] = "sess-b"

    released = router_helpers._release_client_disconnect_dispatch(srv, "sess-a")

    decision = router_helpers.evaluate_session_tenancy(
        srv, "sess-c", tenants=2, idle_seconds=30.0, now=now
    )
    assert released is True
    assert decision.admitted is True
    assert ("", 0) not in session_registry._slot_owners


def test_silent_drop_bounded_by_no_progress_window(monkeypatch):
    """A silent half-open drop is reclaimed inside the 180 s watchdog bound."""
    assert router_helpers.DEFAULT_TENANCY_IDLE_SECONDS <= 180.0

    now = 1000.0
    srv = _SrvStub()
    srv.record("sess-a", now=now)
    srv.record("sess-b", now=now)

    within = router_helpers.evaluate_session_tenancy(
        srv, "sess-c", tenants=2, idle_seconds=180.0, now=now + 179.0
    )
    after = router_helpers.evaluate_session_tenancy(
        srv, "sess-c", tenants=2, idle_seconds=180.0, now=now + 180.0
    )

    assert within.admitted is False
    assert after.admitted is True


def test_reserve_prices_from_live_residency(monkeypatch):
    """``POST /llama/local/reserve`` prices the same live prefix as the guard.

    A caller that predicted ``cached_ratio: 0.99`` but whose slot actually
    holds 60 % of the prompt must be priced at 0.6, so a reservation can never
    promise capacity the guard would refuse.
    """
    from proxy import admission_reserve

    admission_reserve.reset()
    _install_slot_state(
        monkeypatch,
        slots=[_slot(0, 60_000)],
        owners={("", 0): "sess-a"},
    )

    status, payload = admission_reserve.reserve(
        {
            "estimated_tokens": 100_000,
            "cached_ratio": 0.99,
            "session_id": "sess-a",
            "dry_run": True,
        },
        {"server": {}},
    )

    admission_reserve.reset()
    assert status == 200
    assert payload["cached_ratio_source"] == "live"
    assert payload["cached_ratio"] == pytest.approx(0.6, abs=1e-6)


def test_reserve_falls_back_to_body_ratio_without_measurement(monkeypatch):
    """No fresh measurement (or no slot) leaves the caller's ratio alone."""
    from proxy import admission_reserve

    admission_reserve.reset()
    _install_slot_state(monkeypatch, slots=[], owners={})

    status, payload = admission_reserve.reserve(
        {
            "estimated_tokens": 100_000,
            "cached_ratio": 0.99,
            "session_id": "sess-a",
            "dry_run": True,
        },
        {"server": {}},
    )

    admission_reserve.reset()
    assert status == 200
    assert payload["cached_ratio_source"] == "body"
    assert payload["cached_ratio"] == pytest.approx(0.99, abs=1e-6)


# ---------------------------------------------------------------------------
# C3 — gate plumbing: config knob, facts provenance, no hot-path /slots call
# ---------------------------------------------------------------------------

def test_resident_prefix_min_ratio_config_resolution():
    """The knob's default, disable semantics and clamping."""
    assert provider._effective_resident_prefix_min_ratio({"server": {}}) == 0.5
    assert provider._effective_resident_prefix_min_ratio({}) == 0.5
    assert provider._effective_resident_prefix_min_ratio(None) == 0.5
    assert (
        provider._effective_resident_prefix_min_ratio(
            {"server": {"local_resident_prefix_min_ratio": 0}}
        )
        == 0.0
    )
    assert (
        provider._effective_resident_prefix_min_ratio(
            {"server": {"local_resident_prefix_min_ratio": 0.25}}
        )
        == 0.25
    )
    # Out-of-range / malformed values fail safe rather than opening or closing
    # the gate silently.
    assert (
        provider._effective_resident_prefix_min_ratio(
            {"server": {"local_resident_prefix_min_ratio": 4}}
        )
        == 1.0
    )
    assert (
        provider._effective_resident_prefix_min_ratio(
            {"server": {"local_resident_prefix_min_ratio": "nonsense"}}
        )
        == 0.5
    )


def test_resolve_resident_prefix_provenance(monkeypatch):
    """The fallback ladder's facts are reported with their provenance."""
    measured = provider.resolve_resident_prefix(
        "sess-a",
        EST_TOKENS,
        view=_view(
            slots=[_slot(0, N_PAST_COLD)],
            owners={("", 0): "sess-a"},
            monkeypatch=monkeypatch,
        ),
    )
    assert measured.source == "measured"
    assert measured.resident is True
    assert measured.n_past_actual == N_PAST_COLD
    assert measured.reuse_ratio == pytest.approx(N_PAST_COLD / EST_TOKENS)

    restore = provider.resolve_resident_prefix(
        "sess-a",
        EST_TOKENS,
        view=_view(
            slots=[_slot(0, None)], owners={("", 0): "sess-a"},
            monkeypatch=monkeypatch,
        ),
    )
    assert restore.source == "session_slot"
    assert restore.resident is True
    assert restore.reuse_ratio == 1.0

    evicted = provider.resolve_resident_prefix(
        "sess-a",
        EST_TOKENS,
        view=_view(
            slots=[_slot(0, EST_TOKENS)], owners={("", 0): "sess-other"},
            monkeypatch=monkeypatch,
        ),
    )
    assert evicted.source == "non_resident"
    assert evicted.resident is False
    assert evicted.slot_owner_session == "sess-other"

    unknown = provider.resolve_resident_prefix(
        "sess-a",
        EST_TOKENS,
        view=_view(slots=[], owners={}, monkeypatch=monkeypatch),
    )
    assert unknown.source == "unknown"
    assert unknown.resident is False


def test_gate_never_issues_an_upstream_slots_query(monkeypatch):
    """AC1's constraint: the ladder consumes only the cached view.

    llama-server holds ``/slots`` for >15 s while processing a prompt batch
    (LP-0MUXTW4MD003TVIC), so a synchronous query on the admission hot path
    would stall every routing decision. Both query helpers are patched to
    raise; running the ladder with no explicit view must still work.
    """

    def _boom(*args, **kwargs):
        raise AssertionError("/slots must not be queried on the admission path")

    _install_slot_state(
        monkeypatch, slots=[_slot(0, EST_TOKENS)], owners={("", 0): "sess-a"}
    )
    monkeypatch.setattr(observability, "_query_slots_detail", _boom)
    monkeypatch.setattr(observability, "_query_slots_progress", _boom)
    monkeypatch.setattr(provider, "expected_cached_ratio", lambda *a, **k: 0.0)

    skip, reason = provider._should_skip_local(
        "Qwen3",
        "sess-a",
        {"messages": []},
        COLD_THRESHOLD,
        estimated_tokens=EST_TOKENS,
        resident_prefix_min_ratio=0.5,
        residency_view=None,
    )

    assert (skip, reason) == (False, None)


# ---------------------------------------------------------------------------
# C4 — the tenancy rung inside the admission ladder
# ---------------------------------------------------------------------------

def _install_tenancy(sessions, *, endpoint=None, ttl=30.0, age=0.0):
    """Populate the real dispatch registry with live leases for *sessions*."""
    import proxy.server as server_module

    now = time.monotonic() - age
    for name in sessions:
        key = router_helpers._dispatch_lease_key(endpoint, name)
        server_module.local_dispatch_records[key] = {
            "backend": endpoint or "local",
            "started_at": now,
            "active": True,
            "expires_at": now + ttl,
            "model_name": "Qwen3",
            "last_progress": 0,
            "last_progress_ts": now,
        }


@pytest.fixture(autouse=True)
def _reset_dispatch_registry_after_test():
    yield
    import proxy.server as server_module

    server_module.local_dispatch_records.clear()


def _gate_with_tenancy(*, session_id, tenants=2, idle_seconds=30.0, view=None,
                       monkeypatch, min_ratio=0.5):
    return provider._should_skip_local(
        "Qwen3",
        session_id,
        {"messages": []},
        COLD_THRESHOLD,
        estimated_tokens=EST_TOKENS,
        resident_prefix_min_ratio=min_ratio,
        residency_view=view,
        session_tenants=tenants,
        tenancy_idle_seconds=idle_seconds,
    )


def test_ladder_denies_a_third_session_with_tenant_limit(monkeypatch):
    """Two live tenants hold the pool ⇒ a third session's prompt falls through.

    The residency view is unknown here (the case the gate fails open on), so
    tenancy is the rung that keeps a newcomer from evicting a resident.
    """
    _install_tenancy(["sess-a", "sess-b"])
    view = _view(slots=[], owners={}, monkeypatch=monkeypatch)

    skip, reason = _gate_with_tenancy(
        session_id="sess-c", view=view, monkeypatch=monkeypatch
    )

    assert skip is True
    assert reason == SKIP_TENANT_LIMIT


def test_ladder_admits_a_tenant_session_reentry(monkeypatch):
    """A resident's own next turn is never blocked by the bound."""
    _install_tenancy(["sess-a", "sess-b"])
    view = _view(
        slots=[_slot(0, EST_TOKENS), _slot(1, EST_TOKENS)],
        owners={("", 0): "sess-a", ("", 1): "sess-b"},
        monkeypatch=monkeypatch,
    )

    skip, reason = _gate_with_tenancy(
        session_id="sess-a", view=view, monkeypatch=monkeypatch
    )

    assert (skip, reason) == (False, None)


def test_ladder_admits_when_tenant_idle_lapsed(monkeypatch):
    """A lapsed resident tenancy frees the pool for a newcomer."""
    # Live leases (unexpired) whose last activity is 5 s old: they lapse on a
    # 1 s idle bound but not on the default 30 s one.
    _install_tenancy(["sess-a", "sess-b"], age=5.0)
    view = _view(slots=[], owners={}, monkeypatch=monkeypatch)

    skip, reason = _gate_with_tenancy(
        session_id="sess-c",
        idle_seconds=1.0,
        view=view,
        monkeypatch=monkeypatch,
    )

    # The tenancy rung released the pool; what remains is the (separately
    # covered) cold-pricing decision, never a `tenant_limit` denial.
    assert reason != SKIP_TENANT_LIMIT
    assert reason == "large_context_bypass"


def test_tenancy_bound_disabled_with_zero(monkeypatch):
    """``local_session_tenants: 0`` disables the bound in the ladder."""
    _install_tenancy(["sess-a", "sess-b", "sess-c"])
    view = _view(slots=[], owners={}, monkeypatch=monkeypatch)
    monkeypatch.setattr(provider, "expected_cached_ratio", lambda *a, **k: 0.99)

    skip, reason = _gate_with_tenancy(
        session_id="sess-d", tenants=0, view=view, monkeypatch=monkeypatch
    )

    assert (skip, reason) == (False, None)


def test_tenancy_config_resolution():
    """The two tenancy knobs resolve with their documented defaults."""
    assert router_helpers.resolve_session_tenants({"server": {}}, max_local=2) == 2
    assert (
        router_helpers.resolve_session_tenants(
            {"server": {"session_slot_pool_size": 4}}
        )
        == 4
    )
    assert (
        router_helpers.resolve_session_tenants(
            {"server": {"local_session_tenants": 1}}
        )
        == 1
    )
    assert (
        router_helpers.resolve_session_tenants(
            {"server": {"local_session_tenants": 0}}
        )
        == 0
    )
    assert (
        router_helpers.resolve_session_tenants(
            {"server": {"local_session_tenants": "nonsense", "session_slot_pool_size": 2}}
        )
        == 2
    )
    assert (
        router_helpers.resolve_session_tenants(
            {"server": {"local_session_tenants": "nonsense"}}, max_local=2
        )
        == 2
    )

    assert (
        router_helpers.resolve_tenancy_idle_seconds({"server": {}})
        == router_helpers.DEFAULT_TENANCY_IDLE_SECONDS
    )
    assert (
        router_helpers.resolve_tenancy_idle_seconds(
            {"server": {"local_session_tenancy_idle_seconds": 12}}
        )
        == 12.0
    )
    # Derived from the adaptive dispatch lease when the explicit knob is absent.
    assert (
        router_helpers.resolve_tenancy_idle_seconds(
            {"server": {"local_dispatch_lease_timeout_seconds": 45}}
        )
        == 45.0
    )
    # A zero would lapse tenancies instantly: fall back to the default.
    assert (
        router_helpers.resolve_tenancy_idle_seconds(
            {"server": {"local_session_tenancy_idle_seconds": 0}}
        )
        == router_helpers.DEFAULT_TENANCY_IDLE_SECONDS
    )


def test_tenancy_fails_open_when_registry_is_unreadable(monkeypatch):
    """An unreadable registry must never deny local (fail-open)."""

    class _Broken:
        @property
        def local_dispatch_records(self):
            raise RuntimeError("registry unavailable")

    decision = router_helpers.evaluate_session_tenancy(
        _Broken(), "sess-new", tenants=1, idle_seconds=30.0
    )

    assert decision.admitted is True
    assert decision.reason is None


# ---------------------------------------------------------------------------
# C5 — reclaim is a lease: eviction event + snapshot so the client restores warm
# ---------------------------------------------------------------------------

class _LoggingSrvStub(_SrvStub):
    """Server stub whose logger is the real module logger (caplog-visible)."""

    def __init__(self, records=None):
        super().__init__(records)
        import logging

        self.logger = logging.getLogger("llama-proxy")
        self.current_model = "Qwen3"


def _eviction_srv(tmp_path, sessions=("sess-a",), *, slots=2):
    """Server stub with slot persistence enabled for eviction-save tests."""
    srv = _LoggingSrvStub()
    srv.config = {
        "server": {
            "session_slot_save_path": str(tmp_path / "slots"),
            "session_slot_pool_size": slots,
            "session_slot_skip_when_busy": False,
            "session_slot_timeout_seconds": 3.0,
            "llama_server_port": 8080,
        }
    }
    now = time.monotonic()
    for index, name in enumerate(sessions):
        srv.record(name, now=now)
        session_registry._slot_owners[("", index)] = name
    return srv


def test_eviction_saves_slot_snapshot(tmp_path, monkeypatch, caplog):
    """A reclaim persists the evicted session's KV as ``slot_<session>.bin``."""
    import asyncio
    import logging

    from proxy import admission_events
    from proxy import session as session_mod

    calls = []

    async def _fake_save(port, slot_id, filename, timeout, model=None, endpoint=None):
        calls.append((port, slot_id, filename))
        return True

    monkeypatch.setattr(session_mod, "_save_slot_snapshot", _fake_save)
    srv = _eviction_srv(tmp_path)

    async def _run():
        released = await router_helpers._release_prefill_deadline_dispatch(
            srv, "sess-a"
        )
        # Let the detached snapshot task run.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return released

    with caplog.at_level(logging.ERROR, logger="llama-proxy"):
        released = asyncio.run(_run())

    assert released is True
    assert calls == [(8080, 0, str(tmp_path / "slots" / "slot_sess-a.bin"))]
    # The shared eviction event (ERROR, post-acceptance eviction) names the
    # cause and reports that the snapshot was planned.
    events = [r for r in caplog.records if "local_eviction" in r.getMessage()]
    assert events, "the shared eviction event must be emitted"
    assert events[0].levelno == logging.ERROR
    assert events[0].reason in admission_events.EVICTION_REASONS
    assert events[0].snapshot_planned is True


def test_eviction_skips_snapshot_when_persistence_gated(tmp_path, monkeypatch):
    """No slot assignment ⇒ no snapshot attempt (never allocate a slot to save)."""
    from proxy import session as session_mod

    calls = []

    async def _fake_save(*args, **kwargs):
        calls.append(args)
        return True

    monkeypatch.setattr(session_mod, "_save_slot_snapshot", _fake_save)
    srv = _eviction_srv(tmp_path, sessions=())
    srv.local_dispatch_records[router_helpers._dispatch_lease_key(None, "ghost")] = {
        "backend": "local",
        "started_at": time.monotonic(),
        "active": True,
        "expires_at": time.monotonic() + 30.0,
        "last_progress_ts": time.monotonic(),
    }

    planned = router_helpers._record_local_eviction(
        srv, "ghost", reason="lease_evicted"
    )

    assert planned is False
    assert calls == []


def test_eviction_snapshot_respects_pool_size_gate(tmp_path):
    """Persistence gates (here: zero-slot pool) suppress the snapshot plan."""
    from proxy import session as session_mod

    srv = _eviction_srv(tmp_path, slots=0)

    plan = session_mod.plan_eviction_snapshot(
        "sess-a", srv.config["server"], endpoint=None
    )

    assert plan is None


def test_eviction_event_reason_vocabulary(caplog):
    """The eviction event carries a closed reason vocabulary (single convention)."""
    import logging

    from proxy import admission_events

    srv = _LoggingSrvStub()
    with caplog.at_level(logging.ERROR, logger="llama-proxy"):
        for reason in admission_events.EVICTION_REASONS:
            admission_events.log_local_eviction(
                srv,
                "sess-a",
                reason,
                endpoint="http://localhost:8080",
                slot_id=1,
                slot_owner_session="sess-a",
                snapshot_planned=False,
            )

    records = [r for r in caplog.records if "local_eviction" in r.getMessage()]
    assert [r.reason for r in records] == list(admission_events.EVICTION_REASONS)
    assert all(r.levelno == logging.ERROR for r in records)
    assert all(r.event == "local_eviction" for r in records)


def test_tenancy_idle_bound_stays_inside_the_lease_ceiling():
    """The idle bound never becomes the *only* release path (AC6 bounds)."""
    srv = _SrvStub()
    srv.config = {"server": {}}

    ceiling = router_helpers._get_max_prefill_seconds(srv)

    assert router_helpers.DEFAULT_TENANCY_IDLE_SECONDS <= 180.0
    if ceiling > 0:
        assert router_helpers.DEFAULT_TENANCY_IDLE_SECONDS < ceiling


# ---------------------------------------------------------------------------
# C6 — observability: every admission decision is logged and machine-parseable
# ---------------------------------------------------------------------------

def _parse_logged_caught(caplog, needle: str):
    """Parse the first caught log record whose message contains *needle*.

    Uses the real ``proxy-usage-analysis`` parser so the assertion is about the
    operator-visible contract (the report consumes these lines), not about the
    format string.
    """
    import sys
    from pathlib import Path

    scripts = (
        Path(__file__).resolve().parents[2]
        / ".pi/skills/proxy-usage-analysis/scripts"
    )
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import log_parser

    record = next(r for r in caplog.records if needle in r.getMessage())
    line = f"2026-10-25 01:14:37,460 - INFO - {record.getMessage()}"
    event = log_parser.parse_log_line(line)
    assert event is not None, f"unparseable decision line: {line!r}"
    return event


def test_admission_decision_logging_is_parseable(caplog, monkeypatch):
    """A resident admission logs decision, n_past, ratio and slot owner."""
    import logging

    view = _view(
        slots=[_slot(0, EST_TOKENS)], owners={("", 0): "sess-a"},
        monkeypatch=monkeypatch,
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _gate(view=view, monkeypatch=monkeypatch)

    assert (skip, reason) == (False, None)
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.kind == "residency_decision"
    assert event.decision == "admit"
    assert event.session == "sess-a"
    assert event.n_past_actual == EST_TOKENS
    assert event.estimated_tokens == EST_TOKENS
    assert event.reuse_ratio == pytest.approx(1.0)
    assert event.slot_owner_session == "sess-a"
    assert event.residency_source == "measured"


def test_evicted_denial_logging_is_parseable(caplog, monkeypatch):
    """An evicted denial logs the measured cheap-prefix ratio, not a prediction."""
    import logging

    view = _view(
        slots=[_slot(0, N_PAST_COLD)], owners={("", 0): "sess-a"},
        monkeypatch=monkeypatch,
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _gate(view=view, monkeypatch=monkeypatch)

    assert skip is True
    assert reason == SKIP_RESIDENT_PREFIX
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.decision == SKIP_RESIDENT_PREFIX
    assert event.n_past_actual == N_PAST_COLD
    assert event.reuse_ratio == pytest.approx(N_PAST_COLD / EST_TOKENS, abs=1e-3)


def test_tenant_denial_logging_is_parseable(caplog, monkeypatch):
    """The tenancy denial is logged with the bound and the occupant count."""
    import logging

    _install_tenancy(["sess-a", "sess-b"])
    view = _view(slots=[], owners={}, monkeypatch=monkeypatch)

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _gate_with_tenancy(
            session_id="sess-c", view=view, monkeypatch=monkeypatch
        )

    assert skip is True
    assert reason == SKIP_TENANT_LIMIT
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.decision == SKIP_TENANT_LIMIT
    assert event.tenant_limit == 2
    assert event.tenant_occupants == 2


# ---------------------------------------------------------------------------
# LP-0MVNPZY22008B536 — reuse_ratio is logged only when it was measured
# ---------------------------------------------------------------------------

def test_non_resident_denial_logs_no_ratio(caplog, monkeypatch):
    """A session owning no slot has *no* measurement: log ``reuse_ratio=none``.

    Logging the internal 0.0 sentinel fabricated a 0 % measurement, which the
    usage report then counted in its "measured ratio" denominator and averaged
    into the evicted row.
    """
    import logging

    view = _view(
        slots=[_slot(0, EST_TOKENS)],
        owners={("", 0): "sess-other"},
        monkeypatch=monkeypatch,
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _gate(view=view, predicted_ratio=0.99, monkeypatch=monkeypatch)

    assert skip is True
    assert reason == SKIP_RESIDENT_PREFIX
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.decision == SKIP_RESIDENT_PREFIX
    assert event.residency_source == "non_resident"
    assert event.n_past_actual is None
    assert event.reuse_ratio is None, "a non-measurement must not be logged as 0.0"


def test_unknown_state_logs_no_ratio(caplog, monkeypatch):
    """Genuinely unknown state (stale view, no ownership) also logs ``none``.

    LP-0MVN4LANP0069QQM closed the gap this test used to document: the
    fail-open path (no fresh snapshot, no ownership) *does* emit the decision
    line — with ``n_past_actual=none`` and ``reuse_ratio=none`` — so a
    residency-blind admission is visible in the usage report instead of
    vanishing. Asserted through the ladder, not through a direct call.
    """
    import logging

    from proxy.residency import RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS

    view = _view(
        slots=[_slot(0, None)],
        owners={},
        age=RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS + 1.0,
        monkeypatch=monkeypatch,
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _gate(view=view, predicted_ratio=0.99, monkeypatch=monkeypatch)

    assert (skip, reason) == (False, None)
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.residency_source == "unknown"
    assert event.decision == provider.DECISION_ADMIT_PREDICTED
    assert event.n_past_actual is None
    assert event.reuse_ratio is None, "a non-measurement must not be logged as 0.0"


def test_unknown_state_denial_logs_the_predicted_decision(caplog, monkeypatch):
    """A fail-open *denial* is logged too, at INFO with ``n_past_actual=none``.

    AC5 requires every admission/denial decision to be observable; before
    LP-0MVN4LANP0069QQM the residency-blind denial was invisible, so the
    usage report could not distinguish "no residency measurement" from "no
    large prompts at all".
    """
    import logging

    from proxy.residency import RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS

    view = _view(
        slots=[_slot(0, None)],
        owners={},
        age=RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS + 1.0,
        monkeypatch=monkeypatch,
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _gate(view=view, predicted_ratio=0.0, monkeypatch=monkeypatch)

    assert skip is True
    assert reason == "large_context_bypass"
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.residency_source == "unknown"
    assert event.decision == provider.DECISION_SKIP_PREDICTED
    assert event.n_past_actual is None
    assert event.reuse_ratio is None


def test_decision_line_is_logged_at_info(caplog, monkeypatch):
    """The decision line rides INFO — the level the usage report parses.

    Asserted on the fail-open path, which had emitted no line at all.
    """
    import logging

    from proxy.residency import RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS

    view = _view(
        slots=[_slot(0, None)],
        owners={},
        age=RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS + 1.0,
        monkeypatch=monkeypatch,
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        _gate(view=view, predicted_ratio=0.0, monkeypatch=monkeypatch)

    records = [
        r for r in caplog.records if "routing_resident_prefix" in r.getMessage()
    ]
    assert records, "the fail-open decision must be logged"
    assert all(r.levelno == logging.INFO for r in records)


def test_measured_path_still_emits_one_line(caplog, monkeypatch):
    """The fail-open logging never doubles up on the measured path."""
    import logging

    view = _view(
        slots=[_slot(0, EST_TOKENS)], owners={("", 0): "sess-a"},
        monkeypatch=monkeypatch,
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _gate(view=view, monkeypatch=monkeypatch)

    assert (skip, reason) == (False, None)
    records = [
        r for r in caplog.records if "routing_resident_prefix" in r.getMessage()
    ]
    assert len(records) == 1
    assert "decision=admit" in records[0].getMessage()


def test_measured_zero_ratio_is_still_logged_as_zero(caplog, monkeypatch):
    """A genuine measured 0 (slot reports ``context_tokens=0``) stays numeric."""
    import logging

    view = _view(
        slots=[_slot(0, 0)], owners={("", 0): "sess-a"},
        monkeypatch=monkeypatch,
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _gate(view=view, monkeypatch=monkeypatch)

    assert skip is True
    assert reason == SKIP_RESIDENT_PREFIX
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.n_past_actual == 0
    assert event.reuse_ratio == pytest.approx(0.0)
    assert event.reuse_ratio is not None


# ---------------------------------------------------------------------------
# AC2 — a tenancy denial never costs a resident its prefix
# ---------------------------------------------------------------------------

def test_tenancy_denial_does_not_evict_a_resident(monkeypatch, caplog):
    """The (N+1)th session falls through; nobody's lease or slot is reclaimed.

    AC2 requires the denials to follow the normal fall-through chain "not
    evicting a resident" — so the bound must be enforced purely as a routing
    decision. This asserts the residents' dispatch records survive untouched
    and that the shared eviction event stays silent (an eviction here would be
    exactly the 2026-10-24 thrash this work removes).
    """
    import logging

    now = 1000.0
    srv = _SrvStub()
    srv.record("sess-a", now=now)
    srv.record("sess-b", now=now)
    before = {k: dict(v) for k, v in srv.local_dispatch_records.items()}

    with caplog.at_level(logging.ERROR, logger="llama-proxy"):
        decision = router_helpers.evaluate_session_tenancy(
            srv, "sess-c", tenants=2, idle_seconds=30.0, now=now
        )

    assert decision.admitted is False
    assert decision.reason == SKIP_TENANT_LIMIT
    assert srv.local_dispatch_records == before, "a denial must not reclaim a lease"
    assert all(rec["active"] for rec in srv.local_dispatch_records.values())
    assert not [r for r in caplog.records if "local_eviction" in r.getMessage()]


def test_live_resident_is_never_idle_lapsed(monkeypatch):
    """A resident still processing in its slot keeps its tenancy.

    Idle-lapse only frees a *silent* tenancy; a session whose slot is
    ``is_processing`` in the live view is provably active, so it must stay an
    occupant even when its dispatch record looks stale.
    """
    now = 1000.0
    srv = _SrvStub()
    srv.record("sess-a", now=now - 600.0)  # stale lease record
    srv.record("sess-b", now=now)
    view = _view(
        slots=[_slot(0, EST_TOKENS)], owners={("", 0): "sess-a"},
        monkeypatch=monkeypatch,
    )

    decision = router_helpers.evaluate_session_tenancy(
        srv, "sess-c", tenants=2, idle_seconds=30.0, now=now, view=view
    )

    assert decision.admitted is False
    assert "sess-a" in decision.occupants, (
        "a live (is_processing) resident must not be lapsed out"
    )


# ---------------------------------------------------------------------------
# AC1 — the idle override still serves a residency-gate denial
# ---------------------------------------------------------------------------

def test_idle_backend_admits_cold_work_after_a_residency_denial(monkeypatch):
    """End-to-end at the gate: a residency denial is idle-overridable.

    ``local_idle_admit_*`` is the last rung of the ladder: the ladder refuses
    cold work from a non-resident session, and an idle GPU then admits that
    same request locally (composition, not merely reason-list membership).
    """
    view = _view(
        slots=[_slot(0, EST_TOKENS)], owners={("", 0): "sess-other"},
        monkeypatch=monkeypatch,
    )

    skip, reason = _gate(view=view, monkeypatch=monkeypatch)

    assert skip is True
    assert reason == SKIP_RESIDENT_PREFIX
    assert reason in provider.IDLE_OVERRIDABLE_SKIP_REASONS

    monkeypatch.setattr(
        "proxy.local_activity.local_idle_seconds", lambda *a, **k: 600.0
    )
    monkeypatch.setattr(provider, "_local_idle_blocker", lambda *a, **k: None)
    monkeypatch.setattr(
        provider, "_idle_admission_budget_verdict", lambda *a, **k: None
    )

    decision = provider._evaluate_idle_admission_override(
        {"server": {"local_idle_admit_enabled": True,
                    "local_idle_admit_seconds": 60.0}},
        {"llama_model": "Qwen3"},
        EST_TOKENS,
    )

    assert decision is not None, "an idle backend must still admit cold work"
    assert decision["admitted"] is True


# ---------------------------------------------------------------------------
# AC6 — the eviction snapshot obeys the existing persistence gates
# ---------------------------------------------------------------------------

def test_eviction_snapshot_respects_the_busy_gate(tmp_path, monkeypatch):
    """Eviction saves go through ``session_slot_skip_when_busy`` like any save.

    A save issued while another local stream is generating serialises behind
    that work, so the busy gate must apply to the eviction path too — and an
    idle eviction must still produce a plan.
    """
    from proxy import session as session_mod

    srv = _eviction_srv(tmp_path)
    srv.config["server"]["session_slot_skip_when_busy"] = True
    monkeypatch.setattr(session_mod, "_srv", lambda: srv)

    srv.record("sess-b", now=time.monotonic())  # another session streaming
    assert session_mod.plan_eviction_snapshot(
        "sess-a", srv.config["server"], endpoint=None
    ) is None

    srv.local_dispatch_records.clear()
    srv.record("sess-a", now=time.monotonic())  # only the evicted session left
    assert session_mod.plan_eviction_snapshot(
        "sess-a", srv.config["server"], endpoint=None
    ) is not None


def test_local_prefill_deadline_default_is_unchanged():
    """AC6: the prefill deadline stays the pre-existing 90 s default.

    The 90→110 s change belongs to ``LP-0MVN3KMGI009BZ0D``; this item's tenancy
    work must not move it (the deadline is only the lease backstop).
    """
    assert router_helpers._get_prefill_deadline_seconds(_SrvStub()) == 90.0
