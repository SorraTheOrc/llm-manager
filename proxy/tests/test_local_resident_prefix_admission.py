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

import re
import time

import prefill_thrash_replay_harness as replay_harness
import proxy.measured_prefill as measured_prefill
import proxy.observability as observability
import proxy.provider as provider
import proxy.router_helpers as router_helpers
import proxy.session as session_registry
import pytest
from proxy.measured_prefill import MeasuredPrefill

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

#: Model the ladder is driven with (``_gate``); the measurement store is keyed
#: ``(model, session)``, so the constant is part of the pin.
MODEL = "Qwen3"
#: A measured dispatch smaller than the prompt: the rung must report
#: ``min(total, estimated_tokens) / estimated_tokens`` (≈0.61), not the 1.0
#: sentinel, so this fixture distinguishes a real measurement from the guess.
MEASURED_TOTAL = 30_000

#: ``measurement=`` values from the parent's closed vocabulary (AC2).
MEASUREMENT_MEASURED = "measured"
MEASUREMENT_NO_SLOT = "unavailable:no_slot"
MEASUREMENT_NO_PRIOR = "unavailable:no_prior_measurement"
MEASUREMENT_STALE_VIEW = "unavailable:stale_view"
MEASUREMENT_CAPTURE_DISABLED = "unavailable:capture_disabled"


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
    session_registry._slot_last_tenant.clear()
    yield
    observability._last_slot_details_cache[:] = []
    observability._last_slot_details_at = 0.0
    session_registry._slot_owners.clear()
    session_registry._slot_last_tenant.clear()


@pytest.fixture(autouse=True)
def _reset_measured_prefill_state():
    """Keep the in-process measurement store isolated between tests.

    The store is the third documented measurement source the admission rung
    reads (``proxy.measured_prefill``); it is process-global, so every test in
    this module starts and ends with it empty.
    """
    measured_prefill.reset_measured_prefill()
    yield
    measured_prefill.reset_measured_prefill()


def _prime_measurement(
    total: int = MEASURED_TOTAL,
    *,
    cache: int = 0,
    model: str = MODEL,
    session: str = "sess-a",
) -> None:
    """Record a prior measured dispatch for ``(model, session)``."""
    measured_prefill.set_measured_prefill(
        model,
        session,
        MeasuredPrefill(total=total, cache=cache, processed=0, time_ms=1234.0),
    )


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
    model_name: str = MODEL,
    estimated_tokens: int = EST_TOKENS,
    min_ratio: float = 0.5,
    predicted_ratio: float = 0.0,
    view=None,
    cold_threshold: int = COLD_THRESHOLD,
    warm_threshold: int = 0,
    unmeasured_cold_enabled: bool = True,
    monkeypatch: pytest.MonkeyPatch | None = None,
):
    """Run the local admission ladder as the router does for a large prompt."""
    if monkeypatch is not None:
        monkeypatch.setattr(
            provider, "expected_cached_ratio", lambda *a, **k: predicted_ratio
        )
    return provider._should_skip_local(
        model_name,
        session_id,
        {"messages": []},
        cold_threshold,
        estimated_tokens=estimated_tokens,
        warm_cache_threshold=warm_threshold,
        resident_prefix_min_ratio=min_ratio,
        unmeasured_cold_enabled=unmeasured_cold_enabled,
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


def test_unknown_state_denies_cold_work(monkeypatch):
    """No fresh view and no ownership ⇒ the prompt is *unmeasured*, hence cold.

    LP-0MW81DFML006O5KL retired the fail-open fallback here: a stale/missing
    snapshot with no ownership used to be decided by the predicted sidecar
    ratio (the 2026-11-07 55 492-token / ``new_tokens=172`` admission). The
    prediction is now irrelevant — the cold threshold governs and the prompt
    follows the remote chain with an explicit reason.
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

    assert (skip_cached, reason_cached) == (
        True, provider.SKIP_REASON_UNMEASURED_COLD
    )
    assert (skip_cold, reason_cold) == (
        True, provider.SKIP_REASON_UNMEASURED_COLD
    )


# ---------------------------------------------------------------------------
# AC5 rollback — one config flag restores the pre-change decision
# (LP-0MW81DJZ60049X5W)
# ---------------------------------------------------------------------------

#: The single rollback switch for the unmeasured-cold enforcement.
UNMEASURED_COLD_FLAG = "local_unmeasured_residency_cold_enabled"


def _unmeasured_view(monkeypatch):
    """A stale snapshot with no ownership ⇒ prefix residency is ``unknown``."""
    from proxy.residency import RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS

    return _view(
        slots=[_slot(0, None)],
        owners={},
        age=RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS + 1.0,
        monkeypatch=monkeypatch,
    )


def test_unmeasured_cold_rollback_flag_restores_the_predicted_decision(monkeypatch):
    """A 55k unmeasured prompt: cold by default, predicted with the rollback.

    AC2 of LP-0MW81DJZ60049X5W pins both halves: unset (the shipped default)
    keeps the LP-0MW81DFML006O5KL enforcement, and the single rollback flag
    restores the *pre-change* decision byte-for-byte — the 55 492-token /
    ``new_tokens=172`` case that motivated the change is admitted again on the
    predicted ratio.
    """
    view = _unmeasured_view(monkeypatch)

    # Shipped default: nothing was measured, so the cold threshold governs and
    # the predicted sidecar ratio cannot make the prompt cheap.
    assert _gate(
        view=view,
        estimated_tokens=55_000,
        predicted_ratio=1.0,
        monkeypatch=monkeypatch,
    ) == (True, provider.SKIP_REASON_UNMEASURED_COLD)

    # Rollback: one value restores the pre-change predicted-ratio decision,
    # with both of its verdicts (predicted-warm admit / predicted-cold bypass).
    rollback = provider._unmeasured_residency_cold_enabled(
        {"server": {UNMEASURED_COLD_FLAG: False}}
    )
    assert rollback is False
    assert _gate(
        view=view,
        estimated_tokens=55_000,
        predicted_ratio=1.0,
        unmeasured_cold_enabled=rollback,
        monkeypatch=monkeypatch,
    ) == (False, None)
    assert _gate(
        view=view,
        estimated_tokens=55_000,
        predicted_ratio=0.0,
        unmeasured_cold_enabled=rollback,
        monkeypatch=monkeypatch,
    ) == (True, "large_context_bypass")


def test_unmeasured_cold_rollback_restores_the_predicted_log_line(caplog, monkeypatch):
    """The rollback also restores the pre-change decision *line* (``admit_predicted``)."""
    import logging

    view = _unmeasured_view(monkeypatch)
    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, _ = _gate(
            view=view,
            estimated_tokens=55_000,
            predicted_ratio=1.0,
            unmeasured_cold_enabled=False,
            monkeypatch=monkeypatch,
        )

    assert skip is False
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.decision == provider.DECISION_ADMIT_PREDICTED
    assert event.n_past_actual is None
    assert event.residency_source == provider.RESIDENCY_SOURCE_UNKNOWN


def test_unmeasured_cold_flag_config_resolution():
    """The rollback flag defaults on and reads both nested and flat forms."""
    resolve = provider._unmeasured_residency_cold_enabled

    # Default = the intended post-rollout behaviour: enforcement ON.
    assert resolve(None) is True
    assert resolve({}) is True
    assert resolve({"server": {}}) is True

    # Shipped (nested ``server:``) form, as a YAML bool and as 0/1.
    assert resolve({"server": {UNMEASURED_COLD_FLAG: False}}) is False
    assert resolve({"server": {UNMEASURED_COLD_FLAG: 0}}) is False
    assert resolve({"server": {UNMEASURED_COLD_FLAG: True}}) is True

    # Flat form (single-level configs and tests).
    assert resolve({UNMEASURED_COLD_FLAG: False}) is False
    assert resolve({UNMEASURED_COLD_FLAG: "false"}) is False

    # A malformed value keeps the default rather than silently rolling back.
    assert resolve({"server": {UNMEASURED_COLD_FLAG: "nonsense"}}) is True


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

    The newcomer owns a slot with a measured resident prefix, so it passes the
    residency gate and reaches the tenancy rung — the rung that keeps it from
    evicting a resident. A genuinely *unmeasured* newcomer never gets this far:
    it is denied earlier with ``unmeasured_residency_cold``
    (LP-0MW81DFML006O5KL). The newcomer's slot is not ``is_processing``, so it
    is not one of the occupants the bound counts.
    """
    _install_tenancy(["sess-a", "sess-b"])
    view = _view(
        slots=[_slot(0, EST_TOKENS, processing=False)],
        owners={("", 0): "sess-c"},
        monkeypatch=monkeypatch,
    )

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
    view = _view(
        slots=[_slot(0, EST_TOKENS, processing=False)],
        owners={("", 0): "sess-c"},
        monkeypatch=monkeypatch,
    )

    skip, reason = _gate_with_tenancy(
        session_id="sess-c",
        idle_seconds=1.0,
        view=view,
        monkeypatch=monkeypatch,
    )

    # The tenancy rung released the pool and the newcomer's own prefix is
    # measured resident, so the prompt is served locally.
    assert (skip, reason) == (False, None)


def test_tenancy_bound_disabled_with_zero(monkeypatch):
    """``local_session_tenants: 0`` disables the bound in the ladder."""
    _install_tenancy(["sess-a", "sess-b", "sess-c"])
    view = _view(
        slots=[_slot(0, EST_TOKENS, processing=False)],
        owners={("", 0): "sess-d"},
        monkeypatch=monkeypatch,
    )

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

    async def _fake_save(
        port, slot_id, filename, timeout, model=None, endpoint=None, source=None
    ):
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
    # LP-0MVT2PY3J00077W6: provenance is auditable — a live measurement says
    # so, in the additive field that follows ``residency_source=``.
    assert _measurement_field(caplog) == MEASUREMENT_MEASURED


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
    view = _view(
        slots=[_slot(0, EST_TOKENS, processing=False)],
        owners={("", 0): "sess-c"},
        monkeypatch=monkeypatch,
    )

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


def test_unknown_state_denial_logs_its_basis(caplog, monkeypatch):
    """Genuinely unknown state logs ``none`` for the ratio and *why* it is unknown.

    LP-0MVN4LANP0069QQM made residency-blind decisions observable;
    LP-0MW81DFML006O5KL changed their verdict from an admit to an explicit
    ``unmeasured_residency_cold`` denial. The line therefore records the basis
    (``residency_source=unknown`` / ``measurement=unavailable:stale_view``)
    and the decision, so any unmeasured 55k skip is explainable from the log,
    and no ``reuse_ratio`` is fabricated for a non-measurement
    (LP-0MVNPZY22008B536). Asserted through the ladder, not through a direct
    call.
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

    assert (skip, reason) == (True, provider.SKIP_REASON_UNMEASURED_COLD)
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.residency_source == "unknown"
    assert event.decision == provider.SKIP_REASON_UNMEASURED_COLD
    assert event.n_past_actual is None
    assert event.reuse_ratio is None, "a non-measurement must not be logged as 0.0"


def test_decision_line_is_logged_at_info(caplog, monkeypatch):
    """The decision line rides INFO — the level the usage report parses.

    Asserted on the unmeasured-cold denial, the path whose verdict changed in
    LP-0MW81DFML006O5KL.
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
        _gate(view=view, predicted_ratio=0.99, monkeypatch=monkeypatch)

    records = [
        r for r in caplog.records if "routing_resident_prefix" in r.getMessage()
    ]
    assert records, "the unmeasured-cold decision must be logged"
    assert all(r.levelno == logging.INFO for r in records)


def test_measured_path_still_emits_one_line(caplog, monkeypatch):
    """The unmeasured-cold logging never doubles up on the measured path."""
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


# ---------------------------------------------------------------------------
# AC2 — the thrash cooldown and the tenancy bound compose (never widen)
# ---------------------------------------------------------------------------

def test_thrash_cooldown_and_tenancy_bound_compose():
    """A tenancy-denied session under thrash cooldown sees both gates.

    The routing loop checks the post-deadline thrash cooldown
    (``LP-0MVKUKFHH005UDWL``) *before* the ladder, so the cooldown must not
    widen admission: a session whose tenancy is denied stays denied while its
    cooldown remains armed, on its own grounds.
    """
    from proxy import router_helpers as rh

    rh._reset_prefill_thrash_cooldowns()
    try:
        at = 1000.0  # cooldown-arming instant
        ev = at + 60  # evaluation instant: records must still be live (TTL 30 s)
        srv = _SrvStub()
        srv.record("sess-a", now=ev)
        srv.record("sess-b", now=ev)

        # A cooldown armed just before (deadline far in the future)…
        rh._record_prefill_thrash_cooldown(
            "sess-c",
            {"server": {"local_prefill_thrash_cooldown_seconds": 600}},
            now=at,
        )
        assert rh._prefill_thrash_cooldown_remaining("sess-c", now=at + 60) > 0

        # …does not widen admission: the tenancy bound still denies a third
        # session on its own grounds while the cooldown stays armed.
        decision = rh.evaluate_session_tenancy(
            srv, "sess-c", tenants=2, idle_seconds=30.0, now=ev
        )
        assert decision.admitted is False
        assert decision.reason == SKIP_TENANT_LIMIT
    finally:
        rh._reset_prefill_thrash_cooldowns()


# ---------------------------------------------------------------------------
# AC5 — a single eviction logging convention (LP-0MVN3KMGI009BZ0D AC7)
# ---------------------------------------------------------------------------

def test_single_eviction_logging_convention():
    """Every reclaim path funnels through ``admission_events`` — one sink.

    The one-logging-convention rule forbids a second, ad-hoc eviction line:
    whatever reason text a reclaim path wants, it must reach
    ``admission_events.log_local_eviction``. This structural contract pins the
    routing so a future path cannot invent a new format silently.
    """
    import inspect
    import re

    from proxy import admission_events, router_helpers

    src = inspect.getsource(router_helpers)
    # The shared sinks, by name, are the only eviction emitters the reclaim
    # module may use.
    sinks = sorted(
        set(re.findall(r"(?:log_local_eviction|record_eviction_after_acceptance)", src))
    )
    assert sinks == ["log_local_eviction", "record_eviction_after_acceptance"], (
        f"reclaim paths must use the shared admission_events sinks only, "
        f"found: {sinks}"
    )
    # …and every import of an *eviction* function comes from
    # proxy.admission_events — never a second module defining its own format.
    eviction_imports = re.findall(
        r"from proxy\.([\w.]+) import ([\w, ]+)", src
    )
    stray = [
        m for m in eviction_imports
        if "eviction" in m[1].lower() and "admission_events" not in m[0]
    ]
    assert not stray, f"eviction emitters must come from admission_events only: {stray}"
    assert callable(admission_events.log_local_eviction)


def test_tenancy_denial_leaves_the_contention_queue_budget_alone(monkeypatch):
    """A tenancy denial is an ordinary skip: the queue budget is untouched.

    AC2 requires the denial to follow the normal fall-through chain "not
    evicting a resident" and to spend the ``contention_queue_*`` budget exactly
    as the pre-existing bypass does — i.e. the ladder must not enter (or
    consume) the queue at all when it denies on tenancy grounds.
    """
    from proxy import router as router_mod

    config = {"server": {"contention_queue_max_wait_seconds": 60.0}}
    before = dict(router_mod._get_contention_queue_config(config["server"]))

    def _boom(*_a, **_k):  # pragma: no cover - must never run
        raise AssertionError("the admission ladder must not consult the queue")

    _install_tenancy(["sess-a", "sess-b"])
    view = _view(
        slots=[_slot(0, EST_TOKENS, processing=False)],
        owners={("", 0): "sess-c"},
        monkeypatch=monkeypatch,
    )
    monkeypatch.setattr(provider, "evaluate_admission", _boom, raising=False)
    monkeypatch.setattr("proxy.prefill_eta.evaluate_admission", _boom)

    skip, reason = _gate_with_tenancy(
        session_id="sess-c", view=view, monkeypatch=monkeypatch
    )

    assert skip is True
    assert reason == SKIP_TENANT_LIMIT
    after = router_mod._get_contention_queue_config(config["server"])
    assert after == before, "a tenancy denial must not change the queue budget"


def test_lifetime_ceiling_releases_a_lease_that_keeps_reporting_progress():
    """The 900 s ceiling bounds a lease the no-progress watchdog cannot.

    A wedged slot can keep reporting progress/is_processing forever, so the
    lease TTL refresh is bounded from above by ``started_at`` + ceiling: the
    900 s ceiling is never optional, even when every other bound is satisfied.
    """
    srv = _SrvStub()
    now = time.monotonic()
    record = {
        "backend": "local",
        "active": True,
        "started_at": now - 901.0,
        "last_progress_ts": now - 1.0,  # fresh: the no-progress watchdog is silent
        "expires_at": now + 3600.0,
    }
    assert router_helpers._watchdog_no_progress(srv, "sess-a", record) is True

    # Inside the ceiling the same well-refreshed lease stays live.
    record["started_at"] = now - 899.0
    assert router_helpers._watchdog_no_progress(srv, "sess-a", record) is False


# ---------------------------------------------------------------------------
# LP-0MVT2PB6U007PKD1 — admission-time measurement (parent AC3 (i)–(iv))
#
# The parent defect (LP-0MVRZXPH70025ZNR) is that the deployed llama.cpp b8782
# build reports no per-slot ``context_tokens``, so rung 1 of
# ``resolve_resident_prefix`` is unreachable in production and every
# ``routing_resident_prefix`` line logs ``n_past_actual=none`` /
# ``reuse_ratio=none``. The fix measures from the in-process
# ``proxy.measured_prefill`` store (llama-server's own ``prompt_progress``
# ``total`` for the session's last dispatch) and labels every non-measurement
# with a closed ``measurement=`` reason instead of a bare ``none``.
#
# These pins are red-first: they are marked ``xfail(strict=True)`` and each
# marker names the implementation child that turns it green, so the tree stays
# green for the dev-push gate while the behaviour is pinned before any
# production code changes.
# ---------------------------------------------------------------------------


def _measurement_field(caplog, needle: str = "routing_resident_prefix") -> str | None:
    """The ``measurement=`` value on the first caught decision line.

    Read from the raw message rather than ``log_parser``: the log grammar
    consumer (``measurement=`` parsing and the report breakdown) is pinned by
    the sibling test item LP-0MVT2PF1C008LAPB, while these tests pin the
    admission behaviour and provenance.
    """
    record = next(r for r in caplog.records if needle in r.getMessage())
    match = re.search(r"\bmeasurement=(\S+)", record.getMessage())
    return match.group(1) if match else None


def test_session_slot_admission_uses_the_last_measured_dispatch(caplog, monkeypatch):
    """AC3(i): ownership + a prior measurement ⇒ a *numeric* reusable prefix.

    The slot reports no ``context_tokens`` (the deployed b8782 case), so the
    ladder must consult the ``(model, session)`` measurement store and report
    the last dispatch's ``total`` as ``n_past_actual`` with
    ``reuse_ratio = min(total, estimated_tokens) / estimated_tokens``. The
    internal 1.0 gate sentinel must not be logged as if it were measured.
    """
    import logging

    _prime_measurement(MEASURED_TOTAL)

    # The store is an in-process measurement source: the rung must not reach
    # for ``/slots`` (llama-server holds it >15 s while prefilling, so an
    # upstream query on the admission path is forbidden — LP-0MUXTW4MD003TVIC).
    def _boom(*_args, **_kwargs):
        raise AssertionError("/slots must not be queried on the admission path")

    monkeypatch.setattr(observability, "_query_slots_detail", _boom)
    monkeypatch.setattr(observability, "_query_slots_progress", _boom)

    view = _view(
        slots=[_slot(0, None)], owners={("", 0): "sess-a"}, monkeypatch=monkeypatch
    )
    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _gate(view=view, monkeypatch=monkeypatch)

    assert (skip, reason) == (False, None)
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.decision == "admit"
    assert event.residency_source == "measured_last_dispatch"
    assert event.n_past_actual == MEASURED_TOTAL
    assert event.reuse_ratio == pytest.approx(
        min(MEASURED_TOTAL, EST_TOKENS) / EST_TOKENS
    )
    assert _measurement_field(caplog) == MEASUREMENT_MEASURED


def test_session_slot_admission_without_a_measurement_is_labelled(
    caplog, monkeypatch
):
    """AC3(ii): a resident admission with no prior measurement is explicit.

    The session owns its slot but nothing has been captured for the pair, so
    the decision must carry ``measurement=unavailable:no_prior_measurement``
    and ``reuse_ratio=none`` — never a numeric ratio (the 1.0 sentinel is an
    internal gate value, not a measurement).
    """
    import logging

    view = _view(
        slots=[_slot(0, None)], owners={("", 0): "sess-a"}, monkeypatch=monkeypatch
    )
    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _gate(view=view, monkeypatch=monkeypatch)

    assert (skip, reason) == (False, None)
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.residency_source == "session_slot"
    assert event.decision == "admit"
    assert event.n_past_actual is None
    assert event.reuse_ratio is None
    assert _measurement_field(caplog) == MEASUREMENT_NO_PRIOR


def test_non_resident_decision_is_labelled_no_slot(caplog, monkeypatch):
    """AC3(iii): a non-resident decision is labelled ``unavailable:no_slot``.

    A fresh view shows the pool holding another session's prefix, so this
    session's reusable prefix is genuinely unmeasurable — not zero. A
    measurement left over from an earlier residency must not be reused either
    (the stale-measurement failure mode the parent exists to fix), and the
    LP-0MVNPZY22008B536 guarantee still holds: no fabricated
    ``reuse_ratio=0.0000``.
    """
    import logging

    _prime_measurement(MEASURED_TOTAL, session="sess-a")
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
    assert event.residency_source == "non_resident"
    assert event.n_past_actual is None
    assert event.reuse_ratio is None, "a non-measurement must not be logged as 0.0"
    assert _measurement_field(caplog) == MEASUREMENT_NO_SLOT
    record = next(
        r for r in caplog.records if "routing_resident_prefix" in r.getMessage()
    )
    assert "reuse_ratio=0.0000" not in record.getMessage()


def test_eviction_invalidates_a_stored_measurement(caplog, monkeypatch):
    """AC3(iv) eviction: a reclaimed slot's measurement is not reused.

    ``_record_local_eviction`` is the single reclaim chokepoint, so clearing
    the measurement there covers every eviction reason. The slot registry
    still names the session afterwards here, which is exactly the
    stale-measurement case: ownership alone must not resurrect the old
    dispatch's ``total`` as a numeric ratio.

    The chokepoint knows the session but not the routed model name, so the
    clear is model-agnostic: a measurement stored under *any* model for the
    session must go (a partial clear would leave an orphan the next dispatch
    could read as warm).
    """
    import logging

    from proxy import admission_events

    _prime_measurement(MEASURED_TOTAL)
    _prime_measurement(MEASURED_TOTAL, model="Qwen3-Next")
    srv = _SrvStub()
    router_helpers._record_local_eviction(
        srv,
        "sess-a",
        reason=admission_events.EVICTION_RESIDENT_PREFIX_EVICTED,
        endpoint=None,
    )

    assert measured_prefill.get_measured_prefill(MODEL, "sess-a") is None
    assert measured_prefill.get_measured_prefill("Qwen3-Next", "sess-a") is None

    view = _view(
        slots=[_slot(0, None)], owners={("", 0): "sess-a"}, monkeypatch=monkeypatch
    )
    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        _gate(view=view, monkeypatch=monkeypatch)

    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.n_past_actual is None
    assert event.reuse_ratio is None
    assert (_measurement_field(caplog) or "").startswith("unavailable:")


def test_same_session_slot_release_keeps_the_measurement():
    """AC1d (survive half): a same-session release keeps the measurement.

    Every local dispatch ends with the slot handed back, so clearing on the
    plain release made the session's next admission unmeasurable
    (``unavailable:no_slot`` on ~75 % of local admissions). The measurement
    survives the release and the same session's re-assignment, so its next
    turn is priced from the last measured dispatch instead of the predicted
    sidecar ratio (LP-0MW959P9S005DJY7 AC1d).
    """
    _prime_measurement(MEASURED_TOTAL, session="sess-a")
    _prime_measurement(MEASURED_TOTAL, model="Qwen3-Next", session="sess-a")
    _prime_measurement(MEASURED_TOTAL, session="sess-b")

    slot_id = session_registry._slot_id_for_session("sess-a", 2)
    session_registry._free_slot_assignment("sess-a")

    # Survival across the release, for every model the session was measured
    # under (the release path knows neither the routed model nor the tenant).
    assert measured_prefill.get_measured_prefill(MODEL, "sess-a") is not None
    assert (
        measured_prefill.get_measured_prefill("Qwen3-Next", "sess-a") is not None
    )
    # Other sessions' measurements are untouched (session-scoped bookkeeping).
    assert measured_prefill.get_measured_prefill(MODEL, "sess-b") is not None

    # The same session re-entering its slot is not a change of hands.
    assert session_registry._slot_id_for_session("sess-a", 2) == slot_id
    assert measured_prefill.get_measured_prefill(MODEL, "sess-a") is not None


def test_slot_taken_by_a_different_session_forgets_the_previous_tenant():
    """AC1d (never inherit half): a new tenant clears the previous tenant.

    A slot that changes hands is exactly where the old measurement becomes a
    stale ratio the new owner must not inherit — the reclaimed slot's previous
    tenant loses its total, and the new tenant has none of its own. Asserted in
    both directions.
    """
    _prime_measurement(MEASURED_TOTAL, session="sess-a")
    _prime_measurement(MEASURED_TOTAL, model="Qwen3-Next", session="sess-a")

    slot_id = session_registry._slot_id_for_session("sess-a", 2)
    session_registry._free_slot_assignment("sess-a")
    assert measured_prefill.get_measured_prefill(MODEL, "sess-a") is not None

    # A different session takes the released slot: the previous tenant's
    # measurement is invalidated (model-agnostic), and the new tenant inherits
    # nothing.
    assert session_registry._slot_id_for_session("sess-b", 2) == slot_id
    assert measured_prefill.get_measured_prefill(MODEL, "sess-a") is None
    assert measured_prefill.get_measured_prefill("Qwen3-Next", "sess-a") is None
    assert measured_prefill.get_measured_prefill(MODEL, "sess-b") is None


def test_model_switch_does_not_reuse_the_previous_models_measurement(caplog, monkeypatch):
    """AC3(iv) model switch: the measurement is scoped to the model.

    The store key is ``(model, session)``, so a switch must leave the new model
    unmeasured. The owning model is checked first: that control is what makes
    this a red-first pin (today neither model reports a numeric ratio), and it
    stops a "always none" implementation from satisfying the switched case.
    """
    import logging

    _prime_measurement(MEASURED_TOTAL, model=MODEL)
    view = _view(
        slots=[_slot(0, None)], owners={("", 0): "sess-a"}, monkeypatch=monkeypatch
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        _gate(view=view, monkeypatch=monkeypatch)

    owned = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert owned.n_past_actual == MEASURED_TOTAL
    assert owned.residency_source == "measured_last_dispatch"

    caplog.clear()
    view = _view(
        slots=[_slot(0, None)], owners={("", 0): "sess-a"}, monkeypatch=monkeypatch
    )
    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        _gate(model_name="Qwen3-Next", view=view, monkeypatch=monkeypatch)

    switched = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert switched.n_past_actual is None
    assert switched.reuse_ratio is None
    assert _measurement_field(caplog) == MEASUREMENT_NO_PRIOR


# ---------------------------------------------------------------------------
# LP-0MVT2Q1SD009JLKD — the closed unmeasured-reason vocabulary
#
# A decision that *cannot* be measured must say **why**, so "the ladder cannot
# measure" (no prior measurement / capture rolled back) is distinguishable from
# "there is no slot to measure" (no_slot) and from "the residency view is
# absent or stale" (stale_view). Every ``routing_resident_prefix`` line carries
# exactly one ``measurement=`` value drawn from module-level constants, with
# deterministic precedence:
#
#   measured  >  capture_disabled  >  no_prior_measurement  >  no_slot  >  stale_view
#
# ``measured`` always wins (a measurement needs no reason); for a slot-owning
# session the capture rollback wins over a missing prior measurement; ``no_slot``
# needs a *fresh* view (otherwise the honest answer is ``stale_view``).
# ---------------------------------------------------------------------------

#: Every vocabulary value paired with the trigger that produces it. The table
#: is also the coverage statement checked below: the implementation may not add
#: (or drop) a reason without extending this table.
VOCABULARY_TRIGGERS = (
    ("measured_live_slot", MEASUREMENT_MEASURED),
    ("measured_last_dispatch", MEASUREMENT_MEASURED),
    ("capture_disabled", MEASUREMENT_CAPTURE_DISABLED),
    ("no_prior_measurement", MEASUREMENT_NO_PRIOR),
    ("no_slot", MEASUREMENT_NO_SLOT),
    ("stale_view", MEASUREMENT_STALE_VIEW),
)


def _run_vocabulary_trigger(case, caplog, monkeypatch):
    """Install the state *case* names and run one admission decision."""
    import logging

    if case == "capture_disabled":
        import proxy.server as server_module

        # The documented rollback: capture is off, so a slot-owning session
        # must say *why* it has no measurement (capture, not absence).
        monkeypatch.setattr(
            server_module,
            "config",
            {"server": {"local_measured_cache_abort_enabled": 0}},
        )
        _prime_measurement(MEASURED_TOTAL)
        view = _view(
            slots=[_slot(0, None)], owners={("", 0): "sess-a"}, monkeypatch=monkeypatch
        )
        kwargs = {}
    elif case == "no_prior_measurement":
        view = _view(
            slots=[_slot(0, None)], owners={("", 0): "sess-a"}, monkeypatch=monkeypatch
        )
        kwargs = {}
    elif case == "measured_last_dispatch":
        _prime_measurement(MEASURED_TOTAL)
        view = _view(
            slots=[_slot(0, None)], owners={("", 0): "sess-a"}, monkeypatch=monkeypatch
        )
        kwargs = {}
    elif case == "measured_live_slot":
        view = _view(
            slots=[_slot(0, EST_TOKENS)], owners={("", 0): "sess-a"}, monkeypatch=monkeypatch
        )
        kwargs = {}
    elif case == "no_slot":
        _prime_measurement(MEASURED_TOTAL, session="sess-a")
        view = _view(
            slots=[_slot(0, EST_TOKENS)],
            owners={("", 0): "sess-other"},
            monkeypatch=monkeypatch,
        )
        kwargs = {"predicted_ratio": 0.99}
    elif case == "stale_view":
        from proxy.residency import RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS

        # No fresh snapshot and no ownership: the residency view is absent, so
        # the honest reason is staleness, not "no slot".
        view = _view(
            slots=[_slot(0, EST_TOKENS)],
            owners={},
            age=RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS + 1.0,
            monkeypatch=monkeypatch,
        )
        kwargs = {"predicted_ratio": 0.0}
    else:  # pragma: no cover - the table above is exhaustive
        raise AssertionError(f"unhandled vocabulary trigger: {case}")

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        _gate(view=view, monkeypatch=monkeypatch, **kwargs)


@pytest.mark.parametrize("case,expected", VOCABULARY_TRIGGERS)
def test_every_vocabulary_value_has_exactly_one_trigger(
    case, expected, caplog, monkeypatch
):
    """AC2: each decision logs exactly one closed-vocabulary reason (AC2).

    One admission per trigger: the line must carry exactly one
    ``measurement=`` token (never zero, never two) and its value must be the
    constant the trigger stands for.
    """
    _run_vocabulary_trigger(case, caplog, monkeypatch)

    message = next(
        r.getMessage()
        for r in caplog.records
        if "routing_resident_prefix" in r.getMessage()
    )
    assert message.count("measurement=") == 1
    assert _measurement_field(caplog) == expected


def test_vocabulary_table_covers_the_closed_vocabulary():
    """The table above exercises every constant the provider may emit.

    A new value in ``provider.MEASUREMENT_VOCABULARY`` without a trigger here
    (or a trigger for a value the provider cannot emit) fails, so the
    vocabulary cannot silently widen.
    """
    assert {expected for _, expected in VOCABULARY_TRIGGERS} == set(
        provider.MEASUREMENT_VOCABULARY
    )


def test_capture_rollback_wins_over_a_missing_prior_measurement(monkeypatch):
    """Precedence: for a slot owner, ``capture_disabled`` beats ``no_prior``.

    With capture on, the same state reports ``no_prior_measurement``; with the
    rollback off it must report ``capture_disabled``, so an operator can tell
    "the feature is off" from "no dispatch was ever measured".
    """
    import logging

    enabled = provider.measurement_reason(
        provider.RESIDENCY_SOURCE_SESSION_SLOT, view_known=True, capture_enabled=True
    )
    disabled = provider.measurement_reason(
        provider.RESIDENCY_SOURCE_SESSION_SLOT, view_known=True, capture_enabled=False
    )

    assert enabled == MEASUREMENT_NO_PRIOR
    assert disabled == MEASUREMENT_CAPTURE_DISABLED
    # A measured source is never re-labelled by the rollback switch.
    assert (
        provider.measurement_reason(
            provider.RESIDENCY_SOURCE_MEASURED_LAST_DISPATCH,
            view_known=True,
            capture_enabled=False,
        )
        == MEASUREMENT_MEASURED
    )
    # And a fresh view with no slot is a *measured* zero prefix, never
    # ``stale_view``; only an absent/stale view reports staleness.
    assert (
        provider.measurement_reason(
            provider.RESIDENCY_SOURCE_NON_RESIDENT, view_known=True
        )
        == MEASUREMENT_NO_SLOT
    )
    assert (
        provider.measurement_reason(
            provider.RESIDENCY_SOURCE_UNKNOWN, view_known=False
        )
        == MEASUREMENT_STALE_VIEW
    )


def test_tenancy_decision_without_residency_facts_is_still_labelled(
    caplog, monkeypatch
):
    """A decision logged without residency facts still carries one reason.

    With the residency gate disabled (``local_resident_prefix_min_ratio: 0``)
    the tenancy rung logs with ``facts=None``. The line must still carry
    exactly one vocabulary value instead of no provenance at all:
    ``unavailable:stale_view`` — no fresh residency view informed it.
    """
    import logging

    _install_tenancy(["sess-a", "sess-b"])

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _gate_with_tenancy(
            session_id="sess-c",
            view=_view(slots=[], owners={}, monkeypatch=monkeypatch),
            min_ratio=0.0,
            monkeypatch=monkeypatch,
        )

    assert skip is True
    assert reason == SKIP_TENANT_LIMIT
    message = next(
        r.getMessage()
        for r in caplog.records
        if "routing_resident_prefix" in r.getMessage()
    )
    assert message.count("measurement=") == 1
    assert _measurement_field(caplog) == MEASUREMENT_STALE_VIEW


# ---------------------------------------------------------------------------
# LP-0MVT2PY3J00077W6 — the measured-last-dispatch rung's guard rails
#
# The rung is the only place a *measurement* (rather than the ownership
# sentinel) drives the gate, so its guards are pinned one by one: it may only
# fire for a session that still owns a slot, only while capture is enabled,
# only for a usable ``total``, and never by raising into the request path.
# Anything else must degrade to the pre-change ``session_slot`` behaviour.
# ---------------------------------------------------------------------------


def _assert_degraded_to_session_slot(
    caplog, *, measurement: str = MEASUREMENT_NO_PRIOR
) -> None:
    """The rung did not fire: ownership only, no fabricated measurement.

    The decision still carries exactly one closed-vocabulary ``measurement=``
    reason (LP-0MVT2Q1SD009JLKD): ``unavailable:no_prior_measurement`` when
    nothing was captured for the pair, ``unavailable:capture_disabled`` when
    the capture rollback is off.
    """
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.residency_source == "session_slot"
    assert event.n_past_actual is None
    assert event.reuse_ratio is None
    assert _measurement_field(caplog) == measurement


@pytest.mark.parametrize(
    "total",
    [None, 0, -1, float("nan"), float("inf")],
    ids=["none", "zero", "negative", "nan", "inf"],
)
def test_unusable_measured_total_degrades_to_session_slot(
    caplog, monkeypatch, total
):
    """AC fail-open: ``None``/``0``/non-finite totals are not measurements."""
    import logging

    measured_prefill.set_measured_prefill(
        MODEL,
        "sess-a",
        MeasuredPrefill(total=total, cache=None, processed=None, time_ms=None),
    )
    view = _view(
        slots=[_slot(0, None)], owners={("", 0): "sess-a"}, monkeypatch=monkeypatch
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _gate(view=view, monkeypatch=monkeypatch)

    assert (skip, reason) == (False, None)
    _assert_degraded_to_session_slot(caplog)


def test_measured_rung_is_not_used_when_the_session_owns_no_slot(
    caplog, monkeypatch
):
    """Ownership is the guard: a stale measurement must not resurrect a ratio.

    The session has a captured dispatch but the pool now holds another
    session's prefix, so the ladder must report the fresh-view non-resident
    measurement fact rather than the stale ``total``.
    """
    import logging

    _prime_measurement(MEASURED_TOTAL, session="sess-a")
    view = _view(
        slots=[_slot(0, EST_TOKENS)],
        owners={("", 0): "sess-other"},
        monkeypatch=monkeypatch,
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _gate(view=view, monkeypatch=monkeypatch)

    assert skip is True
    assert reason == SKIP_RESIDENT_PREFIX
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.residency_source == "non_resident"
    assert event.n_past_actual is None
    assert event.reuse_ratio is None


def test_measured_rung_is_guarded_by_the_capture_rollback(caplog, monkeypatch):
    """The feature switch restores the pre-change rung end to end."""
    import logging

    import proxy.server as server_module

    monkeypatch.setattr(
        server_module,
        "config",
        {"server": {"local_measured_cache_abort_enabled": 0}},
    )
    _prime_measurement(MEASURED_TOTAL)
    view = _view(
        slots=[_slot(0, None)], owners={("", 0): "sess-a"}, monkeypatch=monkeypatch
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _gate(view=view, monkeypatch=monkeypatch)

    assert (skip, reason) == (False, None)
    _assert_degraded_to_session_slot(
        caplog, measurement=MEASUREMENT_CAPTURE_DISABLED
    )


def test_measured_rung_degrades_when_the_store_raises(caplog, monkeypatch):
    """An unreadable store is "no measurement", never a request-path error."""
    import logging

    def _boom(*_args, **_kwargs):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(measured_prefill, "get_measured_prefill", _boom)
    view = _view(
        slots=[_slot(0, None)], owners={("", 0): "sess-a"}, monkeypatch=monkeypatch
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _gate(view=view, monkeypatch=monkeypatch)

    assert (skip, reason) == (False, None)
    _assert_degraded_to_session_slot(caplog)


def test_measured_rung_degrades_when_the_config_cannot_be_read(
    caplog, monkeypatch
):
    """An unreadable config falls back to today's rung, not to a measurement.

    The rollback switch cannot be read, so capture resolves to *not enabled*
    (the pre-change behaviour) — and the line says so with
    ``unavailable:capture_disabled`` rather than claiming nothing was measured.
    """
    import logging

    import proxy.server as server_module

    class _BrokenConfig(dict):
        def get(self, *_args, **_kwargs):
            raise RuntimeError("config unavailable")

    monkeypatch.setattr(server_module, "config", _BrokenConfig())
    _prime_measurement(MEASURED_TOTAL)
    view = _view(
        slots=[_slot(0, None)], owners={("", 0): "sess-a"}, monkeypatch=monkeypatch
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _gate(view=view, monkeypatch=monkeypatch)

    assert (skip, reason) == (False, None)
    _assert_degraded_to_session_slot(
        caplog, measurement=MEASUREMENT_CAPTURE_DISABLED
    )


def test_grown_prompt_is_denied_on_the_measured_ratio(caplog, monkeypatch):
    """A grown prompt's *measured* reuse is below the gate ⇒ deny (AC4 rule (a)).

    ``MEASURED_TOTAL`` is 30k against a 100k prompt (ratio 0.30 < 0.5), so the
    decision is a denial with a numeric ratio — the signal AC4 rule (a) needs
    and that the ownership sentinel (implicit ratio 1.0) could never produce.
    """
    import logging

    _prime_measurement(MEASURED_TOTAL)
    view = _view(
        slots=[_slot(0, None)], owners={("", 0): "sess-a"}, monkeypatch=monkeypatch
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _gate(
            view=view, estimated_tokens=100_000, cold_threshold=60_000,
            monkeypatch=monkeypatch,
        )

    assert skip is True
    assert reason == SKIP_RESIDENT_PREFIX
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.decision == SKIP_RESIDENT_PREFIX
    assert event.residency_source == "measured_last_dispatch"
    assert event.n_past_actual == MEASURED_TOTAL
    assert event.reuse_ratio == pytest.approx(MEASURED_TOTAL / 100_000)
    assert _measurement_field(caplog) == MEASUREMENT_MEASURED


def test_pricing_uses_the_measured_prefix_not_the_ownership_sentinel(
    caplog, monkeypatch
):
    """AC: pricing reads ``min(n_past_actual, estimated_tokens)`` reusable.

    The measured dispatch only explains 30k of a 49k prompt, so 19k new tokens
    exceed a 10k cold cap and local is bypassed. The old ``session_slot``
    rung's implicit "everything is reusable" assumption would have priced this
    at zero new tokens and admitted it.
    """
    import logging

    _prime_measurement(MEASURED_TOTAL)
    view = _view(
        slots=[_slot(0, None)], owners={("", 0): "sess-a"}, monkeypatch=monkeypatch
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _gate(view=view, cold_threshold=10_000, monkeypatch=monkeypatch)

    assert skip is True
    assert reason == "large_context_bypass"
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.decision == "large_context_bypass"
    assert event.n_past_actual == MEASURED_TOTAL
    assert event.reuse_ratio == pytest.approx(MEASURED_TOTAL / EST_TOKENS)

# ---------------------------------------------------------------------------
# F1 — candidate-first admission harness and five-scenario fixtures
# (LP-0MW959KH8008FL42, parent AC5)
#
# This item pins the candidate-first admission contract *before* any
# implementation lands. The harness reuses the residency-ladder fixtures above
# (``_install_slot_state`` / ``_slot`` / ``_view`` / ``_prime_measurement`` /
# ``_measurement_field`` / ``_parse_logged_caught``) — no second ladder.
#
# Pinned production seam (owned by F2–F4; absent today, so every scenario
# below is red-first):
#
#   proxy/session.py
#     @dataclass(frozen=True)
#     class SlotCandidate:
#         session_id: str
#         slot_id: int
#         endpoint: str = ""
#
#     def acquire_slot_candidate(session_id, *, endpoint=None,
#                                pool_size=None) -> SlotCandidate | None:
#         """Lowest free slot for the session on the endpoint; same-session
#         re-entry idempotent; None when the pool is full."""
#
#     def release_slot_candidate(candidate, *, reason=None) -> bool:
#         """Idempotent release through the existing release seam
#         (``_free_slot_assignment``); never a double-free. When the release
#         closes a *refused* admission it names the refusing reason
#         (``candidate_released reason=<skip_reason>``, F3 AC2b); a second
#         release returns False and logs nothing (F3 AC2e)."""
#
# ``acquire_slot_candidate`` shipped with F2; ``release_slot_candidate`` and
# the release contract at the admission call site shipped with F3
# (LP-0MW959PWL001WYSM).
#
# Accepted on every refusal rung: ``context_too_large``,
# ``unmeasured_residency_cold``, ``resident_prefix_insufficient``,
# ``tenant_limit``, ``large_context_bypass``, the contention-queue refusal and
# any exception raised inside the decision (F3 AC2c).
#
# The admission call site (F2) acquires the candidate **before** the residency
# view is materialised, threads it into the ladder, and releases it on every
# refusal (F3). Residency resolution needs no new input: the acquire registers
# ownership in ``session._slot_owners``, the registry the residency rungs
# already read, so the measurement is of the candidate slot.
#
# Markers: each pending scenario is ``xfail(strict=True)`` naming the owning
# item (F3–F5). Strict xfail means the implementation child **must delete the
# marker** when its code lands — a silently-passing pin fails the gate instead
# of rotting (same discipline the file used for the red-first
# LP-0MVT2PB6U007PKD1 pins). The implemented scenarios move from
# ``SCENARIO_OWNERS`` into ``IMPLEMENTED_SCENARIOS`` in the same edit that
# removes their marker (F2, LP-0MW959P9S005DJY7);
# ``test_f1_scenario_owners_are_pinned`` audits that no scenario is dropped,
# left un-pinned, or passes while still claiming to be pending (AC5d).
# ---------------------------------------------------------------------------

#: Owning implementation items (parent's F-plan children).
F2_ASSIGN = "LP-0MW959P9S005DJY7"  # assign the candidate before the verdict
F3_RELEASE = "LP-0MW959PWL001WYSM"  # release cleanly on a refused admission
F4_FALLBACK = "LP-0MW959QJ9002MKEI"  # exclusivity + deterministic busy fallback

#: Pending ``test_scenario*`` → owning item(s). The audit test below reads this
#: map, so adding a scenario without naming its owner fails the suite (AC5d).
#: Empty today: scenario 2 went green with F4 (LP-0MW959QJ9002MKEI), the last
#: pending scenario. A future scenario must be added here with its owner.
SCENARIO_OWNERS: dict[str, tuple[str, ...]] = {}

#: Implemented ``test_scenario*`` → the item(s) that made it green once the
#: requirement exists. These carry **no** xfail marker; the audit asserts that,
#: so a scenario cannot be marked pending after it ships (nor silently
#: re-xfailed). ``test_scenario3`` went green with F2's atomic assignment, which
#: is F4's AC3a requirement (F4 still owns the busy-pool fallback AC3b, the
#: bounded wait AC3c and the deadlock/consistency tests AC3d).
IMPLEMENTED_SCENARIOS = {
    "test_scenario1_assign_then_measure_admits": (F2_ASSIGN,),
    "test_scenario2_busy_pool_falls_back_deterministically": (F4_FALLBACK,),
    "test_scenario1_refusal_releases_the_candidate": (F3_RELEASE,),
    "test_scenario3_concurrent_admissions_never_share_a_candidate": (
        F2_ASSIGN,
        F4_FALLBACK,
    ),
    "test_scenario4_probe_error_fails_open_and_logs_the_basis": (F2_ASSIGN,),
    "test_scenario5_stale_view_rereads_once_then_falls_back": (F2_ASSIGN,),
}


def _free_slots(ids, *, context_tokens: int | None = None) -> list[dict]:
    """Build a fake ``/slots`` snapshot of *ids* free slots.

    Same snapshot shape the ladder fixtures use (``slot_id`` /
    ``is_processing`` / ``context_tokens``) — one builder for the candidate
    harness, no second ladder (AC5c). The snapshot only carries the slot
    *details*; ownership comes from the live registry at view-build time,
    exactly as in ``proxy.residency.get_residency_view``.
    """
    return [_slot(sid, context_tokens) for sid in ids]


def _candidate_decision(
    *,
    session_id: str = "sess-a",
    pool_size: int = 2,
    **gate_kw,
):
    """Run one candidate-first admission the way F2's call site will.

    Order pinned by AC5a scenario 1: acquire the candidate **before** the
    residency view is materialised (the ladder builds its view from the
    snapshot + the live ownership registry, so the measurement is of the
    candidate slot), decide, and release the candidate when the admission was
    refused (F3's release contract). Returns ``(candidate, skip, reason)``.

    Fails on the pre-change tree because the production seam does not exist —
    that is exactly the state AC5d pins with the strict xfail markers.
    """
    candidate = session_registry.acquire_slot_candidate(
        session_id, endpoint="", pool_size=pool_size
    )
    skip, reason = provider._should_skip_local(
        gate_kw.pop("model_name", MODEL),
        session_id,
        {"messages": []},
        gate_kw.pop("cold_threshold", COLD_THRESHOLD),
        estimated_tokens=gate_kw.pop("estimated_tokens", EST_TOKENS),
        warm_cache_threshold=0,
        resident_prefix_min_ratio=gate_kw.pop("min_ratio", 0.5),
        unmeasured_cold_enabled=gate_kw.pop("unmeasured_cold_enabled", True),
        **gate_kw,
    )
    if candidate is not None and skip:
        session_registry.release_slot_candidate(candidate, reason=reason)
    return candidate, skip, reason


def test_scenario1_assign_then_measure_admits(caplog, monkeypatch):
    """Scenario 1 (admit half) — no slot at decision → assign → measure → admit.

    ``sess-a`` owns nothing at decision time; a candidate is assigned before
    the verdict, so the measurement is of the candidate slot
    (``residency_source=measured_last_dispatch``, ``measurement=measured``)
    instead of ``unavailable:no_slot``. AC5b: ``unavailable:no_slot`` must not
    appear — the pool is *not* exhausted.
    """
    import logging

    _prime_measurement(MEASURED_TOTAL)
    _install_slot_state(monkeypatch, slots=_free_slots([0]), owners={})

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        candidate, skip, reason = _candidate_decision(session_id="sess-a")

    assert candidate is not None
    assert candidate.session_id == "sess-a"
    assert candidate.slot_id == 0
    assert (skip, reason) == (False, None)
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.decision == "admit"
    assert event.residency_source == provider.RESIDENCY_SOURCE_MEASURED_LAST_DISPATCH
    assert event.n_past_actual == MEASURED_TOTAL
    assert _measurement_field(caplog) == MEASUREMENT_MEASURED
    assert MEASUREMENT_NO_SLOT not in caplog.text, (
        "a free pool must not be reported as unavailable:no_slot (AC5b)"
    )


def test_scenario1_refusal_releases_the_candidate(caplog, monkeypatch):
    """Scenario 1 (refuse half) — assign → measure → refuse → clean release.

    The measured prefix is below the ratio, so the admission is refused; the
    candidate is released through the production seam: no registry entry left,
    one ``candidate_released reason=...`` line, no ``abandoned_slot_recorded``
    (an abandoned slot means a *dispatched* request, never a refusal — F3
    AC2a/AC2b/AC2e).
    """
    import logging

    _prime_measurement(5_000)  # reusable fraction ≈ 0.10 < min_ratio 0.5
    _install_slot_state(monkeypatch, slots=_free_slots([0]), owners={})

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        candidate, skip, reason = _candidate_decision(session_id="sess-a")

    assert candidate is not None
    assert (skip, reason) == (True, SKIP_RESIDENT_PREFIX)
    # Measured basis, then a clean release: registry back to its baseline.
    assert dict(session_registry._slot_owners) == {}
    assert session_registry._assigned_slot_for_session("sess-a") is None
    assert f"candidate_released reason={SKIP_RESIDENT_PREFIX}" in caplog.text
    assert "abandoned_slot_recorded" not in caplog.text
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert _measurement_field(caplog) == MEASUREMENT_MEASURED
    assert event.residency_source == provider.RESIDENCY_SOURCE_MEASURED_LAST_DISPATCH


def test_scenario2_busy_pool_falls_back_deterministically(caplog, monkeypatch):
    """Scenario 2 — slot busy: no free candidate → deterministic fallback.

    The whole pool is occupied by another session, so ``acquire_slot_candidate``
    returns ``None`` and the decision falls back through the existing ladder.
    AC5b: ``unavailable:no_slot`` is emitted here *only* because the pool is
    genuinely exhausted, and the fallback reason is stable across identical
    runs (F4 AC3b) — measured against identical inputs, twice.
    """
    import logging

    _install_slot_state(
        monkeypatch,
        slots=[_slot(0, EST_TOKENS)],
        owners={("", 0): "sess-other"},
    )
    monkeypatch.setattr(provider, "expected_cached_ratio", lambda *a, **k: 0.99)

    outcomes = []
    bases = []
    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        for _ in range(2):  # deterministic across runs (F4 AC3b)
            candidate, skip, reason = _candidate_decision(
                session_id="sess-a", pool_size=1
            )
            assert candidate is None, "a full 1-slot pool has no candidate"
            outcomes.append((skip, reason))
            # Parse each run before clearing: the identical basis (source +
            # measurement) is part of the stable fallback, not only the reason.
            bases.append(
                (
                    _parse_logged_caught(caplog, "routing_resident_prefix"),
                    _measurement_field(caplog),
                )
            )
            caplog.clear()

    assert outcomes[0] == outcomes[1], "fallback must be deterministic (F4 AC3b)"
    assert outcomes[0] == (True, SKIP_RESIDENT_PREFIX)
    # Pool unchanged: the occupant keeps the slot, the session gains nothing.
    assert dict(session_registry._slot_owners) == {("", 0): "sess-other"}
    for event, measurement in bases:
        assert event.residency_source == provider.RESIDENCY_SOURCE_NON_RESIDENT
        assert measurement == MEASUREMENT_NO_SLOT


def test_scenario3_concurrent_admissions_never_share_a_candidate(caplog, monkeypatch):
    """Scenario 3 — two concurrent admissions never receive the same candidate.

    Two sessions admit concurrently on a 3-slot pool (one occupied, two free);
    the federated decisions must end on *distinct* candidate slots (or an
    explicit refusal), the registry must stay consistent (no slot with two
    owners) and the run must complete under a hard timeout (F4 AC3a/AC3d).
    """
    import asyncio
    import logging

    _prime_measurement(MEASURED_TOTAL, session="sess-a")
    _prime_measurement(MEASURED_TOTAL, session="sess-b")
    _install_slot_state(
        monkeypatch,
        slots=[_slot(0, EST_TOKENS), _slot(1, None), _slot(2, None)],
        owners={("", 0): "sess-other"},
    )

    async def _race():
        a = asyncio.to_thread(
            _candidate_decision, session_id="sess-a", pool_size=3
        )
        b = asyncio.to_thread(
            _candidate_decision, session_id="sess-b", pool_size=3
        )
        return await asyncio.wait_for(asyncio.gather(a, b), timeout=10)

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        (a_cand, a_skip, a_reason), (b_cand, b_skip, b_reason) = asyncio.run(_race())

    assert a_cand is not None and b_cand is not None
    assert a_cand.slot_id != b_cand.slot_id, "two admissions shared a candidate"
    assert {a_cand.slot_id, b_cand.slot_id} <= {1, 2}
    assert (a_skip, a_reason) == (False, None)
    assert (b_skip, b_reason) == (False, None)
    # Registry consistent afterwards: the occupant plus two distinct tenants.
    owners = dict(session_registry._slot_owners)
    assert owners[("", 0)] == "sess-other"
    tenants = [sid for (ep, sid), owner in owners.items()
               if owner in ("sess-a", "sess-b")]
    assert len(tenants) == 2 and len(set(tenants)) == 2


def test_scenario4_probe_error_fails_open_and_logs_the_basis(caplog, monkeypatch):
    """Scenario 4 — the residency probe raises → fail open, but logged.

    A candidate was still assigned before the verdict, yet every residency
    source becomes unavailable (the snapshot probe raises; ownership
    consultation degrades to ``None``). The ladder must not propagate the
    error nor fabricate a measurement: it fails open to the predicted path
    (rollback flag off), and one decision line records the honest basis
    (``residency_source=unknown`` / ``measurement=unavailable:stale_view``).
    """
    import logging

    from proxy import residency as residency_mod

    def _boom(*_a, **_k):
        raise RuntimeError("probe error (F1 scenario 4 fixture)")

    monkeypatch.setattr(residency_mod, "get_residency_view", _boom)
    monkeypatch.setattr(provider, "_session_owns_slot", lambda *a, **k: None)
    monkeypatch.setattr(provider, "expected_cached_ratio", lambda *a, **k: 1.0)
    _install_slot_state(monkeypatch, slots=_free_slots([0]), owners={})

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        candidate, skip, reason = _candidate_decision(
            session_id="sess-a", unmeasured_cold_enabled=False
        )

    assert candidate is not None, "the candidate is assigned before the verdict"
    assert (skip, reason) == (False, None), "fail-open to the predicted path"
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.decision == provider.DECISION_ADMIT_PREDICTED
    assert event.residency_source == provider.RESIDENCY_SOURCE_UNKNOWN
    assert _measurement_field(caplog) == MEASUREMENT_STALE_VIEW
    assert event.n_past_actual is None, "a probe error must not fabricate a size"


def test_scenario5_stale_view_rereads_once_then_falls_back(caplog, monkeypatch):
    """Scenario 5 — ``stale_view``: re-read once, then fall back with reason.

    The snapshot is older than the residency trust window and the pool is
    full for this session; the ladder re-reads the residency view **exactly
    once** before falling back, and the decision line carries
    ``measurement=unavailable:stale_view`` / ``residency_source=unknown`` with
    the unmeasured-cold skip. ``unavailable:no_slot`` must not appear — the
    truth is that the view is stale, not that the pool is measured full.
    """
    import logging

    from proxy.residency import RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS

    from proxy import residency as residency_mod

    _install_slot_state(
        monkeypatch,
        slots=[_slot(0, EST_TOKENS)],
        owners={("", 0): "sess-other"},
        age=RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS + 1.0,
    )
    calls = {"n": 0}
    real_view = residency_mod.get_residency_view

    def _counted(*args, **kwargs):
        calls["n"] += 1
        return real_view(*args, **kwargs)

    monkeypatch.setattr(residency_mod, "get_residency_view", _counted)

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        candidate, skip, reason = _candidate_decision(
            session_id="sess-a", pool_size=1
        )

    assert candidate is None
    assert (skip, reason) == (True, provider.SKIP_REASON_UNMEASURED_COLD)
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.residency_source == provider.RESIDENCY_SOURCE_UNKNOWN
    assert _measurement_field(caplog) == MEASUREMENT_STALE_VIEW
    assert MEASUREMENT_NO_SLOT not in _measurement_field(caplog)
    assert calls["n"] == 2, (
        "stale view must be re-read exactly once before the fallback "
        f"(got {calls['n']} residency-view reads)"
    )


def test_f1_scenario_owners_are_pinned():
    """AC5d audit — every scenario is pinned, and only pending ones are xfail.

    Introspects the module's ``test_scenario*`` functions against
    ``SCENARIO_OWNERS`` (pending) and ``IMPLEMENTED_SCENARIOS`` (shipped): a
    scenario that is dropped, un-pinned, softened or mis-owner-tagged fails
    here, so no AC5a scenario can silently disappear from the suite or lose
    its owning-item attribution. An implemented scenario must carry **no**
    xfail marker, so a shipped pin cannot rot into a permanently-skipped one.
    (Named ``test_f1_``, not ``test_scenario_``, so the audit is not mistaken
    for a scenario.)
    """
    import sys

    module = sys.modules[__name__]
    missing = [
        name
        for name in list(SCENARIO_OWNERS) + list(IMPLEMENTED_SCENARIOS)
        if not hasattr(module, name)
    ]
    assert not missing, f"F1 scenarios missing from the module: {missing}"
    assert not set(SCENARIO_OWNERS) & set(IMPLEMENTED_SCENARIOS), (
        "a scenario cannot be both pending and implemented"
    )
    extras = [
        name for name in dir(module)
        if name.startswith("test_scenario")
        and name not in SCENARIO_OWNERS
        and name not in IMPLEMENTED_SCENARIOS
    ]
    assert not extras, f"F1 scenarios without an owning item: {extras}"
    for name, owners in SCENARIO_OWNERS.items():
        fn = getattr(module, name)
        marks = list(getattr(fn, "pytestmark", []) or [])
        xfails = [
            m for m in marks
            if getattr(getattr(m, "mark", m), "name", None) == "xfail"
        ]
        assert xfails, f"{name}: AC5d requires an xfail marker naming its owner"
        kwargs = getattr(getattr(xfails[0], "mark", xfails[0]), "kwargs", {})
        assert kwargs.get("strict") is True, f"{name}: xfail must be strict"
        reason = kwargs.get("reason", "")
        assert any(owner in reason for owner in owners), (
            f"{name}: xfail reason {reason!r} must name one of {owners}"
        )
    for name, owner in IMPLEMENTED_SCENARIOS.items():
        fn = getattr(module, name)
        marks = list(getattr(fn, "pytestmark", []) or [])
        assert not [
            m for m in marks
            if getattr(getattr(m, "mark", m), "name", None) == "xfail"
        ], (
            f"{name}: implemented by {owner} — delete its xfail marker "
            "instead of leaving a skipped pin"
        )


# ---------------------------------------------------------------------------
# F2 — candidate-first admission (LP-0MW959P9S005DJY7)
#
# The verdict is taken with a slot candidate already assigned (parent AC1):
# ``_should_skip_local`` acquires the candidate itself from the configured slot
# pool, so the residency probe measures the candidate slot instead of
# predicting from a not-yet-existing assignment. These pins cover the
# implementation child's own acceptance surface — AC1a (assignment before the
# verdict), AC1c (``unavailable:no_slot`` only when the pool is exhausted),
# AC1d (measurement survives a same-session release only) and AC1e (honest
# degradation, no fabricated context size).
# ---------------------------------------------------------------------------


def _candidate_config(tmp_path, *, pool_size: int = 2) -> dict:
    """A ``server:`` config with the slot bookkeeping candidate-first needs."""
    return {
        "session_slot_pool_size": pool_size,
        "session_slot_save_path": str(tmp_path),
    }


def _decide_with_pool(
    *,
    session_id: str = "sess-a",
    server_config: dict | None = None,
    estimated_tokens: int = EST_TOKENS,
    **gate_kw,
):
    """Run one admission through the ladder's **own** candidate acquisition."""
    return provider._should_skip_local(
        MODEL,
        session_id,
        {"messages": []},
        COLD_THRESHOLD,
        estimated_tokens=estimated_tokens,
        warm_cache_threshold=0,
        resident_prefix_min_ratio=0.5,
        unmeasured_cold_enabled=gate_kw.pop("unmeasured_cold_enabled", True),
        endpoint=gate_kw.pop("endpoint", ""),
        server_config=server_config,
        **gate_kw,
    )


def test_admission_assigns_the_candidate_before_the_verdict(tmp_path, caplog, monkeypatch):
    """AC1a — the verdict is measured against a candidate assigned first.

    ``sess-a`` owns nothing at decision time and the pool has a free slot, so
    the ladder assigns it before probing. The measurement is of that candidate
    (``measured_last_dispatch``), not the predicted sidecar ratio, and the
    ownership registry records the candidate for the dispatch that follows.
    """
    import logging

    _prime_measurement(MEASURED_TOTAL)
    _install_slot_state(monkeypatch, slots=_free_slots([0]), owners={})

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _decide_with_pool(
            server_config=_candidate_config(tmp_path)
        )

    assert (skip, reason) == (False, None)
    assert dict(session_registry._slot_owners) == {("", 0): "sess-a"}
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.decision == "admit"
    assert event.residency_source == provider.RESIDENCY_SOURCE_MEASURED_LAST_DISPATCH
    assert event.n_past_actual == MEASURED_TOTAL
    assert _measurement_field(caplog) == MEASUREMENT_MEASURED


def test_candidate_assignment_is_idempotent_for_the_same_session():
    """AC1a — a session that already owns its candidate re-enters it.

    Re-entry never consumes a second slot and reports the same candidate, so
    the repeated probes/decisions in one request cycle cannot walk the pool.
    """
    config = {"session_slot_pool_size": 2, "session_slot_save_path": "/tmp/slots"}

    first = session_registry.acquire_slot_candidate(
        "sess-a", endpoint="", pool_size=session_registry.candidate_pool_size(config)
    )
    second = session_registry.acquire_slot_candidate(
        "sess-a", endpoint="", pool_size=session_registry.candidate_pool_size(config)
    )

    assert first is not None and second is not None
    assert (first.slot_id, first.session_id) == (second.slot_id, second.session_id)
    assert first.slot_id == 0
    assert dict(session_registry._slot_owners) == {("", 0): "sess-a"}


def test_no_slot_is_reported_only_when_the_pool_is_exhausted(
    tmp_path, caplog, monkeypatch
):
    """AC1c — ``unavailable:no_slot`` means the pool is full, nothing else.

    A free pool with no measurement reports ``no_prior_measurement`` (the
    candidate *was* measured — it simply has no prior dispatch); only a pool
    with no free candidate reports ``no_slot``.
    """
    import logging

    # Free pool, no measurement: measured against the candidate.
    _install_slot_state(monkeypatch, slots=_free_slots([0]), owners={})
    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        _decide_with_pool(server_config=_candidate_config(tmp_path))
    free_pool_measurement = _measurement_field(caplog)
    assert free_pool_measurement == MEASUREMENT_NO_PRIOR
    assert MEASUREMENT_NO_SLOT not in caplog.text
    caplog.clear()

    # Pool exhausted: the one slot belongs to another session.
    _install_slot_state(
        monkeypatch,
        slots=[_slot(0, EST_TOKENS)],
        owners={("", 0): "sess-other"},
    )
    monkeypatch.setattr(provider, "expected_cached_ratio", lambda *a, **k: 0.99)
    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _decide_with_pool(
            server_config=_candidate_config(tmp_path, pool_size=1)
        )

    assert (skip, reason) == (True, SKIP_RESIDENT_PREFIX)
    assert _measurement_field(caplog) == MEASUREMENT_NO_SLOT
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.residency_source == provider.RESIDENCY_SOURCE_NON_RESIDENT
    # The occupant keeps the slot; the refused session gained nothing.
    assert dict(session_registry._slot_owners) == {("", 0): "sess-other"}


def test_no_pool_configured_keeps_the_pre_change_behaviour(caplog, monkeypatch):
    """Fail-open — no slot bookkeeping means no candidate and no ownership.

    A deployment without the slot pool must keep today's residency verdict
    (nothing is registered, the session owns no slot), never a fabricated
    candidate.
    """
    import logging

    _install_slot_state(monkeypatch, slots=_free_slots([0]), owners={})

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _decide_with_pool(server_config=None)

    assert dict(session_registry._slot_owners) == {}
    assert (skip, reason) == (True, SKIP_RESIDENT_PREFIX)
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.residency_source == provider.RESIDENCY_SOURCE_NON_RESIDENT
    assert _measurement_field(caplog) == MEASUREMENT_NO_SLOT


def test_candidate_without_a_measurement_degrades_to_session_slot(
    tmp_path, caplog, monkeypatch
):
    """AC1e — no live ``n_past`` and no prior dispatch degrades one rung.

    The candidate is owned but nothing measures its prefix, so the ladder
    reports ``session_slot`` / ``unavailable:no_prior_measurement`` with
    ``n_past_actual=none``: the rung is explicit and never logs a fabricated
    context size.
    """
    import logging

    _install_slot_state(monkeypatch, slots=[_slot(0, None)], owners={})

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        _decide_with_pool(server_config=_candidate_config(tmp_path))

    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.residency_source == provider.RESIDENCY_SOURCE_SESSION_SLOT
    assert event.n_past_actual is None
    assert event.reuse_ratio is None
    assert _measurement_field(caplog) == MEASUREMENT_NO_PRIOR


def test_measurement_survives_the_dispatch_turn_boundary(tmp_path, caplog, monkeypatch):
    """AC1d — the released slot's measurement prices the session's next turn.

    A dispatch ends with ``_free_slot_assignment``; the session's next
    admission must still be priced from its last measured dispatch (that is
    what makes ``unavailable:no_slot`` stop being the normal path), and it
    must do so against the candidate it is re-assigned before the verdict.
    """
    import logging

    _prime_measurement(MEASURED_TOTAL)
    _install_slot_state(monkeypatch, slots=_free_slots([0]), owners={})
    session_registry._slot_id_for_session("sess-a", 2)
    session_registry._free_slot_assignment("sess-a")
    assert measured_prefill.get_measured_prefill(MODEL, "sess-a") is not None

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _decide_with_pool(
            server_config=_candidate_config(tmp_path)
        )

    assert (skip, reason) == (False, None)
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.residency_source == provider.RESIDENCY_SOURCE_MEASURED_LAST_DISPATCH
    assert event.n_past_actual == MEASURED_TOTAL
    assert MEASUREMENT_NO_SLOT not in caplog.text


# ---------------------------------------------------------------------------
# F3 — release the slot candidate cleanly on a refused admission
# (LP-0MW959PWL001WYSM, parent AC2)
#
# Candidate-first admission acquires a slot before the verdict, so an
# acquired-but-refused candidate is a *fabricated occupancy* that would deny
# the pool to real work. The release contract pinned here:
#
# - nothing is left behind (no slot registry entry, no dispatch lease, no
#   ``slot_<session>.bin``, no ``abandoned_slot_recorded``) — AC2a;
# - exactly one ``candidate_released reason=<skip_reason>`` line — AC2b;
# - every refusal rung releases, including an exception inside the decision —
#   AC2c;
# - an admitted admission keeps its candidate with no second assignment and no
#   release — AC2d;
# - a run of refusals returns the registry to baseline and a second release is
#   a no-op — AC2e.
#
# The release seam is ``proxy.session.release_slot_candidate`` (F1 pins it); the
# production call site owns it through ``provider._AdmissionCandidateLease``.
# ---------------------------------------------------------------------------

#: The one line a refused admission must emit (AC2b).
CANDIDATE_RELEASED = "candidate_released"


def _release_config(tmp_path, *, pool_size: int = 2, **extra) -> dict:
    """A ``server:`` config with the slot bookkeeping a candidate needs."""
    config = _candidate_config(tmp_path, pool_size=pool_size)
    config.update(extra)
    return config


def _decide_with_release(tmp_path, *, session_id: str = "sess-a", pool_size: int = 2, **gate_kw):
    """One admission through the ladder's **own** candidate acquisition.

    The caller does not hold a candidate, so the ladder acquires one and owns
    its release (the ``_queue_context_bypass`` / direct-call shape).
    """
    return provider._should_skip_local(
        MODEL,
        session_id,
        {"messages": []},
        gate_kw.pop("cold_threshold", COLD_THRESHOLD),
        estimated_tokens=gate_kw.pop("estimated_tokens", EST_TOKENS),
        warm_cache_threshold=gate_kw.pop("warm_threshold", 0),
        resident_prefix_min_ratio=gate_kw.pop("min_ratio", 0.5),
        unmeasured_cold_enabled=gate_kw.pop("unmeasured_cold_enabled", True),
        endpoint=gate_kw.pop("endpoint", ""),
        server_config=_release_config(tmp_path, pool_size=pool_size),
        **gate_kw,
    )


def _assert_clean_release(caplog, *, session_id: str = "sess-a", reason: str) -> None:
    """AC2a/AC2b: registry back to baseline, one line naming the reason."""
    assert dict(session_registry._slot_owners) == {}, (
        "a refused admission must leave no fabricated occupancy behind"
    )
    assert session_registry._assigned_slot_for_session(session_id) is None
    assert f"{CANDIDATE_RELEASED} reason={reason}" in caplog.text, (
        f"the release must be operationally visible, naming {reason!r}"
    )
    assert "abandoned_slot_recorded" not in caplog.text, (
        "an abandoned slot means a dispatched request that never reported, "
        "never a refused admission (AC2a)"
    )


# Rung → the setup that reaches it with a candidate held. ``unmeasured_``
# ``residency_cold`` is deliberately absent: it is only reachable when **no**
# candidate could be acquired (it requires an unknown residency source, which
# owning a candidate rules out), and that case is pinned separately below.
_REFUSAL_RUNG_CASES = ("context_too_large", SKIP_RESIDENT_PREFIX, SKIP_TENANT_LIMIT,
                       "large_context_bypass")


@pytest.mark.parametrize("rung", _REFUSAL_RUNG_CASES)
def test_every_refusal_rung_releases_the_candidate(tmp_path, caplog, monkeypatch, rung):
    """AC2c — every refusal rung that can hold a candidate releases it.

    Each rung is driven through the ladder's own acquisition, so the assertion
    is about the *decision* releasing, not about a hand-written release in the
    test: the registry is back to its baseline and the line names that rung's
    reason. ``context_too_large`` is the outermost rung (it refuses before the
    residency probe), so it proves the candidate is acquired early enough for
    every rung to be able to return it.
    """
    import logging

    kwargs: dict = {}
    session_id = "sess-a"
    if rung == "context_too_large":
        kwargs["warm_threshold"] = 1_000
    elif rung == SKIP_RESIDENT_PREFIX:
        _prime_measurement(5_000)  # reusable fraction ≈ 0.10 < min_ratio 0.5
        _install_slot_state(monkeypatch, slots=_free_slots([0]), owners={})
    elif rung == SKIP_TENANT_LIMIT:
        # Two live tenants occupy the bound; the newcomer (``sess-c``) owns its
        # candidate and passes the residency gate, so it reaches the tenancy
        # rung — the one that must return the candidate.
        session_id = "sess-c"
        _install_tenancy(["sess-a", "sess-b"])
        _install_slot_state(
            monkeypatch, slots=[_slot(0, None, processing=False)], owners={}
        )
        kwargs["session_tenants"] = 2
        kwargs["tenancy_idle_seconds"] = 30.0
    elif rung == "large_context_bypass":
        _prime_measurement(MEASURED_TOTAL)
        _install_slot_state(monkeypatch, slots=_free_slots([0]), owners={})
        kwargs["cold_threshold"] = 10_000

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _decide_with_release(tmp_path, session_id=session_id, **kwargs)

    assert (skip, reason) == (True, rung)
    _assert_clean_release(caplog, session_id=session_id, reason=rung)


def test_a_refusal_leaves_no_lease_and_no_slot_snapshot(tmp_path, caplog, monkeypatch):
    """AC2a — a refusal never dispatches, so nothing is persisted.

    The refused candidate leaves no dispatch-lease record, no
    ``slot_<session>.bin`` snapshot under the configured save path and no
    ``abandoned_slot_recorded`` event: the release is bookkeeping only.
    """
    import logging

    import proxy.server as server_module

    _prime_measurement(5_000)
    _install_slot_state(monkeypatch, slots=_free_slots([0]), owners={})
    before = dict(server_module.local_dispatch_records)

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _decide_with_release(tmp_path)

    assert (skip, reason) == (True, SKIP_RESIDENT_PREFIX)
    _assert_clean_release(caplog, reason=SKIP_RESIDENT_PREFIX)
    assert dict(server_module.local_dispatch_records) == before, (
        "a refused admission must not create a dispatch lease"
    )
    assert not list(tmp_path.glob("slot_*.bin")), (
        "a refused admission must not save a slot snapshot"
    )


def test_release_is_idempotent_and_never_double_frees(tmp_path, caplog):
    """AC2e — a second release is a no-op: one line, one pop.

    The ``finally``-style safety release and an explicit refusal release both
    call the same seam, so the seam itself must be idempotent: the second call
    reports ``False`` and emits no second ``candidate_released`` line.
    """
    import logging

    candidate = session_registry.acquire_slot_candidate(
        "sess-a",
        endpoint="",
        pool_size=session_registry.candidate_pool_size(_release_config(tmp_path)),
    )
    assert candidate is not None

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        assert session_registry.release_slot_candidate(
            candidate, reason=SKIP_RESIDENT_PREFIX
        ) is True
        assert session_registry.release_slot_candidate(
            candidate, reason=SKIP_RESIDENT_PREFIX
        ) is False

    assert dict(session_registry._slot_owners) == {}
    assert caplog.text.count(CANDIDATE_RELEASED) == 1
    assert f"{CANDIDATE_RELEASED} reason={SKIP_RESIDENT_PREFIX}" in caplog.text


def test_a_run_of_refusals_returns_the_registry_to_baseline(
    tmp_path, caplog, monkeypatch
):
    """AC2e — refusals leak nothing and leave the pool usable.

    Five refusals from four sessions neither grow the registry nor strand a
    slot: the pool the refusals borrowed from is still fully available
    afterwards, and each refusal logged its own reason.
    """
    import logging

    _prime_measurement(5_000)
    baseline = dict(session_registry._slot_owners)
    assert baseline == {}

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        for index, session_id in enumerate(
            ["sess-a", "sess-b", "sess-c", "sess-d", "sess-a"]
        ):
            # Each refusal must be a *refusal*: a measured prefix below the
            # ratio, re-primed per session because a change of hands forgets the
            # previous tenant's measurement (AC1d).
            _prime_measurement(5_000, session=session_id)
            _install_slot_state(monkeypatch, slots=_free_slots([0, 1]), owners={})
            skip, reason = _decide_with_release(
                tmp_path, session_id=session_id, pool_size=2
            )
            assert (skip, reason) == (True, SKIP_RESIDENT_PREFIX), index

    assert dict(session_registry._slot_owners) == baseline
    assert caplog.text.count(CANDIDATE_RELEASED) == 5

    # The pool still hands out candidates: nothing was stranded by the run.
    reclaimed = session_registry.acquire_slot_candidate(
        "sess-new",
        endpoint="",
        pool_size=session_registry.candidate_pool_size(_release_config(tmp_path)),
    )
    assert reclaimed is not None
    assert reclaimed.slot_id == 0


def test_an_exception_inside_the_decision_releases_the_candidate(
    tmp_path, caplog, monkeypatch
):
    """AC2c — a raise inside the decision never leaks the candidate.

    The residency probe raising is the shape a partial failure takes; the
    candidate acquired for the verdict is returned before the exception is
    re-raised, with the reason named on the release line.
    """
    import logging

    _install_slot_state(monkeypatch, slots=_free_slots([0]), owners={})

    def _boom(*_args, **_kwargs):
        raise RuntimeError("residency probe exploded")

    monkeypatch.setattr(provider, "resolve_resident_prefix", _boom)

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        with pytest.raises(RuntimeError):
            _decide_with_release(tmp_path)

    assert dict(session_registry._slot_owners) == {}
    assert f"{CANDIDATE_RELEASED} reason=exception" in caplog.text


def test_no_candidate_means_nothing_to_release(caplog, monkeypatch):
    """AC2c — the unmeasured-cold rung cannot leak: it has no candidate.

    ``unmeasured_residency_cold`` requires an *unknown* residency source, and
    owning a candidate rules that out; it is therefore only reachable when the
    pool could not hand one out (here: the single slot belongs to another
    session). Nothing is acquired, so nothing is released and the occupant
    keeps its slot.
    """
    import logging

    from proxy.residency import RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS

    _install_slot_state(
        monkeypatch,
        slots=[_slot(0, EST_TOKENS)],
        owners={("", 0): "sess-other"},
        age=RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS + 1.0,
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = provider._should_skip_local(
            MODEL,
            "sess-a",
            {"messages": []},
            COLD_THRESHOLD,
            estimated_tokens=EST_TOKENS,
            resident_prefix_min_ratio=0.5,
            endpoint="",
            server_config={
                "session_slot_pool_size": 1,
                "session_slot_save_path": "/tmp/slots",
            },
        )

    assert (skip, reason) == (True, provider.SKIP_REASON_UNMEASURED_COLD)
    assert dict(session_registry._slot_owners) == {("", 0): "sess-other"}
    assert CANDIDATE_RELEASED not in caplog.text


def test_admission_keeps_the_candidate_through_dispatch(tmp_path, caplog, monkeypatch):
    """AC2d — an admitted request keeps its candidate, unchanged.

    When the caller owns the lifecycle, a refusal-free verdict releases
    nothing; keeping the lease hands ownership to the dispatch bookkeeping with
    no second assignment, and a later safety release is a no-op.
    """
    import logging

    _prime_measurement(MEASURED_TOTAL)
    _install_slot_state(monkeypatch, slots=_free_slots([0]), owners={})

    config = _release_config(tmp_path)
    candidate = session_registry.acquire_slot_candidate(
        "sess-a",
        endpoint="",
        pool_size=session_registry.candidate_pool_size(config),
    )
    assert candidate is not None
    lease = provider._AdmissionCandidateLease(candidate)

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = provider._should_skip_local(
            MODEL,
            "sess-a",
            {"messages": []},
            COLD_THRESHOLD,
            estimated_tokens=EST_TOKENS,
            resident_prefix_min_ratio=0.5,
            endpoint="",
            server_config=config,
            candidate_slot=candidate,
        )

        assert (skip, reason) == (False, None)
        assert dict(session_registry._slot_owners) == {("", 0): "sess-a"}
        assert CANDIDATE_RELEASED not in caplog.text

        # The dispatch path keeps the lease and still holds the same slot.
        lease.keep()
        assert lease.release() is False
        assert dict(session_registry._slot_owners) == {("", 0): "sess-a"}
        assert CANDIDATE_RELEASED not in caplog.text


def test_lease_release_names_the_refusing_reason(tmp_path, caplog):
    """AC2b/AC2e — the lease releases once, naming the recorded reason."""
    import logging

    candidate = session_registry.acquire_slot_candidate(
        "sess-a",
        endpoint="",
        pool_size=session_registry.candidate_pool_size(_release_config(tmp_path)),
    )
    assert candidate is not None
    lease = provider._AdmissionCandidateLease(candidate)
    lease.note_refusal(SKIP_TENANT_LIMIT)

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        assert lease.release() is True
        assert lease.release() is False  # idempotent safety release

    assert dict(session_registry._slot_owners) == {}
    assert caplog.text.count(CANDIDATE_RELEASED) == 1
    assert f"{CANDIDATE_RELEASED} reason={SKIP_TENANT_LIMIT}" in caplog.text


# ---------------------------------------------------------------------------
# F4 — candidate exclusivity and deterministic busy-pool fallback
# (LP-0MW959QJ9002MKEI, parent AC3)
#
# Candidate-first admission (F2/F3) moved capacity acquisition into the
# admission path. That turns a previously-impossible race into an admission-time
# one, so the contract is proven here rather than assumed:
#
# - AC3a: two concurrent admissions never receive the same candidate; when the
#   pool cannot serve both, the loser is refused explicitly (``None``) instead
#   of sharing;
# - AC3b: a full pool falls back deterministically — the same reason for the
#   same inputs, on both decision seams (the direct admission ladder and the
#   contention-queue mirror), and the occupant's occupancy is untouched;
# - AC3c: acquisition is bounded — a busy pool returns immediately and never
#   introduces a wait primitive of its own (any real wait is the existing
#   contention queue, whose budget is ``prefill_eta.wait_budget_seconds``);
# - AC3d: concurrent acquire/release cycles leave the registry consistent (no
#   slot shared, no double release) and complete under a hard timeout;
# - AC3e: a session that already owns a candidate re-enters it idempotently,
#   concurrently too, and never consumes a second slot.
#
# Every test drives the real registry (``session._slot_owners`` +
# ``_slot_registry_lock``); the residency snapshot stub is the one documented in
# the module header. No sleeps: concurrency is real threads, bounded by a
# join/acquire timeout instead of a clock.
# ---------------------------------------------------------------------------

#: Hard bound for a concurrency test to complete (a deadlock fails here).
CONCURRENCY_TIMEOUT = 30.0


def _busy_pool_config(tmp_path, *, pool_size: int = 1) -> dict:
    """A full ``server:`` config with the slot bookkeeping a candidate needs.

    Nested so the residency readers (``_effective_resident_prefix_min_ratio``,
    ``_effective_large_context_thresholds``) resolve exactly as in production.
    """
    return {
        "server": {
            "session_slot_pool_size": pool_size,
            "session_slot_save_path": str(tmp_path),
            "local_large_context_cold_cache_threshold": COLD_THRESHOLD,
            "local_resident_prefix_min_ratio": 0.5,
        }
    }


def _busy_pool_decision(
    config: dict, *, session_id: str = "sess-a"
) -> tuple[bool, str | None, object]:
    """One admission decision in the shape the dispatch loop uses.

    Acquire the attempt's candidate, take the verdict without letting the
    ladder acquire a second one (``own_candidate=False``) and hand the
    candidate back on a refusal — the production sequence of
    ``_proxy_with_fallback_cycle`` + ``_AdmissionCandidateLease``. Returns
    ``(skip, reason, candidate)``.
    """
    server_config = provider._server_section(config)
    candidate = provider._acquire_admission_candidate(
        session_id, endpoint="", server_config=server_config
    )
    lease = provider._AdmissionCandidateLease(candidate)
    skip, reason = provider._should_skip_local(
        MODEL,
        session_id,
        {"messages": []},
        COLD_THRESHOLD,
        estimated_tokens=EST_TOKENS,
        warm_cache_threshold=0,
        resident_prefix_min_ratio=provider._effective_resident_prefix_min_ratio(
            config
        ),
        unmeasured_cold_enabled=provider._unmeasured_residency_cold_enabled(config),
        endpoint="",
        server_config=server_config,
        candidate_slot=candidate,
        own_candidate=False,
    )
    if skip:
        lease.note_refusal(reason)
    else:
        lease.keep()  # admitted: ownership passes to the dispatch bookkeeping
    lease.release()
    return skip, reason, candidate


def _registry_snapshot() -> dict:
    """Copy the ownership registry under its own lock (never blocks unbounded).

    The registry documents one locking discipline for mutation; taking the same
    lock for a consistent read also proves it is never held unboundedly — a
    lock held across a wait would fail here instead of hanging the suite.
    """
    lock = session_registry._slot_registry_lock
    assert lock.acquire(timeout=CONCURRENCY_TIMEOUT), (
        "slot registry lock held unboundedly (AC3c/AC3d)"
    )
    try:
        return dict(session_registry._slot_owners)
    finally:
        lock.release()


def _registry_violations(owners: dict, *, pool_size: int) -> list[str]:
    """AC3d invariants: one owner per slot, one slot per session per endpoint."""
    violations: list[str] = []
    per_session: dict[tuple[str, str], list[int]] = {}
    for key, owner in owners.items():
        endpoint = session_registry._slot_key_endpoint(key)
        slot_id = session_registry._slot_key_slot_id(key)
        if not isinstance(owner, str) or not owner:
            violations.append(f"slot {key!r} has no owning session")
            continue
        if not 0 <= slot_id < pool_size:
            violations.append(f"slot {slot_id} outside the {pool_size}-slot pool")
        per_session.setdefault((endpoint, owner), []).append(slot_id)
    for (endpoint, owner), slots in per_session.items():
        if len(slots) > 1:
            violations.append(
                f"{owner} owns slots {sorted(slots)} on endpoint {endpoint!r}"
            )
    return violations


def _run_concurrently(target, count: int):
    """Run *target(index)* in *count* daemon threads, joined under a timeout.

    Daemon threads (never a pool) so a deadlocked run fails the assertion
    instead of hanging interpreter teardown. Returns ``(results, alive)``.
    """
    import threading

    results: list = [None] * count
    barrier = threading.Barrier(count)

    def _worker(index: int) -> None:
        barrier.wait(timeout=CONCURRENCY_TIMEOUT)
        results[index] = target(index)

    threads = [
        threading.Thread(target=_worker, args=(index,), daemon=True)
        for index in range(count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=CONCURRENCY_TIMEOUT)
    return results, [thread for thread in threads if thread.is_alive()]


def _install_full_pool(monkeypatch: pytest.MonkeyPatch, *, session: str = "sess-other"):
    """Install the 1-slot pool occupied by another session (a busy pool)."""
    _install_slot_state(
        monkeypatch,
        slots=[_slot(0, EST_TOKENS)],
        owners={("", 0): session},
    )


def test_busy_pool_reason_is_stable_across_identical_decisions(
    tmp_path, caplog, monkeypatch
):
    """AC3b — a full pool falls back with one stable reason, run after run.

    A fully warm prediction is stubbed (0.99): a busy pool must not be rescued
    by the predictive ratio, so the fallback is the measured non-resident
    denial every time, and the occupant's occupancy is never touched.
    """
    import logging

    _install_full_pool(monkeypatch)
    monkeypatch.setattr(provider, "expected_cached_ratio", lambda *a, **k: 0.99)
    config = _busy_pool_config(tmp_path)

    outcomes = []
    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        for _ in range(12):
            skip, reason, candidate = _busy_pool_decision(config)
            assert candidate is None, "a full 1-slot pool cannot hand out a slot"
            outcomes.append((skip, reason))

    assert set(outcomes) == {(True, SKIP_RESIDENT_PREFIX)}, (
        f"the busy-pool fallback must be deterministic (AC3b): {set(outcomes)}"
    )
    # The occupant keeps its slot; no admission gains occupancy by refusing.
    assert _registry_snapshot() == {("", 0): "sess-other"}
    event = _parse_logged_caught(caplog, "routing_resident_prefix")
    assert event.residency_source == provider.RESIDENCY_SOURCE_NON_RESIDENT
    assert _measurement_field(caplog) == MEASUREMENT_NO_SLOT


def test_busy_pool_contention_queue_mirror_agrees_with_the_ladder(
    tmp_path, caplog, monkeypatch
):
    """AC3b — both decision seams name the same fallback for one busy pool.

    The contention-queue decision point runs the same ladder through
    ``_queue_context_bypass``. For a pinned-local client a *busy pool* is a
    residency denial, not a capacity wait, so the mirror refuses
    deterministically (the request steps to the next provider / retryable
    refusal) and never enqueues: the queue stays empty.
    """
    import asyncio
    import logging

    from proxy import contention_queue

    _install_full_pool(monkeypatch)
    monkeypatch.setattr(provider, "expected_cached_ratio", lambda *a, **k: 0.99)
    config = _busy_pool_config(tmp_path)
    model_config = {"providers": [{"name": "local", "type": "local"}]}
    provider_cfg = {"name": "local", "type": "local", "llama_model": MODEL}

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        direct = _busy_pool_decision(config)[:2]
        caplog.clear()
        mirrored = [
            asyncio.run(
                provider._queue_context_bypass(
                    config,
                    model_config,
                    provider_cfg,
                    None,  # request is unused when estimated_tokens is supplied
                    {"messages": []},
                    "sess-a",
                    priority="medium",
                    estimated_tokens=EST_TOKENS,
                    endpoint="",
                )
            )
            for _ in range(2)
        ]

    assert direct == (True, SKIP_RESIDENT_PREFIX)
    assert mirrored == [direct, direct], (
        "the queue mirror must agree with the admission ladder (AC3b)"
    )
    assert contention_queue.queue_depth() == 0, (
        "a busy pool is not a capacity wait: nothing may be enqueued"
    )
    assert _registry_snapshot() == {("", 0): "sess-other"}


def test_busy_pool_acquisition_is_bounded_by_the_existing_wait_budget(
    tmp_path, monkeypatch
):
    """AC3c — a busy pool returns at once; no new wait primitive is introduced.

    Twenty contended decisions on a full pool all return well inside the
    *existing* local wait budget (``prefill_eta.wait_budget_seconds``, the
    contention-queue window), with nothing enqueued. The bound is asserted
    against that shipped budget rather than a new constant, so an acquisition
    that started waiting would have to spend it. The budget's own identity is
    pinned by ``tests/test_contention_queue.py`` and
    ``tests/test_prefill_eta.py`` — not duplicated here.
    """
    import time

    from proxy import contention_queue, prefill_eta

    _install_full_pool(monkeypatch)
    monkeypatch.setattr(provider, "expected_cached_ratio", lambda *a, **k: 0.99)
    config = _busy_pool_config(tmp_path)
    budget_seconds = prefill_eta.wait_budget_seconds(config, "medium")
    assert budget_seconds > 0, "the shipped contention-queue budget is enabled"

    started = time.monotonic()
    for _ in range(20):
        skip, reason, candidate = _busy_pool_decision(config)
        assert (skip, reason, candidate) == (True, SKIP_RESIDENT_PREFIX, None)
    elapsed = time.monotonic() - started

    assert elapsed < budget_seconds, (
        f"acquisition must not wait: {elapsed:.3f}s against the existing "
        f"{budget_seconds:.1f}s wait budget (AC3c)"
    )
    assert contention_queue.queue_depth() == 0


def test_concurrent_pool_exhaustion_hands_out_distinct_candidates(tmp_path):
    """AC3a — concurrent admissions get distinct candidates or a refusal.

    Eight sessions race for a two-slot pool: exactly two candidates are handed
    out, on distinct slots, and the other six are refused explicitly (never a
    shared slot). The registry then holds exactly the two winners.
    """
    pool_size = 2
    config = _busy_pool_config(tmp_path, pool_size=pool_size)
    sessions = [f"sess-{index}" for index in range(8)]

    results, alive = _run_concurrently(
        lambda index: session_registry.acquire_slot_candidate(
            sessions[index],
            endpoint="",
            pool_size=session_registry.candidate_pool_size(
                provider._server_section(config)
            ),
        ),
        len(sessions),
    )

    assert not alive, "concurrent acquisition deadlocked (AC3d)"
    granted = [(session, candidate) for session, candidate in zip(sessions, results)
               if candidate is not None]
    assert len(granted) == pool_size, (
        f"a {pool_size}-slot pool must grant exactly {pool_size} candidates, "
        f"got {[(s, c.slot_id) for s, c in granted]}"
    )
    slots = [candidate.slot_id for _, candidate in granted]
    assert len(set(slots)) == pool_size, f"two admissions shared a candidate: {slots}"
    assert set(slots) == {0, 1}
    assert all(candidate is None for session, candidate in zip(sessions, results)
               if session not in {s for s, _ in granted})

    owners = _registry_snapshot()
    assert owners == {("", candidate.slot_id): session for session, candidate in granted}
    assert _registry_violations(owners, pool_size=pool_size) == []

    for session, candidate in granted:
        assert session_registry.release_slot_candidate(candidate, reason="test") is True
    assert _registry_snapshot() == {}


def test_concurrent_acquire_release_cycles_keep_the_registry_consistent():
    """AC3d — acquire/release churn leaves no shared slot and no double free.

    Eight threads run interleaved acquire / double-release cycles over a
    four-slot pool. Every observed registry state is consistent (one owner per
    slot, one slot per session), a second release of the same candidate is a
    no-op, and the whole cycle completes under the hard join timeout — the
    registry has no lock held across a wait, so no deadlock.
    """
    pool_size = 4
    sessions = [f"sess-{index}" for index in range(12)]
    cycles = 40
    violations: list[str] = []

    def _cycle(index: int):
        import random

        rng = random.Random(index)
        for _ in range(cycles):
            session = sessions[rng.randrange(len(sessions))]
            candidate = session_registry.acquire_slot_candidate(
                session, endpoint="", pool_size=pool_size
            )
            owners = _registry_snapshot()
            violations.extend(_registry_violations(owners, pool_size=pool_size))
            if candidate is None:
                continue
            owned = owners.get(("", candidate.slot_id))
            if owned != session:
                violations.append(
                    f"{session} was handed slot {candidate.slot_id} owned by {owned}"
                )
            released = session_registry.release_slot_candidate(candidate, reason="churn")
            if session_registry.release_slot_candidate(candidate, reason="churn"):
                violations.append(f"slot {candidate.slot_id} was double-released")
            after = _registry_snapshot()
            violations.extend(_registry_violations(after, pool_size=pool_size))
            if released and after.get(("", candidate.slot_id)) == session:
                violations.append(
                    f"slot {candidate.slot_id} still owned by {session} after release"
                )
        return session

    _, alive = _run_concurrently(_cycle, 8)

    assert not alive, "concurrent acquire/release deadlocked (AC3d)"
    assert violations == [], f"registry inconsistencies under contention: {violations[:5]}"
    for session in sessions:
        session_registry._free_slot_assignment(session)
    assert _registry_snapshot() == {}, "the churn leaked an occupied slot"


def test_same_session_candidate_re_entry_is_idempotent_under_concurrency():
    """AC3e — concurrent re-entry consumes exactly one slot.

    Eight threads acquire for the *same* session on a two-slot pool: every call
    re-enters the same candidate, so the registry holds one entry and the
    second slot is still free for another session.
    """
    pool_size = 2

    results, alive = _run_concurrently(
        lambda _index: session_registry.acquire_slot_candidate(
            "sess-a", endpoint="", pool_size=pool_size
        ),
        8,
    )

    assert not alive, "concurrent same-session re-entry deadlocked (AC3d)"
    assert all(candidate is not None for candidate in results)
    slots = {candidate.slot_id for candidate in results}
    assert slots == {0}, f"re-entry walked the pool: {sorted(slots)} (AC3e)"
    assert _registry_snapshot() == {("", 0): "sess-a"}

    other = session_registry.acquire_slot_candidate(
        "sess-b", endpoint="", pool_size=pool_size
    )
    assert other is not None and other.slot_id == 1, (
        "re-entry must leave the remaining slot free for another session"
    )


def test_concurrent_assignments_stay_exclusive_under_a_widened_window(
    monkeypatch,
):
    """AC3a — the assign step is mutually exclusive, not merely fast.

    Two sessions race for a two-slot pool while the assignment's critical
    section is deliberately widened (the change-of-hands bookkeeping yields the
    GIL). Without mutual exclusion both threads would observe an empty registry
    inside the widened window and both be handed slot 0; with the registry lock
    the second thread waits and takes slot 1. This is the pressure that makes
    the pin a proof of exclusivity rather than of a lucky interleaving.
    """
    import time

    real_forget = session_registry._forget_previous_tenant

    def _slow_forget(key, session_id):
        time.sleep(0.05)
        real_forget(key, session_id)

    monkeypatch.setattr(session_registry, "_forget_previous_tenant", _slow_forget)

    results, alive = _run_concurrently(
        lambda index: session_registry.acquire_slot_candidate(
            f"sess-{index}", endpoint="", pool_size=2
        ),
        2,
    )

    assert not alive, "the assign step deadlocked (AC3d)"
    assert all(candidate is not None for candidate in results)
    slots = [candidate.slot_id for candidate in results]
    assert len(set(slots)) == 2, f"two admissions shared a candidate: {slots}"
    # Which session wins slot 0 is scheduling-dependent; that each holds exactly
    # one slot of the pair is not.
    owners = _registry_snapshot()
    assert sorted(owners) == [("", 0), ("", 1)]
    assert sorted(owners.values()) == ["sess-0", "sess-1"]
    assert _registry_violations(owners, pool_size=2) == []


def test_concurrent_release_of_one_candidate_frees_it_once(monkeypatch):
    """AC3d — two concurrent releases of one candidate free it exactly once.

    The registry mapping's ``pop`` is widened so a check-then-pop without
    mutual exclusion would free the slot twice (both releases report success
    and both would emit a release line). The release is one atomic step over
    the registry, so exactly one caller frees it and the other is a no-op.
    """
    import threading
    import time

    candidate = session_registry.acquire_slot_candidate(
        "sess-a", endpoint="", pool_size=2
    )
    assert candidate is not None

    class _SlowPopDict(dict):
        """Registry mapping whose pop yields the GIL (pressure, not behaviour)."""

        def pop(self, key, *default):
            time.sleep(0.05)
            return super().pop(key, *default)

    registry = _SlowPopDict(session_registry._slot_owners)
    monkeypatch.setattr(session_registry, "_slot_owners", registry)

    outcomes: list[bool] = []
    outcomes_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def _release() -> None:
        barrier.wait(timeout=CONCURRENCY_TIMEOUT)
        released = session_registry.release_slot_candidate(candidate, reason="race")
        with outcomes_lock:
            outcomes.append(released)

    threads = [threading.Thread(target=_release, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=CONCURRENCY_TIMEOUT)

    assert not any(thread.is_alive() for thread in threads), (
        "the release step deadlocked (AC3d)"
    )
    assert outcomes.count(True) == 1, (
        f"exactly one release may free the slot, got {outcomes}"
    )
    assert dict(registry) == {}


# ---------------------------------------------------------------------------
# F5 — the added admission latency is measured and bounded
# (LP-0MW959R5S009WDM8, parent AC4)
#
# Candidate-first admission (F2) put two steps on the request hot path: the slot
# candidate acquisition and the residency resolution that prices the verdict
# against it. F3/F4 made those safe; this slice makes them *bounded*. The
# production decision line carries the measured cost per admission
# (``admission_overhead_ms=``), and the checked-in 2026-10-24 thrash fixture is
# replayed through the real admission path to aggregate that figure into the
# p50/p95 the budget is enforced against.
#
# Budget: p95 <= 50 ms. The added work is one in-process registry lookup plus
# the residency resolution — no upstream ``/slots`` query (llama-server holds it
# >15 s under prefill, LP-0MUXTW4MD003TVIC), no new poller and no background
# task, so the guards below are the only new moving parts on the hot path.
# ---------------------------------------------------------------------------

#: The added work's latency budget (parent AC4), owned by the replay harness.
BUDGET_MS = replay_harness.ADMISSION_OVERHEAD_BUDGET_MS

#: Delay injected by the synthetic slow probe (AC4d): comfortably above the
#: budget, small enough that the bounded replay stays fast.
SLOW_PROBE_SECONDS = 0.06


def _overhead_field(caplog, needle: str = "routing_resident_prefix") -> str:
    """The raw ``admission_overhead_ms=`` token on the first decision line."""
    record = next(r for r in caplog.records if needle in r.getMessage())
    match = re.search(r"\badmission_overhead_ms=(\S+)", record.getMessage())
    assert match is not None, record.getMessage()
    return match.group(1)


def _replay_admission(monkeypatch, record, server_config):
    """Run **one** fixture attempt through the real candidate-first admission.

    One admission against an empty 2-slot pool is exactly the case
    candidate-first admission exists for (no slot at decision → assign →
    measure), and it is the path whose added cost the budget bounds.
    """
    _install_slot_state(monkeypatch, slots=_free_slots([0, 1]), owners={})
    _prime_measurement(MEASURED_TOTAL, session=str(record["session"]))
    return provider._should_skip_local(
        MODEL,
        str(record["session"]),
        {"messages": []},
        COLD_THRESHOLD,
        estimated_tokens=int(record["estimated_tokens"]),
        warm_cache_threshold=0,
        resident_prefix_min_ratio=0.5,
        endpoint="",
        server_config=server_config,
    )


def test_candidate_first_admission_logs_its_measured_overhead(
    caplog, monkeypatch, tmp_path
):
    """AC4a/AC4e — a measured admission reports its added cost on the line.

    The figure is the operator's only view of this latency without a replay, so
    it must be present, numeric and non-negative on the ordinary measured path —
    and it must be the *added* work (candidate acquisition + residency
    resolution), which on this host is far below the 50 ms budget.
    """
    import logging

    _prime_measurement(MEASURED_TOTAL)
    _install_slot_state(monkeypatch, slots=_free_slots([0]), owners={})

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _decide_with_pool(server_config=_candidate_config(tmp_path))

    assert (skip, reason) == (False, None)
    token = _overhead_field(caplog)
    assert token != "none", "a measured admission must report its overhead"
    value = float(token)
    assert 0.0 <= value < BUDGET_MS
    # The replay reads this exact token back (AC4a), so the harness and the
    # production field name cannot drift apart silently.
    assert (
        f"{provider.ADMISSION_OVERHEAD_FIELD}=" == replay_harness.ADMISSION_OVERHEAD_TOKEN
    )


def test_disabled_ladder_logs_no_overhead_figure(caplog, monkeypatch):
    """AC4e — an unmeasured admission reports ``none``, never a fabricated 0.

    With the residency gate disabled nothing is acquired or resolved, so there
    is no figure to report: a ``0.000`` would be a fabricated measurement the
    usage report could average in as a real (impossibly fast) admission.
    """
    import logging

    _install_tenancy(["sess-a", "sess-b"])

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _gate_with_tenancy(
            session_id="sess-c",
            view=_view(slots=[], owners={}, monkeypatch=monkeypatch),
            min_ratio=0.0,
            monkeypatch=monkeypatch,
        )

    assert (skip, reason) == (True, SKIP_TENANT_LIMIT)
    assert _overhead_field(caplog) == "none"


def test_measured_admission_issues_no_upstream_slots_query(
    caplog, monkeypatch, tmp_path
):
    """AC4c — the measured path is in-process: no ``/slots`` call.

    Both upstream query helpers raise, so a synchronous ``/slots`` query on the
    admission path — the >15 s hold this item must not regress
    (LP-0MUXTW4MD003TVIC) — fails the test instead of slowing it down. The
    admission still measures its overhead from cached state.
    """
    import logging

    def _boom(*args, **kwargs):
        raise AssertionError("/slots must not be queried on the admission path")

    _prime_measurement(MEASURED_TOTAL)
    _install_slot_state(monkeypatch, slots=_free_slots([0]), owners={})
    monkeypatch.setattr(observability, "_query_slots_detail", _boom)
    monkeypatch.setattr(observability, "_query_slots_progress", _boom)

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        skip, reason = _decide_with_pool(server_config=_candidate_config(tmp_path))

    assert (skip, reason) == (False, None)
    assert _overhead_field(caplog) != "none"


def test_replayed_admission_overhead_is_within_budget(monkeypatch, tmp_path):
    """AC4a/AC4b — the fixture replay's p50/p95 stay inside the 50 ms budget.

    The 2026-10-24 thrash capture (1 978 real admission attempts, 38 sessions)
    is replayed through the **real** admission path; every attempt's own
    ``admission_overhead_ms=`` figure is aggregated into the p50/p95 the AC is
    held to. The figures are fixture-derived (the parent item's open question
    about production representativeness is recorded on the work item), so the
    report carries its provenance.
    """
    _meta, records = replay_harness.load_fixture()
    assert records, "the fixture must carry the captured attempts"
    server_config = _candidate_config(tmp_path)
    sessions = {str(record["session"]) for record in records}

    report = replay_harness.measure_admission_overhead(
        records,
        lambda record: _replay_admission(monkeypatch, record, server_config),
        provenance=(
            "prefill_thrash_2026_10_24.jsonl "
            f"({len(records)} attempts, {len(sessions)} sessions)"
        ),
    )

    assert report.measured > 0, report.to_json()
    assert report.p50_ms is not None and report.p95_ms is not None
    assert 0.0 <= report.p50_ms <= report.p95_ms
    assert report.within_budget, report.to_json()
    assert not report.over_budget


def test_synthetic_slow_probe_trips_the_latency_budget(monkeypatch, tmp_path):
    """AC4d — a synthetic slow probe trips the > 50 ms failure.

    The probe delays the residency resolution by 60 ms, so the measured p95 must
    exceed the budget and the report must say so. Without this pin the budget
    could pass on a degraded measurement and silently drop the guard.
    """
    _meta, records = replay_harness.load_fixture()
    measurable = [
        record
        for record in records
        if int(record["estimated_tokens"]) > COLD_THRESHOLD
    ][:3]
    assert measurable, "the fixture must carry admissions the ladder prices"
    server_config = _candidate_config(tmp_path)
    real_resolve = provider.resolve_resident_prefix

    def _slow_resolve(*args, **kwargs):
        time.sleep(SLOW_PROBE_SECONDS)
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(provider, "resolve_resident_prefix", _slow_resolve)

    report = replay_harness.measure_admission_overhead(
        measurable,
        lambda record: _replay_admission(monkeypatch, record, server_config),
        provenance="synthetic slow probe (AC4d)",
    )

    assert report.measured == len(measurable), report.to_json()
    assert report.p95_ms > BUDGET_MS, report.to_json()
    assert report.over_budget is True
    assert report.within_budget is False


def test_overhead_budget_verdict_uses_the_measured_p95():
    """AC4d — the verdict reads the nearest-rank p95, and never an absent one.

    An unmeasured report is neither over budget nor within it: a missing figure
    must fail closed rather than read as a passed budget.
    """
    assert replay_harness.percentile([], 95) is None
    assert replay_harness.percentile([4.0], 95) == 4.0
    samples = [1.0, 2.0, 3.0, 4.0, 100.0]
    # Nearest rank: the reported percentile is a duration the path really paid,
    # never an interpolation of two samples.
    assert replay_harness.percentile(samples, 50) == 3.0
    assert replay_harness.percentile(samples, 95) == 100.0

    inside = replay_harness.admission_overhead_report(samples[:4], attempts=4)
    assert (inside.p50_ms, inside.p95_ms) == (2.0, 4.0)
    assert inside.within_budget is True
    assert inside.over_budget is False

    breached = replay_harness.admission_overhead_report(samples, attempts=5)
    assert breached.over_budget is True
    assert breached.within_budget is False

    unmeasured = replay_harness.admission_overhead_report([], attempts=7)
    assert unmeasured.p95_ms is None
    assert unmeasured.over_budget is False
    assert unmeasured.within_budget is False, "an absent figure is not a pass"
