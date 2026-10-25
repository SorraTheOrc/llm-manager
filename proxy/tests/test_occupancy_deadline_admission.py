"""Occupancy-aware deadline admission (LP-0MVOFVCA2008DQVE).

Closes the parent's re-scoped residual (producer decision 2026-10-25): a
within-budget, mostly warm prompt was admitted locally, then cancelled by
``local_prefill_deadline_seconds`` with **zero prefill progress** because the
other slot was mid-prefill with a long occupant. Measured 2026-10-25
11:46:50–19:11 across 7.4 h: 22 ``local_eviction_after_acceptance
reason=prefill_deadline`` events (12 user-traffic), the common signature
``eta_prefill_s=7.1 elapsed_s=120.1 progress=None alive=False in_flight=True``
— the guard priced the occupant at the newcomer's own ETA (symmetric
``eta * busy_slots``) and the deadline bound ignored occupancy entirely.

Covered acceptance criteria:

- AC1: own ETA within the deadline but projected time to first byte (own +
  remaining occupant prefill) beyond it ⇒ ``reason=prefill_eta_exceeds_deadline``
  and the request falls through — the 2026-10-25 residual shape.
- AC2: the remaining occupant service is projected from each occupant's own
  dispatch record (``eta_prefill_s`` minus elapsed) and never raises on
  malformed evidence.
- AC3: the symmetric floor is never lowered (no occupant evidence ⇒ the
  pre-change verdict, byte-for-byte).
- AC4: both the queue and the direct free-slot admission paths carry the
  verdict, visible in the ``local_admission_guard`` log.
- AC5: ``POST /llama/local/reserve`` feeds the same projection (verdicts
  cannot drift).
- AC6: ``observe`` logs without refusing; deadline 0 and an uncalibrated
  bucket still admit.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from unittest.mock import patch

import proxy.provider as provider
import pytest

from proxy import admission_reserve, contention_queue, prefill_eta


def _cfg(**server_overrides) -> dict:
    server = {
        "session_slot_pool_size": 2,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": 60,
        "contention_queue_max_depth": 8,
        "local_prefill_deadline_seconds": 120,
    }
    server.update(server_overrides)
    return {"provider_cooldown_seconds": 60, "server": server}


@pytest.fixture(autouse=True)
def _clean_state():
    contention_queue.reset()
    prefill_eta.reset()
    admission_reserve.reset()
    yield
    contention_queue.reset()
    prefill_eta.reset()
    admission_reserve.reset()


#: Small-prompt calibration → own ETA in the residual's ~7 s band.
_WARM_TOKENS = 2_000
_WARM_MS_PER_TOKEN = 3.5


def _seed_calibration(
    tokens: int = _WARM_TOKENS, ms_per_token: float = _WARM_MS_PER_TOKEN,
    count: int = 6,
) -> None:
    for _ in range(count):
        prefill_eta.record_dispatch_sample("Qwen3", tokens, 0.0, tokens * ms_per_token)


# ---------------------------------------------------------------------------
# AC1 — the 2026-10-25 residual shape
# ---------------------------------------------------------------------------

def test_warm_prompt_behind_a_long_occupant_is_refused_at_the_deadline():
    """AC1: own ETA ≈ 7 s and a single occupant owing ~115 s ⇒ projected
    first byte ~122 s >= the 120 s deadline ⇒ refused with the deadline
    reason, even though both the own ETA and the projected queue wait
    (7+7=14 s) fit the 60 s budget. This is the exact shape the old guard
    admitted and let the deadline cancel."""
    _seed_calibration()
    cfg = _cfg()

    own = prefill_eta.evaluate_admission(cfg, "Qwen3", _WARM_TOKENS, 0.0, "low")
    assert own.admit is True, "fixture sanity: admitted without occupancy"
    assert own.eta_prefill_seconds < 120.0

    # A context_tokens-less projection with the symmetric floor (busy=1)
    # still admits — the fix must come from the measured occupant evidence.
    symmetric = prefill_eta.evaluate_admission(
        cfg, "Qwen3", _WARM_TOKENS, 0.0, "low", busy_slots=1
    )
    assert symmetric.admit is True

    decision = prefill_eta.evaluate_admission(
        cfg, "Qwen3", _WARM_TOKENS, 0.0, "low",
        busy_slots=1, occupant_prefill_seconds=115.0,
    )

    assert decision.admit is False
    assert decision.reason == "prefill_eta_exceeds_deadline"
    assert decision.active is True
    assert decision.prefill_deadline_seconds == 120.0
    assert decision.eta_prefill_seconds < decision.prefill_deadline_seconds
    assert decision.eta_occupancy_seconds == pytest.approx(115.0, abs=0.5)
    assert decision.projected_wait_seconds >= decision.prefill_deadline_seconds


def test_occupant_remaining_slightly_under_the_deadline_still_admits():
    """The occupancy-aware deadline is not a blanket refusal behind any
    occupant: a warm prompt whose projected first byte fits stays admitted."""
    _seed_calibration()
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", _WARM_TOKENS, 0.0, "low",
        busy_slots=1, occupant_prefill_seconds=5.0,
    )
    assert decision.admit is True
    assert decision.reason == "within_budget"


def test_occupancy_deadline_boundary_is_an_inclusive_refusal():
    """projected first byte == deadline refuses (>= semantics, matching the
    own-ETA boundary the deadline bound already used)."""
    _seed_calibration()
    own = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", _WARM_TOKENS, 0.0, "low"
    )
    eta = own.eta_prefill_seconds
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", _WARM_TOKENS, 0.0, "low",
        busy_slots=1,
        occupant_prefill_seconds=120.0 - eta,
    )
    assert decision.admit is False
    assert decision.reason == "prefill_eta_exceeds_deadline"


# ---------------------------------------------------------------------------
# AC2 — occupant-evidence projection from live dispatch records
# ---------------------------------------------------------------------------

def _occupant_record(
    eta: float = 115.0,
    *,
    started_ago: float = 2.0,
    active: bool = True,
    first_byte: bool = False,
    endpoint: str | None = None,
) -> dict:
    return {
        "active": active,
        "first_byte_seen": first_byte,
        "eta_prefill_s": eta,
        "started_at": time.monotonic() - started_ago,
        "endpoint": endpoint,
        "backend": endpoint or "http://192.0.2.9:8080",
    }


def test_remaining_seconds_prices_each_occupant_from_its_own_evidence():
    from proxy.router_helpers import _live_prefill_remaining_seconds

    srv = SimpleNamespace(
        local_dispatch_records={
            "occ-a": _occupant_record(eta=115.0, started_ago=2.0),
            "occ-b": _occupant_record(eta=30.0, started_ago=40.0),  # fully elapsed → 0
            "done": _occupant_record(eta=9.0, first_byte=True),
            "idle": _occupant_record(eta=9.0, active=False),
        }
    )
    assert _live_prefill_remaining_seconds(srv) == pytest.approx(113.0, abs=0.5)


def test_remaining_seconds_excludes_the_requesting_session():
    from proxy.router_helpers import _live_prefill_remaining_seconds

    srv = SimpleNamespace(
        local_dispatch_records={
            ("http://192.0.2.9:8080", "s-1"): _occupant_record(eta=60.0, started_ago=0.0),
            "s-1": _occupant_record(eta=60.0, started_ago=0.0),
            "other": _occupant_record(eta=12.0, started_ago=0.0),
        }
    )
    assert _live_prefill_remaining_seconds(srv, exclude_session="s-1") == pytest.approx(12.0)


def test_remaining_seconds_scopes_to_endpoint_when_known():
    from proxy.router_helpers import _live_prefill_remaining_seconds

    srv = SimpleNamespace(
        local_dispatch_records={
            "a": _occupant_record(eta=10.0, started_ago=0.0, endpoint="http://a:8080"),
            "b": _occupant_record(eta=99.0, started_ago=0.0, endpoint="http://b:8080"),
            "legacy": _occupant_record(eta=5.0, started_ago=0.0, endpoint=None),
        }
    )
    assert (
        _live_prefill_remaining_seconds(srv, endpoint="http://a:8080")
        == pytest.approx(15.0)  # 10 (scoped) + 5 (legacy, unscoped)
    )


def test_remaining_seconds_fails_open_on_malformed_evidence():
    from proxy.router_helpers import _live_prefill_remaining_seconds

    bad_records = {
        "no-eta": {"active": True, "first_byte_seen": False, "started_at": 0.0},
        "bad-eta": _occupant_record(eta="slow"),
        "nan-eta": _occupant_record(eta=float("nan")),
        "neg-eta": _occupant_record(eta=-3.0),
        "no-started": {"active": True, "first_byte_seen": False, "eta_prefill_s": 9.0},
        "nan-started": {**_occupant_record(eta=9.0), "started_at": float("inf")},
        "not-a-record": "junk",
    }
    assert _live_prefill_remaining_seconds(SimpleNamespace(local_dispatch_records=bad_records)) == 0.0
    assert _live_prefill_remaining_seconds(SimpleNamespace()) == 0.0
    assert _live_prefill_remaining_seconds(SimpleNamespace(local_dispatch_records=[])) == 0.0


# ---------------------------------------------------------------------------
# AC3 — the symmetric floor is never lowered
# ---------------------------------------------------------------------------

def test_symmetric_floor_is_never_lowered_by_a_smaller_measured_remnant():
    _seed_calibration()
    floor = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", _WARM_TOKENS, 0.0, "low", busy_slots=2
    )
    measured = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", _WARM_TOKENS, 0.0, "low",
        busy_slots=2, occupant_prefill_seconds=1.0,
    )
    assert measured.eta_occupancy_seconds == pytest.approx(
        floor.eta_occupancy_seconds, abs=0.1
    )
    assert measured.projected_wait_seconds == pytest.approx(
        floor.projected_wait_seconds, abs=0.1
    )
    assert measured.admit == floor.admit


def test_no_occupant_evidence_is_byte_for_byte_the_old_verdict():
    _seed_calibration()
    for busy in (0, 1, 2):
        old = prefill_eta.evaluate_admission(
            _cfg(), "Qwen3", _WARM_TOKENS, 0.0, "low", busy_slots=busy
        )
        new = prefill_eta.evaluate_admission(
            _cfg(), "Qwen3", _WARM_TOKENS, 0.0, "low",
            busy_slots=busy, occupant_prefill_seconds=None,
        )
        assert new.reason == old.reason
        assert new.admit == old.admit
        assert new.projected_wait_seconds == old.projected_wait_seconds


# ---------------------------------------------------------------------------
# AC6 — fail-open semantics preserved
# ---------------------------------------------------------------------------

def test_observe_policy_logs_the_occupancy_deadline_refusal_without_refusing():
    _seed_calibration()
    decision = prefill_eta.evaluate_admission(
        _cfg(local_admission_guard_policy="observe"),
        "Qwen3", _WARM_TOKENS, 0.0, "low",
        busy_slots=1, occupant_prefill_seconds=115.0,
    )
    assert decision.admit is True
    assert decision.reason == "prefill_eta_exceeds_deadline"
    assert decision.policy == "observe"


def test_deadline_zero_disables_the_occupancy_bound():
    """Deadline 0 = documented rollback: the occupancy evidence no longer
    refuses at the deadline (the queue budget still binds, as it always did —
    here it is raised out of the way so the deadline is the only possible
    bound)."""
    _seed_calibration()
    decision = prefill_eta.evaluate_admission(
        _cfg(local_prefill_deadline_seconds=0, contention_queue_max_wait_seconds=2_000),
        "Qwen3", _WARM_TOKENS, 0.0, "low", busy_slots=1,
        occupant_prefill_seconds=1_000.0,
    )
    assert decision.admit is True
    assert decision.reason == "within_budget"
    assert decision.prefill_deadline_seconds == 0.0


def test_uncalibrated_bucket_still_admits_with_occupant_evidence():
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", _WARM_TOKENS, 0.0, "low", busy_slots=1,
        occupant_prefill_seconds=115.0,
    )
    assert decision.admit is True
    assert decision.reason == "insufficient_samples"


# ---------------------------------------------------------------------------
# AC4 — the guard paths carry the verdict (queue + direct free-slot)
# ---------------------------------------------------------------------------

class _RecordSrv:
    """Stand-in for the proxy server with a live occupant record."""

    def __init__(self, records: dict, occupied: bool = True):
        self.local_dispatch_records = records
        self.local_prefill_in_flight = (
            {"occ-session"} if occupied else set()
        )
        self.local_generating_queries = 0
        self.local_generating_sessions = set()


async def _guard_decision(records: dict, *, session_id: str = "s-new"):
    srv = _RecordSrv(records)
    cfg = _cfg()
    with (
        patch("proxy.server", srv),
        patch("proxy.provider._get_local_concurrency_info", lambda _c, endpoint=None: (1, 2)),
    ):
        return await provider._prefill_admission_guard(
            cfg,
            {"name": "local-llama", "llama_model": "Qwen3"},
            _WARM_TOKENS,
            session_id,
            priority="low",
            max_local=2,
            endpoint=None,
        )


@pytest.mark.asyncio
async def test_guard_refuses_when_a_live_occupant_owes_the_budget(caplog):
    """AC4 queue path: a live occupant record (``eta_prefill_s=115`` started
    2 s ago, no first byte) pushes the projection past the deadline ⇒ the
    guard refuses, and the log carries the occupancy evidence."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_calibration()

    decision = await _guard_decision(
        {"occ-session": _occupant_record(eta=115.0, started_ago=2.0)}
    )

    assert decision is not None
    assert decision.admit is False
    assert decision.reason == "prefill_eta_exceeds_deadline"
    assert decision.eta_occupancy_seconds == pytest.approx(113.0, abs=0.5)

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "local_admission_guard" in messages
    assert "decision=reject" in messages
    assert "reason=prefill_eta_exceeds_deadline" in messages
    assert "eta_occupancy_s=" in messages
    assert "projected_wait_s=" in messages


@pytest.mark.asyncio
async def test_guard_excludes_own_session_record_from_occupancy():
    """AC4: a session's own record never blocks its own re-entry."""
    _seed_calibration()
    decision = await _guard_decision(
        {"s-new": _occupant_record(eta=115.0, started_ago=2.0)},
        session_id="s-new",
    )
    assert decision is not None
    assert decision.admit is True


@pytest.mark.asyncio
async def test_direct_free_slot_path_keeps_the_deadline_reason(caplog):
    """AC4 direct path: the same guard invocation with ``free_slot=True`` /
    ``reason_label`` still reports the deadline reason (never the direct
    suffix) and refuses — the exact verdict a queued newcomer gets."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_calibration()
    srv = _RecordSrv({"occ-session": _occupant_record(eta=115.0, started_ago=2.0)})
    with (
        patch("proxy.server", srv),
        patch("proxy.provider._get_local_concurrency_info", lambda _c, endpoint=None: (1, 2)),
    ):
        decision = await provider._prefill_admission_guard(
            _cfg(),
            {"name": "local-llama", "llama_model": "Qwen3"},
            _WARM_TOKENS,
            "s-new",
            priority="low",
            max_local=2,
            reason_label="eta_exceeds_budget_direct",
            free_slot=True,
        )

    assert decision is not None
    assert decision.admit is False
    assert decision.reason == "prefill_eta_exceeds_deadline"
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "local_admission_guard decision=reject reason=prefill_eta_exceeds_deadline" in messages


# ---------------------------------------------------------------------------
# AC5 — the reserve endpoint shares the projection
# ---------------------------------------------------------------------------

def test_reserve_verdict_matches_the_guard_for_the_same_live_state():
    _seed_calibration()
    cfg = _cfg()
    body = {"estimated_tokens": _WARM_TOKENS, "session_id": "s-1"}

    guard = prefill_eta.evaluate_admission(
        cfg, "Qwen3", _WARM_TOKENS, 0.0, "low",
        busy_slots=1, in_flight_prefills=1, occupant_prefill_seconds=115.0,
    )
    status, payload = admission_reserve.reserve(
        body, cfg, model="Qwen3",
        busy_slots=1, in_flight_prefills=1, occupant_prefill_seconds=115.0,
    )

    assert status == 429
    assert payload["admit"] is False
    assert payload["reason"] == guard.reason == "prefill_eta_exceeds_deadline"
    assert payload["queue_eta_s"] >= 120.0


@pytest.mark.asyncio
async def test_reserve_handler_probes_the_same_live_records():
    """AC5 wiring: ``POST /llama/local/reserve`` feeds the live occupant
    projection into the shared reserve verdict (captured by stub), so the
    handler and the guard cannot drift for identical live state."""
    from proxy.handlers import reserve_local_slot

    _seed_calibration()
    seen: dict = {}

    class _Body:
        async def json(self):
            return {"estimated_tokens": _WARM_TOKENS, "session_id": "s-1"}

    class _Request:
        def __init__(self):
            self.body = _Body()
            self.headers = {}

        async def json(self):
            return await self.body.json()

    def _fake_reserve(body, config, **kwargs):
        seen.update(kwargs)
        seen["body"] = body
        return 200, {"admit": True}

    srv = SimpleNamespace(
        config=_cfg(),
        current_model="Qwen3",
        local_dispatch_records={"occ-session": _occupant_record(eta=115.0, started_ago=2.0)},
        local_prefill_in_flight={"occ-session"},
        local_generating_queries=0,
        local_generating_sessions=set(),
    )
    with (
        patch("proxy.server", srv),
        patch("proxy.provider._local_concurrency_info", lambda _c, endpoint=None: (1, 2)),
        patch("proxy.admission_reserve.reserve", new=_fake_reserve),
    ):
        response = await reserve_local_slot(_Request())

    assert response.status_code == 200
    assert seen["body"]["session_id"] == "s-1"
    assert seen["busy_slots"] == 1
    assert seen["in_flight_prefills"] == 1
    assert seen["occupant_prefill_seconds"] == pytest.approx(113.0, abs=0.5)


# ---------------------------------------------------------------------------
# AC8 — the guard never breaks routing on an observability gap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_guard_never_raises_when_records_are_unavailable():
    _seed_calibration()
    # A server without a records attribute must be treated like an idle gate.
    decision = await _guard_decision({})
    assert decision is not None
    assert decision.admit is True
