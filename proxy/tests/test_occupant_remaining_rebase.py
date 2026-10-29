"""Token-based occupant-remaining re-pricing (LP-0MVS9YK9P006M1NQ).

The admission guard's occupant-remaining term used to decay the occupant's
*optimistic* admission ETA: ``max(0, eta_prefill_s - (now - started_at))``.
Because ``eta_prefill_s`` is the uncontended estimate (measured 2-4x optimistic
for an occupant that shares the batch), that term under-priced the occupant and
converged to ``0`` the instant the occupant passed its optimistic ETA — exactly
when it was most expensive (measured 2026-10-28 00:28:55: a cold 65-89k-token
audit prompt priced ``eta_occupancy_s=16.0``; the newcomer queued the full 60 s,
the gate never freed, and the dispatch fell back remote).

This file pins the re-priced contract:

- AC1: the remaining service is priced from the occupant's remaining *uncached
  new tokens* at the contended rate; the optimistic ``eta - elapsed`` decay no
  longer determines the result (an occupant past its optimistic ETA with 60k
  new tokens still reports a large remaining service).
- AC2: a cold 60k-token occupant prices at >= the contended rate x its
  remaining new tokens, and a 6,659-token newcomer behind it still projects
  >= 55 s.
- AC3: fail-open preserved: records with missing/invalid ``new_tokens`` or
  ``eta_prefill_s`` contribute ``0.0``; inactive / first-byte / self /
  other-endpoint records are skipped; an uncalibrated contended window never
  guesses a multiplier.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from proxy.router_helpers import _live_prefill_remaining_seconds

from proxy import prefill_eta

#: The measured prompt-eval median for the contended case (~120 t/s). The
#: estimator applies its 1.2x conservatism factor on top, so the *priced*
#: rate is ~100 t/s — every assertion below is a floor, never an exact fit.
_CONTENDED_MS_PER_TOKEN = 8.33

#: Seed a token count inside each context bucket the tests price:
#: bucket 0 (0..16 383) for the 6 659-token newcomer, bucket 1 (16 384..65 535)
#: for the 60 000-token occupant.
_BUCKET0_SEED_TOKENS = 8_000
_BUCKET1_SEED_TOKENS = 40_000

_MODEL = "Qwen3"


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
def _clean_calibration():
    prefill_eta.reset()
    yield
    prefill_eta.reset()


def _seed_contended(
    tokens: int,
    ms_per_token: float = _CONTENDED_MS_PER_TOKEN,
    count: int = 6,
) -> None:
    """Seed a calibrated contended window for *tokens*' context bucket."""
    for _ in range(count):
        prefill_eta.record_dispatch_sample(
            _MODEL, tokens, 0.0, tokens * ms_per_token, contended=True
        )


def _occupant(
    *,
    new_tokens,
    eta=100.0,
    started_ago: float = 0.0,
    active: bool = True,
    first_byte: bool = False,
    endpoint: str | None = None,
    model: str | None = _MODEL,
) -> dict:
    """A live dispatch record shaped like the proxy's ``_dispatch_evidence``."""
    return {
        "active": active,
        "first_byte_seen": first_byte,
        "eta_prefill_s": eta,
        "started_at": time.monotonic() - started_ago,
        "new_tokens": new_tokens,
        "cached_ratio": 0.0,
        "model_name": model,
        "endpoint": endpoint,
        "backend": endpoint or "http://192.0.2.9:8080",
    }


def _srv(records: dict) -> SimpleNamespace:
    # No ``config`` attribute → the estimator's default guard settings
    # (min_samples=5); each test seeds 6 samples so the contended window fits.
    return SimpleNamespace(local_dispatch_records=records)


# ---------------------------------------------------------------------------
# AC1 — the optimistic eta - elapsed decay no longer determines the result
# ---------------------------------------------------------------------------

def test_occupant_past_its_optimistic_eta_still_owes_its_remaining_new_tokens():
    """AC1: an occupant 150 s into a prompt whose *optimistic* ETA was 100 s
    still owes most of its 60k new tokens. The old decay reports 0.0 here."""
    _seed_contended(_BUCKET1_SEED_TOKENS)
    srv = _srv({
        "occ": _occupant(new_tokens=60_000, eta=100.0, started_ago=150.0),
    })

    remaining = _live_prefill_remaining_seconds(srv)

    # The pre-change arithmetic, ``max(0, 100 - 150)``, is 0.0; the re-priced
    # term reports at least half of the full contended service (>= 300 s).
    assert remaining > 0.0
    assert remaining >= 60_000 * (_CONTENDED_MS_PER_TOKEN / 1000.0) * 0.5


def test_remaining_service_shrinks_as_the_occupant_makes_progress():
    """The re-pricing still decays with progress, just against the contended
    full duration rather than the optimistic ETA — not a constant."""
    _seed_contended(_BUCKET1_SEED_TOKENS)
    fresh = _live_prefill_remaining_seconds(
        _srv({"occ": _occupant(new_tokens=60_000, eta=100.0, started_ago=0.0)})
    )
    advanced = _live_prefill_remaining_seconds(
        _srv({"occ": _occupant(new_tokens=60_000, eta=100.0, started_ago=300.0)})
    )

    assert advanced < fresh
    assert advanced > 0.0


# ---------------------------------------------------------------------------
# AC2 — 60k cold occupant and a small newcomer behind it
# ---------------------------------------------------------------------------

def test_cold_occupant_prices_at_least_the_contended_rate_times_remaining_tokens():
    """AC2: a freshly admitted 60k-token occupant owes at least the contended
    rate x its remaining new tokens (>= 60 000 x 8.33 ms ≈ 500 s)."""
    _seed_contended(_BUCKET1_SEED_TOKENS)
    srv = _srv({
        "occ": _occupant(new_tokens=60_000, eta=100.0, started_ago=0.0),
    })

    occupancy = _live_prefill_remaining_seconds(srv)

    assert occupancy >= 60_000 * (_CONTENDED_MS_PER_TOKEN / 1000.0)


def test_small_newcomer_behind_the_cold_occupant_projects_past_55s():
    """AC2: the 6 659-token newcomer's projected wait includes the re-priced
    occupant and clears the 55 s floor from the measured 13.9 s case."""
    _seed_contended(_BUCKET0_SEED_TOKENS)
    _seed_contended(_BUCKET1_SEED_TOKENS)
    occupancy = _live_prefill_remaining_seconds(
        _srv({"occ": _occupant(new_tokens=60_000, eta=100.0, started_ago=0.0)})
    )

    decision = prefill_eta.evaluate_admission(
        _cfg(), _MODEL, 6_659, 0.0, "low",
        busy_slots=1, occupant_prefill_seconds=occupancy,
    )

    assert decision.admit is False
    assert decision.projected_wait_seconds >= 55.0
    assert decision.eta_occupancy_seconds >= 55.0


# ---------------------------------------------------------------------------
# AC3 — fail-open preserved
# ---------------------------------------------------------------------------

def test_missing_or_malformed_evidence_contributes_nothing():
    """AC3: missing/invalid ``new_tokens`` or ``eta_prefill_s`` contributes
    0.0; inactive and first-byte records are skipped."""
    _seed_contended(_BUCKET1_SEED_TOKENS)
    records = {
        "no-new-tokens": _occupant(new_tokens=None, eta=100.0),
        "bad-new-tokens": _occupant(new_tokens="lots", eta=100.0),
        "negative-new-tokens": _occupant(new_tokens=-5, eta=100.0),
        "no-eta": _occupant(new_tokens=60_000, eta=None),
        "bad-eta": _occupant(new_tokens=60_000, eta="slow"),
        "no-started-at": {
            k: v
            for k, v in _occupant(new_tokens=60_000, eta=100.0).items()
            if k != "started_at"
        },
        "first-byte": _occupant(new_tokens=60_000, eta=100.0, first_byte=True),
        "inactive": _occupant(new_tokens=60_000, eta=100.0, active=False),
        "junk": "not-a-record",
    }

    assert _live_prefill_remaining_seconds(_srv(records)) == 0.0


def test_uncalibrated_contended_window_never_guesses():
    """AC3: with no calibrated window the term fails open to 0.0 rather than
    pricing from a guessed multiplier (the guard's founding rule)."""
    # No seeding at all → neither the contended nor the solitary window is
    # calibrated, so the remaining service is not priced.
    srv = _srv({"occ": _occupant(new_tokens=60_000, eta=100.0, started_ago=0.0)})
    assert _live_prefill_remaining_seconds(srv) == 0.0


def test_self_and_other_endpoint_records_still_excluded():
    """AC3: the session/endpoint filters behave exactly as before."""
    _seed_contended(_BUCKET0_SEED_TOKENS)
    records = {
        ("http://a:8080", "me"): _occupant(
            new_tokens=8_000, endpoint="http://a:8080"
        ),
        ("http://a:8080", "other"): _occupant(
            new_tokens=8_000, endpoint="http://a:8080"
        ),
        ("http://b:8080", "far"): _occupant(
            new_tokens=8_000, endpoint="http://b:8080"
        ),
    }
    srv = _srv(records)

    included = _live_prefill_remaining_seconds(srv)
    without_self = _live_prefill_remaining_seconds(srv, exclude_session="me")
    only_a = _live_prefill_remaining_seconds(srv, endpoint="http://a:8080")

    assert without_self < included
    assert only_a < included
    # Excluding self *and* scoping to endpoint "a" leaves only "other", so
    # the combined filter is strictly smaller than the endpoint scope alone.
    combined = _live_prefill_remaining_seconds(
        srv, exclude_session="me", endpoint="http://a:8080"
    )
    assert 0.0 < combined < only_a
