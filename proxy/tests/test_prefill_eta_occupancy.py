"""Occupancy-aware prefill admission (LP-0MVICD2YW004DJPV).

The guard used to compare the caller's *own* projected prefill against the
wait budget, ignoring the prefill already in flight ahead of it. With prefills
serialised (``local_max_concurrent_prefills: 1``) the true newcomer wait is its
own prefill **plus** the occupied gate, a quantity the proxy already publishes
as ``queue_eta_seconds``. Result: the guard admitted waits its own status
payload had flagged as unservable (2026-10-21 window: 302 queued waits, 261
fell back after queueing, 215 at the full 120 s budget, 25,804 s wasted).

Covered acceptance criteria:

- AC1: own ETA within budget but projected wait (own + occupancy) above it ⇒
  ``admit=False`` and the request falls through without queueing.
- AC2: the guard's projected wait equals the published ``queue_eta_seconds``
  for the same ``(model, tokens, busy_slots)`` inputs — the two can never
  drift apart again.
- AC3: fail-open preserved — uncalibrated, no budget (``high`` priority),
  ``observe`` policy all still admit; ``policy=off`` never evaluates.
- AC4: an in-flight prefill with unobservable progress occupies the gate for
  at least the calibrated estimate, and the decision is logged with the
  occupancy evidence.
- AC6: the admission log line carries own ETA, occupancy term, projected wait,
  budget, busy slots and in-flight prefills.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import proxy.provider as provider
import pytest
from fastapi import Response

from proxy import contention_queue, prefill_eta


class _DummyRequest:
    """Minimal request stub (matches the other admission tests)."""

    def __init__(self, body: bytes = b'{"model":"test"}'):
        self._body = body
        self.headers = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


class _MutableConcurrency:
    """Mutable ``_get_local_concurrency_info`` stand-in."""

    def __init__(self, active: int = 0, max_: int = 1):
        self.active = active
        self.max = max_

    def __call__(self, config, endpoint=None) -> tuple[int, int]:
        return (self.active, self.max)


def _ok_response() -> Response:
    return Response(
        content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
        status_code=200,
        media_type="application/json",
    )


def _cfg(**server_overrides) -> dict:
    server = {
        "session_slot_pool_size": 1,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": 60,
        "contention_queue_max_depth": 4,
    }
    server.update(server_overrides)
    return {"provider_cooldown_seconds": 60, "server": server}


#: Tokens whose calibrated own-ETA (~56.2 s) fits the 60 s budget while the
#: projected wait with one occupant (~112.4 s) does not. This is exactly the
#: shape the old guard got wrong.
_TOKENS = 12_000
_MS_PER_TOKEN = 3.9
_BUDGET = 60.0


@pytest.fixture(autouse=True)
def _clean_state():
    contention_queue.reset()
    prefill_eta.reset()
    yield
    contention_queue.reset()
    prefill_eta.reset()


def _seed_calibration(
    tokens: int = _TOKENS, ms_per_token: float = _MS_PER_TOKEN, count: int = 6
) -> None:
    for _ in range(count):
        prefill_eta.record_dispatch_sample("Qwen3", tokens, 0.0, tokens * ms_per_token)


@contextmanager
def _guard_patches(tokens: int = _TOKENS, cached_ratio: float = 0.0):
    with (
        patch(
            "proxy.provider._estimate_effective_prompt_tokens_for_routing",
            new=AsyncMock(return_value=tokens),
        ),
        patch("proxy.provider._get_tokenizer_for_model", lambda *_: (None, 1.0)),
        patch("proxy.provider._should_skip_local", lambda *a, **k: (False, None)),
        patch("proxy.provider._get_cached_ratio", lambda *_: cached_ratio),
    ):
        yield


# ---------------------------------------------------------------------------
# Shared estimator
# ---------------------------------------------------------------------------

def test_projected_wait_is_own_prefill_times_occupancy():
    _seed_calibration()
    projected = prefill_eta.projected_wait_seconds("Qwen3", _TOKENS, 0.0, 2, _cfg())

    assert projected.busy_slots == 2
    assert projected.own_prefill_seconds == pytest.approx(56.2, abs=0.1)
    assert projected.occupancy_seconds == pytest.approx(
        projected.own_prefill_seconds * 2, abs=0.1
    )
    assert projected.projected_seconds == pytest.approx(168.5, abs=0.2)


def test_projected_wait_negative_or_missing_occupancy_is_zero():
    _seed_calibration()
    for busy in (-3, 0, None):
        projected = prefill_eta.projected_wait_seconds(
            "Qwen3", _TOKENS, 0.0, busy, _cfg()
        )
        assert projected.busy_slots == 0
        assert projected.projected_seconds == pytest.approx(
            projected.own_prefill_seconds
        )


# ---------------------------------------------------------------------------
# AC1: occupancy-aware refusal
# ---------------------------------------------------------------------------

def test_own_eta_within_budget_but_projected_wait_above_it_is_refused():
    """AC1: own ETA < budget < projected wait ⇒ refuse."""
    _seed_calibration()
    own = prefill_eta.evaluate_admission(_cfg(), "Qwen3", _TOKENS, 0.0, "low")
    assert own.admit is True, "fixture sanity: own prefill fits the budget"
    assert own.eta_prefill_seconds < _BUDGET

    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", _TOKENS, 0.0, "low", busy_slots=1
    )

    assert decision.admit is False
    assert decision.reason == "eta_exceeds_budget"
    assert decision.projected_wait_seconds > _BUDGET
    assert decision.eta_occupancy_seconds == pytest.approx(
        decision.eta_prefill_seconds
    )
    assert decision.busy_slots == 1


def test_zero_occupancy_keeps_legacy_verdict():
    """Backcompat: with no occupants the verdict is the pre-change one."""
    _seed_calibration()
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", _TOKENS, 0.0, "low", busy_slots=0
    )
    assert decision.admit is True
    assert decision.reason == "within_budget"
    assert decision.projected_wait_seconds == pytest.approx(
        decision.eta_prefill_seconds
    )


@pytest.mark.asyncio
async def test_occupancy_refusal_never_enters_the_queue(caplog):
    """AC1 end-to-end: no queue wait, no local dispatch, remote takes over."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_calibration()
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {"name": "remote-fallback", "type": "remote",
             "endpoint": "https://api.example.com/v1"},
        ],
        "aliases": ["hybrid*"],
    }
    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch(
            "proxy.provider._get_local_concurrency_info", _MutableConcurrency(1, 1)
        ),
        patch("proxy.provider.request_priority", return_value="low"),
        _guard_patches(),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", config, _cfg()
        )

    assert result.status_code == 200
    assert calls == ["remote"], "a refused request must not dispatch local"
    assert contention_queue.queue_depth() == 0
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "local_admission_guard decision=reject" in messages
    assert "contention_queue_wait" not in messages
    assert "contention_queue_fallback_after_queue" not in messages


# ---------------------------------------------------------------------------
# AC2: guard and published status agree
# ---------------------------------------------------------------------------

def test_guard_projected_wait_matches_published_queue_eta():
    """AC2: same ``(model, tokens, busy_slots)`` ⇒ identical numbers."""
    _seed_calibration()
    busy = 1

    published = prefill_eta.status_estimate(
        _cfg(), "Qwen3", total_slots=1, available_slots=1 - busy
    )
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", _TOKENS, 0.0, "low", busy_slots=busy
    )

    assert published["queue_eta_busy_slots"] == busy
    assert published["prefill_eta_seconds"] == pytest.approx(
        decision.eta_prefill_seconds, abs=0.1
    )
    assert published["queue_eta_seconds"] == pytest.approx(
        decision.projected_wait_seconds, abs=0.1
    )
    assert published["queue_eta_occupancy_seconds"] == pytest.approx(
        decision.eta_occupancy_seconds, abs=0.1
    )


def test_published_queue_eta_flags_the_budget_it_enforces():
    """The measured smoking gun: published queue ETA > budget while the old
    own-ETA comparison still admitted."""
    _seed_calibration()
    published = prefill_eta.status_estimate(
        _cfg(), "Qwen3", total_slots=1, available_slots=0
    )

    assert published["queue_eta_seconds"] > published["prefill_budget_seconds"]
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", _TOKENS, 0.0, "low",
        busy_slots=published["queue_eta_busy_slots"],
    )
    assert decision.admit is False


# ---------------------------------------------------------------------------
# AC3: fail-open preserved
# ---------------------------------------------------------------------------

def test_uncalibrated_bucket_still_admits_under_occupancy():
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", _TOKENS, 0.0, "low", busy_slots=3
    )
    assert decision.admit is True
    assert decision.reason == "insufficient_samples"


def test_high_priority_admits_and_never_queues():
    _seed_calibration()
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", _TOKENS, 0.0, "high", busy_slots=3
    )
    assert decision.admit is True
    assert decision.reason == "no_budget"


def test_observe_policy_logs_refusal_but_admits_under_occupancy():
    _seed_calibration()
    decision = prefill_eta.evaluate_admission(
        _cfg(local_admission_guard_policy="observe"),
        "Qwen3", _TOKENS, 0.0, "low", busy_slots=1,
    )
    assert decision.admit is True
    assert decision.reason == "eta_exceeds_budget"
    assert decision.policy == "observe"


def test_policy_off_never_costs_the_prompt():
    _seed_calibration()
    decision = prefill_eta.evaluate_admission(
        _cfg(local_admission_guard_policy="off"),
        "Qwen3", _TOKENS, 0.0, "low", busy_slots=1,
    )
    assert decision.admit is True
    assert decision.reason == "policy_off"
    assert decision.projected_wait_seconds is None


# ---------------------------------------------------------------------------
# AC4: unobservable prefill progress occupies the gate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_in_flight_prefill_with_unobservable_progress_is_counted(caplog):
    """AC4: ``progress=None alive=False in_flight=True`` must not read as
    "free soon" — the in-flight prefill occupies the gate for at least the
    calibrated estimate."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_calibration()

    with (
        # No generating occupancy at all: the only occupant is the prefill.
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(0, 1)),
        patch("proxy.router_helpers._get_live_prefill_in_flight_count", lambda _srv: 1),
        patch("proxy.provider.request_priority", return_value="low"),
        _guard_patches(),
    ):
        action, reason, elapsed = await provider._maybe_queue_for_local_slot(
            _cfg(),
            0,
            1,
            _DummyRequest(),
            {},
            {},
            {"name": "local-llama", "llama_model": "Qwen3"},
            "session-1",
            priority="low",
        )

    assert action == "admission_reject"
    assert reason == "eta_exceeds_budget"
    assert elapsed is None
    assert contention_queue.queue_depth() == 0

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "busy_slots=1" in messages
    assert "in_flight_prefills=1" in messages
    assert "eta_occupancy_s=" in messages
    assert "projected_wait_s=" in messages


@pytest.mark.asyncio
async def test_free_gate_with_no_in_flight_prefill_still_admits(caplog):
    """AC4 counter-case: an idle gate must not be charged occupancy."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_calibration()

    with (
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(0, 1)),
        patch("proxy.router_helpers._get_live_prefill_in_flight_count", lambda _srv: 0),
        patch("proxy.provider.request_priority", return_value="low"),
        _guard_patches(),
    ):
        decision = await provider._prefill_admission_guard(
            _cfg(),
            {"name": "local-llama", "llama_model": "Qwen3"},
            _TOKENS,
            "session-1",
            priority="low",
            max_local=1,
        )

    assert decision is not None
    assert decision.admit is True
    assert decision.busy_slots == 0
    assert decision.in_flight_prefills == 0


# ---------------------------------------------------------------------------
# AC6: logging carries the occupancy evidence
# ---------------------------------------------------------------------------

def test_decision_log_carries_occupancy_evidence(caplog):
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_calibration()
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", _TOKENS, 0.0, "low",
        busy_slots=1, in_flight_prefills=1,
    )
    prefill_eta.record_guard_decision(
        decision, provider="local-llama", session_id="s-1", model="Qwen3",
        priority="low",
    )

    line = next(
        r.getMessage() for r in caplog.records if r.getMessage().startswith(
            "local_admission_guard"
        )
    )
    for field in (
        "decision=reject",
        "reason=eta_exceeds_budget",
        "eta_prefill_s=",
        "eta_occupancy_s=",
        "projected_wait_s=",
        "budget_s=",
        "busy_slots=1",
        "in_flight_prefills=1",
        "samples=",
        "provider=local-llama",
        "session=s-1",
    ):
        assert field in line, f"missing {field} in: {line}"
