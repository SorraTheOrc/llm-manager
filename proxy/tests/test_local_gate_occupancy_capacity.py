"""Prefill-gate occupancy and the capacity-admission/cap interaction
(LP-0MVXWQQ20002HAX4).

The residual local-share gap after the cooldown fix decomposes into two
*eligibility* defects, both measured on live traffic (2026-11-01
13:12:48-14:00:03, 58 sessions, 1 788 requests, local share 2.6 %):

1. **Double-counted co-residency.** ``local_max_concurrent_prefills: 1``
   serialises prefill-vs-prefill only — decode keeps running in parallel
   (documented in ``proxy/config.yaml`` and ``proxy/docs/routing.md``). The
   admission guard nevertheless costed its wait/deadline bounds against
   ``max(generating concurrency, prefills in flight)``, so a *decoder-only*
   neighbour whose throughput cost is already inside the contended rate
   bucket was charged a second time as an occupancy multiplier. Measured: 40
   refusals in one hour with ``in_flight_prefills=0 busy_slots=1``. The
   correction ships as part of the opt-in free-gate rung (below); the
   ``gate_occupancy`` field reports the distinction on every decision.

2. **Two gates that contradicted each other.** Capacity-based admission
   (:mod:`proxy.capacity_admission`) admits the economic band against
   MEASURED idle slot-time, but the dispatch guard's flat cold-prefill token
   cap refused every request it admitted, so enabling the capacity gate was a
   no-op. The gate now also enforces the guard's own prefill deadline bound,
   and a request it admits is stamped on the request so the flat cap yields.

3. **The queue-wait budget applied to a request that never queues.**
   ``contention_queue_max_wait_seconds`` is calibrated for *queue* losses,
   but the direct (free-slot) path compared a request that starts its
   prefill immediately against it. Measured: 20 of the 76 gate-free refusals
   in the same window had a measured own ETA of 36-102 s — inside the 114 s
   deadline-relative safety budget — and were refused only against the 30 s
   queue budget. A free gate is now bounded by the prefill deadline. Every
   contended admission keeps the queue bound.

Every test asserts behaviour through the public API: a decision object, a
dispatch, or a log line a consumer parses.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, patch

import proxy.provider as provider
import pytest
from fastapi import Response

from proxy import capacity_admission, contention_queue, prefill_eta


class _DummyRequest:
    """Minimal request stub (matches the other admission tests)."""

    def __init__(self, body: bytes = b'{"model":"test"}', headers=None):
        self._body = body
        self.headers = headers or {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


def _ok_response() -> Response:
    return Response(
        content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
        status_code=200,
        media_type="application/json",
    )


MODEL = "Qwen3"
#: Tokens whose calibrated own-ETA (~56 s at 3.9 ms/token) fits a 60 s wait
#: budget but whose doubled projection (~112 s) does not.
TOKENS = 12_000
MS_PER_TOKEN = 3.9
BUDGET = 60.0


def _cfg(**server_overrides) -> dict:
    server = {
        "session_slot_pool_size": 1,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": BUDGET,
        "contention_queue_max_depth": 4,
        # Well clear of the ~112 s doubled projection so the deadline bound is
        # never the reason under test.
        "local_prefill_deadline_seconds": 600,
    }
    server.update(server_overrides)
    return {"provider_cooldown_seconds": 60, "server": server}


@pytest.fixture(autouse=True)
def _clean_state():
    contention_queue.reset()
    prefill_eta.reset()
    capacity_admission.reset_capacity_state()
    yield
    contention_queue.reset()
    prefill_eta.reset()
    capacity_admission.reset_capacity_state()


def _seed_calibration(tokens: int = TOKENS, count: int = 6) -> None:
    for _ in range(count):
        prefill_eta.record_dispatch_sample(
            MODEL, tokens, 0.0, tokens * MS_PER_TOKEN
        )


class TestPrefillGateObservability:
    """``gate_occupancy`` names the part of ``busy_slots`` that is a prefill.

    The wait term keeps its historical ``busy_slots`` meaning unless the
    opt-in free-gate rung is enabled, so this field is pure observability for
    every default deployment — it is what makes "the pool is generating"
    distinguishable from "the prefill gate is busy" in the log, which is how
    the 2026-11-01 decomposition was measured.
    """

    def test_decoder_only_neighbour_is_reported_as_a_free_gate(self):
        _seed_calibration()

        decision = prefill_eta.evaluate_admission(
            _cfg(), MODEL, TOKENS, 0.0, "low", busy_slots=1, in_flight_prefills=0
        )

        assert decision.busy_slots == 1, "the decoder is still reported"
        assert decision.gate_occupancy == 0, "but it does not hold the gate"
        # Default deployment: the wait term is unchanged (``busy_slots``).
        assert decision.projected_wait_seconds > BUDGET

    def test_gate_occupancy_is_capped_by_the_prefill_count(self):
        """Two generating slots and one prefill ⇒ one gate occupant."""
        _seed_calibration()

        decision = prefill_eta.evaluate_admission(
            _cfg(), MODEL, TOKENS, 0.0, "low", busy_slots=2, in_flight_prefills=1
        )

        assert decision.gate_occupancy == 1
        assert decision.busy_slots == 2

    def test_unobserved_gate_reports_busy_slots(self):
        """A caller that does not report the gate reports ``busy_slots``."""
        _seed_calibration()

        decision = prefill_eta.evaluate_admission(
            _cfg(), MODEL, TOKENS, 0.0, "low", busy_slots=1
        )

        assert decision.admit is False
        assert decision.gate_occupancy == 1

    def test_decision_log_carries_the_gate_occupancy(self, caplog):
        """Consumers parse ``gate_occupancy=`` off the admission line."""
        caplog.set_level(logging.INFO, logger="llama-proxy")
        _seed_calibration()

        prefill_eta.record_guard_decision(
            prefill_eta.evaluate_admission(
                _cfg(), MODEL, TOKENS, 0.0, "low", busy_slots=1, in_flight_prefills=0
            ),
            provider="local-qwen3",
            model=MODEL,
            priority="low",
        )

        line = next(
            r.getMessage()
            for r in caplog.records
            if "local_admission_guard decision=" in r.getMessage()
        )
        assert "busy_slots=1" in line
        assert "in_flight_prefills=0" in line
        assert "gate_occupancy=0" in line


class TestFreeGateBound:
    """A free prefill gate may be bounded by the deadline (opt-in).

    ``contention_queue_max_wait_seconds`` is calibrated for *queue* losses.
    A request that starts its prefill immediately never queues, so its only
    wait is its own prefill — bounded by the prefill deadline and its safety
    budget. Measured 2026-11-01 13:12-14:00: 20 of 76 gate-free refusals had
    a measured own ETA of 36-102 s (inside the 114 s safety budget) and were
    refused only against the 30 s queue budget.

    The rung is OPT-IN (`local_admission_guard_free_gate_deadline_budget`),
    because the deadline bound still admits the occasional prompt the
    measured deadline then cancels (6 of those 20 in the thrash replay
    window). OFF is the shipped state and the rollback.
    """

    def _gate_free_cfg(self, **server_overrides) -> dict:
        server = {
            "session_slot_pool_size": 1,
            "contention_queue_policy": "queue",
            "contention_queue_max_wait_seconds": 30,
            "contention_queue_max_depth": 4,
            "local_prefill_deadline_seconds": 120,
            "local_admission_guard_budget_safety_fraction": 0.95,
            "local_admission_guard_free_gate_deadline_budget": True,
        }
        server.update(server_overrides)
        return {"provider_cooldown_seconds": 60, "server": server}

    def test_free_gate_admits_a_prefill_that_fits_the_deadline(self):
        _seed_calibration()  # own ETA 56.2 s: over the 30 s queue budget

        decision = prefill_eta.evaluate_admission(
            self._gate_free_cfg(), MODEL, TOKENS, 0.0, "low",
            busy_slots=1, in_flight_prefills=0,
        )

        assert decision.admit is True
        assert decision.reason == "within_budget"
        assert decision.budget_seconds == pytest.approx(30.0)
        assert decision.bound_seconds == pytest.approx(114.0)
        assert decision.gate_occupancy == 0
        # Under the rung the decoder contributes no occupancy at all.
        assert decision.eta_occupancy_seconds == pytest.approx(0.0)
        assert decision.projected_wait_seconds == pytest.approx(
            decision.eta_prefill_seconds
        )

    def test_rung_is_off_by_default(self):
        """The shipped default keeps the queue-wait budget in charge."""
        _seed_calibration()

        decision = prefill_eta.evaluate_admission(
            self._gate_free_cfg(
                local_admission_guard_free_gate_deadline_budget=False
            ),
            MODEL, TOKENS, 0.0, "low",
            busy_slots=1, in_flight_prefills=0,
        )

        assert decision.admit is False
        assert decision.reason == "eta_exceeds_budget"
        assert decision.bound_seconds == pytest.approx(30.0)

    def test_contended_admission_still_honours_the_queue_budget(self):
        """With a prefill in flight the queue bound is unchanged."""
        _seed_calibration()

        decision = prefill_eta.evaluate_admission(
            self._gate_free_cfg(), MODEL, TOKENS, 0.0, "low",
            busy_slots=1, in_flight_prefills=1,
        )

        assert decision.admit is False
        assert decision.reason == "eta_exceeds_budget"
        assert decision.bound_seconds == pytest.approx(30.0)

    def test_safety_budget_still_refuses_on_a_free_gate(self):
        """The deadline-relative bound is not widened by this rung."""
        # 30,000 tokens at the calibrated 3.9 ms/token ≈ 140 s > 114 s.
        _seed_calibration(tokens=30_000)

        decision = prefill_eta.evaluate_admission(
            self._gate_free_cfg(), MODEL, 30_000, 0.0, "low",
            busy_slots=1, in_flight_prefills=0,
        )

        assert decision.admit is False
        assert decision.reason == "prefill_eta_exceeds_safety_budget"

    def test_legacy_caller_keeps_the_queue_budget(self):
        """Without an observed gate, ``busy_slots`` keeps its old meaning."""
        _seed_calibration()

        decision = prefill_eta.evaluate_admission(
            self._gate_free_cfg(), MODEL, TOKENS, 0.0, "low", busy_slots=1
        )

        assert decision.admit is False
        assert decision.reason == "eta_exceeds_budget"
        assert decision.bound_seconds == pytest.approx(30.0)
    def test_disabled_safety_budget_leaves_the_queue_floor(self):
        """``budget_safety_fraction: 0`` (documented rollback) ⇒ no widening."""
        _seed_calibration()

        decision = prefill_eta.evaluate_admission(
            self._gate_free_cfg(local_admission_guard_budget_safety_fraction=0.0),
            MODEL, TOKENS, 0.0, "low",
            busy_slots=1, in_flight_prefills=0,
        )

        assert decision.admit is False
        assert decision.bound_seconds == pytest.approx(30.0)

    def test_decision_log_carries_the_bound(self, caplog):
        caplog.set_level(logging.INFO, logger="llama-proxy")
        _seed_calibration()

        prefill_eta.record_guard_decision(
            prefill_eta.evaluate_admission(
                self._gate_free_cfg(), MODEL, TOKENS, 0.0, "low",
                busy_slots=1, in_flight_prefills=0,
            ),
            provider="local-qwen3",
            model=MODEL,
            priority="low",
            reason_label="eta_exceeds_budget_direct",
            free_slot=True,
        )

        line = next(
            r.getMessage()
            for r in caplog.records
            if "local_admission_guard decision=" in r.getMessage()
        )
        assert "budget_s=30.0" in line
        assert "bound_s=114.0" in line


class TestFlatCapYieldsToCapacityAdmission:
    """A capacity-admitted band request is not re-refused by the flat cap."""

    CAP = 25_000
    TOKENS_OVER_CAP = 26_000

    def test_flat_cap_still_refuses_without_a_capacity_admission(self):
        decision = prefill_eta.evaluate_admission(
            _cfg(local_cold_prefill_budget_tokens=self.CAP),
            MODEL,
            self.TOKENS_OVER_CAP,
            0.0,
            "low",
        )

        assert decision.admit is False
        assert decision.reason == "prefill_budget_exceeded"
        assert decision.new_tokens == self.TOKENS_OVER_CAP
        assert decision.budget_tokens == self.CAP

    def test_capacity_admission_is_not_overridden_by_the_flat_cap(self):
        decision = prefill_eta.evaluate_admission(
            _cfg(
                local_cold_prefill_budget_tokens=self.CAP,
                contention_queue_max_wait_seconds=300,
            ),
            MODEL,
            self.TOKENS_OVER_CAP,
            0.0,
            "low",
            capacity_admitted=True,
        )

        assert decision.admit is True
        assert decision.reason != "prefill_budget_exceeded"
        # The evidence of the cap it out-priced stays on the decision.
        assert decision.budget_tokens == self.CAP
        assert decision.new_tokens == self.TOKENS_OVER_CAP

    def test_measured_bounds_still_refuse_a_capacity_admission(self):
        """Deferring the flat cap never bypasses the measured deadline bound."""
        _seed_calibration(tokens=self.TOKENS_OVER_CAP)

        decision = prefill_eta.evaluate_admission(
            _cfg(
                local_cold_prefill_budget_tokens=self.CAP,
                local_prefill_deadline_seconds=90,
            ),
            MODEL,
            self.TOKENS_OVER_CAP,
            0.0,
            "low",
            capacity_admitted=True,
        )

        assert decision.admit is False
        assert decision.reason == "prefill_eta_exceeds_deadline"


class TestCapacityAdmissionWiring:
    """End to end: routing admission → dispatch guard → local dispatch."""

    CAP = 25_000
    TOKENS_OVER_CAP = 26_000

    def _guard_patches(self, capacity_admit: dict | None):
        return (
            patch(
                "proxy.provider._estimate_effective_prompt_tokens_for_routing",
                new=AsyncMock(return_value=self.TOKENS_OVER_CAP),
            ),
            patch("proxy.provider._get_tokenizer_for_model", lambda *_: (None, 1.0)),
            patch(
                "proxy.provider._should_skip_local",
                lambda *a, **k: (True, "large_context_bypass"),
            ),
            patch("proxy.provider._get_cached_ratio", lambda *_: 0.0),
            patch(
                "proxy.capacity_admission.evaluate_capacity_admission",
                lambda *a, **k: capacity_admit,
            ),
        )

    async def _drive(self, calls, capacity_admit):
        config = {
            "providers": [
                {"name": "local-llama", "type": "local", "llama_model": MODEL},
                {
                    "name": "remote-fallback",
                    "type": "remote",
                    "endpoint": "https://api.example.com/v1",
                },
            ],
            "aliases": ["hybrid*"],
        }

        async def _local(_req, _path):
            calls.append("local")
            return _ok_response()

        async def _remote(_req, _path, _pc):
            calls.append("remote")
            return _ok_response()

        _cfg_patch, _tok, _skip, _ratio, _cap = self._guard_patches(capacity_admit)
        with (
            patch("proxy.router.proxy_to_local", _local),
            patch("proxy.server.proxy_to_remote", _remote),
            patch("proxy.provider.request_priority", return_value="low"),
            _cfg_patch,
            _tok,
            _skip,
            _ratio,
            _cap,
        ):
            return await provider.proxy_with_fallback(
                _DummyRequest(), "v1/chat/completions", config,
                _cfg(
                    local_cold_prefill_budget_tokens=self.CAP,
                    contention_queue_max_wait_seconds=300,
                ),
            )

    @pytest.mark.asyncio
    async def test_capacity_admit_reaches_local_dispatch(self, caplog):
        caplog.set_level(logging.INFO, logger="llama-proxy")
        calls: list[str] = []
        admit = {
            "reason": "capacity_admission",
            "projected_seconds": 130.0,
            "prefill_seconds": 101.0,
            "deadline_seconds": 600.0,
            "deadline_budget_seconds": 0.0,
            "admitted_seconds": 130.0,
            "budget_seconds": 4000.0,
            "idle_seconds": 5000.0,
            "budget_ratio": 0.8,
            "horizon_seconds": 3600.0,
        }

        await self._drive(calls, admit)

        messages = " ".join(r.getMessage() for r in caplog.records)
        assert calls == ["local"], (
            "the guard must not re-refuse a request the capacity gate admitted"
        )
        assert "routing_capacity_admit" in messages
        assert "prefill_budget_exceeded" not in messages

    @pytest.mark.asyncio
    async def test_without_a_capacity_admit_the_flat_cap_still_refuses(
        self, caplog
    ):
        caplog.set_level(logging.INFO, logger="llama-proxy")
        calls: list[str] = []

        await self._drive(calls, None)

        messages = " ".join(r.getMessage() for r in caplog.records)
        assert calls == ["remote"], "the rollback path must keep today's bypass"
        assert "routing_capacity_admit" not in messages
        assert "reason=large_context_bypass" in messages
