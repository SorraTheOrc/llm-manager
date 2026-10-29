"""Prefill-ETA admission guard wiring (LP-0MVGDUV5E005UWBQ).

The guard sits at the contention-queue decision point (``provider.py``): when
the projected prefill cannot finish inside the caller's wait budget the request
is refused *immediately* and falls through to the next provider instead of
parking in the queue for up to the full budget.

Covered acceptance criteria:

- AC3: ETA above budget ⇒ immediate refusal (no queueing) + retryable status
  with ``Retry-After`` when the chain is exhausted; ETA within budget ⇒ the
  queue behaves exactly as today; ``high`` priority still never queues;
  missing calibration ⇒ today's behaviour unchanged.
- AC5: a 90 %-cached 100k prompt is still admitted (and queued) because its
  ``new_tokens`` term is small.

Also pins the per-reason retry contract for ``reason=prefill_budget_exceeded``
(LP-0MVQ239VQ001RXZ0, AC1 of LP-0MVNTT4X8009IO2X): a prompt that is
permanently over the flat cold-prefill cap must NOT advertise ``Retry-After``.

The ETA/occupancy reasons take the opposite treatment (LP-0MVQ23A6S007DVDM,
AC2/AC4 of LP-0MVNTT4X8009IO2X): waiting *does* help, so each refusal must
advertise ``ceil(projected_wait_seconds - budget_seconds)`` (min 1 s, capped)
as ``Retry-After`` together with an absolute ``retry_at`` and the admission
token evidence.

The final section (LP-0MVQ23AGI006JNH4, AC6/AC7 of LP-0MVNTT4X8009IO2X) pins
the same contract as a single table across **every** refusal reason — status,
``Retry-After``/``retry_at`` presence-or-absence, reason naming and the token
evidence — plus a dedicated cooldown case whose timer must come from the real
cooldown remainder rather than the ETA formula.

Both builders resolve their retry story through the single reason→timing
decision ``provider._refusal_retry_timing()`` (LP-0MVQ23API001UOET), so the two
refusal paths cannot drift apart.

The capacity-verdict slice (LP-0MVSMRSU5001JTMT, parent AC2/AC3/AC4) extends
that contract to the *kind* of refusal: the 429 body and the single
``local_only_refused`` line must both name the verdict from
``admission_reserve.local_capacity_verdict`` — ``deferrable`` when waiting can
help, ``too_large`` when the request exceeds the local serving envelope — while
retry timing and every pre-existing field stay untouched.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import proxy.provider as provider
import pytest
from fastapi import Response

from proxy import contention_queue, prefill_eta


class _DummyRequest:
    """Minimal request stub (matches the contention-queue test helpers)."""

    def __init__(self, body: bytes = b'{"model":"test"}', headers=None):
        self._body = body
        self.headers = headers or {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


class _MutableConcurrency:
    """Mutable ``_get_local_concurrency_info`` stand-in."""

    def __init__(self, active: int = 1, max_: int = 1):
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


def _queue_cfg(**server_overrides) -> dict:
    server = {
        "session_slot_pool_size": 1,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": 60,
        "contention_queue_max_depth": 4,
        # Isolate the BUDGET bound in this file: the deadline bound
        # (LP-0MVKUKEU80022C3U) is covered by tests/test_prefill_eta_deadline.py.
        "local_prefill_deadline_seconds": 0,
    }
    server.update(server_overrides)
    return {"provider_cooldown_seconds": 60, "server": server}


@pytest.fixture(autouse=True)
def _clean_state():
    contention_queue.reset()
    prefill_eta.reset()
    yield
    contention_queue.reset()
    prefill_eta.reset()


@pytest.fixture
def mixed_model_config():
    return {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {
                "name": "remote-fallback",
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
            },
        ],
        "aliases": ["hybrid*"],
    }


def _seed_calibration(tokens: int = 100_000, ms_per_token: float = 3.9) -> None:
    for _ in range(6):
        prefill_eta.record_dispatch_sample(
            "Qwen3", tokens, 0.0, tokens * ms_per_token
        )


@contextmanager
def _guard_patches(
    tokens: int, cached_ratio: float = 0.0, skip_reason: str | None = None
):
    """Patch the estimate pipeline so the guard costs a known prompt size.

    *skip_reason* forces the local-eligibility gate to skip the local entry
    with that reason (used to reach the non-admission skip paths), and leaves
    today's ``(False, None)`` admission untouched when omitted.
    """
    with (
        patch(
            "proxy.provider._estimate_effective_prompt_tokens_for_routing",
            new=AsyncMock(return_value=tokens),
        ),
        patch("proxy.provider._get_tokenizer_for_model", lambda *_: (None, 1.0)),
        patch(
            "proxy.provider._should_skip_local",
            lambda *a, **k: (skip_reason is not None, skip_reason),
        ),
        patch("proxy.provider._get_cached_ratio", lambda *_: cached_ratio),
    ):
        yield


# ---------------------------------------------------------------------------
# AC3: ETA above the budget ⇒ immediate refusal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_eta_above_budget_falls_back_immediately_without_queueing(
    mixed_model_config, caplog
):
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_calibration(tokens=100_000, ms_per_token=3.9)
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(1, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _guard_patches(tokens=100_000),
    ):
        started = time.monotonic()
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", mixed_model_config, _queue_cfg()
        )
        elapsed = time.monotonic() - started

    assert result.status_code == 200
    assert calls == ["remote"], "a refused request must never dispatch or wait local"
    assert elapsed < 2.0, "the refusal must be immediate (no queue wait)"
    assert contention_queue.queue_depth() == 0

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "local_admission_guard decision=reject" in messages
    assert "eta_prefill_s=" in messages
    assert "budget_s=" in messages
    assert "contention_queue_wait" not in messages, "must not enter the queue"
    assert "contention_queue_fallback_after_queue" not in messages


@pytest.mark.asyncio
async def test_refused_request_records_eta_diagnostics(mixed_model_config):
    """The refusal is visible in the exhaustion diagnostics (ETA + budget)."""
    _seed_calibration(tokens=100_000, ms_per_token=3.9)

    with (
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(1, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _guard_patches(tokens=100_000),
    ):
        action, reason, elapsed = await provider._maybe_queue_for_local_slot(
            _queue_cfg(),
            1,
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


# ---------------------------------------------------------------------------
# AC3: ETA within the budget ⇒ today's queue behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_eta_within_budget_still_queues_and_dispatches_local(
    mixed_model_config, caplog
):
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_calibration(tokens=2_000, ms_per_token=3.9)
    concurrency = _MutableConcurrency(1, 1)
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    import asyncio

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
        patch("proxy.provider.request_priority", return_value="low"),
        _guard_patches(tokens=2_000),
    ):
        task = asyncio.create_task(
            provider.proxy_with_fallback(
                _DummyRequest(), "v1/chat/completions", mixed_model_config,
                _queue_cfg(),
            )
        )
        for _ in range(200):
            if contention_queue.queue_depth() > 0:
                break
            await asyncio.sleep(0.005)
        assert contention_queue.queue_depth() == 1, "admissible request should queue"

        concurrency.active = 0
        await contention_queue.wake_all()
        result = await asyncio.wait_for(task, timeout=5)

    assert result.status_code == 200
    assert calls == ["local"]
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "local_admission_guard decision=admit" in messages
    assert "contention_queue_dispatch" in messages


@pytest.mark.asyncio
async def test_warm_cache_prompt_is_admitted_and_queued(mixed_model_config, caplog):
    """AC5 (LP-0MVGDUV5E005UWBQ) under occupancy: a cache-warm prompt stays
    queueable even when the prefill gate is fully occupied (own ETA ~17 s,
    projected wait ~34 s — inside the 60 s budget). With the 120 s budget
    the original fixture (40k tokens) was also admissible; LP-0MVIQ0EXV00673HO
    halved the budget so the prompt was reduced to stay under the new cap."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_calibration(tokens=40_000, ms_per_token=3.9)
    concurrency = _MutableConcurrency(1, 1)
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    import asyncio

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
        patch("proxy.provider.request_priority", return_value="low"),
        _guard_patches(tokens=20_000, cached_ratio=0.9),
    ):
        task = asyncio.create_task(
            provider.proxy_with_fallback(
                _DummyRequest(), "v1/chat/completions", mixed_model_config,
                _queue_cfg(),
            )
        )
        for _ in range(200):
            if contention_queue.queue_depth() > 0:
                break
            await asyncio.sleep(0.005)
        assert contention_queue.queue_depth() == 1, (
            "a cache-warm prompt must stay admissible under occupancy"
        )

        concurrency.active = 0
        await contention_queue.wake_all()
        result = await asyncio.wait_for(task, timeout=5)

    assert result.status_code == 200
    assert calls == ["local"]
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "local_admission_guard decision=admit" in messages


@pytest.mark.asyncio
async def test_missing_calibration_keeps_queueing_unchanged(
    mixed_model_config, caplog
):
    """AC3: with no calibration the guard is inert and today's queue applies."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    concurrency = _MutableConcurrency(1, 1)
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    import asyncio

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
        patch("proxy.provider.request_priority", return_value="low"),
        _guard_patches(tokens=100_000),
    ):
        task = asyncio.create_task(
            provider.proxy_with_fallback(
                _DummyRequest(), "v1/chat/completions", mixed_model_config,
                _queue_cfg(),
            )
        )
        for _ in range(200):
            if contention_queue.queue_depth() > 0:
                break
            await asyncio.sleep(0.005)
        assert contention_queue.queue_depth() == 1, "uncalibrated guard must not refuse"

        concurrency.active = 0
        await contention_queue.wake_all()
        result = await asyncio.wait_for(task, timeout=5)

    assert result.status_code == 200
    assert calls == ["local"]
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "local_admission_guard" not in messages


@pytest.mark.asyncio
async def test_observe_policy_never_refuses(mixed_model_config, caplog):
    """``observe`` logs the refusal verdict but keeps queueing."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_calibration(tokens=100_000, ms_per_token=3.9)
    concurrency = _MutableConcurrency(1, 1)
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    import asyncio

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
        patch("proxy.provider.request_priority", return_value="low"),
        _guard_patches(tokens=100_000),
    ):
        task = asyncio.create_task(
            provider.proxy_with_fallback(
                _DummyRequest(), "v1/chat/completions", mixed_model_config,
                _queue_cfg(local_admission_guard_policy="observe"),
            )
        )
        for _ in range(200):
            if contention_queue.queue_depth() > 0:
                break
            await asyncio.sleep(0.005)
        queued = contention_queue.queue_depth()

        concurrency.active = 0
        await contention_queue.wake_all()
        result = await asyncio.wait_for(task, timeout=5)

    assert queued == 1
    assert result.status_code == 200
    assert calls == ["local"], "observe must keep today's queue-then-dispatch path"
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "local_admission_guard decision=reject" in messages
    assert "enforced=false" in messages
    assert "reason=eta_exceeds_budget" in messages


# ---------------------------------------------------------------------------
# AC3 regression guard: high priority never queues
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_high_priority_never_queues_or_waits(mixed_model_config):
    _seed_calibration(tokens=100_000, ms_per_token=3.9)
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(1, 1)),
        patch("proxy.provider.request_priority", return_value="high"),
        _guard_patches(tokens=100_000),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", mixed_model_config, _queue_cfg()
        )

    assert result.status_code == 200
    assert calls == ["remote"]
    assert contention_queue.queue_depth() == 0


# ---------------------------------------------------------------------------
# AC3: retryable status when the refused request finds no remote provider
# ---------------------------------------------------------------------------

def test_exhausted_chain_after_refusal_returns_retryable_429_with_retry_after():
    diagnostics = [
        {
            "provider": "local-llama",
            "type": "local",
            "status": "local_admission_reject",
            "reason": "eta_exceeds_budget",
            "eta_prefill_seconds": 468.0,
            "budget_seconds": 60.0,  # halved from 120 (LP-0MVIQ0EXV00673HO)
        }
    ]
    response = provider._build_exhausted_response(
        all_local_slot_exhaustion=False,
        unavailable_providers={},
        diagnostics=diagnostics,
        config=_queue_cfg(),
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "408"  # 468 - 60 = 408
    assert response.headers["X-Local-Admission"] == "reject"
    assert response.headers["X-Local-Admission-Eta"] == "468.0"
    assert response.headers["X-Local-Admission-Budget"] == "60.0"

    payload = json.loads(response.body)
    assert payload["code"] == provider.EXHAUSTION_CODE_LOCAL_ADMISSION
    assert payload["eta_prefill_seconds"] == 468.0
    assert payload["budget_seconds"] == 60.0
    assert payload["retry_after"] == 408
    assert payload["diagnostics"] == diagnostics


def test_generic_exhaustion_is_unchanged_without_a_refusal():
    """Regression: a chain that never hit the guard keeps the generic 503."""
    response = provider._build_exhausted_response(
        all_local_slot_exhaustion=False,
        unavailable_providers={},
        diagnostics=[{"provider": "remote", "status": "HTTP 500"}],
        config=_queue_cfg(),
    )

    assert response.status_code == 503
    assert "X-Local-Admission" not in response.headers
    assert json.loads(response.body)["code"] == provider.EXHAUSTION_CODE_ALL


# ---------------------------------------------------------------------------
# Direct (free-slot) dispatch guard (LP-0MVIQ0F8T002FS42)
#
# The queue-entry guard above only runs when every local slot is busy. A
# request that finds a FREE slot used to dispatch with no ETA ceiling, so a
# 40-80k-token prefill that could not finish inside the budget was started
# anyway and abandoned by the caller (baseline: 48 of 606 local dispatches
# had a first byte after >120 s). These tests cover the same decision applied
# one step earlier, on the free-slot path.
# ---------------------------------------------------------------------------

def _free_slot_cfg(**server_overrides) -> dict:
    """Config for the direct-dispatch path: a slot is free (cur < max).

    The deadline bound (LP-0MVKUKEU80022C3U) is disabled here to isolate the
    budget bound — the deadline bound is covered by
    tests/test_prefill_eta_deadline.py.
    """
    return _queue_cfg(**server_overrides)


@pytest.mark.asyncio
async def test_direct_dispatch_refused_when_projected_prefill_exceeds_budget(
    mixed_model_config, caplog
):
    """AC1: a request whose projected prefill exceeds the budget is NEVER
    dispatched locally, including when a local slot is free."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    # Cache-cold 100k prompt at ~3.9 ms/token ≈ 468 s ≫ 60 s budget.
    _seed_calibration(tokens=100_000, ms_per_token=3.9)
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        # Free slot: 0 active of 1 — the direct-dispatch path, not the queue.
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(0, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _guard_patches(tokens=100_000),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", mixed_model_config,
            _free_slot_cfg(),
        )

    assert result.status_code == 200
    assert calls == ["remote"], (
        "a free slot must NOT bypass the ETA ceiling: over-budget local "
        "dispatches must fall through to remote"
    )
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "local_admission_guard decision=reject" in messages
    assert "reason=eta_exceeds_budget_direct" in messages
    assert "eta_prefill_s=" in messages
    assert "budget_s=" in messages
    assert "contention_queue_wait" not in messages, "must not enter the queue"


@pytest.mark.asyncio
async def test_direct_dispatch_admitted_when_projected_prefill_fits_budget(
    mixed_model_config,
):
    """AC2: a request whose projected prefill fits the budget is still
    dispatched locally on a free slot — no regression for the common case."""
    _seed_calibration(tokens=100_000, ms_per_token=3.9)
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(0, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        # 10k cold ≈ 47 s < 60 s budget ⇒ admit.
        _guard_patches(tokens=10_000),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", mixed_model_config,
            _free_slot_cfg(),
        )

    assert result.status_code == 200
    assert calls == ["local"], "within-budget free-slot dispatch must stay local"


@pytest.mark.asyncio
async def test_direct_dispatch_inert_without_calibration(mixed_model_config, caplog):
    """AC3: cold start — with no estimator samples the guard is inert and the
    free-slot dispatch behaves exactly as today (conservative fail-open)."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(0, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _guard_patches(tokens=100_000),  # no calibration seeded
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", mixed_model_config,
            _free_slot_cfg(),
        )

    assert result.status_code == 200
    assert calls == ["local"], "an uncalibrated guard must not refuse local work"
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "reason=eta_exceeds_budget_direct" not in messages


@pytest.mark.asyncio
async def test_direct_dispatch_uses_the_single_wait_budget(mixed_model_config, caplog):
    """AC3/AC4: the direct guard compares against the SAME budget source as
    the queue guard (``contention_queue_max_wait_seconds``) — no second knob.
    Raising the budget admits work the 60 s budget refused."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_calibration(tokens=100_000, ms_per_token=3.9)
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(0, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _guard_patches(tokens=100_000),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", mixed_model_config,
            _free_slot_cfg(contention_queue_max_wait_seconds=600),
        )

    assert result.status_code == 200
    assert calls == ["local"], (
        "the same budget knob must govern the direct path: a 600 s budget "
        "admits the 468 s projection"
    )


@pytest.mark.asyncio
async def test_direct_dispatch_logs_one_labelled_guard_line(mixed_model_config, caplog):
    """AC4: the direct refusal emits exactly ONE guard line, labelled
    ``reason=eta_exceeds_budget_direct free_slot=true`` — it must not be
    double-counted as a queue-entry refusal, and it must carry the ETA and
    budget that drove the decision."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_calibration(tokens=100_000, ms_per_token=3.9)

    async def _local(_req, _path):
        return _ok_response()

    async def _remote(_req, _path, _pc):
        return _ok_response()

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(0, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _guard_patches(tokens=100_000),
    ):
        await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", mixed_model_config,
            _free_slot_cfg(),
        )

    guard_lines = [
        r.getMessage() for r in caplog.records
        if "local_admission_guard decision=" in r.getMessage()
    ]
    assert len(guard_lines) == 1, (
        f"expected exactly one guard line, got {len(guard_lines)}: {guard_lines}"
    )
    line = guard_lines[0]
    assert "decision=reject" in line
    assert "reason=eta_exceeds_budget_direct" in line
    assert "free_slot=true" in line
    assert "eta_prefill_s=" in line
    assert "budget_s=" in line
    # The fallback target is recorded for the operator (AC4): the refusal
    # names the reason and the chain records which provider serves instead.
    assert any(
        "local_admission_guard_direct_fallback" in r.getMessage()
        and "free_slot=true" in r.getMessage()
        for r in caplog.records
    ), "the direct refusal must name its reason at the dispatch point"
    assert any(
        "Fallback triggered" in r.getMessage()
        and "to=remote-fallback" in r.getMessage()
        and "reason=local_admission_guard" in r.getMessage()
        # LP-0MVMBMSWR0036RES: the line now carries session/attempt evidence
        # after ``reason=`` so the fallback is attributable to a session.
        and "session=" in r.getMessage()
        and "attempt=" in r.getMessage()
        for r in caplog.records
    ), "the chosen fallback provider must be logged"


# ---------------------------------------------------------------------------
# AC1 (LP-0MVNTT4X8009IO2X): ``prefill_budget_exceeded`` is NOT time-retryable
#
# Regression pin for the 2026-10-25 11:57:09 UTC refusal: a prompt of 31 246
# new (uncached) tokens against the flat 25 000-token cold-prefill cap was
# answered with ``Retry-After: 1`` / ``"retry_after": 1``. Waiting cannot help
# — the prompt is permanently over a flat cap — so the client backed off seven
# times (up to 2 m 8 s) and then reported the 429 as non-retryable. Both
# refusal builders are covered because either can answer a refused client:
# ``_build_local_only_refusal_response()`` (local-only intent, remote still
# available) and ``_build_local_admission_reject_response()`` (the chain is
# exhausted as well).
#
# TDD red phase removed by LP-0MVQ23API001UOET (reason-aware retry timing) and
# LP-0MVQ23AYQ000CGTW (token evidence): both builders now resolve their retry
# story through ``provider._refusal_retry_timing()``, so these tests fail
# loudly instead of silently xfailing.
# ---------------------------------------------------------------------------

#: The 2026-10-25 incident verdict, verbatim.
INCIDENT_NEW_TOKENS = 31_246
INCIDENT_BUDGET_TOKENS = 25_000


def _cold_budget_cfg(**server_overrides) -> dict:
    """Shipped post-incident admission config: flat 25k cold-prefill cap."""
    return _queue_cfg(
        local_cold_prefill_budget_tokens=INCIDENT_BUDGET_TOKENS,
        **server_overrides,
    )


async def _issue_local_only_prefill_budget_refusal(mixed_model_config) -> tuple:
    """Issue the 2026-10-25 shape: a local-only client over the cold cap.

    A remote provider is available, so falling through to it would silently
    break the client's ``local-only`` intent. Returns ``(response, calls)``.
    """
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(1, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _guard_patches(tokens=INCIDENT_NEW_TOKENS),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(headers={"x-local-preference": "local-only"}),
            "v1/chat/completions",
            mixed_model_config,
            _cold_budget_cfg(),
        )

    return result, calls


def _assert_prefill_budget_refusal_contract(result) -> dict:
    """Assert the AC1 contract shared by both refusal paths; return the body.

    A refusal whose cause is a permanent over-budget prompt must be explicit
    about the fact that time does not heal it: no ``Retry-After``, an explicit
    ``retryable: false`` / ``retry_after: null``, the token evidence the client
    needs to size a compaction, and a remedy that names compaction.
    """
    assert result.status_code == 429
    assert result.headers.get("Retry-After") is None, (
        "an over-budget prompt is not time-retryable: advertising a delay "
        "invites the retry storm observed on 2026-10-25"
    )

    body = json.loads(result.body)
    assert body.get("retryable") is False
    assert body.get("retry_after") is None
    # AC4 / LP-0MVQ23AYQ000CGTW AC1: even a NOT-time-retryable refusal must
    # carry the token evidence on headers AND body, so a client can decide to
    # compact rather than hammer the proxy (LP-0MVNTT4X8009IO2X AC4 on both
    # refusal builders).
    assert (
        int(result.headers["X-Local-Admission-New-Tokens"]) == INCIDENT_NEW_TOKENS
    ), "the token headers must be present on the over-budget refusal too"
    assert (
        int(result.headers["X-Local-Admission-Budget-Tokens"]) == INCIDENT_BUDGET_TOKENS
    )
    assert body.get("new_tokens") == INCIDENT_NEW_TOKENS, (
        "the refusal must carry the token evidence the client compacts against"
    )
    assert body.get("budget_tokens") == INCIDENT_BUDGET_TOKENS
    assert "compact" in str(body.get("suggested_action") or "").lower(), (
        "the suggested action must name compaction as the remedy"
    )
    return body


@pytest.mark.asyncio
async def test_prefill_budget_exceeded_pin_local_is_promoted_not_refused(
    mixed_model_config,
):
    """The 2026-10-25 shape: a too-large pin-local request is promoted.

    The incident prompt outgrows the local serving envelope, so waiting cannot
    help: the pin is declined and the remote tiers serve the request — no 429,
    no ``Retry-After``, no local-capacity refusal body (LP-0MVSLZY470051ZB2
    AC4; parent AC5). The no-timer contract for this reason still holds on the
    exhaustion path, asserted by
    ``test_prefill_budget_exceeded_exhaustion_refusal_is_not_time_retryable``.
    """
    result, calls = await _issue_local_only_prefill_budget_refusal(
        mixed_model_config
    )

    assert calls == ["remote"], (
        f"a too-large pin-local request must be served by the remote tier; "
        f"got dispatches={calls}"
    )
    assert result.status_code == 200, f"got {result.status_code}"
    assert result.headers.get("X-Local-Route") == "remote"
    assert result.headers.get("Retry-After") is None, (
        "waiting cannot help; the promoted response must not advertise a delay"
    )
    body = json.loads(result.body)
    assert body.get("code") != provider.EXHAUSTION_CODE_LOCAL_ADMISSION, (
        "promotion must not produce a local-capacity refusal"
    )


@pytest.mark.asyncio
async def test_prefill_budget_exceeded_exhaustion_refusal_is_not_time_retryable():
    """The chain is exhausted too: same contract on the exhaustion refusal."""
    local_only_chain = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"}
        ],
        "aliases": ["hybrid*"],
    }

    async def _local(_req, _path):
        return _ok_response()

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(1, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _guard_patches(tokens=INCIDENT_NEW_TOKENS),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", local_only_chain,
            _cold_budget_cfg(),
        )

    body = _assert_prefill_budget_refusal_contract(result)
    assert body["code"] == provider.EXHAUSTION_CODE_LOCAL_ADMISSION
    assert body["error"] == "All providers exhausted"


# ---------------------------------------------------------------------------
# LP-0MVQ23AYQ000CGTW: admission token evidence is threaded, never recomputed
#
# AC1 of this item: ``X-Local-Admission-New-Tokens`` /
# ``X-Local-Admission-Budget-Tokens`` must appear on every refusal from both
# builders, and their body equivalents too. AC3: those numbers must be the
# guard decision's own (threaded verbatim from ``_local_admission_decision``
# via ``_local_admission_evidence``), never recomputed by the refusal code.
# The end-to-end cases above show the values equal the priced prompt, which a
# recomputation could reproduce; these builder-level pins hand each builder an
# attempt whose token evidence is deliberately unrelated to its ETA/budget
# fields, so only a threaded value can be echoed — and an evidence-less
# attempt must not fabricate fields (additive contract).
# ---------------------------------------------------------------------------

#: Token evidence deliberately unrelated to the attempt's ETA/budget/projected
#: fields, so a refactor that "recomputed" from the refusal's own inputs could
#: never reproduce it.
_THREADED_NEW_TOKENS = 12_345
_THREADED_BUDGET_TOKENS = 6_789


def _evidence_attempt(
    reason: str = "eta_exceeds_budget",
    *,
    new_tokens: int | None = _THREADED_NEW_TOKENS,
    budget_tokens: int | None = _THREADED_BUDGET_TOKENS,
) -> dict:
    """An admission attempt record shaped like ``_local_admission_evidence``."""
    attempt: dict = {
        "status": "local_admission_reject",
        "reason": reason,
        "eta_prefill_seconds": 100.0,
        "budget_seconds": 60.0,
        "projected_wait_seconds": 150.0,
    }
    if new_tokens is not None:
        attempt["new_tokens"] = new_tokens
    if budget_tokens is not None:
        attempt["budget_tokens"] = budget_tokens
    return attempt


def _assert_threaded_token_evidence(
    result, new_tokens: int, budget_tokens: int
) -> dict:
    """The refusal advertises *exactly* the threaded numbers, headers and body."""
    assert result.status_code == 429
    body = json.loads(result.body)
    assert int(result.headers["X-Local-Admission-New-Tokens"]) == new_tokens
    assert int(result.headers["X-Local-Admission-Budget-Tokens"]) == budget_tokens
    assert body.get("new_tokens") == new_tokens
    assert body.get("budget_tokens") == budget_tokens
    return body


def test_local_admission_reject_refusal_echoes_the_threaded_evidence():
    """Exhaustion-path builder returns the decision's numbers verbatim (AC3)."""
    result = provider._build_local_admission_reject_response(
        _evidence_attempt()
    )
    body = _assert_threaded_token_evidence(
        result, _THREADED_NEW_TOKENS, _THREADED_BUDGET_TOKENS
    )
    # The values echo neither the ETA nor the budget seconds, proving the
    # builder read the attempt's evidence rather than deriving a token count.
    assert body["new_tokens"] not in (100, 60, 150)


def test_local_only_refusal_echoes_the_threaded_evidence():
    """Local-only builder returns the decision's numbers verbatim (AC3)."""
    result = provider._build_local_only_refusal_response(_evidence_attempt())
    _assert_threaded_token_evidence(
        result, _THREADED_NEW_TOKENS, _THREADED_BUDGET_TOKENS
    )


def test_refusal_without_token_evidence_fabricates_none():
    """No evidence in the thread ⇒ no token headers/fields on either builder.

    Pins the additive contract: the builders must never invent a token count
    when the admission decision carries none (AC3 — sourced, not computed).
    """
    for builder in (
        provider._build_local_admission_reject_response,
        provider._build_local_only_refusal_response,
    ):
        result = builder(_evidence_attempt(new_tokens=None, budget_tokens=None))
        assert "X-Local-Admission-New-Tokens" not in result.headers
        assert "X-Local-Admission-Budget-Tokens" not in result.headers
        body = json.loads(result.body)
        assert "new_tokens" not in body
        assert "budget_tokens" not in body


# ---------------------------------------------------------------------------
# AC2/AC4 (LP-0MVNTT4X8009IO2X): ETA/occupancy refusals advertise a truthful
# retry time
#
# Every ETA/occupancy refusal describes a *wait the client can outlast*, so the
# refusal must advertise:
#
# - ``Retry-After`` = ``ceil(projected_wait_seconds - budget_seconds)``, never
#   below 1 s and capped by ``server.max_retry_after_seconds`` (AC2);
# - ``retry_at`` — the same instant as an RFC 3339 UTC timestamp, so a
#   scheduler need not trust its own clock (AC2);
# - ``X-Local-Admission-New-Tokens`` / ``X-Local-Admission-Budget-Tokens`` (and
#   the body equivalents) so a client can decide to compact instead of retrying
#   (AC4).
#
# Both refusal builders are pinned, because either can answer a refused
# client: ``_build_local_only_refusal_response()`` (a local-only client, remote
# still available) and ``_build_local_admission_reject_response()`` (the chain
# is exhausted as well, reached via ``_build_exhausted_response()``). The
# projected wait is pinned per case so the advertised arithmetic is exactly
# checkable — the estimator itself is covered by ``test_prefill_eta_*``.
#
# TDD red phase removed by LP-0MVQ23API001UOET (reason-aware retry timing) and
# LP-0MVQ23AYQ000CGTW (token evidence): both builders now derive
# ``Retry-After``/``retry_at`` from the projected wait and emit the token
# evidence, so the cases assert the real contract rather than xfailing.
# ---------------------------------------------------------------------------

#: Prompt size priced by every case. Unecached, so ``new_tokens`` == this.
PROMPT_NEW_TOKENS = 20_000
#: Flat cold-prefill cap configured for every case: above the prompt, so the
#: ``prefill_budget_exceeded`` bound never pre-empts an occupancy bound.
COLD_PREFILL_BUDGET_TOKENS = 25_000
#: Tolerance (seconds) on the absolute ``retry_at`` instant.
RETRY_AT_TOLERANCE_SECONDS = 5


@dataclass(frozen=True)
class _OccupancyRefusalCase:
    """One reason→timing row of the ETA/occupancy refusal contract table."""

    reason: str
    #: ``"queue"`` = every local slot is busy, ``"direct"`` = a slot is free.
    path: str
    eta_prefill_seconds: float
    projected_wait_seconds: float
    budget_seconds: float
    expected_retry_after: int
    #: ``"hybrid"`` = a remote provider is available (local-only refusal) or
    #: ``"local_only"`` = the chain has no fallback (exhaustion refusal).
    chain: str = "hybrid"
    #: Extra ``server`` config for this case.
    server: dict = field(default_factory=dict)


#: Refusal whose *wait budget* is the binding bound, behind an occupied gate:
#: own prefill 100 s + 50 s of occupant service = 150 s projected wait.
_ETA_EXCEEDS_BUDGET_CASE = dict(
    reason="eta_exceeds_budget",
    path="queue",
    eta_prefill_seconds=100.0,
    projected_wait_seconds=150.0,
    budget_seconds=60.0,
    expected_retry_after=90,  # ceil(150 - 60)
)

#: The same bound applied one step earlier, on the free-slot dispatch path.
_ETA_EXCEEDS_BUDGET_DIRECT_CASE = dict(
    reason="eta_exceeds_budget_direct",
    path="direct",
    eta_prefill_seconds=150.0,
    projected_wait_seconds=150.0,
    budget_seconds=60.0,
    expected_retry_after=90,  # ceil(150 - 60)
)

#: Deadline refusal: the *deadline* binds, not the wait budget, so the honest
#: hint is the 1 s floor rather than "wait 540 s".
_PREFILL_DEADLINE_CASE = dict(
    reason="prefill_eta_exceeds_deadline",
    path="queue",
    eta_prefill_seconds=150.0,
    projected_wait_seconds=150.0,
    budget_seconds=600.0,
    expected_retry_after=1,  # wait < budget ⇒ never below 1 s
    server={"local_prefill_deadline_seconds": 120},
)

#: Deadline-relative safety budget: a 115 s ETA fits the 120 s deadline but not
#: ``0.9 x 120 s = 108 s``, so the safety bound is the binding one.
_PREFILL_SAFETY_BUDGET_CASE = dict(
    reason="prefill_eta_exceeds_safety_budget",
    path="queue",
    eta_prefill_seconds=115.0,
    projected_wait_seconds=115.0,
    budget_seconds=60.0,
    expected_retry_after=55,  # ceil(115 - 60)
    server={
        "local_prefill_deadline_seconds": 120,
        "local_admission_guard_budget_safety_fraction": 0.9,
    },
)

#: The advertised duration is capped by ``server.max_retry_after_seconds``.
_CAPPED_CASE = dict(
    reason="eta_exceeds_budget",
    path="queue",
    eta_prefill_seconds=4_000.0,
    projected_wait_seconds=5_000.0,
    budget_seconds=60.0,
    expected_retry_after=120,  # ceil(5000 - 60) = 4940, capped at 120
    server={"max_retry_after_seconds": 120},
)


_OCCUPANCY_REFUSAL_CASES = [
    pytest.param(
        _OccupancyRefusalCase(**_ETA_EXCEEDS_BUDGET_CASE),
        id="eta_exceeds_budget-local-only",
    ),
    pytest.param(
        _OccupancyRefusalCase(**_ETA_EXCEEDS_BUDGET_CASE, chain="local_only"),
        id="eta_exceeds_budget-exhausted",
    ),
    pytest.param(
        _OccupancyRefusalCase(**_ETA_EXCEEDS_BUDGET_DIRECT_CASE),
        id="eta_exceeds_budget_direct-local-only",
    ),
    pytest.param(
        _OccupancyRefusalCase(**_PREFILL_DEADLINE_CASE),
        id="prefill_eta_exceeds_deadline-local-only",
    ),
    pytest.param(
        _OccupancyRefusalCase(**_PREFILL_DEADLINE_CASE, chain="local_only"),
        id="prefill_eta_exceeds_deadline-exhausted",
    ),
    pytest.param(
        _OccupancyRefusalCase(**_PREFILL_SAFETY_BUDGET_CASE),
        id="prefill_eta_exceeds_safety_budget-local-only",
    ),
    pytest.param(
        _OccupancyRefusalCase(**_PREFILL_SAFETY_BUDGET_CASE, chain="local_only"),
        id="prefill_eta_exceeds_safety_budget-exhausted",
    ),
    pytest.param(
        _OccupancyRefusalCase(**_CAPPED_CASE),
        id="eta_exceeds_budget-capped",
    ),
]


def _case_cfg(case: _OccupancyRefusalCase) -> dict:
    """Merge the case's server overrides onto the queue-admission fixture."""
    return _queue_cfg(
        local_cold_prefill_budget_tokens=COLD_PREFILL_BUDGET_TOKENS,
        contention_queue_max_wait_seconds=case.budget_seconds,
        **case.server,
    )


def _local_only_model_config() -> dict:
    """A chain with no fallback: the local provider only (AC4 exhaustion path)."""
    return {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"}
        ],
        "aliases": ["hybrid*"],
    }


@contextmanager
def _fixed_wait_projection(case: _OccupancyRefusalCase):
    """Pin the guard's occupancy-aware projection to the case's numbers.

    The advertised timing must be derived from the *projected wait* (own
    prefill plus the occupied gate), not from the own-prefill ETA, so the
    projection is fixed here and the arithmetic in the assertions is exact.
    """
    wait = prefill_eta.ProjectedWait(
        own_prefill_seconds=case.eta_prefill_seconds,
        occupancy_seconds=max(
            0.0, case.projected_wait_seconds - case.eta_prefill_seconds
        ),
        projected_seconds=case.projected_wait_seconds,
        busy_slots=(
            1 if case.projected_wait_seconds > case.eta_prefill_seconds else 0
        ),
        coefficients=prefill_eta.Coefficients(
            coded_ms_per_token=1.0,
            prefill_ms_per_token=1.0,
            samples=6,
            calibrated=True,
        ),
    )
    with patch("proxy.prefill_eta.projected_wait_seconds", lambda *a, **k: wait):
        yield


def _parse_rfc3339_utc(value) -> datetime:
    """Parse an RFC 3339 UTC timestamp, failing loudly for anything else."""
    assert value, "the refusal body must carry an absolute retry_at"
    text = str(value)
    assert text.endswith(("Z", "+00:00")), (
        f"retry_at must be RFC 3339 UTC, got {text!r}"
    )
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    assert parsed.utcoffset() == timedelta(0), (
        f"retry_at must be UTC, got {text!r}"
    )
    return parsed


def _assert_occupancy_refusal_contract(result, case: _OccupancyRefusalCase) -> dict:
    """Assert the AC2/AC4 contract shared by both refusal builders."""
    assert result.status_code == 429
    assert result.headers["X-Local-Route"] == "refused"
    assert result.headers.get("X-Local-Fallback-Reason") == case.reason, (
        "the refusal must name the occupancy reason that refused it"
    )

    # AC2: Retry-After is ceil(projected wait - budget), min 1 s, capped.
    assert result.headers.get("Retry-After") == str(case.expected_retry_after), (
        "Retry-After must be ceil(projected_wait_seconds - budget_seconds), "
        "never below 1 s and capped by server.max_retry_after_seconds"
    )

    body = json.loads(result.body)
    assert body.get("retryable") is True, "a wait the client can outlast is retryable"
    assert body.get("retry_after") == case.expected_retry_after, (
        "the body must carry the same seconds as the Retry-After header"
    )

    # AC2: an absolute instant so the client need not trust its own clock.
    retry_at = _parse_rfc3339_utc(body.get("retry_at"))
    expected_at = datetime.now(UTC) + timedelta(seconds=case.expected_retry_after)
    assert abs((retry_at - expected_at).total_seconds()) <= RETRY_AT_TOLERANCE_SECONDS, (
        f"retry_at {retry_at.isoformat()} must be now + "
        f"{case.expected_retry_after}s, got {expected_at.isoformat()}"
    )

    # AC4: the token evidence a client compacts against, on headers and body.
    assert int(result.headers["X-Local-Admission-New-Tokens"]) == PROMPT_NEW_TOKENS
    assert (
        int(result.headers["X-Local-Admission-Budget-Tokens"])
        == COLD_PREFILL_BUDGET_TOKENS
    )
    assert body.get("new_tokens") == PROMPT_NEW_TOKENS
    assert body.get("budget_tokens") == COLD_PREFILL_BUDGET_TOKENS
    return body


def test_occupancy_refusal_case_table_matches_the_advertised_formula():
    """Every table row's expectation *is* the AC2 formula (cap applied)."""
    for param in _OCCUPANCY_REFUSAL_CASES:
        case = param.values[0]
        cap = int(case.server.get("max_retry_after_seconds", 3_600))
        expected = max(
            1,
            min(
                cap,
                math.ceil(case.projected_wait_seconds - case.budget_seconds),
            ),
        )
        assert case.expected_retry_after == expected, (
            f"{param.id}: expected_retry_after={case.expected_retry_after} "
            f"!= ceil({case.projected_wait_seconds} - {case.budget_seconds}) "
            f"capped at {cap} = {expected}"
        )


async def _issue_occupancy_refusal(case: _OccupancyRefusalCase, mixed_model_config):
    """Issue one ETA/occupancy-refused request; return ``(response, dispatches)``."""
    dispatched: list[str] = []

    async def _local(_req, _path):
        dispatched.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        dispatched.append("remote")
        return _ok_response()

    model_config = (
        mixed_model_config if case.chain == "hybrid" else _local_only_model_config()
    )
    headers = (
        {"x-local-preference": "local-only"} if case.chain == "hybrid" else {}
    )
    concurrency = _MutableConcurrency(1 if case.path == "queue" else 0, 1)

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
        patch("proxy.provider.request_priority", return_value="low"),
        _guard_patches(tokens=PROMPT_NEW_TOKENS),
        _fixed_wait_projection(case),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(headers=headers),
            "v1/chat/completions",
            model_config,
            _case_cfg(case),
        )

    return result, dispatched


@pytest.mark.parametrize("case", _OCCUPANCY_REFUSAL_CASES)
@pytest.mark.asyncio
async def test_occupancy_refusal_advertises_a_truthful_retry_time(
    case: _OccupancyRefusalCase, mixed_model_config
):
    """Each ETA/occupancy reason → Retry-After + retry_at + token evidence."""
    result, dispatched = await _issue_occupancy_refusal(case, mixed_model_config)

    assert dispatched == [], (
        "a refused ETA/occupancy request must never dispatch or move remote"
    )
    body = _assert_occupancy_refusal_contract(result, case)
    assert body["code"] == provider.EXHAUSTION_CODE_LOCAL_ADMISSION


# ---------------------------------------------------------------------------
# AC6/AC7 (LP-0MVNTT4X8009IO2X): the refusal contract as ONE table
#
# The sections above pin each family in depth. This section pins the
# *cross-reason* contract in a single table, so a refusal reason cannot ship
# with an inconsistent retry story:
#
# - every refusal is a 429, is stamped ``X-Local-Route: refused`` and names the
#   reason that refused it (AC6);
# - ``Retry-After`` / ``retry_at`` are present exactly when waiting helps —
#   the ETA/occupancy reasons and a real cooldown — and absent exactly when it
#   cannot (``prefill_budget_exceeded``, ``context_too_large``) (AC2/AC6);
# - the admission token evidence is present on *every* refusal (AC2);
# - a refused request is never dispatched, and never moved to a remote provider
#   behind a ``local-only`` client's back (AC6).
#
# ``reason=cooldown`` (the local entry cooling down while a remote provider is
# available) is the one reason whose timer must come from *real* state instead
# of the ETA formula, so it gets a dedicated test as well (AC7): the advertised
# remainder is the cooldown's, not ``suggested_retry_after_seconds()``'s.
#
# Non-admission skips: ``cooldown`` and ``context_too_large`` are pinned here
# because their contract is already decided (a real cooldown remainder, and
# "compact, do not wait"). The lease-held and contention-queue-expiry causes
# are pinned by ``test_local_non_admission_retry_timing.py``
# (LP-0MVQ23B80009OZ9V), which also pins this file's non-admission rows green.
#
# Every row is green: the AC1/AC2 rows since LP-0MVQ23API001UOET (reason-aware
# timing) and LP-0MVQ23AYQ000CGTW (token evidence); the two non-admission rows
# (``cooldown`` and ``context_too_large``) since LP-0MVQ23B80009OZ9V
# (non-admission retry timing). A red row is a real regression.
# ---------------------------------------------------------------------------

#: Real cooldown remaining for the cooldown-refusal cases. Kept far enough above
#: the 1 s ``Retry-After`` floor that an ETA-formula answer is distinguishable.
COOLDOWN_REMAINDER_SECONDS = 90
#: What ``suggested_retry_after_seconds()`` is stubbed to return while the
#: cooldown cases run, so an ETA-derived answer can never masquerade as (or be
#: mistaken for) the cooldown remainder.
ETA_FORMULA_SENTINEL_SECONDS = 999
#: Local entry used by every case in this file.
LOCAL_PROVIDER_NAME = "local-llama"


@contextmanager
def _local_entry_in_cooldown(seconds: float = COOLDOWN_REMAINDER_SECONDS):
    """Put the local entry into a *real* cooldown for the current request.

    The expiry is set half a second beyond *seconds* so ``int(remaining)`` —
    what ``_provider_cooldown_remaining()`` reports and what a truthful
    refusal must advertise — is exactly *seconds*.
    """
    previous = provider._provider_unavailable_until.get(LOCAL_PROVIDER_NAME)
    provider._provider_unavailable_until[LOCAL_PROVIDER_NAME] = (
        time.time() + seconds + 0.5
    )
    try:
        yield
    finally:
        if previous is None:
            provider._provider_unavailable_until.pop(LOCAL_PROVIDER_NAME, None)
        else:
            provider._provider_unavailable_until[LOCAL_PROVIDER_NAME] = previous


@dataclass(frozen=True)
class _RefusalContractRow:
    """One reason row of the cross-reason refusal contract table."""

    reason: str
    #: How the row reaches its refusal: ``"occupancy"`` (guard refusal, case
    #: taken from the AC2 table), ``"prefill_budget"`` (2026-10-25 shape),
    #: ``"context_bypass"`` (physical context cap) or ``"cooldown"``.
    trigger: str
    #: ``True`` when waiting helps, so the refusal must advertise a timer.
    time_retryable: bool
    #: ``True`` when the reason is ``too_large`` and the pin is therefore
    #: declined (LP-0MVSLZY470051ZB2 AC4): the request is promoted to the
    #: remote tiers rather than refused, so the row asserts the promotion
    #: shape (no 429, no timer) instead of the refusal shape.
    promoted: bool = False
    #: The AC2 occupancy case that reaches this reason, when applicable.
    occupancy_case: _OccupancyRefusalCase | None = None
    #: ``new_tokens`` the refusal must advertise. Defaults to the shared
    #: ``PROMPT_NEW_TOKENS``; the ``prefill_budget_exceeded`` row reproduces the
    #: 2026-10-25 incident verbatim, so it advertises that prompt's size.
    expected_new_tokens: int = PROMPT_NEW_TOKENS


def _occupancy_row(reason: str) -> _RefusalContractRow:
    """Row for an ETA/occupancy reason, reusing the AC2 case table verbatim."""
    case = next(
        param.values[0]
        for param in _OCCUPANCY_REFUSAL_CASES
        if param.values[0].reason == reason and param.values[0].chain == "hybrid"
    )
    return _RefusalContractRow(
        reason=reason,
        trigger="occupancy",
        time_retryable=True,
        occupancy_case=case,
    )


#: The contract table: exactly one row per refusal reason, verbatim from the
#: per-reason contract of LP-0MVNTT4X8009IO2X (AC1/AC2/AC3) and the docs item
#: LP-0MVQ23BH7006HLKB (wait / compact / refuse).
_REFUSAL_CONTRACT_ROWS = [
    _occupancy_row("eta_exceeds_budget"),
    _occupancy_row("eta_exceeds_budget_direct"),
    _occupancy_row("prefill_eta_exceeds_deadline"),
    _occupancy_row("prefill_eta_exceeds_safety_budget"),
    # Permanently over the flat cold-prefill cap: time cannot heal it. The
    # pin is declined for a too-large verdict (LP-0MVSLZY470051ZB2 AC4), so
    # the row asserts promotion to the remote tiers — the no-timer contract
    # still holds because waiting cannot help.
    _RefusalContractRow(
        reason="prefill_budget_exceeded",
        trigger="prefill_budget",
        time_retryable=False,
        promoted=True,
        expected_new_tokens=INCIDENT_NEW_TOKENS,
    ),
    # Cannot fit a KV slot: the remedy is compaction, not a wait. Same
    # promotion boundary as the row above.
    _RefusalContractRow(
        reason="context_too_large",
        trigger="context_bypass",
        time_retryable=False,
        promoted=True,
    ),
    # A real, finite unavailability: waiting helps, for the cooldown's length.
    _RefusalContractRow(
        reason="cooldown", trigger="cooldown", time_retryable=True
    ),
]

#: Every row is asserted green: the choke point refuses every exit, the
#: reason-token child records the per-branch tokens, and the non-admission
#: rows (cooldown, ``context_too_large``) advertise their real expiry/no-timer
#: contract since LP-0MVQ23B80009OZ9V. A future red row is a real failure.
_REFUSAL_CONTRACT_PARAMS = [
    pytest.param(row, id=row.reason) for row in _REFUSAL_CONTRACT_ROWS
]


async def _issue_cooldown_refusal(mixed_model_config) -> tuple:
    """A local-only client while its local entry cools down; remote available.

    Returns ``(response, dispatches)``. The ETA formula's answer is stubbed to
    a sentinel so a test can prove the advertised timer is not derived from it.
    """
    dispatched: list[str] = []

    async def _local(_req, _path):
        dispatched.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        dispatched.append("remote")
        return _ok_response()

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(1, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        patch(
            "proxy.prefill_eta.suggested_retry_after_seconds",
            lambda *a, **k: ETA_FORMULA_SENTINEL_SECONDS,
        ),
        _guard_patches(tokens=PROMPT_NEW_TOKENS),
        _local_entry_in_cooldown(),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(headers={"x-local-preference": "local-only"}),
            "v1/chat/completions",
            mixed_model_config,
            _cold_budget_cfg(),
        )

    return result, dispatched


async def _issue_refusal(row: _RefusalContractRow, mixed_model_config) -> tuple:
    """Issue the request that must be refused for *row*; ``(response, calls)``."""
    if row.trigger == "occupancy":
        assert row.occupancy_case is not None
        return await _issue_occupancy_refusal(row.occupancy_case, mixed_model_config)
    if row.trigger == "prefill_budget":
        return await _issue_local_only_prefill_budget_refusal(mixed_model_config)
    if row.trigger == "context_bypass":
        calls: list[str] = []

        async def _local(_req, _path):
            calls.append("local")
            return _ok_response()

        async def _remote(_req, _path, _pc):
            calls.append("remote")
            return _ok_response()

        with (
            patch("proxy.router.proxy_to_local", _local),
            patch("proxy.server.proxy_to_remote", _remote),
            patch(
                "proxy.provider._get_local_concurrency_info",
                _MutableConcurrency(0, 1),
            ),
            patch("proxy.provider.request_priority", return_value="low"),
            _guard_patches(tokens=PROMPT_NEW_TOKENS, skip_reason=row.reason),
        ):
            result = await provider.proxy_with_fallback(
                _DummyRequest(headers={"x-local-preference": "local-only"}),
                "v1/chat/completions",
                mixed_model_config,
                _cold_budget_cfg(),
            )
        return result, calls
    if row.trigger == "cooldown":
        return await _issue_cooldown_refusal(mixed_model_config)
    raise AssertionError(f"unknown refusal trigger {row.trigger!r}")


def test_refusal_contract_table_has_exactly_one_row_per_reason():
    """The table is the contract: one row per reason, no silent drop.

    A reason that disappears from the table would silently lose its retry
    story, so the reason set is pinned here (and only here — the per-reason
    expectations live in the rows themselves).
    """
    reasons = [row.reason for row in _REFUSAL_CONTRACT_ROWS]
    assert len(reasons) == len(set(reasons)), f"duplicate reason rows: {reasons}"
    assert set(reasons) == {
        "eta_exceeds_budget",
        "eta_exceeds_budget_direct",
        "prefill_eta_exceeds_deadline",
        "prefill_eta_exceeds_safety_budget",
        "prefill_budget_exceeded",
        "context_too_large",
        "cooldown",
    }, "a refusal reason is missing from (or unexpected in) the contract table"

    not_time_retryable = {
        row.reason for row in _REFUSAL_CONTRACT_ROWS if not row.time_retryable
    }
    assert not_time_retryable == {"prefill_budget_exceeded", "context_too_large"}, (
        "only the causes waiting cannot fix may omit Retry-After/retry_at"
    )

    for row in _REFUSAL_CONTRACT_ROWS:
        if row.trigger == "occupancy":
            assert row.occupancy_case is not None
            assert row.occupancy_case.reason == row.reason, (
                "an occupancy row must reuse the AC2 case table, not restate "
                "its numbers"
            )


@pytest.mark.parametrize(
    "row",
    _REFUSAL_CONTRACT_PARAMS,
    ids=[row.reason for row in _REFUSAL_CONTRACT_ROWS],
)
@pytest.mark.asyncio
async def test_refusal_reason_follows_the_contract_table(
    row: _RefusalContractRow, mixed_model_config
):
    """Per reason: status, timer presence/absence, reason naming, token evidence.

    A ``too_large`` row (``promoted=True``) asserts the promotion shape
    instead: remote serves the request with no refusal and no timer, because
    waiting cannot help and the pin is declined (LP-0MVSLZY470051ZB2 AC4).
    """
    result, dispatched = await _issue_refusal(row, mixed_model_config)

    if row.promoted:
        assert dispatched == ["remote"], (
            f"{row.reason}: a too-large pin-local request must reach the "
            f"remote tiers (dispatched={dispatched})"
        )
        assert result.status_code == 200, (
            f"{row.reason}: too-large must not be refused for local capacity, "
            f"got {result.status_code}"
        )
        headers = result.headers
        body = json.loads(result.body)
        assert headers.get("X-Local-Route") == "remote", (
            f"{row.reason}: promotion reports the remote route"
        )
        # Waiting cannot help: even a promoted request never advertises a
        # delay header it would not honour (parent retry contract).
        assert headers.get("Retry-After") is None, (
            f"{row.reason}: waiting cannot help a too-large request"
        )
        assert body.get("retryable") is not True, (
            f"{row.reason}: no retryable flag on a promoted request"
        )
        assert body.get("code") != provider.EXHAUSTION_CODE_LOCAL_ADMISSION, (
            f"{row.reason}: promotion is not a local-capacity refusal"
        )
        return

    assert dispatched == [], (
        f"{row.reason}: a refused request must never be dispatched locally or "
        "moved to a remote provider"
    )
    assert result.status_code == 429, (
        f"{row.reason}: every admission refusal answers 429"
    )
    headers = result.headers
    body = json.loads(result.body)

    assert headers.get("X-Local-Route") == "refused"
    advertised = " ".join(
        str(value or "")
        for value in (headers.get("X-Local-Fallback-Reason"), body.get("reason"))
    ).lower()
    assert row.reason.lower() in advertised, (
        f"{row.reason}: the refusal must name the reason that refused it, "
        f"got header={headers.get('X-Local-Fallback-Reason')!r} "
        f"body={body.get('reason')!r}"
    )

    # AC2/AC6: the timer is present exactly when waiting helps.
    assert (headers.get("Retry-After") is not None) is row.time_retryable, (
        f"{row.reason}: Retry-After must be "
        f"{'present' if row.time_retryable else 'absent'}"
    )
    assert (bool(body.get("retry_at"))) is row.time_retryable, (
        f"{row.reason}: retry_at must be "
        f"{'an RFC 3339 UTC instant' if row.time_retryable else 'absent'}"
    )
    assert (body.get("retry_after") is not None) is row.time_retryable, (
        f"{row.reason}: body retry_after must mirror the header presence"
    )
    assert body.get("retryable") is row.time_retryable, (
        f"{row.reason}: the body must say whether a retry can help"
    )
    if row.time_retryable:
        assert headers["Retry-After"] == str(body["retry_after"]), (
            f"{row.reason}: the header and the body must advertise one delay"
        )
        _parse_rfc3339_utc(body["retry_at"])

    # AC2: the admission token evidence is present on *every* refusal.
    assert (
        int(headers["X-Local-Admission-New-Tokens"]) == row.expected_new_tokens
    ), f"{row.reason}: the client needs the token evidence to size a compaction"
    assert (
        int(headers["X-Local-Admission-Budget-Tokens"]) == COLD_PREFILL_BUDGET_TOKENS
    )
    assert body.get("new_tokens") == row.expected_new_tokens
    assert body.get("budget_tokens") == COLD_PREFILL_BUDGET_TOKENS


@pytest.mark.asyncio
async def test_cooldown_refusal_advertises_the_cooldown_remainder(
    mixed_model_config,
):
    """AC7: a cooldown refusal advertises the cooldown, not the ETA formula.

    The local entry is cooling down for :data:`COOLDOWN_REMAINDER_SECONDS` and
    ``suggested_retry_after_seconds()`` is stubbed to
    :data:`ETA_FORMULA_SENTINEL_SECONDS`, so the advertised delay can only be
    the real cooldown remainder.
    """
    result, dispatched = await _issue_cooldown_refusal(mixed_model_config)

    assert dispatched == [], (
        "X-Local-Preference: local-only means refuse, not fall through to remote"
    )
    assert result.status_code == 429
    assert result.headers.get("X-Local-Route") == "refused"

    body = json.loads(result.body)
    advertised_reason = " ".join(
        str(value or "")
        for value in (
            result.headers.get("X-Local-Fallback-Reason"),
            body.get("reason"),
        )
    ).lower()
    assert "cooldown" in advertised_reason, (
        "the refusal must attribute the refusal to the cooldown, got "
        f"{advertised_reason!r}"
    )

    retry_after = int(result.headers["Retry-After"])
    assert retry_after == body["retry_after"], (
        "the Retry-After header and the body must advertise one delay"
    )
    assert retry_after != ETA_FORMULA_SENTINEL_SECONDS, (
        "the cooldown remainder must come from real availability state, not "
        "from the ETA formula"
    )
    assert COOLDOWN_REMAINDER_SECONDS - 2 <= retry_after <= COOLDOWN_REMAINDER_SECONDS, (
        f"Retry-After must be the cooldown remainder "
        f"(~{COOLDOWN_REMAINDER_SECONDS}s), got {retry_after}s"
    )

    retry_at = _parse_rfc3339_utc(body["retry_at"])
    expected_at = datetime.now(UTC) + timedelta(seconds=retry_after)
    assert abs((retry_at - expected_at).total_seconds()) <= RETRY_AT_TOLERANCE_SECONDS, (
        f"retry_at {retry_at.isoformat()} must be the advertised instant, got "
        f"{expected_at.isoformat()}"
    )
    assert body.get("retryable") is True, "a cooldown is a wait the client can outlast"


# ---------------------------------------------------------------------------
# Capacity verdict in the refusal body and the single verdict log line
# (LP-0MVSMRSU5001JTMT, child of LP-0MVSMI7JK002FKUF)
#
# The pin-local refusal must say whether waiting can help: ``deferrable`` for a
# temporal exit (stay local, retry later) or ``too_large`` when the request
# itself exceeds the local serving envelope. Both surfaces are pinned here:
#
# - the 429 body gains exactly one field — ``verdict`` — beside ``reason``;
# - the ``local_only_refused`` line gains exactly one key — ``verdict`` — and
#   still names the parsed ``preference`` and the recorded ``reason``.
#
# TDD red phase: these assertions started ``xfail(strict=True)`` while the
# wiring child (LP-0MVSMINPR006SWT6) had not landed. That child removed the
# markers in the same change that added the behaviour, so a failure here now
# means the contract broke rather than that the wiring is pending.
# ---------------------------------------------------------------------------

#: Reused ``key=value`` parser shape (same as the log-grep tooling).
_VERDICT_KV_RE = re.compile(r"(\w+)=(\S+)")


def _parse_verdict_kv(line: str) -> dict[str, str]:
    """Parse a log line's ``key=value`` pairs (the operator grep shape)."""
    return {match.group(1): match.group(2) for match in _VERDICT_KV_RE.finditer(line)}


def _local_only_refusal_lines(caplog) -> list[str]:
    """Every ``local_only_refused`` line captured in *caplog*."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("local_only_refused ")
    ]


#: The observed 2026-10-28 incident shape: the priced own-ETA (31.7 s) fits the
#: 60 s budget, the projection with one occupant (63.4 s) does not, and the
#: prompt (6 627 new tokens) is far inside the 25 000-token cold-prefill cap.
_INCIDENT_ATTEMPT: dict = {
    "status": "local_admission_reject",
    "reason": "eta_exceeds_budget_direct",
    "eta_prefill_seconds": 31.7,
    "budget_seconds": 60.0,
    "projected_wait_seconds": 63.4,
    "new_tokens": 6_627,
    "budget_tokens": 25_000,
}

#: Body keys the local-only refusal carries today (before ``verdict``). Pinned
#: literally so a rename or a silent removal fails the additive-only guard.
_BODY_KEYS_BEFORE_VERDICT = frozenset(
    {
        "error",
        "code",
        "detail",
        "reason",
        "eta_prefill_seconds",
        "budget_seconds",
        "retryable",
        "retry_after",
        "suggested_action",
        "projected_wait_seconds",
        "retry_at",
        "new_tokens",
        "budget_tokens",
    }
)

def test_incident_refusal_body_carries_the_deferrable_verdict():
    """AC1: the incident shape is temporal, and nothing else about it moves."""
    result = provider._build_local_only_refusal_response(dict(_INCIDENT_ATTEMPT))

    assert result.status_code == 429
    body = json.loads(result.body)
    assert body["verdict"] == "deferrable"
    assert body["reason"] == "eta_exceeds_budget_direct"

    # The retry story stays exactly what ``_refusal_retry_timing`` produced —
    # recomputed here so a timing change could never ride along with the
    # verdict (re-specified, not re-implemented).
    expected = provider._refusal_retry_timing(
        "eta_exceeds_budget_direct",
        eta_prefill_seconds=31.7,
        budget_seconds=60.0,
        projected_wait_seconds=63.4,
    )
    assert body["retryable"] is expected.retryable
    assert body["retry_after"] == expected.retry_after
    # ``retry_at`` is stamped from ``now`` inside the builder, so a recomputed
    # expectation taken microseconds later cannot be string-equal — compare the
    # absolute instants within a tolerance instead of re-specifying the value.
    delta = abs(
        _parse_rfc3339_utc(body["retry_at"])
        - _parse_rfc3339_utc(expected.retry_at)
    )
    assert delta <= timedelta(seconds=2), (
        f"retry_at must be the recomputed instant (now + retry_after); "
        f"body={body['retry_at']!r} expected≈{expected.retry_at!r}"
    )
    assert result.headers["Retry-After"] == str(expected.retry_after)

    # Existing fields and headers are untouched.
    assert body["projected_wait_seconds"] == 63.4
    assert body["new_tokens"] == 6_627
    assert body["budget_tokens"] == 25_000
    assert result.headers["X-Local-Route"] == "refused"
    assert (
        result.headers["X-Local-Fallback-Reason"] == "eta_exceeds_budget_direct"
    )
    assert result.headers["X-Local-Admission"] == "reject"


def test_refusal_body_keys_are_additive_only():
    """AC6: exactly one new key — nothing renamed, nothing removed."""
    result = provider._build_local_only_refusal_response(dict(_INCIDENT_ATTEMPT))
    body = json.loads(result.body)

    assert set(body) == _BODY_KEYS_BEFORE_VERDICT | {"verdict"}


@pytest.mark.parametrize(
    "reason", ["prefill_budget_exceeded", "context_too_large"]
)
def test_too_large_refusals_carry_the_too_large_verdict(reason):
    """AC2: both permanent size exits are ``too_large`` with no retry timer."""
    result = provider._build_local_only_refusal_response(
        {
            "status": "local_admission_reject",
            "reason": reason,
            "eta_prefill_seconds": 0.0,
            "budget_seconds": 60.0,
            "new_tokens": 31_246,
            "budget_tokens": 25_000,
        }
    )

    assert result.status_code == 429
    body = json.loads(result.body)
    assert body["verdict"] == "too_large"
    assert body["reason"] == reason
    assert "Retry-After" not in result.headers
    assert "retry_at" not in body
    assert body["retryable"] is False


def test_one_verdict_log_line_per_refusal(caplog):
    """AC3: one line per refusal, naming preference + verdict + reason."""
    caplog.set_level(logging.INFO, logger="llama-proxy")

    provider._build_local_only_refusal_response(dict(_INCIDENT_ATTEMPT))

    lines = _local_only_refusal_lines(caplog)
    assert len(lines) == 1, f"one line per refusal; got {lines!r}"
    fields = _parse_verdict_kv(lines[0])
    assert fields.get("reason") == "eta_exceeds_budget_direct"
    assert fields.get("verdict") == "deferrable"
    assert fields.get("preference"), (
        f"the line must name the intent that produced the refusal; got {fields!r}"
    )


async def _issue_guard_refusal(
    preference: str | None, *, tokens: int = INCIDENT_NEW_TOKENS
) -> tuple:
    """Drive the end-to-end over-budget refusal for *preference*.

    Returns ``(response, provider_calls)``; a ``prefer-local`` request falls
    through to remote (no refusal), a pin-local one is refused locally.
    *tokens* is the priced prompt size — the default is the incident shape
    (over the cold-prefill cap), a smaller one keeps the sole blocking signal
    temporal so a temporal exit can be asserted as ``deferrable``.
    """
    model_config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {
                "name": "remote-fallback",
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
            },
        ],
        "aliases": ["hybrid*"],
    }
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    headers = {"x-local-preference": preference} if preference else {}
    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(1, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _guard_patches(tokens=tokens),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(headers=headers),
            "v1/chat/completions",
            model_config,
            _cold_budget_cfg(),
        )
    return result, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("preference", ["prefer-local", "remote-ok", None])
async def test_non_refusal_paths_log_no_verdict_line(preference, caplog):
    """AC3: only a refusal produces a verdict; fall-through logs none."""
    caplog.set_level(logging.INFO, logger="llama-proxy")

    result, calls = await _issue_guard_refusal(preference)

    assert result.status_code == 200, "the default client keeps the remote answer"
    assert calls == ["remote"]
    assert _local_only_refusal_lines(caplog) == []


@pytest.mark.asyncio
async def test_choke_point_refusal_logs_one_verdict_line(caplog):
    """AC3: the chain-cooldown choke-point refusal logs exactly one line."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    provider._provider_unavailable_until["local-llama"] = time.time() + 60
    try:
        # A prompt that fits the cold-prefill cap: the cooldown is then the
        # *only* blocking signal, so the verdict must be ``deferrable``
        # (waiting helps) rather than ``too_large`` (the prompt never fits).
        result, calls = await _issue_guard_refusal(
            "local-only", tokens=INCIDENT_BUDGET_TOKENS // 2
        )
    finally:
        provider._provider_unavailable_until.pop("local-llama", None)

    assert result.status_code == 429
    assert calls == [], "a pin-local client must never be moved remote"
    assert result.headers["X-Local-Fallback-Reason"] == "local_provider_cooldown"

    lines = _local_only_refusal_lines(caplog)
    assert len(lines) == 1, f"one line per refusal; got {lines!r}"
    fields = _parse_verdict_kv(lines[0])
    assert fields.get("reason") == "local_provider_cooldown"
    assert fields.get("verdict") == "deferrable"


@pytest.mark.asyncio
async def test_too_large_pin_local_promotes_instead_of_refusing(caplog):
    """Pin AC4: a too-large pin-local request must not be refused for capacity.

    The incident shape (``prefill_budget_exceeded``) outgrows the local
    serving envelope; waiting cannot help, so the pin is declined and the
    remote tiers serve the request — no 429, no refusal line, no promotion of
    a fits-local request either (LP-0MVSLZY470051ZB2 AC4; the chain-level
    strip belongs to LP-0MVSM06LB001PRU4).
    """
    caplog.set_level(logging.INFO, logger="llama-proxy")

    result, calls = await _issue_guard_refusal("local-only")

    assert result.status_code == 200, (
        f"a too-large request must fall through to remote, got "
        f"{result.status_code}"
    )
    assert calls == ["remote"], f"the remote tiers must serve; got {calls!r}"
    assert _local_only_refusal_lines(caplog) == [], (
        "a too-large request must not produce a pin-local refusal line"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("preference", ["local-preferred", "local-only"])
async def test_pin_local_with_a_free_slot_is_served_locally(
    mixed_model_config, preference
):
    """Pin AC2: a pin-local request that fits and finds a slot is served locally.

    The pin only ever removes the *remote* option; a request local can serve
    right now must be served locally with no refusal, no queueing and a single
    ``X-Local-Route: local`` stamp (parent AC2 / AC6). Both wire spellings are
    asserted because the ``local-only`` alias must keep working.
    """
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(0, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(headers={"x-local-preference": preference}),
            "v1/chat/completions",
            mixed_model_config,
            _queue_cfg(),
        )

    assert result.status_code == 200, f"got {result.status_code}"
    assert calls == ["local"], f"the local route must serve; got {calls!r}"
    assert result.headers.get("X-Local-Route") == "local"
    assert contention_queue.queue_depth() == 0


# ---------------------------------------------------------------------------
# AC4/AC5 (LP-0MVSM06BK008FZUE): queue invariants of the deferral refusal
#
# A deferral ends at a contention-queue exit: the wait budget expired, the
# depth cap refused to enqueue the request, or the held lease outlived the
# lease queue. For a ``local-preferred`` request whose verdict is
# ``deferrable`` each of those exits must produce the retryable 429 exactly
# once, without leaving queue or lease state behind:
#
# - enqueued at most once (one ``wait_for_local_slot()`` call per request);
# - the queue depth back to its pre-request value (a seeded holder untouched);
# - no dispatch/lease record and no lease grant registered for the session;
# - ``X-Local-Route`` stamped exactly once by the refusal builder.
#
# The rows are driven through the real queue where the exit is a queue
# decision (a seeded holder reaches the depth cap deterministically) and by a
# stubbed expiry where the point is the bookkeeping, not the wait itself.
# ---------------------------------------------------------------------------

PIN_LOCAL = "local-preferred"


def _lease_active_response() -> Response:
    """The router's lease-held 503 (``no_slots_available``)."""
    return Response(
        content=json.dumps(
            {
                "error": {
                    "type": "server_busy",
                    "code": "no_slots_available",
                    "message": "Local backend busy. Owner session holds the lease.",
                },
                "status": 503,
                "reason": "local_lease_active",
                "local_owner_session_id": "owner-session",
            }
        ),
        status_code=503,
        media_type="application/json",
    )


async def _issue_deferral_exit(
    model_config: dict, trigger: str, extra_patches: list | None = None
) -> tuple:
    """Drive one deferral exit; return ``(response, dispatches, wait_calls)``.

    ``wait_calls`` records every ``contention_queue.wait_for_local_slot()``
    invocation, which is the enqueue-once invariant's observable evidence: a
    refused request may enter the queue at most once.
    """
    dispatches: list[str] = []
    wait_calls: list[int] = []
    real_wait = contention_queue.wait_for_local_slot

    async def _counting_wait(**kwargs):
        wait_calls.append(1)
        if trigger in ("queue_wait_expiry", "lease_active_fallback"):
            # Budget already spent and no slot freed within it (no real
            # sleeping in this test); the depth-cap row below exercises the
            # real cap check instead.
            return None
        return await real_wait(**kwargs)

    async def _local(_req, _path):
        dispatches.append("local")
        if trigger == "lease_active_fallback":
            return _lease_active_response()
        return _ok_response()

    async def _remote(_req, _path, _pc):  # pragma: no cover - must never run
        dispatches.append("remote")
        return _ok_response()

    if trigger == "queue_depth_cap":
        # One concurrent waiter holds the queue's only depth slot.
        contention_queue._waiters.append((0.0, None, "queue-holder", False, None))

    config = _queue_cfg(
        contention_queue_max_depth=1 if trigger == "queue_depth_cap" else 4,
    )
    concurrency = (1, 1) if trigger != "lease_active_fallback" else (0, 1)

    with ExitStack() as stack:
        stack.enter_context(patch("proxy.router.proxy_to_local", _local))
        stack.enter_context(patch("proxy.server.proxy_to_remote", _remote))
        stack.enter_context(
            patch(
                "proxy.provider._get_local_concurrency_info",
                _MutableConcurrency(*concurrency),
            )
        )
        stack.enter_context(
            patch("proxy.provider.request_priority", return_value="low")
        )
        stack.enter_context(
            patch("proxy.contention_queue.wait_for_local_slot", _counting_wait)
        )
        for ctx in extra_patches or []:
            stack.enter_context(ctx)
        result = await provider.proxy_with_fallback(
            _DummyRequest(headers={"X-Local-Preference": PIN_LOCAL}),
            "v1/chat/completions",
            model_config,
            config,
        )
    return result, dispatches, wait_calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trigger", "expected_reason", "expected_dispatches", "seeded"),
    [
        ("queue_depth_cap", "fallback_after_queue", [], 1),
        ("queue_wait_expiry", "fallback_after_queue", [], 0),
        ("lease_active_fallback", "local_lease_active", ["local"], 0),
    ],
    ids=["queue_depth_cap", "queue_wait_expiry", "lease_active_fallback"],
)
async def test_deferral_exit_refusal_leaks_no_queue_slot_or_lease(
    mixed_model_config, trigger, expected_reason, expected_dispatches, seeded
):
    """Every deferral exit: one enqueue attempt, no queue/lease leak, one stamp."""
    import proxy.server as proxy_server

    depth_before = contention_queue.queue_depth()
    records_before = dict(proxy_server.local_dispatch_records)
    grants_before = dict(contention_queue._local_grant_times)

    result, dispatches, wait_calls = await _issue_deferral_exit(
        mixed_model_config, trigger
    )

    assert dispatches == expected_dispatches, (
        f"{trigger}: a deferred pin-local request is refused, never promoted "
        f"(dispatched={dispatches})"
    )
    # Enqueue-once: the request is offered to the queue exactly once, never in
    # a loop (a second attempt would mean the refusal re-entered the queue).
    assert wait_calls == [1], (
        f"{trigger}: expected exactly one queue attempt, got {len(wait_calls)}"
    )
    # Depth restored: only the deliberately seeded holder remains.
    assert contention_queue.queue_depth() == depth_before + seeded, (
        f"{trigger}: the refusal leaked a queue slot "
        f"(depth {contention_queue.queue_depth()}, seeded {seeded})"
    )
    # No lease was taken for a request that is being refused.
    assert proxy_server.local_dispatch_records == records_before, (
        f"{trigger}: a refused request must not leave a dispatch/lease record"
    )
    assert contention_queue._local_grant_times == grants_before, (
        f"{trigger}: a refused request must not be registered as a lease owner"
    )

    assert result.status_code == 429, f"{trigger}: got {result.status_code}"
    assert result.headers.getlist("x-local-route") == ["refused"], (
        f"{trigger}: X-Local-Route must be stamped exactly once, got "
        f"{result.headers.getlist('x-local-route')!r}"
    )
    body = json.loads(result.body)
    assert body.get("reason") == expected_reason, (
        f"{trigger}: the refusal must name {expected_reason!r}, got "
        f"{body.get('reason')!r}"
    )
    assert result.headers.get("X-Local-Fallback-Reason") == expected_reason
    assert body.get("retryable") is True, (
        f"{trigger}: a bounded local-capacity wait is retryable"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["queue_depth_cap", "queue_wait_expiry"])
async def test_deferral_exit_refuses_before_resolving_a_remote_candidate(
    mixed_model_config, trigger
):
    """The queue exit refuses *where it happens*, not after a remote lookup.

    Deliverable: the refusal is returned before any other candidate is
    resolved — so the chain is resolved exactly once (the local entry) and a
    remote provider is never even considered for a request that is being
    refused. A regression to the loop-head-only refusal would resolve twice.
    """
    resolutions: list = []
    real_resolve = provider._resolve_provider_with_exclusions

    def _spy(*args, **kwargs):
        resolved = real_resolve(*args, **kwargs)
        resolutions.append((resolved or {}).get("name"))
        return resolved

    result, _dispatches, _waits = await _issue_deferral_exit(
        mixed_model_config,
        trigger,
        extra_patches=[
            patch.object(provider, "_resolve_provider_with_exclusions", _spy)
        ],
    )

    assert result.status_code == 429, f"{trigger}: got {result.status_code}"
    assert resolutions == ["local-llama"], (
        f"{trigger}: the refusal must come from the queue exit itself, before "
        f"the next candidate is resolved (resolved={resolutions})"
    )
