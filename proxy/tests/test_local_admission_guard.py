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
"""

from __future__ import annotations

import json
import logging
import math
import time
from contextlib import contextmanager
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
def _guard_patches(tokens: int, cached_ratio: float = 0.0):
    """Patch the estimate pipeline so the guard costs a known prompt size."""
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
# TDD red phase (the repo convention of ``test_provider_context_limit.py``):
# the two tests are marked ``xfail(strict=True)`` because neither builder is
# reason-aware yet. LP-0MVQ23API001UOET (reason-aware retry timing) and
# LP-0MVQ23AYQ000CGTW (token evidence) remove the markers, at which point a
# regression fails loudly instead of silently xfailing.
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
    assert body.get("new_tokens") == INCIDENT_NEW_TOKENS, (
        "the refusal must carry the token evidence the client compacts against"
    )
    assert body.get("budget_tokens") == INCIDENT_BUDGET_TOKENS
    assert "compact" in str(body.get("suggested_action") or "").lower(), (
        "the suggested action must name compaction as the remedy"
    )
    return body


@pytest.mark.xfail(
    strict=True,
    reason=(
        "red phase: refusal builders are not reason-aware yet — "
        "LP-0MVQ23API001UOET / LP-0MVQ23AYQ000CGTW turn this green"
    ),
)
@pytest.mark.asyncio
async def test_prefill_budget_exceeded_local_only_refusal_is_not_time_retryable(
    mixed_model_config,
):
    """Local-only client, remote still available: 429 without ``Retry-After``."""
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

    assert calls == [], "a local-only client must never be moved remote"
    body = _assert_prefill_budget_refusal_contract(result)
    assert body["reason"] == "prefill_budget_exceeded"
    assert result.headers.get("X-Local-Route") == "refused"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "red phase: refusal builders are not reason-aware yet — "
        "LP-0MVQ23API001UOET / LP-0MVQ23AYQ000CGTW turn this green"
    ),
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
# TDD red phase (the repo convention used by ``test_provider_context_limit.py``
# and this file's AC1 section): neither builder is reason-aware yet and neither
# emits ``retry_at`` nor the token evidence, so the cases are
# ``xfail(strict=True)``. LP-0MVQ23API001UOET (reason-aware retry timing) and
# LP-0MVQ23AYQ000CGTW (token evidence) remove the markers, at which point a
# regression fails loudly instead of silently xfailing.
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "red phase: neither refusal builder is reason-aware yet and neither "
        "emits retry_at or the token evidence — LP-0MVQ23API001UOET / "
        "LP-0MVQ23AYQ000CGTW turn this green"
    ),
)
@pytest.mark.parametrize("case", _OCCUPANCY_REFUSAL_CASES)
@pytest.mark.asyncio
async def test_occupancy_refusal_advertises_a_truthful_retry_time(
    case: _OccupancyRefusalCase, mixed_model_config
):
    """Each ETA/occupancy reason → Retry-After + retry_at + token evidence."""
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

    assert dispatched == [], (
        "a refused ETA/occupancy request must never dispatch or move remote"
    )
    body = _assert_occupancy_refusal_contract(result, case)
    assert body["code"] == provider.EXHAUSTION_CODE_LOCAL_ADMISSION
