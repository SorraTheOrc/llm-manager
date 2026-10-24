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
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import proxy.provider as provider
import pytest
from fastapi import Response

from proxy import contention_queue, prefill_eta


class _DummyRequest:
    """Minimal request stub (matches the contention-queue test helpers)."""

    def __init__(self, body: bytes = b'{"model":"test"}'):
        self._body = body
        self.headers = {}
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
