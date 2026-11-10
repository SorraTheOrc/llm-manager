"""
F1: Contention-queue behavior suite (LP-0MSORQVK50012Q4D).

Pins all bounded cross-session contention-queue behaviors BEFORE the
production change lands (F2 config / F3 core / F4 metrics). Extends
``test_provider_fallback.py`` patterns (provider fallback specs) and
``test_mode_*.py`` (mode gating).

Covered acceptance criteria (F1 LP-0MSOZESW90057SRR):
- AC1: queue-on-contention in cheap mode dispatches local when a slot frees
  in time.
- AC2: fallback to the next remote provider after the wait cap
  (contention_queue_max_wait_seconds) is exceeded.
- AC3: fallback after the depth cap (contention_queue_max_depth) is exceeded.
- AC4: context_too_large / large_context_bypass are NEVER queued — they fall
  back exactly as today.
- AC5: fast mode (fallback policy) is byte-for-byte unchanged.
- AC6: wake fires on BOTH local_active_queries decrement AND
  slot-persistence / lease release.
- AC7: queued wait subtracts from the client-visible adaptive timeout budget
  (Q2=a).
- AC8: queue depth is logged on dispatch; no cumulative counters exposed.

Time mocking: the 60s wait cap is exercised with tiny caps (0.05-0.2s) or by
patching ``_get_contention_queue_config`` — never a real 60s sleep.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import httpx
import proxy.provider as provider
import pytest
from fastapi import Response

from proxy import contention_queue

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: The contention queue is a **pin-local** surface: since
#: LP-0MVUH9PUU005HVS4 the default (``remote-ok``) never queues, so a stub that
#: exercises a queue/admission-refusal path must send the pin-local value.
PIN_LOCAL_HEADERS = {"X-Local-Preference": "local-preferred"}


class _DummyRequest:
    """Minimal request stub for use in fallback tests."""
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


def _remote_passthrough_response() -> Response:
    """A distinctive remote response whose exact bytes must pass through to
    the client unchanged in fast mode (F1 AC5 byte-for-byte guarantee).

    Real UTF-8 (non-ASCII) bytes so byte-identity is non-trivial."""
    return Response(
        content='{"choices":[{"message":{"content":"é中文 bytes"}}]}'.encode(),
        status_code=200,
        media_type="application/json",
    )


class _MutableConcurrency:
    """Mutable stand-in for ``_get_local_concurrency_info``.

    Tests flip ``active`` between calls to simulate a slot freeing while a
    request is queued (or never freeing, to hit the caps).
    """

    def __init__(self, active: int = 1, max_: int = 1):
        self.active = active
        self.max = max_

    def __call__(self, config) -> tuple[int, int]:
        return (self.active, self.max)


class _FakeSrv:
    """Minimal server stand-in for wake-site tests (decrement / lease release)."""

    def __init__(self):
        self.local_active_queries = 1
        self.local_active_queries_lock = asyncio.Lock()
        self.local_dispatch_records_lock = asyncio.Lock()
        self.local_dispatch_records: dict = {}
        self.backend_signal_counts: dict = {}

        class _Logger:
            def info(self, *a, **k):
                pass

            def warning(self, *a, **k):
                pass

            def debug(self, *a, **k):
                pass

        self.logger = _Logger()
        self.config = {}


@pytest.fixture(autouse=True)
def reset_queue():
    """Reset the cross-session queue between tests."""
    contention_queue.reset()
    yield
    contention_queue.reset()


@pytest.fixture
def mixed_model_config():
    """A model config with both local and remote providers."""
    return {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {
                "name": "remote-fallback",
                "type": "remote",
                "endpoint": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
        ],
        "aliases": ["hybrid*"],
    }


def _queue_cfg(**overrides) -> dict:
    """Cheap-mode config with the contention-queue keys present."""
    cfg = {
        "provider_cooldown_seconds": 60,
        "server": {
            "session_slot_pool_size": 1,
            "contention_queue_policy": "queue",
            "contention_queue_max_wait_seconds": 60,
            "contention_queue_max_depth": 4,
        },
    }
    cfg["server"].update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# AC1: queue-on-contention dispatches local when a slot frees in time
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queue_on_contention_dispatches_local_when_slot_frees(
    mixed_model_config, caplog,
):
    """Cheap mode + queue policy: a request that finds slots busy QUEUES; when
    the slot frees within the caps it dispatches local. The dispatch log line
    carries queue depth + policy (F4 AC1)."""
    import logging

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    concurrency = _MutableConcurrency(active=1, max_=1)
    call_log = []

    async def _mock_proxy_to_local(_req, _path):
        call_log.append("local")
        return _ok_response()

    async def _mock_proxy_to_remote(_req, _path, _pc):
        call_log.append("remote")
        return _ok_response()

    request = _DummyRequest(headers=dict(PIN_LOCAL_HEADERS))
    cfg = _queue_cfg()

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
        patch("proxy.provider.request_priority", return_value="low"),
    ):
        task = asyncio.create_task(
            provider.proxy_with_fallback(
                request, "v1/chat/completions", mixed_model_config, cfg
            )
        )
        # Let the request enqueue behind the busy slot.
        for _ in range(200):
            if contention_queue.queue_depth() > 0:
                break
            await asyncio.sleep(0.005)
        assert contention_queue.queue_depth() == 1, "request should be queued"

        # Slot frees in time → wake the queue.
        concurrency.active = 0
        await contention_queue.wake_all()

        result = await asyncio.wait_for(task, timeout=5)

    assert result.status_code == 200
    assert call_log == ["local"], "queued request must dispatch local"

    # F1 AC1: the dispatch log line carries queue depth + policy.
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "contention_queue_dispatch" in messages
    assert "policy=queue" in messages
    assert "depth=" in messages


# ---------------------------------------------------------------------------
# AC2: fallback to next remote provider after max_wait exceeded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_after_max_wait_exceeded(mixed_model_config, caplog):
    """When the wait cap is exceeded (no slot frees) the queue reports a
    fallback and the event records the elapsed wait time (F4 AC2).

    The *client* that waited is pin-local, so it is never silently moved to a
    paid provider: the fallback becomes the retryable ``429`` deferral
    (LP-0MVUH9PUU005HVS4 — the non-pinned default never queues at all, so it
    cannot reach this exit).
    """
    import logging

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    concurrency = _MutableConcurrency(active=1, max_=1)
    call_log = []

    async def _mock_proxy_to_local(_req, _path):
        call_log.append("local")
        return _ok_response()

    async def _mock_proxy_to_remote(_req, _path, _pc):
        call_log.append("remote")
        return _ok_response()

    request = _DummyRequest(headers=dict(PIN_LOCAL_HEADERS))
    cfg = _queue_cfg()

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
        patch("proxy.provider.request_priority", return_value="low"),
        # Tiny cap: 60s real wait would be far too slow for the suite.
        patch(
            "proxy.router._get_contention_queue_config",
            return_value={"policy": "queue", "max_wait_seconds": 0.05, "max_depth": 4},
        ),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 429, (
        "a pin-local client that spent its wait budget gets the retryable "
        f"deferral, never a paid dispatch; got {result.status_code}"
    )
    assert call_log == [], "the pin must not be broken by a queue expiry"
    assert result.headers.get("X-Local-Route") == "refused"
    assert result.headers.get("X-Local-Fallback-Reason") == "fallback_after_queue"

    # F1 AC2: the fallback-after-queue log line carries the elapsed wait.
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "contention_queue_fallback_after_queue" in messages
    assert "queued_duration=" in messages


# ---------------------------------------------------------------------------
# AC3: fallback after max_depth exceeded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queue_module_depth_cap_returns_none():
    """Queue module: when max_depth waiters already queued, a new request is
    NOT enqueued — it returns None immediately (fallback signal)."""
    held = asyncio.Event()

    def _slot_free() -> bool:
        return False  # slot never frees in this test

    # First waiter fills the single depth slot.
    first = asyncio.create_task(
        contention_queue.wait_for_local_slot(5.0, max_depth=1, slot_free_check=_slot_free)
    )
    await asyncio.sleep(0.02)
    assert contention_queue.queue_depth() == 1

    # Second request: depth cap exceeded → immediate fallback (None).
    result = await contention_queue.wait_for_local_slot(
        5.0, max_depth=1, slot_free_check=_slot_free
    )
    assert result is None
    assert contention_queue.queue_depth() == 1, "second request must not enqueue"

    # Cleanup: cancel the held waiter.
    held.set()
    first.cancel()
    try:
        await first
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_fallback_after_max_depth_exceeded_integration(mixed_model_config):
    """Provider level: with max_depth=1, a second concurrent request is turned
    away immediately while the first is queued.

    The turned-away request is pin-local, so the depth-cap fallback surfaces as
    the retryable ``429`` deferral (never a paid dispatch).
    """
    concurrency = _MutableConcurrency(active=1, max_=1)
    call_log = []

    async def _mock_proxy_to_local(_req, _path):
        call_log.append("local")
        return _ok_response()

    async def _mock_proxy_to_remote(_req, _path, _pc):
        call_log.append("remote")
        return _ok_response()

    cfg = _queue_cfg(contention_queue_max_depth=1)

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
        patch("proxy.provider.request_priority", return_value="low"),
        patch(
            "proxy.router._get_contention_queue_config",
            return_value={"policy": "queue", "max_wait_seconds": 0.2, "max_depth": 1},
        ),
    ):
        first = asyncio.create_task(
            provider.proxy_with_fallback(
                _DummyRequest(headers=dict(PIN_LOCAL_HEADERS)), "v1/chat/completions", mixed_model_config, cfg
            )
        )
        # Let the first request occupy the single depth slot.
        for _ in range(200):
            if contention_queue.queue_depth() > 0:
                break
            await asyncio.sleep(0.005)
        assert contention_queue.queue_depth() == 1

        # Second request: depth exceeded → immediate remote fallback.
        result2 = await provider.proxy_with_fallback(
            _DummyRequest(headers=dict(PIN_LOCAL_HEADERS)), "v1/chat/completions", mixed_model_config, cfg
        )
        assert result2.status_code == 429, (
            "the depth cap turns a pinned request away with the retryable "
            f"deferral; got {result2.status_code}"
        )
        assert result2.headers.get("X-Local-Fallback-Reason") == (
            "fallback_after_queue"
        )
        assert "remote" not in call_log, "the pin must not be broken at the depth cap"
        assert "local" not in call_log, "the second request never got a slot"

        # Free the slot → the queued first request dispatches local.
        concurrency.active = 0
        await contention_queue.wake_all()
        result1 = await asyncio.wait_for(first, timeout=5)
        assert result1.status_code == 200
        assert "local" in call_log


# ---------------------------------------------------------------------------
# AC4: context bypasses never queued
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_bypass_never_queued(mixed_model_config):
    """A physically-oversized request (context_too_large) must NEVER wait in
    the contention queue — it falls back exactly as today in every mode
    (LP-0MSORQVK50012Q4D AC4, LP-0MU5A4QBR003YJM0 AC3)."""
    concurrency = _MutableConcurrency(active=1, max_=1)
    call_log = []

    async def _mock_proxy_to_local(_req, _path):
        call_log.append("local")
        return _ok_response()

    async def _mock_proxy_to_remote(_req, _path, _pc):
        call_log.append("remote")
        return _ok_response()

    # Large body → estimated tokens exceed the tiny warm threshold.
    big_body = json.dumps(
        {"model": "test", "messages": [{"role": "user", "content": "x" * 20_000}]}
    ).encode()
    request = _DummyRequest(body=big_body)

    cfg = _queue_cfg(
        local_large_context_cold_cache_threshold=100,
        local_large_context_warm_cache_threshold=200,
        local_large_context_economic_bypass_serves_local=True,
    )

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert call_log == ["remote"], "context bypass must fall back to remote"
    assert contention_queue.queue_depth() == 0


# ---------------------------------------------------------------------------
# AC1/AC2: config-driven economic bypass recovery (LP-0MU5A4QBR003YJM0)
# ---------------------------------------------------------------------------


def _economic_bypass_body() -> bytes:
    """A body whose estimate sits between a small cold threshold and a large
    warm threshold — an economic (not physical) bypass."""
    phrase = "test message content for token estimation "
    return json.dumps(
        {"model": "test", "messages": [{"role": "user", "content": phrase * 2000}]}
    ).encode()


@pytest.mark.asyncio
async def test_cheap_mode_economic_bypass_serves_local(mixed_model_config, caplog):
    """AC1 (LP-0MU5A4QBR003YJM0): with the cheap profile's recovery flag an
    economic bypass (new_tokens > cold, total <= warm) is served locally — no
    remote attempt.  A distinct ``routing_economic_bypass_local`` log line
    records the recovery (AC5).
    """
    import logging

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    # Slots available → local dispatch directly.
    concurrency = _MutableConcurrency(active=0, max_=1)
    call_log = []

    async def _mock_proxy_to_local(_req, _path):
        call_log.append("local")
        return _ok_response()

    async def _mock_proxy_to_remote(_req, _path, _pc):
        call_log.append("remote")
        return _ok_response()

    request = _DummyRequest(body=_economic_bypass_body())
    cfg = _queue_cfg(
        local_large_context_cold_cache_threshold=100,
        local_large_context_warm_cache_threshold=100_000,
        # Isolate the economic band. The synthetic request has no residency
        # measurement, which the residency ladder now treats as cold
        # (``unmeasured_residency_cold``, LP-0MW81DFML006O5KL) and would deny
        # before the economic rung is reached; ``0`` is the documented rollback
        # that restores the predicted-ratio decision this test pins
        # (``proxy/tests/test_local_resident_prefix_admission.py`` covers the
        # ladder and its interplay with the queue).
        local_resident_prefix_min_ratio=0,
        local_large_context_economic_bypass_serves_local=True,
    )

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert call_log == ["local"], (
        "cheap mode must serve the economic bypass locally (no remote)"
    )
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "routing_economic_bypass_local" in messages
    assert "routing_skip_local" not in messages


@pytest.mark.asyncio
async def test_cheap_mode_economic_bypass_queues_when_busy(mixed_model_config):
    """AC1 (LP-0MU5A4QBR003YJM0): with the recovery flag an economic bypass
    enters the existing contention queue when all slots are busy and
    dispatches local when a slot frees — it never falls back to a paid
    remote."""
    concurrency = _MutableConcurrency(active=1, max_=1)
    call_log = []

    async def _mock_proxy_to_local(_req, _path):
        call_log.append("local")
        return _ok_response()

    async def _mock_proxy_to_remote(_req, _path, _pc):
        call_log.append("remote")
        return _ok_response()

    request = _DummyRequest(headers=dict(PIN_LOCAL_HEADERS), body=_economic_bypass_body())
    cfg = _queue_cfg(
        local_large_context_cold_cache_threshold=100,
        local_large_context_warm_cache_threshold=100_000,
        # Isolate the economic band. The synthetic request has no residency
        # measurement, which the residency ladder now treats as cold
        # (``unmeasured_residency_cold``, LP-0MW81DFML006O5KL) and would deny
        # before the economic rung is reached; ``0`` is the documented rollback
        # that restores the predicted-ratio decision this test pins
        # (``proxy/tests/test_local_resident_prefix_admission.py`` covers the
        # ladder and its interplay with the queue).
        local_resident_prefix_min_ratio=0,
        local_large_context_economic_bypass_serves_local=True,
    )

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
    ):
        task = asyncio.create_task(
            provider.proxy_with_fallback(
                request, "v1/chat/completions", mixed_model_config, cfg
            )
        )
        for _ in range(200):
            if contention_queue.queue_depth() > 0:
                break
            await asyncio.sleep(0.005)
        assert contention_queue.queue_depth() == 1, (
            "economic bypass must queue behind the busy slot in cheap mode"
        )

        concurrency.active = 0
        await contention_queue.wake_all()
        result = await asyncio.wait_for(task, timeout=5)

    assert result.status_code == 200
    assert call_log == ["local"], (
        "queued economic bypass must dispatch local, not remote"
    )


@pytest.mark.asyncio
async def test_fast_mode_economic_bypass_falls_back_remote(
    mixed_model_config, caplog
):
    """AC2 (LP-0MU5A4QBR003YJM0): without the recovery flag (fast/default
    profile) the same economic bypass still skips local
    (``large_context_bypass``) and falls back to the next remote provider —
    no local attempt, no queue."""
    import logging

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    concurrency = _MutableConcurrency(active=0, max_=1)
    call_log = []

    async def _mock_proxy_to_local(_req, _path):
        call_log.append("local")
        return _ok_response()

    async def _mock_proxy_to_remote(_req, _path, _pc):
        call_log.append("remote")
        return _ok_response()

    request = _DummyRequest(body=_economic_bypass_body())
    cfg = _queue_cfg(
        local_large_context_cold_cache_threshold=100,
        local_large_context_warm_cache_threshold=100_000,
        # See ``test_cheap_mode_economic_bypass_serves_local``: the residency
        # ladder is disabled so the economic rung is the one under test.
        local_resident_prefix_min_ratio=0,
    )

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
        # Fast-mode semantics = high priority (never recovers the economic
        # bypass locally); pinned so the test is deterministic inside the
        # cheap window (01:00–10:00), where an unpinned request would cap
        # to low and serve the bypass locally instead.
        patch("proxy.provider.request_priority", return_value="high"),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert call_log == ["remote"], (
        "fast mode must keep the economic bypass remote fallback"
    )
    assert contention_queue.queue_depth() == 0
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "routing_skip_local" in messages
    assert "reason=large_context_bypass" in messages
    assert "routing_economic_bypass_local" not in messages


# ---------------------------------------------------------------------------
# AC5: fast mode byte-for-byte unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fast_mode_fallback_policy_unchanged(mixed_model_config):
    """Fast mode (contention_queue_policy: fallback / absent) keeps today's
    immediate fallback: no queueing, no queue metrics, no queue logs."""
    concurrency = _MutableConcurrency(active=1, max_=1)
    call_log = []

    async def _mock_proxy_to_local(_req, _path):
        call_log.append("local")
        return _ok_response()

    async def _mock_proxy_to_remote(_req, _path, _pc):
        call_log.append("remote")
        return _remote_passthrough_response()

    request = _DummyRequest()
    cfg = {
        "provider_cooldown_seconds": 60,
        "server": {
            "session_slot_pool_size": 1,
            "contention_queue_policy": "fallback",
        },
    }

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
        patch("proxy.provider.request_priority", return_value="high"),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert call_log == ["remote"], "fast mode falls back immediately, no queue"
    assert contention_queue.queue_depth() == 0
    # F1 AC5 byte-for-byte: the client-visible response is byte-identical to
    # what the remote provider returned (no queue layer mutates the payload).
    assert result.body == _remote_passthrough_response().body, (
        "fast-mode fallback response must reach the client byte-for-byte"
    )


@pytest.mark.asyncio
async def test_fast_mode_fallback_dispatch_bytes_unchanged(mixed_model_config):
    """AC5 literal wire check: in fast mode the contention fallback forwards
    the ORIGINAL request body bytes and headers to the remote provider and
    returns the remote response body byte-for-byte — the queue feature must
    not mutate the wire bytes on the fast-mode (fallback policy) path."""
    concurrency = _MutableConcurrency(active=1, max_=1)
    call_log = []

    # Non-ASCII payload so byte-identity is non-trivial (UTF-8 encoding).
    request_body = (
        '{"model":"test","messages":[{"role":"user","content":"é中文 payload"}]}'
    ).encode()
    request = _DummyRequest(body=request_body)
    request.headers = {"x-test-header": "abc", "content-type": "application/json"}
    remote_response = _remote_passthrough_response()

    forwarded_body: bytes | None = None
    forwarded_headers: dict | None = None

    async def _mock_proxy_to_remote(_req, _path, _pc):
        nonlocal forwarded_body, forwarded_headers
        call_log.append("remote")
        forwarded_body = await _req.body()
        forwarded_headers = dict(_req.headers)
        return remote_response

    cfg = {
        "provider_cooldown_seconds": 60,
        "server": {
            "session_slot_pool_size": 1,
            "contention_queue_policy": "fallback",
        },
    }

    with (
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
        patch("proxy.provider.request_priority", return_value="high"),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert call_log == ["remote"], "fast mode must fall back to remote immediately"
    # The exact request bytes reach the remote provider untouched.
    assert forwarded_body == request_body, (
        "fallback dispatch must forward the original request body byte-for-byte, "
        f"got {forwarded_body!r}"
    )
    # The original request headers are forwarded unchanged (no rewrite).
    assert forwarded_headers == request.headers, (
        "fallback dispatch must forward the original request headers, "
        f"got {forwarded_headers!r}"
    )
    # The client receives the remote response body byte-for-byte.
    assert result.body == remote_response.body, (
        "client must receive the remote response body unchanged, "
        f"got {result.body!r}"
    )
    # No queue involvement in fast mode.
    assert contention_queue.queue_depth() == 0


@pytest.mark.asyncio
async def test_absent_contention_keys_default_to_fallback(mixed_model_config):
    """Absent contention_queue_* keys → fallback (backward compatible)."""
    concurrency = _MutableConcurrency(active=1, max_=1)
    call_log = []

    async def _mock_proxy_to_remote(_req, _path, _pc):
        call_log.append("remote")
        return _ok_response()

    cfg = {"provider_cooldown_seconds": 60, "server": {"session_slot_pool_size": 1}}

    with (
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
        patch("proxy.provider.request_priority", return_value="low"),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert call_log == ["remote"]
    assert contention_queue.queue_depth() == 0


# ---------------------------------------------------------------------------
# AC6: wake fires on BOTH local_active_queries decrement AND lease release
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wake_fires_on_local_active_queries_decrement():
    """``_decrement_local_active_queries`` (stream end) wakes a queued waiter."""
    from proxy.router_helpers import _decrement_local_active_queries

    srv = _FakeSrv()
    srv.local_active_queries = 1

    # Waiter waits for the fake counter to drop below max.
    waiter = asyncio.create_task(
        contention_queue.wait_for_local_slot(
            5.0,
            max_depth=4,
            slot_free_check=lambda: srv.local_active_queries < 1,
        )
    )
    await asyncio.sleep(0.02)
    assert contention_queue.queue_depth() == 1

    # Stream end → decrement → must wake the queue.
    await _decrement_local_active_queries(srv, session_key="s1")
    elapsed = await asyncio.wait_for(waiter, timeout=2)
    assert elapsed is not None
    assert srv.local_active_queries == 0


@pytest.mark.asyncio
async def test_wake_fires_on_lease_release():
    """``_release_local_dispatch`` (slot persistence / explicit release)
    wakes a queued waiter."""
    from proxy.router_helpers import _release_local_dispatch

    srv = _FakeSrv()
    srv.local_dispatch_records["s1"] = {
        "backend": "local",
        "started_at": time.monotonic(),
        "active": True,
        "expires_at": time.monotonic() + 300,
    }

    waiter = asyncio.create_task(
        contention_queue.wait_for_local_slot(
            5.0,
            max_depth=4,
            slot_free_check=lambda: "s1" not in srv.local_dispatch_records,
        )
    )
    await asyncio.sleep(0.02)
    assert contention_queue.queue_depth() == 1

    # Lease release → must wake the queue.
    removed = await _release_local_dispatch(srv, "s1")
    assert removed is True
    elapsed = await asyncio.wait_for(waiter, timeout=2)
    assert elapsed is not None


# ---------------------------------------------------------------------------
# AC7: queued wait subtracts from the client-visible adaptive timeout budget
# ---------------------------------------------------------------------------


def test_apply_queue_wait_to_timeout():
    """Q2=a: the queued wait shrinks the client-visible adaptive timeout."""
    from proxy.router_helpers import _apply_queue_wait_to_timeout

    base = httpx.Timeout(60.0)
    reduced = _apply_queue_wait_to_timeout(base, 10.0)
    assert reduced.connect == pytest.approx(50.0), (
        f"60s budget minus 10s queue wait should leave 50s, got {reduced.connect}"
    )

    # Never below a minimal serve floor.
    floored = _apply_queue_wait_to_timeout(httpx.Timeout(2.0), 10.0)
    assert floored.connect >= 1.0

    # Zero wait is a no-op.
    unchanged = _apply_queue_wait_to_timeout(httpx.Timeout(60.0), 0.0)
    assert unchanged.connect == pytest.approx(60.0)


@pytest.mark.asyncio
async def test_queued_dispatch_marks_request_budget(mixed_model_config):
    """After a queued dispatch, the request carries the elapsed wait so
    proxy_to_local can shrink the adaptive timeout (Q2=a)."""
    concurrency = _MutableConcurrency(active=1, max_=1)
    captured = {}

    async def _mock_proxy_to_local(req, _path):
        captured["wait_seconds"] = getattr(req, "_contention_queue_wait_seconds", None)
        return _ok_response()

    async def _mock_proxy_to_remote(_req, _path, _pc):
        return _ok_response()

    cfg = _queue_cfg()

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
        patch("proxy.provider.request_priority", return_value="low"),
    ):
        request = _DummyRequest(headers=dict(PIN_LOCAL_HEADERS))
        task = asyncio.create_task(
            provider.proxy_with_fallback(
                request, "v1/chat/completions", mixed_model_config, cfg
            )
        )
        for _ in range(200):
            if contention_queue.queue_depth() > 0:
                break
            await asyncio.sleep(0.005)
        concurrency.active = 0
        await contention_queue.wake_all()
        result = await asyncio.wait_for(task, timeout=5)

    assert result.status_code == 200
    assert captured.get("wait_seconds") is not None
    assert captured["wait_seconds"] >= 0.0


# ---------------------------------------------------------------------------
# AC8: metrics emitted when policy is queue; not when fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queue_log_lines_emitted_for_queue_policy(mixed_model_config, caplog):
    """Dispatch / fallback log lines carry queue context in queue policy.

    A pin-local client that spends its queue budget is refused, so the
    fallback-after-queue line is the *log* evidence of the expiry while no
    provider is dispatched.
    """
    import logging

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    concurrency = _MutableConcurrency(active=1, max_=1)
    call_log = []

    async def _mock_proxy_to_remote(_req, _path, _pc):
        call_log.append("remote")
        return _ok_response()

    cfg = _queue_cfg()

    with (
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
        patch("proxy.provider.request_priority", return_value="low"),
        patch(
            "proxy.router._get_contention_queue_config",
            return_value={"policy": "queue", "max_wait_seconds": 0.05, "max_depth": 4},
        ),
    ):
        await provider.proxy_with_fallback(
            _DummyRequest(headers=dict(PIN_LOCAL_HEADERS)), "v1/chat/completions", mixed_model_config, cfg
        )

    assert call_log == []
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "contention_queue_fallback_after_queue" in messages


@pytest.mark.asyncio
async def test_no_queue_log_lines_for_fallback_policy(mixed_model_config, caplog):
    """Fast mode (fallback policy) emits no contention-queue log lines."""
    import logging

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    concurrency = _MutableConcurrency(active=1, max_=1)
    call_log = []

    async def _mock_proxy_to_remote(_req, _path, _pc):
        call_log.append("remote")
        return _ok_response()

    cfg = {
        "provider_cooldown_seconds": 60,
        "server": {"session_slot_pool_size": 1, "contention_queue_policy": "fallback"},
    }

    with (
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
        patch("proxy.provider.request_priority", return_value="high"),
    ):
        await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", mixed_model_config, cfg
        )

    assert call_log == ["remote"]
    for record in caplog.records:
        assert "contention_queue" not in record.getMessage(), (
            f"fast mode must not emit queue logs: {record.getMessage()}"
        )


# ---------------------------------------------------------------------------
# Config parsing (F2 interface): keys, defaults, clamps
# ---------------------------------------------------------------------------


def test_contention_queue_config_defaults_to_fallback():
    """Absent keys → fallback policy (backward compatible)."""
    from proxy.router import _get_contention_queue_config

    cfg = _get_contention_queue_config({})
    assert cfg["policy"] == "fallback"
    assert cfg["max_wait_seconds"] == 60
    assert cfg["max_depth"] == 4


def test_contention_queue_config_clamps():
    """Sane clamps: wait in [1, max_runtime], depth in [1, 16]; invalid values
    are clamped, never crash."""
    from proxy.router import _get_contention_queue_config

    # Wait below the floor → clamped to 1s.
    cfg = _get_contention_queue_config(
        {
            "contention_queue_policy": "queue",
            "contention_queue_max_wait_seconds": 0,
            "contention_queue_max_depth": 0,
        }
    )
    assert cfg["max_wait_seconds"] == 1
    assert cfg["max_depth"] == 1

    # Depth above the cap → clamped to 16.
    cfg = _get_contention_queue_config(
        {
            "contention_queue_policy": "queue",
            "contention_queue_max_wait_seconds": 99999,
            "contention_queue_max_depth": 999,
        }
    )
    assert cfg["max_wait_seconds"] <= 1800  # session_guardrail_max_runtime_seconds
    assert cfg["max_depth"] == 16

    # Garbage values never crash.
    cfg = _get_contention_queue_config(
        {
            "contention_queue_policy": "bogus",
            "contention_queue_max_wait_seconds": "abc",
            "contention_queue_max_depth": "xyz",
        }
    )
    assert cfg["policy"] == "fallback"
    assert cfg["max_wait_seconds"] == 60
    assert cfg["max_depth"] == 4


def test_contention_queue_config_logs_invalid_values(caplog):
    """Invalid contention values are logged (F2 AC3): the resolved policy is
    coerced and caps are clamped with a warning, never silently."""
    import logging

    from proxy.router import _get_contention_queue_config

    caplog.set_level(logging.WARNING, logger="llama-proxy.router")
    _get_contention_queue_config(
        {
            "contention_queue_policy": "bogus",
            "contention_queue_max_wait_seconds": "abc",
            "contention_queue_max_depth": "xyz",
        }
    )
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "Invalid contention_queue_policy" in messages
    assert "Invalid contention_queue_max_wait_seconds" in messages
    assert "Invalid contention_queue_max_depth" in messages

    # Clamped values are also logged.
    caplog.clear()
    _get_contention_queue_config(
        {
            "contention_queue_policy": "queue",
            "contention_queue_max_wait_seconds": 99999,
            "contention_queue_max_depth": 999,
        }
    )
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "clamped" in messages


def test_cheap_config_declares_queue_policy():
    """The merged cheap profile declares queue policy + the swept caps (F2 AC1,
    caps tuned 60→120 / 4→8 per LP-0MTF6EVLW007PEHN; halved 120→60 in
    LP-0MVIQ0EXV00673HO; swept 60→30 / 8→2 in LP-0MVT1146V007WXA4)."""
    from config_test_utils import get_merged_config

    server = get_merged_config("cheap")["server"]
    assert server["contention_queue_policy"] == "queue"
    assert server["contention_queue_max_wait_seconds"] == 30
    assert server["contention_queue_max_depth"] == 2


def test_fast_config_declares_queue_policy():
    """The merged fast profile declares queue policy + the same swept caps as
    cheap (depth 2, wait 30s). LP-0MV881O5C007PSL4 raised fast to cheap caps
    (wait 120s); LP-0MVIQ0EXV00673HO halved the unified cap to 60s;
    LP-0MVT1146V007WXA4 swept both to 30s / depth 2 (straggler net).

    Fast mode originally used a smaller queue (depth 3 vs 8, wait 45s vs 120s,
    LP-0MTQYIK4Z008XF2V) so burst traffic spilled to remotes sooner during
    peak hours.  LP-0MV881O5C007PSL4 raised fast to the cheap caps so a burst
    of local-eligible requests waits for a slot instead of spilling while the
    average local stream (~31s) is still running.
    """
    from config_test_utils import get_merged_config

    server = get_merged_config("fast")["server"]
    assert server["contention_queue_policy"] == "queue"
    assert server["contention_queue_max_wait_seconds"] == 30
    assert server["contention_queue_max_depth"] == 2


def test_fast_queue_caps_match_cheap():
    """Fast and cheap share identical queue caps (LP-0MV881O5C007PSL4)."""
    from config_test_utils import get_merged_config

    cheap = get_merged_config("cheap")["server"]
    fast = get_merged_config("fast")["server"]

    assert fast["contention_queue_max_depth"] == cheap["contention_queue_max_depth"], (
        f"fast depth {fast['contention_queue_max_depth']} must equal cheap depth "
        f"{cheap['contention_queue_max_depth']}"
    )
    assert fast["contention_queue_max_wait_seconds"] == cheap["contention_queue_max_wait_seconds"], (
        f"fast wait {fast['contention_queue_max_wait_seconds']} must equal cheap wait "
        f"{cheap['contention_queue_max_wait_seconds']}"
    )


# ---------------------------------------------------------------------------
# Sweep 60 s / depth 8 → 30 s / depth 2 (LP-0MVT1146V007WXA4)
#
# The swept values must reach the queue through the one resolver, and both
# must behave as a straggler net: depth 2 spills the third waiter immediately
# and a parked waiter gives up at 30 s. Never a real 30 s sleep below.
# ---------------------------------------------------------------------------


def _shipped_server_block() -> dict:
    """The shipped ``proxy/config.yaml`` server block (single source of truth)."""
    from config_test_utils import get_merged_config

    return get_merged_config("fast")["server"]


def _shipped_runtime_config() -> dict:
    """Runtime config whose queue caps come from the shipped file.

    ``proxy_with_fallback`` resolves the caps through
    ``_get_contention_queue_config(config)``, so passing the real server block
    exercises the shipped values end to end (no patched resolver).
    """
    return {"provider_cooldown_seconds": 60, "server": dict(_shipped_server_block())}


def test_shipped_sweep_reaches_one_resolver_and_both_caps():
    """AC2: one resolver feeds the queue, the admission guard and ``/reserve``.

    The shipped ``contention_queue_max_wait_seconds`` (30) and
    ``contention_queue_max_depth`` (2) are read through
    ``_get_contention_queue_config``; ``wait_budget_seconds`` is that same
    number on both local paths, and ``high`` priority (which never queues)
    resolves to 0.
    """
    from proxy.router import _get_contention_queue_config

    from proxy import prefill_eta

    server = _shipped_server_block()
    queue_cfg = _get_contention_queue_config(server)
    resolved = {"server": server}

    assert queue_cfg["max_wait_seconds"] == 30.0
    # Depth 2 is inside the [1, 16] clamp: accepted unchanged, not coerced.
    assert queue_cfg["max_depth"] == 2
    assert prefill_eta.wait_budget_seconds(resolved, "medium") == 30.0
    assert prefill_eta.wait_budget_seconds(resolved, "low") == 30.0
    assert prefill_eta.wait_budget_seconds(resolved, "high") == 0.0


@pytest.mark.asyncio
async def test_shipped_sweep_caps_reach_the_provider_queue_wait(
    mixed_model_config,
):
    """AC2 end to end: the provider hands the shipped 30 s / depth 2 to the
    queue wait (the resolver is NOT patched in this test).

    The spy reports "caps exceeded", which for the pin-local client that waited
    is the retryable ``429`` deferral rather than a paid dispatch.
    """
    calls: list[dict] = []
    call_log: list[str] = []

    async def _spy_wait(**kwargs):
        calls.append(kwargs)
        return None  # caps exceeded → the caller falls back remote

    async def _mock_proxy_to_local(_req, _path):
        call_log.append("local")
        return _ok_response()

    async def _mock_proxy_to_remote(_req, _path, _pc):
        call_log.append("remote")
        return _ok_response()

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(1, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        # ``provider.py`` imports the queue lazily (``from proxy import
        # contention_queue``), so patching the module attribute is what the
        # wait site resolves at call time.
        patch("proxy.contention_queue.wait_for_local_slot", _spy_wait),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(headers=dict(PIN_LOCAL_HEADERS)),
            "v1/chat/completions",
            mixed_model_config,
            _shipped_runtime_config(),
        )

    assert result.status_code == 429
    assert call_log == []
    assert [(c["max_wait_seconds"], c["max_depth"]) for c in calls] == [(30.0, 2)]


@pytest.mark.asyncio
async def test_swept_depth_two_third_waiter_falls_back_without_enqueueing():
    """AC4: depth 2 is a straggler net — with two waiters already parked a
    third request falls back immediately instead of enqueueing."""
    from proxy.router import _get_contention_queue_config

    queue_cfg = _get_contention_queue_config(_shipped_server_block())
    assert queue_cfg["max_depth"] == 2

    parked = [
        asyncio.create_task(
            contention_queue.wait_for_local_slot(
                queue_cfg["max_wait_seconds"],
                queue_cfg["max_depth"],
                slot_free_check=lambda: False,
            )
        )
        for _ in range(2)
    ]
    try:
        for _ in range(200):
            if contention_queue.queue_depth() == 2:
                break
            await asyncio.sleep(0.005)
        assert contention_queue.queue_depth() == 2

        third = await contention_queue.wait_for_local_slot(
            queue_cfg["max_wait_seconds"],
            queue_cfg["max_depth"],
            slot_free_check=lambda: False,
        )

        assert third is None, "the third waiter must not park (depth cap 2)"
        assert contention_queue.queue_depth() == 2
    finally:
        for task in parked:
            task.cancel()
        await asyncio.gather(*parked, return_exceptions=True)


@pytest.mark.asyncio
async def test_swept_budget_expires_at_thirty_seconds(monkeypatch):
    """AC4: a parked waiter that is never served falls back at the shipped
    30 s budget.

    The clock and the wait primitive are faked, so the assertion is on the
    *budget handed to the wait* (30.0 s, resolved from the shipped config) —
    never a real 30 s sleep.
    """
    from types import SimpleNamespace

    from proxy.router import _get_contention_queue_config

    queue_cfg = _get_contention_queue_config(_shipped_server_block())
    assert queue_cfg["max_wait_seconds"] == 30.0

    clock = {"now": 1_000.0}
    timeouts: list[float] = []

    async def _fake_wait_for(awaitable, timeout):
        timeouts.append(timeout)
        awaitable.close()
        clock["now"] += timeout  # the wait consumed the whole budget
        raise TimeoutError

    monkeypatch.setattr(
        contention_queue,
        "asyncio",
        SimpleNamespace(
            Condition=asyncio.Condition,
            Event=asyncio.Event,
            wait_for=_fake_wait_for,
        ),
    )
    monkeypatch.setattr(contention_queue.time, "monotonic", lambda: clock["now"])

    result = await contention_queue.wait_for_local_slot(
        queue_cfg["max_wait_seconds"],
        queue_cfg["max_depth"],
        slot_free_check=lambda: False,
    )

    assert result is None
    assert timeouts == [30.0], "the wait must be bounded by the shipped 30 s"
    assert contention_queue.queue_depth() == 0


# ===================================================================
# Post-dispatch local-lease denial → queue before remote fallback
# (LP-0MU5BOMJM008F17C)
# ===================================================================


def _lease_active_response() -> Response:
    """The synthetic 503 returned when the local dispatch lease is held by
    another session (router: server_busy / reason=local_lease_active)."""
    return Response(
        content=json.dumps({
            "error": {
                "code": "no_slots_available",
                "type": "server_busy",
                "message": "Local dispatch lease is held by another session",
            },
            "total_slots": 1,
            "available_slots": 0,
            "reason": "local_lease_active",
        }).encode("utf-8"),
        status_code=503,
        media_type="application/json",
    )


def _free_concurrency() -> "_MutableConcurrency":
    """Concurrency stub reporting a free slot so the pre-dispatch queue path
    is not taken (the denial is a post-dispatch lease race)."""
    return _MutableConcurrency(active=0, max_=1)


@pytest.fixture
def lease_state():
    """Isolate cooldown/quarantine state for the lease-denial tests."""
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    yield
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()


@pytest.mark.asyncio
async def test_lease_denied_queue_admits_and_redispatches_local(lease_state, caplog):
    """AC1/AC2: a post-dispatch lease denial queues for the protected local
    slot; when it frees within the budget the request is served locally."""
    import logging

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    config = {
        "providers": [{"name": "local-llama", "type": "local", "llama_model": "Qwen3"}]
    }
    local_calls = []

    async def _mock_proxy_to_local(_req, _path):
        local_calls.append(1)
        if len(local_calls) == 1:
            return _lease_active_response()
        return _ok_response()

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.provider._get_local_concurrency_info", _free_concurrency()),
        patch("proxy.provider.request_priority", return_value="low"),
        patch(
            "proxy.provider._maybe_queue_for_local_slot",
            new=AsyncMock(return_value=("dispatch", None, 0.5)),
        ),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", config, _queue_cfg()
        )

    assert result.status_code == 200
    assert len(local_calls) == 2, "lease denial must re-dispatch local after the wait"
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "contention_queue_dispatch" in messages
    assert "reason=local_lease_active" in messages
    assert "queued_duration=0.50s" in messages


@pytest.mark.asyncio
async def test_lease_denied_queue_budget_expires_falls_back_to_remote(lease_state, caplog):
    """AC1: when the queue budget expires, the request falls back to an
    available remote provider (normal fallback, not a silent hold)."""
    import logging

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {"name": "remote-ok", "type": "remote",
             "endpoint": "https://api.example.com/v1"},
        ]
    }
    remote_calls = []

    async def _mock_proxy_to_local(_req, _path):
        return _lease_active_response()

    async def _mock_proxy_to_remote(_req, _path, _pc):
        remote_calls.append(1)
        return _ok_response()

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", _free_concurrency()),
        patch("proxy.provider.request_priority", return_value="low"),
        patch(
            "proxy.provider._maybe_queue_for_local_slot",
            new=AsyncMock(return_value=("fallback", None, 300.0)),
        ),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", config, _queue_cfg()
        )

    assert result.status_code == 200
    assert remote_calls == [1], "remote fallback must run after the queue budget expires"
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "contention_queue_fallback_after_queue" in messages
    assert "reason=local_lease_active" in messages


@pytest.mark.asyncio
async def test_lease_denied_dispatch_log_carries_own_new_tokens(lease_state, caplog):
    """AC2: the dispatch line that ends a queue wait carries the wait's own
    ``new_tokens`` (stamped at enqueue), never a join to another request."""
    import logging

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    config = {
        "providers": [{"name": "local-llama", "type": "local", "llama_model": "Qwen3"}]
    }
    local_calls = []

    async def _mock_proxy_to_local(_req, _path):
        local_calls.append(1)
        if len(local_calls) == 1:
            return _lease_active_response()
        return _ok_response()

    request = _DummyRequest()
    setattr(request, provider.CONTENTION_QUEUE_NEW_TOKENS_ATTR, 35077)

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.provider._get_local_concurrency_info", _free_concurrency()),
        patch("proxy.provider.request_priority", return_value="low"),
        patch(
            "proxy.provider._maybe_queue_for_local_slot",
            new=AsyncMock(return_value=("dispatch", None, 0.5)),
        ),
    ):
        await provider.proxy_with_fallback(
            request, "v1/chat/completions", config, _queue_cfg()
        )

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "contention_queue_dispatch" in messages
    assert "new_tokens=35077" in messages


@pytest.mark.asyncio
async def test_lease_denied_fallback_log_carries_own_new_tokens(lease_state, caplog):
    """AC2: the abandon line carries the same own-request ``new_tokens``."""
    import logging

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {"name": "remote-ok", "type": "remote",
             "endpoint": "https://api.example.com/v1"},
        ]
    }

    async def _mock_proxy_to_local(_req, _path):
        return _lease_active_response()

    async def _mock_proxy_to_remote(_req, _path, _pc):
        return _ok_response()

    request = _DummyRequest()
    setattr(request, provider.CONTENTION_QUEUE_NEW_TOKENS_ATTR, 6000)

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", _free_concurrency()),
        patch("proxy.provider.request_priority", return_value="low"),
        patch(
            "proxy.provider._maybe_queue_for_local_slot",
            new=AsyncMock(return_value=("fallback", None, 300.0)),
        ),
    ):
        await provider.proxy_with_fallback(
            request, "v1/chat/completions", config, _queue_cfg()
        )

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "contention_queue_fallback_after_queue" in messages
    assert "new_tokens=6000" in messages


@pytest.mark.asyncio
async def test_lease_denied_dispatch_log_degrades_without_enqueue_evidence(
    lease_state, caplog
):
    """Missing enqueue evidence logs ``new_tokens=unknown`` — a parseable
    absence, never a guessed band."""
    import logging

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    config = {
        "providers": [{"name": "local-llama", "type": "local", "llama_model": "Qwen3"}]
    }
    local_calls = []

    async def _mock_proxy_to_local(_req, _path):
        local_calls.append(1)
        if len(local_calls) == 1:
            return _lease_active_response()
        return _ok_response()

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.provider._get_local_concurrency_info", _free_concurrency()),
        patch("proxy.provider.request_priority", return_value="low"),
        patch(
            "proxy.provider._maybe_queue_for_local_slot",
            new=AsyncMock(return_value=("dispatch", None, 0.5)),
        ),
    ):
        await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", config, _queue_cfg()
        )

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "contention_queue_dispatch" in messages
    assert "new_tokens=unknown" in messages


@pytest.mark.asyncio
async def test_lease_denied_budget_expires_all_remotes_out_of_window_terminal(lease_state, caplog):
    """AC3/AC4: lease denial + budget expiry + all remotes out of window
    yields a prompt terminal error (with a stable code), never a silent
    multi-cycle hold."""
    import logging
    from datetime import UTC, datetime

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {"name": "remote-windowed", "type": "remote",
             "endpoint": "https://api.example.com/v1",
             "available_times": ["00:00-01:00"]},
        ]
    }

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 5, 13, 0, 0, tzinfo=UTC)

    async def _mock_proxy_to_local(_req, _path):
        return _lease_active_response()

    async def _mock_proxy_to_remote(_req, _path, _pc):
        raise AssertionError("out-of-window remote must not be dispatched")

    # chain-hold enabled with a large budget: the window edge (11h) exceeds it,
    # so the hold must not run at all (LP-0MU56ZKQD005SX08).
    cfg = _queue_cfg()
    cfg["chain_hold_seconds"] = 300
    cfg["chain_hold_max_cycles"] = 3

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", _free_concurrency()),
        patch("proxy.provider.request_priority", return_value="low"),
        patch("proxy.provider.datetime", _FixedDateTime),
        patch(
            "proxy.provider._maybe_queue_for_local_slot",
            new=AsyncMock(return_value=("fallback", None, 300.0)),
        ),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", config, cfg
        )

    assert result.status_code == 503
    body = json.loads(result.body)
    assert body["error"] == "All providers exhausted"
    assert body["code"] == "all_exhausted", (
        "a lease-denied provider makes this generic exhaustion, not a window one"
    )
    assert "local_lease_active" in json.dumps(body.get("diagnostics", []))
    assert "contention_queue_fallback_after_queue" in " ".join(
        r.getMessage() for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Lease-owner retention grace resolution (LP-0MVI63OM1009K51W)
# ---------------------------------------------------------------------------


def test_lease_grace_default_is_not_clamped_and_logs_nothing(caplog):
    """AC1/AC3: the shipped 300 s default survives the resolver silently.

    Regression: the grace was clamped to ``contention_queue_max_wait_seconds``
    (120), so every per-request resolution emitted a WARNING and the default
    could never take effect.
    """
    import logging

    from proxy.router import _get_contention_queue_config

    caplog.set_level(logging.WARNING, logger="llama-proxy")
    cfg = {
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": 60,
        "contention_queue_max_depth": 8,
    }

    resolved = _get_contention_queue_config(cfg)

    assert resolved["lease_grace_seconds"] == 300.0
    resolved_again = _get_contention_queue_config(cfg)
    assert resolved_again["lease_grace_seconds"] == 300.0
    assert caplog.records == [], (
        "a valid configuration must not log per request: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


def test_lease_grace_clamps_to_the_runtime_ceiling(caplog):
    """AC2: above the guardrail ceiling → clamped once, with a warning."""
    import logging

    from proxy.router import _get_contention_queue_config

    caplog.set_level(logging.WARNING, logger="llama-proxy")
    resolved = _get_contention_queue_config(
        {
            "contention_queue_policy": "queue",
            "contention_queue_max_wait_seconds": 60,
            "contention_queue_lease_grace_seconds": 99999,
            "session_guardrail_max_runtime_seconds": 1800,
        }
    )

    assert resolved["lease_grace_seconds"] == 1800.0
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "contention_queue_lease_grace_seconds" in messages
    assert "clamped" in messages
    assert "session_guardrail_max_runtime_seconds" in messages


def test_lease_grace_zero_disables_retention_without_warning(caplog):
    """AC2: 0 is a valid value (rollback path), not a clamp."""
    import logging

    from proxy.router import _get_contention_queue_config

    caplog.set_level(logging.WARNING, logger="llama-proxy")
    assert (
        _get_contention_queue_config(
            {"contention_queue_lease_grace_seconds": 0}
        )["lease_grace_seconds"]
        == 0.0
    )
    assert caplog.records == []


def test_lease_grace_negative_and_garbage_clamp_to_zero(caplog):
    """AC2: negative clamps to 0; garbage falls back to the 300 s default."""
    import logging

    from proxy.router import _get_contention_queue_config

    caplog.set_level(logging.WARNING, logger="llama-proxy")
    assert (
        _get_contention_queue_config(
            {"contention_queue_lease_grace_seconds": -5}
        )["lease_grace_seconds"]
        == 0.0
    )
    assert (
        _get_contention_queue_config(
            {"contention_queue_lease_grace_seconds": "soon"}
        )["lease_grace_seconds"]
        == 300.0
    )
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "Invalid contention_queue_lease_grace_seconds" in messages


# ---------------------------------------------------------------------------
# LP-0MVV29X2E004INFL: trigger=/cause= attribution on queue lines
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wait_for_local_slot_reports_the_budget_expiry_cause():
    """AC1: a wait that spends its budget reports ``queue_wait_expired`` —
    the cause is explicit, not inferred from the duration."""
    cause: list[str] = []
    result = await contention_queue.wait_for_local_slot(
        0.05,
        max_depth=4,
        slot_free_check=lambda: False,
        cause_out=cause,
    )
    assert result is None
    assert cause == ["queue_wait_expired"]


@pytest.mark.asyncio
async def test_wait_for_local_slot_reports_the_depth_cap_cause():
    """AC1: a request that never enqueues reports ``queue_depth_exceeded``."""
    holder = asyncio.create_task(
        contention_queue.wait_for_local_slot(
            5.0, max_depth=1, slot_free_check=lambda: False
        )
    )
    await asyncio.sleep(0.02)
    try:
        cause: list[str] = []
        result = await contention_queue.wait_for_local_slot(
            5.0,
            max_depth=1,
            slot_free_check=lambda: False,
            cause_out=cause,
        )
        assert result is None
        assert cause == ["queue_depth_exceeded"]
    finally:
        holder.cancel()
        try:
            await holder
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_wait_for_local_slot_leaves_cause_empty_on_dispatch():
    """No abandon cause is reported when the slot is free at entry."""
    cause: list[str] = []
    result = await contention_queue.wait_for_local_slot(
        5.0,
        max_depth=4,
        slot_free_check=lambda: True,
        cause_out=cause,
    )
    assert result == 0.0
    assert cause == []


@pytest.mark.asyncio
async def test_concurrency_limit_queue_passes_its_entry_trigger(
    mixed_model_config,
):
    """AC1: the concurrency-limit call site names its entry path."""
    captured: dict = {}

    async def _fake_queue(*_args, **kwargs):
        captured.update(kwargs)
        return ("fallback", None, 1.0)

    request = _DummyRequest(headers=dict(PIN_LOCAL_HEADERS))
    with (
        patch("proxy.provider._get_local_concurrency_info",
              _MutableConcurrency(active=1, max_=1)),
        patch("proxy.provider._maybe_queue_for_local_slot", new=_fake_queue),
        patch("proxy.provider.request_priority", return_value="low"),
    ):
        await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, _queue_cfg()
        )

    assert captured.get("trigger") == "concurrency_limit"


@pytest.mark.asyncio
async def test_concurrency_limit_fallback_log_carries_trigger_and_cause(
    mixed_model_config, caplog,
):
    """AC1: the concurrency-limit fallback line (previously reason-less) names
    both the entry path and the explicit abandon cause."""
    import logging

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")

    async def _fake_queue(*_args, **_kwargs):
        return ("fallback", None, 1.0)

    request = _DummyRequest(headers=dict(PIN_LOCAL_HEADERS))
    setattr(request, provider.CONTENTION_QUEUE_CAUSE_ATTR, "queue_wait_expired")
    with (
        patch("proxy.provider._get_local_concurrency_info",
              _MutableConcurrency(active=1, max_=1)),
        patch("proxy.provider._maybe_queue_for_local_slot", new=_fake_queue),
        patch("proxy.provider.request_priority", return_value="low"),
    ):
        await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, _queue_cfg()
        )

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "contention_queue_fallback_after_queue" in messages
    assert "trigger=concurrency_limit" in messages
    assert "cause=queue_wait_expired" in messages


@pytest.mark.asyncio
async def test_lease_active_queue_passes_its_entry_trigger(lease_state):
    """AC1: the lease-active retry call site names its entry path."""
    captured: dict = {}

    async def _mock_proxy_to_local(_req, _path):
        return _lease_active_response()

    async def _fake_queue(*_args, **kwargs):
        captured.update(kwargs)
        return ("fallback", None, 1.0)

    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {"name": "remote-ok", "type": "remote",
             "endpoint": "https://api.example.com/v1"},
        ]
    }
    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.provider._get_local_concurrency_info", _free_concurrency()),
        patch("proxy.provider.request_priority", return_value="low"),
        patch("proxy.provider._maybe_queue_for_local_slot", new=_fake_queue),
    ):
        await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", config, _queue_cfg()
        )

    assert captured.get("trigger") == "local_lease_active"


@pytest.mark.asyncio
async def test_lease_active_fallback_log_carries_trigger_and_cause(
    lease_state, caplog,
):
    """AC1: the lease-active fallback line names the entry path and the
    explicit abandon cause, alongside the pre-existing ``reason=`` token."""
    import logging

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")

    async def _mock_proxy_to_local(_req, _path):
        return _lease_active_response()

    async def _fake_queue(*_args, **_kwargs):
        return ("fallback", None, 1.0)

    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {"name": "remote-ok", "type": "remote",
             "endpoint": "https://api.example.com/v1"},
        ]
    }
    request = _DummyRequest()
    setattr(request, provider.CONTENTION_QUEUE_CAUSE_ATTR, "queue_depth_exceeded")
    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.provider._get_local_concurrency_info", _free_concurrency()),
        patch("proxy.provider.request_priority", return_value="low"),
        patch("proxy.provider._maybe_queue_for_local_slot", new=_fake_queue),
    ):
        await provider.proxy_with_fallback(
            request, "v1/chat/completions", config, _queue_cfg()
        )

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "contention_queue_fallback_after_queue" in messages
    assert "reason=local_lease_active" in messages
    assert "trigger=local_lease_active" in messages
    assert "cause=queue_depth_exceeded" in messages


# ===================================================================
# The lease-active re-queue path: request priority + ONE queue budget
# per turn (LP-0MVV29Y1M004GAW0)
# ===================================================================


def _tiny_queue_config(max_wait: float = 0.2, depth: int = 4) -> dict:
    """Unclamped queue config for the sub-second budget assertions.

    ``_get_contention_queue_config`` clamps the wait to a 1 s floor, so a
    "the second wait is only the remainder" assertion would take a full second
    per case. Patching the resolver keeps the assertion on the *arithmetic*;
    the clamped production value is pinned by the shipped-config tests.
    """
    return {
        "policy": "queue",
        "max_wait_seconds": max_wait,
        "max_depth": depth,
        "lease_grace_seconds": 300,
        "eta_ordering": False,
        "eta_promotion_seconds": 0.0,
    }


@pytest.mark.asyncio
async def test_lease_active_queue_passes_the_request_priority(lease_state):
    """AC1: the lease-active retry passes the request's priority through.

    Without it ``priority=None`` reached both ``_contention_queue_enabled``
    and the admission guard, so the documented ``high`` never-queues contract
    was not enforced on this path at all.
    """
    captured: dict = {}

    async def _mock_proxy_to_local(_req, _path):
        return _lease_active_response()

    async def _fake_queue(*_args, **kwargs):
        captured.update(kwargs)
        return ("fallback_policy", None, None)

    async def _mock_proxy_to_remote(_req, _path, _pc):
        return _ok_response()

    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {"name": "remote-ok", "type": "remote",
             "endpoint": "https://api.example.com/v1"},
        ]
    }
    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", _free_concurrency()),
        patch("proxy.provider.request_priority", return_value="high"),
        patch("proxy.provider._maybe_queue_for_local_slot", new=_fake_queue),
    ):
        await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", config, _queue_cfg()
        )

    assert captured.get("priority") == "high"


@pytest.mark.asyncio
async def test_lease_active_queue_passes_the_wait_already_spent(lease_state):
    """AC2: the retry tells the queue how much of the budget the turn spent."""
    captured: dict = {}

    async def _mock_proxy_to_local(_req, _path):
        return _lease_active_response()

    async def _fake_queue(*_args, **kwargs):
        captured.update(kwargs)
        return ("fallback_policy", None, None)

    async def _mock_proxy_to_remote(_req, _path, _pc):
        return _ok_response()

    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {"name": "remote-ok", "type": "remote",
             "endpoint": "https://api.example.com/v1"},
        ]
    }
    request = _DummyRequest()
    provider._set_queue_wait_on_request(request, 12.5)
    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", _free_concurrency()),
        patch("proxy.provider.request_priority", return_value="low"),
        patch("proxy.provider._maybe_queue_for_local_slot", new=_fake_queue),
    ):
        await provider.proxy_with_fallback(
            request, "v1/chat/completions", config, _queue_cfg()
        )

    assert captured.get("already_waited_seconds") == 12.5


@pytest.mark.asyncio
async def test_lease_active_queue_passes_no_spend_without_a_marker(lease_state):
    """AC2 (untouched path): no earlier wait → the default ``0.0`` is passed."""
    captured: dict = {}

    async def _mock_proxy_to_local(_req, _path):
        return _lease_active_response()

    async def _fake_queue(*_args, **kwargs):
        captured.update(kwargs)
        return ("fallback_policy", None, None)

    async def _mock_proxy_to_remote(_req, _path, _pc):
        return _ok_response()

    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {"name": "remote-ok", "type": "remote",
             "endpoint": "https://api.example.com/v1"},
        ]
    }
    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", _free_concurrency()),
        patch("proxy.provider.request_priority", return_value="low"),
        patch("proxy.provider._maybe_queue_for_local_slot", new=_fake_queue),
    ):
        await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", config, _queue_cfg()
        )

    assert captured.get("already_waited_seconds") == 0.0


@pytest.mark.asyncio
async def test_high_priority_lease_active_request_never_queues(lease_state, caplog):
    """AC1: a ``high`` priority request denied by a lease goes straight remote.

    Behavioural proof, not a keyword assertion: the real
    ``_maybe_queue_for_local_slot`` runs (a 0.2 s window, so a regression
    fails fast instead of hanging for the full production budget), no queue
    wait is logged, the queue stays empty, and the next provider serves.
    """
    import logging

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {"name": "remote-ok", "type": "remote",
             "endpoint": "https://api.example.com/v1"},
        ]
    }
    remote_calls = []

    async def _mock_proxy_to_local(_req, _path):
        return _lease_active_response()

    async def _mock_proxy_to_remote(_req, _path, _pc):
        remote_calls.append(1)
        return _ok_response()

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", _free_concurrency()),
        patch("proxy.provider.request_priority", return_value="high"),
        patch(
            "proxy.router._get_contention_queue_config",
            lambda _server: _tiny_queue_config(0.2),
        ),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", config, _queue_cfg()
        )

    assert result.status_code == 200
    assert remote_calls == [1], "a lease denial at high priority falls straight through"
    assert contention_queue.queue_depth() == 0, "high priority must never enter the queue"
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "contention_queue_wait" not in messages
    assert "reason=local_lease_active" in messages


def test_queue_wait_budget_deducts_the_wait_already_spent():
    """AC2 unit: the second wait's budget is the remainder of one window."""
    cfg = {
        "server": {
            "contention_queue_policy": "queue",
            "contention_queue_max_wait_seconds": 30,
        }
    }

    # Nothing spent → the full configured window (today's behaviour).
    assert provider._contention_queue_wait_budget_seconds(cfg, 0.0) == 30.0
    assert provider._contention_queue_wait_budget_seconds(cfg, None) == 30.0
    # Spent some → only the remainder.
    assert provider._contention_queue_wait_budget_seconds(cfg, 12.5) == 17.5
    # Spent it all (or more) → no budget left, never a negative window.
    assert provider._contention_queue_wait_budget_seconds(cfg, 30.0) == 0.0
    assert provider._contention_queue_wait_budget_seconds(cfg, 45.0) == 0.0
    # A nonsense negative spend can never inflate the budget past the cap.
    assert provider._contention_queue_wait_budget_seconds(cfg, -5.0) == 30.0


@pytest.mark.asyncio
async def test_second_lease_queue_wait_uses_only_the_remaining_budget():
    """AC2: a turn that already waited is bounded by the measured remainder.

    Window 0.3 s, 0.25 s already spent → the second wait may run for at most
    ~0.05 s, so the cumulative queue wait of the turn stays inside one budget
    instead of accumulating two full windows.
    """
    cfg = _queue_cfg()
    spent = 0.25

    with (
        patch(
            "proxy.router._get_contention_queue_config",
            lambda _server: _tiny_queue_config(0.3),
        ),
        patch(
            "proxy.provider._get_local_concurrency_info",
            _MutableConcurrency(active=1, max_=1),
        ),
        patch(
            "proxy.provider._queue_context_bypass",
            new=AsyncMock(return_value=(False, None)),
        ),
    ):
        started = time.monotonic()
        action, _reason, elapsed = await provider._maybe_queue_for_local_slot(
            cfg, 1, 1, _DummyRequest(headers=dict(PIN_LOCAL_HEADERS)), {}, {}, {},
            "turn-session",
            priority="low",
            trigger="local_lease_active",
            already_waited_seconds=spent,
        )
        second_wait = time.monotonic() - started

    assert action == "fallback"
    assert second_wait < 0.2, (
        "the second wait must be the ~0.05 s remainder, not a fresh 0.3 s window"
    )
    assert elapsed is not None
    assert elapsed >= spent, "the reported wait is cumulative for the turn"
    assert elapsed <= 0.3 + 0.1, (
        "cumulative queue wait for one turn must stay inside one budget"
    )


@pytest.mark.asyncio
async def test_lease_queue_enqueue_logs_the_spent_wait_and_effective_budget(
    caplog,
):
    """AC2/AC3 observability: the enqueue line names the spend and the bound.

    A turn that already burned the whole window gets a ``0.0`` budget, so the
    enqueue line must show both numbers: ``max_wait=`` stays the configured
    window (diagnosis reads it as the configuration), ``budget=`` is what this
    wait may actually consume.
    """
    import logging

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    cfg = _queue_cfg()

    with (
        patch(
            "proxy.router._get_contention_queue_config",
            lambda _server: _tiny_queue_config(0.5),
        ),
        patch(
            "proxy.provider._get_local_concurrency_info",
            _MutableConcurrency(active=1, max_=1),
        ),
        patch(
            "proxy.provider._queue_context_bypass",
            new=AsyncMock(return_value=(False, None)),
        ),
    ):
        action, _reason, elapsed = await provider._maybe_queue_for_local_slot(
            cfg, 1, 1, _DummyRequest(headers=dict(PIN_LOCAL_HEADERS)), {}, {}, {},
            "spent-out-session",
            priority="low",
            trigger="local_lease_active",
            already_waited_seconds=0.5,
        )

    assert action == "fallback", "no budget left → the wait expires immediately"
    assert elapsed is not None and elapsed >= 0.5
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "contention_queue_wait" in messages
    assert "max_wait=0.5s" in messages, "max_wait= keeps the configured window"
    assert "spent_wait=0.5s" in messages
    assert "budget=0.0s" in messages


@pytest.mark.asyncio
async def test_lease_active_dispatch_records_cumulative_wait_on_the_request(
    lease_state, caplog,
):
    """AC3: the admitted wait is recorded cumulatively for the turn.

    The marker is what bounds a later re-entry and what the adaptive timeout
    subtracts, so the dispatch path must write the turn's whole wait.
    """
    import logging

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    config = {
        "providers": [{"name": "local-llama", "type": "local", "llama_model": "Qwen3"}]
    }
    local_calls = []

    async def _mock_proxy_to_local(_req, _path):
        local_calls.append(1)
        if len(local_calls) == 1:
            return _lease_active_response()
        return _ok_response()

    request = _DummyRequest()
    provider._set_queue_wait_on_request(request, 2.5)
    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.provider._get_local_concurrency_info", _free_concurrency()),
        patch("proxy.provider.request_priority", return_value="low"),
        patch(
            "proxy.provider._maybe_queue_for_local_slot",
            new=AsyncMock(return_value=("dispatch", None, 4.25)),
        ),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", config, _queue_cfg()
        )

    assert result.status_code == 200
    assert local_calls == [1, 1]
    assert provider._queue_wait_spent_on_request(request) == 4.25
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "queued_duration=4.25s" in messages


async def _settle(loops: int = 5) -> None:
    """Let queued coroutines reach their ``await`` point."""
    for _ in range(loops):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_wait_for_local_slot_reports_the_cumulative_turn_wait():
    """AC3 unit: an admitted second wait returns the turn's total wait."""
    free = False

    def _slot_free() -> bool:
        return free

    task = asyncio.create_task(
        contention_queue.wait_for_local_slot(
            max_wait_seconds=5.0,
            max_depth=4,
            slot_free_check=_slot_free,
            session_id="cumulative-session",
            already_waited_seconds=5.0,
        )
    )
    await _settle()
    free = True
    await contention_queue.wake(1)
    elapsed = await asyncio.wait_for(task, timeout=2)

    assert elapsed is not None and elapsed >= 5.0


@pytest.mark.asyncio
async def test_wait_for_local_slot_defaults_to_the_measured_wait():
    """An untouched caller still gets its own measured wait (no regression)."""
    free = False

    def _slot_free() -> bool:
        return free

    task = asyncio.create_task(
        contention_queue.wait_for_local_slot(
            max_wait_seconds=5.0,
            max_depth=4,
            slot_free_check=_slot_free,
            session_id="first-wait-session",
        )
    )
    await _settle()
    free = True
    await contention_queue.wake(1)
    elapsed = await asyncio.wait_for(task, timeout=2)

    assert elapsed is not None and elapsed < 1.0
