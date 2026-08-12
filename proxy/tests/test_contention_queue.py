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
- AC8: queue metrics (queued count, queued duration, fallback-after-queue
  count) are emitted when policy is queue; not emitted when fallback.

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

class _DummyRequest:
    """Minimal request stub for use in fallback tests."""
    def __init__(self, body: bytes = b'{"model":"test"}'):
        self._body = body
        self.headers = {}
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
        content='{"choices":[{"message":{"content":"é中文 bytes"}}]}'.encode("utf-8"),
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

    request = _DummyRequest()
    cfg = _queue_cfg()

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
        patch("proxy.mode.read_mode", return_value="cheap"),
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
    metrics = contention_queue.metrics()
    assert metrics["contention_queued_count"] == 1
    assert metrics["contention_queued_duration_seconds"] >= 0.0
    assert metrics["contention_fallback_after_queue_count"] == 0

    # F4 AC1: the dispatch log line carries queue depth + policy.
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "contention_queue_dispatch" in messages
    assert "policy=queue" in messages
    assert "depth=" in messages


# ---------------------------------------------------------------------------
# AC2: fallback to next remote provider after max_wait exceeded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_after_max_wait_exceeded(mixed_model_config, caplog):
    """When the wait cap is exceeded (no slot frees), fall back to the next
    remote provider exactly as today. The fallback-after-queue event records
    the elapsed wait time (F4 AC2)."""
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

    request = _DummyRequest()
    cfg = _queue_cfg()

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
        patch("proxy.mode.read_mode", return_value="cheap"),
        # Tiny cap: 60s real wait would be far too slow for the suite.
        patch(
            "proxy.router._get_contention_queue_config",
            return_value={"policy": "queue", "max_wait_seconds": 0.05, "max_depth": 4},
        ),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert call_log == ["remote"], "wait-cap exceeded must fall back to remote"
    metrics = contention_queue.metrics()
    assert metrics["contention_queued_count"] == 1
    assert metrics["contention_fallback_after_queue_count"] == 1

    # F4 AC2: the fallback-after-queue log line carries the elapsed wait.
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
    metrics = contention_queue.metrics()
    assert metrics["contention_fallback_after_queue_count"] == 1
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
    """Provider level: with max_depth=1, a second concurrent request falls
    back to remote immediately while the first is queued."""
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
        patch("proxy.mode.read_mode", return_value="cheap"),
        patch(
            "proxy.router._get_contention_queue_config",
            return_value={"policy": "queue", "max_wait_seconds": 0.2, "max_depth": 1},
        ),
    ):
        first = asyncio.create_task(
            provider.proxy_with_fallback(
                _DummyRequest(), "v1/chat/completions", mixed_model_config, cfg
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
            _DummyRequest(), "v1/chat/completions", mixed_model_config, cfg
        )
        assert result2.status_code == 200
        assert "remote" in call_log
        metrics = contention_queue.metrics()
        assert metrics["contention_fallback_after_queue_count"] >= 1

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
    """A request that would be context-bypassed (context_too_large /
    large_context_bypass) must NEVER wait in the contention queue — it falls
    back exactly as today."""
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
    )

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", concurrency),
        patch("proxy.mode.read_mode", return_value="cheap"),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert call_log == ["remote"], "context bypass must fall back to remote"
    metrics = contention_queue.metrics()
    assert metrics["contention_queued_count"] == 0, "context bypass must not queue"
    assert contention_queue.queue_depth() == 0
    assert metrics["contention_fallback_after_queue_count"] == 0


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
        patch("proxy.mode.read_mode", return_value="fast"),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert call_log == ["remote"], "fast mode falls back immediately, no queue"
    metrics = contention_queue.metrics()
    assert metrics["contention_queued_count"] == 0
    assert metrics["contention_fallback_after_queue_count"] == 0
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
    ).encode("utf-8")
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
        patch("proxy.mode.read_mode", return_value="fast"),
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
    metrics = contention_queue.metrics()
    assert metrics["contention_queued_count"] == 0
    assert metrics["contention_fallback_after_queue_count"] == 0


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
        patch("proxy.mode.read_mode", return_value="cheap"),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert call_log == ["remote"]
    assert contention_queue.metrics()["contention_queued_count"] == 0


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
        patch("proxy.mode.read_mode", return_value="cheap"),
    ):
        request = _DummyRequest()
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


def test_metrics_not_emitted_when_fallback_policy():
    """status_request fields helper returns no queue fields for fallback."""
    from proxy.contention_queue import status_fields

    assert status_fields({"contention_queue_policy": "fallback"}) == {}


def test_metrics_emitted_when_queue_policy():
    """status_request fields helper exposes queue metrics for queue policy
    while in cheap mode."""
    from proxy.contention_queue import status_fields

    with patch("proxy.mode.read_mode", return_value="cheap"):
        fields = status_fields(
            {
                "contention_queue_policy": "queue",
                "contention_queue_max_wait_seconds": 60,
                "contention_queue_max_depth": 4,
            }
        )
    assert fields.get("contention_queue_policy") == "queue"
    assert "contention_queue_depth" in fields
    assert "contention_queued_count" in fields
    assert "contention_queued_duration_seconds" in fields
    assert "contention_fallback_after_queue_count" in fields


def test_metrics_suppressed_when_queue_policy_but_fast_mode():
    """status_fields returns {} for queue-policy config while mode=fast (F4
    AC4): a config override must never emit queue fields unless the proxy is
    actually in cheap operating mode."""
    from proxy.contention_queue import status_fields

    with patch("proxy.mode.read_mode", return_value="fast"):
        fields = status_fields(
            {
                "contention_queue_policy": "queue",
                "contention_queue_max_wait_seconds": 60,
                "contention_queue_max_depth": 4,
            }
        )
    assert fields == {}


@pytest.mark.asyncio
async def test_queue_log_lines_emitted_for_queue_policy(mixed_model_config, caplog):
    """Dispatch / fallback log lines carry queue context in queue policy."""
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
        patch("proxy.mode.read_mode", return_value="cheap"),
        patch(
            "proxy.router._get_contention_queue_config",
            return_value={"policy": "queue", "max_wait_seconds": 0.05, "max_depth": 4},
        ),
    ):
        await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", mixed_model_config, cfg
        )

    assert call_log == ["remote"]
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
        patch("proxy.mode.read_mode", return_value="fast"),
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
    """config-cheap.yaml declares queue policy + caps (F2 AC1)."""
    import yaml
    from proxy.mode import proxy_dir

    with open(proxy_dir() / "config-cheap.yaml") as fh:
        server = yaml.safe_load(fh)["server"]
    assert server["contention_queue_policy"] == "queue"
    assert server["contention_queue_max_wait_seconds"] == 60
    assert server["contention_queue_max_depth"] == 4


def test_fast_config_declares_fallback_policy():
    """config-fast.yaml declares fallback policy (F2 AC2)."""
    import yaml
    from proxy.mode import proxy_dir

    with open(proxy_dir() / "config-fast.yaml") as fh:
        server = yaml.safe_load(fh)["server"]
    assert server["contention_queue_policy"] == "fallback"
