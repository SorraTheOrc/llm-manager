"""Tests for contention-queue coverage of prefill-saturated dispatches.

LP-0MU498TWH008DBCI: a request denied solely by the router's prefill-aware
guard (``prefill_count >= session_slot_pool_size`` while the generating
count is 0) must wait in the bounded contention queue instead of immediately
falling back to a remote provider. The queue wake predicate must therefore
account for BOTH the generating pool and the prefill-in-flight count.

Before the fix, ``slot_free_check`` watched only the generating count, so a
prefill-saturated waiter woke immediately (``0 < max_local``), re-dispatched,
was denied again and fell through to remote as ``local_lease_active``
(fast mode: 425 such fallbacks in the 2026-09-15/16 window).
"""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, patch

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


class _Concurrency:
    """Mutable stand-in for ``_get_local_concurrency_info``.

    Reports the generating-only count. Accepts the optional ``endpoint``
    kwarg so ``_local_concurrency_info`` forwards it (LP-0MRPILSMW004T4H8).
    """

    def __init__(self, active: int = 0, max_: int = 1):
        self.active = active
        self.max = max_

    def __call__(self, config, endpoint=None) -> tuple[int, int]:
        return (self.active, self.max)


def _cfg(max_wait: float = 60.0, max_depth: int = 4, slots: int = 1) -> dict:
    """Cheap-mode config with the contention-queue keys present."""
    return {
        "provider_cooldown_seconds": 60,
        "server": {
            "session_slot_pool_size": slots,
            "contention_queue_policy": "queue",
            "contention_queue_max_wait_seconds": max_wait,
            "contention_queue_max_depth": max_depth,
        },
    }


def _ok_response() -> Response:
    return Response(
        content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
        status_code=200,
        media_type="application/json",
    )


def _lease_active_response() -> Response:
    """Synthetic 503 for a router prefill-gate denial (``local_lease_active``)."""
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


@pytest.fixture(autouse=True)
def reset_queue():
    """Reset the cross-session queue between tests."""
    contention_queue.reset()
    yield
    contention_queue.reset()


@pytest.fixture
def prefill_state(monkeypatch):
    """Expose a mutable ``local_prefill_in_flight`` dict on proxy.server."""
    import proxy.server as srv

    state: dict = {}
    monkeypatch.setattr(srv, "local_prefill_in_flight", state, raising=False)
    return state


# ---------------------------------------------------------------------------
# AC1/AC4: the wake predicate accounts for prefill-in-flight
# ---------------------------------------------------------------------------

class TestLocalSlotSaturation:
    def test_prefill_at_cap_reports_prefill(self, prefill_state):
        """generating==0, prefill==max → prefill-saturated, NOT free."""
        prefill_state["other"] = True
        with patch("proxy.provider._get_local_concurrency_info", _Concurrency(0, 1)):
            assert provider._local_slot_saturation(_cfg(), 1) == "prefill"
            assert provider._local_slot_free(_cfg(), 1) is False

    def test_generating_at_cap_reports_generating(self, prefill_state):
        with patch("proxy.provider._get_local_concurrency_info", _Concurrency(1, 1)):
            assert provider._local_slot_saturation(_cfg(), 1) == "generating"
            assert provider._local_slot_free(_cfg(), 1) is False

    def test_both_below_cap_is_free(self, prefill_state):
        with patch("proxy.provider._get_local_concurrency_info", _Concurrency(0, 2)):
            assert provider._local_slot_saturation(_cfg(slots=2), 2) is None
            assert provider._local_slot_free(_cfg(slots=2), 2) is True

    def test_prefill_guard_not_defeated_at_higher_slot_count(self, prefill_state):
        """AC4: with N slots, N prefills block; N-1 do not."""
        for i in range(3):
            prefill_state[f"s{i}"] = True
        with patch("proxy.provider._get_local_concurrency_info", _Concurrency(0, 3)):
            assert provider._local_slot_free(_cfg(slots=3), 3) is False
        prefill_state.pop("s0")
        with patch("proxy.provider._get_local_concurrency_info", _Concurrency(0, 3)):
            assert provider._local_slot_free(_cfg(slots=3), 3) is True

    def test_missing_prefill_state_fails_open(self, monkeypatch):
        """No prefill tracking attribute → generating-only behaviour."""
        import proxy.server as srv

        monkeypatch.setattr(srv, "local_prefill_in_flight", None, raising=False)
        with patch("proxy.provider._get_local_concurrency_info", _Concurrency(0, 1)):
            assert provider._local_slot_free(_cfg(), 1) is True


# ---------------------------------------------------------------------------
# AC1/AC2: _maybe_queue_for_local_slot waits on prefill
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_queue_waits_for_prefill_then_dispatches(prefill_state):
    """A prefill-saturated request waits; when the prefill completes it is
    dispatched local (elapsed wait is returned for the Q2=a budget)."""
    cfg = _cfg(max_wait=5.0)
    prefill_state["other"] = True

    async def _clear_prefill():
        await asyncio.sleep(0.05)
        prefill_state.clear()
        await contention_queue.wake_all()

    with (
        patch("proxy.provider._get_local_concurrency_info", _Concurrency(0, 1)),
        patch(
            "proxy.provider._queue_context_bypass",
            new=AsyncMock(return_value=(False, None)),
        ),
    ):
        asyncio.create_task(_clear_prefill())
        action, reason, elapsed = await provider._maybe_queue_for_local_slot(
            cfg, 0, 1, _DummyRequest(), {}, {}, {}, "s2",
        )

    assert action == "dispatch", "prefill-saturated request must wait, not fall back"
    assert reason is None
    assert elapsed is not None and elapsed >= 0.05


@pytest.mark.asyncio
async def test_queue_cap_exceeded_when_prefill_never_clears(prefill_state):
    """A prefill that never completes exhausts the wait cap → bounded fallback."""
    cfg = _cfg(max_wait=0.1)
    prefill_state["other"] = True

    with (
        patch("proxy.provider._get_local_concurrency_info", _Concurrency(0, 1)),
        patch(
            "proxy.provider._queue_context_bypass",
            new=AsyncMock(return_value=(False, None)),
        ),
    ):
        action, reason, elapsed = await provider._maybe_queue_for_local_slot(
            cfg, 0, 1, _DummyRequest(), {}, {}, {}, "s2",
        )

    assert action == "fallback"
    assert elapsed is not None and elapsed >= 0.1


@pytest.mark.asyncio
async def test_queue_logs_prefill_saturation(prefill_state, caplog):
    """AC5: the queue-wait log distinguishes prefill from generating waits."""
    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    cfg = _cfg(max_wait=0.05)
    prefill_state["other"] = True

    with (
        patch("proxy.provider._get_local_concurrency_info", _Concurrency(0, 1)),
        patch(
            "proxy.provider._queue_context_bypass",
            new=AsyncMock(return_value=(False, None)),
        ),
    ):
        await provider._maybe_queue_for_local_slot(
            cfg, 0, 1, _DummyRequest(), {}, {}, {}, "s2",
        )

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "contention_queue_wait" in messages
    assert "saturation=prefill" in messages


@pytest.mark.asyncio
async def test_queue_logs_generating_saturation(caplog):
    """AC5: generating saturation is labelled distinctly from prefill."""
    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    cfg = _cfg(max_wait=0.05)

    with (
        patch("proxy.provider._get_local_concurrency_info", _Concurrency(1, 1)),
        patch(
            "proxy.provider._queue_context_bypass",
            new=AsyncMock(return_value=(False, None)),
        ),
    ):
        await provider._maybe_queue_for_local_slot(
            cfg, 1, 1, _DummyRequest(), {}, {}, {}, "s2",
        )

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "saturation=generating" in messages


# ---------------------------------------------------------------------------
# AC2/AC6: end-to-end — post-dispatch lease denial queues on prefill, then
# dispatches local (instead of falling back as local_lease_active)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prefill_denied_request_queues_and_redispatches_local(
    prefill_state, caplog
):
    """The router denies on the prefill gate; the request waits for the
    prefill to complete and is served locally — no remote dispatch."""
    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {
                "name": "remote-ok",
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
            },
        ],
    }
    local_calls = []
    remote_calls = []

    async def _mock_proxy_to_local(_req, _path):
        local_calls.append(1)
        # Deny while the prefill is in flight; succeed once it has cleared.
        # With the pre-fix predicate the queue returned immediately, so this
        # second dispatch happened while the prefill was still in flight and
        # was denied again → remote fallback.
        if prefill_state:
            return _lease_active_response()
        return _ok_response()

    async def _mock_proxy_to_remote(_req, _path, _pc):
        remote_calls.append(1)
        return _ok_response()

    async def _clear_prefill():
        await asyncio.sleep(0.05)
        prefill_state.clear()
        await contention_queue.wake_all()

    prefill_state["other"] = True

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", _Concurrency(0, 1)),
        patch("proxy.mode.read_mode", return_value="cheap"),
        patch(
            "proxy.provider._queue_context_bypass",
            new=AsyncMock(return_value=(False, None)),
        ),
    ):
        asyncio.create_task(_clear_prefill())
        result = await asyncio.wait_for(
            provider.proxy_with_fallback(
                _DummyRequest(), "v1/chat/completions", config, _cfg(max_wait=5.0),
            ),
            timeout=5,
        )

    assert result.status_code == 200
    assert remote_calls == [], "prefill-saturated request must not fall back remote"
    assert len(local_calls) == 2, "lease denial must re-dispatch local after the wait"
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "contention_queue_dispatch" in messages


@pytest.mark.asyncio
async def test_prefill_denied_request_cap_exceeded_falls_back_to_remote(
    prefill_state, caplog
):
    """AC6: when the prefill outlasts the wait cap, the request falls back to
    an available remote provider and records ``fallback_after_queue``."""
    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {
                "name": "remote-ok",
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
            },
        ],
    }
    remote_calls = []

    async def _mock_proxy_to_local(_req, _path):
        return _lease_active_response()

    async def _mock_proxy_to_remote(_req, _path, _pc):
        remote_calls.append(1)
        return _ok_response()

    prefill_state["other"] = True  # never clears

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", _Concurrency(0, 1)),
        patch("proxy.mode.read_mode", return_value="cheap"),
        patch(
            "proxy.provider._queue_context_bypass",
            new=AsyncMock(return_value=(False, None)),
        ),
    ):
        result = await asyncio.wait_for(
            provider.proxy_with_fallback(
                _DummyRequest(), "v1/chat/completions", config, _cfg(max_wait=0.1),
            ),
            timeout=5,
        )

    assert result.status_code == 200
    assert remote_calls == [1], "cap-exceeded prefill wait must fall back remote"
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "contention_queue_fallback_after_queue" in messages
