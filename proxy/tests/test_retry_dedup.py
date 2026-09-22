"""
Hermetic tests for in-flight retry de-duplication and Retry-After
(LP-0MUCEFCV5006UKSZ).

Context
-------
When a client times out waiting for a slow local prefill it retries the same
prompt. Previously the retry either queued behind the original (then
re-prefilled) or, via the client-disconnect path, cancelled the original
prefill — so retries made no net progress and the backend churned context
checkpoints.

The fix de-duplicates by session + messages hash: an identical in-flight
request is rejected with a retryable 409 + Retry-After and the original
prefill is left untouched. Lease-denial (``local_dispatch_denied``) responses
also carry a ``Retry-After`` header so clients can back off.

These tests verify hermetically (no live llama-server).
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# _request_dedup_hash
# ═══════════════════════════════════════════════════════════════════════════════


def test_dedup_hash_stable_for_identical_messages():
    from proxy.router_helpers import _request_dedup_hash

    body = {"model": "x", "messages": [{"role": "user", "content": "hello"}]}
    assert _request_dedup_hash(body) == _request_dedup_hash(dict(body))


def test_dedup_hash_ignores_sampling_params():
    from proxy.router_helpers import _request_dedup_hash

    msgs = [{"role": "user", "content": "hello"}]
    a = {"messages": msgs, "temperature": 0.1}
    b = {"messages": msgs, "temperature": 0.9, "max_tokens": 100}
    assert _request_dedup_hash(a) == _request_dedup_hash(b)


def test_dedup_hash_differs_for_different_messages():
    from proxy.router_helpers import _request_dedup_hash

    a = {"messages": [{"role": "user", "content": "hello"}]}
    b = {"messages": [{"role": "user", "content": "world"}]}
    assert _request_dedup_hash(a) != _request_dedup_hash(b)


def test_dedup_hash_none_for_invalid_body():
    from proxy.router_helpers import _request_dedup_hash

    assert _request_dedup_hash(None) is None
    assert _request_dedup_hash({}) is None
    assert _request_dedup_hash({"messages": "not-a-list"}) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Coordinator duplicate detection
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_identical_inflight_request_rejected_as_duplicate():
    from proxy.session import (
        SessionSingleFlightDuplicateError,
    )
    from proxy.session import (
        session_single_flight_coordinator as coord,
    )

    async with coord.acquire("sess-dup", "queue", 1, request_hash="H1"):
        with pytest.raises(SessionSingleFlightDuplicateError):
            async with coord.acquire("sess-dup", "queue", 1, request_hash="H1"):
                pass


@pytest.mark.asyncio
async def test_different_prompt_not_treated_as_duplicate():
    from proxy.session import (
        SessionSingleFlightDuplicateError,
        SessionSingleFlightRejectedError,
    )
    from proxy.session import (
        session_single_flight_coordinator as coord,
    )

    async with coord.acquire("sess-diff", "queue", 1, request_hash="H1"):
        # A different prompt is not a duplicate; with reject mode it is a
        # normal single-flight rejection, never a duplicate.
        with pytest.raises(SessionSingleFlightRejectedError):
            async with coord.acquire(
                "sess-diff", "reject", 1, request_hash="H2"
            ):
                pass
        # ...and it must not raise DuplicateError.
        try:
            async with coord.acquire("sess-diff", "reject", 1, request_hash="H2"):
                pass
        except SessionSingleFlightDuplicateError:
            pytest.fail("Different prompt must not be a duplicate")
        except SessionSingleFlightRejectedError:
            pass


@pytest.mark.asyncio
async def test_duplicate_after_release_is_not_flagged():
    from proxy.session import session_single_flight_coordinator as coord

    async with coord.acquire("sess-seq", "queue", 1, request_hash="H1"):
        pass
    # The first request finished; a new identical request is not a duplicate.
    async with coord.acquire("sess-seq", "queue", 1, request_hash="H1"):
        pass


@pytest.mark.asyncio
async def test_duplicate_observability_counter():
    from proxy.session import (
        SessionSingleFlightDuplicateError,
    )
    from proxy.session import (
        session_single_flight_coordinator as coord,
    )

    from proxy import server as srv

    srv.session_single_flight_observability["duplicate_events_total"] = 0
    async with coord.acquire("sess-metric", "queue", 1, request_hash="H1"):
        with pytest.raises(SessionSingleFlightDuplicateError):
            async with coord.acquire("sess-metric", "queue", 1, request_hash="H1"):
                pass
    assert srv.session_single_flight_observability["duplicate_events_total"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Router: 409 + Retry-After for a duplicate, Retry-After on lease denial
# ═══════════════════════════════════════════════════════════════════════════════


class _DummyRequest:
    def __init__(self, body: bytes):
        self._body = body
        self.headers = {"content-type": "application/json"}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


def _install_router_harness(monkeypatch, server_cfg: dict):
    from proxy import server as srv

    monkeypatch.setattr(srv, "config", server_cfg)
    proc = MagicMock()
    proc.poll.return_value = None
    monkeypatch.setattr(srv, "llama_process", proc)
    monkeypatch.setattr(srv, "backend_ready", True)
    monkeypatch.setattr(srv, "current_model", "Qwen3")
    monkeypatch.setattr(srv, "active_queries", 0)
    monkeypatch.setattr(srv, "active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "local_active_queries", 0)
    monkeypatch.setattr(srv, "local_active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "local_generating_queries", 0)
    monkeypatch.setattr(srv, "local_generating_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "local_generating_sessions", set())
    monkeypatch.setattr(srv, "local_dispatch_records", {})
    monkeypatch.setattr(srv, "local_dispatch_records_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "local_prefill_in_flight", {})
    monkeypatch.setattr(srv, "local_prefill_in_flight_lock", asyncio.Lock())

    import proxy.router as router_mod

    monkeypatch.setattr(router_mod, "_is_self_healing_active", lambda: False)
    monkeypatch.setattr(
        router_mod,
        "_handle_session",
        AsyncMock(
            return_value={
                "session_id": "sess-1",
                "session_id_header": "sess-1",
                "session_explicit": True,
                "session_created": False,
                "is_delta_request": False,
                "session_fallback_reason": None,
                "delta_messages": None,
                "original_message_count": 1,
                "body_json": {
                    "model": "plan",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                "body_override": None,
            }
        ),
    )
    monkeypatch.setattr(router_mod, "_build_slot_context", lambda *_: (None, None, 3.0))
    monkeypatch.setattr(router_mod, "_resolve_slot_model_name", lambda model, *_: model)
    monkeypatch.setattr(router_mod, "_check_slot_availability", AsyncMock(return_value=None))


@pytest.mark.asyncio
async def test_router_duplicate_returns_409_with_retry_after(monkeypatch):
    from proxy.router import proxy_to_local
    from proxy.session import SessionSingleFlightDuplicateError

    server_cfg = {
        "server": {
            "llama_server_port": 8080,
            "session_slot_pool_size": 3,
            "max_concurrent_queries": 16,
            "session_single_flight_mode": "queue",
            "session_single_flight_max_queue_depth": 1,
            "session_single_flight_duplicate_retry_after_seconds": 12,
        }
    }
    _install_router_harness(monkeypatch, server_cfg)

    import proxy.router as router_mod

    class _DuplicateGuard:
        async def __aenter__(self):
            raise SessionSingleFlightDuplicateError("duplicate_inflight")

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(
        router_mod.session_single_flight_coordinator,
        "acquire",
        lambda *a, **k: _DuplicateGuard(),
    )

    req = _DummyRequest(
        json.dumps(
            {
                "model": "plan",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            }
        ).encode("utf-8")
    )
    resp = await proxy_to_local(req, "v1/chat/completions")

    assert resp.status_code == 409
    payload = json.loads(resp.body)
    assert payload["error"]["code"] == "duplicate_inflight"
    assert resp.headers.get("Retry-After") == "12"


@pytest.mark.asyncio
async def test_router_lease_denial_carries_retry_after_header(monkeypatch):
    from proxy.router import proxy_to_local

    server_cfg = {
        "server": {
            "llama_server_port": 8080,
            "session_slot_pool_size": 1,
            "max_concurrent_queries": 16,
            "local_dispatch_lease_timeout_seconds": 30,
            "session_single_flight_mode": "bypass",
        }
    }
    _install_router_harness(monkeypatch, server_cfg)

    # Deny the dispatch (pool occupied by another session).
    import proxy.router as router_mod

    async def _deny(*args, **kwargs):
        return (False, "other-session", 1, 7.0)

    monkeypatch.setattr(router_mod, "_try_acquire_local_dispatch", _deny)

    req = _DummyRequest(
        json.dumps(
            {
                "model": "plan",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            }
        ).encode("utf-8")
    )
    resp = await proxy_to_local(req, "v1/chat/completions")

    assert resp.status_code == 503
    payload = json.loads(resp.body)
    assert payload["reason"] == "local_lease_active"
    assert resp.headers.get("Retry-After") == "7"
