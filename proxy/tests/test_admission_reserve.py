"""Atomic local-admission reservations + explicit fallback intent
(LP-0MVIKPCSG002B0NL).

Covers:

- AC1: reserve returns a ticket when the occupancy-aware projected wait fits
  the budget, and a follow-up request with the ticket dispatches local without
  queueing (no ``contention_queue_wait`` log for that request).
- AC2: reserve returns 429 with ``reason``/``queue_eta_s``/``budget_s``/
  ``retry_after_s`` when the projected wait exceeds the budget; the refused
  request never enters the contention queue.
- AC3: an expired or cancelled ticket is not honoured (falls back to the
  ordinary path); a cancelled reservation releases the gate for the next
  ticket.
- AC4: ``X-Local-Preference: local-only`` turns a would-be remote fallback
  into the retryable 429, even while remote providers are available.
- AC5: ``remote-ok`` never attempts the local route.
- AC6: route headers ``X-Local-Route: local|remote|refused`` and
  ``X-Local-Fallback-Reason``.
- AC7: the reserve verdict and the admission-guard verdict are computed by the
  same function — identical outcomes for identical inputs.
- AC8: fail-open/backcompat — unknown preference values, missing fields and a
  missing ticket behave exactly as today; an uncalibrated reserve admits.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import proxy.provider as provider
import pytest
from fastapi import Response

from proxy import admission_reserve, contention_queue, prefill_eta

#: Tokens whose calibrated own-ETA (~56.2 s) fits the 60 s budget while the
#: projected wait with one occupant (~112.4 s) does not (same fixture as the
#: occupancy test).
_TOKENS = 12_000
_BUDGET = 60.0


class _DummyRequest:
    """Minimal request stub (matches the admission-guard test helpers)."""

    def __init__(self, body: bytes = b'{"model":"test"}', headers: dict | None = None):
        self._body = body
        self.headers = headers or {}
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


@pytest.fixture(autouse=True)
def _clean_state():
    contention_queue.reset()
    prefill_eta.reset()
    admission_reserve.reset()
    yield
    contention_queue.reset()
    prefill_eta.reset()
    admission_reserve.reset()


def _seed_calibration(
    tokens: int = _TOKENS, ms_per_token: float = 3.9, count: int = 6
) -> None:
    for _ in range(count):
        prefill_eta.record_dispatch_sample("Qwen3", tokens, 0.0, tokens * ms_per_token)


@contextmanager
def _estimate_patches(tokens: int = _TOKENS, cached_ratio: float = 0.0):
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
# AC1 — reserve end-to-end (ticket → follow-up dispatch, no queueing)
# ---------------------------------------------------------------------------

def test_reserve_returns_ticket_when_projected_wait_fits_budget():
    _seed_calibration()
    status, payload = admission_reserve.reserve(
        {"estimated_tokens": _TOKENS, "session_id": "s-1"},
        _cfg(),
        model="Qwen3",
        busy_slots=0,
    )

    assert status == 200
    assert payload["admit"] is True
    assert payload["reservation_id"]
    assert payload["ttl_s"] == admission_reserve.DEFAULT_RESERVE_TTL_SECONDS


@pytest.mark.asyncio
async def test_follow_up_with_ticket_dispatches_local_without_queueing(caplog):
    """AC1 end-to-end: the ticket bypasses admission re-check and the queue."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_calibration()
    # Gate is occupied (busy=1 ⇒ projected 112 s > budget), so without the
    # ticket this request would be refused. The ticket was reserved for an
    # idle gate (busy=0 ⇒ own ETA 56.2 s < budget) and is honoured now.
    status, payload = admission_reserve.reserve(
        {"estimated_tokens": _TOKENS, "session_id": "s-1"},
        _cfg(),
        model="Qwen3",
        busy_slots=0,
    )
    assert status == 200

    request = _DummyRequest(headers={"X-Local-Reservation": payload["reservation_id"]})
    with (
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(1, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _estimate_patches(),
    ):
        action, reason, elapsed = await provider._maybe_queue_for_local_slot(
            _cfg(),
            1,
            1,
            request,
            {},
            {},
            {"name": "local-llama", "llama_model": "Qwen3"},
            "s-1",
            priority="low",
        )

    assert action == "dispatch"
    assert reason == "reserved"
    assert elapsed == 0.0
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "local_admission_reserve_claimed" in messages
    assert "contention_queue_wait" not in messages, "reserved dispatch must not queue"
    # Single-use: the ticket is consumed by the honoured follow-up.
    assert admission_reserve.peek(payload["reservation_id"]) is None


# ---------------------------------------------------------------------------
# AC2 — reserve refusal (429 with reasons; never enters the queue)
# ---------------------------------------------------------------------------

def test_reserve_refuses_with_full_payload_when_wait_exceeds_budget():
    _seed_calibration()
    status, payload = admission_reserve.reserve(
        {"estimated_tokens": _TOKENS, "session_id": "s-1"},
        _cfg(),
        model="Qwen3",
        busy_slots=1,  # occupied gate ⇒ projected ~112 s > 60 s budget
    )

    assert status == 429
    assert payload["admit"] is False
    assert payload["reason"] == "eta_exceeds_budget"
    assert payload["queue_eta_s"] > _BUDGET
    assert payload["budget_s"] == _BUDGET
    assert isinstance(payload["retry_after_s"], int)
    assert payload["retry_after_s"] >= 1


@pytest.mark.asyncio
async def test_refused_reserve_request_never_enters_the_contention_queue(caplog):
    """AC2 end-to-end: a refused refund request has no queue side effects."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_calibration()
    status, payload = admission_reserve.reserve(
        {"estimated_tokens": _TOKENS}, _cfg(), model="Qwen3", busy_slots=1
    )
    assert status == 429
    assert "reservation_id" not in payload
    assert contention_queue.queue_depth() == 0
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "local_admission_reserve admit=false reason=eta_exceeds_budget" in messages


# ---------------------------------------------------------------------------
# AC3 — TTL and cancellation
# ---------------------------------------------------------------------------

def test_expired_ticket_is_not_honoured(monkeypatch):
    import proxy.admission_reserve as ar_mod

    # Ticket created at monotonic 0 with a 2 s TTL.
    admission_reserve._reservations["t-expired"] = admission_reserve.Reservation(
        ticket_id="t-expired",
        estimated_tokens=_TOKENS,
        cached_ratio=0.0,
        priority=None,
        provider="Qwen3",
        session_id=None,
        created_at=0.0,
        ttl_seconds=2.0,
    )
    monkeypatch.setattr(ar_mod.time, "monotonic", lambda: 1.0)
    assert ar_mod.peek("t-expired") is not None, "ticket valid inside its TTL"

    # Fast-forward past the TTL: the ticket must not be honoured.
    monkeypatch.setattr(ar_mod.time, "monotonic", lambda: 3.0)
    assert ar_mod.peek("t-expired") is None, "expired ticket must not be honoured"
    assert ar_mod.claim("t-expired") is None, "expired claim falls back (AC3)"


def test_cancelled_ticket_is_not_honoured_and_releases_the_gate():
    _seed_calibration()
    status, payload = admission_reserve.reserve(
        {"estimated_tokens": _TOKENS, "session_id": "s-1"},
        _cfg(),
        model="Qwen3",
        busy_slots=0,
    )
    ticket = payload["reservation_id"]

    cancel_status, cancel_payload = admission_reserve.cancel(ticket)
    assert cancel_status == 200
    assert cancel_payload["cancelled"] is True
    assert admission_reserve.peek(ticket) is None

    # The next client can reserve again (the gate claim was released).
    status2, payload2 = admission_reserve.reserve(
        {"estimated_tokens": _TOKENS, "session_id": "s-2"},
        _cfg(),
        model="Qwen3",
        busy_slots=0,
    )
    assert status2 == 200
    assert payload2["reservation_id"] != ticket


def test_cancel_unknown_ticket_returns_404():
    status, payload = admission_reserve.cancel("does-not-exist")
    assert status == 404
    assert payload["cancelled"] is False


def test_missing_ticket_header_takes_the_ordinary_path():
    _seed_calibration()
    status, payload = admission_reserve.reserve(
        {"estimated_tokens": _TOKENS}, _cfg(), model="Qwen3", busy_slots=0
    )
    # A follow-up WITHOUT the header has no reservation.
    assert admission_reserve.claim(None) is None


# ---------------------------------------------------------------------------
# AC7 — single estimator: reserve verdict == guard verdict
# ---------------------------------------------------------------------------

def test_reserve_verdict_matches_admission_guard_for_identical_inputs():
    _seed_calibration()
    for busy in (0, 1):
        reserve_status, reserve_payload = admission_reserve.reserve(
            {"estimated_tokens": _TOKENS}, _cfg(), model="Qwen3", busy_slots=busy
        )
        guard = prefill_eta.evaluate_admission(
            _cfg(), "Qwen3", _TOKENS, 0.0, "medium", busy_slots=busy
        )
        assert reserve_status == (200 if guard.admit else 429)
        if guard.admit:
            assert reserve_payload["admit"] is True
        else:
            assert reserve_payload["reason"] == guard.reason
            assert reserve_payload["queue_eta_s"] == pytest.approx(
                guard.projected_wait_seconds, abs=0.1
            )


# ---------------------------------------------------------------------------
# AC8 — fail-open / backcompat
# ---------------------------------------------------------------------------

def test_reserve_missing_or_invalid_fields_are_400():
    status, payload = admission_reserve.reserve({}, _cfg())
    assert status == 400
    assert "estimated_tokens" in payload.get("error", "")

    status, payload = admission_reserve.reserve(
        {"estimated_tokens": -5}, _cfg()
    )
    assert status == 400


def test_uncalibrated_reserve_admits_even_with_full_occupancy():
    # No samples seeded ⇒ inert guard ⇒ admit (never reject on a guess).
    status, payload = admission_reserve.reserve(
        {"estimated_tokens": _TOKENS}, _cfg(), model="Qwen3", busy_slots=8
    )
    assert status == 200
    assert payload["admit"] is True
    assert payload["reservation_id"]
    # The verdict's reason shows the guard was inert, not calibrated.
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", _TOKENS, 0.0, None, busy_slots=8
    )
    assert decision.admit is True
    assert decision.reason == "insufficient_samples"


def test_unknown_preference_value_falls_back_to_prefer_local():
    assert (
        admission_reserve.parse_preference({"X-Local-Preference": "purple"})
        == "prefer-local"
    )
    assert admission_reserve.parse_preference({}) == "prefer-local"
    assert (
        admission_reserve.parse_preference({"X-Local-Preference": "remote-ok"})
        == "remote-ok"
    )


def test_dry_run_returns_verdict_without_claiming():
    _seed_calibration()
    status, payload = admission_reserve.reserve(
        {"estimated_tokens": _TOKENS, "dry_run": True},
        _cfg(),
        model="Qwen3",
        busy_slots=0,
    )
    assert status == 200
    assert payload["dry_run"] is True
    assert "reservation_id" not in payload


# ---------------------------------------------------------------------------
# AC4 — local-only: 429 instead of remote fallback (remotes available)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_local_only_refuses_429_when_guard_would_fall_back_remote(
    caplog,
):
    """AC4: with remotes available, local-only still yields the 429."""
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
    request = _DummyRequest(headers={"X-Local-Preference": "local-only"})
    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(1, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _estimate_patches(),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", config, _cfg()
        )

    # Occupied gate + 20k tokens ⇒ the guard refuses; local-only must NOT
    # proceed to the remote provider.
    assert calls == []
    assert result.status_code == 429
    assert result.headers.get("X-Local-Route") == "refused"
    assert result.headers.get("X-Local-Fallback-Reason") == "eta_exceeds_budget"
    assert result.headers.get("Retry-After")
    assert result.headers.get("X-Local-Admission") == "reject"
    body = json.loads(result.body)
    assert body["code"] == "local_admission_reject"
    assert body["retry_after"] >= 1


@pytest.mark.asyncio
async def test_prefer_local_still_falls_back_remote_when_refused(caplog):
    """Backcompat (AC8): prefer-local (and absent header) keep the fallback."""
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
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(1, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _estimate_patches(),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", config, _cfg()
        )

    assert calls == ["remote"], "prefer-local keeps today's remote fallback"
    assert result.status_code == 200


# ---------------------------------------------------------------------------
# AC5 — remote-ok never attempts the local route
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_remote_ok_never_attempts_local(caplog):
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
    request = _DummyRequest(headers={"X-Local-Preference": "remote-ok"})
    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(0, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _estimate_patches(),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", config, _cfg()
        )

    assert calls == ["remote"]
    assert "local" not in calls
    assert result.status_code == 200
    assert result.headers.get("X-Local-Route") == "remote"


@pytest.mark.asyncio
async def test_remote_ok_with_local_only_chain_never_calls_local(caplog):
    """AC5: even a local-only chain is not attempted under remote-ok."""
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
        ],
        "aliases": ["hybrid*"],
    }
    request = _DummyRequest(headers={"X-Local-Preference": "remote-ok"})
    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(0, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _estimate_patches(),
        patch("proxy.router_helpers.evaluate_and_apply_compaction", AsyncMock()),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", config, _cfg()
        )

    assert calls == [], "local must never be attempted under remote-ok"
    # The empty chain exhausts: 503 all_exhausted, not a local call.
    assert result.status_code in (503, 429)


# ---------------------------------------------------------------------------
# AC6 — route headers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_local_success_carries_route_local(caplog):
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
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(0, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _estimate_patches(tokens=2_000),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", config, _cfg()
        )

    assert calls == ["local"]
    assert result.status_code == 200
    assert result.headers.get("X-Local-Route") == "local"


@pytest.mark.asyncio
async def test_remote_success_carries_route_remote_and_reason(caplog):
    """AC6: remote-served responses carry the recorded fallback reason."""
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
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(1, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        # fallback policy: local slot exhausted ⇒ fall through immediately
        _estimate_patches(),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", config,
            _cfg(contention_queue_policy="fallback"),
        )

    assert calls == ["remote"]
    assert result.status_code == 200
    assert result.headers.get("X-Local-Route") == "remote"
    assert result.headers.get("X-Local-Fallback-Reason")


# ---------------------------------------------------------------------------
# Route registration (the endpoint exists and is additive)
# ---------------------------------------------------------------------------

def test_reserve_routes_are_registered():
    from proxy.handlers import router

    paths = {route.path for route in router.routes}
    assert "/llama/local/reserve" in paths
    assert "/llama/local/reserve/{reservation_id}" in paths
