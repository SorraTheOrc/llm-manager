"""Routing occupancy must not count the requesting session's own idle lease.

LP-0MW0VH2LB008JSKQ. Two gates decide local admission and they disagreed about
the requesting session's own dispatch record:

- the **routing** gate (``_proxy_with_fallback_cycle`` →
  ``_local_concurrency_info``) counted *every* active-or-unexpired record,
  including the requester's own, and fell through to a remote provider with
  ``status=local_concurrency_limit`` when the total reached
  ``session_slot_pool_size``;
- the **lease** gate (``_try_acquire_local_dispatch``) deliberately permits
  same-session re-acquisition (``own_has_lease``), so it would have granted the
  slot.

Because the routing gate fired first, the lease gate's exemption was
unreachable: a session returning inside its own 30 s no-preemption idle lease
was pushed remote while *nothing was generating* (measured 2026-11-03:
15 of 82 ``local_concurrency_limit`` fall-throughs, with
``herdr-1793718235`` pushed remote on four consecutive turns 4.5–26.2 s after
its own local turn).

The rule pinned here: the requester's own record is excluded from the routing
count **only while it is an idle lease** (``active`` false). A record that is
still ``active`` is genuine in-flight work and keeps counting, so concurrent
requests within one session are gated exactly as before.
"""

from __future__ import annotations

import json
import logging
import time
from unittest.mock import AsyncMock, patch

import proxy.provider as provider
import proxy.server as server
import pytest
from starlette.responses import Response

ENDPOINT = "http://localhost:8080"
POOL = 2
ME = "session-me"
OTHER = "session-other"
OTHER2 = "session-other-2"


def _config(**server_overrides) -> dict:
    server_cfg = {
        "llama_server_port": 8080,
        "session_slot_pool_size": POOL,
        "local_dispatch_lease_timeout_seconds": 30,
        "max_concurrent_queries": 8,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": 30,
        "contention_queue_max_depth": 4,
        "local_prefill_deadline_seconds": 0,
        "local_cold_prefill_budget_tokens": 25000,
    }
    server_cfg.update(server_overrides)
    return {"server": server_cfg, "provider_cooldown_seconds": 60}


def _record(*, active: bool = False, expires_in: float = 25.0, model: str = "Qwen3") -> dict:
    now = time.monotonic()
    return {
        "backend": "local",
        "started_at": now,
        "active": active,
        "expires_at": now + expires_in,
        "model_name": model,
    }


@pytest.fixture
def install_records(monkeypatch):
    """Install dispatch records on the ``proxy.server`` module state."""

    def _install(records: dict, generating: int = 0) -> dict:
        monkeypatch.setattr(server, "local_dispatch_records", records, raising=False)
        monkeypatch.setattr(server, "local_generating_queries", generating, raising=False)
        return records

    return _install


# ======================================================================
# The count itself
# ======================================================================


def test_own_idle_lease_is_excluded_from_the_routing_count(install_records):
    """AC1: own idle lease + one other idle lease on a 2-slot pool counts 1."""
    install_records(
        {
            (ENDPOINT, ME): _record(active=False),
            (ENDPOINT, OTHER): _record(active=False),
        }
    )

    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    ) == (1, POOL)


def test_without_a_requester_the_count_is_unchanged(install_records):
    """AC3: anonymous/legacy callers exclude nothing — byte-identical count."""
    install_records(
        {
            (ENDPOINT, ME): _record(active=False),
            (ENDPOINT, OTHER): _record(active=False),
        }
    )

    assert provider._get_local_concurrency_info(_config(), endpoint=ENDPOINT) == (POOL, POOL)
    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=None
    ) == (POOL, POOL)


def test_two_other_sessions_still_saturate_the_pool(install_records):
    """AC2: genuine occupancy is never weakened by the exclusion."""
    install_records(
        {
            (ENDPOINT, OTHER): _record(active=False),
            (ENDPOINT, OTHER2): _record(active=False),
        }
    )

    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    ) == (POOL, POOL)


def test_own_active_record_still_counts(install_records):
    """A session with work in flight is genuinely occupying its slot.

    Excluding an *active* record would hide real concurrency, so only the idle
    (``active=False``) lease is subtracted.
    """
    install_records(
        {
            (ENDPOINT, ME): _record(active=True),
            (ENDPOINT, OTHER): _record(active=False),
        }
    )

    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    ) == (POOL, POOL)


def test_expired_records_are_not_counted_at_all(install_records):
    """An expired record is not occupancy, own or otherwise."""
    install_records(
        {
            (ENDPOINT, ME): _record(active=False, expires_in=-5.0),
            (ENDPOINT, OTHER): _record(active=False, expires_in=-5.0),
        }
    )

    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    ) == (0, POOL)


def test_legacy_plain_keys_exclude_the_own_idle_lease(install_records):
    """AC4: legacy plain-key records obey the same rule in a mixed registry.

    A *pure* legacy registry (no tuple keys at all) does not iterate records at
    all — the global ``local_generating_queries`` counter is authoritative
    there, and that path is deliberately unchanged. Plain-key records are
    counted alongside tuple-keyed ones only once the registry is
    endpoint-aware (any tuple key present), which is the mixed shape tested
    here: the own plain-key idle lease is excluded, another session's is not,
    and a tuple-keyed record for a *different* endpoint is not counted.
    """
    install_records(
        {
            ("http://localhost:9999", "elsewhere"): _record(active=True),
            ME: _record(active=False),
            OTHER: _record(active=False),
        }
    )

    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    ) == (1, POOL)
    assert provider._get_local_concurrency_info(_config(), endpoint=ENDPOINT) == (POOL, POOL)


def test_pure_legacy_registry_keeps_the_global_counter(install_records):
    """AC3/AC4: a pure legacy registry still uses the global counter unchanged."""
    install_records({ME: _record(active=False), OTHER: _record(active=False)}, generating=2)

    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    ) == (POOL, POOL)


def test_another_endpoints_own_record_does_not_leak(install_records):
    """AC4: only the matching ``(endpoint, session)`` pair is excluded."""
    install_records(
        {
            ("http://localhost:9999", ME): _record(active=False),
            (ENDPOINT, OTHER): _record(active=False),
        }
    )

    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    ) == (1, POOL)


def test_wrapper_forwards_session_id_only_when_accepted(monkeypatch):
    """The signature-aware wrapper must not break ``(config)``-only stubs."""
    calls: list[tuple] = []

    def config_only(config):  # no endpoint, no session_id
        calls.append(("config",))
        return (7, 9)

    monkeypatch.setattr(provider, "_get_local_concurrency_info", config_only)
    assert provider._local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    ) == (7, 9)

    def with_session(config, endpoint=None, session_id=None):
        calls.append(("session", endpoint, session_id))
        return (1, 2)

    monkeypatch.setattr(provider, "_get_local_concurrency_info", with_session)
    assert provider._local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    ) == (1, 2)
    assert calls[-1] == ("session", ENDPOINT, ME)


# ======================================================================
# End-to-end through the routing gate
# ======================================================================


class _Request:
    """Minimal request stub carrying the session header the gate reads."""

    def __init__(self, session_id: str | None = ME):
        headers = {"host": "localhost"}
        if session_id:
            headers["x-session-id"] = session_id
        self.headers = headers
        self._body = json.dumps(
            {"model": "code", "messages": [{"role": "user", "content": "hi"}]}
        ).encode("utf-8")
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


def _model_config() -> dict:
    return {
        "providers": [
            {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"},
            {"name": "opencode-go-2", "type": "remote", "endpoint": "https://api.example.com/v1"},
        ]
    }


def _ok() -> Response:
    return Response(
        content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
        status_code=200,
        media_type="application/json",
    )


async def _route(monkeypatch, calls: list[str]) -> Response:
    """Drive ``proxy_with_fallback`` past routing with local-looking conditions."""

    async def _local(_req, _path):
        calls.append("local")
        return _ok()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok()

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider.request_priority", return_value="low"),
        patch(
            "proxy.provider._estimate_effective_prompt_tokens_for_routing",
            new=AsyncMock(return_value=2_000),
        ),
        patch("proxy.provider._get_tokenizer_for_model", lambda *_: (None, 1.0)),
        # Warm cache + small prompt: nothing but occupancy may divert the request.
        patch("proxy.provider._get_cached_ratio", lambda *_: 0.95),
        patch("proxy.provider._should_skip_local", lambda *a, **k: (False, None)),
    ):
        return await provider.proxy_with_fallback(
            _Request(), "v1/chat/completions", _model_config(), _config()
        )


@pytest.mark.asyncio
async def test_returning_session_is_served_locally_with_its_own_lease(
    monkeypatch, install_records, caplog
):
    """AC1/AC6: own idle lease + one other idle lease ⇒ still local.

    This is the production shape: both slots are held by *finished* turns
    inside their 30 s no-preemption lease, so nothing is generating, and the
    returning session is entitled to the slot its lease reserved.
    """
    caplog.set_level(logging.INFO, logger="llama-proxy")
    install_records(
        {
            (ENDPOINT, ME): _record(active=False),
            (ENDPOINT, OTHER): _record(active=False),
        }
    )
    calls: list[str] = []
    monkeypatch.setattr(provider, "_get_local_provider_endpoint", lambda *a, **k: ENDPOINT)

    result = await _route(monkeypatch, calls)

    assert result.status_code == 200
    assert calls == ["local"], f"the returning session must be served locally, got {calls}"
    assert "local_concurrency_limit" not in caplog.text


@pytest.mark.asyncio
async def test_a_genuinely_full_pool_still_falls_through(monkeypatch, install_records, caplog):
    """AC2/AC5: two *other* sessions' leases keep today's fall-through."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    install_records(
        {
            (ENDPOINT, OTHER): _record(active=False),
            (ENDPOINT, OTHER2): _record(active=False),
        }
    )
    calls: list[str] = []
    monkeypatch.setattr(provider, "_get_local_provider_endpoint", lambda *a, **k: ENDPOINT)

    result = await _route(monkeypatch, calls)

    assert result.status_code == 200
    assert calls == ["remote"], "a full pool must still route to the next provider"
    assert "reason=local_concurrency_limit" in caplog.text
