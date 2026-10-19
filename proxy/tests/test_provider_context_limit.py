"""Contract tests for the per-entry ``max_context_tokens`` gate
(LP-0MVF9MY0W002J2N3 tests, LP-0MVF9N2QL001CDA7 gate, parent
LP-0MVF2DNCX003Z5CO).

These tests were written test-first (F1) as ``xfail(strict=True)`` and the
markers were removed by F2 when the gate landed, so they are now the live
regression suite for the per-entry context ceiling.

Agreed vocabulary (parent plan Appendix):

- provider-entry field: ``max_context_tokens``
- remote skip status: ``provider_context_exceeded`` (attempt carries
  ``estimated_tokens`` and the entry ``limit``)
- helper contract pinned by these tests::

      _provider_context_limit(provider_cfg) -> int          # 0 = ungated
      _should_skip_provider_context(provider_cfg, estimated_tokens)
          -> (skip: bool, limit: int)                       # limit 0 = ungated

F1 AC coverage:

- (a) an entry above its limit is skipped with the correct reason
- (b) an entry at or below its limit is dispatched
- (c) absent / ``0`` (and any non-positive) limit is ungated
- (d) a skip falls through to the next provider
- (e) no cooldown / affinity / usage-limit state after a skip
- (f) local chains are unaffected (field absent = byte-for-byte today; a
  local limit only ever tightens the existing hard cap)
- (g) config parsing of the new field (tolerant reader, mirroring
  ``rate_limit_rpm`` parsing)
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import proxy.provider as provider
import proxy.provider_budget as provider_budget
import pytest
from fastapi import Response
from proxy.provider_budget import BudgetDecision

_OK_BODY = json.dumps({"choices": [{"message": {"content": "ok"}}]})


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_router_state(monkeypatch):
    """Reset cooldown/usage/affinity state and pin a deterministic priority.

    Mirrors the autouse fixture in ``test_provider_fallback.py``: provider
    cooldowns, usage-limit quarantines, sibling-failure streaks and the
    free-tier affinity store are module-level state that would otherwise leak
    between tests (and make "no state recorded" assertions meaningless).
    """
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()
    provider_budget._affinity.clear()
    # Both bindings matter: ``proxy.provider`` imports the callable into its own
    # namespace, while ``effective_chain_for_request`` resolves it from
    # ``proxy.priority``. Pinning both keeps the cycle deterministic inside the
    # cheap window (01:00-10:00 UTC), where an unpinned request caps to ``low``.
    monkeypatch.setattr(provider, "request_priority", lambda *a, **k: "medium")
    monkeypatch.setattr("proxy.priority.request_priority", lambda *a, **k: "medium")
    yield
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()
    provider_budget._affinity.clear()


class _Request:
    """Minimal request stub (body + session header) for the fallback cycles."""

    def __init__(self, body: bytes | None = None, session_id: str | None = "ctx-limit-session"):
        self._body = body if body is not None else json.dumps(
            {"model": "ctx-limit-test", "messages": [{"role": "user", "content": "hi"}]}
        ).encode()
        self.headers = {"content-type": "application/json"}
        if session_id:
            self.headers["x-session-id"] = session_id
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


def _remote_entry(name: str, **extra) -> dict:
    entry = {
        "name": name,
        "type": "remote",
        "endpoint": f"https://{name}.example.com/v1",
        "api_key_env": "TEST_CONTEXT_LIMIT_KEY",
    }
    entry.update(extra)
    return entry


def _remote_ptr(calls: list[str]):
    async def _ptr(_request, _path, provider_cfg):
        calls.append(provider_cfg["name"])
        return Response(content=_OK_BODY, status_code=200, media_type="application/json")

    return _ptr


def _budget_spy(calls: list[str]):
    async def _budget(provider_cfg, config, session_id, request=None):
        calls.append(provider_cfg.get("name"))
        return BudgetDecision(allowed=True, reason="unlimited")

    return _budget


def _patch_remote_cycle(
    monkeypatch,
    *,
    providers: list[dict],
    estimate: int | AsyncMock,
    remote_calls: list[str],
    budget_calls: list[str],
) -> tuple[dict, AsyncMock]:
    """Wire a deterministic remote fallback cycle; returns (model_config, estimate_mock)."""
    model_config = {"name": "ctx-limit-test", "providers": providers}
    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: _remote_ptr(remote_calls))
    monkeypatch.setattr(provider, "acquire_provider_budget", _budget_spy(budget_calls))
    monkeypatch.setattr(provider, "_get_tokenizer_for_model", MagicMock(return_value=(None, 1.0)))
    if isinstance(estimate, AsyncMock):
        estimate_mock = estimate
    else:
        estimate_mock = AsyncMock(return_value=estimate)
    monkeypatch.setattr(
        provider, "_estimate_effective_prompt_tokens_for_routing", estimate_mock
    )
    return model_config, estimate_mock


@pytest.fixture
def local_and_remote_calls(monkeypatch):
    """Deterministic local+remote cycle harness (returns call recorders).

    The local branch's routing estimate + thresholds are stubbed so the test
    isolates the per-entry context gate: cold threshold high enough that the
    economic bypass never fires, warm threshold 100000 (today's shipped value).
    """
    state = {"local_calls": [], "remote_calls": [], "budget_calls": []}

    async def _local_ptr(_request, _path, endpoint=None):
        state["local_calls"].append(endpoint)
        return Response(content=_OK_BODY, status_code=200, media_type="application/json")

    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: _local_ptr)
    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: _remote_ptr(state["remote_calls"]))
    monkeypatch.setattr(
        provider, "acquire_provider_budget", _budget_spy(state["budget_calls"])
    )
    monkeypatch.setattr(provider, "_get_tokenizer_for_model", MagicMock(return_value=(None, 1.0)))
    monkeypatch.setattr(provider, "_get_cached_ratio", lambda *_: 0.0)
    monkeypatch.setattr(
        provider, "_effective_large_context_thresholds", lambda *_: (1_000_000, 100_000)
    )
    monkeypatch.setattr(provider, "_local_concurrency_info", lambda *a, **k: (0, 4))
    monkeypatch.setattr(
        provider, "_get_local_provider_endpoint", lambda *a, **k: "http://127.0.0.1:8080"
    )
    monkeypatch.setattr(
        provider,
        "_evaluate_compaction_for_bypass",
        AsyncMock(
            return_value={
                "evaluated": True,
                "action": "noop",
                "applied": False,
                "estimated_before": 0,
                "estimated_after": 0,
                "reason": None,
            }
        ),
    )
    return state


def _local_entry(**extra) -> dict:
    entry = {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"}
    entry.update(extra)
    return entry


# ---------------------------------------------------------------------------
# (g) Config parsing: _provider_context_limit
# ---------------------------------------------------------------------------


def test_context_limit_positive_int_is_parsed():
    """A positive ``max_context_tokens`` is the entry's estimate ceiling."""
    assert provider._provider_context_limit({"max_context_tokens": 128_000}) == 128_000
    assert provider._provider_context_limit({"max_context_tokens": 1}) == 1


def test_context_limit_absent_is_ungated():
    """An entry without the field is ungated (0), byte-for-byte as today."""
    assert provider._provider_context_limit({}) == 0
    assert provider._provider_context_limit({"name": "open"}) == 0
    assert provider._provider_context_limit({"max_context_tokens": None}) == 0


def test_context_limit_zero_and_negative_are_ungated():
    """``0`` and any non-positive value disable the gate (AC1 contract)."""
    assert provider._provider_context_limit({"max_context_tokens": 0}) == 0
    assert provider._provider_context_limit({"max_context_tokens": -1}) == 0
    assert provider._provider_context_limit({"max_context_tokens": -128_000}) == 0


def test_context_limit_malformed_value_is_ungated_not_fatal():
    """A malformed value must fail open (mirrors ``rate_limit_rpm`` parsing)."""
    assert provider._provider_context_limit({"max_context_tokens": "many"}) == 0
    assert provider._provider_context_limit({"max_context_tokens": {"nested": 1}}) == 0
    assert provider._provider_context_limit({"max_context_tokens": [1]}) == 0


def test_context_limit_numeric_string_is_tolerated():
    """A numeric string parses like ``int()`` does for ``rate_limit_rpm``."""
    assert provider._provider_context_limit({"max_context_tokens": "64000"}) == 64_000


# ---------------------------------------------------------------------------
# (a)(b)(c) Gate helper: _should_skip_provider_context
# ---------------------------------------------------------------------------


def test_gate_skips_entry_above_its_limit():
    """Above the limit -> skip, with the evaluated limit returned."""
    skip, limit = provider._should_skip_provider_context(
        {"max_context_tokens": 50_000}, 50_001
    )
    assert skip is True
    assert limit == 50_000


def test_gate_allows_entry_at_its_limit():
    """At the limit is allowed: the ceiling is inclusive (AC1: at or below)."""
    skip, limit = provider._should_skip_provider_context(
        {"max_context_tokens": 50_000}, 50_000
    )
    assert skip is False
    assert limit == 50_000


def test_gate_allows_entry_below_its_limit():
    skip, limit = provider._should_skip_provider_context(
        {"max_context_tokens": 50_000}, 49_999
    )
    assert skip is False
    assert limit == 50_000


@pytest.mark.parametrize("field_value", [None, 0, -10])
def test_gate_never_skips_an_ungated_entry(field_value):
    """Absent / 0 / non-positive -> ungated, however large the estimate."""
    entry = {} if field_value is None else {"max_context_tokens": field_value}
    skip, limit = provider._should_skip_provider_context(entry, 10_000_000)
    assert skip is False
    assert limit == 0


@pytest.mark.parametrize("estimated", [None, 0, -1])
def test_gate_never_skips_without_a_positive_estimate(estimated):
    """No usable estimate -> no skip (fail open, like the local gate)."""
    skip, limit = provider._should_skip_provider_context({"max_context_tokens": 10}, estimated)
    assert skip is False
    assert limit == 10


# ---------------------------------------------------------------------------
# (b)(c) Remote fallback cycle: dispatch when at/below limit or ungated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("estimated", [50_000, 40_000])
async def test_remote_entry_at_or_below_limit_is_dispatched(monkeypatch, estimated):
    """At or below the limit the entry is dispatched normally (AC1: at/below)."""
    remote_calls: list[str] = []
    budget_calls: list[str] = []
    model_config, _ = _patch_remote_cycle(
        monkeypatch,
        providers=[
            _remote_entry("capped", max_context_tokens=50_000, rate_limit_rpm=5),
            _remote_entry("open"),
        ],
        estimate=estimated,
        remote_calls=remote_calls,
        budget_calls=budget_calls,
    )

    response = await provider._proxy_with_remote_fallback_cycle(
        _Request(), "v1/chat/completions", model_config, {"provider_cooldown_seconds": 60}
    )

    assert response.status_code == 200
    assert remote_calls == ["capped"]
    assert budget_calls == ["capped"]


@pytest.mark.asyncio
@pytest.mark.parametrize("field_value", [None, 0, -1])
async def test_ungated_remote_entry_dispatches_at_any_size(monkeypatch, field_value):
    """Absent / ``0`` / negative limit stays ungated for an oversized request."""
    remote_calls: list[str] = []
    budget_calls: list[str] = []
    extra = {} if field_value is None else {"max_context_tokens": field_value}
    model_config, _ = _patch_remote_cycle(
        monkeypatch,
        providers=[_remote_entry("open", **extra), _remote_entry("tail")],
        estimate=1_000_000,
        remote_calls=remote_calls,
        budget_calls=budget_calls,
    )

    response = await provider._proxy_with_remote_fallback_cycle(
        _Request(), "v1/chat/completions", model_config, {"provider_cooldown_seconds": 60}
    )

    assert response.status_code == 200
    assert remote_calls == ["open"]
    assert budget_calls == ["open"]


# ---------------------------------------------------------------------------
# (a)(d) Remote fallback cycle: above-limit skip + fall-through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remote_entry_above_limit_is_skipped_before_budget_and_next_serves(monkeypatch):
    """Above the limit the entry is skipped BEFORE ``acquire_provider_budget``.

    AC2/AC6: the gate runs before the budget check and dispatch, so an
    oversized request never consumes a rate-limit slot (queue-safe).
    """
    remote_calls: list[str] = []
    budget_calls: list[str] = []
    model_config, _ = _patch_remote_cycle(
        monkeypatch,
        providers=[
            _remote_entry("capped", max_context_tokens=50_000, rate_limit_rpm=5),
            _remote_entry("open"),
        ],
        estimate=60_000,
        remote_calls=remote_calls,
        budget_calls=budget_calls,
    )

    response = await provider._proxy_with_remote_fallback_cycle(
        _Request(), "v1/chat/completions", model_config, {"provider_cooldown_seconds": 60}
    )

    assert response.status_code == 200
    assert remote_calls == ["open"], "the capped entry must be skipped"
    assert budget_calls == ["open"], "a skipped entry must not consume budget"


@pytest.mark.asyncio
async def test_context_skip_chain_exhausts_as_context_not_slots(monkeypatch, caplog):
    """Every entry above its limit -> a context exhaustion, not a slot 429.

    AC3/AC5/AC6: the skip is recorded as ``provider_context_exceeded`` carrying
    the estimate and the limit, logged with provider/estimate/limit, and it
    leaves no cooldown, failure-count, usage-limit or affinity state behind.
    A context skip must also never masquerade as slot exhaustion.
    """
    remote_calls: list[str] = []
    budget_calls: list[str] = []
    model_config, _ = _patch_remote_cycle(
        monkeypatch,
        providers=[
            _remote_entry("capped-a", max_context_tokens=50_000, rate_limit_rpm=5),
            _remote_entry("capped-b", max_context_tokens=50_000),
        ],
        estimate=60_000,
        remote_calls=remote_calls,
        budget_calls=budget_calls,
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy.provider"):
        with pytest.raises(provider.ChainExhaustedError) as exc_info:
            await provider._proxy_with_remote_fallback_cycle(
                _Request(),
                "v1/chat/completions",
                model_config,
                {"provider_cooldown_seconds": 60},
            )

    exhausted = exc_info.value.response
    assert exhausted.status_code == 503
    payload = json.loads(exhausted.body)
    assert payload["code"] == provider.EXHAUSTION_CODE_ALL
    diagnostics = [a for a in payload["diagnostics"] if a.get("status") == "provider_context_exceeded"]
    assert {a["provider"] for a in diagnostics} == {"capped-a", "capped-b"}
    for attempt in diagnostics:
        assert attempt["estimated_tokens"] == 60_000
        assert attempt["limit"] == 50_000
        assert attempt["reason"] == "provider_context_exceeded"

    # (e) no failure-side state was applied
    assert remote_calls == []
    assert budget_calls == []
    assert provider._provider_unavailable_until == {}
    assert provider._provider_failure_count == {}
    assert provider._usage_reset_at == {}
    assert provider._sibling_failure_count == {}
    assert provider_budget._affinity == {}

    # AC3 logging: provider name, estimate, limit and the reason vocabulary
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "provider_context_exceeded" in logged
    for needle in ("capped-a", "60000", "50000"):
        assert needle in logged


@pytest.mark.asyncio
async def test_estimate_is_computed_once_per_cycle(monkeypatch):
    """AC4: one routing estimate per cycle, reused for every gated entry.

    Two gated entries are skipped and the third serves; the authoritative
    estimate pipeline (tokenizer + routing estimate + multiplier) must run
    once, not once per provider.
    """
    estimate_mock = AsyncMock(return_value=60_000)
    tokenizer_mock = MagicMock(return_value=(None, 1.0))
    remote_calls: list[str] = []
    budget_calls: list[str] = []
    providers = [
        _remote_entry("capped-a", max_context_tokens=50_000),
        _remote_entry("capped-b", max_context_tokens=50_000),
        _remote_entry("open"),
    ]
    model_config, _ = _patch_remote_cycle(
        monkeypatch,
        providers=providers,
        estimate=estimate_mock,
        remote_calls=remote_calls,
        budget_calls=budget_calls,
    )
    monkeypatch.setattr(provider, "_get_tokenizer_for_model", tokenizer_mock)

    response = await provider._proxy_with_remote_fallback_cycle(
        _Request(), "v1/chat/completions", model_config, {"provider_cooldown_seconds": 60}
    )

    assert response.status_code == 200
    assert remote_calls == ["open"]
    assert tokenizer_mock.call_count == 1
    assert estimate_mock.await_count == 1


@pytest.mark.asyncio
async def test_skip_leaves_no_state_so_the_entry_serves_a_later_smaller_request(monkeypatch):
    """AC5: a skip records nothing, so the entry stays eligible within the cycle.

    First request (60k) skips the 50k entry and serves the open tail; a second,
    smaller request (40k) is dispatched to the previously skipped entry — proof
    that no cooldown / failure-domain / affinity state was recorded.
    """
    estimates = iter([60_000, 40_000])

    async def _next_estimate(*_args, **_kwargs):
        return next(estimates)

    remote_calls: list[str] = []
    budget_calls: list[str] = []
    model_config, _ = _patch_remote_cycle(
        monkeypatch,
        providers=[
            _remote_entry("capped", max_context_tokens=50_000, rate_limit_rpm=5),
            _remote_entry("open"),
        ],
        estimate=AsyncMock(side_effect=_next_estimate),
        remote_calls=remote_calls,
        budget_calls=budget_calls,
    )
    config = {"provider_cooldown_seconds": 60}

    first = await provider._proxy_with_remote_fallback_cycle(
        _Request(), "v1/chat/completions", model_config, config
    )
    assert first.status_code == 200
    assert remote_calls == ["open"]
    assert provider._provider_unavailable_until == {}
    assert provider_budget._affinity == {}

    second = await provider._proxy_with_remote_fallback_cycle(
        _Request(), "v1/chat/completions", model_config, config
    )
    assert second.status_code == 200
    assert remote_calls == ["open", "capped"]


# ---------------------------------------------------------------------------
# (d) Local+remote cycle: the second dispatch loop enforces the same gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_and_remote_cycle_skips_an_over_limit_remote_entry(monkeypatch):
    """AC2: ``_proxy_with_fallback_cycle`` skips an over-limit remote entry."""
    remote_calls: list[str] = []
    budget_calls: list[str] = []
    model_config, _ = _patch_remote_cycle(
        monkeypatch,
        providers=[
            _remote_entry("capped", max_context_tokens=50_000),
            _remote_entry("open"),
        ],
        estimate=60_000,
        remote_calls=remote_calls,
        budget_calls=budget_calls,
    )

    response = await provider._proxy_with_fallback_cycle(
        _Request(), "v1/chat/completions", model_config, {"provider_cooldown_seconds": 60}
    )

    assert response.status_code == 200
    assert remote_calls == ["open"]
    assert budget_calls == ["open"]


# ---------------------------------------------------------------------------
# (f) Local chains: unaffected when unset, tightened (never loosened) when set
# ---------------------------------------------------------------------------


def test_local_skip_decision_is_unchanged_without_the_field():
    """(f) The existing local gate is untouched when the field is absent."""
    body = {"model": "ctx-limit-test", "messages": [{"role": "user", "content": "hi"}]}
    assert provider._should_skip_local(
        "Qwen3",
        None,
        body,
        1_000_000,
        estimated_tokens=60_000,
        warm_cache_threshold=100_000,
    ) == (False, None)
    assert provider._should_skip_local(
        "Qwen3",
        None,
        body,
        1_000_000,
        estimated_tokens=120_000,
        warm_cache_threshold=100_000,
    ) == (True, "context_too_large")


@pytest.mark.asyncio
async def test_local_entry_limit_composes_as_an_additional_hard_cap(
    monkeypatch, local_and_remote_calls
):
    """AC7: a local ``max_context_tokens`` tightens the existing warm cap.

    With warm=100000 and the entry limit at 50000, a 60k request is skipped
    locally (``context_too_large``) and falls through to the remote tail — the
    new field can only make the existing gate stricter.
    """
    monkeypatch.setattr(
        provider, "_estimate_effective_prompt_tokens_for_routing", AsyncMock(return_value=60_000)
    )
    model_config = {
        "name": "ctx-limit-test",
        "providers": [
            _local_entry(max_context_tokens=50_000),
            _remote_entry("remote-tail"),
        ],
    }

    response = await provider._proxy_with_fallback_cycle(
        _Request(), "v1/chat/completions", model_config, {"provider_cooldown_seconds": 60}
    )

    assert response.status_code == 200
    assert local_and_remote_calls["local_calls"] == []
    assert local_and_remote_calls["remote_calls"] == ["remote-tail"]


@pytest.mark.asyncio
async def test_local_entry_limit_never_loosens_the_existing_cap(
    monkeypatch, local_and_remote_calls
):
    """AC7: a limit ABOVE the existing warm cap leaves local routing unchanged.

    The composed cap is ``min(existing, limit)``: with limit=200000 the warm cap
    stays 100000, so a 60k request still routes local (today's behaviour).
    """
    monkeypatch.setattr(
        provider, "_estimate_effective_prompt_tokens_for_routing", AsyncMock(return_value=60_000)
    )
    model_config = {
        "name": "ctx-limit-test",
        "providers": [
            _local_entry(max_context_tokens=200_000),
            _remote_entry("remote-tail"),
        ],
    }

    response = await provider._proxy_with_fallback_cycle(
        _Request(), "v1/chat/completions", model_config, {"provider_cooldown_seconds": 60}
    )

    assert response.status_code == 200
    assert local_and_remote_calls["local_calls"] == ["http://127.0.0.1:8080"]
    assert local_and_remote_calls["remote_calls"] == []


@pytest.mark.asyncio
async def test_local_context_too_large_still_falls_through_without_a_limit(
    monkeypatch, local_and_remote_calls
):
    """(f) The pre-existing ``context_too_large`` bypass is unchanged."""
    monkeypatch.setattr(
        provider, "_estimate_effective_prompt_tokens_for_routing", AsyncMock(return_value=120_000)
    )
    model_config = {
        "name": "ctx-limit-test",
        "providers": [
            _local_entry(),
            _remote_entry("remote-tail"),
        ],
    }

    response = await provider._proxy_with_fallback_cycle(
        _Request(), "v1/chat/completions", model_config, {"provider_cooldown_seconds": 60}
    )

    assert response.status_code == 200
    assert local_and_remote_calls["local_calls"] == []
    assert local_and_remote_calls["remote_calls"] == ["remote-tail"]


@pytest.mark.asyncio
async def test_queue_context_bypass_honours_a_local_entry_limit(monkeypatch):
    """AC6/AC7: a queued local request never waits for a slot it cannot use.

    ``_queue_context_bypass`` mirrors the smart-routing decision at the
    contention-queue point. A local entry whose ``max_context_tokens`` is below
    the request estimate must therefore be reported as a context bypass (never
    queued), while the same entry without the field still queues as today.
    """
    monkeypatch.setattr(
        provider, "_get_tokenizer_for_model", MagicMock(return_value=(None, 1.0))
    )
    monkeypatch.setattr(
        provider, "_estimate_effective_prompt_tokens_for_routing", AsyncMock(return_value=60_000)
    )
    monkeypatch.setattr(provider, "_get_cached_ratio", lambda *_: 0.0)
    monkeypatch.setattr(
        provider, "_effective_large_context_thresholds", lambda *_: (1_000_000, 100_000)
    )
    config = {"local_large_context_warm_cache_threshold": 100_000}
    body = {"model": "ctx-limit-test", "messages": [{"role": "user", "content": "hi"}]}

    capped = await provider._queue_context_bypass(
        config, {"name": "ctx-limit-test"}, _local_entry(max_context_tokens=50_000),
        _Request(), body, "ctx-limit-session",
    )
    assert capped == (True, "context_too_large")

    ungated = await provider._queue_context_bypass(
        config, {"name": "ctx-limit-test"}, _local_entry(),
        _Request(), body, "ctx-limit-session",
    )
    assert ungated == (False, None)
