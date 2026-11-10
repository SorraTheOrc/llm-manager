"""Tests for the overloaded-429 retry and balance-exhaustion quarantine
(LP-0MW46PBWL008T317, epic LP-0MW46P7QU0064LM3).

Goal: once a ``use_balance: true`` account may dispatch on credit, the proxy
must (a) stop cleanly when the credit balance is genuinely exhausted, and
(b) retry briefly, not quarantine, when the upstream is transiently
overloaded — so the chain neither spends forever nor collapses on a
momentary blip.

Covered acceptance criteria:

- AC1: a simulated ``[overloaded]`` 429 with ``Retry-After: 2`` produces a
  short cooldown (~2 s, bounded) and a retry, and does NOT add
  ``https://opencode.ai/zen/go`` to the permanent attempted-domains set.
- AC2: after the short cooldown the same entry is re-tried and can succeed
  (the test simulates 429 then 200 across a chain-hold cycle).
- AC3: a genuine balance-exhaustion response (402 / usage-limit body)
  quarantines the account with real reset timing and is not retried in a
  loop.
- AC4: balance-exhaustion quarantine is cumulative with, and does not
  shorten, an existing usage-limit quarantine.
- AC5: a chain with ``opencode-go`` failing transient overloads falls
  through to the next provider after bounded retries rather than exhausting
  or hanging indefinitely.
- AC6: logs distinguish ``overloaded`` retry from ``quota_exhausted`` and
  from a hard usage-limit quarantine; tests assert the reasons.
- AC7: unit and integration tests cover the transient 429 retry and the
  balance exhaustion quarantine.
- AC10: the transient ``[overloaded]`` 429 is a short retryable cooldown
  honouring ``Retry-After``, never a permanent domain failure.
"""

from __future__ import annotations

import json
import logging
import time as _real_time
from unittest.mock import MagicMock

import proxy.provider as provider
import pytest
from fastapi import Response

_PROVIDER_LOGGER = "llama-proxy.provider"
_OK_BODY = json.dumps({"choices": [{"message": {"content": "ok"}}]})

#: The failure domain of the opencode-go entries — AC1/AC10 require it is
#: never poisoned by an overloaded 429.
_GO_DOMAIN = "https://opencode.ai/zen/go:deepseek-v4-flash"


class _FakeTime:
    """Stand-in for ``provider.time`` with an advanceable wall clock.

    The cooldown / quarantine machinery reads ``time.time()``; advancing the
    wall clock by the recorded chain-hold wait simulates the real elapsed
    time between cycles, so a 2-second overloaded cooldown is expired when
    the next cycle reads it.
    """

    def __init__(self, start: float = 1_000_000.0):
        self.wall = start

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return _real_time.monotonic()


def _config(**server: object) -> dict:
    return {"server": dict(server), "provider_cooldown_seconds": 60}


def _opencode_entry(
    name: str = "opencode-go",
    api_key_env: str = "OPENCODE_KEY_A",
    *,
    use_balance: bool = True,
    **extra,
) -> dict:
    entry = {
        "name": name,
        "type": "remote",
        "provider": "opencode-go",
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": api_key_env,
        "model": "deepseek-v4-flash",
        "use_balance": use_balance,
    }
    entry.update(extra)
    return entry


def _fallback_entry(name: str = "munitoai") -> dict:
    return {
        "name": name,
        "type": "remote",
        "provider": "munitoai",
        "endpoint": "https://api.munito.ai/v1",
        "api_key_env": "MUNITOAI_KEY",
        "model": "deepseek-v4-flash",
    }


def _overloaded_body() -> bytes:
    return json.dumps({
        "error": {
            "type": "[overloaded]",
            "message": "the server is overloaded; retry shortly",
        }
    }).encode()


def _overloaded_response(retry_after: str | None = "2", ms: str | None = None) -> Response:
    headers: dict[str, str] = {}
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    if ms is not None:
        headers["retry-after-ms"] = ms
    return Response(
        status_code=429,
        content=_overloaded_body(),
        media_type="application/json",
        headers=headers,
    )


def _balance_402_body() -> bytes:
    return json.dumps({
        "error": {
            "type": "insufficient_quota",
            "message": "Prepaid credit exhausted. Resets in 2 hours 0 minutes.",
        }
    }).encode()


def _balance_402_response() -> Response:
    return Response(
        status_code=402,
        content=_balance_402_body(),
        media_type="application/json",
    )


class _Request:
    """Minimal request stub for the fallback chain."""

    def __init__(self, session_id: str | None = "overload-session"):
        self._body = json.dumps(
            {"model": "bal-test", "messages": [{"role": "user", "content": "hi"}]}
        ).encode()
        self.headers = {"content-type": "application/json"}
        if session_id:
            self.headers["x-session-id"] = session_id
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


async def _noop_async(*args, **kwargs):
    return None


def _patch_remote_cycle(monkeypatch, dispatch):
    """Patch the remote dispatch pointer with a caller-controlled dispatcher.

    ``dispatch(provider_cfg, call_index)`` returns the Response to serve for
    that provider; call_index is the 0-based count of prior dispatches to any
    provider (so a test can serve 429-then-200 across the chain).
    """
    calls: list[tuple[str, int]] = []
    counter = 0

    async def _ptr(_request, _path, provider_cfg):
        nonlocal counter
        idx = counter
        counter += 1
        calls.append((provider_cfg["name"], idx))
        return dispatch(provider_cfg, idx)

    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: _ptr)
    monkeypatch.setattr(
        provider, "_get_tokenizer_for_model", MagicMock(return_value=(None, 1.0))
    )
    monkeypatch.setattr(provider, "_maybe_probe_quarantined_accounts", _noop_async)
    return calls


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path):
    """Fresh cooldown/quarantine state and a deterministic fake wall clock."""
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._usage_reset_guessed.clear()
    provider._usage_reset_window.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()
    monkeypatch.setenv(
        provider._PROVIDER_STATE_FILE_ENV, str(tmp_path / "provider-state.json")
    )
    fake_time = _FakeTime()
    monkeypatch.setattr(provider, "time", fake_time)
    monkeypatch.setattr(provider, "request_priority", lambda *a, **k: "medium")
    monkeypatch.setattr("proxy.priority.request_priority", lambda *a, **k: "medium")
    yield
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._usage_reset_guessed.clear()
    provider._usage_reset_window.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()


def _drive_cycle(model_config: dict, config: dict):
    return provider._proxy_with_remote_fallback_cycle(
        _Request(), "v1/chat/completions", model_config, config
    )


# ---------------------------------------------------------------------------
# Classifier units (AC7)
# ---------------------------------------------------------------------------


def test_overloaded_classifier_matches_observed_shapes():
    """The OpenCode ``[overloaded]`` 429 shapes are classified as overloaded."""
    body = _overloaded_body().decode()
    # Observed: 429 ... [overloaded] the server is overloaded; retry shortly
    assert provider._is_overloaded_error(
        Response(status_code=429, content=body.encode()), body
    ) is True
    # error.type == "overloaded"
    body = json.dumps({"error": {"type": "overloaded"}})
    assert provider._is_overloaded_error(
        Response(status_code=429, content=body.encode()), body
    ) is True
    # Plain-wording fallback (no bracket marker).
    body = json.dumps({"error": {"message": "server is overloaded retry shortly"}})
    assert provider._is_overloaded_error(
        Response(status_code=429, content=body.encode()), body
    ) is True
    # Not an overload: a quota 429 must NOT be classified as overloaded.
    quota = json.dumps({
        "error": {"type": "GoUsageLimitError", "message": "Weekly usage limit reached."}
    })
    assert provider._is_overloaded_error(
        Response(status_code=429, content=quota.encode()), quota
    ) is False
    # Not an overload: a non-429 status is never classified.
    assert provider._is_overloaded_error(
        Response(status_code=402, content=_overloaded_body()), _overloaded_body().decode()
    ) is False


def test_balance_exhaustion_classifier_shapes():
    """402 and usage-limit bodies on non-429 statuses are balance exhaustion."""
    assert provider._is_balance_exhaustion_error(
        _balance_402_response(), "x"
    ) is True
    # Usage-limit body on a 403 (non-429) is balance exhaustion.
    body = json.dumps({
        "error": {"type": "insufficient_quota", "code": "usage_limit_reached"}
    }).encode()
    assert provider._is_balance_exhaustion_error(
        Response(status_code=403, content=body), body.decode()
    ) is True
    # A 429 is never balance exhaustion (the quota/usage-limit path owns it).
    assert provider._is_balance_exhaustion_error(
        Response(status_code=429, content=_overloaded_body()), "x"
    ) is False
    # A 200 is not an error at all.
    assert provider._is_balance_exhaustion_error(
        Response(status_code=200, content=b"ok"), "ok"
    ) is False


def test_overloaded_cooldown_is_bounded_and_honours_retry_after():
    """AC1/AC10: the overloaded cooldown honours Retry-After, bounded."""
    # Retry-After: 2 -> ~2s cooldown.
    assert provider._overloaded_cooldown_seconds(_overloaded_response("2"), {}) == pytest.approx(2.0)
    # retry-after-ms: 2000 -> the same ~2s.
    resp_ms = _overloaded_response(retry_after=None, ms="2000")
    assert provider._overloaded_cooldown_seconds(resp_ms, {}) == pytest.approx(2.0)
    # A wildly large Retry-After is capped (never a long cooldown).
    assert provider._overloaded_cooldown_seconds(
        _overloaded_response("3600"), {}
    ) == pytest.approx(provider._OVERLOADED_COOLDOWN_DEFAULT_MAX_SECONDS)
    # The cap is configurable.
    assert provider._overloaded_cooldown_seconds(
        _overloaded_response("3600"), {"server": {"overloaded_cooldown_max_seconds": 8}}
    ) == pytest.approx(8)
    # No header at all -> the 1-second base, still bounded.
    assert provider._overloaded_cooldown_seconds(
        _overloaded_response(retry_after=None), {}
    ) == pytest.approx(provider._OVERLOADED_BASE_COOLDOWN_SECONDS)


def test_balance_exhaustion_quarantine_reset_resolution():
    """AC3: a 402 with a reset message resolves the reset-aware quarantine."""
    entry = _opencode_entry()
    info = provider._balance_exhaustion_reset_info(
        _balance_402_response(), _balance_402_body().decode(), {}, entry
    )
    assert info is not None
    assert info.explicit is True
    # "Resets in 2 hours 0 minutes" + the 2-minute margin.
    assert info.seconds == pytest.approx(2 * 3600 + provider._USAGE_LIMIT_RESET_MARGIN_SECONDS)


# ---------------------------------------------------------------------------
# AC1 / AC10 — overloaded 429: short cooldown, no domain poisoning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_overloaded_429_short_cooldown_no_domain_poisoning(monkeypatch, caplog):
    """AC1/AC10: a transient overload poisons neither the domain nor the account."""
    entry = _opencode_entry()
    model_config = {"name": "bal-test", "providers": [entry]}
    config = _config()

    calls = _patch_remote_cycle(
        monkeypatch, lambda _cfg, _idx: _overloaded_response("2")
    )
    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        with pytest.raises(provider.ChainExhaustedError):
            await _drive_cycle(model_config, config)

    assert [name for name, _ in calls] == ["opencode-go"]

    # Short cooldown (~2s), not the 60s default and not 30s max.
    remaining = provider._provider_cooldown_remaining(entry["name"])
    assert 1 <= remaining <= 4, f"expected a ~2s cooldown, got {remaining}s"

    # The failure DOMAIN is NOT cooldowned and the ACCOUNT is NOT quarantined:
    # transient overload never becomes a permanent domain failure (AC1/AC10).
    assert provider._is_provider_unavailable(_GO_DOMAIN) is False
    assert provider._is_provider_unavailable("opencode-go") is True
    account = provider._usage_limit_account_key(entry)
    assert provider._usage_reset_remaining(account) == 0.0

    # The attempt carries the distinct overloaded status and log reason (AC6).
    assert any(
        record.levelno == logging.INFO
        and "reason=overloaded" in record.getMessage()
        and "cooldown=" in record.getMessage()
        for record in caplog.records
    ), [r.getMessage() for r in caplog.records]


# ---------------------------------------------------------------------------
# AC2 — after the short cooldown the same entry is re-tried successfully
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_overloaded_retry_after_cooldown_succeeds(monkeypatch):
    """AC2: 429 [overloaded] then 200 — the chain-hold waits the cooldown then
    re-tries the SAME entry."""
    entry = _opencode_entry()
    model_config = {"name": "bal-test", "providers": [entry]}
    config = {
        **_config(),
        "chain_hold_seconds": 300,
        "chain_hold_max_cycles": 3,
    }
    fake_time = provider.time
    holds: list[float] = []

    async def fake_hold(_req, seconds):
        holds.append(seconds)
        # Simulate the elapsed wall time so the cooldown expires for the
        # next cycle (exactly what the real sleep would do).
        fake_time.wall += seconds
        return False

    monkeypatch.setattr(provider, "_hold_sleep", fake_hold)

    def _dispatch(_cfg, idx):
        if idx == 0:
            return _overloaded_response("2")
        return Response(content=_OK_BODY, status_code=200, media_type="application/json")

    calls = _patch_remote_cycle(monkeypatch, _dispatch)
    result = await provider.proxy_with_remote_fallback(
        _Request(), "v1/chat/completions", model_config, config
    )

    assert result.status_code == 200
    assert len(calls) == 2, calls
    assert [name for name, _ in calls] == ["opencode-go", "opencode-go"]
    # The hold is bounded by the upstream's own wait (2s), not the 300s hold.
    assert holds == [2.0], holds
    # The entry came out of cooldown (the clock advanced) and served.
    assert provider._is_provider_unavailable(entry["name"]) is False


# ---------------------------------------------------------------------------
# AC3 — balance exhaustion quarantines the account with real reset timing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_balance_exhaustion_402_quarantines_account(monkeypatch, caplog):
    """AC3: a 402 with a usage-limit body quarantines the account for the
    advertised reset (reusing the reset-aware quarantine) and the next cycle
    skips the entry instead of retrying it in a loop."""
    entry = _opencode_entry()
    model_config = {"name": "bal-test", "providers": [entry]}
    config = _config()

    calls = _patch_remote_cycle(
        monkeypatch, lambda _cfg, _idx: _balance_402_response()
    )
    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        with pytest.raises(provider.ChainExhaustedError):
            await _drive_cycle(model_config, config)

    assert [name for name, _ in calls] == ["opencode-go"]
    account = provider._usage_limit_account_key(entry)
    remaining = provider._usage_reset_remaining(account)
    # Real reset timing: ~2h from the body + the 2-minute margin.
    assert remaining == pytest.approx(
        2 * 3600 + provider._USAGE_LIMIT_RESET_MARGIN_SECONDS, abs=2
    )
    assert account not in provider._usage_reset_guessed  # explicit, hard
    # A second cycle must NOT re-dispatch the quarantined account.
    calls2 = _patch_remote_cycle(
        monkeypatch, lambda _cfg, _idx: _balance_402_response()
    )
    with pytest.raises(provider.ChainExhaustedError):
        await _drive_cycle(model_config, config)
    assert calls2 == [], "a quarantined account must not be retried in a loop"
    # The 402 went through the reset-aware quarantine, not the overload retry.
    assert any(
        "usage-limit reset source=message" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]
    assert all(
        "reason=overloaded" not in r.getMessage() for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


@pytest.mark.asyncio
async def test_bare_402_without_body_evidence_quarantines_via_balance_branch(
    monkeypatch, caplog
):
    """A bare 402 (no usage-limit body, no reset message) is the balance-
    exhaustion branch: the account is still quarantined (bounded guess) and
    the log names ``reason=balance_exhaustion`` (AC3/AC6)."""
    entry = _opencode_entry()
    model_config = {"name": "bal-test", "providers": [entry]}
    config = _config()

    def _dispatch(_cfg, _idx):
        return Response(status_code=402, content=b"Payment Required")

    calls = _patch_remote_cycle(monkeypatch, _dispatch)
    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        with pytest.raises(provider.ChainExhaustedError):
            await _drive_cycle(model_config, config)
    assert calls
    account = provider._usage_limit_account_key(entry)
    remaining = provider._usage_reset_remaining(account)
    assert remaining > 0, "a bare 402 must still quarantine the account"
    assert remaining <= (
        provider._usage_limit_guess_cap_seconds(config)
        + provider._USAGE_LIMIT_RESET_MARGIN_SECONDS
    )
    assert any(
        "reason=balance_exhaustion" in r.getMessage()
        and "reset_in=" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


# ---------------------------------------------------------------------------
# AC4 — cumulative: never shorten an existing usage-limit quarantine
# ---------------------------------------------------------------------------


def test_quarantine_is_cumulative_and_never_shortened():
    """AC4: ``_set_usage_limit_quarantine`` only extends — a shorter reset
    never shortens or clears a longer quarantine already recorded."""
    account = provider._usage_limit_account_key(_opencode_entry())
    margin = provider._USAGE_LIMIT_RESET_MARGIN_SECONDS
    long_reset = provider._UsageLimitReset(3600 + margin, True, "weekly", "weekly")
    short_reset = provider._UsageLimitReset(120 + margin, True, "daily", "daily")

    # Existing LONG quarantine + a SHORT signal: the long one survives.
    provider._set_usage_limit_quarantine(account, long_reset)
    provider._set_usage_limit_quarantine(account, short_reset)
    assert provider._usage_reset_remaining(account) == pytest.approx(
        3600 + margin, abs=2
    )
    # The recorded metadata stays the longer quarantine's.
    assert provider._usage_reset_window[account] == "weekly"

    # Reverse order: a LONG signal extends a short quarantine.
    provider.clear_usage_limit_quarantine()
    provider._set_usage_limit_quarantine(account, short_reset)
    provider._set_usage_limit_quarantine(account, long_reset)
    assert provider._usage_reset_remaining(account) == pytest.approx(
        3600 + margin, abs=2
    )
    assert provider._usage_reset_window[account] == "weekly"

    # An equal-length reset keeps the same quarantine (idempotent).
    provider._set_usage_limit_quarantine(account, long_reset)
    assert provider._usage_reset_remaining(account) == pytest.approx(
        3600 + margin, abs=2
    )


# ---------------------------------------------------------------------------
# AC5 — falls through to the next provider after bounded retries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chain_falls_through_to_next_provider_on_transient_overload(monkeypatch):
    """AC5: opencode-go overloading transiently does not exhaust or hang the
    chain — the next provider serves the request in the same cycle."""
    entry = _opencode_entry()
    fallback = _fallback_entry()
    model_config = {"name": "bal-test", "providers": [entry, fallback]}
    config = _config()

    def _dispatch(cfg, _idx):
        if cfg["name"] == "opencode-go":
            return _overloaded_response("2")
        return Response(content=_OK_BODY, status_code=200, media_type="application/json")

    calls = _patch_remote_cycle(monkeypatch, _dispatch)
    result = await provider.proxy_with_remote_fallback(
        _Request(), "v1/chat/completions", model_config, config
    )

    assert result.status_code == 200
    assert [name for name, _ in calls] == ["opencode-go", "munitoai"]
    # The overloaded entry is only ever on a SHORT cooldown — it did not get
    # a long quarantine, and the domain was never poisoned (AC10).
    remaining = provider._provider_cooldown_remaining(entry["name"])
    assert 1 <= remaining <= 4, remaining
    assert provider._is_provider_unavailable(_GO_DOMAIN) is False
    account = provider._usage_limit_account_key(entry)
    assert provider._usage_reset_remaining(account) == 0.0


@pytest.mark.asyncio
async def test_repeated_overloads_stay_short_and_bounded(monkeypatch):
    """AC10: repeated overloaded 429s never grow into a long cooldown or a
    domain failure — the cooldown stays the bounded Retry-After."""
    entry = _opencode_entry()
    model_config = {"name": "bal-test", "providers": [entry]}
    config = _config()

    calls = _patch_remote_cycle(
        monkeypatch, lambda _cfg, _idx: _overloaded_response("2")
    )
    for _ in range(3):
        with pytest.raises(provider.ChainExhaustedError):
            await _drive_cycle(model_config, config)
        # Immediately after each dispatch the cooldown is still the SHORT
        # bounded wait (never grown by repeated failures).
        remaining = provider._provider_cooldown_remaining(entry["name"])
        assert 1 <= remaining <= 4, (
            f"repeated overloads must stay short, got {remaining}s"
        )
        # Then let the short cooldown elapse so the entry is re-dispatched
        # (rather than skipped while still cooling) next cycle.
        provider.time.wall += 3

    assert len(calls) == 3
    assert provider._is_provider_unavailable(_GO_DOMAIN) is False
    assert provider._usage_reset_remaining(
        provider._usage_limit_account_key(entry)
    ) == 0.0


@pytest.mark.asyncio
async def test_mixed_cycle_overloaded_branch(monkeypatch, caplog):
    """The mixed local+remote cycle applies the same overloaded handling: a
    local 400 falls through, opencode-go's transient overload is a short
    cooldown (no domain poisoning), and the remote fallback serves."""
    local_entry = {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"}
    go = _opencode_entry()
    fallback = _fallback_entry()
    model_config = {"name": "mixed-test", "providers": [local_entry, go, fallback]}
    config = _config()
    call_log: list[str] = []

    async def _mock_local(_req, _path):
        call_log.append("local")
        return Response(status_code=400, content=b"bad request")

    async def _mock_remote(_req, _path, provider_cfg):
        call_log.append(provider_cfg["name"])
        if provider_cfg["name"] == "opencode-go":
            return _overloaded_response("2")
        return Response(content=_OK_BODY, status_code=200, media_type="application/json")

    monkeypatch.setattr(
        provider, "_get_proxy_to_local", lambda: _mock_local, raising=True
    )
    monkeypatch.setattr(
        provider, "_get_proxy_to_remote", lambda: _mock_remote, raising=True
    )
    monkeypatch.setattr(
        provider, "_get_tokenizer_for_model", MagicMock(return_value=(None, 1.0))
    )
    monkeypatch.setattr(provider, "_maybe_probe_quarantined_accounts", _noop_async)
    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        result = await provider.proxy_with_fallback(
            _Request(), "v1/chat/completions", model_config, config
        )

    assert result.status_code == 200
    assert call_log == ["local", "opencode-go", "munitoai"], call_log
    # The overloaded branch in the mixed cycle logged its distinct reason and
    # did NOT poison the failure domain or quarantine the account.
    assert any(
        "reason=overloaded" in r.getMessage()
        and "provider=opencode-go" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]
    assert provider._is_provider_unavailable(_GO_DOMAIN) is False
    assert provider._usage_reset_remaining(
        provider._usage_limit_account_key(go)
    ) == 0.0


# ---------------------------------------------------------------------------
# AC6 — reason vocabulary stays distinct
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reason_vocabulary_distinct(monkeypatch, caplog):
    """The three vocabularies never mix: ``overloaded`` vs ``quota_exhausted``
    vs ``usage_limit_reset`` are separate attempt statuses / log reasons."""
    entry = _opencode_entry()
    model_config = {"name": "bal-test", "providers": [entry]}
    config = _config()

    # 1. Transient overload.
    calls = _patch_remote_cycle(
        monkeypatch, lambda _cfg, _idx: _overloaded_response("2")
    )
    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        with pytest.raises(provider.ChainExhaustedError):
            await _drive_cycle(model_config, config)
    assert [name for name, _ in calls] == ["opencode-go"]
    assert any(
        "reason=overloaded" in r.getMessage() for r in caplog.records
    ), caplog.text
    assert all(
        "reason=quota_exhausted" not in r.getMessage() for r in caplog.records
    ), caplog.text

    # 2. Hard usage-limit reset (a GoUsageLimitError 429) keeps its own
    #    quarantine reason, distinct from both overloaded and balance.
    provider._provider_unavailable_until.clear()
    provider._usage_reset_at.clear()
    provider._usage_reset_guessed.clear()
    provider._usage_reset_window.clear()
    caplog.clear()
    go_body = json.dumps({
        "error": {
            "type": "GoUsageLimitError",
            "message": "Weekly usage limit reached. Resets in 22hr 43min.",
        },
        "metadata": {"limitName": "weekly"},
    }).encode()

    def _go_dispatch(_cfg, _idx):
        return Response(status_code=429, content=go_body, media_type="application/json")

    calls2 = _patch_remote_cycle(monkeypatch, _go_dispatch)
    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        with pytest.raises(provider.ChainExhaustedError):
            await _drive_cycle(model_config, config)
    assert calls2
    account = provider._usage_limit_account_key(entry)
    assert provider._usage_reset_remaining(account) > 0
    # The weekly message reset is explicit/hard — a hard quarantine, distinct
    # from a guessed one.
    assert account not in provider._usage_reset_guessed
    assert any(
        "reason=usage_limit_reset" in r.getMessage()
        or "usage-limit reset source=" in r.getMessage()
        for r in caplog.records
    ), caplog.text
    assert all(
        "reason=overloaded" not in r.getMessage() for r in caplog.records
    ), caplog.text
