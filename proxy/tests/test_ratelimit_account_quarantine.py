"""Tests for the reset-aware, account-scoped 429 quarantine (LP-0MVFA0PAV0022ZHO).

MunitoAI answers a per-account quota 429 with custom ``x-ratelimit-*`` headers
(no standard ``Retry-After``). The generic header parsing engine
(LP-0MVBHDNLB007ALLJ) already sets an ENTRY-level cooldown from the reset
header, and the account-key eligibility helpers (LP-0MVCX0E2V001VB5N) already
scope the per-request exclusion. This module pins down the remaining gap:

- a quota 429 carrying a parseable ``x-ratelimit-reset-*`` header must ALSO
  quarantine the failing API-key ACCOUNT in ``_usage_reset_at`` for the
  reported reset plus the existing ``_USAGE_LIMIT_RESET_MARGIN_SECONDS``
  (2-minute) margin, so the wait survives the request and is reflected in
  ``_compute_retry_after()``;
- the quarantine stays ACCOUNT-scoped: a paid sibling sharing the same
  endpoint/model (different ``api_key_env``) is neither quarantined nor
  cooled down;
- a 429 without any reset signal keeps the pre-change behaviour exactly
  (short entry cooldown, account-key poisoning only, no quarantine);
- the ``respect_ratelimit_headers`` flag stays the gate: when it is off the
  reset headers are ignored and no account quarantine is created;
- both fallback loops (``_proxy_with_remote_fallback_cycle`` and
  ``_proxy_with_fallback_cycle``) apply the quarantine, a quota 429 still
  advances the chain, and chain-hold cycles still apply when the whole chain
  is exhausted;
- the ``FreeUsageLimitError`` (3 h cooldown) and ``GoUsageLimitError``
  (account quarantine) paths are regression-guarded.

Coverage map: AC1/AC4 (helper + cycle), AC2 (sibling eligibility in
``test_failure_domain_grouping.py``), AC3 (chain-hold exhaustion), AC5 (no
reset signal), AC6 (Free/Go regressions), AC7 (flag gate), AC8 (both cycles).

TDD red phase: the reset-aware quarantine assertions are marked
``xfail(strict=True)`` so ``dev`` stays green while this red phase is in
flight — the same convention as ``test_compaction_headers_contract.py``
(LP-0MU44KR5F0015D7A) and ``test_restart_lifecycle.py`` (LP-0MVF7DO29009T8MH).
``strict=True`` means an unexpected pass fails the suite, so the
implementation item LP-0MVFA1VRT007T9HT must remove these markers when it
makes the quarantine real. The pre-xfail red run is recorded on the work item.
"""

import json
import time
from unittest.mock import patch

import proxy.provider as provider
import pytest
from fastapi import Response

_MUNITO_ENDPOINT = "https://api.munito.ai/inference"
_MUNITO_MODEL = "deepseek-ai/deepseek-v4-flash"

# Red-phase marker (see module docstring): removed by LP-0MVFA1VRT007T9HT.
_RED_REASON = "reset-derived account quarantine added by LP-0MVFA1VRT007T9HT"

# Reset headers mirroring a live MunitoAI free-tier quota 429
# (LP-0MVBHDNLB007ALLJ): requests exhausted, reset in 32 s.
_RESET_HEADERS = {
    "x-ratelimit-limit-requests": "10",
    "x-ratelimit-remaining-requests": "0",
    "x-ratelimit-reset-requests": "32s",
    "x-ratelimit-over-limit": "yes",
}

# Reset is 32 s, the header-derived ENTRY cooldown adds the 2 s
# ``_RATELIMIT_COOLDOWN_MARGIN_SECONDS``, while the ACCOUNT quarantine adds the
# 2-minute ``_USAGE_LIMIT_RESET_MARGIN_SECONDS``.
_RESET_SECONDS = 32.0
_ENTRY_COOLDOWN_SECONDS = _RESET_SECONDS + 2.0
_ACCOUNT_QUARANTINE_SECONDS = (
    _RESET_SECONDS + provider._USAGE_LIMIT_RESET_MARGIN_SECONDS
)

# The observed api.munito.ai per-account quota 429 body
# (LP-0MVCX0E2V001VB5N).
_QUOTA_BODY = json.dumps(
    {"code": "rate_limit_exceeded", "type": "rate_limit_error"}
)


class _DummyRequest:
    """Minimal request stub (mirrors test_failure_domain_grouping)."""

    def __init__(self, body: bytes = b'{"model":"test"}'):
        self._body = body
        self.headers = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body

    async def is_disconnected(self):
        return False


def _free_entry(**extra):
    """The free-tier MunitoAI entry that opts into ``x-ratelimit-*`` headers."""
    return {
        "name": "munitoai-free",
        "type": "remote",
        "provider": "MunitoAI",
        "endpoint": _MUNITO_ENDPOINT,
        "api_key_env": "MUNITOAI_FREE_API_KEY",
        "model": _MUNITO_MODEL,
        "respect_ratelimit_headers": True,
        **extra,
    }


def _paid_entry():
    """The paid MunitoAI entry: same endpoint+model, different API-key account."""
    return {
        "name": "munitoai",
        "type": "remote",
        "provider": "MunitoAI",
        "endpoint": _MUNITO_ENDPOINT,
        "api_key_env": "MUNITOAI_API_KEY",
        "model": _MUNITO_MODEL,
    }


@pytest.fixture
def free_paid_chain():
    """Free MunitoAI followed by its paid sibling on the SAME failure domain."""
    return {"providers": [_free_entry(), _paid_entry()], "aliases": ["test*"]}


@pytest.fixture(autouse=True)
def _isolate_provider_state(tmp_path, monkeypatch):
    """Isolate cooldown/quarantine maps and the state file for every test."""
    monkeypatch.setenv(
        provider._PROVIDER_STATE_FILE_ENV,
        str(tmp_path / "provider-state.json"),
    )
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._usage_reset_guessed.clear()
    provider._usage_limit_last_probe.clear()
    yield
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._usage_reset_guessed.clear()
    provider._usage_limit_last_probe.clear()


def _quota_429(headers: dict | None = None) -> Response:
    """A per-account quota 429 (rate_limit_error / rate_limit_exceeded)."""
    return Response(
        content=_QUOTA_BODY.encode("utf-8"),
        status_code=429,
        media_type="application/json",
        headers=headers or {},
    )


def _ok_json_response() -> Response:
    return Response(
        content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
        status_code=200,
        media_type="application/json",
    )


def _account_keys():
    return (
        provider._usage_limit_account_key(_free_entry()),
        provider._usage_limit_account_key(_paid_entry()),
    )


def _assert_failing_account_quarantined(expected_seconds: float, abs_tol: float = 2.0):
    """Assert only the failing account is quarantined, for the reset window."""
    free_account, paid_account = _account_keys()
    assert free_account in provider._usage_reset_at, (
        "a quota 429 with an x-ratelimit-reset-* header must quarantine the "
        "failing API-key account"
    )
    remaining = provider._usage_reset_remaining(free_account)
    assert remaining == pytest.approx(expected_seconds, abs=abs_tol), (
        "the account quarantine must last the reported reset plus the "
        f"{provider._USAGE_LIMIT_RESET_MARGIN_SECONDS}s margin; got {remaining}"
    )
    assert paid_account not in provider._usage_reset_at, (
        "the paid sibling account must not be quarantined"
    )
    assert free_account not in provider._usage_reset_guessed, (
        "a reset derived from an explicit header is not a guessed quarantine"
    )
    return free_account, paid_account


# ===========================================================================
# AC1 / AC4 / AC5 / AC7 — the shared 429 cooldown primitive
# ===========================================================================


class TestResetAwareAccountQuarantine:
    def test_reset_header_sets_accurate_entry_cooldown(self):
        """AC1: 32 s reset -> accurate 34 s entry cooldown (reset + the 2 s
        ``_RATELIMIT_COOLDOWN_MARGIN_SECONDS``) and attempt diagnostics."""
        attempts = []
        response = _quota_429(_RESET_HEADERS)
        with patch("time.time", return_value=1000.0):
            cooldown = provider._handle_http_error_with_cooldown(
                response,
                "munitoai-free",
                "remote",
                60.0,
                attempts,
                _QUOTA_BODY,
                {"respect_ratelimit_headers": True},
                provider_cfg=_free_entry(),
            )
            assert cooldown == pytest.approx(_ENTRY_COOLDOWN_SECONDS)
            assert provider._provider_unavailable_until["munitoai-free"] == (
                pytest.approx(1000.0 + _ENTRY_COOLDOWN_SECONDS)
            )
            assert attempts[-1]["ratelimit_reset_seconds"] == pytest.approx(
                _RESET_SECONDS
            )
            assert attempts[-1]["ratelimit_over_limit"] is True

    @pytest.mark.xfail(strict=True, reason=_RED_REASON)
    def test_reset_header_quarantines_failing_account_beyond_entry_cooldown(self):
        """AC1/AC4: the same 32 s reset ALSO quarantines the failing account
        for reset + the 2-minute ``_USAGE_LIMIT_RESET_MARGIN_SECONDS``, which
        outlives the entry cooldown and stays account-scoped."""
        attempts = []
        response = _quota_429(_RESET_HEADERS)
        with patch("time.time", return_value=1000.0):
            cooldown = provider._handle_http_error_with_cooldown(
                response,
                "munitoai-free",
                "remote",
                60.0,
                attempts,
                _QUOTA_BODY,
                {"respect_ratelimit_headers": True},
                provider_cfg=_free_entry(),
            )
            # The account quarantine outlives the entry cooldown and is
            # recorded against the failing account only.
            _assert_failing_account_quarantined(
                _ACCOUNT_QUARANTINE_SECONDS, abs_tol=0.01
            )
            assert provider._usage_reset_at[
                provider._usage_limit_account_key(_free_entry())
            ] == pytest.approx(1000.0 + _ACCOUNT_QUARANTINE_SECONDS)
            assert provider._usage_reset_remaining(
                provider._usage_limit_account_key(_free_entry())
            ) > cooldown

    def test_quota_429_without_reset_signal_keeps_short_cooldown(self):
        """AC5: no x-ratelimit-* reset -> exponential backoff, no quarantine,
        account-key poisoning unchanged (the pre-existing behaviour)."""
        attempts = []
        response = _quota_429()
        with patch("time.time", return_value=1000.0):
            cooldown = provider._handle_http_error_with_cooldown(
                response,
                "munitoai-free",
                "remote",
                60.0,
                attempts,
                _QUOTA_BODY,
                {"respect_ratelimit_headers": True},
                provider_cfg=_free_entry(),
            )
            assert cooldown == 1.0  # _BACKOFF_BASE_SECONDS
            assert provider._usage_reset_at == {}, (
                "a quota 429 without any reset signal must not create an "
                "account quarantine"
            )
            assert provider._provider_unavailable_until["munitoai-free"] == 1001.0
            assert "ratelimit_reset_seconds" not in attempts[-1]

    def test_disabled_feature_flag_ignores_reset_headers(self):
        """AC7: respect_ratelimit_headers=False -> headers ignored, no
        account quarantine, plain backoff."""
        attempts = []
        response = _quota_429(_RESET_HEADERS)
        entry = _free_entry(respect_ratelimit_headers=False)
        with patch("time.time", return_value=1000.0):
            cooldown = provider._handle_http_error_with_cooldown(
                response,
                "munitoai-free",
                "remote",
                60.0,
                attempts,
                _QUOTA_BODY,
                {"respect_ratelimit_headers": True},  # global on, entry off
                provider_cfg=entry,
            )
            assert cooldown == 1.0
            assert provider._usage_reset_at == {}
            assert "ratelimit_reset_seconds" not in attempts[-1]


# ===========================================================================
# AC1 / AC4 / AC8 — both fallback loops apply the account quarantine
# ===========================================================================


def _chain_mock(call_order: list[str], headers: dict | None = _RESET_HEADERS):
    """Free entry answers the quota 429, the paid sibling answers 200."""

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        name = provider_cfg.get("name")
        call_order.append(name)
        if name == "munitoai-free":
            return _quota_429(headers)
        return _ok_json_response()

    return _mock_proxy_to_remote


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason=_RED_REASON)
async def test_remote_fallback_cycle_quarantines_failing_account(free_paid_chain):
    """AC1/AC8: the remote-only loop quarantines the failing account and still
    advances to the paid sibling within the same request."""
    call_order: list[str] = []
    with patch(
        "proxy.server.proxy_to_remote", _chain_mock(call_order)
    ):
        result = await provider._proxy_with_remote_fallback_cycle(
            _DummyRequest(),
            "v1/chat/completions",
            free_paid_chain,
            {"provider_cooldown_seconds": 60},
        )

    assert result.status_code == 200
    assert call_order == ["munitoai-free", "munitoai"], (
        f"a quota 429 must advance the chain to the sibling, got {call_order}"
    )
    _free_account, _paid_account = _assert_failing_account_quarantined(
        _ACCOUNT_QUARANTINE_SECONDS
    )
    assert not provider._is_provider_unavailable("munitoai"), (
        "the paid sibling must not be cooled down"
    )
    assert not provider._is_provider_unavailable(
        provider._failure_domain_key(_free_entry())
    ), "the account quota must not poison the shared failure domain"


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason=_RED_REASON)
async def test_combined_fallback_cycle_quarantines_failing_account(free_paid_chain):
    """AC8: the combined local+remote loop applies the same quarantine."""
    call_order: list[str] = []
    with patch(
        "proxy.server.proxy_to_remote", _chain_mock(call_order)
    ):
        result = await provider._proxy_with_fallback_cycle(
            _DummyRequest(),
            "v1/chat/completions",
            free_paid_chain,
            {"provider_cooldown_seconds": 60},
        )

    assert result.status_code == 200
    assert call_order == ["munitoai-free", "munitoai"], (
        f"a quota 429 must advance the chain to the sibling, got {call_order}"
    )
    _assert_failing_account_quarantined(_ACCOUNT_QUARANTINE_SECONDS)
    assert not provider._is_provider_unavailable("munitoai")


@pytest.mark.asyncio
async def test_combined_fallback_cycle_without_reset_signal_does_not_quarantine(
    free_paid_chain,
):
    """AC5/AC8: a reset-header-less quota 429 leaves the account unquarantined
    in both loops' shared path (only the short entry cooldown applies)."""
    call_order: list[str] = []
    with patch(
        "proxy.server.proxy_to_remote", _chain_mock(call_order, headers=None)
    ):
        result = await provider._proxy_with_fallback_cycle(
            _DummyRequest(),
            "v1/chat/completions",
            free_paid_chain,
            {"provider_cooldown_seconds": 60},
        )

    assert result.status_code == 200
    assert call_order == ["munitoai-free", "munitoai"]
    assert provider._usage_reset_at == {}
    assert provider._provider_unavailable_until["munitoai-free"] - time.time() <= (
        provider._BACKOFF_BASE_SECONDS + 1
    )


# ===========================================================================
# AC3 — chain-hold exhaustion derives retry_after from the quarantine
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason=_RED_REASON)
async def test_quota_429_exhaustion_retry_after_derives_from_account_quarantine():
    """AC3: when the quota 429 exhausts the whole chain, the chain-hold cycles
    still apply and the terminal exhaustion response advertises a retry_after
    taken from the account quarantine (not the short entry cooldown)."""
    model_config = {"providers": [_free_entry()]}
    call_count = 0

    async def _always_quota_429(_req, _path, _provider_cfg):
        nonlocal call_count
        call_count += 1
        return _quota_429(_RESET_HEADERS)

    config = {
        "provider_cooldown_seconds": 60,
        # Tiny holds keep the test fast; the total budget stays the default.
        "chain_hold_seconds": 0.01,
        "chain_hold_max_cycles": 2,
    }

    with patch("proxy.server.proxy_to_remote", _always_quota_429):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", model_config, config
        )

    body = json.loads(result.body)
    assert result.status_code == 503
    assert body["code"] == provider.EXHAUSTION_CODE_ALL
    # The quarantine (152 s) outlives the 34 s entry cooldown, so the
    # advertised wait is the account reset + margin.
    assert body["retry_after"] == pytest.approx(
        _ACCOUNT_QUARANTINE_SECONDS, abs=2
    )
    assert result.headers.get("retry-after") == str(body["retry_after"])
    # Chain-hold held the exhausted request, but the quarantined account was
    # never re-dispatched in the later cycles.
    assert call_count == 1, (
        f"the quarantined account must not be re-dispatched, got {call_count} calls"
    )


# ===========================================================================
# AC6 — FreeUsageLimitError (3 h) and GoUsageLimitError regressions
# ===========================================================================


@pytest.mark.asyncio
async def test_free_usage_limit_error_keeps_three_hour_cooldown():
    """AC6: FreeUsageLimitError without a reset time keeps the 3 h cooldown
    and creates no account quarantine (unchanged behaviour). Uses a DIFFERENT-
    domain fallback because the Free branch poisons the shared failure domain."""
    chain = {
        "providers": [
            _free_entry(),
            {
                "name": "deepseek-flash",
                "type": "remote",
                "provider": "deepseek",
                "endpoint": "https://api.deepseek.com",
                "api_key_env": "DEEPSEEK_API_KEY",
                "model": "deepseek-flash",
            },
        ],
        "aliases": ["test*"],
    }
    body = json.dumps(
        {
            "type": "error",
            "error": {
                "type": "FreeUsageLimitError",
                "message": "Rate limit exceeded. Please try again later.",
            },
            "metadata": {},
        }
    )
    call_order: list[str] = []

    async def _mock(_req, _path, provider_cfg):
        call_order.append(provider_cfg.get("name"))
        if provider_cfg.get("name") == "munitoai-free":
            return Response(
                content=body.encode("utf-8"),
                status_code=429,
                media_type="application/json",
            )
        return _ok_json_response()

    with patch("proxy.server.proxy_to_remote", _mock):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(),
            "v1/chat/completions",
            chain,
            {"provider_cooldown_seconds": 60},
        )

    assert result.status_code == 200
    assert call_order == ["munitoai-free", "deepseek-flash"]
    cooldown = provider._provider_unavailable_until["munitoai-free"] - time.time()
    assert cooldown == pytest.approx(
        provider._FREE_USAGE_LIMIT_COOLDOWN_SECONDS, abs=5
    )
    assert provider._usage_reset_at == {}


@pytest.mark.asyncio
async def test_go_usage_limit_error_still_quarantines_account(free_paid_chain):
    """AC6: GoUsageLimitError keeps its explicit-reset account quarantine
    (message duration + 2-minute margin) with no entry cooldown."""
    body = json.dumps(
        {
            "type": "error",
            "error": {
                "type": "GoUsageLimitError",
                "message": "Weekly usage limit reached. Resets in 2hr.",
            },
            "metadata": {"limitName": "weekly"},
        }
    )
    call_order: list[str] = []

    async def _mock(_req, _path, provider_cfg):
        call_order.append(provider_cfg.get("name"))
        if provider_cfg.get("name") == "munitoai-free":
            return Response(
                content=body.encode("utf-8"),
                status_code=429,
                media_type="application/json",
            )
        return _ok_json_response()

    with patch("proxy.server.proxy_to_remote", _mock):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(),
            "v1/chat/completions",
            free_paid_chain,
            {"provider_cooldown_seconds": 60},
        )

    assert result.status_code == 200
    assert call_order == ["munitoai-free", "munitoai"]
    free_account, paid_account = _account_keys()
    assert provider._usage_reset_at[free_account] - time.time() == pytest.approx(
        2 * 3600 + provider._USAGE_LIMIT_RESET_MARGIN_SECONDS, abs=5
    )
    assert paid_account not in provider._usage_reset_at
    assert not provider._is_provider_unavailable("munitoai-free")
    assert not provider._is_provider_unavailable("munitoai")
