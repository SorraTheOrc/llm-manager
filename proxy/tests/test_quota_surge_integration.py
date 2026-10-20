"""End-to-end integration test and regression gate for the quota surge epic.

This is the final verification slice of the quota-aware OpenCode Go surge
epic (LP-0MVF4FIP1001RD8P, feature F9 — LP-0MVF7KSJW007L2QF): it stands up a
**fake upstream HTTP server** on 127.0.0.1 serving both ``GET /v1/usage``
and ``POST /v1/chat/completions``, wires the proxy's real provider
resolution to it, and drives a representative request sequence through the
**real** fallback dispatch cycle — no live network, no sockets to
opencode.ai.

The flow (each step asserted independently so a failure names the phase):

1. Three ``opencode-*`` accounts with the observed live percentages
   (88% / 95% / 5%) → headroom-aware ordering sends the dispatch to the 5%
   account (the pacing ramp makes the big-residual account urgent even
   though its reset is still outside the final week).
2. Advance the clock into the account's final week → the ``available_times``
   bypass becomes active and the dispatch is allowed at 02:00 UTC (outside
   the configured ``10:00-00:00`` window), with ``reason=quota_surge``
   logged exactly once per state change.
3. Cross the hard stop (99.6%) mid-surge → the account is skipped on the
   very next dispatch with **zero** further upstream calls (the chain
   exhausts instead of dispatching).
4. Age the snapshots past ``server.quota_state_max_age_seconds`` → surge is
   suspended (fail-safe): outside the window nothing is dispatched, and back
   inside the window routing reverts to configured ``available_times`` +
   config order instead of the headroom ranking.

Regression assertions (the "existing suite is unaffected" claim):

- With ``server.quota_surge_enabled: false`` the same flow keeps the
  configured ``available_times`` window and config order, byte-identical to
  the pre-feature behaviour (AC8).
- The pre-existing routing gates the epic must never weaken — usage-limit
  quarantine, the ``rate_limit_rpm`` budget and mid-chain fallback — keep
  working under surge through the same real resolution path (AC6 / R7).

The poller runs through its **default HTTP transport** (a real GET against
the fake ``/v1/usage`` endpoint) and the dispatch through the real
``_proxy_with_remote_fallback_cycle``; only the upstream *host* is the fake
server. Keys are served per ``Authorization`` Bearer token so the upstream
counts exactly which account was polled/dispatched.

Full-suite regression evidence (the existing ``available_times``, chain
fallback, usage-limit quarantine and provider-budget suites passing
unchanged) is recorded per AC11c at the branch head, not re-asserted here —
see the work-item comment.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock

import httpx
import proxy.provider as provider
import proxy.provider_budget as provider_budget
import proxy.quota_state as quota_state
import pytest
from fastapi import Response
from proxy.quota_state import reset_quota_state
from proxy.rate_limiter import get_rate_limiter

pytestmark = pytest.mark.integration

#: Frozen start of the simulation: 12:00 UTC, inside the configured
#: ``10:00-00:00`` window.
BASE = datetime(2026, 10, 19, 12, 0, 0, tzinfo=UTC)

#: 02:00 UTC on day 3 is outside the ``10:00-00:00`` window.
_OFF_PEAK = datetime(2026, 10, 22, 2, 0, 0, tzinfo=UTC)

_PROVIDER_LOGGER = "llama-proxy.provider"

KEY_GO = "OPENCODE_API_KEY"
KEY_GO_2 = "OPENCODE_2_API_KEY"
KEY_GO_3 = "OPENCODE_3_API_KEY"

#: Env var -> the Bearer token the fixture sets for it (what the upstream sees).
_TOKENS = {KEY_GO: "token-go", KEY_GO_2: "token-go-2", KEY_GO_3: "token-go-3"}
_TOKEN_TO_ENV = {token: env for env, token in _TOKENS.items()}

_OK_CHAT = {
    "id": "chatcmpl_integration",
    "object": "chat.completion",
    "model": "deepseek-v4-flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}


# ---------------------------------------------------------------------------
# Fake upstream: a real local HTTP server serving /v1/usage + chat/completions
# ---------------------------------------------------------------------------


class _FakeUpstream:
    """Loopback ``GET /v1/usage`` + ``POST /v1/chat/completions`` server.

    Each account is identified by its ``Authorization: Bearer <token>``; the
    usage payload is per-token mutable state (the scenario evolves the
    accounts' quota between steps) and every call is counted per token so
    the test can assert exactly which account was polled or dispatched —
    including the "**zero** further upstream calls" hard-stop assertion.
    """

    def __init__(self) -> None:
        self.usage: dict[str, dict] = {}
        self.fail_chat: dict[str, int] = {}
        self.usage_calls: list[str] = []
        self.chat_calls: list[str] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> _FakeUpstream:
        upstream = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # silence test noise
                pass

            def _token(self) -> str:
                header = self.headers.get("Authorization", "")
                return header.removeprefix("Bearer ").strip()

            def do_GET(self) -> None:
                if self.path.endswith("/v1/usage"):
                    token = self._token()
                    upstream.usage_calls.append(token)
                    body = upstream.usage.get(token)
                    if body is None:
                        upstream._respond(self, 404, {"error": "unknown key"})
                        return
                    upstream._respond(self, 200, body)
                    return
                upstream._respond(self, 404, {"error": "not found"})

            def do_POST(self) -> None:
                if self.path.endswith("/v1/chat/completions"):
                    token = self._token()
                    length = int(self.headers.get("Content-Length") or 0)
                    if length:
                        self.rfile.read(length)
                    upstream.chat_calls.append(token)
                    status = upstream.fail_chat.get(token)
                    if status is not None:
                        upstream._respond(
                            self, status, {"error": {"message": "upstream failure"}}
                        )
                        return
                    upstream._respond(self, 200, _OK_CHAT)
                    return
                upstream._respond(self, 404, {"error": "not found"})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()
        return self

    @staticmethod
    def _respond(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # -- counters -----------------------------------------------------------

    @staticmethod
    def _token_for(name: str) -> str:
        """Accept an env-var name or the raw token the upstream records."""
        return _TOKENS.get(name, name)

    def chat_count(self, name: str) -> int:
        """Dispatch calls for an account, named by env var or token."""
        return self.chat_calls.count(self._token_for(name))

    def chat_total(self) -> int:
        return len(self.chat_calls)

    def usage_count(self, name: str) -> int:
        """``/v1/usage`` polls for an account, named by env var or token."""
        return self.usage_calls.count(self._token_for(name))

    @property
    def endpoint(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}/zen/go"


# ---------------------------------------------------------------------------
# Fixtures / scenario helpers
# ---------------------------------------------------------------------------


class _Clock:
    """Deterministic wall clock shared by quota_state and provider.datetime."""

    def __init__(self, start: datetime = BASE):
        self.now = start

    def __call__(self) -> datetime:
        """Serve as the injectable quota clock."""
        return self.now

    def set(self, value: datetime) -> None:
        self.now = value

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


class _FakeDatetime:
    """Stand-in for ``proxy.provider.datetime`` bound to the frozen clock."""

    def __init__(self, clock: _Clock):
        self._clock = clock

    def now(self, tz=None):
        return self._clock.now

    def fromtimestamp(self, ts, tz=None):
        return datetime.fromtimestamp(ts, tz)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _usage_body(
    monthly: float,
    *,
    monthly_resets_at: datetime,
    rolling: float = 0.0,
    weekly: float = 0.0,
) -> dict:
    """A ``/v1/usage`` body whose monthly reset sits at an absolute instant."""
    return {
        "usage": {
            "rolling": {
                "status": "ok",
                "percent": rolling,
                "resetsAt": _iso(monthly_resets_at + timedelta(hours=5)),
            },
            "weekly": {
                "status": "ok",
                "percent": weekly,
                "resetsAt": _iso(monthly_resets_at - timedelta(days=2)),
            },
            "monthly": {
                "status": "ok",
                "percent": monthly,
                "resetsAt": _iso(monthly_resets_at),
            },
        }
    }


def _provider(name: str, api_key_env: str) -> dict:
    return {
        "name": name,
        "type": "remote",
        "provider": "opencode-go",
        "endpoint": None,  # filled from the fake upstream by the fixture
        "api_key_env": api_key_env,
        "model": "deepseek-v4-flash",
        "available_times": ["10:00-00:00"],
    }


def _config(**server: object) -> dict:
    return {
        "server": {
            "quota_poll_interval_seconds": 900,
            "quota_state_max_age_seconds": 3600,
            "quota_surge_days": 7,
            "quota_exhaustion_percent": 99.5,
            **server,
        },
        "provider_cooldown_seconds": 60,
    }


class _Request:
    """Minimal request stub (body + session header) for the fallback cycle."""

    def __init__(self, session_id: str = "surge-integration-session"):
        self._body = json.dumps(
            {"model": "surge-test", "messages": [{"role": "user", "content": "hi"}]}
        ).encode()
        self.headers = {"content-type": "application/json"}
        if session_id:
            self.headers["x-session-id"] = session_id
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body

    async def is_disconnected(self):
        return False


async def _noop_async(*args, **kwargs):
    return None


def _fake_remote_dispatch(upstream: _FakeUpstream):
    """The ``_get_proxy_to_remote`` shim: a REAL HTTP POST to the fake upstream.

    This is the only stubbed "proxy layer" — the resolution, the chain
    gating, the quota decisions and the wire call itself are all real; only
    the upstream host is the loopback fake instead of opencode.ai.
    """

    async def _dispatch(request, path: str, provider_cfg: dict) -> Response:
        endpoint = provider_cfg["endpoint"]
        token = os.environ[provider_cfg["api_key_env"]]
        async with httpx.AsyncClient(timeout=10) as client:
            upstream_response = await client.post(
                f"{endpoint}/{path}",
                content=await request.body(),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            media_type="application/json",
        )

    return _dispatch


@pytest.fixture
def s(monkeypatch):
    """Fresh fake upstream + frozen clock + isolated provider/quota state."""
    upstream = _FakeUpstream().start()
    clock = _Clock(BASE)

    quota_state.reset_quota_state()
    provider.reset_surge_bypass_state()
    quota_state.set_clock(clock)
    monkeypatch.setattr(provider, "datetime", _FakeDatetime(clock))
    monkeypatch.setattr(provider, "request_priority", lambda *a, **k: "medium")
    monkeypatch.setattr("proxy.priority.request_priority", lambda *a, **k: "medium")
    monkeypatch.setattr(
        provider, "_get_tokenizer_for_model", MagicMock(return_value=(None, 1.0))
    )
    monkeypatch.setattr(provider, "_maybe_probe_quarantined_accounts", _noop_async)
    monkeypatch.setattr(
        provider, "_get_proxy_to_remote", lambda: _fake_remote_dispatch(upstream)
    )
    for key, token in _TOKENS.items():
        monkeypatch.setenv(key, token)

    model_config = {
        "name": "surge-integration",
        "providers": [
            _provider("opencode-go", KEY_GO),
            _provider("opencode-go-2", KEY_GO_2),
            _provider("opencode-go-3", KEY_GO_3),
        ],
    }
    for entry in model_config["providers"]:
        entry["endpoint"] = upstream.endpoint
    config = _config()

    class _Scenario:
        pass

    s = _Scenario()
    s.upstream = upstream
    s.clock = clock
    s.model_config = model_config
    s.config = config

    yield s

    upstream.stop()
    reset_quota_state()
    provider.reset_surge_bypass_state()
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()
    provider_budget._affinity.clear()


async def _poll(s, *, force: bool = True) -> None:
    """Poll every account through the real default HTTP transport."""
    await quota_state.refresh_quota_accounts(s.model_config, s.config, force=force)


async def _dispatch(s) -> Response:
    """Drive the REAL fallback dispatch cycle against the fake upstream."""
    return await provider._proxy_with_remote_fallback_cycle(
        _Request(), "v1/chat/completions", s.model_config, s.config
    )


def _surge_logs(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno <= logging.INFO
        and "quota_surge_bypass" in record.getMessage()
        and "entered" in record.getMessage()
    ]


def _ranked_names(s) -> list[str]:
    """The providers in the order the chain would actually iterate them."""
    return [
        entry["name"]
        for entry in provider._quota_ranked_providers(s.model_config, s.config)
    ]


# ---------------------------------------------------------------------------
# The four-step integrated flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_flow_poll_order_surge_hard_stop_stale(s, caplog):
    """AC11b: poll → derive → order → surge → hard stop → stale, one flow.

    The whole feature is proven as one flow against the fake upstream with
    the real poller transport and the real fallback dispatch cycle.
    """
    # Observed live percentages; account 3 (5%) keeps a large residual whose
    # pacing ramp makes it the urgent pick before its final week begins. The
    # other two resets sit beyond the 7-day surge horizon for the whole flow,
    # so account 3 is the ONLY in-window-eligible entry at off-peak hours.
    month_11 = BASE + timedelta(days=11)
    month_13 = BASE + timedelta(days=13)
    month_8_5 = BASE + timedelta(days=8, hours=12)
    s.upstream.usage = {
        "token-go": _usage_body(88.0, monthly_resets_at=month_11),
        "token-go-2": _usage_body(95.0, monthly_resets_at=month_13),
        "token-go-3": _usage_body(5.0, monthly_resets_at=month_8_5),
    }

    # -- Step 1: ordering prefers the 5% account (AC3) ---------------------
    await _poll(s)
    # 12:00 UTC: all three inside the configured window; the pacing ramp
    # (R5) ranks the 95%-headroom account first even beyond the final week.
    assert _ranked_names(s) == ["opencode-go-3", "opencode-go", "opencode-go-2"]
    assert s.upstream.usage_count(KEY_GO) == 1, "each account polled exactly once"
    assert s.upstream.usage_count(KEY_GO_3) == 1

    response = await _dispatch(s)
    assert response.status_code == 200
    assert s.upstream.chat_count(KEY_GO_3) == 1, "dispatch must go to the 5% account"
    assert s.upstream.chat_total() == 1

    # -- Step 2: final-week bypass at 02:00 UTC (AC5) ----------------------
    # Advance 2d14h → 02:00 UTC, outside the 10:00-00:00 window. Account 3's
    # reset is now 5d22h away (inside the 7-day surge window): the bypass
    # lifts its window restriction and the dispatch is allowed.
    s.clock.set(_OFF_PEAK)
    await _poll(s)  # fresh poll at the new clock -> fresh surge horizon
    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        assert _ranked_names(s)[0] == "opencode-go-3"
        for _ in range(2):
            response = await _dispatch(s)
            assert response.status_code == 200
    assert s.upstream.chat_count(KEY_GO_3) == 3, "bypass admits off-peak dispatch"
    assert s.upstream.chat_count(KEY_GO) == 0, "no other account called"

    entered = _surge_logs(caplog)
    assert len(entered) == 1, f"expected exactly one entering transition: {entered}"
    assert "reason=quota_surge" in entered[0]
    assert f"account={KEY_GO_3}" in entered[0]

    # -- Step 3: hard stop (99.6%) mid-surge, zero further upstream calls --
    # Account 3 crosses the 99.5% money guard while still surged: the very
    # next dispatch skips it and — with every sibling outside the window —
    # the chain exhausts instead of calling out (AC10).
    s.upstream.usage["token-go-3"] = _usage_body(99.6, monthly_resets_at=month_8_5)
    s.clock.advance(3600)  # 03:00 UTC
    await _poll(s)
    calls_before = s.upstream.chat_calls.copy()

    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        with pytest.raises(provider.ChainExhaustedError):
            await _dispatch(s)

    assert s.upstream.chat_calls == calls_before, (
        "the hard-stopped account must not be called again"
    )
    assert any(
        "reason=quota_exhausted" in record.getMessage()
        and f"account={KEY_GO_3}" in record.getMessage()
        for record in caplog.records
    ), "expected an INFO quota_exhausted skip"

    # -- Step 4: stale snapshots suspend the surge (AC9) -------------------
    # Restore account 3 to 5% and re-poll (fresh at 03:00 UTC, still surged),
    # then age past quota_state_max_age_seconds WITHOUT polling again: the
    # fail-safe suspends the bypass — at 05:00 UTC (outside the window) the
    # account the bypass would otherwise admit is now blocked, so nothing is
    # dispatched.
    s.upstream.usage["token-go-3"] = _usage_body(5.0, monthly_resets_at=month_8_5)
    await _poll(s)
    s.clock.advance(2 * 3600)
    behaviour = quota_state.quota_behaviour(s.config, now=s.clock.now)
    assert behaviour.suspended is True, "aged snapshots must suspend the surge"

    calls_before = s.upstream.chat_calls.copy()
    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        with pytest.raises(provider.ChainExhaustedError):
            await _dispatch(s)
    assert s.upstream.chat_calls == calls_before, (
        "stale state must revert to the configured window (nothing outside it)"
    )
    suspended_warns = [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
        and "quota surge suspended" in record.getMessage()
    ]
    assert suspended_warns, "expected a WARN for the stale-state suspension"

    # Back inside the window the suspension also reverts ORDERING to config
    # order: with a fresh (surged) snapshot the 5% account is ranked first;
    # with stale state the configured order (go, go-2, go-3) wins.
    s.clock.set(BASE + timedelta(days=3))  # 12:00 UTC, inside the window
    await _poll(s)
    assert _ranked_names(s) == [
        "opencode-go-3",
        "opencode-go",
        "opencode-go-2",
    ], "fresh surge state keeps the headroom ranking"
    s.clock.advance(2 * 3600)  # age past max age again
    assert _ranked_names(s) == [
        "opencode-go",
        "opencode-go-2",
        "opencode-go-3",
    ], "suspended routing must revert to configured order"

    response = await _dispatch(s)
    assert response.status_code == 200
    assert s.upstream.chat_count(KEY_GO) == 1, "config-order entry receives the call"
    assert s.upstream.chat_count(KEY_GO_3) == 3, "5% account untouched while suspended"


# ---------------------------------------------------------------------------
# Regression: the feature's quiet mode keeps the pre-surge routing (AC8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_feature_keeps_window_and_config_order(s, caplog):
    """AC8: ``quota_surge_enabled: false`` → byte-identical routing.

    With the kill switch off the snapshots are still polled but the chain
    reverts to the configured ``available_times`` window and config order:
    at 02:00 UTC nothing dispatches; at 12:00 UTC the first configured entry
    — not the headroom-favourite — receives the call.
    """
    s.config = _config(quota_surge_enabled=False)
    month_8 = BASE + timedelta(days=8)
    month_12 = BASE + timedelta(days=12)
    month_8_5 = BASE + timedelta(days=8, hours=12)
    s.upstream.usage = {
        "token-go": _usage_body(88.0, monthly_resets_at=month_8),
        "token-go-2": _usage_body(95.0, monthly_resets_at=month_12),
        "token-go-3": _usage_body(5.0, monthly_resets_at=month_8_5),
    }
    await _poll(s)

    assert provider._quota_surge_applies(s.config) is False
    assert _ranked_names(s) == ["opencode-go", "opencode-go-2", "opencode-go-3"]

    s.clock.set(_OFF_PEAK)
    with pytest.raises(provider.ChainExhaustedError):
        await _dispatch(s)
    assert s.upstream.chat_total() == 0, "kill switch keeps the off-peak window closed"

    s.clock.set(BASE + timedelta(days=3))
    response = await _dispatch(s)
    assert response.status_code == 200
    assert s.upstream.chat_count(KEY_GO) == 1, "config order, not headroom order"


# ---------------------------------------------------------------------------
# Regression: pre-existing routing gates keep working under surge (AC6/R7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_existing_routing_gates_unaffected_under_surge(s):
    """AC6/R7: the surge never weakens quarantine / budget / fallback gates.

    Uses the same real resolution path the pre-existing suites cover and
    proves each gate still excludes the headroom-favourite (surged) account
    on the next decision.
    """
    # Account 3 (5%, 4 days to reset) is the surged headroom favourite.
    month_4 = BASE + timedelta(days=4)
    s.upstream.usage = {
        "token-go": _usage_body(88.0, monthly_resets_at=BASE + timedelta(days=8)),
        "token-go-2": _usage_body(95.0, monthly_resets_at=BASE + timedelta(days=12)),
        "token-go-3": _usage_body(5.0, monthly_resets_at=month_4),
    }
    await _poll(s)
    go, _go2, go3 = s.model_config["providers"]
    assert provider._quota_surge_bypass_active(go3, s.config) is True
    assert provider._quota_surge_bypass_active(go, s.config) is False
    assert _ranked_names(s)[0] == "opencode-go-3"

    # 1. Usage-limit quarantine (LP-0MSLJPOCC0001ROJ): a pending reset for
    #    the favourite skips it even though the quota ranking wants it.
    quarantined = provider._usage_limit_account_key(go3)
    provider._usage_reset_at[quarantined] = time.time() + 600
    try:
        chosen = provider.resolve_provider(s.model_config, config=s.config)
        assert chosen["name"] != "opencode-go-3", "quarantine must precede quota ranking"
    finally:
        provider._usage_reset_at.pop(quarantined, None)

    # 2. Rate-limit RPM budget on the favourite (LP-0MVDQ1GMO00801PL): a
    #    spent window excludes it even under surge — the chain hops to the
    #    next entry without calling the exhausted account.
    go3["rate_limit_rpm"] = 1
    limiter = get_rate_limiter()
    limiter.reset("opencode-go-3")
    assert await limiter.check_and_increment("opencode-go-3", 1, 60) is True
    try:
        response = await _dispatch(s)
        assert response.status_code == 200
        assert s.upstream.chat_count(KEY_GO_3) == 0, "rate-limited entry not called"
        assert s.upstream.chat_count(KEY_GO) == 1, "chain falls through to next entry"
    finally:
        limiter.reset("opencode-go-3")
        go3.pop("rate_limit_rpm", None)

    # 3. Mid-chain fallback keeps the same (quota) ranking for the NEXT
    #    eligible entry after the favourite has been attempted (AC3b).
    #    ``_resolve_provider_with_exclusions`` is the real fallback path (the
    #    same-gateway sibling exclusion of ``failed_provider`` is a separate,
    #    intentional rule and is covered by the chain-fallback suites).
    assert (
        provider._resolve_provider_with_exclusions(
            s.model_config, {"opencode-go-3"}, set(), config=s.config
        )["name"]
        == "opencode-go"
    )
