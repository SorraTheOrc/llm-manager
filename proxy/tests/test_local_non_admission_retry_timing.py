"""Non-admission local-route exits advertise their *real* retry expiry.

LP-0MVQ23B80009OZ9V (AC1–AC5; covers AC3 of LP-0MVNTT4X8009IO2X).

The parent contract says a local-only refusal answers two questions: *whether*
waiting helps and, when it does, *when*. The ETA/occupancy reasons answer
"when" from the projected prefill (:func:`proxy.prefill_eta.
suggested_retry_after_seconds`). Every other local-route exit has a real,
finite availability expiry of its own, and this module pins that the refusal
advertises **that** expiry instead of the ETA formula:

- a cooling-down local entry → the cooldown remainder
  (``_provider_unavailable_until`` — the state behind
  ``unavailable_providers``);
- a quarantined local account → the usage-limit reset remainder
  (``_usage_reset_at``);
- a held dispatch lease → the lease remainder
  (``router_helpers._get_lease_timeout_seconds`` / the live dispatch record);
- a contention-queue exit → the queue's wait window
  (``contention_queue_max_wait_seconds``).

``context_too_large`` is the exception: no amount of waiting makes a prompt
fit a KV slot, so it must behave like ``prefill_budget_exceeded`` — no
``Retry-After``, ``retryable: false`` and compaction guidance (AC5).

Each case stubs the ETA formula to a sentinel value, so an ETA-derived answer
can never masquerade as a real expiry (and vice versa). Every case is a
``local-only`` request with a remote provider available: the assertion
``calls == []`` proves the refusal was returned instead of silently moving the
request remote.
"""

from __future__ import annotations

import json
import time
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import proxy.provider as provider
import pytest
from fastapi import Response

from proxy import contention_queue, prefill_eta

LOCAL_PROVIDER_NAME = "local-llama"
REMOTE_PROVIDER_NAME = "remote-fallback"
LOCAL_ONLY = "local-only"

#: What the ETA formula returns while these cases run. Real-expiry timers are
#: asserted to be a different value, so a regression to the ETA formula is an
#: explicit assertion failure rather than a coincidental pass.
ETA_FORMULA_SENTINEL_SECONDS = 999

#: Real expiries the cases advertise. Each is far above the 1 s floor so an
#: ETA-formula answer cannot be mistaken for the real value.
COOLDOWN_REMAINDER_SECONDS = 90
USAGE_RESET_REMAINDER_SECONDS = 120
LEASE_REMAINDER_SECONDS = 75
QUEUE_WINDOW_SECONDS = 45

#: Tolerance (seconds) on the absolute ``retry_at`` instant.
RETRY_AT_TOLERANCE_SECONDS = 5

#: Prompt priced by the ``context_too_large`` case, and the flat cold-prefill
#: cap advertised as token evidence (the same numbers the refusal contract
#: tests use).
PROMPT_NEW_TOKENS = 20_000
COLD_PREFILL_BUDGET_TOKENS = 25_000


class _DummyRequest:
    """Minimal request stub (matches the other admission test helpers)."""

    def __init__(self, body: bytes = b'{"model":"test"}', headers=None):
        self._body = body
        self.headers = headers or {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


class _MutableConcurrency:
    """``_get_local_concurrency_info`` stand-in."""

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


def _lease_active_response() -> Response:
    """The router's lease-held 503 (``no_slots_available``)."""
    return Response(
        content=json.dumps(
            {
                "error": {
                    "type": "server_busy",
                    "code": "no_slots_available",
                    "message": "Local backend busy. Owner session holds the lease.",
                },
                "status": 503,
                "reason": "local_lease_active",
                "local_owner_session_id": "owner-session",
            }
        ),
        status_code=503,
        media_type="application/json",
    )


def _model_config() -> dict:
    return {
        "providers": [
            {"name": LOCAL_PROVIDER_NAME, "type": "local", "llama_model": "Qwen3"},
            {
                "name": REMOTE_PROVIDER_NAME,
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
            },
        ],
        "aliases": ["hybrid*"],
    }


def _config(**server_overrides) -> dict:
    """Server config with every unrelated gate disabled/inert."""
    server = {
        "session_slot_pool_size": 1,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": QUEUE_WINDOW_SECONDS,
        "contention_queue_max_depth": 4,
        "local_model_ctx_size": 0,
        "local_prefill_deadline_seconds": 0,
        "local_cold_prefill_budget_tokens": COLD_PREFILL_BUDGET_TOKENS,
    }
    server.update(server_overrides)
    return {"provider_cooldown_seconds": 60, "server": server}


@contextmanager
def _estimate_patches(tokens: int, cached_ratio: float = 0.0):
    """Price every prompt exactly *tokens* (cache-cold by default)."""
    with (
        patch(
            "proxy.provider._estimate_effective_prompt_tokens_for_routing",
            new=AsyncMock(return_value=tokens),
        ),
        patch("proxy.provider._get_tokenizer_for_model", lambda *_: (None, 1.0)),
        patch("proxy.provider._get_cached_ratio", lambda *_: cached_ratio),
    ):
        yield


@contextmanager
def _local_entry_in_cooldown(seconds: float):
    """Put the local entry into a *real* provider cooldown."""
    key = LOCAL_PROVIDER_NAME
    previous = provider._provider_unavailable_until.get(key)
    # Half a second past the whole second, so ``int(remaining)`` is exact.
    provider._provider_unavailable_until[key] = time.time() + seconds + 0.5
    try:
        yield
    finally:
        if previous is None:
            provider._provider_unavailable_until.pop(key, None)
        else:
            provider._provider_unavailable_until[key] = previous


@contextmanager
def _local_account_quarantined(seconds: float):
    """Quarantine the local entry's usage-limit account (``_usage_reset_at``)."""
    account = provider._usage_limit_account_key(
        {"name": LOCAL_PROVIDER_NAME, "type": "local", "llama_model": "Qwen3"}
    )
    previous = provider._usage_reset_at.get(account)
    provider._usage_reset_at[account] = time.time() + seconds + 0.5
    try:
        yield
    finally:
        if previous is None:
            provider._usage_reset_at.pop(account, None)
        else:
            provider._usage_reset_at[account] = previous


@pytest.fixture(autouse=True)
def _clean_state():
    contention_queue.reset()
    prefill_eta.reset()
    provider._provider_unavailable_until.pop(LOCAL_PROVIDER_NAME, None)
    local_account = provider._usage_limit_account_key(
        {"name": LOCAL_PROVIDER_NAME, "type": "local", "llama_model": "Qwen3"}
    )
    provider._usage_reset_at.pop(local_account, None)
    yield
    contention_queue.reset()
    prefill_eta.reset()
    provider._provider_unavailable_until.pop(LOCAL_PROVIDER_NAME, None)
    provider._usage_reset_at.pop(local_account, None)


async def _issue(
    *,
    preference: str | None = LOCAL_ONLY,
    config: dict | None = None,
    concurrency: tuple[int, int] = (0, 1),
    local_responder=None,
    extra_patches: list | None = None,
) -> tuple[Response, list[str]]:
    """Issue a request; return ``(response, upstream dispatches)``.

    ``dispatches`` is the ordered record of upstream calls: ``[]`` proves the
    refusal was returned instead of being dispatched anywhere.
    """
    calls: list[str] = []
    responder = local_responder or _local_ok

    async def _local(_req, _path):
        calls.append("local")
        return responder()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    headers = {} if preference is None else {"X-Local-Preference": preference}
    with ExitStack() as stack:
        stack.enter_context(patch("proxy.router.proxy_to_local", _local))
        stack.enter_context(patch("proxy.server.proxy_to_remote", _remote))
        stack.enter_context(
            patch(
                "proxy.provider._get_local_concurrency_info",
                _MutableConcurrency(*concurrency),
            )
        )
        stack.enter_context(patch("proxy.provider.request_priority", return_value="low"))
        stack.enter_context(
            patch(
                "proxy.prefill_eta.suggested_retry_after_seconds",
                lambda *a, **k: ETA_FORMULA_SENTINEL_SECONDS,
            )
        )
        for ctx in extra_patches or []:
            stack.enter_context(ctx)
        result = await provider.proxy_with_fallback(
            _DummyRequest(headers=headers),
            "v1/chat/completions",
            _model_config(),
            config or _config(),
        )
    return result, calls


def _local_ok() -> Response:
    return _ok_response()


def _parse_rfc3339_utc(value) -> datetime:
    """Parse an RFC 3339 UTC timestamp, failing loudly for anything else."""
    assert value, "the refusal body must carry an absolute retry_at"
    text = str(value)
    assert text.endswith(("Z", "+00:00")), (
        f"retry_at must be RFC 3339 UTC, got {text!r}"
    )
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    assert parsed.utcoffset() == timedelta(0), f"retry_at must be UTC, got {text!r}"
    return parsed


def _assert_real_expiry_refusal(
    result: Response,
    dispatched: list[str],
    expected_seconds: int,
    *,
    allow_local_attempt: bool = False,
) -> dict:
    """The shared contract of every real-expiry refusal.

    *allow_local_attempt* is for the lease-held case: a lease denial is only
    observable by attempting the local dispatch, so ``["local"]`` is expected
    there — what must never happen is a *remote* dispatch behind a
    ``local-only`` client's back.
    """
    expected_dispatches = ["local"] if allow_local_attempt else []
    assert dispatched == expected_dispatches, (
        "a local-only request must be refused, never moved to a remote provider "
        f"(dispatched={dispatched})"
    )
    assert result.status_code == 429, f"got {result.status_code}"
    assert result.headers.get("X-Local-Route") == "refused"
    assert result.headers.get("Retry-After") == str(expected_seconds), (
        "Retry-After must be the real expiry remainder, got "
        f"{result.headers.get('Retry-After')!r}"
    )
    assert int(result.headers["Retry-After"]) != ETA_FORMULA_SENTINEL_SECONDS, (
        "the timer must come from real availability state, not the ETA formula"
    )

    body = json.loads(result.body)
    assert body.get("retryable") is True, "a bounded wait is retryable"
    assert body.get("retry_after") == expected_seconds, (
        "the body must mirror the Retry-After header"
    )
    retry_at = _parse_rfc3339_utc(body.get("retry_at"))
    expected_at = datetime.now(UTC) + timedelta(seconds=expected_seconds)
    assert abs((retry_at - expected_at).total_seconds()) <= RETRY_AT_TOLERANCE_SECONDS, (
        f"retry_at {retry_at.isoformat()} must be now + {expected_seconds}s, got "
        f"{expected_at.isoformat()}"
    )
    return body


# ---------------------------------------------------------------------------
# AC2: cooldown / usage-limit reset remainders (real cooldown state)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cooldown_refusal_advertises_the_cooldown_remainder():
    """A cooling-down local entry refuses for exactly its cooldown remainder."""
    with _local_entry_in_cooldown(COOLDOWN_REMAINDER_SECONDS):
        result, dispatched = await _issue()

    body = _assert_real_expiry_refusal(
        result, dispatched, COOLDOWN_REMAINDER_SECONDS
    )
    assert body.get("reason") == "local_provider_cooldown", (
        "the refusal must attribute the timer to the cooldown that caused it"
    )


@pytest.mark.asyncio
async def test_usage_limit_reset_refusal_advertises_the_reset_remainder():
    """A quarantined local account refuses for its reset remainder (AC2)."""
    with _local_account_quarantined(USAGE_RESET_REMAINDER_SECONDS):
        result, dispatched = await _issue()

    body = _assert_real_expiry_refusal(
        result, dispatched, USAGE_RESET_REMAINDER_SECONDS
    )
    assert body.get("reason") == "usage_limit_reset_pending", (
        "the refusal must attribute the timer to the pending quota reset"
    )


# ---------------------------------------------------------------------------
# AC3: lease-held refusal advertises the lease remainder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lease_held_refusal_advertises_the_lease_remainder():
    """A held dispatch lease is a bounded wait: advertise its remainder (AC3).

    No live dispatch record exists in this unit test, so the remainder is the
    configured ``local_dispatch_lease_timeout_seconds`` — the same bound
    ``router_helpers._get_lease_timeout_seconds`` reports.
    """
    with patch(
        "proxy.contention_queue.wait_for_local_slot",
        new=AsyncMock(return_value=None),
    ):
        result, dispatched = await _issue(
            config=_config(
                local_dispatch_lease_timeout_seconds=LEASE_REMAINDER_SECONDS,
            ),
            local_responder=_lease_active_response,
        )

    body = _assert_real_expiry_refusal(
        result, dispatched, LEASE_REMAINDER_SECONDS, allow_local_attempt=True
    )
    assert body.get("reason") == "local_lease_active", (
        "the refusal must attribute the timer to the held lease"
    )


# ---------------------------------------------------------------------------
# AC4: contention-queue expiry advertises the queue window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contention_queue_expiry_refusal_advertises_the_queue_window():
    """A request that waited out the queue window refuses with that window (AC4).

    The queue's expiry is its wait budget (``contention_queue_max_wait_seconds``):
    a client that retries immediately would spend the same window again, so the
    advertised delay is the window, not the ETA formula's 1 s floor.
    """
    with patch(
        "proxy.contention_queue.wait_for_local_slot",
        new=AsyncMock(return_value=None),
    ):
        result, dispatched = await _issue(
            config=_config(
                contention_queue_max_wait_seconds=QUEUE_WINDOW_SECONDS,
            ),
            concurrency=(1, 1),
        )

    body = _assert_real_expiry_refusal(result, dispatched, QUEUE_WINDOW_SECONDS)
    assert body.get("reason") == "fallback_after_queue", (
        "the refusal must attribute the timer to the queue exit"
    )


# ---------------------------------------------------------------------------
# AC1: the real expiry drives the timer for every non-admission exit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trigger", "expected_seconds"),
    [
        ("cooldown", COOLDOWN_REMAINDER_SECONDS),
        ("usage_limit_reset", USAGE_RESET_REMAINDER_SECONDS),
        ("lease_held", LEASE_REMAINDER_SECONDS),
        ("queue_expiry", QUEUE_WINDOW_SECONDS),
    ],
    ids=["cooldown", "usage_limit_reset", "lease_held", "queue_expiry"],
)
async def test_non_admission_exit_never_uses_the_eta_formula(
    trigger: str, expected_seconds: int
):
    """AC1 as one table: each exit's timer is its real expiry, not the ETA."""
    stack = ExitStack()
    with stack:
        local_responder = None
        concurrency = (0, 1)
        config = _config()
        if trigger == "cooldown":
            stack.enter_context(_local_entry_in_cooldown(expected_seconds))
        elif trigger == "usage_limit_reset":
            stack.enter_context(_local_account_quarantined(expected_seconds))
        elif trigger == "lease_held":
            local_responder = _lease_active_response
            config = _config(
                local_dispatch_lease_timeout_seconds=expected_seconds
            )
        elif trigger == "queue_expiry":
            concurrency = (1, 1)
            config = _config(
                contention_queue_max_wait_seconds=expected_seconds
            )
        else:  # pragma: no cover - a new trigger must be wired above
            raise AssertionError(f"unknown trigger {trigger!r}")

        if trigger in {"lease_held", "queue_expiry"}:
            stack.enter_context(
                patch(
                    "proxy.contention_queue.wait_for_local_slot",
                    new=AsyncMock(return_value=None),
                )
            )

        result, dispatched = await _issue(
            config=config,
            concurrency=concurrency,
            local_responder=local_responder,
        )

    _assert_real_expiry_refusal(
        result,
        dispatched,
        expected_seconds,
        allow_local_attempt=(trigger == "lease_held"),
    )


# ---------------------------------------------------------------------------
# AC5: context_too_large is not time-retryable (compact, do not wait)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_too_large_pin_local_is_promoted_not_refused():
    """An unfittable prompt: promoted to remote, never given a retry timer.

    The same shape as ``prefill_budget_exceeded`` (AC5): waiting cannot make
    the prompt fit a KV slot, so a delay header would invite futile retries.
    Since the pin slice (LP-0MVSLZY470051ZB2 AC4) the request is not refused at
    all — the pin is declined and the remote tiers serve it, which is the
    strongest form of "no timer": there is no local-capacity 429 to retry.
    """
    with (
        patch(
            "proxy.provider._should_skip_local",
            lambda *a, **k: (True, "context_too_large"),
        ),
        patch(
            "proxy.router_helpers.evaluate_and_apply_compaction",
            AsyncMock(
                return_value={
                    "evaluated": False,
                    "action": "noop",
                    "applied": False,
                    "estimated_before": 0,
                    "estimated_after": 0,
                    "reason": None,
                    "messages": None,
                }
            ),
        ),
        _estimate_patches(PROMPT_NEW_TOKENS),
    ):
        result, dispatched = await _issue()

    assert dispatched == ["remote"], (
        f"an oversized prompt must be promoted to the remote tier; "
        f"dispatched={dispatched}"
    )
    assert result.status_code == 200, f"got {result.status_code}"
    assert result.headers.get("X-Local-Route") == "remote"
    assert result.headers.get("Retry-After") is None, (
        "waiting cannot make an oversized prompt fit: a promoted request must "
        "not advertise a delay"
    )
    body = json.loads(result.body)
    assert body.get("code") != provider.EXHAUSTION_CODE_LOCAL_ADMISSION, (
        "a too-large request must not receive a local-capacity refusal"
    )
    assert body.get("retryable") is not True


@pytest.mark.asyncio
async def test_context_too_large_refusal_keeps_its_no_timer_contract():
    """The refusal builder still refuses this reason with no timer and evidence.

    Direct-builder contract (the routing path now promotes, but the builder is
    still the single refusal implementation and must keep its AC5 shape for any
    caller that does refuse an oversized prompt): no ``Retry-After`` /
    ``retry_at``, ``retryable: false``, compaction guidance and the token
    evidence a client needs to size the compaction.
    """
    attempt = {
        "reason": "context_too_large",
        "new_tokens": PROMPT_NEW_TOKENS,
        "budget_tokens": COLD_PREFILL_BUDGET_TOKENS,
    }
    result = provider._build_local_only_refusal_response(attempt)

    assert result.status_code == 429
    assert result.headers.get("X-Local-Route") == "refused"
    assert result.headers.get("X-Local-Fallback-Reason") == "context_too_large"

    body = json.loads(result.body)
    assert result.headers.get("Retry-After") is None, (
        "waiting cannot make an oversized prompt fit: advertising a delay "
        "invites futile retries"
    )
    assert body.get("retryable") is False
    assert body.get("retry_after") is None
    assert body.get("retry_at") is None
    assert "compact" in str(body.get("suggested_action") or "").lower(), (
        "the suggested action must name compaction as the remedy"
    )
    # The verdict is the body's classification of the refusal (wiring slice).
    assert body.get("verdict") == "too_large"

    # The token evidence lets the client size the compaction (parent AC4).
    assert (
        int(result.headers["X-Local-Admission-New-Tokens"]) == PROMPT_NEW_TOKENS
    ), "the context refusal must carry the token evidence on its headers"
    assert (
        int(result.headers["X-Local-Admission-Budget-Tokens"])
        == COLD_PREFILL_BUDGET_TOKENS
    )
    assert body.get("new_tokens") == PROMPT_NEW_TOKENS
    assert body.get("budget_tokens") == COLD_PREFILL_BUDGET_TOKENS


@pytest.mark.asyncio
async def test_prefer_local_keeps_todays_remote_fall_through_on_real_expiry():
    """Byte-compatibility: a default client still falls through to remote.

    The real-expiry timer is added to the `local-only` refusal only; a
    ``prefer-local`` client must keep today's remote dispatch unchanged.
    """
    with _local_entry_in_cooldown(COOLDOWN_REMAINDER_SECONDS):
        result, dispatched = await _issue(preference="prefer-local")

    assert dispatched == ["remote"], (
        "prefer-local must keep today's remote fall-through, got "
        f"{dispatched}"
    )
    assert result.status_code == 200
    assert result.headers.get("X-Local-Route") == "remote"
