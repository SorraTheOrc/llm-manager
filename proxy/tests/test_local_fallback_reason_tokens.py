"""Stable reason tokens for every exit from the local route.

LP-0MVRMC1W6003MV3Y (child of LP-0MVNU2KTK009D4AG, parent AC2/AC4/AC6).

The sibling choke point (LP-0MVRMC1L3006F0F3) refuses every ``local-only``
exit with *a* reason; this child pins the vocabulary: each exit records the
token from the canonical table
(:data:`proxy.admission_reserve.LOCAL_FALLBACK_REASON_TOKENS`), and the token
must equal the ``reason=`` log field verbatim so the header, the refusal body
and the log can never drift.

The 2026-10-25 incident was a silent fall-through: the chain-resolution skips
inside ``_resolve_provider_with_exclusions`` (cooldown / quota hard stop /
usage-limit reset / ``available_times`` window) and the lease-active
``continue`` left no reason on the request, so the refusal fell back to the
generic ``local_route_unavailable`` token. These tests pin the per-branch
tokens for exactly those exits:

- chain-resolution skip: ``local_provider_cooldown``,
  ``quota_hard_stop``, ``usage_limit_reset_pending``,
  ``outside_available_window``
- lease path: ``local_lease_active``

Every case asserts three surfaces at once (AC "unit tests assert each token is
recorded for its branch and equals the log ``reason=`` value verbatim"):

- the ``local-only`` refusal carries the token in both the
  ``X-Local-Fallback-Reason`` header and the body ``reason``;
- the router's own ``Skipping provider= ... reason=<token>`` log line matches
  verbatim;
- ``prefer-local`` records the SAME token (observability is
  preference-independent) but never refuses — the remote fall-through still
  completes (parent AC: preferred status only).
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

from proxy import admission_reserve, contention_queue, prefill_eta

LOCAL_PROVIDER_NAME = "local-llama"
REMOTE_PROVIDER_NAME = "remote-fallback"

LOCAL_ONLY = "local-only"
PREFER_LOCAL = "prefer-local"

#: Wait budget every route runs with (seconds).
WAIT_BUDGET_SECONDS = 60


class _DummyRequest:
    """Minimal request stub (matches the matrix test helpers)."""

    def __init__(self, body: bytes = b'{"model":"test"}', headers=None):
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


def _lease_active_response() -> Response:
    """A llama-server 503 naming ``local_lease_active`` (the lease shows busy)."""
    return Response(
        content=json.dumps(
            {
                "error": {"code": "no_slots_available"},
                "total_slots": 1,
                "available_slots": 0,
                "reason": "local_lease_active",
            }
        ),
        status_code=503,
        media_type="application/json",
    )


def _model_config(*, local_entry: dict | None = None) -> dict:
    """The hybrid chain (local first, then remote) every test runs."""
    local = {"name": LOCAL_PROVIDER_NAME, "type": "local", "llama_model": "Qwen3"}
    if local_entry:
        local.update(local_entry)
    return {
        "providers": [
            local,
            {
                "name": REMOTE_PROVIDER_NAME,
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
            },
        ],
        "aliases": ["hybrid*"],
    }


def _config(**server_extra) -> dict:
    """Server config with the admission thresholds disabled."""
    server = {
        "session_slot_pool_size": 1,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": WAIT_BUDGET_SECONDS,
        "contention_queue_max_depth": 4,
        "local_model_ctx_size": 0,
        "local_prefill_deadline_seconds": 0,
    }
    server.update(server_extra)
    return {"provider_cooldown_seconds": 60, "server": server}


# ---------------------------------------------------------------------------
# Exit fixtures
# ---------------------------------------------------------------------------


@contextmanager
def _local_entry_in_cooldown(seconds: float = 60.0):
    """Put the LOCAL entry into a real cooldown."""
    previous = provider._provider_unavailable_until.get(LOCAL_PROVIDER_NAME)
    provider._provider_unavailable_until[LOCAL_PROVIDER_NAME] = time.time() + seconds
    try:
        yield
    finally:
        if previous is None:
            provider._provider_unavailable_until.pop(LOCAL_PROVIDER_NAME, None)
        else:
            provider._provider_unavailable_until[LOCAL_PROVIDER_NAME] = previous


def _quota_hard_stop_for_local(original):
    """Quota hard stop on the LOCAL entry only, so a remote fallback exists."""

    def _check(provider_cfg: dict, config: dict):
        if provider_cfg.get("type") == "local":
            return True, "quota_hard_stop (monthly quota reached)"
        return original(provider_cfg, config)

    return _check


@contextmanager
def _usage_reset_pending_for_local():
    """Quarantine the LOCAL entry's account until a future reset time."""
    key = provider._usage_limit_account_key(
        {"name": LOCAL_PROVIDER_NAME, "type": "local", "llama_model": "Qwen3"}
    )
    previous_at = provider._usage_reset_at.get(key)
    previous_window = provider._usage_reset_window.get(key)
    provider._usage_reset_at[key] = time.time() + 60.0
    provider._usage_reset_window[key] = "weekly"
    try:
        yield
    finally:
        if previous_at is None:
            provider._usage_reset_at.pop(key, None)
        else:
            provider._usage_reset_at[key] = previous_at
        if previous_window is None:
            provider._usage_reset_window.pop(key, None)
        else:
            provider._usage_reset_window[key] = previous_window


def _outside_available_window_entry() -> dict:
    """A LOCAL entry whose ``available_times`` window can never contain now."""
    now = datetime.now(UTC)
    start = (now + timedelta(minutes=30)).time()
    end = (now + timedelta(minutes=31)).time()
    return {"available_times": [f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"]}


#: One row per chain-resolution skip this child pins.
_CHAIN_SKIP_ROWS: tuple[tuple[str, str, dict | None], ...] = (
    ("cooldown", "local_provider_cooldown", None),
    ("quota_hard_stop", "quota_hard_stop", None),
    ("usage_limit_reset_pending", "usage_limit_reset_pending", None),
    ("outside_available_window", "outside_available_window", None),
)


def _enter_exit(trigger: str, stack: ExitStack) -> None:
    """Arm *trigger*'s exit on top of the shared environment."""
    if trigger == "cooldown":
        stack.enter_context(_local_entry_in_cooldown())
    elif trigger == "quota_hard_stop":
        stack.enter_context(
            patch(
                "proxy.provider._quota_exhausted",
                _quota_hard_stop_for_local(provider._quota_exhausted),
            )
        )
    elif trigger == "usage_limit_reset_pending":
        stack.enter_context(_usage_reset_pending_for_local())
    elif trigger == "outside_available_window":
        pass  # armed via the local entry's ``available_times``
    else:  # pragma: no cover - a new trigger must name its exit
        raise AssertionError(f"unknown local-route exit trigger {trigger!r}")


async def _issue(
    trigger: str | None,
    preference: str | None,
    caplog_bind: Exception | None = None,
):
    """Issue one request through ``proxy_with_fallback``.

    *trigger* ``"lease"`` forces a lease-active local response with a busy
    slot pool; ``None`` means "no local exit at all" (the entry is skipped for
    the trigger named on the row, or the lease path for the lease row).

    Returns ``(response, calls)`` where *calls* is the ordered upstream
    dispatch record (``[]`` ⇒ nothing was dispatched).
    """
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _lease_active_response() if trigger == "lease" else _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    headers = {} if preference is None else {"X-Local-Preference": preference}
    local_entry = _outside_available_window_entry() if trigger == "outside_available_window" else None
    with ExitStack() as stack:
        stack.enter_context(patch("proxy.router.proxy_to_local", _local))
        stack.enter_context(patch("proxy.server.proxy_to_remote", _remote))
        stack.enter_context(
            patch(
                "proxy.provider._get_local_concurrency_info",
                _MutableConcurrency(0, 1),
            )
        )
        stack.enter_context(patch("proxy.provider.request_priority", return_value="low"))
        if trigger is not None and trigger != "outside_available_window":
            _enter_exit(trigger, stack)
        result = await provider.proxy_with_fallback(
            _DummyRequest(headers=headers),
            "v1/chat/completions",
            _model_config(local_entry=local_entry),
            _config(),
        )
    return result, calls


@pytest.fixture(autouse=True)
def _clean_state():
    contention_queue.reset()
    prefill_eta.reset()
    provider._provider_unavailable_until.pop(LOCAL_PROVIDER_NAME, None)
    yield
    contention_queue.reset()
    prefill_eta.reset()
    provider._provider_unavailable_until.pop(LOCAL_PROVIDER_NAME, None)


# ---------------------------------------------------------------------------
# AC1: the canonical token table is the single source of truth
# ---------------------------------------------------------------------------


def test_canonical_token_table_covers_the_contract_tokens():
    """Every token the chain/lease exits pin on is in the canonical table."""
    tokens = admission_reserve.LOCAL_FALLBACK_REASON_TOKENS
    for token in (
        "local_provider_cooldown",
        "quota_hard_stop",
        "usage_limit_reset_pending",
        "outside_available_window",
        "local_lease_active",
        "local_only_no_local_route",
    ):
        assert token in tokens, f"{token!r} must be a canonical reason token"

    # The provider-side constants resolve through the table, so the code
    # cannot record a spelling the table does not know.
    assert provider.SKIP_REASON_LOCAL_PROVIDER_COOLDOWN == "local_provider_cooldown"
    assert provider.SKIP_REASON_QUOTA_HARD_STOP == "quota_hard_stop"
    assert provider.SKIP_REASON_USAGE_LIMIT_RESET_PENDING == (
        "usage_limit_reset_pending"
    )
    assert provider.SKIP_REASON_OUTSIDE_AVAILABLE_WINDOW == (
        "outside_available_window"
    )
    assert provider.SKIP_REASON_LOCAL_LEASE_ACTIVE == "local_lease_active"
    assert provider.LOCAL_ONLY_NO_LOCAL_ROUTE == "local_only_no_local_route"


def test_canonical_local_fallback_reason_rejects_unknown_tokens():
    """The table is the vocabulary: an unknown token fails loudly, not silently."""
    with pytest.raises(ValueError):
        admission_reserve.canonical_local_fallback_reason("no_such_exit")
    assert (
        admission_reserve.canonical_local_fallback_reason("quota_hard_stop")
        == "quota_hard_stop"
    )


# ---------------------------------------------------------------------------
# AC2: chain-resolution skips record their token; local-only refuses; the
# log reason= field matches the token verbatim
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "trigger,token,local_entry",
    _CHAIN_SKIP_ROWS,
    ids=[r[0] for r in _CHAIN_SKIP_ROWS],
)
async def test_chain_skip_records_token_on_local_only_refusal(
    trigger: str, token: str, local_entry: dict | None, caplog
):
    """local-only + a chain skip ⇒ 429 naming the token, matched by the log."""
    with caplog.at_level("INFO", logger="llama-proxy.provider"):
        result, calls = await _issue(trigger, LOCAL_ONLY)

    assert calls == [], (
        f"{trigger}: a local-only request must be refused, not dispatched "
        f"(dispatched={calls})"
    )
    assert result.status_code == 429, f"{trigger}: got {result.status_code}"
    assert result.headers.get("X-Local-Route") == "refused"
    assert result.headers.get("X-Local-Fallback-Reason") == token, (
        f"{trigger}: header must carry the stable token {token!r}, got "
        f"{result.headers.get('X-Local-Fallback-Reason')!r}"
    )
    body = json.loads(result.body)
    assert body.get("reason") == token, (
        f"{trigger}: body reason and header must agree, got {body.get('reason')!r}"
    )

    # AC: the recorded token equals the log reason= value verbatim.
    matches = [
        record.getMessage()
        for record in caplog.records
        if "Skipping provider=" in record.getMessage()
        and f"reason={token}" in record.getMessage()
    ]
    assert matches, (
        f"{trigger}: the chain-skip log must carry reason={token} verbatim; "
        f"got:\n{caplog.text}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "trigger,token,local_entry",
    _CHAIN_SKIP_ROWS,
    ids=[r[0] for r in _CHAIN_SKIP_ROWS],
)
async def test_chain_skip_records_token_for_prefer_local_without_refusal(
    trigger: str, token: str, local_entry: dict | None
):
    """prefer-local records the SAME token but never refuses (AC6).

    The remote fall-through completes exactly as before — only the
    ``X-Local-Fallback-Reason`` header now names *why* local was skipped,
    so observability is preference-independent without changing routing.
    """
    result, calls = await _issue(trigger, PREFER_LOCAL)

    assert calls == ["remote"], (
        f"{trigger}: prefer-local must keep today's remote fall-through "
        f"(dispatched={calls})"
    )
    assert result.status_code == 200, f"{trigger}: got {result.status_code}"
    assert result.headers.get("X-Local-Route") == "remote"
    assert result.headers.get("X-Local-Fallback-Reason") == token, (
        f"{trigger}: prefer-local carries the same token as the refusal, got "
        f"{result.headers.get('X-Local-Fallback-Reason')!r}"
    )
    body = json.loads(result.body)
    assert "code" not in body or body.get("code") != provider.EXHAUSTION_CODE_LOCAL_ADMISSION, (
        f"{trigger}: no refusal is allowed behind prefer-local"
    )


# ---------------------------------------------------------------------------
# AC2: the lease-active continue records local_lease_active
# ---------------------------------------------------------------------------


async def _issue_lease(preference: str | None):
    """A local dispatch that answers lease-active on a free slot pool.

    The pool is free, so the local route is probed; llama-server answers that
    the slot is lease-active (reserved for another session), and the lease
    block re-queues briefly (slot still free ⇒ the re-dispatch repeats the
    answer) before the bounded fall through to the next provider.
    """
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _lease_active_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    headers = {} if preference is None else {"X-Local-Preference": preference}
    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch(
            "proxy.provider._get_local_concurrency_info",
            _MutableConcurrency(0, 1),
        ),
        patch("proxy.provider.request_priority", return_value="low"),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(headers=headers),
            "v1/chat/completions",
            _model_config(),
            _config(),
        )
    return result, calls


@pytest.mark.asyncio
async def test_lease_active_records_local_lease_active_on_local_only_refusal(caplog):
    """A lease-active local answer ⇒ refusal with ``local_lease_active``.

    This is the exit that silently fell through in the 2026-10-25 incident:
    the lease-active ``continue`` left no reason, so the refusal (when the
    next candidate was remote) arrived attributed with the generic fallback
    instead of the contention cause.
    """
    with caplog.at_level("INFO", logger="llama-proxy.provider"):
        result, calls = await _issue_lease(LOCAL_ONLY)

    assert calls == ["local", "local"], (
        "the lease route is probed and re-probed after the queue; the refusal "
        f"must then stop the chain (dispatched={calls})"
    )
    assert result.status_code == 429, f"got {result.status_code}"
    assert result.headers.get("X-Local-Route") == "refused"
    assert result.headers.get("X-Local-Fallback-Reason") == "local_lease_active", (
        "the lease exit must name local_lease_active, got "
        f"{result.headers.get('X-Local-Fallback-Reason')!r}"
    )
    body = json.loads(result.body)
    assert body.get("reason") == "local_lease_active"
    assert "reason=local_lease_active" in caplog.text, (
        "the log must carry reason=local_lease_active verbatim:\n" + caplog.text
    )


@pytest.mark.asyncio
async def test_lease_active_records_token_for_prefer_local_without_refusal():
    """prefer-local records local_lease_active but still completes remotely."""
    result, calls = await _issue_lease(PREFER_LOCAL)

    assert calls == ["local", "local", "remote"], (
        "prefer-local keeps the lease fall-through to remote "
        f"(dispatched={calls})"
    )
    assert result.status_code == 200, f"got {result.status_code}"
    assert result.headers.get("X-Local-Route") == "remote"
    assert result.headers.get("X-Local-Fallback-Reason") == "local_lease_active", (
        f"got {result.headers.get('X-Local-Fallback-Reason')!r}"
    )
