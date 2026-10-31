"""Local-pinned refusal matrix (``local-preferred``, legacy ``local-only`` alias):
one row per exit from the local route, plus the cross-cutting behaviour matrix.

LP-0MVRMC0Z500883O0 (child of LP-0MVNU2KTK009D4AG, AC1/AC2/AC3/AC6).

The file keeps its historical name (``test_local_only_refusal_matrix.py``): a
rename would break the citations in ``proxy/docs/routing.md``, in the sibling
work items and in ``test_local_admission_retry_docs.py`` for no behavioural
gain (LP-0MVSHXTX1007D58Z planning decision 2). ``local-only`` is the deployed
legacy alias of the canonical ``local-preferred`` value, and every row below is
asserted for both spellings.

The contract under test is one sentence: with ``X-Local-Preference: local-only``
a request either completes **locally** or receives the local-only refusal — it
is never handed to a remote provider. Every distinct way out of the local route
is one row of :data:`_EXIT_ROWS`, and every row asserts the same two outcomes:

- ``local-only`` ⇒ HTTP 429, ``X-Local-Route: refused``,
  ``X-Local-Fallback-Reason: <token>``, and **no upstream call at all** (a
  refused request is neither dispatched locally nor moved remote);
- ``prefer-local`` and an absent header ⇒ today's remote dispatch, with no
  refusal and no refusal body (AC3 byte-compatibility).

The structural counter-case closes the loop: no preference strips local entries
from the chain (LP-0MVUH9OYZ007GMWF AC1 — ``remote-ok`` means "local allowed,
remote acceptable"), so a remote-less chain under ``remote-ok`` is served
locally rather than answered with the exhaustion response.

Rows are pinned, not the prose: nothing here asserts the human-readable
``error``/``detail`` text — the stable ``code``/header contract is what clients
discriminate on (LP-0MVNTJJGX00375VQ owns the wording), and this module reuses
the refusal builders rather than re-describing their payloads.

Why a matrix instead of one test per branch: the 2026-10-25 incident happened
because ``local-only`` was enforced at *two* admission decisions while every
other ``continue`` leaked to remote. A table with one row per exit makes a new
skip path (or a dropped enforcement point) a visible, failing row instead of an
untested branch.

Test-first: the rows that a skip path still leaks are marked
``xfail(strict=True)`` — ``strict`` so that the implementing children
(LP-0MVRMC1L3006F0F3 choke point, LP-0MVRMC1W6003MV3Y reason tokens) cannot land
without flipping the marker off with the behaviour. The choke point flipped
all but two rows green; the reason-token child's per-branch tokens
(LP-0MVRMC1W6003MV3Y) turned the remaining two (``quota_hard_stop``,
``chain_cooldown``) green, so no row is pending today.

Capacity verdict (LP-0MVSMRT5R009EKP9, child of LP-0MVSMI7JK002FKUF): every row
also states *what the refusal means* — ``too_large`` for the request-size exits
(``context_too_large``) and ``deferrable`` for every temporal exit, including
the fail-safe ``local_only_no_local_route`` default. The verdict assertions were
the TDD red phase for the wiring child (LP-0MVSMINPR006SWT6): they started
``xfail(strict=True)`` because the 429 body had no ``verdict`` field, and that
child flipped them green in the same change that added it — a red row now means
the verdict contract broke.

Pin slice (LP-0MVSLZY470051ZB2): the verdict now decides *whether* the pin holds.
A ``deferrable`` exit is refused (``calls == []``, 429, ``X-Local-Route:
refused``); a ``too_large`` exit declines the pin and is served by the remote
tiers (``calls == ["remote"]``, 200, no local-capacity 429) — the boundary that
removed the incident's "All providers exhausted" dead end. Both wire spellings
(the ``local-only`` alias and ``local-preferred``) are asserted per row, and the
two post-dispatch rows prove the pin also holds *after* a local dispatch failed.

Every row now also asserts the refusal body's verdict, including the free-slot
direct path (``admission_guard_free_slot``): LP-0MVNTJJGX00375VQ routed that
branch through the one pin-local refusal builder, so the admission-reject /
exhaustion payload no longer answers a pinned request and the row's former
``_VERDICT_NOT_APPLICABLE`` skip is gone (LP-0MVSS435600370D6's deliverable).
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import proxy.provider as provider
import pytest
from fastapi import Response
from proxy.priority import apply_local_preference

from proxy import admission_reserve, contention_queue, prefill_eta

#: Prompt size priced by the rows that need a known measurement.
PROMPT_TOKENS = 20_000
#: Per-entry ``max_context_tokens`` cap for the entry-ceiling row: half the
#: priced prompt, so the cap (not a global threshold) is what binds.
ENTRY_CONTEXT_CAP_TOKENS = PROMPT_TOKENS // 2
#: Calibrated budget refusal: a cache-cold 100k prompt at ~3.9 ms/token is
#: ~390 s of prefill against a 60 s wait budget.
ADMISSION_PROMPT_TOKENS = 100_000
#: Wait budget every row runs with (seconds).
WAIT_BUDGET_SECONDS = 60
#: Queue wait used by the expiry row — small enough to keep the suite fast.
QUEUE_EXPIRY_WAIT_SECONDS = 0.05
#: Local entry used by every row.
LOCAL_PROVIDER_NAME = "local-llama"
REMOTE_PROVIDER_NAME = "remote-fallback"

LOCAL_ONLY = "local-only"
LOCAL_PREFERRED = "local-preferred"
PREFER_LOCAL = "prefer-local"
REMOTE_OK = "remote-ok"


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


async def _no_compaction(_server, _result, _body_json, _server_config) -> dict:
    """Server-side compaction is a no-op: these rows pin *routing*, not rescue."""
    return {
        "evaluated": False,
        "action": "noop",
        "applied": False,
        "estimated_before": 0,
        "estimated_after": 0,
        "reason": None,
        "messages": None,
    }


@dataclass(frozen=True)
class _ExitRow:
    """One exit from the local route.

    ``trigger`` names the mechanism that takes the exit (the fixture), ``token``
    is the stable ``X-Local-Fallback-Reason`` the refusal must carry,
    ``expected_verdict`` is the capacity verdict the refusal body must report
    (:data:`proxy.admission_reserve.CAPACITY_DEFERRABLE` or
    :data:`proxy.admission_reserve.CAPACITY_TOO_LARGE`), and
    ``exit_description`` is the operator-facing name of the exit — used in
    failure messages so a red row says which route leaked.
    """

    trigger: str
    token: str
    expected_verdict: str
    exit_description: str
    #: ``(active, max)`` local slots the row runs with — busy slots are what
    #: bring the contention queue into play.
    concurrency: tuple[int, int] = (0, 1)
    #: Reason forced through ``_should_skip_local`` for the router-skip rows.
    router_skip_reason: str | None = None
    #: Extra ``server`` config for the row.
    server: dict = field(default_factory=dict)
    #: Extra keys merged onto the local provider entry (e.g. the per-entry
    #: ``max_context_tokens`` ceiling).
    local_entry: dict = field(default_factory=dict)
    #: Drop the local entry from the chain entirely (no local route at all).
    remote_only_chain: bool = False
    #: A callable building the response the patched local stub returns. Used
    #: by the post-dispatch rows, which prove the pin holds even after the
    #: local route was genuinely attempted and then failed. ``None`` means
    #: "serve like a healthy local" (the shared stub in :func:`_issue`).
    local_response: Callable[[], Response] | None = None


_EXIT_ROWS: tuple[_ExitRow, ...] = (
    _ExitRow(
        trigger="thrash_cooldown",
        token="prefill_thrash_cooldown",
        expected_verdict=admission_reserve.CAPACITY_DEFERRABLE,
        exit_description=(
            "session in post-deadline prefill-thrash cooldown "
            "(router_skip_local reason=prefill_thrash_cooldown)"
        ),
    ),
    _ExitRow(
        trigger="router_skip_physical_cap",
        token="context_too_large",
        expected_verdict=admission_reserve.CAPACITY_TOO_LARGE,
        exit_description="router skip: prompt cannot fit the KV slot",
        router_skip_reason="context_too_large",
    ),
    _ExitRow(
        trigger="router_skip_economic",
        token="large_context_bypass",
        expected_verdict=admission_reserve.CAPACITY_DEFERRABLE,
        exit_description="router skip: economic cold-prefill bypass",
        router_skip_reason="large_context_bypass",
    ),
    _ExitRow(
        trigger="router_skip_residency",
        token="resident_prefix_insufficient",
        expected_verdict=admission_reserve.CAPACITY_DEFERRABLE,
        exit_description=(
            "router skip: residency ladder found too little reusable prefix"
        ),
        router_skip_reason="resident_prefix_insufficient",
    ),
    _ExitRow(
        trigger="router_skip_tenancy",
        token="tenant_limit",
        expected_verdict=admission_reserve.CAPACITY_DEFERRABLE,
        exit_description="router skip: per-tenant residency cap reached",
        router_skip_reason="tenant_limit",
    ),
    _ExitRow(
        trigger="queue_depth_cap",
        token="fallback_after_queue",
        expected_verdict=admission_reserve.CAPACITY_DEFERRABLE,
        exit_description="contention-queue depth cap reached (never queued)",
        concurrency=(1, 1),
        server={"contention_queue_max_depth": 1},
    ),
    _ExitRow(
        trigger="queue_wait_expiry",
        token="fallback_after_queue",
        expected_verdict=admission_reserve.CAPACITY_DEFERRABLE,
        exit_description="contention-queue wait budget expired",
        concurrency=(1, 1),
        server={
            "contention_queue_max_wait_seconds": QUEUE_EXPIRY_WAIT_SECONDS,
            "contention_queue_max_depth": 4,
        },
    ),
    _ExitRow(
        trigger="admission_guard_free_slot",
        token="eta_exceeds_budget_direct",
        expected_verdict=admission_reserve.CAPACITY_DEFERRABLE,
        exit_description=(
            "prefill-gate denial on the free-slot (direct dispatch) path"
        ),
    ),
    _ExitRow(
        trigger="entry_context_cap",
        token="context_too_large",
        expected_verdict=admission_reserve.CAPACITY_TOO_LARGE,
        exit_description="per-entry max_context_tokens ceiling",
        # The router gate is inert while the cold threshold is 0 ("large-context
        # bypass disabled"), so the row enables the gate and lets the entry cap
        # be the bound that binds: cold 1k < prompt 20k, entry cap 10k.
        server={"local_large_context_cold_cache_threshold": 1_000},
        local_entry={"max_context_tokens": ENTRY_CONTEXT_CAP_TOKENS},
    ),
    _ExitRow(
        trigger="quota_hard_stop",
        token="quota_hard_stop",
        expected_verdict=admission_reserve.CAPACITY_DEFERRABLE,
        exit_description="chain resolution skipped local on a quota hard stop",
    ),
    _ExitRow(
        trigger="chain_cooldown",
        token="local_provider_cooldown",
        expected_verdict=admission_reserve.CAPACITY_DEFERRABLE,
        exit_description="chain resolution skipped local while it cools down",
    ),
    _ExitRow(
        trigger="no_local_route",
        token="local_only_no_local_route",
        expected_verdict=admission_reserve.CAPACITY_DEFERRABLE,
        exit_description="the chain has no local entry at all",
        remote_only_chain=True,
    ),
    # Post-dispatch exits (LP-0MVSLZY470051ZB2 AC1): the local route is
    # genuinely attempted once and then fails, so the refusal must come from
    # the loop-head choke point on the next iteration — the structural proof
    # that the pin holds *after* the local dispatch, not just before it.
    _ExitRow(
        trigger="local_error_status",
        token=provider.LOCAL_ROUTE_UNAVAILABLE,
        expected_verdict=admission_reserve.CAPACITY_DEFERRABLE,
        exit_description=(
            "post-dispatch: the local provider returned an upstream 500 error"
        ),
        local_response=lambda: Response(
            content=b'{"error":"internal"}',
            status_code=500,
            media_type="application/json",
        ),
    ),
    _ExitRow(
        trigger="local_empty_response",
        token=provider.LOCAL_ROUTE_UNAVAILABLE,
        expected_verdict=admission_reserve.CAPACITY_DEFERRABLE,
        exit_description=(
            "post-dispatch: the local provider returned an empty response"
        ),
        local_response=lambda: Response(content=b"", status_code=200),
    ),
)

#: Every row is asserted green: the choke point (LP-0MVRMC1L3006F0F3)
#: refuses every deferrable exit, the reason-token child
#: (LP-0MVRMC1W6003MV3Y) records the per-branch chain tokens, and the pin
#: slice (LP-0MVSLZY470051ZB2) declines the pin on the ``too_large`` rows.
#: A future red row is a real failure — the xfail marker machinery died with
#: the behaviour it pinned.


def _model_config(row: _ExitRow) -> dict:
    """The model chain for *row* (local + remote, or remote only)."""
    providers = [
        {"name": REMOTE_PROVIDER_NAME, "type": "remote",
         "endpoint": "https://api.example.com/v1"},
    ]
    if not row.remote_only_chain:
        local_entry = {
            "name": LOCAL_PROVIDER_NAME,
            "type": "local",
            "llama_model": "Qwen3",
        }
        local_entry.update(row.local_entry)
        providers.insert(0, local_entry)
    return {"providers": providers, "aliases": ["hybrid*"]}


def _config(row: _ExitRow) -> dict:
    """Server config for *row*.

    Thresholds are deliberately disabled (``local_model_ctx_size: 0`` and no
    economic ceiling) so each row is skipped for the reason it names and not
    for an ambient context threshold.
    """
    server = {
        "session_slot_pool_size": 1,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": WAIT_BUDGET_SECONDS,
        "contention_queue_max_depth": 4,
        "local_model_ctx_size": 0,
        # Isolate the budget bound from the deadline bound.
        "local_prefill_deadline_seconds": 0,
    }
    server.update(row.server)
    return {"provider_cooldown_seconds": 60, "server": server}


@contextmanager
def _estimate_patches(tokens: int, cached_ratio: float = 0.0):
    """Price every prompt exactly *tokens*, cache-cold by default."""
    with (
        patch(
            "proxy.provider._estimate_effective_prompt_tokens_for_routing",
            new=AsyncMock(return_value=tokens),
        ),
        patch("proxy.provider._get_tokenizer_for_model", lambda *_: (None, 1.0)),
        patch("proxy.provider._get_cached_ratio", lambda *_: cached_ratio),
    ):
        yield


def _seed_calibration(tokens: int, ms_per_token: float = 3.9) -> None:
    for _ in range(6):
        prefill_eta.record_dispatch_sample("Qwen3", tokens, 0.0, tokens * ms_per_token)


@contextmanager
def _local_entry_in_cooldown(seconds: float = 60.0):
    """Put the local entry into a *real* cooldown."""
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
    """Quota hard stop on the *local* entry only, so a remote fallback exists.

    Chain resolution walks every provider, so a blanket patch would exhaust the
    remote entry as well and the row would test the exhaustion path instead of
    the local skip. *original* is the real predicate, captured before the patch.
    """

    def _check(provider_cfg: dict, config: dict):
        if provider_cfg.get("type") == "local":
            return True, "quota_hard_stop (monthly quota reached)"
        return original(provider_cfg, config)

    return _check


def _occupy_the_only_queue_slot() -> None:
    """Take the queue's single depth slot, as a concurrently waiting request would.

    The contention-queue module documents its waiter tuple as
    ``(enqueued_at, event, session_id, retained, eta_seconds)``; seeding one
    waiter reaches the depth cap deterministically (the real code returns
    ``None`` *before* enqueueing) instead of racing two live tasks inside a
    table-driven matrix. ``contention_queue.reset()`` in the autouse fixture
    clears it again.
    """
    contention_queue._waiters.append((0.0, None, "queue-holder", False, None))


def _row_patches(row: _ExitRow) -> list:
    """The row-specific patches that take *row*'s exit (on top of the shared ones)."""
    trigger = row.trigger
    patches: list = []
    if row.router_skip_reason is not None:
        patches.append(
            patch(
                "proxy.provider._should_skip_local",
                lambda *a, **k: (True, row.router_skip_reason),
            )
        )
    elif trigger == "thrash_cooldown":
        patches.append(
            patch(
                "proxy.router_helpers._prefill_thrash_cooldown_remaining",
                lambda *a, **k: 90.0,
            )
        )
    elif trigger == "queue_depth_cap":
        _occupy_the_only_queue_slot()
    elif trigger == "queue_wait_expiry":
        pass
    elif trigger == "admission_guard_free_slot":
        _seed_calibration(ADMISSION_PROMPT_TOKENS)
        patches.append(_estimate_patches(ADMISSION_PROMPT_TOKENS))
    elif trigger == "entry_context_cap":
        patches.append(_estimate_patches(PROMPT_TOKENS))
    elif trigger == "quota_hard_stop":
        patches.append(
            patch(
                "proxy.provider._quota_exhausted",
                _quota_hard_stop_for_local(provider._quota_exhausted),
            )
        )
    elif trigger == "chain_cooldown":
        patches.append(_local_entry_in_cooldown())
    elif trigger == "no_local_route":
        pass
    elif trigger in ("local_error_status", "local_empty_response"):
        # Post-dispatch failures: the shared patches already leave a free slot
        # so the loop discovers the failure at the dispatch itself.
        pass
    elif trigger == HEALTHY_FREE_SLOT:
        # The healthy baseline: no exit to take, no row-specific patch. Named
        # here so a new trigger cannot silently reuse it.
        pass
    else:  # pragma: no cover - a new row must name its trigger
        raise AssertionError(f"unknown local-route exit trigger {trigger!r}")
    return patches


async def _issue(row: _ExitRow, preference: str | None) -> tuple:
    """Issue *row*'s request with *preference*; return ``(response, calls)``.

    *calls* is the ordered record of upstream dispatches: exactly one entry per
    attempt, so ``[]`` proves nothing was dispatched and ``["remote"]`` proves
    today's remote fall-through still happens.
    """
    result, calls, _ = await _issue_with_request(row, preference)
    return result, calls


async def _issue_with_request(
    row: _ExitRow, preference: str | None
) -> tuple:
    """Like :func:`_issue`, but also return the request the cycle saw.

    The request carries the per-request routing state this suite asserts on
    (the cached capacity verdict and the stamped outcome), which a response
    cannot expose.
    """
    calls: list[str] = []
    headers = {} if preference is None else {"X-Local-Preference": preference}
    request = _DummyRequest(headers=headers)

    async def _local(_req, _path):
        calls.append("local")
        response_factory = row.local_response or _ok_response
        return response_factory()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    with ExitStack() as stack:
        stack.enter_context(patch("proxy.router.proxy_to_local", _local))
        stack.enter_context(patch("proxy.server.proxy_to_remote", _remote))
        stack.enter_context(
            patch(
                "proxy.provider._get_local_concurrency_info",
                _MutableConcurrency(*row.concurrency),
            )
        )
        stack.enter_context(
            patch("proxy.provider.request_priority", return_value="low")
        )
        stack.enter_context(
            patch(
                "proxy.router_helpers.evaluate_and_apply_compaction",
                _no_compaction,
            )
        )
        for ctx in _row_patches(row):
            stack.enter_context(ctx)
        result = await provider.proxy_with_fallback(
            request,
            "v1/chat/completions",
            _model_config(row),
            _config(row),
        )
    return result, calls, request


@pytest.fixture(autouse=True)
def _clean_state():
    contention_queue.reset()
    prefill_eta.reset()
    provider._provider_unavailable_until.pop(LOCAL_PROVIDER_NAME, None)
    yield
    contention_queue.reset()
    prefill_eta.reset()
    provider._provider_unavailable_until.pop(LOCAL_PROVIDER_NAME, None)


def test_matrix_has_exactly_one_row_per_local_route_exit():
    """The table is the contract: no duplicate exits, none silently dropped.

    A new exit from the local route must arrive with a row, and a row that
    disappears would silently lose its ``local-only`` guarantee — so both are
    pinned here, in one place, rather than restated per test.
    """
    triggers = [row.trigger for row in _EXIT_ROWS]
    assert len(triggers) == len(set(triggers)), f"duplicate rows: {triggers}"
    assert set(triggers) == {
        "thrash_cooldown",
        "router_skip_physical_cap",
        "router_skip_economic",
        "router_skip_residency",
        "router_skip_tenancy",
        "queue_depth_cap",
        "queue_wait_expiry",
        "admission_guard_free_slot",
        "entry_context_cap",
        "quota_hard_stop",
        "chain_cooldown",
        "no_local_route",
        "local_error_status",
        "local_empty_response",
    }, "a local-route exit is missing from (or unexpected in) the matrix"

    for row in _EXIT_ROWS:
        assert row.token, f"{row.trigger}: every exit needs a stable reason token"
        assert row.exit_description, f"{row.trigger}: describe the exit it pins"
        # Vocabulary guard: the wire contract documents exactly two verdicts,
        # so a row naming a third value must fail here rather than surface as
        # an unexplained body mismatch.
        assert row.expected_verdict in {
            admission_reserve.CAPACITY_DEFERRABLE,
            admission_reserve.CAPACITY_TOO_LARGE,
        }, (
            f"{row.trigger}: expected_verdict must be one of the documented "
            f"capacity verdicts, got {row.expected_verdict!r}"
        )


# ---------------------------------------------------------------------------
# AC1 + AC2: a pin-local client is NEVER moved to a paid provider
# ---------------------------------------------------------------------------

#: Both wire spellings of the pin-local intent. ``local-only`` is accepted as a
#: deprecated alias and normalised to ``local-preferred`` by
#: ``admission_reserve.parse_preference()``, so every row must behave
#: identically for both — asserted, not assumed (parent AC5).
PIN_LOCAL_PREFERENCES = (LOCAL_ONLY, LOCAL_PREFERRED)

#: Rows whose exit is temporal (waiting can help): the pin holds and the
#: request is refused instead of being promoted.
_DEFERRABLE_ROWS = tuple(
    row
    for row in _EXIT_ROWS
    if row.expected_verdict == admission_reserve.CAPACITY_DEFERRABLE
)
#: Rows whose exit is a request-size/context condition (waiting cannot help):
#: the pin is declined and the remote tiers serve the request.
_TOO_LARGE_ROWS = tuple(
    row for row in _EXIT_ROWS if row.expected_verdict == admission_reserve.CAPACITY_TOO_LARGE
)


def _row(trigger: str) -> _ExitRow:
    """Return the matrix row named *trigger* (single-selection helper)."""
    for row in _EXIT_ROWS:
        if row.trigger == trigger:
            return row
    raise AssertionError(f"no matrix row named {trigger!r}")  # pragma: no cover


@pytest.mark.asyncio
@pytest.mark.parametrize("row", _DEFERRABLE_ROWS, ids=[r.trigger for r in _DEFERRABLE_ROWS])
@pytest.mark.parametrize("preference", PIN_LOCAL_PREFERENCES)
async def test_pin_local_refuses_instead_of_dispatching_remote(
    row: _ExitRow, preference: str
):
    """Every deferrable exit: 429 refusal naming the reason, nothing dispatched.

    The assertion that carries the incident is ``calls == []``: a pin-local
    client with a **deferrable** verdict must be neither served locally by a
    remote provider's turn nor handed to one, whatever reason the local route
    was unavailable (parent assumption A2 — a fits-local request is never
    promoted to a paid provider).

    The queue/lease invariants of AC6 ride along here: the refusal is terminal,
    so no waiter survives it and ``X-Local-Route`` is stamped exactly once.
    """
    depth_before = contention_queue.queue_depth()
    result, calls = await _issue(row, preference)

    expected = ["local"] if row.local_response is not None else []
    assert calls == expected, (
        f"{row.exit_description}: a pin-local ({preference}) request must never "
        f"reach a remote provider (dispatched={calls}); at most the single "
        f"local attempt of a post-dispatch exit is expected {expected}"
    )
    assert result.status_code == 429, (
        f"{row.exit_description}: a deferrable pin-local request means the "
        f"refusal, got {result.status_code}"
    )
    assert result.headers.get("X-Local-Route") == "refused", (
        f"{row.exit_description}: the response must declare the refusal"
    )
    assert result.headers.get("X-Local-Fallback-Reason") == row.token, (
        f"{row.exit_description}: the refusal must carry the stable reason token "
        f"{row.token!r}, got "
        f"{result.headers.get('X-Local-Fallback-Reason')!r}"
    )
    assert result.headers.getlist("x-local-route") == ["refused"], (
        f"{row.exit_description}: X-Local-Route must be stamped exactly once, "
        f"got {result.headers.getlist('x-local-route')!r}"
    )
    # The refusal is the stable code/header contract, not prose: an
    # unparseable body or a 5xx would break clients that classify on it.
    body = json.loads(result.body)
    assert body.get("code"), f"{row.exit_description}: refusal needs a stable code"
    assert body.get("reason") == row.token, (
        f"{row.exit_description}: body reason and header must agree, got "
        f"{body.get('reason')!r}"
    )
    # AC6: the refusal neither enqueues a new waiter nor leaks the queue —
    # the depth is unchanged from before the request (the ``queue_depth_cap``
    # row pre-seeds its single holder and must be excluded from the count).
    seeded = 1 if row.trigger == "queue_depth_cap" else 0
    assert contention_queue.queue_depth() == depth_before + seeded, (
        f"{row.exit_description}: the refusal must leave the queue exactly as "
        f"it found it (depth {depth_before} + {seeded} seeded -> "
        f"{contention_queue.queue_depth()})"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("row", _TOO_LARGE_ROWS, ids=[r.trigger for r in _TOO_LARGE_ROWS])
@pytest.mark.parametrize("preference", PIN_LOCAL_PREFERENCES)
async def test_too_large_exits_promote_instead_of_refusing(
    row: _ExitRow, preference: str
):
    """Every request-size exit: the pin is declined, remote serves, no 429.

    Parent AC4 / pin slice AC4: a prompt that outgrows the local serving
    envelope cannot be helped by waiting, so it must reach the remote tiers
    and must **not** receive a local-capacity 429 (the "All providers
    exhausted" dead end the incident reported). Chain-level stripping of the
    local entry and the promoted outcome line belong to LP-0MVSM06LB001PRU4;
    this row pins the refusal boundary only.
    """
    result, calls = await _issue(row, preference)

    assert calls == ["remote"], (
        f"{row.exit_description}: a too-large pin-local request must fall "
        f"through to the remote tiers (dispatched={calls})"
    )
    assert result.status_code == 200, (
        f"{row.exit_description}: a too-large request must not be refused for "
        f"local capacity, got {result.status_code}"
    )
    assert result.headers.get("X-Local-Route") == "remote", (
        f"{row.exit_description}: the remote route must be reported, got "
        f"{result.headers.get('X-Local-Route')!r}"
    )
    body = json.loads(result.body)
    assert body.get("code") != provider.EXHAUSTION_CODE_LOCAL_ADMISSION, (
        f"{row.exit_description}: no local-capacity refusal may be produced"
    )
    assert b"All providers exhausted" not in result.body, (
        f"{row.exit_description}: promotion must not report chain exhaustion"
    )
    assert contention_queue.queue_depth() == 0, (
        f"{row.exit_description}: a promoted request never enters the queue"
    )




# ---------------------------------------------------------------------------
# Too-large promotion: local is stripped BEFORE the dispatch loop
# (LP-0MVSM06LB001PRU4 — AC1/AC2/AC3/AC4/AC6)
# ---------------------------------------------------------------------------


class _VerdictRecording:
    """Recording passthrough for the single capacity-verdict helper."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.results: list = []
        self._original = admission_reserve.local_capacity_verdict

    def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": dict(kwargs)})
        verdict = self._original(*args, **kwargs)
        self.results.append(verdict)
        return verdict


@pytest.mark.asyncio
@pytest.mark.parametrize("preference", PIN_LOCAL_PREFERENCES)
async def test_too_large_promotion_never_touches_the_local_route(preference: str):
    """AC1/AC6: free local slot, yet the local dispatch stub is never invoked.

    The row's ``(0, 1)`` slot state means a local slot is genuinely free, so
    ``calls == ["remote"]`` cannot be explained by contention: the local entry
    was removed from the chain before the loop ever resolved a provider. One
    dispatch, one ``X-Local-Route`` header, nothing queued — the promoted path
    takes no queue slot and no lease (a lease is only ever taken by a local
    dispatch, which never happens).
    """
    row = _row("entry_context_cap")
    assert row.concurrency == (0, 1), "this row must run with a free local slot"

    result, calls, request = await _issue_with_request(row, preference)

    assert calls == ["remote"], (
        f"{row.exit_description}: the local entry must be stripped pre-loop, "
        f"not skipped in-loop (dispatched={calls})"
    )
    assert len(calls) == 1, "the promoted request is dispatched exactly once"
    assert result.status_code == 200, f"got {result.status_code}"
    assert result.headers.getlist("x-local-route") == ["remote"], (
        "exactly one X-Local-Route header is stamped on the promoted path"
    )
    assert contention_queue.queue_depth() == 0, (
        "a promoted request engages no contention queue"
    )
    assert request._local_routing_outcome == provider.LOCAL_ROUTING_OUTCOME_PROMOTED, (
        "the promotion outcome is stamped for the per-request outcome line"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("preference", PIN_LOCAL_PREFERENCES)
async def test_too_large_verdict_is_computed_once_and_cached(preference: str):
    """AC2: one verdict per request before the loop, reused from the request.

    The pre-loop computation prices the routing estimate against the local
    entry's ceiling; the pin predicate then reuses the cached value (it never
    re-prices the prompt), so the helper is consulted exactly once for a
    promoted request.
    """
    row = _row("entry_context_cap")
    spy = _VerdictRecording()

    with patch("proxy.admission_reserve.local_capacity_verdict", new=spy):
        result, calls, request = await _issue_with_request(row, preference)

    assert calls == ["remote"]
    assert result.status_code == 200
    assert len(spy.calls) == 1, (
        "the verdict is computed once per request per cycle; "
        f"got {spy.calls!r}"
    )
    cached = request._local_capacity_verdict
    assert cached is not None, "the verdict must be cached on the request"
    assert cached.verdict == admission_reserve.CAPACITY_TOO_LARGE
    priced = spy.calls[0]["kwargs"]
    assert priced["estimated_tokens"] == PROMPT_TOKENS, (
        "the verdict must price the shared routing estimate"
    )
    assert priced["context_limit"] == ENTRY_CONTEXT_CAP_TOKENS, (
        "the verdict must use the local entry's own context ceiling"
    )


@pytest.mark.asyncio
async def test_too_large_strip_reuses_the_single_chain_filter():
    """AC4: the strip goes through ``priority.apply_local_preference``.

    ``provider.py`` owns no second ``type == "local"`` filtering loop: the
    verdict is handed to the one chain filter, which is the same call
    ``remote-ok`` makes. A hand-rolled comprehension would leave this spy
    empty while the request still promoted.
    """
    row = _row("entry_context_cap")
    recorded: list[dict] = []
    original = provider.apply_local_preference

    def _spy(chain, preference, verdict=None):
        recorded.append({"preference": preference, "verdict": verdict})
        return original(chain, preference, verdict)

    with patch.object(provider, "apply_local_preference", _spy):
        result, calls, _ = await _issue_with_request(row, LOCAL_PREFERRED)

    assert calls == ["remote"]
    assert result.status_code == 200
    assert len(recorded) == 1, (
        f"exactly one verdict-aware filter call expected; got {recorded!r}"
    )
    assert recorded[0]["verdict"] == admission_reserve.CAPACITY_TOO_LARGE
    assert recorded[0]["preference"] == admission_reserve.PREF_LOCAL_PREFERRED


@pytest.mark.asyncio
async def test_deferrable_verdict_never_applies_the_capacity_strip():
    """AC4: a temporal verdict leaves the chain alone (never promoted).

    The ``chain_cooldown`` row is plainly deferrable, so the filter must not be
    called with a verdict at all — a fits-local request stays pinned and is
    refused, exactly as the deferrable matrix rows assert.
    """
    row = _row("chain_cooldown")
    recorded: list[dict] = []
    original = provider.apply_local_preference

    def _spy(chain, preference, verdict=None):
        recorded.append({"preference": preference, "verdict": verdict})
        return original(chain, preference, verdict)

    with patch.object(provider, "apply_local_preference", _spy):
        result, calls, request = await _issue_with_request(row, LOCAL_PREFERRED)

    assert calls == [], "a deferrable pin-local request is refused, not promoted"
    assert result.status_code == 429
    assert recorded == [], f"no capacity strip for a deferrable verdict: {recorded!r}"
    assert (
        getattr(request, "_local_routing_outcome", None)
        != provider.LOCAL_ROUTING_OUTCOME_PROMOTED
    ), (
        "a refused request is never stamped as promoted (it is stamped "
        "``pinned`` by the per-request outcome line, LP-0MVSM06Z8003UQTM)"
    )


@pytest.mark.asyncio
async def test_remote_less_chain_is_never_stripped_by_a_too_large_verdict():
    """AC3 (parent A5): a ``private`` chain keeps its local route.

    Every entry is ``type: local``, so capacity stripping would leave nothing
    to serve the request. The filter's carve-out keeps the chain intact: the
    verdict is still ``too_large`` (asserted through the recording helper) but
    it is not cached and the local route is dispatched as before — no empty
    chain and therefore no "All providers exhausted" produced by the filter.
    """
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):  # pragma: no cover - must never run
        calls.append("remote")
        return _ok_response()

    request = _DummyRequest(headers={"X-Local-Preference": LOCAL_PREFERRED})
    model_config = {
        "providers": [
            {"name": LOCAL_PROVIDER_NAME, "type": "local", "llama_model": "Qwen3"}
        ],
        "aliases": ["private*"],
    }
    config = {
        "provider_cooldown_seconds": 60,
        "server": {
            "session_slot_pool_size": 1,
            "local_model_ctx_size": 0,
            "local_prefill_deadline_seconds": 0,
            # A tight cold-prefill budget makes the *verdict* too_large, while
            # the admission guard is off so this row proves the filter's
            # carve-out rather than a guard refusal.
            "local_cold_prefill_budget_tokens": 1_000,
            "local_admission_guard_policy": "off",
        },
    }
    spy = _VerdictRecording()

    with (
        patch("proxy.admission_reserve.local_capacity_verdict", new=spy),
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch(
            "proxy.provider._get_local_concurrency_info",
            _MutableConcurrency(0, 1),
        ),
        patch("proxy.provider.request_priority", return_value="low"),
        patch(
            "proxy.router_helpers.evaluate_and_apply_compaction",
            _no_compaction,
        ),
        _estimate_patches(PROMPT_TOKENS),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", model_config, config
        )

    assert calls == ["local"], (
        "a remote-less chain keeps its local route even when too_large "
        f"(dispatched={calls})"
    )
    assert result.status_code == 200, f"got {result.status_code}"
    assert result.headers.get("X-Local-Route") == "local"
    assert spy.calls, "the verdict must still be computed before the loop"
    assert any(
        v.verdict == admission_reserve.CAPACITY_TOO_LARGE for v in spy.results
    ), (
        "the row must actually exercise a too_large verdict"
    )
    assert getattr(request, "_local_capacity_verdict", None) is None, (
        "an unstripped chain must not cache a verdict that would flip the pin"
    )
    assert getattr(request, "_local_routing_outcome", None) is None, (
        "a served-local request is not stamped as promoted"
    )


# ---------------------------------------------------------------------------
# AC3: one pin decision, one verdict source
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["chain_cooldown", "admission_guard_free_slot"])
async def test_pin_predicate_is_the_only_pin_decision(trigger: str):
    """Declining the predicate is enough to turn the refusal into a fall-through.

    Both routes that can refuse a deferrable pin-local request are exercised —
    the loop-head choke point (``chain_cooldown``) and the admission refusal
    (``admission_guard_free_slot``) — so neither may carry its own preference
    comparison: patching the one predicate changes both outcomes (parent AC3).
    """
    row = _row(trigger)
    with patch.object(provider, "_local_preference_pins_local", lambda *a, **k: False):
        result, calls = await _issue(row, LOCAL_PREFERRED)

    assert calls == ["remote"], (
        f"{row.exit_description}: the pin predicate alone must decide the "
        f"refusal (dispatched={calls})"
    )
    assert result.status_code == 200, (
        f"{row.exit_description}: a declined pin must fall through, got "
        f"{result.status_code}"
    )


@pytest.mark.asyncio
async def test_pin_verdict_comes_only_from_the_shared_capacity_helper():
    """A spy verdict flips the routing decision — provider.py owns no table.

    Forcing ``too_large`` out of
    :func:`admission_reserve.local_capacity_verdict` promotes a row whose
    reason is plainly temporal. Only a single verdict source can make that
    happen, so any re-derived classification inside ``provider.py`` would fail
    here (parent AC3).
    """
    row = _row("chain_cooldown")
    forced = admission_reserve.CapacityVerdict(
        verdict=admission_reserve.CAPACITY_TOO_LARGE, reason=row.token
    )
    with patch(
        "proxy.admission_reserve.local_capacity_verdict", return_value=forced
    ) as spy:
        result, calls = await _issue(row, LOCAL_PREFERRED)

    assert spy.called, (
        "the pin decision must consult local_capacity_verdict(); re-deriving "
        "the classification in provider.py is the defect this pins"
    )
    assert calls == ["remote"], (
        f"{row.exit_description}: the helper's verdict must drive the pin "
        f"(dispatched={calls})"
    )
    assert result.status_code == 200, f"got {result.status_code}"

# ---------------------------------------------------------------------------
# AC1 (LP-0MVSMRT5R009EKP9): the refusal body reports the row's verdict
# ---------------------------------------------------------------------------

#: Rows that no longer produce a refusal body at all: the pin slice
#: (LP-0MVSLZY470051ZB2 AC4) declines the pin for a ``too_large`` verdict, so
#: those requests are served remotely instead of being refused.
_VERDICT_PARAMS = [
    pytest.param(
        row,
        id=row.trigger,
        marks=pytest.mark.skip(
            reason=(
                "no refusal body on this row: a too-large exit is promoted to "
                "the remote tiers instead of refused (LP-0MVSLZY470051ZB2 AC4)"
            )
        ),
    )
    if row.expected_verdict == admission_reserve.CAPACITY_TOO_LARGE
    else pytest.param(row, id=row.trigger)
    for row in _EXIT_ROWS
]


@pytest.mark.asyncio
@pytest.mark.parametrize("row", _VERDICT_PARAMS)
@pytest.mark.parametrize("preference", PIN_LOCAL_PREFERENCES)
async def test_refusal_body_reports_the_row_verdict(row: _ExitRow, preference: str):
    """Every refusing exit: the body classifies why local could not serve.

    ``deferrable`` means the block is temporal, so a pin-local client should
    retry rather than re-size. The mapping is pinned per row so a new exit
    cannot ship without stating its verdict; both wire spellings are asserted
    because the alias must not change the body (parent AC5).
    """
    result, calls = await _issue(row, preference)

    expected = ["local"] if row.local_response is not None else []
    assert calls == expected, (
        f"{row.exit_description}: a verdict assertion still assumes the "
        f"refusal path (dispatched={calls}, expected={expected})"
    )
    assert result.status_code == 429, (
        f"{row.exit_description}: expected the pin-local refusal, got "
        f"{result.status_code}"
    )
    body = json.loads(result.body)
    assert body.get("verdict") == row.expected_verdict, (
        f"{row.exit_description}: refusal body verdict must be "
        f"{row.expected_verdict!r} for reason {row.token!r}, got "
        f"{body.get('verdict')!r}"
    )


# ---------------------------------------------------------------------------
# AC3: prefer-local (and no header) ⇒ today's remote fall-through, unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("row", _EXIT_ROWS, ids=[r.trigger for r in _EXIT_ROWS])
@pytest.mark.parametrize("preference", [PREFER_LOCAL, None], ids=["header", "absent"])
async def test_prefer_local_keeps_todays_remote_fall_through(
    row: _ExitRow, preference: str | None
):
    """Every exit: the pre-existing remote dispatch happens, with no refusal.

    This is the byte-compatibility half of the contract (AC3): enforcing
    ``local-only`` must not turn the default client's fall-through into a
    refusal.
    """
    result, calls = await _issue(row, preference)

    expected = ["local", "remote"] if row.local_response is not None else ["remote"]
    assert calls == expected, (
        f"{row.exit_description}: prefer-local must fall through to the next "
        f"provider exactly as today (dispatched={calls}, expected={expected})"
    )
    assert result.status_code == 200, (
        f"{row.exit_description}: the default client keeps the remote answer, "
        f"got {result.status_code}"
    )
    assert result.headers.get("X-Local-Route") == "remote"
    body = json.loads(result.body)
    assert "code" not in body or body.get("code") != provider.EXHAUSTION_CODE_LOCAL_ADMISSION, (
        f"{row.exit_description}: no refusal body behind a default client"
    )


# ---------------------------------------------------------------------------
# Regression: no preference strips local entries (LP-0MVUH9OYZ007GMWF AC1)
# ---------------------------------------------------------------------------


def test_no_preference_strips_local_entries():
    """AC1: only a capacity verdict strips; a preference value never does.

    ``remote-ok`` used to remove every local entry, which is the opposite of
    what the value means ("local allowed, remote acceptable"). Its name is
    historic, so this row is what stops a future reader restoring the strip.
    """
    chain = [
        {"name": LOCAL_PROVIDER_NAME, "type": "local", "llama_model": "Qwen3"},
        {"name": REMOTE_PROVIDER_NAME, "type": "remote",
         "endpoint": "https://api.example.com/v1"},
    ]

    assert apply_local_preference(chain, REMOTE_OK) == chain, (
        "remote-ok must keep the local entry in the chain"
    )
    assert apply_local_preference(chain, PREFER_LOCAL) == chain, (
        "prefer-local keeps the chain intact (its fall-through is the contract)"
    )
    assert apply_local_preference(chain, LOCAL_ONLY) == chain, (
        "local-only keeps the chain intact so the refusal can report the truth"
    )


@pytest.mark.asyncio
async def test_remote_ok_with_no_remote_entry_serves_locally():
    """AC1/AC3: an all-local chain under ``remote-ok`` is served locally.

    A preference may never empty a chain: filtering the only local entry out
    would manufacture an empty chain and answer "all providers exhausted" to a
    model that can serve the request perfectly well locally.
    """
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):  # pragma: no cover - must never run
        calls.append("remote")
        return _ok_response()

    model_config = {
        "providers": [
            {"name": LOCAL_PROVIDER_NAME, "type": "local", "llama_model": "Qwen3"}
        ],
        "aliases": ["hybrid*"],
    }
    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(0, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(headers={"X-Local-Preference": REMOTE_OK}),
            "v1/chat/completions",
            model_config,
            {"provider_cooldown_seconds": 60, "server": {"session_slot_pool_size": 1}},
        )

    assert calls == ["local"], (
        f"remote-ok: the local route serves a remote-less chain, got {calls}"
    )
    assert result.status_code == 200, (
        f"an empty chain must never be manufactured from a preference, got "
        f"{result.status_code}"
    )
    assert result.headers.get("X-Local-Route") == "local"
    assert json.loads(result.body).get("code") != provider.EXHAUSTION_CODE_ALL


# ---------------------------------------------------------------------------
# AC6: the remote-fallback dispatch cycle carries the same choke point
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remote_fallback_cycle_enforces_the_local_only_choke_point():
    """The other dispatch cycle refuses too; prefer-local keeps its fall.

    ``proxy_with_fallback`` always runs the local+remote cycle, so a remote-
    only model takes ``proxy_with_remote_fallback`` — the second dispatch
    cycle the choke point must cover (LP-0MVRMC1L3006F0F3 AC ``both dispatch
    cycles are covered``). Its chain has no local entry, so ``local-only``
    answers the ``local_only_no_local_route`` refusal before any remote
    dispatch, while the default client still completes remotely with
    ``X-Local-Route: remote``.
    """
    calls: list[str] = []

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    model_config = {
        "providers": [
            {"name": REMOTE_PROVIDER_NAME, "type": "remote",
             "endpoint": "https://api.example.com/v1"}
        ],
        "aliases": ["hybrid*"],
    }
    config = {"provider_cooldown_seconds": 60,
              "server": {"session_slot_pool_size": 1}}

    async def _issue(preference: str | None):
        headers = {} if preference is None else {"X-Local-Preference": preference}
        with patch("proxy.server.proxy_to_remote", _remote):
            return await provider.proxy_with_remote_fallback(
                _DummyRequest(headers=headers),
                "v1/chat/completions",
                model_config,
                config,
            )

    refused = await _issue(LOCAL_ONLY)
    assert calls == [], (
        "local-only must refuse before the first remote dispatch "
        f"(dispatched={calls})"
    )
    assert refused.status_code == 429, f"got {refused.status_code}"
    assert refused.headers.get("X-Local-Route") == "refused"
    assert refused.headers.get("X-Local-Fallback-Reason") == (
        "local_only_no_local_route"
    ), "a remote-only chain has no local route; the refusal must say so"

    served = await _issue(PREFER_LOCAL)
    assert calls == ["remote"], (
        "prefer-local keeps today's remote dispatch on the remote-only cycle "
        f"(dispatched={calls})"
    )
    assert served.status_code == 200, f"got {served.status_code}"
    assert served.headers.get("X-Local-Route") == "remote"


# ---------------------------------------------------------------------------
# AC: a genuine local success after a skip still wins (no masking refusal)
# ---------------------------------------------------------------------------


async def _compaction_rescue_outcome(_server, _result, _body_json, _server_config):
    """Compaction applied: the bypassed session fits local again."""
    return {
        "evaluated": True,
        "action": "apply",
        "applied": True,
        "estimated_before": 99999,
        "estimated_after": 1_000,
        "reason": "context_too_large",
        "messages": [{"role": "user", "content": "compacted"}],
    }


@pytest.mark.asyncio
async def test_local_only_still_serves_a_compaction_rescued_local_dispatch():
    """AC4: a skip followed by a local rescue is NOT refused.

    The choke point lives at the loop head only: once the local branch
    rescues the request (server-side compaction success re-runs local
    eligibility and re-admits), the same iteration dispatches local — the
    refusal must not mask that later local success (LP-0MVRMC1L3006F0F3 AC).
    """
    calls: list[str] = []
    skip_calls = {"n": 0}

    def _skip_once_then_admit(*_a, **_k):
        skip_calls["n"] += 1
        if skip_calls["n"] == 1:
            return (True, "context_too_large")
        return (False, None)

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch(
            "proxy.provider._get_local_concurrency_info",
            _MutableConcurrency(0, 1),
        ),
        patch("proxy.provider.request_priority", return_value="low"),
        patch("proxy.provider._should_skip_local", _skip_once_then_admit),
        patch(
            "proxy.router_helpers.evaluate_and_apply_compaction",
            _compaction_rescue_outcome,
        ),
        _estimate_patches(PROMPT_TOKENS),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(headers={"X-Local-Preference": LOCAL_ONLY}),
            "v1/chat/completions",
            _model_config(_EXIT_ROWS[0]),
            _config(_EXIT_ROWS[0]),
        )

    assert calls == ["local"], (
        "a compaction-rescued request must still be served locally, got "
        f"{calls}"
    )
    assert result.status_code == 200
    assert result.headers.get("X-Local-Route") != "refused", (
        "the refusal must not mask a later local success"
    )


# ---------------------------------------------------------------------------
# Deferral (LP-0MVSM06BK008FZUE): a queue exit ends the wait in a 429
#
# The pin (LP-0MVSLZY470051ZB2) already stops a local-preferred request from
# being promoted once the loop resolves another candidate. This section pins
# the *exit-time* contract instead: the two contention-queue exits — wait
# budget expired, and the depth cap refusing to enqueue — must end the
# deferral where it happened, with a retryable 429 built by the one refusal
# builder and timed from the queue's own measured remainder. That is what makes
# the same guarantee hold when no remote candidate exists at all (a private /
# remote-less chain), where the loop-head choke point is never reached.
# ---------------------------------------------------------------------------

#: The two contention-queue exits (never-enqueued depth cap, and budget
#: expiry). Both share the ``fallback_after_queue`` reason token.
QUEUE_EXIT_TRIGGERS = ("queue_depth_cap", "queue_wait_expiry")
#: Tolerance (seconds) on the absolute ``retry_at`` instant.
RETRY_AT_TOLERANCE_SECONDS = 5


def _parse_retry_at(value) -> datetime:
    """Parse the refusal's RFC 3339 UTC ``retry_at``."""
    assert value, "a retryable refusal must carry an absolute retry_at"
    parsed = datetime.fromisoformat(str(value))
    assert parsed.utcoffset() == timedelta(0), f"retry_at must be UTC, got {value!r}"
    return parsed


def _expected_queue_remainder(row: _ExitRow) -> int:
    """The measured queue remainder a row's exit leaves (AC6).

    The depth cap never starts the queue clock, so the whole window remains;
    a wait-budget expiry has consumed it, so only the 1 s floor is left.
    """
    if row.trigger == "queue_depth_cap":
        return provider._contention_queue_expiry_seconds(_config(row))
    return 1


async def _issue_with_chain(
    row: _ExitRow, preference: str | None, model_config: dict
) -> tuple:
    """Issue *row*'s request against an explicit *model_config*.

    Mirrors :func:`_issue` (same row fixtures/patches) so a row can be run
    against a chain that has no remote entry at all. *preference* ``None``
    sends no ``X-Local-Preference`` header (the absent baseline).
    """
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):  # pragma: no cover - never reached
        calls.append("remote")
        return _ok_response()

    with ExitStack() as stack:
        stack.enter_context(patch("proxy.router.proxy_to_local", _local))
        stack.enter_context(patch("proxy.server.proxy_to_remote", _remote))
        stack.enter_context(
            patch(
                "proxy.provider._get_local_concurrency_info",
                _MutableConcurrency(*row.concurrency),
            )
        )
        stack.enter_context(patch("proxy.provider.request_priority", return_value="low"))
        for ctx in _row_patches(row):
            stack.enter_context(ctx)
        headers = (
            {} if preference is None else {"X-Local-Preference": preference}
        )
        result = await provider.proxy_with_fallback(
            _DummyRequest(headers=headers),
            "v1/chat/completions",
            model_config,
            _config(row),
        )
    return result, calls


def _assert_queue_exit_refusal(row: _ExitRow, result, calls: list[str]) -> dict:
    """The retryable-429 contract of a queue exit (AC1/AC2/AC3/AC6)."""
    assert calls == [], (
        f"{row.exit_description}: a deferred pin-local request must be refused, "
        f"never dispatched anywhere (dispatched={calls})"
    )
    assert result.status_code == 429, (
        f"{row.exit_description}: the queue exit must end in the retryable 429, "
        f"got {result.status_code}"
    )
    assert result.headers.getlist("x-local-route") == ["refused"], (
        f"{row.exit_description}: X-Local-Route must be stamped exactly once, "
        f"got {result.headers.getlist('x-local-route')!r}"
    )
    body = json.loads(result.body)
    # AC1: the header and the body agree on the canonical queue token.
    assert body.get("reason") == "fallback_after_queue"
    assert result.headers.get("X-Local-Fallback-Reason") == "fallback_after_queue"
    assert body.get("verdict") == admission_reserve.CAPACITY_DEFERRABLE, (
        f"{row.exit_description}: a queue exit is a bounded wait, so the "
        f"verdict must be deferrable, got {body.get('verdict')!r}"
    )
    # AC1: retryable, with a coherent Retry-After / retry_at pair.
    assert body.get("retryable") is True, "a bounded wait is retryable"
    retry_after = body.get("retry_after")
    assert isinstance(retry_after, int) and retry_after >= 1, (
        f"retry_after must be a whole number of seconds, got {retry_after!r}"
    )
    assert result.headers.get("Retry-After") == str(retry_after), (
        "Retry-After and body retry_after must agree"
    )
    expected_at = datetime.now(UTC) + timedelta(seconds=retry_after)
    assert (
        abs((_parse_retry_at(body.get("retry_at")) - expected_at).total_seconds())
        <= RETRY_AT_TOLERANCE_SECONDS
    ), "retry_at must be now + Retry-After"
    # AC6: the timer is `_refusal_retry_timing` fed with the *measured*
    # remainder — never a re-derived estimate of its own.
    expected_timing = provider._refusal_retry_timing(
        "fallback_after_queue",
        advertised_retry_after=_expected_queue_remainder(row),
        config=_config(row),
    )
    assert retry_after == expected_timing.retry_after, (
        f"{row.exit_description}: the refusal's timing must be the shared "
        f"decision fed with the measured remainder, got {retry_after} "
        f"(expected {expected_timing.retry_after})"
    )
    # AC3: this is the deferral refusal, never chain exhaustion.
    assert body.get("code") == provider.EXHAUSTION_CODE_LOCAL_ADMISSION
    assert b"All providers exhausted" not in result.body
    return body


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", QUEUE_EXIT_TRIGGERS)
async def test_queue_exit_refuses_with_a_retryable_429(trigger: str):
    """Both queue exits: a local-preferred deferrable request gets the 429.

    Explicit field assertions (not a shape smoke test): reason/verdict,
    retryable + Retry-After/retry_at coherence, the shared timing decision, and
    ``calls == []`` proving no remote provider was reached.
    """
    row = _row(trigger)
    depth_before = contention_queue.queue_depth()

    result, calls = await _issue(row, LOCAL_PREFERRED)

    _assert_queue_exit_refusal(row, result, calls)
    # AC2/AC4: the queue is left exactly as found — the depth-cap row's held
    # waiter is untouched and nothing was enqueued twice.
    seeded = 1 if trigger == "queue_depth_cap" else 0
    assert contention_queue.queue_depth() == depth_before + seeded, (
        f"{row.exit_description}: the refusal must not leak (or hold) a queue "
        f"slot, got depth {contention_queue.queue_depth()}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", QUEUE_EXIT_TRIGGERS)
async def test_queue_exit_refuses_on_a_remote_less_chain(trigger: str):
    """A private (no remote entry) chain defers and refuses — never exhausts.

    Parent assumption A5: a remote-less chain still defers and never promotes.
    The loop therefore never resolves another candidate, so the refusal must
    be produced at the queue exit itself rather than by the loop-head pin —
    otherwise this request would report "All providers exhausted" for a purely
    temporal condition (AC1/AC3).
    """
    row = _row(trigger)
    local_only_chain = {
        "providers": [
            {"name": LOCAL_PROVIDER_NAME, "type": "local", "llama_model": "Qwen3"},
        ],
        "aliases": ["hybrid*"],
    }

    result, calls = await _issue_with_chain(row, LOCAL_PREFERRED, local_only_chain)

    assert calls == [], "no local dispatch may happen on a queue exit"
    _assert_queue_exit_refusal(row, result, calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", QUEUE_EXIT_TRIGGERS)
async def test_queue_exit_refusal_is_not_restamped(trigger: str):
    """AC5: the top-level stamp helper sees an already-stamped refusal once.

    The refusal is stamped by its builder; the top-level
    ``_stamp_local_route_headers()`` choke point must leave it alone (one
    ``X-Local-Route`` value, one invocation) instead of adding a second stamp.
    """
    row = _row(trigger)
    seen: list[list[str]] = []
    real_stamp = provider._stamp_local_route_headers

    def _spy(response, request, model_config):
        seen.append(list(response.headers.getlist("x-local-route")))
        return real_stamp(response, request, model_config)

    with patch.object(provider, "_stamp_local_route_headers", _spy):
        result, _calls = await _issue(row, LOCAL_PREFERRED)

    assert seen == [["refused"]], (
        f"{row.exit_description}: the refusal must reach the top-level stamp "
        f"already stamped, got {seen!r}"
    )
    assert result.headers.getlist("x-local-route") == ["refused"], (
        "the refusal must be stamped exactly once"
    )


# ---------------------------------------------------------------------------
# AC1/AC5 (LP-0MVSM06Z8003UQTM): exactly one routing-outcome line per request
#
# Every row of the matrix is one request, and every row must produce exactly
# **one** ``local_preference_resolved`` line — not one per skip branch, not one
# per refusal builder. The line is emitted from the top-level stamp choke point
# both entries call once, so the count assertion is the structural proof; the
# outcome/reason assertions are what make a refusal distinguishable from chain
# exhaustion for an operator reading ``proxy.log``.
# ---------------------------------------------------------------------------

#: Outcome the single line must report for each row whose outcome is *not*
#: simply derived from the pin verdict: a spent deferral (``deferred``) and a
#: pre-loop capacity strip (``promoted``) are stamped at their decision point.
_ROW_EXPECTED_OUTCOME = {
    "queue_depth_cap": provider.LOCAL_ROUTING_OUTCOME_DEFERRED,
    "queue_wait_expiry": provider.LOCAL_ROUTING_OUTCOME_DEFERRED,
    "entry_context_cap": provider.LOCAL_ROUTING_OUTCOME_PROMOTED,
}


def _resolved_lines(caplog) -> list[str]:
    """Every ``local_preference_resolved`` line captured in *caplog*."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("local_preference_resolved ")
    ]


#: ``key=value`` pairs on the outcome line — the shape log-grep tooling parses.
_RESOLVED_LINE_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


def _resolved_line_fields(line: str) -> dict[str, str]:
    """Parse the ``key=value`` pairs of the outcome *line* into a dict."""
    return {
        match.group(1): match.group(2)
        for match in _RESOLVED_LINE_KV_RE.finditer(line)
    }


def _expected_outcome(row: _ExitRow) -> str:
    """The outcome *row* must report on its single per-request line."""
    pinned = row.expected_verdict != admission_reserve.CAPACITY_TOO_LARGE
    default = (
        provider.LOCAL_ROUTING_OUTCOME_PINNED
        if pinned
        else provider.LOCAL_ROUTING_OUTCOME_REMOTE
    )
    return _ROW_EXPECTED_OUTCOME.get(row.trigger, default)


@pytest.mark.asyncio
@pytest.mark.parametrize("row", _EXIT_ROWS, ids=[r.trigger for r in _EXIT_ROWS])
@pytest.mark.parametrize("preference", PIN_LOCAL_PREFERENCES)
async def test_every_row_logs_exactly_one_outcome_line(
    row: _ExitRow, preference: str, caplog
):
    """One request, one line — whatever path the local route took (AC1/AC5).

    The count is asserted per row, so a future per-branch log call (the
    duplication this item exists to prevent) fails the row instead of quietly
    doubling the operator's log volume. The line names the resolved preference
    (the legacy ``local-only`` alias reports ``local-preferred``), the outcome
    and the row's own reason token, so a deferral can never be mistaken for
    chain exhaustion (AC2).
    """
    caplog.set_level(logging.INFO, logger="llama-proxy.provider")

    await _issue(row, preference)

    lines = _resolved_lines(caplog)
    assert len(lines) == 1, (
        f"{row.exit_description}: exactly one local_preference_resolved line per "
        f"request is required; got {len(lines)}: {lines!r}"
    )
    fields = _resolved_line_fields(lines[0])
    assert fields["preference"] == LOCAL_PREFERRED, (
        f"{row.exit_description}: the resolved preference is always reported "
        f"under its canonical spelling (legacy local-only normalises), got "
        f"{fields.get('preference')!r}"
    )
    assert fields["outcome"] == _expected_outcome(row), (
        f"{row.exit_description}: expected outcome={_expected_outcome(row)}, got "
        f"{fields.get('outcome')!r}. Line: {lines[0]!r}"
    )
    assert fields["outcome"] in provider.LOCAL_ROUTING_OUTCOMES, (
        f"{row.exit_description}: the outcome must come from the closed "
        f"vocabulary, got {fields.get('outcome')!r}"
    )
    expected_reason = (
        "none"
        if fields["outcome"] == provider.LOCAL_ROUTING_OUTCOME_PROMOTED
        else row.token
    )
    assert fields["reason"] == expected_reason, (
        f"{row.exit_description}: expected reason={expected_reason}, got "
        f"{fields.get('reason')!r}. Line: {lines[0]!r}"
    )


# ---------------------------------------------------------------------------
# The behaviour matrix (LP-0MVSHXTX1007D58Z)
#
# One table for the whole contract: for every (preference, slot state, capacity
# verdict) combination a client can hit, the matrix states what it observes —
# HTTP status, ``X-Local-Route``, the ``X-Provider`` that served it, whether
# *anything* was dispatched upstream, and, for a deferral, the retry story.
#
# The rows cite the exit fixtures above (``exit_trigger``) instead of restating
# what each exit means: those tables own "how local became unavailable", this
# one owns "what the client sees". Deep per-exit coverage (queue invariants,
# one-line-per-request logging, the single verdict source) stays in the
# sections above; the matrix is the cross-cutting summary, so a red parameter
# names the behaviour that broke rather than an implementation detail (AC4).
#
# Two baseline rows (an absent header and ``prefer-local`` on a healthy local
# route) additionally assert *byte-compatibility*: a client that never asked
# for pin-local behaviour keeps today's response shape, with no ``retryable``
# or ``verdict`` field invented for it (parent AC5).
# ---------------------------------------------------------------------------

#: Pseudo-trigger for the healthy route: a local entry, a free slot, no exit at
#: all — the unchanged baseline the preference work must not disturb.
HEALTHY_FREE_SLOT = "healthy_free_slot"

_HEALTHY_ROW = _ExitRow(
    trigger=HEALTHY_FREE_SLOT,
    token="",
    expected_verdict=admission_reserve.CAPACITY_DEFERRABLE,
    exit_description="a healthy local route with a free slot",
    concurrency=(0, 1),
)


@dataclass(frozen=True)
class _BehaviourMatrixRow:
    """One row of the local-preference behaviour matrix.

    Every field is an *observation* (what the client sees), never a mechanism:
    the mechanism is the exit fixture the row points at.
    """

    #: Parametrisation id, composed ``preference-slot-outcome`` so a red row
    #: reads as behaviour (AC4).
    behaviour: str
    #: Wire value sent in ``X-Local-Preference``; ``None`` = header absent.
    preference: str | None
    #: The exit fixture the row runs against (or :data:`HEALTHY_FREE_SLOT`).
    exit_trigger: str = HEALTHY_FREE_SLOT
    #: Slot-state override; ``None`` keeps the exit fixture's own state.
    concurrency: tuple[int, int] | None = None
    #: Serve from a chain with no remote entry at all (a ``private`` model).
    remote_less_chain: bool = False
    expected_status: int = 200
    expected_route: str = "local"
    #: ``X-Provider`` of the served response; ``None`` when nothing served.
    expected_provider: str | None = LOCAL_PROVIDER_NAME
    #: Ordered upstream dispatches: ``("remote",)`` proves remote dispatch,
    #: ``("local",)`` proves a local serve, ``()`` proves nothing dispatched.
    expected_dispatches: tuple[str, ...] = ()
    #: Refusal contract: the body ``reason`` and the body ``verdict``.
    #: ``None`` reason = the row is served, not refused (and every refusal row
    #: here is deferrable, so its retry story is asserted in the test).
    expected_refusal_reason: str | None = None
    expected_verdict: str | None = None


_BEHAVIOUR_MATRIX = (
    pytest.param(
        _BehaviourMatrixRow(
            behaviour="prefer-local-free-slot-serves-local",
            preference=PREFER_LOCAL,
            expected_dispatches=("local",),
        ),
        id="prefer-local-free-slot-serves-local",
    ),
    pytest.param(
        _BehaviourMatrixRow(
            behaviour="absent-free-slot-serves-local",
            preference=None,
            expected_dispatches=("local",),
        ),
        id="absent-free-slot-serves-local",
    ),
    pytest.param(
        _BehaviourMatrixRow(
            behaviour="prefer-local-busy-deferrable-falls-through-remote",
            preference=PREFER_LOCAL,
            exit_trigger="chain_cooldown",
            concurrency=(1, 1),
            expected_route="remote",
            expected_provider=REMOTE_PROVIDER_NAME,
            expected_dispatches=("remote",),
        ),
        id="prefer-local-busy-deferrable-falls-through-remote",
    ),
    pytest.param(
        _BehaviourMatrixRow(
            behaviour="remote-ok-free-slot-serves-local",
            preference=REMOTE_OK,
            expected_dispatches=("local",),
        ),
        id="remote-ok-free-slot-serves-local",
    ),
    pytest.param(
        _BehaviourMatrixRow(
            behaviour="local-preferred-free-slot-serves-local",
            preference=LOCAL_PREFERRED,
            expected_dispatches=("local",),
        ),
        id="local-preferred-free-slot-serves-local",
    ),
    pytest.param(
        _BehaviourMatrixRow(
            behaviour="local-preferred-busy-deferrable-refuses",
            preference=LOCAL_PREFERRED,
            exit_trigger="queue_wait_expiry",
            expected_status=429,
            expected_route="refused",
            expected_provider=None,
            expected_dispatches=(),
            expected_refusal_reason="fallback_after_queue",
            expected_verdict=admission_reserve.CAPACITY_DEFERRABLE,
        ),
        id="local-preferred-busy-deferrable-refuses",
    ),
    pytest.param(
        _BehaviourMatrixRow(
            behaviour="local-preferred-busy-too-large-promotes-remote",
            preference=LOCAL_PREFERRED,
            exit_trigger="entry_context_cap",
            concurrency=(1, 1),
            expected_route="remote",
            expected_provider=REMOTE_PROVIDER_NAME,
            expected_dispatches=("remote",),
        ),
        id="local-preferred-busy-too-large-promotes-remote",
    ),
    pytest.param(
        _BehaviourMatrixRow(
            behaviour="local-only-legacy-busy-deferrable-refuses",
            preference=LOCAL_ONLY,
            exit_trigger="queue_wait_expiry",
            expected_status=429,
            expected_route="refused",
            expected_provider=None,
            expected_dispatches=(),
            expected_refusal_reason="fallback_after_queue",
            expected_verdict=admission_reserve.CAPACITY_DEFERRABLE,
        ),
        id="local-only-legacy-busy-deferrable-refuses",
    ),
    pytest.param(
        _BehaviourMatrixRow(
            behaviour="local-preferred-remote-less-chain-still-defers",
            preference=LOCAL_PREFERRED,
            exit_trigger="queue_wait_expiry",
            remote_less_chain=True,
            expected_status=429,
            expected_route="refused",
            expected_provider=None,
            expected_dispatches=(),
            expected_refusal_reason="fallback_after_queue",
            expected_verdict=admission_reserve.CAPACITY_DEFERRABLE,
        ),
        id="local-preferred-remote-less-chain-still-defers",
    ),
)

#: The deprecation line the legacy alias must emit exactly once per request.
_DEPRECATION_MARKER = "deprecated X-Local-Preference"


def _matrix_exit_row(row: _BehaviourMatrixRow) -> _ExitRow:
    """The exit fixture *row* runs against, with its slot state applied."""
    exit_row = (
        _HEALTHY_ROW
        if row.exit_trigger == HEALTHY_FREE_SLOT
        else _row(row.exit_trigger)
    )
    if row.concurrency is None:
        return exit_row
    return replace(exit_row, concurrency=row.concurrency)


def _matrix_model_config(row: _BehaviourMatrixRow, exit_row: _ExitRow) -> dict:
    """The model chain for *row* (remote-less when the row asks for it)."""
    model_config = _model_config(exit_row)
    if row.remote_less_chain:
        model_config["providers"] = [
            entry
            for entry in model_config["providers"]
            if entry.get("type") == "local"
        ]
    return model_config


async def _issue_matrix_row(row: _BehaviourMatrixRow) -> tuple:
    """Issue *row* end-to-end; return ``(response, dispatches)``."""
    exit_row = _matrix_exit_row(row)
    return await _issue_with_chain(
        exit_row, row.preference, _matrix_model_config(row, exit_row)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("row", _BEHAVIOUR_MATRIX)
async def test_each_matrix_row_serves_or_refuses_as_the_preference_matrix_says(
    row: _BehaviourMatrixRow, caplog
):
    """One matrix row: status, route, serving provider, dispatches, retry story.

    Each row carries its own expected values, so a red parameter names the
    behaviour that regressed (AC1) and ``calls`` makes "never dispatched" a
    first-class assertion rather than an inference from the response (AC2).
    """
    caplog.set_level(logging.WARNING, logger="llama-proxy")

    result, calls = await _issue_matrix_row(row)

    assert result.status_code == row.expected_status, (
        f"{row.behaviour}: expected HTTP {row.expected_status}, got "
        f"{result.status_code}"
    )
    assert result.headers.get("X-Local-Route") == row.expected_route, (
        f"{row.behaviour}: expected X-Local-Route={row.expected_route!r}, got "
        f"{result.headers.get('X-Local-Route')!r}"
    )
    assert result.headers.get("X-Provider") == row.expected_provider, (
        f"{row.behaviour}: expected X-Provider={row.expected_provider!r}, got "
        f"{result.headers.get('X-Provider')!r}"
    )
    assert calls == list(row.expected_dispatches), (
        f"{row.behaviour}: expected upstream dispatches "
        f"{list(row.expected_dispatches)}, got {calls}"
    )

    body = json.loads(result.body)
    if row.expected_refusal_reason is None:
        # Byte-compatibility: no retry story is invented for a served request.
        assert "retryable" not in body, (
            f"{row.behaviour}: a served response must keep today's body shape "
            f"(no 'retryable' field), got {sorted(body)}"
        )
        assert "verdict" not in body, (
            f"{row.behaviour}: a served response must keep today's body shape "
            f"(no 'verdict' field), got {sorted(body)}"
        )
        assert result.headers.get("Retry-After") is None, (
            f"{row.behaviour}: a served response must not advertise a retry"
        )
        assert body.get("code") != provider.EXHAUSTION_CODE_LOCAL_ADMISSION, (
            f"{row.behaviour}: a served request must not carry the "
            f"local-capacity refusal code"
        )
    else:
        assert body.get("reason") == row.expected_refusal_reason, (
            f"{row.behaviour}: refusal reason must be "
            f"{row.expected_refusal_reason!r}, got {body.get('reason')!r}"
        )
        assert result.headers.get("X-Local-Fallback-Reason") == (
            row.expected_refusal_reason
        ), f"{row.behaviour}: the header and the body must agree on the reason"
        assert body.get("verdict") == row.expected_verdict, (
            f"{row.behaviour}: refusal verdict must be {row.expected_verdict!r}, "
            f"got {body.get('verdict')!r}"
        )
        assert bool(body.get("retryable")) is True, (
            f"{row.behaviour}: a deferral is a bounded wait, so it is retryable"
        )
        retry_after = body.get("retry_after")
        assert isinstance(retry_after, int) and retry_after >= 1, (
            f"{row.behaviour}: retry_after must be whole seconds >= 1, got "
            f"{retry_after!r}"
        )
        assert result.headers.get("Retry-After") == str(retry_after), (
            f"{row.behaviour}: Retry-After and body retry_after must agree"
        )
        _parse_retry_at(body.get("retry_at"))

    # The legacy alias normalises silently-but-logged: exactly one deprecation
    # line per request, and none for a canonical value (parent AC5).
    deprecations = [
        record.getMessage()
        for record in caplog.records
        if _DEPRECATION_MARKER in record.getMessage()
    ]
    expected_deprecations = 1 if row.preference == LOCAL_ONLY else 0
    assert len(deprecations) == expected_deprecations, (
        f"{row.behaviour}: expected {expected_deprecations} deprecation line(s) "
        f"for preference {row.preference!r}, got {deprecations!r}"
    )


def test_behaviour_matrix_covers_every_documented_behaviour():
    """The table is the contract: one row per documented behaviour, unique ids.

    The previous incident (2026-10-28) was a matrix row that did not exist:
    a fits-local, pin-local request promoted to a paid provider. A new
    preference/slot/verdict combination must therefore arrive with a row, and a
    row that disappears loses its guarantee loudly rather than silently.
    """
    ids = [param.id for param in _BEHAVIOUR_MATRIX]
    assert len(ids) == len(set(ids)), f"duplicate behaviour rows: {ids}"
    assert set(ids) == {
        "prefer-local-free-slot-serves-local",
        "absent-free-slot-serves-local",
        "prefer-local-busy-deferrable-falls-through-remote",
        "remote-ok-free-slot-serves-local",
        "local-preferred-free-slot-serves-local",
        "local-preferred-busy-deferrable-refuses",
        "local-preferred-busy-too-large-promotes-remote",
        "local-only-legacy-busy-deferrable-refuses",
        "local-preferred-remote-less-chain-still-defers",
    }, "a behaviour matrix row is missing from (or unexpected in) the table"

    for param in _BEHAVIOUR_MATRIX:
        row = param.values[0]
        assert row.behaviour == param.id, (
            f"{param.id}: the row's behaviour label must be its "
            f"parametrisation id, got {row.behaviour!r} (AC4)"
        )
        assert row.behaviour.startswith(
            ("prefer-local", "absent", "remote-ok", "local-preferred", "local-only")
        ), f"{param.id}: the id must start with the preference it pins"
