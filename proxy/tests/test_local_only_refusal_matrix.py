"""Local-only refusal matrix: one row per exit from the local route.

LP-0MVRMC0Z500883O0 (child of LP-0MVNU2KTK009D4AG, AC1/AC2/AC3/AC6).

The contract under test is one sentence: with ``X-Local-Preference: local-only``
a request either completes **locally** or receives the local-only refusal — it
is never handed to a remote provider. Every distinct way out of the local route
is one row of :data:`_EXIT_ROWS`, and every row asserts the same two outcomes:

- ``local-only`` ⇒ HTTP 429, ``X-Local-Route: refused``,
  ``X-Local-Fallback-Reason: <token>``, and **no upstream call at all** (a
  refused request is neither dispatched locally nor moved remote);
- ``prefer-local`` and an absent header ⇒ today's remote dispatch, with no
  refusal and no refusal body (AC3 byte-compatibility).

The structural counter-case closes the loop: ``remote-ok`` still strips local
entries from the chain, and an all-local chain under ``remote-ok`` still
answers today's exhaustion response.

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
the fail-safe ``local_only_no_local_route`` default. The verdict assertions are
the TDD red phase for the wiring child (LP-0MVSMINPR006SWT6): they start
``xfail(strict=True)`` because today's 429 body has no ``verdict`` field, and
that child flips them green in the same change that adds it.
"""

from __future__ import annotations

import json
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
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
)

#: Every row is asserted green: the choke point (LP-0MVRMC1L3006F0F3)
#: refuses every exit and the reason-token child (LP-0MVRMC1W6003MV3Y)
#: records the per-branch chain tokens. A future red row is a real failure —
#: the xfail marker machinery died with the behaviour it pinned.

_LOCAL_ONLY_PARAMS = [
    pytest.param(row, id=row.trigger) for row in _EXIT_ROWS
]


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
    else:  # pragma: no cover - a new row must name its trigger
        raise AssertionError(f"unknown local-route exit trigger {trigger!r}")
    return patches


async def _issue(row: _ExitRow, preference: str | None) -> tuple:
    """Issue *row*'s request with *preference*; return ``(response, calls)``.

    *calls* is the ordered record of upstream dispatches: exactly one entry per
    attempt, so ``[]`` proves nothing was dispatched and ``["remote"]`` proves
    today's remote fall-through still happens.
    """
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

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
            _DummyRequest(headers=headers),
            "v1/chat/completions",
            _model_config(row),
            _config(row),
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
# AC1 + AC2: local-only ⇒ refusal, on every exit, with no remote dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("row", _LOCAL_ONLY_PARAMS)
async def test_local_only_refuses_instead_of_dispatching_remote(row: _ExitRow):
    """Every exit: 429 refusal naming the reason, and nothing dispatched.

    The assertion that carries the incident is ``calls == []``: a
    ``local-only`` client must be neither served locally *nor* handed to a
    remote provider, whatever reason the local route was unavailable.
    """
    result, calls = await _issue(row, LOCAL_ONLY)

    assert calls == [], (
        f"{row.exit_description}: a local-only request must never be dispatched "
        f"(dispatched={calls}) — it is refused, not moved to a paid provider"
    )
    assert result.status_code == 429, (
        f"{row.exit_description}: local-only means the refusal, got "
        f"{result.status_code}"
    )
    assert result.headers.get("X-Local-Route") == "refused", (
        f"{row.exit_description}: the response must declare the refusal"
    )
    assert result.headers.get("X-Local-Fallback-Reason") == row.token, (
        f"{row.exit_description}: the refusal must carry the stable reason token "
        f"{row.token!r}, got "
        f"{result.headers.get('X-Local-Fallback-Reason')!r}"
    )
    # The refusal is the stable code/header contract, not prose: an
    # unparseable body or a 5xx would break clients that classify on it.
    body = json.loads(result.body)
    assert body.get("code"), f"{row.exit_description}: refusal needs a stable code"
    assert body.get("reason") == row.token, (
        f"{row.exit_description}: body reason and header must agree, got "
        f"{body.get('reason')!r}"
    )


# ---------------------------------------------------------------------------
# AC1 (LP-0MVSMRT5R009EKP9): the refusal body reports the row's verdict
# ---------------------------------------------------------------------------

#: TDD red phase for the wiring child (LP-0MVSMINPR006SWT6): today's 429 body
#: has no ``verdict`` field, so every row's verdict assertion is a genuine
#: expected failure until that child adds it. ``strict=True`` means the marker
#: must be removed in the same change that adds the behaviour.
_VERDICT_XFAIL = pytest.mark.xfail(
    strict=True,
    reason=(
        "capacity-verdict wiring pending (LP-0MVSMINPR006SWT6): the 429 "
        "refusal body has no 'verdict' field"
    ),
)


@_VERDICT_XFAIL
@pytest.mark.asyncio
@pytest.mark.parametrize("row", _LOCAL_ONLY_PARAMS)
async def test_local_only_refusal_body_reports_the_row_verdict(row: _ExitRow):
    """Every exit: the refusal body classifies why local could not serve.

    ``too_large`` means waiting cannot help (the request exceeds the local
    serving envelope); ``deferrable`` means the block is temporal, so a
    ``local-only`` client should retry rather than re-size. The mapping is
    pinned per row so a new exit cannot ship without stating its verdict.
    """
    result, calls = await _issue(row, LOCAL_ONLY)

    assert calls == [], (
        f"{row.exit_description}: a verdict assertion still assumes the "
        f"refusal path (dispatched={calls})"
    )
    assert result.status_code == 429, (
        f"{row.exit_description}: expected the local-only refusal, got "
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

    assert calls == ["remote"], (
        f"{row.exit_description}: prefer-local must fall through to the next "
        f"provider exactly as today (dispatched={calls})"
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
# Regression: remote-ok is enforced structurally and stays as it is
# ---------------------------------------------------------------------------


def test_remote_ok_still_strips_local_entries():
    """AC5: ``remote-ok`` removes local entries; the other two keep the chain."""
    chain = [
        {"name": LOCAL_PROVIDER_NAME, "type": "local", "llama_model": "Qwen3"},
        {"name": REMOTE_PROVIDER_NAME, "type": "remote",
         "endpoint": "https://api.example.com/v1"},
    ]

    assert apply_local_preference(chain, REMOTE_OK) == [chain[1]], (
        "remote-ok must strip every local entry from the chain"
    )
    assert apply_local_preference(chain, PREFER_LOCAL) == chain, (
        "prefer-local keeps the chain intact (its fall-through is the contract)"
    )
    assert apply_local_preference(chain, LOCAL_ONLY) == chain, (
        "local-only keeps the chain intact so the refusal can report the truth"
    )


@pytest.mark.asyncio
async def test_remote_ok_with_no_remote_entry_returns_todays_exhaustion():
    """AC5: an all-local chain under ``remote-ok`` exhausts, it does not refuse.

    ``remote-ok`` is filtered structurally (never a 429 refusal): the chain
    becomes genuinely empty and the pre-existing exhaustion path answers.
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

    assert calls == [], "remote-ok: the local route is never attempted"
    assert result.status_code == 503, (
        f"an empty chain exhausts as today, got {result.status_code}"
    )
    assert json.loads(result.body).get("code") == provider.EXHAUSTION_CODE_ALL


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
