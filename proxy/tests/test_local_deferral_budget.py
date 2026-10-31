"""Bounded ``local-preferred`` deferral: three refusals, then remote.

LP-0MVUH9OMY004MQ3U (child of LP-0MVU1GSDY003EJ0O; covers parent AC5, AC6, AC7,
AC8 — child AC3, AC6-deferral).

The contract, written before it exists: a pin-local client that keeps being
told to retry must eventually get an answer. Concretely, per session:

``refusals 1–3``
    Unchanged retryable ``429``s (``X-Local-Route: refused``, ``retryable:
    true``, ``Retry-After`` + ``retry_at``) that now also report the budget —
    ``deferral_refusals`` (count so far) and ``deferral_refusals_remaining``
    (``0`` on the third). The body change is *purely additive*: every
    pre-existing field is still there.

``the fourth request``
    Is **not** refused. The local entry leaves the chain for that request, a
    remote tier serves it, and the promotion is visible: ``X-Local-Route:
    remote``, ``X-Local-Fallback-Reason: deferral_refusals_exhausted``, one log
    line naming the session, the count and the threshold, and no ``Retry-After``
    (the terminal path does not advertise a wait it will not honour).

``resets``
    A successful local dispatch, a ``too_large`` promotion and the entry's TTL
    expiry each clear the counter — pinned one test per trigger, because a
    stale counter would escalate a session that had already been served
    locally (parent risk R3).

``unchanged``
    A request with no resolvable session id keeps today's unbounded deferral
    (there is no stable key to count against), a remote-less chain never
    escalates (the cost-capped escape hatch of parent assumption A4), and
    ``local_deferral_max_refusals: 0`` restores today's behaviour exactly.

Red phase: the ``xfail(strict=True)`` specs are red today (no counter, no body
fields, no escalation token) and LP-0MVUH9PL0005PW90 turns them green. The
unmarked specs are guards that hold today and must keep holding.

The counter store itself is exercised through the pre-agreed surface documented
in :mod:`local_preference_v2_support` (``record_refusal`` / ``refusal_count`` /
``reset`` / ``_now``), so the store's internals stay the implementing child's
choice while its observable behaviour is pinned here.
"""

from __future__ import annotations

import local_preference_v2_support as support
import pytest
from local_preference_v2_support import (
    BODY_REFUSALS,
    BODY_REFUSALS_REMAINING,
    DEFAULT_DEFERRAL_MAX_REFUSALS,
    ESCALATION_TOKEN,
    LOCAL_PREFERRED,
)

from proxy import admission_reserve

#: A deterministic *deferrable* refusal: a cache-cold 100k prompt priced far
#: outside the 60 s wait budget, refused by the free-slot admission guard while
#: a local slot is free. Retryable, so it is exactly what the counter counts.
_DEFERRAL_ROW = support.matrix._row("admission_guard_free_slot")
#: A healthy free slot: a *successful local dispatch* (the first reset trigger).
_HEALTHY_ROW = support.matrix._HEALTHY_ROW
#: A ``too_large`` promotion (the second reset trigger): the local entry leaves
#: the chain and the remote tier serves the request.
_TOO_LARGE_ROW = support.matrix._row("entry_context_cap")

SESSION = "contract-deferral-session"
OTHER_SESSION = "contract-deferral-session-2"

#: The pre-existing refusal-body keys that must survive additively (parent AC7).
_PRE_EXISTING_BODY_KEYS = (
    "error",
    "code",
    "detail",
    "reason",
    "verdict",
    "eta_prefill_seconds",
    "budget_seconds",
    "retryable",
    "retry_after",
    "suggested_action",
)


@pytest.fixture(autouse=True)
def _clean_state():
    support.reset_state()
    yield
    support.reset_state()


# ---------------------------------------------------------------------------
# AC3 (parent AC5/AC7): refusals 1–3 keep the timing contract and report budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "LP-0MVUH9PL0005PW90: the refusal body has no deferral_refusals / "
        "deferral_refusals_remaining fields yet"
    ),
)
async def test_refusals_one_to_three_advertise_the_remaining_budget():
    """Every retryable refusal counts itself down; nothing else changes.

    One sequence, three rows: refusal *n* reports ``deferral_refusals == n`` and
    ``deferral_refusals_remaining == 3 - n`` (so the third says ``0`` and the
    client can see the escalation coming), while the timing contract of
    LP-0MVNTT4X8009IO2X stays exactly as it was.
    """
    results = await support.issue_sequence(
        _DEFERRAL_ROW,
        count=DEFAULT_DEFERRAL_MAX_REFUSALS,
        preference=LOCAL_PREFERRED,
        session_id=SESSION,
    )

    for count, (result, calls, _request) in enumerate(results, start=1):
        context = f"refusal {count} of {DEFAULT_DEFERRAL_MAX_REFUSALS}"
        assert result.status_code == 429, (
            f"{context}: a refusal inside the budget is still a retryable 429, "
            f"got {result.status_code}"
        )
        assert support.route_of(result) == "refused", (
            f"{context}: the refusal must declare X-Local-Route: refused"
        )
        assert support.fallback_reason_of(result) == _DEFERRAL_ROW.token, (
            f"{context}: the refusal reason token must stay "
            f"{_DEFERRAL_ROW.token!r}"
        )
        body = support.body_of(result)
        assert body.get("retryable") is True, (
            f"{context}: a refusal inside the budget is retryable"
        )
        assert result.headers.get("Retry-After") == str(body.get("retry_after")), (
            f"{context}: Retry-After and body retry_after must agree"
        )
        assert body.get("retry_at"), (
            f"{context}: a retryable refusal carries an absolute retry_at"
        )
        assert calls == [], (
            f"{context}: a refused pin-local request reaches no provider "
            f"(dispatched={calls})"
        )
        assert body.get(BODY_REFUSALS) == count, (
            f"{context}: the body must report deferral_refusals="
            f"{count}, got {body.get(BODY_REFUSALS)!r}"
        )
        assert body.get(BODY_REFUSALS_REMAINING) == (
            DEFAULT_DEFERRAL_MAX_REFUSALS - count
        ), (
            f"{context}: deferral_refusals_remaining must be "
            f"{DEFAULT_DEFERRAL_MAX_REFUSALS - count}, got "
            f"{body.get(BODY_REFUSALS_REMAINING)!r}"
        )
        for key in _PRE_EXISTING_BODY_KEYS:
            assert key in body, (
                f"{context}: the budget fields are additive — the pre-existing "
                f"key {key!r} must still be present (body keys: {sorted(body)})"
            )


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "LP-0MVUH9PL0005PW90: the fourth request from a spent session is still "
        "a fourth retryable 429 instead of a remote dispatch"
    ),
)
async def test_fourth_request_escalates_to_remote_visibly(caplog):
    """After three refusals the next request is served remotely, and says so.

    The escalation must be *visible and distinguishable* from a ``too_large``
    promotion: ``X-Local-Route: remote`` plus the dedicated reason token, one
    log line naming the session/count/threshold, and no ``Retry-After`` — the
    terminal path must not advertise a wait it will not honour
    (LP-0MVNTT4X8009IO2X).
    """
    support.set_debug_logging(caplog)
    refusals = await support.issue_sequence(
        _DEFERRAL_ROW,
        count=DEFAULT_DEFERRAL_MAX_REFUSALS,
        preference=LOCAL_PREFERRED,
        session_id=SESSION,
    )
    assert [result.status_code for result, _c, _r in refusals] == [429, 429, 429], (
        "the first three requests must be refusals, whatever the fourth does"
    )

    result, calls, request = await support.issue(
        _DEFERRAL_ROW, preference=LOCAL_PREFERRED, session_id=SESSION
    )

    assert result.status_code == 200, (
        "the fourth request from a spent session must be served, not refused "
        f"again (got {result.status_code})"
    )
    assert support.route_of(result) == "remote", (
        "the escalation is a remote dispatch: X-Local-Route must be 'remote', "
        f"got {support.route_of(result)!r}"
    )
    assert support.fallback_reason_of(result) == ESCALATION_TOKEN, (
        "the escalation must be distinguishable from a too_large promotion by "
        f"its reason token ({ESCALATION_TOKEN!r}), got "
        f"{support.fallback_reason_of(result)!r}"
    )
    assert calls == ["remote"], (
        f"exactly one upstream dispatch serves the escalated request, got {calls}"
    )
    assert support.queue_wait_of(request) == 0.0, (
        "an escalated request must not spend any contention-queue wait "
        f"(waited {support.queue_wait_of(request)}s)"
    )
    assert result.headers.get("Retry-After") is None, (
        "the terminal path must not advertise a retry timer"
    )

    token_lines = support.escalation_lines(caplog)
    outcome_lines = [
        line for line in token_lines if "local_preference_resolved" in line
    ]
    escalation_lines = [line for line in token_lines if line not in outcome_lines]
    assert len(escalation_lines) == 1, (
        "exactly one escalation log line is emitted (the routing-outcome line "
        f"may name the same token), got {escalation_lines!r}"
    )
    assert SESSION in escalation_lines[0], (
        "the escalation line must be attributable by session, got "
        f"{escalation_lines[0]!r}"
    )
    assert str(DEFAULT_DEFERRAL_MAX_REFUSALS) in escalation_lines[0], (
        "the escalation line must name the refusal count/threshold, got "
        f"{escalation_lines[0]!r}"
    )


@pytest.mark.xfail(
    strict=True,
    reason="LP-0MVUH9PL0005PW90: the escalation token is not in the token table",
)
def test_escalation_reason_token_is_canonical():
    """One token table: header, body and log field cannot drift apart."""
    assert ESCALATION_TOKEN in admission_reserve.LOCAL_FALLBACK_REASON_TOKENS, (
        f"{ESCALATION_TOKEN!r} must be registered in "
        "admission_reserve.LOCAL_FALLBACK_REASON_TOKENS"
    )
    assert (
        admission_reserve.canonical_local_fallback_reason(ESCALATION_TOKEN)
        == ESCALATION_TOKEN
    )


# ---------------------------------------------------------------------------
# AC3 (parent AC8): the counter resets — one row per trigger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "LP-0MVUH9PL0005PW90: the counter does not exist, so a local success "
        "cannot reset the budget and the body reports no count"
    ),
)
async def test_local_success_resets_the_counter():
    """Being served locally clears the session's budget.

    Run with ``local_deferral_max_refusals: 1`` so the reset is *observable*:
    without it the post-success refusal would be the escalation.
    """
    refusal, _calls, _request = await support.issue(
        _DEFERRAL_ROW,
        preference=LOCAL_PREFERRED,
        session_id=SESSION,
        config=support.config_for(_DEFERRAL_ROW, deferral_max_refusals=1),
    )
    assert refusal.status_code == 429
    assert support.body_of(refusal).get(BODY_REFUSALS) == 1

    served, served_calls, _request = await support.issue(
        _HEALTHY_ROW,
        preference=LOCAL_PREFERRED,
        session_id=SESSION,
        config=support.config_for(_HEALTHY_ROW, deferral_max_refusals=1),
    )
    assert served.status_code == 200, (
        f"the interleaved local dispatch must succeed, got {served.status_code}"
    )
    assert served_calls == ["local"], (
        f"the reset trigger is a *local* serve, got dispatches {served_calls}"
    )

    after, after_calls, _request = await support.issue(
        _DEFERRAL_ROW,
        preference=LOCAL_PREFERRED,
        session_id=SESSION,
        config=support.config_for(_DEFERRAL_ROW, deferral_max_refusals=1),
    )
    assert after.status_code == 429, (
        "after a local success the session is back inside its budget, so the "
        f"next refusal is a 429, got {after.status_code}"
    )
    assert support.body_of(after).get(BODY_REFUSALS) == 1, (
        "the count restarts at 1 after a local success, got "
        f"{support.body_of(after).get(BODY_REFUSALS)!r}"
    )
    assert after_calls == [], (
        f"a refused request dispatches nothing, got {after_calls}"
    )


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "LP-0MVUH9PL0005PW90: a too_large promotion does not clear the counter "
        "(and the body reports no count yet)"
    ),
)
async def test_too_large_promotion_resets_the_counter():
    """A ``too_large`` promotion is a different outcome: it clears the budget."""
    refusal, _calls, _request = await support.issue(
        _DEFERRAL_ROW,
        preference=LOCAL_PREFERRED,
        session_id=SESSION,
        config=support.config_for(_DEFERRAL_ROW, deferral_max_refusals=1),
    )
    assert refusal.status_code == 429

    promoted, promoted_calls, _request = await support.issue(
        _TOO_LARGE_ROW,
        preference=LOCAL_PREFERRED,
        session_id=SESSION,
        config=support.config_for(_TOO_LARGE_ROW, deferral_max_refusals=1),
    )
    assert promoted.status_code == 200, (
        f"a too_large request is promoted, got {promoted.status_code}"
    )
    assert promoted_calls == ["remote"], (
        f"the too_large promotion dispatches remotely, got {promoted_calls}"
    )

    after, _after_calls, _request = await support.issue(
        _DEFERRAL_ROW,
        preference=LOCAL_PREFERRED,
        session_id=SESSION,
        config=support.config_for(_DEFERRAL_ROW, deferral_max_refusals=1),
    )
    assert after.status_code == 429, (
        "a too_large promotion resets the deferral budget, so the next "
        f"deferrable refusal is a 429, got {after.status_code}"
    )
    assert support.body_of(after).get(BODY_REFUSALS) == 1


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason="LP-0MVUH9PL0005PW90: the counter store (and its clock hook) does not exist",
)
async def test_ttl_expiry_resets_the_counter():
    """A stale entry is not a budget: TTL expiry restarts the count.

    Asserted with the deterministic clock (no sleeping): the first refusal is
    recorded, the store's clock is pushed past ``TTL_SECONDS``, and the next
    request is refused *again* rather than escalated.
    """
    local_deferral = support.require_local_deferral()
    ttl_seconds = support.require_attribute(local_deferral, "TTL_SECONDS")

    with support.frozen_clock() as clock:
        first, _calls, _request = await support.issue(
            _DEFERRAL_ROW,
            preference=LOCAL_PREFERRED,
            session_id=SESSION,
            config=support.config_for(_DEFERRAL_ROW, deferral_max_refusals=1),
        )
        assert first.status_code == 429
        assert support.body_of(first).get(BODY_REFUSALS) == 1

        clock.advance(ttl_seconds + 1)

        after, _after_calls, _request = await support.issue(
            _DEFERRAL_ROW,
            preference=LOCAL_PREFERRED,
            session_id=SESSION,
            config=support.config_for(_DEFERRAL_ROW, deferral_max_refusals=1),
        )
    assert after.status_code == 429, (
        "an expired counter entry must not escalate the session "
        f"(got {after.status_code})"
    )
    assert support.body_of(after).get(BODY_REFUSALS) == 1, (
        "the count restarts after the entry's TTL, got "
        f"{support.body_of(after).get(BODY_REFUSALS)!r}"
    )


# ---------------------------------------------------------------------------
# Guards (parent AC8/AC9/AC10): unchanged paths stay unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_session_id_keeps_todays_unbounded_deferral():
    """Guard: without a stable key every request is still a retryable 429.

    There is nothing to count a keyless request against, so it keeps today's
    behaviour — one refusal per attempt, for ever. The row is deliberately
    *not* xfail: it must stay true across the change and across the default
    inversion that follows it.
    """
    results = await support.issue_sequence(
        _DEFERRAL_ROW,
        count=DEFAULT_DEFERRAL_MAX_REFUSALS + 1,
        preference=LOCAL_PREFERRED,
    )
    for attempt, (result, calls, _request) in enumerate(results, start=1):
        assert result.status_code == 429, (
            f"session-less attempt {attempt}: today's refusal must be kept, "
            f"got {result.status_code}"
        )
        assert support.route_of(result) == "refused"
        assert calls == []


@pytest.mark.asyncio
async def test_remote_less_chain_never_escalates():
    """Guard: a chain with no remote tier keeps refusing (the cost-capped pin).

    ``private``-shaped chains are the documented escape hatch for a client that
    cannot tolerate remote spend (parent assumption A4), so the budget must not
    turn the fourth refusal into a remote dispatch — there is nothing to
    dispatch to.
    """
    model_config = support.remote_less_model_config(_DEFERRAL_ROW)
    results = []
    for _ in range(DEFAULT_DEFERRAL_MAX_REFUSALS + 1):
        results.append(
            await support.issue(
                _DEFERRAL_ROW,
                preference=LOCAL_PREFERRED,
                session_id=SESSION,
                model_config=model_config,
            )
        )
    for attempt, (result, calls, _request) in enumerate(results, start=1):
        assert result.status_code == 429, (
            f"remote-less attempt {attempt}: the pin keeps refusing, got "
            f"{result.status_code}"
        )
        assert support.fallback_reason_of(result) != ESCALATION_TOKEN, (
            f"remote-less attempt {attempt}: no escalation without a remote tier"
        )
        assert calls == []


@pytest.mark.asyncio
async def test_threshold_zero_restores_unbounded_deferral():
    """Guard: ``local_deferral_max_refusals: 0`` disables the escalation.

    The knob's off-switch must reproduce today's behaviour exactly, because it
    is the documented rollback for the whole bounded-deferral change.
    """
    config = support.config_for(_DEFERRAL_ROW, deferral_max_refusals=0)
    results = await support.issue_sequence(
        _DEFERRAL_ROW,
        count=DEFAULT_DEFERRAL_MAX_REFUSALS + 1,
        preference=LOCAL_PREFERRED,
        session_id=SESSION,
        config=config,
    )
    for attempt, (result, _calls, _request) in enumerate(results, start=1):
        assert result.status_code == 429, (
            f"attempt {attempt} with the knob at 0: today's refusal must be "
            f"kept, got {result.status_code}"
        )
        assert support.fallback_reason_of(result) != ESCALATION_TOKEN


# ---------------------------------------------------------------------------
# The counter store itself (pre-agreed surface, see the support module)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="LP-0MVUH9PL0005PW90: proxy.local_deferral does not exist yet",
)
def test_counter_is_keyed_per_session():
    local_deferral = support.require_local_deferral()
    record = support.require_attribute(local_deferral, "record_refusal")
    count = support.require_attribute(local_deferral, "refusal_count")

    assert count(SESSION) == 0, "a session with no refusals has no budget used"
    assert record(SESSION) == 1
    assert record(SESSION) == 2
    assert count(SESSION) == 2
    assert count(OTHER_SESSION) == 0, (
        "one session's refusals must never spend another session's budget"
    )


@pytest.mark.xfail(
    strict=True,
    reason="LP-0MVUH9PL0005PW90: proxy.local_deferral does not exist yet",
)
def test_counter_reset_drops_only_the_named_session():
    local_deferral = support.require_local_deferral()
    record = support.require_attribute(local_deferral, "record_refusal")
    count = support.require_attribute(local_deferral, "refusal_count")
    reset = support.require_attribute(local_deferral, "reset")

    record(SESSION)
    record(OTHER_SESSION)
    reset(SESSION)

    assert count(SESSION) == 0
    assert count(OTHER_SESSION) == 1, (
        "reset(session) must drop exactly one session (bounded per-session "
        "state, not a global flush)"
    )


@pytest.mark.xfail(
    strict=True,
    reason="LP-0MVUH9PL0005PW90: proxy.local_deferral does not exist yet",
)
def test_counter_entries_expire_after_ttl():
    local_deferral = support.require_local_deferral()
    ttl_seconds = support.require_attribute(local_deferral, "TTL_SECONDS")
    record = support.require_attribute(local_deferral, "record_refusal")
    count = support.require_attribute(local_deferral, "refusal_count")

    with support.frozen_clock() as clock:
        record(SESSION)
        assert count(SESSION) == 1
        clock.advance(ttl_seconds + 1)
        assert count(SESSION) == 0, (
            "an entry older than TTL_SECONDS must not count towards a budget"
        )


@pytest.mark.xfail(
    strict=True,
    reason="LP-0MVUH9PL0005PW90: proxy.local_deferral does not exist yet",
)
def test_counter_store_is_bounded():
    """The store is capped: no unbounded growth across sessions (parent AC8)."""
    local_deferral = support.require_local_deferral()
    max_entries = support.require_attribute(local_deferral, "MAX_ENTRIES")
    record = support.require_attribute(local_deferral, "record_refusal")
    count = support.require_attribute(local_deferral, "refusal_count")

    assert isinstance(max_entries, int) and max_entries > 0, (
        f"MAX_ENTRIES must be a positive int, got {max_entries!r}"
    )
    earliest = f"{SESSION}-0"
    for index in range(max_entries + 3):
        record(f"{SESSION}-{index}")

    assert count(earliest) == 0, (
        "the oldest entry must be evicted once the store is at its cap "
        f"(MAX_ENTRIES={max_entries}); an unbounded store leaks per-session "
        "state"
    )
