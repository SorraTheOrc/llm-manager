"""``remote-ok`` means "local allowed, remote acceptable" — never "never wait".

LP-0MVUH9OMY004MQ3U (child of LP-0MVU1GSDY003EJ0O; covers parent AC15, AC16,
AC20 — child AC4, AC6-remote-ok).

The shipped meaning of ``remote-ok`` is the opposite of its name: today it
*strips* every local entry from the chain, so a ``remote-ok`` client can never
use the GPU. The intended meaning (operator-confirmed, parent assumption A5) is:

- the local route **stays** in the chain and is attempted exactly as a
  headerless request attempts it today — so with a free slot the request is
  served **locally**;
- when local cannot serve the request **right now** the request steps straight
  to the next provider — **no queue entry, no deferral, no local-capacity
  ``429``** — and the response reports ``X-Local-Route: remote`` with a
  canonical fall-through reason and a contention-queue wait of exactly ``0``.

One row per local-exit cause (parent AC20's enumeration), because the failure
mode this contract exists to prevent is a *new* exit that keeps the old
strip-and-go-remote shape: with ``remote-ok`` retained in the chain the exit is
taken and recorded, so the reason token proves the local route was actually
consulted rather than skipped by a preference filter.

Red phase
---------

The ``xfail(strict=True)`` specs are red today (the strip is still there) and
are turned green by LP-0MVUH9OYZ007GMWF, *before* the default inversion lands
so the suite never sees a headerless caller on the new semantics. The unmarked
specs are guards: the ``private`` model's absent-header behaviour, the two live
values' byte-compatibility and the local-down fall-through hold today and must
keep holding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import local_preference_v2_support as support
import proxy.provider as provider
import pytest
from fastapi import Response
from local_preference_v2_support import LOCAL_PREFERRED, REMOTE_OK

from proxy import priority

#: A free local slot: the row that proves ``remote-ok`` *attempts* local.
_HEALTHY_ROW = support.matrix._HEALTHY_ROW


def _lease_active_response() -> Response:
    """The router's lease-held 503 (``no_slots_available``).

    Mirrors the fixture ``test_local_admission_guard.py`` uses, so the
    lease-held exit is exercised through the same production predicate
    (``provider._is_local_lease_active_response``) rather than a stand-in.
    """
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


#: Post-dispatch lease exit: the local entry is *attempted* and reports the
#: lease as held, so the request steps to the next provider with the local
#: attempt on the record.
_LEASE_ROW = replace(
    support.matrix._row("local_error_status"),
    exit_description="dispatch lease held by another session",
    token="local_lease_active",
    local_response=_lease_active_response,
)


@dataclass(frozen=True)
class RemoteOkExitRow:
    """One local-exit cause a ``remote-ok`` request must step over instantly."""

    #: Parent AC20's own wording, so the table cannot drift from the contract.
    cause: str
    row: object


#: Parent AC20's enumeration, one row each: slot saturated, prefill-ETA over
#: the wait budget, local cooldown, residency/thrash skip, lease held and the
#: per-entry context cap.
REMOTE_OK_EXIT_ROWS: tuple[RemoteOkExitRow, ...] = (
    RemoteOkExitRow(
        cause="slot saturated (the queue policy would otherwise engage)",
        row=support.matrix._row("queue_depth_cap"),
    ),
    RemoteOkExitRow(
        cause="prefill-ETA over the wait budget",
        row=support.matrix._row("admission_guard_free_slot"),
    ),
    RemoteOkExitRow(
        cause="local entry in cooldown",
        row=support.matrix._row("chain_cooldown"),
    ),
    RemoteOkExitRow(
        cause="residency skip (resident_prefix_insufficient)",
        row=support.matrix._row("router_skip_residency"),
    ),
    RemoteOkExitRow(
        cause="prefill-thrash skip",
        row=support.matrix._row("thrash_cooldown"),
    ),
    RemoteOkExitRow(
        cause="per-entry max_context_tokens cap",
        row=support.matrix._row("entry_context_cap"),
    ),
    RemoteOkExitRow(cause="dispatch lease held", row=_LEASE_ROW),
)

REMOTE_OK_EXIT_IDS = [row.row.trigger for row in REMOTE_OK_EXIT_ROWS]


@pytest.fixture(autouse=True)
def _clean_state():
    support.reset_state()
    yield
    support.reset_state()


# ---------------------------------------------------------------------------
# AC4 (parent AC15): a free slot is used — remote-ok no longer strips local
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "LP-0MVUH9OYZ007GMWF: remote-ok still strips the local entry, so a "
        "free local slot is never used"
    ),
)
async def test_remote_ok_serves_locally_when_a_slot_is_free():
    """The value means "local allowed": a free slot is served locally.

    ``X-Local-Route: local`` with the local dispatch on the record, and no
    fall-through reason (nothing fell through).
    """
    result, calls, _request = await support.issue(
        _HEALTHY_ROW, preference=REMOTE_OK
    )

    assert result.status_code == 200, f"got {result.status_code}"
    assert calls == ["local"], (
        "a remote-ok request must attempt the local route (parent AC15): "
        f"dispatched={calls}"
    )
    assert support.route_of(result) == "local", (
        f"X-Local-Route must be 'local', got {support.route_of(result)!r}"
    )
    assert support.fallback_reason_of(result) is None, (
        "a locally served request has nothing to fall through to "
        f"(got {support.fallback_reason_of(result)!r})"
    )


#: Red-first rows for the chain filter: ``remote-ok`` still strips today, the
#: other live value does not. The split is per-parameter so the guard half
#: cannot silently xfail along with the red half.
_STRIP_ROWS = (
    pytest.param(
        REMOTE_OK,
        marks=pytest.mark.xfail(
            strict=True,
            reason=(
                "LP-0MVUH9OYZ007GMWF: apply_local_preference still strips local "
                "entries for PREF_REMOTE_OK"
            ),
        ),
        id=REMOTE_OK,
    ),
    pytest.param(LOCAL_PREFERRED, id=LOCAL_PREFERRED),
)


@pytest.mark.parametrize("preference", _STRIP_ROWS)
def test_no_live_preference_strips_the_local_route(preference: str):
    """No preference may remove the local entry: only a capacity verdict can.

    The chain filter is the single strip site, so this is the structural form
    of AC15/AC16: after the change it can only strip for ``too_large``, and a
    preference value never empties a chain.
    """
    chain = [
        {
            "name": support.matrix.LOCAL_PROVIDER_NAME,
            "type": "local",
            "llama_model": "Qwen3",
        },
        {
            "name": support.matrix.REMOTE_PROVIDER_NAME,
            "type": "remote",
            "endpoint": "https://api.example.com/v1",
        },
    ]

    filtered = priority.apply_local_preference(chain, preference)

    assert filtered == chain, (
        f"{preference}: the local entry must stay in the chain — "
        f"got {filtered}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "LP-0MVUH9OYZ007GMWF: effective_chain_for_request() still applies the "
        "remote-ok strip"
    ),
)
def test_effective_chain_keeps_the_local_entry_for_remote_ok():
    """The convenience used by both dispatch cycles keeps the local route."""
    request = support.build_request(preference=REMOTE_OK)
    model_config = support.matrix._model_config(_HEALTHY_ROW)

    chain = priority.effective_chain_for_request(request, model_config, {})

    names = [entry.get("name") for entry in chain]
    assert support.matrix.LOCAL_PROVIDER_NAME in names, (
        "a generic remote-ok request must see the local entry in its "
        f"effective chain, got {names}"
    )
    assert support.matrix.REMOTE_PROVIDER_NAME in names, (
        f"the remote entry must stay too, got {names}"
    )


# ---------------------------------------------------------------------------
# AC4 (parent AC20): every local exit steps over instantly — no wait, no 429
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "LP-0MVUH9OYZ007GMWF: remote-ok never consults the local route, so no "
        "exit is taken and no canonical fall-through reason is recorded"
    ),
)
@pytest.mark.parametrize("exit_row", REMOTE_OK_EXIT_ROWS, ids=REMOTE_OK_EXIT_IDS)
async def test_remote_ok_steps_to_the_next_provider_without_waiting(
    exit_row: RemoteOkExitRow,
):
    """One row per exit: served remotely, zero queue wait, no deferral.

    *calls* distinguishes the pre-dispatch exits (one remote dispatch) from the
    post-dispatch lease exit (the doomed local attempt is expected, exactly as
    the refusal matrix models it). The recorded fall-through reason is the
    proof that the local route was *consulted*: a preference-stripped chain
    would report ``no_local_provider`` because no exit was ever taken.
    """
    row = exit_row.row
    result, calls, request = await support.issue(row, preference=REMOTE_OK)
    context = f"remote-ok / {exit_row.cause}"

    expected_calls = ["local", "remote"] if row.local_response is not None else [
        "remote"
    ]
    assert result.status_code == 200, (
        f"{context}: the request must be served by the next provider, got "
        f"{result.status_code}"
    )
    assert support.route_of(result) == "remote", (
        f"{context}: X-Local-Route must be 'remote', got "
        f"{support.route_of(result)!r}"
    )
    assert calls == expected_calls, (
        f"{context}: expected exactly one upstream dispatch beyond any doomed "
        f"local attempt ({expected_calls}), got {calls}"
    )
    assert support.queue_wait_of(request) == 0.0, (
        f"{context}: a remote-ok request never waits for a local slot "
        f"(waited {support.queue_wait_of(request)}s)"
    )

    reason = support.fallback_reason_of(result)
    support.assert_canonical_fallback_reason(reason, context=context)
    assert reason != support.NO_LOCAL_PROVIDER, (
        f"{context}: the local route was skipped by the preference instead of "
        "being consulted — a preference-based strip is exactly what AC15 "
        f"removes (reason={reason!r})"
    )


# ---------------------------------------------------------------------------
# AC4 (parent AC16): a remote-less chain keeps working under every preference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_private_model_with_absent_header_is_served_locally():
    """Guard: the ``private`` model's behaviour must not change.

    Today's default (``prefer-local``) already serves this chain locally; the
    default inversion must not alter it, so this row stays green throughout.
    """
    result, calls, _request = await support.issue(
        _HEALTHY_ROW, preference=None, model_config=support.private_model_config()
    )

    assert result.status_code == 200, f"got {result.status_code}"
    assert calls == ["local"], (
        f"a headerless private-model request is served locally, got {calls}"
    )
    assert support.route_of(result) == "local"
    assert support.body_of(result).get("code") != provider.EXHAUSTION_CODE_ALL, (
        "the private model must never answer with 'all providers exhausted'"
    )


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "LP-0MVUH9OYZ007GMWF: an explicit remote-ok on a remote-less chain is "
        "stripped to an empty chain and answered with exhaustion"
    ),
)
async def test_private_model_with_explicit_remote_ok_is_served_locally():
    """A remote-less chain is never emptied by a preference (parent AC16).

    ``private`` has one ``type: local`` provider. Filtering it out on a
    preference would manufacture an empty chain and answer "all providers
    exhausted" to a model that can serve perfectly well locally.
    """
    result, calls, _request = await support.issue(
        _HEALTHY_ROW, preference=REMOTE_OK, model_config=support.private_model_config()
    )

    assert result.status_code == 200, (
        "a remote-ok request on a remote-less chain is served locally, got "
        f"{result.status_code}"
    )
    assert calls == ["local"], (
        f"no remote tier exists to step to; the local route serves, got {calls}"
    )
    assert support.route_of(result) == "local"
    body = support.body_of(result)
    assert body.get("code") != provider.EXHAUSTION_CODE_ALL, (
        "an empty chain must never be manufactured from a preference "
        f"(body code={body.get('code')!r})"
    )


@pytest.mark.asyncio
async def test_remote_ok_uses_the_remote_entry_when_the_chain_has_no_local_route():
    """Guard: a genuinely remote-only chain serves remotely (nothing to wait on)."""
    row = support.matrix._row("no_local_route")
    result, calls, request = await support.issue(row, preference=REMOTE_OK)

    assert result.status_code == 200, f"got {result.status_code}"
    assert calls == ["remote"], f"the only entry must serve, got {calls}"
    assert support.route_of(result) == "remote"
    assert support.queue_wait_of(request) == 0.0
