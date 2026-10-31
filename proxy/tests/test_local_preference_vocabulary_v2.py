"""Vocabulary-v2 contract: two live values, two loud retirements, one default.

LP-0MVUH9OMY004MQ3U (child of LP-0MVU1GSDY003EJ0O; covers parent AC1, AC4, AC14,
AC21 — child AC1, AC2, AC6-docs).

This module is the **definition of done** for the vocabulary change, written
before it exists. It pins three things a silent revert would otherwise be able
to break:

1. **The retired values are rejected, loudly and early.** ``local-only``
   (retired by LP-0MVUH9PAZ002UAPA) and ``prefer-local`` (retired by
   LP-0MVUH9PUU005HVS4) each answer a stable ``400`` with
   ``code: local_preference_retired`` and a ``detail`` naming the live
   successor(s) — on **both** dispatch entry points, while the local route is
   free, and before any routing/queue/lease/dispatch state exists. Neither is
   normalised to ``local-preferred`` and neither is resolved to the default: a
   silent demotion would bill a cost-capped client without consent
   (LP-0MVNU2KTK009D4AG).
2. **The live vocabulary is exactly two values with one default.**
   ``VALID_PREFERENCES == ("local-preferred", "remote-ok")``,
   ``DEFAULT_PREFERENCE == "remote-ok"``, and ``local-only`` has left the alias
   table — so a third value creeping back, or the default drifting, fails here.
3. **The documents say what the code does.** ``proxy/docs/routing.md`` and the
   ``proxy/config.yaml`` commentary state the two live values, the
   ``remote-ok`` default and the bounded-deferral fields a client must honour.

Red phase
---------

Every spec marked ``xfail(strict=True)`` is red **today** and will be turned
green by the child named in its ``reason``. ``strict`` matters twice: the
implementing child cannot land without flipping the marker, and a *removal* of
the behaviour later cannot pass silently (an xfail that starts passing is a
failure, not a warning). The unmarked specs are guards: they hold today and
must keep holding across the change, so the retirement cannot take a live value
or the unknown-value fallback with it.

The parse-level spec accepts either signalling shape — a returned retired
outcome *or* a raised dedicated error — because the parent fixes the *outcome*
("never ``local-preferred``, never the default") and leaves the mechanism to
the owning child; the request-level rows are the authoritative surface.
"""

from __future__ import annotations

import local_preference_v2_support as support
import pytest
from local_preference_v2_support import (
    DEFAULT_PREFERENCE,
    HEADER_PREFERENCE,
    LIVE_PREFERENCES,
    LOCAL_ONLY,
    LOCAL_PREFERRED,
    PREFER_LOCAL,
    REMOTE_OK,
    RETIRED_CODE,
    RETIRED_VALUES,
)

from proxy import admission_reserve

#: A healthy free-slot row: the local route *can* serve, so a rejection here
#: proves the retired value was refused on its own account, not because no
#: local capacity happened to exist.
_HEALTHY_ROW = support.matrix._HEALTHY_ROW


def _retired_param(row: support.RetiredValue):
    """Parametrise one retired value, marked red until *its own* child lands.

    A single module-level ``xfail`` cannot express the staggered retirement:
    ``local-only`` is retired by LP-0MVUH9PAZ002UAPA and ``prefer-local`` by
    LP-0MVUH9PUU005HVS4, so each row carries its own marker. ``strict`` keeps
    its two jobs — the owning child cannot land without removing it, and a
    later re-introduction of the value cannot pass silently.
    """
    if row.owner == "LP-0MVUH9PAZ002UAPA":
        return pytest.param(row, id=row.value)
    return pytest.param(
        row,
        id=row.value,
        marks=pytest.mark.xfail(
            strict=True,
            reason=(
                f"{row.owner}: {row.value} is still a live value — its "
                "retirement is that child's job"
            ),
        ),
    )


RETIRED_PARAMS = [_retired_param(row) for row in RETIRED_VALUES]

#: A chain with no local entry at all, driven through the remote-only cycle.
_REMOTE_ONLY_ROW = support.matrix._row("no_local_route")


@pytest.fixture(autouse=True)
def _clean_state():
    support.reset_state()
    yield
    support.reset_state()


# ---------------------------------------------------------------------------
# AC1 (parent AC1/AC4/AC14): a retired value answers 400, before any routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", support.ENTRY_POINTS)
@pytest.mark.parametrize("retired", RETIRED_PARAMS)
async def test_retired_value_is_rejected_with_a_stable_400(
    retired: support.RetiredValue, entry_point: str
):
    """A withdrawn wire value is a ``400`` with the same shape on every entry.

    The row is deliberately run against a **free** local slot on the
    local-first cycle: the retired value must be refused even when the request
    would otherwise have been served locally (child AC2 — never a silent
    demotion), and the remote-only cycle proves the rejection is not a local
    capacity decision.
    """
    row = _HEALTHY_ROW if entry_point == "local_first" else _REMOTE_ONLY_ROW
    result, calls, _request = await support.issue(
        row, preference=retired.value, entry_point=entry_point
    )

    assert result.status_code == 400, (
        f"{retired.value} ({entry_point}): a retired value is a 400, got "
        f"{result.status_code}"
    )
    body = support.body_of(result)
    assert body.get("code") == RETIRED_CODE, (
        f"{retired.value} ({entry_point}): the body must carry "
        f"code={RETIRED_CODE!r}, got {body.get('code')!r}"
    )
    assert body.get("error"), (
        f"{retired.value} ({entry_point}): the rejection body must carry an "
        f"'error' field, got {sorted(body)}"
    )
    detail = str(body.get("detail") or "")
    for successor in retired.successors:
        assert successor in detail, (
            f"{retired.value} ({entry_point}): the detail must name the live "
            f"successor {successor!r} so a client can migrate from the body "
            f"alone, got {detail!r}"
        )
    support.assert_no_upstream_calls(
        calls, context=f"{retired.value} ({entry_point})"
    )
    support.assert_no_queue_or_lease_writes(
        calls, context=f"{retired.value} ({entry_point})"
    )


@pytest.mark.parametrize("retired", RETIRED_PARAMS)
def test_retired_value_is_never_normalised_or_resolved_to_the_default(
    retired: support.RetiredValue,
):
    """The single parse site never yields a live value for a retired one.

    ``local-only`` must not come back as ``local-preferred`` (silent
    demotion) and ``prefer-local`` must not come back as the new default
    (silent reinterpretation); both are the exact failure modes parent AC1 and
    AC14 forbid. A returned retired outcome and a raised dedicated error both
    satisfy "never a live value", so this spec does not pin which one.
    """
    try:
        outcome = admission_reserve.parse_preference(
            {HEADER_PREFERENCE.lower(): retired.value}
        )
    except Exception:
        # Signalling the retirement by raising is an acceptable shape: the
        # request-level rows above pin the user-visible contract.
        outcome = "<raised>"

    assert outcome not in admission_reserve.VALID_PREFERENCES, (
        f"{retired.value}: parse_preference must not return a live preference "
        f"(got {outcome!r} of {admission_reserve.VALID_PREFERENCES})"
    )
    assert outcome != admission_reserve.DEFAULT_PREFERENCE, (
        f"{retired.value}: parse_preference must not fall back to the default "
        f"(got {outcome!r})"
    )


# ---------------------------------------------------------------------------
# AC2 (parent AC21): the live vocabulary is exactly two values, one default
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "LP-0MVUH9PUU005HVS4: VALID_PREFERENCES still carries prefer-local and "
        "DEFAULT_PREFERENCE is still prefer-local"
    ),
)
def test_live_vocabulary_is_exactly_the_two_documented_values():
    """The live set and the default, asserted exactly (no subset check)."""
    assert admission_reserve.VALID_PREFERENCES == LIVE_PREFERENCES, (
        "the live vocabulary must be exactly "
        f"{LIVE_PREFERENCES}, got {admission_reserve.VALID_PREFERENCES}"
    )
    assert admission_reserve.DEFAULT_PREFERENCE == DEFAULT_PREFERENCE, (
        f"the default must be {DEFAULT_PREFERENCE!r}, got "
        f"{admission_reserve.DEFAULT_PREFERENCE!r} — an absent header is the "
        "policy every headerless client inherits"
    )


def test_local_only_has_left_the_alias_table_and_the_live_set():
    """A retired value cannot silently re-enter through the alias table.

    Split from the ``prefer-local`` half (LP-0MVUH9PAZ002UAPA): one marker over
    both retirements would stop guarding the *withdrawn* value the moment the
    still-live one kept the row red.
    """
    aliases = admission_reserve.LEGACY_PREFERENCE_ALIASES
    assert LOCAL_ONLY not in aliases, (
        "local-only must not be normalised any more; it is a loud 400 "
        f"(alias table: {aliases!r})"
    )
    assert LOCAL_ONLY not in admission_reserve.VALID_PREFERENCES
    assert set(aliases.values()) <= set(admission_reserve.VALID_PREFERENCES), (
        "every alias must still resolve to a live value "
        f"(alias table: {aliases!r})"
    )


@pytest.mark.xfail(
    strict=True,
    reason="LP-0MVUH9PUU005HVS4: prefer-local is still a live value",
)
def test_prefer_local_has_left_the_alias_table_and_the_live_set():
    """The second retirement's half of the same guarantee."""
    aliases = admission_reserve.LEGACY_PREFERENCE_ALIASES
    assert PREFER_LOCAL not in aliases, (
        f"prefer-local must not be normalised (alias table: {aliases!r})"
    )
    assert PREFER_LOCAL not in admission_reserve.VALID_PREFERENCES
    assert getattr(admission_reserve, "PREF_LOCAL_ONLY", LOCAL_ONLY) not in (
        admission_reserve.VALID_PREFERENCES
    ), "the retired local-only constant must not be part of the live set"


@pytest.mark.xfail(
    strict=True,
    reason="LP-0MVUH9PUU005HVS4: an absent header still resolves to prefer-local",
)
def test_absent_header_resolves_to_the_documented_default():
    assert admission_reserve.parse_preference({}) == DEFAULT_PREFERENCE, (
        "an absent header must resolve to the documented default "
        f"{DEFAULT_PREFERENCE!r}"
    )


@pytest.mark.parametrize("value", LIVE_PREFERENCES)
def test_live_values_still_parse_to_themselves(value: str):
    """Byte-compatibility guard: the two live values keep their exact result."""
    assert admission_reserve.parse_preference({HEADER_PREFERENCE.lower(): value}) == (
        value
    ), f"{value!r} must still parse to itself"


def test_unknown_value_still_falls_back_to_the_default():
    """Guard: the header stays additive — an unknown value never raises."""
    assert (
        admission_reserve.parse_preference(
            {HEADER_PREFERENCE.lower(): "local-preffered"}
        )
        == admission_reserve.DEFAULT_PREFERENCE
    ), "an unrecognised value resolves to the default (parent assumption: the header is additive)"


# ---------------------------------------------------------------------------
# AC6-docs (parent AC12): the documents match the two-value implementation
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "LP-0MVUH9Q4O008RHDX owns the docs: routing.md still documents "
        "prefer-local as the default, the local-only alias and the remote-ok "
        "strip, and config.yaml has no deferral-budget key"
    ),
)
def test_routing_docs_document_the_two_value_vocabulary_and_the_budget():
    """Docs-contract: a client author can learn the whole vocabulary here.

    Pins the *tokens and the retirements*, not the prose: every live value, the
    escalation token and both refusal-body fields must appear, and the two
    paragraphs that promised the old behaviour (unknown values behave as
    ``prefer-local``; clients may keep sending ``local-only``) must be gone.
    """
    text = support.collapsed(support.ROUTING_DOC)

    for token in (
        LOCAL_PREFERRED,
        REMOTE_OK,
        support.ESCALATION_TOKEN,
        support.BODY_REFUSALS,
        support.BODY_REFUSALS_REMAINING,
        support.DEFERRAL_CONFIG_KEY,
    ):
        assert token in text, (
            f"proxy/docs/routing.md must document {token!r}: the client-facing "
            "contract includes the value, the escalation token and the budget "
            "fields"
        )

    # The default must be stated, in either word order ("`remote-ok` (default)"
    # or "the default is `remote-ok`"): the prose is the docs owner's choice, the
    # identification of the default is not.
    default_position = text.find(f"`{REMOTE_OK}`")
    assert default_position != -1, "the default value must be named in backticks"
    neighbourhood = text[
        max(0, default_position - 80) : default_position + 80
    ]
    assert "default" in neighbourhood, (
        f"routing.md must identify `{REMOTE_OK}` as the default (not just list "
        f"it): ...{neighbourhood!r}..."
    )
    assert "behave as `prefer-local`" not in text, (
        "routing.md must stop resolving unknown values to the retired "
        "prefer-local value"
    )
    assert "may keep sending" not in text, (
        "routing.md must stop telling clients the local-only alias is accepted"
    )


def test_config_documents_the_deferral_budget_knob():
    """One knob, one documented default, restart-required (parent AC10)."""
    text = support.CONFIG_YAML.read_text(encoding="utf-8")
    assert (
        f"{support.DEFERRAL_CONFIG_KEY}: {support.DEFAULT_DEFERRAL_MAX_REFUSALS}"
        in text
    ), (
        f"proxy/config.yaml must carry {support.DEFERRAL_CONFIG_KEY}: "
        f"{support.DEFAULT_DEFERRAL_MAX_REFUSALS} with its restart-required "
        "commentary"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "LP-0MVUH9Q4O008RHDX owns the docs: routing.md still lists "
        "`prefer-local` as a selectable preference-table row"
    ),
)
def test_routing_docs_still_name_the_live_values_only_as_live():
    """The retired spellings remain migration *notes*, not live policy.

    The docs must mention the retirements (there is a migration note to write),
    so this spec pins only that neither retired value appears as a preference
    table row — a client must never be told to send them. A migration note that
    keeps the old row would keep shipping a value the proxy now rejects.
    """
    text = support.collapsed(support.ROUTING_DOC)
    for retired in RETIRED_VALUES:
        assert f"| `{retired.value}`" not in text, (
            f"{retired.value!r} must not appear as a documented preference "
            "table row (it is retired)"
        )
