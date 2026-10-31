"""Docs contract: the pin-local every-exit table must match the exits.

LP-0MVRMC2HT004NPO6 (child of LP-0MVNU2KTK009D4AG, parent AC7); section
renamed by LP-0MVSHXU8E004BL57 (the pin-local value is ``local-preferred``).

The deliverable here *is* the prose in ``proxy/docs/routing.md``: the
"``local-preferred``: pinned while deferrable, promoted when too large"
section must state the enforcement and enumerate the exit→token vocabulary a
client sees. The
assertions pin the contract, not the words — the documented token set must be
exactly the set the refusal paths emit, so a new local-route exit (or a
renamed token) that does not land in the docs fails here, and a documented
exit the code cannot produce fails the other way. No prose beyond the
refusal statement is pinned (mirroring ``test_local_admission_retry_docs.py``).

The expected set is the union of three pinned vocabularies:

- the refusal-matrix rows (``test_local_only_refusal_matrix.py``) — one token
  per exit from the local route;
- the canonical chain/lease/structural tokens
  (``admission_reserve.LOCAL_FALLBACK_REASON_TOKENS``);
- the admission-guard reasons the refusal builders emit on the queue and
  direct paths (``test_local_admission_guard.py``'s cross-reason table).

``local_only_no_local_route`` keeps its own focused assertion: a chain with no
local entry is documented as a **refusal**, never a remote fall-through (AC5).

The deferrable/too-large split (LP-0MVTF2WYK007QI53, parent AC7) is pinned the
same way: every exit row carries an **Outcome** cell (`refusing` or
`promoting`), and the promoting set must equal the implementation's
``admission_reserve.CAPACITY_TOO_LARGE_REASONS`` — so a documented exit that
promotes (``context_too_large``, ``prefill_budget_exceeded``) can never read as
a refusal, and a temporal exit can never read as a promotion.
"""

from __future__ import annotations

import re
from pathlib import Path

ROUTING_DOC = Path(__file__).resolve().parents[1] / "docs" / "routing.md"

#: Heading of the documented section; the section ends at the next heading.
#: Renamed by LP-0MVSHXU8E004BL57: the value carrying the per-exit pin is
#: ``local-preferred`` (``local-only`` is its deprecated alias), and the section
#: now also documents the ``too_large`` promotion that declines the pin.
SECTION_HEADING = (
    "### `local-preferred`: pinned while deferrable, promoted when too large"
)
NEXT_HEADING = "### Response headers"

#: The stable ``X-Local-Fallback-Reason`` tokens a local-only client can be
#: refused with. Union of the matrix rows, the canonical reason-token table and
#: the admission-guard reasons — see the module docstring.
DOCUMENTED_EXIT_TOKENS = {
    # Admission guard, queue path.
    "eta_exceeds_budget",
    "prefill_eta_exceeds_deadline",
    "prefill_eta_exceeds_safety_budget",
    "prefill_budget_exceeded",
    # Admission guard, direct free-slot path.
    "eta_exceeds_budget_direct",
    # Chain resolution.
    "local_provider_cooldown",
    "quota_hard_stop",
    "usage_limit_reset_pending",
    "outside_available_window",
    # Router skips.
    "context_too_large",
    "large_context_bypass",
    "resident_prefix_insufficient",
    "tenant_limit",
    "prefill_thrash_cooldown",
    # Reason-scoped local 5xx release (LP-0MVWFKESK0005G61): the measured
    # cache-deficit abort detours the request without quarantining the entry.
    "measured_cache_deficit",
    # Contention queue.
    "fallback_after_queue",
    "local_concurrency_limit",
    # Lease denial.
    "local_lease_active",
    # Structural refusals.
    "local_only_no_local_route",
    "local_route_unavailable",
    # Bounded deferral (LP-0MVUH9PL0005PW90): the session spent its
    # retryable-refusal budget, so the pin is declined and a remote tier serves
    # the request. A promotion with its own token — not a capacity reason.
    "deferral_refusals_exhausted",
}


def _section(text: str) -> str:
    """The documented section, heading to next heading."""
    start = text.index(SECTION_HEADING)
    rest = text[start + len(SECTION_HEADING) :]
    end = rest.index(NEXT_HEADING)
    return rest[:end]


def _exit_rows(section: str) -> list[list[str]]:
    """Data rows of the exit→token table, markdown separator rows removed."""
    rows: list[list[str]] = []
    in_table = False
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            in_table = False
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and cells[0].lower().startswith("exit from the local route"):
            in_table = True
            continue
        if not in_table:
            continue
        if all(set(cell) <= set("-: ") for cell in cells):
            continue  # markdown separator row
        rows.append(cells)
    return rows


def _documented_tokens(section: str) -> set[str]:
    """The ``X-Local-Fallback-Reason`` values documented in the exit table.

    Each data row's second cell may carry prose after the token (e.g. the
    direct-path row's "deadline/safety/budget reasons keep their own token"),
    so the first backticked span is the stable token the row pins.
    """
    tokens: set[str] = set()
    for cells in _exit_rows(section):
        token = re.search(r"`([^`]+)`", cells[1])
        assert token, f"exit row {cells[0]!r} must carry a ticked token"
        tokens.add(token.group(1))
    assert tokens, "the section must contain an exit→token table"
    return tokens


#: The two outcomes a pinned ``local-preferred`` request can meet at an exit.
DOCUMENTED_OUTCOMES = {"refusing", "promoting"}


def _documented_outcomes(section: str) -> dict[str, str]:
    """Map each documented exit token to its documented outcome.

    The third cell of each row states what a pinned request does with that
    exit — ``refusing`` (retryable ``429`` deferral) or ``promoting`` (the
    local entry leaves the chain and a remote tier serves the request) — so a
    client can tell the two apart from the table alone.
    """
    outcomes: dict[str, str] = {}
    for cells in _exit_rows(section):
        token = re.search(r"`([^`]+)`", cells[1])
        assert token, f"exit row {cells[0]!r} must carry a ticked token"
        assert len(cells) >= 3, (
            f"exit row {token.group(1)!r} must carry an Outcome cell "
            f"(one of {sorted(DOCUMENTED_OUTCOMES)})"
        )
        outcomes[token.group(1)] = cells[2].strip("*` ").lower()
    assert outcomes, "the section must contain an exit→token table"
    return outcomes


def test_documented_tokens_are_exactly_the_refusal_vocabulary():
    """One documented exit token per emitted token — no invented, no missing."""
    section = _section(ROUTING_DOC.read_text())

    assert _documented_tokens(section) == DOCUMENTED_EXIT_TOKENS, (
        "routing.md's every-exit table must document every token a local-only "
        "client can be refused with and nothing else; missing="
        f"{sorted(DOCUMENTED_EXIT_TOKENS - _documented_tokens(section))} "
        f"unexpected={sorted(_documented_tokens(section) - DOCUMENTED_EXIT_TOKENS)}"
    )


def test_every_canonical_reason_token_is_documented():
    """The canonical table's vocabulary (chain/lease/structural) is all there."""
    from proxy import admission_reserve

    section = _section(ROUTING_DOC.read_text())
    documented = _documented_tokens(section)

    assert admission_reserve.LOCAL_FALLBACK_REASON_TOKENS <= documented, (
        "every canonical reason token must be documented; missing="
        f"{sorted(admission_reserve.LOCAL_FALLBACK_REASON_TOKENS - documented)}"
    )


def test_no_local_route_is_documented_as_a_refusal_not_a_fall_through():
    """AC5: a chain with no local entry refuses, never falls through remote."""
    section = _section(ROUTING_DOC.read_text())

    assert "local_only_no_local_route" in section, (
        "the structural no-local-route refusal must be documented"
    )
    # The refusal statement is the contract, not the wording — anything that
    # says "refused" near the no-local-route token satisfies it, while a
    # silent-fall-through wording fails.
    refusals = [
        line for line in section.splitlines() if "local_only_no_local_route" in line
    ]
    assert refusals, "the exit table must name the no-local-route token"
    assert any("refus" in line.lower() for line in refusals) or (
        "never a remote fall-through" in section
    ), "the no-local-route exit must read as a refusal, not a fall-through"


def test_response_headers_section_keeps_the_other_two_preferences_accurate():
    """prefer-local unchanged / no preference strips, per the X-Local-Route docs.

    The section must state that ``prefer-local`` falls through exactly as
    before **and** that the only chain-level strip is the capacity filter — a
    ``remote-ok`` request keeps the local route (LP-0MVUH9OYZ007GMWF AC1/AC5),
    so a sentence claiming the preference removes local entries fails here.
    """
    text = ROUTING_DOC.read_text()

    headers = text[text.index("### Response headers") :]
    headers = headers[: headers.index("### Per-reason retry contract")]
    # Line wrapping is not part of the contract: compare on normalised prose.
    plain = " ".join(headers.split())
    assert "`prefer-local` (the default) keeps today's behaviour unchanged" in plain, (
        "the response-header section must state that prefer-local falls through "
        "exactly as before"
    )
    assert "The only chain-level strip that exists is the capacity filter" in plain, (
        "the response-header section must re-anchor stripping on the capacity "
        "verdict — no preference removes a local entry"
    )
    assert "so a `remote-ok` request reaches a local-route exit exactly like an " in plain, (
        "the response-header section must state that remote-ok keeps the local "
        "route (its name is historic)"
    )


def test_every_exit_row_declares_a_refusal_or_a_promotion():
    """AC3: each row's Outcome is one of the two documented outcomes."""
    section = _section(ROUTING_DOC.read_text())
    outcomes = _documented_outcomes(section)

    assert set(outcomes) == _documented_tokens(section), (
        "each documented exit token must carry exactly one Outcome cell"
    )
    unknown = {
        token: outcome
        for token, outcome in outcomes.items()
        if outcome not in DOCUMENTED_OUTCOMES
    }
    assert not unknown, (
        "every exit row must be marked refusing or promoting; got "
        f"{sorted(unknown.items())}"
    )


def test_promoting_rows_are_exactly_the_implementation_too_large_reasons():
    """AC1/AC3: the documented promoting set is the code's too-large set.

    ``admission_reserve.CAPACITY_TOO_LARGE_REASONS`` is the single source of
    truth for which exits decline the pin, so a row marked promoting that the
    code refuses (or vice versa) fails here.
    """
    from proxy import admission_reserve

    outcomes = _documented_outcomes(_section(ROUTING_DOC.read_text()))
    promoting = {token for token, outcome in outcomes.items() if outcome == "promoting"}

    # Two promotion *causes* exist and both must read as promoting: the
    # capacity verdict (this prompt can never fit local) and the bounded
    # deferral (this session has been told to retry too many times). The
    # equivalence stays exact in both directions — a new promoting row that is
    # neither fails here, and a cause that stops promoting fails too
    # (LP-0MVUH9PL0005PW90).
    expected_promoting = set(admission_reserve.CAPACITY_TOO_LARGE_REASONS) | {
        admission_reserve.LOCAL_FALLBACK_DEFERRAL_EXHAUSTED
    }
    assert promoting == expected_promoting, (
        "routing.md must mark exactly the implementation's promotion reasons as "
        "promoting; documented-but-not-implemented="
        f"{sorted(promoting - expected_promoting)} "
        "implemented-but-not-documented="
        f"{sorted(expected_promoting - promoting)}"
    )
    # AC3 names the two promoting exits explicitly.
    assert {"context_too_large", "prefill_budget_exceeded"} <= promoting, (
        "the context cap (including the per-entry max_context_tokens ceiling) "
        "and the cold-prefill cap must read as promoting rows"
    )


def test_temporal_exits_are_documented_as_refusing_rows():
    """AC3: cooldown/residency/thrash/queue/lease/quota rows all refuse."""
    outcomes = _documented_outcomes(_section(ROUTING_DOC.read_text()))

    temporal_exit_groups = {
        "cooldown": {"local_provider_cooldown", "prefill_thrash_cooldown"},
        "residency": {"resident_prefix_insufficient", "tenant_limit"},
        "thrash": {"prefill_thrash_cooldown"},
        "queue": {"fallback_after_queue", "local_concurrency_limit"},
        "lease": {"local_lease_active"},
        "quota": {"quota_hard_stop", "usage_limit_reset_pending"},
    }
    for group, tokens in temporal_exit_groups.items():
        missing = tokens - set(outcomes)
        assert not missing, f"{group} exit(s) {sorted(missing)} are not documented"
        promoted = sorted(
            token for token in tokens if outcomes[token] != "refusing"
        )
        assert not promoted, (
            f"{group} exits are temporal (waiting can help) and must read as "
            f"refusing rows, not promotions: {promoted}"
        )


PINNED_HEADING = "#### Pinned: a deferrable request never goes remote"
PROMOTED_HEADING = "#### Promoted: a too-large request leaves the local chain"


def _pinned_section(text: str) -> str:
    """The pinned-enforcement section, heading to the promotion section."""
    start = text.index(PINNED_HEADING)
    rest = text[start + len(PINNED_HEADING) :]
    return rest[: rest.index(PROMOTED_HEADING)]


def test_pinned_section_states_both_outcomes_not_a_universal_refusal():
    """AC1: the old "every exit refuses" claim is replaced by the split."""
    text = ROUTING_DOC.read_text()
    plain = " ".join(text.split())

    assert (
        "every exit from the local route is an exit into the deferral or the "
        "refusal"
    ) not in plain, (
        "the universal claim is false for too-large exits — the document must "
        "state the deferrable/too-large split instead"
    )

    section = " ".join(_pinned_section(text).split())
    assert "X-Local-Route: refused" in section, (
        "the pinned section must document the deferrable refusal "
        "(X-Local-Route: refused)"
    )
    assert "X-Local-Route: remote" in section, (
        "the pinned section must document the too-large promotion "
        "(X-Local-Route: remote)"
    )
    assert "#promoted-a-too-large-request-leaves-the-local-chain" in section, (
        "the pinned section must point at the promotion mechanism"
    )


def test_response_headers_document_promotion_and_the_unstrippable_chain():
    """AC4: X-Local-Route documents promotion and never strips an all-local chain."""
    text = ROUTING_DOC.read_text()
    headers = text[text.index("### Response headers") :]
    headers = headers[: headers.index("### Per-reason retry contract")]
    plain = " ".join(headers.replace("`", "").replace("*", "").split())

    assert "remote" in plain and "promoted" in plain, (
        "the X-Local-Route section must document the remote value as a promotion"
    )
    assert "refused" in plain, (
        "the X-Local-Route section must keep documenting the refused value"
    )
    assert "no non-local entry" in plain and "is never stripped" in plain, (
        "the X-Local-Route section must state that a chain with no non-local "
        "entry is never stripped (it keeps deferring/refusing)"
    )


def test_refusal_line_fields_and_the_fallback_claim_are_documented():
    """LP-0MVNTJQVB007QWEZ AC2/AC3: the countable token, its context fields and
    the truthful fallback claim are documented where operators look.

    An operator counts this failure class by one token, so the prose must name
    that token and the fields that make a refusal attributable; and it must
    state that the ``… → routing to next provider`` line belongs to a real
    fall-through only — otherwise the docs would bless the false claim this
    item removed.
    """
    section = _section(ROUTING_DOC.read_text())

    for field in (
        "local_only_refused",
        "route=refused",
        "session=",
        "provider=",
        "priority=",
        "new_tokens=",
        "budget_tokens=",
        "eta_prefill_seconds=",
        "budget_seconds=",
    ):
        assert field in section, (
            f"the refusal-line contract must document {field!r} — an operator "
            "cannot count what the docs do not name (AC2/AC3)"
        )

    fallback_lines = [
        line
        for line in section.splitlines()
        if "local_admission_guard_direct_fallback" in line
    ]
    assert fallback_lines, (
        "the direct-guard dispatch line must stay documented"
    )
    assert any("real fall-through" in line for line in fallback_lines) or any(
        "only" in line.lower() and "fall" in line.lower()
        for line in fallback_lines
    ), (
        "the docs must state that the '→ routing to next provider' line is "
        "emitted only on a real fall-through (AC1)"
    )


#: Heading of the worked-example subsection that shows the reproduced incident
#: (LP-0MVNTJQVB007QWEZ AC5).
WORKED_EXAMPLE_HEADING = "**Worked example"

#: The 2026-10-25 refusal's session id — the case the example reproduces.
INCIDENT_REFUSAL_SESSION = "herdr-1792897705-4039879-16152"

#: The test that reproduces the incident end-to-end (and asserts the example is
#: the exact line it emits).
REPRODUCTION_TEST = (
    "test_direct_guard_pin_refusal_is_attributable_and_claims_no_fallback"
)


def _worked_example_block(section: str) -> str:
    """The fenced log block of the worked example, fences stripped."""
    rest = section[section.index(WORKED_EXAMPLE_HEADING) :]
    fence_start = rest.index("```")
    fence_end = rest.index("```", fence_start + 3)
    return rest[fence_start + 3 : fence_end]


def test_worked_example_shows_the_reproduced_incident_log_output():
    """LP-0MVNTJQVB007QWEZ AC5: the incident's output is written down.

    "Reproducible in a test" is only verifiable if the output an operator
    actually sees is recorded next to the reproducer: the worked example must
    show the refusal line for the 2026-10-25 session, name the test that
    reproduces it, and — because the refusal was never routed anywhere — claim
    no fall-through inside the example. (The reproducer additionally asserts
    the example *is* the line it emits, so the sample cannot be invented.)
    """
    section = _section(ROUTING_DOC.read_text())
    block = _worked_example_block(section)

    assert INCIDENT_REFUSAL_SESSION in block, (
        "the worked example must show the log output of the 2026-10-25 "
        f"incident (session {INCIDENT_REFUSAL_SESSION}); got {block!r}"
    )
    example_lines = [line.strip() for line in block.splitlines() if line.strip()]
    assert any(
        line.startswith("local_only_refused ")
        and f"session={INCIDENT_REFUSAL_SESSION}" in line
        for line in example_lines
    ), (
        "the worked example must show the countable refusal line for the "
        f"incident; got {example_lines!r}"
    )
    assert any(
        line.startswith("local_admission_guard decision=reject") for line in example_lines
    ), (
        "the worked example must also show the guard decision line the refusal "
        f"accompanies; got {example_lines!r}"
    )
    assert "routing to next provider" not in block, (
        "the worked example must not show a fall-through claim for a refused "
        f"request — nothing was routed; got {block!r}"
    )
    assert REPRODUCTION_TEST in section, (
        f"the worked example must name its reproducer ({REPRODUCTION_TEST}) "
        "so an operator can regenerate the output"
    )
