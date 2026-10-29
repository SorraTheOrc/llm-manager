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
    # Contention queue.
    "fallback_after_queue",
    "local_concurrency_limit",
    # Lease denial.
    "local_lease_active",
    # Structural refusals.
    "local_only_no_local_route",
    "local_route_unavailable",
}


def _section(text: str) -> str:
    """The documented section, heading to next heading."""
    start = text.index(SECTION_HEADING)
    rest = text[start + len(SECTION_HEADING) :]
    end = rest.index(NEXT_HEADING)
    return rest[:end]


def _documented_tokens(section: str) -> set[str]:
    """The ``X-Local-Fallback-Reason`` values documented in the exit table.

    Each data row's second cell may carry prose after the token (e.g. the
    direct-path row's "deadline/safety/budget reasons keep their own token"),
    so the first backticked span is the stable token the row pins.
    """
    tokens: set[str] = set()
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
        token = re.search(r"`([^`]+)`", cells[1])
        assert token, f"exit row {cells[0]!r} must carry a ticked token"
        tokens.add(token.group(1))
    assert tokens, "the section must contain an exit→token table"
    return tokens


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
    """prefer-local unchanged / remote-ok structural, per the X-Local-Route docs."""
    text = ROUTING_DOC.read_text()

    headers = text[text.index("### Response headers") :]
    headers = headers[: headers.index("### Per-reason retry contract")]
    # Line wrapping is not part of the contract: compare on normalised prose.
    plain = " ".join(headers.split())
    assert "`prefer-local` (the default) keeps today's behaviour unchanged" in plain, (
        "the response-header section must state that prefer-local falls through "
        "exactly as before"
    )
    assert "removed from the chain *before* dispatch" in plain, (
        "the response-header section must state that remote-ok strips local "
        "structurally"
    )
